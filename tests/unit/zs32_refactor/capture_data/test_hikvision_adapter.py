# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Linux-authoritative contract tests for the Hikvision capture boundary."""

from __future__ import annotations

import sys
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from zs32_inspection.capture import CameraBinding, CaptureRequest, CaptureRoundPlan, PartialRoundCaptureError
from zs32_inspection.capture.hikvision import (
    DeviceDescription,
    HikvisionCameraAdapter,
    HikvisionCaptureConfig,
    HikvisionCaptureError,
    select_devices_by_serial,
)
from zs32_inspection.domain.identity import Hand, PartIdentity
from zs32_inspection.timing import TimingRecorder


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="camera runtime is Linux only")


def _bindings(count: int = 3) -> tuple[CameraBinding, ...]:
    return tuple(
        CameraBinding(
            slot_id=f"slot_{index}",
            serial=f"SERIAL_{index}",
            views={"front": f"front_{index}", "back": f"back_{index}"},
        )
        for index in range(count)
    )


@pytest.mark.parametrize("camera_count", [3, 4, 5])
def test_serial_binding_is_topology_driven(camera_count: int) -> None:
    """Enumeration order cannot change the stable topology slot order."""
    bindings = _bindings(camera_count)
    available = tuple(
        DeviceDescription(index, "model", f"SERIAL_{camera_count - index - 1}")
        for index in range(camera_count)
    )
    selected = select_devices_by_serial(bindings, available)
    assert tuple(device.serial for device in selected) == tuple(
        binding.serial for binding in bindings
    )


def test_missing_topology_serial_fails_closed() -> None:
    """A device index is never substituted for a missing stable serial."""
    with pytest.raises(HikvisionCaptureError, match="unavailable"):
        select_devices_by_serial(
            _bindings(),
            (
                DeviceDescription(0, "model", "SERIAL_0"),
                DeviceDescription(1, "model", "SERIAL_1"),
            ),
        )


def test_duplicate_enumerated_serial_fails_closed() -> None:
    """Ambiguous SDK identity cannot be resolved by enumeration index."""
    with pytest.raises(HikvisionCaptureError, match="duplicate"):
        select_devices_by_serial(
            _bindings(),
            (
                DeviceDescription(0, "model", "SERIAL_0"),
                DeviceDescription(1, "model", "SERIAL_0"),
                DeviceDescription(2, "model", "SERIAL_1"),
                DeviceDescription(3, "model", "SERIAL_2"),
            ),
        )


def test_adapter_construction_does_not_load_sdk() -> None:
    """Importing and configuring capture on a Mac must not touch production modules."""
    called = False

    def forbidden_loader() -> Any:
        nonlocal called
        called = True
        raise AssertionError("dependency loader must remain lazy")

    HikvisionCameraAdapter(HikvisionCaptureConfig(), dependency_loader=forbidden_loader)
    assert not called


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("exposure", 0),
        ("capture_interval", -0.1),
        ("timeout_ms", 0),
        ("hdr_settle_frames", -1),
        ("hdr_max_clip_pct", 101),
        ("png_compression", 10),
        ("hdr", "yes"),
    ],
)
def test_invalid_acquisition_parameters_are_rejected(field: str, value: Any) -> None:
    """Hardware is never opened for an invalid acquisition profile."""
    with pytest.raises((TypeError, ValueError)):
        HikvisionCaptureConfig(**{field: value})


def test_hdr_exposure_order_is_explicit() -> None:
    """Misnamed/reversed HDR exposure profiles fail before hardware access."""
    with pytest.raises(ValueError, match="short_exposure"):
        HikvisionCaptureConfig(hdr=True, short_exposure=40000, long_exposure=4000)


def test_versioned_acquisition_mapping_requires_every_exact_field() -> None:
    payload = HikvisionCaptureConfig().as_dict()
    assert HikvisionCaptureConfig.from_mapping(payload).as_dict() == payload

    missing = dict(payload)
    missing.pop("timeout_ms")
    with pytest.raises(ValueError, match="strict schema"):
        HikvisionCaptureConfig.from_mapping(missing)

    extra = {**payload, "camera_index": 0}
    with pytest.raises(ValueError, match="strict schema"):
        HikvisionCaptureConfig.from_mapping(extra)


def test_hdr_and_encoding_timing_accumulates_required_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = TimingRecorder("capture-test")
    adapter = HikvisionCameraAdapter(
        HikvisionCaptureConfig(hdr=True, hdr_max_retries=0),
        timing_recorder=recorder,
    )
    images = tuple(
        SimpleNamespace(binding=binding, device_index=index, image=object())
        for index, binding in enumerate(_bindings())
    )
    monkeypatch.setattr(adapter, "_capture_exposure_pass", lambda _exposure: images)
    monkeypatch.setattr(adapter, "_fuse_exposures", lambda _short, _long: object())
    monkeypatch.setattr(adapter, "_image_clip_pct", lambda _image: 0.0)

    adapter._capture_hdr()

    class _Cv2:
        IMWRITE_PNG_COMPRESSION = 16

        @staticmethod
        def imencode(_extension, _image, _parameters):
            return True, SimpleNamespace(tobytes=lambda: b"\x89PNG\r\n\x1a\nencoded")

    adapter._dependencies = SimpleNamespace(cv2=_Cv2(), numpy=None)
    captured = SimpleNamespace(
        binding=_bindings(1)[0],
        device_index=0,
        image=SimpleNamespace(ndim=3, shape=(2, 3, 3)),
    )
    round_plan = SimpleNamespace(round_id="front")
    adapter._encode_frame(
        captured,
        round_plan,
        capture_mode="hdr_fused",
        exposure=None,
        parameters={},
    )

    stages = recorder.payload()["stages"]
    assert stages["hdr_short_exposure"]["count"] == 1
    assert stages["hdr_long_exposure"]["count"] == 1
    assert stages["hdr_fusion"]["count"] == len(images)
    assert stages["image_encoding"]["count"] == 1


def test_hdr_fuses_four_cameras_concurrently_in_topology_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Independent in-memory HDR pairs should overlap without reordering cameras."""
    recorder = TimingRecorder("capture-test")
    adapter = HikvisionCameraAdapter(
        HikvisionCaptureConfig(hdr=True, hdr_max_retries=0),
        timing_recorder=recorder,
    )
    bindings = _bindings(4)
    short_images = tuple(
        SimpleNamespace(binding=binding, device_index=index, image=("short", index))
        for index, binding in enumerate(bindings)
    )
    long_images = tuple(
        SimpleNamespace(binding=binding, device_index=index, image=("long", index))
        for index, binding in enumerate(bindings)
    )
    exposure_passes = iter((short_images, long_images))
    monkeypatch.setattr(adapter, "_capture_exposure_pass", lambda _exposure: next(exposure_passes))
    monkeypatch.setattr(adapter, "_image_clip_pct", lambda image: float(image[1]))
    lock = threading.Lock()
    release = threading.Event()
    active = 0
    entered = 0
    max_active = 0

    def fuse(short: tuple[str, int], long: tuple[str, int]) -> tuple[str, int]:
        nonlocal active, entered, max_active
        assert short[1] == long[1]
        with lock:
            active += 1
            entered += 1
            max_active = max(max_active, active)
            if entered == 4:
                release.set()
        assert release.wait(timeout=2.0)
        with lock:
            active -= 1
        return "fused", short[1]

    monkeypatch.setattr(adapter, "_fuse_exposures", fuse)

    fused, attempt, clips = adapter._capture_hdr()

    assert max_active == 4
    assert attempt == 1
    assert tuple(item.binding for item in fused) == bindings
    assert tuple(item.device_index for item in fused) == tuple(range(4))
    assert tuple(item.image for item in fused) == tuple(("fused", index) for index in range(4))
    assert clips == (0.0, 1.0, 2.0, 3.0)
    stages = recorder.payload()["stages"]
    assert stages["hdr_fusion"]["count"] == 4
    assert stages["hdr_fusion_parallel_wall"]["count"] == 1


def test_capture_round_encodes_four_cameras_concurrently_in_topology_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Independent PNG encodes should overlap while frames retain topology order and bytes."""
    recorder = TimingRecorder("capture-test")
    adapter = HikvisionCameraAdapter(
        HikvisionCaptureConfig(hdr=True, hdr_max_retries=0),
        timing_recorder=recorder,
    )
    bindings = _bindings(4)
    images = tuple(
        SimpleNamespace(
            binding=binding,
            device_index=index,
            image=SimpleNamespace(ndim=3, shape=(2, 3, 3), marker=index),
        )
        for index, binding in enumerate(bindings)
    )
    monkeypatch.setattr(adapter, "_ensure_open", lambda _bindings: None)
    monkeypatch.setattr(adapter, "_capture_hdr", lambda: (images, 1, (0.0, 1.0, 2.0, 3.0)))
    lock = threading.Lock()
    release = threading.Event()
    active = 0
    entered = 0
    max_active = 0

    class _Cv2:
        IMWRITE_PNG_COMPRESSION = 16

        @staticmethod
        def imencode(_extension, image, _parameters):
            nonlocal active, entered, max_active
            with lock:
                active += 1
                entered += 1
                max_active = max(max_active, active)
                if entered == 4:
                    release.set()
            assert release.wait(timeout=2.0)
            with lock:
                active -= 1
            payload = b"\x89PNG\r\n\x1a\n" + bytes([image.marker])
            return True, SimpleNamespace(tobytes=lambda: payload)

    adapter._dependencies = SimpleNamespace(cv2=_Cv2(), numpy=None)
    request = CaptureRequest("session", "capture", PartIdentity("part", Hand.RIGHT))
    round_plan = CaptureRoundPlan("front", "capture front")

    frames = adapter.capture_round(request, round_plan, bindings)

    assert max_active == 4
    assert tuple(frame.camera_slot_id for frame in frames) == tuple(binding.slot_id for binding in bindings)
    assert tuple(frame.view_id for frame in frames) == tuple(binding.views["front"] for binding in bindings)
    assert tuple(frame.image_bytes[-1] for frame in frames) == tuple(range(4))
    stages = recorder.payload()["stages"]
    assert stages["image_encoding"]["count"] == 4
    assert stages["image_encoding_parallel_wall"]["count"] == 1


def test_parallel_hdr_is_pixel_identical_to_serial_fusion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parallel scheduling must not alter the established HDR algorithm output."""
    import cv2
    import numpy as np

    config = HikvisionCaptureConfig(
        hdr=True,
        hdr_max_retries=0,
        hdr_max_clip_pct=100.0,
        align_hdr=False,
        blur_size=5,
    )
    adapter = HikvisionCameraAdapter(config)
    adapter._dependencies = SimpleNamespace(cv2=cv2, numpy=np)
    bindings = _bindings(4)
    generator = np.random.default_rng(42)
    short_arrays = tuple(generator.integers(0, 180, (24, 32, 3), dtype=np.uint8) for _ in bindings)
    long_arrays = tuple(generator.integers(30, 240, (24, 32, 3), dtype=np.uint8) for _ in bindings)
    expected = tuple(
        adapter._fuse_exposures(short_image, long_image)
        for short_image, long_image in zip(short_arrays, long_arrays, strict=True)
    )
    short_images = tuple(
        SimpleNamespace(binding=binding, device_index=index, image=image)
        for index, (binding, image) in enumerate(zip(bindings, short_arrays, strict=True))
    )
    long_images = tuple(
        SimpleNamespace(binding=binding, device_index=index, image=image)
        for index, (binding, image) in enumerate(zip(bindings, long_arrays, strict=True))
    )
    exposure_passes = iter((short_images, long_images))
    monkeypatch.setattr(adapter, "_capture_exposure_pass", lambda _exposure: next(exposure_passes))

    actual, attempt, _clips = adapter._capture_hdr()

    assert attempt == 1
    assert all(
        np.array_equal(item.image, expected_image)
        for item, expected_image in zip(actual, expected, strict=True)
    )


def test_parallel_png_is_byte_identical_to_serial_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parallel scheduling must preserve OpenCV PNG bytes exactly."""
    import cv2
    import numpy as np

    config = HikvisionCaptureConfig(hdr=True, hdr_max_retries=0, png_compression=3)
    adapter = HikvisionCameraAdapter(config)
    adapter._dependencies = SimpleNamespace(cv2=cv2, numpy=np)
    bindings = _bindings(4)
    generator = np.random.default_rng(7)
    arrays = tuple(generator.integers(0, 256, (24, 32, 3), dtype=np.uint8) for _ in bindings)
    images = tuple(
        SimpleNamespace(binding=binding, device_index=index, image=image)
        for index, (binding, image) in enumerate(zip(bindings, arrays, strict=True))
    )
    expected = []
    for image in arrays:
        success, encoded = cv2.imencode(
            ".png",
            image,
            [cv2.IMWRITE_PNG_COMPRESSION, config.png_compression],
        )
        assert success
        expected.append(encoded.tobytes())
    monkeypatch.setattr(adapter, "_ensure_open", lambda _bindings: None)
    monkeypatch.setattr(adapter, "_capture_hdr", lambda: (images, 1, (0.0, 0.0, 0.0, 0.0)))
    request = CaptureRequest("session", "capture", PartIdentity("part", Hand.RIGHT))
    round_plan = CaptureRoundPlan("front", "capture front")

    frames = adapter.capture_round(request, round_plan, bindings)

    assert tuple(frame.image_bytes for frame in frames) == tuple(expected)


def test_parallel_png_failure_retains_only_canonical_success_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed parallel encode must retain the same prefix evidence as the serial path."""
    adapter = HikvisionCameraAdapter(HikvisionCaptureConfig(hdr=True, hdr_max_retries=0))
    bindings = _bindings(4)
    images = tuple(
        SimpleNamespace(
            binding=binding,
            device_index=index,
            image=SimpleNamespace(ndim=3, shape=(2, 3, 3), marker=index),
        )
        for index, binding in enumerate(bindings)
    )
    monkeypatch.setattr(adapter, "_ensure_open", lambda _bindings: None)
    monkeypatch.setattr(adapter, "_capture_hdr", lambda: (images, 1, (0.0, 0.0, 0.0, 0.0)))
    release = threading.Event()
    lock = threading.Lock()
    entered = 0

    class _Cv2:
        IMWRITE_PNG_COMPRESSION = 16

        @staticmethod
        def imencode(_extension, image, _parameters):
            nonlocal entered
            with lock:
                entered += 1
                if entered == 4:
                    release.set()
            assert release.wait(timeout=2.0)
            if image.marker == 2:
                return False, None
            payload = b"\x89PNG\r\n\x1a\n" + bytes([image.marker])
            return True, SimpleNamespace(tobytes=lambda: payload)

    adapter._dependencies = SimpleNamespace(cv2=_Cv2(), numpy=None)
    request = CaptureRequest("session", "capture", PartIdentity("part", Hand.RIGHT))
    round_plan = CaptureRoundPlan("front", "capture front")

    with pytest.raises(PartialRoundCaptureError) as error_info:
        adapter.capture_round(request, round_plan, bindings)

    assert tuple(frame.camera_slot_id for frame in error_info.value.partial_frames) == ("slot_0", "slot_1")
    assert tuple(frame.image_bytes[-1] for frame in error_info.value.partial_frames) == (0, 1)


def test_parallel_hdr_reports_first_canonical_failure_when_later_failure_finishes_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker completion order must not replace the first topology-ordered HDR error."""
    adapter = HikvisionCameraAdapter(HikvisionCaptureConfig(hdr=True, hdr_max_retries=0))
    bindings = _bindings(4)
    short_images = tuple(
        SimpleNamespace(binding=binding, device_index=index, image=("short", index))
        for index, binding in enumerate(bindings)
    )
    long_images = tuple(
        SimpleNamespace(binding=binding, device_index=index, image=("long", index))
        for index, binding in enumerate(bindings)
    )
    exposure_passes = iter((short_images, long_images))
    monkeypatch.setattr(adapter, "_capture_exposure_pass", lambda _exposure: next(exposure_passes))
    later_failed = threading.Event()

    def fuse(short: tuple[str, int], _long: tuple[str, int]) -> tuple[str, int]:
        index = short[1]
        if index == 0:
            assert later_failed.wait(timeout=2.0)
            raise ValueError("canonical-front-failure")
        if index == 2:
            later_failed.set()
            raise RuntimeError("later-fast-failure")
        return "fused", index

    monkeypatch.setattr(adapter, "_fuse_exposures", fuse)
    monkeypatch.setattr(adapter, "_image_clip_pct", lambda _image: 0.0)

    with pytest.raises(ValueError, match="canonical-front-failure"):
        adapter._capture_hdr()
