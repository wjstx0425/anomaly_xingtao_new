"""Tests for the BMW laboratory six-view acquisition boundary."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import runpy

import cv2
import numpy as np
import pytest

from capture_data.collect_multicamera_dataset import DeviceDescription

from bmw_inspection.lab.capture import (
    LabCameraSession,
    build_capture_set,
    load_capture_set,
    map_round_images,
    select_serial_bound_devices,
)
from bmw_inspection.lab.config import CaptureSettings, load_experiment_config
from bmw_inspection.lab.contracts import ViewId


SERIALS = ("DA9805574", "DA9625347", "DB0968108")


def _device(index: int, serial: str) -> DeviceDescription:
    return DeviceDescription(index=index, model="MV-CU120-10UM", serial=serial)


def _image(value: int) -> np.ndarray:
    return np.full((4, 6, 3), value, dtype=np.uint8)


class _FakeAdapter:
    def __init__(
        self,
        devices: list[DeviceDescription],
        images: list[np.ndarray],
        *,
        read_error: BaseException | None = None,
    ) -> None:
        self.devices = devices
        self.images = list(images)
        self.read_error = read_error
        self.calls: list[object] = []

    @property
    def opened_serials(self) -> list[str]:
        return [call[1] for call in self.calls if isinstance(call, tuple) and call[0] == "open"]

    def list_devices(self) -> list[DeviceDescription]:
        self.calls.append("list_devices")
        return self.devices

    def open(self, device: DeviceDescription, gain: float):
        from capture_data.collect_multicamera_dataset import CameraHandle

        self.calls.append(("open", device.serial, gain))
        return CameraHandle(device=device, cam=object(), frame_info=object(), data_buf=object(), started=False)

    def start(self, handle) -> None:
        self.calls.append(("start", handle.device.serial))
        handle.started = True

    def set_exposure(self, handle, exposure: float) -> None:
        self.calls.append(("set_exposure", handle.device.serial, exposure))

    def trigger(self, handle) -> None:
        self.calls.append(("trigger", handle.device.serial))

    def read(self, handle, timeout_ms: int) -> np.ndarray:
        self.calls.append(("read", handle.device.serial, timeout_ms))
        if self.read_error is not None:
            raise self.read_error
        return self.images.pop(0)

    def stop(self, handle) -> None:
        self.calls.append(("stop", handle.device.serial))
        handle.started = False

    def restore_continuous(self, handle) -> None:
        self.calls.append(("restore", handle.device.serial))

    def close(self, handle) -> None:
        self.calls.append(("close", handle.device.serial))

    def destroy(self, handle) -> None:
        self.calls.append(("destroy", handle.device.serial))


@pytest.fixture
def config():
    project_root = Path(__file__).resolve().parents[4]
    loaded = load_experiment_config(project_root / "configs/bmw/experiments/bmw_lab_v1.json")
    return replace(
        loaded,
        capture=CaptureSettings(
            image_width=6,
            image_height=4,
            exposure=4000.0,
            gain=0.0,
            timeout_ms=3000,
            warmup_frames=1,
        ),
    )


def test_serial_binding_uses_declared_slot_order_not_enumeration_order() -> None:
    devices = (
        _device(8, "DB0968108"),
        _device(3, "DA9625347"),
        _device(1, "DA9805574"),
    )

    selected = select_serial_bound_devices(devices, SERIALS)

    assert tuple(device.serial for device in selected) == SERIALS


def test_serial_binding_rejects_missing_or_duplicate_camera_identity() -> None:
    with pytest.raises(ValueError, match="DA9625347.*unavailable"):
        select_serial_bound_devices((_device(1, "DA9805574"),), SERIALS)
    with pytest.raises(ValueError, match="DA9805574.*multiple"):
        select_serial_bound_devices(
            (_device(1, "DA9805574"), _device(2, "DA9805574"), _device(3, "DA9625347")),
            SERIALS,
        )


def test_two_rounds_map_three_camera_images_to_the_canonical_six_views() -> None:
    front = map_round_images("front", (_image(10), _image(20), _image(30)))
    back = map_round_images("back", (_image(40), _image(50), _image(60)))
    capture_set = build_capture_set(front, back)

    assert tuple(front) == (ViewId.FRONT, ViewId.FRONT_LEFT, ViewId.FRONT_RIGHT)
    assert tuple(back) == (ViewId.BACK, ViewId.BACK_LEFT, ViewId.BACK_RIGHT)
    assert tuple(capture_set.views) == tuple(ViewId)


def test_two_rounds_bind_declared_serials_and_map_six_views(config) -> None:
    adapter = _FakeAdapter(
        [_device(11, "DB0968108"), _device(3, "DA9625347"), _device(7, "DA9805574")],
        [_image(value) for value in range(12)],
    )

    with LabCameraSession(config, adapter=adapter) as session:
        front = session.capture_round("front")
        back = session.capture_round("back")

    assert tuple(front) == (ViewId.FRONT, ViewId.FRONT_LEFT, ViewId.FRONT_RIGHT)
    assert tuple(back) == (ViewId.BACK, ViewId.BACK_LEFT, ViewId.BACK_RIGHT)
    assert adapter.opened_serials == list(SERIALS)
    assert [call[1] for call in adapter.calls if isinstance(call, tuple) and call[0] == "trigger"] == list(SERIALS) * 4


def test_camera_session_releases_every_opened_handle_after_read_failure(config) -> None:
    error = TimeoutError("frame timeout")
    adapter = _FakeAdapter(
        [_device(1, "DA9805574"), _device(2, "DA9625347"), _device(3, "DB0968108")],
        [],
        read_error=error,
    )

    with pytest.raises(TimeoutError) as raised:
        with LabCameraSession(config, adapter=adapter) as session:
            session.capture_round("front")

    assert raised.value is error
    assert adapter.calls[-12:] == [
        ("stop", "DB0968108"), ("restore", "DB0968108"), ("close", "DB0968108"), ("destroy", "DB0968108"),
        ("stop", "DA9625347"), ("restore", "DA9625347"), ("close", "DA9625347"), ("destroy", "DA9625347"),
        ("stop", "DA9805574"), ("restore", "DA9805574"), ("close", "DA9805574"), ("destroy", "DA9805574"),
    ]


@pytest.mark.parametrize("round_id", ("", "side", "FRONT"))
def test_round_mapping_rejects_unknown_round_ids(round_id: str) -> None:
    with pytest.raises(ValueError, match="round"):
        map_round_images(round_id, (_image(10), _image(20), _image(30)))


def test_round_mapping_rejects_incomplete_or_dimension_mismatched_frames() -> None:
    with pytest.raises(ValueError, match="three"):
        map_round_images("front", (_image(10), _image(20)))
    with pytest.raises(ValueError, match="same dimensions"):
        map_round_images("front", (_image(10), _image(20), np.zeros((5, 6, 3), dtype=np.uint8)))


def test_build_capture_set_rejects_incomplete_or_duplicate_canonical_views() -> None:
    front = map_round_images("front", (_image(10), _image(20), _image(30)))
    back = map_round_images("back", (_image(40), _image(50), _image(60)))

    with pytest.raises(ValueError, match="missing"):
        build_capture_set(front, {ViewId.BACK: _image(40), ViewId.BACK_LEFT: _image(50)})
    with pytest.raises(ValueError, match="overlap"):
        build_capture_set(front, {**back, ViewId.FRONT: _image(70)})


def test_build_capture_set_owns_read_only_copies_of_source_images() -> None:
    front = map_round_images("front", (_image(10), _image(20), _image(30)))
    back = map_round_images("back", (_image(40), _image(50), _image(60)))

    capture_set = build_capture_set(front, back)
    stored = capture_set.views[ViewId.FRONT].image
    front[ViewId.FRONT][0, 0, 0] = 99

    assert int(stored[0, 0, 0]) == 10
    assert not np.shares_memory(stored, front[ViewId.FRONT])
    assert stored.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        stored[0, 0, 0] = 77


@pytest.mark.parametrize(
    "invalid_image",
    (
        np.zeros((4, 6, 3), dtype=np.float32),
        [[0]],
    ),
)
def test_build_capture_set_rejects_non_uint8_ndarrays(invalid_image: object) -> None:
    front = map_round_images("front", (_image(10), _image(20), _image(30)))
    back: dict[ViewId, object] = {
        ViewId.BACK: invalid_image,
        ViewId.BACK_LEFT: _image(50),
        ViewId.BACK_RIGHT: _image(60),
    }

    with pytest.raises(ValueError, match="uint8 numpy array"):
        build_capture_set(front, back)  # type: ignore[arg-type]


def test_build_capture_set_rejects_dimension_drift_between_rounds() -> None:
    front = map_round_images("front", (_image(10), _image(20), _image(30)))
    back = map_round_images(
        "back",
        tuple(np.full((5, 6, 3), value, dtype=np.uint8) for value in (40, 50, 60)),
    )

    with pytest.raises(ValueError, match="six capture images must have the same dimensions"):
        build_capture_set(front, back)


def test_load_capture_set_accepts_exactly_six_canonical_bmp_or_png_views(tmp_path: Path) -> None:
    expected = {
        ViewId.FRONT: 10,
        ViewId.FRONT_LEFT: 20,
        ViewId.FRONT_RIGHT: 30,
        ViewId.BACK: 40,
        ViewId.BACK_LEFT: 50,
        ViewId.BACK_RIGHT: 60,
    }
    for index, (view_id, value) in enumerate(expected.items()):
        suffix = ".png" if index % 2 else ".bmp"
        assert cv2.imwrite(str(tmp_path / f"{view_id.value}{suffix}"), _image(value))

    capture_set = load_capture_set(tmp_path)

    assert tuple(capture_set.views) == tuple(ViewId)
    assert {view_id: int(view.image[0, 0, 0]) for view_id, view in capture_set.views.items()} == expected


def test_load_capture_set_rejects_missing_or_duplicate_views_before_inference(tmp_path: Path) -> None:
    assert cv2.imwrite(str(tmp_path / "front.png"), _image(10))

    with pytest.raises(ValueError, match="missing"):
        load_capture_set(tmp_path)

    for view_id in tuple(ViewId)[1:]:
        assert cv2.imwrite(str(tmp_path / f"{view_id.value}.png"), _image(20))
    assert cv2.imwrite(str(tmp_path / "front.bmp"), _image(30))

    with pytest.raises(ValueError, match="duplicate"):
        load_capture_set(tmp_path)


def test_lab_cli_exposes_only_capture_inputs_needed_before_runtime_exists() -> None:
    project_root = Path(__file__).resolve().parents[4]

    namespace = runpy.run_path(str(project_root / "pipeline/bmw_lab_inspection.py"))
    args = namespace["build_parser"]().parse_args(
        ["--config", "lab.json", "--capture-set", "saved-set", "--no-gui", "--output-root", "out"]
    )

    assert args.config == Path("lab.json")
    assert args.capture_set == Path("saved-set")
    assert args.no_gui is True
    assert args.output_root == Path("out")


def test_lab_cli_does_not_depend_on_python_311_strenum() -> None:
    project_root = Path(__file__).resolve().parents[4]
    cli_path = project_root / "pipeline/bmw_lab_inspection.py"
    source = cli_path.read_text(encoding="utf-8")
    namespace = runpy.run_path(str(cli_path))

    assert "StrEnum" not in source
    assert issubclass(namespace["CaptureState"], str)
    assert namespace["CaptureState"].WAITING_FRONT.value == "WAITING_FRONT"
