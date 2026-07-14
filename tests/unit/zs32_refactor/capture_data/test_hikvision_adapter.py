# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Linux-authoritative contract tests for the Hikvision capture boundary."""

from __future__ import annotations

import sys
from typing import Any

import pytest

from zs32_inspection.capture import CameraBinding
from zs32_inspection.capture.hikvision import (
    DeviceDescription,
    HikvisionCameraAdapter,
    HikvisionCaptureConfig,
    HikvisionCaptureError,
    select_devices_by_serial,
)


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
