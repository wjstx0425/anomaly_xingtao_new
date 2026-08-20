"""Tests for serial-bound single-camera acquisition in the BMW Demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from capture_data.collect_multicamera_dataset import CameraHandle, DeviceDescription

from bmw_inspection.camera import SingleCameraSession, select_unique_device


@dataclass(frozen=True)
class _CameraConfig:
    camera_serial: str = "DA9625347"
    exposure: float = 4000.0
    gain: float = 0.0
    timeout_ms: int = 3000
    warmup_frames: int = 1


class _FakeAdapter:
    def __init__(
        self,
        devices: list[DeviceDescription],
        images: list[np.ndarray] | None = None,
        *,
        read_error: BaseException | None = None,
        start_error: BaseException | None = None,
    ) -> None:
        self.devices = devices
        self.images = list(images or [])
        self.read_error = read_error
        self.start_error = start_error
        self.calls: list[object] = []

    def list_devices(self) -> list[DeviceDescription]:
        self.calls.append("list_devices")
        return self.devices

    def open(self, device: DeviceDescription, gain: float) -> CameraHandle:
        self.calls.append(("open", device.serial, gain))
        return CameraHandle(
            device=device,
            cam=object(),
            frame_info=object(),
            data_buf=object(),
            started=False,
        )

    def start(self, handle: CameraHandle) -> None:
        self.calls.append(("start", handle.device.serial))
        if self.start_error is not None:
            raise self.start_error
        handle.started = True

    def set_exposure(self, handle: CameraHandle, exposure: float) -> None:
        self.calls.append(("set_exposure", handle.device.serial, exposure))

    def trigger(self, handle: CameraHandle) -> None:
        self.calls.append(("trigger", handle.device.serial))

    def read(self, handle: CameraHandle, timeout_ms: int) -> np.ndarray:
        self.calls.append(("read", handle.device.serial, timeout_ms))
        if self.images:
            return self.images.pop(0)
        if self.read_error is not None:
            raise self.read_error
        raise AssertionError("fake adapter has no image or configured read error")

    def stop(self, handle: CameraHandle) -> None:
        self.calls.append(("stop", handle.device.serial))
        handle.started = False

    def restore_continuous(self, handle: CameraHandle) -> None:
        self.calls.append(("restore", handle.device.serial))

    def close(self, handle: CameraHandle) -> None:
        self.calls.append(("close", handle.device.serial))

    def destroy(self, handle: CameraHandle) -> None:
        self.calls.append(("destroy", handle.device.serial))


def _device(index: int, serial: str) -> DeviceDescription:
    return DeviceDescription(index=index, model="MV-CU120-10UM", serial=serial)


def test_select_unique_device_returns_the_exact_serial_identity() -> None:
    target = _device(7, "DA9625347")

    selected = select_unique_device(
        (_device(0, "OTHER"), target),
        "DA9625347",
    )

    assert selected is target


def test_select_unique_device_never_falls_back_to_enumeration_index() -> None:
    with pytest.raises(ValueError, match="DA9625347.*unavailable"):
        select_unique_device((_device(0, "OTHER"),), "DA9625347")


def test_select_unique_device_rejects_duplicate_serial_identity() -> None:
    with pytest.raises(ValueError, match="DA9625347.*multiple"):
        select_unique_device(
            (_device(1, "DA9625347"), _device(8, "DA9625347")),
            "DA9625347",
        )


def test_capture_discards_one_warmup_and_returns_one_accepted_frame() -> None:
    warmup = np.full((3, 4, 3), 11, dtype=np.uint8)
    accepted = np.full((3, 4, 3), 22, dtype=np.uint8)
    adapter = _FakeAdapter([_device(9, "DA9625347")], [warmup, accepted])

    with SingleCameraSession(_CameraConfig(), adapter=adapter) as session:
        image = session.capture()

    assert image is accepted
    assert adapter.calls == [
        "list_devices",
        ("open", "DA9625347", 0.0),
        ("start", "DA9625347"),
        ("set_exposure", "DA9625347", 4000.0),
        ("trigger", "DA9625347"),
        ("read", "DA9625347", 3000),
        ("set_exposure", "DA9625347", 4000.0),
        ("trigger", "DA9625347"),
        ("read", "DA9625347", 3000),
        ("stop", "DA9625347"),
        ("restore", "DA9625347"),
        ("close", "DA9625347"),
        ("destroy", "DA9625347"),
    ]


def test_capture_propagates_read_timeout_and_still_releases_the_camera() -> None:
    timeout = TimeoutError("frame timeout")
    adapter = _FakeAdapter(
        [_device(3, "DA9625347")],
        [np.zeros((2, 2, 3), dtype=np.uint8)],
        read_error=timeout,
    )

    with pytest.raises(TimeoutError) as raised:
        with SingleCameraSession(_CameraConfig(), adapter=adapter) as session:
            session.capture()

    assert raised.value is timeout
    assert adapter.calls[-4:] == [
        ("stop", "DA9625347"),
        ("restore", "DA9625347"),
        ("close", "DA9625347"),
        ("destroy", "DA9625347"),
    ]


def test_start_failure_cleans_the_partially_opened_handle() -> None:
    failure = RuntimeError("start failed")
    adapter = _FakeAdapter(
        [_device(3, "DA9625347")],
        start_error=failure,
    )

    with pytest.raises(RuntimeError) as raised:
        with SingleCameraSession(_CameraConfig(), adapter=adapter):
            pass

    assert raised.value is failure
    assert adapter.calls == [
        "list_devices",
        ("open", "DA9625347", 0.0),
        ("start", "DA9625347"),
        ("restore", "DA9625347"),
        ("close", "DA9625347"),
        ("destroy", "DA9625347"),
    ]
