"""Serial-bound single-camera acquisition for the BMW bright-streak Demo."""

from __future__ import annotations

from contextlib import AbstractContextManager
from types import TracebackType
from typing import Sequence

import numpy as np

from capture_data.collect_multicamera_dataset import (
    CameraAdapter,
    CameraHandle,
    DeviceDescription,
    GroupedTriggerPacer,
    HikvisionAdapter,
    capture_single_round,
    open_cameras,
)

from .contracts import BrightStreakConfig


def select_unique_device(
    devices: Sequence[DeviceDescription],
    serial: str,
) -> DeviceDescription:
    """Return the only enumerated camera with ``serial``.

    Device indexes are intentionally ignored because MVS enumeration order can
    change between sessions.
    """
    if not isinstance(serial, str) or not serial.strip():
        raise ValueError("camera serial must be a non-empty string")
    matches = [device for device in devices if device.serial == serial]
    if not matches:
        raise ValueError(f"camera serial {serial!r} is unavailable")
    if len(matches) > 1:
        raise ValueError(f"camera serial {serial!r} appears multiple times")
    return matches[0]


class SingleCameraSession:
    """Keep one serial-bound camera open for software-triggered inspections."""

    def __init__(
        self,
        config: BrightStreakConfig,
        *,
        adapter: CameraAdapter | None = None,
    ) -> None:
        self.config = config
        self._adapter = HikvisionAdapter.load() if adapter is None else adapter
        self._camera_context: AbstractContextManager[list[CameraHandle]] | None = None
        self._handles: list[CameraHandle] | None = None
        self._pacer = GroupedTriggerPacer(0.0)

    def __enter__(self) -> SingleCameraSession:
        if self._camera_context is not None:
            raise RuntimeError("single-camera session is already open")
        selected = select_unique_device(
            self._adapter.list_devices(),
            self.config.camera_serial,
        )
        camera_context = open_cameras((selected,), self._adapter, self.config.gain)
        self._camera_context = camera_context
        try:
            handles = camera_context.__enter__()
        except BaseException:
            self._camera_context = None
            raise
        if len(handles) != 1:
            self._camera_context = None
            camera_context.__exit__(None, None, None)
            raise RuntimeError(f"single-camera session opened {len(handles)} handles")
        self._handles = handles
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        camera_context = self._camera_context
        self._camera_context = None
        self._handles = None
        if camera_context is None:
            return False
        return bool(camera_context.__exit__(exc_type, exc_value, traceback))

    def capture(self) -> np.ndarray:
        """Discard configured warm-up frames and return one accepted image."""
        if self._handles is None:
            raise RuntimeError("single-camera session is not open")
        for _ in range(self.config.warmup_frames):
            capture_single_round(
                self._handles,
                self._adapter,
                self.config.exposure,
                self.config.timeout_ms,
                self._pacer,
            )
        results = capture_single_round(
            self._handles,
            self._adapter,
            self.config.exposure,
            self.config.timeout_ms,
            self._pacer,
        )
        if len(results) != 1:
            raise RuntimeError(f"single-camera capture returned {len(results)} frames")
        return results[0].final_image


__all__ = ["SingleCameraSession", "select_unique_device"]
