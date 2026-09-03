"""Hikvision camera and HDR primitives owned by the BMW runtime."""

from __future__ import annotations

import ctypes
import importlib
import logging
import math
import os
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .exposure_fusion import fuse_exposures


DEFAULT_MVS_SDK_PATH = Path("/opt/MVS/Samples/64/Python/MvImport")
FRAME_BUFFER_SIZE = 50 * 1024 * 1024
LOGGER = logging.getLogger(__name__)


def mvs_sdk_path() -> Path:
    """Return the configurable Hikvision SDK Python directory."""
    return Path(os.environ.get("BMW_MVS_SDK_PATH", DEFAULT_MVS_SDK_PATH)).expanduser().resolve()


@dataclass(frozen=True)
class DeviceDescription:
    index: int
    model: str
    serial: str


@dataclass
class CameraHandle:
    device: DeviceDescription
    cam: object
    frame_info: object
    data_buf: object
    started: bool


class CameraAdapter(Protocol):
    def list_devices(self) -> list[DeviceDescription]: ...
    def open(self, device: DeviceDescription, gain: float) -> CameraHandle: ...
    def start(self, handle: CameraHandle) -> None: ...
    def trigger(self, handle: CameraHandle) -> None: ...
    def read(self, handle: CameraHandle, timeout_ms: int) -> np.ndarray: ...
    def set_exposure(self, handle: CameraHandle, exposure: float) -> None: ...
    def stop(self, handle: CameraHandle) -> None: ...
    def restore_continuous(self, handle: CameraHandle) -> None: ...
    def close(self, handle: CameraHandle) -> None: ...
    def destroy(self, handle: CameraHandle) -> None: ...


def _decode_sdk_text(value: object) -> str:
    return bytes(value).split(b"\0", maxsplit=1)[0].decode("utf-8", errors="replace")


def _frame_to_bgr(sdk: object, data_buf: object, frame_info: object) -> np.ndarray:
    width, height = int(frame_info.nWidth), int(frame_info.nHeight)
    data = np.frombuffer(data_buf, dtype=np.uint8, count=int(frame_info.nFrameLen))
    if frame_info.enPixelType == sdk.PixelType_Gvsp_Mono8:
        return cv2.cvtColor(data.reshape(height, width), cv2.COLOR_GRAY2BGR)
    if frame_info.enPixelType == sdk.PixelType_Gvsp_BayerRG8:
        return cv2.cvtColor(data.reshape(height, width), cv2.COLOR_BAYER_RG2BGR)
    if frame_info.enPixelType == sdk.PixelType_Gvsp_BGR8_Packed:
        return data.reshape(height, width, 3).copy()
    raise RuntimeError(f"unsupported Hikvision pixel type: {frame_info.enPixelType}")


class HikvisionAdapter:
    """Lazily loaded adapter for the Hikvision MVS SDK."""

    def __init__(self, sdk: object) -> None:
        self.sdk = sdk
        self._device_list: object | None = None

    @classmethod
    def load(cls) -> "HikvisionAdapter":
        sdk_path = mvs_sdk_path()
        if not (sdk_path / "MvCameraControl_class.py").is_file():
            raise RuntimeError(f"Hikvision MVS Python SDK not found: {sdk_path}")
        if str(sdk_path) not in sys.path:
            sys.path.insert(0, str(sdk_path))
        return cls(importlib.import_module("MvCameraControl_class"))

    def _check(self, code: int, operation: str, device: DeviceDescription | None = None) -> None:
        if code == 0:
            return
        identity = "" if device is None else f" for serial {device.serial}"
        raise RuntimeError(f"{operation} failed{identity}, ret=0x{code:x}")

    def list_devices(self) -> list[DeviceDescription]:
        device_list = self.sdk.MV_CC_DEVICE_INFO_LIST()
        types = self.sdk.MV_GIGE_DEVICE | self.sdk.MV_USB_DEVICE
        self._check(self.sdk.MvCamera.MV_CC_EnumDevices(types, device_list), "EnumDevices")
        self._device_list = device_list
        devices: list[DeviceDescription] = []
        for index in range(device_list.nDeviceNum):
            info = ctypes.cast(device_list.pDeviceInfo[index], ctypes.POINTER(self.sdk.MV_CC_DEVICE_INFO)).contents
            transport = (
                info.SpecialInfo.stGigEInfo
                if info.nTLayerType == self.sdk.MV_GIGE_DEVICE
                else info.SpecialInfo.stUsb3VInfo
            )
            devices.append(DeviceDescription(index, _decode_sdk_text(transport.chModelName), _decode_sdk_text(transport.chSerialNumber)))
        return devices

    def open(self, device: DeviceDescription, gain: float) -> CameraHandle:
        if self._device_list is None:
            raise RuntimeError("list_devices() must be called before open()")
        info = ctypes.cast(
            self._device_list.pDeviceInfo[device.index], ctypes.POINTER(self.sdk.MV_CC_DEVICE_INFO)
        ).contents
        cam = self.sdk.MvCamera()
        handle: CameraHandle | None = None
        try:
            self._check(cam.MV_CC_CreateHandle(info), "CreateHandle", device)
            handle = CameraHandle(device, cam, None, None, False)
            self._check(cam.MV_CC_OpenDevice(self.sdk.MV_ACCESS_Exclusive, 0), "OpenDevice", device)
            for name in ("ExposureAuto", "GainAuto"):
                self._check(cam.MV_CC_SetEnumValue(name, 0), name, device)
            self._set_float(cam, "Gain", gain, device)
            self._check(cam.MV_CC_SetEnumValue("TriggerMode", 1), "TriggerMode", device)
            self._check(cam.MV_CC_SetEnumValue("TriggerSource", 7), "TriggerSource", device)
            handle.frame_info = self.sdk.MV_FRAME_OUT_INFO_EX()
            handle.data_buf = (ctypes.c_ubyte * FRAME_BUFFER_SIZE)()
            return handle
        except BaseException:
            if handle is not None:
                _cleanup_camera_handles([handle], self, include_stop=False)
            raise

    def _set_float(self, cam: object, name: str, value: float, device: DeviceDescription) -> None:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        bounds = self.sdk.MVCC_FLOATVALUE()
        self._check(cam.MV_CC_GetFloatValue(name, bounds), f"GetFloatValue({name})", device)
        if not float(bounds.fMin) <= value <= float(bounds.fMax):
            raise ValueError(f"{name} {value} outside [{bounds.fMin}, {bounds.fMax}]")
        self._check(cam.MV_CC_SetFloatValue(name, float(value)), name, device)

    def start(self, handle: CameraHandle) -> None:
        self._check(handle.cam.MV_CC_StartGrabbing(), "StartGrabbing", handle.device)
        handle.started = True

    def trigger(self, handle: CameraHandle) -> None:
        self._check(handle.cam.MV_CC_SetCommandValue("TriggerSoftware"), "TriggerSoftware", handle.device)

    def read(self, handle: CameraHandle, timeout_ms: int) -> np.ndarray:
        self._check(
            handle.cam.MV_CC_GetOneFrameTimeout(handle.data_buf, len(handle.data_buf), handle.frame_info, timeout_ms),
            "GetOneFrameTimeout",
            handle.device,
        )
        return _frame_to_bgr(self.sdk, handle.data_buf, handle.frame_info)

    def set_exposure(self, handle: CameraHandle, exposure: float) -> None:
        self._set_float(handle.cam, "ExposureTime", exposure, handle.device)

    def stop(self, handle: CameraHandle) -> None:
        self._check(handle.cam.MV_CC_StopGrabbing(), "StopGrabbing", handle.device)
        handle.started = False

    def restore_continuous(self, handle: CameraHandle) -> None:
        self._check(handle.cam.MV_CC_SetEnumValue("TriggerMode", 0), "TriggerMode", handle.device)

    def close(self, handle: CameraHandle) -> None:
        self._check(handle.cam.MV_CC_CloseDevice(), "CloseDevice", handle.device)

    def destroy(self, handle: CameraHandle) -> None:
        self._check(handle.cam.MV_CC_DestroyHandle(), "DestroyHandle", handle.device)


class GroupedTriggerPacer:
    def __init__(
        self,
        interval_seconds: float,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not math.isfinite(interval_seconds) or interval_seconds < 0:
            raise ValueError("capture interval must be non-negative")
        self._interval = interval_seconds
        self._monotonic = monotonic
        self._sleep = sleep
        self._last_pass_at: float | None = None

    def wait(self) -> None:
        now = self._monotonic()
        if self._last_pass_at is not None:
            remaining = self._interval - (now - self._last_pass_at)
            if remaining > 0:
                self._sleep(remaining)
                now = self._monotonic()
        self._last_pass_at = now


@dataclass(frozen=True)
class HdrViewResult:
    camera_slot: int
    short_image: np.ndarray
    long_image: np.ndarray
    fused_image: np.ndarray
    fused_clip_pct: float
    attempt: int


@dataclass(frozen=True)
class SingleViewResult:
    camera_slot: int
    final_image: np.ndarray


def _trigger_and_read(handles: Sequence[CameraHandle], adapter: CameraAdapter, timeout_ms: int) -> list[np.ndarray]:
    for handle in handles:
        adapter.trigger(handle)
    return [adapter.read(handle, timeout_ms) for handle in handles]


def _capture_pass(handles: Sequence[CameraHandle], adapter: CameraAdapter, exposure: float, settle: int, timeout_ms: int, pacer: GroupedTriggerPacer) -> list[np.ndarray]:
    for handle in handles:
        adapter.set_exposure(handle, exposure)
    for _ in range(settle):
        pacer.wait()
        _trigger_and_read(handles, adapter, timeout_ms)
    pacer.wait()
    return _trigger_and_read(handles, adapter, timeout_ms)


def capture_hdr_round(handles: Sequence[CameraHandle], adapter: CameraAdapter, config: object, *, pacer: GroupedTriggerPacer) -> list[HdrViewResult]:
    for attempt in range(int(config.hdr_max_retries) + 1):
        short_images = _capture_pass(handles, adapter, config.short_exposure, config.hdr_settle_frames, config.timeout_ms, pacer)
        long_images = _capture_pass(handles, adapter, config.long_exposure, config.hdr_settle_frames, config.timeout_ms, pacer)
        results: list[HdrViewResult] = []
        for slot, (short_image, long_image) in enumerate(zip(short_images, long_images, strict=True)):
            fused = fuse_exposures(
                [short_image, long_image],
                align=config.align_hdr,
                short_dark_threshold=config.short_dark_threshold,
                long_clip_threshold=config.long_clip_threshold,
                blend_width=config.blend_width,
                blur_size=config.blur_size,
            )
            gray = cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY)
            results.append(HdrViewResult(slot, short_image, long_image, fused, float(np.mean(gray >= 250) * 100), attempt + 1))
        if all(item.fused_clip_pct <= config.hdr_max_clip_pct for item in results):
            return results
    return results


def capture_single_round(handles: Sequence[CameraHandle], adapter: CameraAdapter, exposure: float, timeout_ms: int, pacer: GroupedTriggerPacer) -> list[SingleViewResult]:
    images = _capture_pass(handles, adapter, exposure, 0, timeout_ms, pacer)
    return [SingleViewResult(slot, image) for slot, image in enumerate(images)]


def _cleanup_camera_handles(handles: Sequence[CameraHandle], adapter: CameraAdapter, *, include_stop: bool = True) -> list[BaseException]:
    errors: list[BaseException] = []
    for handle in reversed(handles):
        operations = ([adapter.stop] if include_stop and handle.started else []) + [adapter.restore_continuous, adapter.close, adapter.destroy]
        for operation in operations:
            try:
                operation(handle)
            except BaseException as error:
                errors.append(error)
    return errors


@contextmanager
def open_cameras(devices: Sequence[DeviceDescription], adapter: CameraAdapter, gain: float) -> Iterator[list[CameraHandle]]:
    handles: list[CameraHandle] = []
    try:
        for device in devices:
            handle = adapter.open(device, gain)
            handles.append(handle)
            adapter.start(handle)
        yield handles
    finally:
        errors = _cleanup_camera_handles(handles, adapter)
        if errors and sys.exc_info()[0] is None:
            raise RuntimeError(f"camera cleanup failed: {errors[0]}") from errors[0]


__all__ = [
    "CameraAdapter",
    "CameraHandle",
    "DeviceDescription",
    "GroupedTriggerPacer",
    "HdrViewResult",
    "HikvisionAdapter",
    "capture_hdr_round",
    "capture_single_round",
    "mvs_sdk_path",
    "open_cameras",
]
