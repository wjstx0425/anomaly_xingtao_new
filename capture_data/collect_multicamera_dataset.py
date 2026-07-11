# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pure mapping and command-line schema for ZS32 multi-camera capture.

Hardware SDK imports intentionally live outside this module's import path so
the mapping and CLI contract remain testable on machines without the MVS SDK.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

SDK_PATH = "/opt/MVS/Samples/64/Python/MvImport"
FRAME_BUFFER_SIZE = 50 * 1024 * 1024


@dataclass(frozen=True)
class DeviceDescription:
    """Stable camera identity reported by the Hikvision SDK."""

    index: int
    model: str
    serial: str


@dataclass
class CameraHandle:
    """Resources owned by one opened camera."""

    device: DeviceDescription
    cam: object
    frame_info: object
    data_buf: object
    started: bool


class CameraAdapter(Protocol):
    """Camera operations used by grouped acquisition and lifecycle helpers."""

    def open(self, device: DeviceDescription, gain: float, fps: float) -> CameraHandle: ...

    def start(self, handle: CameraHandle) -> None: ...

    def trigger(self, handle: CameraHandle) -> None: ...

    def read(self, handle: CameraHandle, timeout_ms: int) -> np.ndarray: ...

    def stop(self, handle: CameraHandle) -> None: ...

    def close(self, handle: CameraHandle) -> None: ...

    def destroy(self, handle: CameraHandle) -> None: ...


def _decode_sdk_text(value: object) -> str:
    """Decode a null-terminated SDK character array."""
    return bytes(value).split(b"\0", maxsplit=1)[0].decode("utf-8", errors="replace")


class HikvisionAdapter:
    """Thin, lazily loaded adapter around the Hikvision MVS SDK."""

    def __init__(self, sdk: object) -> None:
        self.sdk = sdk

    @classmethod
    def load(cls) -> HikvisionAdapter:
        """Load the MVS SDK only when hardware access is requested."""
        if SDK_PATH not in sys.path:
            sys.path.append(SDK_PATH)
        return cls(importlib.import_module("MvCameraControl_class"))

    def _check(self, ret: int, operation: str, device: DeviceDescription | None = None) -> None:
        if ret == 0:
            return
        identity = "" if device is None else f" for device {device.index} serial {device.serial}"
        raise RuntimeError(f"{operation} failed{identity}, ret=0x{ret:x}")

    def list_devices(self) -> list[DeviceDescription]:
        """Enumerate attached GigE and USB cameras."""
        device_list = self.sdk.MV_CC_DEVICE_INFO_LIST()
        device_types = self.sdk.MV_GIGE_DEVICE | self.sdk.MV_USB_DEVICE
        self._check(self.sdk.MvCamera.MV_CC_EnumDevices(device_types, device_list), "EnumDevices")
        devices: list[DeviceDescription] = []
        for index in range(device_list.nDeviceNum):
            info = ctypes.cast(
                device_list.pDeviceInfo[index], ctypes.POINTER(self.sdk.MV_CC_DEVICE_INFO)
            ).contents
            if info.nTLayerType == self.sdk.MV_GIGE_DEVICE:
                transport = info.SpecialInfo.stGigEInfo
            else:
                transport = info.SpecialInfo.stUsb3VInfo
            devices.append(
                DeviceDescription(
                    index,
                    _decode_sdk_text(transport.chModelName),
                    _decode_sdk_text(transport.chSerialNumber),
                )
            )
        return devices

    def open(self, device: DeviceDescription, gain: float, fps: float) -> CameraHandle:
        """Create, open, and configure a camera for software triggering."""
        device_list = self.sdk.MV_CC_DEVICE_INFO_LIST()
        device_types = self.sdk.MV_GIGE_DEVICE | self.sdk.MV_USB_DEVICE
        self._check(self.sdk.MvCamera.MV_CC_EnumDevices(device_types, device_list), "EnumDevices", device)
        if device.index < 0 or device.index >= device_list.nDeviceNum:
            raise RuntimeError(f"device index {device.index} serial {device.serial} is unavailable")
        info = ctypes.cast(
            device_list.pDeviceInfo[device.index], ctypes.POINTER(self.sdk.MV_CC_DEVICE_INFO)
        ).contents
        cam = self.sdk.MvCamera()
        created = False
        try:
            self._check(cam.MV_CC_CreateHandle(info), "CreateHandle", device)
            created = True
            self._check(cam.MV_CC_OpenDevice(self.sdk.MV_ACCESS_Exclusive, 0), "OpenDevice", device)
            for name, value in (("ExposureAuto", 0), ("GainAuto", 0), ("TriggerMode", 1), ("TriggerSource", 7)):
                self._check(cam.MV_CC_SetEnumValue(name, value), name, device)
            self._check(cam.MV_CC_SetFloatValue("Gain", float(gain)), "Gain", device)
            self._check(cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True), "FrameRateEnable", device)
            self._check(cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(fps)), "FrameRate", device)
        except Exception:
            if created:
                try:
                    cam.MV_CC_CloseDevice()
                except Exception:
                    pass
                try:
                    cam.MV_CC_DestroyHandle()
                except Exception:
                    pass
            raise
        return CameraHandle(
            device=device,
            cam=cam,
            frame_info=self.sdk.MV_FRAME_OUT_INFO_EX(),
            data_buf=(ctypes.c_ubyte * FRAME_BUFFER_SIZE)(),
            started=False,
        )

    def start(self, handle: CameraHandle) -> None:
        """Start acquisition on one camera."""
        self._check(handle.cam.MV_CC_StartGrabbing(), "StartGrabbing", handle.device)
        handle.started = True

    def trigger(self, handle: CameraHandle) -> None:
        """Issue one software trigger."""
        self._check(handle.cam.MV_CC_SetCommandValue("TriggerSoftware"), "TriggerSoftware", handle.device)

    def read(self, handle: CameraHandle, timeout_ms: int) -> np.ndarray:
        """Read and convert one triggered frame."""
        self._check(
            handle.cam.MV_CC_GetOneFrameTimeout(handle.data_buf, len(handle.data_buf), handle.frame_info, timeout_ms),
            "GetOneFrameTimeout",
            handle.device,
        )
        capture_path = str(Path(__file__).resolve().parent)
        if capture_path not in sys.path:
            sys.path.append(capture_path)
        converter = importlib.import_module("collect_dataset").convert_frame_to_bgr
        return converter(handle.data_buf, handle.frame_info)

    def stop(self, handle: CameraHandle) -> None:
        """Stop acquisition."""
        self._check(handle.cam.MV_CC_StopGrabbing(), "StopGrabbing", handle.device)
        handle.started = False

    def close(self, handle: CameraHandle) -> None:
        """Close the camera device."""
        self._check(handle.cam.MV_CC_CloseDevice(), "CloseDevice", handle.device)

    def destroy(self, handle: CameraHandle) -> None:
        """Destroy the SDK camera handle."""
        self._check(handle.cam.MV_CC_DestroyHandle(), "DestroyHandle", handle.device)


def trigger_and_read(
    handles: Sequence[CameraHandle], adapter: CameraAdapter, timeout_ms: int
) -> list[np.ndarray]:
    """Trigger every camera before reading any frame."""
    for handle in handles:
        adapter.trigger(handle)
    return [adapter.read(handle, timeout_ms) for handle in handles]


def close_cameras(handles: Sequence[CameraHandle], adapter: CameraAdapter) -> None:
    """Best-effort cleanup of camera resources in reverse setup order."""
    for handle in reversed(handles):
        if handle.started:
            try:
                adapter.stop(handle)
            except Exception:
                pass
        try:
            adapter.close(handle)
        except Exception:
            pass
        try:
            adapter.destroy(handle)
        except Exception:
            pass


@contextmanager
def open_cameras(
    devices: Sequence[DeviceDescription], adapter: CameraAdapter, gain: float, fps: float
) -> Iterator[list[CameraHandle]]:
    """Open and start cameras, cleaning partial setup on every exit path."""
    handles: list[CameraHandle] = []
    try:
        for device in devices:
            handle = adapter.open(device, gain, fps)
            handles.append(handle)
            adapter.start(handle)
        yield handles
    finally:
        close_cameras(handles, adapter)


ROUND_VIEWS: dict[str, tuple[str, str, str]] = {
    "front": ("front", "front_left", "front_right"),
    "back": ("back", "back_left", "back_right"),
}


def validate_devices(devices: Sequence[int]) -> tuple[int, int, int]:
    """Validate and preserve the physical ordering of three camera indices.

    Args:
        devices: Device indices ordered as front, left-side, and right-side camera slots.

    Returns:
        The three device indices with their input order preserved.

    Raises:
        ValueError: If the input does not contain exactly three unique indices.
    """
    if len(devices) != 3 or len(set(devices)) != 3:
        msg = "--devices requires exactly three unique device indices"
        raise ValueError(msg)
    return devices[0], devices[1], devices[2]


def view_for(round_name: str, camera_slot: int) -> str:
    """Return the view assigned to a camera slot for one capture round.

    Args:
        round_name: Capture round name, either ``front`` or ``back``.
        camera_slot: Zero-based physical camera slot.

    Returns:
        The dataset view name assigned to the slot.
    """
    return ROUND_VIEWS[round_name][camera_slot]


class _CaptureArgumentParser(argparse.ArgumentParser):
    """Argument parser with capture-only required options."""

    def parse_args(
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        """Allow discovery alone while requiring label and HDR for capture."""
        parsed = super().parse_args(args, namespace)
        if not parsed.list_devices:
            if parsed.label is None:
                self.error("the following arguments are required: --label")
            if not parsed.hdr:
                self.error("the following arguments are required: --hdr")
        return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the SDK-free CLI parser for six-view ZS32 capture.

    Returns:
        The configured argument parser.
    """
    parser = _CaptureArgumentParser(description="Collect six-view ZS32 HDR images from three cameras.")
    parser.add_argument("--devices", nargs=3, type=int, default=[0, 1, 2])
    parser.add_argument("--hand", choices=("left",), default="left")
    parser.add_argument("--label", choices=("normal", "defect"))
    parser.add_argument("--defect-type", default="")
    parser.add_argument("--part-id", default="part001")
    parser.add_argument("--group-count", type=int, default=1)
    parser.add_argument("--images-per-group", type=int, default=1)
    parser.add_argument("--manual-load", action="store_true")

    parser.add_argument("--hdr", action="store_true")
    parser.add_argument("--short-exposure", type=float, default=7000.0)
    parser.add_argument("--long-exposure", type=float, default=40000.0)
    parser.add_argument("--gain", type=float)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--hdr-settle-frames", type=int, default=5)
    parser.add_argument("--timeout-ms", type=int, default=3000)
    parser.add_argument("--short-dark-threshold", type=float, default=70.0)
    parser.add_argument("--long-clip-threshold", type=float, default=245.0)
    parser.add_argument("--blend-width", type=float, default=18.0)
    parser.add_argument("--blur-size", type=int, default=31)
    parser.add_argument("--align-hdr", action="store_true")
    parser.add_argument("--save-hdr-sources", action="store_true")
    parser.add_argument("--hdr-max-retries", type=int, default=2)
    parser.add_argument("--hdr-max-clip-pct", type=float, default=12.0)

    parser.add_argument("--root", default="./dataset")
    parser.add_argument("--list-devices", action="store_true")
    return parser
