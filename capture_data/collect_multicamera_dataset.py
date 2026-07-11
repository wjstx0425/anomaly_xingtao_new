# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pure mapping and command-line schema for ZS32 multi-camera capture.

Hardware SDK imports intentionally live outside this module's import path so
the mapping and CLI contract remain testable on machines without the MVS SDK.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import importlib
import re
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from capture_data.exposure_fusion import fuse_exposures

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

    def set_exposure(self, handle: CameraHandle, exposure: float) -> None: ...

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

    def set_exposure(self, handle: CameraHandle, exposure: float) -> None:
        """Set a manual exposure time in microseconds."""
        self._check(
            handle.cam.MV_CC_SetFloatValue("ExposureTime", float(exposure)),
            "ExposureTime",
            handle.device,
        )

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


@dataclass(frozen=True)
class HdrViewResult:
    """Short, long, and fused images for one physical camera slot."""

    camera_slot: int
    short_image: np.ndarray
    long_image: np.ndarray
    fused_image: np.ndarray
    fused_clip_pct: float
    attempt: int


class HdrCaptureConfig(Protocol):
    """Configuration fields required by grouped HDR capture."""

    short_exposure: float
    long_exposure: float
    hdr_settle_frames: int
    timeout_ms: int
    align_hdr: bool
    short_dark_threshold: float
    long_clip_threshold: float
    blend_width: float
    blur_size: int
    hdr_max_retries: int
    hdr_max_clip_pct: float


def capture_exposure_pass(
    handles: Sequence[CameraHandle],
    adapter: CameraAdapter,
    exposure: float,
    settle_frames: int,
    timeout_ms: int,
) -> list[np.ndarray]:
    """Set one exposure on all cameras and return one grouped frame pass."""
    for handle in handles:
        adapter.set_exposure(handle, exposure)
    for _ in range(settle_frames):
        trigger_and_read(handles, adapter, timeout_ms)
    return trigger_and_read(handles, adapter, timeout_ms)


def _image_clip_pct(image: np.ndarray, threshold: int = 250) -> float:
    """Return the percentage of grayscale pixels at or above the clip threshold."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray >= threshold) * 100.0)


def capture_hdr_round(
    handles: Sequence[CameraHandle], adapter: CameraAdapter, config: HdrCaptureConfig
) -> list[HdrViewResult]:
    """Capture and fuse one complete short/long HDR pair for every camera."""
    for attempt_index in range(config.hdr_max_retries + 1):
        short_images = capture_exposure_pass(
            handles,
            adapter,
            config.short_exposure,
            config.hdr_settle_frames,
            config.timeout_ms,
        )
        long_images = capture_exposure_pass(
            handles,
            adapter,
            config.long_exposure,
            config.hdr_settle_frames,
            config.timeout_ms,
        )
        results: list[HdrViewResult] = []
        for camera_slot, (short_image, long_image) in enumerate(
            zip(short_images, long_images, strict=True)
        ):
            fused_image = fuse_exposures(
                [short_image, long_image],
                method="selective",
                align=config.align_hdr,
                short_dark_threshold=config.short_dark_threshold,
                long_clip_threshold=config.long_clip_threshold,
                blend_width=config.blend_width,
                blur_size=config.blur_size,
            )
            results.append(
                HdrViewResult(
                    camera_slot=camera_slot,
                    short_image=short_image,
                    long_image=long_image,
                    fused_image=fused_image,
                    fused_clip_pct=_image_clip_pct(fused_image),
                    attempt=attempt_index + 1,
                )
            )
        if all(result.fused_clip_pct <= config.hdr_max_clip_pct for result in results):
            return results
        if attempt_index == config.hdr_max_retries:
            return results
    raise RuntimeError("unreachable HDR retry state")


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

MANIFEST_COLUMNS = (
    "record_type",
    "session_id",
    "sample_id",
    "group_id",
    "image_index",
    "round",
    "view",
    "device_index",
    "camera_serial",
    "file",
    "source_short",
    "source_long",
    "short_exposure",
    "long_exposure",
    "hdr_attempt",
    "fused_clip_pct",
    "captured_at",
    "sample_status",
    "failed_round",
    "failed_view",
    "failed_device_index",
    "error",
)


@dataclass(frozen=True)
class SessionPaths:
    """Directories and manifest shared by one capture program run."""

    session_id: str
    root: Path
    view_dirs: dict[str, Path]
    manifest_path: Path


class RoundStorageError(RuntimeError):
    """A round image could not be persisted with its capture identity."""

    def __init__(self, message: str, round_name: str, view: str, device_index: int) -> None:
        super().__init__(message)
        self.round_name = round_name
        self.view = view
        self.device_index = device_index


def create_session(args: argparse.Namespace, now: datetime) -> SessionPaths:
    """Create the six view directories and one microsecond-resolution manifest path."""
    session_id = now.strftime("%Y%m%d_%H%M%S_%f")
    root = Path(args.root)
    view_dirs: dict[str, Path] = {}
    for view in (*ROUND_VIEWS["front"], *ROUND_VIEWS["back"]):
        label_parts = (args.label,) if args.label == "normal" else (args.label, args.defect_type)
        directory = root / args.hand / view
        for part in label_parts:
            directory /= part
        directory = directory / session_id / "images"
        directory.mkdir(parents=True, exist_ok=True)
        view_dirs[view] = directory
    manifest_path = root / "manifests" / f"{session_id}.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    return SessionPaths(session_id, root, view_dirs, manifest_path)


def _image_name(args: argparse.Namespace, view: str, group_id: str, image_index: int, kind: str) -> str:
    label = args.label if args.label == "normal" else f"{args.label}_{args.defect_type}"
    return f"{args.hand}_{view}_{label}_{args.part_id}_{group_id}_{image_index:06d}_{kind}.png"


def _write_round_images(
    images: Sequence[tuple[Path, np.ndarray, str, int]], round_name: str
) -> None:
    """Write a complete round through temporary files before publishing any image."""
    staged: list[tuple[Path, Path]] = []
    try:
        for destination, image, view, device_index in images:
            temporary = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
            staged.append((temporary, destination))
            if not cv2.imwrite(str(temporary), image):
                raise RoundStorageError(
                    f"cv2.imwrite returned false for {destination}", round_name, view, device_index
                )
        for temporary, destination in staged:
            temporary.replace(destination)
    except Exception:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        raise


def save_round(
    results: Sequence[HdrViewResult],
    round_name: str,
    sample_id: str,
    group_id: str,
    image_index: int,
    handles: Sequence[CameraHandle],
    args: argparse.Namespace,
    paths: SessionPaths,
) -> list[dict[str, str]]:
    """Atomically store one three-view HDR round and return its manifest rows."""
    if len(results) != 3:
        raise RuntimeError(f"{round_name} round returned {len(results)} results instead of 3")
    captured_at = datetime.now().isoformat(timespec="microseconds")
    writes: list[tuple[Path, np.ndarray, str, int]] = []
    rows: list[dict[str, str]] = []
    for result in results:
        slot = result.camera_slot
        view = view_for(round_name, slot)
        handle = handles[slot]
        fused_path = paths.view_dirs[view] / _image_name(args, view, group_id, image_index, "fused")
        short_path = paths.view_dirs[view] / _image_name(args, view, group_id, image_index, "short")
        long_path = paths.view_dirs[view] / _image_name(args, view, group_id, image_index, "long")
        writes.append((fused_path, result.fused_image, view, handle.device.index))
        if args.save_hdr_sources:
            writes.extend(
                (
                    (short_path, result.short_image, view, handle.device.index),
                    (long_path, result.long_image, view, handle.device.index),
                )
            )
        rows.append(
            {
                "record_type": "image",
                "session_id": paths.session_id,
                "sample_id": sample_id,
                "group_id": group_id,
                "image_index": str(image_index),
                "round": round_name,
                "view": view,
                "device_index": str(handle.device.index),
                "camera_serial": handle.device.serial,
                "file": str(fused_path),
                "source_short": str(short_path) if args.save_hdr_sources else "",
                "source_long": str(long_path) if args.save_hdr_sources else "",
                "short_exposure": str(args.short_exposure),
                "long_exposure": str(args.long_exposure),
                "hdr_attempt": str(result.attempt),
                "fused_clip_pct": str(result.fused_clip_pct),
                "captured_at": captured_at,
                "sample_status": "incomplete",
                "failed_round": "",
                "failed_view": "",
                "failed_device_index": "",
                "error": "",
            }
        )
    _write_round_images(writes, round_name)
    return rows


def _failure_identity(
    error: Exception, round_name: str, handles: Sequence[CameraHandle]
) -> tuple[str, str]:
    if isinstance(error, RoundStorageError):
        return error.view, str(error.device_index)
    match = re.search(r"device\s+(\d+)", str(error), flags=re.IGNORECASE)
    if match is None:
        return "", ""
    device_index = int(match.group(1))
    for slot, handle in enumerate(handles):
        if handle.device.index == device_index:
            return view_for(round_name, slot), str(device_index)
    return "", str(device_index)


def _write_manifest(paths: SessionPaths, rows: Sequence[dict[str, str]]) -> None:
    with paths.manifest_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        if file.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)


def capture_sample(
    handles: Sequence[CameraHandle],
    adapter: CameraAdapter,
    args: argparse.Namespace,
    paths: SessionPaths,
    group_id: str,
    image_index: int,
) -> bool:
    """Capture a paired front/back sample and record explicit completeness."""
    sample_id = f"{args.part_id}_{group_id}_{image_index:06d}"
    rows: list[dict[str, str]] = []
    current_round = "front"
    try:
        for current_round in ("front", "back"):
            results = capture_hdr_round(handles, adapter, args)
            rows.extend(
                save_round(results, current_round, sample_id, group_id, image_index, handles, args, paths)
            )
    except Exception as error:
        failed_view, failed_device = _failure_identity(error, current_round, handles)
        for row in rows:
            row["sample_status"] = "incomplete"
        rows.append(
            {
                **dict.fromkeys(MANIFEST_COLUMNS, ""),
                "record_type": "sample",
                "session_id": paths.session_id,
                "sample_id": sample_id,
                "group_id": group_id,
                "image_index": str(image_index),
                "sample_status": "incomplete",
                "failed_round": current_round,
                "failed_view": failed_view,
                "failed_device_index": failed_device,
                "error": str(error),
            }
        )
        _write_manifest(paths, rows)
        return False
    for row in rows:
        row["sample_status"] = "complete"
    rows.append(
        {
            **dict.fromkeys(MANIFEST_COLUMNS, ""),
            "record_type": "sample",
            "session_id": paths.session_id,
            "sample_id": sample_id,
            "group_id": group_id,
            "image_index": str(image_index),
            "sample_status": "complete",
        }
    )
    _write_manifest(paths, rows)
    return True


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
