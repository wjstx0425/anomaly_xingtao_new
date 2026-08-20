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
import logging
import math
import re
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

if __package__:
    from capture_data.exposure_fusion import fuse_exposures
else:
    from exposure_fusion import fuse_exposures

SDK_PATH = "/opt/MVS/Samples/64/Python/MvImport"
FRAME_BUFFER_SIZE = 50 * 1024 * 1024
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceDescription:
    """Stable camera identity reported by the Hikvision SDK."""

    index: int
    model: str
    serial: str


@dataclass(frozen=True)
class CameraSerials:
    """USB serial identities ordered by physical camera slot."""

    front: str
    left: str
    right: str


DEFAULT_SERIALS = CameraSerials("DA9805574", "DA9625347", "DB0998274")


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
    """Decode a null-terminated SDK character array."""
    return bytes(value).split(b"\0", maxsplit=1)[0].decode("utf-8", errors="replace")


class HikvisionAdapter:
    """Thin, lazily loaded adapter around the Hikvision MVS SDK."""

    def __init__(self, sdk: object) -> None:
        self.sdk = sdk
        self._device_list: object | None = None

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
        self._device_list = device_list
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

    def open(self, device: DeviceDescription, gain: float) -> CameraHandle:
        """Create, open, and configure a camera for software triggering."""
        if self._device_list is None:
            raise RuntimeError("list_devices() must be called before open()")
        device_list = self._device_list
        if device.index < 0 or device.index >= device_list.nDeviceNum:
            raise RuntimeError(f"device index {device.index} serial {device.serial} is unavailable")
        info = ctypes.cast(
            device_list.pDeviceInfo[device.index], ctypes.POINTER(self.sdk.MV_CC_DEVICE_INFO)
        ).contents
        if info.nTLayerType == self.sdk.MV_GIGE_DEVICE:
            transport = info.SpecialInfo.stGigEInfo
        else:
            transport = info.SpecialInfo.stUsb3VInfo
        cached_serial = _decode_sdk_text(transport.chSerialNumber)
        if cached_serial != device.serial:
            raise RuntimeError(
                f"cached device serial mismatch for index {device.index}: "
                f"requested {device.serial}, enumerated {cached_serial}"
            )
        cam = self.sdk.MvCamera()
        handle: CameraHandle | None = None
        try:
            self._check(cam.MV_CC_CreateHandle(info), "CreateHandle", device)
            handle = CameraHandle(device=device, cam=cam, frame_info=None, data_buf=None, started=False)
            self._check(cam.MV_CC_OpenDevice(self.sdk.MV_ACCESS_Exclusive, 0), "OpenDevice", device)
            for name, value in (("ExposureAuto", 0), ("GainAuto", 0)):
                self._check(cam.MV_CC_SetEnumValue(name, value), name, device)
            self.validate_float_range(cam, "Gain", gain)
            self._check(cam.MV_CC_SetFloatValue("Gain", float(gain)), "Gain", device)
            for name, value in (("TriggerMode", 1), ("TriggerSource", 7)):
                self._check(cam.MV_CC_SetEnumValue(name, value), name, device)
            handle.frame_info = self.sdk.MV_FRAME_OUT_INFO_EX()
            handle.data_buf = (ctypes.c_ubyte * FRAME_BUFFER_SIZE)()
        except BaseException as error:
            if handle is not None:
                cleanup_errors = _cleanup_camera_handles([handle], self, include_stop=False)
                _report_cleanup_errors(error, cleanup_errors)
            raise
        if handle is None:  # pragma: no cover - CreateHandle success always assigns it
            raise RuntimeError("camera handle setup ended without a handle")
        return handle

    def validate_float_range(self, cam: object, name: str, value: float) -> None:
        """Validate a finite float value against one SDK camera node range.

        Args:
            cam: Open Hikvision SDK camera object.
            name: Float node name.
            value: Requested node value.

        Raises:
            RuntimeError: If the SDK range query fails.
            ValueError: If the value is non-finite or outside the SDK range.
        """
        if not math.isfinite(value):
            raise ValueError(f"{name} requested value {value} must be finite")
        float_value = self.sdk.MVCC_FLOATVALUE()
        self._check(cam.MV_CC_GetFloatValue(name, float_value), f"GetFloatValue({name})")
        minimum = float(float_value.fMin)
        maximum = float(float_value.fMax)
        if value < minimum or value > maximum:
            raise ValueError(
                f"{name} requested value {value} is outside SDK range [{minimum}, {maximum}]"
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
        self.validate_float_range(handle.cam, "ExposureTime", exposure)
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

    def restore_continuous(self, handle: CameraHandle) -> None:
        """Restore continuous acquisition before releasing a camera.

        Args:
            handle: Camera whose trigger mode should be disabled.

        Raises:
            RuntimeError: If the SDK cannot disable trigger mode.
        """
        self._check(handle.cam.MV_CC_SetEnumValue("TriggerMode", 0), "TriggerMode", handle.device)

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
    capture_interval: float


class GroupedTriggerPacer:
    """Keep a minimum application-level delay between grouped trigger passes."""

    def __init__(
        self,
        interval_seconds: float,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not math.isfinite(interval_seconds) or interval_seconds < 0:
            raise ValueError("capture interval must be greater than or equal to zero")
        self._interval = interval_seconds
        self._monotonic = monotonic
        self._sleep = sleep
        self._last_pass_at: float | None = None

    def wait(self) -> None:
        """Wait until the next grouped trigger pass is allowed to start."""
        now = self._monotonic()
        if self._last_pass_at is not None:
            remaining = self._interval - (now - self._last_pass_at)
            if remaining > 0:
                self._sleep(remaining)
                now = self._monotonic()
        self._last_pass_at = now


def capture_exposure_pass(
    handles: Sequence[CameraHandle],
    adapter: CameraAdapter,
    exposure: float,
    settle_frames: int,
    timeout_ms: int,
    pacer: GroupedTriggerPacer,
) -> list[np.ndarray]:
    """Set one exposure on all cameras and return one grouped frame pass."""
    for handle in handles:
        adapter.set_exposure(handle, exposure)
    for _ in range(settle_frames):
        pacer.wait()
        trigger_and_read(handles, adapter, timeout_ms)
    pacer.wait()
    return trigger_and_read(handles, adapter, timeout_ms)


def _image_clip_pct(image: np.ndarray, threshold: int = 250) -> float:
    """Return the percentage of grayscale pixels at or above the clip threshold."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray >= threshold) * 100.0)


def capture_hdr_round(
    handles: Sequence[CameraHandle],
    adapter: CameraAdapter,
    config: HdrCaptureConfig,
    *,
    pacer: GroupedTriggerPacer | None = None,
) -> list[HdrViewResult]:
    """Capture and fuse one complete short/long HDR pair for every camera."""
    pacer = GroupedTriggerPacer(config.capture_interval) if pacer is None else pacer
    for attempt_index in range(config.hdr_max_retries + 1):
        short_images = capture_exposure_pass(
            handles,
            adapter,
            config.short_exposure,
            config.hdr_settle_frames,
            config.timeout_ms,
            pacer,
        )
        long_images = capture_exposure_pass(
            handles,
            adapter,
            config.long_exposure,
            config.hdr_settle_frames,
            config.timeout_ms,
            pacer,
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


@dataclass(frozen=True)
class SingleViewResult:
    """One single-exposure image for a physical camera slot."""

    camera_slot: int
    final_image: np.ndarray
    source_short: np.ndarray | None = None
    source_long: np.ndarray | None = None


def capture_single_round(
    handles: Sequence[CameraHandle],
    adapter: CameraAdapter,
    exposure: float,
    timeout_ms: int,
    pacer: GroupedTriggerPacer,
) -> list[SingleViewResult]:
    """Set all exposures, then trigger all cameras before reading any camera."""
    for handle in handles:
        adapter.set_exposure(handle, exposure)
    pacer.wait()
    images = trigger_and_read(handles, adapter, timeout_ms)
    return [SingleViewResult(slot, image) for slot, image in enumerate(images)]


def capture_round(
    handles: Sequence[CameraHandle],
    adapter: CameraAdapter,
    args: argparse.Namespace,
    pacer: GroupedTriggerPacer,
) -> list[HdrViewResult] | list[SingleViewResult]:
    """Dispatch one round to explicit HDR or the default single-exposure path."""
    if args.hdr:
        return capture_hdr_round(handles, adapter, args, pacer=pacer)
    return capture_single_round(handles, adapter, args.exposure, args.timeout_ms, pacer)


def _cleanup_camera_handles(
    handles: Sequence[CameraHandle],
    adapter: CameraAdapter,
    *,
    include_stop: bool = True,
) -> list[tuple[str, DeviceDescription, BaseException]]:
    """Attempt every cleanup operation and return all failures."""
    errors: list[tuple[str, DeviceDescription, BaseException]] = []
    for handle in reversed(handles):
        operations: list[tuple[str, Callable[[CameraHandle], None]]] = []
        if include_stop and handle.started:
            operations.append(("stop", adapter.stop))
        operations.extend(
            (
                ("restore", adapter.restore_continuous),
                ("close", adapter.close),
                ("destroy", adapter.destroy),
            )
        )
        for operation, callback in operations:
            try:
                callback(handle)
            except BaseException as error:
                errors.append((operation, handle.device, error))
    return errors


def _cleanup_message(errors: Sequence[tuple[str, DeviceDescription, BaseException]]) -> str:
    """Format cleanup failures with camera identities."""
    details = "; ".join(
        f"{operation} device {device.index} serial {device.serial}: {error}"
        for operation, device, error in errors
    )
    return f"camera cleanup failed: {details}"


def _report_cleanup_errors(
    primary_error: BaseException,
    cleanup_errors: Sequence[tuple[str, DeviceDescription, BaseException]],
) -> None:
    """Report cleanup failures without replacing an active primary exception."""
    if not cleanup_errors:
        return
    message = _cleanup_message(cleanup_errors)
    if hasattr(primary_error, "add_note"):
        primary_error.add_note(message)
    else:  # pragma: no cover - Python 3.10 compatibility
        LOGGER.error(message)


def close_cameras(handles: Sequence[CameraHandle], adapter: CameraAdapter) -> None:
    """Clean every camera in reverse order and report all cleanup failures.

    Args:
        handles: Camera handles to release.
        adapter: Camera operations used for cleanup.

    Raises:
        RuntimeError: After all cleanup attempts if any operation failed.
    """
    errors = _cleanup_camera_handles(handles, adapter)
    if errors:
        raise RuntimeError(_cleanup_message(errors)) from errors[0][2]


@contextmanager
def open_cameras(
    devices: Sequence[DeviceDescription], adapter: CameraAdapter, gain: float
) -> Iterator[list[CameraHandle]]:
    """Open and start cameras, cleaning partial setup on every exit path."""
    handles: list[CameraHandle] = []
    try:
        for device in devices:
            handle = adapter.open(device, gain)
            handles.append(handle)
            adapter.start(handle)
        yield handles
    except BaseException as error:
        _report_cleanup_errors(error, _cleanup_camera_handles(handles, adapter))
        raise
    else:
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
    "capture_mode",
    "exposure",
    "gain",
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
    staged: list[tuple[Path, Path, str, int]] = []
    published: list[Path] = []
    current_view = ""
    current_device_index = -1
    try:
        for destination, image, view, device_index in images:
            current_view = view
            current_device_index = device_index
            temporary = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
            staged.append((temporary, destination, view, device_index))
            if not cv2.imwrite(str(temporary), image):
                raise RoundStorageError(
                    f"cv2.imwrite returned false for {destination}", round_name, view, device_index
                )
        for temporary, destination, view, device_index in staged:
            current_view = view
            current_device_index = device_index
            temporary.replace(destination)
            published.append(destination)
    except Exception as error:
        for temporary, _, _, _ in staged:
            temporary.unlink(missing_ok=True)
        for destination in published:
            destination.unlink(missing_ok=True)
        if isinstance(error, RoundStorageError):
            raise
        raise RoundStorageError(
            f"failed to publish {round_name} view {current_view} for device "
            f"{current_device_index}: {error}",
            round_name,
            current_view,
            current_device_index,
        ) from error


def save_round(
    results: Sequence[HdrViewResult] | Sequence[SingleViewResult],
    round_name: str,
    sample_id: str,
    group_id: str,
    image_index: int,
    handles: Sequence[CameraHandle],
    args: argparse.Namespace,
    paths: SessionPaths,
) -> list[dict[str, str]]:
    """Atomically store one three-view round and return its manifest rows."""
    if len(results) != 3:
        raise RuntimeError(f"{round_name} round returned {len(results)} results instead of 3")
    slots = {result.camera_slot for result in results}
    expected_slots = {0, 1, 2}
    if slots != expected_slots:
        raise RuntimeError(
            f"{round_name} round camera slots must be exactly {expected_slots}, got {slots}"
        )
    captured_at = datetime.now().isoformat(timespec="microseconds")
    writes: list[tuple[Path, np.ndarray, str, int]] = []
    rows: list[dict[str, str]] = []
    for result in results:
        slot = result.camera_slot
        view = view_for(round_name, slot)
        handle = handles[slot]
        is_hdr = isinstance(result, HdrViewResult)
        final_kind = "fused" if is_hdr else "single"
        final_path = paths.view_dirs[view] / _image_name(
            args, view, group_id, image_index, final_kind
        )
        short_path = paths.view_dirs[view] / _image_name(args, view, group_id, image_index, "short")
        long_path = paths.view_dirs[view] / _image_name(args, view, group_id, image_index, "long")
        final_image = result.fused_image if is_hdr else result.final_image
        writes.append((final_path, final_image, view, handle.device.index))
        if is_hdr and args.save_hdr_sources:
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
                "capture_mode": "hdr_fused" if is_hdr else "single",
                "exposure": "" if is_hdr else str(args.exposure),
                "gain": str(0.0 if args.gain is None else args.gain),
                "file": str(final_path),
                "source_short": str(short_path) if is_hdr and args.save_hdr_sources else "",
                "source_long": str(long_path) if is_hdr and args.save_hdr_sources else "",
                "short_exposure": str(args.short_exposure) if is_hdr else "",
                "long_exposure": str(args.long_exposure) if is_hdr else "",
                "hdr_attempt": str(result.attempt) if is_hdr else "",
                "fused_clip_pct": str(result.fused_clip_pct) if is_hdr else "",
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
    prompt: Callable[[str], object] | None = None,
) -> bool:
    """Capture a paired front/back sample and record explicit completeness."""
    pacer = GroupedTriggerPacer(args.capture_interval)
    sample_id = f"{args.part_id}_{group_id}_{image_index:06d}"
    rows: list[dict[str, str]] = []
    current_round = "front"
    hand_part = "左手件" if args.hand == "left" else "右手件"
    try:
        for current_round in ("front", "back"):
            if prompt is not None:
                prompt_text = (
                    f"放好 ZS32 {hand_part}正面后按 Enter 或 s..."
                    if current_round == "front"
                    else f"将同一个 ZS32 {hand_part}翻到背面后按 Enter 或 s..."
                )
                prompt(prompt_text)
            results = capture_round(handles, adapter, args, pacer)
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
                "capture_mode": "hdr_fused" if args.hdr else "single",
                "sample_status": "incomplete",
                "failed_round": current_round,
                "failed_view": failed_view,
                "failed_device_index": failed_device,
                "error": str(error),
            }
        )
        _write_manifest(paths, rows)
        return False
    expected_views = set(ROUND_VIEWS["front"] + ROUND_VIEWS["back"])
    stored_views = {row["view"] for row in rows if row["record_type"] == "image"}
    if len(rows) != 6 or stored_views != expected_views:
        error = RuntimeError(
            "sample requires exactly six distinct canonical views before completion; "
            f"got {sorted(stored_views)}"
        )
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
                "capture_mode": "hdr_fused" if args.hdr else "single",
                "sample_status": "incomplete",
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
            "capture_mode": "hdr_fused" if args.hdr else "single",
            "sample_status": "complete",
        }
    )
    _write_manifest(paths, rows)
    return True


def capture_group(
    handles: Sequence[CameraHandle],
    adapter: CameraAdapter,
    args: argparse.Namespace,
    paths: SessionPaths,
    group_id: str,
    prompt: Callable[[str], object] | None = None,
    *,
    pacer: GroupedTriggerPacer | None = None,
) -> list[bool]:
    """Capture every sample in a group front-first, then back-first.

    The placement prompt belongs to the physical group transition, while each
    image index retains its own sample ID across the two capture rounds.
    """
    pacer = GroupedTriggerPacer(args.capture_interval) if pacer is None else pacer
    image_indices = range(1, args.images_per_group + 1)
    rows_by_index: dict[int, list[dict[str, str]]] = {index: [] for index in image_indices}
    failures: dict[int, tuple[str, Exception]] = {}
    hand_part = "左手件" if args.hand == "left" else "右手件"
    for round_name, prompt_text in (
        ("front", f"放好 ZS32 {hand_part}正面后按 Enter 或 s..."),
        ("back", f"将同一个 ZS32 {hand_part}翻到背面后按 Enter 或 s..."),
    ):
        if prompt is not None:
            prompt(prompt_text)
        for image_index in image_indices:
            sample_id = f"{args.part_id}_{group_id}_{image_index:06d}"
            try:
                results = capture_round(handles, adapter, args, pacer)
                rows_by_index[image_index].extend(
                    save_round(
                        results,
                        round_name,
                        sample_id,
                        group_id,
                        image_index,
                        handles,
                        args,
                        paths,
                    )
                )
            except Exception as error:
                failures.setdefault(image_index, (round_name, error))

    statuses: list[bool] = []
    expected_views = set(ROUND_VIEWS["front"] + ROUND_VIEWS["back"])
    for image_index in image_indices:
        sample_id = f"{args.part_id}_{group_id}_{image_index:06d}"
        rows = rows_by_index[image_index]
        failure = failures.get(image_index)
        stored_views = {row["view"] for row in rows if row["record_type"] == "image"}
        if failure is None and len(rows) == 6 and stored_views == expected_views:
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
                    "capture_mode": "hdr_fused" if args.hdr else "single",
                    "sample_status": "complete",
                }
            )
            statuses.append(True)
        else:
            if failure is None:
                round_name = ""
                error = RuntimeError(
                    "sample requires exactly six distinct canonical views before completion; "
                    f"got {sorted(stored_views)}"
                )
                failed_view = failed_device = ""
            else:
                round_name, error = failure
                failed_view, failed_device = _failure_identity(error, round_name, handles)
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
                    "capture_mode": "hdr_fused" if args.hdr else "single",
                    "sample_status": "incomplete",
                    "failed_round": round_name,
                    "failed_view": failed_view,
                    "failed_device_index": failed_device,
                    "error": str(error),
                }
            )
            statuses.append(False)
        _write_manifest(paths, rows)
    return statuses


def select_devices_by_serial(
    serials: CameraSerials, available_devices: Sequence[DeviceDescription]
) -> list[DeviceDescription]:
    """Resolve physical camera slots from one SDK enumeration snapshot.

    Args:
        serials: Requested serials ordered as front, left, and right camera slots.
        available_devices: Devices from one SDK enumeration snapshot.

    Returns:
        Three device descriptions ordered as front, left, and right.

    Raises:
        ValueError: If requested or enumerated identities are ambiguous or missing.
    """
    requested = (serials.front, serials.left, serials.right)
    if len(set(requested)) != len(requested):
        raise ValueError("requested camera serials must be unique")
    counts: dict[str, int] = {}
    for device in available_devices:
        counts[device.serial] = counts.get(device.serial, 0) + 1
    duplicates = sorted(serial for serial, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"enumeration contains duplicate camera serials: {duplicates}")
    by_serial = {device.serial: device for device in available_devices}
    missing = [serial for serial in requested if serial not in by_serial]
    if missing:
        raise ValueError(f"camera serials are unavailable: {missing}")
    return [by_serial[serial] for serial in requested]


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
        """Allow discovery alone while validating capture-specific options."""
        parsed = super().parse_args(args, namespace)
        if not parsed.list_devices:
            if parsed.label is None:
                self.error("the following arguments are required: --label")
            if not parsed.hdr and (parsed.save_hdr_sources or parsed.align_hdr):
                self.error("--save-hdr-sources and --align-hdr require --hdr")
            if parsed.label == "defect" and not parsed.defect_type.strip():
                self.error("the following arguments are required for defect capture: --defect-type")
            positive_integer_options = (
                ("group_count", "--group-count"),
                ("images_per_group", "--images-per-group"),
                ("timeout_ms", "--timeout-ms"),
            )
            for attribute, option in positive_integer_options:
                if getattr(parsed, attribute) <= 0:
                    self.error(f"{option} must be greater than zero")
            positive_float_options = (
                ("exposure", "--exposure"),
                ("short_exposure", "--short-exposure"),
                ("long_exposure", "--long-exposure"),
            )
            for attribute, option in positive_float_options:
                value = getattr(parsed, attribute)
                if not math.isfinite(value) or value <= 0:
                    self.error(f"{option} must be greater than zero")
            capture_interval = parsed.capture_interval
            if not math.isfinite(capture_interval) or capture_interval < 0:
                self.error("--capture-interval must be greater than or equal to zero")
        return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the SDK-free CLI parser for six-view ZS32 capture.

    Returns:
        The configured argument parser.
    """
    parser = _CaptureArgumentParser(description="Collect six-view ZS32 images from three cameras.")
    parser.add_argument("--front-serial", default=DEFAULT_SERIALS.front)
    parser.add_argument("--left-serial", default=DEFAULT_SERIALS.left)
    parser.add_argument("--right-serial", default=DEFAULT_SERIALS.right)
    parser.add_argument("--hand", choices=("left", "right"), default="left")
    parser.add_argument("--label", choices=("normal", "defect"))
    parser.add_argument("--defect-type", default="")
    parser.add_argument("--part-id", default="part001")
    parser.add_argument("--group-count", type=int, default=1)
    parser.add_argument("--images-per-group", type=int, default=1)
    parser.add_argument("--manual-load", action="store_true")

    parser.add_argument("--hdr", action="store_true")
    parser.add_argument("--exposure", type=float, default=4000.0)
    parser.add_argument("--short-exposure", type=float, default=7000.0)
    parser.add_argument("--long-exposure", type=float, default=40000.0)
    parser.add_argument("--gain", type=float)
    parser.add_argument(
        "--capture-interval",
        type=float,
        default=0.2,
        help="minimum seconds between grouped software-trigger passes",
    )
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run device discovery or one interactive six-view capture session."""
    parser = build_parser()
    args = parser.parse_args(argv)
    adapter = HikvisionAdapter.load()
    available_devices = adapter.list_devices()
    if args.list_devices:
        for device in available_devices:
            print(f"{device.index}\t{device.model}\t{device.serial}")
        return 0

    try:
        devices = select_devices_by_serial(
            CameraSerials(args.front_serial, args.left_serial, args.right_serial),
            available_devices,
        )
    except ValueError as error:
        parser.error(str(error))
    paths = create_session(args, datetime.now())
    try:
        with open_cameras(
            devices,
            adapter,
            gain=0.0 if args.gain is None else args.gain,
        ) as handles:
            pacer = GroupedTriggerPacer(args.capture_interval)
            for group_index in range(1, args.group_count + 1):
                group_id = f"group{group_index:03d}"
                capture_group(
                    handles,
                    adapter,
                    args,
                    paths,
                    group_id,
                    prompt=input if args.manual_load else None,
                    pacer=pacer,
                )
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
