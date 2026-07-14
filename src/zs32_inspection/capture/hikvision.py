# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Topology-driven Hikvision MVS capture adapter.

The module is intentionally safe to import on a developer Mac.  The vendor
SDK, OpenCV, and NumPy are imported only when hardware access is requested.
The Linux deployment must expose ``MvCameraControl_class`` through its normal
Python environment (for example, ``PYTHONPATH`` set by the service unit).  This
adapter never mutates ``sys.path`` and never imports a legacy pipeline script.
"""

from __future__ import annotations

import ctypes
import importlib
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any

from .contracts import (
    CameraBinding,
    CaptureFrame,
    CaptureRequest,
    CaptureRoundPlan,
    utc_now,
)
from .service import PartialRoundCaptureError


DEFAULT_FRAME_BUFFER_SIZE = 50 * 1024 * 1024
HIKVISION_CONFIG_FIELDS = (
    "exposure",
    "gain",
    "timeout_ms",
    "capture_interval",
    "hdr",
    "short_exposure",
    "long_exposure",
    "hdr_settle_frames",
    "align_hdr",
    "short_dark_threshold",
    "long_clip_threshold",
    "blend_width",
    "blur_size",
    "hdr_max_retries",
    "hdr_max_clip_pct",
    "png_compression",
    "frame_buffer_size",
)


class HikvisionSdkUnavailableError(RuntimeError):
    """The production camera runtime is missing or cannot be imported."""


class HikvisionCaptureError(RuntimeError):
    """A camera identity, acquisition, conversion, or encoding operation failed."""


@dataclass(frozen=True, slots=True)
class HikvisionCaptureConfig:
    """Versionable acquisition parameters shared by all topology camera slots."""

    exposure: float = 4000.0
    gain: float = 0.0
    timeout_ms: int = 3000
    capture_interval: float = 0.2
    hdr: bool = False
    short_exposure: float = 4000.0
    long_exposure: float = 35000.0
    hdr_settle_frames: int = 8
    align_hdr: bool = True
    short_dark_threshold: float = 80.0
    long_clip_threshold: float = 245.0
    blend_width: float = 50.0
    blur_size: int = 101
    hdr_max_retries: int = 2
    hdr_max_clip_pct: float = 5.0
    png_compression: int = 3
    frame_buffer_size: int = DEFAULT_FRAME_BUFFER_SIZE

    def __post_init__(self) -> None:
        for name in ("hdr", "align_hdr"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        for name in ("exposure", "short_exposure", "long_exposure"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite positive number")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        for name in (
            "gain",
            "capture_interval",
            "short_dark_threshold",
            "long_clip_threshold",
            "blend_width",
            "hdr_max_clip_pct",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.gain < 0:
            raise ValueError("gain must be greater than or equal to zero")
        if self.capture_interval < 0:
            raise ValueError("capture_interval must be greater than or equal to zero")
        if not 0 <= self.short_dark_threshold <= 255:
            raise ValueError("short_dark_threshold must be in [0, 255]")
        if not 0 <= self.long_clip_threshold <= 255:
            raise ValueError("long_clip_threshold must be in [0, 255]")
        if self.blend_width <= 0:
            raise ValueError("blend_width must be greater than zero")
        if not 0 <= self.hdr_max_clip_pct <= 100:
            raise ValueError("hdr_max_clip_pct must be in [0, 100]")
        for name in ("timeout_ms", "frame_buffer_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("hdr_settle_frames", "hdr_max_retries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if isinstance(self.blur_size, bool) or not isinstance(self.blur_size, int):
            raise TypeError("blur_size must be an integer")
        if self.blur_size <= 0:
            raise ValueError("blur_size must be greater than zero")
        if isinstance(self.png_compression, bool) or not isinstance(self.png_compression, int):
            raise TypeError("png_compression must be an integer")
        if not 0 <= self.png_compression <= 9:
            raise ValueError("png_compression must be in [0, 9]")
        if self.hdr and self.short_exposure >= self.long_exposure:
            raise ValueError("HDR short_exposure must be lower than long_exposure")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "HikvisionCaptureConfig":
        """Parse the exact versioned acquisition-asset schema without defaults."""
        if not isinstance(payload, Mapping) or set(payload) != set(HIKVISION_CONFIG_FIELDS):
            raise ValueError(
                "Hikvision acquisition config fields differ from the strict schema"
            )
        return cls(**dict(payload))  # type: ignore[arg-type]

    def as_dict(self) -> dict[str, object]:
        """Return the canonical JSON-native acquisition parameter payload."""
        return {field: getattr(self, field) for field in HIKVISION_CONFIG_FIELDS}


@dataclass(frozen=True, slots=True)
class DeviceDescription:
    """Stable identity reported by one SDK enumeration snapshot."""

    index: int
    model: str
    serial: str


@dataclass(slots=True)
class _CameraHandle:
    device: DeviceDescription
    camera: Any
    frame_info: Any | None = None
    data_buffer: Any | None = None
    created: bool = False
    opened: bool = False
    started: bool = False


@dataclass(frozen=True, slots=True)
class _RuntimeDependencies:
    sdk: Any
    cv2: Any
    numpy: Any


@dataclass(frozen=True, slots=True)
class _CapturedImage:
    binding: CameraBinding
    device_index: int
    image: Any


class _GroupedReadError(HikvisionCaptureError):
    def __init__(self, message: str, partial_images: Sequence[_CapturedImage]) -> None:
        super().__init__(message)
        self.partial_images = tuple(partial_images)


def _load_runtime_dependencies() -> _RuntimeDependencies:
    """Load production-only modules without changing the process import path."""
    modules: dict[str, Any] = {}
    for module_name in ("MvCameraControl_class", "cv2", "numpy"):
        try:
            modules[module_name] = importlib.import_module(module_name)
        except (ImportError, OSError) as error:
            if module_name == "MvCameraControl_class":
                guidance = (
                    "Hikvision MVS SDK module 'MvCameraControl_class' is unavailable. "
                    "Install the vendor SDK on Linux and expose its MvImport directory "
                    "through the deployment environment/PYTHONPATH before starting the service."
                )
            else:
                guidance = (
                    f"production capture dependency {module_name!r} is unavailable in the "
                    "Linux deployment environment"
                )
            raise HikvisionSdkUnavailableError(guidance) from error
    return _RuntimeDependencies(
        sdk=modules["MvCameraControl_class"],
        cv2=modules["cv2"],
        numpy=modules["numpy"],
    )


def _decode_sdk_text(value: Any) -> str:
    return bytes(value).split(b"\0", maxsplit=1)[0].decode("utf-8", errors="replace")


def select_devices_by_serial(
    cameras: Sequence[CameraBinding],
    available_devices: Sequence[DeviceDescription],
) -> tuple[DeviceDescription, ...]:
    """Bind topology slots to one enumeration snapshot, preserving slot order."""
    bindings = tuple(cameras)
    if not bindings:
        raise ValueError("at least one topology camera binding is required")
    requested = tuple(binding.serial for binding in bindings)
    if len(requested) != len(set(requested)):
        raise ValueError("topology camera serials must be unique")
    counts: dict[str, int] = {}
    for device in available_devices:
        counts[device.serial] = counts.get(device.serial, 0) + 1
    duplicates = sorted(serial for serial, count in counts.items() if count > 1)
    if duplicates:
        raise HikvisionCaptureError(
            f"SDK enumeration contains duplicate camera serials: {duplicates}"
        )
    by_serial = {device.serial: device for device in available_devices}
    missing = [serial for serial in requested if serial not in by_serial]
    if missing:
        raise HikvisionCaptureError(f"topology camera serials are unavailable: {missing}")
    return tuple(by_serial[serial] for serial in requested)


class _MvsSdkAdapter:
    """Small ownership boundary around the vendor's C-style Python API."""

    def __init__(self, dependencies: _RuntimeDependencies, *, frame_buffer_size: int) -> None:
        self.dependencies = dependencies
        self.sdk = dependencies.sdk
        self._frame_buffer_size = frame_buffer_size
        self._device_list: Any | None = None

    def _check(
        self,
        return_code: int,
        operation: str,
        device: DeviceDescription | None = None,
    ) -> None:
        if return_code == 0:
            return
        identity = "" if device is None else f" device={device.index} serial={device.serial}"
        raise HikvisionCaptureError(
            f"Hikvision {operation} failed{identity}, ret=0x{return_code:x}"
        )

    def list_devices(self) -> tuple[DeviceDescription, ...]:
        device_list = self.sdk.MV_CC_DEVICE_INFO_LIST()
        device_types = self.sdk.MV_GIGE_DEVICE | self.sdk.MV_USB_DEVICE
        self._check(
            self.sdk.MvCamera.MV_CC_EnumDevices(device_types, device_list),
            "EnumDevices",
        )
        self._device_list = device_list
        devices: list[DeviceDescription] = []
        for index in range(device_list.nDeviceNum):
            info = ctypes.cast(
                device_list.pDeviceInfo[index],
                ctypes.POINTER(self.sdk.MV_CC_DEVICE_INFO),
            ).contents
            if info.nTLayerType == self.sdk.MV_GIGE_DEVICE:
                transport = info.SpecialInfo.stGigEInfo
            elif info.nTLayerType == self.sdk.MV_USB_DEVICE:
                transport = info.SpecialInfo.stUsb3VInfo
            else:
                raise HikvisionCaptureError(
                    f"enumerated unsupported camera transport at device index {index}"
                )
            devices.append(
                DeviceDescription(
                    index=index,
                    model=_decode_sdk_text(transport.chModelName),
                    serial=_decode_sdk_text(transport.chSerialNumber),
                )
            )
        return tuple(devices)

    def open(self, device: DeviceDescription, gain: float) -> _CameraHandle:
        if self._device_list is None:
            raise HikvisionCaptureError("list_devices() must run before opening a camera")
        if device.index < 0 or device.index >= self._device_list.nDeviceNum:
            raise HikvisionCaptureError(
                f"enumerated camera index is unavailable: {device.index} serial={device.serial}"
            )
        info = ctypes.cast(
            self._device_list.pDeviceInfo[device.index],
            ctypes.POINTER(self.sdk.MV_CC_DEVICE_INFO),
        ).contents
        transport = (
            info.SpecialInfo.stGigEInfo
            if info.nTLayerType == self.sdk.MV_GIGE_DEVICE
            else info.SpecialInfo.stUsb3VInfo
        )
        enumerated_serial = _decode_sdk_text(transport.chSerialNumber)
        if enumerated_serial != device.serial:
            raise HikvisionCaptureError(
                "camera enumeration changed before open: "
                f"index={device.index} expected={device.serial} actual={enumerated_serial}"
            )
        camera = self.sdk.MvCamera()
        handle = _CameraHandle(device=device, camera=camera)
        try:
            self._check(camera.MV_CC_CreateHandle(info), "CreateHandle", device)
            handle.created = True
            self._check(
                camera.MV_CC_OpenDevice(self.sdk.MV_ACCESS_Exclusive, 0),
                "OpenDevice",
                device,
            )
            handle.opened = True
            self._check(camera.MV_CC_SetEnumValue("ExposureAuto", 0), "ExposureAuto", device)
            self._check(camera.MV_CC_SetEnumValue("GainAuto", 0), "GainAuto", device)
            self._set_checked_float(handle, "Gain", gain)
            self._check(camera.MV_CC_SetEnumValue("TriggerMode", 1), "TriggerMode", device)
            self._check(camera.MV_CC_SetEnumValue("TriggerSource", 7), "TriggerSource", device)
            handle.frame_info = self.sdk.MV_FRAME_OUT_INFO_EX()
            handle.data_buffer = (ctypes.c_ubyte * self._frame_buffer_size)()
            return handle
        except BaseException as error:  # noqa: BLE001 - hardware resources need signal-safe cleanup
            _attach_cleanup_errors(error, self.release((handle,)))
            raise

    def _set_checked_float(self, handle: _CameraHandle, name: str, value: float) -> None:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        float_range = self.sdk.MVCC_FLOATVALUE()
        self._check(
            handle.camera.MV_CC_GetFloatValue(name, float_range),
            f"GetFloatValue({name})",
            handle.device,
        )
        minimum = float(float_range.fMin)
        maximum = float(float_range.fMax)
        if value < minimum or value > maximum:
            raise HikvisionCaptureError(
                f"{name}={value} is outside [{minimum}, {maximum}] "
                f"for serial={handle.device.serial}"
            )
        self._check(
            handle.camera.MV_CC_SetFloatValue(name, float(value)),
            name,
            handle.device,
        )

    def start(self, handle: _CameraHandle) -> None:
        self._check(handle.camera.MV_CC_StartGrabbing(), "StartGrabbing", handle.device)
        handle.started = True

    def set_exposure(self, handle: _CameraHandle, exposure: float) -> None:
        self._set_checked_float(handle, "ExposureTime", exposure)

    def trigger(self, handle: _CameraHandle) -> None:
        self._check(
            handle.camera.MV_CC_SetCommandValue("TriggerSoftware"),
            "TriggerSoftware",
            handle.device,
        )

    def read(self, handle: _CameraHandle, timeout_ms: int) -> Any:
        if handle.data_buffer is None or handle.frame_info is None:
            raise HikvisionCaptureError(
                f"camera read buffers are not initialized for serial={handle.device.serial}"
            )
        self._check(
            handle.camera.MV_CC_GetOneFrameTimeout(
                handle.data_buffer,
                len(handle.data_buffer),
                handle.frame_info,
                timeout_ms,
            ),
            "GetOneFrameTimeout",
            handle.device,
        )
        return self._convert_frame(handle.data_buffer, handle.frame_info)

    def _convert_frame(self, data_buffer: Any, frame_info: Any) -> Any:
        cv2 = self.dependencies.cv2
        np = self.dependencies.numpy
        width = int(frame_info.nWidth)
        height = int(frame_info.nHeight)
        frame_length = int(frame_info.nFrameLen)
        if width <= 0 or height <= 0 or frame_length <= 0:
            raise HikvisionCaptureError(
                f"SDK returned invalid frame dimensions/length: {width}x{height}, {frame_length}"
            )
        data = np.frombuffer(data_buffer, dtype=np.uint8, count=frame_length)
        pixel_type = frame_info.enPixelType
        if pixel_type == self.sdk.PixelType_Gvsp_Mono8:
            return cv2.cvtColor(data.reshape(height, width), cv2.COLOR_GRAY2BGR)
        if pixel_type == self.sdk.PixelType_Gvsp_BayerRG8:
            return cv2.cvtColor(data.reshape(height, width), cv2.COLOR_BAYER_RG2BGR)
        if pixel_type == self.sdk.PixelType_Gvsp_BGR8_Packed:
            return data.reshape(height, width, 3).copy()
        raise HikvisionCaptureError(f"unsupported Hikvision pixel type: {pixel_type!r}")

    def release(
        self,
        handles: Sequence[_CameraHandle],
    ) -> tuple[tuple[str, DeviceDescription, BaseException], ...]:
        """Attempt every cleanup operation in reverse acquisition order."""
        errors: list[tuple[str, DeviceDescription, BaseException]] = []
        for handle in reversed(tuple(handles)):
            operations: list[tuple[str, Callable[[], Any]]] = []
            if handle.started:
                operations.append(("stop", handle.camera.MV_CC_StopGrabbing))
            if handle.opened:
                operations.extend(
                    (
                        ("restore_trigger_mode", lambda h=handle: h.camera.MV_CC_SetEnumValue("TriggerMode", 0)),
                        ("close", handle.camera.MV_CC_CloseDevice),
                    )
                )
            if handle.created:
                operations.append(("destroy", handle.camera.MV_CC_DestroyHandle))
            for operation, callback in operations:
                try:
                    self._check(callback(), operation, handle.device)
                except BaseException as error:  # noqa: BLE001 - attempt all cleanup operations
                    errors.append((operation, handle.device, error))
            handle.started = False
            handle.opened = False
            handle.created = False
        return tuple(errors)


class _GroupedTriggerPacer:
    def __init__(
        self,
        interval_seconds: float,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
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


def _format_cleanup_errors(
    errors: Sequence[tuple[str, DeviceDescription, BaseException]],
) -> str:
    details = "; ".join(
        f"{operation} device={device.index} serial={device.serial}: {error}"
        for operation, device, error in errors
    )
    return f"Hikvision camera cleanup failed: {details}"


def _attach_cleanup_errors(
    primary_error: BaseException,
    errors: Sequence[tuple[str, DeviceDescription, BaseException]],
) -> None:
    if not errors:
        return
    message = _format_cleanup_errors(errors)
    if hasattr(primary_error, "add_note"):
        try:
            primary_error.add_note(message)
        except (AttributeError, TypeError):
            # Frozen dataclass exceptions can reject BaseException.add_note's
            # attribute write.  Keep cleanup diagnostics without replacing the
            # primary hardware/capture failure.
            notes = list(getattr(primary_error, "__notes__", ()))
            object.__setattr__(primary_error, "__notes__", [*notes, message])


class HikvisionCameraAdapter:
    """CaptureSource implementation with topology serial binding and grouped trigger/read.

    Use it as a context manager for one capture session.  Successful rounds keep
    the exclusive camera handles open; any capture failure immediately releases
    all handles, and normal context exit also performs complete cleanup.
    """

    def __init__(
        self,
        config: HikvisionCaptureConfig,
        *,
        dependency_loader: Callable[[], _RuntimeDependencies] = _load_runtime_dependencies,
    ) -> None:
        if not isinstance(config, HikvisionCaptureConfig):
            raise TypeError("config must be HikvisionCaptureConfig")
        self.config = config
        self._dependency_loader = dependency_loader
        self._dependencies: _RuntimeDependencies | None = None
        self._sdk_adapter: _MvsSdkAdapter | None = None
        self._handles: tuple[_CameraHandle, ...] = ()
        self._bindings: tuple[CameraBinding, ...] = ()
        self._pacer = _GroupedTriggerPacer(config.capture_interval)

    def __enter__(self) -> HikvisionCameraAdapter:
        """Return the session-scoped source without opening hardware yet."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Release every camera and never suppress the active application error."""
        del exc_type, traceback
        try:
            self.close()
        except BaseException as cleanup_error:  # noqa: BLE001 - preserve primary exception
            if exc_value is None:
                raise
            _attach_cleanup_errors(
                exc_value,
                (("context_close", DeviceDescription(-1, "session", "unknown"), cleanup_error),),
            )
        return False

    def _runtime(self) -> _MvsSdkAdapter:
        if self._sdk_adapter is None:
            self._dependencies = self._dependency_loader()
            self._sdk_adapter = _MvsSdkAdapter(
                self._dependencies,
                frame_buffer_size=self.config.frame_buffer_size,
            )
        return self._sdk_adapter

    def list_devices(self) -> tuple[DeviceDescription, ...]:
        """Enumerate devices without opening them; still requires the Linux SDK."""
        return self._runtime().list_devices()

    def _ensure_open(self, cameras: Sequence[CameraBinding]) -> None:
        requested = tuple(cameras)
        if self._handles:
            if requested != self._bindings:
                raise HikvisionCaptureError(
                    "camera topology bindings changed during one open capture session"
                )
            return
        sdk_adapter = self._runtime()
        available = sdk_adapter.list_devices()
        selected = select_devices_by_serial(requested, available)
        handles: list[_CameraHandle] = []
        try:
            for device in selected:
                handle = sdk_adapter.open(device, self.config.gain)
                handles.append(handle)
                sdk_adapter.start(handle)
        except BaseException as error:  # noqa: BLE001 - clean partial hardware setup
            _attach_cleanup_errors(error, sdk_adapter.release(handles))
            raise
        self._handles = tuple(handles)
        self._bindings = requested

    def close(self) -> None:
        """Release all cameras, attempting every cleanup operation before failing."""
        if not self._handles:
            self._bindings = ()
            return
        sdk_adapter = self._runtime()
        handles = self._handles
        self._handles = ()
        self._bindings = ()
        errors = sdk_adapter.release(handles)
        if errors:
            raise HikvisionCaptureError(_format_cleanup_errors(errors)) from errors[0][2]

    def _close_after_failure(self, error: BaseException) -> None:
        if not self._handles or self._sdk_adapter is None:
            return
        handles = self._handles
        self._handles = ()
        self._bindings = ()
        _attach_cleanup_errors(error, self._sdk_adapter.release(handles))

    def _group_trigger_and_read(self) -> tuple[_CapturedImage, ...]:
        adapter = self._runtime()
        for handle in self._handles:
            adapter.trigger(handle)
        images: list[_CapturedImage] = []
        for binding, handle in zip(self._bindings, self._handles, strict=True):
            try:
                image = adapter.read(handle, self.config.timeout_ms)
            except BaseException as error:  # noqa: BLE001 - retain partial grouped-read evidence
                raise _GroupedReadError(
                    f"grouped read failed for slot={binding.slot_id} "
                    f"serial={binding.serial}: {error}",
                    images,
                ) from error
            images.append(_CapturedImage(binding, handle.device.index, image))
        return tuple(images)

    def _capture_exposure_pass(self, exposure: float) -> tuple[_CapturedImage, ...]:
        adapter = self._runtime()
        for handle in self._handles:
            adapter.set_exposure(handle, exposure)
        for _ in range(self.config.hdr_settle_frames):
            self._pacer.wait()
            self._group_trigger_and_read()
        self._pacer.wait()
        return self._group_trigger_and_read()

    def _capture_single(self) -> tuple[_CapturedImage, ...]:
        adapter = self._runtime()
        for handle in self._handles:
            adapter.set_exposure(handle, self.config.exposure)
        self._pacer.wait()
        return self._group_trigger_and_read()

    def _capture_hdr(self) -> tuple[tuple[_CapturedImage, ...], int, tuple[float, ...]]:
        final_images: tuple[_CapturedImage, ...] = ()
        final_clips: tuple[float, ...] = ()
        for attempt in range(1, self.config.hdr_max_retries + 2):
            short_images = self._capture_exposure_pass(self.config.short_exposure)
            long_images = self._capture_exposure_pass(self.config.long_exposure)
            fused: list[_CapturedImage] = []
            clips: list[float] = []
            for short, long in zip(short_images, long_images, strict=True):
                if short.binding != long.binding or short.device_index != long.device_index:
                    raise HikvisionCaptureError("HDR exposure passes changed camera identity/order")
                image = self._fuse_exposures(short.image, long.image)
                fused.append(_CapturedImage(short.binding, short.device_index, image))
                clips.append(self._image_clip_pct(image))
            final_images = tuple(fused)
            final_clips = tuple(clips)
            if all(value <= self.config.hdr_max_clip_pct for value in final_clips):
                return final_images, attempt, final_clips
        return final_images, self.config.hdr_max_retries + 1, final_clips

    def _fuse_exposures(self, short_image: Any, long_image: Any) -> Any:
        if self._dependencies is None:
            raise HikvisionCaptureError("capture runtime dependencies are not loaded")
        cv2 = self._dependencies.cv2
        np = self._dependencies.numpy
        if short_image.shape != long_image.shape:
            raise HikvisionCaptureError(
                f"HDR image shapes differ: {short_image.shape!r} vs {long_image.shape!r}"
            )
        if short_image.dtype != np.uint8 or long_image.dtype != np.uint8:
            raise HikvisionCaptureError("HDR source images must be uint8")
        working = [short_image, long_image]
        if self.config.align_hdr:
            aligned = [image.copy() for image in working]
            cv2.createAlignMTB().process(working, aligned)
            working = aligned
        short_gray = cv2.cvtColor(working[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
        long_gray = cv2.cvtColor(working[1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        width = max(float(self.config.blend_width), 1.0)
        dark_weight = 1.0 / (
            1.0 + np.exp((short_gray - self.config.short_dark_threshold) / width)
        )
        valid_long_weight = 1.0 / (
            1.0 + np.exp((long_gray - self.config.long_clip_threshold) / width)
        )
        weight = dark_weight * valid_long_weight
        blur_size = self.config.blur_size
        if blur_size > 1:
            if blur_size % 2 == 0:
                blur_size += 1
            weight = cv2.GaussianBlur(weight, (blur_size, blur_size), 0)
        weight_3c = weight[..., None]
        fused = (
            working[0].astype(np.float32) * (1.0 - weight_3c)
            + working[1].astype(np.float32) * weight_3c
        )
        return np.clip(fused, 0, 255).astype(np.uint8)

    def _image_clip_pct(self, image: Any) -> float:
        if self._dependencies is None:
            raise HikvisionCaptureError("capture runtime dependencies are not loaded")
        cv2 = self._dependencies.cv2
        np = self._dependencies.numpy
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray >= 250) * 100.0)

    def _encode_frame(
        self,
        captured: _CapturedImage,
        round_plan: CaptureRoundPlan,
        *,
        capture_mode: str,
        exposure: float | None,
        parameters: dict[str, str | int | float | bool | None],
    ) -> CaptureFrame:
        if self._dependencies is None:
            raise HikvisionCaptureError("capture runtime dependencies are not loaded")
        image = captured.image
        if getattr(image, "ndim", None) != 3 or image.shape[2] != 3:
            raise HikvisionCaptureError(
                f"canonical camera image must be HxWx3 BGR, got {getattr(image, 'shape', None)!r}"
            )
        cv2 = self._dependencies.cv2
        success, encoded = cv2.imencode(
            ".png",
            image,
            [cv2.IMWRITE_PNG_COMPRESSION, self.config.png_compression],
        )
        if not success:
            raise HikvisionCaptureError(
                f"PNG encoding failed for camera slot={captured.binding.slot_id}"
            )
        return CaptureFrame(
            round_id=round_plan.round_id,
            view_id=captured.binding.views[round_plan.round_id],
            camera_slot_id=captured.binding.slot_id,
            camera_serial=captured.binding.serial,
            device_index=captured.device_index,
            image_bytes=encoded.tobytes(),
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            capture_mode=capture_mode,
            exposure=exposure,
            gain=self.config.gain,
            captured_at=utc_now(),
            capture_parameters=parameters,
        )

    def _common_parameters(self) -> dict[str, str | int | float | bool | None]:
        """Return operational settings needed to reproduce one acquisition pass."""
        return {
            "timeout_ms": self.config.timeout_ms,
            "capture_interval": self.config.capture_interval,
            "png_compression": self.config.png_compression,
            "frame_buffer_size": self.config.frame_buffer_size,
        }

    def capture_round(
        self,
        request: CaptureRequest,
        round_plan: CaptureRoundPlan,
        cameras: Sequence[CameraBinding],
    ) -> Sequence[CaptureFrame]:
        """Capture one topology round; every exposure pass triggers all cameras first."""
        if not isinstance(request, CaptureRequest):
            raise TypeError("request must be CaptureRequest")
        if not isinstance(round_plan, CaptureRoundPlan):
            raise TypeError("round_plan must be CaptureRoundPlan")
        bindings = tuple(cameras)
        if not bindings:
            raise ValueError("capture round requires topology camera bindings")
        for binding in bindings:
            if round_plan.round_id not in binding.views:
                raise ValueError(
                    f"camera slot {binding.slot_id!r} has no view for round "
                    f"{round_plan.round_id!r}"
                )
        encoded: list[CaptureFrame] = []
        try:
            self._ensure_open(bindings)
            if self.config.hdr:
                images, attempt, clip_percentages = self._capture_hdr()
                for captured, clip_pct in zip(images, clip_percentages, strict=True):
                    frame = self._encode_frame(
                        captured,
                        round_plan,
                        capture_mode="hdr_fused",
                        exposure=None,
                        parameters={
                            **self._common_parameters(),
                            "short_exposure": self.config.short_exposure,
                            "long_exposure": self.config.long_exposure,
                            "hdr_settle_frames": self.config.hdr_settle_frames,
                            "hdr_attempt": attempt,
                            "align_hdr": self.config.align_hdr,
                            "short_dark_threshold": self.config.short_dark_threshold,
                            "long_clip_threshold": self.config.long_clip_threshold,
                            "blend_width": self.config.blend_width,
                            "blur_size": self.config.blur_size,
                            "hdr_max_retries": self.config.hdr_max_retries,
                            "hdr_max_clip_pct": self.config.hdr_max_clip_pct,
                            "fused_clip_pct": clip_pct,
                            "hdr_clip_limit_exceeded": (
                                clip_pct > self.config.hdr_max_clip_pct
                            ),
                        },
                    )
                    encoded.append(frame)
            else:
                try:
                    images = self._capture_single()
                except _GroupedReadError as error:
                    for captured in error.partial_images:
                        encoded.append(
                            self._encode_frame(
                                captured,
                                round_plan,
                                capture_mode="single",
                                exposure=self.config.exposure,
                                parameters={
                                    **self._common_parameters(),
                                    "exposure": self.config.exposure,
                                },
                            )
                        )
                    raise PartialRoundCaptureError(str(error), tuple(encoded)) from error
                for captured in images:
                    encoded.append(
                        self._encode_frame(
                            captured,
                            round_plan,
                            capture_mode="single",
                            exposure=self.config.exposure,
                            parameters={
                                **self._common_parameters(),
                                "exposure": self.config.exposure,
                            },
                        )
                    )
            return tuple(encoded)
        except BaseException as error:  # noqa: BLE001 - always release exclusive camera handles
            failure: BaseException = error
            if not isinstance(error, PartialRoundCaptureError) and encoded:
                failure = PartialRoundCaptureError(str(error), tuple(encoded))
            self._close_after_failure(failure)
            if failure is error:
                raise
            raise failure from error


__all__ = [
    "DeviceDescription",
    "HikvisionCameraAdapter",
    "HikvisionCaptureConfig",
    "HikvisionCaptureError",
    "HikvisionSdkUnavailableError",
    "select_devices_by_serial",
]
