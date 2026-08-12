"""Minimal four-camera/two-round HDR acquisition for the BMW eight-view Demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from bmw_inspection.capture.config import BmwCaptureProfile, load_capture_profile


@dataclass(frozen=True, slots=True)
class _HdrRuntimeConfig:
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


@dataclass(frozen=True, slots=True)
class HdrSourceImages:
    """Owned source frames retained for one semantic view of an HDR capture."""

    short_image: np.ndarray
    long_image: np.ndarray
    fused_image: np.ndarray
    fused_clip_pct: float
    attempt: int
    source_kind: str = "hdr_pair"

    def __post_init__(self) -> None:
        if self.source_kind not in {"hdr_pair", "fused_only"}:
            raise ValueError("source_kind must be hdr_pair or fused_only")
        if not isinstance(self.attempt, int) or self.attempt <= 0:
            raise ValueError("attempt must be a positive integer")
        for name in ("short_image", "long_image", "fused_image"):
            image = getattr(self, name)
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim not in {2, 3} or image.size == 0:
                raise ValueError(f"{name} must be a non-empty uint8 grayscale/BGR image")
            owned = image.copy()
            owned.flags.writeable = False
            object.__setattr__(self, name, owned)


def _runtime_config(profile: BmwCaptureProfile) -> _HdrRuntimeConfig:
    hdr = profile.hdr
    return _HdrRuntimeConfig(
        short_exposure=hdr.short_exposure_us,
        long_exposure=hdr.long_exposure_us,
        hdr_settle_frames=hdr.settle_frames,
        timeout_ms=hdr.timeout_ms,
        align_hdr=hdr.align,
        short_dark_threshold=hdr.short_dark_threshold,
        long_clip_threshold=hdr.long_clip_threshold,
        blend_width=hdr.blend_width,
        blur_size=hdr.blur_size,
        hdr_max_retries=hdr.max_retries,
        hdr_max_clip_pct=hdr.max_clip_pct,
        capture_interval=hdr.trigger_interval_s,
    )


class FourCameraHdrSession:
    """Open the four configured serials once and capture front/back HDR rounds."""

    def __init__(self, capture_config: Path) -> None:
        self.profile = load_capture_profile(Path(capture_config))
        self._adapter: Any = None
        self._camera_context: Any = None
        self._handles: Any = None
        self._pacer: Any = None
        self.last_sources: Mapping[str, HdrSourceImages] = MappingProxyType({})

    def __enter__(self) -> FourCameraHdrSession:
        from capture_data.collect_multicamera_dataset import GroupedTriggerPacer, HikvisionAdapter, open_cameras

        adapter = HikvisionAdapter.load()
        available = {device.serial: device for device in adapter.list_devices()}
        missing = [slot.serial for slot in self.profile.slots if slot.serial not in available]
        if missing:
            raise RuntimeError(f"未找到BMW相机：{', '.join(missing)}")
        devices = [available[slot.serial] for slot in self.profile.slots]
        context = open_cameras(devices, adapter, self.profile.hdr.gain)
        handles = context.__enter__()
        self._adapter = adapter
        self._camera_context = context
        self._handles = handles
        self._pacer = GroupedTriggerPacer(self.profile.hdr.trigger_interval_s)
        return self

    def capture_round(self, round_id: str) -> Mapping[str, np.ndarray]:
        """Capture one HDR round and map physical slots to four semantic views."""
        if round_id not in {"front", "back"}:
            raise ValueError("round_id必须是front或back")
        if self._handles is None:
            raise RuntimeError("四相机尚未打开")
        from capture_data.collect_multicamera_dataset import capture_hdr_round

        results = capture_hdr_round(
            self._handles,
            self._adapter,
            _runtime_config(self.profile),
            pacer=self._pacer,
        )
        if len(results) != 4:
            raise RuntimeError(f"HDR采集应返回4张融合图，实际为{len(results)}")
        images: dict[str, np.ndarray] = {}
        sources: dict[str, HdrSourceImages] = {}
        for result in results:
            if result.camera_slot < 0 or result.camera_slot >= len(self.profile.slots):
                raise RuntimeError(f"HDR返回未知相机槽位：{result.camera_slot}")
            slot = self.profile.slots[result.camera_slot]
            view = slot.front_view if round_id == "front" else slot.back_view
            images[view] = result.fused_image.copy()
            sources[view] = HdrSourceImages(
                short_image=result.short_image,
                long_image=result.long_image,
                fused_image=result.fused_image,
                fused_clip_pct=result.fused_clip_pct,
                attempt=result.attempt,
            )
        expected = self.profile.front_views if round_id == "front" else self.profile.back_views
        if tuple(images) != expected:
            raise RuntimeError(f"HDR视角顺序不完整：{tuple(images)}")
        cached = dict(self.last_sources)
        cached.update(sources)
        self.last_sources = MappingProxyType(cached)
        return MappingProxyType(images)

    def close(self) -> None:
        if self._camera_context is not None:
            context = self._camera_context
            self._camera_context = None
            self._handles = None
            context.__exit__(None, None, None)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if self._camera_context is None:
            return False
        context = self._camera_context
        self._camera_context = None
        self._handles = None
        return bool(context.__exit__(exc_type, exc, traceback))


__all__ = ["FourCameraHdrSession", "HdrSourceImages"]
