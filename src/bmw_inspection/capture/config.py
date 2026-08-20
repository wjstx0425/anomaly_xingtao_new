"""Strict configuration for the minimal BMW four-camera HDR collector."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_EXPECTED_VIEWS = (
    "front",
    "front_left",
    "front_right",
    "front_secondary",
    "back",
    "back_left",
    "back_right",
    "back_secondary",
)


@dataclass(frozen=True, slots=True)
class CameraSlot:
    """One fixed camera and its view in both operator rounds."""

    slot_id: str
    serial: str
    front_view: str
    back_view: str


@dataclass(frozen=True, slots=True)
class HdrSettings:
    """Image-formation settings forwarded to the existing HDR collector."""

    short_exposure_us: float
    long_exposure_us: float
    gain: float
    trigger_interval_s: float
    settle_frames: int
    timeout_ms: int
    align: bool
    short_dark_threshold: float
    long_clip_threshold: float
    blend_width: float
    blur_size: int
    max_retries: int
    max_clip_pct: float


@dataclass(frozen=True, slots=True)
class BmwCaptureProfile:
    """Validated BMW acquisition profile and resolved bootstrap topology."""

    path: Path
    profile_id: str
    bootstrap_topology_path: Path
    slots: tuple[CameraSlot, ...]
    hdr: HdrSettings

    @property
    def front_views(self) -> tuple[str, ...]:
        return tuple(slot.front_view for slot in self.slots)

    @property
    def back_views(self) -> tuple[str, ...]:
        return tuple(slot.back_view for slot in self.slots)

    @property
    def all_views(self) -> tuple[str, ...]:
        return self.front_views + self.back_views


def _number(payload: dict[str, Any], field: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return float(value)


def _positive(payload: dict[str, Any], field: str) -> float:
    value = _number(payload, field)
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def load_capture_profile(path: Path) -> BmwCaptureProfile:
    """Load the exact approved four-camera/eight-view HDR capture contract."""
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    profile_id = payload.get("profile_id")
    if not isinstance(profile_id, str) or _SAFE_ID.fullmatch(profile_id) is None:
        raise ValueError("profile_id must be path-safe")

    raw_slots = payload.get("slots")
    if not isinstance(raw_slots, list) or len(raw_slots) != 4:
        raise ValueError("slots must contain exactly four cameras")
    slots = tuple(
        CameraSlot(
            slot_id=str(item["slot_id"]),
            serial=str(item["serial"]),
            front_view=str(item["front_view"]),
            back_view=str(item["back_view"]),
        )
        for item in raw_slots
    )
    if len({slot.serial for slot in slots}) != 4:
        raise ValueError("camera serials must be unique")
    views = tuple(slot.front_view for slot in slots) + tuple(slot.back_view for slot in slots)
    if views != _EXPECTED_VIEWS:
        raise ValueError(f"views must equal {_EXPECTED_VIEWS}")

    raw_hdr = payload.get("hdr")
    if not isinstance(raw_hdr, dict):
        raise ValueError("hdr must be an object")
    short = _positive(raw_hdr, "short_exposure_us")
    long = _positive(raw_hdr, "long_exposure_us")
    if short >= long:
        raise ValueError("short_exposure_us must be below long_exposure_us")
    settle = raw_hdr.get("settle_frames")
    timeout = raw_hdr.get("timeout_ms")
    blur = raw_hdr.get("blur_size")
    retries = raw_hdr.get("max_retries")
    if isinstance(settle, bool) or not isinstance(settle, int) or settle <= 0:
        raise ValueError("settle_frames must be positive")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise ValueError("timeout_ms must be positive")
    if isinstance(blur, bool) or not isinstance(blur, int) or blur <= 0 or blur % 2 == 0:
        raise ValueError("blur_size must be a positive odd integer")
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError("max_retries must be non-negative")
    align = raw_hdr.get("align")
    if not isinstance(align, bool):
        raise ValueError("align must be boolean")
    hdr = HdrSettings(
        short_exposure_us=short,
        long_exposure_us=long,
        gain=_number(raw_hdr, "gain"),
        trigger_interval_s=_positive(raw_hdr, "trigger_interval_s"),
        settle_frames=settle,
        timeout_ms=timeout,
        align=align,
        short_dark_threshold=_positive(raw_hdr, "short_dark_threshold"),
        long_clip_threshold=_positive(raw_hdr, "long_clip_threshold"),
        blend_width=_positive(raw_hdr, "blend_width"),
        blur_size=blur,
        max_retries=retries,
        max_clip_pct=_positive(raw_hdr, "max_clip_pct"),
    )
    topology_text = payload.get("bootstrap_topology_path")
    if not isinstance(topology_text, str) or not topology_text:
        raise ValueError("bootstrap_topology_path must be non-empty")
    topology_path = (resolved.parent / topology_text).resolve()
    if not topology_path.is_file():
        raise ValueError(f"bootstrap topology does not exist: {topology_path}")
    return BmwCaptureProfile(resolved, profile_id, topology_path, slots, hdr)
