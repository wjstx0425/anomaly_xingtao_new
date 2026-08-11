"""Immutable contracts for the BMW bright-streak Demo."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any, NoReturn


class DemoStatus(str, Enum):
    """Stable customer-facing outcomes."""

    OK = "OK"
    NG_NO_STREAK = "NG_NO_STREAK"
    NG_BROKEN = "NG_BROKEN"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class BrightStreakConfig:
    """Validated camera, ROI, and rule settings for one Demo configuration."""

    path: Path
    camera_serial: str
    image_width: int
    image_height: int
    exposure: float
    gain: float
    timeout_ms: int
    warmup_frames: int
    roi_xyxy: tuple[int, int, int, int] | None
    result_root: Path
    min_mean_intensity: float
    max_mean_intensity: float
    max_dark_clip_ratio: float
    max_bright_clip_ratio: float
    min_laplacian_variance: float
    background_kernel_px: int
    response_mad_scale: float
    min_component_area_px: int
    min_component_width_px: float
    max_component_width_px: float
    center_tolerance_px: float
    micro_gap_close_px: int
    min_contrast_snr: float
    min_coverage_ratio: float
    min_longest_run_ratio: float
    max_gap_ratio: float
    max_gap_count: int

    def __post_init__(self) -> None:
        """Defend direct construction as strictly as JSON loading."""
        if not isinstance(self.path, Path):
            raise TypeError("path must be a Path")
        if not isinstance(self.camera_serial, str) or not self.camera_serial.strip():
            raise ValueError("camera_serial must be a non-empty string")
        _positive_integer(self.image_width, "image_width")
        _positive_integer(self.image_height, "image_height")
        _finite_number(self.exposure, "exposure", minimum=0.0, minimum_inclusive=False)
        _finite_number(self.gain, "gain", minimum=0.0)
        _positive_integer(self.timeout_ms, "timeout_ms")
        if self.warmup_frames not in {1, 2}:
            raise ValueError("warmup_frames must be 1 or 2")
        if not isinstance(self.result_root, Path) or not str(self.result_root).strip():
            raise ValueError("result_root must be a non-empty path")
        _validate_roi(self.roi_xyxy, self.image_width, self.image_height)

        for name in _FINITE_THRESHOLD_NAMES:
            value = getattr(self, name)
            bounds = _THRESHOLD_BOUNDS.get(name, (0.0, None))
            _finite_number(value, name, minimum=bounds[0], maximum=bounds[1])
        for name in _INTEGER_THRESHOLD_NAMES:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.background_kernel_px <= 0 or self.background_kernel_px % 2 == 0:
            raise ValueError("background_kernel_px must be a positive odd integer")
        if self.min_mean_intensity >= self.max_mean_intensity:
            raise ValueError("min_mean_intensity must be less than max_mean_intensity")
        if self.min_component_width_px > self.max_component_width_px:
            raise ValueError("min_component_width_px must not exceed max_component_width_px")

    def require_detection_roi(self) -> tuple[int, int, int, int]:
        """Return the selected ROI or fail closed before detection."""
        if self.roi_xyxy is None:
            raise ValueError("bright-streak ROI has not been selected")
        return self.roi_xyxy

    @property
    def detection_roi(self) -> tuple[int, int, int, int]:
        """Half-open ROI required by the detector."""
        return self.require_detection_roi()

    @property
    def allowed_max_gap_px(self) -> int:
        """Convert the editable ratio limit using the ROI streak-axis length."""
        _x1, y1, _x2, y2 = self.require_detection_roi()
        return max(0, int((y2 - y1) * self.max_gap_ratio))


@dataclass(frozen=True, slots=True)
class BrightStreakMetrics:
    """Numerical detector and image-quality evidence for one inspection."""

    contrast_snr: float
    coverage_ratio: float
    longest_run_ratio: float
    max_gap_px: int
    max_gap_ratio: float
    gap_count: int
    mean_width_px: float
    lateral_offset_px: float
    mean_intensity: float
    dark_clip_ratio: float
    bright_clip_ratio: float
    laplacian_variance: float
    capture_elapsed_ms: float = 0.0
    processing_elapsed_ms: float = 0.0
    total_elapsed_ms: float = 0.0

    def __post_init__(self) -> None:
        """Reject incomplete or non-finite published measurements."""
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name in {"max_gap_px", "gap_count"}:
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"{item.name} must be a non-negative integer")
            elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{item.name} must be finite")
        for name in ("coverage_ratio", "longest_run_ratio", "max_gap_ratio", "dark_clip_ratio", "bright_clip_ratio"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class BrightStreakResult:
    """One fail-closed BMW bright-streak decision."""

    status: DemoStatus
    reason: str
    roi_xyxy: tuple[int, int, int, int]
    metrics: BrightStreakMetrics | None

    def __post_init__(self) -> None:
        """Require explicit status, reason, and half-open ROI evidence."""
        if not isinstance(self.status, DemoStatus):
            raise TypeError("status must be DemoStatus")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")
        if (
            not isinstance(self.roi_xyxy, tuple)
            or len(self.roi_xyxy) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) for value in self.roi_xyxy)
        ):
            raise ValueError("roi_xyxy must be four integer half-open coordinates")
        if self.metrics is not None and not isinstance(self.metrics, BrightStreakMetrics):
            raise TypeError("metrics must be BrightStreakMetrics or None")


_FINITE_THRESHOLD_NAMES = (
    "min_mean_intensity",
    "max_mean_intensity",
    "max_dark_clip_ratio",
    "max_bright_clip_ratio",
    "min_laplacian_variance",
    "response_mad_scale",
    "min_component_width_px",
    "max_component_width_px",
    "center_tolerance_px",
    "min_contrast_snr",
    "min_coverage_ratio",
    "min_longest_run_ratio",
    "max_gap_ratio",
)
_INTEGER_THRESHOLD_NAMES = (
    "background_kernel_px",
    "min_component_area_px",
    "micro_gap_close_px",
    "max_gap_count",
)
_THRESHOLD_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "min_mean_intensity": (0.0, 255.0),
    "max_mean_intensity": (0.0, 255.0),
    "max_dark_clip_ratio": (0.0, 1.0),
    "max_bright_clip_ratio": (0.0, 1.0),
    "min_coverage_ratio": (0.0, 1.0),
    "min_longest_run_ratio": (0.0, 1.0),
    "max_gap_ratio": (0.0, 1.0),
}
_REQUIRED_ROOT_FIELDS = {
    "schema_version",
    "mode",
    "camera_serial",
    "image_width",
    "image_height",
    "exposure",
    "gain",
    "timeout_ms",
    "warmup_frames",
    "roi_xyxy",
    "result_root",
    "thresholds",
}
_REQUIRED_THRESHOLD_FIELDS = set(_FINITE_THRESHOLD_NAMES) | set(_INTEGER_THRESHOLD_NAMES)


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        payload[key] = value
    return payload


def read_json_object(path: Path) -> dict[str, Any]:
    """Read one strict JSON object while rejecting duplicates and NaN/Infinity."""
    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(
            resolved.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON configuration {resolved}: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ValueError("BMW Demo configuration must be a JSON object")
    return payload


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _positive_integer(value: object, name: str) -> int:
    parsed = _integer(value, name)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _finite_number(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and (parsed < minimum if minimum_inclusive else parsed <= minimum):
        operator = ">=" if minimum_inclusive else ">"
        raise ValueError(f"{name} must be {operator} {minimum:g}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{name} must be <= {maximum:g}")
    return parsed


def _validate_roi(
    value: tuple[int, int, int, int] | None,
    image_width: int,
    image_height: int,
) -> None:
    if value is None:
        return
    if (
        not isinstance(value, tuple)
        or len(value) != 4
        or any(isinstance(coordinate, bool) or not isinstance(coordinate, int) for coordinate in value)
    ):
        raise ValueError("roi_xyxy must be null or four integer coordinates")
    x1, y1, x2, y2 = value
    if not (0 <= x1 < x2 <= image_width and 0 <= y1 < y2 <= image_height):
        raise ValueError("roi_xyxy must be a positive-area half-open rectangle within the configured image")


def _roi(value: object) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("roi_xyxy must be null or a four-element array")
    if any(isinstance(coordinate, bool) or not isinstance(coordinate, int) for coordinate in value):
        raise ValueError("roi_xyxy coordinates must be integers")
    return value[0], value[1], value[2], value[3]


def config_from_payload(path: Path, payload: dict[str, Any]) -> BrightStreakConfig:
    """Validate a parsed payload without discarding extension fields."""
    missing = sorted(_REQUIRED_ROOT_FIELDS - set(payload))
    if missing:
        raise ValueError(f"BMW Demo configuration is missing fields: {missing}")
    if payload["schema_version"] != 1:
        raise ValueError("schema_version must be exactly 1")
    if payload["mode"] != "demo":
        raise ValueError("mode must be exactly 'demo'")
    thresholds = payload["thresholds"]
    if not isinstance(thresholds, dict):
        raise TypeError("thresholds must be a JSON object")
    missing_thresholds = sorted(_REQUIRED_THRESHOLD_FIELDS - set(thresholds))
    if missing_thresholds:
        raise ValueError(f"thresholds are missing fields: {missing_thresholds}")
    serial = payload["camera_serial"]
    result_root = payload["result_root"]
    if not isinstance(serial, str):
        raise TypeError("camera_serial must be a string")
    if not isinstance(result_root, str) or not result_root.strip():
        raise ValueError("result_root must be a non-empty string")
    return BrightStreakConfig(
        path=Path(path).expanduser().resolve(),
        camera_serial=serial,
        image_width=_integer(payload["image_width"], "image_width"),
        image_height=_integer(payload["image_height"], "image_height"),
        exposure=_finite_number(payload["exposure"], "exposure"),
        gain=_finite_number(payload["gain"], "gain"),
        timeout_ms=_integer(payload["timeout_ms"], "timeout_ms"),
        warmup_frames=_integer(payload["warmup_frames"], "warmup_frames"),
        roi_xyxy=_roi(payload["roi_xyxy"]),
        result_root=Path(result_root),
        **{name: thresholds[name] for name in _REQUIRED_THRESHOLD_FIELDS},
    )


def load_config(path: Path) -> BrightStreakConfig:
    """Load and strictly validate one editable BMW Demo JSON configuration."""
    resolved = Path(path).expanduser().resolve()
    return config_from_payload(resolved, read_json_object(resolved))
