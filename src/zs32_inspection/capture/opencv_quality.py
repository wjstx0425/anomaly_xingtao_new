"""Configuration-bound OpenCV image-quality gate.

The adapter is intentionally independent from the capture service and imports
OpenCV/NumPy only when :meth:`OpenCvQualityGate.evaluate` is called.  Every
required view must have an explicit threshold record; there is no disabled,
warning-only, or implicit-pass mode.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from zs32_inspection.domain.contracts import canonical_sha256

from .contracts import (
    CaptureFrame,
    CapturePlan,
    CaptureRequest,
    require_identifier,
    require_sha256,
)
from .gates import CaptureGateResult


DependencyLoader = Callable[[], tuple[Any, Any]]


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _bounded(value: object, field: str, minimum: float, maximum: float) -> float:
    result = _finite_number(value, field)
    if not minimum <= result <= maximum:
        raise ValueError(f"{field} must be in [{minimum}, {maximum}]")
    return result


@dataclass(frozen=True, slots=True)
class QualityViewThresholds:
    """All required image-quality thresholds for one topology view."""

    brightness_mean_min: float
    brightness_mean_max: float
    brightness_std_min: float
    dark_pixel_level: int
    dark_ratio_max: float
    saturated_pixel_level: int
    saturated_ratio_max: float
    laplacian_variance_min: float

    def __post_init__(self) -> None:
        minimum = _bounded(
            self.brightness_mean_min,
            "brightness_mean_min",
            0.0,
            255.0,
        )
        maximum = _bounded(
            self.brightness_mean_max,
            "brightness_mean_max",
            0.0,
            255.0,
        )
        if minimum >= maximum:
            raise ValueError("brightness_mean_min must be below brightness_mean_max")
        object.__setattr__(self, "brightness_mean_min", minimum)
        object.__setattr__(self, "brightness_mean_max", maximum)
        object.__setattr__(
            self,
            "brightness_std_min",
            _bounded(self.brightness_std_min, "brightness_std_min", 0.0, 127.5),
        )
        for field in ("dark_pixel_level", "saturated_pixel_level"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
                raise ValueError(f"{field} must be an integer in [0, 255]")
        if self.dark_pixel_level >= self.saturated_pixel_level:
            raise ValueError("dark_pixel_level must be below saturated_pixel_level")
        object.__setattr__(
            self,
            "dark_ratio_max",
            _bounded(self.dark_ratio_max, "dark_ratio_max", 0.0, 1.0),
        )
        object.__setattr__(
            self,
            "saturated_ratio_max",
            _bounded(self.saturated_ratio_max, "saturated_ratio_max", 0.0, 1.0),
        )
        laplacian_minimum = _finite_number(
            self.laplacian_variance_min,
            "laplacian_variance_min",
        )
        if laplacian_minimum < 0.0:
            raise ValueError("laplacian_variance_min must be non-negative")
        object.__setattr__(self, "laplacian_variance_min", laplacian_minimum)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "brightness_mean_min": self.brightness_mean_min,
            "brightness_mean_max": self.brightness_mean_max,
            "brightness_std_min": self.brightness_std_min,
            "dark_pixel_level": self.dark_pixel_level,
            "dark_ratio_max": self.dark_ratio_max,
            "saturated_pixel_level": self.saturated_pixel_level,
            "saturated_ratio_max": self.saturated_ratio_max,
            "laplacian_variance_min": self.laplacian_variance_min,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "QualityViewThresholds":
        if not isinstance(payload, Mapping):
            raise TypeError("quality view thresholds must be an object")
        expected = {
            "brightness_mean_min",
            "brightness_mean_max",
            "brightness_std_min",
            "dark_pixel_level",
            "dark_ratio_max",
            "saturated_pixel_level",
            "saturated_ratio_max",
            "laplacian_variance_min",
        }
        if set(payload) != expected:
            raise ValueError(
                "quality view thresholds fields differ from the strict schema; "
                f"missing={sorted(expected - set(payload))}, "
                f"extra={sorted(set(payload) - expected)}"
            )
        return cls(**{key: payload[key] for key in expected})  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class QualityGateProfile:
    """Immutable, topology-bound quality thresholds for every required view."""

    schema_version: int
    profile_id: str
    product: str
    hand: str
    topology_id: str
    topology_sha256: str
    views: Mapping[str, QualityViewThresholds]
    profile_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != 1
        ):
            raise ValueError("quality gate schema_version must be 1")
        if self.product != "ZS32":
            raise ValueError("quality gate product must be ZS32")
        if not isinstance(self.hand, str) or self.hand.strip().lower() not in {"left", "right"}:
            raise ValueError("quality gate hand must be explicitly left or right")
        object.__setattr__(self, "hand", self.hand.strip().lower())
        object.__setattr__(self, "profile_id", require_identifier(self.profile_id, "profile_id"))
        object.__setattr__(
            self,
            "topology_id",
            require_identifier(self.topology_id, "topology_id"),
        )
        object.__setattr__(
            self,
            "topology_sha256",
            require_sha256(self.topology_sha256, "topology_sha256"),
        )
        object.__setattr__(
            self,
            "profile_sha256",
            require_sha256(self.profile_sha256, "quality profile_sha256"),
        )
        if not isinstance(self.views, Mapping) or not self.views:
            raise ValueError("quality gate profile must contain explicit per-view thresholds")
        frozen: dict[str, QualityViewThresholds] = {}
        for view_id, thresholds in self.views.items():
            checked_view = require_identifier(view_id, "quality view_id")
            if not isinstance(thresholds, QualityViewThresholds):
                raise TypeError(f"quality thresholds for {checked_view!r} have the wrong type")
            frozen[checked_view] = thresholds
        object.__setattr__(self, "views", MappingProxyType(frozen))
        if canonical_sha256(self.as_dict(include_sha256=False)) != self.profile_sha256:
            raise ValueError("quality profile_sha256 does not match the canonical profile payload")

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "product": self.product,
            "hand": self.hand,
            "topology_id": self.topology_id,
            "topology_sha256": self.topology_sha256,
            "views": {
                view: thresholds.as_dict()
                for view, thresholds in sorted(self.views.items())
            },
        }
        if include_sha256:
            payload["profile_sha256"] = self.profile_sha256
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "QualityGateProfile":
        if not isinstance(payload, Mapping):
            raise TypeError("quality gate profile must be an object")
        expected = {
            "schema_version",
            "profile_id",
            "product",
            "hand",
            "topology_id",
            "topology_sha256",
            "views",
            "profile_sha256",
        }
        if set(payload) != expected:
            raise ValueError(
                "quality gate profile fields differ from the strict schema; "
                f"missing={sorted(expected - set(payload))}, "
                f"extra={sorted(set(payload) - expected)}"
            )
        raw_views = payload["views"]
        if not isinstance(raw_views, Mapping) or any(
            not isinstance(key, str) for key in raw_views
        ):
            raise TypeError("quality gate views must be an object keyed by view_id")
        return cls(
            schema_version=payload["schema_version"],  # type: ignore[arg-type]
            profile_id=payload["profile_id"],  # type: ignore[arg-type]
            product=payload["product"],  # type: ignore[arg-type]
            hand=payload["hand"],  # type: ignore[arg-type]
            topology_id=payload["topology_id"],  # type: ignore[arg-type]
            topology_sha256=payload["topology_sha256"],  # type: ignore[arg-type]
            views={
                view: QualityViewThresholds.from_mapping(item)  # type: ignore[arg-type]
                for view, item in raw_views.items()
            },
            profile_sha256=payload["profile_sha256"],  # type: ignore[arg-type]
        )

    def validate_plan(self, plan: CapturePlan) -> None:
        if plan.product != self.product:
            raise ValueError("quality gate product conflicts with capture plan")
        if plan.topology_id != self.topology_id or plan.topology_sha256 != self.topology_sha256:
            raise ValueError("quality gate topology identity conflicts with capture plan")
        expected = set(plan.required_views)
        actual = set(self.views)
        if actual != expected:
            raise ValueError(
                "quality gate view set differs from required topology views; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )


def _load_opencv_dependencies() -> tuple[Any, Any]:
    return importlib.import_module("cv2"), importlib.import_module("numpy")


class OpenCvQualityGate:
    """Measure deterministic full-frame quality metrics for every view."""

    name = "quality"

    def __init__(
        self,
        profile: QualityGateProfile,
        *,
        dependency_loader: DependencyLoader | None = None,
    ) -> None:
        if not isinstance(profile, QualityGateProfile):
            raise TypeError("profile must be a QualityGateProfile")
        self.profile = profile
        self._dependency_loader = dependency_loader or _load_opencv_dependencies

    def evaluate(
        self,
        request: CaptureRequest,
        plan: CapturePlan,
        frames: Sequence[CaptureFrame],
    ) -> Sequence[CaptureGateResult]:
        self.profile.validate_plan(plan)
        if request.hand != self.profile.hand:
            raise ValueError(
                f"quality gate hand {self.profile.hand!r} conflicts with capture hand "
                f"{request.hand!r}"
            )
        frame_by_view = _require_exact_frames(plan, frames, "quality")
        cv2, np = self._dependency_loader()
        results: list[CaptureGateResult] = []
        for view_id in plan.required_views:
            frame = frame_by_view[view_id]
            image = _decode_color_frame(frame, cv2, np)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresholds = self.profile.views[view_id]
            brightness_mean = float(gray.mean())
            brightness_std = float(gray.std())
            dark_ratio = float((gray <= thresholds.dark_pixel_level).mean())
            saturated_ratio = float(
                (image >= thresholds.saturated_pixel_level).any(axis=2).mean()
            )
            laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            metrics = {
                "brightness_mean": brightness_mean,
                "brightness_std": brightness_std,
                "dark_ratio": dark_ratio,
                "saturated_ratio": saturated_ratio,
                "laplacian_variance": laplacian_variance,
            }
            if any(not math.isfinite(value) for value in metrics.values()):
                raise RuntimeError(f"quality metrics are non-finite for view {view_id!r}")
            failures: list[str] = []
            if not thresholds.brightness_mean_min <= brightness_mean <= thresholds.brightness_mean_max:
                failures.append(
                    "brightness_mean outside "
                    f"[{thresholds.brightness_mean_min:.6g},{thresholds.brightness_mean_max:.6g}]"
                )
            if brightness_std < thresholds.brightness_std_min:
                failures.append(
                    f"brightness_std below {thresholds.brightness_std_min:.6g}"
                )
            if dark_ratio > thresholds.dark_ratio_max:
                failures.append(f"dark_ratio above {thresholds.dark_ratio_max:.6g}")
            if saturated_ratio > thresholds.saturated_ratio_max:
                failures.append(
                    f"saturated_ratio above {thresholds.saturated_ratio_max:.6g}"
                )
            if laplacian_variance < thresholds.laplacian_variance_min:
                failures.append(
                    "laplacian_variance below "
                    f"{thresholds.laplacian_variance_min:.6g}"
                )
            metric_text = ",".join(
                f"{key}={value:.6g}" for key, value in sorted(metrics.items())
            )
            prefix = f"profile_sha256={self.profile.profile_sha256}; {metric_text}"
            reason = prefix if not failures else f"{'; '.join(failures)}; {prefix}"
            results.append(CaptureGateResult("quality", not failures, reason, view_id))
        return tuple(results)


def _require_exact_frames(
    plan: CapturePlan,
    frames: Sequence[CaptureFrame],
    gate_name: str,
) -> dict[str, CaptureFrame]:
    frame_by_view: dict[str, CaptureFrame] = {}
    for frame in frames:
        if frame.view_id in frame_by_view:
            raise ValueError(f"{gate_name} gate received duplicate view {frame.view_id!r}")
        frame_by_view[frame.view_id] = frame
    expected = set(plan.required_views)
    actual = set(frame_by_view)
    if actual != expected:
        raise ValueError(
            f"{gate_name} gate frames differ from required topology views; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return frame_by_view


def _decode_color_frame(frame: CaptureFrame, cv2: Any, np: Any) -> Any:
    encoded = np.frombuffer(frame.image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or getattr(image, "ndim", 0) != 3 or image.shape[2] != 3:
        raise ValueError(f"quality gate cannot decode color PNG for view {frame.view_id!r}")
    actual_height, actual_width = image.shape[:2]
    if (actual_width, actual_height) != (frame.width, frame.height):
        raise ValueError(
            f"quality frame dimensions conflict for view {frame.view_id!r}: "
            f"decoded={actual_width}x{actual_height}, "
            f"declared={frame.width}x{frame.height}"
        )
    return image
