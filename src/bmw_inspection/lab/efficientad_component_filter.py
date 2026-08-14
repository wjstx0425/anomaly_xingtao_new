"""Deterministic component-aware scoring for EfficientAD anomaly maps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class ComponentFilterPolicy:
    """Threshold and geometry gates for one view's anomaly components."""

    low_threshold: float
    seed_threshold: float
    p95_threshold: float
    minimum_area: int
    hard_peak_threshold: float
    line_minimum_length: int
    line_minimum_area: int

    def __post_init__(self) -> None:
        numeric_fields = (
            "low_threshold",
            "seed_threshold",
            "p95_threshold",
            "hard_peak_threshold",
        )
        for name in numeric_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite number")
            if float(value) < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, float(value))
        integer_fields = ("minimum_area", "line_minimum_length", "line_minimum_area")
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if self.seed_threshold < self.low_threshold:
            raise ValueError("seed_threshold must be at least low_threshold")
        if self.p95_threshold < self.low_threshold:
            raise ValueError("p95_threshold must be at least low_threshold")
        if self.hard_peak_threshold < self.seed_threshold:
            raise ValueError("hard_peak_threshold must be at least seed_threshold")


@dataclass(frozen=True, slots=True)
class ComponentStatistics:
    """Stable JSON-shaped evidence for one low-threshold component."""

    label: int
    area: int
    peak: float
    mean: float
    p95: float
    bounding_box_xyxy: tuple[int, int, int, int]
    contains_seed: bool
    accepted: bool
    acceptance_reason: str


@dataclass(frozen=True, slots=True)
class ComponentScoreResult:
    """Decision score and component evidence derived from one anomaly map."""

    score: float
    accepted_components: tuple[ComponentStatistics, ...]
    rejected_components: tuple[ComponentStatistics, ...]
    accepted_mask: np.ndarray
    hotspot: tuple[int, int] | None
    ignored_pixel_count: int

    def __post_init__(self) -> None:
        owned_mask = np.asarray(self.accepted_mask, dtype=np.uint8).copy()
        owned_mask.flags.writeable = False
        object.__setattr__(self, "accepted_mask", owned_mask)
        object.__setattr__(self, "accepted_components", tuple(self.accepted_components))
        object.__setattr__(self, "rejected_components", tuple(self.rejected_components))


def _ignore_pixels(ignore_mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    """Return a boolean ignore mask resized to the anomaly-map grid."""
    if ignore_mask is None:
        return np.zeros(shape, dtype=bool)
    array = np.asarray(ignore_mask)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("ignore_mask must be a non-empty two-dimensional array")
    if not np.issubdtype(array.dtype, np.bool_):
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError("ignore_mask must be finite and binary")
        unique = set(np.unique(array).tolist())
        if not unique.issubset({0, 1, 255}):
            raise ValueError("ignore_mask must be binary")
    binary = array.astype(bool, copy=False).astype(np.uint8)
    if binary.shape != shape:
        binary = cv2.resize(binary, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return binary.astype(bool, copy=False)


def _acceptance_reason(
    *,
    area: int,
    peak: float,
    p95: float,
    longest_side: int,
    contains_seed: bool,
    policy: ComponentFilterPolicy,
) -> tuple[bool, str]:
    if not contains_seed:
        return False, "below_seed"
    if area >= policy.minimum_area and p95 >= policy.p95_threshold:
        return True, "area_p95"
    if peak >= policy.hard_peak_threshold:
        return True, "hard_peak"
    if area >= policy.line_minimum_area and longest_side >= policy.line_minimum_length:
        return True, "line"
    return False, "below_component_gates"


def score_anomaly_components(
    anomaly_map: np.ndarray,
    ignore_mask: np.ndarray | None,
    policy: ComponentFilterPolicy,
) -> ComponentScoreResult:
    """Score accepted 8-connected anomaly components with one shared policy."""
    values = np.asarray(anomaly_map)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("anomaly_map must be a non-empty two-dimensional array")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("anomaly_map must contain only finite numbers")
    working = values.astype(np.float32, copy=True)
    ignored = _ignore_pixels(ignore_mask, working.shape)
    working[ignored] = 0.0
    low_mask = (working >= policy.low_threshold).astype(np.uint8)
    component_count, labels, statistics, _centroids = cv2.connectedComponentsWithStats(
        low_mask,
        connectivity=8,
    )

    accepted: list[ComponentStatistics] = []
    rejected: list[ComponentStatistics] = []
    accepted_mask = np.zeros(working.shape, dtype=np.uint8)
    for label in range(1, component_count):
        component_mask = labels == label
        component_values = working[component_mask]
        x = int(statistics[label, cv2.CC_STAT_LEFT])
        y = int(statistics[label, cv2.CC_STAT_TOP])
        width = int(statistics[label, cv2.CC_STAT_WIDTH])
        height = int(statistics[label, cv2.CC_STAT_HEIGHT])
        area = int(statistics[label, cv2.CC_STAT_AREA])
        peak = float(np.max(component_values))
        mean = float(np.mean(component_values, dtype=np.float64))
        p95 = float(np.percentile(component_values, 95))
        contains_seed = bool(peak >= policy.seed_threshold)
        is_accepted, reason = _acceptance_reason(
            area=area,
            peak=peak,
            p95=p95,
            longest_side=max(width, height),
            contains_seed=contains_seed,
            policy=policy,
        )
        component = ComponentStatistics(
            label=label,
            area=area,
            peak=peak,
            mean=mean,
            p95=p95,
            bounding_box_xyxy=(x, y, x + width, y + height),
            contains_seed=contains_seed,
            accepted=is_accepted,
            acceptance_reason=reason,
        )
        if is_accepted:
            accepted.append(component)
            accepted_mask[component_mask] = 255
        else:
            rejected.append(component)

    if not accepted:
        score = 0.0
        hotspot = None
    else:
        score = max(component.p95 for component in accepted)
        hotspot_flat = int(np.argmax(np.where(accepted_mask != 0, working, -np.inf)))
        hotspot_y, hotspot_x = np.unravel_index(hotspot_flat, working.shape)
        hotspot = (int(hotspot_x), int(hotspot_y))
    return ComponentScoreResult(
        score=score,
        accepted_components=tuple(accepted),
        rejected_components=tuple(rejected),
        accepted_mask=accepted_mask,
        hotspot=hotspot,
        ignored_pixel_count=int(np.count_nonzero(ignored)),
    )


__all__ = [
    "ComponentFilterPolicy",
    "ComponentScoreResult",
    "ComponentStatistics",
    "score_anomaly_components",
]
