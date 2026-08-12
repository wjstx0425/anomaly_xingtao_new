"""Pure tracked-profile evidence for the fixed BMW bright-streak ROI."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Mapping, Sequence

import numpy as np


TRACKED_PROFILE_ROI_SHAPE = (613, 81)


@dataclass(frozen=True, slots=True)
class TrackedProfileGeometry:
    """Geometry and path constraints for the fixed bright-streak ROI."""

    candidate_width: int = 7
    background_width: int = 10
    background_gap: int = 3
    smooth_window: int = 5
    max_step: int = 2
    step_penalty: float = 1.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.smooth_window, bool)
            or not isinstance(self.smooth_window, int)
            or self.smooth_window <= 0
            or self.smooth_window % 2 == 0
        ):
            raise ValueError("tracked-profile smooth_window must be a positive odd integer")
        if (
            isinstance(self.candidate_width, bool)
            or not isinstance(self.candidate_width, int)
            or self.candidate_width <= 0
            or self.candidate_width % 2 == 0
        ):
            raise ValueError("tracked-profile candidate_width must be a positive odd integer")
        if (
            isinstance(self.background_width, bool)
            or not isinstance(self.background_width, int)
            or self.background_width <= 0
        ):
            raise ValueError("tracked-profile background_width must be a positive integer")
        if (
            isinstance(self.background_gap, bool)
            or not isinstance(self.background_gap, int)
            or self.background_gap < 0
        ):
            raise ValueError("tracked-profile background_gap must be a non-negative integer")
        if (
            isinstance(self.max_step, bool)
            or not isinstance(self.max_step, int)
            or self.max_step < 0
        ):
            raise ValueError("tracked-profile max_step must be a non-negative integer")
        if (
            isinstance(self.step_penalty, bool)
            or not isinstance(self.step_penalty, (int, float))
            or not np.isfinite(self.step_penalty)
            or self.step_penalty < 0.0
        ):
            raise ValueError("tracked-profile step_penalty must be a finite non-negative number")


# The narrower candidate band was selected from calibration-only evidence: it
# doubles the weakest normal accepted coverage and reduces the calibration gap
# envelope while retaining strict normal/no-streak peak separation. Task-1's
# dataclass defaults remain the general synthetic reference geometry.
BMW_TRACKED_PROFILE_V3_GEOMETRY = TrackedProfileGeometry(candidate_width=5)


@dataclass(frozen=True, slots=True)
class TrackedProfileThresholds:
    """Presence and continuity thresholds applied to one tracked path."""

    strong_row_score: float
    weak_row_score: float
    min_presence_coverage_ratio: float
    min_longest_run_ratio: float
    max_gap_ratio: float
    max_gap_count: int

    def __post_init__(self) -> None:
        numeric = (
            self.strong_row_score,
            self.weak_row_score,
            self.min_presence_coverage_ratio,
            self.min_longest_run_ratio,
            self.max_gap_ratio,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
            for value in numeric
        ):
            raise ValueError("tracked-profile thresholds must be finite")
        if self.weak_row_score > self.strong_row_score:
            raise ValueError("tracked-profile weak_row_score must not exceed strong_row_score")
        ratios = (
            self.min_presence_coverage_ratio,
            self.min_longest_run_ratio,
            self.max_gap_ratio,
        )
        if any(value < 0.0 or value > 1.0 for value in ratios):
            raise ValueError("tracked-profile ratio thresholds must be within [0, 1]")
        if (
            isinstance(self.max_gap_count, bool)
            or not isinstance(self.max_gap_count, int)
            or self.max_gap_count < 0
        ):
            raise ValueError("tracked-profile max_gap_count must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class TrackedProfileMetrics:
    """Owned immutable arrays and explainable statistics for one tracked path."""

    response_map: np.ndarray
    path_x: np.ndarray
    path_scores: np.ndarray
    strong_mask: np.ndarray
    accepted_mask: np.ndarray
    bridged_mask: np.ndarray
    active_start_row: int | None
    active_stop_row: int | None
    path_total_score: float
    coverage_ratio: float
    longest_run_px: int
    longest_run_ratio: float
    max_gap_px: int
    max_gap_ratio: float
    gap_count: int


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    result = np.array(values, copy=True)
    result.setflags(write=False)
    return result


def _median_smooth_rows(values: np.ndarray, window: int) -> np.ndarray:
    if window == 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0)
    return np.median(windows, axis=-1)


def _response_map(
    image: np.ndarray,
    geometry: TrackedProfileGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    half_candidate = geometry.candidate_width // 2
    edge = half_candidate + geometry.background_gap + geometry.background_width
    centres = np.arange(edge, image.shape[1] - edge, dtype=np.int64)
    if centres.size == 0:
        raise ValueError("tracked-profile bands do not fit inside the fixed 81-pixel ROI")

    grayscale = image.astype(np.float64, copy=False)
    responses = np.empty((image.shape[0], centres.size), dtype=np.float64)
    for index, centre in enumerate(centres):
        candidate_start = centre - half_candidate
        candidate_stop = centre + half_candidate + 1
        left_stop = candidate_start - geometry.background_gap
        left_start = left_stop - geometry.background_width
        right_start = candidate_stop + geometry.background_gap
        right_stop = right_start + geometry.background_width
        candidate = grayscale[:, candidate_start:candidate_stop].mean(axis=1)
        left = grayscale[:, left_start:left_stop].mean(axis=1)
        right = grayscale[:, right_start:right_stop].mean(axis=1)
        responses[:, index] = candidate - (left + right) / 2.0
    return _median_smooth_rows(responses, geometry.smooth_window), centres


def _track_path(
    responses: np.ndarray,
    centres: np.ndarray,
    geometry: TrackedProfileGeometry,
) -> tuple[np.ndarray, float]:
    height, width = responses.shape
    cumulative = responses[0].copy()
    predecessors = np.empty((height, width), dtype=np.int64)
    predecessors[0] = np.arange(width)
    for row in range(1, height):
        previous = cumulative
        cumulative = np.empty(width, dtype=np.float64)
        for current in range(width):
            first = max(0, current - geometry.max_step)
            stop = min(width, current + geometry.max_step + 1)
            previous_indexes = np.arange(first, stop)
            transition_scores = previous[first:stop] - geometry.step_penalty * np.abs(
                centres[current] - centres[first:stop]
            )
            relative_predecessor = int(np.argmax(transition_scores))
            predecessor = int(previous_indexes[relative_predecessor])
            predecessors[row, current] = predecessor
            cumulative[current] = responses[row, current] + transition_scores[relative_predecessor]

    final_index = int(np.argmax(cumulative))
    total_score = float(cumulative[final_index])
    path_indexes = np.empty(height, dtype=np.int64)
    path_indexes[-1] = final_index
    for row in range(height - 1, 0, -1):
        path_indexes[row - 1] = predecessors[row, path_indexes[row]]
    return centres[path_indexes], total_score


def _runs(mask: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        elif not active and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    gaps = [(left[1], right[0]) for left, right in zip(runs, runs[1:], strict=False)]
    return runs, gaps


def _metrics_from_path(
    responses: np.ndarray,
    path_x: np.ndarray,
    path_scores: np.ndarray,
    path_total_score: float,
    thresholds: TrackedProfileThresholds,
) -> TrackedProfileMetrics:
    strong_mask = path_scores >= thresholds.strong_row_score
    weak_mask = path_scores >= thresholds.weak_row_score
    accepted_mask = np.zeros(path_scores.shape, dtype=np.bool_)
    weak_runs, _ = _runs(weak_mask)
    for start, stop in weak_runs:
        if strong_mask[start:stop].any():
            accepted_mask[start:stop] = True
    bridged_mask = accepted_mask & ~strong_mask
    runs, gaps = _runs(accepted_mask)
    height = len(path_scores)
    longest_run_px = max((stop - start for start, stop in runs), default=0)
    max_gap_px = max((stop - start for start, stop in gaps), default=0)
    return TrackedProfileMetrics(
        response_map=_readonly_copy(responses),
        path_x=_readonly_copy(path_x),
        path_scores=_readonly_copy(path_scores),
        strong_mask=_readonly_copy(strong_mask),
        accepted_mask=_readonly_copy(accepted_mask),
        bridged_mask=_readonly_copy(bridged_mask),
        active_start_row=(runs[0][0] if runs else None),
        active_stop_row=(runs[-1][1] if runs else None),
        path_total_score=path_total_score,
        coverage_ratio=float(accepted_mask.mean()),
        longest_run_px=longest_run_px,
        longest_run_ratio=longest_run_px / height,
        max_gap_px=max_gap_px,
        max_gap_ratio=max_gap_px / height,
        gap_count=len(gaps),
    )


def analyze_tracked_profile(
    image: np.ndarray,
    geometry: TrackedProfileGeometry,
    thresholds: TrackedProfileThresholds,
) -> TrackedProfileMetrics:
    """Track the highest-scoring smooth local-contrast path in one fixed ROI."""
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")
    if not isinstance(geometry, TrackedProfileGeometry):
        raise TypeError("geometry must be TrackedProfileGeometry")
    if not isinstance(thresholds, TrackedProfileThresholds):
        raise TypeError("thresholds must be TrackedProfileThresholds")
    if image.ndim != 2 or image.shape != TRACKED_PROFILE_ROI_SHAPE:
        raise ValueError(
            "tracked bright-streak profile requires a grayscale ROI with shape (613, 81)"
        )

    responses, centres = _response_map(image, geometry)
    path_x, path_total_score = _track_path(responses, centres, geometry)
    first_centre = int(centres[0])
    path_scores = responses[np.arange(image.shape[0]), path_x - first_centre]
    return _metrics_from_path(responses, path_x, path_scores, path_total_score, thresholds)


def classify_tracked_profile(
    metrics: TrackedProfileMetrics,
    thresholds: TrackedProfileThresholds,
) -> str:
    """Classify presence before applying bounded continuity rules."""
    if not isinstance(metrics, TrackedProfileMetrics):
        raise TypeError("metrics must be TrackedProfileMetrics")
    if not isinstance(thresholds, TrackedProfileThresholds):
        raise TypeError("thresholds must be TrackedProfileThresholds")
    if not metrics.strong_mask.any() or metrics.active_start_row is None:
        return "NG_NO_STREAK"
    if metrics.coverage_ratio < thresholds.min_presence_coverage_ratio:
        return "NG_NO_STREAK"
    if (
        metrics.longest_run_ratio < thresholds.min_longest_run_ratio
        or metrics.max_gap_ratio > thresholds.max_gap_ratio
        or metrics.gap_count > thresholds.max_gap_count
    ):
        return "NG_BROKEN"
    return "OK"


def fit_tracked_profile_thresholds(
    records: Sequence[Mapping[str, object]],
    geometry: TrackedProfileGeometry,
) -> TrackedProfileThresholds:
    """Fit separated presence seeds and normal-only continuity envelopes."""
    if not isinstance(geometry, TrackedProfileGeometry):
        raise TypeError("geometry must be TrackedProfileGeometry")

    normal: list[np.ndarray] = []
    no_streak: list[np.ndarray] = []
    for record in records:
        if "split" in record and record["split"] != "calibration":
            raise ValueError("tracked-profile threshold fitting accepts only split='calibration'")
        if "final_test" in record and record["final_test"] is not False:
            raise ValueError(
                "tracked-profile fitting final_test must be the boolean false when provided"
            )
        label = record.get("label")
        if label not in {"normal", "no_streak"}:
            raise ValueError("tracked-profile calibration labels must be normal or no_streak")
        path_scores = record.get("path_scores")
        if not isinstance(path_scores, np.ndarray):
            raise TypeError("tracked-profile calibration path_scores must be numpy arrays")
        if path_scores.ndim != 1 or path_scores.shape != (TRACKED_PROFILE_ROI_SHAPE[0],):
            raise ValueError("tracked-profile calibration path_scores must have shape (613,)")
        scores = path_scores.astype(np.float64, copy=False)
        if not np.isfinite(scores).all():
            raise ValueError("tracked-profile calibration path_scores must be finite")
        (normal if label == "normal" else no_streak).append(scores)

    if not normal or not no_streak:
        raise ValueError("tracked-profile calibration needs both normal and no_streak samples")

    no_streak_peak = max(float(scores.max()) for scores in no_streak)
    normal_peak_floor = min(float(scores.max()) for scores in normal)
    if no_streak_peak >= normal_peak_floor:
        raise ValueError("tracked-profile calibration peak envelopes must be strictly separated")
    separation = normal_peak_floor - no_streak_peak
    strong_row_score = no_streak_peak + separation / 2.0
    weak_row_score = no_streak_peak + separation / 4.0
    provisional = TrackedProfileThresholds(
        strong_row_score=strong_row_score,
        weak_row_score=weak_row_score,
        min_presence_coverage_ratio=0.0,
        min_longest_run_ratio=0.0,
        max_gap_ratio=1.0,
        max_gap_count=TRACKED_PROFILE_ROI_SHAPE[0],
    )
    dummy_responses = np.empty((TRACKED_PROFILE_ROI_SHAPE[0], 0), dtype=np.float64)
    dummy_path_x = np.zeros(TRACKED_PROFILE_ROI_SHAPE[0], dtype=np.int64)
    normal_metrics = [
        _metrics_from_path(
            dummy_responses,
            dummy_path_x,
            scores,
            float(scores.sum()),
            provisional,
        )
        for scores in normal
    ]
    return TrackedProfileThresholds(
        strong_row_score=strong_row_score,
        weak_row_score=weak_row_score,
        min_presence_coverage_ratio=min(item.coverage_ratio for item in normal_metrics),
        min_longest_run_ratio=min(item.longest_run_ratio for item in normal_metrics),
        max_gap_ratio=max(item.max_gap_ratio for item in normal_metrics),
        max_gap_count=max(item.gap_count for item in normal_metrics),
    )


__all__ = [
    "BMW_TRACKED_PROFILE_V3_GEOMETRY",
    "TRACKED_PROFILE_ROI_SHAPE",
    "TrackedProfileGeometry",
    "TrackedProfileMetrics",
    "TrackedProfileThresholds",
    "analyze_tracked_profile",
    "classify_tracked_profile",
    "fit_tracked_profile_thresholds",
]
