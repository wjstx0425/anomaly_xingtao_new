"""Deterministic OpenCV rules for the BMW bright-streak Demo."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

import cv2
import numpy as np

from .contracts import BrightStreakConfig, BrightStreakMetrics, BrightStreakResult, DemoStatus


_DARK_CLIP_LEVEL = 1
_BRIGHT_CLIP_LEVEL = 254


@dataclass(frozen=True, slots=True)
class BrightStreakDecision:
    """Structured pixels and geometry used to make one bright-streak decision."""

    result: BrightStreakResult
    roi: np.ndarray
    response: np.ndarray
    mask: np.ndarray
    mask_used_for_metrics: np.ndarray
    runs: tuple[tuple[int, int], ...]
    gaps: tuple[tuple[int, int], ...]

    @property
    def status(self) -> DemoStatus:
        """Return the business status without requiring callers to unpack ``result``."""
        return self.result.status

    @property
    def metrics(self) -> BrightStreakMetrics | None:
        """Return the measured decision metrics, if image validation reached detection."""
        return self.result.metrics

    @property
    def accepted_mask(self) -> np.ndarray:
        """Return the accepted decision mask under its descriptive public name."""
        return self.mask


def _empty_result(config: BrightStreakConfig, reason: str) -> BrightStreakResult:
    roi = config.roi_xyxy if config.roi_xyxy is not None else (0, 0, 0, 0)
    return BrightStreakResult(DemoStatus.ERROR, reason, roi, None)


def _gray_image(image: np.ndarray) -> np.ndarray | None:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        return None
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return None


def _runs(presence: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(presence.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def _fill_micro_gaps(presence: np.ndarray, maximum: int) -> np.ndarray:
    filled = presence.copy()
    if maximum <= 0:
        return filled
    present_runs = _runs(filled)
    for (_left_start, left_end), (right_start, _right_end) in zip(
        present_runs,
        present_runs[1:],
        strict=False,
    ):
        if right_start - left_end <= maximum:
            filled[left_end:right_start] = True
    return filled


def _metrics(
    *,
    roi: np.ndarray,
    selected_mask: np.ndarray,
    contrast_snr: float,
    micro_gap_close_px: int,
) -> tuple[BrightStreakMetrics, list[tuple[int, int]], list[tuple[int, int]]]:
    presence = _fill_micro_gaps(selected_mask.any(axis=1), micro_gap_close_px)
    present_runs = _runs(presence)
    roi_height = max(1, roi.shape[0])
    if present_runs:
        coverage_ratio = float(presence.sum() / roi_height)
        longest_run = max(end - start for start, end in present_runs)
        longest_run_ratio = longest_run / roi_height
        gaps = [
            (left_end, right_start)
            for (_left_start, left_end), (right_start, _right_end) in zip(
                present_runs,
                present_runs[1:],
                strict=False,
            )
        ]
    else:
        coverage_ratio = 0.0
        longest_run_ratio = 0.0
        gaps = []
    max_gap_px = max((end - start for start, end in gaps), default=0)
    row_count = max(1, int(selected_mask.any(axis=1).sum()))
    mean_width = float(selected_mask.sum() / row_count) if selected_mask.any() else 0.0
    if selected_mask.any():
        _ys, xs = np.nonzero(selected_mask)
        lateral_offset = float(xs.mean() - (roi.shape[1] - 1) / 2.0)
    else:
        lateral_offset = 0.0
    metrics = BrightStreakMetrics(
        contrast_snr=float(max(0.0, contrast_snr)),
        coverage_ratio=float(coverage_ratio),
        longest_run_ratio=float(longest_run_ratio),
        max_gap_px=int(max_gap_px),
        max_gap_ratio=float(max_gap_px / roi_height),
        gap_count=len(gaps),
        mean_width_px=mean_width,
        lateral_offset_px=lateral_offset,
        mean_intensity=float(roi.mean()),
        dark_clip_ratio=float((roi <= _DARK_CLIP_LEVEL).mean()),
        bright_clip_ratio=float((roi >= _BRIGHT_CLIP_LEVEL).mean()),
        laplacian_variance=float(cv2.Laplacian(roi, cv2.CV_64F).var()),
    )
    return metrics, present_runs, gaps


def _quality_metrics(roi: np.ndarray) -> BrightStreakMetrics:
    return BrightStreakMetrics(
        contrast_snr=0.0,
        coverage_ratio=0.0,
        longest_run_ratio=0.0,
        max_gap_px=0,
        max_gap_ratio=0.0,
        gap_count=0,
        mean_width_px=0.0,
        lateral_offset_px=0.0,
        mean_intensity=float(roi.mean()),
        dark_clip_ratio=float((roi <= _DARK_CLIP_LEVEL).mean()),
        bright_clip_ratio=float((roi >= _BRIGHT_CLIP_LEVEL).mean()),
        laplacian_variance=float(cv2.Laplacian(roi, cv2.CV_64F).var()),
    )


def _timed(
    metrics: BrightStreakMetrics,
    started: float,
    capture_elapsed_ms: float,
) -> BrightStreakMetrics:
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return replace(
        metrics,
        capture_elapsed_ms=capture_elapsed_ms,
        processing_elapsed_ms=elapsed_ms,
        total_elapsed_ms=capture_elapsed_ms + elapsed_ms,
    )


def _empty_decision(config: BrightStreakConfig, reason: str) -> BrightStreakDecision:
    placeholder = np.zeros((1, 1), dtype=np.uint8)
    return BrightStreakDecision(
        result=_empty_result(config, reason),
        roi=placeholder,
        response=placeholder,
        mask=placeholder,
        mask_used_for_metrics=placeholder.astype(bool),
        runs=(),
        gaps=(),
    )


def _decision(
    result: BrightStreakResult,
    *,
    roi: np.ndarray,
    response: np.ndarray,
    mask: np.ndarray,
    runs: list[tuple[int, int]],
    gaps: list[tuple[int, int]],
) -> BrightStreakDecision:
    return BrightStreakDecision(
        result=result,
        roi=roi,
        response=response,
        mask=mask,
        mask_used_for_metrics=mask.astype(bool),
        runs=tuple(runs),
        gaps=tuple(gaps),
    )


def detect_bright_streak_evidence(
    image: np.ndarray,
    config: BrightStreakConfig,
    *,
    capture_elapsed_ms: float = 0.0,
) -> BrightStreakDecision:
    """Classify one frame and return the exact response and mask used by the rule."""
    if (
        isinstance(capture_elapsed_ms, bool)
        or not isinstance(capture_elapsed_ms, (int, float))
        or not math.isfinite(float(capture_elapsed_ms))
        or capture_elapsed_ms < 0
    ):
        raise ValueError("capture_elapsed_ms must be a finite non-negative number")
    capture_elapsed_ms = float(capture_elapsed_ms)
    started = time.perf_counter()
    gray = _gray_image(image)
    if gray is None:
        return _empty_decision(config, "image must be a uint8 grayscale, BGR, or BGRA array")
    expected_shape = (config.image_height, config.image_width)
    if gray.shape != expected_shape:
        return _empty_decision(
            config,
            f"image dimensions {gray.shape[1]}x{gray.shape[0]} do not match "
            f"configured {config.image_width}x{config.image_height}",
        )
    try:
        x1, y1, x2, y2 = config.require_detection_roi()
    except ValueError as error:
        return _empty_decision(config, str(error))
    roi = np.ascontiguousarray(gray[y1:y2, x1:x2])
    if roi.size == 0:
        return _empty_decision(config, "configured bright-streak ROI is empty")

    quality = _quality_metrics(roi)
    quality_failures: list[str] = []
    if not config.min_mean_intensity <= quality.mean_intensity <= config.max_mean_intensity:
        quality_failures.append("ROI mean intensity outside configured range")
    if quality.dark_clip_ratio > config.max_dark_clip_ratio:
        quality_failures.append("ROI dark clipping exceeds configured limit")
    if quality.bright_clip_ratio > config.max_bright_clip_ratio:
        quality_failures.append("ROI bright clipping exceeds configured limit")
    if quality.laplacian_variance < config.min_laplacian_variance:
        quality_failures.append("ROI sharpness is below configured limit")
    if quality_failures:
        empty_mask = np.zeros_like(roi, dtype=np.uint8)
        return _decision(
            BrightStreakResult(
                DemoStatus.ERROR,
                "; ".join(quality_failures),
                (x1, y1, x2, y2),
                _timed(quality, started, capture_elapsed_ms),
            ),
            roi=roi,
            response=np.zeros_like(roi, dtype=np.uint8),
            mask=empty_mask,
            runs=[],
            gaps=[],
        )

    kernel_width = min(config.background_kernel_px, roi.shape[1] | 1)
    if kernel_width > roi.shape[1]:
        kernel_width -= 2
    kernel_width = max(1, kernel_width)
    background_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 3))
    response = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, background_kernel)
    response_median = float(np.median(response))
    response_mad = float(np.median(np.abs(response.astype(np.float32) - response_median)))
    core_threshold = response_median + config.response_mad_scale * max(1.0, response_mad)
    core_mask = (response.astype(np.float32) >= core_threshold).astype(np.uint8)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(core_mask, 8)
    center_x = (roi.shape[1] - 1) / 2.0
    candidates: list[int] = []
    for index in range(1, count):
        _x, _y, width, height, area = (int(value) for value in stats[index])
        component_center = float(centroids[index, 0])
        if area < config.min_component_area_px:
            continue
        if not config.min_component_width_px <= width <= config.max_component_width_px:
            continue
        if abs(component_center - center_x) > config.center_tolerance_px:
            continue
        if height < width:
            continue
        candidates.append(index)

    selected_mask = np.zeros_like(core_mask, dtype=bool)
    contrast_snr = 0.0
    saturated_mask = roi >= _BRIGHT_CLIP_LEVEL
    saturated_presence = saturated_mask.any(axis=1)
    saturated_row_support = float(saturated_presence.mean())
    saturated_row_count = int(saturated_presence.sum())
    saturated_mean_width = (
        float(saturated_mask.sum() / saturated_row_count) if saturated_row_count else 0.0
    )
    if saturated_mask.any():
        _saturated_ys, saturated_xs = np.nonzero(saturated_mask)
        saturated_offset = float(saturated_xs.mean() - center_x)
    else:
        saturated_offset = 0.0
    saturated_candidate = (
        saturated_row_support >= config.min_coverage_ratio
        and config.min_component_width_px <= saturated_mean_width <= config.max_component_width_px
        and abs(saturated_offset) <= config.center_tolerance_px
    )
    if saturated_candidate:
        selected_mask = cv2.morphologyEx(
            saturated_mask.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (1, max(1, 2 * config.micro_gap_close_px + 1)),
            ),
        ).astype(bool)
        raw_median = float(np.median(roi))
        raw_mad = float(np.median(np.abs(roi.astype(np.float32) - raw_median)))
        contrast_snr = float(
            max(0.0, (float(roi[saturated_mask].mean()) - raw_median) / max(1.0, raw_mad))
        )
    elif candidates:
        anchor = max(candidates, key=lambda index: int(stats[index, cv2.CC_STAT_AREA]))
        anchor_center = float(centroids[anchor, 0])
        alignment_tolerance = max(1.0, config.max_component_width_px / 2.0)
        aligned = [
            index
            for index in candidates
            if abs(float(centroids[index, 0]) - anchor_center) <= alignment_tolerance
        ]
        selected_mask = np.isin(labels, np.asarray(aligned, dtype=labels.dtype))
        raw_median = float(np.median(roi))
        raw_mad = float(np.median(np.abs(roi.astype(np.float32) - raw_median)))
        anchor_values = response[labels == anchor].astype(np.float32)
        contrast_snr = float(
            max(0.0, (float(anchor_values.mean()) - response_median) / max(1.0, raw_mad))
        )

    mask = selected_mask.astype(np.uint8) * 255
    metrics, runs, gaps = _metrics(
        roi=roi,
        selected_mask=selected_mask,
        contrast_snr=contrast_snr,
        micro_gap_close_px=config.micro_gap_close_px,
    )
    metrics = _timed(metrics, started, capture_elapsed_ms)
    if metrics.contrast_snr < config.min_contrast_snr or metrics.coverage_ratio < config.min_coverage_ratio:
        return _decision(
            BrightStreakResult(
                DemoStatus.NG_NO_STREAK,
                "bright streak contrast or coverage is below the configured minimum",
                (x1, y1, x2, y2),
                metrics,
            ),
            roi=roi,
            response=response,
            mask=mask,
            runs=runs,
            gaps=gaps,
        )
    if (
        metrics.longest_run_ratio < config.min_longest_run_ratio
        or metrics.max_gap_ratio > config.max_gap_ratio
        or metrics.gap_count > config.max_gap_count
    ):
        return _decision(
            BrightStreakResult(
                DemoStatus.NG_BROKEN,
                "bright streak is present but continuity is outside configured limits",
                (x1, y1, x2, y2),
                metrics,
            ),
            roi=roi,
            response=response,
            mask=mask,
            runs=runs,
            gaps=gaps,
        )
    return _decision(
        BrightStreakResult(
            DemoStatus.OK,
            "bright streak is present and continuous",
            (x1, y1, x2, y2),
            metrics,
        ),
        roi=roi,
        response=response,
        mask=mask,
        runs=runs,
        gaps=gaps,
    )


def detect_bright_streak(
    image: np.ndarray,
    config: BrightStreakConfig,
    *,
    capture_elapsed_ms: float = 0.0,
) -> BrightStreakResult:
    """Classify one frame using only its pixels and the validated rule config."""
    return detect_bright_streak_evidence(
        image,
        config,
        capture_elapsed_ms=capture_elapsed_ms,
    ).result


def render_evidence(image: np.ndarray, result: BrightStreakResult | BrightStreakDecision) -> np.ndarray:
    """Render a deterministic ROI, accepted-streak, and continuity overlay."""
    decision = result if isinstance(result, BrightStreakDecision) else None
    if decision is not None:
        result = decision.result
    gray = _gray_image(image)
    if gray is None:
        raise ValueError("image must be a uint8 grayscale, BGR, or BGRA array")
    if image.ndim == 2:
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        canvas = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        canvas = image.copy()
    x1, y1, x2, y2 = result.roi_xyxy
    if not (0 <= x1 < x2 <= canvas.shape[1] and 0 <= y1 < y2 <= canvas.shape[0]):
        return canvas
    status_color = {
        DemoStatus.OK: (0, 200, 0),
        DemoStatus.NG_NO_STREAK: (0, 0, 255),
        DemoStatus.NG_BROKEN: (0, 140, 255),
        DemoStatus.ERROR: (255, 0, 255),
    }[result.status]
    cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), status_color, 3)
    if result.metrics is not None:
        roi_center = (x2 - x1 - 1) / 2.0
        center = int(round(x1 + roi_center + result.metrics.lateral_offset_px))
        half_width = max(1, int(round(result.metrics.mean_width_px / 2.0)))
        runs = decision.runs if decision is not None else ()
        gaps = decision.gaps if decision is not None else ()
        for start, end in runs:
            cv2.rectangle(
                canvas,
                (max(x1, center - half_width), y1 + start),
                (min(x2 - 1, center + half_width), y1 + max(start, end - 1)),
                (0, 255, 0),
                1,
            )
        for start, end in gaps:
            cv2.rectangle(
                canvas,
                (x1, y1 + start),
                (x2 - 1, y1 + max(start, end - 1)),
                (0, 0, 255),
                1,
            )
    cv2.putText(
        canvas,
        result.status.value,
        (x1, max(24, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2,
        cv2.LINE_AA,
    )
    return canvas
