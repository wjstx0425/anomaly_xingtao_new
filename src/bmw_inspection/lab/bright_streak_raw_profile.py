"""Offline raw-grayscale profile features for the BMW front-left bright streak."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from bmw_inspection.contracts import DemoStatus, load_config
from bmw_inspection.detector import detect_bright_streak_evidence


RAW_PROFILE_ROI_SHAPE = (613, 81)


@dataclass(frozen=True, slots=True)
class RawProfileMetrics:
    """Explainable one-dimensional evidence extracted from one fixed bright-streak ROI."""

    row_scores: np.ndarray
    mask: np.ndarray
    coverage_ratio: float
    longest_run_px: int
    longest_run_ratio: float
    max_gap_px: int
    max_gap_ratio: float
    gap_count: int


@dataclass(frozen=True, slots=True)
class RawProfileThresholds:
    """Calibration-only acceptance thresholds for raw-profile evidence."""

    min_row_score: float
    min_presence_coverage_ratio: float
    min_longest_run_ratio: float
    max_gap_ratio: float
    max_gap_count: int


def _smoothed(values: np.ndarray, window: int) -> np.ndarray:
    if window == 1:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, window), axis=-1)


def _runs(mask: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return foreground runs and only the gaps bounded by foreground runs."""
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


def _metrics_from_row_scores(row_scores: np.ndarray, min_row_score: float) -> RawProfileMetrics:
    if row_scores.ndim != 1 or row_scores.shape != (RAW_PROFILE_ROI_SHAPE[0],):
        raise ValueError("row_scores must have shape (613,)")
    mask = row_scores >= min_row_score
    runs, gaps = _runs(mask)
    height = len(mask)
    longest_run_px = max((stop - start for start, stop in runs), default=0)
    max_gap_px = max((stop - start for start, stop in gaps), default=0)
    return RawProfileMetrics(
        row_scores=row_scores,
        mask=mask,
        coverage_ratio=float(mask.mean()),
        longest_run_px=longest_run_px,
        longest_run_ratio=longest_run_px / height,
        max_gap_px=max_gap_px,
        max_gap_ratio=max_gap_px / height,
        gap_count=len(gaps),
    )


def analyze_raw_profile(
    image: np.ndarray,
    *,
    min_row_score: float,
    candidate_width: int = 9,
    background_width: int = 24,
    background_gap: int = 4,
    smooth_window: int = 5,
) -> RawProfileMetrics:
    """Score a fixed 81x613 grayscale ROI against its left and right backgrounds.

    The candidate band is centred in the ROI.  The two equal-width background
    bands are separated from it by ``background_gap`` pixels, so the local
    response remains insensitive to left-right illumination drift.

    Args:
        image: Grayscale front-left ROI with shape ``(613, 81)``.
        min_row_score: Local contrast threshold used to form the row mask.
        candidate_width: Width in pixels of the centre candidate band.
        background_width: Width in pixels of each side background band.
        background_gap: Pixels excluded between candidate and each background.
        smooth_window: Odd moving-average window applied to row scores.

    Returns:
        Row scores, binary evidence mask, and coverage/continuity statistics.

    Raises:
        TypeError: If the ROI is not a NumPy array or threshold is not numeric.
        ValueError: If the ROI/band geometry is invalid.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")
    if isinstance(min_row_score, bool) or not isinstance(min_row_score, (int, float)):
        raise TypeError("min_row_score must be numeric")
    if image.ndim != 2 or image.shape != RAW_PROFILE_ROI_SHAPE:
        raise ValueError("raw bright-streak profile requires a grayscale ROI with shape (613, 81)")
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (candidate_width, background_width, smooth_window)
        )
        or isinstance(background_gap, bool)
        or not isinstance(background_gap, int)
        or background_gap < 0
        or smooth_window % 2 == 0
    ):
        raise ValueError(
            "profile bands need positive widths, an odd smooth_window, "
            "and a non-negative background_gap"
        )

    _, width = image.shape
    candidate_start = (width - candidate_width) // 2
    candidate_stop = candidate_start + candidate_width
    left_stop = candidate_start - background_gap
    left_start = left_stop - background_width
    right_start = candidate_stop + background_gap
    right_stop = right_start + background_width
    if left_start < 0 or right_stop > width:
        raise ValueError("profile bands do not fit inside the fixed 81-pixel ROI")

    grayscale = image.astype(np.float64, copy=False)
    centre = grayscale[:, candidate_start:candidate_stop].mean(axis=1)
    left = grayscale[:, left_start:left_stop].mean(axis=1)
    right = grayscale[:, right_start:right_stop].mean(axis=1)
    row_scores = _smoothed(centre - (left + right) / 2.0, smooth_window)
    return _metrics_from_row_scores(row_scores, float(min_row_score))


def _threshold_candidates(values: np.ndarray) -> np.ndarray:
    unique = np.unique(values.astype(np.float64, copy=False))
    if len(unique) > 256:
        unique = np.unique(np.quantile(unique, np.linspace(0.0, 1.0, 256)))
    mids = (unique[:-1] + unique[1:]) / 2.0
    return np.unique(np.concatenate((np.array([0.0]), unique, mids)))


def _classify_metrics(metrics: RawProfileMetrics, thresholds: RawProfileThresholds) -> str:
    if metrics.coverage_ratio < thresholds.min_presence_coverage_ratio:
        return DemoStatus.NG_NO_STREAK.value
    if (
        metrics.longest_run_ratio < thresholds.min_longest_run_ratio
        or metrics.max_gap_ratio > thresholds.max_gap_ratio
        or metrics.gap_count > thresholds.max_gap_count
    ):
        return DemoStatus.NG_BROKEN.value
    return DemoStatus.OK.value


def classify_raw_profile(metrics: RawProfileMetrics, thresholds: RawProfileThresholds) -> str:
    """Convert raw-profile evidence into the established BMW bright-streak status."""
    if not isinstance(metrics, RawProfileMetrics):
        raise TypeError("metrics must be RawProfileMetrics")
    if not isinstance(thresholds, RawProfileThresholds):
        raise TypeError("thresholds must be RawProfileThresholds")
    return _classify_metrics(metrics, thresholds)


def fit_raw_profile_thresholds(records: Sequence[Mapping[str, object]]) -> RawProfileThresholds:
    """Fit raw-profile thresholds from calibration rows only.

    Each row must provide ``label`` (``normal`` or ``no_streak``) and a
    613-value ``row_scores`` array.  The caller controls split selection, so
    passing final-test rows is intentionally not possible through the
    evaluation entrypoint.
    """
    normal: list[np.ndarray] = []
    missing: list[np.ndarray] = []
    for record in records:
        label = record.get("label")
        scores = record.get("row_scores")
        if label not in {"normal", "no_streak"}:
            raise ValueError("raw-profile calibration labels must be normal or no_streak")
        if not isinstance(scores, np.ndarray):
            raise TypeError("raw-profile calibration row_scores must be numpy arrays")
        if scores.ndim != 1 or scores.shape != (RAW_PROFILE_ROI_SHAPE[0],):
            raise ValueError("raw-profile calibration row_scores must have shape (613,)")
        (normal if label == "normal" else missing).append(scores.astype(np.float64, copy=False))
    if not normal or not missing:
        raise ValueError("raw-profile calibration needs both normal and no_streak samples")

    best: tuple[float, float, int, int, int, float, float, float, float, int] | None = None
    score_values = np.concatenate([*normal, *missing])
    for min_row_score in _threshold_candidates(score_values):
        normal_metrics = [
            _metrics_from_row_scores(scores, float(min_row_score)) for scores in normal
        ]
        missing_metrics = [
            _metrics_from_row_scores(scores, float(min_row_score)) for scores in missing
        ]
        min_longest = min(item.longest_run_ratio for item in normal_metrics)
        max_gap = max(item.max_gap_ratio for item in normal_metrics)
        max_gap_count = max(item.gap_count for item in normal_metrics)
        coverage_values = np.array(
            [item.coverage_ratio for item in [*normal_metrics, *missing_metrics]],
            dtype=np.float64,
        )
        for min_coverage in _threshold_candidates(coverage_values):
            thresholds = RawProfileThresholds(
                min_row_score=float(min_row_score),
                min_presence_coverage_ratio=float(min_coverage),
                min_longest_run_ratio=min_longest,
                max_gap_ratio=max_gap,
                max_gap_count=max_gap_count,
            )
            normal_statuses = [_classify_metrics(item, thresholds) for item in normal_metrics]
            missing_statuses = [_classify_metrics(item, thresholds) for item in missing_metrics]
            false_rejects = sum(status != DemoStatus.OK.value for status in normal_statuses)
            false_accepts = sum(status == DemoStatus.OK.value for status in missing_statuses)
            missing_status_errors = sum(
                status != DemoStatus.NG_NO_STREAK.value for status in missing_statuses
            )
            balanced_accuracy = (
                (1.0 - false_rejects / len(normal))
                + (1.0 - false_accepts / len(missing))
            ) / 2.0
            candidate = (
                sum(status == DemoStatus.OK.value for status in normal_statuses)
                + sum(status == DemoStatus.NG_NO_STREAK.value for status in missing_statuses),
                balanced_accuracy,
                -false_rejects,
                -false_accepts,
                -missing_status_errors,
                float(min_row_score),
                -float(min_coverage),
                min_longest,
                max_gap,
                max_gap_count,
            )
            if best is None or candidate > best:
                best = candidate
    assert best is not None
    _, _, _, _, _, min_row_score, min_coverage_negated, min_longest, max_gap, max_gap_count = best
    return RawProfileThresholds(
        min_row_score=min_row_score,
        min_presence_coverage_ratio=-min_coverage_negated,
        min_longest_run_ratio=min_longest,
        max_gap_ratio=max_gap,
        max_gap_count=max_gap_count,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _profile_roi(image: np.ndarray, roi_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("bright-streak source image must be grayscale, BGR, or BGRA")
    x1, y1, x2, y2 = roi_xyxy
    roi = gray[y1:y2, x1:x2]
    if roi.shape != RAW_PROFILE_ROI_SHAPE:
        raise ValueError("bright-streak config ROI must remain 81x613 for raw-profile v2")
    return roi


def _accuracy(outcomes: Sequence[Mapping[str, object]]) -> float:
    if not outcomes:
        return 0.0
    return (
        sum(row["predicted_status"] == row["expected_status"] for row in outcomes)
        / len(outcomes)
    )


def evaluate_bright_streak_raw_profile(
    manifest_path: Path,
    base_config_path: Path,
    output_dir: Path,
    *,
    roi_xyxy: tuple[int, int, int, int] | None = None,
) -> dict[str, object]:
    """Fit v2 only on calibration and compare it to the current detector on final test.

    The function is deliberately offline: it publishes only a new report
    directory and never changes the current detector or Demo configuration.
    """
    manifest_path = Path(manifest_path).expanduser().resolve()
    base_config_path = Path(base_config_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if not manifest_path.is_file():
        raise ValueError(f"bright-streak manifest does not exist: {manifest_path}")
    if not base_config_path.is_file():
        raise ValueError(f"bright-streak base config does not exist: {base_config_path}")
    base_config = load_config(base_config_path)
    base_roi_xyxy = base_config.require_detection_roi()
    resolved_roi_xyxy = base_roi_xyxy if roi_xyxy is None else tuple(roi_xyxy)
    if (
        len(resolved_roi_xyxy) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) for value in resolved_roi_xyxy)
        or (resolved_roi_xyxy[3] - resolved_roi_xyxy[1], resolved_roi_xyxy[2] - resolved_roi_xyxy[0])
        != RAW_PROFILE_ROI_SHAPE
    ):
        raise ValueError("bright-streak config ROI must remain 81x613 for raw-profile v2")

    records: list[dict[str, object]] = []
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"sample_id", "split", "expected_status", "source_path"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("bright-streak manifest is missing required fields")
        for row_number, row in enumerate(reader, start=2):
            if row["split"] not in {"calibration", "final_test"}:
                continue
            if row["expected_status"] not in {DemoStatus.OK.value, DemoStatus.NG_NO_STREAK.value}:
                continue
            source_path = Path(row["source_path"]).expanduser()
            if not source_path.is_absolute():
                source_path = manifest_path.parent / source_path
            source_path = source_path.resolve()
            image = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"cannot decode bright-streak image: {source_path}")
            roi = _profile_roi(image, resolved_roi_xyxy)
            raw_scores = analyze_raw_profile(roi, min_row_score=0.0).row_scores
            current = detect_bright_streak_evidence(image, base_config)
            records.append(
                {
                    "row_number": row_number,
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "expected_status": row["expected_status"],
                    "label": (
                        "normal"
                        if row["expected_status"] == DemoStatus.OK.value
                        else "no_streak"
                    ),
                    "source_path": str(source_path),
                    "row_scores": raw_scores,
                    "current_status": current.status.value,
                }
            )
    if not records:
        raise ValueError("bright-streak manifest has no calibration or final_test decision rows")
    calibration = [row for row in records if row["split"] == "calibration"]
    final_test = [row for row in records if row["split"] == "final_test"]
    thresholds = fit_raw_profile_thresholds(calibration)

    output_dir.mkdir(parents=True)
    metrics_path = output_dir / "metrics.csv"
    report_path = output_dir / "report.json"
    profiles_dir = output_dir / "profiles"
    profiles_dir.mkdir()
    raw_outcomes: list[dict[str, object]] = []
    current_outcomes: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for record in records:
        metrics = _metrics_from_row_scores(
            record["row_scores"],  # type: ignore[arg-type]
            thresholds.min_row_score,
        )
        profile_path = profiles_dir / f"profile_{int(record['row_number']):04d}.npz"
        np.savez_compressed(profile_path, row_scores=metrics.row_scores, mask=metrics.mask)
        raw_status = classify_raw_profile(metrics, thresholds)
        outcome = {
            "row_number": record["row_number"],
            "sample_id": record["sample_id"],
            "expected_status": record["expected_status"],
            "predicted_status": raw_status,
            "correct": raw_status == record["expected_status"],
        }
        current_outcome = {
            "row_number": record["row_number"],
            "sample_id": record["sample_id"],
            "expected_status": record["expected_status"],
            "predicted_status": record["current_status"],
            "correct": record["current_status"] == record["expected_status"],
        }
        if record["split"] == "final_test":
            raw_outcomes.append(outcome)
            current_outcomes.append(current_outcome)
        metric_rows.append(
            {
                **{
                    key: record[key]
                    for key in (
                        "row_number",
                        "sample_id",
                        "split",
                        "expected_status",
                        "source_path",
                    )
                },
                "current_predicted_status": record["current_status"],
                "raw_profile_predicted_status": raw_status,
                "raw_profile_npz": str(profile_path),
                "row_score_max": float(np.max(metrics.row_scores)),
                "coverage_ratio": metrics.coverage_ratio,
                "longest_run_px": metrics.longest_run_px,
                "longest_run_ratio": metrics.longest_run_ratio,
                "max_gap_px": metrics.max_gap_px,
                "max_gap_ratio": metrics.max_gap_ratio,
                "gap_count": metrics.gap_count,
            }
        )
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    report: dict[str, object] = {
        "status": "complete",
        "fit_split": "calibration",
        "final_test_used_for_fit": False,
        "manifest": str(manifest_path),
        "base_config": str(base_config_path),
        "base_roi_xyxy": list(base_roi_xyxy),
        "roi_xyxy": list(resolved_roi_xyxy),
        "metrics_csv": str(metrics_path),
        "profiles_dir": str(profiles_dir),
        "report_json": str(report_path),
        "calibration_count": len(calibration),
        "final_test_count": len(final_test),
        "raw_profile_v2": {
            "thresholds": {
                "min_row_score": thresholds.min_row_score,
                "min_presence_coverage_ratio": thresholds.min_presence_coverage_ratio,
                "min_longest_run_ratio": thresholds.min_longest_run_ratio,
                "max_gap_ratio": thresholds.max_gap_ratio,
                "max_gap_count": thresholds.max_gap_count,
            },
            "final_test_count": len(raw_outcomes),
            "final_test_accuracy": _accuracy(raw_outcomes),
            "outcomes": raw_outcomes,
        },
        "current_algorithm": {
            "final_test_count": len(current_outcomes),
            "final_test_accuracy": _accuracy(current_outcomes),
            "outcomes": current_outcomes,
        },
        "identities": {
            "manifest_sha256": _sha256(manifest_path),
            "base_config_sha256": _sha256(base_config_path),
            "raw_profile_source_sha256": _sha256(Path(__file__).resolve()),
        },
    }
    _write_json(report_path, report)
    return report


__all__ = [
    "RAW_PROFILE_ROI_SHAPE",
    "RawProfileMetrics",
    "RawProfileThresholds",
    "analyze_raw_profile",
    "classify_raw_profile",
    "evaluate_bright_streak_raw_profile",
    "fit_raw_profile_thresholds",
]
