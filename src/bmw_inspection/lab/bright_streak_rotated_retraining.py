"""Fast laboratory retraining for the rotated tracked-profile bright streak."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2

from bmw_inspection.lab.bright_streak_rotated_roi import (
    load_rotated_bright_streak_roi,
    rectify_bright_streak_roi,
)


ROTATED_HDR_FIELD_WEAK_ROW_SCORE = 95.0
from bmw_inspection.lab.bright_streak_tracked_profile import (
    TRACKED_PROFILE_ROI_SHAPE,
    TrackedProfileGeometry,
    TrackedProfileThresholds,
    analyze_tracked_profile,
    classify_tracked_profile,
    fit_tracked_profile_thresholds,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gray_roi(path: Path, roi: Any) -> Any:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot decode bright-streak image: {path}")
    rectified = rectify_bright_streak_roi(image, roi)
    return cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY) if rectified.ndim == 3 else rectified


def retrain_rotated_bright_streak(
    normal_manifest: Path,
    no_streak_image: Path,
    rotated_roi_path: Path,
    rotated_roi_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Fit one candidate from current normals and one current no-streak sample."""
    manifest = Path(normal_manifest).expanduser().resolve()
    no_streak = Path(no_streak_image).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"bright-streak candidate already exists: {output}")
    roi = load_rotated_bright_streak_roi(
        Path(rotated_roi_path).expanduser().resolve(),
        expected_sha256=rotated_roi_sha256,
    )
    if not no_streak.is_file():
        raise ValueError(f"no-streak image does not exist: {no_streak}")
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"session_id", "sample_id", "source_class", "source_path"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("normal bright-streak manifest is invalid")
        normal_rows = [row for row in reader if row["source_class"] == "normal"]
    if not normal_rows:
        raise ValueError("normal bright-streak manifest contains no normal rows")

    geometry = TrackedProfileGeometry(candidate_width=5)
    provisional = TrackedProfileThresholds(
        strong_row_score=0.0,
        weak_row_score=0.0,
        min_presence_coverage_ratio=0.0,
        min_longest_run_ratio=0.0,
        max_gap_ratio=1.0,
        max_gap_count=TRACKED_PROFILE_ROI_SHAPE[0],
    )
    records: list[dict[str, Any]] = []
    for row in normal_rows:
        path = Path(row["source_path"]).expanduser().resolve()
        metrics = analyze_tracked_profile(_gray_roi(path, roi), geometry, provisional)
        records.append(
            {
                "sample_key": f"{row['session_id']}::{row['sample_id']}",
                "label": "normal",
                "source_path": str(path),
                "source_sha256": _sha256(path),
                "path_scores": metrics.path_scores,
                "split": "calibration",
                "final_test": False,
            }
        )
    no_streak_metrics = analyze_tracked_profile(_gray_roi(no_streak, roi), geometry, provisional)
    records.append(
        {
            "sample_key": f"20260814_094431_343851::{no_streak.stem}",
            "label": "no_streak",
            "source_path": str(no_streak),
            "source_sha256": _sha256(no_streak),
            "path_scores": no_streak_metrics.path_scores,
            "split": "calibration",
            "final_test": False,
        }
    )
    presence_fit = fit_tracked_profile_thresholds(records, geometry)
    if ROTATED_HDR_FIELD_WEAK_ROW_SCORE > presence_fit.strong_row_score:
        raise ValueError("field weak-row score exceeds the newly fitted strong-row score")
    continuity_probe = TrackedProfileThresholds(
        strong_row_score=presence_fit.strong_row_score,
        weak_row_score=ROTATED_HDR_FIELD_WEAK_ROW_SCORE,
        min_presence_coverage_ratio=0.0,
        min_longest_run_ratio=0.0,
        max_gap_ratio=1.0,
        max_gap_count=TRACKED_PROFILE_ROI_SHAPE[0],
    )
    normal_probe_metrics = [
        analyze_tracked_profile(
            _gray_roi(Path(record["source_path"]), roi), geometry, continuity_probe
        )
        for record in records
        if record["label"] == "normal"
    ]
    thresholds = TrackedProfileThresholds(
        strong_row_score=presence_fit.strong_row_score,
        weak_row_score=ROTATED_HDR_FIELD_WEAK_ROW_SCORE,
        min_presence_coverage_ratio=min(item.coverage_ratio for item in normal_probe_metrics),
        min_longest_run_ratio=min(item.longest_run_ratio for item in normal_probe_metrics),
        max_gap_ratio=max(item.max_gap_ratio for item in normal_probe_metrics),
        max_gap_count=max(item.gap_count for item in normal_probe_metrics),
    )

    outcomes: list[dict[str, Any]] = []
    for record in records:
        metrics = analyze_tracked_profile(
            _gray_roi(Path(record["source_path"]), roi), geometry, thresholds
        )
        predicted = classify_tracked_profile(metrics, thresholds)
        expected = "OK" if record["label"] == "normal" else "NG_NO_STREAK"
        outcomes.append(
            {
                "sample_key": record["sample_key"],
                "label": record["label"],
                "expected_status": expected,
                "predicted_status": predicted,
                "correct": predicted == expected,
                "source_path": record["source_path"],
                "source_sha256": record["source_sha256"],
                "peak_score": float(record["path_scores"].max()),
                "coverage_ratio": metrics.coverage_ratio,
                "longest_run_ratio": metrics.longest_run_ratio,
                "max_gap_ratio": metrics.max_gap_ratio,
                "gap_count": metrics.gap_count,
            }
        )

    output.mkdir(parents=True)
    metrics_path = output / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(outcomes[0]))
        writer.writeheader()
        writer.writerows(outcomes)
    report: dict[str, Any] = {
        "schema": "bmw.bright_streak_rotated_v3_candidate/1.0",
        "status": "complete",
        "algorithm": "tracked_profile_v3_manual_rotated_roi",
        "candidate_only": True,
        "normal_count": len(normal_rows),
        "no_streak_count": 1,
        "no_streak_independent_test_count": 0,
        "fit_data": "all_current_normals_plus_one_current_no_streak",
        "presence_fit_weak_row_score": presence_fit.weak_row_score,
        "weak_row_score_policy": "reuse_rotated_hdr_field_recalibration_v1",
        "geometry": asdict(geometry),
        "thresholds": asdict(thresholds),
        "normal_false_rejects": sum(
            row["label"] == "normal" and not row["correct"] for row in outcomes
        ),
        "no_streak_false_accepts": sum(
            row["label"] == "no_streak" and not row["correct"] for row in outcomes
        ),
        "normal_manifest": str(manifest),
        "normal_manifest_sha256": _sha256(manifest),
        "no_streak_image": str(no_streak),
        "no_streak_image_sha256": _sha256(no_streak),
        "rotated_roi": str(Path(rotated_roi_path).expanduser().resolve()),
        "rotated_roi_sha256": rotated_roi_sha256,
        "metrics_csv": str(metrics_path),
    }
    report_path = output / "report.json"
    report["report_json"] = str(report_path)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = ["ROTATED_HDR_FIELD_WEAK_ROW_SCORE", "retrain_rotated_bright_streak"]
