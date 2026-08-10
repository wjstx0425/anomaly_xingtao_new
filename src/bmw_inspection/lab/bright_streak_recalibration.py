"""Standalone real-data recalibration for the BMW bright-streak detector."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Mapping

import cv2

from bmw_inspection.contracts import DemoStatus, load_config, read_json_object
from bmw_inspection.detector import detect_bright_streak_evidence, render_evidence
from bmw_inspection.lab.eight_view_train_all import (
    _balanced_accuracy,
    _fit_bright_streak_thresholds,
    _is_continuous,
)


_FITTED_THRESHOLD_NAMES = (
    "min_contrast_snr",
    "min_coverage_ratio",
    "min_longest_run_ratio",
    "max_gap_ratio",
    "max_gap_count",
)
_METRIC_FIELDS = (
    "row_number",
    "sample_id",
    "split",
    "expected_status",
    "predicted_status",
    "correct",
    "source_path",
    "contrast_snr",
    "coverage_ratio",
    "longest_run_ratio",
    "max_gap_ratio",
    "gap_count",
    "continuous",
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


def _manifest_records(manifest_path: Path, base_config_path: Path) -> list[dict[str, Any]]:
    base = load_config(base_config_path)
    records: list[dict[str, Any]] = []
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"sample_id", "split", "expected_status", "source_path"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("bright-streak manifest is missing required fields")
        for row_number, row in enumerate(reader, start=2):
            expected = row["expected_status"]
            split = row["split"]
            if expected not in {DemoStatus.OK.value, DemoStatus.NG_NO_STREAK.value}:
                continue
            if split not in {"calibration", "final_test"}:
                continue
            source_path = Path(row["source_path"]).expanduser()
            if not source_path.is_absolute():
                source_path = manifest_path.parent / source_path
            source_path = source_path.resolve()
            image = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"cannot decode bright-streak image: {source_path}")
            decision = detect_bright_streak_evidence(image, base)
            metrics = decision.metrics
            if metrics is None:
                raise ValueError(f"bright-streak metrics are unavailable: {source_path}")
            records.append(
                {
                    "row_number": row_number,
                    "sample_id": row["sample_id"],
                    "split": split,
                    "expected_status": expected,
                    "source_path": str(source_path),
                    "label": "normal" if expected == DemoStatus.OK.value else "no_streak",
                    "contrast_snr": metrics.contrast_snr,
                    "coverage_ratio": metrics.coverage_ratio,
                    "longest_run_ratio": metrics.longest_run_ratio,
                    "max_gap_ratio": metrics.max_gap_ratio,
                    "gap_count": metrics.gap_count,
                }
            )
    if not records:
        raise ValueError("bright-streak manifest has no calibration or final_test decision rows")
    return records


def recalibrate_bright_streak(
    manifest_path: Path,
    base_config_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Fit on calibration rows and report untouched final-test outcomes.

    Args:
        manifest_path (Path): Prepared ``bright_streak.csv`` containing explicit split and expected-status fields.
        base_config_path (Path): Loadable BMW bright-streak JSON whose ROI and detector settings are retained.
        output_dir (Path): New directory for the calibrated config, metrics, report, and final-test evidence.

    Returns:
        dict[str, object]: Published artifact paths, identities, fitted thresholds, and final-test outcomes.

    Raises:
        FileExistsError: If ``output_dir`` already exists.
        OSError: If an artifact or evidence image cannot be written.
        ValueError: If inputs, manifest rows, images, metrics, or class coverage are invalid.
    """
    manifest_path = Path(manifest_path).expanduser().resolve()
    base_config_path = Path(base_config_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError(f"bright-streak manifest does not exist: {manifest_path}")
    if not base_config_path.is_file():
        raise ValueError(f"bright-streak base config does not exist: {base_config_path}")

    records = _manifest_records(manifest_path, base_config_path)
    calibration = [row for row in records if row["split"] == "calibration"]
    final_test = [row for row in records if row["split"] == "final_test"]
    fitted = _fit_bright_streak_thresholds(calibration)
    final_balanced, final_false_rejects, final_false_accepts = _balanced_accuracy(
        final_test,
        fitted["min_contrast_snr"],
        fitted["min_coverage_ratio"],
        fitted["min_longest_run_ratio"],
        fitted["max_gap_ratio"],
        fitted["max_gap_count"],
    )

    output_dir.mkdir(parents=True)
    calibrated_path = output_dir / "calibrated_config.json"
    metrics_path = output_dir / "metrics.csv"
    report_path = output_dir / "report.json"
    evidence_dir = output_dir / "final_test_errors"
    evidence_dir.mkdir()

    payload = read_json_object(base_config_path)
    thresholds = dict(payload["thresholds"])
    for name in _FITTED_THRESHOLD_NAMES:
        thresholds[name] = fitted[name]
    payload["thresholds"] = thresholds
    payload["mode"] = "demo"
    payload["result_root"] = str(output_dir / "runtime")
    _write_json(calibrated_path, payload)
    calibrated = load_config(calibrated_path)

    final_outcomes: list[dict[str, object]] = []
    final_errors: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    error_index = 0
    for record in records:
        source_path = Path(record["source_path"])
        image = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"cannot decode bright-streak image: {source_path}")
        decision = detect_bright_streak_evidence(image, calibrated)
        predicted_status = decision.status.value
        correct = predicted_status == record["expected_status"]
        metric_rows.append(
            {
                "row_number": record["row_number"],
                "sample_id": record["sample_id"],
                "split": record["split"],
                "expected_status": record["expected_status"],
                "predicted_status": predicted_status,
                "correct": correct,
                "source_path": record["source_path"],
                "contrast_snr": record["contrast_snr"],
                "coverage_ratio": record["coverage_ratio"],
                "longest_run_ratio": record["longest_run_ratio"],
                "max_gap_ratio": record["max_gap_ratio"],
                "gap_count": record["gap_count"],
                "continuous": _is_continuous(
                    record,
                    fitted["min_longest_run_ratio"],
                    fitted["max_gap_ratio"],
                    fitted["max_gap_count"],
                ),
            }
        )
        if record["split"] != "final_test":
            continue
        outcome: dict[str, object] = {
            "row_number": record["row_number"],
            "sample_id": record["sample_id"],
            "source_path": record["source_path"],
            "expected_status": record["expected_status"],
            "predicted_status": predicted_status,
            "correct": correct,
        }
        if not correct:
            error_index += 1
            evidence_path = evidence_dir / f"final_test_error_{error_index:04d}.png"
            if not cv2.imwrite(str(evidence_path), render_evidence(image, decision)):
                raise OSError(f"cannot write bright-streak evidence: {evidence_path}")
            outcome["evidence_path"] = str(evidence_path)
            final_errors.append(outcome)
        final_outcomes.append(outcome)

    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(metric_rows)

    detector_source = Path(inspect.getsourcefile(detect_bright_streak_evidence) or "").resolve()
    report: dict[str, object] = {
        "status": "complete",
        "fit_split": "calibration",
        "final_test_used_for_fit": False,
        "manifest": str(manifest_path),
        "base_config": str(base_config_path),
        "config": str(calibrated_path),
        "metrics_csv": str(metrics_path),
        "report_json": str(report_path),
        "evidence_dir": str(evidence_dir),
        "calibration_count": len(calibration),
        "final_test_count": len(final_test),
        "fitted_thresholds": {name: fitted[name] for name in _FITTED_THRESHOLD_NAMES},
        "calibration_balanced_accuracy": fitted["balanced_accuracy"],
        "calibration_false_rejects": fitted["false_rejects"],
        "calibration_false_accepts": fitted["false_accepts"],
        "final_test_balanced_accuracy": final_balanced,
        "final_test_false_rejects": final_false_rejects,
        "final_test_false_accepts": final_false_accepts,
        "final_test_outcomes": final_outcomes,
        "final_test_errors": final_errors,
        "identities": {
            "manifest_sha256": _sha256(manifest_path),
            "base_config_sha256": _sha256(base_config_path),
            "calibrated_config_sha256": _sha256(calibrated_path),
            "detector_source": str(detector_source),
            "detector_sha256": _sha256(detector_source),
        },
    }
    _write_json(report_path, report)
    return report


__all__ = ["recalibrate_bright_streak"]
