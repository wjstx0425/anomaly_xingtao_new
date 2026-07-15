# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: EM102, TRY003

"""Precision-first per-view thresholds for YOLO auxiliary strong evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from capture_data.zs32_inspection_orchestrator import CANONICAL_VIEWS

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

SCHEMA = "anomalib.zs32_yolo_auxiliary_thresholds"
SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class AuxiliaryRow:
    """One validated per-view YOLO image score and bbox-presence label."""

    part_id: str
    physical_part_id: str
    hand: str
    view: str
    score: float
    label: int
    split: str
    model_version: str
    roi_version: str


def _read_rows(path: Path) -> tuple[AuxiliaryRow, ...]:
    """Read and validate Stage33 YOLO annotation rows."""
    with path.open(encoding="utf-8-sig", newline="") as file:
        raw_rows = list(csv.DictReader(file))
    if not raw_rows:
        raise ValueError(f"YOLO auxiliary calibration rows are empty: {path}")
    rows = []
    identities = set()
    for row_number, raw in enumerate(raw_rows, start=2):
        required = {
            field: str(raw.get(field, "")).strip()
            for field in (
                "part_id",
                "hand",
                "view",
                "branch",
                "raw_score",
                "gt_label",
                "split",
                "model_version",
                "roi_version",
            )
        }
        if any(not value for value in required.values()):
            raise ValueError(f"{path} row {row_number} has an empty required field")
        view = required["view"]
        suffix = f"::{view}"
        if view not in CANONICAL_VIEWS or not required["part_id"].endswith(suffix):
            raise ValueError(f"{path} row {row_number} has inconsistent part/view identity")
        if required["branch"] != "yolo" or required["hand"] != "right":
            raise ValueError(f"{path} row {row_number} is not a right-hand YOLO score")
        if required["gt_label"] not in {"0", "1"} or required["split"] not in {"calibration", "test"}:
            raise ValueError(f"{path} row {row_number} has an invalid label or split")
        try:
            score = float(required["raw_score"])
        except ValueError as exc:
            raise ValueError(f"{path} row {row_number} has a non-numeric score") from exc
        if not math.isfinite(score) or score < 0:
            raise ValueError(f"{path} row {row_number} score must be finite and nonnegative")
        identity = (required["part_id"], required["split"])
        if identity in identities:
            raise ValueError(f"duplicate YOLO auxiliary row identity: {identity}")
        identities.add(identity)
        rows.append(
            AuxiliaryRow(
                part_id=required["part_id"],
                physical_part_id=required["part_id"][: -len(suffix)],
                hand=required["hand"],
                view=view,
                score=score,
                label=int(required["gt_label"]),
                split=required["split"],
                model_version=required["model_version"],
                roi_version=required["roi_version"],
            ),
        )
    validated_rows = tuple(rows)
    _validate_part_contract(validated_rows)
    return validated_rows


def _validate_part_contract(rows: Sequence[AuxiliaryRow]) -> None:
    """Require split-isolated physical parts with one row for every canonical view."""
    grouped: dict[str, list[AuxiliaryRow]] = defaultdict(list)
    for row in rows:
        grouped[row.physical_part_id].append(row)
    expected_views = set(CANONICAL_VIEWS)
    for physical_part_id, part_rows in grouped.items():
        splits = {row.split for row in part_rows}
        if len(splits) != 1:
            raise ValueError(f"physical part crosses calibration/test splits: {physical_part_id}")
        views = [row.view for row in part_rows]
        if len(views) != len(CANONICAL_VIEWS) or set(views) != expected_views:
            raise ValueError(
                f"physical part does not contain exactly {len(CANONICAL_VIEWS)} canonical views: {physical_part_id}",
            )
        versions = {(row.hand, row.model_version, row.roi_version) for row in part_rows}
        if len(versions) != 1:
            raise ValueError(f"physical part has inconsistent runtime versions: {physical_part_id}")


def _metrics(rows: Sequence[AuxiliaryRow], threshold: float) -> dict[str, int | float | None]:
    """Compute deterministic binary image metrics for one inclusive threshold."""
    tp = sum(row.label == 1 and row.score >= threshold for row in rows)
    fp = sum(row.label == 0 and row.score >= threshold for row in rows)
    fn = sum(row.label == 1 and row.score < threshold for row in rows)
    tn = sum(row.label == 0 and row.score < threshold for row in rows)
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "positive_count": tp + fn,
        "negative_count": fp + tn,
        "predicted_positive_count": tp + fp,
        "image_presence_precision": tp / (tp + fp) if tp + fp else None,
        "image_presence_recall": tp / (tp + fn) if tp + fn else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
    }


def select_high_precision_threshold(
    rows: Sequence[AuxiliaryRow],
    minimum_image_precision: float,
) -> tuple[float | None, dict[str, int | float | None] | None, str]:
    """Select the highest-recall threshold satisfying fit image-presence precision."""
    if not 0 < minimum_image_precision <= 1:
        message = "minimum_image_precision must be within (0, 1]"
        raise ValueError(message)
    if not any(row.label == 1 for row in rows) or not any(row.label == 0 for row in rows):
        return None, None, "insufficient_data"
    feasible = []
    for threshold in sorted({row.score for row in rows if row.score > 0}):
        metrics = _metrics(rows, threshold)
        precision = metrics["image_presence_precision"]
        recall = metrics["image_presence_recall"]
        if precision is not None and recall is not None and precision >= minimum_image_precision:
            feasible.append((recall, precision, -threshold, threshold, metrics))
    if not feasible:
        return None, None, "no_feasible_threshold"
    _, _, _, threshold, metrics = max(feasible, key=lambda item: item[:3])
    return threshold, metrics, "ok"


def _validate_view_contract(rows: Sequence[AuxiliaryRow], view: str) -> tuple[tuple[AuxiliaryRow, ...], ...]:
    """Return fit/test rows after exact version and split validation."""
    view_rows = tuple(row for row in rows if row.view == view)
    versions = {(row.hand, row.model_version, row.roi_version) for row in view_rows}
    if len(versions) != 1:
        raise ValueError(f"YOLO auxiliary view {view!r} has inconsistent runtime versions")
    fit = tuple(row for row in view_rows if row.split == "calibration")
    test = tuple(row for row in view_rows if row.split == "test")
    if not fit or not test:
        raise ValueError(f"YOLO auxiliary view {view!r} requires calibration and test rows")
    return fit, test


def _part_metrics(
    rows: Sequence[AuxiliaryRow],
    thresholds: Mapping[str, float],
    split: str,
) -> dict[str, int | float | None]:
    """Evaluate canonical-view OR evidence at physical-part level."""
    grouped: dict[str, list[AuxiliaryRow]] = defaultdict(list)
    for row in rows:
        if row.split == split:
            grouped[row.physical_part_id].append(row)
    tp = fp = fn = tn = 0
    positive_parts_triggered_on_labeled_view = 0
    positive_parts_triggered_only_on_unlabeled_view = 0
    for part_rows in grouped.values():
        label = int(any(row.label == 1 for row in part_rows))
        prediction = int(any(row.score >= thresholds[row.view] for row in part_rows))
        gt_aligned_trigger = any(row.label == 1 and row.score >= thresholds[row.view] for row in part_rows)
        tp += label == 1 and prediction == 1
        fp += label == 0 and prediction == 1
        fn += label == 1 and prediction == 0
        tn += label == 0 and prediction == 0
        positive_parts_triggered_on_labeled_view += label == 1 and gt_aligned_trigger
        positive_parts_triggered_only_on_unlabeled_view += label == 1 and prediction == 1 and not gt_aligned_trigger
    positive_count = tp + fn
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "positive_part_count": positive_count,
        "negative_part_count": fp + tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "positive_parts_triggered_on_labeled_view": positive_parts_triggered_on_labeled_view,
        "positive_parts_triggered_only_on_unlabeled_view": positive_parts_triggered_only_on_unlabeled_view,
        "gt_aligned_trigger_recall": (
            positive_parts_triggered_on_labeled_view / positive_count if positive_count else None
        ),
    }


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_yolo_auxiliary_calibration(
    input_csv: Path,
    output_dir: Path,
    *,
    minimum_fit_image_precision: float,
    use_test_for_selection: bool = False,
) -> dict[str, Any]:
    """Publish precision-first thresholds without altering recall-first artifacts.

    ``use_test_for_selection`` is an explicitly leaked commissioning escape hatch.
    Its artifacts are marked as non-independent and must remain opt-in downstream.
    """
    if output_dir.exists():
        raise FileExistsError(f"YOLO auxiliary output already exists: {output_dir}")
    rows = _read_rows(input_csv)
    records = []
    selected_thresholds = {}
    for view in CANONICAL_VIEWS:
        fit, test = _validate_view_contract(rows, view)
        selection_rows = fit + test if use_test_for_selection else fit
        threshold, fit_metrics, status = select_high_precision_threshold(
            selection_rows,
            minimum_fit_image_precision,
        )
        test_metrics = None if threshold is None else _metrics(test, threshold)
        warnings = []
        if sum(row.label == 1 for row in fit) < 5:
            warnings.append("fit_positive_count_below_5")
        if sum(row.label == 1 for row in test) < 5:
            warnings.append("test_positive_count_below_5")
        if threshold is not None:
            selected_thresholds[view] = threshold
        records.append(
            {
                "hand": fit[0].hand,
                "view": view,
                "branch": "yolo",
                "model_version": fit[0].model_version,
                "roi_version": fit[0].roi_version,
                "threshold": threshold,
                "comparison": "score >= threshold",
                "proposed_low_threshold": threshold,
                "proposed_high_threshold": threshold,
                "status": status,
                "fit": fit_metrics,
                "test": test_metrics,
                "warnings": warnings,
            },
        )
    all_views_have_candidate = len(selected_thresholds) == len(CANONICAL_VIEWS)
    part_metrics = (
        {split: _part_metrics(rows, selected_thresholds, split) for split in ("calibration", "test")}
        if all_views_have_candidate
        else None
    )
    threshold_payload = {
        "schema": SCHEMA,
        "version": SCHEMA_VERSION,
        "profile": "high_precision_auxiliary",
        "commissioning_only": True,
        "runtime_injection_supported": False,
        "decision_rule": "score >= threshold emits STRONG YOLO evidence; lower scores emit CLEAR",
        "fit_split": "calibration",
        "evaluation_split": "test",
        "test_used_for_selection": False,
        "minimum_fit_image_presence_precision": minimum_fit_image_precision,
        "all_views_have_candidate": all_views_have_candidate,
        "thresholds": records,
    }
    if use_test_for_selection:
        threshold_payload.update(
            {
                "fit_split": "calibration+test",
                "evaluation_split": "test_reused_for_selection",
                "test_used_for_selection": True,
                "data_leakage": True,
                "leakage_notice": "TEST DATA WAS USED FOR THRESHOLD SELECTION; METRICS ARE NOT HELD-OUT.",
            },
        )
    threshold_payload["threshold_records_sha256"] = _canonical_sha256(records)
    summary = {
        "schema": "anomalib.zs32_yolo_auxiliary_summary",
        "version": SCHEMA_VERSION,
        "commissioning_only": True,
        "runtime_injection_supported": False,
        "minimum_fit_image_presence_precision": minimum_fit_image_precision,
        "threshold_selection_used_test": use_test_for_selection,
        "all_views_have_candidate": all_views_have_candidate,
        "views": {record["view"]: record for record in records},
        "six_view_or_part_metrics": part_metrics,
        "limitations": [
            "Per-view positive calibration counts are small; observed precision is not a confidence bound.",
            "Image-level max confidence does not prove that a predicted box overlaps ground truth.",
            "A future locked profile would use this as direct-NG YOLO strong evidence, not cross-model confirmation.",
            "This commissioning artifact is not a complete locked Stage31 threshold bundle.",
        ],
    }
    if use_test_for_selection:
        summary.update(
            {
                "data_leakage": True,
                "leakage_notice": "TEST DATA WAS USED FOR THRESHOLD SELECTION; METRICS ARE NOT HELD-OUT.",
            },
        )
    staging = output_dir.with_name(f".{output_dir.name}.tmp")
    staging.mkdir(parents=True)
    try:
        (staging / "thresholds.json").write_text(
            json.dumps(threshold_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output_dir)
    except Exception:
        for path in staging.glob("*"):
            path.unlink(missing_ok=True)
        staging.rmdir()
        raise
    return summary
