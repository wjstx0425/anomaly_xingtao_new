"""Small deterministic helpers for masked Template offline A/B reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER
from bmw_inspection.lab.template import fit_risk_threshold


def fit_masked_thresholds(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """Fit one threshold per view from calibration rows only."""
    thresholds: dict[str, float] = {}
    reports: dict[str, dict[str, Any]] = {}
    for view in VIEW_ORDER:
        calibration = [
            row for row in records if row.get("view_id") == view and row.get("split") == "calibration"
        ]
        fit = fit_risk_threshold(
            [(str(row["label"]), float(row["masked_risk"])) for row in calibration]
        )
        thresholds[view] = fit.threshold
        reports[view] = {
            "calibration_row_count": len(calibration),
            "threshold": fit.threshold,
            "balanced_accuracy": fit.balanced_accuracy,
            "normal_false_rejects": fit.normal_false_rejects,
            "defect_false_accepts": fit.defect_false_accepts,
        }
    return thresholds, reports


def summarize_decisions(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count old/masked NG and both decision-flip directions."""
    rows = tuple(records)
    return {
        "row_count": len(rows),
        "old_ng_count": sum(row.get("old_status") == "NG" for row in rows),
        "masked_ng_count": sum(row.get("masked_status") == "NG" for row in rows),
        "ng_to_pass_count": sum(
            row.get("old_status") == "NG" and row.get("masked_status") == "PASS" for row in rows
        ),
        "pass_to_ng_count": sum(
            row.get("old_status") == "PASS" and row.get("masked_status") == "NG" for row in rows
        ),
    }
