from __future__ import annotations

import pytest

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER
from bmw_inspection.lab.template_mask_ab import fit_masked_thresholds, summarize_decisions


def _records() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for view in VIEW_ORDER:
        rows.extend(
            (
                {"view_id": view, "split": "calibration", "label": "normal", "masked_risk": 0.1},
                {"view_id": view, "split": "calibration", "label": "defect", "masked_risk": 0.9},
                {"view_id": view, "split": "final_test", "label": "normal", "masked_risk": 100.0},
            )
        )
    return rows


def test_fit_masked_thresholds_uses_only_calibration() -> None:
    thresholds, reports = fit_masked_thresholds(_records())

    assert tuple(thresholds) == VIEW_ORDER
    assert thresholds == pytest.approx({view: 0.1 for view in VIEW_ORDER})
    assert all(report["calibration_row_count"] == 2 for report in reports.values())


def test_summarize_decisions_counts_both_flip_directions() -> None:
    summary = summarize_decisions(
        [
            {"old_status": "NG", "masked_status": "PASS"},
            {"old_status": "PASS", "masked_status": "NG"},
            {"old_status": "NG", "masked_status": "NG"},
        ]
    )

    assert summary == {
        "row_count": 3,
        "old_ng_count": 2,
        "masked_ng_count": 2,
        "ng_to_pass_count": 1,
        "pass_to_ng_count": 1,
    }
