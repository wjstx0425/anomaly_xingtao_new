# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for grouped ZS32 fusion threshold calibration."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from capture_data.fusion_calibration import (
    fit_dual_thresholds,
    part_level_metrics,
    run_calibration,
)


def _row(
    part_id: str,
    *,
    view: str = "front",
    raw_score: float,
    gt_label: int,
    split: str = "calibration",
    hand: str = "left",
    branch: str = "anomaly",
    model_version: str = "model-v1",
    roi_version: str = "roi-v1",
) -> dict[str, str]:
    """Build one calibration row using the public CSV schema."""
    return {
        "part_id": part_id,
        "hand": hand,
        "view": view,
        "branch": branch,
        "raw_score": str(raw_score),
        "gt_label": str(gt_label),
        "split": split,
        "model_version": model_version,
        "roi_version": roi_version,
    }


def test_rejects_physical_part_split_leakage_across_views() -> None:
    """All rows belonging to one physical part must remain in one split."""
    rows = [
        _row("part-001", view="front", raw_score=0.1, gt_label=0, split="calibration"),
        _row("part-001", view="back", raw_score=0.2, gt_label=0, split="test"),
    ]

    with pytest.raises(ValueError, match=r"part-001.*split"):
        fit_dual_thresholds(rows, target_recall=1.0, normal_quantile=0.995)


def test_thresholds_group_by_full_versioned_branch_key() -> None:
    """Hand, view, branch, model version, and ROI version define independent thresholds."""
    rows = [
        _row("normal-a", raw_score=0.1, gt_label=0),
        _row("defect-a", raw_score=0.6, gt_label=1),
        _row("normal-b", raw_score=0.2, gt_label=0, roi_version="roi-v2"),
        _row("defect-b", raw_score=0.8, gt_label=1, roi_version="roi-v2"),
    ]

    thresholds = fit_dual_thresholds(rows, target_recall=1.0, normal_quantile=1.0)

    assert [(record.roi_version, record.low_threshold, record.high_threshold) for record in thresholds] == [
        ("roi-v1", 0.6, 0.6),
        ("roi-v2", 0.8, 0.8),
    ]
    assert all(record.low_threshold <= record.high_threshold for record in thresholds)


def test_target_recall_uses_highest_threshold_that_keeps_required_defects_non_clear() -> None:
    """Low threshold should maximize CLEAR range without violating observed target recall."""
    rows = [
        _row("normal", raw_score=0.1, gt_label=0),
        *[_row(f"defect-{index}", raw_score=score, gt_label=1) for index, score in enumerate((0.2, 0.4, 0.6, 0.8))],
    ]

    full_recall = fit_dual_thresholds(rows, target_recall=1.0, normal_quantile=1.0)[0]
    three_quarter_recall = fit_dual_thresholds(rows, target_recall=0.75, normal_quantile=1.0)[0]

    assert full_recall.low_threshold == pytest.approx(0.2)
    assert three_quarter_recall.low_threshold == pytest.approx(0.4)


def test_required_view_without_both_classes_is_explicitly_insufficient() -> None:
    """Required groups must never receive fabricated deployment thresholds."""
    rows = [
        _row("normal-front", raw_score=0.1, gt_label=0),
        _row("defect-front", raw_score=0.8, gt_label=1),
    ]

    thresholds = fit_dual_thresholds(
        rows,
        target_recall=1.0,
        normal_quantile=0.995,
        required_groups=(("left", "back", "anomaly", "model-v1", "roi-v1"),),
    )

    missing = next(record for record in thresholds if record.view == "back")
    assert missing.status == "insufficient_data"
    assert missing.low_threshold is None
    assert missing.high_threshold is None
    assert missing.normal_count == 0
    assert missing.defect_count == 0


def test_defect_escape_is_counted_once_per_physical_part_not_once_per_view() -> None:
    """Six CLEAR defect rows from one part represent one escaped physical part."""
    views = ("front", "front_left", "front_right", "back", "back_left", "back_right")
    rows = [
        *[_row("normal", view=view, raw_score=0.1, gt_label=0, split="test") for view in views],
        *[_row("escaped-defect", view=view, raw_score=0.2, gt_label=1, split="test") for view in views],
    ]
    thresholds = [
        {
            "hand": "left",
            "view": view,
            "branch": "anomaly",
            "model_version": "model-v1",
            "roi_version": "roi-v1",
            "low_threshold": 0.3,
            "high_threshold": 0.7,
            "normal_count": 1,
            "defect_count": 1,
            "status": "ok",
        }
        for view in views
    ]

    metrics = part_level_metrics(rows, thresholds)

    assert metrics["overall"]["defect_part_count"] == 1
    assert metrics["overall"]["escape_count"] == 1
    assert metrics["overall"]["escape_rate"] == pytest.approx(1.0)
    assert metrics["overall"]["recall"] == pytest.approx(0.0)


def test_zero_escape_reports_exact_one_sided_95_percent_upper_bound() -> None:
    """A zero count still needs its finite-sample binomial upper bound."""
    rows = [
        _row("normal", raw_score=0.1, gt_label=0),
        _row("defect-1", raw_score=0.6, gt_label=1),
        _row("defect-2", raw_score=0.7, gt_label=1),
        _row("defect-3", raw_score=0.8, gt_label=1),
    ]
    thresholds = fit_dual_thresholds(rows, target_recall=1.0, normal_quantile=1.0)

    eval_rows = [
        _row("test-defect-1", raw_score=0.6, gt_label=1, split="test"),
        _row("test-defect-2", raw_score=0.7, gt_label=1, split="test"),
        _row("test-defect-3", raw_score=0.8, gt_label=1, split="test"),
    ]
    overall = part_level_metrics(eval_rows, thresholds)["overall"]

    assert overall["escape_count"] == 0
    assert overall["escape_rate_95_upper"] == pytest.approx(1 - 0.05 ** (1 / 3))


def test_defect_in_gray_band_is_not_mistaken_for_clear_escape() -> None:
    """GRAY evidence catches a defect even when it is not a strong rejection."""
    rows = [_row("defect", raw_score=0.5, gt_label=1, split="test")]
    thresholds = [
        {
            "hand": "left",
            "view": "front",
            "branch": "anomaly",
            "model_version": "model-v1",
            "roi_version": "roi-v1",
            "low_threshold": 0.4,
            "high_threshold": 0.8,
            "normal_count": 1,
            "defect_count": 1,
            "status": "ok",
        },
    ]

    overall = part_level_metrics(rows, thresholds)["overall"]

    assert overall["escape_count"] == 0
    assert overall["review_count"] == 1


@pytest.mark.parametrize(
    ("low_threshold", "high_threshold", "status"),
    [
        (0.8, 0.2, "ok"),
        (float("nan"), 0.8, "ok"),
        (0.2, 0.8, "unexpected"),
    ],
)
def test_part_metrics_reject_invalid_external_thresholds(
    low_threshold: float,
    high_threshold: float,
    status: str,
) -> None:
    """Serialized thresholds must obey the same validity contract as the fusion engine."""
    rows = [_row("defect", raw_score=0.5, gt_label=1, split="test")]
    thresholds = [
        {
            "hand": "left",
            "view": "front",
            "branch": "anomaly",
            "model_version": "model-v1",
            "roi_version": "roi-v1",
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "normal_count": 1,
            "defect_count": 1,
            "status": status,
        },
    ]

    with pytest.raises(ValueError, match="invalid threshold"):
        part_level_metrics(rows, thresholds)


def test_run_calibration_atomically_writes_all_reports(tmp_path: Path) -> None:
    """The offline entrypoint should publish the four immutable report artifacts."""
    input_csv = tmp_path / "calibration.csv"
    fieldnames = list(_row("part", raw_score=0.1, gt_label=0))
    with input_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [
                _row("normal", raw_score=0.1, gt_label=0),
                _row("defect", raw_score=0.8, gt_label=1),
                _row("test-normal", raw_score=0.1, gt_label=0, split="test"),
                _row("test-defect", raw_score=0.8, gt_label=1, split="test"),
            ],
        )

    output_dir = tmp_path / "reports"
    run_calibration(
        input_csv,
        output_dir,
        target_recall=1.0,
        normal_quantile=0.995,
        required_views=("front",),
    )

    assert {path.name for path in output_dir.iterdir()} == {
        "thresholds.json",
        "thresholds.csv",
        "calibration_metrics.json",
        "calibration_summary.md",
    }
    payload = json.loads((output_dir / "thresholds.json").read_text(encoding="utf-8"))
    assert payload["thresholds"][0]["low_threshold"] == pytest.approx(0.8)
    assert payload["required_groups"] == [["left", "front", "anomaly", "model-v1", "roi-v1"]]
    summary = (output_dir / "calibration_summary.md").read_text(encoding="utf-8")
    assert "100% recall" in summary
    assert "production escapes" in summary


def test_threshold_report_binds_exact_stage18_deployment_contract(tmp_path: Path) -> None:
    """Stage 31 artifacts must identify the strict profile and immutable threshold records."""
    input_csv = tmp_path / "calibration.csv"
    fieldnames = list(_row("part", raw_score=0.1, gt_label=0))
    with input_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [
                _row("normal", raw_score=0.1, gt_label=0),
                _row("defect", raw_score=0.8, gt_label=1),
                _row("test-normal", raw_score=0.1, gt_label=0, split="test"),
                _row("test-defect", raw_score=0.8, gt_label=1, split="test"),
            ],
        )
    expected_versions = [
        {
            "hand": "left",
            "side": "zs32",
            "view": "front",
            "branch": "anomaly",
            "model_version": "model-v1",
            "threshold_version": "threshold-v7",
            "roi_version": "roi-v1",
            "template_version": "template-v3",
        },
    ]
    deployment_contract = {
        "product": "ZS32",
        "profile": "zs32_six_view_v1",
        "allowed_hands": ["left", "right"],
        "required_side": "zs32",
        "config_sha256": "a" * 64,
        "expected_versions": expected_versions,
    }

    run_calibration(
        input_csv,
        tmp_path / "reports",
        required_views=("front",),
        deployment_contract=deployment_contract,
    )

    payload = json.loads((tmp_path / "reports/thresholds.json").read_text(encoding="utf-8"))
    assert payload["deployment_contract"] == deployment_contract
    assert payload["threshold_versions"] == ["threshold-v7"]
    canonical = json.dumps(payload["thresholds"], separators=(",", ":"), sort_keys=True).encode()
    assert payload["threshold_records_sha256"] == hashlib.sha256(canonical).hexdigest()
    assert payload["artifact_schema"] == "anomalib.zs32_fusion_thresholds"
    assert payload["artifact_version"] == "1.0"
    assert payload["calibration_valid"] is True
    assert payload["profile_sha256"] == deployment_contract["config_sha256"]
    artifact_hash = payload.pop("artifact_sha256")
    canonical_artifact = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    assert artifact_hash == hashlib.sha256(canonical_artifact).hexdigest()


def test_fit_uses_only_selected_split_without_test_leakage() -> None:
    """Held-out defect scores must not influence fitted deployment thresholds."""
    rows = [
        _row("cal-normal", raw_score=0.1, gt_label=0),
        _row("cal-defect", raw_score=0.2, gt_label=1),
        _row("test-normal", raw_score=0.1, gt_label=0, split="test"),
        _row("test-defect", raw_score=0.9, gt_label=1, split="test"),
    ]

    threshold = fit_dual_thresholds(
        rows,
        target_recall=1.0,
        normal_quantile=0.995,
        fit_split="calibration",
    )[0]

    assert threshold.low_threshold == pytest.approx(0.2)


def test_fit_and_metrics_fail_when_selected_split_is_missing() -> None:
    """A misspelled or absent split must fail closed instead of falling back to all rows."""
    rows = [
        _row("normal", raw_score=0.1, gt_label=0),
        _row("defect", raw_score=0.8, gt_label=1),
    ]
    thresholds = fit_dual_thresholds(rows, target_recall=1.0, normal_quantile=0.995)

    with pytest.raises(ValueError, match=r"fit split.*missing"):
        fit_dual_thresholds(rows, target_recall=1.0, normal_quantile=0.995, fit_split="missing")
    with pytest.raises(ValueError, match=r"evaluation split.*missing"):
        part_level_metrics(rows, thresholds, eval_split="missing")


def test_exact_required_groups_do_not_cross_product_view_specific_branches() -> None:
    """Exact group contracts must preserve heterogeneous view-specific branches and versions."""
    rows = [
        _row("front-normal", raw_score=0.1, gt_label=0, branch="anomaly_front"),
        _row("front-defect", raw_score=0.8, gt_label=1, branch="anomaly_front"),
        _row("back-normal", view="back", raw_score=0.1, gt_label=0, branch="anomaly_back", model_version="back-v2"),
        _row("back-defect", view="back", raw_score=0.8, gt_label=1, branch="anomaly_back", model_version="back-v2"),
    ]
    required_groups = (
        ("left", "front", "anomaly_front", "model-v1", "roi-v1"),
        ("left", "back", "anomaly_back", "back-v2", "roi-v1"),
    )

    thresholds = fit_dual_thresholds(
        rows,
        target_recall=1.0,
        normal_quantile=0.995,
        required_views=("front", "back"),
        required_groups=required_groups,
    )

    assert [
        (record.hand, record.view, record.branch, record.model_version, record.roi_version) for record in thresholds
    ] == sorted(required_groups)


def test_required_view_is_only_an_existence_gate() -> None:
    """A required view should fail closed when no fit row provides it."""
    rows = [
        _row("normal", raw_score=0.1, gt_label=0),
        _row("defect", raw_score=0.8, gt_label=1),
    ]

    with pytest.raises(ValueError, match=r"required view.*back"):
        fit_dual_thresholds(
            rows,
            target_recall=1.0,
            normal_quantile=0.995,
            required_views=("front", "back"),
        )


def test_missing_required_eval_group_forces_review_and_invalidates_safety_metrics() -> None:
    """Absent required evidence must not produce apparently safe recall or confidence bounds."""
    required_groups = (
        ("left", "front", "anomaly_front", "model-v1", "roi-v1"),
        ("left", "back", "anomaly_back", "model-v1", "roi-v1"),
    )
    thresholds = [
        {
            "hand": hand,
            "view": view,
            "branch": branch,
            "model_version": model,
            "roi_version": roi,
            "low_threshold": 0.4,
            "high_threshold": 0.8,
            "normal_count": 2,
            "defect_count": 2,
            "status": "ok",
        }
        for hand, view, branch, model, roi in required_groups
    ]
    rows = [
        _row("test-defect", raw_score=0.1, gt_label=1, split="test", branch="anomaly_front"),
    ]

    overall = part_level_metrics(rows, thresholds, required_groups=required_groups)["overall"]

    assert overall["calibration_valid"] is False
    assert overall["review_count"] == 1
    assert overall["escape_rate"] is None
    assert overall["non_clear_recall"] is None
    assert overall["recall"] is None
    assert overall["normal_reject_rate"] is None
    assert overall["review_rate"] is None
    assert overall["escape_rate_95_upper"] is None
    assert overall["diagnostic_observed_escape_rate"] == pytest.approx(0.0)
    assert overall["diagnostic_observed_normal_reject_rate"] is None
    assert overall["diagnostic_observed_review_rate"] == pytest.approx(1.0)


def test_required_group_for_absent_evaluation_hand_invalidates_contract() -> None:
    """A required hand absent from the whole eval split must remain an explicit contract miss."""
    required_group = ("right", "front", "anomaly", "model-v1", "roi-v1")
    thresholds = [
        {
            "hand": "left",
            "view": "front",
            "branch": "anomaly",
            "model_version": "model-v1",
            "roi_version": "roi-v1",
            "low_threshold": 0.4,
            "high_threshold": 0.8,
            "normal_count": 2,
            "defect_count": 2,
            "status": "ok",
        },
        {
            "hand": required_group[0],
            "view": required_group[1],
            "branch": required_group[2],
            "model_version": required_group[3],
            "roi_version": required_group[4],
            "low_threshold": 0.4,
            "high_threshold": 0.8,
            "normal_count": 2,
            "defect_count": 2,
            "status": "ok",
        },
    ]
    rows = [_row("left-defect", raw_score=0.9, gt_label=1, split="test")]

    metrics = part_level_metrics(rows, thresholds, required_groups=(required_group,))

    assert metrics["overall"]["calibration_valid"] is False
    assert metrics["overall"]["escape_rate"] is None
    assert metrics["overall"]["recall"] is None
    assert metrics["overall"]["normal_reject_rate"] is None
    assert metrics["overall"]["review_rate"] is None
    assert metrics["overall"]["escape_rate_95_upper"] is None
    assert metrics["missing_required_groups"] == [list(required_group)]


def test_insufficient_required_threshold_invalidates_otherwise_complete_view(tmp_path: Path) -> None:
    """Another branch covering the same view must not hide an insufficient exact group."""
    input_csv = tmp_path / "calibration.csv"
    fieldnames = list(_row("part", raw_score=0.1, gt_label=0))
    required_groups = (
        ("left", "front", "primary", "model-v1", "roi-v1"),
        ("left", "front", "auxiliary", "model-v1", "roi-v1"),
    )
    rows = [
        _row("cal-normal", branch="primary", raw_score=0.1, gt_label=0),
        _row("cal-defect", branch="primary", raw_score=0.8, gt_label=1),
        _row("cal-normal", branch="auxiliary", raw_score=0.2, gt_label=0),
        _row("test-normal", branch="primary", raw_score=0.1, gt_label=0, split="test"),
        _row("test-defect", branch="primary", raw_score=0.8, gt_label=1, split="test"),
    ]
    with input_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    output_dir = tmp_path / "reports"
    metrics = run_calibration(
        input_csv,
        output_dir,
        required_views=("front",),
        required_groups=required_groups,
    )

    assert metrics["overall"]["calibration_valid"] is False
    assert metrics["overall"]["recall"] is None
    assert metrics["overall"]["escape_rate_95_upper"] is None
    assert metrics["invalid_threshold_groups"] == [list(required_groups[1])]
    summary = (output_dir / "calibration_summary.md").read_text(encoding="utf-8")
    assert "invalid threshold groups" in summary
    assert "auxiliary" in summary


@pytest.mark.parametrize(
    ("low", "high", "status"),
    [(float("nan"), 0.8, "ok"), (0.9, 0.8, "ok"), (0.4, 0.8, "unknown")],
)
def test_metrics_reject_invalid_external_threshold_records(low: float, high: float, status: str) -> None:
    """Serialized threshold input must preserve the fusion engine's finite ordered contract."""
    rows = [_row("test-defect", raw_score=0.5, gt_label=1, split="test")]
    threshold = {
        "hand": "left",
        "view": "front",
        "branch": "anomaly",
        "model_version": "model-v1",
        "roi_version": "roi-v1",
        "low_threshold": low,
        "high_threshold": high,
        "normal_count": 1,
        "defect_count": 1,
        "status": status,
    }

    with pytest.raises(ValueError, match="threshold"):
        part_level_metrics(rows, [threshold])


def test_atomic_replace_failure_does_not_publish_partial_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed replace must leave the corresponding final artifact unpublished."""
    input_csv = tmp_path / "calibration.csv"
    fieldnames = list(_row("part", raw_score=0.1, gt_label=0))
    with input_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [
                _row("normal", raw_score=0.1, gt_label=0),
                _row("defect", raw_score=0.8, gt_label=1),
                _row("test-normal", raw_score=0.1, gt_label=0, split="test"),
                _row("test-defect", raw_score=0.8, gt_label=1, split="test"),
            ],
        )

    def fail_replace(_self: Path, _target: Path) -> Path:
        msg = "injected replace failure"
        raise OSError(msg)

    monkeypatch.setattr(Path, "replace", fail_replace)
    output_dir = tmp_path / "reports"

    with pytest.raises(OSError, match="injected replace failure"):
        run_calibration(input_csv, output_dir)

    assert not output_dir.exists()


@pytest.mark.parametrize("failed_write", [2, 3])
def test_report_write_failure_never_publishes_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_write: int,
) -> None:
    """Failure while staging any report must leave no final or temporary report directory."""
    input_csv = tmp_path / "calibration.csv"
    fieldnames = list(_row("part", raw_score=0.1, gt_label=0))
    with input_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [
                _row("cal-normal", raw_score=0.1, gt_label=0),
                _row("cal-defect", raw_score=0.8, gt_label=1),
                _row("test-normal", raw_score=0.1, gt_label=0, split="test"),
                _row("test-defect", raw_score=0.8, gt_label=1, split="test"),
            ],
        )
    original_write_text = Path.write_text
    write_count = 0

    def fail_selected_write(path: Path, content: str, *, encoding: str | None = None) -> int:
        nonlocal write_count
        write_count += 1
        if write_count == failed_write:
            msg = "injected report write failure"
            raise OSError(msg)
        return original_write_text(path, content, encoding=encoding)

    monkeypatch.setattr(Path, "write_text", fail_selected_write)
    output_dir = tmp_path / "reports"

    with pytest.raises(OSError, match="injected report write failure"):
        run_calibration(input_csv, output_dir, required_views=("front",))

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".reports.tmp-*"))


def test_existing_output_directory_is_rejected_without_changes(tmp_path: Path) -> None:
    """Immutable report snapshots must never overwrite or mix with an existing directory."""
    input_csv = tmp_path / "calibration.csv"
    fieldnames = list(_row("part", raw_score=0.1, gt_label=0))
    with input_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [
                _row("cal-normal", raw_score=0.1, gt_label=0),
                _row("cal-defect", raw_score=0.8, gt_label=1),
                _row("test-normal", raw_score=0.1, gt_label=0, split="test"),
                _row("test-defect", raw_score=0.8, gt_label=1, split="test"),
            ],
        )
    output_dir = tmp_path / "reports"
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("unchanged", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        run_calibration(input_csv, output_dir, required_views=("front",))

    assert {path.name for path in output_dir.iterdir()} == {"keep.txt"}
    assert marker.read_text(encoding="utf-8") == "unchanged"


def test_run_calibration_defaults_to_all_six_zs32_views(tmp_path: Path) -> None:
    """The default ZS32 workflow must fail closed when an entire required view is absent."""
    input_csv = tmp_path / "calibration.csv"
    fieldnames = list(_row("part", raw_score=0.1, gt_label=0))
    with input_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [
                _row("cal-normal", raw_score=0.1, gt_label=0),
                _row("cal-defect", raw_score=0.8, gt_label=1),
                _row("test-normal", raw_score=0.1, gt_label=0, split="test"),
                _row("test-defect", raw_score=0.8, gt_label=1, split="test"),
            ],
        )

    metrics = run_calibration(input_csv, tmp_path / "reports")

    assert metrics["overall"]["calibration_valid"] is False
    assert "back_right" in metrics["missing_fit_views"]
    assert "back_right" in metrics["missing_evaluation_views"]
    assert metrics["overall"]["escape_rate"] is None
    assert metrics["overall"]["normal_reject_rate"] is None
    assert metrics["overall"]["review_rate"] is None
    assert metrics["overall"]["recall"] is None
    assert metrics["overall"]["escape_rate_95_upper"] is None
    assert metrics["overall"]["diagnostic_observed_escape_rate"] == pytest.approx(0.0)
    assert metrics["overall"]["diagnostic_observed_normal_reject_rate"] == pytest.approx(0.0)
    assert metrics["overall"]["diagnostic_observed_review_rate"] == pytest.approx(0.0)


@pytest.mark.parametrize("missing_from", ["calibration", "test"])
def test_run_calibration_reports_split_specific_missing_view(tmp_path: Path, missing_from: str) -> None:
    """Fit and evaluation view coverage must be checked independently."""
    input_csv = tmp_path / f"{missing_from}.csv"
    views = ("front", "front_left", "front_right", "back", "back_left", "back_right")
    fieldnames = list(_row("part", raw_score=0.1, gt_label=0))
    rows = []
    for split in ("calibration", "test"):
        for view in views:
            if split == missing_from and view == "back_right":
                continue
            rows.extend(
                [
                    _row(f"{split}-normal", view=view, raw_score=0.1, gt_label=0, split=split),
                    _row(f"{split}-defect", view=view, raw_score=0.8, gt_label=1, split=split),
                ],
            )
    with input_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metrics = run_calibration(input_csv, tmp_path / "reports")

    expected_fit = ["back_right"] if missing_from == "calibration" else []
    expected_evaluation = ["back_right"] if missing_from == "test" else []
    assert metrics["missing_fit_views"] == expected_fit
    assert metrics["missing_evaluation_views"] == expected_evaluation
    assert metrics["overall"]["calibration_valid"] is False
    assert metrics["overall"]["recall"] is None
    assert metrics["overall"]["escape_rate_95_upper"] is None
