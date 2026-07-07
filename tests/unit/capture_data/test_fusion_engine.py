# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for robust industrial inspection fusion helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_fusion_module() -> ModuleType:
    """Load the fusion helper from its file path."""
    script_path = Path(__file__).resolve().parents[3] / "capture_data" / "fusion_engine.py"
    spec = importlib.util.spec_from_file_location("capture_data_fusion_engine", script_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load fusion engine script from {script_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_missing_required_view_returns_invalid_capture() -> None:
    """Missing required side/view inputs should fail closed before branch fusion."""
    fusion = load_fusion_module()

    decision = fusion.fuse_part_predictions(
        "part001",
        [],
        missing_required=("top:uniform",),
    )

    assert decision.final_status == "INVALID_CAPTURE"
    assert decision.final_label is None
    assert "top:uniform" in decision.reason


def test_quality_failure_returns_retake_before_model_positive() -> None:
    """Quality failures should ask for a retake instead of forcing OK/NG inference."""
    fusion = load_fusion_module()
    predictions = [
        fusion.BranchPrediction(
            part_id="part001",
            side="top",
            view="uniform",
            slot_id=None,
            branch="quality",
            pred_label=1,
            score=None,
            threshold=None,
            defect_type=None,
            reason="blur_laplacian_var below minimum",
            source_path="part001_top_uniform.png",
        ),
        fusion.BranchPrediction(
            part_id="part001",
            side="top",
            view="uniform",
            slot_id="slot02",
            branch="geometry",
            pred_label=1,
            score=12.0,
            threshold=10.0,
            defect_type="less",
            reason=None,
            source_path="part001_top_uniform.png",
        ),
    ]

    decision = fusion.fuse_part_predictions("part001", predictions)

    assert decision.final_status == "RETAKE"
    assert decision.final_label is None
    assert decision.triggered_branch == "quality"
    assert "blur_laplacian_var" in decision.reason


def test_branch_order_maps_positive_predictions_to_ng_statuses() -> None:
    """The first positive branch in configured order should explain the NG status."""
    fusion = load_fusion_module()
    predictions = [
        fusion.BranchPrediction(
            part_id="part001",
            side="top",
            view="uniform",
            slot_id="slot02",
            branch="anomaly_dino",
            pred_label=1,
            score=0.72,
            threshold=0.5,
            defect_type="surface",
            reason=None,
            source_path="part001_top_uniform.png",
        ),
        fusion.BranchPrediction(
            part_id="part001",
            side="top",
            view="uniform",
            slot_id="slot02",
            branch="geometry",
            pred_label=1,
            score=12.0,
            threshold=10.0,
            defect_type="less",
            reason=None,
            source_path="part001_top_uniform.png",
        ),
    ]

    decision = fusion.fuse_part_predictions("part001", predictions)

    assert decision.final_status == "NG_GEOMETRY"
    assert decision.final_label == 1
    assert decision.triggered_branch == "geometry"
    assert decision.defect_slot == "slot02"
    assert decision.defect_type == "less"


def test_anomaly_positive_maps_to_ng_anomaly() -> None:
    """AnomalyDINO positives should produce NG_ANOMALY when geometry is clean or absent."""
    fusion = load_fusion_module()
    predictions = [
        fusion.BranchPrediction(
            part_id="part001",
            side="top",
            view="uniform",
            slot_id="slot05",
            branch="anomaly_dino",
            pred_label=1,
            score=0.72,
            threshold=0.5,
            defect_type="surface",
            reason=None,
            source_path="part001_top_uniform.png",
        ),
    ]

    decision = fusion.fuse_part_predictions("part001", predictions)

    assert decision.final_status == "NG_ANOMALY"
    assert decision.final_label == 1
    assert decision.triggered_branch == "anomaly_dino"


def test_ok_requires_quality_and_registration_pass_rows() -> None:
    """Configured OK prerequisites should fail closed when PASS rows are missing."""
    fusion = load_fusion_module()
    predictions = [
        fusion.BranchPrediction(
            part_id="part001",
            side="top",
            view="uniform",
            slot_id="slot01",
            branch="geometry",
            pred_label=0,
            score=1.0,
            threshold=10.0,
            defect_type="none",
            reason=None,
            source_path="part001_top_uniform.png",
        ),
    ]

    decision = fusion.fuse_part_predictions(
        "part001",
        predictions,
        config={"ok_requires": {"quality_gate": "PASS", "registration": "PASS"}},
    )

    assert decision.final_status == "RETAKE"
    assert decision.final_label is None
    assert decision.triggered_branch == "quality_gate"
    assert "missing required quality_gate PASS" in decision.reason


def test_ok_requires_rejects_warn_quality_status() -> None:
    """Configured PASS prerequisites should not treat WARN gate rows as OK."""
    fusion = load_fusion_module()
    predictions = [
        fusion.BranchPrediction(
            part_id="part001",
            side="top",
            view="uniform",
            slot_id=None,
            branch="quality_gate",
            pred_label=0,
            score=None,
            threshold=None,
            defect_type=None,
            reason="brightness near limit",
            source_path="part001_top_uniform.png",
            status="WARN",
        ),
        fusion.BranchPrediction(
            part_id="part001",
            side="top",
            view="uniform",
            slot_id="slot01",
            branch="geometry",
            pred_label=0,
            score=1.0,
            threshold=10.0,
            defect_type="none",
            reason=None,
            source_path="part001_top_uniform.png",
        ),
    ]

    decision = fusion.fuse_part_predictions(
        "part001",
        predictions,
        config={"ok_requires": {"quality_gate": "PASS"}},
    )

    assert decision.final_status == "RETAKE"
    assert decision.triggered_branch == "quality_gate"
    assert "required quality_gate PASS" in decision.reason


def test_required_view_config_is_checked_without_manifest() -> None:
    """Fusion config required side/view keys should work even without a manifest CSV."""
    fusion = load_fusion_module()
    predictions = [
        fusion.BranchPrediction(
            part_id="part001",
            side="top",
            view="uniform",
            slot_id="slot01",
            branch="geometry",
            pred_label=0,
            score=1.0,
            threshold=10.0,
            defect_type="none",
            reason=None,
            source_path="part001_top_uniform_slot01.png",
        ),
    ]

    missing = fusion.missing_required_views(
        predictions,
        {"ok_requires": {"required_sides": ["top", "bottom"], "required_views": ["uniform"]}},
    )
    decision = fusion.fuse_part_predictions("part001", predictions, missing_required=missing)

    assert missing == ["bottom:uniform"]
    assert decision.final_status == "INVALID_CAPTURE"


def test_config_scalar_values_are_not_split_into_characters() -> None:
    """Scalar config values should be treated as one value, not a character sequence."""
    fusion = load_fusion_module()
    predictions = [
        fusion.BranchPrediction(
            part_id="part001",
            side="top",
            view="uniform",
            slot_id="slot01",
            branch="geometry",
            pred_label=1,
            score=12.0,
            threshold=10.0,
            defect_type="less",
            reason=None,
            source_path="part001_top_uniform_slot01.png",
        ),
        fusion.BranchPrediction(
            part_id="part001",
            side="top",
            view="uniform",
            slot_id="slot01",
            branch="anomaly_dino",
            pred_label=1,
            score=0.9,
            threshold=0.5,
            defect_type="surface",
            reason=None,
            source_path="part001_top_uniform_slot01.png",
        ),
    ]

    missing = fusion.missing_required_views(
        predictions,
        {"ok_requires": {"required_sides": "top", "required_views": "uniform"}},
    )
    decision = fusion.fuse_part_predictions("part001", predictions, config={"branch_order": "anomaly_dino"})

    assert missing == []
    assert decision.final_status == "NG_ANOMALY"


def test_no_positive_predictions_return_ok() -> None:
    """Clean branch predictions should produce a strict OK decision."""
    fusion = load_fusion_module()
    predictions = [
        fusion.BranchPrediction(
            part_id="part001",
            side="top",
            view="uniform",
            slot_id="slot01",
            branch="geometry",
            pred_label=0,
            score=3.0,
            threshold=10.0,
            defect_type="none",
            reason=None,
            source_path="part001_top_uniform.png",
        ),
    ]

    decision = fusion.fuse_part_predictions("part001", predictions)

    assert decision.final_status == "OK"
    assert decision.final_label == 0
    assert decision.triggered_branch is None


def test_near_threshold_prediction_returns_suspect_when_enabled() -> None:
    """Near-threshold negatives should be reviewable SUSPECT cases."""
    fusion = load_fusion_module()
    predictions = [
        fusion.BranchPrediction(
            part_id="part001",
            side="top",
            view="uniform",
            slot_id="slot04",
            branch="geometry",
            pred_label=0,
            score=9.2,
            threshold=10.0,
            defect_type="more",
            reason=None,
            source_path="part001_top_uniform.png",
        ),
    ]

    decision = fusion.fuse_part_predictions(
        "part001",
        predictions,
        config={"suspect_policy": {"enable": True, "near_threshold_ratio": 0.9}},
    )

    assert decision.final_status == "SUSPECT"
    assert decision.final_label is None
    assert decision.triggered_branch == "geometry"
    assert "near threshold" in decision.reason


def test_surface_texture_positive_maps_to_suspect_not_ng() -> None:
    """Surface texture positives should request review without counting as hard NG."""
    fusion = load_fusion_module()
    predictions = [
        fusion.BranchPrediction(
            part_id="part001",
            side="top",
            view="uniform",
            slot_id="slot03",
            branch="surface_texture",
            pred_label=1,
            score=21.0,
            threshold=18.0,
            defect_type="surface",
            reason="surface_texture local_residual=21 threshold=18",
            source_path="part001_top_uniform_slot03.png",
            status="SUSPECT",
        ),
    ]

    decision = fusion.fuse_part_predictions("part001", predictions)

    assert decision.final_status == "SUSPECT"
    assert decision.final_label is None
    assert decision.triggered_branch == "surface_texture"


def test_load_branch_csvs_normalize_geometry_and_anomaly_predictions(tmp_path: Path) -> None:
    """Existing geometry/anomaly CSV columns should become unified branch rows."""
    fusion = load_fusion_module()
    geometry_csv = tmp_path / "geometry_predictions.csv"
    geometry_csv.write_text(
        "\n".join(
            [
                "source_path,image_path,slot,geometry_pred_label,geometry_score,geometry_threshold,geometry_type",
                "dataset/part001_top_uniform_slot02.png,,slot02,1,12.0,10.0,less",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    anomaly_csv = tmp_path / "predictions.csv"
    anomaly_csv.write_text(
        "\n".join(
            [
                "source_path,processed_path,model,view,pred_score,deploy_threshold,deploy_pred_label",
                "dataset/part002_top_uniform_slot05.png,processed/part002.png,anomaly_dino,uniform,0.7,0.5,1",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    predictions = [
        *fusion.load_branch_predictions_csv(geometry_csv, branch="geometry"),
        *fusion.load_branch_predictions_csv(anomaly_csv, branch="anomaly_dino"),
    ]

    assert [(row.part_id, row.branch, row.pred_label, row.score, row.threshold) for row in predictions] == [
        ("part001_top_uniform_slot02", "geometry", 1, 12.0, 10.0),
        ("part002_top_uniform_slot05", "anomaly_dino", 1, 0.7, 0.5),
    ]
    assert predictions[0].slot_id == "slot02"
    assert predictions[1].view == "uniform"


def test_group_predictions_merges_source_processed_and_basename_aliases(tmp_path: Path) -> None:
    """Rows from different branch CSV schemas should merge when any path alias matches."""
    fusion = load_fusion_module()
    geometry_csv = tmp_path / "geometry_predictions.csv"
    shared_processed = tmp_path / "processed" / "part001_top_uniform_slot02.png"
    geometry_csv.write_text(
        "\n".join(
            [
                "source_path,image_path,slot,geometry_pred_label,geometry_score,geometry_threshold,geometry_type",
                f"{shared_processed},,slot02,0,2.0,10.0,none",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    anomaly_csv = tmp_path / "predictions.csv"
    anomaly_csv.write_text(
        "\n".join(
            [
                "source_path,processed_path,model,view,pred_score,deploy_threshold,deploy_pred_label",
                f"{tmp_path / 'raw' / 'part001.png'},{shared_processed},anomaly_dino,uniform,0.8,0.5,1",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    predictions = [
        *fusion.load_branch_predictions_csv(geometry_csv, branch="geometry"),
        *fusion.load_branch_predictions_csv(anomaly_csv, branch="anomaly_dino"),
    ]

    grouped = fusion.group_predictions_by_part(predictions)
    decisions = fusion.fuse_grouped_predictions(grouped)

    assert len(grouped) == 1
    assert [prediction.branch for prediction in next(iter(grouped.values()))] == ["geometry", "anomaly_dino"]
    assert decisions[0].final_status == "NG_ANOMALY"


def test_group_predictions_merges_unique_basename_only_matches(tmp_path: Path) -> None:
    """Rows with a unique basename/stem match should merge when full paths differ."""
    fusion = load_fusion_module()
    geometry_csv = tmp_path / "geometry_predictions.csv"
    geometry_csv.write_text(
        "\n".join(
            [
                "source_path,image_path,slot,geometry_pred_label,geometry_score,geometry_threshold,geometry_type",
                f"{tmp_path / 'raw' / 'part001_slot02.png'},,slot02,0,2.0,10.0,none",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    anomaly_csv = tmp_path / "predictions.csv"
    anomaly_csv.write_text(
        "\n".join(
            [
                "source_path,processed_path,model,view,pred_score,deploy_threshold,deploy_pred_label",
                f"{tmp_path / 'processed' / 'part001_slot02.png'},,anomaly_dino,uniform,0.8,0.5,1",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    predictions = [
        *fusion.load_branch_predictions_csv(geometry_csv, branch="geometry"),
        *fusion.load_branch_predictions_csv(anomaly_csv, branch="anomaly_dino"),
    ]

    grouped = fusion.group_predictions_by_part(predictions)
    decisions = fusion.fuse_grouped_predictions(grouped)

    assert len(grouped) == 1
    assert [prediction.branch for prediction in next(iter(grouped.values()))] == ["geometry", "anomaly_dino"]
    assert decisions[0].final_status == "NG_ANOMALY"


def test_group_predictions_does_not_merge_ambiguous_basename_only_matches(tmp_path: Path) -> None:
    """Basename/stem aliases should not merge rows when full paths point to different inputs."""
    fusion = load_fusion_module()
    geometry_csv = tmp_path / "geometry_predictions.csv"
    geometry_csv.write_text(
        "\n".join(
            [
                "source_path,image_path,slot,geometry_pred_label,geometry_score,geometry_threshold,geometry_type",
                f"{tmp_path / 'normal' / 'shared_slot02.png'},,slot02,0,2.0,10.0,none",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    anomaly_csv = tmp_path / "predictions.csv"
    anomaly_csv.write_text(
        "\n".join(
            [
                "source_path,processed_path,model,view,pred_score,deploy_threshold,deploy_pred_label",
                f"{tmp_path / 'defect' / 'shared_slot02.png'},,anomaly_dino,uniform,0.8,0.5,1",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    predictions = [
        *fusion.load_branch_predictions_csv(geometry_csv, branch="geometry"),
        *fusion.load_branch_predictions_csv(anomaly_csv, branch="anomaly_dino"),
    ]

    grouped = fusion.group_predictions_by_part(predictions)

    assert len(grouped) == 2
    assert sorted(len(part_predictions) for part_predictions in grouped.values()) == [1, 1]


def test_write_fusion_outputs_and_benchmark_summary(tmp_path: Path) -> None:
    """Fusion outputs should be persisted and summarized for robustness reports."""
    fusion = load_fusion_module()
    predictions = [
        fusion.BranchPrediction(
            part_id="normal001",
            side="top",
            view="uniform",
            slot_id="slot01",
            branch="geometry",
            pred_label=0,
            score=1.0,
            threshold=10.0,
            defect_type="none",
            reason=None,
            source_path=str(tmp_path / "normal_test" / "normal001.png"),
        ),
        fusion.BranchPrediction(
            part_id="defect001",
            side="top",
            view="uniform",
            slot_id="slot02",
            branch="geometry",
            pred_label=1,
            score=12.0,
            threshold=10.0,
            defect_type="less",
            reason=None,
            source_path=str(tmp_path / "defect" / "defect001.png"),
        ),
    ]
    grouped = fusion.group_predictions_by_part(predictions)
    decisions = fusion.fuse_grouped_predictions(grouped)

    branch_csv = tmp_path / "branch_predictions.csv"
    fused_csv = tmp_path / "fused_predictions.csv"
    fusion.write_branch_predictions_csv(predictions, branch_csv)
    fusion.write_fused_decisions_csv(decisions, fused_csv)
    summary = fusion.compute_benchmark_summary(decisions, grouped)

    assert branch_csv.read_text(encoding="utf-8").splitlines()[0].startswith("part_id,side,view")
    assert "NG_GEOMETRY" in fused_csv.read_text(encoding="utf-8")
    assert summary["total_parts"] == 2
    assert summary["defect_recall"] == 1.0
    assert summary["fused_recall"] == 1.0
    assert summary["geometry_recall"] == 1.0
    assert summary["anomaly_dino_recall"] == 0.0
    assert summary["normal_false_positives"] == 0


def test_benchmark_summary_keeps_unknown_out_of_normal_denominator(tmp_path: Path) -> None:
    """Unknown predictions should not silently dilute normal false-positive metrics."""
    fusion = load_fusion_module()
    grouped = {
        "unknown001": [
            fusion.BranchPrediction(
                part_id="unknown001",
                side="unknown",
                view=None,
                slot_id=None,
                branch="geometry",
                pred_label=0,
                score=0.0,
                threshold=1.0,
                defect_type="none",
                reason=None,
                source_path=str(tmp_path / "misc" / "unknown001.png"),
            ),
        ],
        "normal001": [
            fusion.BranchPrediction(
                part_id="normal001",
                side="top",
                view="uniform",
                slot_id="slot01",
                branch="geometry",
                pred_label=1,
                score=12.0,
                threshold=10.0,
                defect_type="more",
                reason=None,
                source_path=str(tmp_path / "normal_test" / "normal001.png"),
            ),
        ],
    }
    decisions = fusion.fuse_grouped_predictions(grouped)

    summary = fusion.compute_benchmark_summary(
        decisions,
        grouped,
        clean_normal_root=tmp_path / "normal_test",
    )

    assert summary["unknown_total"] == 1
    assert summary["normal_total"] == 1
    assert summary["clean_normal_false_positives"] == 1
    assert summary["normal_fp_rate"] == 1.0
