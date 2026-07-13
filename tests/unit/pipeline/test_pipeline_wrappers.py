# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for numbered pipeline wrapper helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import cv2
import numpy as np
import pytest


def _load_module(name: str, relative_path: str) -> ModuleType:
    """Load a pipeline module from a file path."""
    script_path = Path(__file__).resolve().parents[3] / relative_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load {script_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(script_path.parent))
    spec.loader.exec_module(module)
    return module


def test_multicamera_collection_wrapper_forwards_arguments_unchanged() -> None:
    """The numbered wrapper should delegate every capture option unchanged."""
    wrapper = _load_module("pipeline_collect_multicamera", "pipeline/1_collect_multicamera_data.py")
    argv = [
        "pipeline/1_collect_multicamera_data.py",
        "--devices",
        "0",
        "1",
        "2",
        "--hand",
        "left",
        "--label",
        "normal",
        "--hdr",
    ]

    with patch.object(wrapper, "run_repo_script") as run_repo_script, patch.object(sys, "argv", argv):
        wrapper.main()

    run_repo_script.assert_called_once_with(
        "capture_data/collect_multicamera_dataset.py",
        ["--devices", "0", "1", "2", "--hand", "left", "--label", "normal", "--hdr"],
    )


def test_with_default_command_inserts_all_for_workflow_options() -> None:
    """Workflow wrappers should insert the default subcommand before options."""
    common = _load_module("pipeline_common", "pipeline/_common.py")

    args = common.with_default_command(["--models", "anomaly_dino"], {"all", "train"}, "all")

    assert args == ["all", "--models", "anomaly_dino"]


def test_with_default_command_keeps_explicit_workflow_command() -> None:
    """Workflow wrappers should preserve explicit subcommands."""
    common = _load_module("pipeline_common_explicit", "pipeline/_common.py")

    args = common.with_default_command(["train", "--models", "patchcore"], {"all", "train"}, "all")

    assert args == ["train", "--models", "patchcore"]


def test_process_wrapper_resolves_default_auto_mode() -> None:
    """The processing wrapper should default option-style arguments to auto mode."""
    process = _load_module("pipeline_process", "pipeline/2_process_data.py")

    mode, args = process._resolve_mode(["--data-root", "dataset/c789"])

    assert mode == "auto"
    assert args == ["--data-root", "dataset/c789"]


def test_process_wrapper_resolves_manual_mode() -> None:
    """The processing wrapper should route manual mode explicitly."""
    process = _load_module("pipeline_process_manual", "pipeline/2_process_data.py")

    mode, args = process._resolve_mode(["manual", "--slot-count", "6"])

    assert mode == "manual"
    assert args == ["--slot-count", "6"]


def test_process_wrapper_forwards_manual_slot_arguments() -> None:
    """Manual fixed-slot arguments should be forwarded unchanged."""
    process = _load_module("pipeline_process_manual_slots", "pipeline/2_process_data.py")

    mode, args = process._resolve_mode(["manual", "--slot", "slot01:1,1,832,43,3200,446"])

    assert mode == "manual"
    assert args == ["--slot", "slot01:1,1,832,43,3200,446"]


def test_process_wrapper_resolves_fx11_defect_mode() -> None:
    """The processing wrapper should route the FX11 defect preset."""
    process = _load_module("pipeline_process_fx11_defect", "pipeline/2_process_data.py")

    mode, args = process._resolve_mode(["fx11-defect", "--face", "bottom"])

    assert mode == "fx11-defect"
    assert args == ["--face", "bottom"]


def test_fx11_defect_preset_builds_bottom_manual_args(tmp_path: Path) -> None:
    """The FX11 defect preset should build fixed bottom slot crop arguments."""
    preset = _load_module("pipeline_fx11_defect_crop", "pipeline/fx11_defect_crop.py")
    args = preset.build_parser().parse_args(
        [
            "--face",
            "bottom",
            "--data-root",
            str(tmp_path / "dataset"),
            "--output-root",
            str(tmp_path / "parts"),
            "--results-root",
            str(tmp_path / "results"),
            "--overwrite",
        ],
    )

    manual_args = preset.build_manual_args("bottom", args)

    assert manual_args[:8] == [
        "--data-root",
        str(tmp_path / "dataset"),
        "--output-root",
        str(tmp_path / "parts"),
        "--hand",
        "no_hand",
        "--position",
        "bottom",
    ]
    assert "--labels" in manual_args
    assert "defect" in manual_args
    assert "--defect-slot-mode" in manual_args
    assert "from-name" in manual_args
    assert ["--slot", "slot01:1,1,830,10,3230,420"] == manual_args[
        manual_args.index("--slot") : manual_args.index("--slot") + 2
    ]
    assert "--overwrite" in manual_args


def test_run_all_builds_stages_in_order() -> None:
    """The all-in-one wrapper should preserve pipeline stage order."""
    run_all = _load_module("pipeline_run_all", "pipeline/0_run_all.py")
    args = run_all.build_parser().parse_args(
        [
            "--collect",
            "--hand left --position top --label normal",
            "--process",
            "auto --data-root dataset/c789",
            "--train",
            "--models anomaly_dino",
            "--inference",
            "image.png --model anomaly_dino",
        ],
    )

    stages = run_all.build_stages(args)

    assert [stage.name for stage in stages] == [
        "1_collect_data",
        "2_process_data",
        "3_train_model",
        "4_inference",
    ]
    assert stages[0].args == ["--hand", "left", "--position", "top", "--label", "normal"]
    assert stages[1].args == ["auto", "--data-root", "dataset/c789"]


def test_run_all_supports_repeated_collection() -> None:
    """The all-in-one wrapper should allow normal and defect collection runs."""
    run_all = _load_module("pipeline_run_all_collect", "pipeline/0_run_all.py")
    args = run_all.build_parser().parse_args(
        [
            "--collect",
            "--label normal",
            "--collect",
            "--label defect --defect-type scratch",
        ],
    )

    stages = run_all.build_stages(args)

    assert [stage.name for stage in stages] == ["1_collect_data", "1_collect_data"]
    assert stages[0].args == ["--label", "normal"]
    assert stages[1].args == ["--label", "defect", "--defect-type", "scratch"]


def test_compare_models_builds_safe_default_workflow_args(tmp_path: Path) -> None:
    """The model-comparison wrapper should build conservative multi-model workflow args."""
    compare = _load_module("pipeline_compare_models", "pipeline/6_compare_models.py")
    args = compare.build_parser().parse_args(
        [
            "--data-root",
            str(tmp_path / "parts"),
            "--output-root",
            str(tmp_path / "results"),
            "--views",
            "no_hand_top",
            "--skip-blue-removal",
        ],
    )

    workflow_args = compare.build_workflow_args("all", args)

    assert workflow_args[:5] == [
        "all",
        "--data-root",
        str(tmp_path / "parts"),
        "--output-root",
        str(tmp_path / "results"),
    ]
    assert workflow_args[workflow_args.index("--models") + 1 : workflow_args.index("--roi")] == [
        "patchcore",
        "efficient_ad",
        "anomaly_dino",
    ]
    assert ["--patchcore-layers", "layer2"] == workflow_args[
        workflow_args.index("--patchcore-layers") : workflow_args.index("--patchcore-layers") + 2
    ]
    assert "--anomaly-dino-coreset-subsampling" in workflow_args
    assert "--skip-missing-efficientad-assets" in workflow_args
    assert ["--deploy-fpr", "0.05"] == workflow_args[
        workflow_args.index("--deploy-fpr") : workflow_args.index("--deploy-fpr") + 2
    ]


def test_compare_models_can_require_efficientad_assets(tmp_path: Path) -> None:
    """The comparison wrapper should allow strict EfficientAD asset checking."""
    compare = _load_module("pipeline_compare_models_require_assets", "pipeline/6_compare_models.py")
    args = compare.build_parser().parse_args(
        [
            "--data-root",
            str(tmp_path / "parts"),
            "--output-root",
            str(tmp_path / "results"),
            "--views",
            "no_hand_top",
            "--require-efficientad-assets",
        ],
    )

    workflow_args = compare.build_workflow_args("all", args)

    assert "--skip-missing-efficientad-assets" not in workflow_args


def test_compare_models_skip_preprocess_runs_train_then_evaluate(tmp_path: Path) -> None:
    """The comparison wrapper should reuse preprocessing when requested."""
    compare = _load_module("pipeline_compare_models_skip", "pipeline/6_compare_models.py")
    args = compare.build_parser().parse_args(
        [
            "--data-root",
            str(tmp_path / "parts"),
            "--output-root",
            str(tmp_path / "results"),
            "--views",
            "no_hand_bottom",
            "--skip-preprocess",
        ],
    )

    commands = compare.build_commands(args)

    assert [command.name for command in commands] == ["train", "evaluate"]
    assert commands[0].args[0] == "train"
    assert commands[1].args[0] == "evaluate"


def test_compare_models_expands_short_efficientad_image_size(tmp_path: Path) -> None:
    """The comparison wrapper should avoid EfficientAD feature maps smaller than its kernel."""
    compare = _load_module("pipeline_compare_models_efficientad_size", "pipeline/6_compare_models.py")
    args = compare.build_parser().parse_args(
        [
            "--data-root",
            str(tmp_path / "parts"),
            "--output-root",
            str(tmp_path / "results"),
            "--views",
            "no_hand_top",
            "--image-size",
            "224,1008",
        ],
    )

    workflow_args = compare.build_workflow_args("all", args)

    assert ["--image-size", "256,1152"] == workflow_args[
        workflow_args.index("--image-size") : workflow_args.index("--image-size") + 2
    ]


def test_compare_models_writes_sorted_report(tmp_path: Path) -> None:
    """The comparison wrapper should write a compact sorted comparison report."""
    compare = _load_module("pipeline_compare_models_report", "pipeline/6_compare_models.py")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "summary.csv").write_text(
        "\n".join(
            [
                "model,view,deploy_threshold,deploy_target_fpr,image_fpr,image_accuracy,image_recall,image_f1,"
                "sample_fpr,sample_accuracy,sample_recall,sample_f1",
                "patchcore,no_hand_top,0.4,0.05,0.04,0.9,0.8,0.85,0.03,0.91,0.82,0.86",
                "anomaly_dino,no_hand_top,0.5,0.05,0.02,0.92,0.7,0.8,0.01,0.93,0.7,0.78",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    report_path = compare.write_comparison_report(tmp_path)

    report = report_path.read_text(encoding="utf-8")
    assert report.index("patchcore") < report.index("anomaly_dino")
    assert "sample_f1" in report


def test_split_stress_wrapper_targets_stress_split_script() -> None:
    """The stress split wrapper should forward to the capture_data splitter."""
    wrapper = _load_module("pipeline_split_stress", "pipeline/9_split_stress_normal.py")

    assert wrapper.TARGET_SCRIPT == "capture_data/prepare_stress_splits.py"


def test_build_hardened_wrapper_targets_hardened_dataset_script() -> None:
    """The hardened dataset wrapper should forward to the capture_data builder."""
    wrapper = _load_module("pipeline_build_hardened", "pipeline/10_build_hardened_dataset.py")

    assert wrapper.TARGET_SCRIPT == "capture_data/build_hardened_dataset.py"


def test_build_geometry_templates_wrapper_targets_geometry_builder() -> None:
    """The geometry template wrapper should forward to the template builder."""
    wrapper = _load_module("pipeline_build_geometry_templates", "pipeline/11_build_geometry_templates.py")

    assert wrapper.TARGET_SCRIPT == "capture_data/build_geometry_templates.py"


def test_geometry_eval_wrapper_targets_geometry_evaluator() -> None:
    """The geometry evaluation wrapper should forward to the geometry evaluator."""
    wrapper = _load_module("pipeline_geometry_eval", "pipeline/12_geometry_eval.py")

    assert wrapper.TARGET_SCRIPT == "capture_data/evaluate_geometry_shape.py"


def test_export_geometry_review_pack_wrapper_targets_exporter() -> None:
    """The manual review-pack wrapper should forward to the review exporter."""
    wrapper = _load_module("pipeline_export_geometry_review", "pipeline/13_export_geometry_review_pack.py")

    assert wrapper.TARGET_SCRIPT == "capture_data/export_geometry_review_pack.py"


def test_build_manual_geometry_templates_wrapper_targets_builder() -> None:
    """The manual geometry wrapper should forward to the manual template builder."""
    wrapper = _load_module("pipeline_build_manual_geometry", "pipeline/14_build_manual_geometry_templates.py")

    assert wrapper.TARGET_SCRIPT == "capture_data/build_manual_geometry_templates.py"


def test_fuse_inspection_results_parser_accepts_optional_branch_csvs(tmp_path: Path) -> None:
    """The fusion wrapper should accept existing geometry and anomaly report paths."""
    wrapper = _load_module("pipeline_fuse_inspection", "pipeline/18_fuse_inspection_results.py")

    args = wrapper.build_parser().parse_args(
        [
            "--geometry-csv",
            str(tmp_path / "geometry_predictions.csv"),
            "--anomaly-csv",
            str(tmp_path / "predictions.csv"),
            "--branch-csv",
            f"traditional={tmp_path / 'traditional_predictions.csv'}",
            "--fusion-config",
            str(tmp_path / "fusion.yaml"),
            "--output-dir",
            str(tmp_path / "fused"),
        ],
    )

    assert args.geometry_csv == tmp_path / "geometry_predictions.csv"
    assert args.anomaly_csv == tmp_path / "predictions.csv"
    assert args.branch_csv == [f"traditional={tmp_path / 'traditional_predictions.csv'}"]
    assert args.output_dir == tmp_path / "fused"


def test_fuse_inspection_results_parser_accepts_zs32_audit_options(tmp_path: Path) -> None:
    """Stage 18 should expose strict ZS32 audit controls."""
    wrapper = _load_module("pipeline_fuse_inspection_zs32_parser", "pipeline/18_fuse_inspection_results.py")

    args = wrapper.build_parser().parse_args(
        [
            "--profile",
            "zs32",
            "--audit-dir",
            str(tmp_path / "audit"),
            "--require-complete-evidence",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert args.profile == "zs32"
    assert args.audit_dir == tmp_path / "audit"
    assert args.require_complete_evidence is True


def test_calibrate_zs32_fusion_parser_accepts_grouped_calibration_options(tmp_path: Path) -> None:
    """Stage 30 should expose bounded calibration controls and repeatable required views."""
    wrapper = _load_module("pipeline_calibrate_zs32_fusion", "pipeline/30_calibrate_zs32_fusion.py")

    args = wrapper.build_parser().parse_args(
        [
            "--input-csv",
            str(tmp_path / "calibration.csv"),
            "--output-dir",
            str(tmp_path / "reports"),
            "--target-recall",
            "0.99",
            "--normal-quantile",
            "0.995",
            "--fit-split",
            "calibration",
            "--eval-split",
            "test",
            "--required-view",
            "front",
            "--required-view",
            "back",
            "--required-group",
            "left:front:anomaly_front:model-v1:roi-v1",
        ],
    )

    assert args.input_csv == tmp_path / "calibration.csv"
    assert args.output_dir == tmp_path / "reports"
    assert args.target_recall == pytest.approx(0.99)
    assert args.normal_quantile == pytest.approx(0.995)
    assert args.fit_split == "calibration"
    assert args.eval_split == "test"
    assert args.required_view == ["front", "back"]
    assert args.required_group == [("left", "front", "anomaly_front", "model-v1", "roi-v1")]


@pytest.mark.parametrize(("option", "value"), [("--target-recall", "0"), ("--normal-quantile", "1.1")])
def test_calibrate_zs32_fusion_parser_rejects_out_of_range_rates(option: str, value: str, tmp_path: Path) -> None:
    """Stage 30 rate arguments must stay in the open-zero, closed-one interval."""
    wrapper = _load_module(f"pipeline_calibrate_zs32_fusion_{option}", "pipeline/30_calibrate_zs32_fusion.py")

    with pytest.raises(SystemExit):
        wrapper.build_parser().parse_args(
            [
                "--input-csv",
                str(tmp_path / "calibration.csv"),
                "--output-dir",
                str(tmp_path / "reports"),
                option,
                value,
            ],
        )


def test_fuse_inspection_results_run_writes_expected_decisions(tmp_path: Path) -> None:
    """The fusion wrapper should run an end-to-end CSV smoke test."""
    wrapper = _load_module("pipeline_fuse_inspection_run", "pipeline/18_fuse_inspection_results.py")
    geometry_csv = tmp_path / "geometry_predictions.csv"
    geometry_csv.write_text(
        "\n".join(
            [
                "part_id,side,view,slot,geometry_pred_label,geometry_score,"
                "geometry_threshold,geometry_type,source_path",
                "part_geometry,top,uniform,slot02,1,12.0,10.0,less,part_geometry_top_uniform_slot02.png",
                "part_ok,top,uniform,slot01,0,1.0,10.0,none,part_ok_top_uniform_slot01.png",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    anomaly_csv = tmp_path / "predictions.csv"
    anomaly_csv.write_text(
        "\n".join(
            [
                "part_id,side,view,slot_id,pred_score,deploy_threshold,deploy_pred_label,defect_type,source_path",
                "part_anomaly,top,uniform,slot05,0.8,0.5,1,surface,part_anomaly_top_uniform_slot05.png",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    quality_csv = tmp_path / "quality_gate.csv"
    quality_csv.write_text(
        "\n".join(
            [
                "part_id,side,view,status,reason,source_path",
                "part_quality,top,uniform,FAIL,blur_laplacian_var below minimum,part_quality_top_uniform.png",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_csv = tmp_path / "manifest.csv"
    manifest_csv.write_text(
        "part_id,side,view,image_path,label,defect_type,slot_id,group_id,notes\n"
        "part_missing,bottom,uniform,part_missing_bottom_uniform.png,normal,,,,\n",
        encoding="utf-8",
    )
    args = wrapper.build_parser().parse_args(
        [
            "--manifest",
            str(manifest_csv),
            "--required-view",
            "top:uniform",
            "--quality-csv",
            str(quality_csv),
            "--geometry-csv",
            str(geometry_csv),
            "--anomaly-csv",
            str(anomaly_csv),
            "--output-dir",
            str(tmp_path / "fused"),
        ],
    )

    decisions = wrapper.run_fusion(args)
    statuses = {decision.part_id: decision.final_status for decision in decisions}

    assert statuses["part_geometry"] == "NG_GEOMETRY"
    assert statuses["part_anomaly"] == "NG_ANOMALY"
    assert statuses["part_quality"] == "RETAKE"
    assert statuses["part_missing"] == "INVALID_CAPTURE"
    assert statuses["part_ok"] == "OK"
    assert "part_geometry" in (tmp_path / "fused" / "branch_predictions.csv").read_text(encoding="utf-8")


def test_fuse_inspection_results_loads_custom_branch_csv(tmp_path: Path) -> None:
    """Custom branch CSVs should participate in fail-closed fusion."""
    wrapper = _load_module("pipeline_fuse_inspection_custom_branch", "pipeline/18_fuse_inspection_results.py")
    branch_csv = tmp_path / "traditional_predictions.csv"
    branch_csv.write_text(
        "\n".join(
            [
                "part_id,side,view,slot_id,branch,pred_label,score,threshold,defect_type,evidence_type,"
                "gt_defect_type,reason,source_path,evidence_path",
                "part_crack,top,uniform,slot03,crack,1,0.62,0.5,crack,dark_line,crack,thin dark line,"
                "part_crack_top_uniform_slot03.png,evidence/part_crack.png",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    args = wrapper.build_parser().parse_args(
        [
            "--branch-csv",
            f"traditional={branch_csv}",
            "--output-dir",
            str(tmp_path / "fused"),
        ],
    )

    decisions = wrapper.run_fusion(args)

    assert decisions[0].final_status == "NG_CRACK"
    branch_output = (tmp_path / "fused" / "branch_predictions.csv").read_text(encoding="utf-8")
    assert "evidence/part_crack.png" in branch_output
    assert "dark_line" in branch_output
    assert "gt_defect_type" in branch_output


def test_fuse_inspection_results_loads_yolo_branch_csv_by_default(tmp_path: Path) -> None:
    """YOLO branch CSVs should trigger NG_YOLO without custom fusion config."""
    wrapper = _load_module("pipeline_fuse_inspection_yolo_branch", "pipeline/18_fuse_inspection_results.py")
    branch_csv = tmp_path / "yolo_predictions.csv"
    branch_csv.write_text(
        "\n".join(
            [
                "part_id,side,view,slot_id,pred_label,score,threshold,defect_type,reason,"
                "source_path,evidence_path",
                "part_yolo,top,uniform,slot06,1,0.91,0.5,defect,yolo box confidence above threshold,"
                "part_yolo_top_uniform_slot06.png,evidence/part_yolo.png",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    args = wrapper.build_parser().parse_args(
        [
            "--branch-csv",
            f"yolo={branch_csv}",
            "--output-dir",
            str(tmp_path / "fused"),
        ],
    )

    decisions = wrapper.run_fusion(args)

    assert decisions[0].final_status == "NG_YOLO"
    assert decisions[0].defect_slot == "slot06"
    assert decisions[0].triggered_branch == "yolo"


def test_run_traditional_operators_parser_accepts_c789_options(tmp_path: Path) -> None:
    """Stage 20 should expose the C789 traditional-operator CLI surface."""
    wrapper = _load_module("pipeline_traditional_operators", "pipeline/20_run_traditional_operators.py")

    args = wrapper.build_parser().parse_args(
        [
            "--input-root",
            str(tmp_path / "images"),
            "--preset",
            "c789_left_top_3x2",
            "--side",
            "top",
            "--view",
            "uniform",
            "--config",
            str(tmp_path / "traditional.yaml"),
            "--calibrate-normal-root",
            str(tmp_path / "normal"),
            "--template-dir",
            str(tmp_path / "templates"),
            "--geometry-thresholds",
            str(tmp_path / "geometry_thresholds.csv"),
            "--label",
            "normal",
            "--no-progress",
            "--output-dir",
            str(tmp_path / "traditional"),
        ],
    )

    assert args.input_root == tmp_path / "images"
    assert args.preset == "c789_left_top_3x2"
    assert args.side == "top"
    assert args.output_dir == tmp_path / "traditional"
    assert args.calibrate_normal_root == tmp_path / "normal"
    assert args.template_dir == tmp_path / "templates"
    assert args.geometry_thresholds == tmp_path / "geometry_thresholds.csv"
    assert args.label == "normal"
    assert args.show_progress is False


def test_prepare_yolo_dataset_parser_accepts_c789_options(tmp_path: Path) -> None:
    """Stage 21 should expose the C789 YOLO export CLI surface."""
    wrapper = _load_module("pipeline_prepare_yolo_dataset", "pipeline/21_prepare_yolo_dataset.py")

    args = wrapper.build_parser().parse_args(
        [
            "--manifest",
            str(tmp_path / "part_crop_manifest.csv"),
            "--annotations",
            str(tmp_path / "bbox_annotations.csv"),
            "--output-root",
            str(tmp_path / "yolo"),
            "--positive-val-ratio",
            "0.25",
            "--normal-test-split",
            "val",
            "--preview-dir",
            str(tmp_path / "previews"),
            "--overwrite",
        ],
    )

    assert args.manifest == tmp_path / "part_crop_manifest.csv"
    assert args.annotations == tmp_path / "bbox_annotations.csv"
    assert args.output_root == tmp_path / "yolo"
    assert args.positive_val_ratio == 0.25
    assert args.normal_test_split == "val"
    assert args.preview_dir == tmp_path / "previews"
    assert args.overwrite is True


def test_prepare_yolo_same_dist_dataset_parser_accepts_c789_options(tmp_path: Path) -> None:
    """Stage 25 should expose the C789 same-distribution YOLO dataset CLI surface."""
    wrapper = _load_module("pipeline_prepare_yolo_same_dist_dataset", "pipeline/25_prepare_yolo_same_dist_dataset.py")

    args = wrapper.build_parser().parse_args(
        [
            "--input-root",
            str(tmp_path / "flat"),
            "--output-root",
            str(tmp_path / "same_dist"),
            "--normal-source-root",
            str(tmp_path / "balanced"),
            "--val-groups",
            "g002",
            "g008",
            "--train-normal-limit",
            "600",
            "--val-normal-limit",
            "150",
            "--preview-limit",
            "16",
            "--overwrite",
        ],
    )

    assert args.input_root == tmp_path / "flat"
    assert args.output_root == tmp_path / "same_dist"
    assert args.normal_source_root == tmp_path / "balanced"
    assert tuple(args.val_groups) == ("g002", "g008")
    assert args.train_normal_limit == 600
    assert args.val_normal_limit == 150
    assert args.preview_limit == 16
    assert args.overwrite is True


def test_prepare_yolo_roi_dataset_parser_accepts_c789_options(tmp_path: Path) -> None:
    """Stage 26 should expose the C789 ROI YOLO dataset CLI surface."""
    wrapper = _load_module("pipeline_prepare_yolo_roi_dataset", "pipeline/26_prepare_yolo_roi_dataset.py")

    args = wrapper.build_parser().parse_args(
        [
            "--input-root",
            str(tmp_path / "flat"),
            "--output-root",
            str(tmp_path / "roi"),
            "--mode",
            "gt-center",
            "--roi-size",
            "512",
            "--stride",
            "256",
            "--normal-source-root",
            str(tmp_path / "balanced"),
            "--train-normal-ratio",
            "2",
            "--val-normal-limit",
            "150",
            "--preview-limit",
            "12",
            "--overwrite",
        ],
    )

    assert args.input_root == tmp_path / "flat"
    assert args.output_root == tmp_path / "roi"
    assert args.mode == "gt-center"
    assert args.roi_size == 512
    assert args.stride == 256
    assert args.normal_source_root == tmp_path / "balanced"
    assert args.train_normal_ratio == 2
    assert args.val_normal_limit == 150
    assert args.preview_limit == 12
    assert args.overwrite is True


def test_prepare_zs32_label_studio_parser_accepts_paths_and_overwrite(tmp_path: Path) -> None:
    wrapper = _load_module(
        "pipeline_prepare_zs32_label_studio",
        "pipeline/27_prepare_zs32_label_studio.py",
    )
    args = wrapper.build_parser().parse_args(
        [
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-root",
            str(tmp_path / "labeling"),
            "--overwrite",
        ],
    )
    assert args.dataset_root == tmp_path / "dataset"
    assert args.output_root == tmp_path / "labeling"
    assert args.overwrite is True


def test_collect_c789_yolo_defects_builds_capture_and_crop_args(tmp_path: Path) -> None:
    """Stage 22 should capture C789 defect trays and crop all six slots as defects."""
    wrapper = _load_module("pipeline_collect_c789_yolo_defects", "pipeline/22_collect_c789_yolo_defects.py")

    args = wrapper.build_parser().parse_args(
        [
            "--raw-root",
            str(tmp_path / "raw"),
            "--parts-root",
            str(tmp_path / "parts"),
            "--defect-output-dir",
            str(tmp_path / "defect_images"),
            "--hand",
            "right",
            "--position",
            "bottom",
            "--defect-type",
            "scratch",
            "--part-id",
            "batch001",
            "--group-count",
            "3",
            "--images-per-group",
            "2",
            "--overwrite",
        ],
    )

    collect_args = wrapper.build_collect_args(args)
    crop_args = wrapper.build_crop_args(args)

    assert collect_args[:6] == ["--hand", "right", "--position", "bottom", "--label", "defect"]
    assert ["--defect-type", "scratch"] == collect_args[
        collect_args.index("--defect-type") : collect_args.index("--defect-type") + 2
    ]
    assert ["--group-count", "3"] == collect_args[
        collect_args.index("--group-count") : collect_args.index("--group-count") + 2
    ]
    assert "--hdr" in collect_args
    assert ["--preset", "c789_left_bottom_3x2"] == crop_args[
        crop_args.index("--preset") : crop_args.index("--preset") + 2
    ]
    assert ["--output-position", "bottom_ZS32"] == crop_args[
        crop_args.index("--output-position") : crop_args.index("--output-position") + 2
    ]
    assert ["--labels", "defect"] == crop_args[crop_args.index("--labels") : crop_args.index("--labels") + 2]
    assert ["--hole-mask-method", "none"] == crop_args[
        crop_args.index("--hole-mask-method") : crop_args.index("--hole-mask-method") + 2
    ]
    assert not hasattr(args, "hole_mask_method")
    assert ["--defect-slot-mode", "all"] == crop_args[
        crop_args.index("--defect-slot-mode") : crop_args.index("--defect-slot-mode") + 2
    ]
    assert "--overwrite" in crop_args

    no_hdr_args = wrapper.build_parser().parse_args(
        ["--no-hdr", "--no-manual-load", "--no-save-hdr-sources", "--no-align-hdr"],
    )
    assert no_hdr_args.hdr is False
    assert no_hdr_args.manual_load is False
    assert no_hdr_args.save_hdr_sources is False
    assert no_hdr_args.align_hdr is False


def test_collect_c789_yolo_defects_flattens_defect_crops(tmp_path: Path) -> None:
    """Stage 22 should put processed defect crop images into one flat folder."""
    wrapper = _load_module("pipeline_collect_c789_yolo_defects_flatten", "pipeline/22_collect_c789_yolo_defects.py")
    parts_root = tmp_path / "parts"
    defect_output = tmp_path / "defect_images"
    first = parts_root / "left" / "top" / "defect" / "sample_a_slot01" / "images" / "crop.png"
    second = parts_root / "left" / "top" / "defect" / "sample_b_slot02" / "images" / "crop.png"
    for path, value in ((first, 80), (second, 160)):
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.full((8, 8, 3), value, dtype=np.uint8)
        assert cv2.imwrite(str(path), image)

    manifest_path = wrapper.copy_defect_crops(
        parts_root=parts_root,
        hand="left",
        output_position="top",
        defect_output_dir=defect_output,
        overwrite=False,
    )

    output_images = sorted(path.name for path in defect_output.glob("*.png"))
    manifest_text = manifest_path.read_text(encoding="utf-8")

    assert output_images == ["crop.png", "crop_57b34d5b.png"]
    assert "sample_a_slot01" in manifest_text
    assert "sample_b_slot02" in manifest_text


def test_build_multiview_manifest_parser_accepts_profile_and_regex(tmp_path: Path) -> None:
    """Stage 16 should expose the manifest builder CLI surface from the implementation plan."""
    wrapper = _load_module("pipeline_multiview_manifest", "pipeline/16_build_multiview_manifest.py")

    args = wrapper.build_parser().parse_args(
        [
            "--input-root",
            str(tmp_path / "raw"),
            "--profile",
            str(tmp_path / "profile.yaml"),
            "--filename-regex",
            r"(?P<part_id>.+)_(?P<side>top)_(?P<view>uniform)\.png",
            "--required-side",
            "top",
            "--required-view",
            "uniform",
            "--output-csv",
            str(tmp_path / "manifest.csv"),
        ],
    )

    assert args.input_root == tmp_path / "raw"
    assert args.output_csv == tmp_path / "manifest.csv"
    assert args.required_side == ["top"]
    assert args.required_view == ["uniform"]


def test_calibrate_quality_gate_parser_accepts_optional_invalid_root(tmp_path: Path) -> None:
    """Stage 17 should support normal/stress roots and optional invalid data."""
    wrapper = _load_module("pipeline_quality_gate", "pipeline/17_calibrate_quality_gate.py")

    args = wrapper.build_parser().parse_args(
        [
            "--normal-root",
            str(tmp_path / "normal"),
            "--stress-root",
            str(tmp_path / "stress"),
            "--invalid-root",
            str(tmp_path / "invalid"),
            "--profile",
            str(tmp_path / "profile.yaml"),
            "--output-yaml",
            str(tmp_path / "quality.yaml"),
            "--output-report",
            str(tmp_path / "report.md"),
        ],
    )

    assert args.normal_root == tmp_path / "normal"
    assert args.invalid_root == tmp_path / "invalid"
    assert args.output_yaml == tmp_path / "quality.yaml"


def test_robustness_benchmark_parser_accepts_existing_prediction_csvs(tmp_path: Path) -> None:
    """The benchmark wrapper should support a cheap CSV-only robustness run."""
    wrapper = _load_module("pipeline_robustness_benchmark", "pipeline/19_run_robustness_benchmark.py")

    args = wrapper.build_parser().parse_args(
        [
            "--clean-normal-root",
            str(tmp_path / "normal_test"),
            "--stress-normal-root",
            str(tmp_path / "stress"),
            "--defect-root",
            str(tmp_path / "defect"),
            "--geometry-csv",
            str(tmp_path / "geometry_predictions.csv"),
            "--anomaly-predictions",
            str(tmp_path / "predictions.csv"),
            "--output-dir",
            str(tmp_path / "benchmark"),
        ],
    )

    assert args.clean_normal_root == tmp_path / "normal_test"
    assert args.geometry_csv == tmp_path / "geometry_predictions.csv"
    assert args.output_dir == tmp_path / "benchmark"


def test_robustness_benchmark_counts_unpredicted_defect_inputs(tmp_path: Path) -> None:
    """Defect files with no branch prediction should stay in the recall denominator."""
    wrapper = _load_module("pipeline_robustness_benchmark_missing", "pipeline/19_run_robustness_benchmark.py")
    normal_root = tmp_path / "normal_test"
    defect_root = tmp_path / "defect"
    normal_root.mkdir(exist_ok=True)
    defect_root.mkdir(exist_ok=True)
    (defect_root / "less_1_2_slot02.png").write_text("placeholder", encoding="utf-8")
    geometry_csv = tmp_path / "geometry_predictions.csv"
    geometry_csv.write_text(
        "\n".join(
            [
                "source_path,image_path,slot,geometry_pred_label,geometry_score,geometry_threshold,geometry_type",
                f"{normal_root / 'normal001_slot01.png'},,slot01,0,1.0,10.0,none",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    args = wrapper.build_parser().parse_args(
        [
            "--clean-normal-root",
            str(normal_root),
            "--defect-root",
            str(defect_root),
            "--geometry-csv",
            str(geometry_csv),
            "--output-dir",
            str(tmp_path / "benchmark"),
        ],
    )

    summary = wrapper.run_benchmark(args)

    assert summary["defect_total"] == 1
    assert summary["defect_recall"] == 0.0
    assert summary["missing_prediction_count"] == 1
    assert "less_1_2_slot02" in (tmp_path / "benchmark" / "misses.csv").read_text(encoding="utf-8")
    assert "benchmark_input" in (tmp_path / "benchmark" / "branch_predictions.csv").read_text(encoding="utf-8")
    assert "less,1,1,0" in (tmp_path / "benchmark" / "by_defect_type.csv").read_text(encoding="utf-8")
    assert "slot02,1,1,0" in (tmp_path / "benchmark" / "by_slot.csv").read_text(encoding="utf-8")


def test_robustness_benchmark_groups_geometry_delta_by_gt_defect_type(tmp_path: Path) -> None:
    """Geometry evidence labels should not replace the human GT defect type buckets."""
    wrapper = _load_module("pipeline_robustness_benchmark_gt_defect", "pipeline/19_run_robustness_benchmark.py")
    defect_root = tmp_path / "defect"
    defect_root.mkdir(exist_ok=True)
    defect_path = defect_root / "less_1_2_slot02.png"
    defect_path.write_text("placeholder", encoding="utf-8")
    geometry_csv = tmp_path / "geometry_predictions.csv"
    geometry_csv.write_text(
        "\n".join(
            [
                "part_id,side,view,slot_id,branch,pred_label,score,threshold,defect_type,evidence_type,"
                "gt_defect_type,reason,source_path",
                f"less_1_2_slot02,top,uniform,slot02,geometry,1,10,5,geometry_delta,missing_mask,less,"
                f"shape diff,{defect_path}",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    args = wrapper.build_parser().parse_args(
        [
            "--defect-root",
            str(defect_root),
            "--geometry-csv",
            str(geometry_csv),
            "--output-dir",
            str(tmp_path / "benchmark"),
        ],
    )

    summary = wrapper.run_benchmark(args)

    assert summary["defect_total"] == 1
    assert summary["less_recall"] == 1.0
    by_defect = (tmp_path / "benchmark" / "by_defect_type.csv").read_text(encoding="utf-8")
    details = (tmp_path / "benchmark" / "robustness_details.csv").read_text(encoding="utf-8")
    assert "less,1,1,1" in by_defect
    assert "geometry_delta" not in by_defect
    assert "missing_mask" in details


def test_visualize_traditional_results_writes_contact_sheet(tmp_path: Path) -> None:
    """Stage 24 should render GT labels separately from operator evidence labels."""
    visualizer = _load_module("pipeline_visualize_traditional_results", "pipeline/24_visualize_traditional_results.py")
    image_path = tmp_path / "surface_1_1_slot01.png"
    image = np.zeros((60, 100, 3), dtype=np.uint8)
    image[15:45, 20:80] = (160, 160, 160)
    assert cv2.imwrite(str(image_path), image)
    cases_csv = tmp_path / "traditional_cases.csv"
    predictions_csv = tmp_path / "traditional_predictions.csv"
    cases_csv.write_text(
        "\n".join(
            [
                "case_id,label,part_id,side,view,slot_id,pred_label,positive_branches,gt_defect_type,source_path",
                f"surface_1_1_slot01,defect,surface_1_1,top,uniform,slot01,1,geometry,surface,{image_path}",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    predictions_csv.write_text(
        "\n".join(
            [
                "part_id,side,view,slot_id,branch,pred_label,score,threshold,defect_type,evidence_type,"
                "gt_defect_type,reason,source_path,evidence_path,status",
                f"surface_1_1,top,uniform,slot01,geometry,1,12,5,geometry_delta,missing_mask,surface,"
                f"geometry_template raw_delta=less evidence=missing_mask region=r02_c06 score=12.0000 "
                f"threshold=5.0000 dx=0 dy=0 iou=0.9,{image_path},,",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    args = visualizer.build_parser().parse_args(
        [
            "--predictions-csv",
            str(predictions_csv),
            "--cases-csv",
            str(cases_csv),
            "--output-dir",
            str(tmp_path / "visual"),
            "--report-name",
            "surface_review",
            "--mode",
            "defect-all",
        ],
    )

    outputs = visualizer.run_visualization(args)

    assert outputs == [tmp_path / "visual" / "surface_review_defect-all_page01.jpg"]
    assert outputs[0].is_file()
    localization_path = tmp_path / "visual" / "localization" / "surface_review_surface_1_1_slot01_slot01_localization.jpg"
    assert localization_path.is_file()
    localization = cv2.imread(str(localization_path))
    assert localization is not None
    assert localization[38, 81, 2] > localization[38, 81, 1] + 40


def test_robustness_benchmark_marks_unpredicted_invalid_as_not_evaluated(tmp_path: Path) -> None:
    """Invalid inputs without predictions should not be counted as successful invalid rejects."""
    wrapper = _load_module("pipeline_robustness_benchmark_invalid_missing", "pipeline/19_run_robustness_benchmark.py")
    normal_root = tmp_path / "normal_test"
    invalid_root = tmp_path / "invalid"
    normal_root.mkdir(exist_ok=True)
    invalid_root.mkdir(exist_ok=True)
    (normal_root / "shared_slot01.png").write_text("placeholder", encoding="utf-8")
    (invalid_root / "shared_slot01.png").write_text("placeholder", encoding="utf-8")
    geometry_csv = tmp_path / "geometry_predictions.csv"
    geometry_csv.write_text(
        "\n".join(
            [
                "source_path,image_path,slot,geometry_pred_label,geometry_score,geometry_threshold,geometry_type",
                f"{normal_root / 'shared_slot01.png'},,slot01,0,1.0,10.0,none",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    args = wrapper.build_parser().parse_args(
        [
            "--clean-normal-root",
            str(normal_root),
            "--invalid-root",
            str(invalid_root),
            "--geometry-csv",
            str(geometry_csv),
            "--output-dir",
            str(tmp_path / "benchmark"),
        ],
    )

    summary = wrapper.run_benchmark(args)
    details = (tmp_path / "benchmark" / "robustness_details.csv").read_text(encoding="utf-8")

    assert summary["invalid_total"] == 1
    assert summary["invalid_rejected"] == 0
    assert summary["invalid_reject_rate"] == 0.0
    assert summary["missing_prediction_count"] == 1
    assert summary["not_evaluated_missing_prediction_count"] == 1
    assert "not_evaluated/missing_prediction" in details
