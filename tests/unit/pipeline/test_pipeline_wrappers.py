# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for numbered pipeline wrapper helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


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
