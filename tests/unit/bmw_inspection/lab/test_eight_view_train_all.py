"""Minimal contracts for the BMW eight-view laboratory training orchestrator."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER
import pytest

from bmw_inspection.lab.eight_view_train_all import (
    LabTrainingConfig,
    _fit_bright_streak_thresholds,
    build_training_plan,
    run_training,
)


def test_lab_defaults_keep_efficientad_batch_one_and_yolo_batch_32(tmp_path: Path) -> None:
    config = LabTrainingConfig.defaults(tmp_path)

    assert config.views == VIEW_ORDER
    assert config.efficientad_batch == 1
    assert config.yolo_batch == 32
    assert config.efficientad_epochs == 30
    assert config.yolo_epochs == 100


def test_training_plan_is_fail_fast_and_covers_all_requested_branches(tmp_path: Path) -> None:
    config = LabTrainingConfig.defaults(tmp_path)

    plan = build_training_plan(config)

    assert [step.name for step in plan] == [
        "materialize",
        "template",
        "bright_streak",
        "efficientad",
        "yolo",
    ]
    assert all(step.fail_fast for step in plan)
    assert plan[-1].parameters["batch"] == 32
    assert plan[-1].parameters["model"] == str(config.yolo_checkpoint)
    assert plan[-2].parameters["views"] == list(VIEW_ORDER)


def test_dry_run_plan_does_not_require_output_directory(tmp_path: Path) -> None:
    config = LabTrainingConfig.defaults(tmp_path)

    plan = build_training_plan(config)

    assert not config.run_dir.exists()
    assert all(step.status == "planned" for step in plan)


def test_run_training_dry_run_never_calls_stage_handlers(tmp_path: Path) -> None:
    config = LabTrainingConfig.defaults(tmp_path)

    report = run_training(
        config,
        dry_run=True,
        stage_handlers={name: lambda _config: pytest.fail(name) for name in (step.name for step in build_training_plan(config))},
    )

    assert report["status"] == "dry_run"
    assert report["experimental_only"] is True
    assert not config.run_dir.exists()


def test_run_training_stops_after_first_failed_stage(tmp_path: Path) -> None:
    config = LabTrainingConfig.defaults(tmp_path)
    called: list[str] = []

    def ok(name: str):
        def handler(_config: LabTrainingConfig) -> dict[str, str]:
            called.append(name)
            return {"status": "complete"}

        return handler

    def fail(_config: LabTrainingConfig) -> dict[str, str]:
        called.append("template")
        raise RuntimeError("template failed")

    handlers = {step.name: ok(step.name) for step in build_training_plan(config)}
    handlers["template"] = fail

    with pytest.raises(RuntimeError, match="template failed"):
        run_training(config, stage_handlers=handlers)

    assert called == ["materialize", "template"]


def test_one_click_cli_uses_confirmed_laboratory_defaults() -> None:
    script = Path(__file__).resolve().parents[4] / "pipeline/bmw_lab_train_all.py"
    spec = importlib.util.spec_from_file_location("bmw_lab_train_all_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.build_parser().parse_args([])

    assert args.efficientad_epochs == 30
    assert args.yolo_epochs == 100
    assert args.yolo_batch == 32
    assert args.yolo_imgsz == 640

    all_normal_args = module.build_parser().parse_args(["--efficientad-all-normal-train"])
    assert all_normal_args.efficientad_all_normal_train is True


def test_all_normal_train_mode_disables_internal_efficientad_validation(tmp_path: Path) -> None:
    from bmw_inspection.lab import eight_view_train_all as training

    config = replace(LabTrainingConfig.defaults(tmp_path), efficientad_all_normal_train=True)

    options = training._efficientad_data_options(config)

    assert options == {
        "normal_test_dir": None,
        "test_split_mode": "none",
        "val_split_mode": "none",
        "run_test": False,
        "limit_val_batches": 0,
        "validation_status": "pending_external_validation",
    }
    efficientad_step = next(step for step in build_training_plan(config) if step.name == "efficientad")
    assert efficientad_step.parameters["all_normal_train"] is True
    assert efficientad_step.parameters["validation"] == "pending_external_validation"


def test_bright_streak_fit_keeps_explicit_continuity_gates() -> None:
    calibration = [
        {
            "label": "normal",
            "contrast_snr": 4.0,
            "coverage_ratio": 0.35,
            "longest_run_ratio": 0.34,
            "max_gap_ratio": 0.06,
            "gap_count": 1,
        },
        {
            "label": "normal",
            "contrast_snr": 3.7,
            "coverage_ratio": 0.33,
            "longest_run_ratio": 0.33,
            "max_gap_ratio": 0.07,
            "gap_count": 2,
        },
        {
            "label": "no_streak",
            "contrast_snr": 2.8,
            "coverage_ratio": 0.05,
            "longest_run_ratio": 0.03,
            "max_gap_ratio": 0.30,
            "gap_count": 2,
        },
    ]

    fitted = _fit_bright_streak_thresholds(calibration)

    assert fitted["min_longest_run_ratio"] == pytest.approx(0.33)
    assert fitted["max_gap_ratio"] == pytest.approx(0.07)
    assert fitted["max_gap_count"] == 2
    assert 2.8 < fitted["min_contrast_snr"] < 3.7
    assert 0.05 < fitted["min_coverage_ratio"] < 0.33
    assert fitted["balanced_accuracy"] == pytest.approx(1.0)
