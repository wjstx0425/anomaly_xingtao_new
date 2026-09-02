# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for sequential BMW right/left Template and EfficientAD training."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _config(module: object, tmp_path: Path):
    return module.BilateralNormalTrainingConfig(
        repo_root=tmp_path,
        right_prepared_root=tmp_path / "prepared/right",
        right_roi_config=tmp_path / "right_roi.json",
        left_prepared_root=tmp_path / "prepared/left",
        left_roi_config=tmp_path / "left_roi.json",
        training_root=tmp_path / "training",
        output_root=tmp_path / "results",
        right_training_id="right_training_v1",
        right_run_id="right_models_v1",
        left_training_id="left_training_v1",
        left_run_id="left_models_v1",
        efficientad_epochs=30,
        gpu=0,
        workers=8,
        seed=42,
    )


def test_builds_two_hand_configs_with_all_normal_efficientad_and_distinct_outputs(tmp_path: Path) -> None:
    from bmw_inspection.lab import bilateral_normal_training as training

    right, left = training.build_hand_configs(_config(training, tmp_path))

    assert (right.capture_scope, left.capture_scope) == ("right", "left")
    assert right.efficientad_all_normal_train is True
    assert left.efficientad_all_normal_train is True
    assert right.training_release != left.training_release
    assert right.run_dir != left.run_dir


def test_runs_right_then_left_with_train_stage_only(tmp_path: Path) -> None:
    from bmw_inspection.lab import bilateral_normal_training as training

    calls: list[tuple[str, bool, str]] = []

    def runner(config, *, dry_run: bool, stage: str):
        calls.append((config.capture_scope, dry_run, stage))
        return {"status": "dry_run" if dry_run else "complete"}

    report = training.run_bilateral_normal_training(
        _config(training, tmp_path),
        dry_run=True,
        runner=runner,
    )

    assert calls == [("right", True, "train"), ("left", True, "train")]
    assert report["status"] == "dry_run"
    assert [item["capture_scope"] for item in report["hands"]] == ["right", "left"]


def test_stops_before_left_when_right_training_fails(tmp_path: Path) -> None:
    from bmw_inspection.lab import bilateral_normal_training as training

    calls: list[str] = []

    def runner(config, *, dry_run: bool, stage: str):
        del dry_run, stage
        calls.append(config.capture_scope)
        raise RuntimeError("right failed")

    with pytest.raises(RuntimeError, match="right failed"):
        training.run_bilateral_normal_training(_config(training, tmp_path), runner=runner)

    assert calls == ["right"]


def test_cli_defaults_to_current_0820_releases_and_rois() -> None:
    script = Path(__file__).resolve().parents[4] / "pipeline/bmw_lab_train_bilateral_normal.py"
    spec = importlib.util.spec_from_file_location("bmw_lab_train_bilateral_normal_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.build_parser().parse_args([])

    assert args.right_prepared_root.name == "bmw_right_0820_v1"
    assert args.right_roi_config.name == "bmw_right_0820_v1.json"
    assert args.left_prepared_root.name == "bmw_left_0820_v1"
    assert args.left_roi_config.name == "bmw_left_0820_v1.json"
