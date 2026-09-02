"""Contracts for single-view BMW Template training with a fixed threshold."""

from __future__ import annotations

import csv
import importlib.util
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from bmw_inspection.lab import eight_view_train_all as train_all


REPO_ROOT = Path(__file__).resolve().parents[4]


def _config(tmp_path: Path) -> train_all.LabTrainingConfig:
    return replace(
        train_all.LabTrainingConfig.defaults(tmp_path),
        training_root=tmp_path / "dataset/bmw_lab_training",
        training_id="bmw_right_front_right_template_0823_v1",
        output_root=tmp_path / "results/bmw_lab_one_click",
        run_id="bmw_right_front_right_template_0823_v1",
        views=("front_right",),
    )


def _template_release(config: train_all.LabTrainingConfig) -> None:
    image_dir = config.training_release / "crops/front_right"
    image_dir.mkdir(parents=True)
    rows: list[dict[str, str]] = []
    for index, split in enumerate(("train", "train", "train", "calibration", "final_test")):
        image_path = image_dir / f"sample-{index}.png"
        assert cv2.imwrite(str(image_path), np.full((8, 10, 3), 20 + index, dtype=np.uint8))
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "part_id": f"part-{index}",
                "view_id": "front_right",
                "image_path": str(image_path),
                "split": split,
                "label": "normal",
            }
        )
    manifest = config.training_release / "template/trainer_manifest.csv"
    manifest.parent.mkdir(parents=True)
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_template_only_plan_and_runner_select_no_other_stage(tmp_path: Path) -> None:
    config = _config(tmp_path)
    calls: list[float] = []

    plan = train_all.build_template_only_plan(config, threshold=0.012455999851226807)
    report = train_all.run_template_only_training(
        config,
        threshold=0.012455999851226807,
        stage_handler=lambda _config, threshold: calls.append(threshold) or {"status": "complete"},
    )

    assert [step.name for step in plan] == ["template"]
    assert plan[0].parameters == {
        "views": ["front_right"],
        "template_count": 5,
        "fixed_threshold": 0.012455999851226807,
    }
    assert calls == [0.012455999851226807]
    assert [step["name"] for step in report["steps"]] == ["template"]


def test_template_only_preflight_accepts_one_selected_view(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _template_release(config)

    report = train_all.preflight_template_only(config)

    assert report["views"] == ["front_right"]
    assert report["image_counts"] == {"front_right": 5}


def test_template_only_cli_keeps_selected_view_and_fixed_threshold() -> None:
    script = REPO_ROOT / "pipeline/bmw_lab_train_template_only.py"
    spec = importlib.util.spec_from_file_location("bmw_lab_train_template_only_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.build_parser().parse_args(
        [
            "--training-id",
            "bmw_left_front_right_template_0823_v1",
            "--run-id",
            "bmw_left_front_right_template_0823_v1",
            "--view",
            "front_right",
            "--threshold",
            "0.008",
        ]
    )
    config = module._config_from_args(args)

    assert config.views == ("front_right",)
    assert args.threshold == pytest.approx(0.008)
