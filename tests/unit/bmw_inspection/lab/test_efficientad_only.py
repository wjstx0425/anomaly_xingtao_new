"""Contracts for the 21:00 BMW EfficientAD-only trainer."""

from __future__ import annotations

import importlib.util
import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest

from bmw_inspection.lab import eight_view_train_all as train_all
from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER

REPO_ROOT = Path(__file__).resolve().parents[4]


def _config(tmp_path: Path) -> train_all.LabTrainingConfig:
    return replace(
        train_all.LabTrainingConfig.defaults(tmp_path),
        training_root=tmp_path / "dataset/bmw_lab_training",
        training_id="bmw_right_batch_20260810_21_roi_v1",
        output_root=tmp_path / "results/bmw_lab_one_click",
        run_id="bmw_right_batch_20260810_21_efficientad_v1",
    )


def test_efficientad_only_plan_selects_no_other_training_stage(tmp_path: Path) -> None:
    planner = getattr(train_all, "build_efficientad_only_plan", None)

    assert planner is not None, "EfficientAD-only planning entrypoint must exist"
    plan = planner(_config(tmp_path))

    assert [step.name for step in plan] == ["efficientad", "score_normal_test", "calibrate_normal_thresholds"]
    assert plan[0].parameters == {
        "views": list(VIEW_ORDER),
        "model_size": "small",
        "batch": 1,
        "epochs": 30,
        "image_size": [256, 256],
        "seed": 42,
    }


def test_efficientad_only_runner_scores_held_out_normals_then_fits_thresholds(tmp_path: Path) -> None:
    runner = getattr(train_all, "run_efficientad_only_training", None)
    assert runner is not None, "EfficientAD-only runner must exist"
    parameters = inspect.signature(runner).parameters
    assert {"score_handler", "threshold_handler"}.issubset(parameters)
    calls: list[str] = []

    def efficientad(_config: train_all.LabTrainingConfig) -> dict[str, str]:
        calls.append("efficientad")
        return {"status": "complete"}

    def score(_config: train_all.LabTrainingConfig) -> dict[str, str]:
        calls.append("score_normal_test")
        return {"status": "complete", "scores_csv": "scores.csv"}

    def threshold(_config: train_all.LabTrainingConfig, score_report: dict[str, str]) -> dict[str, str]:
        calls.append("calibrate_normal_thresholds")
        assert score_report["scores_csv"] == "scores.csv"
        return {"status": "complete", "threshold_asset": "part_thresholds.json"}

    report = runner(
        _config(tmp_path),
        stage_handler=efficientad,
        score_handler=score,
        threshold_handler=threshold,
    )

    assert calls == ["efficientad", "score_normal_test", "calibrate_normal_thresholds"]
    assert [step["name"] for step in report["steps"]] == calls


def test_efficientad_only_runner_never_enters_unselected_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = getattr(train_all, "run_efficientad_only_training", None)
    assert runner is not None, "EfficientAD-only runner must exist"
    calls: list[str] = []

    def forbidden(_config: train_all.LabTrainingConfig) -> dict[str, str]:
        pytest.fail("EfficientAD-only run entered an unselected stage")

    def efficientad(_config: train_all.LabTrainingConfig) -> dict[str, str]:
        calls.append("efficientad")
        return {"status": "complete"}

    for stage in ("_materialize_stage", "_template_stage", "_bright_streak_stage", "_yolo_stage"):
        monkeypatch.setattr(train_all, stage, forbidden)
    monkeypatch.setattr(train_all, "_efficientad_stage", efficientad)

    report = runner(
        _config(tmp_path),
        score_handler=lambda _config: {"status": "complete", "scores_csv": "scores.csv"},
        threshold_handler=lambda _config, _scores: {"status": "complete"},
    )

    assert calls == ["efficientad"]
    assert [step["name"] for step in report["steps"]] == [
        "efficientad",
        "score_normal_test",
        "calibrate_normal_thresholds",
    ]


def test_efficientad_only_preflight_ignores_yolo_ready_release_gate(tmp_path: Path) -> None:
    preflight = getattr(train_all, "preflight_efficientad_only", None)
    assert preflight is not None, "EfficientAD-only preflight must exist"
    config = _config(tmp_path)
    release = config.training_release
    release.mkdir(parents=True)
    (release / "report.json").write_text(json.dumps({"yolo_training_ready": False}), encoding="utf-8")
    for view in VIEW_ORDER:
        for split in ("normal", "normal_test"):
            image_dir = release / "efficientad" / view / split
            image_dir.mkdir(parents=True)
            (image_dir / "sample.png").write_bytes(b"placeholder")

    report = preflight(config)

    assert report["training_release"] == str(release)
    assert report["yolo_training_ready_checked"] is False
    assert report["views"] == list(VIEW_ORDER)


def test_efficientad_only_writes_a_21_part_normal_only_threshold_asset(tmp_path: Path) -> None:
    calibrate = getattr(train_all, "_calibrate_efficientad_normal_thresholds", None)
    assert calibrate is not None, "normal-only threshold calibration stage must exist"
    config = _config(tmp_path)
    score_dir = config.run_dir / "efficientad" / "score_analysis"
    score_dir.mkdir(parents=True)
    score_csv = score_dir / "efficientad_normal_test_scores.csv"
    rows = ["part_id,view_id,label,score,image_path"]
    for part_index in range(21):
        for view_index, view in enumerate(VIEW_ORDER):
            part_id = f"bmw_right_normal_group{part_index:03d}"
            image_path = score_dir / f"session__{part_id}_000001__{view}.png"
            rows.append(f"{part_id},{view},normal,{0.1 + part_index / 1000 + view_index / 10000},{image_path}")
            checkpoint = config.run_dir / "efficientad" / view / "model.ckpt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(view.encode("utf-8"))
    score_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")

    report = calibrate(config, {"scores_csv": str(score_csv)})
    payload = json.loads(Path(report["threshold_asset"]).read_text(encoding="utf-8"))

    assert payload["calibration_source"] == "normal_test_only"
    assert payload["normal_part_count"] == 21
    assert payload["allowed_normal_false_positive_count"] == 1
    assert payload["normal_false_positive_count"] <= 1
    assert payload["defect_metrics"] == "not_evaluated"
    assert set(payload["thresholds"]) == set(VIEW_ORDER)


def test_efficientad_only_runner_refuses_existing_output_directory(tmp_path: Path) -> None:
    runner = getattr(train_all, "run_efficientad_only_training", None)
    assert runner is not None, "EfficientAD-only runner must exist"
    config = _config(tmp_path)
    config.run_dir.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="new --run-id"):
        runner(config, stage_handler=lambda _config: {"status": "complete"})


def test_efficientad_only_cli_defaults_to_the_published_21_point_release() -> None:
    script = REPO_ROOT / "pipeline/bmw_lab_train_efficientad_only.py"
    assert script.is_file(), "EfficientAD-only CLI must exist"
    spec = importlib.util.spec_from_file_location("bmw_lab_train_efficientad_only_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.build_parser().parse_args([])

    assert args.training_id == "bmw_right_batch_20260810_21_roi_v1"
    assert args.run_id == "bmw_right_batch_20260810_21_efficientad_v1"
    assert args.efficientad_epochs == 30
    assert args.gpu == 0
    assert args.seed == 42
