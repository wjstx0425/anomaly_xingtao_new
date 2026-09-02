# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the BMW left-hand normal-only training run."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from types import SimpleNamespace

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _left_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    prepared = tmp_path / "prepared"
    _write_json(prepared / "report.json", {"capture_scope": "left", "release_status": "published"})
    roi = _write_json(tmp_path / "left_roi.json", {"capture_scope": "left"})
    mask = _write_json(tmp_path / "mask_index.json", {"capture_scope": "left"})
    policy = _write_json(tmp_path / "component_policy.json", {"capture_scope": "left"})
    return prepared, roi, mask, policy


def _config(module: object, tmp_path: Path):
    prepared, roi, mask, policy = _left_inputs(tmp_path)
    return module.LeftNormalTrainingConfig(
        repo_root=tmp_path,
        prepared_root=prepared,
        roi_config=roi,
        training_root=tmp_path / "training",
        training_id="left_roi_v1",
        output_root=tmp_path / "results",
        run_id="left_normal_v1",
        mask_index=mask,
        component_policy=policy,
    )


def test_plan_has_the_exact_left_normal_stage_order_without_yolo_or_bright_streak(tmp_path: Path) -> None:
    from bmw_inspection.lab import left_normal_training as training

    plan = training.build_left_normal_plan(_config(training, tmp_path))

    assert tuple(step.name for step in plan) == (
        "materialize",
        "template",
        "efficientad",
        "score_component_maps",
        "calibrate_component_thresholds",
    )
    assert all("yolo" not in step.name and "bright" not in step.name for step in plan)


def test_all_normal_train_mode_is_checkpoint_only_until_external_validation(tmp_path: Path) -> None:
    from bmw_inspection.lab import left_normal_training as training

    config = replace(_config(training, tmp_path), efficientad_all_normal_train=True)
    plan = training.build_left_normal_plan(config)

    efficientad_step = next(step for step in plan if step.name == "efficientad")
    assert efficientad_step.parameters["all_normal_train"] is True
    with pytest.raises(ValueError, match="all-normal.*stage train.*external validation"):
        training.run_left_normal_training(config, dry_run=True, stage="all")


def test_dry_run_refuses_any_non_left_capture_scope_before_gpu_work(tmp_path: Path) -> None:
    from bmw_inspection.lab import left_normal_training as training

    config = _config(training, tmp_path)
    _write_json(config.prepared_root / "report.json", {"capture_scope": "right", "release_status": "published"})

    with pytest.raises(ValueError, match="capture_scope.*left"):
        training.run_left_normal_training(config, dry_run=True)


def test_right_capture_scope_accepts_matching_prepared_release_and_roi(tmp_path: Path) -> None:
    from bmw_inspection.lab import left_normal_training as training

    config = replace(_config(training, tmp_path), capture_scope="right")
    manifest = config.prepared_root / "manifests/dataset_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("sample_id\n", encoding="utf-8")
    _write_json(
        config.prepared_root / "report.json",
        {"capture_scope": "right", "dataset_id": config.prepared_root.name, "release_status": "published"},
    )
    _write_json(
        config.roi_config,
        {
            "schema_version": 1,
            "coordinate_system": "pixel_xyxy_half_open",
            "dataset_id": config.prepared_root.name,
            "source_manifest": str(manifest),
            "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "representative_sample_id": "right-normal-001",
            "image_width": 10,
            "image_height": 10,
            "part_rois": {view: [0, 0, 10, 10] for view in VIEW_ORDER},
        },
    )

    report = training.run_left_normal_training(config, dry_run=True, stage="train")

    assert report["status"] == "dry_run"
    assert report["capture_scope"] == "right"


def test_template_candidate_thresholds_bind_all_eight_model_json_hashes(tmp_path: Path) -> None:
    from bmw_inspection.lab import left_normal_training as training

    models = {
        view: _write_json(tmp_path / "template" / view / "model.json", {"view": view}) for view in VIEW_ORDER
    }

    artifact = training.build_template_normal_threshold_artifact(
        models,
        {view: 0.1 + index / 100 for index, view in enumerate(VIEW_ORDER)},
        calibration_row_count=24,
    )

    assert artifact["candidate_only"] is True
    assert artifact["fit_split"] == "calibration"
    assert artifact["defect_metrics"] == "not_evaluated"
    assert artifact["model_json_sha256_by_view"] == {
        view: hashlib.sha256(models[view].read_bytes()).hexdigest() for view in VIEW_ORDER
    }


def test_component_threshold_artifact_binds_all_required_left_provenance(tmp_path: Path) -> None:
    from bmw_inspection.lab import left_normal_training as training

    checkpoints = {}
    for view in VIEW_ORDER:
        checkpoint = tmp_path / "efficientad" / view / "model.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(view.encode("utf-8"))
        checkpoints[view] = checkpoint
    mask = _write_json(tmp_path / "mask_index.json", {"capture_scope": "left"})
    policy = _write_json(tmp_path / "policy.json", {"capture_scope": "left"})

    artifact = training.build_component_threshold_artifact(
        checkpoint_paths=checkpoints,
        mask_index=mask,
        component_policy=policy,
        score_source="efficientad_component_p95_v1",
        sample_count=40,
        fpr_resolution=1 / 40,
        thresholds={view: 0.2 for view in VIEW_ORDER},
    )

    assert artifact["target_part_fpr"] == 0.05
    assert artifact["score_source"] == "efficientad_component_p95_v1"
    assert artifact["sample_count"] == 40
    assert artifact["fpr_resolution"] == 1 / 40
    assert artifact["mask_index_sha256"] == hashlib.sha256(mask.read_bytes()).hexdigest()
    assert artifact["component_policy_sha256"] == hashlib.sha256(policy.read_bytes()).hexdigest()
    assert artifact["checkpoint_sha256_by_view"] == {
        view: hashlib.sha256(checkpoints[view].read_bytes()).hexdigest() for view in VIEW_ORDER
    }


def test_run_uses_injected_handlers_in_order(tmp_path: Path) -> None:
    from bmw_inspection.lab import left_normal_training as training

    config = _config(training, tmp_path)
    calls: list[str] = []

    def handler(name: str):
        def run(_config, _state):
            calls.append(name)
            return {"stage": name}

        return run

    report = training.run_left_normal_training(
        config,
        stage_handlers={
            name: handler(name)
            for name in (
                "materialize",
                "template",
                "efficientad",
                "score_component_maps",
                "calibrate_component_thresholds",
            )
        },
    )

    assert calls == [
        "materialize",
        "template",
        "efficientad",
        "score_component_maps",
        "calibrate_component_thresholds",
    ]
    assert [step["name"] for step in report["steps"]] == calls


def test_train_stage_uses_real_default_handlers_without_mask_or_policy(tmp_path: Path, monkeypatch) -> None:
    from bmw_inspection.lab import left_normal_training as training

    config = _config(training, tmp_path)
    config = training.LeftNormalTrainingConfig(
        repo_root=config.repo_root, prepared_root=config.prepared_root, roi_config=config.roi_config,
        training_root=config.training_root, training_id=config.training_id, output_root=config.output_root,
        run_id=config.run_id,
    )
    calls: list[str] = []
    monkeypatch.setattr(training, "materialize_left_normal_data", lambda _config: calls.append("materialize") or {})
    monkeypatch.setattr(training, "train_left_normal_templates", lambda _config: calls.append("template") or {})
    monkeypatch.setattr(training, "_efficientad_stage", lambda _config: calls.append("efficientad") or {})

    report = training.run_left_normal_training(config, stage="train")

    assert calls == ["materialize", "template", "efficientad"]
    assert [step["name"] for step in report["steps"]] == calls


def test_component_calibration_rescores_all_eight_views_with_an_injected_predictor(tmp_path: Path, monkeypatch) -> None:
    from bmw_inspection.lab import left_normal_training as training
    from bmw_inspection.lab.efficientad_component_filter import ComponentFilterPolicy

    config = _config(training, tmp_path)
    for view in VIEW_ORDER:
        image_path = config.training_release / "efficientad" / view / "normal_test" / "part-a" / "images" / f"session__part-a_000001__{view}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(image_path), np.zeros((5, 5), dtype=np.uint8))
        checkpoint = config.run_dir / "efficientad" / view / "model.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(view.encode())
    policy = ComponentFilterPolicy(0.3, 0.5, 0.65, 4, 0.9, 5, 3)
    monkeypatch.setattr(
        "bmw_inspection.lab.efficientad_ignore_mask.load_ignore_mask_asset",
        lambda *_args, **_kwargs: SimpleNamespace(masks={view: np.zeros((5, 5), dtype=np.uint8) for view in VIEW_ORDER}, index_sha256="a" * 64),
    )
    monkeypatch.setattr(
        "bmw_inspection.lab.eight_view_demo._load_efficientad_component_policies",
        lambda _path: {view: policy for view in VIEW_ORDER},
    )

    score = training.score_component_maps(
        config, {}, predictor_factory=lambda _checkpoint: lambda _image: (0.9, True, np.pad(np.array([[0.55]], dtype=np.float32), ((2, 2), (2, 2))))
    )
    csv_text = Path(score["scores_csv"]).read_text(encoding="utf-8")
    assert csv_text.count("\n") == 9
    assert ",0," in csv_text
    artifact = training.calibrate_component_thresholds(config, {"score_component_maps": score})

    assert artifact["score_source"] == "accepted_component_max_p95"
    assert artifact["source_csv"] == score["scores_csv"]
    assert Path(artifact["threshold_asset"]).is_file()
