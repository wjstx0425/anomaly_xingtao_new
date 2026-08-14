# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the atomic BMW V6 composite Demo publisher."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER
from bmw_inspection.lab.v6_demo_publisher import publish_v6_demo


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@dataclass(slots=True)
class _V6Fixture:
    repo_root: Path
    output_run: Path
    output_config: Path
    v5_config: Path
    yolo_sha256: str

    @property
    def arguments(self) -> dict[str, Path]:
        return {
            "repo_root": self.repo_root,
            "output_run": self.output_run,
            "output_config": self.output_config,
        }

    def remove_checkpoint(self, view: str) -> None:
        (self.repo_root / "results/bmw_lab_one_click/bmw_right_normal_20260814_models_v1/efficientad" / view / "model.ckpt").unlink()


@pytest.fixture
def fixture(tmp_path: Path) -> _V6Fixture:
    repo = tmp_path / "repo"
    public_roi = {"part_rois": {view: [1, 2, 30, 40] for view in VIEW_ORDER}}
    _write_json(repo / "configs/bmw/rois/bmw_right_hdr_eight_view_v1.json", public_roi)
    _write_json(repo / "configs/bmw/rois/bmw_right_normal_20260814_roi_v1.json", public_roi)

    mask = repo / "results/masks/index.json"
    _write_json(mask, {"mask": "v5"})
    bright_report = repo / "results/bmw_bright_streak_rotated_retrain/bmw_right_normal50_no_streak1_20260814_v2/report.json"
    _write_json(bright_report, {"status": "complete", "candidate_only": True})
    rotated_roi = repo / "results/bmw_bright_streak_rotated_roi/bmw_demo_20260813_164043_v3/roi.json"
    _write_json(rotated_roi, {"roi": [1, 2, 3, 4]})

    v5_run = repo / "results/bmw_lab_one_click/bmw_right_batch_20260810_21_v3_ng_evidence_demo_v1"
    yolo = v5_run / "yolo/train/weights/best.pt"
    yolo.parent.mkdir(parents=True)
    yolo.write_bytes(b"v5-yolo")
    yolo_sha256 = _sha256(yolo)

    template_asset = repo / "results/bmw_template_manual_ignore_ab/bmw_right_manual_ignore_v3_template_ab_v1/template_masked_thresholds.json"
    template_asset_payload = {
        "thresholds": {view: 0.01 for view in VIEW_ORDER},
        "views": {view: {"model_json_sha256": "old-template-sha"} for view in VIEW_ORDER},
        "manual_ignore_mask_index_sha256": _sha256(mask),
    }
    _write_json(template_asset, template_asset_payload)
    efficientad_asset = v5_run / "efficientad_thresholds_deployment_v3.json"
    _write_json(
        efficientad_asset,
        {
            "thresholds": {view: 0.55 for view in VIEW_ORDER},
            "base_thresholds": {view: 0.50 for view in VIEW_ORDER},
            "checkpoint_sha256_by_view": {view: "old-efficientad-sha" for view in VIEW_ORDER},
        },
    )

    v5_config = repo / "configs/bmw/experiments/bmw_eight_view_demo_v5_template_manual_ignore_mask_v1.json"
    _write_json(
        v5_config,
        {
            "schema_version": 1,
            "demo_id": "v5",
            "capture_config": "../capture.json",
            "roi_config": "../rois/bmw_right_hdr_eight_view_v1.json",
            "prepared_manifest": "../../../dataset/manifest.csv",
            "training_run": "../../../results/bmw_lab_one_click/bmw_right_batch_20260810_21_v3_ng_evidence_demo_v1",
            "result_root": "../../../results/v5-results",
            "bright_streak": {
                "engine": "tracked_profile_v3_manual_rotated_roi",
                "config": "../../../results/v5-bright.json",
                "config_sha256": "0" * 64,
                "rotated_roi": "../../../results/bmw_bright_streak_rotated_roi/bmw_demo_20260813_164043_v3/roi.json",
                "rotated_roi_sha256": _sha256(rotated_roi),
                "weak_row_score_override": 95.0,
            },
            "template": {
                "ignore_mask_index": "../../../results/masks/index.json",
                "ignore_mask_index_sha256": _sha256(mask),
                "threshold_artifact": "../../../results/bmw_template_manual_ignore_ab/bmw_right_manual_ignore_v3_template_ab_v1/template_masked_thresholds.json",
                "threshold_artifact_sha256": _sha256(template_asset),
            },
            "efficientad": {
                "threshold_artifact": "../../../results/bmw_lab_one_click/bmw_right_batch_20260810_21_v3_ng_evidence_demo_v1/efficientad_thresholds_deployment_v3.json",
                "threshold_artifact_sha256": _sha256(efficientad_asset),
                "ignore_mask_index": "../../../results/masks/index.json",
                "ignore_mask_index_sha256": _sha256(mask),
            },
            "yolo": {"candidate_conf": 0.1, "final_threshold": 0.25, "imgsz": 640},
        },
    )

    new_run = repo / "results/bmw_lab_one_click/bmw_right_normal_20260814_models_v1"
    _write_json(new_run / "run_report.json", {"status": "complete"})
    _write_json(new_run / "efficientad/score_analysis/part_thresholds.json", {"thresholds": {view: 0.5 for view in VIEW_ORDER}})
    for view in VIEW_ORDER:
        template = new_run / "template" / view / "model.json"
        template.parent.mkdir(parents=True, exist_ok=True)
        template.write_text(json.dumps({"view": view}), encoding="utf-8")
        _write_json(new_run / "template" / view / "metrics.json", {"view": view})
        checkpoint = new_run / "efficientad" / view / "model.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(view.encode())
        _write_json(new_run / "efficientad" / view / "metrics.json", {"view": view})

    return _V6Fixture(
        repo_root=repo,
        output_run=repo / "results/bmw_v6_composite",
        output_config=repo / "configs/bmw/experiments/bmw_eight_view_demo_v6_right_normal_20260814_v1.json",
        v5_config=v5_config,
        yolo_sha256=yolo_sha256,
    )


def test_rejects_incomplete_eight_view_training(fixture: _V6Fixture) -> None:
    fixture.remove_checkpoint("back_secondary")

    with pytest.raises(ValueError, match="八视角"):
        publish_v6_demo(**fixture.arguments)


def test_rebinds_sha_but_preserves_threshold_values(fixture: _V6Fixture) -> None:
    report = publish_v6_demo(**fixture.arguments)

    assert report["template_thresholds_unchanged"] is True
    assert report["efficientad_thresholds_unchanged"] is True
    assert report["yolo_sha256"] == fixture.yolo_sha256
    assert (fixture.output_run / "template").resolve() == (
        fixture.repo_root / "results/bmw_lab_one_click/bmw_right_normal_20260814_models_v1/template"
    ).resolve()
    assert (fixture.output_run / "efficientad").resolve() == (
        fixture.repo_root / "results/bmw_lab_one_click/bmw_right_normal_20260814_models_v1/efficientad"
    ).resolve()
    composition = json.loads((fixture.output_run / "composition.json").read_text(encoding="utf-8"))
    assert composition["yolo"]["sha256"] == fixture.yolo_sha256
    template = json.loads((fixture.output_run / "template_thresholds_v5_rebound.json").read_text(encoding="utf-8"))
    efficientad = json.loads((fixture.output_run / "efficientad_thresholds_v5_rebound.json").read_text(encoding="utf-8"))
    assert template["thresholds"] == {view: 0.01 for view in VIEW_ORDER}
    assert efficientad["thresholds"] == {view: 0.55 for view in VIEW_ORDER}
    assert template["thresholds_recalibrated"] is False
    assert efficientad["thresholds_recalibrated"] is False
    assert all(item["model_json_sha256"] != "old-template-sha" for item in template["views"].values())
    assert all(value != "old-efficientad-sha" for value in efficientad["checkpoint_sha256_by_view"].values())


def test_keeps_v5_bytes_unchanged_and_refuses_existing_destinations(fixture: _V6Fixture) -> None:
    before = fixture.v5_config.read_bytes()

    publish_v6_demo(**fixture.arguments)

    assert fixture.v5_config.read_bytes() == before
    config = json.loads(fixture.output_config.read_text(encoding="utf-8"))
    assert config["roi_config"] == "../rois/bmw_right_hdr_eight_view_v1.json"
    assert config["bright_streak"]["engine"] == "tracked_profile_v3_manual_rotated_candidate"
    with pytest.raises(FileExistsError, match="refuse to overwrite"):
        publish_v6_demo(**fixture.arguments)
