# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the atomic BMW left-hand Demo publisher."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER
from bmw_inspection.lab.eight_view_demo import load_demo_config
import bmw_inspection.lab.left_demo_publisher as left_demo_publisher
from bmw_inspection.lab.left_demo_publisher import publish_left_demo


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _component_policy() -> dict[str, object]:
    policy = {
        "low_threshold": 0.2,
        "seed_threshold": 0.35,
        "p95_threshold": 0.35,
        "minimum_area": 8,
        "hard_peak_threshold": 0.9,
        "line_minimum_length": 10,
        "line_minimum_area": 4,
    }
    return {
        "schema_version": "bmw.efficientad_component_filter/1.0",
        "candidate_only": True,
        "score_source": "accepted_component_max_p95",
        "policies": {view: policy for view in VIEW_ORDER},
    }


@dataclass(frozen=True, slots=True)
class _LeftFixture:
    repo_root: Path
    output_run: Path
    output_config: Path
    source_run: Path
    yolo_checkpoint: Path

    @property
    def arguments(self) -> dict[str, Path]:
        return {
            "repo_root": self.repo_root,
            "output_run": self.output_run,
            "output_config": self.output_config,
        }


@pytest.fixture
def fixture(tmp_path: Path) -> _LeftFixture:
    repo = tmp_path / "repo"
    _write_json(repo / "configs/bmw/capture/bmw_4cam_eight_view_hdr_v1.json", {"capture": "left"})
    _write_json(repo / "configs/bmw/rois/bmw_left_normal_20260814_roi_v2.json", {"part_rois": {}})
    manifest = repo / "dataset/bmw_lab_prepared/bmw_left_normal_20260814_v2/manifests/dataset_manifest.csv"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("sample_id,view_id,source_path\n", encoding="utf-8")

    source_run = repo / "results/bmw_lab_one_click/bmw_left_normal_20260814_models_v3"
    _write_json(source_run / "run_report.json", {"status": "complete"})
    checkpoint_sha256: dict[str, str] = {}
    for view in VIEW_ORDER:
        template = source_run / "template" / view / "model.json"
        template.parent.mkdir(parents=True, exist_ok=True)
        template.write_text(json.dumps({"view_id": view, "threshold": 0.1}), encoding="utf-8")
        checkpoint = source_run / "efficientad" / view / "model.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(view.encode())
        checkpoint_sha256[view] = _sha256(checkpoint)

    mask = repo / "results/bmw_efficientad_manual_ignore_masks/bmw_left_normal_20260814_models_v3_mask_v1/index.json"
    _write_json(mask, {"mask": "left"})
    component_policy = repo / "configs/bmw/efficientad_component_filter_lab_v1.json"
    _write_json(component_policy, _component_policy())
    thresholds = source_run / "efficientad/component_score_analysis/part_thresholds.json"
    _write_json(
        thresholds,
        {
            "demo_only": True,
            "test_used_for_selection": True,
            "score_source": "accepted_component_max_p95",
            "source_csv": "component_scores.csv",
            "source_csv_sha256": "a" * 64,
            "mask_index_sha256": _sha256(mask),
            "component_policy_sha256": _sha256(component_policy),
            "thresholds": {view: 0.5 for view in VIEW_ORDER},
            "checkpoint_sha256_by_view": checkpoint_sha256,
        },
    )
    bright_report = repo / "results/bmw_bright_streak_rotated_retrain/bmw_left_normal52_no_streak1_20260814_v1/report.json"
    _write_json(bright_report, {"status": "complete"})
    rotated_roi = repo / "results/bmw_bright_streak_rotated_roi/bmw_left_20260814_v2/roi.json"
    _write_json(rotated_roi, {"roi": [1, 2, 3, 4]})
    yolo_checkpoint = repo / "results/bmw_lab_one_click/bmw_right_multisource_left_yolo_v1/yolo/train/weights/best.pt"
    yolo_checkpoint.parent.mkdir(parents=True)
    yolo_checkpoint.write_bytes(b"shared-left-yolo")

    return _LeftFixture(
        repo_root=repo,
        output_run=repo / "results/bmw_lab_one_click/bmw_left_normal_20260814_demo_v1",
        output_config=repo / "configs/bmw/experiments/bmw_eight_view_demo_left_normal_20260814_v1.json",
        source_run=source_run,
        yolo_checkpoint=yolo_checkpoint,
    )


def test_publishes_three_links_receipt_and_loadable_left_config(
    fixture: _LeftFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(left_demo_publisher, "_SHARED_YOLO_SHA256", _sha256(fixture.yolo_checkpoint))
    receipt = publish_left_demo(**fixture.arguments)

    assert receipt["status"] == "complete"
    assert (fixture.output_run / "template").resolve() == (fixture.source_run / "template").resolve()
    assert (fixture.output_run / "efficientad").resolve() == (fixture.source_run / "efficientad").resolve()
    assert (fixture.output_run / "yolo").resolve() == fixture.yolo_checkpoint.parent.parent.parent.resolve()
    composition = json.loads((fixture.output_run / "composition.json").read_text(encoding="utf-8"))
    assert composition["schema"] == "bmw.left_demo_composition/1.0"
    assert composition["yolo"]["sha256"] == _sha256(fixture.yolo_checkpoint)

    payload = json.loads(fixture.output_config.read_text(encoding="utf-8"))
    assert "template" not in payload
    assert "trusted_ok_reference" not in payload
    assert payload["bright_streak"]["engine"] == "tracked_profile_v3_manual_rotated_candidate"
    assert "weak_row_score_override" not in payload["bright_streak"]
    loaded = load_demo_config(fixture.output_config)
    assert loaded.roi_config == fixture.repo_root / "configs/bmw/rois/bmw_left_normal_20260814_roi_v2.json"
    assert loaded.training_run == fixture.output_run
    assert loaded.yolo_checkpoint == fixture.output_run / "yolo/train/weights/best.pt"
    assert loaded.efficientad_component_filter_artifact == (
        fixture.repo_root / "configs/bmw/efficientad_component_filter_lab_v1.json"
    )


def test_refuses_existing_left_destinations(fixture: _LeftFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(left_demo_publisher, "_SHARED_YOLO_SHA256", _sha256(fixture.yolo_checkpoint))
    publish_left_demo(**fixture.arguments)

    with pytest.raises(FileExistsError, match="refuse to overwrite"):
        publish_left_demo(**fixture.arguments)
