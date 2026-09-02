"""Focused contract tests for the editable BMW laboratory Demo profile."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER
from bmw_inspection.lab.eight_view_demo import load_demo_config


def _touch(path: Path, content: str = "asset") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return str(path)


def _simple_profile(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    capture = _touch(tmp_path / "capture.json", "{}")
    roi = _touch(tmp_path / "roi.json", "{}")
    manifest = _touch(tmp_path / "dataset.csv", "sample_id,view_id,source_path\n")
    mask = _touch(tmp_path / "mask_index.json", "{}")
    trusted = _touch(tmp_path / "reference_index.json", '{"references": []}')
    rotated_roi = _touch(tmp_path / "rotated_roi.json", "{}")
    component_filter_path = tmp_path / "component_filter.json"
    policy = {
        "low_threshold": 0.2,
        "seed_threshold": 0.35,
        "p95_threshold": 0.35,
        "minimum_area": 8,
        "hard_peak_threshold": 0.9,
        "line_minimum_length": 10,
        "line_minimum_area": 4,
    }
    component_filter = _touch(
        component_filter_path,
        json.dumps({"policies": {view: policy for view in VIEW_ORDER}}),
    )
    template_models = {
        view: _touch(tmp_path / "template" / view / "model.json", "{}")
        for view in VIEW_ORDER
    }
    efficientad_checkpoints = {
        view: _touch(tmp_path / "efficientad" / view / "model.ckpt")
        for view in VIEW_ORDER
    }
    yolo_checkpoint = _touch(tmp_path / "yolo" / "best.pt")
    profile: dict[str, object] = {
        "demo_id": "editable-lab-profile",
        "capture_config": capture,
        "roi_config": roi,
        "prepared_manifest": manifest,
        "result_root": str(tmp_path / "results"),
        "template": {
            "models": template_models,
            "thresholds": {view: 0.01 for view in VIEW_ORDER},
            "ignore_mask_index": mask,
        },
        "bright_streak": {
            "engine": "tracked_profile_v3_manual_rotated_roi",
            "rotated_roi": rotated_roi,
            "geometry": {
                "candidate_width": 5,
                "background_width": 10,
                "background_gap": 3,
                "smooth_window": 5,
                "max_step": 2,
                "step_penalty": 1.0,
            },
            "thresholds": {
                "strong_row_score": 124.025,
                "weak_row_score": 95.0,
                "min_presence_coverage_ratio": 0.37,
                "min_longest_run_ratio": 0.23,
                "max_gap_ratio": 0.55,
                "max_gap_count": 4,
            },
        },
        "efficientad": {
            "checkpoints": efficientad_checkpoints,
            "thresholds": {view: 0.55 for view in VIEW_ORDER},
            "threshold_source": "legacy_reuse_for_0820_candidate",
            "validation_status": "pending_independent_validation",
            "ignore_mask_index": mask,
            "component_filter": component_filter,
        },
        "yolo": {
            "checkpoint": yolo_checkpoint,
            "candidate_conf": 0.1,
            "final_threshold": 0.25,
            "imgsz": 640,
        },
        "trusted_ok_reference": {"index": trusted},
    }
    path = tmp_path / "demo.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    return path, profile


def test_lab_profile_loads_direct_paths_and_thresholds_without_sha(tmp_path: Path) -> None:
    path, _payload = _simple_profile(tmp_path)

    config = load_demo_config(path)

    assert config.template_models["front"] == tmp_path / "template/front/model.json"
    assert config.template_thresholds["front"] == pytest.approx(0.01)
    assert config.efficientad_checkpoints["back"] == tmp_path / "efficientad/back/model.ckpt"
    assert config.efficientad_thresholds["back"] == pytest.approx(0.55)
    assert config.efficientad_threshold_source == "legacy_reuse_for_0820_candidate"
    assert config.efficientad_validation_status == "pending_independent_validation"
    assert config.yolo_checkpoint == tmp_path / "yolo/best.pt"
    assert config.shared_ignore_mask_index == tmp_path / "mask_index.json"
    assert config.trusted_ok_reference_index == tmp_path / "reference_index.json"
    assert config.template_weighted_regions is None
    assert not any("sha256" in name for name in config.__dataclass_fields__)


def test_lab_profile_loads_optional_weighted_template_regions(tmp_path: Path) -> None:
    path, payload = _simple_profile(tmp_path)
    regions = tmp_path / "weighted_regions.json"
    regions.write_text(json.dumps({"views": {view: [] for view in VIEW_ORDER}}), encoding="utf-8")
    template = payload["template"]
    assert isinstance(template, dict)
    legacy_thresholds = dict(template["thresholds"])  # type: ignore[arg-type]
    template["weighted_regions"] = {
        "enabled": True,
        "weight": 3.0,
        "outside_weight": 0.5,
        "roi_config": regions.name,
        "thresholds": {view: 0.02 for view in VIEW_ORDER},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_demo_config(path)

    assert dict(config.template_thresholds) == legacy_thresholds
    assert config.template_weighted_regions is not None
    assert config.template_weighted_regions.enabled is True
    assert config.template_weighted_regions.weight == pytest.approx(3.0)
    assert config.template_weighted_regions.outside_weight == pytest.approx(0.5)
    assert config.template_weighted_regions.roi_config == regions
    assert config.template_weighted_regions.thresholds["front"] == pytest.approx(0.02)


def test_lab_profile_defaults_weighted_template_outside_to_one(tmp_path: Path) -> None:
    path, payload = _simple_profile(tmp_path)
    regions = tmp_path / "weighted_regions.json"
    regions.write_text(json.dumps({"views": {view: [] for view in VIEW_ORDER}}), encoding="utf-8")
    template = payload["template"]
    assert isinstance(template, dict)
    template["weighted_regions"] = {
        "enabled": True,
        "weight": 3.0,
        "roi_config": regions.name,
        "thresholds": {view: 0.02 for view in VIEW_ORDER},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_demo_config(path)

    assert config.template_weighted_regions is not None
    assert config.template_weighted_regions.outside_weight == pytest.approx(1.0)


def test_lab_profile_accepts_manual_threshold_change_without_rebinding(tmp_path: Path) -> None:
    path, payload = _simple_profile(tmp_path)
    template = payload["template"]
    efficientad = payload["efficientad"]
    yolo = payload["yolo"]
    assert isinstance(template, dict) and isinstance(efficientad, dict) and isinstance(yolo, dict)
    template["thresholds"]["front"] = 0.123  # type: ignore[index]
    efficientad["thresholds"]["front"] = 0.678  # type: ignore[index]
    yolo["final_threshold"] = 0.42
    path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_demo_config(path)

    assert config.template_thresholds["front"] == pytest.approx(0.123)
    assert config.efficientad_thresholds["front"] == pytest.approx(0.678)
    assert config.yolo_final_threshold == pytest.approx(0.42)


def test_lab_profile_rejects_missing_model_file(tmp_path: Path) -> None:
    path, payload = _simple_profile(tmp_path)
    template = payload["template"]
    assert isinstance(template, dict)
    template["models"]["front"] = str(tmp_path / "missing-model.json")  # type: ignore[index]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Template front.*不存在"):
        load_demo_config(path)
