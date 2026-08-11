"""Strict configuration tests for BMW six-view laboratory experiments."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bmw_inspection.lab.config import load_experiment_config
from bmw_inspection.lab.contracts import BranchName, ViewId


VIEWS = ("front", "front_left", "front_right", "back", "back_left", "back_right")
REPO_ROOT = Path(__file__).parents[4]


def _topology() -> dict[str, object]:
    return {
        "topology_id": "bmw-3cam-double-side-v1",
        "camera_slots": [
            {"slot_id": "center", "serial": "DA9805574", "front": "front", "back": "back"},
            {"slot_id": "left", "serial": "DA9625347", "front": "front_left", "back": "back_left"},
            {"slot_id": "right", "serial": "DB0968108", "front": "front_right", "back": "back_right"},
        ],
    }


def _profile() -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_id": "bmw-lab-v1",
        "topology_path": "topology.json",
        "capture": {
            "image_width": 4024,
            "image_height": 3036,
            "exposure": 4000.0,
            "gain": 0.0,
            "timeout_ms": 3000,
            "warmup_frames": 1,
        },
        "part_rois": {view: [0, 0, 4024, 3036] for view in VIEWS},
        "template": {
            "enabled": False,
            "groups": {
                view: {"model_path": None, "threshold": 0.2} for view in VIEWS
            },
        },
        "bright_streak": {"config_paths": {"front_left": "bright_streak_demo.json"}},
        "yolo": {
            "enabled": False,
            "checkpoint": None,
            "class_name": "defect",
            "candidate_conf": 0.25,
            "final_threshold": 0.5,
        },
        "patchcore": {
            "enabled": False,
            "checkpoints": {view: None for view in VIEWS},
            "thresholds": {view: 0.5 for view in VIEWS},
        },
        "required_for_ok": ["template", "bright_streak", "yolo", "patchcore"],
        "result_root": "results/bmw_lab_v1",
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_valid_files(tmp_path: Path) -> Path:
    _write_json(tmp_path / "topology.json", _topology())
    profile = tmp_path / "profile.json"
    _write_json(profile, _profile())
    return profile


def test_loads_disabled_learned_models_and_exact_topology(tmp_path: Path) -> None:
    config = load_experiment_config(_write_valid_files(tmp_path))

    assert tuple(config.topology.view_ids) == tuple(ViewId)
    assert tuple(slot.serial for slot in config.topology.camera_slots) == (
        "DA9805574",
        "DA9625347",
        "DB0968108",
    )
    assert config.yolo.enabled is False
    assert config.patchcore.enabled is False
    assert config.bright_streak.enabled_views == (ViewId.FRONT_LEFT,)
    assert config.bright_streak.config_paths[ViewId.FRONT_LEFT] == tmp_path / "bright_streak_demo.json"
    assert set(config.template.groups) == set(ViewId)
    assert config.yolo.candidate_conf == 0.25
    assert config.yolo.final_threshold == 0.5
    assert config.required_for_ok == frozenset(BranchName)


@pytest.mark.parametrize("mutation", ["topology_id", "unique_serial", "swapped_serials", "swapped_slots"])
def test_rejects_any_topology_that_does_not_match_the_fixed_fixture(tmp_path: Path, mutation: str) -> None:
    profile = _write_valid_files(tmp_path)
    topology = _topology()
    slots = topology["camera_slots"]
    assert isinstance(slots, list)
    center = slots[0]
    left = slots[1]
    assert isinstance(center, dict)
    assert isinstance(left, dict)

    if mutation == "topology_id":
        topology["topology_id"] = "bmw-3cam-double-side-v2"
    elif mutation == "unique_serial":
        center["serial"] = "UNEXPECTED-BUT-UNIQUE"
    elif mutation == "swapped_serials":
        center["serial"], left["serial"] = left["serial"], center["serial"]
    else:
        center["slot_id"], left["slot_id"] = left["slot_id"], center["slot_id"]

    _write_json(tmp_path / "topology.json", topology)
    with pytest.raises(ValueError, match="fixed BMW fixture topology"):
        load_experiment_config(profile)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_conf", -0.01),
        ("candidate_conf", 1.01),
        ("final_threshold", -0.01),
        ("final_threshold", 1.01),
    ],
)
def test_rejects_yolo_confidence_outside_unit_interval(tmp_path: Path, field: str, value: float) -> None:
    profile = _write_valid_files(tmp_path)
    payload = _profile()
    yolo = payload["yolo"]
    assert isinstance(yolo, dict)
    yolo[field] = value
    _write_json(profile, payload)

    with pytest.raises(ValueError, match=rf"yolo\.{field} must be between 0 and 1"):
        load_experiment_config(profile)


def test_rejects_yolo_candidate_conf_above_final_threshold(tmp_path: Path) -> None:
    profile = _write_valid_files(tmp_path)
    payload = _profile()
    yolo = payload["yolo"]
    assert isinstance(yolo, dict)
    yolo["candidate_conf"] = 0.75
    yolo["final_threshold"] = 0.5
    _write_json(profile, payload)

    with pytest.raises(ValueError, match="candidate_conf must not exceed yolo.final_threshold"):
        load_experiment_config(profile)


def test_shipped_profile_resolves_result_root_from_repository_root() -> None:
    config = load_experiment_config(REPO_ROOT / "configs/bmw/experiments/bmw_lab_v1.json")

    assert config.result_root == REPO_ROOT / "results/bmw_lab_v1"


def test_rejects_duplicate_json_key(tmp_path: Path) -> None:
    profile = _write_valid_files(tmp_path)
    profile.write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_experiment_config(profile)


def test_rejects_unknown_view_missing_required_view_and_out_of_bounds_roi(tmp_path: Path) -> None:
    profile = _write_valid_files(tmp_path)
    payload = _profile()
    rois = payload["part_rois"]
    assert isinstance(rois, dict)
    rois.pop("back_right")
    rois["unknown"] = [0, 0, 4024, 3036]
    _write_json(profile, payload)

    with pytest.raises(ValueError, match="part_rois"):
        load_experiment_config(profile)

    payload = _profile()
    rois = payload["part_rois"]
    assert isinstance(rois, dict)
    rois["front"] = [0, 0, 4025, 3036]
    _write_json(profile, payload)
    with pytest.raises(ValueError, match="part_rois.front"):
        load_experiment_config(profile)


def test_rejects_unknown_branch_and_nonfinite_threshold(tmp_path: Path) -> None:
    profile = _write_valid_files(tmp_path)
    payload = _profile()
    payload["required_for_ok"] = ["template", "not-a-branch"]
    _write_json(profile, payload)
    with pytest.raises(ValueError, match="unknown branch"):
        load_experiment_config(profile)

    payload = _profile()
    yolo = payload["yolo"]
    assert isinstance(yolo, dict)
    yolo["final_threshold"] = float("nan")
    _write_json(profile, payload)
    with pytest.raises(ValueError, match="finite"):
        load_experiment_config(profile)


def test_rejects_duplicate_camera_serial_and_enabled_missing_models(tmp_path: Path) -> None:
    profile = _write_valid_files(tmp_path)
    topology = _topology()
    slots = topology["camera_slots"]
    assert isinstance(slots, list)
    duplicate_slot = slots[2]
    assert isinstance(duplicate_slot, dict)
    duplicate_slot["serial"] = "DA9625347"
    _write_json(tmp_path / "topology.json", topology)
    with pytest.raises(ValueError, match="duplicate camera serial"):
        load_experiment_config(profile)

    _write_json(tmp_path / "topology.json", _topology())
    topology = _topology()
    slots = topology["camera_slots"]
    assert isinstance(slots, list)
    missing_view_slot = slots[2]
    assert isinstance(missing_view_slot, dict)
    missing_view_slot["front"] = "front_left"
    _write_json(tmp_path / "topology.json", topology)
    with pytest.raises(ValueError, match="six required views"):
        load_experiment_config(profile)

    _write_json(tmp_path / "topology.json", _topology())
    payload = _profile()
    yolo = payload["yolo"]
    assert isinstance(yolo, dict)
    yolo["enabled"] = True
    yolo["checkpoint"] = "missing.pt"
    _write_json(profile, payload)
    with pytest.raises(ValueError, match="yolo.checkpoint.*does not exist"):
        load_experiment_config(profile)

    payload = _profile()
    template = payload["template"]
    assert isinstance(template, dict)
    template["enabled"] = True
    _write_json(profile, payload)
    with pytest.raises(ValueError, match="template.groups.front.model_path.*required"):
        load_experiment_config(profile)

    payload = _profile()
    patchcore = payload["patchcore"]
    assert isinstance(patchcore, dict)
    patchcore["enabled"] = True
    _write_json(profile, payload)
    with pytest.raises(ValueError, match="patchcore.checkpoints.front.*required"):
        load_experiment_config(profile)
