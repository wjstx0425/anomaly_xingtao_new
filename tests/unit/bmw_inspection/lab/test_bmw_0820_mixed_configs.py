"""Contracts for the right/left 0820 mixed laboratory candidates."""

from __future__ import annotations

import json
from pathlib import Path

from bmw_inspection.views import VIEW_ORDER
from bmw_inspection.lab.eight_view_demo import load_demo_config
from bmw_inspection.lab.eight_view_demo_models import load_part_rois


ROOT = Path(__file__).resolve().parents[4]
CONFIGS = {
    "right": ROOT / "configs/bmw/experiments/bmw_eight_view_demo_right_0820_mixed_v1.json",
    "left": ROOT / "configs/bmw/experiments/bmw_eight_view_demo_left_0820_mixed_v1.json",
}


def _strings(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def test_active_profiles_and_trusted_ok_indexes_are_location_independent() -> None:
    for path in CONFIGS.values():
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert not any(Path(value).is_absolute() for value in _strings(payload) if "/" in value)
        trusted = path.parent / payload["trusted_ok_reference"]["index"]
        trusted_payload = json.loads(trusted.resolve().read_text(encoding="utf-8"))
        for row in trusted_payload["references"]:
            assert not Path(row["full_image_path"]).is_absolute()
            assert not Path(row["roi_image_path"]).is_absolute()
def test_0820_mixed_configs_use_0823_front_right_models_and_new_public_rois() -> None:
    for hand, path in CONFIGS.items():
        config = load_demo_config(path)
        assert config.roi_config == ROOT / f"configs/bmw/rois/bmw_{hand}_0820_v1.json"
        assert tuple(config.template_models) == VIEW_ORDER
        assert tuple(config.efficientad_checkpoints) == VIEW_ORDER
        for view in VIEW_ORDER:
            template_run = (
                f"bmw_{hand}_front_right_template_0823_v1"
                if view == "front_right"
                else f"bmw_{hand}_0820_template_efficientad_all_normal_models_v1"
            )
            assert config.template_models[view] == (
                ROOT
                / "results/bmw_lab_one_click"
                / template_run
                / f"template/{view}/model.json"
            ).resolve()
            efficientad_run = (
                f"bmw_{hand}_front_right_efficientad_0823_v2"
                if view == "front_right"
                else f"bmw_{hand}_0820_template_efficientad_all_normal_models_v1"
            )
            assert config.efficientad_checkpoints[view] == (
                ROOT
                / "results/bmw_lab_one_click"
                / efficientad_run
                / f"efficientad/{view}/model.ckpt"
            ).resolve()
        if hand == "left":
            assert config.efficientad_threshold_source == "field_ok_envelope_20260821"
            assert config.efficientad_validation_status == (
                "lab_only_8_train_normal_plus_4_field_ok_pending_independent_validation"
            )
        else:
            assert config.efficientad_threshold_source == "legacy_reuse_for_0820_candidate"
            assert config.efficientad_validation_status == "pending_independent_validation"
        assert config.efficientad_ignore_mask_index == (
            ROOT
            / "results/bmw_efficientad_manual_ignore_masks"
            / f"bmw_{hand}_0820_manual_ignore_v2/index.json"
        ).resolve()
        assert config.bright_streak_rotated_roi == (
            ROOT
            / "results/bmw_bright_streak_rotated_roi"
            / f"bmw_{hand}_0820_manual_v1/roi.json"
        ).resolve()
        assert config.yolo_checkpoint == (
            ROOT
            / "results/bmw_lab_one_click/bmw_right_multisource_left_yolo_0820_v2/yolo/train/weights/best.pt"
        ).resolve()

    right = load_demo_config(CONFIGS["right"])
    left = load_demo_config(CONFIGS["left"])
    assert right.template_ignore_mask_index == (
        ROOT / "results/bmw_efficientad_manual_ignore_masks/bmw_right_manual_ignore_v3/index.json"
    ).resolve()
    assert left.template_ignore_mask_index is None


def test_left_0820_keeps_template_strict_and_uses_field_tolerant_efficientad_thresholds() -> None:
    config = load_demo_config(CONFIGS["left"])

    assert {view: config.template_thresholds[view] for view in VIEW_ORDER[:3]} == {
        "front": 0.015,
        "front_left": 0.011,
        "front_right": 0.008,
    }
    assert {view: config.efficientad_thresholds[view] for view in VIEW_ORDER[:3]} == {
        "front": 0.39,
        "front_left": 0.40,
        "front_right": 0.43,
    }


def test_left_0820_uses_current_training_normal_trusted_ok_bank() -> None:
    config = load_demo_config(CONFIGS["left"])

    assert config.trusted_ok_reference_index == (
        ROOT / "dataset/bmw_trusted_ok_reference/bmw_left_0820_train_normal_v1/reference_index.json"
    ).resolve()


def test_right_0820_uses_current_training_normal_trusted_ok_bank() -> None:
    config = load_demo_config(CONFIGS["right"])

    assert config.trusted_ok_reference_index == (
        ROOT / "dataset/bmw_trusted_ok_reference/bmw_right_0820_train_normal_v1/reference_index.json"
    ).resolve()


def test_right_0820_front_efficientad_thresholds_use_original_values() -> None:
    config = load_demo_config(CONFIGS["right"])

    assert {view: config.efficientad_thresholds[view] for view in VIEW_ORDER[:3]} == {
        "front": 0.5502086162570001,
        "front_left": 0.5500145435330002,
        "front_right": 0.549995529652,
    }


def test_0820_model_inputs_match_each_new_roi_shape() -> None:
    import json

    for path in CONFIGS.values():
        config = load_demo_config(path)
        rois = load_part_rois(config.roi_config)
        for view in VIEW_ORDER:
            model = json.loads(config.template_models[view].read_text(encoding="utf-8"))
            x1, y1, x2, y2 = rois[view]
            assert (model["input_width"], model["input_height"]) == (x2 - x1, y2 - y1)


def test_0820_mixed_configs_use_hand_specific_weighted_template_regions() -> None:
    for hand, path in CONFIGS.items():
        config = load_demo_config(path)
        weighted = config.template_weighted_regions
        assert weighted is not None
        assert weighted.enabled is True
        assert weighted.weight == 3.0
        assert weighted.roi_config == (
            ROOT / f"configs/bmw/key_template/bmw_{hand}_0823_key_rois_v1.json"
        ).resolve()
        assert tuple(weighted.thresholds) == VIEW_ORDER
