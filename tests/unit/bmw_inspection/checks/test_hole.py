# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Synthetic contract tests; thresholds here are not production acceptance limits."""

import json

import cv2
import numpy as np
import pytest

from bmw_inspection.checks.hole import inspect_hole, load_config


@pytest.fixture
def sample(tmp_path):
    mask = np.zeros((60, 60), dtype=np.uint8)
    cv2.circle(mask, (30, 30), 15, 255, -1)
    assert cv2.imwrite(str(tmp_path / "expected.png"), mask)
    config = {
        "schema_version": 1, "status": "ready", "part_id": "SYNTHETIC_ONLY",
        "hand": "left", "feature_id": "hole_1", "view_id": "front_left",
        "source_channel": "fused", "pixel_channel": "gray",
        "reference_image_size": [160, 100], "coordinate_space": "full_image",
        "roi_xyxy": [20, 20, 80, 80], "expected_mask": "expected.png",
        "registration": {"mode": "fixture_fixed", "validated": True, "evidence": "synthetic fixture"},
        "association": {"exclusive_search_confirmed": True, "evidence": "synthetic isolated ROI"},
        "segmentation": {"threshold": 128, "polarity": "bright", "stability_delta": 10,
                         "max_changed_fraction": 0.05},
        "limits": {"pass_min_open_fraction": 0.95, "ng_max_open_fraction": 0.6,
                   "pass_max_area_difference": 0.05, "ng_min_area_difference": 0.4,
                   "pass_max_shape_difference": 0.05, "ng_min_shape_difference": 0.4,
                   "pass_max_position_px": 1.0, "ng_min_position_px": 6.0},
        "validation_evidence": "Synthetic software test only; not a real BMW reference",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    image = np.full((100, 160, 3), 20, dtype=np.uint8)
    image[20:80, 20:80][mask > 0] = 230
    return image, config, path


def run(sample, image=None, **changes):
    source, _, path = sample
    kwargs = {"part_id": "SYNTHETIC_ONLY", "view_id": "front_left", "source_channel": "fused",
              "source_kind": "fused_only", "observation": {"raw_unmasked": True,
              "fixture_verified": True, "observable": True, "evidence": "synthetic fixture"},
              "synthesized_test_data": True}
    kwargs.update(changes)
    return inspect_hole(source if image is None else image, load_config(path), **kwargs)


def test_normal_open_hole_and_input_preservation(sample):
    original = sample[0].copy()
    result = run(sample)
    assert result.payload["status"] == "PASS"
    assert result.payload["metrics"]["open_fraction"] == pytest.approx(1)
    assert result.payload["metrics"]["shape_difference"] == pytest.approx(0)
    assert result.payload["metrics"]["position_deviation_px"] == pytest.approx(0)
    assert result.images
    np.testing.assert_array_equal(sample[0], original)


def test_absent_hole_has_no_fabricated_process_cause(sample):
    result = run(sample, np.full_like(sample[0], 20))
    assert result.payload["status"] == "NG"
    assert result.payload["metrics"]["open_fraction"] == 0
    assert result.payload["metrics"]["position_deviation_px"] is None
    assert "孔未正常开放，原因待确认" in json.dumps(result.payload, ensure_ascii=False)


def test_partial_blockage_is_ng(sample):
    image = sample[0].copy()
    image[:, :50] = 20
    result = run(sample, image)
    assert result.payload["status"] == "NG"
    assert 0 < result.payload["metrics"]["open_fraction"] < 0.6


def test_neighbor_cannot_substitute_missing_target(sample):
    image = np.full_like(sample[0], 20)
    cv2.circle(image, (115, 50), 15, (230, 230, 230), -1)
    result = run(sample, image)
    assert result.payload["status"] == "NG"
    assert result.payload["metrics"]["open_fraction"] == 0


def test_shifted_target_reports_position_difference(sample):
    image = np.full_like(sample[0], 20)
    cv2.circle(image, (59, 50), 15, (230, 230, 230), -1)
    result = run(sample, image)
    assert result.payload["status"] == "NG"
    assert result.payload["metrics"]["position_deviation_px"] == pytest.approx(9)


def test_reflection_like_threshold_instability_is_review(sample):
    image = sample[0].copy()
    image[image == 230] = 130
    assert run(sample, image).payload["status"] == "REVIEW"


@pytest.mark.parametrize("field", ["fixture_verified", "observable"])
def test_fixture_occlusion_or_registration_uncertainty_is_review(sample, field):
    observation = {"raw_unmasked": True, "fixture_verified": True, "observable": True,
                   "evidence": "synthetic occlusion/uncertain registration"}
    observation[field] = False
    assert run(sample, observation=observation).payload["status"] == "REVIEW"


def test_missing_observation_evidence_is_review(sample):
    observation = {"raw_unmasked": True, "fixture_verified": True, "observable": True}
    assert run(sample, observation=observation).payload["status"] == "REVIEW"


@pytest.mark.parametrize("changes", [
    {"part_id": "other"}, {"view_id": "back"}, {"source_channel": "short"},
    {"source_kind": "unknown"},
    {"observation": {"raw_unmasked": False, "fixture_verified": True,
                     "observable": True, "evidence": "masked"}},
])
def test_input_contract_errors(sample, changes):
    assert run(sample, **changes).payload["status"] == "ERROR"


@pytest.mark.parametrize("kind", ["wrong_size", "float_image", "empty"])
def test_invalid_image_is_error(sample, kind):
    image = {"wrong_size": np.zeros((99, 160, 3), dtype=np.uint8),
             "float_image": sample[0].astype(np.float32),
             "empty": np.zeros((0, 0, 3), dtype=np.uint8)}[kind]
    assert run(sample, image).payload["status"] == "ERROR"


@pytest.mark.parametrize("case", ["missing_limits", "reversed_limits", "bad_roi", "missing_mask",
                                  "wrong_mask_size", "unvalidated_registration", "neighbor_ambiguous"])
def test_invalid_configuration_has_no_silent_defaults(sample, case):
    _, config, path = sample
    if case == "missing_limits":
        del config["limits"]
    elif case == "reversed_limits":
        config["limits"]["ng_max_open_fraction"] = 0.99
    elif case == "bad_roi":
        config["roi_xyxy"] = [-1, 20, 80, 80]
    elif case == "missing_mask":
        config["expected_mask"] = "absent.png"
    elif case == "wrong_mask_size":
        cv2.imwrite(str(path.parent / "expected.png"), np.full((2, 2), 255, dtype=np.uint8))
    elif case == "unvalidated_registration":
        config["registration"]["validated"] = False
    else:
        config["association"]["exclusive_search_confirmed"] = False
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises((ValueError, OSError)):
        load_config(path)


def test_evidence_preserves_existing_run(sample, tmp_path):
    result = run(sample)
    output = tmp_path / "evidence"
    result.save(output, source="synthetic.png")
    before = {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}
    payload = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert any(name.endswith(".png") for name in before)
    with pytest.raises(FileExistsError):
        result.save(output, source="must-not-overwrite.png")
    assert before == {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}


def test_boundary_band_is_review_not_forced_ng(sample):
    image = np.full_like(sample[0], 20)
    cv2.circle(image, (53, 50), 15, (230, 230, 230), -1)
    result = run(sample, image)
    assert result.payload["status"] == "REVIEW"
    assert result.payload["metrics"]["position_deviation_px"] == pytest.approx(3)


def test_explicit_red_channel_does_not_use_bgr_blue(sample):
    _, config, path = sample
    config["pixel_channel"] = "red"
    path.write_text(json.dumps(config), encoding="utf-8")
    image = np.full_like(sample[0], 20)
    image[:, :, 0] = sample[0][:, :, 0]
    assert run(sample, image).payload["status"] == "NG"
    image[:, :, 2] = sample[0][:, :, 2]
    assert run(sample, image).payload["status"] == "PASS"


def test_dark_polarity_can_measure_open_region(sample):
    _, config, path = sample
    config["segmentation"]["polarity"] = "dark"
    path.write_text(json.dumps(config), encoding="utf-8")
    image = 255 - sample[0]
    assert run(sample, image).payload["status"] == "PASS"


@pytest.mark.parametrize("case", ["missing_limits", "missing_image", "invalid_observation",
                                  "diagnostic_missing_threshold"])
def test_cli_input_errors_save_reviewable_error(sample, tmp_path, monkeypatch, case):
    from bmw_inspection.cli.check_hole import main

    image, config, path = sample
    image_path = tmp_path / "source.png"
    observation_path = tmp_path / "observation.json"
    output = tmp_path / "cli_output"
    if case != "missing_image":
        assert cv2.imwrite(str(image_path), image)
    if case == "missing_limits":
        del config["limits"]
        path.write_text(json.dumps(config), encoding="utf-8")
    elif case == "diagnostic_missing_threshold":
        make_diagnostic(sample)
        del config["segmentation"]["threshold"]
        path.write_text(json.dumps(config), encoding="utf-8")
    observation = [] if case == "invalid_observation" else {
        "raw_unmasked": True, "fixture_verified": True, "observable": True, "evidence": "synthetic only",
    }
    observation_path.write_text(json.dumps(observation), encoding="utf-8")
    monkeypatch.setattr("sys.argv", [
        "check_hole", "--config", str(path), "--image", str(image_path),
        "--part-id", "SYNTHETIC_ONLY", "--view-id", "front_left", "--source-channel", "fused",
        "--source-kind", "fused_only", "--observation", str(observation_path),
        "--output", str(output), "--synthesized-test-data",
    ])
    assert main() == 2
    saved = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert saved["status"] == "ERROR"
    assert saved["source"] == str(image_path)
    assert saved["synthesized_test_data"] is True


def test_threshold_sensitive_reflection_outside_expected_hole_requests_review(sample):
    image = sample[0].copy()
    image[22:30, 22:70] = 130
    result = run(sample, image=image)
    assert result.payload["metrics"]["open_fraction"] == 1
    assert result.payload["metrics"]["threshold_changed_fraction"] > 0.05
    assert result.payload["status"] == "REVIEW"


def make_diagnostic(sample):
    """Keep measurement settings explicit while declaring absent acceptance evidence."""
    _, config, path = sample
    config["status"] = "diagnostic"
    config["limits"] = None
    config["registration"]["validated"] = False
    config["registration"]["evidence"] = "Independent fixture positioning is not verified"
    config["segmentation"]["max_changed_fraction"] = None
    path.write_text(json.dumps(config), encoding="utf-8")


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("observation_verified", [False, True])
def test_diagnostic_never_passes_or_rejects_even_with_verified_observation(sample, tmp_path, empty,
                                                                         observation_verified):
    make_diagnostic(sample)
    observation = {"raw_unmasked": True, "fixture_verified": observation_verified,
                   "observable": observation_verified, "evidence": "Synthetic capture only"}
    result = run(sample, np.full_like(sample[0], 20) if empty else None, observation=observation)
    assert result.payload["status"] == "REVIEW"
    assert result.payload["diagnostic_only"] is True
    assert "fact" not in result.payload
    assert result.payload["metrics"]["open_fraction"] == (0 if empty else 1)
    assert result.payload["metrics"]["position_deviation_px"] == (None if empty else 0)
    assert "仅诊断测量" in result.payload["reason"]
    assert {"overlay", "full_overlay", "open_mask", "expected_mask"} <= result.images.keys()
    output = tmp_path / "diagnostic_evidence"
    result.save(output, source="synthetic.png")
    saved = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert saved["diagnostic_only"] is True
    assert saved["config"]["limits"] is None
    assert saved["evidence"]["full_overlay"] == "hole_1_full_overlay.png"


def test_diagnostic_retains_input_error_checks(sample):
    make_diagnostic(sample)
    assert run(sample, part_id="other").payload["status"] == "ERROR"
    assert run(sample, observation={"raw_unmasked": False}).payload["status"] == "ERROR"


@pytest.mark.parametrize("case", ["threshold", "stability_delta", "registration_evidence",
                                  "registration_mode", "association", "draft"])
def test_diagnostic_does_not_default_missing_measurement_contract(sample, case):
    make_diagnostic(sample)
    _, config, path = sample
    if case in ("threshold", "stability_delta"):
        del config["segmentation"][case]
    elif case == "registration_evidence":
        config["registration"]["evidence"] = " "
    elif case == "registration_mode":
        config["registration"]["mode"] = "target_hole"
    elif case == "association":
        config["association"]["exclusive_search_confirmed"] = False
    else:
        config["status"] = "draft"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize("field", ["limits", "max_changed_fraction"])
def test_ready_still_rejects_null_acceptance_fields(sample, field):
    _, config, path = sample
    if field == "limits":
        config[field] = None
    else:
        config["segmentation"][field] = None
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)
