"""Conservative automatic hole localization beyond the taught anchor windows."""
from copy import deepcopy

import cv2
import numpy as np
import pytest

from bmw_inspection.checks.contour_compare.registration import estimate_rigid_pose


def fixture():
    image = np.full((420, 520), 220, np.uint8)
    cv2.circle(image, (150, 140), 17, 20, -1)
    cv2.circle(image, (340, 260), 22, 20, -1)
    config = {"anchor_rois_xyxy": [[120, 110, 180, 170], [307, 227, 373, 293]],
              "max_rotation_deg": 8, "max_anchor_motion_px": 100,
              "max_anchor_spacing_change_px": 1.5, "max_rms_residual_px": 1,
              "coarse_search": {"enabled": True, "search_margin_px": 110}}
    return image, config


@pytest.mark.parametrize("angle,offset", [(0, (65, -38)), (4, (55, -22)), (-4, (-50, 35))])
def test_automatic_pose_recovers_shift_and_rotation(angle, offset):
    reference, config = fixture()
    matrix = cv2.getRotationMatrix2D((260, 210), angle, 1)
    matrix[:, 2] += offset
    test = cv2.warpAffine(reference, matrix, (520, 420), borderValue=220)
    out = estimate_rigid_pose(test, reference, config)
    assert out["valid"], out
    expected = np.linalg.inv(np.vstack([matrix, [0, 0, 1]]))
    np.testing.assert_allclose(out["T_test_to_reference"], expected, atol=.2)
    assert out["coarse_search"]["feasible_assignment_count"] == 1
    assert np.linalg.det(out["T_test_to_reference"][:2, :2]) == pytest.approx(1.)


def test_disabled_search_preserves_fixed_roi_behavior():
    reference, config = fixture()
    matrix = np.float32([[1, 0, 65], [0, 1, -38]])
    test = cv2.warpAffine(reference, matrix, (520, 420), borderValue=220)
    config.pop("coarse_search")
    out = estimate_rigid_pose(test, reference, config)
    assert not out["valid"]
    assert out["reason_codes"] == ["anchor_extraction_failed"]
    assert "coarse_search" not in out
    config["coarse_search"] = {"enabled": False}
    disabled = estimate_rigid_pose(test, reference, config)
    assert disabled == out


def test_missing_hole_fails_closed():
    reference, config = fixture()
    test = reference.copy()
    cv2.circle(test, (340, 260), 24, 220, -1)
    out = estimate_rigid_pose(test, reference, config)
    assert not out["valid"]
    assert out["T_test_to_reference"] is None
    assert out["coarse_search"]["feasible_assignment_count"] == 0


def test_second_plausible_hole_pair_is_rejected_as_ambiguous():
    reference, config = fixture()
    test = reference.copy()
    cv2.circle(test, (150, 195), 17, 20, -1)
    cv2.circle(test, (340, 315), 22, 20, -1)
    out = estimate_rigid_pose(test, reference, config)
    assert not out["valid"]
    assert out["T_test_to_reference"] is None
    assert out["coarse_search"]["feasible_assignment_count"] == 2


def test_scaled_hole_constellation_is_rejected():
    reference, config = fixture()
    matrix = cv2.getRotationMatrix2D((260, 210), 0, 1.1)
    test = cv2.warpAffine(reference, matrix, (520, 420), borderValue=220)
    out = estimate_rigid_pose(test, reference, config)
    assert not out["valid"]
    assert out["T_test_to_reference"] is None
    assert out["coarse_search"]["rejected_assignments"]["spacing"] >= 1


@pytest.mark.parametrize("field,value", [("max_rotation_deg", 1), ("max_anchor_motion_px", 10)])
def test_search_never_overrides_motion_limits(field, value):
    reference, config = fixture()
    config[field] = value
    matrix = cv2.getRotationMatrix2D((260, 210), 4, 1)
    matrix[:, 2] += (55, -22)
    test = cv2.warpAffine(reference, matrix, (520, 420), borderValue=220)
    out = estimate_rigid_pose(test, reference, config)
    assert not out["valid"]
    assert out["T_test_to_reference"] is None


def test_disabled_search_still_accepts_legacy_small_shift():
    reference, config = fixture()
    config = deepcopy(config)
    config.pop("coarse_search")
    test = cv2.warpAffine(reference, np.float32([[1, 0, 3], [0, 1, -2]]), (520, 420), borderValue=220)
    out = estimate_rigid_pose(test, reference, config)
    assert out["valid"]
    np.testing.assert_allclose(out["T_test_to_reference"][:2, 2], [-3, 2], atol=.01)


def test_missing_hole_cannot_be_replaced_by_wrong_size_fixture_hole():
    reference, config = fixture()
    test = reference.copy()
    cv2.circle(test, (340, 260), 24, 220, -1)
    cv2.circle(test, (340, 260), 8, 20, -1)
    out = estimate_rigid_pose(test, reference, config)
    assert not out["valid"]
    assert out["coarse_search"]["candidate_counts"][1] >= 1
    assert out["coarse_search"]["shape_filtered_counts"][1] == 0


def test_missing_hole_cannot_be_replaced_by_wrong_spacing_fixture_hole():
    reference, config = fixture()
    test = reference.copy()
    cv2.circle(test, (340, 260), 24, 220, -1)
    cv2.circle(test, (360, 280), 22, 20, -1)
    out = estimate_rigid_pose(test, reference, config)
    assert not out["valid"]
    assert out["coarse_search"]["rejected_assignments"]["spacing"] == 1


@pytest.mark.parametrize("option,value", [("search_margin_px", -1), ("max_candidates_per_anchor", 0),
                                          ("max_axis_change_fraction", float("nan"))])
def test_invalid_search_options_fail_closed(option, value):
    reference, config = fixture()
    config["coarse_search"][option] = value
    out = estimate_rigid_pose(reference, reference, config)
    assert not out["valid"]
    assert out["reason_codes"] == ["coarse_search_invalid_settings"]
    assert out["T_test_to_reference"] is None


def test_candidate_limit_fails_closed():
    reference, config = fixture()
    config["coarse_search"]["max_candidates_per_anchor"] = 1
    test = reference.copy()
    cv2.circle(test, (150, 195), 17, 20, -1)
    out = estimate_rigid_pose(test, reference, config)
    assert not out["valid"]
    assert out["reason_codes"] == ["coarse_search_candidate_limit_exceeded"]


@pytest.mark.parametrize("options", [True, [], {"enabled": 1}, {"enabled": "true"},
                                     {"enabled": True, "search_margin_px": True},
                                     {"enabled": True, "max_candidates_per_anchor": True},
                                     {"enabled": True, "max_axis_change_fraction": True}])
def test_malformed_search_configuration_is_rejected(options):
    reference, config = fixture()
    config["coarse_search"] = options
    out = estimate_rigid_pose(reference, reference, config)
    assert not out["valid"]
    assert out["reason_codes"] == ["coarse_search_invalid_settings"]


def test_reference_self_registration_retains_original_refinement_windows():
    reference, config = fixture()
    out = estimate_rigid_pose(reference, reference, config)
    assert out["valid"]
    np.testing.assert_allclose(out["T_test_to_reference"], np.eye(3), atol=1e-10)
    assert [item["roi_xyxy"] for item in out["coarse_search"]["refinement"]] == config["anchor_rois_xyxy"]
    assert all(item["center_shift_from_coarse_px"] == 0 for item in out["coarse_search"]["refinement"])
