"""Image-based regressions for whole-part component identity."""

import cv2
import numpy as np
import pytest

from bmw_inspection.checks.contour_compare.segmentation import segment_part


def scene():
    mask = np.zeros((200, 240), np.uint8)
    mask[50:150, 90:160] = 1
    image = np.repeat((20 + 180 * mask)[..., None], 3, axis=2)
    return image, mask, {"extraction": {"component_unknown_distance_px": 10}}


def test_background_bigger_than_part_is_excluded_by_identity():
    image, mask, config = scene()
    image[10:190, 10:65] = 200
    result = segment_part(image, mask, config)
    assert result["mask"][100, 110] == 1
    assert not result["mask"][100, 40]
    assert result["diagnostics"]["raw_component_count"] == 2
    assert result["diagnostics"]["excluded_component_count"] == 1
    assert result["diagnostics"]["component_count"] == 1
    assert np.min(result["dense_xy"][:, 0]) >= 90


def test_connected_large_excess_is_retained_beyond_search_band():
    image, mask, config = scene()
    image[90:115, 155:225] = 200
    result = segment_part(image, mask, config)
    assert result["mask"][100, 220] == 1
    assert np.max(result["dense_xy"][:, 0]) == 224


def test_broken_part_retains_both_observed_components_for_review():
    image, mask, config = scene()
    image[95:105, 90:160] = 20
    result = segment_part(image, mask, config)
    assert result["diagnostics"]["multiple_components"]
    assert result["diagnostics"]["component_count"] == 2
    assert np.isnan(result["dense_xy"]).any()
    assert not result["mask"][100, 120]


def test_detached_fragment_is_unknown_and_not_measured_excess():
    image, mask, config = scene()
    image[70:85, 165:175] = 200
    result = segment_part(image, mask, config)
    assert result["diagnostics"]["component_assignment_uncertain"]
    assert result["diagnostics"]["unresolved_component_count"] == 1
    assert not result["mask"][75, 170]
    assert np.max(result["dense_xy"][:, 0]) == 159


def test_dark_current_region_is_never_filled_with_nominal_foreground():
    image, mask, config = scene()
    image[120:150, 90:160] = 20
    result = segment_part(image, mask, config)
    assert not result["mask"][140, 110]
    assert result["mask"][110, 110]


def test_teach_roi_and_registered_component_identity():
    image, mask, config = scene()
    config["part_roi_xyxy"] = [80, 40, 170, 160]
    taught = segment_part(image, None, config)
    np.testing.assert_array_equal(taught["mask"], mask)
    forward = np.array([[1., 0, 10], [0, 1, -5], [0, 0, 1]])
    current = cv2.warpAffine(image, forward[:2], (240, 200), borderValue=(20, 20, 20))
    result = segment_part(current, mask, config, {"T_test_to_reference": np.linalg.inv(forward)})
    expected = cv2.warpAffine(mask, forward[:2], (240, 200))
    np.testing.assert_array_equal(result["mask"], expected)


def test_absent_core_reports_missing_and_settings_are_validated():
    image, mask, config = scene()
    config["extraction"]["component_core_inset_px"] = 1000
    result = segment_part(image, mask, config)
    assert result["diagnostics"]["part_component_missing"]
    assert not result["mask"].any()
    config["extraction"]["component_core_inset_px"] = -1
    with pytest.raises(ValueError, match="component assignment"):
        segment_part(image, mask, config)


def test_internal_hole_island_is_not_a_second_outer_component():
    image, mask, config = scene()
    image[75:125, 105:145] = 20
    image[90:110, 115:135] = 200
    result = segment_part(image, mask, config)
    assert result["diagnostics"]["core_overlapping_connected_component_count"] == 2
    assert result["diagnostics"]["component_count"] == 1
    assert not result["diagnostics"]["multiple_components"]
    assert np.all((result["dense_xy"][:, 0] == 90) | (result["dense_xy"][:, 0] == 159) | (result["dense_xy"][:, 1] == 50) | (result["dense_xy"][:, 1] == 149))
