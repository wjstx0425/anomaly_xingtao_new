"""R02a software evidence contracts, not industrial line-defect acceptance."""

import copy

import numpy as np
import pytest

from bmw_inspection.benchmark.line_rule import predict_lines


def scene():
    image = np.full((100, 140, 3), 120, np.uint8)
    config = {"image_shape_hw": [100, 140], "roi_xyxy": [10, 10, 130, 90],
              "response_threshold": 25.0, "min_area_px": 4,
              "widths_px": [3, 7, 15], "background_sigma_px": 15.0}
    return image, config


def test_dark_and_bright_lines_keep_independent_native_maps_and_original_boxes():
    image, config = scene()
    image[25:28, 25:65] = 20
    image[55:58, 75:115] = 225
    result = predict_lines(image, config)
    assert result["response_dark"][26, 40] >= 25
    assert result["response_bright"][56, 90] >= 25
    assert result["candidate_mask"][26, 40] == 255
    assert result["candidate_mask"][56, 90] == 255
    for x, y in [(40, 26), (90, 56)]:
        assert any(box["xyxy"][0] <= x < box["xyxy"][2] and box["xyxy"][1] <= y < box["xyxy"][3] for box in result["boxes"])
    assert result["response_dark"].shape == image.shape[:2]
    assert result["response_dark"].dtype == np.float32
    assert result["decision"] == "REVIEW"
    assert result["calibration_status"] == "development_uncalibrated"


def test_short_wide_candidate_is_not_discarded_by_aspect_ratio():
    image, config = scene()
    image[40:48, 50:59] = 10
    result = predict_lines(image, config)
    assert result["candidate_mask"][44, 54] == 255
    containing = [b for b in result["boxes"] if b["xyxy"][0] <= 54 < b["xyxy"][2] and b["xyxy"][1] <= 44 < b["xyxy"][3]]
    assert containing and containing[0]["area_px"] >= 60
    assert containing[0]["width_px"] > 1


def test_uniform_image_has_no_candidates_but_is_not_claimed_pass():
    image, config = scene()
    result = predict_lines(image, config)
    assert not result["candidate_mask"].any()
    assert not result["boxes"]
    assert result["raw_score"] == 0
    assert result["decision"] == "REVIEW"


def test_roi_limits_observation_without_changing_input_or_configuration():
    image, config = scene()
    image[40:43, :140] = 20
    original, original_config = image.copy(), copy.deepcopy(config)
    image.flags.writeable = False
    result = predict_lines(image, config)
    np.testing.assert_array_equal(image, original)
    assert config == original_config
    assert not result["candidate_mask"][:, :10].any()
    assert not result["response_dark"][:, 130:].any()
    assert result["observed_regions"] == [[10, 10, 130, 90]]
    observed_area = 120 * 80
    assert observed_area + sum((r[2] - r[0]) * (r[3] - r[1]) for r in result["missing_regions"]) == 100 * 140
    assert any(box["touches_observed_roi_boundary"] for box in result["boxes"])


def test_line_descriptors_use_pixel_units_and_finite_timings():
    image, config = scene()
    image[40:43, 40:90] = 10
    result = predict_lines(image, config)
    box = max(result["boxes"], key=lambda value: value["area_px"])
    assert box["skeleton_length_px"] > 40
    assert 1 <= box["width_px"] <= 5
    assert box["orientation_deg"] == pytest.approx(0)
    for key in ("preprocess_ms", "inference_ms", "postprocess_ms"):
        assert np.isfinite(result[key]) and result[key] >= 0


@pytest.mark.parametrize("key,value", [
    ("image_shape_hw", [90, 140]), ("roi_xyxy", [-1, 10, 30, 40]),
    ("roi_xyxy", [10, 10, 150, 90]), ("roi_xyxy", [10, 10, 10, 90]),
    ("response_threshold", None), ("response_threshold", float("nan")),
    ("response_threshold", 0), ("min_area_px", 0), ("min_area_px", True),
    ("widths_px", [2, 7]), ("widths_px", []), ("orientations_deg", [180]),
    ("background_sigma_px", float("inf")),
])
def test_invalid_geometry_and_development_parameters_raise(key, value):
    image, config = scene()
    config[key] = value
    with pytest.raises(ValueError):
        predict_lines(image, config)


def test_wrong_image_dtype_or_channel_layout_is_rejected():
    image, config = scene()
    for invalid in (image.astype(float), image[:, :, 0], np.zeros((100, 140, 4), np.uint8)):
        with pytest.raises(ValueError, match="uint8 BGR"):
            predict_lines(invalid, config)
