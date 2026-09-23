"""Synthetic image evidence tests, not BMW production acceptance."""

import cv2
import numpy as np
import pytest

from bmw_inspection.checks.contour_compare.extraction import coarse_segment, extract_full_outline


def scene():
    mask = np.zeros((150, 180), np.uint8)
    mask[30:120, 45:135] = 1
    image = np.repeat((20 + mask * 180)[..., None], 3, axis=2)
    # Sample the four sides away from corners to isolate image edge measurements.
    y = np.arange(42, 108, dtype=float)
    x = np.arange(58, 122, dtype=float)
    points = np.concatenate([np.column_stack([y * 0 + 45, y]), np.column_stack([x, x * 0 + 119]), np.column_stack([y * 0 + 134, y[::-1]]), np.column_stack([x[::-1], x * 0 + 30])])
    normals = np.concatenate([np.tile(n, (count, 1)) for n, count in [([1, 0], len(y)), ([0, -1], len(x)), ([-1, 0], len(y)), ([0, 1], len(x))]])
    contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    reference = {"mask": mask, "image": image.copy(), "sample_xy": points, "dense_xy": contour[0][:, 0].astype(float), "inward_normal_xy": normals}
    config = {"extraction": {"search_inward_px": 20, "search_outward_px": 20, "profile_step_px": 0.5, "min_edge_amplitude": 10, "min_candidate_margin": 0.2}}
    return image, reference, config


def extract(image, reference, config, transform=None):
    return extract_full_outline(image, reference, {"T_test_to_reference": np.eye(3) if transform is None else transform}, config)


def test_normal_four_sides_and_read_only_input():
    image, reference, config = scene()
    original = image.copy()
    result = extract(image, reference, config)
    assert result["valid"].all()
    assert np.max(abs(result["normal_offset_u_px"])) < 1e-10
    np.testing.assert_array_equal(image, original)


def test_notch_is_not_filled_by_reference_prior():
    image, reference, config = scene()
    image[65:88, 45:58] = 20
    result = extract(image, reference, config)
    selected = (reference["sample_xy"][:, 0] == 45) & (reference["sample_xy"][:, 1] >= 68) & (reference["sample_xy"][:, 1] <= 84)
    assert result["valid"][selected].all()
    np.testing.assert_allclose(result["normal_offset_u_px"][selected], 13, atol=1e-10)
    assert result["mask"][75, 50] == 0


def test_protrusion_outside_profile_is_retained_globally():
    image, reference, config = scene()
    image[65:88, 10:50] = 200
    result = extract(image, reference, config)
    assert result["mask"][75, 12] == 1
    assert result["diagnostics"]["global_shape_change_candidate"]
    selected = (reference["sample_xy"][:, 0] == 45) & (reference["sample_xy"][:, 1] == 75)
    assert not result["valid"][selected].any()
    assert np.isnan(result["sample_xy"][selected]).all()


def test_transparency_produces_missing_observations():
    image, reference, config = scene()
    image = np.dstack([image, np.full(image.shape[:2], 255, np.uint8)])
    image[65:88, 30:70, 3] = 0
    result = extract(image, reference, config)
    selected = (reference["sample_xy"][:, 0] == 45) & (reference["sample_xy"][:, 1] == 75)
    assert not result["valid"][selected].any()
    assert np.isnan(result["sample_xy"][selected]).all()


def test_reference_to_current_prior_uses_inverse_transform():
    image, reference, config = scene()
    current = cv2.warpAffine(image, np.array([[1, 0, 7], [0, 1, -4]], dtype=float), (180, 150), borderValue=(20, 20, 20))
    transform = np.array([[1, 0, -7], [0, 1, 4], [0, 0, 1]], dtype=float)
    result = extract(current, reference, config, transform)
    assert result["valid"].all()
    assert np.max(abs(result["normal_offset_u_px"])) <= 1


def test_components_are_never_silently_dropped():
    image, reference, config = scene()
    image[65:72, 45:135] = 20
    result = coarse_segment(image, reference["mask"], config)
    assert result["diagnostics"]["component_count"] == 2
    assert np.isnan(result["dense_xy"]).any()


def test_displaced_corner_uses_unknown_not_nominal_fill():
    image, reference, config = scene()
    reference["corner_mask"] = np.ones(len(reference["sample_xy"]), bool)
    image[65:88, 45:58] = 20
    result = extract(image, reference, config)
    selected = (reference["sample_xy"][:, 0] == 45) & (reference["sample_xy"][:, 1] == 75)
    assert not result["valid"][selected].any()
    assert np.isnan(result["sample_xy"][selected]).all()


def test_missing_calibration_and_nonrigid_transform_rejected():
    image, reference, config = scene()
    config["extraction"]["min_edge_amplitude"] = None
    with pytest.raises(ValueError, match="explicit"):
        extract(image, reference, config)
    config["extraction"]["min_edge_amplitude"] = 10
    with pytest.raises(ValueError, match="rigid"):
        extract(image, reference, config, np.diag([1.1, 1.1, 1]))


def test_teach_requires_roi_and_rejects_non_uint8():
    image, _, config = scene()
    with pytest.raises(ValueError, match="part_roi"):
        coarse_segment(image, None, config)
    with pytest.raises(ValueError, match="uint8"):
        coarse_segment(image.astype(float), None, config)
    config["part_roi_xyxy"] = [35, 20, 145, 130]
    result = coarse_segment(image, None, config)
    assert result["mask"][75, 75] == 1
    assert result["mask"][15, 15] == 0


def test_low_contrast_is_unknown_even_when_segmentation_is_closed():
    image, reference, config = scene()
    image[reference["mask"] != 0] = 24
    result = extract(image, reference, config)
    assert not result["valid"].any()
    assert np.isnan(result["sample_xy"]).all()
    assert not result["dense_valid"].any()


def test_known_rotation_preserves_reference_pixel_coordinates():
    image, reference, config = scene()
    forward = np.eye(3)
    forward[:2] = cv2.getRotationMatrix2D((90, 75), 3, 1)
    current = cv2.warpAffine(image, forward[:2], (180, 150), borderValue=(20, 20, 20))
    result = extract(current, reference, config, np.linalg.inv(forward))
    assert result["valid"].all()
    assert np.max(abs(result["normal_offset_u_px"])) <= 1


def test_missing_or_failed_registration_cannot_fall_back_to_identity():
    image, reference, config = scene()
    with pytest.raises(ValueError, match="registration"):
        extract_full_outline(image, reference, {}, config)
    with pytest.raises(ValueError, match="registration"):
        extract_full_outline(image, reference, {"valid": False, "T_test_to_reference": np.eye(3)}, config)


def test_quarter_pixel_motion_is_preserved_without_nominal_snapping():
    image, reference, config = scene()
    current = cv2.warpAffine(image, np.array([[1, 0, .25], [0, 1, 0]], dtype=float), (180, 150), borderValue=(20, 20, 20))
    result = extract(current, reference, config)
    left = reference["sample_xy"][:, 0] == 45
    assert result["valid"][left].all()
    np.testing.assert_allclose(result["normal_offset_u_px"][left], .25, atol=1e-8)
    assert all(c[0]["reference_pixel_boundary_correction_px"] == .5 for c in np.asarray(result["candidates"], dtype=object)[left])


def test_low_contrast_reference_cannot_silently_define_edge_calibration():
    image, reference, config = scene()
    reference["image"][:] = 20
    result = extract(image, reference, config)
    assert not result["valid"].any()
    assert np.isnan(result["sample_xy"]).all()
    assert np.all(result["invalid_reason"] == "reference_image_edge_calibration_unconfirmed")


@pytest.mark.parametrize('key,value', [('min_profile_similarity', float('nan')), ('min_profile_similarity', -2), ('min_profile_similarity', 2), ('min_profile_similarity_margin', float('nan')), ('coarse_reference_agreement_px', float('inf'))])
def test_invalid_new_profile_parameters_rejected_by_pure_api(key, value):
    image, reference, config = scene()
    config['extraction'][key] = value
    with pytest.raises(ValueError, match=key):
        extract(image, reference, config)


def test_stronger_inner_texture_does_not_change_same_image_material_edge():
    image, reference, config = scene()
    image[30:120, 63:66] = 255
    image[30:120, 67:70] = 10
    reference['image'] = image.copy()
    result = extract(image, reference, config)
    assert np.max(np.abs(result['normal_offset_u_px'][result['valid']])) < 1e-10
