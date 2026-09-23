"""Current-image acquisition preserves geometry and unknown evidence."""
import cv2
import numpy as np

from bmw_inspection.checks.contour_compare.acquisition import acquire_current_outline


def scene(shift=(0, 0)):
    mask = np.zeros((100, 120), np.uint8)
    polygon = np.array([[20, 20], [80, 20], [80, 40], [88, 40], [88, 47],
                        [80, 47], [80, 80], [50, 80], [50, 72], [44, 72],
                        [44, 80], [20, 80]]) + shift
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
    dense = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0][:, 0].astype(float)
    return (mask * 180 + 30).astype(np.uint8), {'mask': mask, 'dense_xy': dense, 'diagnostics': {}}


def test_translation_and_narrow_features_preserved():
    image, coarse = scene()
    result = acquire_current_outline(image, coarse, min_edge_amplitude=10.)
    shifted_image, shifted_coarse = scene((7, 4))
    shifted = acquire_current_outline(shifted_image, shifted_coarse, min_edge_amplitude=10.)
    np.testing.assert_array_equal(result['candidate_xy'], coarse['dense_xy'])
    np.testing.assert_allclose(shifted['candidate_xy'], result['candidate_xy'] + [7, 4])
    np.testing.assert_array_equal(shifted['valid'], result['valid'])
    np.testing.assert_allclose(shifted['observed_xy'][result['valid']], result['observed_xy'][result['valid']] + [7, 4])
    assert result['valid'].mean() > .5
    assert np.any(np.all(result['candidate_xy'] == [88, 40], axis=1))
    assert np.any(np.all(result['candidate_xy'] == [47, 72], axis=1))
    assert not result['semantic_boundary_confirmed']


def test_no_contrast_has_no_observations():
    image, coarse = scene()
    result = acquire_current_outline(np.full_like(image, 50), coarse, min_edge_amplitude=10.)
    assert not result['valid'].any()
    assert np.isnan(result['observed_xy']).all()
    assert result['image_supported_arc_length_px'] == 0
    assert result['candidate_arc_length_px'] > 0


def test_nan_separator_and_missing_image_evidence_cannot_be_bridged():
    image, coarse = scene()
    coarse['dense_xy'] = np.vstack([coarse['dense_xy'][:30], [np.nan, np.nan], coarse['dense_xy'][30:]])
    image[35:45, 16:25] = 50
    result = acquire_current_outline(image, coarse, min_edge_amplitude=10.)
    assert not result['valid'][30]
    assert not result['segment_valid'][29:31].any()
    assert np.isnan(result['observed_xy'][~result['valid']]).all()
    for i in np.flatnonzero(result['segment_valid']):
        j = result['segment_end_index'][i]
        assert result['valid'][i] and result['valid'][j]


def test_corners_are_explicit_pixel_candidates():
    image, coarse = scene()
    result = acquire_current_outline(image, coarse, min_edge_amplitude=10.)
    index = np.flatnonzero(np.all(coarse['dense_xy'] == [88, 40], axis=1))[0]
    assert not result['valid'][index]
    assert result['invalid_reason'][index] == 'pixel_candidate_corner'


def test_current_protrusion_deformation_is_measured():
    image, coarse = scene()
    original = acquire_current_outline(image, coarse, min_edge_amplitude=10.)
    mask = coarse['mask'].copy()
    mask[40:48, 80:95] = 1
    contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0][:, 0].astype(float)
    deformed = acquire_current_outline((mask * 180 + 30).astype(np.uint8),
                                       {'mask': mask, 'dense_xy': contour}, min_edge_amplitude=10.)
    assert np.nanmax(deformed['observed_xy'][:, 0]) > np.nanmax(original['observed_xy'][:, 0]) + 5


def test_transparent_image_domain_never_supports_edges():
    image, coarse = scene()
    bgra = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    bgra[:, :, 3] = 0
    result = acquire_current_outline(bgra, coarse, min_edge_amplitude=10.)
    assert not result['valid'].any()
