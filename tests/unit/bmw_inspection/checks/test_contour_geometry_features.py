"""Synthetic independent line-caliper checks; no industrial accuracy claim."""
import cv2
import numpy as np

from bmw_inspection.checks.contour_compare.geometry_features import measure_line_feature, hole_line_distances


def scene():
    image = np.zeros((240, 240, 3), np.uint8)
    image[20:220, 30:130] = 180
    feature = {'name': 'right_edge', 'endpoints_xy': [[130, 50], [130, 190]], 'polarity': 1,
               'search_half_width_px': 20}
    return image, feature


def test_pose_moves_calipers_without_moving_measured_geometry():
    image, feature = scene()
    original = image.copy()
    base = measure_line_feature(image, feature, np.eye(3))
    assert base['valid'] and base['support_fraction'] == 1
    h = np.array([[80., 110.]])
    expected = hole_line_distances(h, base)[0]['distance_px']
    np.testing.assert_allclose(expected, 49.5, atol=.1)
    matrix = np.eye(3); matrix[:2] = cv2.getRotationMatrix2D((120,120), 9, 1)
    matrix[:2,2] += [14,-5]
    moved = cv2.warpAffine(image, matrix[:2], (240,240))
    result = measure_line_feature(moved, feature, matrix)
    hole = h @ matrix[:2,:2].T + matrix[:2,2]
    assert result['valid']
    np.testing.assert_allclose(hole_line_distances(hole, result)[0]['distance_px'], expected, atol=.25)
    np.testing.assert_array_equal(image, original)


def test_actual_edge_displacement_is_not_snapped_to_standard():
    image, feature = scene()
    base = measure_line_feature(image, feature, np.eye(3))
    image[20:220,130:139] = 180
    changed = measure_line_feature(image, feature, np.eye(3))
    h=np.array([[80.,110.]])
    difference=hole_line_distances(h,changed)[0]['distance_px']-hole_line_distances(h,base)[0]['distance_px']
    np.testing.assert_allclose(difference,9,atol=.1)


def test_missing_and_ambiguous_edges_remain_unknown():
    image, feature = scene()
    blank = measure_line_feature(np.zeros_like(image), feature, np.eye(3))
    assert not blank['valid'] and not blank['point_valid'].any()
    assert hole_line_distances(np.array([[80,110]]),blank)[0]['distance_px'] is None
    image[:]=0
    image[:,100:120]=180;image[:,126:136]=180
    ambiguous = measure_line_feature(image,feature,np.eye(3))
    assert not ambiguous['valid']
    assert 'ambiguous_edges' in ambiguous['invalid_reason']


def test_extrapolated_foot_is_explicit():
    image,feature=scene()
    line=measure_line_feature(image,feature,np.eye(3))
    value=hole_line_distances(np.array([[80,5]]),line)[0]
    assert value['valid'] and not value['foot_within_observed_span']


def test_coarse_config_rejects_silent_invalid_options():
    import json
    from pathlib import Path
    import pytest
    from bmw_inspection.checks.contour_compare.contracts import validate_config, ContourInputError
    root=Path(__file__).resolve().parents[4]
    config=json.loads((root/'configs/bmw/checks/contour/left_front_4024_automatic_geometry_v1.json').read_text())
    assert validate_config(config,draft=True)['registration']['coarse_search']['enabled']
    for bad in ({'enabled':'yes'}, {'enabled':True,'search_margin_px':True}, {'enabled':False,'typo':4}):
        config['registration']['coarse_search']=bad
        with pytest.raises(ContourInputError):validate_config(config,draft=True)


def test_stronger_neighbouring_fixture_edge_cannot_replace_material_edge():
    image,feature=scene()
    image[:]=0
    image[:,30:130]=60
    image[:,138:145]=180
    line=measure_line_feature(image,feature,np.eye(3))
    assert not line['valid']
    assert not line['point_valid'].any()
    assert set(line['invalid_reason'])=={'ambiguous_edges'}


def test_current_material_mask_disambiguates_stronger_fixture_without_snapping():
    image,feature=scene()
    image[:]=0;image[:,30:130]=60;image[:,138:145]=180
    mask=np.zeros(image.shape[:2],np.uint8);mask[:,30:130]=1
    line=measure_line_feature(image,feature,np.eye(3),mask)
    assert line['valid']
    np.testing.assert_allclose(line['points_xy'][:,0],129.5,atol=.1)
    assert not line['semantic_boundary_confirmed']
    image[:,130:135]=60;mask[:,130:135]=1
    line=measure_line_feature(image,feature,np.eye(3),mask)
    assert line['valid']
    np.testing.assert_allclose(line['points_xy'][:,0],134.5,atol=.1)
