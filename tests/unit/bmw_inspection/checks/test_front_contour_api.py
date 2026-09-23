"""Integration contracts for the in-memory front contour diagnostic module."""
from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from bmw_inspection.checks.contour_compare import api
from bmw_inspection.checks.contour_compare.contracts import ContourInputError


@pytest.fixture
def inputs(monkeypatch):
    root = Path(__file__).resolve().parents[4]
    config = json.loads((root / 'configs/bmw/checks/contour/left_front_4024_automatic_geometry_v1.json').read_text())
    config['image'].update(width=240, height=240)
    config['part_roi_xyxy'] = [10, 10, 230, 230]
    config['registration']['anchor_rois_xyxy'] = [[45, 55, 75, 85], [75, 145, 105, 175]]
    config['registration']['coarse_search'] = {'enabled': False}
    config['geometry_features'] = [{
        'name': 'right_edge', 'role': 'independent_check',
        'endpoints_xy': [[130, 50], [130, 190]], 'polarity': 1,
        'material_side': 1, 'search_half_width_px': 15,
        'min_edge_amplitude': 8, 'spacing_px': 2,
    }]
    mask = np.zeros((240, 240), np.uint8)
    mask[20:220, 30:130] = 1
    image = np.repeat((mask * 180)[..., None], 3, axis=2)
    cv2.circle(image, (60, 70), 7, (0, 0, 0), -1)
    cv2.circle(image, (90, 160), 9, (0, 0, 0), -1)

    def coarse(image, reference_mask, config, registration):
        contours, _ = cv2.findContours(reference_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return {'mask': (reference_mask != 0).astype(np.uint8), 'dense_xy': contours[0][:, 0].astype(float)}

    monkeypatch.setattr(api, 'coarse_segment', coarse)
    return config, image, mask


def process(inspector, image, **overrides):
    metadata = {'hand': 'left', 'view_id': 'front', 'channel': 'fused'}
    metadata.update(overrides)
    return inspector.process(image, **metadata)


def test_reference_self_measurement_and_serialization(inputs):
    config, image, mask = inputs
    inspector = api.FrontContourInspector(config, image, mask)
    result = process(inspector, image)
    assert result.record['status'] == 'REVIEW'
    assert result.record['diagnostic_only'] is True
    assert result.record['pose']['valid']
    assert result.current_mask.shape == mask.shape
    assert result.contour is not None
    feature = result.record['features'][0]
    assert feature['valid']
    np.testing.assert_allclose(feature['delta_center_to_line_px'], [0, 0], atol=1e-8)
    assert not result.contour['semantic_boundary_confirmed']
    assert np.isnan(result.contour['observed_xy']).any()
    summary = result.to_dict()
    encoded = json.dumps(summary, allow_nan=False)
    assert 'observed_xy' not in encoded
    assert 'current_mask' not in summary
    full = result.to_dict(include_arrays=True)
    json.dumps(full, allow_nan=False)
    assert full['current_mask'] == result.current_mask.tolist()
    assert any(None in point for point in full['contour']['observed_xy'])


def test_pose_failure_does_not_segment_or_fabricate_outline(inputs, monkeypatch):
    config, image, mask = inputs
    inspector = api.FrontContourInspector(config, image, mask)

    def forbidden(*args, **kwargs):
        pytest.fail('Segmentation must not run after unsuccessful registration')

    monkeypatch.setattr(api, 'coarse_segment', forbidden)
    result = process(inspector, np.zeros_like(image))
    assert not result.record['pose']['valid']
    assert result.record['status'] == 'REVIEW'
    assert result.record['features'] == []
    assert result.current_mask is None
    assert result.contour is None
    json.dumps(result.to_dict(include_arrays=True), allow_nan=False)


@pytest.mark.parametrize('metadata', [
    {'hand': 'right'}, {'view_id': 'back'}, {'channel': 'short'}, {'acquire_contour': 1},
])
def test_incompatible_metadata_rejected(inputs, metadata):
    config, image, mask = inputs
    inspector = api.FrontContourInspector(config, image, mask)
    with pytest.raises(ContourInputError):
        process(inspector, image, **metadata)


@pytest.mark.parametrize('transform', [lambda x: x[:-1], lambda x: x.astype(float), lambda x: x[..., 0]])
def test_invalid_image_not_implicitly_resized_or_converted(inputs, transform):
    config, image, mask = inputs
    inspector = api.FrontContourInspector(config, image, mask)
    with pytest.raises(ContourInputError):
        process(inspector, transform(image))


@pytest.mark.parametrize('transform', [lambda x: x[:-1], lambda x: np.zeros_like(x), lambda x: np.ones_like(x)])
def test_invalid_reference_mask_rejected(inputs, transform):
    config, image, mask = inputs
    with pytest.raises(ContourInputError):
        api.FrontContourInspector(config, image, transform(mask))


@pytest.mark.parametrize('section', ['geometry_features', 'geometry_validation'])
def test_missing_geometry_contract_rejected(inputs, section):
    config, image, mask = inputs
    config.pop(section)
    with pytest.raises(ContourInputError):
        api.FrontContourInspector(config, image, mask)


def test_reference_hole_localization_failure_rejected(inputs):
    config, image, mask = inputs
    with pytest.raises(ContourInputError):
        api.FrontContourInspector(config, np.zeros_like(image), mask)


def test_reference_cache_isolated_from_caller_mutation(inputs):
    config, image, mask = inputs
    original = image.copy()
    inspector = api.FrontContourInspector(config, image, mask)
    expected = deepcopy(inspector.reference_features)
    config['geometry_features'][0]['endpoints_xy'][0][0] = 0
    image[:] = 0
    mask[:] = 0
    exposed = inspector.reference_features
    exposed[0]['name'] = 'changed'
    exposed[0]['points_xy'][:] = 0
    result = process(inspector, original, acquire_contour=False)
    assert result.record['pose']['valid']
    assert result.record['features'][0]['name'] == 'right_edge'
    np.testing.assert_allclose(result.record['features'][0]['delta_center_to_line_px'], [0, 0], atol=1e-8)
    assert inspector.reference_features[0]['name'] == expected[0]['name']
    np.testing.assert_allclose(inspector.reference_features[0]['points_xy'], expected[0]['points_xy'], equal_nan=True)
    assert result.current_mask.any()
    assert result.contour is None


def test_returned_observations_cannot_mutate_future_recipe_or_baseline(inputs):
    config, image, mask = inputs
    inspector = api.FrontContourInspector(config, image, mask)
    first = process(inspector, image, acquire_contour=False)
    first.record['features'][0]['parameter_snapshot']['name'] = 'mutated'
    first.record['features'][0]['parameter_snapshot']['endpoints_xy'][0] = [0, 0]
    first.record['pose']['anchors'][0]['roi_xyxy'][:] = [0, 0, 1, 1]
    first.current_mask[:] = 0
    next_result = process(inspector, image, acquire_contour=False)
    assert next_result.record['pose']['valid']
    assert next_result.record['features'][0]['name'] == 'right_edge'
    assert inspector.reference_features[0]['parameter_snapshot']['name'] == 'right_edge'
    np.testing.assert_allclose(next_result.record['features'][0]['delta_center_to_line_px'], [0, 0], atol=1e-8)
    assert next_result.current_mask.any()
