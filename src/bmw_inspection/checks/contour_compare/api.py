"""In-memory front contour acquisition and geometry integration API.

This development API does not issue a production acceptance decision. It reuses
the same algorithms as the offline geometry report, without camera or file I/O.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .acquisition import acquire_current_outline
from .contracts import ContourInputError, check_image, json_ready, read_json, validate_config
from .extraction import coarse_segment
from .geometry_features import hole_line_distances, measure_line_feature
from .registration import estimate_rigid_pose


@dataclass
class FrontContourResult:
    """Per-call owned observations in input-image pixel coordinates.

    Attributes:
        record: Pose, features and diagnostic metadata; status remains REVIEW.
        current_mask: Segmentation candidate mask, or None when pose failed.
        contour: Candidate/observed arrays with validity and connectivity, or None.
    """

    record: dict
    current_mask: np.ndarray | None
    contour: dict | None

    def to_dict(self, *, include_arrays: bool = False) -> dict:
        """Return strict-JSON-safe data; include dense contour/mask only on request.

        Feature arrays in record are retained in both modes. Unknown coordinates
        become null; in-memory arrays retain NaN and their explicit validity flags.
        """
        result = dict(self.record)
        if include_arrays:
            result.update(current_mask=self.current_mask, contour=self.contour)
        return json_ready(result)


def _validate_geometry(config: dict) -> None:
    """Validate the geometry extension before reference preparation."""
    try:
        features = config['geometry_features']
        limits = config['geometry_validation']
        if not isinstance(features, list) or not features or not isinstance(limits, dict):
            raise ValueError('geometry features and validation are required')
        names = set()
        w, h = config['image']['width'], config['image']['height']
        for feature in features:
            name = feature['name']
            if not isinstance(name, str) or not name.strip() or name in names:
                raise ValueError('geometry feature names must be nonempty and unique')
            names.add(name)
            if feature.get('role') not in ('independent_check', 'diagnostic_check', 'measurement'):
                raise ValueError('invalid geometry feature role')
            points = np.asarray(feature['endpoints_xy'], dtype=float)
            if (points.shape != (2, 2) or not np.isfinite(points).all()
                    or np.any(points < 0) or np.any(points >= [w, h])
                    or np.linalg.norm(points[1] - points[0]) < 10):
                raise ValueError('feature endpoints must span >=10 pixels inside the image')
            for key, default in (('search_half_width_px', 40.), ('spacing_px', 2.), ('min_edge_amplitude', 8.)):
                value = feature.get(key, default)
                if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
                    raise ValueError(f'{key} must be finite positive')
            if not 2 <= feature.get('search_half_width_px', 40.) <= 200:
                raise ValueError('search_half_width_px must be in [2, 200]')
            for key, default in (('polarity', None), ('material_side', 1)):
                value = feature.get(key, default)
                if type(value) not in (int, float) or value not in (-1, 1):
                    raise ValueError(f'{key} must be -1 or +1')
            if any(type(i) is not int or i not in (0, 1) for i in feature.get('display_hole_indices', [])):
                raise ValueError('display_hole_indices must identify hole 0 or 1')
        for key in ('max_center_to_line_delta_px', 'max_line_direction_delta_deg'):
            value = limits[key]
            if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
                raise ValueError(f'{key} must be finite nonnegative')
        if type(limits['physical_datum_confirmed']) is not bool:
            raise ValueError('physical_datum_confirmed must be boolean')
    except (KeyError, TypeError, ValueError) as exc:
        raise ContourInputError(f'Invalid geometry configuration: {exc}') from exc


class FrontContourInspector:
    """Prepare reference once, then measure individual BGR/BGRA uint8 frames.

    Config and reference inputs are copied. Calls perform no resize, fusion,
    camera access or file writes. Use serial calls per instance; concurrent
    workers should use separate processes because GrabCut seeds OpenCV's RNG.
    """

    def __init__(self, config: dict, reference_image: np.ndarray, reference_mask: np.ndarray):
        try:
            self._config = validate_config(config, draft=True)
        except (KeyError, TypeError, ValueError) as exc:
            raise ContourInputError(f'Invalid contour configuration: {exc}') from exc
        _validate_geometry(self._config)
        check_image(reference_image, self._config)
        if (not isinstance(reference_mask, np.ndarray) or reference_mask.ndim != 2
                or reference_mask.shape != reference_image.shape[:2]
                or reference_mask.dtype.kind not in 'buif'
                or not np.isfinite(reference_mask).all()
                or not np.isin(reference_mask, [0, 1, 255]).all()
                or not np.any(reference_mask) or np.all(reference_mask)):
            raise ContourInputError('reference_mask must be binary, nonempty, with background, and match image size')
        self._reference = reference_image.copy()
        self._mask = (reference_mask != 0).astype(np.uint8)
        pose = estimate_rigid_pose(self._reference, self._reference, self._config)
        if not pose['valid']:
            raise ContourInputError(f'Reference hole localization failed: {pose}')
        coarse = coarse_segment(self._reference, self._mask, self._config,
                                {'valid': True, 'T_test_to_reference': np.eye(3)})
        holes = np.array([a['reference']['center_xy'] for a in pose['anchors']])
        self._reference_features = []
        for spec in self._config['geometry_features']:
            line = measure_line_feature(self._reference, spec, np.eye(3), coarse['mask'])
            line['hole_distances'] = hole_line_distances(holes, line)
            self._reference_features.append(line)

    @classmethod
    def from_files(cls, config_path: str | Path, reference_image_path: str | Path,
                   reference_mask_path: str | Path) -> FrontContourInspector:
        """Load explicitly selected assets once; paths resolve from caller cwd."""
        config = read_json(config_path)
        image = cv2.imread(str(reference_image_path), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(reference_mask_path), cv2.IMREAD_GRAYSCALE)
        return cls(config, image, mask)

    @property
    def reference_features(self) -> list[dict]:
        """Return an independent copy of the automatically measured baseline."""
        return deepcopy(self._reference_features)

    def process(self, image: np.ndarray, *, hand: str, view_id: str, channel: str,
                acquire_contour: bool = True) -> FrontContourResult:
        """Measure one frame; invalid inputs raise ContourInputError.

        Channel is caller-declared provenance (fused/short/long), not inferred
        from pixels. Missing test anchors return REVIEW with no mask or contour.
        """
        started = perf_counter()
        config = self._config
        check_image(image, config)
        if (hand, view_id, channel) != (config['hand'], config['view_id'], config['image']['channel']):
            raise ContourInputError('Image hand/view/channel mismatch')
        if type(acquire_contour) is not bool:
            raise ContourInputError('acquire_contour must be boolean')
        pose = estimate_rigid_pose(image, self._reference, config)
        record = {'schema_version': 1, 'status': 'REVIEW', 'diagnostic_only': True,
                  'whole_part_release': None, 'coordinate_system': 'input_pixel_xy',
                  'profile_id': config.get('profile_id'), 'hand': hand, 'view_id': view_id,
                  'channel': channel, 'pose': pose, 'features': [], 'hole_spacing_px': None,
                  'physical_datum_confirmed': config['geometry_validation']['physical_datum_confirmed'],
                  'independent_check_consistent': False, 'independent_check_used_for_fit': False,
                  'contour_acquisition': None}
        mask, contour = None, None
        if pose['valid']:
            coarse = coarse_segment(image, self._mask, config, pose)
            mask = coarse['mask']
            if acquire_contour:
                contour = acquire_current_outline(image, coarse,
                    min_edge_amplitude=config['extraction']['min_edge_amplitude'])
                record['contour_acquisition'] = {k: v for k, v in contour.items() if not isinstance(v, np.ndarray)}
            holes = np.array([a['test']['center_xy'] for a in pose['anchors']])
            record['hole_spacing_px'] = float(np.linalg.norm(holes[1] - holes[0]))
            checks = []
            for spec, baseline in zip(config['geometry_features'], self._reference_features):
                line = measure_line_feature(image, spec, pose['T_reference_to_test'], mask)
                line['hole_distances'] = hole_line_distances(holes, line)
                line.update(delta_center_to_line_px=None, direction_delta_deg=None)
                check = None
                if line['valid'] and baseline['valid']:
                    line['delta_center_to_line_px'] = [a['distance_px'] - b['distance_px']
                        for a, b in zip(line['hole_distances'], baseline['hole_distances'])]
                    direction = np.asarray(pose['T_test_to_reference'])[:2, :2] @ line['line_direction_xy']
                    base = baseline['line_direction_xy']
                    line['direction_delta_deg'] = float(np.degrees(np.arctan2(
                        np.linalg.det(np.stack([base, direction])), np.dot(base, direction))))
                    limits = config['geometry_validation']
                    check = (max(abs(v) for v in line['delta_center_to_line_px']) <= limits['max_center_to_line_delta_px']
                             and abs(line['direction_delta_deg']) <= limits['max_line_direction_delta_deg'])
                if spec['role'] == 'independent_check':
                    checks.append(check)
                line['development_consistency_check'] = check if spec['role'] == 'independent_check' else None
                record['features'].append(line)
            record['independent_check_consistent'] = bool(checks and all(v is True for v in checks))
        record['elapsed_ms'] = (perf_counter() - started) * 1000
        # Low-level results retain config references (ROI and parameter_snapshot).
        # Do not expose those references to callers of the cached inspector.
        return FrontContourResult(deepcopy(record), mask, contour)
