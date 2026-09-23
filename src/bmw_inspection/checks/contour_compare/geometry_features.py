"""Independent image calipers for hole-to-edge geometry in input pixels.

These measurements never alter the hole-derived pose or fill an outer contour.
Line endpoints define a taught sampling span, not certified material endpoints.
"""
from __future__ import annotations

import cv2
import numpy as np

from .extraction import _candidate_transitions, _channels, _sample
from .geometry import transform_points


def measure_line_feature(image: np.ndarray, feature: dict, reference_to_test: np.ndarray,
                         current_mask: np.ndarray | None = None) -> dict:
    """Measure a taught straight span using only current-image edge responses.

    Return every caliper's validity and raw points, robust line fit, residuals,
    and observed support span. Missing or ambiguous responses remain unknown.
    The robust line is a measurement summary, never replacement edge pixels.
    """
    _, gray, domain = _channels(image)
    endpoints = np.asarray(feature['endpoints_xy'], float)
    matrix = np.asarray(reference_to_test, float)
    search = float(feature.get('search_half_width_px', 40.))
    amplitude = float(feature.get('min_edge_amplitude', 8.))
    spacing = float(feature.get('spacing_px', 2.))
    polarity = feature.get('polarity')
    if (endpoints.shape != (2, 2) or not np.isfinite(endpoints).all()
            or matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[2], [0, 0, 1])
            or not np.allclose(matrix[:2, :2].T @ matrix[:2, :2], np.eye(2), atol=1e-6)
            or not np.isclose(np.linalg.det(matrix[:2, :2]), 1.)
            or not np.isfinite([search, amplitude, spacing]).all()
            or not 2 <= search <= 200 or amplitude <= 0 or spacing <= 0
            or polarity not in (-1, 1)):
        raise ValueError('Invalid line feature or non-rigid pose')
    delta = endpoints[1] - endpoints[0]
    length = float(np.linalg.norm(delta))
    if length < 10:
        raise ValueError('Line feature must span at least 10 pixels')
    tangent = delta / length
    normal = np.array([-tangent[1], tangent[0]])
    n = min(1500, max(6, int(length / spacing) + 1))
    centers = endpoints[0] + np.linspace(0, 1, n)[:, None] * delta
    offsets = np.arange(-search, search + .25, .5)
    grid_ref = centers[:, None, :] + offsets[None, :, None] * normal
    grid = transform_points(grid_ref.reshape(-1, 2), matrix).reshape(grid_ref.shape)
    intensity = _sample(gray, grid)
    material = None
    if current_mask is not None:
        if current_mask.shape != gray.shape or not np.isfinite(current_mask).all():
            raise ValueError('Current mask must match image geometry')
        material = _sample((current_mask != 0).astype(np.uint8), grid)
    material_side = feature.get('material_side', 1)
    if material_side not in (-1, 1):
        raise ValueError('material_side must be -1 or +1')
    safe = cv2.erode(domain.astype(np.uint8), np.ones((5, 5), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0)
    valid_domain = _sample(safe, grid) > .999
    contrasts = intensity[:, 4:] - intensity[:, :-4]
    points = np.full((n, 2), np.nan)
    reasons = np.full(n, 'missing_edge', dtype=object)
    strengths = np.zeros(n)
    candidates = []
    for i in range(n):
        inside = np.logical_and.reduce([valid_domain[i, k:len(offsets)-4+k] for k in range(5)])
        choices = _candidate_transitions(intensity[i], contrasts[i],
            (polarity * contrasts[i] >= amplitude) & inside, offsets[2:-2], offsets)
        if material is not None:
            # Current foreground identity disambiguates interior texture and
            # neighbouring fixture edges; reference coordinates never fill it.
            choices = [item for item in choices
                       if np.interp(item[0] + 3 * material_side, offsets, material[i]) >= .75
                       and np.interp(item[0] - 3 * material_side, offsets, material[i]) <= .25
                       and abs(item[0]) <= search - 3]
        choices = sorted(choices, key=lambda item: item[1], reverse=True)
        candidates.append([{'offset_px': u, 'amplitude': a} for u, a, _ in choices])
        if not choices:
            reasons[i] = 'missing_edge' if inside.all() else 'missing_or_clipped_edge'
            continue
        # Brightness is not edge identity: a stronger fixture edge cannot win
        # over a weaker material edge merely because its gradient is larger.
        if len(choices) > 1:
            reasons[i] = 'ambiguous_edges'
            continue
        u, strengths[i], _ = choices[0]
        points[i] = transform_points(np.array([centers[i] + u * normal]), matrix)[0]
        reasons[i] = ''
    valid = np.isfinite(points).all(axis=1)
    result = {'name': feature['name'], 'role': feature.get('role', 'measurement'),
              'valid': False, 'reason': 'insufficient_observed_support',
              'points_xy': points, 'point_valid': valid, 'invalid_reason': reasons,
              'edge_amplitudes': strengths, 'candidates': candidates,
              'support_fraction': float(valid.mean()), 'taught_span_px': length,
              'length_semantics': 'observed support within taught span, not full physical edge length',
              'semantic_boundary_confirmed': False,
              'edge_identity_basis': 'current_segmentation' if material is not None else 'unique_signed_gradient',
              'distance_unit': 'input_pixel', 'parameter_snapshot': feature}
    if valid.sum() < 6 or valid.mean() < .65:
        return result
    line = cv2.fitLine(points[valid].astype(np.float32), cv2.DIST_HUBER, 0, .01, .001).ravel().astype(float)
    direction, origin = line[:2], line[2:]
    direction /= np.linalg.norm(direction)
    target_direction = matrix[:2, :2] @ tangent
    if np.dot(direction, target_direction) < 0:
        direction = -direction
    fitted_normal = np.array([-direction[1], direction[0]])
    residuals = (points[valid] - origin) @ fitted_normal
    abscissa = (points[valid] - origin) @ direction
    rms = float(np.sqrt(np.mean(residuals**2)))
    angle = float(np.degrees(np.arccos(np.clip(np.dot(direction, target_direction), -1, 1))))
    # The whole observed residual distribution is retained despite robust fitting.
    result.update(line_origin_xy=origin, line_direction_xy=direction, line_normal_xy=fitted_normal,
                  fit_rms_px=rms, fit_max_abs_residual_px=float(np.max(np.abs(residuals))),
                  angle_to_taught_deg=angle, observed_support_span_px=float(np.ptp(abscissa)),
                  observed_support_endpoints_xy=origin + np.array([abscissa.min(), abscissa.max()])[:, None]*direction)
    if rms > 2. or angle > 15.:
        result['reason'] = 'nonstraight_or_mismatched_edge'
    else:
        result.update(valid=True, reason='observed_line')
    return result


def hole_line_distances(centers_xy: np.ndarray, line: dict) -> list[dict]:
    """Measure center-to-infinite-line perpendicular distances and foot points.

    Report whether the foot lies within the observed line support; outside-foot
    distances involve line extrapolation and are not point-to-segment distances.
    """
    centers = np.asarray(centers_xy, float)
    if centers.ndim != 2 or centers.shape[1] != 2 or not np.isfinite(centers).all():
        raise ValueError('Finite N x 2 hole centers required')
    if not line.get('valid'):
        return [{'valid': False, 'distance_px': None, 'reason': line.get('reason')} for _ in centers]
    origin, normal = np.asarray(line['line_origin_xy']), np.asarray(line['line_normal_xy'])
    direction = np.asarray(line['line_direction_xy'])
    support = (np.asarray(line['observed_support_endpoints_xy']) - origin) @ direction
    rows = []
    for point in centers:
        signed = float((point-origin) @ normal)
        foot = point - signed * normal
        pos = float((foot-origin) @ direction)
        rows.append({'valid': True, 'distance_px': abs(signed), 'signed_distance_px': signed,
                     'perpendicular_foot_xy': foot,
                     'foot_within_observed_span': bool(support.min() <= pos <= support.max()),
                     'distance_kind': 'center_to_fitted_infinite_line'})
    return rows
