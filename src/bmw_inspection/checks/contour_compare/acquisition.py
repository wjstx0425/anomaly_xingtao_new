"""Acquire current segmentation candidates and independently supported edges.

This diagnostic output measures current candidate arc support, not coverage of
an approved reference. Image contrast cannot confirm material identity.
"""
from __future__ import annotations

import numpy as np

from .extraction import _candidate_transitions, _channels, _sample


def acquire_current_outline(image: np.ndarray, coarse: dict, *, min_edge_amplitude: float) -> dict:
    """Refine unique current-image transitions within three input pixels.

    Preserve every supplied dense vertex and separator as candidate evidence.
    Local mask context only determines inward direction and candidate support;
    no reference coordinates or shape priors enter the measurements. Corners,
    missing contrast, ambiguous transitions and disconnected search domains
    remain UNKNOWN, with NaN observed coordinates. Arc support counts original
    candidate segments whose two endpoints were independently observed.
    """
    _, gray, domain = _channels(image)
    mask = np.asarray(coarse['mask'])
    dense = np.asarray(coarse['dense_xy'], float)
    if (mask.shape != gray.shape or dense.ndim != 2 or dense.shape[1] != 2
            or np.isinf(dense).any() or not np.isfinite(mask).all()
            or isinstance(min_edge_amplitude, (bool, np.bool_))
            or not np.isfinite(min_edge_amplitude) or min_edge_amplitude <= 0):
        raise ValueError('Acquisition requires finite mask, dense XY and positive edge amplitude')
    finite = np.isfinite(dense).all(axis=1)
    n = len(dense)
    observed = np.full((n, 2), np.nan)
    valid = np.zeros(n, bool)
    reasons = np.full(n, 'component_separator', object)
    reasons[finite] = 'pixel_candidate_topology_unconfirmed'
    normals = np.full((n, 2), np.nan)
    ends = np.full(n, -1, int)
    lengths = np.zeros(n, float)
    components = np.split(np.flatnonzero(finite), np.flatnonzero(np.diff(np.flatnonzero(finite)) != 1) + 1)
    for ids in components:
        if len(ids) < 3:
            continue
        points = dense[ids]
        closed = 0 < np.linalg.norm(points[-1] - points[0]) <= 1.5
        for k, i in enumerate(ids):
            if k + 1 < len(ids) or closed:
                j = ids[(k + 1) % len(ids)]
                distance = np.linalg.norm(dense[j] - dense[i])
                if 0 < distance <= 1.5:
                    ends[i], lengths[i] = j, distance
            if not closed and (k < 2 or k >= len(ids) - 2):
                continue
            local_ids = ids[(np.arange(k - 2, k + 3) % len(ids))]
            local = dense[local_ids]
            steps = np.linalg.norm(np.diff(local, axis=0), axis=1)
            if np.any(steps <= 0) or np.any(steps > 1.5):
                continue
            incoming, outgoing = local[2] - local[0], local[4] - local[2]
            norm = np.linalg.norm(incoming) * np.linalg.norm(outgoing)
            if norm <= 0 or np.dot(incoming, outgoing) / norm < np.cos(np.deg2rad(45)):
                reasons[i] = 'pixel_candidate_corner'
                continue
            tangent = local[4] - local[0]
            normal = np.array([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
            normals[i] = normal
            reasons[i] = 'current_material_direction_unconfirmed'
    eligible = np.flatnonzero(np.isfinite(normals).all(axis=1))
    # Two additional pixels supply context around a transition searched within
    # +/-3 pixels; they cannot become output transition locations.
    offsets = np.arange(-5., 5.01, .5)
    if len(eligible):
        probes = dense[eligible, None] + np.array([-2., 2.])[None, :, None] * normals[eligible, None]
        sides = _sample((mask != 0).astype(np.uint8), probes)
        flip = (sides[:, 0] > .75) & (sides[:, 1] < .25)
        forward = (sides[:, 1] > .75) & (sides[:, 0] < .25)
        normals[eligible[flip]] *= -1
        eligible = eligible[flip | forward]
    if len(eligible):
        profiles = dense[eligible, None] + offsets[None, :, None] * normals[eligible, None]
        intensities = _sample(gray, profiles)
        materials = _sample((mask != 0).astype(np.uint8), profiles)
        domains = _sample(domain, profiles) > .999
        h, w = gray.shape
        domains &= (profiles[..., 0] >= 0) & (profiles[..., 0] <= w - 1) & (profiles[..., 1] >= 0) & (profiles[..., 1] <= h - 1)
        for row, i in enumerate(eligible):
            reasons[i] = 'clipped_or_transparent_search'
            if not domains[row].all():
                continue
            values = intensities[row]
            contrasts = values[4:] - values[:-4]
            accepted = (np.abs(contrasts) >= min_edge_amplitude) & (np.abs(offsets[2:-2]) <= 3.)
            transitions = _candidate_transitions(values, contrasts, accepted, offsets[2:-2], offsets)
            context = float(np.median(values[-3:]) - np.median(values[:3]))
            supported = []
            for u, strength, contrast in transitions:
                # All support probes must remain inside the acquired profile.
                if abs(u) > 3. or contrast * context <= 0 or abs(context) < min_edge_amplitude:
                    continue
                inside = np.interp(u + 2., offsets, materials[row])
                outside = np.interp(u - 2., offsets, materials[row])
                if inside >= .75 and outside <= .25:
                    supported.append(u)
            reasons[i] = 'ambiguous_current_image_edges' if len(supported) > 1 else 'no_supported_current_image_edge'
            if len(supported) == 1:
                observed[i] = dense[i] + supported[0] * normals[i]
                valid[i] = True
                reasons[i] = ''
    segment_valid = np.zeros(n, bool)
    connected = np.flatnonzero(ends >= 0)
    segment_valid[connected] = valid[connected] & valid[ends[connected]]
    candidate_length = float(lengths.sum())
    supported_length = float(lengths[segment_valid].sum())
    return {
        'candidate_xy': dense.copy(), 'observed_xy': observed, 'valid': valid,
        'invalid_reason': reasons, 'segment_end_index': ends, 'segment_valid': segment_valid,
        'candidate_segment_length_px': lengths, 'candidate_arc_length_px': candidate_length,
        'image_supported_arc_length_px': supported_length,
        'image_supported_candidate_arc_fraction': supported_length / candidate_length if candidate_length else 0.,
        'semantic_boundary_confirmed': False, 'diagnostic_only': True,
        'coverage_basis': 'current_segmentation_candidate_arc_not_fixed_reference',
        'diagnostics': {'segmentation': coarse.get('diagnostics', {}), 'search_radius_px': 3., 'context_radius_px': 5.,
                        'min_edge_amplitude': float(min_edge_amplitude), 'corner_angle_deg': 45.,
                        'measurement_method': 'unique_current_mask_supported_image_transition'},
    }
