"""Conservative corner measurements along an observed, supported dense chain."""
from __future__ import annotations

import numpy as np
import cv2


def track_corner_runs(nominal_xy, corner_mask, sample_xy, valid, dense_xy, dense_valid, *, max_distance_px=40.) -> dict:
    """Track corner runs between adjacent reliable profile anchors.

    Every returned coordinate lies on a supported dense segment. A unique
    observed turn anchors signed arc distances from the nominal corner; the
    reference never supplies replacement coordinates. Endpoints are sought on
    nearby supported flanks, bounded by max_distance_px of nominal arc travel.
    NaN separators, unsupported vertices, ambiguous turns/anchors and implausible
    path lengths leave the entire run unknown. Dense contours must use pixel-adjacent
    vertices, as produced by CHAIN_APPROX_NONE. Inputs are never modified.
    """
    nominal = np.asarray(nominal_xy, float)
    points = np.asarray(sample_xy, float).copy()
    corners = np.asarray(corner_mask, bool)
    good = np.asarray(valid, bool).copy()
    dense = np.asarray(dense_xy, float)
    supported = np.asarray(dense_valid, bool)
    n = len(nominal)
    if (nominal.shape != (n, 2) or points.shape != nominal.shape or corners.shape != (n,)
            or good.shape != (n,) or dense.ndim != 2 or dense.shape[1] != 2
            or supported.shape != (len(dense),) or n < 3 or not np.isfinite(nominal).all()
            or np.isinf(dense).any() or np.any(supported & ~np.isfinite(dense).all(axis=1))
            or np.any(good & ~np.isfinite(points).all(axis=1))
            or not np.isfinite(max_distance_px) or max_distance_px <= 0):
        raise ValueError("Invalid corner tracking inputs")
    original_good = good.copy()
    good[corners] = False
    points[corners] = np.nan
    reasons = np.full(n, "", dtype=object)
    reasons[corners] = "corner_chain_unconfirmed"
    diagnostics = {"strategy": "supported_dense_chain_unique_turn_two_flanks", "runs": []}
    if corners.all() or not corners.any():
        return {"sample_xy": points, "valid": good, "invalid_reason": reasons, "diagnostics": diagnostics}
    finite = np.isfinite(dense).all(axis=1)
    # Separate components before considering cyclic closure. A closing edge is
    # allowed only when its endpoints are actually pixel adjacent.
    components = []
    for ids in np.split(np.flatnonzero(finite), np.flatnonzero(np.diff(np.flatnonzero(finite)) != 1) + 1):
        if len(ids) >= 2:
            components.append(ids)
    starts = np.flatnonzero(corners & ~np.roll(corners, 1))
    def turn_landmark(chain):
        approx = cv2.approxPolyDP(np.asarray(chain, np.float32), .75, False)[:, 0]
        if len(approx) < 3:
            return None
        vectors = np.diff(approx, axis=0)
        angles = np.arctan2(vectors[:-1, 0] * vectors[1:, 1] - vectors[:-1, 1] * vectors[1:, 0],
                            np.sum(vectors[:-1] * vectors[1:], axis=1))
        turns = np.flatnonzero(np.abs(angles) >= np.deg2rad(30))
        if len(turns) != 1:
            return None
        k = int(turns[0])
        index = int(np.argmin(np.linalg.norm(chain - approx[k + 1], axis=1)))
        return index, float(angles[k]), vectors[k] / np.linalg.norm(vectors[k]), vectors[k + 1] / np.linalg.norm(vectors[k + 1])

    for start in starts:
        run = []
        j = int(start)
        while corners[j]:
            run.append(j)
            j = (j + 1) % n
        left, right = (start - 1) % n, j
        record = {"sample_ids": run, "status": "UNKNOWN"}
        diagnostics["runs"].append(record)
        # Search outwards along the existing topology, bounded by physical arc
        # distance and by other corner runs. Never manufacture a missing anchor.
        for side in (-1, 1):
            k = left if side == -1 else right
            travelled = 0.
            first_supported_at = None
            while not corners[k] and travelled <= max_distance_px:
                if original_good[k]:
                    if first_supported_at is None:
                        first_supported_at = travelled
                    # Keep a small supported flank beyond the first candidate;
                    # that candidate can lie exactly on the displaced turn.
                    if travelled - first_supported_at >= 3.:
                        break
                following = (k + side) % n
                travelled += np.linalg.norm(nominal[following] - nominal[k])
                k = following
            if travelled > max_distance_px:
                k = int(start)  # corner sentinel, explicitly rejected below
            if side == -1:
                left = k
            else:
                right = k
        if not original_good[left] or not original_good[right] or corners[left] or corners[right]:
            record["reason"] = "missing_profile_anchor"
            continue
        record["anchor_sample_ids"] = [int(left), int(right)]
        anchors = points[[left, right]]
        matches = []
        for component in components:
            ds = np.linalg.norm(dense[component, None, :] - anchors[None, :, :], axis=2)
            for k in range(2):
                ds[~supported[component], k] = np.inf
            ia, ib = np.argmin(ds, axis=0)
            if max(ds[ia, 0], ds[ib, 1]) > 2.:
                continue
            if ia == ib:
                continue
            matches.append((component, int(ia), int(ib)))
        if len(matches) != 1:
            record["reason"] = "ambiguous_or_different_component_anchors"
            continue
        component, ia, ib = matches[0]
        expected_ids = [int(left)]
        while expected_ids[-1] != right:
            expected_ids.append((expected_ids[-1] + 1) % n)
        expected = nominal[expected_ids]
        nominal_steps = np.linalg.norm(np.diff(expected, axis=0), axis=1)
        expected_length = nominal_steps.sum()
        nominal_arc = np.r_[0., np.cumsum(nominal_steps)]
        expected_turn = turn_landmark(expected)
        if expected_turn is None:
            record["reason"] = "nominal_turn_ambiguous"
            continue
        targets_nominal = nominal_arc[[expected_ids.index(i) for i in run]]
        candidates = []
        for direction in (1, -1):
            local = [ia]
            k = ia
            while k != ib and len(local) <= len(component):
                k = (k + direction) % len(component)
                local.append(k)
            ids = component[local]
            chain = dense[ids]
            steps = np.linalg.norm(np.diff(chain, axis=0), axis=1)
            if not supported[ids].all() or np.any(steps > 1.5) or np.any(steps <= 0):
                continue
            length = steps.sum()
            if not .25 * expected_length <= length <= 4 * expected_length:
                continue
            arc = np.r_[0., np.cumsum(steps)]
            observed_turn = turn_landmark(chain)
            if observed_turn is None:
                continue
            ni, na, nt0, nt1 = expected_turn
            ci, ca, ct0, ct1 = observed_turn
            if abs(na - ca) > np.deg2rad(25) or min(np.dot(nt0, ct0), np.dot(nt1, ct1)) < np.cos(np.deg2rad(30)):
                continue
            # Anchor the material turn itself; endpoint arc-fraction interpolation
            # would spread tangential motion across the corner and erase its shift.
            targets = arc[ci] + targets_nominal - nominal_arc[ni]
            if np.any(targets < 0) or np.any(targets > length):
                continue
            measured = np.column_stack([np.interp(targets, arc, chain[:, axis]) for axis in (0, 1)])
            # Constrain the entire path, not just resampled points: long detours
            # cannot hide between output samples.
            near = np.min(np.linalg.norm(chain[:, None, :] - expected[None, :, :], axis=2), axis=1)
            if np.any(near > max_distance_px) or np.any(np.linalg.norm(measured - nominal[run], axis=1) > max_distance_px):
                continue
            candidates.append((measured, length, ids))
        if len(candidates) != 1:
            record["reason"] = "ambiguous_or_unsupported_dense_path"
            continue
        measured, length, ids = candidates[0]
        points[run] = measured
        good[run] = True
        reasons[run] = ""
        record.update(status="OBSERVED", nominal_span_px=float(expected_length), measured_path_length_px=float(length),
                      dense_vertex_ids=ids.tolist(), anchor_sample_ids=[int(left), int(right)])
    for record in diagnostics["runs"]:
        if record["status"] == "UNKNOWN":
            reasons[record["sample_ids"]] = record["reason"]
    return {"sample_xy": points, "valid": good, "invalid_reason": reasons, "diagnostics": diagnostics}


def persistent_corner_mask(nominal_xy, corner_mask, cell_length_px, *, radius_px=6., min_angle_deg=25.) -> np.ndarray:
    """Retain original corner candidates with a persistent 25-degree turn.

    Tangents use vertices six arc pixels before and after each candidate. Arc
    travel is capped below half the perimeter, so narrow closed contours never
    wrap onto the same flank. Only the returned classification changes; geometry,
    sampling lengths, normals and required coverage are not modified. Default
    development settings are 6 input pixels and 25 degrees, not physical limits.
    """
    xy = np.asarray(nominal_xy, float)
    raw = np.asarray(corner_mask)
    lengths = np.asarray(cell_length_px, float)
    if (xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 or raw.shape != (len(xy),)
            or lengths.shape != (len(xy),) or not np.isfinite(xy).all()
            or not np.isfinite(lengths).all() or np.any(lengths <= 0)
            or raw.dtype.kind != 'b' or isinstance(radius_px, (bool, np.bool_))
            or isinstance(min_angle_deg, (bool, np.bool_)) or not np.isfinite([radius_px, min_angle_deg]).all()
            or radius_px <= 0 or not 0 < min_angle_deg <= 180):
        raise ValueError('Invalid persistent corner mask inputs')
    perimeter = float(lengths.sum())
    # Interpolate only for estimating coarse tangent directions; these points
    # never become observations or replace the reference geometry.
    arc = np.r_[0., np.cumsum(lengths)]
    closed = np.vstack([xy, xy[0]])
    span = min(float(radius_px), perimeter / 4.)
    centers = arc[:-1]
    before = np.column_stack([np.interp((centers - span) % perimeter, arc, closed[:, axis]) for axis in (0, 1)])
    after = np.column_stack([np.interp((centers + span) % perimeter, arc, closed[:, axis]) for axis in (0, 1)])
    incoming, outgoing = xy - before, after - xy
    norms = np.linalg.norm(incoming, axis=1) * np.linalg.norm(outgoing, axis=1)
    cosine = np.divide(np.sum(incoming * outgoing, axis=1), norms, out=np.ones(len(xy)), where=norms > 1e-12)
    angle = np.arccos(np.clip(cosine, -1., 1.))
    # Degenerate tangent evidence is not grounds to erase an existing corner.
    return raw.copy() & ((angle >= np.deg2rad(min_angle_deg)) | (norms <= 1e-12))
