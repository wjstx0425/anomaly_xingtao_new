"""Pixel geometry for periodic contours; measurement never inserts missing edges."""
from __future__ import annotations

import numpy as np


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a forward homogeneous rigid transform to xy points."""
    points = np.asarray(points, dtype=float)
    return points @ np.asarray(matrix)[:2, :2].T + np.asarray(matrix)[:2, 2]


def rigid_fit(test_xy: np.ndarray, ref_xy: np.ndarray) -> np.ndarray:
    """Fit rotation and translation without scale, reflection, or affine freedom."""
    source, target = np.asarray(test_xy, float), np.asarray(ref_xy, float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2 or len(source) < 2:
        raise ValueError("Rigid fitting needs matching arrays of at least two xy anchors")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Anchor coordinates must be finite")
    x, y = source - source.mean(0), target - target.mean(0)
    if np.linalg.norm(x) < 1e-8 or np.linalg.norm(y) < 1e-8:
        raise ValueError("Coincident anchors cannot determine rotation")
    u, _, vt = np.linalg.svd(x.T @ y)
    correction = np.diag([1., np.linalg.det(vt.T @ u.T)])
    rotation = vt.T @ correction @ u.T
    matrix = np.eye(3)
    matrix[:2, :2] = rotation
    matrix[:2, 2] = target.mean(0) - rotation @ source.mean(0)
    return matrix


def point_to_segments(points: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact nearest segment distances, feet and indices with bounded memory."""
    points, starts, ends = (np.asarray(a, float).reshape(-1, 2) for a in (points, starts, ends))
    if starts.shape != ends.shape:
        raise ValueError("Segment endpoint arrays must have matching shapes")
    distance = np.full(len(points), np.inf)
    closest = np.full(points.shape, np.nan)
    indices = np.full(len(points), -1, dtype=int)
    for p0 in range(0, len(points), 128):
        p = points[p0:p0 + 128]
        best = np.full(len(p), np.inf)
        for s0 in range(0, len(starts), 1024):
            a, b = starts[s0:s0 + 1024], ends[s0:s0 + 1024]
            ab = b - a
            denominator = np.sum(ab * ab, axis=1)
            t = np.sum((p[:, None] - a) * ab, axis=2) / np.maximum(denominator, 1e-30)
            feet = a + np.clip(t, 0, 1)[..., None] * ab
            d2 = np.sum((p[:, None] - feet) ** 2, axis=2)
            d2 = np.where(np.isfinite(d2), d2, np.inf)
            j = np.argmin(d2, axis=1)
            candidate = d2[np.arange(len(p)), j]
            better = candidate < best
            rows = np.flatnonzero(better)
            best[rows] = candidate[rows]
            closest[p0 + rows] = feet[rows, j[rows]]
            indices[p0 + rows] = s0 + j[rows]
        distance[p0:p0 + len(p)] = np.sqrt(best)
    return distance, closest, indices


def resample_closed(dense_xy: np.ndarray, spacing: float) -> dict:
    """Uniformly parameterize an unsimplified closed curve with fixed arc cells."""
    dense = np.asarray(dense_xy, float).reshape(-1, 2)
    if len(dense) < 3 or not np.isfinite(dense).all() or not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("A finite contour and positive spacing are required")
    dense = dense[np.linalg.norm(dense - np.roll(dense, 1, axis=0), axis=1) > 1e-10]
    if len(dense) < 3:
        raise ValueError("Contour has fewer than three distinct vertices")
    delta = np.roll(dense, -1, axis=0) - dense
    lengths = np.linalg.norm(delta, axis=1)
    perimeter = lengths.sum()
    count = max(3, int(np.ceil(perimeter / spacing)))
    cell = perimeter / count
    s = np.arange(count) * cell
    cumulative = np.r_[0., np.cumsum(lengths)]
    segment = np.minimum(np.searchsorted(cumulative, s, side="right") - 1, len(dense) - 1)
    points = dense[segment] + ((s - cumulative[segment]) / lengths[segment])[:, None] * delta[segment]
    # Signed polygon area determines the inside mathematically; reference teaching
    # additionally verifies these normals against its foreground mask.
    area2 = np.sum(dense[:, 0] * np.roll(dense[:, 1], -1) - dense[:, 1] * np.roll(dense[:, 0], -1))
    if abs(area2) < 1e-8:
        raise ValueError("Contour encloses no material")
    tangent = np.roll(points, -2, axis=0) - np.roll(points, 2, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-12)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0])) * np.sign(area2)
    before, after = points - np.roll(points, 3, axis=0), np.roll(points, -3, axis=0) - points
    cosine = np.sum(before * after, axis=1) / np.maximum(np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1), 1e-12)
    return {"sample_xy": points, "s_px": s, "cell_length_px": np.full(count, cell),
            "inward_normal_xy": normal, "corner_mask": cosine < np.cos(np.deg2rad(35.))}


def cyclic_runs(mask: np.ndarray) -> list[np.ndarray]:
    """Return contiguous true runs, joining the periodic first and last cells."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return []
    if mask.all():
        return [np.arange(len(mask))]
    starts = np.flatnonzero(mask & ~np.roll(mask, 1))
    runs = []
    for start in starts:
        ids = []
        i = int(start)
        while mask[i]:
            ids.append(i)
            i = (i + 1) % len(mask)
        runs.append(np.asarray(ids, int))
    return runs
