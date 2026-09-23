"""Assign observed foreground to the part without imposing its nominal boundary."""

from __future__ import annotations

import cv2
import numpy as np


def segment_part(image: np.ndarray, reference_mask: np.ndarray | None, config: dict, registration: dict | None = None) -> dict:
    """Segment the full image, then identify current components by core overlap.

    GrabCut receives only mutable probability labels inside the valid image.
    A reference core is used *after* segmentation for component identity, never
    to add foreground pixels. Connected protrusions are retained regardless of
    their distance to the nominal outline. Detached foreground near the nominal
    part is recorded as unresolved, and cannot be interpreted as measured excess.

    Settings in ``extraction`` are ``grabcut_iterations`` (5),
    ``component_core_inset_px`` (4), ``component_min_core_overlap_px`` (4), and
    ``component_unknown_distance_px`` (the larger profile range, default 40).
    All effective settings are included in the returned diagnostics. Teaching
    without a mask uses the interior of ``part_roi_xyxy`` for component identity;
    ambiguous multiple components remain explicit and require reference review.
    """
    # Imported at call time to preserve extraction's backwards-compatible public
    # helpers while letting coarse_segment delegate here without import cycles.
    from .extraction import _channels, _matrix

    bgr, _, domain = _channels(image)
    height, width = domain.shape
    settings = config.get("extraction", config)
    iterations = int(settings.get("grabcut_iterations", 5))
    inset = float(settings.get("component_core_inset_px", 4))
    overlap_min = int(settings.get("component_min_core_overlap_px", 4))
    unknown_distance = float(settings.get("component_unknown_distance_px", max(float(settings.get("search_inward_px", 40)), float(settings.get("search_outward_px", 40)))))
    if iterations < 1 or overlap_min < 1 or not np.isfinite([inset, unknown_distance]).all() or min(inset, unknown_distance) < 0:
        raise ValueError("Invalid component assignment settings")
    labels = np.full(domain.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    if reference_mask is not None:
        if reference_mask.shape != domain.shape:
            raise ValueError("Reference and test image sizes differ")
        prior = cv2.warpAffine((reference_mask != 0).astype(np.uint8), np.linalg.inv(_matrix(registration))[:2], (width, height), flags=cv2.INTER_NEAREST)
    else:
        roi = config.get("part_roi_xyxy")
        if roi is None:
            raise ValueError("Teaching requires part_roi_xyxy or an explicit reference mask")
        x1, y1, x2, y2 = map(int, roi)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("Invalid half-open part ROI")
        prior = np.zeros(domain.shape, np.uint8)
        prior[y1:y2, x1:x2] = 1
    labels[prior != 0] = cv2.GC_PR_FGD
    labels[~domain] = cv2.GC_BGD
    if not np.any(labels == cv2.GC_PR_FGD) or not np.any(labels == cv2.GC_PR_BGD):
        raise ValueError("Segmentation requires foreground and background probability seeds")
    # Fix initialization for reproducible reference/current observation. This
    # seeds the algorithm; it does not branch on file identity or reference fit.
    seed = int(settings.get("grabcut_seed", 0))
    cv2.setRNGSeed(seed)
    cv2.grabCut(bgr, labels, None, np.zeros((1, 65)), np.zeros((1, 65)), iterations, cv2.GC_INIT_WITH_MASK)
    raw = ((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)).astype(np.uint8)
    core = (cv2.distanceTransform(prior, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) > inset) & domain
    count, ids, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    core_overlap = np.bincount(ids[core], minlength=count)
    distance = cv2.distanceTransform(1 - prior, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    # Vectorized reductions avoid a full image scan for every tiny speck.
    nearest = np.full(count, np.inf)
    np.minimum.at(nearest, ids.ravel(), distance.ravel())
    owned_ids = np.flatnonzero(core_overlap >= overlap_min)
    owned_ids = owned_ids[owned_ids != 0]
    owned_lookup = np.zeros(count, np.uint8)
    owned_lookup[owned_ids] = 1
    mask = owned_lookup[ids]
    components = []
    uncertain_ids = []
    excluded_ids = []
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        owned = bool(owned_lookup[label])
        uncertain = not owned and nearest[label] <= unknown_distance
        if uncertain:
            uncertain_ids.append(label)
        elif not owned:
            excluded_ids.append(label)
        components.append({"label": label, "area_px2": area, "bbox_xywh": [x, y, w, h], "core_overlap_px": int(core_overlap[label]), "min_distance_to_prior_px": float(nearest[label]), "image_clipped": x == 0 or y == 0 or x + w == width or y + h == height, "assignment": "part_core_overlap" if owned else "unresolved_detached_foreground" if uncertain else "unassigned_distant_foreground"})
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    pieces = []
    for contour in contours:
        if pieces:
            pieces.append(np.full((1, 2), np.nan))
        pieces.append(contour[:, 0].astype(float))
    # A bright island inside an internal hole is a connected component, but
    # RETR_EXTERNAL correctly excludes it from the whole outer perimeter.
    owned_clipped = any(item["image_clipped"] for item in components if item["assignment"] == "part_core_overlap")
    diagnostics = {"components": components, "component_count": len(contours), "core_overlapping_connected_component_count": len(owned_ids), "raw_component_count": count - 1,
                   "excluded_component_count": len(excluded_ids), "unresolved_component_count": len(uncertain_ids),
                   "multiple_components": len(contours) > 1, "image_clipped": owned_clipped,
                   "component_assignment_uncertain": bool(uncertain_ids), "part_component_missing": not len(owned_ids),
                   "probable_reference_seed_only": True, "component_assignment_method": "current_foreground_reference_core_overlap",
                   "component_settings": {"grabcut_iterations": iterations, "grabcut_seed": seed, "component_core_inset_px": inset,
                                          "component_min_core_overlap_px": overlap_min, "component_unknown_distance_px": unknown_distance}}
    return {"mask": mask, "dense_xy": np.concatenate(pieces) if pieces else np.empty((0, 2)), "diagnostics": diagnostics}
