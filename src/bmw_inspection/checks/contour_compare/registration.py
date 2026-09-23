"""Independent hole anchors and constrained SE(2) pose estimation."""
from __future__ import annotations

from itertools import product

import cv2
import numpy as np

from .geometry import rigid_fit, transform_points


def _anchor_candidates(image: np.ndarray, roi: list, settings: dict, *, local_contrast: bool = False) -> dict:
    x1, y1, x2, y2 = map(int, roi)
    if not (0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]):
        return {"valid": False, "reason": "anchor_roi_out_of_bounds"}
    crop = image[y1:y2, x1:x2]
    if crop.ndim == 3 and crop.shape[2] == 4 and np.any(crop[..., 3] != 255):
        return {"valid": False, "reason": "anchor_alpha_invalid"}
    gray = cv2.cvtColor(crop[..., :3], cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if len(contour) < 12 or area < settings.get("min_anchor_area_px", 30) or x <= 0 or y <= 0 or x + w >= x2-x1 or y+h >= y2-y1:
            continue
        (cx, cy), (a, b), angle = cv2.fitEllipse(contour)
        if min(a, b) < 3 or max(a, b) / min(a, b) > settings.get("max_anchor_axis_ratio", 3.):
            continue
        theta = np.deg2rad(angle)
        rotation = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        xy = (contour[:, 0].astype(float) - [cx, cy]) @ rotation.T
        radial = np.sqrt((xy[:, 0] / (a / 2)) ** 2 + (xy[:, 1] / (b / 2)) ** 2)
        residual = float(np.sqrt(np.mean(((radial - 1) * min(a, b)/2) ** 2)))
        contrast_crop = gray
        if local_contrast:
            padding = max(3, int(max(w, h) * .25))
            contrast_crop = gray[max(0, y-padding):min(gray.shape[0], y+h+padding),
                                 max(0, x-padding):min(gray.shape[1], x+w+padding)]
        contrast = float(np.percentile(contrast_crop, 90) - np.percentile(contrast_crop, 10))
        if residual > settings.get("max_anchor_ellipse_residual_px", 2.) or contrast < settings.get("min_anchor_contrast", 15.):
            continue
        candidates.append({"valid": True, "center_xy": [cx+x1, cy+y1], "area_px": area,
                           "ellipse_axes_px": [a, b], "ellipse_residual_px": residual, "contrast": contrast})
    return {"valid": True, "candidates": candidates}


def _anchor(image: np.ndarray, roi: list, settings: dict) -> dict:
    extraction = _anchor_candidates(image, roi, settings)
    if not extraction["valid"]:
        return extraction
    candidates = extraction["candidates"]
    if len(candidates) != 1:
        return {"valid": False, "reason": "anchor_missing_or_ambiguous", "candidate_count": len(candidates)}
    return candidates[0]


def _coarse_anchors(image: np.ndarray, anchors: list[dict], settings: dict) -> dict:
    """Find a unique bounded rigid constellation of full-resolution hole ellipses.

    Only hole candidates participate. Enlarged ROIs change search coverage, never
    the final fit's scale or reflection. Multiple feasible assignments fail closed.
    """
    options = settings["coarse_search"]
    margin = options.get("search_margin_px", settings.get("max_anchor_motion_px", 0))
    axis_fraction = options.get("max_axis_change_fraction", .2)
    limit = options.get("max_candidates_per_anchor", 20)
    diagnostics = {"method": "bounded_hole_constellation", "search_rois_xyxy": [],
                   "candidate_counts": [], "shape_filtered_counts": [], "feasible_assignment_count": 0,
                   "rejected_assignments": {"duplicate_hole": 0, "spacing": 0, "rotation": 0,
                                            "motion": 0, "residual": 0}}
    out = {"valid": False, "reason": "coarse_anchor_missing_or_ambiguous", "diagnostics": diagnostics}
    thresholds = [settings.get(key) for key in ("max_anchor_spacing_change_px", "max_rotation_deg",
                  "max_anchor_motion_px", "max_rms_residual_px")]
    if (isinstance(margin, bool) or not isinstance(margin, (int, float)) or not np.isfinite(margin) or margin <= 0
            or isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100
            or isinstance(axis_fraction, bool) or not isinstance(axis_fraction, (int, float)) or not np.isfinite(axis_fraction)
            or not 0 < axis_fraction < 1
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0
                   for value in thresholds)):
        out["reason"] = "coarse_search_invalid_settings"
        return out
    spacing_limit, rotation_limit, motion_limit, residual_limit = thresholds
    pools = []
    for item in anchors:
        x1, y1, x2, y2 = item["roi_xyxy"]
        roi = [max(0, int(np.floor(x1-margin))), max(0, int(np.floor(y1-margin))),
               min(image.shape[1], int(np.ceil(x2+margin))), min(image.shape[0], int(np.ceil(y2+margin)))]
        diagnostics["search_rois_xyxy"].append(roi)
        # A large Otsu window includes dark background and can swallow the hole.
        # Keep the taught window size while scanning the bounded search region.
        width, height = int(x2-x1), int(y2-y1)
        ref_axes = np.sort(item["reference"]["ellipse_axes_px"])
        max_diameter = float(max(ref_axes))*(1+axis_fraction)
        step_x = max(1, min(width//3, int(width-max_diameter-4)))
        step_y = max(1, min(height//3, int(height-max_diameter-4)))
        xs = sorted(set(range(roi[0], roi[2]-width+1, step_x)) | {int(x1), roi[2]-width})
        ys = sorted(set(range(roi[1], roi[3]-height+1, step_y)) | {int(y1), roi[3]-height})
        if len(xs)*len(ys) > 2500:
            out["reason"] = "coarse_search_window_limit_exceeded"
            return out
        diagnostics.setdefault("window_counts", []).append(len(xs)*len(ys))
        candidates = []
        for x in xs:
            for y in ys:
                extraction = _anchor_candidates(image, [x, y, x+width, y+height], settings, local_contrast=True)
                if extraction["valid"]:
                    candidates.extend(extraction["candidates"])
        diagnostics["candidate_counts"].append(len(candidates))
        ref_axes = np.sort(item["reference"]["ellipse_axes_px"])
        candidates = [candidate for candidate in candidates
                      if np.max(np.abs(np.sort(candidate["ellipse_axes_px"])/ref_axes-1)) <= axis_fraction]
        # Repeated local windows observe the same hole. Cluster only sub-hole
        # offsets, retaining the ellipse closest in size to the reference.
        candidates.sort(key=lambda candidate: (float(np.max(np.abs(
            np.sort(candidate["ellipse_axes_px"])/ref_axes-1))), candidate["ellipse_residual_px"]))
        unique = []
        cluster_radius = max(2., float(min(ref_axes))*.1)
        for candidate in candidates:
            if all(np.linalg.norm(np.array(candidate["center_xy"])-other["center_xy"]) > cluster_radius
                   for other in unique):
                unique.append(candidate)
        candidates = unique
        diagnostics["shape_filtered_counts"].append(len(candidates))
        if len(candidates) > limit:
            out["reason"] = "coarse_search_candidate_limit_exceeded"
            return out
        pools.append(candidates)
    if np.prod([len(pool) for pool in pools], dtype=float) > 10000:
        out["reason"] = "coarse_search_assignment_limit_exceeded"
        return out
    ref = np.array([item["reference"]["center_xy"] for item in anchors])
    pairs = np.triu_indices(len(ref), 1)
    ref_spacing = np.linalg.norm((ref[:, None]-ref)[pairs], axis=1)
    accepted = []
    for assignment in product(*pools):
        test = np.array([candidate["center_xy"] for candidate in assignment])
        test_spacing = np.linalg.norm((test[:, None]-test)[pairs], axis=1)
        if np.min(test_spacing) < 1:
            diagnostics["rejected_assignments"]["duplicate_hole"] += 1
            continue
        spacing = float(np.max(np.abs(test_spacing-ref_spacing)))
        if spacing > spacing_limit:
            diagnostics["rejected_assignments"]["spacing"] += 1
            continue
        try:
            matrix = rigid_fit(test, ref)
        except ValueError:
            diagnostics["rejected_assignments"]["duplicate_hole"] += 1
            continue
        rotation = abs(float(np.rad2deg(np.arctan2(matrix[1, 0], matrix[0, 0]))))
        motion = float(np.max(np.linalg.norm(test-ref, axis=1)))
        rms = float(np.sqrt(np.mean(np.sum((transform_points(test, matrix)-ref)**2, axis=1))))
        violation = next((key for key, value, threshold in (("rotation", rotation, rotation_limit),
                          ("motion", motion, motion_limit), ("residual", rms, residual_limit))
                          if value > threshold), None)
        if violation:
            diagnostics["rejected_assignments"][violation] += 1
            continue
        accepted.append((assignment, {"anchor_spacing_change_px": spacing, "rms_residual_px": rms,
                                      "rotation_deg": rotation, "max_anchor_motion_px": motion}))
    diagnostics["feasible_assignment_count"] = len(accepted)
    diagnostics["feasible_assignment_quality"] = [quality for _, quality in accepted]
    if len(accepted) == 1:
        out.update(valid=True, reason=None, candidates=list(accepted[0][0]))
    return out


def estimate_rigid_pose(image: np.ndarray, reference_image: np.ndarray, config: dict) -> dict:
    """Estimate test-to-reference rigid pose or return explicit unavailable matrices."""
    settings = config.get("registration", config)
    result = {"valid": False, "reason_codes": [], "T_test_to_reference": None,
              "T_reference_to_test": None, "anchors": [], "quality": {}}
    if image.shape != reference_image.shape or image.dtype != np.uint8 or reference_image.dtype != np.uint8:
        result["reason_codes"] = ["registration_image_spec_mismatch"]
        return result
    rois = settings.get("anchor_rois_xyxy", [])
    if len(rois) < 2:
        result["reason_codes"] = ["insufficient_anchor_rois"]
        return result
    coarse = settings.get("coarse_search", {})
    if not isinstance(coarse, dict) or not isinstance(coarse.get("enabled", False), bool):
        result["reason_codes"] = ["coarse_search_invalid_settings"]
        return result
    enabled = coarse.get("enabled", False)
    for roi in rois:
        result["anchors"].append({"roi_xyxy": roi, "reference": _anchor(reference_image, roi, settings),
                                  "test": {"valid": False} if enabled else _anchor(image, roi, settings)})
    if enabled and all(item["reference"]["valid"] for item in result["anchors"]):
        coarse_result = _coarse_anchors(image, result["anchors"], settings)
        result["coarse_search"] = coarse_result["diagnostics"]
        if not coarse_result["valid"]:
            result["reason_codes"] = [coarse_result["reason"]]
            return result
        result["coarse_search"]["refinement"] = []
        for item, candidate in zip(result["anchors"], coarse_result["candidates"], strict=True):
            delta = np.array(candidate["center_xy"])-item["reference"]["center_xy"]
            shift = np.rint(delta).astype(int)
            roi = (np.array(item["roi_xyxy"])+np.tile(shift, 2)).tolist()
            refined = _anchor(image, roi, settings)
            item["test"] = refined
            refinement = {"roi_xyxy": roi, "coarse_center_xy": candidate["center_xy"],
                          "valid": refined["valid"]}
            if refined["valid"]:
                refinement["center_shift_from_coarse_px"] = float(np.linalg.norm(
                    np.array(refined["center_xy"])-candidate["center_xy"]))
            else:
                refinement["reason"] = refined["reason"]
            result["coarse_search"]["refinement"].append(refinement)
    if any(not item[side]["valid"] for item in result["anchors"] for side in ("reference", "test")):
        result["reason_codes"] = ["anchor_extraction_failed"]
        return result
    ref = np.array([a["reference"]["center_xy"] for a in result["anchors"]])
    test = np.array([a["test"]["center_xy"] for a in result["anchors"]])
    try:
        matrix = rigid_fit(test, ref)
    except ValueError:
        result["reason_codes"] = ["anchor_geometry_degenerate"]
        return result
    pairs = np.triu_indices(len(ref), 1)
    spacing_change = float(np.max(np.abs(np.linalg.norm((ref[:, None]-ref)[pairs], axis=1) - np.linalg.norm((test[:, None]-test)[pairs], axis=1))))
    rms = float(np.sqrt(np.mean(np.sum((transform_points(test, matrix)-ref)**2, axis=1))))
    rotation = float(np.rad2deg(np.arctan2(matrix[1, 0], matrix[0, 0])))
    motion = float(np.max(np.linalg.norm(test-ref, axis=1)))
    result["quality"] = {"rms_residual_px": rms, "anchor_spacing_change_px": spacing_change,
                         "rotation_deg": rotation, "max_anchor_motion_px": motion,
                         "translation_xy_px": matrix[:2, 2].tolist(),
                         "independent_pose_validation": False,
                         "limitation": "Anchor fit alone does not validate 3D pose or deformation of anchor-bearing material."}
    checks = [("max_rms_residual_px", rms), ("max_rotation_deg", abs(rotation)),
              ("max_anchor_motion_px", motion), ("max_anchor_spacing_change_px", spacing_change)]
    for key, measured in checks:
        threshold = settings.get(key)
        if threshold is None or not np.isfinite(threshold) or threshold <= 0:
            result["reason_codes"].append("missing_or_invalid_" + key)
        elif measured > threshold:
            result["reason_codes"].append("exceeded_" + key)
    if not result["reason_codes"]:
        result.update(valid=True, T_test_to_reference=matrix, T_reference_to_test=np.linalg.inv(matrix))
    return result
