"""Strict offline contour recipe and image contracts."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import cv2
import numpy as np


class ContourInputError(ValueError):
    """An input cannot be measured under this recipe."""


def json_ready(value):
    """Copy numerical metadata to strict JSON, preserving unknown as null."""
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(v) for v in value]
    return value


def read_json(path):
    """Reject nonstandard JSON floating point constants."""
    def reject(value):
        raise ContourInputError(f"Nonfinite JSON constant: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def write_json(path, value):
    Path(path).write_text(json.dumps(json_ready(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def check_image(image, config):
    """Validate uint8 image geometry without resizing or editing caller storage."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ContourInputError("image must be uint8")
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ContourInputError("image must be BGR or BGRA")
    expected = (config["image"]["height"], config["image"]["width"])
    if image.shape[:2] != expected:
        raise ContourInputError(f"image size {image.shape[:2]} does not match {expected}")


def read_image(path, config):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    check_image(image, config)
    return image


def validate_config(config, *, draft=False):
    """Validate all safety-critical recipe fields before extraction."""
    if not isinstance(config, dict):
        raise ContourInputError("configuration must be an object")
    c = copy.deepcopy(config)
    # A serialization round trip rejects NaN also for programmatic callers.
    try:
        json.dumps(c, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ContourInputError("configuration must contain finite JSON values") from exc
    for section in ("image", "registration", "extraction", "comparison", "coverage"):
        if not isinstance(c.get(section), dict):
            raise ContourInputError(f"{section} must be an object")
    if c.get("schema_version") != 1 or c.get("check_type") != "contour_compare":
        raise ContourInputError("unsupported schema/check_type")
    if c.get("hand") not in ("left", "right") or c.get("view_id") not in (
        "front", "front_left", "front_right", "front_secondary", "back", "back_left", "back_right", "back_secondary"
    ):
        raise ContourInputError("invalid hand/view_id")
    if c.get("deployment_enabled") is not False or c.get("validation_level") != "development":
        raise ContourInputError("this offline implementation supports development recipes only")
    for key in ("width", "height"):
        v = c["image"][key]
        if type(v) is not int or v < 2:
            raise ContourInputError(f"image.{key} must be a positive integer")
    if c["image"].get("channel") not in ("fused", "short", "long"):
        raise ContourInputError("unsupported image.channel")
    w, h = c["image"]["width"], c["image"]["height"]
    def roi(value, name):
        if value is None and draft:
            return
        if not isinstance(value, list) or len(value) != 4 or any(type(x) is not int for x in value):
            raise ContourInputError(f"{name}: integer xyxy ROI required")
        x1, y1, x2, y2 = value
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ContourInputError(f"{name}: ROI outside image")
    roi(c.get("part_roi_xyxy"), "part_roi_xyxy")
    reg = c["registration"]
    if "coarse_search" in reg:
        coarse = reg["coarse_search"]
        if not isinstance(coarse, dict) or type(coarse.get("enabled")) is not bool:
            raise ContourInputError("coarse_search requires an explicit boolean enabled")
        allowed = {"enabled", "search_margin_px", "max_candidates_per_anchor", "max_axis_change_fraction"}
        if set(coarse) - allowed:
            raise ContourInputError("unsupported coarse_search option")
        if coarse["enabled"]:
            for key in ("search_margin_px", "max_axis_change_fraction"):
                value = coarse.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                    raise ContourInputError(f"coarse_search.{key} must be positive finite")
            if coarse["max_axis_change_fraction"] >= 1:
                raise ContourInputError("coarse_search.max_axis_change_fraction must be below one")
            limit = coarse.get("max_candidates_per_anchor")
            if type(limit) is not int or not 1 <= limit <= 100:
                raise ContourInputError("coarse_search.max_candidates_per_anchor must be an integer in 1..100")
    if reg.get("method") != "rigid_2d_anchors" or len(reg.get("anchor_rois_xyxy", [])) < 2:
        raise ContourInputError("at least two explicit rigid anchors required")
    for r in reg["anchor_rois_xyxy"]:
        roi(r, "anchor_rois_xyxy")
    for r in reg.get("independent_check_regions", []):
        roi(r, "independent_check_regions")
    if reg.get("independent_check_regions"):
        raise ContourInputError("independent check regions are not implemented; use empty list and retain pose limitation")
    required = {
        "registration": ("max_rotation_deg", "max_anchor_motion_px", "max_anchor_spacing_change_px", "max_rms_residual_px"),
        "extraction": ("arc_spacing_px", "profile_step_px", "search_inward_px", "search_outward_px", "min_edge_amplitude", "min_candidate_margin"),
        "comparison": ("default_inward_tolerance_px", "default_outward_tolerance_px", "min_exceedance_arc_px", "short_large_peak_review_px"),
    }
    missing = []
    for section, keys in required.items():
        for key in keys:
            v = c[section].get(key)
            if v is None:
                missing.append(f"{section}.{key}")
            elif isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0:
                raise ContourInputError(f"{section}.{key} must be positive finite")
    ex = c["extraction"]
    if ex.get("edge_selection_mode", "independent_v2") not in ("independent_v2", "continuous_v3"):
        raise ContourInputError("unsupported extraction.edge_selection_mode")
    for key in ("reference_edge_search_px", "path_continuity_weight", "path_ambiguity_margin", "path_jump_cap_px", "corner_persistence_radius_px", "corner_persistence_angle_deg"):
        if key in ex and (isinstance(ex[key], bool) or not isinstance(ex[key], (int, float)) or ex[key] <= 0):
            raise ContourInputError(f"extraction.{key} must be positive finite")
    if ex.get("corner_persistence_angle_deg", 25.) > 180:
        raise ContourInputError("extraction.corner_persistence_angle_deg must not exceed 180")
    for key in ("min_profile_similarity", "min_profile_similarity_margin"):
        if key in ex and (isinstance(ex[key], bool) or not isinstance(ex[key], (int, float)) or not 0 < ex[key] <= 1):
            raise ContourInputError(f"extraction.{key} must be in (0, 1]")
    for key in ("coarse_reference_agreement_px", "component_core_inset_px", "component_unknown_distance_px"):
        if key in ex and (isinstance(ex[key], bool) or not isinstance(ex[key], (int, float)) or ex[key] <= 0):
            raise ContourInputError(f"extraction.{key} must be positive finite")
    for key in ("grabcut_iterations", "component_min_core_overlap_px"):
        if key in ex and (type(ex[key]) is not int or ex[key] < 1):
            raise ContourInputError(f"extraction.{key} must be a positive integer")
    if "grabcut_seed" in ex and (type(ex["grabcut_seed"]) is not int or not 0 <= ex["grabcut_seed"] < 2**31):
        raise ContourInputError("extraction.grabcut_seed must be a nonnegative int32")
    if ex.get("nominal_snap_allowed") is not False or ex.get("interpolate_missing_for_measurement") is not False:
        raise ContourInputError("nominal snapping/interpolation forbidden")
    if ex.get("morphology_open_size_px", 0) != 0 or ex.get("morphology_close_size_px", 0) != 0:
        raise ContourInputError("measurement morphology is unsupported")
    if ex.get("coarse_method") != "grabcut_probable_seed" or ex.get("corner_strategy") != "local_2d":
        raise ContourInputError("unsupported extraction method")
    if c["comparison"].get("distance_mode") != "bidirectional_point_to_segment":
        raise ContourInputError("unsupported distance mode")
    if c["comparison"].get("merge_valid_gap_px", 0) != 0:
        raise ContourInputError("nonzero event gap merging not yet supported")
    seen_arcs = set()
    for override in c["comparison"].get("arc_overrides", []):
        if not isinstance(override, dict):
            raise ContourInputError("arc override must be an object")
        arc = override.get("arc_id")
        if type(arc) is not int or arc < 0 or arc in seen_arcs:
            raise ContourInputError("arc override IDs must be unique nonnegative integers")
        seen_arcs.add(arc)
        for key, value in override.items():
            if key == "arc_id":
                continue
            if key not in ("inward_tolerance_px", "outward_tolerance_px", "min_exceedance_arc_px"):
                raise ContourInputError(f"unsupported arc override: {key}")
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or value <= 0:
                raise ContourInputError(f"arc override {key} must be positive finite")
    if ex.get("min_candidate_margin") is not None and ex["min_candidate_margin"] > 1:
        raise ContourInputError("min_candidate_margin must not exceed 1")
    coverage = c["coverage"]
    if coverage.get("excluded_arc_ids") or coverage.get("require_full_reference_perimeter") is not True:
        raise ContourInputError("this version requires the complete perimeter")
    if coverage.get("max_unobserved_required_arc_px") != 0:
        raise ContourInputError("unknown required arcs cannot be accepted")
    if not draft:
        if missing:
            raise ContourInputError("draft fields unresolved: " + ", ".join(missing))
        if not c.get("reference_bundle") or c.get("reference_confirmed") is not True:
            raise ContourInputError("confirmed reference_bundle required; recipe remains draft")
    return c
