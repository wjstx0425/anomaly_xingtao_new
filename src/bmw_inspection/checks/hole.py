"""Single-hole image geometry diagnostics; no manufacturing-cause inference."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from bmw_inspection.checks.stamp_config import _rectangle
from bmw_inspection.views import VIEW_ORDER


def _number(value: object, name: str, low: float, high: float) -> float:
    if type(value) not in (float, int) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite in [{low}, {high}]")
    return float(value)


def _load_config(path: str | Path) -> dict:
    """Load explicit geometry or diagnostic measurements, resolving relative masks."""
    path = Path(path)
    c = json.loads(path.read_text(encoding="utf-8"))
    if c.get("schema_version") != 1 or c.get("status") not in ("ready", "diagnostic"):
        raise ValueError("schema_version=1 and status=ready or diagnostic required; draft cannot run")
    diagnostic = c["status"] == "diagnostic"
    for key in ("part_id", "feature_id", "validation_evidence"):
        if not isinstance(c.get(key), str) or not c[key].strip():
            raise ValueError(f"Missing {key}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", c["feature_id"]):
        raise ValueError("feature_id must be a safe filename identifier")
    if c.get("hand") not in ("left", "right") or c.get("view_id") not in VIEW_ORDER:
        raise ValueError("Explicit hand and canonical view_id required")
    if c.get("coordinate_space") != "full_image":
        raise ValueError("Only full_image xyxy coordinates are supported")
    size = c.get("reference_image_size")
    if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
        raise ValueError("reference_image_size must be [width,height]")
    x1, y1, x2, y2 = _rectangle(c.get("roi_xyxy"), tuple(size), "roi_xyxy")
    if c.get("source_channel") not in ("fused", "short", "long"):
        raise ValueError("Explicit source_channel required")
    if c.get("pixel_channel") not in ("gray", "blue", "green", "red"):
        raise ValueError("Explicit pixel_channel required")
    reg, assoc = c.get("registration", {}), c.get("association", {})
    registration_allowed = reg.get("validated") is True or (diagnostic and reg.get("validated") is False)
    if (reg.get("mode") != "fixture_fixed"
            or not registration_allowed
            or not isinstance(reg.get("evidence"), str) or not reg["evidence"].strip()):
        raise ValueError("Independent fixture registration with evidence required; ready needs validated=true")
    if assoc.get("exclusive_search_confirmed") is not True or not assoc.get("evidence"):
        raise ValueError("Target-exclusive ROI excluding neighboring holes must be confirmed")
    seg = c["segmentation"]
    _number(seg["threshold"], "threshold", 0, 255)
    _number(seg["stability_delta"], "stability_delta", 1, 255)
    if not diagnostic or seg["max_changed_fraction"] is not None:
        _number(seg["max_changed_fraction"], "max_changed_fraction", 0, 1)
    if seg["polarity"] not in ("bright", "dark"):
        raise ValueError("polarity must be bright or dark")
    limits = c["limits"]
    limit_fields = (("open_fraction", 1), ("area_difference", float("inf")),
                    ("shape_difference", 1), ("position_px", float("inf")))
    for suffix, upper in (() if diagnostic and limits is None else limit_fields):
        pkey = "pass_min_" + suffix if suffix == "open_fraction" else "pass_max_" + suffix
        nkey = "ng_max_" + suffix if suffix == "open_fraction" else "ng_min_" + suffix
        p, n = _number(limits[pkey], pkey, 0, upper), _number(limits[nkey], nkey, 0, upper)
        if (n >= p if suffix == "open_fraction" else n <= p):
            raise ValueError(f"{suffix} must have a nonempty REVIEW band")
    mask_path = path.parent / c["expected_mask"]
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.ndim != 2 or mask.shape != (y2-y1, x2-x1):
        raise ValueError("expected_mask must be ROI-sized grayscale PNG")
    if not np.isin(mask, [0, 255]).all() or not np.any(mask):
        raise ValueError("expected_mask must be nonempty binary 0/255")
    if np.any(mask[[0, -1], :]) or np.any(mask[:, [0, -1]]):
        raise ValueError("Expected hole must lie strictly inside exclusive ROI")
    c["_mask"] = mask > 0
    return c


def load_config(path: str | Path) -> dict:
    """Reject missing or malformed configuration fields with a diagnostic error."""
    try:
        return _load_config(path)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"Missing or malformed hole configuration: {exc}") from exc


@dataclass
class HoleResult:
    """JSON metadata and independently owned diagnostic images."""
    payload: dict
    images: dict[str, np.ndarray]

    def save(self, output: str | Path, *, source: str | None = None) -> None:
        """Persist to a fresh directory, including feature-specific evidence paths."""
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        payload = json.loads(json.dumps(self.payload, allow_nan=False))
        payload["source"] = source
        payload["evidence"] = {}
        for name, array in self.images.items():
            filename = f"{payload.get('feature_id') or 'input_error'}_{name}.png"
            if not cv2.imwrite(str(output / filename), array):
                raise OSError(f"Failed to save {filename}")
            payload["evidence"][name] = filename
        (output / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "summary.md").write_text(
            f"{payload['status']}: {payload['reason']}\n\n"
            + json.dumps(payload.get("metrics", {}), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def inspect_hole(image: np.ndarray, config: dict, *, part_id: str, view_id: str,
                 source_channel: str, source_kind: str, observation: dict,
                 synthesized_test_data: bool = False) -> HoleResult:
    """Measure one target in fixed fixture coordinates without seeking other holes.

    Observation evidence is caller supplied per capture: it is not an automated
    fixture/occlusion detector. Unknown observation quality never produces PASS/NG.
    """
    c = config
    payload = {"schema_version": 1, "check_id": "hole_open", "feature_id": c["feature_id"],
               "part_id": part_id, "hand": c["hand"], "view_id": view_id,
               "source_channel": source_channel, "source_kind": source_kind,
               "pixel_channel": c["pixel_channel"], "coordinate_space": "full_image",
               "roi_xyxy": c["roi_xyxy"], "observation": observation,
               "synthesized_test_data": synthesized_test_data,
               "diagnostic_only": c["status"] == "diagnostic",
               "config": {k: v for k, v in c.items() if not k.startswith("_")},
               "metrics": {}, "cause_uncertain": True,
               "requirement_results": {"11": "NOT_DETERMINED", "15": "NOT_DETERMINED"},
               "phase1_coverage_complete": False}
    images = {}

    def result(status: str, reason: str) -> HoleResult:
        payload.update(status=status, reason=reason)
        return HoleResult(payload, images)

    if (part_id != c["part_id"] or view_id != c["view_id"] or source_channel != c["source_channel"]):
        return result("ERROR", "Input part/view/source channel does not match configured target")
    if source_kind not in ("hdr", "fused_only") or (source_kind == "fused_only" and source_channel != "fused"):
        return result("ERROR", "Source provenance does not support requested exposure")
    if observation.get("raw_unmasked") is not True:
        return result("ERROR", "Unmasked original image must be explicitly confirmed")
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim not in (2, 3)
            or (image.ndim == 3 and image.shape[2] != 3)
            or list(image.shape[1::-1]) != c["reference_image_size"]):
        return result("ERROR", "Expected full-resolution uint8 gray or BGR image; no resizing")
    x1, y1, x2, y2 = c["roi_xyxy"]
    roi = image[y1:y2, x1:x2].copy()
    images["roi_original"] = roi
    if c["pixel_channel"] == "gray":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
    elif roi.ndim != 3:
        return result("ERROR", "Requested color channel is unavailable")
    else:
        gray = roi[:, :, {"blue": 0, "green": 1, "red": 2}[c["pixel_channel"]]].copy()
    h = c["_mask"]
    seg = c["segmentation"]
    def segment(t: float) -> np.ndarray:
        return gray > t if seg["polarity"] == "bright" else gray <= t
    b = segment(seg["threshold"])
    low = segment(max(0, seg["threshold"]-seg["stability_delta"]))
    high = segment(min(255, seg["threshold"]+seg["stability_delta"]))
    unstable = low != high
    stability_region = h | low | high
    n, _, stats, _ = cv2.connectedComponentsWithStats(b.astype(np.uint8), 8)
    contours, _ = cv2.findContours(b.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    ha, ba = int(h.sum()), int(b.sum())
    intersection = int((b & h).sum())
    pos = None
    if ba:
        hy, hx = np.nonzero(h)
        by, bx = np.nonzero(b)
        pos = float(np.hypot(bx.mean()-hx.mean(), by.mean()-hy.mean()))
    edge = h & ~cv2.erode(h.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    metrics = {"open_fraction": intersection/ha, "blocked_fraction": 1-intersection/ha,
               "area_px2": ba, "expected_area_px2": ha, "area_difference": abs(ba-ha)/ha,
               "shape_difference": 1-intersection/int((b | h).sum()),
               "position_deviation_px": pos, "component_count": n-1,
               "component_areas_px2": stats[1:, cv2.CC_STAT_AREA].tolist(),
               "edge_coverage": float((b & edge).sum()/edge.sum()),
               "threshold_changed_fraction": float(unstable.sum()/stability_region.sum())}
    payload["metrics"] = metrics
    payload["opening_observed"] = bool(intersection)
    payload["physical_hole_presence"] = "UNDETERMINED"
    images.update(selected_channel=gray, expected_mask=h.astype(np.uint8)*255,
                  open_mask=b.astype(np.uint8)*255, blocked_mask=(h & ~b).astype(np.uint8)*255)
    overlay = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR) if roi.ndim == 2 else roi.copy()
    expected_contours, _ = cv2.findContours(h.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, expected_contours, -1, (255, 0, 0), 1)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
    overlay[h & ~b] = (0, 0, 255)
    images["overlay"] = overlay
    full = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    full[y1:y2, x1:x2] = overlay
    cv2.rectangle(full, (x1, y1), (x2-1, y2-1), (255, 255, 0), 1)
    images["full_overlay"] = full
    if payload["diagnostic_only"]:
        return result("REVIEW", "仅诊断测量：孔验收边界/独立定位尚未完成确认，不作合格或判废结论")
    if (observation.get("fixture_verified") is not True or observation.get("observable") is not True
            or not isinstance(observation.get("evidence"), str) or not observation["evidence"].strip()):
        return result("REVIEW", "Independent fixture position / unobstructed observation requires confirmation")
    if metrics["threshold_changed_fraction"] > seg["max_changed_fraction"]:
        return result("REVIEW", "Threshold segmentation unstable; reflection/illumination requires review")
    if np.any(b[[0, -1], :]) or np.any(b[:, [0, -1]]):
        return result("REVIEW", "Open component touches search boundary; target association is uncertain")
    lim = c["limits"]
    if metrics["open_fraction"] <= lim["ng_max_open_fraction"]:
        payload["fact"] = "HOLE_NOT_OPEN"
        return result("NG", "孔未正常开放，原因待确认")
    comparisons = [(metrics["area_difference"], "area_difference"),
                   (metrics["shape_difference"], "shape_difference"), (pos, "position_px")]
    if any(v is not None and v >= lim["ng_min_"+key] for v, key in comparisons):
        payload["fact"] = "HOLE_GEOMETRY_OUT_OF_RANGE"
        return result("NG", "指定孔开放区域几何异常，原因待确认")
    if metrics["open_fraction"] >= lim["pass_min_open_fraction"] and all(
            v is not None and v <= lim["pass_max_"+key] for v, key in comparisons):
        return result("PASS", "Configured single-hole opening and geometry within confirmed limits")
    return result("REVIEW", "Measurements fall between confirmed PASS and NG boundaries")
