"""Once-only reference teaching and immutable numeric bundle loading."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np

from .contracts import ContourInputError, check_image, read_image, read_json, validate_config, write_json


@contextmanager
def atomic_directory(output):
    """Publish a complete new directory, refusing existing or concurrent runs."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(output.name + ".lock")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    temporary = None
    try:
        os.close(fd)
        if output.exists():
            raise FileExistsError(output)
        temporary = Path(tempfile.mkdtemp(prefix=".contour-", dir=output.parent))
        yield temporary
        if output.exists():
            raise FileExistsError(output)
        temporary.rename(output)
        temporary = None
    finally:
        if temporary is not None:
            shutil.rmtree(temporary)
        lock.unlink(missing_ok=True)


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"image write failed: {path}")


def _material_probe(mask, points):
    probe = np.rint(points).astype(int)
    valid = ((probe[:, 0] >= 0) & (probe[:, 0] < mask.shape[1])
             & (probe[:, 1] >= 0) & (probe[:, 1] < mask.shape[0]))
    result = np.zeros(len(points), dtype=bool)
    result[valid] = mask[probe[valid, 1], probe[valid, 0]] > 0
    return result


def _normal_support(mask, xy, normals):
    """Report directional material support at one and two pixel distances.

    A thin exterior sliver can make both two-pixel probes foreground even
    when the one-pixel probes establish the direction. Equal occupancy supplies
    no evidence. Opposing directions at different distances remain ambiguous.
    """
    forward = np.zeros(len(xy), dtype=bool)
    backward = np.zeros(len(xy), dtype=bool)
    for distance in (1, 2):
        positive = _material_probe(mask, xy + normals * distance)
        negative = _material_probe(mask, xy - normals * distance)
        forward |= positive & ~negative
        backward |= negative & ~positive
    return forward, backward


def _arc_labels(ranges, count):
    if ranges is None:
        return np.zeros(count, dtype=np.int32)
    if not isinstance(ranges, list) or not ranges:
        raise ContourInputError("arc_ranges must be a nonempty full-coverage list")
    labels = np.full(count, -1, dtype=np.int32)
    for item in ranges:
        if not isinstance(item, dict):
            raise ContourInputError("arc_ranges entries must be objects")
        arc, start, end = (item.get(k) for k in ("arc_id", "start_sample", "end_sample"))
        if any(type(v) is not int for v in (arc, start, end)) or not (0 <= arc <= np.iinfo(np.int32).max and 0 <= start < end <= count):
            raise ContourInputError("invalid half-open arc range")
        if np.any(labels[start:end] != -1):
            raise ContourInputError("arc_ranges overlap")
        labels[start:end] = arc
    if np.any(labels < 0):
        raise ContourInputError("arc_ranges must cover the full reference")
    return labels


def _check_overrides(config, arc_ids):
    seen = set()
    for item in config["comparison"].get("arc_overrides", []):
        if not isinstance(item, dict):
            raise ContourInputError("arc override must be an object")
        arc = item.get("arc_id")
        if type(arc) is not int or arc in seen or not np.any(arc_ids == arc):
            raise ContourInputError("arc override must target a unique existing reference arc")
        seen.add(arc)


def teach_reference(image, draft_config, annotations):
    """Build full dense reference from an operator mask; never invent hidden edges.

    Annotations require ``mask`` and ``confirmed``. Optional ``unknown_sample_ids``
    leaves the model draft. Config overrides must be supplied before this call.
    """
    from .geometry import resample_closed
    c = validate_config(draft_config, draft=True)
    check_image(image, c)
    mask = np.asarray(annotations["mask"])
    if mask.shape != image.shape[:2] or mask.dtype != np.uint8:
        raise ContourInputError("annotation mask must be uint8 and match the image")
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    if image.shape[2] == 4 and np.any((image[:, :, 3] < 255) & (mask > 0)):
        raise ContourInputError("reference foreground includes transparent/invalid pixels")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 1:
        raise ContourInputError("reference must contain exactly one connected outer component")
    dense = contours[0][:, 0, :].astype(float)
    if len(dense) < 8 or cv2.contourArea(contours[0]) <= 4:
        raise ContourInputError("reference contour too small")
    roi = c.get("part_roi_xyxy")
    if roi is None:
        raise ContourInputError("part_roi_xyxy must include full part and surrounding background")
    x1, y1, x2, y2 = roi
    if np.any((dense[:, 0] <= x1) | (dense[:, 0] >= x2 - 1) | (dense[:, 1] <= y1) | (dense[:, 1] >= y2 - 1)):
        raise ContourInputError("reference contour touches/exceeds part ROI")
    data = resample_closed(dense, c["extraction"]["arc_spacing_px"])
    data["dense_xy"] = dense
    n = len(data["sample_xy"])
    data["required_mask"] = np.ones(n, dtype=bool)
    data["arc_id"] = _arc_labels(annotations.get("arc_ranges"), n)
    c["reference_arc_ranges"] = annotations.get("arc_ranges")
    _check_overrides(c, data["arc_id"])
    data["reference_valid"] = np.ones(n, dtype=bool)
    unknown = np.asarray(annotations.get("unknown_sample_ids", []), dtype=int)
    if len(unknown) and (unknown.min() < 0 or unknown.max() >= n):
        raise ContourInputError("unknown_sample_ids outside reference")
    data["reference_valid"][unknown] = False
    # Check normals against actual material rather than relying on winding.
    xy = data["sample_xy"]
    normals = data["inward_normal_xy"]
    inside, opposite_inside = _normal_support(mask, xy, normals)
    normals[~inside & opposite_inside] *= -1
    # No directional evidence, or conflicting evidence across distances,
    # leaves polarity unconfirmed. Neither case may be repaired by flipping.
    data["reference_valid"] &= inside != opposite_inside
    c["reference_confirmed"] = bool(annotations.get("confirmed", False) and data["reference_valid"].all())
    c["reference_normal_unconfirmed_sample_ids"] = np.flatnonzero(inside == opposite_inside).tolist()
    c["reference_bundle"] = "."
    c["reference_version"] = annotations.get("reference_version", "v001")
    c["config_version"] = c.get("config_version", "v001")
    c["reference_identity"] = {"hand": c["hand"], "view_id": c["view_id"], "image": copy.deepcopy(c["image"])}
    # Derived development thresholds are suggestions with explicit recorded statistics.
    gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
    def sample(points):
        return cv2.remap(gray, points[:, 0].astype(np.float32).reshape(-1, 1), points[:, 1].astype(np.float32).reshape(-1, 1), cv2.INTER_LINEAR).ravel()
    amplitudes = np.abs(sample(xy + normals * 2) - sample(xy - normals * 2))
    q10 = float(np.quantile(amplitudes, .1))
    suggestions = {
        ("extraction", "min_edge_amplitude"): max(3.0, q10 * .35),
        ("extraction", "min_candidate_margin"): .2,
        ("registration", "max_rms_residual_px"): 1.5,
        ("comparison", "short_large_peak_review_px"): c["comparison"]["default_inward_tolerance_px"] * 2,
    }
    provenance = c.setdefault("parameter_provenance", {})
    provenance["reference_contrast_statistics"] = {"q10_amplitude": q10, "sample_count": n}
    for (section, key), value in suggestions.items():
        if c[section].get(key) is None:
            c[section][key] = value
            provenance[f"{section}.{key}"] = {"value": value, "source": "single_reference_development_suggestion" if key == "min_edge_amplitude" else "explicit_development_start", "independently_validated": False}
    provenance["input_draft"] = {"distance_and_length": "legacy_local_baseline_only", "search_and_pose": "development_start_only"}
    return {**data, "image": image.copy(), "mask": mask, "config": c}


def save_reference(reference, output):
    """Atomically save recipe, exact input, full mask and sampled geometry."""
    with atomic_directory(output) as tmp:
        save_image(tmp / "reference.png", reference["image"])
        save_image(tmp / "reference_foreground_mask.png", reference["mask"])
        overlay = reference["image"][:, :, :3].copy()
        cv2.polylines(overlay, [np.rint(reference["dense_xy"]).astype(np.int32)], True, (0, 220, 0), 1)
        save_image(tmp / "reference_overlay.png", overlay)
        keys = ("dense_xy", "sample_xy", "s_px", "cell_length_px", "inward_normal_xy", "arc_id", "required_mask", "corner_mask", "reference_valid")
        np.savez_compressed(tmp / "contour.npz", **{k: reference[k] for k in keys})
        c = copy.deepcopy(reference["config"])
        c["asset_sha256"] = {name: hashlib.sha256((tmp / name).read_bytes()).hexdigest() for name in ("reference.png", "reference_foreground_mask.png", "contour.npz")}
        write_json(tmp / "recipe.json", c)


def load_reference(config_path):
    """Load and validate non-pickled reference geometry and image identity."""
    config_path = Path(config_path)
    c = validate_config(read_json(config_path))
    identity = c.get("reference_identity", {})
    if identity != {"hand": c["hand"], "view_id": c["view_id"], "image": c["image"]}:
        raise ContourInputError("reference hand/view/image identity mismatch")
    base = config_path.parent / c["reference_bundle"]
    for name in ("reference.png", "reference_foreground_mask.png", "contour.npz"):
        digest = hashlib.sha256((base / name).read_bytes()).hexdigest()
        if c.get("asset_sha256", {}).get(name) != digest:
            raise ContourInputError(f"reference asset checksum mismatch: {name}")
    image = read_image(base / "reference.png", c)
    mask = cv2.imread(str(base / "reference_foreground_mask.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != image.shape[:2]:
        raise ContourInputError("reference mask dimensions invalid")
    with np.load(base / "contour.npz", allow_pickle=False) as bundle:
        data = {k: bundle[k].copy() for k in bundle.files}
    required_keys = {"dense_xy", "sample_xy", "s_px", "cell_length_px", "inward_normal_xy", "arc_id", "required_mask", "corner_mask", "reference_valid"}
    if not required_keys <= data.keys():
        raise ContourInputError("reference geometry missing required arrays")
    if data["sample_xy"].ndim != 2:
        raise ContourInputError("invalid reference sample_xy")
    n = len(data["sample_xy"])
    for key in ("sample_xy", "inward_normal_xy"):
        if data[key].shape != (n, 2) or data[key].dtype.kind not in "fiu" or not np.isfinite(data[key]).all():
            raise ContourInputError(f"invalid reference {key}")
    for key in ("s_px", "cell_length_px", "arc_id", "required_mask", "corner_mask", "reference_valid"):
        if data[key].shape != (n,) or data[key].dtype.kind not in "bifu" or not np.isfinite(data[key]).all():
            raise ContourInputError(f"invalid reference {key}")
    for key in ("required_mask", "corner_mask", "reference_valid"):
        if data[key].dtype.kind != "b":
            raise ContourInputError(f"reference {key} must be a boolean array")
    if data["arc_id"].dtype.kind not in "iu" or np.any(data["arc_id"] < 0):
        raise ContourInputError("reference arc_id must be nonnegative integers")
    if not np.array_equal(data["arc_id"], _arc_labels(c.get("reference_arc_ranges"), n)):
        raise ContourInputError("reference arc labels do not match recorded ranges")
    _check_overrides(c, data["arc_id"])
    if n < 8 or not np.all(data["required_mask"]) or not np.all(data["reference_valid"]) or np.any(data["cell_length_px"] <= 0):
        raise ContourInputError("reference incomplete or invalid arc lengths")
    dense = data["dense_xy"]
    if dense.ndim != 2 or dense.shape[1] != 2 or len(dense) < 8 or dense.dtype.kind not in "fiu" or not np.isfinite(dense).all():
        raise ContourInputError("invalid dense contour")
    x1, y1, x2, y2 = c["part_roi_xyxy"]
    if np.any((dense[:, 0] <= x1) | (dense[:, 0] >= x2 - 1) | (dense[:, 1] <= y1) | (dense[:, 1] >= y2 - 1)):
        raise ContourInputError("dense contour touches/exceeds image or part ROI")
    if image.shape[2] == 4 and np.any((image[:, :, 3] < 255) & (mask > 0)):
        raise ContourInputError("reference foreground includes transparent pixels")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 1:
        raise ContourInputError("reference mask must have one outer component")
    actual = contours[0][:, 0].astype(float)
    # Allow a fixed alternate start/winding, but no geometry absent from the mask.
    matches = np.flatnonzero(np.all(actual == dense[0], axis=1))
    if len(actual) != len(dense) or not any(
        np.array_equal(dense, np.roll(actual, -int(i), axis=0))
        or np.array_equal(dense, np.roll(actual, -int(i), axis=0)[np.r_[0, np.arange(len(actual) - 1, 0, -1)]])
        for i in matches
    ):
        raise ContourInputError("dense contour does not match the reference mask boundary")
    from .geometry import resample_closed
    expected = resample_closed(dense, c["extraction"]["arc_spacing_px"])
    for key in ("sample_xy", "s_px", "cell_length_px"):
        if data[key].shape != expected[key].shape or not np.allclose(data[key], expected[key], rtol=1e-9, atol=1e-7):
            raise ContourInputError(f"reference {key} is inconsistent with dense arc parameterization")
    perimeter = float(np.linalg.norm(np.roll(dense, -1, axis=0) - dense, axis=1).sum())
    if abs(data["s_px"][0]) > 1e-8 or np.any(np.diff(data["s_px"]) <= 0) or not np.isclose(data["cell_length_px"].sum(), perimeter, rtol=1e-9):
        raise ContourInputError("reference arc length origin/order/perimeter invalid")
    normals = data["inward_normal_xy"]
    if not np.allclose(np.linalg.norm(normals, axis=1), 1, rtol=1e-7, atol=1e-7):
        raise ContourInputError("reference normals must have unit length")
    inside, outside = _normal_support(mask, data["sample_xy"], normals)
    if not np.all(inside & ~outside):
        raise ContourInputError("reference normals lack verified inward material support")
    if not np.allclose(np.abs(np.sum(normals * expected["inward_normal_xy"], axis=1)), 1, rtol=1e-7, atol=1e-7):
        raise ContourInputError("reference normals inconsistent with local tangents")
    if not np.array_equal(data["corner_mask"], expected["corner_mask"]):
        raise ContourInputError("reference corner labels inconsistent with geometry")
    for value in data.values():
        value.flags.writeable = False
    image.flags.writeable = mask.flags.writeable = False
    return {**data, "image": image, "mask": mask, "config": c}
