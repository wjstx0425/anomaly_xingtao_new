"""R02a development evidence from directional dark and bright line responses."""

from __future__ import annotations

from time import perf_counter

import cv2
import numpy as np
from skimage.morphology import skeletonize


def _settings(image, config):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("R02a requires an unchanged uint8 BGR image")
    if not isinstance(config, dict):
        raise ValueError("R02a config must be a dictionary")
    shape = config.get("image_shape_hw")
    if not isinstance(shape, (list, tuple)) or len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape):
        raise ValueError("image_shape_hw must explicitly contain positive integer height and width")
    if tuple(shape) != image.shape[:2]:
        raise ValueError("Input image shape differs from the configured image_shape_hw")
    roi = config.get("roi_xyxy")
    if not isinstance(roi, (list, tuple)) or len(roi) != 4 or any(type(v) is not int for v in roi):
        raise ValueError("roi_xyxy must explicitly contain four integer coordinates")
    x1, y1, x2, y2 = roi
    if not 0 <= x1 < x2 <= shape[1] or not 0 <= y1 < y2 <= shape[0]:
        raise ValueError("roi_xyxy must be a nonempty half-open region inside the input")
    threshold = config.get("response_threshold")
    sigma = config.get("background_sigma_px", 15.0)
    for name, value in (("response_threshold", threshold), ("background_sigma_px", sigma)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be explicit positive finite grayscale/pixel units")
    area = config.get("min_area_px")
    if type(area) is not int or area <= 0:
        raise ValueError("min_area_px must explicitly be a positive integer development threshold")
    widths = config.get("widths_px", [3, 7, 15])
    if not isinstance(widths, (list, tuple)) or not 1 <= len(widths) <= 8 or any(type(v) is not int or v < 3 or v > 255 or v % 2 != 1 for v in widths):
        raise ValueError("widths_px requires one to eight odd kernel spans between 3 and 255 pixels")
    angles = config.get("orientations_deg", [0, 45, 90, 135])
    if not isinstance(angles, (list, tuple)) or not 1 <= len(angles) <= 16 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or not 0 <= v < 180 for v in angles):
        raise ValueError("orientations_deg requires one to sixteen finite angles in [0, 180)")
    return tuple(roi), float(threshold), area, tuple(widths), tuple(angles), float(sigma)


def _line_kernel(span, angle):
    kernel = np.zeros((span, span), np.uint8)
    radius = (span - 1) / 2
    delta = radius * np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
    start, end = (np.rint(radius + sign * delta).astype(int) for sign in (-1, 1))
    cv2.line(kernel, tuple(start), tuple(end), 1, 1)
    return kernel


def _shape_description(component):
    skeleton = skeletonize(component)
    horizontal = np.count_nonzero(skeleton[:, :-1] & skeleton[:, 1:])
    vertical = np.count_nonzero(skeleton[:-1] & skeleton[1:])
    # Do not count a diagonal shortcut when an orthogonal skeleton route exists.
    diagonal_a = skeleton[:-1, :-1] & skeleton[1:, 1:] & ~skeleton[:-1, 1:] & ~skeleton[1:, :-1]
    diagonal_b = skeleton[:-1, 1:] & skeleton[1:, :-1] & ~skeleton[:-1, :-1] & ~skeleton[1:, 1:]
    length = float(horizontal + vertical + np.sqrt(2) * (diagonal_a.sum() + diagonal_b.sum()))
    if length == 0:
        length = float(skeleton.sum())
    padded = np.pad(component.astype(np.uint8), 1)
    distances = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    widths = np.maximum(1.0, 2.0 * distances[skeleton] - 1.0)
    y, x = np.nonzero(component)
    orientation = None
    if len(x) >= 3:
        covariance = np.cov(np.column_stack([x, y]), rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if eigenvalues[-1] > max(1e-9, 1.1 * eigenvalues[0]):
            direction = eigenvectors[:, -1]
            orientation = float(np.degrees(np.arctan2(direction[1], direction[0])) % 180)
    return length, float(np.median(widths)) if len(widths) else 0.0, orientation


def predict_lines(image: np.ndarray, config: dict) -> dict:
    """Return native-pixel line evidence without using labels or image paths.

    Required config fields are ``image_shape_hw``, ``roi_xyxy``,
    ``response_threshold`` and ``min_area_px``. Thresholds are development
    settings, never defect probabilities or acceptance limits. Optional kernel
    spans, angles and Gaussian background sigma are recorded in the result.
    Kernel spans control probe width, not a measured defect-width guarantee.

    Arrays retain original image dimensions; zeros outside the observed ROI
    mean unobserved, not normal. Boxes use original half-open xyxy coordinates.
    Dark and bright responses are kept independently. This function does not
    assign crack/scratch classes or promote REVIEW to PASS/NG without truth
    calibration. Inputs are not modified, resized, or read from disk.
    """
    start = perf_counter()
    roi, threshold, min_area, widths, angles, sigma = _settings(image, config)
    x1, y1, x2, y2 = roi
    height, width = image.shape[:2]
    # Read surrounding image context so an ROI edge is not itself an artificial
    # morphology/background boundary. Only the approved ROI is reported.
    padding = int(np.ceil(4 * sigma)) + max(widths)
    left, top = max(0, x1 - padding), max(0, y1 - padding)
    right, bottom = min(width, x2 + padding), min(height, y2 + padding)
    gray = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY).astype(np.float32)
    background = cv2.GaussianBlur(gray, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
    flattened = gray - background
    preprocessed = perf_counter()

    dark = np.zeros_like(flattened)
    bright = np.zeros_like(flattened)
    for span in widths:
        for angle in angles:
            kernel = _line_kernel(span, angle)
            np.maximum(dark, cv2.morphologyEx(flattened, cv2.MORPH_BLACKHAT, kernel, borderType=cv2.BORDER_REFLECT_101), out=dark)
            np.maximum(bright, cv2.morphologyEx(flattened, cv2.MORPH_TOPHAT, kernel, borderType=cv2.BORDER_REFLECT_101), out=bright)
    dark = dark[y1 - top:y2 - top, x1 - left:x2 - left]
    bright = bright[y1 - top:y2 - top, x1 - left:x2 - left]
    inferred = perf_counter()

    binary = ((dark >= threshold) | (bright >= threshold)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    retained = np.zeros_like(binary)
    boxes = []
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        if area < min_area:
            continue
        component = labels[y:y + h, x:x + w] == label
        retained[y:y + h, x:x + w][component] = 255
        length, local_width, orientation = _shape_description(component)
        peak_dark = float(dark[y:y + h, x:x + w][component].max())
        peak_bright = float(bright[y:y + h, x:x + w][component].max())
        boxes.append({
            "xyxy": [x + x1, y + y1, x + x1 + w, y + y1 + h],
            "area_px": area, "skeleton_length_px": length, "width_px": local_width,
            "orientation_deg": orientation, "peak_dark_response": peak_dark,
            "peak_bright_response": peak_bright,
            "evidence_polarity": "mixed" if min(peak_dark, peak_bright) >= threshold else "dark" if peak_dark >= threshold else "bright",
            "touches_observed_roi_boundary": bool(x == 0 or y == 0 or x + w == x2 - x1 or y + h == y2 - y1),
        })
    response_dark = np.zeros((height, width), np.float32)
    response_bright = np.zeros((height, width), np.float32)
    candidate_mask = np.zeros((height, width), np.uint8)
    response_dark[y1:y2, x1:x2] = dark
    response_bright[y1:y2, x1:x2] = bright
    candidate_mask[y1:y2, x1:x2] = retained
    missing = [box for box in ([0, 0, width, y1], [0, y2, width, height], [0, y1, x1, y2], [x2, y1, width, y2]) if box[0] < box[2] and box[1] < box[3]]
    finished = perf_counter()
    return {
        "algorithm_id": "R02a", "execution_status": "SUCCESS", "decision": "REVIEW",
        "reason": "development_line_evidence_without_crack_or_scratch_truth_calibration",
        "calibration_status": "development_uncalibrated",
        "raw_score": float(max(dark.max(), bright.max())), "score_direction": "higher_more_line_response",
        "score_units": "grayscale_response", "threshold": threshold,
        "response_dark": response_dark, "response_bright": response_bright,
        "candidate_mask": candidate_mask, "boxes": boxes,
        "coordinate_transform": np.eye(3).tolist(), "observed_roi_xyxy": list(roi),
        "observed_regions": [list(roi)], "missing_regions": missing,
        "parameters": {"image_shape_hw": [height, width], "roi_xyxy": list(roi),
                       "response_threshold": threshold, "min_area_px": min_area,
                       "widths_px": list(widths), "orientations_deg": list(angles), "background_sigma_px": sigma,
                       "width_method": "median_skeleton_distance_diameter_minus_one",
                       "skeleton_length_method": "pixel_graph_euclidean_without_diagonal_shortcuts",
                       "orientation_convention": "PCA_degrees_from_image_x_axis_modulo_180_null_if_isotropic"},
        "preprocess_ms": (preprocessed - start) * 1000,
        "inference_ms": (inferred - preprocessed) * 1000,
        "postprocess_ms": (finished - inferred) * 1000,
    }
