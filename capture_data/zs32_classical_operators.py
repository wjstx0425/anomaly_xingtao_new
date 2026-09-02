# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Standalone classical diagnostic operators for ZS32 surface ROIs.

This module deliberately returns continuous evidence rather than production
inspection decisions. It is not connected to Stage 32, Stage 18, or Dashboard.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import cv2
import numpy as np

OperatorName = Literal["thin_line", "pit_spot"]


@dataclass(frozen=True)
class OperatorParameters:
    """Configuration shared by the standalone surface operators."""

    material_min_gray: int = 24
    material_blur_kernel: int = 31
    material_close_kernel: int = 11
    material_min_area_ratio: float = 0.02
    material_erode_pixels: int = 8
    normalize_clip_limit: float = 2.0
    normalize_tile_size: int = 8
    robust_mad_scale: float = 4.0
    thin_min_area: int = 12
    thin_max_area_ratio: float = 0.10
    thin_min_elongation: float = 3.0
    thin_max_thickness: float = 8.0
    spot_kernel_sizes: tuple[int, ...] = (9, 17, 25)
    spot_min_area: int = 12
    spot_max_area_ratio: float = 0.10
    spot_min_circularity: float = 0.18
    spot_max_elongation: float = 3.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe parameter mapping."""
        data = asdict(self)
        data["spot_kernel_sizes"] = list(self.spot_kernel_sizes)
        return data


@dataclass(frozen=True)
class ComponentEvidence:
    """Shape and contrast evidence for one retained connected component."""

    x: int
    y: int
    width: int
    height: int
    area: int
    mean_response: float
    max_response: float
    elongation: float
    thickness: float
    circularity: float

    def to_dict(self) -> dict[str, int | float]:
        """Return JSON-safe component evidence."""
        return asdict(self)


@dataclass(frozen=True)
class OperatorResult:
    """Continuous diagnostic output from one standalone operator."""

    operator: OperatorName
    score: float
    mask: np.ndarray
    response: np.ndarray
    material_mask: np.ndarray
    components: tuple[ComponentEvidence, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe scalar metadata without embedding image arrays."""
        return {
            "operator": self.operator,
            "score": self.score,
            "component_count": len(self.components),
            "components": [component.to_dict() for component in self.components],
            "mask_shape": list(self.mask.shape),
            "response_shape": list(self.response.shape),
            "material_mask_shape": list(self.material_mask.shape),
        }


def _validate_bgr_image(image: np.ndarray) -> None:
    """Validate the uint8 BGR input contract."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        msg = "Expected a BGR image with shape (height, width, 3)"
        raise ValueError(msg)
    if image.dtype != np.uint8:
        msg = "Expected a uint8 BGR image"
        raise ValueError(msg)
    if image.shape[0] == 0 or image.shape[1] == 0:
        msg = "Expected a non-empty BGR image"
        raise ValueError(msg)


def build_material_mask(image: np.ndarray, parameters: OperatorParameters) -> np.ndarray:
    """Segment substantial non-black material and remove its silhouette edge."""
    _validate_bgr_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_size = max(1, int(parameters.material_blur_kernel))
    if blur_size % 2 == 0:
        blur_size += 1
    smoothed = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    mask = np.where(smoothed >= parameters.material_min_gray, 255, 0).astype(np.uint8)

    close_size = max(1, int(parameters.material_close_kernel))
    if close_size % 2 == 0:
        close_size += 1
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    retained = np.zeros_like(mask)
    min_area = max(1, round(mask.size * parameters.material_min_area_ratio))
    for label in range(1, component_count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            retained[labels == label] = 255

    erode_pixels = max(0, int(parameters.material_erode_pixels))
    if erode_pixels:
        kernel_size = 2 * erode_pixels + 1
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        retained = cv2.erode(retained, erode_kernel)
    return retained


def _normalized_gray(image: np.ndarray, parameters: OperatorParameters) -> np.ndarray:
    """Return locally contrast-normalized grayscale."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tile_size = max(1, int(parameters.normalize_tile_size))
    clahe = cv2.createCLAHE(
        clipLimit=max(0.1, float(parameters.normalize_clip_limit)),
        tileGridSize=(tile_size, tile_size),
    )
    return clahe.apply(gray)


def _robust_binary(
    response: np.ndarray,
    material_mask: np.ndarray,
    parameters: OperatorParameters,
) -> np.ndarray:
    """Threshold positive response values using material-local robust statistics."""
    values = response[material_mask > 0]
    if values.size == 0 or float(np.max(values)) <= 0.0:
        return np.zeros(response.shape, dtype=np.uint8)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = 1.4826 * mad
    threshold = max(1.0, median + parameters.robust_mad_scale * robust_sigma)
    binary = np.where((response > threshold) & (material_mask > 0), 255, 0).astype(np.uint8)
    return cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )


def _component_measurements(
    binary: np.ndarray,
    response: np.ndarray,
    min_area: int,
    max_area: int,
) -> tuple[tuple[ComponentEvidence, np.ndarray], ...]:
    """Measure viable components and retain only bbox-local masks."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    measured: list[tuple[ComponentEvidence, np.ndarray]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if not min_area <= area <= max_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        label_roi = labels[y : y + height, x : x + width]
        component_mask = np.where(label_roi == label, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        (_, _), (rect_width, rect_height), _ = cv2.minAreaRect(contour)
        major = max(float(rect_width), float(rect_height), 1.0)
        minor = max(min(float(rect_width), float(rect_height)), 1.0)
        perimeter = float(cv2.arcLength(contour, closed=True))
        circularity = 0.0 if perimeter <= 0.0 else float(4.0 * np.pi * area / perimeter**2)
        response_roi = response[y : y + height, x : x + width]
        component_values = response_roi[component_mask > 0]
        measured.append(
            (
                ComponentEvidence(
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    area=area,
                    mean_response=float(np.mean(component_values)),
                    max_response=float(np.max(component_values)),
                    elongation=major / minor,
                    thickness=minor,
                    circularity=min(circularity, 1.0),
                ),
                component_mask,
            ),
        )
    return tuple(measured)


def _make_result(
    operator: OperatorName,
    response: np.ndarray,
    material_mask: np.ndarray,
    retained: list[tuple[ComponentEvidence, np.ndarray]],
) -> OperatorResult:
    """Build a result mask and continuous uncalibrated evidence score."""
    result_mask = np.zeros(response.shape, dtype=np.uint8)
    material_area = max(1, int(np.count_nonzero(material_mask)))
    weighted_evidence = 0.0
    for component, component_mask in retained:
        result_roi = result_mask[
            component.y : component.y + component.height,
            component.x : component.x + component.width,
        ]
        result_roi[component_mask > 0] = 255
        shape_weight = component.elongation if operator == "thin_line" else component.circularity
        shape_weight = max(0.1, min(shape_weight, 20.0))
        weighted_evidence += component.mean_response * component.area * shape_weight
    score = weighted_evidence / (255.0 * material_area)
    components = tuple(component for component, _ in retained)
    return OperatorResult(
        operator=operator,
        score=float(score),
        mask=result_mask,
        response=response.astype(np.float32, copy=False),
        material_mask=material_mask,
        components=components,
    )


def detect_thin_lines(image: np.ndarray, parameters: OperatorParameters) -> OperatorResult:
    """Detect elongated surface evidence."""
    _validate_bgr_image(image)
    normalized = _normalized_gray(image, parameters)
    gradient_x = cv2.Scharr(normalized, cv2.CV_32F, 1, 0)
    gradient_y = cv2.Scharr(normalized, cv2.CV_32F, 0, 1)
    gradient = cv2.magnitude(gradient_x, gradient_y) / 32.0
    laplacian = np.abs(cv2.Laplacian(normalized, cv2.CV_32F, ksize=3)) / 4.0
    response = np.maximum(gradient, laplacian)
    material_mask = build_material_mask(image, parameters)
    binary = _robust_binary(response, material_mask, parameters)

    material_area = max(1, int(np.count_nonzero(material_mask)))
    max_area = max(parameters.thin_min_area, int(material_area * parameters.thin_max_area_ratio))
    retained = [
        (component, component_mask)
        for component, component_mask in _component_measurements(
            binary,
            response,
            parameters.thin_min_area,
            max_area,
        )
        if component.elongation >= parameters.thin_min_elongation
        and component.thickness <= parameters.thin_max_thickness
    ]
    return _make_result("thin_line", response, material_mask, retained)


def detect_pits_spots(image: np.ndarray, parameters: OperatorParameters) -> OperatorResult:
    """Detect compact bright or dark surface evidence."""
    _validate_bgr_image(image)
    normalized = _normalized_gray(image, parameters)
    response = np.zeros(normalized.shape, dtype=np.float32)
    for requested_size in parameters.spot_kernel_sizes:
        size = max(3, int(requested_size))
        if size % 2 == 0:
            size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        black_hat = cv2.morphologyEx(normalized, cv2.MORPH_BLACKHAT, kernel)
        top_hat = cv2.morphologyEx(normalized, cv2.MORPH_TOPHAT, kernel)
        scale_response = np.maximum(black_hat, top_hat).astype(np.float32)
        np.maximum(response, scale_response, out=response)
    material_mask = build_material_mask(image, parameters)
    binary = _robust_binary(response, material_mask, parameters)

    material_area = max(1, int(np.count_nonzero(material_mask)))
    max_area = max(parameters.spot_min_area, int(material_area * parameters.spot_max_area_ratio))
    retained = [
        (component, component_mask)
        for component, component_mask in _component_measurements(
            binary,
            response,
            parameters.spot_min_area,
            max_area,
        )
        if component.circularity >= parameters.spot_min_circularity
        and component.elongation <= parameters.spot_max_elongation
    ]
    return _make_result("pit_spot", response, material_mask, retained)


def render_overlay(image: np.ndarray, result: OperatorResult) -> np.ndarray:
    """Render diagnostic evidence without mutating the source image."""
    _validate_bgr_image(image)
    if result.mask.shape != image.shape[:2]:
        msg = "Operator result mask shape must match the BGR image"
        raise ValueError(msg)
    colors: dict[OperatorName, tuple[int, int, int]] = {
        "thin_line": (0, 0, 255),
        "pit_spot": (0, 165, 255),
    }
    overlay = image.copy()
    color = colors[result.operator]
    overlay[result.mask > 0] = color
    for component in result.components:
        cv2.rectangle(
            overlay,
            (component.x, component.y),
            (component.x + component.width - 1, component.y + component.height - 1),
            color,
            1,
        )
    return overlay
