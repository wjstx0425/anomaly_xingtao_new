"""Masked correlation helpers for the BMW eight-view Template branch."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class MaskedTemplateMatch:
    """Best finite masked Template match across templates and translations."""

    similarity: float
    template_index: int
    location: tuple[int, int]
    template: np.ndarray
    aligned_query: np.ndarray
    aligned_inspect_mask: np.ndarray


def prepare_template_inspect_mask(ignore_mask: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Apply Template aspect-fit geometry to a ``255=ignore`` original-ROI mask."""
    mask = np.asarray(ignore_mask)
    if mask.dtype != np.uint8 or mask.ndim != 2 or mask.size == 0:
        raise ValueError("Template ignore mask must be a non-empty uint8 2D array")
    if not set(np.unique(mask).tolist()).issubset({0, 255}):
        raise ValueError("Template ignore mask must contain only 0 and 255")
    if (
        len(target_size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in target_size)
    ):
        raise ValueError("Template target_size must contain positive integer width and height")
    target_width, target_height = target_size
    scale = min(target_width / mask.shape[1], target_height / mask.shape[0])
    width = max(1, min(target_width, int(round(mask.shape[1] * scale))))
    height = max(1, min(target_height, int(round(mask.shape[0] * scale))))
    inspect = cv2.resize((mask == 0).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    left = (target_width - width) // 2
    top = (target_height - height) // 2
    return cv2.copyMakeBorder(
        inspect,
        top,
        target_height - height - top,
        left,
        target_width - width - left,
        cv2.BORDER_REFLECT_101,
    )


def _correlate(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    return cv2.matchTemplate(
        np.ascontiguousarray(image, dtype=np.float32),
        np.ascontiguousarray(template, dtype=np.float32),
        cv2.TM_CCORR,
    )


def masked_ccoeff_normed_map(
    padded_query: np.ndarray,
    template: np.ndarray,
    padded_inspect_mask: np.ndarray,
    *,
    minimum_valid_pixels: int = 256,
) -> np.ndarray:
    """Compute position-specific masked CCOEFF_NORMED for a translation window."""
    query = np.asarray(padded_query)
    reference = np.asarray(template)
    inspect = np.asarray(padded_inspect_mask)
    if query.ndim != 2 or reference.ndim != 2 or inspect.ndim != 2:
        raise ValueError("masked Template arrays must be two-dimensional")
    if query.shape != inspect.shape or reference.shape[0] > query.shape[0] or reference.shape[1] > query.shape[1]:
        raise ValueError("masked Template array shapes are incompatible")
    if not np.isfinite(query).all() or not np.isfinite(reference).all():
        raise ValueError("masked Template image values must be finite")
    if not set(np.unique(inspect).tolist()).issubset({0, 1, False, True}):
        raise ValueError("Template inspect mask must contain only 0 and 1")
    if isinstance(minimum_valid_pixels, bool) or not isinstance(minimum_valid_pixels, int) or minimum_valid_pixels < 3:
        raise ValueError("minimum_valid_pixels must be an integer of at least 3")

    valid = inspect.astype(np.float32, copy=False)
    query_value = query.astype(np.float32, copy=False)
    template_value = reference.astype(np.float32, copy=False)
    query_value = query_value - float(np.mean(query_value, dtype=np.float64))
    template_value = template_value - float(np.mean(template_value, dtype=np.float64))
    ones = np.ones(reference.shape, dtype=np.float32)
    weighted_query = valid * query_value

    count = _correlate(valid, ones)
    sum_query = _correlate(weighted_query, ones)
    sum_query_squared = _correlate(valid * query_value * query_value, ones)
    sum_template = _correlate(valid, template_value)
    sum_template_squared = _correlate(valid, template_value * template_value)
    sum_product = _correlate(weighted_query, template_value)

    safe_count = np.maximum(count, 1.0)
    numerator = sum_product - sum_query * sum_template / safe_count
    query_energy = sum_query_squared - sum_query * sum_query / safe_count
    template_energy = sum_template_squared - sum_template * sum_template / safe_count
    denominator = np.sqrt(np.maximum(query_energy, 0.0) * np.maximum(template_energy, 0.0))
    valid_candidate = np.logical_and.reduce(
        (
            count >= minimum_valid_pixels,
            query_energy > 1e-8,
            template_energy > 1e-8,
            denominator > 1e-8,
        )
    )
    response = np.full(count.shape, np.nan, dtype=np.float32)
    np.divide(numerator, denominator, out=response, where=valid_candidate)
    np.clip(response, -1.0, 1.0, out=response)
    return response


def select_masked_template_match(
    padded_query: np.ndarray,
    templates: Sequence[np.ndarray],
    padded_inspect_mask: np.ndarray,
    *,
    minimum_valid_pixels: int = 256,
) -> MaskedTemplateMatch:
    """Return the highest finite masked correlation across all templates and shifts."""
    if not templates:
        raise ValueError("masked Template matching requires at least one template")
    candidates: list[tuple[float, int, tuple[int, int]]] = []
    for index, template in enumerate(templates):
        response = masked_ccoeff_normed_map(
            padded_query,
            template,
            padded_inspect_mask,
            minimum_valid_pixels=minimum_valid_pixels,
        )
        if np.isfinite(response).any():
            flat_index = int(np.nanargmax(response))
            y, x = np.unravel_index(flat_index, response.shape)
            candidates.append((float(response[y, x]), index, (int(x), int(y))))
    if not candidates:
        raise ValueError("no valid masked Template candidate")
    similarity, template_index, location = max(candidates, key=lambda item: item[0])
    template = np.asarray(templates[template_index])
    x, y = location
    aligned_query = np.asarray(padded_query)[y : y + template.shape[0], x : x + template.shape[1]].copy()
    aligned_inspect = np.asarray(padded_inspect_mask)[
        y : y + template.shape[0], x : x + template.shape[1]
    ].copy()
    template_copy = template.copy()
    for value in (template_copy, aligned_query, aligned_inspect):
        value.flags.writeable = False
    return MaskedTemplateMatch(
        similarity=similarity,
        template_index=template_index,
        location=location,
        template=template_copy,
        aligned_query=aligned_query,
        aligned_inspect_mask=aligned_inspect,
    )
