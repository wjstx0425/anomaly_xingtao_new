"""Critical-region weighting for the existing BMW whole-ROI Template score."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType

import cv2
import numpy as np

from bmw_inspection.views import VIEW_ORDER
from bmw_inspection.lab.template_ignore_mask import prepare_template_inspect_mask


@dataclass(frozen=True, slots=True)
class TemplateWeightedRegion:
    """One operator-selected rectangle in public-ROI-local half-open coordinates."""

    region_id: str
    roi_xyxy: tuple[int, int, int, int]


def load_template_weighted_regions(
    path: Path,
    *,
    expected_shapes: Mapping[str, tuple[int, int]],
) -> Mapping[str, tuple[TemplateWeightedRegion, ...]]:
    """Load hand-specific weighted rectangles without release or digest binding."""
    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Template关键区域JSON无法读取：{resolved}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("views"), dict):
        raise ValueError("Template关键区域JSON必须包含views对象")
    raw_views = payload["views"]
    unknown = set(raw_views) - set(VIEW_ORDER)
    if unknown:
        raise ValueError(f"Template关键区域包含未知视角：{sorted(unknown)}")
    if set(expected_shapes) != set(VIEW_ORDER):
        raise ValueError("Template关键区域期望尺寸必须覆盖八个视角")

    parsed: dict[str, tuple[TemplateWeightedRegion, ...]] = {}
    for view in VIEW_ORDER:
        raw_regions = raw_views.get(view, [])
        if not isinstance(raw_regions, list):
            raise ValueError(f"{view} Template关键区域必须是列表")
        height, width = expected_shapes[view]
        seen: set[str] = set()
        regions: list[TemplateWeightedRegion] = []
        for raw in raw_regions:
            if not isinstance(raw, dict):
                raise ValueError(f"{view} Template关键区域条目必须是对象")
            region_id = raw.get("id")
            roi = raw.get("roi_xyxy")
            if not isinstance(region_id, str) or not region_id.strip() or region_id in seen:
                raise ValueError(f"{view} Template关键区域id为空或重复")
            if (
                not isinstance(roi, list)
                or len(roi) != 4
                or any(isinstance(value, bool) or not isinstance(value, int) for value in roi)
            ):
                raise ValueError(f"{view} Template关键区域坐标必须是四个整数")
            x1, y1, x2, y2 = roi
            if x1 < 0 or y1 < 0 or x1 >= x2 or y1 >= y2 or x2 > width or y2 > height:
                raise ValueError(f"{view} Template关键区域越界：{roi}")
            seen.add(region_id)
            regions.append(TemplateWeightedRegion(region_id, (x1, y1, x2, y2)))
        parsed[view] = tuple(regions)
    return MappingProxyType(parsed)


def _fit_geometry(
    input_shape: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    input_height, input_width = input_shape
    target_width, target_height = target_size
    if min(input_height, input_width, target_width, target_height) <= 0:
        raise ValueError("Template加权图像尺寸必须为正")
    scale = min(target_width / input_width, target_height / input_height)
    fitted_width = max(1, min(target_width, int(round(input_width * scale))))
    fitted_height = max(1, min(target_height, int(round(input_height * scale))))
    left = (target_width - fitted_width) // 2
    top = (target_height - fitted_height) // 2
    return fitted_width, fitted_height, left, top


def build_aligned_template_weights(
    input_shape: tuple[int, int],
    target_size: tuple[int, int],
    max_shift: int,
    best_location: tuple[int, int],
    regions: Sequence[TemplateWeightedRegion],
    region_weight: float,
    ignore_mask: np.ndarray | None,
    outside_weight: float = 1.0,
) -> np.ndarray:
    """Build weights in the selected Template alignment coordinate system."""
    if not math.isfinite(float(region_weight)) or float(region_weight) < 1.0:
        raise ValueError("Template关键区域权重必须是有限且不小于1的数值")
    if not math.isfinite(float(outside_weight)) or float(outside_weight) <= 0.0:
        raise ValueError("Template关键区域外权重必须是有限正数")
    target_width, target_height = target_size
    fitted_width, fitted_height, left, top = _fit_geometry(input_shape, target_size)
    input_height, input_width = input_shape
    weights = np.full(
        (target_height, target_width),
        float(outside_weight),
        dtype=np.float64,
    )
    for region in regions:
        x1, y1, x2, y2 = region.roi_xyxy
        mapped_x1 = left + math.floor(x1 * fitted_width / input_width)
        mapped_y1 = top + math.floor(y1 * fitted_height / input_height)
        mapped_x2 = left + math.ceil(x2 * fitted_width / input_width)
        mapped_y2 = top + math.ceil(y2 * fitted_height / input_height)
        weights[mapped_y1:mapped_y2, mapped_x1:mapped_x2] = np.maximum(
            weights[mapped_y1:mapped_y2, mapped_x1:mapped_x2],
            float(region_weight),
        )

    padded_weights = cv2.copyMakeBorder(
        weights,
        max_shift,
        max_shift,
        max_shift,
        max_shift,
        cv2.BORDER_REFLECT_101,
    )
    best_x, best_y = best_location
    aligned = padded_weights[best_y : best_y + target_height, best_x : best_x + target_width].copy()
    if aligned.shape != (target_height, target_width):
        raise ValueError("Template关键区域权重与对齐位置不兼容")

    if ignore_mask is not None:
        inspect = prepare_template_inspect_mask(ignore_mask, target_size)
        padded_inspect = cv2.copyMakeBorder(
            inspect,
            max_shift,
            max_shift,
            max_shift,
            max_shift,
            cv2.BORDER_REFLECT_101,
        )
        aligned_inspect = padded_inspect[
            best_y : best_y + target_height,
            best_x : best_x + target_width,
        ]
        if aligned_inspect.shape != aligned.shape:
            raise ValueError("Template ignore mask与对齐位置不兼容")
        aligned[aligned_inspect == 0] = 0.0
    aligned.flags.writeable = False
    return aligned


def weighted_ccoeff(
    query: np.ndarray,
    template: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Return weighted CCOEFF_NORMED for one already-selected alignment."""
    q = np.asarray(query, dtype=np.float64)
    t = np.asarray(template, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if (
        q.ndim != 2
        or q.shape != t.shape
        or q.shape != w.shape
        or not np.all(np.isfinite(q))
        or not np.all(np.isfinite(t))
        or not np.all(np.isfinite(w))
        or np.any(w < 0)
    ):
        raise ValueError("Template加权输入形状或权重无效")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("Template有效权重为空")
    q_centered = q - float(np.sum(w * q) / total)
    t_centered = t - float(np.sum(w * t) / total)
    denominator = math.sqrt(
        float(np.sum(w * q_centered**2) * np.sum(w * t_centered**2))
    )
    if denominator <= 0:
        raise ValueError("Template加权相关系数分母为零")
    similarity = float(np.sum(w * q_centered * t_centered) / denominator)
    return float(np.clip(similarity, -1.0, 1.0))


def normal_envelope_threshold(
    scores: Sequence[float],
    margin_ratio: float = 0.10,
) -> float:
    """Fit a normal-only envelope with the requested relative lab margin."""
    values = [float(score) for score in scores]
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Template标定分数必须是非空有限非负序列")
    if not math.isfinite(float(margin_ratio)) or margin_ratio < 0:
        raise ValueError("Template标定余量必须是有限非负数值")
    maximum = max(values)
    if maximum == 0.0:
        return math.nextafter(0.0, math.inf)
    return maximum * (1.0 + float(margin_ratio))


__all__ = [
    "TemplateWeightedRegion",
    "build_aligned_template_weights",
    "load_template_weighted_regions",
    "normal_envelope_threshold",
    "weighted_ccoeff",
]
