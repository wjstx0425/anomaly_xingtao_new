"""Resident model adapters for the BMW eight-view laboratory Demo."""

from __future__ import annotations

import json
import math
import tempfile
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Any

import cv2
import numpy as np

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER
from bmw_inspection.lab.efficientad_analysis import fixed_scale_heatmap
from bmw_inspection.lab.efficientad_component_filter import (
    ComponentFilterPolicy,
    ComponentStatistics,
    score_anomaly_components,
)
from bmw_inspection.lab.eight_view_demo import (
    BranchStatus,
    DemoBranch,
    DemoBranchResult,
    EightViewInspection,
    EightViewDemoConfig,
    fuse_demo_status,
)
from bmw_inspection.lab.trusted_ok_reference import (
    TrustedOkMatch,
    TrustedOkMatcher,
)
from bmw_inspection.lab.template_region_weighting import (
    TemplateWeightedRegion,
    build_aligned_template_weights,
    load_template_weighted_regions,
    weighted_ccoeff,
)


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """Small model-independent output consumed by the Demo runtime."""

    status: BranchStatus
    score: float | None
    threshold: float | None
    reason: str
    overlay: np.ndarray | None
    raw_pred_label: bool | None = None
    details: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not isinstance(self.status, BranchStatus):
            raise TypeError("status must be BranchStatus")
        for name in ("score", "threshold"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite or None")
        if not self.reason.strip():
            raise ValueError("reason must not be empty")
        if self.raw_pred_label is not None and not isinstance(self.raw_pred_label, bool):
            raise TypeError("raw_pred_label must be bool or None")
        if not isinstance(self.details, Mapping):
            raise TypeError("details must be a mapping")


ViewPredictor = Callable[[str, np.ndarray], ModelOutput]
BrightPredictor = Callable[[np.ndarray], ModelOutput]


def _round_views(round_name: str) -> tuple[str, ...]:
    if round_name == "front":
        return VIEW_ORDER[:4]
    if round_name == "back":
        return VIEW_ORDER[4:]
    raise ValueError("round_name must be 'front' or 'back'")


def _round_result_order(round_name: str) -> tuple[tuple[DemoBranch, str], ...]:
    views = _round_views(round_name)
    bright_streak = (
        ((DemoBranch.BRIGHT_STREAK, "front_left"),)
        if round_name == "front"
        else ()
    )
    return (
        *((DemoBranch.TEMPLATE, view) for view in views),
        *bright_streak,
        *((DemoBranch.YOLO, view) for view in views),
        *((DemoBranch.EFFICIENTAD, view) for view in views),
    )


@dataclass(frozen=True, slots=True)
class RoundInspectionResult:
    """Intermediate result for one four-view capture round."""

    round_name: str
    roi_images: Mapping[str, np.ndarray]
    results: tuple[DemoBranchResult, ...]
    elapsed_ms: float

    def __post_init__(self) -> None:
        views = _round_views(self.round_name)
        if tuple(self.roi_images) != views:
            raise ValueError(f"{self.round_name} roi_images must use the round view order")
        actual_order = tuple((row.branch, row.view_id) for row in self.results)
        if actual_order != _round_result_order(self.round_name):
            raise ValueError(f"{self.round_name} results are incomplete or out of order")
        if not math.isfinite(float(self.elapsed_ms)) or self.elapsed_ms < 0:
            raise ValueError("elapsed_ms must be finite and non-negative")
        object.__setattr__(self, "roi_images", MappingProxyType(dict(self.roi_images)))


@dataclass(frozen=True, slots=True)
class _TemplateModel:
    input_width: int
    input_height: int
    target_width: int
    target_height: int
    max_shift: int
    model_threshold: float
    threshold: float
    templates: tuple[np.ndarray, ...]


def load_part_rois(path: Path) -> Mapping[str, tuple[int, int, int, int]]:
    """Load the eight ROI rectangles needed by the Demo, without dataset binding."""
    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取ROI配置：{resolved}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("ROI配置必须是JSON对象")
    width, height = payload.get("image_width"), payload.get("image_height")
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
        or isinstance(height, bool)
        or not isinstance(height, int)
        or height <= 0
    ):
        raise ValueError("ROI配置必须包含正整数image_width和image_height")
    raw_rois = payload.get("part_rois")
    if not isinstance(raw_rois, dict) or set(raw_rois) != set(VIEW_ORDER):
        raise ValueError("ROI配置必须覆盖八个标准视角")
    rois: dict[str, tuple[int, int, int, int]] = {}
    for view in VIEW_ORDER:
        raw = raw_rois[view]
        if (
            not isinstance(raw, list)
            or len(raw) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) for value in raw)
        ):
            raise ValueError(f"ROI {view}必须是四个整数")
        x1, y1, x2, y2 = raw
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"ROI {view}超出配置图像范围")
        rois[view] = (x1, y1, x2, y2)
    return MappingProxyType(rois)


def _load_demo_ignore_masks(
    index_path: Path,
    expected_shapes: Mapping[str, tuple[int, int]],
) -> Mapping[str, np.ndarray]:
    """Load eight readable binary masks; publishing metadata is intentionally ignored."""
    path = Path(index_path).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取ignore mask索引：{path}: {error}") from error
    records = payload.get("views") if isinstance(payload, dict) else None
    if not isinstance(records, dict) or set(records) != set(VIEW_ORDER):
        raise ValueError("ignore mask索引必须覆盖八个标准视角")
    masks: dict[str, np.ndarray] = {}
    for view in VIEW_ORDER:
        record = records[view]
        relative = record.get("mask_path") if isinstance(record, dict) else None
        if not isinstance(relative, str) or not relative.strip():
            raise ValueError(f"ignore mask路径无效：{view}")
        mask_path = (path.parent / relative).resolve()
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"ignore mask无法读取：{mask_path}")
        if mask.dtype != np.uint8:
            raise ValueError(f"ignore mask格式无效：{view}")
        if not set(np.unique(mask).tolist()).issubset({0, 255}):
            raise ValueError(f"ignore mask不是二值图：{view}")
        if tuple(mask.shape) != expected_shapes[view]:
            target_height, target_width = expected_shapes[view]
            mask = cv2.resize(
                mask,
                (target_width, target_height),
                interpolation=cv2.INTER_NEAREST,
            )
            if not set(np.unique(mask).tolist()).issubset({0, 255}):
                raise ValueError(f"ignore mask缩放后不是二值图：{view}")
        mask.flags.writeable = False
        masks[view] = mask
    return MappingProxyType(masks)


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError("Template输入必须是灰度、BGR或BGRA图像")


def _prepare_template_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    gray = _gray(image)
    target_width, target_height = size
    scale = min(target_width / gray.shape[1], target_height / gray.shape[0])
    width = max(1, min(target_width, int(round(gray.shape[1] * scale))))
    height = max(1, min(target_height, int(round(gray.shape[0] * scale))))
    resized = cv2.resize(
        gray,
        (width, height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    left = (target_width - width) // 2
    top = (target_height - height) // 2
    fitted = cv2.copyMakeBorder(
        resized,
        top,
        target_height - height - top,
        left,
        target_width - width - left,
        cv2.BORDER_REFLECT_101,
    )
    return cv2.GaussianBlur(fitted, (3, 3), 0)


class EightViewTemplatePredictor:
    """Load and score all eight independently trained Template models."""

    def __init__(
        self,
        model_paths: Mapping[str, Path],
        *,
        thresholds: Mapping[str, float],
        ignore_masks: Mapping[str, np.ndarray] | None = None,
        weighted_regions: Mapping[str, tuple[TemplateWeightedRegion, ...]] | None = None,
        weighted_region_weight: float = 1.0,
        weighted_outside_weight: float = 1.0,
        weighted_thresholds: Mapping[str, float] | None = None,
    ) -> None:
        if tuple(model_paths) != VIEW_ORDER:
            raise ValueError("Template模型必须按标准顺序覆盖八个视角")
        if tuple(thresholds) != VIEW_ORDER:
            raise ValueError("Template阈值必须按标准顺序覆盖八个视角")
        self._models = MappingProxyType(
            {
                view: self._load(
                    view,
                    Path(model_paths[view]).expanduser().resolve(),
                    float(thresholds[view]),
                )
                for view in VIEW_ORDER
            }
        )
        if weighted_regions is None:
            self._weighted_regions = None
            self._weighted_region_weight = 1.0
            self._weighted_outside_weight = 1.0
            self._weighted_thresholds = None
        else:
            if tuple(weighted_regions) != VIEW_ORDER or weighted_thresholds is None:
                raise ValueError("Template关键区域和加权阈值必须按标准顺序覆盖八个视角")
            if tuple(weighted_thresholds) != VIEW_ORDER:
                raise ValueError("Template加权阈值必须按标准顺序覆盖八个视角")
            if not math.isfinite(float(weighted_region_weight)) or weighted_region_weight < 1.0:
                raise ValueError("Template关键区域权重必须是有限且不小于1的数值")
            if not math.isfinite(float(weighted_outside_weight)) or weighted_outside_weight <= 0.0:
                raise ValueError("Template关键区域外权重必须是有限正数")
            parsed_thresholds = {view: float(weighted_thresholds[view]) for view in VIEW_ORDER}
            if any(not math.isfinite(value) or value < 0 for value in parsed_thresholds.values()):
                raise ValueError("Template加权阈值必须是有限非负数值")
            self._weighted_regions = MappingProxyType(
                {view: tuple(weighted_regions[view]) for view in VIEW_ORDER}
            )
            self._weighted_region_weight = float(weighted_region_weight)
            self._weighted_outside_weight = float(weighted_outside_weight)
            self._weighted_thresholds = MappingProxyType(parsed_thresholds)
        if ignore_masks is None:
            self._ignore_masks: Mapping[str, np.ndarray] | None = None
        else:
            if tuple(ignore_masks) != VIEW_ORDER:
                raise ValueError("Template手动忽略区必须按标准顺序覆盖八个视角")
            owned_masks: dict[str, np.ndarray] = {}
            for view in VIEW_ORDER:
                mask = np.asarray(ignore_masks[view])
                model = self._models[view]
                expected_shape = (model.input_height, model.input_width)
                if mask.dtype != np.uint8 or mask.ndim != 2 or mask.shape != expected_shape:
                    raise ValueError(f"{view} Template ignore mask必须是{expected_shape}的uint8二维图")
                if not set(np.unique(mask).tolist()).issubset({0, 255}):
                    raise ValueError(f"{view} Template ignore mask只能包含0和255")
                owned = mask.copy()
                owned.flags.writeable = False
                owned_masks[view] = owned
            self._ignore_masks = MappingProxyType(owned_masks)

    @staticmethod
    def _load(view: str, path: Path, threshold: float) -> _TemplateModel:
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError(f"{view} Template阈值必须是有限非负数值")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("view_id") != view:
            raise ValueError(f"Template模型视角不匹配：{path}")
        preprocess = payload.get("preprocess")
        items = payload.get("templates")
        if not isinstance(preprocess, dict) or not isinstance(items, list) or not items:
            raise ValueError(f"Template模型结构不正确：{path}")
        model_threshold = payload.get("threshold")
        if (
            isinstance(model_threshold, bool)
            or not isinstance(model_threshold, Real)
            or not math.isfinite(float(model_threshold))
            or float(model_threshold) < 0
        ):
            raise ValueError(f"Template模型内阈值无效：{path}")
        target_width = int(preprocess["target_width"])
        target_height = int(preprocess["target_height"])
        templates: list[np.ndarray] = []
        for item in items:
            template_path = (path.parent / item["path"]).resolve()
            image = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (target_height, target_width):
                raise ValueError(f"Template图片不可用：{template_path}")
            templates.append(image)
        return _TemplateModel(
            input_width=int(payload["input_width"]),
            input_height=int(payload["input_height"]),
            target_width=target_width,
            target_height=target_height,
            max_shift=int(preprocess["max_shift"]),
            model_threshold=float(model_threshold),
            threshold=threshold,
            templates=tuple(templates),
        )

    def predict(self, view: str, image: np.ndarray) -> ModelOutput:
        """Return Template risk and a display-only difference heatmap."""
        model = self._models[view]
        if image.shape[:2] != (model.input_height, model.input_width):
            raise ValueError(
                f"{view} Template要求{model.input_width}x{model.input_height}，实际{image.shape[1]}x{image.shape[0]}"
            )
        query = _prepare_template_image(image, (model.target_width, model.target_height))
        padded = cv2.copyMakeBorder(
            query,
            model.max_shift,
            model.max_shift,
            model.max_shift,
            model.max_shift,
            cv2.BORDER_REFLECT_101,
        )
        matches: list[tuple[float, np.ndarray, tuple[int, int]]] = []
        for template in model.templates:
            response = cv2.matchTemplate(padded, template, cv2.TM_CCOEFF_NORMED)
            _minimum, maximum, _minimum_location, maximum_location = cv2.minMaxLoc(response)
            matches.append((float(maximum), template, maximum_location))
        raw_similarity, raw_best, raw_best_location = max(matches, key=lambda item: item[0])
        raw_risk = max(0.0, 1.0 - raw_similarity)
        similarity, best, best_location = raw_similarity, raw_best, raw_best_location
        risk = raw_risk
        threshold = model.threshold
        score_source = "unmasked_ccoeff_normed"
        aligned_inspect_mask: np.ndarray | None = None
        if self._ignore_masks is not None and np.any(self._ignore_masks[view]):
            from bmw_inspection.lab.template_ignore_mask import (
                prepare_template_inspect_mask,
                select_masked_template_match,
            )

            inspect_mask = prepare_template_inspect_mask(
                self._ignore_masks[view],
                (model.target_width, model.target_height),
            )
            padded_inspect_mask = cv2.copyMakeBorder(
                inspect_mask,
                model.max_shift,
                model.max_shift,
                model.max_shift,
                model.max_shift,
                cv2.BORDER_REFLECT_101,
            )
            minimum_valid_pixels = max(
                256,
                math.ceil(0.10 * model.target_width * model.target_height),
            )
            masked_match = select_masked_template_match(
                padded,
                model.templates,
                padded_inspect_mask,
                minimum_valid_pixels=minimum_valid_pixels,
            )
            similarity = masked_match.similarity
            best = masked_match.template
            best_location = masked_match.location
            aligned_inspect_mask = masked_match.aligned_inspect_mask
            risk = max(0.0, 1.0 - similarity)
            score_source = "manual_ignore_masked_ccoeff_normed"
        best_x, best_y = best_location
        aligned_query = padded[
            best_y : best_y + model.target_height,
            best_x : best_x + model.target_width,
        ]
        legacy_similarity = similarity
        legacy_risk = risk
        weighted_regions = (
            () if self._weighted_regions is None else self._weighted_regions[view]
        )
        aligned_weights: np.ndarray | None = None
        if weighted_regions:
            if self._weighted_thresholds is None:
                raise ValueError("Template加权阈值未配置")
            aligned_weights = build_aligned_template_weights(
                input_shape=(model.input_height, model.input_width),
                target_size=(model.target_width, model.target_height),
                max_shift=model.max_shift,
                best_location=best_location,
                regions=weighted_regions,
                region_weight=self._weighted_region_weight,
                ignore_mask=None if self._ignore_masks is None else self._ignore_masks[view],
                outside_weight=self._weighted_outside_weight,
            )
            similarity = weighted_ccoeff(aligned_query, best, aligned_weights)
            risk = max(0.0, 1.0 - similarity)
            threshold = self._weighted_thresholds[view]
            score_source = "weighted_region_ccoeff_normed"
        difference = cv2.absdiff(aligned_query, best)
        if aligned_inspect_mask is None:
            mean_absolute_difference = float(difference.mean())
            valid_target_pixel_count = int(difference.size)
        else:
            valid = aligned_inspect_mask.astype(bool)
            if not np.any(valid):
                raise ValueError(f"{view} Template ignore mask排除了全部对齐像素")
            mean_absolute_difference = float(difference[valid].mean())
            valid_target_pixel_count = int(np.count_nonzero(valid))
            difference = difference.copy()
            difference[~valid] = 0
        heatmap = cv2.applyColorMap(difference, cv2.COLORMAP_TURBO)
        base = cv2.cvtColor(aligned_query, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(base, 0.65, heatmap, 0.35, 0.0)
        if aligned_inspect_mask is not None:
            overlay[~aligned_inspect_mask.astype(bool)] = base[~aligned_inspect_mask.astype(bool)]
        if aligned_weights is not None:
            emphasized = np.logical_and(aligned_weights > 1.0, aligned_weights > 0.0).astype(np.uint8)
            contours, _hierarchy = cv2.findContours(emphasized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)
        passed = risk <= threshold
        masked = aligned_inspect_mask is not None
        reason_prefix = "手动忽略区外 Template 诊断热区" if masked else "Template 诊断热区"
        return ModelOutput(
            BranchStatus.PASS if passed else BranchStatus.NG,
            risk,
            threshold,
            (
                f"{reason_prefix}：模板风险 {risk:.4f}，部署阈值 {threshold:.4f}；"
                f"最佳平移 ({best_x - model.max_shift}, {best_y - model.max_shift})，"
                f"对齐后平均绝对差 {mean_absolute_difference:.3f}"
            ),
            overlay,
            details={
                "evidence_type": "诊断热区",
                "score_source": score_source,
                "similarity": similarity,
                "risk": risk,
                "model_threshold": model.model_threshold,
                "deployment_threshold": threshold,
                "threshold_exceedance": risk - threshold,
                "best_shift_x": best_x - model.max_shift,
                "best_shift_y": best_y - model.max_shift,
                "aligned_mean_absolute_difference": mean_absolute_difference,
                "raw_unmasked_similarity": raw_similarity,
                "raw_unmasked_risk": raw_risk,
                "raw_unmasked_best_shift_x": raw_best_location[0] - model.max_shift,
                "raw_unmasked_best_shift_y": raw_best_location[1] - model.max_shift,
                "masked_similarity": legacy_similarity if masked else None,
                "masked_risk": legacy_risk if masked else None,
                "ignored_input_pixel_count": (
                    int(np.count_nonzero(self._ignore_masks[view])) if masked and self._ignore_masks is not None else 0
                ),
                "valid_target_pixel_count": valid_target_pixel_count,
                "valid_target_pixel_fraction": valid_target_pixel_count / difference.size,
                "legacy_similarity": legacy_similarity if weighted_regions else None,
                "legacy_risk": legacy_risk if weighted_regions else None,
                "weighted_similarity": similarity if weighted_regions else None,
                "weighted_risk": risk if weighted_regions else None,
                "region_weight": self._weighted_region_weight if weighted_regions else None,
                "outside_weight": self._weighted_outside_weight if weighted_regions else None,
                "effective_region_ratio": (
                    self._weighted_region_weight / self._weighted_outside_weight
                    if weighted_regions
                    else None
                ),
                "weighted_region_count": len(weighted_regions),
                "weighted_regions_xyxy": [list(region.roi_xyxy) for region in weighted_regions],
            },
        )


class EightViewLabTrackedProfileBrightStreakPredictor:
    """Tracked-profile light-streak inference from directly editable parameters."""

    def __init__(
        self,
        geometry: Mapping[str, object],
        thresholds: Mapping[str, object],
        rotated_roi_path: Path,
    ) -> None:
        from bmw_inspection.lab.bright_streak_rotated_roi import load_rotated_bright_streak_roi
        from bmw_inspection.lab.bright_streak_tracked_profile import (
            TrackedProfileGeometry,
            TrackedProfileThresholds,
        )

        try:
            self._geometry = TrackedProfileGeometry(**dict(geometry))
            self._thresholds = TrackedProfileThresholds(**dict(thresholds))
        except (TypeError, ValueError) as error:
            raise ValueError(f"光痕geometry或thresholds无效：{error}") from error
        required_width = self._geometry.candidate_width + 2 * (
            self._geometry.background_gap + self._geometry.background_width
        )
        if required_width > 81:
            raise ValueError("光痕geometry超出81像素旋转ROI宽度")
        self._rotated_roi = load_rotated_bright_streak_roi(rotated_roi_path)

    def predict(self, image: np.ndarray) -> ModelOutput:
        """Detect whether the rectified light streak is present and continuous."""
        from bmw_inspection.lab.bright_streak_rotated_roi import rectify_bright_streak_roi
        from bmw_inspection.lab.bright_streak_tracked_profile import (
            analyze_tracked_profile,
            classify_tracked_profile,
        )

        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim not in {2, 3}:
            raise ValueError("追踪光痕输入必须是uint8灰度或BGR图像")
        roi_gray = _gray(rectify_bright_streak_roi(image, self._rotated_roi))
        metrics = analyze_tracked_profile(roi_gray, self._geometry, self._thresholds)
        decision = classify_tracked_profile(metrics, self._thresholds)
        status = BranchStatus.PASS if decision == "OK" else BranchStatus.NG
        overlay = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
        for row, column in enumerate(metrics.path_x):
            if metrics.strong_mask[row]:
                color = (0, 255, 0)
            elif metrics.bridged_mask[row]:
                color = (0, 200, 255)
            elif (
                metrics.active_start_row is not None
                and metrics.active_stop_row is not None
                and metrics.active_start_row <= row < metrics.active_stop_row
            ):
                color = (0, 0, 255)
            else:
                color = (128, 128, 128)
            overlay[row, int(column)] = color
        present = bool(metrics.strong_mask.any())
        reason = (
            f"追踪光痕判定 {decision}；是否存在 {'是' if present else '否'}；"
            f"覆盖率 {metrics.coverage_ratio:.3f}（阈值 "
            f"{self._thresholds.min_presence_coverage_ratio:.3f}）；"
            f"最长连续段 {metrics.longest_run_ratio:.3f}（阈值 "
            f"{self._thresholds.min_longest_run_ratio:.3f}）；"
            f"最大断点 {metrics.max_gap_ratio:.3f}（上限 {self._thresholds.max_gap_ratio:.3f}）；"
            f"断点数 {metrics.gap_count}（上限 {self._thresholds.max_gap_count}）"
        )
        return ModelOutput(
            status,
            metrics.coverage_ratio,
            self._thresholds.min_presence_coverage_ratio,
            reason,
            overlay,
            details={
                "evidence_type": "追踪中心线诊断证据",
                "decision": decision,
                "roi_points_xy": self._rotated_roi.points_xy,
                "roi_mode": "manual_rotated_perspective",
                "presence": present,
                "coverage_ratio": metrics.coverage_ratio,
                "longest_run_px": metrics.longest_run_px,
                "longest_run_ratio": metrics.longest_run_ratio,
                "max_gap_px": metrics.max_gap_px,
                "max_gap_ratio": metrics.max_gap_ratio,
                "gap_count": metrics.gap_count,
                "strong_row_score": self._thresholds.strong_row_score,
                "weak_row_score": self._thresholds.weak_row_score,
                "bridged_rows": int(np.count_nonzero(metrics.bridged_mask)),
                "active_start_row": metrics.active_start_row,
                "active_stop_row": metrics.active_stop_row,
                "max_step": self._geometry.max_step,
                "tracked_centerline_x": tuple(int(value) for value in metrics.path_x),
            },
        )


def _default_yolo_factory(path: Path) -> Any:
    from ultralytics import YOLO

    return YOLO(str(path))


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class EightViewYoloPredictor:
    """One resident one-class YOLO model shared by all ROI crops."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        candidate_conf: float,
        final_threshold: float,
        imgsz: int,
        ignore_regions: Mapping[str, tuple[tuple[int, int, int, int], ...]] | None = None,
        model_factory: Callable[[Path], Any] = _default_yolo_factory,
    ) -> None:
        resolved = Path(checkpoint).expanduser().resolve()
        if not resolved.is_file():
            raise ValueError(f"YOLO模型不存在：{resolved}")
        if not 0 <= candidate_conf <= final_threshold <= 1:
            raise ValueError("YOLO阈值必须满足0 <= candidate_conf <= final_threshold <= 1")
        self._candidate_conf = float(candidate_conf)
        self._final_threshold = float(final_threshold)
        self._imgsz = int(imgsz)
        self._ignore_regions = {
            view: tuple(tuple(int(value) for value in region) for region in regions)
            for view, regions in (ignore_regions or {}).items()
        }
        self._model = model_factory(resolved)
        names = getattr(self._model, "names", None)
        parsed = {index: name for index, name in enumerate(names)} if isinstance(names, list) else dict(names or {})
        if parsed != {0: "defect"}:
            raise ValueError("YOLO模型必须只包含类别defect")
        self._lock = threading.Lock()

    def predict(self, view: str, image: np.ndarray) -> ModelOutput:
        with self._lock:
            predictions = self._model.predict(
                source=image,
                conf=self._candidate_conf,
                imgsz=self._imgsz,
                verbose=False,
            )
        if len(predictions) != 1:
            raise ValueError("YOLO必须返回一张图的结果")
        boxes = getattr(predictions[0], "boxes", None)
        if boxes is None:
            coordinates = np.empty((0, 4), dtype=np.float32)
            confidences = np.empty((0,), dtype=np.float32)
            classes = np.empty((0,), dtype=np.float32)
        else:
            coordinates = _array(boxes.xyxy).reshape((-1, 4))
            confidences = _array(boxes.conf).reshape(-1)
            classes = _array(boxes.cls).reshape(-1)
        if not (len(coordinates) == len(confidences) == len(classes)):
            raise ValueError("YOLO检测框字段长度不一致")
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
        view_ignore_regions = self._ignore_regions.get(view, ())
        for region_x1, region_y1, region_x2, region_y2 in view_ignore_regions:
            cv2.rectangle(
                overlay,
                (region_x1, region_y1),
                (region_x2, region_y2),
                (128, 128, 128),
                2,
            )
        final_count = 0
        ignored_count = 0
        eligible_confidences: list[float] = []
        box_details: list[dict[str, Any]] = []
        for xyxy, confidence, class_id in zip(coordinates, confidences, classes, strict=True):
            if int(class_id) != 0:
                raise ValueError("YOLO输出了非defect类别")
            center_x = (float(xyxy[0]) + float(xyxy[2])) / 2.0
            center_y = (float(xyxy[1]) + float(xyxy[3])) / 2.0
            ignored = any(
                region_x1 <= center_x <= region_x2 and region_y1 <= center_y <= region_y2
                for region_x1, region_y1, region_x2, region_y2 in view_ignore_regions
            )
            ignored_count += int(ignored)
            if not ignored:
                eligible_confidences.append(float(confidence))
            final = not ignored and float(confidence) >= self._final_threshold
            final_count += int(final)
            color = (160, 160, 160) if ignored else ((0, 0, 255) if final else (0, 170, 255))
            x1, y1, x2, y2 = (int(round(float(value))) for value in xyxy)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3 if final else 2)
            cv2.putText(
                overlay,
                (
                    f"ignored fixture {float(confidence):.2f}"
                    if ignored
                    else f"defect {float(confidence):.2f}"
                ),
                (max(0, x1), max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
            box_details.append(
                {
                    "xyxy": (x1, y1, x2, y2),
                    "confidence": float(confidence),
                    "class_name": "defect",
                    "is_final": final,
                    "is_ignored": ignored,
                    "ignore_reason": "fixed_yolo_ignore_region" if ignored else None,
                }
            )
        score = max(eligible_confidences, default=0.0)
        return ModelOutput(
            BranchStatus.NG if final_count else BranchStatus.PASS,
            score,
            self._final_threshold,
            (
                f"YOLO 真实检测框：最终缺陷框 {final_count} 个，候选框 {len(confidences)} 个，"
                f"固定区域忽略框 {ignored_count} 个；"
                f"最高置信度 {score:.4f}，部署阈值 {self._final_threshold:.4f}，"
                f"候选阈值 {self._candidate_conf:.4f}"
            ),
            overlay,
            details={
                "evidence_type": "真实检测框",
                "candidate_threshold": self._candidate_conf,
                "deployment_threshold": self._final_threshold,
                "threshold_exceedance": score - self._final_threshold,
                "candidate_box_count": len(confidences),
                "eligible_candidate_box_count": len(eligible_confidences),
                "ignored_box_count": ignored_count,
                "final_box_count": final_count,
                "boxes": box_details,
            },
        )


EfficientPrediction = tuple[float, bool, np.ndarray]
EfficientPredictorFactory = Callable[[Path], Callable[[np.ndarray], EfficientPrediction]]


class _AnomalibEfficientPredictor:
    """One resident EfficientAD checkpoint with Anomalib preprocessing."""

    def __init__(self, checkpoint: Path) -> None:
        from anomalib.engine import Engine
        from anomalib.models import EfficientAd

        self._model = EfficientAd.load_from_checkpoint(
            checkpoint,
            map_location="cpu",
            weights_only=False,
            visualizer=False,
        )
        self._engine = Engine(logger=False)

    def __call__(self, image: np.ndarray) -> EfficientPrediction:
        with tempfile.TemporaryDirectory(prefix="bmw-efficientad-demo-") as directory:
            path = Path(directory) / "roi.png"
            if not cv2.imwrite(str(path), image):
                raise RuntimeError("无法编码EfficientAD临时ROI")
            predictions = self._engine.predict(
                model=self._model,
                data_path=path,
                ckpt_path=None,
                return_predictions=True,
            )
        items = [] if predictions is None else [item for batch in predictions for item in batch]
        if len(items) != 1:
            raise RuntimeError("EfficientAD必须返回一个预测")
        item = items[0]
        score_values = _array(item.pred_score).reshape(-1)
        label_values = _array(item.pred_label).reshape(-1)
        anomaly_map = _array(item.anomaly_map)
        while anomaly_map.ndim > 2 and anomaly_map.shape[0] == 1:
            anomaly_map = anomaly_map[0]
        if score_values.size != 1 or label_values.size != 1 or anomaly_map.ndim != 2:
            raise RuntimeError("EfficientAD预测字段形状不正确")
        return float(score_values[0]), bool(label_values[0]), anomaly_map.astype(np.float32, copy=False)


class EightViewEfficientAdPredictor:
    """Keep eight EfficientAD-S predictors resident and expose comparable normalized scores."""

    def __init__(
        self,
        checkpoints: Mapping[str, Path],
        *,
        thresholds: Mapping[str, float],
        base_thresholds: Mapping[str, float] | None = None,
        threshold_margin: float = 0.0,
        predictor_factory: EfficientPredictorFactory = _AnomalibEfficientPredictor,
        ignore_masks: Mapping[str, np.ndarray] | None = None,
        component_policies: Mapping[str, ComponentFilterPolicy] | None = None,
        threshold_source: str | None = None,
        validation_status: str | None = None,
    ) -> None:
        if tuple(checkpoints) != VIEW_ORDER:
            raise ValueError("EfficientAD模型必须按标准顺序覆盖八个视角")
        if tuple(thresholds) != VIEW_ORDER:
            raise ValueError("EfficientAD阈值必须按标准顺序覆盖八个视角")
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in thresholds.values()):
            raise ValueError("EfficientAD阈值必须是有限数值")
        from bmw_inspection.lab.efficientad_thresholds import _validated_thresholds

        try:
            _views, threshold_values = _validated_thresholds(thresholds)
        except ValueError as error:
            raise ValueError("EfficientAD阈值必须是有限数值") from error
        self._thresholds = MappingProxyType(threshold_values)
        base_input = thresholds if base_thresholds is None else base_thresholds
        if tuple(base_input) != VIEW_ORDER:
            raise ValueError("EfficientAD基础阈值必须按标准顺序覆盖八个视角")
        try:
            _base_views, base_values = _validated_thresholds(base_input)
        except ValueError as error:
            raise ValueError("EfficientAD基础阈值必须是有限数值") from error
        if isinstance(threshold_margin, bool) or not isinstance(threshold_margin, Real):
            raise ValueError("EfficientAD阈值余量必须是有限非负数值")
        margin = float(threshold_margin)
        if not math.isfinite(margin) or margin < 0:
            raise ValueError("EfficientAD阈值余量必须是有限非负数值")
        if any(
            not math.isclose(threshold_values[view], base_values[view] + margin, abs_tol=1e-12)
            for view in VIEW_ORDER
        ):
            raise ValueError("EfficientAD基础阈值加余量必须等于部署阈值")
        self._base_thresholds = MappingProxyType(base_values)
        self._threshold_margin = margin
        for name, value in (
            ("threshold_source", threshold_source),
            ("validation_status", validation_status),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"EfficientAD {name} must be a non-empty string")
        self._threshold_source = threshold_source
        self._validation_status = validation_status
        if component_policies is None:
            self._component_policies: Mapping[str, ComponentFilterPolicy] | None = None
        else:
            if tuple(component_policies) != VIEW_ORDER or any(
                not isinstance(policy, ComponentFilterPolicy) for policy in component_policies.values()
            ):
                raise ValueError("EfficientAD component policies must use the canonical eight-view order")
            self._component_policies = MappingProxyType(dict(component_policies))
        if ignore_masks is None:
            self._ignore_masks: Mapping[str, np.ndarray] | None = None
        else:
            if tuple(ignore_masks) != VIEW_ORDER:
                raise ValueError("EfficientAD ignore masks must use the canonical eight-view order")
            owned_masks: dict[str, np.ndarray] = {}
            for view, mask in ignore_masks.items():
                array = np.asarray(mask)
                if array.dtype != np.uint8 or array.ndim != 2 or array.size == 0:
                    raise ValueError(f"EfficientAD ignore mask is invalid: {view}")
                if not set(np.unique(array).tolist()).issubset({0, 255}):
                    raise ValueError(f"EfficientAD ignore mask is not binary: {view}")
                copied = array.copy()
                copied.flags.writeable = False
                owned_masks[view] = copied
            self._ignore_masks = MappingProxyType(owned_masks)
        predictors: dict[str, Callable[[np.ndarray], EfficientPrediction]] = {}
        for view in VIEW_ORDER:
            checkpoint = Path(checkpoints[view]).expanduser().resolve()
            if not checkpoint.is_file():
                raise ValueError(f"EfficientAD模型不存在：{checkpoint}")
            predictors[view] = predictor_factory(checkpoint)
        self._predictors = MappingProxyType(predictors)
        self._lock = threading.Lock()

    def predict(self, view: str, image: np.ndarray) -> ModelOutput:
        with self._lock:
            score, raw_pred_label, anomaly_map = self._predictors[view](image)
        if not math.isfinite(score) or not np.isfinite(anomaly_map).all():
            raise ValueError("EfficientAD输出包含非有限数值")
        raw_pred_score = score
        scoring_map = anomaly_map
        score_source = "pred_score"
        ignored_roi_pixel_count = 0
        ignored_map_pixel_count = 0
        raw_map_max = float(np.max(anomaly_map))
        view_ignore_mask = None if self._ignore_masks is None else self._ignore_masks[view]
        if view_ignore_mask is not None and np.any(view_ignore_mask):
            from bmw_inspection.lab.efficientad_ignore_mask import mask_anomaly_map

            masked = mask_anomaly_map(anomaly_map, view_ignore_mask)
            score = masked.score
            scoring_map = masked.masked_map
            score_source = "manual_ignore_masked_anomaly_map_max"
            ignored_roi_pixel_count = int(np.count_nonzero(view_ignore_mask))
            ignored_map_pixel_count = masked.ignored_map_pixel_count
            raw_map_max = masked.raw_max

        component_result = None
        if self._component_policies is not None:
            component_result = score_anomaly_components(
                anomaly_map,
                view_ignore_mask,
                self._component_policies[view],
            )
            score = component_result.score
            score_source = "accepted_component_max_p95"
            ignored_map_pixel_count = component_result.ignored_pixel_count
            if view_ignore_mask is not None:
                ignored_roi_pixel_count = int(np.count_nonzero(view_ignore_mask))

        threshold = self._thresholds[view]
        base_threshold = self._base_thresholds[view]
        status = BranchStatus.NG if score >= threshold else BranchStatus.PASS
        heatmap = fixed_scale_heatmap(scoring_map)
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
        heatmap = cv2.resize(heatmap, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
        overlay = cv2.addWeighted(base, 0.6, heatmap, 0.4, 0.0)

        def put_label(text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
            cv2.putText(overlay, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(overlay, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        if component_result is None:
            hotspot_y, hotspot_x = np.unravel_index(int(np.argmax(scoring_map)), scoring_map.shape)
            hotspot: tuple[int, int] | None = (int(hotspot_x), int(hotspot_y))
        else:
            hotspot = component_result.hotspot
            accepted_display_mask = cv2.resize(
                component_result.accepted_mask,
                (base.shape[1], base.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            contours, _hierarchy = cv2.findContours(
                accepted_display_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            accepted_color = (0, 0, 255) if status is BranchStatus.NG else (0, 255, 0)
            cv2.drawContours(overlay, contours, -1, accepted_color, 2)
            for component, color, annotate in (
                *((item, accepted_color, True) for item in component_result.accepted_components),
                *((item, (0, 165, 255), False) for item in component_result.rejected_components),
            ):
                x1, y1, x2, y2 = component.bounding_box_xyxy
                display_box = (
                    int(round(x1 * base.shape[1] / anomaly_map.shape[1])),
                    int(round(y1 * base.shape[0] / anomaly_map.shape[0])),
                    max(0, int(round(x2 * base.shape[1] / anomaly_map.shape[1])) - 1),
                    max(0, int(round(y2 * base.shape[0] / anomaly_map.shape[0])) - 1),
                )
                cv2.rectangle(overlay, display_box[:2], display_box[2:], color, 2)
                if annotate:
                    label_y = max(18, display_box[1] - 5)
                    put_label(
                        f"P95={component.p95:.3f} area={component.area} reason={component.acceptance_reason}",
                        (display_box[0], label_y),
                        accepted_color,
                    )

            put_label(
                f"{status.value} P95={score:.3f} threshold={threshold:.3f} "
                f"exceedance={score - threshold:+.3f}",
                (8, max(18, overlay.shape[0] - 10)),
                (0, 0, 255) if status is BranchStatus.NG else (0, 255, 0),
            )

        if hotspot is not None:
            hotspot_x, hotspot_y = hotspot
            display_x = int(round(hotspot_x * max(0, base.shape[1] - 1) / max(1, anomaly_map.shape[1] - 1)))
            display_y = int(round(hotspot_y * max(0, base.shape[0] - 1) / max(1, anomaly_map.shape[0] - 1)))
            cv2.drawMarker(overlay, (display_x, display_y), (0, 0, 255), cv2.MARKER_CROSS, 13, 2)
            hotspot_value: float | None = float(scoring_map[hotspot_y, hotspot_x])
        else:
            hotspot_x = hotspot_y = None
            hotspot_value = None

        def component_details(component: ComponentStatistics) -> dict[str, object]:
            return {
                "area": component.area,
                "peak": component.peak,
                "mean": component.mean,
                "p95": component.p95,
                "bounding_box_xyxy": component.bounding_box_xyxy,
                "acceptance_reason": component.acceptance_reason,
                "contains_seed": component.contains_seed,
            }

        accepted_components = (
            tuple(component_details(item) for item in component_result.accepted_components)
            if component_result is not None
            else ()
        )
        rejected_components = (
            tuple(component_details(item) for item in component_result.rejected_components)
            if component_result is not None
            else ()
        )
        score_label = {
            "manual_ignore_masked_anomaly_map_max": "手动忽略区外异常图最大值",
            "accepted_component_max_p95": "有效连通域最大P95",
        }.get(score_source, "异常分数")
        details = {
            "evidence_type": "诊断热区",
            "score": score,
            "raw_pred_score": raw_pred_score,
            "raw_anomaly_map_max": raw_map_max,
            "score_source": score_source,
            "base_threshold": base_threshold,
            "deployment_threshold": threshold,
            "threshold_margin": self._threshold_margin,
            "threshold_exceedance": score - threshold,
            "hotspot_x": hotspot_x,
            "hotspot_y": hotspot_y,
            "hotspot_value": hotspot_value,
            "ignored_roi_pixel_count": ignored_roi_pixel_count,
            "ignored_map_pixel_count": ignored_map_pixel_count,
            "raw_pred_label": raw_pred_label,
        }
        if self._threshold_source is not None:
            details["threshold_source"] = self._threshold_source
        if self._validation_status is not None:
            details["validation_status"] = self._validation_status
        if component_result is not None:
            details.update(
                {
                    "accepted_component_count": len(accepted_components),
                    "rejected_component_count": len(rejected_components),
                    "accepted_components": accepted_components,
                    "rejected_components": rejected_components,
                }
            )
        return ModelOutput(
            status,
            score,
            threshold,
            (
                f"EfficientAD{score_label} {score:.4f}，基础阈值 {base_threshold:.4f}，"
                f"部署阈值 {threshold:.4f}，余量 {self._threshold_margin:.4f}"
            ),
            overlay,
            raw_pred_label=raw_pred_label,
            details=details,
        )


class EightViewModelSuite:
    """Run every configured branch and retain evidence even after an earlier NG."""

    def __init__(
        self,
        *,
        rois: Mapping[str, tuple[int, int, int, int]],
        template_predictor: ViewPredictor,
        bright_streak_predictor: BrightPredictor,
        yolo_predictor: ViewPredictor,
        efficientad_predictor: ViewPredictor,
        trusted_ok_matcher: TrustedOkMatcher | None = None,
        trusted_ok_matcher_error: str | None = None,
    ) -> None:
        if tuple(rois) != VIEW_ORDER:
            raise ValueError("rois must use the canonical BMW eight-view order")
        parsed: dict[str, tuple[int, int, int, int]] = {}
        for view, roi in rois.items():
            if len(roi) != 4 or any(isinstance(value, bool) or not isinstance(value, int) for value in roi):
                raise ValueError(f"ROI for {view} must contain four integers")
            x1, y1, x2, y2 = roi
            if not (0 <= x1 < x2 and 0 <= y1 < y2):
                raise ValueError(f"ROI for {view} must have positive area")
            parsed[view] = roi
        self._rois = parsed
        self._template = template_predictor
        self._bright_streak = bright_streak_predictor
        self._yolo = yolo_predictor
        self._efficientad = efficientad_predictor
        self._trusted_ok_matcher = trusted_ok_matcher
        self._trusted_ok_matcher_error = trusted_ok_matcher_error

    def inspect(self, images: Mapping[str, np.ndarray], *, capture_id: str) -> EightViewInspection:
        """Run 8 Template + 1 light streak + 8 YOLO + 8 EfficientAD checks."""
        if tuple(images) != VIEW_ORDER:
            raise ValueError("images must use the canonical BMW eight-view order")
        front = self.inspect_round(
            {view: images[view] for view in _round_views("front")},
            "front",
        )
        back = self.inspect_round(
            {view: images[view] for view in _round_views("back")},
            "back",
        )
        return self.finalize_rounds(images, front, back, capture_id)

    def inspect_round(
        self,
        images: Mapping[str, np.ndarray],
        round_name: str,
    ) -> RoundInspectionResult:
        """Run the configured branches for one four-view capture round."""
        views = _round_views(round_name)
        if tuple(images) != views:
            raise ValueError(f"{round_name} images must use the round view order")
        started = perf_counter()
        crops = {view: self._crop(view, images[view]) for view in views}
        results: list[DemoBranchResult] = []
        for view in views:
            results.append(self._call(DemoBranch.TEMPLATE, view, self._template, view, crops[view]))
        if round_name == "front":
            results.append(
                self._call(
                    DemoBranch.BRIGHT_STREAK,
                    "front_left",
                    self._bright_streak,
                    images["front_left"],
                )
            )
        for view in views:
            results.append(self._call(DemoBranch.YOLO, view, self._yolo, view, crops[view]))
        for view in views:
            results.append(self._call(DemoBranch.EFFICIENTAD, view, self._efficientad, view, crops[view]))
        return RoundInspectionResult(
            round_name=round_name,
            roi_images=crops,
            results=tuple(results),
            elapsed_ms=(perf_counter() - started) * 1000.0,
        )

    def finalize_rounds(
        self,
        images: Mapping[str, np.ndarray],
        front: RoundInspectionResult,
        back: RoundInspectionResult,
        capture_id: str,
    ) -> EightViewInspection:
        """Merge two completed rounds and run final diagnostics once."""
        if tuple(images) != VIEW_ORDER:
            raise ValueError("images must use the canonical BMW eight-view order")
        if not isinstance(front, RoundInspectionResult) or front.round_name != "front":
            raise ValueError("front must be a front RoundInspectionResult")
        if not isinstance(back, RoundInspectionResult) or back.round_name != "back":
            raise ValueError("back must be a back RoundInspectionResult")
        started = perf_counter()
        all_results = (*front.results, *back.results)
        result_by_key = {(row.branch, row.view_id): row for row in all_results}
        if len(result_by_key) != len(all_results):
            raise ValueError("round results contain duplicate branch/view rows")
        final_order = (
            *((DemoBranch.TEMPLATE, view) for view in VIEW_ORDER),
            (DemoBranch.BRIGHT_STREAK, "front_left"),
            *((DemoBranch.YOLO, view) for view in VIEW_ORDER),
            *((DemoBranch.EFFICIENTAD, view) for view in VIEW_ORDER),
        )
        if set(result_by_key) != set(final_order):
            raise ValueError("round results are incomplete or duplicated")
        rows = tuple(result_by_key[key] for key in final_order)
        crops = {
            **front.roi_images,
            **back.roi_images,
        }
        final_status = fuse_demo_status(rows)
        actionable_comparisons = tuple(
            dict.fromkeys(
                (
                    row.view_id,
                    "full" if row.branch is DemoBranch.BRIGHT_STREAK else "roi",
                )
                for row in rows
                if row.status in {BranchStatus.NG, BranchStatus.ERROR}
            )
        )
        trusted_ok_by_comparison: dict[tuple[str, str], TrustedOkMatch] = {}
        match_errors: dict[str, str] = {}
        for view, comparison_mode in actionable_comparisons:
            comparison_id = f"{view}/{comparison_mode}"
            if self._trusted_ok_matcher is None:
                if self._trusted_ok_matcher_error is not None:
                    match_errors[comparison_id] = self._trusted_ok_matcher_error
                continue
            try:
                trusted_ok_by_comparison[(view, comparison_mode)] = self._trusted_ok_matcher.match(
                    view,
                    images[view],
                    crops[view],
                    comparison_mode=comparison_mode,
                )
            except Exception as error:
                match_errors[comparison_id] = str(error)
        diagnostic_metadata = (
            {"trusted_ok_match_errors": match_errors} if match_errors else {}
        )
        return EightViewInspection(
            capture_id=capture_id,
            images=images,
            results=rows,
            final_status=final_status,
            elapsed_ms=(
                front.elapsed_ms
                + back.elapsed_ms
                + (perf_counter() - started) * 1000.0
            ),
            roi_images=crops,
            trusted_ok_by_comparison=trusted_ok_by_comparison,
            diagnostic_metadata=diagnostic_metadata,
        )

    def _crop(self, view: str, image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim not in {2, 3}:
            raise ValueError(f"{view} must be a uint8 grayscale/BGR image")
        x1, y1, x2, y2 = self._rois[view]
        height, width = image.shape[:2]
        if x2 > width or y2 > height:
            raise ValueError(f"ROI for {view} lies outside {width}x{height}")
        return image[y1:y2, x1:x2].copy()

    @staticmethod
    def _call(branch: DemoBranch, view: str, predictor: Callable[..., ModelOutput], *args: object) -> DemoBranchResult:
        started = perf_counter()
        try:
            output = predictor(*args)
            if not isinstance(output, ModelOutput):
                raise TypeError("模型后端必须返回ModelOutput")
            return DemoBranchResult(
                branch=branch,
                view_id=view,
                status=output.status,
                score=output.score,
                threshold=output.threshold,
                elapsed_ms=(perf_counter() - started) * 1000.0,
                reason=output.reason,
                overlay=output.overlay,
                raw_pred_label=output.raw_pred_label,
                details=output.details,
            )
        except Exception as error:
            return DemoBranchResult(
                branch=branch,
                view_id=view,
                status=BranchStatus.ERROR,
                score=None,
                threshold=None,
                elapsed_ms=(perf_counter() - started) * 1000.0,
                reason=f"{branch.value}推理失败：{error}",
                overlay=None,
                details={"evidence_type": "不可用", "error": str(error)},
            )


def build_model_suite(
    config: EightViewDemoConfig,
    *,
    status_callback: Callable[[str], None] | None = None,
) -> EightViewModelSuite:
    """Build the four resident model branches from one resolved Demo profile."""
    if config.bright_streak_engine != "tracked_profile_v3_manual_rotated_roi":
        raise ValueError(f"不支持的实验室光痕引擎：{config.bright_streak_engine}")
    if config.bright_streak_rotated_roi is None:
        raise ValueError("光痕旋转ROI路径不能为空")
    bright_streak = EightViewLabTrackedProfileBrightStreakPredictor(
        config.bright_streak_geometry,
        config.bright_streak_thresholds,
        config.bright_streak_rotated_roi,
    )
    yolo = EightViewYoloPredictor(
        config.yolo_checkpoint,
        candidate_conf=config.yolo_candidate_conf,
        final_threshold=config.yolo_final_threshold,
        imgsz=config.yolo_imgsz,
        ignore_regions=getattr(config, "yolo_ignore_regions", None),
    )
    rois = load_part_rois(config.roi_config)
    expected_shapes = {
        view: (rois[view][3] - rois[view][1], rois[view][2] - rois[view][0])
        for view in VIEW_ORDER
    }
    mask_cache: dict[Path, Mapping[str, np.ndarray]] = {}

    def masks(path: Path | None) -> Mapping[str, np.ndarray] | None:
        if path is None:
            return None
        if path not in mask_cache:
            mask_cache[path] = _load_demo_ignore_masks(path, expected_shapes)
        return mask_cache[path]

    template_masks = masks(config.template_ignore_mask_index)
    efficientad_masks = masks(config.efficientad_ignore_mask_index)
    weighted_config = config.template_weighted_regions
    weighted_regions = None
    if weighted_config is not None and weighted_config.enabled:
        weighted_regions = load_template_weighted_regions(
            weighted_config.roi_config,
            expected_shapes=expected_shapes,
        )
    template = EightViewTemplatePredictor(
        config.template_models,
        thresholds=config.template_thresholds,
        ignore_masks=template_masks,
        weighted_regions=weighted_regions,
        weighted_region_weight=(1.0 if weighted_config is None else weighted_config.weight),
        weighted_outside_weight=(
            1.0 if weighted_config is None else weighted_config.outside_weight
        ),
        weighted_thresholds=(None if weighted_config is None else weighted_config.thresholds),
    )
    efficientad = EightViewEfficientAdPredictor(
        config.efficientad_checkpoints,
        thresholds=config.efficientad_thresholds,
        base_thresholds=config.efficientad_base_thresholds,
        threshold_margin=config.efficientad_threshold_margin,
        ignore_masks=efficientad_masks,
        component_policies=config.efficientad_component_policies,
        threshold_source=config.efficientad_threshold_source,
        validation_status=config.efficientad_validation_status,
    )
    trusted_ok_matcher: TrustedOkMatcher | None = None
    trusted_ok_matcher_error: str | None = None
    index_path = getattr(config, "trusted_ok_reference_index", None)
    if index_path is not None:
        announce = status_callback if status_callback is not None else (lambda _message: None)
        announce("正在读取并预热可信OK参考库……")
        try:
            candidate = TrustedOkMatcher(index_path)
            candidate.preload()
        except Exception as error:
            trusted_ok_matcher_error = f"可信OK参考不可用：{error}"
            announce(f"可信OK参考库不可用，已仅禁用参考诊断：{error}")
        else:
            trusted_ok_matcher = candidate
            announce("可信OK参考库预热完成。")
    return EightViewModelSuite(
        rois=rois,
        template_predictor=template.predict,
        bright_streak_predictor=bright_streak.predict,
        yolo_predictor=yolo.predict,
        efficientad_predictor=efficientad.predict,
        trusted_ok_matcher=trusted_ok_matcher,
        trusted_ok_matcher_error=trusted_ok_matcher_error,
    )


__all__ = [
    "EightViewEfficientAdPredictor",
    "EightViewLabTrackedProfileBrightStreakPredictor",
    "EightViewModelSuite",
    "EightViewTemplatePredictor",
    "EightViewYoloPredictor",
    "ModelOutput",
    "RoundInspectionResult",
    "build_model_suite",
    "load_part_rois",
]
