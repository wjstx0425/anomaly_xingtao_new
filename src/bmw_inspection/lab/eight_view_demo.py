"""Small contracts and configuration for the BMW eight-view laboratory Demo."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import cv2
import numpy as np

from bmw_inspection.views import VIEW_ORDER
from bmw_inspection.lab.efficientad_component_filter import ComponentFilterPolicy
from bmw_inspection.lab.trusted_ok_reference import TrustedOkMatch


class DemoBranch(str, Enum):
    """Evidence branches shown by the laboratory Demo."""

    TEMPLATE = "template"
    BRIGHT_STREAK = "bright_streak"
    YOLO = "yolo"
    EFFICIENTAD = "efficientad"


class BranchStatus(str, Enum):
    """One branch result for one view."""

    PASS = "PASS"
    NG = "NG"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


class DemoFinalStatus(str, Enum):
    """The result displayed after all configured checks finish."""

    OK = "OK"
    NG = "NG"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class DemoBranchResult:
    """Display-ready result from one model/view pair."""

    branch: DemoBranch
    view_id: str
    status: BranchStatus
    score: float | None
    threshold: float | None
    elapsed_ms: float
    reason: str
    overlay: np.ndarray | None
    raw_pred_label: bool | None = None
    details: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        if self.view_id not in VIEW_ORDER:
            raise ValueError(f"unknown BMW view: {self.view_id}")
        if not isinstance(self.branch, DemoBranch) or not isinstance(self.status, BranchStatus):
            raise TypeError("branch and status must use Demo enums")
        for name in ("score", "threshold"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite or None")
        if not math.isfinite(float(self.elapsed_ms)) or self.elapsed_ms < 0:
            raise ValueError("elapsed_ms must be finite and non-negative")
        if not self.reason.strip():
            raise ValueError("reason must not be empty")
        if self.raw_pred_label is not None and not isinstance(self.raw_pred_label, bool):
            raise TypeError("raw_pred_label must be bool or None")
        object.__setattr__(self, "details", _immutable_details(self.details))
        if self.overlay is not None:
            if not isinstance(self.overlay, np.ndarray) or self.overlay.size == 0:
                raise TypeError("overlay must be a non-empty numpy image or None")
            owned = self.overlay.copy()
            owned.flags.writeable = False
            object.__setattr__(self, "overlay", owned)


@dataclass(frozen=True, slots=True)
class EightViewInspection:
    """One complete, immutable eight-view laboratory result."""

    capture_id: str
    images: Mapping[str, np.ndarray]
    results: tuple[DemoBranchResult, ...]
    final_status: DemoFinalStatus
    elapsed_ms: float
    roi_images: Mapping[str, np.ndarray] = MappingProxyType({})
    trusted_ok_by_comparison: Mapping[tuple[str, str], TrustedOkMatch] = MappingProxyType({})
    diagnostic_metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not self.capture_id.strip():
            raise ValueError("capture_id must not be empty")
        object.__setattr__(self, "images", _owned_images(self.images))
        if not isinstance(self.results, tuple) or not all(isinstance(row, DemoBranchResult) for row in self.results):
            raise TypeError("results must contain DemoBranchResult values")
        if self.final_status is not fuse_demo_status(self.results):
            raise ValueError("final_status does not match branch results")
        if not math.isfinite(float(self.elapsed_ms)) or self.elapsed_ms < 0:
            raise ValueError("elapsed_ms must be finite and non-negative")
        if not isinstance(self.roi_images, Mapping):
            raise TypeError("roi_images must be a mapping")
        if self.roi_images:
            object.__setattr__(self, "roi_images", _owned_images(self.roi_images))
        else:
            object.__setattr__(self, "roi_images", MappingProxyType({}))
        if not isinstance(self.trusted_ok_by_comparison, Mapping):
            raise TypeError("trusted_ok_by_comparison must be a mapping")
        trusted: dict[tuple[str, str], TrustedOkMatch] = {}
        for key, match in self.trusted_ok_by_comparison.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or key[0] not in VIEW_ORDER
                or key[1] not in {"roi", "full"}
                or not isinstance(match, TrustedOkMatch)
                or (match.view_id, match.comparison_mode) != key
            ):
                raise ValueError("trusted_ok_by_comparison must use matching (view, mode) keys")
            trusted[key] = match
        object.__setattr__(self, "trusted_ok_by_comparison", MappingProxyType(trusted))
        object.__setattr__(self, "diagnostic_metadata", _immutable_details(self.diagnostic_metadata))

    def actionable_results(self) -> tuple[DemoBranchResult, ...]:
        """Return NG and ERROR evidence in the model execution order."""
        return tuple(
            row for row in self.results if row.status in {BranchStatus.NG, BranchStatus.ERROR}
        )


@dataclass(frozen=True, slots=True)
class TemplateWeightedRegionsConfig:
    """Optional critical-region weighting for the existing Template branch."""

    enabled: bool
    weight: float
    outside_weight: float
    roi_config: Path
    thresholds: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "thresholds", MappingProxyType(dict(self.thresholds)))


@dataclass(frozen=True, slots=True)
class EightViewDemoConfig:
    """Resolved paths and editable thresholds for one Demo profile."""

    path: Path
    demo_id: str
    capture_config: Path
    roi_config: Path
    prepared_manifest: Path
    result_root: Path
    template_models: Mapping[str, Path]
    template_thresholds: Mapping[str, float]
    efficientad_checkpoints: Mapping[str, Path]
    efficientad_thresholds: Mapping[str, float]
    efficientad_base_thresholds: Mapping[str, float]
    efficientad_threshold_margin: float
    bright_streak_engine: str
    bright_streak_geometry: Mapping[str, float | int]
    bright_streak_thresholds: Mapping[str, float | int]
    yolo_checkpoint: Path
    yolo_candidate_conf: float
    yolo_final_threshold: float
    yolo_imgsz: int
    efficientad_threshold_source: str | None = None
    efficientad_validation_status: str | None = None
    yolo_ignore_regions: Mapping[str, tuple[tuple[int, int, int, int], ...]] = MappingProxyType({})
    bright_streak_rotated_roi: Path | None = None
    efficientad_ignore_mask_index: Path | None = None
    efficientad_component_filter_config: Path | None = None
    efficientad_component_policies: Mapping[str, ComponentFilterPolicy] | None = None
    template_ignore_mask_index: Path | None = None
    template_weighted_regions: TemplateWeightedRegionsConfig | None = None
    trusted_ok_reference_index: Path | None = None
    views: tuple[str, ...] = VIEW_ORDER

    def __post_init__(self) -> None:
        object.__setattr__(self, "template_models", MappingProxyType(dict(self.template_models)))
        object.__setattr__(self, "template_thresholds", MappingProxyType(dict(self.template_thresholds)))
        object.__setattr__(self, "efficientad_checkpoints", MappingProxyType(dict(self.efficientad_checkpoints)))
        object.__setattr__(self, "efficientad_thresholds", MappingProxyType(dict(self.efficientad_thresholds)))
        object.__setattr__(self, "bright_streak_geometry", MappingProxyType(dict(self.bright_streak_geometry)))
        object.__setattr__(self, "bright_streak_thresholds", MappingProxyType(dict(self.bright_streak_thresholds)))
        object.__setattr__(
            self,
            "yolo_ignore_regions",
            MappingProxyType(
                {
                    view: tuple(tuple(region) for region in regions)
                    for view, regions in self.yolo_ignore_regions.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "efficientad_base_thresholds",
            MappingProxyType(dict(self.efficientad_base_thresholds)),
        )
        if self.efficientad_component_policies is not None:
            object.__setattr__(
                self,
                "efficientad_component_policies",
                MappingProxyType(dict(self.efficientad_component_policies)),
            )

    @property
    def shared_ignore_mask_index(self) -> Path | None:
        """Return the one shared path when Template and EfficientAD use the same mask."""
        if self.template_ignore_mask_index == self.efficientad_ignore_mask_index:
            return self.template_ignore_mask_index
        return None


def _immutable_value(value: Any) -> Any:
    """Recursively own structured evidence while keeping it JSON-shaped."""
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("details must not contain non-finite values")
        return value
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _immutable_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_immutable_value(item) for item in value)
    raise TypeError(f"details contains unsupported value: {type(value).__name__}")


def _immutable_details(details: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(details, Mapping):
        raise TypeError("details must be a mapping")
    return MappingProxyType({str(key): _immutable_value(value) for key, value in details.items()})


def fuse_demo_status(results: tuple[DemoBranchResult, ...]) -> DemoFinalStatus:
    """Fuse only after all model calls have already completed."""
    if not results or any(row.status is BranchStatus.ERROR for row in results):
        return DemoFinalStatus.ERROR
    if any(row.status is BranchStatus.NG for row in results):
        return DemoFinalStatus.NG
    return DemoFinalStatus.OK


def _owned_images(images: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    if tuple(images) != VIEW_ORDER:
        raise ValueError("images must use the canonical BMW eight-view order")
    owned: dict[str, np.ndarray] = {}
    for view, image in images.items():
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim not in {2, 3}:
            raise ValueError(f"{view} must be a uint8 grayscale/BGR image")
        copied = image.copy()
        copied.flags.writeable = False
        owned[view] = copied
    return MappingProxyType(owned)


def load_capture_directory(path: Path) -> Mapping[str, np.ndarray]:
    """Load ``<view>.<image extension>`` files from one offline directory."""
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"八视图目录不存在：{root}")
    supported = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
    by_stem: dict[str, list[Path]] = {view: [] for view in VIEW_ORDER}
    for candidate in root.iterdir():
        if candidate.is_file() and candidate.suffix.lower() in supported and candidate.stem in by_stem:
            by_stem[candidate.stem].append(candidate)
    images: dict[str, np.ndarray] = {}
    for view in VIEW_ORDER:
        matches = by_stem[view]
        if len(matches) != 1:
            raise ValueError(f"视角 {view} 需要且只能有一张同名图片，实际为 {len(matches)}")
        image = cv2.imread(str(matches[0]), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"无法读取视角图片：{matches[0]}")
        images[view] = image
    return _owned_images(images)


def load_manifest_sample(manifest_path: Path, sample_id: str) -> Mapping[str, np.ndarray]:
    """Load one complete prepared sample by ID without copying dataset files."""
    manifest = Path(manifest_path).expanduser().resolve()
    if not sample_id.strip():
        raise ValueError("sample_id不能为空")
    try:
        with manifest.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"sample_id", "view_id", "source_path"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError("数据清单缺少sample_id、view_id或source_path列")
            rows = [row for row in reader if row["sample_id"] == sample_id]
    except OSError as error:
        raise ValueError(f"无法读取数据清单：{manifest}: {error}") from error
    by_view: dict[str, str] = {}
    for row in rows:
        view = row["view_id"]
        if view not in VIEW_ORDER or view in by_view:
            raise ValueError(f"样本 {sample_id} 包含重复或未知视角：{view}")
        by_view[view] = row["source_path"]
    if tuple(view for view in VIEW_ORDER if view in by_view) != VIEW_ORDER or len(by_view) != len(VIEW_ORDER):
        missing = [view for view in VIEW_ORDER if view not in by_view]
        raise ValueError(f"样本 {sample_id} 不是完整八视图，缺少：{','.join(missing)}")
    images: dict[str, np.ndarray] = {}
    for view in VIEW_ORDER:
        image_path = Path(by_view[view]).expanduser()
        if not image_path.is_absolute():
            image_path = manifest.parent / image_path
        image = cv2.imread(str(image_path.resolve()), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"无法读取样本图片：{image_path}")
        images[view] = image
    return _owned_images(images)


def _json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取BMW八视图Demo配置：{path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("BMW八视图Demo配置必须是JSON对象")
    return payload


def _probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name}必须是有限数值")
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise ValueError(f"{name}必须位于0到1之间")
    return parsed


def _load_efficientad_component_policies(path: Path) -> Mapping[str, ComponentFilterPolicy]:
    """Load only the component-filter parameters used by inference."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取EfficientAD组件过滤配置：{path}: {error}") from error
    policies = payload.get("policies") if isinstance(payload, dict) else None
    if not isinstance(policies, dict) or set(policies) != set(VIEW_ORDER):
        raise ValueError("EfficientAD组件过滤配置必须覆盖八个标准视角")
    parsed: dict[str, ComponentFilterPolicy] = {}
    for view in VIEW_ORDER:
        item = policies[view]
        if not isinstance(item, dict):
            raise ValueError(f"EfficientAD组件过滤参数无效：{view}")
        try:
            parsed[view] = ComponentFilterPolicy(**item)
        except (TypeError, ValueError) as error:
            raise ValueError(f"EfficientAD组件过滤参数无效：{view}") from error
    return MappingProxyType(parsed)


def _view_numbers(raw: object, name: str, *, non_negative: bool = False) -> Mapping[str, float]:
    if not isinstance(raw, dict) or set(raw) != set(VIEW_ORDER):
        raise ValueError(f"{name}必须覆盖八个标准视角")
    parsed: dict[str, float] = {}
    for view in VIEW_ORDER:
        value = raw[view]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError(f"{name}.{view}必须是有限数值")
        number = float(value)
        if non_negative and number < 0:
            raise ValueError(f"{name}.{view}不能为负数")
        parsed[view] = number
    return MappingProxyType(parsed)


def load_demo_config(path: Path) -> EightViewDemoConfig:
    """Load one editable laboratory profile without release or digest binding."""
    resolved = Path(path).expanduser().resolve()
    payload = _json_object(resolved)
    base = resolved.parent

    def section(name: str) -> dict[str, Any]:
        value = payload.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"{name}配置必须是JSON对象")
        return value

    def resolve(raw: object, name: str, *, must_exist: bool = True) -> Path:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"{name}路径必须是非空字符串")
        candidate = Path(raw).expanduser()
        candidate = (candidate if candidate.is_absolute() else base / candidate).resolve()
        if must_exist and not candidate.is_file():
            raise ValueError(f"{name}不存在：{candidate}")
        return candidate

    def view_paths(raw: object, name: str) -> Mapping[str, Path]:
        if not isinstance(raw, dict) or set(raw) != set(VIEW_ORDER):
            raise ValueError(f"{name}必须覆盖八个标准视角")
        return MappingProxyType(
            {view: resolve(raw[view], f"{name} {view}") for view in VIEW_ORDER}
        )

    template = section("template")
    bright = section("bright_streak")
    efficientad = section("efficientad")
    yolo = section("yolo")
    template_models = view_paths(template.get("models"), "Template")
    template_thresholds = _view_numbers(
        template.get("thresholds"), "Template阈值", non_negative=True
    )
    weighted_regions_raw = template.get("weighted_regions")
    template_weighted_regions = None
    if weighted_regions_raw is not None:
        if not isinstance(weighted_regions_raw, dict):
            raise ValueError("Template weighted_regions必须是JSON对象")
        enabled = weighted_regions_raw.get("enabled")
        if not isinstance(enabled, bool):
            raise ValueError("Template weighted_regions.enabled必须是布尔值")
        weight_raw = weighted_regions_raw.get("weight")
        if isinstance(weight_raw, bool) or not isinstance(weight_raw, Real):
            raise ValueError("Template关键区域权重必须是有限且不小于1的数值")
        weight = float(weight_raw)
        if not math.isfinite(weight) or weight < 1.0:
            raise ValueError("Template关键区域权重必须是有限且不小于1的数值")
        outside_weight_raw = weighted_regions_raw.get("outside_weight", 1.0)
        if isinstance(outside_weight_raw, bool) or not isinstance(outside_weight_raw, Real):
            raise ValueError("Template关键区域外权重必须是有限正数")
        outside_weight = float(outside_weight_raw)
        if not math.isfinite(outside_weight) or outside_weight <= 0.0:
            raise ValueError("Template关键区域外权重必须是有限正数")
        template_weighted_regions = TemplateWeightedRegionsConfig(
            enabled=enabled,
            weight=weight,
            outside_weight=outside_weight,
            roi_config=resolve(weighted_regions_raw.get("roi_config"), "Template关键区域ROI"),
            thresholds=_view_numbers(
                weighted_regions_raw.get("thresholds"),
                "Template加权阈值",
                non_negative=True,
            ),
        )
    efficientad_checkpoints = view_paths(efficientad.get("checkpoints"), "EfficientAD")
    efficientad_thresholds = _view_numbers(
        efficientad.get("thresholds"), "EfficientAD阈值"
    )
    base_thresholds_raw = efficientad.get("base_thresholds")
    efficientad_base_thresholds = (
        efficientad_thresholds
        if base_thresholds_raw is None
        else _view_numbers(base_thresholds_raw, "EfficientAD基础阈值")
    )
    margin_raw = efficientad.get("threshold_margin", 0.0)
    if isinstance(margin_raw, bool) or not isinstance(margin_raw, Real):
        raise ValueError("EfficientAD threshold_margin必须是有限非负数值")
    efficientad_threshold_margin = float(margin_raw)
    if not math.isfinite(efficientad_threshold_margin) or efficientad_threshold_margin < 0:
        raise ValueError("EfficientAD threshold_margin必须是有限非负数值")
    threshold_source = efficientad.get("threshold_source")
    validation_status = efficientad.get("validation_status")
    for name, value in (
        ("EfficientAD threshold_source", threshold_source),
        ("EfficientAD validation_status", validation_status),
    ):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{name}必须是非空字符串")

    candidate = _probability(yolo["candidate_conf"], "candidate_conf")
    final = _probability(yolo["final_threshold"], "final_threshold")
    if candidate > final:
        raise ValueError("candidate_conf不能高于final_threshold")
    imgsz = yolo["imgsz"]
    if isinstance(imgsz, bool) or not isinstance(imgsz, int) or imgsz <= 0:
        raise ValueError("yolo.imgsz必须是正整数")
    yolo_ignore_regions: dict[str, tuple[tuple[int, int, int, int], ...]] = {}
    raw_ignore_regions = yolo.get("ignore_regions", {})
    if not isinstance(raw_ignore_regions, dict):
        raise ValueError("YOLO忽略区域必须按视角配置")
    for view, raw_regions in raw_ignore_regions.items():
        if view not in VIEW_ORDER or not isinstance(raw_regions, list):
            raise ValueError("YOLO忽略区域包含未知视角或无效区域列表")
        parsed_regions: list[tuple[int, int, int, int]] = []
        for raw_region in raw_regions:
            if (
                not isinstance(raw_region, list)
                or len(raw_region) != 4
                or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_region)
            ):
                raise ValueError("YOLO忽略区域坐标必须是四个整数")
            x1, y1, x2, y2 = raw_region
            if x1 < 0 or y1 < 0 or x1 >= x2 or y1 >= y2:
                raise ValueError("YOLO忽略区域坐标范围无效")
            parsed_regions.append((x1, y1, x2, y2))
        yolo_ignore_regions[view] = tuple(parsed_regions)
    bright_streak_engine = bright.get("engine")
    if bright_streak_engine != "tracked_profile_v3_manual_rotated_roi":
        raise ValueError("实验室版本只保留tracked_profile_v3_manual_rotated_roi光痕引擎")
    geometry = bright.get("geometry")
    thresholds = bright.get("thresholds")
    if not isinstance(geometry, dict) or not isinstance(thresholds, dict):
        raise ValueError("光痕geometry和thresholds必须直接写在Demo配置中")
    bright_streak_rotated_roi = resolve(bright.get("rotated_roi"), "光痕旋转ROI")
    template_mask = (
        resolve(template["ignore_mask_index"], "Template ignore mask")
        if "ignore_mask_index" in template
        else None
    )
    efficientad_mask = (
        resolve(efficientad["ignore_mask_index"], "EfficientAD ignore mask")
        if "ignore_mask_index" in efficientad
        else None
    )
    component_filter = (
        resolve(efficientad["component_filter"], "EfficientAD组件过滤配置")
        if "component_filter" in efficientad
        else None
    )
    component_policies = (
        None if component_filter is None else _load_efficientad_component_policies(component_filter)
    )
    trusted_config = payload.get("trusted_ok_reference")
    trusted_index = None
    if trusted_config is not None:
        if not isinstance(trusted_config, dict):
            raise ValueError("trusted_ok_reference配置必须是JSON对象")
        trusted_index = resolve(trusted_config.get("index"), "可信OK索引")
    demo_id = payload["demo_id"]
    if not isinstance(demo_id, str) or not demo_id.strip():
        raise ValueError("demo_id不能为空")
    return EightViewDemoConfig(
        path=resolved,
        demo_id=demo_id,
        capture_config=resolve(payload.get("capture_config"), "capture_config"),
        roi_config=resolve(payload.get("roi_config"), "roi_config"),
        prepared_manifest=resolve(
            payload.get("prepared_manifest"),
            "prepared_manifest",
            must_exist=False,
        ),
        result_root=resolve(payload.get("result_root"), "result_root", must_exist=False),
        template_models=template_models,
        template_thresholds=template_thresholds,
        efficientad_checkpoints=efficientad_checkpoints,
        efficientad_thresholds=efficientad_thresholds,
        efficientad_base_thresholds=efficientad_base_thresholds,
        efficientad_threshold_margin=efficientad_threshold_margin,
        efficientad_threshold_source=threshold_source,
        efficientad_validation_status=validation_status,
        bright_streak_engine=bright_streak_engine,
        bright_streak_geometry=geometry,
        bright_streak_thresholds=thresholds,
        yolo_checkpoint=resolve(yolo.get("checkpoint"), "YOLO checkpoint"),
        yolo_candidate_conf=candidate,
        yolo_final_threshold=final,
        yolo_imgsz=imgsz,
        yolo_ignore_regions=yolo_ignore_regions,
        bright_streak_rotated_roi=bright_streak_rotated_roi,
        efficientad_ignore_mask_index=efficientad_mask,
        efficientad_component_filter_config=component_filter,
        efficientad_component_policies=component_policies,
        template_ignore_mask_index=template_mask,
        template_weighted_regions=template_weighted_regions,
        trusted_ok_reference_index=trusted_index,
    )


__all__ = [
    "BranchStatus",
    "DemoBranch",
    "DemoBranchResult",
    "DemoFinalStatus",
    "EightViewDemoConfig",
    "EightViewInspection",
    "TemplateWeightedRegionsConfig",
    "fuse_demo_status",
    "load_capture_directory",
    "load_demo_config",
]
