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

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER


class DemoBranch(str, Enum):
    """The four evidence branches shown by the laboratory Demo."""

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


@dataclass(frozen=True, slots=True)
class EightViewDemoConfig:
    """Resolved paths and editable thresholds for one Demo profile."""

    path: Path
    demo_id: str
    capture_config: Path
    roi_config: Path
    prepared_manifest: Path
    training_run: Path
    result_root: Path
    template_models: Mapping[str, Path]
    efficientad_checkpoints: Mapping[str, Path]
    efficientad_thresholds: Mapping[str, float]
    bright_streak_config: Path
    yolo_checkpoint: Path
    yolo_candidate_conf: float
    yolo_final_threshold: float
    yolo_imgsz: int
    views: tuple[str, ...] = VIEW_ORDER

    def __post_init__(self) -> None:
        object.__setattr__(self, "template_models", MappingProxyType(dict(self.template_models)))
        object.__setattr__(self, "efficientad_checkpoints", MappingProxyType(dict(self.efficientad_checkpoints)))
        object.__setattr__(self, "efficientad_thresholds", MappingProxyType(dict(self.efficientad_thresholds)))


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


def _load_efficientad_thresholds(path: Path) -> Mapping[str, float]:
    """Load the Task 3 threshold contract and reject unsafe Demo assets."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取EfficientAD整件阈值资产：{path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("EfficientAD整件阈值资产必须是JSON对象")
    for flag in ("demo_only", "test_used_for_selection"):
        if payload.get(flag) is not True:
            raise ValueError(f"EfficientAD整件阈值资产必须显式标记{flag}=true")
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, dict) or tuple(thresholds) != VIEW_ORDER:
        raise ValueError("EfficientAD整件阈值资产必须按标准顺序覆盖八个视角")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in thresholds.values()):
        raise ValueError("EfficientAD整件阈值必须是有限数值")

    # Reuse the Task 3 threshold contract rather than maintaining a second
    # numerical validator in the Demo loader.
    from bmw_inspection.lab.efficientad_thresholds import _validated_thresholds

    try:
        views, parsed = _validated_thresholds(thresholds)
    except ValueError as error:
        raise ValueError("EfficientAD整件阈值必须是有限数值") from error
    if views != VIEW_ORDER:
        raise ValueError("EfficientAD整件阈值资产必须按标准顺序覆盖八个视角")
    return MappingProxyType(parsed)


def load_demo_config(path: Path) -> EightViewDemoConfig:
    """Load the small profile and resolve the exact trained model assets."""
    resolved = Path(path).expanduser().resolve()
    payload = _json_object(resolved)
    required = {
        "schema_version",
        "demo_id",
        "capture_config",
        "roi_config",
        "prepared_manifest",
        "training_run",
        "result_root",
        "bright_streak",
        "efficientad",
        "yolo",
    }
    if set(payload) != required or payload["schema_version"] != 1:
        raise ValueError("BMW八视图Demo配置字段或schema_version不正确")
    yolo = payload["yolo"]
    if not isinstance(yolo, dict) or set(yolo) != {"candidate_conf", "final_threshold", "imgsz"}:
        raise ValueError("yolo配置字段不正确")
    candidate = _probability(yolo["candidate_conf"], "candidate_conf")
    final = _probability(yolo["final_threshold"], "final_threshold")
    if candidate > final:
        raise ValueError("candidate_conf不能高于final_threshold")
    imgsz = yolo["imgsz"]
    if isinstance(imgsz, bool) or not isinstance(imgsz, int) or imgsz <= 0:
        raise ValueError("yolo.imgsz必须是正整数")
    bright_streak_config = payload["bright_streak"]
    if not isinstance(bright_streak_config, dict) or set(bright_streak_config) != {"config"}:
        raise ValueError("bright_streak配置字段不正确")
    efficientad_config = payload["efficientad"]
    if not isinstance(efficientad_config, dict) or set(efficientad_config) != {"threshold_artifact"}:
        raise ValueError("efficientad配置字段不正确")
    base = resolved.parent

    def resolve(raw: object) -> Path:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("配置路径必须是非空字符串")
        candidate_path = Path(raw).expanduser()
        return (candidate_path if candidate_path.is_absolute() else base / candidate_path).resolve()

    capture_config = resolve(payload["capture_config"])
    roi_config = resolve(payload["roi_config"])
    training_run = resolve(payload["training_run"])
    bright = resolve(bright_streak_config["config"])
    efficientad_thresholds = _load_efficientad_thresholds(resolve(efficientad_config["threshold_artifact"]))
    for label, asset in (("capture_config", capture_config), ("roi_config", roi_config)):
        if not asset.is_file():
            raise ValueError(f"{label}不存在：{asset}")
    template_models = {view: training_run / "template" / view / "model.json" for view in VIEW_ORDER}
    efficientad = {view: training_run / "efficientad" / view / "model.ckpt" for view in VIEW_ORDER}
    yolo_checkpoint = training_run / "yolo/train/weights/best.pt"
    for label, asset in {
        **{f"Template {view}": model for view, model in template_models.items()},
        **{f"EfficientAD {view}": model for view, model in efficientad.items()},
        "bright_streak": bright,
        "YOLO": yolo_checkpoint,
    }.items():
        if not asset.is_file():
            raise ValueError(f"{label}模型不存在：{asset}")
    demo_id = payload["demo_id"]
    if not isinstance(demo_id, str) or not demo_id.strip():
        raise ValueError("demo_id不能为空")
    return EightViewDemoConfig(
        path=resolved,
        demo_id=demo_id,
        capture_config=capture_config,
        roi_config=roi_config,
        prepared_manifest=resolve(payload["prepared_manifest"]),
        training_run=training_run,
        result_root=resolve(payload["result_root"]),
        template_models=template_models,
        efficientad_checkpoints=efficientad,
        efficientad_thresholds=efficientad_thresholds,
        bright_streak_config=bright,
        yolo_checkpoint=yolo_checkpoint,
        yolo_candidate_conf=candidate,
        yolo_final_threshold=final,
        yolo_imgsz=imgsz,
    )


__all__ = [
    "BranchStatus",
    "DemoBranch",
    "DemoBranchResult",
    "DemoFinalStatus",
    "EightViewDemoConfig",
    "EightViewInspection",
    "fuse_demo_status",
    "load_capture_directory",
    "load_demo_config",
]
