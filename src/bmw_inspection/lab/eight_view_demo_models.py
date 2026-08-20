"""Resident model adapters for the BMW eight-view laboratory Demo."""

from __future__ import annotations

import json
import math
import tempfile
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Any

import cv2
import numpy as np

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER
from bmw_inspection.lab.eight_view_demo import (
    BranchStatus,
    DemoBranch,
    DemoBranchResult,
    EightViewInspection,
    EightViewDemoConfig,
    fuse_demo_status,
)
from bmw_inspection.lab.eight_view_roi import load_roi_config


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """Small model-independent output consumed by the Demo runtime."""

    status: BranchStatus
    score: float | None
    threshold: float | None
    reason: str
    overlay: np.ndarray | None

    def __post_init__(self) -> None:
        if not isinstance(self.status, BranchStatus):
            raise TypeError("status must be BranchStatus")
        for name in ("score", "threshold"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite or None")
        if not self.reason.strip():
            raise ValueError("reason must not be empty")


ViewPredictor = Callable[[str, np.ndarray], ModelOutput]
BrightPredictor = Callable[[np.ndarray], ModelOutput]


@dataclass(frozen=True, slots=True)
class _TemplateModel:
    input_width: int
    input_height: int
    target_width: int
    target_height: int
    max_shift: int
    threshold: float
    templates: tuple[np.ndarray, ...]


def load_part_rois(path: Path) -> Mapping[str, tuple[int, int, int, int]]:
    """Load the manifest-bound eight-view ROI asset in canonical order."""
    config = load_roi_config(Path(path).expanduser().resolve())
    return MappingProxyType({view: config.part_rois[view] for view in VIEW_ORDER})


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

    def __init__(self, model_paths: Mapping[str, Path]) -> None:
        if tuple(model_paths) != VIEW_ORDER:
            raise ValueError("Template模型必须按标准顺序覆盖八个视角")
        self._models = MappingProxyType(
            {view: self._load(view, Path(model_paths[view]).expanduser().resolve()) for view in VIEW_ORDER}
        )

    @staticmethod
    def _load(view: str, path: Path) -> _TemplateModel:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("view_id") != view:
            raise ValueError(f"Template模型视角不匹配：{path}")
        preprocess = payload.get("preprocess")
        items = payload.get("templates")
        if not isinstance(preprocess, dict) or not isinstance(items, list) or not items:
            raise ValueError(f"Template模型结构不正确：{path}")
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
            threshold=float(payload["threshold"]),
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
        matches: list[tuple[float, np.ndarray]] = []
        for template in model.templates:
            response = cv2.matchTemplate(padded, template, cv2.TM_CCOEFF_NORMED)
            _minimum, maximum, _minimum_location, _maximum_location = cv2.minMaxLoc(response)
            matches.append((float(maximum), template))
        similarity, best = max(matches, key=lambda item: item[0])
        risk = max(0.0, 1.0 - similarity)
        difference = cv2.absdiff(query, best)
        heatmap = cv2.applyColorMap(difference, cv2.COLORMAP_TURBO)
        base = cv2.cvtColor(query, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(base, 0.65, heatmap, 0.35, 0.0)
        passed = risk <= model.threshold
        return ModelOutput(
            BranchStatus.PASS if passed else BranchStatus.NG,
            risk,
            model.threshold,
            f"模板风险 {risk:.4f}，阈值 {model.threshold:.4f}",
            overlay,
        )


class EightViewBrightStreakPredictor:
    """Use the calibrated full-image `front_left` rule and render its evidence."""

    def __init__(self, config_path: Path) -> None:
        from bmw_inspection.contracts import load_config

        self._config = load_config(Path(config_path).expanduser().resolve())

    def predict(self, image: np.ndarray) -> ModelOutput:
        from bmw_inspection.contracts import DemoStatus
        from bmw_inspection.detector import detect_bright_streak_evidence, render_evidence

        decision = detect_bright_streak_evidence(image, self._config)
        metrics = decision.metrics
        if decision.status is DemoStatus.ERROR:
            status = BranchStatus.ERROR
        elif decision.status is DemoStatus.OK:
            status = BranchStatus.PASS
        else:
            status = BranchStatus.NG
        return ModelOutput(
            status,
            None if metrics is None else metrics.coverage_ratio,
            self._config.min_coverage_ratio,
            {
                DemoStatus.OK: "亮痕存在且连续",
                DemoStatus.NG_NO_STREAK: "未检测到有效亮痕",
                DemoStatus.NG_BROKEN: "亮痕存在但不连续",
                DemoStatus.ERROR: "光痕检测输入异常",
            }[decision.status],
            render_evidence(image, decision),
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
        self._model = model_factory(resolved)
        names = getattr(self._model, "names", None)
        parsed = {index: name for index, name in enumerate(names)} if isinstance(names, list) else dict(names or {})
        if parsed != {0: "defect"}:
            raise ValueError("YOLO模型必须只包含类别defect")
        self._lock = threading.Lock()

    def predict(self, view: str, image: np.ndarray) -> ModelOutput:
        del view
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
        final_count = 0
        for xyxy, confidence, class_id in zip(coordinates, confidences, classes, strict=True):
            if int(class_id) != 0:
                raise ValueError("YOLO输出了非defect类别")
            final = float(confidence) >= self._final_threshold
            final_count += int(final)
            color = (0, 0, 255) if final else (0, 170, 255)
            x1, y1, x2, y2 = (int(round(float(value))) for value in xyxy)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3 if final else 2)
            cv2.putText(
                overlay,
                f"defect {float(confidence):.2f}",
                (max(0, x1), max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        score = max((float(value) for value in confidences), default=0.0)
        return ModelOutput(
            BranchStatus.NG if final_count else BranchStatus.PASS,
            score,
            self._final_threshold,
            f"最终缺陷框 {final_count} 个，候选框 {len(confidences)} 个",
            overlay,
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
        predictor_factory: EfficientPredictorFactory = _AnomalibEfficientPredictor,
    ) -> None:
        if tuple(checkpoints) != VIEW_ORDER:
            raise ValueError("EfficientAD模型必须按标准顺序覆盖八个视角")
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
            score, anomalous, anomaly_map = self._predictors[view](image)
        if not math.isfinite(score) or not np.isfinite(anomaly_map).all():
            raise ValueError("EfficientAD输出包含非有限数值")
        minimum = float(anomaly_map.min())
        spread = float(np.ptp(anomaly_map))
        normalized = np.zeros(anomaly_map.shape, dtype=np.uint8)
        if spread > 0:
            normalized = np.rint((anomaly_map - minimum) * 255.0 / spread).astype(np.uint8)
        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
        heatmap = cv2.resize(heatmap, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
        overlay = cv2.addWeighted(base, 0.6, heatmap, 0.4, 0.0)
        return ModelOutput(
            BranchStatus.NG if anomalous else BranchStatus.PASS,
            score,
            0.5,
            f"EfficientAD异常分数 {score:.4f}",
            overlay,
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

    def inspect(self, images: Mapping[str, np.ndarray], *, capture_id: str) -> EightViewInspection:
        """Run 8 Template + 1 light streak + 8 YOLO + 8 EfficientAD checks."""
        if tuple(images) != VIEW_ORDER:
            raise ValueError("images must use the canonical BMW eight-view order")
        started = perf_counter()
        crops = {view: self._crop(view, images[view]) for view in VIEW_ORDER}
        results: list[DemoBranchResult] = []
        for view in VIEW_ORDER:
            results.append(self._call(DemoBranch.TEMPLATE, view, self._template, view, crops[view]))
        results.append(
            self._call(
                DemoBranch.BRIGHT_STREAK,
                "front_left",
                self._bright_streak,
                images["front_left"],
            )
        )
        for view in VIEW_ORDER:
            results.append(self._call(DemoBranch.YOLO, view, self._yolo, view, crops[view]))
        for view in VIEW_ORDER:
            results.append(self._call(DemoBranch.EFFICIENTAD, view, self._efficientad, view, crops[view]))
        rows = tuple(results)
        return EightViewInspection(
            capture_id=capture_id,
            images=images,
            results=rows,
            final_status=fuse_demo_status(rows),
            elapsed_ms=(perf_counter() - started) * 1000.0,
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
            )


def build_model_suite(config: EightViewDemoConfig) -> EightViewModelSuite:
    """Build the four resident model branches from one resolved Demo profile."""
    template = EightViewTemplatePredictor(config.template_models)
    bright_streak = EightViewBrightStreakPredictor(config.bright_streak_config)
    yolo = EightViewYoloPredictor(
        config.yolo_checkpoint,
        candidate_conf=config.yolo_candidate_conf,
        final_threshold=config.yolo_final_threshold,
        imgsz=config.yolo_imgsz,
    )
    efficientad = EightViewEfficientAdPredictor(config.efficientad_checkpoints)
    return EightViewModelSuite(
        rois=load_part_rois(config.roi_config),
        template_predictor=template.predict,
        bright_streak_predictor=bright_streak.predict,
        yolo_predictor=yolo.predict,
        efficientad_predictor=efficientad.predict,
    )


__all__ = [
    "EightViewBrightStreakPredictor",
    "EightViewEfficientAdPredictor",
    "EightViewModelSuite",
    "EightViewTemplatePredictor",
    "EightViewYoloPredictor",
    "ModelOutput",
    "build_model_suite",
    "load_part_rois",
]
