"""Linux-only Ultralytics runtime for the one global ZS32 YOLO model.

Ultralytics is imported only when the verified loader is invoked.  The backend
may resize/letterbox the already-canonical ROI crop, but it never chooses or
applies another business ROI.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .base import (
    DeviceSpec,
    ModelContractError,
    ModelInput,
    RawModelScore,
    YoloDetection,
    require_text,
)
from .yolo import YoloDeploymentSpec, YoloPredictor
from .yolo_receipt import parse_yolo_trainer_source, runtime_package_tree_sha256


def _class_names(model: object) -> dict[int, str]:
    raw = getattr(model, "names", None)
    if isinstance(raw, list):
        return {index: str(value) for index, value in enumerate(raw)}
    if isinstance(raw, dict):
        try:
            return {int(key): str(value) for key, value in raw.items()}
        except (TypeError, ValueError) as error:
            raise ModelContractError("Ultralytics model names contain a non-integer class id") from error
    raise ModelContractError("Ultralytics model did not expose a class-name mapping")


def _scalar(value: object, *, field: str) -> float:
    """Read one scalar tensor/number and reject non-finite backend output."""
    try:
        item = value.item() if hasattr(value, "item") else value
        output = float(item)
    except (TypeError, ValueError) as error:
        raise ModelContractError(f"Ultralytics {field} is not a scalar") from error
    if not math.isfinite(output):
        raise ModelContractError(f"Ultralytics {field} must be finite")
    return output


def _canonical_file_path(value: object, *, field: str) -> Path:
    """Resolve one backend path to an existing regular file identity."""
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str) and value.strip():
        path = Path(value)
    else:
        raise ModelContractError(f"{field} must be a non-empty filesystem path")
    try:
        resolved = path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ModelContractError(f"{field} does not resolve to an existing file: {path}") from error
    if not resolved.is_file():
        raise ModelContractError(f"{field} must resolve to a regular file: {resolved}")
    return resolved


def _verify_result_path_identity(
    expected_paths: tuple[Path, ...],
    results: tuple[object, ...],
) -> None:
    """Fail closed unless Ultralytics returns each input path once and in order."""
    result_paths = tuple(
        _canonical_file_path(
            getattr(result, "path", None),
            field=f"Ultralytics result[{index}].path",
        )
        for index, result in enumerate(results)
    )
    if len(set(result_paths)) != len(result_paths):
        duplicates = sorted(
            str(path) for path in set(result_paths) if result_paths.count(path) > 1
        )
        raise ModelContractError(
            f"Ultralytics returned duplicate result paths: {duplicates}"
        )
    missing = sorted(str(path) for path in set(expected_paths) - set(result_paths))
    extra = sorted(str(path) for path in set(result_paths) - set(expected_paths))
    if missing or extra:
        raise ModelContractError(
            f"Ultralytics result path set mismatch; missing={missing}, extra={extra}"
        )
    if len(result_paths) != len(expected_paths):
        raise ModelContractError(
            f"Ultralytics returned {len(result_paths)} result paths for "
            f"{len(expected_paths)} inputs"
        )
    if result_paths != expected_paths:
        raise ModelContractError(
            "Ultralytics result path order differs from the canonical input order"
        )


class UltralyticsBatchPredictor(YoloPredictor):
    """Loaded global predictor that preserves every canonical sample identity."""

    def __init__(self, model: object, spec: YoloDeploymentSpec, device: DeviceSpec) -> None:
        self._model = model
        self._spec = spec
        self._device = device

    def predict_batch(
        self,
        samples: Sequence[ModelInput],
        *,
        inspection_id: str,
    ) -> Sequence[RawModelScore]:
        inspection_id = require_text(inspection_id, "inspection_id")
        materialized = tuple(samples)
        if not materialized:
            raise ModelContractError("YOLO prediction requires at least one canonical crop")
        identities = {
            (item.sample.capture_set_id, item.sample.view_id) for item in materialized
        }
        if len(identities) != len(materialized):
            raise ModelContractError("YOLO batch contains duplicate capture/view identities")
        for sample in materialized:
            sample.verify_crop()
        source_paths = tuple(
            _canonical_file_path(item.crop_path, field="YOLO canonical crop path")
            for item in materialized
        )
        if len(set(source_paths)) != len(source_paths):
            raise ModelContractError("YOLO batch contains duplicate canonical crop paths")
        settings = self._spec.runtime
        try:
            results = tuple(
                self._model.predict(
                    source=[str(path) for path in source_paths],
                    imgsz=settings.imgsz,
                    conf=settings.candidate_conf,
                    iou=settings.iou,
                    max_det=settings.max_det,
                    classes=[0],
                    agnostic_nms=False,
                    device=self._device.device,
                    save=False,
                    save_txt=False,
                    save_conf=False,
                    stream=False,
                    verbose=False,
                )
            )
        except Exception as error:
            raise RuntimeError(f"Ultralytics batch inference failed: {error}") from error
        _verify_result_path_identity(source_paths, results)
        return tuple(
            self._result(sample, result, inspection_id=inspection_id)
            for sample, result in zip(materialized, results, strict=True)
        )

    def _result(
        self,
        model_input: ModelInput,
        result: object,
        *,
        inspection_id: str,
    ) -> RawModelScore:
        sample = model_input.sample
        shape = getattr(result, "orig_shape", None)
        if (
            not isinstance(shape, (tuple, list))
            or len(shape) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape)
        ):
            raise ModelContractError("Ultralytics result orig_shape must be positive (height, width)")
        height, width = shape
        if (width, height) != (sample.crop_width, sample.crop_height):
            raise ModelContractError("Ultralytics result shape differs from canonical crop dimensions")
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            raise ModelContractError("Ultralytics result is missing boxes")
        raw_xyxy = getattr(boxes, "xyxy", None)
        raw_conf = getattr(boxes, "conf", None)
        raw_cls = getattr(boxes, "cls", None)
        if raw_xyxy is None or raw_conf is None or raw_cls is None:
            raise ModelContractError("Ultralytics boxes are missing xyxy/conf/cls")
        if not (len(raw_xyxy) == len(raw_conf) == len(raw_cls)):
            raise ModelContractError("Ultralytics xyxy/conf/cls lengths disagree")
        detections: list[YoloDetection] = []
        for xyxy, confidence, class_id in zip(raw_xyxy, raw_conf, raw_cls, strict=True):
            class_value = _scalar(class_id, field="class id")
            if not class_value.is_integer() or int(class_value) != 0:
                raise ModelContractError("Ultralytics emitted a class outside the single defect class")
            try:
                coordinates = tuple(_scalar(value, field="box coordinate") for value in xyxy)
            except TypeError as error:
                raise ModelContractError("Ultralytics xyxy row is not iterable") from error
            if len(coordinates) != 4:
                raise ModelContractError("Ultralytics xyxy row must contain four coordinates")
            x1, y1, x2, y2 = coordinates
            if x2 <= x1 or y2 <= y1:
                raise ModelContractError("Ultralytics emitted a non-positive-area box")
            normalized_unclipped = (x1 / width, y1 / height, x2 / width, y2 / height)
            normalized = tuple(min(1.0, max(0.0, value)) for value in normalized_unclipped)
            clipped = normalized != normalized_unclipped
            nx1, ny1, nx2, ny2 = normalized
            if nx2 <= nx1 or ny2 <= ny1:
                raise ModelContractError("Ultralytics box lies outside the canonical crop")
            detections.append(
                YoloDetection(
                    class_id=0,
                    class_name="defect",
                    confidence=_scalar(confidence, field="confidence"),
                    xyxy_norm=(nx1, ny1, nx2, ny2),
                    area_ratio=(nx2 - nx1) * (ny2 - ny1),
                    clipped=clipped,
                )
            )
        detections.sort(
            key=lambda item: (-item.confidence, *item.xyxy_norm)
        )
        maximum = max((item.confidence for item in detections), default=0.0)
        return RawModelScore(
            inspection_id=inspection_id,
            part_instance_id=sample.part.part_instance_id,
            capture_set_id=sample.capture_set_id,
            hand=sample.part.hand.value,
            view=sample.view_id,
            branch="yolo",
            score=maximum,
            model_family="yolo",
            model_digest=self._spec.model_digest,
            roi_config_id=sample.roi_config_id,
            roi_digest=model_input.roi_digest,
            source_sha256=sample.source_sha256,
            crop_sha256=sample.crop_sha256,
            detections=tuple(item.to_dict() for item in detections),
        )


class UltralyticsRuntimeBackend:
    """Verified Linux loader passed to :class:`UltralyticsYoloAdapter`."""

    def __call__(
        self,
        spec: YoloDeploymentSpec,
        device: DeviceSpec,
    ) -> UltralyticsBatchPredictor:
        if device.accelerator != "gpu":
            raise ModelContractError("production YOLO runtime requires accelerator='gpu'")
        spec.verify()
        try:
            import torch
            import ultralytics
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(f"Ultralytics Linux runtime dependency is unavailable: {error}") from error
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for the production YOLO runtime")
        trainer_source = parse_yolo_trainer_source(spec.provenance.trainer_source)
        package_file = getattr(ultralytics, "__file__", None)
        if not isinstance(package_file, str) or not package_file:
            raise ModelContractError("Ultralytics package did not expose an import source path")
        actual_runtime_tree = runtime_package_tree_sha256(Path(package_file).resolve().parent)
        if actual_runtime_tree != trainer_source.runtime_package_tree_sha256:
            raise ModelContractError(
                "Ultralytics runtime package tree mismatch: expected "
                f"{trainer_source.runtime_package_tree_sha256!r}, "
                f"found {actual_runtime_tree!r}"
            )
        try:
            model = YOLO(str(spec.weights.path), task="detect")
        except Exception as error:
            raise RuntimeError(f"Ultralytics failed to load verified best.pt: {error}") from error
        if _class_names(model) != {0: "defect"}:
            raise ModelContractError("loaded YOLO model must expose exactly class 0='defect'")
        return UltralyticsBatchPredictor(model, spec, device)


__all__ = ["UltralyticsBatchPredictor", "UltralyticsRuntimeBackend"]
