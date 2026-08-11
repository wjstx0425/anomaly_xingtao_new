# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""One resident global one-class YOLO backend for BMW laboratory inspection."""

from __future__ import annotations

import csv
import ctypes
import errno
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import math
import os
import re
import shutil
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Any

import cv2
import numpy as np
import yaml

from bmw_inspection.lab.config import LabExperimentConfig
from bmw_inspection.lab.contracts import BranchEvidence, BranchName, BranchStatus, ViewId


ModelFactory = Callable[[Path], Any]
_VERSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _default_model_factory(checkpoint: Path) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Ultralytics is required; install the bmw-lab optional dependency") from error
    return YOLO(str(checkpoint))


def _class_names(model: Any) -> dict[int, str]:
    names = getattr(model, "names", None)
    if isinstance(names, list):
        parsed = {index: str(name) for index, name in enumerate(names)}
    elif isinstance(names, Mapping):
        try:
            parsed = {int(index): str(name) for index, name in names.items()}
        except (TypeError, ValueError) as error:
            raise ValueError("YOLO class names are invalid") from error
    else:
        raise ValueError("YOLO model must expose class names")
    if parsed != {0: "defect"}:
        raise ValueError("YOLO model must have exactly one class named 'defect'")
    return parsed


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _view_id(value: ViewId | str) -> ViewId:
    try:
        return ViewId(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unknown BMW view_id: {value!r}") from error


def _validate_rois(
    part_rois: Mapping[ViewId, tuple[int, int, int, int]],
    *,
    require_all: bool,
) -> Mapping[ViewId, tuple[int, int, int, int]]:
    normalized: dict[ViewId, tuple[int, int, int, int]] = {}
    for raw_view, raw_roi in part_rois.items():
        view_id = _view_id(raw_view)
        if (
            not isinstance(raw_roi, tuple)
            or len(raw_roi) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_roi)
        ):
            raise ValueError(f"part ROI for {view_id.value} must contain four integers")
        x1, y1, x2, y2 = raw_roi
        if not (0 <= x1 < x2 and 0 <= y1 < y2):
            raise ValueError(f"part ROI for {view_id.value} must have positive area")
        normalized[view_id] = raw_roi
    if require_all and set(normalized) != set(ViewId):
        raise ValueError("YOLO part_rois must contain exactly the six BMW views")
    return MappingProxyType(normalized)


def _prepare_roi(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim not in {2, 3}:
        raise ValueError("YOLO input must be a uint8 grayscale/BGR/BGRA image")
    x1, y1, x2, y2 = roi
    height, width = image.shape[:2]
    if x2 > width or y2 > height:
        raise ValueError(f"YOLO ROI lies outside {width}x{height} input")
    crop = image[y1:y2, x1:x2]
    if crop.ndim == 2:
        return np.repeat(crop[..., None], 3, axis=2)
    if crop.shape[2] == 3:
        return crop.copy()
    if crop.shape[2] == 4:
        return cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
    raise ValueError("YOLO input must be a uint8 grayscale/BGR/BGRA image")


@dataclass(frozen=True, slots=True)
class DetectionBox:
    """One immutable full-view one-class detection candidate."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str = "defect"

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in (self.x1, self.y1, self.x2, self.y2, self.confidence))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("YOLO box coordinates and confidence must be finite")
        if not (values[0] < values[2] and values[1] < values[3]):
            raise ValueError("YOLO box must have positive area")
        _probability(self.confidence, "YOLO confidence")
        if self.class_name != "defect":
            raise ValueError("YOLO predicted class must be defect")


@dataclass(frozen=True, slots=True)
class YoloEvidence(BranchEvidence):
    """Branch evidence with candidates kept separate from business-final boxes."""

    candidates: tuple[DetectionBox, ...]
    final_boxes: tuple[DetectionBox, ...]
    roi_xyxy: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        BranchEvidence.__post_init__(self)
        if not isinstance(self.candidates, tuple) or not all(
            isinstance(box, DetectionBox) for box in self.candidates
        ):
            raise TypeError("candidates must be a tuple of DetectionBox")
        if not isinstance(self.final_boxes, tuple) or not all(
            isinstance(box, DetectionBox) for box in self.final_boxes
        ):
            raise TypeError("final_boxes must be a tuple of DetectionBox")
        if any(box not in self.candidates for box in self.final_boxes):
            raise ValueError("final_boxes must be drawn from candidates")
        if self.threshold is None:
            raise ValueError("YOLO threshold must be present")
        threshold = _probability(self.threshold, "YOLO final threshold")
        expected = tuple(box for box in self.candidates if box.confidence >= threshold)
        if self.final_boxes != expected:
            raise ValueError("final_boxes must exactly match candidates at or above threshold")
        expected_status = BranchStatus.NG if expected else BranchStatus.PASS
        if self.status is not expected_status:
            raise ValueError("YOLO status must agree with final_boxes")
        if (
            not isinstance(self.roi_xyxy, tuple)
            or len(self.roi_xyxy) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) for value in self.roi_xyxy)
        ):
            raise ValueError("roi_xyxy must contain four integers")
        roi_x1, roi_y1, roi_x2, roi_y2 = self.roi_xyxy
        if not (0 <= roi_x1 < roi_x2 and 0 <= roi_y1 < roi_y2):
            raise ValueError("roi_xyxy must have positive area")


def _infer_candidates(
    model: Any,
    image: np.ndarray,
    *,
    roi: tuple[int, int, int, int],
    candidate_conf: float,
    imgsz: int,
) -> tuple[DetectionBox, ...]:
    prepared = _prepare_roi(image, roi)
    results = model.predict(source=prepared, conf=candidate_conf, imgsz=imgsz, verbose=False)
    if not isinstance(results, Sequence) or len(results) != 1:
        raise ValueError("YOLO predict must return exactly one result")
    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return ()
    xyxy = _to_numpy(getattr(boxes, "xyxy", None)).astype(np.float64, copy=False)
    confidences = _to_numpy(getattr(boxes, "conf", None)).astype(np.float64, copy=False).reshape(-1)
    class_ids = _to_numpy(getattr(boxes, "cls", None)).astype(np.float64, copy=False).reshape(-1)
    if xyxy.size == 0:
        xyxy = xyxy.reshape((0, 4))
    if xyxy.ndim != 2 or xyxy.shape[1] != 4 or len(xyxy) != len(confidences) or len(xyxy) != len(class_ids):
        raise ValueError("YOLO result boxes have inconsistent dimensions")
    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    roi_width, roi_height = roi_x2 - roi_x1, roi_y2 - roi_y1
    candidates: list[DetectionBox] = []
    for coordinates, confidence, class_id in zip(xyxy, confidences, class_ids, strict=True):
        parsed_confidence = _probability(float(confidence), "YOLO confidence")
        if not math.isfinite(float(class_id)) or int(class_id) != class_id or int(class_id) != 0:
            raise ValueError("YOLO predicted class must be defect")
        if not np.isfinite(coordinates).all():
            raise ValueError("YOLO box coordinates must be finite")
        local_x1, local_y1, local_x2, local_y2 = (float(value) for value in coordinates)
        local_x1 = min(max(local_x1, 0.0), float(roi_width))
        local_y1 = min(max(local_y1, 0.0), float(roi_height))
        local_x2 = min(max(local_x2, 0.0), float(roi_width))
        local_y2 = min(max(local_y2, 0.0), float(roi_height))
        if parsed_confidence >= candidate_conf and local_x1 < local_x2 and local_y1 < local_y2:
            candidates.append(
                DetectionBox(
                    x1=local_x1 + roi_x1,
                    y1=local_y1 + roi_y1,
                    x2=local_x2 + roi_x1,
                    y2=local_y2 + roi_y1,
                    confidence=parsed_confidence,
                )
            )
    return tuple(candidates)


class YoloBackend:
    """One checkpoint loaded once and shared across all six fixed BMW views."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        part_rois: Mapping[ViewId, tuple[int, int, int, int]],
        candidate_conf: float,
        final_threshold: float,
        required_for_ok: bool,
        imgsz: int = 1280,
        model_factory: ModelFactory = _default_model_factory,
    ) -> None:
        resolved = Path(checkpoint).expanduser().resolve()
        if not resolved.is_file():
            raise ValueError(f"YOLO checkpoint does not exist: {resolved}")
        if not isinstance(required_for_ok, bool):
            raise TypeError("required_for_ok must be bool")
        if isinstance(imgsz, bool) or not isinstance(imgsz, int) or imgsz <= 0:
            raise ValueError("imgsz must be a positive integer")
        self._candidate_conf = _probability(candidate_conf, "candidate_conf")
        self._final_threshold = _probability(final_threshold, "final_threshold")
        if self._candidate_conf > self._final_threshold:
            raise ValueError("candidate_conf must not exceed final_threshold")
        self._part_rois = _validate_rois(part_rois, require_all=True)
        self._required_for_ok = required_for_ok
        self._imgsz = imgsz
        self._checkpoint = resolved
        self._model_id = _sha256(resolved)
        self._model = model_factory(resolved)
        _class_names(self._model)
        self._inference_lock = threading.Lock()

    @classmethod
    def from_experiment_config(
        cls,
        config: LabExperimentConfig,
        *,
        model_factory: ModelFactory = _default_model_factory,
        imgsz: int = 1280,
    ) -> YoloBackend:
        """Load the checkpoint and hot-editable thresholds from one profile snapshot."""
        if not config.yolo.enabled or config.yolo.checkpoint is None:
            raise ValueError("YOLO branch is disabled or missing its checkpoint")
        return cls(
            config.yolo.checkpoint,
            part_rois=config.part_rois,
            candidate_conf=config.yolo.candidate_conf,
            final_threshold=config.yolo.final_threshold,
            required_for_ok=BranchName.YOLO in config.required_for_ok,
            imgsz=imgsz,
            model_factory=model_factory,
        )

    def predict(self, view_id: ViewId | str, image: np.ndarray) -> YoloEvidence:
        """Return all diagnostic candidates and only business-final detections."""
        parsed_view = _view_id(view_id)
        started = perf_counter()
        with self._inference_lock:
            candidates = _infer_candidates(
                self._model,
                image,
                roi=self._part_rois[parsed_view],
                candidate_conf=self._candidate_conf,
                imgsz=self._imgsz,
            )
        final_boxes = tuple(box for box in candidates if box.confidence >= self._final_threshold)
        elapsed_ms = (perf_counter() - started) * 1000.0
        max_confidence = max((box.confidence for box in candidates), default=0.0)
        return YoloEvidence(
            branch=BranchName.YOLO,
            view_id=parsed_view,
            status=BranchStatus.NG if final_boxes else BranchStatus.PASS,
            required_for_ok=self._required_for_ok,
            score=max_confidence,
            threshold=self._final_threshold,
            elapsed_ms=elapsed_ms,
            reason=(
                f"YOLO final detections={len(final_boxes)}, candidates={len(candidates)}, "
                f"max_confidence={max_confidence:.6f}"
            ),
            model_id=self._model_id,
            artifact_paths={"checkpoint": str(self._checkpoint)},
            candidates=candidates,
            final_boxes=final_boxes,
            roi_xyxy=self._part_rois[parsed_view],
        )


def render_final_overlay(image: np.ndarray, evidence: YoloEvidence) -> np.ndarray:
    """Draw red rectangles for final detections only, never raw candidates."""
    if not isinstance(evidence, YoloEvidence):
        raise TypeError("evidence must be YoloEvidence")
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim not in {2, 3}:
        raise ValueError("overlay input must be a uint8 image")
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    for box in evidence.final_boxes:
        cv2.rectangle(
            overlay,
            (int(round(box.x1)), int(round(box.y1))),
            (int(round(box.x2)), int(round(box.y2))),
            (0, 0, 255),
            2,
        )
    return overlay


@dataclass(frozen=True, slots=True)
class YoloScoreRow:
    """One image-level confidence used for calibration-only F1 fitting."""

    sample_id: str
    split: str
    label: str
    max_confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must be non-empty")
        if self.split not in {"calibration", "final_test"}:
            raise ValueError("YOLO score split must be calibration or final_test")
        if self.label not in {"normal", "defect"}:
            raise ValueError("YOLO score label must be normal or defect")
        _probability(self.max_confidence, "max_confidence")


@dataclass(frozen=True, slots=True)
class YoloThresholdFit:
    threshold: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


def fit_yolo_final_threshold(
    rows: Sequence[YoloScoreRow],
    *,
    candidate_conf: float,
) -> YoloThresholdFit:
    """Maximise image-level defect F1 using calibration rows only."""
    floor = _probability(candidate_conf, "candidate_conf")
    calibration = [row for row in rows if row.split == "calibration"]
    if not calibration or {row.label for row in calibration} != {"normal", "defect"}:
        raise ValueError("YOLO threshold calibration requires normal and defect calibration rows")
    candidates = sorted({floor, *(max(floor, row.max_confidence) for row in calibration)})
    fits: list[YoloThresholdFit] = []
    for threshold in candidates:
        true_positives = sum(row.label == "defect" and row.max_confidence >= threshold for row in calibration)
        false_positives = sum(row.label == "normal" and row.max_confidence >= threshold for row in calibration)
        false_negatives = sum(row.label == "defect" and row.max_confidence < threshold for row in calibration)
        denominator = 2 * true_positives + false_positives + false_negatives
        f1 = 0.0 if denominator == 0 else 2 * true_positives / denominator
        fits.append(
            YoloThresholdFit(
                threshold=threshold,
                f1=f1,
                true_positives=true_positives,
                false_positives=false_positives,
                false_negatives=false_negatives,
            )
        )
    return max(fits, key=lambda fit: (fit.f1, fit.threshold))


@dataclass(frozen=True, slots=True)
class YoloCalibrationSample:
    sample_id: str
    view_id: ViewId
    image_path: Path
    split: str
    label: str

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must be non-empty")
        object.__setattr__(self, "view_id", _view_id(self.view_id))
        resolved = Path(self.image_path).expanduser().resolve()
        if not resolved.is_file():
            raise ValueError(f"YOLO calibration image does not exist: {resolved}")
        object.__setattr__(self, "image_path", resolved)
        if self.split not in {"calibration", "final_test"}:
            raise ValueError("YOLO calibration sample split must be calibration or final_test")
        if self.label not in {"normal", "defect"}:
            raise ValueError("YOLO calibration sample label must be normal or defect")


@dataclass(frozen=True, slots=True)
class YoloTrainingRun:
    output_dir: Path
    best_checkpoint: Path
    final_threshold: float
    metrics_path: Path


def _validate_data_yaml(path: Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"YOLO data.yaml does not exist: {resolved}")
    try:
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"invalid YOLO data.yaml: {resolved}") from error
    names = payload.get("names") if isinstance(payload, dict) else None
    if names not in ({0: "defect"}, {"0": "defect"}, ["defect"]):
        raise ValueError("YOLO data.yaml must define exactly one class named 'defect'")
    return resolved


def _ultralytics_version() -> str:
    try:
        return package_version("ultralytics")
    except PackageNotFoundError as error:  # pragma: no cover - guarded by the optional dependency
        raise RuntimeError("cannot determine installed Ultralytics version") from error


def _training_best_path(trainer: Any) -> Path:
    trainer_state = getattr(trainer, "trainer", None)
    best = getattr(trainer_state, "best", None)
    if best is None:
        raise ValueError("YOLO trainer did not expose trainer.best")
    return Path(best).expanduser().resolve()


def _plain_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    if hasattr(value, "__dict__"):
        return {str(key): _plain_value(item) for key, item in vars(value).items()}
    return str(value)


def _actual_training_args(trainer: Any) -> dict[str, Any]:
    trainer_state = getattr(trainer, "trainer", None)
    save_dir = Path(getattr(trainer_state, "save_dir", ""))
    args_path = save_dir / "args.yaml"
    if args_path.is_file():
        try:
            payload = yaml.safe_load(args_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(f"cannot read actual Ultralytics args.yaml: {args_path}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"actual Ultralytics args.yaml must contain a mapping: {args_path}")
        return {str(key): _plain_value(value) for key, value in payload.items()}
    args = getattr(trainer_state, "args", None)
    payload = _plain_value(args)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("YOLO trainer did not expose actual training args")
    return payload


def _atomic_publish_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename a staged directory without ever replacing a destination."""
    if os.name == "posix":
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is not None:
            renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
            renameat2.restype = ctypes.c_int
            result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
            if result == 0:
                return
            error_number = ctypes.get_errno()
            if error_number == errno.EEXIST:
                raise FileExistsError(f"refuse to overwrite existing YOLO training version: {destination}")
            if error_number not in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
                raise OSError(error_number, os.strerror(error_number), destination)
    reservation = destination.parent / f".{destination.name}.publish-reservation"
    try:
        reservation.mkdir()
    except FileExistsError as error:
        raise FileExistsError(f"another publisher is reserving YOLO version: {destination}") from error
    try:
        if destination.exists():
            raise FileExistsError(f"refuse to overwrite existing YOLO training version: {destination}")
        source.rename(destination)
    finally:
        reservation.rmdir()


def _validate_staged_training_run(publish_dir: Path, expected_rows: int) -> None:
    required = {
        "best.pt",
        "args.yaml",
        "calibration_predictions.csv",
        "metrics.json",
    }
    if {path.name for path in publish_dir.iterdir()} != required:
        raise ValueError("staged YOLO run does not contain the exact required artifacts")
    args = yaml.safe_load((publish_dir / "args.yaml").read_text(encoding="utf-8"))
    if not isinstance(args, dict) or not args.get("ultralytics_version"):
        raise ValueError("staged YOLO args.yaml lacks Ultralytics version metadata")
    metrics = json.loads((publish_dir / "metrics.json").read_text(encoding="utf-8"))
    if metrics.get("fit_split") != "calibration" or metrics.get("final_test_used") is not False:
        raise ValueError("staged YOLO metrics violate calibration-only fitting")
    with (publish_dir / "calibration_predictions.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected_rows or any(row["split"] != "calibration" for row in rows):
        raise ValueError("staged YOLO predictions violate calibration-only fitting")


def train_versioned_yolo(
    *,
    data_yaml: Path,
    base_checkpoint: Path,
    output_root: Path,
    version: str,
    calibration_samples: Sequence[YoloCalibrationSample],
    part_rois: Mapping[ViewId, tuple[int, int, int, int]],
    candidate_conf: float = 0.01,
    imgsz: int = 1280,
    model_factory: ModelFactory = _default_model_factory,
) -> YoloTrainingRun:
    """Train and publish one immutable version without editing the active profile."""
    if not isinstance(version, str) or _VERSION_ID.fullmatch(version) is None:
        raise ValueError("version must be a path-safe identifier")
    if isinstance(imgsz, bool) or not isinstance(imgsz, int) or imgsz <= 0:
        raise ValueError("imgsz must be a positive integer")
    floor = _probability(candidate_conf, "candidate_conf")
    destination = (Path(output_root).expanduser().resolve() / version).resolve()
    if destination.exists():
        raise FileExistsError(f"refuse to overwrite existing YOLO training version: {destination}")
    data_path = _validate_data_yaml(data_yaml)
    base_path = Path(base_checkpoint).expanduser().resolve()
    if not base_path.is_file():
        raise ValueError(f"base YOLO checkpoint does not exist: {base_path}")
    rois = _validate_rois(part_rois, require_all=False)
    calibration = [sample for sample in calibration_samples if sample.split == "calibration"]
    if any(sample.view_id not in rois for sample in calibration):
        raise ValueError("every YOLO calibration sample requires its configured part ROI")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{version}.", dir=destination.parent))
    publish_dir = temporary_root / "publish"
    publish_dir.mkdir()
    try:
        trainer = model_factory(base_path)
        trainer.train(
            data=str(data_path),
            imgsz=imgsz,
            seed=42,
            project=str(temporary_root / "ultralytics"),
            name="fit",
            exist_ok=False,
        )
        trained_best = _training_best_path(trainer)
        if not trained_best.is_file():
            raise ValueError(f"YOLO training did not produce best.pt: {trained_best}")
        training_args = _actual_training_args(trainer)
        best_checkpoint = publish_dir / "best.pt"
        shutil.copy2(trained_best, best_checkpoint)
        predictor = model_factory(best_checkpoint)
        _class_names(predictor)
        prediction_rows: list[dict[str, Any]] = []
        scores: list[YoloScoreRow] = []
        for sample in sorted(calibration, key=lambda item: item.sample_id):
            image = cv2.imread(str(sample.image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"cannot decode YOLO calibration image: {sample.image_path}")
            candidates = _infer_candidates(
                predictor,
                image,
                roi=rois[sample.view_id],
                candidate_conf=floor,
                imgsz=imgsz,
            )
            max_confidence = max((box.confidence for box in candidates), default=0.0)
            scores.append(YoloScoreRow(sample.sample_id, "calibration", sample.label, max_confidence))
            prediction_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "view_id": sample.view_id.value,
                    "split": "calibration",
                    "label": sample.label,
                    "max_confidence": max_confidence,
                    "candidates_json": json.dumps(
                        [
                            {
                                "x1": box.x1,
                                "y1": box.y1,
                                "x2": box.x2,
                                "y2": box.y2,
                                "confidence": box.confidence,
                                "class_name": box.class_name,
                            }
                            for box in candidates
                        ],
                        sort_keys=True,
                    ),
                }
            )
        fit = fit_yolo_final_threshold(scores, candidate_conf=floor)
        with (publish_dir / "calibration_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("sample_id", "view_id", "split", "label", "max_confidence", "candidates_json"),
            )
            writer.writeheader()
            writer.writerows(prediction_rows)
        args = dict(training_args)
        args.update(
            {
                "base_checkpoint": str(base_path),
                "candidate_conf": floor,
                "data": str(data_path),
                "final_threshold": fit.threshold,
                "imgsz": imgsz,
                "seed": 42,
                "ultralytics_version": _ultralytics_version(),
                "version": version,
            }
        )
        (publish_dir / "args.yaml").write_text(
            yaml.safe_dump(args, allow_unicode=True, sort_keys=True),
            encoding="utf-8",
        )
        metrics = {
            "schema_version": 1,
            "fit_split": "calibration",
            "final_test_used": False,
            "candidate_conf": floor,
            "final_threshold": fit.threshold,
            "image_level_f1": fit.f1,
            "true_positives": fit.true_positives,
            "false_positives": fit.false_positives,
            "false_negatives": fit.false_negatives,
            "checkpoint_sha256": _sha256(best_checkpoint),
        }
        (publish_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _validate_staged_training_run(publish_dir, len(calibration))
        _atomic_publish_noreplace(publish_dir, destination)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    shutil.rmtree(temporary_root, ignore_errors=True)
    return YoloTrainingRun(
        output_dir=destination,
        best_checkpoint=destination / "best.pt",
        final_threshold=fit.threshold,
        metrics_path=destination / "metrics.json",
    )
