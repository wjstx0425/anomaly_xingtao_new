# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Persistent six-view PatchCore and YOLO inference runtime for ZS32."""

# Runtime adapters intentionally translate heterogeneous third-party exceptions into
# fail-closed evidence while retaining the original diagnostic text.
# ruff: noqa: EM102, PLW0717, TRY003, TRY301

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast
from uuid import uuid4

import cv2
import numpy as np
from capture_data.fusion_engine import BranchPrediction, write_branch_predictions_csv
from capture_data.zs32_inspection_orchestrator import CANONICAL_VIEWS, InspectionRequest
from capture_data.zs32_patchcore_roi_dataset import load_patchcore_roi_config
from capture_data.zs32_view_roi_dataset import load_roi_config

from zs32_inspection.dashboard.contracts import MODELED_VIEWS

REPO_ROOT = Path(__file__).resolve().parents[1]
SHA256_LENGTH = 64


def sha256_file(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PatchcoreSpec:
    """Immutable checkpoint identity for one canonical view."""

    view: str
    checkpoint: Path
    checkpoint_sha256: str
    model_version: str


@dataclass(frozen=True, slots=True)
class YoloSpec:
    """Immutable YOLO runtime and checkpoint settings."""

    weights: Path
    weights_sha256: str
    model_version: str
    imgsz: int
    candidate_conf: float
    iou: float
    max_det: int
    class_map: Mapping[int, str]


@dataclass(frozen=True, slots=True)
class RuntimeVersions:
    """Shared deployment version identifiers."""

    threshold: str
    patchcore_roi: str
    yolo_roi: str
    template: str


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Validated model, ROI, and provenance contract for online inference."""

    path: Path
    product: str
    profile: str
    supported_hands: tuple[str, ...]
    patchcore_roi_config: Path
    yolo_roi_config: Path
    image_width: int
    image_height: int
    patchcore_rois: Mapping[str, Mapping[str, tuple[int, int, int, int]]]
    yolo_rois: Mapping[str, tuple[int, int, int, int]]
    patchcore: Mapping[str, PatchcoreSpec]
    yolo: YoloSpec
    versions: RuntimeVersions


@dataclass(frozen=True, slots=True)
class PatchcoreArtifacts:
    """Lossless PatchCore map and display-only binary-mask artifacts."""

    raw_anomaly_map_path: Path
    mask_path: Path
    mask_source: Literal["pred_mask", "diagnostic_anomaly_map"]
    diagnostic_mask_threshold: float | None
    raw_anomaly_map_shape: tuple[int, int]
    mask_shape: tuple[int, int]


@dataclass(frozen=True, slots=True)
class ModelEvidence:
    """Continuous model output plus a human-readable evidence artifact."""

    score: float
    evidence_path: Path
    detections: tuple[dict[str, Any], ...] | None = None
    patchcore_artifacts: PatchcoreArtifacts | None = None


@dataclass(frozen=True, slots=True)
class RuntimeResult:
    """Fail-closed result from one six-view model execution."""

    machine_status: str
    inspection_complete: bool
    output_dir: Path
    patchcore_csv: Path
    yolo_csv: Path
    calibration_csv: Path | None
    errors: tuple[str, ...]
    missing_required_evidence: tuple[str, ...]


class PatchcoreBackend(Protocol):
    """Interface used by the runtime for per-view PatchCore inference."""

    def predict(
        self,
        view: str,
        crop_path: Path,
        evidence_path: Path,
        *,
        diagnostic_mask_threshold: float = 0.65,
    ) -> ModelEvidence:
        """Predict one already-cropped view."""


class YoloBackend(Protocol):
    """Interface used by the runtime for one six-image YOLO batch."""

    def predict(self, crops: dict[str, Path], evidence_dir: Path) -> dict[str, ModelEvidence]:
        """Predict all canonical views in one batch."""


def _require_mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        msg = f"runtime config {field} must be an object"
        raise TypeError(msg)
    return cast("dict[str, Any]", value)


def _required_text(mapping: Mapping[str, Any], field: str) -> str:
    value = mapping.get(field)
    text = "" if value is None else str(value).strip()
    if not text:
        msg = f"runtime config {field} must be non-empty"
        raise ValueError(msg)
    return text


def _resolve_path(value: object, *, field: str) -> Path:
    text = "" if value is None else str(value).strip()
    if not text:
        msg = f"runtime config {field} must be a file path"
        raise ValueError(msg)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _verified_asset(path: Path, expected_sha256: object, *, field: str) -> str:
    expected = str(expected_sha256).strip().lower()
    if len(expected) != SHA256_LENGTH or any(character not in "0123456789abcdef" for character in expected):
        msg = f"runtime config {field} must contain a 64-character SHA-256"
        raise ValueError(msg)
    if not path.is_file():
        msg = f"runtime asset does not exist: {path}"
        raise FileNotFoundError(msg)
    actual = sha256_file(path)
    if actual != expected:
        msg = f"runtime asset SHA-256 mismatch for {path}: expected {expected}, found {actual}"
        raise ValueError(msg)
    return actual


def load_runtime_config(path: Path) -> RuntimeConfig:
    """Load and verify all model assets and both ROI configurations."""
    path = path.expanduser().resolve()
    payload = _require_mapping(json.loads(path.read_text(encoding="utf-8")), "root")
    if payload.get("schema_version") != 1:
        msg = "runtime config schema_version must be 1"
        raise ValueError(msg)
    supported_hands = tuple(str(item).strip() for item in payload.get("supported_hands", []))
    if supported_hands != ("right",):
        msg = "the checked-in ZS32 weights currently support exactly the right hand"
        raise ValueError(msg)
    patchcore_roi_path = _resolve_path(
        payload.get("patchcore_roi_config"),
        field="patchcore_roi_config",
    )
    yolo_roi_path = _resolve_path(payload.get("yolo_roi_config"), field="yolo_roi_config")
    pc_width, pc_height, patchcore_rois, _ = load_patchcore_roi_config(patchcore_roi_path)
    yolo_width, yolo_height, yolo_rois, _ = load_roi_config(yolo_roi_path)
    if (pc_width, pc_height) != (yolo_width, yolo_height):
        msg = "PatchCore and YOLO ROI configs must declare the same source dimensions"
        raise ValueError(msg)

    patchcore_payload = _require_mapping(payload.get("patchcore"), "patchcore")
    if set(patchcore_payload) != set(CANONICAL_VIEWS) or len(patchcore_payload) != len(CANONICAL_VIEWS):
        msg = f"runtime config patchcore must contain exactly the six views: {CANONICAL_VIEWS}"
        raise ValueError(msg)
    patchcore: dict[str, PatchcoreSpec] = {}
    checkpoints: set[Path] = set()
    for view in CANONICAL_VIEWS:
        record = _require_mapping(patchcore_payload[view], f"patchcore.{view}")
        checkpoint = _resolve_path(record.get("checkpoint"), field=f"patchcore.{view}.checkpoint")
        _verified_asset(checkpoint, record.get("checkpoint_sha256"), field=f"patchcore.{view}.checkpoint_sha256")
        if checkpoint in checkpoints:
            msg = f"PatchCore views must use six distinct checkpoints; duplicate: {checkpoint}"
            raise ValueError(msg)
        checkpoints.add(checkpoint)
        patchcore[view] = PatchcoreSpec(
            view=view,
            checkpoint=checkpoint,
            checkpoint_sha256=str(record["checkpoint_sha256"]).lower(),
            model_version=_required_text(record, "model_version"),
        )

    yolo_payload = _require_mapping(payload.get("yolo"), "yolo")
    weights = _resolve_path(yolo_payload.get("weights"), field="yolo.weights")
    _verified_asset(weights, yolo_payload.get("weights_sha256"), field="yolo.weights_sha256")
    yolo = YoloSpec(
        weights=weights,
        weights_sha256=str(yolo_payload["weights_sha256"]).lower(),
        model_version=_required_text(yolo_payload, "model_version"),
        imgsz=int(yolo_payload.get("imgsz", 640)),
        candidate_conf=float(yolo_payload.get("candidate_conf", 0.001)),
        iou=float(yolo_payload.get("iou", 0.7)),
        max_det=int(yolo_payload.get("max_det", 300)),
        class_map={
            int(class_id): str(class_name).strip()
            for class_id, class_name in _require_mapping(yolo_payload.get("class_map"), "yolo.class_map").items()
        },
    )
    if yolo.imgsz <= 0 or yolo.max_det <= 0:
        msg = "YOLO imgsz and max_det must be positive"
        raise ValueError(msg)
    if not (0 < yolo.candidate_conf < 1) or not (0 < yolo.iou <= 1):
        msg = "YOLO candidate_conf and iou must be finite probabilities"
        raise ValueError(msg)
    if not yolo.class_map or any(not class_name for class_name in yolo.class_map.values()):
        msg = "YOLO class_map must contain non-empty deployment semantics"
        raise ValueError(msg)

    versions_payload = _require_mapping(payload.get("versions"), "versions")
    versions = RuntimeVersions(
        threshold=_required_text(versions_payload, "threshold"),
        patchcore_roi=_required_text(versions_payload, "patchcore_roi"),
        yolo_roi=_required_text(versions_payload, "yolo_roi"),
        template=_required_text(versions_payload, "template"),
    )
    return RuntimeConfig(
        path=path,
        product=_required_text(payload, "product"),
        profile=_required_text(payload, "profile"),
        supported_hands=supported_hands,
        patchcore_roi_config=patchcore_roi_path,
        yolo_roi_config=yolo_roi_path,
        image_width=pc_width,
        image_height=pc_height,
        patchcore_rois=patchcore_rois,
        yolo_rois=yolo_rois,
        patchcore=patchcore,
        yolo=yolo,
        versions=versions,
    )


def _prediction_items(predictions: object) -> list[object]:
    items: list[object] = []
    if predictions is None:
        return items
    if hasattr(predictions, "image_path"):
        try:
            return list(predictions) if isinstance(predictions.image_path, list) else [predictions]
        except TypeError:
            return [predictions]
    if isinstance(predictions, list | tuple):
        for value in predictions:
            items.extend(_prediction_items(value))
    return items


def _scalar(value: object) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1)[0].item()  # type: ignore[union-attr]
    elif isinstance(value, np.ndarray):
        value = value.reshape(-1)[0].item()
    score = float(value)
    if not math.isfinite(score):
        msg = f"model emitted a non-finite score: {score}"
        raise ValueError(msg)
    return score


def _array2d(value: object) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()  # type: ignore[union-attr]
    array = np.asarray(value)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        return None
    return np.nan_to_num(array.astype(np.float32), copy=False)


def _finite_array2d(value: object, *, field: str) -> np.ndarray:
    """Convert a tensor-like value to a finite float32 two-dimensional array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()  # type: ignore[union-attr]
    array = np.asarray(value)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        msg = f"PatchCore {field} must be a 2D array"
        raise ValueError(msg)
    array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        msg = f"PatchCore {field} must contain only finite values"
        raise ValueError(msg)
    return array


def _build_patchcore_mask(
    anomaly_map: object,
    pred_mask: object | None,
    *,
    output_shape: tuple[int, int],
    diagnostic_mask_threshold: float,
) -> tuple[np.ndarray, Literal["pred_mask", "diagnostic_anomaly_map"], float | None]:
    """Build a display-only binary mask, preferring Anomalib's real prediction mask."""
    threshold = float(diagnostic_mask_threshold)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        msg = "diagnostic_mask_threshold must be finite and within [0, 1]"
        raise ValueError(msg)
    if len(output_shape) != 2 or any(not isinstance(value, int) or value <= 0 for value in output_shape):
        msg = "output_shape must contain two positive integers"
        raise ValueError(msg)

    raw = _finite_array2d(anomaly_map, field="anomaly_map")
    if pred_mask is not None:
        source: Literal["pred_mask", "diagnostic_anomaly_map"] = "pred_mask"
        source_mask = _finite_array2d(pred_mask, field="pred_mask")
        mask_threshold = None
    else:
        source = "diagnostic_anomaly_map"
        minimum, maximum = float(raw.min()), float(raw.max())
        source_mask = np.zeros_like(raw, dtype=np.float32)
        if maximum > minimum:
            source_mask = (raw - minimum) / (maximum - minimum)
        mask_threshold = threshold

    resized = cv2.resize(
        source_mask,
        (output_shape[1], output_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    cutoff = 0.5 if source == "pred_mask" else threshold
    return (resized >= cutoff).astype(np.uint8) * 255, source, mask_threshold


def _write_patchcore_overlay(crop_path: Path, anomaly_map: object, output_path: Path, score: float) -> None:
    image = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
    if image is None:
        msg = f"could not read PatchCore crop: {crop_path}"
        raise ValueError(msg)
    array = _array2d(anomaly_map)
    if array is not None:
        array = cv2.resize(array, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        minimum, maximum = float(array.min()), float(array.max())
        normalized = np.zeros_like(array, dtype=np.uint8)
        if maximum > minimum:
            normalized = np.clip((array - minimum) * 255 / (maximum - minimum), 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        image = cv2.addWeighted(image, 0.55, heatmap, 0.45, 0)
    cv2.putText(image, f"PatchCore score={score:.6f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        msg = f"failed to write PatchCore evidence: {output_path}"
        raise OSError(msg)


class AnomalibPatchcoreBackend:
    """PatchCore adapter that retains six restored models and six engines."""

    def __init__(self, specs: Mapping[str, PatchcoreSpec], *, accelerator: str = "auto", devices: int = 1) -> None:
        from anomalib.engine import Engine
        from anomalib.models import Patchcore

        self.models: dict[str, object] = {}
        self.engines: dict[str, object] = {}
        engine_root = Path(tempfile.gettempdir()) / "anomalib-zs32-runtime-engine"
        for view in CANONICAL_VIEWS:
            self.models[view] = Patchcore.load_from_checkpoint(
                specs[view].checkpoint,
                weights_only=False,
                visualizer=False,
            )
            self.engines[view] = Engine(
                accelerator=accelerator,
                devices=devices,
                default_root_dir=engine_root / view,
                logger=False,
            )

    def predict(
        self,
        view: str,
        crop_path: Path,
        evidence_path: Path,
        *,
        diagnostic_mask_threshold: float = 0.65,
    ) -> ModelEvidence:
        """Run PatchCore and persist its raw map, display mask, and overlay."""
        predictions = self.engines[view].predict(  # type: ignore[union-attr]
            model=self.models[view],
            data_path=crop_path,
            ckpt_path=None,
            return_predictions=True,
        )
        items = _prediction_items(predictions)
        if len(items) != 1:
            msg = f"PatchCore {view} returned {len(items)} prediction items, expected 1"
            raise RuntimeError(msg)
        item = items[0]
        score = _scalar(getattr(item, "pred_score", None))
        raw = _finite_array2d(getattr(item, "anomaly_map", None), field="anomaly_map")
        crop = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        if crop is None:
            msg = f"could not read PatchCore crop: {crop_path}"
            raise ValueError(msg)
        mask, mask_source, applied_threshold = _build_patchcore_mask(
            raw,
            getattr(item, "pred_mask", None),
            output_shape=crop.shape[:2],
            diagnostic_mask_threshold=diagnostic_mask_threshold,
        )
        raw_path = evidence_path.parent / "raw_maps" / f"{view}.npy"
        mask_path = evidence_path.parent / "masks" / f"{view}.png"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(raw_path, raw.astype(np.float32, copy=False), allow_pickle=False)
        if not cv2.imwrite(str(mask_path), mask):
            msg = f"failed to write PatchCore mask: {mask_path}"
            raise OSError(msg)
        _write_patchcore_overlay(crop_path, raw, evidence_path, score)
        return ModelEvidence(
            score=score,
            evidence_path=evidence_path,
            patchcore_artifacts=PatchcoreArtifacts(
                raw_anomaly_map_path=raw_path,
                mask_path=mask_path,
                mask_source=mask_source,
                diagnostic_mask_threshold=applied_threshold,
                raw_anomaly_map_shape=cast("tuple[int, int]", raw.shape),
                mask_shape=cast("tuple[int, int]", mask.shape),
            ),
        )


class UltralyticsYoloBackend:
    """YOLO adapter that retains one model and predicts six views as one batch."""

    def __init__(self, spec: YoloSpec, *, device: str | None = None) -> None:
        from ultralytics import YOLO

        self.model = YOLO(spec.weights)
        self.spec = spec
        self.device = device

    def predict(self, crops: dict[str, Path], evidence_dir: Path) -> dict[str, ModelEvidence]:
        """Run one six-view YOLO batch and write one annotated image per view."""
        sources = [str(crops[view]) for view in CANONICAL_VIEWS]
        kwargs: dict[str, Any] = {
            "source": sources,
            "imgsz": self.spec.imgsz,
            "conf": self.spec.candidate_conf,
            "iou": self.spec.iou,
            "max_det": self.spec.max_det,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        predictions = list(self.model.predict(**kwargs))
        if len(predictions) != len(CANONICAL_VIEWS):
            msg = f"YOLO returned {len(predictions)} results, expected {len(CANONICAL_VIEWS)}"
            raise RuntimeError(msg)
        evidence_dir.mkdir(parents=True, exist_ok=True)
        output: dict[str, ModelEvidence] = {}
        for view, result in zip(CANONICAL_VIEWS, predictions, strict=True):
            detections: list[dict[str, Any]] = []
            boxes = getattr(result, "boxes", None)
            if boxes is not None:
                for box in boxes:
                    class_id = int(_scalar(box.cls))
                    confidence = _scalar(box.conf)
                    xyxy = [float(value) for value in box.xyxy.detach().cpu().reshape(-1).tolist()]
                    if len(xyxy) != 4 or not all(math.isfinite(value) for value in xyxy):
                        msg = f"YOLO emitted malformed xyxy for {view}: {xyxy}"
                        raise ValueError(msg)
                    x1, y1, x2, y2 = xyxy
                    if x2 <= x1 or y2 <= y1:
                        msg = f"YOLO emitted a non-positive box for {view}: {xyxy}"
                        raise ValueError(msg)
                    names = getattr(result, "names", {})
                    checkpoint_class_name = (
                        str(names.get(class_id, class_id)) if isinstance(names, Mapping) else str(class_id)
                    )
                    class_name = self.spec.class_map.get(class_id)
                    if class_name is None:
                        msg = f"YOLO class {class_id} has no deployment semantic mapping"
                        raise ValueError(msg)
                    detections.append(
                        {
                            "class": class_id,
                            "class_name": class_name,
                            "checkpoint_class_name": checkpoint_class_name,
                            "confidence": confidence,
                            "xyxy": xyxy,
                            "area": (x2 - x1) * (y2 - y1),
                        },
                    )
            evidence_path = evidence_dir / f"{view}.png"
            plotted = result.plot()
            if not cv2.imwrite(str(evidence_path), plotted):
                msg = f"failed to write YOLO evidence: {evidence_path}"
                raise OSError(msg)
            score = max((float(item["confidence"]) for item in detections), default=0.0)
            output[view] = ModelEvidence(score=score, evidence_path=evidence_path, detections=tuple(detections))
        return output


def load_threshold_map(path: Path | None) -> dict[tuple[str, str, str, str, str], tuple[float, float]]:
    """Read dual thresholds for evidence annotation; Stage 18 performs authoritative validation."""
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("thresholds")
    if not isinstance(records, list):
        msg = "threshold artifact thresholds must be a list"
        raise TypeError(msg)
    output: dict[tuple[str, str, str, str, str], tuple[float, float]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            msg = "threshold artifact records must be objects"
            raise TypeError(msg)
        key = tuple(str(record[field]) for field in ("hand", "view", "branch", "model_version", "roi_version"))
        low, high = float(record["low_threshold"]), float(record["high_threshold"])
        if len(key) != 5 or not math.isfinite(low) or not math.isfinite(high) or low > high:
            msg = f"invalid dual threshold record: {key}"
            raise ValueError(msg)
        if key in output:
            msg = f"duplicate threshold record: {key}"
            raise ValueError(msg)
        output[cast("tuple[str, str, str, str, str]", key)] = (low, high)
    return output


class ZS32ModelRuntime:
    """Run both model families while preserving continuous, hashed evidence."""

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        patchcore_backend: PatchcoreBackend | None = None,
        yolo_backend: YoloBackend | None = None,
        accelerator: str = "auto",
        devices: int = 1,
        yolo_device: str | None = None,
    ) -> None:
        self.config = config
        self._patchcore_backend = patchcore_backend
        self._yolo_backend = yolo_backend
        self.accelerator = accelerator
        self.devices = devices
        self.yolo_device = yolo_device

    def _ensure_backends(self) -> tuple[PatchcoreBackend, YoloBackend]:
        if self._patchcore_backend is None:
            self._patchcore_backend = AnomalibPatchcoreBackend(
                self.config.patchcore,
                accelerator=self.accelerator,
                devices=self.devices,
            )
        if self._yolo_backend is None:
            self._yolo_backend = UltralyticsYoloBackend(self.config.yolo, device=self.yolo_device)
        return self._patchcore_backend, self._yolo_backend

    def _validate_request(self, request: InspectionRequest, output_dir: Path) -> dict[str, np.ndarray]:
        if output_dir.exists():
            msg = f"output directory already exists: {output_dir}"
            raise FileExistsError(msg)
        if request.hand not in self.config.supported_hands:
            msg = f"unsupported hand {request.hand!r}; available weights: {self.config.supported_hands}"
            raise ValueError(msg)
        for field in (request.part_id, request.capture_session, request.group_id):
            if not isinstance(field, str) or not field.strip():
                msg = "part_id, capture_session, and group_id must be non-empty"
                raise ValueError(msg)
        if set(request.images) != set(CANONICAL_VIEWS) or len(request.images) != len(CANONICAL_VIEWS):
            msg = f"request must contain exactly the six canonical views: {CANONICAL_VIEWS}"
            raise ValueError(msg)
        paths = [Path(request.images[view]).expanduser().resolve() for view in CANONICAL_VIEWS]
        if len(set(paths)) != len(paths):
            msg = "six-view request must use six distinct source image paths"
            raise ValueError(msg)
        images: dict[str, np.ndarray] = {}
        for view, path in zip(CANONICAL_VIEWS, paths, strict=True):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                msg = f"could not read source image for {view}: {path}"
                raise ValueError(msg)
            if (image.shape[1], image.shape[0]) != (self.config.image_width, self.config.image_height):
                msg = (
                    f"source image dimensions for {view} must be "
                    f"{self.config.image_width}x{self.config.image_height}, found {image.shape[1]}x{image.shape[0]}"
                )
                raise ValueError(msg)
            images[view] = image
        return images

    @staticmethod
    def _thresholds_for(
        *,
        hand: str,
        view: str,
        branch: str,
        model_version: str,
        roi_version: str,
        threshold_map: Mapping[tuple[str, str, str, str, str], tuple[float, float]],
    ) -> tuple[float, float] | None:
        return threshold_map.get((hand, view, branch, model_version, roi_version))

    def _prediction(
        self,
        request: InspectionRequest,
        *,
        view: str,
        branch: str,
        evidence: ModelEvidence | None,
        evidence_path: Path,
        model_version: str,
        roi_version: str,
        threshold_map: Mapping[tuple[str, str, str, str, str], tuple[float, float]],
        reason: str | None = None,
    ) -> BranchPrediction:
        source_path = Path(request.images[view]).expanduser().resolve()
        thresholds = self._thresholds_for(
            hand=request.hand,
            view=view,
            branch=branch,
            model_version=model_version,
            roi_version=roi_version,
            threshold_map=threshold_map,
        )
        score = None if evidence is None else float(evidence.score)
        if score is not None and not math.isfinite(score):
            msg = f"non-finite {branch} score for {view}: {score}"
            raise ValueError(msg)
        low, high = thresholds if thresholds is not None else (None, None)
        return BranchPrediction(
            part_id=request.part_id,
            side="zs32",
            view=view,
            slot_id=None,
            branch=branch,
            pred_label=int(score is not None and high is not None and score >= high),
            score=score,
            threshold=high,
            defect_type="defect" if score is not None and high is not None and score >= high else None,
            reason=reason
            if reason is not None
            else (None if thresholds is not None else "locked dual thresholds unavailable"),
            source_path=str(source_path),
            status=None if thresholds is not None and evidence is not None else "ERROR",
            evidence_path=str(evidence_path.resolve()),
            evidence_type="anomaly_heatmap" if branch.startswith("anomaly_") else "object_detection_overlay",
            low_threshold=low,
            high_threshold=high,
            evidence_level=None,
            model_version=model_version,
            threshold_version=self.config.versions.threshold,
            roi_version=roi_version,
            template_version=self.config.versions.template,
            hand=request.hand,
            product=self.config.product,
            profile=self.config.profile,
            source_hash=sha256_file(source_path),
            evidence_hash=sha256_file(evidence_path),
            manifest_identity=f"{request.part_id}:{request.hand}:{view}",
            capture_session=request.capture_session,
            group_id=request.group_id,
            detections=None if branch.startswith("anomaly_") else (() if evidence is None else evidence.detections),
        )

    @staticmethod
    def _write_error_evidence(path: Path, *, branch: str, view: str, error: Exception) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"branch": branch, "view": view, "error_type": type(error).__name__, "error": str(error)},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def run(  # noqa: C901
        self,
        request: InspectionRequest,
        output_dir: Path,
        *,
        threshold_artifact: Path | None = None,
        gt_label: int | None = None,
        split: str | None = None,
        diagnostic_mask_threshold: float = 0.65,
    ) -> RuntimeResult:
        """Run all models, publish immutable evidence, and remain REVIEW until strict fusion."""
        output_dir = output_dir.expanduser().resolve()
        images = self._validate_request(request, output_dir)
        if (gt_label is None) != (split is None):
            msg = "gt_label and split must be provided together"
            raise ValueError(msg)
        if gt_label not in {None, 0, 1}:
            msg = "gt_label must be 0, 1, or omitted"
            raise ValueError(msg)
        if not math.isfinite(diagnostic_mask_threshold) or not 0 <= diagnostic_mask_threshold <= 1:
            msg = "diagnostic_mask_threshold must be finite and within [0, 1]"
            raise ValueError(msg)
        threshold_map = load_threshold_map(threshold_artifact)
        staging = output_dir.parent / f".{output_dir.name}.tmp-{uuid4().hex}"
        staging.mkdir(parents=True)
        final_patchcore_dir = output_dir / "evidence" / "patchcore"
        final_yolo_dir = output_dir / "evidence" / "yolo"
        errors: list[str] = []
        patchcore_rows: list[BranchPrediction] = []
        yolo_rows: list[BranchPrediction] = []
        calibration_rows: list[dict[str, Any]] = []
        view_manifest: dict[str, dict[str, Any]] = {}
        try:
            patchcore_crops: dict[str, Path] = {}
            yolo_crops: dict[str, Path] = {}
            for view in MODELED_VIEWS:
                image = images[view]
                pc_x1, pc_y1, pc_x2, pc_y2 = self.config.patchcore_rois[request.hand][view]
                yo_x1, yo_y1, yo_x2, yo_y2 = self.config.yolo_rois[view]
                pc_crop = staging / "crops" / "patchcore" / f"{view}.png"
                yo_crop = staging / "crops" / "yolo" / f"{view}.png"
                pc_crop.parent.mkdir(parents=True, exist_ok=True)
                yo_crop.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(pc_crop), image[pc_y1:pc_y2, pc_x1:pc_x2]):
                    raise OSError(f"failed to write PatchCore crop: {pc_crop}")
                if not cv2.imwrite(str(yo_crop), image[yo_y1:yo_y2, yo_x1:yo_x2]):
                    raise OSError(f"failed to write YOLO crop: {yo_crop}")
                patchcore_crops[view] = pc_crop
                yolo_crops[view] = yo_crop

            backend_load_error: Exception | None = None
            try:
                patchcore_backend, yolo_backend = self._ensure_backends()
            except Exception as error:  # noqa: BLE001 - publish a fail-closed initialization generation
                backend_load_error = error
                errors.append(f"model initialization: {type(error).__name__}: {error}")
                patchcore_backend = self._patchcore_backend
                yolo_backend = self._yolo_backend

            for view in MODELED_VIEWS:
                temp_evidence = staging / "evidence" / "patchcore" / f"{view}.png"
                final_evidence = final_patchcore_dir / f"{view}.png"
                evidence: ModelEvidence | None = None
                error_reason = None
                try:
                    if patchcore_backend is None:
                        raise backend_load_error or RuntimeError("PatchCore backend initialization failed")
                    evidence = patchcore_backend.predict(
                        view,
                        patchcore_crops[view],
                        temp_evidence,
                        diagnostic_mask_threshold=diagnostic_mask_threshold,
                    )
                    if evidence.evidence_path.resolve() != temp_evidence.resolve() or not temp_evidence.is_file():
                        msg = f"PatchCore backend did not publish the requested evidence for {view}"
                        raise RuntimeError(msg)
                    if not math.isfinite(float(evidence.score)):
                        raise ValueError(f"PatchCore emitted a non-finite score for {view}")
                    artifacts = evidence.patchcore_artifacts
                    if artifacts is None:
                        raise ValueError(f"PatchCore artifacts missing for {view}")
                    expected_raw = temp_evidence.parent / "raw_maps" / f"{view}.npy"
                    expected_mask = temp_evidence.parent / "masks" / f"{view}.png"
                    if artifacts.raw_anomaly_map_path.resolve() != expected_raw.resolve():
                        raise ValueError(f"PatchCore raw anomaly map path is invalid for {view}")
                    if artifacts.mask_path.resolve() != expected_mask.resolve():
                        raise ValueError(f"PatchCore mask path is invalid for {view}")
                    saved_raw = np.load(expected_raw, allow_pickle=False)
                    if saved_raw.dtype != np.float32 or saved_raw.ndim != 2 or not np.isfinite(saved_raw).all():
                        raise ValueError(f"PatchCore raw anomaly map is invalid for {view}")
                    if tuple(saved_raw.shape) != artifacts.raw_anomaly_map_shape:
                        raise ValueError(f"PatchCore raw anomaly map shape is invalid for {view}")
                    saved_mask = cv2.imread(str(expected_mask), cv2.IMREAD_GRAYSCALE)
                    if saved_mask is None or tuple(saved_mask.shape) != artifacts.mask_shape:
                        raise ValueError(f"PatchCore mask shape is invalid for {view}")
                    roi_x1, roi_y1, roi_x2, roi_y2 = self.config.patchcore_rois[request.hand][view]
                    expected_mask_shape = (roi_y2 - roi_y1, roi_x2 - roi_x1)
                    if artifacts.mask_shape != expected_mask_shape:
                        raise ValueError(f"PatchCore mask does not match the crop geometry for {view}")
                    if not set(np.unique(saved_mask)) <= {0, 255}:
                        raise ValueError(f"PatchCore mask must be binary for {view}")
                    if artifacts.mask_source == "pred_mask":
                        if artifacts.diagnostic_mask_threshold is not None:
                            raise ValueError(f"PatchCore pred_mask must not record a diagnostic threshold for {view}")
                    elif artifacts.mask_source == "diagnostic_anomaly_map":
                        if artifacts.diagnostic_mask_threshold != diagnostic_mask_threshold:
                            raise ValueError(f"PatchCore diagnostic mask threshold is invalid for {view}")
                    else:
                        raise ValueError(f"PatchCore mask source is invalid for {view}")
                    final_raw = final_patchcore_dir / "raw_maps" / f"{view}.npy"
                    final_mask = final_patchcore_dir / "masks" / f"{view}.png"
                    view_manifest[view] = {
                        "view": view,
                        "source_path": str(Path(request.images[view]).resolve()),
                        "model_supported": True,
                        "patchcore": {
                            "score": float(evidence.score),
                            "status": "available",
                            "evidence_path": str(final_evidence.resolve()),
                            "raw_anomaly_map_path": str(final_raw.resolve()),
                            "mask_path": str(final_mask.resolve()),
                            "mask_source": artifacts.mask_source,
                            "diagnostic_mask_threshold": artifacts.diagnostic_mask_threshold,
                            "display_only": True,
                            "roi_xyxy": list(self.config.patchcore_rois[request.hand][view]),
                            "raw_anomaly_map_shape": list(artifacts.raw_anomaly_map_shape),
                            "mask_shape": list(artifacts.mask_shape),
                        },
                    }
                except Exception as error:  # noqa: BLE001 - retain other view evidence
                    errors.append(f"anomaly_{view}: {type(error).__name__}: {error}")
                    error_reason = errors[-1]
                    diagnostic_score = None if evidence is None else float(evidence.score)
                    evidence = None
                    temp_evidence = temp_evidence.with_suffix(".error.json")
                    final_evidence = final_evidence.with_suffix(".error.json")
                    self._write_error_evidence(temp_evidence, branch=f"anomaly_{view}", view=view, error=error)
                    view_manifest[view] = {
                        "view": view,
                        "source_path": str(Path(request.images[view]).resolve()),
                        "model_supported": True,
                        "patchcore": {
                            "score": diagnostic_score,
                            "status": "error",
                            "reason": error_reason,
                            "evidence_path": str(final_evidence.resolve()),
                            "display_only": True,
                            "roi_xyxy": list(self.config.patchcore_rois[request.hand][view]),
                        },
                    }
                row = self._prediction(
                    request,
                    view=view,
                    branch=f"anomaly_{view}",
                    evidence=evidence,
                    evidence_path=temp_evidence,
                    model_version=self.config.patchcore[view].model_version,
                    roi_version=self.config.versions.patchcore_roi,
                    threshold_map=threshold_map,
                    reason=error_reason,
                )
                patchcore_rows.append(dataclass_replace_paths(row, evidence_path=final_evidence))

            try:
                if yolo_backend is None:
                    raise backend_load_error or RuntimeError("YOLO backend initialization failed")
                yolo_results = yolo_backend.predict(yolo_crops, staging / "evidence" / "yolo")
                if set(yolo_results) != set(CANONICAL_VIEWS):
                    msg = "YOLO backend must return exactly one result per canonical view"
                    raise RuntimeError(msg)
            except Exception as error:  # noqa: BLE001 - publish fail-closed per-view diagnostics
                yolo_results = {}
                errors.append(f"yolo batch: {type(error).__name__}: {error}")
                yolo_batch_error: Exception | None = error
            else:
                yolo_batch_error = None
            for view in CANONICAL_VIEWS:
                temp_evidence = staging / "evidence" / "yolo" / f"{view}.png"
                final_evidence = final_yolo_dir / f"{view}.png"
                evidence = yolo_results.get(view)
                error_reason = None
                try:
                    if evidence is None:
                        raise yolo_batch_error or RuntimeError(f"YOLO result missing for {view}")
                    if evidence.detections is None:
                        raise ValueError(f"YOLO detections must be an explicit list for {view}")
                    confidences = [float(item["confidence"]) for item in evidence.detections]
                    expected_score = max(confidences, default=0.0)
                    if not math.isclose(float(evidence.score), expected_score, rel_tol=0, abs_tol=1e-12):
                        raise ValueError(f"YOLO score must equal max detection confidence for {view}")
                    if evidence.evidence_path.resolve() != temp_evidence.resolve() or not temp_evidence.is_file():
                        raise RuntimeError(f"YOLO backend did not publish the requested evidence for {view}")
                except Exception as error:  # noqa: BLE001 - retain other model evidence
                    errors.append(f"yolo/{view}: {type(error).__name__}: {error}")
                    error_reason = errors[-1]
                    evidence = None
                    temp_evidence = temp_evidence.with_suffix(".error.json")
                    final_evidence = final_evidence.with_suffix(".error.json")
                    self._write_error_evidence(temp_evidence, branch="yolo", view=view, error=error)
                row = self._prediction(
                    request,
                    view=view,
                    branch="yolo",
                    evidence=evidence,
                    evidence_path=temp_evidence,
                    model_version=self.config.yolo.model_version,
                    roi_version=self.config.versions.yolo_roi,
                    threshold_map=threshold_map,
                    reason=error_reason,
                )
                yolo_rows.append(dataclass_replace_paths(row, evidence_path=final_evidence))

            write_branch_predictions_csv(patchcore_rows, staging / "patchcore.csv")
            write_branch_predictions_csv(yolo_rows, staging / "yolo.csv")
            if gt_label is not None and split is not None:
                calibration_rows.extend(
                    ({
                        "part_id": row.part_id,
                        "hand": row.hand,
                        "view": row.view,
                        "branch": row.branch,
                        "raw_score": row.score,
                        "gt_label": gt_label,
                        "split": split,
                        "model_version": row.model_version,
                        "roi_version": row.roi_version,
                    })
                    for row in [*patchcore_rows, *yolo_rows]
                    if row.score is not None
                )
                with (staging / "calibration_rows.csv").open("w", encoding="utf-8", newline="") as file:
                    fieldnames = [
                        "part_id",
                        "hand",
                        "view",
                        "branch",
                        "raw_score",
                        "gt_label",
                        "split",
                        "model_version",
                        "roi_version",
                    ]
                    writer = csv.DictWriter(file, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(calibration_rows)

            missing = [
                "locked_threshold_artifact" if not threshold_map else "strict_fusion_not_run",
                "template_match",
                "quality_gate",
                "registration",
                "geometry",
            ]
            if errors:
                missing.append("complete_model_evidence")
            manifest = {
                "schema_version": "1.0",
                "product": self.config.product,
                "profile": self.config.profile,
                "part_id": request.part_id,
                "capture_session": request.capture_session,
                "group_id": request.group_id,
                "hand": request.hand,
                "machine_status": "REVIEW",
                "inspection_complete": False,
                "source_images": {view: str(Path(request.images[view]).resolve()) for view in CANONICAL_VIEWS},
                "views": view_manifest,
                "runtime_config": str(self.config.path),
                "runtime_config_sha256": sha256_file(self.config.path),
                "patchcore_roi_config": str(self.config.patchcore_roi_config),
                "patchcore_roi_sha256": sha256_file(self.config.patchcore_roi_config),
                "yolo_roi_config": str(self.config.yolo_roi_config),
                "yolo_roi_sha256": sha256_file(self.config.yolo_roi_config),
                "patchcore_models": {view: asdict(self.config.patchcore[view]) for view in CANONICAL_VIEWS},
                "yolo_model": asdict(self.config.yolo),
                "errors": errors,
                "missing_required_evidence": missing,
                "note": "Continuous evidence only; OK is forbidden until strict Stage 18 fusion succeeds.",
            }
            (staging / "runtime_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            (staging / "runtime_summary.json").write_text(
                json.dumps(
                    {
                        "machine_status": "REVIEW",
                        "inspection_complete": False,
                        "errors": errors,
                        "missing_required_evidence": missing,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            staging.replace(output_dir)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

        return RuntimeResult(
            machine_status="REVIEW",
            inspection_complete=False,
            output_dir=output_dir,
            patchcore_csv=output_dir / "patchcore.csv",
            yolo_csv=output_dir / "yolo.csv",
            calibration_csv=(output_dir / "calibration_rows.csv") if calibration_rows else None,
            errors=tuple(errors),
            missing_required_evidence=tuple(missing),
        )


def dataclass_replace_paths(prediction: BranchPrediction, *, evidence_path: Path) -> BranchPrediction:
    """Replace a staging evidence path with its immutable final output path."""
    from dataclasses import replace

    return replace(prediction, evidence_path=str(evidence_path.resolve()))
