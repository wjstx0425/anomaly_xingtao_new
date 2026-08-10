"""Six-view PatchCore runtime, calibration, and training utilities."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import cv2
import numpy as np

from bmw_inspection.lab.contracts import BranchEvidence, BranchName, BranchStatus, ViewId


@dataclass(frozen=True, slots=True)
class RawPatchCorePrediction:
    """Unmodified scalar score and anomaly map produced by PatchCore."""

    score: float
    anomaly_map: np.ndarray


@dataclass(frozen=True, slots=True)
class PatchCoreDecision:
    """Validated runtime prediction, including raw and display-only maps."""

    view_id: ViewId
    score: float
    threshold: float | None
    raw_anomaly_map: np.ndarray
    display_heatmap: np.ndarray
    checkpoint_path: Path
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class ThresholdFit:
    """Result of fitting an image-level threshold on calibration data."""

    threshold: float | None
    f1: float | None
    status: BranchStatus


@dataclass(frozen=True, slots=True)
class EquivalenceReceipt:
    """Evidence that training and runtime preserve the same raw prediction."""

    checkpoint_sha256: str
    score_equal: bool
    map_equal: bool


@dataclass(frozen=True, slots=True)
class PatchCoreTrainingProfile:
    """Frozen, serializable defaults for the six per-view PatchCore models."""

    backbone: str = "wide_resnet50_2"
    layers: tuple[str, ...] = ("layer2", "layer3")
    image_size: tuple[int, int] = (512, 512)
    coreset_sampling_ratio: float = 0.1
    num_neighbors: int = 9
    seed: int = 42
    max_epochs: int = 1

    def __post_init__(self) -> None:
        if not self.backbone.strip():
            raise ValueError("backbone must not be empty")
        if not self.layers or any(not layer.strip() for layer in self.layers):
            raise ValueError("layers must contain non-empty layer names")
        if len(self.image_size) != 2 or any(size <= 0 for size in self.image_size):
            raise ValueError("image_size must contain two positive dimensions")
        if not 0 < self.coreset_sampling_ratio <= 1:
            raise ValueError("coreset_sampling_ratio must be in (0, 1]")
        if self.num_neighbors <= 0:
            raise ValueError("num_neighbors must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.max_epochs != 1:
            raise ValueError("BMW laboratory PatchCore training is fixed to one epoch")

    def as_dict(self) -> dict[str, Any]:
        """Return the exact JSON-safe training parameters."""
        return {
            "backbone": self.backbone,
            "layers": list(self.layers),
            "image_size": list(self.image_size),
            "coreset_sampling_ratio": self.coreset_sampling_ratio,
            "num_neighbors": self.num_neighbors,
            "seed": self.seed,
            "max_epochs": self.max_epochs,
        }


class RuntimePredictor(Protocol):
    """Minimal resident predictor interface used by the runtime backend."""

    def predict(self, image: np.ndarray) -> RawPatchCorePrediction:
        """Predict from one RGB uint8 ROI."""


class CalibrationPredictor(Protocol):
    """Minimal file predictor interface used during calibration."""

    def predict_path(self, image_path: Path) -> RawPatchCorePrediction:
        """Predict from one calibration image without changing its split."""


class PatchCoreTrainer(Protocol):
    """Injectable per-view training adapter."""

    def train(
        self,
        view_id: ViewId,
        view_root: Path,
        run_dir: Path,
        profile: PatchCoreTrainingProfile,
    ) -> Path:
        """Train one view and return its checkpoint path."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _validated_prediction(prediction: RawPatchCorePrediction) -> RawPatchCorePrediction:
    try:
        score_array = _numpy(prediction.score)
        if score_array.size != 1:
            raise ValueError
        score = float(score_array.reshape(-1)[0])
    except (TypeError, ValueError) as error:
        raise ValueError("PatchCore score must be a scalar") from error
    anomaly_map = _numpy(prediction.anomaly_map)
    while anomaly_map.ndim > 2 and anomaly_map.shape[0] == 1:
        anomaly_map = anomaly_map[0]
    anomaly_map = np.array(anomaly_map, copy=True)
    if anomaly_map.ndim != 2 or anomaly_map.size == 0:
        raise ValueError("PatchCore anomaly map must be a non-empty 2D array")
    if not math.isfinite(score):
        raise ValueError("PatchCore score must be finite")
    if not np.isfinite(anomaly_map).all():
        raise ValueError("PatchCore anomaly map must contain only finite values")
    return RawPatchCorePrediction(score=score, anomaly_map=anomaly_map)


def _display_heatmap(anomaly_map: np.ndarray) -> np.ndarray:
    display_map = anomaly_map.astype(np.float32, copy=False)
    minimum = float(display_map.min())
    peak_to_peak = float(np.ptp(display_map))
    if peak_to_peak == 0:
        normalized = np.zeros(display_map.shape, dtype=np.uint8)
    else:
        normalized = np.rint((display_map - minimum) * (255.0 / peak_to_peak)).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


def _exact_view_mapping(mapping: Mapping[ViewId, Any], name: str) -> dict[ViewId, Any]:
    if set(mapping) != set(ViewId):
        raise ValueError(f"{name} must contain exactly the six required views")
    return dict(mapping)


class PatchCoreBackend:
    """Resident six-model PatchCore runtime with view-specific ROI and threshold."""

    def __init__(
        self,
        *,
        checkpoint_paths: Mapping[ViewId, Path],
        thresholds: Mapping[ViewId, float | None],
        rois: Mapping[ViewId, tuple[int, int, int, int]],
        predictor_factory: Callable[[Path], RuntimePredictor] | None = None,
        required_for_ok: bool = True,
    ) -> None:
        checkpoints = _exact_view_mapping(checkpoint_paths, "checkpoint_paths")
        self._thresholds = _exact_view_mapping(thresholds, "thresholds")
        self._rois = _exact_view_mapping(rois, "rois")
        self._required_for_ok = required_for_ok
        self._lock = threading.Lock()
        self._decisions: dict[ViewId, PatchCoreDecision] = {}
        self._checkpoints: dict[ViewId, Path] = {}
        self._checkpoint_hashes: dict[ViewId, str] = {}
        self._predictors: dict[ViewId, RuntimePredictor] = {}
        factory = predictor_factory or _AnomalibPredictor

        for view_id in ViewId:
            threshold = self._thresholds[view_id]
            if threshold is not None and (not math.isfinite(float(threshold)) or threshold < 0):
                raise ValueError(f"threshold for {view_id.value} must be finite, non-negative, or None")
            roi = self._rois[view_id]
            if len(roi) != 4 or not all(isinstance(value, int) for value in roi):
                raise ValueError(f"ROI for {view_id.value} must contain four integers")
            checkpoint = Path(checkpoints[view_id]).expanduser().resolve()
            if not checkpoint.is_file():
                raise ValueError(f"checkpoint for {view_id.value} does not exist: {checkpoint}")
            try:
                predictor = factory(checkpoint)
            except Exception as error:
                raise RuntimeError(
                    f"failed to load PatchCore checkpoint for {view_id.value}: {error}"
                ) from error
            self._checkpoints[view_id] = checkpoint
            self._checkpoint_hashes[view_id] = _sha256(checkpoint)
            self._predictors[view_id] = predictor

    @classmethod
    def from_experiment_config(
        cls,
        config: Any,
        *,
        predictor_factory: Callable[[Path], RuntimePredictor] | None = None,
    ) -> PatchCoreBackend:
        """Build a backend from a validated ``LabExperimentConfig``."""
        checkpoints = {
            view_id: checkpoint
            for view_id, checkpoint in config.patchcore.checkpoints.items()
            if checkpoint is not None
        }
        return cls(
            checkpoint_paths=checkpoints,
            thresholds=config.patchcore.thresholds,
            rois=config.part_rois,
            predictor_factory=predictor_factory,
            required_for_ok=BranchName.PATCHCORE in config.required_for_ok,
        )

    def decision_for(self, view_id: ViewId) -> PatchCoreDecision | None:
        """Return the latest valid decision for a view."""
        return self._decisions.get(view_id)

    def predict(self, view_id: ViewId, image: np.ndarray) -> BranchEvidence:
        """Crop, convert, predict, and threshold one captured view."""
        started = time.perf_counter()
        self._decisions.pop(view_id, None)
        try:
            rgb = self._crop_as_rgb(view_id, image)
            with self._lock:
                raw = self._predictors[view_id].predict(rgb)
            prediction = _validated_prediction(raw)
            raw_map = prediction.anomaly_map.copy()
            raw_map.flags.writeable = False
            heatmap = _display_heatmap(prediction.anomaly_map)
            heatmap.flags.writeable = False
            threshold = self._thresholds[view_id]
            status = (
                BranchStatus.REVIEW
                if threshold is None
                else BranchStatus.NG
                if prediction.score > threshold
                else BranchStatus.PASS
            )
            reason = (
                "calibration defect examples are missing; threshold is unavailable"
                if threshold is None
                else f"raw anomaly score {prediction.score:.6g} "
                f"{'exceeds' if status is BranchStatus.NG else 'does not exceed'} "
                f"threshold {threshold:.6g}"
            )
            self._decisions[view_id] = PatchCoreDecision(
                view_id=view_id,
                score=prediction.score,
                threshold=threshold,
                raw_anomaly_map=raw_map,
                display_heatmap=heatmap,
                checkpoint_path=self._checkpoints[view_id],
                checkpoint_sha256=self._checkpoint_hashes[view_id],
            )
            return self._evidence(view_id, status, prediction.score, threshold, started, reason)
        except Exception as error:
            return self._evidence(
                view_id,
                BranchStatus.ERROR,
                None,
                self._thresholds.get(view_id),
                started,
                f"PatchCore prediction failed: {error}",
            )

    def _crop_as_rgb(self, view_id: ViewId, image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise TypeError("captured image must be a uint8 numpy array")
        if image.ndim not in (2, 3):
            raise ValueError("captured image must be Mono8, BGR, or BGRA")
        height, width = image.shape[:2]
        x1, y1, x2, y2 = self._rois[view_id]
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"ROI for {view_id.value} is outside the captured image")
        crop = image[y1:y2, x1:x2]
        if crop.ndim == 2:
            return np.repeat(crop[:, :, None], 3, axis=2)
        if crop.shape[2] == 3:
            return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        if crop.shape[2] == 4:
            return cv2.cvtColor(crop, cv2.COLOR_BGRA2RGB)
        raise ValueError("captured image must be Mono8, BGR, or BGRA")

    def _evidence(
        self,
        view_id: ViewId,
        status: BranchStatus,
        score: float | None,
        threshold: float | None,
        started: float,
        reason: str,
    ) -> BranchEvidence:
        return BranchEvidence(
            branch=BranchName.PATCHCORE,
            view_id=view_id,
            status=status,
            required_for_ok=self._required_for_ok,
            score=score,
            threshold=threshold,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            reason=reason,
            model_id=self._checkpoint_hashes.get(view_id),
            artifact_paths={},
        )


class _AnomalibPredictor:
    """Thin adapter around Anomalib that keeps a checkpoint resident."""

    def __init__(self, checkpoint: Path) -> None:
        from anomalib.engine import Engine
        from anomalib.models import Patchcore

        self._model = Patchcore.load_from_checkpoint(
            checkpoint,
            map_location="cpu",
            weights_only=False,
            visualizer=False,
        )
        self._engine = Engine(logger=False, enable_checkpointing=False)

    def predict_path(self, image_path: Path) -> RawPatchCorePrediction:
        predictions = self._engine.predict(
            model=self._model,
            data_path=image_path,
            ckpt_path=None,
            return_predictions=True,
        )
        items = [] if predictions is None else [item for batch in predictions for item in batch]
        if len(items) != 1:
            raise RuntimeError("Anomalib must return exactly one prediction")
        item = items[0]
        return RawPatchCorePrediction(score=item.pred_score, anomaly_map=item.anomaly_map)

    def predict(self, image: np.ndarray) -> RawPatchCorePrediction:
        with tempfile.TemporaryDirectory(prefix="bmw-patchcore-") as directory:
            image_path = Path(directory) / "roi.png"
            if not cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                raise RuntimeError("failed to encode PatchCore ROI")
            return self.predict_path(image_path)


class _AnomalibTrainer:
    """Default adapter using Anomalib's supported Folder/Engine lifecycle."""

    def __init__(self, *, accelerator: str = "auto", devices: int | str = 1) -> None:
        self._accelerator = accelerator
        self._devices = devices

    def train(
        self,
        view_id: ViewId,
        view_root: Path,
        run_dir: Path,
        profile: PatchCoreTrainingProfile,
    ) -> Path:
        from anomalib.data import Folder
        from anomalib.engine import Engine
        from anomalib.models import Patchcore
        from lightning import seed_everything

        seed_everything(profile.seed, workers=True)
        datamodule = Folder(
            name=f"bmw_{view_id.value}",
            root=view_root,
            normal_dir="train/good",
            normal_test_dir=None,
            abnormal_dir=None,
            normal_split_ratio=0.0,
            test_split_mode="none",
            val_split_mode="none",
            seed=profile.seed,
            train_batch_size=1,
            eval_batch_size=1,
        )
        model = Patchcore(
            backbone=profile.backbone,
            layers=list(profile.layers),
            pre_trained=True,
            coreset_sampling_ratio=profile.coreset_sampling_ratio,
            num_neighbors=profile.num_neighbors,
            pre_processor=Patchcore.configure_pre_processor(image_size=profile.image_size),
            post_processor=False,
            evaluator=False,
            visualizer=False,
        )
        engine = Engine(
            default_root_dir=run_dir,
            max_epochs=profile.max_epochs,
            accelerator=self._accelerator,
            devices=self._devices,
            deterministic=True,
            precision="32-true",
            logger=False,
        )
        engine.fit(model=model, datamodule=datamodule)
        checkpoint = run_dir / "model.ckpt"
        engine.trainer.save_checkpoint(checkpoint, weights_only=False)
        return checkpoint


def fit_f1_threshold(samples: Sequence[tuple[str, float]]) -> ThresholdFit:
    """Maximize calibration image-level F1 using ``score > threshold``."""
    parsed: list[tuple[bool, float]] = []
    for label, score in samples:
        if label not in {"normal", "defect"}:
            raise ValueError(f"unknown calibration label: {label}")
        if not math.isfinite(float(score)):
            raise ValueError("calibration scores must be finite")
        parsed.append((label == "defect", float(score)))
    if not any(label for label, _ in parsed) or not any(not label for label, _ in parsed):
        return ThresholdFit(threshold=None, f1=None, status=BranchStatus.REVIEW)

    best_threshold: float | None = None
    best_f1 = -1.0
    observed_scores = {score for _, score in parsed}
    below_minimum = math.nextafter(min(observed_scores), -math.inf)
    if not math.isfinite(below_minimum):
        raise ValueError("cannot construct a finite threshold below the minimum calibration score")
    for threshold in sorted(observed_scores | {below_minimum}):
        true_positive = sum(label and score > threshold for label, score in parsed)
        false_positive = sum(not label and score > threshold for label, score in parsed)
        false_negative = sum(label and score <= threshold for label, score in parsed)
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 0.0 if denominator == 0 else 2 * true_positive / denominator
        if f1 > best_f1:
            best_threshold = threshold
            best_f1 = f1
    return ThresholdFit(threshold=best_threshold, f1=best_f1, status=BranchStatus.PASS)


def _image_paths(directory: Path) -> list[Path]:
    suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
    return sorted(path for path in directory.rglob("*") if path.suffix.lower() in suffixes)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_rgb_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot read calibration image: {path}")
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    raise ValueError(f"unsupported calibration image shape for {path}: {image.shape}")


def train_patchcore_views(
    *,
    dataset_root: Path,
    output_root: Path,
    profile: PatchCoreTrainingProfile | None = None,
    trainer: PatchCoreTrainer | None = None,
    calibration_predictor_factory: Callable[[Path], CalibrationPredictor] | None = None,
    runtime_predictor_factory: Callable[[Path], RuntimePredictor] | None = None,
    accelerator: str = "auto",
    devices: int | str = 1,
) -> dict[str, Any]:
    """Train and calibrate exactly six versioned per-view PatchCore models."""
    dataset_root = Path(dataset_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"versioned output already exists: {output_root}")
    profile = profile or PatchCoreTrainingProfile()
    selected_trainer = trainer or _AnomalibTrainer(accelerator=accelerator, devices=devices)
    predictor_factory = calibration_predictor_factory or _AnomalibPredictor
    resident_factory = runtime_predictor_factory or _AnomalibPredictor
    for view_id in ViewId:
        if not _image_paths(dataset_root / view_id.value / "train/good"):
            raise ValueError(f"missing normal training images for {view_id.value}")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent))
    review_views: list[str] = []
    try:
        for view_id in ViewId:
            view_root = dataset_root / view_id.value
            run_dir = staging / view_id.value
            run_dir.mkdir(parents=True)
            checkpoint = Path(selected_trainer.train(view_id, view_root, run_dir, profile)).resolve()
            if not checkpoint.is_file():
                raise RuntimeError(f"trainer did not create checkpoint for {view_id.value}")
            canonical_checkpoint = run_dir / "model.ckpt"
            if checkpoint != canonical_checkpoint.resolve():
                shutil.copy2(checkpoint, canonical_checkpoint)
            checkpoint = canonical_checkpoint
            checkpoint_sha256 = _sha256(checkpoint)
            _write_json(run_dir / "training_config.json", profile.as_dict())

            predictor = predictor_factory(checkpoint)
            raw_dir = run_dir / "calibration/raw_maps"
            heatmap_dir = run_dir / "calibration/heatmaps"
            raw_dir.mkdir(parents=True)
            heatmap_dir.mkdir(parents=True)
            rows: list[tuple[str, str, float, str, str]] = []
            threshold_samples: list[tuple[str, float]] = []
            calibration_predictions: dict[Path, RawPatchCorePrediction] = {}
            for label, split_name in (("normal", "good"), ("defect", "defect")):
                paths = _image_paths(view_root / "calibration" / split_name)
                for index, image_path in enumerate(paths):
                    prediction = _validated_prediction(predictor.predict_path(image_path))
                    stem = f"{label}_{index:04d}_{image_path.stem}"
                    raw_path = raw_dir / f"{stem}.npy"
                    heatmap_path = heatmap_dir / f"{stem}.png"
                    np.save(raw_path, prediction.anomaly_map, allow_pickle=False)
                    if not cv2.imwrite(str(heatmap_path), _display_heatmap(prediction.anomaly_map)):
                        raise RuntimeError(f"failed to write heatmap for {image_path}")
                    relative_image = image_path.relative_to(dataset_root).as_posix()
                    rows.append((view_id.value, label, prediction.score, relative_image, raw_path.name))
                    threshold_samples.append((label, prediction.score))
                    calibration_predictions[image_path] = prediction
            with (run_dir / "calibration_scores.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("view_id", "label", "score", "image_path", "raw_map"))
                writer.writerows(rows)

            fit = fit_f1_threshold(tuple(threshold_samples))
            if fit.status is BranchStatus.REVIEW:
                review_views.append(view_id.value)
            _write_json(
                run_dir / "threshold.json",
                {
                    "view_id": view_id.value,
                    "threshold": fit.threshold,
                    "f1": fit.f1,
                    "status": fit.status.value,
                    "checkpoint_sha256": checkpoint_sha256,
                },
            )
            if not calibration_predictions:
                raise ValueError(f"no calibration image is available for equivalence: {view_id.value}")
            equivalence_sample = next(iter(calibration_predictions))
            runtime_predictor = resident_factory(checkpoint)
            runtime_prediction = runtime_predictor.predict(_read_rgb_image(equivalence_sample))
            receipt = verify_prediction_equivalence(
                checkpoint_path=checkpoint,
                expected_checkpoint_sha256=checkpoint_sha256,
                training_prediction=calibration_predictions[equivalence_sample],
                runtime_prediction=runtime_prediction,
                atol=1e-6,
            )
            _write_json(
                run_dir / "equivalence_receipt.json",
                {
                    "atol": 1e-6,
                    "checkpoint_sha256": receipt.checkpoint_sha256,
                    "sample": equivalence_sample.relative_to(dataset_root).as_posix(),
                    "score_equal": receipt.score_equal,
                    "map_equal": receipt.map_equal,
                },
            )
        report: dict[str, Any] = {
            "ok_profile_allowed": not review_views,
            "review_views": review_views,
            "views": [view_id.value for view_id in ViewId],
            "profile": profile.as_dict(),
        }
        _write_json(staging / "training_report.json", report)
        staging.rename(output_root)
        return report
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_prediction_equivalence(
    *,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    training_prediction: RawPatchCorePrediction,
    runtime_prediction: RawPatchCorePrediction,
    atol: float = 1e-6,
) -> EquivalenceReceipt:
    """Verify score/map equivalence for one image and one exact checkpoint."""
    if atol < 0 or not math.isfinite(atol):
        raise ValueError("atol must be finite and non-negative")
    checkpoint_sha256 = _sha256(Path(checkpoint_path))
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError("checkpoint SHA256 differs from the expected training checkpoint")
    training = _validated_prediction(training_prediction)
    runtime = _validated_prediction(runtime_prediction)
    score_equal = math.isclose(training.score, runtime.score, rel_tol=0.0, abs_tol=atol)
    if not score_equal:
        raise ValueError("raw anomaly score differs between training and runtime")
    map_equal = training.anomaly_map.shape == runtime.anomaly_map.shape and np.allclose(
        training.anomaly_map,
        runtime.anomaly_map,
        rtol=0.0,
        atol=atol,
    )
    if not map_equal:
        raise ValueError("raw anomaly map differs between training and runtime")
    return EquivalenceReceipt(
        checkpoint_sha256=checkpoint_sha256,
        score_equal=score_equal,
        map_equal=map_equal,
    )


__all__ = [
    "EquivalenceReceipt",
    "PatchCoreBackend",
    "PatchCoreDecision",
    "PatchCoreTrainingProfile",
    "RawPatchCorePrediction",
    "ThresholdFit",
    "fit_f1_threshold",
    "train_patchcore_views",
    "verify_prediction_equivalence",
]
