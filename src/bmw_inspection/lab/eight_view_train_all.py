"""BMW eight-view laboratory training orchestration.

This module intentionally produces experimental artifacts only. It does not
activate checkpoints in the legacy six-view runtime.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

import cv2

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER


@dataclass(frozen=True, slots=True)
class LabTrainingConfig:
    """Paths and small laboratory defaults for one training run."""

    repo_root: Path
    prepared_root: Path
    roi_config: Path
    reviewed_yolo_labels: Path
    training_root: Path
    training_id: str
    output_root: Path
    run_id: str
    yolo_checkpoint: Path
    imagenette_dir: Path
    views: tuple[str, ...] = VIEW_ORDER
    efficientad_batch: int = 1
    efficientad_epochs: int = 30
    efficientad_image_size: tuple[int, int] = (256, 256)
    yolo_batch: int = 32
    yolo_epochs: int = 100
    yolo_image_size: int = 640
    gpu: int = 0
    workers: int = 8
    seed: int = 42

    @classmethod
    def defaults(cls, repo_root: Path) -> LabTrainingConfig:
        """Build repository-relative defaults used by the one-click CLI."""
        root = Path(repo_root).expanduser().resolve()
        return cls(
            repo_root=root,
            prepared_root=root / "dataset/bmw_lab_prepared/bmw_hdr_eight_view_v1",
            roi_config=root / "configs/bmw/rois/bmw_hdr_eight_view_v1.json",
            reviewed_yolo_labels=(
                root
                / "dataset/bmw_lab_labeling/exports/"
                "bmw_label_review_project14_final_20260810_032057/reviewed_yolo_labels"
            ),
            training_root=root / "dataset/bmw_lab_training",
            training_id="bmw_hdr_roi_training_reviewed_v1",
            output_root=root / "results/bmw_lab_one_click",
            run_id="bmw_lab_eight_view_v1",
            yolo_checkpoint=root / "yolo26n.pt",
            imagenette_dir=Path("/home/yunjing/anomalib/.cache/anomalib/imagenette/imagenette2"),
        )

    @property
    def training_release(self) -> Path:
        """Return the immutable materialized training release path."""
        return self.training_root / self.training_id

    @property
    def run_dir(self) -> Path:
        """Return the experiment result directory."""
        return self.output_root / self.run_id


@dataclass(frozen=True, slots=True)
class TrainingStep:
    """One fail-fast stage shown by ``--dry-run``."""

    name: str
    parameters: dict[str, Any]
    status: str = "planned"
    fail_fast: bool = True


def build_training_plan(config: LabTrainingConfig) -> tuple[TrainingStep, ...]:
    """Return the deterministic stage order without starting training."""
    return (
        TrainingStep(
            "materialize",
            {
                "prepared_root": str(config.prepared_root),
                "roi_config": str(config.roi_config),
                "reviewed_yolo_labels": str(config.reviewed_yolo_labels),
                "output": str(config.training_release),
            },
        ),
        TrainingStep(
            "template",
            {"views": list(config.views), "template_count": 5},
        ),
        TrainingStep(
            "bright_streak",
            {"view": "front_left", "fit_split": "calibration"},
        ),
        TrainingStep(
            "efficientad",
            {
                "views": list(config.views),
                "model_size": "small",
                "batch": config.efficientad_batch,
                "epochs": config.efficientad_epochs,
                "image_size": list(config.efficientad_image_size),
            },
        ),
        TrainingStep(
            "yolo",
            {
                "model": str(config.yolo_checkpoint),
                "batch": config.yolo_batch,
                "epochs": config.yolo_epochs,
                "imgsz": config.yolo_image_size,
            },
        ),
    )


StageHandler = Callable[[LabTrainingConfig], dict[str, Any]]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_paths(path: Path) -> list[Path]:
    suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
    return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in suffixes)


def preflight_training(config: LabTrainingConfig) -> dict[str, Any]:
    """Check only the inputs needed before the expensive run starts."""
    if config.views != VIEW_ORDER:
        raise ValueError("BMW laboratory training requires the canonical eight-view order")
    if config.efficientad_batch != 1:
        raise ValueError("EfficientAD train batch is fixed to 1 by the model implementation")
    for name, value in (
        ("efficientad_epochs", config.efficientad_epochs),
        ("yolo_batch", config.yolo_batch),
        ("yolo_epochs", config.yolo_epochs),
        ("yolo_image_size", config.yolo_image_size),
        ("workers", config.workers),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    required_files = {
        "ROI config": config.roi_config,
        "prepared manifest": config.prepared_root / "manifests/dataset_manifest.csv",
        "bright-streak manifest": config.prepared_root / "manifests/bright_streak.csv",
        "YOLO checkpoint": config.yolo_checkpoint,
        "bright-streak base config": config.repo_root / "configs/bmw/bright_streak_demo.json",
    }
    for label, path in required_files.items():
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
    if not config.reviewed_yolo_labels.is_dir():
        raise ValueError(f"reviewed YOLO label root does not exist: {config.reviewed_yolo_labels}")
    label_files = sorted(config.reviewed_yolo_labels.glob("*.txt"))
    if len(label_files) != 248:
        raise ValueError(f"reviewed YOLO label root must contain exactly 248 txt files, got {len(label_files)}")
    if not config.imagenette_dir.is_dir() or not _image_paths(config.imagenette_dir):
        raise ValueError(f"ImageNette data is missing: {config.imagenette_dir}")
    teacher = (
        Path.home()
        / ".cache/anomalib/pre_trained/efficientad_pretrained_weights/pretrained_teacher_small.pth"
    )
    if not teacher.is_file():
        raise ValueError(f"EfficientAD-S teacher weights are missing: {teacher}")
    return {
        "status": "ready",
        "views": list(config.views),
        "reviewed_label_count": len(label_files),
        "yolo_batch": config.yolo_batch,
        "teacher_weights": str(teacher),
        "imagenette_dir": str(config.imagenette_dir),
    }


def _materialize_stage(config: LabTrainingConfig) -> dict[str, Any]:
    from bmw_inspection.lab.eight_view_training_data import materialize_training_data

    report_path = config.training_release / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("yolo_training_ready") is not True or report.get("view_crop_counts") != {
            view: 132 for view in VIEW_ORDER
        }:
            raise ValueError(f"existing training release is incomplete: {config.training_release}")
        return {"status": "reused", "training_release": str(config.training_release)}
    report = materialize_training_data(
        prepared_root=config.prepared_root,
        roi_config_path=config.roi_config,
        output_root=config.training_root,
        training_id=config.training_id,
        yolo_label_root=config.reviewed_yolo_labels,
    )
    if report.get("yolo_training_ready") is not True:
        raise RuntimeError("materialized dataset is not YOLO training ready")
    return {"status": "created", "training_release": str(config.training_release), "report": report}


@dataclass(frozen=True, slots=True)
class _NamedView:
    value: str


@dataclass(frozen=True, slots=True)
class _EightViewTemplateSample:
    sample_id: str
    part_id: str
    view_id: _NamedView
    image_path: Path
    split: str
    label: str


def _load_eight_view_template_rows(path: Path) -> dict[str, list[_EightViewTemplateSample]]:
    groups = {view: [] for view in VIEW_ORDER}
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"sample_id", "part_id", "view_id", "image_path", "split", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("template trainer manifest is missing required fields")
        for row_number, row in enumerate(reader, start=2):
            view = row["view_id"]
            if view not in groups:
                raise ValueError(f"unknown template view at row {row_number}: {view}")
            image_path = Path(row["image_path"])
            image_path = image_path if image_path.is_absolute() else path.parent / image_path
            image_path = image_path.resolve()
            if not image_path.is_file():
                raise ValueError(f"template image does not exist: {image_path}")
            groups[view].append(
                _EightViewTemplateSample(
                    sample_id=row["sample_id"],
                    part_id=row["part_id"],
                    view_id=_NamedView(view),
                    image_path=image_path,
                    split=row["split"],
                    label=row["label"],
                )
            )
    if any(not rows for rows in groups.values()):
        raise ValueError("template manifest must contain all eight views")
    return groups


def _template_stage(config: LabTrainingConfig) -> dict[str, Any]:
    from bmw_inspection.lab.template import train_template_group

    manifest = config.training_release / "template/trainer_manifest.csv"
    groups = _load_eight_view_template_rows(manifest)
    output_root = config.run_dir / "template"
    output_root.mkdir(parents=True, exist_ok=True)
    models: dict[str, str] = {}
    for view in config.views:
        output_dir = output_root / view
        model_path = output_dir / "model.json"
        if model_path.is_file() and (output_dir / "model.sha256").is_file():
            models[view] = str(model_path)
            continue
        if output_dir.exists():
            raise FileExistsError(f"incomplete Template output exists; choose a new --run-id: {output_dir}")
        first = cv2.imread(str(groups[view][0].image_path), cv2.IMREAD_UNCHANGED)
        if first is None:
            raise ValueError(f"cannot decode Template input: {groups[view][0].image_path}")
        height, width = first.shape[:2]
        trained = train_template_group(  # type: ignore[arg-type]
            groups[view],
            (0, 0, width, height),
            output_dir,
            target_size=(512, 512),
            max_shift=12,
            template_count=5,
        )
        models[view] = str(trained)
    report = {"status": "complete", "model_count": len(models), "models": models}
    _write_json(output_root / "training_report.json", report)
    return report


def _is_continuous(
    row: Mapping[str, Any],
    min_longest_run_ratio: float,
    max_gap_ratio: float,
    max_gap_count: int,
) -> bool:
    return bool(
        row["longest_run_ratio"] >= min_longest_run_ratio
        and row["max_gap_ratio"] <= max_gap_ratio
        and row["gap_count"] <= max_gap_count
    )


def _balanced_accuracy(
    records: list[dict[str, Any]],
    contrast: float,
    coverage: float,
    min_longest_run_ratio: float,
    max_gap_ratio: float,
    max_gap_count: int,
) -> tuple[float, int, int]:
    normal = [row for row in records if row["label"] == "normal"]
    missing = [row for row in records if row["label"] == "no_streak"]
    if not normal or not missing:
        raise ValueError("bright-streak evaluation needs both normal and no_streak samples")
    false_rejects = sum(
        not (
            row["contrast_snr"] >= contrast
            and row["coverage_ratio"] >= coverage
            and _is_continuous(row, min_longest_run_ratio, max_gap_ratio, max_gap_count)
        )
        for row in normal
    )
    false_accepts = sum(
        row["contrast_snr"] >= contrast
        and row["coverage_ratio"] >= coverage
        and _is_continuous(row, min_longest_run_ratio, max_gap_ratio, max_gap_count)
        for row in missing
    )
    true_normal_rate = 1.0 - false_rejects / len(normal)
    true_missing_rate = 1.0 - false_accepts / len(missing)
    return (true_normal_rate + true_missing_rate) / 2.0, false_rejects, false_accepts


def _candidate_thresholds(values: list[float]) -> list[float]:
    unique = sorted(set(float(value) for value in values))
    mids = [(left + right) / 2.0 for left, right in zip(unique, unique[1:], strict=False)]
    return sorted(set([0.0, *unique, *mids]))


def _fit_bright_streak_thresholds(calibration: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit presence thresholds while retaining explicit continuity checks."""
    normal = [row for row in calibration if row["label"] == "normal"]
    missing = [row for row in calibration if row["label"] == "no_streak"]
    if not normal or not missing:
        raise ValueError("bright-streak calibration needs both normal and no_streak samples")

    # No interrupted-streak samples exist yet. Use the strictest continuity
    # envelope that still accepts every calibration normal instead of silently
    # disabling continuity or reusing thresholds from a different exposure mode.
    min_longest_run_ratio = min(float(row["longest_run_ratio"]) for row in normal)
    max_gap_ratio = max(float(row["max_gap_ratio"]) for row in normal)
    max_gap_count = max(int(row["gap_count"]) for row in normal)
    minimum_normal_contrast = min(float(row["contrast_snr"]) for row in normal)
    maximum_missing_contrast = max(float(row["contrast_snr"]) for row in missing)
    contrast_floor = (
        (minimum_normal_contrast + maximum_missing_contrast) / 2.0
        if maximum_missing_contrast < minimum_normal_contrast
        else 0.0
    )
    minimum_normal_coverage = min(float(row["coverage_ratio"]) for row in normal)
    maximum_missing_coverage = max(float(row["coverage_ratio"]) for row in missing)
    coverage_floor = (
        (minimum_normal_coverage + maximum_missing_coverage) / 2.0
        if maximum_missing_coverage < minimum_normal_coverage
        else 0.0
    )
    fits: list[tuple[float, int, int, float, float]] = []
    contrast_candidates = [
        value
        for value in _candidate_thresholds([row["contrast_snr"] for row in calibration])
        if value >= contrast_floor
    ]
    coverage_candidates = [
        value
        for value in _candidate_thresholds([row["coverage_ratio"] for row in calibration])
        if value >= coverage_floor
    ]
    for contrast in contrast_candidates:
        for coverage in coverage_candidates:
            balanced, false_rejects, false_accepts = _balanced_accuracy(
                calibration,
                contrast,
                coverage,
                min_longest_run_ratio,
                max_gap_ratio,
                max_gap_count,
            )
            fits.append((balanced, false_rejects, false_accepts, contrast, coverage))
    balanced, false_rejects, false_accepts, contrast, coverage = max(
        fits,
        key=lambda item: (item[0], -item[2], -item[1], -item[4], -item[3]),
    )
    return {
        "min_contrast_snr": contrast,
        "min_coverage_ratio": coverage,
        "min_longest_run_ratio": min_longest_run_ratio,
        "max_gap_ratio": max_gap_ratio,
        "max_gap_count": max_gap_count,
        "balanced_accuracy": balanced,
        "false_rejects": false_rejects,
        "false_accepts": false_accepts,
    }


def _bright_streak_stage(config: LabTrainingConfig) -> dict[str, Any]:
    from bmw_inspection.contracts import load_config, read_json_object
    from bmw_inspection.detector import detect_bright_streak_evidence

    output_root = config.run_dir / "bright_streak"
    model_path = output_root / "calibrated_config.json"
    report_path = output_root / "training_report.json"
    if model_path.is_file() and report_path.is_file():
        return {"status": "reused", "config": str(model_path)}
    if output_root.exists():
        raise FileExistsError(f"incomplete bright-streak output exists; choose a new --run-id: {output_root}")
    output_root.mkdir(parents=True)
    base_path = config.repo_root / "configs/bmw/bright_streak_demo.json"
    base = load_config(base_path)
    manifest = config.prepared_root / "manifests/bright_streak.csv"
    records: list[dict[str, Any]] = []
    with manifest.open("r", newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            expected = row["expected_status"]
            if expected not in {"OK", "NG_NO_STREAK"} or row["split"] not in {"calibration", "final_test"}:
                continue
            image = cv2.imread(row["source_path"], cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"cannot decode bright-streak image: {row['source_path']}")
            decision = detect_bright_streak_evidence(image, base)
            metrics = decision.metrics
            if metrics is None:
                raise ValueError(f"bright-streak metrics are unavailable: {row['source_path']}")
            records.append(
                {
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "label": "normal" if expected == "OK" else "no_streak",
                    "contrast_snr": metrics.contrast_snr,
                    "coverage_ratio": metrics.coverage_ratio,
                    "longest_run_ratio": metrics.longest_run_ratio,
                    "max_gap_ratio": metrics.max_gap_ratio,
                    "gap_count": metrics.gap_count,
                }
            )
    calibration = [row for row in records if row["split"] == "calibration"]
    fitted = _fit_bright_streak_thresholds(calibration)
    final_rows = [row for row in records if row["split"] == "final_test"]
    final_balanced, final_false_rejects, final_false_accepts = _balanced_accuracy(
        final_rows,
        fitted["min_contrast_snr"],
        fitted["min_coverage_ratio"],
        fitted["min_longest_run_ratio"],
        fitted["max_gap_ratio"],
        fitted["max_gap_count"],
    )
    payload = read_json_object(base_path)
    thresholds = dict(payload["thresholds"])
    for name in (
        "min_contrast_snr",
        "min_coverage_ratio",
        "min_longest_run_ratio",
        "max_gap_ratio",
        "max_gap_count",
    ):
        thresholds[name] = fitted[name]
    payload["thresholds"] = thresholds
    payload["mode"] = "demo"
    payload["result_root"] = str(config.run_dir / "bright_streak_runtime")
    _write_json(model_path, payload)
    with (output_root / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        for row in records:
            row["continuous"] = _is_continuous(
                row,
                fitted["min_longest_run_ratio"],
                fitted["max_gap_ratio"],
                fitted["max_gap_count"],
            )
        fieldnames = list(records[0])
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    report = {
        "status": "complete",
        "fit_split": "calibration",
        "final_test_used_for_fit": False,
        "min_contrast_snr": fitted["min_contrast_snr"],
        "min_coverage_ratio": fitted["min_coverage_ratio"],
        "min_longest_run_ratio": fitted["min_longest_run_ratio"],
        "max_gap_ratio": fitted["max_gap_ratio"],
        "max_gap_count": fitted["max_gap_count"],
        "calibration_balanced_accuracy": fitted["balanced_accuracy"],
        "calibration_false_rejects": fitted["false_rejects"],
        "calibration_false_accepts": fitted["false_accepts"],
        "final_test_balanced_accuracy": final_balanced,
        "final_test_false_rejects": final_false_rejects,
        "final_test_false_accepts": final_false_accepts,
        "config": str(model_path),
    }
    _write_json(report_path, report)
    return report


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (RuntimeError, TypeError, ValueError):
            pass
    return str(value)


def _efficientad_stage(config: LabTrainingConfig) -> dict[str, Any]:
    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.models import EfficientAd
    from lightning import seed_everything

    dataset_root = config.training_release / "efficientad"
    output_root = config.run_dir / "efficientad"
    output_root.mkdir(parents=True, exist_ok=True)
    models: dict[str, dict[str, Any]] = {}
    for view in config.views:
        view_data = dataset_root / view
        run_dir = output_root / view
        checkpoint = run_dir / "model.ckpt"
        metrics_path = run_dir / "metrics.json"
        if checkpoint.is_file() and metrics_path.is_file():
            models[view] = {
                "status": "reused",
                "checkpoint": str(checkpoint),
                "sha256": _sha256(checkpoint),
            }
            continue
        if run_dir.exists():
            raise FileExistsError(f"incomplete EfficientAD output exists; choose a new --run-id: {run_dir}")
        abnormal_dir = "defect" if _image_paths(view_data / "defect") else None
        datamodule = Folder(
            name=f"bmw_{view}",
            root=view_data,
            normal_dir="normal",
            abnormal_dir=abnormal_dir,
            normal_test_dir="normal_test",
            normal_split_ratio=0.0,
            extensions=(".png",),
            train_batch_size=1,
            eval_batch_size=1,
            num_workers=config.workers,
            test_split_mode="from_dir",
            val_split_mode="same_as_test",
            seed=config.seed,
        )
        seed_everything(config.seed, workers=True)
        model = EfficientAd(
            imagenet_dir=config.imagenette_dir,
            model_size="small",
            lr=1e-4,
            pre_processor=EfficientAd.configure_pre_processor(image_size=config.efficientad_image_size),
            visualizer=False,
        )
        engine = Engine(
            accelerator="gpu",
            devices=1,
            max_epochs=config.efficientad_epochs,
            default_root_dir=run_dir,
            deterministic=True,
            precision="32-true",
            logger=False,
        )
        engine.fit(model=model, datamodule=datamodule)
        run_dir.mkdir(parents=True, exist_ok=True)
        engine.trainer.save_checkpoint(checkpoint, weights_only=False)
        test_metrics = engine.test(model=model, datamodule=datamodule)
        report = {
            "status": "complete",
            "view": view,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "epochs": config.efficientad_epochs,
            "batch": 1,
            "image_size": list(config.efficientad_image_size),
            "test_metrics": _json_safe(test_metrics),
        }
        _write_json(metrics_path, report)
        models[view] = report
        del engine, model, datamodule
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass
    summary = {"status": "complete", "model_count": len(models), "models": models}
    _write_json(output_root / "training_report.json", summary)
    return summary


def _yolo_stage(config: LabTrainingConfig) -> dict[str, Any]:
    from ultralytics import YOLO

    data_yaml = config.training_release / "yolo/data.yaml"
    if not data_yaml.is_file():
        raise ValueError(f"YOLO data.yaml is missing: {data_yaml}")
    output_root = config.run_dir / "yolo"
    report_path = output_root / "training_report.json"
    expected_best = output_root / "train/weights/best.pt"
    if expected_best.is_file() and report_path.is_file():
        return {
            "status": "reused",
            "best_checkpoint": str(expected_best),
            "checkpoint_sha256": _sha256(expected_best),
        }
    if output_root.exists():
        raise FileExistsError(f"incomplete YOLO output exists; choose a new --run-id: {output_root}")
    model = YOLO(str(config.yolo_checkpoint))
    model.train(
        data=str(data_yaml),
        epochs=config.yolo_epochs,
        batch=config.yolo_batch,
        imgsz=config.yolo_image_size,
        device=config.gpu,
        workers=config.workers,
        seed=config.seed,
        project=str(output_root),
        name="train",
        exist_ok=False,
    )
    best = Path(model.trainer.best).expanduser().resolve()
    if not best.is_file():
        raise RuntimeError(f"YOLO training did not produce best.pt: {best}")
    report = {
        "status": "complete",
        "base_checkpoint": str(config.yolo_checkpoint),
        "best_checkpoint": str(best),
        "checkpoint_sha256": _sha256(best),
        "batch": config.yolo_batch,
        "epochs": config.yolo_epochs,
        "imgsz": config.yolo_image_size,
        "device": config.gpu,
        "data": str(data_yaml),
    }
    _write_json(report_path, report)
    return report


def default_stage_handlers() -> Mapping[str, StageHandler]:
    """Return the concrete laboratory stage implementations."""
    return MappingProxyType(
        {
            "materialize": _materialize_stage,
            "template": _template_stage,
            "bright_streak": _bright_streak_stage,
            "efficientad": _efficientad_stage,
            "yolo": _yolo_stage,
        }
    )


def run_training(
    config: LabTrainingConfig,
    *,
    dry_run: bool = False,
    stage_handlers: Mapping[str, StageHandler] | None = None,
) -> dict[str, Any]:
    """Execute the deterministic plan and stop at the first failed stage."""
    plan = build_training_plan(config)
    if dry_run:
        return {
            "status": "dry_run",
            "experimental_only": True,
            "run_dir": str(config.run_dir),
            "steps": [
                {"name": step.name, "parameters": step.parameters, "status": step.status}
                for step in plan
            ],
        }
    handlers = dict(default_stage_handlers() if stage_handlers is None else stage_handlers)
    missing = [step.name for step in plan if step.name not in handlers]
    if missing:
        raise ValueError(f"missing stage handlers: {', '.join(missing)}")
    config.run_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "status": "running",
        "experimental_only": True,
        "run_dir": str(config.run_dir),
        "steps": [],
    }
    report_path = config.run_dir / "run_report.json"
    for step in plan:
        try:
            result = handlers[step.name](config)
        except BaseException as error:
            report["status"] = "failed"
            report["failed_step"] = step.name
            report["error"] = f"{type(error).__name__}: {error}"
            _write_json(report_path, report)
            raise
        report["steps"].append({"name": step.name, "status": "complete", "result": result})
        _write_json(report_path, report)
    report["status"] = "complete"
    _write_json(report_path, report)
    return report
