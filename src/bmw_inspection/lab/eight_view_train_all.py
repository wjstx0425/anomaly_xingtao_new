# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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
    efficientad_all_normal_train: bool = False
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
                "all_normal_train": config.efficientad_all_normal_train,
                "validation": (
                    "pending_external_validation"
                    if config.efficientad_all_normal_train
                    else "materialized_normal_test"
                ),
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


def build_efficientad_only_plan(config: LabTrainingConfig) -> tuple[TrainingStep, ...]:
    """Return the isolated EfficientAD-S plan for an existing ROI release.

    The 21:00 diagnostic release has pending YOLO labels, so this plan must
    not select materialization or any branch that depends on YOLO readiness.
    """
    efficientad_step = TrainingStep(
        "efficientad",
        {
            "views": list(config.views),
            "model_size": "small",
            "batch": config.efficientad_batch,
            "epochs": config.efficientad_epochs,
            "image_size": list(config.efficientad_image_size),
            "seed": config.seed,
        },
    )
    if config.efficientad_all_normal_train:
        return (efficientad_step,)
    return (
        efficientad_step,
        TrainingStep(
            "score_normal_test",
            {
                "views": list(config.views),
                "split": "normal_test",
                "expected_normal_part_count": 21,
            },
        ),
        TrainingStep(
            "calibrate_normal_thresholds",
            {
                "target_part_fpr": 1 / 21,
                "allowed_normal_false_positive_count": 1,
                "defect_metrics": "not_evaluated",
            },
        ),
    )


def build_template_only_plan(config: LabTrainingConfig, *, threshold: float) -> tuple[TrainingStep, ...]:
    """Return a single Template stage that reuses an explicit deployment threshold."""
    fixed_threshold = _template_threshold(threshold)
    return (
        TrainingStep(
            "template",
            {
                "views": list(config.views),
                "template_count": 5,
                "fixed_threshold": fixed_threshold,
            },
        ),
    )


StageHandler = Callable[[LabTrainingConfig], dict[str, Any]]
TemplateOnlyStageHandler = Callable[[LabTrainingConfig, float], dict[str, Any]]


def _template_threshold(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("Template fixed threshold must be finite")
    threshold = float(value)
    if threshold < 0.0:
        raise ValueError("Template fixed threshold must be non-negative")
    return threshold


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


def required_yolo_label_names(prepared_root: Path) -> tuple[str, ...]:
    """Return the reviewed YOLO files required by one prepared dataset.

    Args:
        prepared_root (Path): Prepared eight-view dataset root.

    Returns:
        tuple[str, ...]: Sorted YOLO text filenames required for non-normal source rows.

    Raises:
        OSError: If the prepared manifest cannot be read.
        ValueError: If the manifest schema is invalid or contains no defect rows.
    """
    manifest = Path(prepared_root).expanduser().resolve() / "manifests/dataset_manifest.csv"
    with manifest.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"sample_id", "session_id", "view_id", "source_class"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("prepared dataset manifest is missing YOLO identity fields")
        names = {
            f"{row['session_id']}__{row['sample_id']}__{row['view_id']}.txt"
            for row in reader
            if row["source_class"] not in {"normal", "no_streak"}
        }
    if not names:
        raise ValueError("prepared dataset has no defect rows requiring YOLO review")
    return tuple(sorted(names))


def training_release_report_is_complete(report: Mapping[str, Any], views: tuple[str, ...]) -> bool:
    """Return whether a materialized release can be safely reused.

    Args:
        report (Mapping[str, Any]): Materialized training release report.
        views (tuple[str, ...]): Canonical views expected in the report.

    Returns:
        bool: True when YOLO data is ready and every view has the same positive crop count.
    """
    counts = report.get("view_crop_counts")
    if report.get("yolo_training_ready") is not True or not isinstance(counts, Mapping):
        return False
    if set(counts) != set(views):
        return False
    values = [counts[view] for view in views]
    return all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in values) and len(
        set(values)
    ) == 1


def preflight_model_assets(config: LabTrainingConfig) -> dict[str, Any]:
    """Validate shared model settings and local pretrained assets.

    Args:
        config (LabTrainingConfig): Laboratory training configuration.

    Returns:
        dict[str, Any]: Validated views, model assets, and YOLO batch.

    Raises:
        ValueError: If model settings or required local assets are invalid.
    """
    if config.views != VIEW_ORDER:
        raise ValueError("BMW laboratory training requires the canonical eight-view order")
    if config.efficientad_batch != 1:
        raise ValueError("EfficientAD train batch is fixed to 1 by the model implementation")
    if not isinstance(config.efficientad_all_normal_train, bool):
        raise TypeError("efficientad_all_normal_train must be bool")
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
        "YOLO checkpoint": config.yolo_checkpoint,
        "bright-streak base config": config.repo_root / "configs/bmw/bright_streak_demo.json",
    }
    for label, path in required_files.items():
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
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
        "yolo_batch": config.yolo_batch,
        "teacher_weights": str(teacher),
        "imagenette_dir": str(config.imagenette_dir),
    }


def preflight_training(config: LabTrainingConfig) -> dict[str, Any]:
    """Check materialization inputs and shared model assets before training."""
    assets = preflight_model_assets(config)
    required_files = {
        "ROI config": config.roi_config,
        "prepared manifest": config.prepared_root / "manifests/dataset_manifest.csv",
        "bright-streak manifest": config.prepared_root / "manifests/bright_streak.csv",
    }
    for label, path in required_files.items():
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
    if not config.reviewed_yolo_labels.is_dir():
        raise ValueError(f"reviewed YOLO label root does not exist: {config.reviewed_yolo_labels}")
    expected_label_names = set(required_yolo_label_names(config.prepared_root))
    label_files = sorted(config.reviewed_yolo_labels.glob("*.txt"))
    actual_label_names = {path.name for path in label_files}
    if actual_label_names != expected_label_names:
        missing = len(expected_label_names - actual_label_names)
        extra = len(actual_label_names - expected_label_names)
        raise ValueError(
            "reviewed YOLO label root does not match prepared defect rows: "
            f"expected={len(expected_label_names)}, actual={len(actual_label_names)}, missing={missing}, extra={extra}"
        )
    return {**assets, "reviewed_label_count": len(label_files)}


def preflight_efficientad_only(config: LabTrainingConfig) -> dict[str, Any]:
    """Validate only the published EfficientAD data layout.

    This deliberately does not read ``report.json`` or invoke the shared
    preflight because both include YOLO/materialization requirements that are
    irrelevant to the 21:00 EfficientAD-only diagnostic.
    """
    if not config.views or len(set(config.views)) != len(config.views):
        raise ValueError("EfficientAD-only views must be a non-empty unique subset")
    canonical_views = tuple(view for view in VIEW_ORDER if view in config.views)
    if config.views != canonical_views:
        raise ValueError("EfficientAD-only views must follow the canonical BMW view order")
    if config.efficientad_batch != 1:
        raise ValueError("EfficientAD train batch is fixed to 1 by the model implementation")
    if isinstance(config.efficientad_epochs, bool) or not isinstance(config.efficientad_epochs, int):
        raise ValueError("efficientad_epochs must be a positive integer")
    if config.efficientad_epochs <= 0:
        raise ValueError("efficientad_epochs must be a positive integer")
    if isinstance(config.workers, bool) or not isinstance(config.workers, int) or config.workers <= 0:
        raise ValueError("workers must be a positive integer")
    dataset_root = config.training_release / "efficientad"
    image_counts: dict[str, dict[str, int]] = {}
    required_splits = ("normal",) if config.efficientad_all_normal_train else ("normal", "normal_test")
    for view in config.views:
        counts = {
            split: len(_image_paths(dataset_root / view / split))
            for split in required_splits
        }
        if not all(counts.values()):
            raise ValueError(f"EfficientAD release is missing normal data for {view}: {dataset_root / view}")
        image_counts[view] = counts
    return {
        "status": "ready",
        "training_release": str(config.training_release),
        "views": list(config.views),
        "image_counts": image_counts,
        "yolo_training_ready_checked": False,
    }


def preflight_template_only(config: LabTrainingConfig) -> dict[str, Any]:
    """Validate a selected-view Template release without checking other model branches."""
    if not config.views or len(set(config.views)) != len(config.views):
        raise ValueError("Template-only views must be a non-empty unique subset")
    canonical_views = tuple(view for view in VIEW_ORDER if view in config.views)
    if config.views != canonical_views:
        raise ValueError("Template-only views must follow the canonical BMW view order")
    manifest = config.training_release / "template/trainer_manifest.csv"
    groups = _load_eight_view_template_rows(manifest, config.views)
    image_counts: dict[str, int] = {}
    for view, rows in groups.items():
        split_counts = {split: sum(row.split == split for row in rows) for split in ("train", "calibration", "final_test")}
        if split_counts["train"] < 3 or not split_counts["calibration"] or not split_counts["final_test"]:
            raise ValueError(f"Template release lacks train/calibration/final_test rows for {view}")
        for row in rows:
            if cv2.imread(str(row.image_path), cv2.IMREAD_UNCHANGED) is None:
                raise ValueError(f"cannot decode Template input: {row.image_path}")
        image_counts[view] = len(rows)
    return {
        "status": "ready",
        "training_release": str(config.training_release),
        "views": list(config.views),
        "image_counts": image_counts,
    }


def _materialize_stage(config: LabTrainingConfig) -> dict[str, Any]:
    from bmw_inspection.lab.eight_view_training_data import materialize_training_data

    report_path = config.training_release / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not training_release_report_is_complete(report, config.views):
            raise ValueError(f"existing training release is incomplete: {config.training_release}")
        if bool(report.get("efficientad_all_normal_train", False)) != config.efficientad_all_normal_train:
            raise ValueError(
                "existing training release EfficientAD all-normal mode does not match the requested run"
            )
        return {"status": "reused", "training_release": str(config.training_release)}
    report = materialize_training_data(
        prepared_root=config.prepared_root,
        roi_config_path=config.roi_config,
        output_root=config.training_root,
        training_id=config.training_id,
        yolo_label_root=config.reviewed_yolo_labels,
        efficientad_all_normal_train=config.efficientad_all_normal_train,
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


def _load_eight_view_template_rows(
    path: Path,
    views: tuple[str, ...] = VIEW_ORDER,
) -> dict[str, list[_EightViewTemplateSample]]:
    groups = {view: [] for view in views}
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"sample_id", "part_id", "view_id", "image_path", "split", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("template trainer manifest is missing required fields")
        for row_number, row in enumerate(reader, start=2):
            view = row["view_id"]
            if view not in VIEW_ORDER:
                raise ValueError(f"unknown template view at row {row_number}: {view}")
            if view not in groups:
                continue
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
        raise ValueError("template manifest must contain every selected view")
    return groups


def _template_stage(config: LabTrainingConfig) -> dict[str, Any]:
    from bmw_inspection.lab.template import train_template_group

    manifest = config.training_release / "template/trainer_manifest.csv"
    groups = _load_eight_view_template_rows(manifest, config.views)
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


def _template_fixed_stage(config: LabTrainingConfig, threshold: float) -> dict[str, Any]:
    from bmw_inspection.lab.template import train_template_group_fixed_threshold

    fixed_threshold = _template_threshold(threshold)
    manifest = config.training_release / "template/trainer_manifest.csv"
    groups = _load_eight_view_template_rows(manifest, config.views)
    output_root = config.run_dir / "template"
    output_root.mkdir(parents=True, exist_ok=True)
    models: dict[str, str] = {}
    for view in config.views:
        output_dir = output_root / view
        first = cv2.imread(str(groups[view][0].image_path), cv2.IMREAD_UNCHANGED)
        if first is None:
            raise ValueError(f"cannot decode Template input: {groups[view][0].image_path}")
        height, width = first.shape[:2]
        trained = train_template_group_fixed_threshold(  # type: ignore[arg-type]
            groups[view],
            (0, 0, width, height),
            output_dir,
            threshold=fixed_threshold,
            target_size=(512, 512),
            max_shift=12,
            template_count=5,
        )
        models[view] = str(trained)
    report = {
        "status": "complete",
        "model_count": len(models),
        "fixed_threshold": fixed_threshold,
        "models": models,
    }
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


def _efficientad_data_options(config: LabTrainingConfig) -> dict[str, Any]:
    """Return the EfficientAD data contract for checkpoint training."""
    if not isinstance(config.efficientad_all_normal_train, bool):
        raise TypeError("efficientad_all_normal_train must be bool")
    if config.efficientad_all_normal_train:
        return {
            "normal_test_dir": None,
            "test_split_mode": "none",
            "val_split_mode": "none",
            "run_test": False,
            "limit_val_batches": 0,
            "validation_status": "pending_external_validation",
        }
    return {
        "normal_test_dir": "normal_test",
        "test_split_mode": "from_dir",
        "val_split_mode": "same_as_test",
        "run_test": True,
        "limit_val_batches": 1.0,
        "validation_status": "internal_release_validation",
    }


def _efficientad_stage(config: LabTrainingConfig) -> dict[str, Any]:
    from lightning import seed_everything

    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.models import EfficientAd

    dataset_root = config.training_release / "efficientad"
    output_root = config.run_dir / "efficientad"
    output_root.mkdir(parents=True, exist_ok=True)
    data_options = _efficientad_data_options(config)
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
        abnormal_dir = (
            None
            if config.efficientad_all_normal_train
            else "defect" if _image_paths(view_data / "defect") else None
        )
        datamodule = Folder(
            name=f"bmw_{view}",
            root=view_data,
            normal_dir="normal",
            abnormal_dir=abnormal_dir,
            normal_test_dir=data_options["normal_test_dir"],
            normal_split_ratio=0.0,
            extensions=(".png",),
            train_batch_size=1,
            eval_batch_size=1,
            num_workers=config.workers,
            test_split_mode=data_options["test_split_mode"],
            val_split_mode=data_options["val_split_mode"],
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
            limit_val_batches=data_options["limit_val_batches"],
        )
        engine.fit(model=model, datamodule=datamodule)
        run_dir.mkdir(parents=True, exist_ok=True)
        engine.trainer.save_checkpoint(checkpoint, weights_only=False)
        test_metrics = (
            engine.test(model=model, datamodule=datamodule)
            if data_options["run_test"]
            else []
        )
        report = {
            "status": "complete",
            "view": view,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "epochs": config.efficientad_epochs,
            "batch": 1,
            "image_size": list(config.efficientad_image_size),
            "efficientad_all_normal_train": config.efficientad_all_normal_train,
            "validation_status": data_options["validation_status"],
            "threshold_status": (
                "pending_external_validation"
                if config.efficientad_all_normal_train
                else "not_generated_by_training_stage"
            ),
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


def _score_efficientad_normal_test(config: LabTrainingConfig) -> dict[str, Any]:
    """Score the 21 held-out normal physical parts from the new checkpoints."""
    from anomalib.engine import Engine
    from anomalib.models import EfficientAd

    from bmw_inspection.lab.efficientad_thresholds import PartScore, part_id_from_image_path

    output_dir = config.run_dir / "efficientad" / "score_analysis"
    score_path = output_dir / "efficientad_normal_test_scores.csv"
    report_path = output_dir / "normal_test_score_report.json"
    if output_dir.exists():
        raise FileExistsError(f"EfficientAD score output already exists; choose a new --run-id: {output_dir}")
    output_dir.mkdir(parents=True)
    records: list[PartScore] = []
    try:
        for view in config.views:
            checkpoint = config.run_dir / "efficientad" / view / "model.ckpt"
            normal_test_dir = config.training_release / "efficientad" / view / "normal_test"
            if not checkpoint.is_file():
                raise ValueError(f"EfficientAD checkpoint does not exist: {checkpoint}")
            if not _image_paths(normal_test_dir):
                raise ValueError(f"EfficientAD normal_test data is missing: {normal_test_dir}")
            model = EfficientAd.load_from_checkpoint(
                checkpoint,
                map_location="cpu",
                weights_only=False,
                visualizer=False,
            )
            engine = Engine(accelerator="gpu", devices=1, logger=False)
            predictions = engine.predict(
                model=model,
                data_path=normal_test_dir,
                ckpt_path=None,
                return_predictions=True,
            )
            for batch in predictions or []:
                paths = tuple(Path(path) for path in batch.image_path)
                scores = tuple(float(score) for score in batch.pred_score.reshape(-1))
                if len(paths) != len(scores):
                    raise RuntimeError(f"EfficientAD score output count mismatch: {view}")
                records.extend(
                    PartScore(
                        part_id=part_id_from_image_path(path, view),
                        view_id=view,
                        label="normal",
                        score=score,
                        image_path=path,
                    )
                    for path, score in zip(paths, scores, strict=True)
                )
            del engine, model
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except (ImportError, RuntimeError):
                pass
        part_ids = {row.part_id for row in records}
        expected_count = len(config.views) * len(part_ids)
        if not part_ids or len(records) != expected_count:
            raise ValueError("EfficientAD normal_test scores must contain every view for each physical part")
        with score_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("part_id", "view_id", "label", "score", "image_path"))
            writer.writeheader()
            writer.writerows(
                {
                    "part_id": row.part_id,
                    "view_id": row.view_id,
                    "label": row.label,
                    "score": f"{row.score:.12g}",
                    "image_path": str(row.image_path),
                }
                for row in sorted(records, key=lambda row: (row.part_id, row.view_id))
            )
        report = {
            "status": "complete",
            "calibration_source": "normal_test_only",
            "normal_part_count": len(part_ids),
            "score_count": len(records),
            "views": list(config.views),
            "scores_csv": str(score_path),
            "defect_metrics": "not_evaluated",
        }
        _write_json(report_path, report)
        return report
    except BaseException:
        raise


def _calibrate_efficientad_normal_thresholds(
    config: LabTrainingConfig,
    score_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit a <=1/21 false-NG threshold artifact from normal-test scores only."""
    from bmw_inspection.lab.efficientad_thresholds import PartScore, fit_part_thresholds

    raw_score_path = score_report.get("scores_csv")
    score_path = Path(raw_score_path) if isinstance(raw_score_path, str) else None
    if score_path is None or not score_path.is_file():
        raise ValueError("EfficientAD normal-test score report has no readable scores_csv")
    rows: list[PartScore] = []
    with score_path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"part_id", "view_id", "label", "score", "image_path"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("EfficientAD normal-test score CSV is missing required fields")
        for row_number, row in enumerate(reader, start=2):
            if row.get("label") != "normal":
                raise ValueError(f"EfficientAD normal-only score CSV has non-normal row {row_number}")
            try:
                score = float(row.get("score") or "")
            except ValueError as error:
                raise ValueError(f"EfficientAD normal-only score CSV has invalid score at row {row_number}") from error
            rows.append(
                PartScore(
                    part_id=row.get("part_id") or "",
                    view_id=row.get("view_id") or "",
                    label="normal",
                    score=score,
                    image_path=Path(row.get("image_path") or ""),
                )
            )
    fit = fit_part_thresholds(rows, views=config.views, target_part_fpr=1 / 21)
    if fit.normal_part_count <= 0:
        raise ValueError("EfficientAD normal-test calibration requires at least one complete part")
    output_dir = config.run_dir / "efficientad" / "score_analysis"
    threshold_path = output_dir / "part_thresholds.json"
    report_path = output_dir / "part_threshold_report.json"
    if threshold_path.exists() or report_path.exists():
        raise FileExistsError(f"EfficientAD threshold output already exists: {output_dir}")
    checkpoint_sha256_by_view = {
        view: _sha256(config.run_dir / "efficientad" / view / "model.ckpt") for view in config.views
    }
    payload = {
        "schema_version": "bmw.efficientad_normal_only_thresholds/1.0",
        **fit.to_dict(),
        "calibration_source": "normal_test_only",
        "source_csv": str(score_path),
        "source_csv_sha256": _sha256(score_path),
        "checkpoint_sha256_by_view": checkpoint_sha256_by_view,
        "defect_metrics": "not_evaluated",
        "candidate_only": True,
        "default_demo_config_updated": False,
    }
    _write_json(threshold_path, payload)
    report = {"status": "complete", **payload, "threshold_asset": str(threshold_path)}
    _write_json(report_path, report)
    return report


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


def run_efficientad_only_training(
    config: LabTrainingConfig,
    *,
    dry_run: bool = False,
    stage_handler: StageHandler | None = None,
    score_handler: StageHandler | None = None,
    threshold_handler: Callable[[LabTrainingConfig, Mapping[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Train only EfficientAD-S from an already-published ROI release.

    Existing output directories are rejected even when they contain complete
    checkpoints. This keeps each diagnostic candidate immutable and prevents
    accidental reuse of a different run identity.
    """
    plan = build_efficientad_only_plan(config)
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
    if config.run_dir.exists() or config.run_dir.is_symlink():
        raise FileExistsError(f"EfficientAD-only output already exists; choose a new --run-id: {config.run_dir}")
    efficientad_handler = _efficientad_stage if stage_handler is None else stage_handler
    normal_score_handler = _score_efficientad_normal_test if score_handler is None else score_handler
    normal_threshold_handler = (
        _calibrate_efficientad_normal_thresholds if threshold_handler is None else threshold_handler
    )
    config.run_dir.mkdir(parents=True)
    report: dict[str, Any] = {
        "status": "running",
        "experimental_only": True,
        "run_dir": str(config.run_dir),
        "steps": [],
    }
    report_path = config.run_dir / "run_report.json"
    score_result: dict[str, Any] | None = None
    for step, handler in (
        ("efficientad", efficientad_handler),
        ("score_normal_test", normal_score_handler),
    ):
        if config.efficientad_all_normal_train and step == "score_normal_test":
            break
        try:
            result = handler(config)
        except BaseException as error:
            report["status"] = "failed"
            report["failed_step"] = step
            report["error"] = f"{type(error).__name__}: {error}"
            _write_json(report_path, report)
            raise
        report["steps"].append({"name": step, "status": "complete", "result": result})
        _write_json(report_path, report)
        if step == "score_normal_test":
            score_result = result
    if config.efficientad_all_normal_train:
        report["status"] = "complete"
        _write_json(report_path, report)
        return report
    assert score_result is not None
    try:
        threshold_result = normal_threshold_handler(config, score_result)
    except BaseException as error:
        report["status"] = "failed"
        report["failed_step"] = "calibrate_normal_thresholds"
        report["error"] = f"{type(error).__name__}: {error}"
        _write_json(report_path, report)
        raise
    report["steps"].append(
        {"name": "calibrate_normal_thresholds", "status": "complete", "result": threshold_result}
    )
    report["status"] = "complete"
    _write_json(report_path, report)
    return report


def run_template_only_training(
    config: LabTrainingConfig,
    *,
    threshold: float,
    dry_run: bool = False,
    stage_handler: TemplateOnlyStageHandler | None = None,
) -> dict[str, Any]:
    """Train only selected Template views while preserving a fixed threshold."""
    fixed_threshold = _template_threshold(threshold)
    plan = build_template_only_plan(config, threshold=fixed_threshold)
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
    if config.run_dir.exists() or config.run_dir.is_symlink():
        raise FileExistsError(f"Template-only output already exists; choose a new --run-id: {config.run_dir}")
    handler = _template_fixed_stage if stage_handler is None else stage_handler
    config.run_dir.mkdir(parents=True)
    report: dict[str, Any] = {
        "status": "running",
        "experimental_only": True,
        "run_dir": str(config.run_dir),
        "steps": [],
    }
    report_path = config.run_dir / "run_report.json"
    try:
        result = handler(config, fixed_threshold)
    except BaseException as error:
        report["status"] = "failed"
        report["failed_step"] = "template"
        report["error"] = f"{type(error).__name__}: {error}"
        _write_json(report_path, report)
        raise
    report["steps"].append({"name": "template", "status": "complete", "result": result})
    report["status"] = "complete"
    _write_json(report_path, report)
    return report


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
