"""Minimal non-YOLO retraining orchestration for a normal-only BMW release."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2

from bmw_inspection.lab.bright_streak_recalibration import recalibrate_bright_streak
from bmw_inspection.lab.eight_view_roi import EightViewRoiConfig, load_roi_config, save_roi_config
from bmw_inspection.lab.eight_view_train_all import (
    LabTrainingConfig,
    _calibrate_efficientad_normal_thresholds,
    _efficientad_stage,
    _load_eight_view_template_rows,
    _score_efficientad_normal_test,
)
from bmw_inspection.lab.eight_view_training_data import materialize_training_data


STEP_ORDER = (
    "derive_roi",
    "materialize",
    "prepare_bright_streak",
    "template",
    "bright_streak",
    "efficientad",
    "score_normal_test",
    "calibrate_normal_thresholds",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class NormalOnlyRetrainingConfig:
    """Inputs and fresh output identities for one selective laboratory run."""

    repo_root: Path
    prepared_root: Path
    reuse_roi: Path
    old_no_streak_release: Path
    training_root: Path
    training_id: str
    output_root: Path
    run_id: str
    derived_roi: Path
    bright_base_config: Path
    imagenette_dir: Path
    efficientad_epochs: int = 30
    gpu: int = 0
    workers: int = 8
    seed: int = 42
    baseline_template_root: Path | None = None

    @property
    def training_release(self) -> Path:
        return self.training_root / self.training_id

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_id


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return tuple(reader.fieldnames or ()), list(reader)


def validate_normal_only_prepared(prepared_root: Path) -> dict[str, Any]:
    """Reject a prepared release containing anything except normal parts."""
    root = Path(prepared_root).expanduser().resolve()
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    counts = report.get("source_class_part_counts")
    if not isinstance(counts, dict) or set(counts) != {"normal"} or not isinstance(counts["normal"], int):
        raise ValueError("prepared release must contain normal parts only")
    if counts["normal"] <= 0:
        raise ValueError("prepared release must contain at least one normal part")
    return report


def derive_fixed_setup_roi(config: NormalOnlyRetrainingConfig) -> dict[str, Any]:
    """Copy existing coordinates into a fixed-setup ROI bound to the new release."""
    source = load_roi_config(config.reuse_roi)
    report = validate_normal_only_prepared(config.prepared_root)
    manifest = (config.prepared_root / "manifests/dataset_manifest.csv").resolve()
    fields, rows = _read_csv(manifest)
    if "sample_id" not in fields or not rows:
        raise ValueError("prepared dataset manifest is empty or invalid")
    capture_scope = report.get("capture_scope")
    if capture_scope not in {"left", "right"}:
        raise ValueError("prepared release capture_scope must be left or right")
    asset = EightViewRoiConfig(
        dataset_id=config.training_id,
        source_manifest=manifest,
        source_manifest_sha256=_sha256(manifest),
        representative_sample_id=rows[0]["sample_id"],
        image_width=source.image_width,
        image_height=source.image_height,
        part_rois=source.part_rois,
        binding_mode="fixed_setup",
        capture_scope=capture_scope,
    )
    if config.derived_roi.exists():
        existing = load_roi_config(config.derived_roi)
        if existing != asset:
            raise FileExistsError(f"ROI config already exists with different content: {config.derived_roi}")
        return {"status": "reused", "roi_config": str(config.derived_roi.resolve())}
    save_roi_config(config.derived_roi, asset)
    return {"status": "complete", "roi_config": str(config.derived_roi.resolve())}


def materialize_normal_only_training_data(config: NormalOnlyRetrainingConfig) -> dict[str, Any]:
    """Materialize once or reuse the exact matching published release."""
    report_path = config.training_release / "report.json"
    prepared_manifest = (config.prepared_root / "manifests/dataset_manifest.csv").resolve()
    expected = {
        "release_status": "published",
        "prepared_manifest": str(prepared_manifest),
        "prepared_manifest_sha256": _sha256(prepared_manifest),
        "roi_config": str(config.derived_roi.resolve()),
        "roi_config_sha256": _sha256(config.derived_roi),
    }
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if any(report.get(name) != value for name, value in expected.items()):
            raise FileExistsError(f"training release already exists with different content: {config.training_release}")
        return {**report, "status": "reused"}
    if config.training_release.exists():
        raise FileExistsError(f"incomplete training release already exists: {config.training_release}")
    report = materialize_training_data(
        prepared_root=config.prepared_root,
        roi_config_path=config.derived_roi,
        output_root=config.training_root,
        training_id=config.training_id,
    )
    return {**report, "status": "complete"}


def build_bright_streak_manifest(
    prepared_root: Path,
    old_no_streak_release: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Merge new normal rows with historical no-streak rows only."""
    new_manifest = Path(prepared_root).resolve() / "manifests/bright_streak.csv"
    fields, new_rows = _read_csv(new_manifest)
    old_release = Path(old_no_streak_release).resolve()
    old_report = json.loads((old_release / "report.json").read_text(encoding="utf-8"))
    prepared_manifest = old_report.get("prepared_manifest")
    if not isinstance(prepared_manifest, str):
        raise ValueError("historical release has no prepared_manifest")
    old_manifest = Path(prepared_manifest).resolve().parent / "bright_streak.csv"
    old_fields, old_rows = _read_csv(old_manifest)
    if old_fields != fields:
        raise ValueError("bright-streak manifest schemas differ")
    normals = [row for row in new_rows if row.get("source_class") == "normal"]
    missing = [row for row in old_rows if row.get("source_class") == "no_streak"]
    missing_splits = {row.get("split") for row in missing}
    if not normals or not {"calibration", "final_test"}.issubset(missing_splits):
        raise ValueError("bright-streak calibration needs new normals and historical no_streak calibration/final_test")
    combined = [*normals, *({**row, "sample_id": f"historical::{row['sample_id']}"} for row in missing)]
    destination = Path(output_path).resolve()
    if destination.exists():
        existing_fields, existing_rows = _read_csv(destination)
        if existing_fields != fields or existing_rows != combined:
            raise FileExistsError(f"bright-streak manifest already exists with different content: {destination}")
        return {
            "status": "reused",
            "manifest": str(destination),
            "normal_count": len(normals),
            "no_streak_count": len(missing),
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(combined)
    return {
        "status": "complete",
        "manifest": str(destination),
        "normal_count": len(normals),
        "no_streak_count": len(missing),
    }


def _lab_config(config: NormalOnlyRetrainingConfig) -> LabTrainingConfig:
    defaults = LabTrainingConfig.defaults(config.repo_root)
    return replace(
        defaults,
        prepared_root=config.prepared_root,
        roi_config=config.derived_roi,
        training_root=config.training_root,
        training_id=config.training_id,
        output_root=config.output_root,
        run_id=config.run_id,
        imagenette_dir=config.imagenette_dir,
        efficientad_epochs=config.efficientad_epochs,
        gpu=config.gpu,
        workers=config.workers,
        seed=config.seed,
    )


def train_normal_only_templates(
    config: LabTrainingConfig,
    baseline_template_root: Path,
    *,
    trainer: Callable[..., Path] | None = None,
) -> dict[str, Any]:
    """Train new templates from normal rows while retaining baseline thresholds."""
    from bmw_inspection.lab.template import train_template_group_fixed_threshold

    train_group = trainer or train_template_group_fixed_threshold
    manifest = config.training_release / "template/trainer_manifest.csv"
    groups = _load_eight_view_template_rows(manifest)
    baseline = Path(baseline_template_root).expanduser().resolve()
    output_root = config.run_dir / "template"
    output_root.mkdir(parents=True, exist_ok=True)
    models: dict[str, dict[str, Any]] = {}
    for view in config.views:
        baseline_model = baseline / view / "model.json"
        try:
            threshold = float(json.loads(baseline_model.read_text(encoding="utf-8"))["threshold"])
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"baseline Template threshold is unavailable for {view}: {baseline_model}") from error
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError(f"baseline Template threshold is invalid for {view}: {baseline_model}")
        output_dir = output_root / view
        model_path = output_dir / "model.json"
        if model_path.is_file() and (output_dir / "model.sha256").is_file():
            models[view] = {"status": "reused", "model": str(model_path), "threshold": threshold}
            continue
        if output_dir.exists():
            raise FileExistsError(f"incomplete Template output exists: {output_dir}")
        first = cv2.imread(str(groups[view][0].image_path), cv2.IMREAD_UNCHANGED)
        if first is None:
            raise ValueError(f"cannot decode Template input: {groups[view][0].image_path}")
        height, width = first.shape[:2]
        trained = train_group(
            groups[view],
            (0, 0, width, height),
            output_dir,
            threshold=threshold,
            target_size=(512, 512),
            max_shift=12,
            template_count=5,
        )
        models[view] = {"status": "complete", "model": str(trained), "threshold": threshold}
    report = {"status": "complete", "model_count": len(models), "models": models}
    (output_root / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def recalibrate_bright_streak_or_reuse(
    manifest_path: Path,
    base_config_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Recalibrate once or reuse an output bound to the same inputs."""
    manifest = Path(manifest_path).expanduser().resolve()
    base_config = Path(base_config_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not output.exists():
        return recalibrate_bright_streak(manifest, base_config, output)
    report_path = output / "report.json"
    calibrated_path = output / "calibrated_config.json"
    if not report_path.is_file() or not calibrated_path.is_file():
        raise FileExistsError(f"incomplete bright-streak output exists: {output}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    identities = report.get("identities")
    expected = {
        "status": "complete",
        "manifest": str(manifest),
        "base_config": str(base_config),
        "config": str(calibrated_path),
    }
    if any(report.get(name) != value for name, value in expected.items()):
        raise FileExistsError(f"bright-streak output is bound to different inputs: {output}")
    if not isinstance(identities, dict) or any(
        identities.get(name) != digest
        for name, digest in {
            "manifest_sha256": _sha256(manifest),
            "base_config_sha256": _sha256(base_config),
            "calibrated_config_sha256": _sha256(calibrated_path),
        }.items()
    ):
        raise FileExistsError(f"bright-streak output identity mismatch: {output}")
    return {**report, "status": "reused"}


StageHandler = Callable[[], dict[str, Any]]


def run_normal_only_retraining(
    config: NormalOnlyRetrainingConfig,
    *,
    dry_run: bool = False,
    stage_handlers: Mapping[str, StageHandler] | None = None,
) -> dict[str, Any]:
    """Run the fixed non-YOLO stage order and stop on the first failure."""
    validate_normal_only_prepared(config.prepared_root)
    if dry_run:
        return {"status": "dry_run", "steps": list(STEP_ORDER), "yolo_trained": False}
    lab = _lab_config(config)
    baseline_template_root = config.baseline_template_root or (
        config.repo_root
        / "results/bmw_lab_one_click/bmw_right_batch_20260810_21_v3_ng_evidence_demo_v1/template"
    )
    bright_manifest = config.run_dir / "bright_streak_input/bright_streak.csv"
    score_result: dict[str, Any] = {}
    handlers: dict[str, StageHandler] = {
        "derive_roi": lambda: derive_fixed_setup_roi(config),
        "materialize": lambda: materialize_normal_only_training_data(config),
        "prepare_bright_streak": lambda: build_bright_streak_manifest(
            config.prepared_root, config.old_no_streak_release, bright_manifest
        ),
        "template": lambda: train_normal_only_templates(lab, baseline_template_root),
        "bright_streak": lambda: recalibrate_bright_streak_or_reuse(
            bright_manifest, config.bright_base_config, config.run_dir / "bright_streak"
        ),
        "efficientad": lambda: _efficientad_stage(lab),
        "score_normal_test": lambda: _score_efficientad_normal_test(lab),
        "calibrate_normal_thresholds": lambda: _calibrate_efficientad_normal_thresholds(lab, score_result),
    }
    if stage_handlers is not None:
        handlers.update(stage_handlers)
    report_path = config.run_dir / "run_report.json"
    resumed_from: dict[str, Any] | None = None
    if config.run_dir.exists():
        if not report_path.is_file():
            raise FileExistsError(f"normal-only output is incomplete: {config.run_dir}")
        resumed_from = json.loads(report_path.read_text(encoding="utf-8"))
        if resumed_from.get("status") != "failed":
            raise FileExistsError(f"normal-only output already exists; choose a new --run-id: {config.run_dir}")
    else:
        config.run_dir.mkdir(parents=True)
    report: dict[str, Any] = {
        "status": "running",
        "steps": [],
        "yolo_trained": False,
        "resumed_from_failed_step": resumed_from.get("failed_step") if resumed_from else None,
    }
    for name in STEP_ORDER:
        try:
            result = handlers[name]()
        except BaseException as error:
            report.update(status="failed", failed_step=name, error=f"{type(error).__name__}: {error}")
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            raise
        if name == "score_normal_test":
            score_result.clear()
            score_result.update(result)
        report["steps"].append({"name": name, "status": "complete", "result": result})
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["status"] = "complete"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


__all__ = [
    "NormalOnlyRetrainingConfig",
    "STEP_ORDER",
    "build_bright_streak_manifest",
    "derive_fixed_setup_roi",
    "materialize_normal_only_training_data",
    "recalibrate_bright_streak_or_reuse",
    "run_normal_only_retraining",
    "train_normal_only_templates",
    "validate_normal_only_prepared",
]
