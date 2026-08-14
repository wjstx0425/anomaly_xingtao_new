"""Fail-closed orchestration for left-hand normal-only candidate training."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER
from bmw_inspection.lab.eight_view_train_all import _efficientad_stage

TARGET_PART_FPR = 0.05
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class LeftNormalTrainingConfig:
    """Paths for one immutable left-hand normal-only candidate."""

    repo_root: Path
    prepared_root: Path
    roi_config: Path
    training_root: Path
    training_id: str
    output_root: Path
    run_id: str
    mask_index: Path | None = None
    component_policy: Path | None = None
    views: tuple[str, ...] = VIEW_ORDER
    efficientad_epochs: int = 30
    efficientad_image_size: tuple[int, int] = (256, 256)
    gpu: int = 0
    workers: int = 8
    seed: int = 42

    def __post_init__(self) -> None:
        for field in ("repo_root", "prepared_root", "roi_config", "training_root", "output_root"):
            object.__setattr__(self, field, Path(getattr(self, field)).expanduser().resolve())
        if self.views != VIEW_ORDER:
            raise ValueError("left normal-only training requires the canonical eight-view order")
        for field in ("training_id", "run_id"):
            if not _IDENTIFIER.fullmatch(getattr(self, field)):
                raise ValueError(f"{field} contains unsupported characters")

    @property
    def training_release(self) -> Path:
        return self.training_root / self.training_id

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_id


@dataclass(frozen=True, slots=True)
class LeftTrainingStep:
    name: str
    parameters: Mapping[str, Any]


StageHandler = Callable[[LeftNormalTrainingConfig, Mapping[str, Any]], dict[str, Any]]


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file() or candidate.is_symlink():
        raise ValueError(f"{label} does not exist as a regular file: {candidate}")
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {candidate}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {candidate}")
    return value


def _left_sha(path: Path, label: str) -> str:
    if _read_object(path, label).get("capture_scope") != "left":
        raise ValueError(f"{label} capture_scope must be left: {Path(path).expanduser().resolve()}")
    return _sha256(path)


def validate_left_normal_inputs(config: LeftNormalTrainingConfig) -> dict[str, str]:
    """Validate all cross-hand-sensitive identities before any stage may run."""

    return {
        "prepared_report_sha256": _left_sha(config.prepared_root / "report.json", "prepared release report"),
        "roi_config_sha256": _left_sha(config.roi_config, "ROI config"),
    }


def build_left_normal_plan(config: LeftNormalTrainingConfig) -> tuple[LeftTrainingStep, ...]:
    """Return the exact five-stage plan; it intentionally contains no YOLO or bright streak."""

    return (
        LeftTrainingStep("materialize", {"prepared_root": str(config.prepared_root), "training_release": str(config.training_release)}),
        LeftTrainingStep("template", {"train_split": "train", "fit_split": "calibration", "views": list(config.views)}),
        LeftTrainingStep("efficientad", {"views": list(config.views), "batch": 1, "epochs": config.efficientad_epochs}),
        LeftTrainingStep("score_component_maps", {"score_source": "accepted_component_max_p95"}),
        LeftTrainingStep("calibrate_component_thresholds", {"fit_split": "calibration", "target_part_fpr": TARGET_PART_FPR}),
    )


def _view_hashes(paths: Mapping[str, Path], label: str) -> dict[str, str]:
    if tuple(paths) != VIEW_ORDER:
        raise ValueError(f"{label} must contain all eight BMW views in canonical order")
    result: dict[str, str] = {}
    for view in VIEW_ORDER:
        path = Path(paths[view]).expanduser().resolve()
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"{label} does not exist as a regular file: {path}")
        result[view] = _sha256(path)
    return result


def _thresholds(values: Mapping[str, float], label: str) -> dict[str, float]:
    if tuple(values) != VIEW_ORDER:
        raise ValueError(f"{label} must contain all eight BMW views in canonical order")
    result = {view: float(values[view]) for view in VIEW_ORDER}
    if any(not math.isfinite(value) or value < 0 for value in result.values()):
        raise ValueError(f"{label} must be finite non-negative values")
    return result


def build_template_normal_threshold_artifact(
    model_json_paths: Mapping[str, Path], thresholds: Mapping[str, float], *, calibration_row_count: int
) -> dict[str, Any]:
    """Describe per-view candidate thresholds fit from normal calibration rows."""

    if isinstance(calibration_row_count, bool) or calibration_row_count <= 0:
        raise ValueError("calibration_row_count must be positive")
    return {
        "schema_version": "bmw.left_template_normal_thresholds/1.0",
        "candidate_only": True,
        "defect_metrics": "not_evaluated",
        "fit_split": "calibration",
        "calibration_label": "normal",
        "calibration_row_count": calibration_row_count,
        "thresholds": _thresholds(thresholds, "Template thresholds"),
        "model_json_sha256_by_view": _view_hashes(model_json_paths, "Template model JSON paths"),
    }


def build_component_threshold_artifact(
    *,
    checkpoint_paths: Mapping[str, Path],
    mask_index: Path,
    component_policy: Path,
    score_source: str,
    sample_count: int,
    fpr_resolution: float,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """Describe the SHA-bound whole-part component-score calibration asset."""

    if not isinstance(score_source, str) or not score_source.strip():
        raise ValueError("score_source must be non-empty")
    if isinstance(sample_count, bool) or sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if not math.isfinite(float(fpr_resolution)) or not 0 < float(fpr_resolution) <= 1:
        raise ValueError("fpr_resolution must be finite and between zero and one")
    mask = Path(mask_index).expanduser().resolve()
    policy = Path(component_policy).expanduser().resolve()
    _read_object(mask, "mask index")
    _read_object(policy, "component policy")
    return {
        "schema_version": "bmw.left_efficientad_component_thresholds/1.0",
        "candidate_only": True,
        "defect_metrics": "not_evaluated",
        "demo_only": True,
        "test_used_for_selection": True,
        "target_part_fpr": TARGET_PART_FPR,
        "thresholds": _thresholds(thresholds, "EfficientAD thresholds"),
        "checkpoint_sha256_by_view": _view_hashes(checkpoint_paths, "EfficientAD checkpoint paths"),
        "source_csv": str(mask),
        "source_csv_sha256": _sha256(mask),
        "mask_index_sha256": _sha256(mask),
        "component_policy_sha256": _sha256(policy),
        "score_source": score_source,
        "sample_count": sample_count,
        "fpr_resolution": float(fpr_resolution),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def materialize_left_normal_data(config: LeftNormalTrainingConfig) -> dict[str, Any]:
    """Publish the fresh ROI release without creating or training YOLO assets."""
    from bmw_inspection.lab.eight_view_training_data import materialize_training_data
    return materialize_training_data(
        prepared_root=config.prepared_root, roi_config_path=config.roi_config,
        output_root=config.training_root, training_id=config.training_id,
    )


def train_left_normal_templates(config: LeftNormalTrainingConfig) -> dict[str, Any]:
    """Train from normal train rows and envelope-fit each threshold on normal calibration rows."""
    from bmw_inspection.lab.eight_view_train_all import _load_eight_view_template_rows
    from bmw_inspection.lab.template import train_template_group_fixed_threshold
    groups = _load_eight_view_template_rows(config.training_release / "template/trainer_manifest.csv")
    root = config.run_dir / "template"
    root.mkdir(parents=True, exist_ok=True)
    models: dict[str, Path] = {}
    thresholds: dict[str, float] = {}
    for view in config.views:
        output = root / view
        model = output / "model.json"
        if not model.is_file():
            first = cv2.imread(str(groups[view][0].image_path), cv2.IMREAD_UNCHANGED)
            if first is None:
                raise ValueError(f"cannot decode Template input: {groups[view][0].image_path}")
            height, width = first.shape[:2]
            train_template_group_fixed_threshold(groups[view], (0, 0, width, height), output, threshold=1.0)
        risks: list[float] = []
        with (output / "calibration_rows.csv").open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("label") != "normal":
                    raise ValueError("left Template calibration must contain normal rows only")
                risks.append(float(row["risk"]))
        if not risks:
            raise ValueError(f"left Template calibration has no normal rows: {view}")
        threshold = math.nextafter(max(risks), math.inf)
        payload = _read_object(model, "Template model")
        payload["threshold"] = threshold
        payload["threshold_fit"] = {"split": "calibration", "metric": "normal_envelope", "final_test_used": False}
        _write_json(model, payload)
        (output / "model.sha256").write_text(_sha256(model) + "\n", encoding="ascii")
        models[view] = model
        thresholds[view] = threshold
    artifact = build_template_normal_threshold_artifact(models, thresholds, calibration_row_count=sum(1 for _ in (root / config.views[0] / "calibration_rows.csv").open(encoding="utf-8")) - 1)
    _write_json(root / "normal_thresholds.json", artifact)
    return {"status": "complete", "models": {view: str(models[view]) for view in config.views}, "threshold_asset": str(root / "normal_thresholds.json")}


def _lab_config(config: LeftNormalTrainingConfig):
    from dataclasses import replace
    from bmw_inspection.lab.eight_view_train_all import LabTrainingConfig
    defaults = LabTrainingConfig.defaults(config.repo_root)
    return replace(defaults, prepared_root=config.prepared_root, roi_config=config.roi_config, training_root=config.training_root, training_id=config.training_id, output_root=config.output_root, run_id=config.run_id, efficientad_epochs=config.efficientad_epochs, gpu=config.gpu, workers=config.workers, seed=config.seed)


def score_component_maps(config: LeftNormalTrainingConfig, state: Mapping[str, Any]) -> dict[str, Any]:
    """Require the completed component-map scorer rather than silently use pred_score."""
    if config.mask_index is None or config.component_policy is None:
        raise ValueError("calibrate stage requires --mask-index and --component-policy")
    mask = _read_object(config.mask_index, "mask index")
    policy = _read_object(config.component_policy, "component policy")
    if mask.get("public_roi_config_sha256") != _sha256(config.roi_config):
        raise ValueError("mask index ROI SHA does not match the left ROI config")
    if policy.get("schema_version") != "bmw.efficientad_component_filter/1.0":
        raise ValueError("unsupported component policy schema")
    raise RuntimeError("component-map predictor must be provided by the completed EfficientAD integration")


def calibrate_component_thresholds(config: LeftNormalTrainingConfig, state: Mapping[str, Any]) -> dict[str, Any]:
    raise RuntimeError("component-map predictor must publish a component score CSV before calibration")


def run_left_normal_training(config: LeftNormalTrainingConfig, *, dry_run: bool = False, stage: str = "all", stage_handlers: Mapping[str, StageHandler] | None = None) -> dict[str, Any]:
    """Run real default train handlers; calibration is a separate post-mask stage."""
    if stage not in {"all", "train", "calibrate"}:
        raise ValueError("stage must be one of: all, train, calibrate")
    identities = validate_left_normal_inputs(config)
    plan = build_left_normal_plan(config)
    selected = plan[:3] if stage == "train" else plan[3:] if stage == "calibrate" else plan
    if dry_run:
        if stage in {"all", "calibrate"} and (config.mask_index is None or config.component_policy is None):
            raise ValueError("calibrate stage requires --mask-index and --component-policy")
        return {"status": "dry_run", "candidate_only": True, "gpu_work_started": False, "run_dir": str(config.run_dir), "identities": identities, "steps": [{"name": step.name, "parameters": dict(step.parameters), "status": "planned"} for step in selected]}
    if stage in {"train", "all"}:
        if config.run_dir.exists() or config.run_dir.is_symlink():
            raise FileExistsError(f"left normal-only output already exists; choose a new --run-id: {config.run_dir}")
        config.run_dir.mkdir(parents=True)
    elif not config.run_dir.is_dir():
        raise ValueError("calibrate stage requires the completed train run directory")
    handlers: dict[str, StageHandler] = {
        "materialize": lambda current, _state: materialize_left_normal_data(current),
        "template": lambda current, _state: train_left_normal_templates(current),
        "efficientad": lambda current, _state: _efficientad_stage(_lab_config(current)),
        "score_component_maps": score_component_maps,
        "calibrate_component_thresholds": calibrate_component_thresholds,
    }
    handlers.update(stage_handlers or {})
    report: dict[str, Any] = {"status": "running", "candidate_only": True, "run_dir": str(config.run_dir), "identities": identities, "steps": []}
    state: dict[str, Any] = {}
    report_path = config.run_dir / "run_report.json"
    for step_item in selected:
        try:
            result = handlers[step_item.name](config, state)
        except BaseException as error:
            report.update({"status": "failed", "failed_step": step_item.name, "error": f"{type(error).__name__}: {error}"})
            _write_json(report_path, report)
            raise
        state[step_item.name] = result
        report["steps"].append({"name": step_item.name, "status": "complete", "result": result})
        _write_json(report_path, report)
    report["status"] = "complete"
    _write_json(report_path, report)
    return report
