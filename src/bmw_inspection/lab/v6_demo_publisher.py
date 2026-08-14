# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Atomic publisher for the BMW V6 composite Demo assets."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER


_V5_CONFIG = Path("configs/bmw/experiments/bmw_eight_view_demo_v5_template_manual_ignore_mask_v1.json")
_NEW_RUN = Path("results/bmw_lab_one_click/bmw_right_normal_20260814_models_v1")
_NEW_TRAINING_ROI = Path("configs/bmw/rois/bmw_right_normal_20260814_roi_v1.json")
_NEW_PREPARED_MANIFEST = Path(
    "dataset/bmw_lab_prepared/bmw_right_normal_20260814_v1/manifests/dataset_manifest.csv"
)
_BRIGHT_CANDIDATE = Path(
    "results/bmw_bright_streak_rotated_retrain/bmw_right_normal50_no_streak1_20260814_v2/report.json"
)


def publish_v6_demo(repo_root: Path, *, output_run: Path, output_config: Path) -> dict[str, object]:
    """Publish a fresh V6 composite without changing the V5 inputs.

    The V6 models come from the completed normal-only training run. Its
    numerical thresholds are deliberately *not* selected for this composite:
    V5 threshold objects and the shared V5 mask remain intact, with only model
    identities rebound to the new Template and EfficientAD assets.
    """
    root = Path(repo_root).expanduser().resolve()
    final_run = _under_root(root, output_run)
    final_config = _under_root(root, output_config)
    if final_run.exists() or final_config.exists():
        existing = final_run if final_run.exists() else final_config
        raise FileExistsError(f"refuse to overwrite existing V6 destination: {existing}")

    v5_config_path = root / _V5_CONFIG
    v5_config = _load_object(v5_config_path, "V5 Demo config")
    new_run = root / _NEW_RUN
    _validate_new_training_run(new_run)
    new_prepared_manifest = root / _NEW_PREPARED_MANIFEST
    if not new_prepared_manifest.is_file():
        raise ValueError(f"new prepared manifest is missing: {new_prepared_manifest}")
    _assert_public_roi_equivalent(
        _resolve_config_path(v5_config_path, v5_config["roi_config"]),
        root / _NEW_TRAINING_ROI,
    )

    v5_training_run = _resolve_config_path(v5_config_path, v5_config["training_run"])
    yolo_checkpoint = v5_training_run / "yolo/train/weights/best.pt"
    if not yolo_checkpoint.is_file():
        raise ValueError(f"V5 YOLO checkpoint is missing: {yolo_checkpoint}")
    yolo_sha256 = _sha256(yolo_checkpoint)

    template_source = _resolve_config_path(v5_config_path, _mapping(v5_config, "template")["threshold_artifact"])
    efficientad_source = _resolve_config_path(v5_config_path, _mapping(v5_config, "efficientad")["threshold_artifact"])
    template_rebound = _rebind_template_thresholds(template_source, new_run)
    efficientad_rebound = _rebind_efficientad_thresholds(efficientad_source, new_run)
    template_unchanged = _same_numeric_leaves(_load_object(template_source, "V5 Template thresholds"), template_rebound)
    efficientad_unchanged = _same_numeric_leaves(
        _load_object(efficientad_source, "V5 EfficientAD thresholds"), efficientad_rebound
    )
    if not template_unchanged or not efficientad_unchanged:
        raise AssertionError("V5 numeric thresholds must not be changed while rebinding model identities")

    final_run.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{final_run.name}.staging-", dir=final_run.parent))
    try:
        _write_json(staging / "template_thresholds_v5_rebound.json", template_rebound)
        _write_json(staging / "efficientad_thresholds_v5_rebound.json", efficientad_rebound)
        _symlink_directory(staging / "template", new_run / "template")
        _symlink_directory(staging / "efficientad", new_run / "efficientad")
        _symlink_directory(staging / "yolo", v5_training_run / "yolo")
        composition = {
            "schema": "bmw.v6_demo_composition/1.0",
            "template": {"path": str(new_run / "template")},
            "efficientad": {"path": str(new_run / "efficientad")},
            "yolo": {"path": str(yolo_checkpoint), "sha256": yolo_sha256, "source": "V5"},
            "public_roi": {
                "path": str(_resolve_config_path(v5_config_path, v5_config["roi_config"])),
                "training_roi": str(root / _NEW_TRAINING_ROI),
                "coordinates_equal": True,
            },
            "bright_streak": {"candidate_report": str(root / _BRIGHT_CANDIDATE)},
        }
        _write_json(staging / "composition.json", composition)
        staging.rename(final_run)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    try:
        v6_config = _build_v6_config(
            v5_config,
            output_run=final_run,
            output_config=final_config,
            prepared_manifest=new_prepared_manifest,
            template_thresholds=final_run / "template_thresholds_v5_rebound.json",
            efficientad_thresholds=final_run / "efficientad_thresholds_v5_rebound.json",
            bright_candidate=root / _BRIGHT_CANDIDATE,
        )
        _write_config_atomically(final_config, v6_config)
    except Exception:
        shutil.rmtree(final_run, ignore_errors=True)
        raise

    return {
        "status": "complete",
        "output_run": str(final_run),
        "output_config": str(final_config),
        "template_thresholds_unchanged": template_unchanged,
        "efficientad_thresholds_unchanged": efficientad_unchanged,
        "yolo_sha256": yolo_sha256,
    }


def _under_root(root: Path, path: Path) -> Path:
    candidate = Path(path).expanduser()
    lexical_root = Path(os.path.abspath(root))
    lexical_path = Path(os.path.abspath(candidate if candidate.is_absolute() else root / candidate))
    try:
        lexical_path.relative_to(lexical_root)
    except ValueError as error:
        raise ValueError(f"V6 output must be inside repo_root: {lexical_path}") from error
    return lexical_path


def _validate_new_training_run(run: Path) -> None:
    report = _load_object(run / "run_report.json", "eight-view training report")
    if report.get("status") != "complete":
        raise ValueError("八视角训练尚未完成")
    missing: list[str] = []
    for view in VIEW_ORDER:
        for label, path in (
            ("Template model.json", run / "template" / view / "model.json"),
            ("Template metrics", run / "template" / view / "metrics.json"),
            ("EfficientAD model.ckpt", run / "efficientad" / view / "model.ckpt"),
            ("EfficientAD metrics", run / "efficientad" / view / "metrics.json"),
        ):
            if not path.is_file():
                missing.append(f"{view}/{label}")
    thresholds = run / "efficientad/score_analysis/part_thresholds.json"
    if not thresholds.is_file():
        missing.append("efficientad/score_analysis/part_thresholds.json")
    if missing:
        raise ValueError(f"八视角训练资产不完整: {', '.join(missing)}")


def _assert_public_roi_equivalent(public_roi: Path, training_roi: Path) -> None:
    public = _load_object(public_roi, "V5 public ROI")
    training = _load_object(training_roi, "new training ROI")
    if public.get("part_rois") != training.get("part_rois"):
        raise ValueError("V5 public ROI坐标与新训练ROI不等价")


def _rebind_template_thresholds(source: Path, new_run: Path) -> dict[str, Any]:
    payload = _load_object(source, "V5 Template thresholds")
    views = _mapping(payload, "views")
    if set(views) != set(VIEW_ORDER):
        raise ValueError("V5 Template thresholds缺少八视角模型身份")
    for view in VIEW_ORDER:
        item = _mapping(views, view)
        item["model_json_sha256"] = _sha256(new_run / "template" / view / "model.json")
    payload.update(
        {
            "thresholds_recalibrated": False,
            "rebound_from": str(source),
            "rebound_from_sha256": _sha256(source),
        }
    )
    return payload


def _rebind_efficientad_thresholds(source: Path, new_run: Path) -> dict[str, Any]:
    payload = _load_object(source, "V5 EfficientAD thresholds")
    checkpoints = _mapping(payload, "checkpoint_sha256_by_view")
    if set(checkpoints) != set(VIEW_ORDER):
        raise ValueError("V5 EfficientAD thresholds缺少八视角模型身份")
    for view in VIEW_ORDER:
        checkpoints[view] = _sha256(new_run / "efficientad" / view / "model.ckpt")
    payload.update(
        {
            "thresholds_recalibrated": False,
            "rebound_from": str(source),
            "rebound_from_sha256": _sha256(source),
        }
    )
    return payload


def _build_v6_config(
    v5_config: dict[str, Any],
    *,
    output_run: Path,
    output_config: Path,
    prepared_manifest: Path,
    template_thresholds: Path,
    efficientad_thresholds: Path,
    bright_candidate: Path,
) -> dict[str, Any]:
    v6 = json.loads(json.dumps(v5_config))
    v6["demo_id"] = "bmw-eight-view-right-v6-normal-20260814-v1"
    v6["training_run"] = _relative_path(output_config.parent, output_run)
    v6["prepared_manifest"] = _relative_path(output_config.parent, prepared_manifest)
    v6["result_root"] = _relative_path(output_config.parent, output_run.parent / "bmw_eight_view_demo_v6_right_normal_20260814_v1")
    template = _mapping(v6, "template")
    template["threshold_artifact"] = _relative_path(output_config.parent, template_thresholds)
    template["threshold_artifact_sha256"] = _sha256(template_thresholds)
    efficientad = _mapping(v6, "efficientad")
    efficientad["threshold_artifact"] = _relative_path(output_config.parent, efficientad_thresholds)
    efficientad["threshold_artifact_sha256"] = _sha256(efficientad_thresholds)
    bright = _mapping(v6, "bright_streak")
    bright["engine"] = "tracked_profile_v3_manual_rotated_candidate"
    bright["config"] = _relative_path(output_config.parent, bright_candidate)
    bright["config_sha256"] = _sha256(bright_candidate)
    bright.pop("weak_row_score_override", None)
    return v6


def _same_numeric_leaves(before: object, after: object) -> bool:
    return _numeric_leaves(before) == _numeric_leaves(after)


def _numeric_leaves(value: object, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], float]:
    if isinstance(value, bool):
        return {}
    if isinstance(value, (int, float)):
        return {prefix: float(value)}
    if isinstance(value, Mapping):
        result: dict[tuple[str, ...], float] = {}
        for key, child in value.items():
            result.update(_numeric_leaves(child, (*prefix, str(key))))
        return result
    if isinstance(value, list):
        result = {}
        for index, child in enumerate(value):
            result.update(_numeric_leaves(child, (*prefix, str(index))))
        return result
    return {}


def _mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"required object is missing: {key}")
    return value


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _resolve_config_path(config: Path, raw: object) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("V5 config path must be a non-empty string")
    candidate = Path(raw).expanduser()
    return (candidate if candidate.is_absolute() else config.parent / candidate).resolve()


def _relative_path(parent: Path, target: Path) -> str:
    return os.path.relpath(target, parent)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _symlink_directory(link: Path, target: Path) -> None:
    if not target.is_dir():
        raise ValueError(f"composite source directory is missing: {target}")
    link.symlink_to(target, target_is_directory=True)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_config_atomically(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".tmp", prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    try:
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
