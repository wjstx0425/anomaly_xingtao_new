# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Atomic publisher for the BMW left-hand Demo composite assets."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER


_LEFT_RUN = Path("results/bmw_lab_one_click/bmw_left_normal_20260814_models_v3")
_SHARED_YOLO_RUN = Path("results/bmw_lab_one_click/bmw_right_multisource_left_yolo_v1")
_SHARED_YOLO_SHA256 = "0e9591f2fa2487ad12000847f1d80137901ed69989e0e95cb5c8907b95ba3913"
_CAPTURE_CONFIG = Path("configs/bmw/capture/bmw_4cam_eight_view_hdr_v1.json")
_LEFT_ROI = Path("configs/bmw/rois/bmw_left_normal_20260814_roi_v2.json")
_LEFT_PREPARED_MANIFEST = Path(
    "dataset/bmw_lab_prepared/bmw_left_normal_20260814_v2/manifests/dataset_manifest.csv"
)
_LEFT_MASK = Path(
    "results/bmw_efficientad_manual_ignore_masks/bmw_left_normal_20260814_models_v3_mask_v1/index.json"
)
_COMPONENT_POLICY = Path("configs/bmw/efficientad_component_filter_lab_v1.json")
_LEFT_THRESHOLDS = Path(
    "results/bmw_lab_one_click/bmw_left_normal_20260814_models_v3/efficientad/"
    "component_score_analysis/part_thresholds.json"
)
_BRIGHT_REPORT = Path(
    "results/bmw_bright_streak_rotated_retrain/bmw_left_normal52_no_streak1_20260814_v1/report.json"
)
_BRIGHT_ROTATED_ROI = Path("results/bmw_bright_streak_rotated_roi/bmw_left_20260814_v2/roi.json")


def publish_left_demo(repo_root: Path, *, output_run: Path, output_config: Path) -> dict[str, object]:
    """Publish one no-overwrite left-hand composite and its loadable Demo config."""
    root = Path(repo_root).expanduser().resolve()
    final_run = _under_root(root, output_run)
    final_config = _under_root(root, output_config)
    if final_run.exists() or final_config.exists():
        existing = final_run if final_run.exists() else final_config
        raise FileExistsError(f"refuse to overwrite existing left Demo destination: {existing}")

    source_run = root / _LEFT_RUN
    yolo_run = root / _SHARED_YOLO_RUN
    _validate_left_training_run(source_run)
    yolo_checkpoint = yolo_run / "yolo/train/weights/best.pt"
    if not yolo_checkpoint.is_file():
        raise ValueError(f"shared YOLO checkpoint is missing: {yolo_checkpoint}")
    yolo_sha256 = _sha256(yolo_checkpoint)
    if yolo_sha256 != _SHARED_YOLO_SHA256:
        raise ValueError("shared YOLO checkpoint SHA256 does not match the approved left Demo asset")

    assets = {
        "capture config": root / _CAPTURE_CONFIG,
        "left ROI": root / _LEFT_ROI,
        "left prepared manifest": root / _LEFT_PREPARED_MANIFEST,
        "left EfficientAD mask": root / _LEFT_MASK,
        "EfficientAD component policy": root / _COMPONENT_POLICY,
        "left EfficientAD thresholds": root / _LEFT_THRESHOLDS,
        "left bright-streak candidate": root / _BRIGHT_REPORT,
        "left bright-streak rotated ROI": root / _BRIGHT_ROTATED_ROI,
    }
    for label, path in assets.items():
        if not path.is_file():
            raise ValueError(f"{label} is missing: {path}")

    final_run.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{final_run.name}.staging-", dir=final_run.parent))
    try:
        _symlink_directory(staging / "template", source_run / "template")
        _symlink_directory(staging / "efficientad", source_run / "efficientad")
        _symlink_directory(staging / "yolo", yolo_run / "yolo")
        _write_json(
            staging / "composition.json",
            {
                "schema": "bmw.left_demo_composition/1.0",
                "template": {"path": str(source_run / "template")},
                "efficientad": {"path": str(source_run / "efficientad")},
                "yolo": {"path": str(yolo_checkpoint), "sha256": yolo_sha256},
                "roi": {"path": str(assets["left ROI"]), "sha256": _sha256(assets["left ROI"])},
                "prepared_manifest": str(assets["left prepared manifest"]),
                "efficientad_deployment": {
                    "mask": str(assets["left EfficientAD mask"]),
                    "mask_sha256": _sha256(assets["left EfficientAD mask"]),
                    "component_policy": str(assets["EfficientAD component policy"]),
                    "component_policy_sha256": _sha256(assets["EfficientAD component policy"]),
                    "thresholds": str(assets["left EfficientAD thresholds"]),
                    "thresholds_sha256": _sha256(assets["left EfficientAD thresholds"]),
                },
                "bright_streak": {
                    "candidate_report": str(assets["left bright-streak candidate"]),
                    "candidate_report_sha256": _sha256(assets["left bright-streak candidate"]),
                    "rotated_roi": str(assets["left bright-streak rotated ROI"]),
                    "rotated_roi_sha256": _sha256(assets["left bright-streak rotated ROI"]),
                },
            },
        )
        staging.rename(final_run)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    try:
        config = _build_config(final_run, final_config, assets)
        _write_config_atomically(final_config, config)
    except Exception:
        shutil.rmtree(final_run, ignore_errors=True)
        raise

    return {
        "status": "complete",
        "output_run": str(final_run),
        "output_config": str(final_config),
        "shared_yolo_sha256": yolo_sha256,
    }


def _under_root(root: Path, path: Path) -> Path:
    candidate = Path(path).expanduser()
    lexical_root = Path(os.path.abspath(root))
    lexical_path = Path(os.path.abspath(candidate if candidate.is_absolute() else root / candidate))
    try:
        lexical_path.relative_to(lexical_root)
    except ValueError as error:
        raise ValueError(f"left Demo output must be inside repo_root: {lexical_path}") from error
    return lexical_path


def _validate_left_training_run(run: Path) -> None:
    report_path = run / "run_report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"left training report is missing or invalid: {report_path}") from error
    if not isinstance(report, dict) or report.get("status") != "complete":
        raise ValueError("left eight-view training is not complete")
    missing = [
        f"{view}/{label}"
        for view in VIEW_ORDER
        for label, path in (
            ("Template model.json", run / "template" / view / "model.json"),
            ("EfficientAD model.ckpt", run / "efficientad" / view / "model.ckpt"),
        )
        if not path.is_file()
    ]
    if missing:
        raise ValueError(f"left eight-view training assets are incomplete: {', '.join(missing)}")


def _build_config(final_run: Path, output_config: Path, assets: dict[str, Path]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "demo_id": "bmw-eight-view-left-normal-20260814-v1",
        "capture_config": _relative_path(output_config.parent, assets["capture config"]),
        "roi_config": _relative_path(output_config.parent, assets["left ROI"]),
        "prepared_manifest": _relative_path(output_config.parent, assets["left prepared manifest"]),
        "training_run": _relative_path(output_config.parent, final_run),
        "result_root": _relative_path(
            output_config.parent, final_run.parent / "bmw_eight_view_demo_left_normal_20260814_v1"
        ),
        "bright_streak": {
            "engine": "tracked_profile_v3_manual_rotated_candidate",
            "config": _relative_path(output_config.parent, assets["left bright-streak candidate"]),
            "config_sha256": _sha256(assets["left bright-streak candidate"]),
            "rotated_roi": _relative_path(output_config.parent, assets["left bright-streak rotated ROI"]),
            "rotated_roi_sha256": _sha256(assets["left bright-streak rotated ROI"]),
        },
        "efficientad": {
            "threshold_artifact": _relative_path(output_config.parent, assets["left EfficientAD thresholds"]),
            "threshold_artifact_sha256": _sha256(assets["left EfficientAD thresholds"]),
            "ignore_mask_index": _relative_path(output_config.parent, assets["left EfficientAD mask"]),
            "ignore_mask_index_sha256": _sha256(assets["left EfficientAD mask"]),
            "component_filter_artifact": _relative_path(output_config.parent, assets["EfficientAD component policy"]),
            "component_filter_artifact_sha256": _sha256(assets["EfficientAD component policy"]),
        },
        "yolo": {"candidate_conf": 0.1, "final_threshold": 0.25, "imgsz": 640},
    }


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
