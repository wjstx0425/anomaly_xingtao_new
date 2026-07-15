# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CLI contract tests for Stage 33 commissioning calibration."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


def _load_stage33() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "pipeline/33_run_zs32_offline_calibration.py"
    spec = importlib.util.spec_from_file_location("pipeline_stage33", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_stage33_parser_requires_explicit_commissioning_assets(tmp_path: Path) -> None:
    """The batch CLI must keep every loose asset path explicit and default to GPU 0."""
    stage33 = _load_stage33()
    args = stage33.build_parser().parse_args(
        [
            "--crop-manifest",
            str(tmp_path / "crop.csv"),
            "--template-calibration-csv",
            str(tmp_path / "template.csv"),
            "--template-model-dir",
            str(tmp_path / "template-model"),
            "--runtime-config",
            str(tmp_path / "runtime.json"),
            "--yolo-dataset-root",
            str(tmp_path / "yolo-dataset"),
            "--path-root",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "output"),
            "--commissioning-only",
        ],
    )

    assert args.accelerator == "gpu"
    assert args.devices == 1
    assert args.yolo_device == "0"
    assert args.target_recall == pytest.approx(1.0)
    assert args.normal_quantile == pytest.approx(0.995)
    assert args.yolo_aux_min_image_precision is None
    assert args.yolo_aux_use_test_for_selection is False
    assert args.reuse_existing_inference is False
    assert args.resume is False


def test_stage33_parser_accepts_explicit_yolo_auxiliary_precision(tmp_path: Path) -> None:
    """The optional auxiliary policy must be explicit and independently configurable."""
    stage33 = _load_stage33()
    args = stage33.build_parser().parse_args(
        [
            "--crop-manifest",
            str(tmp_path / "crop.csv"),
            "--template-calibration-csv",
            str(tmp_path / "template.csv"),
            "--template-model-dir",
            str(tmp_path / "template-model"),
            "--runtime-config",
            str(tmp_path / "runtime.json"),
            "--yolo-dataset-root",
            str(tmp_path / "yolo-dataset"),
            "--path-root",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "output"),
            "--commissioning-only",
            "--yolo-aux-min-image-precision",
            "0.9",
            "--yolo-aux-use-test-for-selection",
            "--reuse-existing-inference",
        ],
    )

    assert args.yolo_aux_min_image_precision == pytest.approx(0.9)
    assert args.yolo_aux_use_test_for_selection is True
    assert args.reuse_existing_inference is True


def test_threshold_quality_rejects_zero_yolo_thresholds_and_tiny_test_set() -> None:
    """An algorithm-valid fit must still fail the commissioning usability gate."""
    stage33 = _load_stage33()
    report = stage33.build_threshold_quality_report(
        {
            "thresholds": [
                {
                    "view": "front_left",
                    "branch": "yolo",
                    "low_threshold": 0.0,
                    "high_threshold": 0.0,
                },
            ],
        },
        {
            "overall": {
                "calibration_valid": True,
                "normal_reject_rate": 1.0,
                "non_clear_recall": 1.0,
                "defect_part_count": 4,
            },
        },
    )

    assert report["algorithm_calibration_valid"] is True
    assert report["usable_for_runtime_threshold_injection"] is False
    assert report["yolo_zero_high_views"] == ["front_left"]
    assert len(report["rejection_reasons"]) == 6


def test_threshold_quality_rejects_invalid_fit_and_low_recall() -> None:
    """Structural-looking thresholds cannot hide an invalid or low-recall fit."""
    stage33 = _load_stage33()
    report = stage33.build_threshold_quality_report(
        {
            "thresholds": [
                {
                    "view": "front",
                    "branch": "yolo",
                    "low_threshold": None,
                    "high_threshold": None,
                },
            ],
        },
        {
            "overall": {
                "calibration_valid": False,
                "normal_reject_rate": 0.0,
                "non_clear_recall": 0.75,
                "defect_part_count": 20,
            },
        },
        target_recall=1.0,
    )

    assert report["usable_for_runtime_threshold_injection"] is False
    assert report["yolo_invalid_threshold_views"] == ["front"]
    assert "Stage31 algorithm calibration is invalid" in report["rejection_reasons"]
    assert any("recall is below target" in reason for reason in report["rejection_reasons"])


def test_prepare_output_contract_rejects_resume_drift(tmp_path: Path) -> None:
    """Resume must bind an existing output directory to identical inputs and parameters."""
    stage33 = _load_stage33()
    output = tmp_path / "output"
    payload = {"schema_version": "1.0", "runtime_config_sha256": "a" * 64}

    path = stage33.prepare_output_contract(output, payload, resume=False)
    resumed_path = stage33.prepare_output_contract(output, payload, resume=True)

    assert resumed_path == path
    with pytest.raises(ValueError, match="differs from current inputs"):
        stage33.prepare_output_contract(
            output,
            {**payload, "runtime_config_sha256": "b" * 64},
            resume=True,
        )


def test_prepare_output_contract_rejects_legacy_directory_without_contract(tmp_path: Path) -> None:
    """Unproven legacy outputs cannot be silently adopted by resume."""
    stage33 = _load_stage33()
    output = tmp_path / "legacy"
    output.mkdir()

    with pytest.raises(ValueError, match="no commissioning run contract"):
        stage33.prepare_output_contract(output, {"schema_version": "1.0"}, resume=True)


def test_stage33_rejects_runtime_template_roi_version_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Published runtime and Template assets must identify one ROI generation."""
    stage33 = _load_stage33()
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    (template_dir / "model.json").write_text("{}", encoding="utf-8")
    crop_manifest = tmp_path / "crop.csv"
    template_csv = tmp_path / "template.csv"
    runtime_config = tmp_path / "runtime.json"
    for path in (crop_manifest, template_csv, runtime_config):
        path.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(
        stage33,
        "load_model",
        lambda _path: {"versions": {"roi": "template-roi-v1", "template": "template-v1"}},
    )
    monkeypatch.setattr(
        stage33,
        "load_runtime_config",
        lambda _path: SimpleNamespace(
            versions=SimpleNamespace(
                patchcore_roi="runtime-roi-v2",
                yolo_roi="runtime-roi-v2",
                template="template-v1",
            ),
        ),
    )
    monkeypatch.setattr(stage33, "validate_yolo_annotation_dataset", lambda *_args: None)
    monkeypatch.setattr(stage33, "_yolo_dataset_contract", lambda _path: {})
    args = SimpleNamespace(
        template_model_dir=template_dir,
        runtime_config=runtime_config,
        crop_manifest=crop_manifest,
        template_calibration_csv=template_csv,
        yolo_dataset_root=tmp_path / "yolo",
        hand="right",
        target_recall=1.0,
        normal_quantile=0.995,
    )

    with pytest.raises(ValueError, match="ROI version mismatch"):
        stage33.build_run_contract(args, ())


def test_stage33_calibration_requires_secondary_normal_and_defect_rows(tmp_path: Path) -> None:
    """The Stage31 adapter must not fall back to its historical six-view default."""
    stage33 = _load_stage33()
    input_csv = tmp_path / "rows.csv"
    fields = (
        "part_id",
        "hand",
        "view",
        "branch",
        "raw_score",
        "gt_label",
        "split",
        "model_version",
        "roi_version",
    )
    primary_views = ("front", "front_left", "front_right", "back", "back_left", "back_right")
    rows = [
        {
            "part_id": f"{split}-{label}",
            "hand": "right",
            "view": view,
            "branch": "template_match",
            "raw_score": str(label),
            "gt_label": str(label),
            "split": split,
            "model_version": "template-v1",
            "roi_version": "roi-v1",
        }
        for split in ("calibration", "test")
        for label in (0, 1)
        for view in primary_views
    ]
    with input_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    args = SimpleNamespace(target_recall=1.0, normal_quantile=0.995, resume=False)

    with pytest.raises(ValueError, match="secondary"):
        stage33._run_or_validate_calibration(input_csv, tmp_path / "thresholds", args)  # noqa: SLF001
