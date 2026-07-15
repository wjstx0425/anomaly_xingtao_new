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


def _contract_fixture(
    stage33: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    runtime_roi_version: str = "roi-v1",
) -> tuple[SimpleNamespace, dict[str, object], dict[str, object]]:
    """Build one fully bound Task2 assets fixture for Stage33 contract tests."""
    runtime_dir = tmp_path / "runtime-publication"
    runtime_dir.mkdir()
    runtime_config = runtime_dir / "runtime_assets.json"
    assets_manifest = runtime_dir / "assets_manifest.json"
    roi_config = tmp_path / "roi.json"
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    template_model = template_dir / "model.json"
    crop_manifest = tmp_path / "crop.csv"
    template_csv = tmp_path / "template.csv"
    for path, content in (
        (runtime_config, "runtime"),
        (assets_manifest, "manifest"),
        (roi_config, "roi"),
        (template_model, "template"),
        (crop_manifest, "crop"),
        (template_csv, "template-csv"),
    ):
        path.write_text(content, encoding="utf-8")
    versions = {
        "model": "template-model-v1",
        "threshold": "threshold-v1",
        "roi": "roi-v1",
        "template": "template-generation-v1",
    }
    model = {
        "required_hands": ["right"],
        "required_views": list(stage33.VIEW_ORDER),
        "groups": {f"right/{view}": {} for view in stage33.VIEW_ORDER},
        "versions": versions,
    }
    config = SimpleNamespace(
        path=runtime_config.resolve(),
        patchcore_roi_config=roi_config.resolve(),
        yolo_roi_config=roi_config.resolve(),
        versions=SimpleNamespace(
            patchcore_roi=runtime_roi_version,
            yolo_roi=runtime_roi_version,
            template=versions["template"],
        ),
    )
    manifest = {
        "asset_set_sha256": "c" * 64,
        "runtime_assets": {"path": str(runtime_config.resolve()), "sha256": stage33._sha256(runtime_config)},
        "roi_config": {"path": str(roi_config.resolve()), "sha256": stage33._sha256(roi_config)},
        "template_model": {
            "path": str(template_dir.resolve()),
            "model_json": {"path": str(template_model.resolve()), "sha256": stage33._sha256(template_model)},
        },
    }
    monkeypatch.setattr(stage33, "load_model", lambda _path: model)
    monkeypatch.setattr(stage33, "load_runtime_config", lambda _path: config)
    monkeypatch.setattr(stage33, "validate_yolo_annotation_dataset", lambda *_args: None)
    monkeypatch.setattr(stage33, "_yolo_dataset_contract", lambda _path: {})
    monkeypatch.setattr(stage33, "load_runtime_assets_manifest", lambda _path: manifest)
    monkeypatch.setattr(stage33, "_template_calibration_versions", lambda *_args: versions, raising=False)
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
    return args, model, manifest


def test_stage33_rejects_runtime_template_roi_version_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Published runtime and Template assets must identify one ROI generation."""
    stage33 = _load_stage33()
    args, _, _ = _contract_fixture(stage33, tmp_path, monkeypatch, runtime_roi_version="runtime-roi-v2")

    with pytest.raises(ValueError, match="ROI version mismatch"):
        stage33.build_run_contract(args, ())


@pytest.mark.parametrize("mutation", ["hands", "group_order"])
def test_stage33_requires_exact_right_eight_template_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Strict Stage33 must reject generic or reordered Template models."""
    stage33 = _load_stage33()
    args, model, _ = _contract_fixture(stage33, tmp_path, monkeypatch)
    if mutation == "hands":
        model["required_hands"] = ["left", "right"]
    else:
        model["groups"] = dict(reversed(tuple(model["groups"].items())))

    with pytest.raises(ValueError, match="exact right-hand eight-view Template"):
        stage33.build_run_contract(args, ())


def test_stage33_run_contract_binds_task2_manifest_asset_set_and_roi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resume provenance must bind the validated Task2 generation and physical ROI bytes."""
    stage33 = _load_stage33()
    args, _, manifest = _contract_fixture(stage33, tmp_path, monkeypatch)

    contract = stage33.build_run_contract(args, ())

    assert contract["assets_manifest"] == str((args.runtime_config.parent / "assets_manifest.json").resolve())
    assert contract["assets_manifest_sha256"] == stage33._sha256(args.runtime_config.parent / "assets_manifest.json")
    assert contract["asset_set_sha256"] == manifest["asset_set_sha256"]
    assert contract["roi_config"] == manifest["roi_config"]
    assert contract["template_model_binding"] == manifest["template_model"]["model_json"]


def test_stage33_rejects_roi_bytes_drifting_from_task2_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Matching ROI version strings cannot hide changed physical ROI bytes."""
    stage33 = _load_stage33()
    args, _, manifest = _contract_fixture(stage33, tmp_path, monkeypatch)
    Path(manifest["roi_config"]["path"]).write_text("drifted-roi", encoding="utf-8")

    with pytest.raises(ValueError, match="ROI.*SHA-256"):
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


def test_stage33_calibration_publication_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stage33 = _load_stage33()
    input_csv = tmp_path / "rows.csv"
    with input_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("view", "gt_label", "split"))
        writer.writeheader()
        writer.writerows(
            {"view": view, "gt_label": str(label), "split": "calibration"}
            for view in stage33.VIEW_ORDER
            for label in (0, 1)
        )
    output = tmp_path / "thresholds"

    def fail(_input: Path, report_dir: Path, **_kwargs: object) -> dict[str, object]:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "thresholds.json").write_text("{}", encoding="utf-8")
        raise RuntimeError("injected")

    monkeypatch.setattr(stage33, "run_calibration", fail)
    args = SimpleNamespace(target_recall=1.0, normal_quantile=0.995, resume=False)

    with pytest.raises(RuntimeError, match="injected"):
        stage33._run_or_validate_calibration(input_csv, output, args)  # noqa: SLF001

    assert not output.exists()
    assert not tuple(tmp_path.glob(".thresholds.*.tmp"))


def test_preflight_rejects_reuse_without_existing_cases_before_gpu(tmp_path: Path) -> None:
    stage33 = _load_stage33()
    calibration = tmp_path / "template.csv"
    with calibration.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("view", "gt_label", "split"))
        writer.writeheader()
        writer.writerows(
            {"view": view, "gt_label": str(label), "split": "calibration"}
            for view in stage33.VIEW_ORDER
            for label in (0, 1)
        )
    args = SimpleNamespace(
        template_calibration_csv=calibration,
        output_dir=tmp_path / "missing",
        reuse_existing_inference=True,
        resume=True,
    )

    with pytest.raises(ValueError, match="requires an existing"):
        stage33.validate_preflight(args, (), {}, object())
