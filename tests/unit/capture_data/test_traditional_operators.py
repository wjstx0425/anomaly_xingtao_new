# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for C789 traditional operator branch predictions."""

from __future__ import annotations

import csv
import io
import importlib.util
import json
import sys
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np


def load_traditional_module() -> ModuleType:
    """Load the traditional operators module from its file path."""
    script_path = Path(__file__).resolve().parents[3] / "capture_data" / "traditional_operators.py"
    spec = importlib.util.spec_from_file_location("capture_data_traditional_operators", script_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load traditional operators module from {script_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_image(path: Path, image: np.ndarray) -> None:
    """Write an OpenCV fixture image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def _slot_image() -> np.ndarray:
    """Create a simple metal-like slot crop on a dark background."""
    image = np.zeros((160, 260, 3), dtype=np.uint8)
    cv2.rectangle(image, (35, 45), (225, 120), (150, 150, 150), -1)
    cv2.rectangle(image, (65, 65), (90, 90), (5, 5, 5), -1)
    cv2.rectangle(image, (170, 65), (195, 90), (5, 5, 5), -1)
    cv2.rectangle(image, (120, 35), (145, 120), (180, 180, 180), -1)
    return image


def test_traditional_operators_emit_explainable_branch_rows(tmp_path: Path) -> None:
    """Geometry, crack, and registration checks should emit fusion-compatible rows."""
    traditional = load_traditional_module()
    normal_path = tmp_path / "part001_top_uniform_slot01.png"
    missing_edge_path = tmp_path / "part002_top_uniform_slot01.png"
    crack_path = tmp_path / "part003_top_uniform_slot01.png"
    shifted_path = tmp_path / "part004_top_uniform_slot01.png"
    normal = _slot_image()
    missing_edge = normal.copy()
    cv2.rectangle(missing_edge, (35, 45), (105, 70), (0, 0, 0), -1)
    crack = normal.copy()
    cv2.line(crack, (55, 105), (205, 55), (0, 0, 0), 3)
    shifted = np.zeros_like(normal)
    shifted[:, 40:] = normal[:, :-40]
    for path, image in (
        (normal_path, normal),
        (missing_edge_path, missing_edge),
        (crack_path, crack),
        (shifted_path, shifted),
    ):
        _write_image(path, image)

    config = {
        "operators": {
            "registration": {
                "enabled": True,
                "min_area_ratio": 0.15,
                "max_area_ratio": 0.7,
                "expected_center_x_ratio": 0.5,
                "expected_center_y_ratio": 0.52,
                "max_center_shift_ratio": 0.08,
                "mode": "fail",
            },
            "geometry": {"enabled": True, "min_area_ratio": 0.31},
            "crack": {"enabled": True, "min_length_ratio": 0.35, "dark_delta": 55},
            "surface_texture": {"enabled": False},
            "feature_presence": {"enabled": False},
        },
    }

    results = []
    for path in (normal_path, missing_edge_path, crack_path, shifted_path):
        results.extend(
            traditional.evaluate_slot_image(
                path,
                side="top",
                view="uniform",
                slot_id="slot01",
                config=config,
                evidence_dir=tmp_path / "evidence",
            ),
        )

    traditional.write_branch_csv(results, tmp_path / "traditional_predictions.csv")

    positives = {(result.part_id, result.branch) for result in results if result.pred_label == 1}
    assert ("part002", "geometry") in positives
    assert ("part003", "crack") in positives
    assert ("part004", "registration") in positives
    with (tmp_path / "traditional_predictions.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert {"part_id", "branch", "reason", "evidence_path"} <= set(rows[0])
    assert any(row["evidence_path"] and Path(row["evidence_path"]).is_file() for row in rows)


def test_registration_warn_does_not_emit_blocking_label(tmp_path: Path) -> None:
    """Registration WARN should be visible without forcing RETAKE in permissive fusion."""
    traditional = load_traditional_module()
    image_path = tmp_path / "part001_top_uniform_slot01.png"
    image = np.zeros((160, 260, 3), dtype=np.uint8)
    image[:, 50:] = _slot_image()[:, :-50]
    _write_image(image_path, image)
    config = {
        "operators": {
            "registration": {
                "enabled": True,
                "mode": "warn",
                "expected_center_x_ratio": 0.5,
                "expected_center_y_ratio": 0.52,
                "max_center_shift_ratio": 0.05,
            },
            "geometry": {"enabled": False},
            "crack": {"enabled": False},
            "surface_texture": {"enabled": False},
            "feature_presence": {"enabled": False},
        },
    }

    results = traditional.evaluate_slot_image(
        image_path,
        side="top",
        view="uniform",
        slot_id="slot01",
        config=config,
        evidence_dir=tmp_path / "evidence",
    )

    assert results[0].branch == "registration"
    assert results[0].status == "WARN"
    assert results[0].pred_label == 0


def test_run_traditional_operators_writes_progress_and_error_summary(tmp_path: Path) -> None:
    """Stage 20 should report progress and summarize false positives/false negatives."""
    traditional = load_traditional_module()
    normal_path = tmp_path / "normal" / "part001_top_uniform_slot01.png"
    defect_path = tmp_path / "defect" / "part002_top_uniform_slot01.png"
    normal = _slot_image()
    defect = normal.copy()
    cv2.rectangle(defect, (35, 45), (105, 70), (0, 0, 0), -1)
    _write_image(normal_path, normal)
    _write_image(defect_path, defect)
    config_path = tmp_path / "traditional.json"
    config_path.write_text(
        json.dumps(
            {
                "operators": {
                    "registration": {"enabled": False},
                    "geometry": {"enabled": True, "min_area_ratio": 0.31},
                    "crack": {"enabled": False},
                    "surface_texture": {"enabled": False},
                    "feature_presence": {"enabled": False},
                },
            },
        ),
        encoding="utf-8",
    )
    args = Namespace(
        input_root=tmp_path,
        preset="c789_left_top_3x2",
        side="top",
        view="uniform",
        input_mode="slot",
        config=config_path,
        output_dir=tmp_path / "out",
        max_images=None,
        save_evidence=False,
        show_progress=True,
        label="auto",
    )

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        traditional.run_traditional_operators(args)

    summary_csv = args.output_dir / "traditional_summary.csv"
    summary_md = args.output_dir / "traditional_summary.md"
    assert "[traditional] 1/2" in stdout.getvalue()
    assert summary_csv.is_file()
    assert summary_md.is_file()
    with summary_csv.open(newline="", encoding="utf-8") as file:
        summary = {row["metric"]: row["value"] for row in csv.DictReader(file)}
    assert summary["normal_total"] == "1"
    assert summary["normal_false_positive"] == "0"
    assert summary["defect_total"] == "1"
    assert summary["defect_false_negative"] == "0"
    assert "false_positive_rate" in summary


def test_calibrated_slot_template_geometry_detects_less_and_more(tmp_path: Path) -> None:
    """Normal-slot calibration should turn template differences into geometry positives."""
    traditional = load_traditional_module()
    normal_path = tmp_path / "normal" / "part001_slot01" / "images" / "normal_slot01.png"
    less_path = tmp_path / "defect" / "part002_slot01" / "images" / "less_slot01.png"
    more_path = tmp_path / "defect" / "part003_slot01" / "images" / "more_slot01.png"
    normal = _slot_image()
    less = normal.copy()
    cv2.rectangle(less, (35, 45), (95, 82), (0, 0, 0), -1)
    more = normal.copy()
    cv2.rectangle(more, (12, 74), (42, 118), (160, 160, 160), -1)
    for path, image in ((normal_path, normal), (less_path, less), (more_path, more)):
        _write_image(path, image)
    config_path = tmp_path / "traditional.json"
    config_path.write_text(
        json.dumps(
            {
                "operators": {
                    "registration": {"enabled": False},
                    "geometry": {"enabled": True, "threshold_margin": 0.05, "min_component_area": 16},
                    "crack": {"enabled": False},
                    "surface_texture": {"enabled": False},
                    "feature_presence": {"enabled": False},
                },
            },
        ),
        encoding="utf-8",
    )
    args = Namespace(
        input_root=tmp_path / "defect",
        preset="c789_left_top_3x2",
        side="top",
        view="uniform",
        input_mode="slot",
        config=config_path,
        output_dir=tmp_path / "out",
        max_images=None,
        save_evidence=True,
        show_progress=False,
        label="auto",
        calibrate_normal_root=tmp_path / "normal",
        template_dir=None,
        geometry_thresholds=None,
    )

    results = traditional.run_traditional_operators(args)

    positives = {Path(result.source_path).name: result for result in results if result.branch == "geometry"}
    assert positives["less_slot01.png"].pred_label == 1
    assert positives["less_slot01.png"].defect_type == "geometry_delta"
    assert positives["less_slot01.png"].evidence_type == "missing_mask"
    assert positives["less_slot01.png"].gt_defect_type == "less"
    assert positives["more_slot01.png"].pred_label == 1
    assert positives["more_slot01.png"].defect_type == "geometry_delta"
    assert positives["more_slot01.png"].evidence_type == "extra_mask"
    assert positives["more_slot01.png"].gt_defect_type == "more"
    assert all(Path(result.evidence_path).is_file() for result in positives.values())
    with (args.output_dir / "traditional_predictions.csv").open(newline="", encoding="utf-8") as file:
        rows = {Path(row["source_path"]).name: row for row in csv.DictReader(file) if row["branch"] == "geometry"}
    assert rows["less_slot01.png"]["defect_type"] == "geometry_delta"
    assert rows["less_slot01.png"]["evidence_type"] == "missing_mask"
    assert rows["less_slot01.png"]["gt_defect_type"] == "less"
    assert "raw_delta=less" in rows["less_slot01.png"]["reason"]
    assert "type=less" not in rows["less_slot01.png"]["reason"]


def test_traditional_summary_reports_gt_vs_evidence_type(tmp_path: Path) -> None:
    """Stage 20 summaries should separate file-name GT type from operator evidence type."""
    traditional = load_traditional_module()
    normal_path = tmp_path / "normal" / "part001_slot01" / "images" / "normal_slot01.png"
    surface_path = tmp_path / "defect" / "part002_slot01" / "images" / "surface_slot01.png"
    crack_path = tmp_path / "defect" / "part003_slot01" / "images" / "crack_slot01.png"
    normal = _slot_image()
    surface = normal.copy()
    cv2.rectangle(surface, (35, 45), (95, 82), (0, 0, 0), -1)
    crack = normal.copy()
    cv2.line(crack, (55, 104), (210, 58), (0, 0, 0), 3)
    for path, image in ((normal_path, normal), (surface_path, surface), (crack_path, crack)):
        _write_image(path, image)
    config_path = tmp_path / "traditional.json"
    config_path.write_text(
        json.dumps(
            {
                "operators": {
                    "registration": {"enabled": False},
                    "geometry": {"enabled": True, "threshold_margin": 0.05, "min_component_area": 16},
                    "crack": {"enabled": False},
                    "surface_texture": {"enabled": False},
                    "feature_presence": {"enabled": False},
                },
            },
        ),
        encoding="utf-8",
    )
    args = Namespace(
        input_root=tmp_path / "defect",
        preset="c789_left_top_3x2",
        side="top",
        view="uniform",
        input_mode="slot",
        config=config_path,
        output_dir=tmp_path / "out",
        max_images=None,
        save_evidence=False,
        show_progress=False,
        label="auto",
        calibrate_normal_root=tmp_path / "normal",
        template_dir=None,
        geometry_thresholds=None,
    )

    traditional.run_traditional_operators(args)

    summary_md = (args.output_dir / "traditional_summary.md").read_text(encoding="utf-8")
    assert "GT Defect Type vs Primary Evidence" in summary_md
    assert "| `surface` | `geometry:missing_mask` | 1 |" in summary_md
    confusion_csv = args.output_dir / "traditional_defect_evidence_confusion.csv"
    with confusion_csv.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert {
        "gt_defect_type": "surface",
        "primary_evidence": "geometry:missing_mask",
        "count": "1",
    } in rows


def test_calibrated_crack_scores_only_template_material_roi(tmp_path: Path) -> None:
    """Crack scoring should ignore dark lines outside the calibrated material ROI."""
    traditional = load_traditional_module()
    normal_path = tmp_path / "normal" / "part001_slot01" / "images" / "normal_slot01.png"
    inside_path = tmp_path / "defect" / "part002_slot01" / "images" / "crack_inside_slot01.png"
    outside_path = tmp_path / "defect" / "part003_slot01" / "images" / "crack_outside_slot01.png"
    normal = _slot_image()
    inside = normal.copy()
    cv2.line(inside, (55, 104), (210, 58), (0, 0, 0), 3)
    outside = normal.copy()
    cv2.line(outside, (5, 8), (245, 18), (0, 0, 0), 4)
    for path, image in ((normal_path, normal), (inside_path, inside), (outside_path, outside)):
        _write_image(path, image)
    config_path = tmp_path / "traditional.json"
    config_path.write_text(
        json.dumps(
            {
                "operators": {
                    "registration": {"enabled": False},
                    "geometry": {"enabled": False},
                    "crack": {
                        "enabled": True,
                        "threshold_margin": 0.05,
                        "min_length_ratio": 0.05,
                        "dark_delta": 40,
                        "min_component_area": 8,
                    },
                    "surface_texture": {"enabled": False},
                    "feature_presence": {"enabled": False},
                },
            },
        ),
        encoding="utf-8",
    )
    args = Namespace(
        input_root=tmp_path / "defect",
        preset="c789_left_top_3x2",
        side="top",
        view="uniform",
        input_mode="slot",
        config=config_path,
        output_dir=tmp_path / "out",
        max_images=None,
        save_evidence=False,
        show_progress=False,
        label="auto",
        calibrate_normal_root=tmp_path / "normal",
        template_dir=None,
        geometry_thresholds=None,
    )

    results = traditional.run_traditional_operators(args)

    by_name = {Path(result.source_path).name: result for result in results if result.branch == "crack"}
    assert by_name["crack_inside_slot01.png"].pred_label == 1
    assert by_name["crack_outside_slot01.png"].pred_label == 0


def test_calibrated_surface_texture_positive_is_suspect(tmp_path: Path) -> None:
    """Surface texture positives should be review signals rather than hard NG rows."""
    traditional = load_traditional_module()
    normal_path = tmp_path / "normal" / "part001_slot01" / "images" / "normal_slot01.png"
    surface_path = tmp_path / "defect" / "part002_slot01" / "images" / "surface_slot01.png"
    normal = _slot_image()
    surface = normal.copy()
    surface[68:112, 70:170] = np.random.default_rng(0).integers(70, 230, size=(44, 100, 3), dtype=np.uint8)
    _write_image(normal_path, normal)
    _write_image(surface_path, surface)
    config_path = tmp_path / "traditional.json"
    config_path.write_text(
        json.dumps(
            {
                "operators": {
                    "registration": {"enabled": False},
                    "geometry": {"enabled": False},
                    "crack": {"enabled": False},
                    "surface_texture": {"enabled": True, "threshold_margin": 0.05},
                    "feature_presence": {"enabled": False},
                },
            },
        ),
        encoding="utf-8",
    )
    args = Namespace(
        input_root=tmp_path / "defect",
        preset="c789_left_top_3x2",
        side="top",
        view="uniform",
        input_mode="slot",
        config=config_path,
        output_dir=tmp_path / "out",
        max_images=None,
        save_evidence=False,
        show_progress=False,
        label="auto",
        calibrate_normal_root=tmp_path / "normal",
        template_dir=None,
        geometry_thresholds=None,
    )

    results = traditional.run_traditional_operators(args)

    surface_result = next(result for result in results if result.branch == "surface_texture")
    assert surface_result.pred_label == 1
    assert surface_result.status == "SUSPECT"
