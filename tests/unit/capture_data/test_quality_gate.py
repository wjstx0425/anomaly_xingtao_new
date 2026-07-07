# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for rule-based image quality gate metrics."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np


def load_quality_module() -> ModuleType:
    """Load the quality gate module from its file path."""
    script_path = Path(__file__).resolve().parents[3] / "capture_data" / "quality_gate.py"
    spec = importlib.util.spec_from_file_location("capture_data_quality_gate", script_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load quality gate module from {script_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_image(path: Path, image: np.ndarray) -> None:
    """Write an OpenCV image fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def _checkerboard(size: int = 96) -> np.ndarray:
    """Return a sharp, mid-brightness BGR checkerboard."""
    yy, xx = np.indices((size, size))
    gray = np.where((xx // 8 + yy // 8) % 2 == 0, 90, 170).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_synthetic_quality_gate_warns_bad_inputs(tmp_path: Path) -> None:
    """Synthetic images should separate normal, saturated, dark, and blurred inputs."""
    quality = load_quality_module()
    config = {
        "mode": "warn",
        "metrics": {
            "brightness_mean": {"min": 30.0, "max": 230.0},
            "saturation_ratio": {"max": 0.02},
            "dark_ratio": {"max": 0.05},
            "blur_laplacian_var": {"min": 80.0},
        },
    }
    normal_path = tmp_path / "part001_top_uniform.png"
    white_path = tmp_path / "part001_top_overexposed.png"
    dark_path = tmp_path / "part001_top_dark.png"
    blur_path = tmp_path / "part001_top_blur.png"
    _write_image(normal_path, _checkerboard())
    _write_image(white_path, np.full((96, 96, 3), 255, dtype=np.uint8))
    _write_image(dark_path, np.zeros((96, 96, 3), dtype=np.uint8))
    _write_image(blur_path, np.full((96, 96, 3), 120, dtype=np.uint8))

    normal = quality.evaluate_quality_gate(quality.compute_quality_metrics(normal_path), config)
    saturated = quality.evaluate_quality_gate(quality.compute_quality_metrics(white_path), config)
    dark = quality.evaluate_quality_gate(quality.compute_quality_metrics(dark_path), config)
    blurred = quality.evaluate_quality_gate(
        quality.compute_quality_metrics(blur_path),
        {"mode": "warn", "metrics": {"blur_laplacian_var": {"min": 80.0}}},
    )

    assert normal.status == "PASS"
    assert saturated.status == "WARN"
    assert any("saturation_ratio" in reason for reason in saturated.reasons)
    assert dark.status == "WARN"
    assert any("dark_ratio" in reason or "brightness_mean" in reason for reason in dark.reasons)
    assert blurred.status == "WARN"
    assert any("blur_laplacian_var" in reason for reason in blurred.reasons)


def test_write_quality_gate_csv_contains_fusion_fields(tmp_path: Path) -> None:
    """Quality CSV should be directly consumable by the fusion branch loader."""
    quality = load_quality_module()
    image_path = tmp_path / "part001_top_uniform.png"
    _write_image(image_path, _checkerboard())
    metrics = quality.compute_quality_metrics(image_path)
    result = quality.evaluate_quality_gate(
        metrics,
        {"mode": "warn", "metrics": {"blur_laplacian_var": {"min": metrics.blur_laplacian_var + 1.0}}},
    )
    output_csv = tmp_path / "quality_gate.csv"

    quality.write_quality_gate_csv([result], output_csv)

    with output_csv.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["branch"] == "quality_gate"
    assert rows[0]["status"] == "WARN"
    assert rows[0]["fail_label"] == "0"
    assert "blur_laplacian_var" in rows[0]["reason"]
    assert rows[0]["source_path"] == str(image_path.resolve())


def test_quality_gate_fail_mode_blocks_saturated_inputs(tmp_path: Path) -> None:
    """Fail mode should emit FAIL for synthetic overexposed images."""
    quality = load_quality_module()
    white_path = tmp_path / "part001_top_overexposed.png"
    _write_image(white_path, np.full((96, 96, 3), 255, dtype=np.uint8))

    result = quality.evaluate_quality_gate(
        quality.compute_quality_metrics(white_path),
        {"mode": "fail", "metrics": {"saturation_ratio": {"max": 0.02}}},
    )

    assert result.status == "FAIL"
    assert any("saturation_ratio" in reason for reason in result.reasons)


def test_evaluate_image_quality_returns_fail_for_unreadable_inputs(tmp_path: Path) -> None:
    """Unreadable images should become explicit FAIL quality results."""
    quality = load_quality_module()
    missing_path = tmp_path / "missing.png"

    result = quality.evaluate_image_quality(missing_path, {"mode": "warn"})

    assert result.status == "FAIL"
    assert "Could not read image" in result.reasons[0]


def test_calibrate_quality_config_uses_normal_distribution() -> None:
    """Calibration should build warn-mode thresholds from normal/stress metrics only."""
    quality = load_quality_module()
    metrics = [
        quality.ImageQualityMetrics(
            image_path="a.png",
            side="top",
            view="uniform",
            brightness_mean=100.0,
            brightness_std=20.0,
            saturation_ratio=0.0,
            dark_ratio=0.0,
            blur_laplacian_var=100.0,
            highlight_ratio=0.01,
        ),
        quality.ImageQualityMetrics(
            image_path="b.png",
            side="top",
            view="uniform",
            brightness_mean=120.0,
            brightness_std=25.0,
            saturation_ratio=0.01,
            dark_ratio=0.02,
            blur_laplacian_var=150.0,
            highlight_ratio=0.02,
        ),
    ]

    config = quality.calibrate_quality_config(metrics, mode="warn", margin_ratio=0.0)

    assert config["mode"] == "warn"
    assert config["metrics"]["brightness_mean"] == {"min": 100.0, "max": 120.0}
    assert config["metrics"]["blur_laplacian_var"]["min"] == 100.0
    assert config["metrics"]["saturation_ratio"]["max"] == 0.01
