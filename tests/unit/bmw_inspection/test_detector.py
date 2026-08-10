"""Regression tests for the offline BMW bright-streak detector."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from bmw_inspection.contracts import DemoStatus, load_config
from bmw_inspection.detector import (
    detect_bright_streak,
    detect_bright_streak_evidence,
    render_evidence,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEMO_CONFIG = REPO_ROOT / "configs/bmw/bright_streak_demo.json"
OK_ROOT = REPO_ROOT / "dataset/bmw/OK"
NG_ROOT = REPO_ROOT / "dataset/bmw/NG"


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert image is not None
    return image


def _write_synthetic_config(path: Path) -> None:
    payload = {
        "schema_version": 1,
        "mode": "demo",
        "camera_serial": "DA9625347",
        "image_width": 160,
        "image_height": 120,
        "exposure": 4000.0,
        "gain": 0.0,
        "timeout_ms": 3000,
        "warmup_frames": 1,
        "roi_xyxy": [60, 10, 100, 110],
        "result_root": "results/test-only",
        "thresholds": {
            "min_mean_intensity": 20.0,
            "max_mean_intensity": 245.0,
            "max_dark_clip_ratio": 0.25,
            "max_bright_clip_ratio": 0.10,
            "min_laplacian_variance": 1.0,
            "background_kernel_px": 15,
            "response_mad_scale": 4.0,
            "min_component_area_px": 8,
            "min_component_width_px": 1.0,
            "max_component_width_px": 12.0,
            "center_tolerance_px": 8.0,
            "micro_gap_close_px": 2,
            "min_contrast_snr": 4.0,
            "min_coverage_ratio": 0.40,
            "min_longest_run_ratio": 0.45,
            "max_gap_ratio": 0.10,
            "max_gap_count": 1,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _broken_streak_image() -> np.ndarray:
    yy, xx = np.indices((120, 160))
    image = (76 + ((3 * xx + 5 * yy) % 9)).astype(np.uint8)
    image[20:48, 78:82] = 210
    image[68:98, 78:82] = 210
    return image


def _curved_streak_image() -> np.ndarray:
    yy, xx = np.indices((120, 160))
    image = (76 + ((3 * xx + 5 * yy) % 9)).astype(np.uint8)
    for y in range(15, 105):
        center = 80 + int(round(5 * np.sin(y / 18.0)))
        half_width = 1 + int(y % 17 == 0)
        image[y, center - half_width : center + half_width + 1] = 210
    return image


def _gradually_drifting_streak_image() -> np.ndarray:
    yy, xx = np.indices((120, 160))
    image = (76 + ((3 * xx + 5 * yy) % 9)).astype(np.uint8)
    for y in range(15, 105):
        center = 73 + int(round(14 * (y - 15) / 89))
        image[y, center - 1 : center + 2] = 210
    return image


def test_curved_variable_width_streak_is_tracked(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_synthetic_config(config_path)
    decision = detect_bright_streak_evidence(_curved_streak_image(), load_config(config_path))
    assert decision.status is DemoStatus.OK
    assert decision.metrics is not None
    assert decision.metrics.coverage_ratio >= 0.75


def test_gradual_drift_beyond_width_allowance_is_tracked(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_synthetic_config(config_path)
    decision = detect_bright_streak_evidence(
        _gradually_drifting_streak_image(),
        load_config(config_path),
    )
    assert decision.status is DemoStatus.OK
    assert decision.metrics is not None
    assert decision.metrics.coverage_ratio >= 0.75


def test_unstructured_center_texture_is_not_a_streak(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_synthetic_config(config_path)
    image = np.full((120, 160), 80, dtype=np.uint8)
    image[15:105:7, 73:88] = 150
    decision = detect_bright_streak_evidence(image, load_config(config_path))
    assert decision.status is DemoStatus.NG_NO_STREAK


def test_complete_new_dataset_has_the_locked_demo_labels() -> None:
    config = load_config(DEMO_CONFIG)
    labeled_paths = {
        DemoStatus.OK: sorted(OK_ROOT.glob("*.bmp")),
        DemoStatus.NG_NO_STREAK: sorted(NG_ROOT.glob("*.bmp")),
    }

    assert len(labeled_paths[DemoStatus.OK]) == 6
    assert len(labeled_paths[DemoStatus.NG_NO_STREAK]) == 7

    for expected, paths in labeled_paths.items():
        for path in paths:
            result = detect_bright_streak(_read_image(path), config)
            assert result.status is expected, (path.name, result)


def test_broken_streak_is_reported_separately_from_absence(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_synthetic_config(config_path)
    config = load_config(config_path)

    result = detect_bright_streak(_broken_streak_image(), config)

    assert result.status is DemoStatus.NG_BROKEN
    assert result.metrics.gap_count >= 1
    assert result.metrics.max_gap_ratio > config.max_gap_ratio


def test_wrong_dimensions_fail_closed(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_synthetic_config(config_path)
    config = load_config(config_path)

    result = detect_bright_streak(np.full((119, 160), 80, dtype=np.uint8), config)

    assert result.status is DemoStatus.ERROR


def test_clipped_roi_fails_closed(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_synthetic_config(config_path)
    config = load_config(config_path)

    result = detect_bright_streak(np.full((120, 160), 255, dtype=np.uint8), config)

    assert result.status is DemoStatus.ERROR


def test_detection_and_evidence_are_deterministic() -> None:
    config = load_config(DEMO_CONFIG)
    image = _read_image(OK_ROOT / "Image_20260805172921398.bmp")

    first = detect_bright_streak_evidence(image, config)
    second = detect_bright_streak_evidence(image.copy(), config)

    assert first.result.status is second.result.status
    assert first.result.reason == second.result.reason
    assert first.result.roi_xyxy == second.result.roi_xyxy
    assert first.result.metrics is not None and second.result.metrics is not None
    assert first.result.metrics.capture_elapsed_ms == second.result.metrics.capture_elapsed_ms == 0.0
    assert np.array_equal(first.response, second.response)
    assert np.array_equal(first.mask, second.mask)
    assert first.runs == second.runs
    assert first.gaps == second.gaps
    first_evidence = render_evidence(image, first)
    second_evidence = render_evidence(image.copy(), second)
    assert first_evidence.shape == (config.image_height, config.image_width, 3)
    assert first_evidence.dtype == np.uint8
    assert np.array_equal(first_evidence, second_evidence)


def test_saturated_path_returns_the_mask_used_for_decision() -> None:
    config = load_config(DEMO_CONFIG)
    image = _read_image(OK_ROOT / "Image_20260805172921398.bmp")

    decision = detect_bright_streak_evidence(image, config)

    assert decision.result.status is DemoStatus.OK
    assert decision.status is decision.result.status
    assert decision.metrics is decision.result.metrics
    assert decision.mask_used_for_metrics.dtype == np.bool_
    assert np.array_equal(decision.mask.astype(bool), decision.mask_used_for_metrics)
    assert np.array_equal(decision.accepted_mask.astype(bool), decision.mask_used_for_metrics)
    assert decision.mask.dtype == np.uint8
    assert set(np.unique(decision.mask)) <= {0, 255}

    nonzero_widths = np.count_nonzero(decision.mask, axis=1)
    expected_mean_width = float(nonzero_widths[nonzero_widths > 0].mean())
    assert decision.result.metrics is not None
    assert decision.result.metrics.mean_width_px == pytest.approx(expected_mean_width)
    assert decision.result.metrics.mean_width_px < decision.roi.shape[1]

    overlay = render_evidence(image, decision)
    x1, y1, x2, y2 = decision.result.roi_xyxy
    original_roi = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)[y1:y2, x1:x2]
    changed = np.any(overlay[y1:y2, x1:x2] != original_roi, axis=2)
    assert not changed.all()


@pytest.mark.parametrize("capture_elapsed_ms", [-1.0, float("nan"), float("inf")])
def test_detector_rejects_invalid_capture_elapsed_ms(capture_elapsed_ms: float) -> None:
    config = load_config(DEMO_CONFIG)
    image = _read_image(OK_ROOT / "Image_20260805172921398.bmp")

    with pytest.raises(ValueError, match="capture_elapsed_ms"):
        detect_bright_streak_evidence(image, config, capture_elapsed_ms=capture_elapsed_ms)


def test_structured_runs_gaps_and_timings_do_not_use_reason_encoding(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_synthetic_config(config_path)
    config = load_config(config_path)

    decision = detect_bright_streak_evidence(_broken_streak_image(), config)

    assert decision.result.status is DemoStatus.NG_BROKEN
    assert decision.runs
    assert decision.gaps
    assert "evidence_" not in decision.result.reason
    assert decision.result.metrics is not None
    for value in (
        decision.result.metrics.capture_elapsed_ms,
        decision.result.metrics.processing_elapsed_ms,
        decision.result.metrics.total_elapsed_ms,
    ):
        assert math.isfinite(value)
        assert value >= 0.0
