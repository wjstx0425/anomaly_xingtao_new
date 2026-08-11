"""Contract tests for the BMW bright-streak Demo configuration."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from bmw_inspection.contracts import BrightStreakConfig, DemoStatus, load_config


def _payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "mode": "demo",
        "camera_serial": "DA9625347",
        "image_width": 4024,
        "image_height": 3036,
        "exposure": 4000.0,
        "gain": 0.0,
        "timeout_ms": 3000,
        "warmup_frames": 1,
        "roi_xyxy": [1825, 1290, 1870, 1405],
        "result_root": "results/bmw_bright_streak_demo",
        "thresholds": {
            "min_mean_intensity": 20.0,
            "max_mean_intensity": 245.0,
            "max_dark_clip_ratio": 0.25,
            "max_bright_clip_ratio": 0.10,
            "min_laplacian_variance": 10.0,
            "background_kernel_px": 21,
            "response_mad_scale": 4.0,
            "min_component_area_px": 8,
            "min_component_width_px": 1.0,
            "max_component_width_px": 20.0,
            "center_tolerance_px": 18.0,
            "micro_gap_close_px": 3,
            "min_contrast_snr": 4.0,
            "min_coverage_ratio": 0.60,
            "min_longest_run_ratio": 0.50,
            "max_gap_ratio": 0.05,
            "max_gap_count": 1,
        },
    }


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_demo_status_values_are_the_business_contract() -> None:
    assert tuple(status.value for status in DemoStatus) == (
        "OK",
        "NG_NO_STREAK",
        "NG_BROKEN",
        "ERROR",
    )


def test_demo_status_imports_without_python_311_strenum() -> None:
    contracts_path = Path(__file__).parents[3] / "src/bmw_inspection/contracts.py"
    script = f"""
import enum
import importlib.util
import sys

delattr(enum, "StrEnum")
spec = importlib.util.spec_from_file_location("bmw_demo_contracts_py310", {str(contracts_path)!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert issubclass(module.DemoStatus, str)
assert module.DemoStatus.OK.value == "OK"
"""

    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_load_config_returns_a_frozen_validated_contract(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _write(path, _payload())

    config = load_config(path)

    assert isinstance(config, BrightStreakConfig)
    assert config.roi_xyxy == (1825, 1290, 1870, 1405)
    assert config.allowed_max_gap_px == 5
    with pytest.raises(FrozenInstanceError):
        config.gain = 2.0  # type: ignore[misc]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_load_config_rejects_non_finite_thresholds(tmp_path: Path, value: float) -> None:
    payload = _payload()
    thresholds = payload["thresholds"]
    assert isinstance(thresholds, dict)
    thresholds["min_contrast_snr"] = value
    path = tmp_path / "config.json"
    _write(path, payload)

    with pytest.raises(ValueError, match="finite"):
        load_config(path)


@pytest.mark.parametrize(
    "roi",
    [
        [-1, 10, 20, 30],
        [10, -1, 20, 30],
        [10, 10, 10, 30],
        [10, 10, 20, 10],
        [10, 10, 4025, 30],
        [10, 10, 20, 3037],
    ],
)
def test_load_config_rejects_roi_outside_half_open_image_bounds(
    tmp_path: Path,
    roi: list[int],
) -> None:
    payload = _payload()
    payload["roi_xyxy"] = roi
    path = tmp_path / "config.json"
    _write(path, payload)

    with pytest.raises(ValueError, match="roi_xyxy"):
        load_config(path)


def test_null_roi_loads_for_selection_but_is_rejected_for_detection(tmp_path: Path) -> None:
    payload = _payload()
    payload["roi_xyxy"] = None
    path = tmp_path / "config.json"
    _write(path, payload)

    config = load_config(path)

    assert config.roi_xyxy is None
    with pytest.raises(ValueError, match="ROI.*selected"):
        config.require_detection_roi()
