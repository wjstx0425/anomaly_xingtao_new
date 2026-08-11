"""Tests for atomic BMW ROI selection and persistence."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from bmw_inspection.roi_selector import save_roi, select_roi


REPO_ROOT = Path(__file__).resolve().parents[3]


def _payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "mode": "demo",
        "camera_serial": "DA9625347",
        "image_width": 100,
        "image_height": 80,
        "exposure": 4000.0,
        "gain": 0.0,
        "timeout_ms": 3000,
        "warmup_frames": 1,
        "roi_xyxy": None,
        "result_root": "results/untouched",
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
        "operator_note": {"keep": ["all", "non-ROI", "fields"]},
    }


def _write(path: Path) -> dict[str, object]:
    payload = _payload()
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def test_save_roi_atomically_preserves_every_non_roi_json_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "config.json"
    original = _write(path)
    real_replace = Path.replace
    replacements: list[tuple[Path, Path]] = []

    def track_replace(source: Path, destination: Path) -> Path:
        replacements.append((source, destination))
        assert source.parent == destination.parent
        assert source != destination
        json.loads(source.read_text(encoding="utf-8"))
        return real_replace(source, destination)

    monkeypatch.setattr(Path, "replace", track_replace)

    save_roi(path, (10, 12, 40, 50))

    updated = json.loads(path.read_text(encoding="utf-8"))
    assert replacements and replacements[-1][1] == path
    assert updated["roi_xyxy"] == [10, 12, 40, 50]
    assert {key: value for key, value in updated.items() if key != "roi_xyxy"} == {
        key: value for key, value in original.items() if key != "roi_xyxy"
    }
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


def test_save_roi_rejects_invalid_bounds_without_touching_config(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _write(path)
    before = path.read_bytes()

    with pytest.raises(ValueError, match="roi_xyxy"):
        save_roi(path, (10, 12, 101, 50))

    assert path.read_bytes() == before


def test_select_roi_converts_xywh_and_requires_enter_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.json"
    _write(config_path)
    image_path = tmp_path / "ok.bmp"
    assert cv2.imwrite(str(image_path), np.full((80, 100), 127, dtype=np.uint8))
    monkeypatch.setattr(cv2, "selectROI", lambda *_args, **_kwargs: (10, 12, 30, 38))
    monkeypatch.setattr(cv2, "imshow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cv2, "waitKey", lambda _delay=0: 13)
    monkeypatch.setattr(cv2, "destroyWindow", lambda *_args, **_kwargs: None)

    roi = select_roi(image_path, config_path)

    assert roi == (10, 12, 40, 50)
    assert json.loads(config_path.read_text(encoding="utf-8"))["roi_xyxy"] == [10, 12, 40, 50]


def test_select_roi_does_not_persist_when_confirmation_is_not_enter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.json"
    _write(config_path)
    image_path = tmp_path / "ok.bmp"
    assert cv2.imwrite(str(image_path), np.full((80, 100), 127, dtype=np.uint8))
    monkeypatch.setattr(cv2, "selectROI", lambda *_args, **_kwargs: (10, 12, 30, 38))
    monkeypatch.setattr(cv2, "imshow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cv2, "waitKey", lambda _delay=0: ord("q"))
    monkeypatch.setattr(cv2, "destroyWindow", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="Enter"):
        select_roi(image_path, config_path)

    assert json.loads(config_path.read_text(encoding="utf-8"))["roi_xyxy"] is None


def test_select_roi_scales_large_image_and_maps_selection_to_source_coordinates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload()
    payload["image_width"] = 4000
    payload["image_height"] = 3000
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    image_path = tmp_path / "large.bmp"
    assert cv2.imwrite(str(image_path), np.full((3000, 4000), 127, dtype=np.uint8))
    displayed_shapes: list[tuple[int, ...]] = []

    def select_roi_stub(_window: str, displayed: np.ndarray, **_kwargs: object) -> tuple[int, int, int, int]:
        displayed_shapes.append(displayed.shape)
        return 240, 120, 120, 240

    monkeypatch.setattr(cv2, "selectROI", select_roi_stub)
    monkeypatch.setattr(cv2, "imshow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cv2, "waitKey", lambda _delay=0: 13)
    monkeypatch.setattr(cv2, "destroyWindow", lambda *_args, **_kwargs: None)

    roi = select_roi(image_path, config_path)

    assert displayed_shapes == [(720, 960)]
    assert roi == (1000, 500, 1500, 1500)
    assert json.loads(config_path.read_text(encoding="utf-8"))["roi_xyxy"] == [1000, 500, 1500, 1500]


def test_roi_selector_cli_defaults_to_first_new_ok_image() -> None:
    script_path = REPO_ROOT / "pipeline/bmw_select_bright_streak_roi.py"
    spec = importlib.util.spec_from_file_location("bmw_select_bright_streak_roi", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    expected = REPO_ROOT / "dataset/bmw/OK/Image_20260805172921398.bmp"
    assert module.DEFAULT_IMAGE == expected
    assert module.DEFAULT_IMAGE.is_file()
