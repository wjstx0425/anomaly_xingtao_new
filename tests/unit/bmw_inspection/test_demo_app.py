"""Focused tests for the BMW Demo controller and result publication."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from bmw_inspection.contracts import BrightStreakResult, DemoStatus, load_config
from bmw_inspection.demo_app import (
    Action,
    _prepare_roi_evidence,
    action_enabled,
    format_metric_rows,
    key_to_action,
    mouse_to_action,
    publish_result,
    render_dashboard,
    render_waiting_dashboard,
    run_image,
    run_gui,
    run_offline,
    status_presentation,
)
from bmw_inspection.detector import detect_bright_streak, detect_bright_streak_evidence


REPO_ROOT = Path(__file__).resolve().parents[3]
DEMO_CONFIG = REPO_ROOT / "configs/bmw/bright_streak_demo.json"
OK_IMAGE = REPO_ROOT / "dataset/bmw/OK/Image_20260805172921398.bmp"
NO_STREAK_IMAGE = REPO_ROOT / "dataset/bmw/NG/Image_20260805173316482.bmp"
REQUIRED_ARTIFACTS = {
    "source.png",
    "roi.png",
    "response.png",
    "mask.png",
    "evidence.png",
    "result.json",
}


def _only_result_dir(root: Path) -> Path:
    result_files = list(root.rglob("result.json"))
    assert len(result_files) == 1
    return result_files[0].parent


def _assert_finite_json(value: object) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _assert_finite_json(item)
    elif isinstance(value, list):
        for item in value:
            _assert_finite_json(item)
    elif isinstance(value, float):
        assert math.isfinite(value)


def test_offline_demo_publishes_complete_explainable_results(tmp_path: Path) -> None:
    result = run_offline(OK_IMAGE, DEMO_CONFIG, output_root=tmp_path)

    assert result.status is DemoStatus.OK
    result_dir = _only_result_dir(tmp_path)
    assert REQUIRED_ARTIFACTS <= {path.name for path in result_dir.iterdir()}
    payload = json.loads((result_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "OK"
    assert payload["camera_serial"] == "DA9625347"
    assert payload["source_image"] == str(OK_IMAGE.resolve())
    assert payload["metrics"]["capture_elapsed_ms"] > 0.0
    assert payload["metrics"]["processing_elapsed_ms"] > 0.0
    assert payload["metrics"]["total_elapsed_ms"] == pytest.approx(
        payload["metrics"]["capture_elapsed_ms"] + payload["metrics"]["processing_elapsed_ms"],
    )
    _assert_finite_json(payload)
    for name in REQUIRED_ARTIFACTS - {"result.json"}:
        assert cv2.imread(str(result_dir / name), cv2.IMREAD_UNCHANGED) is not None


def test_offline_demo_preserves_no_streak_business_result(tmp_path: Path) -> None:
    result = run_offline(NO_STREAK_IMAGE, DEMO_CONFIG, output_root=tmp_path)

    assert result.status is DemoStatus.NG_NO_STREAK
    payload = json.loads((_only_result_dir(tmp_path) / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "NG_NO_STREAK"


def test_demo_publishes_the_detector_response_and_decision_mask(tmp_path: Path) -> None:
    config = load_config(DEMO_CONFIG)
    image = cv2.imread(str(OK_IMAGE), cv2.IMREAD_UNCHANGED)
    assert image is not None
    decision = detect_bright_streak_evidence(image, config)

    result, result_dir = run_image(
        image,
        config,
        output_root=tmp_path,
        source_ref="test:image",
        capture_elapsed_ms=12.5,
    )

    assert result.status is decision.result.status
    assert result.reason == decision.result.reason
    assert result.roi_xyxy == decision.result.roi_xyxy
    response = cv2.imread(str(result_dir / "response.png"), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(result_dir / "mask.png"), cv2.IMREAD_UNCHANGED)
    assert response is not None and mask is not None
    assert np.array_equal(response, decision.response)
    assert np.array_equal(mask, decision.mask)
    assert result.metrics is not None
    assert result.metrics.capture_elapsed_ms == 12.5
    assert result.metrics.total_elapsed_ms == pytest.approx(
        result.metrics.capture_elapsed_ms + result.metrics.processing_elapsed_ms,
    )


def test_publish_result_requires_one_structured_decision(tmp_path: Path) -> None:
    config = load_config(DEMO_CONFIG)
    image = cv2.imread(str(OK_IMAGE), cv2.IMREAD_UNCHANGED)
    assert image is not None
    decision = detect_bright_streak_evidence(image, config, capture_elapsed_ms=3.0)

    result_dir = publish_result(
        image,
        decision,
        config,
        output_root=tmp_path,
        source_ref="test:decision",
    )

    payload = json.loads((result_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == decision.result.status.value
    assert payload["reason"] == decision.result.reason
    assert payload["metrics"] == asdict(decision.result.metrics)


def test_publication_becomes_visible_only_after_all_files_exist(
    tmp_path: Path,
    monkeypatch,
) -> None:
    real_replace = Path.replace
    observed: list[set[str]] = []

    def track_replace(source: Path, destination: Path) -> Path:
        if source.is_dir():
            observed.append({path.name for path in source.iterdir()})
        return real_replace(source, destination)

    monkeypatch.setattr(Path, "replace", track_replace)

    run_offline(OK_IMAGE, DEMO_CONFIG, output_root=tmp_path)

    assert observed and REQUIRED_ARTIFACTS <= observed[-1]


def test_invalid_image_is_published_as_error_not_part_ng(tmp_path: Path) -> None:
    image_path = tmp_path / "wrong-size.png"
    assert cv2.imwrite(str(image_path), np.full((10, 10), 80, dtype=np.uint8))
    output_root = tmp_path / "results"

    result = run_offline(image_path, DEMO_CONFIG, output_root=output_root)

    assert result.status is DemoStatus.ERROR
    payload = json.loads((_only_result_dir(output_root) / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "ERROR"
    assert payload["reason"]


def test_demo_keyboard_mapping_is_small_and_explicit() -> None:
    assert key_to_action(ord(" ")) is Action.CAPTURE
    assert key_to_action(ord("r")) is Action.RETRY
    assert key_to_action(ord("R")) is Action.RETRY
    assert key_to_action(ord("q")) is Action.QUIT
    assert key_to_action(ord("Q")) is Action.QUIT
    assert key_to_action(-1) is None
    assert key_to_action(ord("x")) is None
    assert mouse_to_action(1000, 830) is Action.CAPTURE
    assert mouse_to_action(1200, 830) is Action.RETRY
    assert mouse_to_action(1400, 830) is Action.QUIT
    assert mouse_to_action(100, 100) is None


def test_demo_actions_import_without_python_311_strenum() -> None:
    script = f"""
import enum
import sys

sys.path.insert(0, {str(REPO_ROOT / 'src')!r})
if hasattr(enum, "StrEnum"):
    delattr(enum, "StrEnum")
import bmw_inspection.demo_app as module
assert issubclass(module.Action, str)
assert module.Action.CAPTURE.value == "capture"
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_all_business_statuses_have_exact_chinese_presentation() -> None:
    assert (status_presentation(DemoStatus.OK).title, status_presentation(DemoStatus.OK).detail) == (
        "检测通过",
        "亮痕存在且连续",
    )
    assert (
        status_presentation(DemoStatus.NG_NO_STREAK).title,
        status_presentation(DemoStatus.NG_NO_STREAK).detail,
    ) == ("检测不通过", "未检测到有效亮痕")
    assert (
        status_presentation(DemoStatus.NG_BROKEN).title,
        status_presentation(DemoStatus.NG_BROKEN).detail,
    ) == ("检测不通过", "亮痕存在但不连续")
    assert (status_presentation(DemoStatus.ERROR).title, status_presentation(DemoStatus.ERROR).detail) == (
        "设备或图像异常",
        "请检查相机、光源与工件位置",
    )


def test_metric_rows_are_simplified_chinese_business_copy() -> None:
    config = load_config(DEMO_CONFIG)
    image = cv2.imread(str(OK_IMAGE), cv2.IMREAD_UNCHANGED)
    assert image is not None
    result = detect_bright_streak(image, config)

    rows = format_metric_rows(result, config)

    assert tuple(row.label for row in rows) == ("亮痕覆盖率", "连续率", "最大断点", "对比度")
    assert rows[0].value.endswith("%")
    assert rows[1].value.endswith("%")
    assert "像素" in rows[2].value
    assert "良好" in rows[3].value

    error = BrightStreakResult(DemoStatus.ERROR, "camera error", config.detection_roi, None)
    assert format_metric_rows(error, config)[3].value == "不可用"

    no_streak_image = cv2.imread(str(NO_STREAK_IMAGE), cv2.IMREAD_UNCHANGED)
    assert no_streak_image is not None
    no_streak = detect_bright_streak(no_streak_image, config)
    assert format_metric_rows(no_streak, config)[2].value == "不可用"


def test_dashboard_is_deterministic_and_has_hover_feedback(tmp_path: Path) -> None:
    config = load_config(DEMO_CONFIG)
    image = cv2.imread(str(OK_IMAGE), cv2.IMREAD_UNCHANGED)
    assert image is not None
    result = detect_bright_streak(image, config)
    fixed_time = datetime(2026, 8, 5, 18, 30, 0)

    first = render_dashboard(image, result, config, tmp_path, now=fixed_time)
    second = render_dashboard(image.copy(), result, config, tmp_path, now=fixed_time)
    hovered = render_dashboard(
        image,
        result,
        config,
        tmp_path,
        hovered_action=Action.CAPTURE,
        now=fixed_time,
    )

    assert first.shape == (900, 1600, 3)
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)
    assert not np.array_equal(first[810:870, 985:1165], hovered[810:870, 985:1165])


def test_roi_evidence_rotates_clockwise_and_preserves_aspect_ratio() -> None:
    zoom = np.zeros((6, 2, 3), dtype=np.uint8)
    zoom[0, 0] = (255, 0, 0)

    prepared = _prepare_roi_evidence(zoom, width=12, height=8)

    assert prepared.shape == (8, 12, 3)
    blue_y, blue_x = np.unravel_index(np.argmax(prepared[:, :, 0]), prepared[:, :, 0].shape)
    assert blue_x >= 10
    assert blue_y <= 3
    background = np.array((15, 18, 21), dtype=np.uint8)
    content_y, content_x = np.nonzero(np.any(prepared != background, axis=2))
    content_width = int(content_x.max() - content_x.min() + 1)
    content_height = int(content_y.max() - content_y.min() + 1)
    assert content_width / content_height == 3.0


def test_waiting_state_disables_retry_and_renders_deterministically() -> None:
    config = load_config(DEMO_CONFIG)
    fixed_time = datetime(2026, 8, 5, 18, 45, 0)

    first = render_waiting_dashboard(config, now=fixed_time)
    second = render_waiting_dashboard(config, now=fixed_time)

    assert action_enabled(Action.CAPTURE, has_result=False)
    assert not action_enabled(Action.RETRY, has_result=False)
    assert action_enabled(Action.QUIT, has_result=False)
    assert action_enabled(Action.RETRY, has_result=True)
    assert first.shape == (900, 1600, 3)
    assert np.array_equal(first, second)


def _patch_highgui(monkeypatch, keys: list[int]) -> None:
    key_iterator = iter(keys)
    monkeypatch.setattr(cv2, "namedWindow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cv2, "resizeWindow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cv2, "imshow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cv2, "waitKey", lambda _delay=0: next(key_iterator))
    monkeypatch.setattr(cv2, "destroyWindow", lambda *_args, **_kwargs: None)


def test_live_gui_quit_before_inspection_does_not_acquire(tmp_path: Path, monkeypatch) -> None:
    config = load_config(DEMO_CONFIG)
    calls: list[str] = []
    _patch_highgui(monkeypatch, [ord("q")])

    def acquire() -> np.ndarray:
        calls.append("capture")
        raise AssertionError("startup must not acquire")

    result = run_gui(config, acquire, output_root=tmp_path)

    assert result is None
    assert calls == []
    assert not list(tmp_path.rglob("result.json"))


def test_live_gui_space_then_quit_acquires_exactly_once(tmp_path: Path, monkeypatch) -> None:
    config = load_config(DEMO_CONFIG)
    image = cv2.imread(str(OK_IMAGE), cv2.IMREAD_UNCHANGED)
    assert image is not None
    calls: list[str] = []
    _patch_highgui(monkeypatch, [ord(" "), ord("q")])

    def acquire() -> np.ndarray:
        calls.append("capture")
        return image.copy()

    capture_clock = iter((100.0, 100.025))
    real_perf_counter = time.perf_counter

    def perf_counter() -> float:
        return next(capture_clock, real_perf_counter())

    monkeypatch.setattr("bmw_inspection.demo_app.time.perf_counter", perf_counter)

    result = run_gui(config, acquire, output_root=tmp_path)

    assert result is not None and result.status is DemoStatus.OK
    assert calls == ["capture"]
    assert len(list(tmp_path.rglob("result.json"))) == 1
    payload = json.loads((_only_result_dir(tmp_path) / "result.json").read_text(encoding="utf-8"))
    assert payload["metrics"]["capture_elapsed_ms"] == pytest.approx(25.0)
    assert payload["metrics"]["total_elapsed_ms"] == pytest.approx(
        payload["metrics"]["capture_elapsed_ms"] + payload["metrics"]["processing_elapsed_ms"],
    )
