# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared keyboard, mouse, reducer, and HighGUI loop contracts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import pytest

from zs32_inspection.dashboard import app as dashboard_app
from zs32_inspection.dashboard.app import (
    Action,
    DashboardState,
    key_to_action,
    mouse_to_action,
    reduce_state,
    run_dashboard,
    service_live_state,
)
from zs32_inspection.dashboard.contracts import EvidenceLayer, ProgressRecord
from zs32_inspection.dashboard.parser import load_inspection_result
from zs32_inspection.dashboard.render import render_dashboard


@pytest.fixture
def state(eight_view_result_dir: Path) -> DashboardState:
    return DashboardState(result=load_inspection_result(eight_view_result_dir))


def test_s_key_and_detection_button_share_one_action(state: DashboardState) -> None:
    state = replace(state, result=None)
    frame = render_dashboard(state)

    assert key_to_action(ord("s")) == Action.INSPECTION
    assert mouse_to_action(frame.inspection_button.rect.center, frame.hit_regions) == Action.INSPECTION


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        (ord("1"), Action.SELECT_FUSION),
        (ord("2"), Action.SELECT_ORIGINAL),
        (ord("3"), Action.SELECT_PATCHCORE),
        (ord("4"), Action.SELECT_YOLO),
        (ord("5"), Action.SELECT_TEMPLATE),
        (ord("q"), Action.QUIT),
        (27, Action.ESCAPE),
        (ord("X"), None),
    ],
)
def test_key_mapping_is_stable(key: int, expected: Action | None) -> None:
    assert key_to_action(key) == expected


def test_reducer_selects_layers_and_views_and_escape_is_predictable(state: DashboardState) -> None:
    selected = reduce_state(state, Action.SELECT_VIEW, value="back_right")
    layered = reduce_state(selected, Action.SELECT_YOLO)

    assert layered.selected_view == "back_right"
    assert layered.layer is EvidenceLayer.YOLO
    assert reduce_state(layered, Action.ESCAPE).selected_view is None
    assert reduce_state(state, Action.ESCAPE).exit_requested is True


def test_disabled_inspection_action_does_not_request_work(state: DashboardState) -> None:
    state = replace(
        state,
        running=True,
        progress=ProgressRecord("ZS32-0001", None, "inference", "running", "now"),
    )

    assert reduce_state(state, Action.INSPECTION).inspection_requested is False


def test_offline_inspection_shortcut_does_not_request_live_work(state: DashboardState) -> None:
    assert state.result is not None
    assert state.result.mode == "offline"

    assert reduce_state(state, Action.INSPECTION) is state


def test_complete_live_result_is_loaded_only_once(
    state: DashboardState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress = ProgressRecord("ZS32-0001", "session", "complete", "/result", "now")

    class Controller:
        running = False

        @staticmethod
        def poll() -> ProgressRecord:
            return progress

    loads: list[Path] = []

    def load_once(path: Path):
        loads.append(path)
        return state.result

    monkeypatch.setattr("zs32_inspection.dashboard.app.load_inspection_result", load_once)

    first = service_live_state(state, Controller(), "ZS32-0001")  # type: ignore[arg-type]
    service_live_state(first, Controller(), "ZS32-0001")  # type: ignore[arg-type]

    assert loads == [Path("/result")]


def test_disabled_keyboard_inspection_preserves_existing_request_and_matches_mouse(
    state: DashboardState,
) -> None:
    state = replace(
        state,
        running=True,
        inspection_requested=True,
        progress=ProgressRecord("ZS32-0001", None, "inference", "running", "now"),
    )
    frame = render_dashboard(state)

    assert mouse_to_action(frame.inspection_button.rect.center, frame.hit_regions) is None
    action = key_to_action(ord("s"))
    assert action is Action.INSPECTION
    assert reduce_state(state, action) is state


def test_retry_waits_for_worker_readiness_before_starting_stage35() -> None:
    progress = ProgressRecord("part-1", None, "worker_starting", "loading", "now")

    class Controller:
        running = False
        worker_ready = False
        starts = 0
        worker_starts = 0

        def start_worker(self) -> None:
            self.worker_starts += 1

        def start(self, _part_id: str) -> None:
            self.starts += 1

        @staticmethod
        def poll() -> ProgressRecord:
            return progress

    controller = Controller()
    state = DashboardState(result=None, inspection_requested=True)

    updated = service_live_state(state, controller, "part-1")  # type: ignore[arg-type]

    assert controller.worker_starts == 1
    assert controller.starts == 0
    assert updated.inspection_requested is False
    assert updated.progress == progress


def test_no_gui_never_calls_highgui(
    state: DashboardState,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("HighGUI must not be called")

    monkeypatch.setattr(cv2, "namedWindow", forbidden)
    monkeypatch.setattr(cv2, "imshow", forbidden)
    monkeypatch.setattr(cv2, "waitKey", forbidden)

    screenshot = tmp_path / "nested" / "dashboard.png"
    final = run_dashboard(state, no_gui=True, save_screenshot=screenshot)

    assert final == state
    assert screenshot.is_file()
    assert cv2.imread(str(screenshot)).shape == (920, 1600, 3)


def test_live_complete_no_gui_records_parse_and_first_render_once(
    eight_view_result_dir: Path,
) -> None:
    progress = ProgressRecord(
        "ZS32-0001",
        "session",
        "complete",
        str(eight_view_result_dir),
        "now",
    )

    class Controller:
        running = False

        @staticmethod
        def poll() -> ProgressRecord:
            return progress

        @staticmethod
        def close() -> None:
            return None

    final = run_dashboard(
        DashboardState(result=None),
        no_gui=True,
        live_controller=Controller(),  # type: ignore[arg-type]
        part_id="ZS32-0001",
    )

    assert final.result is not None
    timing = json.loads((eight_view_result_dir / "timing.json").read_text(encoding="utf-8"))
    assert timing["stages"]["dashboard_result_parse"]["count"] == 1
    assert timing["stages"]["dashboard_first_frame_render"]["count"] == 1
    assert timing["stages"]["dashboard_result_parse"]["seconds"] >= 0
    assert timing["stages"]["dashboard_first_frame_render"]["seconds"] >= 0
    assert "dashboard_first_frame_display" not in timing["stages"]


def test_live_complete_gui_records_display_after_imshow_returns(
    eight_view_result_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress = ProgressRecord(
        "ZS32-0001",
        "session",
        "complete",
        str(eight_view_result_dir),
        "now",
    )

    class Controller:
        running = False

        @staticmethod
        def poll() -> ProgressRecord:
            return progress

        @staticmethod
        def close() -> None:
            return None

    imshow_returned = False

    def imshow(*_args: object, **_kwargs: object) -> None:
        nonlocal imshow_returned
        imshow_returned = True

    publish_timing = dashboard_app._publish_dashboard_timing

    def publish_after_imshow(session: object) -> None:
        assert imshow_returned is True
        publish_timing(session)  # type: ignore[arg-type]

    monkeypatch.setattr(cv2, "namedWindow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cv2, "resizeWindow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cv2, "imshow", imshow)
    monkeypatch.setattr(cv2, "waitKey", lambda _delay: ord("q"))
    monkeypatch.setattr(cv2, "destroyWindow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dashboard_app, "_publish_dashboard_timing", publish_after_imshow)

    run_dashboard(
        DashboardState(result=None),
        live_controller=Controller(),  # type: ignore[arg-type]
        part_id="ZS32-0001",
    )

    timing = json.loads((eight_view_result_dir / "timing.json").read_text(encoding="utf-8"))
    assert imshow_returned is True
    assert timing["stages"]["dashboard_result_parse"]["count"] == 1
    assert timing["stages"]["dashboard_first_frame_render"]["count"] == 1
    assert timing["stages"]["dashboard_first_frame_display"]["count"] == 1


def test_malformed_timing_json_is_rebuilt_without_failing_dashboard(
    eight_view_result_dir: Path,
) -> None:
    (eight_view_result_dir / "timing.json").write_text("{not-json", encoding="utf-8")
    progress = ProgressRecord(
        "ZS32-0001",
        "session",
        "complete",
        str(eight_view_result_dir),
        "now",
    )

    class Controller:
        running = False

        @staticmethod
        def poll() -> ProgressRecord:
            return progress

        @staticmethod
        def close() -> None:
            return None

    run_dashboard(
        DashboardState(result=None),
        no_gui=True,
        live_controller=Controller(),  # type: ignore[arg-type]
        part_id="ZS32-0001",
    )

    timing = json.loads((eight_view_result_dir / "timing.json").read_text(encoding="utf-8"))
    assert set(timing["stages"]) == {"dashboard_first_frame_render", "dashboard_result_parse"}


@pytest.mark.parametrize(
    "existing",
    [
        {"stages": {"capture": {"count": 1, "seconds": 1.25}}},
        {"stages": {"broken": {"count": 1, "seconds": float("nan")}}},
    ],
)
def test_dashboard_timing_preserves_valid_stages_and_replaces_invalid_payload(
    eight_view_result_dir: Path,
    existing: dict[str, object],
) -> None:
    (eight_view_result_dir / "timing.json").write_text(json.dumps(existing), encoding="utf-8")
    progress = ProgressRecord(
        "ZS32-0001",
        "session",
        "complete",
        str(eight_view_result_dir),
        "now",
    )

    class Controller:
        running = False

        @staticmethod
        def poll() -> ProgressRecord:
            return progress

        @staticmethod
        def close() -> None:
            return None

    run_dashboard(
        DashboardState(result=None),
        no_gui=True,
        live_controller=Controller(),  # type: ignore[arg-type]
        part_id="ZS32-0001",
    )

    timing = json.loads((eight_view_result_dir / "timing.json").read_text(encoding="utf-8"))
    assert timing["stages"]["dashboard_result_parse"]["count"] == 1
    assert timing["stages"]["dashboard_first_frame_render"]["count"] == 1
    if "capture" in existing["stages"]:  # type: ignore[operator]
        assert timing["stages"]["capture"] == {"count": 1, "seconds": 1.25}
    else:
        assert "broken" not in timing["stages"]


def test_screenshot_write_failure_raises(
    state: DashboardState,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cv2, "imwrite", lambda *_args, **_kwargs: False)

    with pytest.raises(OSError, match="screenshot"):
        run_dashboard(state, no_gui=True, save_screenshot=tmp_path / "failed.png")


def test_gui_loop_uses_50ms_poll(state: DashboardState, monkeypatch: pytest.MonkeyPatch) -> None:
    waits: list[int] = []
    monkeypatch.setattr(cv2, "namedWindow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cv2, "resizeWindow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cv2, "imshow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cv2, "destroyWindow", lambda *_args, **_kwargs: None)

    def wait_key(delay: int) -> int:
        waits.append(delay)
        return ord("q")

    monkeypatch.setattr(cv2, "waitKey", wait_key)

    final = run_dashboard(state)

    assert final.exit_requested is True
    assert waits == [50]
