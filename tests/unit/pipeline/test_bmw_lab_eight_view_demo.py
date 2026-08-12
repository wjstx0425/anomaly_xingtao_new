"""Entry-point integration for BMW v3 evidence persistence."""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER
from bmw_inspection.lab.eight_view_demo import (
    BranchStatus,
    DemoBranch,
    DemoBranchResult,
    DemoFinalStatus,
    EightViewInspection,
)
from pipeline import bmw_lab_eight_view_demo as entrypoint
from bmw_inspection.lab.eight_view_demo_ui import DemoUiPage, EightViewUiState, select_branch


def test_persist_result_adapts_offline_images_and_preserves_live_sources(monkeypatch) -> None:
    images = {view: np.zeros((4, 5, 3), dtype=np.uint8) for view in VIEW_ORDER}
    inspection = SimpleNamespace(images=images)
    config = object()
    live_sources = {view: object() for view in VIEW_ORDER}
    calls = []
    monkeypatch.setattr(entrypoint, "fused_only_sources", lambda value: ("offline", value))
    monkeypatch.setattr(
        entrypoint,
        "persist_inspection",
        lambda used_config, used_inspection, used_sources: calls.append(
            (used_config, used_inspection, used_sources)
        ),
    )

    entrypoint._persist_result(config, inspection, None)
    entrypoint._persist_result(config, inspection, live_sources)

    assert calls == [
        (config, inspection, ("offline", images)),
        (config, inspection, live_sources),
    ]


def test_o_shortcut_handler_toggles_only_when_inspection_exists() -> None:
    idle = EightViewUiState()
    assert entrypoint._handle_trusted_ok_shortcut(idle, ord("o")) is idle
    legacy = SimpleNamespace(results=("unchanged",), trusted_ok_by_comparison={}, diagnostic_metadata={})
    legacy_state = EightViewUiState(inspection=legacy)  # type: ignore[arg-type]
    assert entrypoint._handle_trusted_ok_shortcut(legacy_state, ord("o")) is legacy_state
    inspection = SimpleNamespace(
        results=("unchanged",), trusted_ok_by_comparison={("front", "roi"): object()}, diagnostic_metadata={}
    )
    state = EightViewUiState(inspection=inspection)  # type: ignore[arg-type]

    toggled = entrypoint._handle_trusted_ok_shortcut(state, ord("O"))

    assert toggled.trusted_ok_mode is True
    assert toggled.inspection is inspection
    assert toggled.inspection.results == ("unchanged",)
    assert entrypoint._handle_trusted_ok_shortcut(toggled, ord("x")) is toggled


def test_logical_canvas_coordinates_scale_default_and_resized_viewports() -> None:
    assert entrypoint._logical_canvas_coordinates(720, 405, 1440, 810) == (800, 450)
    assert entrypoint._logical_canvas_coordinates(1000, 500, 2000, 1000) == (800, 450)
    assert entrypoint._logical_canvas_coordinates(-1, 0, 1440, 810) is None
    assert entrypoint._logical_canvas_coordinates(1440, 0, 1440, 810) is None
    assert entrypoint._logical_canvas_coordinates(0, 810, 1440, 810) is None
    assert entrypoint._logical_canvas_coordinates(0, 0, 0, 810) is None


def test_mouse_callback_only_queues_left_button_releases_and_consumption_applies_hit(monkeypatch) -> None:
    releases: list[tuple[int, int]] = []

    entrypoint._queue_left_button_release(cv2.EVENT_MOUSEMOVE, 10, 20, 0, releases)
    entrypoint._queue_left_button_release(cv2.EVENT_LBUTTONUP, 720, 405, 0, releases)

    assert releases == [(720, 405)]
    selected = []
    monkeypatch.setattr(entrypoint, "dashboard_hit_test", lambda x, y: (x, y))
    monkeypatch.setattr(
        entrypoint,
        "apply_dashboard_click",
        lambda state, hit: selected.append(hit) or state,
    )
    state = EightViewUiState()

    assert entrypoint._consume_mouse_releases(state, releases, 1440, 810) is state
    assert selected == [(800, 450)]
    assert releases == []


def test_gui_window_is_realized_before_mouse_callback_registration(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []
    dashboard = np.zeros((900, 1600, 3), dtype=np.uint8)
    releases: list[tuple[int, int]] = []
    state = EightViewUiState(experiment_mode=True)
    monkeypatch.setattr(
        entrypoint.cv2,
        "namedWindow",
        lambda title, mode: calls.append(("named", title, mode)),
    )
    monkeypatch.setattr(
        entrypoint.cv2,
        "resizeWindow",
        lambda title, width, height: calls.append(("resize", title, width, height)),
    )
    monkeypatch.setattr(entrypoint, "render_eight_view_screen", lambda used_state: dashboard)
    monkeypatch.setattr(entrypoint.cv2, "imshow", lambda title, image: calls.append(("show", title, image)))
    monkeypatch.setattr(entrypoint.cv2, "waitKey", lambda delay: calls.append(("wait", delay)) or -1)
    monkeypatch.setattr(
        entrypoint.cv2,
        "setMouseCallback",
        lambda title, callback, userdata: calls.append(("mouse", title, callback, userdata)),
    )
    monkeypatch.setattr(
        entrypoint.cv2,
        "setWindowTitle",
        lambda title, visible_title: calls.append(("caption", title, visible_title)),
    )

    entrypoint._initialize_gui_window("BMW_EIGHT_VIEW_DEMO", "BMW 零件八视图检测", state, releases)

    assert [call[0] for call in calls] == ["named", "resize", "show", "wait", "mouse", "caption"]
    assert {calls[index][1] for index in (0, 1, 2, 4, 5)} == {"BMW_EIGHT_VIEW_DEMO"}
    assert calls[2][2] is dashboard
    assert calls[3] == ("wait", 1)
    assert calls[4][2] is entrypoint._queue_left_button_release
    assert calls[4][3] is releases
    assert calls[5] == ("caption", "BMW_EIGHT_VIEW_DEMO", "BMW 零件八视图检测")


def test_page_key_handling_returns_detail_to_dashboard_and_preserves_dashboard_escape() -> None:
    detail = EightViewUiState(page=DemoUiPage.DETAIL, experiment_mode=True)

    returned, action = entrypoint._handle_gui_key(detail, 27)
    dashboard, dashboard_action = entrypoint._handle_gui_key(EightViewUiState(), 27)
    _, quit_action = entrypoint._handle_gui_key(detail, ord("q"))
    reset, reset_action = entrypoint._handle_gui_key(detail, ord("r"))

    assert returned.page is DemoUiPage.DASHBOARD
    assert action is entrypoint._GuiAction.CONTINUE
    for key in (ord(" "), ord("t"), ord("l"), ord("y"), ord("e"), ord("n"), ord("p"), ord("o"), *range(ord("1"), ord("8") + 1)):
        ignored, ignored_action = entrypoint._handle_gui_key(detail, key)
        assert ignored is detail
        assert ignored_action is entrypoint._GuiAction.CONTINUE
    assert dashboard.page is DemoUiPage.DASHBOARD
    assert dashboard_action is entrypoint._GuiAction.CONTINUE
    assert quit_action is entrypoint._GuiAction.EXIT
    assert reset.page is DemoUiPage.DASHBOARD
    assert reset.experiment_mode is True
    assert reset_action is entrypoint._GuiAction.RESET


def test_keyboard_branch_selection_matches_card_selection_and_jumps_to_actionable_view() -> None:
    images = {view: np.zeros((4, 5, 3), dtype=np.uint8) for view in VIEW_ORDER}
    yolo_result = DemoBranchResult(
        DemoBranch.YOLO,
        "front_right",
        BranchStatus.NG,
        0.9,
        0.5,
        1.0,
        "发现缺陷",
        np.zeros((4, 5, 3), dtype=np.uint8),
    )
    inspection = EightViewInspection(
        "capture",
        images,
        (yolo_result,),
        DemoFinalStatus.NG,
        1.0,
    )
    state = EightViewUiState(
        inspection=inspection,
        selected_view="back",
        selected_branch=DemoBranch.TEMPLATE,
        experiment_mode=True,
    )

    keyboard_state, action = entrypoint._handle_gui_key(state, ord("y"))

    assert keyboard_state == select_branch(state, DemoBranch.YOLO)
    assert keyboard_state.selected_view == "front_right"
    assert action is entrypoint._GuiAction.CONTINUE
