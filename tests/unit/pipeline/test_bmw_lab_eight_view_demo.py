"""Entry-point integration for BMW v3 evidence persistence."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from bmw_inspection.views import VIEW_ORDER
from bmw_inspection.lab.eight_view_demo import (
    BranchStatus,
    DemoBranch,
    DemoBranchResult,
    DemoFinalStatus,
    EightViewInspection,
)
from pipeline import bmw_lab_eight_view_demo as entrypoint
from bmw_inspection.lab.eight_view_demo_ui import DemoUiPage, DemoUiPhase, EightViewUiState, select_branch
from bmw_inspection.lab.live_cycle_timing import LiveCycleTiming


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


def test_qt_mouse_coordinates_are_already_in_logical_canvas_pixels() -> None:
    assert entrypoint._logical_canvas_coordinates(720, 405) == (720, 405)
    assert entrypoint._logical_canvas_coordinates(1599, 899) == (1599, 899)
    assert entrypoint._logical_canvas_coordinates(-1, 0) is None
    assert entrypoint._logical_canvas_coordinates(1600, 0) is None
    assert entrypoint._logical_canvas_coordinates(0, 900) is None


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

    assert entrypoint._consume_mouse_releases(state, releases) is state
    assert selected == [(720, 405)]
    assert releases == []


def test_qt_mouse_coordinates_select_all_algorithm_cards_without_rescaling() -> None:
    card_centres = {
        DemoBranch.TEMPLATE: (1200, 312),
        DemoBranch.BRIGHT_STREAK: (1450, 312),
        DemoBranch.YOLO: (1200, 444),
        DemoBranch.EFFICIENTAD: (1450, 444),
    }

    for branch, point in card_centres.items():
        state = EightViewUiState(selected_branch=DemoBranch.BRIGHT_STREAK)
        selected = entrypoint._consume_mouse_releases(state, [point])

        assert selected.selected_branch is branch


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


def test_processing_space_is_ignored() -> None:
    state = EightViewUiState(phase=DemoUiPhase.PROCESSING)

    unchanged, action = entrypoint._handle_gui_key(state, ord(" "))

    assert unchanged is state
    assert action is entrypoint._GuiAction.CONTINUE


def test_live_gui_overlaps_front_inference_with_back_capture_and_persists_once(monkeypatch) -> None:
    main_thread = threading.get_ident()
    front_started = threading.Event()
    release_front = threading.Event()
    persisted = threading.Event()
    calls: list[tuple[object, ...]] = []
    wait_count = 0

    front_images = {
        view: np.full((4, 5, 3), index, dtype=np.uint8)
        for index, view in enumerate(VIEW_ORDER[:4])
    }
    back_images = {
        view: np.full((4, 5, 3), index, dtype=np.uint8)
        for index, view in enumerate(VIEW_ORDER[4:], start=4)
    }
    inspection = SimpleNamespace(capture_id="inspection", images={**front_images, **back_images})

    class FakeCamera:
        def __init__(self) -> None:
            self.last_sources = {view: object() for view in VIEW_ORDER}

        def __enter__(self):
            calls.append(("camera_enter", threading.get_ident()))
            return self

        def __exit__(self, *_args):
            calls.append(("camera_exit", threading.get_ident()))
            return False

        def capture_round(self, round_name: str):
            calls.append(("capture", round_name, threading.get_ident(), front_started.is_set()))
            return front_images if round_name == "front" else back_images

    class FakeModels:
        def inspect(self, _images, *, capture_id):
            calls.append(("legacy_inspect", capture_id, threading.get_ident()))
            return inspection

        def inspect_round(self, _images, round_name: str):
            calls.append(("inspect_round", round_name, threading.get_ident()))
            if round_name == "front":
                front_started.set()
                assert release_front.wait(timeout=2)
            return f"{round_name}-partial"

        def finalize_rounds(self, _images, front, back, capture_id):
            calls.append(("finalize", front, back, capture_id, threading.get_ident()))
            return inspection

    camera = FakeCamera()

    def fake_wait_key(_delay: int) -> int:
        nonlocal wait_count
        calls.append(("wait_key", threading.get_ident()))
        wait_count += 1
        if wait_count == 1:
            return ord(" ")
        if wait_count == 2:
            assert front_started.wait(timeout=2)
            return ord(" ")
        if wait_count in {3, 4, 5}:
            return ord(" ")
        if wait_count == 6:
            release_front.set()
            return -1
        if persisted.wait(timeout=0.001):
            return ord("q")
        return ord("q") if wait_count > 5_000 else -1

    def fake_persist(used_config, used_inspection, used_sources) -> None:
        calls.append(
            (
                "persist",
                used_config,
                used_inspection,
                used_sources,
                threading.get_ident(),
            )
        )
        persisted.set()

    gui_threads: list[int] = []
    monkeypatch.setattr(entrypoint, "FourCameraHdrSession", lambda _path: camera)
    monkeypatch.setattr(entrypoint, "_initialize_gui_window", lambda *_args: None)
    monkeypatch.setattr(entrypoint, "render_eight_view_screen", lambda _state: np.zeros((2, 2, 3), dtype=np.uint8))
    monkeypatch.setattr(entrypoint.cv2, "imshow", lambda *_args: gui_threads.append(threading.get_ident()))
    monkeypatch.setattr(entrypoint.cv2, "waitKey", fake_wait_key)
    monkeypatch.setattr(entrypoint.cv2, "destroyAllWindows", lambda: gui_threads.append(threading.get_ident()))
    monkeypatch.setattr(entrypoint, "persist_inspection", fake_persist)
    monkeypatch.setattr(entrypoint, "preferred_selection", lambda _inspection: ("front", DemoBranch.TEMPLATE))

    status = entrypoint._run_gui(
        SimpleNamespace(experiment_mode=False, save_screenshot=None),
        SimpleNamespace(capture_config="capture.json"),
        FakeModels(),
        None,
    )

    capture_calls = [call for call in calls if call[0] == "capture"]
    round_calls = [call for call in calls if call[0] == "inspect_round"]
    finalize_calls = [call for call in calls if call[0] == "finalize"]
    persist_calls = [call for call in calls if call[0] == "persist"]
    assert status == 0
    assert [(call[1], call[2]) for call in capture_calls] == [
        ("front", main_thread),
        ("back", main_thread),
    ]
    assert capture_calls[1][3] is True
    assert [call[1] for call in round_calls] == ["front", "back"]
    assert {call[2] for call in round_calls} == {finalize_calls[0][4], persist_calls[0][4]}
    assert round_calls[0][2] != main_thread
    assert finalize_calls[0][1:3] == ("front-partial", "back-partial")
    assert len(finalize_calls) == 1
    assert len(persist_calls) == 1
    assert not any(call[0] == "legacy_inspect" for call in calls)
    assert set(gui_threads) == {main_thread}


def test_live_gui_records_first_space_to_first_result_frame(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class ManualClock:
        def __init__(self) -> None:
            self._nanoseconds = 0
            self._lock = threading.Lock()

        def __call__(self) -> int:
            with self._lock:
                return self._nanoseconds

        def advance_ms(self, milliseconds: float) -> None:
            with self._lock:
                self._nanoseconds += int(milliseconds * 1_000_000)

    clock = ManualClock()
    front_started = threading.Event()
    release_front = threading.Event()
    timing_written = threading.Event()
    timing_calls: list[tuple[Path, LiveCycleTiming]] = []
    lifecycle_events: list[str] = []
    main_thread = threading.get_ident()
    worker_threads: list[int] = []
    gui_threads: list[int] = []
    rendered_phase = DemoUiPhase.IDLE
    result_frame_submitted = False
    wait_count = 0

    front_images = {view: np.zeros((4, 5, 3), dtype=np.uint8) for view in VIEW_ORDER[:4]}
    back_images = {view: np.zeros((4, 5, 3), dtype=np.uint8) for view in VIEW_ORDER[4:]}

    class FakeCamera:
        last_sources = {view: object() for view in VIEW_ORDER}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def capture_round(self, round_name: str):
            assert threading.get_ident() == main_thread
            clock.advance_ms(100.0)
            if round_name == "back":
                release_front.set()
                return back_images
            return front_images

    class FakeModels:
        def inspect_round(self, _images, round_name: str):
            worker_threads.append(threading.get_ident())
            if round_name == "front":
                front_started.set()
                assert release_front.wait(timeout=2)
                clock.advance_ms(200.0)
            else:
                clock.advance_ms(300.0)
            return f"{round_name}-partial"

        def finalize_rounds(self, images, front, back, capture_id):
            worker_threads.append(threading.get_ident())
            assert (front, back) == ("front-partial", "back-partial")
            clock.advance_ms(50.0)
            return SimpleNamespace(capture_id=capture_id, images=images)

    def fake_wait_key(_delay: int) -> int:
        nonlocal wait_count
        wait_count += 1
        if wait_count == 1:
            return ord(" ")
        if wait_count == 2:
            assert front_started.wait(timeout=2)
            clock.advance_ms(200.0)
            return ord(" ")
        if timing_written.wait(timeout=0.001):
            return ord("q")
        return ord("q") if wait_count > 2_000 else -1

    def fake_render(state):
        nonlocal rendered_phase
        rendered_phase = state.phase
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def fake_imshow(*_args) -> None:
        nonlocal result_frame_submitted
        gui_threads.append(threading.get_ident())
        if rendered_phase is DemoUiPhase.RESULT and not result_frame_submitted:
            result_frame_submitted = True
            clock.advance_ms(20.0)
            lifecycle_events.append("result_frame")

    def fake_persist(_config, _inspection, _sources) -> None:
        worker_threads.append(threading.get_ident())
        lifecycle_events.append("persist")
        clock.advance_ms(150.0)

    def fake_persist_timing(result_root: Path, timing: LiveCycleTiming) -> Path:
        timing_calls.append((result_root, timing))
        timing_written.set()
        return result_root / timing.capture_id / "cycle_timing.json"

    wall_times = iter(
        (
            datetime(2026, 8, 25, 12, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 8, 25, 12, 0, 1, 120000, tzinfo=timezone.utc),
        )
    )
    monkeypatch.setattr(entrypoint, "FourCameraHdrSession", lambda _path: FakeCamera())
    monkeypatch.setattr(entrypoint, "_initialize_gui_window", lambda *_args: None)
    monkeypatch.setattr(entrypoint, "render_eight_view_screen", fake_render)
    monkeypatch.setattr(entrypoint.cv2, "imshow", fake_imshow)
    monkeypatch.setattr(entrypoint.cv2, "waitKey", fake_wait_key)
    monkeypatch.setattr(entrypoint.cv2, "destroyAllWindows", lambda: gui_threads.append(threading.get_ident()))
    monkeypatch.setattr(entrypoint, "persist_inspection", fake_persist)
    monkeypatch.setattr(entrypoint, "persist_cycle_timing", fake_persist_timing)
    monkeypatch.setattr(entrypoint, "preferred_selection", lambda _inspection: ("front", DemoBranch.TEMPLATE))

    status = entrypoint._run_gui(
        SimpleNamespace(experiment_mode=False, save_screenshot=None),
        SimpleNamespace(capture_config="capture.json", result_root=tmp_path),
        FakeModels(),
        None,
        clock_ns=clock,
        wall_clock=lambda: next(wall_times),
    )

    assert status == 0
    assert result_frame_submitted is True
    assert len(timing_calls) == 1
    result_root, timing = timing_calls[0]
    assert result_root == tmp_path
    assert timing.front_capture_ms == 100.0
    assert timing.flip_wait_ms == 200.0
    assert timing.back_capture_ms == 100.0
    assert timing.front_inference_ms == 500.0
    assert timing.back_inference_ms == 300.0
    assert timing.finalize_ms == 50.0
    assert timing.persist_ms == 150.0
    assert timing.result_display_ms == 20.0
    assert timing.front_overlap_ms == 300.0
    assert timing.total_cycle_ms == 970.0
    assert lifecycle_events == ["result_frame", "persist"]
    assert set(gui_threads) == {main_thread}
    assert len(set(worker_threads)) == 1
    assert worker_threads[0] != main_thread


def test_reset_discards_completion_from_an_older_gui_generation(monkeypatch) -> None:
    final_started = threading.Event()
    release_final = threading.Event()
    persisted = threading.Event()
    reset_sent = False
    wait_count = 0
    rendered_phases: list[DemoUiPhase] = []
    phase_count_at_reset = 0

    front_images = {view: np.zeros((4, 5, 3), dtype=np.uint8) for view in VIEW_ORDER[:4]}
    back_images = {view: np.zeros((4, 5, 3), dtype=np.uint8) for view in VIEW_ORDER[4:]}
    inspection = SimpleNamespace(capture_id="stale", images={**front_images, **back_images})

    class FakeCamera:
        last_sources = {view: object() for view in VIEW_ORDER}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def capture_round(self, round_name: str):
            return front_images if round_name == "front" else back_images

    class FakeModels:
        def inspect(self, _images, *, capture_id):
            raise RuntimeError(f"legacy synchronous path used for {capture_id}")

        def inspect_round(self, _images, round_name: str):
            return round_name

        def finalize_rounds(self, _images, _front, _back, _capture_id):
            final_started.set()
            assert release_final.wait(timeout=2)
            return inspection

    def fake_render(state):
        rendered_phases.append(state.phase)
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def fake_wait_key(_delay: int) -> int:
        nonlocal phase_count_at_reset, reset_sent, wait_count
        wait_count += 1
        if wait_count > 2_000:
            return ord("q")
        if wait_count in {1, 2}:
            return ord(" ")
        if not reset_sent:
            if final_started.is_set():
                reset_sent = True
                phase_count_at_reset = len(rendered_phases)
                return ord("r")
            threading.Event().wait(0.001)
            return -1
        if not release_final.is_set():
            release_final.set()
            assert persisted.wait(timeout=2)
            return -1
        return ord("q")

    monkeypatch.setattr(entrypoint, "FourCameraHdrSession", lambda _path: FakeCamera())
    monkeypatch.setattr(entrypoint, "_initialize_gui_window", lambda *_args: None)
    monkeypatch.setattr(entrypoint, "render_eight_view_screen", fake_render)
    monkeypatch.setattr(entrypoint.cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(entrypoint.cv2, "waitKey", fake_wait_key)
    monkeypatch.setattr(entrypoint.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(entrypoint, "persist_inspection", lambda *_args: persisted.set())
    monkeypatch.setattr(entrypoint, "preferred_selection", lambda _inspection: ("front", DemoBranch.TEMPLATE))

    status = entrypoint._run_gui(
        SimpleNamespace(experiment_mode=False, save_screenshot=None),
        SimpleNamespace(capture_config="capture.json"),
        FakeModels(),
        None,
    )

    assert status == 0
    assert final_started.is_set()
    assert persisted.is_set()
    assert DemoUiPhase.RESULT not in rendered_phases[phase_count_at_reset:]


def test_live_gui_waits_for_model_worker_before_closing_camera(monkeypatch) -> None:
    front_started = threading.Event()
    release_front = threading.Event()
    order: list[str] = []
    wait_count = 0
    front_images = {view: np.zeros((4, 5, 3), dtype=np.uint8) for view in VIEW_ORDER[:4]}

    class FakeCamera:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            order.append("camera_exit")
            return False

        def capture_round(self, round_name: str):
            assert round_name == "front"
            return front_images

    class FakeModels:
        def inspect_round(self, _images, round_name: str):
            assert round_name == "front"
            front_started.set()
            assert release_front.wait(timeout=2)
            order.append("worker_done")
            return "front"

    def fake_wait_key(_delay: int) -> int:
        nonlocal wait_count
        wait_count += 1
        if wait_count == 1:
            return ord(" ")
        assert front_started.wait(timeout=2)
        threading.Timer(0.02, release_front.set).start()
        return ord("q")

    monkeypatch.setattr(entrypoint, "FourCameraHdrSession", lambda _path: FakeCamera())
    monkeypatch.setattr(entrypoint, "_initialize_gui_window", lambda *_args: None)
    monkeypatch.setattr(entrypoint, "render_eight_view_screen", lambda _state: np.zeros((2, 2, 3), dtype=np.uint8))
    monkeypatch.setattr(entrypoint.cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(entrypoint.cv2, "waitKey", fake_wait_key)
    monkeypatch.setattr(entrypoint.cv2, "destroyAllWindows", lambda: None)

    status = entrypoint._run_gui(
        SimpleNamespace(experiment_mode=False, save_screenshot=None),
        SimpleNamespace(capture_config="capture.json"),
        FakeModels(),
        None,
    )

    assert status == 0
    assert order == ["worker_done", "camera_exit"]
