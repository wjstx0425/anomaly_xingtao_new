# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Reducer and optional OpenCV HighGUI loop for the ZS32 dashboard."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import cv2

from zs32_inspection.timing import TimingRecorder, merge_timing_payloads

from .contracts import VIEW_ORDER, EvidenceLayer
from .live import Stage35Controller
from .parser import load_inspection_result
from .render import DashboardState, HitRegion, RenderFrame, render_dashboard

_WINDOW_NAME = "ZS32 Inspection Dashboard"
_DASHBOARD_TIMING_STAGES = frozenset(
    {
        "dashboard_result_parse",
        "dashboard_first_frame_render",
        "dashboard_first_frame_display",
    },
)


@dataclass(slots=True)
class _DashboardTimingSession:
    """Track the first parsed and displayed frame for one completed result."""

    result_dir: Path
    recorder: TimingRecorder
    render_recorded: bool = False
    display_recorded: bool = False
    request_started: float | None = None


_DASHBOARD_TIMINGS: dict[Path, _DashboardTimingSession] = {}


def _timing_session(result_dir: Path, *, request_started: float | None = None) -> _DashboardTimingSession:
    resolved = result_dir.expanduser().resolve()
    session = _DashboardTimingSession(resolved, TimingRecorder("dashboard"), request_started=request_started)
    _DASHBOARD_TIMINGS[resolved] = session
    return session


def _timing_for_state(state: DashboardState) -> _DashboardTimingSession | None:
    progress = state.progress
    if progress is None or progress.state != "complete":
        return None
    try:
        result_dir = Path(progress.message).expanduser().resolve()
    except (OSError, RuntimeError):
        return None
    return _DASHBOARD_TIMINGS.get(result_dir)


def _load_existing_timing(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _publish_dashboard_timing(session: _DashboardTimingSession) -> None:
    """Best-effort merge dashboard timing without making the UI fail."""
    path = session.result_dir / "timing.json"
    temporary = path.with_name(f".{path.name}.tmp-dashboard")
    try:
        existing = _load_existing_timing(path)
        new_payload = session.recorder.payload()
        existing_stages = existing.get("stages", {})
        if isinstance(existing_stages, Mapping):
            new_payload["stages"] = {
                name: value
                for name, value in new_payload["stages"].items()
                if name not in _DASHBOARD_TIMING_STAGES or name not in existing_stages
            }
        try:
            total_seconds = (
                time.perf_counter() - session.request_started if session.request_started is not None else None
            )
            merged = merge_timing_payloads(existing, new_payload, total_seconds=total_seconds)
        except (TypeError, ValueError):
            merged = merge_timing_payloads(new_payload)
        temporary.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    except Exception:  # noqa: BLE001 - timing must never make the dashboard fail
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    finally:
        _DASHBOARD_TIMINGS.pop(session.result_dir, None)


def _render_with_timing(state: DashboardState) -> tuple[RenderFrame, _DashboardTimingSession | None]:
    session = _timing_for_state(state)
    if session is None or session.render_recorded:
        return render_dashboard(state), session
    with session.recorder.measure("dashboard_first_frame_render"):
        frame = render_dashboard(state)
    session.render_recorded = True
    return frame, session


class Action(StrEnum):
    """All keyboard and mouse actions accepted by the reducer."""

    INSPECTION = "inspection"
    QUIT = "quit"
    ESCAPE = "escape"
    SELECT_VIEW = "select_view"
    SELECT_FUSION = "select_fusion"
    SELECT_ORIGINAL = "select_original"
    SELECT_PATCHCORE = "select_patchcore"
    SELECT_YOLO = "select_yolo"
    SELECT_TEMPLATE = "select_template"


_KEY_ACTIONS = {
    ord("s"): Action.INSPECTION,
    ord("S"): Action.INSPECTION,
    ord("q"): Action.QUIT,
    ord("Q"): Action.QUIT,
    27: Action.ESCAPE,
    ord("1"): Action.SELECT_FUSION,
    ord("2"): Action.SELECT_ORIGINAL,
    ord("3"): Action.SELECT_PATCHCORE,
    ord("4"): Action.SELECT_YOLO,
    ord("5"): Action.SELECT_TEMPLATE,
}

_LAYER_ACTIONS = {
    Action.SELECT_FUSION: EvidenceLayer.FUSION,
    Action.SELECT_ORIGINAL: EvidenceLayer.ORIGINAL,
    Action.SELECT_PATCHCORE: EvidenceLayer.PATCHCORE,
    Action.SELECT_YOLO: EvidenceLayer.YOLO,
    Action.SELECT_TEMPLATE: EvidenceLayer.TEMPLATE,
}


def key_to_action(key: int) -> Action | None:
    """Map one HighGUI key code to a reducer action."""
    return _KEY_ACTIONS.get(key & 0xFF) if key >= 0 else None


def _hit_at(point: tuple[int, int], hit_regions: tuple[HitRegion, ...]) -> HitRegion | None:
    return next((hit for hit in reversed(hit_regions) if hit.rect.contains(point)), None)


def _hit_action(hit: HitRegion | None) -> Action | None:
    if hit is None or not hit.enabled:
        return None
    if hit.action == "inspection_action":
        return Action.INSPECTION
    if hit.action == "quit":
        return Action.QUIT
    if hit.action == "select_view":
        return Action.SELECT_VIEW
    if hit.action == "select_layer" and hit.value is not None:
        return {
            EvidenceLayer.FUSION.value: Action.SELECT_FUSION,
            EvidenceLayer.ORIGINAL.value: Action.SELECT_ORIGINAL,
            EvidenceLayer.PATCHCORE.value: Action.SELECT_PATCHCORE,
            EvidenceLayer.YOLO.value: Action.SELECT_YOLO,
            EvidenceLayer.TEMPLATE.value: Action.SELECT_TEMPLATE,
        }.get(hit.value)
    return None


def mouse_to_action(point: tuple[int, int], hit_regions: tuple[HitRegion, ...]) -> Action | None:
    """Map a click point to the same reducer actions used by keys."""
    return _hit_action(_hit_at(point, hit_regions))


def reduce_state(state: DashboardState, action: Action, *, value: str | None = None) -> DashboardState:
    """Apply one action without side effects."""
    if action in _LAYER_ACTIONS:
        return replace(state, layer=_LAYER_ACTIONS[action])
    if action is Action.SELECT_VIEW:
        if value not in VIEW_ORDER:
            raise ValueError("selected view must be in VIEW_ORDER")
        return replace(state, selected_view=value)
    if action is Action.INSPECTION:
        live_enabled = state.result is None or state.result.mode == "live"
        enabled = live_enabled and (
            not state.running
            or (state.progress is not None and state.progress.state in {"waiting_front", "waiting_back"})
        )
        return replace(state, inspection_requested=True) if enabled else state
    if action is Action.ESCAPE:
        return replace(state, selected_view=None) if state.selected_view else replace(state, exit_requested=True)
    if action is Action.QUIT:
        return replace(state, exit_requested=True)
    raise ValueError(f"unsupported dashboard action: {action!r}")


def _save_screenshot(frame: RenderFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame.canvas):
        raise OSError(f"failed to save dashboard screenshot: {path}")


def service_live_state(
    state: DashboardState,
    controller: Stage35Controller,
    part_id: str,
) -> DashboardState:
    """Apply one contextual CTA and one progress poll without blocking the UI."""
    if state.inspection_requested:
        if controller.running:
            controller.confirm()
        else:
            setattr(controller, "_dashboard_request_started", time.perf_counter())
            start_worker = getattr(controller, "start_worker", None)
            if callable(start_worker):
                start_worker()
            if getattr(controller, "worker_ready", True):
                controller.start(part_id)
        state = replace(state, inspection_requested=False)
    progress = controller.poll()
    result = state.result
    result_changed = (
        progress is not None
        and progress.state == "complete"
        and (
            result is None
            or state.progress is None
            or state.progress.state != "complete"
            or state.progress.message != progress.message
        )
    )
    if result_changed:
        assert progress is not None
        request_started = getattr(controller, "_dashboard_request_started", None)
        session = _timing_session(
            Path(progress.message),
            request_started=request_started if isinstance(request_started, float) else None,
        )
        setattr(controller, "_dashboard_request_started", None)
        try:
            with session.recorder.measure("dashboard_result_parse"):
                result = replace(load_inspection_result(session.result_dir), mode="live")
        except Exception:  # noqa: BLE001 - discard session and preserve the parser error
            _DASHBOARD_TIMINGS.pop(session.result_dir, None)
            raise
    return replace(state, progress=progress, running=controller.running, result=result)


def run_dashboard(
    initial_state: DashboardState,
    *,
    no_gui: bool = False,
    save_screenshot: Path | None = None,
    live_controller: Stage35Controller | None = None,
    part_id: str | None = None,
) -> DashboardState:
    """Render once headlessly or run the 50 ms HighGUI event loop."""
    state = initial_state
    if live_controller is not None:
        if not part_id:
            raise ValueError("part_id is required for live dashboard mode")
        start_worker = getattr(live_controller, "start_worker", None)
        if callable(start_worker):
            start_worker()
        state = service_live_state(state, live_controller, part_id)
    frame, timing_session = _render_with_timing(state)
    if save_screenshot is not None:
        _save_screenshot(frame, save_screenshot)
    if no_gui:
        if timing_session is not None:
            _publish_dashboard_timing(timing_session)
        if live_controller is not None:
            live_controller.close()
        return state

    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_WINDOW_NAME, 1600, 920)
    current: dict[str, RenderFrame] = {"frame": frame}
    mouse_events: list[tuple[Action, str | None]] = []

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event != cv2.EVENT_LBUTTONUP:
            return
        hit = _hit_at((x, y), current["frame"].hit_regions)
        action = _hit_action(hit)
        if action is not None:
            mouse_events.append((action, None if hit is None else hit.value))

    cv2.setMouseCallback(_WINDOW_NAME, on_mouse)
    try:
        while not state.exit_requested:
            if live_controller is not None:
                assert part_id is not None
                state = service_live_state(state, live_controller, part_id)
            frame, timing_session = _render_with_timing(state)
            current["frame"] = frame
            if timing_session is not None and not timing_session.display_recorded:
                with timing_session.recorder.measure("dashboard_first_frame_display"):
                    cv2.imshow(_WINDOW_NAME, frame.canvas)
                timing_session.display_recorded = True
                _publish_dashboard_timing(timing_session)
            else:
                cv2.imshow(_WINDOW_NAME, frame.canvas)
            key_action = key_to_action(cv2.waitKey(50))
            if key_action is not None:
                state = reduce_state(state, key_action)
            while mouse_events:
                action, value = mouse_events.pop(0)
                state = reduce_state(state, action, value=value)
    finally:
        if live_controller is not None:
            live_controller.close()
        cv2.destroyWindow(_WINDOW_NAME)
    return state
