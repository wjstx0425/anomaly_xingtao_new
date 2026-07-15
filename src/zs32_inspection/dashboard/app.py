# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Reducer and optional OpenCV HighGUI loop for the ZS32 dashboard."""

from __future__ import annotations

from dataclasses import replace
from enum import StrEnum
from pathlib import Path

import cv2

from .contracts import VIEW_ORDER, EvidenceLayer
from .live import Stage35Controller
from .parser import load_inspection_result
from .render import DashboardState, HitRegion, RenderFrame, render_dashboard

_WINDOW_NAME = "ZS32 Inspection Dashboard"


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
        enabled = not state.running or (
            state.progress is not None and state.progress.state in {"waiting_front", "waiting_back"}
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
            controller.start(part_id)
        state = replace(state, inspection_requested=False)
    progress = controller.poll()
    result = state.result
    if progress is not None and progress.state == "complete":
        result = replace(load_inspection_result(Path(progress.message)), mode="live")
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
        state = service_live_state(state, live_controller, part_id)
    frame = render_dashboard(state)
    if save_screenshot is not None:
        _save_screenshot(frame, save_screenshot)
    if no_gui:
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
            frame = render_dashboard(state)
            current["frame"] = frame
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
