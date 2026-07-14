# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared keyboard, mouse, reducer, and HighGUI loop contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import pytest

from zs32_inspection.dashboard.app import (
    Action,
    DashboardState,
    key_to_action,
    mouse_to_action,
    reduce_state,
    run_dashboard,
)
from zs32_inspection.dashboard.contracts import EvidenceLayer, ProgressRecord
from zs32_inspection.dashboard.parser import load_inspection_result
from zs32_inspection.dashboard.render import render_dashboard


@pytest.fixture
def state(eight_view_result_dir: Path) -> DashboardState:
    return DashboardState(result=load_inspection_result(eight_view_result_dir))


def test_s_key_and_detection_button_share_one_action(state: DashboardState) -> None:
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
