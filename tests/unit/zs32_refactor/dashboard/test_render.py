# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pure canvas contracts for the ZS32 eight-view dashboard."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from zs32_inspection.dashboard.app import DashboardState
from zs32_inspection.dashboard.contracts import EvidenceLayer, ProgressRecord
from zs32_inspection.dashboard.parser import load_inspection_result
from zs32_inspection.dashboard.render import inspection_button_label, render_dashboard


@pytest.fixture
def state(eight_view_result_dir: Path) -> DashboardState:
    return DashboardState(result=load_inspection_result(eight_view_result_dir))


def test_render_has_eight_cards_and_single_detection_action(state: DashboardState) -> None:
    frame = render_dashboard(state)

    assert frame.canvas.shape == (920, 1600, 3)
    assert frame.canvas.dtype == np.uint8
    assert [hit.action for hit in frame.hit_regions].count("select_view") == 8
    assert [hit.action for hit in frame.hit_regions].count("inspection_action") == 1
    assert [hit.action for hit in frame.hit_regions].count("quit") == 1
    assert len([hit for hit in frame.hit_regions if hit.action == "select_layer"]) == 5


def test_render_uses_four_by_two_front_back_bands(state: DashboardState) -> None:
    frame = render_dashboard(state)
    cards = [hit for hit in frame.hit_regions if hit.action == "select_view"]

    assert [card.value for card in cards] == [view.view for view in state.result.views]
    assert len({card.rect.y for card in cards[:4]}) == 1
    assert len({card.rect.y for card in cards[4:]}) == 1
    assert cards[0].rect.y < cards[4].rect.y
    assert all(card.rect.width >= 48 and card.rect.height >= 48 for card in cards)


def test_enlarged_view_keeps_identity_status_and_notice(state: DashboardState) -> None:
    state = replace(state, selected_view="front_secondary", layer=EvidenceLayer.FUSION)

    frame = render_dashboard(state)

    assert frame.selected_view == "front_secondary"
    assert frame.notice == "暂未接入模型"
    assert frame.canvas.shape == (920, 1600, 3)
    assert not [hit for hit in frame.hit_regions if hit.action == "select_view"]


@pytest.mark.parametrize(
    ("progress", "running", "expected"),
    [
        (None, False, ("开始检测 [S]", True)),
        (ProgressRecord("p", None, "waiting_front", "front", "now"), True, ("确认正面并拍摄 [S]", True)),
        (ProgressRecord("p", None, "waiting_back", "back", "now"), True, ("确认背面并拍摄 [S]", True)),
        (ProgressRecord("p", None, "inference", "running", "now"), True, ("检测运行中", False)),
        (None, True, ("检测运行中", False)),
    ],
)
def test_inspection_button_label_is_contextual(
    progress: ProgressRecord | None,
    running: bool,
    expected: tuple[str, bool],
) -> None:
    assert inspection_button_label(progress, running) == expected


def test_running_non_waiting_button_has_clear_disabled_region(state: DashboardState) -> None:
    state = replace(
        state,
        running=True,
        progress=ProgressRecord("ZS32-0001", None, "inference", "检测中", "now"),
    )

    frame = render_dashboard(state)

    assert frame.inspection_button.enabled is False
    assert [hit.action for hit in frame.hit_regions].count("inspection_action") == 1
    assert frame.inspection_button.rect.height >= 48
