"""Focused renderer checks for the independent eight-view Demo."""

from __future__ import annotations

import numpy as np

from bmw_inspection.lab.eight_view_demo_ui import (
    DemoUiPhase,
    EightViewUiState,
    _demo_font_path,
    render_eight_view_dashboard,
)


def test_demo_uses_distinct_medium_and_bold_cjk_fonts() -> None:
    medium = _demo_font_path("medium")
    bold = _demo_font_path("bold")

    assert medium.name == "NotoSansCJK-Medium.ttc"
    assert bold.name == "NotoSansCJK-Bold.ttc"
    assert medium != bold


def test_idle_dashboard_is_blank_and_uses_fixed_presentation_size() -> None:
    dashboard = render_eight_view_dashboard(EightViewUiState(phase=DemoUiPhase.IDLE))

    assert dashboard.shape == (900, 1600, 3)
    assert dashboard.dtype == np.uint8
