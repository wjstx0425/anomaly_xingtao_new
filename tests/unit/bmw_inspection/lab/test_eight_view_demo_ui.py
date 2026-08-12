"""Focused renderer checks for the independent eight-view Demo."""

from __future__ import annotations

import numpy as np

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER
from bmw_inspection.lab.eight_view_demo import (
    BranchStatus,
    DemoBranch,
    DemoBranchResult,
    DemoFinalStatus,
    EightViewInspection,
)
from bmw_inspection.lab.eight_view_demo_ui import (
    DemoUiPhase,
    EightViewUiState,
    _demo_font_path,
    evidence_comparison_images,
    render_eight_view_dashboard,
    step_actionable_selection,
)


def _inspection() -> EightViewInspection:
    images = {view: np.full((24, 32, 3), index, dtype=np.uint8) for index, view in enumerate(VIEW_ORDER)}
    rows = (
        DemoBranchResult(
            DemoBranch.TEMPLATE,
            "front",
            BranchStatus.PASS,
            0.1,
            0.2,
            1.0,
            "模板通过，但说明文本完整保留",
            np.full((10, 10, 3), 10, dtype=np.uint8),
            details={"evidence_type": "诊断热区"},
        ),
        DemoBranchResult(
            DemoBranch.YOLO,
            "front_right",
            BranchStatus.NG,
            0.8,
            0.25,
            2.0,
            "发现一个置信度为0.8的真实缺陷框，必须完整显示原因",
            np.full((10, 10, 3), 20, dtype=np.uint8),
            details={"evidence_type": "真实检测框", "threshold_exceedance": 0.55},
        ),
        DemoBranchResult(
            DemoBranch.EFFICIENTAD,
            "back_left",
            BranchStatus.ERROR,
            None,
            None,
            3.0,
            "异常模型的完整错误原因",
            None,
            details={"evidence_type": "诊断热区"},
        ),
    )
    return EightViewInspection("capture", images, rows, DemoFinalStatus.ERROR, 6.0)


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


def test_actionable_results_and_step_selection_wrap_in_result_order() -> None:
    inspection = _inspection()

    assert [(row.view_id, row.branch) for row in inspection.actionable_results()] == [
        ("front_right", DemoBranch.YOLO),
        ("back_left", DemoBranch.EFFICIENTAD),
    ]
    assert step_actionable_selection(inspection, "front_right", DemoBranch.YOLO, 1) == (
        "back_left",
        DemoBranch.EFFICIENTAD,
    )
    assert step_actionable_selection(inspection, "back_left", DemoBranch.EFFICIENTAD, 1) == (
        "front_right",
        DemoBranch.YOLO,
    )
    assert step_actionable_selection(inspection, "front_right", DemoBranch.YOLO, -1) == (
        "back_left",
        DemoBranch.EFFICIENTAD,
    )


def test_evidence_comparison_exposes_short_long_hdr_and_model_evidence() -> None:
    inspection = _inspection()
    sources = {
        view: {
            "short": np.full((8, 8, 3), 30, dtype=np.uint8),
            "long": np.full((8, 8, 3), 60, dtype=np.uint8),
            "hdr": inspection.images[view],
        }
        for view in VIEW_ORDER
    }
    state = EightViewUiState(
        phase=DemoUiPhase.RESULT,
        inspection=inspection,
        selected_view="front_right",
        selected_branch=DemoBranch.YOLO,
        source_images=sources,
        experiment_mode=True,
    )

    panels = evidence_comparison_images(state)

    assert tuple(label for label, _image in panels) == ("短曝光", "长曝光", "融合 HDR", "真实检测框")
    assert tuple(int(image.mean()) for _label, image in panels) == (30, 60, 2, 20)
    assert render_eight_view_dashboard(state).shape == (900, 1600, 3)


def test_evidence_comparison_does_not_present_offline_fused_image_as_real_exposures() -> None:
    inspection = _inspection()
    sources = {
        view: {
            "short": inspection.images[view],
            "long": inspection.images[view],
            "hdr": inspection.images[view],
            "source_kind": "fused_only",
        }
        for view in VIEW_ORDER
    }
    state = EightViewUiState(
        phase=DemoUiPhase.RESULT,
        inspection=inspection,
        selected_view="front_right",
        selected_branch=DemoBranch.YOLO,
        source_images=sources,
    )

    panels = evidence_comparison_images(state)

    assert tuple(label for label, _image in panels[:3]) == (
        "无短曝光原图",
        "无长曝光原图",
        "历史融合图",
    )
    assert panels[0][1] is None
    assert panels[1][1] is None
    assert panels[2][1] is inspection.images["front_right"]


def test_bright_streak_evidence_is_rotated_clockwise_for_display() -> None:
    images = {view: np.zeros((24, 32, 3), dtype=np.uint8) for view in VIEW_ORDER}
    overlay = np.zeros((5, 20, 3), dtype=np.uint8)
    overlay[:, 0] = 255
    row = DemoBranchResult(
        DemoBranch.BRIGHT_STREAK,
        "front_left",
        BranchStatus.NG,
        0.0,
        0.1,
        1.0,
        "未检测到光痕",
        overlay,
        details={"evidence_type": "规则 ROI 证据"},
    )
    inspection = EightViewInspection("bright", images, (row,), DemoFinalStatus.NG, 1.0)
    state = EightViewUiState(
        phase=DemoUiPhase.RESULT,
        inspection=inspection,
        selected_view="front_left",
        selected_branch=DemoBranch.BRIGHT_STREAK,
    )

    evidence = evidence_comparison_images(state)[3][1]

    assert evidence is not None
    assert evidence.shape == (20, 5, 3)
    assert np.all(evidence[0] == 255)
