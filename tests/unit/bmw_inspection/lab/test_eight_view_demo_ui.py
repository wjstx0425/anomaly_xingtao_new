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
    _trusted_ok_available,
    _trusted_reference_details,
    evidence_comparison_images,
    render_eight_view_dashboard,
    step_actionable_selection,
    toggle_trusted_ok_mode,
)
from bmw_inspection.lab.trusted_ok_reference import TrustedOkMatch


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


def _match(inspection: EightViewInspection, view: str) -> TrustedOkMatch:
    current = inspection.images[view]
    return TrustedOkMatch(
        view_id=view,
        physical_part_id="normal-train-001",
        sample_id="normal-train-001_000001",
        similarity=0.9234,
        shift_x=3,
        shift_y=-2,
        current_full_image=current,
        reference_full_image=np.full_like(current, 40),
        current_roi=np.full((12, 16, 3), 50, dtype=np.uint8),
        reference_roi=np.full((12, 16, 3), 60, dtype=np.uint8),
        aligned_reference_roi=np.full((12, 16, 3), 70, dtype=np.uint8),
        difference_overlay=np.full((12, 16, 3), 80, dtype=np.uint8),
        source_sha256="a" * 64,
        reference_full_sha256="b" * 64,
        reference_roi_sha256="c" * 64,
        index_sha256="d" * 64,
        whitelist_sha256="e" * 64,
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


def test_o_toggle_requires_inspection_and_preserves_model_results() -> None:
    idle = EightViewUiState()
    assert toggle_trusted_ok_mode(idle) is idle
    legacy_inspection = _inspection()
    legacy = EightViewUiState(phase=DemoUiPhase.RESULT, inspection=legacy_inspection)
    assert _trusted_ok_available(legacy) is False
    assert toggle_trusted_ok_mode(legacy) is legacy
    match = _match(legacy_inspection, "front_right")
    inspection = EightViewInspection(
        legacy_inspection.capture_id,
        legacy_inspection.images,
        legacy_inspection.results,
        legacy_inspection.final_status,
        legacy_inspection.elapsed_ms,
        trusted_ok_by_view={"front_right": match},
    )
    state = EightViewUiState(phase=DemoUiPhase.RESULT, inspection=inspection)
    assert _trusted_ok_available(state) is True

    enabled = toggle_trusted_ok_mode(state)
    disabled = toggle_trusted_ok_mode(enabled)

    assert enabled.trusted_ok_mode is True
    assert disabled.trusted_ok_mode is False
    assert enabled.inspection is inspection
    assert enabled.inspection.results is inspection.results


def test_trusted_ok_mode_uses_four_comparison_panels_and_selected_model_evidence() -> None:
    base = _inspection()
    match = _match(base, "front_right")
    inspection = EightViewInspection(
        base.capture_id, base.images, base.results, base.final_status, base.elapsed_ms,
        trusted_ok_by_view={"front_right": match},
    )
    state = EightViewUiState(
        phase=DemoUiPhase.RESULT,
        inspection=inspection,
        selected_view="front_right",
        selected_branch=DemoBranch.YOLO,
        trusted_ok_mode=True,
    )

    panels = evidence_comparison_images(state)

    assert tuple(label for label, _image in panels) == (
        "现场NG", "可信OK", "对齐差异", "真实检测框"
    )
    assert tuple(int(image.mean()) for _label, image in panels) == (50, 70, 80, 20)
    assert _trusted_reference_details(state) == (
        ("参考零件", "normal-train-001"),
        ("参考样本", "normal-train-001_000001"),
        ("相似度", "0.923400"),
        ("对齐平移", "(+3, -2)"),
    )
    assert render_eight_view_dashboard(state).shape == (900, 1600, 3)


def test_trusted_ok_mode_explicitly_reports_missing_reference() -> None:
    state = EightViewUiState(
        phase=DemoUiPhase.RESULT,
        inspection=_inspection(),
        selected_view="front_right",
        selected_branch=DemoBranch.YOLO,
        trusted_ok_mode=True,
    )

    panels = evidence_comparison_images(state)
    dashboard = render_eight_view_dashboard(state)

    assert tuple(label for label, _image in panels) == (
        "现场NG", "可信OK", "对齐差异", "真实检测框"
    )
    assert panels[1][1] is None and panels[2][1] is None
    assert _trusted_reference_details(state) == (("可信参考", "无可信OK参考"),)
    assert dashboard.shape == (900, 1600, 3)
