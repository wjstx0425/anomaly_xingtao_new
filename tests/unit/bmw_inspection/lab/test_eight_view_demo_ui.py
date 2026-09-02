"""Focused renderer checks for the independent eight-view Demo."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER
from bmw_inspection.lab.bright_streak_rotated_roi import (
    RotatedBrightStreakRoi,
    rectify_bright_streak_roi,
)
from bmw_inspection.lab.eight_view_demo import (
    BranchStatus,
    DemoBranch,
    DemoBranchResult,
    DemoFinalStatus,
    EightViewInspection,
)
from bmw_inspection.lab.eight_view_demo_ui import (
    DashboardHit,
    DemoUiPage,
    DemoUiPhase,
    EightViewUiState,
    _demo_font_path,
    _experiment_threshold_details,
    _result_timing_lines,
    _trusted_ok_available,
    _trusted_reference_details,
    apply_dashboard_click,
    dashboard_hit_test,
    detail_controls_layout,
    detail_reason_lines,
    detail_reference_footer,
    evidence_detail_images,
    evidence_comparison_images,
    render_eight_view_dashboard,
    render_eight_view_screen,
    select_branch,
    step_actionable_selection,
    toggle_trusted_ok_mode,
)
from bmw_inspection.lab.live_cycle_timing import LiveCycleTiming
from bmw_inspection.lab.trusted_ok_reference import TrustedOkMatch


def _inspection() -> EightViewInspection:
    images = {view: np.full((24, 32, 3), index, dtype=np.uint8) for index, view in enumerate(VIEW_ORDER)}
    rois = {view: np.full((12, 16, 3), 100 + index, dtype=np.uint8) for index, view in enumerate(VIEW_ORDER)}
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
    return EightViewInspection("capture", images, rows, DemoFinalStatus.ERROR, 6.0, roi_images=rois)


def _cycle_timing() -> LiveCycleTiming:
    return LiveCycleTiming(
        capture_id="capture",
        started_at="2026-08-25T12:00:00.000+08:00",
        displayed_at="2026-08-25T12:00:12.000+08:00",
        front_capture_ms=1200.0,
        flip_wait_ms=2500.0,
        back_capture_ms=1300.0,
        front_inference_ms=3900.0,
        back_inference_ms=3800.0,
        finalize_ms=200.0,
        persist_ms=5100.0,
        result_display_ms=30.0,
        front_overlap_ms=3800.0,
        total_cycle_ms=12000.0,
    )


def test_result_timing_lines_show_total_and_compact_breakdown() -> None:
    assert _result_timing_lines(_cycle_timing()) == (
        "完整周期 12.000 s",
        "采F/B 1.20/1.30  翻2.50  推F/B 3.90/3.80  融0.20  存5.10",
    )


def _match(
    inspection: EightViewInspection,
    view: str,
    comparison_mode: str = "roi",
    current_value: int = 50,
) -> TrustedOkMatch:
    current = inspection.images[view]
    region_shape = current.shape if comparison_mode == "full" else (12, 16, 3)
    return TrustedOkMatch(
        view_id=view,
        comparison_mode=comparison_mode,
        physical_part_id="normal-train-001",
        sample_id="normal-train-001_000001",
        similarity=0.9234,
        shift_x=3,
        shift_y=-2,
        current_full_image=current,
        reference_full_image=np.full_like(current, 40),
        current_roi=np.full(region_shape, current_value, dtype=np.uint8),
        reference_roi=np.full((12, 16, 3), 60, dtype=np.uint8),
        aligned_reference_roi=np.full((12, 16, 3), 70, dtype=np.uint8),
        difference_overlay=np.full((12, 16, 3), 80, dtype=np.uint8),
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


def test_efficientad_candidate_metadata_is_visible_in_ui_details() -> None:
    row = DemoBranchResult(
        DemoBranch.EFFICIENTAD,
        "front",
        BranchStatus.PASS,
        0.3,
        0.55,
        1.0,
        "candidate",
        None,
        details={
            "threshold_source": "legacy_reuse_for_0820_candidate",
            "validation_status": "pending_independent_validation",
        },
    )

    assert _experiment_threshold_details(row) == (
        ("阈值来源", "legacy_reuse_for_0820_candidate"),
        ("验证状态", "pending_independent_validation"),
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
        roi_images=legacy_inspection.roi_images,
        trusted_ok_by_comparison={("front_right", "roi"): match},
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
        roi_images=base.roi_images,
        trusted_ok_by_comparison={("front_right", "roi"): match},
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
    assert panels[0][1] is not None
    assert panels[0][1].shape[:2] == (12, 16)
    assert int(panels[0][1].mean()) == 102
    assert _trusted_reference_details(state) == (("可信参考", "无可信OK参考"),)
    assert dashboard.shape == (900, 1600, 3)


def test_trusted_mode_does_not_present_reference_for_manually_selected_pass_row() -> None:
    base = _inspection()
    match = _match(base, "front_right")
    inspection = EightViewInspection(
        base.capture_id, base.images, base.results, base.final_status, base.elapsed_ms,
        roi_images=base.roi_images,
        trusted_ok_by_comparison={("front_right", "roi"): match},
    )
    state = EightViewUiState(
        phase=DemoUiPhase.RESULT,
        inspection=inspection,
        selected_view="front",
        selected_branch=DemoBranch.TEMPLATE,
        trusted_ok_mode=True,
    )

    panels = evidence_comparison_images(state)

    assert tuple(label for label, _image in panels) == (
        "当前项目通过，无需NG对比", "无需可信OK参考", "无需对齐差异", "无需模型证据"
    )
    assert all(image is None for _label, image in panels)

    missing_row_panels = evidence_comparison_images(EightViewUiState(
        phase=DemoUiPhase.RESULT,
        inspection=inspection,
        selected_view="front",
        selected_branch=DemoBranch.BRIGHT_STREAK,
        trusted_ok_mode=True,
    ))
    assert missing_row_panels[0][0] == "当前项目无检测结果，无需NG对比"
    assert all(image is None for _label, image in missing_row_panels)


def test_trusted_mode_selects_front_left_full_or_roi_match_by_branch() -> None:
    images = {view: np.zeros((24, 32, 3), dtype=np.uint8) for view in VIEW_ORDER}
    rois = {view: np.full((12, 16, 3), 11, dtype=np.uint8) for view in VIEW_ORDER}
    rows = (
        DemoBranchResult(DemoBranch.TEMPLATE, "front_left", BranchStatus.NG, 0.3, 0.2, 1.0, "template ng", None),
        DemoBranchResult(DemoBranch.BRIGHT_STREAK, "front_left", BranchStatus.NG, 0.0, 0.1, 1.0, "streak ng", None),
    )
    base = EightViewInspection("dual", images, rows, DemoFinalStatus.NG, 2.0, roi_images=rois)
    roi_match = _match(base, "front_left", "roi", 21)
    full_match = _match(base, "front_left", "full", 31)
    inspection = EightViewInspection(
        "dual", images, rows, DemoFinalStatus.NG, 2.0, roi_images=rois,
        trusted_ok_by_comparison={
            ("front_left", "roi"): roi_match,
            ("front_left", "full"): full_match,
        },
    )

    template_panels = evidence_comparison_images(EightViewUiState(
        phase=DemoUiPhase.RESULT, inspection=inspection, selected_view="front_left",
        selected_branch=DemoBranch.TEMPLATE, trusted_ok_mode=True,
    ))
    streak_panels = evidence_comparison_images(EightViewUiState(
        phase=DemoUiPhase.RESULT, inspection=inspection, selected_view="front_left",
        selected_branch=DemoBranch.BRIGHT_STREAK, trusted_ok_mode=True,
    ))

    assert int(template_panels[0][1].mean()) == 21
    assert int(streak_panels[0][1].mean()) == 31
    assert template_panels[0][1].shape[:2] == (12, 16)
    assert streak_panels[0][1].shape[:2] == (24, 32)


def test_algorithm_card_prioritizes_error_then_ng_and_falls_back_to_first_pass() -> None:
    inspection = _inspection()
    state = EightViewUiState(
        phase=DemoUiPhase.RESULT,
        inspection=inspection,
        selected_view="front",
        selected_branch=DemoBranch.TEMPLATE,
    )

    yolo = select_branch(state, DemoBranch.YOLO)
    efficientad = select_branch(state, DemoBranch.EFFICIENTAD)
    template = select_branch(state, DemoBranch.TEMPLATE)

    assert (yolo.selected_view, yolo.selected_branch) == ("front_right", DemoBranch.YOLO)
    assert (efficientad.selected_view, efficientad.selected_branch) == ("back_left", DemoBranch.EFFICIENTAD)
    assert (template.selected_view, template.selected_branch) == ("front", DemoBranch.TEMPLATE)


def test_dashboard_view_card_only_changes_view_and_empty_evidence_panel_is_noop() -> None:
    state = EightViewUiState(
        phase=DemoUiPhase.RESULT,
        inspection=_inspection(),
        selected_view="front_right",
        selected_branch=DemoBranch.YOLO,
    )

    view_changed = apply_dashboard_click(state, dashboard_hit_test(300, 120))
    unchanged = apply_dashboard_click(state, DashboardHit(evidence_index=0))

    assert (view_changed.selected_view, view_changed.selected_branch) == ("front_left", DemoBranch.YOLO)
    assert view_changed.page is DemoUiPage.DASHBOARD
    assert unchanged is state


def test_dashboard_hit_regions_and_selected_borders_are_stable() -> None:
    state = EightViewUiState(
        phase=DemoUiPhase.RESULT,
        inspection=_inspection(),
        selected_view="front_right",
        selected_branch=DemoBranch.YOLO,
    )

    assert dashboard_hit_test(24, 92) == DashboardHit(view_id="front")
    assert dashboard_hit_test(254, 274) == DashboardHit(view_id="front")
    assert dashboard_hit_test(279, 92) == DashboardHit(view_id="front_left")
    assert dashboard_hit_test(1084, 258) == DashboardHit(branch=DemoBranch.TEMPLATE)
    assert dashboard_hit_test(1334, 366) == DashboardHit(branch=DemoBranch.BRIGHT_STREAK)
    assert dashboard_hit_test(24, 610) == DashboardHit(evidence_index=0)
    assert dashboard_hit_test(254, 840) == DashboardHit(evidence_index=0)
    assert dashboard_hit_test(23, 610) is None

    dashboard = render_eight_view_dashboard(state)
    assert tuple(int(value) for value in dashboard[124, 534]) == (167, 47, 0)
    assert tuple(int(value) for value in dashboard[390, 1084]) == (167, 47, 0)


def test_evidence_detail_pairs_keep_algorithm_coordinate_domains() -> None:
    base = _inspection()
    template_row = DemoBranchResult(
        DemoBranch.TEMPLATE,
        "front",
        BranchStatus.NG,
        0.3,
        0.2,
        1.0,
        "模板不通过",
        np.full((10, 10, 3), 10, dtype=np.uint8),
    )
    template_match = _match(base, "front")
    yolo_match = _match(base, "front_right")
    efficientad_match = _match(base, "back_left")
    reference_full = np.zeros((10, 20, 3), dtype=np.uint8)
    reference_full[2:8, 4:12] = 90
    streak_row = DemoBranchResult(
        DemoBranch.BRIGHT_STREAK,
        "front_left",
        BranchStatus.NG,
        0.0,
        0.1,
        1.0,
        "光痕断续",
        np.full((6, 8, 3), 120, dtype=np.uint8),
        details={"roi_xyxy": (4, 2, 12, 8)},
    )
    full_match = _match(base, "front_left", "full")
    full_match = TrustedOkMatch(
        view_id=full_match.view_id,
        comparison_mode=full_match.comparison_mode,
        physical_part_id=full_match.physical_part_id,
        sample_id=full_match.sample_id,
        similarity=full_match.similarity,
        shift_x=full_match.shift_x,
        shift_y=full_match.shift_y,
        current_full_image=full_match.current_full_image,
        reference_full_image=reference_full,
            current_roi=full_match.current_roi,
            reference_roi=full_match.reference_roi,
            aligned_reference_roi=full_match.aligned_reference_roi,
            difference_overlay=full_match.difference_overlay,
        )
    inspection = EightViewInspection(
        base.capture_id,
        base.images,
        (template_row, *base.results[1:], streak_row),
        base.final_status,
        base.elapsed_ms,
        roi_images=base.roi_images,
        trusted_ok_by_comparison={
            ("front", "roi"): template_match,
            ("front_right", "roi"): yolo_match,
            ("back_left", "roi"): efficientad_match,
            ("front_left", "full"): full_match,
        },
    )

    template = evidence_detail_images(EightViewUiState(
        phase=DemoUiPhase.RESULT, inspection=inspection, selected_view="front", selected_branch=DemoBranch.TEMPLATE,
    ))
    yolo = evidence_detail_images(EightViewUiState(
        phase=DemoUiPhase.RESULT, inspection=inspection, selected_view="front_right", selected_branch=DemoBranch.YOLO,
    ))
    efficientad = evidence_detail_images(EightViewUiState(
        phase=DemoUiPhase.RESULT,
        inspection=inspection,
        selected_view="back_left",
        selected_branch=DemoBranch.EFFICIENTAD,
    ))
    streak = evidence_detail_images(EightViewUiState(
        phase=DemoUiPhase.RESULT,
        inspection=inspection,
        selected_view="front_left",
        selected_branch=DemoBranch.BRIGHT_STREAK,
    ))

    assert int(template.left_image.mean()) == 70
    assert int(template.right_image.mean()) == 10
    assert int(yolo.left_image.mean()) == 60
    assert int(yolo.right_image.mean()) == 20
    assert int(efficientad.left_image.mean()) == 60
    assert efficientad.right_image is None
    assert streak.left_image.shape == (8, 6, 3)
    assert np.all(streak.left_image == 90)
    assert streak.right_image.shape == (8, 6, 3)
    assert int(streak.right_image.mean()) == 120


def test_bright_streak_reference_uses_same_rotated_perspective() -> None:
    images = {view: np.zeros((100, 120, 3), dtype=np.uint8) for view in VIEW_ORDER}
    reference_full = np.indices((100, 120)).sum(axis=0).astype(np.uint8)
    reference_full = np.repeat(reference_full[:, :, None], 3, axis=2)
    points = ((40, 10), (70, 15), (65, 90), (35, 85))
    asset = RotatedBrightStreakRoi(
        points_xy=points,
        source_width=120,
        source_height=100,
        output_width=81,
        output_height=613,
    )
    row = DemoBranchResult(
        DemoBranch.BRIGHT_STREAK,
        "front_left",
        BranchStatus.NG,
        0.0,
        0.1,
        1.0,
        "光痕断续",
        np.zeros((613, 81, 3), dtype=np.uint8),
        details={"roi_points_xy": points},
    )
    inspection = EightViewInspection(
        "rotated",
        images,
        (row,),
        DemoFinalStatus.NG,
        1.0,
        trusted_ok_by_comparison={
            ("front_left", "full"): replace(
                _match(
                    EightViewInspection("base", images, (row,), DemoFinalStatus.NG, 1.0),
                    "front_left",
                    "full",
                ),
                reference_full_image=reference_full,
            )
        },
    )

    detail = evidence_detail_images(
        EightViewUiState(
            phase=DemoUiPhase.RESULT,
            inspection=inspection,
            selected_view="front_left",
            selected_branch=DemoBranch.BRIGHT_STREAK,
        )
    )

    expected = np.rot90(rectify_bright_streak_roi(reference_full, asset), k=3)
    assert detail.left_image is not None
    assert detail.left_image.shape == (81, 613, 3)
    assert np.array_equal(detail.left_image, expected)


def test_detail_page_is_opened_only_for_valid_evidence_and_reports_pass_or_missing_reference() -> None:
    base = _inspection()
    pass_state = EightViewUiState(
        phase=DemoUiPhase.RESULT,
        inspection=base,
        selected_view="front",
        selected_branch=DemoBranch.TEMPLATE,
    )
    missing_reference = EightViewUiState(
        phase=DemoUiPhase.RESULT,
        inspection=base,
        selected_view="front_right",
        selected_branch=DemoBranch.YOLO,
    )

    opened = apply_dashboard_click(pass_state, DashboardHit(evidence_index=3))
    missing = evidence_detail_images(missing_reference)

    assert opened.page is DemoUiPage.DETAIL
    assert opened.inspection is base
    assert "无需NG参考" in evidence_detail_images(pass_state).left_label
    assert missing.left_image is None
    assert "无可信OK参考" in missing.left_label
    assert render_eight_view_screen(opened).shape == (900, 1600, 3)


def test_pass_detail_footer_hides_reference_identity_shared_by_another_ng_branch() -> None:
    base = _inspection()
    yolo_ng = DemoBranchResult(
        DemoBranch.YOLO,
        "front",
        BranchStatus.NG,
        0.8,
        0.2,
        1.0,
        "同一视角的YOLO不通过",
        np.full((10, 10, 3), 20, dtype=np.uint8),
    )
    inspection = EightViewInspection(
        base.capture_id,
        base.images,
        base.results + (yolo_ng,),
        base.final_status,
        base.elapsed_ms,
        roi_images=base.roi_images,
        trusted_ok_by_comparison={("front", "roi"): _match(base, "front")},
    )
    pass_state = EightViewUiState(
        phase=DemoUiPhase.RESULT,
        inspection=inspection,
        selected_view="front",
        selected_branch=DemoBranch.TEMPLATE,
    )

    footer = detail_reference_footer(pass_state)

    assert footer == "当前项目通过，无需NG参考"
    assert "normal-train-001" not in footer


def test_long_detail_reason_wraps_without_losing_chinese_semantics() -> None:
    long_reason = "模板差异超过阈值，需核对定位、边缘、纹理和夹具状态。" * 5
    base = _inspection()
    long_template = DemoBranchResult(
        DemoBranch.TEMPLATE,
        "front",
        BranchStatus.NG,
        0.3,
        0.2,
        1.0,
        long_reason,
        np.full((10, 10, 3), 10, dtype=np.uint8),
    )
    inspection = EightViewInspection(
        base.capture_id,
        base.images,
        (long_template, *base.results[1:]),
        base.final_status,
        base.elapsed_ms,
        roi_images=base.roi_images,
    )
    state = EightViewUiState(
        phase=DemoUiPhase.RESULT,
        inspection=inspection,
        selected_view="front",
        selected_branch=DemoBranch.TEMPLATE,
    )

    lines = detail_reason_lines(state)

    assert 2 <= len(lines) <= 3
    assert "".join(lines) == long_reason
    assert render_eight_view_screen(replace(state, page=DemoUiPage.DETAIL)).shape == (900, 1600, 3)


def test_detail_controls_use_a_safe_top_right_baseline() -> None:
    text, x, y, anchor = detail_controls_layout()

    assert text == "Esc：返回主页面　Q：退出　R：重置"
    assert anchor == "ra"
    assert 0 < x < 1600
    assert 0 < y <= 900 - 13
