"""All-Chinese Swiss-grid UI for the independent BMW eight-view laboratory Demo."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER
from bmw_inspection.lab.eight_view_demo import (
    BranchStatus,
    DemoBranch,
    DemoBranchResult,
    DemoFinalStatus,
    EightViewInspection,
)
from bmw_inspection.lab.ui import _fit_image, _rgb_to_bgr


class DemoUiPhase(str, Enum):
    IDLE = "IDLE"
    WAITING_FLIP = "WAITING_FLIP"
    PROCESSING = "PROCESSING"
    RESULT = "RESULT"
    ERROR = "ERROR"


class DemoUiPage(str, Enum):
    """The in-window page currently shown to the operator."""

    DASHBOARD = "DASHBOARD"
    DETAIL = "DETAIL"


@dataclass(frozen=True, slots=True)
class DashboardHit:
    """One logical 1600x900 dashboard target, or no target when all fields are empty."""

    branch: DemoBranch | None = None
    view_id: str | None = None
    evidence_index: int | None = None


@dataclass(frozen=True, slots=True)
class EvidenceDetailImages:
    """The truthful fixed-reference/current pair for one selected branch."""

    left_label: str
    left_image: np.ndarray | None
    right_label: str
    right_image: np.ndarray | None


@dataclass(frozen=True, slots=True)
class EightViewUiState:
    phase: DemoUiPhase = DemoUiPhase.IDLE
    message: str = "按空格键拍摄正面"
    images: Mapping[str, np.ndarray] | None = None
    inspection: EightViewInspection | None = None
    selected_view: str = "front_left"
    selected_branch: DemoBranch = DemoBranch.BRIGHT_STREAK
    experiment_mode: bool = False
    source_images: Mapping[str, Any] | None = None
    trusted_ok_mode: bool = False
    page: DemoUiPage = DemoUiPage.DASHBOARD


_WHITE = (255, 255, 255)
_SURFACE = (247, 247, 248)
_INK = (20, 24, 31)
_MUTED = (76, 84, 96)
_GRID = (211, 214, 220)
_BLUE = (0, 47, 167)
_GREEN = (23, 132, 75)
_RED = (228, 0, 43)
_AMBER = (197, 117, 0)
_MAGENTA = (191, 0, 112)

_VIEW_LABELS = {
    "front": "正面中间",
    "front_left": "正面左侧",
    "front_right": "正面右侧",
    "front_secondary": "正面辅助",
    "back": "反面中间",
    "back_left": "反面左侧",
    "back_right": "反面右侧",
    "back_secondary": "反面辅助",
}
_BRANCH_LABELS = {
    DemoBranch.TEMPLATE: "模板匹配",
    DemoBranch.BRIGHT_STREAK: "光痕检测",
    DemoBranch.YOLO: "缺陷检测",
    DemoBranch.EFFICIENTAD: "异常检测",
}

_VIEW_CARD_X = 24
_VIEW_CARD_Y = 92
_VIEW_CARD_WIDTH = 230
_VIEW_CARD_HEIGHT = 182
_VIEW_CARD_STEP_X = 255
_VIEW_CARD_STEP_Y = 210
_EVIDENCE_Y = 610
_EVIDENCE_HEIGHT = 230
_BRANCH_CARD_X = 1084
_BRANCH_CARD_Y = 258
_BRANCH_CARD_WIDTH = 232
_BRANCH_CARD_HEIGHT = 108
_BRANCH_CARD_STEP_X = 250
_BRANCH_CARD_STEP_Y = 132


@lru_cache(maxsize=2)
def _demo_font_path(weight: str) -> Path:
    """Resolve the installed CJK sans font for one semantic weight."""
    if weight not in {"medium", "bold"}:
        raise ValueError("字体字重必须是medium或bold")
    generic_override = os.environ.get("BMW_DEMO_FONT")
    specific_override = os.environ.get(f"BMW_DEMO_FONT_{weight.upper()}")
    system_name = "NotoSansCJK-Medium.ttc" if weight == "medium" else "NotoSansCJK-Bold.ttc"
    fallbacks = (
        Path(specific_override).expanduser() if specific_override else None,
        Path(generic_override).expanduser() if generic_override else None,
        Path("/usr/share/fonts/opentype/noto") / system_name,
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for candidate in fallbacks:
        if candidate is not None and candidate.is_file():
            return candidate
    raise RuntimeError("未找到可用字体，请安装Noto Sans CJK或设置BMW_DEMO_FONT")


@lru_cache(maxsize=64)
def _demo_font(size: int, weight: str = "medium") -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(_demo_font_path(weight)), size=size)


def _status(status: BranchStatus) -> tuple[str, tuple[int, int, int]]:
    return {
        BranchStatus.PASS: ("通过", _GREEN),
        BranchStatus.NG: ("不通过", _RED),
        BranchStatus.ERROR: ("异常", _MAGENTA),
        BranchStatus.SKIPPED: ("未执行", _MUTED),
    }[status]


def _aggregate(rows: tuple[DemoBranchResult, ...]) -> BranchStatus:
    if not rows:
        return BranchStatus.SKIPPED
    if any(row.status is BranchStatus.ERROR for row in rows):
        return BranchStatus.ERROR
    if any(row.status is BranchStatus.NG for row in rows):
        return BranchStatus.NG
    return BranchStatus.PASS


def preferred_selection(inspection: EightViewInspection) -> tuple[str, DemoBranch]:
    """Prefer the first actionable evidence, otherwise show the light-streak proof."""
    for status in (BranchStatus.ERROR, BranchStatus.NG):
        for row in inspection.results:
            if row.status is status:
                return row.view_id, row.branch
    return "front_left", DemoBranch.BRIGHT_STREAK


def step_actionable_selection(
    inspection: EightViewInspection,
    current_view: str,
    current_branch: DemoBranch,
    step: int,
) -> tuple[str, DemoBranch]:
    """Move through NG/ERROR evidence with deterministic circular navigation."""
    if step == 0:
        raise ValueError("step不能为0")
    actionable = inspection.actionable_results()
    if not actionable:
        return current_view, current_branch
    selected = next(
        (
            index
            for index, row in enumerate(actionable)
            if row.view_id == current_view and row.branch is current_branch
        ),
        -1 if step > 0 else 0,
    )
    row = actionable[(selected + (1 if step > 0 else -1)) % len(actionable)]
    return row.view_id, row.branch


def dashboard_hit_test(x: int, y: int) -> DashboardHit | None:
    """Map one logical canvas coordinate to a stable clickable dashboard target."""
    for index, view in enumerate(VIEW_ORDER):
        card_x = _VIEW_CARD_X + (index % 4) * _VIEW_CARD_STEP_X
        card_y = _VIEW_CARD_Y + (index // 4) * _VIEW_CARD_STEP_Y
        if card_x <= x <= card_x + _VIEW_CARD_WIDTH and card_y <= y <= card_y + _VIEW_CARD_HEIGHT:
            return DashboardHit(view_id=view)
    for index, branch in enumerate(DemoBranch):
        card_x = _BRANCH_CARD_X + (index % 2) * _BRANCH_CARD_STEP_X
        card_y = _BRANCH_CARD_Y + (index // 2) * _BRANCH_CARD_STEP_Y
        if card_x <= x <= card_x + _BRANCH_CARD_WIDTH and card_y <= y <= card_y + _BRANCH_CARD_HEIGHT:
            return DashboardHit(branch=branch)
    for index in range(4):
        panel_x = _VIEW_CARD_X + index * _VIEW_CARD_STEP_X
        if panel_x <= x <= panel_x + _VIEW_CARD_WIDTH and _EVIDENCE_Y <= y <= _EVIDENCE_Y + _EVIDENCE_HEIGHT:
            return DashboardHit(evidence_index=index)
    return None


def _selected_row(state: EightViewUiState) -> DemoBranchResult | None:
    if state.inspection is None:
        return None
    return next(
        (
            row
            for row in state.inspection.results
            if row.view_id == state.selected_view and row.branch is state.selected_branch
        ),
        None,
    )


def select_branch(state: EightViewUiState, branch: DemoBranch) -> EightViewUiState:
    """Select a branch, prioritising its first ERROR, then NG, then PASS evidence."""
    if state.inspection is None:
        return replace(state, selected_branch=branch)
    branch_rows = tuple(row for row in state.inspection.results if row.branch is branch)
    target = next(
        (
            row
            for status in (BranchStatus.ERROR, BranchStatus.NG, BranchStatus.PASS)
            for row in branch_rows
            if row.status is status
        ),
        None,
    )
    if target is None:
        return replace(state, selected_branch=branch)
    return replace(state, selected_view=target.view_id, selected_branch=branch)


def apply_dashboard_click(state: EightViewUiState, hit: DashboardHit | None) -> EightViewUiState:
    """Apply one dashboard target without changing the immutable inspection result."""
    if state.page is not DemoUiPage.DASHBOARD or hit is None:
        return state
    if hit.branch is not None:
        return select_branch(state, hit.branch)
    if hit.view_id is not None:
        return replace(state, selected_view=hit.view_id)
    if hit.evidence_index is None:
        return state
    panels = evidence_comparison_images(state)
    if not 0 <= hit.evidence_index < len(panels) or panels[hit.evidence_index][1] is None:
        return state
    if _selected_row(state) is None:
        return state
    return replace(state, page=DemoUiPage.DETAIL)


def _trusted_ok_available(state: EightViewUiState) -> bool:
    """Report whether this inspection came from an enabled reference diagnostic."""
    if state.inspection is None:
        return False
    return bool(state.inspection.trusted_ok_by_comparison) or bool(
        state.inspection.diagnostic_metadata.get("trusted_ok_match_errors")
    )


def toggle_trusted_ok_mode(state: EightViewUiState) -> EightViewUiState:
    """Toggle diagnostic reference comparison only when it is available."""
    if not _trusted_ok_available(state):
        return state
    return replace(state, trusted_ok_mode=not state.trusted_ok_mode)


def _trusted_reference_details(state: EightViewUiState) -> tuple[tuple[str, str], ...]:
    """Return compact, always-visible reference identity rows for the right column."""
    if not state.trusted_ok_mode or state.inspection is None:
        return ()
    selected_row = next(
        (
            row for row in state.inspection.results
            if row.view_id == state.selected_view and row.branch is state.selected_branch
        ),
        None,
    )
    if selected_row is None or selected_row.status not in {BranchStatus.NG, BranchStatus.ERROR}:
        return ()
    mode = "full" if state.selected_branch is DemoBranch.BRIGHT_STREAK else "roi"
    match = state.inspection.trusted_ok_by_comparison.get((state.selected_view, mode))
    if match is None:
        return (("可信参考", "无可信OK参考"),)
    return (
        ("参考零件", match.physical_part_id),
        ("参考样本", match.sample_id),
        ("相似度", f"{match.similarity:.6f}"),
        ("对齐平移", f"({match.shift_x:+d}, {match.shift_y:+d})"),
    )


def _source_image(state: EightViewUiState, kind: str) -> np.ndarray | None:
    sources = state.source_images
    if sources is None:
        return None
    by_view = sources.get(state.selected_view)
    if isinstance(by_view, Mapping):
        image = by_view.get(kind)
    else:
        by_kind = sources.get(kind)
        image = by_kind.get(state.selected_view) if isinstance(by_kind, Mapping) else None
        if image is None and by_view is not None:
            attribute = {"short": "short_image", "long": "long_image", "hdr": "fused_image"}[kind]
            image = getattr(by_view, attribute, None)
    return image if isinstance(image, np.ndarray) and image.size else None


def _source_kind(state: EightViewUiState) -> str | None:
    sources = state.source_images
    if sources is None:
        return None
    by_view = sources.get(state.selected_view)
    if isinstance(by_view, Mapping):
        value = by_view.get("source_kind")
    else:
        value = getattr(by_view, "source_kind", None)
    return value if isinstance(value, str) else None


def evidence_comparison_images(
    state: EightViewUiState,
) -> tuple[tuple[str, np.ndarray | None], ...]:
    """Expose the exact four image panels used by the result dashboard."""
    inspection = state.inspection
    selected_row = None if inspection is None else next(
        (
            row
            for row in inspection.results
            if row.view_id == state.selected_view and row.branch is state.selected_branch
        ),
        None,
    )
    hdr = _source_image(state, "hdr")
    if hdr is None and inspection is not None:
        hdr = inspection.images[state.selected_view]
    evidence_type = {
        DemoBranch.YOLO: "真实检测框",
        DemoBranch.BRIGHT_STREAK: "规则 ROI 证据",
        DemoBranch.TEMPLATE: "诊断热区",
        DemoBranch.EFFICIENTAD: "诊断热区",
    }[state.selected_branch]
    if selected_row is not None:
        evidence_type = str(selected_row.details.get("evidence_type", evidence_type))
    evidence = None if selected_row is None else selected_row.overlay
    if evidence is not None and state.selected_branch is DemoBranch.BRIGHT_STREAK:
        evidence = cv2.rotate(evidence, cv2.ROTATE_90_CLOCKWISE)
    if state.trusted_ok_mode and inspection is not None:
        if selected_row is None:
            return (
                ("当前项目无检测结果，无需NG对比", None),
                ("无需可信OK参考", None),
                ("无需对齐差异", None),
                ("无需模型证据", None),
            )
        if selected_row.status not in {BranchStatus.NG, BranchStatus.ERROR}:
            return (
                ("当前项目通过，无需NG对比", None),
                ("无需可信OK参考", None),
                ("无需对齐差异", None),
                ("无需模型证据", None),
            )
        mode = "full" if state.selected_branch is DemoBranch.BRIGHT_STREAK else "roi"
        match = inspection.trusted_ok_by_comparison.get((state.selected_view, mode))
        if match is not None:
            current = match.current_roi
        elif mode == "full":
            current = inspection.images[state.selected_view]
        else:
            current = inspection.roi_images.get(state.selected_view)
        return (
            ("现场NG", current),
            ("可信OK", None if match is None else match.aligned_reference_roi),
            ("对齐差异", None if match is None else match.difference_overlay),
            (evidence_type, evidence),
        )
    if _source_kind(state) == "fused_only":
        return (
            ("无短曝光原图", None),
            ("无长曝光原图", None),
            ("历史融合图", hdr),
            (evidence_type, evidence),
        )
    return (
        ("短曝光", _source_image(state, "short")),
        ("长曝光", _source_image(state, "long")),
        ("融合 HDR", hdr),
        (evidence_type, evidence),
    )


def _bright_streak_reference_roi(match: Any, row: DemoBranchResult) -> np.ndarray | None:
    roi = row.details.get("roi_xyxy")
    if (
        not isinstance(roi, tuple)
        or len(roi) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) for value in roi)
    ):
        return None
    x1, y1, x2, y2 = roi
    reference = match.reference_full_image
    if not (0 <= x1 < x2 <= reference.shape[1] and 0 <= y1 < y2 <= reference.shape[0]):
        return None
    return reference[y1:y2, x1:x2]


def evidence_detail_images(state: EightViewUiState) -> EvidenceDetailImages:
    """Build the approved-reference/current pair in the selected algorithm's native domain."""
    row = _selected_row(state)
    if row is None:
        return EvidenceDetailImages("当前项目无检测结果", None, "当前检测证据", None)
    current = row.overlay
    if row.status is BranchStatus.PASS:
        if state.selected_branch is DemoBranch.BRIGHT_STREAK and current is not None:
            current = cv2.rotate(current, cv2.ROTATE_90_CLOCKWISE)
        return EvidenceDetailImages("当前项目通过，无需NG参考", None, "当前检测证据", current)
    mode = "full" if state.selected_branch is DemoBranch.BRIGHT_STREAK else "roi"
    match = None if state.inspection is None else state.inspection.trusted_ok_by_comparison.get((state.selected_view, mode))
    if match is None:
        if state.selected_branch is DemoBranch.BRIGHT_STREAK and current is not None:
            current = cv2.rotate(current, cv2.ROTATE_90_CLOCKWISE)
        return EvidenceDetailImages("无可信OK参考", None, "当前检测证据", current)
    if state.selected_branch is DemoBranch.TEMPLATE:
        return EvidenceDetailImages("可信OK对齐参考", match.aligned_reference_roi, "当前差异证据", match.difference_overlay)
    if state.selected_branch in {DemoBranch.YOLO, DemoBranch.EFFICIENTAD}:
        return EvidenceDetailImages("可信OK参考ROI", match.reference_roi, "当前ROI覆盖图", current)
    reference_roi = _bright_streak_reference_roi(match, row)
    if reference_roi is None:
        return EvidenceDetailImages("可信OK参考ROI不可用", None, "当前光痕ROI证据", None if current is None else cv2.rotate(current, cv2.ROTATE_90_CLOCKWISE))
    return EvidenceDetailImages(
        "可信OK光痕ROI",
        cv2.rotate(reference_roi, cv2.ROTATE_90_CLOCKWISE),
        "当前光痕ROI证据",
        None if current is None else cv2.rotate(current, cv2.ROTATE_90_CLOCKWISE),
    )


def _wrapped_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    font: ImageFont.FreeTypeFont,
    width: int,
) -> list[str]:
    """Wrap Chinese/Latin result text without truncating its reason."""
    lines: list[str] = []
    current = ""
    for character in text:
        candidate = current + character
        if current and draw.textlength(candidate, font=font) > width:
            lines.append(current)
            current = character
        else:
            current = candidate
    if current or not lines:
        lines.append(current)
    return lines


def render_eight_view_dashboard(state: EightViewUiState) -> np.ndarray:
    """Render a deterministic 1600x900 presentation/experiment dashboard."""
    canvas = np.full((900, 1600, 3), _rgb_to_bgr(_SURFACE), dtype=np.uint8)
    cv2.line(canvas, (0, 68), (1599, 68), _rgb_to_bgr(_GRID), 1)
    cv2.line(canvas, (1060, 68), (1060, 899), _rgb_to_bgr(_GRID), 1)
    cv2.line(canvas, (0, 530), (1060, 530), _rgb_to_bgr(_GRID), 1)

    images = state.inspection.images if state.inspection is not None else state.images
    results = () if state.inspection is None else state.inspection.results
    for index, view in enumerate(VIEW_ORDER):
        column, row_index = index % 4, index // 4
        x = _VIEW_CARD_X + column * _VIEW_CARD_STEP_X
        y = _VIEW_CARD_Y + row_index * _VIEW_CARD_STEP_Y
        cv2.rectangle(canvas, (x, y + 32), (x + 230, y + 182), _rgb_to_bgr(_WHITE), -1)
        view_rows = tuple(item for item in results if item.view_id == view)
        selected = view == state.selected_view
        border = _rgb_to_bgr(_BLUE if selected else (_GRID if not view_rows else _status(_aggregate(view_rows))[1]))
        if images is not None and view in images:
            canvas[y + 32 : y + 182, x : x + 230] = _fit_image(images[view], 230, 150)
        cv2.rectangle(canvas, (x, y + 32), (x + 229, y + 181), border, 4 if selected else 2)

    selected_row = next(
        (row for row in results if row.view_id == state.selected_view and row.branch is state.selected_branch),
        None,
    )
    comparison = evidence_comparison_images(state)
    panel_rectangles: list[tuple[str, np.ndarray | None, int]] = []
    for index, (label, panel_image) in enumerate(comparison):
        x = _VIEW_CARD_X + index * _VIEW_CARD_STEP_X
        panel_rectangles.append((label, panel_image, x))
        cv2.rectangle(canvas, (x, 610), (x + 230, 840), _rgb_to_bgr(_WHITE), -1)
        if panel_image is not None:
            canvas[610:840, x : x + 230] = _fit_image(panel_image, 230, 230)
        border_color = _status(selected_row.status)[1] if index == 3 and selected_row is not None else _GRID
        cv2.rectangle(canvas, (x, 610), (x + 229, 839), _rgb_to_bgr(border_color), 2 if index == 3 else 1)

    if state.phase is DemoUiPhase.RESULT and state.inspection is not None:
        final = state.inspection.final_status
        status_title, status_detail, color = {
            DemoFinalStatus.OK: ("检测通过", "25 项检测全部通过", _GREEN),
            DemoFinalStatus.NG: ("检测不通过", "至少一个检测项目判定为不通过", _RED),
            DemoFinalStatus.ERROR: ("检测异常", "至少一个模型未能完成推理", _MAGENTA),
        }[final]
    else:
        status_title, status_detail, color = {
            DemoUiPhase.IDLE: ("等待检测", "按空格键拍摄正面", _BLUE),
            DemoUiPhase.WAITING_FLIP: ("请翻转零件", "固定反面后按空格键继续", _AMBER),
            DemoUiPhase.PROCESSING: ("正在检测", "正在运行四类模型", _BLUE),
            DemoUiPhase.ERROR: ("运行异常", state.message, _MAGENTA),
            DemoUiPhase.RESULT: ("检测完成", state.message, _BLUE),
        }[state.phase]
    cv2.rectangle(canvas, (1084, 96), (1574, 224), _rgb_to_bgr(color), 4)

    branch_rectangles: list[tuple[DemoBranch, int, int]] = []
    for index, branch in enumerate(DemoBranch):
        x = _BRANCH_CARD_X + (index % 2) * _BRANCH_CARD_STEP_X
        y = _BRANCH_CARD_Y + (index // 2) * _BRANCH_CARD_STEP_Y
        branch_rectangles.append((branch, x, y))
        branch_rows = tuple(row for row in results if row.branch is branch)
        branch_status = _aggregate(branch_rows)
        branch_color = _status(branch_status)[1]
        cv2.rectangle(canvas, (x, y), (x + 232, y + 108), _rgb_to_bgr(_WHITE), -1)
        cv2.rectangle(
            canvas,
            (x, y),
            (x + _BRANCH_CARD_WIDTH, y + _BRANCH_CARD_HEIGHT),
            _rgb_to_bgr(_BLUE if branch is state.selected_branch else branch_color),
            4 if branch is state.selected_branch else 2,
        )

    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    draw.text((24, 14), "BMW 零件八视图检测", font=_demo_font(31, "bold"), fill=_INK)
    draw.text((408, 22), "实验室快速验证", font=_demo_font(19), fill=_BLUE)
    draw.text(
        (1574, 22),
        "实验模式" if state.experiment_mode else "展示模式",
        font=_demo_font(18),
        fill=_MUTED,
        anchor="ra",
    )
    for index, view in enumerate(VIEW_ORDER):
        column, row_index = index % 4, index // 4
        x, y = 24 + column * 255, 92 + row_index * 210
        draw.text((x, y - 1), f"{index + 1:02d}", font=_demo_font(24, "bold"), fill=_BLUE)
        draw.text((x + 44, y + 3), _VIEW_LABELS[view], font=_demo_font(17), fill=_INK)
    draw.text((1104, 112), status_title, font=_demo_font(33, "bold"), fill=color)
    draw.text((1104, 170), status_detail[:24], font=_demo_font(17), fill=_MUTED)
    for branch, x, y in branch_rectangles:
        branch_status = _aggregate(tuple(row for row in results if row.branch is branch))
        label, branch_color = _status(branch_status)
        draw.text((x + 16, y + 12), _BRANCH_LABELS[branch], font=_demo_font(21, "bold"), fill=_INK)
        draw.text((x + 16, y + 56), label, font=_demo_font(25, "bold"), fill=branch_color)
    draw.text(
        (24, 548),
        f"证据对比：{_VIEW_LABELS[state.selected_view]} · {_BRANCH_LABELS[state.selected_branch]}",
        font=_demo_font(20, "bold"),
        fill=_INK,
    )
    for label, panel_image, x in panel_rectangles:
        draw.text((x, 580), label, font=_demo_font(17, "bold"), fill=_INK)
        if panel_image is None:
            unavailable = (
                "无可信OK参考"
                if state.trusted_ok_mode and label in {"可信OK", "对齐差异"}
                else "暂无图像"
            )
            draw.text((x + 115, 725), unavailable, font=_demo_font(16), fill=_MUTED, anchor="mm")
    controls = "空格：拍摄/继续　R：重置　Q：退出"
    if state.experiment_mode:
        controls += "\n1–8：视角　T/L/Y/E：证据\nN：下一条　P：上一条"
        if _trusted_ok_available(state):
            controls += "　O：可信OK对比"
    for index, line in enumerate(controls.splitlines()):
        draw.text((1084, 520 + index * 26), line, font=_demo_font(15), fill=_MUTED)
    if state.inspection is not None:
        actionable = state.inspection.actionable_results()
        current_index = next(
            (
                index
                for index, row in enumerate(actionable)
                if row.view_id == state.selected_view and row.branch is state.selected_branch
            ),
            None,
        )
        queue_text = "当前为手动选择" if current_index is None else f"NG/异常 {current_index + 1}/{len(actionable)}"
        draw.text((1084, 602), queue_text, font=_demo_font(17, "bold"), fill=_RED if actionable else _GREEN)
    detail_y = 630
    detail_rows = [
        ("视角", _VIEW_LABELS[state.selected_view]),
        ("项目", _BRANCH_LABELS[state.selected_branch]),
        ("结果", "未执行" if selected_row is None else _status(selected_row.status)[0]),
    ]
    if selected_row is not None:
        base_threshold = selected_row.details.get("base_threshold")
        deployment_threshold = selected_row.details.get("deployment_threshold", selected_row.threshold)
        margin = selected_row.details.get("threshold_margin")
        exceedance = selected_row.details.get("threshold_exceedance")
        numeric_rows = [
            ("分数", "不可用" if selected_row.score is None else f"{selected_row.score:.6g}"),
            ("部署阈值", "不可用" if deployment_threshold is None else f"{float(deployment_threshold):.6g}"),
            ("超限量", "不可用" if exceedance is None else f"{float(exceedance):+.6g}"),
        ]
        if not state.trusted_ok_mode:
            numeric_rows[1:1] = [
                ("基础阈值", "不可用" if base_threshold is None else f"{float(base_threshold):.6g}"),
            ]
            numeric_rows.insert(
                3,
                ("阈值余量", "不可用" if margin is None else f"{float(margin):.6g}"),
            )
        detail_rows.extend(numeric_rows)
    detail_rows.extend(_trusted_reference_details(state))
    for index, (label, value) in enumerate(detail_rows):
        y = detail_y + index * 20
        draw.text((1084, y), label, font=_demo_font(13), fill=_MUTED)
        draw.text((1192, y), value, font=_demo_font(13), fill=_INK)
    if selected_row is not None:
        reason_y = detail_y + len(detail_rows) * 20 + 5
        draw.text((1084, reason_y), "完整原因", font=_demo_font(13, "bold"), fill=_MUTED)
        for index, line in enumerate(_wrapped_lines(draw, selected_row.reason, font=_demo_font(12), width=465)):
            draw.text((1084, reason_y + 21 + index * 17), line, font=_demo_font(12), fill=_INK)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _detail_status_text(state: EightViewUiState) -> tuple[str, str, str, tuple[int, int, int]]:
    row = _selected_row(state)
    if row is None:
        return (_BRANCH_LABELS[state.selected_branch], _VIEW_LABELS[state.selected_view], "未执行", _MUTED)
    label, color = _status(row.status)
    return (_BRANCH_LABELS[row.branch], _VIEW_LABELS[row.view_id], label, color)


def render_eight_view_detail(state: EightViewUiState) -> np.ndarray:
    """Render one 1600x900 fixed-reference detail page from immutable source arrays."""
    canvas = np.full((900, 1600, 3), _rgb_to_bgr(_SURFACE), dtype=np.uint8)
    detail = evidence_detail_images(state)
    branch_label, view_label, status_label, status_color = _detail_status_text(state)
    panels = ((24, detail.left_label, detail.left_image), (820, detail.right_label, detail.right_image))
    for x, _label, panel_image in panels:
        cv2.rectangle(canvas, (x, 158), (x + 756, 806), _rgb_to_bgr(_WHITE), -1)
        if panel_image is not None:
            canvas[158:806, x : x + 756] = _fit_image(panel_image, 756, 648)
        cv2.rectangle(canvas, (x, 158), (x + 755, 805), _rgb_to_bgr(_BLUE), 3)
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), "BMW 检测证据详情", font=_demo_font(31, "bold"), fill=_INK)
    draw.text((24, 72), f"{branch_label} · {view_label}", font=_demo_font(23, "bold"), fill=_INK)
    draw.text((560, 76), status_label, font=_demo_font(22, "bold"), fill=status_color)
    for x, label, panel_image in panels:
        draw.text((x, 118), label, font=_demo_font(20, "bold"), fill=_INK)
        if panel_image is None:
            draw.text((x + 378, 482), label, font=_demo_font(21), fill=_MUTED, anchor="mm")
    row = _selected_row(state)
    if row is not None:
        score = "不可用" if row.score is None else f"{row.score:.6g}"
        threshold = "不可用" if row.threshold is None else f"{row.threshold:.6g}"
        draw.text((24, 830), f"分数：{score}　阈值：{threshold}　原因：{row.reason}", font=_demo_font(16), fill=_INK)
    match = None
    if state.inspection is not None:
        mode = "full" if state.selected_branch is DemoBranch.BRIGHT_STREAK else "roi"
        match = state.inspection.trusted_ok_by_comparison.get((state.selected_view, mode))
    reference_text = "参考样本：无可信OK参考" if match is None else f"参考零件：{match.physical_part_id}　参考样本：{match.sample_id}"
    draw.text((24, 862), f"{reference_text}　Esc：返回主页面　Q：退出　R：重置", font=_demo_font(15), fill=_MUTED)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def render_eight_view_screen(state: EightViewUiState) -> np.ndarray:
    """Dispatch the fixed-size window to dashboard or trusted-reference detail page."""
    if state.page is DemoUiPage.DETAIL:
        return render_eight_view_detail(state)
    return render_eight_view_dashboard(state)


__all__ = [
    "DashboardHit",
    "DemoUiPage",
    "DemoUiPhase",
    "EvidenceDetailImages",
    "EightViewUiState",
    "apply_dashboard_click",
    "dashboard_hit_test",
    "evidence_detail_images",
    "evidence_comparison_images",
    "preferred_selection",
    "render_eight_view_detail",
    "render_eight_view_dashboard",
    "render_eight_view_screen",
    "select_branch",
    "step_actionable_selection",
    "toggle_trusted_ok_mode",
]
