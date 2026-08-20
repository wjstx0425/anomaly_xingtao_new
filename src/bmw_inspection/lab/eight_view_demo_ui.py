"""All-Chinese Swiss-grid UI for the independent BMW eight-view laboratory Demo."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path

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


@dataclass(frozen=True, slots=True)
class EightViewUiState:
    phase: DemoUiPhase = DemoUiPhase.IDLE
    message: str = "按空格键拍摄正面"
    images: Mapping[str, np.ndarray] | None = None
    inspection: EightViewInspection | None = None
    selected_view: str = "front_left"
    selected_branch: DemoBranch = DemoBranch.BRIGHT_STREAK
    experiment_mode: bool = False


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
        x, y = 24 + column * 255, 92 + row_index * 210
        cv2.rectangle(canvas, (x, y + 32), (x + 230, y + 182), _rgb_to_bgr(_WHITE), -1)
        view_rows = tuple(item for item in results if item.view_id == view)
        border = _rgb_to_bgr(_GRID if not view_rows else _status(_aggregate(view_rows))[1])
        if images is not None and view in images:
            canvas[y + 32 : y + 182, x : x + 230] = _fit_image(images[view], 230, 150)
        cv2.rectangle(canvas, (x, y + 32), (x + 229, y + 181), border, 2)

    selected_row = next(
        (row for row in results if row.view_id == state.selected_view and row.branch is state.selected_branch),
        None,
    )
    evidence = None if selected_row is None else selected_row.overlay
    if evidence is not None and state.selected_branch is DemoBranch.BRIGHT_STREAK:
        evidence = cv2.rotate(evidence, cv2.ROTATE_90_CLOCKWISE)
    cv2.rectangle(canvas, (24, 584), (648, 840), _rgb_to_bgr(_WHITE), -1)
    if evidence is not None:
        canvas[584:840, 24:648] = _fit_image(evidence, 624, 256)
        evidence_border = _status(selected_row.status)[1] if selected_row is not None else _BLUE
        cv2.rectangle(canvas, (24, 584), (647, 839), _rgb_to_bgr(evidence_border), 2)
    else:
        cv2.rectangle(canvas, (24, 584), (647, 839), _rgb_to_bgr(_GRID), 1)
    cv2.rectangle(canvas, (674, 584), (1034, 840), _rgb_to_bgr(_WHITE), -1)
    cv2.rectangle(canvas, (674, 584), (1034, 840), _rgb_to_bgr(_GRID), 1)

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
        x = 1084 + (index % 2) * 250
        y = 258 + (index // 2) * 132
        branch_rectangles.append((branch, x, y))
        branch_rows = tuple(row for row in results if row.branch is branch)
        branch_status = _aggregate(branch_rows)
        branch_color = _status(branch_status)[1]
        cv2.rectangle(canvas, (x, y), (x + 232, y + 108), _rgb_to_bgr(_WHITE), -1)
        cv2.rectangle(canvas, (x, y), (x + 232, y + 108), _rgb_to_bgr(branch_color), 2)

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
        f"证据放大：{_VIEW_LABELS[state.selected_view]} · {_BRANCH_LABELS[state.selected_branch]}",
        font=_demo_font(20, "bold"),
        fill=_INK,
    )
    if evidence is None:
        draw.text((336, 712), "当前项目暂无证据", font=_demo_font(21), fill=_MUTED, anchor="mm")
    draw.text((694, 602), "检测信息", font=_demo_font(23, "bold"), fill=_INK)
    detail_rows = [
        ("视角", _VIEW_LABELS[state.selected_view]),
        ("检测项目", _BRANCH_LABELS[state.selected_branch]),
        ("结果", "未执行" if selected_row is None else _status(selected_row.status)[0]),
    ]
    if state.experiment_mode:
        detail_rows.extend(
            [
                ("分数", "不可用" if selected_row is None or selected_row.score is None else f"{selected_row.score:.6g}"),
                ("阈值", "不可用" if selected_row is None or selected_row.threshold is None else f"{selected_row.threshold:.6g}"),
                ("耗时", "不可用" if selected_row is None else f"{selected_row.elapsed_ms:.1f} 毫秒"),
            ]
        )
    elif selected_row is not None:
        detail_rows.append(("说明", selected_row.reason[:18]))
    for index, (label, value) in enumerate(detail_rows[:6]):
        y = 650 + index * 29
        draw.text((694, y), label, font=_demo_font(16), fill=_MUTED)
        draw.text((790, y), value, font=_demo_font(16), fill=_INK)
    controls = "空格：拍摄/继续　R：重置　Q：退出"
    if state.experiment_mode:
        controls += "\n1–8：视角　T/L/Y/E：证据"
    for index, line in enumerate(controls.splitlines()):
        draw.text((1084, 542 + index * 29), line, font=_demo_font(16), fill=_MUTED)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


__all__ = [
    "DemoUiPhase",
    "EightViewUiState",
    "preferred_selection",
    "render_eight_view_dashboard",
]
