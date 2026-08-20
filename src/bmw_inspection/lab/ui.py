# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Chinese dual-mode UI for fast BMW laboratory inspection iteration."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bmw_inspection.lab.contracts import BranchEvidence, BranchName, BranchStatus, CaptureSet, FinalStatus, ViewId
from bmw_inspection.lab.runtime import RuntimeOutcome, RuntimeState
from bmw_inspection.lab.yolo import YoloEvidence


class UiMode(str, Enum):
    """The audience-specific amount of information shown on screen."""

    PRESENTATION = "presentation"
    EXPERIMENT = "experiment"


class UiPhase(str, Enum):
    """Explicit operator workflow states; startup is always idle."""

    IDLE = "IDLE"
    WAITING_FLIP = "WAITING_FLIP"
    PROCESSING = "PROCESSING"
    RESULT = "RESULT"
    ERROR = "ERROR"
    RELOAD_REQUIRED = "RELOAD_REQUIRED"


@dataclass(frozen=True, slots=True)
class StatusCopy:
    """Chinese status label with one truthful semantic colour."""

    title: str
    detail: str
    color: tuple[int, int, int]


_WHITE = (255, 255, 255)
_SURFACE = (247, 247, 248)
_INK = (20, 24, 31)
_MUTED = (102, 109, 120)
_GRID = (211, 214, 220)
_BLUE = (0, 47, 167)
_GREEN = (23, 132, 75)
_RED = (228, 0, 43)
_AMBER = (197, 117, 0)
_MAGENTA = (191, 0, 112)
_SKIPPED = (125, 130, 138)


_STATUS_COPY = {
    FinalStatus.OK: StatusCopy("检测通过", "所有必检项目均已通过", _GREEN),
    FinalStatus.NG_TEMPLATE: StatusCopy("模板匹配不通过", "零件位置或外观与模板不一致", _RED),
    FinalStatus.NG_BRIGHT_STREAK: StatusCopy("光痕检测不通过", "亮痕缺失或不连续", _RED),
    FinalStatus.NG_YOLO: StatusCopy("缺陷检测不通过", "检测到达到业务阈值的缺陷框", _RED),
    FinalStatus.NG_ANOMALY: StatusCopy("异常检测不通过", "异常模型分数超过阈值", _RED),
    FinalStatus.REVIEW: StatusCopy("待复核", "必检模型尚未完整配置或证据不完整", _AMBER),
    FinalStatus.RETAKE: StatusCopy("请重新拍摄", "图像质量不满足检测要求", _AMBER),
    FinalStatus.ERROR: StatusCopy("系统异常", "请检查相机、配置和模型后重试", _MAGENTA),
}


_VIEW_LABELS = {
    ViewId.FRONT: "正面中间",
    ViewId.FRONT_LEFT: "正面左侧",
    ViewId.FRONT_RIGHT: "正面右侧",
    ViewId.BACK: "反面中间",
    ViewId.BACK_LEFT: "反面左侧",
    ViewId.BACK_RIGHT: "反面右侧",
}
_BRANCH_LABELS = {
    BranchName.TEMPLATE: "模板匹配",
    BranchName.BRIGHT_STREAK: "光痕检测",
    BranchName.YOLO: "缺陷检测",
    BranchName.PATCHCORE: "异常检测",
}
_PHASE_COPY = {
    UiPhase.IDLE: StatusCopy("等待检测", "按空格键拍摄正面并开始本次检测", _BLUE),
    UiPhase.WAITING_FLIP: StatusCopy(
        "请翻转零件",
        "保持同一零件和固定工装，然后拍摄反面",
        _AMBER,
    ),
    UiPhase.PROCESSING: StatusCopy("正在检测", "正在依次执行模板、光痕、缺陷和异常检测", _BLUE),
    UiPhase.ERROR: _STATUS_COPY[FinalStatus.ERROR],
    UiPhase.RELOAD_REQUIRED: StatusCopy(
        "需要重新加载模型",
        "冷配置已变化，请按 R 重新加载模型",
        _AMBER,
    ),
}


class InspectionRuntime(Protocol):
    """Runtime boundary required by the event controller."""

    def inspect(self, capture_set: CaptureSet) -> RuntimeOutcome:
        """Inspect one complete six-view capture set."""


@dataclass(frozen=True, slots=True)
class UiSnapshot:
    """One deterministic render state independent from the GUI event loop."""

    phase: UiPhase = UiPhase.IDLE
    message: str = "等待检测"
    outcome: RuntimeOutcome | None = None
    previous_outcome: RuntimeOutcome | None = None
    selected_view: ViewId = ViewId.FRONT_LEFT
    selected_branch: BranchName = BranchName.BRIGHT_STREAK
    experiment_id: str = ""
    evidence_image: np.ndarray | None = None

    @property
    def previous_label(self) -> str:
        """The only label under which a stale result may be displayed."""
        return "上次结果"


class LabUiController:
    """Keep capture and inference exceptions inside an operator-recoverable state machine."""

    def __init__(
        self,
        *,
        capture_front: Callable[[], Any],
        capture_back: Callable[[], Any],
        assemble_capture: Callable[[Any, Any], CaptureSet],
        runtime: InspectionRuntime,
        experiment_id: str = "",
    ) -> None:
        self._capture_front = capture_front
        self._capture_back = capture_back
        self._assemble_capture = assemble_capture
        self._runtime = runtime
        self._front: Any = None
        self._snapshot = UiSnapshot(experiment_id=experiment_id)

    @property
    def snapshot(self) -> UiSnapshot:
        """Return the immutable current state."""
        return self._snapshot

    def capture_front(self) -> None:
        """Acquire only the first three views after an explicit action."""
        try:
            previous = self._snapshot.outcome or self._snapshot.previous_outcome
            self._front = self._capture_front()
            self._snapshot = replace(
                self._snapshot,
                phase=UiPhase.WAITING_FLIP,
                message="正面拍摄完成，请翻转同一零件",
                outcome=None,
                previous_outcome=previous,
            )
        except Exception as error:  # event-loop boundary
            self._set_error(error)

    def capture_back_and_inspect(self, before_inspect: Callable[[], None] | None = None) -> None:
        """Acquire the second round and synchronously inspect the complete set."""
        if self._snapshot.phase is not UiPhase.WAITING_FLIP:
            return
        try:
            back = self._capture_back()
            capture_set = self._assemble_capture(self._front, back)
            if not isinstance(capture_set, CaptureSet):
                raise TypeError("采集组合器必须返回完整的六视图 CaptureSet")
            self._snapshot = replace(self._snapshot, phase=UiPhase.PROCESSING, message="正在检测")
            if before_inspect is not None:
                before_inspect()
            outcome = self._runtime.inspect(capture_set)
            phase = UiPhase.RELOAD_REQUIRED if outcome.state is RuntimeState.RELOAD_REQUIRED else UiPhase.RESULT
            self._snapshot = replace(self._snapshot, phase=phase, message=outcome.reason, outcome=outcome)
        except Exception as error:  # event-loop boundary
            self._set_error(error)

    def retry(self) -> None:
        """Return to idle while retaining stale evidence under an explicit label."""
        previous = self._snapshot.outcome or self._snapshot.previous_outcome
        self._front = None
        self._snapshot = replace(
            self._snapshot,
            phase=UiPhase.IDLE,
            message="等待检测",
            outcome=None,
            previous_outcome=previous,
            evidence_image=None,
        )

    def select(self, view_id: ViewId, branch: BranchName) -> None:
        """Select one real view/branch evidence panel."""
        self._snapshot = replace(self._snapshot, selected_view=view_id, selected_branch=branch)

    def report_error(self, error: Exception) -> None:
        """Expose an event-loop error without terminating Retry or Quit handling."""
        self._set_error(error)

    def _set_error(self, error: Exception) -> None:
        previous = self._snapshot.outcome or self._snapshot.previous_outcome
        self._snapshot = replace(
            self._snapshot,
            phase=UiPhase.ERROR,
            message=f"错误：{error}",
            outcome=None,
            previous_outcome=previous,
        )


def status_copy(status: FinalStatus) -> StatusCopy:
    """Return stable, fully Chinese business copy."""
    return _STATUS_COPY[status]


def rotate_evidence_for_display(image: np.ndarray, branch: BranchName) -> np.ndarray:
    """Rotate only the bright-streak display bitmap, never detector coordinates."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("证据图必须是非空 numpy 图像")
    copied = image.copy()
    return cv2.rotate(copied, cv2.ROTATE_90_CLOCKWISE) if branch is BranchName.BRIGHT_STREAK else copied


def render_yolo_diagnostic_overlay(image: np.ndarray, evidence: YoloEvidence) -> np.ndarray:
    """Draw amber diagnostic candidates and red business-final detections."""
    if not isinstance(evidence, YoloEvidence):
        raise TypeError("缺陷诊断证据必须是 YoloEvidence")
    if image.ndim == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        overlay = image.copy()
    elif image.ndim == 3 and image.shape[2] == 4:
        overlay = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        raise ValueError(f"不支持的缺陷诊断图像尺寸: {image.shape}")
    final_boxes = set(evidence.final_boxes)
    for box in evidence.candidates:
        color = _RED if box in final_boxes else _AMBER
        cv2.rectangle(
            overlay,
            (int(round(box.x1)), int(round(box.y1))),
            (int(round(box.x2)), int(round(box.y2))),
            _rgb_to_bgr(color),
            2,
        )
    return overlay


def _result(snapshot: UiSnapshot) -> tuple[Any | None, bool]:
    if snapshot.outcome is not None and snapshot.outcome.inspection is not None:
        return snapshot.outcome.inspection, False
    if snapshot.previous_outcome is not None and snapshot.previous_outcome.inspection is not None:
        return snapshot.previous_outcome.inspection, True
    return None, False


def _screen_status(snapshot: UiSnapshot) -> StatusCopy:
    inspection, _stale = _result(snapshot)
    if snapshot.phase is UiPhase.RESULT and inspection is not None:
        return status_copy(inspection.final_status)
    return _PHASE_COPY.get(snapshot.phase, StatusCopy("检测状态未知", snapshot.message, _MAGENTA))


def _branch_status(evidence: tuple[BranchEvidence, ...], branch: BranchName) -> BranchStatus:
    rows = [row.status for row in evidence if row.branch is branch]
    for status in (BranchStatus.ERROR, BranchStatus.NG, BranchStatus.REVIEW):
        if status in rows:
            return status
    if rows and all(status is BranchStatus.PASS for status in rows):
        return BranchStatus.PASS
    return BranchStatus.SKIPPED


def view_frame_status(inspection: Any, view_id: ViewId) -> BranchStatus:
    """Return a truthful thumbnail frame state when evidence may be incomplete."""
    rows = [
        item.status
        for item in inspection.evidence
        if item.view_id is view_id and item.required_for_ok
    ]
    if inspection.final_status is FinalStatus.RETAKE:
        return BranchStatus.REVIEW
    if rows and all(status is BranchStatus.PASS for status in rows):
        return BranchStatus.PASS
    for status in (BranchStatus.ERROR, BranchStatus.NG, BranchStatus.REVIEW):
        if status in rows:
            return status
    return BranchStatus.SKIPPED


def _semantic(status: BranchStatus) -> tuple[tuple[int, int, int], str]:
    return {
        BranchStatus.PASS: (_GREEN, "通过"),
        BranchStatus.NG: (_RED, "不通过"),
        BranchStatus.REVIEW: (_AMBER, "待复核"),
        BranchStatus.ERROR: (_MAGENTA, "异常"),
        BranchStatus.SKIPPED: (_SKIPPED, "未执行"),
    }[status]


@lru_cache(maxsize=1)
def _font_path() -> Path:
    override = os.environ.get("BMW_DEMO_FONT")
    candidates = (
        Path(override).expanduser() if override else None,
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    raise RuntimeError("未找到可用字体，请安装 Noto Sans CJK 或设置 BMW_DEMO_FONT")


@lru_cache(maxsize=32)
def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(_font_path()), size=size)


def _fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        bgr = image.copy()
    elif image.ndim == 3 and image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        raise ValueError(f"不支持的图像尺寸: {image.shape}")
    scale = min(width / bgr.shape[1], height / bgr.shape[0])
    resized = cv2.resize(
        bgr,
        (max(1, int(round(bgr.shape[1] * scale))), max(1, int(round(bgr.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    panel = np.full((height, width, 3), 238, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return panel


def _rgb_to_bgr(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return color[2], color[1], color[0]


def render_dashboard(snapshot: UiSnapshot, mode: UiMode) -> np.ndarray:
    """Render a deterministic 1600x900 Swiss-grid dashboard."""
    if not isinstance(snapshot, UiSnapshot) or not isinstance(mode, UiMode):
        raise TypeError("render_dashboard 需要 UiSnapshot 和 UiMode")
    canvas = np.full((900, 1600, 3), _rgb_to_bgr(_SURFACE), dtype=np.uint8)
    inspection, stale = _result(snapshot)
    evidence = inspection.evidence if inspection is not None else ()

    # Structural one-pixel Swiss grid.
    cv2.line(canvas, (0, 68), (1599, 68), _rgb_to_bgr(_GRID), 1)
    cv2.line(canvas, (1060, 68), (1060, 899), _rgb_to_bgr(_GRID), 1)
    cv2.line(canvas, (0, 590), (1060, 590), _rgb_to_bgr(_GRID), 1)

    if inspection is not None:
        for index, view_id in enumerate(ViewId):
            column = index % 3
            row = index // 3
            x, y = 24 + column * 342, 92 + row * 244
            image = inspection.capture_set.views[view_id].image
            panel = _fit_image(image, 318, 178)
            canvas[y + 34 : y + 212, x : x + 318] = panel
            frame_status = view_frame_status(inspection, view_id)
            color, _ = _semantic(frame_status)
            cv2.rectangle(canvas, (x, y + 34), (x + 317, y + 211), _rgb_to_bgr(color), 2)
    status = _screen_status(snapshot)
    cv2.rectangle(canvas, (1084, 96), (1574, 224), _rgb_to_bgr(status.color), 4)

    branch_rects = {}
    for index, branch in enumerate(BranchName):
        x = 1084 + (index % 2) * 250
        y = 258 + (index // 2) * 132
        branch_rects[branch] = (x, y, x + 232, y + 108)
        color, _label = _semantic(_branch_status(evidence, branch))
        cv2.rectangle(canvas, (x, y), (x + 232, y + 108), _rgb_to_bgr(_WHITE), -1)
        cv2.rectangle(canvas, (x, y), (x + 232, y + 108), _rgb_to_bgr(color), 2)

    selected = snapshot.evidence_image
    if (
        mode is UiMode.EXPERIMENT
        and inspection is not None
        and snapshot.selected_branch is BranchName.YOLO
    ):
        yolo_row = next(
            (
                row
                for row in evidence
                if row.branch is BranchName.YOLO
                and row.view_id is snapshot.selected_view
                and isinstance(row, YoloEvidence)
            ),
            None,
        )
        if yolo_row is not None:
            source = inspection.capture_set.views[snapshot.selected_view].image
            selected = render_yolo_diagnostic_overlay(source, yolo_row)
    is_branch_artifact = selected is not None
    if selected is not None:
        shown = (
            rotate_evidence_for_display(selected, snapshot.selected_branch)
            if is_branch_artifact
            else selected.copy()
        )
        canvas[642:858, 24:648] = _fit_image(shown, 624, 216)
        cv2.rectangle(canvas, (24, 642), (647, 857), _rgb_to_bgr(_BLUE), 2)
    else:
        cv2.rectangle(canvas, (24, 642), (647, 857), _rgb_to_bgr(_WHITE), -1)
        cv2.rectangle(canvas, (24, 642), (647, 857), _rgb_to_bgr(_GRID), 1)
    cv2.rectangle(canvas, (674, 642), (1034, 857), _rgb_to_bgr(_WHITE), -1)
    cv2.rectangle(canvas, (674, 642), (1034, 857), _rgb_to_bgr(_GRID), 1)

    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    draw.text((24, 16), "BMW 零件六视图检测", font=_font(30), fill=_INK)
    draw.text((390, 24), "展示模式" if mode is UiMode.PRESENTATION else "实验模式", font=_font(18), fill=_BLUE)
    if snapshot.experiment_id:
        draw.text((1574, 24), f"实验：{snapshot.experiment_id}", font=_font(17), fill=_MUTED, anchor="ra")
    for index, view_id in enumerate(ViewId):
        column = index % 3
        row = index // 3
        x, y = 24 + column * 342, 92 + row * 244
        draw.text((x, y - 2), f"{index + 1:02d}", font=_font(29), fill=_BLUE)
        draw.text((x + 54, y + 5), _VIEW_LABELS[view_id], font=_font(18), fill=_INK)
    draw.text((1104, 116), status.title, font=_font(31), fill=status.color)
    draw.text((1104, 172), status.detail, font=_font(16), fill=_MUTED)
    if stale:
        draw.text((1518, 74), snapshot.previous_label, font=_font(16), fill=_AMBER, anchor="ra")
    for branch, (x1, y1, _x2, _y2) in branch_rects.items():
        branch_status = _branch_status(evidence, branch)
        color, label = _semantic(branch_status)
        draw.text((x1 + 16, y1 + 14), _BRANCH_LABELS[branch], font=_font(20), fill=_INK)
        draw.text((x1 + 16, y1 + 58), label, font=_font(23), fill=color)
    evidence_title = (
        f"证据放大：{_VIEW_LABELS[snapshot.selected_view]} · "
        f"{_BRANCH_LABELS[snapshot.selected_branch]}"
    )
    draw.text((24, 606), evidence_title, font=_font(19), fill=_INK)
    if selected is None:
        draw.text((336, 748), "当前视角没有该项目证据", font=_font(20), fill=_SKIPPED, anchor="mm")
    draw.text((694, 662), "检测信息", font=_font(22), fill=_INK)
    if mode is UiMode.EXPERIMENT:
        selected_rows = [
            row for row in evidence if row.view_id is snapshot.selected_view and row.branch is snapshot.selected_branch
        ]
        row = selected_rows[0] if selected_rows else None
        rows = (
            ("分数", "不可用" if row is None or row.score is None else f"{row.score:.6g}"),
            ("阈值", "不可用" if row is None or row.threshold is None else f"{row.threshold:.6g}"),
            ("耗时", "不可用" if row is None else f"{row.elapsed_ms:.2f} 毫秒"),
            ("模型", "未配置" if row is None or row.model_id is None else Path(row.model_id).name),
        )
    else:
        rows = (
            ("当前步骤", status.title),
            ("视角", _VIEW_LABELS[snapshot.selected_view]),
            ("检测项目", _BRANCH_LABELS[snapshot.selected_branch]),
        )
    for index, (label, value) in enumerate(rows):
        y = 708 + index * 36
        draw.text((694, y), label, font=_font(16), fill=_MUTED)
        draw.text((790, y), value[:23], font=_font(16), fill=_INK)
    if snapshot.phase is UiPhase.RELOAD_REQUIRED:
        operation_lines = ("R：重新加载模型　Q：退出",)
    elif mode is UiMode.EXPERIMENT:
        operation_lines = (
            "空格：拍摄　R：重新检测　Q：退出",
            "1–6：切换视角　T/L/Y/P：切换证据　M：重载模型",
        )
    else:
        operation_lines = ("空格：拍摄　R：重新检测　Q：退出",)
    for index, line in enumerate(operation_lines):
        draw.text((1084, 538 + index * 28), line, font=_font(15), fill=_MUTED)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


__all__ = [
    "LabUiController",
    "StatusCopy",
    "UiMode",
    "UiPhase",
    "UiSnapshot",
    "render_dashboard",
    "render_yolo_diagnostic_overlay",
    "rotate_evidence_for_display",
    "status_copy",
    "view_frame_status",
]
