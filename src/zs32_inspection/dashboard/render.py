# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pure OpenCV canvas renderer for the ZS32 eight-view dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .compositor import compose_view, fit_letterbox
from .contracts import EvidenceLayer, InspectionResult, ProgressRecord, ViewResult

_BACKGROUND = (14, 15, 16)
_SURFACE = (22, 23, 24)
_SURFACE_ACTIVE = (36, 37, 38)
_BORDER = (76, 78, 80)
_TEXT = (232, 232, 228)
_MUTED = (158, 160, 158)
_GREEN = (92, 201, 120)
_RED = (80, 92, 235)
_AMBER = (72, 186, 238)
_DISABLED = (70, 72, 73)

_LAYER_LABELS = (
    (EvidenceLayer.FUSION, "Fusion [1]"),
    (EvidenceLayer.ORIGINAL, "Original [2]"),
    (EvidenceLayer.PATCHCORE, "PatchCore [3]"),
    (EvidenceLayer.YOLO, "YOLO [4]"),
    (EvidenceLayer.TEMPLATE, "Template [5]"),
)


@dataclass(frozen=True, slots=True)
class Rect:
    """Integer rectangle used for drawing and hit testing."""

    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> tuple[int, int]:
        """Return the integer center point."""
        return self.x + self.width // 2, self.y + self.height // 2

    def contains(self, point: tuple[int, int]) -> bool:
        """Return whether *point* lies inside the half-open rectangle."""
        x, y = point
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height


@dataclass(frozen=True, slots=True)
class HitRegion:
    """One mouse target represented with a reducer action and optional value."""

    rect: Rect
    action: str
    value: str | None = None
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class ButtonVisual:
    """Rendered button geometry and availability."""

    rect: Rect
    label: str
    enabled: bool


@dataclass(frozen=True, slots=True)
class DashboardState:
    """Complete immutable state consumed by the renderer and reducer."""

    result: InspectionResult
    layer: EvidenceLayer = EvidenceLayer.FUSION
    selected_view: str | None = None
    progress: ProgressRecord | None = None
    running: bool = False
    inspection_requested: bool = False
    exit_requested: bool = False


@dataclass(frozen=True, slots=True)
class RenderFrame:
    """One BGR frame plus all interactive geometry."""

    canvas: np.ndarray
    hit_regions: tuple[HitRegion, ...]
    inspection_button: ButtonVisual
    selected_view: str | None = None
    notice: str = ""


@dataclass(frozen=True, slots=True)
class _Text:
    text: str
    xy: tuple[int, int]
    size: int
    color: tuple[int, int, int]
    bold: bool = False


@lru_cache(maxsize=32)
def _font(size: int, bold: bool) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
        if bold
        else Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf")
        if bold
        else Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def _paint_text(canvas: np.ndarray, items: list[_Text]) -> None:
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    pillow = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pillow)
    for item in items:
        draw.text(item.xy, item.text, font=_font(item.size, item.bold), fill=item.color[::-1])
    canvas[:] = cv2.cvtColor(np.asarray(pillow), cv2.COLOR_RGB2BGR)


def _outline(canvas: np.ndarray, rect: Rect, color: tuple[int, int, int], thickness: int = 1) -> None:
    cv2.rectangle(
        canvas,
        (rect.x, rect.y),
        (rect.x + rect.width - 1, rect.y + rect.height - 1),
        color,
        thickness,
        lineType=cv2.LINE_8,
    )


def _status_color(status: str) -> tuple[int, int, int]:
    normalized = status.strip().upper()
    if normalized in {"OK", "PASS", "CLEAR", "AVAILABLE"}:
        return _GREEN
    if normalized.startswith("NG") or normalized in {"FAIL", "ERROR", "SYSTEM_ERROR"}:
        return _RED
    return _AMBER


def _ellipsize(text: str, limit: int) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[: max(0, limit - 3)] + "..."


def inspection_button_label(progress: ProgressRecord | None, running: bool) -> tuple[str, bool]:
    """Return the frozen context label and enabled state for the sole CTA."""
    if not running:
        return "开始检测 [S]", True
    if progress and progress.state == "waiting_front":
        return "确认正面并拍摄 [S]", True
    if progress and progress.state == "waiting_back":
        return "确认背面并拍摄 [S]", True
    return "检测运行中", False


def _view_status(view: ViewResult, layer: EvidenceLayer) -> tuple[str, float | None, str]:
    if layer is EvidenceLayer.ORIGINAL:
        return "", None, ""
    branch = view.branches.get(layer.value)
    if branch is None:
        return "ERROR", None, "branch missing"
    return branch.status, branch.score, branch.reason


def _place_image(canvas: np.ndarray, image: np.ndarray, rect: Rect) -> None:
    fitted = fit_letterbox(image, rect.width, rect.height)
    x = rect.x + fitted.offset_x
    y = rect.y + fitted.offset_y
    canvas[y : y + fitted.image.shape[0], x : x + fitted.image.shape[1]] = fitted.image


def _render_header(
    canvas: np.ndarray,
    state: DashboardState,
    texts: list[_Text],
    hits: list[HitRegion],
) -> None:
    identity = state.result.identity
    status = state.result.machine_status
    texts.extend([
        _Text("ZS32 / RIGHT  八视角检测", (16, 10), 28, _TEXT, True),
        _Text(
            f"part_id={identity.part_id}  capture_session={identity.capture_session}  group_id={identity.group_id}",
            (16, 50),
            18,
            _MUTED,
        ),
        _Text(f"总状态  {status}", (1260, 12), 24, _status_color(status), True),
        _Text(_ellipsize(state.result.reason, 46), (1090, 50), 16, _MUTED),
    ])
    cv2.line(canvas, (16, 92), (1584, 92), _BORDER, 1, cv2.LINE_8)
    for index, (layer, label) in enumerate(_LAYER_LABELS):
        rect = Rect(16 + index * 186, 104, 174, 52)
        active = state.layer is layer
        cv2.rectangle(
            canvas,
            (rect.x, rect.y),
            (rect.x + rect.width - 1, rect.y + rect.height - 1),
            _SURFACE_ACTIVE if active else _SURFACE,
            -1,
        )
        _outline(canvas, rect, _TEXT if active else _BORDER)
        texts.append(_Text(label, (rect.x + 12, rect.y + 13), 17, _TEXT if active else _MUTED, active))
        hits.append(HitRegion(rect, "select_layer", layer.value))


def _render_grid(
    canvas: np.ndarray,
    state: DashboardState,
    texts: list[_Text],
    hits: list[HitRegion],
) -> str:
    notice = ""
    card_width, card_height = 380, 302
    for index, view in enumerate(state.result.views):
        column, row = index % 4, index // 4
        rect = Rect(16 + column * 392, 170 + row * 314, card_width, card_height)
        cv2.rectangle(canvas, (rect.x, rect.y), (rect.x + rect.width - 1, rect.y + rect.height - 1), _SURFACE, -1)
        _outline(canvas, rect, _BORDER)
        composed = compose_view(view, state.layer)
        image_rect = Rect(rect.x + 1, rect.y + 35, rect.width - 2, 202)
        _place_image(canvas, composed.image, image_rect)
        status, score, reason = _view_status(view, state.layer)
        texts.append(_Text(view.view, (rect.x + 10, rect.y + 7), 18, _TEXT, True))
        if status:
            texts.append(_Text(status, (rect.x + 220, rect.y + 7), 16, _status_color(status), True))
        score_text = "" if score is None else f"score={score:.6g}"
        if score_text:
            texts.append(_Text(score_text, (rect.x + 10, rect.y + 243), 15, _TEXT))
        detail = composed.notice or reason
        if detail:
            texts.append(_Text(_ellipsize(detail, 45), (rect.x + 10, rect.y + 270), 14, _AMBER))
        hits.append(HitRegion(rect, "select_view", view.view))
    return notice


def _render_selected(canvas: np.ndarray, state: DashboardState, texts: list[_Text]) -> str:
    view = next(view for view in state.result.views if view.view == state.selected_view)
    composed = compose_view(view, state.layer)
    rect = Rect(16, 170, 1568, 616)
    cv2.rectangle(canvas, (rect.x, rect.y), (rect.x + rect.width - 1, rect.y + rect.height - 1), _SURFACE, -1)
    _outline(canvas, rect, _BORDER)
    _place_image(canvas, composed.image, Rect(rect.x + 1, rect.y + 45, rect.width - 2, rect.height - 92))
    status, score, reason = _view_status(view, state.layer)
    texts.append(_Text(view.view, (rect.x + 14, rect.y + 8), 22, _TEXT, True))
    if status:
        texts.append(_Text(status, (rect.x + 350, rect.y + 9), 18, _status_color(status), True))
    if score is not None:
        texts.append(_Text(f"score={score:.6g}", (rect.x + 530, rect.y + 9), 18, _TEXT))
    detail = composed.notice or reason
    if detail:
        texts.append(_Text(_ellipsize(detail, 120), (rect.x + 14, rect.y + rect.height - 39), 17, _AMBER))
    texts.append(_Text("Esc 返回八视角", (rect.x + rect.width - 190, rect.y + 10), 16, _MUTED))
    return composed.notice


def _render_footer(
    canvas: np.ndarray,
    state: DashboardState,
    texts: list[_Text],
    hits: list[HitRegion],
) -> ButtonVisual:
    cv2.line(canvas, (16, 800), (1584, 800), _BORDER, 1, cv2.LINE_8)
    quit_rect = Rect(16, 824, 132, 64)
    cv2.rectangle(
        canvas,
        (quit_rect.x, quit_rect.y),
        (quit_rect.x + quit_rect.width - 1, quit_rect.y + quit_rect.height - 1),
        _SURFACE,
        -1,
    )
    _outline(canvas, quit_rect, _BORDER)
    texts.append(_Text("退出 [Q]", (quit_rect.x + 21, quit_rect.y + 19), 19, _TEXT, True))
    hits.append(HitRegion(quit_rect, "quit"))

    label, enabled = inspection_button_label(state.progress, state.running)
    rect = Rect(600, 816, 400, 72)
    fill = (42, 78, 55) if enabled else _DISABLED
    cv2.rectangle(canvas, (rect.x, rect.y), (rect.x + rect.width - 1, rect.y + rect.height - 1), fill, -1)
    _outline(canvas, rect, _GREEN if enabled else _BORDER)
    texts.append(_Text(label, (rect.x + 52, rect.y + 20), 22, _TEXT if enabled else _MUTED, True))
    hits.append(HitRegion(rect, "inspection_action", enabled=enabled))
    return ButtonVisual(rect, label, enabled)


def render_dashboard(state: DashboardState, width: int = 1600, height: int = 920) -> RenderFrame:
    """Render the fixed industrial dashboard as a uint8 BGR frame."""
    if (width, height) != (1600, 920):
        raise ValueError("dashboard canvas is fixed at 1600x920")
    canvas = np.full((height, width, 3), _BACKGROUND, dtype=np.uint8)
    texts: list[_Text] = []
    hits: list[HitRegion] = []
    _render_header(canvas, state, texts, hits)
    notice = _render_selected(canvas, state, texts) if state.selected_view else _render_grid(canvas, state, texts, hits)
    button = _render_footer(canvas, state, texts, hits)
    _paint_text(canvas, texts)
    return RenderFrame(canvas, tuple(hits), button, state.selected_view, notice)
