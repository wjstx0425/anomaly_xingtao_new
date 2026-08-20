"""Minimal local UI and atomic publication for the BMW bright-streak Demo."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .camera import SingleCameraSession
from .contracts import BrightStreakConfig, BrightStreakResult, DemoStatus, load_config
from .detector import BrightStreakDecision, detect_bright_streak_evidence, render_evidence


class Action(str, Enum):
    """Small operator action set shared by keyboard and buttons."""

    CAPTURE = "capture"
    RETRY = "retry"
    QUIT = "quit"


@dataclass(frozen=True, slots=True)
class StatusPresentation:
    """Chinese business copy and semantic signal color for one outcome."""

    title: str
    detail: str
    color: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class MetricPresentation:
    """One simplified metric row for the customer dashboard."""

    label: str
    value: str
    progress: float | None


@dataclass(frozen=True, slots=True)
class _TextSpec:
    text: str
    xy: tuple[int, int]
    size: int
    color: tuple[int, int, int]
    anchor: str = "la"


_CANVAS_SIZE = (1600, 900)
_BG = (15, 18, 21)
_CARD = (24, 31, 37)
_CARD_ALT = (29, 39, 47)
_BORDER = (61, 78, 91)
_BORDER_ACTIVE = (191, 174, 48)
_TEXT = (238, 239, 235)
_MUTED = (174, 163, 145)
_CYAN = (200, 184, 57)
_OK = (118, 230, 0)
_NG = (48, 59, 255)
_ERROR = (0, 184, 255)
_ROI = (0, 212, 255)


_STATUS_PRESENTATIONS = {
    DemoStatus.OK: StatusPresentation("检测通过", "亮痕存在且连续", _OK),
    DemoStatus.NG_NO_STREAK: StatusPresentation("检测不通过", "未检测到有效亮痕", _NG),
    DemoStatus.NG_BROKEN: StatusPresentation("检测不通过", "亮痕存在但不连续", _NG),
    DemoStatus.ERROR: StatusPresentation("设备或图像异常", "请检查相机、光源与工件位置", _ERROR),
}


_KEY_ACTIONS = {
    ord(" "): Action.CAPTURE,
    ord("r"): Action.RETRY,
    ord("R"): Action.RETRY,
    ord("q"): Action.QUIT,
    ord("Q"): Action.QUIT,
    27: Action.QUIT,
}

_BUTTONS = (
    (Action.CAPTURE, (985, 810, 1165, 870), "开始检测"),
    (Action.RETRY, (1180, 810, 1360, 870), "重新检测"),
    (Action.QUIT, (1375, 810, 1555, 870), "退出"),
)


def status_presentation(status: DemoStatus) -> StatusPresentation:
    """Return stable Chinese copy for a business outcome."""
    return _STATUS_PRESENTATIONS[status]


def format_metric_rows(
    result: BrightStreakResult,
    config: BrightStreakConfig,
) -> tuple[MetricPresentation, ...]:
    """Convert detector evidence into four customer-readable metric rows."""
    metrics = result.metrics
    if metrics is None:
        return (
            MetricPresentation("亮痕覆盖率", "不可用", None),
            MetricPresentation("连续率", "不可用", None),
            MetricPresentation("最大断点", "不可用", None),
            MetricPresentation("对比度", "不可用", None),
        )
    gap_quality = "合格" if metrics.max_gap_ratio <= config.max_gap_ratio else "超限"
    contrast_quality = "良好" if metrics.contrast_snr >= config.min_contrast_snr else "偏低"
    contrast_progress = min(1.0, metrics.contrast_snr / max(1.0, config.min_contrast_snr * 1.5))
    gap_progress = max(0.0, 1.0 - metrics.max_gap_ratio / max(config.max_gap_ratio, 1e-6))
    gap_row = (
        MetricPresentation("最大断点", "不可用", None)
        if result.status is DemoStatus.NG_NO_STREAK
        else MetricPresentation("最大断点", f"{metrics.max_gap_px} 像素 · {gap_quality}", gap_progress)
    )
    return (
        MetricPresentation("亮痕覆盖率", f"{metrics.coverage_ratio * 100:.1f}%", metrics.coverage_ratio),
        MetricPresentation("连续率", f"{metrics.longest_run_ratio * 100:.1f}%", metrics.longest_run_ratio),
        gap_row,
        MetricPresentation("对比度", f"{contrast_quality} · {metrics.contrast_snr:.2f}", contrast_progress),
    )


@lru_cache(maxsize=1)
def _font_path() -> Path:
    override = os.environ.get("BMW_DEMO_FONT")
    candidates = [
        Path(override).expanduser() if override else None,
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    raise RuntimeError(
        "未找到中文字体；请安装 Noto Sans CJK SC 或通过 BMW_DEMO_FONT 指定字体文件",
    )


@lru_cache(maxsize=16)
def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(_font_path()), size=size)


def _draw_text(draw: ImageDraw.ImageDraw, spec: _TextSpec) -> None:
    """Draw one antialiased Chinese text item onto an RGB Pillow surface."""
    b, g, r = spec.color
    draw.text(spec.xy, spec.text, font=_font(spec.size), fill=(r, g, b), anchor=spec.anchor)


def _apply_text(canvas: np.ndarray, specs: list[_TextSpec]) -> np.ndarray:
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    for spec in specs:
        _draw_text(draw, spec)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def key_to_action(key: int) -> Action | None:
    """Map one OpenCV key code to a Demo action."""
    return _KEY_ACTIONS.get(key & 0xFF) if key >= 0 else None


def mouse_to_action(x: int, y: int) -> Action | None:
    """Map one click point to a drawn operator button."""
    for action, (x1, y1, x2, y2), _label in _BUTTONS:
        if x1 <= x < x2 and y1 <= y < y2:
            return action
    return None


def action_enabled(action: Action, *, has_result: bool) -> bool:
    """Return whether one operator action is available in the current UI state."""
    if action is Action.RETRY:
        return has_result
    return action in {Action.CAPTURE, Action.QUIT}


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"unsupported image shape: {image.shape}")


def _write_image(path: Path, image: np.ndarray) -> None:
    if image.size == 0 or not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write Demo artifact: {path}")


def _result_payload(
    result: BrightStreakResult,
    config: BrightStreakConfig,
    source_ref: str,
    captured_at: datetime,
) -> dict[str, object]:
    metrics = asdict(result.metrics) if result.metrics is not None else None
    return {
        "schema_version": 1,
        "captured_at": captured_at.isoformat(timespec="milliseconds"),
        "camera_serial": config.camera_serial,
        "source_image": source_ref,
        "status": result.status.value,
        "reason": result.reason,
        "roi_xyxy": list(result.roi_xyxy),
        "metrics": metrics,
    }


def publish_result(
    image: np.ndarray,
    decision: BrightStreakDecision,
    config: BrightStreakConfig,
    *,
    output_root: Path,
    source_ref: str,
    captured_at: datetime | None = None,
) -> Path:
    """Atomically publish one complete, explainable inspection directory."""
    timestamp = captured_at or datetime.now().astimezone()
    day_root = Path(output_root).expanduser().resolve() / timestamp.strftime("%Y%m%d")
    day_root.mkdir(parents=True, exist_ok=True)
    stem = timestamp.strftime("%H%M%S_%f")
    final_dir = day_root / stem
    suffix = 1
    while final_dir.exists() or (day_root / f".{final_dir.name}.tmp").exists():
        final_dir = day_root / f"{stem}_{suffix:02d}"
        suffix += 1
    temp_dir = day_root / f".{final_dir.name}.tmp"
    temp_dir.mkdir()

    result = decision.result
    try:
        evidence = render_evidence(image, decision)
    except (ValueError, cv2.error):
        evidence = cv2.cvtColor(_gray(image), cv2.COLOR_GRAY2BGR)
    _write_image(temp_dir / "source.png", image)
    _write_image(temp_dir / "roi.png", decision.roi)
    _write_image(temp_dir / "response.png", decision.response)
    _write_image(temp_dir / "mask.png", decision.mask)
    _write_image(temp_dir / "evidence.png", evidence)
    payload = _result_payload(result, config, source_ref, timestamp)
    (temp_dir / "result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp_dir.replace(final_dir)
    return final_dir


def _default_output_root(config: BrightStreakConfig) -> Path:
    if config.result_root.is_absolute():
        return config.result_root
    repo_root = config.path.parents[2] if len(config.path.parents) >= 3 else Path.cwd()
    return repo_root / config.result_root


def run_image(
    image: np.ndarray,
    config: BrightStreakConfig,
    *,
    output_root: Path | None = None,
    source_ref: str,
    capture_elapsed_ms: float = 0.0,
) -> tuple[BrightStreakResult, Path]:
    """Detect and publish one already acquired image."""
    decision = detect_bright_streak_evidence(
        image,
        config,
        capture_elapsed_ms=capture_elapsed_ms,
    )
    result = decision.result
    result_dir = publish_result(
        image,
        decision,
        config,
        output_root=output_root or _default_output_root(config),
        source_ref=source_ref,
    )
    return result, result_dir


def run_offline(
    image_path: Path,
    config_path: Path,
    output_root: Path | None = None,
) -> BrightStreakResult:
    """Run the exact Demo decision path on one saved image."""
    resolved_image = Path(image_path).expanduser().resolve()
    capture_started = time.perf_counter()
    image = cv2.imread(str(resolved_image), cv2.IMREAD_UNCHANGED)
    capture_elapsed_ms = (time.perf_counter() - capture_started) * 1000.0
    if image is None:
        raise FileNotFoundError(f"unable to read image: {resolved_image}")
    config = load_config(config_path)
    result, _result_dir = run_image(
        image,
        config,
        output_root=output_root,
        source_ref=str(resolved_image),
        capture_elapsed_ms=capture_elapsed_ms,
    )
    return result


def capture_once(
    config: BrightStreakConfig,
    *,
    output_root: Path | None = None,
) -> tuple[np.ndarray, BrightStreakResult, Path]:
    """Open the configured serial camera, acquire one frame, and inspect it."""
    with SingleCameraSession(config) as session:
        capture_started = time.perf_counter()
        image = session.capture()
        capture_elapsed_ms = (time.perf_counter() - capture_started) * 1000.0
    result, result_dir = run_image(
        image,
        config,
        output_root=output_root,
        source_ref=f"camera:{config.camera_serial}",
        capture_elapsed_ms=capture_elapsed_ms,
    )
    return image, result, result_dir


def _fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), _BG, dtype=np.uint8)
    bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    scale = min(width / bgr.shape[1], height / bgr.shape[0])
    resized = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _prepare_roi_evidence(zoom: np.ndarray, width: int, height: int) -> np.ndarray:
    """Rotate ROI evidence clockwise and fit it proportionally for presentation."""
    if zoom.size == 0:
        return np.full((height, width, 3), _BG, dtype=np.uint8)
    rotated = cv2.rotate(zoom, cv2.ROTATE_90_CLOCKWISE)
    return _fit(rotated, width, height)


def render_dashboard(
    image: np.ndarray,
    result: BrightStreakResult,
    config: BrightStreakConfig,
    result_dir: Path,
    *,
    hovered_action: Action | None = None,
    camera_connected: bool = False,
    now: datetime | None = None,
) -> np.ndarray:
    """Render the fixed 1600x900 fully Chinese customer Demo screen."""
    del result_dir  # Publication paths intentionally stay off the presentation surface.
    canvas = np.full((_CANVAS_SIZE[1], _CANVAS_SIZE[0], 3), _BG, dtype=np.uint8)
    text_specs: list[_TextSpec] = []
    displayed_at = now or datetime.now().astimezone()
    presentation = status_presentation(result.status)

    # Header: flat industrial rail with real runtime identity.
    cv2.rectangle(canvas, (0, 0), (1599, 94), _CARD, -1)
    cv2.line(canvas, (0, 94), (1599, 94), _BORDER, 1)
    cv2.rectangle(canvas, (28, 24), (34, 72), _CYAN, -1)
    text_specs.extend(
        [
            _TextSpec("BMW 零件亮痕检测演示系统", (50, 22), 30, _TEXT),
            _TextSpec("固定相机 · 固定光源 · 单次触发检测", (51, 62), 15, _MUTED),
            _TextSpec(displayed_at.strftime("%Y-%m-%d  %H:%M:%S"), (1568, 25), 18, _TEXT, "ra"),
            _TextSpec(f"相机序列号  {config.camera_serial}", (1568, 58), 15, _MUTED, "ra"),
        ],
    )
    connection_color = _OK if camera_connected else _CYAN
    connection_text = "相机已连接" if camera_connected else "离线图像回放"
    cv2.circle(canvas, (1170, 45), 6, connection_color, -1, cv2.LINE_AA)
    text_specs.append(_TextSpec(connection_text, (1186, 34), 16, _TEXT))

    # Main source card.
    cv2.rectangle(canvas, (28, 112), (950, 790), _CARD, -1)
    cv2.rectangle(canvas, (28, 112), (950, 790), _BORDER, 1)
    cv2.line(canvas, (28, 154), (950, 154), _BORDER, 1)
    text_specs.append(_TextSpec("实时采集图像", (44, 122), 18, _TEXT))
    source_view = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    roi_x1, roi_y1, roi_x2, roi_y2 = config.require_detection_roi()
    cv2.rectangle(source_view, (roi_x1, roi_y1), (roi_x2 - 1, roi_y2 - 1), _ROI, 7)
    source_panel = _fit(source_view, 894, 620)
    canvas[162:782, 42:936] = source_panel

    # ROI evidence card. Horizontal magnification makes the narrow streak legible.
    cv2.rectangle(canvas, (974, 112), (1572, 380), _CARD, -1)
    cv2.rectangle(canvas, (974, 112), (1572, 380), _BORDER, 1)
    cv2.line(canvas, (974, 154), (1572, 154), _BORDER, 1)
    text_specs.append(_TextSpec("亮痕检测区域", (990, 122), 18, _TEXT))
    evidence = render_evidence(image, result)
    x1, y1, x2, y2 = config.require_detection_roi()
    zoom = evidence[y1:y2, x1:x2]
    canvas[164:364, 990:1556] = _prepare_roi_evidence(zoom, width=566, height=200)
    text_specs.append(_TextSpec("ROI 放大视图", (1544, 337), 13, _MUTED, "ra"))

    # Business result card with a strong signal rail.
    cv2.rectangle(canvas, (974, 396), (1572, 548), _CARD_ALT, -1)
    cv2.rectangle(canvas, (974, 396), (1572, 548), _BORDER, 1)
    cv2.rectangle(canvas, (974, 396), (986, 548), presentation.color, -1)
    text_specs.extend(
        [
            _TextSpec(presentation.title, (1012, 417), 36, presentation.color),
            _TextSpec(presentation.detail, (1014, 474), 20, _TEXT),
            _TextSpec("本次检测结果", (1546, 418), 14, _MUTED, "ra"),
        ],
    )

    # Simplified metric card: real values, no internal detector prose.
    cv2.rectangle(canvas, (974, 564), (1572, 790), _CARD, -1)
    cv2.rectangle(canvas, (974, 564), (1572, 790), _BORDER, 1)
    text_specs.append(_TextSpec("检测指标", (990, 576), 18, _TEXT))
    for index, row in enumerate(format_metric_rows(result, config)):
        row_y = 616 + index * 42
        text_specs.append(_TextSpec(row.label, (994, row_y), 15, _MUTED))
        text_specs.append(_TextSpec(row.value, (1548, row_y), 15, _TEXT, "ra"))
        cv2.rectangle(canvas, (1135, row_y + 5), (1395, row_y + 13), _BG, -1)
        cv2.rectangle(canvas, (1135, row_y + 5), (1395, row_y + 13), _BORDER, 1)
        if row.progress is not None:
            progress = max(0.0, min(1.0, row.progress))
            fill_x = 1135 + int(round(260 * progress))
            if fill_x > 1135:
                cv2.rectangle(canvas, (1135, row_y + 5), (fill_x, row_y + 13), _CYAN, -1)

    # Footer and flat industrial controls.
    cv2.line(canvas, (0, 802), (1599, 802), _BORDER, 1)
    cv2.circle(canvas, (34, 842), 5, presentation.color, -1, cv2.LINE_AA)
    text_specs.append(_TextSpec("检测完成 · 结果已保存", (50, 829), 16, _TEXT))
    text_specs.append(_TextSpec("空格键开始检测  ·  R 重新检测  ·  Q 退出", (50, 858), 13, _MUTED))
    for action, (bx1, by1, bx2, by2), label in _BUTTONS:
        hovered = action is hovered_action
        if action is Action.QUIT:
            fill = (48, 43, 63) if not hovered else (58, 52, 91)
        else:
            fill = (44, 57, 67) if not hovered else (58, 76, 89)
        border = _BORDER_ACTIVE if hovered else _BORDER
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), fill, -1)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), border, 2 if hovered else 1)
        text_specs.append(_TextSpec(label, ((bx1 + bx2) // 2, (by1 + by2) // 2), 17, _TEXT, "mm"))
    return _apply_text(canvas, text_specs)


def render_waiting_dashboard(
    config: BrightStreakConfig,
    *,
    hovered_action: Action | None = None,
    camera_connected: bool = True,
    now: datetime | None = None,
) -> np.ndarray:
    """Render the initial connected state without acquiring or reusing an image."""
    canvas = np.full((_CANVAS_SIZE[1], _CANVAS_SIZE[0], 3), _BG, dtype=np.uint8)
    text_specs: list[_TextSpec] = []
    displayed_at = now or datetime.now().astimezone()

    cv2.rectangle(canvas, (0, 0), (1599, 94), _CARD, -1)
    cv2.line(canvas, (0, 94), (1599, 94), _BORDER, 1)
    cv2.rectangle(canvas, (28, 24), (34, 72), _CYAN, -1)
    text_specs.extend(
        [
            _TextSpec("BMW 零件亮痕检测演示系统", (50, 22), 30, _TEXT),
            _TextSpec("固定相机 · 固定光源 · 单次触发检测", (51, 62), 15, _MUTED),
            _TextSpec(displayed_at.strftime("%Y-%m-%d  %H:%M:%S"), (1568, 25), 18, _TEXT, "ra"),
            _TextSpec(f"相机序列号  {config.camera_serial}", (1568, 58), 15, _MUTED, "ra"),
        ],
    )
    connection_color = _OK if camera_connected else _CYAN
    connection_text = "相机已连接" if camera_connected else "相机未连接"
    cv2.circle(canvas, (1170, 45), 6, connection_color, -1, cv2.LINE_AA)
    text_specs.append(_TextSpec(connection_text, (1186, 34), 16, _TEXT))

    cv2.rectangle(canvas, (28, 112), (950, 790), _CARD, -1)
    cv2.rectangle(canvas, (28, 112), (950, 790), _BORDER, 1)
    cv2.line(canvas, (28, 154), (950, 154), _BORDER, 1)
    text_specs.append(_TextSpec("实时采集图像", (44, 122), 18, _TEXT))
    cv2.rectangle(canvas, (42, 162), (936, 782), _BG, -1)
    cv2.line(canvas, (419, 472), (559, 472), _BORDER, 1)
    cv2.line(canvas, (489, 402), (489, 542), _BORDER, 1)
    cv2.circle(canvas, (489, 472), 46, _BORDER, 1, cv2.LINE_AA)
    text_specs.extend(
        [
            _TextSpec("等待采集图像", (489, 568), 24, _MUTED, "ma"),
            _TextSpec("点击“开始检测”或按空格键", (489, 610), 16, _MUTED, "ma"),
        ],
    )

    cv2.rectangle(canvas, (974, 112), (1572, 380), _CARD, -1)
    cv2.rectangle(canvas, (974, 112), (1572, 380), _BORDER, 1)
    cv2.line(canvas, (974, 154), (1572, 154), _BORDER, 1)
    cv2.rectangle(canvas, (990, 164), (1556, 364), _BG, -1)
    text_specs.extend(
        [
            _TextSpec("亮痕检测区域", (990, 122), 18, _TEXT),
            _TextSpec("检测后显示 ROI 证据", (1273, 246), 18, _MUTED, "mm"),
        ],
    )

    cv2.rectangle(canvas, (974, 396), (1572, 548), _CARD_ALT, -1)
    cv2.rectangle(canvas, (974, 396), (1572, 548), _BORDER, 1)
    cv2.rectangle(canvas, (974, 396), (986, 548), _CYAN, -1)
    text_specs.extend(
        [
            _TextSpec("等待检测", (1012, 417), 36, _CYAN),
            _TextSpec("点击“开始检测”或按空格键开始", (1014, 474), 20, _TEXT),
            _TextSpec("系统就绪", (1546, 418), 14, _MUTED, "ra"),
        ],
    )

    cv2.rectangle(canvas, (974, 564), (1572, 790), _CARD, -1)
    cv2.rectangle(canvas, (974, 564), (1572, 790), _BORDER, 1)
    text_specs.append(_TextSpec("检测指标", (990, 576), 18, _TEXT))
    for index, label in enumerate(("亮痕覆盖率", "连续率", "最大断点", "对比度")):
        row_y = 616 + index * 42
        text_specs.append(_TextSpec(label, (994, row_y), 15, _MUTED))
        text_specs.append(_TextSpec("尚未检测", (1548, row_y), 15, _MUTED, "ra"))
        cv2.rectangle(canvas, (1135, row_y + 5), (1395, row_y + 13), _BG, -1)
        cv2.rectangle(canvas, (1135, row_y + 5), (1395, row_y + 13), _BORDER, 1)

    cv2.line(canvas, (0, 802), (1599, 802), _BORDER, 1)
    cv2.circle(canvas, (34, 842), 5, _CYAN, -1, cv2.LINE_AA)
    text_specs.append(_TextSpec("系统就绪 · 等待检测", (50, 829), 16, _TEXT))
    text_specs.append(_TextSpec("空格键开始检测  ·  Q 退出", (50, 858), 13, _MUTED))
    for action, (bx1, by1, bx2, by2), label in _BUTTONS:
        enabled = action_enabled(action, has_result=False)
        hovered = enabled and action is hovered_action
        if not enabled:
            fill, border, text_color = (27, 31, 34), (45, 51, 56), (91, 96, 100)
        elif action is Action.QUIT:
            fill = (48, 43, 63) if not hovered else (58, 52, 91)
            border, text_color = (_BORDER_ACTIVE if hovered else _BORDER), _TEXT
        else:
            fill = (44, 57, 67) if not hovered else (58, 76, 89)
            border, text_color = (_BORDER_ACTIVE if hovered else _BORDER), _TEXT
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), fill, -1)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), border, 2 if hovered else 1)
        text_specs.append(_TextSpec(label, ((bx1 + bx2) // 2, (by1 + by2) // 2), 17, text_color, "mm"))
    return _apply_text(canvas, text_specs)


def run_gui(
    config: BrightStreakConfig,
    acquire: Callable[[], np.ndarray],
    *,
    output_root: Path | None = None,
    save_screenshot: Path | None = None,
) -> BrightStreakResult | None:
    """Run the local Demo, waiting for explicit inspection input at startup."""
    window = "BMW Bright Streak Demo"
    last_result: BrightStreakResult | None = None
    clicked: list[Action] = []
    hovered: list[Action | None] = [None]

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event == cv2.EVENT_MOUSEMOVE:
            hovered[0] = mouse_to_action(x, y)
        if event == cv2.EVENT_LBUTTONUP:
            action = mouse_to_action(x, y)
            if action is not None:
                clicked.append(action)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1600, 900)
    cv2.setMouseCallback(window, on_mouse)
    dashboard = render_waiting_dashboard(
        config,
        hovered_action=hovered[0],
        camera_connected=True,
    )
    if save_screenshot is not None:
        Path(save_screenshot).parent.mkdir(parents=True, exist_ok=True)
        _write_image(Path(save_screenshot), dashboard)
    cv2.imshow(window, dashboard)
    displayed_hover = hovered[0]
    while True:
        if hovered[0] is not displayed_hover:
            if last_result is None:
                dashboard = render_waiting_dashboard(
                    config,
                    hovered_action=hovered[0],
                    camera_connected=True,
                )
            else:
                dashboard = render_dashboard(
                    image,
                    last_result,
                    config,
                    result_dir,
                    hovered_action=hovered[0],
                    camera_connected=True,
                )
            cv2.imshow(window, dashboard)
            displayed_hover = hovered[0]
        action = clicked.pop(0) if clicked else key_to_action(cv2.waitKey(50))
        if action is None or not action_enabled(action, has_result=last_result is not None):
            continue
        if action is Action.QUIT:
            cv2.destroyWindow(window)
            return last_result
        capture_started = time.perf_counter()
        image = acquire()
        capture_elapsed_ms = (time.perf_counter() - capture_started) * 1000.0
        last_result, result_dir = run_image(
            image,
            config,
            output_root=output_root,
            source_ref=f"camera:{config.camera_serial}",
            capture_elapsed_ms=capture_elapsed_ms,
        )
        dashboard = render_dashboard(
            image,
            last_result,
            config,
            result_dir,
            hovered_action=hovered[0],
            camera_connected=True,
        )
        if save_screenshot is not None:
            _write_image(Path(save_screenshot), dashboard)
        cv2.imshow(window, dashboard)
        displayed_hover = hovered[0]
