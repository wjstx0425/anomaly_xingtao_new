#!/usr/bin/env python3
"""Run the independent Chinese BMW four-camera/eight-view laboratory Demo."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime
from enum import Enum
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER  # noqa: E402
from bmw_inspection.lab.eight_view_demo import (  # noqa: E402
    DemoBranch,
    EightViewDemoConfig,
    load_capture_directory,
    load_demo_config,
    load_manifest_sample,
)
from bmw_inspection.lab.eight_view_demo_capture import FourCameraHdrSession  # noqa: E402
from bmw_inspection.lab.eight_view_demo_models import EightViewModelSuite, build_model_suite  # noqa: E402
from bmw_inspection.lab.eight_view_demo_persistence import (  # noqa: E402
    fused_only_sources,
    persist_inspection,
)
from bmw_inspection.lab.eight_view_demo_ui import (  # noqa: E402
    DemoUiPhase,
    DemoUiPage,
    EightViewUiState,
    apply_dashboard_click,
    dashboard_hit_test,
    preferred_selection,
    render_eight_view_dashboard,
    render_eight_view_screen,
    select_branch,
    step_actionable_selection,
    toggle_trusted_ok_mode,
)


_LOGICAL_CANVAS_WIDTH = 1600
_LOGICAL_CANVAS_HEIGHT = 900


class _GuiAction(str, Enum):
    """Non-rendering commands consumed by the OpenCV loop."""

    CONTINUE = "CONTINUE"
    EXIT = "EXIT"
    RESET = "RESET"
    CAPTURE = "CAPTURE"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/bmw/experiments/bmw_eight_view_demo_v1.json",
        help="八视图Demo配置。",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--sample-id", help="从准备清单加载一个完整八视图样本。")
    source.add_argument("--capture-set", type=Path, help="包含八张<view>图片的离线目录。")
    parser.add_argument("--experiment-mode", action="store_true", help="显示分数、阈值和耗时并允许切换证据。")
    parser.add_argument("--no-gui", action="store_true", help="只运行一次并输出JSON摘要。")
    parser.add_argument("--save-screenshot", type=Path, help="检测完成后保存1600x900界面截图。")
    return parser


def _offline_images(args: argparse.Namespace, config: EightViewDemoConfig) -> tuple[str, Mapping[str, np.ndarray]] | None:
    if args.sample_id:
        return args.sample_id, load_manifest_sample(config.prepared_manifest, args.sample_id)
    if args.capture_set:
        path = Path(args.capture_set).expanduser().resolve()
        return path.name, load_capture_directory(path)
    return None


def _merge_rounds(front: Mapping[str, np.ndarray], back: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    merged = {view: front[view] if view in front else back[view] for view in VIEW_ORDER}
    return merged


def _save_dashboard(path: Path, dashboard: np.ndarray) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), dashboard):
        raise RuntimeError(f"无法保存界面截图：{target}")


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _persist_result(config: object, inspection: object, source_images: Mapping[str, object] | None) -> None:
    """Persist live HDR sources, or explicitly adapt an offline fused-only sample."""
    sources = fused_only_sources(inspection.images) if source_images is None else source_images
    persist_inspection(config, inspection, sources)


def _handle_trusted_ok_shortcut(state: EightViewUiState, key: int) -> EightViewUiState:
    """Handle O/o without rerunning or mutating any model result."""
    if key not in {ord("o"), ord("O")}:
        return state
    return toggle_trusted_ok_mode(state)


def _logical_canvas_coordinates(
    x: int,
    y: int,
    display_width: int,
    display_height: int,
) -> tuple[int, int] | None:
    """Convert one displayed-window click into the fixed 1600x900 canvas domain."""
    if (
        display_width <= 0
        or display_height <= 0
        or x < 0
        or y < 0
        or x >= display_width
        or y >= display_height
    ):
        return None
    return (
        x * _LOGICAL_CANVAS_WIDTH // display_width,
        y * _LOGICAL_CANVAS_HEIGHT // display_height,
    )


def _queue_left_button_release(
    event: int,
    x: int,
    y: int,
    _flags: int,
    releases: list[tuple[int, int]],
) -> None:
    """Queue a mouse release; state transitions stay in the main OpenCV loop."""
    if event == cv2.EVENT_LBUTTONUP:
        releases.append((x, y))


def _consume_mouse_releases(
    state: EightViewUiState,
    releases: list[tuple[int, int]],
    display_width: int,
    display_height: int,
) -> EightViewUiState:
    """Apply queued dashboard clicks using the current displayed image dimensions."""
    while releases:
        x, y = releases.pop(0)
        logical = _logical_canvas_coordinates(x, y, display_width, display_height)
        if logical is not None:
            state = apply_dashboard_click(state, dashboard_hit_test(*logical))
    return state


def _initialize_gui_window(
    title: str,
    state: EightViewUiState,
    mouse_releases: list[tuple[int, int]],
) -> None:
    """Realize the Qt window before registering its mouse callback."""
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1440, 810)
    cv2.imshow(title, render_eight_view_screen(state))
    cv2.waitKey(1)
    cv2.setMouseCallback(title, _queue_left_button_release, mouse_releases)


def _handle_gui_key(state: EightViewUiState, key: int) -> tuple[EightViewUiState, _GuiAction]:
    """Apply page-aware shortcuts without capturing or rerunning inspection work."""
    if key in {ord("q"), ord("Q")}:
        return state, _GuiAction.EXIT
    if key in {ord("r"), ord("R")}:
        return EightViewUiState(experiment_mode=state.experiment_mode), _GuiAction.RESET
    if state.page is DemoUiPage.DETAIL:
        if key == 27:
            return replace(state, page=DemoUiPage.DASHBOARD), _GuiAction.CONTINUE
        return state, _GuiAction.CONTINUE
    if key == 27:
        return state, _GuiAction.CONTINUE
    toggled = _handle_trusted_ok_shortcut(state, key)
    if toggled is not state:
        return toggled, _GuiAction.CONTINUE
    if state.experiment_mode and ord("1") <= key <= ord("8"):
        return replace(state, selected_view=VIEW_ORDER[key - ord("1")]), _GuiAction.CONTINUE
    if state.experiment_mode and key in {ord("t"), ord("T"), ord("l"), ord("L"), ord("y"), ord("Y"), ord("e"), ord("E")}:
        branch = {
            ord("t"): DemoBranch.TEMPLATE,
            ord("l"): DemoBranch.BRIGHT_STREAK,
            ord("y"): DemoBranch.YOLO,
            ord("e"): DemoBranch.EFFICIENTAD,
        }[ord(chr(key).lower())]
        return select_branch(state, branch), _GuiAction.CONTINUE
    if state.inspection is not None and key in {ord("n"), ord("N"), ord("p"), ord("P")}:
        selected_view, selected_branch = step_actionable_selection(
            state.inspection,
            state.selected_view,
            state.selected_branch,
            1 if key in {ord("n"), ord("N")} else -1,
        )
        return replace(state, selected_view=selected_view, selected_branch=selected_branch), _GuiAction.CONTINUE
    if key == ord(" "):
        return state, _GuiAction.CAPTURE
    return state, _GuiAction.CONTINUE


def _summary(inspection: object) -> dict[str, object]:
    rows = inspection.results
    return {
        "capture_id": inspection.capture_id,
        "final_status": inspection.final_status.value,
        "elapsed_ms": round(inspection.elapsed_ms, 3),
        "checks": len(rows),
        "counts": {
            status: sum(row.status.value == status for row in rows)
            for status in ("PASS", "NG", "ERROR", "SKIPPED")
        },
        "failures": [
            {
                "view": row.view_id,
                "branch": row.branch.value,
                "status": row.status.value,
                "score": row.score,
                "threshold": row.threshold,
                "reason": row.reason,
                "details": _jsonable(row.details),
            }
            for row in rows
            if row.status.value in {"NG", "ERROR"}
        ],
    }


def _run_no_gui(
    args: argparse.Namespace,
    config: EightViewDemoConfig,
    models: EightViewModelSuite,
    offline: tuple[str, Mapping[str, np.ndarray]] | None,
) -> int:
    source_images: Mapping[str, object] | None = None
    if offline is not None:
        capture_id, images = offline
    else:
        with FourCameraHdrSession(config.capture_config) as camera:
            input("固定零件正面后按回车拍摄：")
            front = camera.capture_round("front")
            input("翻转同一零件并固定，按回车拍摄反面：")
            back = camera.capture_round("back")
            source_images = camera.last_sources
        capture_id = datetime.now().strftime("bmw_demo_%Y%m%d_%H%M%S")
        images = _merge_rounds(front, back)
    inspection = models.inspect(images, capture_id=capture_id)
    _persist_result(config, inspection, source_images)
    state = EightViewUiState(
        phase=DemoUiPhase.RESULT,
        message="检测完成",
        images=images,
        inspection=inspection,
        source_images=fused_only_sources(images) if source_images is None else source_images,
        experiment_mode=args.experiment_mode,
    )
    selected_view, selected_branch = preferred_selection(inspection)
    state = replace(state, selected_view=selected_view, selected_branch=selected_branch)
    if args.save_screenshot:
        _save_dashboard(args.save_screenshot, render_eight_view_dashboard(state))
    print(json.dumps(_summary(inspection), ensure_ascii=False, indent=2))
    return 0 if inspection.final_status.value == "OK" else 1


def _run_gui(
    args: argparse.Namespace,
    config: EightViewDemoConfig,
    models: EightViewModelSuite,
    offline: tuple[str, Mapping[str, np.ndarray]] | None,
) -> int:
    title = "BMW 零件八视图检测"
    mouse_releases: list[tuple[int, int]] = []
    state = EightViewUiState(experiment_mode=args.experiment_mode)
    _initialize_gui_window(title, state, mouse_releases)
    front: Mapping[str, np.ndarray] | None = None
    camera_context = nullcontext(None) if offline is not None else FourCameraHdrSession(config.capture_config)
    try:
        with camera_context as camera:
            while True:
                _, _, display_width, display_height = cv2.getWindowImageRect(title)
                state = _consume_mouse_releases(state, mouse_releases, display_width, display_height)
                cv2.imshow(title, render_eight_view_screen(state))
                key = cv2.waitKey(30) & 0xFF
                state, action = _handle_gui_key(state, key)
                if action is _GuiAction.EXIT:
                    return 0
                if action is _GuiAction.RESET:
                    front = None
                    continue
                if action is not _GuiAction.CAPTURE:
                    continue
                try:
                    if offline is not None:
                        capture_id, images = offline
                    elif state.phase in {DemoUiPhase.IDLE, DemoUiPhase.RESULT, DemoUiPhase.ERROR}:
                        front = camera.capture_round("front")
                        state = EightViewUiState(
                            phase=DemoUiPhase.WAITING_FLIP,
                            message="正面拍摄完成，请翻转零件",
                            images=front,
                            experiment_mode=args.experiment_mode,
                        )
                        continue
                    else:
                        if front is None:
                            raise RuntimeError("尚未拍摄正面")
                        back = camera.capture_round("back")
                        images = _merge_rounds(front, back)
                        capture_id = datetime.now().strftime("bmw_demo_%Y%m%d_%H%M%S")
                    state = replace(state, phase=DemoUiPhase.PROCESSING, message="四类模型正在推理", images=images)
                    cv2.imshow(title, render_eight_view_screen(state))
                    cv2.waitKey(1)
                    inspection = models.inspect(images, capture_id=capture_id)
                    source_images = (
                        fused_only_sources(images)
                        if offline is not None
                        else camera.last_sources
                    )
                    persist_inspection(config, inspection, source_images)
                    selected_view, selected_branch = preferred_selection(inspection)
                    state = replace(
                        state,
                        phase=DemoUiPhase.RESULT,
                        message="检测完成",
                        images=images,
                        inspection=inspection,
                        source_images=source_images,
                        selected_view=selected_view,
                        selected_branch=selected_branch,
                    )
                    if args.save_screenshot:
                        _save_dashboard(args.save_screenshot, render_eight_view_dashboard(state))
                except Exception as error:
                    state = replace(state, phase=DemoUiPhase.ERROR, message=str(error), inspection=None)
    finally:
        cv2.destroyAllWindows()


def main() -> int:
    args = build_parser().parse_args()
    config = load_demo_config(args.config)
    offline = _offline_images(args, config)
    print("正在加载 Template、光痕、YOLO 和八个 EfficientAD 模型……", flush=True)
    models = build_model_suite(config, status_callback=lambda message: print(message, flush=True))
    print("模型加载完成。", flush=True)
    if args.no_gui:
        return _run_no_gui(args, config, models, offline)
    return _run_gui(args, config, models, offline)


if __name__ == "__main__":
    raise SystemExit(main())
