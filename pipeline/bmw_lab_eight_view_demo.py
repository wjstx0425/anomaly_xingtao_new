#!/usr/bin/env python3
"""Run the independent Chinese BMW four-camera/eight-view laboratory Demo."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from time import perf_counter_ns

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
from bmw_inspection.lab.live_cycle_timing import LiveCycleTiming, persist_cycle_timing  # noqa: E402
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
_WINDOW_HANDLE = "BMW_EIGHT_VIEW_DEMO"
_WINDOW_TITLE = "BMW 零件八视图检测"


class _GuiAction(str, Enum):
    """Non-rendering commands consumed by the OpenCV loop."""

    CONTINUE = "CONTINUE"
    EXIT = "EXIT"
    RESET = "RESET"
    CAPTURE = "CAPTURE"


@dataclass(slots=True)
class _LiveGuiWork:
    """One live capture moving through the single model worker."""

    generation: int
    front_images: Mapping[str, np.ndarray]
    front_future: Future[object]
    cycle_started_ns: int
    cycle_started_at: str
    front_capture_finished_ns: int
    back_future: Future[object] | None = None
    final_future: Future[object] | None = None
    images: Mapping[str, np.ndarray] | None = None
    source_images: Mapping[str, object] | None = None
    capture_id: str | None = None
    back_capture_started_ns: int | None = None
    back_capture_finished_ns: int | None = None


@dataclass(frozen=True, slots=True)
class _TimedRoundResult:
    """One model round with worker-wall timing marks."""

    value: object
    started_ns: int
    finished_ns: int


@dataclass(frozen=True, slots=True)
class _TimedFinalResult:
    """Final inspection plus its worker-wall timing marks."""

    inspection: object
    finalize_started_ns: int
    finalize_finished_ns: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            REPO_ROOT
            / "configs/bmw/experiments/bmw_eight_view_demo_v6_right_normal_20260814_v1.json"
        ),
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


def _copy_round(images: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    """Give the model worker images that the UI cannot mutate."""
    return {view: image.copy() for view, image in images.items()}


def _milliseconds(nanoseconds: int) -> float:
    return nanoseconds / 1_000_000.0


def _overlap_ms(first_start: int, first_end: int, second_start: int, second_end: int) -> float:
    return _milliseconds(max(0, min(first_end, second_end) - max(first_start, second_start)))


def _inspect_round_timed(
    models: EightViewModelSuite,
    images: Mapping[str, np.ndarray],
    round_name: str,
    clock_ns: Callable[[], int],
) -> _TimedRoundResult:
    """Run one model round and retain its actual worker interval."""
    started_ns = clock_ns()
    value = models.inspect_round(images, round_name)
    return _TimedRoundResult(value=value, started_ns=started_ns, finished_ns=clock_ns())


def _finalize_live(
    models: EightViewModelSuite,
    images: Mapping[str, np.ndarray],
    capture_id: str,
    front_result: _TimedRoundResult,
    back_result: _TimedRoundResult,
    clock_ns: Callable[[], int],
) -> _TimedFinalResult:
    """Finalize both rounds on the single model worker."""
    finalize_started_ns = clock_ns()
    inspection = models.finalize_rounds(
        images,
        front_result.value,
        back_result.value,
        capture_id,
    )
    finalize_finished_ns = clock_ns()
    return _TimedFinalResult(
        inspection=inspection,
        finalize_started_ns=finalize_started_ns,
        finalize_finished_ns=finalize_finished_ns,
    )


def _build_live_cycle_timing(
    work: _LiveGuiWork,
    front: _TimedRoundResult,
    back: _TimedRoundResult,
    final: _TimedFinalResult,
    displayed_ns: int,
    displayed_at: str,
    persist_started_ns: int,
    persist_finished_ns: int,
) -> LiveCycleTiming:
    """Build the final record after the first RESULT frame was submitted."""
    if (
        work.capture_id is None
        or work.back_capture_started_ns is None
        or work.back_capture_finished_ns is None
    ):
        raise RuntimeError("完整周期计时缺少反面采集时间点")
    return LiveCycleTiming(
        capture_id=work.capture_id,
        started_at=work.cycle_started_at,
        displayed_at=displayed_at,
        front_capture_ms=_milliseconds(work.front_capture_finished_ns - work.cycle_started_ns),
        flip_wait_ms=_milliseconds(work.back_capture_started_ns - work.front_capture_finished_ns),
        back_capture_ms=_milliseconds(work.back_capture_finished_ns - work.back_capture_started_ns),
        front_inference_ms=_milliseconds(front.finished_ns - front.started_ns),
        back_inference_ms=_milliseconds(back.finished_ns - back.started_ns),
        finalize_ms=_milliseconds(final.finalize_finished_ns - final.finalize_started_ns),
        persist_ms=_milliseconds(persist_finished_ns - persist_started_ns),
        result_display_ms=_milliseconds(displayed_ns - final.finalize_finished_ns),
        front_overlap_ms=_overlap_ms(
            front.started_ns,
            front.finished_ns,
            work.front_capture_finished_ns,
            work.back_capture_finished_ns,
        ),
        total_cycle_ms=_milliseconds(displayed_ns - work.cycle_started_ns),
    )


def _persist_after_display(
    config: EightViewDemoConfig,
    work: _LiveGuiWork,
    front: _TimedRoundResult,
    back: _TimedRoundResult,
    final: _TimedFinalResult,
    displayed_ns: int,
    displayed_at: str,
    clock_ns: Callable[[], int],
) -> LiveCycleTiming:
    """Save the completed inspection after the operator has seen its result."""
    if work.source_images is None:
        raise RuntimeError("实时检测结果缺少HDR源图")
    persist_started_ns = clock_ns()
    persist_inspection(config, final.inspection, work.source_images)
    persist_finished_ns = clock_ns()
    timing = _build_live_cycle_timing(
        work,
        front,
        back,
        final,
        displayed_ns,
        displayed_at,
        persist_started_ns,
        persist_finished_ns,
    )
    persist_cycle_timing(config.result_root, timing)
    return timing


def _persist_discarded_final(
    config: EightViewDemoConfig,
    work: _LiveGuiWork,
    future: Future[object],
) -> None:
    """Keep the established save behavior when RESET hides an in-flight final result."""
    final = future.result()
    if not isinstance(final, _TimedFinalResult) or work.source_images is None:
        raise RuntimeError("被重置的实时检测结果不完整")
    persist_inspection(config, final.inspection, work.source_images)


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
) -> tuple[int, int] | None:
    """Validate coordinates already mapped to image pixels by OpenCV's Qt backend."""
    if (
        x < 0
        or y < 0
        or x >= _LOGICAL_CANVAS_WIDTH
        or y >= _LOGICAL_CANVAS_HEIGHT
    ):
        return None
    return x, y


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
) -> EightViewUiState:
    """Apply queued Qt image-coordinate clicks to the logical dashboard."""
    while releases:
        x, y = releases.pop(0)
        logical = _logical_canvas_coordinates(x, y)
        if logical is not None:
            state = apply_dashboard_click(state, dashboard_hit_test(*logical))
    return state


def _initialize_gui_window(
    window_handle: str,
    visible_title: str,
    state: EightViewUiState,
    mouse_releases: list[tuple[int, int]],
) -> None:
    """Realize an ASCII-addressable Qt window, then apply its Chinese caption."""
    cv2.namedWindow(window_handle, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_handle, 1440, 810)
    cv2.imshow(window_handle, render_eight_view_screen(state))
    cv2.waitKey(1)
    cv2.setMouseCallback(window_handle, _queue_left_button_release, mouse_releases)
    cv2.setWindowTitle(window_handle, visible_title)


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
    if key == ord(" ") and state.phase is not DemoUiPhase.PROCESSING:
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
    *,
    clock_ns: Callable[[], int] = perf_counter_ns,
    wall_clock: Callable[[], datetime] | None = None,
) -> int:
    wall_clock = (lambda: datetime.now().astimezone()) if wall_clock is None else wall_clock
    title = _WINDOW_HANDLE
    mouse_releases: list[tuple[int, int]] = []
    state = EightViewUiState(experiment_mode=args.experiment_mode)
    _initialize_gui_window(title, _WINDOW_TITLE, state, mouse_releases)
    generation = 0
    work: _LiveGuiWork | None = None
    pending_timing: tuple[
        _LiveGuiWork,
        _TimedRoundResult,
        _TimedRoundResult,
        _TimedFinalResult,
    ] | None = None
    save_future: Future[object] | None = None
    camera_context = nullcontext(None) if offline is not None else FourCameraHdrSession(config.capture_config)
    executor_context = (
        nullcontext(None)
        if offline is not None
        else ThreadPoolExecutor(max_workers=1, thread_name_prefix="bmw-eight-view-model")
    )
    try:
        with camera_context as camera, executor_context as executor:
            while True:
                if save_future is not None and save_future.done():
                    try:
                        timing = save_future.result()
                        if not isinstance(timing, LiveCycleTiming):
                            raise RuntimeError("实时检测保存结果缺少完整周期计时")
                        state = replace(
                            state,
                            message=f"完整周期 {timing.total_cycle_ms / 1000.0:.3f} s，保存完成",
                            cycle_timing=timing,
                        )
                        if args.save_screenshot:
                            _save_dashboard(args.save_screenshot, render_eight_view_dashboard(state))
                    except Exception as error:
                        state = replace(state, phase=DemoUiPhase.ERROR, message=str(error))
                    save_future = None
                if work is not None and work.generation == generation:
                    if work.final_future is not None and work.final_future.done():
                        completed = work
                        work = None
                        try:
                            final_result = completed.final_future.result()
                            front_result = completed.front_future.result()
                            if completed.back_future is None:
                                raise RuntimeError("实时检测结果缺少反面推理")
                            back_result = completed.back_future.result()
                            if not isinstance(final_result, _TimedFinalResult):
                                raise RuntimeError("实时检测结果缺少完整周期计时")
                            if not isinstance(front_result, _TimedRoundResult) or not isinstance(
                                back_result,
                                _TimedRoundResult,
                            ):
                                raise RuntimeError("实时检测结果缺少分轮计时")
                            inspection = final_result.inspection
                            if completed.images is None or completed.source_images is None:
                                raise RuntimeError("实时检测结果缺少八视图或HDR源图")
                            selected_view, selected_branch = preferred_selection(inspection)
                            state = replace(
                                state,
                                phase=DemoUiPhase.RESULT,
                                message="检测完成",
                                images=completed.images,
                                inspection=inspection,
                                source_images=completed.source_images,
                                selected_view=selected_view,
                                selected_branch=selected_branch,
                            )
                            pending_timing = (completed, front_result, back_result, final_result)
                        except Exception as error:
                            state = replace(state, phase=DemoUiPhase.ERROR, message=str(error), inspection=None)
                    elif (
                        work.back_future is not None
                        and work.final_future is None
                        and work.front_future.done()
                        and work.back_future.done()
                    ):
                        try:
                            front_result = work.front_future.result()
                            back_result = work.back_future.result()
                            if work.images is None or work.source_images is None or work.capture_id is None:
                                raise RuntimeError("实时检测任务不完整")
                            work.final_future = executor.submit(
                                _finalize_live,
                                models,
                                work.images,
                                work.capture_id,
                                front_result,
                                back_result,
                                clock_ns,
                            )
                        except Exception as error:
                            work = None
                            state = replace(state, phase=DemoUiPhase.ERROR, message=str(error), inspection=None)
                    elif work.back_future is None and work.front_future.done():
                        error = work.front_future.exception()
                        if error is not None:
                            work = None
                            state = replace(state, phase=DemoUiPhase.ERROR, message=str(error), inspection=None)
                state = _consume_mouse_releases(state, mouse_releases)
                cv2.imshow(title, render_eight_view_screen(state))
                if pending_timing is not None:
                    completed, front_result, back_result, final_result = pending_timing
                    try:
                        displayed_ns = clock_ns()
                        displayed_at = wall_clock().astimezone().isoformat(timespec="milliseconds")
                        visible_cycle_ms = _milliseconds(displayed_ns - completed.cycle_started_ns)
                        state = replace(
                            state,
                            message=f"完整周期 {visible_cycle_ms / 1000.0:.3f} s，结果保存中",
                        )
                        save_future = executor.submit(
                            _persist_after_display,
                            config,
                            completed,
                            front_result,
                            back_result,
                            final_result,
                            displayed_ns,
                            displayed_at,
                            clock_ns,
                        )
                    except Exception as error:
                        state = replace(state, phase=DemoUiPhase.ERROR, message=str(error))
                    pending_timing = None
                key = cv2.waitKey(30) & 0xFF
                state, action = _handle_gui_key(state, key)
                if action is _GuiAction.EXIT:
                    return 0
                if action is _GuiAction.RESET:
                    if work is not None and work.final_future is not None:
                        discarded = work
                        work.final_future.add_done_callback(
                            lambda future, discarded=discarded: _persist_discarded_final(
                                config,
                                discarded,
                                future,
                            )
                        )
                    generation += 1
                    work = None
                    continue
                if action is not _GuiAction.CAPTURE:
                    continue
                if save_future is not None:
                    state = replace(state, message="结果保存中，请稍候")
                    continue
                try:
                    if offline is not None:
                        capture_id, images = offline
                    elif state.phase in {DemoUiPhase.IDLE, DemoUiPhase.RESULT, DemoUiPhase.ERROR}:
                        generation += 1
                        cycle_started_ns = clock_ns()
                        cycle_started_at = wall_clock().astimezone().isoformat(timespec="milliseconds")
                        front_images = camera.capture_round("front")
                        front_capture_finished_ns = clock_ns()
                        work = _LiveGuiWork(
                            generation=generation,
                            front_images=front_images,
                            front_future=executor.submit(
                                _inspect_round_timed,
                                models,
                                _copy_round(front_images),
                                "front",
                                clock_ns,
                            ),
                            cycle_started_ns=cycle_started_ns,
                            cycle_started_at=cycle_started_at,
                            front_capture_finished_ns=front_capture_finished_ns,
                        )
                        state = EightViewUiState(
                            phase=DemoUiPhase.WAITING_FLIP,
                            message="正面推理中，请翻转零件",
                            images=front_images,
                            experiment_mode=args.experiment_mode,
                        )
                        continue
                    elif state.phase is DemoUiPhase.WAITING_FLIP:
                        if work is None:
                            raise RuntimeError("尚未拍摄正面")
                        work.back_capture_started_ns = clock_ns()
                        back = camera.capture_round("back")
                        work.back_capture_finished_ns = clock_ns()
                        images = _merge_rounds(work.front_images, back)
                        capture_id = datetime.now().strftime("bmw_demo_%Y%m%d_%H%M%S")
                        work.images = images
                        work.source_images = dict(camera.last_sources)
                        work.capture_id = capture_id
                        work.back_future = executor.submit(
                            _inspect_round_timed,
                            models,
                            _copy_round(back),
                            "back",
                            clock_ns,
                        )
                        state = replace(
                            state,
                            phase=DemoUiPhase.PROCESSING,
                            message="四类模型正在推理",
                            images=images,
                        )
                        continue
                    else:
                        continue
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
