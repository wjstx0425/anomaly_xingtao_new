#!/usr/bin/env python3
"""Run the Chinese BMW six-view laboratory inspection UI."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import replace
from enum import Enum
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bmw_inspection.lab.capture import LabCameraSession, build_capture_set, load_capture_set  # noqa: E402
from bmw_inspection.lab.config import LabExperimentConfig, load_experiment_config  # noqa: E402
from bmw_inspection.lab.contracts import BranchName, BranchStatus, CaptureSet, FinalStatus, ViewId  # noqa: E402
from bmw_inspection.lab.runtime import RuntimeState, build_lab_runtime  # noqa: E402
from bmw_inspection.lab.ui import (  # noqa: E402
    LabUiController,
    UiMode,
    UiPhase,
    UiSnapshot,
    render_dashboard,
    status_copy,
)


class CaptureState(str, Enum):
    """Terminal-mode progress labels for the two explicit capture rounds."""

    WAITING_FRONT = "WAITING_FRONT"
    CAPTURE_FRONT = "CAPTURE_FRONT"
    WAITING_FLIP = "WAITING_FLIP"
    CAPTURE_BACK = "CAPTURE_BACK"
    READY_FOR_INSPECTION = "READY_FOR_INSPECTION"


_FRONT_VIEWS = (ViewId.FRONT, ViewId.FRONT_LEFT, ViewId.FRONT_RIGHT)
_BACK_VIEWS = (ViewId.BACK, ViewId.BACK_LEFT, ViewId.BACK_RIGHT)


def build_parser() -> argparse.ArgumentParser:
    """Build the intentionally small laboratory UI interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/bmw/experiments/bmw_lab_v1.json",
        help="BMW 实验配置 JSON。",
    )
    parser.add_argument(
        "--capture-set",
        type=Path,
        help="包含六张 front/front_left/front_right/back/back_left/back_right 图片的离线目录。",
    )
    parser.add_argument(
        "--mode",
        choices=tuple(mode.value for mode in UiMode),
        default=UiMode.PRESENTATION.value,
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="不打开窗口，仅执行一次检测并输出结果。",
    )
    parser.add_argument("--save-screenshot", type=Path, help="保存一张 1600x900 中文界面截图。")
    parser.add_argument("--output-root", type=Path, help="覆盖本次运行的结果目录。")
    return parser


def _validate_capture_set_dimensions(capture_set: CaptureSet, config: LabExperimentConfig) -> None:
    for captured in capture_set.views.values():
        height, width = captured.image.shape[:2]
        if (width, height) != (config.capture.image_width, config.capture.image_height):
            raise ValueError(
                f"视角 {captured.view_id.value} 的尺寸为 {width}x{height}，"
                f"配置要求 {config.capture.image_width}x{config.capture.image_height}"
            )


def _capture_live_terminal(config: LabExperimentConfig) -> CaptureSet:
    """Keep terminal acquisition explicit when no GUI is requested."""
    print("等待拍摄正面")
    with LabCameraSession(config) as session:
        input("固定零件正面后按回车拍摄：")
        print("正在拍摄正面")
        front = session.capture_round("front")
        print("等待翻转零件")
        input("翻转同一零件并固定后按回车拍摄：")
        print("正在拍摄反面")
        back = session.capture_round("back")
    print("六视图采集完成")
    return build_capture_set(front, back)


def _offline_rounds(capture_set: CaptureSet) -> tuple[dict[ViewId, object], dict[ViewId, object]]:
    front = {view_id: capture_set.views[view_id].image for view_id in _FRONT_VIEWS}
    back = {view_id: capture_set.views[view_id].image for view_id in _BACK_VIEWS}
    return front, back


def _save_screenshot(path: Path, dashboard: object) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), dashboard):
        raise RuntimeError(f"无法保存界面截图：{target}")


def _artifact_filenames(branch: BranchName) -> tuple[str, ...]:
    return {
        BranchName.TEMPLATE: ("overlay.png", "difference.png"),
        BranchName.BRIGHT_STREAK: ("roi.png",),
        BranchName.YOLO: ("overlay.png",),
        BranchName.PATCHCORE: ("overlay.png", "heatmap.png"),
    }[branch]


def _load_artifact(run_dir: Path, branch: BranchName, view_id: ViewId) -> np.ndarray | None:
    branch_dir = run_dir / "evidence" / branch.value / view_id.value
    for filename in _artifact_filenames(branch):
        artifact = branch_dir / filename
        if not artifact.is_file():
            continue
        image = cv2.imread(str(artifact), cv2.IMREAD_UNCHANGED)
        if image is not None:
            return image
    return None


def _with_selected_evidence(snapshot: UiSnapshot, mode: UiMode) -> UiSnapshot:
    """Load a published display artifact without changing detector coordinates or decisions."""
    outcome = snapshot.outcome
    if outcome is None or outcome.run_dir is None:
        return snapshot
    inspection = outcome.inspection
    selected_branch = snapshot.selected_branch
    selected_view = snapshot.selected_view
    if inspection is not None and mode is UiMode.PRESENTATION:
        preferred_branch = {
            FinalStatus.NG_TEMPLATE: BranchName.TEMPLATE,
            FinalStatus.NG_BRIGHT_STREAK: BranchName.BRIGHT_STREAK,
            FinalStatus.NG_YOLO: BranchName.YOLO,
            FinalStatus.NG_ANOMALY: BranchName.PATCHCORE,
        }.get(inspection.final_status)
        triggered_rows = [
            row
            for row in inspection.evidence
            if row.status in {BranchStatus.NG, BranchStatus.ERROR, BranchStatus.REVIEW}
            and (preferred_branch is None or row.branch is preferred_branch)
        ]
        ranked = sorted(
            triggered_rows,
            key=lambda row: (
                tuple(BranchName).index(row.branch),
                tuple(ViewId).index(row.view_id),
            ),
        )
        for row in ranked:
            if row.status is BranchStatus.SKIPPED:
                continue
            artifact = _load_artifact(outcome.run_dir, row.branch, row.view_id)
            if artifact is not None:
                return replace(
                    snapshot,
                    selected_branch=row.branch,
                    selected_view=row.view_id,
                    evidence_image=artifact,
                )
        if ranked:
            return replace(
                snapshot,
                selected_branch=ranked[0].branch,
                selected_view=ranked[0].view_id,
                evidence_image=None,
            )
        if inspection.final_status is not FinalStatus.OK:
            return replace(snapshot, evidence_image=None)
        pass_rows = sorted(
            (row for row in inspection.evidence if row.status is BranchStatus.PASS),
            key=lambda row: (
                row.branch is not BranchName.BRIGHT_STREAK,
                tuple(BranchName).index(row.branch),
                tuple(ViewId).index(row.view_id),
            ),
        )
        for row in pass_rows:
            artifact = _load_artifact(outcome.run_dir, row.branch, row.view_id)
            if artifact is not None:
                return replace(
                    snapshot,
                    selected_branch=row.branch,
                    selected_view=row.view_id,
                    evidence_image=artifact,
                )
    artifact = _load_artifact(outcome.run_dir, selected_branch, selected_view)
    return replace(snapshot, evidence_image=artifact)


def _run_once(
    *,
    config: LabExperimentConfig,
    capture_set: CaptureSet,
    mode: UiMode,
    screenshot: Path | None,
    config_loader: Callable[[], LabExperimentConfig],
) -> int:
    runtime = build_lab_runtime(config, config_loader=config_loader)
    outcome = runtime.inspect(capture_set)
    phase = UiPhase.RELOAD_REQUIRED if outcome.state is RuntimeState.RELOAD_REQUIRED else UiPhase.RESULT
    controller = LabUiController(
        capture_front=lambda: None,
        capture_back=lambda: None,
        assemble_capture=lambda _front, _back: capture_set,
        runtime=runtime,
        experiment_id=config.experiment_id,
    )
    snapshot = replace(controller.snapshot, phase=phase, outcome=outcome, message=outcome.reason)
    dashboard = render_dashboard(_with_selected_evidence(snapshot, mode), mode)
    if screenshot is not None:
        _save_screenshot(screenshot, dashboard)
        print(f"界面截图：{Path(screenshot).expanduser().resolve()}")
    if outcome.inspection is None:
        print("检测状态：需要重新加载模型")
        return 3
    print(f"检测结果：{status_copy(outcome.inspection.final_status).title}")
    print(f"证据目录：{outcome.run_dir}")
    return 0


def _run_gui(
    *,
    config: LabExperimentConfig,
    mode: UiMode,
    capture_set: CaptureSet | None,
    screenshot: Path | None,
    config_loader: Callable[[], LabExperimentConfig],
) -> int:
    """Own the camera context around the complete HighGUI lifecycle."""
    if capture_set is None:
        while True:
            runtime = build_lab_runtime(config, config_loader=config_loader)
            with LabCameraSession(config) as session:
                status = _run_gui_loop(
                    config=config,
                    mode=mode,
                    screenshot=screenshot,
                    config_loader=config_loader,
                    runtime=runtime,
                    capture_front=lambda: session.capture_round("front"),
                    capture_back=lambda: session.capture_round("back"),
                    restart_on_reload=True,
                )
            if status != 10:
                return status
            config = config_loader()
    runtime = build_lab_runtime(config, config_loader=config_loader)
    front, back = _offline_rounds(capture_set)
    return _run_gui_loop(
        config=config,
        mode=mode,
        screenshot=screenshot,
        config_loader=config_loader,
        runtime=runtime,
        capture_front=lambda: front,
        capture_back=lambda: back,
        restart_on_reload=False,
    )


def _run_gui_loop(
    *,
    config: LabExperimentConfig,
    mode: UiMode,
    screenshot: Path | None,
    config_loader: Callable[[], LabExperimentConfig],
    runtime: object,
    capture_front: Callable[[], object],
    capture_back: Callable[[], object],
    restart_on_reload: bool,
) -> int:
    """Run the recoverable event loop while an optional outer camera context is alive."""
    window = "BMW 六视图检测"
    window_created = False
    restart_requested = False
    controller = LabUiController(
        capture_front=capture_front,
        capture_back=capture_back,
        assemble_capture=build_capture_set,
        runtime=runtime,
        experiment_id=config.experiment_id,
    )

    def show_current(wait_ms: int = 1) -> None:
        dashboard = render_dashboard(_with_selected_evidence(controller.snapshot, mode), mode)
        cv2.imshow(window, dashboard)
        cv2.waitKey(wait_ms)

    def reload_controller() -> None:
        nonlocal config, runtime, controller
        config = config_loader()
        runtime = build_lab_runtime(config, config_loader=config_loader)
        controller = LabUiController(
            capture_front=capture_front,
            capture_back=capture_back,
            assemble_capture=build_capture_set,
            runtime=runtime,
            experiment_id=config.experiment_id,
        )

    def fallback_dashboard() -> np.ndarray:
        dashboard = np.full((900, 1600, 3), 247, dtype=np.uint8)
        cv2.putText(
            dashboard,
            "UI ERROR  |  R: RETRY  |  Q: QUIT",
            (90, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (143, 0, 191),
            3,
            cv2.LINE_AA,
        )
        return dashboard

    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        window_created = True
        cv2.resizeWindow(window, 1600, 900)
        while True:
            try:
                dashboard = render_dashboard(_with_selected_evidence(controller.snapshot, mode), mode)
            except Exception as error:
                controller.report_error(error)
                dashboard = fallback_dashboard()
            cv2.imshow(window, dashboard)
            key = cv2.waitKey(30) & 0xFF
            if key in {ord("q"), ord("Q"), 27}:
                break
            try:
                if key == ord(" "):
                    if controller.snapshot.phase is UiPhase.WAITING_FLIP:
                        controller.capture_back_and_inspect(before_inspect=show_current)
                    elif controller.snapshot.phase in {UiPhase.IDLE, UiPhase.ERROR, UiPhase.RESULT}:
                        controller.capture_front()
                elif key in {ord("r"), ord("R")}:
                    if controller.snapshot.phase is UiPhase.RELOAD_REQUIRED:
                        if restart_on_reload:
                            restart_requested = True
                            break
                        reload_controller()
                    else:
                        controller.retry()
                elif key in {ord("m"), ord("M")} and mode is UiMode.EXPERIMENT:
                    if restart_on_reload:
                        restart_requested = True
                        break
                    reload_controller()
                elif ord("1") <= key <= ord("6"):
                    controller.select(tuple(ViewId)[key - ord("1")], controller.snapshot.selected_branch)
                elif mode is UiMode.EXPERIMENT and chr(key).lower() in {"t", "l", "y", "p"}:
                    branch = {
                        "t": BranchName.TEMPLATE,
                        "l": BranchName.BRIGHT_STREAK,
                        "y": BranchName.YOLO,
                        "p": BranchName.PATCHCORE,
                    }[chr(key).lower()]
                    controller.select(controller.snapshot.selected_view, branch)
            except Exception as error:
                controller.report_error(error)
        if screenshot is not None:
            _save_screenshot(
                screenshot,
                render_dashboard(_with_selected_evidence(controller.snapshot, mode), mode),
            )
    finally:
        if window_created:
            try:
                cv2.destroyWindow(window)
            except cv2.error:
                pass
    return 10 if restart_requested else 0


def main(argv: list[str] | None = None) -> int:
    """Load the profile and run an offline or live Chinese inspection UI."""
    args = build_parser().parse_args(argv)
    try:
        config = load_experiment_config(args.config)
        if args.output_root is not None:
            config = replace(config, result_root=args.output_root.expanduser().resolve())
            config_loader = lambda: replace(
                load_experiment_config(config.path),
                result_root=config.result_root,
            )
        else:
            config_loader = lambda: load_experiment_config(config.path)
        capture_set = load_capture_set(args.capture_set) if args.capture_set is not None else None
        if capture_set is not None:
            _validate_capture_set_dimensions(capture_set, config)
        mode = UiMode(args.mode)
        if args.no_gui:
            capture_set = capture_set or _capture_live_terminal(config)
            return _run_once(
                config=config,
                capture_set=capture_set,
                mode=mode,
                screenshot=args.save_screenshot,
                config_loader=config_loader,
            )
        return _run_gui(
            config=config,
            mode=mode,
            capture_set=capture_set,
            screenshot=args.save_screenshot,
            config_loader=config_loader,
        )
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"BMW 检测启动失败：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
