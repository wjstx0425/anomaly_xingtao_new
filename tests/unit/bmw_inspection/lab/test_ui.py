"""Regression tests for the Chinese BMW laboratory UI."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from bmw_inspection.lab.contracts import (
    BranchEvidence,
    BranchName,
    BranchStatus,
    CapturedView,
    CaptureSet,
    FinalStatus,
    InspectionResult,
    ViewId,
)
from bmw_inspection.lab.runtime import RuntimeOutcome, RuntimeState
from bmw_inspection.lab.ui import (
    LabUiController,
    UiMode,
    UiPhase,
    UiSnapshot,
    render_dashboard,
    render_yolo_diagnostic_overlay,
    rotate_evidence_for_display,
    status_copy,
    view_frame_status,
)
from bmw_inspection.lab.yolo import DetectionBox, YoloEvidence


def _capture_set() -> CaptureSet:
    captured_at = datetime(2026, 8, 5, 12, 0, 0)
    return CaptureSet(
        capture_set_id="part-001",
        views={
            view_id: CapturedView(
                view_id=view_id,
                camera_serial=f"camera-{index}",
                image=np.full((48, 64), 25 + index * 20, dtype=np.uint8),
                captured_at=captured_at,
            )
            for index, view_id in enumerate(ViewId)
        },
        created_at=captured_at,
    )


def _result(status: FinalStatus = FinalStatus.OK) -> InspectionResult:
    evidence = tuple(
        BranchEvidence(
            branch=branch,
            view_id=view_id,
            status=BranchStatus.PASS,
            required_for_ok=True,
            score=0.1,
            threshold=0.2,
            elapsed_ms=1.5,
            reason="通过",
            model_id="model-v1",
            artifact_paths={},
        )
        for branch in BranchName
        for view_id in ViewId
    )
    return InspectionResult(_capture_set(), evidence, status, "检测完成", True)


class _Runtime:
    def __init__(self, outcome: RuntimeOutcome) -> None:
        self.outcome = outcome
        self.calls = 0

    def inspect(self, capture_set: CaptureSet) -> RuntimeOutcome:
        self.calls += 1
        assert isinstance(capture_set, CaptureSet)
        return self.outcome


def test_controller_is_idle_without_capture_at_startup(tmp_path: Path) -> None:
    calls = {"front": 0, "back": 0}
    result = _result()
    runtime = _Runtime(RuntimeOutcome(RuntimeState.PUBLISHED, result, tmp_path, (), "检测完成"))

    controller = LabUiController(
        capture_front=lambda: calls.__setitem__("front", calls["front"] + 1) or "front",
        capture_back=lambda: calls.__setitem__("back", calls["back"] + 1) or "back",
        assemble_capture=lambda _front, _back: _capture_set(),
        runtime=runtime,
    )

    assert controller.snapshot.phase is UiPhase.IDLE
    assert calls == {"front": 0, "back": 0}
    assert runtime.calls == 0


def test_controller_runs_two_explicit_rounds_and_preserves_previous_label(tmp_path: Path) -> None:
    result = _result()
    runtime = _Runtime(RuntimeOutcome(RuntimeState.PUBLISHED, result, tmp_path, (), "检测完成"))
    controller = LabUiController(
        capture_front=lambda: "front",
        capture_back=lambda: "back",
        assemble_capture=lambda front, back: _capture_set() if (front, back) == ("front", "back") else None,
        runtime=runtime,
    )

    controller.capture_front()
    assert controller.snapshot.phase is UiPhase.WAITING_FLIP
    phases: list[UiPhase] = []
    controller.capture_back_and_inspect(before_inspect=lambda: phases.append(controller.snapshot.phase))
    assert phases == [UiPhase.PROCESSING]
    assert controller.snapshot.phase is UiPhase.RESULT
    assert controller.snapshot.outcome is runtime.outcome
    assert runtime.calls == 1

    controller.retry()
    assert controller.snapshot.phase is UiPhase.IDLE
    assert controller.snapshot.outcome is None
    assert controller.snapshot.previous_outcome is runtime.outcome
    assert controller.snapshot.previous_label == "上次结果"


def test_controller_converts_exceptions_and_reload_required_to_usable_states(tmp_path: Path) -> None:
    reload_runtime = _Runtime(RuntimeOutcome(RuntimeState.RELOAD_REQUIRED, None, None, (), "配置已变化"))
    controller = LabUiController(
        capture_front=lambda: "front",
        capture_back=lambda: "back",
        assemble_capture=lambda _front, _back: _capture_set(),
        runtime=reload_runtime,
    )
    controller.capture_front()
    controller.capture_back_and_inspect()
    assert controller.snapshot.phase is UiPhase.RELOAD_REQUIRED
    controller.retry()
    assert controller.snapshot.phase is UiPhase.IDLE

    failing = LabUiController(
        capture_front=lambda: (_ for _ in ()).throw(RuntimeError("相机断开")),
        capture_back=lambda: None,
        assemble_capture=lambda _front, _back: _capture_set(),
        runtime=reload_runtime,
    )
    failing.capture_front()
    assert failing.snapshot.phase is UiPhase.ERROR
    assert "相机断开" in failing.snapshot.message
    failing.retry()
    assert failing.snapshot.phase is UiPhase.IDLE


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (FinalStatus.OK, "检测通过"),
        (FinalStatus.NG_TEMPLATE, "模板匹配不通过"),
        (FinalStatus.NG_BRIGHT_STREAK, "光痕检测不通过"),
        (FinalStatus.NG_YOLO, "缺陷检测不通过"),
        (FinalStatus.NG_ANOMALY, "异常检测不通过"),
        (FinalStatus.REVIEW, "待复核"),
        (FinalStatus.RETAKE, "请重新拍摄"),
        (FinalStatus.ERROR, "系统异常"),
    ],
)
def test_all_business_outcomes_have_chinese_copy(status: FinalStatus, expected: str) -> None:
    assert status_copy(status).title == expected


def test_render_is_deterministic_1600x900_and_modes_differ() -> None:
    result = _result()
    base = UiSnapshot(
        phase=UiPhase.RESULT,
        message="检测完成",
        outcome=RuntimeOutcome(RuntimeState.PUBLISHED, result, Path("/tmp/run"), (), "检测完成"),
        previous_outcome=None,
        selected_view=ViewId.FRONT_LEFT,
        selected_branch=BranchName.BRIGHT_STREAK,
        experiment_id="bmw-lab-v1",
    )

    presentation = render_dashboard(base, UiMode.PRESENTATION)
    experiment = render_dashboard(base, UiMode.EXPERIMENT)

    assert presentation.shape == (900, 1600, 3)
    assert presentation.dtype == np.uint8
    assert np.array_equal(presentation, render_dashboard(base, UiMode.PRESENTATION))
    assert not np.array_equal(presentation, experiment)
    assert tuple(presentation[700, 100]) == (255, 255, 255)


def test_retake_with_no_branch_evidence_never_draws_pass_frames() -> None:
    capture_set = _capture_set()
    result = InspectionResult(capture_set, (), FinalStatus.RETAKE, "请重新拍摄", False)

    assert {view_frame_status(result, view_id) for view_id in ViewId} == {BranchStatus.REVIEW}


def test_bright_streak_evidence_rotates_clockwise_for_display_only() -> None:
    evidence = np.arange(6, dtype=np.uint8).reshape(2, 3)
    original = evidence.copy()

    rotated = rotate_evidence_for_display(evidence, BranchName.BRIGHT_STREAK)

    assert rotated.tolist() == [[3, 0], [4, 1], [5, 2]]
    assert np.array_equal(evidence, original)
    assert rotate_evidence_for_display(evidence, BranchName.YOLO).shape == evidence.shape


def test_experiment_yolo_overlay_uses_amber_for_candidate_and_red_only_for_final() -> None:
    image = np.zeros((40, 60), dtype=np.uint8)
    candidate = DetectionBox(5, 5, 20, 20, 0.3)
    final = DetectionBox(30, 5, 50, 20, 0.8)
    evidence = YoloEvidence(
        branch=BranchName.YOLO,
        view_id=ViewId.FRONT,
        status=BranchStatus.NG,
        required_for_ok=True,
        score=0.8,
        threshold=0.5,
        elapsed_ms=1.0,
        reason="检测到缺陷",
        model_id="model-v1",
        artifact_paths={},
        candidates=(candidate, final),
        final_boxes=(final,),
        roi_xyxy=(0, 0, 60, 40),
    )

    overlay = render_yolo_diagnostic_overlay(image, evidence)

    assert tuple(overlay[5, 5]) == (0, 117, 197)  # amber in BGR
    assert tuple(overlay[5, 30]) == (43, 0, 228)  # red in BGR
