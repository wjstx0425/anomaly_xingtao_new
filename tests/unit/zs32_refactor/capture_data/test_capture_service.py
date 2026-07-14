"""Linux-authoritative contract tests for topology-driven capture."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from zs32_inspection.capture import (
    CameraBinding,
    CaptureFrame,
    CapturePlan,
    CaptureRequest,
    CaptureRetakeRequired,
    CaptureRoundPlan,
    CaptureService,
    CaptureSystemError,
    IncompleteCaptureError,
    PartialRoundCaptureError,
    RoundConfirmation,
)
from zs32_inspection.capture.gates import CaptureGateResult
from zs32_inspection.capture.gate_policy import CaptureGateProvenance
from zs32_inspection.domain.identity import Hand, PartIdentity


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="authoritative runtime is Linux only")


def _plan(camera_count: int) -> CapturePlan:
    cameras = tuple(
        CameraBinding(
            slot_id=f"slot{index}",
            serial=f"SERIAL{index}",
            views={"front": f"front_{index}", "back": f"back_{index}"},
        )
        for index in range(camera_count)
    )
    return CapturePlan(
        product="ZS32",
        topology_id=f"zs32-{camera_count}cam",
        topology_sha256="a" * 64,
        rounds=(CaptureRoundPlan("front", "front prompt"), CaptureRoundPlan("back", "back prompt")),
        cameras=cameras,
        required_views=tuple(
            [f"front_{index}" for index in range(camera_count)]
            + [f"back_{index}" for index in range(camera_count)]
        ),
    )


class _Source:
    def __init__(self, *, omit: tuple[str, str] | None = None, wrong_serial: bool = False) -> None:
        self.omit = omit
        self.wrong_serial = wrong_serial

    def capture_round(self, request, round_plan, cameras):
        frames = []
        for device_index, camera in enumerate(cameras):
            if self.omit == (round_plan.round_id, camera.slot_id):
                continue
            serial = "WRONG" if self.wrong_serial and device_index == 0 else camera.serial
            frames.append(
                CaptureFrame(
                    round_id=round_plan.round_id,
                    view_id=camera.views[round_plan.round_id],
                    camera_slot_id=camera.slot_id,
                    camera_serial=serial,
                    device_index=device_index,
                    image_bytes=(
                        b"\x89PNG\r\n\x1a\n"
                        + f"{round_plan.round_id}:{camera.slot_id}".encode()
                    ),
                    width=4024,
                    height=3036,
                    capture_mode="single",
                    exposure=4000.0,
                    gain=0.0,
                    captured_at="2026-07-14T00:00:00Z",
                )
            )
        return frames


class _Coordinator:
    def __init__(self) -> None:
        self.rounds: list[str] = []

    def confirm_round(self, request, round_plan, *, round_index, round_count):
        self.rounds.append(round_plan.round_id)
        return RoundConfirmation(
            round_plan.round_id,
            round_plan.prompt,
            "operator-test",
            "2026-07-14T00:00:00Z",
            "2026-07-14T00:00:01Z",
        )


class _Store:
    def __init__(self) -> None:
        self.complete = None
        self.incomplete = None

    def publish_complete(self, plan, bundle):
        self.complete = bundle
        return Path("/published")

    def publish_incomplete(
        self,
        request,
        plan,
        frames,
        *,
        started_at,
        failure_reason,
        gate_provenance,
        round_confirmations,
    ):
        self.incomplete = (
            request,
            tuple(frames),
            failure_reason,
            gate_provenance,
            tuple(round_confirmations),
        )
        return Path("/quarantine")


class _PassGate:
    def __init__(self, name: str) -> None:
        self.name = name

    def evaluate(self, request, plan, frames):
        return tuple(CaptureGateResult(self.name, True, "", view) for view in plan.required_views)


def _provenance(plan: CapturePlan) -> CaptureGateProvenance:
    return CaptureGateProvenance(
            policy_id="test-policy",
            policy_sha256="1" * 64,
            hand=Hand.RIGHT,
            topology_sha256=plan.topology_sha256,
            acquisition_config_sha256="5" * 64,
            quality_profile_sha256="2" * 64,
            registration_profile_sha256="3" * 64,
            registration_reference_sha256_by_view={view: "4" * 64 for view in plan.required_views},
        )


def _service(source, store, plan, *, quality=None, registration=None, coordinator=None):
    return CaptureService(
        source,
        store,
        gate_provenance=_provenance(plan),
        gates=(quality or _PassGate("quality"), registration or _PassGate("registration")),
        round_coordinator=coordinator or _Coordinator(),
    )


@pytest.mark.parametrize(("camera_count", "view_count"), [(3, 6), (4, 8), (5, 10)])
def test_capture_service_has_no_three_camera_constant(camera_count: int, view_count: int) -> None:
    store = _Store()
    request = CaptureRequest("session", "capture", PartIdentity("part", Hand.RIGHT))
    plan = _plan(camera_count)
    result = _service(_Source(), store, plan).capture(request, plan)
    assert len(result.image_sha256_by_view) == view_count
    assert len(store.complete.gate_results) == view_count * 2
    assert store.complete.domain_capture_set.part == request.part
    assert store.incomplete is None


def test_missing_one_camera_frame_is_quarantined_and_never_published() -> None:
    store = _Store()
    request = CaptureRequest("session", "capture", PartIdentity("part", Hand.RIGHT))
    plan = _plan(3)
    with pytest.raises(IncompleteCaptureError) as raised:
        _service(_Source(omit=("back", "slot2")), store, plan).capture(request, plan)
    assert raised.value.failure_kind == "INVALID_CAPTURE"
    assert store.complete is None
    assert store.incomplete is not None
    assert "returned 2 frames" in store.incomplete[2]
    assert store.incomplete[3].acquisition_config_sha256 == "5" * 64


class _RejectBackRound(_Coordinator):
    def confirm_round(self, request, round_plan, *, round_index, round_count):
        if round_plan.round_id == "back":
            raise CaptureRetakeRequired("operator did not confirm flipped part")
        return super().confirm_round(
            request,
            round_plan,
            round_index=round_index,
            round_count=round_count,
        )


def test_back_round_never_triggers_without_explicit_operator_confirmation() -> None:
    store = _Store()
    request = CaptureRequest("session", "capture", PartIdentity("part", Hand.RIGHT))
    plan = _plan(3)
    with pytest.raises(IncompleteCaptureError) as raised:
        _service(
            _Source(),
            store,
            plan,
            coordinator=_RejectBackRound(),
        ).capture(request, plan)
    assert raised.value.failure_kind == "RETAKE"
    assert {frame.round_id for frame in store.incomplete[1]} == {"front"}
    assert tuple(item.round_id for item in store.incomplete[4]) == ("front",)
    assert store.complete is None


def test_serial_identity_conflict_is_incomplete() -> None:
    store = _Store()
    request = CaptureRequest("session", "capture", PartIdentity("part", Hand.RIGHT))
    plan = _plan(3)
    with pytest.raises(IncompleteCaptureError) as raised:
        _service(_Source(wrong_serial=True), store, plan).capture(request, plan)
    assert raised.value.failure_kind == "INVALID_CAPTURE"
    assert "camera serial mismatch" in store.incomplete[2]


class _FailedQualityGate:
    name = "quality"

    def evaluate(self, request, plan, frames):
        return tuple(
            CaptureGateResult(
                "quality",
                view != plan.required_views[0],
                "" if view != plan.required_views[0] else "blurred",
                view,
            )
            for view in plan.required_views
        )


def test_quality_failure_is_retake_quarantine() -> None:
    store = _Store()
    request = CaptureRequest("session", "capture", PartIdentity("part", Hand.RIGHT))
    plan = _plan(3)
    with pytest.raises(IncompleteCaptureError) as raised:
        _service(_Source(), store, plan, quality=_FailedQualityGate()).capture(request, plan)
    assert raised.value.failure_kind == "RETAKE"
    assert store.complete is None
    assert "RETAKE: capture gate failed" in store.incomplete[2]


def test_missing_required_gate_is_rejected_before_capture() -> None:
    store = _Store()
    plan = _plan(3)
    with pytest.raises(
        CaptureSystemError,
        match="exactly one quality gate and one registration gate",
    ):
        CaptureService(
            _Source(),
            store,
            gate_provenance=_provenance(plan),
            gates=(_PassGate("quality"),),
            round_coordinator=_Coordinator(),
        )
    assert store.complete is None
    assert store.incomplete is None


class _MissingViewGate:
    name = "registration"

    def evaluate(self, request, plan, frames):
        return tuple(
            CaptureGateResult("registration", True, "", view)
            for view in plan.required_views[:-1]
        )


def test_gate_missing_one_required_view_is_system_error() -> None:
    store = _Store()
    request = CaptureRequest("session", "capture", PartIdentity("part", Hand.RIGHT))
    plan = _plan(3)
    with pytest.raises(IncompleteCaptureError) as raised:
        _service(_Source(), store, plan, registration=_MissingViewGate()).capture(request, plan)
    assert raised.value.failure_kind == "SYSTEM_ERROR"
    assert store.complete is None
    assert "evidence is incomplete or duplicated" in store.incomplete[2]


class _DuplicateViewGate:
    name = "registration"

    def evaluate(self, request, plan, frames):
        results = [
            CaptureGateResult("registration", True, "", view)
            for view in plan.required_views
        ]
        results.append(CaptureGateResult("registration", True, "", plan.required_views[0]))
        return tuple(results)


def test_gate_duplicate_view_is_system_error() -> None:
    store = _Store()
    request = CaptureRequest("session", "capture", PartIdentity("part", Hand.RIGHT))
    plan = _plan(3)
    with pytest.raises(IncompleteCaptureError) as raised:
        _service(_Source(), store, plan, registration=_DuplicateViewGate()).capture(request, plan)
    assert raised.value.failure_kind == "SYSTEM_ERROR"
    assert store.complete is None
    assert "evidence is incomplete or duplicated" in store.incomplete[2]


class _RuntimeFailureSource:
    def capture_round(self, request, round_plan, cameras):
        raise RuntimeError("camera SDK timeout")


class _PartialRuntimeFailureSource:
    def capture_round(self, request, round_plan, cameras):
        partial = tuple(_Source().capture_round(request, round_plan, cameras)[:1])
        raise PartialRoundCaptureError("grouped read failed", partial)


@pytest.mark.parametrize(
    "source",
    [_RuntimeFailureSource(), _PartialRuntimeFailureSource()],
)
def test_source_runtime_failure_is_system_error_and_never_invalid_capture(source) -> None:
    store = _Store()
    request = CaptureRequest("session", "capture", PartIdentity("part", Hand.RIGHT))
    plan = _plan(3)
    with pytest.raises(IncompleteCaptureError) as raised:
        _service(source, store, plan).capture(request, plan)
    assert raised.value.failure_kind == "SYSTEM_ERROR"
    assert store.complete is None
    assert store.incomplete[2].startswith("SYSTEM_ERROR:")


class _IncompletePublicationFailureStore(_Store):
    def publish_incomplete(
        self,
        request,
        plan,
        frames,
        *,
        started_at,
        failure_reason,
        gate_provenance,
        round_confirmations,
    ):
        raise PermissionError("quarantine is read-only")


def test_incomplete_publication_failure_is_typed_system_error() -> None:
    request = CaptureRequest("session", "capture", PartIdentity("part", Hand.RIGHT))
    plan = _plan(3)
    with pytest.raises(CaptureSystemError, match="diagnostic publication also failed") as raised:
        _service(
            _Source(omit=("back", "slot2")),
            _IncompletePublicationFailureStore(),
            plan,
        ).capture(request, plan)
    assert raised.value.failure_kind == "SYSTEM_ERROR"


class _CompletePublicationFailureStore(_Store):
    def publish_complete(self, plan, bundle):
        raise PermissionError("complete store is read-only")


def test_complete_publication_failure_is_typed_system_error() -> None:
    request = CaptureRequest("session", "capture", PartIdentity("part", Hand.RIGHT))
    plan = _plan(3)
    with pytest.raises(CaptureSystemError, match="complete capture publication failed") as raised:
        _service(_Source(), _CompletePublicationFailureStore(), plan).capture(request, plan)
    assert raised.value.failure_kind == "SYSTEM_ERROR"
