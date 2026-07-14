"""Topology-driven capture orchestration with fail-closed publication."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from .contracts import (
    CameraBinding,
    CaptureFrame,
    CapturePlan,
    CaptureRequest,
    CaptureResult,
    CaptureRoundPlan,
    RoundConfirmation,
    ensure_unique_frames,
    utc_now,
)
from .errors import (
    CaptureFailure,
    CaptureRetakeRequired,
    CaptureSystemError,
    InvalidCaptureError,
)
from .gates import CaptureGate, CaptureGateResult
from .gate_policy import CaptureGateProvenance
from .manifests import CaptureBundle, build_capture_rows


class CaptureSource(Protocol):
    """Adapter port implemented by the Hikvision grouped-trigger layer."""

    def capture_round(
        self,
        request: CaptureRequest,
        round_plan: CaptureRoundPlan,
        cameras: Sequence[CameraBinding],
    ) -> Sequence[CaptureFrame]: ...


class RoundCoordinator(Protocol):
    """Require an explicit physical-part/side confirmation before every round."""

    def confirm_round(
        self,
        request: CaptureRequest,
        round_plan: CaptureRoundPlan,
        *,
        round_index: int,
        round_count: int,
    ) -> RoundConfirmation: ...


@dataclass(frozen=True, slots=True)
class PartialRoundCaptureError(CaptureSystemError):
    """Adapter failure that retains frames acquired before a round aborted."""

    reason: str
    partial_frames: tuple[CaptureFrame, ...] = ()

    def __str__(self) -> str:
        return self.reason


class CaptureStore(Protocol):
    """Atomic persistence port used by the capture service."""

    def publish_complete(self, plan: CapturePlan, bundle: CaptureBundle) -> Path: ...

    def publish_incomplete(
        self,
        request: CaptureRequest,
        plan: CapturePlan,
        frames: Sequence[CaptureFrame],
        *,
        started_at: str,
        failure_reason: str,
        gate_provenance: CaptureGateProvenance,
        round_confirmations: Sequence[RoundConfirmation],
    ) -> Path: ...


@dataclass(frozen=True, slots=True)
class IncompleteCaptureError(RuntimeError):
    """A classified capture failure was retained in the quarantine store."""

    capture_set_id: str
    failure_kind: str
    reason: str
    diagnostic_path: str

    def __str__(self) -> str:
        return (
            f"capture set {self.capture_set_id} is incomplete ({self.failure_kind}): "
            f"{self.reason}; "
            f"diagnostic={self.diagnostic_path}"
        )


class CaptureService:
    """Acquire one physical part and publish only a complete capture set."""

    def __init__(
        self,
        source: CaptureSource,
        store: CaptureStore,
        *,
        gates: Sequence[CaptureGate] = (),
        gate_provenance: CaptureGateProvenance,
        round_coordinator: RoundCoordinator,
    ) -> None:
        self._source = source
        self._store = store
        self._gates = tuple(gates)
        if not hasattr(round_coordinator, "confirm_round"):
            raise CaptureSystemError("capture requires an explicit round coordinator")
        self._round_coordinator = round_coordinator
        if not isinstance(gate_provenance, CaptureGateProvenance):
            raise CaptureSystemError("capture requires structured gate policy provenance")
        self._gate_provenance = gate_provenance
        gate_names = tuple(getattr(gate, "name", None) for gate in self._gates)
        if set(gate_names) != {"quality", "registration"} or len(gate_names) != 2:
            raise CaptureSystemError(
                "capture requires exactly one quality gate and one registration gate"
            )

    def capture(self, request: CaptureRequest, plan: CapturePlan) -> CaptureResult:
        """Capture all topology rounds or raise after quarantining partial data."""
        started_at = utc_now()
        if (
            self._gate_provenance.hand.value != request.hand
            or self._gate_provenance.topology_sha256 != plan.topology_sha256
            or set(self._gate_provenance.registration_reference_sha256_by_view)
            != set(plan.required_views)
        ):
            raise CaptureSystemError("capture gate provenance differs from request/topology")
        frames: list[CaptureFrame] = []
        gate_results: list[CaptureGateResult] = []
        round_confirmations: list[RoundConfirmation] = []
        try:
            for round_index, round_plan in enumerate(plan.rounds, start=1):
                try:
                    confirmation = self._round_coordinator.confirm_round(
                        request,
                        round_plan,
                        round_index=round_index,
                        round_count=len(plan.rounds),
                    )
                except CaptureFailure:
                    raise
                except Exception as error:
                    raise CaptureSystemError(
                        f"round confirmation failed for {round_plan.round_id!r}: {error}"
                    ) from error
                if not isinstance(confirmation, RoundConfirmation) or (
                    confirmation.round_id,
                    confirmation.prompt,
                ) != (round_plan.round_id, round_plan.prompt):
                    raise CaptureSystemError(
                        f"round confirmation identity differs for {round_plan.round_id!r}"
                    )
                round_confirmations.append(confirmation)
                try:
                    round_frames = tuple(
                        self._source.capture_round(request, round_plan, plan.cameras)
                    )
                except PartialRoundCaptureError as error:
                    frames.extend(error.partial_frames)
                    raise
                except Exception as error:
                    raise CaptureSystemError(
                        f"capture source runtime failed in round {round_plan.round_id!r}: {error}"
                    ) from error
                self._validate_round(plan, round_plan, round_frames)
                frames.extend(round_frames)
            try:
                ensure_unique_frames(frames)
            except (TypeError, ValueError) as error:
                raise InvalidCaptureError(str(error)) from error
            actual_views = {frame.view_id for frame in frames}
            if actual_views != set(plan.required_views):
                raise InvalidCaptureError(
                    f"capture set views do not match topology: missing="
                    f"{sorted(set(plan.required_views) - actual_views)}, unexpected="
                    f"{sorted(actual_views - set(plan.required_views))}"
                )
            gate_failures = []
            gate_names: set[str] = set()
            for gate in self._gates:
                if gate.name not in {"quality", "registration"}:
                    raise CaptureSystemError(
                        f"unsupported capture gate: {gate.name!r}",
                    )
                if gate.name in gate_names:
                    raise CaptureSystemError(
                        f"duplicate capture gate: {gate.name!r}",
                    )
                gate_names.add(gate.name)
                try:
                    results = tuple(gate.evaluate(request, plan, frames))
                except Exception as error:
                    raise CaptureSystemError(
                        f"capture gate {gate.name!r} runtime failed: {error}",
                    ) from error
                if any(result.gate != gate.name for result in results):
                    raise CaptureSystemError(
                        f"capture gate {gate.name!r} returned conflicting gate identity",
                    )
                view_ids = tuple(result.view_id for result in results)
                if any(view_id is None for view_id in view_ids) or set(view_ids) != set(
                    plan.required_views
                ) or len(view_ids) != len(set(view_ids)):
                    raise CaptureSystemError(
                        f"capture gate {gate.name!r} evidence is incomplete or duplicated; "
                        f"expected={sorted(plan.required_views)}, "
                        f"got={sorted(str(item) for item in view_ids)}",
                    )
                gate_failures.extend(result for result in results if not result.passed)
                gate_results.extend(results)
            if gate_failures:
                details = "; ".join(
                    f"{item.gate}{'/' + item.view_id if item.view_id else ''}: {item.reason}"
                    for item in gate_failures
                )
                raise CaptureRetakeRequired(f"capture gate failed: {details}")
        except Exception as error:
            classified = (
                error
                if isinstance(error, CaptureFailure)
                else CaptureSystemError(f"capture orchestration runtime failed: {error}")
            )
            try:
                diagnostic = self._store.publish_incomplete(
                    request,
                    plan,
                    frames,
                    started_at=started_at,
                    failure_reason=f"{classified.failure_kind}: {classified}",
                    gate_provenance=self._gate_provenance,
                    round_confirmations=round_confirmations,
                )
            except Exception as publication_error:
                raise CaptureSystemError(
                    "capture failed and its diagnostic publication also failed: "
                    f"original={classified.failure_kind}:{classified}; "
                    f"publication={publication_error}"
                ) from publication_error
            raise IncompleteCaptureError(
                request.capture_set_id,
                classified.failure_kind,
                str(classified),
                str(diagnostic),
            ) from classified

        try:
            bundle = build_capture_rows(
                request,
                plan,
                frames,
                gate_results,
                self._gate_provenance,
                round_confirmations,
                started_at=started_at,
            )
        except Exception as error:
            raise CaptureSystemError(f"capture bundle construction failed: {error}") from error
        try:
            published = self._store.publish_complete(plan, bundle)
        except Exception as error:
            raise CaptureSystemError(f"complete capture publication failed: {error}") from error
        return CaptureResult(
            capture_session=request.capture_session,
            capture_set_id=request.capture_set_id,
            part_instance_id=request.part_instance_id,
            published_path=str(published),
            image_sha256_by_view={row.view_id: row.image_sha256 for row in bundle.images},
        )

    @staticmethod
    def _validate_round(
        plan: CapturePlan,
        round_plan: CaptureRoundPlan,
        frames: Sequence[CaptureFrame],
    ) -> None:
        expected_by_slot = {
            camera.slot_id: (camera.serial, camera.views[round_plan.round_id])
            for camera in plan.cameras
        }
        if len(frames) != len(expected_by_slot):
            raise InvalidCaptureError(
                f"round {round_plan.round_id!r} returned {len(frames)} frames; "
                f"expected {len(expected_by_slot)}"
            )
        observed_slots: set[str] = set()
        for frame in frames:
            if frame.round_id != round_plan.round_id:
                raise InvalidCaptureError(
                    f"frame round identity mismatch: expected {round_plan.round_id!r}, "
                    f"got {frame.round_id!r}"
                )
            if frame.camera_slot_id in observed_slots:
                raise InvalidCaptureError(
                    f"round {round_plan.round_id!r} has duplicate camera slot "
                    f"{frame.camera_slot_id!r}"
                )
            observed_slots.add(frame.camera_slot_id)
            try:
                expected_serial, expected_view = expected_by_slot[frame.camera_slot_id]
            except KeyError as error:
                raise InvalidCaptureError(
                    f"unexpected camera slot: {frame.camera_slot_id!r}"
                ) from error
            if frame.camera_serial != expected_serial:
                raise InvalidCaptureError(
                    f"camera serial mismatch for slot {frame.camera_slot_id!r}: "
                    f"expected {expected_serial!r}, got {frame.camera_serial!r}"
                )
            if frame.view_id != expected_view:
                raise InvalidCaptureError(
                    f"view mismatch for round/slot {round_plan.round_id!r}/"
                    f"{frame.camera_slot_id!r}: expected {expected_view!r}, got {frame.view_id!r}"
                )
        if observed_slots != set(expected_by_slot):
            raise InvalidCaptureError(
                f"round {round_plan.round_id!r} camera slots differ from topology"
            )
