"""Registration-gate adapter contract; failures require RETAKE."""

from __future__ import annotations

from typing import Protocol, Sequence

from .contracts import CaptureFrame, CapturePlan, CaptureRequest
from .gates import CaptureGateResult


class RegistrationGate(Protocol):
    """Evaluate pose/registration without turning runtime faults into part NG."""

    name: str

    def evaluate(
        self,
        request: CaptureRequest,
        plan: CapturePlan,
        frames: Sequence[CaptureFrame],
    ) -> Sequence[CaptureGateResult]: ...


def registration_failure(view_id: str, reason: str) -> CaptureGateResult:
    """Build a canonical fail-closed per-view registration result."""
    return CaptureGateResult("registration", False, reason, view_id)
