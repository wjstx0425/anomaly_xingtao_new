"""Quality-gate adapter contract; failures require RETAKE."""

from __future__ import annotations

from typing import Protocol, Sequence

from .contracts import CaptureFrame, CapturePlan, CaptureRequest
from .gates import CaptureGateResult


class QualityGate(Protocol):
    """Evaluate blur, clipping, darkness, and other recapturable image faults."""

    name: str

    def evaluate(
        self,
        request: CaptureRequest,
        plan: CapturePlan,
        frames: Sequence[CaptureFrame],
    ) -> Sequence[CaptureGateResult]: ...


def quality_failure(view_id: str, reason: str) -> CaptureGateResult:
    """Build a canonical fail-closed per-view quality result."""
    return CaptureGateResult("quality", False, reason, view_id)
