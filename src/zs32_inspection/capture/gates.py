"""Ports for image-quality and registration RETAKE gates.

Concrete OpenCV or learned implementations live in adapters.  The capture
business service consumes only this deterministic result contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .contracts import CaptureFrame, CapturePlan, CaptureRequest


@dataclass(frozen=True, slots=True)
class CaptureGateResult:
    """One fail-closed quality or registration gate result."""

    gate: str
    passed: bool
    reason: str = ""
    view_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.gate, str) or self.gate not in {"quality", "registration"}:
            raise ValueError(f"unsupported capture gate: {self.gate!r}")
        if not isinstance(self.passed, bool):
            raise ValueError("capture gate passed must be a boolean")
        if not isinstance(self.reason, str):
            raise ValueError("capture gate reason must be a string")
        if self.view_id is not None and (
            not isinstance(self.view_id, str) or not self.view_id.strip()
        ):
            raise ValueError("capture gate view_id must be a non-empty string when present")
        if not self.passed and not self.reason.strip():
            raise ValueError("a failed capture gate must provide a reason")


class CaptureGate(Protocol):
    """Hardware-independent quality/registration evaluator."""

    name: str

    def evaluate(
        self,
        request: CaptureRequest,
        plan: CapturePlan,
        frames: Sequence[CaptureFrame],
    ) -> Sequence[CaptureGateResult]: ...
