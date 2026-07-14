"""Linux console boundary for explicit multi-round part positioning."""

from __future__ import annotations

import select
import sys
from dataclasses import dataclass

from .contracts import (
    CaptureRequest,
    CaptureRoundPlan,
    RoundConfirmation,
    require_identifier,
    utc_now,
)
from .errors import CaptureRetakeRequired, CaptureSystemError


@dataclass(frozen=True, slots=True)
class ConsoleRoundCoordinator:
    """Require an exact operator token before each camera trigger round."""

    operator_id: str
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operator_id",
            require_identifier(self.operator_id, "operator_id"),
        )
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or self.timeout_seconds <= 0
        ):
            raise ValueError("round confirmation timeout_seconds must be positive")
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))

    def confirm_round(
        self,
        request: CaptureRequest,
        round_plan: CaptureRoundPlan,
        *,
        round_index: int,
        round_count: int,
    ) -> RoundConfirmation:
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            raise CaptureSystemError(
                "interactive round confirmation requires a foreground TTY"
            )
        expected = f"CONFIRM {round_plan.round_id}"
        prompted_at = utc_now()
        print(
            f"\n[{round_index}/{round_count}] part={request.part_instance_id} "
            f"hand={request.hand}: {round_plan.prompt}\n"
            f"Type exactly '{expected}' within {self.timeout_seconds:g}s, or type CANCEL:",
            flush=True,
        )
        readable, _, _ = select.select([sys.stdin], [], [], self.timeout_seconds)
        if not readable:
            raise CaptureRetakeRequired(
                f"operator confirmation timed out for round {round_plan.round_id!r}"
            )
        response = sys.stdin.readline()
        if response == "":
            raise CaptureRetakeRequired(
                f"operator confirmation input closed for round {round_plan.round_id!r}"
            )
        response = response.strip()
        if response == "CANCEL":
            raise CaptureRetakeRequired(
                f"operator cancelled round {round_plan.round_id!r}"
            )
        if response != expected:
            raise CaptureRetakeRequired(
                f"operator confirmation token was invalid for round {round_plan.round_id!r}"
            )
        return RoundConfirmation(
            round_id=round_plan.round_id,
            prompt=round_plan.prompt,
            confirmed_by=self.operator_id,
            prompted_at=prompted_at,
            confirmed_at=utc_now(),
        )


__all__ = ["ConsoleRoundCoordinator"]
