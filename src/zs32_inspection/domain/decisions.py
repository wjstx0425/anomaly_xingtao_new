"""Final ZS32 inspection states and fail-closed publication semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .errors import DecisionValidationError
from .identity import require_non_empty


class InspectionStatus(str, Enum):
    """Mutually distinct final inspection statuses."""

    OK = "OK"
    NG_TEMPLATE = "NG_TEMPLATE"
    NG_ANOMALY = "NG_ANOMALY"
    NG_YOLO = "NG_YOLO"
    REVIEW = "REVIEW"
    RETAKE = "RETAKE"
    INVALID_CAPTURE = "INVALID_CAPTURE"
    SYSTEM_ERROR = "SYSTEM_ERROR"


_RELEASABLE_STATUSES = frozenset(
    {
        InspectionStatus.OK,
        InspectionStatus.NG_TEMPLATE,
        InspectionStatus.NG_ANOMALY,
        InspectionStatus.NG_YOLO,
    },
)
_EVIDENCE_STATUSES = frozenset(
    {
        InspectionStatus.NG_TEMPLATE,
        InspectionStatus.NG_ANOMALY,
        InspectionStatus.NG_YOLO,
    },
)


@dataclass(frozen=True, slots=True)
class InspectionDecision:
    """Separate observed evidence, run integrity, review, and released result."""

    inspection_id: str
    evidence_status: InspectionStatus | None
    inspection_status: InspectionStatus
    review_status: str | None
    released_status: InspectionStatus | None
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject unsafe or internally contradictory decision states."""
        try:
            object.__setattr__(self, "inspection_id", require_non_empty(self.inspection_id, "inspection_id"))
            if self.evidence_status is not None:
                object.__setattr__(self, "evidence_status", InspectionStatus(self.evidence_status))
            object.__setattr__(self, "inspection_status", InspectionStatus(self.inspection_status))
            if self.released_status is not None:
                object.__setattr__(self, "released_status", InspectionStatus(self.released_status))
        except ValueError as error:
            raise DecisionValidationError(str(error)) from error
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))
        for reason in self.reason_codes:
            try:
                require_non_empty(reason, "reason_code")
            except ValueError as error:
                raise DecisionValidationError(str(error)) from error
        if self.evidence_status is not None and self.evidence_status not in _EVIDENCE_STATUSES:
            msg = f"evidence_status must be a concrete NG status, got {self.evidence_status.value!r}"
            raise DecisionValidationError(msg)
        if self.released_status is not None and self.released_status not in _RELEASABLE_STATUSES:
            msg = f"released_status cannot be {self.released_status.value!r}"
            raise DecisionValidationError(msg)
        if self.inspection_status in {
            InspectionStatus.SYSTEM_ERROR,
            InspectionStatus.INVALID_CAPTURE,
            InspectionStatus.RETAKE,
            InspectionStatus.REVIEW,
        } and self.released_status is not None:
            msg = f"{self.inspection_status.value} must not publish released_status"
            raise DecisionValidationError(msg)
        if self.released_status is not None and self.released_status is not self.inspection_status:
            msg = "released_status must equal inspection_status for a complete releasable inspection"
            raise DecisionValidationError(msg)
        if self.inspection_status in _RELEASABLE_STATUSES and self.released_status is None:
            raise DecisionValidationError(
                f"complete {self.inspection_status.value} must publish the same released_status"
            )
        if self.inspection_status is InspectionStatus.OK and self.evidence_status is not None:
            msg = "OK cannot coexist with an NG evidence_status"
            raise DecisionValidationError(msg)
        if self.inspection_status in _EVIDENCE_STATUSES and self.evidence_status is not self.inspection_status:
            raise DecisionValidationError(
                "a completed NG inspection must preserve the same evidence_status"
            )
        if self.inspection_status in {InspectionStatus.RETAKE, InspectionStatus.INVALID_CAPTURE}:
            if self.evidence_status is not None:
                raise DecisionValidationError(
                    f"{self.inspection_status.value} cannot carry model/template NG evidence"
                )
        if self.inspection_status is InspectionStatus.REVIEW:
            try:
                object.__setattr__(
                    self,
                    "review_status",
                    require_non_empty(self.review_status, "review_status"),
                )
            except ValueError as error:
                raise DecisionValidationError(str(error)) from error
        elif self.review_status is not None:
            raise DecisionValidationError(
                f"review_status is only legal when inspection_status is REVIEW, got {self.inspection_status.value}"
            )
