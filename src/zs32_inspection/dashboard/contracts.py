"""Stable contracts shared by the ZS32 inspection dashboard and runner."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Mapping


VIEW_ORDER = (
    "front",
    "front_left",
    "front_right",
    "front_secondary",
    "back",
    "back_left",
    "back_right",
    "back_secondary",
)
MODELED_VIEWS = (
    "front",
    "front_left",
    "front_right",
    "back",
    "back_left",
    "back_right",
)


def _require_non_empty(field_name: str, value: object) -> None:
    """Reject missing or blank identity fields at runtime."""
    if not isinstance(value, str) or not value.strip():
        msg = f"{field_name} must be a non-empty string"
        raise ValueError(msg)


class EvidenceLayer(StrEnum):
    """Dashboard evidence layers in their stable presentation order."""

    FUSION = "fusion"
    ORIGINAL = "original"
    PATCHCORE = "patchcore"
    YOLO = "yolo"
    TEMPLATE = "template"


class BranchState(StrEnum):
    """Execution availability of one evidence branch."""

    AVAILABLE = "available"
    SKIPPED = "skipped"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class InspectionIdentity:
    """Identity shared by all eight views of one inspection."""

    part_id: str
    capture_session: str
    group_id: str
    hand: Literal["right"]

    def __post_init__(self) -> None:
        """Validate the stable right-hand identity contract at runtime."""
        for field_name in ("part_id", "capture_session", "group_id"):
            _require_non_empty(field_name, getattr(self, field_name))
        if self.hand != "right":
            msg = "hand must be 'right'"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class BranchEvidence:
    """Evidence emitted by one branch for one view."""

    branch: str
    state: BranchState
    status: str
    score: float | None
    reason: str
    evidence_path: Path | None = None
    mask_path: Path | None = None
    mask_source: str | None = None
    roi_xyxy: tuple[int, int, int, int] | None = None
    detections: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class ViewResult:
    """Source and branch results for one position in the eight-view grid."""

    view: str
    source_path: Path
    source_sha256: str
    source_shape: tuple[int, int]
    model_supported: bool
    branches: Mapping[str, BranchEvidence]
    capture: Mapping[str, str]

    def __post_init__(self) -> None:
        """Validate view membership and its fixed model-support policy."""
        if self.view not in VIEW_ORDER:
            msg = f"view must be one of VIEW_ORDER, got {self.view!r}"
            raise ValueError(msg)
        expected_support = self.view in MODELED_VIEWS
        if self.model_supported is not expected_support:
            msg = f"model_supported must be {expected_support} for view {self.view!r}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class InspectionResult:
    """Complete result consumed by the offline and live dashboards."""

    identity: InspectionIdentity
    views: tuple[ViewResult, ...]
    machine_status: str
    reason: str
    mode: Literal["offline", "live"] = "offline"

    def __post_init__(self) -> None:
        """Require the exact closed, ordered eight-view result set."""
        actual_order = tuple(view.view for view in self.views)
        if actual_order != VIEW_ORDER:
            msg = f"views must match VIEW_ORDER exactly, got {actual_order!r}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ProgressRecord:
    """Atomic progress message published by a live inspection runner."""

    part_id: str
    capture_session: str | None
    state: str
    message: str
    timestamp: str
    confirmation_id: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ConfirmationCommand:
    """Single-use operator confirmation for one capture round."""

    action: Literal["confirm_round"]
    round: Literal["front", "back"]
    confirmation_id: str
    part_id: str

    def __post_init__(self) -> None:
        """Validate the single supported command shape at runtime."""
        if self.action != "confirm_round":
            msg = "action must be 'confirm_round'"
            raise ValueError(msg)
        if self.round not in ("front", "back"):
            msg = "round must be 'front' or 'back'"
            raise ValueError(msg)
        _require_non_empty("confirmation_id", self.confirmation_id)
        _require_non_empty("part_id", self.part_id)
