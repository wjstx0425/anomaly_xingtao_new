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


@dataclass(frozen=True, slots=True)
class InspectionResult:
    """Complete result consumed by the offline and live dashboards."""

    identity: InspectionIdentity
    views: tuple[ViewResult, ...]
    machine_status: str
    reason: str
    mode: Literal["offline", "live"] = "offline"


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
