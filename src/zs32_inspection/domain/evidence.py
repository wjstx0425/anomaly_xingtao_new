"""Stable evidence semantics independent of model frameworks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .errors import EvidenceValidationError
from .identity import Hand, require_non_empty, require_sha256


class EvidenceLevel(str, Enum):
    """Calibrated evidence strength used by strict fusion."""

    CLEAR = "CLEAR"
    GRAY = "GRAY"
    STRONG = "STRONG"


class EvidenceBranch(str, Enum):
    """The only second-layer branches permitted by the blueprint."""

    ANOMALY = "anomaly"
    YOLO = "yolo"


class TemplateOutcome(str, Enum):
    """Binary template gate outcome; it deliberately has no review state."""

    PASS = "PASS"
    NG_TEMPLATE = "NG_TEMPLATE"


@dataclass(frozen=True, slots=True)
class ModelEvidence:
    """One calibrated second-layer result for one hand/view/branch."""

    inspection_id: str
    capture_set_id: str
    part_instance_id: str
    hand: Hand
    view_id: str
    branch: EvidenceBranch
    level: EvidenceLevel
    score: float
    model_family: str
    model_sha256: str
    roi_config_id: str
    source_sha256: str
    crop_sha256: str
    threshold_sha256: str

    def __post_init__(self) -> None:
        """Reject incomplete identity, non-finite scores, or invalid hashes."""
        for field_name in (
            "inspection_id",
            "capture_set_id",
            "part_instance_id",
            "view_id",
            "model_family",
            "roi_config_id",
        ):
            try:
                object.__setattr__(self, field_name, require_non_empty(getattr(self, field_name), field_name))
            except ValueError as error:
                raise EvidenceValidationError(str(error)) from error
        try:
            object.__setattr__(self, "hand", Hand.parse(self.hand))
            object.__setattr__(self, "branch", EvidenceBranch(self.branch))
            object.__setattr__(self, "level", EvidenceLevel(self.level))
        except ValueError as error:
            raise EvidenceValidationError(str(error)) from error
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)) or not math.isfinite(self.score):
            msg = f"score must be finite, got {self.score!r}"
            raise EvidenceValidationError(msg)
        object.__setattr__(self, "score", float(self.score))
        for field_name in ("model_sha256", "source_sha256", "crop_sha256", "threshold_sha256"):
            try:
                object.__setattr__(self, field_name, require_sha256(getattr(self, field_name), field_name))
            except ValueError as error:
                raise EvidenceValidationError(str(error)) from error
        if self.branch is EvidenceBranch.YOLO and self.model_family != "yolo":
            msg = "YOLO evidence must identify model_family='yolo'"
            raise EvidenceValidationError(msg)
        if self.branch is EvidenceBranch.ANOMALY and self.model_family not in {
            "patchcore",
            "efficientad",
            "anomalydino",
        }:
            msg = f"unsupported anomaly model_family {self.model_family!r}"
            raise EvidenceValidationError(msg)


@dataclass(frozen=True, slots=True)
class TemplateEvidence:
    """One binary template result for one canonical ROI crop."""

    inspection_id: str
    capture_set_id: str
    part_instance_id: str
    hand: Hand
    view_id: str
    outcome: TemplateOutcome
    score: float
    threshold: float
    template_sha256: str
    threshold_sha256: str
    source_sha256: str
    crop_sha256: str
    roi_config_id: str

    def __post_init__(self) -> None:
        """Validate template evidence without inventing a gray interval."""
        for field_name in (
            "inspection_id",
            "capture_set_id",
            "part_instance_id",
            "view_id",
            "roi_config_id",
        ):
            try:
                object.__setattr__(self, field_name, require_non_empty(getattr(self, field_name), field_name))
            except ValueError as error:
                raise EvidenceValidationError(str(error)) from error
        try:
            object.__setattr__(self, "hand", Hand.parse(self.hand))
            object.__setattr__(self, "outcome", TemplateOutcome(self.outcome))
        except ValueError as error:
            raise EvidenceValidationError(str(error)) from error
        for field_name in ("score", "threshold"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                msg = f"{field_name} must be finite, got {value!r}"
                raise EvidenceValidationError(msg)
            object.__setattr__(self, field_name, float(value))
        for field_name in ("template_sha256", "threshold_sha256", "source_sha256", "crop_sha256"):
            try:
                object.__setattr__(self, field_name, require_sha256(getattr(self, field_name), field_name))
            except ValueError as error:
                raise EvidenceValidationError(str(error)) from error
