"""Immutable public result contracts for BMW six-view laboratory inspection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


class ViewId(str, Enum):
    """The fixed physical viewpoints used by the BMW laboratory fixture."""

    FRONT = "front"
    FRONT_LEFT = "front_left"
    FRONT_RIGHT = "front_right"
    BACK = "back"
    BACK_LEFT = "back_left"
    BACK_RIGHT = "back_right"


class BranchName(str, Enum):
    """Inspection branches that can contribute evidence to a final result."""

    TEMPLATE = "template"
    BRIGHT_STREAK = "bright_streak"
    YOLO = "yolo"
    PATCHCORE = "patchcore"


class BranchStatus(str, Enum):
    """Status of one branch for one view."""

    PASS = "PASS"
    NG = "NG"
    SKIPPED = "SKIPPED"
    REVIEW = "REVIEW"
    ERROR = "ERROR"


class FinalStatus(str, Enum):
    """Business-facing inspection outcomes."""

    OK = "OK"
    NG_TEMPLATE = "NG_TEMPLATE"
    NG_BRIGHT_STREAK = "NG_BRIGHT_STREAK"
    NG_YOLO = "NG_YOLO"
    NG_ANOMALY = "NG_ANOMALY"
    REVIEW = "REVIEW"
    RETAKE = "RETAKE"
    ERROR = "ERROR"


def _non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _finite_or_none(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite or None")
    return float(value)


@dataclass(frozen=True, slots=True)
class CapturedView:
    """One image acquired from its explicitly bound camera serial."""

    view_id: ViewId
    camera_serial: str
    image: Any
    captured_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.view_id, ViewId):
            raise TypeError("view_id must be ViewId")
        _non_empty_string(self.camera_serial, "camera_serial")
        if self.image is None:
            raise ValueError("image must not be None")
        if not isinstance(self.captured_at, datetime):
            raise TypeError("captured_at must be datetime")


@dataclass(frozen=True, slots=True)
class CaptureSet:
    """The exact six views of one physical part acquired in two manual rounds."""

    capture_set_id: str
    views: Mapping[ViewId, CapturedView]
    created_at: datetime

    def __post_init__(self) -> None:
        _non_empty_string(self.capture_set_id, "capture_set_id")
        if not isinstance(self.created_at, datetime):
            raise TypeError("created_at must be datetime")
        if not isinstance(self.views, Mapping):
            raise TypeError("views must be a mapping")
        if set(self.views) != set(ViewId):
            raise ValueError("views must contain exactly the six required ViewId values")
        copied: dict[ViewId, CapturedView] = {}
        for view_id, captured in self.views.items():
            if not isinstance(view_id, ViewId) or not isinstance(captured, CapturedView):
                raise TypeError("views must map ViewId to CapturedView")
            if captured.view_id != view_id:
                raise ValueError("captured view_id must match its views key")
            if not isinstance(captured.image, np.ndarray):
                raise TypeError("captured image must be a numpy array")
            owned_image = captured.image.copy()
            owned_image.flags.writeable = False
            copied[view_id] = CapturedView(
                view_id=captured.view_id,
                camera_serial=captured.camera_serial,
                image=owned_image,
                captured_at=captured.captured_at,
            )
        object.__setattr__(self, "views", MappingProxyType(copied))


@dataclass(frozen=True, slots=True)
class BranchEvidence:
    """Structured branch evidence kept separate from the fused business result."""

    branch: BranchName
    view_id: ViewId
    status: BranchStatus
    required_for_ok: bool
    score: float | None
    threshold: float | None
    elapsed_ms: float
    reason: str
    model_id: str | None
    artifact_paths: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.branch, BranchName):
            raise TypeError("branch must be BranchName")
        if not isinstance(self.view_id, ViewId):
            raise TypeError("view_id must be ViewId")
        if not isinstance(self.status, BranchStatus):
            raise TypeError("status must be BranchStatus")
        if not isinstance(self.required_for_ok, bool):
            raise TypeError("required_for_ok must be bool")
        _finite_or_none(self.score, "score")
        _finite_or_none(self.threshold, "threshold")
        elapsed = _finite_or_none(self.elapsed_ms, "elapsed_ms")
        assert elapsed is not None
        if elapsed < 0:
            raise ValueError("elapsed_ms must be non-negative")
        _non_empty_string(self.reason, "reason")
        if self.model_id is not None:
            _non_empty_string(self.model_id, "model_id")
        if not isinstance(self.artifact_paths, Mapping):
            raise TypeError("artifact_paths must be a mapping")
        artifacts: dict[str, str] = {}
        for key, value in self.artifact_paths.items():
            artifacts[_non_empty_string(key, "artifact path key")] = _non_empty_string(value, "artifact path value")
        object.__setattr__(self, "artifact_paths", MappingProxyType(artifacts))


def validate_final_status(status: FinalStatus, required_complete: bool) -> FinalStatus:
    """Reject an OK result unless every required branch completed successfully."""
    if not isinstance(status, FinalStatus):
        raise TypeError("status must be FinalStatus")
    if not isinstance(required_complete, bool):
        raise TypeError("required_complete must be bool")
    if status is FinalStatus.OK and not required_complete:
        raise ValueError("required branch is disabled, skipped, or incomplete; cannot publish OK")
    return status


@dataclass(frozen=True, slots=True)
class InspectionResult:
    """One immutable six-view inspection outcome and its branch evidence."""

    capture_set: CaptureSet
    evidence: tuple[BranchEvidence, ...]
    final_status: FinalStatus
    reason: str
    required_complete: bool

    def __post_init__(self) -> None:
        if not isinstance(self.capture_set, CaptureSet):
            raise TypeError("capture_set must be CaptureSet")
        if not isinstance(self.evidence, tuple) or not all(isinstance(item, BranchEvidence) for item in self.evidence):
            raise TypeError("evidence must be a tuple of BranchEvidence")
        validate_final_status(self.final_status, self.required_complete)
        _non_empty_string(self.reason, "reason")
