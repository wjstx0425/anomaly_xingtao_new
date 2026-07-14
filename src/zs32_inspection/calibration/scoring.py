# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Validated score rows and score-to-level pure functions for ZS32."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

from zs32_inspection.domain.contracts import ThresholdRecord
from zs32_inspection.domain.evidence import EvidenceBranch, EvidenceLevel, ModelEvidence
from zs32_inspection.domain.identity import Hand
from zs32_inspection.models.base import ModelContractError, RawModelScore, require_sha256, require_text


class SplitRole(str, Enum):
    """Semantic split roles; test rows are never legal fitter inputs."""

    CALIBRATION = "calibration"
    TEST = "test"


class ScoreBranch(str, Enum):
    """The only score branches accepted by the redesigned calibration layer."""

    TEMPLATE = "template"
    ANOMALY = "anomaly"
    YOLO = "yolo"


class GroundTruth(str, Enum):
    """Physical-part label for threshold calibration and held-out evaluation."""

    NORMAL = "normal"
    DEFECT = "defect"


@dataclass(frozen=True, slots=True, order=True)
class ScoreGroup:
    """Exact threshold key from the blueprint."""

    hand: str
    view: str
    branch: ScoreBranch
    model_digest: str
    roi_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "hand", Hand.parse(self.hand).value)
        object.__setattr__(self, "view", require_text(self.view, "group.view"))
        object.__setattr__(self, "branch", ScoreBranch(self.branch))
        object.__setattr__(self, "hand", Hand.parse(self.hand).value)
        object.__setattr__(self, "model_digest", require_sha256(self.model_digest, "group.model_digest"))
        object.__setattr__(self, "roi_version", require_text(self.roi_version, "group.roi_version"))

    def __str__(self) -> str:
        """Return a human-readable stable group identity."""
        return f"{self.hand}/{self.view}/{self.branch.value}/{self.model_digest}/{self.roi_version}"


@dataclass(frozen=True, slots=True)
class ScoreRow:
    """One raw, higher-is-riskier score with complete split and model identity."""

    dataset_release_id: str
    part_instance_id: str
    capture_set_id: str
    hand: str
    view: str
    branch: ScoreBranch
    score: float
    ground_truth: GroundTruth
    split_role: SplitRole
    split_id: str
    model_digest: str
    roi_version: str
    roi_digest: str
    part_ground_truth: GroundTruth | None = None

    def __post_init__(self) -> None:
        for field in (
            "dataset_release_id",
            "part_instance_id",
            "capture_set_id",
            "hand",
            "view",
            "split_id",
            "roi_version",
        ):
            object.__setattr__(self, field, require_text(getattr(self, field), f"score.{field}"))
        object.__setattr__(self, "branch", ScoreBranch(self.branch))
        object.__setattr__(self, "ground_truth", GroundTruth(self.ground_truth))
        object.__setattr__(
            self,
            "part_ground_truth",
            self.ground_truth
            if self.part_ground_truth is None
            else GroundTruth(self.part_ground_truth),
        )
        object.__setattr__(self, "split_role", SplitRole(self.split_role))
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)) or not math.isfinite(self.score):
            raise ModelContractError("score.score must be finite")
        object.__setattr__(self, "model_digest", require_sha256(self.model_digest, "score.model_digest"))
        object.__setattr__(self, "roi_digest", require_sha256(self.roi_digest, "score.roi_digest"))

    @property
    def group(self) -> ScoreGroup:
        """Return the exact threshold lookup key."""
        return ScoreGroup(self.hand, self.view, self.branch, self.model_digest, self.roi_version)


def validate_part_split_isolation(rows: Iterable[ScoreRow]) -> tuple[ScoreRow, ...]:
    """Reject physical-part leakage and inconsistent labels across all views/repeats."""
    materialized = tuple(rows)
    if not materialized:
        raise ModelContractError("score rows must not be empty")
    identity_by_part: dict[str, tuple[str, SplitRole, str, GroundTruth, str]] = {}
    part_by_capture: dict[str, str] = {}
    roi_digest_by_version: dict[str, str] = {}
    for row in materialized:
        identity = (
            row.dataset_release_id,
            row.split_role,
            row.split_id,
            row.part_ground_truth,
            row.hand,
        )
        previous = identity_by_part.setdefault(row.part_instance_id, identity)
        if previous != identity:
            raise ModelContractError(
                f"physical part {row.part_instance_id!r} leaks across release/split/label/hand identities",
            )
        previous_part = part_by_capture.setdefault(row.capture_set_id, row.part_instance_id)
        if previous_part != row.part_instance_id:
            raise ModelContractError(
                f"capture set {row.capture_set_id!r} maps to multiple physical parts",
            )
        previous_roi_digest = roi_digest_by_version.setdefault(row.roi_version, row.roi_digest)
        if previous_roi_digest != row.roi_digest:
            raise ModelContractError(f"ROI version {row.roi_version!r} maps to multiple digests")
    return materialized


def rows_for_fit(rows: Iterable[ScoreRow], calibration_split_id: str) -> tuple[ScoreRow, ...]:
    """Return calibration-role rows only; held-out test data cannot influence fitting."""
    split_id = require_text(calibration_split_id, "calibration_split_id")
    materialized = validate_part_split_isolation(rows)
    selected = tuple(
        row
        for row in materialized
        if row.split_role is SplitRole.CALIBRATION and row.split_id == split_id
    )
    if not selected:
        raise ModelContractError(f"calibration split is missing: {split_id}")
    if len({row.dataset_release_id for row in selected}) != 1:
        raise ModelContractError("calibration rows must belong to exactly one dataset release")
    return selected


def rows_for_test(rows: Iterable[ScoreRow], test_split_id: str) -> tuple[ScoreRow, ...]:
    """Return held-out test-role rows only."""
    split_id = require_text(test_split_id, "test_split_id")
    materialized = validate_part_split_isolation(rows)
    selected = tuple(
        row
        for row in materialized
        if row.split_role is SplitRole.TEST and row.split_id == split_id
    )
    if not selected:
        raise ModelContractError(f"held-out test split is missing: {split_id}")
    if len({row.dataset_release_id for row in selected}) != 1:
        raise ModelContractError("held-out test rows must belong to exactly one dataset release")
    return selected


def classify_dual(score: float, low: float, high: float) -> EvidenceLevel:
    """Classify a finite higher-is-riskier score with inclusive boundaries."""
    values = (score, low, high)
    if (
        any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values)
        or not all(math.isfinite(value) for value in values)
        or low >= high
    ):
        raise ModelContractError("dual thresholds require finite score and ordered low/high")
    if score >= high:
        return EvidenceLevel.STRONG
    if score >= low:
        return EvidenceLevel.GRAY
    return EvidenceLevel.CLEAR


def to_domain_model_evidence(
    raw: RawModelScore,
    *,
    threshold: ThresholdRecord,
) -> ModelEvidence:
    """Convert one raw adapter score into the canonical calibrated evidence type."""
    try:
        branch = EvidenceBranch(raw.branch)
    except ValueError as error:
        raise ModelContractError(f"raw model branch must be anomaly or yolo, got {raw.branch!r}") from error
    if (
        raw.hand != threshold.hand.value
        or raw.view != threshold.view_id
        or branch is not threshold.branch
        or raw.model_digest != threshold.model_sha256
        or raw.roi_config_id != threshold.roi_config_id
    ):
        raise ModelContractError("raw score identity does not match the canonical deployment threshold")
    return ModelEvidence(
        inspection_id=raw.inspection_id,
        capture_set_id=raw.capture_set_id,
        part_instance_id=raw.part_instance_id,
        hand=Hand.parse(raw.hand),
        view_id=raw.view,
        branch=branch,
        level=classify_dual(raw.score, threshold.low, threshold.high),
        score=raw.score,
        model_family=raw.model_family,
        model_sha256=raw.model_digest,
        roi_config_id=raw.roi_config_id,
        source_sha256=raw.source_sha256,
        crop_sha256=raw.crop_sha256,
        threshold_sha256=threshold.calibration_sha256,
    )


def require_unique_part_group_scores(rows: Sequence[ScoreRow]) -> None:
    """Reject duplicate scores for the same physical part and exact group."""
    seen: set[tuple[str, ScoreGroup]] = set()
    for row in rows:
        key = (row.part_instance_id, row.group)
        if key in seen:
            raise ModelContractError(f"duplicate score for physical part/group: {row.part_instance_id}, {row.group}")
        seen.add(key)
