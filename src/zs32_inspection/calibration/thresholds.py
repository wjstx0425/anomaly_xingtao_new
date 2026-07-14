# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pure threshold fitting for binary template and dual anomaly/YOLO groups."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from zs32_inspection.domain.contracts import TemplateThresholdRecord, ThresholdRecord
from zs32_inspection.domain.evidence import EvidenceBranch
from zs32_inspection.domain.identity import Hand
from zs32_inspection.models.base import ModelContractError, ModelSlot, require_sha256, require_text
from zs32_inspection.template.artifacts import TemplateThresholdFit

from .scoring import GroundTruth, ScoreBranch, ScoreGroup, ScoreRow, rows_for_fit


@dataclass(frozen=True, slots=True)
class FitParameters:
    """Versioned calibration parameters; none are hard-coded release policy."""

    target_defect_recall: float
    normal_quantile: float
    min_normal_parts: int
    min_defect_parts: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.target_defect_recall, bool)
            or not isinstance(self.target_defect_recall, (int, float))
            or not math.isfinite(self.target_defect_recall)
            or not 0 < self.target_defect_recall <= 1
        ):
            raise ModelContractError("target_defect_recall must be in (0, 1]")
        if (
            isinstance(self.normal_quantile, bool)
            or not isinstance(self.normal_quantile, (int, float))
            or not math.isfinite(self.normal_quantile)
            or not 0 < self.normal_quantile <= 1
        ):
            raise ModelContractError("normal_quantile must be in (0, 1]")
        if (
            isinstance(self.min_normal_parts, bool)
            or not isinstance(self.min_normal_parts, int)
            or isinstance(self.min_defect_parts, bool)
            or not isinstance(self.min_defect_parts, int)
            or self.min_normal_parts <= 0
            or self.min_defect_parts <= 0
        ):
            raise ModelContractError("minimum calibration part counts must be positive")

    def to_dict(self) -> dict[str, float | int]:
        """Return parameters for calibration provenance."""
        return {
            "target_defect_recall": self.target_defect_recall,
            "normal_quantile": self.normal_quantile,
            "min_normal_parts": self.min_normal_parts,
            "min_defect_parts": self.min_defect_parts,
        }


@dataclass(frozen=True, slots=True)
class DualThreshold:
    """Low/high thresholds or an explicit insufficient-data record."""

    group: ScoreGroup
    dataset_release_id: str
    calibration_split_id: str
    low: float | None
    high: float | None
    normal_count: int
    defect_count: int
    status: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dataset_release_id",
            require_text(self.dataset_release_id, "threshold.dataset_release_id"),
        )
        object.__setattr__(
            self,
            "calibration_split_id",
            require_text(self.calibration_split_id, "threshold.calibration_split_id"),
        )
        if self.group.branch not in {ScoreBranch.ANOMALY, ScoreBranch.YOLO}:
            raise ModelContractError("dual threshold branch must be anomaly or yolo")
        if (
            isinstance(self.normal_count, bool)
            or not isinstance(self.normal_count, int)
            or isinstance(self.defect_count, bool)
            or not isinstance(self.defect_count, int)
            or self.normal_count < 0
            or self.defect_count < 0
        ):
            raise ModelContractError("threshold counts must be non-negative")
        if self.status == "ok":
            if self.normal_count == 0 or self.defect_count == 0:
                raise ModelContractError("valid dual threshold requires normal and defect parts")
            if self.low is None or self.high is None:
                raise ModelContractError("valid dual threshold must contain low and high")
            if (
                isinstance(self.low, bool)
                or not isinstance(self.low, (int, float))
                or isinstance(self.high, bool)
                or not isinstance(self.high, (int, float))
                or not math.isfinite(self.low)
                or not math.isfinite(self.high)
                or self.low >= self.high
            ):
                raise ModelContractError("dual threshold values must be finite and ordered")
        elif self.status in {"insufficient_data", "non_separable"}:
            if self.low is not None or self.high is not None:
                raise ModelContractError("invalid dual threshold must not contain deployable low/high")
        else:
            raise ModelContractError(f"unsupported dual threshold status: {self.status!r}")

    def to_dict(self) -> dict[str, object]:
        """Serialize one group record."""
        return {
            "dataset_release_id": self.dataset_release_id,
            "calibration_split_id": self.calibration_split_id,
            "hand": self.group.hand,
            "view": self.group.view,
            "branch": self.group.branch.value,
            "model_digest": self.group.model_digest,
            "roi_version": self.group.roi_version,
            "low": self.low,
            "high": self.high,
            "normal_count": self.normal_count,
            "defect_count": self.defect_count,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True, order=True)
class TemplateCalibrationGroup:
    """Exact binary template threshold identity."""

    hand: str
    view: str
    model_digest: str
    roi_version: str
    roi_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "hand", Hand.parse(self.hand).value)
        object.__setattr__(self, "view", require_text(self.view, "template_group.view"))
        object.__setattr__(
            self,
            "model_digest",
            require_sha256(self.model_digest, "template_group.model_digest"),
        )
        object.__setattr__(self, "roi_version", require_text(self.roi_version, "template_group.roi_version"))
        object.__setattr__(self, "roi_digest", require_sha256(self.roi_digest, "template_group.roi_digest"))

    @property
    def slot(self) -> ModelSlot:
        """Return hand/view identity shared with template assets."""
        return ModelSlot(self.hand, self.view)


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    """Return an inclusive nearest-rank quantile."""
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _defect_recall_threshold(values: Sequence[float], target_recall: float) -> float:
    """Return the largest inclusive threshold meeting observed target recall."""
    ordered = sorted(values)
    required_hits = math.ceil(target_recall * len(ordered))
    return ordered[len(ordered) - required_hits]


def _part_scores(rows: Sequence[ScoreRow], label: GroundTruth) -> list[float]:
    """Aggregate repeated captures conservatively to one maximum score per part."""
    by_part: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.ground_truth is label:
            by_part[row.part_instance_id].append(row.score)
    return [max(by_part[part_id]) for part_id in sorted(by_part)]


def fit_dual_thresholds(
    rows: Iterable[ScoreRow],
    *,
    required_groups: Sequence[ScoreGroup],
    calibration_split_id: str,
    parameters: FitParameters,
) -> tuple[DualThreshold, ...]:
    """Fit only calibration-role rows for exact anomaly/YOLO groups.

    Held-out test rows may be present in ``rows`` but are discarded by
    ``rows_for_fit`` before any statistic is computed.
    """
    fit_rows = rows_for_fit(rows, calibration_split_id)
    required = tuple(sorted(set(required_groups)))
    if not required:
        raise ModelContractError("at least one required dual-threshold group is needed")
    if any(group.branch not in {ScoreBranch.ANOMALY, ScoreBranch.YOLO} for group in required):
        raise ModelContractError("required dual groups may contain only anomaly/yolo branches")
    unexpected = sorted({row.group for row in fit_rows if row.branch is not ScoreBranch.TEMPLATE} - set(required))
    if unexpected:
        raise ModelContractError(f"calibration scores contain unexpected model groups: {unexpected}")
    output: list[DualThreshold] = []
    dataset_release_id = fit_rows[0].dataset_release_id
    for group in required:
        group_rows = [row for row in fit_rows if row.group == group]
        normal_scores = _part_scores(group_rows, GroundTruth.NORMAL)
        defect_scores = _part_scores(group_rows, GroundTruth.DEFECT)
        sufficient = (
            len(normal_scores) >= parameters.min_normal_parts
            and len(defect_scores) >= parameters.min_defect_parts
        )
        if sufficient:
            candidate_low = _nearest_rank(normal_scores, parameters.normal_quantile)
            candidate_high = _defect_recall_threshold(defect_scores, parameters.target_defect_recall)
            if candidate_low < candidate_high:
                low = candidate_low
                high = candidate_high
                status = "ok"
            else:
                low = None
                high = None
                status = "non_separable"
        else:
            low = None
            high = None
            status = "insufficient_data"
        output.append(
            DualThreshold(
                group=group,
                dataset_release_id=dataset_release_id,
                calibration_split_id=calibration_split_id,
                low=low,
                high=high,
                normal_count=len(normal_scores),
                defect_count=len(defect_scores),
                status=status,
            ),
        )
    return tuple(output)


def fit_template_thresholds(
    rows: Iterable[ScoreRow],
    *,
    required_groups: Sequence[TemplateCalibrationGroup],
    calibration_split_id: str,
    parameters: FitParameters,
) -> tuple[TemplateThresholdFit, ...]:
    """Fit one binary rejection threshold per template hand/view group."""
    fit_rows = rows_for_fit(rows, calibration_split_id)
    required = tuple(sorted(set(required_groups)))
    if not required:
        raise ModelContractError("at least one required template calibration group is needed")
    required_identities = {
        (group.hand, group.view, group.model_digest, group.roi_version, group.roi_digest)
        for group in required
    }
    observed_identities = {
        (row.hand, row.view, row.model_digest, row.roi_version, row.roi_digest)
        for row in fit_rows
        if row.branch is ScoreBranch.TEMPLATE
    }
    unexpected = sorted(observed_identities - required_identities)
    if unexpected:
        raise ModelContractError(f"calibration scores contain unexpected template groups: {unexpected}")
    output: list[TemplateThresholdFit] = []
    dataset_release_id = fit_rows[0].dataset_release_id
    for group in required:
        group_rows = [
            row
            for row in fit_rows
            if row.branch is ScoreBranch.TEMPLATE
            and row.hand == group.hand
            and row.view == group.view
            and row.model_digest == group.model_digest
            and row.roi_version == group.roi_version
            and row.roi_digest == group.roi_digest
        ]
        normal_scores = _part_scores(group_rows, GroundTruth.NORMAL)
        defect_scores = _part_scores(group_rows, GroundTruth.DEFECT)
        sufficient = (
            len(normal_scores) >= parameters.min_normal_parts
            and len(defect_scores) >= parameters.min_defect_parts
        )
        threshold = (
            _defect_recall_threshold(defect_scores, parameters.target_defect_recall)
            if sufficient
            else None
        )
        output.append(
            TemplateThresholdFit(
                slot=group.slot,
                model_digest=group.model_digest,
                roi_version=group.roi_version,
                roi_digest=group.roi_digest,
                dataset_release_id=dataset_release_id,
                calibration_split_id=calibration_split_id,
                threshold=threshold,
                normal_count=len(normal_scores),
                defect_count=len(defect_scores),
                status="ok" if sufficient else "insufficient_data",
            ),
        )
    return tuple(output)


def to_domain_threshold_record(record: DualThreshold, calibration_sha256: str) -> ThresholdRecord:
    """Promote one valid fit record into the canonical deployment threshold type."""
    if record.status != "ok" or record.low is None or record.high is None:
        raise ModelContractError(f"dual threshold is not deployable: {record.group}")
    return ThresholdRecord(
        hand=Hand.parse(record.group.hand),
        view_id=record.group.view,
        branch=EvidenceBranch(record.group.branch.value),
        low=record.low,
        high=record.high,
        model_sha256=record.group.model_digest,
        roi_config_id=record.group.roi_version,
        calibration_sha256=calibration_sha256,
    )


def to_domain_template_threshold_record(
    record: TemplateThresholdFit,
    calibration_sha256: str,
) -> TemplateThresholdRecord:
    """Promote one valid binary fit into the canonical deployment record."""
    if record.status != "ok" or record.threshold is None:
        raise ModelContractError(f"template threshold is not deployable: {record.slot.key}")
    return TemplateThresholdRecord(
        hand=Hand.parse(record.slot.hand),
        view_id=record.slot.view,
        threshold=record.threshold,
        template_sha256=record.model_digest,
        calibration_sha256=calibration_sha256,
    )
