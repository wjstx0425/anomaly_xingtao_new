# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Held-out part metrics and canonical calibration artifact provenance."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from zs32_inspection.models.base import ModelContractError, canonical_sha256, require_sha256, require_text
from zs32_inspection.template.artifacts import TemplateThresholdFit

from .scoring import GroundTruth, ScoreBranch, ScoreGroup, ScoreRow, classify_dual, rows_for_test
from .thresholds import DualThreshold, FitParameters, TemplateCalibrationGroup


@dataclass(frozen=True, slots=True)
class CalibrationProvenance:
    """All version identities required to reproduce a threshold artifact."""

    recipe_digest: str
    profile_digest: str
    topology_digest: str
    roi_digest: str
    dataset_release_id: str
    dataset_manifest_digest: str
    calibration_split_id: str
    test_split_id: str
    model_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in (
            "recipe_digest",
            "profile_digest",
            "topology_digest",
            "roi_digest",
            "dataset_manifest_digest",
        ):
            object.__setattr__(self, field, require_sha256(getattr(self, field), f"provenance.{field}"))
        for field in ("dataset_release_id", "calibration_split_id", "test_split_id"):
            object.__setattr__(self, field, require_text(getattr(self, field), f"provenance.{field}"))
        if self.calibration_split_id == self.test_split_id:
            raise ModelContractError("calibration and held-out test split IDs must be distinct")
        digests = tuple(sorted({require_sha256(value, "provenance.model_digest") for value in self.model_digests}))
        if not digests:
            raise ModelContractError("calibration provenance requires model digests")
        object.__setattr__(self, "model_digests", digests)

    def to_dict(self) -> dict[str, object]:
        """Return canonical provenance content."""
        return {
            "recipe_digest": self.recipe_digest,
            "profile_digest": self.profile_digest,
            "topology_digest": self.topology_digest,
            "roi_digest": self.roi_digest,
            "dataset_release_id": self.dataset_release_id,
            "dataset_manifest_digest": self.dataset_manifest_digest,
            "calibration_split_id": self.calibration_split_id,
            "test_split_id": self.test_split_id,
            "model_digests": list(self.model_digests),
        }


@dataclass(frozen=True, slots=True)
class HeldOutPartMetrics:
    """Strict part-level metrics from the independent test split."""

    dataset_release_id: str
    test_split_id: str
    part_count: int
    normal_part_count: int
    defect_part_count: int
    clear_count: int
    gray_count: int
    strong_count: int
    template_ng_count: int
    incomplete_count: int
    defect_escape_count: int
    normal_reject_count: int
    defect_escape_rate: float | None
    normal_reject_rate: float | None
    review_rate: float | None
    evaluation_complete: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dataset_release_id",
            require_text(self.dataset_release_id, "metrics.dataset_release_id"),
        )
        object.__setattr__(self, "test_split_id", require_text(self.test_split_id, "metrics.test_split_id"))
        count_fields = (
            "part_count",
            "normal_part_count",
            "defect_part_count",
            "clear_count",
            "gray_count",
            "strong_count",
            "template_ng_count",
            "incomplete_count",
            "defect_escape_count",
            "normal_reject_count",
        )
        for field in count_fields:
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ModelContractError(f"metrics.{field} must be a non-negative integer")
        if self.part_count != self.normal_part_count + self.defect_part_count:
            raise ModelContractError("metrics part_count must equal normal_part_count + defect_part_count")
        outcome_count = (
            self.clear_count
            + self.gray_count
            + self.strong_count
            + self.template_ng_count
            + self.incomplete_count
        )
        if self.part_count != outcome_count:
            raise ModelContractError("metrics part_count must equal the sum of mutually exclusive part outcomes")
        if self.defect_escape_count > self.defect_part_count:
            raise ModelContractError("metrics defect_escape_count exceeds defect_part_count")
        if self.normal_reject_count > self.normal_part_count:
            raise ModelContractError("metrics normal_reject_count exceeds normal_part_count")
        expected_rates = {
            "defect_escape_rate": (
                None if self.defect_part_count == 0 else self.defect_escape_count / self.defect_part_count
            ),
            "normal_reject_rate": (
                None if self.normal_part_count == 0 else self.normal_reject_count / self.normal_part_count
            ),
            "review_rate": None if self.part_count == 0 else self.gray_count / self.part_count,
        }
        for field, expected in expected_rates.items():
            actual = getattr(self, field)
            if expected is None:
                if actual is not None:
                    raise ModelContractError(f"metrics.{field} must be null when its denominator is zero")
            elif (
                actual is None
                or isinstance(actual, bool)
                or not isinstance(actual, (int, float))
                or not math.isfinite(actual)
                or not 0 <= actual <= 1
                or not math.isclose(float(actual), expected, rel_tol=0, abs_tol=1e-12)
            ):
                raise ModelContractError(f"metrics.{field} contradicts its count fields")
        expected_complete = self.incomplete_count == 0 and self.normal_part_count > 0 and self.defect_part_count > 0
        if not isinstance(self.evaluation_complete, bool) or self.evaluation_complete is not expected_complete:
            raise ModelContractError("metrics.evaluation_complete contradicts completeness and label counts")

    def to_dict(self) -> dict[str, object]:
        """Return canonical part-level metrics."""
        return {
            "dataset_release_id": self.dataset_release_id,
            "test_split_id": self.test_split_id,
            "part_count": self.part_count,
            "normal_part_count": self.normal_part_count,
            "defect_part_count": self.defect_part_count,
            "clear_count": self.clear_count,
            "gray_count": self.gray_count,
            "strong_count": self.strong_count,
            "template_ng_count": self.template_ng_count,
            "incomplete_count": self.incomplete_count,
            "defect_escape_count": self.defect_escape_count,
            "normal_reject_count": self.normal_reject_count,
            "defect_escape_rate": self.defect_escape_rate,
            "normal_reject_rate": self.normal_reject_rate,
            "review_rate": self.review_rate,
            "evaluation_complete": self.evaluation_complete,
        }


def _max_score_by_part_group(rows: Sequence[ScoreRow]) -> dict[tuple[str, ScoreGroup], float]:
    values: dict[tuple[str, ScoreGroup], float] = {}
    for row in rows:
        key = (row.part_instance_id, row.group)
        values[key] = max(row.score, values.get(key, row.score))
    return values


def evaluate_heldout(
    rows: Iterable[ScoreRow],
    *,
    dual_thresholds: Sequence[DualThreshold],
    template_thresholds: Sequence[TemplateThresholdFit],
    required_dual_groups: Sequence[ScoreGroup],
    required_template_groups: Sequence[TemplateCalibrationGroup],
    test_split_id: str,
) -> HeldOutPartMetrics:
    """Evaluate fixed thresholds without fitting or mutating them."""
    test_rows = rows_for_test(rows, test_split_id)
    labels = {row.part_instance_id: row.part_ground_truth for row in test_rows}
    hands = {row.part_instance_id: row.hand for row in test_rows}
    scores = _max_score_by_part_group(test_rows)
    dual_by_group = {record.group: record for record in dual_thresholds}
    if len(dual_by_group) != len(dual_thresholds):
        raise ModelContractError("dual threshold groups must be unique")
    template_by_identity = {
        (record.slot.hand, record.slot.view, record.model_digest, record.roi_version, record.roi_digest): record
        for record in template_thresholds
    }
    if len(template_by_identity) != len(template_thresholds):
        raise ModelContractError("template threshold groups must be unique")

    outcomes: dict[str, str] = {}
    for part_id in sorted(labels):
        hand = hands[part_id]
        incomplete = False
        template_ng = False
        levels = []
        part_template_groups = [required for required in required_template_groups if required.hand == hand]
        part_dual_groups = [required for required in required_dual_groups if required.hand == hand]
        if not part_template_groups or not part_dual_groups:
            incomplete = True
        for required in part_template_groups:
            key = (
                required.hand,
                required.view,
                required.model_digest,
                required.roi_version,
                required.roi_digest,
            )
            threshold = template_by_identity.get(key)
            score_group = ScoreGroup(
                required.hand,
                required.view,
                ScoreBranch.TEMPLATE,
                required.model_digest,
                required.roi_version,
            )
            score = scores.get((part_id, score_group))
            if threshold is None or threshold.status != "ok" or threshold.threshold is None or score is None:
                incomplete = True
            elif score >= threshold.threshold:
                template_ng = True
        for required in part_dual_groups:
            threshold = dual_by_group.get(required)
            score = scores.get((part_id, required))
            if (
                threshold is None
                or threshold.status != "ok"
                or threshold.low is None
                or threshold.high is None
                or score is None
            ):
                incomplete = True
            else:
                levels.append(classify_dual(score, threshold.low, threshold.high).value)
        if incomplete:
            outcomes[part_id] = "INCOMPLETE"
        elif template_ng:
            outcomes[part_id] = "NG_TEMPLATE"
        elif "STRONG" in levels:
            outcomes[part_id] = "STRONG"
        elif "GRAY" in levels:
            outcomes[part_id] = "GRAY"
        else:
            outcomes[part_id] = "CLEAR"

    defect_parts = [part_id for part_id, label in labels.items() if label is GroundTruth.DEFECT]
    normal_parts = [part_id for part_id, label in labels.items() if label is GroundTruth.NORMAL]
    defect_escape_count = sum(outcomes[part_id] == "CLEAR" for part_id in defect_parts)
    normal_reject_count = sum(outcomes[part_id] in {"STRONG", "NG_TEMPLATE"} for part_id in normal_parts)
    incomplete_count = sum(value == "INCOMPLETE" for value in outcomes.values())
    dataset_release_ids = {row.dataset_release_id for row in test_rows}
    if len(dataset_release_ids) != 1:
        raise ModelContractError("held-out rows must belong to exactly one dataset release")
    return HeldOutPartMetrics(
        dataset_release_id=next(iter(dataset_release_ids)),
        test_split_id=test_split_id,
        part_count=len(outcomes),
        normal_part_count=len(normal_parts),
        defect_part_count=len(defect_parts),
        clear_count=sum(value == "CLEAR" for value in outcomes.values()),
        gray_count=sum(value == "GRAY" for value in outcomes.values()),
        strong_count=sum(value == "STRONG" for value in outcomes.values()),
        template_ng_count=sum(value == "NG_TEMPLATE" for value in outcomes.values()),
        incomplete_count=incomplete_count,
        defect_escape_count=defect_escape_count,
        normal_reject_count=normal_reject_count,
        defect_escape_rate=defect_escape_count / len(defect_parts) if defect_parts else None,
        normal_reject_rate=normal_reject_count / len(normal_parts) if normal_parts else None,
        review_rate=sum(value == "GRAY" for value in outcomes.values()) / len(outcomes) if outcomes else None,
        evaluation_complete=incomplete_count == 0 and bool(defect_parts) and bool(normal_parts),
    )


@dataclass(frozen=True, slots=True)
class CalibrationArtifact:
    """Canonical template/model threshold artifact with held-out metrics."""

    provenance: CalibrationProvenance
    parameters: FitParameters
    required_dual_groups: tuple[ScoreGroup, ...]
    required_template_groups: tuple[TemplateCalibrationGroup, ...]
    dual_thresholds: tuple[DualThreshold, ...]
    template_thresholds: tuple[TemplateThresholdFit, ...]
    heldout_metrics: HeldOutPartMetrics
    calibration_valid: bool

    def __post_init__(self) -> None:
        """Prevent callers from manually asserting an unsafe valid state."""
        if not isinstance(self.calibration_valid, bool):
            raise ModelContractError("calibration_valid must be a boolean")
        for field in (
            "required_dual_groups",
            "required_template_groups",
            "dual_thresholds",
            "template_thresholds",
        ):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if not isinstance(self.provenance, CalibrationProvenance):
            raise ModelContractError("calibration provenance must be CalibrationProvenance")
        if not isinstance(self.parameters, FitParameters):
            raise ModelContractError("calibration parameters must be FitParameters")
        if not isinstance(self.heldout_metrics, HeldOutPartMetrics):
            raise ModelContractError("calibration heldout_metrics must be HeldOutPartMetrics")
        if any(not isinstance(group, ScoreGroup) for group in self.required_dual_groups):
            raise ModelContractError("required_dual_groups must contain ScoreGroup values")
        if any(not isinstance(group, TemplateCalibrationGroup) for group in self.required_template_groups):
            raise ModelContractError("required_template_groups must contain TemplateCalibrationGroup values")
        if any(not isinstance(record, DualThreshold) for record in self.dual_thresholds):
            raise ModelContractError("dual_thresholds must contain DualThreshold values")
        if any(not isinstance(record, TemplateThresholdFit) for record in self.template_thresholds):
            raise ModelContractError("template_thresholds must contain TemplateThresholdFit values")
        if self.calibration_valid:
            if not self.required_dual_groups or not self.required_template_groups:
                raise ModelContractError("valid calibration requires both template and model threshold groups")
            if any(record.status != "ok" for record in self.dual_thresholds):
                raise ModelContractError("valid calibration cannot contain an invalid dual threshold")
            if any(record.status != "ok" for record in self.template_thresholds):
                raise ModelContractError("valid calibration cannot contain an invalid template threshold")
            if any(
                record.normal_count < self.parameters.min_normal_parts
                or record.defect_count < self.parameters.min_defect_parts
                for record in (*self.dual_thresholds, *self.template_thresholds)
            ):
                raise ModelContractError("valid calibration threshold counts do not meet fitted minimums")
            if not self.heldout_metrics.evaluation_complete:
                raise ModelContractError("valid calibration requires complete held-out evaluation")
            dual_groups = {record.group for record in self.dual_thresholds}
            if len(self.dual_thresholds) != len(dual_groups) or dual_groups != set(self.required_dual_groups):
                raise ModelContractError("valid calibration dual groups must exactly match required groups")
            template_groups = {
                (record.slot.hand, record.slot.view, record.model_digest, record.roi_version, record.roi_digest)
                for record in self.template_thresholds
            }
            required_template_groups = {
                (group.hand, group.view, group.model_digest, group.roi_version, group.roi_digest)
                for group in self.required_template_groups
            }
            if len(self.template_thresholds) != len(template_groups) or template_groups != required_template_groups:
                raise ModelContractError("valid calibration template groups must exactly match required groups")
            expected_digests = {group.model_digest for group in self.required_dual_groups} | {
                group.model_digest for group in self.required_template_groups
            }
            if expected_digests != set(self.provenance.model_digests):
                raise ModelContractError("valid calibration provenance model digests do not match required groups")
            if (
                self.heldout_metrics.dataset_release_id != self.provenance.dataset_release_id
                or self.heldout_metrics.test_split_id != self.provenance.test_split_id
            ):
                raise ModelContractError("valid calibration held-out provenance does not match artifact provenance")
            fit_records = (*self.dual_thresholds, *self.template_thresholds)
            if any(
                record.dataset_release_id != self.provenance.dataset_release_id
                or record.calibration_split_id != self.provenance.calibration_split_id
                for record in fit_records
            ):
                raise ModelContractError("valid calibration fit provenance does not match artifact provenance")

    @property
    def artifact_sha256(self) -> str:
        """Return the content digest of the artifact payload (excluding the digest itself)."""
        return canonical_sha256(self.payload())

    def payload(self) -> dict[str, object]:
        """Return deterministic artifact content."""
        return {
            "schema": "zs32.calibration",
            "schema_version": 1,
            "provenance": self.provenance.to_dict(),
            "parameters": self.parameters.to_dict(),
            "required_dual_groups": [
                {
                    "hand": group.hand,
                    "view": group.view,
                    "branch": group.branch.value,
                    "model_digest": group.model_digest,
                    "roi_version": group.roi_version,
                }
                for group in self.required_dual_groups
            ],
            "required_template_groups": [
                {
                    "hand": group.hand,
                    "view": group.view,
                    "model_digest": group.model_digest,
                    "roi_version": group.roi_version,
                    "roi_digest": group.roi_digest,
                }
                for group in self.required_template_groups
            ],
            "dual_thresholds": [record.to_dict() for record in self.dual_thresholds],
            "template_thresholds": [record.to_dict() for record in self.template_thresholds],
            "heldout_metrics": self.heldout_metrics.to_dict(),
            "calibration_valid": self.calibration_valid,
        }

    def to_dict(self) -> dict[str, object]:
        """Return payload plus its canonical content hash."""
        return {**self.payload(), "artifact_sha256": self.artifact_sha256}


def build_calibration_artifact(
    *,
    provenance: CalibrationProvenance,
    parameters: FitParameters,
    required_dual_groups: Sequence[ScoreGroup],
    required_template_groups: Sequence[TemplateCalibrationGroup],
    dual_thresholds: Sequence[DualThreshold],
    template_thresholds: Sequence[TemplateThresholdFit],
    heldout_metrics: HeldOutPartMetrics,
) -> CalibrationArtifact:
    """Assemble a fail-closed artifact after fitting and independent evaluation."""
    required_dual = tuple(sorted(set(required_dual_groups)))
    required_template = tuple(sorted(set(required_template_groups)))
    dual = tuple(sorted(dual_thresholds, key=lambda record: record.group))
    template = tuple(
        sorted(template_thresholds, key=lambda record: (record.slot.hand, record.slot.view, record.model_digest))
    )
    actual_dual = {record.group for record in dual}
    expected_template = {
        (group.hand, group.view, group.model_digest, group.roi_version, group.roi_digest)
        for group in required_template
    }
    actual_template = {
        (record.slot.hand, record.slot.view, record.model_digest, record.roi_version, record.roi_digest)
        for record in template
    }
    model_digests = {group.model_digest for group in required_dual} | {
        group.model_digest for group in required_template
    }
    provenance_matches = model_digests == set(provenance.model_digests)
    fit_provenance_matches = all(
        record.dataset_release_id == provenance.dataset_release_id
        and record.calibration_split_id == provenance.calibration_split_id
        for record in (*dual, *template)
    )
    valid = (
        bool(required_dual)
        and bool(required_template)
        and actual_dual == set(required_dual)
        and len(dual) == len(required_dual)
        and actual_template == expected_template
        and len(template) == len(required_template)
        and all(record.status == "ok" for record in dual)
        and all(record.status == "ok" for record in template)
        and all(
            record.normal_count >= parameters.min_normal_parts
            and record.defect_count >= parameters.min_defect_parts
            for record in (*dual, *template)
        )
        and heldout_metrics.evaluation_complete
        and heldout_metrics.dataset_release_id == provenance.dataset_release_id
        and heldout_metrics.test_split_id == provenance.test_split_id
        and provenance_matches
        and fit_provenance_matches
    )
    return CalibrationArtifact(
        provenance=provenance,
        parameters=parameters,
        required_dual_groups=required_dual,
        required_template_groups=required_template,
        dual_thresholds=dual,
        template_thresholds=template,
        heldout_metrics=heldout_metrics,
        calibration_valid=valid,
    )
