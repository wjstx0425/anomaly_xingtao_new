"""Linux-authoritative contract tests for strict ZS32 fusion."""

from __future__ import annotations

import sys

import pytest

from tests.unit.zs32_refactor.domain._fixtures import (
    DIGESTS,
    FUSION_POLICY_BYTES,
    recipe_mapping,
    roi_mapping,
    topology_mapping,
)
from zs32_inspection.config.compiler import compile_deployment_contract
from zs32_inspection.config.schemas import parse_recipe, parse_roi_config, parse_topology
from zs32_inspection.domain.contracts import DeploymentContract
from zs32_inspection.domain.decisions import InspectionStatus
from zs32_inspection.domain.evidence import (
    EvidenceBranch,
    EvidenceLevel,
    ModelEvidence,
    TemplateEvidence,
    TemplateOutcome,
)
from zs32_inspection.domain.errors import EvidenceValidationError
from zs32_inspection.domain.identity import Hand
from zs32_inspection.fusion.completeness import EvidenceContext
from zs32_inspection.fusion.engine import StrictFusionEngine

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="ZS32 verification is Linux-only")

HASH = "1" * 64
VIEWS = ("front_0", "back_0", "front_1", "back_1", "front_2", "back_2")


def _contract() -> DeploymentContract:
    topology = parse_topology(topology_mapping(3))
    roi = parse_roi_config(roi_mapping(topology))
    recipe = parse_recipe(recipe_mapping(topology, roi))
    return compile_deployment_contract(
        topology,
        roi,
        recipe,
        fusion_policy_bytes=FUSION_POLICY_BYTES,
    )


def _context(contract: DeploymentContract) -> EvidenceContext:
    return EvidenceContext.from_contract(
        inspection_id="inspection-1",
        capture_set_id="capture-1",
        part_instance_id="part-1",
        hand=Hand.RIGHT,
        contract=contract,
    )


def _templates(outcomes: dict[str, TemplateOutcome] | None = None) -> list[TemplateEvidence]:
    outcomes = outcomes or {}
    return [
        TemplateEvidence(
            inspection_id="inspection-1",
            capture_set_id="capture-1",
            part_instance_id="part-1",
            hand=Hand.RIGHT,
            view_id=view,
            outcome=outcomes.get(view, TemplateOutcome.PASS),
            score=0.9 if outcomes.get(view) is TemplateOutcome.NG_TEMPLATE else 0.1,
            threshold=0.5,
            template_sha256=DIGESTS["template"],
            threshold_sha256=DIGESTS["template_calibration"],
            source_sha256=HASH,
            crop_sha256=HASH,
            roi_config_id="zs32-roi-v2",
        )
        for view in VIEWS
    ]


def _models(levels: dict[tuple[str, EvidenceBranch], EvidenceLevel] | None = None) -> list[ModelEvidence]:
    levels = levels or {}
    return [
        ModelEvidence(
            inspection_id="inspection-1",
            capture_set_id="capture-1",
            part_instance_id="part-1",
            hand=Hand.RIGHT,
            view_id=view,
            branch=branch,
            level=levels.get((view, branch), EvidenceLevel.CLEAR),
            score={
                EvidenceLevel.CLEAR: 0.1,
                EvidenceLevel.GRAY: 0.5,
                EvidenceLevel.STRONG: 0.9,
            }[levels.get((view, branch), EvidenceLevel.CLEAR)],
            model_family="yolo" if branch is EvidenceBranch.YOLO else "patchcore",
            model_sha256=(
                DIGESTS["yolo"] if branch is EvidenceBranch.YOLO else DIGESTS["anomaly"]
            ),
            roi_config_id="zs32-roi-v2",
            source_sha256=HASH,
            crop_sha256=HASH,
            threshold_sha256=DIGESTS["model_calibration"],
        )
        for view in VIEWS
        for branch in EvidenceBranch
    ]


def test_template_mismatch_short_circuits_to_releasable_ng() -> None:
    contract = _contract()
    decision = StrictFusionEngine().fuse_templates(
        context=_context(contract),
        evidence=_templates({"front_2": TemplateOutcome.NG_TEMPLATE}),
    )

    assert decision is not None
    assert decision.inspection_status is InspectionStatus.NG_TEMPLATE
    assert decision.released_status is InspectionStatus.NG_TEMPLATE


def test_missing_template_is_system_contract_failure_not_template_ng() -> None:
    contract = _contract()
    with pytest.raises(EvidenceValidationError, match="template evidence view mismatch"):
        StrictFusionEngine().fuse_templates(context=_context(contract), evidence=_templates()[:-1])


def test_all_model_rows_must_be_clear_for_ok() -> None:
    contract = _contract()
    decision = StrictFusionEngine().fuse_models(context=_context(contract), evidence=_models())

    assert decision.inspection_status is InspectionStatus.OK
    assert decision.released_status is InspectionStatus.OK


def test_gray_never_releases_ok() -> None:
    contract = _contract()
    levels = {("back_1", EvidenceBranch.ANOMALY): EvidenceLevel.GRAY}
    decision = StrictFusionEngine().fuse_models(context=_context(contract), evidence=_models(levels))

    assert decision.inspection_status is InspectionStatus.REVIEW
    assert decision.released_status is None


def test_any_strong_is_ng_without_majority_vote() -> None:
    contract = _contract()
    levels = {("back_0", EvidenceBranch.YOLO): EvidenceLevel.STRONG}
    decision = StrictFusionEngine().fuse_models(context=_context(contract), evidence=_models(levels))

    assert decision.inspection_status is InspectionStatus.NG_YOLO
    assert decision.released_status is InspectionStatus.NG_YOLO


def test_missing_second_layer_row_cannot_produce_ok() -> None:
    contract = _contract()
    with pytest.raises(EvidenceValidationError, match="model evidence group mismatch"):
        StrictFusionEngine().fuse_models(context=_context(contract), evidence=_models()[:-1])


def test_simultaneous_strong_rows_are_all_retained() -> None:
    contract = _contract()
    levels = {
        ("front_0", EvidenceBranch.YOLO): EvidenceLevel.STRONG,
        ("back_0", EvidenceBranch.ANOMALY): EvidenceLevel.STRONG,
    }
    decision = StrictFusionEngine().fuse_models(context=_context(contract), evidence=_models(levels))

    assert decision.inspection_status is InspectionStatus.NG_ANOMALY
    assert len(decision.reason_codes) == 2
