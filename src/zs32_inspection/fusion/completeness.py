"""Identity and completeness checks preceding ZS32 evidence fusion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from zs32_inspection.domain.evidence import EvidenceBranch, ModelEvidence, TemplateEvidence
from zs32_inspection.domain.errors import EvidenceValidationError
from zs32_inspection.domain.identity import Hand, require_non_empty

if TYPE_CHECKING:
    from zs32_inspection.domain.contracts import DeploymentContract


@dataclass(frozen=True, slots=True)
class EvidenceContext:
    """Identity expected for every row produced by one inspection run."""

    inspection_id: str
    capture_set_id: str
    part_instance_id: str
    hand: Hand
    required_views: tuple[str, ...]
    contract: DeploymentContract | None = None

    def __post_init__(self) -> None:
        for field in ("inspection_id", "capture_set_id", "part_instance_id"):
            try:
                object.__setattr__(self, field, require_non_empty(getattr(self, field), field))
            except ValueError as error:
                raise EvidenceValidationError(str(error)) from error
        object.__setattr__(self, "hand", Hand.parse(self.hand))
        required = tuple(self.required_views)
        if not required or len(required) != len(set(required)) or any(not view for view in required):
            raise EvidenceValidationError("required_views must be non-empty and unique")
        object.__setattr__(self, "required_views", required)
        if self.contract is None:
            raise EvidenceValidationError("strict fusion requires an exact DeploymentContract")
        if self.hand not in self.contract.allowed_hands:
            raise EvidenceValidationError(f"hand {self.hand.value!r} is not enabled by deployment contract")
        if required != self.contract.topology.required_views:
            raise EvidenceValidationError("required_views differ from the deployment topology")

    @classmethod
    def from_contract(
        cls,
        *,
        inspection_id: str,
        capture_set_id: str,
        part_instance_id: str,
        hand: Hand,
        contract: DeploymentContract,
    ) -> EvidenceContext:
        """Bind completeness checks to the exact deployment contract."""
        return cls(
            inspection_id=inspection_id,
            capture_set_id=capture_set_id,
            part_instance_id=part_instance_id,
            hand=hand,
            required_views=contract.topology.required_views,
            contract=contract,
        )


def _validate_identity(row: ModelEvidence | TemplateEvidence, context: EvidenceContext) -> None:
    actual = (row.inspection_id, row.capture_set_id, row.part_instance_id, row.hand)
    expected = (
        context.inspection_id,
        context.capture_set_id,
        context.part_instance_id,
        context.hand,
    )
    if actual != expected:
        raise EvidenceValidationError(f"evidence identity mismatch: expected={expected!r}, actual={actual!r}")


def require_complete_templates(
    rows: Sequence[TemplateEvidence], context: EvidenceContext
) -> dict[str, TemplateEvidence]:
    """Require exactly one template result for every topology view."""
    if context.contract is None:  # defensive against forged/uninitialized objects
        raise EvidenceValidationError("strict fusion requires an exact DeploymentContract")
    by_view: dict[str, TemplateEvidence] = {}
    for row in rows:
        _validate_identity(row, context)
        try:
            context.contract.validate_template_evidence(row)
        except ValueError as error:
            raise EvidenceValidationError(str(error)) from error
        if row.view_id in by_view:
            raise EvidenceValidationError(f"duplicate template evidence for view {row.view_id!r}")
        by_view[row.view_id] = row
    expected = set(context.required_views)
    actual = set(by_view)
    if actual != expected:
        raise EvidenceValidationError(
            f"template evidence view mismatch; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return by_view


def require_complete_models(
    rows: Sequence[ModelEvidence], context: EvidenceContext
) -> dict[tuple[str, EvidenceBranch], ModelEvidence]:
    """Require ``views x {anomaly,yolo}`` with no duplicate or extra row."""
    if context.contract is None:  # defensive against forged/uninitialized objects
        raise EvidenceValidationError("strict fusion requires an exact DeploymentContract")
    indexed: dict[tuple[str, EvidenceBranch], ModelEvidence] = {}
    for row in rows:
        _validate_identity(row, context)
        try:
            context.contract.validate_model_evidence(row)
        except ValueError as error:
            raise EvidenceValidationError(str(error)) from error
        key = (row.view_id, row.branch)
        if key in indexed:
            raise EvidenceValidationError(
                f"duplicate model evidence for view={row.view_id!r}, branch={row.branch.value!r}"
            )
        indexed[key] = row
    expected = {(view, branch) for view in context.required_views for branch in EvidenceBranch}
    actual = set(indexed)
    if actual != expected:
        missing = sorted((view, branch.value) for view, branch in expected - actual)
        extra = sorted((view, branch.value) for view, branch in actual - expected)
        raise EvidenceValidationError(f"model evidence group mismatch; missing={missing}, extra={extra}")
    return indexed
