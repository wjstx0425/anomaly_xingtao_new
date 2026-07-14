"""Strict template and second-layer fusion state machine."""

from __future__ import annotations

from collections.abc import Sequence

from zs32_inspection.domain.decisions import InspectionDecision, InspectionStatus
from zs32_inspection.domain.evidence import (
    EvidenceBranch,
    EvidenceLevel,
    ModelEvidence,
    TemplateEvidence,
    TemplateOutcome,
)

from .completeness import EvidenceContext, require_complete_models, require_complete_templates
from .policy import DEFAULT_POLICY, FusionPolicy


class StrictFusionEngine:
    """Apply the frozen ZS32 contract without voting or evidence defaults."""

    def __init__(self, policy: FusionPolicy = DEFAULT_POLICY) -> None:
        self.policy = policy

    def fuse_templates(
        self,
        *,
        context: EvidenceContext,
        evidence: Sequence[TemplateEvidence],
    ) -> InspectionDecision | None:
        """Return ``NG_TEMPLATE`` on mismatch, otherwise allow layer two.

        Missing or malformed rows raise an evidence contract error.  The
        orchestrator maps that runtime failure to ``SYSTEM_ERROR``; it is never
        interpreted as a product mismatch.
        """
        indexed = require_complete_templates(evidence, context)
        failed = [view for view in context.required_views if indexed[view].outcome is TemplateOutcome.NG_TEMPLATE]
        if not failed:
            return None
        reasons = tuple(f"template_strong:{view}" for view in failed)
        return InspectionDecision(
            inspection_id=context.inspection_id,
            evidence_status=InspectionStatus.NG_TEMPLATE,
            inspection_status=InspectionStatus.NG_TEMPLATE,
            review_status=None,
            released_status=InspectionStatus.NG_TEMPLATE,
            reason_codes=reasons,
        )

    def fuse_models(
        self,
        *,
        context: EvidenceContext,
        evidence: Sequence[ModelEvidence],
    ) -> InspectionDecision:
        """Require all rows and apply STRONG > GRAY > all-CLEAR semantics."""
        indexed = require_complete_models(evidence, context)
        ordered = [
            indexed[(view, branch)]
            for view in context.required_views
            for branch in EvidenceBranch
        ]
        strong = [row for row in ordered if row.level is EvidenceLevel.STRONG]
        if strong:
            strong_branches = {row.branch for row in strong}
            primary = next(branch for branch in self.policy.strong_priority if branch in strong_branches)
            status = (
                InspectionStatus.NG_ANOMALY
                if primary is EvidenceBranch.ANOMALY
                else InspectionStatus.NG_YOLO
            )
            reasons = tuple(
                f"strong:{row.branch.value}:{row.view_id}:{row.score:.17g}" for row in strong
            )
            return InspectionDecision(
                inspection_id=context.inspection_id,
                evidence_status=status,
                inspection_status=status,
                review_status=None,
                released_status=status,
                reason_codes=reasons,
            )

        gray = [row for row in ordered if row.level is EvidenceLevel.GRAY]
        if gray:
            return InspectionDecision(
                inspection_id=context.inspection_id,
                evidence_status=None,
                inspection_status=InspectionStatus.REVIEW,
                review_status="PENDING",
                released_status=None,
                reason_codes=tuple(
                    f"gray:{row.branch.value}:{row.view_id}:{row.score:.17g}" for row in gray
                ),
            )

        return InspectionDecision(
            inspection_id=context.inspection_id,
            evidence_status=None,
            inspection_status=InspectionStatus.OK,
            review_status=None,
            released_status=InspectionStatus.OK,
            reason_codes=("all_required_model_evidence_clear",),
        )
