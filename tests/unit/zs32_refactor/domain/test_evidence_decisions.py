"""Evidence identity and final decision safety tests."""

from __future__ import annotations

import pytest

from zs32_inspection.domain.decisions import InspectionDecision, InspectionStatus
from zs32_inspection.domain.errors import DecisionValidationError, EvidenceValidationError
from zs32_inspection.domain.evidence import EvidenceBranch, EvidenceLevel, ModelEvidence
from zs32_inspection.domain.identity import Hand


def test_non_finite_model_score_is_rejected() -> None:
    """NaN cannot silently classify into CLEAR/GRAY/STRONG."""
    with pytest.raises(EvidenceValidationError, match="score must be finite"):
        ModelEvidence(
            inspection_id="inspection-1",
            capture_set_id="capture-1",
            part_instance_id="part-1",
            hand=Hand.RIGHT,
            view_id="front",
            branch=EvidenceBranch.ANOMALY,
            level=EvidenceLevel.CLEAR,
            score=float("nan"),
            model_family="patchcore",
            model_sha256="1" * 64,
            roi_config_id="zs32-roi-v2",
            source_sha256="2" * 64,
            crop_sha256="3" * 64,
            threshold_sha256="4" * 64,
        )


def test_system_error_can_keep_ng_evidence_but_never_release_it() -> None:
    """Known defect evidence survives a later system fault without formal release."""
    decision = InspectionDecision(
        inspection_id="inspection-1",
        evidence_status=InspectionStatus.NG_ANOMALY,
        inspection_status=InspectionStatus.SYSTEM_ERROR,
        review_status=None,
        released_status=None,
        reason_codes=("ANOMALY_STRONG", "YOLO_RUNTIME_FAILED"),
    )

    assert decision.evidence_status is InspectionStatus.NG_ANOMALY
    assert decision.released_status is None


@pytest.mark.parametrize(
    "status",
    [
        InspectionStatus.SYSTEM_ERROR,
        InspectionStatus.INVALID_CAPTURE,
        InspectionStatus.RETAKE,
        InspectionStatus.REVIEW,
    ],
)
def test_non_releasable_status_cannot_publish_ok(status: InspectionStatus) -> None:
    """Fault, capture, retake, and review states can never publish OK."""
    with pytest.raises(DecisionValidationError, match="must not publish"):
        InspectionDecision(
            inspection_id="inspection-1",
            evidence_status=None,
            inspection_status=status,
            review_status=None,
            released_status=InspectionStatus.OK,
        )


def test_completed_ng_requires_matching_evidence_and_release_channels() -> None:
    with pytest.raises(DecisionValidationError, match="same evidence_status"):
        InspectionDecision(
            inspection_id="inspection-1",
            evidence_status=None,
            inspection_status=InspectionStatus.NG_YOLO,
            review_status=None,
            released_status=InspectionStatus.NG_YOLO,
        )


def test_review_requires_review_status() -> None:
    with pytest.raises(DecisionValidationError, match="review_status"):
        InspectionDecision(
            inspection_id="inspection-1",
            evidence_status=None,
            inspection_status=InspectionStatus.REVIEW,
            review_status=None,
            released_status=None,
        )
