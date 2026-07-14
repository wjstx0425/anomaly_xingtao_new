# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Binary template-gate decision and matcher boundary."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from zs32_inspection.domain.contracts import TemplateThresholdRecord
from zs32_inspection.domain.evidence import TemplateEvidence, TemplateOutcome
from zs32_inspection.domain.identity import Hand
from zs32_inspection.models.base import (
    ModelContractError,
    ModelInput,
    ModelSlot,
    require_sha256,
    require_text,
)


@dataclass(frozen=True, slots=True)
class TemplateScore:
    """Continuous risk emitted by a matcher before the binary gate."""

    inspection_id: str
    part_instance_id: str
    capture_set_id: str
    hand: str
    view: str
    risk_score: float
    similarity: float
    model_digest: str
    roi_config_id: str
    roi_digest: str
    source_sha256: str
    crop_sha256: str
    best_template_path: Path
    best_template_sha256: str
    offset_xy: tuple[int, int]

    def __post_init__(self) -> None:
        for field in (
            "inspection_id",
            "part_instance_id",
            "capture_set_id",
            "hand",
            "view",
            "roi_config_id",
        ):
            object.__setattr__(self, field, require_text(getattr(self, field), f"template_score.{field}"))
        object.__setattr__(self, "hand", Hand.parse(self.hand).value)
        if (
            isinstance(self.risk_score, bool)
            or not isinstance(self.risk_score, (int, float))
            or isinstance(self.similarity, bool)
            or not isinstance(self.similarity, (int, float))
            or not math.isfinite(self.risk_score)
            or not math.isfinite(self.similarity)
        ):
            raise ModelContractError("template score and similarity must be finite")
        if not -1 <= self.similarity <= 1 or not 0 <= self.risk_score <= 2:
            raise ModelContractError("template TM_CCOEFF_NORMED similarity/risk must lie in [-1,1]/[0,2]")
        if not math.isclose(self.risk_score, 1 - self.similarity, rel_tol=0, abs_tol=1e-12):
            raise ModelContractError("template risk_score must equal 1 - similarity")
        for field in ("model_digest", "roi_digest", "source_sha256", "crop_sha256", "best_template_sha256"):
            object.__setattr__(self, field, require_sha256(getattr(self, field), f"template_score.{field}"))
        object.__setattr__(self, "best_template_path", Path(self.best_template_path).expanduser())

    @property
    def slot(self) -> ModelSlot:
        """Return the score's exact threshold slot."""
        return ModelSlot(self.hand, self.view)


def decide_template(score: TemplateScore, threshold: TemplateThresholdRecord) -> TemplateEvidence:
    """Apply the one-threshold PASS/NG_TEMPLATE rule.

    Invalid threshold or identity data raises ``ModelContractError``. The
    orchestrator must map that system fault to ``SYSTEM_ERROR``, never part NG.
    """
    if score.hand != threshold.hand.value or score.view != threshold.view_id:
        raise ModelContractError("template score and threshold hand/view do not match")
    if score.model_digest != threshold.template_sha256:
        raise ModelContractError("template score and threshold model digests do not match")
    outcome = TemplateOutcome.NG_TEMPLATE if score.risk_score >= threshold.threshold else TemplateOutcome.PASS
    return TemplateEvidence(
        inspection_id=score.inspection_id,
        capture_set_id=score.capture_set_id,
        part_instance_id=score.part_instance_id,
        hand=Hand.parse(score.hand),
        view_id=score.view,
        outcome=outcome,
        score=score.risk_score,
        threshold=float(threshold.threshold),
        template_sha256=score.model_digest,
        threshold_sha256=threshold.calibration_sha256,
        source_sha256=score.source_sha256,
        crop_sha256=score.crop_sha256,
        roi_config_id=score.roi_config_id,
    )


@runtime_checkable
class TemplatePredictor(Protocol):
    """Matcher boundary that emits raw scores from canonical crops only."""

    def score_batch(
        self,
        samples: Sequence[ModelInput],
        *,
        inspection_id: str,
    ) -> Sequence[TemplateScore]:
        """Score samples without applying thresholds or orchestration decisions."""
