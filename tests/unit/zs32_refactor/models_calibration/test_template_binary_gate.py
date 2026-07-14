# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Linux test definitions for the binary template gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from zs32_inspection.domain.contracts import TemplateThresholdRecord
from zs32_inspection.domain.evidence import TemplateOutcome
from zs32_inspection.domain.errors import RecipeValidationError
from zs32_inspection.models import ModelContractError
from zs32_inspection.template import TemplateScore, decide_template

HEX = "c" * 64


def _score(value: float) -> TemplateScore:
    return TemplateScore(
        inspection_id="inspection-1",
        part_instance_id="part-1",
        capture_set_id="capture-1",
        hand="right",
        view="front",
        risk_score=value,
        similarity=1 - value,
        model_digest=HEX,
        roi_config_id="roi-v1",
        roi_digest=HEX,
        source_sha256=HEX,
        crop_sha256=HEX,
        best_template_path=Path("template.png"),
        best_template_sha256=HEX,
        offset_xy=(0, 0),
    )


def _threshold(value: float) -> TemplateThresholdRecord:
    return TemplateThresholdRecord(
        hand="right",
        view_id="front",
        threshold=value,
        template_sha256=HEX,
        calibration_sha256=HEX,
    )


def test_template_has_exactly_pass_and_ng_outcomes() -> None:
    assert decide_template(_score(0.19), _threshold(0.2)).outcome is TemplateOutcome.PASS
    assert decide_template(_score(0.20), _threshold(0.2)).outcome is TemplateOutcome.NG_TEMPLATE
    assert {item.value for item in TemplateOutcome} == {"PASS", "NG_TEMPLATE"}


def test_invalid_template_threshold_is_system_fault_not_ng_evidence() -> None:
    with pytest.raises(RecipeValidationError, match="finite"):
        _threshold(float("nan"))


def test_template_rejects_model_identity_mismatch() -> None:
    threshold = TemplateThresholdRecord(
        hand="right",
        view_id="front",
        threshold=0.2,
        template_sha256="a" * 64,
        calibration_sha256=HEX,
    )
    with pytest.raises(ModelContractError, match="model digests"):
        decide_template(_score(0.1), threshold)
