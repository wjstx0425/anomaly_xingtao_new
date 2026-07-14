# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Calibration, held-out evaluation, and production acceptance contracts for ZS32."""

from .acceptance import (
    HeldOutAcceptanceDecision,
    HeldOutAcceptanceLimits,
    HeldOutAcceptancePolicy,
    evaluate_heldout_acceptance,
    load_heldout_acceptance_decision_bytes,
    load_heldout_acceptance_policy_bytes,
)
from .reports import (
    CalibrationArtifact,
    CalibrationProvenance,
    HeldOutPartMetrics,
    build_calibration_artifact,
    evaluate_heldout,
)
from .scoring import (
    GroundTruth,
    ScoreBranch,
    ScoreGroup,
    ScoreRow,
    SplitRole,
    classify_dual,
    to_domain_model_evidence,
)
from .thresholds import (
    DualThreshold,
    FitParameters,
    TemplateCalibrationGroup,
    fit_dual_thresholds,
    fit_template_thresholds,
    to_domain_template_threshold_record,
    to_domain_threshold_record,
)

__all__ = [
    "CalibrationArtifact",
    "CalibrationProvenance",
    "DualThreshold",
    "FitParameters",
    "GroundTruth",
    "HeldOutAcceptanceDecision",
    "HeldOutAcceptanceLimits",
    "HeldOutAcceptancePolicy",
    "HeldOutPartMetrics",
    "ScoreBranch",
    "ScoreGroup",
    "ScoreRow",
    "SplitRole",
    "TemplateCalibrationGroup",
    "build_calibration_artifact",
    "classify_dual",
    "evaluate_heldout",
    "evaluate_heldout_acceptance",
    "fit_dual_thresholds",
    "fit_template_thresholds",
    "load_heldout_acceptance_decision_bytes",
    "load_heldout_acceptance_policy_bytes",
    "to_domain_template_threshold_record",
    "to_domain_threshold_record",
    "to_domain_model_evidence",
]
