# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Binary template gate for ZS32 canonical ROI crops."""

from zs32_inspection.domain.evidence import TemplateEvidence, TemplateOutcome

from .artifacts import TemplateAssetSpec, TemplateThresholdFit, validate_template_assets
from .opencv_backend import OpenCvTemplatePredictor, OpenCvTemplateTrainingBackend
from .predictor import TemplatePredictor, TemplateScore, decide_template
from .trainer import TemplateTrainer, TemplateTrainSpec, TemplateTrainingBackend

__all__ = [
    "TemplateAssetSpec",
    "TemplateEvidence",
    "TemplateOutcome",
    "OpenCvTemplatePredictor",
    "OpenCvTemplateTrainingBackend",
    "TemplatePredictor",
    "TemplateScore",
    "TemplateThresholdFit",
    "TemplateTrainer",
    "TemplateTrainSpec",
    "TemplateTrainingBackend",
    "decide_template",
    "validate_template_assets",
]
