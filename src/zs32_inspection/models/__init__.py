# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Model plugin and deployment-asset contracts for ZS32."""

from .anomalydino import AnomalyDINOAdapter
from .anomalib_backend import (
    AnomalibBackendError,
    AnomalibBatchPredictor,
    AnomalibRuntimeBackend,
    AnomalibTrainerBackend,
    sha256_auxiliary_tree,
)
from .base import (
    AnomalyAdapter,
    AnomalyFamily,
    AnomalyModelArtifact,
    AnomalyPredictor,
    AssetFile,
    BackendUnavailableError,
    DeviceSpec,
    ModelContractError,
    ModelInput,
    ModelSlot,
    RawModelScore,
    TrainSpec,
    YoloDetection,
)
from .deployment import DeploymentAssetDescription
from .efficientad import EfficientADAdapter
from .patchcore import PatchCoreAdapter
from .registry import CandidateStatus, ModelCandidate, register_candidate, transition_candidate
from .yolo import UltralyticsYoloAdapter, YoloDeploymentSpec, YoloRuntimeSettings, import_yolo_bundle
from .yolo_receipt import YoloTrainerSource, YoloTrainingReceipt, load_yolo_training_receipt
from .ultralytics_backend import UltralyticsBatchPredictor, UltralyticsRuntimeBackend

__all__ = [
    "AnomalyAdapter",
    "AnomalyDINOAdapter",
    "AnomalibBackendError",
    "AnomalibBatchPredictor",
    "AnomalibRuntimeBackend",
    "AnomalibTrainerBackend",
    "sha256_auxiliary_tree",
    "AnomalyFamily",
    "AnomalyModelArtifact",
    "AnomalyPredictor",
    "AssetFile",
    "BackendUnavailableError",
    "DeviceSpec",
    "DeploymentAssetDescription",
    "EfficientADAdapter",
    "ModelContractError",
    "ModelCandidate",
    "ModelInput",
    "ModelSlot",
    "RawModelScore",
    "PatchCoreAdapter",
    "CandidateStatus",
    "TrainSpec",
    "UltralyticsYoloAdapter",
    "UltralyticsBatchPredictor",
    "UltralyticsRuntimeBackend",
    "YoloDeploymentSpec",
    "YoloRuntimeSettings",
    "YoloTrainerSource",
    "YoloTrainingReceipt",
    "YoloDetection",
    "import_yolo_bundle",
    "load_yolo_training_receipt",
    "register_candidate",
    "transition_candidate",
]
