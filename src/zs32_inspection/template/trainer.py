# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Explicit Linux training boundary for template assets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from zs32_inspection.models.base import (
    BackendUnavailableError,
    ModelContractError,
    ModelSlot,
    freeze_json_mapping,
    require_sha256,
    require_text,
)

from .artifacts import TemplateAssetSpec


@dataclass(frozen=True, slots=True)
class TemplateTrainSpec:
    """One slot's immutable template-training request."""

    slot: ModelSlot
    dataset_release_id: str
    dataset_manifest_digest: str
    dataset_root: Path
    train_split_id: str
    recipe_digest: str
    roi_version: str
    roi_digest: str
    output_dir: Path
    parameters: Mapping[str, Any]
    device: str
    execution_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        for field in ("dataset_release_id", "train_split_id", "roi_version"):
            object.__setattr__(self, field, require_text(getattr(self, field), f"template_train.{field}"))
        for field in ("dataset_manifest_digest", "recipe_digest", "roi_digest"):
            object.__setattr__(self, field, require_sha256(getattr(self, field), f"template_train.{field}"))
        dataset_root = Path(self.dataset_root).expanduser()
        if dataset_root.is_symlink() or not dataset_root.is_dir():
            raise ModelContractError(
                f"template_train.dataset_root must be a verified materialized export: {dataset_root}"
            )
        object.__setattr__(self, "dataset_root", dataset_root.resolve())
        object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser())
        object.__setattr__(
            self,
            "parameters",
            freeze_json_mapping(self.parameters, "template_train.parameters"),
        )
        object.__setattr__(self, "device", require_text(self.device, "template_train.device"))
        object.__setattr__(
            self,
            "execution_receipt",
            freeze_json_mapping(
                self.execution_receipt,
                "template_train.execution_receipt",
            ),
        )
        from zs32_inspection.runtime.execution_receipt import ExecutionReceipt

        receipt = ExecutionReceipt.from_mapping(self.execution_receipt)
        if receipt.operation != "train_template":
            raise ModelContractError("template TrainSpec requires a train_template receipt")


@runtime_checkable
class TemplateTrainingBackend(Protocol):
    """Linux implementation that builds reference templates but does not fit thresholds."""

    def __call__(self, spec: TemplateTrainSpec) -> TemplateAssetSpec:
        """Train matcher assets from only the requested training split."""


class TemplateTrainer:
    """Keep real OpenCV/data work behind an explicit injected boundary."""

    def __init__(self, backend: TemplateTrainingBackend | None = None) -> None:
        self._backend = backend

    def train(self, spec: TemplateTrainSpec) -> TemplateAssetSpec:
        """Train without silently fitting thresholds or promoting a release."""
        if self._backend is None:
            raise BackendUnavailableError(
                "template training backend is not configured; run with the Linux integration",
            )
        artifact = self._backend(spec)
        if artifact.slot != spec.slot:
            raise ModelContractError("template backend returned an artifact for the wrong hand/view slot")
        if artifact.roi_version != spec.roi_version or artifact.roi_digest != spec.roi_digest:
            raise ModelContractError("template backend returned an artifact with incompatible ROI provenance")
        if (
            artifact.dataset_release_id != spec.dataset_release_id
            or artifact.dataset_manifest_digest != spec.dataset_manifest_digest
            or artifact.train_split_id != spec.train_split_id
            or artifact.recipe_digest != spec.recipe_digest
        ):
            raise ModelContractError("template backend returned an artifact with incompatible dataset provenance")
        if (
            dict(artifact.training_parameters) != dict(spec.parameters)
            or dict(artifact.execution_receipt) != dict(spec.execution_receipt)
        ):
            raise ModelContractError(
                "template backend returned an artifact with incompatible execution provenance"
            )
        return artifact
