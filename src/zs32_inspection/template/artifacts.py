# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed template assets and single-threshold records."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from zs32_inspection.models.base import (
    AssetFile,
    ModelContractError,
    ModelSlot,
    freeze_json_mapping,
    require_sha256,
    require_text,
)


@dataclass(frozen=True, slots=True)
class TemplateAssetSpec:
    """Hashed template matcher assets for one hand/view."""

    slot: ModelSlot
    model: AssetFile
    templates: tuple[AssetFile, ...]
    model_digest: str
    template_version: str
    roi_version: str
    roi_digest: str
    dataset_release_id: str
    dataset_manifest_digest: str
    train_split_id: str
    recipe_digest: str
    framework_version: str
    training_parameters: Mapping[str, Any]
    execution_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.slot, ModelSlot):
            raise ModelContractError("template slot must be a ModelSlot")
        if not isinstance(self.model, AssetFile):
            raise ModelContractError("template model must be an AssetFile")
        object.__setattr__(self, "templates", tuple(self.templates))
        if any(not isinstance(asset, AssetFile) for asset in self.templates):
            raise ModelContractError("reference templates must contain only AssetFile values")
        object.__setattr__(self, "model_digest", require_sha256(self.model_digest, "template.model_digest"))
        object.__setattr__(
            self,
            "template_version",
            require_text(self.template_version, "template.template_version"),
        )
        object.__setattr__(self, "roi_version", require_text(self.roi_version, "template.roi_version"))
        object.__setattr__(self, "roi_digest", require_sha256(self.roi_digest, "template.roi_digest"))
        object.__setattr__(
            self,
            "dataset_release_id",
            require_text(self.dataset_release_id, "template.dataset_release_id"),
        )
        object.__setattr__(self, "train_split_id", require_text(self.train_split_id, "template.train_split_id"))
        object.__setattr__(
            self,
            "dataset_manifest_digest",
            require_sha256(self.dataset_manifest_digest, "template.dataset_manifest_digest"),
        )
        object.__setattr__(self, "recipe_digest", require_sha256(self.recipe_digest, "template.recipe_digest"))
        object.__setattr__(
            self,
            "framework_version",
            require_text(self.framework_version, "template.framework_version"),
        )
        for field in ("training_parameters", "execution_receipt"):
            object.__setattr__(
                self,
                field,
                freeze_json_mapping(getattr(self, field), f"template.{field}"),
            )
        from zs32_inspection.runtime.execution_receipt import ExecutionReceipt

        receipt = ExecutionReceipt.from_mapping(self.execution_receipt)
        if receipt.operation != "train_template":
            raise ModelContractError("template asset requires a train_template receipt")
        if self.model_digest != self.model.sha256:
            raise ModelContractError("template model_digest must equal model asset digest")
        if self.model.role != "template_model":
            raise ModelContractError("template model asset role must be 'template_model'")
        if not self.templates:
            raise ModelContractError("template asset group must contain at least one reference template")
        if any(asset.role != "reference_template" for asset in self.templates):
            raise ModelContractError("all reference template asset roles must be 'reference_template'")
        paths = [asset.path for asset in self.templates]
        if len(paths) != len(set(paths)):
            raise ModelContractError("reference template paths must be unique within a slot")

    def verify(self) -> None:
        """Verify all assets before OpenCV or another matcher reads them."""
        self.model.verify()
        for template in self.templates:
            template.verify()


@dataclass(frozen=True, slots=True)
class TemplateThresholdFit:
    """Candidate single threshold, including explicit non-deployable status."""

    slot: ModelSlot
    model_digest: str
    roi_version: str
    roi_digest: str
    dataset_release_id: str
    calibration_split_id: str
    threshold: float | None
    normal_count: int
    defect_count: int
    status: str

    def __post_init__(self) -> None:
        if not isinstance(self.slot, ModelSlot):
            raise ModelContractError("template threshold slot must be a ModelSlot")
        object.__setattr__(self, "model_digest", require_sha256(self.model_digest, "threshold.model_digest"))
        object.__setattr__(self, "roi_version", require_text(self.roi_version, "threshold.roi_version"))
        object.__setattr__(self, "roi_digest", require_sha256(self.roi_digest, "threshold.roi_digest"))
        object.__setattr__(
            self,
            "dataset_release_id",
            require_text(self.dataset_release_id, "threshold.dataset_release_id"),
        )
        object.__setattr__(
            self,
            "calibration_split_id",
            require_text(self.calibration_split_id, "threshold.calibration_split_id"),
        )
        if (
            isinstance(self.normal_count, bool)
            or not isinstance(self.normal_count, int)
            or isinstance(self.defect_count, bool)
            or not isinstance(self.defect_count, int)
            or self.normal_count < 0
            or self.defect_count < 0
        ):
            raise ModelContractError("template threshold counts must be non-negative")
        if self.status == "ok":
            if self.normal_count == 0 or self.defect_count == 0:
                raise ModelContractError("valid template threshold requires normal and defect parts")
            if (
                self.threshold is None
                or isinstance(self.threshold, bool)
                or not isinstance(self.threshold, (int, float))
                or not math.isfinite(self.threshold)
            ):
                raise ModelContractError("valid template threshold must be finite")
        elif self.status == "insufficient_data":
            if self.threshold is not None:
                raise ModelContractError("insufficient template threshold must not contain a threshold")
        else:
            raise ModelContractError(f"unsupported template threshold status: {self.status!r}")

    def to_dict(self) -> dict[str, object]:
        """Return the canonical threshold record."""
        return {
            "hand": self.slot.hand,
            "view": self.slot.view,
            "model_digest": self.model_digest,
            "roi_version": self.roi_version,
            "roi_digest": self.roi_digest,
            "dataset_release_id": self.dataset_release_id,
            "calibration_split_id": self.calibration_split_id,
            "threshold": self.threshold,
            "normal_count": self.normal_count,
            "defect_count": self.defect_count,
            "status": self.status,
        }


def validate_template_assets(
    assets: Mapping[ModelSlot, TemplateAssetSpec],
    required_slots: Sequence[ModelSlot],
) -> None:
    """Validate dynamic-topology coverage and exact hand/view keys."""
    expected = set(required_slots)
    actual = set(assets)
    if actual != expected:
        missing = sorted(slot.key for slot in expected - actual)
        unexpected = sorted(slot.key for slot in actual - expected)
        raise ModelContractError(f"template asset slot mismatch; missing={missing}, unexpected={unexpected}")
    for slot, asset in assets.items():
        if slot != asset.slot:
            raise ModelContractError(f"template asset mapping key does not match slot: {slot.key}")
        asset.verify()
