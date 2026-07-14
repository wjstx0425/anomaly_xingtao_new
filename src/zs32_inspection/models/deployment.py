# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Complete model/template/calibration asset description for release assembly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from .base import AssetFile, ModelContractError, ModelSlot, canonical_sha256, require_sha256
from .registry import ModelCandidate

if TYPE_CHECKING:
    from zs32_inspection.template.artifacts import TemplateAssetSpec


@dataclass(frozen=True, slots=True)
class DeploymentAssetDescription:
    """All Phase 4-5 files a release assembler must copy and re-hash.

    This is a description, not a promoted release. It has no method that can
    change a production pointer or overwrite an existing release directory.
    """

    candidate: ModelCandidate
    required_slots: tuple[ModelSlot, ...]
    template_assets: Mapping[ModelSlot, TemplateAssetSpec]
    template_thresholds: AssetFile
    model_thresholds: AssetFile
    calibration_metrics: AssetFile
    calibration_input_provenance: AssetFile
    calibration_artifact: AssetFile
    calibration_artifact_digest: str

    def __post_init__(self) -> None:
        required = tuple(self.required_slots)
        if not required or len(set(required)) != len(required):
            raise ModelContractError("deployment required_slots must be non-empty and unique")
        object.__setattr__(self, "required_slots", required)
        object.__setattr__(self, "template_assets", MappingProxyType(dict(self.template_assets)))
        object.__setattr__(
            self,
            "calibration_artifact_digest",
            require_sha256(self.calibration_artifact_digest, "deployment.calibration_artifact_digest"),
        )
        self.candidate.validate_slots(required)
        if set(self.template_assets) != set(required):
            missing = sorted(slot.key for slot in set(required) - set(self.template_assets))
            unexpected = sorted(slot.key for slot in set(self.template_assets) - set(required))
            raise ModelContractError(f"deployment template slot mismatch; missing={missing}, unexpected={unexpected}")
        for slot, template in self.template_assets.items():
            if slot != template.slot:
                raise ModelContractError(f"template mapping key does not match descriptor: {slot.key}")
            anomaly = self.candidate.anomaly_artifacts[slot]
            if template.roi_digest != anomaly.roi_digest or template.roi_version != anomaly.roi_version:
                raise ModelContractError(f"template/anomaly ROI identity mismatch for {slot.key}")
            if (
                template.dataset_release_id != self.candidate.dataset_release_id
                or template.dataset_manifest_digest != self.candidate.dataset_manifest_digest
                or template.recipe_digest != self.candidate.recipe_digest
            ):
                raise ModelContractError(f"template candidate provenance mismatch for {slot.key}")
        roles = {
            "template_thresholds": self.template_thresholds,
            "model_thresholds": self.model_thresholds,
            "calibration_metrics": self.calibration_metrics,
            "calibration_input_provenance": self.calibration_input_provenance,
            "calibration_artifact": self.calibration_artifact,
        }
        for role, asset in roles.items():
            if asset.role != role:
                raise ModelContractError(f"deployment calibration asset role must be {role!r}")
        if self.calibration_artifact.sha256 != self.calibration_artifact_digest:
            raise ModelContractError(
                "calibration_artifact_digest must equal the canonical calibration artifact file digest"
            )

    @property
    def descriptor_digest(self) -> str:
        """Return the canonical release-input descriptor digest."""
        return canonical_sha256(self.to_dict())

    def verify(self) -> None:
        """Verify every byte before a release assembler starts copying files."""
        for artifact in self.candidate.anomaly_artifacts.values():
            artifact.verify()
        self.candidate.yolo.verify()
        for template in self.template_assets.values():
            template.verify()
        self.template_thresholds.verify()
        self.model_thresholds.verify()
        self.calibration_metrics.verify()
        self.calibration_input_provenance.verify()
        self.calibration_artifact.verify()

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic deployment-input manifest."""
        return {
            "schema": "zs32.deployment_assets",
            "schema_version": 1,
            "candidate_id": self.candidate.candidate_id,
            "candidate_digest": self.candidate.digest,
            "required_slots": [slot.key for slot in self.required_slots],
            "templates": [
                {
                    "hand": slot.hand,
                    "view": slot.view,
                    "model": self.template_assets[slot].model.to_dict(),
                    "templates": [asset.to_dict() for asset in self.template_assets[slot].templates],
                    "model_digest": self.template_assets[slot].model_digest,
                    "template_version": self.template_assets[slot].template_version,
                    "roi_version": self.template_assets[slot].roi_version,
                    "roi_digest": self.template_assets[slot].roi_digest,
                    "dataset_release_id": self.template_assets[slot].dataset_release_id,
                    "dataset_manifest_digest": self.template_assets[slot].dataset_manifest_digest,
                    "train_split_id": self.template_assets[slot].train_split_id,
                    "recipe_digest": self.template_assets[slot].recipe_digest,
                }
                for slot in sorted(self.template_assets, key=lambda value: (value.hand, value.view))
            ],
            "calibration": {
                "template_thresholds": self.template_thresholds.to_dict(),
                "model_thresholds": self.model_thresholds.to_dict(),
                "metrics": self.calibration_metrics.to_dict(),
                "input_provenance": self.calibration_input_provenance.to_dict(),
                "artifact": self.calibration_artifact.to_dict(),
                "artifact_digest": self.calibration_artifact_digest,
            },
        }


def expected_asset_roles() -> Sequence[str]:
    """Expose the fixed Phase 5 calibration asset roles to release validation."""
    return (
        "template_thresholds",
        "model_thresholds",
        "calibration_metrics",
        "calibration_input_provenance",
        "calibration_artifact",
    )
