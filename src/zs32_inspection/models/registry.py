# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Immutable candidate-registry records for ZS32 model promotion workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType

from .base import (
    AnomalyFamily,
    AnomalyModelArtifact,
    ModelContractError,
    ModelSlot,
    canonical_sha256,
    require_sha256,
    require_text,
    validate_required_slots,
)
from .yolo import YoloDeploymentSpec


class CandidateStatus(str, Enum):
    """Offline lifecycle; registration never implies production promotion."""

    REGISTERED = "registered"
    VALIDATED = "validated"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class ModelCandidate:
    """A content-addressed candidate, not a mutable production pointer."""

    candidate_id: str
    product: str
    anomaly_family: AnomalyFamily
    anomaly_artifacts: Mapping[ModelSlot, AnomalyModelArtifact]
    yolo: YoloDeploymentSpec
    recipe_digest: str
    topology_digest: str
    roi_digest: str
    roi_version: str
    dataset_release_id: str
    dataset_manifest_digest: str
    status: CandidateStatus = CandidateStatus.REGISTERED

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", require_text(self.candidate_id, "candidate.candidate_id"))
        object.__setattr__(self, "product", require_text(self.product, "candidate.product"))
        object.__setattr__(self, "anomaly_family", AnomalyFamily(self.anomaly_family))
        object.__setattr__(
            self,
            "dataset_release_id",
            require_text(self.dataset_release_id, "candidate.dataset_release_id"),
        )
        object.__setattr__(self, "roi_version", require_text(self.roi_version, "candidate.roi_version"))
        for field in ("recipe_digest", "topology_digest", "roi_digest", "dataset_manifest_digest"):
            object.__setattr__(self, field, require_sha256(getattr(self, field), f"candidate.{field}"))
        object.__setattr__(self, "status", CandidateStatus(self.status))
        if self.product != "ZS32":
            raise ModelContractError("model registry accepts only product ZS32")
        if not self.anomaly_artifacts:
            raise ModelContractError("candidate must contain anomaly artifacts")
        object.__setattr__(self, "anomaly_artifacts", MappingProxyType(dict(self.anomaly_artifacts)))
        if not isinstance(self.yolo, YoloDeploymentSpec):
            raise ModelContractError("candidate yolo must be a YoloDeploymentSpec")
        for slot, artifact in self.anomaly_artifacts.items():
            if slot != artifact.slot:
                raise ModelContractError(f"candidate anomaly mapping key does not match artifact slot: {slot.key}")
            if artifact.family is not self.anomaly_family:
                raise ModelContractError("one candidate cannot mix anomaly families")
            if (
                artifact.recipe_digest != self.recipe_digest
                or artifact.roi_digest != self.roi_digest
                or artifact.roi_version != self.roi_version
            ):
                raise ModelContractError("anomaly artifact provenance does not match candidate recipe/ROI")
            if (
                artifact.dataset_release_id != self.dataset_release_id
                or artifact.dataset_manifest_digest != self.dataset_manifest_digest
            ):
                raise ModelContractError("anomaly artifact dataset provenance does not match candidate")
        if (
            self.yolo.provenance.dataset_release_id != self.dataset_release_id
            or self.yolo.provenance.dataset_manifest_digest != self.dataset_manifest_digest
        ):
            raise ModelContractError("YOLO dataset provenance does not match candidate")
        model_digests = [artifact.model_digest for artifact in self.anomaly_artifacts.values()]
        if len(model_digests) != len(set(model_digests)):
            raise ModelContractError("each hand/view anomaly slot must use a distinct checkpoint digest")

    @property
    def digest(self) -> str:
        """Return the canonical candidate descriptor digest."""
        return canonical_sha256(self._content_payload())

    def validate_slots(self, required_slots: Sequence[ModelSlot]) -> None:
        """Require exactly one artifact for every compiled hand/view slot."""
        validate_required_slots(self.anomaly_artifacts, required_slots)

    def _content_payload(self) -> dict[str, object]:
        """Return content identity independent of registry ID/lifecycle status."""
        return {
            "product": self.product,
            "anomaly_family": self.anomaly_family.value,
            "anomaly_artifacts": [
                self.anomaly_artifacts[slot].to_dict()
                for slot in sorted(self.anomaly_artifacts, key=lambda value: (value.hand, value.view))
            ],
            "yolo": self.yolo.to_dict(),
            "recipe_digest": self.recipe_digest,
            "topology_digest": self.topology_digest,
            "roi_digest": self.roi_digest,
            "roi_version": self.roi_version,
            "dataset_release_id": self.dataset_release_id,
            "dataset_manifest_digest": self.dataset_manifest_digest,
        }

    def to_dict(self) -> dict[str, object]:
        """Return deterministic registry content."""
        return {
            "candidate_id": self.candidate_id,
            **self._content_payload(),
            "status": self.status.value,
            "candidate_digest": self.digest,
        }


def register_candidate(existing: Sequence[ModelCandidate], candidate: ModelCandidate) -> tuple[ModelCandidate, ...]:
    """Return a new registry snapshot without touching any production release."""
    if any(record.candidate_id == candidate.candidate_id for record in existing):
        raise ModelContractError(f"candidate_id already exists: {candidate.candidate_id}")
    if any(record.digest == candidate.digest for record in existing):
        raise ModelContractError(f"candidate content is already registered: {candidate.digest}")
    return (*existing, candidate)


def transition_candidate(
    candidate: ModelCandidate,
    target: CandidateStatus,
) -> ModelCandidate:
    """Return an immutable lifecycle record without changing content identity.

    Only an explicitly registered candidate may be accepted or rejected.  A
    validated/rejected record cannot be silently rewritten or rolled back.
    The caller remains responsible for proving calibration and held-out-test
    acceptance before requesting ``VALIDATED``.
    """
    target = CandidateStatus(target)
    if candidate.status is not CandidateStatus.REGISTERED:
        raise ModelContractError(
            f"candidate lifecycle is terminal after {candidate.status.value}: "
            f"{candidate.candidate_id}"
        )
    if target not in {CandidateStatus.VALIDATED, CandidateStatus.REJECTED}:
        raise ModelContractError(
            "registered candidate may transition only to validated or rejected"
        )
    transitioned = replace(candidate, status=target)
    if transitioned.digest != candidate.digest:
        raise ModelContractError("candidate lifecycle transition changed content digest")
    return transitioned
