# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed contracts shared by ZS32 model adapters.

This module deliberately contains no torch, anomalib, or Ultralytics imports.
Third-party model construction belongs behind an injected Linux runtime
boundary; asset and identity validation happens before that boundary is called.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from zs32_inspection.domain.contracts import AnomalyFamily
from zs32_inspection.domain.errors import ZS32ContractError
from zs32_inspection.domain.evidence import EvidenceBranch
from zs32_inspection.domain.identity import Hand, RoiSample

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ModelContractError(ZS32ContractError):
    """A model asset, identity, or score violates the deployment contract."""


class BackendUnavailableError(RuntimeError):
    """The Linux model backend has not been provided to an adapter."""


def require_text(value: object, field: str) -> str:
    """Return a stripped non-empty string."""
    text = "" if value is None else str(value).strip()
    if not text:
        raise ModelContractError(f"{field} must be non-empty")
    return text


def require_sha256(value: object, field: str) -> str:
    """Return a normalized hexadecimal SHA-256 digest."""
    digest = require_text(value, field).lower()
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise ModelContractError(f"{field} must be a 64-character lowercase SHA-256")
    return digest


def sha256_file(path: Path) -> str:
    """Hash one file without loading a model or deserializing its contents."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    """Hash a deterministic JSON-compatible representation."""
    import json

    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def freeze_json_mapping(value: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    """Copy, JSON-validate, and freeze provenance/configuration parameters."""
    if not isinstance(value, Mapping):
        raise ModelContractError(f"{field} must be a mapping")
    copied = dict(value)
    try:
        canonical_sha256(copied)
    except (TypeError, ValueError) as error:
        raise ModelContractError(f"{field} must contain finite JSON-serializable values") from error
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class ModelSlot:
    """Identity of one hand/view-specific anomaly model."""

    hand: str
    view: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "hand", Hand.parse(self.hand).value)
        object.__setattr__(self, "view", require_text(self.view, "slot.view"))

    @property
    def key(self) -> str:
        """Return the stable serialized slot key."""
        return f"{self.hand}/{self.view}"


@dataclass(frozen=True, slots=True)
class AssetFile:
    """One content-addressed file crossing an adapter load boundary."""

    role: str
    path: Path
    sha256: str
    media_type: str = "application/octet-stream"
    logical_path: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", require_text(self.role, "asset.role"))
        object.__setattr__(self, "path", Path(self.path).expanduser())
        object.__setattr__(self, "sha256", require_sha256(self.sha256, "asset.sha256"))
        object.__setattr__(self, "media_type", require_text(self.media_type, "asset.media_type"))
        logical_path = (
            self.path.name
            if not self.logical_path
            else require_text(self.logical_path, "asset.logical_path")
        )
        relative = PurePosixPath(logical_path)
        if relative.as_posix() in {"", "."} or relative.is_absolute() or ".." in relative.parts:
            raise ModelContractError(f"asset.logical_path must stay inside a deployment bundle: {logical_path!r}")
        object.__setattr__(self, "logical_path", relative.as_posix())

    def verify(self) -> None:
        """Fail before model loading if the file is missing or changed."""
        if not self.path.is_file():
            raise ModelContractError(f"required {self.role} asset does not exist: {self.path}")
        actual = sha256_file(self.path)
        if actual != self.sha256:
            raise ModelContractError(
                f"{self.role} asset SHA-256 mismatch: expected {self.sha256}, found {actual}: {self.path}",
            )

    def to_dict(self) -> dict[str, str]:
        """Return an auditable descriptor without reading the file."""
        return {
            "role": self.role,
            "relative_path": self.logical_path,
            "sha256": self.sha256,
            "media_type": self.media_type,
        }


@dataclass(frozen=True, slots=True)
class DeviceSpec:
    """Runtime device request passed to an injected Linux backend."""

    accelerator: str = "gpu"
    device: str = "0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "accelerator", require_text(self.accelerator, "device.accelerator"))
        object.__setattr__(self, "device", require_text(self.device, "device.device"))


@dataclass(frozen=True, slots=True)
class ModelInput:
    """Filesystem binding for the canonical domain ROI sample.

    The crop geometry and identity stay owned by ``domain.RoiSample``. Model
    adapters receive only its already-cropped file and may resize/normalize it,
    but may not apply another spatial crop.
    """

    sample: RoiSample
    crop_path: Path
    roi_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.sample, RoiSample):
            raise ModelContractError("model input sample must be a canonical domain.RoiSample")
        object.__setattr__(self, "crop_path", Path(self.crop_path).expanduser())
        object.__setattr__(self, "roi_digest", require_sha256(self.roi_digest, "model_input.roi_digest"))

    @property
    def slot(self) -> ModelSlot:
        """Return this sample's anomaly-model slot."""
        return ModelSlot(self.sample.part.hand.value, self.sample.view_id)

    def verify_crop(self) -> None:
        """Verify the canonical crop bytes immediately before model inference."""
        if self.crop_path.is_symlink() or not self.crop_path.is_file():
            raise ModelContractError(
                f"canonical ROI crop must be a regular non-symlink file: {self.crop_path}"
            )
        actual = sha256_file(self.crop_path)
        if actual != self.sample.crop_sha256:
            raise ModelContractError(
                f"canonical ROI crop SHA-256 mismatch: expected {self.sample.crop_sha256}, found {actual}",
            )


@dataclass(frozen=True, slots=True)
class YoloDetection:
    """Normalized single-class detection emitted by the global YOLO backend."""

    class_id: int
    class_name: str
    confidence: float
    xyxy_norm: tuple[float, float, float, float]
    area_ratio: float
    clipped: bool

    def __post_init__(self) -> None:
        if isinstance(self.class_id, bool) or self.class_id != 0 or self.class_name != "defect":
            raise ModelContractError("ZS32 YOLO detections must use class_id=0 and class_name='defect'")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(self.confidence)
            or not 0 <= self.confidence <= 1
        ):
            raise ModelContractError("YOLO detection confidence must be finite and in [0, 1]")
        coordinates = tuple(self.xyxy_norm)
        if len(coordinates) != 4 or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in coordinates
        ):
            raise ModelContractError("YOLO xyxy_norm must contain four finite numbers")
        x1, y1, x2, y2 = (float(value) for value in coordinates)
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ModelContractError("YOLO xyxy_norm must be an ordered box inside [0, 1]")
        object.__setattr__(self, "xyxy_norm", (x1, y1, x2, y2))
        expected_area = (x2 - x1) * (y2 - y1)
        if (
            isinstance(self.area_ratio, bool)
            or not isinstance(self.area_ratio, (int, float))
            or not math.isfinite(self.area_ratio)
            or not math.isclose(float(self.area_ratio), expected_area, rel_tol=1e-9, abs_tol=1e-12)
        ):
            raise ModelContractError("YOLO area_ratio must equal the normalized box area")
        if not isinstance(self.clipped, bool):
            raise ModelContractError("YOLO detection clipped must be boolean")
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "area_ratio", float(self.area_ratio))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> YoloDetection:
        """Reject unknown/missing backend detection fields before fusion."""
        expected = {"class_id", "class_name", "confidence", "xyxy_norm", "area_ratio", "clipped"}
        if set(payload) != expected:
            raise ModelContractError(
                f"YOLO detection keys invalid; missing={sorted(expected - set(payload))}, "
                f"unknown={sorted(set(payload) - expected)}"
            )
        xyxy = payload["xyxy_norm"]
        if isinstance(xyxy, (str, bytes)) or not isinstance(xyxy, Sequence):
            raise ModelContractError("YOLO detection xyxy_norm must be an array")
        return cls(
            class_id=payload["class_id"],
            class_name=payload["class_name"],
            confidence=payload["confidence"],
            xyxy_norm=tuple(xyxy),
            area_ratio=payload["area_ratio"],
            clipped=payload["clipped"],
        )

    def to_dict(self) -> dict[str, object]:
        """Return the frozen normalized detection schema."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "xyxy_norm": list(self.xyxy_norm),
            "area_ratio": self.area_ratio,
            "clipped": self.clipped,
        }


@dataclass(frozen=True, slots=True)
class RawModelScore:
    """Uncalibrated continuous score before domain evidence construction."""

    inspection_id: str
    part_instance_id: str
    capture_set_id: str
    hand: str
    view: str
    branch: str
    score: float
    model_family: str
    model_digest: str
    roi_config_id: str
    roi_digest: str
    source_sha256: str
    crop_sha256: str
    heatmap_path: Path | None = None
    heatmap_sha256: str | None = None
    overlay_path: Path | None = None
    overlay_sha256: str | None = None
    detections: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        for field in (
            "part_instance_id",
            "capture_set_id",
            "inspection_id",
            "hand",
            "view",
            "branch",
            "model_family",
            "roi_config_id",
        ):
            object.__setattr__(self, field, require_text(getattr(self, field), f"raw_score.{field}"))
        object.__setattr__(self, "hand", Hand.parse(self.hand).value)
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)) or not math.isfinite(self.score):
            raise ModelContractError("raw_score.score must be finite")
        try:
            branch = EvidenceBranch(self.branch)
        except ValueError as error:
            raise ModelContractError(f"raw_score.branch must be anomaly or yolo, got {self.branch!r}") from error
        object.__setattr__(self, "branch", branch.value)
        if branch is EvidenceBranch.YOLO:
            if self.model_family != "yolo":
                raise ModelContractError("raw YOLO score must identify model_family='yolo'")
            if not 0 <= self.score <= 1:
                raise ModelContractError("raw YOLO confidence score must be in [0, 1]")
        elif self.model_family not in {family.value for family in AnomalyFamily}:
            raise ModelContractError(f"raw anomaly score has unsupported model_family {self.model_family!r}")
        object.__setattr__(self, "model_digest", require_sha256(self.model_digest, "raw_score.model_digest"))
        object.__setattr__(self, "roi_digest", require_sha256(self.roi_digest, "raw_score.roi_digest"))
        object.__setattr__(self, "source_sha256", require_sha256(self.source_sha256, "raw_score.source_sha256"))
        object.__setattr__(self, "crop_sha256", require_sha256(self.crop_sha256, "raw_score.crop_sha256"))
        if self.heatmap_path is None and self.heatmap_sha256 is not None:
            raise ModelContractError("heatmap_sha256 requires heatmap_path")
        if self.heatmap_path is not None:
            object.__setattr__(self, "heatmap_path", Path(self.heatmap_path).expanduser())
            object.__setattr__(
                self,
                "heatmap_sha256",
                require_sha256(self.heatmap_sha256, "evidence.heatmap_sha256"),
            )
        if self.overlay_path is None and self.overlay_sha256 is not None:
            raise ModelContractError("overlay_sha256 requires overlay_path")
        if self.overlay_path is not None:
            object.__setattr__(self, "overlay_path", Path(self.overlay_path).expanduser())
            object.__setattr__(
                self,
                "overlay_sha256",
                require_sha256(self.overlay_sha256, "evidence.overlay_sha256"),
            )
        detections = tuple(self.detections)
        if any(not isinstance(item, Mapping) for item in detections):
            raise ModelContractError("raw_score.detections must contain mappings")
        if branch is EvidenceBranch.ANOMALY and detections:
            raise ModelContractError("raw anomaly scores must not contain YOLO detections")
        if branch is EvidenceBranch.YOLO:
            normalized = tuple(YoloDetection.from_mapping(item) for item in detections)
            expected_score = max((item.confidence for item in normalized), default=0.0)
            if not math.isclose(self.score, expected_score, rel_tol=0, abs_tol=1e-12):
                raise ModelContractError(
                    "raw YOLO score must equal maximum detection confidence (or 0 when empty)"
                )
            detections = tuple(MappingProxyType(item.to_dict()) for item in normalized)
        object.__setattr__(self, "detections", detections)


@dataclass(frozen=True, slots=True)
class TrainSpec:
    """Immutable request for one hand/view anomaly training job."""

    family: AnomalyFamily
    slot: ModelSlot
    dataset_release_id: str
    dataset_manifest_digest: str
    train_split_id: str
    recipe_digest: str
    roi_version: str
    roi_digest: str
    materialized_export_root: Path
    materialized_manifest_digest: str
    device: DeviceSpec
    output_dir: Path
    parameters: Mapping[str, Any]
    execution_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", AnomalyFamily(self.family))
        if not isinstance(self.slot, ModelSlot):
            raise ModelContractError("train.slot must be a ModelSlot")
        for field in ("dataset_release_id", "train_split_id", "roi_version"):
            object.__setattr__(self, field, require_text(getattr(self, field), f"train.{field}"))
        for field in (
            "dataset_manifest_digest",
            "recipe_digest",
            "roi_digest",
            "materialized_manifest_digest",
        ):
            object.__setattr__(self, field, require_sha256(getattr(self, field), f"train.{field}"))
        object.__setattr__(
            self,
            "materialized_export_root",
            Path(self.materialized_export_root).expanduser(),
        )
        if not isinstance(self.device, DeviceSpec):
            raise ModelContractError("train.device must be a DeviceSpec")
        object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser())
        object.__setattr__(self, "parameters", freeze_json_mapping(self.parameters, "train.parameters"))
        object.__setattr__(
            self,
            "execution_receipt",
            freeze_json_mapping(self.execution_receipt, "train.execution_receipt"),
        )
        from zs32_inspection.runtime.execution_receipt import ExecutionReceipt

        receipt = ExecutionReceipt.from_mapping(self.execution_receipt)
        if receipt.operation != "train_anomaly":
            raise ModelContractError("anomaly TrainSpec requires a train_anomaly receipt")


@dataclass(frozen=True, slots=True)
class AnomalyModelArtifact:
    """Content-addressed anomaly checkpoint plus complete training provenance."""

    family: AnomalyFamily
    slot: ModelSlot
    checkpoint: AssetFile
    model_digest: str
    dataset_release_id: str
    dataset_manifest_digest: str
    train_split_id: str
    recipe_digest: str
    roi_version: str
    roi_digest: str
    framework_version: str
    training_parameters: Mapping[str, Any]
    execution_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", AnomalyFamily(self.family))
        if not isinstance(self.slot, ModelSlot):
            raise ModelContractError("artifact.slot must be a ModelSlot")
        if not isinstance(self.checkpoint, AssetFile):
            raise ModelContractError("artifact.checkpoint must be an AssetFile")
        object.__setattr__(self, "model_digest", require_sha256(self.model_digest, "artifact.model_digest"))
        if self.model_digest != self.checkpoint.sha256:
            raise ModelContractError("artifact.model_digest must equal the checkpoint content digest")
        if self.checkpoint.role != "checkpoint":
            raise ModelContractError("anomaly checkpoint asset role must be 'checkpoint'")
        for field in ("dataset_release_id", "train_split_id", "roi_version", "framework_version"):
            object.__setattr__(self, field, require_text(getattr(self, field), f"artifact.{field}"))
        for field in ("dataset_manifest_digest", "recipe_digest", "roi_digest"):
            object.__setattr__(self, field, require_sha256(getattr(self, field), f"artifact.{field}"))
        object.__setattr__(
            self,
            "training_parameters",
            freeze_json_mapping(self.training_parameters, "artifact.training_parameters"),
        )
        object.__setattr__(
            self,
            "execution_receipt",
            freeze_json_mapping(self.execution_receipt, "artifact.execution_receipt"),
        )
        from zs32_inspection.runtime.execution_receipt import ExecutionReceipt

        receipt = ExecutionReceipt.from_mapping(self.execution_receipt)
        if receipt.operation != "train_anomaly":
            raise ModelContractError("anomaly artifact requires a train_anomaly receipt")

    def verify(self) -> None:
        """Verify checkpoint bytes before passing the artifact to a backend."""
        self.checkpoint.verify()

    def to_dict(self) -> dict[str, object]:
        """Serialize the stable artifact contract for a registry or release."""
        return {
            "family": self.family.value,
            "hand": self.slot.hand,
            "view": self.slot.view,
            "checkpoint": self.checkpoint.to_dict(),
            "model_digest": self.model_digest,
            "dataset_release_id": self.dataset_release_id,
            "dataset_manifest_digest": self.dataset_manifest_digest,
            "train_split_id": self.train_split_id,
            "recipe_digest": self.recipe_digest,
            "roi_version": self.roi_version,
            "roi_digest": self.roi_digest,
            "framework_version": self.framework_version,
            "training_parameters": dict(self.training_parameters),
            "execution_receipt": dict(self.execution_receipt),
        }


@runtime_checkable
class AnomalyPredictor(Protocol):
    """Loaded backend that preserves sample identity and emits raw scores."""

    def predict_batch(
        self,
        samples: Sequence[ModelInput],
        *,
        inspection_id: str,
    ) -> Sequence[RawModelScore]:
        """Predict canonical crops without thresholding, fusion, or recropping."""


@runtime_checkable
class AnomalyRuntimeLoader(Protocol):
    """Linux-only boundary implemented by a concrete framework integration."""

    def __call__(
        self,
        artifacts: Mapping[ModelSlot, AnomalyModelArtifact],
        device: DeviceSpec,
    ) -> AnomalyPredictor:
        """Restore already-verified checkpoints."""


@runtime_checkable
class AnomalyTrainer(Protocol):
    """Linux-only boundary for one anomaly training job."""

    def __call__(self, spec: TrainSpec) -> AnomalyModelArtifact:
        """Train without changing any production release or recipe pointer."""


@runtime_checkable
class AnomalyAdapter(Protocol):
    """Uniform plugin contract for the three supported anomaly families."""

    family: AnomalyFamily

    def train(self, spec: TrainSpec) -> AnomalyModelArtifact:
        """Run one explicitly requested training job."""

    def load(
        self,
        artifacts: Mapping[ModelSlot, AnomalyModelArtifact],
        device: DeviceSpec,
    ) -> AnomalyPredictor:
        """Verify all slots and restore a predictor through the runtime boundary."""


class DelegatingAnomalyAdapter:
    """Validation-heavy adapter base with explicitly injected real backends."""

    family: AnomalyFamily

    def __init__(
        self,
        *,
        runtime_loader: AnomalyRuntimeLoader | None = None,
        trainer: AnomalyTrainer | None = None,
    ) -> None:
        self._runtime_loader = runtime_loader
        self._trainer = trainer

    def train(self, spec: TrainSpec) -> AnomalyModelArtifact:
        """Delegate training only after the family contract is verified."""
        if spec.family is not self.family:
            raise ModelContractError(
                f"{self.family.value} adapter cannot train family {spec.family.value}",
            )
        if self._trainer is None:
            raise BackendUnavailableError(
                f"{self.family.value} training backend is not configured; run with the Linux GPU integration",
            )
        artifact = self._trainer(spec)
        if artifact.family is not self.family or artifact.slot != spec.slot:
            raise ModelContractError("training backend returned an artifact with the wrong family or slot")
        if (
            artifact.recipe_digest != spec.recipe_digest
            or artifact.roi_digest != spec.roi_digest
            or artifact.roi_version != spec.roi_version
            or artifact.dataset_release_id != spec.dataset_release_id
            or artifact.dataset_manifest_digest != spec.dataset_manifest_digest
            or artifact.train_split_id != spec.train_split_id
            or dict(artifact.execution_receipt) != dict(spec.execution_receipt)
        ):
            raise ModelContractError("training backend returned an artifact with incompatible provenance")
        return artifact

    def load(
        self,
        artifacts: Mapping[ModelSlot, AnomalyModelArtifact],
        device: DeviceSpec,
    ) -> AnomalyPredictor:
        """Verify checkpoint identity and bytes before third-party deserialization."""
        if not artifacts:
            raise ModelContractError("at least one anomaly artifact is required")
        if any(
            not isinstance(key, ModelSlot) or not isinstance(value, AnomalyModelArtifact)
            for key, value in artifacts.items()
        ):
            raise ModelContractError("anomaly artifact mapping must contain ModelSlot/AnomalyModelArtifact pairs")
        provenance_signatures = {
            (
                artifact.dataset_release_id,
                artifact.dataset_manifest_digest,
                artifact.recipe_digest,
                artifact.roi_version,
                artifact.roi_digest,
            )
            for artifact in artifacts.values()
        }
        if len(provenance_signatures) != 1:
            raise ModelContractError("anomaly artifacts do not share one dataset/recipe/ROI provenance")
        for key, artifact in artifacts.items():
            if key != artifact.slot:
                raise ModelContractError(f"artifact mapping key does not match slot: {key.key}")
            if artifact.family is not self.family:
                raise ModelContractError(
                    f"{self.family.value} release cannot contain {artifact.family.value} artifact {key.key}",
                )
            artifact.verify()
        if self._runtime_loader is None:
            raise BackendUnavailableError(
                f"{self.family.value} runtime backend is not configured; run with the Linux GPU integration",
            )
        return self._runtime_loader(artifacts, device)


def validate_required_slots(
    artifacts: Mapping[ModelSlot, AnomalyModelArtifact],
    required_slots: Sequence[ModelSlot],
) -> None:
    """Require exact dynamic-topology coverage without a six-view constant."""
    required = set(required_slots)
    actual = set(artifacts)
    if actual != required:
        missing = sorted(slot.key for slot in required - actual)
        unexpected = sorted(slot.key for slot in actual - required)
        raise ModelContractError(f"anomaly artifact slot mismatch; missing={missing}, unexpected={unexpected}")
