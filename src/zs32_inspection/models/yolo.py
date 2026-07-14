# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed import and load boundary for the external YOLO model."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from .base import (
    AssetFile,
    BackendUnavailableError,
    DeviceSpec,
    ModelContractError,
    ModelInput,
    RawModelScore,
    canonical_sha256,
    require_sha256,
    require_text,
    sha256_file,
)
from .yolo_receipt import (
    load_yolo_training_receipt,
    parse_yolo_trainer_source,
)

RUN_SEED_PATTERN = re.compile(r"(?:^|[_-])seed(?P<seed>\d+)(?:$|[_-])", re.IGNORECASE)
YAML_SEED_PATTERN = re.compile(r"^seed\s*:\s*['\"]?(?P<seed>-?\d+)['\"]?\s*(?:#.*)?$", re.MULTILINE)
YAML_DATA_PATTERN = re.compile(
    r"^data\s*:\s*(?P<data>[^#\r\n]+?)\s*(?:#.*)?$",
    re.MULTILINE,
)
CLASS_NAME_PATTERN = re.compile(
    r"^\s*(?P<class_id>\d+)\s*:\s*['\"]?(?P<name>[^#'\"\r\n]+?)['\"]?\s*(?:#.*)?$",
    re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class YoloRuntimeSettings:
    """Frozen production settings, separate from external training arguments."""

    imgsz: int = 640
    candidate_conf: float = 0.001
    iou: float = 0.7
    max_det: int = 300
    single_class_name: str = "defect"

    def __post_init__(self) -> None:
        if (
            isinstance(self.imgsz, bool)
            or not isinstance(self.imgsz, int)
            or isinstance(self.max_det, bool)
            or not isinstance(self.max_det, int)
            or self.imgsz <= 0
            or self.max_det <= 0
        ):
            raise ModelContractError("YOLO imgsz and max_det must be positive")
        if (
            isinstance(self.candidate_conf, bool)
            or not isinstance(self.candidate_conf, (int, float))
            or not math.isfinite(self.candidate_conf)
            or not 0 < self.candidate_conf < 1
        ):
            raise ModelContractError("YOLO candidate_conf must be in (0, 1)")
        if (
            isinstance(self.iou, bool)
            or not isinstance(self.iou, (int, float))
            or not math.isfinite(self.iou)
            or not 0 < self.iou <= 1
        ):
            raise ModelContractError("YOLO iou must be in (0, 1]")
        object.__setattr__(
            self,
            "single_class_name",
            require_text(self.single_class_name, "yolo.single_class_name"),
        )
        if self.single_class_name != "defect":
            raise ModelContractError("ZS32 YOLO deployment class must be exactly 'defect'")

    def to_dict(self) -> dict[str, int | float | str]:
        """Return the production-only settings."""
        return {
            "imgsz": self.imgsz,
            "candidate_conf": self.candidate_conf,
            "iou": self.iou,
            "max_det": self.max_det,
            "single_class_name": self.single_class_name,
        }


@dataclass(frozen=True, slots=True)
class YoloTrainingProvenance:
    """External-training facts that are audited but never used as runtime paths."""

    dataset_release_id: str
    dataset_manifest_digest: str
    yolo_export_publication_id: str
    yolo_export_root_sha256: str
    yolo_export_manifest_digest: str
    yolo_export_data_yaml_digest: str
    yolo_export_policy_digest: str
    args_data_reference: str
    ultralytics_version_or_commit: str
    trainer_source: dict[str, str]
    training_receipt_digest: str
    training_seed: int
    run_id: str
    run_name: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dataset_release_id",
            require_text(self.dataset_release_id, "yolo.dataset_release_id"),
        )
        object.__setattr__(
            self,
            "yolo_export_data_yaml_digest",
            require_sha256(
                self.yolo_export_data_yaml_digest,
                "yolo.yolo_export_data_yaml_digest",
            ),
        )
        object.__setattr__(
            self,
            "yolo_export_policy_digest",
            require_sha256(
                self.yolo_export_policy_digest,
                "yolo.yolo_export_policy_digest",
            ),
        )
        object.__setattr__(
            self,
            "args_data_reference",
            require_text(self.args_data_reference, "yolo.args_data_reference"),
        )
        object.__setattr__(
            self,
            "dataset_manifest_digest",
            require_sha256(self.dataset_manifest_digest, "yolo.dataset_manifest_digest"),
        )
        object.__setattr__(
            self,
            "yolo_export_publication_id",
            require_text(self.yolo_export_publication_id, "yolo.yolo_export_publication_id"),
        )
        object.__setattr__(
            self,
            "yolo_export_root_sha256",
            require_sha256(self.yolo_export_root_sha256, "yolo.yolo_export_root_sha256"),
        )
        object.__setattr__(
            self,
            "yolo_export_manifest_digest",
            require_sha256(
                self.yolo_export_manifest_digest,
                "yolo.yolo_export_manifest_digest",
            ),
        )
        object.__setattr__(
            self,
            "ultralytics_version_or_commit",
            require_text(self.ultralytics_version_or_commit, "yolo.ultralytics_version_or_commit"),
        )
        source = parse_yolo_trainer_source(self.trainer_source)
        if self.ultralytics_version_or_commit != source.runtime_identity:
            raise ModelContractError(
                "YOLO runtime identity must equal the attested trainer source identity"
            )
        object.__setattr__(self, "trainer_source", source.to_dict())
        object.__setattr__(
            self,
            "training_receipt_digest",
            require_sha256(self.training_receipt_digest, "yolo.training_receipt_digest"),
        )
        object.__setattr__(self, "run_id", require_text(self.run_id, "yolo.run_id"))
        object.__setattr__(self, "run_name", require_text(self.run_name, "yolo.run_name"))
        if (
            isinstance(self.training_seed, bool)
            or not isinstance(self.training_seed, int)
            or self.training_seed < 0
        ):
            raise ModelContractError("YOLO training_seed must be non-negative")
        match = RUN_SEED_PATTERN.search(self.run_name)
        if match is not None and int(match.group("seed")) != self.training_seed:
            raise ModelContractError(
                f"YOLO run name seed {match.group('seed')} contradicts args.yaml seed {self.training_seed}",
            )

    def to_dict(self) -> dict[str, object]:
        """Return training provenance without leaking training-machine paths into runtime settings."""
        return {
            "dataset_release_id": self.dataset_release_id,
            "dataset_manifest_digest": self.dataset_manifest_digest,
            "yolo_export_publication_id": self.yolo_export_publication_id,
            "yolo_export_root_sha256": self.yolo_export_root_sha256,
            "yolo_export_manifest_digest": self.yolo_export_manifest_digest,
            "yolo_export_data_yaml_digest": self.yolo_export_data_yaml_digest,
            "yolo_export_policy_digest": self.yolo_export_policy_digest,
            "args_data_reference": self.args_data_reference,
            "ultralytics_version_or_commit": self.ultralytics_version_or_commit,
            "trainer_source": self.trainer_source,
            "training_receipt_digest": self.training_receipt_digest,
            "training_seed": self.training_seed,
            "run_id": self.run_id,
            "run_name": self.run_name,
        }


@dataclass(frozen=True, slots=True)
class YoloDeploymentSpec:
    """One global YOLO checkpoint for both hands and every topology view."""

    weights: AssetFile
    args_yaml: AssetFile
    data_yaml: AssetFile
    class_names_yaml: AssetFile
    training_receipt: AssetFile
    model_digest: str
    bundle_digest: str
    provenance: YoloTrainingProvenance
    runtime: YoloRuntimeSettings

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_digest", require_sha256(self.model_digest, "yolo.model_digest"))
        object.__setattr__(self, "bundle_digest", require_sha256(self.bundle_digest, "yolo.bundle_digest"))
        if self.weights.path.name != "best.pt":
            raise ModelContractError("YOLO deployment weights must be the trained best.pt, never yolo26n.pt")
        if self.model_digest != self.weights.sha256:
            raise ModelContractError("YOLO model_digest must equal best.pt content digest")
        expected_roles = {
            "weights": self.weights,
            "train_args": self.args_yaml,
            "dataset_config": self.data_yaml,
            "class_names": self.class_names_yaml,
            "training_receipt": self.training_receipt,
        }
        for role, asset in expected_roles.items():
            if asset.role != role:
                raise ModelContractError(f"YOLO asset role mismatch: expected {role!r}, found {asset.role!r}")
        if self.bundle_digest != canonical_sha256(self._bundle_payload()):
            raise ModelContractError("YOLO bundle_digest does not match the deployment descriptor")

    @property
    def assets(self) -> tuple[AssetFile, ...]:
        """Return every file that must be verified before Ultralytics loads weights."""
        return (
            self.weights,
            self.args_yaml,
            self.data_yaml,
            self.class_names_yaml,
            self.training_receipt,
        )

    def _bundle_payload(self) -> dict[str, object]:
        return {
            "schema": "zs32.yolo_deployment",
            "schema_version": 3,
            "assets": [asset.to_dict() for asset in self.assets],
            "model_digest": self.model_digest,
            "provenance": self.provenance.to_dict(),
            "runtime": self.runtime.to_dict(),
        }

    def verify(self) -> None:
        """Verify all bundle bytes before any unsafe checkpoint deserialization."""
        for asset in self.assets:
            asset.verify()
        seed = read_training_seed(self.args_yaml.path)
        if seed != self.provenance.training_seed:
            raise ModelContractError(
                f"YOLO args.yaml seed changed: expected {self.provenance.training_seed}, found {seed}",
            )
        if read_class_names(self.class_names_yaml.path) != ("defect",):
            raise ModelContractError("YOLO class_names.yaml must map exactly class 0 to 'defect'")
        if read_training_data_reference(self.args_yaml.path) != self.provenance.args_data_reference:
            raise ModelContractError("YOLO args.yaml data reference differs from training receipt")
        receipt = load_yolo_training_receipt(self.training_receipt.path)
        actual_receipt_assets = {
            "best.pt": self.weights.sha256,
            "args.yaml": self.args_yaml.sha256,
            "data.yaml": self.data_yaml.sha256,
            "class_names.yaml": self.class_names_yaml.sha256,
        }
        if receipt.artifact_sha256 != actual_receipt_assets:
            raise ModelContractError("packaged YOLO assets differ from training receipt")
        if receipt.provenance_dict() != self.provenance.to_dict():
            raise ModelContractError("packaged YOLO receipt differs from deployment provenance")

    def to_dict(self) -> dict[str, object]:
        """Return a release-ready asset descriptor."""
        return {**self._bundle_payload(), "bundle_digest": self.bundle_digest}


def read_training_seed(args_yaml: Path) -> int:
    """Read the unique top-level integer seed without importing a YAML runtime."""
    if not args_yaml.is_file():
        raise ModelContractError(f"YOLO args.yaml does not exist: {args_yaml}")
    matches = [
        int(match.group("seed"))
        for match in YAML_SEED_PATTERN.finditer(args_yaml.read_text(encoding="utf-8"))
    ]
    if len(matches) != 1 or matches[0] < 0:
        raise ModelContractError(
            "YOLO args.yaml must contain exactly one non-negative top-level integer seed"
        )
    return matches[0]


def read_class_names(class_names_yaml: Path) -> tuple[str, ...]:
    """Read an explicit integer-to-name mapping from the dedicated class file."""
    if not class_names_yaml.is_file():
        raise ModelContractError(f"YOLO class_names.yaml does not exist: {class_names_yaml}")
    parsed: dict[int, str] = {}
    for match in CLASS_NAME_PATTERN.finditer(class_names_yaml.read_text(encoding="utf-8")):
        class_id = int(match.group("class_id"))
        name = match.group("name").strip()
        if class_id in parsed:
            raise ModelContractError(f"YOLO class_names.yaml repeats class id {class_id}")
        parsed[class_id] = name
    if set(parsed) != {0}:
        raise ModelContractError("YOLO class_names.yaml must contain exactly one class with id 0")
    return (parsed[0],)


def read_training_data_reference(args_yaml: Path) -> str:
    """Read the one top-level data path attested by the external trainer."""
    if not args_yaml.is_file():
        raise ModelContractError(f"YOLO args.yaml does not exist: {args_yaml}")
    matches = [
        match.group("data").strip().strip("'\"")
        for match in YAML_DATA_PATTERN.finditer(args_yaml.read_text(encoding="utf-8"))
    ]
    if len(matches) != 1 or not matches[0]:
        raise ModelContractError("YOLO args.yaml must contain exactly one non-empty top-level data reference")
    return matches[0]


def _asset(bundle_dir: Path, name: str, role: str, media_type: str) -> AssetFile:
    path = bundle_dir / name
    if not path.is_file():
        raise ModelContractError(f"external YOLO bundle is missing {name}: {path}")
    return AssetFile(role=role, path=path, sha256=sha256_file(path), media_type=media_type)


def import_yolo_bundle(
    bundle_dir: Path,
    *,
    training_receipt_path: Path,
    runtime: YoloRuntimeSettings | None = None,
) -> YoloDeploymentSpec:
    """Describe and hash an external training result without copying or loading weights.

    The immutable release assembler owns copying these verified files into a
    candidate/release directory. This importer cannot mutate production state.
    """
    bundle_dir = Path(bundle_dir).expanduser()
    weights = _asset(bundle_dir, "best.pt", "weights", "application/x-pytorch")
    args_yaml = _asset(bundle_dir, "args.yaml", "train_args", "application/yaml")
    data_yaml = _asset(bundle_dir, "data.yaml", "dataset_config", "application/yaml")
    class_names_yaml = _asset(bundle_dir, "class_names.yaml", "class_names", "application/yaml")
    receipt = load_yolo_training_receipt(training_receipt_path)
    receipt.verify_bundle(bundle_dir)
    training_receipt = AssetFile(
        role="training_receipt",
        path=Path(training_receipt_path).expanduser(),
        sha256=receipt.receipt_sha256,
        media_type="application/json",
        logical_path="training_receipt.json",
    )
    if read_class_names(class_names_yaml.path) != ("defect",):
        raise ModelContractError("YOLO class_names.yaml must map exactly class 0 to 'defect'")
    training_seed = read_training_seed(args_yaml.path)
    if training_seed != receipt.actual_seed:
        raise ModelContractError(
            f"YOLO args.yaml seed {training_seed} differs from receipt seed {receipt.actual_seed}"
        )
    if read_training_data_reference(args_yaml.path) != receipt.args_data_reference:
        raise ModelContractError("YOLO args.yaml data reference differs from training receipt")
    if data_yaml.sha256 != receipt.export_data_yaml_sha256:
        raise ModelContractError(
            "trainer data.yaml must be byte-identical to the attested atomic YOLO export data.yaml"
        )
    provenance = YoloTrainingProvenance(**receipt.provenance_dict())
    runtime_settings = runtime or YoloRuntimeSettings()
    payload = {
        "schema": "zs32.yolo_deployment",
        "schema_version": 3,
        "assets": [
            asset.to_dict()
            for asset in (weights, args_yaml, data_yaml, class_names_yaml, training_receipt)
        ],
        "model_digest": weights.sha256,
        "provenance": provenance.to_dict(),
        "runtime": runtime_settings.to_dict(),
    }
    return YoloDeploymentSpec(
        weights=weights,
        args_yaml=args_yaml,
        data_yaml=data_yaml,
        class_names_yaml=class_names_yaml,
        training_receipt=training_receipt,
        model_digest=weights.sha256,
        bundle_digest=canonical_sha256(payload),
        provenance=provenance,
        runtime=runtime_settings,
    )


@runtime_checkable
class YoloPredictor(Protocol):
    """Global YOLO predictor preserving the identity of every input sample."""

    def predict_batch(
        self,
        samples: Sequence[ModelInput],
        *,
        inspection_id: str,
    ) -> Sequence[RawModelScore]:
        """Return raw maximum-confidence evidence and normalized detections."""


@runtime_checkable
class YoloRuntimeLoader(Protocol):
    """Linux-only Ultralytics restoration boundary."""

    def __call__(self, spec: YoloDeploymentSpec, device: DeviceSpec) -> YoloPredictor:
        """Load a previously verified global best.pt."""


class UltralyticsYoloAdapter:
    """Verify a global bundle before delegating to an injected Ultralytics loader."""

    def __init__(self, runtime_loader: YoloRuntimeLoader | None = None) -> None:
        self._runtime_loader = runtime_loader

    def load(self, spec: YoloDeploymentSpec, device: DeviceSpec) -> YoloPredictor:
        """Verify provenance and all files before invoking Ultralytics."""
        spec.verify()
        if self._runtime_loader is None:
            raise BackendUnavailableError(
                "Ultralytics runtime backend is not configured; run with the Linux GPU integration",
            )
        return self._runtime_loader(spec, device)
