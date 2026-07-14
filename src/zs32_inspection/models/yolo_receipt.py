# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Strict external-trainer attestation for one YOLO training result.

The external trainer is deliberately outside this repository.  This receipt
records the exact inputs and outputs that the trainer attests it observed; it
is provenance evidence, not proof that the inputs causally produced the
checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from zs32_inspection.runtime.publisher import canonical_json_bytes

from .base import ModelContractError, require_sha256, require_text, sha256_file


_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
_RFC3339_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})"
)
_ATTESTATION_SCOPE = "external_trainer_observation_not_causal_proof"
_ARTIFACT_NAMES = ("best.pt", "args.yaml", "data.yaml", "class_names.yaml")


def _strict_text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ModelContractError(f"{field} must be a string")
    return require_text(value, field)


def _strict_sha256(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ModelContractError(f"{field} must be a lowercase SHA256 string")
    return require_sha256(value, field)


def _exact_keys(value: object, expected: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ModelContractError(
            f"{field} must have exact keys {sorted(expected)}; found {actual}"
        )
    return value


def _load_unique_json(content: bytes) -> object:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ModelContractError(f"training_receipt.json repeats key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(content, object_pairs_hook=unique_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ModelContractError(f"training_receipt.json is invalid UTF-8 JSON: {error}") from error


@dataclass(frozen=True, slots=True)
class YoloTrainerSource:
    """Exact clean source identity asserted by the external trainer."""

    kind: str
    repository: str = ""
    commit: str = ""
    bundle_sha256: str = ""
    immutable_storage_reference: str = ""
    runtime_package_tree_sha256: str = ""

    def __post_init__(self) -> None:
        runtime_tree = _strict_sha256(
            self.runtime_package_tree_sha256,
            "yolo.trainer_source.runtime_package_tree_sha256",
        )
        object.__setattr__(self, "runtime_package_tree_sha256", runtime_tree)
        if self.kind == "git_commit":
            repository = _strict_text(self.repository, "yolo.trainer_source.repository")
            commit = _strict_text(self.commit, "yolo.trainer_source.commit").lower()
            if _COMMIT_PATTERN.fullmatch(commit) is None:
                raise ModelContractError(
                    "yolo.trainer_source.commit must be an exact lowercase 40- or 64-hex commit"
                )
            if self.bundle_sha256 or self.immutable_storage_reference:
                raise ModelContractError("git_commit trainer source cannot contain source-bundle fields")
            object.__setattr__(self, "repository", repository)
            object.__setattr__(self, "commit", commit)
            return
        if self.kind == "content_addressed_bundle":
            digest = _strict_sha256(
                self.bundle_sha256,
                "yolo.trainer_source.bundle_sha256",
            )
            reference = _strict_text(
                self.immutable_storage_reference,
                "yolo.trainer_source.immutable_storage_reference",
            )
            if digest not in reference:
                raise ModelContractError(
                    "source-bundle immutable_storage_reference must contain bundle_sha256"
                )
            if self.repository or self.commit:
                raise ModelContractError("content-addressed trainer source cannot contain git fields")
            object.__setattr__(self, "bundle_sha256", digest)
            object.__setattr__(self, "immutable_storage_reference", reference)
            return
        raise ModelContractError(
            "yolo.trainer_source.kind must be git_commit or content_addressed_bundle"
        )

    @property
    def runtime_identity(self) -> str:
        """Identity expected from the production Ultralytics runtime boundary."""
        return self.commit if self.kind == "git_commit" else self.bundle_sha256

    def to_dict(self) -> dict[str, str]:
        if self.kind == "git_commit":
            return {
                "kind": self.kind,
                "repository": self.repository,
                "commit": self.commit,
                "worktree_state": "clean",
                "runtime_package_tree_sha256": self.runtime_package_tree_sha256,
            }
        return {
            "kind": self.kind,
            "bundle_sha256": self.bundle_sha256,
            "immutable_storage_reference": self.immutable_storage_reference,
            "runtime_package_tree_sha256": self.runtime_package_tree_sha256,
        }


def runtime_package_tree_sha256(root: Path) -> str:
    """Hash the exact importable package tree, excluding generated bytecode caches."""
    package_root = Path(root).expanduser()
    if package_root.is_symlink() or not package_root.is_dir():
        raise ModelContractError(
            f"Ultralytics runtime package root must be a non-symlink directory: {package_root}"
        )
    package_root = package_root.resolve()
    digest = hashlib.sha256()
    count = 0
    for path in sorted(
        package_root.rglob("*"),
        key=lambda item: item.relative_to(package_root).as_posix(),
    ):
        relative = path.relative_to(package_root)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            raise ModelContractError(f"Ultralytics runtime package contains a symlink: {path}")
        metadata = path.stat(follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ModelContractError(
                f"Ultralytics runtime package contains a special filesystem node: {path}"
            )
        relative_bytes = relative.as_posix().encode("utf-8")
        digest.update(relative_bytes)
        digest.update(b"\x00")
        digest.update(bytes.fromhex(sha256_file(path)))
        digest.update(b"\x00")
        digest.update(str(metadata.st_size).encode("ascii"))
        digest.update(b"\n")
        count += 1
    if count == 0:
        raise ModelContractError("Ultralytics runtime package tree contains no source files")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class YoloTrainingReceipt:
    """Canonical, strict-schema observation receipt supplied by the external trainer."""

    attested_by: str
    attested_at: str
    run_id: str
    run_name: str
    actual_seed: int
    artifact_sha256: dict[str, str]
    yolo_export_publication_id: str
    yolo_export_root_sha256: str
    export_manifest_sha256: str
    export_data_yaml_sha256: str
    export_policy_sha256: str
    args_data_reference: str
    dataset_release_id: str
    dataset_manifest_sha256: str
    trainer_source: YoloTrainerSource
    receipt_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "attested_by",
            "attested_at",
            "run_id",
            "run_name",
            "yolo_export_publication_id",
            "dataset_release_id",
            "args_data_reference",
        ):
            object.__setattr__(
                self,
                field,
                _strict_text(getattr(self, field), f"yolo.receipt.{field}"),
            )
        if _RFC3339_PATTERN.fullmatch(self.attested_at) is None:
            raise ModelContractError("yolo.receipt.attested_at must be RFC3339 with an explicit timezone")
        try:
            parsed_time = datetime.fromisoformat(self.attested_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ModelContractError("yolo.receipt.attested_at is not a real RFC3339 timestamp") from error
        if parsed_time.tzinfo is None:
            raise ModelContractError("yolo.receipt.attested_at must include a timezone")
        if (
            isinstance(self.actual_seed, bool)
            or not isinstance(self.actual_seed, int)
            or self.actual_seed < 0
        ):
            raise ModelContractError("yolo.receipt.actual_seed must be a non-negative integer")
        if not isinstance(self.artifact_sha256, dict) or set(self.artifact_sha256) != set(
            _ARTIFACT_NAMES
        ):
            raise ModelContractError("YOLO receipt artifact set must be exact")
        normalized = {
            name: _strict_sha256(
                self.artifact_sha256[name],
                f"yolo.receipt.artifacts.{name}",
            )
            for name in _ARTIFACT_NAMES
        }
        object.__setattr__(self, "artifact_sha256", normalized)
        for field in (
            "yolo_export_root_sha256",
            "export_manifest_sha256",
            "export_data_yaml_sha256",
            "export_policy_sha256",
            "dataset_manifest_sha256",
            "receipt_sha256",
        ):
            object.__setattr__(
                self,
                field,
                _strict_sha256(getattr(self, field), f"yolo.receipt.{field}"),
            )
        if not isinstance(self.trainer_source, YoloTrainerSource):
            raise ModelContractError("yolo.receipt.trainer_source must be a YoloTrainerSource")

    def verify_bundle(self, bundle_dir: Path) -> None:
        """Rehash the exact four trainer-output files named by the receipt."""
        root = Path(bundle_dir).expanduser()
        for name in _ARTIFACT_NAMES:
            path = root / name
            if path.is_symlink() or not path.is_file():
                raise ModelContractError(f"external YOLO bundle is missing regular file {name}: {path}")
            actual = sha256_file(path)
            if actual != self.artifact_sha256[name]:
                raise ModelContractError(
                    f"YOLO receipt digest mismatch for {name}: {actual} != {self.artifact_sha256[name]}"
                )

    def provenance_dict(self) -> dict[str, object]:
        """Return all receipt bindings that must survive candidate and release packaging."""
        return {
            "dataset_release_id": self.dataset_release_id,
            "dataset_manifest_digest": self.dataset_manifest_sha256,
            "yolo_export_publication_id": self.yolo_export_publication_id,
            "yolo_export_root_sha256": self.yolo_export_root_sha256,
            "yolo_export_manifest_digest": self.export_manifest_sha256,
            "yolo_export_data_yaml_digest": self.export_data_yaml_sha256,
            "yolo_export_policy_digest": self.export_policy_sha256,
            "args_data_reference": self.args_data_reference,
            "ultralytics_version_or_commit": self.trainer_source.runtime_identity,
            "trainer_source": self.trainer_source.to_dict(),
            "training_receipt_digest": self.receipt_sha256,
            "training_seed": self.actual_seed,
            "run_id": self.run_id,
            "run_name": self.run_name,
        }


def load_yolo_training_receipt(path: Path) -> YoloTrainingReceipt:
    """Load only the training-safe canonical v3 receipt schema."""
    receipt_path = Path(path).expanduser()
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ModelContractError(f"training receipt must be a regular non-symlink file: {receipt_path}")
    content = receipt_path.read_bytes()
    payload = _load_unique_json(content)
    root = _exact_keys(
        payload,
        {
            "schema",
            "schema_version",
            "attestation_scope",
            "attested_by",
            "attested_at",
            "run",
            "artifacts",
            "training_data",
            "trainer_source",
        },
        "training_receipt.json",
    )
    if canonical_json_bytes(root) != content:
        raise ModelContractError("training_receipt.json must use canonical JSON bytes")
    if root["schema"] != "zs32.yolo_external_training_receipt" or root["schema_version"] != 3:
        raise ModelContractError(
            "training_receipt.json schema must be zs32.yolo_external_training_receipt v3"
        )
    if root["attestation_scope"] != _ATTESTATION_SCOPE:
        raise ModelContractError(
            "training receipt must explicitly state external observation, not causal proof"
        )
    run = _exact_keys(root["run"], {"run_id", "run_name", "actual_seed"}, "receipt.run")
    artifacts = _exact_keys(root["artifacts"], set(_ARTIFACT_NAMES), "receipt.artifacts")
    training_data = _exact_keys(
        root["training_data"],
        {
            "yolo_export_publication_id",
            "yolo_export_root_sha256",
            "export_manifest_sha256",
            "export_data_yaml_sha256",
            "export_policy_sha256",
            "args_data_reference",
            "dataset_release_id",
            "dataset_manifest_sha256",
        },
        "receipt.training_data",
    )
    if not isinstance(root["trainer_source"], dict):
        raise ModelContractError("receipt.trainer_source must be an object")
    source_raw = root["trainer_source"]
    source = parse_yolo_trainer_source(source_raw)
    return YoloTrainingReceipt(
        attested_by=root["attested_by"],
        attested_at=root["attested_at"],
        run_id=run["run_id"],
        run_name=run["run_name"],
        actual_seed=run["actual_seed"],
        artifact_sha256=dict(artifacts),
        yolo_export_publication_id=training_data["yolo_export_publication_id"],
        yolo_export_root_sha256=training_data["yolo_export_root_sha256"],
        export_manifest_sha256=training_data["export_manifest_sha256"],
        export_data_yaml_sha256=training_data["export_data_yaml_sha256"],
        export_policy_sha256=training_data["export_policy_sha256"],
        args_data_reference=training_data["args_data_reference"],
        dataset_release_id=training_data["dataset_release_id"],
        dataset_manifest_sha256=training_data["dataset_manifest_sha256"],
        trainer_source=source,
        receipt_sha256=sha256_file(receipt_path),
    )


def parse_yolo_trainer_source(payload: object) -> YoloTrainerSource:
    """Parse the exact source-identity union used in receipt and runtime metadata."""
    if not isinstance(payload, dict):
        raise ModelContractError("yolo trainer_source must be an object")
    source_raw = payload
    kind = source_raw.get("kind")
    if kind == "git_commit":
        _exact_keys(
            source_raw,
            {
                "kind", "repository", "commit", "worktree_state",
                "runtime_package_tree_sha256",
            },
            "receipt.trainer_source",
        )
        if source_raw["worktree_state"] != "clean":
            raise ModelContractError("git trainer source must attest worktree_state='clean'")
        source = YoloTrainerSource(
            kind="git_commit",
            repository=source_raw["repository"],
            commit=source_raw["commit"],
            runtime_package_tree_sha256=source_raw["runtime_package_tree_sha256"],
        )
    elif kind == "content_addressed_bundle":
        _exact_keys(
            source_raw,
            {
                "kind", "bundle_sha256", "immutable_storage_reference",
                "runtime_package_tree_sha256",
            },
            "receipt.trainer_source",
        )
        source = YoloTrainerSource(
            kind="content_addressed_bundle",
            bundle_sha256=source_raw["bundle_sha256"],
            immutable_storage_reference=source_raw["immutable_storage_reference"],
            runtime_package_tree_sha256=source_raw["runtime_package_tree_sha256"],
        )
    else:
        raise ModelContractError("receipt.trainer_source has unsupported kind")
    return source


__all__ = [
    "YoloTrainerSource",
    "YoloTrainingReceipt",
    "load_yolo_training_receipt",
    "parse_yolo_trainer_source",
    "runtime_package_tree_sha256",
]
