"""Verify an immutable deployment release before importing model frameworks."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from zs32_inspection.calibration import (
    CalibrationArtifact,
    CalibrationProvenance,
    DualThreshold,
    FitParameters,
    HeldOutPartMetrics,
    ScoreBranch,
    ScoreGroup,
    TemplateCalibrationGroup,
)
from zs32_inspection.calibration.acceptance import (
    evaluate_heldout_acceptance,
    load_heldout_acceptance_decision_bytes,
    load_heldout_acceptance_policy_bytes,
)
from zs32_inspection.capture.gate_policy import (
    CaptureGatePolicy,
    CaptureGateProvenance,
    load_capture_gate_policy_bytes,
)
from zs32_inspection.capture.gate_publication import verify_capture_gate_asset_semantics
from zs32_inspection.capture.contracts import CapturePlan
from zs32_inspection.capture.hikvision import HikvisionCaptureConfig
from zs32_inspection.capture.opencv_quality import QualityGateProfile
from zs32_inspection.capture.opencv_registration import RegistrationGateProfile
from zs32_inspection.config.schemas import (
    parse_deployment_contract,
    parse_recipe,
    parse_release_manifest,
    parse_roi_config,
    parse_topology,
)
from zs32_inspection.data.codec_contract import CanonicalPngCodec
from zs32_inspection.data.calibration_targets import CalibrationTargetsSnapshot
from zs32_inspection.data.manifests import (
    CanonicalSemanticsSnapshot,
    CaptureProvenanceRow,
    DatasetReleaseManifest,
)
from zs32_inspection.data.splitter import SplitAssignment, SplitPolicy, assign_grouped_splits
from zs32_inspection.domain.contracts import DeploymentContract, ReleaseManifest
from zs32_inspection.domain.identity import Hand, PRODUCT
from zs32_inspection.fusion.policy import FusionPolicy, parse_fusion_policy
from zs32_inspection.models.anomalib_backend import validate_anomaly_artifact_metadata
from zs32_inspection.models.base import (
    AnomalyModelArtifact,
    AssetFile,
    ModelSlot,
    canonical_sha256 as model_canonical_sha256,
)
from zs32_inspection.models.yolo import (
    YoloDeploymentSpec,
    YoloRuntimeSettings,
    YoloTrainingProvenance,
)
from zs32_inspection.template.artifacts import TemplateThresholdFit

from .environment_receipt import RuntimeEnvironmentReceipt
from .execution_receipt import ExecutionReceipt
from .promotion_receipt import PromotionReceipt
from .publisher import PublicationError, canonical_json_bytes

_CHECKSUM_LINE = re.compile(r"(?P<digest>[0-9a-f]{64})  (?P<path>[^\r\n]+)")


class ReleaseLoadError(RuntimeError):
    """A release failed byte-level or envelope validation."""


def _safe_index_path(text: str) -> str:
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in text
        or any(character in text for character in ("\x00", "\n", "\r"))
    ):
        raise ReleaseLoadError(f"unsafe path in checksums.sha256: {text!r}")
    return path.as_posix()


def _load_json_bytes(content: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ReleaseLoadError(f"cannot decode release JSON {label}: {error}") from error
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ReleaseLoadError(f"release JSON root must be an object: {label}")
    return payload


def _load_config_bytes(content: bytes, *, relative_path: str) -> dict[str, Any]:
    suffix = PurePosixPath(relative_path).suffix.lower()
    if suffix not in {".json", ".yaml", ".yml"}:
        raise ReleaseLoadError(f"unsupported release config suffix: {relative_path}")
    try:
        text = content.decode("utf-8")
        payload = json.loads(text) if suffix == ".json" else yaml.safe_load(text)
    except (UnicodeError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise ReleaseLoadError(f"cannot decode release config {relative_path}: {error}") from error
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ReleaseLoadError(f"release config root must be an object: {relative_path}")
    return payload


def _read_confined_regular_file(root: Path, relative_path: str) -> bytes:
    """Read below ``root`` without following any path-component symlink.

    Every component is opened relative to an already-open directory descriptor.
    The final descriptor must remain the same private regular file for the whole
    read.  This is the use-boundary counterpart to the initial full-tree audit.
    """
    relative = _safe_index_path(relative_path)
    parts = PurePosixPath(relative).parts
    descriptors: list[int] = []
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        current = os.open(
            root,
            flags | getattr(os, "O_DIRECTORY", 0),
        )
        descriptors.append(current)
        root_metadata = os.fstat(current)
        if not stat.S_ISDIR(root_metadata.st_mode):
            raise ReleaseLoadError(f"release root descriptor is not a directory: {root}")
        for component in parts[:-1]:
            current = os.open(
                component,
                flags | getattr(os, "O_DIRECTORY", 0),
                dir_fd=current,
            )
            descriptors.append(current)
            metadata = os.fstat(current)
            if not stat.S_ISDIR(metadata.st_mode):
                raise ReleaseLoadError(
                    f"release path component is not a directory: {relative_path}"
                )
        file_descriptor = os.open(parts[-1], flags, dir_fd=current)
        descriptors.append(file_descriptor)
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ReleaseLoadError(
                f"release file must be a private regular file: {relative_path}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_before != identity_after:
            raise ReleaseLoadError(f"release file changed while being read: {relative_path}")
        return b"".join(chunks)
    except ReleaseLoadError:
        raise
    except OSError as error:
        raise ReleaseLoadError(f"cannot safely read release file {relative_path}: {error}") from error
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _parse_checksum_index(content: bytes) -> Mapping[str, str]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ReleaseLoadError("checksums.sha256 must be UTF-8") from error
    if not text.endswith("\n"):
        raise ReleaseLoadError("checksums.sha256 must end with one newline")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = _CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise ReleaseLoadError(f"malformed checksums.sha256 line {line_number}")
        relative = _safe_index_path(match.group("path"))
        if relative in {"checksums.sha256", "publication_root.json"}:
            raise ReleaseLoadError(f"checksum index must not contain its publication metadata: {relative}")
        if relative in entries:
            raise ReleaseLoadError(f"duplicate checksum path: {relative}")
        entries[relative] = match.group("digest")
    if not entries:
        raise ReleaseLoadError("checksums.sha256 must contain release files")
    if list(entries) != sorted(entries):
        raise ReleaseLoadError("checksums.sha256 paths must be sorted canonically")
    return MappingProxyType(entries)


@dataclass(frozen=True, slots=True)
class VerifiedReleaseFiles:
    """Byte-verified release envelope safe for semantic contract parsing."""

    root: Path
    release_id: str
    root_sha256: str
    manifest: Mapping[str, Any]
    deployment_contract: Mapping[str, Any]
    checksums: Mapping[str, str]

    def path(self, relative_path: str) -> Path:
        """Resolve an indexed regular file without leaving the release."""
        relative = _safe_index_path(relative_path)
        if relative not in self.checksums:
            raise ReleaseLoadError(f"file is not covered by release checksums: {relative}")
        return self.root / Path(*PurePosixPath(relative).parts)

    def read_bytes(self, relative_path: str) -> bytes:
        """Re-open and rehash one indexed file at its exact use boundary."""
        relative = _safe_index_path(relative_path)
        expected = self.checksums.get(relative)
        if expected is None:
            raise ReleaseLoadError(f"file is not covered by release checksums: {relative}")
        content = _read_confined_regular_file(self.root, relative)
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise ReleaseLoadError(
                f"release file changed after verification: {relative}: {actual} != {expected}"
            )
        return content

    def read_json(self, relative_path: str) -> dict[str, Any]:
        """Read one indexed JSON object through the verified boundary."""
        return _load_json_bytes(self.read_bytes(relative_path), label=relative_path)

    def read_config(self, relative_path: str) -> dict[str, Any]:
        """Read one indexed JSON/YAML object through the verified boundary."""
        return _load_config_bytes(self.read_bytes(relative_path), relative_path=relative_path)


@dataclass(frozen=True, slots=True)
class VerifiedDeploymentRelease:
    """Semantically rehashed release safe to pass to model asset loaders."""

    files: VerifiedReleaseFiles
    manifest: ReleaseManifest
    contract: DeploymentContract
    fusion_policy: FusionPolicy
    fusion_policy_file_sha256: str
    capture_gate_policy: CaptureGatePolicy
    capture_gate_policy_file_sha256: str
    runtime_environment: RuntimeEnvironmentReceipt
    runtime_environment_file_sha256: str
    promotion_receipt: PromotionReceipt
    promotion_receipt_file_sha256: str


def _verify_release_files(
    root: Path,
    *,
    expected_publication_id: str | None = None,
) -> VerifiedReleaseFiles:
    """Verify the entire tree and its publication root without loading models."""
    root = Path(root).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise ReleaseLoadError(f"release root must be a regular directory, not a symlink: {root}")
    root = root.resolve()
    checksum_bytes = _read_confined_regular_file(root, "checksums.sha256")
    entries = _parse_checksum_index(checksum_bytes)

    actual: set[str] = set()
    for path in root.rglob("*"):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ReleaseLoadError(f"symlinks are forbidden in a release: {path}")
        if stat.S_ISREG(metadata.st_mode):
            if metadata.st_nlink != 1:
                raise ReleaseLoadError(f"hardlinked files are forbidden in a release: {path}")
            actual.add(path.relative_to(root).as_posix())
        elif not stat.S_ISDIR(metadata.st_mode):
            raise ReleaseLoadError(f"special filesystem nodes are forbidden in a release: {path}")
    expected = set(entries) | {"checksums.sha256", "publication_root.json"}
    if actual != expected:
        raise ReleaseLoadError(
            f"release file set differs from checksum index; missing={sorted(expected - actual)}, "
            f"unindexed={sorted(actual - expected)}"
        )
    for relative, expected_digest in entries.items():
        actual_digest = hashlib.sha256(
            _read_confined_regular_file(root, relative)
        ).hexdigest()
        if actual_digest != expected_digest:
            raise ReleaseLoadError(
                f"release checksum mismatch for {relative}: {actual_digest} != {expected_digest}"
            )

    publication_root = _load_json_bytes(
        _read_confined_regular_file(root, "publication_root.json"),
        label="publication_root.json",
    )
    if set(publication_root) != {"algorithm", "publication_id", "root_sha256"}:
        raise ReleaseLoadError("publication_root.json has unexpected or missing fields")
    if publication_root["algorithm"] != "sha256(checksums.sha256 bytes)":
        raise ReleaseLoadError("unsupported publication root algorithm")
    publication_id = root.name if expected_publication_id is None else expected_publication_id
    if publication_root["publication_id"] != publication_id:
        raise ReleaseLoadError("publication id does not match release directory name")
    root_digest = hashlib.sha256(checksum_bytes).hexdigest()
    if publication_root["root_sha256"] != root_digest:
        raise ReleaseLoadError("publication root digest does not match checksums.sha256 bytes")

    if "manifest.json" not in entries or "deployment_contract.json" not in entries:
        raise ReleaseLoadError("release is missing indexed manifest.json or deployment_contract.json")
    manifest = _load_json_bytes(
        _read_confined_regular_file(root, "manifest.json"), label="manifest.json"
    )
    contract = _load_json_bytes(
        _read_confined_regular_file(root, "deployment_contract.json"),
        label="deployment_contract.json",
    )
    if manifest.get("release_id") != publication_id or manifest.get("product") != PRODUCT:
        raise ReleaseLoadError("release manifest identity does not match its directory/product")
    contract_digest = contract.get("contract_sha256")
    if not isinstance(contract_digest, str) or manifest.get("contract_sha256") != contract_digest:
        raise ReleaseLoadError("manifest and deployment contract digests disagree")
    return VerifiedReleaseFiles(
        root=root,
        release_id=publication_id,
        root_sha256=root_digest,
        manifest=MappingProxyType(manifest),
        deployment_contract=MappingProxyType(contract),
        checksums=entries,
    )


def verify_release_files(root: Path) -> VerifiedReleaseFiles:
    """Verify a release and normalize all low-level failures to one boundary."""
    try:
        return _verify_release_files(root)
    except ReleaseLoadError:
        raise
    except (OSError, PublicationError, TypeError, ValueError) as error:
        raise ReleaseLoadError(f"release byte verification failed: {error}") from error


def _load_verified_deployment_release(
    root: Path,
    *,
    expected_publication_id: str | None = None,
) -> VerifiedDeploymentRelease:
    """Perform byte verification, then strict schema and contract revalidation."""
    files = (
        verify_release_files(root)
        if expected_publication_id is None
        else _verify_release_files(root, expected_publication_id=expected_publication_id)
    )
    try:
        manifest = parse_release_manifest(files.read_json("manifest.json"))
        contract = parse_deployment_contract(files.read_json("deployment_contract.json"))
    except ValueError as error:
        raise ReleaseLoadError(f"release schema/contract validation failed: {error}") from error
    if manifest.release_id != files.release_id:
        raise ReleaseLoadError("validated release manifest id differs from directory name")
    if manifest.contract_sha256 != contract.contract_sha256:
        raise ReleaseLoadError("validated manifest and contract digests disagree")

    policy_relative = "fusion/policy.yaml"
    if policy_relative not in files.checksums:
        raise ReleaseLoadError(f"release is missing indexed {policy_relative}")
    policy_digest = files.checksums[policy_relative]
    if policy_digest != contract.fusion_policy_sha256:
        raise ReleaseLoadError("fusion policy file digest differs from deployment contract")
    try:
        policy_payload = files.read_config(policy_relative)
        policy = parse_fusion_policy(policy_payload)
    except ValueError as error:
        raise ReleaseLoadError(f"fusion policy schema validation failed: {error}") from error

    gate_policy_relative = contract.capture_gate_policy.relative_path
    gate_policy_digest = files.checksums.get(gate_policy_relative)
    if gate_policy_digest != contract.capture_gate_policy.sha256:
        raise ReleaseLoadError(
            "capture gate policy file is missing or differs from deployment contract"
        )
    try:
        gate_policy = load_capture_gate_policy_bytes(
            files.read_bytes(gate_policy_relative)
        )
        gate_policy.validate_topology(
            contract.topology,
            allowed_hands=contract.allowed_hands,
        )
        if (
            files.checksums.get(gate_policy.acquisition_config.relative_path)
            != gate_policy.acquisition_config.sha256
        ):
            raise ReleaseLoadError(
                "capture acquisition config is missing or differs from policy"
            )
        HikvisionCaptureConfig.from_mapping(
            files.read_json(gate_policy.acquisition_config.relative_path)
        )
        for hand_policy in gate_policy.hands.values():
            for artifact in (
                hand_policy.quality_profile,
                hand_policy.registration_profile,
                *hand_policy.registration_references.values(),
            ):
                if files.checksums.get(artifact.relative_path) != artifact.sha256:
                    raise ReleaseLoadError(
                        "capture gate asset is missing or differs from policy: "
                        f"{artifact.relative_path}"
                    )
                files.read_bytes(artifact.relative_path)
            quality_profile = QualityGateProfile.from_mapping(
                files.read_json(hand_policy.quality_profile.relative_path)
            )
            registration_profile = RegistrationGateProfile.from_mapping(
                files.read_json(hand_policy.registration_profile.relative_path)
            )
            plan = CapturePlan.from_topology(contract.topology)
            quality_profile.validate_plan(plan)
            registration_profile.validate_plan(plan)
            if (
                quality_profile.hand != hand_policy.hand.value
                or registration_profile.hand != hand_policy.hand.value
            ):
                raise ReleaseLoadError("capture gate profile hand differs from policy hand")
            profile_references = {
                view: (spec.reference.relative_path, spec.reference.sha256)
                for view, spec in registration_profile.views.items()
            }
            policy_references = {
                view: (artifact.relative_path, artifact.sha256)
                for view, artifact in hand_policy.registration_references.items()
            }
            if profile_references != policy_references:
                raise ReleaseLoadError(
                    "registration profile references differ from capture gate policy"
                )
        verify_capture_gate_asset_semantics(
            contract.topology,
            gate_policy,
            files.read_bytes,
        )
    except ReleaseLoadError:
        raise
    except (TypeError, ValueError) as error:
        raise ReleaseLoadError(f"capture gate policy validation failed: {error}") from error

    required_assets = {
        contract.yolo_model.relative_path: contract.yolo_model.sha256,
        **{binding.asset.relative_path: binding.asset.sha256 for binding in contract.template_bindings},
        **{binding.asset.relative_path: binding.asset.sha256 for binding in contract.anomaly_bindings},
    }
    for relative, digest in required_assets.items():
        indexed = files.checksums.get(relative)
        if indexed != digest:
            raise ReleaseLoadError(
                f"contract asset is missing or has a different indexed digest: {relative}"
            )
    fixed_paths = {
        "recipe.yaml",
        "topology.yaml",
        "roi.yaml",
        "models/yolo/metadata.json",
        "models/yolo/train_args.yaml",
        "models/yolo/data.yaml",
        "models/yolo/class_names.yaml",
        "models/yolo/training_receipt.json",
        "models/yolo/weights.sha256",
        "calibration/calibration_artifact.json",
        "calibration/template_thresholds.json",
        "calibration/model_thresholds.json",
        "calibration/metrics.json",
        "calibration/input_provenance.json",
        "provenance/dataset_release.json",
        "provenance/dataset_capture_provenance.jsonl",
        "provenance/dataset_canonical_semantics.json",
        "provenance/dataset_calibration_targets.json",
        "provenance/dataset_governance.json",
        "provenance/dataset_split_assignments.json",
        "provenance/model_candidate.json",
        "provenance/model_candidate_full.json",
        "provenance/promotion_receipt.json",
        "provenance/code_version.json",
        "provenance/runtime_environment.json",
    }
    missing_fixed = sorted(fixed_paths - set(files.checksums))
    if missing_fixed:
        raise ReleaseLoadError(f"release is missing required deployment files: {missing_fixed}")
    try:
        environment_payload = files.read_json("provenance/runtime_environment.json")
        runtime_environment = RuntimeEnvironmentReceipt.from_mapping(environment_payload)
    except (TypeError, ValueError) as error:
        raise ReleaseLoadError(f"runtime environment receipt validation failed: {error}") from error
    if files.read_bytes("provenance/runtime_environment.json") != runtime_environment.canonical_bytes():
        raise ReleaseLoadError("runtime environment receipt must use canonical JSON bytes")
    runtime_environment_file_sha256 = files.checksums[
        "provenance/runtime_environment.json"
    ]
    try:
        promotion_payload = files.read_json("provenance/promotion_receipt.json")
        promotion_receipt = PromotionReceipt.from_mapping(promotion_payload)
    except (TypeError, ValueError) as error:
        raise ReleaseLoadError(f"promotion receipt validation failed: {error}") from error
    if files.read_bytes("provenance/promotion_receipt.json") != promotion_receipt.canonical_bytes():
        raise ReleaseLoadError("promotion receipt must use exact canonical JSON bytes")
    promotion_receipt_file_sha256 = files.checksums[
        "provenance/promotion_receipt.json"
    ]
    if promotion_receipt.sha256 != promotion_receipt_file_sha256:
        raise ReleaseLoadError("promotion receipt semantic and file digests disagree")
    try:
        approved_at = datetime.fromisoformat(
            promotion_receipt.approved_at.replace("Z", "+00:00")
        )
        release_created_at = datetime.fromisoformat(
            manifest.created_at.replace("Z", "+00:00")
        )
    except ValueError as error:
        raise ReleaseLoadError("release/promotion timestamps must be ISO-8601") from error
    if release_created_at.tzinfo is None or release_created_at.utcoffset() is None:
        raise ReleaseLoadError("release created_at must include an explicit timezone")
    if approved_at > release_created_at:
        raise ReleaseLoadError("promotion approval is later than release created_at")
    try:
        recipe = parse_recipe(files.read_config("recipe.yaml"))
        topology = parse_topology(files.read_config("topology.yaml"))
        roi = parse_roi_config(files.read_config("roi.yaml"))
    except ValueError as error:
        raise ReleaseLoadError(f"release config snapshot validation failed: {error}") from error
    recipe_templates = {
        (item.hand, item.view_id): item.asset for item in recipe.template_bindings
    }
    contract_templates = {
        (item.hand, item.view_id): item.asset for item in contract.template_bindings
    }
    recipe_anomaly = {
        (item.hand, item.view_id): (item.family, item.asset)
        for item in recipe.anomaly_bindings
    }
    contract_anomaly = {
        (item.hand, item.view_id): (item.family, item.asset)
        for item in contract.anomaly_bindings
    }
    recipe_template_thresholds = {
        (item.hand, item.view_id): item for item in recipe.template_thresholds
    }
    contract_template_thresholds = {
        (item.hand, item.view_id): item for item in contract.template_thresholds
    }
    recipe_model_thresholds = {
        (item.hand, item.view_id, item.branch): item for item in recipe.model_thresholds
    }
    contract_model_thresholds = {
        (item.hand, item.view_id, item.branch): item for item in contract.model_thresholds
    }
    if (
        recipe.recipe_sha256 != contract.recipe_sha256
        or recipe.recipe_id != contract.recipe_id
        or recipe.product != contract.product
        or recipe.topology_id != contract.topology.topology_id
        or recipe.roi_config_id != contract.roi.roi_config_id
        or set(recipe.allowed_hands) != set(contract.allowed_hands)
        or recipe.anomaly_family is not contract.anomaly_family
        or recipe.capture_gate_policy != contract.capture_gate_policy
        or len(recipe.template_bindings) != len(contract.template_bindings)
        or recipe_templates != contract_templates
        or len(recipe.anomaly_bindings) != len(contract.anomaly_bindings)
        or recipe_anomaly != contract_anomaly
        or recipe.yolo_model != contract.yolo_model
        or len(recipe.template_thresholds) != len(contract.template_thresholds)
        or recipe_template_thresholds != contract_template_thresholds
        or len(recipe.model_thresholds) != len(contract.model_thresholds)
        or recipe_model_thresholds != contract_model_thresholds
        or recipe.fusion_policy_sha256 != contract.fusion_policy_sha256
        or topology.topology_sha256 != contract.topology.topology_sha256
        or roi.roi_sha256 != contract.roi.roi_sha256
    ):
        raise ReleaseLoadError("release config snapshots differ from deployment contract")

    calibration_digest = files.checksums["calibration/calibration_artifact.json"]
    threshold_calibration_digests = {
        item.calibration_sha256
        for item in (*contract.template_thresholds, *contract.model_thresholds)
    }
    if threshold_calibration_digests != {calibration_digest}:
        raise ReleaseLoadError("release thresholds do not bind calibration_artifact.json bytes")
    dataset = _validate_release_provenance(
        files,
        contract,
        promotion_receipt=promotion_receipt,
        promotion_receipt_sha256=promotion_receipt_file_sha256,
        capture_gate_policy=gate_policy,
        capture_gate_policy_sha256=gate_policy_digest,
    )
    _validate_calibration_artifact(files, contract, dataset)
    _validate_yolo_bundle(files, contract)
    _validate_template_groups(
        files,
        contract,
        expected_train_split_id=dataset.split_assignments_sha256,
    )
    _validate_anomaly_groups(
        files,
        contract,
        expected_train_split_id=dataset.split_assignments_sha256,
    )
    _validate_exact_release_file_set(files, contract, gate_policy)
    return VerifiedDeploymentRelease(
        files=files,
        manifest=manifest,
        contract=contract,
        fusion_policy=policy,
        fusion_policy_file_sha256=policy_digest,
        capture_gate_policy=gate_policy,
        capture_gate_policy_file_sha256=gate_policy_digest,
        runtime_environment=runtime_environment,
        runtime_environment_file_sha256=runtime_environment_file_sha256,
        promotion_receipt=promotion_receipt,
        promotion_receipt_file_sha256=promotion_receipt_file_sha256,
    )


def _validate_exact_release_file_set(
    files: VerifiedReleaseFiles,
    contract: DeploymentContract,
    gate_policy: CaptureGatePolicy,
) -> None:
    """Reject even reindexed files that are not derivable from the release contract."""
    expected = {
        "manifest.json",
        "deployment_contract.json",
        "recipe.yaml",
        "topology.yaml",
        "roi.yaml",
        "fusion/policy.yaml",
        contract.capture_gate_policy.relative_path,
        "models/yolo/best.pt",
        "models/yolo/metadata.json",
        "models/yolo/train_args.yaml",
        "models/yolo/data.yaml",
        "models/yolo/class_names.yaml",
        "models/yolo/training_receipt.json",
        "models/yolo/weights.sha256",
        "calibration/calibration_artifact.json",
        "calibration/template_thresholds.json",
        "calibration/model_thresholds.json",
        "calibration/metrics.json",
        "calibration/input_provenance.json",
        "provenance/dataset_release.json",
        "provenance/dataset_capture_provenance.jsonl",
        "provenance/dataset_canonical_semantics.json",
        "provenance/dataset_calibration_targets.json",
        "provenance/dataset_governance.json",
        "provenance/dataset_split_assignments.json",
        "provenance/model_candidate.json",
        "provenance/model_candidate_full.json",
        "provenance/promotion_receipt.json",
        "provenance/code_version.json",
        "provenance/runtime_environment.json",
    }
    expected.add(gate_policy.acquisition_config.relative_path)
    for hand_policy in gate_policy.hands.values():
        expected.update(
            artifact.relative_path
            for artifact in (
                hand_policy.quality_profile,
                hand_policy.registration_profile,
                *hand_policy.registration_references.values(),
            )
        )
    for binding in contract.anomaly_bindings:
        expected.add(binding.asset.relative_path)
        expected.add(
            (PurePosixPath(binding.asset.relative_path).parent / "metadata.json").as_posix()
        )
    for binding in contract.template_bindings:
        expected.add(binding.asset.relative_path)
        metadata_path = (
            PurePosixPath(binding.asset.relative_path).parent / "metadata.json"
        ).as_posix()
        expected.add(metadata_path)
        metadata = files.read_json(metadata_path)
        for reference in metadata["reference_templates"]:
            expected.add(reference["relative_path"])
    candidate = files.read_json("provenance/model_candidate.json")
    validation_id = candidate["validation_publication_id"]
    expected.update(
        f"provenance/validations/{validation_id}/{name}"
        for name in (
            "candidate.json",
            "deployment_assets.json",
            "validation.json",
            "heldout_acceptance_policy.json",
            "heldout_acceptance.json",
            "checksums.sha256",
            "publication_root.json",
        )
    )
    actual = set(files.checksums)
    if actual != expected:
        raise ReleaseLoadError(
            "release file set differs from the contract-derived allowlist; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )


def _validate_release_provenance(
    files: VerifiedReleaseFiles,
    contract: DeploymentContract,
    *,
    promotion_receipt: PromotionReceipt,
    promotion_receipt_sha256: str,
    capture_gate_policy: CaptureGatePolicy,
    capture_gate_policy_sha256: str,
) -> DatasetReleaseManifest:
    """Bind copied dataset/candidate provenance to every compiled identity."""
    candidate = files.read_json("provenance/model_candidate.json")
    expected_candidate_keys = {
        "schema_version",
        "product",
        "candidate_id",
        "candidate_digest",
        "candidate_status",
        "promotion_id",
        "promotion_approver",
        "promotion_approved_at",
        "promotion_receipt_sha256",
        "validation_publication_id",
        "validation_publication_root_sha256",
        "validation_record_sha256",
        "heldout_acceptance_policy_sha256",
        "heldout_acceptance_decision_sha256",
        "dataset_release_id",
        "dataset_manifest_sha256",
        "recipe_sha256",
        "topology_sha256",
        "roi_sha256",
        "roi_version",
        "anomaly_family",
        "yolo_model_sha256",
        "calibration_artifact_sha256",
    }
    if set(candidate) != expected_candidate_keys or candidate.get("schema_version") != 3:
        raise ReleaseLoadError("model candidate provenance envelope is malformed")
    for field in (
        "candidate_id",
        "promotion_id",
        "promotion_approver",
        "promotion_approved_at",
        "validation_publication_id",
        "dataset_release_id",
    ):
        if not isinstance(candidate.get(field), str) or not candidate[field].strip():
            raise ReleaseLoadError(f"model candidate provenance has invalid {field}")
    for field in (
        "candidate_digest",
        "dataset_manifest_sha256",
        "recipe_sha256",
        "topology_sha256",
        "roi_sha256",
        "yolo_model_sha256",
        "calibration_artifact_sha256",
        "validation_publication_root_sha256",
        "validation_record_sha256",
        "promotion_receipt_sha256",
        "heldout_acceptance_policy_sha256",
        "heldout_acceptance_decision_sha256",
    ):
        value = candidate.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ReleaseLoadError(f"model candidate provenance has invalid {field}")
    calibration_digest = files.checksums["calibration/calibration_artifact.json"]
    if (
        candidate.get("product") != PRODUCT
        or candidate.get("candidate_status") != "validated"
        or candidate.get("recipe_sha256") != contract.recipe_sha256
        or candidate.get("topology_sha256") != contract.topology.topology_sha256
        or candidate.get("roi_sha256") != contract.roi.roi_sha256
        or candidate.get("roi_version") != contract.roi.roi_config_id
        or candidate.get("anomaly_family") != contract.anomaly_family.value
        or candidate.get("yolo_model_sha256") != contract.yolo_model.sha256
        or candidate.get("calibration_artifact_sha256") != calibration_digest
        or candidate.get("promotion_id") != promotion_receipt.promotion_id
        or candidate.get("promotion_approver") != promotion_receipt.approver
        or candidate.get("promotion_approved_at") != promotion_receipt.approved_at
        or candidate.get("promotion_receipt_sha256") != promotion_receipt_sha256
        or promotion_receipt.release_id != files.release_id
        or promotion_receipt.candidate_id != candidate.get("candidate_id")
        or promotion_receipt.candidate_digest != candidate.get("candidate_digest")
        or promotion_receipt.validation_publication_id
        != candidate.get("validation_publication_id")
        or promotion_receipt.validation_publication_root_sha256
        != candidate.get("validation_publication_root_sha256")
        or promotion_receipt.validation_record_sha256
        != candidate.get("validation_record_sha256")
        or promotion_receipt.contract_sha256 != contract.contract_sha256
        or promotion_receipt.calibration_artifact_sha256 != calibration_digest
        or promotion_receipt.dataset_manifest_sha256
        != candidate.get("dataset_manifest_sha256")
        or promotion_receipt.heldout_acceptance_policy_sha256
        != candidate.get("heldout_acceptance_policy_sha256")
        or promotion_receipt.heldout_acceptance_decision_sha256
        != candidate.get("heldout_acceptance_decision_sha256")
    ):
        raise ReleaseLoadError(
            "model candidate/promotion receipt provenance differs from deployment contract"
        )
    full_candidate = files.read_json("provenance/model_candidate_full.json")
    expected_full_keys = {
        "candidate_id", "product", "anomaly_family", "anomaly_artifacts", "yolo",
        "recipe_digest", "topology_digest", "roi_digest", "roi_version",
        "dataset_release_id", "dataset_manifest_digest", "status", "candidate_digest",
    }
    if set(full_candidate) != expected_full_keys:
        raise ReleaseLoadError("full model candidate descriptor envelope is malformed")
    full_content = {
        key: value
        for key, value in full_candidate.items()
        if key not in {"candidate_id", "status", "candidate_digest"}
    }
    recomputed_candidate_digest = model_canonical_sha256(full_content)
    if (
        recomputed_candidate_digest != full_candidate.get("candidate_digest")
        or recomputed_candidate_digest != candidate.get("candidate_digest")
        or full_candidate.get("candidate_id") != candidate.get("candidate_id")
        or full_candidate.get("status") != candidate.get("candidate_status")
        or full_candidate.get("product") != candidate.get("product")
        or full_candidate.get("anomaly_family") != candidate.get("anomaly_family")
        or full_candidate.get("recipe_digest") != candidate.get("recipe_sha256")
        or full_candidate.get("topology_digest") != candidate.get("topology_sha256")
        or full_candidate.get("roi_digest") != candidate.get("roi_sha256")
        or full_candidate.get("roi_version") != candidate.get("roi_version")
        or full_candidate.get("dataset_release_id") != candidate.get("dataset_release_id")
        or full_candidate.get("dataset_manifest_digest") != candidate.get("dataset_manifest_sha256")
    ):
        raise ReleaseLoadError("full model candidate descriptor digest/provenance is inconsistent")
    anomaly_rows = full_candidate.get("anomaly_artifacts")
    if not isinstance(anomaly_rows, list):
        raise ReleaseLoadError("full model candidate anomaly_artifacts must be an array")
    full_anomaly = {
        (item.get("hand"), item.get("view"), item.get("family"), item.get("model_digest"))
        for item in anomaly_rows
        if isinstance(item, dict)
    }
    contract_anomaly = {
        (binding.hand.value, binding.view_id, binding.family.value, binding.asset.sha256)
        for binding in contract.anomaly_bindings
    }
    yolo = full_candidate.get("yolo")
    if (
        len(full_anomaly) != len(anomaly_rows)
        or full_anomaly != contract_anomaly
        or not isinstance(yolo, dict)
        or yolo.get("model_digest") != contract.yolo_model.sha256
    ):
        raise ReleaseLoadError("full model candidate assets differ from deployment contract")

    dataset_digest = files.checksums["provenance/dataset_release.json"]
    if candidate.get("dataset_manifest_sha256") != dataset_digest:
        raise ReleaseLoadError("model candidate does not bind copied dataset provenance bytes")
    dataset = files.read_json("provenance/dataset_release.json")
    expected_dataset_keys = {
        "schema_version", "dataset_release_id", "product", "topology_id",
        "topology_sha256", "roi_version", "roi_sha256", "bbox_migration_policy",
        "capture_gate_policy_sha256",
        "required_views", "hands", "sample_count", "part_count", "capture_set_count",
        "canonical_manifest_sha256", "capture_provenance_sha256", "bbox_audit_sha256",
        "canonical_semantics_sha256", "calibration_targets_sha256",
        "calibration_target_count", "dataset_provenance_sha256",
        "split_assignments_sha256", "split_policy", "split_policy_sha256",
        "canonical_png_codec", "canonical_png_codec_sha256",
        "adapter_manifest_sha256", "created_at",
    }
    if set(dataset) != expected_dataset_keys or dataset.get("schema_version") != 4:
        raise ReleaseLoadError("copied dataset release provenance envelope is malformed")
    if (
        dataset.get("dataset_release_id") != candidate.get("dataset_release_id")
        or dataset.get("product") != PRODUCT
        or dataset.get("topology_id") != contract.topology.topology_id
        or dataset.get("topology_sha256") != contract.topology.topology_sha256
        or dataset.get("roi_version") != contract.roi.roi_config_id
        or dataset.get("roi_sha256") != contract.roi.roi_sha256
        or dataset.get("capture_gate_policy_sha256")
        != contract.capture_gate_policy.sha256
        or dataset.get("required_views") != list(contract.topology.required_views)
        or not isinstance(dataset.get("hands"), list)
        or {hand.value for hand in contract.allowed_hands} != set(dataset["hands"])
    ):
        raise ReleaseLoadError("copied dataset provenance differs from deployment contract")
    if not isinstance(dataset.get("adapter_manifest_sha256"), dict) or set(
        dataset["adapter_manifest_sha256"]
    ) != {"yolo", "anomalib", "template"}:
        raise ReleaseLoadError("dataset provenance adapter manifests are incomplete")
    dataset_manifest = DatasetReleaseManifest(
        schema_version=dataset["schema_version"],
        dataset_release_id=dataset["dataset_release_id"],
        product=dataset["product"],
        topology_id=dataset["topology_id"],
        topology_sha256=dataset["topology_sha256"],
        roi_version=dataset["roi_version"],
        roi_sha256=dataset["roi_sha256"],
        capture_gate_policy_sha256=dataset["capture_gate_policy_sha256"],
        bbox_migration_policy=dataset["bbox_migration_policy"],
        required_views=tuple(dataset["required_views"]),
        hands=tuple(dataset["hands"]),
        sample_count=dataset["sample_count"],
        part_count=dataset["part_count"],
        capture_set_count=dataset["capture_set_count"],
        canonical_manifest_sha256=dataset["canonical_manifest_sha256"],
        capture_provenance_sha256=dataset["capture_provenance_sha256"],
        bbox_audit_sha256=dataset["bbox_audit_sha256"],
        canonical_semantics_sha256=dataset["canonical_semantics_sha256"],
        calibration_targets_sha256=dataset["calibration_targets_sha256"],
        calibration_target_count=dataset["calibration_target_count"],
        dataset_provenance_sha256=dataset["dataset_provenance_sha256"],
        split_assignments_sha256=dataset["split_assignments_sha256"],
        split_policy=dataset["split_policy"],
        split_policy_sha256=dataset["split_policy_sha256"],
        canonical_png_codec=dataset["canonical_png_codec"],
        canonical_png_codec_sha256=dataset["canonical_png_codec_sha256"],
        adapter_manifest_sha256=dataset["adapter_manifest_sha256"],
        created_at=dataset["created_at"],
    )
    _validate_dataset_capture_provenance(
        files,
        dataset_manifest,
        capture_gate_policy=capture_gate_policy,
        capture_gate_policy_sha256=capture_gate_policy_sha256,
    )
    _validate_dataset_governance(files, dataset_manifest)
    _validate_validation_publication_snapshot(
        files,
        candidate=candidate,
        full_candidate=full_candidate,
        contract=contract,
        dataset=dataset_manifest,
    )

    code_version = files.read_json("provenance/code_version.json")
    _validate_code_version(code_version)
    return dataset_manifest


def _validate_dataset_governance(
    files: VerifiedReleaseFiles,
    dataset: DatasetReleaseManifest,
) -> None:
    """Rebuild the approved semantics, split policy, and canonical PNG evidence."""
    digest_by_path = {
        "provenance/dataset_canonical_semantics.json": (
            dataset.canonical_semantics_sha256
        ),
        "provenance/dataset_calibration_targets.json": (
            dataset.calibration_targets_sha256
        ),
        "provenance/dataset_governance.json": dataset.dataset_provenance_sha256,
        "provenance/dataset_split_assignments.json": (
            dataset.split_assignments_sha256
        ),
    }
    for relative, expected_digest in digest_by_path.items():
        if files.checksums.get(relative) != expected_digest:
            raise ReleaseLoadError(
                f"copied dataset governance digest mismatch: {relative}"
            )
    try:
        semantics_payload = files.read_json(
            "provenance/dataset_canonical_semantics.json"
        )
        semantics = CanonicalSemanticsSnapshot.from_mapping(semantics_payload)
        targets_payload = files.read_json(
            "provenance/dataset_calibration_targets.json"
        )
        targets = CalibrationTargetsSnapshot.from_mapping(targets_payload)
        split_policy = SplitPolicy.from_mapping(dataset.split_policy)
        png_codec = CanonicalPngCodec.from_mapping(dataset.canonical_png_codec)
        split_payload = files.read_json(
            "provenance/dataset_split_assignments.json"
        )
        governance_payload = files.read_json("provenance/dataset_governance.json")
    except (TypeError, ValueError) as error:
        raise ReleaseLoadError(f"copied dataset governance is malformed: {error}") from error
    if (
        semantics.sha256 != dataset.canonical_semantics_sha256
        or files.read_bytes("provenance/dataset_canonical_semantics.json")
        != semantics.canonical_bytes
        or targets.sha256 != dataset.calibration_targets_sha256
        or files.read_bytes("provenance/dataset_calibration_targets.json")
        != targets.canonical_bytes
        or len(targets.targets) != dataset.calibration_target_count
        or targets.dataset_release_id != dataset.dataset_release_id
        or targets.topology_id != dataset.topology_id
        or targets.roi_version != dataset.roi_version
        or split_policy.sha256 != dataset.split_policy_sha256
        or png_codec.sha256 != dataset.canonical_png_codec_sha256
    ):
        raise ReleaseLoadError("copied dataset governance identity is inconsistent")
    if (
        set(split_payload) != {
            "schema_version",
            "policy",
            "policy_sha256",
            "assignments",
        }
        or split_payload.get("schema_version") != 2
        or split_payload.get("policy") != split_policy.as_dict()
        or split_payload.get("policy_sha256") != split_policy.sha256
        or not isinstance(split_payload.get("assignments"), list)
        or not split_payload["assignments"]
        or any(
            not isinstance(item, dict)
            or set(item) != {"part_instance_id", "split", "stratum"}
            for item in split_payload["assignments"]
        )
        or files.read_bytes("provenance/dataset_split_assignments.json")
        != canonical_json_bytes(split_payload)
    ):
        raise ReleaseLoadError("copied dataset split artifact is malformed or unbound")
    try:
        assignments = tuple(
            SplitAssignment(
                part_instance_id=item["part_instance_id"],
                split=item["split"],
                stratum=item["stratum"],
            )
            for item in split_payload["assignments"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ReleaseLoadError(f"copied split assignments are invalid: {error}") from error
    if assignments != tuple(
        sorted(assignments, key=lambda item: item.part_instance_id)
    ) or len({item.part_instance_id for item in assignments}) != len(assignments):
        raise ReleaseLoadError("copied split assignments are not canonical and unique")
    strata_by_part = {
        item.part_instance_id: item.stratum for item in assignments
    }
    if assignments != assign_grouped_splits(strata_by_part, policy=split_policy):
        raise ReleaseLoadError("copied split assignments are not reproducible from SplitPolicy")
    expected_governance = {
        "schema_version": 2,
        "canonical_semantics": {
            "relative_path": "canonical_semantics.json",
            "sha256": dataset.canonical_semantics_sha256,
            "approval": semantics.approval.as_dict(),
        },
        "calibration_targets": {
            "relative_path": "calibration_targets.json",
            "sha256": dataset.calibration_targets_sha256,
            "target_count": dataset.calibration_target_count,
            "approval": targets.approval.as_dict(),
        },
        "split": {
            "relative_path": "split_assignments.json",
            "artifact_sha256": dataset.split_assignments_sha256,
            "policy": split_policy.as_dict(),
            "policy_sha256": split_policy.sha256,
        },
        "canonical_png": {
            "codec": png_codec.as_dict(),
            "codec_sha256": png_codec.sha256,
        },
    }
    if (
        governance_payload != expected_governance
        or files.read_bytes("provenance/dataset_governance.json")
        != canonical_json_bytes(expected_governance)
    ):
        raise ReleaseLoadError("copied dataset governance summary is malformed or unbound")


def _validate_dataset_capture_provenance(
    files: VerifiedReleaseFiles,
    dataset: DatasetReleaseManifest,
    *,
    capture_gate_policy: CaptureGatePolicy,
    capture_gate_policy_sha256: str,
) -> None:
    """Rebuild the small capture provenance artifact sealed into deployment."""
    content = files.read_bytes("provenance/dataset_capture_provenance.jsonl")
    if hashlib.sha256(content).hexdigest() != dataset.capture_provenance_sha256:
        raise ReleaseLoadError("copied dataset capture provenance digest mismatch")
    lines = content.splitlines()
    if not lines or any(not line for line in lines):
        raise ReleaseLoadError("copied dataset capture provenance contains blank rows")
    try:
        rows = [
            CaptureProvenanceRow.from_mapping(json.loads(line))
            for line in lines
        ]
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ReleaseLoadError(
            f"copied dataset capture provenance is malformed: {error}"
        ) from error
    canonical = b"".join(canonical_json_bytes(row.as_dict()) for row in rows)
    if canonical != content:
        raise ReleaseLoadError("copied dataset capture provenance is not canonical JSONL")
    if rows != sorted(rows, key=lambda row: row.capture_set_id):
        raise ReleaseLoadError("copied dataset capture provenance order is not canonical")
    if len(rows) != dataset.capture_set_count or len(
        {row.capture_set_id for row in rows}
    ) != len(rows):
        raise ReleaseLoadError("copied dataset capture provenance identities are incomplete")
    policy_identities = {(row.policy_id, row.policy_sha256) for row in rows}
    hand_identity: dict[str, tuple[object, ...]] = {}
    for row in rows:
        expected = capture_gate_policy.provenance_for(
            Hand.parse(row.hand),
            policy_sha256=capture_gate_policy_sha256,
        )
        actual = CaptureGateProvenance(
            policy_id=row.policy_id,
            policy_sha256=row.policy_sha256,
            hand=Hand.parse(row.hand),
            topology_sha256=row.topology_sha256,
            acquisition_config_sha256=row.acquisition_config_sha256,
            quality_profile_sha256=row.quality_profile_sha256,
            registration_profile_sha256=row.registration_profile_sha256,
            registration_reference_sha256_by_view=(
                row.registration_reference_sha256_by_view
            ),
        )
        if (
            actual != expected
            or row.policy_sha256 != dataset.capture_gate_policy_sha256
            or row.topology_sha256 != dataset.topology_sha256
            or set(row.registration_reference_sha256_by_view)
            != set(dataset.required_views)
        ):
            raise ReleaseLoadError(
                f"copied dataset capture provenance differs from release: {row.capture_set_id}"
            )
        identity = (
            row.policy_id,
            row.policy_sha256,
            row.topology_sha256,
            row.acquisition_config_sha256,
            row.quality_profile_sha256,
            row.registration_profile_sha256,
            tuple(sorted(row.registration_reference_sha256_by_view.items())),
        )
        previous = hand_identity.setdefault(row.hand, identity)
        if previous != identity:
            raise ReleaseLoadError(
                f"copied dataset capture provenance is mixed within hand {row.hand!r}"
            )
    if len(policy_identities) != 1 or set(hand_identity) != set(dataset.hands):
        raise ReleaseLoadError("copied dataset capture policy/hand identities are inconsistent")


def _validate_validation_publication_snapshot(
    files: VerifiedReleaseFiles,
    *,
    candidate: dict[str, Any],
    full_candidate: dict[str, Any],
    contract: DeploymentContract,
    dataset: DatasetReleaseManifest,
) -> None:
    """Rebuild copied validation/calibration roots and check remaining validated assertions.

    The release contains every byte needed to independently reconstruct the
    validation and calibration publication roots. Registration and score-run
    publications are not copied into the release; their identities remain
    assertions certified by the reconstructed validation publication.
    """
    validation_id = candidate["validation_publication_id"]
    if any(token in validation_id for token in ("/", "\\", "\x00", "\n", "\r")):
        raise ReleaseLoadError("validation publication id is unsafe")
    prefix = f"provenance/validations/{validation_id}/"
    relative_names = {
        relative.removeprefix(prefix)
        for relative in files.checksums
        if relative.startswith(prefix)
    }
    expected_names = {
        "candidate.json",
        "deployment_assets.json",
        "validation.json",
        "heldout_acceptance_policy.json",
        "heldout_acceptance.json",
        "checksums.sha256",
        "publication_root.json",
    }
    if relative_names != expected_names:
        raise ReleaseLoadError("copied validation publication file set is incomplete or unexpected")

    checksum_bytes = files.read_bytes(f"{prefix}checksums.sha256")
    nested_checksums = _parse_checksum_index(checksum_bytes)
    expected_index = {
        "candidate.json",
        "deployment_assets.json",
        "validation.json",
        "heldout_acceptance_policy.json",
        "heldout_acceptance.json",
    }
    if set(nested_checksums) != expected_index:
        raise ReleaseLoadError("copied validation checksum index has an unexpected file set")
    for name, expected_digest in nested_checksums.items():
        content = files.read_bytes(f"{prefix}{name}")
        if hashlib.sha256(content).hexdigest() != expected_digest:
            raise ReleaseLoadError(f"copied validation file differs from nested index: {name}")

    root_sha256 = hashlib.sha256(checksum_bytes).hexdigest()
    publication_root = _load_json_bytes(
        files.read_bytes(f"{prefix}publication_root.json"),
        label="copied validation publication root",
    )
    if (
        set(publication_root) != {"algorithm", "publication_id", "root_sha256"}
        or files.read_bytes(f"{prefix}publication_root.json")
        != canonical_json_bytes(publication_root)
        or publication_root.get("algorithm") != "sha256(checksums.sha256 bytes)"
        or publication_root.get("publication_id") != validation_id
        or publication_root.get("root_sha256") != root_sha256
        or candidate.get("validation_publication_root_sha256") != root_sha256
    ):
        raise ReleaseLoadError("copied validation publication root is malformed or unbound")
    if nested_checksums["validation.json"] != candidate.get("validation_record_sha256"):
        raise ReleaseLoadError("copied validation record digest differs from release provenance")
    try:
        heldout_policy = load_heldout_acceptance_policy_bytes(
            files.read_bytes(f"{prefix}heldout_acceptance_policy.json")
        )
        heldout_decision = load_heldout_acceptance_decision_bytes(
            files.read_bytes(f"{prefix}heldout_acceptance.json")
        )
    except ValueError as error:
        raise ReleaseLoadError(
            f"copied held-out acceptance evidence is malformed: {error}"
        ) from error
    if (
        heldout_policy.sha256
        != candidate.get("heldout_acceptance_policy_sha256")
        or heldout_decision.sha256
        != candidate.get("heldout_acceptance_decision_sha256")
        or heldout_decision.policy_sha256 != heldout_policy.sha256
    ):
        raise ReleaseLoadError(
            "copied held-out acceptance evidence differs from release provenance"
        )

    nested_candidate_bytes = files.read_bytes(f"{prefix}candidate.json")
    if (
        nested_candidate_bytes != canonical_json_bytes(full_candidate)
        or nested_candidate_bytes != files.read_bytes("provenance/model_candidate_full.json")
    ):
        raise ReleaseLoadError("copied validation candidate differs from release candidate")
    deployment = _load_json_bytes(
        files.read_bytes(f"{prefix}deployment_assets.json"),
        label="copied validation deployment assets",
    )
    if (
        files.read_bytes(f"{prefix}deployment_assets.json")
        != canonical_json_bytes(deployment)
        or deployment.get("schema") != "zs32.deployment_assets"
        or deployment.get("schema_version") != 1
        or deployment.get("candidate_id") != candidate.get("candidate_id")
        or deployment.get("candidate_digest") != candidate.get("candidate_digest")
    ):
        raise ReleaseLoadError("copied validation deployment descriptor is malformed or unbound")

    validation = _load_json_bytes(
        files.read_bytes(f"{prefix}validation.json"),
        label="copied candidate validation record",
    )
    validation_keys = {
        "schema", "schema_version", "validation_id", "candidate_id", "candidate_digest",
        "input_candidate_descriptor_sha256", "input_template_assets_descriptor_sha256",
        "dataset_release_id", "dataset_manifest_sha256", "recipe_sha256", "contract_sha256",
        "calibration_artifact_sha256", "calibration_split_id", "test_split_id",
        "candidate_status_before", "candidate_status_after", "manual_promotion_required",
        "score_run_id", "score_run_root_sha256", "scores_sha256", "score_audit_sha256",
        "score_execution_receipt_sha256",
        "calibration_publication_id", "calibration_publication_root_sha256",
        "registration_publication_id", "registration_publication_root_sha256",
        "heldout_acceptance_policy_id", "heldout_acceptance_policy_version",
        "heldout_acceptance_policy_sha256", "heldout_acceptance_decision_sha256",
    }
    digest_fields = {
        "candidate_digest", "input_candidate_descriptor_sha256",
        "input_template_assets_descriptor_sha256", "dataset_manifest_sha256",
        "recipe_sha256", "contract_sha256", "calibration_artifact_sha256",
        "score_run_root_sha256", "scores_sha256", "score_audit_sha256",
        "score_execution_receipt_sha256",
        "calibration_publication_root_sha256", "registration_publication_root_sha256",
        "heldout_acceptance_policy_sha256", "heldout_acceptance_decision_sha256",
    }
    if (
        set(validation) != validation_keys
        or files.read_bytes(f"{prefix}validation.json") != canonical_json_bytes(validation)
        or any(
            not isinstance(validation.get(field), str)
            or re.fullmatch(r"[0-9a-f]{64}", validation[field]) is None
            for field in digest_fields
        )
    ):
        raise ReleaseLoadError("copied candidate validation record is malformed")

    calibration = files.read_json("calibration/calibration_artifact.json")
    try:
        reconstructed_calibration = reconstruct_calibration_artifact(calibration)
        expected_heldout_decision = evaluate_heldout_acceptance(
            heldout_policy,
            reconstructed_calibration.heldout_metrics,
            metrics_sha256=files.checksums["calibration/metrics.json"],
        )
    except (TypeError, ValueError) as error:
        raise ReleaseLoadError(
            f"held-out acceptance re-evaluation failed: {error}"
        ) from error
    if heldout_decision != expected_heldout_decision:
        raise ReleaseLoadError(
            "copied held-out acceptance decision differs from calibration metrics"
        )
    calibration_provenance = calibration.get("provenance")
    calibration_inputs = files.read_json("calibration/input_provenance.json")
    calibration_names = (
        "calibration_artifact.json",
        "input_provenance.json",
        "metrics.json",
        "model_thresholds.json",
        "template_thresholds.json",
    )
    calibration_checksum_bytes = "".join(
        f"{hashlib.sha256(files.read_bytes(f'calibration/{name}')).hexdigest()}  {name}\n"
        for name in sorted(calibration_names)
    ).encode("utf-8")
    calibration_publication_root_sha256 = hashlib.sha256(
        calibration_checksum_bytes
    ).hexdigest()
    if not isinstance(calibration_provenance, dict):
        raise ReleaseLoadError("calibration provenance is unavailable to validation snapshot")
    if (
        validation.get("calibration_publication_root_sha256")
        != calibration_publication_root_sha256
    ):
        raise ReleaseLoadError(
            "copied validation calibration publication root differs from release bytes"
        )
    if (
        validation.get("schema") != "zs32.candidate_validation"
        or validation.get("schema_version") != 3
        or validation.get("validation_id") != validation_id
        or validation.get("candidate_id") != candidate.get("candidate_id")
        or validation.get("candidate_digest") != candidate.get("candidate_digest")
        or validation.get("dataset_release_id") != dataset.dataset_release_id
        or validation.get("dataset_manifest_sha256")
        != candidate.get("dataset_manifest_sha256")
        or validation.get("recipe_sha256") != contract.recipe_sha256
        or validation.get("contract_sha256") != contract.contract_sha256
        or validation.get("calibration_artifact_sha256")
        != files.checksums["calibration/calibration_artifact.json"]
        or validation.get("calibration_split_id")
        != calibration_provenance.get("calibration_split_id")
        or validation.get("test_split_id") != calibration_provenance.get("test_split_id")
        or validation.get("candidate_status_before") != "registered"
        or validation.get("candidate_status_after") != "validated"
        or validation.get("manual_promotion_required") is not True
        or validation.get("heldout_acceptance_policy_id") != heldout_policy.policy_id
        or validation.get("heldout_acceptance_policy_version")
        != heldout_policy.policy_version
        or validation.get("heldout_acceptance_policy_sha256")
        != heldout_policy.sha256
        or validation.get("heldout_acceptance_decision_sha256")
        != heldout_decision.sha256
        or validation.get("score_run_id") != calibration_inputs.get("score_run_id")
        or validation.get("score_run_root_sha256")
        != calibration_inputs.get("score_run_root_sha256")
        or validation.get("scores_sha256") != calibration_inputs.get("scores_sha256")
        or validation.get("score_audit_sha256")
        != calibration_inputs.get("score_audit_sha256")
        or validation.get("score_execution_receipt_sha256")
        != calibration_inputs.get("score_execution_receipt_sha256")
        or validation.get("calibration_publication_id")
        != calibration_inputs.get("calibration_id")
        or not isinstance(validation.get("registration_publication_id"), str)
        or not validation["registration_publication_id"].strip()
    ):
        raise ReleaseLoadError(
            "copied candidate validation record differs from release provenance"
        )


def _validate_code_version(payload: dict[str, Any]) -> None:
    expected = {
        "schema_version",
        "git_commit",
        "git_tree",
        "dirty",
        "dependency_lock_sha256",
        "build_id",
    }
    if set(payload) != expected or payload.get("schema_version") != 1:
        raise ReleaseLoadError("release code_version provenance envelope is malformed")
    for field in ("git_commit", "git_tree"):
        value = payload.get(field)
        if not isinstance(value, str) or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", value) is None:
            raise ReleaseLoadError(f"release code_version has invalid {field}")
    dependency = payload.get("dependency_lock_sha256")
    if not isinstance(dependency, str) or re.fullmatch(r"[0-9a-f]{64}", dependency) is None:
        raise ReleaseLoadError("release code_version has invalid dependency_lock_sha256")
    if payload.get("dirty") is not False:
        raise ReleaseLoadError("release code_version must bind a clean source worktree")
    if not isinstance(payload.get("build_id"), str) or not payload["build_id"].strip():
        raise ReleaseLoadError("release code_version has invalid build_id")


def load_verified_deployment_release(root: Path) -> VerifiedDeploymentRelease:
    """Load a verified release and normalize all schema/I/O failures."""
    try:
        return _load_verified_deployment_release(root)
    except ReleaseLoadError:
        raise
    except (OSError, PublicationError, TypeError, ValueError) as error:
        raise ReleaseLoadError(f"release semantic verification failed: {error}") from error


def validate_staged_deployment_release(
    root: Path,
    *,
    expected_release_id: str,
) -> VerifiedDeploymentRelease:
    """Run the complete loader against a pre-rename staging directory."""
    try:
        return _load_verified_deployment_release(
            root,
            expected_publication_id=expected_release_id,
        )
    except ReleaseLoadError:
        raise
    except (OSError, PublicationError, TypeError, ValueError) as error:
        raise ReleaseLoadError(f"staged release semantic verification failed: {error}") from error


def _validate_yolo_bundle(files: VerifiedReleaseFiles, contract: DeploymentContract) -> None:
    metadata = files.read_json("models/yolo/metadata.json")
    if set(metadata) != {
        "schema_version", "model_sha256", "bundle_sha256", "runtime", "provenance",
        "asset_sha256",
    } or metadata.get("schema_version") != 3:
        raise ReleaseLoadError("YOLO metadata envelope is malformed")
    if metadata.get("model_sha256") != contract.yolo_model.sha256:
        raise ReleaseLoadError("YOLO metadata model digest differs from deployment contract")
    bundle_digest = metadata.get("bundle_sha256")
    if not isinstance(bundle_digest, str) or re.fullmatch(r"[0-9a-f]{64}", bundle_digest) is None:
        raise ReleaseLoadError("YOLO metadata bundle_sha256 is invalid")
    runtime = metadata.get("runtime")
    provenance = metadata.get("provenance")
    if not isinstance(runtime, dict) or set(runtime) != {
        "imgsz", "candidate_conf", "iou", "max_det", "single_class_name"
    }:
        raise ReleaseLoadError("YOLO runtime metadata is malformed")
    if not isinstance(provenance, dict) or set(provenance) != {
        "dataset_release_id", "dataset_manifest_digest", "yolo_export_publication_id",
        "yolo_export_root_sha256", "yolo_export_manifest_digest",
        "yolo_export_data_yaml_digest", "yolo_export_policy_digest",
        "args_data_reference",
        "ultralytics_version_or_commit", "trainer_source", "training_receipt_digest",
        "training_seed", "run_id", "run_name",
    }:
        raise ReleaseLoadError("YOLO training provenance metadata is malformed")
    candidate = files.read_json("provenance/model_candidate.json")
    if (
        provenance.get("dataset_release_id") != candidate.get("dataset_release_id")
        or provenance.get("dataset_manifest_digest") != candidate.get("dataset_manifest_sha256")
        or not isinstance(provenance.get("ultralytics_version_or_commit"), str)
        or not provenance["ultralytics_version_or_commit"].strip()
        or isinstance(provenance.get("training_seed"), bool)
        or not isinstance(provenance.get("training_seed"), int)
        or provenance["training_seed"] < 0
        or not isinstance(provenance.get("run_name"), str)
        or not provenance["run_name"].strip()
        or not isinstance(provenance.get("run_id"), str)
        or not provenance["run_id"].strip()
    ):
        raise ReleaseLoadError("YOLO training provenance differs from release candidate")
    assets = metadata.get("asset_sha256")
    if not isinstance(assets, dict):
        raise ReleaseLoadError("YOLO metadata asset_sha256 must be an object")
    expected = {
        "best.pt": "models/yolo/best.pt",
        "train_args.yaml": "models/yolo/train_args.yaml",
        "data.yaml": "models/yolo/data.yaml",
        "class_names.yaml": "models/yolo/class_names.yaml",
        "training_receipt.json": "models/yolo/training_receipt.json",
    }
    if set(assets) != set(expected):
        raise ReleaseLoadError("YOLO metadata asset_sha256 fields are incomplete or unknown")
    for name, relative in expected.items():
        if assets.get(name) != files.checksums.get(relative):
            raise ReleaseLoadError(f"YOLO metadata/index digest mismatch for {name}")
    try:
        weights_line = files.read_bytes("models/yolo/weights.sha256").decode("utf-8")
    except UnicodeError as error:
        raise ReleaseLoadError("models/yolo/weights.sha256 must be UTF-8") from error
    if weights_line != f"{contract.yolo_model.sha256}  best.pt\n":
        raise ReleaseLoadError("models/yolo/weights.sha256 is not canonical")
    specification = YoloDeploymentSpec(
        weights=AssetFile(
            role="weights",
            path=files.path("models/yolo/best.pt"),
            sha256=files.checksums["models/yolo/best.pt"],
            media_type="application/x-pytorch",
            logical_path="best.pt",
        ),
        args_yaml=AssetFile(
            role="train_args",
            path=files.path("models/yolo/train_args.yaml"),
            sha256=files.checksums["models/yolo/train_args.yaml"],
            media_type="application/yaml",
            logical_path="args.yaml",
        ),
        data_yaml=AssetFile(
            role="dataset_config",
            path=files.path("models/yolo/data.yaml"),
            sha256=files.checksums["models/yolo/data.yaml"],
            media_type="application/yaml",
            logical_path="data.yaml",
        ),
        class_names_yaml=AssetFile(
            role="class_names",
            path=files.path("models/yolo/class_names.yaml"),
            sha256=files.checksums["models/yolo/class_names.yaml"],
            media_type="application/yaml",
            logical_path="class_names.yaml",
        ),
        training_receipt=AssetFile(
            role="training_receipt",
            path=files.path("models/yolo/training_receipt.json"),
            sha256=files.checksums["models/yolo/training_receipt.json"],
            media_type="application/json",
            logical_path="training_receipt.json",
        ),
        model_digest=metadata["model_sha256"],
        bundle_digest=metadata["bundle_sha256"],
        provenance=YoloTrainingProvenance(
            dataset_release_id=provenance["dataset_release_id"],
            dataset_manifest_digest=provenance["dataset_manifest_digest"],
            yolo_export_publication_id=provenance["yolo_export_publication_id"],
            yolo_export_root_sha256=provenance["yolo_export_root_sha256"],
            yolo_export_manifest_digest=provenance["yolo_export_manifest_digest"],
            yolo_export_data_yaml_digest=provenance["yolo_export_data_yaml_digest"],
            yolo_export_policy_digest=provenance["yolo_export_policy_digest"],
            args_data_reference=provenance["args_data_reference"],
            ultralytics_version_or_commit=provenance["ultralytics_version_or_commit"],
            trainer_source=provenance["trainer_source"],
            training_receipt_digest=provenance["training_receipt_digest"],
            training_seed=provenance["training_seed"],
            run_id=provenance["run_id"],
            run_name=provenance["run_name"],
        ),
        runtime=YoloRuntimeSettings(
            imgsz=runtime["imgsz"],
            candidate_conf=runtime["candidate_conf"],
            iou=runtime["iou"],
            max_det=runtime["max_det"],
            single_class_name=runtime["single_class_name"],
        ),
    )
    specification.verify()


def _validate_calibration_artifact(
    files: VerifiedReleaseFiles,
    contract: DeploymentContract,
    dataset: DatasetReleaseManifest,
) -> None:
    payload = files.read_json("calibration/calibration_artifact.json")
    expected_keys = {
        "schema",
        "schema_version",
        "provenance",
        "parameters",
        "required_dual_groups",
        "required_template_groups",
        "dual_thresholds",
        "template_thresholds",
        "heldout_metrics",
        "calibration_valid",
    }
    if set(payload) != expected_keys or payload.get("schema") != "zs32.calibration":
        raise ReleaseLoadError("calibration artifact envelope is malformed")
    if payload.get("schema_version") != 1 or payload.get("calibration_valid") is not True:
        raise ReleaseLoadError("calibration artifact is not valid for deployment")
    artifact = reconstruct_calibration_artifact(payload)
    if artifact.payload() != payload or not artifact.calibration_valid:
        raise ReleaseLoadError("calibration artifact does not round-trip through strict contracts")
    if files.read_bytes("calibration/calibration_artifact.json") != canonical_json_bytes(payload):
        raise ReleaseLoadError("calibration artifact bytes are not canonical JSON")
    artifact_digest = files.checksums["calibration/calibration_artifact.json"]
    template_snapshot = files.read_json("calibration/template_thresholds.json")
    model_snapshot = files.read_json("calibration/model_thresholds.json")
    metrics_snapshot = files.read_json("calibration/metrics.json")
    expected_template_snapshot = {
        "schema_version": 1,
        "calibration_artifact_sha256": artifact_digest,
        "thresholds": payload["template_thresholds"],
    }
    expected_model_snapshot = {
        "schema_version": 1,
        "calibration_artifact_sha256": artifact_digest,
        "thresholds": payload["dual_thresholds"],
    }
    if template_snapshot != expected_template_snapshot:
        raise ReleaseLoadError("template threshold snapshot differs from calibration artifact")
    if model_snapshot != expected_model_snapshot:
        raise ReleaseLoadError("model threshold snapshot differs from calibration artifact")
    if metrics_snapshot != payload["heldout_metrics"]:
        raise ReleaseLoadError("metrics snapshot differs from calibration artifact")
    for relative, snapshot in (
        ("calibration/template_thresholds.json", template_snapshot),
        ("calibration/model_thresholds.json", model_snapshot),
        ("calibration/metrics.json", metrics_snapshot),
    ):
        if files.read_bytes(relative) != canonical_json_bytes(snapshot):
            raise ReleaseLoadError(f"calibration snapshot is not canonical JSON: {relative}")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ReleaseLoadError("calibration provenance must be an object")
    if (
        provenance.get("recipe_digest") != contract.recipe_sha256
        or provenance.get("profile_digest") != contract.fusion_policy_sha256
        or provenance.get("topology_digest") != contract.topology.topology_sha256
        or provenance.get("roi_digest") != contract.roi.roi_sha256
    ):
        raise ReleaseLoadError("calibration provenance differs from deployment contract")
    candidate = files.read_json("provenance/model_candidate.json")
    input_provenance = files.read_json("calibration/input_provenance.json")
    expected_input_keys = {
        "schema", "schema_version", "calibration_id", "score_run_id",
        "score_run_root_sha256", "scores_sha256", "score_audit_sha256",
        "score_execution_receipt_sha256",
        "candidate_id", "candidate_digest", "candidate_descriptor_sha256",
        "recipe_digest", "dataset_release_id", "dataset_manifest_sha256",
        "calibration_targets_sha256", "calibration_target_count",
        "calibration_split_id", "test_split_id",
    }
    digest_fields = (
        "score_run_root_sha256", "scores_sha256", "score_audit_sha256",
        "score_execution_receipt_sha256",
        "candidate_descriptor_sha256",
        "calibration_targets_sha256",
    )
    if (
        set(input_provenance) != expected_input_keys
        or input_provenance.get("schema") != "zs32.calibration_inputs"
        or input_provenance.get("schema_version") != 3
        or any(
            not isinstance(input_provenance.get(field), str)
            or re.fullmatch(r"[0-9a-f]{64}", input_provenance[field]) is None
            for field in digest_fields
        )
        or any(
            not isinstance(input_provenance.get(field), str)
            or not input_provenance[field].strip()
            for field in ("calibration_id", "score_run_id")
        )
        or input_provenance.get("candidate_id") != candidate.get("candidate_id")
        or input_provenance.get("candidate_digest") != candidate.get("candidate_digest")
        or input_provenance.get("recipe_digest") != contract.recipe_sha256
        or input_provenance.get("dataset_release_id") != candidate.get("dataset_release_id")
        or input_provenance.get("dataset_manifest_sha256")
        != candidate.get("dataset_manifest_sha256")
        or input_provenance.get("calibration_targets_sha256")
        != dataset.calibration_targets_sha256
        or input_provenance.get("calibration_target_count")
        != dataset.calibration_target_count
        or input_provenance.get("calibration_split_id")
        != provenance.get("calibration_split_id")
        or input_provenance.get("test_split_id") != provenance.get("test_split_id")
        or files.read_bytes("calibration/input_provenance.json")
        != canonical_json_bytes(input_provenance)
    ):
        raise ReleaseLoadError("calibration input provenance is malformed or inconsistent")
    expected_model_digests = {
        contract.yolo_model.sha256,
        *(binding.asset.sha256 for binding in contract.template_bindings),
        *(binding.asset.sha256 for binding in contract.anomaly_bindings),
    }
    raw_model_digests = provenance.get("model_digests")
    if (
        provenance.get("dataset_release_id") != candidate.get("dataset_release_id")
        or provenance.get("dataset_manifest_digest") != candidate.get("dataset_manifest_sha256")
        or not isinstance(provenance.get("calibration_split_id"), str)
        or not provenance["calibration_split_id"].strip()
        or not isinstance(provenance.get("test_split_id"), str)
        or not provenance["test_split_id"].strip()
        or provenance["calibration_split_id"] == provenance["test_split_id"]
        or not isinstance(raw_model_digests, list)
        or any(not isinstance(item, str) for item in raw_model_digests)
        or set(raw_model_digests) != expected_model_digests
        or len(raw_model_digests) != len(set(raw_model_digests))
    ):
        raise ReleaseLoadError("calibration dataset/split/model provenance is invalid")
    metrics = payload.get("heldout_metrics")
    if not isinstance(metrics, dict) or metrics.get("evaluation_complete") is not True:
        raise ReleaseLoadError("calibration held-out evaluation is incomplete")
    if (
        metrics.get("dataset_release_id") != provenance.get("dataset_release_id")
        or metrics.get("test_split_id") != provenance.get("test_split_id")
    ):
        raise ReleaseLoadError("calibration held-out metrics provenance is inconsistent")

    required_groups = {
        (group.hand.value, group.view_id, group.branch.value)
        for group in contract.required_evidence_groups
    }
    raw_required = payload.get("required_dual_groups")
    if not isinstance(raw_required, list):
        raise ReleaseLoadError("calibration required_dual_groups must be an array")
    artifact_required = {
        (item.get("hand"), item.get("view"), item.get("branch"))
        for item in raw_required
        if isinstance(item, dict)
    }
    if len(artifact_required) != len(raw_required) or artifact_required != required_groups:
        raise ReleaseLoadError("calibration required dual groups differ from deployment contract")

    dual = payload.get("dual_thresholds")
    if not isinstance(dual, list):
        raise ReleaseLoadError("calibration dual_thresholds must be an array")
    dual_by_key = {
        (item.get("hand"), item.get("view"), item.get("branch")): item
        for item in dual
        if isinstance(item, dict)
    }
    if len(dual_by_key) != len(dual) or set(dual_by_key) != required_groups:
        raise ReleaseLoadError("calibration dual threshold groups are incomplete or duplicated")
    for threshold in contract.model_thresholds:
        item = dual_by_key[(threshold.hand.value, threshold.view_id, threshold.branch.value)]
        if (
            item.get("status") != "ok"
            or item.get("low") != threshold.low
            or item.get("high") != threshold.high
            or item.get("model_digest") != threshold.model_sha256
            or item.get("roi_version") != threshold.roi_config_id
            or item.get("dataset_release_id") != provenance.get("dataset_release_id")
            or item.get("calibration_split_id") != provenance.get("calibration_split_id")
        ):
            raise ReleaseLoadError("calibration dual threshold differs from deployment contract")

    template_expected = {
        (item.hand.value, item.view_id): item for item in contract.template_thresholds
    }
    raw_template_required = payload.get("required_template_groups")
    if not isinstance(raw_template_required, list):
        raise ReleaseLoadError("calibration required_template_groups must be an array")
    template_required_keys = {
        (item.get("hand"), item.get("view"))
        for item in raw_template_required
        if isinstance(item, dict)
    }
    if (
        len(template_required_keys) != len(raw_template_required)
        or template_required_keys != set(template_expected)
    ):
        raise ReleaseLoadError("calibration required template groups differ from deployment contract")
    template = payload.get("template_thresholds")
    if not isinstance(template, list):
        raise ReleaseLoadError("calibration template_thresholds must be an array")
    template_by_key = {
        (item.get("hand"), item.get("view")): item
        for item in template
        if isinstance(item, dict)
    }
    if len(template_by_key) != len(template) or set(template_by_key) != set(template_expected):
        raise ReleaseLoadError("calibration template threshold groups are incomplete or duplicated")
    for key, threshold in template_expected.items():
        item = template_by_key[key]
        if (
            item.get("status") != "ok"
            or item.get("threshold") != threshold.threshold
            or item.get("model_digest") != threshold.template_sha256
            or item.get("roi_version") != contract.roi.roi_config_id
            or item.get("dataset_release_id") != provenance.get("dataset_release_id")
            or item.get("calibration_split_id") != provenance.get("calibration_split_id")
            or item.get("roi_digest") != contract.roi.roi_sha256
        ):
            raise ReleaseLoadError("calibration template threshold differs from deployment contract")


def _exact_object(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = set(value) if isinstance(value, dict) else set()
        raise ReleaseLoadError(
            f"{label} fields are malformed; missing={sorted(keys - actual)}, "
            f"unknown={sorted(actual - keys)}"
        )
    return value


def _object_array(value: object, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ReleaseLoadError(f"{label} must be an array")
    output: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ReleaseLoadError(f"{label}[{index}] must be an object")
        output.append(item)
    return output


def reconstruct_calibration_artifact(payload: dict[str, Any]) -> CalibrationArtifact:
    """Rebuild every nested calibration contract instead of trusting flags.

    Candidate validation and final release loading deliberately share this
    exact parser so an artifact cannot pass promotion and later acquire a
    different interpretation at deployment time.
    """
    provenance_payload = _exact_object(
        payload["provenance"],
        {
            "recipe_digest", "profile_digest", "topology_digest", "roi_digest",
            "dataset_release_id", "dataset_manifest_digest", "calibration_split_id",
            "test_split_id", "model_digests",
        },
        "calibration.provenance",
    )
    parameters_payload = _exact_object(
        payload["parameters"],
        {"target_defect_recall", "normal_quantile", "min_normal_parts", "min_defect_parts"},
        "calibration.parameters",
    )
    provenance = CalibrationProvenance(
        recipe_digest=provenance_payload["recipe_digest"],
        profile_digest=provenance_payload["profile_digest"],
        topology_digest=provenance_payload["topology_digest"],
        roi_digest=provenance_payload["roi_digest"],
        dataset_release_id=provenance_payload["dataset_release_id"],
        dataset_manifest_digest=provenance_payload["dataset_manifest_digest"],
        calibration_split_id=provenance_payload["calibration_split_id"],
        test_split_id=provenance_payload["test_split_id"],
        model_digests=tuple(provenance_payload["model_digests"]),
    )
    parameters = FitParameters(**parameters_payload)
    required_dual: list[ScoreGroup] = []
    for index, item in enumerate(_object_array(payload["required_dual_groups"], "required_dual_groups")):
        row = _exact_object(
            item,
            {"hand", "view", "branch", "model_digest", "roi_version"},
            f"required_dual_groups[{index}]",
        )
        required_dual.append(
            ScoreGroup(
                row["hand"], row["view"], ScoreBranch(row["branch"]),
                row["model_digest"], row["roi_version"],
            )
        )
    required_template: list[TemplateCalibrationGroup] = []
    for index, item in enumerate(
        _object_array(payload["required_template_groups"], "required_template_groups")
    ):
        row = _exact_object(
            item,
            {"hand", "view", "model_digest", "roi_version", "roi_digest"},
            f"required_template_groups[{index}]",
        )
        required_template.append(TemplateCalibrationGroup(**row))
    dual_thresholds: list[DualThreshold] = []
    for index, item in enumerate(_object_array(payload["dual_thresholds"], "dual_thresholds")):
        row = _exact_object(
            item,
            {
                "dataset_release_id", "calibration_split_id", "hand", "view", "branch",
                "model_digest", "roi_version", "low", "high", "normal_count",
                "defect_count", "status",
            },
            f"dual_thresholds[{index}]",
        )
        group = ScoreGroup(
            row["hand"], row["view"], ScoreBranch(row["branch"]),
            row["model_digest"], row["roi_version"],
        )
        dual_thresholds.append(
            DualThreshold(
                group=group,
                dataset_release_id=row["dataset_release_id"],
                calibration_split_id=row["calibration_split_id"],
                low=row["low"],
                high=row["high"],
                normal_count=row["normal_count"],
                defect_count=row["defect_count"],
                status=row["status"],
            )
        )
    template_thresholds: list[TemplateThresholdFit] = []
    for index, item in enumerate(
        _object_array(payload["template_thresholds"], "template_thresholds")
    ):
        row = _exact_object(
            item,
            {
                "hand", "view", "model_digest", "roi_version", "roi_digest",
                "dataset_release_id", "calibration_split_id", "threshold",
                "normal_count", "defect_count", "status",
            },
            f"template_thresholds[{index}]",
        )
        template_thresholds.append(
            TemplateThresholdFit(
                slot=ModelSlot(row["hand"], row["view"]),
                model_digest=row["model_digest"],
                roi_version=row["roi_version"],
                roi_digest=row["roi_digest"],
                dataset_release_id=row["dataset_release_id"],
                calibration_split_id=row["calibration_split_id"],
                threshold=row["threshold"],
                normal_count=row["normal_count"],
                defect_count=row["defect_count"],
                status=row["status"],
            )
        )
    metrics_payload = _exact_object(
        payload["heldout_metrics"],
        {
            "dataset_release_id", "test_split_id", "part_count", "normal_part_count",
            "defect_part_count", "clear_count", "gray_count", "strong_count",
            "template_ng_count", "incomplete_count", "defect_escape_count",
            "normal_reject_count", "defect_escape_rate", "normal_reject_rate",
            "review_rate", "evaluation_complete",
        },
        "calibration.heldout_metrics",
    )
    return CalibrationArtifact(
        provenance=provenance,
        parameters=parameters,
        required_dual_groups=tuple(required_dual),
        required_template_groups=tuple(required_template),
        dual_thresholds=tuple(dual_thresholds),
        template_thresholds=tuple(template_thresholds),
        heldout_metrics=HeldOutPartMetrics(**metrics_payload),
        calibration_valid=payload["calibration_valid"],
    )


def _validate_template_groups(
    files: VerifiedReleaseFiles,
    contract: DeploymentContract,
    *,
    expected_train_split_id: str,
) -> None:
    candidate = files.read_json("provenance/model_candidate.json")
    for binding in contract.template_bindings:
        group_root = PurePosixPath(binding.asset.relative_path).parent
        metadata_relative = (group_root / "metadata.json").as_posix()
        if metadata_relative not in files.checksums:
            raise ReleaseLoadError(f"template group metadata is missing: {metadata_relative}")
        metadata = files.read_json(metadata_relative)
        expected_keys = {
            "schema_version", "hand", "view", "model_sha256", "template_version",
            "roi_version", "roi_sha256", "dataset_release_id",
            "dataset_manifest_sha256", "train_split_id", "recipe_sha256",
            "framework_version", "training_parameters", "execution_receipt",
            "reference_templates",
        }
        if set(metadata) != expected_keys or metadata.get("schema_version") != 1:
            raise ReleaseLoadError(f"template metadata envelope is malformed: {metadata_relative}")
        expected_identity = (binding.hand.value, binding.view_id, binding.asset.sha256)
        actual_identity = (metadata.get("hand"), metadata.get("view"), metadata.get("model_sha256"))
        if actual_identity != expected_identity:
            raise ReleaseLoadError(f"template metadata identity mismatch: {metadata_relative}")
        if (
            metadata.get("roi_version") != contract.roi.roi_config_id
            or metadata.get("roi_sha256") != contract.roi.roi_sha256
            or metadata.get("dataset_release_id") != candidate.get("dataset_release_id")
            or metadata.get("dataset_manifest_sha256") != candidate.get("dataset_manifest_sha256")
            or metadata.get("train_split_id") != expected_train_split_id
            or metadata.get("recipe_sha256") != contract.recipe_sha256
            or not isinstance(metadata.get("train_split_id"), str)
            or not metadata["train_split_id"].strip()
            or not isinstance(metadata.get("template_version"), str)
            or not metadata["template_version"].strip()
            or not isinstance(metadata.get("framework_version"), str)
            or not metadata["framework_version"].strip()
            or not isinstance(metadata.get("training_parameters"), dict)
            or not isinstance(metadata.get("execution_receipt"), dict)
        ):
            raise ReleaseLoadError(f"template metadata provenance mismatch: {metadata_relative}")
        try:
            template_receipt = ExecutionReceipt.from_mapping(metadata["execution_receipt"])
        except (TypeError, ValueError) as error:
            raise ReleaseLoadError(
                f"template execution receipt is invalid: {metadata_relative}: {error}"
            ) from error
        if template_receipt.operation != "train_template":
            raise ReleaseLoadError(
                f"template execution receipt operation is invalid: {metadata_relative}"
            )
        references = metadata.get("reference_templates")
        if not isinstance(references, list) or not references:
            raise ReleaseLoadError(f"template group has no reference templates: {metadata_relative}")
        seen: set[str] = set()
        for reference in references:
            if not isinstance(reference, dict) or set(reference) != {"relative_path", "sha256"}:
                raise ReleaseLoadError(f"malformed template reference metadata: {metadata_relative}")
            relative = reference["relative_path"]
            digest = reference["sha256"]
            if not isinstance(relative, str) or relative in seen:
                raise ReleaseLoadError(f"duplicate/invalid template reference path: {metadata_relative}")
            seen.add(relative)
            expected_prefix = (group_root / "templates").as_posix() + "/"
            if not relative.startswith(expected_prefix):
                raise ReleaseLoadError(
                    f"template reference must stay inside its slot group: {relative}"
                )
            if files.checksums.get(relative) != digest:
                raise ReleaseLoadError(f"template reference digest mismatch: {relative}")
        model = files.read_json(binding.asset.relative_path)
        model_keys = {
            "schema", "schema_version", "hand", "view", "method", "risk",
            "preprocessing", "references", "template_version", "roi_version",
            "roi_sha256", "dataset_release_id", "dataset_manifest_sha256",
            "train_split_id", "recipe_sha256", "train_part_count",
            "framework_version", "training_parameters", "execution_receipt",
        }
        if not isinstance(model, dict) or set(model) != model_keys:
            raise ReleaseLoadError(f"template model envelope is malformed: {binding.asset.relative_path}")
        if (
            model.get("schema") != "zs32.opencv_template"
            or model.get("schema_version") != 1
            or model.get("method") != "cv2.TM_CCOEFF_NORMED"
            or model.get("risk") != "1-similarity"
            or (model.get("hand"), model.get("view")) != (binding.hand.value, binding.view_id)
            or model.get("template_version") != metadata.get("template_version")
            or model.get("roi_version") != metadata.get("roi_version")
            or model.get("roi_sha256") != metadata.get("roi_sha256")
            or model.get("dataset_release_id") != metadata.get("dataset_release_id")
            or model.get("dataset_manifest_sha256") != metadata.get("dataset_manifest_sha256")
            or model.get("train_split_id") != metadata.get("train_split_id")
            or model.get("recipe_sha256") != metadata.get("recipe_sha256")
            or model.get("framework_version") != metadata.get("framework_version")
            or model.get("training_parameters") != metadata.get("training_parameters")
            or model.get("execution_receipt") != metadata.get("execution_receipt")
            or isinstance(model.get("train_part_count"), bool)
            or not isinstance(model.get("train_part_count"), int)
            or model["train_part_count"] <= 0
        ):
            raise ReleaseLoadError(f"template model provenance is invalid: {binding.asset.relative_path}")
        preprocessing = model.get("preprocessing")
        if (
            not isinstance(preprocessing, dict)
            or set(preprocessing) != {"color", "resize", "width", "gaussian_kernel", "max_shift"}
            or preprocessing.get("color") != "grayscale"
            or preprocessing.get("resize") != "aspect_preserving_width"
            or preprocessing.get("gaussian_kernel") != 3
            or isinstance(preprocessing.get("width"), bool)
            or not isinstance(preprocessing.get("width"), int)
            or preprocessing["width"] <= 0
            or isinstance(preprocessing.get("max_shift"), bool)
            or not isinstance(preprocessing.get("max_shift"), int)
            or preprocessing["max_shift"] < 0
        ):
            raise ReleaseLoadError(f"template preprocessing is invalid: {binding.asset.relative_path}")
        model_references = model.get("references")
        if not isinstance(model_references, list) or len(model_references) != len(references):
            raise ReleaseLoadError(f"template model reference set is incomplete: {binding.asset.relative_path}")
        for index, (record, reference) in enumerate(
            zip(model_references, references, strict=True),
            start=1,
        ):
            if (
                not isinstance(record, dict)
                or set(record) != {"index", "artifact_sha256", "source_crop_sha256"}
                or record.get("index") != index
                or record.get("artifact_sha256") != reference.get("sha256")
                or not isinstance(record.get("source_crop_sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", record["source_crop_sha256"]) is None
            ):
                raise ReleaseLoadError(
                    f"template model reference provenance is invalid: {binding.asset.relative_path}"
                )


def _validate_anomaly_groups(
    files: VerifiedReleaseFiles,
    contract: DeploymentContract,
    *,
    expected_train_split_id: str,
) -> None:
    candidate = files.read_json("provenance/model_candidate.json")
    for binding in contract.anomaly_bindings:
        group_root = PurePosixPath(binding.asset.relative_path).parent
        metadata_relative = (group_root / "metadata.json").as_posix()
        if metadata_relative not in files.checksums:
            raise ReleaseLoadError(f"anomaly group metadata is missing: {metadata_relative}")
        metadata = files.read_json(metadata_relative)
        expected_keys = {
            "schema_version", "family", "hand", "view", "model_sha256",
            "dataset_release_id", "dataset_manifest_sha256", "train_split_id",
            "recipe_sha256", "roi_version", "roi_sha256", "framework_version",
            "training_parameters", "execution_receipt",
        }
        if set(metadata) != expected_keys or metadata.get("schema_version") != 1:
            raise ReleaseLoadError(f"anomaly metadata envelope is malformed: {metadata_relative}")
        expected = (
            binding.hand.value,
            binding.view_id,
            binding.family.value,
            binding.asset.sha256,
            contract.roi.roi_config_id,
            contract.roi.roi_sha256,
        )
        actual = (
            metadata.get("hand"),
            metadata.get("view"),
            metadata.get("family"),
            metadata.get("model_sha256"),
            metadata.get("roi_version"),
            metadata.get("roi_sha256"),
        )
        if actual != expected:
            raise ReleaseLoadError(f"anomaly metadata identity mismatch: {metadata_relative}")
        if (
            metadata.get("dataset_release_id") != candidate.get("dataset_release_id")
            or metadata.get("dataset_manifest_sha256") != candidate.get("dataset_manifest_sha256")
            or metadata.get("train_split_id") != expected_train_split_id
            or metadata.get("recipe_sha256") != contract.recipe_sha256
            or not isinstance(metadata.get("train_split_id"), str)
            or not metadata["train_split_id"].strip()
            or not isinstance(metadata.get("framework_version"), str)
            or not metadata["framework_version"].strip()
            or not isinstance(metadata.get("training_parameters"), dict)
            or not isinstance(metadata.get("execution_receipt"), dict)
        ):
            raise ReleaseLoadError(f"anomaly metadata provenance mismatch: {metadata_relative}")
        validate_anomaly_artifact_metadata(
            AnomalyModelArtifact(
                family=binding.family,
                slot=ModelSlot(binding.hand.value, binding.view_id),
                checkpoint=AssetFile(
                    role="checkpoint",
                    path=files.path(binding.asset.relative_path),
                    sha256=binding.asset.sha256,
                    media_type="application/x-pytorch",
                    logical_path=binding.asset.relative_path,
                ),
                model_digest=binding.asset.sha256,
                dataset_release_id=metadata["dataset_release_id"],
                dataset_manifest_digest=metadata["dataset_manifest_sha256"],
                train_split_id=metadata["train_split_id"],
                recipe_digest=metadata["recipe_sha256"],
                roi_version=metadata["roi_version"],
                roi_digest=metadata["roi_sha256"],
                framework_version=metadata["framework_version"],
                training_parameters=metadata["training_parameters"],
                execution_receipt=metadata["execution_receipt"],
            )
        )
