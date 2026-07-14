"""Manual, immutable assembly of one validated ZS32 deployment release."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from zs32_inspection.calibration.acceptance import (
    load_heldout_acceptance_decision_bytes,
    load_heldout_acceptance_policy_bytes,
)
from zs32_inspection.capture.gate_policy import (
    CaptureGatePolicy,
    load_capture_gate_policy_bytes,
)
from zs32_inspection.config.schemas import (
    parse_deployment_contract,
    parse_recipe,
    parse_release_manifest,
    parse_roi_config,
    parse_topology,
)
from zs32_inspection.domain.contracts import DeploymentContract, ReleaseManifest
from zs32_inspection.models.base import SHA256_PATTERN, ModelSlot, sha256_file
from zs32_inspection.models.deployment import DeploymentAssetDescription
from zs32_inspection.models.registry import CandidateStatus

from .environment_receipt import (
    RuntimeEnvironmentReceipt,
    load_runtime_environment_receipt,
)
from .promotion_receipt import PromotionReceipt, load_promotion_receipt
from .publisher import (
    AtomicDirectoryPublisher,
    PublicationError,
    VerifiedAtomicPublication,
    canonical_json_bytes,
    verify_atomic_publication,
)


_VALIDATION_FILES = frozenset(
    {
        "candidate.json",
        "deployment_assets.json",
        "validation.json",
        "heldout_acceptance_policy.json",
        "heldout_acceptance.json",
    }
)


@dataclass(frozen=True, slots=True)
class ReleaseAssemblySpec:
    """Complete explicit input for one manual production promotion."""

    release_id: str
    output_root: Path
    created_at: str
    promotion_receipt_path: Path
    promotion_receipt_sha256: str
    validation_publication_path: Path
    validation_publication_id: str
    validation_publication_root_sha256: str
    validation_record_sha256: str
    heldout_acceptance_policy_sha256: str
    heldout_acceptance_decision_sha256: str
    contract: DeploymentContract
    assets: DeploymentAssetDescription
    fusion_policy_path: Path
    capture_gate_policy_path: Path
    capture_gate_asset_root: Path
    dataset_release_manifest_path: Path
    dataset_release_manifest_sha256: str
    code_version: Mapping[str, Any]
    runtime_environment_receipt_path: Path
    runtime_environment_receipt_sha256: str

    def __post_init__(self) -> None:
        if not self.release_id.strip() or any(token in self.release_id for token in ("/", "\\", "\x00", "\n", "\r")):
            raise ValueError(f"unsafe release_id: {self.release_id!r}")
        if not self.created_at.strip():
            raise ValueError("release created_at must be explicit")
        if (
            not self.validation_publication_id.strip()
            or any(
                token in self.validation_publication_id
                for token in ("/", "\\", "\x00", "\n", "\r")
            )
        ):
            raise ValueError("release validation_publication_id must be a safe explicit identity")
        for field in (
            "validation_publication_root_sha256",
            "validation_record_sha256",
            "heldout_acceptance_policy_sha256",
            "heldout_acceptance_decision_sha256",
            "promotion_receipt_sha256",
            "runtime_environment_receipt_sha256",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
                raise ValueError(f"release {field} must be a lowercase SHA256")
        if not isinstance(self.contract, DeploymentContract):
            raise TypeError("release assembly requires a DeploymentContract")
        if not isinstance(self.assets, DeploymentAssetDescription):
            raise TypeError("release assembly requires a DeploymentAssetDescription")
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(
            self,
            "validation_publication_path",
            Path(self.validation_publication_path).expanduser(),
        )
        object.__setattr__(self, "fusion_policy_path", Path(self.fusion_policy_path).expanduser())
        object.__setattr__(
            self,
            "promotion_receipt_path",
            Path(self.promotion_receipt_path).expanduser(),
        )
        object.__setattr__(
            self,
            "capture_gate_policy_path",
            Path(self.capture_gate_policy_path).expanduser(),
        )
        object.__setattr__(
            self,
            "capture_gate_asset_root",
            Path(self.capture_gate_asset_root).expanduser(),
        )
        object.__setattr__(
            self,
            "dataset_release_manifest_path",
            Path(self.dataset_release_manifest_path).expanduser(),
        )
        object.__setattr__(
            self,
            "runtime_environment_receipt_path",
            Path(self.runtime_environment_receipt_path).expanduser(),
        )
        if not isinstance(self.code_version, Mapping) or not self.code_version:
            raise ValueError("code_version provenance must be a non-empty mapping")


def _yaml_bytes(payload: Mapping[str, object]) -> bytes:
    """Serialize a deterministic human-readable release snapshot."""
    return yaml.safe_dump(
        dict(payload),
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=True,
    ).encode("utf-8")


def _recipe_payload(contract: DeploymentContract) -> dict[str, object]:
    template_assets: dict[str, dict[str, object]] = {}
    for binding in contract.template_bindings:
        template_assets.setdefault(binding.hand.value, {})[binding.view_id] = binding.asset.as_dict()
    anomaly_models: dict[str, dict[str, object]] = {}
    for binding in contract.anomaly_bindings:
        anomaly_models.setdefault(binding.hand.value, {})[binding.view_id] = {
            "family": binding.family.value,
            "artifact": binding.asset.as_dict(),
        }
    return {
        "schema_version": 1,
        "recipe_id": contract.recipe_id,
        "product": contract.product,
        "topology_id": contract.topology.topology_id,
        "roi_version": contract.roi.roi_config_id,
        "allowed_hands": [hand.value for hand in contract.allowed_hands],
        "anomaly_family": contract.anomaly_family.value,
        "capture_gate_policy": contract.capture_gate_policy.as_dict(),
        "template_assets": template_assets,
        "anomaly_models": anomaly_models,
        "yolo_model": contract.yolo_model.as_dict(),
        "template_thresholds": [
            {
                "hand": item.hand.value,
                "view": item.view_id,
                "threshold": item.threshold,
                "template_sha256": item.template_sha256,
                "calibration_sha256": item.calibration_sha256,
            }
            for item in contract.template_thresholds
        ],
        "model_thresholds": [
            {
                "hand": item.hand.value,
                "view": item.view_id,
                "branch": item.branch.value,
                "low": item.low,
                "high": item.high,
                "model_sha256": item.model_sha256,
                "roi_version": item.roi_config_id,
                "calibration_sha256": item.calibration_sha256,
            }
            for item in contract.model_thresholds
        ],
        "fusion_policy_sha256": contract.fusion_policy_sha256,
    }


def _topology_payload(contract: DeploymentContract) -> dict[str, object]:
    payload = contract.topology.as_dict(include_sha256=False)
    return payload


def _roi_payload(contract: DeploymentContract) -> dict[str, object]:
    return contract.roi.as_dict(include_sha256=False)


def _candidate_provenance(
    spec: ReleaseAssemblySpec,
    receipt: PromotionReceipt,
) -> dict[str, object]:
    candidate = spec.assets.candidate
    return {
        "schema_version": 3,
        "product": candidate.product,
        "candidate_id": candidate.candidate_id,
        "candidate_digest": candidate.digest,
        "candidate_status": candidate.status.value,
        "promotion_id": receipt.promotion_id,
        "promotion_approver": receipt.approver,
        "promotion_approved_at": receipt.approved_at,
        "promotion_receipt_sha256": spec.promotion_receipt_sha256,
        "validation_publication_id": spec.validation_publication_id,
        "validation_publication_root_sha256": spec.validation_publication_root_sha256,
        "validation_record_sha256": spec.validation_record_sha256,
        "heldout_acceptance_policy_sha256": (
            spec.heldout_acceptance_policy_sha256
        ),
        "heldout_acceptance_decision_sha256": (
            spec.heldout_acceptance_decision_sha256
        ),
        "dataset_release_id": candidate.dataset_release_id,
        "dataset_manifest_sha256": candidate.dataset_manifest_digest,
        "recipe_sha256": candidate.recipe_digest,
        "topology_sha256": candidate.topology_digest,
        "roi_sha256": candidate.roi_digest,
        "roi_version": candidate.roi_version,
        "anomaly_family": candidate.anomaly_family.value,
        "yolo_model_sha256": candidate.yolo.model_digest,
        "calibration_artifact_sha256": spec.assets.calibration_artifact_digest,
    }


def _load_and_validate_promotion_receipt(
    spec: ReleaseAssemblySpec,
) -> PromotionReceipt:
    """Rehash one canonical approval and bind it to every promotion input."""
    receipt = load_promotion_receipt(spec.promotion_receipt_path)
    if receipt.sha256 != spec.promotion_receipt_sha256:
        raise ValueError("promotion receipt bytes differ from release assembly input")
    candidate = spec.assets.candidate
    if (
        receipt.release_id != spec.release_id
        or receipt.candidate_id != candidate.candidate_id
        or receipt.candidate_digest != candidate.digest
        or receipt.validation_publication_id != spec.validation_publication_id
        or receipt.validation_publication_root_sha256
        != spec.validation_publication_root_sha256
        or receipt.validation_record_sha256 != spec.validation_record_sha256
        or receipt.contract_sha256 != spec.contract.contract_sha256
        or receipt.calibration_artifact_sha256
        != spec.assets.calibration_artifact_digest
        or receipt.dataset_manifest_sha256 != spec.dataset_release_manifest_sha256
        or receipt.heldout_acceptance_policy_sha256
        != spec.heldout_acceptance_policy_sha256
        or receipt.heldout_acceptance_decision_sha256
        != spec.heldout_acceptance_decision_sha256
    ):
        raise ValueError(
            "promotion receipt differs from release/candidate/validation/contract/"
            "calibration/dataset identity"
        )
    try:
        approved_at = datetime.fromisoformat(receipt.approved_at.replace("Z", "+00:00"))
        created_at = datetime.fromisoformat(spec.created_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("release created_at must be timezone-aware ISO-8601") from error
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("release created_at must include an explicit timezone")
    if approved_at > created_at:
        raise ValueError("promotion receipt cannot approve a release after release created_at")
    return receipt


def _verify_validation_publication(spec: ReleaseAssemblySpec) -> VerifiedAtomicPublication:
    """Verify and bind the exact candidate/deployment/validation decision bytes."""
    publication = verify_atomic_publication(
        spec.validation_publication_path,
        required_paths=_VALIDATION_FILES,
        allowed_paths=_VALIDATION_FILES,
        expected_publication_id=spec.validation_publication_id,
    )
    if publication.root_sha256 != spec.validation_publication_root_sha256:
        raise ValueError("validation publication root differs from release assembly input")
    if publication.checksums["validation.json"] != spec.validation_record_sha256:
        raise ValueError("validation record digest differs from release assembly input")
    policy_bytes = publication.read_bytes("heldout_acceptance_policy.json")
    decision_bytes = publication.read_bytes("heldout_acceptance.json")
    policy = load_heldout_acceptance_policy_bytes(policy_bytes)
    decision = load_heldout_acceptance_decision_bytes(decision_bytes)
    if (
        publication.checksums["heldout_acceptance_policy.json"]
        != spec.heldout_acceptance_policy_sha256
        or policy.sha256 != spec.heldout_acceptance_policy_sha256
        or publication.checksums["heldout_acceptance.json"]
        != spec.heldout_acceptance_decision_sha256
        or decision.sha256 != spec.heldout_acceptance_decision_sha256
        or decision.policy_sha256 != policy.sha256
    ):
        raise ValueError(
            "held-out acceptance policy/decision differs from release assembly input"
        )
    if publication.read_bytes("candidate.json") != canonical_json_bytes(
        spec.assets.candidate.to_dict()
    ):
        raise ValueError("validation candidate bytes differ from deployment candidate")
    if publication.read_bytes("deployment_assets.json") != canonical_json_bytes(
        spec.assets.to_dict()
    ):
        raise ValueError("validation deployment bytes differ from release assets")
    return publication


def _verify_inputs(spec: ReleaseAssemblySpec) -> VerifiedAtomicPublication:
    contract = spec.contract
    assets = spec.assets
    candidate = assets.candidate
    if candidate.status is not CandidateStatus.VALIDATED:
        raise ValueError("only a manually validated model candidate may enter release assembly")
    assets.verify()
    required_slots = tuple(
        ModelSlot(hand.value, view)
        for hand in contract.allowed_hands
        for view in contract.topology.required_views
    )
    if tuple(assets.required_slots) != required_slots:
        raise ValueError("deployment asset slots differ from compiled contract order/identity")
    if (
        candidate.recipe_digest != contract.recipe_sha256
        or candidate.topology_digest != contract.topology.topology_sha256
        or candidate.roi_digest != contract.roi.roi_sha256
        or candidate.anomaly_family.value != contract.anomaly_family.value
    ):
        raise ValueError("candidate provenance differs from deployment contract")
    anomaly_by_slot = assets.candidate.anomaly_artifacts
    for binding in contract.anomaly_bindings:
        slot = ModelSlot(binding.hand.value, binding.view_id)
        if anomaly_by_slot[slot].model_digest != binding.asset.sha256:
            raise ValueError(f"anomaly contract/candidate digest mismatch for {slot.key}")
    template_by_slot = assets.template_assets
    for binding in contract.template_bindings:
        slot = ModelSlot(binding.hand.value, binding.view_id)
        if template_by_slot[slot].model_digest != binding.asset.sha256:
            raise ValueError(f"template contract/candidate digest mismatch for {slot.key}")
    if candidate.yolo.model_digest != contract.yolo_model.sha256:
        raise ValueError("YOLO contract/candidate digest mismatch")
    calibration_digests = {
        item.calibration_sha256
        for item in (*contract.template_thresholds, *contract.model_thresholds)
    }
    if calibration_digests != {assets.calibration_artifact_digest}:
        raise ValueError("contract thresholds do not bind the selected calibration artifact")
    if sha256_file(spec.fusion_policy_path) != contract.fusion_policy_sha256:
        raise ValueError("fusion policy bytes differ from deployment contract")
    _load_capture_gate_policy(spec)
    if sha256_file(spec.dataset_release_manifest_path) != spec.dataset_release_manifest_sha256:
        raise ValueError("dataset release manifest hash differs from assembly input")
    if spec.dataset_release_manifest_sha256 != candidate.dataset_manifest_digest:
        raise ValueError("dataset release manifest differs from model candidate provenance")
    _dataset_governance_artifacts(spec)
    _load_runtime_environment_receipt(spec)
    _load_and_validate_promotion_receipt(spec)
    return _verify_validation_publication(spec)


def _load_runtime_environment_receipt(
    spec: ReleaseAssemblySpec,
) -> RuntimeEnvironmentReceipt:
    """Require canonical receipt bytes and both internal/external digests."""
    receipt = load_runtime_environment_receipt(spec.runtime_environment_receipt_path)
    if sha256_file(spec.runtime_environment_receipt_path) != spec.runtime_environment_receipt_sha256:
        raise ValueError("runtime environment receipt bytes differ from assembly input")
    return receipt


def _dataset_governance_artifacts(
    spec: ReleaseAssemblySpec,
) -> dict[str, tuple[Path, str]]:
    """Locate and rehash every governance artifact bound by the dataset manifest."""
    try:
        payload = json.loads(spec.dataset_release_manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"dataset release manifest cannot be parsed: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("dataset release manifest must be an object")
    if payload.get("capture_gate_policy_sha256") != spec.contract.capture_gate_policy.sha256:
        raise ValueError("dataset capture gate policy differs from deployment contract")
    bindings = {
        "dataset_capture_provenance.jsonl": (
            "capture_provenance.jsonl",
            "capture_provenance_sha256",
        ),
        "dataset_canonical_semantics.json": (
            "canonical_semantics.json",
            "canonical_semantics_sha256",
        ),
        "dataset_calibration_targets.json": (
            "calibration_targets.json",
            "calibration_targets_sha256",
        ),
        "dataset_governance.json": (
            "dataset_provenance.json",
            "dataset_provenance_sha256",
        ),
        "dataset_split_assignments.json": (
            "split_assignments.json",
            "split_assignments_sha256",
        ),
    }
    artifacts: dict[str, tuple[Path, str]] = {}
    for destination, (source_name, digest_field) in bindings.items():
        digest = payload.get(digest_field)
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"dataset release does not bind {source_name}")
        path = spec.dataset_release_manifest_path.with_name(source_name)
        if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"dataset governance artifact differs from manifest: {source_name}")
        artifacts[destination] = (path, digest)
    return artifacts


def _load_capture_gate_policy(spec: ReleaseAssemblySpec) -> CaptureGatePolicy:
    """Rehash the canonical gate manifest and every referenced asset."""
    if (
        spec.capture_gate_policy_path.is_symlink()
        or not spec.capture_gate_policy_path.is_file()
    ):
        raise ValueError("capture gate policy must be a regular non-symlink file")
    if sha256_file(spec.capture_gate_policy_path) != spec.contract.capture_gate_policy.sha256:
        raise ValueError("capture gate policy bytes differ from deployment contract")
    policy = load_capture_gate_policy_bytes(spec.capture_gate_policy_path.read_bytes())
    policy.validate_topology(
        spec.contract.topology,
        allowed_hands=spec.contract.allowed_hands,
    )
    policy.verify_assets(spec.capture_gate_asset_root)
    return policy


def assemble_release(spec: ReleaseAssemblySpec) -> Path:
    """Copy verified candidates into one immutable, manually promoted release."""
    validation_publication = _verify_inputs(spec)
    promotion_receipt = _load_and_validate_promotion_receipt(spec)
    gate_policy = _load_capture_gate_policy(spec)
    dataset_governance_artifacts = _dataset_governance_artifacts(spec)
    _load_runtime_environment_receipt(spec)
    contract = spec.contract
    assets = spec.assets
    manifest = ReleaseManifest(
        schema_version=1,
        release_id=spec.release_id,
        product=contract.product,
        contract_sha256=contract.contract_sha256,
        created_at=spec.created_at,
    )
    recipe_payload = _recipe_payload(contract)
    topology_payload = _topology_payload(contract)
    roi_payload = _roi_payload(contract)
    if parse_recipe(recipe_payload).recipe_sha256 != contract.recipe_sha256:
        raise ValueError("reconstructed release recipe digest differs from deployment contract")
    if parse_topology(topology_payload).topology_sha256 != contract.topology.topology_sha256:
        raise ValueError("reconstructed release topology digest differs from deployment contract")
    if parse_roi_config(roi_payload).roi_sha256 != contract.roi.roi_sha256:
        raise ValueError("reconstructed release ROI digest differs from deployment contract")

    required = {
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
    validation_destination = (
        Path("provenance") / "validations" / spec.validation_publication_id
    )
    validation_sources = {
        **{
            name: validation_publication.checksums[name]
            for name in _VALIDATION_FILES
        },
        "checksums.sha256": validation_publication.root_sha256,
        "publication_root.json": sha256_file(
            validation_publication.root / "publication_root.json"
        ),
    }
    with AtomicDirectoryPublisher(spec.output_root, spec.release_id) as publisher:
        publisher.write_json("manifest.json", {
            "schema_version": manifest.schema_version,
            "release_id": manifest.release_id,
            "product": manifest.product,
            "contract_sha256": manifest.contract_sha256,
            "created_at": manifest.created_at,
        })
        publisher.write_json("deployment_contract.json", contract.as_dict())
        publisher.write_bytes("recipe.yaml", _yaml_bytes(recipe_payload))
        publisher.write_bytes("topology.yaml", _yaml_bytes(topology_payload))
        publisher.write_bytes("roi.yaml", _yaml_bytes(roi_payload))
        publisher.copy_file(
            spec.runtime_environment_receipt_path,
            "provenance/runtime_environment.json",
            expected_sha256=spec.runtime_environment_receipt_sha256,
        )
        publisher.copy_file(
            spec.promotion_receipt_path,
            "provenance/promotion_receipt.json",
            expected_sha256=spec.promotion_receipt_sha256,
        )
        publisher.copy_file(
            spec.fusion_policy_path,
            "fusion/policy.yaml",
            expected_sha256=contract.fusion_policy_sha256,
        )
        publisher.copy_file(
            spec.capture_gate_policy_path,
            contract.capture_gate_policy.relative_path,
            expected_sha256=contract.capture_gate_policy.sha256,
        )
        publisher.copy_file(
            spec.capture_gate_asset_root / gate_policy.acquisition_config.relative_path,
            gate_policy.acquisition_config.relative_path,
            expected_sha256=gate_policy.acquisition_config.sha256,
        )
        required.add(gate_policy.acquisition_config.relative_path)
        for hand_policy in gate_policy.hands.values():
            for artifact in (
                hand_policy.quality_profile,
                hand_policy.registration_profile,
                *hand_policy.registration_references.values(),
            ):
                publisher.copy_file(
                    spec.capture_gate_asset_root / artifact.relative_path,
                    artifact.relative_path,
                    expected_sha256=artifact.sha256,
                )
                required.add(artifact.relative_path)

        for binding in contract.anomaly_bindings:
            slot = ModelSlot(binding.hand.value, binding.view_id)
            artifact = assets.candidate.anomaly_artifacts[slot]
            publisher.copy_file(
                artifact.checkpoint.path,
                binding.asset.relative_path,
                expected_sha256=binding.asset.sha256,
            )
            required.add(binding.asset.relative_path)
            metadata_path = str(Path(binding.asset.relative_path).parent / "metadata.json")
            publisher.write_json(
                metadata_path,
                {
                    "schema_version": 1,
                    "family": artifact.family.value,
                    "hand": slot.hand,
                    "view": slot.view,
                    "model_sha256": artifact.model_digest,
                    "dataset_release_id": artifact.dataset_release_id,
                    "dataset_manifest_sha256": artifact.dataset_manifest_digest,
                    "train_split_id": artifact.train_split_id,
                    "recipe_sha256": artifact.recipe_digest,
                    "roi_version": artifact.roi_version,
                    "roi_sha256": artifact.roi_digest,
                    "framework_version": artifact.framework_version,
                    "training_parameters": dict(artifact.training_parameters),
                    "execution_receipt": dict(artifact.execution_receipt),
                },
            )
            required.add(metadata_path)

        for binding in contract.template_bindings:
            slot = ModelSlot(binding.hand.value, binding.view_id)
            template = assets.template_assets[slot]
            publisher.copy_file(
                template.model.path,
                binding.asset.relative_path,
                expected_sha256=binding.asset.sha256,
            )
            required.add(binding.asset.relative_path)
            group_root = Path(binding.asset.relative_path).parent
            references: list[dict[str, str]] = []
            for index, reference in enumerate(template.templates):
                relative = (group_root / "templates" / f"{index:03d}_{reference.path.name}").as_posix()
                publisher.copy_file(reference.path, relative, expected_sha256=reference.sha256)
                required.add(relative)
                references.append({"relative_path": relative, "sha256": reference.sha256})
            metadata_path = (group_root / "metadata.json").as_posix()
            publisher.write_json(
                metadata_path,
                {
                    "schema_version": 1,
                    "hand": slot.hand,
                    "view": slot.view,
                    "model_sha256": template.model_digest,
                    "template_version": template.template_version,
                    "roi_version": template.roi_version,
                    "roi_sha256": template.roi_digest,
                    "dataset_release_id": template.dataset_release_id,
                    "dataset_manifest_sha256": template.dataset_manifest_digest,
                    "train_split_id": template.train_split_id,
                    "recipe_sha256": template.recipe_digest,
                    "framework_version": template.framework_version,
                    "training_parameters": dict(template.training_parameters),
                    "execution_receipt": dict(template.execution_receipt),
                    "reference_templates": references,
                },
            )
            required.add(metadata_path)

        yolo = assets.candidate.yolo
        publisher.copy_file(yolo.weights.path, "models/yolo/best.pt", expected_sha256=yolo.weights.sha256)
        publisher.copy_file(yolo.args_yaml.path, "models/yolo/train_args.yaml", expected_sha256=yolo.args_yaml.sha256)
        publisher.copy_file(yolo.data_yaml.path, "models/yolo/data.yaml", expected_sha256=yolo.data_yaml.sha256)
        publisher.copy_file(
            yolo.class_names_yaml.path,
            "models/yolo/class_names.yaml",
            expected_sha256=yolo.class_names_yaml.sha256,
        )
        publisher.copy_file(
            yolo.training_receipt.path,
            "models/yolo/training_receipt.json",
            expected_sha256=yolo.training_receipt.sha256,
        )
        publisher.write_bytes(
            "models/yolo/weights.sha256",
            f"{yolo.model_digest}  best.pt\n".encode("ascii"),
        )
        publisher.write_json(
            "models/yolo/metadata.json",
            {
                "schema_version": 3,
                "model_sha256": yolo.model_digest,
                "bundle_sha256": yolo.bundle_digest,
                "runtime": yolo.runtime.to_dict(),
                "provenance": yolo.provenance.to_dict(),
                "asset_sha256": {
                    "best.pt": yolo.weights.sha256,
                    "train_args.yaml": yolo.args_yaml.sha256,
                    "data.yaml": yolo.data_yaml.sha256,
                    "class_names.yaml": yolo.class_names_yaml.sha256,
                    "training_receipt.json": yolo.training_receipt.sha256,
                },
            },
        )

        for source, relative in (
            (assets.calibration_artifact, "calibration/calibration_artifact.json"),
            (assets.template_thresholds, "calibration/template_thresholds.json"),
            (assets.model_thresholds, "calibration/model_thresholds.json"),
            (assets.calibration_metrics, "calibration/metrics.json"),
            (
                assets.calibration_input_provenance,
                "calibration/input_provenance.json",
            ),
        ):
            publisher.copy_file(source.path, relative, expected_sha256=source.sha256)
        publisher.copy_file(
            spec.dataset_release_manifest_path,
            "provenance/dataset_release.json",
            expected_sha256=spec.dataset_release_manifest_sha256,
        )
        for destination_name, (source_path, digest) in sorted(
            dataset_governance_artifacts.items()
        ):
            publisher.copy_file(
                source_path,
                f"provenance/{destination_name}",
                expected_sha256=digest,
            )
        publisher.write_json(
            "provenance/model_candidate.json",
            _candidate_provenance(spec, promotion_receipt),
        )
        publisher.write_json("provenance/model_candidate_full.json", assets.candidate.to_dict())
        publisher.write_json("provenance/code_version.json", dict(spec.code_version))
        for name, digest in sorted(validation_sources.items()):
            destination = (validation_destination / name).as_posix()
            publisher.copy_file(
                validation_publication.root / name,
                destination,
                expected_sha256=digest,
            )
            required.add(destination)
        reverified_validation = _verify_validation_publication(spec)
        reverified_promotion = _load_and_validate_promotion_receipt(spec)
        if (
            reverified_validation.root_sha256
            != validation_publication.root_sha256
            or dict(reverified_validation.checksums)
            != dict(validation_publication.checksums)
        ):
            raise PublicationError("validation publication changed during release assembly")
        if reverified_promotion.sha256 != promotion_receipt.sha256:
            raise PublicationError("promotion receipt changed during release assembly")
        return publisher.finalize(
            validator=lambda staging: _validate_release_staging(staging, spec),
            required_paths=frozenset(required),
        )


def _validate_release_staging(root: Path, spec: ReleaseAssemblySpec) -> None:
    """Reparse the complete semantic envelope immediately before publication."""
    manifest = parse_release_manifest(json.loads((root / "manifest.json").read_text(encoding="utf-8")))
    contract = parse_deployment_contract(
        json.loads((root / "deployment_contract.json").read_text(encoding="utf-8"))
    )
    if manifest.release_id != spec.release_id or manifest.contract_sha256 != contract.contract_sha256:
        raise PublicationError("staged release manifest/contract identity mismatch")
    if contract.contract_sha256 != spec.contract.contract_sha256:
        raise PublicationError("staged deployment contract differs from assembly input")
    if (
        parse_recipe(
            yaml.safe_load((root / "recipe.yaml").read_text(encoding="utf-8"))
        ).recipe_sha256
        != contract.recipe_sha256
    ):
        raise PublicationError("staged recipe digest mismatch")
    if (
        parse_topology(
            yaml.safe_load((root / "topology.yaml").read_text(encoding="utf-8"))
        ).topology_sha256
        != contract.topology.topology_sha256
    ):
        raise PublicationError("staged topology digest mismatch")
    if (
        parse_roi_config(
            yaml.safe_load((root / "roi.yaml").read_text(encoding="utf-8"))
        ).roi_sha256
        != contract.roi.roi_sha256
    ):
        raise PublicationError("staged ROI digest mismatch")
    if sha256_file(root / "fusion/policy.yaml") != contract.fusion_policy_sha256:
        raise PublicationError("staged fusion policy digest mismatch")
    if (
        sha256_file(root / "provenance/runtime_environment.json")
        != spec.runtime_environment_receipt_sha256
    ):
        raise PublicationError("staged runtime environment receipt digest mismatch")
    if (
        sha256_file(root / "provenance/promotion_receipt.json")
        != spec.promotion_receipt_sha256
    ):
        raise PublicationError("staged promotion receipt digest mismatch")
    if (
        sha256_file(root / contract.capture_gate_policy.relative_path)
        != contract.capture_gate_policy.sha256
    ):
        raise PublicationError("staged capture gate policy digest mismatch")
    if sha256_file(root / "calibration/calibration_artifact.json") != spec.assets.calibration_artifact_digest:
        raise PublicationError("staged calibration artifact digest mismatch")
    from .release_loader import ReleaseLoadError, validate_staged_deployment_release

    try:
        validate_staged_deployment_release(root, expected_release_id=spec.release_id)
    except ReleaseLoadError as error:
        raise PublicationError(f"staged release failed full loader validation: {error}") from error
