"""Manually assemble one immutable, fully verified deployment release."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zs32_inspection.calibration.acceptance import (
    HeldOutAcceptanceDecision,
    HeldOutAcceptancePolicy,
    evaluate_heldout_acceptance,
    load_heldout_acceptance_decision_bytes,
    load_heldout_acceptance_policy_bytes,
)
from zs32_inspection.calibration.reports import CalibrationArtifact
from zs32_inspection.config.loaders import load_deployment_contract
from zs32_inspection.data.dataset_release import verify_dataset_release
from zs32_inspection.models.base import SHA256_PATTERN, sha256_file
from zs32_inspection.models.deployment import DeploymentAssetDescription
from zs32_inspection.models.registry import CandidateStatus, ModelCandidate
from zs32_inspection.runtime.release_assembler import ReleaseAssemblySpec, assemble_release
from zs32_inspection.runtime.release_loader import reconstruct_calibration_artifact
from zs32_inspection.runtime.promotion_receipt import load_promotion_receipt
from zs32_inspection.runtime.publisher import (
    VerifiedAtomicPublication,
    canonical_json_bytes,
    verify_atomic_publication,
)

from ._common import command_error, command_result, load_object
from ._deployment_assets import parse_candidate, parse_deployment_assets
from ._environment import require_linux_nvidia


_VALIDATION_FILES = frozenset(
    {
        "candidate.json",
        "deployment_assets.json",
        "validation.json",
        "heldout_acceptance_policy.json",
        "heldout_acceptance.json",
    }
)
_CALIBRATION_FILES = frozenset(
    {
        "calibration_artifact.json",
        "template_thresholds.json",
        "model_thresholds.json",
        "metrics.json",
        "input_provenance.json",
    }
)
_VALIDATION_RECORD_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "validation_id",
        "candidate_id",
        "candidate_digest",
        "input_candidate_descriptor_sha256",
        "input_template_assets_descriptor_sha256",
        "dataset_release_id",
        "dataset_manifest_sha256",
        "recipe_sha256",
        "contract_sha256",
        "calibration_artifact_sha256",
        "calibration_split_id",
        "test_split_id",
        "candidate_status_before",
        "candidate_status_after",
        "manual_promotion_required",
        "score_run_id",
        "score_run_root_sha256",
        "scores_sha256",
        "score_audit_sha256",
        "score_execution_receipt_sha256",
        "calibration_publication_id",
        "calibration_publication_root_sha256",
        "registration_publication_id",
        "registration_publication_root_sha256",
        "heldout_acceptance_policy_id",
        "heldout_acceptance_policy_version",
        "heldout_acceptance_policy_sha256",
        "heldout_acceptance_decision_sha256",
    }
)


@dataclass(frozen=True, slots=True)
class ValidationPublicationInput:
    """Canonical descriptors read from one exact atomic validation publication."""

    publication: VerifiedAtomicPublication
    candidate_payload: dict[str, Any]
    deployment_payload: dict[str, Any]
    validation_record: dict[str, Any]
    heldout_acceptance_policy: HeldOutAcceptancePolicy
    heldout_acceptance_decision: HeldOutAcceptanceDecision


def _canonical_object(content: bytes, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON: {error}") from error
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ValueError(f"{label} must be a string-keyed JSON object")
    if canonical_json_bytes(payload) != content:
        raise ValueError(f"{label} must use canonical JSON serialization")
    return payload


def _load_validation_publication(
    candidate_path: Path,
    deployment_path: Path,
) -> ValidationPublicationInput:
    """Require both descriptors to come from one exact, immutable publication."""
    if candidate_path.name != "candidate.json" or deployment_path.name != "deployment_assets.json":
        raise ValueError(
            "--candidate/--deployment-assets must name candidate.json and deployment_assets.json"
        )
    candidate_parent = candidate_path.parent.expanduser().resolve()
    deployment_parent = deployment_path.parent.expanduser().resolve()
    if candidate_parent != deployment_parent:
        raise ValueError(
            "candidate and deployment assets must come from the same validation publication"
        )
    publication = verify_atomic_publication(
        candidate_path.parent,
        required_paths=_VALIDATION_FILES,
        allowed_paths=_VALIDATION_FILES,
    )
    return ValidationPublicationInput(
        publication=publication,
        candidate_payload=_canonical_object(
            publication.read_bytes("candidate.json"),
            "validated candidate",
        ),
        deployment_payload=_canonical_object(
            publication.read_bytes("deployment_assets.json"),
            "deployment assets",
        ),
        validation_record=_canonical_object(
            publication.read_bytes("validation.json"),
            "candidate validation record",
        ),
        heldout_acceptance_policy=load_heldout_acceptance_policy_bytes(
            publication.read_bytes("heldout_acceptance_policy.json")
        ),
        heldout_acceptance_decision=load_heldout_acceptance_decision_bytes(
            publication.read_bytes("heldout_acceptance.json")
        ),
    )


def _reverify_publication(verified: VerifiedAtomicPublication) -> None:
    """Reject replacement or mutation of a previously verified publication."""
    current = verify_atomic_publication(
        verified.root,
        required_paths=_VALIDATION_FILES,
        allowed_paths=_VALIDATION_FILES,
        expected_publication_id=verified.publication_id,
    )
    if (
        current.root != verified.root
        or current.root_sha256 != verified.root_sha256
        or dict(current.checksums) != dict(verified.checksums)
    ):
        raise ValueError("validation publication changed during release assembly")


def _verified_calibration_evidence(
    assets: DeploymentAssetDescription,
) -> tuple[
    VerifiedAtomicPublication,
    str,
    str,
    dict[str, Any],
    CalibrationArtifact,
]:
    """Re-open calibration evidence and return its publication/split/score identities."""
    calibration_assets = {
        "calibration_artifact.json": assets.calibration_artifact,
        "template_thresholds.json": assets.template_thresholds,
        "model_thresholds.json": assets.model_thresholds,
        "metrics.json": assets.calibration_metrics,
        "input_provenance.json": assets.calibration_input_provenance,
    }
    parents = {asset.path.expanduser().resolve().parent for asset in calibration_assets.values()}
    if len(parents) != 1 or any(
        asset.path.name != expected_name
        for expected_name, asset in calibration_assets.items()
    ):
        raise ValueError(
            "deployment calibration assets must come from one complete calibration publication"
        )
    root = next(iter(parents))
    publication = verify_atomic_publication(
        root,
        required_paths=_CALIBRATION_FILES,
        allowed_paths=_CALIBRATION_FILES,
    )
    for name, asset in calibration_assets.items():
        if publication.checksums[name] != asset.sha256:
            raise ValueError(f"calibration publication digest differs from deployment assets: {name}")
    artifact_bytes = publication.read_bytes("calibration_artifact.json")
    if hashlib.sha256(artifact_bytes).hexdigest() != assets.calibration_artifact_digest:
        raise ValueError("calibration artifact digest differs from deployment assets")
    artifact_payload = _canonical_object(artifact_bytes, "calibration artifact")
    artifact = reconstruct_calibration_artifact(artifact_payload)
    if artifact.payload() != artifact_payload or not artifact.calibration_valid:
        raise ValueError("calibration artifact is malformed or non-deployable")
    if _canonical_object(
        publication.read_bytes("metrics.json"),
        "calibration metrics",
    ) != artifact.heldout_metrics.to_dict():
        raise ValueError("held-out metrics snapshot differs from calibration artifact")
    input_provenance = _canonical_object(
        publication.read_bytes("input_provenance.json"),
        "calibration input provenance",
    )
    return (
        publication,
        artifact.provenance.calibration_split_id,
        artifact.provenance.test_split_id,
        input_provenance,
        artifact,
    )


def _validate_validation_record(
    record: dict[str, Any],
    *,
    validation_id: str,
    candidate: ModelCandidate,
    assets: DeploymentAssetDescription,
    contract_sha256: str,
    recipe_sha256: str,
    dataset_release_id: str,
    dataset_manifest_sha256: str,
    calibration_split_id: str,
    test_split_id: str,
    calibration_publication: VerifiedAtomicPublication,
    calibration_input_provenance: dict[str, Any],
    heldout_acceptance_policy: HeldOutAcceptancePolicy,
    heldout_acceptance_decision: HeldOutAcceptanceDecision,
) -> None:
    """Bind the validation decision to every release-defining identity."""
    digest_fields = (
        "candidate_digest",
        "input_candidate_descriptor_sha256",
        "input_template_assets_descriptor_sha256",
        "dataset_manifest_sha256",
        "recipe_sha256",
        "contract_sha256",
        "calibration_artifact_sha256",
        "score_run_root_sha256",
        "scores_sha256",
        "score_audit_sha256",
        "score_execution_receipt_sha256",
        "calibration_publication_root_sha256",
        "registration_publication_root_sha256",
        "heldout_acceptance_policy_sha256",
        "heldout_acceptance_decision_sha256",
    )
    if set(record) != _VALIDATION_RECORD_KEYS:
        raise ValueError("candidate validation record has missing or unknown fields")
    if any(
        not isinstance(record.get(field), str)
        or SHA256_PATTERN.fullmatch(record[field]) is None
        for field in digest_fields
    ):
        raise ValueError("candidate validation record contains an invalid SHA256")
    if any(
        not isinstance(record.get(field), str) or not record[field].strip()
        for field in (
            "validation_id",
            "candidate_id",
            "dataset_release_id",
            "calibration_split_id",
            "test_split_id",
            "score_run_id",
            "calibration_publication_id",
            "registration_publication_id",
            "heldout_acceptance_policy_id",
            "heldout_acceptance_policy_version",
        )
    ):
        raise ValueError("candidate validation record contains an invalid identity")
    if (
        record["schema"] != "zs32.candidate_validation"
        or record["schema_version"] != 3
        or record["validation_id"] != validation_id
        or record["candidate_id"] != candidate.candidate_id
        or record["candidate_digest"] != candidate.digest
        or record["dataset_release_id"] != dataset_release_id
        or record["dataset_manifest_sha256"] != dataset_manifest_sha256
        or record["recipe_sha256"] != recipe_sha256
        or record["contract_sha256"] != contract_sha256
        or record["calibration_artifact_sha256"] != assets.calibration_artifact_digest
        or record["calibration_split_id"] != calibration_split_id
        or record["test_split_id"] != test_split_id
        or record["candidate_status_before"] != CandidateStatus.REGISTERED.value
        or record["candidate_status_after"] != CandidateStatus.VALIDATED.value
        or record["manual_promotion_required"] is not True
        or record["heldout_acceptance_policy_id"]
        != heldout_acceptance_policy.policy_id
        or record["heldout_acceptance_policy_version"]
        != heldout_acceptance_policy.policy_version
        or record["heldout_acceptance_policy_sha256"]
        != heldout_acceptance_policy.sha256
        or record["heldout_acceptance_decision_sha256"]
        != heldout_acceptance_decision.sha256
        or record["calibration_publication_id"] != calibration_publication.publication_id
        or record["calibration_publication_root_sha256"]
        != calibration_publication.root_sha256
        or record["score_run_id"] != calibration_input_provenance.get("score_run_id")
        or record["score_run_root_sha256"]
        != calibration_input_provenance.get("score_run_root_sha256")
        or record["scores_sha256"] != calibration_input_provenance.get("scores_sha256")
        or record["score_audit_sha256"]
        != calibration_input_provenance.get("score_audit_sha256")
        or record["score_execution_receipt_sha256"]
        != calibration_input_provenance.get("score_execution_receipt_sha256")
    ):
        raise ValueError(
            "candidate validation record differs from candidate/contract/calibration/dataset "
            "or does not require manual promotion"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manually assemble an immutable ZS32 deployment release")
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--promotion-receipt", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--deployment-assets", type=Path, required=True)
    parser.add_argument("--fusion-policy", type=Path, required=True)
    parser.add_argument("--capture-gate-policy", type=Path, required=True)
    parser.add_argument("--capture-gate-asset-root", type=Path, required=True)
    parser.add_argument("--dataset-release-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--code-version", type=Path, required=True)
    parser.add_argument("--runtime-environment-receipt", type=Path, required=True)
    parser.add_argument("--runtime-environment-sha256", required=True)
    return parser


def _run(argv: Sequence[str] | None) -> int:
    args = _parser().parse_args(argv)
    promotion_receipt_path = args.promotion_receipt.expanduser()
    promotion_receipt = load_promotion_receipt(promotion_receipt_path)
    promotion_receipt_sha256 = sha256_file(promotion_receipt_path)
    if promotion_receipt.sha256 != promotion_receipt_sha256:
        raise ValueError("promotion receipt changed while the command was reading it")
    contract = load_deployment_contract(args.contract)
    validation_input = _load_validation_publication(
        args.candidate,
        args.deployment_assets,
    )
    candidate = parse_candidate(
        validation_input.candidate_payload,
        asset_root=args.asset_root,
    )
    if candidate.status is not CandidateStatus.VALIDATED:
        raise ValueError("release assembly requires a validated candidate publication")
    assets = parse_deployment_assets(
        validation_input.deployment_payload,
        asset_root=args.asset_root,
        candidate=candidate,
    )
    if args.dataset_release_manifest.name != "dataset_release.json":
        raise ValueError("--dataset-release-manifest must name dataset_release.json")
    dataset = verify_dataset_release(args.dataset_release_manifest.parent)
    if (
        dataset.dataset_release_id != candidate.dataset_release_id
        or sha256_file(args.dataset_release_manifest) != args.dataset_manifest_sha256
        or args.dataset_manifest_sha256 != candidate.dataset_manifest_digest
    ):
        raise ValueError("verified dataset release differs from candidate/assembly provenance")
    (
        calibration_publication,
        calibration_split_id,
        test_split_id,
        calibration_input_provenance,
        calibration_artifact,
    ) = _verified_calibration_evidence(assets)
    expected_acceptance_decision = evaluate_heldout_acceptance(
        validation_input.heldout_acceptance_policy,
        calibration_artifact.heldout_metrics,
        metrics_sha256=assets.calibration_metrics.sha256,
    )
    if validation_input.heldout_acceptance_decision != expected_acceptance_decision:
        raise ValueError(
            "held-out acceptance decision differs from policy/calibration metrics"
        )
    _validate_validation_record(
        validation_input.validation_record,
        validation_id=validation_input.publication.publication_id,
        candidate=candidate,
        assets=assets,
        contract_sha256=contract.contract_sha256,
        recipe_sha256=contract.recipe_sha256,
        dataset_release_id=dataset.dataset_release_id,
        dataset_manifest_sha256=args.dataset_manifest_sha256,
        calibration_split_id=calibration_split_id,
        test_split_id=test_split_id,
        calibration_publication=calibration_publication,
        calibration_input_provenance=calibration_input_provenance,
        heldout_acceptance_policy=validation_input.heldout_acceptance_policy,
        heldout_acceptance_decision=validation_input.heldout_acceptance_decision,
    )
    spec = ReleaseAssemblySpec(
        release_id=args.release_id,
        output_root=args.output_root,
        created_at=args.created_at,
        promotion_receipt_path=promotion_receipt_path,
        promotion_receipt_sha256=promotion_receipt_sha256,
        validation_publication_path=validation_input.publication.root,
        validation_publication_id=validation_input.publication.publication_id,
        validation_publication_root_sha256=validation_input.publication.root_sha256,
        validation_record_sha256=validation_input.publication.checksums["validation.json"],
        heldout_acceptance_policy_sha256=(
            validation_input.heldout_acceptance_policy.sha256
        ),
        heldout_acceptance_decision_sha256=(
            validation_input.heldout_acceptance_decision.sha256
        ),
        contract=contract,
        assets=assets,
        fusion_policy_path=args.fusion_policy,
        capture_gate_policy_path=args.capture_gate_policy,
        capture_gate_asset_root=args.capture_gate_asset_root,
        dataset_release_manifest_path=args.dataset_release_manifest,
        dataset_release_manifest_sha256=args.dataset_manifest_sha256,
        code_version=load_object(args.code_version, "code version provenance"),
        runtime_environment_receipt_path=args.runtime_environment_receipt,
        runtime_environment_receipt_sha256=args.runtime_environment_sha256,
    )
    _reverify_publication(validation_input.publication)
    published = assemble_release(spec)
    _reverify_publication(validation_input.publication)
    command_result(
        "zs32-assemble-release",
        {
            "contract_sha256": contract.contract_sha256,
            "heldout_acceptance_policy_sha256": (
                validation_input.heldout_acceptance_policy.sha256
            ),
            "heldout_acceptance_decision_sha256": (
                validation_input.heldout_acceptance_decision.sha256
            ),
            "promotion_id": promotion_receipt.promotion_id,
            "promotion_receipt_sha256": promotion_receipt_sha256,
            "release_id": args.release_id,
            "runtime_environment_sha256": args.runtime_environment_sha256,
            "published_path": str(published),
            "validation_id": validation_input.publication.publication_id,
            "validation_record_sha256": validation_input.publication.checksums[
                "validation.json"
            ],
            "validation_root_sha256": validation_input.publication.root_sha256,
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        require_linux_nvidia()
        return _run(argv)
    except Exception as error:
        return command_error("zs32-assemble-release", error)


if __name__ == "__main__":
    raise SystemExit(main())
