"""Validate calibration evidence and freeze a release-ready candidate snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from zs32_inspection.calibration.acceptance import (
    evaluate_heldout_acceptance,
    load_heldout_acceptance_policy_bytes,
)
from zs32_inspection.config.compiler import compile_deployment_contract
from zs32_inspection.config.loaders import load_recipe, load_roi_config, load_topology
from zs32_inspection.data.dataset_release import verify_dataset_release
from zs32_inspection.data.calibration_targets import load_calibration_targets
from zs32_inspection.models.base import SHA256_PATTERN, AssetFile, ModelSlot, sha256_file
from zs32_inspection.models.deployment import DeploymentAssetDescription
from zs32_inspection.models.registry import CandidateStatus, transition_candidate
from zs32_inspection.runtime.publisher import (
    AtomicDirectoryPublisher,
    PublicationError,
    VerifiedAtomicPublication,
    canonical_json_bytes,
    verify_atomic_publication,
)
from zs32_inspection.runtime.release_loader import reconstruct_calibration_artifact

from ._common import command_error, command_result, load_object
from ._deployment_assets import parse_candidate, parse_template_assets
from ._environment import require_linux_nvidia
from .calibrate import (
    _canonical_jsonl,
    _load_verified_score_run,
    _validate_score_run_provenance,
    _validate_score_target_replay,
)
from .score_calibration import _load_canonical_rows, _selected_rows

_CALIBRATION_FILES = frozenset(
    {
        "calibration_artifact.json",
        "template_thresholds.json",
        "model_thresholds.json",
        "metrics.json",
        "input_provenance.json",
    }
)
_REGISTRATION_FILES = frozenset(
    {
        "candidate.json", "template_assets.json", "bound_recipe.json",
        "calibration_spec.json", "registration.json",
    }
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate one registered ZS32 candidate against immutable calibration evidence"
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--template-assets", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--roi", type=Path, required=True)
    parser.add_argument("--fusion-policy", type=Path, required=True)
    parser.add_argument("--dataset-release", type=Path, required=True)
    parser.add_argument("--score-run", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--heldout-acceptance-policy", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validation-id", required=True)
    return parser


def _verify_calibration_publication(root: Path) -> Mapping[str, bytes]:
    """Verify the complete AtomicDirectoryPublisher envelope before promotion."""
    _publication, contents = _load_calibration_publication(root)
    return contents


def _load_calibration_publication(
    root: Path,
) -> tuple[VerifiedAtomicPublication, Mapping[str, bytes]]:
    """Return one verified calibration publication together with its indexed bytes."""
    publication = verify_atomic_publication(
        root,
        required_paths=_CALIBRATION_FILES,
        allowed_paths=_CALIBRATION_FILES,
    )
    contents = {
        name: publication.read_bytes(name)
        for name in sorted(_CALIBRATION_FILES)
    }
    return publication, contents


def _canonical_object(content: bytes, label: str) -> dict[str, object]:
    try:
        payload = json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON: {error}") from error
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ValueError(f"{label} must be a string-keyed JSON object")
    if canonical_json_bytes(payload) != content:
        raise ValueError(f"{label} must use canonical JSON serialization")
    return payload


def _validate_score_run_input_binding(
    publication: VerifiedAtomicPublication,
    run: Mapping[str, object],
    input_provenance: Mapping[str, object],
) -> None:
    """Require calibration provenance to name the exact score publication and inputs."""
    actual = (
        publication.publication_id,
        publication.root_sha256,
        run.get("scores_sha256"),
        run.get("score_audit_sha256"),
        object_value(
            run.get("execution_receipt"),
            "score_run.execution_receipt",
        ).get("receipt_sha256"),
        run.get("candidate_id"),
        run.get("candidate_digest"),
        run.get("candidate_descriptor_sha256"),
        run.get("recipe_digest"),
        run.get("dataset_release_id"),
        run.get("dataset_manifest_sha256"),
        run.get("calibration_targets_sha256"),
        run.get("calibration_target_count"),
        run.get("calibration_split_id"),
        run.get("test_split_id"),
    )
    expected = (
        input_provenance.get("score_run_id"),
        input_provenance.get("score_run_root_sha256"),
        input_provenance.get("scores_sha256"),
        input_provenance.get("score_audit_sha256"),
        input_provenance.get("score_execution_receipt_sha256"),
        input_provenance.get("candidate_id"),
        input_provenance.get("candidate_digest"),
        input_provenance.get("candidate_descriptor_sha256"),
        input_provenance.get("recipe_digest"),
        input_provenance.get("dataset_release_id"),
        input_provenance.get("dataset_manifest_sha256"),
        input_provenance.get("calibration_targets_sha256"),
        input_provenance.get("calibration_target_count"),
        input_provenance.get("calibration_split_id"),
        input_provenance.get("test_split_id"),
    )
    if actual != expected:
        raise ValueError(
            "score run publication or candidate/recipe/dataset/split binding differs "
            "from calibration input provenance"
        )


def _validation_publication_fields(
    *,
    registration: VerifiedAtomicPublication,
    calibration: VerifiedAtomicPublication,
    score_run: VerifiedAtomicPublication,
    score_run_manifest: Mapping[str, object],
) -> dict[str, object]:
    """Expose every immutable publication consumed by candidate validation."""
    return {
        "registration_publication_id": registration.publication_id,
        "registration_publication_root_sha256": registration.root_sha256,
        "calibration_publication_id": calibration.publication_id,
        "calibration_publication_root_sha256": calibration.root_sha256,
        "score_run_id": score_run.publication_id,
        "score_run_root_sha256": score_run.root_sha256,
        "scores_sha256": score_run_manifest["scores_sha256"],
        "score_audit_sha256": score_run_manifest["score_audit_sha256"],
        "score_execution_receipt_sha256": object_value(
            score_run_manifest["execution_receipt"],
            "score_run.execution_receipt",
        )["receipt_sha256"],
    }


def _calibration_assets(
    root: Path,
    asset_root: Path,
    contents: Mapping[str, bytes],
) -> tuple[AssetFile, AssetFile, AssetFile, AssetFile, AssetFile]:
    base = asset_root.expanduser().resolve()
    try:
        relative_root = root.expanduser().resolve().relative_to(base)
    except ValueError as error:
        raise ValueError("calibration publication must stay below --asset-root") from error

    def asset(name: str, role: str) -> AssetFile:
        relative = (relative_root / name).as_posix()
        return AssetFile(
            role=role,
            path=root / name,
            sha256=hashlib.sha256(contents[name]).hexdigest(),
            media_type="application/json",
            logical_path=relative,
        )

    return (
        asset("template_thresholds.json", "template_thresholds"),
        asset("model_thresholds.json", "model_thresholds"),
        asset("metrics.json", "calibration_metrics"),
        asset("input_provenance.json", "calibration_input_provenance"),
        asset("calibration_artifact.json", "calibration_artifact"),
    )


def _run(argv: Sequence[str] | None) -> int:
    args = _parser().parse_args(argv)
    if (
        not args.validation_id.strip()
        or args.validation_id != args.validation_id.strip()
        or any(character.isspace() for character in args.validation_id)
    ):
        raise ValueError("validation-id must be non-empty and contain no whitespace")
    if (
        args.candidate.name != "candidate.json"
        or args.template_assets.name != "template_assets.json"
        or args.candidate.parent != args.template_assets.parent
    ):
        raise ValueError(
            "candidate/template descriptors must be candidate.json and template_assets.json "
            "from the same registration publication"
        )
    acceptance_policy_path = args.heldout_acceptance_policy.expanduser()
    if acceptance_policy_path.is_symlink() or not acceptance_policy_path.is_file():
        raise ValueError("held-out acceptance policy must be a regular non-symlink file")
    acceptance_policy_bytes = acceptance_policy_path.read_bytes()
    acceptance_policy = load_heldout_acceptance_policy_bytes(
        acceptance_policy_bytes
    )
    registration_publication = verify_atomic_publication(
        args.candidate.parent,
        required_paths=_REGISTRATION_FILES,
        allowed_paths=_REGISTRATION_FILES,
    )
    candidate_bytes = registration_publication.read_bytes("candidate.json")
    template_bytes = registration_publication.read_bytes("template_assets.json")
    candidate_descriptor_sha256 = hashlib.sha256(candidate_bytes).hexdigest()
    template_descriptor_sha256 = hashlib.sha256(template_bytes).hexdigest()
    candidate = parse_candidate(
        _canonical_object(candidate_bytes, "registered model candidate"),
        asset_root=args.asset_root,
    )
    if candidate.status is not CandidateStatus.REGISTERED:
        raise ValueError("candidate validation requires status=registered")
    required_slots, templates = parse_template_assets(
        _canonical_object(template_bytes, "template assets"),
        asset_root=args.asset_root,
        candidate=candidate,
    )
    candidate.validate_slots(required_slots)
    for artifact in candidate.anomaly_artifacts.values():
        artifact.verify()
    candidate.yolo.verify()
    for template in templates.values():
        template.verify()

    recipe = load_recipe(args.recipe)
    topology = load_topology(args.topology)
    roi = load_roi_config(args.roi)
    policy_bytes = args.fusion_policy.read_bytes()
    contract = compile_deployment_contract(
        topology,
        roi,
        recipe,
        fusion_policy_bytes=policy_bytes,
    )
    dataset = verify_dataset_release(args.dataset_release)
    dataset_manifest_digest = sha256_file(args.dataset_release / "dataset_release.json")
    expected_slots = tuple(
        ModelSlot(hand.value, view)
        for hand in contract.allowed_hands
        for view in contract.topology.required_views
    )
    if required_slots != expected_slots:
        raise ValueError("template/candidate slot order differs from the compiled deployment contract")
    if (
        candidate.recipe_digest != contract.recipe_sha256
        or candidate.topology_digest != topology.topology_sha256
        or candidate.roi_digest != roi.roi_sha256
        or candidate.roi_version != roi.roi_config_id
        or candidate.dataset_release_id != dataset.dataset_release_id
        or candidate.dataset_manifest_digest != dataset_manifest_digest
        or candidate.anomaly_family is not contract.anomaly_family
        or dataset.capture_gate_policy_sha256
        != contract.capture_gate_policy.sha256
        or set(dataset.hands)
        != {hand.value for hand in contract.allowed_hands}
        or dataset.split_assignments_sha256
        != next(iter(candidate.anomaly_artifacts.values())).train_split_id
    ):
        raise ValueError("candidate identity/provenance differs from contract or dataset release")
    if any(
        candidate.anomaly_artifacts[slot].train_split_id != dataset.split_assignments_sha256
        or templates[slot].train_split_id != dataset.split_assignments_sha256
        for slot in expected_slots
    ):
        raise ValueError("candidate/template training split differs from dataset split assignments")
    contract_anomaly = {
        ModelSlot(item.hand.value, item.view_id): item.asset.sha256
        for item in contract.anomaly_bindings
    }
    contract_templates = {
        ModelSlot(item.hand.value, item.view_id): item.asset.sha256
        for item in contract.template_bindings
    }
    if (
        any(candidate.anomaly_artifacts[slot].model_digest != contract_anomaly[slot] for slot in expected_slots)
        or any(templates[slot].model_digest != contract_templates[slot] for slot in expected_slots)
        or candidate.yolo.model_digest != contract.yolo_model.sha256
    ):
        raise ValueError("compiled recipe assets differ from the frozen candidate/template models")

    calibration_publication, contents = _load_calibration_publication(args.calibration)
    artifact_payload = _canonical_object(
        contents["calibration_artifact.json"],
        "calibration artifact",
    )
    artifact = reconstruct_calibration_artifact(artifact_payload)
    if artifact.payload() != artifact_payload or not artifact.calibration_valid:
        raise ValueError("calibration artifact is malformed or non-deployable")
    artifact_digest = hashlib.sha256(contents["calibration_artifact.json"]).hexdigest()
    expected_model_digests = {
        candidate.yolo.model_digest,
        *(candidate.anomaly_artifacts[slot].model_digest for slot in expected_slots),
        *(templates[slot].model_digest for slot in expected_slots),
    }
    provenance = artifact.provenance
    if (
        provenance.recipe_digest != contract.recipe_sha256
        or provenance.profile_digest != contract.fusion_policy_sha256
        or provenance.topology_digest != topology.topology_sha256
        or provenance.roi_digest != roi.roi_sha256
        or provenance.dataset_release_id != dataset.dataset_release_id
        or provenance.dataset_manifest_digest != dataset_manifest_digest
        or set(provenance.model_digests) != expected_model_digests
        or artifact.heldout_metrics.evaluation_complete is not True
    ):
        raise ValueError("calibration/held-out provenance differs from frozen release inputs")
    input_provenance = _canonical_object(
        contents["input_provenance.json"],
        "calibration input provenance",
    )
    expected_input_keys = {
        "schema", "schema_version", "calibration_id", "score_run_id",
        "score_run_root_sha256", "scores_sha256", "score_audit_sha256",
        "score_execution_receipt_sha256",
        "candidate_id", "candidate_digest", "candidate_descriptor_sha256",
        "recipe_digest", "dataset_release_id", "dataset_manifest_sha256",
        "calibration_targets_sha256", "calibration_target_count",
        "calibration_split_id", "test_split_id",
    }
    if (
        set(input_provenance) != expected_input_keys
        or input_provenance.get("schema") != "zs32.calibration_inputs"
        or input_provenance.get("schema_version") != 3
        or input_provenance.get("calibration_id") != args.calibration.name
        or input_provenance.get("candidate_id") != candidate.candidate_id
        or input_provenance.get("candidate_digest") != candidate.digest
        or input_provenance.get("candidate_descriptor_sha256")
        != candidate_descriptor_sha256
        or input_provenance.get("recipe_digest") != contract.recipe_sha256
        or input_provenance.get("dataset_release_id") != dataset.dataset_release_id
        or input_provenance.get("dataset_manifest_sha256") != dataset_manifest_digest
        or input_provenance.get("calibration_targets_sha256")
        != dataset.calibration_targets_sha256
        or input_provenance.get("calibration_target_count")
        != dataset.calibration_target_count
        or input_provenance.get("calibration_split_id") != provenance.calibration_split_id
        or input_provenance.get("test_split_id") != provenance.test_split_id
        or any(
            not isinstance(input_provenance.get(field), str)
            or SHA256_PATTERN.fullmatch(input_provenance[field]) is None
            for field in (
                "score_run_root_sha256",
                "scores_sha256",
                "score_audit_sha256",
                "score_execution_receipt_sha256",
                "calibration_targets_sha256",
            )
        )
        or not isinstance(input_provenance.get("score_run_id"), str)
        or not input_provenance["score_run_id"].strip()
        or contents["input_provenance.json"] != canonical_json_bytes(input_provenance)
    ):
        raise ValueError("calibration input provenance differs from candidate/recipe/dataset/splits")
    score_publication, score_run, _score_rows = _load_verified_score_run(args.score_run)
    _validate_score_run_provenance(
        score_run,
        recipe=recipe,
        topology=topology,
        roi=roi,
        dataset=dataset,
        dataset_manifest_digest=dataset_manifest_digest,
        candidate=candidate,
        candidate_descriptor_digest=candidate_descriptor_sha256,
        provenance=provenance,
    )
    _validate_score_run_input_binding(
        score_publication,
        score_run,
        input_provenance,
    )
    calibration_targets = load_calibration_targets(
        args.dataset_release / "calibration_targets.json"
    )
    selected_canonical_rows = _selected_rows(
        _load_canonical_rows(args.dataset_release),
        required_slots=tuple(sorted(expected_slots, key=lambda slot: slot.key)),
        required_views=topology.required_views,
    )
    _validate_score_target_replay(
        audit=_canonical_jsonl(
            score_publication.read_bytes("score_audit.jsonl"),
            "score_audit.jsonl",
        ),
        targets=calibration_targets,
        canonical_rows=selected_canonical_rows,
        calibration_split_id=provenance.calibration_split_id,
        test_split_id=provenance.test_split_id,
    )
    template_snapshot = _canonical_object(
        contents["template_thresholds.json"],
        "template threshold snapshot",
    )
    model_snapshot = _canonical_object(
        contents["model_thresholds.json"],
        "model threshold snapshot",
    )
    metrics_snapshot = _canonical_object(contents["metrics.json"], "calibration metrics")
    if template_snapshot != {
        "schema_version": 1,
        "calibration_artifact_sha256": artifact_digest,
        "thresholds": artifact_payload["template_thresholds"],
    }:
        raise ValueError("template threshold snapshot differs from calibration artifact")
    if model_snapshot != {
        "schema_version": 1,
        "calibration_artifact_sha256": artifact_digest,
        "thresholds": artifact_payload["dual_thresholds"],
    }:
        raise ValueError("model threshold snapshot differs from calibration artifact")
    if metrics_snapshot != artifact.heldout_metrics.to_dict():
        raise ValueError("held-out metrics snapshot differs from calibration artifact")
    acceptance_decision = evaluate_heldout_acceptance(
        acceptance_policy,
        artifact.heldout_metrics,
        metrics_sha256=hashlib.sha256(contents["metrics.json"]).hexdigest(),
    )
    if {item.calibration_sha256 for item in contract.template_thresholds} != {artifact_digest}:
        raise ValueError("compiled template thresholds do not bind this calibration artifact")
    if {item.calibration_sha256 for item in contract.model_thresholds} != {artifact_digest}:
        raise ValueError("compiled model thresholds do not bind this calibration artifact")

    validated = transition_candidate(candidate, CandidateStatus.VALIDATED)
    (
        template_thresholds,
        model_thresholds,
        metrics,
        calibration_input_provenance,
        calibration_artifact,
    ) = _calibration_assets(
        args.calibration,
        args.asset_root,
        contents,
    )
    deployment = DeploymentAssetDescription(
        candidate=validated,
        required_slots=expected_slots,
        template_assets=templates,
        template_thresholds=template_thresholds,
        model_thresholds=model_thresholds,
        calibration_metrics=metrics,
        calibration_input_provenance=calibration_input_provenance,
        calibration_artifact=calibration_artifact,
        calibration_artifact_digest=artifact_digest,
    )
    deployment.verify()
    reverified_calibration, reverified_calibration_contents = (
        _load_calibration_publication(args.calibration)
    )
    if (
        reverified_calibration.root_sha256 != calibration_publication.root_sha256
        or reverified_calibration_contents != contents
    ):
        raise ValueError("calibration publication changed during candidate validation")
    reverified_registration = verify_atomic_publication(
        args.candidate.parent,
        required_paths=_REGISTRATION_FILES,
        allowed_paths=_REGISTRATION_FILES,
    )
    if reverified_registration.root_sha256 != registration_publication.root_sha256:
        raise ValueError("candidate registration publication changed during validation")

    def reverify_score_run() -> None:
        reverified_publication, reverified_run, _reverified_rows = (
            _load_verified_score_run(args.score_run)
        )
        if (
            reverified_publication.root_sha256 != score_publication.root_sha256
            or reverified_run != score_run
        ):
            raise ValueError("score run publication changed during candidate validation")
        _validate_score_run_input_binding(
            reverified_publication,
            reverified_run,
            input_provenance,
        )

    def reverify_acceptance_policy() -> None:
        if (
            acceptance_policy_path.is_symlink()
            or not acceptance_policy_path.is_file()
            or acceptance_policy_path.read_bytes() != acceptance_policy_bytes
        ):
            raise ValueError(
                "held-out acceptance policy changed during candidate validation"
            )

    reverify_score_run()
    reverify_acceptance_policy()
    validation_record = {
        "schema": "zs32.candidate_validation",
        "schema_version": 3,
        "validation_id": args.validation_id,
        "candidate_id": validated.candidate_id,
        "candidate_digest": validated.digest,
        "input_candidate_descriptor_sha256": candidate_descriptor_sha256,
        "input_template_assets_descriptor_sha256": template_descriptor_sha256,
        "dataset_release_id": dataset.dataset_release_id,
        "dataset_manifest_sha256": dataset_manifest_digest,
        "recipe_sha256": contract.recipe_sha256,
        "contract_sha256": contract.contract_sha256,
        "calibration_artifact_sha256": artifact_digest,
        "calibration_split_id": provenance.calibration_split_id,
        "test_split_id": provenance.test_split_id,
        "candidate_status_before": CandidateStatus.REGISTERED.value,
        "candidate_status_after": CandidateStatus.VALIDATED.value,
        "manual_promotion_required": True,
        "heldout_acceptance_policy_id": acceptance_policy.policy_id,
        "heldout_acceptance_policy_version": acceptance_policy.policy_version,
        "heldout_acceptance_policy_sha256": acceptance_policy.sha256,
        "heldout_acceptance_decision_sha256": acceptance_decision.sha256,
        **_validation_publication_fields(
            registration=registration_publication,
            calibration=calibration_publication,
            score_run=score_publication,
            score_run_manifest=score_run,
        ),
    }

    def validate(staging: Path) -> None:
        reverify_score_run()
        reverify_acceptance_policy()
        latest_registration = verify_atomic_publication(
            args.candidate.parent,
            required_paths=_REGISTRATION_FILES,
            allowed_paths=_REGISTRATION_FILES,
        )
        latest_calibration, latest_calibration_contents = (
            _load_calibration_publication(args.calibration)
        )
        if latest_registration.root_sha256 != registration_publication.root_sha256:
            raise PublicationError(
                "candidate registration publication changed before validation publish"
            )
        if (
            latest_calibration.root_sha256 != calibration_publication.root_sha256
            or latest_calibration_contents != contents
        ):
            raise PublicationError(
                "calibration publication changed before validation publish"
            )
        if load_object(staging / "candidate.json", "validated candidate") != validated.to_dict():
            raise PublicationError("staged validated candidate changed")
        if load_object(staging / "deployment_assets.json", "deployment assets") != deployment.to_dict():
            raise PublicationError("staged deployment assets changed")
        if load_object(staging / "validation.json", "validation record") != validation_record:
            raise PublicationError("staged validation record changed")
        if (
            (staging / "heldout_acceptance_policy.json").read_bytes()
            != acceptance_policy_bytes
        ):
            raise PublicationError("staged held-out acceptance policy changed")
        if (
            (staging / "heldout_acceptance.json").read_bytes()
            != acceptance_decision.canonical_bytes()
        ):
            raise PublicationError("staged held-out acceptance decision changed")

    with AtomicDirectoryPublisher(args.output_root, args.validation_id) as publisher:
        publisher.write_json("candidate.json", validated.to_dict())
        publisher.write_json("deployment_assets.json", deployment.to_dict())
        publisher.write_json("validation.json", validation_record)
        publisher.write_bytes(
            "heldout_acceptance_policy.json", acceptance_policy_bytes
        )
        publisher.write_bytes(
            "heldout_acceptance.json", acceptance_decision.canonical_bytes()
        )
        published = publisher.finalize(
            validator=validate,
            required_paths=frozenset(
                {
                    "candidate.json",
                    "deployment_assets.json",
                    "validation.json",
                    "heldout_acceptance_policy.json",
                    "heldout_acceptance.json",
                }
            ),
        )
    command_result(
        "zs32-validate-candidate",
        {
            "candidate_digest": validated.digest,
            "calibration_artifact_sha256": artifact_digest,
            "contract_sha256": contract.contract_sha256,
            "heldout_acceptance_policy_id": acceptance_policy.policy_id,
            "heldout_acceptance_policy_version": acceptance_policy.policy_version,
            "heldout_acceptance_policy_sha256": acceptance_policy.sha256,
            "heldout_acceptance_decision_sha256": acceptance_decision.sha256,
            "published_path": str(published),
            "manual_promotion_required": True,
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        require_linux_nvidia()
        return _run(argv)
    except Exception as error:
        return command_error("zs32-validate-candidate", error)


if __name__ == "__main__":
    raise SystemExit(main())
