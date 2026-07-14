"""Linux-only tests for the manual release validation-publication boundary."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from zs32_inspection.calibration.acceptance import (
    HeldOutAcceptanceDecision,
    HeldOutAcceptancePolicy,
    evaluate_heldout_acceptance,
)
from zs32_inspection.calibration.reports import HeldOutPartMetrics
from zs32_inspection.cli.assemble_release import (
    _load_validation_publication,
    _reverify_publication,
    _validate_validation_record,
)
from zs32_inspection.models.deployment import DeploymentAssetDescription
from zs32_inspection.models.registry import ModelCandidate
from zs32_inspection.runtime.publisher import (
    AtomicDirectoryPublisher,
    PublicationError,
    VerifiedAtomicPublication,
)


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="ZS32 verification is Linux-only")

_CANDIDATE_DIGEST = "1" * 64
_DATASET_DIGEST = "2" * 64
_RECIPE_DIGEST = "3" * 64
_CONTRACT_DIGEST = "4" * 64
_CALIBRATION_DIGEST = "5" * 64


def _acceptance() -> tuple[HeldOutAcceptancePolicy, HeldOutAcceptanceDecision]:
    policy = HeldOutAcceptancePolicy.from_mapping(
        {
            "schema": "zs32.heldout_acceptance_policy",
            "schema_version": 1,
            "product": "ZS32",
            "policy_id": "heldout-strict-v1",
            "policy_version": "1.0.0",
            "limits": {
                "max_defect_escape_rate": 0.0,
                "max_normal_reject_rate": 0.0,
                "max_review_rate": 0.0,
                "min_normal_heldout_parts": 1,
                "min_defect_heldout_parts": 1,
            },
        }
    )
    metrics = HeldOutPartMetrics(
        dataset_release_id="dataset-v1",
        test_split_id="test-split-v1",
        part_count=2,
        normal_part_count=1,
        defect_part_count=1,
        clear_count=1,
        gray_count=0,
        strong_count=1,
        template_ng_count=0,
        incomplete_count=0,
        defect_escape_count=0,
        normal_reject_count=0,
        defect_escape_rate=0.0,
        normal_reject_rate=0.0,
        review_rate=0.0,
        evaluation_complete=True,
    )
    return policy, evaluate_heldout_acceptance(
        policy, metrics, metrics_sha256="e" * 64
    )


def _validation_record(*, validation_id: str = "validation-v1") -> dict[str, object]:
    policy, decision = _acceptance()
    return {
        "schema": "zs32.candidate_validation",
        "schema_version": 3,
        "validation_id": validation_id,
        "candidate_id": "candidate-v1",
        "candidate_digest": _CANDIDATE_DIGEST,
        "input_candidate_descriptor_sha256": "6" * 64,
        "input_template_assets_descriptor_sha256": "7" * 64,
        "dataset_release_id": "dataset-v1",
        "dataset_manifest_sha256": _DATASET_DIGEST,
        "recipe_sha256": _RECIPE_DIGEST,
        "contract_sha256": _CONTRACT_DIGEST,
        "calibration_artifact_sha256": _CALIBRATION_DIGEST,
        "calibration_split_id": "calibration-split-v1",
        "test_split_id": "test-split-v1",
        "candidate_status_before": "registered",
        "candidate_status_after": "validated",
        "manual_promotion_required": True,
        "heldout_acceptance_policy_id": policy.policy_id,
        "heldout_acceptance_policy_version": policy.policy_version,
        "heldout_acceptance_policy_sha256": policy.sha256,
        "heldout_acceptance_decision_sha256": decision.sha256,
        "score_run_id": "score-run-v1",
        "score_run_root_sha256": "8" * 64,
        "scores_sha256": "9" * 64,
        "score_audit_sha256": "a" * 64,
        "score_execution_receipt_sha256": "d" * 64,
        "calibration_publication_id": "calibration-v1",
        "calibration_publication_root_sha256": "b" * 64,
        "registration_publication_id": "registration-v1",
        "registration_publication_root_sha256": "c" * 64,
    }


def _publication(
    output_root: Path,
    publication_id: str,
    *,
    noncanonical_candidate: bool = False,
    include_extra: bool = False,
) -> Path:
    policy, decision = _acceptance()
    with AtomicDirectoryPublisher(output_root, publication_id) as publisher:
        if noncanonical_candidate:
            publisher.write_bytes("candidate.json", b'{"z": 1, "a": 2}\n')
        else:
            publisher.write_json("candidate.json", {"candidate": publication_id})
        publisher.write_json("deployment_assets.json", {"deployment": publication_id})
        publisher.write_json(
            "validation.json",
            _validation_record(validation_id=publication_id),
        )
        publisher.write_bytes(
            "heldout_acceptance_policy.json", policy.canonical_bytes()
        )
        publisher.write_bytes("heldout_acceptance.json", decision.canonical_bytes())
        if include_extra:
            publisher.write_json("forged.json", {"forged": True})
        return publisher.finalize(
            validator=lambda _staging: None,
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


def test_boundary_accepts_one_complete_canonical_validation_publication(
    tmp_path: Path,
) -> None:
    root = _publication(tmp_path, "validation-v1")

    loaded = _load_validation_publication(
        root / "candidate.json",
        root / "deployment_assets.json",
    )

    assert loaded.publication.publication_id == "validation-v1"
    assert loaded.candidate_payload == {"candidate": "validation-v1"}
    assert loaded.deployment_payload == {"deployment": "validation-v1"}
    assert loaded.validation_record["manual_promotion_required"] is True
    assert loaded.heldout_acceptance_decision.policy_sha256 == (
        loaded.heldout_acceptance_policy.sha256
    )


def test_boundary_rejects_descriptors_from_different_publications(tmp_path: Path) -> None:
    first = _publication(tmp_path, "validation-first")
    second = _publication(tmp_path, "validation-second")

    with pytest.raises(ValueError, match="same validation publication"):
        _load_validation_publication(
            first / "candidate.json",
            second / "deployment_assets.json",
        )


def test_boundary_rejects_noncanonical_descriptor_bytes(tmp_path: Path) -> None:
    root = _publication(
        tmp_path,
        "validation-noncanonical",
        noncanonical_candidate=True,
    )

    with pytest.raises(ValueError, match="canonical JSON"):
        _load_validation_publication(
            root / "candidate.json",
            root / "deployment_assets.json",
        )


def test_boundary_rejects_extra_indexed_files(tmp_path: Path) -> None:
    root = _publication(tmp_path, "validation-extra", include_extra=True)

    with pytest.raises(PublicationError, match="unexpected indexed files"):
        _load_validation_publication(
            root / "candidate.json",
            root / "deployment_assets.json",
        )


def test_boundary_reverification_rejects_post_read_mutation(tmp_path: Path) -> None:
    root = _publication(tmp_path, "validation-mutated")
    loaded = _load_validation_publication(
        root / "candidate.json",
        root / "deployment_assets.json",
    )
    root.chmod(0o750)
    candidate_path = root / "candidate.json"
    candidate_path.chmod(0o640)
    candidate_path.write_bytes(b'{"candidate":"forged"}\n')

    with pytest.raises(PublicationError, match="checksum mismatch"):
        _reverify_publication(loaded.publication)


def _validate_record(record: dict[str, Any]) -> None:
    policy, decision = _acceptance()
    candidate = cast(
        ModelCandidate,
        SimpleNamespace(candidate_id="candidate-v1", digest=_CANDIDATE_DIGEST),
    )
    assets = cast(
        DeploymentAssetDescription,
        SimpleNamespace(calibration_artifact_digest=_CALIBRATION_DIGEST),
    )
    calibration_publication = VerifiedAtomicPublication(
        root=Path("/calibration-v1"),
        publication_id="calibration-v1",
        root_sha256="b" * 64,
        checksums={},
        owner_uid=0,
    )
    _validate_validation_record(
        record,
        validation_id="validation-v1",
        candidate=candidate,
        assets=assets,
        contract_sha256=_CONTRACT_DIGEST,
        recipe_sha256=_RECIPE_DIGEST,
        dataset_release_id="dataset-v1",
        dataset_manifest_sha256=_DATASET_DIGEST,
        calibration_split_id="calibration-split-v1",
        test_split_id="test-split-v1",
        calibration_publication=calibration_publication,
        calibration_input_provenance={
            "score_run_id": "score-run-v1",
            "score_run_root_sha256": "8" * 64,
            "scores_sha256": "9" * 64,
            "score_audit_sha256": "a" * 64,
            "score_execution_receipt_sha256": "d" * 64,
        },
        heldout_acceptance_policy=policy,
        heldout_acceptance_decision=decision,
    )


def test_validation_record_binds_release_inputs_and_requires_manual_promotion() -> None:
    _validate_record(cast(dict[str, Any], _validation_record()))


@pytest.mark.parametrize(
    ("field", "forged_value"),
    [
        ("candidate_id", "other-candidate"),
        ("contract_sha256", "8" * 64),
        ("calibration_artifact_sha256", "9" * 64),
        ("dataset_manifest_sha256", "a" * 64),
        ("calibration_split_id", "other-calibration-split"),
        ("test_split_id", "other-test-split"),
        ("score_run_root_sha256", "d" * 64),
        ("calibration_publication_root_sha256", "e" * 64),
        ("heldout_acceptance_policy_sha256", "f" * 64),
        ("heldout_acceptance_decision_sha256", "0" * 64),
        ("manual_promotion_required", False),
    ],
)
def test_validation_record_rejects_unbound_or_automatic_promotion(
    field: str,
    forged_value: object,
) -> None:
    record = cast(dict[str, Any], _validation_record())
    record[field] = forged_value

    with pytest.raises(ValueError, match="differs from candidate/contract/calibration/dataset"):
        _validate_record(record)
