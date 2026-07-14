# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Linux test definitions for the immutable model candidate registry."""

from __future__ import annotations

import hashlib

import pytest

from tests.unit.zs32_refactor.execution_fixtures import execution_receipt_mapping

from zs32_inspection.models import (
    AnomalyFamily,
    AnomalyModelArtifact,
    AssetFile,
    ModelCandidate,
    ModelContractError,
    ModelSlot,
    register_candidate,
    transition_candidate,
)
from zs32_inspection.models.registry import CandidateStatus
from zs32_inspection.models.yolo import import_yolo_bundle
from zs32_inspection.runtime.publisher import canonical_json_bytes

HEX = "1" * 64


def _yolo(tmp_path):
    bundle = tmp_path / "yolo"
    bundle.mkdir(exist_ok=True)
    (bundle / "best.pt").write_bytes(b"global-yolo")
    (bundle / "args.yaml").write_text("data: /exports/yolo/data.yaml\nseed: 43\n", encoding="utf-8")
    (bundle / "data.yaml").write_text("names:\n  0: defect\n", encoding="utf-8")
    (bundle / "class_names.yaml").write_text("0: defect\n", encoding="utf-8")
    receipt = tmp_path / "training_receipt.json"
    receipt.write_bytes(canonical_json_bytes({
        "schema": "zs32.yolo_external_training_receipt",
        "schema_version": 3,
        "attestation_scope": "external_trainer_observation_not_causal_proof",
        "attested_by": "test",
        "attested_at": "2026-07-14T12:00:00Z",
        "run": {"run_id": "run-1", "run_name": "run-seed43", "actual_seed": 43},
        "artifacts": {
            name: hashlib.sha256((bundle / name).read_bytes()).hexdigest()
            for name in ("best.pt", "args.yaml", "data.yaml", "class_names.yaml")
        },
        "training_data": {
            "yolo_export_publication_id": "yolo-export-v1",
            "yolo_export_root_sha256": HEX,
            "export_manifest_sha256": HEX,
            "export_data_yaml_sha256": hashlib.sha256((bundle / "data.yaml").read_bytes()).hexdigest(),
            "export_policy_sha256": "9" * 64,
            "args_data_reference": "/exports/yolo/data.yaml",
            "dataset_release_id": "dataset-v1",
            "dataset_manifest_sha256": HEX,
        },
        "trainer_source": {
            "kind": "git_commit", "repository": "test", "commit": "a" * 40,
            "worktree_state": "clean",
            "runtime_package_tree_sha256": "f" * 64,
        },
    }))
    return import_yolo_bundle(
        bundle,
        training_receipt_path=receipt,
    )


def _anomaly(tmp_path, slot: ModelSlot, family: AnomalyFamily) -> AnomalyModelArtifact:
    path = tmp_path / f"{slot.hand}-{slot.view}-{family.value}.ckpt"
    path.write_bytes(path.name.encode())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return AnomalyModelArtifact(
        family=family,
        slot=slot,
        checkpoint=AssetFile("checkpoint", path, digest),
        model_digest=digest,
        dataset_release_id="dataset-v1",
        dataset_manifest_digest=HEX,
        train_split_id="train-v1",
        recipe_digest=HEX,
        roi_version="roi-v1",
        roi_digest=HEX,
        framework_version="test",
        training_parameters={},
        execution_receipt=execution_receipt_mapping("train_anomaly"),
    )


def _candidate(tmp_path, candidate_id: str) -> ModelCandidate:
    slots = (ModelSlot("right", "front"), ModelSlot("right", "back"))
    artifacts = {slot: _anomaly(tmp_path, slot, AnomalyFamily.PATCHCORE) for slot in slots}
    return ModelCandidate(
        candidate_id=candidate_id,
        product="ZS32",
        anomaly_family=AnomalyFamily.PATCHCORE,
        anomaly_artifacts=artifacts,
        yolo=_yolo(tmp_path),
        recipe_digest=HEX,
        topology_digest=HEX,
        roi_digest=HEX,
        roi_version="roi-v1",
        dataset_release_id="dataset-v1",
        dataset_manifest_digest=HEX,
    )


def test_candidate_rejects_family_mixing(tmp_path) -> None:
    slot = ModelSlot("right", "front")
    artifact = _anomaly(tmp_path, slot, AnomalyFamily.EFFICIENTAD)
    with pytest.raises(ModelContractError, match="mix anomaly families"):
        ModelCandidate(
            candidate_id="candidate-1",
            product="ZS32",
            anomaly_family=AnomalyFamily.PATCHCORE,
            anomaly_artifacts={slot: artifact},
            yolo=_yolo(tmp_path),
            recipe_digest=HEX,
            topology_digest=HEX,
            roi_digest=HEX,
            roi_version="roi-v1",
            dataset_release_id="dataset-v1",
            dataset_manifest_digest=HEX,
        )


def test_candidate_digest_is_independent_of_registry_id(tmp_path) -> None:
    first = _candidate(tmp_path, "candidate-1")
    second = _candidate(tmp_path, "candidate-2")
    assert first.digest == second.digest
    with pytest.raises(ModelContractError, match="already registered"):
        register_candidate((first,), second)


def test_candidate_validation_is_explicit_immutable_transition(tmp_path) -> None:
    registered = _candidate(tmp_path, "candidate-1")

    validated = transition_candidate(registered, CandidateStatus.VALIDATED)

    assert registered.status is CandidateStatus.REGISTERED
    assert validated.status is CandidateStatus.VALIDATED
    assert validated.digest == registered.digest
    with pytest.raises(ModelContractError, match="terminal"):
        transition_candidate(validated, CandidateStatus.REJECTED)
