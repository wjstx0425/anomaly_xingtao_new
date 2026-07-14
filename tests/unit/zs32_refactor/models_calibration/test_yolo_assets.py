# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Linux test definitions for strict external YOLO training receipts."""

from __future__ import annotations

import hashlib
import json

import pytest

from zs32_inspection.models import ModelContractError
from zs32_inspection.models.yolo import import_yolo_bundle
from zs32_inspection.models.yolo_receipt import load_yolo_training_receipt
from zs32_inspection.runtime.publisher import canonical_json_bytes

HEX = "b" * 64
COMMIT = "c" * 40
RUNTIME_TREE = "f" * 64


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(tmp_path, *, seed: int = 43, run_name: str = "final_n640_p1_seed43"):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "best.pt").write_bytes(b"trained-weights")
    (bundle / "args.yaml").write_text(
        f"task: detect\ndata: /immutable/exports/yolo-r1/data.yaml\nseed: {seed}\nproject: /training/only\n",
        encoding="utf-8",
    )
    (bundle / "data.yaml").write_text(
        "train: images/train\nval: images/model_val\n\nnames:\n  0: defect\n",
        encoding="utf-8",
    )
    (bundle / "class_names.yaml").write_text("0: defect\n", encoding="utf-8")
    receipt = tmp_path / "training_receipt.json"
    payload = {
        "schema": "zs32.yolo_external_training_receipt",
        "schema_version": 3,
        "attestation_scope": "external_trainer_observation_not_causal_proof",
        "attested_by": "linux-yolo-operator",
        "attested_at": "2026-07-14T12:00:00+08:00",
        "run": {"run_id": "run-001", "run_name": run_name, "actual_seed": seed},
        "artifacts": {
            name: _sha(bundle / name)
            for name in ("best.pt", "args.yaml", "data.yaml", "class_names.yaml")
        },
        "training_data": {
            "yolo_export_publication_id": "yolo-export-r1",
            "yolo_export_root_sha256": "d" * 64,
            "export_manifest_sha256": "e" * 64,
            "export_data_yaml_sha256": _sha(bundle / "data.yaml"),
            "export_policy_sha256": "9" * 64,
            "args_data_reference": "/immutable/exports/yolo-r1/data.yaml",
            "dataset_release_id": "dataset-v1",
            "dataset_manifest_sha256": HEX,
        },
        "trainer_source": {
            "kind": "git_commit",
            "repository": "ssh://git.example/ultralytics.git",
            "commit": COMMIT,
            "worktree_state": "clean",
            "runtime_package_tree_sha256": RUNTIME_TREE,
        },
    }
    receipt.write_bytes(canonical_json_bytes(payload))
    return bundle, receipt, payload


def test_yolo_import_rejects_the_known_seed_name_contradiction(tmp_path) -> None:
    bundle, receipt, _ = _bundle(tmp_path, seed=43, run_name="final_n640_p1_seed42")
    with pytest.raises(ModelContractError, match="contradicts"):
        import_yolo_bundle(bundle, training_receipt_path=receipt)


def test_yolo_import_hashes_receipt_and_all_required_assets(tmp_path) -> None:
    bundle, receipt, _ = _bundle(tmp_path)
    spec = import_yolo_bundle(bundle, training_receipt_path=receipt)
    assert spec.weights.path.name == "best.pt"
    assert {asset.role for asset in spec.assets} == {
        "weights", "train_args", "dataset_config", "class_names", "training_receipt"
    }
    assert spec.provenance.run_id == "run-001"
    assert spec.provenance.yolo_export_publication_id == "yolo-export-r1"
    assert spec.provenance.yolo_export_policy_digest == "9" * 64
    assert spec.provenance.trainer_source["worktree_state"] == "clean"
    assert spec.runtime.single_class_name == "defect"
    assert "project" not in spec.runtime.to_dict()
    spec.verify()


def test_yolo_load_verification_detects_post_import_weight_change(tmp_path) -> None:
    bundle, receipt, _ = _bundle(tmp_path)
    spec = import_yolo_bundle(bundle, training_receipt_path=receipt)
    (bundle / "best.pt").write_bytes(b"tampered")
    with pytest.raises(ModelContractError, match="SHA-256 mismatch"):
        spec.verify()


def test_yolo_import_rejects_asset_digest_not_attested_by_trainer(tmp_path) -> None:
    bundle, receipt, payload = _bundle(tmp_path)
    payload["artifacts"]["best.pt"] = "f" * 64
    receipt.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ModelContractError, match="receipt digest mismatch for best.pt"):
        import_yolo_bundle(bundle, training_receipt_path=receipt)


def test_yolo_import_rejects_unbound_args_data_reference(tmp_path) -> None:
    bundle, receipt, payload = _bundle(tmp_path)
    payload["training_data"]["args_data_reference"] = "/different/export/data.yaml"
    receipt.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ModelContractError, match="data reference differs"):
        import_yolo_bundle(bundle, training_receipt_path=receipt)


def test_yolo_import_requires_trainer_data_yaml_to_equal_atomic_export_data(tmp_path) -> None:
    bundle, receipt, payload = _bundle(tmp_path)
    payload["training_data"]["export_data_yaml_sha256"] = "a" * 64
    receipt.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ModelContractError, match="byte-identical"):
        import_yolo_bundle(bundle, training_receipt_path=receipt)


def test_yolo_import_rejects_any_business_class_other_than_defect(tmp_path) -> None:
    bundle, receipt, payload = _bundle(tmp_path)
    (bundle / "class_names.yaml").write_text("0: scratch\n", encoding="utf-8")
    payload["artifacts"]["class_names.yaml"] = _sha(bundle / "class_names.yaml")
    receipt.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ModelContractError, match="class 0 to 'defect'"):
        import_yolo_bundle(bundle, training_receipt_path=receipt)


def test_receipt_rejects_unknown_fields_and_noncanonical_json(tmp_path) -> None:
    _, receipt, payload = _bundle(tmp_path)
    payload["unreviewed"] = True
    receipt.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ModelContractError, match="exact keys"):
        load_yolo_training_receipt(receipt)
    payload.pop("unreviewed")
    receipt.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(ModelContractError, match="canonical JSON"):
        load_yolo_training_receipt(receipt)


def test_receipt_v2_is_rejected_because_it_does_not_bind_training_isolation(tmp_path) -> None:
    _, receipt, payload = _bundle(tmp_path)
    payload["schema_version"] = 2
    payload["training_data"].pop("export_policy_sha256")
    receipt.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ModelContractError, match="version 3"):
        load_yolo_training_receipt(receipt)


def test_receipt_requires_timezone_and_clean_exact_source(tmp_path) -> None:
    _, receipt, payload = _bundle(tmp_path)
    payload["attested_at"] = "2026-07-14T12:00:00"
    receipt.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ModelContractError, match="RFC3339"):
        load_yolo_training_receipt(receipt)
    payload["attested_at"] = "2026-07-14T12:00:00Z"
    payload["trainer_source"]["worktree_state"] = "dirty"
    receipt.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ModelContractError, match="worktree_state='clean'"):
        load_yolo_training_receipt(receipt)


def test_receipt_strict_schema_rejects_text_field_with_number_type(tmp_path) -> None:
    _, receipt, payload = _bundle(tmp_path)
    payload["run"]["run_id"] = 123
    receipt.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ModelContractError, match="run_id must be a string"):
        load_yolo_training_receipt(receipt)


def test_receipt_accepts_content_addressed_source_bundle(tmp_path) -> None:
    _, receipt, payload = _bundle(tmp_path)
    digest = "a" * 64
    payload["trainer_source"] = {
        "kind": "content_addressed_bundle",
        "bundle_sha256": digest,
        "immutable_storage_reference": f"sha256/{digest}/ultralytics-source.tar.zst",
        "runtime_package_tree_sha256": RUNTIME_TREE,
    }
    receipt.write_bytes(canonical_json_bytes(payload))
    loaded = load_yolo_training_receipt(receipt)
    assert loaded.trainer_source.runtime_identity == digest
