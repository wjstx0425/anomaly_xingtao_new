"""Linux tests for binding a YOLO receipt to actual immutable publications."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from zs32_inspection.cli import import_yolo as subject
from zs32_inspection.models.base import sha256_file
from zs32_inspection.models.yolo_receipt import YoloTrainerSource, YoloTrainingReceipt
from zs32_inspection.runtime.publisher import (
    AtomicDirectoryPublisher,
    canonical_json_bytes,
    verify_atomic_publication,
)


def _publication(tmp_path):
    dataset = tmp_path / "dataset-v1"
    dataset.mkdir()
    manifest = dataset / "dataset_release.json"
    manifest.write_bytes(canonical_json_bytes({"dataset_release_id": "dataset-v1"}))
    with AtomicDirectoryPublisher(tmp_path / "exports", "yolo-export-v1") as publisher:
        publisher.write_bytes("export_manifest.csv", b"sample_id\nexample\n")
        publisher.write_bytes("data.yaml", b"names:\n  0: defect\n")
        publisher.write_bytes("training_export_policy.json", b"{}\n")
        publisher.copy_file(
            manifest,
            "provenance/dataset_release.json",
            expected_sha256=sha256_file(manifest),
        )
        path = publisher.finalize(
            validator=lambda _staging: None,
            required_paths=frozenset(
                {
                    "export_manifest.csv",
                    "data.yaml",
                    "training_export_policy.json",
                    "provenance/dataset_release.json",
                }
            ),
        )
    return dataset, manifest, verify_atomic_publication(path)


def test_receipt_is_checked_against_actual_dataset_and_export_publications(
    tmp_path,
    monkeypatch,
) -> None:
    dataset, manifest, export = _publication(tmp_path)
    monkeypatch.setattr(
        subject,
        "verify_dataset_release",
        lambda _path: SimpleNamespace(dataset_release_id="dataset-v1"),
    )
    monkeypatch.setattr(subject, "validate_yolo_training_export", lambda _publication: None)
    receipt = YoloTrainingReceipt(
        attested_by="test",
        attested_at="2026-07-14T12:00:00Z",
        run_id="run-1",
        run_name="run-seed43",
        actual_seed=43,
        artifact_sha256={name: "a" * 64 for name in (
            "best.pt", "args.yaml", "data.yaml", "class_names.yaml"
        )},
        yolo_export_publication_id=export.publication_id,
        yolo_export_root_sha256=export.root_sha256,
        export_manifest_sha256=export.checksums["export_manifest.csv"],
        export_data_yaml_sha256=export.checksums["data.yaml"],
        export_policy_sha256=export.checksums["training_export_policy.json"],
        args_data_reference="/exports/yolo-export-v1/data.yaml",
        dataset_release_id="dataset-v1",
        dataset_manifest_sha256=sha256_file(manifest),
        trainer_source=YoloTrainerSource(
            kind="git_commit",
            repository="test",
            commit="b" * 40,
            runtime_package_tree_sha256="f" * 64,
        ),
        receipt_sha256="c" * 64,
    )
    verified = subject._verify_training_data(
        receipt=receipt,
        dataset_release=dataset,
        yolo_export=export.root,
    )
    assert verified.root_sha256 == export.root_sha256

    forged = replace(receipt, yolo_export_root_sha256="d" * 64)
    with pytest.raises(ValueError, match="verified YOLO export publication"):
        subject._verify_training_data(
            receipt=forged,
            dataset_release=dataset,
            yolo_export=export.root,
        )
