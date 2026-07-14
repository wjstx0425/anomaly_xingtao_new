"""Materialized training exports must be complete projections of adapter manifests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tests.unit.zs32_refactor.execution_fixtures import execution_receipt_mapping

import zs32_inspection.models.anomalib_backend as anomalib_backend
import zs32_inspection.template.opencv_backend as opencv_backend
from zs32_inspection.data.anomalib_export import anomalib_materialized_path
from zs32_inspection.data.export_common import csv_bytes
from zs32_inspection.data.manifests import ADAPTER_BASE_COLUMNS
from zs32_inspection.data.template_export import template_materialized_path
from zs32_inspection.models import DeviceSpec, ModelContractError, ModelSlot, TrainSpec
from zs32_inspection.runtime.publisher import (
    AtomicDirectoryPublisher,
    canonical_json_bytes,
    sha256_file,
)


def _rows() -> tuple[dict[str, str], ...]:
    payloads = (b"\x89PNG\r\n\x1a\nfirst", b"\x89PNG\r\n\x1a\nsecond")
    return tuple(
        {
            "sample_id": f"sample-{index}",
            "part_instance_id": f"part-{index}",
            "capture_set_id": f"capture-{index}",
            "hand": "right",
            "view": "front",
            "split": "train",
            "label": "normal",
            "defect_type": "",
            "canonical_crop_path": f"canonical/train/sample-{index}.png",
            "canonical_crop_sha256": hashlib.sha256(payload).hexdigest(),
        }
        for index, payload in enumerate(payloads, start=1)
    )


def _materialized_publication(
    tmp_path: Path,
    *,
    adapter: str,
    adapter_indices: tuple[int, ...] = (0, 1),
    materialized_indices: tuple[int, ...] = (0, 1),
    mismatched_adapter_digest: bool = False,
    tampered_crop: bool = False,
) -> tuple[Path, str, str]:
    all_rows = _rows()
    adapter_rows = [all_rows[index] for index in adapter_indices]
    destination_for = (
        anomalib_materialized_path if adapter == "anomalib" else template_materialized_path
    )
    materialized_rows = [
        {**all_rows[index], "materialized_path": destination_for(all_rows[index])}
        for index in materialized_indices
    ]
    adapter_bytes = csv_bytes(ADAPTER_BASE_COLUMNS, adapter_rows)
    adapter_digest = hashlib.sha256(adapter_bytes).hexdigest()

    def digest_for(name: str, fallback: str) -> str:
        if adapter != name:
            return fallback
        return "f" * 64 if mismatched_adapter_digest else adapter_digest

    dataset_payload = {
        "dataset_release_id": "dataset-v1",
        "adapter_manifest_sha256": {
            "yolo": "1" * 64,
            "anomalib": digest_for("anomalib", "2" * 64),
            "template": digest_for("template", "3" * 64),
        },
    }
    dataset_bytes = canonical_json_bytes(dataset_payload)
    manifest_columns = (*ADAPTER_BASE_COLUMNS, "materialized_path")
    publication_id = (
        f"{adapter}-a{''.join(map(str, adapter_indices))}-"
        f"m{''.join(map(str, materialized_indices))}-"
        f"digest-{int(mismatched_adapter_digest)}-tamper-{int(tampered_crop)}"
    )
    payload_by_sample = {
        "sample-1": b"\x89PNG\r\n\x1a\nfirst",
        "sample-2": b"\x89PNG\r\n\x1a\nsecond",
    }
    with AtomicDirectoryPublisher(tmp_path / "exports", publication_id) as publisher:
        for row in materialized_rows:
            payload = payload_by_sample[row["sample_id"]]
            if tampered_crop and row["sample_id"] == "sample-1":
                payload = b"\x89PNG\r\n\x1a\ntampered"
            publisher.write_bytes(
                row["materialized_path"],
                payload,
            )
        publisher.write_bytes(
            "materialized_manifest.csv",
            csv_bytes(manifest_columns, materialized_rows),
        )
        publisher.write_bytes("provenance/adapter_manifest.csv", adapter_bytes)
        publisher.write_bytes("provenance/dataset_release.json", dataset_bytes)
        required = {
            "materialized_manifest.csv",
            "provenance/adapter_manifest.csv",
            "provenance/dataset_release.json",
            *(row["materialized_path"] for row in materialized_rows),
        }
        root = publisher.finalize(
            validator=lambda _staging: None,
            required_paths=frozenset(required),
        )
    return (
        root,
        sha256_file(root / "provenance/dataset_release.json"),
        sha256_file(root / "materialized_manifest.csv"),
    )


def _verify(adapter: str, root: Path, dataset_digest: str, manifest_digest: str) -> None:
    if adapter == "template":
        opencv_backend._verify_materialized_export(root, dataset_digest)
        return
    spec = TrainSpec(
        family="patchcore",
        slot=ModelSlot("right", "front"),
        dataset_release_id="dataset-v1",
        dataset_manifest_digest=dataset_digest,
        train_split_id="train-v1",
        recipe_digest="4" * 64,
        roi_version="roi-v1",
        roi_digest="5" * 64,
        materialized_export_root=root,
        materialized_manifest_digest=manifest_digest,
        device=DeviceSpec(),
        output_dir=root.parent / "candidate",
        parameters={},
        execution_receipt=execution_receipt_mapping("train_anomaly"),
    )
    anomalib_backend._verify_training_partition(spec)


@pytest.mark.parametrize("adapter", ("anomalib", "template"))
def test_materialized_export_accepts_exact_complete_adapter_projection(
    tmp_path: Path,
    adapter: str,
) -> None:
    root, dataset_digest, manifest_digest = _materialized_publication(
        tmp_path,
        adapter=adapter,
    )
    _verify(adapter, root, dataset_digest, manifest_digest)


@pytest.mark.parametrize("adapter", ("anomalib", "template"))
@pytest.mark.parametrize(
    ("adapter_indices", "materialized_indices"),
    (
        ((0,), (0, 1)),
        ((0, 1), (0,)),
    ),
    ids=("adapter-row-deleted", "materialized-subset"),
)
def test_materialized_export_rejects_any_adapter_projection_subset(
    tmp_path: Path,
    adapter: str,
    adapter_indices: tuple[int, ...],
    materialized_indices: tuple[int, ...],
) -> None:
    root, dataset_digest, manifest_digest = _materialized_publication(
        tmp_path,
        adapter=adapter,
        adapter_indices=adapter_indices,
        materialized_indices=materialized_indices,
    )
    with pytest.raises(ModelContractError, match="complete canonical adapter export"):
        _verify(adapter, root, dataset_digest, manifest_digest)


@pytest.mark.parametrize("adapter", ("anomalib", "template"))
def test_materialized_export_rejects_adapter_manifest_digest_mismatch(
    tmp_path: Path,
    adapter: str,
) -> None:
    root, dataset_digest, manifest_digest = _materialized_publication(
        tmp_path,
        adapter=adapter,
        mismatched_adapter_digest=True,
    )
    with pytest.raises(ModelContractError, match="adapter provenance"):
        _verify(adapter, root, dataset_digest, manifest_digest)


@pytest.mark.parametrize("adapter", ("anomalib", "template"))
def test_materialized_export_rejects_self_consistent_tree_with_tampered_crop(
    tmp_path: Path,
    adapter: str,
) -> None:
    root, dataset_digest, manifest_digest = _materialized_publication(
        tmp_path,
        adapter=adapter,
        tampered_crop=True,
    )
    with pytest.raises(ModelContractError, match="canonical digest|crop hash mismatch"):
        _verify(adapter, root, dataset_digest, manifest_digest)
