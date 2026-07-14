# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Linux verification cases for the immutable calibration score-run boundary."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from zs32_inspection.calibration import CalibrationProvenance
from zs32_inspection.cli.calibrate import (
    _load_verified_score_run,
    _validate_score_run_provenance,
)
from zs32_inspection.domain.contracts import AnomalyFamily
from zs32_inspection.domain.identity import Hand
from zs32_inspection.models import ModelSlot
from zs32_inspection.runtime.publisher import PublicationError, canonical_json_bytes
from tests.unit.zs32_refactor.execution_fixtures import execution_receipt_mapping

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="ZS32 verification is Linux-only")

DATASET_MANIFEST = "1" * 64
CANONICAL_MANIFEST = "2" * 64
SPLIT_ASSIGNMENTS = "3" * 64
CALIBRATION_TARGETS = "0" * 64
CANDIDATE_DIGEST = "4" * 64
CANDIDATE_DESCRIPTOR = "5" * 64
RECIPE_DIGEST = "6" * 64
TEMPLATE_DESCRIPTOR = "7" * 64
TOPOLOGY_DIGEST = "8" * 64
ROI_DIGEST = "9" * 64
TEMPLATE_DIGEST = "a" * 64
REFERENCE_DIGEST = "b" * 64
ANOMALY_DIGEST = "c" * 64
YOLO_DIGEST = "d" * 64
YOLO_BUNDLE_DIGEST = "e" * 64
PROFILE_DIGEST = "f" * 64


def _rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    scores: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    for index, (branch, digest, score) in enumerate(
        (
            ("template", TEMPLATE_DIGEST, 0.1),
            ("anomaly", ANOMALY_DIGEST, 0.2),
            ("yolo", YOLO_DIGEST, 0.0),
        )
    ):
        row = {
            "dataset_release_id": "dataset-v1",
            "part_instance_id": "part-v1",
            "capture_set_id": "capture-v1",
            "hand": "right",
            "view": "v1",
            "branch": branch,
            "score": score,
            "ground_truth": "normal",
            "part_ground_truth": "normal",
            "split_role": "calibration",
            "split_id": "cal-v1",
            "model_digest": digest,
            "roi_version": "roi-v1",
            "roi_digest": ROI_DIGEST,
        }
        scores.append(row)
        audit.append(
            {
                "sequence_index": index,
                "score_sequence_index": index,
                "included_in_calibration": True,
                "dataset_release_id": row["dataset_release_id"],
                "part_instance_id": row["part_instance_id"],
                "capture_set_id": row["capture_set_id"],
                "hand": row["hand"],
                "view": row["view"],
                "split_role": row["split_role"],
                "split_id": row["split_id"],
                "roi_version": row["roi_version"],
                "roi_digest": row["roi_digest"],
                "branch": branch,
                "raw_score": score,
                "model_digest": digest,
                "part_ground_truth": "normal",
                "calibration_target": "normal",
                "target_reason": None,
                "inspection_id": "score-v1:capture-v1",
                "source_sha256": "1" * 64,
                "crop_sha256": "2" * 64,
                **(
                    {
                        "similarity": 0.9,
                        "best_template_sha256": REFERENCE_DIGEST,
                        "offset_xy": [0, 0],
                    }
                    if branch == "template"
                    else {
                        "model_family": "patchcore",
                        "heatmap_sha256": None,
                    }
                    if branch == "anomaly"
                    else {"detections": []}
                ),
            }
        )
    return scores, audit


def _jsonl(rows: list[dict[str, object]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def _run_payload(scores_bytes: bytes, audit_bytes: bytes) -> dict[str, object]:
    return {
        "schema": "zs32.calibration_score_run",
        "schema_version": 3,
        "score_run_id": "score-v1",
        "dataset_release_id": "dataset-v1",
        "dataset_manifest_sha256": DATASET_MANIFEST,
        "canonical_manifest_sha256": CANONICAL_MANIFEST,
        "calibration_targets_sha256": CALIBRATION_TARGETS,
        "calibration_target_count": 3,
        "split_assignments_sha256": SPLIT_ASSIGNMENTS,
        "calibration_split_id": "cal-v1",
        "test_split_id": "test-v1",
        "candidate_id": "candidate-v1",
        "candidate_digest": CANDIDATE_DIGEST,
        "candidate_descriptor_sha256": CANDIDATE_DESCRIPTOR,
        "candidate_status": "registered",
        "recipe_digest": RECIPE_DIGEST,
        "anomaly_family": "patchcore",
        "template_assets_descriptor_sha256": TEMPLATE_DESCRIPTOR,
        "topology_id": "topology-v1",
        "topology_sha256": TOPOLOGY_DIGEST,
        "roi_version": "roi-v1",
        "roi_sha256": ROI_DIGEST,
        "enabled_hands": ["right"],
        "required_slots": ["right/v1"],
        "model_digests": [TEMPLATE_DIGEST, ANOMALY_DIGEST, YOLO_DIGEST],
        "template_models": [
            {
                "hand": "right",
                "view": "v1",
                "model_digest": TEMPLATE_DIGEST,
                "reference_digests": [REFERENCE_DIGEST],
            }
        ],
        "anomaly_models": [
            {"hand": "right", "view": "v1", "model_digest": ANOMALY_DIGEST}
        ],
        "yolo_model_digest": YOLO_DIGEST,
        "yolo_bundle_digest": YOLO_BUNDLE_DIGEST,
        "device": {"accelerator": "gpu", "device": "0"},
        "branch_order": ["template", "anomaly", "yolo"],
        "canonical_row_count": 1,
        "score_row_count": 3,
        "score_audit_row_count": 3,
        "excluded_score_count": 0,
        "scores_sha256": hashlib.sha256(scores_bytes).hexdigest(),
        "score_audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        "execution_receipt": execution_receipt_mapping(
            "score_calibration",
            input_sha256_by_role={
                "dataset_manifest": DATASET_MANIFEST,
                "canonical_manifest": CANONICAL_MANIFEST,
                "calibration_targets": CALIBRATION_TARGETS,
                "candidate_descriptor": CANDIDATE_DESCRIPTOR,
                "template_assets_descriptor": TEMPLATE_DESCRIPTOR,
            },
            parameters_sha256=hashlib.sha256(
                canonical_json_bytes(
                    {
                        "calibration_split_id": "cal-v1",
                        "test_split_id": "test-v1",
                        "device": "0",
                    }
                )
            ).hexdigest(),
        ),
    }


def _write_publication(root: Path, files: dict[str, bytes]) -> None:
    root.mkdir()
    for relative, content in files.items():
        (root / relative).write_bytes(content)
    checksums = {
        relative: hashlib.sha256(content).hexdigest()
        for relative, content in files.items()
    }
    checksum_bytes = (
        "\n".join(f"{digest}  {relative}" for relative, digest in sorted(checksums.items()))
        + "\n"
    ).encode()
    (root / "checksums.sha256").write_bytes(checksum_bytes)
    (root / "publication_root.json").write_bytes(
        canonical_json_bytes(
            {
                "algorithm": "sha256(checksums.sha256 bytes)",
                "publication_id": root.name,
                "root_sha256": hashlib.sha256(checksum_bytes).hexdigest(),
            }
        )
    )


def _valid_publication(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    scores, audit = _rows()
    scores_bytes = _jsonl(scores)
    audit_bytes = _jsonl(audit)
    run = _run_payload(scores_bytes, audit_bytes)
    root = tmp_path / "score-v1"
    _write_publication(
        root,
        {
            "scores.jsonl": scores_bytes,
            "score_audit.jsonl": audit_bytes,
            "score_run.json": canonical_json_bytes(run),
        },
    )
    return root, run


def test_score_run_requires_an_exact_atomic_publication(tmp_path: Path) -> None:
    root, run = _valid_publication(tmp_path)
    publication, loaded, rows = _load_verified_score_run(root)
    assert publication.publication_id == "score-v1"
    assert loaded == run
    assert len(rows) == 3


def test_score_run_rejects_an_indexed_extra_file(tmp_path: Path) -> None:
    root, _run = _valid_publication(tmp_path)
    original = {
        name: (root / name).read_bytes()
        for name in ("scores.jsonl", "score_audit.jsonl", "score_run.json")
    }
    for path in root.iterdir():
        path.unlink()
    root.rmdir()
    _write_publication(root, {**original, "unexpected.json": b"{}\n"})
    with pytest.raises(PublicationError, match="unexpected indexed files"):
        _load_verified_score_run(root)


def test_score_run_rejects_a_false_publication_root_digest(tmp_path: Path) -> None:
    root, _run = _valid_publication(tmp_path)
    (root / "publication_root.json").write_bytes(
        canonical_json_bytes(
            {
                "algorithm": "sha256(checksums.sha256 bytes)",
                "publication_id": "score-v1",
                "root_sha256": "0" * 64,
            }
        )
    )
    with pytest.raises(PublicationError, match="root identity or digest mismatch"):
        _load_verified_score_run(root)


def test_score_run_rejects_scores_resealed_without_updating_run_digest(tmp_path: Path) -> None:
    root, run = _valid_publication(tmp_path)
    scores, audit = _rows()
    scores[0]["score"] = 0.9
    for path in root.iterdir():
        path.unlink()
    root.rmdir()
    _write_publication(
        root,
        {
            "scores.jsonl": _jsonl(scores),
            "score_audit.jsonl": _jsonl(audit),
            "score_run.json": canonical_json_bytes(run),
        },
    )
    with pytest.raises(ValueError, match="scores.jsonl digest differs"):
        _load_verified_score_run(root)


def test_score_run_rejects_noncanonical_manifest_even_when_resealed(tmp_path: Path) -> None:
    root, run = _valid_publication(tmp_path)
    scores, audit = _rows()
    for path in root.iterdir():
        path.unlink()
    root.rmdir()
    _write_publication(
        root,
        {
            "scores.jsonl": _jsonl(scores),
            "score_audit.jsonl": _jsonl(audit),
            "score_run.json": (json.dumps(run, indent=2) + "\n").encode(),
        },
    )
    with pytest.raises(ValueError, match="canonical JSON serialization"):
        _load_verified_score_run(root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing", "keys invalid"),
        ("extra", "keys invalid"),
        ("template_relation", "template score relation"),
        ("inspection_identity", "inspection identity"),
    ),
)
def test_score_run_rejects_incomplete_or_polluted_branch_audit_when_resealed(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    scores, audit = _rows()
    if mutation == "missing":
        del audit[0]["best_template_sha256"]
    elif mutation == "extra":
        audit[1]["unbound_debug_value"] = "not allowed"
    elif mutation == "inspection_identity":
        for row in audit:
            row["inspection_id"] = "forged-inspection"
    else:
        audit[0]["similarity"] = 0.5
    scores_bytes = _jsonl(scores)
    audit_bytes = _jsonl(audit)
    run = _run_payload(scores_bytes, audit_bytes)
    root = tmp_path / "score-v1"
    _write_publication(
        root,
        {
            "scores.jsonl": scores_bytes,
            "score_audit.jsonl": audit_bytes,
            "score_run.json": canonical_json_bytes(run),
        },
    )
    with pytest.raises(ValueError, match=message):
        _load_verified_score_run(root)


def _semantic_inputs():
    slot = ModelSlot("right", "v1")
    recipe = SimpleNamespace(
        recipe_sha256=RECIPE_DIGEST,
        anomaly_family=AnomalyFamily.PATCHCORE,
        allowed_hands=(Hand.RIGHT,),
        template_bindings=(
            SimpleNamespace(hand=Hand.RIGHT, view_id="v1", asset=SimpleNamespace(sha256=TEMPLATE_DIGEST)),
        ),
        anomaly_bindings=(
            SimpleNamespace(hand=Hand.RIGHT, view_id="v1", asset=SimpleNamespace(sha256=ANOMALY_DIGEST)),
        ),
        yolo_model=SimpleNamespace(sha256=YOLO_DIGEST),
    )
    topology = SimpleNamespace(
        topology_id="topology-v1",
        topology_sha256=TOPOLOGY_DIGEST,
        required_views=("v1",),
    )
    roi = SimpleNamespace(roi_config_id="roi-v1", roi_sha256=ROI_DIGEST)
    dataset = SimpleNamespace(
        dataset_release_id="dataset-v1",
        canonical_manifest_sha256=CANONICAL_MANIFEST,
        calibration_targets_sha256=CALIBRATION_TARGETS,
        calibration_target_count=3,
        split_assignments_sha256=SPLIT_ASSIGNMENTS,
    )
    candidate = SimpleNamespace(
        candidate_id="candidate-v1",
        digest=CANDIDATE_DIGEST,
        status=SimpleNamespace(value="registered"),
        recipe_digest=RECIPE_DIGEST,
        topology_digest=TOPOLOGY_DIGEST,
        roi_version="roi-v1",
        roi_digest=ROI_DIGEST,
        dataset_release_id="dataset-v1",
        dataset_manifest_digest=DATASET_MANIFEST,
        anomaly_family=AnomalyFamily.PATCHCORE,
        anomaly_artifacts={slot: SimpleNamespace(model_digest=ANOMALY_DIGEST)},
        yolo=SimpleNamespace(model_digest=YOLO_DIGEST, bundle_digest=YOLO_BUNDLE_DIGEST),
        validate_slots=lambda required: None,
    )
    provenance = CalibrationProvenance(
        recipe_digest=RECIPE_DIGEST,
        profile_digest=PROFILE_DIGEST,
        topology_digest=TOPOLOGY_DIGEST,
        roi_digest=ROI_DIGEST,
        dataset_release_id="dataset-v1",
        dataset_manifest_digest=DATASET_MANIFEST,
        calibration_split_id="cal-v1",
        test_split_id="test-v1",
        model_digests=(TEMPLATE_DIGEST, ANOMALY_DIGEST, YOLO_DIGEST),
    )
    return recipe, topology, roi, dataset, candidate, provenance


def test_score_run_provenance_binds_dataset_recipe_candidate_and_splits() -> None:
    scores, audit = _rows()
    run = _run_payload(_jsonl(scores), _jsonl(audit))
    recipe, topology, roi, dataset, candidate, provenance = _semantic_inputs()
    _validate_score_run_provenance(
        run,
        recipe=recipe,
        topology=topology,
        roi=roi,
        dataset=dataset,
        dataset_manifest_digest=DATASET_MANIFEST,
        candidate=candidate,
        candidate_descriptor_digest=CANDIDATE_DESCRIPTOR,
        provenance=provenance,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("dataset_release_id", "dataset-other", "dataset provenance"),
        ("candidate_digest", "0" * 64, "candidate identity"),
        ("calibration_split_id", "cal-other", "split IDs"),
        ("recipe_digest", "0" * 64, "recipe/topology/ROI"),
    ),
)
def test_score_run_provenance_rejects_cross_input_reuse(field: str, value: str, message: str) -> None:
    scores, audit = _rows()
    run = _run_payload(_jsonl(scores), _jsonl(audit))
    run[field] = value
    recipe, topology, roi, dataset, candidate, provenance = _semantic_inputs()
    with pytest.raises(ValueError, match=message):
        _validate_score_run_provenance(
            run,
            recipe=recipe,
            topology=topology,
            roi=roi,
            dataset=dataset,
            dataset_manifest_digest=DATASET_MANIFEST,
            candidate=candidate,
            candidate_descriptor_digest=CANDIDATE_DESCRIPTOR,
            provenance=provenance,
        )
