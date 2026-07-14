"""Linux-only byte-boundary tests for candidate validation inputs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from zs32_inspection.cli.validate_candidate import (
    _parser,
    _validate_score_run_input_binding,
    _validation_publication_fields,
    _verify_calibration_publication,
)
from zs32_inspection.runtime.publisher import (
    AtomicDirectoryPublisher,
    PublicationError,
    VerifiedAtomicPublication,
)
from tests.unit.zs32_refactor.execution_fixtures import execution_receipt_mapping


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="ZS32 verification is Linux-only")


def test_validation_cli_requires_an_explicit_heldout_acceptance_policy() -> None:
    required = [
        "--candidate",
        "/immutable/registration/candidate.json",
        "--template-assets",
        "/immutable/registration/template_assets.json",
        "--asset-root",
        "/immutable",
        "--recipe",
        "/immutable/recipe.json",
        "--topology",
        "/immutable/topology.json",
        "--roi",
        "/immutable/roi.json",
        "--fusion-policy",
        "/immutable/fusion.json",
        "--dataset-release",
        "/immutable/dataset",
        "--score-run",
        "/immutable/scores",
        "--calibration",
        "/immutable/calibration",
        "--output-root",
        "/immutable/validations",
        "--validation-id",
        "validation-v1",
    ]

    with pytest.raises(SystemExit):
        _parser().parse_args(required)

    parsed = _parser().parse_args(
        [
            *required,
            "--heldout-acceptance-policy",
            "/immutable/heldout-policy.json",
        ]
    )
    assert parsed.heldout_acceptance_policy == Path(
        "/immutable/heldout-policy.json"
    )


def _publication(root: Path) -> Path:
    with AtomicDirectoryPublisher(root, "calibration-v1") as publisher:
        publisher.write_json("calibration_artifact.json", {"calibration_valid": True})
        publisher.write_json("template_thresholds.json", {"thresholds": []})
        publisher.write_json("model_thresholds.json", {"thresholds": []})
        publisher.write_json("metrics.json", {"evaluation_complete": True})
        publisher.write_json("input_provenance.json", {"schema_version": 1})
        return publisher.finalize(
            validator=lambda _staging: None,
            required_paths=frozenset(
                {
                    "calibration_artifact.json",
                    "template_thresholds.json",
                    "model_thresholds.json",
                    "metrics.json",
                    "input_provenance.json",
                }
            ),
        )


def test_calibration_publication_boundary_accepts_complete_atomic_tree(tmp_path: Path) -> None:
    contents = _verify_calibration_publication(_publication(tmp_path))

    assert set(contents) == {
        "calibration_artifact.json",
        "template_thresholds.json",
        "model_thresholds.json",
        "metrics.json",
        "input_provenance.json",
    }


def test_calibration_publication_boundary_rejects_unindexed_file(tmp_path: Path) -> None:
    root = _publication(tmp_path)
    root.chmod(0o750)
    (root / "forged.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(PublicationError, match="file set differs"):
        _verify_calibration_publication(root)


def test_calibration_publication_boundary_rejects_hardlink(tmp_path: Path) -> None:
    root = _publication(tmp_path)
    root.chmod(0o750)
    target = root / "metrics.json"
    target.chmod(0o640)
    os.link(target, root / "metrics-hardlink.json")

    with pytest.raises(PublicationError, match="hardlinked|file set differs"):
        _verify_calibration_publication(root)


def _score_binding() -> tuple[
    VerifiedAtomicPublication,
    dict[str, object],
    dict[str, object],
]:
    publication = VerifiedAtomicPublication(
        root=Path("/immutable/score-v1"),
        publication_id="score-v1",
        root_sha256="1" * 64,
        checksums={},
        owner_uid=os.geteuid(),
    )
    run = {
        "scores_sha256": "2" * 64,
        "score_audit_sha256": "3" * 64,
        "execution_receipt": execution_receipt_mapping("score_calibration"),
        "candidate_id": "candidate-v1",
        "candidate_digest": "4" * 64,
        "candidate_descriptor_sha256": "5" * 64,
        "recipe_digest": "6" * 64,
        "dataset_release_id": "dataset-v1",
        "dataset_manifest_sha256": "7" * 64,
        "calibration_targets_sha256": "8" * 64,
        "calibration_target_count": 30,
        "calibration_split_id": "calibration-v1",
        "test_split_id": "test-v1",
    }
    provenance = {
        "score_run_id": publication.publication_id,
        "score_run_root_sha256": publication.root_sha256,
        **run,
        "score_execution_receipt_sha256": run["execution_receipt"]["receipt_sha256"],
    }
    return publication, run, provenance


def test_validation_binds_the_exact_score_publication_and_semantic_inputs() -> None:
    publication, run, provenance = _score_binding()

    _validate_score_run_input_binding(publication, run, provenance)


@pytest.mark.parametrize(
    "field",
    (
        "score_run_id",
        "score_run_root_sha256",
        "scores_sha256",
        "score_audit_sha256",
        "score_execution_receipt_sha256",
        "candidate_id",
        "candidate_digest",
        "candidate_descriptor_sha256",
        "recipe_digest",
        "dataset_release_id",
        "dataset_manifest_sha256",
        "calibration_targets_sha256",
        "calibration_target_count",
        "calibration_split_id",
        "test_split_id",
    ),
)
def test_validation_rejects_score_run_reuse_across_any_frozen_input(field: str) -> None:
    publication, run, provenance = _score_binding()
    provenance[field] = "different"

    with pytest.raises(ValueError, match="score run publication or candidate/recipe/dataset/split"):
        _validate_score_run_input_binding(publication, run, provenance)


def test_validation_record_names_every_consumed_immutable_publication() -> None:
    score_publication, score_run, _provenance = _score_binding()
    registration = VerifiedAtomicPublication(
        root=Path("/immutable/registration-v1"),
        publication_id="registration-v1",
        root_sha256="8" * 64,
        checksums={},
        owner_uid=os.geteuid(),
    )
    calibration = VerifiedAtomicPublication(
        root=Path("/immutable/calibration-v1"),
        publication_id="calibration-v1",
        root_sha256="9" * 64,
        checksums={},
        owner_uid=os.geteuid(),
    )

    assert _validation_publication_fields(
        registration=registration,
        calibration=calibration,
        score_run=score_publication,
        score_run_manifest=score_run,
    ) == {
        "registration_publication_id": "registration-v1",
        "registration_publication_root_sha256": "8" * 64,
        "calibration_publication_id": "calibration-v1",
        "calibration_publication_root_sha256": "9" * 64,
        "score_run_id": "score-v1",
        "score_run_root_sha256": "1" * 64,
        "scores_sha256": "2" * 64,
        "score_audit_sha256": "3" * 64,
        "score_execution_receipt_sha256": score_run["execution_receipt"]["receipt_sha256"],
    }
