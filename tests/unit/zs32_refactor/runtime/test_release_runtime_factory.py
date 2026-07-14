"""Linux-only release roundtrip and runtime metadata reconstruction tests."""

from __future__ import annotations

import hashlib
import json
import stat
import sys
from pathlib import Path

import pytest

from zs32_inspection.domain.contracts import AnomalyFamily
from zs32_inspection.domain.errors import DeploymentContractError
from zs32_inspection.models.base import DeviceSpec, ModelSlot
from zs32_inspection.runtime.publisher import canonical_json_bytes
from zs32_inspection.runtime.release_loader import ReleaseLoadError, load_verified_deployment_release
from zs32_inspection.runtime.runtime_factory import build_loaded_runtime


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="ZS32 verification is Linux-only")


def _make_tree_writable(root: Path) -> None:
    root.chmod(0o750)
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            path.chmod(0o750)
        elif stat.S_ISREG(mode):
            path.chmod(0o640)


def _reindex_release(root: Path) -> None:
    """Rebuild publication hashes after an intentional semantic mutation."""
    entries: list[tuple[str, str]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in {"checksums.sha256", "publication_root.json"}:
            continue
        entries.append((relative, hashlib.sha256(path.read_bytes()).hexdigest()))
    checksum_bytes = "".join(
        f"{digest}  {relative}\n" for relative, digest in sorted(entries)
    ).encode("utf-8")
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


def _reindex_nested_publication(root: Path) -> tuple[str, str]:
    """Rebuild a nested atomic publication after a deliberate semantic forgery."""
    entries = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.iterdir()
        if path.is_file() and path.name not in {"checksums.sha256", "publication_root.json"}
    }
    checksum_bytes = "".join(
        f"{digest}  {name}\n" for name, digest in sorted(entries.items())
    ).encode("utf-8")
    root_sha256 = hashlib.sha256(checksum_bytes).hexdigest()
    (root / "checksums.sha256").write_bytes(checksum_bytes)
    (root / "publication_root.json").write_bytes(
        canonical_json_bytes(
            {
                "algorithm": "sha256(checksums.sha256 bytes)",
                "publication_id": root.name,
                "root_sha256": root_sha256,
            }
        )
    )
    return root_sha256, entries["validation.json"]


def test_assembler_loader_roundtrip_binds_all_provenance(assembled_release: Path) -> None:
    release = load_verified_deployment_release(assembled_release)
    candidate = release.files.read_json("provenance/model_candidate.json")
    dataset = release.files.read_json("provenance/dataset_release.json")

    assert release.manifest.release_id == assembled_release.name
    assert release.manifest.contract_sha256 == release.contract.contract_sha256
    assert release.runtime_environment.platform == "linux"
    assert release.runtime_environment.device_mapping.gpu_uuid == "GPU-runtime-fixture"
    assert release.runtime_environment_file_sha256 == release.files.checksums[
        "provenance/runtime_environment.json"
    ]
    assert candidate["candidate_status"] == "validated"
    assert candidate["promotion_id"] == "manual-promotion-runtime-v1"
    assert candidate["promotion_approver"] == "runtime-fixture-approver"
    assert candidate["promotion_receipt_sha256"] == release.files.checksums[
        "provenance/promotion_receipt.json"
    ]
    assert release.promotion_receipt.promotion_id == candidate["promotion_id"]
    assert release.promotion_receipt.candidate_digest == candidate["candidate_digest"]
    assert release.promotion_receipt_file_sha256 == candidate[
        "promotion_receipt_sha256"
    ]
    assert candidate["validation_publication_id"] == "validation-runtime-v1"
    validation_root = (
        assembled_release
        / "provenance/validations"
        / candidate["validation_publication_id"]
    )
    assert candidate["validation_publication_root_sha256"] == hashlib.sha256(
        (validation_root / "checksums.sha256").read_bytes()
    ).hexdigest()
    assert candidate["validation_record_sha256"] == hashlib.sha256(
        (validation_root / "validation.json").read_bytes()
    ).hexdigest()
    assert candidate["heldout_acceptance_policy_sha256"] == hashlib.sha256(
        (validation_root / "heldout_acceptance_policy.json").read_bytes()
    ).hexdigest()
    assert candidate["heldout_acceptance_decision_sha256"] == hashlib.sha256(
        (validation_root / "heldout_acceptance.json").read_bytes()
    ).hexdigest()
    assert release.promotion_receipt.heldout_acceptance_policy_sha256 == candidate[
        "heldout_acceptance_policy_sha256"
    ]
    assert release.promotion_receipt.heldout_acceptance_decision_sha256 == candidate[
        "heldout_acceptance_decision_sha256"
    ]
    assert candidate["dataset_release_id"] == dataset["dataset_release_id"]
    assert candidate["dataset_manifest_sha256"] == release.files.checksums[
        "provenance/dataset_release.json"
    ]
    assert candidate["recipe_sha256"] == release.contract.recipe_sha256
    assert candidate["topology_sha256"] == release.contract.topology.topology_sha256
    assert candidate["roi_sha256"] == release.contract.roi.roi_sha256
    assert candidate["roi_version"] == release.contract.roi.roi_config_id
    assert candidate["anomaly_family"] == release.contract.anomaly_family.value
    assert candidate["yolo_model_sha256"] == release.contract.yolo_model.sha256
    assert candidate["calibration_artifact_sha256"] == release.files.checksums[
        "calibration/calibration_artifact.json"
    ]
    assert set(dataset["adapter_manifest_sha256"]) == {"yolo", "anomalib", "template"}
    assert release.files.checksums["provenance/dataset_canonical_semantics.json"] == (
        dataset["canonical_semantics_sha256"]
    )
    assert release.files.checksums["provenance/dataset_calibration_targets.json"] == (
        dataset["calibration_targets_sha256"]
    )
    assert release.files.checksums["provenance/dataset_governance.json"] == (
        dataset["dataset_provenance_sha256"]
    )
    assert release.files.checksums["provenance/dataset_split_assignments.json"] == (
        dataset["split_assignments_sha256"]
    )


def test_loader_rejects_reindexed_runtime_environment_forgery(
    assembled_release: Path,
) -> None:
    _make_tree_writable(assembled_release)
    path = assembled_release / "provenance/runtime_environment.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["nvidia_driver_version"] = "forged-driver"
    path.write_bytes(canonical_json_bytes(payload))
    _reindex_release(assembled_release)

    with pytest.raises(ReleaseLoadError, match="runtime environment receipt"):
        load_verified_deployment_release(assembled_release)


def test_loader_rejects_missing_promotion_receipt_provenance(
    assembled_release: Path,
) -> None:
    _make_tree_writable(assembled_release)
    path = assembled_release / "provenance/model_candidate.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["promotion_receipt_sha256"]
    path.write_bytes(canonical_json_bytes(payload))
    _reindex_release(assembled_release)

    with pytest.raises(ReleaseLoadError, match="candidate provenance envelope is malformed"):
        load_verified_deployment_release(assembled_release)


def test_loader_rejects_reindexed_promotion_receipt_identity_tamper(
    assembled_release: Path,
) -> None:
    _make_tree_writable(assembled_release)
    path = assembled_release / "provenance/promotion_receipt.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["candidate_digest"] = "0" * 64
    path.write_bytes(canonical_json_bytes(payload))
    _reindex_release(assembled_release)

    with pytest.raises(ReleaseLoadError, match="candidate/promotion receipt provenance"):
        load_verified_deployment_release(assembled_release)


def test_loader_rejects_release_timestamp_before_promotion_approval(
    assembled_release: Path,
) -> None:
    _make_tree_writable(assembled_release)
    path = assembled_release / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["created_at"] = "2026-07-13T23:58:00Z"
    path.write_bytes(canonical_json_bytes(payload))
    _reindex_release(assembled_release)

    with pytest.raises(ReleaseLoadError, match="approval is later"):
        load_verified_deployment_release(assembled_release)


def test_loader_rejects_reindexed_outer_release_extra_file(
    assembled_release: Path,
) -> None:
    _make_tree_writable(assembled_release)
    (assembled_release / "rogue.bin").write_bytes(b"not authorized by the release contract")
    _reindex_release(assembled_release)

    with pytest.raises(ReleaseLoadError, match="contract-derived allowlist"):
        load_verified_deployment_release(assembled_release)


def test_loader_rejects_dataset_roi_provenance_changed_with_valid_tree_hashes(
    assembled_release: Path,
) -> None:
    _make_tree_writable(assembled_release)
    dataset_path = assembled_release / "provenance/dataset_release.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset["roi_sha256"] = "0" * 64
    dataset_path.write_bytes(canonical_json_bytes(dataset))
    dataset_digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    candidate_path = assembled_release / "provenance/model_candidate.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["dataset_manifest_sha256"] = dataset_digest
    candidate_path.write_bytes(canonical_json_bytes(candidate))
    _reindex_release(assembled_release)

    with pytest.raises(ReleaseLoadError, match="dataset provenance differs"):
        load_verified_deployment_release(assembled_release)


def test_loader_rejects_reindexed_dataset_semantics_approval_tamper(
    assembled_release: Path,
) -> None:
    _make_tree_writable(assembled_release)
    path = assembled_release / "provenance/dataset_canonical_semantics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["approval"]["reviewed_by"] = "forged-reviewer"
    path.write_bytes(canonical_json_bytes(payload))
    _reindex_release(assembled_release)

    with pytest.raises(ReleaseLoadError, match="dataset governance digest mismatch"):
        load_verified_deployment_release(assembled_release)


@pytest.mark.parametrize("branch", ("template", "anomaly"))
def test_loader_rejects_model_trained_against_another_split_assignment(
    assembled_release: Path,
    branch: str,
) -> None:
    """A reindexed release cannot relabel which immutable train split built a model."""
    _make_tree_writable(assembled_release)
    metadata_paths = sorted((assembled_release / "models" / branch).rglob("metadata.json"))
    assert metadata_paths
    path = metadata_paths[0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["train_split_id"] = "0" * 64
    path.write_bytes(canonical_json_bytes(payload))
    _reindex_release(assembled_release)

    with pytest.raises(ReleaseLoadError, match="metadata provenance mismatch"):
        load_verified_deployment_release(assembled_release)


def test_loader_rejects_reindexed_calibration_input_provenance_tamper(
    assembled_release: Path,
) -> None:
    _make_tree_writable(assembled_release)
    path = assembled_release / "calibration/input_provenance.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["candidate_id"] = "another-candidate"
    path.write_bytes(canonical_json_bytes(payload))
    _reindex_release(assembled_release)

    with pytest.raises(ReleaseLoadError, match="calibration input provenance"):
        load_verified_deployment_release(assembled_release)


def test_loader_rejects_fully_reindexed_automatic_validation_promotion(
    assembled_release: Path,
) -> None:
    """Recomputing inner and outer hashes cannot turn manual validation into auto-promotion."""
    _make_tree_writable(assembled_release)
    candidate_path = assembled_release / "provenance/model_candidate.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    validation_root = (
        assembled_release
        / "provenance/validations"
        / candidate["validation_publication_id"]
    )
    validation_path = validation_root / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["manual_promotion_required"] = False
    validation_path.write_bytes(canonical_json_bytes(validation))
    validation_root_sha256, validation_record_sha256 = _reindex_nested_publication(
        validation_root
    )
    candidate["validation_publication_root_sha256"] = validation_root_sha256
    candidate["validation_record_sha256"] = validation_record_sha256
    candidate_path.write_bytes(canonical_json_bytes(candidate))
    _reindex_release(assembled_release)

    with pytest.raises(ReleaseLoadError, match="validation record differs"):
        load_verified_deployment_release(assembled_release)


def test_loader_reconstructs_calibration_root_after_full_validation_reindex(
    assembled_release: Path,
) -> None:
    """Outer and nested reindexing cannot forge the original calibration root assertion."""
    _make_tree_writable(assembled_release)
    candidate_path = assembled_release / "provenance/model_candidate.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    validation_root = (
        assembled_release
        / "provenance/validations"
        / candidate["validation_publication_id"]
    )
    validation_path = validation_root / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["calibration_publication_root_sha256"] = "f" * 64
    validation_path.write_bytes(canonical_json_bytes(validation))
    validation_root_sha256, validation_record_sha256 = _reindex_nested_publication(
        validation_root
    )
    candidate["validation_publication_root_sha256"] = validation_root_sha256
    candidate["validation_record_sha256"] = validation_record_sha256
    candidate_path.write_bytes(canonical_json_bytes(candidate))
    _reindex_release(assembled_release)

    with pytest.raises(ReleaseLoadError, match="calibration publication root"):
        load_verified_deployment_release(assembled_release)


def test_loader_rejects_reindexed_acceptance_policy_not_bound_by_promotion(
    assembled_release: Path,
) -> None:
    """Reindexing release bytes cannot change the policy approved by the receipt."""
    _make_tree_writable(assembled_release)
    candidate_path = assembled_release / "provenance/model_candidate.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    validation_root = (
        assembled_release
        / "provenance/validations"
        / candidate["validation_publication_id"]
    )
    policy_path = validation_root / "heldout_acceptance_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["policy_version"] = "forged-2.0.0"
    policy_path.write_bytes(canonical_json_bytes(policy))
    validation_root_sha256, validation_record_sha256 = _reindex_nested_publication(
        validation_root
    )
    candidate["validation_publication_root_sha256"] = validation_root_sha256
    candidate["validation_record_sha256"] = validation_record_sha256
    candidate["heldout_acceptance_policy_sha256"] = hashlib.sha256(
        policy_path.read_bytes()
    ).hexdigest()
    candidate_path.write_bytes(canonical_json_bytes(candidate))
    _reindex_release(assembled_release)

    with pytest.raises(ReleaseLoadError, match="promotion receipt provenance"):
        load_verified_deployment_release(assembled_release)


def test_runtime_factory_reconstructs_release_metadata_before_backend_load(
    assembled_release: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = load_verified_deployment_release(assembled_release)
    captured: dict[str, object] = {}
    anomaly_predictor = object()
    yolo_predictor = object()

    class FakeAnomalyAdapter:
        def __init__(self, *, runtime_loader: object) -> None:
            captured["anomaly_loader"] = runtime_loader

        def load(self, artifacts: object, device: object) -> object:
            captured["anomaly"] = artifacts
            captured["anomaly_device"] = device
            return anomaly_predictor

    class FakeYoloAdapter:
        def __init__(self, runtime_loader: object) -> None:
            captured["yolo_loader"] = runtime_loader

        def load(self, spec: object, device: object) -> object:
            captured["yolo"] = spec
            captured["yolo_device"] = device
            return yolo_predictor

    class FakeTemplatePredictor:
        def __init__(self, assets: object) -> None:
            captured["templates"] = assets

        def score_batch(self, samples: object, *, inspection_id: str) -> tuple[object, ...]:
            return ()

    monkeypatch.setattr(
        "zs32_inspection.runtime.runtime_factory.PatchCoreAdapter",
        FakeAnomalyAdapter,
    )
    monkeypatch.setattr(
        "zs32_inspection.runtime.runtime_factory.UltralyticsYoloAdapter",
        FakeYoloAdapter,
    )
    monkeypatch.setattr(
        "zs32_inspection.runtime.runtime_factory.OpenCvTemplatePredictor",
        FakeTemplatePredictor,
    )
    device = DeviceSpec(accelerator="gpu", device="0")

    runtime = build_loaded_runtime(release, device=device)
    assert captured == {}
    template_predictor = runtime.load_template_predictor()
    assert isinstance(template_predictor, FakeTemplatePredictor)
    assert runtime.load_anomaly_predictor() is anomaly_predictor
    assert runtime.load_yolo_predictor() is yolo_predictor

    expected_slots = {
        ModelSlot(hand.value, view)
        for hand in release.contract.allowed_hands
        for view in release.contract.topology.required_views
    }
    templates = captured["templates"]
    anomaly = captured["anomaly"]
    yolo = captured["yolo"]
    assert set(templates) == expected_slots
    assert set(anomaly) == expected_slots
    assert all(
        item.dataset_manifest_digest
        == release.files.checksums["provenance/dataset_release.json"]
        for item in templates.values()
    )
    assert all(item.recipe_digest == release.contract.recipe_sha256 for item in anomaly.values())
    assert yolo.provenance.dataset_release_id == "dataset-zs32-runtime-v1"
    assert yolo.provenance.dataset_manifest_digest == release.files.checksums[
        "provenance/dataset_release.json"
    ]
    assert yolo.provenance.training_seed == 43
    assert yolo.provenance.run_id == "runtime-run-1"
    assert yolo.provenance.yolo_export_publication_id == "yolo-export-runtime-v1"
    assert (
        yolo.provenance.training_receipt_digest
        == release.files.checksums["models/yolo/training_receipt.json"]
    )
    assert yolo.runtime.single_class_name == "defect"
    assert runtime.release_id == release.manifest.release_id
    assert runtime.release_root_sha256 == release.files.root_sha256
    assert set(runtime.template_slot_contracts) == {
        (hand, view)
        for hand in release.contract.allowed_hands
        for view in release.contract.topology.required_views
    }
    assert all(
        slot.references
        and all(sha256_file(reference.path) == reference.sha256 for reference in slot.references)
        for slot in runtime.template_slot_contracts.values()
    )


def test_runtime_factory_rehashes_metadata_at_use_boundary(
    assembled_release: Path,
) -> None:
    release = load_verified_deployment_release(assembled_release)
    metadata = release.files.path("models/yolo/metadata.json")
    metadata.chmod(0o640)
    metadata.write_bytes(b"{}\n")

    runtime = build_loaded_runtime(release, device=DeviceSpec(accelerator="gpu", device="0"))
    with pytest.raises(ReleaseLoadError, match="changed after verification"):
        runtime.load_yolo_predictor()


def test_pending_left_roi_cannot_compile_a_release_contract(
    compiled_contract_factory: object,
) -> None:
    with pytest.raises(DeploymentContractError, match="ROI.*pending"):
        compiled_contract_factory(left_ready=False, hands=("left",))


def test_stable_recipe_identity_breaks_calibration_hash_cycle_but_contract_covers_it(
    compiled_contract_factory: object,
) -> None:
    first = compiled_contract_factory(calibration_digest="4" * 64)
    second = compiled_contract_factory(calibration_digest="5" * 64)

    assert first.recipe_sha256 == second.recipe_sha256
    assert first.contract_sha256 != second.contract_sha256
    assert {item.calibration_sha256 for item in first.template_thresholds} == {"4" * 64}
    assert {item.calibration_sha256 for item in second.model_thresholds} == {"5" * 64}


def test_selected_anomaly_family_is_reconstructed_without_fallback(
    assembled_release: Path,
) -> None:
    release = load_verified_deployment_release(assembled_release)

    assert release.contract.anomaly_family is AnomalyFamily.PATCHCORE
    assert {binding.family for binding in release.contract.anomaly_bindings} == {
        AnomalyFamily.PATCHCORE
    }
