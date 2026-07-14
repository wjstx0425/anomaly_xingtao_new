"""DeploymentContract completeness and fail-closed compiler tests."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from zs32_inspection.config.compiler import compile_deployment_contract
from zs32_inspection.config.schemas import parse_deployment_contract, parse_recipe, parse_roi_config, parse_topology
from zs32_inspection.domain.contracts import DeploymentContract
from zs32_inspection.domain.errors import DeploymentContractError
from zs32_inspection.domain.evidence import (
    EvidenceBranch,
    EvidenceLevel,
    ModelEvidence,
    TemplateEvidence,
    TemplateOutcome,
)
from zs32_inspection.domain.identity import Hand

from ._fixtures import DIGESTS, FUSION_POLICY_BYTES, recipe_mapping, roi_mapping, topology_mapping


def _compile(
    camera_count: int = 3,
    *,
    left_ready: bool = False,
    hands: tuple[str, ...] = ("right",),
) -> DeploymentContract:
    topology = parse_topology(topology_mapping(camera_count))
    roi = parse_roi_config(roi_mapping(topology, left_ready=left_ready))
    recipe = parse_recipe(recipe_mapping(topology, roi, hands=hands))
    return compile_deployment_contract(topology, roi, recipe, fusion_policy_bytes=FUSION_POLICY_BYTES)


@pytest.mark.parametrize(("camera_count", "expected_groups"), [(3, 12), (4, 16), (5, 20)])
def test_compiler_generates_every_required_second_layer_group(camera_count: int, expected_groups: int) -> None:
    """One enabled hand gets views x anomaly/YOLO with no fixed six-view constant."""
    contract = _compile(camera_count)

    assert len(contract.groups_for_hand(Hand.RIGHT)) == expected_groups
    assert len(contract.template_bindings) == expected_groups // 2
    assert len(contract.anomaly_bindings) == expected_groups // 2
    assert len(contract.contract_sha256) == 64


def test_left_pending_blocks_any_recipe_enabling_left() -> None:
    """Right-hand coordinates are never mirrored or reused for left deployment."""
    topology = parse_topology(topology_mapping(3))
    roi = parse_roi_config(roi_mapping(topology, left_ready=False))
    recipe = parse_recipe(recipe_mapping(topology, roi, hands=("left",)))

    with pytest.raises(DeploymentContractError, match="ROI.*pending"):
        compile_deployment_contract(topology, roi, recipe, fusion_policy_bytes=FUSION_POLICY_BYTES)


def test_ready_hand_missing_one_roi_blocks_contract() -> None:
    """A ready marker cannot hide a missing required view coordinate."""
    topology = parse_topology(topology_mapping(3))
    roi_payload = roi_mapping(topology)
    del roi_payload["hands"]["right"]["views"][topology.required_views[-1]]
    roi = parse_roi_config(roi_payload)
    recipe = parse_recipe(recipe_mapping(topology, roi))

    with pytest.raises(DeploymentContractError, match="ROI view set mismatch"):
        compile_deployment_contract(topology, roi, recipe, fusion_policy_bytes=FUSION_POLICY_BYTES)


def test_both_ready_hands_have_independent_complete_groups() -> None:
    """A two-hand release has 12 groups per hand, not one shared identity set."""
    contract = _compile(3, left_ready=True, hands=("right", "left"))

    assert len(contract.groups_for_hand(Hand.RIGHT)) == 12
    assert len(contract.groups_for_hand(Hand.LEFT)) == 12
    assert len(contract.required_evidence_groups) == 24


def test_missing_model_threshold_blocks_contract() -> None:
    """No required anomaly/YOLO group can be defaulted from another view."""
    topology = parse_topology(topology_mapping(3))
    roi = parse_roi_config(roi_mapping(topology))
    payload = recipe_mapping(topology, roi)
    payload["model_thresholds"].pop()
    recipe = parse_recipe(payload)

    with pytest.raises(DeploymentContractError, match="model thresholds identity set mismatch"):
        compile_deployment_contract(topology, roi, recipe, fusion_policy_bytes=FUSION_POLICY_BYTES)


def test_mixed_anomaly_family_blocks_contract() -> None:
    """A deployment cannot mix PatchCore and EfficientAD by view."""
    topology = parse_topology(topology_mapping(3))
    roi = parse_roi_config(roi_mapping(topology))
    payload = recipe_mapping(topology, roi)
    first_view = topology.required_views[0]
    payload["anomaly_models"]["right"][first_view]["family"] = "efficientad"
    payload["anomaly_models"]["right"][first_view]["artifact"]["relative_path"] = (
        f"models/anomaly/efficientad/right/{first_view}/model.ckpt"
    )
    recipe = parse_recipe(payload)

    with pytest.raises(DeploymentContractError, match="family mixing is forbidden"):
        compile_deployment_contract(topology, roi, recipe, fusion_policy_bytes=FUSION_POLICY_BYTES)


def test_yolo_initialization_weight_cannot_enter_release() -> None:
    """The global deployed YOLO artifact must be best.pt, never yolo26n.pt."""
    topology = parse_topology(topology_mapping(3))
    roi = parse_roi_config(roi_mapping(topology))
    payload = recipe_mapping(topology, roi)
    payload["yolo_model"]["relative_path"] = "models/yolo/yolo26n.pt"
    recipe = parse_recipe(payload)

    with pytest.raises(DeploymentContractError, match="best.pt"):
        compile_deployment_contract(topology, roi, recipe, fusion_policy_bytes=FUSION_POLICY_BYTES)


def test_template_evidence_must_use_contract_calibration_digest() -> None:
    """Template score/threshold evidence is content-bound to the release calibration."""
    contract = _compile()
    view_id = contract.topology.required_views[0]
    evidence = TemplateEvidence(
        inspection_id="inspection-1",
        capture_set_id="capture-1",
        part_instance_id="part-1",
        hand=Hand.RIGHT,
        view_id=view_id,
        outcome=TemplateOutcome.PASS,
        score=0.1,
        threshold=0.5,
        template_sha256=DIGESTS["template"],
        threshold_sha256=DIGESTS["template_calibration"],
        source_sha256="7" * 64,
        crop_sha256="8" * 64,
        roi_config_id=contract.roi.roi_config_id,
    )

    contract.validate_template_evidence(evidence)
    wrong = replace(evidence, threshold_sha256="9" * 64)
    with pytest.raises(DeploymentContractError, match="threshold outside"):
        contract.validate_template_evidence(wrong)


def test_model_evidence_must_use_contract_model_threshold_and_level() -> None:
    """A model row cannot self-report a hash or level outside the release contract."""
    contract = _compile()
    view_id = contract.topology.required_views[0]
    evidence = ModelEvidence(
        inspection_id="inspection-1",
        capture_set_id="capture-1",
        part_instance_id="part-1",
        hand=Hand.RIGHT,
        view_id=view_id,
        branch=EvidenceBranch.ANOMALY,
        level=EvidenceLevel.CLEAR,
        score=0.1,
        model_family="patchcore",
        model_sha256=DIGESTS["anomaly"],
        roi_config_id=contract.roi.roi_config_id,
        source_sha256="7" * 64,
        crop_sha256="8" * 64,
        threshold_sha256=DIGESTS["model_calibration"],
    )

    contract.validate_model_evidence(evidence)
    with pytest.raises(DeploymentContractError, match="contradicts"):
        contract.validate_model_evidence(replace(evidence, level=EvidenceLevel.STRONG))


def test_contract_hash_is_deterministic_and_canonical_roi_schema_is_v2() -> None:
    """Equivalent compiled payloads share one hash and serialize the sole ROI schema."""
    first = _compile()
    second = _compile()

    assert first.contract_sha256 == second.contract_sha256
    serialized_roi = first.as_dict()["roi"]
    assert serialized_roi["roi_version"] == first.roi.roi_config_id
    assert "source_image_size" in serialized_roi
    assert "reason" not in serialized_roi["hands"]["right"]
    assert serialized_roi["hands"]["left"]["reason"] == "left ROI is not approved"
    assert "roi_config_id" not in serialized_roi
    assert "source_image" not in serialized_roi


def test_compiler_hashes_actual_fusion_policy_bytes() -> None:
    """A recipe self-report cannot substitute for the actual strict policy artifact."""
    topology = parse_topology(topology_mapping(3))
    roi = parse_roi_config(roi_mapping(topology))
    recipe = parse_recipe(recipe_mapping(topology, roi))

    with pytest.raises(DeploymentContractError, match="policy artifact bytes"):
        compile_deployment_contract(topology, roi, recipe, fusion_policy_bytes=b"tampered-policy")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sha256", "0" * 64),
        ("artifact_id", "unbound-zs32-capture-gate-policy"),
        ("version", "training-profile"),
    ],
)
def test_unbound_capture_gate_policy_can_never_compile(field: str, value: str) -> None:
    """The checked-in recipe template is not a runnable provenance identity."""
    topology = parse_topology(topology_mapping(3))
    roi = parse_roi_config(roi_mapping(topology))
    payload = recipe_mapping(topology, roi)
    payload["capture_gate_policy"][field] = value
    recipe = parse_recipe(payload)

    with pytest.raises(DeploymentContractError, match="capture_gate_policy is unbound"):
        compile_deployment_contract(
            topology,
            roi,
            recipe,
            fusion_policy_bytes=FUSION_POLICY_BYTES,
        )


def test_serialized_contract_is_rehashed_and_revalidated() -> None:
    """Runtime deserialization never trusts contract_sha256 from the JSON file."""
    contract = _compile()
    payload = contract.as_dict()

    restored = parse_deployment_contract(payload)
    assert restored.contract_sha256 == contract.contract_sha256

    tampered = copy.deepcopy(payload)
    tampered["model_thresholds"][0]["high"] = 0.9
    with pytest.raises(DeploymentContractError, match="contract_sha256"):
        parse_deployment_contract(tampered)


def test_recipe_identity_has_no_calibration_hash_cycle_but_contract_covers_thresholds() -> None:
    """Threshold fitting can bind recipe identity, while final contract still changes."""
    topology = parse_topology(topology_mapping(3))
    roi = parse_roi_config(roi_mapping(topology))
    first_payload = recipe_mapping(topology, roi)
    second_payload = copy.deepcopy(first_payload)
    second_payload["template_thresholds"][0]["threshold"] = 0.61
    second_payload["template_thresholds"][0]["calibration_sha256"] = "e" * 64
    second_payload["model_thresholds"][0]["low"] = 0.31
    second_payload["model_thresholds"][0]["high"] = 0.91
    second_payload["model_thresholds"][0]["calibration_sha256"] = "e" * 64

    first_recipe = parse_recipe(first_payload)
    second_recipe = parse_recipe(second_payload)
    assert first_recipe.recipe_sha256 == second_recipe.recipe_sha256
    assert "template_thresholds" not in first_recipe.identity_dict()
    assert "model_thresholds" not in first_recipe.identity_dict()

    first_contract = compile_deployment_contract(
        topology,
        roi,
        first_recipe,
        fusion_policy_bytes=FUSION_POLICY_BYTES,
    )
    second_contract = compile_deployment_contract(
        topology,
        roi,
        second_recipe,
        fusion_policy_bytes=FUSION_POLICY_BYTES,
    )
    assert first_contract.contract_sha256 != second_contract.contract_sha256


def test_recipe_identity_has_no_model_self_reference_but_contract_covers_assets() -> None:
    """Training can bind the stable profile before generated model hashes exist."""
    topology = parse_topology(topology_mapping(3))
    roi = parse_roi_config(roi_mapping(topology))
    first_payload = recipe_mapping(topology, roi)
    second_payload = copy.deepcopy(first_payload)
    first_view = topology.required_views[0]
    second_payload["template_assets"]["right"][first_view]["sha256"] = "e" * 64
    for threshold in second_payload["template_thresholds"]:
        if threshold["hand"] == "right" and threshold["view"] == first_view:
            threshold["template_sha256"] = "e" * 64

    first_recipe = parse_recipe(first_payload)
    second_recipe = parse_recipe(second_payload)
    assert first_recipe.recipe_sha256 == second_recipe.recipe_sha256
    assert "template_assets" not in first_recipe.identity_dict()
    assert "anomaly_models" not in first_recipe.identity_dict()
    assert "yolo_model" not in first_recipe.identity_dict()

    first_contract = compile_deployment_contract(
        topology,
        roi,
        first_recipe,
        fusion_policy_bytes=FUSION_POLICY_BYTES,
    )
    second_contract = compile_deployment_contract(
        topology,
        roi,
        second_recipe,
        fusion_policy_bytes=FUSION_POLICY_BYTES,
    )
    assert first_contract.contract_sha256 != second_contract.contract_sha256
