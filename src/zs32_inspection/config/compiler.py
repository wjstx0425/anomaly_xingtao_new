"""Fail-closed compiler from validated configs to DeploymentContract."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Hashable
from pathlib import PurePosixPath
from typing import TypeVar

from zs32_inspection.domain.contracts import (
    AnomalyBinding,
    DeploymentContract,
    InspectionRecipe,
    RequiredEvidenceGroup,
    RoiConfig,
    TemplateBinding,
    TemplateThresholdRecord,
    ThresholdRecord,
)
from zs32_inspection.domain.errors import DeploymentContractError, RoiValidationError
from .guards import require_bound_capture_gate_policy
from zs32_inspection.domain.evidence import EvidenceBranch
from zs32_inspection.domain.identity import PRODUCT
from zs32_inspection.domain.topology import CaptureTopology

Record = TypeVar("Record")


def _unique_index(
    records: tuple[Record, ...],
    key: Callable[[Record], Hashable],
    label: str,
) -> dict[Hashable, Record]:
    """Build a unique record index using one callable key."""
    index: dict[Hashable, Record] = {}
    for record in records:
        identity = key(record)
        if identity in index:
            msg = f"duplicate {label} for identity {identity!r}"
            raise DeploymentContractError(msg)
        index[identity] = record
    return index


def _require_exact_keys(actual: set[object], expected: set[object], label: str) -> None:
    if actual != expected:
        missing = sorted(expected - actual, key=str)
        extra = sorted(actual - expected, key=str)
        msg = f"{label} identity set mismatch; missing={missing}, extra={extra}"
        raise DeploymentContractError(msg)


def _validate_asset_paths(
    template_bindings: tuple[TemplateBinding, ...],
    anomaly_bindings: tuple[AnomalyBinding, ...],
    recipe: InspectionRecipe,
) -> None:
    for binding in template_bindings:
        prefix = PurePosixPath("template") / binding.hand.value / binding.view_id
        if not binding.asset.relative_path.startswith(f"{prefix.as_posix()}/"):
            msg = (
                f"template asset path for {binding.hand.value}/{binding.view_id} must be under "
                f"{prefix.as_posix()}/"
            )
            raise DeploymentContractError(msg)
    for binding in anomaly_bindings:
        prefix = PurePosixPath("models") / "anomaly" / binding.family.value / binding.hand.value / binding.view_id
        if not binding.asset.relative_path.startswith(f"{prefix.as_posix()}/"):
            msg = (
                f"anomaly asset path for {binding.hand.value}/{binding.view_id} must be under "
                f"{prefix.as_posix()}/"
            )
            raise DeploymentContractError(msg)
    if PurePosixPath(recipe.yolo_model.relative_path) != PurePosixPath("models/yolo/best.pt"):
        msg = "global YOLO deployment asset must be models/yolo/best.pt; yolo26n.pt is initialization-only"
        raise DeploymentContractError(msg)
    if PurePosixPath(recipe.capture_gate_policy.relative_path) != PurePosixPath(
        "capture/gates/policy.json"
    ):
        msg = "capture gate policy asset must be capture/gates/policy.json"
        raise DeploymentContractError(msg)


def compile_deployment_contract(
    topology: CaptureTopology,
    roi: RoiConfig,
    recipe: InspectionRecipe,
    *,
    fusion_policy_bytes: bytes,
) -> DeploymentContract:
    """Compile and exhaustively validate one immutable deployment contract.

    Validation happens before any model is loaded.  No ROI, asset, threshold,
    view, hand, model family, or hash is defaulted or inferred from a filename.
    ``fusion_policy_bytes`` must be the exact bytes later stored in the release;
    callers must not hash a reformatted JSON/YAML representation.
    """
    if not isinstance(topology, CaptureTopology):
        msg = "topology must be a validated CaptureTopology"
        raise DeploymentContractError(msg)
    if not isinstance(roi, RoiConfig):
        msg = "roi must be a validated RoiConfig"
        raise DeploymentContractError(msg)
    if not isinstance(recipe, InspectionRecipe):
        msg = "recipe must be a validated InspectionRecipe"
        raise DeploymentContractError(msg)
    try:
        require_bound_capture_gate_policy(recipe)
    except (TypeError, ValueError) as error:
        raise DeploymentContractError(str(error)) from error
    if not isinstance(fusion_policy_bytes, bytes) or not fusion_policy_bytes:
        msg = "fusion_policy_bytes must contain the exact non-empty policy artifact bytes"
        raise DeploymentContractError(msg)
    actual_fusion_policy_sha256 = hashlib.sha256(fusion_policy_bytes).hexdigest()
    if actual_fusion_policy_sha256 != recipe.fusion_policy_sha256:
        msg = "recipe fusion_policy_sha256 does not match the supplied policy artifact bytes"
        raise DeploymentContractError(msg)
    if topology.product != PRODUCT or roi.product != PRODUCT or recipe.product != PRODUCT:
        msg = "topology, ROI, and recipe must all target product ZS32"
        raise DeploymentContractError(msg)
    if recipe.topology_id != topology.topology_id or roi.topology_id != topology.topology_id:
        msg = (
            "topology identity mismatch: "
            f"topology={topology.topology_id!r}, roi={roi.topology_id!r}, recipe={recipe.topology_id!r}"
        )
        raise DeploymentContractError(msg)
    if recipe.roi_config_id != roi.roi_config_id:
        msg = f"recipe ROI {recipe.roi_config_id!r} does not match authoritative ROI {roi.roi_config_id!r}"
        raise DeploymentContractError(msg)

    allowed_hands = tuple(sorted(recipe.allowed_hands, key=lambda hand: hand.value))
    for hand in allowed_hands:
        try:
            roi.require_ready(hand, topology.required_views)
        except RoiValidationError as error:
            msg = f"deployment blocked by ROI readiness: {error}"
            raise DeploymentContractError(msg) from error

    expected_hand_views = {
        (hand, view_id)
        for hand in allowed_hands
        for view_id in topology.required_views
    }
    template_index = _unique_index(
        recipe.template_bindings,
        lambda binding: (binding.hand, binding.view_id),
        "template binding",
    )
    anomaly_index = _unique_index(
        recipe.anomaly_bindings,
        lambda binding: (binding.hand, binding.view_id),
        "anomaly binding",
    )
    template_threshold_index = _unique_index(
        recipe.template_thresholds,
        lambda threshold: (threshold.hand, threshold.view_id),
        "template threshold",
    )
    _require_exact_keys(set(template_index), expected_hand_views, "template bindings")
    _require_exact_keys(set(anomaly_index), expected_hand_views, "anomaly bindings")
    _require_exact_keys(set(template_threshold_index), expected_hand_views, "template thresholds")

    for identity, binding in anomaly_index.items():
        if binding.family is not recipe.anomaly_family:
            msg = (
                f"anomaly family mixing is forbidden: {identity!r} uses {binding.family.value!r}, "
                f"recipe selects {recipe.anomaly_family.value!r}"
            )
            raise DeploymentContractError(msg)
    for identity, threshold in template_threshold_index.items():
        binding = template_index[identity]
        if threshold.template_sha256 != binding.asset.sha256:
            msg = f"template threshold for {identity!r} is bound to a different template digest"
            raise DeploymentContractError(msg)

    expected_threshold_keys = {
        (hand, view_id, branch)
        for hand in allowed_hands
        for view_id in topology.required_views
        for branch in (EvidenceBranch.ANOMALY, EvidenceBranch.YOLO)
    }
    threshold_index = _unique_index(
        recipe.model_thresholds,
        lambda threshold: (threshold.hand, threshold.view_id, threshold.branch),
        "model threshold",
    )
    _require_exact_keys(set(threshold_index), expected_threshold_keys, "model thresholds")
    for (hand, view_id, branch), threshold in threshold_index.items():
        expected_model_sha256 = (
            anomaly_index[(hand, view_id)].asset.sha256
            if branch is EvidenceBranch.ANOMALY
            else recipe.yolo_model.sha256
        )
        if threshold.model_sha256 != expected_model_sha256:
            msg = f"threshold for {(hand, view_id, branch.value)!r} uses a different model digest"
            raise DeploymentContractError(msg)
        if threshold.roi_config_id != roi.roi_config_id:
            msg = f"threshold for {(hand, view_id, branch.value)!r} uses a different ROI version"
            raise DeploymentContractError(msg)

    template_bindings = tuple(
        template_index[(hand, view_id)]
        for hand in allowed_hands
        for view_id in topology.required_views
    )
    anomaly_bindings = tuple(
        anomaly_index[(hand, view_id)]
        for hand in allowed_hands
        for view_id in topology.required_views
    )
    template_thresholds = tuple(
        template_threshold_index[(hand, view_id)]
        for hand in allowed_hands
        for view_id in topology.required_views
    )
    model_thresholds = tuple(
        threshold_index[(hand, view_id, branch)]
        for hand in allowed_hands
        for view_id in topology.required_views
        for branch in (EvidenceBranch.ANOMALY, EvidenceBranch.YOLO)
    )
    _validate_asset_paths(template_bindings, anomaly_bindings, recipe)

    required_groups = tuple(
        RequiredEvidenceGroup(hand=hand, view_id=view_id, branch=branch)
        for hand in allowed_hands
        for view_id in topology.required_views
        for branch in (EvidenceBranch.ANOMALY, EvidenceBranch.YOLO)
    )
    contract = DeploymentContract(
        schema_version=1,
        product=PRODUCT,
        recipe_id=recipe.recipe_id,
        recipe_sha256=recipe.recipe_sha256,
        topology=topology,
        roi=roi,
        allowed_hands=allowed_hands,
        anomaly_family=recipe.anomaly_family,
        capture_gate_policy=recipe.capture_gate_policy,
        template_bindings=template_bindings,
        anomaly_bindings=anomaly_bindings,
        yolo_model=recipe.yolo_model,
        template_thresholds=template_thresholds,
        model_thresholds=model_thresholds,
        required_evidence_groups=required_groups,
        fusion_policy_sha256=actual_fusion_policy_sha256,
        contract_sha256="",
    )
    validate_deployment_contract(contract)
    return contract


def validate_deployment_contract(contract: DeploymentContract) -> None:
    """Revalidate a deserialized contract without trusting its stored digest alone."""
    if not isinstance(contract, DeploymentContract):
        msg = "contract must be a DeploymentContract"
        raise DeploymentContractError(msg)
    if contract.roi.topology_id != contract.topology.topology_id:
        msg = "deployment contract ROI and topology identities differ"
        raise DeploymentContractError(msg)
    if not contract.allowed_hands or len(set(contract.allowed_hands)) != len(contract.allowed_hands):
        msg = "deployment contract allowed_hands must be non-empty and unique"
        raise DeploymentContractError(msg)
    for hand in contract.allowed_hands:
        try:
            contract.roi.require_ready(hand, contract.topology.required_views)
        except RoiValidationError as error:
            msg = f"deployment blocked by ROI readiness: {error}"
            raise DeploymentContractError(msg) from error

    expected_hand_views = {
        (hand, view_id)
        for hand in contract.allowed_hands
        for view_id in contract.topology.required_views
    }
    template_index = _unique_index(
        contract.template_bindings,
        lambda binding: (binding.hand, binding.view_id),
        "template binding",
    )
    anomaly_index = _unique_index(
        contract.anomaly_bindings,
        lambda binding: (binding.hand, binding.view_id),
        "anomaly binding",
    )
    template_threshold_index = _unique_index(
        contract.template_thresholds,
        lambda threshold: (threshold.hand, threshold.view_id),
        "template threshold",
    )
    _require_exact_keys(set(template_index), expected_hand_views, "template bindings")
    _require_exact_keys(set(anomaly_index), expected_hand_views, "anomaly bindings")
    _require_exact_keys(set(template_threshold_index), expected_hand_views, "template thresholds")
    for identity, binding in anomaly_index.items():
        if binding.family is not contract.anomaly_family:
            msg = f"anomaly binding {identity!r} differs from contract anomaly family"
            raise DeploymentContractError(msg)
    for identity, threshold in template_threshold_index.items():
        if threshold.template_sha256 != template_index[identity].asset.sha256:
            msg = f"template threshold for {identity!r} uses a different template digest"
            raise DeploymentContractError(msg)

    expected_threshold_keys = {
        (hand, view_id, branch)
        for hand in contract.allowed_hands
        for view_id in contract.topology.required_views
        for branch in (EvidenceBranch.ANOMALY, EvidenceBranch.YOLO)
    }
    threshold_index = _unique_index(
        contract.model_thresholds,
        lambda threshold: (threshold.hand, threshold.view_id, threshold.branch),
        "model threshold",
    )
    _require_exact_keys(set(threshold_index), expected_threshold_keys, "model thresholds")
    for (hand, view_id, branch), threshold in threshold_index.items():
        expected_digest = (
            anomaly_index[(hand, view_id)].asset.sha256
            if branch is EvidenceBranch.ANOMALY
            else contract.yolo_model.sha256
        )
        if threshold.model_sha256 != expected_digest:
            msg = f"threshold for {(hand, view_id, branch.value)!r} uses a different model digest"
            raise DeploymentContractError(msg)
        if threshold.roi_config_id != contract.roi.roi_config_id:
            msg = f"threshold for {(hand, view_id, branch.value)!r} uses a different ROI version"
            raise DeploymentContractError(msg)

    expected_groups = {
        RequiredEvidenceGroup(hand=hand, view_id=view_id, branch=branch)
        for hand in contract.allowed_hands
        for view_id in contract.topology.required_views
        for branch in (EvidenceBranch.ANOMALY, EvidenceBranch.YOLO)
    }
    actual_groups = set(contract.required_evidence_groups)
    if len(actual_groups) != len(contract.required_evidence_groups):
        msg = "required_evidence_groups contains duplicates"
        raise DeploymentContractError(msg)
    _require_exact_keys(actual_groups, expected_groups, "required evidence groups")

    for binding in contract.template_bindings:
        prefix = f"template/{binding.hand.value}/{binding.view_id}/"
        if not binding.asset.relative_path.startswith(prefix):
            msg = f"template asset path must be under {prefix}"
            raise DeploymentContractError(msg)
    for binding in contract.anomaly_bindings:
        prefix = f"models/anomaly/{binding.family.value}/{binding.hand.value}/{binding.view_id}/"
        if not binding.asset.relative_path.startswith(prefix):
            msg = f"anomaly asset path must be under {prefix}"
            raise DeploymentContractError(msg)
    if PurePosixPath(contract.yolo_model.relative_path) != PurePosixPath("models/yolo/best.pt"):
        msg = "global YOLO deployment asset must be models/yolo/best.pt"
        raise DeploymentContractError(msg)
    if PurePosixPath(contract.capture_gate_policy.relative_path) != PurePosixPath(
        "capture/gates/policy.json"
    ):
        msg = "capture gate policy asset must be capture/gates/policy.json"
        raise DeploymentContractError(msg)
