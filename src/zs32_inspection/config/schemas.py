"""Strict mapping parsers for versioned ZS32 configuration schemas."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, TypeVar

from zs32_inspection.domain.contracts import (
    AnomalyBinding,
    AnomalyFamily,
    ArtifactRef,
    DeploymentContract,
    InspectionRecipe,
    ReleaseManifest,
    RequiredEvidenceGroup,
    RoiBox,
    RoiConfig,
    RoiHandConfig,
    RoiReadiness,
    TemplateBinding,
    TemplateThresholdRecord,
    ThresholdRecord,
    canonical_sha256,
)
from zs32_inspection.domain.errors import SchemaValidationError
from zs32_inspection.domain.evidence import EvidenceBranch
from zs32_inspection.domain.identity import Hand
from zs32_inspection.domain.topology import CameraSlot, CaptureRound, CaptureTopology

from .compiler import validate_deployment_contract

EnumValue = TypeVar("EnumValue", bound=Enum)


def _object(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        msg = f"{path} must be an object with string keys"
        raise SchemaValidationError(msg)
    return value


def _array(value: object, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        msg = f"{path} must be an array"
        raise SchemaValidationError(msg)
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        msg = f"{path} must be a non-empty string"
        raise SchemaValidationError(msg)
    return value


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{path} must be an integer"
        raise SchemaValidationError(msg)
    return value


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{path} must be a number"
        raise SchemaValidationError(msg)
    return float(value)


def _enum(enum_type: type[EnumValue], value: object, path: str) -> EnumValue:
    raw = _string(value, path)
    try:
        return enum_type(raw)
    except ValueError as error:
        allowed = [item.value for item in enum_type]
        msg = f"{path} must be one of {allowed}, got {raw!r}"
        raise SchemaValidationError(msg) from error


def _plain(value: Any) -> Any:
    """Recursively copy schema containers into JSON-native dict/list values."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item) for item in value]
    return value


def _keys(
    value: Mapping[str, Any],
    path: str,
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> None:
    actual = set(value)
    missing = sorted(required - actual)
    unknown = sorted(actual - required - optional)
    if missing or unknown:
        msg = f"{path} keys invalid; missing={missing}, unknown={unknown}"
        raise SchemaValidationError(msg)


def _artifact(value: object, path: str) -> ArtifactRef:
    item = _object(value, path)
    _keys(
        item,
        path,
        required=frozenset({"artifact_id", "version", "sha256", "relative_path"}),
    )
    return ArtifactRef(
        artifact_id=_string(item["artifact_id"], f"{path}.artifact_id"),
        version=_string(item["version"], f"{path}.version"),
        sha256=_string(item["sha256"], f"{path}.sha256"),
        relative_path=_string(item["relative_path"], f"{path}.relative_path"),
    )


def parse_topology(value: Mapping[str, Any]) -> CaptureTopology:
    """Parse a strict topology and derive its canonical digest."""
    root = _object(value, "topology")
    _keys(
        root,
        "topology",
        required=frozenset(
            {"schema_version", "topology_id", "product", "rounds", "camera_slots", "required_views"},
        ),
    )
    rounds: list[CaptureRound] = []
    for index, raw_round in enumerate(_array(root["rounds"], "topology.rounds")):
        path = f"topology.rounds[{index}]"
        item = _object(raw_round, path)
        _keys(item, path, required=frozenset({"round_id", "prompt"}))
        rounds.append(
            CaptureRound(
                round_id=_string(item["round_id"], f"{path}.round_id"),
                prompt=_string(item["prompt"], f"{path}.prompt"),
            ),
        )
    slots: list[CameraSlot] = []
    for index, raw_slot in enumerate(_array(root["camera_slots"], "topology.camera_slots")):
        path = f"topology.camera_slots[{index}]"
        item = _object(raw_slot, path)
        _keys(item, path, required=frozenset({"slot_id", "serial", "views"}))
        raw_views = _object(item["views"], f"{path}.views")
        slots.append(
            CameraSlot(
                slot_id=_string(item["slot_id"], f"{path}.slot_id"),
                serial=_string(item["serial"], f"{path}.serial"),
                views={
                    _string(round_id, f"{path}.views round_id"): _string(view_id, f"{path}.views[{round_id!r}]")
                    for round_id, view_id in raw_views.items()
                },
            ),
        )
    required_views = tuple(
        _string(view_id, f"topology.required_views[{index}]")
        for index, view_id in enumerate(_array(root["required_views"], "topology.required_views"))
    )
    return CaptureTopology(
        schema_version=_integer(root["schema_version"], "topology.schema_version"),
        topology_id=_string(root["topology_id"], "topology.topology_id"),
        product=_string(root["product"], "topology.product"),
        rounds=tuple(rounds),
        camera_slots=tuple(slots),
        required_views=required_views,
        topology_sha256=canonical_sha256(root),
    )


def parse_roi_config(value: Mapping[str, Any]) -> RoiConfig:
    """Parse the hand-aware v2 ROI schema without defaults or mirroring."""
    root = _object(value, "roi")
    _keys(
        root,
        "roi",
        required=frozenset(
            {
                "schema_version",
                "roi_version",
                "product",
                "topology_id",
                "coordinate_system",
                "source_image_size",
                "hands",
            },
        ),
    )
    source_image = _object(root["source_image_size"], "roi.source_image_size")
    _keys(source_image, "roi.source_image_size", required=frozenset({"width", "height"}))
    raw_hands = _object(root["hands"], "roi.hands")
    hands: dict[Hand, RoiHandConfig] = {}
    for hand_name, raw_hand in raw_hands.items():
        path = f"roi.hands.{hand_name}"
        try:
            hand = Hand(hand_name)
        except ValueError as error:
            msg = f"{path} is not a supported hand"
            raise SchemaValidationError(msg) from error
        item = _object(raw_hand, path)
        _keys(item, path, required=frozenset({"status", "views"}), optional=frozenset({"reason"}))
        raw_views = _object(item["views"], f"{path}.views")
        views: dict[str, RoiBox] = {}
        for view_id, raw_roi in raw_views.items():
            roi_path = f"{path}.views.{view_id}"
            roi_item = _object(raw_roi, roi_path)
            _keys(roi_item, roi_path, required=frozenset({"xyxy"}))
            coordinates = _array(roi_item["xyxy"], f"{roi_path}.xyxy")
            if len(coordinates) != 4:
                msg = f"{roi_path}.xyxy must contain exactly four coordinates"
                raise SchemaValidationError(msg)
            views[_string(view_id, f"{roi_path} key")] = RoiBox(
                *(
                    _integer(coordinate, f"{roi_path}.xyxy[{index}]")
                    for index, coordinate in enumerate(coordinates)
                ),
            )
        reason = item.get("reason")
        if reason is not None:
            reason = _string(reason, f"{path}.reason")
        hands[hand] = RoiHandConfig(
            hand=hand,
            status=_enum(RoiReadiness, item["status"], f"{path}.status"),
            views=views,
            reason=reason,
        )
    return RoiConfig(
        schema_version=_integer(root["schema_version"], "roi.schema_version"),
        roi_config_id=_string(root["roi_version"], "roi.roi_version"),
        product=_string(root["product"], "roi.product"),
        topology_id=_string(root["topology_id"], "roi.topology_id"),
        coordinate_system=_string(root["coordinate_system"], "roi.coordinate_system"),
        source_width=_integer(source_image["width"], "roi.source_image_size.width"),
        source_height=_integer(source_image["height"], "roi.source_image_size.height"),
        hands=hands,
        roi_sha256=canonical_sha256(root),
    )


def parse_recipe(value: Mapping[str, Any]) -> InspectionRecipe:
    """Parse a complete, deployment-oriented ZS32 recipe schema."""
    root = _object(value, "recipe")
    _keys(
        root,
        "recipe",
        required=frozenset(
            {
                "schema_version",
                "recipe_id",
                "product",
                "topology_id",
                "roi_version",
                "allowed_hands",
                "anomaly_family",
                "capture_gate_policy",
                "template_assets",
                "anomaly_models",
                "yolo_model",
                "template_thresholds",
                "model_thresholds",
                "fusion_policy_sha256",
            },
        ),
    )
    allowed_hands = tuple(
        _enum(Hand, hand, f"recipe.allowed_hands[{index}]")
        for index, hand in enumerate(_array(root["allowed_hands"], "recipe.allowed_hands"))
    )
    anomaly_family = _enum(AnomalyFamily, root["anomaly_family"], "recipe.anomaly_family")

    template_bindings: list[TemplateBinding] = []
    for hand_name, raw_views in _object(root["template_assets"], "recipe.template_assets").items():
        hand = _enum(Hand, hand_name, f"recipe.template_assets.{hand_name}")
        for view_id, raw_asset in _object(raw_views, f"recipe.template_assets.{hand_name}").items():
            template_bindings.append(
                TemplateBinding(
                    hand=hand,
                    view_id=_string(view_id, "template asset view_id"),
                    asset=_artifact(raw_asset, f"recipe.template_assets.{hand_name}.{view_id}"),
                ),
            )

    anomaly_bindings: list[AnomalyBinding] = []
    for hand_name, raw_views in _object(root["anomaly_models"], "recipe.anomaly_models").items():
        hand = _enum(Hand, hand_name, f"recipe.anomaly_models.{hand_name}")
        for view_id, raw_model in _object(raw_views, f"recipe.anomaly_models.{hand_name}").items():
            path = f"recipe.anomaly_models.{hand_name}.{view_id}"
            model = _object(raw_model, path)
            _keys(model, path, required=frozenset({"family", "artifact"}))
            anomaly_bindings.append(
                AnomalyBinding(
                    hand=hand,
                    view_id=_string(view_id, "anomaly model view_id"),
                    family=_enum(AnomalyFamily, model["family"], f"{path}.family"),
                    asset=_artifact(model["artifact"], f"{path}.artifact"),
                ),
            )

    template_thresholds: list[TemplateThresholdRecord] = []
    for index, raw_threshold in enumerate(_array(root["template_thresholds"], "recipe.template_thresholds")):
        path = f"recipe.template_thresholds[{index}]"
        item = _object(raw_threshold, path)
        _keys(
            item,
            path,
            required=frozenset({"hand", "view", "threshold", "template_sha256", "calibration_sha256"}),
        )
        template_thresholds.append(
            TemplateThresholdRecord(
                hand=_enum(Hand, item["hand"], f"{path}.hand"),
                view_id=_string(item["view"], f"{path}.view"),
                threshold=_number(item["threshold"], f"{path}.threshold"),
                template_sha256=_string(item["template_sha256"], f"{path}.template_sha256"),
                calibration_sha256=_string(item["calibration_sha256"], f"{path}.calibration_sha256"),
            ),
        )

    model_thresholds: list[ThresholdRecord] = []
    for index, raw_threshold in enumerate(_array(root["model_thresholds"], "recipe.model_thresholds")):
        path = f"recipe.model_thresholds[{index}]"
        item = _object(raw_threshold, path)
        _keys(
            item,
            path,
            required=frozenset(
                {
                    "hand",
                    "view",
                    "branch",
                    "low",
                    "high",
                    "model_sha256",
                    "roi_version",
                    "calibration_sha256",
                },
            ),
        )
        model_thresholds.append(
            ThresholdRecord(
                hand=_enum(Hand, item["hand"], f"{path}.hand"),
                view_id=_string(item["view"], f"{path}.view"),
                branch=_enum(EvidenceBranch, item["branch"], f"{path}.branch"),
                low=_number(item["low"], f"{path}.low"),
                high=_number(item["high"], f"{path}.high"),
                model_sha256=_string(item["model_sha256"], f"{path}.model_sha256"),
                roi_config_id=_string(item["roi_version"], f"{path}.roi_version"),
                calibration_sha256=_string(item["calibration_sha256"], f"{path}.calibration_sha256"),
            ),
        )

    normalized_root = _plain(root)
    for threshold in normalized_root["template_thresholds"]:
        threshold["threshold"] = float(threshold["threshold"])
    for threshold in normalized_root["model_thresholds"]:
        threshold["low"] = float(threshold["low"])
        threshold["high"] = float(threshold["high"])
    normalized_identity = {
        key: normalized_root[key]
        for key in (
            "schema_version",
            "recipe_id",
            "product",
            "topology_id",
            "roi_version",
            "allowed_hands",
            "anomaly_family",
            "fusion_policy_sha256",
        )
    }
    normalized_identity["capture_gate_policy_sha256"] = _object(
        root["capture_gate_policy"],
        "recipe.capture_gate_policy",
    )["sha256"]
    return InspectionRecipe(
        schema_version=_integer(root["schema_version"], "recipe.schema_version"),
        recipe_id=_string(root["recipe_id"], "recipe.recipe_id"),
        product=_string(root["product"], "recipe.product"),
        topology_id=_string(root["topology_id"], "recipe.topology_id"),
        roi_config_id=_string(root["roi_version"], "recipe.roi_version"),
        allowed_hands=allowed_hands,
        anomaly_family=anomaly_family,
        capture_gate_policy=_artifact(
            root["capture_gate_policy"],
            "recipe.capture_gate_policy",
        ),
        template_bindings=tuple(template_bindings),
        anomaly_bindings=tuple(anomaly_bindings),
        yolo_model=_artifact(root["yolo_model"], "recipe.yolo_model"),
        template_thresholds=tuple(template_thresholds),
        model_thresholds=tuple(model_thresholds),
        fusion_policy_sha256=_string(root["fusion_policy_sha256"], "recipe.fusion_policy_sha256"),
        recipe_sha256=canonical_sha256(normalized_identity),
    )


def parse_deployment_contract(value: Mapping[str, Any]) -> DeploymentContract:
    """Deserialize a contract, recompute its digest, and revalidate completeness."""
    root = _object(value, "deployment_contract")
    _keys(
        root,
        "deployment_contract",
        required=frozenset(
            {
                "schema_version",
                "product",
                "recipe_id",
                "recipe_sha256",
                "topology",
                "roi",
                "allowed_hands",
                "anomaly_family",
                "capture_gate_policy",
                "template_bindings",
                "anomaly_bindings",
                "yolo_model",
                "template_thresholds",
                "model_thresholds",
                "required_evidence_groups",
                "fusion_policy_sha256",
                "contract_sha256",
            },
        ),
    )

    topology_payload = dict(_object(root["topology"], "deployment_contract.topology"))
    topology_digest = _string(
        topology_payload.pop("topology_sha256", None),
        "deployment_contract.topology.topology_sha256",
    )
    topology = parse_topology(topology_payload)
    if topology.topology_sha256 != topology_digest:
        msg = "deployment contract topology_sha256 does not match its canonical topology payload"
        raise SchemaValidationError(msg)

    roi_payload = dict(_object(root["roi"], "deployment_contract.roi"))
    roi_digest = _string(roi_payload.pop("roi_sha256", None), "deployment_contract.roi.roi_sha256")
    roi = parse_roi_config(roi_payload)
    if roi.roi_sha256 != roi_digest:
        msg = "deployment contract roi_sha256 does not match its canonical ROI payload"
        raise SchemaValidationError(msg)

    template_bindings: list[TemplateBinding] = []
    for index, raw_binding in enumerate(_array(root["template_bindings"], "deployment_contract.template_bindings")):
        path = f"deployment_contract.template_bindings[{index}]"
        item = _object(raw_binding, path)
        _keys(item, path, required=frozenset({"hand", "view_id", "asset"}))
        template_bindings.append(
            TemplateBinding(
                hand=_enum(Hand, item["hand"], f"{path}.hand"),
                view_id=_string(item["view_id"], f"{path}.view_id"),
                asset=_artifact(item["asset"], f"{path}.asset"),
            ),
        )

    anomaly_bindings: list[AnomalyBinding] = []
    for index, raw_binding in enumerate(_array(root["anomaly_bindings"], "deployment_contract.anomaly_bindings")):
        path = f"deployment_contract.anomaly_bindings[{index}]"
        item = _object(raw_binding, path)
        _keys(item, path, required=frozenset({"hand", "view_id", "family", "asset"}))
        anomaly_bindings.append(
            AnomalyBinding(
                hand=_enum(Hand, item["hand"], f"{path}.hand"),
                view_id=_string(item["view_id"], f"{path}.view_id"),
                family=_enum(AnomalyFamily, item["family"], f"{path}.family"),
                asset=_artifact(item["asset"], f"{path}.asset"),
            ),
        )

    template_thresholds: list[TemplateThresholdRecord] = []
    for index, raw_threshold in enumerate(
        _array(root["template_thresholds"], "deployment_contract.template_thresholds"),
    ):
        path = f"deployment_contract.template_thresholds[{index}]"
        item = _object(raw_threshold, path)
        _keys(
            item,
            path,
            required=frozenset({"hand", "view_id", "threshold", "template_sha256", "calibration_sha256"}),
        )
        template_thresholds.append(
            TemplateThresholdRecord(
                hand=_enum(Hand, item["hand"], f"{path}.hand"),
                view_id=_string(item["view_id"], f"{path}.view_id"),
                threshold=_number(item["threshold"], f"{path}.threshold"),
                template_sha256=_string(item["template_sha256"], f"{path}.template_sha256"),
                calibration_sha256=_string(item["calibration_sha256"], f"{path}.calibration_sha256"),
            ),
        )

    model_thresholds: list[ThresholdRecord] = []
    for index, raw_threshold in enumerate(_array(root["model_thresholds"], "deployment_contract.model_thresholds")):
        path = f"deployment_contract.model_thresholds[{index}]"
        item = _object(raw_threshold, path)
        _keys(
            item,
            path,
            required=frozenset(
                {
                    "hand",
                    "view_id",
                    "branch",
                    "low",
                    "high",
                    "model_sha256",
                    "roi_version",
                    "calibration_sha256",
                },
            ),
        )
        model_thresholds.append(
            ThresholdRecord(
                hand=_enum(Hand, item["hand"], f"{path}.hand"),
                view_id=_string(item["view_id"], f"{path}.view_id"),
                branch=_enum(EvidenceBranch, item["branch"], f"{path}.branch"),
                low=_number(item["low"], f"{path}.low"),
                high=_number(item["high"], f"{path}.high"),
                model_sha256=_string(item["model_sha256"], f"{path}.model_sha256"),
                roi_config_id=_string(item["roi_version"], f"{path}.roi_version"),
                calibration_sha256=_string(item["calibration_sha256"], f"{path}.calibration_sha256"),
            ),
        )

    required_groups: list[RequiredEvidenceGroup] = []
    for index, raw_group in enumerate(
        _array(root["required_evidence_groups"], "deployment_contract.required_evidence_groups"),
    ):
        path = f"deployment_contract.required_evidence_groups[{index}]"
        item = _object(raw_group, path)
        _keys(item, path, required=frozenset({"hand", "view_id", "branch"}))
        required_groups.append(
            RequiredEvidenceGroup(
                hand=_enum(Hand, item["hand"], f"{path}.hand"),
                view_id=_string(item["view_id"], f"{path}.view_id"),
                branch=_enum(EvidenceBranch, item["branch"], f"{path}.branch"),
            ),
        )

    contract = DeploymentContract(
        schema_version=_integer(root["schema_version"], "deployment_contract.schema_version"),
        product=_string(root["product"], "deployment_contract.product"),
        recipe_id=_string(root["recipe_id"], "deployment_contract.recipe_id"),
        recipe_sha256=_string(root["recipe_sha256"], "deployment_contract.recipe_sha256"),
        topology=topology,
        roi=roi,
        allowed_hands=tuple(
            _enum(Hand, hand, f"deployment_contract.allowed_hands[{index}]")
            for index, hand in enumerate(
                _array(root["allowed_hands"], "deployment_contract.allowed_hands"),
            )
        ),
        anomaly_family=_enum(AnomalyFamily, root["anomaly_family"], "deployment_contract.anomaly_family"),
        capture_gate_policy=_artifact(
            root["capture_gate_policy"],
            "deployment_contract.capture_gate_policy",
        ),
        template_bindings=tuple(template_bindings),
        anomaly_bindings=tuple(anomaly_bindings),
        yolo_model=_artifact(root["yolo_model"], "deployment_contract.yolo_model"),
        template_thresholds=tuple(template_thresholds),
        model_thresholds=tuple(model_thresholds),
        required_evidence_groups=tuple(required_groups),
        fusion_policy_sha256=_string(root["fusion_policy_sha256"], "deployment_contract.fusion_policy_sha256"),
        contract_sha256=_string(root["contract_sha256"], "deployment_contract.contract_sha256"),
    )
    validate_deployment_contract(contract)
    return contract


def parse_release_manifest(value: Mapping[str, Any]) -> ReleaseManifest:
    """Parse the non-cyclic immutable release manifest."""
    root = _object(value, "release_manifest")
    _keys(
        root,
        "release_manifest",
        required=frozenset({"schema_version", "release_id", "product", "contract_sha256", "created_at"}),
    )
    return ReleaseManifest(
        schema_version=_integer(root["schema_version"], "release_manifest.schema_version"),
        release_id=_string(root["release_id"], "release_manifest.release_id"),
        product=_string(root["product"], "release_manifest.product"),
        contract_sha256=_string(root["contract_sha256"], "release_manifest.contract_sha256"),
        created_at=_string(root["created_at"], "release_manifest.created_at"),
    )
