"""Strict parsing of content-addressed Phase 4-5 release inputs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from zs32_inspection.domain.contracts import AnomalyFamily
from zs32_inspection.models.base import AnomalyModelArtifact, AssetFile, ModelSlot
from zs32_inspection.models.deployment import DeploymentAssetDescription
from zs32_inspection.models.registry import CandidateStatus, ModelCandidate
from zs32_inspection.models.yolo import (
    YoloDeploymentSpec,
    YoloRuntimeSettings,
    YoloTrainingProvenance,
)
from zs32_inspection.template.artifacts import TemplateAssetSpec

from ._common import array_value, object_value, require_keys, resolved_asset, string_value


def _asset(payload: object, *, root: Path, context: str) -> AssetFile:
    item = object_value(payload, context)
    require_keys(
        item,
        required=("role", "relative_path", "sha256", "media_type"),
        context=context,
    )
    relative = string_value(item["relative_path"], f"{context}.relative_path")
    return AssetFile(
        role=string_value(item["role"], f"{context}.role"),
        path=resolved_asset(root, relative, context),
        sha256=string_value(item["sha256"], f"{context}.sha256"),
        media_type=string_value(item["media_type"], f"{context}.media_type"),
        logical_path=relative,
    )


def _anomaly(payload: object, *, root: Path, index: int) -> AnomalyModelArtifact:
    context = f"candidate.anomaly_artifacts[{index}]"
    item = object_value(payload, context)
    require_keys(
        item,
        required=(
            "family", "hand", "view", "checkpoint", "model_digest", "dataset_release_id",
            "dataset_manifest_digest", "train_split_id", "recipe_digest", "roi_version",
            "roi_digest", "framework_version", "training_parameters",
            "execution_receipt",
        ),
        context=context,
    )
    return AnomalyModelArtifact(
        family=AnomalyFamily(item["family"]),
        slot=ModelSlot(item["hand"], item["view"]),
        checkpoint=_asset(item["checkpoint"], root=root, context=f"{context}.checkpoint"),
        model_digest=item["model_digest"],
        dataset_release_id=item["dataset_release_id"],
        dataset_manifest_digest=item["dataset_manifest_digest"],
        train_split_id=item["train_split_id"],
        recipe_digest=item["recipe_digest"],
        roi_version=item["roi_version"],
        roi_digest=item["roi_digest"],
        framework_version=item["framework_version"],
        training_parameters=object_value(item["training_parameters"], f"{context}.training_parameters"),
        execution_receipt=object_value(item["execution_receipt"], f"{context}.execution_receipt"),
    )


def _yolo(payload: object, *, root: Path) -> YoloDeploymentSpec:
    item = object_value(payload, "candidate.yolo")
    require_keys(
        item,
        required=("schema", "schema_version", "assets", "model_digest", "provenance", "runtime", "bundle_digest"),
        context="candidate.yolo",
    )
    if item["schema"] != "zs32.yolo_deployment" or item["schema_version"] != 3:
        raise ValueError("candidate.yolo schema must be zs32.yolo_deployment version 3")
    asset_rows = tuple(
        asset
        for index, raw in enumerate(array_value(item["assets"], "candidate.yolo.assets"))
        for asset in (_asset(raw, root=root, context=f"candidate.yolo.assets[{index}]"),)
    )
    assets = {asset.role: asset for asset in asset_rows}
    if len(assets) != len(asset_rows):
        raise ValueError("candidate.yolo contains duplicate asset roles")
    if set(assets) != {
        "weights", "train_args", "dataset_config", "class_names", "training_receipt"
    }:
        raise ValueError(f"candidate.yolo asset roles are incomplete: {sorted(assets)}")
    provenance = object_value(item["provenance"], "candidate.yolo.provenance")
    require_keys(
        provenance,
        required=(
            "dataset_release_id", "dataset_manifest_digest", "yolo_export_publication_id",
            "yolo_export_root_sha256", "yolo_export_manifest_digest",
            "yolo_export_data_yaml_digest", "yolo_export_policy_digest",
            "args_data_reference",
            "ultralytics_version_or_commit", "trainer_source", "training_receipt_digest",
            "training_seed", "run_id", "run_name",
        ),
        context="candidate.yolo.provenance",
    )
    runtime = object_value(item["runtime"], "candidate.yolo.runtime")
    require_keys(
        runtime,
        required=("imgsz", "candidate_conf", "iou", "max_det", "single_class_name"),
        context="candidate.yolo.runtime",
    )
    return YoloDeploymentSpec(
        weights=assets["weights"],
        args_yaml=assets["train_args"],
        data_yaml=assets["dataset_config"],
        class_names_yaml=assets["class_names"],
        training_receipt=assets["training_receipt"],
        model_digest=item["model_digest"],
        bundle_digest=item["bundle_digest"],
        provenance=YoloTrainingProvenance(**provenance),
        runtime=YoloRuntimeSettings(**runtime),
    )


def parse_candidate(payload: dict[str, Any], *, asset_root: Path) -> ModelCandidate:
    """Parse a complete ModelCandidate.to_dict document and re-check its digest."""
    require_keys(
        payload,
        required=(
            "candidate_id", "product", "anomaly_family", "anomaly_artifacts", "yolo",
            "recipe_digest", "topology_digest", "roi_digest", "roi_version",
            "dataset_release_id", "dataset_manifest_digest", "status", "candidate_digest",
        ),
        context="candidate",
    )
    artifacts = tuple(
        _anomaly(item, root=asset_root, index=index)
        for index, item in enumerate(array_value(payload["anomaly_artifacts"], "candidate.anomaly_artifacts"))
    )
    by_slot = {item.slot: item for item in artifacts}
    if len(by_slot) != len(artifacts):
        raise ValueError("candidate contains duplicate anomaly slots")
    candidate = ModelCandidate(
        candidate_id=payload["candidate_id"],
        product=payload["product"],
        anomaly_family=AnomalyFamily(payload["anomaly_family"]),
        anomaly_artifacts=by_slot,
        yolo=_yolo(payload["yolo"], root=asset_root),
        recipe_digest=payload["recipe_digest"],
        topology_digest=payload["topology_digest"],
        roi_digest=payload["roi_digest"],
        roi_version=payload["roi_version"],
        dataset_release_id=payload["dataset_release_id"],
        dataset_manifest_digest=payload["dataset_manifest_digest"],
        status=CandidateStatus(payload["status"]),
    )
    if candidate.digest != payload["candidate_digest"]:
        raise ValueError("candidate_digest does not match candidate content")
    return candidate


def parse_candidate_registration(
    payload: dict[str, Any],
    *,
    asset_root: Path,
    candidate_id: str,
) -> tuple[ModelCandidate, tuple[ModelSlot, ...], Mapping[ModelSlot, TemplateAssetSpec]]:
    """Parse the one-time aggregate of frozen training outputs.

    The registration document has no lifecycle status or self-declared digest;
    both are derived here.  It includes template outputs so registration can
    emit the exact descriptor consumed by calibration scoring without manual
    copy/paste of candidate identities.
    """
    require_keys(
        payload,
        required=(
            "schema", "schema_version", "product", "anomaly_family",
            "anomaly_artifacts", "yolo", "recipe_digest", "topology_digest",
            "roi_digest", "roi_version", "dataset_release_id",
            "dataset_manifest_digest", "required_slots", "templates",
        ),
        context="candidate_registration",
    )
    if payload["schema"] != "zs32.candidate_registration" or payload["schema_version"] != 1:
        raise ValueError(
            "candidate_registration schema must be zs32.candidate_registration version 1"
        )
    artifacts = tuple(
        _anomaly(item, root=asset_root, index=index)
        for index, item in enumerate(
            array_value(payload["anomaly_artifacts"], "candidate_registration.anomaly_artifacts")
        )
    )
    anomaly_by_slot = {item.slot: item for item in artifacts}
    if len(anomaly_by_slot) != len(artifacts):
        raise ValueError("candidate_registration contains duplicate anomaly slots")
    candidate = ModelCandidate(
        candidate_id=candidate_id,
        product=payload["product"],
        anomaly_family=AnomalyFamily(payload["anomaly_family"]),
        anomaly_artifacts=anomaly_by_slot,
        yolo=_yolo(payload["yolo"], root=asset_root),
        recipe_digest=payload["recipe_digest"],
        topology_digest=payload["topology_digest"],
        roi_digest=payload["roi_digest"],
        roi_version=payload["roi_version"],
        dataset_release_id=payload["dataset_release_id"],
        dataset_manifest_digest=payload["dataset_manifest_digest"],
        status=CandidateStatus.REGISTERED,
    )
    required_slots = tuple(
        _slot(item, f"candidate_registration.required_slots[{index}]")
        for index, item in enumerate(
            array_value(payload["required_slots"], "candidate_registration.required_slots")
        )
    )
    if not required_slots or len(required_slots) != len(set(required_slots)):
        raise ValueError("candidate_registration required_slots must be non-empty and unique")
    candidate.validate_slots(required_slots)
    templates = tuple(
        _template(
            item,
            root=asset_root,
            index=index,
            context_prefix="candidate_registration",
        )
        for index, item in enumerate(
            array_value(payload["templates"], "candidate_registration.templates")
        )
    )
    template_by_slot = {item.slot: item for item in templates}
    if len(template_by_slot) != len(templates) or set(template_by_slot) != set(required_slots):
        raise ValueError("candidate_registration template slots are duplicated or incomplete")
    for slot in required_slots:
        template = template_by_slot[slot]
        anomaly = candidate.anomaly_artifacts[slot]
        if (
            template.roi_version != candidate.roi_version
            or template.roi_digest != candidate.roi_digest
            or template.dataset_release_id != candidate.dataset_release_id
            or template.dataset_manifest_digest != candidate.dataset_manifest_digest
            or template.train_split_id != anomaly.train_split_id
            or template.recipe_digest != candidate.recipe_digest
        ):
            raise ValueError(f"candidate_registration template provenance mismatch: {slot.key}")
    return candidate, required_slots, MappingProxyType(template_by_slot)


def template_assets_payload(
    candidate: ModelCandidate,
    required_slots: tuple[ModelSlot, ...],
    templates: Mapping[ModelSlot, TemplateAssetSpec],
) -> dict[str, object]:
    """Serialize the exact pre-calibration template descriptor."""
    return {
        "schema": "zs32.template_assets",
        "schema_version": 1,
        "candidate_id": candidate.candidate_id,
        "candidate_digest": candidate.digest,
        "required_slots": [slot.key for slot in required_slots],
        "templates": [
            {
                "hand": slot.hand,
                "view": slot.view,
                "model": templates[slot].model.to_dict(),
                "templates": [asset.to_dict() for asset in templates[slot].templates],
                "model_digest": templates[slot].model_digest,
                "template_version": templates[slot].template_version,
                "roi_version": templates[slot].roi_version,
                "roi_digest": templates[slot].roi_digest,
                "dataset_release_id": templates[slot].dataset_release_id,
                "dataset_manifest_digest": templates[slot].dataset_manifest_digest,
                "train_split_id": templates[slot].train_split_id,
                "recipe_digest": templates[slot].recipe_digest,
                "framework_version": templates[slot].framework_version,
                "training_parameters": dict(templates[slot].training_parameters),
                "execution_receipt": dict(templates[slot].execution_receipt),
            }
            for slot in required_slots
        ],
    }


def _slot(text: object, context: str) -> ModelSlot:
    if not isinstance(text, str) or text.count("/") != 1:
        raise ValueError(f"{context} must use hand/view")
    hand, view = text.split("/", maxsplit=1)
    return ModelSlot(hand, view)


def _template(
    payload: object,
    *,
    root: Path,
    index: int,
    context_prefix: str = "deployment_assets",
) -> TemplateAssetSpec:
    context = f"{context_prefix}.templates[{index}]"
    item = object_value(payload, context)
    require_keys(
        item,
        required=(
            "hand", "view", "model", "templates", "model_digest", "template_version",
            "roi_version", "roi_digest", "dataset_release_id", "dataset_manifest_digest",
            "train_split_id", "recipe_digest",
            "framework_version", "training_parameters", "execution_receipt",
        ),
        context=context,
    )
    references = tuple(
        _asset(raw, root=root, context=f"{context}.templates[{template_index}]")
        for template_index, raw in enumerate(array_value(item["templates"], f"{context}.templates"))
    )
    return TemplateAssetSpec(
        slot=ModelSlot(item["hand"], item["view"]),
        model=_asset(item["model"], root=root, context=f"{context}.model"),
        templates=references,
        model_digest=item["model_digest"],
        template_version=item["template_version"],
        roi_version=item["roi_version"],
        roi_digest=item["roi_digest"],
        dataset_release_id=item["dataset_release_id"],
        dataset_manifest_digest=item["dataset_manifest_digest"],
        train_split_id=item["train_split_id"],
        recipe_digest=item["recipe_digest"],
        framework_version=item["framework_version"],
        training_parameters=object_value(
            item["training_parameters"],
            f"{context}.training_parameters",
        ),
        execution_receipt=object_value(
            item["execution_receipt"],
            f"{context}.execution_receipt",
        ),
    )


def parse_template_assets(
    payload: dict[str, Any],
    *,
    asset_root: Path,
    candidate: ModelCandidate,
) -> tuple[tuple[ModelSlot, ...], Mapping[ModelSlot, TemplateAssetSpec]]:
    """Parse frozen template candidates before calibration exists.

    Threshold fitting happens after the frozen models have scored the
    calibration and test partitions.  This descriptor therefore binds only
    the already-trained template assets to the model candidate; it cannot
    contain threshold or calibration placeholders.
    """
    require_keys(
        payload,
        required=(
            "schema", "schema_version", "candidate_id", "candidate_digest",
            "required_slots", "templates",
        ),
        context="template_assets",
    )
    if payload["schema"] != "zs32.template_assets" or payload["schema_version"] != 1:
        raise ValueError("template_assets schema must be zs32.template_assets version 1")
    if payload["candidate_id"] != candidate.candidate_id or payload["candidate_digest"] != candidate.digest:
        raise ValueError("template_assets candidate identity differs from candidate document")
    required_slots = tuple(
        _slot(item, f"template_assets.required_slots[{index}]")
        for index, item in enumerate(array_value(payload["required_slots"], "template_assets.required_slots"))
    )
    if not required_slots or len(required_slots) != len(set(required_slots)):
        raise ValueError("template_assets required_slots must be non-empty and unique")
    candidate.validate_slots(required_slots)
    templates = tuple(
        _template(item, root=asset_root, index=index, context_prefix="template_assets")
        for index, item in enumerate(array_value(payload["templates"], "template_assets.templates"))
    )
    by_slot = {item.slot: item for item in templates}
    if len(by_slot) != len(templates):
        raise ValueError("template_assets contains duplicate template slots")
    if set(by_slot) != set(required_slots):
        missing = sorted(slot.key for slot in set(required_slots) - set(by_slot))
        unexpected = sorted(slot.key for slot in set(by_slot) - set(required_slots))
        raise ValueError(
            f"template_assets slot mismatch; missing={missing}, unexpected={unexpected}"
        )
    for slot, template in by_slot.items():
        anomaly = candidate.anomaly_artifacts[slot]
        if (
            template.roi_version != candidate.roi_version
            or template.roi_digest != candidate.roi_digest
            or template.roi_version != anomaly.roi_version
            or template.roi_digest != anomaly.roi_digest
        ):
            raise ValueError(f"template/anomaly/candidate ROI provenance mismatch for {slot.key}")
        if (
            template.dataset_release_id != candidate.dataset_release_id
            or template.dataset_manifest_digest != candidate.dataset_manifest_digest
            or template.recipe_digest != candidate.recipe_digest
            or template.train_split_id != anomaly.train_split_id
        ):
            raise ValueError(f"template candidate provenance mismatch for {slot.key}")
    return required_slots, MappingProxyType(by_slot)


def parse_deployment_assets(
    payload: dict[str, Any],
    *,
    asset_root: Path,
    candidate: ModelCandidate,
) -> DeploymentAssetDescription:
    """Parse DeploymentAssetDescription.to_dict without trusting source paths."""
    require_keys(
        payload,
        required=(
            "schema", "schema_version", "candidate_id", "candidate_digest",
            "required_slots", "templates", "calibration",
        ),
        context="deployment_assets",
    )
    if payload["schema"] != "zs32.deployment_assets" or payload["schema_version"] != 1:
        raise ValueError("deployment_assets schema must be zs32.deployment_assets version 1")
    if payload["candidate_id"] != candidate.candidate_id or payload["candidate_digest"] != candidate.digest:
        raise ValueError("deployment_assets candidate identity differs from candidate document")
    required_slots = tuple(
        _slot(item, f"deployment_assets.required_slots[{index}]")
        for index, item in enumerate(array_value(payload["required_slots"], "deployment_assets.required_slots"))
    )
    templates = tuple(
        _template(item, root=asset_root, index=index)
        for index, item in enumerate(array_value(payload["templates"], "deployment_assets.templates"))
    )
    by_slot = {item.slot: item for item in templates}
    if len(by_slot) != len(templates):
        raise ValueError("deployment_assets contains duplicate template slots")
    calibration = object_value(payload["calibration"], "deployment_assets.calibration")
    require_keys(
        calibration,
        required=(
            "template_thresholds", "model_thresholds", "metrics",
            "input_provenance", "artifact", "artifact_digest",
        ),
        context="deployment_assets.calibration",
    )
    return DeploymentAssetDescription(
        candidate=candidate,
        required_slots=required_slots,
        template_assets=by_slot,
        template_thresholds=_asset(
            calibration["template_thresholds"], root=asset_root, context="calibration.template_thresholds"
        ),
        model_thresholds=_asset(
            calibration["model_thresholds"], root=asset_root, context="calibration.model_thresholds"
        ),
        calibration_metrics=_asset(
            calibration["metrics"], root=asset_root, context="calibration.metrics"
        ),
        calibration_input_provenance=_asset(
            calibration["input_provenance"],
            root=asset_root,
            context="calibration.input_provenance",
        ),
        calibration_artifact=_asset(
            calibration["artifact"], root=asset_root, context="calibration.artifact"
        ),
        calibration_artifact_digest=calibration["artifact_digest"],
    )
