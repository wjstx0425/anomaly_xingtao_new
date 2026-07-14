"""Construct production predictors only from a fully verified release."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePosixPath

from zs32_inspection.models.anomalib_backend import AnomalibRuntimeBackend
from zs32_inspection.models.anomalydino import AnomalyDINOAdapter
from zs32_inspection.models.base import (
    AnomalyModelArtifact,
    AssetFile,
    DeviceSpec,
    ModelContractError,
    ModelSlot,
)
from zs32_inspection.models.efficientad import EfficientADAdapter
from zs32_inspection.models.patchcore import PatchCoreAdapter
from zs32_inspection.models.ultralytics_backend import UltralyticsRuntimeBackend
from zs32_inspection.models.yolo import (
    UltralyticsYoloAdapter,
    YoloDeploymentSpec,
    YoloRuntimeSettings,
    YoloTrainingProvenance,
)
from zs32_inspection.template.artifacts import TemplateAssetSpec
from zs32_inspection.template.opencv_backend import OpenCvTemplatePredictor

from .orchestrator import LoadedRuntime, RawPredictor
from .release_loader import ReleaseLoadError, VerifiedDeploymentRelease


def _asset(
    release: VerifiedDeploymentRelease,
    *,
    role: str,
    relative: str,
    media_type: str,
    logical_path: str | None = None,
) -> AssetFile:
    """Bind one indexed release file and rehash it at this use boundary."""
    release.files.read_bytes(relative)
    return AssetFile(
        role=role,
        path=release.files.path(relative),
        sha256=release.files.checksums[relative],
        media_type=media_type,
        logical_path=logical_path or relative,
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ReleaseLoadError(f"{label} must be an object")
    return value


def _template_assets(
    release: VerifiedDeploymentRelease,
) -> dict[ModelSlot, TemplateAssetSpec]:
    output: dict[ModelSlot, TemplateAssetSpec] = {}
    for binding in release.contract.template_bindings:
        slot = ModelSlot(binding.hand.value, binding.view_id)
        group = PurePosixPath(binding.asset.relative_path).parent
        metadata = release.files.read_json((group / "metadata.json").as_posix())
        raw_references = metadata["reference_templates"]
        if not isinstance(raw_references, list):
            raise ReleaseLoadError(f"template references must be an array for {slot.key}")
        references = tuple(
            _asset(
                release,
                role="reference_template",
                relative=reference["relative_path"],
                media_type="image/png",
                logical_path=reference["relative_path"],
            )
            for reference in raw_references
        )
        model = _asset(
            release,
            role="template_model",
            relative=binding.asset.relative_path,
            media_type="application/json",
            logical_path=binding.asset.relative_path,
        )
        output[slot] = TemplateAssetSpec(
            slot=slot,
            model=model,
            templates=references,
            model_digest=binding.asset.sha256,
            template_version=metadata["template_version"],
            roi_version=metadata["roi_version"],
            roi_digest=metadata["roi_sha256"],
            dataset_release_id=metadata["dataset_release_id"],
            dataset_manifest_digest=metadata["dataset_manifest_sha256"],
            train_split_id=metadata["train_split_id"],
            recipe_digest=metadata["recipe_sha256"],
            framework_version=metadata["framework_version"],
            training_parameters=_mapping(
                metadata["training_parameters"],
                "template training_parameters",
            ),
            execution_receipt=_mapping(
                metadata["execution_receipt"],
                "template execution_receipt",
            ),
        )
    return output


def _anomaly_artifacts(
    release: VerifiedDeploymentRelease,
) -> dict[ModelSlot, AnomalyModelArtifact]:
    output: dict[ModelSlot, AnomalyModelArtifact] = {}
    for binding in release.contract.anomaly_bindings:
        slot = ModelSlot(binding.hand.value, binding.view_id)
        group = PurePosixPath(binding.asset.relative_path).parent
        metadata = release.files.read_json((group / "metadata.json").as_posix())
        checkpoint = _asset(
            release,
            role="checkpoint",
            relative=binding.asset.relative_path,
            media_type="application/x-pytorch",
            logical_path=binding.asset.relative_path,
        )
        output[slot] = AnomalyModelArtifact(
            family=binding.family,
            slot=slot,
            checkpoint=checkpoint,
            model_digest=binding.asset.sha256,
            dataset_release_id=metadata["dataset_release_id"],
            dataset_manifest_digest=metadata["dataset_manifest_sha256"],
            train_split_id=metadata["train_split_id"],
            recipe_digest=metadata["recipe_sha256"],
            roi_version=metadata["roi_version"],
            roi_digest=metadata["roi_sha256"],
            framework_version=metadata["framework_version"],
            training_parameters=_mapping(metadata["training_parameters"], "anomaly training_parameters"),
            execution_receipt=_mapping(
                metadata["execution_receipt"],
                "anomaly execution_receipt",
            ),
        )
    return output


def _yolo_spec(release: VerifiedDeploymentRelease) -> YoloDeploymentSpec:
    metadata = release.files.read_json("models/yolo/metadata.json")
    runtime = _mapping(metadata["runtime"], "YOLO runtime")
    provenance = _mapping(metadata["provenance"], "YOLO provenance")
    if set(runtime) != {"imgsz", "candidate_conf", "iou", "max_det", "single_class_name"}:
        raise ReleaseLoadError("YOLO runtime settings have missing or unknown fields")
    if set(provenance) != {
        "dataset_release_id", "dataset_manifest_digest", "yolo_export_publication_id",
        "yolo_export_root_sha256", "yolo_export_manifest_digest",
        "yolo_export_data_yaml_digest", "yolo_export_policy_digest",
        "args_data_reference",
        "ultralytics_version_or_commit", "trainer_source", "training_receipt_digest",
        "training_seed", "run_id", "run_name",
    }:
        raise ReleaseLoadError("YOLO provenance has missing or unknown fields")
    weights = _asset(
        release,
        role="weights",
        relative="models/yolo/best.pt",
        media_type="application/x-pytorch",
        logical_path="best.pt",
    )
    args_yaml = _asset(
        release,
        role="train_args",
        relative="models/yolo/train_args.yaml",
        media_type="application/yaml",
        logical_path="args.yaml",
    )
    data_yaml = _asset(
        release,
        role="dataset_config",
        relative="models/yolo/data.yaml",
        media_type="application/yaml",
        logical_path="data.yaml",
    )
    classes = _asset(
        release,
        role="class_names",
        relative="models/yolo/class_names.yaml",
        media_type="application/yaml",
        logical_path="class_names.yaml",
    )
    training_receipt = _asset(
        release,
        role="training_receipt",
        relative="models/yolo/training_receipt.json",
        media_type="application/json",
        logical_path="training_receipt.json",
    )
    return YoloDeploymentSpec(
        weights=weights,
        args_yaml=args_yaml,
        data_yaml=data_yaml,
        class_names_yaml=classes,
        training_receipt=training_receipt,
        model_digest=metadata["model_sha256"],
        bundle_digest=metadata["bundle_sha256"],
        provenance=YoloTrainingProvenance(
            dataset_release_id=provenance["dataset_release_id"],
            dataset_manifest_digest=provenance["dataset_manifest_digest"],
            yolo_export_publication_id=provenance["yolo_export_publication_id"],
            yolo_export_root_sha256=provenance["yolo_export_root_sha256"],
            yolo_export_manifest_digest=provenance["yolo_export_manifest_digest"],
            yolo_export_data_yaml_digest=provenance["yolo_export_data_yaml_digest"],
            yolo_export_policy_digest=provenance["yolo_export_policy_digest"],
            args_data_reference=provenance["args_data_reference"],
            ultralytics_version_or_commit=provenance["ultralytics_version_or_commit"],
            trainer_source=provenance["trainer_source"],
            training_receipt_digest=provenance["training_receipt_digest"],
            training_seed=provenance["training_seed"],
            run_id=provenance["run_id"],
            run_name=provenance["run_name"],
        ),
        runtime=YoloRuntimeSettings(
            imgsz=runtime["imgsz"],
            candidate_conf=runtime["candidate_conf"],
            iou=runtime["iou"],
            max_det=runtime["max_det"],
            single_class_name=runtime["single_class_name"],
        ),
    )


class VerifiedReleasePredictorLoader:
    """Reconstruct and load only the model stage requested by the orchestrator."""

    def __init__(self, release: VerifiedDeploymentRelease, device: DeviceSpec) -> None:
        self._release = release
        self._device = device

    def load_template_predictor(self) -> OpenCvTemplatePredictor:
        return OpenCvTemplatePredictor(_template_assets(self._release))

    def load_anomaly_predictor(self) -> RawPredictor:
        anomaly = _anomaly_artifacts(self._release)
        family = self._release.contract.anomaly_family
        adapter_class = {
            "patchcore": PatchCoreAdapter,
            "efficientad": EfficientADAdapter,
            "anomalydino": AnomalyDINOAdapter,
        }[family.value]
        return adapter_class(runtime_loader=AnomalibRuntimeBackend()).load(
            anomaly,
            self._device,
        )

    def load_yolo_predictor(self) -> RawPredictor:
        return UltralyticsYoloAdapter(UltralyticsRuntimeBackend()).load(
            _yolo_spec(self._release),
            self._device,
        )


def build_loaded_runtime(
    release: VerifiedDeploymentRelease,
    *,
    device: DeviceSpec,
) -> LoadedRuntime:
    """Bind a verified release without importing or loading any model stage."""
    if not isinstance(release, VerifiedDeploymentRelease):
        raise TypeError("runtime factory requires a VerifiedDeploymentRelease")
    if not isinstance(device, DeviceSpec):
        raise TypeError("runtime factory requires a DeviceSpec")
    expected_slots = {
        ModelSlot(hand.value, view)
        for hand in release.contract.allowed_hands
        for view in release.contract.topology.required_views
    }
    template_slots = {
        ModelSlot(binding.hand.value, binding.view_id)
        for binding in release.contract.template_bindings
    }
    anomaly_slots = {
        ModelSlot(binding.hand.value, binding.view_id)
        for binding in release.contract.anomaly_bindings
    }
    if template_slots != expected_slots or anomaly_slots != expected_slots:
        raise ModelContractError("release runtime slot groups are incomplete")
    return LoadedRuntime.from_verified_release(
        release,
        predictor_loader=VerifiedReleasePredictorLoader(release, device),
    )


__all__ = ["VerifiedReleasePredictorLoader", "build_loaded_runtime"]
