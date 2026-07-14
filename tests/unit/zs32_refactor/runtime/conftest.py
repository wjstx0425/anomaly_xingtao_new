"""Linux-only fixtures for the complete immutable ZS32 runtime envelope."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.unit.zs32_refactor.execution_fixtures import execution_receipt_mapping
from zs32_inspection.calibration.acceptance import (
    HeldOutAcceptancePolicy,
    evaluate_heldout_acceptance,
)
from zs32_inspection.calibration.reports import HeldOutPartMetrics
from zs32_inspection.config.compiler import compile_deployment_contract
from zs32_inspection.config.schemas import parse_recipe, parse_roi_config, parse_topology
from zs32_inspection.domain.contracts import DeploymentContract, canonical_sha256
from zs32_inspection.models.base import AnomalyModelArtifact, AssetFile, ModelSlot, sha256_file
from zs32_inspection.models.deployment import DeploymentAssetDescription
from zs32_inspection.models.registry import CandidateStatus, ModelCandidate
from zs32_inspection.models.yolo import import_yolo_bundle
from zs32_inspection.runtime.environment_receipt import (
    GpuDeviceMapping,
    PackageInstallation,
    RuntimeEnvironmentReceipt,
)
from zs32_inspection.runtime.publisher import (
    AtomicDirectoryPublisher,
    canonical_json_bytes,
    verify_atomic_publication,
)
from zs32_inspection.runtime.release_assembler import ReleaseAssemblySpec, assemble_release
from zs32_inspection.template.artifacts import TemplateAssetSpec


FUSION_POLICY_BYTES = (
    b'{"all_required_clear_for_ok":true,"no_majority_vote":true,'
    b'"policy_id":"zs32-strict-fusion-v1","schema_version":1,'
    b'"strong_priority":["anomaly","yolo"]}'
)
ACQUISITION_CONFIG_BYTES = canonical_json_bytes(
    {
        "exposure": 4000.0, "gain": 0.0, "timeout_ms": 3000,
        "capture_interval": 0.2, "hdr": False,
        "short_exposure": 4000.0, "long_exposure": 35000.0,
        "hdr_settle_frames": 8, "align_hdr": True,
        "short_dark_threshold": 80.0, "long_clip_threshold": 245.0,
        "blend_width": 50.0, "blur_size": 101,
        "hdr_max_retries": 2, "hdr_max_clip_pct": 5.0,
        "png_compression": 3, "frame_buffer_size": 52428800,
    }
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _topology_payload(camera_count: int = 3) -> dict[str, Any]:
    slots: list[dict[str, object]] = []
    required_views: list[str] = []
    for index in range(camera_count):
        front = f"front_{index}"
        back = f"back_{index}"
        slots.append(
            {
                "slot_id": f"slot_{index}",
                "serial": f"SERIAL_{index}",
                "views": {"front": front, "back": back},
            }
        )
        required_views.extend((front, back))
    return {
        "schema_version": 1,
        "topology_id": f"zs32-{camera_count}cam-double-side-v1",
        "product": "ZS32",
        "rounds": [
            {"round_id": "front", "prompt": "capture_front"},
            {"round_id": "back", "prompt": "flip_then_capture_back"},
        ],
        "camera_slots": slots,
        "required_views": required_views,
    }


def _roi_payload(topology: object, *, left_ready: bool) -> dict[str, Any]:
    required_views = topology.required_views
    left: dict[str, object]
    if left_ready:
        left = {
            "status": "ready",
            "views": {
                view: {"xyxy": [20, 30, 880, 690]}
                for view in required_views
            },
        }
    else:
        left = {
            "status": "pending",
            "reason": "left ROI is not approved",
            "views": {},
        }
    return {
        "schema_version": 2,
        "roi_version": f"{topology.topology_id}-roi-v2",
        "product": "ZS32",
        "topology_id": topology.topology_id,
        "coordinate_system": "pixel_xyxy_half_open",
        "source_image_size": {"width": 1000, "height": 800},
        "hands": {
            "right": {
                "status": "ready",
                "views": {
                    view: {"xyxy": [10, 20, 900, 700]}
                    for view in required_views
                },
            },
            "left": left,
        },
    }


def _recipe_payload(
    topology: object,
    roi: object,
    *,
    template_digests: dict[str, str],
    anomaly_digests: dict[str, str],
    yolo_digest: str,
    calibration_digest: str,
    capture_gate_policy_digest: str = "6" * 64,
    hands: tuple[str, ...] = ("right",),
) -> dict[str, Any]:
    template_assets: dict[str, dict[str, object]] = {}
    anomaly_models: dict[str, dict[str, object]] = {}
    template_thresholds: list[dict[str, object]] = []
    model_thresholds: list[dict[str, object]] = []
    for hand in hands:
        template_assets[hand] = {}
        anomaly_models[hand] = {}
        for view in topology.required_views:
            template_assets[hand][view] = {
                "artifact_id": f"template-{hand}-{view}",
                "version": "template-v1",
                "sha256": template_digests[view],
                "relative_path": f"template/{hand}/{view}/model.json",
            }
            anomaly_models[hand][view] = {
                "family": "patchcore",
                "artifact": {
                    "artifact_id": f"patchcore-{hand}-{view}",
                    "version": "patchcore-v1",
                    "sha256": anomaly_digests[view],
                    "relative_path": f"models/anomaly/patchcore/{hand}/{view}/model.ckpt",
                },
            }
            template_thresholds.append(
                {
                    "hand": hand,
                    "view": view,
                    "threshold": 0.5,
                    "template_sha256": template_digests[view],
                    "calibration_sha256": calibration_digest,
                }
            )
            for branch, model_digest in (
                ("anomaly", anomaly_digests[view]),
                ("yolo", yolo_digest),
            ):
                model_thresholds.append(
                    {
                        "hand": hand,
                        "view": view,
                        "branch": branch,
                        "low": 0.2,
                        "high": 0.8,
                        "model_sha256": model_digest,
                        "roi_version": roi.roi_config_id,
                        "calibration_sha256": calibration_digest,
                    }
                )
    return {
        "schema_version": 1,
        "recipe_id": "zs32-runtime-contract-test-v1",
        "product": "ZS32",
        "topology_id": topology.topology_id,
        "roi_version": roi.roi_config_id,
        "allowed_hands": list(hands),
        "anomaly_family": "patchcore",
        "capture_gate_policy": {
            "artifact_id": "zs32-capture-gate-policy",
            "version": "capture-gates-v1",
            "sha256": capture_gate_policy_digest,
            "relative_path": "capture/gates/policy.json",
        },
        "template_assets": template_assets,
        "anomaly_models": anomaly_models,
        "yolo_model": {
            "artifact_id": "zs32-global-yolo",
            "version": "yolo-v1",
            "sha256": yolo_digest,
            "relative_path": "models/yolo/best.pt",
        },
        "template_thresholds": template_thresholds,
        "model_thresholds": model_thresholds,
        "fusion_policy_sha256": _sha(FUSION_POLICY_BYTES),
    }


@pytest.fixture
def compiled_contract_factory() -> Callable[..., DeploymentContract]:
    """Build a strict 3-camera contract without touching model frameworks."""

    def build(
        *,
        template_digests: dict[str, str] | None = None,
        anomaly_digests: dict[str, str] | None = None,
        yolo_digest: str = "3" * 64,
        calibration_digest: str = "4" * 64,
        left_ready: bool = False,
        hands: tuple[str, ...] = ("right",),
    ) -> DeploymentContract:
        topology = parse_topology(_topology_payload())
        roi = parse_roi_config(_roi_payload(topology, left_ready=left_ready))
        templates = template_digests or {
            view: _sha(f"template:{view}".encode())
            for view in topology.required_views
        }
        anomaly = anomaly_digests or {
            view: _sha(f"anomaly:{view}".encode())
            for view in topology.required_views
        }
        recipe = parse_recipe(
            _recipe_payload(
                topology,
                roi,
                template_digests=templates,
                anomaly_digests=anomaly,
                yolo_digest=yolo_digest,
                calibration_digest=calibration_digest,
                hands=hands,
            )
        )
        return compile_deployment_contract(
            topology,
            roi,
            recipe,
            fusion_policy_bytes=FUSION_POLICY_BYTES,
        )

    return build


@pytest.fixture
def assembled_release(tmp_path: Path) -> Iterator[Path]:
    """Assemble a real complete release and return its immutable directory."""
    topology = parse_topology(_topology_payload())
    roi = parse_roi_config(_roi_payload(topology, left_ready=False))
    source = tmp_path / "candidate-assets"
    source.mkdir(parents=True, exist_ok=True)
    split_policy = {
        "algorithm": "sha256_seed_stratum_part_v1",
        "seed": 43,
        "calibration_ratio": 0.0,
        "test_ratio": 0.0,
    }
    split_policy_sha256 = hashlib.sha256(
        canonical_json_bytes(split_policy)
    ).hexdigest()
    split_payload = {
        "schema_version": 2,
        "policy": split_policy,
        "policy_sha256": split_policy_sha256,
        "assignments": [
            {
                "part_instance_id": "part-runtime-v1",
                "split": "train",
                "stratum": "right:normal:none",
            }
        ],
    }
    split_path = source / "split_assignments.json"
    split_path.write_bytes(canonical_json_bytes(split_payload))
    semantics_payload = {
        "schema_version": 2,
        "approval": {
            "reviewed_by": "runtime-test",
            "reviewed_at": "2026-07-14T00:00:00Z",
            "approved": True,
        },
        "captures": [{"fixture": "runtime"}],
    }
    semantics_path = source / "canonical_semantics.json"
    semantics_path.write_bytes(canonical_json_bytes(semantics_payload))
    calibration_target_rows = sorted(
        (
            {
                "capture_set_id": "capture-runtime-v1",
                "part_instance_id": "part-runtime-v1",
                "hand": "right",
                "view": view,
                "branch": branch,
                "part_ground_truth": "normal",
                "target": "normal",
                "reason": None,
            }
            for view in topology.required_views
            for branch in ("template", "anomaly", "yolo")
        ),
        key=lambda item: (
            item["capture_set_id"], item["part_instance_id"], item["hand"],
            item["view"], item["branch"],
        ),
    )
    calibration_targets_payload = {
        "schema": "zs32.calibration_targets",
        "schema_version": 1,
        "product": "ZS32",
        "dataset_release_id": "dataset-zs32-runtime-v1",
        "topology_id": topology.topology_id,
        "roi_version": roi.roi_config_id,
        "approval": {
            "reviewed_by": "runtime-test",
            "reviewed_at": "2026-07-14T00:00:00Z",
            "approved": True,
        },
        "targets": calibration_target_rows,
    }
    calibration_targets_path = source / "calibration_targets.json"
    calibration_targets_path.write_bytes(
        canonical_json_bytes(calibration_targets_payload)
    )
    png_codec = {
        "encoder": "opencv.imencode",
        "implementation_version": "4.11.0",
        "format": "png",
        "media_type": "image/png",
        "compression": 1,
        "spatial_operation": "single_roi_xyxy_half_open_crop_no_resize",
    }
    png_codec_sha256 = hashlib.sha256(
        canonical_json_bytes(png_codec)
    ).hexdigest()
    governance_payload = {
        "schema_version": 2,
        "canonical_semantics": {
            "relative_path": "canonical_semantics.json",
            "sha256": sha256_file(semantics_path),
            "approval": semantics_payload["approval"],
        },
        "calibration_targets": {
            "relative_path": "calibration_targets.json",
            "sha256": sha256_file(calibration_targets_path),
            "target_count": len(calibration_target_rows),
            "approval": calibration_targets_payload["approval"],
        },
        "split": {
            "relative_path": "split_assignments.json",
            "artifact_sha256": sha256_file(split_path),
            "policy": split_policy,
            "policy_sha256": split_policy_sha256,
        },
        "canonical_png": {
            "codec": png_codec,
            "codec_sha256": png_codec_sha256,
        },
    }
    governance_path = source / "dataset_provenance.json"
    governance_path.write_bytes(canonical_json_bytes(governance_payload))
    dataset_payload = {
        "schema_version": 4,
        "dataset_release_id": "dataset-zs32-runtime-v1",
        "product": "ZS32",
        "topology_id": topology.topology_id,
        "topology_sha256": topology.topology_sha256,
        "roi_version": roi.roi_config_id,
        "roi_sha256": roi.roi_sha256,
        "capture_gate_policy_sha256": "6" * 64,
        "bbox_migration_policy": "clip_partial_drop_outside_audit_all",
        "required_views": list(topology.required_views),
        "hands": ["right"],
        "sample_count": len(topology.required_views),
        "part_count": 1,
        "capture_set_count": 1,
        "canonical_manifest_sha256": "a" * 64,
        "capture_provenance_sha256": "7" * 64,
        "bbox_audit_sha256": "b" * 64,
        "canonical_semantics_sha256": sha256_file(semantics_path),
        "calibration_targets_sha256": sha256_file(calibration_targets_path),
        "calibration_target_count": len(calibration_target_rows),
        "dataset_provenance_sha256": sha256_file(governance_path),
        "split_assignments_sha256": sha256_file(split_path),
        "split_policy": split_policy,
        "split_policy_sha256": split_policy_sha256,
        "canonical_png_codec": png_codec,
        "canonical_png_codec_sha256": png_codec_sha256,
        "adapter_manifest_sha256": {
            "yolo": "d" * 64,
            "anomalib": "e" * 64,
            "template": "f" * 64,
        },
        "created_at": "2026-07-14T00:00:00Z",
    }
    dataset_path = source / "dataset_release.json"
    dataset_path.write_bytes(canonical_json_bytes(dataset_payload))
    dataset_digest = sha256_file(dataset_path)

    yolo_root = source / "yolo"
    yolo_root.mkdir(parents=True)
    (yolo_root / "best.pt").write_bytes(b"runtime-test-yolo-best")
    (yolo_root / "args.yaml").write_text("data: /exports/yolo/data.yaml\nseed: 43\n", encoding="utf-8")
    (yolo_root / "data.yaml").write_text("path: canonical\n", encoding="utf-8")
    (yolo_root / "class_names.yaml").write_text("0: defect\n", encoding="utf-8")
    receipt_path = source / "training_receipt.json"
    receipt_path.write_bytes(canonical_json_bytes({
        "schema": "zs32.yolo_external_training_receipt",
        "schema_version": 3,
        "attestation_scope": "external_trainer_observation_not_causal_proof",
        "attested_by": "runtime-test",
        "attested_at": "2026-07-14T00:00:00Z",
        "run": {
            "run_id": "runtime-run-1", "run_name": "final_n640_p1_seed43", "actual_seed": 43,
        },
        "artifacts": {
            name: sha256_file(yolo_root / name)
            for name in ("best.pt", "args.yaml", "data.yaml", "class_names.yaml")
        },
        "training_data": {
            "yolo_export_publication_id": "yolo-export-runtime-v1",
            "yolo_export_root_sha256": "8" * 64,
            "export_manifest_sha256": "9" * 64,
            "export_data_yaml_sha256": sha256_file(yolo_root / "data.yaml"),
            "export_policy_sha256": "5" * 64,
            "args_data_reference": "/exports/yolo/data.yaml",
            "dataset_release_id": dataset_payload["dataset_release_id"],
            "dataset_manifest_sha256": dataset_digest,
        },
        "trainer_source": {
            "kind": "git_commit", "repository": "runtime-test", "commit": "7" * 40,
            "worktree_state": "clean",
            "runtime_package_tree_sha256": "6" * 64,
        },
    }))
    yolo = import_yolo_bundle(
        yolo_root,
        training_receipt_path=receipt_path,
    )

    template_files: dict[str, AssetFile] = {}
    reference_files: dict[str, AssetFile] = {}
    anomaly_files: dict[str, AssetFile] = {}
    for view in topology.required_views:
        template_path = source / "templates" / view / "model.json"
        reference_path = source / "templates" / view / "reference.png"
        anomaly_path = source / "anomaly" / view / "model.ckpt"
        template_path.parent.mkdir(parents=True, exist_ok=True)
        anomaly_path.parent.mkdir(parents=True, exist_ok=True)
        template_path.write_bytes(canonical_json_bytes({"view": view, "kind": "template"}))
        reference_path.write_bytes(b"\x89PNG\r\n\x1a\n" + view.encode())
        anomaly_path.write_bytes(f"patchcore:{view}".encode())
        template_files[view] = AssetFile(
            role="template_model",
            path=template_path,
            sha256=sha256_file(template_path),
            media_type="application/json",
        )
        reference_files[view] = AssetFile(
            role="reference_template",
            path=reference_path,
            sha256=sha256_file(reference_path),
            media_type="image/png",
        )
        anomaly_files[view] = AssetFile(
            role="checkpoint",
            path=anomaly_path,
            sha256=sha256_file(anomaly_path),
            media_type="application/x-pytorch",
        )

    gate_root = source / "capture" / "gates"
    gate_root.mkdir(parents=True)
    acquisition_path = source / "capture" / "acquisition" / "hikvision.json"
    acquisition_path.parent.mkdir(parents=True)
    acquisition_path.write_bytes(ACQUISITION_CONFIG_BYTES)
    quality_path = gate_root / "right" / "quality.json"
    registration_path = gate_root / "right" / "registration.json"
    quality_path.parent.mkdir(parents=True)
    reference_artifacts: dict[str, dict[str, str]] = {}
    for view in topology.required_views:
        reference_path = gate_root / "right" / "references" / f"{view}.png"
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        reference_path.write_bytes(b"\x89PNG\r\n\x1a\n" + view.encode())
        relative = reference_path.relative_to(source).as_posix()
        reference_artifacts[view] = {
            "artifact_id": f"registration-reference-right-{view}",
            "version": "registration-reference-v1",
            "sha256": sha256_file(reference_path),
            "relative_path": relative,
        }
    quality_payload: dict[str, object] = {
        "schema_version": 1,
        "profile_id": "quality-right-v1",
        "product": "ZS32",
        "hand": "right",
        "topology_id": topology.topology_id,
        "topology_sha256": topology.topology_sha256,
        "views": {
            view: {
                "brightness_mean_min": 10.0,
                "brightness_mean_max": 245.0,
                "brightness_std_min": 1.0,
                "dark_pixel_level": 5,
                "dark_ratio_max": 0.5,
                "saturated_pixel_level": 250,
                "saturated_ratio_max": 0.5,
                "laplacian_variance_min": 1.0,
            }
            for view in topology.required_views
        },
    }
    quality_payload["profile_sha256"] = canonical_sha256(quality_payload)
    quality_path.write_bytes(canonical_json_bytes(quality_payload))
    registration_payload: dict[str, object] = {
        "schema_version": 1,
        "profile_id": "registration-right-v1",
        "product": "ZS32",
        "hand": "right",
        "topology_id": topology.topology_id,
        "topology_sha256": topology.topology_sha256,
        "views": {
            view: {
                "reference": {
                    "asset_id": reference_artifacts[view]["artifact_id"],
                    "relative_path": reference_artifacts[view]["relative_path"],
                    "sha256": reference_artifacts[view]["sha256"],
                    "width": 1000,
                    "height": 800,
                },
                "thresholds": {
                    "analysis_scale": 0.5,
                    "min_ecc_correlation": 0.5,
                    "max_translation_x_px": 10.0,
                    "max_translation_y_px": 10.0,
                    "max_rotation_deg": 5.0,
                    "max_scale_deviation": 0.1,
                    "max_shear": 0.1,
                    "max_iterations": 50,
                    "epsilon": 0.0001,
                    "gaussian_filter_size": 5,
                },
            }
            for view in topology.required_views
        },
    }
    registration_payload["profile_sha256"] = canonical_sha256(registration_payload)
    registration_path.write_bytes(canonical_json_bytes(registration_payload))
    gate_policy_path = gate_root / "policy.json"
    gate_policy_path.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": 3,
                "policy_id": "zs32-right-capture-gates-v1",
                "product": "ZS32",
                "topology_id": topology.topology_id,
                "topology_sha256": topology.topology_sha256,
                "acquisition_config": {
                    "artifact_id": "hikvision-acquisition-v1",
                    "version": "acquisition-v1",
                    "sha256": sha256_file(acquisition_path),
                    "relative_path": "capture/acquisition/hikvision.json",
                },
                "hands": {
                    "right": {
                        "quality_profile": {
                            "artifact_id": "quality-right-v1",
                            "version": "quality-v1",
                            "sha256": sha256_file(quality_path),
                            "relative_path": quality_path.relative_to(source).as_posix(),
                        },
                        "registration_profile": {
                            "artifact_id": "registration-right-v1",
                            "version": "registration-v1",
                            "sha256": sha256_file(registration_path),
                            "relative_path": registration_path.relative_to(source).as_posix(),
                        },
                        "registration_references": reference_artifacts,
                    }
                },
            }
        )
    )
    gate_policy_digest = sha256_file(gate_policy_path)
    capture_provenance_path = source / "capture_provenance.jsonl"
    capture_provenance_path.write_bytes(
        canonical_json_bytes(
            {
                "capture_set_id": "capture-runtime-v1",
                "part_instance_id": "part-runtime-v1",
                "hand": "right",
                "policy_id": "zs32-right-capture-gates-v1",
                "policy_sha256": gate_policy_digest,
                "topology_sha256": topology.topology_sha256,
                "acquisition_config_sha256": sha256_file(acquisition_path),
                "quality_profile_sha256": sha256_file(quality_path),
                "registration_profile_sha256": sha256_file(registration_path),
                "registration_reference_sha256_by_view": {
                    view: reference_artifacts[view]["sha256"]
                    for view in topology.required_views
                },
            }
        )
    )
    dataset_payload["capture_gate_policy_sha256"] = gate_policy_digest
    dataset_payload["capture_provenance_sha256"] = sha256_file(
        capture_provenance_path
    )
    dataset_path.write_bytes(canonical_json_bytes(dataset_payload))
    dataset_digest = sha256_file(dataset_path)
    receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_payload["training_data"]["dataset_manifest_sha256"] = dataset_digest
    receipt_path.write_bytes(canonical_json_bytes(receipt_payload))
    yolo = import_yolo_bundle(
        yolo_root,
        training_receipt_path=receipt_path,
    )

    placeholder = _recipe_payload(
        topology,
        roi,
        template_digests={view: item.sha256 for view, item in template_files.items()},
        anomaly_digests={view: item.sha256 for view, item in anomaly_files.items()},
        yolo_digest=yolo.model_digest,
        calibration_digest="0" * 64,
        capture_gate_policy_digest=gate_policy_digest,
    )
    recipe_identity = parse_recipe(placeholder).recipe_sha256
    for view in topology.required_views:
        template_path = source / "templates" / view / "model.json"
        reference = reference_files[view]
        template_path.write_bytes(
            canonical_json_bytes(
                {
                    "schema": "zs32.opencv_template",
                    "schema_version": 1,
                    "hand": "right",
                    "view": view,
                    "method": "cv2.TM_CCOEFF_NORMED",
                    "risk": "1-similarity",
                    "preprocessing": {
                        "color": "grayscale",
                        "resize": "aspect_preserving_width",
                        "width": 256,
                        "gaussian_kernel": 3,
                        "max_shift": 8,
                    },
                    "references": [
                        {
                            "index": 1,
                            "artifact_sha256": reference.sha256,
                            "source_crop_sha256": "7" * 64,
                        }
                    ],
                    "template_version": "template-v1",
                    "roi_version": roi.roi_config_id,
                    "roi_sha256": roi.roi_sha256,
                    "dataset_release_id": dataset_payload["dataset_release_id"],
                    "dataset_manifest_sha256": dataset_digest,
                    "train_split_id": dataset_payload["split_assignments_sha256"],
                    "recipe_sha256": recipe_identity,
                    "train_part_count": 2,
                    "framework_version": "opencv-4.11.0",
                    "training_parameters": {"width": 256},
                    "execution_receipt": execution_receipt_mapping("train_template"),
                }
            )
        )
        template_files[view] = AssetFile(
            role="template_model",
            path=template_path,
            sha256=sha256_file(template_path),
            media_type="application/json",
        )
    dual_groups: list[dict[str, object]] = []
    dual_thresholds: list[dict[str, object]] = []
    template_groups: list[dict[str, object]] = []
    template_thresholds: list[dict[str, object]] = []
    for view in topology.required_views:
        template_groups.append(
            {
                "hand": "right",
                "view": view,
                "model_digest": template_files[view].sha256,
                "roi_version": roi.roi_config_id,
                "roi_digest": roi.roi_sha256,
            }
        )
        template_thresholds.append(
            {
                "hand": "right",
                "view": view,
                "model_digest": template_files[view].sha256,
                "roi_version": roi.roi_config_id,
                "roi_digest": roi.roi_sha256,
                "dataset_release_id": dataset_payload["dataset_release_id"],
                "calibration_split_id": "cal-v1",
                "threshold": 0.5,
                "normal_count": 2,
                "defect_count": 2,
                "status": "ok",
            }
        )
        for branch, digest in (
            ("anomaly", anomaly_files[view].sha256),
            ("yolo", yolo.model_digest),
        ):
            dual_groups.append(
                {
                    "hand": "right",
                    "view": view,
                    "branch": branch,
                    "model_digest": digest,
                    "roi_version": roi.roi_config_id,
                }
            )
            dual_thresholds.append(
                {
                    "hand": "right",
                    "view": view,
                    "branch": branch,
                    "model_digest": digest,
                    "roi_version": roi.roi_config_id,
                    "dataset_release_id": dataset_payload["dataset_release_id"],
                    "calibration_split_id": "cal-v1",
                    "low": 0.2,
                    "high": 0.8,
                    "normal_count": 2,
                    "defect_count": 2,
                    "status": "ok",
                }
            )
    calibration_payload = {
        "schema": "zs32.calibration",
        "schema_version": 1,
        "provenance": {
            "recipe_digest": recipe_identity,
            "profile_digest": _sha(FUSION_POLICY_BYTES),
            "topology_digest": topology.topology_sha256,
            "roi_digest": roi.roi_sha256,
            "dataset_release_id": dataset_payload["dataset_release_id"],
            "dataset_manifest_digest": dataset_digest,
            "calibration_split_id": "cal-v1",
            "test_split_id": "test-v1",
            "model_digests": sorted(
                {
                    *(item.sha256 for item in template_files.values()),
                    *(item.sha256 for item in anomaly_files.values()),
                    yolo.model_digest,
                }
            ),
        },
        "parameters": {
            "target_defect_recall": 1.0,
            "normal_quantile": 1.0,
            "min_normal_parts": 1,
            "min_defect_parts": 1,
        },
        "required_dual_groups": dual_groups,
        "required_template_groups": template_groups,
        "dual_thresholds": dual_thresholds,
        "template_thresholds": template_thresholds,
        "heldout_metrics": {
            "dataset_release_id": dataset_payload["dataset_release_id"],
            "test_split_id": "test-v1",
            "part_count": 2,
            "normal_part_count": 1,
            "defect_part_count": 1,
            "clear_count": 1,
            "gray_count": 0,
            "strong_count": 1,
            "template_ng_count": 0,
            "incomplete_count": 0,
            "defect_escape_count": 0,
            "normal_reject_count": 0,
            "defect_escape_rate": 0.0,
            "normal_reject_rate": 0.0,
            "review_rate": 0.0,
            "evaluation_complete": True,
        },
        "calibration_valid": True,
    }
    calibration_root = source / "calibration"
    calibration_root.mkdir(parents=True)
    calibration_path = calibration_root / "calibration_artifact.json"
    calibration_path.write_bytes(canonical_json_bytes(calibration_payload))
    calibration_digest = sha256_file(calibration_path)
    final_recipe = parse_recipe(
        _recipe_payload(
            topology,
            roi,
            template_digests={view: item.sha256 for view, item in template_files.items()},
            anomaly_digests={view: item.sha256 for view, item in anomaly_files.items()},
            yolo_digest=yolo.model_digest,
            calibration_digest=calibration_digest,
            capture_gate_policy_digest=gate_policy_digest,
        )
    )
    assert final_recipe.recipe_sha256 == recipe_identity
    contract = compile_deployment_contract(
        topology,
        roi,
        final_recipe,
        fusion_policy_bytes=FUSION_POLICY_BYTES,
    )

    slots = tuple(ModelSlot("right", view) for view in topology.required_views)
    patchcore_training_parameters = {
        "schema": "zs32.anomalib_backend",
        "schema_version": 2,
        "source_parameters_sha256": "7" * 64,
        "family": "patchcore",
        "image_size": [256, 256],
        "normalization": "imagenet",
        "model": {
            "backbone": "wide_resnet50_2",
            "layers": ["layer2", "layer3"],
            "pre_trained": False,
            "initialization": "explicit_state_dict",
            "coreset_sampling_ratio": 0.1,
            "num_neighbors": 9,
            "precision": "float32",
        },
        "trainer": {
            "max_epochs": 1,
            "precision": "32-true",
            "train_batch_size": 1,
            "num_workers": 0,
            "seed": 43,
            "deterministic": True,
        },
        "dataset": {
            "materialized_manifest_sha256": "6" * 64,
            "selected_train_normal_count": 2,
        },
        "auxiliary": {
            "backbone_weights_asset_id": (
                "imagenet1k-wide_resnet50_2-anomalib-timm-feature-extractor-v1"
            ),
            "backbone_weights_sha256": "5" * 64,
            "state_dict_scope": "anomalib_timm_feature_extractor",
        },
    }
    anomaly_artifacts = {
        slot: AnomalyModelArtifact(
            family="patchcore",
            slot=slot,
            checkpoint=anomaly_files[slot.view],
            model_digest=anomaly_files[slot.view].sha256,
            dataset_release_id=dataset_payload["dataset_release_id"],
            dataset_manifest_digest=dataset_digest,
            train_split_id=dataset_payload["split_assignments_sha256"],
            recipe_digest=recipe_identity,
            roi_version=roi.roi_config_id,
            roi_digest=roi.roi_sha256,
            framework_version="anomalib-test-commit",
            training_parameters={
                **patchcore_training_parameters,
                "slot": {"hand": slot.hand, "view": slot.view},
            },
            execution_receipt=execution_receipt_mapping(
                "train_anomaly",
                input_sha256_by_role={"patchcore_backbone": "5" * 64},
                parameters_sha256="7" * 64,
            ),
        )
        for slot in slots
    }
    templates = {
        slot: TemplateAssetSpec(
            slot=slot,
            model=template_files[slot.view],
            templates=(reference_files[slot.view],),
            model_digest=template_files[slot.view].sha256,
            template_version="template-v1",
            roi_version=roi.roi_config_id,
            roi_digest=roi.roi_sha256,
            dataset_release_id=dataset_payload["dataset_release_id"],
            dataset_manifest_digest=dataset_digest,
            train_split_id=dataset_payload["split_assignments_sha256"],
            recipe_digest=recipe_identity,
            framework_version="opencv-4.11.0",
            training_parameters={"width": 256},
            execution_receipt=execution_receipt_mapping("train_template"),
        )
        for slot in slots
    }
    candidate = ModelCandidate(
        candidate_id="candidate-runtime-v1",
        product="ZS32",
        anomaly_family="patchcore",
        anomaly_artifacts=anomaly_artifacts,
        yolo=yolo,
        recipe_digest=recipe_identity,
        topology_digest=topology.topology_sha256,
        roi_digest=roi.roi_sha256,
        roi_version=roi.roi_config_id,
        dataset_release_id=dataset_payload["dataset_release_id"],
        dataset_manifest_digest=dataset_digest,
        status=CandidateStatus.VALIDATED,
    )

    def calibration_asset(name: str, role: str, payload: object) -> AssetFile:
        path = calibration_root / name
        if name != "calibration_artifact.json":
            path.write_bytes(canonical_json_bytes(payload))
        return AssetFile(role=role, path=path, sha256=sha256_file(path), media_type="application/json")

    calibration_artifact = calibration_asset(
        "calibration_artifact.json", "calibration_artifact", calibration_payload
    )
    deployment_assets = DeploymentAssetDescription(
        candidate=candidate,
        required_slots=slots,
        template_assets=templates,
        template_thresholds=calibration_asset(
            "template_thresholds.json",
            "template_thresholds",
            {
                "schema_version": 1,
                "calibration_artifact_sha256": calibration_digest,
                "thresholds": template_thresholds,
            },
        ),
        model_thresholds=calibration_asset(
            "model_thresholds.json",
            "model_thresholds",
            {
                "schema_version": 1,
                "calibration_artifact_sha256": calibration_digest,
                "thresholds": dual_thresholds,
            },
        ),
        calibration_metrics=calibration_asset(
            "metrics.json", "calibration_metrics", calibration_payload["heldout_metrics"]
        ),
        calibration_input_provenance=calibration_asset(
            "input_provenance.json",
            "calibration_input_provenance",
            {
                "schema": "zs32.calibration_inputs",
                "schema_version": 3,
                "calibration_id": "calibration-runtime-v1",
                "score_run_id": "score-runtime-v1",
                "score_run_root_sha256": "8" * 64,
                "scores_sha256": "9" * 64,
                "score_audit_sha256": "a" * 64,
                "score_execution_receipt_sha256": "d" * 64,
                "candidate_id": candidate.candidate_id,
                "candidate_digest": candidate.digest,
                "candidate_descriptor_sha256": "b" * 64,
                "recipe_digest": recipe_identity,
                "dataset_release_id": dataset_payload["dataset_release_id"],
                "dataset_manifest_sha256": dataset_digest,
                "calibration_targets_sha256": dataset_payload["calibration_targets_sha256"],
                "calibration_target_count": dataset_payload["calibration_target_count"],
                "calibration_split_id": "cal-v1",
                "test_split_id": "test-v1",
            },
        ),
        calibration_artifact=calibration_artifact,
        calibration_artifact_digest=calibration_digest,
    )
    fusion_path = source / "fusion_policy.json"
    fusion_path.write_bytes(FUSION_POLICY_BYTES)
    runtime_environment = RuntimeEnvironmentReceipt(
        schema="zs32.runtime_environment_receipt",
        schema_version=2,
        platform="linux",
        architecture="x86_64",
        kernel_release="6.8.0-test",
        os_release_id="ubuntu",
        os_release_version_id="24.04",
        python_version="3.12.9",
        torch_version="2.7.1+cu128",
        torchvision_version="0.22.1+cu128",
        anomalib_version="2.4.3-dev.0",
        ultralytics_version="8.3.0",
        opencv_version="4.11.0",
        numpy_version="2.2.0",
        torch_cuda_version="12.8",
        cudnn_version=90701,
        nvidia_driver_version="570.124.06",
        device_mapping=GpuDeviceMapping(
            requested_device="0",
            cuda_visible_devices="3",
            torch_device_index=0,
            nvidia_smi_index=3,
            gpu_uuid="GPU-runtime-fixture",
            gpu_name="NVIDIA RTX fixture",
        ),
        package_installations={
            role: PackageInstallation(
                distribution_name=("opencv-python" if role == "opencv" else role),
                distribution_version="1.0",
                metadata_sha256=character * 64,
                record_sha256=character * 64,
                record_entry_count=1,
                installed_tree_sha256=character * 64,
                import_origin_relative_path=f"{role}/__init__.py",
                import_origin_sha256=character * 64,
            )
            for role, character in zip(
                ("torch", "torchvision", "anomalib", "ultralytics", "opencv", "numpy"),
                "abcdef",
                strict=True,
            )
        },
    )
    runtime_environment_path = source / "runtime_environment.json"
    runtime_environment_path.write_bytes(runtime_environment.canonical_bytes())
    calibration_publication_names = (
        "calibration_artifact.json",
        "input_provenance.json",
        "metrics.json",
        "model_thresholds.json",
        "template_thresholds.json",
    )
    calibration_checksum_bytes = "".join(
        f"{sha256_file(calibration_root / name)}  {name}\n"
        for name in sorted(calibration_publication_names)
    ).encode("utf-8")
    calibration_publication_root_sha256 = hashlib.sha256(
        calibration_checksum_bytes
    ).hexdigest()
    validation_id = "validation-runtime-v1"
    heldout_policy = HeldOutAcceptancePolicy.from_mapping(
        {
            "schema": "zs32.heldout_acceptance_policy",
            "schema_version": 1,
            "product": "ZS32",
            "policy_id": "runtime-heldout-strict",
            "policy_version": "1.0.0",
            "limits": {
                "max_defect_escape_rate": 0.0,
                "max_normal_reject_rate": 0.0,
                "max_review_rate": 0.0,
                "min_normal_heldout_parts": 1,
                "min_defect_heldout_parts": 1,
            },
        }
    )
    heldout_decision = evaluate_heldout_acceptance(
        heldout_policy,
        HeldOutPartMetrics(**calibration_payload["heldout_metrics"]),
        metrics_sha256=deployment_assets.calibration_metrics.sha256,
    )
    validation_record = {
        "schema": "zs32.candidate_validation",
        "schema_version": 3,
        "validation_id": validation_id,
        "candidate_id": candidate.candidate_id,
        "candidate_digest": candidate.digest,
        "input_candidate_descriptor_sha256": "b" * 64,
        "input_template_assets_descriptor_sha256": "c" * 64,
        "dataset_release_id": candidate.dataset_release_id,
        "dataset_manifest_sha256": dataset_digest,
        "recipe_sha256": contract.recipe_sha256,
        "contract_sha256": contract.contract_sha256,
        "calibration_artifact_sha256": calibration_digest,
        "calibration_split_id": "cal-v1",
        "test_split_id": "test-v1",
        "candidate_status_before": "registered",
        "candidate_status_after": "validated",
        "manual_promotion_required": True,
        "heldout_acceptance_policy_id": heldout_policy.policy_id,
        "heldout_acceptance_policy_version": heldout_policy.policy_version,
        "heldout_acceptance_policy_sha256": heldout_policy.sha256,
        "heldout_acceptance_decision_sha256": heldout_decision.sha256,
        "score_run_id": "score-runtime-v1",
        "score_run_root_sha256": "8" * 64,
        "scores_sha256": "9" * 64,
        "score_audit_sha256": "a" * 64,
        "score_execution_receipt_sha256": "d" * 64,
        "calibration_publication_id": "calibration-runtime-v1",
        "calibration_publication_root_sha256": calibration_publication_root_sha256,
        "registration_publication_id": "registration-runtime-v1",
        "registration_publication_root_sha256": "e" * 64,
    }
    with AtomicDirectoryPublisher(tmp_path / "validations", validation_id) as publisher:
        publisher.write_json("candidate.json", candidate.to_dict())
        publisher.write_json("deployment_assets.json", deployment_assets.to_dict())
        publisher.write_json("validation.json", validation_record)
        publisher.write_bytes(
            "heldout_acceptance_policy.json", heldout_policy.canonical_bytes()
        )
        publisher.write_bytes(
            "heldout_acceptance.json", heldout_decision.canonical_bytes()
        )
        validation_path = publisher.finalize(
            validator=lambda _staging: None,
            required_paths=frozenset(
                {
                    "candidate.json",
                    "deployment_assets.json",
                    "validation.json",
                    "heldout_acceptance_policy.json",
                    "heldout_acceptance.json",
                }
            ),
        )
    validation_publication = verify_atomic_publication(
        validation_path,
        required_paths=frozenset(
            {
                "candidate.json",
                "deployment_assets.json",
                "validation.json",
                "heldout_acceptance_policy.json",
                "heldout_acceptance.json",
            }
        ),
        allowed_paths=frozenset(
            {
                "candidate.json",
                "deployment_assets.json",
                "validation.json",
                "heldout_acceptance_policy.json",
                "heldout_acceptance.json",
            }
        ),
    )
    promotion_receipt_path = source / "promotion_receipt.json"
    promotion_receipt_path.write_bytes(
        canonical_json_bytes(
            {
                "schema": "zs32.promotion_receipt",
                "schema_version": 2,
                "product": "ZS32",
                "release_id": "release-runtime-v1",
                "promotion_id": "manual-promotion-runtime-v1",
                "approver": "runtime-fixture-approver",
                "approved_at": "2026-07-13T23:59:00Z",
                "candidate_id": candidate.candidate_id,
                "candidate_digest": candidate.digest,
                "validation_publication_id": validation_publication.publication_id,
                "validation_publication_root_sha256": validation_publication.root_sha256,
                "validation_record_sha256": validation_publication.checksums[
                    "validation.json"
                ],
                "contract_sha256": contract.contract_sha256,
                "calibration_artifact_sha256": calibration_digest,
                "dataset_manifest_sha256": dataset_digest,
                "heldout_acceptance_policy_sha256": heldout_policy.sha256,
                "heldout_acceptance_decision_sha256": heldout_decision.sha256,
                "evidence": {
                    role: {
                        "status": "PASS",
                        "evidence_sha256": character * 64,
                        "reason": None,
                    }
                    for role, character in zip(
                        ("golden_parity", "fat", "sat"),
                        "456",
                        strict=True,
                    )
                },
            }
        )
    )
    promotion_receipt_path.chmod(0o444)
    published = assemble_release(
        ReleaseAssemblySpec(
            release_id="release-runtime-v1",
            output_root=tmp_path / "releases",
            created_at="2026-07-14T00:00:00Z",
            promotion_receipt_path=promotion_receipt_path,
            promotion_receipt_sha256=sha256_file(promotion_receipt_path),
            validation_publication_path=validation_publication.root,
            validation_publication_id=validation_publication.publication_id,
            validation_publication_root_sha256=validation_publication.root_sha256,
            validation_record_sha256=validation_publication.checksums["validation.json"],
            heldout_acceptance_policy_sha256=heldout_policy.sha256,
            heldout_acceptance_decision_sha256=heldout_decision.sha256,
            contract=contract,
            assets=deployment_assets,
            fusion_policy_path=fusion_path,
            capture_gate_policy_path=gate_policy_path,
            capture_gate_asset_root=source,
            dataset_release_manifest_path=dataset_path,
            dataset_release_manifest_sha256=dataset_digest,
            code_version={
                "schema_version": 1,
                "git_commit": "1" * 40,
                "git_tree": "2" * 40,
                "dirty": False,
                "dependency_lock_sha256": "3" * 64,
                "build_id": "runtime-contract-test",
            },
            runtime_environment_receipt_path=runtime_environment_path,
            runtime_environment_receipt_sha256=sha256_file(runtime_environment_path),
        )
    )
    try:
        yield published
    finally:
        if published.exists():
            published.chmod(0o700)
            for path in published.rglob("*"):
                if path.is_dir() and not path.is_symlink():
                    path.chmod(0o700)
                elif path.is_file() and not path.is_symlink():
                    path.chmod(0o600)
