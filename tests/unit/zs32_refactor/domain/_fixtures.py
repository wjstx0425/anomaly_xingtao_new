"""Pure mapping fixtures for strict ZS32 contract tests."""

from __future__ import annotations

import hashlib
from typing import Any

from zs32_inspection.domain.contracts import RoiConfig
from zs32_inspection.domain.topology import CaptureTopology

FUSION_POLICY_BYTES = (
    b'{"all_required_clear_for_ok":true,"no_majority_vote":true,'
    b'"policy_id":"zs32-strict-fusion-v1","schema_version":1,'
    b'"strong_priority":["anomaly","yolo"]}'
)

DIGESTS = {
    "template": "1" * 64,
    "anomaly": "2" * 64,
    "yolo": "3" * 64,
    "template_calibration": "4" * 64,
    "model_calibration": "5" * 64,
    "fusion": hashlib.sha256(FUSION_POLICY_BYTES).hexdigest(),
    "capture_gate_policy": "6" * 64,
}


def topology_mapping(camera_count: int) -> dict[str, Any]:
    """Build a two-round 3/4/5-camera topology mapping."""
    rounds = [
        {"round_id": "front", "prompt": "capture_front"},
        {"round_id": "back", "prompt": "flip_then_capture_back"},
    ]
    slots = []
    required_views = []
    for camera_index in range(camera_count):
        front_view = f"front_{camera_index}"
        back_view = f"back_{camera_index}"
        slots.append(
            {
                "slot_id": f"slot_{camera_index}",
                "serial": f"SERIAL_{camera_index}",
                "views": {"front": front_view, "back": back_view},
            },
        )
        required_views.extend((front_view, back_view))
    return {
        "schema_version": 1,
        "topology_id": f"zs32-{camera_count}cam-double-side-v1",
        "product": "ZS32",
        "rounds": rounds,
        "camera_slots": slots,
        "required_views": required_views,
    }


def roi_mapping(topology: CaptureTopology, *, left_ready: bool = False) -> dict[str, Any]:
    """Build a v2 ROI mapping with independently controlled hand readiness."""
    right_views = {
        view_id: {"xyxy": [10, 20, 900, 700]}
        for view_id in topology.required_views
    }
    left_views = {
        view_id: {"xyxy": [20, 30, 880, 690]}
        for view_id in topology.required_views
    }
    left: dict[str, Any]
    if left_ready:
        left = {"status": "ready", "views": left_views}
    else:
        left = {"status": "pending", "reason": "left ROI is not approved", "views": {}}
    return {
        "schema_version": 2,
        "roi_version": f"{topology.topology_id}-roi-v2",
        "product": "ZS32",
        "topology_id": topology.topology_id,
        "coordinate_system": "pixel_xyxy_half_open",
        "source_image_size": {"width": 1000, "height": 800},
        "hands": {
            "right": {"status": "ready", "views": right_views},
            "left": left,
        },
    }


def recipe_mapping(
    topology: CaptureTopology,
    roi: RoiConfig,
    *,
    hands: tuple[str, ...] = ("right",),
) -> dict[str, Any]:
    """Build a complete recipe for all enabled hand/view/groups."""
    template_assets: dict[str, dict[str, Any]] = {}
    anomaly_models: dict[str, dict[str, Any]] = {}
    template_thresholds: list[dict[str, Any]] = []
    model_thresholds: list[dict[str, Any]] = []
    for hand in hands:
        template_assets[hand] = {}
        anomaly_models[hand] = {}
        for view_id in topology.required_views:
            template_assets[hand][view_id] = {
                "artifact_id": f"template-{hand}-{view_id}",
                "version": "template-v1",
                "sha256": DIGESTS["template"],
                "relative_path": f"template/{hand}/{view_id}/model.json",
            }
            anomaly_models[hand][view_id] = {
                "family": "patchcore",
                "artifact": {
                    "artifact_id": f"patchcore-{hand}-{view_id}",
                    "version": "patchcore-v1",
                    "sha256": DIGESTS["anomaly"],
                    "relative_path": f"models/anomaly/patchcore/{hand}/{view_id}/model.ckpt",
                },
            }
            template_thresholds.append(
                {
                    "hand": hand,
                    "view": view_id,
                    "threshold": 0.5,
                    "template_sha256": DIGESTS["template"],
                    "calibration_sha256": DIGESTS["template_calibration"],
                },
            )
            for branch, model_digest in (
                ("anomaly", DIGESTS["anomaly"]),
                ("yolo", DIGESTS["yolo"]),
            ):
                model_thresholds.append(
                    {
                        "hand": hand,
                        "view": view_id,
                        "branch": branch,
                        "low": 0.2,
                        "high": 0.8,
                        "model_sha256": model_digest,
                        "roi_version": roi.roi_config_id,
                        "calibration_sha256": DIGESTS["model_calibration"],
                    },
                )
    return {
        "schema_version": 1,
        "recipe_id": f"recipe-{topology.topology_id}",
        "product": "ZS32",
        "topology_id": topology.topology_id,
        "roi_version": roi.roi_config_id,
        "allowed_hands": list(hands),
        "anomaly_family": "patchcore",
        "capture_gate_policy": {
            "artifact_id": "zs32-capture-gate-policy",
            "version": "capture-gates-v1",
            "sha256": DIGESTS["capture_gate_policy"],
            "relative_path": "capture/gates/policy.json",
        },
        "template_assets": template_assets,
        "anomaly_models": anomaly_models,
        "yolo_model": {
            "artifact_id": "zs32-global-yolo",
            "version": "yolo-v1",
            "sha256": DIGESTS["yolo"],
            "relative_path": "models/yolo/best.pt",
        },
        "template_thresholds": template_thresholds,
        "model_thresholds": model_thresholds,
        "fusion_policy_sha256": DIGESTS["fusion"],
    }
