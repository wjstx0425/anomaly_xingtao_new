"""Register frozen training outputs without changing production state."""

from __future__ import annotations

import argparse
import os
import stat
from collections.abc import Sequence
from pathlib import Path

from zs32_inspection.calibration import CalibrationProvenance, FitParameters
from zs32_inspection.config.loaders import load_recipe, load_roi_config, load_topology
from zs32_inspection.config.guards import require_bound_capture_gate_policy
from zs32_inspection.config.schemas import parse_recipe
from zs32_inspection.data.dataset_release import verify_dataset_release
from zs32_inspection.models.base import ModelSlot, sha256_file
from zs32_inspection.runtime.publisher import (
    AtomicDirectoryPublisher,
    PublicationError,
)

from ._common import command_error, command_result, load_object
from ._deployment_assets import (
    parse_candidate_registration,
    template_assets_payload,
)
from ._environment import require_linux_nvidia


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Register one complete frozen ZS32 model/template candidate"
    )
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--registration-id", required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--base-recipe", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--roi", type=Path, required=True)
    parser.add_argument("--dataset-release", type=Path, required=True)
    parser.add_argument("--calibration-split-id", required=True)
    parser.add_argument("--test-split-id", required=True)
    parser.add_argument("--fit-parameters", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def _identifier(value: str, label: str) -> str:
    if not value.strip() or value != value.strip() or any(character.isspace() for character in value):
        raise ValueError(f"{label} must be non-empty and contain no whitespace")
    return value


def _private_descriptor(path: Path, label: str) -> str:
    metadata = path.stat(follow_symlinks=False)
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.getuid()
    ):
        raise ValueError(f"{label} must be an owned private regular file: {path}")
    return sha256_file(path)


def _verify_private_assets(candidate: object, templates: object) -> None:
    assets = [
        *(artifact.checkpoint for artifact in candidate.anomaly_artifacts.values()),
        *candidate.yolo.assets,
        *(template.model for template in templates.values()),
        *(asset for template in templates.values() for asset in template.templates),
    ]
    for asset in assets:
        metadata = asset.path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.getuid()
        ):
            raise ValueError(f"registration asset must be an owned private regular file: {asset.path}")
        asset.verify()
    candidate.yolo.verify()


def _bound_recipe_payload(
    base: dict[str, object],
    *,
    candidate: object,
    required_slots: tuple[ModelSlot, ...],
    templates: object,
) -> dict[str, object]:
    output = dict(base)
    template_assets: dict[str, dict[str, object]] = {}
    anomaly_models: dict[str, dict[str, object]] = {}
    for slot in required_slots:
        template_assets.setdefault(slot.hand, {})[slot.view] = {
            "artifact_id": f"template-{slot.hand}-{slot.view}",
            "version": templates[slot].template_version,
            "sha256": templates[slot].model_digest,
            "relative_path": f"template/{slot.hand}/{slot.view}/model.json",
        }
        anomaly_models.setdefault(slot.hand, {})[slot.view] = {
            "family": candidate.anomaly_family.value,
            "artifact": {
                "artifact_id": f"{candidate.anomaly_family.value}-{slot.hand}-{slot.view}",
                "version": candidate.candidate_id,
                "sha256": candidate.anomaly_artifacts[slot].model_digest,
                "relative_path": (
                    f"models/anomaly/{candidate.anomaly_family.value}/"
                    f"{slot.hand}/{slot.view}/model.ckpt"
                ),
            },
        }
    output["template_assets"] = template_assets
    output["anomaly_models"] = anomaly_models
    output["yolo_model"] = {
        "artifact_id": "yolo-global",
        "version": candidate.candidate_id,
        "sha256": candidate.yolo.model_digest,
        "relative_path": "models/yolo/best.pt",
    }
    output["template_thresholds"] = []
    output["model_thresholds"] = []
    return output


def _run(argv: Sequence[str] | None) -> int:
    args = _parser().parse_args(argv)
    candidate_id = _identifier(args.candidate_id, "candidate-id")
    registration_id = _identifier(args.registration_id, "registration-id")
    calibration_split_id = _identifier(args.calibration_split_id, "calibration-split-id")
    test_split_id = _identifier(args.test_split_id, "test-split-id")
    if calibration_split_id == test_split_id:
        raise ValueError("calibration and held-out test split IDs must differ")
    source_registration_sha256 = _private_descriptor(
        args.registration,
        "candidate registration descriptor",
    )
    base_payload = load_object(args.base_recipe, "base recipe")
    base_recipe = load_recipe(args.base_recipe)
    require_bound_capture_gate_policy(base_recipe)
    if base_recipe.template_thresholds or base_recipe.model_thresholds:
        raise ValueError("candidate registration requires a pre-calibration recipe with empty thresholds")
    topology = load_topology(args.topology)
    roi = load_roi_config(args.roi)
    dataset = verify_dataset_release(args.dataset_release)
    candidate, required_slots, templates = parse_candidate_registration(
        load_object(args.registration, "candidate registration"),
        asset_root=args.asset_root,
        candidate_id=candidate_id,
    )
    expected_slots = tuple(
        ModelSlot(hand.value, view)
        for hand in sorted(base_recipe.allowed_hands, key=lambda item: item.value)
        for view in topology.required_views
    )
    base_template_slots = {
        ModelSlot(item.hand.value, item.view_id) for item in base_recipe.template_bindings
    }
    base_anomaly_slots = {
        ModelSlot(item.hand.value, item.view_id) for item in base_recipe.anomaly_bindings
    }
    if (
        required_slots != expected_slots
        or base_template_slots != set(expected_slots)
        or base_anomaly_slots != set(expected_slots)
    ):
        raise ValueError("registration/base recipe slots differ from enabled hands and topology")
    if (
        candidate.recipe_digest != base_recipe.recipe_sha256
        or candidate.topology_digest != topology.topology_sha256
        or candidate.roi_digest != roi.roi_sha256
        or candidate.roi_version != roi.roi_config_id
        or candidate.anomaly_family is not base_recipe.anomaly_family
        or base_recipe.topology_id != topology.topology_id
        or base_recipe.roi_config_id != roi.roi_config_id
    ):
        raise ValueError("registration identity differs from base recipe/topology/ROI")
    for hand in base_recipe.allowed_hands:
        roi.require_ready(hand, topology.required_views)
    dataset_manifest_digest = sha256_file(args.dataset_release / "dataset_release.json")
    if (
        candidate.dataset_release_id != dataset.dataset_release_id
        or candidate.dataset_manifest_digest != dataset_manifest_digest
        or dataset.topology_sha256 != topology.topology_sha256
        or dataset.roi_sha256 != roi.roi_sha256
        or dataset.capture_gate_policy_sha256
        != base_recipe.capture_gate_policy.sha256
        or {hand.value for hand in base_recipe.allowed_hands} != set(dataset.hands)
    ):
        raise ValueError("registration dataset provenance differs from candidate/configuration")
    if any(
        candidate.anomaly_artifacts[slot].train_split_id != dataset.split_assignments_sha256
        or templates[slot].train_split_id != dataset.split_assignments_sha256
        for slot in required_slots
    ):
        raise ValueError("registered model train split differs from dataset split assignments")
    _verify_private_assets(candidate, templates)
    bound_payload = _bound_recipe_payload(
        base_payload,
        candidate=candidate,
        required_slots=required_slots,
        templates=templates,
    )
    bound_recipe = parse_recipe(bound_payload)
    if bound_recipe.recipe_sha256 != base_recipe.recipe_sha256:
        raise RuntimeError("binding trained assets changed the stable recipe identity")
    template_payload = template_assets_payload(candidate, required_slots, templates)
    fit_payload = load_object(args.fit_parameters, "calibration fit parameters")
    if set(fit_payload) != {
        "target_defect_recall", "normal_quantile", "min_normal_parts", "min_defect_parts"
    }:
        raise ValueError("calibration fit parameter keys are incomplete or unknown")
    fit_parameters = FitParameters(**fit_payload)
    provenance = CalibrationProvenance(
        recipe_digest=candidate.recipe_digest,
        profile_digest=base_recipe.fusion_policy_sha256,
        topology_digest=candidate.topology_digest,
        roi_digest=candidate.roi_digest,
        dataset_release_id=candidate.dataset_release_id,
        dataset_manifest_digest=candidate.dataset_manifest_digest,
        calibration_split_id=calibration_split_id,
        test_split_id=test_split_id,
        model_digests=tuple(
            {
                candidate.yolo.model_digest,
                *(candidate.anomaly_artifacts[slot].model_digest for slot in required_slots),
                *(templates[slot].model_digest for slot in required_slots),
            }
        ),
    )
    calibration_spec = {
        "schema_version": 1,
        "provenance": provenance.to_dict(),
        "fit_parameters": fit_parameters.to_dict(),
        "required_dual_groups": [
            {
                "hand": slot.hand,
                "view": slot.view,
                "branch": branch,
                "model_digest": (
                    candidate.anomaly_artifacts[slot].model_digest
                    if branch == "anomaly"
                    else candidate.yolo.model_digest
                ),
                "roi_version": candidate.roi_version,
            }
            for slot in required_slots
            for branch in ("anomaly", "yolo")
        ],
        "required_template_groups": [
            {
                "hand": slot.hand,
                "view": slot.view,
                "model_digest": templates[slot].model_digest,
                "roi_version": candidate.roi_version,
                "roi_digest": candidate.roi_digest,
            }
            for slot in required_slots
        ],
    }
    registration_record = {
        "schema": "zs32.candidate_registration_record",
        "schema_version": 1,
        "registration_id": registration_id,
        "candidate_id": candidate.candidate_id,
        "candidate_digest": candidate.digest,
        "candidate_status": candidate.status.value,
        "dataset_release_id": candidate.dataset_release_id,
        "dataset_manifest_sha256": candidate.dataset_manifest_digest,
        "train_split_id": dataset.split_assignments_sha256,
        "calibration_split_id": calibration_split_id,
        "test_split_id": test_split_id,
        "recipe_sha256": candidate.recipe_digest,
        "topology_sha256": candidate.topology_digest,
        "roi_sha256": candidate.roi_digest,
        "anomaly_family": candidate.anomaly_family.value,
        "required_slots": [slot.key for slot in required_slots],
        "source_registration_sha256": source_registration_sha256,
        "promotion_status": "not_promoted",
    }

    def validate(staging: Path) -> None:
        expected = {
            "candidate.json": candidate.to_dict(),
            "template_assets.json": template_payload,
            "bound_recipe.json": bound_recipe.as_dict(),
            "calibration_spec.json": calibration_spec,
            "registration.json": registration_record,
        }
        for name, payload in expected.items():
            if load_object(staging / name, f"staged {name}") != payload:
                raise PublicationError(f"staged registration output changed: {name}")

    with AtomicDirectoryPublisher(args.output_root, registration_id) as publisher:
        if sha256_file(args.registration) != source_registration_sha256:
            raise ValueError("candidate registration descriptor changed during validation")
        publisher.write_json("candidate.json", candidate.to_dict())
        publisher.write_json("template_assets.json", template_payload)
        publisher.write_json("bound_recipe.json", bound_recipe.as_dict())
        publisher.write_json("calibration_spec.json", calibration_spec)
        publisher.write_json("registration.json", registration_record)
        published = publisher.finalize(
            validator=validate,
            required_paths=frozenset(
                {
                    "candidate.json", "template_assets.json", "bound_recipe.json",
                    "calibration_spec.json", "registration.json",
                }
            ),
        )
    command_result(
        "zs32-register-candidate",
        {
            "candidate_digest": candidate.digest,
            "recipe_sha256": candidate.recipe_digest,
            "published_path": str(published),
            "promotion_status": "not_promoted",
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        require_linux_nvidia()
        return _run(argv)
    except Exception as error:
        return command_error("zs32-register-candidate", error)


if __name__ == "__main__":
    raise SystemExit(main())
