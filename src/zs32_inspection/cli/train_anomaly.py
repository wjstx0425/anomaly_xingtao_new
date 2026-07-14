"""Train one hand/view anomaly candidate without promoting production state."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

from zs32_inspection.config.loaders import load_recipe, load_roi_config, load_topology
from zs32_inspection.config.guards import require_bound_capture_gate_policy
from zs32_inspection.data.dataset_release import verify_dataset_release
from zs32_inspection.domain.contracts import AnomalyFamily
from zs32_inspection.domain.identity import Hand
from zs32_inspection.models import (
    AnomalyDINOAdapter,
    AnomalibTrainerBackend,
    DeviceSpec,
    EfficientADAdapter,
    ModelSlot,
    PatchCoreAdapter,
    TrainSpec,
)
from zs32_inspection.models.base import require_sha256, sha256_file
from zs32_inspection.runtime.execution_receipt import collect_execution_receipt

from ._common import command_error, command_result, load_object
from ._environment import require_linux_nvidia


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one ZS32 anomaly candidate slot")
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--roi", type=Path, required=True)
    parser.add_argument("--dataset-release", type=Path, required=True)
    parser.add_argument("--anomalib-export", type=Path, required=True)
    parser.add_argument("--anomalib-export-manifest-sha256", required=True)
    parser.add_argument("--hand", choices=[item.value for item in Hand], required=True)
    parser.add_argument("--view", required=True)
    parser.add_argument("--train-split-id", required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    return parser


def _adapter(family: AnomalyFamily):
    """Return the family adapter with the concrete Linux Anomalib trainer."""
    return {
        AnomalyFamily.PATCHCORE: PatchCoreAdapter,
        AnomalyFamily.EFFICIENTAD: EfficientADAdapter,
        AnomalyFamily.ANOMALYDINO: AnomalyDINOAdapter,
    }[family](trainer=AnomalibTrainerBackend())


def _run(argv: Sequence[str] | None) -> int:
    args = _parser().parse_args(argv)
    recipe = load_recipe(args.recipe)
    require_bound_capture_gate_policy(recipe)
    topology = load_topology(args.topology)
    roi = load_roi_config(args.roi)
    dataset = verify_dataset_release(args.dataset_release)
    hand = Hand.parse(args.hand)
    slot = ModelSlot(hand.value, args.view)
    if recipe.topology_id != topology.topology_id or recipe.roi_config_id != roi.roi_config_id:
        raise ValueError("recipe/topology/ROI identity mismatch")
    if hand not in recipe.allowed_hands or args.view not in topology.required_views:
        raise ValueError(f"training slot is not enabled by recipe/topology: {slot.key}")
    roi.require_ready(hand, topology.required_views)
    if (
        dataset.topology_sha256 != topology.topology_sha256
        or dataset.roi_sha256 != roi.roi_sha256
        or dataset.capture_gate_policy_sha256 != recipe.capture_gate_policy.sha256
        or {item.value for item in recipe.allowed_hands} != set(dataset.hands)
    ):
        raise ValueError("dataset release provenance differs from training topology/ROI/gates/hands")
    bindings = {
        (item.hand.value, item.view_id, item.family)
        for item in recipe.anomaly_bindings
    }
    if (hand.value, args.view, recipe.anomaly_family) not in bindings:
        raise ValueError(f"recipe has no anomaly binding for {slot.key}")
    if args.train_split_id != dataset.split_assignments_sha256:
        raise ValueError(
            "--train-split-id must equal the verified dataset split_assignments_sha256"
        )
    manifest_path = args.dataset_release / "dataset_release.json"
    parameters = load_object(args.parameters, "anomaly training parameters")
    input_sha256_by_role = {
        "dataset_manifest": sha256_file(manifest_path),
        "materialized_manifest": args.anomalib_export_manifest_sha256,
        "recipe": sha256_file(args.recipe),
        "roi": sha256_file(args.roi),
        "topology": sha256_file(args.topology),
    }
    if recipe.anomaly_family is AnomalyFamily.PATCHCORE:
        auxiliary = parameters.get("auxiliary")
        if not isinstance(auxiliary, Mapping):
            raise ValueError("PatchCore parameters.auxiliary must be an object")
        backbone_path_value = auxiliary.get("backbone_weights_path")
        if not isinstance(backbone_path_value, str) or not backbone_path_value.strip():
            raise ValueError("PatchCore backbone_weights_path must be non-empty")
        backbone_path = Path(backbone_path_value).expanduser()
        expected_backbone = require_sha256(
            auxiliary.get("backbone_weights_sha256"),
            "auxiliary.backbone_weights_sha256",
        )
        actual_backbone = sha256_file(backbone_path)
        if actual_backbone != expected_backbone:
            raise ValueError("PatchCore backbone weights digest mismatch before receipt capture")
        input_sha256_by_role["patchcore_backbone"] = actual_backbone
    execution_receipt = collect_execution_receipt(
        operation="train_anomaly",
        device=args.device,
        input_sha256_by_role=input_sha256_by_role,
        parameters=parameters,
    )
    spec = TrainSpec(
        family=recipe.anomaly_family,
        slot=slot,
        dataset_release_id=dataset.dataset_release_id,
        dataset_manifest_digest=sha256_file(manifest_path),
        train_split_id=args.train_split_id,
        recipe_digest=recipe.recipe_sha256,
        roi_version=roi.roi_config_id,
        roi_digest=roi.roi_sha256,
        materialized_export_root=args.anomalib_export,
        materialized_manifest_digest=args.anomalib_export_manifest_sha256,
        device=DeviceSpec(accelerator="gpu", device=args.device),
        output_dir=args.output_dir,
        parameters=parameters,
        execution_receipt=execution_receipt.as_dict(),
    )
    artifact = _adapter(recipe.anomaly_family).train(spec)
    artifact.verify()
    candidate_root = artifact.checkpoint.path.parent.parent
    metadata_files = sorted(
        path.relative_to(candidate_root).as_posix()
        for path in (candidate_root / "metadata").glob("*.json")
    )
    if len(metadata_files) != 1:
        raise ValueError("published anomaly candidate must contain exactly one metadata document")
    command_result(
        "zs32-train-anomaly",
        {
            "artifact": artifact.to_dict(),
            "candidate_root": str(candidate_root),
            "metadata_path": metadata_files[0],
            "promotion_status": "candidate_only",
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        require_linux_nvidia()
        return _run(argv)
    except Exception as error:
        return command_error("zs32-train-anomaly", error)


if __name__ == "__main__":
    raise SystemExit(main())
