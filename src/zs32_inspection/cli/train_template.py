"""Train one hand/view template candidate without fitting its threshold."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from zs32_inspection.config.loaders import load_recipe, load_roi_config, load_topology
from zs32_inspection.config.guards import require_bound_capture_gate_policy
from zs32_inspection.data.dataset_release import verify_dataset_release
from zs32_inspection.domain.identity import Hand
from zs32_inspection.models.base import ModelSlot, sha256_file
from zs32_inspection.runtime.execution_receipt import collect_execution_receipt
from zs32_inspection.template.opencv_backend import OpenCvTemplateTrainingBackend
from zs32_inspection.template.trainer import TemplateTrainer, TemplateTrainSpec

from ._common import command_error, command_result, load_object
from ._environment import require_linux_nvidia


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one ZS32 template candidate slot")
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--roi", type=Path, required=True)
    parser.add_argument("--dataset-release", type=Path, required=True)
    parser.add_argument(
        "--template-export",
        type=Path,
        required=True,
        help="immutable template materialized export built from this dataset release",
    )
    parser.add_argument("--template-export-manifest-sha256", required=True)
    parser.add_argument("--hand", choices=[item.value for item in Hand], required=True)
    parser.add_argument("--view", required=True)
    parser.add_argument("--train-split-id", required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    return parser


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
    bindings = {(item.hand.value, item.view_id) for item in recipe.template_bindings}
    if (hand.value, args.view) not in bindings:
        raise ValueError(f"recipe has no template binding for {slot.key}")
    if args.train_split_id != dataset.split_assignments_sha256:
        raise ValueError(
            "--train-split-id must equal the verified dataset split_assignments_sha256"
        )
    template_export_manifest = args.template_export / "provenance/adapter_manifest.csv"
    if sha256_file(template_export_manifest) != args.template_export_manifest_sha256:
        raise ValueError("template materialized manifest differs from its expected SHA256")
    parameters = load_object(args.parameters, "template training parameters")
    dataset_manifest_path = args.dataset_release / "dataset_release.json"
    execution_receipt = collect_execution_receipt(
        operation="train_template",
        device=args.device,
        input_sha256_by_role={
            "dataset_manifest": sha256_file(dataset_manifest_path),
            "materialized_manifest": args.template_export_manifest_sha256,
            "recipe": sha256_file(args.recipe),
            "roi": sha256_file(args.roi),
            "topology": sha256_file(args.topology),
        },
        parameters=parameters,
    )
    spec = TemplateTrainSpec(
        slot=slot,
        dataset_release_id=dataset.dataset_release_id,
        dataset_manifest_digest=sha256_file(dataset_manifest_path),
        dataset_root=args.template_export,
        train_split_id=args.train_split_id,
        recipe_digest=recipe.recipe_sha256,
        roi_version=roi.roi_config_id,
        roi_digest=roi.roi_sha256,
        output_dir=args.output_dir,
        parameters=parameters,
        device=args.device,
        execution_receipt=execution_receipt.as_dict(),
    )
    artifact = TemplateTrainer(OpenCvTemplateTrainingBackend()).train(spec)
    artifact.verify()
    command_result(
        "zs32-train-template",
        {
            "artifact": {
                "hand": artifact.slot.hand,
                "view": artifact.slot.view,
                "model": artifact.model.to_dict(),
                "templates": [item.to_dict() for item in artifact.templates],
                "model_digest": artifact.model_digest,
                "template_version": artifact.template_version,
                "roi_version": artifact.roi_version,
                "roi_digest": artifact.roi_digest,
                "dataset_release_id": artifact.dataset_release_id,
                "dataset_manifest_digest": artifact.dataset_manifest_digest,
                "train_split_id": artifact.train_split_id,
                "recipe_digest": artifact.recipe_digest,
                "framework_version": artifact.framework_version,
                "training_parameters": dict(artifact.training_parameters),
                "execution_receipt": dict(artifact.execution_receipt),
            },
            "output_dir": str(args.output_dir),
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        require_linux_nvidia()
        return _run(argv)
    except Exception as error:
        return command_error("zs32-train-template", error)


if __name__ == "__main__":
    raise SystemExit(main())
