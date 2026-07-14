"""Build one deterministic registration descriptor from frozen model metadata."""

from __future__ import annotations

import argparse
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path

from zs32_inspection.config.loaders import load_recipe, load_roi_config, load_topology
from zs32_inspection.config.guards import require_bound_capture_gate_policy
from zs32_inspection.data.dataset_release import verify_dataset_release
from zs32_inspection.models.base import ModelSlot, sha256_file
from zs32_inspection.runtime.publisher import canonical_json_bytes

from ._common import command_error, command_result, load_object, resolved_asset
from ._deployment_assets import parse_candidate_registration
from ._environment import require_linux_nvidia


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a strict ZS32 candidate-registration descriptor",
    )
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--base-recipe", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--roi", type=Path, required=True)
    parser.add_argument("--dataset-release", type=Path, required=True)
    parser.add_argument(
        "--anomaly-metadata",
        type=Path,
        action="append",
        required=True,
        help="Repeat once for every required hand/view anomaly metadata JSON",
    )
    parser.add_argument(
        "--template-model",
        type=Path,
        action="append",
        required=True,
        help="Repeat once for every required hand/view template model.json",
    )
    parser.add_argument("--yolo-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _private_regular(path: Path, field: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink() or not expanded.is_file():
        raise ValueError(f"{field} must be a regular non-symlink file: {expanded}")
    metadata = expanded.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(f"{field} must be a private regular file: {expanded}")
    return expanded.resolve()


def _asset_root(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink() or not expanded.is_dir():
        raise ValueError(f"asset root must be a regular non-symlink directory: {expanded}")
    return expanded.resolve()


def _global_asset(
    payload: object,
    *,
    publication_root: Path,
    asset_root: Path,
    field: str,
) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field} must be an object")
    item = dict(payload)
    if set(item) != {"role", "relative_path", "sha256", "media_type"}:
        raise ValueError(f"{field} asset fields are incomplete or unknown")
    relative = item["relative_path"]
    if not isinstance(relative, str):
        raise ValueError(f"{field}.relative_path must be a string")
    resolved = resolved_asset(publication_root, relative, field)
    _private_regular(resolved, field)
    try:
        global_relative = resolved.relative_to(asset_root).as_posix()
    except ValueError as error:
        raise ValueError(f"{field} is outside the declared asset root") from error
    if sha256_file(resolved) != item["sha256"]:
        raise ValueError(f"{field} SHA-256 differs from frozen metadata")
    return {**item, "relative_path": global_relative}


def _anomaly_row(path: Path, asset_root: Path) -> dict[str, object]:
    metadata_path = _private_regular(path, "anomaly metadata")
    payload = load_object(metadata_path, "anomaly metadata")
    expected = {
        "schema",
        "schema_version",
        "family",
        "hand",
        "view",
        "checkpoint",
        "model_digest",
        "dataset_release_id",
        "dataset_manifest_digest",
        "train_split_id",
        "recipe_digest",
        "roi_version",
        "roi_digest",
        "framework_version",
        "training_parameters",
        "execution_receipt",
        "promotion_status",
    }
    if set(payload) != expected or (
        payload["schema"], payload["schema_version"], payload["promotion_status"]
    ) != ("zs32.anomaly_model_artifact", 1, "candidate_only"):
        raise ValueError(f"anomaly metadata schema is invalid: {metadata_path}")
    publication_root = metadata_path.parent.parent
    row = {
        key: value
        for key, value in payload.items()
        if key not in {"schema", "schema_version", "promotion_status"}
    }
    row["checkpoint"] = _global_asset(
        row["checkpoint"],
        publication_root=publication_root,
        asset_root=asset_root,
        field="anomaly checkpoint",
    )
    return row


def _template_row(path: Path, asset_root: Path) -> dict[str, object]:
    model_path = _private_regular(path, "template model")
    payload = load_object(model_path, "template model")
    expected = {
        "schema",
        "schema_version",
        "hand",
        "view",
        "method",
        "risk",
        "preprocessing",
        "references",
        "template_version",
        "roi_version",
        "roi_sha256",
        "dataset_release_id",
        "dataset_manifest_sha256",
        "train_split_id",
        "recipe_sha256",
        "train_part_count",
        "framework_version",
        "training_parameters",
        "execution_receipt",
    }
    if set(payload) != expected or (
        payload["schema"], payload["schema_version"]
    ) != ("zs32.template_model", 1):
        raise ValueError(f"template model schema is invalid: {model_path}")
    references = payload["references"]
    if not isinstance(references, list) or not references:
        raise ValueError(f"template model has no frozen references: {model_path}")
    publication_root = model_path.parent
    template_assets: list[dict[str, object]] = []
    for expected_index, reference in enumerate(references, start=1):
        if not isinstance(reference, Mapping) or set(reference) != {
            "index",
            "artifact_sha256",
            "source_crop_sha256",
        }:
            raise ValueError(f"template reference metadata is malformed: {model_path}")
        if reference["index"] != expected_index:
            raise ValueError(f"template reference indices are not canonical: {model_path}")
        template_assets.append(
            _global_asset(
                {
                    "role": "reference_template",
                    "relative_path": f"templates/template_{expected_index:03d}.png",
                    "sha256": reference["artifact_sha256"],
                    "media_type": "image/png",
                },
                publication_root=publication_root,
                asset_root=asset_root,
                field=f"template reference {expected_index}",
            )
        )
    model_asset = _global_asset(
        {
            "role": "template_model",
            "relative_path": model_path.name,
            "sha256": sha256_file(model_path),
            "media_type": "application/json",
        },
        publication_root=publication_root,
        asset_root=asset_root,
        field="template model",
    )
    return {
        "hand": payload["hand"],
        "view": payload["view"],
        "model": model_asset,
        "templates": template_assets,
        "model_digest": model_asset["sha256"],
        "template_version": payload["template_version"],
        "roi_version": payload["roi_version"],
        "roi_digest": payload["roi_sha256"],
        "dataset_release_id": payload["dataset_release_id"],
        "dataset_manifest_digest": payload["dataset_manifest_sha256"],
        "train_split_id": payload["train_split_id"],
        "recipe_digest": payload["recipe_sha256"],
        "framework_version": payload["framework_version"],
        "training_parameters": payload["training_parameters"],
        "execution_receipt": payload["execution_receipt"],
    }


def _yolo_payload(path: Path, asset_root: Path) -> dict[str, object]:
    metadata_path = _private_regular(path, "YOLO metadata")
    payload = load_object(metadata_path, "YOLO metadata")
    assets = payload.get("assets")
    if not isinstance(assets, list):
        raise ValueError("YOLO metadata assets must be an array")
    publication_root = metadata_path.parent
    return {
        **payload,
        "assets": [
            _global_asset(
                item,
                publication_root=publication_root,
                asset_root=asset_root,
                field=f"YOLO asset {index}",
            )
            for index, item in enumerate(assets)
        ],
    }


def _write_private_no_replace(path: Path, payload: Mapping[str, object]) -> None:
    output = path.expanduser()
    parent = output.parent
    if parent.is_symlink():
        raise ValueError(f"output parent must not be a symlink: {parent}")
    parent.mkdir(parents=True, exist_ok=True)
    if not parent.is_dir():
        raise ValueError(f"output parent is not a directory: {parent}")
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        content = canonical_json_bytes(dict(payload))
        written = 0
        while written < len(content):
            count = os.write(descriptor, content[written:])
            if count <= 0:
                raise OSError("candidate registration write made no progress")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _run(argv: Sequence[str] | None) -> int:
    args = _parser().parse_args(argv)
    asset_root = _asset_root(args.asset_root)
    recipe = load_recipe(args.base_recipe)
    require_bound_capture_gate_policy(recipe)
    if recipe.template_thresholds or recipe.model_thresholds:
        raise ValueError("candidate registration requires a pre-calibration recipe")
    topology = load_topology(args.topology)
    roi = load_roi_config(args.roi)
    dataset = verify_dataset_release(args.dataset_release)
    if (
        recipe.topology_id != topology.topology_id
        or recipe.roi_config_id != roi.roi_config_id
        or roi.topology_id != topology.topology_id
        or dataset.topology_sha256 != topology.topology_sha256
        or dataset.roi_sha256 != roi.roi_sha256
        or dataset.capture_gate_policy_sha256 != recipe.capture_gate_policy.sha256
        or set(dataset.hands) != {hand.value for hand in recipe.allowed_hands}
    ):
        raise ValueError("recipe/topology/ROI/dataset identity is inconsistent")
    required_slots = tuple(
        ModelSlot(hand.value, view)
        for hand in sorted(recipe.allowed_hands, key=lambda item: item.value)
        for view in topology.required_views
    )
    for hand in recipe.allowed_hands:
        roi.require_ready(hand, topology.required_views)
    anomaly_rows = [_anomaly_row(path, asset_root) for path in args.anomaly_metadata]
    template_rows = [_template_row(path, asset_root) for path in args.template_model]
    payload = {
        "schema": "zs32.candidate_registration",
        "schema_version": 1,
        "product": "ZS32",
        "anomaly_family": recipe.anomaly_family.value,
        "anomaly_artifacts": anomaly_rows,
        "yolo": _yolo_payload(args.yolo_metadata, asset_root),
        "recipe_digest": recipe.recipe_sha256,
        "topology_digest": topology.topology_sha256,
        "roi_digest": roi.roi_sha256,
        "roi_version": roi.roi_config_id,
        "dataset_release_id": dataset.dataset_release_id,
        "dataset_manifest_digest": sha256_file(
            args.dataset_release / "dataset_release.json"
        ),
        "required_slots": [slot.key for slot in required_slots],
        "templates": template_rows,
    }
    candidate, parsed_slots, parsed_templates = parse_candidate_registration(
        payload,
        asset_root=asset_root,
        candidate_id="candidate-registration-preflight",
    )
    if parsed_slots != required_slots or set(parsed_templates) != set(required_slots):
        raise ValueError("generated candidate registration differs from recipe/topology slots")
    if candidate.anomaly_family is not recipe.anomaly_family:
        raise ValueError("generated candidate registration changed anomaly family")
    _write_private_no_replace(args.output, payload)
    command_result(
        "zs32-build-candidate-registration",
        {
            "output": str(args.output.expanduser().resolve()),
            "output_sha256": sha256_file(args.output),
            "required_slots": [slot.key for slot in required_slots],
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        require_linux_nvidia()
        return _run(argv)
    except Exception as error:
        return command_error("zs32-build-candidate-registration", error)


if __name__ == "__main__":
    raise SystemExit(main())
