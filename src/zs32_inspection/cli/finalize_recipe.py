"""Materialize calibrated threshold records into a final recipe snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path

from zs32_inspection.config.schemas import parse_recipe
from zs32_inspection.config.guards import require_bound_capture_gate_policy
from zs32_inspection.runtime.publisher import canonical_json_bytes, verify_atomic_publication
from zs32_inspection.runtime.release_loader import reconstruct_calibration_artifact

from ._common import array_value, command_error, command_result, load_object, object_value, require_keys
from ._environment import require_linux_nvidia


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finalize one ZS32 recipe from calibration outputs")
    parser.add_argument("--base-recipe", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


_CALIBRATION_FILES = frozenset(
    {
        "calibration_artifact.json",
        "template_thresholds.json",
        "model_thresholds.json",
        "metrics.json",
        "input_provenance.json",
    }
)


def _object(content: bytes, label: str) -> dict[str, object]:
    try:
        payload = json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON: {error}") from error
    if not isinstance(payload, dict) or canonical_json_bytes(payload) != content:
        raise ValueError(f"{label} must be a canonical JSON object")
    return payload


def _write_no_replace(path: Path, payload: bytes) -> None:
    path = path.expanduser()
    if path.suffix.lower() != ".json" or path.is_symlink() or path.parent.is_symlink():
        raise ValueError("final recipe output must be a non-symlink .json path")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        0o440,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _run(argv: Sequence[str] | None) -> int:
    args = _parser().parse_args(argv)
    base = load_object(args.base_recipe, "base recipe")
    required_base = {
        "schema_version", "recipe_id", "product", "topology_id", "roi_version",
        "allowed_hands", "anomaly_family", "template_assets", "anomaly_models",
        "capture_gate_policy", "yolo_model", "template_thresholds", "model_thresholds",
        "fusion_policy_sha256",
    }
    require_keys(base, required=tuple(required_base), context="base recipe")
    parsed_base = parse_recipe(base)
    require_bound_capture_gate_policy(parsed_base)
    if parsed_base.template_thresholds or parsed_base.model_thresholds:
        raise ValueError("base recipe must be the bound pre-calibration recipe with empty thresholds")
    stable_recipe_digest = parsed_base.recipe_sha256

    calibration = verify_atomic_publication(
        args.calibration,
        required_paths=_CALIBRATION_FILES,
        allowed_paths=_CALIBRATION_FILES,
    )
    artifact_bytes = calibration.read_bytes("calibration_artifact.json")
    artifact = _object(artifact_bytes, "calibration artifact")
    artifact_digest = hashlib.sha256(artifact_bytes).hexdigest()
    reconstructed = reconstruct_calibration_artifact(artifact)
    if reconstructed.payload() != artifact or not reconstructed.calibration_valid:
        raise ValueError("calibration artifact is non-deployable")
    provenance = object_value(artifact.get("provenance"), "calibration provenance")
    if (
        provenance.get("recipe_digest") != stable_recipe_digest
        or provenance.get("profile_digest") != base["fusion_policy_sha256"]
    ):
        raise ValueError("calibration artifact differs from base recipe identity/profile")

    template_file = _object(
        calibration.read_bytes("template_thresholds.json"),
        "template thresholds",
    )
    model_file = _object(
        calibration.read_bytes("model_thresholds.json"),
        "model thresholds",
    )
    for label, payload in (("template", template_file), ("model", model_file)):
        require_keys(
            payload,
            required=("schema_version", "calibration_artifact_sha256", "thresholds"),
            context=f"{label} threshold file",
        )
        if payload["schema_version"] != 1 or payload["calibration_artifact_sha256"] != artifact_digest:
            raise ValueError(f"{label} threshold file does not bind calibration artifact bytes")
    if template_file["thresholds"] != artifact["template_thresholds"]:
        raise ValueError("template threshold snapshot differs from calibration artifact")
    if model_file["thresholds"] != artifact["dual_thresholds"]:
        raise ValueError("model threshold snapshot differs from calibration artifact")
    metrics = _object(calibration.read_bytes("metrics.json"), "held-out metrics")
    if metrics != artifact["heldout_metrics"]:
        raise ValueError("held-out metrics snapshot differs from calibration artifact")
    input_provenance = _object(
        calibration.read_bytes("input_provenance.json"),
        "calibration input provenance",
    )
    if (
        input_provenance.get("recipe_digest") != stable_recipe_digest
        or input_provenance.get("dataset_release_id")
        != provenance.get("dataset_release_id")
        or input_provenance.get("dataset_manifest_sha256")
        != provenance.get("dataset_manifest_digest")
        or input_provenance.get("calibration_split_id")
        != provenance.get("calibration_split_id")
        or input_provenance.get("test_split_id") != provenance.get("test_split_id")
    ):
        raise ValueError("calibration input provenance differs from artifact/base recipe")

    template_records = []
    for index, raw in enumerate(array_value(template_file["thresholds"], "template thresholds")):
        item = object_value(raw, f"template thresholds[{index}]")
        required = {
            "hand", "view", "model_digest", "roi_version", "roi_digest",
            "dataset_release_id", "calibration_split_id", "threshold", "normal_count",
            "defect_count", "status",
        }
        require_keys(item, required=tuple(required), context=f"template thresholds[{index}]")
        if item["status"] != "ok" or item["threshold"] is None:
            raise ValueError(f"template threshold group is not deployable at index {index}")
        template_records.append(
            {
                "hand": item["hand"],
                "view": item["view"],
                "threshold": item["threshold"],
                "template_sha256": item["model_digest"],
                "calibration_sha256": artifact_digest,
            }
        )

    model_records = []
    for index, raw in enumerate(array_value(model_file["thresholds"], "model thresholds")):
        item = object_value(raw, f"model thresholds[{index}]")
        required = {
            "dataset_release_id", "calibration_split_id", "hand", "view", "branch",
            "model_digest", "roi_version", "low", "high", "normal_count",
            "defect_count", "status",
        }
        require_keys(item, required=tuple(required), context=f"model thresholds[{index}]")
        if item["status"] != "ok" or item["low"] is None or item["high"] is None:
            raise ValueError(f"model threshold group is not deployable at index {index}")
        model_records.append(
            {
                "hand": item["hand"],
                "view": item["view"],
                "branch": item["branch"],
                "low": item["low"],
                "high": item["high"],
                "model_sha256": item["model_digest"],
                "roi_version": item["roi_version"],
                "calibration_sha256": artifact_digest,
            }
        )

    final_payload = dict(base)
    final_payload["template_thresholds"] = template_records
    final_payload["model_thresholds"] = model_records
    recipe = parse_recipe(final_payload)
    if recipe.recipe_sha256 != stable_recipe_digest:
        raise RuntimeError("final recipe changed its stable pre-calibration identity")
    reverified = verify_atomic_publication(
        args.calibration,
        required_paths=_CALIBRATION_FILES,
        allowed_paths=_CALIBRATION_FILES,
    )
    if reverified.root_sha256 != calibration.root_sha256:
        raise ValueError("calibration publication changed while recipe was finalized")
    _write_no_replace(args.output, canonical_json_bytes(recipe.as_dict()))
    command_result(
        "zs32-finalize-recipe",
        {
            "calibration_artifact_sha256": artifact_digest,
            "recipe_sha256": recipe.recipe_sha256,
            "output": str(args.output),
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        require_linux_nvidia()
        return _run(argv)
    except Exception as error:
        return command_error("zs32-finalize-recipe", error)


if __name__ == "__main__":
    raise SystemExit(main())
