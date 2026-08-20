# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Publish and validate immutable ZS32 commissioning runtime bundles."""

# Validation paths intentionally raise detailed fail-closed errors with asset context.
# ruff: noqa: C901, EM101, EM102, TC003, TRY003, TRY301

from __future__ import annotations

import csv
import ctypes
import errno
import hashlib
import json
import math
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from capture_data.fusion_calibration import THRESHOLD_ARTIFACT_SCHEMA, THRESHOLD_ARTIFACT_VERSION
from capture_data.zs32_model_runtime import load_runtime_config, sha256_file

from zs32_inspection.domain.views import VIEW_ORDER

REPO_ROOT = Path(__file__).resolve().parents[1]
SHA256_LENGTH = 64
SCHEMA_VERSION = 1
AT_FDCWD = -100
RENAME_NOREPLACE = 1


@dataclass(frozen=True, slots=True)
class RuntimeAssetsPublication:
    """Paths and identity of one immutable runtime-assets publication."""

    runtime_assets: Path
    assets_manifest: Path
    fusion_profile: Path
    asset_set_sha256: str


@dataclass(frozen=True, slots=True)
class RuntimeBundle:
    """Validated paths selected by one finalized runtime bundle."""

    path: Path
    bundle_id: str
    asset_set_sha256: str
    runtime_assets: Path
    template_model_dir: Path
    fusion_profile: Path
    threshold_artifact: Path


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return cast("dict[str, Any]", payload)


def _canonical_bytes(payload: object) -> bytes:
    try:
        return json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    except ValueError as exc:
        raise ValueError("canonical JSON cannot contain non-finite numbers") from exc


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolve_source(value: object, *, field: str) -> Path:
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"source declaration {field} must be a non-empty path")
    path = Path(text).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _contained_file(root: Path, value: object, *, field: str) -> Path:
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty relative path")
    path = (root / text).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes its containing directory: {text}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"{field} does not exist: {path}")
    return path


def _validate_binding(binding: object, *, label: str) -> tuple[Path, str]:
    if not isinstance(binding, dict):
        raise TypeError(f"{label} binding must be an object")
    path = Path(str(binding.get("path", ""))).expanduser().resolve()
    expected = binding.get("sha256")
    if not _valid_sha256(expected):
        raise ValueError(f"{label} binding must contain a lowercase SHA-256")
    if not path.is_file():
        raise FileNotFoundError(f"{label} bound file does not exist: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, found {actual}")
    return path, str(expected)


def _require_identity(source: dict[str, Any]) -> None:
    if source.get("schema_version") != 1:
        raise ValueError("source declaration schema_version must be 1")
    if source.get("product") != "ZS32":
        raise ValueError("source declaration product must be exactly ZS32")
    if source.get("hand") != "right":
        raise ValueError("source declaration hand must be exactly right")
    if tuple(source.get("view_order", ())) != VIEW_ORDER:
        raise ValueError(f"source declaration must contain exactly the canonical eight views: {VIEW_ORDER}")
    if source.get("commissioning_only") is not True or source.get("production_release_allowed") is not False:
        raise ValueError("source declaration must be commissioning-only and forbid production release")
    if not str(source.get("bundle_id", "")).strip():
        raise ValueError("source declaration bundle_id must be non-empty")
    if not str(source.get("roi_version", "")).strip():
        raise ValueError("source declaration roi_version must explicitly identify the ROI generation")


def _validate_roi(path: Path) -> dict[str, Any]:
    payload = _read_object(path, "ROI config")
    views = payload.get("views")
    if not isinstance(views, dict) or tuple(views) != VIEW_ORDER:
        raise ValueError(f"ROI config must contain exactly the canonical eight views in order: {VIEW_ORDER}")
    for view in VIEW_ORDER:
        record = views[view]
        roi = record.get("roi") if isinstance(record, dict) else None
        if not isinstance(roi, list) or len(roi) != 4 or any(isinstance(value, bool) for value in roi):
            raise ValueError(f"ROI config has an invalid ROI for {view}")
        x1, y1, x2, y2 = (int(value) for value in roi)
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError(f"ROI config has a non-positive ROI for {view}")
    return payload


def _validate_template_model(template_dir: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    model_path = template_dir / "model.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"template model does not exist: {model_path}")
    sidecar = template_dir / "model.sha256"
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != sha256_file(model_path):
        raise ValueError("template model SHA-256 sidecar does not match model.json")
    model = _read_object(model_path, "template model")
    if tuple(model.get("required_hands", ())) != ("right",):
        raise ValueError("template model must support exactly the right hand")
    if tuple(model.get("required_views", ())) != VIEW_ORDER:
        raise ValueError("template model must contain exactly the canonical eight views")
    groups = model.get("groups")
    expected_groups = tuple(f"right/{view}" for view in VIEW_ORDER)
    if not isinstance(groups, dict) or set(groups) != set(expected_groups) or len(groups) != len(expected_groups):
        raise ValueError("template model must contain exactly eight right-hand groups")
    bindings = [
        {"path": "model.json", "sha256": sha256_file(model_path)},
        {"path": "model.sha256", "sha256": sha256_file(sidecar)},
    ]
    seen: set[Path] = set()
    for group_name in expected_groups:
        group = groups[group_name]
        templates = group.get("templates") if isinstance(group, dict) else None
        if not isinstance(templates, list) or not templates:
            raise ValueError(f"template group {group_name} must contain templates")
        for index, record in enumerate(templates):
            if not isinstance(record, dict):
                raise TypeError(f"template record {group_name}:{index} must be an object")
            path = _contained_file(template_dir, record.get("path"), field=f"template {group_name}:{index}")
            expected = record.get("sha256")
            if not _valid_sha256(expected) or sha256_file(path) != expected:
                raise ValueError(f"template {group_name}:{index} SHA-256 mismatch")
            if path not in seen:
                bindings.append({"path": str(path.relative_to(template_dir)), "sha256": str(expected)})
                seen.add(path)
    return model, bindings


def _read_patchcore_summary(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"PatchCore summary does not exist: {path}")
    records: dict[str, dict[str, Any]] = {}
    checkpoint_hashes: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"view", "checkpoint", "deploy_threshold"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"PatchCore summary must contain columns {sorted(required)}")
        for row_number, row in enumerate(reader, start=2):
            semantic = str(row.get("view", "")).strip()
            if not semantic.startswith("right_"):
                raise ValueError(f"PatchCore summary row {row_number} must use right_<view> identity")
            view = semantic.removeprefix("right_")
            if view not in VIEW_ORDER or view in records:
                raise ValueError(f"PatchCore summary has duplicate or unsupported view: {semantic}")
            checkpoint = Path(str(row.get("checkpoint", ""))).expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"PatchCore checkpoint does not exist: {checkpoint}")
            try:
                reported = float(str(row.get("deploy_threshold", "")))
            except ValueError as exc:
                raise ValueError(f"reported deploy threshold must be finite at row {row_number}") from exc
            if not math.isfinite(reported):
                raise ValueError(f"reported deploy threshold must be finite at row {row_number}")
            digest = sha256_file(checkpoint)
            if digest in checkpoint_hashes:
                raise ValueError(
                    f"PatchCore views must use eight distinct checkpoints; duplicate content: {checkpoint}",
                )
            checkpoint_hashes.add(digest)
            model_name = str(row.get("model", "patchcore")).strip() or "patchcore"
            family = model_name.split("_s", maxsplit=1)[0].replace("_", "-")
            records[view] = {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": digest,
                "model_version": f"{family}-{digest[:12]}",
                "reported_deploy_threshold": reported,
            }
    if tuple(records) != VIEW_ORDER:
        raise ValueError(f"PatchCore summary must contain exactly the canonical eight views in order: {VIEW_ORDER}")
    return records


def _required_branches_by_view(profile: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    branches = profile.get("required_branches_by_view")
    if not isinstance(branches, dict) or tuple(branches) != VIEW_ORDER:
        raise ValueError("fusion profile template must contain exactly the canonical eight views")
    normalized: dict[str, tuple[str, ...]] = {}
    for view in VIEW_ORDER:
        required = tuple(branches[view]) if isinstance(branches[view], list) else ()
        expected_without_template = (f"anomaly_{view}", "yolo")
        allowed = {("template_match", *expected_without_template), expected_without_template}
        if view in {"front_secondary", "back_secondary"}:
            allowed.add(("yolo",))
        if required not in allowed:
            raise ValueError(f"fusion profile template branches for {view} are invalid: {required}")
        normalized[view] = required
    return normalized


def _profile_branch_order(branches: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    required = {branch for view_branches in branches.values() for branch in view_branches}
    canonical = ("template_match", "yolo", *(f"anomaly_{view}" for view in VIEW_ORDER))
    return tuple(branch for branch in canonical if branch in required)


def _validate_profile_template(profile: dict[str, Any], template_versions: dict[str, str]) -> None:
    if profile.get("commissioning_only") is not True or profile.get("production_release_allowed") is not False:
        raise ValueError("fusion profile template must be commissioning-only and forbid production release")
    identity = profile.get("identity")
    if (
        not isinstance(identity, dict)
        or identity.get("product") != "ZS32"
        or identity.get("allowed_hands") != ["right"]
    ):
        raise ValueError("fusion profile template must use exact ZS32/right identity")
    branches = _required_branches_by_view(profile)
    expected_branch_order = _profile_branch_order(branches)
    if tuple(profile.get("branch_order", ())) != expected_branch_order:
        raise ValueError(f"fusion profile template branch_order must be exactly {expected_branch_order}")
    rules = profile.get("rules")
    if (
        not isinstance(rules, dict)
        or set(rules) != set(expected_branch_order)
        or len(rules) != len(expected_branch_order)
    ):
        raise ValueError("fusion profile template rules must reference exactly the required branches")
    if any(not isinstance(rules[branch], dict) for branch in expected_branch_order):
        raise TypeError("fusion profile template rules must be objects")
    existing = profile.get("expected_versions")
    if isinstance(existing, list) and existing:
        roi_versions = {record.get("roi_version") for record in existing if isinstance(record, dict)}
        if roi_versions != {template_versions["roi"]}:
            raise ValueError("ROI/template generation mismatch in fusion profile template")
        template_generations = {record.get("template_version") for record in existing if isinstance(record, dict)}
        if template_generations != {template_versions["template"]}:
            raise ValueError("template generation mismatch in fusion profile template")


def _expected_versions(
    patchcore: dict[str, dict[str, Any]],
    yolo_version: str,
    template_versions: dict[str, str],
    required_branches_by_view: dict[str, tuple[str, ...]],
) -> list[dict[str, str]]:
    records = []
    for view in VIEW_ORDER:
        common = {
            "hand": "right",
            "side": "zs32",
            "view": view,
            "roi_version": template_versions["roi"],
            "template_version": template_versions["template"],
        }
        for branch in required_branches_by_view[view]:
            model_version = (
                template_versions["model"]
                if branch == "template_match"
                else yolo_version
                if branch == "yolo"
                else patchcore[view]["model_version"]
            )
            records.append(
                {
                    **common,
                    "branch": branch,
                    "model_version": model_version,
                    "threshold_version": template_versions["threshold"],
                },
            )
    return records


def _rename_directory_no_replace(source: Path, target: Path) -> None:
    """Atomically rename a publication directory without replacing a concurrent target."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "atomic no-replace rename is unavailable on this Linux runtime")
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    result = renameat2(AT_FDCWD, os.fsencode(source), AT_FDCWD, os.fsencode(target), RENAME_NOREPLACE)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, f"publication output directory already exists: {target}", target)
    raise OSError(error_number, os.strerror(error_number), target)


def _publish_directory(output_dir: Path, writer: Callable[[Path], None]) -> None:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"publication output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f".{output_dir.name}.{uuid4().hex}.tmp")
    staging.mkdir()
    try:
        writer(staging)
        if output_dir.exists():
            raise FileExistsError(f"publication output directory already exists: {output_dir}")
        _rename_directory_no_replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def publish_directory_no_replace(output_dir: Path, writer: Callable[[Path], None]) -> None:
    """Atomically publish one immutable directory without replacing an existing generation."""
    _publish_directory(output_dir, writer)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def publish_runtime_assets(source_path: Path, output_dir: Path) -> RuntimeAssetsPublication:
    """Validate selected assets and atomically publish a new immutable asset set."""
    source_path = source_path.expanduser().resolve()
    source = _read_object(source_path, "source declaration")
    _require_identity(source)
    roi_path = _resolve_source(source.get("roi_config"), field="roi_config")
    _validate_roi(roi_path)
    template_dir = _resolve_source(source.get("template_model_dir"), field="template_model_dir")
    template_model, template_files = _validate_template_model(template_dir)
    template_versions_raw = template_model.get("versions")
    if not isinstance(template_versions_raw, dict):
        raise TypeError("template model versions must be an object")
    version_fields = ("model", "threshold", "roi", "template")
    template_versions = {field: str(template_versions_raw.get(field, "")).strip() for field in version_fields}
    if any(not value for value in template_versions.values()):
        raise ValueError(f"template model versions must contain non-empty {version_fields}")
    declared_roi_version = str(source["roi_version"]).strip()
    if declared_roi_version != template_versions["roi"]:
        raise ValueError("ROI source generation does not match the template ROI version")

    summary_path = _resolve_source(source.get("patchcore_summary"), field="patchcore_summary")
    patchcore = _read_patchcore_summary(summary_path)
    yolo_path = _resolve_source(source.get("yolo_weights"), field="yolo_weights")
    if not yolo_path.is_file():
        raise FileNotFoundError(f"YOLO weights do not exist: {yolo_path}")
    yolo_sha256 = sha256_file(yolo_path)
    yolo_version = f"yolo-{yolo_path.stem}-{yolo_sha256[:12]}"
    yolo_imgsz = source.get("yolo_imgsz", 640)
    if isinstance(yolo_imgsz, bool) or not isinstance(yolo_imgsz, int) or yolo_imgsz <= 0:
        raise ValueError("yolo_imgsz must be a positive integer")
    profile_path = _resolve_source(source.get("fusion_profile_template"), field="fusion_profile_template")
    profile = _read_object(profile_path, "fusion profile template")
    _validate_profile_template(profile, template_versions)

    runtime = {
        "schema_version": 1,
        "bundle_id": source["bundle_id"],
        "product": "ZS32",
        "profile": profile["profile"],
        "supported_hands": ["right"],
        "view_order": list(VIEW_ORDER),
        "commissioning_only": True,
        "production_release_allowed": False,
        "patchcore_roi_config": str(roi_path),
        "yolo_roi_config": str(roi_path),
        "versions": {
            "threshold": template_versions["threshold"],
            "patchcore_roi": template_versions["roi"],
            "yolo_roi": template_versions["roi"],
            "template": template_versions["template"],
        },
        "patchcore": patchcore,
        "yolo": {
            "weights": str(yolo_path),
            "weights_sha256": yolo_sha256,
            "model_version": yolo_version,
            "imgsz": yolo_imgsz,
            "candidate_conf": 0.001,
            "iou": 0.7,
            "max_det": 300,
            "class_map": {"0": "defect"},
        },
    }
    generated_profile = dict(profile)
    required_branches = _required_branches_by_view(profile)
    generated_profile["expected_versions"] = _expected_versions(
        patchcore, yolo_version, template_versions, required_branches,
    )

    asset_set = {
        "schema_version": 1,
        "bundle_id": source["bundle_id"],
        "product": "ZS32",
        "hand": "right",
        "view_order": list(VIEW_ORDER),
        "commissioning_only": True,
        "production_release_allowed": False,
        "roi_config_sha256": sha256_file(roi_path),
        "template_files": template_files,
        "patchcore_summary_sha256": sha256_file(summary_path),
        "patchcore_checkpoints": [
            {"view": view, "sha256": patchcore[view]["checkpoint_sha256"]} for view in VIEW_ORDER
        ],
        "yolo_weights_sha256": yolo_sha256,
        "fusion_profile_template": {"path": str(profile_path), "sha256": sha256_file(profile_path)},
    }
    asset_set_sha256 = _canonical_sha256(asset_set)
    output_dir = output_dir.expanduser().resolve()

    def write(staging: Path) -> None:
        runtime_path = staging / "runtime_assets.json"
        fusion_path = staging / "fusion_profile.json"
        _write_json(runtime_path, runtime)
        _write_json(fusion_path, generated_profile)
        manifest = {
            "schema_version": 1,
            "bundle_id": source["bundle_id"],
            "product": "ZS32",
            "hand": "right",
            "commissioning_only": True,
            "production_release_allowed": False,
            "asset_set": asset_set,
            "asset_set_sha256": asset_set_sha256,
            "runtime_assets": {"path": str(output_dir / runtime_path.name), "sha256": sha256_file(runtime_path)},
            "fusion_profile": {"path": str(output_dir / fusion_path.name), "sha256": sha256_file(fusion_path)},
            "fusion_profile_template": {"path": str(profile_path), "sha256": sha256_file(profile_path)},
            "roi_config": {"path": str(roi_path), "sha256": sha256_file(roi_path)},
            "template_model": {
                "path": str(template_dir),
                "files": template_files,
                "files_sha256": _canonical_sha256(template_files),
                "model_json": {
                    "path": str(template_dir / "model.json"),
                    "sha256": sha256_file(template_dir / "model.json"),
                },
            },
            "patchcore_summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
            "yolo_weights": {"path": str(yolo_path), "sha256": yolo_sha256},
        }
        _write_json(staging / "assets_manifest.json", manifest)

    _publish_directory(output_dir, write)
    publication = RuntimeAssetsPublication(
        runtime_assets=output_dir / "runtime_assets.json",
        assets_manifest=output_dir / "assets_manifest.json",
        fusion_profile=output_dir / "fusion_profile.json",
        asset_set_sha256=asset_set_sha256,
    )
    _validate_assets_manifest(publication.assets_manifest)
    return publication


def _validate_assets_manifest(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    manifest = _read_object(path, "assets manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"assets manifest schema_version must be {SCHEMA_VERSION}")
    if manifest.get("product") != "ZS32" or manifest.get("hand") != "right":
        raise ValueError("assets manifest must use exact ZS32/right identity")
    if manifest.get("commissioning_only") is not True or manifest.get("production_release_allowed") is not False:
        raise ValueError("assets manifest must be commissioning-only and forbid production release")
    asset_set = manifest.get("asset_set")
    if not isinstance(asset_set, dict) or _canonical_sha256(asset_set) != manifest.get("asset_set_sha256"):
        raise ValueError("assets manifest asset-set canonical SHA-256 mismatch")
    if asset_set.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"asset set schema_version must be {SCHEMA_VERSION}")
    if tuple(asset_set.get("view_order", ())) != VIEW_ORDER:
        raise ValueError(f"asset set view_order must be exactly the canonical eight views: {VIEW_ORDER}")
    for field in ("bundle_id", "product", "hand", "commissioning_only", "production_release_allowed"):
        if asset_set.get(field) != manifest.get(field):
            raise ValueError(f"assets manifest asset-set identity differs for {field}")
    runtime_payload: dict[str, Any] | None = None
    fusion_payload: dict[str, Any] | None = None
    for field in ("runtime_assets", "fusion_profile"):
        bound_path, _ = _validate_binding(manifest.get(field), label=field.replace("_", " "))
        try:
            bound_path.relative_to(path.parent)
        except ValueError as exc:
            raise ValueError(f"{field} path escapes the immutable assets publication") from exc
        if field == "runtime_assets":
            load_runtime_config(bound_path)
            runtime_payload = _read_object(bound_path, "runtime assets")
            if runtime_payload.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"runtime assets schema_version must be {SCHEMA_VERSION}")
            if tuple(runtime_payload.get("view_order", ())) != VIEW_ORDER:
                raise ValueError(f"runtime assets view_order must be exactly the canonical eight views: {VIEW_ORDER}")
        else:
            fusion_payload = _read_object(bound_path, "fusion profile")
    profile_template_path, profile_template_sha256 = _validate_binding(
        manifest.get("fusion_profile_template"),
        label="fusion profile template",
    )
    profile_template_binding = {
        "path": str(profile_template_path),
        "sha256": profile_template_sha256,
    }
    if asset_set.get("fusion_profile_template") != profile_template_binding:
        raise ValueError("assets manifest fusion profile template path/SHA-256 differs from the asset set")
    _, roi_sha256 = _validate_binding(manifest.get("roi_config"), label="ROI config")
    _, summary_sha256 = _validate_binding(manifest.get("patchcore_summary"), label="PatchCore summary")
    _, yolo_sha256 = _validate_binding(manifest.get("yolo_weights"), label="YOLO weights")
    if asset_set.get("roi_config_sha256") != roi_sha256:
        raise ValueError("assets manifest asset-set ROI SHA-256 differs from the ROI binding")
    if asset_set.get("patchcore_summary_sha256") != summary_sha256:
        raise ValueError("assets manifest asset-set PatchCore summary SHA-256 differs from its binding")
    if asset_set.get("yolo_weights_sha256") != yolo_sha256:
        raise ValueError("assets manifest asset-set YOLO SHA-256 differs from its binding")
    if runtime_payload is None:
        raise ValueError("assets manifest must bind runtime assets")
    patchcore = runtime_payload.get("patchcore")
    if not isinstance(patchcore, dict):
        raise TypeError("runtime assets patchcore must be an object")
    checkpoint_bindings = [{"view": view, "sha256": patchcore[view]["checkpoint_sha256"]} for view in VIEW_ORDER]
    if asset_set.get("patchcore_checkpoints") != checkpoint_bindings:
        raise ValueError("assets manifest asset-set PatchCore checkpoint hashes differ from runtime assets")
    template = manifest.get("template_model")
    if not isinstance(template, dict):
        raise TypeError("template model manifest must be an object")
    template_dir = Path(str(template.get("path", ""))).expanduser().resolve()
    files = template.get("files")
    if not isinstance(files, list) or _canonical_sha256(files) != template.get("files_sha256"):
        raise ValueError("template model files canonical SHA-256 mismatch")
    if asset_set.get("template_files") != files:
        raise ValueError("assets manifest asset-set template files differ from the template binding")
    for index, binding in enumerate(files):
        if not isinstance(binding, dict):
            raise TypeError(f"template model file binding {index} must be an object")
        file_path = _contained_file(template_dir, binding.get("path"), field=f"template model file {index}")
        if not _valid_sha256(binding.get("sha256")) or sha256_file(file_path) != binding["sha256"]:
            raise ValueError(f"template model file {index} SHA-256 mismatch")
    model_path, _ = _validate_binding(template.get("model_json"), label="template model")
    try:
        model_path.relative_to(template_dir)
    except ValueError as exc:
        raise ValueError("template model path escapes its publication directory") from exc
    model_payload = _read_object(model_path, "template model")
    versions = model_payload.get("versions")
    if not isinstance(versions, dict):
        raise TypeError("template model versions must be an object")
    template_versions = {
        field: str(versions.get(field, "")).strip() for field in ("model", "threshold", "roi", "template")
    }
    if not all(template_versions.values()):
        raise ValueError("template model versions must be complete")
    source_profile = _read_object(profile_template_path, "fusion profile template")
    _validate_profile_template(source_profile, template_versions)
    if fusion_payload is None:
        raise ValueError("assets manifest must bind a fusion profile")
    _validate_profile_template(fusion_payload, template_versions)
    yolo = runtime_payload.get("yolo")
    if not isinstance(yolo, dict):
        raise TypeError("runtime assets yolo must be an object")
    required_branches = _required_branches_by_view(fusion_payload)
    expected_versions = _expected_versions(
        patchcore, str(yolo.get("model_version", "")), template_versions, required_branches,
    )
    if fusion_payload.get("expected_versions") != expected_versions:
        raise ValueError("fusion profile expected_versions differ from the bound runtime and template assets")
    return manifest


def load_runtime_assets_manifest(path: Path) -> dict[str, Any]:
    """Load and fully revalidate a Task 2 immutable runtime-assets manifest."""
    return _validate_assets_manifest(path)


def _validate_threshold_binding(path: Path, assets: dict[str, Any]) -> dict[str, Any]:
    path = path.expanduser().resolve()
    threshold = _read_object(path, "threshold artifact")
    if threshold.get("commissioning_only") is not True or threshold.get("production_release_allowed") is not False:
        raise ValueError("threshold artifact must be commissioning-only and forbid production release")
    if (
        threshold.get("artifact_schema") != THRESHOLD_ARTIFACT_SCHEMA
        or threshold.get("artifact_version") != THRESHOLD_ARTIFACT_VERSION
    ):
        raise ValueError("threshold artifact schema/version does not match the Stage18 contract")
    if threshold.get("calibration_valid") is not True:
        raise ValueError("threshold artifact calibration_valid must be true")
    unsigned = dict(threshold)
    claimed = unsigned.pop("artifact_sha256", None)
    if not _valid_sha256(claimed) or claimed != _canonical_sha256(unsigned):
        raise ValueError("threshold artifact canonical SHA-256 mismatch")
    fusion_path, fusion_sha256 = _validate_binding(assets.get("fusion_profile"), label="fusion profile")
    profile = _read_object(fusion_path, "fusion profile")
    expected = profile.get("expected_versions")
    if not isinstance(expected, list) or not expected or not all(isinstance(record, dict) for record in expected):
        raise ValueError("fusion profile must contain non-empty expected version records")
    expected_count = len(expected)
    if threshold.get("profile_sha256") != fusion_sha256 or threshold.get("config_sha256") != fusion_sha256:
        raise ValueError("threshold artifact is not bound to the exact fusion profile SHA-256")
    if threshold.get("required_views") != list(VIEW_ORDER):
        raise ValueError("threshold artifact required_views must contain exactly the canonical eight views")
    deployment = threshold.get("deployment_contract")
    identity = profile.get("identity")
    if not isinstance(deployment, dict) or not isinstance(identity, dict):
        raise TypeError("threshold deployment contract and fusion profile identity must be objects")
    if deployment.get("config_sha256") != fusion_sha256 or deployment.get("expected_versions") != expected:
        raise ValueError("threshold deployment contract does not match the exact fusion profile")
    for field, value in identity.items():
        if deployment.get(field) != value:
            raise ValueError(f"threshold deployment contract identity differs for {field}")

    identity_fields = ("hand", "view", "branch", "model_version", "roi_version")
    expected_by_identity = {
        tuple(str(record.get(field, "")) for field in identity_fields): record for record in expected
    }
    if len(expected_by_identity) != expected_count or any(not all(key) for key in expected_by_identity):
        raise ValueError("fusion profile expected versions contain duplicate or incomplete identities")
    required_groups = [list(key) for key in sorted(expected_by_identity)]
    if threshold.get("required_groups") != required_groups:
        raise ValueError("threshold artifact required_groups do not match the exact profile")

    records = threshold.get("thresholds")
    if not isinstance(records, list) or len(records) != expected_count or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"threshold artifact must contain exactly {expected_count} threshold records")
    records_by_identity = {tuple(str(record.get(field, "")) for field in identity_fields): record for record in records}
    if len(records_by_identity) != expected_count or set(records_by_identity) != set(expected_by_identity):
        raise ValueError(f"threshold artifact must contain the exact {expected_count} unique profile identities")
    expected_record_order = sorted(expected_by_identity)
    actual_record_order = [tuple(str(record.get(field, "")) for field in identity_fields) for record in records]
    if actual_record_order != expected_record_order:
        raise ValueError("threshold artifact records must use the exact canonical profile order")
    expected_threshold_versions = sorted({str(record.get("threshold_version", "")) for record in expected})
    if not all(expected_threshold_versions) or threshold.get("threshold_versions") != expected_threshold_versions:
        raise ValueError("threshold artifact threshold_versions do not match the fusion profile")
    for index, record in enumerate(records):
        if record.get("status") != "ok":
            raise ValueError(f"threshold record {index} status must be ok")
        finite_values: dict[str, float] = {}
        for field in ("low_threshold", "high_threshold"):
            value = record.get(field)
            if isinstance(value, bool):
                raise TypeError(f"threshold record {index} {field} must be a finite number")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"threshold record {index} {field} must be finite") from exc
            if not math.isfinite(numeric):
                raise ValueError(f"threshold record {index} {field} must be finite")
            finite_values[field] = numeric
        if finite_values["low_threshold"] > finite_values["high_threshold"]:
            raise ValueError(f"threshold record {index} low_threshold must not exceed high_threshold")
    if threshold.get("threshold_records_sha256") != _canonical_sha256(records):
        raise ValueError("threshold records canonical SHA-256 mismatch")
    sources = threshold.get("source_artifacts")
    if not isinstance(sources, dict):
        raise TypeError("threshold artifact source_artifacts must be an object")
    expected_runtime = assets["runtime_assets"]
    runtime_binding = sources.get("runtime_config", sources.get("runtime_assets"))
    if not isinstance(runtime_binding, dict) or runtime_binding.get("sha256") != expected_runtime["sha256"]:
        raise ValueError("threshold artifact is not bound to the exact runtime-assets SHA-256")
    if Path(str(runtime_binding.get("path", ""))).expanduser().resolve() != Path(expected_runtime["path"]).resolve():
        raise ValueError("threshold artifact runtime config path differs from the runtime-assets binding")
    expected_template = assets["template_model"]["model_json"]
    template_binding = sources.get("template_model")
    if not isinstance(template_binding, dict) or template_binding.get("sha256") != expected_template["sha256"]:
        raise ValueError("threshold artifact is not bound to the exact template model SHA-256")
    if Path(str(template_binding.get("path", ""))).expanduser().resolve() != Path(expected_template["path"]).resolve():
        raise ValueError("threshold artifact template model path differs from the template binding")
    return threshold


def finalize_runtime_bundle(assets_manifest: Path, threshold_artifact: Path, output_dir: Path) -> Path:
    """Bind calibrated thresholds to an immutable asset publication."""
    assets_manifest = assets_manifest.expanduser().resolve()
    threshold_artifact = threshold_artifact.expanduser().resolve()
    assets = _validate_assets_manifest(assets_manifest)
    threshold = _validate_threshold_binding(threshold_artifact, assets)
    output_dir = output_dir.expanduser().resolve()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "bundle_id": assets["bundle_id"],
        "product": "ZS32",
        "hand": "right",
        "view_order": list(VIEW_ORDER),
        "commissioning_only": True,
        "production_release_allowed": False,
        "asset_set_sha256": assets["asset_set_sha256"],
        "assets_manifest": {"path": str(assets_manifest), "sha256": sha256_file(assets_manifest)},
        "runtime_assets": assets["runtime_assets"],
        "fusion_profile": assets["fusion_profile"],
        "template_model": assets["template_model"],
        "threshold_artifact": {
            "path": str(threshold_artifact),
            "sha256": sha256_file(threshold_artifact),
            "artifact_sha256": threshold["artifact_sha256"],
        },
    }
    payload["bundle_sha256"] = _canonical_sha256(payload)

    def write(staging: Path) -> None:
        _write_json(staging / "runtime_bundle.json", payload)

    _publish_directory(output_dir, write)
    path = output_dir / "runtime_bundle.json"
    load_runtime_bundle(path)
    return path


def load_runtime_bundle(path: Path) -> RuntimeBundle:
    """Load a finalized bundle and fail closed on identity, path, or asset drift."""
    path = path.expanduser().resolve()
    payload = _read_object(path, "runtime bundle")
    unsigned = dict(payload)
    claimed = unsigned.pop("bundle_sha256", None)
    if not _valid_sha256(claimed) or claimed != _canonical_sha256(unsigned):
        raise ValueError("runtime bundle canonical SHA-256 mismatch")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"runtime bundle schema_version must be {SCHEMA_VERSION}")
    if tuple(payload.get("view_order", ())) != VIEW_ORDER:
        raise ValueError(f"runtime bundle view_order must be exactly the canonical eight views: {VIEW_ORDER}")
    if payload.get("product") != "ZS32" or payload.get("hand") != "right":
        raise ValueError("runtime bundle must use exact ZS32/right identity")
    if payload.get("commissioning_only") is not True or payload.get("production_release_allowed") is not False:
        raise ValueError("runtime bundle must be commissioning-only and forbid production release")
    manifest_path, _ = _validate_binding(payload.get("assets_manifest"), label="assets manifest")
    assets = _validate_assets_manifest(manifest_path)
    if payload.get("bundle_id") != assets.get("bundle_id"):
        raise ValueError("runtime bundle bundle_id differs from the assets manifest")
    if payload.get("asset_set_sha256") != assets.get("asset_set_sha256"):
        raise ValueError("runtime bundle asset-set SHA-256 differs from assets manifest")
    runtime_path, _ = _validate_binding(payload.get("runtime_assets"), label="runtime assets")
    fusion_path, _ = _validate_binding(payload.get("fusion_profile"), label="fusion profile")
    if payload.get("runtime_assets") != assets.get("runtime_assets") or payload.get("fusion_profile") != assets.get(
        "fusion_profile",
    ):
        raise ValueError("runtime bundle asset bindings differ from the assets manifest")
    threshold_path, _ = _validate_binding(payload.get("threshold_artifact"), label="threshold artifact")
    threshold = _validate_threshold_binding(threshold_path, assets)
    threshold_binding = payload["threshold_artifact"]
    if threshold_binding.get("artifact_sha256") != threshold.get("artifact_sha256"):
        raise ValueError("runtime bundle threshold artifact canonical SHA-256 mismatch")
    template = payload.get("template_model")
    if template != assets.get("template_model") or not isinstance(template, dict):
        raise ValueError("runtime bundle template binding differs from the assets manifest")
    load_runtime_config(runtime_path)
    return RuntimeBundle(
        path=path,
        bundle_id=str(payload["bundle_id"]),
        asset_set_sha256=str(payload["asset_set_sha256"]),
        runtime_assets=runtime_path,
        template_model_dir=Path(str(template["path"])).resolve(),
        fusion_profile=fusion_path,
        threshold_artifact=threshold_path,
    )
