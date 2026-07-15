# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: EM101, EM102, TRY003

"""Publish traceable ZS32 offline-fusion commissioning artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from typing import TYPE_CHECKING, Any

from capture_data.fusion_calibration import THRESHOLD_ARTIFACT_SCHEMA, THRESHOLD_ARTIFACT_VERSION
if TYPE_CHECKING:
    from pathlib import Path

GroupKey = tuple[str, str, str, str, str]
LEGACY_VIEWS = ("front", "front_left", "front_right", "back", "back_left", "back_right")
EIGHT_VIEWS = (
    "front",
    "front_left",
    "front_right",
    "front_secondary",
    "back",
    "back_left",
    "back_right",
    "back_secondary",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_commissioning_source_assets(
    threshold_artifact_path: Path,
    runtime_config_path: Path,
    template_model_path: Path,
) -> None:
    """Validate the live files bound by a commissioning artifact."""
    payload = _read_object(threshold_artifact_path)
    unsigned = dict(payload)
    artifact_sha256 = unsigned.pop("artifact_sha256", None)
    if artifact_sha256 != _canonical_sha256(unsigned):
        raise ValueError("commissioning threshold artifact immutable SHA256 mismatch")
    if payload.get("commissioning_only") is not True or payload.get("production_release_allowed") is not False:
        raise ValueError("source asset validation requires an explicit non-production commissioning artifact")
    sources = payload.get("source_artifacts")
    if not isinstance(sources, dict):
        raise TypeError("commissioning artifact source_artifacts must be an object")
    for name, path in (("runtime_config", runtime_config_path), ("template_model", template_model_path)):
        binding = sources.get(name)
        expected = binding.get("sha256") if isinstance(binding, dict) else None
        if not _valid_sha256(expected):
            raise ValueError(f"commissioning artifact has no valid {name} SHA256 binding")
        actual = _sha256(path)
        if actual != expected:
            label = "runtime config" if name == "runtime_config" else "template model"
            raise ValueError(f"{label} SHA256 differs from the locked commissioning threshold artifact")


def validate_18_group_source_assets(
    threshold_artifact_path: Path,
    runtime_config_path: Path,
    template_model_path: Path,
) -> None:
    """Validate legacy 18-group assets through the profile-driven validator."""
    validate_commissioning_source_assets(threshold_artifact_path, runtime_config_path, template_model_path)


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON payload must be an object: {path}")
    return payload


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _group_key(record: dict[str, Any]) -> GroupKey:
    try:
        return tuple(str(record[field]) for field in ("hand", "view", "branch", "model_version", "roi_version"))  # type: ignore[return-value]
    except KeyError as exc:
        raise ValueError(f"incomplete threshold identity: {record}") from exc


def _validate_stage33_binding(path: Path) -> dict[str, Any]:
    sidecar_path = path.parent / "stage33_publication.json"
    sidecar = _read_object(sidecar_path)
    if sidecar.get("commissioning_only") is not True:
        raise ValueError(f"Stage33 source is not explicitly commissioning-only: {sidecar_path}")
    if sidecar.get("thresholds_sha256") != _sha256(path):
        raise ValueError(f"Stage33 source hash differs from its publication sidecar: {path}")
    return sidecar


def _validate_profile(profile: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    if profile.get("commissioning_only") is not True or profile.get("production_release_allowed") is not False:
        raise ValueError("commissioning profile must be commissioning-only and forbid production release")
    by_view = profile.get("required_branches_by_view")
    if not isinstance(by_view, dict) or not by_view:
        raise ValueError("commissioning profile must define required branches by view")
    views = list(by_view)
    if tuple(views) not in {LEGACY_VIEWS, EIGHT_VIEWS}:
        raise ValueError("commissioning profile must use the exact legacy six-view or canonical eight-view order")
    records = profile.get("expected_versions")
    expected_count = len(views) * 3
    if (
        not isinstance(records, list)
        or len(records) != expected_count
        or not all(isinstance(item, dict) for item in records)
    ):
        raise ValueError(f"commissioning profile must contain exactly {expected_count} expected version records")
    for view in views:
        expected = {"template_match", f"anomaly_{view}", "yolo"}
        if set(by_view[view]) != expected:
            raise ValueError(f"commissioning profile has unexpected branches for {view}")
    keys = [_group_key(record) for record in records]
    if len(set(keys)) != expected_count:
        raise ValueError("commissioning profile contains duplicate identities")
    actual_pairs = {(str(record["view"]), str(record["branch"])) for record in records}
    expected_pairs = {
        (view, branch)
        for view in views
        for branch in ("template_match", f"anomaly_{view}", "yolo")
    }
    if actual_pairs != expected_pairs or any(record.get("hand") != "right" for record in records):
        raise ValueError("commissioning profile must contain exact right-hand view and branch identities")
    return views, records


def _validate_runtime_contract(
    profile: dict[str, Any],
    expected: list[dict[str, Any]],
    runtime: dict[str, Any],
    template: dict[str, Any],
) -> None:
    if runtime.get("profile") != profile.get("profile"):
        raise ValueError("runtime profile does not match the commissioning fusion profile")
    runtime_versions = runtime.get("versions")
    template_versions = template.get("versions")
    patchcore = runtime.get("patchcore")
    yolo = runtime.get("yolo")
    if not all(isinstance(value, dict) for value in (runtime_versions, template_versions, patchcore, yolo)):
        raise TypeError("runtime/template version payloads must be objects")
    for record in expected:
        view = str(record["view"])
        branch = str(record["branch"])
        if branch == "template_match":
            actual = {
                "model_version": template_versions.get("model"),
                "threshold_version": template_versions.get("threshold"),
                "roi_version": template_versions.get("roi"),
                "template_version": template_versions.get("template"),
            }
        elif branch == "yolo":
            actual = {
                "model_version": yolo.get("model_version"),
                "threshold_version": runtime_versions.get("threshold"),
                "roi_version": runtime_versions.get("yolo_roi"),
                "template_version": runtime_versions.get("template"),
            }
        else:
            model = patchcore.get(view)
            if not isinstance(model, dict):
                raise ValueError(f"runtime has no PatchCore contract for {view}")
            actual = {
                "model_version": model.get("model_version"),
                "threshold_version": runtime_versions.get("threshold"),
                "roi_version": runtime_versions.get("patchcore_roi"),
                "template_version": runtime_versions.get("template"),
            }
        for field, value in actual.items():
            if value != record.get(field):
                raise ValueError(f"runtime version mismatch for {view}:{branch}:{field}")


def _threshold_map(path: Path, expected_groups: set[GroupKey]) -> dict[GroupKey, dict[str, Any]]:
    payload = _read_object(path)
    records = payload.get("thresholds")
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise TypeError(f"threshold records must be objects: {path}")
    output = {_group_key(record): record for record in records}
    if len(output) != len(records) or set(output) != expected_groups:
        raise ValueError(f"source threshold groups do not match the required commissioning groups: {path}")
    return output


def _validate_split_contract(payload: dict[str, Any], path: Path) -> None:
    if payload.get("fit_split") != "calibration" or payload.get("evaluation_split") != "test":
        raise ValueError(f"Stage33 source split contract must be calibration fit and test evaluation: {path}")


def _validate_base_source(path: Path) -> None:
    payload = _read_object(path)
    if payload.get("calibration_valid") is not True:
        raise ValueError(f"Stage33 template/PatchCore calibration_valid must be true: {path}")
    _validate_split_contract(payload, path)
    records = payload.get("thresholds")
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise TypeError(f"Stage33 template/PatchCore threshold records must be objects: {path}")
    if any(record.get("status") != "ok" for record in records):
        raise ValueError(f"Stage33 template/PatchCore record is not usable: {path}")


def _validate_yolo_source(path: Path, *, allow_test_leakage: bool = False) -> bool:
    payload = _read_object(path)
    if payload.get("commissioning_only") is not True:
        raise ValueError(f"Stage33 YOLO source must be commissioning-only: {path}")
    if payload.get("all_views_have_candidate") is not True:
        raise ValueError(f"Stage33 YOLO source must provide candidates for all required views: {path}")
    test_leakage = payload.get("test_used_for_selection") is True
    if test_leakage:
        if not allow_test_leakage:
            raise ValueError(f"Stage33 YOLO source uses test data; pass explicit test-leakage opt-in: {path}")
        if payload.get("data_leakage") is not True:
            raise ValueError(f"Stage33 leaked YOLO source must declare data_leakage=true: {path}")
        if payload.get("fit_split") != "calibration+test" or payload.get("evaluation_split") != "test_reused_for_selection":
            raise ValueError(f"Stage33 leaked YOLO source split contract is malformed: {path}")
    else:
        _validate_split_contract(payload, path)
    minimum_precision = _finite_threshold(
        payload.get("minimum_fit_image_presence_precision"),
        "minimum_fit_image_presence_precision",
    )
    if not 0 < minimum_precision <= 1:
        raise ValueError("minimum_fit_image_presence_precision must be within (0, 1]")
    records = payload.get("thresholds")
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise TypeError(f"Stage33 YOLO threshold records must be objects: {path}")
    for record in records:
        if record.get("status") != "ok":
            raise ValueError(f"Stage33 YOLO record is not usable: {path}")
        fit = record.get("fit")
        precision = (
            _finite_threshold(fit.get("image_presence_precision"), "YOLO fit image-presence precision")
            if isinstance(fit, dict)
            else None
        )
        if precision is None or precision < minimum_precision:
            raise ValueError(f"Stage33 YOLO record precision is below the declared minimum: {path}")
    return test_leakage


def _finite_threshold(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be finite")
    return numeric


def _compose_records(
    expected: list[dict[str, Any]],
    template: dict[str, Any],
    base: dict[GroupKey, dict[str, Any]],
    yolo: dict[GroupKey, dict[str, Any]],
) -> list[dict[str, Any]]:
    groups = template.get("groups")
    if not isinstance(groups, dict):
        raise TypeError("template model groups must be an object")
    records = []
    for expectation in sorted(expected, key=_group_key):
        key = _group_key(expectation)
        view = str(expectation["view"])
        branch = str(expectation["branch"])
        if branch == "template_match":
            model_group = groups.get(f"right/{view}")
            if not isinstance(model_group, dict):
                raise ValueError(f"template model has no right/{view} group")
            source = base[key]
            low = _finite_threshold(model_group.get("low_threshold"), "template low_threshold")
            high = _finite_threshold(model_group.get("high_threshold"), "template high_threshold")
            provenance = "online_template_model"
        elif branch == "yolo":
            source = yolo[key]
            low = high = _finite_threshold(source.get("threshold"), "YOLO threshold")
            provenance = "stage33_high_precision_auxiliary"
        else:
            source = base[key]
            low = _finite_threshold(source.get("low_threshold"), "PatchCore low_threshold")
            high = _finite_threshold(source.get("high_threshold"), "PatchCore high_threshold")
            provenance = "stage33_template_patchcore_calibration"
        if low > high:
            raise ValueError(f"low threshold exceeds high threshold for {key}")
        fit = source.get("fit") if isinstance(source.get("fit"), dict) else {}
        records.append(
            {
                "hand": expectation["hand"],
                "view": view,
                "branch": branch,
                "model_version": expectation["model_version"],
                "roi_version": expectation["roi_version"],
                "low_threshold": low,
                "high_threshold": high,
                "normal_count": int(source.get("normal_count", fit.get("negative_count", 0))),
                "defect_count": int(source.get("defect_count", fit.get("positive_count", 0))),
                "status": "ok",
                "commissioning_source": provenance,
            },
        )
    return records


def publish_commissioning_artifact(
    profile_path: Path,
    runtime_config_path: Path,
    template_model_path: Path,
    template_patchcore_thresholds_path: Path,
    yolo_auxiliary_thresholds_path: Path,
    output_dir: Path,
    *,
    allow_test_leakage: bool = False,
) -> Path:
    """Compose and atomically publish a non-production Stage18-compatible artifact."""
    if output_dir.exists():
        raise FileExistsError(f"commissioning output already exists: {output_dir}")
    profile = _read_object(profile_path)
    views, expected = _validate_profile(profile)
    runtime = _read_object(runtime_config_path)
    template = _read_object(template_model_path)
    _validate_runtime_contract(profile, expected, runtime, template)
    model_sha_path = template_model_path.parent / "model.sha256"
    if model_sha_path.read_text(encoding="utf-8").strip() != _sha256(template_model_path):
        raise ValueError("template model SHA256 sidecar does not match model.json")
    base_sidecar = _validate_stage33_binding(template_patchcore_thresholds_path)
    yolo_sidecar = _validate_stage33_binding(yolo_auxiliary_thresholds_path)
    _validate_base_source(template_patchcore_thresholds_path)
    test_leakage = _validate_yolo_source(
        yolo_auxiliary_thresholds_path,
        allow_test_leakage=allow_test_leakage,
    )

    all_keys = {_group_key(record) for record in expected}
    yolo_keys = {key for key in all_keys if key[2] == "yolo"}
    base_keys = all_keys - yolo_keys
    base = _threshold_map(template_patchcore_thresholds_path, base_keys)
    yolo = _threshold_map(yolo_auxiliary_thresholds_path, yolo_keys)
    records = _compose_records(expected, template, base, yolo)
    profile_sha256 = _sha256(profile_path)
    deployment_contract = {
        **profile["identity"],
        "config_sha256": profile_sha256,
        "expected_versions": expected,
    }
    payload = {
        "artifact_schema": THRESHOLD_ARTIFACT_SCHEMA,
        "artifact_version": THRESHOLD_ARTIFACT_VERSION,
        "calibration_valid": True,
        "commissioning_only": True,
        "production_release_allowed": False,
        "profile_sha256": profile_sha256,
        "config_sha256": profile_sha256,
        "target_recall": 1.0,
        "normal_quantile": 0.995,
        "fit_split": "calibration+test" if test_leakage else "calibration",
        "evaluation_split": "test_reused_for_selection" if test_leakage else "test",
        "test_used_for_selection": test_leakage,
        "data_leakage": test_leakage,
        "leakage_notice": (
            "TEST DATA WAS USED FOR YOLO THRESHOLD SELECTION; THIS ARTIFACT HAS NO HELD-OUT YOLO EVALUATION."
            if test_leakage
            else None
        ),
        "required_views": views,
        "required_groups": [list(_group_key(record)) for record in sorted(expected, key=_group_key)],
        "thresholds": records,
        "threshold_records_sha256": _canonical_sha256(records),
        "threshold_versions": sorted({str(record["threshold_version"]) for record in expected}),
        "deployment_contract": deployment_contract,
        "source_artifacts": {
            "runtime_config": {"path": str(runtime_config_path.resolve()), "sha256": _sha256(runtime_config_path)},
            "template_model": {"path": str(template_model_path.resolve()), "sha256": _sha256(template_model_path)},
            "template_patchcore_thresholds": {
                "path": str(template_patchcore_thresholds_path.resolve()),
                "sha256": _sha256(template_patchcore_thresholds_path),
                "stage33_publication": base_sidecar,
            },
            "yolo_auxiliary_thresholds": {
                "path": str(yolo_auxiliary_thresholds_path.resolve()),
                "sha256": _sha256(yolo_auxiliary_thresholds_path),
                "stage33_publication": yolo_sidecar,
            },
        },
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    summary = {
        "profile": profile["profile"],
        "threshold_group_count": len(records),
        "commissioning_only": True,
        "production_release_allowed": False,
        "test_used_for_selection": test_leakage,
        "data_leakage": test_leakage,
        "included_branches": ["template_match", "patchcore", "yolo"],
        "deferred_branches": ["quality_gate", "registration", "geometry"],
        "limitations": [
            f"Inspection completeness applies only to the explicit {len(records)}-group commissioning profile.",
            "This artifact must never be used as a production release approval.",
            "YOLO thresholds are image-presence candidates selected from small commissioning samples.",
            *(
                ["TEST DATA WAS USED FOR YOLO THRESHOLD SELECTION; reported YOLO metrics are not held-out."]
                if test_leakage
                else []
            ),
        ],
    }
    staging = output_dir.with_name(f".{output_dir.name}.tmp")
    if staging.exists():
        raise FileExistsError(f"stale commissioning publication staging exists: {staging}")
    staging.mkdir(parents=True)
    try:
        (staging / "thresholds.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir / "thresholds.json"


def publish_18_group_artifact(
    profile_path: Path,
    runtime_config_path: Path,
    template_model_path: Path,
    template_patchcore_thresholds_path: Path,
    yolo_auxiliary_thresholds_path: Path,
    output_dir: Path,
) -> Path:
    """Publish a legacy 18-group artifact through the profile-driven publisher."""
    return publish_commissioning_artifact(
        profile_path,
        runtime_config_path,
        template_model_path,
        template_patchcore_thresholds_path,
        yolo_auxiliary_thresholds_path,
        output_dir,
    )
