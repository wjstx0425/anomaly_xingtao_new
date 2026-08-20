# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: EM101, EM102, TRY003

"""Publish traceable ZS32 offline-fusion commissioning artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping

from capture_data.fusion_calibration import THRESHOLD_ARTIFACT_SCHEMA, THRESHOLD_ARTIFACT_VERSION

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
    normalized_by_view = {}
    for view in views:
        branches = tuple(by_view[view]) if isinstance(by_view[view], list) else ()
        required = (f"anomaly_{view}", "yolo")
        allowed = {("template_match", *required), required}
        if view in {"front_secondary", "back_secondary"}:
            allowed.add(("yolo",))
        if branches not in allowed:
            raise ValueError(f"commissioning profile has unexpected branches for {view}")
        normalized_by_view[view] = branches
    records = profile.get("expected_versions")
    expected_count = sum(len(branches) for branches in normalized_by_view.values())
    if (
        not isinstance(records, list)
        or len(records) != expected_count
        or not all(isinstance(item, dict) for item in records)
    ):
        raise ValueError(f"commissioning profile must contain exactly {expected_count} expected version records")
    keys = [_group_key(record) for record in records]
    if len(set(keys)) != expected_count:
        raise ValueError("commissioning profile contains duplicate identities")
    actual_pairs = {(str(record["view"]), str(record["branch"])) for record in records}
    expected_pairs = {
        (view, branch)
        for view, branches in normalized_by_view.items()
        for branch in branches
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


def _threshold_map(
    path: Path,
    expected_groups: set[GroupKey],
    *,
    allow_model_version_rebind: bool = False,
    allow_roi_version_rebind: bool = False,
) -> dict[GroupKey, dict[str, Any]]:
    payload = _read_object(path)
    records = payload.get("thresholds")
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise TypeError(f"threshold records must be objects: {path}")
    output = {_group_key(record): record for record in records}
    if len(output) != len(records):
        raise ValueError(f"source threshold groups contain duplicate identities: {path}")
    if allow_model_version_rebind or allow_roi_version_rebind:
        rebound: dict[GroupKey, dict[str, Any]] = {}
        for expected in expected_groups:
            candidates = [
                record
                for key, record in output.items()
                if key[:3] == expected[:3]
                and (allow_model_version_rebind or key[3] == expected[3])
                and (allow_roi_version_rebind or key[4] == expected[4])
            ]
            if len(candidates) != 1:
                raise ValueError(f"source threshold groups cannot be uniquely rebound to {expected}: {path}")
            rebound[expected] = candidates[0]
        return rebound
    if not expected_groups.issubset(output):
        raise ValueError(
            f"source threshold groups do not match: they do not contain the required commissioning groups: {path}",
        )
    return {key: output[key] for key in expected_groups}


def _validate_roi_version_rebind(
    runtime_config_path: Path,
    runtime: dict[str, Any],
    template_patchcore_thresholds_path: Path,
    yolo_auxiliary_thresholds_path: Path,
    yolo_sidecar: dict[str, Any],
    *,
    allow_roi_version_rebind: bool,
    source_roi_config_path: Path | None,
) -> dict[str, Any] | None:
    if not allow_roi_version_rebind:
        if source_roi_config_path is not None:
            raise ValueError("source ROI config requires explicit ROI-version-rebind authorization")
        return None
    if source_roi_config_path is None:
        raise ValueError("ROI-version rebind requires an explicit source ROI config")

    source_path = source_roi_config_path.resolve()
    source_sha256 = _sha256(source_path)
    base_root = template_patchcore_thresholds_path.resolve().parent.parent
    yolo_root = yolo_auxiliary_thresholds_path.resolve().parent.parent
    if base_root != yolo_root:
        raise ValueError("ROI-version rebind requires Stage33 threshold artifacts from the same commissioning run root")
    run_contract_path = base_root / "commissioning_run_contract.json"
    run_contract_sha256 = _sha256(run_contract_path)
    if yolo_sidecar.get("commissioning_run_contract_sha256") != run_contract_sha256:
        raise ValueError("Stage33 commissioning run contract SHA256 differs from the signed YOLO publication")
    run_contract = _read_object(run_contract_path)
    if run_contract.get("commissioning_only") is not True:
        raise ValueError("Stage33 commissioning run contract must be commissioning-only")
    source_runtime_value = run_contract.get("runtime_config")
    source_runtime_sha256 = run_contract.get("runtime_config_sha256")
    if not isinstance(source_runtime_value, str) or not _valid_sha256(source_runtime_sha256):
        raise ValueError("Stage33 commissioning run contract has no valid source runtime binding")
    source_runtime_path = Path(source_runtime_value)
    if not source_runtime_path.is_absolute():
        source_runtime_path = run_contract_path.parent / source_runtime_path
    source_runtime_path = source_runtime_path.resolve()
    if _sha256(source_runtime_path) != source_runtime_sha256:
        raise ValueError("source runtime config SHA256 differs from the Stage33 commissioning run contract")
    source_runtime = _read_object(source_runtime_path)
    source_runtime_versions = source_runtime.get("versions")
    if not isinstance(source_runtime_versions, dict):
        raise ValueError("source runtime config has no ROI version contract")

    expected_source_versions = {
        template_patchcore_thresholds_path: source_runtime_versions.get("patchcore_roi"),
        yolo_auxiliary_thresholds_path: source_runtime_versions.get("yolo_roi"),
    }
    for threshold_path, expected_roi_version in expected_source_versions.items():
        if not isinstance(expected_roi_version, str) or not expected_roi_version:
            raise ValueError("source runtime config has no valid ROI version identity")
        threshold_records = _read_object(threshold_path).get("thresholds")
        if not isinstance(threshold_records, list) or any(not isinstance(record, dict) for record in threshold_records):
            raise TypeError(f"threshold records must be objects: {threshold_path}")
        actual_roi_versions = {record.get("roi_version") for record in threshold_records}
        if actual_roi_versions != {expected_roi_version}:
            raise ValueError(f"source threshold ROI identities differ from the signed source runtime: {threshold_path}")

    bindings: dict[str, dict[str, str]] = {
        "commissioning_run_contract": {
            "path": str(run_contract_path.resolve()),
            "sha256": run_contract_sha256,
        },
        "source_runtime_config": {
            "path": str(source_runtime_path),
            "sha256": str(source_runtime_sha256),
        },
    }
    for field in ("patchcore_roi_config", "yolo_roi_config"):
        value = source_runtime.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"source runtime config has no valid {field} for ROI-version rebind")
        source_runtime_roi_path = Path(value)
        if not source_runtime_roi_path.is_absolute():
            source_runtime_roi_path = source_runtime_path.parent / source_runtime_roi_path
        source_runtime_roi_path = source_runtime_roi_path.resolve()
        source_runtime_roi_sha256 = _sha256(source_runtime_roi_path)
        if source_runtime_roi_sha256 != source_sha256:
            raise ValueError(f"source runtime {field} SHA256 differs from the explicit source ROI config")
        bindings[f"source_{field}"] = {
            "path": str(source_runtime_roi_path),
            "sha256": source_runtime_roi_sha256,
        }
    for field in ("patchcore_roi_config", "yolo_roi_config"):
        value = runtime.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"runtime config has no valid {field} for ROI-version rebind")
        runtime_roi_path = Path(value)
        if not runtime_roi_path.is_absolute():
            runtime_roi_path = runtime_config_path.parent / runtime_roi_path
        runtime_roi_path = runtime_roi_path.resolve()
        runtime_roi_sha256 = _sha256(runtime_roi_path)
        if runtime_roi_sha256 != source_sha256:
            raise ValueError(f"{field} SHA256 differs from the explicit source ROI config")
        bindings[field] = {"path": str(runtime_roi_path), "sha256": runtime_roi_sha256}
    return {
        "authorized": True,
        "source_roi_config": {"path": str(source_path), "sha256": source_sha256},
        **bindings,
    }


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


def _nonnegative_count(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _compose_records(
    expected: list[dict[str, Any]],
    template: dict[str, Any],
    base: dict[GroupKey, dict[str, Any]],
    yolo: dict[GroupKey, dict[str, Any]],
    *,
    allow_yolo_model_rebind: bool = False,
    allow_patchcore_model_rebind: bool = False,
    yolo_threshold_override: float | None = None,
    patchcore_threshold_overrides: Mapping[str, tuple[float, float]] | None = None,
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
            normal_count = _nonnegative_count(
                model_group.get("calibration_normal_count"),
                "template calibration_normal_count",
            )
            defect_count = _nonnegative_count(
                model_group.get("calibration_defect_count"),
                "template calibration_defect_count",
            )
            count_provenance = "template_model_calibration"
        elif branch == "yolo":
            source = yolo[key]
            fit = source.get("fit") if isinstance(source.get("fit"), dict) else {}
            if yolo_threshold_override is None:
                low = high = _finite_threshold(source.get("threshold"), "YOLO threshold")
                provenance = "stage33_high_precision_auxiliary"
            else:
                low = high = yolo_threshold_override
                provenance = "manual_demo_override"
            if allow_yolo_model_rebind:
                normal_count = defect_count = 0
                count_provenance = "unavailable_for_rebound_model"
            else:
                normal_count = int(source.get("normal_count", fit.get("negative_count", 0)))
                defect_count = int(source.get("defect_count", fit.get("positive_count", 0)))
                count_provenance = "stage33_threshold_record"
        else:
            source = base[key]
            fit = source.get("fit") if isinstance(source.get("fit"), dict) else {}
            override = (patchcore_threshold_overrides or {}).get(view)
            if override is None:
                low = _finite_threshold(source.get("low_threshold"), "PatchCore low_threshold")
                high = _finite_threshold(source.get("high_threshold"), "PatchCore high_threshold")
                provenance = "stage33_template_patchcore_calibration"
            else:
                low, high = override
                provenance = "manual_demo_override"
            if allow_patchcore_model_rebind:
                normal_count = defect_count = 0
                count_provenance = "unavailable_for_rebound_model"
            else:
                normal_count = int(source.get("normal_count", fit.get("negative_count", 0)))
                defect_count = int(source.get("defect_count", fit.get("positive_count", 0)))
                count_provenance = "stage33_threshold_record"
        if low > high:
            raise ValueError(f"low threshold exceeds high threshold for {key}")
        records.append(
            {
                "hand": expectation["hand"],
                "view": view,
                "branch": branch,
                "model_version": expectation["model_version"],
                "roi_version": expectation["roi_version"],
                "low_threshold": low,
                "high_threshold": high,
                "normal_count": normal_count,
                "defect_count": defect_count,
                "count_provenance": count_provenance,
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
    allow_yolo_model_rebind: bool = False,
    allow_patchcore_model_rebind: bool = False,
    allow_roi_version_rebind: bool = False,
    source_roi_config_path: Path | None = None,
    yolo_threshold_override: float | None = None,
    patchcore_threshold_overrides: Mapping[str, tuple[float, float]] | None = None,
) -> Path:
    """Compose and atomically publish a non-production Stage18-compatible artifact."""
    if output_dir.exists():
        raise FileExistsError(f"commissioning output already exists: {output_dir}")
    profile = _read_object(profile_path)
    if yolo_threshold_override is not None:
        yolo_threshold_override = _finite_threshold(yolo_threshold_override, "YOLO threshold override")
        if not 0 < yolo_threshold_override <= 1:
            raise ValueError("YOLO threshold override must be within (0, 1]")
    patchcore_overrides = dict(patchcore_threshold_overrides or {})
    for view, thresholds in patchcore_overrides.items():
        if view not in EIGHT_VIEWS or len(thresholds) != 2:
            raise ValueError(f"invalid PatchCore threshold override view: {view!r}")
        low = _finite_threshold(thresholds[0], f"PatchCore {view} low override")
        high = _finite_threshold(thresholds[1], f"PatchCore {view} high override")
        if not 0 <= low <= high <= 2:
            raise ValueError(f"PatchCore threshold override must satisfy 0 <= low <= high <= 2 for {view}")
        patchcore_overrides[view] = (low, high)
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
    roi_version_rebind = _validate_roi_version_rebind(
        runtime_config_path,
        runtime,
        template_patchcore_thresholds_path,
        yolo_auxiliary_thresholds_path,
        yolo_sidecar,
        allow_roi_version_rebind=allow_roi_version_rebind,
        source_roi_config_path=source_roi_config_path,
    )

    all_keys = {_group_key(record) for record in expected}
    yolo_keys = {key for key in all_keys if key[2] == "yolo"}
    base_keys = all_keys - yolo_keys
    base = _threshold_map(
        template_patchcore_thresholds_path,
        base_keys,
        allow_model_version_rebind=allow_patchcore_model_rebind,
        allow_roi_version_rebind=allow_roi_version_rebind,
    )
    yolo = _threshold_map(
        yolo_auxiliary_thresholds_path,
        yolo_keys,
        allow_model_version_rebind=allow_yolo_model_rebind,
        allow_roi_version_rebind=allow_roi_version_rebind,
    )
    records = _compose_records(
        expected,
        template,
        base,
        yolo,
        allow_yolo_model_rebind=allow_yolo_model_rebind,
        allow_patchcore_model_rebind=allow_patchcore_model_rebind,
        yolo_threshold_override=yolo_threshold_override,
        patchcore_threshold_overrides=patchcore_overrides,
    )
    required_patchcore_views = {str(record["view"]) for record in expected if str(record["branch"]).startswith("anomaly_")}
    if not set(patchcore_overrides).issubset(required_patchcore_views):
        raise ValueError("PatchCore threshold overrides must target required PatchCore views")
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
        "yolo_threshold_model_rebound": allow_yolo_model_rebind,
        "patchcore_threshold_model_rebound": allow_patchcore_model_rebind,
        "roi_version_rebound": allow_roi_version_rebind,
        "roi_version_rebind": roi_version_rebind,
        "yolo_threshold_override": yolo_threshold_override,
        "patchcore_threshold_overrides": {
            view: {"low_threshold": low, "high_threshold": high}
            for view, (low, high) in sorted(patchcore_overrides.items())
        },
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
            **(
                {
                    name: roi_version_rebind[name]
                    for name in (
                        "source_roi_config",
                        "commissioning_run_contract",
                        "source_runtime_config",
                    )
                }
                if roi_version_rebind is not None
                else {}
            ),
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
        "roi_version_rebound": allow_roi_version_rebind,
        "included_branches": ["template_match", "patchcore", "yolo"],
        "deferred_branches": ["quality_gate", "registration", "geometry"],
        "limitations": [
            f"Inspection completeness applies only to the explicit {len(records)}-group commissioning profile.",
            "This artifact must never be used as a production release approval.",
            "YOLO thresholds are image-presence candidates selected from small commissioning samples.",
            *(
                [
                    "ROI version identities were explicitly rebound only after byte-identical source/runtime "
                    "ROI verification through the signed Stage33 run contract "
                    f"(SHA256 {roi_version_rebind['source_roi_config']['sha256']}).",
                ]
                if roi_version_rebind is not None
                else []
            ),
            *(
                ["YOLO thresholds were rebound from a different model version for demo-only commissioning."]
                if allow_yolo_model_rebind
                else []
            ),
            *(
                [f"YOLO thresholds were manually overridden to {yolo_threshold_override} for demo use."]
                if yolo_threshold_override is not None
                else []
            ),
            *(
                [f"PatchCore thresholds were manually overridden for: {', '.join(sorted(patchcore_overrides))}."]
                if patchcore_overrides
                else []
            ),
            *(
                ["PatchCore thresholds were rebound from different model versions for demo-only commissioning."]
                if allow_patchcore_model_rebind
                else []
            ),
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
