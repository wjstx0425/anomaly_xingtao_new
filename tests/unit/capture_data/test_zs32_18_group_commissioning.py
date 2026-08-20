# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for publishing the explicit ZS32 18-group commissioning artifact."""

# The test intentionally calls the numbered Stage18 loader contract.
# ruff: noqa: SLF001

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from capture_data.zs32_18_group_commissioning import (
    _validate_profile,
    publish_commissioning_artifact,
    publish_18_group_artifact,
    validate_commissioning_source_assets,
    validate_18_group_source_assets,
)

if TYPE_CHECKING:
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[3]
PROFILE = REPO_ROOT / "config/fusion/zs32_right_unified_roi_18_group_commissioning.json"
PROFILE_24 = REPO_ROOT / "config/fusion/zs32_right_eight_view_24_group_commissioning.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resign_stage33(path: Path) -> None:
    sidecar_path = path.parent / "stage33_publication.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["thresholds_sha256"] = _sha256(path)
    _write_json(sidecar_path, sidecar)


def _load_stage18() -> ModuleType:
    path = REPO_ROOT / "pipeline/18_fuse_inspection_results.py"
    spec = importlib.util.spec_from_file_location("stage18_commissioning_artifact_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _assets(tmp_path: Path, profile_path: Path = PROFILE) -> tuple[Path, Path, Path, Path]:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    expected = profile["expected_versions"]
    template_record = next(record for record in expected if record["branch"] == "template_match")
    patchcore_record = next(record for record in expected if record["branch"].startswith("anomaly_"))
    yolo_record = next(record for record in expected if record["branch"] == "yolo")
    runtime = {
        "profile": profile["profile"],
        "versions": {
            "threshold": patchcore_record["threshold_version"],
            "patchcore_roi": patchcore_record["roi_version"],
            "yolo_roi": yolo_record["roi_version"],
            "template": patchcore_record["template_version"],
        },
        "patchcore": {
            record["view"]: {"model_version": record["model_version"]}
            for record in expected
            if record["branch"].startswith("anomaly_")
        },
        "yolo": {"model_version": yolo_record["model_version"]},
    }
    runtime_path = tmp_path / "runtime.json"
    _write_json(runtime_path, runtime)

    template = {
        "versions": {
            "model": template_record["model_version"],
            "threshold": template_record["threshold_version"],
            "roi": template_record["roi_version"],
            "template": template_record["template_version"],
        },
        "groups": {
            f"right/{view}": {
                "low_threshold": 0.11 if view == "front_left" else 0.1,
                "high_threshold": 0.2,
                "calibration_normal_count": 13,
                "calibration_defect_count": 0,
            }
            for view in {record["view"] for record in expected}
        },
    }
    template_path = tmp_path / "template" / "model.json"
    _write_json(template_path, template)
    (template_path.parent / "model.sha256").write_text(_sha256(template_path) + "\n", encoding="utf-8")

    base_records = []
    for record in expected:
        if record["branch"] == "yolo":
            continue
        base_records.append(
            {
                "hand": record["hand"],
                "view": record["view"],
                "branch": record["branch"],
                "model_version": record["model_version"],
                "roi_version": record["roi_version"],
                "low_threshold": 0.3,
                "high_threshold": 0.5,
                "normal_count": 8,
                "defect_count": 4,
                "status": "ok",
            },
        )
    base_path = tmp_path / "stage33" / "template_patchcore_threshold_calibration" / "thresholds.json"
    _write_json(
        base_path,
        {
            "calibration_valid": True,
            "fit_split": "calibration",
            "evaluation_split": "test",
            "thresholds": base_records,
        },
    )
    _write_json(
        base_path.parent / "stage33_publication.json",
        {"commissioning_only": True, "thresholds_sha256": _sha256(base_path)},
    )

    yolo_records = [
        {
            "hand": record["hand"],
            "view": record["view"],
            "branch": "yolo",
            "model_version": record["model_version"],
            "roi_version": record["roi_version"],
            "threshold": 0.9,
            "status": "ok",
            "fit": {
                "positive_count": 4,
                "negative_count": 8,
                "image_presence_precision": 1.0,
            },
        }
        for record in expected
        if record["branch"] == "yolo"
    ]
    yolo_path = tmp_path / "stage33" / "yolo_high_precision_auxiliary" / "thresholds.json"
    _write_json(
        yolo_path,
        {
            "commissioning_only": True,
            "test_used_for_selection": False,
            "all_views_have_candidate": True,
            "fit_split": "calibration",
            "evaluation_split": "test",
            "minimum_fit_image_presence_precision": 0.95,
            "thresholds": yolo_records,
        },
    )
    _write_json(
        yolo_path.parent / "stage33_publication.json",
        {"commissioning_only": True, "thresholds_sha256": _sha256(yolo_path)},
    )
    return runtime_path, template_path, base_path, yolo_path


def _bind_roi_rebind_source_contract(
    runtime: Path,
    base: Path,
    yolo: Path,
    source_roi: Path,
    *,
    source_runtime_roi: Path | None = None,
) -> None:
    runtime_payload = json.loads(runtime.read_text(encoding="utf-8"))
    runtime_payload["patchcore_roi_config"] = str(source_roi)
    runtime_payload["yolo_roi_config"] = str(source_roi)
    _write_json(runtime, runtime_payload)

    source_runtime_roi = source_runtime_roi or source_roi
    source_runtime = base.parent.parent / "source_runtime.json"
    _write_json(
        source_runtime,
        {
            "patchcore_roi_config": str(source_runtime_roi),
            "yolo_roi_config": str(source_runtime_roi),
            "versions": {
                "patchcore_roi": "legacy-roi-identity",
                "yolo_roi": "legacy-roi-identity",
            },
        },
    )
    run_contract = base.parent.parent / "commissioning_run_contract.json"
    _write_json(
        run_contract,
        {
            "commissioning_only": True,
            "runtime_config": str(source_runtime),
            "runtime_config_sha256": _sha256(source_runtime),
        },
    )
    yolo_sidecar_path = yolo.parent / "stage33_publication.json"
    yolo_sidecar = json.loads(yolo_sidecar_path.read_text(encoding="utf-8"))
    yolo_sidecar["commissioning_run_contract_sha256"] = _sha256(run_contract)
    _write_json(yolo_sidecar_path, yolo_sidecar)

    for threshold_path in (base, yolo):
        payload = json.loads(threshold_path.read_text(encoding="utf-8"))
        for record in payload["thresholds"]:
            record["roi_version"] = "legacy-roi-identity"
        _write_json(threshold_path, payload)
        _resign_stage33(threshold_path)


def test_eight_view_profile_is_exact_right_hand_twenty_four_group_contract() -> None:
    """The new profile must add both secondary views without weakening the old profile."""
    old_profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    profile = json.loads(PROFILE_24.read_text(encoding="utf-8"))
    views = {
        "front",
        "front_left",
        "front_right",
        "front_secondary",
        "back",
        "back_left",
        "back_right",
        "back_secondary",
    }

    assert len(old_profile["expected_versions"]) == 18
    assert profile["commissioning_only"] is True
    assert profile["production_release_allowed"] is False
    assert set(profile["required_branches_by_view"]) == views
    assert len(profile["expected_versions"]) == 24
    assert {
        (record["view"], record["branch"])
        for record in profile["expected_versions"]
    } == {
        (view, branch)
        for view in views
        for branch in ("template_match", f"anomaly_{view}", "yolo")
    }



def test_stage34_accepts_profile_driven_twenty_two_group_contract() -> None:
    profile = json.loads(PROFILE_24.read_text(encoding="utf-8"))
    secondary = {"front_secondary", "back_secondary"}
    for view in secondary:
        profile["required_branches_by_view"][view].remove("template_match")
    profile["expected_versions"] = [
        record for record in profile["expected_versions"]
        if not (record["view"] in secondary and record["branch"] == "template_match")
    ]
    views, records = _validate_profile(profile)
    assert len(views) == 8
    assert len(records) == 22

def test_publisher_builds_stage18_valid_twenty_four_group_artifact(tmp_path: Path) -> None:
    """Profile-driven publication must support exact eight-view commissioning evidence."""
    runtime, template, base, yolo = _assets(tmp_path, PROFILE_24)

    threshold_path = publish_commissioning_artifact(
        PROFILE_24,
        runtime,
        template,
        base,
        yolo,
        tmp_path / "published",
    )
    payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    stage18 = _load_stage18()
    records, metadata = stage18._load_threshold_artifact(threshold_path, profile_path=PROFILE_24)

    assert len(records) == 24
    profile = json.loads(PROFILE_24.read_text(encoding="utf-8"))
    assert payload["required_views"] == list(profile["required_branches_by_view"])
    assert payload["commissioning_only"] is True
    assert payload["production_release_allowed"] is False
    assert metadata["commissioning_only"] is True
    validate_commissioning_source_assets(threshold_path, runtime, template)


def test_publisher_requires_opt_in_and_marks_test_leakage(tmp_path: Path) -> None:
    """A leaked source must be rejected by default and remain obvious after opt-in."""
    runtime, template, base, yolo = _assets(tmp_path, PROFILE_24)
    payload = json.loads(yolo.read_text(encoding="utf-8"))
    payload.update(
        {
            "fit_split": "calibration+test",
            "evaluation_split": "test_reused_for_selection",
            "test_used_for_selection": True,
            "data_leakage": True,
        },
    )
    _write_json(yolo, payload)
    _resign_stage33(yolo)

    with pytest.raises(ValueError, match="explicit test-leakage opt-in"):
        publish_commissioning_artifact(PROFILE_24, runtime, template, base, yolo, tmp_path / "rejected")

    threshold_path = publish_commissioning_artifact(
        PROFILE_24,
        runtime,
        template,
        base,
        yolo,
        tmp_path / "published",
        allow_test_leakage=True,
    )
    artifact = json.loads(threshold_path.read_text(encoding="utf-8"))
    summary = json.loads((threshold_path.parent / "summary.json").read_text(encoding="utf-8"))

    assert artifact["data_leakage"] is True
    assert artifact["test_used_for_selection"] is True
    assert artifact["evaluation_split"] == "test_reused_for_selection"
    assert "TEST DATA" in artifact["leakage_notice"]
    assert summary["data_leakage"] is True
    assert any("TEST DATA" in limitation for limitation in summary["limitations"])


def test_publisher_requires_explicit_yolo_model_rebind_opt_in(tmp_path: Path) -> None:
    runtime, template, base, yolo = _assets(tmp_path, PROFILE_24)
    source = json.loads(yolo.read_text(encoding="utf-8"))
    for record in source["thresholds"]:
        record["model_version"] = "different-yolo-model"
    _write_json(yolo, source)
    _resign_stage33(yolo)

    with pytest.raises(ValueError, match="do not contain"):
        publish_commissioning_artifact(PROFILE_24, runtime, template, base, yolo, tmp_path / "rejected")

    threshold_path = publish_commissioning_artifact(
        PROFILE_24,
        runtime,
        template,
        base,
        yolo,
        tmp_path / "published",
        allow_yolo_model_rebind=True,
    )
    artifact = json.loads(threshold_path.read_text(encoding="utf-8"))
    assert artifact["yolo_threshold_model_rebound"] is True
    assert any("demo-only" in item for item in json.loads((threshold_path.parent / "summary.json").read_text())["limitations"])


def test_model_rebind_never_claims_source_counts_for_replacement_models(tmp_path: Path) -> None:
    runtime, template, base, yolo = _assets(tmp_path, PROFILE_24)
    for threshold_path in (base, yolo):
        source = json.loads(threshold_path.read_text(encoding="utf-8"))
        for record in source["thresholds"]:
            record["model_version"] = "source-model"
            record["normal_count"] = 999
            record["defect_count"] = 777
            fit = record.get("fit")
            if isinstance(fit, dict):
                fit["negative_count"] = 999
                fit["positive_count"] = 777
        _write_json(threshold_path, source)
        _resign_stage33(threshold_path)

    threshold_path = publish_commissioning_artifact(
        PROFILE_24,
        runtime,
        template,
        base,
        yolo,
        tmp_path / "published",
        allow_yolo_model_rebind=True,
        allow_patchcore_model_rebind=True,
    )
    records = json.loads(threshold_path.read_text(encoding="utf-8"))["thresholds"]

    for record in records:
        if record["branch"] == "template_match":
            assert (record["normal_count"], record["defect_count"]) == (13, 0)
            assert record["count_provenance"] == "template_model_calibration"
        else:
            assert (record["normal_count"], record["defect_count"]) == (0, 0)
            assert record["count_provenance"] == "unavailable_for_rebound_model"
    secondary = [record for record in records if str(record["view"]).endswith("secondary")]
    assert secondary
    assert all(record["defect_count"] == 0 for record in secondary)


def test_publisher_rebinds_roi_identity_only_with_explicit_byte_identical_source(tmp_path: Path) -> None:
    runtime, template, base, yolo = _assets(tmp_path, PROFILE_24)
    source_roi = tmp_path / "source_roi.json"
    runtime_roi = tmp_path / "runtime_roi.json"
    source_roi.write_text('{"roi": "same bytes"}\n', encoding="utf-8")
    runtime_roi.write_bytes(source_roi.read_bytes())

    _bind_roi_rebind_source_contract(runtime, base, yolo, runtime_roi)

    with pytest.raises(ValueError, match="do not contain"):
        publish_commissioning_artifact(PROFILE_24, runtime, template, base, yolo, tmp_path / "default-rejected")
    with pytest.raises(ValueError, match="source ROI"):
        publish_commissioning_artifact(
            PROFILE_24,
            runtime,
            template,
            base,
            yolo,
            tmp_path / "missing-source-rejected",
            allow_roi_version_rebind=True,
        )
    with pytest.raises(ValueError, match="explicit ROI-version-rebind authorization"):
        publish_commissioning_artifact(
            PROFILE_24,
            runtime,
            template,
            base,
            yolo,
            tmp_path / "unauthorized-source-rejected",
            source_roi_config_path=source_roi,
        )

    threshold_path = publish_commissioning_artifact(
        PROFILE_24,
        runtime,
        template,
        base,
        yolo,
        tmp_path / "published",
        allow_roi_version_rebind=True,
        source_roi_config_path=source_roi,
    )
    artifact = json.loads(threshold_path.read_text(encoding="utf-8"))
    summary = json.loads((threshold_path.parent / "summary.json").read_text(encoding="utf-8"))

    assert artifact["roi_version_rebound"] is True
    assert artifact["roi_version_rebind"]["authorized"] is True
    assert artifact["roi_version_rebind"]["source_roi_config"] == {
        "path": str(source_roi.resolve()),
        "sha256": _sha256(source_roi),
    }
    assert artifact["source_artifacts"]["source_roi_config"] == {
        "path": str(source_roi.resolve()),
        "sha256": _sha256(source_roi),
    }
    assert artifact["source_artifacts"]["commissioning_run_contract"]["sha256"] == _sha256(
        base.parent.parent / "commissioning_run_contract.json",
    )
    assert artifact["source_artifacts"]["source_runtime_config"]["sha256"] == _sha256(
        base.parent.parent / "source_runtime.json",
    )
    assert {record["roi_version"] for record in artifact["thresholds"]} == {
        json.loads(template.read_text(encoding="utf-8"))["versions"]["roi"],
    }
    assert summary["roi_version_rebound"] is True
    assert any(_sha256(source_roi) in limitation and "byte-identical" in limitation for limitation in summary["limitations"])


def test_publisher_rejects_source_threshold_roi_not_bound_to_signed_runtime(tmp_path: Path) -> None:
    runtime, template, base, yolo = _assets(tmp_path, PROFILE_24)
    source_roi = tmp_path / "source_roi.json"
    source_roi.write_text('{"roi": "same"}\n', encoding="utf-8")
    _bind_roi_rebind_source_contract(runtime, base, yolo, source_roi)

    base_payload = json.loads(base.read_text(encoding="utf-8"))
    duplicate = dict(base_payload["thresholds"][0])
    duplicate["roi_version"] = "second-legacy-roi-identity"
    base_payload["thresholds"].append(duplicate)
    _write_json(base, base_payload)
    _resign_stage33(base)

    with pytest.raises(ValueError, match="source threshold ROI identities"):
        publish_commissioning_artifact(
            PROFILE_24,
            runtime,
            template,
            base,
            yolo,
            tmp_path / "rejected",
            allow_roi_version_rebind=True,
            source_roi_config_path=source_roi,
        )


@pytest.mark.parametrize("runtime_roi_field", ["patchcore_roi_config", "yolo_roi_config"])
def test_publisher_rejects_roi_identity_rebind_when_runtime_roi_bytes_differ(
    tmp_path: Path,
    runtime_roi_field: str,
) -> None:
    runtime, template, base, yolo = _assets(tmp_path, PROFILE_24)
    source_roi = tmp_path / "source_roi.json"
    matching_roi = tmp_path / "matching_roi.json"
    mismatched_roi = tmp_path / "mismatched_roi.json"
    source_roi.write_text('{"roi": "source"}\n', encoding="utf-8")
    matching_roi.write_bytes(source_roi.read_bytes())
    mismatched_roi.write_text('{"roi": "different"}\n', encoding="utf-8")
    runtime_payload = json.loads(runtime.read_text(encoding="utf-8"))
    runtime_payload["patchcore_roi_config"] = str(matching_roi)
    runtime_payload["yolo_roi_config"] = str(matching_roi)
    runtime_payload[runtime_roi_field] = str(mismatched_roi)
    _write_json(runtime, runtime_payload)
    _bind_roi_rebind_source_contract(runtime, base, yolo, source_roi)
    runtime_payload = json.loads(runtime.read_text(encoding="utf-8"))
    runtime_payload[runtime_roi_field] = str(mismatched_roi)
    _write_json(runtime, runtime_payload)

    with pytest.raises(ValueError, match=rf"{runtime_roi_field}.*SHA256"):
        publish_commissioning_artifact(
            PROFILE_24,
            runtime,
            template,
            base,
            yolo,
            tmp_path / "rejected",
            allow_roi_version_rebind=True,
            source_roi_config_path=source_roi,
        )


def test_publisher_rejects_roi_rebind_when_signed_source_runtime_roi_differs(tmp_path: Path) -> None:
    runtime, template, base, yolo = _assets(tmp_path, PROFILE_24)
    source_roi = tmp_path / "source_roi.json"
    signed_source_runtime_roi = tmp_path / "signed_source_runtime_roi.json"
    source_roi.write_text('{"roi": "new-and-explicit-source"}\n', encoding="utf-8")
    signed_source_runtime_roi.write_text('{"roi": "different-signed-runtime-source"}\n', encoding="utf-8")
    _bind_roi_rebind_source_contract(
        runtime,
        base,
        yolo,
        source_roi,
        source_runtime_roi=signed_source_runtime_roi,
    )

    with pytest.raises(ValueError, match="source runtime patchcore_roi_config SHA256"):
        publish_commissioning_artifact(
            PROFILE_24,
            runtime,
            template,
            base,
            yolo,
            tmp_path / "rejected",
            allow_roi_version_rebind=True,
            source_roi_config_path=source_roi,
        )


def test_publisher_applies_fixed_demo_yolo_threshold_override(tmp_path: Path) -> None:
    """An explicit demo override must set every required YOLO boundary to the same score."""
    runtime, template, base, yolo = _assets(tmp_path, PROFILE_24)

    threshold_path = publish_commissioning_artifact(
        PROFILE_24,
        runtime,
        template,
        base,
        yolo,
        tmp_path / "published",
        yolo_threshold_override=0.07,
    )

    artifact = json.loads(threshold_path.read_text(encoding="utf-8"))
    yolo_records = [record for record in artifact["thresholds"] if record["branch"] == "yolo"]
    assert len(yolo_records) == 8
    assert all(record["low_threshold"] == record["high_threshold"] == pytest.approx(0.07) for record in yolo_records)
    assert all(record["commissioning_source"] == "manual_demo_override" for record in yolo_records)
    assert artifact["yolo_threshold_override"] == pytest.approx(0.07)


def test_publisher_applies_per_view_patchcore_demo_overrides(tmp_path: Path) -> None:
    """Only explicitly named PatchCore views may replace calibrated low/high boundaries."""
    runtime, template, base, yolo = _assets(tmp_path, PROFILE_24)

    threshold_path = publish_commissioning_artifact(
        PROFILE_24,
        runtime,
        template,
        base,
        yolo,
        tmp_path / "published",
        patchcore_threshold_overrides={"front_right": (0.5, 0.666), "back_right": (0.5, 0.55)},
    )

    artifact = json.loads(threshold_path.read_text(encoding="utf-8"))
    by_view = {
        record["view"]: record
        for record in artifact["thresholds"]
        if str(record["branch"]).startswith("anomaly_")
    }
    assert (by_view["front_right"]["low_threshold"], by_view["front_right"]["high_threshold"]) == pytest.approx(
        (0.5, 0.666),
    )
    assert (by_view["back_right"]["low_threshold"], by_view["back_right"]["high_threshold"]) == pytest.approx(
        (0.5, 0.55),
    )
    assert by_view["front_right"]["commissioning_source"] == "manual_demo_override"
    assert by_view["front"]["commissioning_source"] == "stage33_template_patchcore_calibration"


def test_publisher_builds_stage18_valid_eighteen_group_artifact(tmp_path: Path) -> None:
    """Published records must match the profile and preserve live template thresholds."""
    runtime, template, base, yolo = _assets(tmp_path)
    output = tmp_path / "published"

    threshold_path = publish_18_group_artifact(PROFILE, runtime, template, base, yolo, output)
    payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    stage18 = _load_stage18()
    records, metadata = stage18._load_threshold_artifact(threshold_path, profile_path=PROFILE)

    assert len(records) == 18
    assert payload["commissioning_only"] is True
    assert payload["production_release_allowed"] is False
    assert metadata["commissioning_only"] is True
    front_left_template = next(
        record
        for record in payload["thresholds"]
        if record["view"] == "front_left" and record["branch"] == "template_match"
    )
    assert front_left_template["low_threshold"] == pytest.approx(0.11)
    assert all(
        record["low_threshold"] == record["high_threshold"] == pytest.approx(0.9)
        for record in payload["thresholds"]
        if record["branch"] == "yolo"
    )


def test_publisher_rejects_missing_base_group(tmp_path: Path) -> None:
    """No profile group may be invented when its source calibration record is absent."""
    runtime, template, base, yolo = _assets(tmp_path)
    payload = json.loads(base.read_text(encoding="utf-8"))
    payload["thresholds"].pop()
    _write_json(base, payload)
    _write_json(
        base.parent / "stage33_publication.json",
        {"commissioning_only": True, "thresholds_sha256": _sha256(base)},
    )

    with pytest.raises(ValueError, match="source threshold groups do not match"):
        publish_18_group_artifact(PROFILE, runtime, template, base, yolo, tmp_path / "published")


@pytest.mark.parametrize(
    ("source", "field", "value", "error"),
    [
        ("base", "calibration_valid", False, "calibration_valid"),
        ("base", "fit_split", "test", "split contract"),
        ("yolo", "all_views_have_candidate", False, "all required views"),
        ("yolo", "test_used_for_selection", True, "test data"),
        ("yolo", "fit_split", "test", "split contract"),
    ],
)
def test_publisher_rejects_invalid_stage33_payload_policy(
    tmp_path: Path,
    source: str,
    field: str,
    value: object,
    error: str,
) -> None:
    """Stage34 must not relabel an invalid or leaking Stage33 publication as deployable."""
    runtime, template, base, yolo = _assets(tmp_path)
    path = base if source == "base" else yolo
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    _write_json(path, payload)
    _resign_stage33(path)

    with pytest.raises(ValueError, match=error):
        publish_18_group_artifact(PROFILE, runtime, template, base, yolo, tmp_path / "published")


@pytest.mark.parametrize(("source", "status"), [("base", "error"), ("yolo", "unusable")])
def test_publisher_rejects_non_ok_stage33_record(tmp_path: Path, source: str, status: str) -> None:
    """Every emitted ok record must originate from a Stage33 record explicitly marked ok."""
    runtime, template, base, yolo = _assets(tmp_path)
    path = base if source == "base" else yolo
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["thresholds"][0]["status"] = status
    _write_json(path, payload)
    _resign_stage33(path)

    with pytest.raises(ValueError, match="record is not usable"):
        publish_18_group_artifact(PROFILE, runtime, template, base, yolo, tmp_path / "published")


def test_publisher_rejects_yolo_precision_below_declared_minimum(tmp_path: Path) -> None:
    """A YOLO candidate below its own precision policy cannot enter the locked profile."""
    runtime, template, base, yolo = _assets(tmp_path)
    payload = json.loads(yolo.read_text(encoding="utf-8"))
    payload["thresholds"][0]["fit"]["image_presence_precision"] = 0.94
    _write_json(yolo, payload)
    _resign_stage33(yolo)

    with pytest.raises(ValueError, match="precision"):
        publish_18_group_artifact(PROFILE, runtime, template, base, yolo, tmp_path / "published")


def test_source_asset_validation_rejects_runtime_drift(tmp_path: Path) -> None:
    """The commissioning artifact must bind the exact runtime configuration bytes."""
    runtime, template, base, yolo = _assets(tmp_path)
    threshold_path = publish_18_group_artifact(PROFILE, runtime, template, base, yolo, tmp_path / "published")
    runtime_payload = json.loads(runtime.read_text(encoding="utf-8"))
    runtime_payload["yolo"]["candidate_conf"] = 0.123
    _write_json(runtime, runtime_payload)

    with pytest.raises(ValueError, match="runtime config SHA256"):
        validate_18_group_source_assets(threshold_path, runtime, template)


def test_source_asset_validation_rejects_template_drift_even_if_sidecar_is_updated(tmp_path: Path) -> None:
    """Re-signing model.sha256 must not detach thresholds from their original model.json."""
    runtime, template, base, yolo = _assets(tmp_path)
    threshold_path = publish_18_group_artifact(PROFILE, runtime, template, base, yolo, tmp_path / "published")
    template_payload = json.loads(template.read_text(encoding="utf-8"))
    template_payload["groups"]["right/front"]["low_threshold"] = 0.12
    _write_json(template, template_payload)
    (template.parent / "model.sha256").write_text(_sha256(template) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="template model SHA256"):
        validate_18_group_source_assets(threshold_path, runtime, template)


def test_source_asset_validation_allows_identical_bytes_at_new_paths(tmp_path: Path) -> None:
    """Source identity is content-based and must not lock deployment to an absolute path."""
    runtime, template, base, yolo = _assets(tmp_path)
    threshold_path = publish_18_group_artifact(PROFILE, runtime, template, base, yolo, tmp_path / "published")
    moved_runtime = tmp_path / "relocated" / "runtime.json"
    moved_template = tmp_path / "relocated" / "template" / "model.json"
    moved_runtime.parent.mkdir(parents=True)
    moved_template.parent.mkdir(parents=True)
    moved_runtime.write_bytes(runtime.read_bytes())
    moved_template.write_bytes(template.read_bytes())

    validate_18_group_source_assets(threshold_path, moved_runtime, moved_template)


def test_stage18_rejects_commissioning_artifact_without_source_bindings(tmp_path: Path) -> None:
    """A commissioning artifact must carry immutable runtime and template model bindings."""
    runtime, template, base, yolo = _assets(tmp_path)
    threshold_path = publish_18_group_artifact(PROFILE, runtime, template, base, yolo, tmp_path / "published")
    payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    payload.pop("source_artifacts")
    unsigned = dict(payload)
    unsigned.pop("artifact_sha256")
    payload["artifact_sha256"] = hashlib.sha256(
        json.dumps(unsigned, separators=(",", ":"), sort_keys=True).encode(),
    ).hexdigest()
    _write_json(threshold_path, payload)
    stage18 = _load_stage18()

    with pytest.raises(ValueError, match="source_artifacts"):
        stage18._load_threshold_artifact(threshold_path, profile_path=PROFILE)


def test_stage18_rejects_artifact_that_lies_about_commissioning_policy(tmp_path: Path) -> None:
    """A correctly re-signed artifact cannot promote a commissioning profile to production."""
    runtime, template, base, yolo = _assets(tmp_path)
    threshold_path = publish_18_group_artifact(PROFILE, runtime, template, base, yolo, tmp_path / "published")
    payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    payload["commissioning_only"] = False
    payload["production_release_allowed"] = True
    unsigned = dict(payload)
    unsigned.pop("artifact_sha256")
    payload["artifact_sha256"] = hashlib.sha256(
        json.dumps(unsigned, separators=(",", ":"), sort_keys=True).encode(),
    ).hexdigest()
    _write_json(threshold_path, payload)
    stage18 = _load_stage18()

    with pytest.raises(ValueError, match="policy flags do not match"):
        stage18._load_threshold_artifact(threshold_path, profile_path=PROFILE)
