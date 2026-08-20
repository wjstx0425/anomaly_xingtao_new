# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for immutable two-phase ZS32 runtime bundle publication."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import capture_data.zs32_runtime_bundle as runtime_bundle
import pytest
from capture_data.zs32_runtime_bundle import (
    finalize_runtime_bundle,
    load_runtime_assets_manifest,
    load_runtime_bundle,
    publish_directory_no_replace,
    publish_runtime_assets,
)
from capture_data.zs32_model_runtime import sha256_file

from zs32_inspection.domain.views import VIEW_ORDER


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def test_public_manifest_loader_revalidates_bound_assets(source_spec: Path, tmp_path: Path) -> None:
    publication = publish_runtime_assets(source_spec, tmp_path / "assets")

    manifest = load_runtime_assets_manifest(publication.assets_manifest)

    assert manifest["asset_set_sha256"] == publication.asset_set_sha256


def test_public_no_replace_publisher_cleans_failed_generation(tmp_path: Path) -> None:
    output = tmp_path / "publication"

    def fail(staging: Path) -> None:
        (staging / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("injected")

    with pytest.raises(RuntimeError, match="injected"):
        publish_directory_no_replace(output, fail)

    assert not output.exists()
    assert not tuple(tmp_path.glob(".publication.*.tmp"))


@pytest.fixture
def source_spec(tmp_path: Path) -> Path:
    roi = tmp_path / "roi.json"
    roi.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "coordinate_system": "pixel_xyxy_half_open",
                "image_size": {"width": 100, "height": 80},
                "views": {view: {"roi": [1, 2, 90, 70]} for view in VIEW_ORDER},
            },
        ),
        encoding="utf-8",
    )
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    groups = {}
    for view in VIEW_ORDER:
        template = template_dir / "templates" / "right" / view / "template.png"
        template.parent.mkdir(parents=True)
        template.write_bytes(f"template-{view}".encode())
        groups[f"right/{view}"] = {
            "low_threshold": 0.1,
            "high_threshold": 0.2,
            "calibration_normal_count": 10,
            "calibration_defect_count": 0,
            "templates": [
                {
                    "path": str(template.relative_to(template_dir)),
                    "sha256": _sha256(template),
                    "source_part_id": f"normal:{view}",
                },
            ],
        }
    model = {
        "schema": "anomalib.zs32_template_gate",
        "schema_version": "1.0",
        "required_hands": ["right"],
        "required_views": list(VIEW_ORDER),
        "groups": groups,
        "versions": {
            "model": "template-model-v1",
            "threshold": "stage33-threshold-v1",
            "roi": "roi-generation-v1",
            "template": "template-generation-v1",
        },
    }
    model_path = template_dir / "model.json"
    model_path.write_text(json.dumps(model), encoding="utf-8")
    (template_dir / "model.sha256").write_text(_sha256(model_path) + "\n", encoding="utf-8")

    summary = tmp_path / "eight_view_summary.csv"
    with summary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("view", "checkpoint", "deploy_threshold"))
        writer.writeheader()
        for index, view in enumerate(VIEW_ORDER):
            checkpoint = tmp_path / "checkpoints" / f"{view}.ckpt"
            checkpoint.parent.mkdir(exist_ok=True)
            checkpoint.write_bytes(f"checkpoint-{view}".encode())
            writer.writerow(
                {
                    "view": f"right_{view}",
                    "checkpoint": checkpoint,
                    "deploy_threshold": 0.4 + index / 100,
                },
            )
    yolo = tmp_path / "best.pt"
    yolo.write_bytes(b"yolo")
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "profile": "zs32_right_eight_view_commissioning_v1",
                "commissioning_only": True,
                "production_release_allowed": False,
                "identity": {
                    "product": "ZS32",
                    "profile": "zs32_right_eight_view_commissioning_v1",
                    "allowed_hands": ["right"],
                    "required_side": "zs32",
                },
                "required_branches_by_view": {
                    view: ["template_match", f"anomaly_{view}", "yolo"] for view in VIEW_ORDER
                },
                "branch_order": ["template_match", "yolo", *(f"anomaly_{view}" for view in VIEW_ORDER)],
                "rules": {
                    "template_match": {"status_on_positive": "NG_TEMPLATE"},
                    "yolo": {"status_on_positive": "NG_YOLO"},
                    **{f"anomaly_{view}": {"status_on_positive": "NG_ANOMALY"} for view in VIEW_ORDER},
                },
                "expected_versions": [],
            },
        ),
        encoding="utf-8",
    )
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bundle_id": "zs32-right-eight-view-commissioning-test-v1",
                "product": "ZS32",
                "hand": "right",
                "view_order": list(VIEW_ORDER),
                "roi_config": str(roi),
                "roi_version": "roi-generation-v1",
                "template_model_dir": str(template_dir),
                "patchcore_summary": str(summary),
                "yolo_weights": str(yolo),
                "fusion_profile_template": str(profile),
                "commissioning_only": True,
                "production_release_allowed": False,
            },
        ),
        encoding="utf-8",
    )
    return source


def _mutate_json(path: Path, mutate: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)  # type: ignore[operator]
    path.write_text(json.dumps(payload), encoding="utf-8")


def _stage34_records(
    expected: list[dict[str, object]],
    template_model: dict[str, object],
    base: dict[tuple[str, ...], dict[str, object]],
    yolo: dict[tuple[str, ...], dict[str, object]],
) -> list[dict[str, object]]:
    """Use the real dirty-worktree producer, with its exact clean-HEAD contract as fallback."""
    try:
        from capture_data.zs32_18_group_commissioning import _compose_records
    except ModuleNotFoundError:
        records = []
        groups = template_model["groups"]
        assert isinstance(groups, dict)
        identity_fields = ("hand", "view", "branch", "model_version", "roi_version")
        for expectation in sorted(
            expected,
            key=lambda record: tuple(str(record[field]) for field in identity_fields),
        ):
            key = tuple(str(expectation[field]) for field in identity_fields)
            view = str(expectation["view"])
            branch = str(expectation["branch"])
            source = yolo[key] if branch == "yolo" else base[key]
            if branch == "template_match":
                group = groups[f"right/{view}"]
                assert isinstance(group, dict)
                low, high = float(group["low_threshold"]), float(group["high_threshold"])
                provenance = "online_template_model"
            elif branch == "yolo":
                low = high = float(source["threshold"])
                provenance = "stage33_high_precision_auxiliary"
            else:
                low, high = float(source["low_threshold"]), float(source["high_threshold"])
                provenance = "stage33_template_patchcore_calibration"
            records.append(
                {
                    "hand": expectation["hand"],
                    "view": view,
                    "branch": branch,
                    "model_version": expectation["model_version"],
                    "roi_version": expectation["roi_version"],
                    "low_threshold": low,
                    "high_threshold": high,
                    "normal_count": int(source["normal_count"]),
                    "defect_count": int(source["defect_count"]),
                    "status": "ok",
                    "commissioning_source": provenance,
                },
            )
        return records
    return _compose_records(expected, template_model, base, yolo)


def _threshold_artifact(manifest_path: Path, path: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile = json.loads(Path(manifest["fusion_profile"]["path"]).read_text(encoding="utf-8"))
    expected = profile["expected_versions"]
    identity_fields = ("hand", "view", "branch", "model_version", "roi_version")
    template_model = json.loads(Path(manifest["template_model"]["model_json"]["path"]).read_text(encoding="utf-8"))
    keyed = {tuple(str(record[field]) for field in identity_fields): record for record in expected}
    base = {
        key: {"low_threshold": 0.2, "high_threshold": 0.4, "normal_count": 10, "defect_count": 5}
        for key in keyed
        if key[2] != "yolo"
    }
    yolo = {
        key: {"threshold": 0.3, "normal_count": 10, "defect_count": 5}
        for key in keyed
        if key[2] == "yolo"
    }
    records = _stage34_records(expected, template_model, base, yolo)
    required_groups = sorted([str(record[field]) for field in identity_fields] for record in expected)
    payload = {
        "artifact_schema": "anomalib.zs32_fusion_thresholds",
        "artifact_version": "1.0",
        "calibration_valid": True,
        "commissioning_only": True,
        "production_release_allowed": False,
        "profile_sha256": manifest["fusion_profile"]["sha256"],
        "config_sha256": manifest["fusion_profile"]["sha256"],
        "required_views": list(VIEW_ORDER),
        "required_groups": required_groups,
        "thresholds": records,
        "threshold_records_sha256": _canonical_sha256(records),
        "threshold_versions": sorted({str(record["threshold_version"]) for record in expected}),
        "deployment_contract": {
            **profile["identity"],
            "config_sha256": manifest["fusion_profile"]["sha256"],
            "expected_versions": expected,
        },
        "source_artifacts": {
            "runtime_config": manifest["runtime_assets"],
            "template_model": manifest["template_model"]["model_json"],
        },
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_runtime_assets_publication_contains_exact_eight_views(source_spec: Path, tmp_path: Path) -> None:
    publication = publish_runtime_assets(source_spec, tmp_path / "assets")
    payload = json.loads(publication.runtime_assets.read_text(encoding="utf-8"))
    assert tuple(payload["patchcore"]) == VIEW_ORDER
    assert tuple(payload["view_order"]) == VIEW_ORDER
    assert payload["commissioning_only"] is True
    assert payload["production_release_allowed"] is False
    assert all("reported_deploy_threshold" in payload["patchcore"][view] for view in VIEW_ORDER)
    assert "deploy_threshold" not in payload["versions"]


def test_runtime_assets_publication_uses_declared_yolo_imgsz(source_spec: Path, tmp_path: Path) -> None:
    source = json.loads(source_spec.read_text(encoding="utf-8"))
    source["yolo_imgsz"] = 1280
    source_spec.write_text(json.dumps(source), encoding="utf-8")
    publication = publish_runtime_assets(source_spec, tmp_path / "assets")
    payload = json.loads(publication.runtime_assets.read_text(encoding="utf-8"))
    assert payload["yolo"]["imgsz"] == 1280


def test_publication_generates_all_24_profile_versions(source_spec: Path, tmp_path: Path) -> None:
    publication = publish_runtime_assets(source_spec, tmp_path / "assets")
    profile = json.loads(publication.fusion_profile.read_text(encoding="utf-8"))
    assert len(profile["expected_versions"]) == 24
    assert [record["view"] for record in profile["expected_versions"]][::3] == list(VIEW_ORDER)
    assert profile["branch_order"] == ["template_match", "yolo", *(f"anomaly_{view}" for view in VIEW_ORDER)]


def test_v11_24group_source_and_profile_bind_exact_commissioning_contract() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    source = json.loads(
        (repo_root / "config/fusion/zs32_eight_view_24group_v11_bundle_source.json").read_text(encoding="utf-8"),
    )
    profile_path = repo_root / source["fusion_profile_template"]
    profile = json.loads(profile_path.read_text(encoding="utf-8"))

    assert source == {
        "schema_version": 1,
        "bundle_id": "zs32-right-eight-view-24group-0723-v11-commissioning",
        "product": "ZS32",
        "hand": "right",
        "view_order": list(VIEW_ORDER),
        "roi_config": "dataset/zs32_all_plus_0723_retraining_release_v2/roi_config.json",
        "roi_version": "zs32-eight-view-roi-9412b2838cdb",
        "template_model_dir": "results/zs32_template_gate_right_0723_eight_view_v11",
        "patchcore_summary": "results/zs32_patchcore_eight_view_all_plus_0723_seed42_v11/eight_view_summary.csv",
        "yolo_weights": "results/yolo/zs32_all_plus_0723_normal_n1280_seed42_v11/weights/best.pt",
        "yolo_imgsz": 1280,
        "fusion_profile_template": "config/fusion/zs32_right_eight_view_24_group_commissioning_v11.json",
        "commissioning_only": True,
        "production_release_allowed": False,
    }
    assert profile["profile"] == "zs32_right_eight_view_24_group_commissioning_v11"
    assert profile["identity"] == {
        "product": "ZS32",
        "profile": "zs32_right_eight_view_24_group_commissioning_v11",
        "allowed_hands": ["right"],
        "required_side": "zs32",
    }
    assert profile["commissioning_only"] is True
    assert profile["production_release_allowed"] is False
    assert profile["expected_versions"] == []
    assert profile["required_branches_by_view"] == {
        view: ["template_match", f"anomaly_{view}", "yolo"] for view in VIEW_ORDER
    }
    assert sum(len(branches) for branches in profile["required_branches_by_view"].values()) == 24
    assert profile["branch_order"] == ["template_match", "yolo", *(f"anomaly_{view}" for view in VIEW_ORDER)]
    assert set(profile["rules"]) == set(profile["branch_order"])


def test_v12_0727_source_and_profile_change_only_template_release_contract() -> None:
    """The v12 source keeps v11 non-Template runtime inputs byte-identical."""
    repo_root = Path(__file__).resolve().parents[3]
    v11_source = json.loads(
        (repo_root / "config/fusion/zs32_eight_view_24group_v11_bundle_source.json").read_text(encoding="utf-8"),
    )
    source = json.loads(
        (
            repo_root / "config/fusion/zs32_eight_view_24group_template_0727_v12_bundle_source.json"
        ).read_text(encoding="utf-8"),
    )
    profile_path = repo_root / source["fusion_profile_template"]
    profile = json.loads(profile_path.read_text(encoding="utf-8"))

    assert source["bundle_id"] == "zs32-right-eight-view-24group-template-0727-v12-commissioning"
    assert source["template_model_dir"] == "results/zs32_template_gate_right_0727_eight_view_v12"
    assert source["fusion_profile_template"] == (
        "config/fusion/zs32_right_eight_view_24_group_commissioning_template_0727_v12.json"
    )
    for field in ("roi_config", "roi_version", "patchcore_summary", "yolo_weights", "yolo_imgsz"):
        assert source[field] == v11_source[field]
    assert source["view_order"] == list(VIEW_ORDER)
    assert source["commissioning_only"] is True
    assert source["production_release_allowed"] is False

    assert profile["profile"] == "zs32_right_eight_view_24_group_commissioning_template_0727_v12"
    assert profile["identity"]["profile"] == profile["profile"]
    assert profile["expected_versions"] == []
    assert profile["required_branches_by_view"] == {
        view: ["template_match", f"anomaly_{view}", "yolo"] for view in VIEW_ORDER
    }
    assert sum(len(branches) for branches in profile["required_branches_by_view"].values()) == 24
    for view in ("front_secondary", "back_secondary"):
        assert profile["required_branches_by_view"][view] == ["template_match", f"anomaly_{view}", "yolo"]


def test_v13_0727_source_and_profile_change_only_template_release_contract() -> None:
    """The v13 source keeps v12 non-Template runtime inputs byte-identical."""
    repo_root = Path(__file__).resolve().parents[3]
    v12_source = json.loads(
        (
            repo_root / "config/fusion/zs32_eight_view_24group_template_0727_v12_bundle_source.json"
        ).read_text(encoding="utf-8"),
    )
    source = json.loads(
        (
            repo_root / "config/fusion/zs32_eight_view_24group_template_0727_v13_bundle_source.json"
        ).read_text(encoding="utf-8"),
    )
    profile_path = repo_root / source["fusion_profile_template"]
    profile = json.loads(profile_path.read_text(encoding="utf-8"))

    assert source["bundle_id"] == "zs32-right-eight-view-24group-template-0727-v13-commissioning"
    assert source["template_model_dir"] == "results/zs32_template_gate_right_0727_eight_view_v13"
    assert source["fusion_profile_template"] == (
        "config/fusion/zs32_right_eight_view_24_group_commissioning_template_0727_v13.json"
    )
    for field in ("roi_config", "roi_version", "patchcore_summary", "yolo_weights", "yolo_imgsz"):
        assert source[field] == v12_source[field]
    assert source["view_order"] == list(VIEW_ORDER)
    assert source["commissioning_only"] is True
    assert source["production_release_allowed"] is False

    assert profile["profile"] == "zs32_right_eight_view_24_group_commissioning_template_0727_v13"
    assert profile["identity"]["profile"] == profile["profile"]
    assert profile["expected_versions"] == []
    assert profile["required_branches_by_view"] == {
        view: ["template_match", f"anomaly_{view}", "yolo"] for view in VIEW_ORDER
    }
    assert sum(len(branches) for branches in profile["required_branches_by_view"].values()) == 24
    for view in ("front_secondary", "back_secondary"):
        assert profile["required_branches_by_view"][view] == ["template_match", f"anomaly_{view}", "yolo"]


def test_published_v12_bundle_replaces_only_template_artifact_contract() -> None:
    """The published v12 bundle must retain every non-Template v11 runtime asset."""
    repo_root = Path(__file__).resolve().parents[3]
    v11_path = repo_root / "results/zs32_runtime_bundle_eight_view_v11/runtime_bundle.json"
    v12_path = repo_root / "results/zs32_runtime_bundle_eight_view_template_0727_v12/runtime_bundle.json"

    v12 = load_runtime_bundle(v12_path)
    v11_payload = json.loads(v11_path.read_text(encoding="utf-8"))
    v12_payload = json.loads(v12_path.read_text(encoding="utf-8"))
    v11_runtime = json.loads(Path(v11_payload["runtime_assets"]["path"]).read_text(encoding="utf-8"))
    v12_runtime = json.loads(v12.runtime_assets.read_text(encoding="utf-8"))
    v11_assets = json.loads(Path(v11_payload["assets_manifest"]["path"]).read_text(encoding="utf-8"))
    v12_assets = json.loads(Path(v12_payload["assets_manifest"]["path"]).read_text(encoding="utf-8"))
    v12_profile = json.loads(v12.fusion_profile.read_text(encoding="utf-8"))
    v12_thresholds = json.loads(v12.threshold_artifact.read_text(encoding="utf-8"))

    for payload in (v12_payload, v12_runtime, v12_profile, v12_thresholds):
        assert payload["commissioning_only"] is True
        assert payload["production_release_allowed"] is False

    expected = v12_profile["expected_versions"]
    thresholds = v12_thresholds["thresholds"]
    assert len(expected) == len(thresholds) == len(v12_thresholds["required_groups"]) == 24
    assert v12_thresholds["required_views"] == list(VIEW_ORDER)
    assert {record["view"] for record in expected if record["branch"] == "template_match"} == set(VIEW_ORDER)
    assert {record["view"] for record in thresholds if record["branch"] == "template_match"} == set(VIEW_ORDER)
    assert {"front_secondary", "back_secondary"}.issubset(
        record["view"] for record in thresholds if record["branch"] == "template_match"
    )

    v11_template = Path(v11_payload["template_model"]["model_json"]["path"])
    v12_template = Path(v12_payload["template_model"]["model_json"]["path"])
    assert v12.template_model_dir == v12_template.parent.resolve()
    assert v12_template == v12.template_model_dir / "model.json"
    assert sha256_file(v12_template) == v12_payload["template_model"]["model_json"]["sha256"]
    assert sha256_file(v11_template) == v11_payload["template_model"]["model_json"]["sha256"]
    assert v12_template != v11_template
    assert v12_payload["template_model"]["model_json"]["sha256"] != v11_payload["template_model"]["model_json"]["sha256"]

    assert tuple(v12_runtime["patchcore"]) == VIEW_ORDER
    for view in VIEW_ORDER:
        v11_checkpoint = Path(v11_runtime["patchcore"][view]["checkpoint"])
        v12_checkpoint = Path(v12_runtime["patchcore"][view]["checkpoint"])
        assert v12_checkpoint == v11_checkpoint
        assert sha256_file(v12_checkpoint) == v12_runtime["patchcore"][view]["checkpoint_sha256"]
        assert v12_runtime["patchcore"][view]["checkpoint_sha256"] == v11_runtime["patchcore"][view]["checkpoint_sha256"]
    assert v12_assets["asset_set"]["patchcore_checkpoints"] == v11_assets["asset_set"]["patchcore_checkpoints"]

    v11_yolo = v11_runtime["yolo"]
    v12_yolo = v12_runtime["yolo"]
    assert v12_yolo["weights"] == v11_yolo["weights"]
    assert sha256_file(Path(v12_yolo["weights"])) == v12_yolo["weights_sha256"]
    assert v12_yolo["weights_sha256"] == v11_yolo["weights_sha256"]
    assert v12_yolo["class_map"] == v11_yolo["class_map"] == {"0": "defect"}
    assert v12_assets["asset_set"]["yolo_weights_sha256"] == v11_assets["asset_set"]["yolo_weights_sha256"]

    for field in ("patchcore_roi_config", "yolo_roi_config"):
        assert v12_runtime[field] == v11_runtime[field]
        assert sha256_file(Path(v12_runtime[field])) == v12_assets["asset_set"]["roi_config_sha256"]
        assert v12_assets["asset_set"]["roi_config_sha256"] == v11_assets["asset_set"]["roi_config_sha256"]


def test_published_v13_bundle_replaces_only_template_artifact_contract() -> None:
    """The published v13 bundle changes Template while retaining v12 non-Template bindings."""
    repo_root = Path(__file__).resolve().parents[3]
    v12_path = repo_root / "results/zs32_runtime_bundle_eight_view_template_0727_v12/runtime_bundle.json"
    v13_path = repo_root / "results/zs32_runtime_bundle_eight_view_template_0727_v13/runtime_bundle.json"

    v13 = load_runtime_bundle(v13_path)
    v12_payload = json.loads(v12_path.read_text(encoding="utf-8"))
    v13_payload = json.loads(v13_path.read_text(encoding="utf-8"))
    v12_runtime = json.loads(Path(v12_payload["runtime_assets"]["path"]).read_text(encoding="utf-8"))
    v13_runtime = json.loads(v13.runtime_assets.read_text(encoding="utf-8"))
    v12_assets = json.loads(Path(v12_payload["assets_manifest"]["path"]).read_text(encoding="utf-8"))
    v13_assets = json.loads(Path(v13_payload["assets_manifest"]["path"]).read_text(encoding="utf-8"))
    v12_thresholds = json.loads(Path(v12_payload["threshold_artifact"]["path"]).read_text(encoding="utf-8"))
    v13_profile = json.loads(v13.fusion_profile.read_text(encoding="utf-8"))
    v13_thresholds = json.loads(v13.threshold_artifact.read_text(encoding="utf-8"))
    v13_model_path = Path(v13_payload["template_model"]["model_json"]["path"])
    v13_model = json.loads(v13_model_path.read_text(encoding="utf-8"))

    for payload in (v13_payload, v13_runtime, v13_profile, v13_thresholds):
        assert payload["commissioning_only"] is True
        assert payload["production_release_allowed"] is False

    expected = v13_profile["expected_versions"]
    thresholds = v13_thresholds["thresholds"]
    assert len(expected) == len(thresholds) == len(v13_thresholds["required_groups"]) == 24
    assert v13_thresholds["required_views"] == list(VIEW_ORDER)
    assert v13_profile["required_branches_by_view"] == {
        view: ["template_match", f"anomaly_{view}", "yolo"] for view in VIEW_ORDER
    }
    for view in ("front_secondary", "back_secondary"):
        assert v13_profile["required_branches_by_view"][view] == [
            "template_match",
            f"anomaly_{view}",
            "yolo",
        ]

    template_expected = {record["view"]: record for record in expected if record["branch"] == "template_match"}
    template_thresholds = {
        record["view"]: record for record in thresholds if record["branch"] == "template_match"
    }
    assert set(template_expected) == set(template_thresholds) == set(VIEW_ORDER)
    assert set(v13_model["groups"]) == {f"right/{view}" for view in VIEW_ORDER}
    assert v13_payload["template_model"]["model_json"]["sha256"] == (
        "eac211cab4ee1d27adda9933c3cb9c6fde71c4a7eb50fae138201a600fa17ab5"
    )
    assert sha256_file(v13_model_path) == v13_payload["template_model"]["model_json"]["sha256"]
    assert v13.template_model_dir == v13_model_path.parent.resolve()
    for view in VIEW_ORDER:
        model_group = v13_model["groups"][f"right/{view}"]
        expected_record = template_expected[view]
        threshold_record = template_thresholds[view]
        assert expected_record["model_version"] == v13_model["versions"]["model"]
        assert expected_record["template_version"] == v13_model["versions"]["template"]
        assert expected_record["threshold_version"] == v13_model["versions"]["threshold"]
        assert threshold_record["model_version"] == v13_model["versions"]["model"]
        assert threshold_record["low_threshold"] == model_group["low_threshold"]
        assert threshold_record["high_threshold"] == model_group["high_threshold"]

    assert [record for record in thresholds if record["branch"] != "template_match"] == [
        record for record in v12_thresholds["thresholds"] if record["branch"] != "template_match"
    ]
    for field in ("patchcore", "yolo", "patchcore_roi_config", "yolo_roi_config"):
        assert v13_runtime[field] == v12_runtime[field]
    assert v13_assets["asset_set"]["patchcore_checkpoints"] == v12_assets["asset_set"]["patchcore_checkpoints"]
    assert v13_assets["asset_set"]["yolo_weights_sha256"] == v12_assets["asset_set"]["yolo_weights_sha256"]
    assert v13_assets["asset_set"]["roi_config_sha256"] == v12_assets["asset_set"]["roi_config_sha256"]


def test_publication_and_finalize_derive_22_groups(source_spec: Path, tmp_path: Path) -> None:
    source = json.loads(source_spec.read_text(encoding="utf-8"))
    profile_path = Path(source["fusion_profile_template"])
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    for view in ("front_secondary", "back_secondary"):
        profile["required_branches_by_view"][view].remove("template_match")
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    publication = publish_runtime_assets(source_spec, tmp_path / "assets")
    generated = json.loads(publication.fusion_profile.read_text(encoding="utf-8"))
    assert len(generated["expected_versions"]) == 22
    assert not any(
        record["branch"] == "template_match" and record["view"].endswith("secondary")
        for record in generated["expected_versions"]
    )
    threshold = _threshold_artifact(publication.assets_manifest, tmp_path / "thresholds.json")
    assert len(json.loads(threshold.read_text(encoding="utf-8"))["required_groups"]) == 22
    bundle = finalize_runtime_bundle(publication.assets_manifest, threshold, tmp_path / "bundle")
    assert load_runtime_bundle(bundle).fusion_profile == publication.fusion_profile


def test_publication_and_finalize_derive_20_groups_with_yolo_only_secondary(
    source_spec: Path,
    tmp_path: Path,
) -> None:
    """The demo profile may require only YOLO on the two secondary views."""
    source = json.loads(source_spec.read_text(encoding="utf-8"))
    profile_path = Path(source["fusion_profile_template"])
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    for view in ("front_secondary", "back_secondary"):
        profile["required_branches_by_view"][view] = ["yolo"]
        profile["rules"].pop(f"anomaly_{view}")
    profile["branch_order"] = [
        branch
        for branch in profile["branch_order"]
        if branch not in {"anomaly_front_secondary", "anomaly_back_secondary"}
    ]
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    publication = publish_runtime_assets(source_spec, tmp_path / "assets")
    generated = json.loads(publication.fusion_profile.read_text(encoding="utf-8"))
    assert len(generated["expected_versions"]) == 20
    assert generated["required_branches_by_view"]["front_secondary"] == ["yolo"]
    assert generated["required_branches_by_view"]["back_secondary"] == ["yolo"]
    threshold = _threshold_artifact(publication.assets_manifest, tmp_path / "thresholds.json")
    bundle = finalize_runtime_bundle(publication.assets_manifest, threshold, tmp_path / "bundle")
    assert load_runtime_bundle(bundle).fusion_profile == publication.fusion_profile

@pytest.mark.parametrize(
    "malformation",
    ["missing_branch", "wrong_branch", "extra_branch", "branch_order", "missing_rule", "extra_rule"],
)
def test_publication_rejects_malformed_profile_branches(
    source_spec: Path,
    tmp_path: Path,
    malformation: str,
) -> None:
    source = json.loads(source_spec.read_text(encoding="utf-8"))
    profile_path = Path(source["fusion_profile_template"])
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if malformation == "missing_branch":
        profile["required_branches_by_view"]["front"].pop()
    elif malformation == "wrong_branch":
        profile["required_branches_by_view"]["front"][1] = "anomaly_back"
    elif malformation == "extra_branch":
        profile["required_branches_by_view"]["front"].append("other")
    elif malformation == "branch_order":
        profile["branch_order"] = list(reversed(profile["branch_order"]))
    elif malformation == "missing_rule":
        profile["rules"].pop("yolo")
    else:
        profile["rules"]["other"] = {"status_on_positive": "NG"}
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ValueError, match="branch|rules"):
        publish_runtime_assets(source_spec, tmp_path / "assets")


@pytest.mark.parametrize("missing_view", ["front_secondary", "back_secondary"])
def test_publication_rejects_missing_view(source_spec: Path, tmp_path: Path, missing_view: str) -> None:
    source = json.loads(source_spec.read_text(encoding="utf-8"))
    summary = Path(source["patchcore_summary"])
    rows = list(csv.DictReader(summary.open(encoding="utf-8")))
    with summary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(row for row in rows if row["view"] != f"right_{missing_view}")
    with pytest.raises(ValueError, match="exactly.*eight|eight.*views"):
        publish_runtime_assets(source_spec, tmp_path / "assets")


def test_publication_rejects_duplicate_checkpoint(source_spec: Path, tmp_path: Path) -> None:
    source = json.loads(source_spec.read_text(encoding="utf-8"))
    summary = Path(source["patchcore_summary"])
    rows = list(csv.DictReader(summary.open(encoding="utf-8")))
    rows[1]["checkpoint"] = rows[0]["checkpoint"]
    with summary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="distinct checkpoints|duplicate"):
        publish_runtime_assets(source_spec, tmp_path / "assets")


def test_publication_rejects_duplicate_checkpoint_content(source_spec: Path, tmp_path: Path) -> None:
    source = json.loads(source_spec.read_text(encoding="utf-8"))
    summary = Path(source["patchcore_summary"])
    rows = list(csv.DictReader(summary.open(encoding="utf-8")))
    copied = tmp_path / "copied-checkpoint.ckpt"
    copied.write_bytes(Path(rows[0]["checkpoint"]).read_bytes())
    rows[1]["checkpoint"] = str(copied)
    with summary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="distinct checkpoints|duplicate"):
        publish_runtime_assets(source_spec, tmp_path / "assets")


def test_publication_requires_explicit_roi_generation(source_spec: Path, tmp_path: Path) -> None:
    payload = json.loads(source_spec.read_text(encoding="utf-8"))
    payload.pop("roi_version")
    source_spec.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="ROI.*generation|roi_version"):
        publish_runtime_assets(source_spec, tmp_path / "assets")


def test_publication_rejects_missing_checkpoint(source_spec: Path, tmp_path: Path) -> None:
    source = json.loads(source_spec.read_text(encoding="utf-8"))
    summary = Path(source["patchcore_summary"])
    rows = list(csv.DictReader(summary.open(encoding="utf-8")))
    Path(rows[0]["checkpoint"]).unlink()
    with pytest.raises(FileNotFoundError, match="checkpoint does not exist"):
        publish_runtime_assets(source_spec, tmp_path / "assets")


def test_publication_rejects_roi_template_generation_mismatch(source_spec: Path, tmp_path: Path) -> None:
    source = json.loads(source_spec.read_text(encoding="utf-8"))
    profile = Path(source["fusion_profile_template"])
    payload = json.loads(profile.read_text(encoding="utf-8"))
    payload["expected_versions"] = [
        {
            "hand": "right",
            "side": "zs32",
            "view": "front",
            "branch": "template_match",
            "model_version": "template-model-v1",
            "threshold_version": "stage33-threshold-v1",
            "roi_version": "other-roi-generation",
            "template_version": "template-generation-v1",
        },
    ]
    profile.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="ROI.*generation|roi.*version"):
        publish_runtime_assets(source_spec, tmp_path / "assets")


def test_publication_rejects_template_generation_mismatch(source_spec: Path, tmp_path: Path) -> None:
    source = json.loads(source_spec.read_text(encoding="utf-8"))
    profile = Path(source["fusion_profile_template"])
    payload = json.loads(profile.read_text(encoding="utf-8"))
    payload["expected_versions"] = [
        {
            "hand": "right",
            "side": "zs32",
            "view": "front",
            "branch": "template_match",
            "model_version": "template-model-v1",
            "threshold_version": "stage33-threshold-v1",
            "roi_version": "roi-generation-v1",
            "template_version": "other-template-generation",
        },
    ]
    profile.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="[Tt]emplate.*generation|template.*version"):
        publish_runtime_assets(source_spec, tmp_path / "assets")


def test_publication_rejects_non_finite_reported_threshold(source_spec: Path, tmp_path: Path) -> None:
    source = json.loads(source_spec.read_text(encoding="utf-8"))
    summary = Path(source["patchcore_summary"])
    rows = list(csv.DictReader(summary.open(encoding="utf-8")))
    rows[0]["deploy_threshold"] = "nan"
    with summary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="finite"):
        publish_runtime_assets(source_spec, tmp_path / "assets")


@pytest.mark.parametrize(("field", "value"), [("product", "C789"), ("hand", "left")])
def test_publication_rejects_wrong_identity(source_spec: Path, tmp_path: Path, field: str, value: str) -> None:
    _mutate_json(source_spec, lambda payload: payload.__setitem__(field, value))
    with pytest.raises(ValueError, match="ZS32|right"):
        publish_runtime_assets(source_spec, tmp_path / "assets")


def test_publication_rejects_existing_output(source_spec: Path, tmp_path: Path) -> None:
    output = tmp_path / "assets"
    publish_runtime_assets(source_spec, output)
    with pytest.raises(FileExistsError, match="already exists"):
        publish_runtime_assets(source_spec, output)


def test_publication_failure_never_exposes_partial_generation(
    source_spec: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "assets"
    original = runtime_bundle._write_json
    writes = 0

    def fail_second_write(path: Path, payload: object) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("injected publication failure")
        original(path, payload)

    monkeypatch.setattr(runtime_bundle, "_write_json", fail_second_write)
    with pytest.raises(OSError, match="injected"):
        publish_runtime_assets(source_spec, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".assets.*.tmp"))


def test_publication_never_replaces_output_created_during_final_rename(
    source_spec: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "assets"
    original = runtime_bundle._rename_directory_no_replace

    def create_competing_output_before_replace(path: Path, target: Path) -> None:
        output.mkdir()
        original(path, target)

    monkeypatch.setattr(runtime_bundle, "_rename_directory_no_replace", create_competing_output_before_replace)
    with pytest.raises(FileExistsError, match="already exists"):
        publish_runtime_assets(source_spec, output)
    assert output.is_dir()
    assert not any(output.iterdir())
    assert not list(tmp_path.glob(".assets.*.tmp"))


def test_publication_rejects_template_path_escape(source_spec: Path, tmp_path: Path) -> None:
    source = json.loads(source_spec.read_text(encoding="utf-8"))
    model = Path(source["template_model_dir"]) / "model.json"
    _mutate_json(
        model,
        lambda payload: payload["groups"]["right/front"]["templates"][0].__setitem__("path", "../escape.png"),
    )
    (model.parent.parent / "escape.png").write_bytes(b"escape")
    (model.parent / "model.sha256").write_text(_sha256(model) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="escape|contained"):
        publish_runtime_assets(source_spec, tmp_path / "assets")


def test_finalize_rejects_threshold_not_bound_to_exact_asset_hashes(source_spec: Path, tmp_path: Path) -> None:
    publication = publish_runtime_assets(source_spec, tmp_path / "assets")
    threshold = _threshold_artifact(publication.assets_manifest, tmp_path / "thresholds.json")
    _mutate_json(
        threshold,
        lambda payload: payload["source_artifacts"]["runtime_config"].__setitem__("sha256", "0" * 64),
    )
    payload = json.loads(threshold.read_text(encoding="utf-8"))
    payload["artifact_sha256"] = _canonical_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"},
    )
    threshold.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime.assets.*SHA-256|exact runtime"):
        finalize_runtime_bundle(publication.assets_manifest, threshold, tmp_path / "bundle")


def test_finalize_accepts_real_stage34_threshold_record_schema(source_spec: Path, tmp_path: Path) -> None:
    publication = publish_runtime_assets(source_spec, tmp_path / "assets")
    threshold = _threshold_artifact(publication.assets_manifest, tmp_path / "thresholds.json")
    payload = json.loads(threshold.read_text(encoding="utf-8"))
    assert "threshold_version" not in payload["thresholds"][0]
    assert "template_version" not in payload["thresholds"][0]
    assert finalize_runtime_bundle(publication.assets_manifest, threshold, tmp_path / "bundle").is_file()


def test_finalize_rejects_runtime_binding_path_substitution(source_spec: Path, tmp_path: Path) -> None:
    publication = publish_runtime_assets(source_spec, tmp_path / "assets")
    threshold = _threshold_artifact(publication.assets_manifest, tmp_path / "thresholds.json")
    substitute = tmp_path / "runtime-copy.json"
    substitute.write_bytes(publication.runtime_assets.read_bytes())
    payload = json.loads(threshold.read_text(encoding="utf-8"))
    payload["source_artifacts"]["runtime_config"]["path"] = str(substitute)
    payload.pop("artifact_sha256")
    payload["artifact_sha256"] = _canonical_sha256(payload)
    threshold.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime.*path|path.*runtime"):
        finalize_runtime_bundle(publication.assets_manifest, threshold, tmp_path / "bundle")


def test_finalize_rejects_template_binding_path_substitution(source_spec: Path, tmp_path: Path) -> None:
    publication = publish_runtime_assets(source_spec, tmp_path / "assets")
    threshold = _threshold_artifact(publication.assets_manifest, tmp_path / "thresholds.json")
    manifest = json.loads(publication.assets_manifest.read_text(encoding="utf-8"))
    model = Path(manifest["template_model"]["model_json"]["path"])
    substitute = tmp_path / "model-copy.json"
    substitute.write_bytes(model.read_bytes())
    payload = json.loads(threshold.read_text(encoding="utf-8"))
    payload["source_artifacts"]["template_model"]["path"] = str(substitute)
    payload.pop("artifact_sha256")
    payload["artifact_sha256"] = _canonical_sha256(payload)
    threshold.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="template.*path|path.*template"):
        finalize_runtime_bundle(publication.assets_manifest, threshold, tmp_path / "bundle")


def test_finalize_rejects_fusion_profile_template_drift(source_spec: Path, tmp_path: Path) -> None:
    publication = publish_runtime_assets(source_spec, tmp_path / "assets")
    threshold = _threshold_artifact(publication.assets_manifest, tmp_path / "thresholds.json")
    source = json.loads(source_spec.read_text(encoding="utf-8"))
    Path(source["fusion_profile_template"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="profile template.*SHA-256|SHA-256.*profile template"):
        finalize_runtime_bundle(publication.assets_manifest, threshold, tmp_path / "bundle")


def test_finalize_rejects_fusion_profile_template_path_substitution(source_spec: Path, tmp_path: Path) -> None:
    publication = publish_runtime_assets(source_spec, tmp_path / "assets")
    threshold = _threshold_artifact(publication.assets_manifest, tmp_path / "thresholds.json")
    manifest = json.loads(publication.assets_manifest.read_text(encoding="utf-8"))
    substitute = tmp_path / "profile-template-copy.json"
    source = json.loads(source_spec.read_text(encoding="utf-8"))
    substitute.write_bytes(Path(source["fusion_profile_template"]).read_bytes())
    manifest["fusion_profile_template"]["path"] = str(substitute)
    publication.assets_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="profile template.*path|path.*profile template"):
        finalize_runtime_bundle(publication.assets_manifest, threshold, tmp_path / "bundle")


def test_finalize_rejects_asset_set_hash_not_bound_to_manifest_asset(source_spec: Path, tmp_path: Path) -> None:
    publication = publish_runtime_assets(source_spec, tmp_path / "assets")
    threshold = _threshold_artifact(publication.assets_manifest, tmp_path / "thresholds.json")
    manifest = json.loads(publication.assets_manifest.read_text(encoding="utf-8"))
    manifest["asset_set"]["roi_config_sha256"] = "0" * 64
    manifest["asset_set_sha256"] = _canonical_sha256(manifest["asset_set"])
    publication.assets_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="asset-set.*ROI|ROI.*asset-set"):
        finalize_runtime_bundle(publication.assets_manifest, threshold, tmp_path / "bundle")


@pytest.mark.parametrize("location", ["manifest", "asset_set"])
def test_finalize_rejects_unknown_manifest_schema(
    source_spec: Path,
    tmp_path: Path,
    location: str,
) -> None:
    publication = publish_runtime_assets(source_spec, tmp_path / "assets")
    threshold = _threshold_artifact(publication.assets_manifest, tmp_path / "thresholds.json")
    manifest = json.loads(publication.assets_manifest.read_text(encoding="utf-8"))
    if location == "manifest":
        manifest["schema_version"] = 999
    else:
        manifest["asset_set"]["schema_version"] = 999
        manifest["asset_set_sha256"] = _canonical_sha256(manifest["asset_set"])
    publication.assets_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version|schema version"):
        finalize_runtime_bundle(publication.assets_manifest, threshold, tmp_path / "bundle")


@pytest.mark.parametrize("mutation", ["reversed", "missing", "extra"])
def test_finalize_rejects_noncanonical_asset_set_view_order(
    source_spec: Path,
    tmp_path: Path,
    mutation: str,
) -> None:
    publication = publish_runtime_assets(source_spec, tmp_path / "assets")
    threshold = _threshold_artifact(publication.assets_manifest, tmp_path / "thresholds.json")
    manifest = json.loads(publication.assets_manifest.read_text(encoding="utf-8"))
    views = list(VIEW_ORDER)
    if mutation == "reversed":
        views.reverse()
    elif mutation == "missing":
        views.pop()
    else:
        views.append("other")
    manifest["asset_set"]["view_order"] = views
    manifest["asset_set_sha256"] = _canonical_sha256(manifest["asset_set"])
    publication.assets_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="view_order|canonical eight"):
        finalize_runtime_bundle(publication.assets_manifest, threshold, tmp_path / "bundle")


def test_finalize_rejects_threshold_not_bound_to_exact_template_hash(source_spec: Path, tmp_path: Path) -> None:
    publication = publish_runtime_assets(source_spec, tmp_path / "assets")
    threshold = _threshold_artifact(publication.assets_manifest, tmp_path / "thresholds.json")
    payload = json.loads(threshold.read_text(encoding="utf-8"))
    payload["source_artifacts"]["template_model"]["sha256"] = "0" * 64
    payload["artifact_sha256"] = _canonical_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"},
    )
    threshold.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="template model SHA-256"):
        finalize_runtime_bundle(publication.assets_manifest, threshold, tmp_path / "bundle")


def test_finalize_rejects_threshold_without_exact_profile_contract(source_spec: Path, tmp_path: Path) -> None:
    publication = publish_runtime_assets(source_spec, tmp_path / "assets")
    threshold = _threshold_artifact(publication.assets_manifest, tmp_path / "thresholds.json")
    payload = json.loads(threshold.read_text(encoding="utf-8"))
    payload["profile_sha256"] = "0" * 64
    payload.pop("artifact_sha256")
    payload["artifact_sha256"] = _canonical_sha256(payload)
    threshold.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema|profile|deployment|contract"):
        finalize_runtime_bundle(publication.assets_manifest, threshold, tmp_path / "bundle")


def test_finalize_rejects_duplicate_threshold_identity(source_spec: Path, tmp_path: Path) -> None:
    publication = publish_runtime_assets(source_spec, tmp_path / "assets")
    threshold = _threshold_artifact(publication.assets_manifest, tmp_path / "thresholds.json")
    payload = json.loads(threshold.read_text(encoding="utf-8"))
    payload["thresholds"] = [payload["thresholds"][0]] * 24
    payload["threshold_records_sha256"] = _canonical_sha256(payload["thresholds"])
    payload.pop("artifact_sha256")
    payload["artifact_sha256"] = _canonical_sha256(payload)
    threshold.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exact 24 unique|duplicate"):
        finalize_runtime_bundle(publication.assets_manifest, threshold, tmp_path / "bundle")


def test_finalize_rejects_non_finite_stage33_threshold(source_spec: Path, tmp_path: Path) -> None:
    publication = publish_runtime_assets(source_spec, tmp_path / "assets")
    threshold = _threshold_artifact(publication.assets_manifest, tmp_path / "thresholds.json")
    payload = json.loads(threshold.read_text(encoding="utf-8"))
    payload["thresholds"][0]["low_threshold"] = float("nan")
    payload["threshold_records_sha256"] = _canonical_sha256(payload["thresholds"])
    payload.pop("artifact_sha256")
    payload["artifact_sha256"] = _canonical_sha256(payload)
    threshold.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        finalize_runtime_bundle(publication.assets_manifest, threshold, tmp_path / "bundle")


def test_finalize_refuses_existing_output(source_spec: Path, tmp_path: Path) -> None:
    publication = publish_runtime_assets(source_spec, tmp_path / "assets")
    threshold = _threshold_artifact(publication.assets_manifest, tmp_path / "thresholds.json")
    output = tmp_path / "bundle"
    finalize_runtime_bundle(publication.assets_manifest, threshold, output)
    with pytest.raises(FileExistsError, match="already exists"):
        finalize_runtime_bundle(publication.assets_manifest, threshold, output)


def test_bundle_rejects_asset_hash_drift(source_spec: Path, tmp_path: Path) -> None:
    publication = publish_runtime_assets(source_spec, tmp_path / "assets")
    threshold = _threshold_artifact(publication.assets_manifest, tmp_path / "thresholds.json")
    bundle_path = finalize_runtime_bundle(publication.assets_manifest, threshold, tmp_path / "bundle")
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    Path(payload["runtime_assets"]["path"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_runtime_bundle(bundle_path)


def test_bundle_rejects_threshold_artifact_hash_drift(source_spec: Path, tmp_path: Path) -> None:
    publication = publish_runtime_assets(source_spec, tmp_path / "assets")
    threshold = _threshold_artifact(publication.assets_manifest, tmp_path / "thresholds.json")
    bundle_path = finalize_runtime_bundle(publication.assets_manifest, threshold, tmp_path / "bundle")
    threshold.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_runtime_bundle(bundle_path)


def test_bundle_rejects_bundle_id_different_from_assets_manifest(source_spec: Path, tmp_path: Path) -> None:
    publication = publish_runtime_assets(source_spec, tmp_path / "assets")
    threshold = _threshold_artifact(publication.assets_manifest, tmp_path / "thresholds.json")
    bundle_path = finalize_runtime_bundle(publication.assets_manifest, threshold, tmp_path / "bundle")
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    payload["bundle_id"] = "other-bundle"
    payload.pop("bundle_sha256")
    payload["bundle_sha256"] = _canonical_sha256(payload)
    bundle_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="bundle_id"):
        load_runtime_bundle(bundle_path)


def test_load_bundle_rejects_unknown_schema_with_recomputed_hash(source_spec: Path, tmp_path: Path) -> None:
    publication = publish_runtime_assets(source_spec, tmp_path / "assets")
    threshold = _threshold_artifact(publication.assets_manifest, tmp_path / "thresholds.json")
    bundle_path = finalize_runtime_bundle(publication.assets_manifest, threshold, tmp_path / "bundle")
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    payload["schema_version"] = 999
    payload.pop("bundle_sha256")
    payload["bundle_sha256"] = _canonical_sha256(payload)
    bundle_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version|schema version"):
        load_runtime_bundle(bundle_path)


@pytest.mark.parametrize("mutation", ["reversed", "missing", "extra"])
def test_load_bundle_rejects_noncanonical_view_order(
    source_spec: Path,
    tmp_path: Path,
    mutation: str,
) -> None:
    publication = publish_runtime_assets(source_spec, tmp_path / "assets")
    threshold = _threshold_artifact(publication.assets_manifest, tmp_path / "thresholds.json")
    bundle_path = finalize_runtime_bundle(publication.assets_manifest, threshold, tmp_path / "bundle")
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    views = list(VIEW_ORDER)
    if mutation == "reversed":
        views.reverse()
    elif mutation == "missing":
        views.pop()
    else:
        views.append("other")
    payload["view_order"] = views
    payload.pop("bundle_sha256")
    payload["bundle_sha256"] = _canonical_sha256(payload)
    bundle_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="view_order|canonical eight"):
        load_runtime_bundle(bundle_path)


def test_load_runtime_bundle_returns_bound_paths(source_spec: Path, tmp_path: Path) -> None:
    publication = publish_runtime_assets(source_spec, tmp_path / "assets")
    threshold = _threshold_artifact(publication.assets_manifest, tmp_path / "thresholds.json")
    path = finalize_runtime_bundle(publication.assets_manifest, threshold, tmp_path / "bundle")
    bundle = load_runtime_bundle(path)
    assert bundle.bundle_id == "zs32-right-eight-view-commissioning-test-v1"
    assert bundle.asset_set_sha256 == publication.asset_set_sha256
    assert bundle.runtime_assets == publication.runtime_assets
    assert bundle.fusion_profile == publication.fusion_profile
    assert bundle.threshold_artifact == threshold.resolve()


def test_new_python_process_loads_finalized_bundle(source_spec: Path, tmp_path: Path) -> None:
    publication = publish_runtime_assets(source_spec, tmp_path / "assets")
    threshold = _threshold_artifact(publication.assets_manifest, tmp_path / "thresholds.json")
    bundle_path = finalize_runtime_bundle(publication.assets_manifest, threshold, tmp_path / "bundle")
    repo_root = Path(__file__).resolve().parents[3]
    environment = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; "
            "from capture_data.zs32_runtime_bundle import load_runtime_bundle; "
            f"print(load_runtime_bundle(Path({str(bundle_path)!r})).bundle_id)",
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "zs32-right-eight-view-commissioning-test-v1"
