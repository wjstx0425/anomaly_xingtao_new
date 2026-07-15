# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration contract test for the explicit ZS32 18-group fusion profile."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[3]
PROFILE = REPO_ROOT / "config/fusion/zs32_right_unified_roi_18_group_commissioning.json"
PROFILE_24 = REPO_ROOT / "config/fusion/zs32_right_eight_view_24_group_commissioning.json"


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_stage18() -> ModuleType:
    path = REPO_ROOT / "pipeline/18_fuse_inspection_results.py"
    spec = importlib.util.spec_from_file_location("stage18_18_group_happy_path", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _artifact(
    tmp_path: Path,
    profile: dict[str, object],
    profile_path: Path = PROFILE,
) -> tuple[Path, dict[tuple[str, ...], tuple[float, float]]]:
    records = []
    threshold_by_key = {}
    expected = profile["expected_versions"]
    assert isinstance(expected, list)
    for item in expected:
        assert isinstance(item, dict)
        branch = str(item["branch"])
        low, high = (0.3, 0.5) if branch != "yolo" else (0.8, 0.8)
        record = {
            "hand": item["hand"],
            "view": item["view"],
            "branch": branch,
            "model_version": item["model_version"],
            "roi_version": item["roi_version"],
            "low_threshold": low,
            "high_threshold": high,
            "normal_count": 8,
            "defect_count": 4,
            "status": "ok",
        }
        records.append(record)
        key = tuple(str(record[field]) for field in ("hand", "view", "branch", "model_version", "roi_version"))
        threshold_by_key[key] = (low, high)
    profile_sha = _sha256(profile_path)
    payload = {
        "artifact_schema": "anomalib.zs32_fusion_thresholds",
        "artifact_version": "1.0",
        "calibration_valid": True,
        "commissioning_only": True,
        "production_release_allowed": False,
        "profile_sha256": profile_sha,
        "config_sha256": profile_sha,
        "required_groups": [
            [str(item[field]) for field in ("hand", "view", "branch", "model_version", "roi_version")]
            for item in expected
        ],
        "thresholds": records,
        "threshold_records_sha256": _canonical_sha256(records),
        "source_artifacts": {
            "runtime_config": {"sha256": "a" * 64},
            "template_model": {"sha256": "b" * 64},
        },
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, threshold_by_key


def _predictions(
    tmp_path: Path,
    profile: dict[str, object],
    threshold_by_key: dict[tuple[str, ...], tuple[float, float]],
) -> Path:
    rows = []
    expected = profile["expected_versions"]
    assert isinstance(expected, list)
    for item in expected:
        assert isinstance(item, dict)
        view, branch = str(item["view"]), str(item["branch"])
        source = tmp_path / f"{view}.png"
        source.write_bytes(view.encode())
        evidence = tmp_path / f"{view}-{branch}.png"
        evidence.write_bytes(branch.encode())
        key = tuple(str(item[field]) for field in ("hand", "view", "branch", "model_version", "roi_version"))
        low, high = threshold_by_key[key]
        rows.append(
            {
                "part_id": "right-normal-001",
                "product": "ZS32",
                "profile": profile["profile"],
                "hand": "right",
                "capture_session": "session-001",
                "group_id": "group-001",
                "manifest_identity": f"right-normal-001:right:{view}",
                "source_hash": _sha256(source),
                "evidence_hash": _sha256(evidence),
                "side": "zs32",
                "view": view,
                "branch": branch,
                "pred_label": 0,
                "raw_score": 0.1 if branch != "yolo" else 0.0,
                "low_threshold": low,
                "high_threshold": high,
                "status": "",
                "reason": "clear commissioning evidence",
                "source_path": source,
                "evidence_path": evidence,
                "model_version": item["model_version"],
                "threshold_version": item["threshold_version"],
                "roi_version": item["roi_version"],
                "template_version": item["template_version"],
                "detections": "[]" if branch == "yolo" else "",
            },
        )
    path = tmp_path / "predictions.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_stage18_accepts_complete_clear_eighteen_group_commissioning(tmp_path: Path) -> None:
    """Complete 18-group evidence may be OK only under the named commissioning policy."""
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    artifact, thresholds = _artifact(tmp_path, profile)
    predictions = _predictions(tmp_path, profile, thresholds)
    stage18 = _load_stage18()
    output = tmp_path / "fusion"
    args = stage18.build_parser().parse_args(
        [
            "--profile",
            "zs32-right-18-commissioning",
            "--threshold-artifact",
            str(artifact),
            "--branch-csv",
            f"combined={predictions}",
            "--output-dir",
            str(output),
        ],
    )

    decisions = stage18.run_fusion(args)
    audit = json.loads((output / "audit/right-normal-001.json").read_text(encoding="utf-8"))
    policy = json.loads((output / "fusion_policy.json").read_text(encoding="utf-8"))

    assert len(decisions) == 1
    assert decisions[0].final_status == "OK"
    assert audit["inspection_complete"] is True
    assert audit["fusion_policy"]["commissioning_only"] is True
    assert policy["production_release_allowed"] is False
    assert policy["required_group_count"] == 18


def test_stage18_accepts_complete_clear_twenty_four_group_commissioning(tmp_path: Path) -> None:
    """The named eight-view profile must require and accept exactly 24 clear groups."""
    profile = json.loads(PROFILE_24.read_text(encoding="utf-8"))
    artifact, thresholds = _artifact(tmp_path, profile, PROFILE_24)
    predictions = _predictions(tmp_path, profile, thresholds)
    stage18 = _load_stage18()
    output = tmp_path / "fusion"
    args = stage18.build_parser().parse_args(
        [
            "--profile",
            "zs32-right-24-commissioning",
            "--threshold-artifact",
            str(artifact),
            "--branch-csv",
            f"combined={predictions}",
            "--output-dir",
            str(output),
        ],
    )

    decisions = stage18.run_fusion(args)
    audit = json.loads((output / "audit/right-normal-001.json").read_text(encoding="utf-8"))
    policy = json.loads((output / "fusion_policy.json").read_text(encoding="utf-8"))

    assert len(decisions) == 1
    assert decisions[0].final_status == "OK"
    assert audit["inspection_complete"] is True
    assert audit["fusion_policy"]["commissioning_only"] is True
    assert policy["production_release_allowed"] is False
    assert policy["required_group_count"] == 24
