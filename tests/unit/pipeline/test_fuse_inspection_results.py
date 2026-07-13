# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for strict ZS32 stage-18 audit integration."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

VIEWS = ("front", "front_left", "front_right", "back", "back_left", "back_right")


def _load_stage18(name: str) -> ModuleType:
    """Load stage 18 without requiring ``pipeline`` to be a Python package."""
    script_path = Path(__file__).resolve().parents[3] / "pipeline/18_fuse_inspection_results.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load {script_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_complete_predictions(
    tmp_path: Path,
    *,
    missing_artifact: str | None = None,
    gray_evidence: bool = True,
    strong_evidence: bool = False,
    part_id: str = "part001",
    omit_view: str | None = None,
) -> Path:
    """Write all five required rows for every ZS32 view."""
    source_path = tmp_path / "source.png"
    evidence_path = tmp_path / "evidence.png"
    source_path.write_bytes(b"source")
    evidence_path.write_bytes(b"evidence")
    csv_path = tmp_path / "zs32_predictions.csv"
    fieldnames = [
        "part_id",
        "side",
        "view",
        "branch",
        "pred_label",
        "raw_score",
        "low_threshold",
        "high_threshold",
        "status",
        "reason",
        "source_path",
        "evidence_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for view in VIEWS:
            if view == omit_view:
                continue
            for branch in ("quality_gate", "registration", f"anomaly_{view}", "yolo", "geometry"):
                target_row = view == "front" and branch == "anomaly_front"
                is_gray = gray_evidence and target_row
                is_strong = strong_evidence and target_row
                writer.writerow(
                    {
                        "part_id": part_id,
                        "side": "zs32",
                        "view": view,
                        "branch": branch,
                        "pred_label": 0,
                        "raw_score": 0.6 if is_strong else 0.4 if is_gray else 0.2,
                        "low_threshold": 0.3,
                        "high_threshold": 0.5,
                        "status": "PASS" if branch in {"quality_gate", "registration"} else "",
                        "reason": "strong anomaly" if is_strong else "gray anomaly" if is_gray else "clear",
                        "source_path": "" if missing_artifact == "source" and target_row else source_path,
                        "evidence_path": "" if missing_artifact == "evidence" and target_row else evidence_path,
                    },
                )
    return csv_path


def test_zs32_profile_writes_default_atomic_audit_and_summary(tmp_path: Path) -> None:
    """A GRAY six-view inspection should emit a REVIEW audit with raw evidence."""
    stage18 = _load_stage18("pipeline_fuse_inspection_zs32_e2e")
    predictions_csv = _write_complete_predictions(tmp_path)
    output_dir = tmp_path / "out"
    args = stage18.build_parser().parse_args(
        [
            "--profile",
            "zs32",
            "--branch-csv",
            f"normalized={predictions_csv}",
            "--output-dir",
            str(output_dir),
        ],
    )

    decisions = stage18.run_fusion(args)

    assert decisions[0].final_status == "REVIEW"
    audit_path = output_dir / "audit/part001.json"
    assert audit_path.is_file()
    assert not (output_dir / "audit/part001.json.tmp").exists()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    anomaly = next(row for row in audit["views"]["front"] if row["branch"] == "anomaly_front")
    assert anomaly["score"] == pytest.approx(0.4)
    assert audit["triggers"][0]["evidence_id"] == "front:anomaly_front"
    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "| review | 1 |" in summary
    assert "| complete_inspections | 1 |" in summary
    assert "| incomplete_inspections | 0 |" in summary


@pytest.mark.parametrize("missing_artifact", ["source", "evidence"])
def test_require_complete_evidence_forces_review_for_missing_artifact(
    tmp_path: Path,
    missing_artifact: str,
) -> None:
    """Missing source or evidence artifacts must prevent an automatic OK release."""
    stage18 = _load_stage18(f"pipeline_fuse_inspection_missing_{missing_artifact}")
    predictions_csv = _write_complete_predictions(tmp_path, missing_artifact=missing_artifact, gray_evidence=False)
    output_dir = tmp_path / "out"
    args = stage18.build_parser().parse_args(
        [
            "--profile",
            "zs32",
            "--branch-csv",
            f"normalized={predictions_csv}",
            "--require-complete-evidence",
            "--output-dir",
            str(output_dir),
        ],
    )

    decisions = stage18.run_fusion(args)

    assert decisions[0].final_status == "REVIEW"
    assert decisions[0].final_status != "OK"
    audit = json.loads((output_dir / "audit/part001.json").read_text(encoding="utf-8"))
    assert any(trigger["evidence_id"] == "system:incomplete_evidence" for trigger in audit["triggers"])
    assert f"missing {missing_artifact}" in decisions[0].reason
    assert "| incomplete_inspections | 1 |" in (output_dir / "summary.md").read_text(encoding="utf-8")


def test_require_complete_evidence_forces_review_even_with_strong_trigger(tmp_path: Path) -> None:
    """A strong model result cannot bypass the incomplete-evidence review gate."""
    stage18 = _load_stage18("pipeline_fuse_inspection_missing_strong")
    predictions_csv = _write_complete_predictions(
        tmp_path,
        missing_artifact="source",
        gray_evidence=False,
        strong_evidence=True,
    )
    output_dir = tmp_path / "out"
    args = stage18.build_parser().parse_args(
        [
            "--profile",
            "zs32",
            "--branch-csv",
            f"normalized={predictions_csv}",
            "--require-complete-evidence",
            "--output-dir",
            str(output_dir),
        ],
    )

    decisions = stage18.run_fusion(args)

    assert decisions[0].final_status == "REVIEW"
    assert decisions[0].final_label is None
    assert any(trigger.evidence_id == "system:incomplete_evidence" for trigger in decisions[0].triggered_evidence)


def test_zs32_profile_rejects_incompatible_explicit_config(tmp_path: Path) -> None:
    """A named profile and a different config must not be merged silently."""
    stage18 = _load_stage18("pipeline_fuse_inspection_profile_conflict")
    config_path = tmp_path / "other.json"
    config_path.write_text("{}\n", encoding="utf-8")
    args = stage18.build_parser().parse_args(
        [
            "--profile",
            "zs32",
            "--fusion-config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    with pytest.raises(ValueError, match=r"incompatible.*--profile.*--fusion-config"):
        stage18.run_fusion(args)


def test_summary_marks_missing_required_view_as_incomplete(tmp_path: Path) -> None:
    """Present artifacts cannot make an inspection complete when a required view is absent."""
    stage18 = _load_stage18("pipeline_fuse_inspection_missing_view_summary")
    predictions_csv = _write_complete_predictions(tmp_path, gray_evidence=False, omit_view="back_right")
    output_dir = tmp_path / "out"
    args = stage18.build_parser().parse_args(
        ["--profile", "zs32", "--branch-csv", f"normalized={predictions_csv}", "--output-dir", str(output_dir)],
    )

    stage18.run_fusion(args)

    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "| complete_inspections | 0 |" in summary
    assert "| incomplete_inspections | 1 |" in summary


def test_zs32_audit_rejects_part_id_path_escape(tmp_path: Path) -> None:
    """Attacker-controlled part identities must not escape the configured audit directory."""
    stage18 = _load_stage18("pipeline_fuse_inspection_part_id_escape")
    predictions_csv = _write_complete_predictions(tmp_path, gray_evidence=False, part_id="../escaped")
    output_dir = tmp_path / "out"
    audit_dir = tmp_path / "audit"
    args = stage18.build_parser().parse_args(
        [
            "--profile",
            "zs32",
            "--branch-csv",
            f"normalized={predictions_csv}",
            "--audit-dir",
            str(audit_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    with pytest.raises(ValueError, match="unsafe part_id"):
        stage18.run_fusion(args)

    assert not (tmp_path / "escaped.json").exists()
