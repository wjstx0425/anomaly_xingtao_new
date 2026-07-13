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
    fault: str | None = None,
) -> Path:
    """Write all five required rows for every ZS32 view."""
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
        "model_version",
        "roi_version",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for view in VIEWS:
            if view == omit_view:
                continue
            source_path = tmp_path / f"{view}_source.png"
            source_path.write_bytes(view.encode())
            for branch in ("quality_gate", "registration", f"anomaly_{view}", "yolo", "geometry"):
                if fault == "missing_branch" and view == "back_right" and branch == "geometry":
                    continue
                evidence_path = tmp_path / f"{view}_{branch}_evidence.png"
                evidence_path.write_bytes(branch.encode())
                target_row = view == "front" and branch == "anomaly_front"
                is_gray = gray_evidence and target_row
                is_strong = strong_evidence and target_row
                row = {
                    "part_id": "wrong-part" if fault == "mismatched_part_id" and view == "back_right" else part_id,
                    "side": "zs32",
                    "view": view,
                    "branch": branch,
                    "pred_label": 0,
                    "raw_score": (
                        "nan"
                        if fault == "nonfinite_score" and target_row
                        else 0.6
                        if is_strong
                        else 0.4
                        if is_gray
                        else 0.2
                    ),
                    "low_threshold": 0.3,
                    "high_threshold": 0.5,
                    "status": (
                        "FAIL"
                        if fault == "quality_fail" and view == "front" and branch == "quality_gate"
                        else "WARN"
                        if fault == "registration_warn" and view == "front" and branch == "registration"
                        else "EMPTY"
                        if fault == "yolo_empty_gray_anomaly" and branch == "yolo"
                        else "PASS"
                        if branch in {"quality_gate", "registration"}
                        else ""
                    ),
                    "reason": "strong anomaly" if is_strong else "gray anomaly" if is_gray else "clear",
                    "source_path": "" if missing_artifact == "source" and target_row else source_path,
                    "evidence_path": "" if missing_artifact == "evidence" and target_row else evidence_path,
                    "model_version": (
                        "m2"
                        if fault == "model_version_mismatch" and view == "back_right" and branch == "yolo"
                        else "m1"
                    ),
                    "roi_version": (
                        "roi2"
                        if fault == "roi_version_mismatch" and view == "back_right" and branch == "yolo"
                        else "roi1"
                    ),
                }
                if fault == "yolo_empty_gray_anomaly" and branch == "yolo":
                    row.update(raw_score=0.0, pred_label=0, reason="no detections")
                writer.writerow(row)
                if fault == "duplicate_view_identity" and view == "front" and branch == "anomaly_front":
                    writer.writerow(row)
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


@pytest.mark.parametrize(
    ("fault", "missing_artifact", "omit_view"),
    [
        pytest.param(None, None, "back_right", id="missing_view"),
        pytest.param("missing_branch", None, None, id="missing_branch"),
        pytest.param("duplicate_view_identity", None, None, id="duplicate_view_identity"),
        pytest.param("mismatched_part_id", None, None, id="mismatched_part_id"),
        pytest.param("quality_fail", None, None, id="quality_fail"),
        pytest.param("registration_warn", None, None, id="registration_warn"),
        pytest.param("nonfinite_score", None, None, id="nonfinite_score"),
        pytest.param("model_version_mismatch", None, None, id="model_version_mismatch"),
        pytest.param("roi_version_mismatch", None, None, id="roi_version_mismatch"),
        pytest.param(None, "source", None, id="missing_source_image"),
        pytest.param(None, "evidence", None, id="missing_evidence_artifact"),
        pytest.param("yolo_empty_gray_anomaly", None, None, id="yolo_empty_with_gray_anomaly"),
    ],
)
def test_fault_injection_never_releases_ok(
    tmp_path: Path,
    fault: str | None,
    missing_artifact: str | None,
    omit_view: str | None,
) -> None:
    """Every required-input, gate, version, and evidence fault must fail closed."""
    stage18 = _load_stage18(f"pipeline_fuse_inspection_fault_{fault or missing_artifact or omit_view}")
    predictions_csv = _write_complete_predictions(
        tmp_path,
        fault=fault,
        missing_artifact=missing_artifact,
        omit_view=omit_view,
        gray_evidence=fault == "yolo_empty_gray_anomaly",
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

    assert decisions
    for decision in decisions:
        assert decision.final_status != "OK"
        audit = json.loads((output_dir / f"audit/{decision.part_id}.json").read_text(encoding="utf-8"))
        assert audit["released_status"] is None
        assert audit["machine_status"] in {
            "REVIEW",
            "RETAKE",
            "INVALID_CAPTURE",
            "NG_ANOMALY",
            "NG_GEOMETRY",
            "NG_YOLO",
        }
