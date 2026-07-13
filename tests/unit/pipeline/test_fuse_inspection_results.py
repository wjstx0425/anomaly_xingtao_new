# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for strict ZS32 stage-18 audit integration."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
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
        "product",
        "profile",
        "hand",
        "session_id",
        "timestamp",
        "manifest_identity",
        "source_hash",
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
        "threshold_version",
        "roi_version",
        "template_version",
        "evidence_level",
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
                    "part_id": "wrong-part" if fault == "split_part_ids" and view == "back_right" else part_id,
                    "product": "ZS32",
                    "profile": "zs32_six_view_v1",
                    "hand": "left",
                    "session_id": "session-7",
                    "timestamp": "2026-07-13T12:34:56+08:00",
                    "manifest_identity": f"{part_id}:left:{view}",
                    "source_hash": hashlib.sha256(source_path.read_bytes()).hexdigest(),
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
                        "wrong-model"
                        if fault == "model_version_mismatch" and view == "back_right" and branch == "yolo"
                        else "zs32-models-2026.07.13"
                    ),
                    "threshold_version": "zs32-thresholds-2026.07.13",
                    "roi_version": (
                        "wrong-roi"
                        if fault == "roi_version_mismatch" and view == "back_right" and branch == "yolo"
                        else "zs32-roi-2026.07.12"
                    ),
                    "template_version": "zs32-templates-2026.07.13",
                    "evidence_level": ("BROKEN" if fault == "malformed_evidence_level" and target_row else ""),
                }
                if fault == "malformed_thresholds" and target_row:
                    row["low_threshold"] = 0.7
                    row["high_threshold"] = 0.5
                if fault == "model_timeout" and target_row:
                    row.update(status="TIMEOUT", raw_score=0.2, reason="model inference timeout")
                if fault == "roi_failure" and view == "front" and branch == "registration":
                    row.update(status="FAIL", pred_label=1, reason="ROI localization failed")
                if fault == "quality_blur" and view == "front" and branch == "quality_gate":
                    row.update(status="FAIL", pred_label=1, reason="blur_laplacian_var below minimum")
                if fault == "quality_overexposure" and view == "front" and branch == "quality_gate":
                    row.update(status="FAIL", pred_label=1, reason="highlight_ratio above maximum")
                if fault == "drift" and target_row:
                    row.update(status="DRIFT", raw_score=0.2, reason="camera score drift")
                if fault == "duplicate_source_hash" and view == "back_right":
                    duplicate = tmp_path / "front_source.png"
                    row["source_path"] = duplicate
                if fault == "yolo_empty_gray_anomaly" and branch == "yolo":
                    row.update(raw_score=0.0, pred_label=0, reason="no detections")
                writer.writerow(row)
                if fault == "duplicate_view_identity" and view == "front" and branch == "anomaly_front":
                    writer.writerow(row)
    return csv_path


def test_zs32_profile_writes_atomic_audit_and_summary(tmp_path: Path) -> None:
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
    assert audit["schema_version"] == "2.0"
    assert audit["product"] == "ZS32"
    assert audit["profile"] == "zs32_six_view_v1"
    assert audit["hand"] == "left"
    assert audit["session_id"] == "session-7"
    assert audit["timestamp"] == "2026-07-13T12:34:56+08:00"
    assert audit["inspection_complete"] is True
    assert audit["machine_status"] == "REVIEW"
    assert audit["review_status"] is None
    assert audit["released_status"] is None
    assert anomaly["score"] == pytest.approx(0.4)
    assert anomaly["computed_evidence_level"] == "GRAY"
    assert anomaly["normalized_risk"] == pytest.approx(0.75)
    assert anomaly["source_sha256"]
    assert anomaly["evidence_sha256"]
    assert anomaly["low_margin"] == pytest.approx(0.1)
    assert anomaly["high_margin"] == pytest.approx(-0.1)
    assert anomaly["model_version"] == "zs32-models-2026.07.13"
    assert anomaly["threshold_version"] == "zs32-thresholds-2026.07.13"
    assert anomaly["roi_version"] == "zs32-roi-2026.07.12"
    assert anomaly["template_version"] == "zs32-templates-2026.07.13"
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
    """Missing artifacts block release without erasing an immutable machine NG."""
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

    assert decisions[0].final_status == "NG_ANOMALY"
    assert decisions[0].final_label == 1
    assert any(trigger.evidence_id == "system:incomplete_evidence" for trigger in decisions[0].triggered_evidence)


def test_capture_fault_blocks_release_without_downgrading_strong_ng(tmp_path: Path) -> None:
    """A duplicate capture remains incomplete but cannot erase machine STRONG evidence."""
    stage18 = _load_stage18("pipeline_fuse_inspection_duplicate_source_strong")
    predictions_csv = _write_complete_predictions(
        tmp_path,
        gray_evidence=False,
        strong_evidence=True,
        fault="duplicate_source_hash",
    )
    output_dir = tmp_path / "out"
    args = stage18.build_parser().parse_args(
        ["--profile", "zs32", "--branch-csv", f"normalized={predictions_csv}", "--output-dir", str(output_dir)],
    )

    decisions = stage18.run_fusion(args)

    assert decisions[0].final_status == "NG_ANOMALY"
    assert decisions[0].final_label == 1
    assert decisions[0].triggered_branch == "anomaly_front"
    audit = json.loads((output_dir / "audit/part001.json").read_text(encoding="utf-8"))
    assert audit["machine_status"] == "NG_ANOMALY"
    assert audit["inspection_complete"] is False
    assert any(trigger["evidence_id"] == "system:identity_fault" for trigger in audit["triggers"])
    assert audit["released_status"] is None


def test_zs32_always_requires_evidence_without_optional_flag(tmp_path: Path) -> None:
    """The strict profile cannot bypass evidence completeness through CLI omission."""
    stage18 = _load_stage18("pipeline_fuse_inspection_mandatory_evidence")
    predictions_csv = _write_complete_predictions(tmp_path, missing_artifact="source", gray_evidence=False)
    output_dir = tmp_path / "out"
    args = stage18.build_parser().parse_args(
        ["--profile", "zs32", "--branch-csv", f"normalized={predictions_csv}", "--output-dir", str(output_dir)],
    )

    decisions = stage18.run_fusion(args)

    assert decisions[0].final_status == "REVIEW"
    assert any(item.evidence_id == "system:incomplete_evidence" for item in decisions[0].triggered_evidence)


def test_generation_refuses_existing_final_output(tmp_path: Path) -> None:
    """A published generation is immutable and cannot be overwritten."""
    stage18 = _load_stage18("pipeline_fuse_inspection_existing_generation")
    predictions_csv = _write_complete_predictions(tmp_path, gray_evidence=False)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "sentinel").write_text("published", encoding="utf-8")
    args = stage18.build_parser().parse_args(
        ["--profile", "zs32", "--branch-csv", f"normalized={predictions_csv}", "--output-dir", str(output_dir)],
    )

    with pytest.raises(FileExistsError, match="already exists"):
        stage18.run_fusion(args)

    assert (output_dir / "sentinel").read_text(encoding="utf-8") == "published"


@pytest.mark.parametrize(
    "failure",
    ["audit", "audit_json_write", "source_hash", "evidence_hash", "write", "rename"],
)
def test_publication_failure_leaves_no_consumable_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """Audit and final rename failures cannot expose an OK CSV generation."""
    stage18 = _load_stage18(f"pipeline_fuse_inspection_publish_failure_{failure}")
    predictions_csv = _write_complete_predictions(tmp_path, gray_evidence=False)
    output_dir = tmp_path / "out"
    args = stage18.build_parser().parse_args(
        ["--profile", "zs32", "--branch-csv", f"normalized={predictions_csv}", "--output-dir", str(output_dir)],
    )

    if failure == "audit":
        failure_message = "audit publication failed"

        def fail_audit(*_args: object, **_kwargs: object) -> Path:
            raise OSError(failure_message)

        monkeypatch.setattr(stage18, "write_part_audit", fail_audit)
    elif failure == "audit_json_write":
        original_write_text = Path.write_text
        failure_message = "audit JSON write failed"

        def fail_audit_json_write(self: Path, *write_args: object, **write_kwargs: object) -> int:
            if self.name.endswith(".json.tmp"):
                raise OSError(failure_message)
            return original_write_text(self, *write_args, **write_kwargs)

        monkeypatch.setattr(Path, "write_text", fail_audit_json_write)
    elif failure == "source_hash":
        failure_message = "source hash failed"

        def fail_hash(_path: Path) -> str:
            raise OSError(failure_message)

        monkeypatch.setattr(stage18, "sha256_file", fail_hash)
    elif failure == "evidence_hash":
        import capture_data.inspection_audit as audit_module

        original_hash = audit_module.sha256_file
        failure_message = "evidence hash failed"

        def fail_evidence_hash(path: Path) -> str:
            if "evidence" in path.name:
                raise OSError(failure_message)
            return original_hash(path)

        monkeypatch.setattr(audit_module, "sha256_file", fail_evidence_hash)
    elif failure == "write":
        failure_message = "CSV write failed"

        def fail_write(*_args: object, **_kwargs: object) -> None:
            raise OSError(failure_message)

        monkeypatch.setattr(stage18.fusion, "write_fused_decisions_csv", fail_write)
    else:
        original_replace = Path.replace
        failure_message = "generation rename failed"

        def fail_generation_replace(self: Path, target: Path) -> Path:
            if target == output_dir:
                raise OSError(failure_message)
            return original_replace(self, target)

        monkeypatch.setattr(Path, "replace", fail_generation_replace)

    with pytest.raises(OSError, match=failure_message):
        stage18.run_fusion(args)

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".out.staging-*"))


@pytest.mark.parametrize("fault", ["malformed_thresholds", "malformed_evidence_level"])
def test_malformed_strict_input_publishes_diagnostic_then_returns_nonzero(tmp_path: Path, fault: str) -> None:
    """Malformed strict evidence publishes a fail-closed audit before raising."""
    stage18 = _load_stage18(f"pipeline_fuse_inspection_diagnostic_{fault}")
    predictions_csv = _write_complete_predictions(tmp_path, gray_evidence=False, fault=fault)
    output_dir = tmp_path / "out"
    args = stage18.build_parser().parse_args(
        ["--profile", "zs32", "--branch-csv", f"normalized={predictions_csv}", "--output-dir", str(output_dir)],
    )

    with pytest.raises(RuntimeError, match="diagnostic generation"):
        stage18.run_fusion(args)

    fused_rows = list(csv.DictReader((output_dir / "fused_predictions.csv").open(encoding="utf-8")))
    assert fused_rows
    assert all(row["final_status"] in {"INVALID_CAPTURE", "REVIEW"} for row in fused_rows)
    audits = list((output_dir / "audit").glob("*.json"))
    assert audits
    assert all(json.loads(path.read_text(encoding="utf-8"))["released_status"] is None for path in audits)


def test_malformed_strict_cli_exits_nonzero_after_publishing_diagnostic(tmp_path: Path) -> None:
    """The actual CLI process must fail only after the diagnostic generation becomes readable."""
    predictions_csv = _write_complete_predictions(tmp_path, gray_evidence=False, fault="malformed_evidence_level")
    output_dir = tmp_path / "out"
    script = Path(__file__).resolve().parents[3] / "pipeline/18_fuse_inspection_results.py"

    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(script),
            "--profile",
            "zs32",
            "--branch-csv",
            f"normalized={predictions_csv}",
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "diagnostic generation published" in result.stderr
    assert (output_dir / "fused_predictions.csv").is_file()
    audit = json.loads((output_dir / "audit/part001.json").read_text(encoding="utf-8"))
    assert audit["released_status"] is None


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
    audit_dir = output_dir / "audit"
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
        pytest.param("split_part_ids", None, None, id="split_part_ids"),
        pytest.param("quality_fail", None, None, id="quality_fail"),
        pytest.param("registration_warn", None, None, id="registration_warn"),
        pytest.param("nonfinite_score", None, None, id="nonfinite_score"),
        pytest.param("model_version_mismatch", None, None, id="model_version_mismatch"),
        pytest.param("roi_version_mismatch", None, None, id="roi_version_mismatch"),
        pytest.param(None, "source", None, id="missing_source_image"),
        pytest.param(None, "evidence", None, id="missing_evidence_artifact"),
        pytest.param("yolo_empty_gray_anomaly", None, None, id="yolo_empty_with_gray_anomaly"),
        pytest.param("model_timeout", None, None, id="model_timeout"),
        pytest.param("roi_failure", None, None, id="roi_failure"),
        pytest.param("drift", None, None, id="camera_drift"),
        pytest.param("duplicate_source_hash", None, None, id="duplicate_source_hash"),
        pytest.param("quality_blur", None, None, id="quality_blur"),
        pytest.param("quality_overexposure", None, None, id="quality_overexposure"),
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

    if fault == "nonfinite_score":
        with pytest.raises(RuntimeError, match="diagnostic generation"):
            stage18.run_fusion(args)
        fused_rows = list(csv.DictReader((output_dir / "fused_predictions.csv").open(encoding="utf-8")))
        assert fused_rows[0]["final_status"] == "INVALID_CAPTURE"
        audit = json.loads((output_dir / "audit/part001.json").read_text(encoding="utf-8"))
        assert audit["released_status"] is None
        assert audit["inspection_complete"] is False
        return

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


def test_fault_mismatched_face_part_ids_are_rejected() -> None:
    """Front/back stages from different physical parts must not be combined."""
    stage18 = _load_stage18("pipeline_fuse_inspection_fault_face_identity")
    front = stage18.fusion.FaceDecision(
        part_id="part001",
        face="front",
        stage_status="FRONT_CLEAR",
        final_status=None,
        inspection_complete=False,
        triggered_evidence=(),
        reason="clear",
    )
    back = stage18.fusion.FaceDecision(
        part_id="wrong-part",
        face="back",
        stage_status="BACK_CLEAR",
        final_status=None,
        inspection_complete=False,
        triggered_evidence=(),
        reason="clear",
    )

    with pytest.raises(ValueError, match="part_id mismatch"):
        stage18.fusion.combine_face_decisions(front, back)
