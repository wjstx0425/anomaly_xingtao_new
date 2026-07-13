# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for immutable ZS32 inspection audit evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


def _load_module(name: str, filename: str) -> ModuleType:
    """Load one capture-data helper from its file path."""
    script_path = Path(__file__).resolve().parents[3] / "capture_data" / filename
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load {script_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_part_audit_retains_complete_machine_evidence(tmp_path: Path) -> None:
    """Audit JSON keeps six views, scores, thresholds, margins, versions, hashes, and triggers."""
    fusion = _load_module("audit_test_fusion", "fusion_engine.py")
    audit_module = _load_module("audit_test_module", "inspection_audit.py")
    source = tmp_path / "source.png"
    evidence = tmp_path / "heatmap.png"
    source.write_bytes(b"x")
    evidence.write_bytes(b"y")
    prediction = fusion.BranchPrediction(
        part_id="p1",
        side="zs32",
        view="front",
        slot_id=None,
        branch="anomaly_front",
        pred_label=0,
        score=0.4,
        threshold=None,
        defect_type=None,
        reason="gray evidence",
        source_path=str(source),
        evidence_path=str(evidence),
        low_threshold=0.3,
        high_threshold=0.5,
        model_version="model-v1",
        threshold_version="threshold-v2",
        roi_version="roi-v3",
        template_version="template-v4",
        hand="left",
        product="ZS32",
        profile="zs32_six_view_v1",
        manifest_identity="manifest-part-1",
    )
    triggers = (
        fusion.TriggerEvidence("front:anomaly_front", "anomaly_front", "zs32", "front", "GRAY", 0.4, 0.3, 0.5, "gray"),
    )

    audit = audit_module.build_part_audit(
        "p1",
        [prediction],
        machine_status="REVIEW",
        triggered_evidence=triggers,
        inspection_complete=False,
        session_id="session-7",
        timestamp="2026-07-13T12:34:56+08:00",
    )

    assert audit["schema_version"] == "2.0"
    assert set(audit["views"]) == {"front", "front_left", "front_right", "back", "back_left", "back_right"}
    branch = audit["views"]["front"][0]
    assert branch["score"] == pytest.approx(0.4)
    assert branch["low_threshold"] == pytest.approx(0.3)
    assert branch["high_threshold"] == pytest.approx(0.5)
    assert branch["low_margin"] == pytest.approx(0.1)
    assert branch["high_margin"] == pytest.approx(-0.1)
    assert branch["model_version"] == "model-v1"
    assert branch["threshold_version"] == "threshold-v2"
    assert branch["roi_version"] == "roi-v3"
    assert branch["template_version"] == "template-v4"
    assert branch["computed_evidence_level"] == "GRAY"
    assert branch["normalized_risk"] == pytest.approx(0.75)
    assert branch["source_sha256"] == hashlib.sha256(b"x").hexdigest()
    assert branch["evidence_sha256"] == hashlib.sha256(b"y").hexdigest()
    assert branch["evidence_path"] == str(evidence)
    assert audit["triggers"][0]["evidence_id"] == "front:anomaly_front"
    assert audit["machine_status"] == "REVIEW"
    assert audit["hand"] == "left"
    assert audit["product"] == "ZS32"
    assert audit["profile"] == "zs32_six_view_v1"
    assert audit["session_id"] == "session-7"
    assert audit["timestamp"] == "2026-07-13T12:34:56+08:00"
    assert audit["inspection_complete"] is False
    assert audit["review"]["status"] == "PENDING"
    assert audit["review_status"] is None
    assert audit["released_status"] is None


def test_malformed_evidence_is_serialized_as_invalid_diagnostic(tmp_path: Path) -> None:
    """Malformed explicit evidence stays auditable instead of aborting serialization."""
    fusion = _load_module("audit_invalid_fusion", "fusion_engine.py")
    audit_module = _load_module("audit_invalid_module", "inspection_audit.py")
    source = tmp_path / "source.png"
    evidence = tmp_path / "evidence.png"
    source.write_bytes(b"source")
    evidence.write_bytes(b"evidence")
    prediction = fusion.BranchPrediction(
        part_id="p1",
        side="zs32",
        view="front",
        slot_id=None,
        branch="anomaly_front",
        pred_label=0,
        score=0.4,
        threshold=None,
        defect_type=None,
        reason="malformed level",
        source_path=str(source),
        evidence_path=str(evidence),
        low_threshold=0.3,
        high_threshold=0.5,
        evidence_level="BROKEN",
    )

    audit = audit_module.build_part_audit(
        "p1",
        [prediction],
        machine_status="INVALID_CAPTURE",
        triggered_evidence=(),
        inspection_complete=False,
    )

    branch = audit["views"]["front"][0]
    assert branch["computed_evidence_level"] == "INVALID"
    assert "BROKEN" in branch["evidence_error"]


def test_audit_preserves_capture_identity_and_does_not_invent_timestamp() -> None:
    """Acquisition evidence stays explicit and an absent timestamp remains JSON null."""
    fusion = _load_module("audit_identity_fusion", "fusion_engine.py")
    audit_module = _load_module("audit_identity_module", "inspection_audit.py")
    prediction = fusion.BranchPrediction(
        "p1",
        "zs32",
        "front",
        None,
        "geometry",
        0,
        0.1,
        None,
        None,
        None,
        None,
        capture_session="session-001",
        group_id="group-001",
    )

    audit = audit_module.build_part_audit("p1", [prediction], machine_status="OK", triggered_evidence=())

    assert audit["capture_session"] == "session-001"
    assert audit["group_id"] == "group-001"
    assert audit["timestamp"] is None


def test_audit_retains_each_yolo_box_and_forces_incomplete_for_gate_trigger() -> None:
    """Structured detections remain per-box while gate evidence blocks inspection completion."""
    fusion = _load_module("audit_yolo_fusion", "fusion_engine.py")
    audit_module = _load_module("audit_yolo_module", "inspection_audit.py")
    detections = (
        {"class": "crack", "confidence": 0.91, "xyxy": [1, 2, 11, 22], "area": 200, "in_roi": True},
        {"class": "chip", "confidence": 0.82, "xyxy": [30, 40, 50, 70], "area": 600, "touches_border": True},
    )
    prediction = fusion.BranchPrediction(
        "p1",
        "zs32",
        "front",
        None,
        "yolo",
        1,
        0.91,
        None,
        "crack",
        None,
        None,
        detections=detections,
    )
    gate = fusion.TriggerEvidence(
        "front:quality_gate",
        "quality_gate",
        "zs32",
        "front",
        "GRAY",
        None,
        None,
        None,
        "WARN",
    )

    audit = audit_module.build_part_audit(
        "p1",
        [prediction],
        machine_status="NG_YOLO",
        triggered_evidence=(gate,),
        inspection_complete=True,
    )

    assert audit["views"]["front"][0]["detections"] == list(detections)
    assert audit["inspection_complete"] is False
    assert audit["machine"]["inspection_complete"] is False


def test_missing_evidence_path_is_retained(tmp_path: Path) -> None:
    """Missing evidence artifacts remain explicit instead of disappearing from the audit."""
    fusion = _load_module("audit_missing_fusion", "fusion_engine.py")
    audit_module = _load_module("audit_missing_module", "inspection_audit.py")
    missing = tmp_path / "missing.png"
    prediction = fusion.BranchPrediction(
        "p1",
        "zs32",
        "front",
        None,
        "geometry",
        1,
        0.8,
        None,
        "less",
        None,
        None,
        evidence_path=str(missing),
    )

    audit = audit_module.build_part_audit("p1", [prediction], machine_status="NG", triggered_evidence=())

    branch = audit["views"]["front"][0]
    assert branch["evidence_path"] == str(missing)
    assert branch["evidence_exists"] is False


def test_atomic_write_failure_does_not_publish_final_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed atomic replace must not publish a partial final audit."""
    audit_module = _load_module("audit_atomic_module", "inspection_audit.py")
    output = tmp_path / "audit" / "p1.json"

    def fail_replace(_self: Path, _target: Path) -> Path:
        msg = "replace failed"
        raise OSError(msg)

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        audit_module.write_part_audit({"schema_version": "1.0"}, output)

    assert not output.exists()
    assert json.loads(output.with_suffix(".json.tmp").read_text(encoding="utf-8"))["schema_version"] == "1.0"
