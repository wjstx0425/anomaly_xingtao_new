"""Contracts for the isolated BMW six-view laboratory workflow."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from bmw_inspection.lab.contracts import (
    BranchEvidence,
    BranchName,
    BranchStatus,
    CapturedView,
    CaptureSet,
    FinalStatus,
    ViewId,
    validate_final_status,
)


def test_required_views_are_exactly_six() -> None:
    assert tuple(ViewId) == (
        ViewId.FRONT,
        ViewId.FRONT_LEFT,
        ViewId.FRONT_RIGHT,
        ViewId.BACK,
        ViewId.BACK_LEFT,
        ViewId.BACK_RIGHT,
    )


def test_public_status_values_are_stable() -> None:
    assert tuple(BranchName) == (
        BranchName.TEMPLATE,
        BranchName.BRIGHT_STREAK,
        BranchName.YOLO,
        BranchName.PATCHCORE,
    )
    assert tuple(BranchStatus) == (
        BranchStatus.PASS,
        BranchStatus.NG,
        BranchStatus.SKIPPED,
        BranchStatus.REVIEW,
        BranchStatus.ERROR,
    )
    assert tuple(FinalStatus) == (
        FinalStatus.OK,
        FinalStatus.NG_TEMPLATE,
        FinalStatus.NG_BRIGHT_STREAK,
        FinalStatus.NG_YOLO,
        FinalStatus.NG_ANOMALY,
        FinalStatus.REVIEW,
        FinalStatus.RETAKE,
        FinalStatus.ERROR,
    )


def test_string_enums_serialize_as_strings_without_python_311_strenum() -> None:
    assert json.dumps({"view_id": ViewId.FRONT, "status": FinalStatus.OK}) == (
        '{"view_id": "front", "status": "OK"}'
    )

    contracts_path = Path(__file__).parents[4] / "src/bmw_inspection/lab/contracts.py"
    script = f"""
import enum
import importlib.util
import sys

if hasattr(enum, "StrEnum"):
    delattr(enum, "StrEnum")
spec = importlib.util.spec_from_file_location("bmw_contracts_py310", {str(contracts_path)!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert issubclass(module.ViewId, str)
assert module.ViewId.FRONT.value == "front"
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_disabled_required_branch_cannot_publish_ok() -> None:
    with pytest.raises(ValueError, match="required branch"):
        validate_final_status(FinalStatus.OK, required_complete=False)


def test_capture_set_owns_read_only_copies_of_captured_images() -> None:
    captured_at = datetime.now(timezone.utc)
    sources = {view_id: np.full((4, 6, 3), index, dtype=np.uint8) for index, view_id in enumerate(ViewId)}
    capture_set = CaptureSet(
        capture_set_id="capture-1",
        views={
            view_id: CapturedView(
                view_id=view_id,
                camera_serial=f"camera-{index}",
                image=sources[view_id],
                captured_at=captured_at,
            )
            for index, view_id in enumerate(ViewId)
        },
        created_at=captured_at,
    )
    stored = capture_set.views[ViewId.FRONT].image
    sources[ViewId.FRONT][0, 0, 0] = 99

    assert int(stored[0, 0, 0]) == 0
    assert not np.shares_memory(stored, sources[ViewId.FRONT])
    assert stored.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        stored[0, 0, 0] = 77


def test_branch_evidence_is_immutable_and_rejects_nonfinite_score() -> None:
    evidence = BranchEvidence(
        branch=BranchName.YOLO,
        view_id=ViewId.FRONT,
        status=BranchStatus.PASS,
        required_for_ok=True,
        score=0.2,
        threshold=0.5,
        elapsed_ms=3.0,
        reason="no candidate reached the final threshold",
        model_id="yolo-v1",
        artifact_paths={"overlay": "front.png"},
    )

    with pytest.raises(FrozenInstanceError):
        evidence.reason = "changed"  # type: ignore[misc]
    assert evidence.required_for_ok is True
    with pytest.raises(ValueError, match="score must be finite"):
        BranchEvidence(
            branch=BranchName.YOLO,
            view_id=ViewId.FRONT,
            status=BranchStatus.PASS,
            required_for_ok=False,
            score=float("nan"),
            threshold=0.5,
            elapsed_ms=3.0,
            reason="invalid",
            model_id=None,
            artifact_paths={},
        )


def test_branch_evidence_rejects_nonboolean_required_for_ok() -> None:
    with pytest.raises(TypeError, match="required_for_ok must be bool"):
        BranchEvidence(
            branch=BranchName.YOLO,
            view_id=ViewId.FRONT,
            status=BranchStatus.PASS,
            required_for_ok=1,  # type: ignore[arg-type]
            score=0.2,
            threshold=0.5,
            elapsed_ms=3.0,
            reason="invalid required flag",
            model_id=None,
            artifact_paths={},
        )
