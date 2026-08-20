# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the template-first ZS32 inspection orchestrator."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from capture_data.zs32_inspection_orchestrator import (
    InspectionRequest,
    ZS32InspectionOrchestrator,
    template_results_to_branch_rows,
)

if TYPE_CHECKING:
    from pathlib import Path


VIEWS = (
    "front",
    "front_left",
    "front_right",
    "front_secondary",
    "back",
    "back_left",
    "back_right",
    "back_secondary",
)


def _request(tmp_path: Path, **overrides: object) -> InspectionRequest:
    images = {}
    for view in VIEWS:
        path = tmp_path / f"{view}.png"
        path.write_bytes(b"image")
        images[view] = path
    values = {
        "part_id": "part-001",
        "capture_session": "session-001",
        "group_id": "group-001",
        "hand": "right",
        "images": images,
    }
    values.update(overrides)
    return InspectionRequest(**values)


class SequenceGate:
    """Return a configured status by canonical view and record calls."""

    def __init__(self, statuses: dict[str, str] | None = None) -> None:
        self.statuses = statuses or {}
        self.calls: list[tuple[Path, str, str]] = []

    def evaluate(self, image_path: Path, hand: str, view: str) -> dict[str, object]:
        """Return one configured template result."""
        self.calls.append((image_path, hand, view))
        status = self.statuses.get(view, "PASS")
        return {
            "status": status,
            "reason": None if status == "PASS" else f"{view} template {status.lower()}",
            "score": 0.02 if status == "PASS" else 0.58,
            "risk_score": 0.02 if status == "PASS" else 0.58,
            "threshold": 0.50,
            "low_threshold": 0.10,
            "high_threshold": 0.50,
            "similarity": 0.98 if status == "PASS" else 0.42,
            "best_template_path": str(image_path),
            "best_template_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "versions": {
                "model": "zs32-models-2026.07.13",
                "threshold": "zs32-thresholds-2026.07.13",
                "roi": "zs32-roi-2026.07.12",
                "template": "zs32-templates-2026.07.13",
            },
        }


def test_all_template_views_pass_before_downstream_runs_once(tmp_path: Path) -> None:
    """Downstream runs once and only after all eight canonical views pass."""
    gate = SequenceGate()
    downstream = Mock(return_value={"machine_status": "OK", "inspection_complete": True})
    request = _request(tmp_path)

    audit = ZS32InspectionOrchestrator(gate, downstream).run(request)

    assert [call[2] for call in gate.calls] == list(VIEWS)
    downstream.assert_called_once()
    called_request, template_results = downstream.call_args.args
    assert called_request == request
    assert len(template_results) == len(VIEWS)
    assert all(result["status"] == "PASS" for result in template_results)
    assert audit["machine_status"] == "OK"
    assert audit["stopped_after"] == "downstream"
    assert audit["early_stop_reason"] is None
    assert audit["evaluated_views"] == list(VIEWS)
    assert audit["skipped_views"] == []
    assert audit["skipped_branches"] == []
    assert audit["inspection_complete"] is True
    assert audit["short_circuited"] is False
    assert audit["downstream"] == {"called": True, "status": "OK", "inspection_complete": True}
    assert audit["review"] == {"status": "PENDING"}
    assert audit["review_status"] is None
    assert audit["released_status"] is None
    assert audit["template_results"][0]["branch"] == "template_match"
    assert audit["template_results"][0]["raw_score"] == pytest.approx(0.02)
    assert audit["template_results"][0]["similarity"] == pytest.approx(0.98)

    rows = template_results_to_branch_rows(request, tuple(audit["template_results"]))
    assert [row["view"] for row in rows] == list(VIEWS)
    assert all(row["branch"] == "template_match" for row in rows)
    assert all(row["evidence_level"] == "CLEAR" for row in rows)
    assert rows[0]["score"] == pytest.approx(0.02)
    assert rows[0]["model_version"] == "zs32-models-2026.07.13"
    assert len(rows[0]["source_hash"]) == 64
    assert rows[0]["evidence_hash"] == audit["template_results"][0]["best_template_sha256"]



@pytest.mark.parametrize(
    "profile",
    ["zs32-right-22-commissioning", "zs32_right_eight_view_22_group_commissioning_v1"],
)
def test_22_group_profile_omits_skipped_secondary_template_rows(tmp_path: Path, profile: str) -> None:
    request = _request(tmp_path)
    audit = ZS32InspectionOrchestrator(SequenceGate(), Mock(return_value={"machine_status": "OK", "inspection_complete": True})).run(request)
    results = list(audit["template_results"])
    for result in results:
        if result["view"] in {"front_secondary", "back_secondary"}:
            result["status"] = "SKIPPED"
    rows = template_results_to_branch_rows(request, tuple(results), profile=profile)
    assert [row["view"] for row in rows] == [view for view in VIEWS if "secondary" not in view]

@pytest.mark.parametrize("invalid_score", ["0.02", True, float("nan"), float("inf")])
def test_non_strict_template_score_stops_orchestrator_downstream(
    tmp_path: Path,
    invalid_score: object,
) -> None:
    """The orchestrator must reject coerced and non-finite continuous evidence."""
    gate = SequenceGate()
    original_evaluate = gate.evaluate

    def evaluate(image_path: Path, hand: str, view: str) -> dict[str, object]:
        result = original_evaluate(image_path, hand, view)
        if view == "front":
            result["risk_score"] = invalid_score
        return result

    gate.evaluate = evaluate  # type: ignore[method-assign]
    downstream = Mock()

    audit = ZS32InspectionOrchestrator(gate, downstream).run(_request(tmp_path))

    downstream.assert_not_called()
    assert audit["machine_status"] == "REVIEW"
    assert audit["inspection_complete"] is False
    assert audit["evaluated_views"] == list(VIEWS)
    assert audit["template_results"][0]["status"] == "REVIEW"


@pytest.mark.parametrize(
    ("gate_status", "machine_status"),
    [("REVIEW", "REVIEW"), ("NG_TEMPLATE", "NG_TEMPLATE"), ("FAIL", "NG_TEMPLATE")],
)
def test_non_pass_template_stops_all_downstream_work(
    tmp_path: Path,
    gate_status: str,
    machine_status: str,
) -> None:
    """A non-pass result still evaluates every template view before skipping expensive branches."""
    gate = SequenceGate({"front_left": gate_status})
    downstream = Mock()

    audit = ZS32InspectionOrchestrator(gate, downstream).run(_request(tmp_path))

    downstream.assert_not_called()
    assert [call[2] for call in gate.calls] == list(VIEWS)
    assert audit["machine_status"] == machine_status
    assert audit["stopped_after"] == "template_match"
    assert audit["evaluated_views"] == list(VIEWS)
    assert audit["skipped_views"] == []
    assert len(audit["template_results"]) == len(VIEWS)
    assert audit["skipped_branches"] == ["PatchCore", "YOLO", "geometry"]
    assert audit["inspection_complete"] is False
    assert audit["short_circuited"] is True
    assert audit["downstream"] == {"called": False, "status": "SKIPPED", "inspection_complete": False}
    assert audit["review"] == {"status": "PENDING"}
    assert audit["review_status"] is None
    assert audit["released_status"] is None
    assert all(result["status"] != "CLEAR" for result in audit["template_results"])


def test_gate_exception_becomes_review_and_stops_downstream(tmp_path: Path) -> None:
    """Adapter failures fail closed to manual review."""
    gate = SequenceGate()
    gate.evaluate = Mock(side_effect=RuntimeError("template asset missing"))
    downstream = Mock()

    audit = ZS32InspectionOrchestrator(gate, downstream).run(_request(tmp_path))

    downstream.assert_not_called()
    assert audit["machine_status"] == "REVIEW"
    assert audit["evaluated_views"] == list(VIEWS)
    assert audit["skipped_views"] == []
    assert len(audit["template_results"]) == len(VIEWS)
    assert "template asset missing" in audit["early_stop_reason"]
    assert audit["template_results"][0]["status"] == "REVIEW"


@pytest.mark.parametrize(
    "overrides",
    [
        {"part_id": ""},
        {"capture_session": ""},
        {"group_id": ""},
        {"hand": "left"},
        {"hand": "unknown"},
    ],
)
def test_invalid_identity_stops_before_template_and_downstream(tmp_path: Path, overrides: dict[str, object]) -> None:
    """Invalid physical-part identity stops before any inference call."""
    gate = SequenceGate()
    downstream = Mock()

    audit = ZS32InspectionOrchestrator(gate, downstream).run(_request(tmp_path, **overrides))

    assert gate.calls == []
    downstream.assert_not_called()
    assert audit["machine_status"] == "INVALID_CAPTURE"
    assert audit["stopped_after"] == "request_validation"
    assert audit["evaluated_views"] == []
    assert audit["skipped_views"] == list(VIEWS)
    assert audit["inspection_complete"] is False
    assert audit["short_circuited"] is True


def test_missing_image_is_invalid_capture_without_running_any_gate(tmp_path: Path) -> None:
    """A missing canonical view invalidates the complete capture."""
    request = _request(tmp_path)
    images = dict(request.images)
    images.pop("back_right")
    gate = SequenceGate()
    downstream = Mock()

    audit = ZS32InspectionOrchestrator(gate, downstream).run(_request(tmp_path, images=images))

    assert gate.calls == []
    downstream.assert_not_called()
    assert audit["machine_status"] == "INVALID_CAPTURE"
    assert "back_right" in audit["early_stop_reason"]


def test_missing_image_file_is_invalid_capture_without_running_any_gate(tmp_path: Path) -> None:
    """A nonexistent image path invalidates the complete capture."""
    request = _request(tmp_path)
    missing = tmp_path / "missing.png"
    images = {**request.images, "back": missing}
    gate = SequenceGate()
    downstream = Mock()

    audit = ZS32InspectionOrchestrator(gate, downstream).run(_request(tmp_path, images=images))

    assert gate.calls == []
    downstream.assert_not_called()
    assert audit["machine_status"] == "INVALID_CAPTURE"
    assert str(missing) in audit["early_stop_reason"]


def test_unknown_or_malformed_gate_result_fails_closed_to_review(tmp_path: Path) -> None:
    """An adapter result cannot pass unless its pass status is explicit."""
    gate = SequenceGate()
    gate.evaluate = Mock(return_value={"score": 0.99})
    downstream = Mock()

    audit = ZS32InspectionOrchestrator(gate, downstream).run(_request(tmp_path))

    downstream.assert_not_called()
    assert audit["machine_status"] == "REVIEW"
    assert audit["template_results"][0]["status"] == "REVIEW"
    assert "missing status" in audit["early_stop_reason"]


def test_pass_without_continuous_threshold_evidence_still_stops_downstream(tmp_path: Path) -> None:
    """A reported PASS cannot bypass the required score and dual-threshold evidence."""
    gate = SequenceGate()
    gate.evaluate = Mock(return_value={"status": "PASS", "score": 0.01})
    downstream = Mock()

    audit = ZS32InspectionOrchestrator(gate, downstream).run(_request(tmp_path))

    downstream.assert_not_called()
    assert audit["machine_status"] == "REVIEW"
    assert audit["short_circuited"] is True
    assert "continuous score or dual thresholds" in audit["early_stop_reason"]


@pytest.mark.parametrize("missing", ["versions", "best_template_sha256", "best_template_path"])
def test_pass_without_complete_deployment_evidence_stops_downstream(tmp_path: Path, missing: str) -> None:
    """A numeric PASS cannot start downstream without versions and verified template identity."""
    gate = SequenceGate()
    result = gate.evaluate(next(iter(_request(tmp_path).images.values())), "right", "front")
    result.pop(missing)
    gate.evaluate = Mock(return_value=result)
    downstream = Mock()

    audit = ZS32InspectionOrchestrator(gate, downstream).run(_request(tmp_path))

    downstream.assert_not_called()
    assert audit["machine_status"] == "REVIEW"
    assert "deployment evidence" in audit["early_stop_reason"]


@pytest.mark.parametrize(
    "downstream_result",
    [
        {"machine_status": "OK"},
        {"machine_status": "OK", "inspection_complete": False},
        {"machine_status": "NOT_A_REAL_STATUS", "inspection_complete": True},
    ],
)
def test_incomplete_or_unknown_downstream_result_cannot_release_ok(
    tmp_path: Path,
    downstream_result: dict[str, object],
) -> None:
    """Only a completed, known downstream result may become the machine status."""
    audit = ZS32InspectionOrchestrator(SequenceGate(), Mock(return_value=downstream_result)).run(_request(tmp_path))

    assert audit["machine_status"] == "REVIEW"
    assert audit["inspection_complete"] is False
    assert audit["downstream"]["status"] == "ERROR"
    assert audit["early_stop_reason"]


def test_downstream_exception_becomes_review_without_claiming_completion(tmp_path: Path) -> None:
    """A downstream failure retains template evidence but does not release the part."""
    gate = SequenceGate()
    downstream = Mock(side_effect=RuntimeError("fusion service unavailable"))

    audit = ZS32InspectionOrchestrator(gate, downstream).run(_request(tmp_path))

    downstream.assert_called_once()
    assert audit["machine_status"] == "REVIEW"
    assert audit["stopped_after"] == "downstream"
    assert audit["inspection_complete"] is False
    assert audit["downstream"]["called"] is True
    assert audit["downstream"]["status"] == "ERROR"
    assert "fusion service unavailable" in audit["early_stop_reason"]
