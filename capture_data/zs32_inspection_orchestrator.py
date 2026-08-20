# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Template-first online orchestration for eight-view ZS32 inspection."""

from __future__ import annotations

import csv
import hashlib
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Protocol

from capture_data.fusion_engine import BRANCH_FIELDNAMES
from zs32_inspection.domain.views import CANONICAL_VIEWS

DOWNSTREAM_BRANCHES = ("PatchCore", "YOLO", "geometry")
TEMPLATE_BRANCH = "template_match"
_TEMPLATE_SKIPPED_PROFILES = {
    "zs32-right-20-commissioning",
    "zs32-right-22-commissioning",
    "zs32_right_eight_view_20_group_commissioning_v1",
    "zs32_right_eight_view_22_group_commissioning_v1",
}
_SECONDARY_VIEWS = {"front_secondary", "back_secondary"}
VALID_HANDS = {"right"}
_PASS_STATUSES = {"PASS"}
_REVIEW_STATUSES = {"REVIEW", "WARN", "WARNING", "SUSPECT", "GRAY", "ERROR"}
_NG_STATUSES = {"NG", "FAIL", "FAILED", "DEFECT", "STRONG", "NG_TEMPLATE"}
_DOWNSTREAM_STATUSES = {"OK", "REVIEW", "SUSPECT", "RETAKE", "INVALID_CAPTURE"}


def _strict_finite_number(value: object) -> float | None:
    """Return a finite number without accepting booleans or coercible strings."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if isfinite(number) else None


@dataclass(frozen=True)
class InspectionRequest:
    """Identify one physical ZS32 part and its complete eight-view capture.

    Args:
        part_id (str): Stable physical-part identity.
        capture_session (str): Acquisition-session identity.
        group_id (str): Front/back capture-group identity.
        hand (str): Product hand; strict eight-view inspection requires ``right``.
        images (Mapping[str, str | Path]): Image path for every canonical view.
    """

    part_id: str
    capture_session: str
    group_id: str
    hand: str
    images: Mapping[str, str | Path]


class TemplateGate(Protocol):
    """Structural protocol for an injected traditional template gate."""

    def evaluate(self, image_path: Path, hand: str, view: str) -> object:
        """Evaluate one canonical ZS32 view."""


DownstreamRunner = Callable[[InspectionRequest, tuple[dict[str, Any], ...]], object]


def _result_mapping(result: object) -> dict[str, Any]:
    """Convert an adapter result into a mutable evidence mapping."""
    if isinstance(result, Mapping):
        return dict(result)
    if is_dataclass(result) and not isinstance(result, type):
        return asdict(result)
    attributes = getattr(result, "__dict__", None)
    if isinstance(attributes, Mapping):
        return {key: value for key, value in attributes.items() if not key.startswith("_")}
    return {}


def _exception_mapping(error: Exception) -> dict[str, Any]:
    """Best-effort preservation of a structured adapter error."""
    converter = getattr(error, "to_result", None)
    if not callable(converter):
        return {}
    try:
        return _result_mapping(converter())
    except Exception:  # noqa: BLE001 - error serialization must not bypass fail-closed handling
        return {}


def _normalize_gate_result(  # noqa: C901
    result: object,
    *,
    image_path: Path,
    view: str,
) -> dict[str, Any]:
    """Normalize adapter-specific output without inventing a passing result."""
    evidence = _result_mapping(result)
    continuous_contract_valid = False
    risk_score = evidence.get("risk_score", evidence.get("raw_score", evidence.get("score")))
    raw_status = evidence.get("status")
    if raw_status in {None, ""}:
        prediction = str(evidence.get("prediction", "")).strip().upper()
        if prediction == "NORMAL":
            raw_status = "PASS"
        elif prediction == "DEFECT":
            raw_status = "NG_TEMPLATE"
    if raw_status in {None, ""}:
        status = "REVIEW"
        reason = "template gate result is missing status"
    else:
        normalized = str(raw_status).strip().upper()
        if normalized in _PASS_STATUSES:
            status = "PASS"
        elif normalized in _NG_STATUSES:
            status = "NG_TEMPLATE"
        elif normalized == "INVALID_CAPTURE":
            status = "INVALID_CAPTURE"
        elif normalized in _REVIEW_STATUSES:
            status = "REVIEW"
        else:
            status = "REVIEW"
        reason = evidence.get("reason")
        if status != "PASS" and not reason:
            reason = f"template gate returned {normalized or 'an empty status'}"
    if raw_status not in {None, ""} and status in {"PASS", "REVIEW", "NG_TEMPLATE"}:
        risk = _strict_finite_number(risk_score)
        similarity = _strict_finite_number(evidence.get("similarity"))
        low = _strict_finite_number(evidence.get("low_threshold"))
        high = _strict_finite_number(evidence.get("high_threshold"))
        if any(value is None for value in (risk, similarity, low, high)):
            status = "REVIEW"
            reason = "template gate result is missing continuous score or dual thresholds"
        else:
            if not 0 <= low <= high <= 2:
                status = "REVIEW"
                reason = "template gate result contains invalid continuous evidence"
            else:
                computed = "PASS" if risk < low else "REVIEW" if risk < high else "NG_TEMPLATE"
                if status != computed:
                    status = "REVIEW"
                    reason = (
                        f"template gate status disagrees with thresholds: reported={raw_status}, computed={computed}"
                    )
                else:
                    continuous_contract_valid = True
    if continuous_contract_valid:
        versions = evidence.get("versions")
        versions = versions if isinstance(versions, Mapping) else {}
        required_versions = (
            evidence.get("model_version", versions.get("model")),
            evidence.get("threshold_version", versions.get("threshold")),
            evidence.get("roi_version", versions.get("roi")),
            evidence.get("template_version", versions.get("template")),
        )
        template_path_text = str(evidence.get("best_template_path", "")).strip()
        declared_hash = str(evidence.get("best_template_sha256", "")).strip().lower()
        template_path = Path(template_path_text) if template_path_text else None
        if (
            any(not str(value or "").strip() for value in required_versions)
            or template_path is None
            or not template_path.is_file()
            or not re.fullmatch(r"[0-9a-f]{64}", declared_hash)
            or hashlib.sha256(template_path.read_bytes()).hexdigest() != declared_hash
        ):
            status = "REVIEW"
            reason = "template gate result is missing or mismatches deployment evidence"
    return {
        **evidence,
        "branch": TEMPLATE_BRANCH,
        "view": view,
        "image_path": str(image_path),
        "score": risk_score,
        "raw_score": risk_score,
        "status": status,
        "reason": reason,
    }


def _downstream_mapping(result: object) -> dict[str, Any]:
    """Normalize the downstream result while defaulting incomplete evidence closed."""
    values = _result_mapping(result)
    status_value = values.get("machine_status", values.get("final_status", values.get("status")))
    if status_value in {None, ""}:
        return {"called": True, "status": "ERROR", "inspection_complete": False}
    status_value = str(getattr(status_value, "value", status_value)).strip().upper()
    inspection_complete = values.get("inspection_complete") is True
    if status_value not in _DOWNSTREAM_STATUSES and not status_value.startswith("NG_"):
        return {"called": True, "status": "ERROR", "inspection_complete": False}
    if status_value == "OK" and not inspection_complete:
        return {"called": True, "status": "ERROR", "inspection_complete": False}
    return {
        "called": True,
        "status": status_value,
        "inspection_complete": inspection_complete,
    }


def template_results_to_branch_rows(
    request: InspectionRequest,
    template_results: tuple[dict[str, Any], ...],
    *,
    profile: str = "zs32_eight_view_v1",
) -> list[dict[str, Any]]:
    """Convert eight validated gate results into strict Stage-18 branch rows."""
    results_by_view = {str(result.get("view", "")): result for result in template_results}
    if set(results_by_view) != set(CANONICAL_VIEWS) or len(template_results) != len(CANONICAL_VIEWS):
        msg = "template results must contain exactly one row for every canonical view"
        raise ValueError(msg)
    rows: list[dict[str, Any]] = []
    for view in CANONICAL_VIEWS:
        result = results_by_view[view]
        status = str(result.get("status", "")).upper()
        if profile in _TEMPLATE_SKIPPED_PROFILES and view in _SECONDARY_VIEWS:
            if status != "SKIPPED":
                msg = f"22-group profile requires SKIPPED template result for {view}"
                raise ValueError(msg)
            continue
        if status not in {"PASS", "REVIEW", "NG_TEMPLATE"}:
            msg = f"unsupported template result status for {view}: {status or 'missing'}"
            raise ValueError(msg)
        image_path = Path(request.images[view])
        versions = result.get("versions")
        versions = versions if isinstance(versions, Mapping) else {}
        risk = result.get("risk_score", result.get("raw_score", result.get("score")))
        high = result.get("high_threshold", result.get("threshold"))
        evidence_level = {"PASS": "CLEAR", "REVIEW": "GRAY", "NG_TEMPLATE": "STRONG"}[status]
        rows.append(
            {
                "part_id": request.part_id,
                "side": "zs32",
                "view": view,
                "hand": request.hand,
                "product": "ZS32",
                "profile": profile,
                "capture_session": request.capture_session,
                "group_id": request.group_id,
                "slot_id": None,
                "branch": TEMPLATE_BRANCH,
                "pred_label": int(status == "NG_TEMPLATE"),
                "score": risk,
                "threshold": high,
                "defect_type": "template_mismatch" if status == "NG_TEMPLATE" else None,
                "evidence_type": "template_similarity",
                "gt_defect_type": None,
                "reason": result.get("reason"),
                "source_path": str(image_path),
                "evidence_path": result.get("best_template_path"),
                "status": status,
                "low_threshold": result.get("low_threshold"),
                "high_threshold": high,
                "evidence_level": evidence_level,
                "model_version": result.get("model_version", versions.get("model")),
                "threshold_version": result.get("threshold_version", versions.get("threshold")),
                "roi_version": result.get("roi_version", versions.get("roi")),
                "template_version": result.get("template_version", versions.get("template")),
                "source_hash": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                "evidence_hash": result.get("best_template_sha256"),
                "manifest_identity": f"{request.part_id}:{request.hand}:{view}",
                "detections": None,
            },
        )
    return rows


def write_template_match_csv(
    request: InspectionRequest,
    template_results: tuple[dict[str, Any], ...],
    output_csv: Path,
    *,
    profile: str = "zs32_eight_view_v1",
) -> Path:
    """Atomically publish strict template evidence for Stage 18."""
    rows = template_results_to_branch_rows(request, template_results, profile=profile)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=BRANCH_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_csv)
    return output_csv


def _validation_error(request: InspectionRequest) -> str | None:
    """Return the first identity or capture validation failure."""
    for field in ("part_id", "capture_session", "group_id"):
        if not isinstance(getattr(request, field), str) or not getattr(request, field).strip():
            return f"{field} must be a non-empty string"
    if request.hand not in VALID_HANDS:
        return f"hand must be one of {sorted(VALID_HANDS)}, found {request.hand!r}"
    if not isinstance(request.images, Mapping):
        return "images must be a view-to-path mapping"
    actual_views = set(request.images)
    missing_views = [view for view in CANONICAL_VIEWS if view not in actual_views]
    extra_views = sorted(str(view) for view in actual_views.difference(CANONICAL_VIEWS))
    if missing_views or extra_views:
        return f"invalid eight-view capture: missing={missing_views}, extra={extra_views}"
    path_error = None
    for view in CANONICAL_VIEWS:
        try:
            image_path = Path(request.images[view])
        except TypeError:
            path_error = f"invalid image path for {view}: {request.images[view]!r}"
            break
        if not image_path.is_file():
            path_error = f"missing image file for {view}: {image_path}"
            break
    return path_error


def _base_audit(request: InspectionRequest) -> dict[str, Any]:
    """Initialize a fail-closed execution audit."""
    return {
        "schema_version": "1.0",
        "part_id": request.part_id,
        "capture_session": request.capture_session,
        "group_id": request.group_id,
        "hand": request.hand,
        "machine_status": "REVIEW",
        "short_circuited": False,
        "stopped_after": None,
        "early_stop_reason": None,
        "evaluated_views": [],
        "skipped_views": list(CANONICAL_VIEWS),
        "skipped_branches": list(DOWNSTREAM_BRANCHES),
        "template_results": [],
        "inspection_complete": False,
        "review": {"status": "PENDING"},
        "review_status": None,
        "released_status": None,
        "downstream": {"called": False, "status": "SKIPPED", "inspection_complete": False},
    }


def _invoke_downstream(
    runner: DownstreamRunner,
    request: InspectionRequest,
    template_results: tuple[dict[str, Any], ...],
) -> object:
    """Call either a function runner or an object exposing ``run``."""
    if callable(runner):
        return runner(request, template_results)
    return runner.run(request, template_results)  # type: ignore[attr-defined]


class ZS32InspectionOrchestrator:
    """Run template matching as a fail-closed gate before expensive branches."""

    def __init__(self, template_gate: TemplateGate, downstream_runner: DownstreamRunner) -> None:
        """Initialize injected gate and downstream runner."""
        self.template_gate = template_gate
        self.downstream_runner = downstream_runner

    def run(self, request: InspectionRequest) -> dict[str, Any]:
        """Inspect one part, calling downstream exactly once only after eight passes.

        Args:
            request (InspectionRequest): Complete eight-view part request.

        Returns:
            dict[str, Any]: Structured execution audit including skipped work.
        """
        audit = _base_audit(request)
        if error := _validation_error(request):
            audit.update(
                machine_status="INVALID_CAPTURE",
                short_circuited=True,
                stopped_after="request_validation",
                early_stop_reason=error,
            )
            return audit

        for view in CANONICAL_VIEWS:
            image_path = Path(request.images[view])
            audit["evaluated_views"].append(view)
            audit["skipped_views"] = list(CANONICAL_VIEWS[len(audit["evaluated_views"]) :])
            try:
                raw_result = self.template_gate.evaluate(image_path, request.hand, view)
                evidence = _normalize_gate_result(raw_result, image_path=image_path, view=view)
            except Exception as error:  # noqa: BLE001 - hardware/model adapters have heterogeneous failures
                structured_error = _exception_mapping(error)
                evidence = {
                    **structured_error,
                    "branch": TEMPLATE_BRANCH,
                    "view": view,
                    "image_path": str(image_path),
                    "status": "REVIEW",
                    "reason": structured_error.get("reason")
                    or f"template gate exception: {type(error).__name__}: {error}",
                }
            audit["template_results"].append(evidence)

        blocked = [result for result in audit["template_results"] if result["status"] != "PASS"]
        if blocked:
            decisive = next((result for result in blocked if result["status"] == "NG_TEMPLATE"), blocked[0])
            audit.update(
                machine_status=decisive["status"],
                short_circuited=True,
                stopped_after=TEMPLATE_BRANCH,
                early_stop_reason=decisive["reason"],
            )
            return audit

        audit["skipped_branches"] = []
        try:
            downstream_result = _invoke_downstream(
                self.downstream_runner,
                request,
                tuple(audit["template_results"]),
            )
        except Exception as error:  # noqa: BLE001 - downstream inference stacks have heterogeneous failures
            audit.update(
                machine_status="REVIEW",
                stopped_after="downstream",
                early_stop_reason=f"downstream exception: {type(error).__name__}: {error}",
                downstream={"called": True, "status": "ERROR", "inspection_complete": False},
                inspection_complete=False,
            )
            return audit
        downstream = _downstream_mapping(downstream_result)
        audit.update(
            machine_status=downstream["status"] if downstream["status"] != "ERROR" else "REVIEW",
            stopped_after="downstream",
            downstream=downstream,
            inspection_complete=downstream["inspection_complete"],
        )
        if downstream["status"] == "ERROR":
            audit["early_stop_reason"] = "downstream result is missing machine status"
        return audit


def run_zs32_inspection(
    request: InspectionRequest,
    *,
    template_gate: TemplateGate,
    downstream_runner: DownstreamRunner,
) -> dict[str, Any]:
    """Run one injected template-first inspection without retaining state."""
    return ZS32InspectionOrchestrator(template_gate, downstream_runner).run(request)
