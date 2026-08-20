# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Deterministic, fail-closed fusion for BMW laboratory inspections."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from bmw_inspection.lab.contracts import (
    BranchEvidence,
    BranchName,
    BranchStatus,
    CaptureSet,
    FinalStatus,
    InspectionResult,
    ViewId,
)


_BRANCH_ORDER = tuple(BranchName)
_NG_STATUS = {
    BranchName.TEMPLATE: FinalStatus.NG_TEMPLATE,
    BranchName.BRIGHT_STREAK: FinalStatus.NG_BRIGHT_STREAK,
    BranchName.YOLO: FinalStatus.NG_YOLO,
    BranchName.PATCHCORE: FinalStatus.NG_ANOMALY,
}
_COMPLETE_STATUSES = frozenset({BranchStatus.PASS, BranchStatus.NG})


@dataclass(frozen=True, slots=True)
class FusionDecision:
    """Immutable fused result plus every branch that produced NG evidence."""

    result: InspectionResult
    triggered_branches: tuple[BranchName, ...]


def fuse_inspection(
    capture_set: CaptureSet,
    evidence: Sequence[BranchEvidence],
    *,
    expected_required: Mapping[BranchName, frozenset[ViewId]],
    retake_reasons: Sequence[str] = (),
) -> FusionDecision:
    """Fuse branch rows with explicit precedence and required-evidence checks."""
    rows = tuple(evidence)
    identities: dict[tuple[BranchName, ViewId], BranchEvidence] = {}
    for row in rows:
        identity = (row.branch, row.view_id)
        if identity in identities:
            raise ValueError(f"duplicate branch evidence: {row.branch.value}/{row.view_id.value}")
        identities[identity] = row

    expected = {branch: frozenset(views) for branch, views in expected_required.items()}
    declared_rows: dict[BranchName, set[ViewId]] = {}
    for row in rows:
        if row.required_for_ok:
            declared_rows.setdefault(row.branch, set()).add(row.view_id)

    complete = True
    for branch in set(expected) | set(declared_rows):
        views = expected.get(branch, frozenset(declared_rows.get(branch, set())))
        if branch in expected and not views:
            complete = False
        for view_id in views:
            row = identities.get((branch, view_id))
            if row is None or not row.required_for_ok or row.status not in _COMPLETE_STATUSES:
                complete = False

    triggered = tuple(
        branch
        for branch in _BRANCH_ORDER
        if any(row.branch is branch and row.status is BranchStatus.NG for row in rows)
    )

    retake = tuple(reason.strip() for reason in retake_reasons if reason.strip())
    if retake:
        status = FinalStatus.RETAKE
        reason = "; ".join(retake)
    elif any(row.status is BranchStatus.ERROR for row in rows):
        status = FinalStatus.ERROR
        reason = _join_reasons(rows, BranchStatus.ERROR)
    elif BranchName.TEMPLATE in triggered:
        status = FinalStatus.NG_TEMPLATE
        reason = _trigger_reason(triggered, rows)
    elif triggered:
        status = _NG_STATUS[triggered[0]]
        reason = _trigger_reason(triggered, rows)
    elif any(row.status is BranchStatus.REVIEW for row in rows):
        status = FinalStatus.REVIEW
        reason = _join_reasons(rows, BranchStatus.REVIEW)
    elif not complete:
        status = FinalStatus.REVIEW
        reason = "required evidence is missing, disabled, skipped, or incomplete"
    else:
        status = FinalStatus.OK
        reason = "all required inspection evidence passed"

    result = InspectionResult(
        capture_set=capture_set,
        evidence=rows,
        final_status=status,
        reason=reason,
        required_complete=complete,
    )
    return FusionDecision(result=result, triggered_branches=triggered)


def _join_reasons(rows: tuple[BranchEvidence, ...], status: BranchStatus) -> str:
    reasons = tuple(row.reason for row in rows if row.status is status)
    return "; ".join(reasons) or status.value


def _trigger_reason(
    triggered: tuple[BranchName, ...],
    rows: tuple[BranchEvidence, ...],
) -> str:
    details = []
    for branch in triggered:
        reasons = [row.reason for row in rows if row.branch is branch and row.status is BranchStatus.NG]
        details.append(f"{branch.value}: {', '.join(reasons)}")
    return "; ".join(details)
