# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Truth-table tests for BMW laboratory evidence fusion."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from bmw_inspection.lab.contracts import (
    BranchEvidence,
    BranchName,
    BranchStatus,
    CaptureSet,
    CapturedView,
    FinalStatus,
    ViewId,
)
from bmw_inspection.lab.fusion import fuse_inspection


def _capture_set() -> CaptureSet:
    now = datetime.now()
    return CaptureSet(
        capture_set_id="capture-fusion",
        views={
            view_id: CapturedView(view_id, f"serial-{view_id.value}", np.full((4, 6), 20, np.uint8), now)
            for view_id in ViewId
        },
        created_at=now,
    )


def _row(
    branch: BranchName,
    view_id: ViewId,
    status: BranchStatus = BranchStatus.PASS,
    *,
    required: bool = True,
) -> BranchEvidence:
    return BranchEvidence(
        branch=branch,
        view_id=view_id,
        status=status,
        required_for_ok=required,
        score=0.1,
        threshold=0.5,
        elapsed_ms=1.0,
        reason=f"{branch.value}/{view_id.value}/{status.value}",
        model_id=f"{branch.value}-model",
        artifact_paths={},
    )


def _all_pass() -> tuple[BranchEvidence, ...]:
    return (
        *(_row(BranchName.TEMPLATE, view_id) for view_id in ViewId),
        _row(BranchName.BRIGHT_STREAK, ViewId.FRONT_LEFT),
        *(_row(BranchName.YOLO, view_id) for view_id in ViewId),
        *(_row(BranchName.PATCHCORE, view_id) for view_id in ViewId),
    )


def test_ok_requires_every_declared_required_row_to_be_present_and_pass() -> None:
    decision = fuse_inspection(
        _capture_set(),
        _all_pass(),
        expected_required={
            BranchName.TEMPLATE: frozenset(ViewId),
            BranchName.BRIGHT_STREAK: frozenset({ViewId.FRONT_LEFT}),
            BranchName.YOLO: frozenset(ViewId),
            BranchName.PATCHCORE: frozenset(ViewId),
        },
    )

    assert decision.result.final_status is FinalStatus.OK
    assert decision.result.required_complete is True
    assert decision.triggered_branches == ()


def test_missing_or_disabled_required_branch_never_becomes_ok() -> None:
    rows = tuple(row for row in _all_pass() if not (row.branch is BranchName.YOLO and row.view_id is ViewId.BACK))
    missing = fuse_inspection(
        _capture_set(),
        rows,
        expected_required={BranchName.YOLO: frozenset(ViewId)},
    )
    disabled_rows = (
        *(_row(BranchName.TEMPLATE, view_id) for view_id in ViewId),
        _row(BranchName.YOLO, ViewId.FRONT, BranchStatus.SKIPPED),
    )
    disabled = fuse_inspection(
        _capture_set(),
        disabled_rows,
        expected_required={
            BranchName.TEMPLATE: frozenset(ViewId),
            BranchName.YOLO: frozenset({ViewId.FRONT}),
        },
    )

    assert missing.result.final_status is FinalStatus.REVIEW
    assert missing.result.required_complete is False
    assert disabled.result.final_status is FinalStatus.REVIEW


def test_multiple_downstream_ng_branches_are_all_recorded() -> None:
    rows = list(_all_pass())
    rows = [
        _row(row.branch, row.view_id, BranchStatus.NG)
        if (row.branch, row.view_id)
        in {
            (BranchName.BRIGHT_STREAK, ViewId.FRONT_LEFT),
            (BranchName.YOLO, ViewId.BACK),
            (BranchName.PATCHCORE, ViewId.BACK_RIGHT),
        }
        else row
        for row in rows
    ]

    decision = fuse_inspection(_capture_set(), rows, expected_required={})

    assert decision.result.final_status is FinalStatus.NG_BRIGHT_STREAK
    assert decision.triggered_branches == (
        BranchName.BRIGHT_STREAK,
        BranchName.YOLO,
        BranchName.PATCHCORE,
    )
    assert all(branch.value in decision.result.reason for branch in decision.triggered_branches)


def test_confirmed_downstream_ng_precedes_review_but_not_error() -> None:
    decision = fuse_inspection(
        _capture_set(),
        (
            _row(BranchName.YOLO, ViewId.FRONT, BranchStatus.NG),
            _row(BranchName.PATCHCORE, ViewId.BACK, BranchStatus.REVIEW),
        ),
        expected_required={},
    )

    assert decision.result.final_status is FinalStatus.NG_YOLO
    assert decision.triggered_branches == (BranchName.YOLO,)
    assert "yolo" in decision.result.reason


def test_retake_precedes_model_error_and_product_ng() -> None:
    rows = (
        _row(BranchName.TEMPLATE, ViewId.FRONT, BranchStatus.ERROR),
        _row(BranchName.YOLO, ViewId.BACK, BranchStatus.NG),
    )

    decision = fuse_inspection(
        _capture_set(),
        rows,
        expected_required={},
        retake_reasons=("front is blurred",),
    )

    assert decision.result.final_status is FinalStatus.RETAKE
    assert "blurred" in decision.result.reason


def test_error_precedes_template_or_downstream_ng_and_review_is_not_ok() -> None:
    error = fuse_inspection(
        _capture_set(),
        (
            _row(BranchName.TEMPLATE, ViewId.FRONT, BranchStatus.NG),
            _row(BranchName.PATCHCORE, ViewId.BACK, BranchStatus.ERROR),
        ),
        expected_required={},
    )
    review = fuse_inspection(
        _capture_set(),
        (_row(BranchName.TEMPLATE, ViewId.FRONT, BranchStatus.REVIEW),),
        expected_required={BranchName.TEMPLATE: frozenset({ViewId.FRONT})},
    )

    assert error.result.final_status is FinalStatus.ERROR
    assert review.result.final_status is FinalStatus.REVIEW
    with pytest.raises(AttributeError):
        review.triggered_branches += (BranchName.YOLO,)  # type: ignore[misc]
