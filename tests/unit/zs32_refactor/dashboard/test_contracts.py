from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from zs32_inspection.dashboard.contracts import (
    MODELED_VIEWS,
    VIEW_ORDER,
    BranchEvidence,
    BranchState,
    EvidenceLayer,
    InspectionIdentity,
    InspectionResult,
    ViewResult,
)


def test_view_order_and_modeled_views_are_closed_sets() -> None:
    assert VIEW_ORDER == (
        "front",
        "front_left",
        "front_right",
        "front_secondary",
        "back",
        "back_left",
        "back_right",
        "back_secondary",
    )
    assert MODELED_VIEWS == (
        "front",
        "front_left",
        "front_right",
        "back",
        "back_left",
        "back_right",
    )


def test_evidence_enums_publish_stable_wire_values() -> None:
    assert tuple(EvidenceLayer) == (
        EvidenceLayer.FUSION,
        EvidenceLayer.ORIGINAL,
        EvidenceLayer.PATCHCORE,
        EvidenceLayer.YOLO,
        EvidenceLayer.TEMPLATE,
    )
    assert tuple(item.value for item in BranchState) == (
        "available",
        "skipped",
        "unsupported",
        "error",
    )


def test_inspection_contract_preserves_eight_ordered_views_and_optional_scores(
    tmp_path: Path,
) -> None:
    branch = BranchEvidence(
        branch="patchcore",
        state=BranchState.SKIPPED,
        status="SKIPPED",
        score=None,
        reason="model was not run",
    )
    views = tuple(
        ViewResult(
            view=view,
            source_path=tmp_path / f"{view}.png",
            source_sha256="a" * 64,
            source_shape=(1080, 1440),
            model_supported=view in MODELED_VIEWS,
            branches={"patchcore": branch},
            capture={"camera": "camera-1"},
        )
        for view in VIEW_ORDER
    )
    result = InspectionResult(
        identity=InspectionIdentity(
            part_id="part-1",
            capture_session="session-1",
            group_id="group-1",
            hand="right",
        ),
        views=views,
        machine_status="REVIEW",
        reason="model branch skipped",
    )

    assert tuple(view.view for view in result.views) == VIEW_ORDER
    assert result.views[0].branches["patchcore"].score is None
    assert result.mode == "offline"
    with pytest.raises(FrozenInstanceError):
        result.reason = "changed"
