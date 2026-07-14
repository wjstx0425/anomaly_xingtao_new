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


def _make_view(tmp_path: Path, view: str, *, model_supported: bool | None = None) -> ViewResult:
    return ViewResult(
        view=view,
        source_path=tmp_path / f"{view}.png",
        source_sha256="a" * 64,
        source_shape=(1080, 1440),
        model_supported=view in MODELED_VIEWS if model_supported is None else model_supported,
        branches={},
        capture={},
    )


@pytest.mark.parametrize("view", ["left", "", "FRONT"])
def test_view_result_rejects_unknown_view(tmp_path: Path, view: str) -> None:
    with pytest.raises(ValueError, match="view"):
        _make_view(tmp_path, view)


@pytest.mark.parametrize(
    ("view", "model_supported"),
    [("front", False), ("back_right", False), ("front_secondary", True), ("back_secondary", True)],
)
def test_view_result_requires_support_to_match_modeled_views(
    tmp_path: Path,
    view: str,
    model_supported: bool,
) -> None:
    with pytest.raises(ValueError, match="model_supported"):
        _make_view(tmp_path, view, model_supported=model_supported)


@pytest.mark.parametrize(
    "views",
    [
        VIEW_ORDER[:-1],
        (*VIEW_ORDER[:-1], "back_right"),
        (VIEW_ORDER[1], VIEW_ORDER[0], *VIEW_ORDER[2:]),
    ],
    ids=["missing", "duplicate", "wrong-order"],
)
def test_inspection_result_requires_exactly_eight_ordered_views(tmp_path: Path, views: tuple[str, ...]) -> None:
    identity = InspectionIdentity("part-1", "session-1", "group-1", "right")

    with pytest.raises(ValueError, match="VIEW_ORDER"):
        InspectionResult(
            identity=identity,
            views=tuple(_make_view(tmp_path, view) for view in views),
            machine_status="REVIEW",
            reason="test",
        )


def test_inspection_result_rejects_unknown_mode(tmp_path: Path) -> None:
    identity = InspectionIdentity("part-1", "session-1", "group-1", "right")
    views = tuple(_make_view(tmp_path, view) for view in VIEW_ORDER)

    with pytest.raises(ValueError, match="mode"):
        InspectionResult(
            identity=identity,
            views=views,
            machine_status="REVIEW",
            reason="test",
            mode="batch",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("part_id", "capture_session", "group_id", "hand"),
    [
        ("part-1", "session-1", "group-1", "left"),
        ("", "session-1", "group-1", "right"),
        ("part-1", "", "group-1", "right"),
        ("part-1", "session-1", "", "right"),
    ],
)
def test_inspection_identity_rejects_invalid_runtime_values(
    part_id: str,
    capture_session: str,
    group_id: str,
    hand: str,
) -> None:
    with pytest.raises(ValueError):
        InspectionIdentity(part_id, capture_session, group_id, hand)  # type: ignore[arg-type]
