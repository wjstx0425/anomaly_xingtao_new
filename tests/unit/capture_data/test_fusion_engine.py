# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for robust industrial inspection fusion helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


def load_fusion_module() -> ModuleType:
    """Load the fusion helper from its file path."""
    script_path = Path(__file__).resolve().parents[3] / "capture_data" / "fusion_engine.py"
    spec = importlib.util.spec_from_file_location("capture_data_fusion_engine", script_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load fusion engine script from {script_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_capture_data_module(name: str, filename: str) -> ModuleType:
    """Load another capture-data helper from its file path."""
    script_path = Path(__file__).resolve().parents[3] / "capture_data" / filename
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load capture-data helper from {script_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ZS32_VIEWS = ("front", "front_left", "front_right", "back", "back_left", "back_right")


def _zs32_config() -> dict[str, object]:
    """Return the strict six-view fusion shape used by production ZS32."""
    return {
        "ok_requires": {"required_view_keys": [f"zs32:{view}" for view in ZS32_VIEWS]},
        "required_branches_by_view": {
            view: ["quality_gate", "registration", f"anomaly_{view}", "yolo", "geometry"] for view in ZS32_VIEWS
        },
        "branch_order": ["geometry", "yolo", *(f"anomaly_{view}" for view in ZS32_VIEWS)],
        "rules": {
            "geometry": {"status_on_positive": "NG_GEOMETRY"},
            "yolo": {"status_on_positive": "NG_YOLO"},
            **{f"anomaly_{view}": {"status_on_positive": "NG_ANOMALY"} for view in ZS32_VIEWS},
        },
    }


def _zs32_prediction(
    fusion: ModuleType,
    *,
    part_id: str,
    view: str,
    branch: str,
    score: float | None = None,
    low_threshold: float | None = None,
    high_threshold: float | None = None,
    status: str | None = None,
) -> object:
    """Build one normalized ZS32 prediction for strict-fusion tests."""
    return fusion.BranchPrediction(
        part_id=part_id,
        side="zs32",
        view=view,
        slot_id=None,
        branch=branch,
        pred_label=0,
        score=score,
        threshold=None,
        defect_type=None,
        reason=None,
        source_path=f"{part_id}_{view}.png",
        status=status,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )


def _clear_zs32_rows(fusion: ModuleType, part_id: str) -> list[object]:
    """Build all required six-view rows with CLEAR/PASS evidence."""
    rows = []
    for view in ZS32_VIEWS:
        rows.extend(
            [
                _zs32_prediction(fusion, part_id=part_id, view=view, branch="quality_gate", status="PASS"),
                _zs32_prediction(fusion, part_id=part_id, view=view, branch="registration", status="PASS"),
                _zs32_prediction(
                    fusion,
                    part_id=part_id,
                    view=view,
                    branch=f"anomaly_{view}",
                    score=0.2,
                    low_threshold=0.3,
                    high_threshold=0.5,
                ),
                _zs32_prediction(fusion, part_id=part_id, view=view, branch="yolo"),
                _zs32_prediction(fusion, part_id=part_id, view=view, branch="geometry"),
            ],
        )
    return rows


STRICT_VERSIONS = {
    "model_version": "zs32-models-2026.07.13",
    "threshold_version": "zs32-thresholds-2026.07.13",
    "roi_version": "zs32-roi-2026.07.12",
    "template_version": "zs32-templates-2026.07.13",
}


def _strict_zs32_config() -> dict[str, object]:
    """Return a complete two-hand production identity/version contract."""
    config = _zs32_config()
    config.update(
        {
            "profile": "zs32_six_view_v1",
            "identity": {
                "product": "ZS32",
                "profile": "zs32_six_view_v1",
                "allowed_hands": ["left", "right"],
                "required_side": "zs32",
            },
            "expected_versions": [
                {
                    "hand": hand,
                    "side": "zs32",
                    "view": view,
                    "branch": branch,
                    **STRICT_VERSIONS,
                }
                for hand in ("left", "right")
                for view in ZS32_VIEWS
                for branch in ("quality_gate", "registration", f"anomaly_{view}", "yolo", "geometry")
            ],
        },
    )
    return config


def _strict_zs32_rows(fusion: ModuleType, part_id: str, *, hand: str = "left") -> list[object]:
    """Build complete strict rows carrying real deployment identity and versions."""
    return [
        fusion.BranchPrediction(
            **{
                **row.__dict__,
                "product": "ZS32",
                "profile": "zs32_six_view_v1",
                "hand": hand,
                "capture_session": "capture-session-001",
                "group_id": "group-001",
                **STRICT_VERSIONS,
            },
        )
        for row in _clear_zs32_rows(fusion, part_id)
    ]


def _face_rows(fusion: ModuleType, part_id: str, face: str) -> list[object]:
    """Return the three rows belonging to one physical face."""
    return [row for row in _clear_zs32_rows(fusion, part_id) if str(row.view).startswith(face)]


def test_front_face_never_returns_final_ok() -> None:
    """A clear front stage must wait for the matching back stage."""
    fusion = load_fusion_module()

    result = fusion.fuse_face_predictions("p1", _face_rows(fusion, "p1", "front"), face="front", config=_zs32_config())

    assert result.stage_status == "FRONT_CLEAR"
    assert result.final_status is None
    assert result.inspection_complete is False


def test_front_strong_is_front_ng_but_marks_inspection_incomplete() -> None:
    """Front-side NG remains staged until back-side evidence is captured."""
    fusion = load_fusion_module()
    rows = _face_rows(fusion, "p1", "front")
    rows[4] = fusion.BranchPrediction(**{**rows[4].__dict__, "pred_label": 1})

    result = fusion.fuse_face_predictions("p1", rows, face="front", config=_zs32_config())

    assert result.stage_status == "FRONT_NG"
    assert result.final_status is None
    assert result.inspection_complete is False


def test_front_review_and_back_stage_equivalents() -> None:
    """Both faces expose CLEAR, REVIEW, and NG staged states without early release."""
    fusion = load_fusion_module()
    front_rows = _face_rows(fusion, "p1", "front")
    front_rows[2] = fusion.BranchPrediction(**{**front_rows[2].__dict__, "score": 0.4})
    back_clear = fusion.fuse_face_predictions(
        "p1",
        _face_rows(fusion, "p1", "back"),
        face="back",
        config=_zs32_config(),
    )
    back_ng_rows = _face_rows(fusion, "p1", "back")
    back_ng_rows[4] = fusion.BranchPrediction(**{**back_ng_rows[4].__dict__, "pred_label": 1})

    front_review = fusion.fuse_face_predictions("p1", front_rows, face="front", config=_zs32_config())
    back_ng = fusion.fuse_face_predictions("p1", back_ng_rows, face="back", config=_zs32_config())

    assert front_review.stage_status == "FRONT_REVIEW"
    assert back_clear.stage_status == "BACK_CLEAR"
    assert back_ng.stage_status == "BACK_NG"
    assert back_clear.final_status is None


def test_back_gray_is_back_review_with_primary_evidence() -> None:
    """Back-side gray evidence must retain the branch that caused review."""
    fusion = load_fusion_module()
    rows = _face_rows(fusion, "p1", "back")
    rows[2] = fusion.BranchPrediction(**{**rows[2].__dict__, "score": 0.4})

    result = fusion.fuse_face_predictions("p1", rows, face="back", config=_zs32_config())

    assert result.stage_status == "BACK_REVIEW"
    assert result.triggered_branch == "anomaly_back"
    assert result.defect_side == "zs32"
    assert result.defect_view == "back"
    assert result.reason == "anomaly_back gray evidence score=0.4 low_threshold=0.3 high_threshold=0.5"


def test_face_prediction_rejects_wrong_face_and_identity() -> None:
    """Face fusion must reject unsupported faces and mixed part identities."""
    fusion = load_fusion_module()
    rows = _face_rows(fusion, "other", "front")

    with pytest.raises(ValueError, match="part_id"):
        fusion.fuse_face_predictions("p1", rows, face="front", config=_zs32_config())
    with pytest.raises(ValueError, match="face"):
        fusion.fuse_face_predictions("p1", [], face="side", config=_zs32_config())


def test_face_prediction_without_config_fails_closed_on_missing_view() -> None:
    """Face fusion must require all three physical views even without a config."""
    fusion = load_fusion_module()
    rows = [row for row in _face_rows(fusion, "p1", "front") if row.view != "front_right"]

    result = fusion.fuse_face_predictions("p1", rows, face="front")

    assert result.stage_status == "FRONT_REVIEW"
    assert "front_right" in result.reason


@pytest.mark.parametrize(
    ("front_stage", "back_stage", "expected"),
    [
        ("FRONT_CLEAR", "BACK_CLEAR", "OK"),
        ("FRONT_REVIEW", "BACK_CLEAR", "REVIEW"),
        ("FRONT_CLEAR", "BACK_REVIEW", "REVIEW"),
        ("FRONT_NG", "BACK_CLEAR", "NG"),
        ("FRONT_CLEAR", "BACK_NG", "NG"),
    ],
)
def test_final_combination(front_stage: str, back_stage: str, expected: str) -> None:
    """Final OK requires two clear faces; REVIEW and NG fail closed."""
    fusion = load_fusion_module()
    front = fusion.FaceDecision(
        part_id="p1",
        face="front",
        stage_status=front_stage,
        final_status=None,
        inspection_complete=False,
        triggered_evidence=(),
        reason=front_stage,
    )
    back = fusion.FaceDecision(
        part_id="p1",
        face="back",
        stage_status=back_stage,
        final_status=None,
        inspection_complete=False,
        triggered_evidence=(),
        reason=back_stage,
    )

    result = fusion.combine_face_decisions(front, back)

    assert result.final_status == expected


def test_final_combination_rejects_identity_mismatch() -> None:
    """Front and back evidence from different parts must never be combined."""
    fusion = load_fusion_module()
    front = fusion.FaceDecision(
        part_id="p1",
        face="front",
        stage_status="FRONT_CLEAR",
        final_status=None,
        inspection_complete=False,
        triggered_evidence=(),
        reason="clear",
    )
    back = fusion.FaceDecision(
        part_id="p2",
        face="back",
        stage_status="BACK_CLEAR",
        final_status=None,
        inspection_complete=False,
        triggered_evidence=(),
        reason="clear",
    )

    with pytest.raises(ValueError, match="part_id"):
        fusion.combine_face_decisions(front, back)


def test_final_combination_rejects_malformed_stage_status() -> None:
    """A stage with the wrong face prefix must never be interpreted as CLEAR."""
    fusion = load_fusion_module()
    front = fusion.FaceDecision("p1", "front", "BACK_CLEAR", None, False, (), "invalid")
    back = fusion.FaceDecision("p1", "back", "BACK_CLEAR", None, False, (), "clear")

    with pytest.raises(ValueError, match="stage_status"):
        fusion.combine_face_decisions(front, back)


@pytest.mark.parametrize(
    ("front_state", "back_state", "expected_status", "expected_primary"),
    [
        ("NG", "REVIEW", "NG", "front"),
        ("REVIEW", "NG", "NG", "back"),
        ("NG", "NG", "NG", "front"),
        ("REVIEW", "CLEAR", "REVIEW", "front"),
        ("CLEAR", "REVIEW", "REVIEW", "back"),
        ("REVIEW", "REVIEW", "REVIEW", "front"),
        ("CLEAR", "CLEAR", "OK", None),
    ],
)
def test_final_combination_preserves_decisive_primary_evidence(
    front_state: str,
    back_state: str,
    expected_status: str,
    expected_primary: str | None,
) -> None:
    """Final decisions retain primary evidence using deterministic face priority."""
    fusion = load_fusion_module()

    def face_decision(face: str, state: str) -> object:
        return fusion.FaceDecision(
            part_id="p1",
            face=face,
            stage_status=f"{face.upper()}_{state}",
            final_status=None,
            inspection_complete=False,
            triggered_evidence=(),
            reason=f"{face}_reason",
            defect_side=f"{face}_side",
            defect_view=f"{face}_view",
            defect_slot=f"{face}_slot",
            defect_type=f"{face}_type",
            triggered_branch=f"{face}_branch",
        )

    result = fusion.combine_face_decisions(face_decision("front", front_state), face_decision("back", back_state))

    assert result.final_status == expected_status
    if expected_primary is None:
        assert result.triggered_branch is None
        assert result.defect_side is None
        assert result.defect_view is None
        assert result.defect_slot is None
        assert result.defect_type is None
        assert result.reason == "FRONT_CLEAR + BACK_CLEAR"
    else:
        assert result.triggered_branch == f"{expected_primary}_branch"
        assert result.defect_side == f"{expected_primary}_side"
        assert result.defect_view == f"{expected_primary}_view"
        assert result.defect_slot == f"{expected_primary}_slot"
        assert result.defect_type == f"{expected_primary}_type"
        assert result.reason == f"{expected_primary}_reason"


def test_gray_evidence_returns_review_and_strong_wins() -> None:
    """GRAY evidence requires review, while any STRONG evidence remains decisive."""
    fusion = load_fusion_module()
    config = _zs32_config()
    p1_rows = _clear_zs32_rows(fusion, "p1")
    p2_rows = _clear_zs32_rows(fusion, "p2")
    p1_rows[2] = _zs32_prediction(
        fusion,
        part_id="p1",
        view="front",
        branch="anomaly_front",
        score=0.4,
        low_threshold=0.3,
        high_threshold=0.5,
    )
    p2_rows[2] = _zs32_prediction(
        fusion,
        part_id="p2",
        view="front",
        branch="anomaly_front",
        score=0.4,
        low_threshold=0.3,
        high_threshold=0.5,
    )
    p2_rows[4] = fusion.BranchPrediction(
        **{
            **p2_rows[4].__dict__,
            "pred_label": 1,
            "score": 0.8,
            "low_threshold": 0.3,
            "high_threshold": 0.5,
        },
    )

    review = fusion.fuse_part_predictions("p1", p1_rows, config=config)
    ng = fusion.fuse_part_predictions("p2", p2_rows, config=config)

    assert review.final_status == "REVIEW"
    assert [item.evidence_id for item in review.triggered_evidence] == ["front:anomaly_front"]
    assert [item.level for item in review.triggered_evidence] == ["GRAY"]
    assert ng.final_status == "NG_GEOMETRY"
    assert ng.triggered_branch == "geometry"
    assert [item.evidence_id for item in ng.triggered_evidence] == [
        "front:geometry",
        "front:anomaly_front",
    ]
    assert [item.level for item in ng.triggered_evidence] == ["STRONG", "GRAY"]


def test_same_view_branch_slots_keep_primary_strong_and_unique_evidence() -> None:
    """Repeated view/branch triggers should retain slot identity and the actual STRONG primary."""
    fusion = load_fusion_module()
    predictions = [
        fusion.BranchPrediction(
            part_id="p1",
            side="zs32",
            view="front",
            slot_id="slot01",
            branch="geometry",
            pred_label=1,
            score=0.8,
            threshold=None,
            defect_type="less",
            reason="slot01 strong geometry",
            source_path="p1_front_slot01.png",
            low_threshold=0.3,
            high_threshold=0.5,
        ),
        fusion.BranchPrediction(
            part_id="p1",
            side="zs32",
            view="front",
            slot_id="slot02",
            branch="geometry",
            pred_label=0,
            score=0.4,
            threshold=None,
            defect_type="more",
            reason="slot02 gray geometry",
            source_path="p1_front_slot02.png",
            low_threshold=0.3,
            high_threshold=0.5,
        ),
    ]

    decision = fusion.fuse_part_predictions("p1", predictions, config={"branch_order": ["geometry"]})

    assert decision.final_status == "NG_GEOMETRY"
    assert decision.defect_slot == "slot01"
    assert decision.defect_type == "less"
    assert decision.reason == "slot01 strong geometry"
    assert [item.evidence_id for item in decision.triggered_evidence] == [
        "front:geometry:slot01",
        "front:geometry:slot02",
    ]
    assert [item.level for item in decision.triggered_evidence] == ["STRONG", "GRAY"]
    assert len({item.evidence_id for item in decision.triggered_evidence}) == 2


def test_ok_requires_every_configured_branch_per_view() -> None:
    """A present view remains incomplete when one of its required branches is absent."""
    fusion = load_fusion_module()
    rows = [
        row
        for row in _clear_zs32_rows(fusion, "p1")
        if not (row.view == "back_right" and row.branch == "anomaly_back_right")
    ]

    decision = fusion.fuse_part_predictions("p1", rows, config=_zs32_config())

    assert decision.final_status == "REVIEW"
    assert "missing required branch: back_right:anomaly_back_right" in decision.reason


def test_all_six_views_and_required_branches_clear_returns_ok() -> None:
    """Only complete six-view CLEAR/PASS evidence may produce OK."""
    fusion = load_fusion_module()

    decision = fusion.fuse_part_predictions("p1", _clear_zs32_rows(fusion, "p1"), config=_zs32_config())

    assert decision.final_status == "OK"
    assert decision.final_label == 0
    assert decision.triggered_evidence == ()


def test_normalized_csv_preserves_full_zs32_identity_including_hand(tmp_path: Path) -> None:
    """CSV normalization must retain the identity dimensions used by the strict gate."""
    fusion = load_fusion_module()
    csv_path = tmp_path / "identity.csv"
    csv_path.write_text(
        "part_id,product,profile,hand,capture_session,group_id,side,view,branch,pred_label,source_hash,manifest_identity\n"
        "p1,ZS32,zs32_six_view_v1,left,session-001,group-001,zs32,front,geometry,0,abc123,capture-001\n",
        encoding="utf-8",
    )

    prediction = fusion.load_branch_predictions_csv(csv_path, branch="geometry")[0]
    output = tmp_path / "normalized.csv"
    fusion.write_branch_predictions_csv([prediction], output)

    assert prediction.product == "ZS32"
    assert prediction.profile == "zs32_six_view_v1"
    assert prediction.hand == "left"
    assert prediction.capture_session == "session-001"
    assert prediction.group_id == "group-001"
    assert prediction.source_hash == "abc123"
    assert prediction.manifest_identity == "capture-001"
    header = output.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header[:3] == ["part_id", "side", "view"]
    assert {"product", "profile", "hand", "capture_session", "group_id"} <= set(header)


@pytest.mark.parametrize("field", ["capture_session", "group_id"])
def test_strict_identity_requires_nonempty_capture_identity(field: str) -> None:
    """Every strict row must carry both acquisition-session and capture-group identity."""
    fusion = load_fusion_module()
    rows = _strict_zs32_rows(fusion, "p1")
    rows[0] = fusion.BranchPrediction(**{**rows[0].__dict__, field: None})

    decision = fusion.fuse_part_predictions("p1", rows, config=_strict_zs32_config())

    assert decision.final_status == "INVALID_CAPTURE"
    assert f"missing {field}" in decision.reason


@pytest.mark.parametrize("field", ["capture_session", "group_id"])
def test_strict_identity_rejects_mixed_capture_identity(field: str) -> None:
    """All branches and all six views of one part must share capture identity."""
    fusion = load_fusion_module()
    rows = _strict_zs32_rows(fusion, "p1")
    rows[-1] = fusion.BranchPrediction(**{**rows[-1].__dict__, field: "other"})

    decision = fusion.fuse_part_predictions("p1", rows, config=_strict_zs32_config())

    assert decision.final_status == "INVALID_CAPTURE"
    assert f"mixed {field}" in decision.reason


@pytest.mark.parametrize(("field", "wrong_value"), [("hand", "right"), ("side", "back")])
def test_wrong_hand_or_side_cannot_satisfy_strict_required_identity(field: str, wrong_value: str) -> None:
    """A row from the wrong physical identity must not satisfy a required branch."""
    fusion = load_fusion_module()
    rows = _strict_zs32_rows(fusion, "p1")
    target = next(index for index, row in enumerate(rows) if row.view == "front" and row.branch == "geometry")
    rows[target] = fusion.BranchPrediction(**{**rows[target].__dict__, field: wrong_value})

    decision = fusion.fuse_part_predictions("p1", rows, config=_strict_zs32_config())

    assert decision.final_status == "INVALID_CAPTURE"
    expected_reason = "mixed hand identity" if field == "hand" else "left:zs32:front:geometry"
    assert expected_reason in decision.reason


@pytest.mark.parametrize("hand", ["left", "right"])
def test_strict_identity_accepts_complete_single_hand_parts(hand: str) -> None:
    """The shared profile accepts either declared hand when the whole part is consistent."""
    fusion = load_fusion_module()

    decision = fusion.fuse_part_predictions(
        "p1",
        _strict_zs32_rows(fusion, "p1", hand=hand),
        config=_strict_zs32_config(),
    )

    assert decision.final_status == "OK"


def test_strict_identity_rejects_mixed_hands_within_one_part() -> None:
    """Left/right evidence may not be combined under one physical part identity."""
    fusion = load_fusion_module()
    rows = _strict_zs32_rows(fusion, "p1")
    rows[-1] = fusion.BranchPrediction(**{**rows[-1].__dict__, "hand": "right"})

    decision = fusion.fuse_part_predictions("p1", rows, config=_strict_zs32_config())

    assert decision.final_status == "INVALID_CAPTURE"
    assert "mixed hand identity" in decision.reason


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_version", None),
        ("threshold_version", "wrong-thresholds"),
        ("roi_version", None),
        ("template_version", "wrong-template"),
    ],
)
def test_strict_expected_versions_are_required_and_exact(field: str, value: str | None) -> None:
    """Every applicable strict identity must provide all four exact deployment versions."""
    fusion = load_fusion_module()
    rows = _strict_zs32_rows(fusion, "p1")
    target = next(index for index, row in enumerate(rows) if row.view == "front" and row.branch == "geometry")
    rows[target] = fusion.BranchPrediction(**{**rows[target].__dict__, field: value})

    decision = fusion.fuse_part_predictions("p1", rows, config=_strict_zs32_config())

    assert decision.final_status == "REVIEW"
    assert f"{field} mismatch for left:zs32:front:geometry" in decision.reason


def test_missing_strict_expected_versions_config_fails_closed() -> None:
    """A strict deployment without an expected-version contract cannot release OK."""
    fusion = load_fusion_module()
    config = _strict_zs32_config()
    config.pop("expected_versions")

    decision = fusion.fuse_part_predictions("p1", _strict_zs32_rows(fusion, "p1"), config=config)

    assert decision.final_status == "REVIEW"
    assert "missing expected_versions contract" in decision.reason


@pytest.mark.parametrize("identity_field", ["source_path", "source_hash", "manifest_identity"])
def test_reused_source_identity_across_required_views_is_invalid_capture(identity_field: str) -> None:
    """Distinct required views of one part must never reuse one captured image identity."""
    fusion = load_fusion_module()
    rows = _strict_zs32_rows(fusion, "p1")
    updates = {
        "source_path": "captures/reused.png",
        "source_hash": "sha256-reused",
        "manifest_identity": "manifest-image-reused",
    }
    for index, row in enumerate(rows):
        if row.view in {"front", "back"}:
            rows[index] = fusion.BranchPrediction(**{**row.__dict__, identity_field: updates[identity_field]})

    decision = fusion.fuse_part_predictions("p1", rows, config=_strict_zs32_config())

    assert decision.final_status == "INVALID_CAPTURE"
    assert f"reused {identity_field} across required views" in decision.reason


def test_strong_ng_is_not_downgraded_by_incomplete_or_version_faults() -> None:
    """Immutable STRONG evidence remains machine NG while system faults block release separately."""
    fusion = load_fusion_module()
    rows = [row for row in _strict_zs32_rows(fusion, "p1") if row.view != "back_right"]
    target = next(index for index, row in enumerate(rows) if row.view == "front" and row.branch == "geometry")
    rows[target] = fusion.BranchPrediction(
        **{
            **rows[target].__dict__,
            "pred_label": 1,
            "score": 0.8,
            "low_threshold": 0.3,
            "high_threshold": 0.5,
            "model_version": "wrong-model",
        },
    )

    decision = fusion.fuse_part_predictions("p1", rows, config=_strict_zs32_config())

    assert decision.final_status == "NG_GEOMETRY"
    assert decision.final_label == 1
    assert decision.triggered_branch == "geometry"
    assert decision.triggered_evidence[0].evidence_id == "front:geometry"
    assert any(item.evidence_id == "system:input_contract" for item in decision.triggered_evidence)


def test_strong_ng_retains_every_nonpass_gate_trigger() -> None:
    """Strong evidence stays NG while every quality/registration failure remains auditable."""
    fusion = load_fusion_module()
    rows = _strict_zs32_rows(fusion, "p1")
    geometry = next(index for index, row in enumerate(rows) if row.view == "front" and row.branch == "geometry")
    quality = next(index for index, row in enumerate(rows) if row.view == "front_left" and row.branch == "quality_gate")
    registration = next(
        index for index, row in enumerate(rows) if row.view == "back_right" and row.branch == "registration"
    )
    rows[geometry] = fusion.BranchPrediction(
        **{**rows[geometry].__dict__, "pred_label": 1, "score": 0.8, "low_threshold": 0.3, "high_threshold": 0.5},
    )
    rows[quality] = fusion.BranchPrediction(**{**rows[quality].__dict__, "status": "WARN"})
    rows[registration] = fusion.BranchPrediction(**{**rows[registration].__dict__, "status": "FAIL", "pred_label": 1})

    decision = fusion.fuse_part_predictions("p1", rows, config=_strict_zs32_config())

    assert decision.final_status == "NG_GEOMETRY"
    assert decision.final_label == 1
    assert {item.evidence_id for item in decision.triggered_evidence} >= {
        "front:geometry",
        "front_left:quality_gate",
        "back_right:registration",
    }


@pytest.mark.parametrize(
    ("branch", "status"),
    [("quality_gate", None), ("registration", "   ")],
)
def test_strict_gate_requires_explicit_pass_status(branch: str, status: str | None) -> None:
    """A present strict gate row without an explicit PASS must never release OK."""
    fusion = load_fusion_module()
    rows = _strict_zs32_rows(fusion, "p1")
    target = next(index for index, row in enumerate(rows) if row.view == "front" and row.branch == branch)
    rows[target] = fusion.BranchPrediction(**{**rows[target].__dict__, "status": status})

    decision = fusion.fuse_part_predictions("p1", rows, config=_strict_zs32_config())

    assert decision.final_status == "RETAKE"
    assert decision.final_label is None
    assert decision.triggered_branch == branch
    assert any(item.evidence_id == f"front:{branch}" for item in decision.triggered_evidence)
    assert "explicit PASS" in decision.reason


def test_strong_ng_retains_missing_gate_status_and_blocks_inspection_completion() -> None:
    """STRONG remains NG while a blank strict gate status stays auditable and incomplete."""
    fusion = load_fusion_module()
    audit_module = _load_capture_data_module("missing_gate_status_audit", "inspection_audit.py")
    rows = _strict_zs32_rows(fusion, "p1")
    geometry = next(index for index, row in enumerate(rows) if row.view == "front" and row.branch == "geometry")
    quality = next(index for index, row in enumerate(rows) if row.view == "front_left" and row.branch == "quality_gate")
    rows[geometry] = fusion.BranchPrediction(
        **{**rows[geometry].__dict__, "pred_label": 1, "score": 0.8, "low_threshold": 0.3, "high_threshold": 0.5},
    )
    rows[quality] = fusion.BranchPrediction(**{**rows[quality].__dict__, "status": None})

    decision = fusion.fuse_part_predictions("p1", rows, config=_strict_zs32_config())
    audit = audit_module.build_part_audit(
        "p1",
        [],
        machine_status=decision.final_status,
        triggered_evidence=decision.triggered_evidence,
        inspection_complete=True,
    )

    assert decision.final_status == "NG_GEOMETRY"
    assert decision.final_label == 1
    assert {item.evidence_id for item in decision.triggered_evidence} >= {
        "front:geometry",
        "front_left:quality_gate",
    }
    assert audit["inspection_complete"] is False


def test_yolo_structured_multi_box_is_one_required_identity(tmp_path: Path) -> None:
    """One YOLO summary may contain multiple boxes without duplicating its required branch identity."""
    fusion = load_fusion_module()
    detections = [
        {"class": "crack", "confidence": 0.91, "xyxy": [1, 2, 11, 22], "area": 200, "in_roi": True},
        {"class": "chip", "confidence": 0.82, "xyxy": [30, 40, 50, 70], "area": 600, "touches_border": True},
    ]
    csv_path = tmp_path / "yolo.csv"
    csv_path.write_text(
        "part_id,side,view,branch,pred_label,detections\n"
        f'p1,zs32,front,yolo,1,"{json.dumps(detections).replace(chr(34), chr(34) * 2)}"\n',
        encoding="utf-8",
    )

    prediction = fusion.load_branch_predictions_csv(csv_path, branch="yolo")[0]
    rows = _strict_zs32_rows(fusion, "p1")
    target = next(index for index, row in enumerate(rows) if row.view == "front" and row.branch == "yolo")
    rows[target] = fusion.BranchPrediction(**{
        **rows[target].__dict__,
        "pred_label": 1,
        "detections": prediction.detections,
    })
    decision = fusion.fuse_part_predictions("p1", rows, config=_strict_zs32_config())
    output = tmp_path / "normalized.csv"
    fusion.write_branch_predictions_csv([rows[target]], output)
    reloaded = fusion.load_branch_predictions_csv(output, branch="yolo")[0]

    assert decision.final_status == "NG_YOLO"
    assert "duplicate required identity" not in decision.reason
    assert list(reloaded.detections) == detections


def test_yolo_empty_detection_list_is_clear_summary(tmp_path: Path) -> None:
    """An explicit no-box YOLO summary is a valid CLEAR required identity."""
    fusion = load_fusion_module()
    csv_path = tmp_path / "yolo-clear.csv"
    csv_path.write_text(
        'part_id,side,view,branch,pred_label,detections\np1,zs32,front,yolo,0,"[]"\n',
        encoding="utf-8",
    )

    prediction = fusion.load_branch_predictions_csv(csv_path, branch="yolo")[0]

    assert prediction.pred_label == 0
    assert prediction.detections == ()
    assert fusion.classify_evidence(prediction) is fusion.EvidenceLevel.CLEAR


def test_dual_thresholds_classify_clear_gray_and_strong(tmp_path: Path) -> None:
    """Raw scores should map to the three dual-threshold evidence bands."""
    fusion = load_fusion_module()
    csv_path = tmp_path / "predictions.csv"
    csv_path.write_text(
        "part_id,side,view,raw_score,low_threshold,high_threshold,model_version,threshold_version\n"
        "p1,zs32,front,0.20,0.30,0.50,m1,t1\n"
        "p2,zs32,front,0.40,0.30,0.50,m1,t1\n"
        "p3,zs32,front,0.60,0.30,0.50,m1,t1\n",
        encoding="utf-8",
    )

    rows = fusion.load_branch_predictions_csv(csv_path, branch="anomaly_front")

    assert [fusion.classify_evidence(row).value for row in rows] == ["CLEAR", "GRAY", "STRONG"]
    assert rows[0].model_version == "m1"
    assert rows[0].threshold_version == "t1"


@pytest.mark.parametrize(("pred_label", "expected"), [(0, "CLEAR"), (1, "STRONG")])
def test_legacy_evidence_uses_pred_label_without_dual_thresholds(pred_label: int, expected: str) -> None:
    """Legacy predictions should retain their binary evidence semantics."""
    fusion = load_fusion_module()
    prediction = fusion.BranchPrediction(
        part_id="legacy-part",
        side="zs32",
        view="front",
        slot_id=None,
        branch="anomaly_front",
        pred_label=pred_label,
        score=0.4,
        threshold=0.5,
        defect_type=None,
        reason=None,
        source_path=None,
    )

    assert fusion.classify_evidence(prediction).value == expected


@pytest.mark.parametrize(
    ("raw_score", "low_threshold", "high_threshold", "error_text"),
    [
        ("0.4", "0.6", "0.5", "invalid dual thresholds"),
        ("nan", "0.3", "0.5", "invalid dual thresholds"),
        ("0.4", "0.3", "", "incomplete dual thresholds"),
    ],
)
def test_invalid_dual_thresholds_include_prediction_identity(
    tmp_path: Path,
    raw_score: str,
    low_threshold: str,
    high_threshold: str,
    error_text: str,
) -> None:
    """Malformed dual-threshold rows should identify the part and branch."""
    fusion = load_fusion_module()
    csv_path = tmp_path / "predictions.csv"
    csv_path.write_text(
        "part_id,side,view,raw_score,low_threshold,high_threshold\n"
        f"bad-part,zs32,front,{raw_score},{low_threshold},{high_threshold}\n",
        encoding="utf-8",
    )
    prediction = fusion.load_branch_predictions_csv(csv_path, branch="anomaly_front")[0]

    with pytest.raises(ValueError, match=rf"{error_text}.*bad-part:anomaly_front"):
        fusion.classify_evidence(prediction)


def test_missing_required_view_returns_invalid_capture() -> None:
    """Missing required side/view inputs should fail closed before branch fusion."""
    fusion = load_fusion_module()

    decision = fusion.fuse_part_predictions(
        "part001",
        [],
        missing_required=("top:uniform",),
    )

    assert decision.final_status == "INVALID_CAPTURE"
    assert decision.final_label is None
    assert "top:uniform" in decision.reason


def test_quality_failure_returns_retake_before_model_positive() -> None:
    """Quality failures should ask for a retake instead of forcing OK/NG inference."""
    fusion = load_fusion_module()
    predictions = [
        fusion.BranchPrediction(
            part_id="part001",
            side="top",
            view="uniform",
            slot_id=None,
            branch="quality",
            pred_label=1,
            score=None,
            threshold=None,
            defect_type=None,
            reason="blur_laplacian_var below minimum",
            source_path="part001_top_uniform.png",
        ),
        fusion.BranchPrediction(
            part_id="part001",
            side="top",
            view="uniform",
            slot_id="slot02",
            branch="geometry",
            pred_label=1,
            score=12.0,
            threshold=10.0,
            defect_type="less",
            reason=None,
            source_path="part001_top_uniform.png",
        ),
    ]

    decision = fusion.fuse_part_predictions("part001", predictions)

    assert decision.final_status == "RETAKE"
    assert decision.final_label is None
    assert decision.triggered_branch == "quality"
    assert "blur_laplacian_var" in decision.reason


def test_branch_order_maps_positive_predictions_to_ng_statuses() -> None:
    """The first positive branch in configured order should explain the NG status."""
    fusion = load_fusion_module()
    predictions = [
        fusion.BranchPrediction(
            part_id="part001",
            side="top",
            view="uniform",
            slot_id="slot02",
            branch="anomaly_dino",
            pred_label=1,
            score=0.72,
            threshold=0.5,
            defect_type="surface",
            reason=None,
            source_path="part001_top_uniform.png",
        ),
        fusion.BranchPrediction(
            part_id="part001",
            side="top",
            view="uniform",
            slot_id="slot02",
            branch="geometry",
            pred_label=1,
            score=12.0,
            threshold=10.0,
            defect_type="less",
            reason=None,
            source_path="part001_top_uniform.png",
        ),
    ]

    decision = fusion.fuse_part_predictions("part001", predictions)

    assert decision.final_status == "NG_GEOMETRY"
    assert decision.final_label == 1
    assert decision.triggered_branch == "geometry"
    assert decision.defect_slot == "slot02"
    assert decision.defect_type == "less"


def test_anomaly_positive_maps_to_ng_anomaly() -> None:
    """AnomalyDINO positives should produce NG_ANOMALY when geometry is clean or absent."""
    fusion = load_fusion_module()
    predictions = [
        fusion.BranchPrediction(
            part_id="part001",
            side="top",
            view="uniform",
            slot_id="slot05",
            branch="anomaly_dino",
            pred_label=1,
            score=0.72,
            threshold=0.5,
            defect_type="surface",
            reason=None,
            source_path="part001_top_uniform.png",
        ),
    ]

    decision = fusion.fuse_part_predictions("part001", predictions)

    assert decision.final_status == "NG_ANOMALY"
    assert decision.final_label == 1
    assert decision.triggered_branch == "anomaly_dino"


def test_ok_requires_quality_and_registration_pass_rows() -> None:
    """Configured OK prerequisites should fail closed when PASS rows are missing."""
    fusion = load_fusion_module()
    predictions = [
        fusion.BranchPrediction(
            part_id="part001",
            side="top",
            view="uniform",
            slot_id="slot01",
            branch="geometry",
            pred_label=0,
            score=1.0,
            threshold=10.0,
            defect_type="none",
            reason=None,
            source_path="part001_top_uniform.png",
        ),
    ]

    decision = fusion.fuse_part_predictions(
        "part001",
        predictions,
        config={"ok_requires": {"quality_gate": "PASS", "registration": "PASS"}},
    )

    assert decision.final_status == "RETAKE"
    assert decision.final_label is None
    assert decision.triggered_branch == "quality_gate"
    assert "missing required quality_gate PASS" in decision.reason


def test_ok_requires_rejects_warn_quality_status() -> None:
    """Configured PASS prerequisites should not treat WARN gate rows as OK."""
    fusion = load_fusion_module()
    predictions = [
        fusion.BranchPrediction(
            part_id="part001",
            side="top",
            view="uniform",
            slot_id=None,
            branch="quality_gate",
            pred_label=0,
            score=None,
            threshold=None,
            defect_type=None,
            reason="brightness near limit",
            source_path="part001_top_uniform.png",
            status="WARN",
        ),
        fusion.BranchPrediction(
            part_id="part001",
            side="top",
            view="uniform",
            slot_id="slot01",
            branch="geometry",
            pred_label=0,
            score=1.0,
            threshold=10.0,
            defect_type="none",
            reason=None,
            source_path="part001_top_uniform.png",
        ),
    ]

    decision = fusion.fuse_part_predictions(
        "part001",
        predictions,
        config={"ok_requires": {"quality_gate": "PASS"}},
    )

    assert decision.final_status == "RETAKE"
    assert decision.triggered_branch == "quality_gate"
    assert "required quality_gate PASS" in decision.reason


def test_legacy_warn_gate_without_pass_contract_remains_ok() -> None:
    """Legacy C789-style fusion must not promote an advisory WARN into REVIEW."""
    fusion = load_fusion_module()
    prediction = fusion.BranchPrediction(
        part_id="legacy-part",
        side="top",
        view="uniform",
        slot_id=None,
        branch="quality_gate",
        pred_label=0,
        score=None,
        threshold=None,
        defect_type=None,
        reason="advisory brightness warning",
        source_path="legacy.png",
        status="WARN",
    )

    decision = fusion.fuse_part_predictions("legacy-part", [prediction])

    assert decision.final_status == "OK"
    assert decision.triggered_evidence == ()


def test_required_view_config_is_checked_without_manifest() -> None:
    """Fusion config required side/view keys should work even without a manifest CSV."""
    fusion = load_fusion_module()
    predictions = [
        fusion.BranchPrediction(
            part_id="part001",
            side="top",
            view="uniform",
            slot_id="slot01",
            branch="geometry",
            pred_label=0,
            score=1.0,
            threshold=10.0,
            defect_type="none",
            reason=None,
            source_path="part001_top_uniform_slot01.png",
        ),
    ]

    missing = fusion.missing_required_views(
        predictions,
        {"ok_requires": {"required_sides": ["top", "bottom"], "required_views": ["uniform"]}},
    )
    decision = fusion.fuse_part_predictions("part001", predictions, missing_required=missing)

    assert missing == ["bottom:uniform"]
    assert decision.final_status == "INVALID_CAPTURE"


def test_config_scalar_values_are_not_split_into_characters() -> None:
    """Scalar config values should be treated as one value, not a character sequence."""
    fusion = load_fusion_module()
    predictions = [
        fusion.BranchPrediction(
            part_id="part001",
            side="top",
            view="uniform",
            slot_id="slot01",
            branch="geometry",
            pred_label=1,
            score=12.0,
            threshold=10.0,
            defect_type="less",
            reason=None,
            source_path="part001_top_uniform_slot01.png",
        ),
        fusion.BranchPrediction(
            part_id="part001",
            side="top",
            view="uniform",
            slot_id="slot01",
            branch="anomaly_dino",
            pred_label=1,
            score=0.9,
            threshold=0.5,
            defect_type="surface",
            reason=None,
            source_path="part001_top_uniform_slot01.png",
        ),
    ]

    missing = fusion.missing_required_views(
        predictions,
        {"ok_requires": {"required_sides": "top", "required_views": "uniform"}},
    )
    decision = fusion.fuse_part_predictions("part001", predictions, config={"branch_order": "anomaly_dino"})

    assert missing == []
    assert decision.final_status == "NG_ANOMALY"


def test_no_positive_predictions_return_ok() -> None:
    """Clean branch predictions should produce a strict OK decision."""
    fusion = load_fusion_module()
    predictions = [
        fusion.BranchPrediction(
            part_id="part001",
            side="top",
            view="uniform",
            slot_id="slot01",
            branch="geometry",
            pred_label=0,
            score=3.0,
            threshold=10.0,
            defect_type="none",
            reason=None,
            source_path="part001_top_uniform.png",
        ),
    ]

    decision = fusion.fuse_part_predictions("part001", predictions)

    assert decision.final_status == "OK"
    assert decision.final_label == 0
    assert decision.triggered_branch is None


def test_near_threshold_prediction_returns_review_when_enabled() -> None:
    """Legacy near-threshold negatives should map to the unified REVIEW status."""
    fusion = load_fusion_module()
    predictions = [
        fusion.BranchPrediction(
            part_id="part001",
            side="top",
            view="uniform",
            slot_id="slot04",
            branch="geometry",
            pred_label=0,
            score=9.2,
            threshold=10.0,
            defect_type="more",
            reason=None,
            source_path="part001_top_uniform.png",
        ),
    ]

    decision = fusion.fuse_part_predictions(
        "part001",
        predictions,
        config={"suspect_policy": {"enable": True, "near_threshold_ratio": 0.9}},
    )

    assert decision.final_status == "REVIEW"
    assert decision.final_label is None
    assert decision.triggered_branch == "geometry"
    assert "near threshold" in decision.reason


def test_legacy_surface_texture_suspect_preserves_suspect_status() -> None:
    """Non-strict C789 fusion must retain its established SUSPECT output."""
    fusion = load_fusion_module()
    predictions = [
        fusion.BranchPrediction(
            part_id="part001",
            side="top",
            view="uniform",
            slot_id="slot03",
            branch="surface_texture",
            pred_label=1,
            score=21.0,
            threshold=18.0,
            defect_type="surface",
            reason="surface_texture local_residual=21 threshold=18",
            source_path="part001_top_uniform_slot03.png",
            status="SUSPECT",
        ),
    ]

    decision = fusion.fuse_part_predictions("part001", predictions)

    assert decision.final_status == "SUSPECT"
    assert decision.final_label == 1
    assert decision.triggered_branch == "surface_texture"
    assert [item.level for item in decision.triggered_evidence] == ["STRONG"]


def test_strict_profile_maps_suspect_input_to_review() -> None:
    """Only the strict ZS32 profile maps legacy SUSPECT evidence into REVIEW."""
    fusion = load_fusion_module()
    rows = _strict_zs32_rows(fusion, "p1")
    target = next(index for index, row in enumerate(rows) if row.view == "front" and row.branch == "geometry")
    rows[target] = fusion.BranchPrediction(
        **{
            **rows[target].__dict__,
            "branch": "surface_texture",
            "status": "SUSPECT",
            "pred_label": 1,
        },
    )
    config = _strict_zs32_config()
    config["required_branches_by_view"]["front"] = [
        "quality_gate",
        "registration",
        "anomaly_front",
        "yolo",
        "surface_texture",
    ]
    expected = config["expected_versions"]
    for item in expected:
        if item["view"] == "front" and item["branch"] == "geometry":
            item["branch"] = "surface_texture"

    decision = fusion.fuse_part_predictions("p1", rows, config=config)

    assert decision.final_status == "REVIEW"
    assert decision.final_label is None
    assert decision.triggered_branch == "surface_texture"
    assert [item.level for item in decision.triggered_evidence] == ["GRAY"]


def test_load_branch_csvs_normalize_geometry_and_anomaly_predictions(tmp_path: Path) -> None:
    """Existing geometry/anomaly CSV columns should become unified branch rows."""
    fusion = load_fusion_module()
    geometry_csv = tmp_path / "geometry_predictions.csv"
    geometry_csv.write_text(
        "\n".join(
            [
                "source_path,image_path,slot,geometry_pred_label,geometry_score,geometry_threshold,geometry_type",
                "dataset/part001_top_uniform_slot02.png,,slot02,1,12.0,10.0,less",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    anomaly_csv = tmp_path / "predictions.csv"
    anomaly_csv.write_text(
        "\n".join(
            [
                "source_path,processed_path,model,view,pred_score,deploy_threshold,deploy_pred_label",
                "dataset/part002_top_uniform_slot05.png,processed/part002.png,anomaly_dino,uniform,0.7,0.5,1",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    predictions = [
        *fusion.load_branch_predictions_csv(geometry_csv, branch="geometry"),
        *fusion.load_branch_predictions_csv(anomaly_csv, branch="anomaly_dino"),
    ]

    assert [(row.part_id, row.branch, row.pred_label, row.score, row.threshold) for row in predictions] == [
        ("part001_top_uniform_slot02", "geometry", 1, 12.0, 10.0),
        ("part002_top_uniform_slot05", "anomaly_dino", 1, 0.7, 0.5),
    ]
    assert predictions[0].slot_id == "slot02"
    assert predictions[1].view == "uniform"


def test_group_predictions_merges_source_processed_and_basename_aliases(tmp_path: Path) -> None:
    """Rows from different branch CSV schemas should merge when any path alias matches."""
    fusion = load_fusion_module()
    geometry_csv = tmp_path / "geometry_predictions.csv"
    shared_processed = tmp_path / "processed" / "part001_top_uniform_slot02.png"
    geometry_csv.write_text(
        "\n".join(
            [
                "source_path,image_path,slot,geometry_pred_label,geometry_score,geometry_threshold,geometry_type",
                f"{shared_processed},,slot02,0,2.0,10.0,none",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    anomaly_csv = tmp_path / "predictions.csv"
    anomaly_csv.write_text(
        "\n".join(
            [
                "source_path,processed_path,model,view,pred_score,deploy_threshold,deploy_pred_label",
                f"{tmp_path / 'raw' / 'part001.png'},{shared_processed},anomaly_dino,uniform,0.8,0.5,1",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    predictions = [
        *fusion.load_branch_predictions_csv(geometry_csv, branch="geometry"),
        *fusion.load_branch_predictions_csv(anomaly_csv, branch="anomaly_dino"),
    ]

    grouped = fusion.group_predictions_by_part(predictions)
    decisions = fusion.fuse_grouped_predictions(grouped)

    assert len(grouped) == 1
    assert [prediction.branch for prediction in next(iter(grouped.values()))] == ["geometry", "anomaly_dino"]
    assert decisions[0].final_status == "NG_ANOMALY"


def test_group_predictions_merges_unique_basename_only_matches(tmp_path: Path) -> None:
    """Rows with a unique basename/stem match should merge when full paths differ."""
    fusion = load_fusion_module()
    geometry_csv = tmp_path / "geometry_predictions.csv"
    geometry_csv.write_text(
        "\n".join(
            [
                "source_path,image_path,slot,geometry_pred_label,geometry_score,geometry_threshold,geometry_type",
                f"{tmp_path / 'raw' / 'part001_slot02.png'},,slot02,0,2.0,10.0,none",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    anomaly_csv = tmp_path / "predictions.csv"
    anomaly_csv.write_text(
        "\n".join(
            [
                "source_path,processed_path,model,view,pred_score,deploy_threshold,deploy_pred_label",
                f"{tmp_path / 'processed' / 'part001_slot02.png'},,anomaly_dino,uniform,0.8,0.5,1",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    predictions = [
        *fusion.load_branch_predictions_csv(geometry_csv, branch="geometry"),
        *fusion.load_branch_predictions_csv(anomaly_csv, branch="anomaly_dino"),
    ]

    grouped = fusion.group_predictions_by_part(predictions)
    decisions = fusion.fuse_grouped_predictions(grouped)

    assert len(grouped) == 1
    assert [prediction.branch for prediction in next(iter(grouped.values()))] == ["geometry", "anomaly_dino"]
    assert decisions[0].final_status == "NG_ANOMALY"


def test_group_predictions_does_not_merge_ambiguous_basename_only_matches(tmp_path: Path) -> None:
    """Basename/stem aliases should not merge rows when full paths point to different inputs."""
    fusion = load_fusion_module()
    geometry_csv = tmp_path / "geometry_predictions.csv"
    geometry_csv.write_text(
        "\n".join(
            [
                "source_path,image_path,slot,geometry_pred_label,geometry_score,geometry_threshold,geometry_type",
                f"{tmp_path / 'normal' / 'shared_slot02.png'},,slot02,0,2.0,10.0,none",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    anomaly_csv = tmp_path / "predictions.csv"
    anomaly_csv.write_text(
        "\n".join(
            [
                "source_path,processed_path,model,view,pred_score,deploy_threshold,deploy_pred_label",
                f"{tmp_path / 'defect' / 'shared_slot02.png'},,anomaly_dino,uniform,0.8,0.5,1",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    predictions = [
        *fusion.load_branch_predictions_csv(geometry_csv, branch="geometry"),
        *fusion.load_branch_predictions_csv(anomaly_csv, branch="anomaly_dino"),
    ]

    grouped = fusion.group_predictions_by_part(predictions)

    assert len(grouped) == 2
    assert sorted(len(part_predictions) for part_predictions in grouped.values()) == [1, 1]


def test_write_fusion_outputs_and_benchmark_summary(tmp_path: Path) -> None:
    """Fusion outputs should be persisted and summarized for robustness reports."""
    fusion = load_fusion_module()
    predictions = [
        fusion.BranchPrediction(
            part_id="normal001",
            side="top",
            view="uniform",
            slot_id="slot01",
            branch="geometry",
            pred_label=0,
            score=1.0,
            threshold=10.0,
            defect_type="none",
            reason=None,
            source_path=str(tmp_path / "normal_test" / "normal001.png"),
        ),
        fusion.BranchPrediction(
            part_id="defect001",
            side="top",
            view="uniform",
            slot_id="slot02",
            branch="geometry",
            pred_label=1,
            score=12.0,
            threshold=10.0,
            defect_type="less",
            reason=None,
            source_path=str(tmp_path / "defect" / "defect001.png"),
        ),
    ]
    grouped = fusion.group_predictions_by_part(predictions)
    decisions = fusion.fuse_grouped_predictions(grouped)

    branch_csv = tmp_path / "branch_predictions.csv"
    fused_csv = tmp_path / "fused_predictions.csv"
    fusion.write_branch_predictions_csv(predictions, branch_csv)
    fusion.write_fused_decisions_csv(decisions, fused_csv)
    summary = fusion.compute_benchmark_summary(decisions, grouped)

    assert branch_csv.read_text(encoding="utf-8").splitlines()[0].startswith("part_id,side,view")
    assert "NG_GEOMETRY" in fused_csv.read_text(encoding="utf-8")
    assert summary["total_parts"] == 2
    assert summary["defect_recall"] == 1.0
    assert summary["fused_recall"] == 1.0
    assert summary["geometry_recall"] == 1.0
    assert summary["anomaly_dino_recall"] == 0.0
    assert summary["normal_false_positives"] == 0


def test_benchmark_summary_keeps_unknown_out_of_normal_denominator(tmp_path: Path) -> None:
    """Unknown predictions should not silently dilute normal false-positive metrics."""
    fusion = load_fusion_module()
    grouped = {
        "unknown001": [
            fusion.BranchPrediction(
                part_id="unknown001",
                side="unknown",
                view=None,
                slot_id=None,
                branch="geometry",
                pred_label=0,
                score=0.0,
                threshold=1.0,
                defect_type="none",
                reason=None,
                source_path=str(tmp_path / "misc" / "unknown001.png"),
            ),
        ],
        "normal001": [
            fusion.BranchPrediction(
                part_id="normal001",
                side="top",
                view="uniform",
                slot_id="slot01",
                branch="geometry",
                pred_label=1,
                score=12.0,
                threshold=10.0,
                defect_type="more",
                reason=None,
                source_path=str(tmp_path / "normal_test" / "normal001.png"),
            ),
        ],
    }
    decisions = fusion.fuse_grouped_predictions(grouped)

    summary = fusion.compute_benchmark_summary(
        decisions,
        grouped,
        clean_normal_root=tmp_path / "normal_test",
    )

    assert summary["unknown_total"] == 1
    assert summary["normal_total"] == 1
    assert summary["clean_normal_false_positives"] == 1
    assert summary["normal_fp_rate"] == 1.0
