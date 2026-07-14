# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Linux test definitions for split-safe ZS32 calibration."""

from __future__ import annotations

import pytest

from zs32_inspection.calibration import (
    CalibrationProvenance,
    FitParameters,
    GroundTruth,
    HeldOutPartMetrics,
    ScoreBranch,
    ScoreGroup,
    ScoreRow,
    SplitRole,
    TemplateCalibrationGroup,
    build_calibration_artifact,
    evaluate_heldout,
    fit_dual_thresholds,
    fit_template_thresholds,
    to_domain_model_evidence,
    to_domain_template_threshold_record,
    to_domain_threshold_record,
)
from zs32_inspection.domain.contracts import ThresholdRecord
from zs32_inspection.domain.evidence import EvidenceLevel
from zs32_inspection.models import ModelContractError, RawModelScore

HEX = "d" * 64
MODEL = "e" * 64
TEMPLATE_MODEL = "f" * 64


def _row(
    *,
    part: str,
    branch: ScoreBranch,
    score: float,
    label: GroundTruth,
    role: SplitRole,
    split_id: str,
    model_digest: str = MODEL,
) -> ScoreRow:
    return ScoreRow(
        dataset_release_id="dataset-v1",
        part_instance_id=part,
        capture_set_id=f"capture-{part}",
        hand="right",
        view="front",
        branch=branch,
        score=score,
        ground_truth=label,
        split_role=role,
        split_id=split_id,
        model_digest=model_digest,
        roi_version="roi-v1",
        roi_digest=HEX,
    )


def _parameters() -> FitParameters:
    return FitParameters(1.0, 0.995, 1, 1)


def test_heldout_test_scores_cannot_change_fitted_thresholds() -> None:
    group = ScoreGroup("right", "front", ScoreBranch.ANOMALY, MODEL, "roi-v1")
    calibration = [
        _row(
            part="cal-normal",
            branch=ScoreBranch.ANOMALY,
            score=0.1,
            label=GroundTruth.NORMAL,
            role=SplitRole.CALIBRATION,
            split_id="cal-v1",
        ),
        _row(
            part="cal-defect",
            branch=ScoreBranch.ANOMALY,
            score=0.8,
            label=GroundTruth.DEFECT,
            role=SplitRole.CALIBRATION,
            split_id="cal-v1",
        ),
    ]
    first = fit_dual_thresholds(
        [
            *calibration,
            _row(
                part="test-normal",
                branch=ScoreBranch.ANOMALY,
                score=-100,
                label=GroundTruth.NORMAL,
                role=SplitRole.TEST,
                split_id="test-v1",
            ),
        ],
        required_groups=[group],
        calibration_split_id="cal-v1",
        parameters=_parameters(),
    )
    second = fit_dual_thresholds(
        [
            *calibration,
            _row(
                part="test-normal",
                branch=ScoreBranch.ANOMALY,
                score=100,
                label=GroundTruth.NORMAL,
                role=SplitRole.TEST,
                split_id="test-v1",
            ),
        ],
        required_groups=[group],
        calibration_split_id="cal-v1",
        parameters=_parameters(),
    )
    assert first == second


def test_missing_defect_group_is_explicitly_not_deployable() -> None:
    group = ScoreGroup("right", "front", ScoreBranch.YOLO, MODEL, "roi-v1")
    records = fit_dual_thresholds(
        [
            _row(
                part="normal",
                branch=ScoreBranch.YOLO,
                score=0.1,
                label=GroundTruth.NORMAL,
                role=SplitRole.CALIBRATION,
                split_id="cal-v1",
            ),
        ],
        required_groups=[group],
        calibration_split_id="cal-v1",
        parameters=_parameters(),
    )
    assert records[0].status == "insufficient_data"
    assert records[0].low is None and records[0].high is None


def test_overlapping_normal_and_defect_scores_are_non_separable() -> None:
    group = ScoreGroup("right", "front", ScoreBranch.ANOMALY, MODEL, "roi-v1")
    records = fit_dual_thresholds(
        [
            _row(
                part="normal",
                branch=ScoreBranch.ANOMALY,
                score=0.8,
                label=GroundTruth.NORMAL,
                role=SplitRole.CALIBRATION,
                split_id="cal-v1",
            ),
            _row(
                part="defect",
                branch=ScoreBranch.ANOMALY,
                score=0.8,
                label=GroundTruth.DEFECT,
                role=SplitRole.CALIBRATION,
                split_id="cal-v1",
            ),
        ],
        required_groups=[group],
        calibration_split_id="cal-v1",
        parameters=_parameters(),
    )
    assert records[0].status == "non_separable"
    assert records[0].low is None and records[0].high is None


def test_raw_score_converts_only_once_to_canonical_domain_evidence() -> None:
    raw = RawModelScore(
        inspection_id="inspection-1",
        part_instance_id="part-1",
        capture_set_id="capture-1",
        hand="right",
        view="front",
        branch="anomaly",
        score=0.5,
        model_family="patchcore",
        model_digest=MODEL,
        roi_config_id="roi-v1",
        roi_digest=HEX,
        source_sha256=HEX,
        crop_sha256=HEX,
    )
    threshold = ThresholdRecord(
        hand="right",
        view_id="front",
        branch="anomaly",
        low=0.2,
        high=0.7,
        model_sha256=MODEL,
        roi_config_id="roi-v1",
        calibration_sha256=HEX,
    )
    evidence = to_domain_model_evidence(raw, threshold=threshold)
    assert evidence.level is EvidenceLevel.GRAY
    assert evidence.model_sha256 == MODEL


def test_raw_score_cannot_use_another_group_threshold() -> None:
    raw = RawModelScore(
        inspection_id="inspection-1",
        part_instance_id="part-1",
        capture_set_id="capture-1",
        hand="right",
        view="front",
        branch="yolo",
        score=0.5,
        model_family="yolo",
        model_digest=MODEL,
        roi_config_id="roi-v1",
        roi_digest=HEX,
        source_sha256=HEX,
        crop_sha256=HEX,
    )
    wrong = ThresholdRecord(
        hand="right",
        view_id="back",
        branch="yolo",
        low=0.2,
        high=0.7,
        model_sha256=MODEL,
        roi_config_id="roi-v1",
        calibration_sha256=HEX,
    )
    with pytest.raises(ModelContractError, match="does not match"):
        to_domain_model_evidence(raw, threshold=wrong)


def test_part_cannot_leak_between_calibration_and_test() -> None:
    group = ScoreGroup("right", "front", ScoreBranch.ANOMALY, MODEL, "roi-v1")
    with pytest.raises(ModelContractError, match="leaks"):
        fit_dual_thresholds(
            [
                _row(
                    part="same-part",
                    branch=ScoreBranch.ANOMALY,
                    score=0.1,
                    label=GroundTruth.NORMAL,
                    role=SplitRole.CALIBRATION,
                    split_id="cal-v1",
                ),
                _row(
                    part="same-part",
                    branch=ScoreBranch.ANOMALY,
                    score=0.2,
                    label=GroundTruth.NORMAL,
                    role=SplitRole.TEST,
                    split_id="test-v1",
                ),
            ],
            required_groups=[group],
            calibration_split_id="cal-v1",
            parameters=_parameters(),
        )


def test_heldout_metrics_reject_negative_or_contradictory_counts() -> None:
    with pytest.raises(ModelContractError, match="sum of mutually exclusive"):
        HeldOutPartMetrics(
            dataset_release_id="dataset-v1",
            test_split_id="test-v1",
            part_count=2,
            normal_part_count=1,
            defect_part_count=1,
            clear_count=2,
            gray_count=1,
            strong_count=0,
            template_ng_count=0,
            incomplete_count=0,
            defect_escape_count=1,
            normal_reject_count=0,
            defect_escape_rate=1.0,
            normal_reject_rate=0.0,
            review_rate=0.5,
            evaluation_complete=True,
        )


def test_heldout_metrics_reject_forged_complete_flag_and_rate() -> None:
    with pytest.raises(ModelContractError, match="defect_escape_rate"):
        HeldOutPartMetrics(
            dataset_release_id="dataset-v1",
            test_split_id="test-v1",
            part_count=2,
            normal_part_count=1,
            defect_part_count=1,
            clear_count=2,
            gray_count=0,
            strong_count=0,
            template_ng_count=0,
            incomplete_count=0,
            defect_escape_count=1,
            normal_reject_count=0,
            defect_escape_rate=0.0,
            normal_reject_rate=0.0,
            review_rate=0.0,
            evaluation_complete=True,
        )
    with pytest.raises(ModelContractError, match="evaluation_complete"):
        HeldOutPartMetrics(
            dataset_release_id="dataset-v1",
            test_split_id="test-v1",
            part_count=2,
            normal_part_count=1,
            defect_part_count=1,
            clear_count=1,
            gray_count=0,
            strong_count=0,
            template_ng_count=0,
            incomplete_count=1,
            defect_escape_count=0,
            normal_reject_count=0,
            defect_escape_rate=0.0,
            normal_reject_rate=0.0,
            review_rate=0.0,
            evaluation_complete=True,
        )


def test_complete_groups_build_hashed_valid_artifact() -> None:
    anomaly = ScoreGroup("right", "front", ScoreBranch.ANOMALY, MODEL, "roi-v1")
    yolo = ScoreGroup("right", "front", ScoreBranch.YOLO, MODEL, "roi-v1")
    template = TemplateCalibrationGroup("right", "front", TEMPLATE_MODEL, "roi-v1", HEX)
    rows = []
    for role, split_id, suffix in (
        (SplitRole.CALIBRATION, "cal-v1", "cal"),
        (SplitRole.TEST, "test-v1", "test"),
    ):
        for label, score, part_kind in (
            (GroundTruth.NORMAL, 0.1, "normal"),
            (GroundTruth.DEFECT, 0.9, "defect"),
        ):
            part = f"{suffix}-{part_kind}"
            rows.extend(
                [
                    _row(
                        part=part,
                        branch=ScoreBranch.ANOMALY,
                        score=score,
                        label=label,
                        role=role,
                        split_id=split_id,
                    ),
                    _row(
                        part=part,
                        branch=ScoreBranch.YOLO,
                        score=score,
                        label=label,
                        role=role,
                        split_id=split_id,
                    ),
                    _row(
                        part=part,
                        branch=ScoreBranch.TEMPLATE,
                        score=score,
                        label=label,
                        role=role,
                        split_id=split_id,
                        model_digest=TEMPLATE_MODEL,
                    ),
                ],
            )
    dual = fit_dual_thresholds(
        rows,
        required_groups=[anomaly, yolo],
        calibration_split_id="cal-v1",
        parameters=_parameters(),
    )
    template_thresholds = fit_template_thresholds(
        rows,
        required_groups=[template],
        calibration_split_id="cal-v1",
        parameters=_parameters(),
    )
    heldout = evaluate_heldout(
        rows,
        dual_thresholds=dual,
        template_thresholds=template_thresholds,
        required_dual_groups=[anomaly, yolo],
        required_template_groups=[template],
        test_split_id="test-v1",
    )
    artifact = build_calibration_artifact(
        provenance=CalibrationProvenance(
            recipe_digest=HEX,
            profile_digest=HEX,
            topology_digest=HEX,
            roi_digest=HEX,
            dataset_release_id="dataset-v1",
            dataset_manifest_digest=HEX,
            calibration_split_id="cal-v1",
            test_split_id="test-v1",
            model_digests=(MODEL, TEMPLATE_MODEL),
        ),
        parameters=_parameters(),
        required_dual_groups=[anomaly, yolo],
        required_template_groups=[template],
        dual_thresholds=dual,
        template_thresholds=template_thresholds,
        heldout_metrics=heldout,
    )
    assert artifact.calibration_valid is True
    assert len(artifact.artifact_sha256) == 64
    assert artifact.to_dict()["heldout_metrics"]["part_count"] == 2
    assert to_domain_threshold_record(dual[0], artifact.artifact_sha256).low < to_domain_threshold_record(
        dual[0], artifact.artifact_sha256
    ).high
    assert (
        to_domain_template_threshold_record(template_thresholds[0], artifact.artifact_sha256).template_sha256
        == TEMPLATE_MODEL
    )
