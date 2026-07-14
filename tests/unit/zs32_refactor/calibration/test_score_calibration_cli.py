# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Linux test definitions for frozen-model calibration/test scoring."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from zs32_inspection.cli._deployment_assets import parse_template_assets
from zs32_inspection.cli.score_calibration import _score_all, _selected_rows
from zs32_inspection.cli.calibrate import _validate_score_target_replay
from tests.unit.zs32_refactor.execution_fixtures import execution_receipt_mapping
from zs32_inspection.data.manifests import CanonicalSampleRow
from zs32_inspection.data.calibration_targets import (
    CalibrationTargetApproval,
    CalibrationTargetRecord,
    CalibrationTargetsSnapshot,
)
from zs32_inspection.models import ModelSlot, RawModelScore
from zs32_inspection.template import TemplateScore

TEMPLATE_DIGEST = "a" * 64
REFERENCE_DIGEST = "b" * 64
ANOMALY_DIGEST = "c" * 64
YOLO_DIGEST = "d" * 64
ROI_DIGEST = "e" * 64
SOURCE_DIGEST = "f" * 64
DATASET_DIGEST = "1" * 64
RECIPE_DIGEST = "2" * 64
CANDIDATE_DIGEST = "3" * 64


def _row(
    tmp_path,
    *,
    capture: str,
    split: str,
    view: str = "v1",
    label: str = "normal",
) -> CanonicalSampleRow:
    relative = f"crops/{capture}-{view}.png"
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"{capture}-{view}".encode()
    path.write_bytes(content)
    return CanonicalSampleRow(
        dataset_release_id="dataset-v1",
        part_instance_id=f"part-{capture}",
        capture_set_id=capture,
        hand="right",
        view=view,
        source_path=f"raw/{capture}-{view}.png",
        source_sha256=SOURCE_DIGEST,
        crop_path=relative,
        crop_sha256=hashlib.sha256(content).hexdigest(),
        roi_version="roi-v1",
        roi_sha256=ROI_DIGEST,
        roi_x1=0,
        roi_y1=0,
        roi_x2=8,
        roi_y2=4,
        crop_width=8,
        crop_height=4,
        label=label,
        defect_type="surface" if label == "defect" else "",
        split=split,
        is_synthetic=False,
        annotation_state="confirmed_empty",
        source_label_sha256="4" * 64,
        crop_label_sha256="5" * 64,
        source_box_count=0,
        crop_box_count=0,
        clipped_box_count=0,
        outside_roi_box_count=0,
        dropped_box_count=0,
    )


class _TemplatePredictor:
    mutate_after_score = False

    def score_batch(self, samples, *, inspection_id):
        output = tuple(
            TemplateScore(
                inspection_id=inspection_id,
                part_instance_id=item.sample.part.part_instance_id,
                capture_set_id=item.sample.capture_set_id,
                hand=item.sample.part.hand.value,
                view=item.sample.view_id,
                risk_score=0.2,
                similarity=0.8,
                model_digest=TEMPLATE_DIGEST,
                roi_config_id=item.sample.roi_config_id,
                roi_digest=item.roi_digest,
                source_sha256=item.sample.source_sha256,
                crop_sha256=item.sample.crop_sha256,
                best_template_path="reference.png",
                best_template_sha256=REFERENCE_DIGEST,
                offset_xy=(0, 0),
            )
            for item in samples
        )
        if self.mutate_after_score:
            samples[0].crop_path.write_bytes(b"mutated-after-template")
        return output


class _ModelPredictor:
    def __init__(self, branch: str) -> None:
        self.branch = branch

    def predict_batch(self, samples, *, inspection_id):
        digest = ANOMALY_DIGEST if self.branch == "anomaly" else YOLO_DIGEST
        family = "patchcore" if self.branch == "anomaly" else "yolo"
        score = 0.3 if self.branch == "anomaly" else 0.0
        return tuple(
            RawModelScore(
                inspection_id=inspection_id,
                part_instance_id=item.sample.part.part_instance_id,
                capture_set_id=item.sample.capture_set_id,
                hand=item.sample.part.hand.value,
                view=item.sample.view_id,
                branch=self.branch,
                score=score,
                model_family=family,
                model_digest=digest,
                roi_config_id=item.sample.roi_config_id,
                roi_digest=item.roi_digest,
                source_sha256=item.sample.source_sha256,
                crop_sha256=item.sample.crop_sha256,
            )
            for item in samples
        )


def _assets():
    slot = ModelSlot("right", "v1")
    template = SimpleNamespace(
        model_digest=TEMPLATE_DIGEST,
        templates=(SimpleNamespace(sha256=REFERENCE_DIGEST),),
    )
    anomaly = SimpleNamespace(model_digest=ANOMALY_DIGEST)
    return slot, {slot: template}, {slot: anomaly}


def _targets(rows, *, overrides=None):
    override_values = overrides or {}
    records = {}
    for row in rows:
        for branch in ("template", "anomaly", "yolo"):
            target, reason = override_values.get(
                (row.capture_set_id, branch),
                (row.label, None),
            )
            record = CalibrationTargetRecord(
                capture_set_id=row.capture_set_id,
                part_instance_id=row.part_instance_id,
                hand=row.hand,
                view=row.view,
                branch=branch,
                part_ground_truth=row.label,
                target=target,
                reason=reason,
            )
            records[record.key] = record
    return records


def test_score_rows_are_interleaved_template_anomaly_yolo(tmp_path) -> None:
    rows = (
        _row(tmp_path, capture="train-1", split="train"),
        _row(tmp_path, capture="cal-1", split="calibration"),
        _row(tmp_path, capture="test-1", split="test"),
    )
    slot, templates, anomalies = _assets()
    selected = _selected_rows(rows, required_slots=(slot,), required_views=("v1",))
    scores, audit = _score_all(
        selected,
        dataset_root=tmp_path,
        score_run_id="score-v1",
        calibration_split_id="cal-v1",
        test_split_id="test-v1",
        template_predictor=_TemplatePredictor(),
        anomaly_predictor=_ModelPredictor("anomaly"),
        yolo_predictor=_ModelPredictor("yolo"),
        template_assets=templates,
        anomaly_artifacts=anomalies,
        yolo_model_digest=YOLO_DIGEST,
        anomaly_family="patchcore",
        calibration_targets=_targets(selected),
    )
    assert [row["branch"] for row in scores] == [
        "template", "anomaly", "yolo", "template", "anomaly", "yolo"
    ]
    assert [row["split_id"] for row in scores] == [
        "cal-v1", "cal-v1", "cal-v1", "test-v1", "test-v1", "test-v1"
    ]
    assert [row["sequence_index"] for row in audit] == list(range(6))
    expected_keys = {
        "dataset_release_id", "part_instance_id", "capture_set_id", "hand", "view",
        "branch", "score", "ground_truth", "split_role", "split_id", "model_digest",
        "roi_version", "roi_digest", "part_ground_truth",
    }
    assert all(set(row) == expected_keys for row in scores)


def test_defect_parts_require_branch_targets_but_are_not_rejected(tmp_path) -> None:
    rows = (
        _row(tmp_path, capture="cal-defect", split="calibration", label="defect"),
        _row(tmp_path, capture="test-normal", split="test"),
    )
    slot, templates, anomalies = _assets()
    selected = _selected_rows(rows, required_slots=(slot,), required_views=("v1",))
    targets = _targets(
        selected,
        overrides={
            ("cal-defect", "template"): ("normal", "defect is not template-visible"),
            ("cal-defect", "yolo"): ("exclude", "YOLO annotation is ambiguous"),
        },
    )
    scores, audit = _score_all(
        selected,
        dataset_root=tmp_path,
        score_run_id="score-v1",
        calibration_split_id="cal-v1",
        test_split_id="test-v1",
        template_predictor=_TemplatePredictor(),
        anomaly_predictor=_ModelPredictor("anomaly"),
        yolo_predictor=_ModelPredictor("yolo"),
        template_assets=templates,
        anomaly_artifacts=anomalies,
        yolo_model_digest=YOLO_DIGEST,
        anomaly_family="patchcore",
        calibration_targets=targets,
    )
    defect_scores = [row for row in scores if row["capture_set_id"] == "cal-defect"]
    assert [(row["branch"], row["ground_truth"]) for row in defect_scores] == [
        ("template", "normal"), ("anomaly", "defect")
    ]
    excluded = [row for row in audit if row["calibration_target"] == "exclude"]
    assert len(excluded) == 1
    assert excluded[0]["included_in_calibration"] is False
    assert excluded[0]["score_sequence_index"] is None


def test_train_row_cannot_cross_the_scoring_boundary(tmp_path) -> None:
    row = _row(tmp_path, capture="train-1", split="train")
    _slot, templates, anomalies = _assets()
    with pytest.raises(ValueError, match="train rows are forbidden"):
        _score_all(
            (row,),
            dataset_root=tmp_path,
            score_run_id="score-v1",
            calibration_split_id="cal-v1",
            test_split_id="test-v1",
            template_predictor=_TemplatePredictor(),
            anomaly_predictor=_ModelPredictor("anomaly"),
            yolo_predictor=_ModelPredictor("yolo"),
            template_assets=templates,
            anomaly_artifacts=anomalies,
            yolo_model_digest=YOLO_DIGEST,
            anomaly_family="patchcore",
            calibration_targets=_targets((row,)),
        )


def test_heldout_exclude_cannot_disappear_from_full_system_evaluation(tmp_path) -> None:
    rows = (
        _row(tmp_path, capture="cal-normal", split="calibration"),
        _row(tmp_path, capture="test-normal", split="test"),
    )
    _slot, templates, anomalies = _assets()
    targets = _targets(
        rows,
        overrides={
            ("test-normal", branch): ("exclude", "held-out visibility is ambiguous")
            for branch in ("template", "anomaly", "yolo")
        },
    )
    _scores, audit = _score_all(
        rows,
        dataset_root=tmp_path,
        score_run_id="score-v1",
        calibration_split_id="cal-v1",
        test_split_id="test-v1",
        template_predictor=_TemplatePredictor(),
        anomaly_predictor=_ModelPredictor("anomaly"),
        yolo_predictor=_ModelPredictor("yolo"),
        template_assets=templates,
        anomaly_artifacts=anomalies,
        yolo_model_digest=YOLO_DIGEST,
        anomaly_family="patchcore",
        calibration_targets=targets,
    )
    snapshot = CalibrationTargetsSnapshot(
        dataset_release_id="dataset-v1",
        topology_id="topology-v1",
        roi_version="roi-v1",
        approval=CalibrationTargetApproval(
            reviewed_by="quality-user-1",
            reviewed_at="2026-07-14T00:00:00+08:00",
            approved=True,
        ),
        targets=tuple(sorted(targets.values(), key=lambda item: item.key)),
    )
    with pytest.raises(ValueError, match="held-out test target cannot be exclude"):
        _validate_score_target_replay(
            audit=audit,
            targets=snapshot,
            canonical_rows=rows,
            calibration_split_id="cal-v1",
            test_split_id="test-v1",
        )


def test_crop_is_rehashed_after_all_three_predictors(tmp_path) -> None:
    row = _row(tmp_path, capture="cal-1", split="calibration")
    _slot, templates, anomalies = _assets()
    template = _TemplatePredictor()
    template.mutate_after_score = True
    with pytest.raises((ValueError, RuntimeError), match="SHA256 mismatch|hash mismatch"):
        _score_all(
            (row,),
            dataset_root=tmp_path,
            score_run_id="score-v1",
            calibration_split_id="cal-v1",
            test_split_id="test-v1",
            template_predictor=template,
            anomaly_predictor=_ModelPredictor("anomaly"),
            yolo_predictor=_ModelPredictor("yolo"),
            template_assets=templates,
            anomaly_artifacts=anomalies,
            yolo_model_digest=YOLO_DIGEST,
            anomaly_family="patchcore",
            calibration_targets=_targets((row,)),
        )


def test_template_descriptor_is_independent_of_calibration_assets(tmp_path) -> None:
    slot = ModelSlot("right", "v1")
    anomaly = SimpleNamespace(
        roi_version="roi-v1",
        roi_digest=ROI_DIGEST,
        train_split_id="train-v1",
    )
    candidate = SimpleNamespace(
        candidate_id="candidate-v1",
        digest=CANDIDATE_DIGEST,
        roi_version="roi-v1",
        roi_digest=ROI_DIGEST,
        dataset_release_id="dataset-v1",
        dataset_manifest_digest=DATASET_DIGEST,
        recipe_digest=RECIPE_DIGEST,
        anomaly_artifacts={slot: anomaly},
        validate_slots=lambda required: None,
    )
    payload = {
        "schema": "zs32.template_assets",
        "schema_version": 1,
        "candidate_id": "candidate-v1",
        "candidate_digest": CANDIDATE_DIGEST,
        "required_slots": ["right/v1"],
        "templates": [
            {
                "hand": "right",
                "view": "v1",
                "model": {
                    "role": "template_model",
                    "relative_path": "templates/right/v1/model.json",
                    "sha256": TEMPLATE_DIGEST,
                    "media_type": "application/json",
                },
                "templates": [
                    {
                        "role": "reference_template",
                        "relative_path": "templates/right/v1/reference.png",
                        "sha256": REFERENCE_DIGEST,
                        "media_type": "image/png",
                    }
                ],
                "model_digest": TEMPLATE_DIGEST,
                "template_version": "template-v1",
                "roi_version": "roi-v1",
                "roi_digest": ROI_DIGEST,
                "dataset_release_id": "dataset-v1",
                "dataset_manifest_digest": DATASET_DIGEST,
                "train_split_id": "train-v1",
                "recipe_digest": RECIPE_DIGEST,
                "framework_version": "opencv-test",
                "training_parameters": {"width": 256},
                "execution_receipt": execution_receipt_mapping("train_template"),
            }
        ],
    }
    required, parsed = parse_template_assets(payload, asset_root=tmp_path, candidate=candidate)
    assert required == (slot,)
    assert parsed[slot].model_digest == TEMPLATE_DIGEST
    assert "calibration" not in payload
    with pytest.raises(ValueError, match="unknown=.*calibration"):
        parse_template_assets(
            {**payload, "calibration": {}},
            asset_root=tmp_path,
            candidate=candidate,
        )
