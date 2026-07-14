"""Linux-only contract tests for human-approved branch calibration targets."""

from __future__ import annotations

import sys

import pytest

from zs32_inspection.data.calibration_targets import (
    CalibrationTargetApproval,
    CalibrationTargetRecord,
    CalibrationTargetsSnapshot,
    validate_calibration_targets,
)
from zs32_inspection.data.manifests import CanonicalSampleRow


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="authoritative runtime is Linux only")


def _row(label: str = "defect") -> CanonicalSampleRow:
    return CanonicalSampleRow(
        dataset_release_id="dataset-v4",
        part_instance_id="part-1",
        capture_set_id="capture-1",
        hand="right",
        view="front_left",
        source_path="raw/session/images/capture-1/front_left.png",
        source_sha256="1" * 64,
        crop_path="crops/right/front_left/capture-1.png",
        crop_sha256="2" * 64,
        roi_version="roi-v2",
        roi_sha256="3" * 64,
        roi_x1=0,
        roi_y1=0,
        roi_x2=10,
        roi_y2=10,
        crop_width=10,
        crop_height=10,
        label=label,
        defect_type="scratch" if label == "defect" else "",
        split="calibration",
        is_synthetic=False,
        annotation_state="annotated" if label == "defect" else "confirmed_empty",
        source_label_sha256="4" * 64,
        crop_label_sha256="5" * 64,
        source_box_count=1 if label == "defect" else 0,
        crop_box_count=1 if label == "defect" else 0,
        clipped_box_count=0,
        outside_roi_box_count=0,
        dropped_box_count=0,
    )


def _snapshot(row: CanonicalSampleRow) -> CalibrationTargetsSnapshot:
    values = {
        "anomaly": ("defect", None),
        "template": ("normal", "scratch is not represented by template matching"),
        "yolo": ("exclude", "YOLO visibility was ambiguous during human review"),
    }
    records = tuple(
        CalibrationTargetRecord(
            capture_set_id=row.capture_set_id,
            part_instance_id=row.part_instance_id,
            hand=row.hand,
            view=row.view,
            branch=branch,
            part_ground_truth=row.label,
            target=target,
            reason=reason,
        )
        for branch, (target, reason) in values.items()
    )
    return CalibrationTargetsSnapshot(
        dataset_release_id=row.dataset_release_id,
        topology_id="topology-v1",
        roi_version=row.roi_version,
        approval=CalibrationTargetApproval(
            reviewed_by="quality-user-1",
            reviewed_at="2026-07-14T00:00:00+08:00",
            approved=True,
        ),
        targets=records,
    )


def test_exact_three_branch_contract_accepts_branch_specific_truth() -> None:
    row = _row()
    snapshot = _snapshot(row)
    validate_calibration_targets(
        snapshot,
        (row,),
        dataset_release_id=row.dataset_release_id,
        topology_id="topology-v1",
        roi_version=row.roi_version,
    )
    assert len(snapshot.targets) == 3
    assert snapshot.by_key[
        ("capture-1", "part-1", "right", "front_left", "template")
    ].target.value == "normal"
    assert CalibrationTargetsSnapshot.from_mapping(snapshot.as_dict) == snapshot


def test_defect_to_normal_and_exclude_require_a_reason() -> None:
    with pytest.raises(ValueError, match="audited reason"):
        CalibrationTargetRecord(
            "capture-1", "part-1", "right", "front_left", "template",
            "defect", "normal", None,
        )
    with pytest.raises(ValueError, match="audited reason"):
        CalibrationTargetRecord(
            "capture-1", "part-1", "right", "front_left", "yolo",
            "defect", "exclude", None,
        )


def test_normal_part_cannot_be_relabeled_as_branch_defect() -> None:
    with pytest.raises(ValueError, match="normal part"):
        CalibrationTargetRecord(
            "capture-1", "part-1", "right", "front_left", "anomaly",
            "normal", "defect", None,
        )


def test_contract_fails_closed_on_missing_branch_or_part_truth_mismatch() -> None:
    row = _row()
    snapshot = _snapshot(row)
    incomplete = CalibrationTargetsSnapshot(
        dataset_release_id=snapshot.dataset_release_id,
        topology_id=snapshot.topology_id,
        roi_version=snapshot.roi_version,
        approval=snapshot.approval,
        targets=snapshot.targets[:-1],
    )
    with pytest.raises(ValueError, match="exactly cover"):
        validate_calibration_targets(
            incomplete,
            (row,),
            dataset_release_id=row.dataset_release_id,
            topology_id="topology-v1",
            roi_version=row.roi_version,
        )
