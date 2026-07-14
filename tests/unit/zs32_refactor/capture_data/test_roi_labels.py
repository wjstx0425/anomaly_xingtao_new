"""Audit-contract tests for the single authoritative ROI transformation."""

from __future__ import annotations

import sys

import pytest

from zs32_inspection.data.roi import LabelDocument, migrate_yolo_labels
from zs32_inspection.domain.contracts import RoiBox


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="authoritative runtime is Linux only")


def test_bbox_inside_clip_and_outside_have_distinct_audit_rows() -> None:
    labels = LabelDocument(
        "annotated",
        "0 0.5 0.5 0.1 0.1\n0 0.3 0.5 0.3 0.1\n0 0.05 0.05 0.05 0.05\n",
        "labels/source.txt",
    )
    migrated = migrate_yolo_labels(
        labels,
        source_width=1000,
        source_height=1000,
        roi=RoiBox(200, 200, 800, 800),
    )
    assert migrated.audit.source_box_count == 3
    assert migrated.audit.crop_box_count == 2
    assert migrated.audit.clipped_box_count == 1
    assert migrated.audit.outside_roi_box_count == 1
    assert migrated.audit.dropped_box_count == 1
    assert migrated.audit.boxes[0].dropped_reason is None
    assert migrated.audit.boxes[1].clipped is True
    assert migrated.audit.boxes[2].dropped_reason == "outside_roi"


def test_missing_annotation_is_not_silently_treated_as_empty() -> None:
    document = LabelDocument("missing", None, "labels/missing.txt")
    with pytest.raises(ValueError, match="missing is not confirmed empty"):
        migrate_yolo_labels(
            document,
            source_width=1000,
            source_height=1000,
            roi=RoiBox(0, 0, 1000, 1000),
        )


def test_confirmed_empty_has_before_and_after_hashes() -> None:
    migrated = migrate_yolo_labels(
        LabelDocument("confirmed_empty", "", "labels/normal.txt"),
        source_width=1000,
        source_height=1000,
        roi=RoiBox(100, 100, 900, 900),
    )
    assert migrated.text == ""
    assert migrated.audit.source_label_sha256 == migrated.audit.crop_label_sha256
    assert migrated.audit.source_box_count == 0
