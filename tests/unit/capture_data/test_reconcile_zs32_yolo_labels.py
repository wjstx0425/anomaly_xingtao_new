# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for content-addressed ZS32 eight-view label reconciliation."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from capture_data.reconcile_zs32_yolo_labels import (
    ImageFingerprint,
    LabelCandidate,
    _label_studio_results_to_yolo,
    _select_defect_preannotation,
    build_balanced_defect_normal_yolo_dataset,
    build_label_studio_task,
    finalize_defect_yolo_export,
    finalize_reconciled_dataset,
    fingerprint_image,
    horizontal_flip_yolo,
    labels_equivalent,
    split_alias_groups,
)


def test_fingerprint_distinguishes_file_bytes_from_decoded_pixels(tmp_path: Path) -> None:
    """Two differently encoded PNGs with identical pixels should share the pixel digest."""
    image = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    assert cv2.imwrite(str(first), image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    assert cv2.imwrite(str(second), image, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    first_hash = fingerprint_image(first)
    second_hash = fingerprint_image(second)

    assert first_hash.file_sha256 != second_hash.file_sha256
    assert first_hash.pixel_sha256 == second_hash.pixel_sha256
    assert first_hash.perceptual_hash == second_hash.perceptual_hash
    assert first_hash.width == second_hash.width == 16
    assert first_hash.height == second_hash.height == 12


def test_fingerprint_records_horizontal_flip_digest(tmp_path: Path) -> None:
    """The flip digest should match the decoded digest of a real horizontal mirror."""
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    image[:, :3] = (10, 20, 200)
    source = tmp_path / "source.png"
    mirrored = tmp_path / "mirrored.png"
    assert cv2.imwrite(str(source), image)
    assert cv2.imwrite(str(mirrored), cv2.flip(image, 1))

    source_hash = fingerprint_image(source)
    mirrored_hash = fingerprint_image(mirrored)

    assert source_hash.flipped_pixel_sha256 == mirrored_hash.pixel_sha256


def test_labels_equivalent_ignores_line_order_and_small_rounding() -> None:
    """Conflict detection should compare canonical boxes rather than exact text bytes."""
    left = "0 0.25 0.30 0.10 0.20\n0 0.75 0.70 0.12 0.22\n"
    right = "0 0.75000001 0.70 0.12 0.22\n0 0.25 0.30 0.10 0.20\n"

    assert labels_equivalent(left, right)
    assert not labels_equivalent(left, "")


def test_horizontal_flip_yolo_transforms_centres_and_preserves_geometry() -> None:
    """A proven mirror relation should transform normalized boxes deterministically."""
    result = horizontal_flip_yolo("0 0.20 0.40 0.10 0.30\n", source="label.txt")

    assert result == "0 0.80000000 0.40000000 0.10000000 0.30000000\n"


def test_alias_groups_are_forced_into_one_split() -> None:
    """Renamed duplicate physical parts may never cross train, val, and test."""
    strata = {
        "normal/session-a/group001": "right:normal:none",
        "normal/session-b/group009": "right:normal:none",
        "normal/session-c/group003": "right:normal:none",
        "normal/session-d/group004": "right:normal:none",
    }

    assignments = split_alias_groups(
        strata,
        alias_pairs=(("normal/session-a/group001", "normal/session-b/group009"),),
        val_ratio=0.25,
        test_ratio=0.25,
        seed=42,
    )

    assert assignments["normal/session-a/group001"] == assignments["normal/session-b/group009"]
    assert set(assignments.values()) <= {"train", "val", "test"}


def test_label_studio_task_contains_roi_preannotation() -> None:
    """Trusted ROI labels should appear as percentage RectangleLabels predictions."""
    fingerprint = ImageFingerprint("a" * 64, "b" * 64, "c" * 64, "d" * 64, 200, 100)

    task = build_label_studio_task(
        task_id=7,
        image_url="/data/local-files/?d=images/front/example.png",
        label_text="0 0.50 0.25 0.20 0.10\n",
        fingerprint=fingerprint,
        metadata={"view": "front", "needs_human_annotation": False},
        model_version="sha256_exact",
    )

    assert json.loads(json.dumps(task)) == task
    result = task["predictions"][0]["result"][0]
    assert result["original_width"] == 200
    assert result["original_height"] == 100
    assert result["value"] == {
        "x": 40.0,
        "y": 20.0,
        "width": 20.0,
        "height": 10.0,
        "rotation": 0,
        "rectanglelabels": ["defect"],
    }


def test_completed_label_studio_json_converts_boxes_and_confirmed_empty() -> None:
    """Finalization should distinguish a completed empty task from missing work."""
    empty = {"id": 1, "annotations": [{"was_cancelled": False, "result": []}]}
    annotated = {
        "id": 2,
        "annotations": [
            {
                "was_cancelled": False,
                "result": [
                    {
                        "type": "rectanglelabels",
                        "value": {"x": 10, "y": 20, "width": 30, "height": 40},
                    },
                ],
            },
        ],
    }

    assert _label_studio_results_to_yolo(empty) == ""
    assert _label_studio_results_to_yolo(annotated) == "0 0.25000000 0.40000000 0.30000000 0.40000000\n"


def test_defect_preannotation_preserves_source_label_but_flags_duplicate_conflict() -> None:
    """Each of 840 source tasks keeps its own old box while conflicting twins require review."""
    first = LabelCandidate("0 0.2 0.2 0.1 0.1\n", "first.txt", "first.png", "old")
    second = LabelCandidate("0 0.8 0.8 0.1 0.1\n", "second.txt", "second.png", "old")

    selected, method, conflict, needs_human = _select_defect_preannotation((first,), (first, second))

    assert selected == first
    assert method == "source_label_studio_preannotation_conflict"
    assert conflict is True
    assert needs_human is True


def test_defect_preannotation_reuses_the_only_exact_duplicate_label() -> None:
    """An unlabeled renamed byte-identical defect may inherit one unambiguous old label."""
    candidate = LabelCandidate("0 0.2 0.2 0.1 0.1\n", "first.txt", "first.png", "old")

    selected, method, conflict, needs_human = _select_defect_preannotation((), (candidate,))

    assert selected == candidate
    assert method == "sha256_duplicate_label_reuse"
    assert conflict is False
    assert needs_human is False


def test_stage36_parser_exposes_prepare_and_finalize() -> None:
    """The numbered wrapper should separate reconciliation from post-review finalization."""
    script = Path(__file__).resolve().parents[3] / "pipeline" / "36_reconcile_zs32_yolo_labels.py"
    spec = importlib.util.spec_from_file_location("pipeline36_reconcile_zs32_yolo_labels", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    prepare = module.build_parser().parse_args(["prepare"])
    finalize = module.build_parser().parse_args(
        ["finalize", "--prepared-root", "prepared", "--label-studio-export-json", "export.json"],
    )

    assert prepare.command == "prepare"
    assert finalize.command == "finalize"
    assert finalize.prepared_root == Path("prepared")


def test_finalize_defect_yolo_export_requires_explicit_empty_confirmation(tmp_path: Path) -> None:
    """Missing exported labels become empty only after an explicit operator confirmation."""
    prepared = tmp_path / "prepared"
    export = tmp_path / "export"
    (prepared / "images/front").mkdir(parents=True)
    (export / "labels").mkdir(parents=True)
    (export / "classes.txt").write_text("defect\n", encoding="utf-8")
    rows = []
    session_split = {}
    for index, (session, split, positive) in enumerate(
        (("session-a", "train", True), ("session-b", "val", False), ("session-c", "test", False)),
        start=1,
    ):
        image = prepared / f"images/front/sample-{index}.png"
        assert cv2.imwrite(str(image), np.zeros((10, 10, 3), dtype=np.uint8))
        if positive:
            (export / f"labels/sample-{index}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        rows.append(
            {
                "original_image_path": str(image),
                "task_image_path": str(image),
                "view": "front",
                "session_id": session,
                "group_id": f"group{index:03d}",
                "sample_id": f"left/defect/deform/{session}/group{index:03d}",
                "split": "train",
                "match_method": "missing_annotation",
                "source_label_path": "",
                "has_preannotation": "false",
                "is_duplicate": "false",
                "is_conflict": "false",
                "conflict_cluster_id": "",
                "needs_human_annotation": "true",
                "pixel_sha256": str(index) * 64,
            },
        )
        session_split[session] = split
    with (prepared / "defect_mapping.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="explicit confirmation"):
        finalize_defect_yolo_export(
            prepared_root=prepared,
            yolo_export_root=export,
            output_root=tmp_path / "rejected",
            confirm_skipped_empty=False,
            session_split=session_split,
        )

    summary = finalize_defect_yolo_export(
        prepared_root=prepared,
        yolo_export_root=export,
        output_root=tmp_path / "final",
        confirm_skipped_empty=True,
        session_split=session_split,
    )

    assert summary["images"] == 3
    assert summary["positive_images"] == 1
    assert summary["confirmed_empty_images"] == 2
    assert (tmp_path / "final/labels/val/sample-2.txt").read_text(encoding="utf-8") == ""
    assert (tmp_path / "final/labels/test/sample-3.txt").read_text(encoding="utf-8") == ""


def test_balanced_dataset_excludes_unreviewed_defects_and_adds_normal_negatives(tmp_path: Path) -> None:
    """A clean rebuild must replace uncertain skipped defects with trusted normal negatives."""
    prepared = tmp_path / "prepared"
    export = tmp_path / "export"
    normal_root = tmp_path / "normal"
    (prepared / "images/front").mkdir(parents=True)
    (export / "labels").mkdir(parents=True)
    normal_root.mkdir()
    (export / "classes.txt").write_text("defect\n", encoding="utf-8")
    defect_rows = []
    normal_rows = []
    group_split = {}
    for index, split in enumerate(("train", "val", "test"), start=1):
        defect_image = prepared / f"images/front/defect-{index}.png"
        normal_image = normal_root / f"normal-{index}.png"
        assert cv2.imwrite(str(defect_image), np.zeros((10, 10, 3), dtype=np.uint8))
        assert cv2.imwrite(str(normal_image), np.zeros((10, 10, 3), dtype=np.uint8))
        (export / f"labels/defect-{index}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        defect_group = f"left/defect/deform/session-d-{index}/group001"
        normal_group = f"right/normal/none/session-n/group{index:03d}"
        group_split.update({defect_group: split, normal_group: split})
        defect_rows.append(
            {
                "original_image_path": str(defect_image),
                "task_image_path": str(defect_image),
                "view": "front",
                "session_id": f"session-d-{index}",
                "group_id": "group001",
                "sample_id": defect_group,
                "split": split,
                "match_method": "missing_annotation",
                "source_label_path": "",
                "has_preannotation": "false",
                "is_duplicate": "false",
                "is_conflict": "false",
                "conflict_cluster_id": "",
                "needs_human_annotation": "true",
                "pixel_sha256": str(index) * 64,
            },
        )
        normal_rows.append(
            {
                "sample_key": normal_image.stem,
                "sample_id": normal_group,
                "view": "front",
                "hand": "right",
                "session_id": "session-n",
                "group_id": f"group{index:03d}",
                "split": split,
                "match_method": "normal_directory_confirmed_empty",
                "is_conflict": "False",
                "needs_human_annotation": "False",
                "image_path": str(normal_image),
                "label_path": str(normal_image.with_suffix(".txt")),
                "box_count": "0",
                "source_label_path": "",
                "crop_pixel_sha256": f"n{index}" * 32,
            },
        )
        normal_image.with_suffix(".txt").write_text("", encoding="utf-8")
    uncertain = prepared / "images/front/uncertain.png"
    assert cv2.imwrite(str(uncertain), np.zeros((10, 10, 3), dtype=np.uint8))
    defect_rows.append({**defect_rows[0], "task_image_path": str(uncertain), "pixel_sha256": "u" * 64})
    with (prepared / "defect_mapping.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(defect_rows[0]))
        writer.writeheader()
        writer.writerows(defect_rows)
    normal_manifest = tmp_path / "normal_manifest.csv"
    with normal_manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(normal_rows[0]))
        writer.writeheader()
        writer.writerows(normal_rows)

    summary = build_balanced_defect_normal_yolo_dataset(
        defect_prepared_root=prepared,
        yolo_export_root=export,
        normal_manifest=normal_manifest,
        output_root=tmp_path / "final",
        group_split=group_split,
        require_full_coverage=False,
    )

    assert summary["images"] == 6
    assert summary["positive_images"] == 3
    assert summary["trusted_normal_images"] == 3
    assert summary["excluded_uncertain_defect_images"] == 1
    assert len(list((tmp_path / "final").glob("labels/*/*.txt"))) == 6


def test_finalize_skips_unreviewed_tasks_that_already_have_reused_labels(tmp_path: Path) -> None:
    """An all-task export may contain reused predictions without human annotations."""
    prepared = tmp_path / "prepared"
    image = prepared / "yolo_draft/images/train/reused.png"
    label = prepared / "yolo_draft/labels/train/reused.txt"
    image.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    assert cv2.imwrite(str(image), np.zeros((10, 10, 3), dtype=np.uint8))
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    review_image = prepared / "yolo_draft/images/train/review.png"
    review_label = prepared / "yolo_draft/labels/train/review.txt"
    assert cv2.imwrite(str(review_image), np.zeros((10, 10, 3), dtype=np.uint8))
    review_label.write_text("", encoding="utf-8")
    rows = [
        {
            "sample_key": "reused",
            "split": "train",
            "needs_human_annotation": "False",
            "image_path": str(image),
            "label_path": str(label),
        },
        {
            "sample_key": "review",
            "split": "train",
            "needs_human_annotation": "True",
            "image_path": str(review_image),
            "label_path": str(review_label),
        },
    ]
    with (prepared / "annotation_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (prepared / "mapping.csv").write_text("source,target\n", encoding="utf-8")
    (prepared / "roi_config.json").write_text("{}\n", encoding="utf-8")
    export = tmp_path / "export.json"
    export.write_text(
        json.dumps(
            [
                {"data": {"sample_key": "reused"}, "predictions": [{}], "annotations": []},
                {"data": {"sample_key": "review"}, "annotations": [{"result": [], "was_cancelled": False}]},
            ],
        ),
        encoding="utf-8",
    )

    summary = finalize_reconciled_dataset(
        prepared_root=prepared,
        label_studio_json=export,
        output_root=tmp_path / "final",
    )

    assert summary == {"images": 2, "labels": 2, "boxes": 1, "reviewed": 1}
