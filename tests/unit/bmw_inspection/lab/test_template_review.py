# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the simple BMW Template candidate review package."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER
from bmw_inspection.lab.template_review import ReviewSource, build_template_review_package


def _write_source_data(tmp_path: Path, *, row_count: int = 42) -> tuple[Path, Path]:
    image_paths: list[Path] = []
    for index in range(row_count + 2):
        image = np.full((60, 80, 3), 20 + index, dtype=np.uint8)
        cv2.rectangle(image, (10 + index % 8, 8), (50, 45), (120 + index, 80, 200), 2)
        path = tmp_path / "images" / f"sample-{index:02d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(path), image)
        image_paths.append(path)

    manifest = tmp_path / "manifest.csv"
    fields = [
        "sample_id",
        "physical_part_id",
        "session_id",
        "view_id",
        "source_path",
        "source_class",
        "business_label",
        "split",
    ]
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for view in VIEW_ORDER:
            for index in range(row_count):
                writer.writerow(
                    {
                        "sample_id": f"sample-{index:02d}",
                        "physical_part_id": f"part-{index % 28:02d}",
                        "session_id": "session-a" if index < 28 else "session-b",
                        "view_id": view,
                        "source_path": image_paths[index],
                        "source_class": "normal",
                        "business_label": "OK",
                        "split": "train",
                    }
                )
            writer.writerow(
                {
                    "sample_id": "calibration-only",
                    "physical_part_id": "part-calibration",
                    "session_id": "session-c",
                    "view_id": view,
                    "source_path": image_paths[-2],
                    "source_class": "normal",
                    "business_label": "OK",
                    "split": "calibration",
                }
            )
            writer.writerow(
                {
                    "sample_id": "defect-only",
                    "physical_part_id": "part-defect",
                    "session_id": "session-d",
                    "view_id": view,
                    "source_path": image_paths[-1],
                    "source_class": "others",
                    "business_label": "NG",
                    "split": "train",
                }
            )

    roi_config = tmp_path / "roi.json"
    roi_config.write_text(
        json.dumps(
            {
                "image_width": 80,
                "image_height": 60,
                "part_rois": {view: [10, 5, 70, 55] for view in VIEW_ORDER},
            }
        ),
        encoding="utf-8",
    )
    return manifest, roi_config


def test_build_review_package_selects_40_per_view_without_using_non_train_rows(tmp_path: Path) -> None:
    manifest, roi_config = _write_source_data(tmp_path)
    output = tmp_path / "review"

    result = build_template_review_package(
        [ReviewSource(hand="left", manifest=manifest, roi_config=roi_config)],
        output,
        candidate_count=40,
    )

    assert result == output.resolve()
    rows = list(csv.DictReader((output / "candidate_manifest.csv").open(encoding="utf-8")))
    assert len(rows) == 8 * 40
    assert {row["view_id"] for row in rows} == set(VIEW_ORDER)
    assert all(row["split"] == "train" for row in rows)
    assert all(row["source_class"] == "normal" for row in rows)
    assert all(row["business_label"] == "OK" for row in rows)
    assert not any(row["sample_id"] in {"calibration-only", "defect-only"} for row in rows)

    for view in VIEW_ORDER:
        view_rows = [row for row in rows if row["view_id"] == view]
        assert [int(row["candidate_index"]) for row in view_rows] == list(range(1, 41))
        assert len({row["physical_part_id"] for row in view_rows[:28]}) == 28
        candidate_files = sorted((output / "left" / view).glob("candidate_*.png"))
        assert len(candidate_files) == 40
        assert cv2.imread(str(candidate_files[0])).shape == (50, 60, 3)
        assert cv2.imread(str(output / "left" / view / "contact_sheet.jpg")) is not None


def test_review_selection_is_deterministic(tmp_path: Path) -> None:
    manifest, roi_config = _write_source_data(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    source = ReviewSource(hand="right", manifest=manifest, roi_config=roi_config)

    build_template_review_package([source], first, candidate_count=40)
    build_template_review_package([source], second, candidate_count=40)

    def identities(root: Path) -> list[tuple[str, str, str, str]]:
        rows = csv.DictReader((root / "candidate_manifest.csv").open(encoding="utf-8"))
        return [(row["view_id"], row["session_id"], row["sample_id"], row["physical_part_id"]) for row in rows]

    assert identities(first) == identities(second)


def test_review_package_rejects_existing_output_and_insufficient_candidates(tmp_path: Path) -> None:
    manifest, roi_config = _write_source_data(tmp_path, row_count=39)
    output = tmp_path / "review"
    source = ReviewSource(hand="left", manifest=manifest, roi_config=roi_config)

    try:
        build_template_review_package([source], output, candidate_count=40)
    except ValueError as error:
        assert "needs 40" in str(error)
    else:
        raise AssertionError("insufficient candidates must fail")

    output.mkdir()
    try:
        build_template_review_package([source], output, candidate_count=39)
    except FileExistsError:
        pass
    else:
        raise AssertionError("an existing output root must not be overwritten")

