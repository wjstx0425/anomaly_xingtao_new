# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: EM102, TC003, TRY003

"""Tests for the quick ZS32 six-view YOLO dataset builder."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np

VIEWS = ("front", "front_left", "front_right", "back", "back_left", "back_right")


def _load_module() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "capture_data" / "prepare_zs32_yolo_dataset.py"
    spec = importlib.util.spec_from_file_location("prepare_zs32_yolo_dataset_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_image(path: Path, *, offset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(6, dtype=np.uint8) + offset
    assert cv2.imwrite(str(path), image)


def test_build_dataset_excludes_group_and_writes_empty_and_mirrored_negatives(tmp_path: Path) -> None:
    """The quick builder should produce a complete trainable dataset without split leakage."""
    module = _load_module()
    repo_root = tmp_path
    dataset_root = repo_root / "dataset"
    labeling_root = dataset_root / "zs32_yolo_labeling"
    export_root = labeling_root / "label_export"
    rows: list[dict[str, str]] = []

    for group_id in ("group001", "group027"):
        for index, view in enumerate(VIEWS):
            name = f"left_{view}_defect_less_zs32_left_less_{group_id}_000001_fused.png"
            session_id = "20260711_181850_552955"
            source = dataset_root / "left" / view / "defect" / "less" / session_id / "images" / name
            staged = labeling_root / "images" / "left" / view / "less" / name
            _write_image(source, offset=index)
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.hardlink_to(source)
            rows.append(
                {
                    "labeling_path": str(staged.relative_to(repo_root)),
                    "source_path": str(source.relative_to(repo_root)),
                    "hand": "left",
                    "view": view,
                    "defect_type": "less",
                    "session_id": session_id,
                    "group_id": group_id,
                    "sample_id": f"left/{session_id}/less/{group_id}",
                },
            )

    labeling_root.mkdir(parents=True, exist_ok=True)
    with (labeling_root / "labeling_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (export_root / "labels").mkdir(parents=True)
    first_stem = Path(rows[0]["source_path"]).stem
    (export_root / "labels" / f"{first_stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    for index, view in enumerate(VIEWS):
        name = f"right_{view}_normal_zs32_right_normal_group001_000001_fused.png"
        normal = dataset_root / "right" / view / "normal" / "normal_session" / "images" / name
        _write_image(normal, offset=20 + index)

    output_root = dataset_root / "zs32_six_view_yolo"
    summary = module.build_zs32_yolo_dataset(
        repo_root=repo_root,
        dataset_root=dataset_root,
        labeling_root=labeling_root,
        label_export_root=export_root,
        output_root=output_root,
        val_ratio=0.0,
        test_ratio=0.0,
        seed=7,
    )

    assert summary == {
        "defect_images": 6,
        "annotated_defect_images": 1,
        "empty_defect_images": 5,
        "normal_real_images": 6,
        "normal_mirror_images": 6,
        "total_images": 18,
        "groups": 2,
    }
    images = sorted((output_root / "images" / "train").glob("*.png"))
    labels = sorted((output_root / "labels" / "train").glob("*.txt"))
    assert len(images) == len(labels) == 18
    assert sum(bool(path.read_text(encoding="utf-8").strip()) for path in labels) == 1
    assert not any("group027" in path.name for path in images)
    assert (output_root / "data.yaml").read_text(encoding="utf-8").endswith("  0: defect\n")

    with (output_root / "split_manifest.csv").open(newline="", encoding="utf-8") as file:
        exported = list(csv.DictReader(file))
    assert {row["split"] for row in exported} == {"train"}
    assert all(
        len({item["split"] for item in exported if item["sample_id"] == sample_id}) == 1
        for sample_id in {row["sample_id"] for row in exported}
    )
    mirrored = next(row for row in exported if row["kind"] == "normal_mirror" and row["source_view"] == "front_left")
    assert mirrored["view"] == "front_right"
    source_image = cv2.imread(str(repo_root / mirrored["source_path"]))
    output_image = cv2.imread(str(repo_root / mirrored["output_image"]))
    assert np.array_equal(output_image, cv2.flip(source_image, 1))
