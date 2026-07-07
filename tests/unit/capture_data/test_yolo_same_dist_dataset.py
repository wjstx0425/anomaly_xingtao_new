# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the same-distribution YOLO dataset builder."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np


def load_yolo_same_dist_dataset_module() -> ModuleType:
    """Load the same-distribution YOLO dataset builder script from its file path."""
    script_path = Path(__file__).resolve().parents[3] / "capture_data" / "yolo_same_dist_dataset.py"
    spec = importlib.util.spec_from_file_location("capture_data_yolo_same_dist_dataset", script_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load same-distribution YOLO dataset script from {script_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_image(path: Path, value: int) -> None:
    """Write a deterministic RGB image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((24, 32, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _write_label(path: Path, text: str) -> None:
    """Write a YOLO label file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_sample(root: Path, stem: str, *, pixel_value: int, label_text: str) -> None:
    """Write one flat YOLO sample."""
    _write_image(root / "images" / f"{stem}.png", pixel_value)
    _write_label(root / "labels" / f"{stem}.txt", label_text)


def test_build_parser_exposes_same_distribution_defaults() -> None:
    """CLI defaults should match the planned same-distribution dataset builder contract."""
    builder = load_yolo_same_dist_dataset_module()

    args = builder.build_parser().parse_args(["--input-root", "in", "--output-root", "out"])

    assert args.input_root == Path("in")
    assert args.output_root == Path("out")
    assert args.normal_source_root is None
    assert tuple(args.val_groups) == ("g002", "g008")
    assert args.seed == 0
    assert args.preview_limit == 80
    assert args.overwrite is False
    assert args.train_normal_limit == 600
    assert args.val_normal_limit == 150


def test_build_same_dist_yolo_dataset_writes_split_outputs_and_reports(tmp_path: Path) -> None:
    """A flat YOLO root should become a split-aware dataset with train-only augmentation and no leakage."""
    builder = load_yolo_same_dist_dataset_module()
    input_root = tmp_path / "input_flat"
    output_root = tmp_path / "output_yolo"
    normal_root = tmp_path / "normal_source"

    _write_sample(input_root, "c789_g001_slot01_defect_a", pixel_value=40, label_text="0 0.500000 0.500000 0.250000 0.250000\n")
    _write_sample(input_root, "c789_g002_slot02_defect_b", pixel_value=60, label_text="0 0.500000 0.500000 0.250000 0.250000\n")
    _write_sample(input_root, "c789_g003_slot03_defect_c", pixel_value=80, label_text="0 0.500000 0.500000 0.250000 0.250000\n")
    _write_sample(input_root, "c789_g008_slot04_defect_d", pixel_value=100, label_text="0 0.500000 0.500000 0.250000 0.250000\n")

    _write_image(normal_root / "images" / "train" / "normal_g101_slot11.png", 120)
    _write_label(normal_root / "labels" / "train" / "normal_g101_slot11.txt", "")
    _write_image(normal_root / "images" / "train" / "not_normal_g103_slot13.png", 130)
    _write_label(normal_root / "labels" / "train" / "not_normal_g103_slot13.txt", "0 0.5 0.5 0.1 0.1\n")
    _write_image(normal_root / "images" / "val" / "normal_g102_slot12.png", 140)
    _write_label(normal_root / "labels" / "val" / "normal_g102_slot12.txt", "")

    summary = builder.build_same_dist_yolo_dataset(
        input_root=input_root,
        output_root=output_root,
        normal_source_root=normal_root,
        overwrite=True,
        preview_limit=4,
        train_normal_limit=1,
        val_normal_limit=1,
    )

    assert summary["status"] == "ok"
    assert summary["train_images"] == 17
    assert summary["val_images"] == 3
    assert summary["train_real_images"] == 3
    assert summary["val_real_images"] == 3
    assert summary["leakage_rows"] == 2
    assert summary["leakage_flagged_rows"] == 0

    train_images = sorted(path.name for path in (output_root / "images" / "train").glob("*.png"))
    val_images = sorted(path.name for path in (output_root / "images" / "val").glob("*.png"))
    train_labels = sorted(path.name for path in (output_root / "labels" / "train").glob("*.txt"))
    val_labels = sorted(path.name for path in (output_root / "labels" / "val").glob("*.txt"))

    assert len(train_images) == 17
    assert len(val_images) == 3
    assert len(train_labels) == 17
    assert len(val_labels) == 3
    assert "c789_g002_slot02_defect_b.png" in val_images
    assert "c789_g008_slot04_defect_d.png" in val_images
    assert "normal_g102_slot12.png" in val_images
    assert any(name.endswith("__hflip.png") for name in train_images)
    assert any(name.endswith("__noise.png") for name in train_images)
    assert "c789_g001_slot01_defect_a.png" not in train_images
    assert "normal_g101_slot11.png" in train_images
    assert not any("not_normal_g103" in name for name in train_images)
    assert not any("normal_g101_slot11__" in name for name in train_images)

    expected_files = [
        output_root / "data.yaml",
        output_root / "split_manifest.csv",
        output_root / "augmentation_manifest.csv",
        output_root / "dataset_summary.csv",
        output_root / "leakage_report.csv",
    ]
    for path in expected_files:
        assert path.is_file(), f"missing expected output: {path}"
    assert len(list((output_root / "previews").glob("*.png"))) == 4

    with (output_root / "split_manifest.csv").open(newline="", encoding="utf-8") as file:
        split_rows = list(csv.DictReader(file))
    assert {row["split"] for row in split_rows} == {"train", "val"}
    assert {row["source_kind"] for row in split_rows} == {"defect_source", "normal_source"}
    assert {row["group_id"] for row in split_rows if row["split"] == "val"} == {"g002", "g008", "g102"}

    with (output_root / "augmentation_manifest.csv").open(newline="", encoding="utf-8") as file:
        augmentation_rows = list(csv.DictReader(file))
    assert len(augmentation_rows) == 16
    assert {row["split"] for row in augmentation_rows} == {"train"}
    assert {row["source_kind"] for row in augmentation_rows} == {"defect_source"}
    assert {row["variant"] for row in augmentation_rows} == {
        "orig",
        "hflip",
        "rot_m5",
        "rot_p5",
        "bright_gamma",
        "blur",
        "noise",
        "scale_translate",
    }

    with (output_root / "dataset_summary.csv").open(newline="", encoding="utf-8") as file:
        summary_rows = list(csv.DictReader(file))
    assert {row["split"] for row in summary_rows} == {"train", "val"}
    train_row = next(row for row in summary_rows if row["split"] == "train")
    val_row = next(row for row in summary_rows if row["split"] == "val")
    assert train_row["image_count"] == "17"
    assert train_row["real_image_count"] == "3"
    assert val_row["image_count"] == "3"
    assert val_row["real_image_count"] == "3"

    with (output_root / "leakage_report.csv").open(newline="", encoding="utf-8") as file:
        leakage_rows = list(csv.DictReader(file))
    assert [row["leakage_kind"] for row in leakage_rows] == ["group_id", "source_stem"]
    assert all(row["is_leakage"] == "false" for row in leakage_rows)

    data_yaml = (output_root / "data.yaml").read_text(encoding="utf-8")
    assert "train: images/train" in data_yaml
    assert "val: images/val" in data_yaml
    assert "0: defect" in data_yaml
