# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the single-part crop preparation helper."""

from __future__ import annotations

import csv
import importlib.util
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np


def load_prepare_part_crops_module() -> ModuleType:
    """Load the part crop script from its file path."""
    script_path = Path(__file__).resolve().parents[3] / "capture_data" / "prepare_part_crops.py"
    spec = importlib.util.spec_from_file_location("capture_data_prepare_part_crops", script_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load part crop script from {script_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_image(path: Path, value: int = 128) -> None:
    """Write a small deterministic test image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((10, 20, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_defect_slots_from_name_maps_row_col_suffix() -> None:
    """Defect filenames should select the matching row/column slot."""
    crops = load_prepare_part_crops_module()

    slots = crops.C789_LEFT_TOP_3X2.slots

    assert crops.defect_slots_from_name(Path("surface_3_2_2.png"), slots) == {"slot06"}
    assert crops.defect_slots_from_name(Path("lack_1_1.png"), slots) == {"slot01"}
    assert crops.defect_slots_from_name(Path("unknown.png"), slots) == set()


def test_defect_slots_from_name_maps_bottom_row_col_suffix() -> None:
    """Bottom-view C789 filenames should use row/column suffixes."""
    crops = load_prepare_part_crops_module()

    slots = crops.C789_LEFT_BOTTOM_3X2.slots

    assert crops.defect_slots_from_name(Path("surface_3_2_1.png"), slots) == {"slot06"}
    assert crops.defect_slots_from_name(Path("less_2_2.png"), slots) == {"slot04"}


def test_defect_slots_from_name_maps_one_column_suffix() -> None:
    """One-column presets should use the first trailing number as the slot index."""
    crops = load_prepare_part_crops_module()

    slots = crops.FX11_NO_HAND_TOP_6X1.slots

    assert crops.defect_slots_from_name(Path("crack_2.png"), slots) == {"slot02"}
    assert crops.defect_slots_from_name(Path("deform_6_2.png"), slots) == {"slot06"}
    assert crops.defect_slots_from_name(Path("surface.png"), slots) == set()


def test_mask_holes_median_fills_masked_region() -> None:
    """Median hole masking should replace the masked hole pixels."""
    crops = load_prepare_part_crops_module()
    crop = np.full((20, 20, 3), 200, dtype=np.uint8)
    crop[8:13, 8:13] = 0

    masked = crops.mask_holes(crop, [crops.EllipseMask(10, 10, 4, 4)], "median", 3)

    assert masked[10, 10].tolist() == [200, 200, 200]
    assert masked[0, 0].tolist() == [200, 200, 200]


def test_prepare_part_crops_writes_nested_folder_dataset(tmp_path: Path) -> None:
    """Normal images should produce all slots; defect images should produce mapped slots."""
    crops = load_prepare_part_crops_module()
    data_root = tmp_path / "dataset"
    output_root = tmp_path / "parts"
    _write_image(data_root / "left" / "top" / "normal" / "normal_a.png", 100)
    _write_image(data_root / "left" / "top" / "defect" / "surface_1_2.png", 180)
    slots = [
        crops.SlotSpec("slot01", 1, 1, crops.Box(0, 0, 10, 10)),
        crops.SlotSpec("slot02", 1, 2, crops.Box(10, 0, 20, 10)),
    ]
    args = Namespace(
        data_root=data_root,
        output_root=output_root,
        hand="left",
        position="top",
        output_hand=None,
        output_position=None,
        labels=["normal", "defect"],
        preset="c789_left_top_3x2",
        roi=crops.Box(0, 0, 20, 10),
        slot=slots,
        hole_mask=[],
        slot_hole_mask=[],
        hole_mask_method="none",
        inpaint_radius=3,
        defect_slot_mode="from-name",
        defect_slot_map=None,
        nondefect_slots_from_defect_images="skip",
        normal_split_mode="never",
        normal_test_ratio=0.25,
        normal_split_seed=0,
        preview_overlay=None,
        overwrite=True,
    )

    manifest_path = crops.prepare_part_crops(args)

    normal_images = sorted((output_root / "left" / "top" / "normal").glob("*/images/*.png"))
    defect_images = sorted((output_root / "left" / "top" / "defect").glob("*/images/*.png"))
    with manifest_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert [path.name for path in normal_images] == ["normal_a_slot01.png", "normal_a_slot02.png"]
    assert [path.name for path in defect_images] == ["surface_1_2_slot02.png"]
    assert len(rows) == 3
    assert [row["label"] for row in rows] == ["normal", "normal", "defect"]
    assert rows[-1]["slot"] == "slot02"
    assert rows[-1]["gt_label"] == "1"
    assert (output_root / "part_crop_counts.csv").is_file()


def test_prepare_part_crops_splits_normal_sources_into_normal_test(tmp_path: Path) -> None:
    """Normal sources should be split into train/test before per-slot cropping."""
    crops = load_prepare_part_crops_module()
    data_root = tmp_path / "dataset"
    output_root = tmp_path / "parts"
    for index in range(4):
        _write_image(data_root / "left" / "top" / "normal" / f"normal_{index}.png", 100 + index)
    slots = [
        crops.SlotSpec("slot01", 1, 1, crops.Box(0, 0, 10, 10)),
        crops.SlotSpec("slot02", 1, 2, crops.Box(10, 0, 20, 10)),
    ]
    args = Namespace(
        data_root=data_root,
        output_root=output_root,
        hand="left",
        position="top",
        output_hand=None,
        output_position=None,
        labels=["normal", "normal_test"],
        preset="c789_left_top_3x2",
        roi=crops.Box(0, 0, 20, 10),
        slot=slots,
        hole_mask=[],
        slot_hole_mask=[],
        hole_mask_method="none",
        inpaint_radius=3,
        defect_slot_mode="from-name",
        defect_slot_map=None,
        nondefect_slots_from_defect_images="skip",
        normal_split_mode="auto",
        normal_test_ratio=0.25,
        normal_split_seed=0,
        preview_overlay=None,
        overwrite=True,
    )

    manifest_path = crops.prepare_part_crops(args)

    with manifest_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    labels_by_source: dict[str, set[str]] = {}
    for row in rows:
        labels_by_source.setdefault(row["source_path"], set()).add(row["label"])

    assert len(rows) == 8
    assert sum(row["label"] == "normal" for row in rows) == 6
    assert sum(row["label"] == "normal_test" for row in rows) == 2
    assert all(len(labels) == 1 for labels in labels_by_source.values())
    assert len(list((output_root / "left" / "top" / "normal_test").glob("*/images/*.png"))) == 2


def test_assign_normal_output_labels_rejects_negative_normal_test_ratio(tmp_path: Path) -> None:
    """Normal split ratio should not accept negative values."""
    crops = load_prepare_part_crops_module()
    images = [
        ("normal", tmp_path / "normal_a.png"),
        ("normal", tmp_path / "normal_b.png"),
    ]

    try:
        crops.assign_normal_output_labels(images, "auto", -0.25, seed=0)
    except ValueError as error:
        assert "--normal-test-ratio" in str(error)
    else:
        raise AssertionError("Expected negative --normal-test-ratio to raise ValueError")


def test_prepare_part_crops_keeps_existing_normal_test_in_auto_mode(tmp_path: Path) -> None:
    """Auto mode should honor an existing normal_test directory instead of resplitting normal."""
    crops = load_prepare_part_crops_module()
    data_root = tmp_path / "dataset"
    output_root = tmp_path / "parts"
    _write_image(data_root / "left" / "top" / "normal" / "normal_a.png", 100)
    _write_image(data_root / "left" / "top" / "normal" / "normal_b.png", 110)
    _write_image(data_root / "left" / "top" / "normal_test" / "normal_test_a.png", 120)
    slots = [crops.SlotSpec("slot01", 1, 1, crops.Box(0, 0, 20, 10))]
    args = Namespace(
        data_root=data_root,
        output_root=output_root,
        hand="left",
        position="top",
        output_hand=None,
        output_position=None,
        labels=["normal", "normal_test"],
        preset="c789_left_top_3x2",
        roi=crops.Box(0, 0, 20, 10),
        slot=slots,
        hole_mask=[],
        slot_hole_mask=[],
        hole_mask_method="none",
        inpaint_radius=3,
        defect_slot_mode="from-name",
        defect_slot_map=None,
        nondefect_slots_from_defect_images="skip",
        normal_split_mode="auto",
        normal_test_ratio=0.25,
        normal_split_seed=0,
        preview_overlay=None,
        overwrite=True,
    )

    manifest_path = crops.prepare_part_crops(args)

    with manifest_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert [row["label"] for row in rows] == ["normal", "normal", "normal_test"]


def test_slot_hole_mask_parser_accepts_optional_angle() -> None:
    """Slot-specific hole masks should parse optional ellipse angles."""
    crops = load_prepare_part_crops_module()

    slot_name, ellipse = crops.parse_slot_ellipse("slot03:10,11,12,13,45")

    assert slot_name == "slot03"
    assert ellipse.as_text() == "10,11,12,13,45"
