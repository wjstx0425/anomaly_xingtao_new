# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the manual part crop helper."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_manual_part_crop_module() -> ModuleType:
    """Load the manual crop script from its file path."""
    script_path = Path(__file__).resolve().parents[3] / "capture_data" / "manual_part_crop.py"
    spec = importlib.util.spec_from_file_location("capture_data_manual_part_crop", script_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load manual crop script from {script_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_slots_orders_vertical_boxes() -> None:
    """Vertical order should map top-to-bottom boxes to slot rows."""
    manual = load_manual_part_crop_module()
    boxes = [
        manual.Box(10, 100, 20, 120),
        manual.Box(10, 10, 20, 30),
        manual.Box(10, 60, 20, 80),
    ]

    slots = manual.build_slots(boxes, "vertical", columns=1, prefix="slot")

    assert [slot.name for slot in slots] == ["slot01", "slot02", "slot03"]
    assert [(slot.row, slot.col) for slot in slots] == [(1, 1), (2, 1), (3, 1)]
    assert [slot.box.y1 for slot in slots] == [10, 60, 100]


def test_build_slots_orders_row_major_boxes() -> None:
    """Row-major order should assign rows and columns using the column count."""
    manual = load_manual_part_crop_module()
    boxes = [
        manual.Box(80, 80, 100, 100),
        manual.Box(10, 10, 30, 30),
        manual.Box(80, 10, 100, 30),
        manual.Box(10, 80, 30, 100),
    ]

    slots = manual.build_slots(boxes, "row-major", columns=2, prefix="slot")

    assert [(slot.row, slot.col) for slot in slots] == [(1, 1), (1, 2), (2, 1), (2, 2)]
    assert [(slot.box.x1, slot.box.y1) for slot in slots] == [(10, 10), (80, 10), (10, 80), (80, 80)]


def test_save_slots_csv_writes_reusable_slot_args(tmp_path: Path) -> None:
    """Slot CSV should include ready-to-reuse --slot arguments."""
    manual = load_manual_part_crop_module()
    slots = [
        manual.SlotSpec("slot01", 1, 1, manual.Box(1, 2, 3, 4)),
        manual.SlotSpec("slot02", 2, 1, manual.Box(5, 6, 7, 8)),
    ]

    csv_path = tmp_path / "slots.csv"
    manual.save_slots_csv(csv_path, slots)

    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows[0]["slot_arg"] == "slot01:1,1,1,2,3,4"
    assert rows[1]["slot_arg"] == "slot02:2,1,5,6,7,8"


def test_parser_accepts_fixed_slot_arguments(tmp_path: Path) -> None:
    """Manual crop mode should accept fixed slot boxes without GUI selection."""
    manual = load_manual_part_crop_module()

    args = manual.build_parser().parse_args(
        [
            "--data-root",
            str(tmp_path / "dataset"),
            "--output-root",
            str(tmp_path / "parts"),
            "--slot",
            "slot01:1,1,832,43,3200,446",
            "--slot",
            "slot02:2,1,835,440,3209,853",
        ],
    )

    assert [slot.name for slot in args.slot] == ["slot01", "slot02"]
    assert [(slot.row, slot.col) for slot in args.slot] == [(1, 1), (2, 1)]
    assert args.slot[0].box.as_text() == "832,43,3200,446"
