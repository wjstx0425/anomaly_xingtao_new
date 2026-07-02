# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for C789 stress-normal group splitting helpers."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np


def load_prepare_stress_splits_module() -> ModuleType:
    """Load the stress split script from its file path."""
    script_path = Path(__file__).resolve().parents[3] / "capture_data" / "prepare_stress_splits.py"
    spec = importlib.util.spec_from_file_location("capture_data_prepare_stress_splits", script_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load stress split script from {script_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_image(path: Path, value: int) -> None:
    """Write a deterministic image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((8, 12, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _write_stress_crop(root: Path, label: str, group: int, slot: int) -> Path:
    """Write one stress crop in the existing nested Folder layout."""
    sample = f"part001_20260618_152639_slot{slot:02d}"
    name = f"left_top_normal_part001_g{group:03d}_000000_slot{slot:02d}.png"
    path = root / "left" / "top" / label / sample / "images" / name
    _write_image(path, 100 + group + slot)
    return path


def test_group_key_from_filename_keeps_slots_together() -> None:
    """Crops from different slots of the same source group should share one split key."""
    splits = load_prepare_stress_splits_module()

    first = Path("left_top_normal_part001_g003_000000_slot01.png")
    second = Path("left_top_normal_part001_g003_000000_slot06.png")
    third = Path("left_top_normal_part001_g004_000000_slot01.png")

    assert splits.group_key_from_path(first) == splits.group_key_from_path(second)
    assert splits.group_key_from_path(first) != splits.group_key_from_path(third)


def test_split_stress_normal_dataset_preserves_group_boundaries(tmp_path: Path) -> None:
    """The train/locked split should never put slots from one group in both outputs."""
    splits = load_prepare_stress_splits_module()
    input_root = tmp_path / "stress_parts"
    output_root = tmp_path / "stress_split"
    for group in (1, 2, 3, 4):
        for slot in (1, 2):
            _write_stress_crop(input_root, "normal", group, slot)

    manifest_path = splits.split_stress_normal_dataset(
        input_root=input_root,
        output_root=output_root,
        hand="left",
        position="top",
        train_ratio=0.5,
        seed=0,
        link_mode="copy",
        overwrite=True,
    )

    with manifest_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    split_by_group: dict[str, set[str]] = {}
    for row in rows:
        split_by_group.setdefault(row["group_key"], set()).add(row["split"])

    assert len(rows) == 8
    assert set(split_by_group) == {
        "part001_20260618_152639_g001",
        "part001_20260618_152639_g002",
        "part001_20260618_152639_g003",
        "part001_20260618_152639_g004",
    }
    assert all(len(values) == 1 for values in split_by_group.values())
    assert len(list((output_root / "train" / "left" / "top" / "normal").glob("*/images/*.png"))) == 4
    assert len(list((output_root / "locked" / "left" / "top" / "normal").glob("*/images/*.png"))) == 4
