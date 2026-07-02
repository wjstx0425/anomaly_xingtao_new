# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for building clean-plus-stress hardened Folder datasets."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np


def load_build_hardened_dataset_module() -> ModuleType:
    """Load the hardened dataset script from its file path."""
    script_path = Path(__file__).resolve().parents[3] / "capture_data" / "build_hardened_dataset.py"
    spec = importlib.util.spec_from_file_location("capture_data_build_hardened_dataset", script_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load hardened dataset script from {script_path}"
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


def test_build_hardened_dataset_keeps_locked_stress_out_of_training_root(tmp_path: Path) -> None:
    """Only stress train normal should be merged into the hardened training dataset."""
    builder = load_build_hardened_dataset_module()
    clean_root = tmp_path / "clean"
    stress_split_root = tmp_path / "stress_split"
    output_root = tmp_path / "hardened"
    _write_image(clean_root / "left" / "top" / "normal" / "clean_n" / "images" / "clean_n.png", 100)
    _write_image(clean_root / "left" / "top" / "normal_test" / "clean_t" / "images" / "clean_t.png", 110)
    _write_image(clean_root / "left" / "top" / "defect" / "defect_a" / "images" / "defect_a.png", 200)
    _write_image(stress_split_root / "train" / "left" / "top" / "normal" / "stress_a" / "images" / "stress_a.png", 120)
    _write_image(stress_split_root / "locked" / "left" / "top" / "normal" / "stress_b" / "images" / "stress_b.png", 130)

    manifest_path = builder.build_hardened_dataset(
        clean_root=clean_root,
        stress_split_root=stress_split_root,
        output_root=output_root,
        hand="left",
        position="top",
        link_mode="copy",
        overwrite=True,
    )

    with manifest_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    normal_images = sorted((output_root / "left" / "top" / "normal").glob("*/images/*.png"))
    normal_test_images = sorted((output_root / "left" / "top" / "normal_test").glob("*/images/*.png"))
    defect_images = sorted((output_root / "left" / "top" / "defect").glob("*/images/*.png"))

    assert [path.name for path in normal_images] == ["clean_n.png", "stress_a.png"]
    assert [path.name for path in normal_test_images] == ["clean_t.png"]
    assert [path.name for path in defect_images] == ["defect_a.png"]
    assert not list(output_root.glob("**/stress_b.png"))
    assert {row["source_split"] for row in rows} == {"clean", "stress_train"}
