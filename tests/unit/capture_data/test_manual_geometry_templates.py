# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for manual C789 geometry template helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np


def _load_module(name: str, relative_path: str) -> ModuleType:
    """Load a repository script from a file path."""
    script_path = Path(__file__).resolve().parents[3] / relative_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load {script_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_mask(path: Path, mask: np.ndarray) -> None:
    """Write a binary mask image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def test_build_manual_template_applies_ignore_to_all_masks(tmp_path: Path) -> None:
    """Manual expected/allowed/watch masks should exclude ignored pixels."""
    builder = _load_module("manual_geometry_builder", "capture_data/build_manual_geometry_templates.py")
    geometry = _load_module("manual_geometry_core", "capture_data/geometry_shape.py")

    expected = np.zeros((20, 30), dtype=bool)
    expected[5:15, 8:22] = True
    allowed = expected.copy()
    allowed[4:16, 7:23] = True
    watch = np.zeros_like(expected)
    watch[4:16, 7:23] = True
    ignore = np.zeros_like(expected)
    ignore[9:12, 12:16] = True
    mask_dir = tmp_path / "manual_masks"
    _write_mask(mask_dir / "slot01_expected.png", expected)
    _write_mask(mask_dir / "slot01_allowed.png", allowed)
    _write_mask(mask_dir / "slot01_watch_edge.png", watch)
    _write_mask(mask_dir / "slot01_ignore.png", ignore)

    template = builder.build_manual_template(mask_dir, "slot01")

    assert template.slot == "slot01"
    assert template.source_count == 1
    assert not bool(template.expected_body[10, 14])
    assert not bool(template.allowed_body[10, 14])
    assert not bool(template.edge_band[10, 14])
    path = geometry.save_template(template, tmp_path / "templates")
    assert path.is_file()


def test_export_review_pack_writes_editable_manual_masks(tmp_path: Path) -> None:
    """Review pack export should create one set of editable mask PNGs per slot."""
    exporter = _load_module("manual_geometry_exporter", "capture_data/export_geometry_review_pack.py")
    image = np.zeros((30, 40, 3), dtype=np.uint8)
    image[6:24, 8:32] = 120
    normal_image = tmp_path / "normal" / "sample_slot01" / "images" / "part_slot01.png"
    normal_image.parent.mkdir(parents=True)
    assert cv2.imwrite(str(normal_image), image)

    manifest_path = exporter.export_review_pack(
        normal_root=tmp_path / "normal",
        output_dir=tmp_path / "review",
        preset="c789_left_top_3x2",
        slots=["slot01"],
        template_dir=None,
        stress_root=None,
        defect_root=None,
        samples_per_split=1,
    )

    assert manifest_path.is_file()
    for suffix in ("expected", "allowed", "ignore", "watch_edge"):
        assert (tmp_path / "review" / "manual_masks" / f"slot01_{suffix}.png").is_file()
    assert (tmp_path / "review" / "sheets" / "slot01_review_sheet.png").is_file()
