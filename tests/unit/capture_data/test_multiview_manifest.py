# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for multi-view inspection manifests."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_manifest_module() -> ModuleType:
    """Load the multi-view manifest module from its file path."""
    script_path = Path(__file__).resolve().parents[3] / "capture_data" / "multiview_manifest.py"
    spec = importlib.util.spec_from_file_location("capture_data_multiview_manifest", script_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load multiview manifest module from {script_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_explicit_manifest_csv_groups_images_by_part(tmp_path: Path) -> None:
    """Explicit CSV manifests should group part_id rows without relying on filenames."""
    manifest = load_manifest_module()
    csv_path = tmp_path / "manifest.csv"
    top_path = tmp_path / "images" / "custom_top.png"
    bottom_path = tmp_path / "images" / "custom_bottom.png"
    top_path.parent.mkdir(parents=True)
    top_path.write_bytes(b"top")
    bottom_path.write_bytes(b"bottom")
    csv_path.write_text(
        "\n".join(
            [
                "part_id,side,view,image_path,label,defect_type,slot_id,group_id,notes",
                f"part001,top,uniform,{top_path},normal,,,g1,",
                f"part001,bottom,uniform,{bottom_path},normal,,,g1,",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    records = manifest.load_multiview_manifest(tmp_path, {}, manifest_csv=csv_path)

    assert len(records) == 1
    assert records[0].part_id == "part001"
    assert [(image.side, image.view) for image in records[0].images] == [("bottom", "uniform"), ("top", "uniform")]


def test_regex_manifest_parses_side_view_and_slot_from_names(tmp_path: Path) -> None:
    """Filename regex parsing should support side/view grouping when no CSV exists."""
    manifest = load_manifest_module()
    for name in [
        "part001_top_uniform_slot02.png",
        "part001_bottom_uniform_slot02.png",
        "part002_top_uniform_slot01.png",
    ]:
        (tmp_path / name).write_bytes(b"image")
    config = {
        "filename_regex": (
            r"(?P<part_id>part\d+)_(?P<side>top|bottom)_(?P<view>uniform)"
            r"_slot(?P<slot_id>\d+)\.png"
        ),
    }

    records = manifest.load_multiview_manifest(tmp_path, config)

    by_part = {record.part_id: record for record in records}
    assert sorted(by_part) == ["part001", "part002"]
    assert {image.side for image in by_part["part001"].images} == {"top", "bottom"}
    assert {image.slot_id for image in by_part["part001"].images} == {"slot02"}


def test_validate_required_views_reports_missing_side_view() -> None:
    """Missing required side/view pairs should make a capture invalid."""
    manifest = load_manifest_module()
    record = manifest.MultiViewPartRecord(
        part_id="part001",
        images=(
            manifest.MultiViewImageRecord(
                part_id="part001",
                side="top",
                view="uniform",
                image_path=Path("part001_top_uniform.png"),
            ),
        ),
    )
    config = {"ok_requires": {"required_sides": ["top", "bottom"], "required_views": ["uniform"]}}

    valid, missing = manifest.validate_required_views(record, config)

    assert not valid
    assert missing == ["bottom:uniform"]


def test_write_multiview_manifest_keeps_expected_schema(tmp_path: Path) -> None:
    """The generated manifest CSV should keep the MVP-5 schema."""
    manifest = load_manifest_module()
    output_csv = tmp_path / "manifest.csv"
    record = manifest.MultiViewPartRecord(
        part_id="part001",
        images=(
            manifest.MultiViewImageRecord(
                part_id="part001",
                side="top",
                view="uniform",
                image_path=tmp_path / "part001_top_uniform.png",
                label="normal",
                slot_id="slot01",
                group_id="g1",
            ),
        ),
    )

    manifest.write_multiview_manifest([record], output_csv)

    with output_csv.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert list(rows[0]) == [
        "part_id",
        "side",
        "view",
        "image_path",
        "label",
        "defect_type",
        "slot_id",
        "group_id",
        "notes",
    ]
    assert rows[0]["part_id"] == "part001"
    assert rows[0]["slot_id"] == "slot01"
