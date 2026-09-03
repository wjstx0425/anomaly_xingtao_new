"""Focused contract tests for the simplified eight-view Demo."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from bmw_inspection.views import VIEW_ORDER
from bmw_inspection.lab.eight_view_demo import (
    BranchStatus,
    DemoBranch,
    DemoBranchResult,
    DemoFinalStatus,
    fuse_demo_status,
    load_capture_directory,
    load_manifest_sample,
)


def _row(status: BranchStatus) -> DemoBranchResult:
    return DemoBranchResult(
        DemoBranch.TEMPLATE,
        "front",
        status,
        0.1,
        0.2,
        1.0,
        "fixture",
        None,
    )


def test_fusion_keeps_error_above_ng_above_ok() -> None:
    assert fuse_demo_status((_row(BranchStatus.PASS),)) is DemoFinalStatus.OK
    assert fuse_demo_status((_row(BranchStatus.PASS), _row(BranchStatus.NG))) is DemoFinalStatus.NG
    assert (
        fuse_demo_status((_row(BranchStatus.NG), _row(BranchStatus.ERROR)))
        is DemoFinalStatus.ERROR
    )


def test_capture_directory_requires_exact_readable_eight_views(tmp_path: Path) -> None:
    for index, view in enumerate(VIEW_ORDER):
        assert cv2.imwrite(
            str(tmp_path / f"{view}.png"),
            np.full((8, 9, 3), index, dtype=np.uint8),
        )
    images = load_capture_directory(tmp_path)
    assert tuple(images) == VIEW_ORDER

    (tmp_path / "back.png").unlink()
    with pytest.raises(ValueError, match="back"):
        load_capture_directory(tmp_path)


def test_manifest_loader_requires_one_row_per_view_and_readable_images(tmp_path: Path) -> None:
    rows: list[dict[str, str]] = []
    for index, view in enumerate(VIEW_ORDER):
        image_path = tmp_path / f"{view}.png"
        assert cv2.imwrite(str(image_path), np.full((8, 9, 3), index, dtype=np.uint8))
        rows.append(
            {
                "sample_id": "sample-001",
                "view_id": view,
                "source_path": image_path.name,
            }
        )
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", "view_id", "source_path"))
        writer.writeheader()
        writer.writerows(rows)

    assert tuple(load_manifest_sample(manifest, "sample-001")) == VIEW_ORDER

    rows.pop()
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", "view_id", "source_path"))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="不是完整八视图"):
        load_manifest_sample(manifest, "sample-001")
