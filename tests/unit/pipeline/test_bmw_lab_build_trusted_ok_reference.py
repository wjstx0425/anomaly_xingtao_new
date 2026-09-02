"""Focused tests for the minimal BMW trusted-OK bank generator."""

from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest


VIEWS = (
    "front",
    "front_left",
    "front_right",
    "front_secondary",
    "back",
    "back_left",
    "back_right",
    "back_secondary",
)
PART_IDS = tuple(
    f"bmw_left_normal_group{group}"
    for group in ("002", "004", "005", "006", "008", "011", "012", "013")
)
MANIFEST_FIELDS = (
    "sample_id",
    "physical_part_id",
    "session_id",
    "group_id",
    "view_id",
    "camera_serial",
    "source_path",
    "source_sha256",
    "source_class",
    "business_label",
    "split",
)


def _module():
    return importlib.import_module("pipeline.bmw_lab_build_trusted_ok_reference")


def _inputs(tmp_path: Path, *, omit: tuple[str, str] | None = None) -> tuple[Path, Path]:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    rows: list[dict[str, str]] = []
    for part_number, part_id in enumerate(PART_IDS, start=1):
        sample_id = f"{part_id}_000001"
        for view_number, view in enumerate(VIEWS, start=1):
            if omit == (part_id, view):
                continue
            image = np.full((12, 16, 3), part_number * 10 + view_number, dtype=np.uint8)
            image_path = source_dir / f"{sample_id}_{view}.png"
            assert cv2.imwrite(str(image_path), image)
            rows.append(
                {
                    "sample_id": sample_id,
                    "physical_part_id": part_id,
                    "session_id": "20260820_162040_682498",
                    "group_id": part_id.rsplit("_", 1)[-1],
                    "view_id": view,
                    "camera_serial": f"camera-{view}",
                    "source_path": str(image_path),
                    # Deliberately stale: the lab generator must not inspect SHA fields.
                    "source_sha256": "stale-sha-is-ignored",
                    "source_class": "normal",
                    "business_label": "OK",
                    "split": "train",
                }
            )
    manifest_path = tmp_path / "dataset_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    roi_path = tmp_path / "left_0820_roi.json"
    roi_path.write_text(
        json.dumps(
            {
                "coordinate_system": "pixel_xyxy_half_open",
                "image_width": 16,
                "image_height": 12,
                "part_rois": {view: [2, 3, 11, 10] for view in VIEWS},
                # Deliberately stale: no config/manifest SHA binding is allowed.
                "source_manifest_sha256": "stale-sha-is-ignored",
            }
        ),
        encoding="utf-8",
    )
    return manifest_path, roi_path


def test_parser_defaults_to_left_0820_inputs_and_eight_selected_parts() -> None:
    builder = _module()

    args = builder.build_parser().parse_args([])

    assert args.manifest == Path(
        "/home/yunjing/anomaly_xingtao_new/dataset/bmw_lab_prepared/"
        "bmw_left_0820_v1/manifests/dataset_manifest.csv"
    )
    assert args.roi_config.name == "bmw_left_0820_v1.json"
    assert args.output == Path(
        "/home/yunjing/anomaly_xingtao_new/dataset/bmw_trusted_ok_reference/"
        "bmw_left_0820_train_normal_v1"
    )
    assert builder.DEFAULT_PART_IDS == PART_IDS


def test_build_writes_64_current_roi_crops_and_minimal_runtime_index(tmp_path: Path) -> None:
    builder = _module()
    manifest_path, roi_path = _inputs(tmp_path)
    output = tmp_path / "trusted-ok"

    index_path = builder.build_trusted_ok_reference(
        manifest_path=manifest_path,
        roi_config_path=roi_path,
        output_dir=output,
        physical_part_ids=PART_IDS,
    )

    assert index_path == output / "reference_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert set(payload) == {"references"}
    assert len(payload["references"]) == 64
    assert not any("sha256" in key for row in payload["references"] for key in row)
    assert len(list((output / "roi").glob("*/*.png"))) == 64
    assert not (output / "full").exists()

    first = payload["references"][0]
    assert set(first) == {
        "physical_part_id",
        "sample_id",
        "view_id",
        "full_image_path",
        "roi_image_path",
    }
    assert first["physical_part_id"] == PART_IDS[0]
    assert first["view_id"] == "front"
    assert Path(first["full_image_path"]).is_absolute()
    crop = cv2.imread(str(output / first["roi_image_path"]), cv2.IMREAD_UNCHANGED)
    assert crop is not None
    assert crop.shape == (7, 9, 3)

    # The generated shape is exactly what the current Demo matcher consumes.
    from bmw_inspection.lab.trusted_ok_reference import TrustedOkMatcher

    TrustedOkMatcher(index_path)


def test_build_rejects_a_selected_part_with_a_missing_view(tmp_path: Path) -> None:
    builder = _module()
    manifest_path, roi_path = _inputs(tmp_path, omit=(PART_IDS[0], "back_secondary"))

    with pytest.raises(ValueError, match="八视图不完整"):
        builder.build_trusted_ok_reference(
            manifest_path=manifest_path,
            roi_config_path=roi_path,
            output_dir=tmp_path / "trusted-ok",
            physical_part_ids=PART_IDS,
        )
