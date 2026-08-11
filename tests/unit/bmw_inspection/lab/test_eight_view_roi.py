"""Tests for BMW eight-view ROI selection contracts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import runpy

import cv2
import numpy as np
import pytest

from bmw_inspection.lab import eight_view_roi
from bmw_inspection.lab.eight_view_dataset import CAPTURE_FIELDS, VIEW_ORDER
from bmw_inspection.lab.eight_view_roi import (
    EightViewRoiConfig,
    fit_image_for_display,
    load_roi_config,
    save_roi_config,
    select_representative_images,
    source_roi,
)


DATASET_FIELDS = (
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
SERIAL_BY_VIEW = {
    "front": "DA9805574",
    "front_left": "DA9625347",
    "front_right": "DB0998274",
    "front_secondary": "DB0968108",
    "back": "DA9805574",
    "back_left": "DA9625347",
    "back_right": "DB0998274",
    "back_secondary": "DB0968108",
}


def _prepared_release(root: Path) -> Path:
    release = root / "bmw-test-v1"
    rows: list[dict[str, str]] = []
    for sample_index, (source_class, split) in enumerate((("normal", "train"), ("normal", "calibration"))):
        sample_id = f"sample-{sample_index:03d}"
        for view_index, view in enumerate(VIEW_ORDER):
            image_path = root / "source" / f"{sample_id}__{view}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image = np.full((80, 100, 3), sample_index * 20 + view_index, dtype=np.uint8)
            assert cv2.imwrite(str(image_path), image)
            rows.append(
                {
                    "sample_id": sample_id,
                    "physical_part_id": f"part-{sample_index:03d}",
                    "session_id": "session-001",
                    "group_id": f"group{sample_index:03d}",
                    "view_id": view,
                    "camera_serial": "serial",
                    "source_path": str(image_path),
                    "source_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    "source_class": source_class,
                    "business_label": "OK",
                    "split": split,
                }
            )
    manifest = release / "manifests/dataset_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=DATASET_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (release / "report.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "release_status": "published",
                "dataset_id": "bmw-test-v1",
                "image_width": 100,
                "image_height": 80,
            }
        ),
        encoding="utf-8",
    )
    return release


def _raw_right_release(root: Path) -> tuple[Path, str]:
    session_id = "20260810_164527_229315"
    sample_id = "bmw_right_others_group001_000001"
    rows: list[dict[str, str]] = []
    for index, view in enumerate(VIEW_ORDER):
        image_path = root / "right" / view / "defect" / "others" / session_id / "images" / f"{view}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(image_path), np.full((80, 100, 3), index, dtype=np.uint8))
        row = dict.fromkeys(CAPTURE_FIELDS, "")
        row.update(
            {
                "record_type": "image",
                "session_id": session_id,
                "sample_id": sample_id,
                "group_id": "group001",
                "image_index": "1",
                "round": "front" if view.startswith("front") else "back",
                "view": view,
                "camera_serial": SERIAL_BY_VIEW[view],
                "capture_mode": "hdr_fused",
                "gain": "0.0",
                "file": str(image_path),
                "short_exposure": "1500.0",
                "long_exposure": "6000.0",
                "sample_status": "complete",
            }
        )
        rows.append(row)
    sample_row = dict.fromkeys(CAPTURE_FIELDS, "")
    sample_row.update(
        {
            "record_type": "sample",
            "session_id": session_id,
            "sample_id": sample_id,
            "group_id": "group001",
            "image_index": "1",
            "capture_mode": "hdr_fused",
            "sample_status": "complete",
        }
    )
    rows.append(sample_row)
    manifest = root / "manifests" / f"{session_id}.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CAPTURE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return manifest, sample_id


def test_selects_one_complete_normal_train_sample_for_all_views(tmp_path: Path) -> None:
    release = _prepared_release(tmp_path)

    selection = select_representative_images(release)

    assert selection.dataset_id == "bmw-test-v1"
    assert selection.sample_id == "sample-000"
    assert tuple(selection.images) == VIEW_ORDER
    assert all(path.is_file() for path in selection.images.values())
    assert selection.image_width == 100
    assert selection.image_height == 80
    assert selection.manifest_sha256 == hashlib.sha256(
        (release / "manifests/dataset_manifest.csv").read_bytes()
    ).hexdigest()


def test_selects_raw_right_representative_for_reusable_roi(tmp_path: Path) -> None:
    manifest, sample_id = _raw_right_release(tmp_path)

    selection = eight_view_roi.select_raw_representative_images(
        tmp_path,
        profile_id="bmw-right-hdr-v1",
        capture_scope="right",
        source_class="others",
        sample_id=sample_id,
    )

    assert selection.dataset_id == "bmw-right-hdr-v1"
    assert selection.sample_id == sample_id
    assert selection.manifest_path == manifest.resolve()
    assert selection.manifest_sha256 == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert tuple(selection.images) == VIEW_ORDER
    assert selection.image_width == 100
    assert selection.image_height == 80


def test_display_mapping_returns_half_open_source_roi() -> None:
    source = np.zeros((3036, 4024, 3), dtype=np.uint8)

    displayed = fit_image_for_display(source, max_display_width=1006, max_display_height=759)
    roi = source_roi((100, 50, 200, 100), displayed.shape[:2], source.shape[:2])

    assert displayed.shape[:2] == (759, 1006)
    assert roi == (400, 200, 1200, 600)


def test_roi_config_is_manifest_bound_and_requires_exact_eight_views(tmp_path: Path) -> None:
    release = _prepared_release(tmp_path)
    selection = select_representative_images(release)
    config = EightViewRoiConfig(
        dataset_id=selection.dataset_id,
        source_manifest=selection.manifest_path,
        source_manifest_sha256=selection.manifest_sha256,
        representative_sample_id=selection.sample_id,
        image_width=selection.image_width,
        image_height=selection.image_height,
        part_rois={view: (1, 2, 90, 70) for view in VIEW_ORDER},
    )
    path = tmp_path / "rois.json"

    save_roi_config(path, config)

    loaded = load_roi_config(path)
    assert loaded == config
    assert tuple(json.loads(path.read_text(encoding="utf-8"))["part_rois"]) == VIEW_ORDER
    with pytest.raises(FileExistsError):
        save_roi_config(path, config)
    save_roi_config(path, config, force=True)

    with pytest.raises(ValueError, match="exactly the eight"):
        EightViewRoiConfig(
            dataset_id=selection.dataset_id,
            source_manifest=selection.manifest_path,
            source_manifest_sha256=selection.manifest_sha256,
            representative_sample_id=selection.sample_id,
            image_width=selection.image_width,
            image_height=selection.image_height,
            part_rois={view: (1, 2, 90, 70) for view in VIEW_ORDER[:-1]},
        )


def test_fixed_setup_roi_config_round_trips_as_reusable_schema_v2(tmp_path: Path) -> None:
    manifest, sample_id = _raw_right_release(tmp_path)
    config = EightViewRoiConfig(
        dataset_id="bmw-right-hdr-v1",
        source_manifest=manifest,
        source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        representative_sample_id=sample_id,
        image_width=100,
        image_height=80,
        part_rois={view: (1, 2, 90, 70) for view in VIEW_ORDER},
        binding_mode="fixed_setup",
        capture_scope="right",
    )
    path = tmp_path / "right-rois.json"

    save_roi_config(path, config)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["binding_mode"] == "fixed_setup"
    assert payload["capture_scope"] == "right"
    assert load_roi_config(path) == config


def test_opencv_roi_window_name_is_ascii_for_qt_compatibility() -> None:
    namespace = runpy.run_path("pipeline/bmw_lab_select_eight_view_rois.py")

    title = namespace["_window_name"]("front_secondary")

    assert title.isascii()
    assert "front_secondary" in title


def test_roi_selector_cli_accepts_raw_right_reusable_profile() -> None:
    namespace = runpy.run_path("pipeline/bmw_lab_select_eight_view_rois.py")

    args = namespace["build_parser"]().parse_args(
        [
            "--raw-root",
            "dataset/bmw_lab_raw",
            "--hand",
            "right",
            "--source-class",
            "others",
            "--sample-id",
            "bmw_right_others_group001_000001",
            "--profile-id",
            "bmw-right-hdr-v1",
        ]
    )

    assert args.raw_root == Path("dataset/bmw_lab_raw")
    assert args.hand == "right"
    assert args.source_class == "others"
    assert args.profile_id == "bmw-right-hdr-v1"
