from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np

from bmw_inspection.lab.efficientad_ignore_mask import save_ignore_mask_asset
from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER
from pipeline.bmw_lab_select_efficientad_ignore_masks import (
    build_parser,
    display_to_source_point,
    load_training_release_roi_images,
    load_seed_polygons,
)


def test_display_to_source_point_clamps_and_scales() -> None:
    assert display_to_source_point(50, 25, display_shape=(50, 100), source_shape=(100, 200)) == (100, 50)
    assert display_to_source_point(1000, -3, display_shape=(50, 100), source_shape=(100, 200)) == (199, 0)


def test_parser_defaults_to_trusted_ok_saved_rois() -> None:
    args = build_parser().parse_args([])

    assert args.capture_id == "bmw_right_normal_group002_000001"
    assert args.roi_config.name == "bmw_right_hdr_eight_view_v1.json"
    assert args.from_index is None
    assert args.output.name == "bmw_right_manual_ignore_v2"


def test_parser_allows_blank_masks_from_a_training_release_sample(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        ["--training-release", str(tmp_path / "training"), "--representative-sample", "sample-2"]
    )

    assert args.training_release == tmp_path / "training"
    assert args.representative_sample == "sample-2"
    assert args.from_index is None


def test_load_training_release_roi_images_loads_one_complete_representative_sample(tmp_path: Path) -> None:
    release = tmp_path / "bmw-left-training-v1"
    rows: list[dict[str, str]] = []
    for sample_index, sample_id in enumerate(("sample-1", "sample-2")):
        for view_index, view in enumerate(VIEW_ORDER):
            relative = Path("crops") / view / f"session-{sample_index}__{sample_id}__{view}.png"
            path = release / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            image = np.full((8, 10, 3), sample_index * 40 + view_index, dtype=np.uint8)
            assert cv2.imwrite(str(path), image)
            rows.append(
                {
                    "sample_id": sample_id,
                    "session_id": f"session-{sample_index}",
                    "view_id": view,
                    "source_class": "normal",
                    "split": "train",
                    "crop_path": relative.as_posix(),
                    "crop_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    manifest = release / "manifests/crop_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    images, source_paths, source_id = load_training_release_roi_images(
        release,
        representative_sample="sample-2",
    )

    assert tuple(images) == VIEW_ORDER
    assert tuple(source_paths) == VIEW_ORDER
    assert source_id == "bmw-left-training-v1:sample-2"
    assert all(
        path == release / "crops" / view / f"session-1__sample-2__{view}"
        ".png"
        for view, path in source_paths.items()
    )
    assert all(image.shape == (8, 10, 3) for image in images.values())


def test_load_seed_polygons_reconstructs_sha_verified_v1_masks(tmp_path: Path) -> None:
    roi_config = tmp_path / "roi.json"
    roi_config.write_text("{}\n", encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    images = {view: np.zeros((8, 10, 3), dtype=np.uint8) for view in VIEW_ORDER}
    source_paths: dict[str, Path] = {}
    for view, image in images.items():
        path = source / f"{view}.png"
        assert cv2.imwrite(str(path), image)
        source_paths[view] = path
    expected = {view: [] for view in VIEW_ORDER}
    expected["back"] = [[(1, 1), (5, 1), (5, 5), (1, 5)]]
    index_path = save_ignore_mask_asset(
        tmp_path / "v1",
        images=images,
        polygons_by_view=expected,
        source_paths=source_paths,
        source_capture_id="capture-1",
        public_roi_config=roi_config,
    )

    actual = load_seed_polygons(
        index_path,
        images=images,
        source_paths=source_paths,
        public_roi_config=roi_config,
    )

    assert actual == expected
    actual["back"][0][0] = (9, 7)
    assert expected["back"][0][0] == (1, 1)
