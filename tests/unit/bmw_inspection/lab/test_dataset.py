"""Tests for physical-part BMW dataset manifests and training exports."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import runpy

import cv2
import numpy as np
import pytest
from ultralytics.data.utils import check_det_dataset

from bmw_inspection.lab.dataset import (
    MANIFEST_FIELDS,
    REGRESSION_FIELDS,
    assign_part_splits,
    build_dataset as _build_dataset,
    find_part_split_leakage,
    read_manifest,
    validate_row,
)
from bmw_inspection.lab.contracts import ViewId


PART_ROIS = {view_id: (1, 1, 9, 7) for view_id in ViewId}


def build_dataset(**kwargs: object) -> dict[str, object]:
    """Call the production builder with the deployment-equivalent six-view ROIs."""
    return _build_dataset(part_rois=PART_ROIS, **kwargs)  # type: ignore[arg-type]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _source_fixture(
    root: Path,
    *,
    part_count: int,
    all_normal: bool = False,
) -> tuple[Path, list[dict[str, str]], dict[Path, str]]:
    rows: list[dict[str, str]] = []
    source_hashes: dict[Path, str] = {}
    for part_index in range(part_count):
        part_id = f"part-{part_index:03d}"
        sample_id = f"sample-{part_index:03d}"
        defect_view = tuple(ViewId)[part_index % len(ViewId)]
        for view_index, view_id in enumerate(ViewId):
            image_path = root / "source" / f"{sample_id}__{view_id.value}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image = np.zeros((8, 10, 3), dtype=np.uint8)
            image[:, :, 0] = part_index
            image[:, :, 1] = view_index
            image[:, :, 2] = part_index + view_index
            assert cv2.imwrite(str(image_path), image)
            source_hashes[image_path] = _sha256(image_path)
            is_defect = not all_normal and view_id is defect_view
            rows.append(
                {
                    "sample_id": sample_id,
                    "part_id": part_id,
                    "session_id": f"session-{part_index // 5:02d}",
                    "view_id": view_id.value,
                    "image_path": str(image_path),
                    "image_sha256": source_hashes[image_path],
                    "label": "defect" if is_defect else "normal",
                    "defect_type": "scratch" if is_defect else "",
                    "x1": "1" if is_defect else "",
                    "y1": "2" if is_defect else "",
                    "x2": "7" if is_defect else "",
                    "y2": "6" if is_defect else "",
                    "split": "",
                }
            )
    manifest = root / "source_manifest.csv"
    _write_csv(manifest, rows)
    return manifest, rows, source_hashes


def _regression_fixture(root: Path) -> Path:
    regression_root = root / "legacy_bright_streak"
    for status, count in (("OK", 6), ("NG", 7)):
        for index in range(count):
            path = regression_root / status / f"Image_20260805_{status}_{index:02d}.bmp"
            path.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(path), np.full((8, 10, 3), index, dtype=np.uint8))
    return regression_root


def test_one_part_cannot_cross_splits() -> None:
    rows = [
        {"part_id": "part-001", "split": "train"},
        {"part_id": "part-001", "split": "final_test"},
        {"part_id": "part-002", "split": "calibration"},
    ]

    assert find_part_split_leakage(rows) == {"part-001": ("final_test", "train")}


def test_yolo_positive_requires_valid_defect_box() -> None:
    row = {
        "sample_id": "sample-1",
        "part_id": "part-1",
        "session_id": "session-1",
        "view_id": "front",
        "image_path": "front.png",
        "image_sha256": "0" * 64,
        "label": "defect",
        "defect_type": "scratch",
        "x1": "",
        "y1": "",
        "x2": "",
        "y2": "",
        "split": "train",
    }

    with pytest.raises(ValueError, match="bounding box"):
        validate_row(row)


def test_manifest_requires_explicit_part_and_session_identity() -> None:
    row = {
        "sample_id": "sample-1",
        "part_id": "",
        "session_id": "",
        "view_id": "front",
        "image_path": "front.png",
        "image_sha256": "0" * 64,
        "label": "normal",
        "defect_type": "",
        "x1": "",
        "y1": "",
        "x2": "",
        "y2": "",
        "split": "train",
    }

    with pytest.raises(ValueError, match="part_id"):
        validate_row(row)
    row["part_id"] = "part-1"
    with pytest.raises(ValueError, match="session_id"):
        validate_row(row)


def test_split_assignment_is_deterministic_sixty_twenty_twenty_by_part() -> None:
    part_ids = tuple(f"part-{index:02d}" for index in range(10))

    first = assign_part_splits(part_ids, seed=42)
    second = assign_part_splits(tuple(reversed(part_ids)), seed=42)

    assert first == second
    assert sorted(first.values()).count("train") == 6
    assert sorted(first.values()).count("calibration") == 2
    assert sorted(first.values()).count("final_test") == 2
    assert not find_part_split_leakage(
        [{"part_id": part_id, "split": split} for part_id, split in first.items()]
    )


def test_small_data_requires_explicit_experimental_flag_before_writing(tmp_path: Path) -> None:
    source_manifest, _rows, _hashes = _source_fixture(tmp_path, part_count=10)
    regression_root = _regression_fixture(tmp_path)
    output_root = tmp_path / "output"

    with pytest.raises(ValueError, match="30 complete six-view physical parts.*experimental-small-data"):
        build_dataset(
            source_manifest=source_manifest,
            output_root=output_root,
            dataset_id="bmw-small-v1",
            bright_streak_root=regression_root,
        )

    assert not output_root.exists()


def test_build_exports_keep_sources_and_enforce_yolo_and_patchcore_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_manifest, source_rows, source_hashes = _source_fixture(tmp_path, part_count=10)
    regression_root = _regression_fixture(tmp_path)
    output_root = tmp_path / "output"

    report = build_dataset(
        source_manifest=source_manifest,
        output_root=output_root,
        dataset_id="bmw-small-v1",
        bright_streak_root=regression_root,
        seed=42,
        experimental_small_data=True,
    )

    assert report["experimental_only"] is True
    assert report["final_evaluation_allowed"] is False
    assert report["physical_part_count"] == 10
    assert report["split_part_counts"] == {"train": 6, "calibration": 2, "final_test": 2}
    assert report["split_leakage_count"] == 0
    assert report["bright_streak_regression_rows"] == 13
    manifest_rows = read_manifest(output_root / "manifests/bmw-small-v1.csv")
    assert len(manifest_rows) == 60
    assert {row.part_id for row in manifest_rows} == {f"part-{index:03d}" for index in range(10)}
    assert all("legacy_bright_streak" not in str(row.image_path) for row in manifest_rows)
    for source_path, digest in source_hashes.items():
        assert source_path.is_file()
        assert _sha256(source_path) == digest

    yolo_root = output_root / "exports/bmw-small-v1/yolo"
    data_yaml = (yolo_root / "data.yaml").read_text(encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    resolved_yolo = check_det_dataset(str(yolo_root / "data.yaml"), autodownload=False)
    assert Path(resolved_yolo["train"]) == yolo_root / "images/train"
    assert Path(resolved_yolo["val"]) == yolo_root / "images/val"
    assert Path(resolved_yolo["test"]) == yolo_root / "images/test"
    assert not data_yaml.startswith("path:")
    assert data_yaml.endswith("names:\n  0: defect\n")
    label_files = sorted((yolo_root / "labels").glob("*/*.txt"))
    assert len(label_files) == len(source_rows)
    assert sum(bool(path.read_text(encoding="utf-8").strip()) for path in label_files) == 10
    assert all(
        not text or text.startswith("0 ")
        for path in label_files
        if (text := path.read_text(encoding="utf-8").strip()) is not None
    )
    yolo_export_rows = list(csv.DictReader((yolo_root / "export_manifest.csv").open()))
    assert all((yolo_root / row["exported_image_path"]).is_file() for row in yolo_export_rows)
    assert all((yolo_root / row["exported_label_path"]).is_file() for row in yolo_export_rows)
    assert {
        (int(row["roi_x1"]), int(row["roi_y1"]), int(row["roi_x2"]), int(row["roi_y2"]))
        for row in yolo_export_rows
    } == {(1, 1, 9, 7)}
    exported_images = sorted((yolo_root / "images").glob("*/*.png"))
    assert exported_images
    assert {cv2.imread(str(path), cv2.IMREAD_UNCHANGED).shape[:2] for path in exported_images} == {(6, 8)}
    positive_labels = [
        path.read_text(encoding="utf-8").strip()
        for path in label_files
        if path.read_text(encoding="utf-8").strip()
    ]
    assert set(positive_labels) == {"0 0.375000 0.500000 0.750000 0.666667"}

    split_by_part = {row.part_id: row.split for row in manifest_rows}
    train_normal_samples = {
        f"{row.sample_id}__{row.view_id.value}"
        for row in manifest_rows
        if split_by_part[row.part_id] == "train" and row.label == "normal"
    }
    exported_train = {
        path.stem for path in (output_root / "exports/bmw-small-v1/patchcore").glob("*/train/good/*")
    }
    assert exported_train == train_normal_samples
    assert not list((output_root / "exports/bmw-small-v1/patchcore").glob("*/train/defect/*"))
    assert {path.parent.parent.parent.name for path in exported_train_paths(output_root)} == {
        view.value for view in ViewId
    }
    assert list((output_root / "exports/bmw-small-v1/patchcore").glob("*/calibration/*/*"))
    assert list((output_root / "exports/bmw-small-v1/patchcore").glob("*/final_test/*/*"))
    assert not list((output_root / "exports/bmw-small-v1/patchcore").glob("*/validation/*/*"))
    for manifest_path in (output_root / "exports/bmw-small-v1/patchcore").glob("*/manifest.csv"):
        export_rows = list(csv.DictReader(manifest_path.open()))
        assert all(
            (output_root / "exports/bmw-small-v1/patchcore" / row["exported_image_path"]).is_file()
            for row in export_rows
        )

    regression_rows = list(csv.DictReader((output_root / "manifests/bright_streak_regression.csv").open()))
    assert tuple(regression_rows[0]) == REGRESSION_FIELDS
    assert len(regression_rows) == 13
    assert Counter(row["expected_bright_streak_status"] for row in regression_rows) == {
        "OK": 6,
        "NG_NO_STREAK": 7,
    }
    assert {row["purpose"] for row in regression_rows} == {"bright_streak_regression_only"}
    assert all("legacy_bright_streak" in row["image_path"] for row in regression_rows)
    stored_report = json.loads((output_root / "manifests/bmw-small-v1.report.json").read_text(encoding="utf-8"))
    assert stored_report == report


def exported_train_paths(output_root: Path) -> list[Path]:
    return list((output_root / "exports/bmw-small-v1/patchcore").glob("*/train/good/*"))


def test_dry_run_validates_and_reports_without_materializing_outputs(tmp_path: Path) -> None:
    source_manifest, rows, _hashes = _source_fixture(tmp_path, part_count=5)
    regression_root = _regression_fixture(tmp_path)
    output_root = tmp_path / "output"

    report = build_dataset(
        source_manifest=source_manifest,
        output_root=output_root,
        dataset_id="dry-run",
        bright_streak_root=regression_root,
        experimental_small_data=True,
        dry_run=True,
    )

    assert report["physical_part_count"] == 5
    assert report["experimental_only"] is True
    assert not output_root.exists()

    rows[0]["x2"] = "999"
    _write_csv(source_manifest, rows)
    with pytest.raises(ValueError, match="bounding box exceeds image dimensions"):
        build_dataset(
            source_manifest=source_manifest,
            output_root=output_root,
            dataset_id="dry-run-invalid-box",
            bright_streak_root=regression_root,
            experimental_small_data=True,
            dry_run=True,
        )


def test_incomplete_capture_and_cross_part_duplicate_content_are_rejected(tmp_path: Path) -> None:
    source_manifest, rows, _hashes = _source_fixture(tmp_path, part_count=2)
    regression_root = _regression_fixture(tmp_path)
    _write_csv(source_manifest, rows[:-1])
    with pytest.raises(ValueError, match="complete six-view capture"):
        build_dataset(
            source_manifest=source_manifest,
            output_root=tmp_path / "incomplete",
            dataset_id="incomplete",
            bright_streak_root=regression_root,
            experimental_small_data=True,
            dry_run=True,
        )

    duplicate = [dict(row) for row in rows]
    duplicate[6]["image_path"] = duplicate[0]["image_path"]
    duplicate[6]["image_sha256"] = duplicate[0]["image_sha256"]
    _write_csv(source_manifest, duplicate)
    with pytest.raises(ValueError, match="same image content.*physical parts"):
        build_dataset(
            source_manifest=source_manifest,
            output_root=tmp_path / "duplicate",
            dataset_id="duplicate",
            bright_streak_root=regression_root,
            experimental_small_data=True,
            dry_run=True,
        )

    same_view_image = [dict(row) for row in rows]
    for row in same_view_image[1:6]:
        row["image_path"] = same_view_image[0]["image_path"]
        row["image_sha256"] = same_view_image[0]["image_sha256"]
    _write_csv(source_manifest, same_view_image)
    with pytest.raises(ValueError, match="distinct source images"):
        build_dataset(
            source_manifest=source_manifest,
            output_root=tmp_path / "same-image-views",
            dataset_id="same-image-views",
            bright_streak_root=regression_root,
            experimental_small_data=True,
            dry_run=True,
        )


def test_thirty_complete_parts_enable_final_evaluation(tmp_path: Path) -> None:
    source_manifest, _rows, _hashes = _source_fixture(tmp_path, part_count=30)
    regression_root = _regression_fixture(tmp_path)

    report = build_dataset(
        source_manifest=source_manifest,
        output_root=tmp_path / "output",
        dataset_id="bmw-final-v1",
        bright_streak_root=regression_root,
        dry_run=True,
    )

    assert report["split_part_counts"] == {"train": 18, "calibration": 6, "final_test": 6}
    assert report["experimental_only"] is False
    assert report["final_evaluation_allowed"] is True
    assert report["defect_bbox_rows"] == 30


def test_final_evaluation_requires_actual_defect_boxes(tmp_path: Path) -> None:
    source_manifest, _rows, _hashes = _source_fixture(tmp_path, part_count=30, all_normal=True)
    regression_root = _regression_fixture(tmp_path)

    with pytest.raises(ValueError, match="actual defect bounding boxes"):
        build_dataset(
            source_manifest=source_manifest,
            output_root=tmp_path / "output",
            dataset_id="all-normal",
            bright_streak_root=regression_root,
            dry_run=True,
        )


def test_library_build_requires_the_fixed_bright_streak_regression_root(tmp_path: Path) -> None:
    source_manifest, _rows, _hashes = _source_fixture(tmp_path, part_count=5)

    with pytest.raises(TypeError, match="bright_streak_root"):
        build_dataset(  # type: ignore[call-arg]
            source_manifest=source_manifest,
            output_root=tmp_path / "output",
            dataset_id="missing-regression",
            experimental_small_data=True,
            dry_run=True,
        )


def test_cli_exposes_explicit_source_identity_and_small_data_controls() -> None:
    project_root = Path(__file__).resolve().parents[4]
    namespace = runpy.run_path(str(project_root / "pipeline/bmw_lab_build_manifest.py"))

    args = namespace["build_parser"]().parse_args(
        [
            "--source-manifest",
            "source.csv",
            "--dataset-id",
            "bmw-v1",
            "--experimental-small-data",
            "--dry-run",
        ]
    )

    assert args.source_manifest == Path("source.csv")
    assert args.dataset_id == "bmw-v1"
    assert args.config == project_root / "configs/bmw/experiments/bmw_lab_v1.json"
    assert args.experimental_small_data is True
    assert args.dry_run is True


def test_yolo_export_clips_boxes_to_part_roi_and_rejects_zero_intersection(tmp_path: Path) -> None:
    source_manifest, rows, _hashes = _source_fixture(tmp_path, part_count=1)
    regression_root = _regression_fixture(tmp_path)
    defect = next(row for row in rows if row["label"] == "defect")
    defect.update({"x1": "0", "y1": "0", "x2": "3", "y2": "4"})
    _write_csv(source_manifest, rows)

    build_dataset(
        source_manifest=source_manifest,
        output_root=tmp_path / "clipped",
        dataset_id="clipped",
        bright_streak_root=regression_root,
        experimental_small_data=True,
    )
    label = next(
        path
        for path in (tmp_path / "clipped/exports/clipped/yolo/labels").glob("*/*.txt")
        if path.read_text(encoding="utf-8").strip()
    )
    assert label.read_text(encoding="utf-8").strip() == "0 0.125000 0.250000 0.250000 0.500000"

    defect.update({"x1": "0", "y1": "0", "x2": "1", "y2": "1"})
    _write_csv(source_manifest, rows)
    with pytest.raises(ValueError, match="does not intersect configured part ROI"):
        build_dataset(
            source_manifest=source_manifest,
            output_root=tmp_path / "outside",
            dataset_id="outside",
            bright_streak_root=regression_root,
            experimental_small_data=True,
        )
    assert not (tmp_path / "outside").exists()
