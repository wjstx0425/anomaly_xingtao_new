"""Tests for the standalone BMW eight-view HDR data preparer."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import runpy

import cv2
import numpy as np
import pytest

from bmw_inspection.lab.eight_view_dataset import (
    CAPTURE_FIELDS,
    VIEW_ORDER,
    PreparedImage,
    _atomic_publish_noreplace,
    assign_stratified_part_splits,
    prepare_eight_view_dataset,
    read_complete_capture_rows,
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


def _capture_row(**updates: str) -> dict[str, str]:
    row = dict.fromkeys(CAPTURE_FIELDS, "")
    row.update(
        {
            "capture_mode": "hdr_fused",
            "gain": "0.0",
            "short_exposure": "1500.0",
            "long_exposure": "6000.0",
            "sample_status": "complete",
        }
    )
    row.update(updates)
    return row


def _write_session(
    root: Path,
    *,
    session_id: str,
    source_class: str,
    groups: int,
    incomplete_last: bool = False,
    hand: str = "left",
) -> None:
    rows: list[dict[str, str]] = []
    for group_index in range(1, groups + 1):
        group_id = f"group{group_index:03d}"
        sample_id = f"bmw_{source_class}_{group_id}_000001"
        for view_index, view in enumerate(VIEW_ORDER):
            label_parts = ("normal",) if source_class == "normal" else ("defect", source_class)
            image_path = root / hand / view
            for part in label_parts:
                image_path /= part
            image_path = image_path / session_id / "images" / f"{sample_id}__{view}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            class_offset = {"normal": 0, "no_streak": 40, "edge": 80}.get(source_class, 120)
            image = np.full((8, 10, 3), class_offset + group_index * 10 + view_index, dtype=np.uint8)
            assert cv2.imwrite(str(image_path), image)
            rows.append(
                _capture_row(
                    record_type="image",
                    session_id=session_id,
                    sample_id=sample_id,
                    group_id=group_id,
                    image_index="1",
                    round="front" if view.startswith("front") else "back",
                    view=view,
                    camera_serial=SERIAL_BY_VIEW[view],
                    file=str(image_path),
                )
            )
        rows.append(
            _capture_row(
                record_type="sample",
                session_id=session_id,
                sample_id=sample_id,
                group_id=group_id,
                image_index="1",
            )
        )
    if incomplete_last:
        rows.append(
            _capture_row(
                record_type="sample",
                session_id=session_id,
                sample_id=f"bmw_{source_class}_group999_000001",
                group_id="group999",
                image_index="1",
                sample_status="incomplete",
                failed_round="front",
            )
        )
    manifest = root / "manifests" / f"{session_id}.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CAPTURE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_reader_keeps_only_complete_exact_eight_view_samples(tmp_path: Path) -> None:
    _write_session(
        tmp_path,
        session_id="20260806_100000_000001",
        source_class="normal",
        groups=2,
        incomplete_last=True,
    )

    rows, audit = read_complete_capture_rows(tmp_path)

    assert len(rows) == 16
    assert audit.complete_sample_count == 2
    assert audit.incomplete_sample_count == 1
    assert {row.view_id for row in rows} == set(VIEW_ORDER)
    assert {row.physical_part_id for row in rows} == {
        "bmw_normal_group001",
        "bmw_normal_group002",
    }
    assert {row.source_class for row in rows} == {"normal"}


def test_reader_rejects_complete_sample_missing_one_view(tmp_path: Path) -> None:
    _write_session(
        tmp_path,
        session_id="20260806_100000_000002",
        source_class="normal",
        groups=1,
    )
    manifest = next((tmp_path / "manifests").glob("*.csv"))
    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    rows = [row for row in rows if not (row["record_type"] == "image" and row["view"] == "back_secondary")]
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CAPTURE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="exactly eight views"):
        read_complete_capture_rows(tmp_path)


def test_reader_capture_scope_keeps_only_right_samples_and_audit_counts(tmp_path: Path) -> None:
    _write_session(
        tmp_path,
        session_id="20260810_100000_000001",
        source_class="normal",
        groups=2,
        hand="left",
    )
    _write_session(
        tmp_path,
        session_id="20260810_100000_000002",
        source_class="edge",
        groups=3,
        incomplete_last=True,
        hand="right",
    )

    rows, audit = read_complete_capture_rows(tmp_path, capture_scope="right")

    assert len(rows) == 24
    assert {row.source_class for row in rows} == {"edge"}
    assert {row.source_path.relative_to(tmp_path).parts[0] for row in rows} == {"right"}
    assert audit.manifest_count == 1
    assert audit.complete_sample_count == 3
    assert audit.incomplete_sample_count == 1


def test_reader_maps_generic_defect_layout_to_others(tmp_path: Path) -> None:
    _write_session(
        tmp_path,
        session_id="20260820_153342_329257",
        source_class="defect",
        groups=1,
        hand="right",
    )

    rows, audit = read_complete_capture_rows(
        tmp_path,
        capture_scope="right",
        session_ids=("20260820_153342_329257",),
    )

    assert len(rows) == len(VIEW_ORDER)
    assert {row.source_class for row in rows} == {"others"}
    assert audit.complete_sample_count == 1


def test_reader_session_filter_keeps_only_requested_manifests(tmp_path: Path) -> None:
    selected_session = "20260810_100000_000011"
    _write_session(
        tmp_path,
        session_id="20260810_100000_000010",
        source_class="normal",
        groups=2,
        hand="right",
    )
    _write_session(
        tmp_path,
        session_id=selected_session,
        source_class="edge",
        groups=3,
        incomplete_last=True,
        hand="right",
    )

    rows, audit = read_complete_capture_rows(
        tmp_path,
        capture_scope="right",
        session_ids=(selected_session,),
    )

    assert len(rows) == 24
    assert {row.session_id for row in rows} == {selected_session}
    assert audit.manifest_count == 1
    assert audit.complete_sample_count == 3
    assert audit.incomplete_sample_count == 1


def test_reader_session_filter_rejects_missing_manifest(tmp_path: Path) -> None:
    _write_session(
        tmp_path,
        session_id="20260810_100000_000012",
        source_class="normal",
        groups=1,
        hand="right",
    )

    with pytest.raises(ValueError, match="capture session manifest does not exist"):
        read_complete_capture_rows(
            tmp_path,
            capture_scope="right",
            session_ids=("20260810_100000_999999",),
        )


def test_stratified_split_is_deterministic_and_keeps_parts_together() -> None:
    rows: list[PreparedImage] = []
    for source_class, count in (("normal", 10), ("no_streak", 4)):
        for index in range(count):
            part_id = f"session-{source_class}__group{index:03d}"
            for view in VIEW_ORDER:
                rows.append(
                    PreparedImage(
                        sample_id=f"sample-{source_class}-{index:03d}",
                        physical_part_id=part_id,
                        session_id=f"session-{source_class}",
                        group_id=f"group{index:03d}",
                        view_id=view,
                        camera_serial=SERIAL_BY_VIEW[view],
                        source_path=Path(f"/{part_id}/{view}.png"),
                        source_class=source_class,
                        source_sha256="",
                    )
                )

    first = assign_stratified_part_splits(rows, seed=42)
    second = assign_stratified_part_splits(tuple(reversed(rows)), seed=42)

    assert first == second
    assert sorted(first.values()).count("train") == 8
    assert sorted(first.values()).count("calibration") == 3
    assert sorted(first.values()).count("final_test") == 3
    assert len(first) == 14


def test_publication_preserves_no_streak_branch_semantics_and_refuses_overwrite(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _write_session(raw_root, session_id="20260806_100000_000010", source_class="normal", groups=3)
    _write_session(raw_root, session_id="20260806_100000_000011", source_class="no_streak", groups=3)
    _write_session(raw_root, session_id="20260806_100000_000012", source_class="edge", groups=3)
    output_root = tmp_path / "prepared"

    dry_report = prepare_eight_view_dataset(
        raw_root=raw_root,
        output_root=output_root,
        dataset_id="bmw-hdr-v1",
        dry_run=True,
    )
    assert dry_report["image_count"] == 72
    assert not output_root.exists()

    report = prepare_eight_view_dataset(
        raw_root=raw_root,
        output_root=output_root,
        dataset_id="bmw-hdr-v1",
    )
    release = output_root / "bmw-hdr-v1"
    assert report["release_status"] == "published"
    assert json.loads((release / "report.json").read_text(encoding="utf-8"))["image_count"] == 72

    efficientad = list(csv.DictReader((release / "manifests/efficientad.csv").open()))
    no_streak_ea = [row for row in efficientad if row["source_class"] == "no_streak"]
    assert len(no_streak_ea) == 24
    assert {row["branch_label"] for row in no_streak_ea} == {"normal"}

    template = list(csv.DictReader((release / "manifests/template.csv").open()))
    assert {row["branch_label"] for row in template if row["source_class"] == "no_streak"} == {"normal"}
    assert {row["branch_label"] for row in template if row["source_class"] == "edge"} == {"review_required"}

    yolo = list(csv.DictReader((release / "manifests/yolo_annotation.csv").open()))
    assert {row["annotation_status"] for row in yolo if row["source_class"] == "no_streak"} == {
        "negative_confirmed"
    }
    assert {row["annotation_status"] for row in yolo if row["source_class"] == "edge"} == {
        "review_required"
    }

    streak = list(csv.DictReader((release / "manifests/bright_streak.csv").open()))
    assert len(streak) == 9
    assert {row["view_id"] for row in streak} == {"front_left"}
    assert {row["expected_status"] for row in streak if row["source_class"] == "normal"} == {"OK"}
    assert {row["expected_status"] for row in streak if row["source_class"] == "no_streak"} == {
        "NG_NO_STREAK"
    }
    assert {row["expected_status"] for row in streak if row["source_class"] == "edge"} == {"REVIEW"}

    with pytest.raises(FileExistsError, match="already exists"):
        prepare_eight_view_dataset(
            raw_root=raw_root,
            output_root=output_root,
            dataset_id="bmw-hdr-v1",
        )


def test_pipeline_wrapper_dry_run_prints_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    raw_root = tmp_path / "raw"
    _write_session(raw_root, session_id="20260806_100000_000019", source_class="normal", groups=2)
    _write_session(
        raw_root,
        session_id="20260806_100000_000020",
        source_class="edge",
        groups=1,
        hand="right",
    )
    module = runpy.run_path("pipeline/bmw_lab_prepare_eight_view_data.py")

    exit_code = module["main"](
        [
            "--raw-root",
            str(raw_root),
            "--output-root",
            str(tmp_path / "prepared"),
            "--dataset-id",
            "bmw-hdr-v1",
            "--hand",
            "right",
            "--session-id",
            "20260806_100000_000020",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["complete_sample_count"] == 1
    assert payload["image_count"] == 8
    assert payload["capture_scope"] == "right"
    assert payload["session_ids"] == ["20260806_100000_000020"]
    assert payload["release_status"] == "dry_run"


def test_atomic_publish_never_replaces_existing_directory(tmp_path: Path) -> None:
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "new.txt").write_text("new", encoding="utf-8")
    destination = tmp_path / "release"
    destination.mkdir()
    (destination / "owned.txt").write_text("owned", encoding="utf-8")

    with pytest.raises(FileExistsError):
        _atomic_publish_noreplace(staged, destination)

    assert (destination / "owned.txt").read_text(encoding="utf-8") == "owned"
    assert (staged / "new.txt").read_text(encoding="utf-8") == "new"
