# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for resumable ZS32 three-model offline calibration."""

from __future__ import annotations

import csv
import hashlib
import json
from typing import TYPE_CHECKING

import pytest
from capture_data.zs32_offline_calibration import (
    CALIBRATION_FIELDS,
    ZS32_VIEWS,
    _completed_runtime_case,
    load_offline_cases,
    merge_calibration_rows,
    write_case_index,
    write_yolo_annotation_calibration_rows,
)

if TYPE_CHECKING:
    from pathlib import Path

TEMPLATE_CALIBRATION_FIELDS = (*CALIBRATION_FIELDS, "threshold_version", "template_version")


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    """Write one deterministic CSV fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _manifests(tmp_path: Path, *, missing_view: str | None = None) -> tuple[Path, Path, Path]:
    """Create a crop manifest plus the template score split contract."""
    root = tmp_path / "root"
    manifest_rows: list[dict[str, object]] = []
    template_rows: list[dict[str, object]] = []
    for part_id, label, split in (
        ("right:normal:group001", "normal", "calibration"),
        ("right:defect:deform:group001", "defect", "test"),
    ):
        context = part_id.split(":")
        defect_type = context[2] if label == "defect" else ""
        group_id = context[-1]
        for view in ZS32_VIEWS:
            source = (
                root
                / "dataset"
                / "right"
                / view
                / f"right_{view}_{label}_zs32_right_{defect_type or 'normal'}_{group_id}_000001_fused.png"
            )
            crop = root / "dataset" / "zs32_patchcore_roi_yolo" / "right" / view / source.name
            source.parent.mkdir(parents=True, exist_ok=True)
            crop.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"source")
            crop.write_bytes(b"crop")
            if not (part_id.startswith("right:defect") and view == missing_view):
                manifest_rows.append(
                    {
                        "source_path": str(source.relative_to(root)),
                        "output_path": str(crop.relative_to(root)),
                        "hand": "right",
                        "source_view": view,
                        "resolved_view": view,
                        "label": label,
                        "defect_type": defect_type,
                        "session_id": "session-001",
                    },
                )
            template_rows.append(
                {
                    "part_id": part_id,
                    "hand": "right",
                    "view": view,
                    "branch": "template_match",
                    "raw_score": "0.1" if label == "normal" else "0.9",
                    "gt_label": "0" if label == "normal" else "1",
                    "split": split,
                    "model_version": "template-v1",
                    "roi_version": "roi-v1",
                    "threshold_version": "threshold-v1",
                    "template_version": "template-generation-v1",
                },
            )
    crop_manifest = tmp_path / "crop_manifest.csv"
    _write_csv(crop_manifest, tuple(manifest_rows[0]), manifest_rows)
    template_csv = tmp_path / "template_calibration.csv"
    _write_csv(template_csv, TEMPLATE_CALIBRATION_FIELDS, template_rows)
    return crop_manifest, template_csv, root


def test_load_offline_cases_uses_template_split_and_raw_source_images(tmp_path: Path) -> None:
    """Calibration cases must use frozen template splits and uncropped source images."""
    crop_manifest, template_csv, root = _manifests(tmp_path)

    cases = load_offline_cases(crop_manifest, template_csv, path_root=root, hand="right")

    assert [(case.part_id, case.gt_label, case.split) for case in cases] == [
        ("right:defect:deform:group001", 1, "test"),
        ("right:normal:group001", 0, "calibration"),
    ]
    assert all(tuple(case.images) == ZS32_VIEWS for case in cases)
    assert all("zs32_patchcore_roi_yolo" not in str(path) for case in cases for path in case.images.values())


def test_load_offline_cases_rejects_incomplete_eight_view_parts(tmp_path: Path) -> None:
    """A selected physical part missing one view must fail before GPU inference."""
    crop_manifest, template_csv, root = _manifests(tmp_path, missing_view="back_right")

    with pytest.raises(ValueError, match=r"exactly the .* canonical views"):
        load_offline_cases(crop_manifest, template_csv, path_root=root, hand="right")


def test_load_offline_cases_rejects_noncanonical_template_csv_view_order(tmp_path: Path) -> None:
    """Template split rows must preserve exact VIEW_ORDER for every physical part."""
    crop_manifest, template_csv, root = _manifests(tmp_path)
    rows = list(csv.DictReader(template_csv.open(encoding="utf-8")))
    defect_rows = [row for row in rows if row["gt_label"] == "1"]
    for row, routed_view in zip(defect_rows, reversed(ZS32_VIEWS), strict=True):
        row["view"] = routed_view
    _write_csv(template_csv, TEMPLATE_CALIBRATION_FIELDS, rows)

    with pytest.raises(ValueError, match="canonical view order"):
        load_offline_cases(crop_manifest, template_csv, path_root=root, hand="right")


def test_load_offline_cases_rejects_duplicate_template_csv_view(tmp_path: Path) -> None:
    """A repeated view cannot hide another missing view in an eight-row part."""
    crop_manifest, template_csv, root = _manifests(tmp_path)
    rows = list(csv.DictReader(template_csv.open(encoding="utf-8")))
    defect_rows = [row for row in rows if row["gt_label"] == "1"]
    defect_rows[-1]["view"] = defect_rows[-2]["view"]
    _write_csv(template_csv, TEMPLATE_CALIBRATION_FIELDS, rows)

    with pytest.raises(ValueError, match="exact canonical views"):
        load_offline_cases(crop_manifest, template_csv, path_root=root, hand="right")


def test_load_offline_cases_rejects_physical_part_reused_across_fit_and_test(tmp_path: Path) -> None:
    """One physical part identity cannot contribute to both selection and evaluation."""
    crop_manifest, template_csv, root = _manifests(tmp_path)
    rows = list(csv.DictReader(template_csv.open(encoding="utf-8")))
    test_part_id = next(row["part_id"] for row in rows if row["split"] == "test")
    for row in rows:
        if row["split"] == "calibration":
            row["part_id"] = test_part_id
    _write_csv(template_csv, TEMPLATE_CALIBRATION_FIELDS, rows)

    with pytest.raises(ValueError, match="inconsistent label/split"):
        load_offline_cases(crop_manifest, template_csv, path_root=root, hand="right")


def test_load_offline_cases_rejects_mixed_template_csv_versions(tmp_path: Path) -> None:
    """All Template split records must bind one complete artifact generation."""
    crop_manifest, template_csv, root = _manifests(tmp_path)
    rows = list(csv.DictReader(template_csv.open(encoding="utf-8")))
    rows[-1]["template_version"] = "substituted-template-generation"
    _write_csv(template_csv, TEMPLATE_CALIBRATION_FIELDS, rows)

    with pytest.raises(ValueError, match="Template artifact versions"):
        load_offline_cases(crop_manifest, template_csv, path_root=root, hand="right")


def test_merge_calibration_rows_requires_three_branches_for_every_view(tmp_path: Path) -> None:
    """Merged calibration must contain template, PatchCore, and YOLO for every selected part/view."""
    crop_manifest, template_csv, root = _manifests(tmp_path)
    cases = load_offline_cases(crop_manifest, template_csv, path_root=root, hand="right")
    case_root = tmp_path / "cases"
    for case in cases:
        rows: list[dict[str, object]] = [
            {
                "part_id": case.part_id,
                "hand": "right",
                "view": view,
                "branch": branch,
                "raw_score": "0.2",
                "gt_label": case.gt_label,
                "split": case.split,
                "model_version": f"{branch}-v1",
                "roi_version": "roi-v1",
            }
            for view in ZS32_VIEWS
            for branch in (f"anomaly_{view}", "yolo")
        ]
        _write_csv(case_root / case.slug / "calibration_rows.csv", CALIBRATION_FIELDS, rows)
        template_rows = [
            {
                "part_id": case.part_id,
                "hand": "right",
                "view": view,
                "branch": "template_match",
                "raw_score": "0.1",
                "gt_label": case.gt_label,
                "split": case.split,
                "model_version": "template-v1",
                "roi_version": "roi-v1",
            }
            for view in ZS32_VIEWS
        ]
        _write_csv(case_root / case.slug / "template_calibration_rows.csv", CALIBRATION_FIELDS, template_rows)

    output = tmp_path / "all_rows.csv"
    merged = merge_calibration_rows(cases, case_root, output)

    expected_count = len(cases) * len(ZS32_VIEWS) * 3
    assert merged == expected_count
    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert len(rows) == expected_count
    assert {row["branch"] for row in rows} == {
        "template_match",
        "yolo",
        *(f"anomaly_{view}" for view in ZS32_VIEWS),
    }


def test_merge_calibration_rows_rejects_a_missing_model_score(tmp_path: Path) -> None:
    """Partial model publications must not produce apparently valid thresholds."""
    crop_manifest, template_csv, root = _manifests(tmp_path)
    cases = load_offline_cases(crop_manifest, template_csv, path_root=root, hand="right")
    case_root = tmp_path / "cases"
    for case in cases:
        rows = [
            {
                "part_id": case.part_id,
                "hand": "right",
                "view": view,
                "branch": branch,
                "raw_score": "0.2",
                "gt_label": case.gt_label,
                "split": case.split,
                "model_version": "model-v1",
                "roi_version": "roi-v1",
            }
            for view in ZS32_VIEWS
            for branch in (f"anomaly_{view}", "yolo")
        ]
        if case == cases[0]:
            rows.pop()
        _write_csv(case_root / case.slug / "calibration_rows.csv", CALIBRATION_FIELDS, rows)
        template_rows = [
            {
                "part_id": case.part_id,
                "hand": "right",
                "view": view,
                "branch": "template_match",
                "raw_score": "0.1",
                "gt_label": case.gt_label,
                "split": case.split,
                "model_version": "template-v1",
                "roi_version": "roi-v1",
            }
            for view in ZS32_VIEWS
        ]
        _write_csv(case_root / case.slug / "template_calibration_rows.csv", CALIBRATION_FIELDS, template_rows)

    with pytest.raises(ValueError, match="18 unique calibration groups"):
        merge_calibration_rows(cases, case_root, tmp_path / "all_rows.csv")


def test_yolo_annotation_calibration_uses_per_view_labels_and_val_test_splits(tmp_path: Path) -> None:
    """YOLO thresholds must use box-label presence, not the physical-part defect label."""
    crop_manifest, template_csv, root = _manifests(tmp_path)
    cases = load_offline_cases(crop_manifest, template_csv, path_root=root, hand="right")
    model_rows = []
    yolo_root = tmp_path / "yolo"
    for split in ("train", "val", "test"):
        (yolo_root / "labels" / split).mkdir(parents=True)
        (yolo_root / "images" / split).mkdir(parents=True)
    for case in cases:
        yolo_split = "test" if case.gt_label else "val"
        for view in ZS32_VIEWS:
            model_rows.append(
                {
                    "part_id": case.part_id,
                    "hand": case.hand,
                    "view": view,
                    "branch": "yolo",
                    "raw_score": "0.8" if case.gt_label else "0.0",
                    "gt_label": case.gt_label,
                    "split": case.split,
                    "model_version": "yolo-v1",
                    "roi_version": "yolo-roi-v1",
                },
            )
            stem = case.images[view].stem
            label = yolo_root / "labels" / yolo_split / f"{stem}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("0 0.5 0.5 0.2 0.2\n" if case.gt_label else "", encoding="utf-8")
            (yolo_root / "images" / yolo_split / f"{stem}.png").write_bytes(b"roi")
    model_csv = tmp_path / "model_rows.csv"
    _write_csv(model_csv, CALIBRATION_FIELDS, model_rows)
    output = tmp_path / "yolo_rows.csv"

    count = write_yolo_annotation_calibration_rows(cases, model_csv, yolo_root, output)

    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert count == len(cases) * len(ZS32_VIEWS)
    assert {row["split"] for row in rows} == {"calibration", "test"}
    assert {row["gt_label"] for row in rows} == {"0", "1"}
    assert all(row["part_id"].endswith(f"::{row['view']}") for row in rows)


def test_yolo_annotation_calibration_rejects_one_part_spanning_splits(tmp_path: Path) -> None:
    """All eight views of a physical part must stay in one YOLO split."""
    crop_manifest, template_csv, root = _manifests(tmp_path)
    cases = load_offline_cases(crop_manifest, template_csv, path_root=root, hand="right")
    model_rows = []
    yolo_root = tmp_path / "yolo"
    for split in ("train", "val", "test"):
        (yolo_root / "labels" / split).mkdir(parents=True)
        (yolo_root / "images" / split).mkdir(parents=True)
    for case in cases:
        for index, view in enumerate(ZS32_VIEWS):
            model_rows.append(
                {
                    "part_id": case.part_id,
                    "hand": case.hand,
                    "view": view,
                    "branch": "yolo",
                    "raw_score": "0.1",
                    "gt_label": case.gt_label,
                    "split": case.split,
                    "model_version": "yolo-v1",
                    "roi_version": "roi-v1",
                },
            )
            split = "train" if case == cases[0] and index == 0 else ("test" if case.gt_label else "val")
            stem = case.images[view].stem
            (yolo_root / "labels" / split / f"{stem}.txt").write_text("", encoding="utf-8")
            (yolo_root / "images" / split / f"{stem}.png").write_bytes(b"roi")
    model_csv = tmp_path / "model_rows.csv"
    _write_csv(model_csv, CALIBRATION_FIELDS, model_rows)

    with pytest.raises(ValueError, match="spans YOLO splits"):
        write_yolo_annotation_calibration_rows(cases, model_csv, yolo_root, tmp_path / "rows.csv")


def test_yolo_annotation_calibration_rejects_record_missing_one_view(tmp_path: Path) -> None:
    """An auxiliary YOLO publication cannot silently omit a canonical view score."""
    crop_manifest, template_csv, root = _manifests(tmp_path)
    cases = load_offline_cases(crop_manifest, template_csv, path_root=root, hand="right")
    yolo_root = tmp_path / "yolo"
    for split in ("train", "val", "test"):
        (yolo_root / "labels" / split).mkdir(parents=True)
        (yolo_root / "images" / split).mkdir(parents=True)
    model_rows = []
    for case in cases:
        split = "test" if case.gt_label else "val"
        for view in ZS32_VIEWS:
            stem = case.images[view].stem
            (yolo_root / "labels" / split / f"{stem}.txt").write_text("", encoding="utf-8")
            (yolo_root / "images" / split / f"{stem}.png").write_bytes(b"roi")
            model_rows.append(
                {
                    "part_id": case.part_id,
                    "hand": case.hand,
                    "view": view,
                    "branch": "yolo",
                    "raw_score": "0.1",
                    "gt_label": case.gt_label,
                    "split": case.split,
                    "model_version": "yolo-v1",
                    "roi_version": "roi-v1",
                },
            )
    model_rows = [
        row
        for row in model_rows
        if not (row["part_id"] == cases[0].part_id and row["view"] == "back_secondary")
    ]
    model_csv = tmp_path / "model_rows.csv"
    _write_csv(model_csv, CALIBRATION_FIELDS, model_rows)

    with pytest.raises(ValueError, match=r"missing YOLO score.*back_secondary"):
        write_yolo_annotation_calibration_rows(cases, model_csv, yolo_root, tmp_path / "rows.csv")


def _sha256(path: Path) -> str:
    """Hash one reuse fixture artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_completed_case(case: object, case_dir: Path, runtime_sha256: str) -> None:
    """Write one internally consistent runtime publication for reuse validation."""
    case_dir.mkdir(parents=True)
    source_images = {view: str(case.images[view]) for view in ZS32_VIEWS}
    manifest_views = {}
    calibration_rows = []
    evidence_rows = {"patchcore": [], "yolo": []}
    for family in ("patchcore", "yolo"):
        for view in ZS32_VIEWS:
            branch = f"anomaly_{view}" if family == "patchcore" else "yolo"
            model_version = f"{family}-{view}-v1" if family == "patchcore" else "yolo-v1"
            roi_version = f"{family}-roi-v1"
            evidence = case_dir / "evidence" / family / f"{view}.png"
            evidence.parent.mkdir(parents=True, exist_ok=True)
            evidence.write_bytes(f"{family}-{view}".encode())
            row = {
                "part_id": case.part_id,
                "hand": case.hand,
                "view": view,
                "branch": branch,
                "score": "0.2",
                "source_path": str(case.images[view]),
                "source_hash": _sha256(case.images[view]),
                "evidence_path": str(evidence),
                "evidence_hash": _sha256(evidence),
                "model_version": model_version,
                "roi_version": roi_version,
            }
            evidence_rows[family].append(row)
            calibration_rows.append(
                {
                    "part_id": case.part_id,
                    "hand": case.hand,
                    "view": view,
                    "branch": branch,
                    "raw_score": "0.2",
                    "gt_label": case.gt_label,
                    "split": case.split,
                    "model_version": model_version,
                    "roi_version": roi_version,
                },
            )
            manifest_views.setdefault(view, {})[family] = {"score": 0.2, "evidence_path": str(evidence)}
    for family, rows in evidence_rows.items():
        _write_csv(case_dir / f"{family}.csv", tuple(rows[0]), rows)
    _write_csv(case_dir / "calibration_rows.csv", CALIBRATION_FIELDS, calibration_rows)
    (case_dir / "runtime_manifest.json").write_text(
        json.dumps(
            {
                "part_id": case.part_id,
                "capture_session": case.session_id,
                "group_id": case.group_id,
                "hand": case.hand,
                "errors": [],
                "runtime_config_sha256": runtime_sha256,
                "source_images": source_images,
                "views": manifest_views,
            },
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("mutation", ["missing_manifest", "raw_score", "model_version", "roi_version"])
def test_completed_runtime_case_rejects_reuse_drift(tmp_path: Path, mutation: str) -> None:
    """Reuse must reject incomplete or altered model evidence before aggregation."""
    crop_manifest, template_csv, root = _manifests(tmp_path)
    case = load_offline_cases(crop_manifest, template_csv, path_root=root, hand="right")[0]
    case_dir = tmp_path / "cases" / case.slug
    runtime_sha256 = "a" * 64
    _write_completed_case(case, case_dir, runtime_sha256)
    if mutation == "missing_manifest":
        (case_dir / "runtime_manifest.json").unlink()
    else:
        rows = list(csv.DictReader((case_dir / "calibration_rows.csv").open(encoding="utf-8")))
        rows[0][mutation] = "0.9" if mutation == "raw_score" else f"tampered-{mutation}"
        _write_csv(case_dir / "calibration_rows.csv", CALIBRATION_FIELDS, rows)

    assert not _completed_runtime_case(case, case_dir, runtime_config_sha256=runtime_sha256)


def test_write_case_index_rejects_existing_drift(tmp_path: Path) -> None:
    """Resume must compare the full canonical source index instead of trusting existence."""
    crop_manifest, template_csv, root = _manifests(tmp_path)
    cases = load_offline_cases(crop_manifest, template_csv, path_root=root, hand="right")
    output = tmp_path / "case_index.csv"
    write_case_index(cases, output)
    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    rows[0]["front_source_path"] = str(tmp_path / "substituted.png")
    _write_csv(output, tuple(rows[0]), rows)

    with pytest.raises(ValueError, match="case index differs"):
        write_case_index(cases, output)
