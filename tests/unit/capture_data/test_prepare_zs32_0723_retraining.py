# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the immutable ZS32 0723 retraining-data release builder."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "capture_data" / "prepare_zs32_0723_retraining.py"
WRAPPER_PATH = REPO_ROOT / "pipeline" / "prepare_zs32_0723_retraining.py"
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


def _load_module():
    assert MODULE_PATH.is_file(), "the retraining release builder has not been implemented"
    spec = importlib.util.spec_from_file_location("prepare_zs32_0723_retraining", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


@pytest.fixture()
def source_manifests(tmp_path: Path) -> tuple[Path, Path, Path]:
    new_rows: list[dict[str, str]] = []
    for group_number in range(1, 107):
        group = f"group{group_number:03d}"
        for view in VIEWS:
            image = tmp_path / "new_images" / f"{group}_{view}.png"
            image.parent.mkdir(exist_ok=True)
            image.write_bytes(f"new:{group}:{view}".encode())
            new_rows.append(
                {
                    "source_path": str(image),
                    "output_path": str(image),
                    "hand": "right",
                    "resolved_view": view,
                    "label": "normal",
                    "defect_type": "",
                    "session_id": "zs32_4cam_accept_20260723_163955_301282113",
                }
            )
    new_manifest = tmp_path / "new_crop_manifest.csv"
    _write_csv(new_manifest, list(new_rows[0]), new_rows)
    (tmp_path / "roi_config.json").write_text(
        json.dumps({"schema_version": 1, "views": {view: {"roi": [0, 0, 10, 10]} for view in VIEWS}}),
        encoding="utf-8",
    )

    old_rows: list[dict[str, str]] = []
    for view in VIEWS:
        for label in ("normal", "defect"):
            image = tmp_path / "old_patchcore" / f"{view}_{label}.png"
            image.parent.mkdir(exist_ok=True)
            image.write_bytes(f"old-patchcore:{view}:{label}".encode())
            old_rows.append(
                {
                    "source_path": str(image),
                    "output_path": str(image),
                    "hand": "right",
                    "resolved_view": view,
                    "label": label,
                    "defect_type": "less" if label == "defect" else "",
                    "session_id": "legacy_session",
                }
            )
    old_patchcore = tmp_path / "old_patchcore_manifest.csv"
    _write_csv(old_patchcore, list(old_rows[0]), old_rows)

    yolo_rows: list[dict[str, str]] = []
    for split, label_text in (("train", "0 0.5 0.5 0.2 0.2\n"), ("val", ""), ("test", "0 0.4 0.4 0.1 0.1\n")):
        image = tmp_path / "old_yolo" / "images" / split / f"{split}.png"
        label = tmp_path / "old_yolo" / "labels" / split / f"{split}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(f"old-yolo:{split}".encode())
        label.write_text(label_text, encoding="utf-8")
        yolo_rows.append(
            {
                "source_image_path": str(image),
                "new_image_path": str(image),
                "new_label_path": str(label),
                "sample_id": f"legacy/{split}",
                "view": "front",
                "hand": "left" if split == "train" else "right",
                "defect_type": "less" if label_text else "",
                "session_id": "legacy_yolo",
                "group_id": split,
                "split": split,
                "source_kind": "defect_positive" if label_text else "trusted_normal",
                "annotation_status": "label_studio_positive" if label_text else "normal_directory_confirmed_empty",
                "pixel_sha256": f"legacy-{split}",
            }
        )
    old_yolo = tmp_path / "old_yolo_mapping.csv"
    _write_csv(old_yolo, list(yolo_rows[0]), yolo_rows)
    return new_manifest, old_patchcore, old_yolo


def test_prepare_builds_deterministic_four_way_release(
    source_manifests: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    module = _load_module()
    new_manifest, old_patchcore, old_yolo = source_manifests
    output = tmp_path / "release"

    summary = module.prepare_release(
        new_crop_manifest=new_manifest,
        old_patchcore_manifest=old_patchcore,
        old_yolo_mapping=old_yolo,
        output_root=output,
        seed=42,
    )

    splits = _read_csv(output / "physical_part_splits.csv")
    assert Counter(row["split"] for row in splits) == {
        "train": 64,
        "model_val": 16,
        "calibration": 13,
        "final_test": 13,
    }
    split_by_part = {row["physical_part_id"]: row["split"] for row in splits}
    assert len(split_by_part) == 106
    assert summary["new_physical_parts"] == 106
    assert summary["new_images"] == 848

    # Every new physical part keeps all eight views in exactly one release split.
    release_rows = _read_csv(output / "release_manifest.csv")
    new_rows = [row for row in release_rows if row["source_origin"] == "zs32_0723"]
    splits_by_part: dict[str, set[str]] = defaultdict(set)
    for row in new_rows:
        splits_by_part[row["physical_part_id"]].add(row["release_split"])
    assert all(len(values) == 1 for values in splits_by_part.values())
    assert {part: next(iter(values)) for part, values in splits_by_part.items()} == split_by_part

    # Six primary views contain old+new; secondary views contain only 0723.
    patchcore = _read_csv(output / "patchcore_manifest.csv")
    for view in VIEWS:
        origins = {row["source_origin"] for row in patchcore if row["view"] == view}
        expected = {"zs32_0723"} if view.endswith("_secondary") else {"legacy", "zs32_0723"}
        assert origins == expected
    assert {row["target_bucket"] for row in patchcore if row["source_origin"] == "legacy"} == {
        "normal",
        "defect",
    }
    legacy_defects = [
        row
        for row in patchcore
        if row["source_origin"] == "legacy" and row["release_split"] == "retrospective_test"
    ]
    assert legacy_defects
    assert all("/defect/" in Path(row["target_path"]).as_posix() for row in legacy_defects)
    assert all(
        f"/{row['session_id']}/images/" in Path(row["target_path"]).as_posix()
        for row in patchcore
    )
    new_buckets = {
        row["release_split"]: row["target_bucket"]
        for row in patchcore
        if row["source_origin"] == "zs32_0723"
    }
    assert new_buckets == {
        "train": "normal",
        "model_val": "holdout/model_val",
        "calibration": "normal_test",
        "final_test": "holdout/final_test",
    }
    assert all(Path(row["target_path"]).is_symlink() for row in patchcore)
    trainer_manifest = _read_csv(output / "patchcore" / "crop_manifest.csv")
    assert len(trainer_manifest) == len(patchcore)
    assert {"output_path", "label", "resolved_view", "session_id"} <= trainer_manifest[0].keys()

    template = _read_csv(output / "template_manifest.csv")
    assert {row["source_origin"] for row in template} == {"zs32_0723"}
    assert {row["view"] for row in template} == set(VIEWS)
    assert len(template) == 848
    assert {
        (row["release_split"], row["split"])
        for row in template
    } == {
        ("train", "train"),
        ("model_val", "model_val"),
        ("calibration", "calibration"),
        ("final_test", "final_test"),
    }


def test_yolo_preserves_legacy_labels_and_separates_new_holdouts(
    source_manifests: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    module = _load_module()
    new_manifest, old_patchcore, old_yolo = source_manifests
    output = tmp_path / "release"
    old_rows = _read_csv(old_yolo)

    module.prepare_release(
        new_crop_manifest=new_manifest,
        old_patchcore_manifest=old_patchcore,
        old_yolo_mapping=old_yolo,
        output_root=output,
    )

    mapping = _read_csv(output / "yolo" / "mapping.csv")
    legacy = [row for row in mapping if row["source_origin"] == "legacy"]
    assert Counter(row["split"] for row in legacy) == {"train": 1, "val": 1, "retrospective_test": 1}
    source_label_bytes = {
        row["split"]: Path(row["new_label_path"]).read_bytes()
        for row in old_rows
    }
    target_label_bytes = {
        row["original_split"]: Path(row["new_label_path"]).read_bytes()
        for row in legacy
    }
    assert target_label_bytes == source_label_bytes

    new_rows = [row for row in mapping if row["source_origin"] == "zs32_0723"]
    assert Counter(row["split"] for row in new_rows).keys() == {
        "train",
        "val",
        "calibration",
        "final_test",
    }
    assert all(Path(row["new_label_path"]).read_bytes() == b"" for row in new_rows)
    assert all(Path(row["new_image_path"]).is_symlink() for row in new_rows)
    yaml_text = (output / "yolo" / "data.yaml").read_text(encoding="utf-8")
    assert "train: images/train" in yaml_text
    assert "val: images/val" in yaml_text
    assert "test: images/retrospective_test" in yaml_text
    assert "images/calibration" not in yaml_text
    assert "images/final_test" not in yaml_text


def test_prepare_is_dry_run_safe_and_refuses_overwrite(
    source_manifests: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    module = _load_module()
    new_manifest, old_patchcore, old_yolo = source_manifests
    output = tmp_path / "release"

    dry_summary = module.prepare_release(
        new_crop_manifest=new_manifest,
        old_patchcore_manifest=old_patchcore,
        old_yolo_mapping=old_yolo,
        output_root=output,
        dry_run=True,
    )
    assert dry_summary["dry_run"] is True
    assert not output.exists()

    module.prepare_release(
        new_crop_manifest=new_manifest,
        old_patchcore_manifest=old_patchcore,
        old_yolo_mapping=old_yolo,
        output_root=output,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.prepare_release(
            new_crop_manifest=new_manifest,
            old_patchcore_manifest=old_patchcore,
            old_yolo_mapping=old_yolo,
            output_root=output,
        )


def test_validate_detects_split_hash_leakage_and_cli_has_subcommands(
    source_manifests: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    module = _load_module()
    new_manifest, old_patchcore, old_yolo = source_manifests
    output = tmp_path / "release"
    module.prepare_release(
        new_crop_manifest=new_manifest,
        old_patchcore_manifest=old_patchcore,
        old_yolo_mapping=old_yolo,
        output_root=output,
    )

    validation = module.validate_release(output)
    assert validation["status"] == "VALID"
    assert validation["physical_part_count"] == 106

    rows = _read_csv(output / "release_manifest.csv")
    victim = next(row for row in rows if row["source_origin"] == "zs32_0723")
    original_split = victim["release_split"]
    victim["release_split"] = "final_test" if original_split != "final_test" else "train"
    _write_csv(output / "release_manifest.csv", list(rows[0]), rows)
    with pytest.raises(ValueError, match="physical-part split leakage"):
        module.validate_release(output)

    result = subprocess.run(
        [sys.executable, str(WRAPPER_PATH), "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "{prepare,validate}" in result.stdout


def test_release_writes_machine_readable_summary_and_sha_receipts(
    source_manifests: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    module = _load_module()
    new_manifest, old_patchcore, old_yolo = source_manifests
    output = tmp_path / "release"
    module.prepare_release(
        new_crop_manifest=new_manifest,
        old_patchcore_manifest=old_patchcore,
        old_yolo_mapping=old_yolo,
        output_root=output,
    )

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["seed"] == 42
    assert summary["materialization"] == "symlink"
    source_roi = new_manifest.parent / "roi_config.json"
    assert (output / "roi_config.json").read_bytes() == source_roi.read_bytes()
    assert (output / "patchcore" / "roi_config.json").read_bytes() == source_roi.read_bytes()
    receipt_rows = _read_csv(output / "sha256_receipts.csv")
    assert {
        "physical_part_splits.csv",
        "release_manifest.csv",
        "patchcore_manifest.csv",
        "patchcore/crop_manifest.csv",
        "template_manifest.csv",
        "yolo/mapping.csv",
        "yolo/data.yaml",
        "roi_config.json",
        "patchcore/roi_config.json",
    } <= {row["relative_path"] for row in receipt_rows}
