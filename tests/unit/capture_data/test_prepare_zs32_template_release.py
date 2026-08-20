# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the immutable ZS32 0727 Template release builder."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "capture_data" / "prepare_zs32_template_release.py"
WRAPPER_PATH = REPO_ROOT / "pipeline" / "prepare_zs32_template_release.py"
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
COUNTS = {"train": 11, "model_val": 3, "calibration": 3, "final_test": 2}


def _load_module():
    assert MODULE_PATH.is_file(), "the Template release builder has not been implemented"
    spec = importlib.util.spec_from_file_location("prepare_zs32_template_release", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


@pytest.fixture()
def crop_manifest(tmp_path: Path) -> Path:
    rows: list[dict[str, str]] = []
    repo_root = tmp_path / "repo"
    manifest_root = repo_root / "dataset" / "zs32_0727_template_roi_v1"
    raw_root = repo_root / "dataset" / "zs32_0727"
    session = "zs32_4cam_accept_20260727_0727"
    for group_number in range(1, 20):
        group = f"group{group_number:03d}"
        for view in VIEWS:
            raw = raw_root / "right" / view / "normal" / session / "images" / f"right_{view}_normal_{group}.png"
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_bytes(f"raw-0727:{group}:{view}".encode())
            image = manifest_root / "crops" / session / group / f"{group}_{view}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(f"0727:{group}:{view}".encode())
            rows.append(
                {
                    "source_path": (
                        f"dataset/zs32_0727/right/{view}/normal/{session}/images/"
                        f"right_{view}_normal_{group}.png"
                    ),
                    "output_path": image.relative_to(repo_root).as_posix(),
                    "hand": "right",
                    "source_view": view,
                    "resolved_view": view,
                    "view_corrected": "false",
                    "label": "normal",
                    "defect_type": "",
                    "session_id": session,
                }
            )
    manifest = manifest_root / "crop_manifest.csv"
    _write_csv(manifest, list(rows[0]), rows)
    (manifest_root / "roi_config.json").write_bytes(b'{"roi_version":"test-roi","views":{}}\n')
    return manifest


@pytest.fixture()
def module(crop_manifest: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    module = _load_module()
    repo_root = crop_manifest.parents[2]
    monkeypatch.setattr(module, "REPO_ROOT", repo_root)
    monkeypatch.setattr(
        module,
        "ZS32_0727_ROI_CONFIG_SHA256",
        hashlib.sha256((crop_manifest.parent / "roi_config.json").read_bytes()).hexdigest(),
        raising=False,
    )
    return module


def _prepare(module: object, manifest: Path, output: Path, **kwargs: object) -> dict[str, object]:
    return module.prepare_template_release(
        crop_manifest=manifest,
        output_root=output,
        source_name=kwargs.pop("source_name", "zs32_0727"),
        expected_parts=19,
        train_count=11,
        model_val_count=3,
        calibration_count=3,
        final_test_count=2,
        seed=42,
        **kwargs,
    )


def test_prepare_builds_deterministic_part_isolated_template_release(module: object, crop_manifest: Path, tmp_path: Path) -> None:
    output = tmp_path / "release"

    summary = _prepare(module, crop_manifest, output)

    assert summary["physical_part_count"] == 19
    assert summary["image_count"] == 152
    assert summary["role_counts"] == COUNTS
    splits = _read_csv(output / "physical_part_splits.csv")
    assert Counter(row["role"] for row in splits) == COUNTS
    assert len({row["physical_part_id"] for row in splits}) == 19

    manifest = _read_csv(output / "template_manifest.csv")
    assert len(manifest) == 152
    assert {row["view"] for row in manifest} == set(VIEWS)
    roles_by_part: dict[str, set[str]] = defaultdict(set)
    views_by_part: dict[str, set[str]] = defaultdict(set)
    for row in manifest:
        roles_by_part[row["physical_part_id"]].add(row["role"])
        views_by_part[row["physical_part_id"]].add(row["view"])
        assert Path(row["image_path"]).is_symlink()
        assert Path(row["image_path"]).is_file()
    assert all(len(roles) == 1 for roles in roles_by_part.values())
    assert all(views == set(VIEWS) for views in views_by_part.values())

    assert (output / "roi_config.json").read_bytes() == (crop_manifest.parent / "roi_config.json").read_bytes()
    receipts = _read_csv(output / "sha256_receipts.csv")
    receipt_by_path = {row["relative_path"]: row["sha256"] for row in receipts}
    assert {"physical_part_splits.csv", "template_manifest.csv", "roi_config.json", "summary.json"} <= set(
        receipt_by_path,
    )
    for relative_path, digest in receipt_by_path.items():
        assert hashlib.sha256((output / relative_path).read_bytes()).hexdigest() == digest


def test_prepare_rejects_incomplete_part_cross_part_duplicate_and_wrong_contract(
    module: object,
    crop_manifest: Path,
    tmp_path: Path,
) -> None:
    rows = _read_csv(crop_manifest)

    _write_csv(crop_manifest, list(rows[0]), rows[:-1])
    with pytest.raises(ValueError, match="all eight canonical views"):
        _prepare(module, crop_manifest, tmp_path / "incomplete-release")

    first = rows[0]
    second = next(row for row in rows if "group002" in row["output_path"])
    (module.REPO_ROOT / second["output_path"]).write_bytes((module.REPO_ROOT / first["output_path"]).read_bytes())
    _write_csv(crop_manifest, list(rows[0]), rows)
    with pytest.raises(ValueError, match="encoded image is duplicated"):
        _prepare(module, crop_manifest, tmp_path / "duplicate-release")

    (module.REPO_ROOT / second["output_path"]).write_bytes(b"0727:group002:front")

    invalid = [dict(row) for row in rows]
    invalid[0]["label"] = "defect"
    _write_csv(crop_manifest, list(rows[0]), invalid)
    with pytest.raises(ValueError, match="normal"):
        _prepare(module, crop_manifest, tmp_path / "wrong-label-release")

    invalid[0]["label"] = "normal"
    invalid[0]["hand"] = "left"
    _write_csv(crop_manifest, list(rows[0]), invalid)
    with pytest.raises(ValueError, match="right-hand"):
        _prepare(module, crop_manifest, tmp_path / "wrong-hand-release")


def test_prepare_rejects_zs32_0727_rows_from_another_dataset_or_group_range(
    module: object,
    crop_manifest: Path,
    tmp_path: Path,
) -> None:
    rows = _read_csv(crop_manifest)

    wrong_dataset = [dict(row) for row in rows]
    for row in wrong_dataset:
        row["source_path"] = row["source_path"].replace("dataset/zs32_0727/", "dataset/zs32_0723/")
    _write_csv(crop_manifest, list(rows[0]), wrong_dataset)
    with pytest.raises(ValueError, match="dataset/zs32_0727"):
        _prepare(module, crop_manifest, tmp_path / "wrong-dataset-release")

    wrong_groups = [dict(row) for row in rows]
    for row in wrong_groups:
        for old_number, new_number in zip(range(1, 20), range(20, 39), strict=True):
            old_group = f"group{old_number:03d}"
            if old_group in row["source_path"]:
                row["source_path"] = row["source_path"].replace(old_group, f"group{new_number:03d}")
                break
    _write_csv(crop_manifest, list(rows[0]), wrong_groups)
    with pytest.raises(ValueError, match="group001.*group019"):
        _prepare(module, crop_manifest, tmp_path / "wrong-groups-release")


def test_prepare_resolves_stage30_repo_relative_assets_outside_repo_cwd(
    module: object,
    crop_manifest: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    _prepare(module, crop_manifest.resolve(), tmp_path / "release")


def test_prepare_validates_staging_before_publishing_final_root(
    module: object,
    crop_manifest: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "release"

    def reject_staged_release(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise ValueError("forced staged validation failure")

    monkeypatch.setattr(module, "validate_template_release", reject_staged_release)
    with pytest.raises(ValueError, match="forced staged validation failure"):
        _prepare(module, crop_manifest, output)
    assert not output.exists()


def test_prepare_is_dry_run_safe_refuses_overwrite_and_validation_fails_closed(
    module: object,
    crop_manifest: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "release"

    dry_run = _prepare(module, crop_manifest, output, dry_run=True)
    assert dry_run["status"] == "DRY_RUN"
    assert not output.exists()

    _prepare(module, crop_manifest, output)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _prepare(module, crop_manifest, output)

    rows = _read_csv(output / "template_manifest.csv")
    rows[0]["role"] = "final_test" if rows[0]["role"] != "final_test" else "train"
    _write_csv(output / "template_manifest.csv", list(rows[0]), rows)
    with pytest.raises(ValueError, match="part-role leakage"):
        module.validate_template_release(output)


def test_validate_rejects_receipt_drift_and_wrapper_exposes_subcommands(
    module: object,
    crop_manifest: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "release"
    _prepare(module, crop_manifest, output)

    validation = module.validate_template_release(output)
    assert validation["status"] == "VALID"
    assert validation["physical_part_count"] == 19
    (output / "roi_config.json").write_bytes(b"drift")
    with pytest.raises(ValueError, match="ROI config SHA256|SHA256 receipt mismatch"):
        module.validate_template_release(output)

    result = subprocess.run(
        [sys.executable, str(WRAPPER_PATH), "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "{prepare,validate}" in result.stdout


def test_prepare_rejects_non_0727_source_name(module: object, crop_manifest: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source_name must be exactly 'zs32_0727'"):
        _prepare(module, crop_manifest, tmp_path / "release", source_name="zs32_0723")


def test_prepare_rejects_nominal_0727_source_with_foreign_output(
    module: object,
    crop_manifest: Path,
    tmp_path: Path,
) -> None:
    rows = _read_csv(crop_manifest)
    foreign = tmp_path / "foreign" / "group001_front.png"
    foreign.parent.mkdir(parents=True)
    foreign.write_bytes(b"foreign crop")
    rows[0]["output_path"] = str(foreign)
    _write_csv(crop_manifest, list(rows[0]), rows)

    with pytest.raises(ValueError, match="output_path must be under"):
        _prepare(module, crop_manifest, tmp_path / "release")


def test_prepare_rejects_missing_raw_0727_source(module: object, crop_manifest: Path, tmp_path: Path) -> None:
    rows = _read_csv(crop_manifest)
    raw = module.REPO_ROOT / rows[0]["source_path"]
    raw.unlink()

    with pytest.raises(FileNotFoundError, match="source_path does not exist"):
        _prepare(module, crop_manifest, tmp_path / "release")


def test_prepare_rejects_raw_and_output_group_mismatch(module: object, crop_manifest: Path, tmp_path: Path) -> None:
    rows = _read_csv(crop_manifest)
    rows[0]["output_path"] = rows[0]["output_path"].replace("group001", "group002")
    _write_csv(crop_manifest, list(rows[0]), rows)

    with pytest.raises(ValueError, match="output_path group identity does not match source_path"):
        _prepare(module, crop_manifest, tmp_path / "release")


def test_prepare_rejects_roi_config_with_wrong_pinned_sha256(
    module: object,
    crop_manifest: Path,
    tmp_path: Path,
) -> None:
    (crop_manifest.parent / "roi_config.json").write_bytes(b'{"roi_version":"foreign"}\n')

    with pytest.raises(ValueError, match="ROI config SHA256 does not match pinned 0727 contract"):
        _prepare(module, crop_manifest, tmp_path / "release")


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("expected_parts", 18),
        ("train_count", 12),
        ("model_val_count", 2),
        ("calibration_count", 4),
        ("final_test_count", 1),
        ("seed", 7),
    ],
)
def test_prepare_rejects_noncanonical_part_split_contract(
    module: object,
    crop_manifest: Path,
    tmp_path: Path,
    override: str,
    value: int,
) -> None:
    arguments = {
        "crop_manifest": crop_manifest,
        "output_root": tmp_path / "release",
        "source_name": "zs32_0727",
        "expected_parts": 19,
        "train_count": 11,
        "model_val_count": 3,
        "calibration_count": 3,
        "final_test_count": 2,
        "seed": 42,
    }
    arguments[override] = value

    with pytest.raises(ValueError, match="fixed 19-part seed-42 split"):
        module.prepare_template_release(**arguments)


def test_prepare_rejects_duplicate_encoded_images_within_one_part(
    module: object,
    crop_manifest: Path,
    tmp_path: Path,
) -> None:
    rows = _read_csv(crop_manifest)
    first, second = rows[0], rows[1]
    first_output = module.REPO_ROOT / first["output_path"]
    second_output = module.REPO_ROOT / second["output_path"]
    second_output.write_bytes(first_output.read_bytes())

    with pytest.raises(ValueError, match="encoded image is duplicated"):
        _prepare(module, crop_manifest, tmp_path / "release")


def test_validate_rejects_recomputed_receipts_for_noncanonical_seed(
    module: object,
    crop_manifest: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "release"
    _prepare(module, crop_manifest, output)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["seed"] = 7
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipts = _read_csv(output / "sha256_receipts.csv")
    next(row for row in receipts if row["relative_path"] == "summary.json")["sha256"] = hashlib.sha256(
        summary_path.read_bytes()
    ).hexdigest()
    _write_csv(output / "sha256_receipts.csv", list(receipts[0]), receipts)

    with pytest.raises(ValueError, match="fixed 19-part seed-42 split"):
        module.validate_template_release(output)
