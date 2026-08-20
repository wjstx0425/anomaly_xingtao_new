# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ZS32 0727 Template/PatchCore v14 release builder."""

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
MODULE_PATH = REPO_ROOT / "capture_data" / "prepare_zs32_0727_template_patchcore.py"
WRAPPER_PATH = REPO_ROOT / "pipeline" / "prepare_zs32_0727_template_patchcore.py"
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
DUPLICATE_DEFECT_SESSION = "zs32_right_deform_20260714_201404_438451409"


def _load_module():
    assert MODULE_PATH.is_file(), "the 0727 Template/PatchCore release builder is missing"
    spec = importlib.util.spec_from_file_location("prepare_zs32_0727_template_patchcore", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture()
def crop_manifests(tmp_path: Path) -> tuple[Path, Path]:
    roi_payload = {
        "schema_version": 1,
        "coordinate_system": "pixel_xyxy_half_open",
        "image_size": {"width": 4024, "height": 3036},
        "views": {view: {"roi": [0, 0, 16, 16]} for view in VIEWS},
    }
    roi_text = json.dumps(roi_payload, sort_keys=True)

    normal_rows: list[dict[str, str]] = []
    for session, count in (("normal_session_a", 20), ("normal_session_b", 84)):
        for number in range(1, count + 1):
            group = f"group{number:03d}"
            for view in VIEWS:
                image = tmp_path / "normal_crops" / session / f"{group}_{view}.png"
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(f"normal:{session}:{group}:{view}".encode())
                normal_rows.append(
                    {
                        "source_path": str(image),
                        "output_path": str(image),
                        "hand": "right",
                        "source_view": view,
                        "resolved_view": view,
                        "label": "normal",
                        "defect_type": "",
                        "session_id": session,
                    }
                )
    normal_manifest = tmp_path / "normal_roi" / "crop_manifest.csv"
    _write_csv(normal_manifest, normal_rows)
    normal_manifest.with_name("roi_config.json").write_text(roi_text, encoding="utf-8")

    defect_rows: list[dict[str, str]] = []
    defect_specs = (
        ("defect_deform", "deform", 12),
        ("defect_less", "less", 2),
        ("defect_others", "others", 9),
    )
    for session, defect_type, count in defect_specs:
        for number in range(1, count + 1):
            group = f"group{number:03d}"
            for view in VIEWS:
                image = tmp_path / "defect_crops" / defect_type / session / f"{group}_{view}.png"
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(f"defect:{session}:{group}:{view}".encode())
                defect_rows.append(
                    {
                        "source_path": str(image),
                        "output_path": str(image),
                        "hand": "right",
                        "source_view": view,
                        "resolved_view": view,
                        "label": "defect",
                        "defect_type": defect_type,
                        "session_id": session,
                    }
                )

    # These five groups are known duplicates and must be filtered by session identity.
    for number in range(1, 6):
        group = f"group{number:03d}"
        for view in VIEWS:
            image = tmp_path / "defect_crops" / "deform" / DUPLICATE_DEFECT_SESSION / f"{group}_{view}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(f"duplicate:{group}:{view}".encode())
            defect_rows.append(
                {
                    "source_path": str(image),
                    "output_path": str(image),
                    "hand": "right",
                    "source_view": view,
                    "resolved_view": view,
                    "label": "defect",
                    "defect_type": "deform",
                    "session_id": DUPLICATE_DEFECT_SESSION,
                }
            )
    defect_manifest = tmp_path / "defect_roi" / "crop_manifest.csv"
    _write_csv(defect_manifest, defect_rows)
    defect_manifest.with_name("roi_config.json").write_text(roi_text, encoding="utf-8")
    return normal_manifest, defect_manifest


def test_prepare_builds_isolated_template_and_patchcore_release(
    crop_manifests: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    module = _load_module()
    normal_manifest, defect_manifest = crop_manifests
    output = tmp_path / "release"

    summary = module.prepare_release(normal_manifest, defect_manifest, output, seed=42)

    assert summary["normal_parts"] == 104
    assert summary["defect_parts"] == 23
    assert summary["normal_images"] == 832
    assert summary["defect_images"] == 184
    splits = _read_csv(output / "physical_part_splits.csv")
    assert Counter((row["label"], row["role"]) for row in splits) == {
        ("normal", "train"): 62,
        ("normal", "model_val"): 16,
        ("normal", "calibration"): 13,
        ("normal", "final_test"): 13,
        ("defect", "model_val"): 6,
        ("defect", "calibration"): 11,
        ("defect", "final_test"): 6,
    }
    assert DUPLICATE_DEFECT_SESSION not in {row["session_id"] for row in splits}

    template = _read_csv(output / "template_manifest.csv")
    assert len(template) == 127 * 8
    assert Counter(row["label"] for row in template) == {"normal": 832, "defect": 184}
    assert all(Path(row["image_path"]).is_symlink() for row in template)
    views_by_part: dict[str, set[str]] = defaultdict(set)
    roles_by_part: dict[str, set[str]] = defaultdict(set)
    for row in template:
        views_by_part[row["physical_part_id"]].add(row["view"])
        roles_by_part[row["physical_part_id"]].add(row["split"])
    assert all(views == set(VIEWS) for views in views_by_part.values())
    assert all(len(roles) == 1 for roles in roles_by_part.values())
    assert not [
        row
        for row in template
        if row["label"] == "defect" and row["split"] == "train"
    ]

    patchcore = _read_csv(output / "patchcore_manifest.csv")
    assert len(patchcore) == len(template)
    assert Counter((row["label"], row["release_role"], row["target_bucket"]) for row in patchcore) == {
        ("normal", "train", "normal"): 62 * 8,
        ("normal", "model_val", "holdout/model_val"): 16 * 8,
        ("normal", "calibration", "normal_test"): 13 * 8,
        ("normal", "final_test", "holdout/final_test"): 13 * 8,
        ("defect", "model_val", "holdout/model_val"): 6 * 8,
        ("defect", "calibration", "defect"): 11 * 8,
        ("defect", "final_test", "holdout/final_test"): 6 * 8,
    }
    trainer_manifest = _read_csv(output / "patchcore" / "crop_manifest.csv")
    assert len(trainer_manifest) == (62 + 13 + 11) * 8
    assert Counter(row["label"] for row in trainer_manifest) == {
        "normal": 62 * 8,
        "normal_test": 13 * 8,
        "defect": 11 * 8,
    }
    assert all(Path(row["output_path"]).is_symlink() for row in trainer_manifest)
    assert module.validate_release(output)["status"] == "VALID"


def test_prepare_refuses_existing_output(
    crop_manifests: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    module = _load_module()
    normal_manifest, defect_manifest = crop_manifests
    output = tmp_path / "release"
    output.mkdir()

    with pytest.raises(FileExistsError):
        module.prepare_release(normal_manifest, defect_manifest, output)


def test_prepare_rejects_session_path_traversal(
    crop_manifests: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    module = _load_module()
    normal_manifest, defect_manifest = crop_manifests
    normal_rows = _read_csv(normal_manifest)
    original_session = normal_rows[0]["session_id"]
    original_group = Path(normal_rows[0]["output_path"]).name.split("_", 1)[0]
    for row in normal_rows:
        if (
            row["session_id"] == original_session
            and Path(row["output_path"]).name.startswith(original_group)
        ):
            row["session_id"] = "../escape"
    _write_csv(normal_manifest, normal_rows)

    with pytest.raises(ValueError, match="session_id"):
        module.prepare_release(normal_manifest, defect_manifest, tmp_path / "release")
    assert not (tmp_path / "escape").exists()


def test_validate_rejects_patchcore_route_tampering(
    crop_manifests: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    module = _load_module()
    normal_manifest, defect_manifest = crop_manifests
    output = tmp_path / "release"
    module.prepare_release(normal_manifest, defect_manifest, output)
    rows = _read_csv(output / "patchcore_manifest.csv")
    rows[0]["target_bucket"] = "defect"
    (output / "patchcore_manifest.csv").unlink()
    _write_csv(output / "patchcore_manifest.csv", rows)

    with pytest.raises(ValueError, match="PatchCore"):
        module.validate_release(output)


def test_validate_rejects_cross_manifest_provenance_tampering(
    crop_manifests: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    module = _load_module()
    normal_manifest, defect_manifest = crop_manifests
    output = tmp_path / "release"
    module.prepare_release(normal_manifest, defect_manifest, output)
    rows = _read_csv(output / "patchcore_manifest.csv")
    rows[0]["source_origin"] = "tampered_origin"
    (output / "patchcore_manifest.csv").unlink()
    _write_csv(output / "patchcore_manifest.csv", rows)

    with pytest.raises(ValueError, match="PatchCore provenance"):
        module.validate_release(output)


def test_failed_post_publish_validation_removes_invalid_release(
    crop_manifests: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    normal_manifest, defect_manifest = crop_manifests
    output = tmp_path / "release"
    monkeypatch.setattr(
        module,
        "validate_release",
        lambda _root: (_ for _ in ()).throw(ValueError("forced validation failure")),
    )

    with pytest.raises(ValueError, match="forced validation failure"):
        module.prepare_release(normal_manifest, defect_manifest, output)
    assert not output.exists()


def test_wrapper_help_is_available() -> None:
    result = subprocess.run(
        [sys.executable, str(WRAPPER_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "{prepare,validate}" in result.stdout
