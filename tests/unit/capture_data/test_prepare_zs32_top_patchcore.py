# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ZS32 front/back-only incremental PatchCore release."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "capture_data/prepare_zs32_top_patchcore.py"


def _load_module() -> ModuleType:
    assert MODULE_PATH.is_file(), "front/back release builder is not implemented"
    spec = importlib.util.spec_from_file_location("prepare_zs32_top_patchcore", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_image(path: Path, value: int, *, shape: tuple[int, int] = (8, 10)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((shape[0], shape[1], 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _source_fixture(root: Path) -> Path:
    source = root / "zs32_top"
    session = "zs32_4cam_accept_20260727_215628_492895805"
    fields = (
        "record_type",
        "session_id",
        "sample_id",
        "group_id",
        "image_index",
        "round",
        "view",
        "device_index",
        "camera_serial",
        "capture_mode",
        "exposure",
        "gain",
        "file",
        "source_short",
        "source_long",
        "short_exposure",
        "long_exposure",
        "hdr_attempt",
        "fused_clip_pct",
        "captured_at",
        "sample_status",
        "failed_round",
        "failed_view",
        "failed_device_index",
        "error",
    )
    rows: list[dict[str, str]] = []
    for group_number in range(1, 29):
        group = f"group{group_number:03d}"
        for view_index, view in enumerate(("front", "back")):
            image = (
                source
                / "right"
                / view
                / "normal"
                / session
                / "images"
                / f"right_{view}_normal_zs32_right_normal_accept_{group}_000001_fused.png"
            )
            _write_image(image, group_number + view_index)
            rows.append(
                {
                    "record_type": "image",
                    "session_id": session,
                    "sample_id": f"zs32_right_normal_accept_{group}_000001",
                    "group_id": group,
                    "image_index": "1",
                    "round": view,
                    "view": view,
                    "device_index": "1",
                    "camera_serial": "DA9805574",
                    "capture_mode": "hdr_fused",
                    "exposure": "",
                    "gain": "0.0",
                    "file": str(image),
                    "source_short": "",
                    "source_long": "",
                    "short_exposure": "1500.0",
                    "long_exposure": "5500.0",
                    "hdr_attempt": "1",
                    "fused_clip_pct": "0.0",
                    "captured_at": "2026-07-27T14:00:00Z",
                    "sample_status": "complete",
                    "failed_round": "",
                    "failed_view": "",
                    "failed_device_index": "",
                    "error": "",
                }
            )
        rows.append(
            {
                "record_type": "sample",
                "session_id": session,
                "sample_id": f"zs32_right_normal_accept_{group}_000001",
                "group_id": group,
                "image_index": "1",
                "round": "",
                "view": "",
                "device_index": "",
                "camera_serial": "",
                "capture_mode": "hdr_fused",
                "exposure": "",
                "gain": "",
                "file": "",
                "source_short": "",
                "source_long": "",
                "short_exposure": "",
                "long_exposure": "",
                "hdr_attempt": "",
                "fused_clip_pct": "",
                "captured_at": "",
                "sample_status": "complete",
                "failed_round": "",
                "failed_view": "",
                "failed_device_index": "",
                "error": "",
            }
        )
    rows.append(
        {
            "record_type": "sample",
            "session_id": session,
            "sample_id": "zs32_right_normal_accept_group029_000001",
            "group_id": "group029",
            "image_index": "1",
            "round": "",
            "view": "",
            "device_index": "",
            "camera_serial": "",
            "capture_mode": "hdr_fused",
            "exposure": "",
            "gain": "",
            "file": "",
            "source_short": "",
            "source_long": "",
            "short_exposure": "",
            "long_exposure": "",
            "hdr_attempt": "",
            "fused_clip_pct": "",
            "captured_at": "",
            "sample_status": "incomplete",
            "failed_round": "front",
            "failed_view": "front",
            "failed_device_index": "",
            "error": "operator cancelled",
        }
    )
    _write_csv(source / "manifests" / f"{session}.csv", fields, rows)
    # Invalid unselected-view content must never be read.
    unselected = source / "right/front_left/normal" / session / "images/broken.png"
    unselected.parent.mkdir(parents=True, exist_ok=True)
    unselected.write_bytes(b"not-a-png")
    return source


def _base_release_fixture(root: Path) -> Path:
    base = root / "base_release"
    fields = (
        "source_origin",
        "physical_part_id",
        "hand",
        "view",
        "resolved_view",
        "label",
        "defect_type",
        "session_id",
        "group_id",
        "release_role",
        "target_bucket",
        "source_path",
        "target_path",
        "content_sha256",
    )
    rows: list[dict[str, str]] = []
    cases = (
        ("normal", "", "old_train", "group001", "train", "normal"),
        ("normal", "", "old_cal", "group001", "calibration", "normal_test"),
        ("normal", "", "old_holdout", "group001", "final_test", "holdout/final_test"),
        ("defect", "deform", "old_defect_cal", "group001", "calibration", "defect"),
        ("defect", "deform", "old_defect_holdout", "group001", "final_test", "holdout/final_test"),
    )
    for case_index, (label, defect_type, session, group, role, bucket) in enumerate(cases):
        for view_index, view in enumerate(("front", "back", "front_left")):
            source = root / "base_assets" / session / f"{view}.png"
            _write_image(source, 80 + case_index + view_index, shape=(4, 6))
            rows.append(
                {
                    "source_origin": "base",
                    "physical_part_id": f"{session}:{group}",
                    "hand": "right",
                    "view": view,
                    "resolved_view": view,
                    "label": label,
                    "defect_type": defect_type,
                    "session_id": session,
                    "group_id": group,
                    "release_role": role,
                    "target_bucket": bucket,
                    "source_path": str(source),
                    "target_path": str(source),
                    "content_sha256": "",
                }
            )
    _write_csv(base / "patchcore_manifest.csv", fields, rows)
    return base


def test_prepare_release_uses_only_front_back_and_keeps_exact_group_splits(tmp_path: Path) -> None:
    module = _load_module()
    source = _source_fixture(tmp_path)
    base = _base_release_fixture(tmp_path)
    roi_config = tmp_path / "roi_config.json"
    roi_config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "image_size": {"width": 10, "height": 8},
                "views": {
                    "front": {"roi": [1, 2, 9, 7]},
                    "back": {"roi": [2, 1, 8, 8]},
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "release"

    summary = module.prepare_release(
        source_root=source,
        base_release=base,
        output_root=output,
        roi_config=roi_config,
        seed=42,
    )

    assert summary["status"] == "VALID"
    assert summary["views"] == ["front", "back"]
    assert summary["new_normal_role_counts"] == {"train": 20, "calibration": 4, "final_test": 4}
    assert not any("front_left" in str(path) for path in output.rglob("*"))
    assert len(list((output / "patchcore/right/front/normal").rglob("*.png"))) == 21
    assert len(list((output / "patchcore/right/back/normal").rglob("*.png"))) == 21
    assert len(list((output / "patchcore/right/front/normal_test").rglob("*.png"))) == 5
    assert len(list((output / "patchcore/right/back/normal_test").rglob("*.png"))) == 5
    assert len(list((output / "patchcore/right/front/defect").rglob("*.png"))) == 1
    assert len(list((output / "patchcore/right/back/defect").rglob("*.png"))) == 1
    assert len(list((output / "holdout/new_normal/final_test/front").rglob("*.png"))) == 4
    assert len(list((output / "holdout/new_normal/final_test/back").rglob("*.png"))) == 4
    front_crop = next(
        path
        for path in (output / "patchcore/right/front").rglob("*.png")
        if "zs32_4cam_accept" in str(path)
    )
    back_crop = next(
        path
        for path in (output / "patchcore/right/back").rglob("*.png")
        if "zs32_4cam_accept" in str(path)
    )
    assert cv2.imread(str(front_crop)).shape[:2] == (5, 8)
    assert cv2.imread(str(back_crop)).shape[:2] == (7, 6)

    split_rows = list(csv.DictReader((output / "new_normal_splits.csv").open(encoding="utf-8")))
    assert len(split_rows) == 28
    assert {row["group_id"] for row in split_rows} == {f"group{index:03d}" for index in range(1, 29)}
    assert all("group029" not in row["physical_part_id"] for row in split_rows)


def test_validate_release_rejects_any_unselected_view(tmp_path: Path) -> None:
    module = _load_module()
    source = _source_fixture(tmp_path)
    base = _base_release_fixture(tmp_path)
    roi_config = tmp_path / "roi_config.json"
    roi_config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "image_size": {"width": 10, "height": 8},
                "views": {
                    "front": {"roi": [1, 2, 9, 7]},
                    "back": {"roi": [2, 1, 8, 8]},
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "release"
    module.prepare_release(
        source_root=source,
        base_release=base,
        output_root=output,
        roi_config=roi_config,
        seed=42,
    )
    leaked = output / "patchcore/right/front_left/normal/leak/images/leak.png"
    _write_image(leaked, 1)

    try:
        module.validate_release(output)
    except ValueError as error:
        assert "unselected view" in str(error)
    else:
        raise AssertionError("validate_release accepted an unselected PatchCore view")


def test_cli_requires_explicit_prepare_paths() -> None:
    module = _load_module()

    args = module.build_parser().parse_args(
        [
            "prepare",
            "--source-root",
            "dataset/zs32_top",
            "--base-release",
            "dataset/base",
            "--output-root",
            "dataset/output",
            "--roi-config",
            "dataset/roi.json",
        ]
    )

    assert args.command == "prepare"
    assert args.seed == 42
    assert args.source_root == Path("dataset/zs32_top")
