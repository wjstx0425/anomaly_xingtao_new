# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the isolated BMW 21-point Template candidate."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import cv2
import numpy as np
import pytest

from bmw_inspection.lab.contracts import ViewId
from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER
from bmw_inspection.lab.template import TemplateSample, validate_template_source_sessions


REPO_ROOT = Path(__file__).resolve().parents[4]
ALLOWED_SESSION = "20260810_210030_527506"
UNALLOWED_SESSION = "20260810_999999_000000"


def _write_image(path: Path, seed: int) -> Path:
    image = np.zeros((48, 64), dtype=np.uint8)
    cv2.circle(image, (8 + seed % 32, 12 + seed % 20), 6, 180, -1)
    cv2.line(image, (0, seed % 48), (63, (seed * 3) % 48), 255, 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)
    return path


def _row(
    root: Path,
    view_id: str,
    part_id: str,
    split: str,
    label: str,
    seed: int,
    *,
    session_id: str = ALLOWED_SESSION,
) -> dict[str, str]:
    image = _write_image(root / "crops" / view_id / f"{session_id}__{part_id}__{view_id}.png", seed)
    return {
        "sample_id": f"{part_id}_000001",
        "part_id": part_id,
        "view_id": view_id,
        "image_path": str(image),
        "split": split,
        "label": label,
    }


def _candidate_manifest(root: Path) -> Path:
    rows: list[dict[str, str]] = []
    for view_index, view_id in enumerate(VIEW_ORDER):
        for sample_index in range(5):
            rows.append(_row(root, view_id, f"normal_train_{sample_index}", "train", "normal", view_index + sample_index))
        rows.append(_row(root, view_id, "defect_train", "train", "defect", view_index + 20))
        rows.append(_row(root, view_id, "normal_calibration", "calibration", "normal", view_index))
        rows.append(_row(root, view_id, "normal_final", "final_test", "normal", view_index + 1))
    manifest = root / "template" / "trainer_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fields = ("sample_id", "part_id", "view_id", "image_path", "split", "label")
    manifest.write_text(
        ",".join(fields)
        + "\n"
        + "\n".join(",".join(row[field] for field in fields) for row in rows)
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _baseline_models(root: Path) -> Path:
    for view_index, view_id in enumerate(VIEW_ORDER):
        model = root / view_id / "model.json"
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_text(json.dumps({"threshold": 1.0 - view_index / 100.0}), encoding="utf-8")
    return root


def _candidate_module() -> dict[str, object]:
    return runpy.run_path(str(REPO_ROOT / "pipeline/bmw_lab_train_template_fixed_thresholds.py"))


def test_fixed_threshold_candidate_trains_only_normal_rows_reuses_thresholds_and_reports_normal_part_pass_rates(
    tmp_path: Path,
) -> None:
    module = _candidate_module()
    manifest = _candidate_manifest(tmp_path)
    baseline_root = _baseline_models(tmp_path / "baseline")
    output_root = tmp_path / "candidate"

    report = module["run_fixed_threshold_candidate"](
        manifest=manifest,
        baseline_template_root=baseline_root,
        output_root=output_root,
        allowed_session_ids=(ALLOWED_SESSION,),
    )

    assert report["source_sessions"] == [ALLOWED_SESSION]
    assert report["parameters"] == {"max_shift": 12, "target_size": [512, 512], "template_count": 5}
    for view_index, view_id in enumerate(VIEW_ORDER):
        model = json.loads((output_root / "template" / view_id / "model.json").read_text(encoding="utf-8"))
        selected = {item["part_id"] for item in model["templates"]}
        assert selected == {f"normal_train_{index}" for index in range(5)}
        assert model["threshold"] == pytest.approx(1.0 - view_index / 100.0)
        assert report["per_view"][view_id]["calibration"]["normal_count"] == 1
        assert report["per_view"][view_id]["final_test"]["normal_count"] == 1
    assert report["normal_part_pass_rates"]["calibration"] == {
        "part_count": 1,
        "passed_part_count": 1,
        "pass_rate": 1.0,
    }
    assert report["normal_part_pass_rates"]["final_test"] == {
        "part_count": 1,
        "passed_part_count": 1,
        "pass_rate": 1.0,
    }
    assert (output_root / "template_report.json").is_file()
    with pytest.raises(FileExistsError, match="refuse to overwrite"):
        module["run_fixed_threshold_candidate"](
            manifest=manifest,
            baseline_template_root=baseline_root,
            output_root=output_root,
            allowed_session_ids=(ALLOWED_SESSION,),
        )


def test_template_source_session_validation_rejects_non_whitelisted_crop_filename(tmp_path: Path) -> None:
    sample = TemplateSample(
        sample_id="normal_000001",
        part_id="normal",
        view_id=ViewId.FRONT,
        image_path=_write_image(
            tmp_path / f"{UNALLOWED_SESSION}__normal_000001__front.png",
            seed=1,
        ),
        split="train",
        label="normal",
    )

    with pytest.raises(ValueError, match="not in the allowed session list"):
        validate_template_source_sessions((sample,), (ALLOWED_SESSION,))
