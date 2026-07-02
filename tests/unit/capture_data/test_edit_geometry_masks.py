# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the geometry mask editor helper functions."""

from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np


def _load_module() -> ModuleType:
    """Load the editor script from the repository checkout."""
    script_path = Path(__file__).resolve().parents[3] / "capture_data" / "edit_geometry_masks.py"
    spec = importlib.util.spec_from_file_location("edit_geometry_masks_for_tests", script_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load {script_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_gray(path: Path, image: np.ndarray) -> None:
    """Write a grayscale image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def _write_manifest(review_pack: Path, slots: tuple[str, ...] = ("slot01", "slot02")) -> None:
    """Write a compact review-pack manifest."""
    fieldnames = [
        "slot",
        "reference_path",
        "reference_image",
        "review_sheet",
        "expected_mask",
        "allowed_mask",
        "ignore_mask",
        "watch_edge_mask",
        "normal_count",
        "stress_count",
        "defect_count",
    ]
    rows = []
    for index, slot in enumerate(slots, start=1):
        reference = review_pack / "references" / f"{slot}_reference.png"
        sheet = review_pack / "sheets" / f"{slot}_review_sheet.png"
        masks = review_pack / "manual_masks"
        image = np.zeros((12, 16, 3), dtype=np.uint8)
        image[3:9, 4:12] = 120 + index
        reference.parent.mkdir(parents=True, exist_ok=True)
        sheet.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(reference), image)
        assert cv2.imwrite(str(sheet), image)
        for suffix in ("expected", "allowed", "ignore", "watch_edge"):
            _write_gray(masks / f"{slot}_{suffix}.png", np.zeros((12, 16), dtype=np.uint8))
        rows.append(
            {
                "slot": slot,
                "reference_path": str(reference),
                "reference_image": str(reference),
                "review_sheet": str(sheet),
                "expected_mask": str(masks / f"{slot}_expected.png"),
                "allowed_mask": str(masks / f"{slot}_allowed.png"),
                "ignore_mask": str(masks / f"{slot}_ignore.png"),
                "watch_edge_mask": str(masks / f"{slot}_watch_edge.png"),
                "normal_count": index,
                "stress_count": index + 10,
                "defect_count": index + 20,
            },
        )
    with (review_pack / "review_pack_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_load_review_manifest_resolves_masks_and_counts() -> None:
    """Review manifest rows should become slot items with mask paths."""
    editor = _load_module()
    with tempfile.TemporaryDirectory() as tmp:
        review_pack = Path(tmp) / "review"
        review_pack.mkdir()
        _write_manifest(review_pack)

        items = editor.load_review_manifest(review_pack)

    assert [item.slot for item in items] == ["slot01", "slot02"]
    assert items[0].normal_count == 1
    assert items[0].stress_count == 11
    assert items[0].defect_count == 21
    assert set(items[0].masks) == set(editor.MASK_TYPES)
    assert items[0].masks["watch_edge"].name == "slot01_watch_edge.png"


def test_write_binary_mask_outputs_only_black_and_white() -> None:
    """Mask writing should normalize any non-zero input to 255."""
    editor = _load_module()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mask.png"
        mask = np.array([[0, 1, 127], [255, 2, 0]], dtype=np.uint8)

        editor.write_binary_mask(path, mask)
        loaded = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    assert loaded is not None
    assert set(np.unique(loaded).tolist()) == {0, 255}


def test_mask_dir_from_review_manifest_uses_manifest_paths() -> None:
    """Validation should compile masks from the manifest's mask directory."""
    editor = _load_module()
    with tempfile.TemporaryDirectory() as tmp:
        review_pack = Path(tmp) / "review"
        review_pack.mkdir()
        _write_manifest(review_pack, slots=("slot01",))

        assert editor.mask_dir_from_review_manifest(review_pack) == review_pack / "manual_masks"


def test_compose_overlay_does_not_mutate_allowed_mask() -> None:
    """Preview overlays should not force allowed to include expected in memory."""
    editor = _load_module()
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    expected = np.zeros((10, 10), dtype=bool)
    expected[2:4, 2:4] = True
    allowed = np.zeros((10, 10), dtype=bool)
    masks = {
        "expected": expected.copy(),
        "allowed": allowed.copy(),
        "ignore": np.zeros((10, 10), dtype=bool),
        "watch_edge": np.zeros((10, 10), dtype=bool),
    }

    overlay = editor.compose_overlay(image, masks, "expected", alpha=0.5)

    assert overlay.shape == image.shape
    assert int(overlay.sum()) > 0
    assert not bool(masks["allowed"][2, 2])


def test_summarize_results_reports_recall_and_key_samples() -> None:
    """CSV summarization should report aggregate and key-sample labels."""
    editor = _load_module()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        stress_dir = root / "stress"
        defect_dir = root / "defect"
        stress_dir.mkdir()
        defect_dir.mkdir()
        with (stress_dir / "geometry_predictions.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["source_path", "gt_label", "geometry_pred_label"])
            writer.writeheader()
            writer.writerows(
                [
                    {"source_path": "normal_a.png", "gt_label": "0", "geometry_pred_label": "0"},
                    {"source_path": "normal_b.png", "gt_label": "0", "geometry_pred_label": "1"},
                ],
            )
        with (defect_dir / "geometry_predictions.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["source_path", "gt_label", "geometry_pred_label"])
            writer.writeheader()
            writer.writerows(
                [
                    {"source_path": "less_1_2_slot02.png", "gt_label": "1", "geometry_pred_label": "0"},
                    {"source_path": "more_2_2_slot04.png", "gt_label": "1", "geometry_pred_label": "1"},
                ],
            )
        with (defect_dir / "fused_predictions.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "source_path",
                    "frame_id",
                    "gt_label",
                    "anomaly_deploy_pred_label",
                    "geometry_pred_label",
                    "final_pred_label",
                ],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "source_path": "less_1_2_slot02.png",
                        "frame_id": "less_1_2_slot02",
                        "gt_label": "1",
                        "anomaly_deploy_pred_label": "0",
                        "geometry_pred_label": "0",
                        "final_pred_label": "0",
                    },
                    {
                        "source_path": "more_2_2_slot04.png",
                        "frame_id": "more_2_2_slot04",
                        "gt_label": "1",
                        "anomaly_deploy_pred_label": "0",
                        "geometry_pred_label": "1",
                        "final_pred_label": "1",
                    },
                ],
            )

        summary = editor.summarize_eval_outputs(stress_dir, defect_dir)

    assert summary.stress_fp == 1
    assert summary.stress_total == 2
    assert summary.geometry_positives == 1
    assert summary.geometry_total == 2
    assert summary.fused_positives == 1
    assert summary.fused_total == 2
    assert summary.key_samples["less_1_2_slot02"].final == 0
    assert summary.key_samples["more_2_2_slot04"].geometry == 1


def test_save_current_slot_creates_backup_and_writes_binary_masks() -> None:
    """Saving a slot should back up original masks before writing edited masks."""
    editor = _load_module()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        review_pack = root / "review"
        review_pack.mkdir()
        _write_manifest(review_pack, slots=("slot01",))
        config = editor.EditorConfig(
            review_pack=review_pack,
            template_dir=root / "templates",
            stress_root=root / "stress",
            defect_root=root / "defect",
            anomaly_predictions=root / "predictions.csv",
            stress_output_dir=root / "stress_out",
            defect_output_dir=root / "defect_out",
        )
        mask_editor = editor.GeometryMaskEditor(config)
        mask_editor.masks_by_slot["slot01"]["expected"][1:4, 1:4] = True

        mask_editor.save_current_slot()

        backups = list((review_pack / "manual_masks_backup").glob("*/slot01_expected.png"))
        saved = cv2.imread(str(review_pack / "manual_masks" / "slot01_expected.png"), cv2.IMREAD_GRAYSCALE)

    assert len(backups) == 1
    assert saved is not None
    assert set(np.unique(saved).tolist()) == {0, 255}


if __name__ == "__main__":
    test_load_review_manifest_resolves_masks_and_counts()
    test_write_binary_mask_outputs_only_black_and_white()
    test_mask_dir_from_review_manifest_uses_manifest_paths()
    test_compose_overlay_does_not_mutate_allowed_mask()
    test_summarize_results_reports_recall_and_key_samples()
    test_save_current_slot_creates_backup_and_writes_binary_masks()
