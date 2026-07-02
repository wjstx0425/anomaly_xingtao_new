# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ZS32 defect workflow example."""

from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pandas as pd


def load_workflow_module() -> ModuleType:
    """Load the workflow script from the examples directory."""
    script_path = Path(__file__).resolve().parents[3] / "examples" / "api" / "03_models" / "zs32_defect_workflow.py"
    spec = importlib.util.spec_from_file_location("zs32_defect_workflow", script_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load workflow script from {script_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_binary_metrics() -> None:
    """Binary metrics should report the expected confusion matrix and F1."""
    workflow = load_workflow_module()

    metrics = workflow._binary_metrics(
        pd.Series([0, 0, 1, 1]),
        pd.Series([0, 1, 1, 0]),
    )

    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["accuracy"] == 0.5
    assert metrics["f1"] == 0.5


def test_parse_roi_supports_full_image_aliases() -> None:
    """ROI parsing should support explicit full-image preprocessing."""
    workflow = load_workflow_module()

    assert workflow._parse_roi("none") is None
    assert workflow._parse_roi("full") is None
    assert workflow._format_roi(None) == "full"


def test_visualizer_field_size_follows_model_aspect() -> None:
    """Visualizer panels should preserve the model input aspect ratio by default."""
    workflow = load_workflow_module()

    assert workflow._visualizer_field_size((182, 1008), Namespace(visualizer_field_size=None)) == (1008, 182)
    assert workflow._parse_field_size("1200,220") == (1200, 220)
    assert workflow._visualizer_field_size((182, 1008), Namespace(visualizer_field_size=(1200, 220))) == (1200, 220)


def test_patchcore_memory_options_are_parsed() -> None:
    """PatchCore memory-control arguments should be available from the CLI."""
    workflow = load_workflow_module()

    args = workflow.build_parser().parse_args(
        [
            "all",
            "--models",
            "patchcore",
            "--patchcore-layers",
            "layer2",
            "--patchcore-coreset-ratio",
            "0.1",
            "--patchcore-num-neighbors",
            "1",
            "--patchcore-precision",
            "float16",
        ],
    )

    assert args.patchcore_layers == ["layer2"]
    assert args.patchcore_coreset_ratio == 0.1
    assert args.patchcore_num_neighbors == 1
    assert args.patchcore_precision == "float16"


def test_preprocess_image_can_use_full_image_and_skip_blue_removal(tmp_path: Path) -> None:
    """Preprocessing should support full images without HSV inpainting."""
    workflow = load_workflow_module()
    source_path = tmp_path / "source.png"
    output_path = tmp_path / "output.png"
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[:, :] = (255, 0, 0)
    assert cv2.imwrite(str(source_path), image)

    stats = workflow._preprocess_image(source_path, output_path, roi=None, remove_blue_marks=False)
    output = cv2.imread(str(output_path), cv2.IMREAD_COLOR)

    assert output is not None
    assert output.shape[:2] == (4, 6)
    assert np.array_equal(output, image)
    assert stats["roi_width"] == 6
    assert stats["roi_height"] == 4
    assert stats["blue_removal_enabled"] is False
    assert stats["blue_mask_area"] is None
    assert stats["blue_mask_area_ratio"] is None


def test_deployment_threshold_uses_normal_test_max() -> None:
    """Deployment threshold should default to the max normal_test score."""
    workflow = load_workflow_module()
    predictions = pd.DataFrame(
        {
            "model": ["patchcore"] * 4,
            "view": ["right_top"] * 4,
            "label": ["normal", "normal_test", "normal_test", "defect"],
            "pred_score": [0.05, 0.2, 0.3, 0.31],
        },
    )

    predictions = workflow._add_deployment_predictions(predictions)

    assert predictions["deploy_threshold"].unique().tolist() == [0.3]
    assert predictions["deploy_pred_label"].tolist() == [0, 0, 0, 1]


def test_deployment_threshold_allows_target_false_positive_rate() -> None:
    """Deployment threshold should allow a configurable normal_test false-positive rate."""
    workflow = load_workflow_module()
    normal_test_scores = [index / 100 for index in range(20)]
    predictions = pd.DataFrame(
        {
            "model": ["anomaly_dino"] * 22,
            "view": ["no_hand_top"] * 22,
            "label": ["normal_test"] * 20 + ["defect", "defect"],
            "pred_score": [*normal_test_scores, 0.181, 0.191],
        },
    )

    predictions = workflow._add_deployment_predictions(predictions, deploy_fpr=0.05)

    assert round(float(predictions["deploy_threshold"].iloc[0]), 2) == 0.18
    assert predictions[predictions["label"] == "normal_test"]["deploy_pred_label"].sum() == 1
    assert predictions[predictions["label"] == "defect"]["deploy_pred_label"].tolist() == [1, 1]


def test_sample_summary_uses_max_frame_score() -> None:
    """Sample-level score should be the max across frames."""
    workflow = load_workflow_module()
    predictions = pd.DataFrame(
        {
            "model": ["efficient_ad", "efficient_ad"],
            "view": ["left_top", "left_top"],
            "label": ["defect", "defect"],
            "sample_id": ["part001", "part001"],
            "processed_path": ["a.png", "b.png"],
            "pred_score": [0.1, 0.9],
            "deploy_threshold": [0.5, 0.5],
        },
    )

    summary = workflow._summarize_sample_level(predictions)

    assert len(summary) == 1
    assert summary.loc[0, "pred_score"] == 0.9
    assert summary.loc[0, "deploy_pred_label"] == 1


def test_iter_raw_images_supports_nested_zs32_dirs(tmp_path: Path) -> None:
    """Raw image iteration should keep the original nested ZS32 layout."""
    workflow = load_workflow_module()
    image_path = tmp_path / "right" / "top" / "normal" / "part001" / "images" / "frame_000001.png"
    image_path.parent.mkdir(parents=True)
    image_path.touch()

    images = list(workflow._iter_raw_images(tmp_path, "right_top", "normal"))

    assert images == [image_path]
    assert workflow._raw_sample_id(image_path) == "part001"


def test_iter_raw_images_supports_flat_label_dirs(tmp_path: Path) -> None:
    """Raw image iteration should support FX11-style label/*.png folders."""
    workflow = load_workflow_module()
    image_path = tmp_path / "no_hand" / "top" / "normal" / "part001_000001.png"
    image_path.parent.mkdir(parents=True)
    image_path.touch()

    images = list(workflow._iter_raw_images(tmp_path, "no_hand_top", "normal"))

    assert images == [image_path]
    assert workflow._raw_sample_id(image_path) == "part001_000001"


def test_iter_raw_images_supports_no_hand_bottom_dirs(tmp_path: Path) -> None:
    """Raw image iteration should support FX11 bottom view folders."""
    workflow = load_workflow_module()
    image_path = tmp_path / "no_hand" / "bottom" / "normal" / "part001_000001.png"
    image_path.parent.mkdir(parents=True)
    image_path.touch()

    images = list(workflow._iter_raw_images(tmp_path, "no_hand_bottom", "normal"))

    assert images == [image_path]


def test_iter_raw_images_supports_left_bottom_alias(tmp_path: Path) -> None:
    """C789 left_bottom should accept both bottom_ZS32 and bottom folders."""
    workflow = load_workflow_module()
    image_path = tmp_path / "left" / "bottom" / "normal" / "part001_000001.png"
    image_path.parent.mkdir(parents=True)
    image_path.touch()

    images = list(workflow._iter_raw_images(tmp_path, "left_bottom", "normal"))

    assert images == [image_path]
    assert workflow._raw_label_dirs(tmp_path, "left_bottom", "normal") == [
        tmp_path / "left" / "bottom_ZS32" / "normal",
        tmp_path / "left" / "bottom" / "normal",
    ]


def test_preprocess_dataset_clears_stale_view_output(tmp_path: Path) -> None:
    """Preprocessing should remove stale generated images before writing a new manifest."""
    workflow = load_workflow_module()
    data_root = tmp_path / "data"
    output_root = tmp_path / "output"
    image_path = data_root / "no_hand" / "top" / "normal" / "part001_000001.png"
    image_path.parent.mkdir(parents=True)
    image = np.full((4, 6, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)

    stale_path = (
        output_root
        / "preprocessed"
        / "no_hand_top"
        / "normal"
        / "old_sample"
        / "images"
        / "stale.png"
    )
    stale_path.parent.mkdir(parents=True)
    stale_path.write_text("stale", encoding="utf-8")

    args = Namespace(
        data_root=data_root,
        output_root=output_root,
        views=["no_hand_top"],
        roi=None,
        skip_blue_removal=True,
    )

    manifest_path = workflow.preprocess_dataset(args)
    manifest = pd.read_csv(manifest_path)
    generated_images = list((output_root / "preprocessed" / "no_hand_top").rglob("*.png"))

    assert not stale_path.exists()
    assert len(manifest) == 1
    assert len(generated_images) == 1
