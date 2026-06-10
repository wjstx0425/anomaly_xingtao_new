# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ZS32 capture inference helper."""

from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np


def load_inference_module() -> ModuleType:
    """Load the capture inference script from its file path."""
    script_path = Path(__file__).resolve().parents[3] / "capture_data" / "inference.py"
    spec = importlib.util.spec_from_file_location("capture_data_inference", script_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load inference script from {script_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_threshold_reads_workflow_summary(tmp_path: Path) -> None:
    """Deployment threshold should come from the workflow summary when present."""
    inference = load_inference_module()
    workflow = inference._load_workflow_module()
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    workflow.pd.DataFrame(
        [
            {"model": "patchcore", "view": "no_hand_top", "deploy_threshold": 0.123},
            {"model": "patchcore", "view": "right_top", "deploy_threshold": 0.456},
        ],
    ).to_csv(reports_dir / "summary.csv", index=False)

    args = Namespace(threshold=None, output_root=tmp_path, model="patchcore", view="no_hand_top")

    assert inference._resolve_threshold(args, workflow) == 0.123

    args.threshold = 0.9
    assert inference._resolve_threshold(args, workflow) == 0.9


def test_resolve_preprocessing_config_reads_manifest(tmp_path: Path) -> None:
    """Auto preprocessing config should use the workflow manifest for the selected view."""
    inference = load_inference_module()
    workflow = inference._load_workflow_module()
    manifest_path = workflow._manifest_path(tmp_path)
    manifest_path.parent.mkdir(parents=True)
    workflow.pd.DataFrame(
        [
            {
                "processed_path": str(tmp_path / "preprocessed" / "no_hand_top" / "normal" / "a.png"),
                "view": "no_hand_top",
                "roi": "1,2,30,40",
                "blue_removal_enabled": False,
            },
        ],
    ).to_csv(manifest_path, index=False)

    args = Namespace(output_root=tmp_path, view="no_hand_top", roi="auto", blue_removal="auto")

    config = inference._resolve_preprocessing_config(args, workflow)

    assert config.roi == (1, 2, 30, 40)
    assert config.remove_blue_marks is False


def test_prepare_prediction_input_can_skip_preprocessing(tmp_path: Path) -> None:
    """Already-preprocessed inputs should be passed through and mapped to themselves."""
    inference = load_inference_module()
    workflow = inference._load_workflow_module()
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    first_image = image_dir / "a.png"
    second_image = image_dir / "nested" / "b.jpg"
    second_image.parent.mkdir()
    first_image.touch()
    second_image.touch()
    (image_dir / "notes.txt").touch()

    prediction_path, source_map = inference._prepare_prediction_input(
        image_dir,
        tmp_path / "out",
        inference.PreprocessingConfig(roi=None, remove_blue_marks=False),
        input_is_preprocessed=True,
        workflow=workflow,
    )

    assert prediction_path == image_dir.resolve()
    assert source_map == {
        str(first_image.resolve()): str(first_image.resolve()),
        str(second_image.resolve()): str(second_image.resolve()),
    }


def test_prediction_frame_adds_source_path_and_deploy_label(tmp_path: Path) -> None:
    """Prediction rows should include source paths and workflow deployment labels."""
    inference = load_inference_module()
    workflow = inference._load_workflow_module()
    processed_path = tmp_path / "processed.png"
    source_path = tmp_path / "source.png"
    args = Namespace(model="patchcore", view="no_hand_top")
    predictions = [SimpleNamespace(image_path=str(processed_path), pred_score=0.2, pred_label=0)]

    frame = inference._prediction_frame(
        predictions,
        args,
        tmp_path / "model.ckpt",
        {str(processed_path.resolve()): str(source_path.resolve())},
        threshold=0.1,
        workflow=workflow,
    )

    assert frame["source_path"].tolist() == [str(source_path.resolve())]
    assert frame["deploy_pred_label"].tolist() == [1]
    assert frame["anomalib_pred_label"].tolist() == [0]


def test_add_review_columns_uses_manifest_labels(tmp_path: Path) -> None:
    """Review columns should classify predictions into TP/TN/FP/FN buckets."""
    inference = load_inference_module()
    workflow = inference._load_workflow_module()
    manifest_path = workflow._manifest_path(tmp_path)
    manifest_path.parent.mkdir(parents=True)
    defect_path = tmp_path / "preprocessed" / "no_hand_top" / "defect" / "a.png"
    normal_path = tmp_path / "preprocessed" / "no_hand_top" / "normal" / "b.png"
    workflow.pd.DataFrame(
        [
            {"processed_path": str(defect_path.resolve()), "view": "no_hand_top", "label": "defect"},
            {"processed_path": str(normal_path.resolve()), "view": "no_hand_top", "label": "normal"},
        ],
    ).to_csv(manifest_path, index=False)
    frame = workflow.pd.DataFrame(
        [
            {
                "source_path": str(defect_path.resolve()),
                "processed_path": str(defect_path.resolve()),
                "deploy_pred_label": 1,
                "anomalib_pred_label": 1,
            },
            {
                "source_path": str(normal_path.resolve()),
                "processed_path": str(normal_path.resolve()),
                "deploy_pred_label": 1,
                "anomalib_pred_label": 1,
            },
        ],
    )
    args = Namespace(output_root=tmp_path)

    frame = inference._add_review_columns(frame, args, workflow)

    assert frame["gt_label"].tolist() == [1, 0]
    assert frame["review_pred_label"].tolist() == [1, 1]
    assert frame["result_type"].tolist() == ["TP", "FP"]


def test_write_review_artifacts_creates_result_folders(tmp_path: Path) -> None:
    """Review artifacts should include image, heatmap, and mask files."""
    inference = load_inference_module()
    workflow = inference._load_workflow_module()
    image_path = tmp_path / "source.png"
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    image[:, :] = (30, 40, 50)
    assert cv2.imwrite(str(image_path), image)
    frame = workflow.pd.DataFrame(
        [
            {
                "source_path": str(image_path.resolve()),
                "processed_path": str(image_path.resolve()),
                "view": "no_hand_top",
                "pred_score": 0.7,
                "deploy_threshold": 0.5,
                "dataset_label": "defect",
                "gt_label": 1,
                "review_pred_label": 1,
                "result_type": "TP",
            },
        ],
    )
    artifacts = {
        str(image_path.resolve()): {
            "anomaly_map": np.arange(12 * 16, dtype=np.float32).reshape(12, 16),
            "pred_mask": np.ones((12, 16), dtype=np.uint8),
        },
    }
    args = Namespace(mask_threshold=0.5)

    review_dir = inference._write_review_artifacts(frame, artifacts, tmp_path / "out", args, workflow)

    assert (review_dir / "TP").is_dir()
    assert (review_dir / "TN").is_dir()
    assert (review_dir / "FP").is_dir()
    assert (review_dir / "FN").is_dir()
    assert len(list((review_dir / "TP").glob("*_image.png"))) == 1
    assert len(list((review_dir / "TP").glob("*_heatmap.png"))) == 1
    assert len(list((review_dir / "TP").glob("*_mask.png"))) == 1


def test_mask_overlay_handles_empty_masks() -> None:
    """Mask overlay should not fail when the predicted mask has no positive pixels."""
    inference = load_inference_module()
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    mask = np.zeros((12, 16), dtype=np.uint8)

    overlay = inference._mask_overlay(
        image,
        mask,
        anomaly_map=None,
        mask_threshold=0.5,
        lines=["test"],
        result_type="TN",
    )

    assert overlay.shape == image.shape
