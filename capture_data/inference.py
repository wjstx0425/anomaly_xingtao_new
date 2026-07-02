# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run ZS32/FX11 image inference with the workflow training settings.

Example:
    source .venv/bin/activate

    python capture_data/inference.py hik_images \\
        --output-root results/c789/left_top_pc_512 \\
        --view no_hand_top \\
        --model patchcore \\
        --image-size 512,1024

    python capture_data/inference.py image.png \
        --ckpt-path results/c789/left_top_pc_512/runs/no_hand_top/patchcore/.../model.ckpt \
        --output-root results/c789/left_top_pc_512 \
        --view no_hand_top
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / "examples" / "api" / "03_models" / "zs32_defect_workflow.py"
DEFAULT_VIEW = "no_hand_top"


@dataclass(frozen=True)
class PreprocessingConfig:
    """Resolved OpenCV preprocessing settings."""

    roi: tuple[int, int, int, int] | None
    remove_blue_marks: bool


@dataclass(frozen=True)
class ValidRegionMaskConfig:
    """Settings for image-specific valid-region score masks."""

    preset: str
    hole_dilation: int
    border_margin: int
    foreground_threshold_scale: float
    min_coverage: float = 0.05


@lru_cache(maxsize=1)
def _load_workflow_module() -> ModuleType:
    """Load the ZS32 workflow module from its example-script path."""
    spec = importlib.util.spec_from_file_location("zs32_defect_workflow", WORKFLOW_PATH)
    if spec is None or spec.loader is None:
        msg = f"Could not load workflow script from {WORKFLOW_PATH}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_image_size(value: str) -> tuple[int, int]:
    """Parse image size using the workflow parser."""
    return _load_workflow_module()._parse_image_size(value)


def _iter_input_images(path: Path, workflow: ModuleType) -> list[Path]:
    """Return sorted image files from a single image or directory."""
    path = path.resolve()
    extensions = tuple(extension.lower() for extension in workflow.IMAGE_EXTENSIONS)
    if path.is_file():
        if path.suffix.lower() not in extensions:
            msg = f"Input is not a supported image file: {path}"
            raise ValueError(msg)
        return [path]
    if not path.is_dir():
        msg = f"Input path does not exist: {path}"
        raise FileNotFoundError(msg)

    image_paths = sorted(child for child in path.rglob("*") if child.is_file() and child.suffix.lower() in extensions)
    if not image_paths:
        msg = f"No images found under {path}"
        raise RuntimeError(msg)
    return image_paths


def _coerce_bool(value: Any) -> bool:
    """Convert manifest CSV values to bool."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    msg = f"Could not parse boolean value from manifest: {value!r}"
    raise ValueError(msg)


def _manifest_for_view(output_root: Path, view: str, workflow: ModuleType) -> Any | None:
    """Load manifest rows for a view when the training manifest exists."""
    manifest_path = workflow._manifest_path(output_root)
    if not manifest_path.is_file():
        return None
    manifest = workflow._load_manifest(output_root)
    rows = manifest[manifest["view"] == view]
    return rows if not rows.empty else None


def _roi_from_manifest(output_root: Path, view: str, workflow: ModuleType) -> tuple[int, int, int, int] | None:
    """Resolve the unique ROI recorded for the selected view."""
    rows = _manifest_for_view(output_root, view, workflow)
    if rows is None or "roi" not in rows:
        return None

    values = sorted(str(value) for value in rows["roi"].dropna().unique())
    if len(values) != 1:
        return None
    return workflow._parse_roi(values[0])


def _blue_removal_from_manifest(output_root: Path, view: str, workflow: ModuleType) -> bool | None:
    """Resolve the unique blue-mark setting recorded for the selected view."""
    rows = _manifest_for_view(output_root, view, workflow)
    if rows is None or "blue_removal_enabled" not in rows:
        return None

    values = rows["blue_removal_enabled"].dropna().unique()
    if len(values) != 1:
        return None
    return _coerce_bool(values[0])


def _resolve_preprocessing_config(args: argparse.Namespace, workflow: ModuleType) -> PreprocessingConfig:
    """Resolve raw-image preprocessing settings from args, manifest, and defaults."""
    output_root = args.output_root.resolve()
    if args.roi == "auto":
        roi = _roi_from_manifest(output_root, args.view, workflow)
        if roi is None:
            roi = workflow.DEFAULT_ROI
    else:
        roi = workflow._parse_roi(args.roi)

    if args.blue_removal == "auto":
        remove_blue_marks = _blue_removal_from_manifest(output_root, args.view, workflow)
        if remove_blue_marks is None:
            remove_blue_marks = True
    else:
        remove_blue_marks = args.blue_removal == "on"

    return PreprocessingConfig(roi=roi, remove_blue_marks=remove_blue_marks)


def _resolve_checkpoint(args: argparse.Namespace, workflow: ModuleType) -> Path:
    """Resolve an explicit or automatically discovered checkpoint."""
    if args.ckpt_path is not None:
        checkpoint_path = args.ckpt_path.resolve()
        if not checkpoint_path.is_file():
            msg = f"Checkpoint does not exist: {checkpoint_path}"
            raise FileNotFoundError(msg)
        return checkpoint_path

    output_root = args.output_root.resolve()
    run_dir = workflow._run_dir(output_root, args.view, args.model)
    try:
        return workflow._find_checkpoint(run_dir)
    except FileNotFoundError:
        pass

    checkpoints = sorted(output_root.rglob("*.ckpt"), key=lambda path: path.stat().st_mtime, reverse=True)
    model_key = args.model.lower()
    filtered = [
        path
        for path in checkpoints
        if args.view in path.parts
        and (model_key in str(path).lower() or model_key.replace("_", "") in str(path).lower())
    ]
    if filtered:
        model_checkpoints = [path for path in filtered if path.name == "model.ckpt"]
        return model_checkpoints[0] if model_checkpoints else filtered[0]
    if len(checkpoints) == 1:
        return checkpoints[0]

    msg = (
        f"No checkpoint found for view={args.view!r}, model={args.model!r} under {output_root}. "
        "Pass --ckpt-path or point --output-root at the workflow output directory."
    )
    raise FileNotFoundError(msg)


def _resolve_threshold(args: argparse.Namespace, workflow: ModuleType) -> float | None:
    """Resolve the deployment threshold from args or workflow summary.csv."""
    if args.threshold is not None:
        return args.threshold

    summary_path = args.output_root.resolve() / "reports" / "summary.csv"
    if not summary_path.is_file():
        return None

    summary = workflow.pd.read_csv(summary_path)
    rows = summary[(summary["model"] == args.model) & (summary["view"] == args.view)]
    if rows.empty or "deploy_threshold" not in rows:
        return None
    return float(rows.iloc[0]["deploy_threshold"])


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    """Resolve the directory used for preprocessed inputs and CSV output."""
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.output_root.resolve() / "inference" / args.view / args.model / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _preprocessed_dir(output_dir: Path) -> Path:
    """Return a preprocessed-input directory that will not include stale images."""
    base_dir = output_dir / "preprocessed"
    if not base_dir.exists() or not any(base_dir.iterdir()):
        return base_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return output_dir / f"preprocessed_{timestamp}"


def _prepare_prediction_input(
    data_path: Path,
    output_dir: Path,
    config: PreprocessingConfig,
    input_is_preprocessed: bool,
    workflow: ModuleType,
) -> tuple[Path, dict[str, str]]:
    """Prepare images for prediction and map processed paths back to source paths."""
    image_paths = _iter_input_images(data_path, workflow)

    if input_is_preprocessed:
        return data_path.resolve(), {str(path.resolve()): str(path.resolve()) for path in image_paths}

    destination_root = _preprocessed_dir(output_dir)
    source_root = data_path.resolve() if data_path.is_dir() else data_path.resolve().parent
    source_map: dict[str, str] = {}

    for source_path in image_paths:
        relative_path = source_path.relative_to(source_root)
        destination_path = destination_root / relative_path
        workflow._preprocess_image(
            source_path,
            destination_path,
            roi=config.roi,
            remove_blue_marks=config.remove_blue_marks,
        )
        source_map[str(destination_path.resolve())] = str(source_path.resolve())

    return destination_root, source_map


def _build_visualizer(args: argparse.Namespace, image_size: tuple[int, int], visualizations_dir: Path) -> Any:
    """Build the optional prediction visualizer."""
    from anomalib.visualization import ImageVisualizer

    if not args.visualize:
        return False
    field_size = args.visualizer_field_size
    if field_size is None:
        field_size = image_size[1], image_size[0]
    return ImageVisualizer(
        fields=["image", "anomaly_map"],
        overlay_fields=[("image", ["anomaly_map"]), ("image", ["pred_mask"])],
        field_size=field_size,
        output_dir=visualizations_dir,
    )


def _build_model(args: argparse.Namespace, image_size: tuple[int, int], visualizations_dir: Path) -> Any:
    """Build a model with the same constructor settings used by the training workflow."""
    from anomalib.models import AnomalyDINO, EfficientAd, Patchcore

    visualizer = _build_visualizer(args, image_size, visualizations_dir)

    if args.model == "patchcore":
        return Patchcore(
            backbone="wide_resnet50_2",
            layers=("layer2", "layer3"),
            pre_trained=True,
            coreset_sampling_ratio=0.3,
            num_neighbors=2,
            pre_processor=Patchcore.configure_pre_processor(image_size=image_size),
            visualizer=visualizer,
        )
    if args.model == "efficient_ad":
        return EfficientAd(
            imagenet_dir=args.imagenet_dir.resolve(),
            teacher_out_channels=384,
            model_size="medium",
            lr=1e-4,
            pre_processor=EfficientAd.configure_pre_processor(image_size=image_size),
            visualizer=visualizer,
        )
    if args.model == "anomaly_dino":
        return AnomalyDINO(
            num_neighbours=args.anomaly_dino_neighbors,
            encoder_name=args.anomaly_dino_encoder,
            masking=args.anomaly_dino_masking,
            coreset_subsampling=args.anomaly_dino_coreset_subsampling,
            sampling_ratio=args.anomaly_dino_sampling_ratio,
            pre_processor=AnomalyDINO.configure_pre_processor(image_size=image_size),
            visualizer=visualizer,
        )

    msg = f"Unsupported model: {args.model}"
    raise ValueError(msg)


def _load_model(
    args: argparse.Namespace,
    checkpoint_path: Path,
    image_size: tuple[int, int],
    visualizations_dir: Path,
) -> Any:
    """Load the trained model, preferring checkpoint hyperparameters."""
    if args.rebuild_model:
        return _build_model(args, image_size, visualizations_dir)

    from anomalib.models import AnomalyDINO, EfficientAd, Patchcore

    model_classes = {
        "patchcore": Patchcore,
        "efficient_ad": EfficientAd,
        "anomaly_dino": AnomalyDINO,
    }
    model_class = model_classes[args.model]
    return model_class.load_from_checkpoint(
        checkpoint_path,
        weights_only=False,
        visualizer=_build_visualizer(args, image_size, visualizations_dir),
    )


def _prediction_frame(
    predictions: Any,
    args: argparse.Namespace,
    checkpoint_path: Path,
    source_map: dict[str, str],
    threshold: float | None,
    workflow: ModuleType,
) -> Any:
    """Convert predictions to a report DataFrame."""
    frame = workflow._predictions_to_frame(predictions, args.model)
    if frame.empty:
        msg = "No predictions were returned."
        raise RuntimeError(msg)

    frame["source_path"] = frame["processed_path"].map(source_map).fillna(frame["processed_path"])
    frame["view"] = args.view
    frame["checkpoint_path"] = str(checkpoint_path)
    frame["deploy_threshold"] = threshold
    if threshold is None:
        frame["deploy_pred_label"] = None
    else:
        frame["deploy_pred_label"] = (frame["pred_score"] > threshold).astype(int)

    columns = [
        "source_path",
        "processed_path",
        "model",
        "view",
        "pred_score",
        "deploy_threshold",
        "deploy_pred_label",
        "anomalib_pred_label",
        "checkpoint_path",
    ]
    return frame[columns].sort_values("source_path")


def _prediction_artifacts(predictions: Any, workflow: ModuleType) -> dict[str, dict[str, Any]]:
    """Collect anomaly maps and masks by processed image path."""
    artifacts = {}
    for item in workflow._iter_prediction_items(predictions):
        image_path = getattr(item, "image_path", None)
        if image_path is None:
            continue
        artifacts[str(Path(image_path).resolve())] = {
            "anomaly_map": getattr(item, "anomaly_map", None),
            "pred_mask": getattr(item, "pred_mask", None),
        }
    return artifacts


def _label_map_from_manifest(output_root: Path, workflow: ModuleType) -> dict[str, str]:
    """Build a processed-path to dataset-label map from the workflow manifest."""
    manifest_path = workflow._manifest_path(output_root)
    if not manifest_path.is_file():
        return {}

    manifest = workflow._load_manifest(output_root)
    if "label" not in manifest:
        return {}
    return dict(zip(manifest["processed_path"].astype(str), manifest["label"].astype(str), strict=False))


def _infer_dataset_label(path: str) -> str | None:
    """Infer a dataset label from the preprocessed folder path."""
    parts = set(Path(path).parts)
    if "defect" in parts:
        return "defect"
    if "normal_test" in parts:
        return "normal_test"
    if "normal" in parts:
        return "normal"
    return None


def _label_to_gt(label: str | None) -> int | None:
    """Convert a dataset label to a binary ground-truth label."""
    if label == "defect":
        return 1
    if label in {"normal", "normal_test"}:
        return 0
    return None


def _coerce_optional_label(value: Any, workflow: ModuleType) -> int | None:
    """Convert a prediction label value to ``0`` or ``1`` when possible."""
    if value is None or workflow.pd.isna(value):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "anomaly", "anomalous", "defect"}:
            return 1
        if text in {"false", "0", "normal"}:
            return 0
    return int(float(value))


def _classification_result(gt_label: int | None, pred_label: int | None) -> str:
    """Return TP/TN/FP/FN for a binary prediction."""
    if gt_label is None or pred_label is None:
        return "UNKNOWN"
    if gt_label == 1 and pred_label == 1:
        return "TP"
    if gt_label == 0 and pred_label == 0:
        return "TN"
    if gt_label == 0 and pred_label == 1:
        return "FP"
    return "FN"


def _add_review_columns(frame: Any, args: argparse.Namespace, workflow: ModuleType) -> Any:
    """Add ground-truth and review-bucket columns to prediction results."""
    label_map = _label_map_from_manifest(args.output_root.resolve(), workflow)
    frame = frame.copy()
    dataset_labels = []
    gt_labels = []
    review_pred_labels = []
    result_types = []

    for row in frame.itertuples(index=False):
        dataset_label = (
            label_map.get(str(row.processed_path))
            or label_map.get(str(row.source_path))
            or _infer_dataset_label(str(row.processed_path))
            or _infer_dataset_label(str(row.source_path))
        )
        gt_label = _label_to_gt(dataset_label)
        pred_value = row.deploy_pred_label
        if pred_value is None or workflow.pd.isna(pred_value):
            pred_value = row.anomalib_pred_label
        pred_label = _coerce_optional_label(pred_value, workflow)

        dataset_labels.append(dataset_label)
        gt_labels.append(gt_label)
        review_pred_labels.append(pred_label)
        result_types.append(_classification_result(gt_label, pred_label))

    frame["dataset_label"] = dataset_labels
    frame["gt_label"] = gt_labels
    frame["review_pred_label"] = review_pred_labels
    frame["result_type"] = result_types
    return frame


def _to_2d_array(value: Any) -> Any | None:
    """Convert a tensor-like map or mask to a 2D numpy array."""
    import numpy as np

    if value is None:
        return None
    if hasattr(value, "detach"):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)

    array = np.squeeze(array)
    if array.ndim == 0:
        return None
    if array.ndim == 3:
        array = array[0] if array.shape[0] in {1, 3} else array[..., 0]
    if array.ndim != 2:
        return None
    return array


def _normalize_uint8(array: Any) -> Any:
    """Normalize a numeric map to uint8 for visualization."""
    import numpy as np

    array = np.nan_to_num(array.astype("float32"), copy=False)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value <= min_value:
        return np.zeros_like(array, dtype=np.uint8)
    return ((array - min_value) / (max_value - min_value) * 255.0).clip(0, 255).astype(np.uint8)


@lru_cache(maxsize=1)
def _load_part_crop_module() -> ModuleType:
    """Load part-crop presets used by valid-region masks."""
    spec = importlib.util.spec_from_file_location(
        "prepare_part_crops_for_valid_region",
        REPO_ROOT / "capture_data" / "prepare_part_crops.py",
    )
    if spec is None or spec.loader is None:
        msg = "Could not load part-crop presets for valid-region masks."
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _valid_region_preset_choices() -> tuple[str, ...]:
    """Return valid-region mask preset names accepted by the CLI."""
    try:
        presets = tuple(sorted(_load_part_crop_module().PRESETS))
    except Exception:
        presets = ()
    return ("none", "auto", *presets)


def _resolve_valid_region_preset(value: str, view: str) -> str:
    """Resolve ``auto`` to the crop preset matching a known C789 view."""
    if value != "auto":
        return value
    if view == "left_top":
        return "c789_left_top_3x2"
    if view == "left_bottom":
        return "c789_left_bottom_3x2"
    return "none"


def _slot_name_from_path(path: str) -> str | None:
    """Infer a slot name from a crop filename or sample folder."""
    import re

    match = re.search(r"_slot(?P<slot>[0-9]+)", str(path))
    if match is None:
        return None
    return f"slot{int(match.group('slot')):02d}"


def _foreground_region_mask(image: Any, config: ValidRegionMaskConfig) -> Any:
    """Build a conservative foreground mask that rejects dark fixture/background."""
    import cv2
    import numpy as np

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu_threshold, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = float(np.clip(max(18.0, otsu_threshold * config.foreground_threshold_scale), 18.0, 55.0))
    raw = (gray > threshold).astype(np.uint8)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8), iterations=1)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    if component_count <= 1:
        foreground = raw.astype(bool)
    else:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        foreground = labels == largest

    height, width = gray.shape
    border = np.zeros((height, width), dtype=bool)
    margin = max(0, config.border_margin)
    if margin == 0:
        border[:, :] = True
    elif height > margin * 2 and width > margin * 2:
        border[margin : height - margin, margin : width - margin] = True
    material = foreground & border & (gray > max(20.0, threshold * 0.9))
    return cv2.morphologyEx(material.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1).astype(bool)


def _preset_hole_mask(shape: tuple[int, int], slot_name: str, preset_name: str, dilation: int) -> Any:
    """Return a boolean mask for preset hole/inpaint regions in one slot crop."""
    import cv2
    import numpy as np

    part_crops = _load_part_crop_module()
    preset = part_crops.PRESETS.get(preset_name)
    if preset is None:
        return np.zeros(shape, dtype=bool)
    slot = next((candidate for candidate in preset.slots if candidate.name == slot_name), None)
    hole_masks = preset.slot_hole_masks.get(slot_name, ())
    if slot is None or not hole_masks:
        return np.zeros(shape, dtype=bool)

    height, width = shape
    scale_x = width / max(slot.box.width, 1)
    scale_y = height / max(slot.box.height, 1)
    mask = np.zeros(shape, dtype=np.uint8)
    for ellipse in hole_masks:
        center = (round(ellipse.cx * scale_x), round(ellipse.cy * scale_y))
        axes = (max(1, round(ellipse.rx * scale_x)), max(1, round(ellipse.ry * scale_y)))
        cv2.ellipse(mask, center, axes, ellipse.angle, 0, 360, 255, -1)
    if dilation > 0:
        kernel = np.ones((dilation, dilation), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask > 0


def _build_valid_region_mask(image: Any, slot_name: str | None, config: ValidRegionMaskConfig) -> Any:
    """Build a valid scoring mask for a crop image."""
    import numpy as np

    if config.preset == "none":
        return np.ones(image.shape[:2], dtype=bool)

    valid_mask = _foreground_region_mask(image, config)
    if slot_name:
        hole_mask = _preset_hole_mask(valid_mask.shape, slot_name, config.preset, config.hole_dilation)
        valid_mask = valid_mask & ~hole_mask
    return valid_mask


def _masked_score_from_map(
    anomaly_map: Any,
    valid_mask: Any,
    original_score: float,
    method: str,
    component_threshold_ratio: float,
    component_min_area: int,
) -> tuple[float, dict[str, float | int | str]]:
    """Return a valid-region score calibrated to the original prediction score scale."""
    import cv2
    import numpy as np

    anomaly_map = np.nan_to_num(anomaly_map.astype("float32"), copy=False)
    valid_mask = valid_mask.astype(bool)
    valid_values = anomaly_map[valid_mask]
    full_max = float(anomaly_map.max()) if anomaly_map.size else 0.0
    details: dict[str, float | int | str] = {
        "full_max": full_max,
        "valid_pixel_count": int(valid_values.size),
        "component_count": 0,
        "largest_component_area": 0,
        "score_fallback": "",
    }
    if valid_values.size == 0 or full_max <= 0:
        details["score_fallback"] = "original_empty_mask"
        return float(original_score), details

    selected_raw = float(valid_values.max())
    if method == "masked_component":
        threshold = full_max * component_threshold_ratio
        candidates = ((anomaly_map >= threshold) & valid_mask).astype(np.uint8)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, 8)
        accepted_raw_scores = []
        largest_area = 0
        accepted_count = 0
        for component_index in range(1, component_count):
            area = int(stats[component_index, cv2.CC_STAT_AREA])
            if area < component_min_area:
                continue
            accepted_count += 1
            largest_area = max(largest_area, area)
            accepted_raw_scores.append(float(anomaly_map[labels == component_index].max()))
        details["component_count"] = accepted_count
        details["largest_component_area"] = largest_area
        if accepted_raw_scores:
            selected_raw = max(accepted_raw_scores)
        else:
            details["score_fallback"] = "masked_max_no_component"
    elif method != "masked_max":
        msg = f"Unsupported valid-region score method: {method}"
        raise ValueError(msg)

    details["selected_raw_score"] = selected_raw
    score = float(original_score) * selected_raw / full_max
    return float(round(score, 8)), details


def _add_valid_region_score_columns(frame: Any, artifacts: dict[str, dict[str, Any]], args: Any, workflow: ModuleType) -> Any:
    """Add optional valid-region score columns and optionally drive deployment labels."""
    import cv2
    import numpy as np

    preset = _resolve_valid_region_preset(args.valid_region_mask_preset, args.view)
    frame = frame.copy()
    frame["deploy_score"] = frame["pred_score"]
    frame["deploy_score_source"] = "original"
    if args.valid_region_score_mode == "off" or preset == "none":
        return frame

    config = ValidRegionMaskConfig(
        preset=preset,
        hole_dilation=args.valid_region_hole_dilation,
        border_margin=args.valid_region_border_margin,
        foreground_threshold_scale=args.valid_region_foreground_threshold_scale,
        min_coverage=args.valid_region_min_coverage,
    )
    scores: list[float] = []
    labels: list[int | None] = []
    coverages: list[float] = []
    slots: list[str | None] = []
    component_counts: list[int] = []
    largest_component_areas: list[int] = []
    fallbacks: list[str] = []

    for row in frame.itertuples(index=False):
        image_path = Path(row.source_path if Path(row.source_path).is_file() else row.processed_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        artifact = artifacts.get(str(Path(row.processed_path).resolve()), {})
        anomaly_map = _to_2d_array(artifact.get("anomaly_map"))
        slot_name = _slot_name_from_path(str(row.source_path)) or _slot_name_from_path(str(row.processed_path))
        slots.append(slot_name)

        if image is None or anomaly_map is None:
            scores.append(float("nan"))
            labels.append(None)
            coverages.append(float("nan"))
            component_counts.append(0)
            largest_component_areas.append(0)
            fallbacks.append("missing_image_or_map")
            continue

        height, width = image.shape[:2]
        resized_map = cv2.resize(anomaly_map.astype("float32"), (width, height), interpolation=cv2.INTER_LINEAR)
        valid_mask = _build_valid_region_mask(image, slot_name, config)
        coverage = float(valid_mask.mean())
        coverages.append(coverage)
        if coverage < config.min_coverage:
            scores.append(float("nan"))
            labels.append(None)
            component_counts.append(0)
            largest_component_areas.append(0)
            fallbacks.append("coverage_below_min")
            continue

        score, details = _masked_score_from_map(
            resized_map,
            valid_mask,
            float(row.pred_score),
            args.valid_region_score_method,
            args.valid_region_component_threshold_ratio,
            args.valid_region_component_min_area,
        )
        scores.append(score)
        labels.append(None if row.deploy_threshold is None or workflow.pd.isna(row.deploy_threshold) else int(score > row.deploy_threshold))
        component_counts.append(int(details.get("component_count", 0)))
        largest_component_areas.append(int(details.get("largest_component_area", 0)))
        fallbacks.append(str(details.get("score_fallback", "")))

    frame["valid_region_mask_preset"] = preset
    frame["valid_region_score_method"] = args.valid_region_score_method
    frame["valid_region_slot"] = slots
    frame["valid_region_coverage"] = coverages
    frame["valid_region_score"] = scores
    frame["valid_region_pred_label"] = labels
    frame["valid_region_component_count"] = component_counts
    frame["valid_region_largest_component_area"] = largest_component_areas
    frame["valid_region_fallback"] = fallbacks

    if args.valid_region_score_mode == "deploy":
        usable_scores = ~workflow.pd.isna(frame["valid_region_score"])
        frame.loc[usable_scores, "deploy_score"] = frame.loc[usable_scores, "valid_region_score"]
        frame.loc[usable_scores, "deploy_score_source"] = "valid_region"
        if "deploy_pred_label" in frame:
            deploy_labels = frame["valid_region_pred_label"].where(usable_scores, frame["deploy_pred_label"])
            frame["deploy_pred_label"] = deploy_labels

    return frame


def _safe_artifact_stem(index: int, source_path: str, score: Any, result_type: str, workflow: ModuleType) -> str:
    """Build a compact, filesystem-safe artifact stem."""
    import re

    score_text = "nan" if workflow.pd.isna(score) else f"{float(score):.4f}"
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(source_path).stem)
    return f"{index:04d}_{result_type}_{stem}_score_{score_text}"


def _annotation_lines(row: Any, mask_note: str | None = None) -> list[str]:
    """Build annotation text for review images."""
    import math

    threshold = "none" if row.deploy_threshold is None else f"{float(row.deploy_threshold):.4f}"
    deploy_score = getattr(row, "deploy_score", row.pred_score)
    score_source = getattr(row, "deploy_score_source", "original")
    lines = [
        f"{row.result_type}  gt={row.gt_label} pred={row.review_pred_label}",
        f"score={float(deploy_score):.6f} source={score_source} threshold={threshold}",
        f"label={row.dataset_label} view={row.view}",
    ]
    valid_region_score = getattr(row, "valid_region_score", None)
    if valid_region_score is not None:
        try:
            if not math.isnan(float(valid_region_score)):
                method = getattr(row, "valid_region_score_method", "")
                coverage = getattr(row, "valid_region_coverage", float("nan"))
                lines.append(f"valid_score={float(valid_region_score):.6f} method={method} coverage={float(coverage):.3f}")
        except (TypeError, ValueError):
            pass
    if mask_note:
        lines.append(mask_note)
    return lines


def _annotate_image(image: Any, lines: list[str], result_type: str) -> Any:
    """Draw a readable text banner on an image."""
    import cv2

    colors = {
        "TP": (0, 180, 0),
        "TN": (235, 235, 235),
        "FP": (0, 165, 255),
        "FN": (0, 0, 255),
        "UNKNOWN": (255, 255, 255),
    }
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.55, min(1.1, image.shape[1] / 1800.0))
    thickness = max(1, round(font_scale * 2))
    line_height = round(32 * font_scale)
    banner_height = max(line_height * (len(lines) + 1), 44)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], banner_height), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.72, image, 0.28, 0)

    color = colors.get(result_type, colors["UNKNOWN"])
    for index, line in enumerate(lines, start=1):
        cv2.putText(image, line, (12, index * line_height), font, font_scale, color, thickness, cv2.LINE_AA)
    return image


def _heatmap_overlay(image: Any, anomaly_map: Any | None, lines: list[str], result_type: str) -> Any:
    """Create a heatmap overlay image."""
    import cv2

    if anomaly_map is None:
        return _annotate_image(image.copy(), [*lines, "anomaly_map=missing"], result_type)

    height, width = image.shape[:2]
    resized = cv2.resize(anomaly_map.astype("float32"), (width, height), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(_normalize_uint8(resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.55, heatmap, 0.45, 0)
    return _annotate_image(overlay, lines, result_type)


def _mask_overlay(
    image: Any,
    pred_mask: Any | None,
    anomaly_map: Any | None,
    mask_threshold: float,
    lines: list[str],
    result_type: str,
) -> Any:
    """Create a predicted-mask overlay image."""
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    mask_note = "mask=pred_mask"
    mask = pred_mask
    if mask is None and anomaly_map is not None:
        resized_map = cv2.resize(anomaly_map.astype("float32"), (width, height), interpolation=cv2.INTER_LINEAR)
        mask = _normalize_uint8(resized_map) > round(mask_threshold * 255)
        mask_note = f"mask=anomaly_map>{mask_threshold:.2f}"
    elif mask is not None:
        mask = cv2.resize(mask.astype("float32"), (width, height), interpolation=cv2.INTER_NEAREST) > 0.5

    if mask is None:
        return _annotate_image(image.copy(), [*lines, "mask=missing"], result_type)

    overlay = image.copy()
    color = np.zeros_like(image)
    color[:, :] = (0, 0, 255)
    if np.any(mask):
        blended = image[mask].astype(np.float32) * 0.35 + color[mask].astype(np.float32) * 0.65
        overlay[mask] = blended.clip(0, 255).astype(image.dtype)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 3)
    else:
        mask_note = f"{mask_note} mask_pixels=0"
    return _annotate_image(overlay, [*lines, mask_note], result_type)


def _write_review_artifacts(
    frame: Any,
    artifacts: dict[str, dict[str, Any]],
    output_dir: Path,
    args: Any,
    workflow: ModuleType,
) -> Path:
    """Write original, heatmap, and mask review images into TP/TN/FP/FN folders."""
    import cv2

    review_dir = output_dir / "review"
    for result_type in ("TP", "TN", "FP", "FN"):
        (review_dir / result_type).mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(frame.itertuples(index=False), start=1):
        result_type = row.result_type if row.result_type in {"TP", "TN", "FP", "FN"} else "UNKNOWN"
        result_dir = review_dir / result_type
        result_dir.mkdir(parents=True, exist_ok=True)

        image_path = Path(row.source_path if Path(row.source_path).is_file() else row.processed_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        artifact = artifacts.get(str(Path(row.processed_path).resolve()), {})
        anomaly_map = _to_2d_array(artifact.get("anomaly_map"))
        pred_mask = _to_2d_array(artifact.get("pred_mask"))
        deploy_score = getattr(row, "deploy_score", row.pred_score)
        base_name = _safe_artifact_stem(index, row.source_path, deploy_score, result_type, workflow)
        lines = _annotation_lines(row)

        cv2.imwrite(str(result_dir / f"{base_name}_image.png"), _annotate_image(image.copy(), lines, result_type))
        cv2.imwrite(
            str(result_dir / f"{base_name}_heatmap.png"),
            _heatmap_overlay(image, anomaly_map, lines, result_type),
        )
        cv2.imwrite(
            str(result_dir / f"{base_name}_mask.png"),
            _mask_overlay(image, pred_mask, anomaly_map, args.mask_threshold, lines, result_type),
        )

    return review_dir


def _label_text(value: Any, workflow: ModuleType) -> str:
    """Format a numeric prediction label."""
    if value is None or workflow.pd.isna(value):
        return "unknown"
    return "anomaly" if int(value) == 1 else "normal"


def _print_predictions(frame: Any, threshold: float | None, workflow: ModuleType) -> None:
    """Print compact prediction lines to stdout."""
    label_column = "deploy_pred_label" if threshold is not None else "anomalib_pred_label"
    for row in frame.itertuples(index=False):
        score_value = getattr(row, "deploy_score", row.pred_score)
        score = "nan" if workflow.pd.isna(score_value) else f"{float(score_value):.6f}"
        label = _label_text(getattr(row, label_column), workflow)
        print(f"{label:7s} score={score} {row.source_path}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    workflow = _load_workflow_module()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_path", type=Path, help="Raw or already-preprocessed image file/directory.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=workflow.DEFAULT_OUTPUT_ROOT,
        help="ZS32 workflow output root.",
    )
    parser.add_argument("--view", choices=sorted(workflow.VIEW_SPECS), default=DEFAULT_VIEW, help="Camera/view name.")
    parser.add_argument("--model", choices=workflow.MODELS, default="patchcore", help="Trained model type.")
    parser.add_argument("--ckpt-path", type=Path, help="Optional checkpoint path. If omitted, search output-root.")
    parser.add_argument(
        "--image-size",
        type=_parse_image_size,
        help="Model input size as height,width. Only used with --rebuild-model.",
    )
    parser.add_argument(
        "--visualizer-field-size",
        type=workflow._parse_field_size,
        help="Visualization panel size as width,height. Defaults to the model input aspect ratio.",
    )
    parser.add_argument(
        "--roi",
        default="auto",
        help="Raw-image ROI as x1,y1,x2,y2, full/none, or auto to read the workflow manifest.",
    )
    parser.add_argument(
        "--blue-removal",
        choices=("auto", "on", "off"),
        default="auto",
        help="Raw-image blue-mark removal. auto reads the workflow manifest when available.",
    )
    parser.add_argument(
        "--input-is-preprocessed",
        action="store_true",
        help="Skip OpenCV ROI/blue-mark preprocessing because input already matches workflow preprocessed images.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Optional deployment threshold. Defaults to reports/summary.csv.",
    )
    parser.add_argument("--output-dir", type=Path, help="Directory for preprocessed inputs and predictions.csv.")
    parser.add_argument("--visualize", action="store_true", help="Save anomalib prediction visualizations.")
    parser.add_argument(
        "--no-review-artifacts",
        action="store_true",
        help="Do not write TP/TN/FP/FN review images with heatmaps and masks.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Fallback normalized anomaly-map threshold used only when pred_mask is missing.",
    )
    parser.add_argument(
        "--valid-region-mask-preset",
        choices=_valid_region_preset_choices(),
        default="none",
        help="Optional crop preset used to exclude background, fixture, and preset hole/inpaint regions.",
    )
    parser.add_argument(
        "--valid-region-score-mode",
        choices=("off", "report", "deploy"),
        default="off",
        help="off disables scoring, report adds CSV columns, deploy also uses the valid-region score for labels.",
    )
    parser.add_argument(
        "--valid-region-score-method",
        choices=("masked_max", "masked_component"),
        default="masked_max",
        help="Valid-region score statistic. Use masked_max first; component mode removes tiny hot components.",
    )
    parser.add_argument(
        "--valid-region-hole-dilation",
        type=int,
        default=24,
        help="Pixels used to dilate preset hole/inpaint exclusion masks.",
    )
    parser.add_argument(
        "--valid-region-border-margin",
        type=int,
        default=8,
        help="Image border pixels excluded from automatic foreground masks.",
    )
    parser.add_argument(
        "--valid-region-foreground-threshold-scale",
        type=float,
        default=0.45,
        help="Scale applied to Otsu threshold when extracting the foreground valid region.",
    )
    parser.add_argument(
        "--valid-region-min-coverage",
        type=float,
        default=0.05,
        help="Minimum valid-mask coverage required before a valid-region score is used.",
    )
    parser.add_argument(
        "--valid-region-component-threshold-ratio",
        type=float,
        default=0.7,
        help="Component-mode raw-map threshold as a fraction of the full-map maximum.",
    )
    parser.add_argument(
        "--valid-region-component-min-area",
        type=int,
        default=64,
        help="Minimum connected-component area in resized image pixels for component-mode scoring.",
    )
    parser.add_argument(
        "--rebuild-model",
        action="store_true",
        help="Rebuild the model from workflow defaults instead of restoring model hyperparameters from the checkpoint.",
    )
    parser.add_argument("--accelerator", choices=("gpu", "cpu", "auto"), default="auto", help="Lightning accelerator.")
    parser.add_argument("--devices", type=int, default=1, help="Number of accelerator devices.")
    parser.add_argument(
        "--imagenet-dir",
        type=Path,
        default=workflow.DEFAULT_IMAGENETTE_DIR,
        help="EfficientAd ImageNette dir.",
    )
    parser.add_argument("--anomaly-dino-neighbors", type=int, default=1, help="AnomalyDINO nearest-neighbor count.")
    parser.add_argument(
        "--anomaly-dino-encoder",
        default="dinov2_vit_small_14",
        help="AnomalyDINO DINOv2 encoder name.",
    )
    parser.add_argument("--anomaly-dino-masking", action="store_true", help="Enable AnomalyDINO foreground masking.")
    parser.add_argument(
        "--anomaly-dino-coreset-subsampling",
        action="store_true",
        help="Enable AnomalyDINO coreset subsampling.",
    )
    parser.add_argument("--anomaly-dino-sampling-ratio", type=float, default=0.1, help="AnomalyDINO coreset ratio.")
    return parser


def main() -> None:
    """Run image inference."""
    workflow = _load_workflow_module()
    args = build_parser().parse_args()
    output_dir = _resolve_output_dir(args)
    image_size = args.image_size or workflow.DEFAULT_IMAGE_SIZE
    checkpoint_path = _resolve_checkpoint(args, workflow)
    threshold = _resolve_threshold(args, workflow)

    config = _resolve_preprocessing_config(args, workflow)
    prediction_path, source_map = _prepare_prediction_input(
        args.data_path,
        output_dir,
        config,
        args.input_is_preprocessed,
        workflow,
    )

    from anomalib.engine import Engine

    model = _load_model(args, checkpoint_path, image_size, output_dir / "visualizations")
    engine = Engine(
        accelerator=args.accelerator,
        devices=args.devices,
        default_root_dir=output_dir / "engine",
        logger=False,
    )
    predictions = engine.predict(
        model=model,
        data_path=prediction_path,
        ckpt_path=checkpoint_path if args.rebuild_model else None,
        return_predictions=True,
    )

    artifacts = _prediction_artifacts(predictions, workflow)
    frame = _prediction_frame(predictions, args, checkpoint_path, source_map, threshold, workflow)
    frame = _add_valid_region_score_columns(frame, artifacts, args, workflow)
    frame = _add_review_columns(frame, args, workflow)
    predictions_path = output_dir / "predictions.csv"
    frame.to_csv(predictions_path, index=False)
    review_dir = None
    if not args.no_review_artifacts:
        review_dir = _write_review_artifacts(frame, artifacts, output_dir, args, workflow)

    print(f"checkpoint: {checkpoint_path}")
    print(f"predictions_csv: {predictions_path}")
    if review_dir is not None:
        print(f"review_dir: {review_dir}")
    if threshold is None:
        print("threshold: not found; using anomalib_pred_label for console labels")
    else:
        print(f"threshold: {threshold:.6f}")
    _print_predictions(frame, threshold, workflow)


if __name__ == "__main__":
    main()
