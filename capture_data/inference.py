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


def _safe_artifact_stem(index: int, source_path: str, score: Any, result_type: str, workflow: ModuleType) -> str:
    """Build a compact, filesystem-safe artifact stem."""
    import re

    score_text = "nan" if workflow.pd.isna(score) else f"{float(score):.4f}"
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(source_path).stem)
    return f"{index:04d}_{result_type}_{stem}_score_{score_text}"


def _annotation_lines(row: Any, mask_note: str | None = None) -> list[str]:
    """Build annotation text for review images."""
    threshold = "none" if row.deploy_threshold is None else f"{float(row.deploy_threshold):.4f}"
    lines = [
        f"{row.result_type}  gt={row.gt_label} pred={row.review_pred_label}",
        f"score={float(row.pred_score):.6f} threshold={threshold}",
        f"label={row.dataset_label} view={row.view}",
    ]
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
        base_name = _safe_artifact_stem(index, row.source_path, row.pred_score, result_type, workflow)
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
        score = "nan" if workflow.pd.isna(row.pred_score) else f"{float(row.pred_score):.6f}"
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
