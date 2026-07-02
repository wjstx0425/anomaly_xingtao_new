# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end part defect detection workflow.

This script prepares local image folders, trains selected anomaly-detection
models per camera view, and writes image/sample-level evaluation reports.

常用命令如下。ROI 参数格式是 ``x1,y1,x2,y2``，例如
``--roi 460,30,3480,2600``；如果不裁剪，可以使用 ``--roi none``。

1. PatchCore 训练 + 评估::

        .venv/bin/python examples/api/03_models/zs32_defect_workflow.py all \
          --data-root dataset/c789 \
          --output-root results/c789/left_top_pc_512 \
          --views left_top \
          --models patchcore \
          --skip-blue-removal \
          --roi 460,30,3480,2600 \
          --image-size 512,1024 \
          --patchcore-batch-size 4 \
          --eval-batch-size 1 \
          --accelerator gpu

2. EfficientAd 训练 + 评估::

        .venv/bin/python examples/api/03_models/zs32_defect_workflow.py all \
          --data-root dataset/c789 \
          --output-root results/c789/left_top_efficientad \
          --views left_top \
          --models efficient_ad \
          --skip-blue-removal \
          --roi 460,30,3480,2600 \
          --image-size 392,392 \
          --efficientad-batch-size 1 \
          --eval-batch-size 1 \
          --efficientad-epochs 20 \
          --accelerator gpu

   EfficientAd 需要提前准备 teacher weights 和 ImageNette 数据；如果资源
   缺失并且网络不可用，可先用 ``--models patchcore`` 或添加
   ``--skip-missing-efficientad-assets`` 跳过 EfficientAd。

3. AnomalyDINO 训练 + 评估，显存安全版::

        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        .venv/bin/python examples/api/03_models/zs32_defect_workflow.py all \
          --data-root dataset/c789 \
          --output-root results/c789/left_top_anomaly_dino \
          --views left_top \
          --models anomaly_dino \
          --skip-blue-removal \
          --roi 460,30,3480,2600 \
          --image-size 392,392 \
          --anomaly-dino-batch-size 1 \
          --eval-batch-size 1 \
          --anomaly-dino-coreset-subsampling \
          --anomaly-dino-sampling-ratio 0.03 \
          --accelerator gpu

   如果显存充足，可逐步尝试 ``--anomaly-dino-sampling-ratio 0.05``、
   ``0.07``、``0.1``，或者把 ``--image-size`` 从 ``392,392`` 提到
   ``448,448``。如果出现 CUDA OOM，优先保持 ``--eval-batch-size 1``，
   再降低 ``--anomaly-dino-sampling-ratio`` 或 ``--image-size``。

也可以分阶段运行::

        .venv/bin/python examples/api/03_models/zs32_defect_workflow.py preprocess ...
        .venv/bin/python examples/api/03_models/zs32_defect_workflow.py train ...
        .venv/bin/python examples/api/03_models/zs32_defect_workflow.py evaluate ...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import urllib.error
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))

import cv2  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = REPO_ROOT / "dataset" / "ZS32"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "zs32"
DEFAULT_IMAGENETTE_DIR = REPO_ROOT / ".cache" / "anomalib" / "imagenette"
DEFAULT_ROI = (400, 1200, 3650, 2900)
DEFAULT_IMAGE_SIZE = (384, 768)
DEFAULT_EXTENSIONS = (".png",)
EXPECTED_ALL_IMAGE_COUNT = 340
LABELS = ("normal", "normal_test", "defect")
MODELS = ("patchcore", "efficient_ad", "anomaly_dino")
EFFICIENTAD_WEIGHTS_URL = (
    "https://github.com/open-edge-platform/anomalib/releases/download/"
    "efficientad_pretrained_weights/efficientad_pretrained_weights.zip"
)
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
IMAGE_EXTENSIONS = (".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff")


def _log(message: str) -> None:
    """Print a progress message immediately."""
    print(message, flush=True)


@dataclass(frozen=True)
class ViewSpec:
    """Location of one ZS32 view in the raw dataset."""

    name: str
    raw_parts: tuple[str, str]
    fallback_raw_parts: tuple[tuple[str, str], ...] = ()


VIEW_SPECS = {
    "left_top": ViewSpec(name="left_top", raw_parts=("left", "top")),
    "left_bottom": ViewSpec(
        name="left_bottom",
        raw_parts=("left", "bottom_ZS32"),
        fallback_raw_parts=(("left", "bottom"),),
    ),
    "right_top": ViewSpec(name="right_top", raw_parts=("right", "top")),
    "right_bottom": ViewSpec(
        name="right_bottom",
        raw_parts=("right", "bottom_ZS32"),
        fallback_raw_parts=(("right", "bottom"),),
    ),
    "no_hand_top": ViewSpec(name="no_hand_top", raw_parts=("no_hand", "top")),
    "no_hand_bottom": ViewSpec(name="no_hand_bottom", raw_parts=("no_hand", "bottom")),
}
DEFAULT_VIEW_NAMES = ("left_top", "left_bottom", "right_top", "right_bottom")


def _repo_relative(path: Path) -> str:
    """Return a readable path relative to the repository when possible."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _parse_roi(value: str) -> tuple[int, int, int, int] | None:
    """Parse an ROI in x1,y1,x2,y2 format, or disable cropping."""
    if value.lower() in {"none", "full"}:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        msg = "ROI must be provided as x1,y1,x2,y2, none, or full."
        raise argparse.ArgumentTypeError(msg)
    try:
        x1, y1, x2, y2 = (int(part) for part in parts)
    except ValueError as error:
        msg = "ROI coordinates must be integers."
        raise argparse.ArgumentTypeError(msg) from error
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        msg = "ROI must satisfy x1 >= 0, y1 >= 0, x2 > x1, y2 > y1."
        raise argparse.ArgumentTypeError(msg)
    return x1, y1, x2, y2


def _format_roi(roi: tuple[int, int, int, int] | None) -> str:
    """Return the manifest/log representation for an ROI."""
    return "full" if roi is None else ",".join(str(value) for value in roi)


def _parse_image_size(value: str) -> tuple[int, int]:
    """Parse an image size in height,width format."""
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        msg = "Image size must be provided as height,width."
        raise argparse.ArgumentTypeError(msg)
    try:
        height, width = (int(part) for part in parts)
    except ValueError as error:
        msg = "Image size values must be integers."
        raise argparse.ArgumentTypeError(msg) from error
    if height <= 0 or width <= 0:
        msg = "Image size values must be positive."
        raise argparse.ArgumentTypeError(msg)
    return height, width


def _parse_field_size(value: str) -> tuple[int, int]:
    """Parse a visualization field size in width,height format."""
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        msg = "Visualizer field size must be provided as width,height."
        raise argparse.ArgumentTypeError(msg)
    try:
        width, height = (int(part) for part in parts)
    except ValueError as error:
        msg = "Visualizer field size values must be integers."
        raise argparse.ArgumentTypeError(msg) from error
    if width <= 0 or height <= 0:
        msg = "Visualizer field size values must be positive."
        raise argparse.ArgumentTypeError(msg)
    return width, height


def _parse_deploy_fpr(value: str) -> float:
    """Parse the allowed normal_test false-positive rate for deployment."""
    try:
        false_positive_rate = float(value)
    except ValueError as error:
        msg = "Deployment false-positive rate must be a number between 0 and 1."
        raise argparse.ArgumentTypeError(msg) from error
    if false_positive_rate < 0 or false_positive_rate >= 1:
        msg = "Deployment false-positive rate must satisfy 0 <= value < 1."
        raise argparse.ArgumentTypeError(msg)
    return false_positive_rate


def _parse_sampling_ratio(value: str) -> float:
    """Parse a sampling ratio in the open-closed range (0, 1]."""
    try:
        sampling_ratio = float(value)
    except ValueError as error:
        msg = "Sampling ratio must be a number between 0 and 1."
        raise argparse.ArgumentTypeError(msg) from error
    if sampling_ratio <= 0 or sampling_ratio > 1:
        msg = "Sampling ratio must satisfy 0 < value <= 1."
        raise argparse.ArgumentTypeError(msg)
    return sampling_ratio


def _parse_run_suffix(value: str) -> str:
    """Parse a safe run suffix used in generated run names."""
    if not value:
        return ""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        msg = "Run suffix may contain only letters, numbers, underscore, dash, and dot."
        raise argparse.ArgumentTypeError(msg)
    return value


def _parse_reports_dir_name(value: str) -> Path:
    """Parse a relative reports directory name."""
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        msg = "Reports directory name must be a relative path without '..'."
        raise argparse.ArgumentTypeError(msg)
    return path


def _visualizer_field_size(image_size: tuple[int, int], args: argparse.Namespace) -> tuple[int, int]:
    """Return visualization field size as width,height."""
    if args.visualizer_field_size is not None:
        return args.visualizer_field_size
    height, width = image_size
    return width, height


def _selected_views(views: Sequence[str] | None) -> list[str]:
    """Return normalized view names."""
    return list(views) if views else list(DEFAULT_VIEW_NAMES)


def _selected_models(models: Sequence[str] | None) -> list[str]:
    """Return normalized model names."""
    return list(models) if models else list(MODELS)


def _contains_images(path: Path) -> bool:
    """Return whether a directory contains at least one common image file."""
    return path.is_dir() and any(child.suffix.lower() in IMAGE_EXTENSIONS for child in path.rglob("*"))


def _efficientad_asset_status(args: argparse.Namespace) -> dict[str, Any]:
    """Return local EfficientAd asset paths and availability flags."""
    from anomalib.utils.path import get_pretrained_weights_dir

    pretrained_dir = get_pretrained_weights_dir()
    teacher_dir = pretrained_dir / "efficientad_pretrained_weights"
    teacher_path = teacher_dir / "pretrained_teacher_medium.pth"
    imagenet_dir = args.imagenet_dir.resolve()
    return {
        "teacher_path": teacher_path,
        "teacher_exists": teacher_path.is_file(),
        "weights_archive_path": pretrained_dir / "efficientad_pretrained_weights.zip",
        "imagenet_dir": imagenet_dir,
        "imagenet_has_images": _contains_images(imagenet_dir),
    }


def _efficientad_missing_assets(args: argparse.Namespace) -> list[str]:
    """Return human-readable EfficientAd assets that are missing locally."""
    status = _efficientad_asset_status(args)
    missing = []
    if not status["teacher_exists"]:
        missing.append(f"teacher weights: {status['teacher_path']}")
    if not status["imagenet_has_images"]:
        missing.append(f"ImageNette images under: {status['imagenet_dir']}")
    return missing


def _log_efficientad_asset_help(args: argparse.Namespace) -> None:
    """Log download/cache instructions for EfficientAd assets."""
    status = _efficientad_asset_status(args)
    _log("EfficientAd needs two local resources before offline training:")
    _log(f"  1. Teacher weights: {status['teacher_path']}")
    _log(f"     Download: {EFFICIENTAD_WEIGHTS_URL}")
    _log(f"     Extract into: {status['teacher_path'].parent}")
    _log(f"  2. ImageNette images under: {status['imagenet_dir']}")
    _log(f"     Download: {IMAGENETTE_URL}")
    _log(f"     Extract into: {status['imagenet_dir']}")
    _log("To run only PatchCore while the network is unavailable, use `--models patchcore`.")


def _looks_like_network_error(error: BaseException) -> bool:
    """Return whether an exception chain looks like a download/network failure."""
    current: BaseException | None = error
    error_text_parts = []
    while current is not None:
        if isinstance(current, urllib.error.URLError | socket.gaierror | TimeoutError | ConnectionError):
            return True
        error_text_parts.append(str(current).lower())
        current = current.__cause__ or current.__context__
    error_text = "\n".join(error_text_parts)
    return any(
        fragment in error_text
        for fragment in [
            "cannot send a request",
            "client has been closed",
            "temporary failure in name resolution",
            "name resolution",
            "urlopen error",
            "connection timed out",
            "network is unreachable",
        ]
    )


def _raise_if_pretrained_resource_error(model_name: str, args: argparse.Namespace, error: BaseException) -> None:
    """Raise a clearer error when model setup needs unavailable pretrained resources."""
    if not _looks_like_network_error(error):
        return

    if model_name == "efficient_ad":
        _log_efficientad_asset_help(args)
        msg = (
            "EfficientAd tried to download required assets but the network/DNS lookup failed. "
            "Fix DNS/network access, place the assets in the paths above, or rerun with "
            "`--models patchcore` / `--skip-missing-efficientad-assets`."
        )
        raise RuntimeError(msg) from None

    msg = (
        f"{model_name} tried to download a pretrained resource but the network/DNS lookup failed. "
        "Fix network access or cache the required pretrained files before rerunning."
    )
    if model_name == "patchcore":
        msg += f" PatchCore needs cached timm/HuggingFace weights for `{args.patchcore_backbone}`."
    raise RuntimeError(msg) from None


def _raw_view_dir(data_root: Path, view: str) -> Path:
    """Return the primary raw directory for one view."""
    spec = VIEW_SPECS[view]
    return data_root / spec.raw_parts[0] / spec.raw_parts[1]


def _raw_view_dirs(data_root: Path, view: str) -> list[Path]:
    """Return primary and fallback raw directories for one view."""
    spec = VIEW_SPECS[view]
    raw_parts = (spec.raw_parts, *spec.fallback_raw_parts)
    view_dirs = []
    seen = set()
    for hand, position in raw_parts:
        view_dir = data_root / hand / position
        if view_dir not in seen:
            view_dirs.append(view_dir)
            seen.add(view_dir)
    return view_dirs


def _raw_label_dirs(data_root: Path, view: str, label: str) -> list[Path]:
    """Return candidate raw label directories for one view."""
    return [view_dir / label for view_dir in _raw_view_dirs(data_root, view)]


def _processed_view_dir(output_root: Path, view: str) -> Path:
    """Return the preprocessed directory for one view."""
    return output_root / "preprocessed" / view


def _manifest_path(output_root: Path) -> Path:
    """Return the workflow manifest path."""
    return output_root / "preprocessed" / "manifest.csv"


def _remove_generated_path(path: Path) -> None:
    """Remove a generated file or directory if it exists."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _clear_preprocess_outputs(output_root: Path, views: Sequence[str]) -> None:
    """Clear generated preprocessing outputs for selected views."""
    for view in views:
        view_dir = _processed_view_dir(output_root, view)
        if view_dir.exists():
            _remove_generated_path(view_dir)
            _log(f"Cleared previous preprocessed view: {_repo_relative(view_dir)}")

    preprocessed_root = output_root / "preprocessed"
    for metadata_path in (_manifest_path(output_root), preprocessed_root / "counts.csv"):
        if metadata_path.exists():
            _remove_generated_path(metadata_path)


def _extract_frame_id(image_path: Path) -> str:
    """Extract the six-digit frame id from a ZS32 image filename."""
    match = re.search(r"_(\d{6})$", image_path.stem)
    return match.group(1) if match else image_path.stem


def _iter_raw_images(data_root: Path, view: str, label: str) -> Iterable[Path]:
    """Yield raw images for one view and label."""
    for label_dir in _raw_label_dirs(data_root, view, label):
        nested_images = sorted(label_dir.glob("*/images/*.png"))
        if nested_images:
            yield from nested_images
            return

        flat_images = sorted(path for path in label_dir.glob("*.png") if path.is_file())
        if flat_images:
            yield from flat_images
            return


def _raw_sample_id(image_path: Path) -> str:
    """Return the sample id for nested ZS32 or flat label-directory images."""
    if image_path.parent.name == "images":
        return image_path.parent.parent.name
    return image_path.stem


def _remove_blue_marks(image: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Remove blue manual marks using HSV thresholding and inpainting."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([90, 50, 40]), np.array([135, 255, 255]))
    blue_mask_area = int(np.count_nonzero(blue_mask))
    kernel = np.ones((5, 5), dtype=np.uint8)
    expanded_mask = cv2.dilate(blue_mask, kernel, iterations=2)
    cleaned = cv2.inpaint(image, expanded_mask, 7, cv2.INPAINT_TELEA) if blue_mask_area else image
    blue_mask_ratio = blue_mask_area / float(image.shape[0] * image.shape[1])
    return cleaned, blue_mask_area, blue_mask_ratio


def _preprocess_image(
    source_path: Path,
    output_path: Path,
    roi: tuple[int, int, int, int] | None,
    remove_blue_marks: bool,
) -> dict[str, Any]:
    """Crop one image to the ROI and optionally remove blue marks."""
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        msg = f"Could not read image: {source_path}"
        raise RuntimeError(msg)

    source_height, source_width = image.shape[:2]
    if roi is None:
        cropped = image
    else:
        x1, y1, x2, y2 = roi
        if x2 > source_width or y2 > source_height:
            msg = f"ROI {roi} is outside image {source_width}x{source_height}: {source_path}"
            raise ValueError(msg)
        cropped = image[y1:y2, x1:x2]

    if remove_blue_marks:
        cleaned, blue_mask_area, blue_mask_ratio = _remove_blue_marks(cropped)
    else:
        cleaned = cropped
        blue_mask_area = None
        blue_mask_ratio = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), cleaned):
        msg = f"Could not write image: {output_path}"
        raise RuntimeError(msg)

    return {
        "source_width": source_width,
        "source_height": source_height,
        "roi_width": cropped.shape[1],
        "roi_height": cropped.shape[0],
        "blue_removal_enabled": remove_blue_marks,
        "blue_mask_area": blue_mask_area,
        "blue_mask_area_ratio": blue_mask_ratio,
    }


def preprocess_dataset(args: argparse.Namespace) -> Path:
    """Preprocess ZS32 raw images and write the manifest."""
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    roi = args.roi
    remove_blue_marks = not args.skip_blue_removal
    manifest_rows: list[dict[str, Any]] = []
    processed_count = 0
    selected_views = _selected_views(args.views)

    _log(f"Preprocessing ZS32 images from {_repo_relative(data_root)}")
    _log(f"Output root: {_repo_relative(output_root)}")
    _log(f"ROI: {_format_roi(roi)}")
    _log(f"Blue mark removal: {'enabled' if remove_blue_marks else 'disabled'}")

    _clear_preprocess_outputs(output_root, selected_views)

    for view in selected_views:
        _log(f"Preparing view: {view}")
        for label in LABELS:
            image_paths = list(_iter_raw_images(data_root, view, label))
            if not image_paths:
                checked_dirs = ", ".join(_repo_relative(path) for path in _raw_label_dirs(data_root, view, label))
                _log(f"Warning: no images found for {view}/{label}; checked: {checked_dirs}.")
                continue

            _log(f"  {view}/{label}: {len(image_paths)} images")

            for index, source_path in enumerate(image_paths, start=1):
                sample_id = _raw_sample_id(source_path)
                frame_id = _extract_frame_id(source_path)
                output_path = (
                    _processed_view_dir(output_root, view)
                    / label
                    / sample_id
                    / "images"
                    / source_path.name
                )
                stats = _preprocess_image(source_path, output_path, roi, remove_blue_marks)
                manifest_rows.append(
                    {
                        "source_path": str(source_path.resolve()),
                        "processed_path": str(output_path.resolve()),
                        "view": view,
                        "label": label,
                        "sample_id": sample_id,
                        "frame_id": frame_id,
                        "roi": _format_roi(roi),
                        **stats,
                    },
                )
                processed_count += 1
                if index == len(image_paths) or index % 25 == 0:
                    _log(f"    processed {index}/{len(image_paths)} for {view}/{label}")

    if not manifest_rows:
        msg = f"No images were preprocessed from {data_root}."
        raise RuntimeError(msg)

    manifest = pd.DataFrame(manifest_rows).sort_values(["view", "label", "sample_id", "frame_id"])
    manifest_path = _manifest_path(output_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    count_table = manifest.groupby(["view", "label"], sort=True).size().rename("count").reset_index()
    count_path = manifest_path.parent / "counts.csv"
    count_table.to_csv(count_path, index=False)

    selected_all_views = set(selected_views) == set(DEFAULT_VIEW_NAMES)
    if selected_all_views and len(manifest) != EXPECTED_ALL_IMAGE_COUNT:
        _log(f"Warning: expected {EXPECTED_ALL_IMAGE_COUNT} images, but preprocessed {len(manifest)}.")

    _log(f"Preprocessed {processed_count} images.")
    _log(f"Manifest: {_repo_relative(manifest_path)}")
    _log(f"Counts: {_repo_relative(count_path)}")
    return manifest_path


def _load_manifest(output_root: Path) -> pd.DataFrame:
    """Load the preprocessing manifest."""
    manifest_path = _manifest_path(output_root)
    if not manifest_path.is_file():
        msg = f"Missing manifest: {manifest_path}. Run the preprocess command first."
        raise FileNotFoundError(msg)
    manifest = pd.read_csv(manifest_path)
    manifest["processed_path"] = manifest["processed_path"].map(lambda value: str(Path(value).resolve()))
    return manifest


def _log_preprocessing_config(manifest: pd.DataFrame) -> None:
    """Log preprocessing settings recorded in the manifest."""
    if "roi" in manifest:
        roi_values = sorted(str(value) for value in manifest["roi"].dropna().unique())
        if roi_values:
            suffix = "..." if len(roi_values) > 3 else ""
            _log(f"Preprocessed ROI: {', '.join(roi_values[:3])}{suffix}")
    if "blue_removal_enabled" in manifest:
        blue_values = sorted(str(value) for value in manifest["blue_removal_enabled"].dropna().unique())
        if blue_values:
            _log(f"Preprocessed blue mark removal: {', '.join(blue_values)}")


def _sample_train_data(datamodule: Any, sampling_ratio: float, seed: int) -> None:
    """Keep a deterministic fraction of training normal images."""
    if sampling_ratio >= 1:
        return

    samples = datamodule.train_data.samples
    sample_count = max(1, int(np.ceil(len(samples) * sampling_ratio)))
    sampled = samples.sample(n=sample_count, random_state=seed).reset_index(drop=True)
    sampled.attrs = samples.attrs.copy()
    datamodule.train_data.samples = sampled
    _log(f"Train sampling ratio: {sampling_ratio:g} ({sample_count}/{len(samples)} normal images).")


def _build_datamodule(
    output_root: Path,
    view: str,
    model_name: str,
    args: argparse.Namespace,
) -> Any:
    """Build an Anomalib Folder datamodule for one preprocessed view."""
    from anomalib.data import Folder

    if model_name == "efficient_ad":
        train_batch_size = args.efficientad_batch_size
    elif model_name == "anomaly_dino":
        train_batch_size = args.anomaly_dino_batch_size
    else:
        train_batch_size = args.patchcore_batch_size
    datamodule = Folder(
        name=f"zs32_{view}",
        root=_processed_view_dir(output_root, view),
        normal_dir="normal",
        abnormal_dir="defect",
        normal_test_dir="normal_test",
        normal_split_ratio=0,
        extensions=DEFAULT_EXTENSIONS,
        train_batch_size=train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        test_split_mode="from_dir",
        test_split_ratio=0.2,
        val_split_mode="same_as_test",
        val_split_ratio=0.5,
        seed=args.seed,
    )
    datamodule.category = view
    datamodule.setup()
    _sample_train_data(datamodule, args.train_sampling_ratio, args.seed)
    datamodule._is_setup = True  # noqa: SLF001
    return datamodule


def _build_model(
    model_name: str,
    image_size: tuple[int, int],
    visualizations_dir: Path,
    args: argparse.Namespace,
) -> Any:
    """Build a configured anomaly detection model."""
    from anomalib.models import AnomalyDINO, EfficientAd, Patchcore
    from anomalib.visualization import ImageVisualizer

    visualizer = ImageVisualizer(
        fields=["image", "anomaly_map"],
        overlay_fields=[("image", ["anomaly_map"]), ("image", ["pred_mask"])],
        field_size=_visualizer_field_size(image_size, args),
        output_dir=visualizations_dir,
    )

    if model_name == "patchcore":
        return Patchcore(
            backbone=args.patchcore_backbone,
            layers=tuple(args.patchcore_layers),
            pre_trained=True,
            coreset_sampling_ratio=args.patchcore_coreset_ratio,
            num_neighbors=args.patchcore_num_neighbors,
            precision=args.patchcore_precision,
            pre_processor=Patchcore.configure_pre_processor(image_size=image_size),
            visualizer=visualizer,
        )
    if model_name == "efficient_ad":
        return EfficientAd(
            imagenet_dir=args.imagenet_dir.resolve(),
            teacher_out_channels=384,
            model_size="medium",
            lr=1e-4,
            pre_processor=EfficientAd.configure_pre_processor(image_size=image_size),
            visualizer=visualizer,
        )
    if model_name == "anomaly_dino":
        return AnomalyDINO(
            num_neighbours=args.anomaly_dino_neighbors,
            encoder_name=args.anomaly_dino_encoder,
            masking=args.anomaly_dino_masking,
            coreset_subsampling=args.anomaly_dino_coreset_subsampling,
            sampling_ratio=args.anomaly_dino_sampling_ratio,
            pre_processor=AnomalyDINO.configure_pre_processor(image_size=image_size),
            visualizer=visualizer,
        )

    msg = f"Unsupported model: {model_name}"
    raise ValueError(msg)


def _assert_accelerator_ready(accelerator: str) -> None:
    """Fail early when GPU training is requested but CUDA is not visible."""
    if accelerator != "gpu":
        return

    import torch

    if not torch.cuda.is_available():
        msg = (
            "GPU execution was requested, but torch.cuda.is_available() is False. "
            "Run `source .venv/bin/activate` in your terminal and verify "
            "`python -c \"import torch; print(torch.cuda.get_device_name(0))\"` before training."
        )
        raise RuntimeError(msg)

    _log(f"Using GPU: {torch.cuda.get_device_name(0)}")


def _json_safe(value: Any) -> Any:
    """Convert tensors and numpy scalars to JSON-safe values."""
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.numel() == 1 else tensor.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value


def _model_run_name(model_name: str, args: argparse.Namespace) -> str:
    """Return the model run/report name for a possibly suffixed experiment."""
    suffix = getattr(args, "model_run_suffix", "")
    return f"{model_name}_{suffix}" if suffix else model_name


def _run_dir(output_root: Path, view: str, run_name: str) -> Path:
    """Return the run directory for one experiment."""
    return output_root / "runs" / view / run_name


def train_experiments(args: argparse.Namespace) -> None:
    """Train all selected ZS32 experiments."""
    output_root = args.output_root.resolve()
    model_names = _selected_models(args.models)
    _log("Starting training stage.")
    _log(f"Views: {', '.join(_selected_views(args.views))}")
    _log(f"Models: {', '.join(model_names)}")
    manifest = _load_manifest(output_root)
    _log_preprocessing_config(manifest)
    _assert_accelerator_ready(args.accelerator)

    if "efficient_ad" in model_names:
        missing_assets = _efficientad_missing_assets(args)
        if missing_assets:
            _log("EfficientAd local assets are missing:")
            for missing_asset in missing_assets:
                _log(f"  - {missing_asset}")
            _log_efficientad_asset_help(args)
            if args.skip_missing_efficientad_assets:
                model_names = [model_name for model_name in model_names if model_name != "efficient_ad"]
                _log("Skipping EfficientAd because --skip-missing-efficientad-assets was set.")
                if not model_names:
                    msg = "No models left to train after skipping EfficientAd."
                    raise RuntimeError(msg)
            else:
                _log("Continuing anyway; anomalib will try to download the missing EfficientAd assets.")

    from anomalib.engine import Engine

    for view in _selected_views(args.views):
        for model_name in model_names:
            run_name = _model_run_name(model_name, args)
            run_dir = _run_dir(output_root, view, run_name)
            visualizations_dir = run_dir / "visualizations" / "test"
            run_dir.mkdir(parents=True, exist_ok=True)

            _log(f"Training {run_name} on {view}...")
            if model_name == "patchcore":
                _log(
                    "PatchCore config: "
                    f"backbone={args.patchcore_backbone}, "
                    f"layers={','.join(args.patchcore_layers)}, "
                    f"coreset_ratio={args.patchcore_coreset_ratio}, "
                    f"num_neighbors={args.patchcore_num_neighbors}, "
                    f"precision={args.patchcore_precision}",
                )
            try:
                datamodule = _build_datamodule(output_root, view, model_name, args)
                model = _build_model(model_name, args.image_size, visualizations_dir, args)
                max_epochs = args.efficientad_epochs if model_name == "efficient_ad" else 1
                engine = Engine(
                    accelerator=args.accelerator,
                    devices=args.devices,
                    max_epochs=max_epochs,
                    default_root_dir=run_dir,
                    logger=False,
                )
                engine.fit(model=model, datamodule=datamodule)
                test_metrics = engine.test(model=model, datamodule=datamodule)
            except Exception as error:
                _raise_if_pretrained_resource_error(model_name, args, error)
                raise

            metrics_path = run_dir / "test_metrics.json"
            with metrics_path.open("w", encoding="utf-8") as file:
                json.dump(_json_safe(test_metrics), file, indent=2)
            _log(f"Finished {run_name}/{view}. Metrics: {_repo_relative(metrics_path)}")


def _find_checkpoint(run_dir: Path) -> Path:
    """Find the newest Lightning checkpoint under a run directory."""
    checkpoints = sorted(run_dir.rglob("*.ckpt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not checkpoints:
        msg = f"No checkpoint found under {run_dir}. Run the train command first."
        raise FileNotFoundError(msg)
    model_checkpoints = [path for path in checkpoints if path.name == "model.ckpt"]
    return model_checkpoints[0] if model_checkpoints else checkpoints[0]


def _iter_prediction_items(predictions: Any) -> Iterable[Any]:
    """Yield item-level predictions from Lightning prediction output."""
    if predictions is None:
        return
    for batch in predictions:
        if batch is None:
            continue
        if hasattr(batch, "image_path") and isinstance(batch.image_path, list):
            yield from batch
        elif hasattr(batch, "image_path"):
            yield batch
        elif isinstance(batch, list | tuple):
            yield from _iter_prediction_items(batch)


def _to_scalar(value: Any) -> Any:
    """Convert tensor-like scalar values to Python scalars."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        if tensor.numel() == 0:
            return None
        return tensor.flatten()[0].item()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return value.reshape(-1)[0].item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _predictions_to_frame(predictions: Any, model_name: str) -> pd.DataFrame:
    """Convert Anomalib prediction batches into a DataFrame."""
    rows = []
    for item in _iter_prediction_items(predictions):
        image_path = getattr(item, "image_path", None)
        if image_path is None:
            continue
        rows.append(
            {
                "processed_path": str(Path(image_path).resolve()),
                "model": model_name,
                "pred_score": _to_scalar(getattr(item, "pred_score", None)),
                "anomalib_pred_label": _to_scalar(getattr(item, "pred_label", None)),
            },
        )
    return pd.DataFrame(rows)


def _binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | int]:
    """Compute binary classification metrics."""
    true = y_true.astype(int)
    pred = y_pred.astype(int)
    tp = int(((true == 1) & (pred == 1)).sum())
    tn = int(((true == 0) & (pred == 0)).sum())
    fp = int(((true == 0) & (pred == 1)).sum())
    fn = int(((true == 1) & (pred == 0)).sum())
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def _summarize_sample_level(group: pd.DataFrame) -> pd.DataFrame:
    """Collapse five frame predictions into one sample-level prediction."""
    return (
        group.groupby(["model", "view", "label", "sample_id"], as_index=False)
        .agg(
            pred_score=("pred_score", "max"),
            deploy_threshold=("deploy_threshold", "first"),
            image_count=("processed_path", "count"),
        )
        .assign(
            gt_label=lambda frame: (frame["label"] == "defect").astype(int),
            deploy_pred_label=lambda frame: (frame["pred_score"] > frame["deploy_threshold"]).astype(int),
        )
    )


def _deployment_threshold(scores: pd.Series, false_positive_rate: float) -> float:
    """Return a threshold that permits about the requested normal false-positive rate."""
    clean_scores = scores.dropna().sort_values().reset_index(drop=True)
    if clean_scores.empty:
        return 0.0

    allowed_false_positives = int(np.floor(len(clean_scores) * false_positive_rate))
    if allowed_false_positives <= 0:
        return float(clean_scores.iloc[-1])

    threshold_index = max(0, len(clean_scores) - allowed_false_positives - 1)
    return float(clean_scores.iloc[threshold_index])


def _add_deployment_predictions(predictions: pd.DataFrame, deploy_fpr: float = 0.0) -> pd.DataFrame:
    """Add normal-test-based deployment thresholds and predictions."""
    frames = []
    for (_model, _view), group in predictions.groupby(["model", "view"], sort=True):
        normal_test = group[group["label"] == "normal_test"]
        threshold_scores = normal_test["pred_score"] if not normal_test.empty else group["pred_score"]
        threshold = _deployment_threshold(threshold_scores, deploy_fpr)
        group = group.copy()
        group["deploy_threshold"] = threshold
        group["deploy_target_fpr"] = deploy_fpr
        group["deploy_pred_label"] = (group["pred_score"] > threshold).astype(int)
        group["gt_label"] = (group["label"] == "defect").astype(int)
        frames.append(group)
    return pd.concat(frames, ignore_index=True) if frames else predictions


def _write_score_histogram(group: pd.DataFrame, output_path: Path) -> None:
    """Write a score histogram for one model/view group."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for label, color in [("normal_test", "tab:green"), ("defect", "tab:red")]:
        scores = group.loc[group["label"] == label, "pred_score"].dropna()
        if not scores.empty:
            plt.hist(scores, bins=20, alpha=0.55, label=label, color=color)
    threshold = float(group["deploy_threshold"].iloc[0])
    plt.axvline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.4f}")
    plt.title(f"{group['model'].iloc[0]} / {group['view'].iloc[0]}")
    plt.xlabel("Prediction score")
    plt.ylabel("Image count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _false_positive_rate(metrics: dict[str, float | int]) -> float:
    """Return the false-positive rate from binary metrics."""
    false_positives = int(metrics["fp"])
    true_negatives = int(metrics["tn"])
    normal_count = false_positives + true_negatives
    return false_positives / normal_count if normal_count else 0.0


def _copy_error_examples(errors: pd.DataFrame, output_dir: Path) -> None:
    """Copy representative false positive or false negative preprocessed images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, row in errors.head(5).reset_index(drop=True).iterrows():
        source = Path(row["processed_path"])
        if source.is_file():
            score = float(row["pred_score"])
            destination = output_dir / f"{index:02d}_score_{score:.4f}_{source.name}"
            shutil.copy2(source, destination)


def _build_summary_and_reports(predictions: pd.DataFrame, reports_dir: Path, deploy_fpr: float = 0.0) -> pd.DataFrame:
    """Write CSV, Markdown, histogram, and error-example reports."""
    predictions = _add_deployment_predictions(predictions, deploy_fpr)
    eval_predictions = predictions[predictions["label"].isin(["normal_test", "defect"])].copy()
    summary_rows = []
    false_positives = []
    false_negatives = []

    for (model_name, view), group in eval_predictions.groupby(["model", "view"], sort=True):
        image_metrics = _binary_metrics(group["gt_label"], group["deploy_pred_label"])
        sample_group = _summarize_sample_level(group)
        sample_metrics = _binary_metrics(sample_group["gt_label"], sample_group["deploy_pred_label"])

        adaptive_metrics = {}
        adaptive_rows = group.dropna(subset=["anomalib_pred_label"])
        if not adaptive_rows.empty:
            adaptive_metrics = _binary_metrics(
                adaptive_rows["gt_label"],
                adaptive_rows["anomalib_pred_label"].astype(int),
            )

        threshold = float(group["deploy_threshold"].iloc[0])
        summary_rows.append(
            {
                "model": model_name,
                "view": view,
                "deploy_threshold": threshold,
                "deploy_target_fpr": deploy_fpr,
                "image_fpr": _false_positive_rate(image_metrics),
                "sample_fpr": _false_positive_rate(sample_metrics),
                **{f"image_{key}": value for key, value in image_metrics.items()},
                **{f"sample_{key}": value for key, value in sample_metrics.items()},
                **{f"anomalib_image_{key}": value for key, value in adaptive_metrics.items()},
            },
        )

        figure_path = reports_dir / "figures" / f"{view}_{model_name}_score_histogram.png"
        figure_rows = predictions[(predictions["model"] == model_name) & (predictions["view"] == view)]
        _write_score_histogram(figure_rows, figure_path)

        fp = group[(group["label"] == "normal_test") & (group["deploy_pred_label"] == 1)].sort_values(
            "pred_score",
            ascending=False,
        )
        fn = group[(group["label"] == "defect") & (group["deploy_pred_label"] == 0)].sort_values("pred_score")
        false_positives.append(fp)
        false_negatives.append(fn)
        _copy_error_examples(fp, reports_dir / "examples" / f"{view}_{model_name}" / "false_positives")
        _copy_error_examples(fn, reports_dir / "examples" / f"{view}_{model_name}" / "false_negatives")

    summary = pd.DataFrame(summary_rows).sort_values(["model", "view"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(reports_dir / "summary.csv", index=False)
    predictions.to_csv(reports_dir / "predictions.csv", index=False)

    if false_positives:
        pd.concat(false_positives, ignore_index=True).to_csv(reports_dir / "false_positives.csv", index=False)
    if false_negatives:
        pd.concat(false_negatives, ignore_index=True).to_csv(reports_dir / "false_negatives.csv", index=False)

    _write_markdown_summary(summary, reports_dir / "summary.md")
    return summary


def _write_markdown_summary(summary: pd.DataFrame, output_path: Path) -> None:
    """Write a compact Markdown report."""
    lines = [
        "# ZS32 Defect Detection Summary",
        "",
        "| model | view | threshold | target_fpr | image_fpr | image_acc | image_recall | image_f1 | "
        "sample_fpr | sample_acc | sample_recall | sample_f1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| {model} | {view} | {threshold:.6f} | {target_fpr:.3f} | {image_fpr:.3f} | "
            "{image_acc:.3f} | {image_recall:.3f} | {image_f1:.3f} | {sample_fpr:.3f} | "
            "{sample_acc:.3f} | {sample_recall:.3f} | {sample_f1:.3f} |".format(
                model=row["model"],
                view=row["view"],
                threshold=float(row["deploy_threshold"]),
                target_fpr=float(row["deploy_target_fpr"]),
                image_fpr=float(row["image_fpr"]),
                image_acc=float(row["image_accuracy"]),
                image_recall=float(row["image_recall"]),
                image_f1=float(row["image_f1"]),
                sample_fpr=float(row["sample_fpr"]),
                sample_acc=float(row["sample_accuracy"]),
                sample_recall=float(row["sample_recall"]),
                sample_f1=float(row["sample_f1"]),
            ),
        )

    lines.extend(
        [
            "",
            "## Next Improvements",
            "",
            "- Review false positives first; if they cluster on fixture edges, tighten the ROI.",
            "- If false negatives are small scratches, try image_size=(512,1024) for PatchCore before changing models.",
            "- If blue-mark removal leaves artifacts, save and inspect masks around marked defect samples.",
            "- When camera placement drifts, use a large stable ROI plus automatic metal boundary alignment.",
        ],
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_experiments(args: argparse.Namespace) -> None:
    """Evaluate selected trained experiments and write reports."""
    output_root = args.output_root.resolve()
    _log("Starting evaluation stage.")
    _log(f"Views: {', '.join(_selected_views(args.views))}")
    _log(f"Models: {', '.join(_selected_models(args.models))}")
    manifest = _load_manifest(output_root)
    _log_preprocessing_config(manifest)
    _assert_accelerator_ready(args.accelerator)

    from anomalib.engine import Engine

    prediction_frames = []
    for view in _selected_views(args.views):
        data_path = _processed_view_dir(output_root, view)
        for model_name in _selected_models(args.models):
            run_name = _model_run_name(model_name, args)
            run_dir = _run_dir(output_root, view, run_name)
            checkpoint_path = _find_checkpoint(run_dir)
            visualizations_dir = run_dir / "visualizations" / "predict"
            _log(f"Predicting {run_name} on {view} from {_repo_relative(checkpoint_path)}...")

            try:
                model = _build_model(model_name, args.image_size, visualizations_dir, args)
                engine = Engine(
                    accelerator=args.accelerator,
                    devices=args.devices,
                    default_root_dir=run_dir / "predict",
                    logger=False,
                )
                predictions = engine.predict(
                    model=model,
                    data_path=data_path,
                    ckpt_path=checkpoint_path,
                    return_predictions=True,
                )
            except Exception as error:
                _raise_if_pretrained_resource_error(model_name, args, error)
                raise
            frame = _predictions_to_frame(predictions, run_name)
            if frame.empty:
                msg = f"No predictions returned for {run_name}/{view}."
                raise RuntimeError(msg)
            prediction_frames.append(frame)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions = predictions.merge(
        manifest,
        on="processed_path",
        how="left",
        validate="many_to_one",
    )
    missing_manifest = predictions["label"].isna().sum()
    if missing_manifest:
        msg = f"{missing_manifest} predictions could not be matched to manifest rows."
        raise RuntimeError(msg)
    predictions = predictions.sort_values(["model", "view", "label", "sample_id", "frame_id"])

    reports_dir = output_root / args.reports_dir_name
    summary = _build_summary_and_reports(predictions, reports_dir, args.deploy_fpr)
    _log(f"Predictions: {_repo_relative(reports_dir / 'predictions.csv')}")
    _log(f"Summary: {_repo_relative(reports_dir / 'summary.md')}")
    _log(summary.to_string(index=False))


def run_all(args: argparse.Namespace) -> None:
    """Run preprocessing, training, and evaluation."""
    _log("Running full ZS32 workflow: preprocess -> train -> evaluate.")
    preprocess_dataset(args)
    _log("Preprocessing stage complete.")
    if args.skip_missing_efficientad_assets and "efficient_ad" in _selected_models(args.models):
        missing_assets = _efficientad_missing_assets(args)
        if missing_assets:
            args.models = [model_name for model_name in _selected_models(args.models) if model_name != "efficient_ad"]
            _log("EfficientAd will be skipped for train/evaluate because required local assets are missing.")
            if not args.models:
                msg = "No models left to run after skipping EfficientAd."
                raise RuntimeError(msg)
    train_experiments(args)
    _log("Training stage complete.")
    evaluate_experiments(args)
    _log("Evaluation stage complete.")


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared command-line arguments."""
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Raw dataset root.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Workflow output root.")
    parser.add_argument("--views", nargs="+", choices=sorted(VIEW_SPECS), help="Views to process.")
    parser.add_argument("--models", nargs="+", choices=MODELS, help="Models to run.")
    parser.add_argument("--roi", type=_parse_roi, default=DEFAULT_ROI, help="Crop ROI as x1,y1,x2,y2, none, or full.")
    parser.add_argument("--image-size", type=_parse_image_size, default=DEFAULT_IMAGE_SIZE, help="Model size as H,W.")
    parser.add_argument(
        "--model-run-suffix",
        type=_parse_run_suffix,
        default="",
        help="Optional suffix for the run directory and reported model name.",
    )
    parser.add_argument(
        "--reports-dir-name",
        type=_parse_reports_dir_name,
        default=Path("reports"),
        help="Relative directory below output-root where evaluation reports are written.",
    )
    parser.add_argument(
        "--visualizer-field-size",
        type=_parse_field_size,
        help="Visualization panel size as W,H. Defaults to the model input aspect ratio.",
    )
    parser.add_argument(
        "--skip-blue-removal",
        action="store_true",
        help="Skip HSV blue-mark inpainting during preprocessing.",
    )
    parser.add_argument("--accelerator", choices=("gpu", "cpu", "auto"), default="gpu", help="Lightning accelerator.")
    parser.add_argument("--devices", type=int, default=1, help="Number of accelerator devices.")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader worker count.")
    parser.add_argument(
        "--train-sampling-ratio",
        type=_parse_sampling_ratio,
        default=1.0,
        help="Fraction of normal training images to use. Validation/test data are unchanged.",
    )
    parser.add_argument("--patchcore-batch-size", type=int, default=8, help="PatchCore training batch size.")
    parser.add_argument("--patchcore-backbone", default="wide_resnet50_2", help="PatchCore timm backbone name.")
    parser.add_argument(
        "--patchcore-layers",
        nargs="+",
        choices=("layer1", "layer2", "layer3", "layer4"),
        default=["layer2", "layer3"],
        help="PatchCore feature layers. Use fewer layers to reduce memory.",
    )
    parser.add_argument(
        "--patchcore-coreset-ratio",
        type=float,
        default=0.5,
        help="PatchCore coreset sampling ratio after collecting embeddings.",
    )
    parser.add_argument("--patchcore-num-neighbors", type=int, default=2, help="PatchCore nearest-neighbor count.")
    parser.add_argument(
        "--patchcore-precision",
        choices=("float32", "float16"),
        default="float32",
        help="PatchCore feature precision. float16 reduces memory.",
    )
    parser.add_argument("--efficientad-batch-size", type=int, default=1, help="EfficientAd training batch size.")
    parser.add_argument("--anomaly-dino-batch-size", type=int, default=1, help="AnomalyDINO training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--efficientad-epochs", type=int, default=20, help="EfficientAd max epochs.")
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
    parser.add_argument(
        "--anomaly-dino-sampling-ratio",
        type=_parse_sampling_ratio,
        default=0.1,
        help="AnomalyDINO coreset ratio.",
    )
    parser.add_argument(
        "--deploy-fpr",
        type=_parse_deploy_fpr,
        default=0.0,
        help="Allowed normal_test false-positive rate for deployment threshold, e.g. 0.05 permits about 5%%.",
    )
    parser.add_argument("--imagenet-dir", type=Path, default=DEFAULT_IMAGENETTE_DIR, help="EfficientAd ImageNette dir.")
    parser.add_argument(
        "--skip-missing-efficientad-assets",
        action="store_true",
        help="Skip EfficientAd when its teacher weights or ImageNette data are not cached locally.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split seed.")


def build_parser() -> argparse.ArgumentParser:
    """Build the workflow CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Crop optional ROI, optionally remove blue marks, and write manifest.",
    )
    _add_common_arguments(preprocess_parser)
    preprocess_parser.set_defaults(func=preprocess_dataset)

    train_parser = subparsers.add_parser("train", help="Train selected PatchCore/EfficientAd experiments.")
    _add_common_arguments(train_parser)
    train_parser.set_defaults(func=train_experiments)

    evaluate_parser = subparsers.add_parser("evaluate", help="Predict and write analysis reports.")
    _add_common_arguments(evaluate_parser)
    evaluate_parser.set_defaults(func=evaluate_experiments)

    all_parser = subparsers.add_parser("all", help="Run preprocess, train, and evaluate.")
    _add_common_arguments(all_parser)
    all_parser.set_defaults(func=run_all)
    return parser


def main() -> None:
    """Run the ZS32 workflow CLI."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
