#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Select one half-open Template/PatchCore/YOLO part ROI for each BMW view."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from bmw_inspection.lab.capture import load_capture_set  # noqa: E402
from bmw_inspection.lab.config import load_experiment_config  # noqa: E402
from bmw_inspection.lab.contracts import ViewId  # noqa: E402


def fit_image_for_display(
    image: np.ndarray,
    *,
    max_display_width: int,
    max_display_height: int,
) -> np.ndarray:
    """Fit a source image on screen without changing its aspect ratio or enlarging it."""
    if not isinstance(image, np.ndarray) or image.ndim not in {2, 3}:
        raise TypeError("ROI source must be a numpy image")
    if max_display_width <= 0 or max_display_height <= 0:
        raise ValueError("ROI display dimensions must be positive")
    source_height, source_width = image.shape[:2]
    scale = min(1.0, max_display_width / source_width, max_display_height / source_height)
    display_width = max(1, int(round(source_width * scale)))
    display_height = max(1, int(round(source_height * scale)))
    if (display_width, display_height) == (source_width, source_height):
        return image
    return cv2.resize(image, (display_width, display_height), interpolation=cv2.INTER_AREA)


def source_roi(
    display_xywh: tuple[int, int, int, int],
    display_shape: tuple[int, int],
    source_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Map display XYWH to clamped, half-open source coordinates."""
    x, y, width, height = display_xywh
    if width <= 0 or height <= 0:
        raise ValueError("ROI selection must have positive width and height")
    display_height, display_width = display_shape
    source_height, source_width = source_shape
    scale_x = source_width / display_width
    scale_y = source_height / display_height
    x1 = max(0, int(math.floor(x * scale_x)))
    y1 = max(0, int(math.floor(y * scale_y)))
    x2 = min(source_width, int(math.ceil((x + width) * scale_x)))
    y2 = min(source_height, int(math.ceil((y + height) * scale_y)))
    if not (0 <= x1 < x2 <= source_width and 0 <= y1 < y2 <= source_height):
        raise ValueError("selected ROI lies outside the source image")
    return x1, y1, x2, y2


def _select_one(
    view_id: ViewId,
    image: np.ndarray,
    *,
    max_display_width: int,
    max_display_height: int,
) -> tuple[int, int, int, int]:
    displayed = fit_image_for_display(
        image,
        max_display_width=max_display_width,
        max_display_height=max_display_height,
    )
    select_window = f"BMW Lab {view_id.value} - Select ROI"
    confirm_window = f"BMW Lab {view_id.value} - Enter to Confirm"
    try:
        selection = tuple(
            int(value)
            for value in cv2.selectROI(select_window, displayed, showCrosshair=True, fromCenter=False)
        )
        roi = source_roi(selection, displayed.shape[:2], image.shape[:2])
        x1, y1, x2, y2 = roi
        selected = image[y1:y2, x1:x2]
        preview_scale = max(1.0, min(8.0, 640.0 / max(selected.shape[:2])))
        preview = cv2.resize(
            selected,
            None,
            fx=preview_scale,
            fy=preview_scale,
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imshow(confirm_window, preview)
        if cv2.waitKey(0) & 0xFF not in {10, 13}:
            raise RuntimeError(f"ROI for {view_id.value} was not confirmed with Enter")
        return roi
    finally:
        for window in (select_window, confirm_window):
            try:
                cv2.destroyWindow(window)
            except cv2.error:
                pass


def save_part_rois(profile_path: Path, rois: dict[ViewId, tuple[int, int, int, int]]) -> None:
    """Atomically update only the six part_rois after strict profile revalidation."""
    path = Path(profile_path).expanduser().resolve()
    if set(rois) != set(ViewId):
        raise ValueError("part_rois must contain exactly the six BMW views")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["part_rois"] = {view_id.value: list(rois[view_id]) for view_id in ViewId}
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".json", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        load_experiment_config(temporary)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit saved-capture ROI selection interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-set", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/bmw/experiments/bmw_lab_v1.json",
    )
    parser.add_argument("--max-display-width", type=int, default=1280)
    parser.add_argument("--max-display-height", type=int, default=720)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Select all six ROIs from one immutable saved capture set."""
    args = build_parser().parse_args(argv)
    try:
        config = load_experiment_config(args.config)
        capture_set = load_capture_set(args.capture_set)
        rois: dict[ViewId, tuple[int, int, int, int]] = {}
        for view_id in ViewId:
            image = capture_set.views[view_id].image
            if image.shape[:2] != (config.capture.image_height, config.capture.image_width):
                raise ValueError(f"capture-set dimensions do not match profile for {view_id.value}")
            rois[view_id] = _select_one(
                view_id,
                image,
                max_display_width=args.max_display_width,
                max_display_height=args.max_display_height,
            )
        save_part_rois(args.config, rois)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"BMW six-view ROI selection failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps({view_id.value: list(rois[view_id]) for view_id in ViewId}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
