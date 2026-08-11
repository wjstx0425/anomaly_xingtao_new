"""Interactive half-open ROI selection for the BMW bright-streak Demo."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import cv2

from .contracts import config_from_payload, load_config, read_json_object

_SELECT_WINDOW = "BMW Bright Streak - Select ROI"
_CONFIRM_WINDOW = "BMW Bright Streak - Enter to Confirm"
_DEFAULT_MAX_DISPLAY_WIDTH = 1280
_DEFAULT_MAX_DISPLAY_HEIGHT = 720


def save_roi(path: Path, roi_xyxy: tuple[int, int, int, int]) -> None:
    """Atomically replace only ``roi_xyxy`` in an otherwise unchanged JSON object."""
    config_path = Path(path).expanduser().resolve()
    payload = read_json_object(config_path)
    updated = dict(payload)
    updated["roi_xyxy"] = list(roi_xyxy)
    config_from_payload(config_path, updated)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{config_path.name}.",
        suffix=".tmp",
        dir=config_path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(updated, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(config_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _selection_image(
    image: cv2.typing.MatLike,
    max_display_width: int,
    max_display_height: int,
) -> cv2.typing.MatLike:
    """Fit an image inside the selection viewport without enlarging it."""
    if max_display_width <= 0 or max_display_height <= 0:
        raise ValueError("ROI display width and height must be positive")
    source_height, source_width = image.shape[:2]
    scale = min(1.0, max_display_width / source_width, max_display_height / source_height)
    display_width = max(1, int(round(source_width * scale)))
    display_height = max(1, int(round(source_height * scale)))
    if (display_width, display_height) == (source_width, source_height):
        return image
    return cv2.resize(image, (display_width, display_height), interpolation=cv2.INTER_AREA)


def _source_roi(
    display_xywh: tuple[int, int, int, int],
    display_shape: tuple[int, int],
    source_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Map one display-space XYWH selection to half-open source coordinates."""
    x, y, width, height = display_xywh
    display_height, display_width = display_shape
    source_height, source_width = source_shape
    scale_x = source_width / display_width
    scale_y = source_height / display_height
    x1 = max(0, int(math.floor(x * scale_x)))
    y1 = max(0, int(math.floor(y * scale_y)))
    x2 = min(source_width, int(math.ceil((x + width) * scale_x)))
    y2 = min(source_height, int(math.ceil((y + height) * scale_y)))
    return x1, y1, x2, y2


def select_roi(
    image_path: Path,
    config_path: Path,
    *,
    max_display_width: int = _DEFAULT_MAX_DISPLAY_WIDTH,
    max_display_height: int = _DEFAULT_MAX_DISPLAY_HEIGHT,
) -> tuple[int, int, int, int]:
    """Select, confirm, and persist one half-open ROI from the configured image."""
    config = load_config(config_path)
    source_path = Path(image_path).expanduser().resolve()
    image = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"could not decode ROI source image: {source_path}")
    if image.shape[:2] != (config.image_height, config.image_width):
        raise ValueError(
            f"ROI source image must be {config.image_width}x{config.image_height}, "
            f"found {image.shape[1]}x{image.shape[0]}",
        )

    selection_image = _selection_image(image, max_display_width, max_display_height)
    try:
        x, y, width, height = (
            int(value)
            for value in cv2.selectROI(
                _SELECT_WINDOW,
                selection_image,
                showCrosshair=True,
                fromCenter=False,
            )
        )
        if width <= 0 or height <= 0:
            raise ValueError("ROI selection must have positive width and height")
        roi = _source_roi(
            (x, y, width, height),
            selection_image.shape[:2],
            image.shape[:2],
        )
        x1, y1, x2, y2 = roi
        if not (0 <= x1 < x2 <= config.image_width and 0 <= y1 < y2 <= config.image_height):
            raise ValueError("selected roi_xyxy lies outside the configured image")
        selected = image[y1:y2, x1:x2]
        scale = max(1.0, min(8.0, 640.0 / max(selected.shape[:2])))
        preview = cv2.resize(selected, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(_CONFIRM_WINDOW, preview)
        if cv2.waitKey(0) & 0xFF not in {10, 13}:
            raise RuntimeError("ROI was not confirmed with Enter")
        save_roi(config_path, roi)
        return roi
    finally:
        for window in (_SELECT_WINDOW, _CONFIRM_WINDOW):
            try:
                cv2.destroyWindow(window)
            except cv2.error:
                pass
