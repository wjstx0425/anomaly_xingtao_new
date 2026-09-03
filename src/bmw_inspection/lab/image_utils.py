"""Small image helpers shared by the BMW eight-view UI."""

from __future__ import annotations

import cv2
import numpy as np


def fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Fit a grayscale/BGR/BGRA image into a neutral BGR panel."""
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        bgr = image.copy()
    elif image.ndim == 3 and image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        raise ValueError(f"不支持的图像尺寸: {image.shape}")
    scale = min(width / bgr.shape[1], height / bgr.shape[0])
    resized = cv2.resize(
        bgr,
        (max(1, round(bgr.shape[1] * scale)), max(1, round(bgr.shape[0] * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    panel = np.full((height, width, 3), 238, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return panel


def rgb_to_bgr(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Convert an RGB tuple to OpenCV's BGR order."""
    return color[2], color[1], color[0]
