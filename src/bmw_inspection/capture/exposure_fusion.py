"""HDR fusion used by the BMW four-camera runtime."""

from __future__ import annotations

import cv2
import numpy as np


def _validate(images: list[np.ndarray]) -> None:
    if len(images) != 2:
        raise ValueError("BMW selective HDR fusion requires two images")
    if any(image.dtype != np.uint8 or image.shape != images[0].shape for image in images):
        raise ValueError("HDR images must have the same shape and uint8 dtype")


def fuse_exposures(
    images: list[np.ndarray],
    *,
    align: bool,
    short_dark_threshold: float,
    long_clip_threshold: float,
    blend_width: float,
    blur_size: int,
) -> np.ndarray:
    """Blend the long exposure only where the short exposure is dark."""
    _validate(images)
    short_image, long_image = (image.copy() for image in images)
    if align:
        aligned = [short_image, long_image]
        cv2.createAlignMTB().process(aligned, aligned)
        short_image, long_image = aligned
    short_gray = cv2.cvtColor(short_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    long_gray = cv2.cvtColor(long_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    width = max(float(blend_width), 1.0)
    dark_weight = 1.0 / (1.0 + np.exp((short_gray - short_dark_threshold) / width))
    usable_long = 1.0 / (1.0 + np.exp((long_gray - long_clip_threshold) / width))
    weight = dark_weight * usable_long
    if blur_size > 1:
        kernel = blur_size if blur_size % 2 else blur_size + 1
        weight = cv2.GaussianBlur(weight, (kernel, kernel), 0)
    fused = short_image.astype(np.float32) * (1.0 - weight[..., None])
    fused += long_image.astype(np.float32) * weight[..., None]
    return np.clip(fused, 0, 255).astype(np.uint8)
