"""Utilities for fusing short- and long-exposure capture images."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_bgr(path: str | Path) -> np.ndarray:
    """Read an image as BGR uint8."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def align_mtb(images: list[np.ndarray]) -> list[np.ndarray]:
    """Align images with OpenCV's median-threshold bitmap alignment."""
    if len(images) < 2:
        return images

    aligned = [image.copy() for image in images]
    cv2.createAlignMTB().process(images, aligned)
    return aligned


def _validate_images(images: list[np.ndarray]) -> None:
    if len(images) < 2:
        raise ValueError("At least two exposure images are required")

    shape = images[0].shape
    for index, image in enumerate(images):
        if image.shape != shape:
            raise ValueError(
                f"Image {index} has shape {image.shape}, expected {shape}"
            )
        if image.dtype != np.uint8:
            raise ValueError(f"Image {index} must be uint8, got {image.dtype}")


def mertens_fusion(
    images: list[np.ndarray],
    contrast_weight: float = 1.0,
    saturation_weight: float = 0.2,
    exposure_weight: float = 1.0,
) -> np.ndarray:
    """Fuse exposures with OpenCV's Mertens exposure fusion."""
    _validate_images(images)
    merge = cv2.createMergeMertens(
        contrast_weight,
        saturation_weight,
        exposure_weight,
    )
    fused = merge.process(images)
    return np.clip(fused * 255.0, 0, 255).astype(np.uint8)


def selective_long_exposure_fusion(
    short_exposure: np.ndarray,
    long_exposure: np.ndarray,
    short_dark_threshold: float = 70.0,
    long_clip_threshold: float = 245.0,
    blend_width: float = 18.0,
    blur_size: int = 31,
) -> np.ndarray:
    """Blend long exposure only where the short exposure is underexposed.

    The short exposure remains the base image. This protects surface defects and
    text from being washed out by the long exposure while still revealing dark
    corners and edge details.
    """
    _validate_images([short_exposure, long_exposure])

    short_gray = cv2.cvtColor(short_exposure, cv2.COLOR_BGR2GRAY).astype(np.float32)
    long_gray = cv2.cvtColor(long_exposure, cv2.COLOR_BGR2GRAY).astype(np.float32)

    width = max(float(blend_width), 1.0)

    # High when short exposure is too dark.
    dark_weight = 1.0 / (1.0 + np.exp((short_gray - short_dark_threshold) / width))

    # High when long exposure is not clipped.
    valid_long_weight = 1.0 / (
        1.0 + np.exp((long_gray - long_clip_threshold) / width)
    )

    weight = dark_weight * valid_long_weight
    if blur_size > 1:
        if blur_size % 2 == 0:
            blur_size += 1
        weight = cv2.GaussianBlur(weight, (blur_size, blur_size), 0)

    weight_3c = weight[..., None]
    fused = (
        short_exposure.astype(np.float32) * (1.0 - weight_3c)
        + long_exposure.astype(np.float32) * weight_3c
    )
    return np.clip(fused, 0, 255).astype(np.uint8)


def fuse_exposures(
    images: list[np.ndarray],
    method: str = "selective",
    align: bool = True,
    short_dark_threshold: float = 70.0,
    long_clip_threshold: float = 245.0,
    blend_width: float = 18.0,
    blur_size: int = 31,
) -> np.ndarray:
    """Fuse exposure images into a display-ready uint8 BGR image."""
    _validate_images(images)
    working_images = align_mtb(images) if align else images

    if method == "mertens":
        return mertens_fusion(working_images)

    if method == "selective":
        if len(working_images) != 2:
            raise ValueError("selective fusion expects exactly two images")
        return selective_long_exposure_fusion(
            working_images[0],
            working_images[1],
            short_dark_threshold=short_dark_threshold,
            long_clip_threshold=long_clip_threshold,
            blend_width=blend_width,
            blur_size=blur_size,
        )

    raise ValueError(f"Unsupported fusion method: {method}")
