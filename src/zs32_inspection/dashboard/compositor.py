# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Side-effect-free evidence composition for ZS32 dashboard views."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

import cv2
import numpy as np

from .contracts import BranchEvidence, BranchState, EvidenceLayer, ViewResult

_MASK_ALPHA = 0.45
_BOX_ALPHA = 0.25
_RED = (0, 0, 255)


@dataclass(frozen=True, slots=True)
class FittedImage:
    """An aspect-preserving resized image and its placement transform."""

    image: np.ndarray
    scale: float
    offset: tuple[int, int]

    @property
    def offset_x(self) -> int:
        """Horizontal placement offset in target pixels."""
        return self.offset[0]

    @property
    def offset_y(self) -> int:
        """Vertical placement offset in target pixels."""
        return self.offset[1]


@dataclass(frozen=True, slots=True)
class ComposedView:
    """One source view with only the requested genuine evidence rendered."""

    image: np.ndarray
    notice: str = ""


def _as_bgr(image: np.ndarray) -> np.ndarray:
    """Return a detached three-channel BGR image."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty NumPy array")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3:
        raise ValueError("image must have two or three dimensions")
    if image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 3:
        return image.copy()
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    raise ValueError("image must have one, three, or four channels")


def fit_letterbox(
    image: np.ndarray,
    width: int | tuple[int, int],
    height: int | None = None,
) -> FittedImage:
    """Resize without stretching and return scale plus centered offset.

    ``width`` and ``height`` may be passed separately. For consistency with
    ``source_shape``, a single tuple is interpreted as ``(height, width)``.
    """
    if isinstance(width, tuple):
        if height is not None or len(width) != 2:
            raise ValueError("target shape must be (height, width)")
        target_height, target_width = width
    else:
        if height is None:
            raise ValueError("height is required when width is an integer")
        target_width, target_height = width, height
    if (
        isinstance(target_width, bool)
        or isinstance(target_height, bool)
        or not isinstance(target_width, int)
        or not isinstance(target_height, int)
        or target_width <= 0
        or target_height <= 0
    ):
        raise ValueError("target width and height must be positive integers")

    source = _as_bgr(image)
    source_height, source_width = source.shape[:2]
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(source, (resized_width, resized_height), interpolation=interpolation)
    offset = ((target_width - resized_width) // 2, (target_height - resized_height) // 2)
    return FittedImage(image=resized, scale=scale, offset=offset)


def place_crop_mask(
    mask: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    source_shape: tuple[int, int],
) -> np.ndarray:
    """Place a crop mask into a full ``(height, width)`` boolean mask."""
    if len(source_shape) != 2 or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in source_shape
    ):
        raise ValueError("source_shape must be positive (height, width)")
    if not isinstance(mask, np.ndarray) or mask.size == 0 or mask.ndim not in (2, 3):
        raise ValueError("mask must be a non-empty image")
    if mask.ndim == 3:
        if mask.shape[2] != 1:
            raise ValueError("mask must have one channel")
        mask = mask[..., 0]
    x1, y1, x2, y2 = roi_xyxy
    source_height, source_width = source_shape
    if not (0 <= x1 < x2 <= source_width and 0 <= y1 < y2 <= source_height):
        raise ValueError("ROI is outside source image")
    resized = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST) > 0
    output = np.zeros(source_shape, dtype=bool)
    output[y1:y2, x1:x2] = resized
    return output


def overlay_red_mask(image: np.ndarray, mask: np.ndarray, alpha: float = _MASK_ALPHA) -> np.ndarray:
    """Blend a red overlay onto true mask pixels without changing the input."""
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0, 1]")
    output = _as_bgr(image)
    if not isinstance(mask, np.ndarray) or mask.shape != output.shape[:2]:
        raise ValueError("mask shape must match image height and width")
    selected = mask.astype(bool, copy=False)
    red = np.zeros_like(output)
    red[..., 2] = 255
    blended = cv2.addWeighted(output, 1.0 - alpha, red, alpha, 0)
    output[selected] = blended[selected]
    return output


def _valid_roi(roi_xyxy: tuple[int, int, int, int] | None, image_shape: tuple[int, int]) -> bool:
    if roi_xyxy is None or len(roi_xyxy) != 4:
        return False
    if any(isinstance(value, bool) or not isinstance(value, int) for value in roi_xyxy):
        return False
    x1, y1, x2, y2 = roi_xyxy
    height, width = image_shape
    return 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height


def _clipped_detection_box(
    detection: Mapping[str, object],
    roi_xyxy: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    coordinates = detection.get("xyxy")
    if (
        not isinstance(coordinates, Sequence)
        or isinstance(coordinates, (str, bytes))
        or len(coordinates) != 4
        or any(isinstance(value, bool) or not isinstance(value, Real) for value in coordinates)
    ):
        return None
    values = tuple(float(value) for value in coordinates)
    if not all(math.isfinite(value) for value in values):
        return None
    x1, y1, x2, y2 = values
    if x1 >= x2 or y1 >= y2:
        return None

    roi_x1, roi_y1, roi_x2, roi_y2 = roi_xyxy
    crop_width, crop_height = roi_x2 - roi_x1, roi_y2 - roi_y1
    clipped_x1 = min(max(x1, 0.0), float(crop_width))
    clipped_y1 = min(max(y1, 0.0), float(crop_height))
    clipped_x2 = min(max(x2, 0.0), float(crop_width))
    clipped_y2 = min(max(y2, 0.0), float(crop_height))
    if clipped_x1 >= clipped_x2 or clipped_y1 >= clipped_y2:
        return None

    source_x1 = max(roi_x1, min(roi_x2 - 1, math.floor(roi_x1 + clipped_x1)))
    source_y1 = max(roi_y1, min(roi_y2 - 1, math.floor(roi_y1 + clipped_y1)))
    source_x2 = max(source_x1 + 1, min(roi_x2, math.ceil(roi_x1 + clipped_x2)))
    source_y2 = max(source_y1 + 1, min(roi_y2, math.ceil(roi_y1 + clipped_y2)))
    return source_x1, source_y1, source_x2, source_y2


def _detection_label(detection: Mapping[str, object]) -> str:
    class_name = detection.get("class_name")
    confidence = detection.get("confidence")
    if not isinstance(class_name, str) or not class_name.strip():
        return ""
    if isinstance(confidence, bool) or not isinstance(confidence, Real):
        return ""
    value = float(confidence)
    if not math.isfinite(value) or not 0 <= value <= 1:
        return ""
    return f"{class_name.strip()} {value:.2f}"


def draw_yolo_detections(
    image: np.ndarray,
    detections: Sequence[Mapping[str, object]],
    roi_xyxy: tuple[int, int, int, int] | None,
) -> np.ndarray:
    """Draw genuine crop-coordinate YOLO boxes mapped into the source image."""
    output = _as_bgr(image)
    if not detections:
        return output
    if not _valid_roi(roi_xyxy, output.shape[:2]):
        return output
    assert roi_xyxy is not None
    for detection in detections:
        if not isinstance(detection, Mapping):
            continue
        box = _clipped_detection_box(detection, roi_xyxy)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        region = output[y1:y2, x1:x2]
        red = np.zeros_like(region)
        red[..., 2] = 255
        output[y1:y2, x1:x2] = cv2.addWeighted(region, 1.0 - _BOX_ALPHA, red, _BOX_ALPHA, 0)
        cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), _RED, 1, lineType=cv2.LINE_8)
        label = _detection_label(detection)
        if label:
            label_region = output[y1:y2, x1:x2]
            label_x = 1 if label_region.shape[1] > 1 else 0
            label_y = max(1, min(label_region.shape[0] - 1, 12))
            cv2.putText(
                label_region,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                _RED,
                1,
                lineType=cv2.LINE_AA,
            )
    return output


def _branch_notice(branch: BranchEvidence, label: str) -> str:
    state_text = {
        BranchState.SKIPPED: "未执行",
        BranchState.UNSUPPORTED: "不支持",
        BranchState.ERROR: "证据错误",
    }.get(branch.state)
    if state_text is None:
        return ""
    detail = f"：{branch.reason}" if branch.reason else ""
    return f"{label}: {state_text}{detail}"


def _load_source(view: ViewResult) -> np.ndarray:
    image = cv2.imread(str(view.source_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"source image could not be decoded: {view.source_path}")
    source = _as_bgr(image)
    if source.shape[:2] != view.source_shape:
        raise ValueError("decoded source shape does not match ViewResult.source_shape")
    return source


def _apply_patchcore(image: np.ndarray, branch: BranchEvidence) -> tuple[np.ndarray, str]:
    unavailable = _branch_notice(branch, "PatchCore")
    if unavailable:
        return image.copy(), unavailable
    if branch.mask_path is None or branch.roi_xyxy is None:
        return image.copy(), "PatchCore: 证据错误：缺少 mask 或 ROI"
    mask = cv2.imread(str(branch.mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return image.copy(), "PatchCore: 证据错误：mask 无法解码"
    try:
        full_mask = place_crop_mask(mask, branch.roi_xyxy, image.shape[:2])
        rendered = overlay_red_mask(image, full_mask)
    except (cv2.error, TypeError, ValueError) as error:
        return image.copy(), f"PatchCore: 证据错误：{error}"
    notice = "DIAGNOSTIC MASK" if branch.mask_source and branch.mask_source != "pred_mask" else ""
    return rendered, notice


def _apply_yolo(image: np.ndarray, branch: BranchEvidence) -> tuple[np.ndarray, str]:
    unavailable = _branch_notice(branch, "YOLO")
    if unavailable:
        return image.copy(), unavailable
    if branch.detections and not _valid_roi(branch.roi_xyxy, image.shape[:2]):
        return image.copy(), "YOLO: 证据错误：缺少或非法 ROI"
    return draw_yolo_detections(image, branch.detections, branch.roi_xyxy), ""


def _required_branch(view: ViewResult, name: str) -> BranchEvidence:
    branch = view.branches.get(name)
    if branch is None:
        return BranchEvidence(name, BranchState.ERROR, "ERROR", None, "branch missing")
    return branch


def compose_view(view: ViewResult, layer: EvidenceLayer) -> ComposedView:
    """Compose one evidence layer while preserving the genuine source pixels."""
    source = _load_source(view)
    if layer is EvidenceLayer.ORIGINAL:
        return ComposedView(source)
    if not view.model_supported:
        return ComposedView(source, "暂未接入模型")

    if layer is EvidenceLayer.PATCHCORE:
        image, notice = _apply_patchcore(source, _required_branch(view, "patchcore"))
        return ComposedView(image, notice)
    if layer is EvidenceLayer.YOLO:
        image, notice = _apply_yolo(source, _required_branch(view, "yolo"))
        return ComposedView(image, notice)
    if layer is EvidenceLayer.TEMPLATE:
        branch = _required_branch(view, "template")
        notice = _branch_notice(branch, "Template") or branch.status
        return ComposedView(source, notice)
    if layer is EvidenceLayer.FUSION:
        notices: list[str] = []
        image, patchcore_notice = _apply_patchcore(source, _required_branch(view, "patchcore"))
        if patchcore_notice:
            notices.append(patchcore_notice)
        image, yolo_notice = _apply_yolo(image, _required_branch(view, "yolo"))
        if yolo_notice:
            notices.append(yolo_notice)
        fusion_notice = _branch_notice(_required_branch(view, "fusion"), "Fusion")
        if fusion_notice:
            notices.append(fusion_notice)
        return ComposedView(image, "; ".join(notices))
    raise ValueError(f"unsupported evidence layer: {layer!r}")
