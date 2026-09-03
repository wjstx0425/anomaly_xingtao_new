"""Runtime-only trusted-OK comparison for the BMW laboratory Demo."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
import threading
from typing import Literal

import cv2
import numpy as np

from bmw_inspection.views import VIEW_ORDER


@dataclass(frozen=True, slots=True)
class TrustedOkMatch:
    """Display-only nearest OK reference for one actionable view."""

    view_id: str
    comparison_mode: Literal["roi", "full"]
    physical_part_id: str
    sample_id: str
    similarity: float
    shift_x: int
    shift_y: int
    current_full_image: np.ndarray
    reference_full_image: np.ndarray
    current_roi: np.ndarray
    reference_roi: np.ndarray
    aligned_reference_roi: np.ndarray
    difference_overlay: np.ndarray

    def __post_init__(self) -> None:
        if self.view_id not in VIEW_ORDER:
            raise ValueError(f"unknown BMW view: {self.view_id}")
        if self.comparison_mode not in {"roi", "full"}:
            raise ValueError("comparison_mode must be roi or full")
        if self.comparison_mode == "full" and self.view_id != "front_left":
            raise ValueError("full comparison_mode is reserved for front_left bright streak")
        if not self.physical_part_id.strip() or not self.sample_id.strip():
            raise ValueError("trusted reference identity must not be empty")
        if not math.isfinite(float(self.similarity)) or not -1.0 <= float(self.similarity) <= 1.0:
            raise ValueError("similarity must be finite and within [-1, 1]")
        for name in ("shift_x", "shift_y"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        for name in (
            "current_full_image",
            "reference_full_image",
            "current_roi",
            "reference_roi",
            "aligned_reference_roi",
            "difference_overlay",
        ):
            image = getattr(self, name)
            if (
                not isinstance(image, np.ndarray)
                or image.dtype != np.uint8
                or image.size == 0
                or image.ndim not in {2, 3}
            ):
                raise ValueError(f"{name} must be a non-empty uint8 image")
            owned = image.copy()
            owned.flags.writeable = False
            object.__setattr__(self, name, owned)


@dataclass(frozen=True, slots=True)
class _Reference:
    physical_part_id: str
    sample_id: str
    view_id: str
    full_path: Path
    roi_path: Path


@dataclass(frozen=True, slots=True)
class _PreparedReference:
    row: _Reference
    prepared: np.ndarray


def _text(raw: Mapping[str, object], field: str, context: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}缺少{field}")
    return value.strip()


def _image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8 or image.size == 0:
        raise ValueError(f"可信OK图片无法读取：{path}")
    return image


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError("可信OK匹配只支持灰度、BGR或BGRA图像")


def _prepare(image: np.ndarray) -> np.ndarray:
    gray = _gray(image)
    target_width = target_height = 512
    scale = min(target_width / gray.shape[1], target_height / gray.shape[0])
    width = max(1, min(target_width, int(round(gray.shape[1] * scale))))
    height = max(1, min(target_height, int(round(gray.shape[0] * scale))))
    resized = cv2.resize(
        gray,
        (width, height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    left = (target_width - width) // 2
    top = (target_height - height) // 2
    fitted = cv2.copyMakeBorder(
        resized,
        top,
        target_height - height - top,
        left,
        target_width - width - left,
        cv2.BORDER_REFLECT_101,
    )
    return cv2.GaussianBlur(fitted, (3, 3), 0)


def _difference_overlay(current: np.ndarray, reference: np.ndarray) -> np.ndarray:
    difference = cv2.absdiff(current, reference)
    maximum = int(difference.max())
    contrast = (
        np.zeros_like(difference)
        if maximum == 0
        else cv2.convertScaleAbs(difference, alpha=255.0 / maximum)
    )
    return cv2.applyColorMap(contrast, cv2.COLORMAP_INFERNO)


class TrustedOkMatcher:
    """Load the reference index directly and retain the existing correlation matcher."""

    def __init__(self, index_path: Path, *, max_shift: int = 12) -> None:
        self._index_path = Path(index_path).expanduser().resolve()
        if isinstance(max_shift, bool) or not isinstance(max_shift, int) or not 0 <= max_shift <= 64:
            raise ValueError("max_shift must be an integer from 0 through 64")
        try:
            payload = json.loads(self._index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"无法读取可信OK索引：{self._index_path}: {error}") from error
        raw_references = payload.get("references") if isinstance(payload, dict) else None
        if not isinstance(raw_references, list) or not raw_references:
            raise ValueError("可信OK索引没有references")
        root = self._index_path.parent
        by_view: dict[str, list[_Reference]] = {view: [] for view in VIEW_ORDER}
        for number, raw in enumerate(raw_references, start=1):
            if not isinstance(raw, dict):
                raise ValueError(f"可信OK记录{number}不是JSON对象")
            view = _text(raw, "view_id", f"可信OK记录{number}")
            if view not in VIEW_ORDER:
                raise ValueError(f"可信OK记录{number}视角无效")
            full = Path(_text(raw, "full_image_path", f"可信OK记录{number}"))
            roi = Path(_text(raw, "roi_image_path", f"可信OK记录{number}"))
            full_path = (full if full.is_absolute() else root / full).resolve()
            roi_path = (roi if roi.is_absolute() else root / roi).resolve()
            if not full_path.is_file() or not roi_path.is_file():
                raise ValueError(f"可信OK记录{number}图片不存在")
            by_view[view].append(
                _Reference(
                    physical_part_id=_text(raw, "physical_part_id", f"可信OK记录{number}"),
                    sample_id=_text(raw, "sample_id", f"可信OK记录{number}"),
                    view_id=view,
                    full_path=full_path,
                    roi_path=roi_path,
                )
            )
        missing = [view for view in VIEW_ORDER if not by_view[view]]
        if missing:
            raise ValueError(f"可信OK索引缺少视角：{','.join(missing)}")
        self._max_shift = max_shift
        self._by_view = {view: tuple(rows) for view, rows in by_view.items()}
        self._prepared: dict[tuple[str, Literal["roi", "full"]], tuple[_PreparedReference, ...]] = {}
        self._lock = threading.RLock()

    def _prepared_references(
        self,
        view_id: str,
        comparison_mode: Literal["roi", "full"],
    ) -> tuple[_PreparedReference, ...]:
        key = (view_id, comparison_mode)
        cached = self._prepared.get(key)
        if cached is not None:
            return cached
        prepared = tuple(
            _PreparedReference(
                row=row,
                prepared=_prepare(_image(row.full_path if comparison_mode == "full" else row.roi_path)),
            )
            for row in self._by_view[view_id]
        )
        self._prepared[key] = prepared
        return prepared

    def preload(self) -> None:
        """Warm all ROI banks and the front-left full-image bank once."""
        with self._lock:
            for view in VIEW_ORDER:
                self._prepared_references(view, "roi")
            self._prepared_references("front_left", "full")

    def match(
        self,
        view_id: str,
        current_full_image: np.ndarray,
        current_roi: np.ndarray,
        *,
        comparison_mode: Literal["roi", "full"],
    ) -> TrustedOkMatch:
        if view_id not in VIEW_ORDER:
            raise ValueError(f"unknown BMW view: {view_id}")
        if comparison_mode not in {"roi", "full"}:
            raise ValueError("comparison_mode must be roi or full")
        if comparison_mode == "full" and view_id != "front_left":
            raise ValueError("full comparison_mode is reserved for front_left bright streak")
        current_region = current_full_image if comparison_mode == "full" else current_roi
        prepared_current = _prepare(current_region)
        with self._lock:
            key = (view_id, comparison_mode)
            if key not in self._prepared:
                raise RuntimeError("TrustedOkMatcher.preload() must complete before match()")
            best: tuple[float, int, int, _PreparedReference, np.ndarray] | None = None
            for candidate in self._prepared[key]:
                padded = cv2.copyMakeBorder(
                    candidate.prepared,
                    self._max_shift,
                    self._max_shift,
                    self._max_shift,
                    self._max_shift,
                    cv2.BORDER_REFLECT_101,
                )
                response = cv2.matchTemplate(padded, prepared_current, cv2.TM_CCOEFF_NORMED)
                _minimum, similarity, _minimum_location, location = cv2.minMaxLoc(response)
                if not math.isfinite(similarity):
                    similarity = -1.0
                x, y = location
                aligned = padded[y : y + 512, x : x + 512].copy()
                if best is None or similarity > best[0]:
                    best = (float(similarity), x - self._max_shift, y - self._max_shift, candidate, aligned)
            if best is None:
                raise ValueError(f"可信OK索引没有{view_id}参考")
            similarity, shift_x, shift_y, selected, aligned = best
            row = selected.row
            reference_full = _image(row.full_path)
            reference_region = _image(row.roi_path)
        return TrustedOkMatch(
            view_id=view_id,
            comparison_mode=comparison_mode,
            physical_part_id=row.physical_part_id,
            sample_id=row.sample_id,
            similarity=similarity,
            shift_x=shift_x,
            shift_y=shift_y,
            current_full_image=current_full_image,
            reference_full_image=reference_full,
            current_roi=current_region,
            reference_roi=reference_region,
            aligned_reference_roi=aligned,
            difference_overlay=_difference_overlay(prepared_current, aligned),
        )


__all__ = ["TrustedOkMatch", "TrustedOkMatcher"]
