"""Atomic on-disk evidence records for the BMW eight-view laboratory Demo."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from bmw_inspection.capture.config import load_capture_profile
from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER, _atomic_publish_noreplace
from bmw_inspection.lab.eight_view_demo import EightViewDemoConfig, EightViewInspection
from bmw_inspection.lab.eight_view_demo_capture import HdrSourceImages
from bmw_inspection.lab.eight_view_demo_models import load_part_rois


_INDEX_FIELDS = ("capture_id", "demo_id", "final_status", "result_path")


def _jsonable(value: Any) -> Any:
    """Convert immutable diagnostic mappings/tuples to JSON containers."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _capture_profile_payload(config: EightViewDemoConfig) -> dict[str, Any]:
    """Snapshot configured acquisition values without claiming camera-register readback."""
    profile = load_capture_profile(config.capture_config)
    hdr = profile.hdr
    return {
        "settings_origin": "configured_not_camera_readback",
        "capture_config": str(profile.path),
        "capture_config_sha256": _sha256(profile.path),
        "profile_id": profile.profile_id,
        "camera_slots": [
            {
                "slot_id": slot.slot_id,
                "serial": slot.serial,
                "front_view": slot.front_view,
                "back_view": slot.back_view,
            }
            for slot in profile.slots
        ],
        "hdr": {
            "short_exposure_us": hdr.short_exposure_us,
            "long_exposure_us": hdr.long_exposure_us,
            "gain": hdr.gain,
            "trigger_interval_s": hdr.trigger_interval_s,
            "settle_frames": hdr.settle_frames,
            "timeout_ms": hdr.timeout_ms,
            "align": hdr.align,
            "short_dark_threshold": hdr.short_dark_threshold,
            "long_clip_threshold": hdr.long_clip_threshold,
            "blend_width": hdr.blend_width,
            "blur_size": hdr.blur_size,
            "max_retries": hdr.max_retries,
            "max_clip_pct": hdr.max_clip_pct,
        },
        "iso": "not_applicable_use_gain",
        "aperture": "not_recorded",
        "unrecorded_physical_controls": ["focus", "lamp_output"],
    }


def fused_only_sources(images: Mapping[str, np.ndarray]) -> Mapping[str, HdrSourceImages]:
    """Adapt an offline fused-image sample without claiming it has HDR source frames."""
    if tuple(images) != VIEW_ORDER:
        raise ValueError("images must use the canonical BMW eight-view order")
    return {
        view: HdrSourceImages(
            short_image=images[view],
            long_image=images[view],
            fused_image=images[view],
            fused_clip_pct=0.0,
            attempt=1,
            source_kind="fused_only",
        )
        for view in VIEW_ORDER
    }


def _validate_sources(source_images: Mapping[str, HdrSourceImages]) -> tuple[str, Mapping[str, HdrSourceImages]]:
    if tuple(source_images) != VIEW_ORDER:
        raise ValueError("source_images must use the canonical BMW eight-view order")
    if not all(isinstance(source_images[view], HdrSourceImages) for view in VIEW_ORDER):
        raise TypeError("source_images must contain HdrSourceImages values")
    source_kinds = {source_images[view].source_kind for view in VIEW_ORDER}
    if len(source_kinds) != 1:
        raise ValueError("source_images must use one consistent source_kind")
    return source_kinds.pop(), source_images


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"无法保存检测证据图片：{path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim != 3:
        raise ValueError("ROI image must be grayscale, BGR, or BGRA")
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError("ROI image must be grayscale, BGR, or BGRA")


def _roi_statistics(path: Path, roi: np.ndarray) -> dict[str, Any]:
    gray = _gray(roi)
    return {
        "mean": float(np.mean(gray)),
        "std": float(np.std(gray)),
        "p1": float(np.percentile(gray, 1)),
        "p99": float(np.percentile(gray, 99)),
        "dark_pixel_ratio": float(np.mean(gray <= 15)),
        "bright_saturation_ratio": float(np.mean(gray >= 250)),
        "sha256": _sha256(path),
    }


def _crop_roi(image: np.ndarray, roi: tuple[int, int, int, int], view: str) -> np.ndarray:
    x1, y1, x2, y2 = roi
    height, width = image.shape[:2]
    if x2 > width or y2 > height:
        raise ValueError(f"ROI for {view} lies outside {width}x{height}")
    return image[y1:y2, x1:x2].copy()


def _inspection_payload(
    config: EightViewDemoConfig,
    inspection: EightViewInspection,
    source_kind: str,
    source_images: Mapping[str, HdrSourceImages],
    roi_statistics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    evidence_by_result = {
        id(result): None if result.overlay is None else f"evidence/{result.branch.value}_{result.view_id}.png"
        for result in inspection.results
    }
    trusted_references = {
        view: {
            "reference_is_diagnostic_only": True,
            "physical_part_id": match.physical_part_id,
            "sample_id": match.sample_id,
            "similarity": match.similarity,
            "shift": {"x": match.shift_x, "y": match.shift_y},
            "index_sha256": match.index_sha256,
            "whitelist_sha256": match.whitelist_sha256,
            "source_sha256": match.source_sha256,
            "reference_full_sha256": match.reference_full_sha256,
            "reference_roi_sha256": match.reference_roi_sha256,
            "files": {
                "full": f"references/{view}/full.png",
                "roi": f"references/{view}/roi.png",
                "aligned_roi": f"references/{view}/aligned_roi.png",
                "difference": f"references/{view}/difference.png",
            },
            "saved_sha256": {},
        }
        for view, match in inspection.trusted_ok_by_view.items()
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "demo_id": config.demo_id,
        "capture_id": inspection.capture_id,
        "final_status": inspection.final_status.value,
        "elapsed_ms": inspection.elapsed_ms,
        "source_kind": source_kind,
        "capture_profile": _capture_profile_payload(config),
        "views": {
            view: {
                "source": {
                    "source_kind": source_images[view].source_kind,
                    "fused_clip_pct": source_images[view].fused_clip_pct,
                    "attempt": source_images[view].attempt,
                    "short": f"images/{view}_short.png",
                    "long": f"images/{view}_long.png",
                    "hdr": f"images/{view}_hdr.png",
                },
                "roi": f"rois/{view}.png",
                "roi_statistics": dict(roi_statistics[view]),
            }
            for view in VIEW_ORDER
        },
        "results": [
            {
                "branch": result.branch.value,
                "view_id": result.view_id,
                "status": result.status.value,
                "score": result.score,
                "threshold": result.threshold,
                "elapsed_ms": result.elapsed_ms,
                "reason": result.reason,
                "raw_pred_label": result.raw_pred_label,
                "overlay": evidence_by_result[id(result)],
                "details": _jsonable(result.details),
            }
            for result in inspection.results
        ],
    }
    if trusted_references:
        payload["reference_is_diagnostic_only"] = True
        payload["trusted_ok_references"] = trusted_references
    if inspection.diagnostic_metadata:
        payload["diagnostic_metadata"] = _jsonable(inspection.diagnostic_metadata)
    return payload


def append_inspection_index(result_root: Path, inspection: EightViewInspection, demo_id: str) -> Path:
    """Append one already-published capture to the operator-facing result index."""
    root = Path(result_root).expanduser().resolve()
    index_path = root / "inspection_index.csv"
    write_header = not index_path.exists()
    with index_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_INDEX_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "capture_id": inspection.capture_id,
                "demo_id": demo_id,
                "final_status": inspection.final_status.value,
                "result_path": inspection.capture_id,
            }
        )
        stream.flush()
        os.fsync(stream.fileno())
    return index_path


def persist_inspection(
    config: EightViewDemoConfig,
    inspection: EightViewInspection,
    source_images: Mapping[str, HdrSourceImages],
) -> Path:
    """Publish raw HDR sources, ROIs, overlays, metrics, and an index row without overwrite."""
    source_kind, validated_sources = _validate_sources(source_images)
    root = Path(config.result_root).expanduser().resolve()
    destination = root / inspection.capture_id
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{inspection.capture_id}.", dir=root))
    try:
        roi_by_view = load_part_rois(config.roi_config)
        statistics: dict[str, Mapping[str, Any]] = {}
        for view in VIEW_ORDER:
            source = validated_sources[view]
            _write_image(staging / "images" / f"{view}_short.png", source.short_image)
            _write_image(staging / "images" / f"{view}_long.png", source.long_image)
            _write_image(staging / "images" / f"{view}_hdr.png", source.fused_image)
            roi = _crop_roi(inspection.images[view], roi_by_view[view], view)
            roi_path = staging / "rois" / f"{view}.png"
            _write_image(roi_path, roi)
            statistics[view] = _roi_statistics(roi_path, roi)
        for result in inspection.results:
            if result.overlay is not None:
                _write_image(staging / "evidence" / f"{result.branch.value}_{result.view_id}.png", result.overlay)
        payload = _inspection_payload(config, inspection, source_kind, validated_sources, statistics)
        for view, match in inspection.trusted_ok_by_view.items():
            reference_images = {
                "full": match.reference_full_image,
                "roi": match.reference_roi,
                "aligned_roi": match.aligned_reference_roi,
                "difference": match.difference_overlay,
            }
            for name, image in reference_images.items():
                relative = Path(payload["trusted_ok_references"][view]["files"][name])
                target = staging / relative
                _write_image(target, image)
                payload["trusted_ok_references"][view]["saved_sha256"][name] = _sha256(target)
        (staging / "inspection.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _atomic_publish_noreplace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    append_inspection_index(root, inspection, config.demo_id)
    return destination.resolve()


__all__ = ["append_inspection_index", "fused_only_sources", "persist_inspection"]
