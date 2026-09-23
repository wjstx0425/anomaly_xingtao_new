# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Read back-view stamp text without applying product or business rules."""

from __future__ import annotations

import math
import string
import time
import unicodedata
from dataclasses import asdict
from importlib.metadata import version
from typing import Protocol

import cv2
import numpy as np

from bmw_inspection.checks.stamp_config import StampReaderConfig
from bmw_inspection.checks.stamp_result import StampReadResult


class OCRBackend(Protocol):
    """Minimal callable contract implemented by RapidOCR."""

    def __call__(self, image: np.ndarray, **kwargs: bool) -> tuple[list | None, list | None]:
        """Return OCR lines and backend timing values."""
        ...


def _normalize_code(raw_code: str) -> dict:
    retained, removed, unsupported = [], [], []
    for position, character in enumerate(raw_code):
        if character.isspace() or character in string.punctuation or unicodedata.category(character).startswith("P"):
            if character in string.whitespace:
                reason = "ascii_whitespace"
            elif character in string.punctuation:
                reason = "ascii_punctuation"
            else:
                reason = "unicode_whitespace" if character.isspace() else "unicode_punctuation"
            removed.append({
                "position": position, "character": character,
                "reason": reason,
            })
        else:
            retained.append(character)
            if character not in string.ascii_letters + string.digits:
                unsupported.append({"position": position, "character": character})
    return {
        "raw_code": raw_code, "normalized_code": "".join(retained),
        "removed_characters": removed, "unsupported_characters": unsupported,
    }


class StampReader:
    """Reuse one CPU backend for one configuration within a single worker.

    Create a separate instance per concurrent worker. Readable/review describes
    recognition confidence only; no product or business verdict is computed.
    """

    def __init__(self, config: StampReaderConfig, backend: OCRBackend | None = None) -> None:
        self.config = config
        self._backend = backend
        self._backend_metadata = {"name": "injected"} if backend is not None else {}

    def initialize(self) -> None:
        """Load and retain the OCR models without inference or warmup.

        Repeated calls reuse the backend; an injected backend is kept as-is.
        """
        self._engine()

    def _engine(self) -> OCRBackend:
        if self._backend is None:
            import onnxruntime as ort
            from rapidocr_onnxruntime import RapidOCR

            ort.disable_telemetry_events()
            self._backend = RapidOCR(
                intra_op_num_threads=self.config.intra_op_num_threads,
                inter_op_num_threads=self.config.inter_op_num_threads, text_score=0.0,
            )
            self._backend_metadata = {
                "name": "rapidocr_onnxruntime", "device": "cpu", "text_score": 0.0,
                "intra_op_num_threads": self.config.intra_op_num_threads,
                "inter_op_num_threads": self.config.inter_op_num_threads,
                "versions": {name: version(name) for name in ("rapidocr-onnxruntime", "onnxruntime")},
            }
        return self._backend

    def _rectified_fallback(self, result: StampReadResult, engine: OCRBackend) -> None:
        cfg, payload = self.config, result.payload
        width, height = cfg.line_image_size
        source = np.asarray(cfg.line_quad_xy, dtype=np.float32)
        target = np.asarray([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
        transform = cv2.getPerspectiveTransform(source, target)
        result.line_original = cv2.warpPerspective(result.roi_upright, transform, (width, height))
        gray = cv2.cvtColor(result.line_original, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        result.line_clahe = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        variants = []
        for name, image in (("original", result.line_original), ("clahe", result.line_clahe)):
            start = time.perf_counter()
            observations, times = engine(image.copy(), use_det=False, use_cls=False, use_rec=True)
            elapsed_ms = (time.perf_counter() - start) * 1000
            readings = []
            for text, score in observations or []:
                score = float(score)
                if not isinstance(text, str) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("OCR backend returned invalid direct text or confidence")
                readings.append({**_normalize_code(text), "score": score})
            variants.append({
                "name": name, "candidates": readings, "elapsed_ms": elapsed_ms,
                "backend_times": None if times is None else [float(value) for value in times],
            })
            payload["elapsed_ms"] += elapsed_ms
        payload["primary_reading"] = {key: payload[key] for key in (
            "raw_code", "normalized_code", "removed_characters", "unsupported_characters", "code_score",
            "state", "reasons", "code_width_fraction", "effective_code_geometry", "code_score_source",
        )}
        payload["rectified_reading"] = {
            "source_quad_xy": source.tolist(), "coordinate_space": "upright_roi",
            "image_size": list(cfg.line_image_size), "upright_to_line_transform": transform.tolist(),
            "variants": variants, "consensus_min_score": cfg.consensus_min_score,
        }
        failure = None
        if any(len(variant["candidates"]) != 1 for variant in variants):
            failure = "rectified_ambiguous_or_missing"
        else:
            first, second = (variant["candidates"][0] for variant in variants)
            code = first["normalized_code"]
            if not code or first["unsupported_characters"] or second["unsupported_characters"]:
                failure = "rectified_unsupported_or_empty"
            elif code != second["normalized_code"]:
                failure = "rectified_disagreement"
            elif min(first["score"], second["score"]) < cfg.consensus_min_score:
                failure = "rectified_low_score"
            else:
                indices = payload["code_line_indices"]
                compatible = False
                if len(indices) == 1:
                    primary = payload["normalized_code"]
                    compatible = bool(primary) and (
                        primary != code and primary in code
                        if "incomplete_code_span" in payload["reasons"] else primary == code
                    )
                elif len(indices) > 1:
                    ordered = sorted((payload["lines"][index] for index in indices), key=lambda item: min(
                        point[0] for point in item["box"]
                    ))
                    # Only separated, vertically aligned fragments provide concatenation evidence.
                    separated = all(
                        max(point[0] for point in left["box"]) <= min(point[0] for point in right["box"])
                        and max(min(point[1] for point in left["box"]), min(point[1] for point in right["box"]))
                        < min(max(point[1] for point in left["box"]), max(point[1] for point in right["box"]))
                        for left, right in zip(ordered, ordered[1:])
                    )
                    fragments = [_normalize_code(item["text"]) for item in ordered]
                    union_width = max(point[0] for item in ordered for point in item["box"]) - min(
                        point[0] for item in ordered for point in item["box"]
                    )
                    region_width = cfg.code_region_xyxy[2] - cfg.code_region_xyxy[0]
                    complete_span = union_width / region_width >= cfg.min_code_width_fraction
                    compatible = separated and complete_span and all(
                        item["normalized_code"] and not item["unsupported_characters"] for item in fragments
                    ) and "".join(item["normalized_code"] for item in fragments) == code
                if not compatible:
                    failure = "rectified_primary_conflict"
                elif not set(payload["reasons"]) <= {"low_score", "ambiguous_code_lines", "incomplete_code_span"}:
                    failure = "rectified_primary_reasons_unresolved"
                else:
                    payload.update({key: first[key] for key in (
                        "raw_code", "normalized_code", "removed_characters", "unsupported_characters",
                    )})
                    payload.update({
                        "state": "readable", "reasons": [], "selected_source": "rectified_consensus",
                        "code_score": min(first["score"], second["score"]),
                        "code_score_source": "minimum_of_rectified_original_and_clahe",
                        "effective_code_geometry": {"coordinate_space": "upright_roi", "box": source.tolist()},
                        "resolution_notes": ["rectified_variants_agree", "compatible_with_primary_observation"],
                    })
        if failure:
            payload["reasons"] = [*payload["reasons"], failure]

    def read(
        self, image_bgr: np.ndarray, *, capture_id: str | None = None,
        inspection_id: str | None = None, source_kind: str = "unknown",
    ) -> StampReadResult:
        """Read one BGR image and carry caller-supplied capture context.

        IDs are optional nonempty strings, never inferred from filenames.
        ``source_kind`` is unknown or fused_only, declared by the caller rather
        than verified by this reader. This interface does not accept source
        exposure groups. Uncertainty affects read state, not a product verdict.
        """
        for name, value in (("capture_id", capture_id), ("inspection_id", inspection_id)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a nonempty string when provided")
        if source_kind not in ("unknown", "fused_only"):
            raise ValueError("source_kind must be unknown or fused_only")
        cfg = self.config
        if not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8:
            raise ValueError("Input must be a uint8 BGR numpy array")
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Input must have three BGR channels")
        if image_bgr.shape[1::-1] != cfg.reference_image_size:
            raise ValueError(f"Image dimensions must be {cfg.reference_image_size}, got {image_bgr.shape[1::-1]}")
        x1, y1, x2, y2 = cfg.roi_xyxy
        original = image_bgr[y1:y2, x1:x2].copy()
        upright = cv2.rotate(original, cv2.ROTATE_90_CLOCKWISE)
        engine = self._engine()
        start = time.perf_counter()
        raw_lines, backend_times = engine(upright.copy())
        elapsed_ms = (time.perf_counter() - start) * 1000
        lines = []
        candidates = []
        reasons = []
        rx1, ry1, rx2, ry2 = cfg.code_region_xyxy
        overlay = upright.copy()
        cv2.rectangle(overlay, (rx1, ry1), (rx2 - 1, ry2 - 1), (255, 128, 0), 2)
        for box, text, score in raw_lines or []:
            points = np.asarray(box, dtype=float)
            if points.shape != (4, 2) or not np.isfinite(points).all():
                raise ValueError("OCR backend returned an invalid quadrilateral")
            score = float(score)
            if not isinstance(text, str) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("OCR backend returned invalid text or confidence")
            lines.append({"box": points.tolist(), "text": text, "score": score})
            center = points.mean(axis=0)
            is_code = rx1 <= center[0] < rx2 and ry1 <= center[1] < ry2
            if is_code:
                candidates.append(len(lines) - 1)
                if score < cfg.min_score:
                    reasons.append("low_score")
                if not text.strip():
                    reasons.append("empty_code")
                left, top = points.min(axis=0)
                right, bottom = points.max(axis=0)
                margin = cfg.border_margin_px
                if left <= rx1 + margin or top <= ry1 + margin or right >= rx2 - margin or bottom >= ry2 - margin:
                    reasons.append("code_box_near_region_border")
            cv2.polylines(overlay, [points.astype(np.int32)], True, (0, 0, 255) if is_code else (0, 180, 0), 2)
        if not candidates:
            reasons.append("missing_code")
        elif len(candidates) > 1:
            reasons.append("ambiguous_code_lines")
        width_fraction = None
        raw_code = None
        normalized_code = None
        removed_characters = []
        unsupported_characters = []
        if len(candidates) == 1:
            raw_code = lines[candidates[0]]["text"]
            normalized_code = raw_code
            if cfg.normalization_policy == "ascii_alphanumeric":
                normalized = _normalize_code(raw_code)
                normalized_code = normalized["normalized_code"]
                removed_characters = normalized["removed_characters"]
                unsupported_characters = normalized["unsupported_characters"]
                if unsupported_characters:
                    reasons.append("unsupported_characters")
                if not normalized_code:
                    reasons.append("empty_code")
            code_points = np.asarray(lines[candidates[0]]["box"])
            width_fraction = float(np.ptp(code_points[:, 0])) / (rx2 - rx1)
            if width_fraction < cfg.min_code_width_fraction:
                reasons.append("incomplete_code_span")
        reasons = list(dict.fromkeys(reasons))
        payload = {
            "scope": "stamp_reading_only", "business_rules_evaluated": False,
            "config": asdict(cfg), "box_coordinate_space": "upright_roi",
            "roi_coordinate_space": "full_image", "source_kind": source_kind,
            "capture_id": capture_id, "inspection_id": inspection_id,
            "backend": self._backend_metadata.copy(),
            "raw_code": raw_code, "normalized_code": normalized_code,
            "removed_characters": removed_characters, "unsupported_characters": unsupported_characters,
            "code_score": lines[candidates[0]]["score"] if len(candidates) == 1 else None,
            "code_width_fraction": width_fraction,
            "lines": lines, "code_line_indices": candidates,
            "state": "review" if reasons else "readable", "reasons": reasons,
            "elapsed_ms": elapsed_ms,
            "backend_times": None if backend_times is None else [float(value) for value in backend_times],
        }
        payload["selected_source"] = "primary"
        payload["code_score_source"] = "primary_detection"
        payload["effective_code_geometry"] = {
            "coordinate_space": "upright_roi",
            "box": lines[candidates[0]]["box"] if len(candidates) == 1 else None,
        }
        result = StampReadResult(payload, original, upright, overlay)
        if reasons and cfg.normalization_policy == "ascii_alphanumeric" and cfg.line_quad_xy is not None:
            self._rectified_fallback(result, engine)
        return result
