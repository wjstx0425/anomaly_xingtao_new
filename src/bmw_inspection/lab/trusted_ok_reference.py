"""Prepare immutable, human-reviewed BMW trusted-OK reference candidates."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import tempfile
import threading
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER, _atomic_publish_noreplace
from bmw_inspection.lab.eight_view_roi import load_roi_config


DATASET_FIELDS = (
    "sample_id",
    "physical_part_id",
    "session_id",
    "group_id",
    "view_id",
    "camera_serial",
    "source_path",
    "source_sha256",
    "source_class",
    "business_label",
    "split",
)
DECISION_FIELDS = ("physical_part_id", "sample_id", "decision", "reviewer", "review_note")
TRUSTED_OK_SESSION_ID = "20260810_210030_527506"
TRUSTED_OK_REFERENCE_RELEASE_ID = "bmw_right_20260810_21_train_normal_approved_v2"
TRUSTED_OK_REVIEW_ID = "bmw_right_20260810_21_train_normal_v2"
TRUSTED_OK_PREPROCESSING_IDENTITY = "pil_rgb_crop_png_v1"
TRUSTED_OK_PART_COUNT = 50
TRUSTED_OK_REFERENCE_COUNT = 400
_INDEX_FIELDS = {
    "approved_part_count",
    "preprocessing_identity",
    "reference_count_by_view",
    "references",
    "roi_config_path",
    "roi_config_sha256",
    "schema_version",
    "status",
    "whitelist_sha256",
}
_REFERENCE_FIELDS = {
    "business_label",
    "camera_serial",
    "full_image_path",
    "full_image_sha256",
    "group_id",
    "physical_part_id",
    "preprocessing_identity",
    "roi_config_sha256",
    "roi_image_path",
    "roi_image_sha256",
    "roi_xyxy",
    "sample_id",
    "session_id",
    "source_class",
    "source_path",
    "source_sha256",
    "split",
    "view_id",
    "whitelist_sha256",
}
_WHITELIST_FIELDS = {
    "approved_decisions",
    "approved_part_ids",
    "candidate_manifest_sha256",
    "preprocessing_identity",
    "review_decisions_sha256",
    "review_dir",
    "roi_config_path",
    "roi_config_sha256",
    "schema_version",
    "status",
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_PART_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_VIEW_LABELS = {
    "front": "前正面",
    "front_left": "前左侧",
    "front_right": "前右侧",
    "front_secondary": "前辅助",
    "back": "后正面",
    "back_left": "后左侧",
    "back_right": "后右侧",
    "back_secondary": "后辅助",
}
_CJK_FONTS = (
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
)


@dataclass(frozen=True, slots=True)
class CandidateImage:
    """One source image eligible for a human trusted-OK decision."""

    sample_id: str
    physical_part_id: str
    session_id: str
    group_id: str
    view_id: str
    camera_serial: str
    source_path: Path
    source_sha256: str
    source_class: str
    business_label: str
    split: str

    def manifest_row(self) -> dict[str, str]:
        """Return this image in the frozen prepared-manifest schema."""
        return {
            "sample_id": self.sample_id,
            "physical_part_id": self.physical_part_id,
            "session_id": self.session_id,
            "group_id": self.group_id,
            "view_id": self.view_id,
            "camera_serial": self.camera_serial,
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "source_class": self.source_class,
            "business_label": self.business_label,
            "split": self.split,
        }


@dataclass(frozen=True, slots=True)
class ReviewPackageSummary:
    """Counts and identity of one immutable manual-review package."""

    manifest_path: Path
    output_dir: Path
    session_id: str
    candidate_part_count: int
    candidate_image_count: int
    pending_decision_count: int
    contact_sheet_count: int


@dataclass(frozen=True, slots=True)
class TrustedIndexSummary:
    """Counts and identity of one immutable approved trusted-OK reference release."""

    review_dir: Path
    output_dir: Path
    approved_part_count: int
    reference_count_by_view: dict[str, int]
    whitelist_sha256: str
    roi_config_sha256: str


@dataclass(frozen=True, slots=True)
class TrustedOkMatch:
    """Display-only nearest approved reference for one actionable view."""

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
    source_sha256: str
    reference_full_sha256: str
    reference_roi_sha256: str
    index_sha256: str
    whitelist_sha256: str

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
        if isinstance(self.shift_x, bool) or not isinstance(self.shift_x, int):
            raise TypeError("shift_x must be an integer")
        if isinstance(self.shift_y, bool) or not isinstance(self.shift_y, int):
            raise TypeError("shift_y must be an integer")
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
        for name in (
            "source_sha256",
            "reference_full_sha256",
            "reference_roi_sha256",
            "index_sha256",
            "whitelist_sha256",
        ):
            if not _SHA256.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class _TrustedReference:
    physical_part_id: str
    sample_id: str
    view_id: str
    source_sha256: str
    full_path: Path
    full_sha256: str
    roi_path: Path
    roi_sha256: str


@dataclass(frozen=True, slots=True)
class _PreparedReference:
    row: _TrustedReference
    prepared: np.ndarray


def _gray_for_match(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError("trusted-OK matching requires grayscale, BGR, or BGRA images")


def _prepare_match_image(image: np.ndarray) -> np.ndarray:
    """Apply the frozen Template fit/reflect/blur preparation at 512 square."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0:
        raise ValueError("trusted-OK matching requires a non-empty uint8 image")
    gray = _gray_for_match(image)
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


def _difference_overlay(current: np.ndarray, aligned_reference: np.ndarray) -> np.ndarray:
    difference = cv2.absdiff(current, aligned_reference)
    maximum = int(difference.max())
    contrast = np.zeros_like(difference) if maximum == 0 else cv2.convertScaleAbs(
        difference, alpha=255.0 / maximum
    )
    overlay = cv2.applyColorMap(contrast, cv2.COLORMAP_INFERNO)
    positive = contrast[contrast > 0]
    if positive.size:
        threshold = max(32, int(np.percentile(positive, 75)))
        mask = np.where(contrast >= threshold, 255, 0).astype(np.uint8)
        count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count > 1:
            strongest = max(
                range(1, count),
                key=lambda label: (int(difference[labels == label].sum()), -label),
            )
            region = np.where(labels == strongest, 255, 0).astype(np.uint8)
            contours, _hierarchy = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    return overlay


class TrustedOkMatcher:
    """Match one actionable BMW view against an immutable approved-v2 bank."""

    def __init__(
        self,
        release_dir: Path,
        *,
        expected_index_sha256: str,
        max_shift: int = 12,
    ) -> None:
        raw_root = Path(release_dir).expanduser().absolute()
        if raw_root.name != TRUSTED_OK_REFERENCE_RELEASE_ID:
            raise ValueError(
                f"trusted-OK release identity must be {TRUSTED_OK_REFERENCE_RELEASE_ID}"
            )
        if not raw_root.is_dir() or raw_root.is_symlink():
            raise ValueError(f"trusted-OK release is not a regular directory: {raw_root}")
        if not isinstance(expected_index_sha256, str) or not _SHA256.fullmatch(
            expected_index_sha256
        ):
            raise ValueError("expected index SHA-256 must be lowercase hexadecimal")
        if isinstance(max_shift, bool) or not isinstance(max_shift, int) or not 0 <= max_shift <= 64:
            raise ValueError("max_shift must be an integer from 0 through 64")
        index_path = raw_root / "reference_index.json"
        if not index_path.is_file() or index_path.is_symlink():
            raise ValueError(f"trusted-OK index is not a regular file: {index_path}")
        actual_index_sha256 = _sha256(index_path)
        if actual_index_sha256 != expected_index_sha256:
            raise ValueError("trusted-OK index SHA-256 differs from expected_index_sha256")
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"trusted-OK index is invalid JSON: {index_path}") from error
        if not isinstance(payload, dict) or set(payload) != _INDEX_FIELDS:
            raise ValueError("trusted-OK index fields differ from the frozen v2 schema")
        if payload.get("schema_version") != 1 or payload.get("status") != "published":
            raise ValueError("trusted-OK index has unsupported schema or status")
        if payload.get("approved_part_count") != TRUSTED_OK_PART_COUNT:
            raise ValueError("trusted-OK index must contain exactly 50 approved parts")
        counts = payload.get("reference_count_by_view")
        if not isinstance(counts, dict) or set(counts) != set(VIEW_ORDER) or any(
            counts[view] != TRUSTED_OK_PART_COUNT for view in VIEW_ORDER
        ):
            raise ValueError("trusted-OK reference_count_by_view must be exactly 50 for every view")
        raw_references = payload.get("references")
        if not isinstance(raw_references, list) or len(raw_references) != TRUSTED_OK_REFERENCE_COUNT:
            raise ValueError("trusted-OK index must contain exactly 400 references")

        whitelist_path = raw_root / "trusted_ok_whitelist.json"
        if not whitelist_path.is_file() or whitelist_path.is_symlink():
            raise ValueError(f"trusted-OK whitelist is not a regular file: {whitelist_path}")
        try:
            whitelist = json.loads(whitelist_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"trusted-OK whitelist is invalid JSON: {whitelist_path}") from error
        if not isinstance(whitelist, dict) or set(whitelist) != _WHITELIST_FIELDS:
            raise ValueError("trusted-OK whitelist fields differ from the frozen v2 schema")
        if whitelist.get("schema_version") != 1 or whitelist.get("status") != "published":
            raise ValueError("trusted-OK whitelist has unsupported schema or status")
        whitelist_sha256 = self._digest(payload, "whitelist_sha256", "trusted-OK index")
        if _sha256(whitelist_path) != whitelist_sha256:
            raise ValueError("trusted-OK whitelist SHA-256 differs from the index")
        candidate_manifest_sha256 = self._digest(
            whitelist,
            "candidate_manifest_sha256",
            "trusted-OK whitelist",
        )
        self._digest(whitelist, "review_decisions_sha256", "trusted-OK whitelist")
        roi_config_sha256 = self._digest(payload, "roi_config_sha256", "trusted-OK index")
        if self._digest(whitelist, "roi_config_sha256", "trusted-OK whitelist") != roi_config_sha256:
            raise ValueError("trusted-OK whitelist and index roi_config_sha256 differ")
        if payload.get("roi_config_path") != whitelist.get("roi_config_path"):
            raise ValueError("trusted-OK whitelist and index roi_config_path differ")
        if (
            payload.get("preprocessing_identity") != TRUSTED_OK_PREPROCESSING_IDENTITY
            or whitelist.get("preprocessing_identity") != TRUSTED_OK_PREPROCESSING_IDENTITY
        ):
            raise ValueError("trusted-OK preprocessing_identity differs from the frozen v2 contract")
        review_dir = whitelist.get("review_dir")
        if not isinstance(review_dir, str) or Path(review_dir).name != TRUSTED_OK_REVIEW_ID:
            raise ValueError("trusted-OK whitelist review_dir differs from the frozen v2 identity")
        approved_part_ids, approved_samples = self._validate_whitelist(whitelist)

        by_view: dict[str, list[_TrustedReference]] = {view: [] for view in VIEW_ORDER}
        seen: set[tuple[str, str]] = set()
        for number, raw in enumerate(raw_references, start=1):
            if not isinstance(raw, dict) or set(raw) != _REFERENCE_FIELDS:
                raise ValueError(
                    f"trusted-OK reference {number} fields differ from the frozen v2 schema"
                )
            context = f"trusted-OK reference {number}"
            view = self._text(raw, "view_id", context)
            if view not in VIEW_ORDER:
                raise ValueError(f"{context} has unknown view_id")
            part_id = self._text(raw, "physical_part_id", context)
            if part_id not in approved_part_ids:
                raise ValueError(f"{context} physical_part_id is not approved by the whitelist")
            sample_id = self._text(raw, "sample_id", context)
            if approved_samples[part_id] != sample_id:
                raise ValueError(f"{context} sample_id differs from the whitelist")
            if self._text(raw, "session_id", context) != TRUSTED_OK_SESSION_ID:
                raise ValueError(f"{context} session_id differs from the frozen trusted session")
            if self._text(raw, "split", context) != "train":
                raise ValueError(f"{context} split must be train")
            if self._text(raw, "source_class", context) != "normal":
                raise ValueError(f"{context} source_class must be normal")
            if self._text(raw, "business_label", context) != "OK":
                raise ValueError(f"{context} business_label must be OK")
            if self._text(raw, "preprocessing_identity", context) != TRUSTED_OK_PREPROCESSING_IDENTITY:
                raise ValueError(f"{context} preprocessing_identity differs from the frozen contract")
            if self._digest(raw, "roi_config_sha256", context) != roi_config_sha256:
                raise ValueError(f"{context} roi_config_sha256 differs from the index")
            if self._digest(raw, "whitelist_sha256", context) != whitelist_sha256:
                raise ValueError(f"{context} whitelist_sha256 differs from the index")
            source_sha256 = self._digest(raw, "source_sha256", context)
            full_sha256 = self._digest(raw, "full_image_sha256", context)
            if source_sha256 != full_sha256:
                raise ValueError(f"{context} source_sha256 must equal full_image_sha256")
            roi_xyxy = raw.get("roi_xyxy")
            if (
                not isinstance(roi_xyxy, list)
                or len(roi_xyxy) != 4
                or any(isinstance(value, bool) or not isinstance(value, int) for value in roi_xyxy)
                or not (0 <= roi_xyxy[0] < roi_xyxy[2] and 0 <= roi_xyxy[1] < roi_xyxy[3])
            ):
                raise ValueError(f"{context} roi_xyxy is invalid")
            for field in ("camera_serial", "group_id", "source_path"):
                self._text(raw, field, context)
            identity = (view, part_id)
            if identity in seen:
                raise ValueError(f"{context} duplicates view/physical_part_id")
            seen.add(identity)
            by_view[view].append(
                _TrustedReference(
                    physical_part_id=part_id,
                    sample_id=sample_id,
                    view_id=view,
                    source_sha256=source_sha256,
                    full_path=self._asset_path(raw_root, raw, "full_image_path", context),
                    full_sha256=full_sha256,
                    roi_path=self._asset_path(raw_root, raw, "roi_image_path", context),
                    roi_sha256=self._digest(raw, "roi_image_sha256", context),
                )
            )
        if any(len(by_view[view]) != TRUSTED_OK_PART_COUNT for view in VIEW_ORDER):
            raise ValueError("trusted-OK reference_count_by_view differs from references")
        views_by_part: dict[str, set[str]] = defaultdict(set)
        for view, rows in by_view.items():
            for row in rows:
                views_by_part[row.physical_part_id].add(view)
        if set(views_by_part) != approved_part_ids or any(
            views != set(VIEW_ORDER) for views in views_by_part.values()
        ):
            raise ValueError("trusted-OK approved parts must each contain all eight views")
        self._index_sha256 = actual_index_sha256
        self._whitelist_sha256 = whitelist_sha256
        self._candidate_manifest_sha256 = candidate_manifest_sha256
        self._max_shift = max_shift
        self._by_view = {view: tuple(rows) for view, rows in by_view.items()}
        self._prepared: dict[tuple[str, Literal["roi", "full"]], tuple[_PreparedReference, ...]] = {}
        self._lock = threading.RLock()

    @classmethod
    def _validate_whitelist(
        cls,
        whitelist: Mapping[str, object],
    ) -> tuple[set[str], dict[str, str]]:
        raw_part_ids = whitelist.get("approved_part_ids")
        if (
            not isinstance(raw_part_ids, list)
            or len(raw_part_ids) != TRUSTED_OK_PART_COUNT
            or any(not isinstance(part_id, str) or not part_id.strip() for part_id in raw_part_ids)
            or raw_part_ids != sorted(set(raw_part_ids))
        ):
            raise ValueError("trusted-OK whitelist must contain 50 unique sorted approved_part_ids")
        raw_decisions = whitelist.get("approved_decisions")
        if not isinstance(raw_decisions, list) or len(raw_decisions) != TRUSTED_OK_PART_COUNT:
            raise ValueError("trusted-OK whitelist must contain 50 approved_decisions")
        approved_samples: dict[str, str] = {}
        for number, raw in enumerate(raw_decisions, start=1):
            context = f"trusted-OK approved decision {number}"
            if not isinstance(raw, dict) or set(raw) != set(DECISION_FIELDS):
                raise ValueError(f"{context} fields differ from the frozen v2 schema")
            if cls._text(raw, "decision", context) != "APPROVED":
                raise ValueError(f"{context} decision must be APPROVED")
            part_id = cls._text(raw, "physical_part_id", context)
            if part_id in approved_samples:
                raise ValueError(f"{context} duplicates physical_part_id")
            approved_samples[part_id] = cls._text(raw, "sample_id", context)
            cls._text(raw, "reviewer", context)
            cls._text(raw, "review_note", context)
        if set(approved_samples) != set(raw_part_ids):
            raise ValueError("trusted-OK approved decisions differ from approved_part_ids")
        return set(raw_part_ids), approved_samples

    @staticmethod
    def _text(raw: Mapping[str, object], field: str, context: str) -> str:
        value = raw.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{context} has invalid {field}")
        return value.strip()

    @classmethod
    def _digest(cls, raw: Mapping[str, object], field: str, context: str) -> str:
        value = cls._text(raw, field, context)
        if not _SHA256.fullmatch(value):
            raise ValueError(f"{context} has invalid {field}")
        return value

    @classmethod
    def _asset_path(
        cls, root: Path, raw: Mapping[str, object], field: str, context: str
    ) -> Path:
        relative = Path(cls._text(raw, field, context))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{context} has unsafe {field}")
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"{context} {field} is not a regular file")
        return path

    @staticmethod
    def _load_verified(path: Path, expected_sha256: str) -> np.ndarray:
        if not path.is_file() or path.is_symlink() or _sha256(path) != expected_sha256:
            raise ValueError(f"trusted-OK reference SHA-256 mismatch: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.dtype != np.uint8 or image.size == 0:
            raise ValueError(f"trusted-OK reference image is unavailable: {path}")
        return image

    def _prepared_references(
        self,
        view_id: str,
        comparison_mode: Literal["roi", "full"],
    ) -> tuple[_PreparedReference, ...]:
        key = (view_id, comparison_mode)
        cached = self._prepared.get(key)
        if cached is not None:
            return cached
        rows = self._by_view[view_id]
        if not rows:
            raise ValueError(f"trusted-OK index has no references for view {view_id}")
        prepared = tuple(
            _PreparedReference(
                row=row,
                prepared=_prepare_match_image(
                    self._load_verified(
                        row.full_path if comparison_mode == "full" else row.roi_path,
                        row.full_sha256 if comparison_mode == "full" else row.roi_sha256,
                    )
                ),
            )
            for row in rows
        )
        self._prepared[key] = prepared
        return prepared

    def preload(self) -> None:
        """Warm all ROI banks and the sole full-image bright-streak bank once."""
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
        """Return the deterministic highest normalized-correlation approved reference."""
        if view_id not in VIEW_ORDER:
            raise ValueError(f"unknown BMW view: {view_id}")
        if comparison_mode not in {"roi", "full"}:
            raise ValueError("comparison_mode must be roi or full")
        if comparison_mode == "full" and view_id != "front_left":
            raise ValueError("full comparison_mode is reserved for front_left bright streak")
        for name, image in (
            ("current_full_image", current_full_image),
            ("current_roi", current_roi),
        ):
            if (
                not isinstance(image, np.ndarray)
                or image.dtype != np.uint8
                or image.size == 0
                or image.ndim not in {2, 3}
            ):
                raise ValueError(f"{name} must be a non-empty uint8 image")
        current_full = current_full_image
        current_region = current_full if comparison_mode == "full" else current_roi
        prepared_current = _prepare_match_image(current_region)
        with self._lock:
            key = (view_id, comparison_mode)
            if key not in self._prepared:
                raise RuntimeError("TrustedOkMatcher.preload() must complete before match()")
            candidates = self._prepared[key]
            best: tuple[float, int, int, _PreparedReference, np.ndarray] | None = None
            for candidate in candidates:
                padded = cv2.copyMakeBorder(
                    candidate.prepared,
                    self._max_shift,
                    self._max_shift,
                    self._max_shift,
                    self._max_shift,
                    cv2.BORDER_REFLECT_101,
                )
                response = cv2.matchTemplate(padded, prepared_current, cv2.TM_CCOEFF_NORMED)
                _minimum, similarity, _min_location, location = cv2.minMaxLoc(response)
                if not math.isfinite(similarity):
                    similarity = -1.0
                x, y = location
                aligned = padded[y : y + 512, x : x + 512].copy()
                if best is None or similarity > best[0]:
                    best = (float(similarity), x - self._max_shift, y - self._max_shift, candidate, aligned)
            if best is None:
                raise ValueError(f"trusted-OK index has no references for view {view_id}")
            similarity, shift_x, shift_y, selected, aligned = best
            row = selected.row
            reference_full = self._load_verified(row.full_path, row.full_sha256)
            reference_region = self._load_verified(row.roi_path, row.roi_sha256)
        return TrustedOkMatch(
            view_id=view_id,
            comparison_mode=comparison_mode,
            physical_part_id=row.physical_part_id,
            sample_id=row.sample_id,
            similarity=similarity,
            shift_x=shift_x,
            shift_y=shift_y,
            current_full_image=current_full,
            reference_full_image=reference_full,
            current_roi=current_region,
            reference_roi=reference_region,
            aligned_reference_roi=aligned,
            difference_overlay=_difference_overlay(prepared_current, aligned),
            source_sha256=row.source_sha256,
            reference_full_sha256=row.full_sha256,
            reference_roi_sha256=row.roi_sha256,
            index_sha256=self._index_sha256,
            whitelist_sha256=self._whitelist_sha256,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _required(row: Mapping[str, str], field: str, *, context: str) -> str:
    value = row.get(field, "").strip()
    if not value:
        raise ValueError(f"{context} has empty {field}")
    return value


def _read_manifest(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != DATASET_FIELDS:
            raise ValueError(f"dataset manifest header differs from the frozen BMW schema: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"dataset manifest is empty: {path}")
    if any(None in row for row in rows):
        raise ValueError(f"dataset manifest contains malformed or extra cells: {path}")
    return tuple(rows)


def _candidate_rows(
    rows: Iterable[Mapping[str, str]], session_id: str, *, include_incomplete: bool = False
) -> tuple[CandidateImage, ...]:
    grouped: dict[str, list[CandidateImage]] = defaultdict(list)
    for row_number, raw in enumerate(rows, start=2):
        context = f"dataset manifest row {row_number}"
        if raw.get("session_id", "").strip() != session_id:
            continue
        if (
            raw.get("source_class", "").strip() != "normal"
            or raw.get("business_label", "").strip() != "OK"
            or raw.get("split", "").strip() != "train"
        ):
            continue
        source_sha256 = _required(raw, "source_sha256", context=context).lower()
        if not _SHA256.fullmatch(source_sha256):
            raise ValueError(f"{context} has invalid source_sha256")
        physical_part_id = _required(raw, "physical_part_id", context=context)
        if not _SAFE_PART_ID.fullmatch(physical_part_id):
            raise ValueError(f"{context} has unsafe physical_part_id")
        grouped[physical_part_id].append(
            CandidateImage(
                sample_id=_required(raw, "sample_id", context=context),
                physical_part_id=physical_part_id,
                session_id=_required(raw, "session_id", context=context),
                group_id=_required(raw, "group_id", context=context),
                view_id=_required(raw, "view_id", context=context),
                camera_serial=_required(raw, "camera_serial", context=context),
                source_path=Path(_required(raw, "source_path", context=context)).expanduser().absolute(),
                source_sha256=source_sha256,
                source_class=_required(raw, "source_class", context=context),
                business_label=_required(raw, "business_label", context=context),
                split=_required(raw, "split", context=context),
            )
        )

    candidates: list[CandidateImage] = []
    view_index = {view: index for index, view in enumerate(VIEW_ORDER)}
    for part_id, part_rows in sorted(grouped.items()):
        identities = {(row.sample_id, row.session_id, row.group_id) for row in part_rows}
        if len(identities) != 1:
            raise ValueError(f"candidate physical part has inconsistent identity: {part_id}")
        view_ids = [row.view_id for row in part_rows]
        if any(view_id not in view_index for view_id in view_ids):
            raise ValueError(f"candidate physical part has unsupported view: {part_id}")
        if len(view_ids) != len(set(view_ids)):
            raise ValueError(f"candidate physical part has duplicate views: {part_id}")
        if set(view_ids) != set(VIEW_ORDER) and not include_incomplete:
            continue
        candidates.extend(sorted(part_rows, key=lambda row: view_index[row.view_id]))
    if not candidates:
        raise ValueError("no complete training-normal eight-view candidates found for session")
    return tuple(candidates)


def _verify_sources(rows: Iterable[CandidateImage]) -> None:
    for row in rows:
        if not row.source_path.is_file() or row.source_path.is_symlink():
            raise ValueError(f"candidate source image is not a regular file: {row.source_path}")
        actual = _sha256(row.source_path)
        if actual != row.source_sha256:
            raise ValueError(f"candidate source_sha256 mismatch: {row.source_path}")


def _write_csv(path: Path, fields: tuple[str, ...], rows: Iterable[Mapping[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _CJK_FONTS:
        if path.is_file():
            return ImageFont.truetype(path, size=22)
    return ImageFont.load_default()


def _contact_sheet(rows: tuple[CandidateImage, ...], destination: Path) -> None:
    cell_width, image_height, label_height = 320, 240, 42
    sheet = Image.new("RGB", (cell_width * 4, (image_height + label_height) * 2), "white")
    draw = ImageDraw.Draw(sheet)
    font = _font()
    for index, row in enumerate(rows):
        column, line = index % 4, index // 4
        origin_x = column * cell_width
        origin_y = line * (image_height + label_height)
        with Image.open(row.source_path) as source:
            rendered = ImageOps.contain(source.convert("RGB"), (cell_width, image_height), Image.Resampling.LANCZOS)
        image_x = origin_x + (cell_width - rendered.width) // 2
        image_y = origin_y + (image_height - rendered.height) // 2
        sheet.paste(rendered, (image_x, image_y))
        label = f"{_VIEW_LABELS[row.view_id]} ({row.view_id})"
        draw.text((origin_x + 8, origin_y + image_height + 8), label, fill="black", font=font)
    sheet.save(destination, format="PNG")


def prepare_review_package(manifest: Path, output_dir: Path, *, session_id: str) -> ReviewPackageSummary:
    """Publish PENDING-only human-review candidates from one BMW capture session."""
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string")
    normalized_session_id = session_id.strip()
    if normalized_session_id != TRUSTED_OK_SESSION_ID:
        raise ValueError(f"trusted-OK review package only supports session_id {TRUSTED_OK_SESSION_ID}")
    manifest_path = Path(manifest).expanduser().resolve()
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"dataset manifest is not a regular file: {manifest_path}")
    output = Path(output_dir).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"review package already exists: {output}")

    candidates = _candidate_rows(_read_manifest(manifest_path), normalized_session_id)
    _verify_sources(candidates)
    part_rows = {
        part_id: tuple(rows)
        for part_id, rows in _group_by_part(candidates).items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        _write_csv(staging / "candidate_manifest.csv", DATASET_FIELDS, (row.manifest_row() for row in candidates))
        candidate_manifest_sha256 = _sha256(staging / "candidate_manifest.csv")
        decisions = [
            {
                "physical_part_id": part_id,
                "sample_id": rows[0].sample_id,
                "decision": "PENDING",
                "reviewer": "",
                "review_note": "",
            }
            for part_id, rows in part_rows.items()
        ]
        _write_csv(staging / "review_decisions.csv", DECISION_FIELDS, decisions)
        contact_sheets = staging / "review" / "contact_sheets"
        contact_sheets.mkdir(parents=True)
        for part_id, rows in part_rows.items():
            _contact_sheet(rows, contact_sheets / f"{part_id}.png")
        payload = {
            "schema_version": 1,
            "status": "awaiting_human_review",
            "automatic_approvals": 0,
            "session_id": normalized_session_id,
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "candidate_manifest_sha256": candidate_manifest_sha256,
            "candidate_part_count": len(part_rows),
            "candidate_image_count": len(candidates),
            "pending_decision_count": len(decisions),
            "contact_sheet_count": len(part_rows),
            "required_decision": "PENDING",
        }
        (staging / "review_package.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _atomic_publish_noreplace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return ReviewPackageSummary(
        manifest_path=manifest_path,
        output_dir=output,
        session_id=normalized_session_id,
        candidate_part_count=len(part_rows),
        candidate_image_count=len(candidates),
        pending_decision_count=len(part_rows),
        contact_sheet_count=len(part_rows),
    )


def _group_by_part(rows: Iterable[CandidateImage]) -> dict[str, list[CandidateImage]]:
    grouped: dict[str, list[CandidateImage]] = defaultdict(list)
    for row in rows:
        grouped[row.physical_part_id].append(row)
    return {part_id: grouped[part_id] for part_id in sorted(grouped)}


def _read_review_decisions(path: Path) -> dict[str, dict[str, str]]:
    """Read exact human decisions without silently normalizing their meaning."""
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"review decisions are not a regular file: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != DECISION_FIELDS:
            raise ValueError(f"review decisions header differs from the frozen schema: {path}")
        rows = list(reader)
    if not rows or any(None in row for row in rows):
        raise ValueError("review decisions are empty or malformed")

    decisions: dict[str, dict[str, str]] = {}
    for row_number, raw in enumerate(rows, start=2):
        context = f"review decision row {row_number}"
        part_id = _required(raw, "physical_part_id", context=context)
        sample_id = _required(raw, "sample_id", context=context)
        decision = _required(raw, "decision", context=context)
        if decision not in {"APPROVED", "REJECTED", "PENDING"}:
            raise ValueError(f"{context} has unsupported decision {decision!r}")
        if part_id in decisions:
            raise ValueError(f"review decisions contain duplicate physical_part_id: {part_id}")
        if decision == "APPROVED":
            _required(raw, "reviewer", context=context)
            _required(raw, "review_note", context=context)
        decisions[part_id] = {
            "physical_part_id": part_id,
            "sample_id": sample_id,
            "decision": decision,
            "reviewer": raw.get("reviewer", "").strip(),
            "review_note": raw.get("review_note", "").strip(),
        }
    return decisions


def _validate_review_package(review_dir: Path) -> tuple[dict[str, list[CandidateImage]], dict[str, dict[str, str]]]:
    """Bind decision rows to the frozen candidate manifest, without evaluating excluded rows."""
    if not review_dir.is_dir() or review_dir.is_symlink():
        raise ValueError(f"review package is not a regular directory: {review_dir}")
    package_path = review_dir / "review_package.json"
    candidate_path = review_dir / "candidate_manifest.csv"
    if not package_path.is_file() or package_path.is_symlink():
        raise ValueError(f"review package metadata is not a regular file: {package_path}")
    try:
        package = json.loads(package_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"review package metadata is invalid JSON: {package_path}") from error
    if package.get("schema_version") != 1 or package.get("status") != "awaiting_human_review":
        raise ValueError("review package metadata has unsupported schema or status")
    if not candidate_path.is_file() or candidate_path.is_symlink():
        raise ValueError(f"candidate manifest is not a regular file: {candidate_path}")
    expected_candidate_sha256 = package.get("candidate_manifest_sha256")
    if not isinstance(expected_candidate_sha256, str) or not _SHA256.fullmatch(expected_candidate_sha256):
        raise ValueError("review package candidate_manifest_sha256 must be a lowercase SHA-256")
    if _sha256(candidate_path) != expected_candidate_sha256:
        raise ValueError("review package candidate_manifest_sha256 differs from candidate_manifest.csv")

    candidates = _group_by_part(
        _candidate_rows(_read_manifest(candidate_path), TRUSTED_OK_SESSION_ID, include_incomplete=True)
    )
    decisions = _read_review_decisions(review_dir / "review_decisions.csv")
    if set(decisions) != set(candidates):
        raise ValueError("review decisions must cover every and only candidate physical part")
    for part_id, decision in decisions.items():
        candidate_sample_ids = {row.sample_id for row in candidates[part_id]}
        if decision["sample_id"] not in candidate_sample_ids:
            raise ValueError(f"review decision sample_id differs from candidate manifest: {part_id}")
    return candidates, decisions


def _copy_approved_reference(
    row: CandidateImage,
    *,
    staging: Path,
    roi: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    roi_config_sha256: str,
    whitelist_sha256: str,
) -> dict[str, object]:
    """Copy one source frame and derive its deterministic RGB part crop."""
    if not row.source_path.is_file() or row.source_path.is_symlink():
        raise ValueError(f"approved source image is not a regular file: {row.source_path}")
    suffix = row.source_path.suffix.lower() or ".img"
    full_relative = Path("references") / row.view_id / "full" / f"{row.physical_part_id}{suffix}"
    roi_relative = Path("references") / row.view_id / "roi" / f"{row.physical_part_id}.png"
    full_path = staging / full_relative
    roi_path = staging / roi_relative
    full_path.parent.mkdir(parents=True, exist_ok=True)
    roi_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(row.source_path, full_path)
    full_sha256 = _sha256(full_path)
    if full_sha256 != row.source_sha256:
        raise ValueError(f"copied full reference SHA-256 mismatch: {row.source_path}")
    with Image.open(row.source_path) as source:
        rgb = source.convert("RGB")
        if rgb.size != (image_width, image_height):
            raise ValueError(f"approved source dimensions differ from ROI config: {row.source_path}")
        x1, y1, x2, y2 = roi
        cropped = rgb.crop((x1, y1, x2, y2))
        cropped.save(roi_path, format="PNG")
    roi_sha256 = _sha256(roi_path)
    return {
        "physical_part_id": row.physical_part_id,
        "sample_id": row.sample_id,
        "session_id": row.session_id,
        "group_id": row.group_id,
        "view_id": row.view_id,
        "camera_serial": row.camera_serial,
        "source_path": str(row.source_path),
        "source_sha256": row.source_sha256,
        "source_class": row.source_class,
        "business_label": row.business_label,
        "split": row.split,
        "full_image_path": str(full_relative),
        "full_image_sha256": full_sha256,
        "roi_image_path": str(roi_relative),
        "roi_image_sha256": roi_sha256,
        "roi_xyxy": list(roi),
        "roi_config_sha256": roi_config_sha256,
        "whitelist_sha256": whitelist_sha256,
        "preprocessing_identity": "pil_rgb_crop_png_v1",
    }


def publish_trusted_reference_index(review_dir: Path, output_dir: Path, roi_config: Path) -> TrustedIndexSummary:
    """Publish an immutable reference index for complete, explicitly approved BMW parts."""
    review = Path(review_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().absolute()
    roi_path = Path(roi_config).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"trusted-OK reference release already exists: {output}")
    candidates, decisions = _validate_review_package(review)
    approved_part_ids = [part_id for part_id, decision in decisions.items() if decision["decision"] == "APPROVED"]
    if not approved_part_ids:
        raise ValueError("no APPROVED physical parts are available for trusted reference publication")
    approved_part_ids.sort()
    for part_id in approved_part_ids:
        rows = candidates[part_id]
        if len(rows) != len(VIEW_ORDER) or {row.view_id for row in rows} != set(VIEW_ORDER):
            raise ValueError(f"approved physical part must contain exactly eight views: {part_id}")
        _verify_sources(rows)

    roi = load_roi_config(roi_path)
    roi_config_sha256 = _sha256(roi_path)
    candidate_manifest_sha256 = _sha256(review / "candidate_manifest.csv")
    decision_sha256 = _sha256(review / "review_decisions.csv")
    whitelist = {
        "schema_version": 1,
        "status": "published",
        "review_dir": str(review),
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "review_decisions_sha256": decision_sha256,
        "roi_config_path": str(roi_path),
        "roi_config_sha256": roi_config_sha256,
        "preprocessing_identity": "pil_rgb_crop_png_v1",
        "approved_part_ids": approved_part_ids,
        "approved_decisions": [decisions[part_id] for part_id in approved_part_ids],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        whitelist_path = staging / "trusted_ok_whitelist.json"
        whitelist_path.write_text(
            json.dumps(whitelist, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        whitelist_sha256 = _sha256(whitelist_path)
        references: list[dict[str, object]] = []
        for part_id in approved_part_ids:
            by_view = {row.view_id: row for row in candidates[part_id]}
            for view in VIEW_ORDER:
                references.append(
                    _copy_approved_reference(
                        by_view[view],
                        staging=staging,
                        roi=roi.part_rois[view],
                        image_width=roi.image_width,
                        image_height=roi.image_height,
                        roi_config_sha256=roi_config_sha256,
                        whitelist_sha256=whitelist_sha256,
                    )
                )
        view_counts = {view: sum(row["view_id"] == view for row in references) for view in VIEW_ORDER}
        index = {
            "schema_version": 1,
            "status": "published",
            "whitelist_sha256": whitelist_sha256,
            "roi_config_path": str(roi_path),
            "roi_config_sha256": roi_config_sha256,
            "preprocessing_identity": "pil_rgb_crop_png_v1",
            "approved_part_count": len(approved_part_ids),
            "reference_count_by_view": view_counts,
            "references": references,
        }
        (staging / "reference_index.json").write_text(
            json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _atomic_publish_noreplace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return TrustedIndexSummary(
        review_dir=review,
        output_dir=output,
        approved_part_count=len(approved_part_ids),
        reference_count_by_view=view_counts,
        whitelist_sha256=whitelist_sha256,
        roi_config_sha256=roi_config_sha256,
    )
