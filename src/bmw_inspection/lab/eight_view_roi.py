"""Dataset-bound ROI contracts for BMW eight-view HDR images."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from bmw_inspection.lab.eight_view_dataset import (
    CAPTURE_SCOPES,
    PreparedImage,
    SOURCE_CLASSES,
    VIEW_ORDER,
    read_complete_capture_rows,
)


_MANIFEST_FIELDS = (
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
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_BINDING_MODES = ("prepared_manifest", "fixed_setup")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RepresentativeSelection:
    """One deterministic complete normal/train sample used to select all ROIs."""

    dataset_id: str
    sample_id: str
    physical_part_id: str
    manifest_path: Path
    manifest_sha256: str
    image_width: int
    image_height: int
    images: Mapping[str, Path]


@dataclass(frozen=True, slots=True)
class EightViewRoiConfig:
    """Eight half-open pixel ROIs with strict or fixed-setup reuse binding."""

    dataset_id: str
    source_manifest: Path
    source_manifest_sha256: str
    representative_sample_id: str
    image_width: int
    image_height: int
    part_rois: Mapping[str, tuple[int, int, int, int]]
    binding_mode: str = "prepared_manifest"
    capture_scope: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str) or not self.dataset_id.strip():
            raise ValueError("dataset_id must be a non-empty string")
        if not isinstance(self.representative_sample_id, str) or not self.representative_sample_id.strip():
            raise ValueError("representative_sample_id must be a non-empty string")
        if self.binding_mode not in _BINDING_MODES:
            raise ValueError(f"binding_mode must be one of {_BINDING_MODES}")
        if self.binding_mode == "fixed_setup" and self.capture_scope not in CAPTURE_SCOPES:
            raise ValueError(f"fixed_setup capture_scope must be one of {CAPTURE_SCOPES}")
        if self.binding_mode == "prepared_manifest" and self.capture_scope is not None:
            raise ValueError("prepared_manifest ROI must not define capture_scope")
        if not _SHA256.fullmatch(self.source_manifest_sha256):
            raise ValueError("source_manifest_sha256 must be lowercase SHA-256")
        if isinstance(self.image_width, bool) or not isinstance(self.image_width, int) or self.image_width <= 0:
            raise ValueError("image_width must be a positive integer")
        if isinstance(self.image_height, bool) or not isinstance(self.image_height, int) or self.image_height <= 0:
            raise ValueError("image_height must be a positive integer")
        if tuple(self.part_rois) != VIEW_ORDER:
            raise ValueError("part_rois must contain exactly the eight BMW views in canonical order")
        normalized: dict[str, tuple[int, int, int, int]] = {}
        for view in VIEW_ORDER:
            roi = self.part_rois[view]
            if (
                not isinstance(roi, (list, tuple))
                or len(roi) != 4
                or any(isinstance(value, bool) or not isinstance(value, int) for value in roi)
            ):
                raise ValueError(f"ROI for {view} must contain four integers")
            x1, y1, x2, y2 = roi
            if not (0 <= x1 < x2 <= self.image_width and 0 <= y1 < y2 <= self.image_height):
                raise ValueError(f"ROI for {view} lies outside the configured image dimensions")
            normalized[view] = (x1, y1, x2, y2)
        object.__setattr__(self, "source_manifest", Path(self.source_manifest).expanduser().resolve())
        object.__setattr__(self, "part_rois", normalized)


def fit_image_for_display(
    image: np.ndarray,
    *,
    max_display_width: int,
    max_display_height: int,
) -> np.ndarray:
    """Fit an image inside the operator screen without changing aspect ratio."""
    if not isinstance(image, np.ndarray) or image.ndim not in {2, 3}:
        raise TypeError("ROI source must be a numpy image")
    if max_display_width <= 0 or max_display_height <= 0:
        raise ValueError("ROI display dimensions must be positive")
    source_height, source_width = image.shape[:2]
    scale = min(1.0, max_display_width / source_width, max_display_height / source_height)
    size = (max(1, round(source_width * scale)), max(1, round(source_height * scale)))
    if size == (source_width, source_height):
        return image
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def source_roi(
    display_xywh: tuple[int, int, int, int],
    display_shape: tuple[int, int],
    source_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Map display XYWH to clamped half-open source coordinates."""
    x, y, width, height = display_xywh
    if width <= 0 or height <= 0:
        raise ValueError("ROI selection must have positive width and height")
    display_height, display_width = display_shape
    source_height, source_width = source_shape
    scale_x = source_width / display_width
    scale_y = source_height / display_height
    x1 = max(0, math.floor(x * scale_x))
    y1 = max(0, math.floor(y * scale_y))
    x2 = min(source_width, math.ceil((x + width) * scale_x))
    y2 = min(source_height, math.ceil((y + height) * scale_y))
    if not (0 <= x1 < x2 <= source_width and 0 <= y1 < y2 <= source_height):
        raise ValueError("selected ROI lies outside the source image")
    return x1, y1, x2, y2


def select_representative_images(
    prepared_root: Path,
    *,
    physical_part_id: str | None = None,
) -> RepresentativeSelection:
    """Select and preflight one complete normal/train sample before opening a GUI."""
    root = Path(prepared_root).expanduser().resolve()
    report_path = root / "report.json"
    manifest_path = root / "manifests/dataset_manifest.csv"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("release_status", "published") != "published":
        raise ValueError("prepared release report is not a published schema-v1 dataset")
    dataset_id = report.get("dataset_id")
    if not isinstance(dataset_id, str) or dataset_id != root.name:
        raise ValueError("prepared release dataset_id differs from its directory")
    manifest_sha256 = _sha256(manifest_path)
    expected_hash = report.get("manifest_sha256", {}).get("dataset_manifest.csv")
    if expected_hash is not None and expected_hash != manifest_sha256:
        raise ValueError("prepared dataset manifest SHA-256 differs from report")
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != _MANIFEST_FIELDS:
            raise ValueError("prepared dataset manifest header differs from the frozen schema")
        rows = list(reader)
    candidates: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row["source_class"] != "normal" or row["business_label"] != "OK" or row["split"] != "train":
            continue
        if physical_part_id is not None and row["physical_part_id"] != physical_part_id:
            continue
        key = (row["physical_part_id"], row["sample_id"], row["session_id"])
        candidates.setdefault(key, []).append(row)
    complete = [
        (key, group)
        for key, group in sorted(candidates.items())
        if len(group) == len(VIEW_ORDER) and {row["view_id"] for row in group} == set(VIEW_ORDER)
    ]
    if not complete:
        raise ValueError("no complete normal/train eight-view representative sample was found")
    (part_id, sample_id, _session_id), selected_rows = complete[0]
    by_view = {row["view_id"]: row for row in selected_rows}
    images: dict[str, Path] = {}
    dimensions: set[tuple[int, int]] = set()
    for view in VIEW_ORDER:
        row = by_view[view]
        path = Path(row["source_path"]).expanduser().resolve()
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"representative source image is missing or a symlink: {path}")
        if _sha256(path) != row["source_sha256"]:
            raise ValueError(f"representative source image SHA-256 mismatch: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.size == 0:
            raise ValueError(f"cannot decode representative source image: {path}")
        height, width = image.shape[:2]
        dimensions.add((width, height))
        images[view] = path
    if len(dimensions) != 1:
        raise ValueError("representative images have inconsistent dimensions")
    width, height = next(iter(dimensions))
    if report.get("image_width") != width or report.get("image_height") != height:
        raise ValueError("representative image dimensions differ from prepared report")
    return RepresentativeSelection(
        dataset_id=dataset_id,
        sample_id=sample_id,
        physical_part_id=part_id,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        image_width=width,
        image_height=height,
        images=images,
    )


def select_raw_representative_images(
    raw_root: Path,
    *,
    profile_id: str,
    capture_scope: str,
    source_class: str | None = None,
    sample_id: str | None = None,
) -> RepresentativeSelection:
    """Select one complete raw capture as a fixed-setup ROI reference.

    Args:
        raw_root: Root containing the shared capture manifests and hand directories.
        profile_id: Stable identifier for the reusable ROI profile.
        capture_scope: Top-level capture hand to select, either ``left`` or ``right``.
        source_class: Optional source class such as ``others`` or ``normal``.
        sample_id: Optional exact complete sample identifier.

    Returns:
        One deterministic complete eight-view sample and its source-manifest provenance.

    Raises:
        ValueError: If the filters are invalid or no complete matching sample exists.
    """
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise ValueError("profile_id must be a non-empty string")
    if source_class is not None and source_class not in SOURCE_CLASSES:
        raise ValueError(f"source_class must be one of {SOURCE_CLASSES}")
    if sample_id is not None and (not isinstance(sample_id, str) or not sample_id.strip()):
        raise ValueError("sample_id must be a non-empty string when provided")
    root = Path(raw_root).expanduser().resolve()
    rows, audit = read_complete_capture_rows(
        root,
        verify_image_hash=False,
        capture_scope=capture_scope,
    )
    candidates: dict[tuple[str, str, str], list[PreparedImage]] = {}
    for row in rows:
        if source_class is not None and row.source_class != source_class:
            continue
        if sample_id is not None and row.sample_id != sample_id:
            continue
        key = (row.session_id, row.sample_id, row.physical_part_id)
        candidates.setdefault(key, []).append(row)
    complete = [
        (key, group)
        for key, group in sorted(candidates.items())
        if len(group) == len(VIEW_ORDER) and {row.view_id for row in group} == set(VIEW_ORDER)
    ]
    if not complete:
        raise ValueError("no complete raw eight-view representative sample matched the requested filters")
    (session_id, selected_sample_id, physical_part_id), selected_rows = complete[0]
    by_view = {row.view_id: row for row in selected_rows}
    manifest_path = (root / "manifests" / f"{session_id}.csv").resolve()
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"raw reference manifest is missing or a symlink: {manifest_path}")
    return RepresentativeSelection(
        dataset_id=profile_id.strip(),
        sample_id=selected_sample_id,
        physical_part_id=physical_part_id,
        manifest_path=manifest_path,
        manifest_sha256=_sha256(manifest_path),
        image_width=audit.image_width,
        image_height=audit.image_height,
        images={view: by_view[view].source_path for view in VIEW_ORDER},
    )


def save_roi_config(path: Path, config: EightViewRoiConfig, *, force: bool = False) -> None:
    """Atomically save one strictly validated ROI asset."""
    destination = Path(path).expanduser().resolve()
    if destination.exists() and not force:
        raise FileExistsError(f"ROI config already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema_version": 1 if config.binding_mode == "prepared_manifest" else 2,
        "coordinate_system": "pixel_xyxy_half_open",
        "representative_sample_id": config.representative_sample_id,
        "image_width": config.image_width,
        "image_height": config.image_height,
        "part_rois": {view: list(config.part_rois[view]) for view in VIEW_ORDER},
    }
    if config.binding_mode == "prepared_manifest":
        payload.update(
            {
                "dataset_id": config.dataset_id,
                "source_manifest": str(config.source_manifest),
                "source_manifest_sha256": config.source_manifest_sha256,
            }
        )
    else:
        payload.update(
            {
                "binding_mode": config.binding_mode,
                "profile_id": config.dataset_id,
                "capture_scope": config.capture_scope,
                "reference_manifest": str(config.source_manifest),
                "reference_manifest_sha256": config.source_manifest_sha256,
            }
        )
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".json", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        load_roi_config(temporary)
        if destination.exists() and not force:
            raise FileExistsError(f"ROI config already exists: {destination}")
        if force:
            temporary.replace(destination)
        else:
            temporary.rename(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_roi_config(path: Path) -> EightViewRoiConfig:
    """Load an ROI asset and verify its prepared or reference manifest."""
    config_path = Path(path).expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    schema_version = payload.get("schema_version")
    if schema_version not in {1, 2} or payload.get("coordinate_system") != "pixel_xyxy_half_open":
        raise ValueError("ROI config schema or coordinate system is unsupported")
    if schema_version == 1:
        dataset_id = payload.get("dataset_id")
        source_manifest = payload.get("source_manifest", "")
        source_manifest_sha256 = payload.get("source_manifest_sha256")
        binding_mode = "prepared_manifest"
        capture_scope = None
    else:
        dataset_id = payload.get("profile_id")
        source_manifest = payload.get("reference_manifest", "")
        source_manifest_sha256 = payload.get("reference_manifest_sha256")
        binding_mode = payload.get("binding_mode")
        capture_scope = payload.get("capture_scope")
    config = EightViewRoiConfig(
        dataset_id=dataset_id,
        source_manifest=Path(source_manifest),
        source_manifest_sha256=source_manifest_sha256,
        representative_sample_id=payload.get("representative_sample_id"),
        image_width=payload.get("image_width"),
        image_height=payload.get("image_height"),
        part_rois=payload.get("part_rois", {}),
        binding_mode=binding_mode,
        capture_scope=capture_scope,
    )
    if not config.source_manifest.is_file() or _sha256(config.source_manifest) != config.source_manifest_sha256:
        raise ValueError("ROI config source manifest is missing or has a different SHA-256")
    return config
