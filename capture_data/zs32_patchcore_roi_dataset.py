# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Discover ZS32 PatchCore images and validate per-hand, per-view ROI configuration."""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from uuid import uuid4

import cv2
from capture_data.select_roi import save_overlay, select_roi
from rich.progress import track
from zs32_inspection.domain.views import CANONICAL_VIEWS

HANDS = ("right", "left")
VIEWS = ("front", "front_left", "front_right", "back", "back_left", "back_right")
IMAGE_EXTENSIONS = (".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff")
ROI = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class SourceImage:
    """Stable identity and destination metadata for one PatchCore source image.

    Args:
        source_path (Path): Discovered source image path.
        hand (str): Source hand, either ``right`` or ``left``.
        source_view (str): View directory containing the source image.
        resolved_view (str): Output view, always equal to the source directory view.
        label (str): Classification label directory.
        defect_type (str): Defect subtype, or an empty string for normal data.
        session_id (str): Capture session directory name.
        relative_tail (Path): Source hierarchy following the view directory.
        view_corrected (bool): Compatibility field, always false because directories are authoritative.
    """

    source_path: Path
    hand: str
    source_view: str
    resolved_view: str
    label: str
    defect_type: str
    session_id: str
    relative_tail: Path
    view_corrected: bool


@dataclass(frozen=True, slots=True)
class _PreparedImage:
    """Read-only conversion metadata validated during preflight."""

    source: SourceImage
    output_relative: Path
    roi: ROI
    source_width: int
    source_height: int
    crop_width: int
    crop_height: int


def _require_mapping(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        msg = f"ROI config {field} must be an object."
        raise ValueError(msg)
    return cast("dict[str, object]", value)


def _require_positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        msg = f"ROI config {field} must be a positive integer."
        raise ValueError(msg)
    return value


def _normalize_hands(hands: tuple[str, ...]) -> tuple[str, ...]:
    """Validate and return a non-empty ordered hand selection."""
    if not hands or len(set(hands)) != len(hands) or any(hand not in HANDS for hand in hands):
        msg = f"Hands must be a non-empty unique subset of {HANDS}: {hands!r}"
        raise ValueError(msg)
    return hands


def load_patchcore_roi_config(
    path: Path,
    *,
    hands: tuple[str, ...] = HANDS,
    expected_views: tuple[str, ...] = VIEWS,
) -> tuple[int, int, dict[str, dict[str, ROI]], dict[str, object]]:
    """Load and strictly validate the selected-hand ROI configuration.

    Args:
        path (Path): Path to the ROI JSON configuration.

    Returns:
        tuple[int, int, dict[str, dict[str, ROI]], dict[str, object]]: Configured width, height, ROI mapping, and
            unmodified decoded JSON payload. ``expected_views`` keeps legacy callers on six views while strict
            eight-view runtimes opt in with :data:`CANONICAL_VIEWS`.

    Raises:
        ValueError: If the JSON payload does not match the required schema or contains an invalid ROI.
    """
    payload = _require_mapping(json.loads(path.read_text(encoding="utf-8")), field="root")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        msg = "ROI config schema_version must be 1."
        raise ValueError(msg)
    if payload.get("coordinate_system") != "pixel_xyxy_half_open":
        msg = "ROI config coordinate_system must be pixel_xyxy_half_open."
        raise ValueError(msg)

    image_size = _require_mapping(payload.get("image_size"), field="image_size")
    width = _require_positive_integer(image_size.get("width"), field="image_size.width")
    height = _require_positive_integer(image_size.get("height"), field="image_size.height")
    hands = _normalize_hands(hands)
    if not expected_views or len(set(expected_views)) != len(expected_views):
        msg = f"Expected views must be a non-empty unique tuple: {expected_views!r}"
        raise ValueError(msg)
    if "views" in payload:
        if hands != ("right",):
            msg = "A top-level views ROI config can only be used with hands=('right',)."
            raise ValueError(msg)
        configured_hands: dict[str, object] = {"right": {"views": payload["views"]}}
    else:
        configured_hands = _require_mapping(payload.get("hands"), field="hands")
        if set(configured_hands) != set(hands):
            msg = f"ROI config hands must be exactly {hands}."
            raise ValueError(msg)

    rois: dict[str, dict[str, ROI]] = {}
    for hand in hands:
        hand_payload = _require_mapping(configured_hands[hand], field=f"hands.{hand}")
        configured_views = _require_mapping(hand_payload.get("views"), field=f"hands.{hand}.views")
        if set(configured_views) != set(expected_views):
            msg = f"ROI config views for {hand} must be exactly {expected_views}."
            raise ValueError(msg)
        hand_rois: dict[str, ROI] = {}
        for view in expected_views:
            view_payload = _require_mapping(configured_views[view], field=f"hands.{hand}.views.{view}")
            values = view_payload.get("roi")
            valid_values = (
                isinstance(values, list)
                and len(values) == 4
                and all(isinstance(value, int) and not isinstance(value, bool) for value in values)
            )
            if not valid_values:
                msg = f"Invalid ROI for {hand}/{view}: {values!r}"
                raise ValueError(msg)
            x1, y1, x2, y2 = cast("list[int]", values)
            invalid_origin_or_extent = x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1
            if invalid_origin_or_extent or x2 > width or y2 > height:
                msg = f"ROI is outside configured image for {hand}/{view}: {values!r}"
                raise ValueError(msg)
            hand_rois[view] = (x1, y1, x2, y2)
        rois[hand] = hand_rois
    return width, height, rois, payload


def _filename_view(path: Path, hand: str, expected_views: tuple[str, ...] = VIEWS) -> str:
    filename = path.name
    for other_hand in HANDS:
        if other_hand != hand and filename.startswith(f"{other_hand}_"):
            msg = f"Filename hand mismatch for {path}: expected {hand}, found {other_hand}."
            raise ValueError(msg)
    for view in sorted(expected_views, key=len, reverse=True):
        if filename.startswith(f"{hand}_{view}_"):
            return view
    msg = f"Could not parse filename view for {path}; expected prefix {hand}_<view>_."
    raise ValueError(msg)


def _source_identity(
    path: Path,
    hand_root: Path,
    hand: str,
    expected_views: tuple[str, ...] = VIEWS,
) -> SourceImage:
    relative_path = path.relative_to(hand_root)
    parts = relative_path.parts
    if len(parts) < 5:
        msg = f"Invalid PatchCore source hierarchy: {path}"
        raise ValueError(msg)
    source_view, label = parts[:2]
    if source_view not in expected_views:
        msg = f"Unknown source view for {path}: {source_view}"
        raise ValueError(msg)
    if label == "defect":
        if len(parts) < 6 or parts[4] != "images":
            msg = f"Invalid defect hierarchy for {path}; expected defect/<type>/<session>/images."
            raise ValueError(msg)
        defect_type, session_id = parts[2:4]
    elif label in {"normal", "normal_test"}:
        if parts[3] != "images":
            msg = f"Invalid {label} hierarchy for {path}; expected {label}/<session>/images."
            raise ValueError(msg)
        defect_type, session_id = "", parts[2]
    else:
        msg = f"Unknown PatchCore label for {path}: {label}"
        raise ValueError(msg)
    _filename_view(path, hand, expected_views)  # Validate filename identity without changing directory routing.
    return SourceImage(
        source_path=path,
        hand=hand,
        source_view=source_view,
        resolved_view=source_view,
        label=label,
        defect_type=defect_type,
        session_id=session_id,
        relative_tail=Path(*parts[1:]),
        view_corrected=False,
    )


def discover_patchcore_images(
    dataset_root: Path,
    *,
    hands: tuple[str, ...] = HANDS,
    expected_views: tuple[str, ...] = VIEWS,
    excluded_session_ids: tuple[str, ...] = (),
) -> list[SourceImage]:
    """Discover and validate ZS32 PatchCore images for selected hands.

    The source directory is the authoritative view. Filenames are validated but never change ROI selection,
    output routing, or statistics. Returned records are sorted by source path and have unique destinations.

    Args:
        dataset_root (Path): Root containing the ``right`` and ``left`` source trees.

    Returns:
        list[SourceImage]: Stable source records with validated hand, view, label, and destination identities.

    Raises:
        FileNotFoundError: If a required hand or view directory is missing.
        ValueError: If required normal data is absent or an image identity or destination is invalid.
    """
    hands = _normalize_hands(hands)
    excluded_sessions = set(excluded_session_ids)
    records: list[SourceImage] = []
    normal_views: set[tuple[str, str]] = set()
    destinations: dict[tuple[str, str, Path], Path] = {}
    for hand in hands:
        hand_root = dataset_root / hand
        if not hand_root.is_dir():
            msg = f"Missing hand directory: {hand_root}"
            raise FileNotFoundError(msg)
        for view in expected_views:
            view_root = hand_root / view
            if not view_root.is_dir():
                msg = f"Missing view directory for {hand}/{view}: {view_root}"
                raise FileNotFoundError(msg)
            image_paths = sorted(
                (path for path in view_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
                key=lambda path: path.as_posix(),
            )
            for image_path in image_paths:
                record = _source_identity(image_path, hand_root, hand, expected_views)
                if record.session_id in excluded_sessions:
                    continue
                destination = (record.hand, record.resolved_view, record.relative_tail)
                if destination in destinations:
                    msg = f"PatchCore target collision: {destinations[destination]} and {record.source_path}"
                    raise ValueError(msg)
                destinations[destination] = record.source_path
                records.append(record)
                if record.label == "normal":
                    normal_views.add((record.hand, record.resolved_view))

    missing_normal = [(hand, view) for hand in hands for view in expected_views if (hand, view) not in normal_views]
    if missing_normal:
        msg = f"Missing normal image data for hand/view pairs: {missing_normal}"
        raise ValueError(msg)
    return sorted(records, key=lambda record: record.source_path.as_posix())


def select_patchcore_rois(
    repo_root: Path,
    dataset_root: Path,
    config_path: Path,
    preview_dir: Path,
    max_window_width: int = 1600,
    max_window_height: int = 1000,
    hands: tuple[str, ...] = HANDS,
    excluded_session_ids: tuple[str, ...] = (),
    expected_views: tuple[str, ...] = VIEWS,
) -> dict[str, object]:
    """Interactively select and atomically publish selected hand/view ROIs.

    Args:
        repo_root (Path): Repository root used to store relative reference paths.
        dataset_root (Path): Root containing the ``right`` and ``left`` source trees.
        config_path (Path): JSON configuration path to replace after all selections succeed.
        preview_dir (Path): Directory for full-resolution ROI overlay previews.
        max_window_width (int): Maximum selector window width.
        max_window_height (int): Maximum selector window height.

    Returns:
        dict[str, object]: The complete configuration written to ``config_path``.

    Raises:
        FileNotFoundError: If no readable normal reference exists for a hand/view.
        ValueError: If references have different sizes, lie outside the repository, or a selected ROI is invalid.
    """
    hands = _normalize_hands(hands)
    records = discover_patchcore_images(
        dataset_root,
        hands=hands,
        expected_views=expected_views,
        excluded_session_ids=excluded_session_ids,
    )
    references: dict[tuple[str, str], tuple[SourceImage, object]] = {}
    common_size: tuple[int, int] | None = None
    for hand in hands:
        for view in expected_views:
            candidates = (
                record
                for record in records
                if record.hand == hand and record.resolved_view == view and record.label == "normal"
            )
            for record in candidates:
                image = cv2.imread(str(record.source_path), cv2.IMREAD_COLOR)
                if image is not None:
                    break
            else:
                msg = f"No readable normal reference image for {hand}/{view}."
                raise FileNotFoundError(msg)
            height, width = image.shape[:2]
            if common_size is None:
                common_size = (width, height)
            elif (width, height) != common_size:
                msg = (
                    "Normal reference images must have the same size: "
                    f"expected {common_size}, found {(width, height)} for {record.source_path}."
                )
                raise ValueError(msg)
            references[(hand, view)] = (record, image)

    if common_size is None:  # pragma: no cover - discovery guarantees candidates for every pair
        msg = "No normal reference images were discovered."
        raise FileNotFoundError(msg)
    width, height = common_size
    existing_rois: dict[str, dict[str, ROI]] | None = None
    if config_path.exists():
        configured_width, configured_height, existing_rois, _ = load_patchcore_roi_config(
            config_path,
            hands=hands,
            expected_views=expected_views,
        )
        if (configured_width, configured_height) != common_size:
            msg = (
                "Existing ROI config dimensions do not match normal references: "
                f"configured {(configured_width, configured_height)}, references {common_size}."
            )
            raise ValueError(msg)

    hands_payload: dict[str, object] = {}
    for hand in hands:
        views_payload: dict[str, object] = {}
        for view in expected_views:
            record, image = references[(hand, view)]
            initial_roi = existing_rois[hand][view] if existing_rois is not None else None
            roi = select_roi(record.source_path, initial_roi, max_window_width, max_window_height)
            x1, y1, x2, y2 = roi
            if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > width or y2 > height:
                msg = f"Selected ROI is outside reference image for {hand}/{view}: {roi!r}"
                raise ValueError(msg)
            try:
                reference_image = record.source_path.relative_to(repo_root).as_posix()
            except ValueError as error:
                msg = f"Reference image must be inside repository root {repo_root}: {record.source_path}"
                raise ValueError(msg) from error
            save_overlay(preview_dir / f"{hand}_{view}_roi.png", image, roi)
            views_payload[view] = {"roi": list(roi), "reference_image": reference_image}
        hands_payload[hand] = {"views": views_payload}

    payload: dict[str, object] = {
        "schema_version": 1,
        "coordinate_system": "pixel_xyxy_half_open",
        "image_size": {"width": width, "height": height},
        "hands": hands_payload,
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = config_path.with_suffix(config_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary_path.replace(config_path)
    return payload


def _paths_overlap(first: Path, second: Path) -> bool:
    """Return whether either resolved path contains the other."""
    return first == second or first in second.parents or second in first.parents


def _validate_conversion_paths(
    repo_root: Path,
    dataset_root: Path,
    output_root: Path,
    hands: tuple[str, ...],
) -> None:
    """Validate the output boundary without changing any filesystem state."""
    safe_dataset_root = (repo_root / "dataset").resolve()
    try:
        dataset_relative = dataset_root.relative_to(safe_dataset_root)
    except ValueError as error:
        msg = f"Input dataset root must be under the repository dataset directory: {safe_dataset_root}"
        raise ValueError(msg) from error
    try:
        output_relative = output_root.relative_to(safe_dataset_root)
    except ValueError as error:
        msg = f"Output must be a safe child directory under {safe_dataset_root}: {output_root}"
        raise ValueError(msg) from error
    if not output_relative.parts:
        msg = f"Output must be a safe child directory under {safe_dataset_root}: {output_root}"
        raise ValueError(msg)

    if dataset_relative.parts and _paths_overlap(output_root, dataset_root):
        msg = f"Output must be separate from the input dataset tree: {output_root}"
        raise ValueError(msg)
    input_roots = ((dataset_root / hand).resolve() for hand in hands)
    if any(_paths_overlap(output_root, input_root) for input_root in input_roots):
        msg = f"Output must be separate from the right/left input trees: {output_root}"
        raise ValueError(msg)


def _preflight_conversion(
    records: list[SourceImage],
    dataset_root: Path,
    image_width: int,
    image_height: int,
    rois: dict[str, dict[str, ROI]],
) -> list[_PreparedImage]:
    """Read and validate every source while producing no output artifacts."""
    prepared: list[_PreparedImage] = []
    resolved_sources: set[Path] = set()
    for record in track(records, description="Checking images"):
        source_path = record.source_path.resolve()
        hand_root = (dataset_root / record.hand).resolve()
        try:
            source_path.relative_to(hand_root)
        except ValueError as error:
            msg = f"Source image escapes its {record.hand} input tree: {record.source_path}"
            raise ValueError(msg) from error
        if source_path in resolved_sources:
            msg = f"Duplicate source image after path resolution: {record.source_path}"
            raise ValueError(msg)
        resolved_sources.add(source_path)

        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            msg = f"Failed to read image: {record.source_path}"
            raise FileNotFoundError(msg)
        height, width = image.shape[:2]
        if (width, height) != (image_width, image_height):
            msg = (
                f"Image size mismatch for {record.source_path}: "
                f"found {width}x{height}, expected {image_width}x{image_height}."
            )
            raise ValueError(msg)
        roi = rois[record.hand][record.resolved_view]
        x1, y1, x2, y2 = roi
        crop_width, crop_height = x2 - x1, y2 - y1
        if crop_width <= 0 or crop_height <= 0 or x2 > width or y2 > height:
            msg = f"ROI is outside source image for {record.hand}/{record.resolved_view}: {roi!r}"
            raise ValueError(msg)
        prepared.append(
            _PreparedImage(
                source=record,
                output_relative=Path(record.hand, record.resolved_view) / record.relative_tail,
                roi=roi,
                source_width=width,
                source_height=height,
                crop_width=crop_width,
                crop_height=crop_height,
            ),
        )
    return prepared


def crop_patchcore_dataset(
    repo_root: Path,
    dataset_root: Path,
    output_root: Path,
    roi_config: Path,
    overwrite: bool = False,
    hands: tuple[str, ...] = HANDS,
    excluded_session_ids: tuple[str, ...] = (),
    expected_views: tuple[str, ...] = VIEWS,
) -> dict[str, object]:
    """Preflight and crop a PatchCore dataset into a separate transactional output tree.

    Args:
        repo_root (Path): Repository root containing the safe ``dataset`` boundary.
        dataset_root (Path): Root containing the ``right`` and ``left`` source trees.
        output_root (Path): Separate output directory below ``repo_root/dataset``.
        roi_config (Path): Strict selected-hand, eight-view JSON configuration.
        overwrite (bool): Whether an existing output may be transactionally replaced.

    Returns:
        dict[str, object]: Conversion paths and exact image statistics.

    Raises:
        FileExistsError: If output exists and overwrite is false.
        FileNotFoundError: If a source image cannot be read.
        ValueError: If paths, source identities, dimensions, or ROIs are invalid.
    """
    repo_root = repo_root.resolve()
    dataset_root = dataset_root.resolve()
    output_root = output_root.resolve()
    roi_config = roi_config.resolve()
    hands = _normalize_hands(hands)
    _validate_conversion_paths(repo_root, dataset_root, output_root, hands)
    if output_root.exists() and not overwrite:
        msg = f"Output already exists; pass overwrite=True to replace it: {output_root}"
        raise FileExistsError(msg)

    width, height, rois, config_payload = load_patchcore_roi_config(
        roi_config,
        hands=hands,
        expected_views=expected_views,
    )
    records = discover_patchcore_images(
        dataset_root,
        hands=hands,
        expected_views=expected_views,
        excluded_session_ids=excluded_session_ids,
    )
    prepared = _preflight_conversion(records, dataset_root, width, height, rois)

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.tmp-",
            dir=output_root.parent,
        ),
    )
    backup_root: Path | None = None
    cleanup_warning = ""
    retained_backup_path = ""
    manifest_rows: list[dict[str, str]] = []
    try:
        for item in track(prepared, description="Cropping images"):
            image = cv2.imread(str(item.source.source_path), cv2.IMREAD_COLOR)
            if image is None:
                msg = f"Failed to reread image during conversion: {item.source.source_path}"
                raise RuntimeError(msg)
            height_now, width_now = image.shape[:2]
            if (width_now, height_now) != (item.source_width, item.source_height):
                msg = f"Image changed after preflight: {item.source.source_path}"
                raise RuntimeError(msg)
            x1, y1, x2, y2 = item.roi
            crop = image[y1:y2, x1:x2]
            if crop.shape[:2] != (item.crop_height, item.crop_width):
                msg = f"Unexpected crop shape for {item.source.source_path}: {crop.shape[:2]}"
                raise RuntimeError(msg)

            temporary_output = temporary_root / item.output_relative
            temporary_output.parent.mkdir(parents=True, exist_ok=True)
            if temporary_output.suffix.lower() == ".png":
                written = cv2.imwrite(
                    str(temporary_output),
                    crop,
                    [cv2.IMWRITE_PNG_COMPRESSION, 1],
                )
            else:
                written = cv2.imwrite(str(temporary_output), crop)
            if not written:
                msg = f"Failed to write cropped image: {output_root / item.output_relative}"
                raise RuntimeError(msg)

            source = item.source
            manifest_rows.append(
                {
                    "source_path": source.source_path.relative_to(repo_root).as_posix(),
                    "output_path": (output_root / item.output_relative).relative_to(repo_root).as_posix(),
                    "hand": source.hand,
                    "source_view": source.source_view,
                    "resolved_view": source.resolved_view,
                    "view_corrected": str(source.view_corrected).lower(),
                    "label": source.label,
                    "defect_type": source.defect_type,
                    "session_id": source.session_id,
                    "roi_x1": str(x1),
                    "roi_y1": str(y1),
                    "roi_x2": str(x2),
                    "roi_y2": str(y2),
                    "source_width": str(item.source_width),
                    "source_height": str(item.source_height),
                    "crop_width": str(item.crop_width),
                    "crop_height": str(item.crop_height),
                },
            )

        by_hand_view: dict[str, dict[str, dict[str, object]]] = {
            hand: {
                view: {
                    "input": 0,
                    "output": 0,
                    "corrected_views": 0,
                    "labels": {},
                }
                for view in expected_views
            }
            for hand in hands
        }
        for item in prepared:
            source = item.source
            counts = by_hand_view[source.hand][source.resolved_view]
            counts["input"] = cast("int", counts["input"]) + 1
            counts["output"] = cast("int", counts["output"]) + 1
            counts["corrected_views"] = cast("int", counts["corrected_views"]) + int(source.view_corrected)
            labels = cast("dict[str, int]", counts["labels"])
            labels[source.label] = labels.get(source.label, 0) + 1

        corrected_views = sum(int(item.source.view_corrected) for item in prepared)
        summary: dict[str, object] = {
            "total_images": len(prepared),
            "output_images": len(manifest_rows),
            "corrected_views": corrected_views,
            "by_hand_view": by_hand_view,
        }
        manifest_path = temporary_root / "crop_manifest.csv"
        fieldnames = [
            "source_path",
            "output_path",
            "hand",
            "source_view",
            "resolved_view",
            "view_corrected",
            "label",
            "defect_type",
            "session_id",
            "roi_x1",
            "roi_y1",
            "roi_x2",
            "roi_y2",
            "source_width",
            "source_height",
            "crop_width",
            "crop_height",
        ]
        with manifest_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)
        (temporary_root / "roi_config.json").write_text(
            json.dumps(config_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (temporary_root / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        if output_root.exists():
            backup_root = output_root.parent / f".{output_root.name}.backup-{uuid4().hex}"
            output_root.replace(backup_root)
        try:
            temporary_root.replace(output_root)
        except BaseException:
            if backup_root is not None and backup_root.exists():
                backup_root.replace(output_root)
                backup_root = None
            raise
        if backup_root is not None:
            try:
                shutil.rmtree(backup_root)
            except OSError as error:
                retained_backup_path = str(backup_root)
                cleanup_warning = f"Failed to remove old output backup {backup_root}: {error}"
            else:
                backup_root = None
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)

    return {
        "output_root": output_root,
        "manifest_path": output_root / "crop_manifest.csv",
        "cleanup_warning": cleanup_warning,
        "backup_path": retained_backup_path,
        **summary,
    }
