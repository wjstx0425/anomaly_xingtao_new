# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Multi-view manifest parsing and validation for inspection captures."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MANIFEST_FIELDNAMES = [
    "part_id",
    "side",
    "view",
    "image_path",
    "label",
    "defect_type",
    "slot_id",
    "group_id",
    "notes",
]
DEFAULT_FILENAME_REGEX = (
    r"(?P<part_id>.+?)_(?P<side>top|bottom|front|back)_"
    r"(?P<view>uniform|darkfield|left_bar|right_bar|brightfield)"
    r"(?:_slot(?P<slot_id>\d+|slot\d+))?"
)


@dataclass(frozen=True)
class MultiViewImageRecord:
    """One captured image belonging to a multi-view part inspection."""

    part_id: str
    side: str
    view: str
    image_path: Path
    label: str | None = None
    defect_type: str | None = None
    slot_id: str | None = None
    group_id: str | None = None


@dataclass(frozen=True)
class MultiViewPartRecord:
    """All available images for one physical part."""

    part_id: str
    images: tuple[MultiViewImageRecord, ...]


def _clean_text(value: Any) -> str | None:
    """Return a stripped string, or ``None`` for empty values."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_sequence(value: Any) -> tuple[str, ...]:
    """Normalize a config value to a tuple of strings."""
    if value is None:
        return ()
    if isinstance(value, str):
        text = _clean_text(value)
        return () if text is None else (text,)
    if isinstance(value, Sequence):
        return tuple(text for item in value if (text := _clean_text(item)) is not None)
    text = _clean_text(value)
    return () if text is None else (text,)


def _normalize_slot(value: str | None) -> str | None:
    """Normalize slot ids to ``slotNN`` when possible."""
    if value is None:
        return None
    match = re.search(r"(\d+)", value)
    if match is None:
        return value
    return f"slot{int(match.group(1)):02d}"


def _resolve_image_path(value: str, *, base_dir: Path, input_root: Path) -> Path:
    """Resolve manifest image paths relative to CSV dir first, then input root."""
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return input_root / path


def _record_from_row(row: Mapping[str, str], *, base_dir: Path, input_root: Path) -> MultiViewImageRecord:
    """Build an image record from one explicit manifest CSV row."""
    part_id = _clean_text(row.get("part_id"))
    side = _clean_text(row.get("side"))
    view = _clean_text(row.get("view"))
    image_path_text = _clean_text(row.get("image_path"))
    if part_id is None or side is None or view is None or image_path_text is None:
        msg = f"Manifest row must include part_id, side, view, and image_path: {row}"
        raise ValueError(msg)
    return MultiViewImageRecord(
        part_id=part_id,
        side=side,
        view=view,
        image_path=_resolve_image_path(image_path_text, base_dir=base_dir, input_root=input_root),
        label=_clean_text(row.get("label")),
        defect_type=_clean_text(row.get("defect_type")),
        slot_id=_normalize_slot(_clean_text(row.get("slot_id"))),
        group_id=_clean_text(row.get("group_id")),
    )


def _group_records(records: Sequence[MultiViewImageRecord]) -> list[MultiViewPartRecord]:
    """Group image records by part id in stable order."""
    grouped: dict[str, list[MultiViewImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.part_id].append(record)
    return [
        MultiViewPartRecord(
            part_id=part_id,
            images=tuple(sorted(images, key=lambda item: (item.side, item.view, str(item.image_path)))),
        )
        for part_id, images in sorted(grouped.items())
    ]


def _filename_regex(config: Mapping[str, Any]) -> str:
    """Return configured filename regex or a conservative default."""
    manifest_config = config.get("manifest", {})
    if isinstance(manifest_config, Mapping):
        value = _clean_text(manifest_config.get("filename_regex"))
        if value is not None:
            return value
    value = _clean_text(config.get("filename_regex"))
    return DEFAULT_FILENAME_REGEX if value is None else value


def _record_from_match(path: Path, match: re.Match[str]) -> MultiViewImageRecord:
    """Build an image record from a regex match."""
    groups = match.groupdict()
    part_id = _clean_text(groups.get("part_id")) or path.stem
    side = _clean_text(groups.get("side"))
    view = _clean_text(groups.get("view"))
    if side is None or view is None:
        msg = f"Filename regex must provide side and view groups for {path}"
        raise ValueError(msg)
    return MultiViewImageRecord(
        part_id=part_id,
        side=side,
        view=view,
        image_path=path,
        label=_clean_text(groups.get("label")),
        defect_type=_clean_text(groups.get("defect_type")),
        slot_id=_normalize_slot(_clean_text(groups.get("slot_id"))),
        group_id=_clean_text(groups.get("group_id")),
    )


def _iter_image_paths(input_root: Path) -> list[Path]:
    """Return image files under an input root."""
    if input_root.is_file() and input_root.suffix.lower() in IMAGE_EXTENSIONS:
        return [input_root]
    return sorted(path for path in input_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def load_multiview_manifest(
    input_root: Path,
    config: Mapping[str, Any],
    *,
    manifest_csv: Path | None = None,
) -> list[MultiViewPartRecord]:
    """Load a multi-view manifest from explicit CSV or filename parsing."""
    if manifest_csv is not None:
        with manifest_csv.open(newline="", encoding="utf-8") as file:
            records = [
                _record_from_row(row, base_dir=manifest_csv.parent, input_root=input_root)
                for row in csv.DictReader(file)
            ]
        return _group_records(records)

    pattern = re.compile(_filename_regex(config))
    records = []
    for image_path in _iter_image_paths(input_root):
        match = pattern.search(image_path.name)
        if match is None:
            continue
        records.append(_record_from_match(image_path, match))
    return _group_records(records)


def required_view_keys(config: Mapping[str, Any]) -> set[str]:
    """Return required side:view pairs from profile or fusion-style config."""
    ok_requires = config.get("ok_requires", {})
    manifest_config = config.get("manifest", {})
    required_keys = set(_string_sequence(config.get("required_view_keys")))
    if isinstance(manifest_config, Mapping):
        required_keys.update(_string_sequence(manifest_config.get("required_view_keys")))
        sides = _string_sequence(manifest_config.get("required_sides"))
        views = _string_sequence(manifest_config.get("required_views"))
    else:
        sides = ()
        views = ()
    if isinstance(ok_requires, Mapping):
        required_keys.update(_string_sequence(ok_requires.get("required_view_keys")))
        sides = sides or _string_sequence(ok_requires.get("required_sides"))
        views = views or _string_sequence(ok_requires.get("required_views"))
    sides = sides or _string_sequence(config.get("required_sides"))
    views = views or _string_sequence(config.get("required_views"))
    for value in views:
        if ":" in value:
            required_keys.add(value)
    plain_views = tuple(value for value in views if ":" not in value)
    for side in sides:
        for view in plain_views:
            required_keys.add(f"{side}:{view}")
    return required_keys


def validate_required_views(part_record: MultiViewPartRecord, config: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Return whether a part has all configured required side/view pairs."""
    required = required_view_keys(config)
    if not required:
        return True, []
    observed = {f"{image.side}:{image.view}" for image in part_record.images}
    missing = sorted(required - observed)
    return not missing, missing


def write_multiview_manifest(records: Sequence[MultiViewPartRecord], output_csv: Path) -> None:
    """Write grouped records to the MVP-5 manifest CSV schema."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        for part in records:
            for image in part.images:
                writer.writerow(
                    {
                        "part_id": image.part_id,
                        "side": image.side,
                        "view": image.view,
                        "image_path": str(image.image_path),
                        "label": image.label or "",
                        "defect_type": image.defect_type or "",
                        "slot_id": image.slot_id or "",
                        "group_id": image.group_id or "",
                        "notes": "",
                    },
                )
