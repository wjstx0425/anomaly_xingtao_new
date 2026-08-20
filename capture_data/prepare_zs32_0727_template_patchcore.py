# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Prepare and validate the ZS32 0727 Template/PatchCore v14 data release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

VIEWS = (
    "front",
    "front_left",
    "front_right",
    "front_secondary",
    "back",
    "back_left",
    "back_right",
    "back_secondary",
)
ROLES = ("train", "model_val", "calibration", "final_test")
GROUP_PATTERN = re.compile(r"(group\d{3})(?:\D|$)", re.IGNORECASE)
DUPLICATE_DEFECT_SESSION = "zs32_right_deform_20260714_201404_438451409"
EXPECTED_NORMAL_PARTS = 104
EXPECTED_DEFECT_PARTS = 23
EXPECTED_DEFECT_TYPES = {"deform": 12, "less": 2, "others": 9}
NORMAL_ROLE_COUNTS = {"train": 62, "model_val": 16, "calibration": 13, "final_test": 13}
DEFECT_TYPE_ROLE_COUNTS = {
    "deform": {"calibration": 6, "model_val": 3, "final_test": 3},
    "less": {"calibration": 1, "model_val": 0, "final_test": 1},
    "others": {"calibration": 4, "model_val": 3, "final_test": 2},
}

SPLIT_FIELDS = (
    "physical_part_id",
    "session_id",
    "group_id",
    "label",
    "defect_type",
    "role",
    "seed",
)
TEMPLATE_FIELDS = (
    "source_origin",
    "part_id",
    "physical_part_id",
    "hand",
    "view",
    "resolved_view",
    "label",
    "defect_type",
    "role",
    "split",
    "session_id",
    "group_id",
    "image_path",
    "source_path",
    "content_sha256",
)
PATCHCORE_FIELDS = (
    "source_origin",
    "physical_part_id",
    "hand",
    "view",
    "resolved_view",
    "label",
    "defect_type",
    "session_id",
    "group_id",
    "release_role",
    "target_bucket",
    "source_path",
    "target_path",
    "content_sha256",
)
TRAINER_FIELDS = (
    "source_path",
    "output_path",
    "hand",
    "source_view",
    "resolved_view",
    "view_corrected",
    "label",
    "defect_type",
    "session_id",
    "group_id",
    "physical_part_id",
    "release_split",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return [
            {str(key): "" if value is None else str(value).strip() for key, value in row.items()}
            for row in reader
        ]


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required(row: dict[str, str], field: str, *, context: str) -> str:
    value = row.get(field, "").strip()
    if not value:
        raise ValueError(f"{context} is missing required field {field!r}")
    return value


def _safe_path_segment(value: str, field: str, *, context: str) -> str:
    """Reject identifiers that could change the intended release directory."""
    if (
        not value
        or value in {".", ".."}
        or Path(value).is_absolute()
        or Path(value).name != value
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"{context} has unsafe {field}: {value!r}")
    return value


def _lexical_path(path: Path) -> Path:
    """Normalize a path without following release symlinks."""
    return Path(os.path.abspath(os.fspath(path)))


def _path_below(root: Path, path: Path) -> bool:
    root = _lexical_path(root)
    path = _lexical_path(path)
    return path != root and root in path.parents


def _asset(row: dict[str, str], manifest: Path, *, context: str) -> Path:
    value = row.get("output_path", "").strip() or row.get("image_path", "").strip()
    if not value:
        value = _required(row, "source_path", context=context)
    path = Path(value)
    candidates = (path,) if path.is_absolute() else (Path.cwd() / path, manifest.parent / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"{context} image does not exist: {value}")


def _group_id(row: dict[str, str], path: Path, *, context: str) -> str:
    for value in (row.get("group_id", ""), path.name, row.get("source_path", "")):
        match = GROUP_PATTERN.search(value)
        if match:
            return match.group(1).lower()
    raise ValueError(f"{context} has no groupNNN identity")


def _load_inventory(
    manifest: Path,
    *,
    label: str,
    expected_parts: int,
    excluded_sessions: set[str] | None = None,
) -> list[dict[str, str]]:
    excluded_sessions = excluded_sessions or set()
    prepared: list[dict[str, str]] = []
    views_by_part: dict[str, set[str]] = defaultdict(set)
    types_by_part: dict[str, set[str]] = defaultdict(set)
    row_keys: set[tuple[str, str]] = set()
    hashes: dict[str, str] = {}
    for number, row in enumerate(_read_csv(manifest), start=2):
        context = f"{manifest}:row {number}"
        if row.get("label", "").lower() != label:
            continue
        session_id = _safe_path_segment(
            _required(row, "session_id", context=context),
            "session_id",
            context=context,
        )
        if session_id in excluded_sessions:
            continue
        if _required(row, "hand", context=context).lower() != "right":
            raise ValueError(f"{context} must contain right-hand data")
        view = (row.get("resolved_view", "") or row.get("source_view", "") or row.get("view", "")).lower()
        if view not in VIEWS:
            raise ValueError(f"{context} has unsupported view {view!r}")
        image = _asset(row, manifest, context=context)
        group_id = _group_id(row, image, context=context)
        part_id = f"{session_id}:{group_id}"
        if (part_id, view) in row_keys:
            raise ValueError(f"{context} duplicates {part_id}/{view}")
        row_keys.add((part_id, view))
        defect_type = row.get("defect_type", "").lower()
        if label == "defect" and defect_type not in EXPECTED_DEFECT_TYPES:
            raise ValueError(f"{context} has unsupported defect_type {defect_type!r}")
        if label == "normal" and defect_type:
            raise ValueError(f"{context} normal row must not have defect_type")
        content_sha256 = _sha256(image)
        previous = hashes.setdefault(content_sha256, context)
        if previous != context:
            raise ValueError(f"{context} duplicates encoded image content from {previous}")
        views_by_part[part_id].add(view)
        types_by_part[part_id].add(defect_type)
        prepared.append(
            {
                "physical_part_id": part_id,
                "session_id": session_id,
                "group_id": group_id,
                "hand": "right",
                "view": view,
                "label": label,
                "defect_type": defect_type,
                "source_path": str(image),
                "content_sha256": content_sha256,
            }
        )
    if len(views_by_part) != expected_parts:
        raise ValueError(f"expected {expected_parts} {label} parts, got {len(views_by_part)}")
    incomplete = {part: sorted(views) for part, views in views_by_part.items() if views != set(VIEWS)}
    if incomplete:
        raise ValueError(f"{label} parts are not exact eight-view sets: {incomplete}")
    if any(len(values) != 1 for values in types_by_part.values()):
        raise ValueError(f"{label} part has inconsistent defect type")
    return sorted(prepared, key=lambda row: (row["physical_part_id"], VIEWS.index(row["view"])))


def _assign_normal_roles(parts: list[str], seed: int) -> dict[str, str]:
    if len(parts) != sum(NORMAL_ROLE_COUNTS.values()):
        raise ValueError("normal role counts do not cover all normal parts")
    shuffled = sorted(parts)
    random.Random(seed).shuffle(shuffled)
    result: dict[str, str] = {}
    offset = 0
    for role in ROLES:
        count = NORMAL_ROLE_COUNTS[role]
        result.update({part: role for part in shuffled[offset : offset + count]})
        offset += count
    return result


def _assign_defect_roles(rows: list[dict[str, str]], seed: int) -> dict[str, str]:
    type_by_part = {row["physical_part_id"]: row["defect_type"] for row in rows}
    counts = Counter(type_by_part.values())
    if counts != Counter(EXPECTED_DEFECT_TYPES):
        raise ValueError(f"unexpected unique defect type counts: {dict(counts)}")
    result: dict[str, str] = {}
    for type_index, defect_type in enumerate(sorted(EXPECTED_DEFECT_TYPES)):
        parts = sorted(part for part, value in type_by_part.items() if value == defect_type)
        random.Random(seed + type_index + 1).shuffle(parts)
        offset = 0
        for role in ("calibration", "model_val", "final_test"):
            count = DEFECT_TYPE_ROLE_COUNTS[defect_type][role]
            result.update({part: role for part in parts[offset : offset + count]})
            offset += count
        if offset != len(parts):
            raise ValueError(f"defect role counts do not cover {defect_type}")
    return result


def _symlink(stage_root: Path, relative: Path, source: Path) -> None:
    if relative.is_absolute():
        raise ValueError(f"release path must be relative: {relative}")
    destination = _lexical_path(stage_root / relative)
    if not _path_below(stage_root, destination):
        raise ValueError(f"release path escapes staging root: {relative}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.symlink_to(source.resolve())


def _template_relative(row: dict[str, str], role: str) -> Path:
    label_tail = (
        Path("defect", row["defect_type"])
        if row["label"] == "defect"
        else Path("normal")
    )
    return (
        Path("template", role, "right", row["view"])
        / label_tail
        / row["session_id"]
        / "images"
        / Path(row["source_path"]).name
    )


def _patchcore_route(row: dict[str, str], role: str) -> tuple[str, Path, str]:
    if row["label"] == "normal" and role == "train":
        bucket, label = "normal", "normal"
        relative = Path("patchcore", "right", row["view"], label)
    elif row["label"] == "normal" and role == "calibration":
        bucket, label = "normal_test", "normal_test"
        relative = Path("patchcore", "right", row["view"], label)
    elif row["label"] == "defect" and role == "calibration":
        bucket, label = "defect", "defect"
        relative = Path("patchcore", "right", row["view"], label, row["defect_type"])
    else:
        bucket, label = f"holdout/{role}", row["label"]
        relative = Path("holdout", "patchcore", role, "right", row["view"], label)
        if row["label"] == "defect":
            relative /= row["defect_type"]
    relative /= Path(row["session_id"], "images", Path(row["source_path"]).name)
    return bucket, relative, label


def prepare_release(
    normal_crop_manifest: Path,
    defect_crop_manifest: Path,
    output_root: Path,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Build a new immutable Template/PatchCore release."""
    normal_crop_manifest = Path(normal_crop_manifest).resolve()
    defect_crop_manifest = Path(defect_crop_manifest).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    normal_roi = normal_crop_manifest.with_name("roi_config.json")
    defect_roi = defect_crop_manifest.with_name("roi_config.json")
    if not normal_roi.is_file() or not defect_roi.is_file():
        raise FileNotFoundError("both crop manifests require an adjacent roi_config.json")
    if normal_roi.read_bytes() != defect_roi.read_bytes():
        raise ValueError("normal and defect ROI configs differ")

    normals = _load_inventory(
        normal_crop_manifest,
        label="normal",
        expected_parts=EXPECTED_NORMAL_PARTS,
    )
    defects = _load_inventory(
        defect_crop_manifest,
        label="defect",
        expected_parts=EXPECTED_DEFECT_PARTS,
        excluded_sessions={DUPLICATE_DEFECT_SESSION},
    )
    all_hashes = [row["content_sha256"] for row in (*normals, *defects)]
    if len(set(all_hashes)) != len(all_hashes):
        raise ValueError("retained normal/defect inventory contains duplicate encoded images")
    normal_roles = _assign_normal_roles(
        sorted({row["physical_part_id"] for row in normals}),
        seed,
    )
    defect_roles = _assign_defect_roles(defects, seed)
    roles = {**normal_roles, **defect_roles}

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent))
    try:
        split_rows: list[dict[str, Any]] = []
        by_part = {row["physical_part_id"]: row for row in (*normals, *defects)}
        for part_id, role in sorted(roles.items()):
            row = by_part[part_id]
            split_rows.append(
                {
                    "physical_part_id": part_id,
                    "session_id": row["session_id"],
                    "group_id": row["group_id"],
                    "label": row["label"],
                    "defect_type": row["defect_type"],
                    "role": role,
                    "seed": seed,
                }
            )
        _write_csv(staging / "physical_part_splits.csv", SPLIT_FIELDS, split_rows)

        template_rows: list[dict[str, Any]] = []
        patchcore_rows: list[dict[str, Any]] = []
        trainer_rows: list[dict[str, Any]] = []
        for row in (*normals, *defects):
            role = roles[row["physical_part_id"]]
            source = Path(row["source_path"])
            template_relative = _template_relative(row, role)
            _symlink(staging, template_relative, source)
            template_rows.append(
                {
                    "source_origin": "zs32_0727" if row["label"] == "normal" else "zs32_all_unique_defect",
                    "part_id": row["physical_part_id"],
                    "physical_part_id": row["physical_part_id"],
                    "hand": "right",
                    "view": row["view"],
                    "resolved_view": row["view"],
                    "label": row["label"],
                    "defect_type": row["defect_type"],
                    "role": role,
                    "split": role,
                    "session_id": row["session_id"],
                    "group_id": row["group_id"],
                    "image_path": str(output_root / template_relative),
                    "source_path": row["source_path"],
                    "content_sha256": row["content_sha256"],
                }
            )

            bucket, patchcore_relative, trainer_label = _patchcore_route(row, role)
            _symlink(staging, patchcore_relative, source)
            target_path = output_root / patchcore_relative
            patchcore_rows.append(
                {
                    "source_origin": "zs32_0727" if row["label"] == "normal" else "zs32_all_unique_defect",
                    "physical_part_id": row["physical_part_id"],
                    "hand": "right",
                    "view": row["view"],
                    "resolved_view": row["view"],
                    "label": row["label"],
                    "defect_type": row["defect_type"],
                    "session_id": row["session_id"],
                    "group_id": row["group_id"],
                    "release_role": role,
                    "target_bucket": bucket,
                    "source_path": row["source_path"],
                    "target_path": str(target_path),
                    "content_sha256": row["content_sha256"],
                }
            )
            if not bucket.startswith("holdout/"):
                trainer_rows.append(
                    {
                        "source_path": row["source_path"],
                        "output_path": str(target_path),
                        "hand": "right",
                        "source_view": row["view"],
                        "resolved_view": row["view"],
                        "view_corrected": "false",
                        "label": trainer_label,
                        "defect_type": row["defect_type"],
                        "session_id": row["session_id"],
                        "group_id": row["group_id"],
                        "physical_part_id": row["physical_part_id"],
                        "release_split": role,
                    }
                )
        _write_csv(staging / "template_manifest.csv", TEMPLATE_FIELDS, template_rows)
        _write_csv(staging / "patchcore_manifest.csv", PATCHCORE_FIELDS, patchcore_rows)
        _write_csv(staging / "patchcore" / "crop_manifest.csv", TRAINER_FIELDS, trainer_rows)
        shutil.copy2(normal_roi, staging / "roi_config.json")
        shutil.copy2(normal_roi, staging / "patchcore" / "roi_config.json")

        summary = {
            "schema_version": 1,
            "seed": seed,
            "normal_parts": EXPECTED_NORMAL_PARTS,
            "defect_parts": EXPECTED_DEFECT_PARTS,
            "normal_images": len(normals),
            "defect_images": len(defects),
            "template_rows": len(template_rows),
            "patchcore_rows": len(patchcore_rows),
            "patchcore_trainer_rows": len(trainer_rows),
            "excluded_defect_sessions": [DUPLICATE_DEFECT_SESSION],
            "normal_role_counts": NORMAL_ROLE_COUNTS,
            "defect_role_counts": dict(Counter(defect_roles.values())),
            "defect_type_counts": dict(
                Counter(
                    row["defect_type"]
                    for row in by_part.values()
                    if row["label"] == "defect"
                )
            ),
            "roi_config_sha256": _sha256(normal_roi),
        }
        (staging / "release_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        staging.replace(output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    try:
        validate_release(output_root)
    except BaseException:
        shutil.rmtree(output_root)
        raise
    return summary


def validate_release(output_root: Path) -> dict[str, Any]:
    """Independently validate a prepared release."""
    root = Path(output_root).resolve()
    summary = json.loads((root / "release_summary.json").read_text(encoding="utf-8"))
    splits = _read_csv(root / "physical_part_splits.csv")
    template = _read_csv(root / "template_manifest.csv")
    patchcore = _read_csv(root / "patchcore_manifest.csv")
    trainer = _read_csv(root / "patchcore" / "crop_manifest.csv")
    if len(splits) != EXPECTED_NORMAL_PARTS + EXPECTED_DEFECT_PARTS:
        raise ValueError("physical_part_splits.csv has the wrong part count")
    expected_roles = {
        ("normal", "train"): 62,
        ("normal", "model_val"): 16,
        ("normal", "calibration"): 13,
        ("normal", "final_test"): 13,
        ("defect", "model_val"): 6,
        ("defect", "calibration"): 11,
        ("defect", "final_test"): 6,
    }
    if Counter((row["label"], row["role"]) for row in splits) != Counter(expected_roles):
        raise ValueError("physical part roles do not match the approved split")
    if any(row["session_id"] == DUPLICATE_DEFECT_SESSION for row in splits):
        raise ValueError("duplicate defect session leaked into the release")
    split_by_part: dict[str, dict[str, str]] = {}
    for row in splits:
        part_id = row["physical_part_id"]
        if part_id in split_by_part:
            raise ValueError(f"duplicate physical part split: {part_id}")
        _safe_path_segment(row["session_id"], "session_id", context=part_id)
        if part_id != f"{row['session_id']}:{row['group_id']}":
            raise ValueError(f"physical part identity mismatch: {part_id}")
        split_by_part[part_id] = row
    actual_defect_type_roles = Counter(
        (row["defect_type"], row["role"])
        for row in splits
        if row["label"] == "defect"
    )
    expected_defect_type_roles = Counter(
        {
            (defect_type, role): count
            for defect_type, role_counts in DEFECT_TYPE_ROLE_COUNTS.items()
            for role, count in role_counts.items()
            if count
        }
    )
    if actual_defect_type_roles != expected_defect_type_roles:
        raise ValueError("defect type roles do not match the approved split")
    if len(template) != (EXPECTED_NORMAL_PARTS + EXPECTED_DEFECT_PARTS) * len(VIEWS):
        raise ValueError("template_manifest.csv has the wrong row count")
    views_by_part: dict[str, set[str]] = defaultdict(set)
    roles_by_part: dict[str, set[str]] = defaultdict(set)
    hashes: set[str] = set()
    template_keys: set[tuple[str, str]] = set()
    template_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in template:
        part_id = row["physical_part_id"]
        split = split_by_part.get(part_id)
        if split is None:
            raise ValueError(f"Template references unknown physical part: {part_id}")
        if (part_id, row["view"]) in template_keys:
            raise ValueError(f"duplicate Template row: {part_id}/{row['view']}")
        key = (part_id, row["view"])
        template_keys.add(key)
        template_by_key[key] = row
        if row["view"] not in VIEWS or row["resolved_view"] != row["view"]:
            raise ValueError(f"invalid Template view: {part_id}/{row['view']}")
        for field, split_field in (
            ("label", "label"),
            ("defect_type", "defect_type"),
            ("role", "role"),
            ("split", "role"),
            ("session_id", "session_id"),
            ("group_id", "group_id"),
        ):
            if row[field] != split[split_field]:
                raise ValueError(f"Template {field} mismatch: {part_id}/{row['view']}")
        if row["part_id"] != part_id or row["hand"] != "right":
            raise ValueError(f"Template identity mismatch: {part_id}/{row['view']}")
        expected_origin = "zs32_0727" if row["label"] == "normal" else "zs32_all_unique_defect"
        if row["source_origin"] != expected_origin:
            raise ValueError(f"Template provenance mismatch: {part_id}/{row['view']}")
        expected_path = root / _template_relative(row, split["role"])
        path = _lexical_path(Path(row["image_path"]))
        if path != _lexical_path(expected_path) or not _path_below(root / "template", path):
            raise ValueError(f"Template route mismatch: {part_id}/{row['view']}")
        if not path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"invalid Template release link: {path}")
        source_path = Path(row["source_path"])
        if not source_path.is_file() or path.resolve() != source_path.resolve():
            raise ValueError(f"Template source mismatch: {part_id}/{row['view']}")
        if _sha256(path) != row["content_sha256"]:
            raise ValueError(f"Template content hash mismatch: {path}")
        if row["content_sha256"] in hashes:
            raise ValueError(f"duplicate Template image content: {path}")
        hashes.add(row["content_sha256"])
        views_by_part[row["physical_part_id"]].add(row["view"])
        roles_by_part[row["physical_part_id"]].add(row["split"])
    if any(views != set(VIEWS) for views in views_by_part.values()):
        raise ValueError("Template manifest contains an incomplete part")
    if set(views_by_part) != set(split_by_part):
        raise ValueError("Template manifest does not cover every physical part")
    if any(len(values) != 1 for values in roles_by_part.values()):
        raise ValueError("Template physical part leaks across roles")
    if len(patchcore) != len(template):
        raise ValueError("patchcore_manifest.csv does not cover the complete release")
    patchcore_keys: set[tuple[str, str]] = set()
    active_patchcore: dict[str, dict[str, str]] = {}
    patchcore_route_counts: Counter[tuple[str, str, str]] = Counter()
    for row in patchcore:
        part_id = row["physical_part_id"]
        split = split_by_part.get(part_id)
        if split is None:
            raise ValueError(f"PatchCore references unknown physical part: {part_id}")
        key = (part_id, row["view"])
        if key in patchcore_keys:
            raise ValueError(f"duplicate PatchCore row: {part_id}/{row['view']}")
        patchcore_keys.add(key)
        if row["view"] not in VIEWS or row["resolved_view"] != row["view"]:
            raise ValueError(f"invalid PatchCore view: {part_id}/{row['view']}")
        for field, split_field in (
            ("label", "label"),
            ("defect_type", "defect_type"),
            ("release_role", "role"),
            ("session_id", "session_id"),
            ("group_id", "group_id"),
        ):
            if row[field] != split[split_field]:
                raise ValueError(f"PatchCore {field} mismatch: {part_id}/{row['view']}")
        template_row = template_by_key.get(key)
        if template_row is None or any(
            row[field] != template_row[field]
            for field in ("source_origin", "source_path", "content_sha256")
        ):
            raise ValueError(f"PatchCore provenance mismatch: {part_id}/{row['view']}")
        expected_bucket, relative, trainer_label = _patchcore_route(row, split["role"])
        if row["target_bucket"] != expected_bucket:
            raise ValueError(f"PatchCore bucket mismatch: {part_id}/{row['view']}")
        path = _lexical_path(Path(row["target_path"]))
        if path != _lexical_path(root / relative) or not _path_below(root, path):
            raise ValueError(f"PatchCore route mismatch: {part_id}/{row['view']}")
        if not path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"invalid PatchCore release link: {path}")
        source_path = Path(row["source_path"])
        if not source_path.is_file() or path.resolve() != source_path.resolve():
            raise ValueError(f"PatchCore source mismatch: {part_id}/{row['view']}")
        if _sha256(path) != row["content_sha256"]:
            raise ValueError(f"PatchCore content hash mismatch: {path}")
        patchcore_route_counts[(row["label"], split["role"], expected_bucket)] += 1
        if not expected_bucket.startswith("holdout/"):
            active_patchcore[str(path)] = {
                **row,
                "trainer_label": trainer_label,
            }
    if patchcore_keys != template_keys:
        raise ValueError("PatchCore manifest does not cover the Template inventory")
    expected_patchcore_route_counts = Counter(
        {
            ("normal", "train", "normal"): 62 * len(VIEWS),
            ("normal", "model_val", "holdout/model_val"): 16 * len(VIEWS),
            ("normal", "calibration", "normal_test"): 13 * len(VIEWS),
            ("normal", "final_test", "holdout/final_test"): 13 * len(VIEWS),
            ("defect", "model_val", "holdout/model_val"): 6 * len(VIEWS),
            ("defect", "calibration", "defect"): 11 * len(VIEWS),
            ("defect", "final_test", "holdout/final_test"): 6 * len(VIEWS),
        }
    )
    if patchcore_route_counts != expected_patchcore_route_counts:
        raise ValueError("PatchCore routes do not match the approved split")
    if len(trainer) != (62 + 13 + 11) * len(VIEWS):
        raise ValueError("PatchCore trainer manifest has the wrong row count")
    trainer_paths: set[str] = set()
    per_view_labels: Counter[tuple[str, str]] = Counter()
    for row in trainer:
        path = _lexical_path(Path(row["output_path"]))
        path_key = str(path)
        if path_key in trainer_paths:
            raise ValueError(f"duplicate PatchCore trainer row: {path}")
        trainer_paths.add(path_key)
        expected = active_patchcore.get(path_key)
        if expected is None:
            raise ValueError(f"PatchCore trainer route is not active: {path}")
        for field in ("source_path", "physical_part_id", "session_id", "group_id", "defect_type"):
            if row[field] != expected[field]:
                raise ValueError(f"PatchCore trainer {field} mismatch: {path}")
        if (
            row["resolved_view"] != expected["view"]
            or row["source_view"] != expected["view"]
            or row["release_split"] != expected["release_role"]
            or row["label"] != expected["trainer_label"]
            or row["hand"] != "right"
            or row["view_corrected"] != "false"
        ):
            raise ValueError(f"PatchCore trainer metadata mismatch: {path}")
        if not path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"invalid PatchCore release link: {path}")
        per_view_labels[(row["resolved_view"], row["label"])] += 1
    if trainer_paths != set(active_patchcore):
        raise ValueError("PatchCore trainer manifest does not cover every active route")
    expected_per_view_labels = Counter(
        {
            (view, label): count
            for view in VIEWS
            for label, count in (("normal", 62), ("normal_test", 13), ("defect", 11))
        }
    )
    if per_view_labels != expected_per_view_labels:
        raise ValueError("PatchCore trainer per-view counts are wrong")
    if summary["roi_config_sha256"] != _sha256(root / "roi_config.json"):
        raise ValueError("release ROI config hash mismatch")
    if summary["roi_config_sha256"] != _sha256(root / "patchcore" / "roi_config.json"):
        raise ValueError("PatchCore ROI config hash mismatch")
    split_seeds = {int(row["seed"]) for row in splits}
    if len(split_seeds) != 1:
        raise ValueError("physical part splits contain inconsistent seeds")
    expected_summary = {
        "schema_version": 1,
        "seed": split_seeds.pop(),
        "normal_parts": EXPECTED_NORMAL_PARTS,
        "defect_parts": EXPECTED_DEFECT_PARTS,
        "normal_images": sum(row["label"] == "normal" for row in template),
        "defect_images": sum(row["label"] == "defect" for row in template),
        "template_rows": len(template),
        "patchcore_rows": len(patchcore),
        "patchcore_trainer_rows": len(trainer),
        "excluded_defect_sessions": [DUPLICATE_DEFECT_SESSION],
        "normal_role_counts": dict(
            Counter(row["role"] for row in splits if row["label"] == "normal")
        ),
        "defect_role_counts": dict(
            Counter(row["role"] for row in splits if row["label"] == "defect")
        ),
        "defect_type_counts": dict(
            Counter(row["defect_type"] for row in splits if row["label"] == "defect")
        ),
        "roi_config_sha256": _sha256(root / "roi_config.json"),
    }
    if summary != expected_summary:
        raise ValueError("release summary does not match validated manifests")
    return {
        "status": "VALID",
        "normal_parts": EXPECTED_NORMAL_PARTS,
        "defect_parts": EXPECTED_DEFECT_PARTS,
        "template_rows": len(template),
        "patchcore_rows": len(patchcore),
        "patchcore_trainer_rows": len(trainer),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the release builder command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--normal-crop-manifest", type=Path, required=True)
    prepare.add_argument("--defect-crop-manifest", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=42)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one prepare or validate command."""
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_release(
            args.normal_crop_manifest,
            args.defect_crop_manifest,
            args.output_root,
            seed=args.seed,
        )
    else:
        result = validate_release(args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
