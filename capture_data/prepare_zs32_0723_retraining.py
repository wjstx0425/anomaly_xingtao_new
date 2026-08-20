# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Build and validate an immutable ZS32 0723 retraining-data release."""

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
PRIMARY_VIEWS = frozenset(view for view in VIEWS if not view.endswith("_secondary"))
SPLIT_COUNTS = (("train", 64), ("model_val", 16), ("calibration", 13), ("final_test", 13))
GROUP_PATTERN = re.compile(r"(group\d{3})(?:\D|$)", re.IGNORECASE)

SPLIT_FIELDS = ("physical_part_id", "session_id", "group_id", "split", "seed")
RELEASE_FIELDS = (
    "component",
    "source_origin",
    "physical_part_id",
    "view",
    "release_split",
    "target_split",
    "source_path",
    "target_path",
    "content_sha256",
)
PATCHCORE_FIELDS = (
    "source_origin",
    "physical_part_id",
    "hand",
    "view",
    "label",
    "defect_type",
    "session_id",
    "group_id",
    "release_split",
    "target_bucket",
    "source_path",
    "target_path",
    "content_sha256",
)
PATCHCORE_TRAINER_FIELDS = (
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
TEMPLATE_FIELDS = (
    "source_origin",
    "part_id",
    "physical_part_id",
    "hand",
    "view",
    "resolved_view",
    "label",
    "release_split",
    "split",
    "image_path",
    "source_path",
    "content_sha256",
)
YOLO_FIELDS = (
    "source_origin",
    "source_image_path",
    "source_label_path",
    "new_image_path",
    "new_label_path",
    "sample_id",
    "view",
    "hand",
    "defect_type",
    "session_id",
    "group_id",
    "physical_part_id",
    "release_split",
    "original_split",
    "split",
    "source_kind",
    "annotation_status",
    "pixel_sha256",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return [
            {str(key): "" if value is None else str(value).strip() for key, value in row.items()}
            for row in reader
        ]


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_file(value: str, manifest: Path) -> Path:
    path = Path(value)
    candidates = (path,) if path.is_absolute() else (Path.cwd() / path, manifest.parent / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"manifest asset does not exist: {value!r} ({manifest})")


def _required(row: dict[str, str], field: str, *, context: str) -> str:
    value = row.get(field, "").strip()
    if not value:
        raise ValueError(f"{context} is missing required field {field!r}")
    return value


def _extract_group(row: dict[str, str], *, context: str) -> str:
    for field in ("group_id", "output_path", "source_path"):
        match = GROUP_PATTERN.search(row.get(field, ""))
        if match:
            return match.group(1).lower()
    raise ValueError(f"{context} has no groupNNN identity")


def _unique_name(prefix: str, index: int, source: Path) -> str:
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.stem)
    return f"{prefix}__{index:06d}__{safe_stem}{source.suffix.lower()}"


def _symlink(stage_root: Path, relative_path: Path, source: Path) -> None:
    target = stage_root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite release asset: {target}")
    target.symlink_to(source.resolve())


def _new_inventory(new_crop_manifest: Path) -> list[dict[str, str]]:
    rows = _read_csv(new_crop_manifest)
    if len(rows) != 848:
        raise ValueError(f"0723 crop manifest must contain exactly 848 rows, got {len(rows)}")
    views_by_part: dict[str, set[str]] = defaultdict(set)
    hashes_by_part: dict[str, set[str]] = defaultdict(set)
    prepared: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=2):
        context = f"{new_crop_manifest}:row {index}"
        if _required(row, "hand", context=context).lower() != "right":
            raise ValueError(f"{context} must be right-hand data")
        if _required(row, "label", context=context).lower() != "normal":
            raise ValueError(f"{context} must be normal data")
        view = row.get("resolved_view", "") or row.get("source_view", "")
        if view not in VIEWS:
            raise ValueError(f"{context} has unsupported view {view!r}")
        session = _required(row, "session_id", context=context)
        group = _extract_group(row, context=context)
        physical_part_id = f"{session}:{group}"
        source = _resolve_file(row.get("output_path", "") or row.get("source_path", ""), new_crop_manifest)
        content_sha256 = _sha256(source)
        views_by_part[physical_part_id].add(view)
        hashes_by_part[content_sha256].add(physical_part_id)
        prepared.append(
            {
                **row,
                "view": view,
                "group_id": group,
                "physical_part_id": physical_part_id,
                "asset_path": str(source),
                "content_sha256": content_sha256,
            }
        )
    if len(views_by_part) != 106:
        raise ValueError(f"0723 crop manifest must contain exactly 106 physical parts, got {len(views_by_part)}")
    incomplete = {part: sorted(views) for part, views in views_by_part.items() if views != set(VIEWS)}
    if incomplete:
        raise ValueError(f"each 0723 physical part must contain all eight canonical views: {incomplete}")
    duplicate_parts = {digest: parts for digest, parts in hashes_by_part.items() if len(parts) > 1}
    if duplicate_parts:
        raise ValueError(f"identical encoded images belong to different physical parts: {duplicate_parts}")
    return prepared


def _assign_splits(parts: list[str], seed: int) -> dict[str, str]:
    if len(parts) != sum(count for _, count in SPLIT_COUNTS):
        raise ValueError("the fixed 64/16/13/13 split requires exactly 106 physical parts")
    shuffled = sorted(parts)
    random.Random(seed).shuffle(shuffled)
    result: dict[str, str] = {}
    offset = 0
    for split, count in SPLIT_COUNTS:
        for part in shuffled[offset : offset + count]:
            result[part] = split
        offset += count
    return result


def _final_path(output_root: Path, relative_path: Path) -> str:
    return str((output_root.resolve() / relative_path).absolute())


def _prepare_patchcore(
    *,
    stage_root: Path,
    output_root: Path,
    new_rows: list[dict[str, str]],
    old_manifest: Path,
    splits: dict[str, str],
    release_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    old_rows = _read_csv(old_manifest)
    work: list[tuple[str, dict[str, str]]] = [("legacy", row) for row in old_rows]
    work.extend(("zs32_0723", row) for row in new_rows)
    for index, (origin, row) in enumerate(work):
        context = f"{old_manifest if origin == 'legacy' else '0723'}:row {index + 2}"
        view = row.get("resolved_view", "") or row.get("view", "") or row.get("source_view", "")
        if view not in VIEWS:
            raise ValueError(f"{context} has unsupported view {view!r}")
        if origin == "legacy" and view not in PRIMARY_VIEWS:
            continue
        label = _required(row, "label", context=context).lower()
        if label not in {"normal", "defect"}:
            raise ValueError(f"{context} has unsupported label {label!r}")
        source = (
            Path(row["asset_path"])
            if origin == "zs32_0723"
            else _resolve_file(row.get("output_path", "") or row.get("source_path", ""), old_manifest)
        )
        if origin == "legacy":
            release_split = "legacy_train" if label == "normal" else "retrospective_test"
            bucket = "normal" if label == "normal" else "defect"
            physical_part_id = ""
            content_sha256 = ""
        else:
            physical_part_id = row["physical_part_id"]
            release_split = splits[physical_part_id]
            bucket = {
                "train": "normal",
                "model_val": "holdout/model_val",
                "calibration": "normal_test",
                "final_test": "holdout/final_test",
            }[release_split]
            content_sha256 = row["content_sha256"]
        defect_type = row.get("defect_type", "")
        session_id = _required(row, "session_id", context=context)
        safe_session = re.sub(r"[^A-Za-z0-9_.-]+", "_", session_id)
        relative = Path("patchcore") / "right" / view / bucket
        if label == "defect" and defect_type:
            relative /= defect_type
        relative = relative / safe_session / "images"
        relative /= _unique_name(origin, index, source)
        _symlink(stage_root, relative, source)
        target_path = _final_path(output_root, relative)
        output_row = {
            "source_origin": origin,
            "physical_part_id": physical_part_id,
            "hand": "right",
            "view": view,
            "label": label,
            "defect_type": defect_type,
            "session_id": session_id,
            "group_id": row.get("group_id", ""),
            "release_split": release_split,
            "target_bucket": bucket,
            "source_path": str(source),
            "target_path": target_path,
            "content_sha256": content_sha256,
        }
        result.append(output_row)
        release_rows.append(
            {
                "component": "patchcore",
                **{key: output_row[key] for key in RELEASE_FIELDS if key in output_row},
                "target_split": bucket,
            }
        )
    return result


def _prepare_template(
    *,
    output_root: Path,
    new_rows: list[dict[str, str]],
    splits: dict[str, str],
    patchcore_rows: list[dict[str, str]],
    release_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    patchcore_path = {
        (row["physical_part_id"], row["view"]): row["target_path"]
        for row in patchcore_rows
        if row["source_origin"] == "zs32_0723"
    }
    result: list[dict[str, str]] = []
    for row in new_rows:
        release_split = splits[row["physical_part_id"]]
        output_row = {
            "source_origin": "zs32_0723",
            "part_id": row["physical_part_id"],
            "physical_part_id": row["physical_part_id"],
            "hand": "right",
            "view": row["view"],
            "resolved_view": row["view"],
            "label": "normal",
            "release_split": release_split,
            "split": release_split,
            "image_path": patchcore_path[(row["physical_part_id"], row["view"])],
            "source_path": row["asset_path"],
            "content_sha256": row["content_sha256"],
        }
        result.append(output_row)
        release_rows.append(
            {
                "component": "template",
                "source_origin": "zs32_0723",
                "physical_part_id": row["physical_part_id"],
                "view": row["view"],
                "release_split": release_split,
                "target_split": release_split,
                "source_path": row["asset_path"],
                "target_path": output_row["image_path"],
                "content_sha256": row["content_sha256"],
            }
        )
    return result


def _prepare_yolo(
    *,
    stage_root: Path,
    output_root: Path,
    new_rows: list[dict[str, str]],
    old_mapping: Path,
    splits: dict[str, str],
    release_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for index, row in enumerate(_read_csv(old_mapping)):
        original_split = _required(row, "split", context=f"{old_mapping}:row {index + 2}")
        if original_split not in {"train", "val", "test"}:
            raise ValueError(f"old YOLO split must be train/val/test, got {original_split!r}")
        target_split = "retrospective_test" if original_split == "test" else original_split
        image_source = _resolve_file(_required(row, "new_image_path", context=str(old_mapping)), old_mapping)
        label_source = _resolve_file(_required(row, "new_label_path", context=str(old_mapping)), old_mapping)
        image_relative = Path("yolo") / "images" / target_split / _unique_name("legacy", index, image_source)
        label_relative = Path("yolo") / "labels" / target_split / f"{image_relative.stem}.txt"
        _symlink(stage_root, image_relative, image_source)
        _symlink(stage_root, label_relative, label_source)
        output_row = {
            **{field: row.get(field, "") for field in YOLO_FIELDS},
            "source_origin": "legacy",
            "source_image_path": str(image_source),
            "source_label_path": str(label_source),
            "new_image_path": _final_path(output_root, image_relative),
            "new_label_path": _final_path(output_root, label_relative),
            "physical_part_id": "",
            "release_split": "retrospective_test" if original_split == "test" else f"legacy_{original_split}",
            "original_split": original_split,
            "split": target_split,
        }
        result.append(output_row)
        release_rows.append(
            {
                "component": "yolo",
                "source_origin": "legacy",
                "physical_part_id": "",
                "view": row.get("view", ""),
                "release_split": output_row["release_split"],
                "target_split": target_split,
                "source_path": str(image_source),
                "target_path": output_row["new_image_path"],
                "content_sha256": row.get("pixel_sha256", ""),
            }
        )
    for index, row in enumerate(new_rows):
        release_split = splits[row["physical_part_id"]]
        target_split = {"train": "train", "model_val": "val"}.get(release_split, release_split)
        source = Path(row["asset_path"])
        stem = f"0723__{row['session_id']}__{row['group_id']}__{row['view']}"
        image_relative = Path("yolo") / "images" / target_split / f"{stem}{source.suffix.lower()}"
        label_relative = Path("yolo") / "labels" / target_split / f"{stem}.txt"
        _symlink(stage_root, image_relative, source)
        label_target = stage_root / label_relative
        label_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.open("x", encoding="utf-8").close()
        output_row = {
            "source_origin": "zs32_0723",
            "source_image_path": str(source),
            "source_label_path": "",
            "new_image_path": _final_path(output_root, image_relative),
            "new_label_path": _final_path(output_root, label_relative),
            "sample_id": row["physical_part_id"],
            "view": row["view"],
            "hand": "right",
            "defect_type": "",
            "session_id": row["session_id"],
            "group_id": row["group_id"],
            "physical_part_id": row["physical_part_id"],
            "release_split": release_split,
            "original_split": "",
            "split": target_split,
            "source_kind": "normal_negative",
            "annotation_status": "strict_empty_label",
            "pixel_sha256": row["content_sha256"],
        }
        result.append(output_row)
        release_rows.append(
            {
                "component": "yolo",
                "source_origin": "zs32_0723",
                "physical_part_id": row["physical_part_id"],
                "view": row["view"],
                "release_split": release_split,
                "target_split": target_split,
                "source_path": str(source),
                "target_path": output_row["new_image_path"],
                "content_sha256": row["content_sha256"],
            }
        )
    return result


def _artifact_receipts(stage_root: Path, relative_paths: tuple[Path, ...]) -> list[dict[str, str]]:
    return [
        {"relative_path": relative.as_posix(), "sha256": _sha256(stage_root / relative)}
        for relative in relative_paths
    ]


def prepare_release(
    *,
    new_crop_manifest: Path,
    old_patchcore_manifest: Path,
    old_yolo_mapping: Path,
    output_root: Path,
    seed: int = 42,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Prepare a deterministic 24-group retraining release without overwriting.

    Args:
        new_crop_manifest (Path): Manifest for the 848 cropped ``zs32_0723`` normal images.
        old_patchcore_manifest (Path): Legacy PatchCore crop manifest.
        old_yolo_mapping (Path): Legacy YOLO image/label mapping.
        output_root (Path): New release directory. It must not already exist.
        seed (int): Seed used for the fixed physical-part partition.
        dry_run (bool): Validate inputs and report the planned partition without writing.

    Returns:
        dict[str, Any]: Machine-readable preparation summary.

    Raises:
        FileExistsError: If ``output_root`` already exists.
        FileNotFoundError: If an input or referenced asset is missing.
        ValueError: If an input violates the release contract.
    """
    inputs = tuple(Path(path).resolve() for path in (new_crop_manifest, old_patchcore_manifest, old_yolo_mapping))
    roi_config = inputs[0].parent / "roi_config.json"
    if not roi_config.is_file():
        raise FileNotFoundError(f"new crop manifest ROI config does not exist: {roi_config}")
    output_root = Path(output_root).resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output root: {output_root}")
    new_rows = _new_inventory(inputs[0])
    splits = _assign_splits(sorted({row["physical_part_id"] for row in new_rows}), seed)
    summary: dict[str, Any] = {
        "status": "DRY_RUN" if dry_run else "PREPARED",
        "dry_run": dry_run,
        "seed": seed,
        "materialization": "symlink",
        "new_physical_parts": len(splits),
        "new_images": len(new_rows),
        "split_counts": dict(Counter(splits.values())),
        "input_sha256": {
            "new_crop_manifest": _sha256(inputs[0]),
            "old_patchcore_manifest": _sha256(inputs[1]),
            "old_yolo_mapping": _sha256(inputs[2]),
            "roi_config": _sha256(roi_config),
        },
    }
    if dry_run:
        return summary

    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        split_rows = []
        for part, split in sorted(splits.items()):
            session, group = part.rsplit(":", 1)
            split_rows.append(
                {"physical_part_id": part, "session_id": session, "group_id": group, "split": split, "seed": seed}
            )
        release_rows: list[dict[str, str]] = []
        patchcore_rows = _prepare_patchcore(
            stage_root=stage_root,
            output_root=output_root,
            new_rows=new_rows,
            old_manifest=inputs[1],
            splits=splits,
            release_rows=release_rows,
        )
        template_rows = _prepare_template(
            output_root=output_root,
            new_rows=new_rows,
            splits=splits,
            patchcore_rows=patchcore_rows,
            release_rows=release_rows,
        )
        yolo_rows = _prepare_yolo(
            stage_root=stage_root,
            output_root=output_root,
            new_rows=new_rows,
            old_mapping=inputs[2],
            splits=splits,
            release_rows=release_rows,
        )
        artifact_paths = (
            Path("physical_part_splits.csv"),
            Path("release_manifest.csv"),
            Path("patchcore_manifest.csv"),
            Path("patchcore/crop_manifest.csv"),
            Path("template_manifest.csv"),
            Path("yolo/mapping.csv"),
            Path("yolo/data.yaml"),
            Path("roi_config.json"),
            Path("patchcore/roi_config.json"),
        )
        _write_csv(stage_root / artifact_paths[0], SPLIT_FIELDS, split_rows)
        _write_csv(stage_root / artifact_paths[1], RELEASE_FIELDS, release_rows)
        _write_csv(stage_root / artifact_paths[2], PATCHCORE_FIELDS, patchcore_rows)
        trainer_rows = [
            {
                "source_path": row["source_path"],
                "output_path": row["target_path"],
                "hand": row["hand"],
                "source_view": row["view"],
                "resolved_view": row["view"],
                "view_corrected": "false",
                "label": row["label"],
                "defect_type": row["defect_type"],
                "session_id": row["session_id"],
                "group_id": row["group_id"],
                "physical_part_id": row["physical_part_id"],
                "release_split": row["release_split"],
            }
            for row in patchcore_rows
        ]
        _write_csv(stage_root / artifact_paths[3], PATCHCORE_TRAINER_FIELDS, trainer_rows)
        _write_csv(stage_root / artifact_paths[4], TEMPLATE_FIELDS, template_rows)
        _write_csv(stage_root / artifact_paths[5], YOLO_FIELDS, yolo_rows)
        yaml_path = stage_root / artifact_paths[6]
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(
            f"path: {output_root / 'yolo'}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/retrospective_test\n\n"
            "names:\n"
            "  0: defect\n",
            encoding="utf-8",
        )
        for relative in artifact_paths[7:]:
            target = stage_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(roi_config, target)
        summary.update(
            patchcore_rows=len(patchcore_rows),
            template_rows=len(template_rows),
            yolo_rows=len(yolo_rows),
        )
        (stage_root / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _write_csv(
            stage_root / "sha256_receipts.csv",
            ("relative_path", "sha256"),
            _artifact_receipts(stage_root, artifact_paths),
        )
        os.replace(stage_root, output_root)
    except BaseException:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise
    validate_release(output_root)
    return summary


def validate_release(output_root: Path) -> dict[str, Any]:
    """Validate manifests, split isolation, symlinks, labels, and receipts.

    Args:
        output_root (Path): Prepared release directory.

    Returns:
        dict[str, Any]: Machine-readable validation counts and status.

    Raises:
        FileNotFoundError: If a required manifest or artifact is missing.
        ValueError: If the release violates the data contract.
    """
    root = Path(output_root).resolve()
    split_rows = _read_csv(root / "physical_part_splits.csv")
    release_rows = _read_csv(root / "release_manifest.csv")
    split_by_part = {row["physical_part_id"]: row["split"] for row in split_rows}
    if len(split_by_part) != 106 or Counter(split_by_part.values()) != dict(SPLIT_COUNTS):
        raise ValueError("physical-part split manifest is not the required 64/16/13/13 partition")
    observed: dict[str, set[str]] = defaultdict(set)
    hashes: dict[str, set[str]] = defaultdict(set)
    for row in release_rows:
        if row["source_origin"] != "zs32_0723":
            continue
        observed[row["physical_part_id"]].add(row["release_split"])
        if row["content_sha256"]:
            hashes[row["content_sha256"]].add(row["release_split"])
    if any(len(splits) != 1 for splits in observed.values()):
        raise ValueError("physical-part split leakage detected in release_manifest.csv")
    if set(observed) != set(split_by_part):
        raise ValueError("release manifest does not cover all 106 physical parts")
    if any(next(iter(splits)) != split_by_part[part] for part, splits in observed.items()):
        raise ValueError("physical-part split leakage detected between central manifests")
    if any(len(splits) != 1 for splits in hashes.values()):
        raise ValueError("encoded-hash split leakage detected in release_manifest.csv")

    patchcore = _read_csv(root / "patchcore_manifest.csv")
    for row in patchcore:
        target = Path(row["target_path"])
        if not target.is_symlink() or not target.is_file():
            raise ValueError(f"PatchCore asset is not a live symlink: {target}")
        if row["view"].endswith("_secondary") and row["source_origin"] != "zs32_0723":
            raise ValueError("secondary PatchCore data must come only from zs32_0723")
        expected_segment = f"/{row['session_id']}/images/"
        if expected_segment not in target.as_posix():
            raise ValueError(f"PatchCore asset lost session identity: {target}")
    trainer_manifest = _read_csv(root / "patchcore" / "crop_manifest.csv")
    if len(trainer_manifest) != len(patchcore):
        raise ValueError("PatchCore trainer manifest does not match the central PatchCore manifest")
    if any(Path(row["output_path"]).is_file() is False for row in trainer_manifest):
        raise ValueError("PatchCore trainer manifest contains a missing output_path")

    root_roi = root / "roi_config.json"
    patchcore_roi = root / "patchcore" / "roi_config.json"
    if _sha256(root_roi) != _sha256(patchcore_roi):
        raise ValueError("release and PatchCore ROI configs do not match")

    template = _read_csv(root / "template_manifest.csv")
    if any(row["source_origin"] != "zs32_0723" for row in template):
        raise ValueError("Template manifest must contain only zs32_0723")
    if {row["view"] for row in template} != set(VIEWS):
        raise ValueError("Template manifest must contain all eight views")
    if {row["split"] for row in template} != {split for split, _count in SPLIT_COUNTS}:
        raise ValueError("Template manifest must preserve all four explicit data roles")

    yolo = _read_csv(root / "yolo" / "mapping.csv")
    for row in yolo:
        image = Path(row["new_image_path"])
        label = Path(row["new_label_path"])
        if not image.is_symlink() or not image.is_file() or not label.is_file():
            raise ValueError(f"YOLO image/label pair is incomplete: {image}")
        if row["source_origin"] == "zs32_0723" and label.read_bytes() != b"":
            raise ValueError(f"new normal YOLO label is not strictly empty: {label}")
        if row["source_origin"] == "legacy":
            source_label = Path(row["source_label_path"])
            if source_label.read_bytes() != label.read_bytes():
                raise ValueError(f"legacy YOLO bbox label changed: {label}")

    for receipt in _read_csv(root / "sha256_receipts.csv"):
        artifact = root / receipt["relative_path"]
        if _sha256(artifact) != receipt["sha256"]:
            raise ValueError(f"SHA256 receipt mismatch: {artifact}")
    return {
        "status": "VALID",
        "physical_part_count": len(split_by_part),
        "release_rows": len(release_rows),
        "patchcore_rows": len(patchcore),
        "template_rows": len(template),
        "yolo_rows": len(yolo),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the prepare/validate CLI parser.

    Returns:
        argparse.ArgumentParser: Configured command-line parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="build a new immutable retraining-data release")
    prepare.add_argument("--new-crop-manifest", type=Path, required=True)
    prepare.add_argument("--old-patchcore-manifest", type=Path, required=True)
    prepare.add_argument("--old-yolo-mapping", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--dry-run", action="store_true")
    validate = subparsers.add_parser("validate", help="validate an existing retraining-data release")
    validate.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one CLI subcommand and print a machine-readable result.

    Args:
        argv (list[str] | None): Optional argument list. Uses process arguments when omitted.

    Returns:
        int: Zero after a successful prepare or validation operation.
    """
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_release(
            new_crop_manifest=args.new_crop_manifest,
            old_patchcore_manifest=args.old_patchcore_manifest,
            old_yolo_mapping=args.old_yolo_mapping,
            output_root=args.output_root,
            seed=args.seed,
            dry_run=args.dry_run,
        )
    else:
        result = validate_release(args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
