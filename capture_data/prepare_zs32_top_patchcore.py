# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Prepare a front/back-only incremental PatchCore release from ``zs32_top``."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2

from zs32_inspection.data.splitter import SPLIT_ALGORITHM, SplitPolicy, assign_grouped_splits


VIEWS = ("front", "back")
NEW_ROLE_COUNTS = {"train": 20, "calibration": 4, "final_test": 4}
CROP_MANIFEST_FIELDS = (
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
RELEASE_MANIFEST_FIELDS = (
    "source_origin",
    "physical_part_id",
    "hand",
    "view",
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
SPLIT_FIELDS = ("physical_part_id", "session_id", "group_id", "role", "seed")


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


def _safe_segment(value: str, field: str) -> str:
    if (
        not value
        or value in {".", ".."}
        or Path(value).is_absolute()
        or Path(value).name != value
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"unsafe {field}: {value!r}")
    return value


def _resolve_file(value: str, *, relative_to: Path) -> Path:
    path = Path(value)
    candidates = (path,) if path.is_absolute() else (Path.cwd() / path, relative_to / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(value)


def _load_roi_config(path: Path) -> tuple[int, int, dict[str, tuple[int, int, int, int]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("ROI config must be a JSON object")
    image_size = payload.get("image_size")
    views = payload.get("views")
    if not isinstance(image_size, dict) or not isinstance(views, dict):
        raise ValueError("ROI config requires image_size and views")
    width, height = int(image_size.get("width", 0)), int(image_size.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("ROI image dimensions must be positive")
    rois: dict[str, tuple[int, int, int, int]] = {}
    output_views: dict[str, Any] = {}
    for view in VIEWS:
        entry = views.get(view)
        if not isinstance(entry, dict) or not isinstance(entry.get("roi"), list):
            raise ValueError(f"ROI config is missing {view}")
        values = entry["roi"]
        if len(values) != 4 or any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError(f"invalid {view} ROI")
        x1, y1, x2, y2 = values
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"{view} ROI is outside {width}x{height}")
        rois[view] = (x1, y1, x2, y2)
        output_views[view] = {"roi": list(rois[view])}
        if isinstance(entry.get("reference_image"), str):
            output_views[view]["reference_image"] = entry["reference_image"]
    output_payload = {
        "schema_version": int(payload.get("schema_version", 1)),
        "coordinate_system": payload.get("coordinate_system", "pixel_xyxy_half_open"),
        "image_size": {"width": width, "height": height},
        "views": output_views,
    }
    return width, height, rois, output_payload


def _load_new_inventory(
    source_root: Path,
    *,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    manifests = sorted((source_root / "manifests").glob("*.csv"))
    if len(manifests) != 1:
        raise ValueError(f"expected exactly one zs32_top manifest, found {len(manifests)}")
    manifest = manifests[0]
    rows = _read_csv(manifest)
    complete_groups = {
        (_safe_segment(row["session_id"], "session_id"), _safe_segment(row["group_id"], "group_id"))
        for row in rows
        if row.get("record_type") == "sample" and row.get("sample_status") == "complete"
    }
    inventory: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        if row.get("record_type") != "image" or row.get("view") not in VIEWS:
            continue
        session_id = _safe_segment(row.get("session_id", ""), "session_id")
        group_id = _safe_segment(row.get("group_id", ""), "group_id")
        view = row["view"]
        if (session_id, group_id) not in complete_groups:
            continue
        key = (session_id, group_id, view)
        if key in seen:
            raise ValueError(f"duplicate new image identity: {key}")
        seen.add(key)
        source = _resolve_file(row.get("file", ""), relative_to=manifest.parent)
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not decode new image: {source}")
        if image.shape[:2] != (height, width):
            raise ValueError(f"new image has shape {image.shape[:2]}, expected {(height, width)}: {source}")
        inventory.append(
            {
                "session_id": session_id,
                "group_id": group_id,
                "physical_part_id": f"{session_id}:{group_id}",
                "view": view,
                "source_path": source,
            }
        )
    by_part: dict[str, set[str]] = defaultdict(set)
    for row in inventory:
        by_part[row["physical_part_id"]].add(row["view"])
    incomplete = {part: sorted(views) for part, views in by_part.items() if views != set(VIEWS)}
    if incomplete:
        raise ValueError(f"new front/back identities are incomplete: {incomplete}")
    if len(by_part) != 28 or len(inventory) != 56:
        raise ValueError(f"expected 28 complete front/back parts and 56 images, got {len(by_part)} and {len(inventory)}")
    return inventory


def _assign_new_roles(inventory: list[dict[str, Any]], seed: int) -> dict[str, str]:
    part_ids = sorted({str(row["physical_part_id"]) for row in inventory})
    policy = SplitPolicy(
        algorithm=SPLIT_ALGORITHM,
        seed=seed,
        calibration_ratio=4 / 28,
        test_ratio=4 / 28,
    )
    roles = {
        assignment.part_instance_id: (
            "final_test" if assignment.split == "test" else assignment.split
        )
        for assignment in assign_grouped_splits(
            {part_id: "normal" for part_id in part_ids},
            policy=policy,
        )
    }
    counts = Counter(roles.values())
    if dict(sorted(counts.items())) != dict(sorted(NEW_ROLE_COUNTS.items())):
        raise ValueError(f"unexpected new normal split counts: {dict(counts)}")
    return roles


def _base_source(row: dict[str, str], manifest: Path) -> Path:
    for field in ("target_path", "source_path"):
        value = row.get(field, "")
        if value:
            try:
                return _resolve_file(value, relative_to=manifest.parent)
            except FileNotFoundError:
                pass
    raise FileNotFoundError(f"base row has no readable source: {row}")


def _base_relative_target(row: dict[str, str], source: Path) -> Path:
    view = _safe_segment(row["view"], "view")
    session = _safe_segment(row["session_id"], "session_id")
    label = row["label"]
    role = row["release_role"]
    if label == "normal" and role == "train":
        return Path("patchcore/right") / view / "normal" / session / "images" / source.name
    if label == "normal" and role == "calibration":
        return Path("patchcore/right") / view / "normal_test" / session / "images" / source.name
    if label == "defect" and role == "calibration":
        defect_type = _safe_segment(row["defect_type"], "defect_type")
        return Path("patchcore/right") / view / "defect" / defect_type / session / "images" / source.name
    defect_type = row.get("defect_type", "") or "normal"
    return (
        Path("holdout/base")
        / role
        / label
        / _safe_segment(defect_type, "defect_type")
        / view
        / session
        / "images"
        / source.name
    )


def _new_relative_target(row: dict[str, Any], role: str, source: Path) -> tuple[Path, str]:
    view = row["view"]
    session = row["session_id"]
    if role == "train":
        return Path("patchcore/right") / view / "normal" / session / "images" / source.name, "normal"
    if role == "calibration":
        return Path("patchcore/right") / view / "normal_test" / session / "images" / source.name, "normal_test"
    return (
        Path("holdout/new_normal/final_test") / view / session / "images" / source.name,
        "holdout/final_test",
    )


def _trainer_row(
    *,
    source: Path,
    target: Path,
    view: str,
    label: str,
    defect_type: str,
    session_id: str,
    group_id: str,
    physical_part_id: str,
    role: str,
) -> dict[str, str]:
    return {
        "source_path": str(source),
        "output_path": str(target),
        "hand": "right",
        "source_view": view,
        "resolved_view": view,
        "view_corrected": "false",
        "label": label,
        "defect_type": defect_type,
        "session_id": session_id,
        "group_id": group_id,
        "physical_part_id": physical_part_id,
        "release_split": role,
    }


def validate_release(output_root: Path) -> dict[str, Any]:
    """Validate one published two-view release and return its summary."""
    root = output_root.expanduser().resolve()
    summary_path = root / "release_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("views") != list(VIEWS):
        raise ValueError(f"release views must be exactly {VIEWS}")
    right_root = root / "patchcore/right"
    actual_views = {path.name for path in right_root.iterdir() if path.is_dir()}
    unexpected = sorted(actual_views - set(VIEWS))
    if unexpected:
        raise ValueError(f"release contains unselected view directories: {unexpected}")
    if actual_views != set(VIEWS):
        raise ValueError(f"release is missing selected views: {sorted(set(VIEWS) - actual_views)}")
    crop_rows = _read_csv(root / "patchcore/crop_manifest.csv")
    if {row.get("resolved_view") for row in crop_rows} != set(VIEWS):
        raise ValueError("crop manifest contains an unselected view or misses front/back")
    release_rows = _read_csv(root / "release_manifest.csv")
    if {row.get("view") for row in release_rows} != set(VIEWS):
        raise ValueError("release manifest contains an unselected view or misses front/back")
    for row in crop_rows:
        if not Path(row["output_path"]).is_file():
            raise FileNotFoundError(row["output_path"])
    for row in release_rows:
        target = Path(row["target_path"])
        if not target.is_file():
            raise FileNotFoundError(target)
        if _sha256(target) != row["content_sha256"]:
            raise ValueError(f"content hash mismatch: {target}")
    split_rows = _read_csv(root / "new_normal_splits.csv")
    if len(split_rows) != 28 or len({row["physical_part_id"] for row in split_rows}) != 28:
        raise ValueError("new normal split manifest must contain 28 unique physical parts")
    split_counts = Counter(row["role"] for row in split_rows)
    if dict(sorted(split_counts.items())) != dict(sorted(NEW_ROLE_COUNTS.items())):
        raise ValueError(f"new normal split counts differ: {dict(split_counts)}")
    if len(crop_rows) != int(summary["active_trainer_rows"]):
        raise ValueError("active trainer row count differs from release summary")
    validated = dict(summary)
    validated["status"] = "VALID"
    return validated


def prepare_release(
    *,
    source_root: Path,
    base_release: Path,
    output_root: Path,
    roi_config: Path,
    seed: int = 42,
) -> dict[str, Any]:
    """Publish an isolated front/back release that incrementally extends v14."""
    source_root = source_root.expanduser().resolve()
    base_release = base_release.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    roi_config = roi_config.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing release: {output_root}")
    width, height, rois, output_roi = _load_roi_config(roi_config)
    inventory = _load_new_inventory(source_root, width=width, height=height)
    new_roles = _assign_new_roles(inventory, seed)
    base_manifest = base_release / "patchcore_manifest.csv"
    base_rows = [
        row
        for row in _read_csv(base_manifest)
        if row.get("hand") == "right" and row.get("view") in VIEWS
    ]
    if not base_rows:
        raise ValueError("base release contains no front/back rows")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent))
    trainer_rows: list[dict[str, str]] = []
    release_rows: list[dict[str, str]] = []
    try:
        for row in base_rows:
            source = _base_source(row, base_manifest)
            relative_target = _base_relative_target(row, source)
            staging_target = staging / relative_target
            final_target = output_root / relative_target
            staging_target.parent.mkdir(parents=True, exist_ok=True)
            staging_target.symlink_to(source)
            role = row["release_role"]
            target_bucket = row["target_bucket"]
            release_rows.append(
                {
                    "source_origin": "v14",
                    "physical_part_id": row["physical_part_id"],
                    "hand": "right",
                    "view": row["view"],
                    "label": row["label"],
                    "defect_type": row.get("defect_type", ""),
                    "session_id": row["session_id"],
                    "group_id": row["group_id"],
                    "release_role": role,
                    "target_bucket": target_bucket,
                    "source_path": str(source),
                    "target_path": str(final_target),
                    "content_sha256": _sha256(source),
                }
            )
            if (
                (row["label"] == "normal" and role in {"train", "calibration"})
                or (row["label"] == "defect" and role == "calibration")
            ):
                trainer_rows.append(
                    _trainer_row(
                        source=source,
                        target=final_target,
                        view=row["view"],
                        label=row["label"],
                        defect_type=row.get("defect_type", ""),
                        session_id=row["session_id"],
                        group_id=row["group_id"],
                        physical_part_id=row["physical_part_id"],
                        role=role,
                    )
                )

        for row in inventory:
            source = Path(row["source_path"])
            role = new_roles[row["physical_part_id"]]
            relative_target, target_bucket = _new_relative_target(row, role, source)
            staging_target = staging / relative_target
            final_target = output_root / relative_target
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"could not decode new image during crop: {source}")
            x1, y1, x2, y2 = rois[row["view"]]
            crop = image[y1:y2, x1:x2]
            staging_target.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(staging_target), crop):
                raise OSError(f"could not write crop: {staging_target}")
            release_rows.append(
                {
                    "source_origin": "zs32_top",
                    "physical_part_id": row["physical_part_id"],
                    "hand": "right",
                    "view": row["view"],
                    "label": "normal",
                    "defect_type": "",
                    "session_id": row["session_id"],
                    "group_id": row["group_id"],
                    "release_role": role,
                    "target_bucket": target_bucket,
                    "source_path": str(source),
                    "target_path": str(final_target),
                    "content_sha256": _sha256(staging_target),
                }
            )
            if role in {"train", "calibration"}:
                trainer_rows.append(
                    _trainer_row(
                        source=source,
                        target=final_target,
                        view=row["view"],
                        label="normal",
                        defect_type="",
                        session_id=row["session_id"],
                        group_id=row["group_id"],
                        physical_part_id=row["physical_part_id"],
                        role=role,
                    )
                )

        split_rows = [
            {
                "physical_part_id": part_id,
                "session_id": part_id.split(":", 1)[0],
                "group_id": part_id.split(":", 1)[1],
                "role": role,
                "seed": str(seed),
            }
            for part_id, role in sorted(new_roles.items())
        ]
        _write_csv(staging / "new_normal_splits.csv", SPLIT_FIELDS, split_rows)
        _write_csv(staging / "patchcore/crop_manifest.csv", CROP_MANIFEST_FIELDS, trainer_rows)
        _write_csv(staging / "release_manifest.csv", RELEASE_MANIFEST_FIELDS, release_rows)
        (staging / "patchcore/roi_config.json").write_text(
            json.dumps(output_roi, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        summary = {
            "schema_version": 1,
            "status": "VALID",
            "seed": seed,
            "views": list(VIEWS),
            "physical_identity_basis": "user_confirmed_each_complete_group_is_a_distinct_part",
            "base_release": str(base_release),
            "source_root": str(source_root),
            "roi_config": str(roi_config),
            "new_normal_parts": 28,
            "new_normal_images": 56,
            "new_normal_role_counts": dict(NEW_ROLE_COUNTS),
            "active_trainer_rows": len(trainer_rows),
            "release_rows": len(release_rows),
            "active_counts": {
                f"{label}/{role}/{view}": count
                for (label, role, view), count in sorted(
                    Counter(
                        (row["label"], row["release_split"], row["resolved_view"])
                        for row in trainer_rows
                    ).items()
                )
            },
        }
        (staging / "release_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output_root)
        return validate_release(output_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    """Build the reproducible front/back release CLI."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source-root", type=Path, required=True)
    prepare.add_argument("--base-release", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--roi-config", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=42)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Prepare or validate one front/back-only incremental release."""
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_release(
            source_root=args.source_root,
            base_release=args.base_release,
            output_root=args.output_root,
            roi_config=args.roi_config,
            seed=args.seed,
        )
    else:
        result = validate_release(args.output_root)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
