# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Prepare and validate an immutable, part-isolated ZS32 Template release."""

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
SPLIT_FIELDS = ("physical_part_id", "session_id", "group_id", "role", "seed")
MANIFEST_FIELDS = (
    "source_name",
    "part_id",
    "physical_part_id",
    "hand",
    "view",
    "resolved_view",
    "label",
    "role",
    "split",
    "session_id",
    "group_id",
    "image_path",
    "source_path",
    "content_sha256",
)
RECEIPT_FIELDS = ("relative_path", "sha256")
REPO_ROOT = Path(__file__).resolve().parents[1]
ZS32_0727_SOURCE_ROOT = Path("dataset") / "zs32_0727"
ZS32_0727_TEMPLATE_ROI_ROOT = Path("dataset") / "zs32_0727_template_roi_v1"
ZS32_0727_CROP_MANIFEST = ZS32_0727_TEMPLATE_ROI_ROOT / "crop_manifest.csv"
ZS32_0727_ROI_CONFIG_SHA256 = "9412b2838cdb96f722db356714cf9bbbb5ea01810671ccf92b67323de77ebe65"
ZS32_0727_GROUP_IDS = frozenset(f"group{number:03d}" for number in range(1, 20))
ZS32_0727_EXPECTED_PARTS = 19
ZS32_0727_SEED = 42
ZS32_0727_ROLE_COUNTS = {
    "train": 11,
    "model_val": 3,
    "calibration": 3,
    "final_test": 2,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return [
            {str(key): ("" if value is None else str(value).strip()) for key, value in row.items()}
            for row in reader
        ]


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _required(row: dict[str, str], field: str, *, context: str) -> str:
    value = row.get(field, "").strip()
    if not value:
        raise ValueError(f"{context} is missing required field {field!r}")
    return value


def _resolve_asset(row: dict[str, str], *, source_group: str, context: str) -> Path:
    """Resolve one Stage30 crop while pinning it to the authoritative ROI release."""
    value = _required(row, "output_path", context=context)
    path = Path(value)
    candidate = path if path.is_absolute() else REPO_ROOT / path
    if not candidate.is_file():
        raise FileNotFoundError(f"{context} output_path does not exist: {value!r}")
    resolved = candidate.resolve()
    expected_root = (REPO_ROOT / ZS32_0727_TEMPLATE_ROI_ROOT).resolve()
    try:
        output_relative = resolved.relative_to(expected_root)
    except ValueError as error:
        raise ValueError(f"{context} output_path must be under {ZS32_0727_TEMPLATE_ROI_ROOT}") from error
    match = GROUP_PATTERN.search(output_relative.as_posix())
    if not match or match.group(1).lower() != source_group:
        raise ValueError(f"{context} output_path group identity does not match source_path")
    return resolved


def _source_path_relative_to_repo(row: dict[str, str], *, context: str) -> Path:
    source_path = Path(_required(row, "source_path", context=context))
    absolute_path = source_path if source_path.is_absolute() else REPO_ROOT / source_path
    try:
        return absolute_path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"{context} source_path must be inside repository root {REPO_ROOT}") from error


def _enforce_source_provenance(row: dict[str, str], *, source_name: str, context: str) -> str:
    """Return the authoritative source group after enforcing named-source rules."""
    if source_name != "zs32_0727":
        raise ValueError("source_name must be exactly 'zs32_0727'")
    source_relative = _source_path_relative_to_repo(row, context=context)
    try:
        source_relative.relative_to(ZS32_0727_SOURCE_ROOT)
    except ValueError as error:
        raise ValueError(
            f"{context} source_name 'zs32_0727' requires source_path under {ZS32_0727_SOURCE_ROOT}"
        ) from error
    match = GROUP_PATTERN.search(source_relative.as_posix())
    if not match:
        raise ValueError(f"{context} source_name 'zs32_0727' has no groupNNN identity in source_path")
    group_id = match.group(1).lower()
    if group_id not in ZS32_0727_GROUP_IDS:
        raise ValueError(
            f"{context} source_name 'zs32_0727' requires exactly groups group001 through group019"
        )
    source = REPO_ROOT / source_relative
    if not source.is_file():
        raise FileNotFoundError(f"{context} source_path does not exist: {source}")
    return group_id


def _extract_group(row: dict[str, str], *, context: str) -> str:
    for field in ("group_id", "output_path", "source_path"):
        match = GROUP_PATTERN.search(row.get(field, ""))
        if match:
            return match.group(1).lower()
    raise ValueError(f"{context} has no groupNNN identity")


def _load_inventory(crop_manifest: Path, *, source_name: str, expected_parts: int) -> list[dict[str, str]]:
    rows = _read_csv(crop_manifest)
    if not rows:
        raise ValueError(f"crop manifest has no rows: {crop_manifest}")
    views_by_part: dict[str, set[str]] = defaultdict(set)
    row_keys: set[tuple[str, str]] = set()
    first_row_by_hash: dict[str, str] = {}
    groups: set[str] = set()
    prepared: list[dict[str, str]] = []
    for number, row in enumerate(rows, start=2):
        context = f"{crop_manifest}:row {number}"
        if _required(row, "hand", context=context).lower() != "right":
            raise ValueError(f"{context} must be right-hand data")
        if _required(row, "label", context=context).lower() != "normal":
            raise ValueError(f"{context} must be normal data")
        view = (row.get("resolved_view", "") or row.get("source_view", "") or row.get("view", "")).lower()
        if view not in VIEWS:
            raise ValueError(f"{context} has unsupported view {view!r}")
        session_id = _required(row, "session_id", context=context)
        group_id = _enforce_source_provenance(row, source_name=source_name, context=context)
        part_id = f"{session_id}:{group_id}"
        if (part_id, view) in row_keys:
            raise ValueError(f"{context} duplicates physical part/view {part_id!r}/{view!r}")
        row_keys.add((part_id, view))
        source = _resolve_asset(row, source_group=group_id, context=context)
        content_sha256 = _sha256(source)
        if content_sha256 in first_row_by_hash:
            raise ValueError(
                f"{context} encoded image is duplicated from {first_row_by_hash[content_sha256]}"
            )
        first_row_by_hash[content_sha256] = context
        views_by_part[part_id].add(view)
        groups.add(group_id)
        prepared.append(
            {
                "part_id": part_id,
                "session_id": session_id,
                "group_id": group_id,
                "view": view,
                "source_path": _required(row, "source_path", context=context),
                "asset_path": str(source),
                "content_sha256": content_sha256,
            },
        )
    if len(views_by_part) != expected_parts:
        raise ValueError(f"expected exactly {expected_parts} physical parts, got {len(views_by_part)}")
    if groups != ZS32_0727_GROUP_IDS:
        raise ValueError("source provenance requires exactly groups group001 through group019")
    incomplete = {part: sorted(views) for part, views in views_by_part.items() if views != set(VIEWS)}
    if incomplete:
        raise ValueError(f"each physical part must contain all eight canonical views: {incomplete}")
    return sorted(prepared, key=lambda row: (row["part_id"], VIEWS.index(row["view"])))


def _assign_roles(
    parts: list[str],
    *,
    seed: int,
    counts: dict[str, int],
) -> dict[str, str]:
    if set(counts) != set(ROLES) or any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError(f"role counts must be non-negative integers for {ROLES}")
    if len(parts) != sum(counts.values()):
        raise ValueError("physical-part count must equal the sum of all four role counts")
    shuffled = sorted(parts)
    random.Random(seed).shuffle(shuffled)
    roles: dict[str, str] = {}
    offset = 0
    for role in ROLES:
        for part in shuffled[offset : offset + counts[role]]:
            roles[part] = role
        offset += counts[role]
    return roles


def _symlink(stage_root: Path, relative_path: Path, source: Path) -> None:
    destination = stage_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite release asset: {destination}")
    destination.symlink_to(source)


def _final_path(output_root: Path, relative_path: Path) -> str:
    return str((output_root / relative_path).resolve())


def prepare_template_release(
    crop_manifest: Path,
    output_root: Path,
    *,
    source_name: str,
    expected_parts: int,
    train_count: int,
    model_val_count: int,
    calibration_count: int,
    final_test_count: int,
    seed: int = 42,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Prepare a new immutable Template release from a Stage30 crop manifest.

    Args:
        crop_manifest: Stage30 CSV with one cropped image per part and view.
        output_root: New release directory, which must not already exist.
        source_name: Immutable source dataset identity recorded in manifests.
        expected_parts: Exact number of physical parts required.
        train_count: Number of parts assigned to template training.
        model_val_count: Number of parts reserved for model validation.
        calibration_count: Number of parts used to place normal-only thresholds.
        final_test_count: Number of parts reserved for final evaluation.
        seed: Deterministic part-role assignment seed.
        dry_run: Validate and report without writing output files.

    Returns:
        A machine-readable release summary.
    """
    source_name = source_name.strip()
    if source_name != "zs32_0727":
        raise ValueError("source_name must be exactly 'zs32_0727'")
    counts = {
        "train": train_count,
        "model_val": model_val_count,
        "calibration": calibration_count,
        "final_test": final_test_count,
    }
    if (
        expected_parts != ZS32_0727_EXPECTED_PARTS
        or seed != ZS32_0727_SEED
        or counts != ZS32_0727_ROLE_COUNTS
    ):
        raise ValueError(
            "zs32_0727 requires the fixed 19-part seed-42 split "
            "train=11/model_val=3/calibration=3/final_test=2"
        )
    crop_manifest = Path(crop_manifest).resolve()
    expected_manifest = (REPO_ROOT / ZS32_0727_CROP_MANIFEST).resolve()
    if crop_manifest != expected_manifest:
        raise ValueError(f"crop_manifest must be exactly {ZS32_0727_CROP_MANIFEST}")
    roi_config = crop_manifest.parent / "roi_config.json"
    if not roi_config.is_file():
        raise FileNotFoundError(f"crop manifest ROI config does not exist: {roi_config}")
    if _sha256(roi_config) != ZS32_0727_ROI_CONFIG_SHA256:
        raise ValueError("ROI config SHA256 does not match pinned 0727 contract")
    output_root = Path(output_root).resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output root: {output_root}")
    inventory = _load_inventory(crop_manifest, source_name=source_name, expected_parts=expected_parts)
    roles = _assign_roles(sorted({row["part_id"] for row in inventory}), seed=seed, counts=counts)
    summary: dict[str, Any] = {
        "status": "DRY_RUN" if dry_run else "PREPARED",
        "dry_run": dry_run,
        "source_name": source_name,
        "seed": seed,
        "materialization": "symlink",
        "expected_parts": expected_parts,
        "physical_part_count": len(roles),
        "image_count": len(inventory),
        "role_counts": dict(Counter(roles.values())),
        "input_sha256": {"crop_manifest": _sha256(crop_manifest), "roi_config": _sha256(roi_config)},
    }
    if dry_run:
        return summary

    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        split_rows: list[dict[str, Any]] = []
        for part_id, role in sorted(roles.items()):
            session_id, group_id = part_id.rsplit(":", 1)
            split_rows.append(
                {
                    "physical_part_id": part_id,
                    "session_id": session_id,
                    "group_id": group_id,
                    "role": role,
                    "seed": seed,
                },
            )
        manifest_rows: list[dict[str, Any]] = []
        for row in inventory:
            source = Path(row["asset_path"])
            role = roles[row["part_id"]]
            relative_path = (
                Path("images")
                / role
                / row["session_id"]
                / row["group_id"]
                / f"{row['view']}{source.suffix.lower()}"
            )
            _symlink(stage_root, relative_path, source)
            manifest_rows.append(
                {
                    "source_name": source_name,
                    "part_id": row["part_id"],
                    "physical_part_id": row["part_id"],
                    "hand": "right",
                    "view": row["view"],
                    "resolved_view": row["view"],
                    "label": "normal",
                    "role": role,
                    "split": role,
                    "session_id": row["session_id"],
                    "group_id": row["group_id"],
                    "image_path": _final_path(output_root, relative_path),
                    "source_path": row["source_path"],
                    "content_sha256": row["content_sha256"],
                },
            )
        _write_csv(stage_root / "physical_part_splits.csv", SPLIT_FIELDS, split_rows)
        _write_csv(stage_root / "template_manifest.csv", MANIFEST_FIELDS, manifest_rows)
        shutil.copyfile(roi_config, stage_root / "roi_config.json")
        (stage_root / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        artifacts = (
            Path("physical_part_splits.csv"),
            Path("template_manifest.csv"),
            Path("roi_config.json"),
            Path("summary.json"),
        )
        _write_csv(
            stage_root / "sha256_receipts.csv",
            RECEIPT_FIELDS,
            [{"relative_path": path.as_posix(), "sha256": _sha256(stage_root / path)} for path in artifacts],
        )
        validate_template_release(stage_root, published_output_root=output_root)
        os.replace(stage_root, output_root)
    except BaseException:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise
    validate_template_release(output_root)
    return summary


def validate_template_release(
    output_root: Path,
    *,
    published_output_root: Path | None = None,
) -> dict[str, Any]:
    """Fail closed on any release data, role, symlink, or receipt violation."""
    root = Path(output_root).resolve()
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_name = summary.get("source_name")
    expected_parts = summary.get("expected_parts")
    counts = summary.get("role_counts")
    if (
        source_name != "zs32_0727"
        or expected_parts != ZS32_0727_EXPECTED_PARTS
        or summary.get("seed") != ZS32_0727_SEED
        or counts != ZS32_0727_ROLE_COUNTS
    ):
        raise ValueError(
            "zs32_0727 requires the fixed 19-part seed-42 split "
            "train=11/model_val=3/calibration=3/final_test=2"
        )

    split_rows = _read_csv(root / "physical_part_splits.csv")
    split_by_part = {row.get("physical_part_id", ""): row.get("role", "") for row in split_rows}
    if "" in split_by_part or len(split_by_part) != expected_parts or Counter(split_by_part.values()) != counts:
        raise ValueError("physical_part_splits.csv is not the expected explicit role partition")
    if any(row.get("seed") != str(ZS32_0727_SEED) for row in split_rows):
        raise ValueError("physical_part_splits.csv does not preserve fixed seed 42")

    manifest = _read_csv(root / "template_manifest.csv")
    if len(manifest) != expected_parts * len(VIEWS):
        raise ValueError("template_manifest.csv does not contain the expected number of rows")
    roles_by_part: dict[str, set[str]] = defaultdict(set)
    views_by_part: dict[str, set[str]] = defaultdict(set)
    seen_hashes: set[str] = set()
    published_root = Path(published_output_root).resolve() if published_output_root is not None else root
    for number, row in enumerate(manifest, start=2):
        context = f"template_manifest.csv:row {number}"
        part_id = _required(row, "physical_part_id", context=context)
        role = _required(row, "role", context=context)
        if row.get("split", "") != role or split_by_part.get(part_id) != role:
            raise ValueError(f"part-role leakage detected: {part_id!r}")
        if (
            row.get("part_id", "") != part_id
            or row.get("source_name", "") != source_name
            or row.get("hand", "").lower() != "right"
            or row.get("label", "").lower() != "normal"
        ):
            raise ValueError(f"{context} violates the Template source contract")
        group_id = _enforce_source_provenance(row, source_name=source_name, context=context)
        if row.get("group_id", "") != group_id:
            raise ValueError(f"{context} group identity does not match source provenance")
        view = _required(row, "view", context=context)
        if view not in VIEWS or row.get("resolved_view", "") != view:
            raise ValueError(f"{context} has invalid canonical view")
        published_image = Path(_required(row, "image_path", context=context))
        try:
            image = root / Path(os.path.abspath(published_image)).relative_to(published_root)
        except ValueError as error:
            raise ValueError(f"{context} image_path is outside the published release root") from error
        if not image.is_symlink() or not image.is_file():
            raise ValueError(f"{context} image asset is not a live symlink")
        try:
            image.resolve().relative_to((REPO_ROOT / ZS32_0727_TEMPLATE_ROI_ROOT).resolve())
        except ValueError as error:
            raise ValueError(f"{context} image asset is outside the authoritative 0727 ROI release") from error
        if _sha256(image) != _required(row, "content_sha256", context=context):
            raise ValueError(f"{context} encoded asset SHA256 mismatch")
        content_sha256 = row["content_sha256"]
        if content_sha256 in seen_hashes:
            raise ValueError("encoded image is duplicated in template_manifest.csv")
        seen_hashes.add(content_sha256)
        roles_by_part[part_id].add(role)
        views_by_part[part_id].add(view)
    if set(roles_by_part) != set(split_by_part) or any(len(roles) != 1 for roles in roles_by_part.values()):
        raise ValueError("part-role leakage detected in template_manifest.csv")
    if any(views != set(VIEWS) for views in views_by_part.values()):
        raise ValueError("each physical part must contain all eight canonical views")
    roi_sha256 = _sha256(root / "roi_config.json")
    if (
        roi_sha256 != ZS32_0727_ROI_CONFIG_SHA256
        or roi_sha256 != summary.get("input_sha256", {}).get("roi_config")
    ):
        raise ValueError("copied ROI config SHA256 does not match the input receipt")
    for receipt in _read_csv(root / "sha256_receipts.csv"):
        artifact = root / _required(receipt, "relative_path", context="sha256_receipts.csv")
        if not artifact.is_file() or _sha256(artifact) != _required(receipt, "sha256", context="sha256_receipts.csv"):
            raise ValueError(f"SHA256 receipt mismatch: {artifact}")
    return {"status": "VALID", "physical_part_count": len(split_by_part), "template_rows": len(manifest)}


def build_parser() -> argparse.ArgumentParser:
    """Build the prepare/validate command line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="build a new immutable Template release")
    prepare.add_argument("--crop-manifest", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--source-name", required=True)
    prepare.add_argument("--expected-parts", type=int, required=True)
    prepare.add_argument("--train-count", type=int, required=True)
    prepare.add_argument("--model-val-count", type=int, required=True)
    prepare.add_argument("--calibration-count", type=int, required=True)
    prepare.add_argument("--final-test-count", type=int, required=True)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--dry-run", action="store_true")
    validate = commands.add_parser("validate", help="validate an immutable Template release")
    validate.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run a release preparation or validation command."""
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_template_release(
            crop_manifest=args.crop_manifest,
            output_root=args.output_root,
            source_name=args.source_name,
            expected_parts=args.expected_parts,
            train_count=args.train_count,
            model_val_count=args.model_val_count,
            calibration_count=args.calibration_count,
            final_test_count=args.final_test_count,
            seed=args.seed,
            dry_run=args.dry_run,
        )
    else:
        result = validate_template_release(args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
