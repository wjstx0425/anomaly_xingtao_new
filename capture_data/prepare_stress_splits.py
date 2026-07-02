# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Prepare group-locked stress-normal splits for C789 robustness validation."""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path


IMAGE_EXTENSIONS = (".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff")
NORMAL_LABELS = ("normal", "normal_test")


def _iter_normal_images(view_root: Path) -> Iterable[Path]:
    """Yield stress-normal crops from normal-like label directories."""
    for label in NORMAL_LABELS:
        label_dir = view_root / label
        if not label_dir.is_dir():
            continue
        yield from sorted(path for path in label_dir.glob("**/images/*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        yield from sorted(
            path for path in label_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )


def _sample_id_from_path(path: Path) -> str:
    """Return the nested sample id for one Folder-style image."""
    if path.parent.name == "images":
        return path.parent.parent.name
    return path.stem


def _source_id_from_sample(sample_id: str) -> str:
    """Remove slot suffix from a crop sample id."""
    return re.sub(r"_slot[0-9]+$", "", sample_id)


def group_key_from_path(path: Path) -> str:
    """Return a group split key parsed from a stress crop filename."""
    match = re.search(r"_g(?P<group>[0-9]+)_", path.stem)
    if match is None:
        msg = f"Could not parse group id from filename: {path.name}"
        raise ValueError(msg)
    source_id = _source_id_from_sample(_sample_id_from_path(path))
    return f"{source_id}_g{int(match.group('group')):03d}"


def _slot_from_path(path: Path) -> str:
    """Return slot name from a stress crop path, or an empty string."""
    match = re.search(r"_slot(?P<slot>[0-9]+)", str(path))
    return f"slot{int(match.group('slot')):02d}" if match else ""


def _copy_or_link(source: Path, destination: Path, link_mode: str, overwrite: bool) -> None:
    """Materialize one image via copy, hardlink, or symlink."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not overwrite:
            msg = f"Destination already exists: {destination}"
            raise FileExistsError(msg)
        destination.unlink()

    if link_mode == "copy":
        shutil.copy2(source, destination)
        return
    if link_mode == "hardlink":
        destination.hardlink_to(source)
        return
    if link_mode == "symlink":
        destination.symlink_to(source.resolve())
        return
    msg = f"Unsupported link mode: {link_mode}"
    raise ValueError(msg)


def _assign_group_splits(group_keys: Sequence[str], train_ratio: float, seed: int) -> dict[str, str]:
    """Assign group keys to train or locked splits."""
    if not 0 < train_ratio < 1:
        msg = "--train-ratio must be > 0 and < 1."
        raise ValueError(msg)
    keys = sorted(set(group_keys))
    if len(keys) < 2:
        msg = "At least two stress groups are required for a train/locked split."
        raise ValueError(msg)
    train_count = int(len(keys) * train_ratio + 0.5)
    train_count = min(max(1, train_count), len(keys) - 1)
    shuffled = list(keys)
    random.Random(seed).shuffle(shuffled)
    train_keys = set(shuffled[:train_count])
    return {key: ("train" if key in train_keys else "locked") for key in keys}


def split_stress_normal_dataset(
    input_root: Path,
    output_root: Path,
    hand: str,
    position: str,
    train_ratio: float,
    seed: int,
    link_mode: str,
    overwrite: bool,
) -> Path:
    """Split an already-cropped stress-normal dataset by source group."""
    input_view_root = input_root.resolve() / hand / position
    image_paths = list(_iter_normal_images(input_view_root))
    if not image_paths:
        msg = f"No stress-normal images found under {input_view_root}"
        raise RuntimeError(msg)

    group_keys = [group_key_from_path(path) for path in image_paths]
    splits = _assign_group_splits(group_keys, train_ratio, seed)
    output_root = output_root.resolve()
    manifest_rows = []

    for image_path, group_key in zip(image_paths, group_keys, strict=True):
        split = splits[group_key]
        sample_id = _sample_id_from_path(image_path)
        relative_name = image_path.name
        destination = output_root / split / hand / position / "normal" / sample_id / "images" / relative_name
        _copy_or_link(image_path, destination, link_mode, overwrite)
        manifest_rows.append(
            {
                "source_path": str(image_path.resolve()),
                "processed_path": str(destination.absolute()),
                "split": split,
                "hand": hand,
                "position": position,
                "label": "normal",
                "group_key": group_key,
                "slot": _slot_from_path(image_path),
                "link_mode": link_mode,
            },
        )

    manifest_path = output_root / "stress_group_split_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-root", type=Path, required=True, help="Cropped stress-normal dataset root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output root for train/locked stress splits.")
    parser.add_argument("--hand", default="left", help="View hand directory.")
    parser.add_argument("--position", default="top", help="View position directory, e.g. top or bottom_ZS32.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Fraction of groups assigned to stress train.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for group assignment.")
    parser.add_argument(
        "--link-mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="How to materialize output images.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output images.")
    return parser


def main() -> None:
    """Run group-locked stress-normal splitting."""
    args = build_parser().parse_args()
    manifest_path = split_stress_normal_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        hand=args.hand,
        position=args.position,
        train_ratio=args.train_ratio,
        seed=args.seed,
        link_mode=args.link_mode,
        overwrite=args.overwrite,
    )
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
