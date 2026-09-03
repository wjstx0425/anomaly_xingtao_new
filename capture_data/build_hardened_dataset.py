# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Build a clean-plus-stress-train Folder dataset for robustness training."""

from __future__ import annotations

import argparse
import csv
import shutil
from collections.abc import Iterable
from pathlib import Path


IMAGE_EXTENSIONS = (".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff")
LABELS = ("normal", "normal_test", "defect")


def _iter_label_images(label_dir: Path) -> Iterable[Path]:
    """Yield flat or nested Folder-style images below a label directory."""
    if not label_dir.is_dir():
        return
    yield from sorted(path for path in label_dir.glob("**/images/*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    yield from sorted(path for path in label_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _sample_id_from_path(path: Path) -> str:
    """Return the nested sample id for one Folder image."""
    if path.parent.name == "images":
        return path.parent.parent.name
    return path.stem


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


def _write_image_row(
    *,
    source_path: Path,
    destination_root: Path,
    hand: str,
    position: str,
    label: str,
    source_split: str,
    sample_prefix: str,
    link_mode: str,
    overwrite: bool,
) -> dict[str, str]:
    """Copy/link one image and return a manifest row."""
    sample_id = f"{sample_prefix}{_sample_id_from_path(source_path)}"
    destination = destination_root / hand / position / label / sample_id / "images" / source_path.name
    _copy_or_link(source_path, destination, link_mode, overwrite)
    return {
        "source_path": str(source_path.resolve()),
        "processed_path": str(destination.absolute()),
        "source_split": source_split,
        "hand": hand,
        "position": position,
        "label": label,
        "sample_id": sample_id,
        "link_mode": link_mode,
    }


def build_hardened_dataset(
    clean_root: Path,
    stress_split_root: Path,
    output_root: Path,
    hand: str,
    position: str,
    link_mode: str,
    overwrite: bool,
) -> Path:
    """Build a workflow-readable dataset from clean data and stress-train normal."""
    clean_view_root = clean_root.resolve() / hand / position
    stress_train_normal_root = stress_split_root.resolve() / "train" / hand / position / "normal"
    output_root = output_root.resolve()
    manifest_rows: list[dict[str, str]] = []

    for label in LABELS:
        for source_path in _iter_label_images(clean_view_root / label):
            manifest_rows.append(
                _write_image_row(
                    source_path=source_path,
                    destination_root=output_root,
                    hand=hand,
                    position=position,
                    label=label,
                    source_split="clean",
                    sample_prefix="",
                    link_mode=link_mode,
                    overwrite=overwrite,
                ),
            )

    for source_path in _iter_label_images(stress_train_normal_root):
        manifest_rows.append(
            _write_image_row(
                source_path=source_path,
                destination_root=output_root,
                hand=hand,
                position=position,
                label="normal",
                source_split="stress_train",
                sample_prefix="stress_",
                link_mode=link_mode,
                overwrite=overwrite,
            ),
        )

    if not manifest_rows:
        msg = "No clean or stress-train images were written."
        raise RuntimeError(msg)

    manifest_path = output_root / "hardened_dataset_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--clean-root", type=Path, required=True, help="Clean cropped Folder dataset root.")
    parser.add_argument("--stress-split-root", type=Path, required=True, help="Output root from prepare_stress_splits.py.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output hardened Folder dataset root.")
    parser.add_argument("--hand", default="left", help="View hand directory.")
    parser.add_argument("--position", default="top", help="View position directory, e.g. top or bottom.")
    parser.add_argument(
        "--link-mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="How to materialize output images.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output images.")
    return parser


def main() -> None:
    """Build a hardened clean-plus-stress training dataset."""
    args = build_parser().parse_args()
    manifest_path = build_hardened_dataset(
        clean_root=args.clean_root,
        stress_split_root=args.stress_split_root,
        output_root=args.output_root,
        hand=args.hand,
        position=args.position,
        link_mode=args.link_mode,
        overwrite=args.overwrite,
    )
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
