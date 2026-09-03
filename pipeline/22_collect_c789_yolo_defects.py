# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Collect and crop C789 defect samples for YOLO bbox annotation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

from _common import repo_path


IMAGE_EXTENSIONS = (".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff")


@dataclass(frozen=True)
class C789PositionConfig:
    """C789 crop preset and output directory naming for one face."""

    preset: str
    output_position: str


C789_POSITION_CONFIGS = {
    "top": C789PositionConfig(preset="c789_left_top_3x2", output_position="top"),
    "bottom": C789PositionConfig(preset="c789_left_bottom_3x2", output_position="bottom"),
}


def position_config(position: str) -> C789PositionConfig:
    """Return the C789 crop config for a supported capture position."""
    try:
        return C789_POSITION_CONFIGS[position]
    except KeyError as error:
        msg = f"Unsupported C789 position: {position!r}. Expected one of {sorted(C789_POSITION_CONFIGS)}."
        raise ValueError(msg) from error


def build_collect_args(args: argparse.Namespace) -> list[str]:
    """Build arguments for ``capture_data/collect_dataset.py``."""
    collect_args = [
        "--hand",
        args.hand,
        "--position",
        args.position,
        "--label",
        "defect",
        "--defect-type",
        args.defect_type,
        "--part-id",
        args.part_id,
        "--group-count",
        str(args.group_count),
        "--images-per-group",
        str(args.images_per_group),
        "--device",
        str(args.device),
        "--gain",
        str(args.gain),
        "--root",
        str(args.raw_root),
    ]
    if args.manual_load:
        collect_args.append("--manual-load")
    if args.hdr:
        collect_args.extend(
            [
                "--hdr",
                "--short-exposure",
                str(args.short_exposure),
                "--long-exposure",
                str(args.long_exposure),
                "--short-dark-threshold",
                str(args.short_dark_threshold),
                "--long-clip-threshold",
                str(args.long_clip_threshold),
                "--blend-width",
                str(args.blend_width),
                "--blur-size",
                str(args.blur_size),
                "--hdr-settle-frames",
                str(args.hdr_settle_frames),
                "--hdr-max-retries",
                str(args.hdr_max_retries),
                "--hdr-max-clip-pct",
                str(args.hdr_max_clip_pct),
            ],
        )
        if args.save_hdr_sources:
            collect_args.append("--save-hdr-sources")
        if args.align_hdr:
            collect_args.append("--align-hdr")
    elif args.exposure is not None:
        collect_args.extend(["--exposure", str(args.exposure)])
    return collect_args


def build_crop_args(args: argparse.Namespace) -> list[str]:
    """Build arguments for ``capture_data/prepare_part_crops.py``."""
    config = position_config(args.position)
    crop_args = [
        "--data-root",
        str(args.raw_root),
        "--output-root",
        str(args.parts_root),
        "--hand",
        args.hand,
        "--position",
        args.position,
        "--output-hand",
        args.hand,
        "--output-position",
        config.output_position,
        "--labels",
        "defect",
        "--preset",
        config.preset,
        "--hole-mask-method",
        "none",
        "--defect-slot-mode",
        "all",
        "--nondefect-slots-from-defect-images",
        "skip",
    ]
    if args.overwrite:
        crop_args.append("--overwrite")
    return crop_args


def _iter_images(root: Path) -> Sequence[Path]:
    """Return sorted image files below ``root``."""
    return tuple(
        sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
    )


def _sample_id_for_crop(path: Path) -> str:
    """Infer the part-crop sample id from a nested crop path."""
    if path.parent.name == "images":
        return path.parent.parent.name
    return path.stem


def _slot_for_sample(sample_id: str) -> str:
    """Return the slot id embedded in a sample id, if present."""
    match = re.search(r"slot\d{2}", sample_id)
    return match.group(0) if match else ""


def _flat_output_path(
    source_path: Path,
    defect_root: Path,
    output_dir: Path,
    used_names: set[str],
    overwrite: bool,
) -> Path:
    """Return a collision-safe flat output path for one crop image."""
    candidate = output_dir / source_path.name
    if candidate.name not in used_names and (overwrite or not candidate.exists()):
        used_names.add(candidate.name)
        return candidate

    relative_key = source_path.relative_to(defect_root).as_posix()
    digest = hashlib.sha1(relative_key.encode("utf-8")).hexdigest()[:8]
    stem = source_path.stem
    suffix = source_path.suffix.lower()
    candidate = output_dir / f"{stem}_{digest}{suffix}"
    index = 1
    while candidate.name in used_names or (candidate.exists() and not overwrite):
        candidate = output_dir / f"{stem}_{digest}_{index:03d}{suffix}"
        index += 1
    used_names.add(candidate.name)
    return candidate


def copy_defect_crops(
    *,
    parts_root: Path,
    hand: str,
    output_position: str,
    defect_output_dir: Path,
    overwrite: bool,
) -> Path:
    """Copy nested defect crop images into one flat folder and write a manifest."""
    defect_root = parts_root / hand / output_position / "defect"
    if not defect_root.is_dir():
        msg = f"No processed defect crop directory found: {defect_root}"
        raise RuntimeError(msg)

    crop_paths = _iter_images(defect_root)
    if not crop_paths:
        msg = f"No processed defect crop images found under: {defect_root}"
        raise RuntimeError(msg)

    defect_output_dir.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    rows: list[dict[str, str]] = []
    for crop_path in crop_paths:
        flat_path = _flat_output_path(crop_path, defect_root, defect_output_dir, used_names, overwrite)
        shutil.copy2(crop_path, flat_path)
        sample_id = _sample_id_for_crop(crop_path)
        rows.append(
            {
                "flat_path": str(flat_path.resolve()),
                "crop_path": str(crop_path.resolve()),
                "sample_id": sample_id,
                "slot": _slot_for_sample(sample_id),
            },
        )

    manifest_path = defect_output_dir / "defect_image_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["flat_path", "crop_path", "sample_id", "slot"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def _run_script(relative_path: str, args: Sequence[str]) -> None:
    """Run one repo script and fail if it returns a non-zero exit code."""
    script = repo_path(relative_path)
    command = [sys.executable, str(script), *args]
    subprocess.run(command, cwd=repo_path("."), check=True)


def run_workflow(args: argparse.Namespace) -> Path:
    """Run collection, cropping, and flat defect-image export."""
    if not args.skip_collect:
        _run_script("capture_data/collect_dataset.py", build_collect_args(args))
    _run_script("capture_data/prepare_part_crops.py", build_crop_args(args))
    config = position_config(args.position)
    return copy_defect_crops(
        parts_root=args.parts_root,
        hand=args.hand,
        output_position=config.output_position,
        defect_output_dir=args.defect_output_dir,
        overwrite=args.overwrite,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the C789 YOLO defect collection parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--raw-root", type=Path, default=Path("dataset/c789_yolo_raw"), help="Raw full-image root.")
    parser.add_argument("--parts-root", type=Path, default=Path("dataset/c789_yolo_parts"), help="Cropped parts root.")
    parser.add_argument(
        "--defect-output-dir",
        type=Path,
        default=Path("dataset/c789_yolo_defect_images"),
        help="Flat folder for processed defect crop images.",
    )
    parser.add_argument("--hand", choices=("left", "right"), default="left", help="C789 hand directory to capture.")
    parser.add_argument(
        "--position",
        choices=sorted(C789_POSITION_CONFIGS),
        default="top",
        help="C789 face to capture.",
    )
    parser.add_argument("--defect-type", default="defect", help="Raw defect-type directory name.")
    parser.add_argument("--part-id", default="c789_yolo", help="Collection part or batch id.")
    parser.add_argument("--group-count", type=int, default=1, help="Number of manual capture groups.")
    parser.add_argument("--images-per-group", type=int, default=1, help="Images captured after each load.")
    parser.add_argument("--device", type=int, default=0, help="Camera device index.")
    parser.add_argument("--gain", type=float, default=0.0, help="Camera gain passed to collection.")
    parser.add_argument("--exposure", type=float, help="Single-exposure time when --no-hdr is used.")

    parser.add_argument(
        "--manual-load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pause before each group.",
    )
    parser.add_argument(
        "--hdr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture HDR fused images.",
    )
    parser.add_argument(
        "--save-hdr-sources",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save HDR source frames.",
    )
    parser.add_argument(
        "--align-hdr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Align HDR source frames.",
    )
    parser.add_argument("--short-exposure", type=float, default=4000.0, help="HDR short exposure.")
    parser.add_argument("--long-exposure", type=float, default=35000.0, help="HDR long exposure.")
    parser.add_argument("--short-dark-threshold", type=float, default=80.0, help="HDR short-image dark threshold.")
    parser.add_argument("--long-clip-threshold", type=float, default=245.0, help="HDR long-image clip threshold.")
    parser.add_argument("--blend-width", type=float, default=50.0, help="HDR fusion blend width.")
    parser.add_argument("--blur-size", type=int, default=101, help="HDR fusion blur size.")
    parser.add_argument("--hdr-settle-frames", type=int, default=8, help="Frames discarded after exposure changes.")
    parser.add_argument("--hdr-max-retries", type=int, default=2, help="HDR retries when the fused image clips.")
    parser.add_argument("--hdr-max-clip-pct", type=float, default=12.0, help="Max accepted HDR clipped-pixel percent.")
    parser.add_argument("--skip-collect", action="store_true", help="Only crop and flatten an existing raw dataset.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite crop filenames and flat output files.")
    return parser


def main() -> None:
    """Run the C789 YOLO defect collection workflow."""
    args = build_parser().parse_args()
    manifest_path = run_workflow(args)
    print(f"processed defect images: {args.defect_output_dir}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
