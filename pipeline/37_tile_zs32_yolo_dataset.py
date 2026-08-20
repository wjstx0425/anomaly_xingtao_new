# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Stage 37: build and run the minimal ZS32 tiled YOLO workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capture_data.tile_zs32_yolo_dataset import (  # noqa: E402
    build_tiled_zs32_yolo_dataset,
    load_ultralytics_model,
    run_tiled_inference,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the Stage 37 command parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build the compact tiled YOLO dataset.")
    build.add_argument("--input-root", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--tile-size", type=int, default=1280)
    build.add_argument("--stride", type=int, default=960)
    build.add_argument("--min-visible-ratio", type=float, default=0.8)
    build.add_argument("--normal-tiles-per-image", type=int, default=1)
    build.add_argument("--padding-value", type=int, default=114)
    build.add_argument("--seed", type=int, default=42)

    infer = subparsers.add_parser("infer", help="Run tiled inference and merge detections in source ROI coordinates.")
    infer.add_argument("--model", type=Path, required=True)
    infer.add_argument("--source", type=Path, required=True)
    infer.add_argument("--output-dir", type=Path, required=True)
    infer.add_argument("--tile-size", type=int, default=1280)
    infer.add_argument("--stride", type=int, default=960)
    infer.add_argument("--conf", type=float, default=0.01)
    infer.add_argument("--tile-iou", type=float, default=0.7)
    infer.add_argument("--merge-iou", type=float, default=0.5)
    infer.add_argument("--batch", type=int, default=16)
    infer.add_argument("--device", default="0")
    infer.add_argument("--padding-value", type=int, default=114)
    return parser


def main() -> None:
    """Run Stage 37."""
    args = build_parser().parse_args()
    if args.command == "build":
        summary = build_tiled_zs32_yolo_dataset(
            input_root=args.input_root,
            output_root=args.output_root,
            tile_size=args.tile_size,
            stride=args.stride,
            min_visible_ratio=args.min_visible_ratio,
            normal_tiles_per_image=args.normal_tiles_per_image,
            padding_value=args.padding_value,
            seed=args.seed,
        )
    else:
        model = load_ultralytics_model(args.model)
        summary = run_tiled_inference(
            model=model,
            source=args.source,
            output_dir=args.output_dir,
            tile_size=args.tile_size,
            stride=args.stride,
            confidence=args.conf,
            tile_iou=args.tile_iou,
            merge_iou=args.merge_iou,
            batch=args.batch,
            device=args.device,
            padding_value=args.padding_value,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
