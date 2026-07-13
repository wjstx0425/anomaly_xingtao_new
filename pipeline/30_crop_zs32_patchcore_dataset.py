# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage 30: crop right/left six-view images for PatchCore retraining."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capture_data.zs32_patchcore_roi_dataset import (  # noqa: E402
    HANDS,
    VIEWS,
    crop_patchcore_dataset,
    select_patchcore_rois,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the stage-30 command line parser."""
    dataset_root = REPO_ROOT / "dataset"
    config_path = dataset_root / "zs32_patchcore_roi_config.json"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    select_parser = subparsers.add_parser("select", help="Select 12 independent right/left view ROIs.")
    select_parser.add_argument("--dataset-root", type=Path, default=dataset_root)
    select_parser.add_argument("--config", type=Path, default=config_path)
    select_parser.add_argument("--preview-dir", type=Path, default=dataset_root / "zs32_patchcore_roi_previews")
    select_parser.add_argument("--max-window-width", type=int, default=1600)
    select_parser.add_argument("--max-window-height", type=int, default=1000)

    convert_parser = subparsers.add_parser("convert", help="Crop both hand datasets using the saved ROIs.")
    convert_parser.add_argument("--dataset-root", type=Path, default=dataset_root)
    convert_parser.add_argument("--config", type=Path, default=config_path)
    convert_parser.add_argument("--output-root", type=Path, default=dataset_root / "zs32_patchcore_roi")
    convert_parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run ROI selection or crop conversion."""
    args = build_parser().parse_args(argv)
    if args.command == "select":
        payload = select_patchcore_rois(
            repo_root=REPO_ROOT,
            dataset_root=args.dataset_root,
            config_path=args.config,
            preview_dir=args.preview_dir,
            max_window_width=args.max_window_width,
            max_window_height=args.max_window_height,
        )
        hands = cast("dict[str, dict[str, dict[str, dict[str, list[int]]]]]", payload["hands"])
        print(f"ROI config: {args.config}")
        for hand in HANDS:
            for view in VIEWS:
                print(f"  {hand}/{view}: {hands[hand]['views'][view]['roi']}")
        return

    result = crop_patchcore_dataset(
        repo_root=REPO_ROOT,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        roi_config=args.config,
        overwrite=args.overwrite,
    )
    print(f"PatchCore ROI dataset: {result['output_root']}")
    print(f"Manifest: {result['manifest_path']}")
    print(f"Images: {result['total_images']} -> {result['output_images']}")
    print(f"Corrected views: {result['corrected_views']}")
    if result.get("cleanup_warning"):
        print(f"WARNING: {result['cleanup_warning']}")


if __name__ == "__main__":
    main()
