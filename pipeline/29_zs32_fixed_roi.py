# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage 29: select six fixed ZS32 ROIs and crop the YOLO dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capture_data.zs32_view_roi_dataset import crop_zs32_yolo_dataset, select_view_rois  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the stage-29 command line parser."""
    dataset_root = REPO_ROOT / "dataset"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    select_parser = subparsers.add_parser("select", help="Open six OpenCV windows and save the fixed ROI config.")
    select_parser.add_argument("--input-root", type=Path, default=dataset_root / "zs32_six_view_yolo")
    select_parser.add_argument("--config", type=Path, default=dataset_root / "zs32_six_view_roi_config.json")
    select_parser.add_argument("--preview-dir", type=Path, default=dataset_root / "zs32_six_view_roi_previews")
    select_parser.add_argument("--max-window-width", type=int, default=1600)
    select_parser.add_argument("--max-window-height", type=int, default=1000)

    convert_parser = subparsers.add_parser("convert", help="Crop images and transform YOLO labels using the config.")
    convert_parser.add_argument("--input-root", type=Path, default=dataset_root / "zs32_six_view_yolo")
    convert_parser.add_argument("--config", type=Path, default=dataset_root / "zs32_six_view_roi_config.json")
    convert_parser.add_argument("--output-root", type=Path, default=dataset_root / "zs32_six_view_roi_yolo")
    convert_parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    """Run ROI selection or dataset conversion."""
    args = build_parser().parse_args()
    if args.command == "select":
        payload = select_view_rois(
            repo_root=REPO_ROOT,
            input_root=args.input_root,
            config_path=args.config,
            preview_dir=args.preview_dir,
            max_window_width=args.max_window_width,
            max_window_height=args.max_window_height,
        )
        print(f"ROI config: {args.config}")
        for view, values in payload["views"].items():
            print(f"  {view}: {values['roi']}")
        return

    summary = crop_zs32_yolo_dataset(
        repo_root=REPO_ROOT,
        input_root=args.input_root,
        output_root=args.output_root,
        roi_config=args.config,
        overwrite=args.overwrite,
    )
    print(f"ROI YOLO dataset: {args.output_root}")
    print(f"data.yaml: {args.output_root / 'data.yaml'}")
    print(f"Boxes clipped at ROI boundary: {summary['clipped_boxes']}")
    print(f"Boxes dropped outside ROI: {summary['dropped_boxes']}")
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
