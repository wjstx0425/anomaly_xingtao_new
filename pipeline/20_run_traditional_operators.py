# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage 20: run C789 traditional operator branches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capture_data import traditional_operators as traditional  # noqa: E402
from capture_data.prepare_part_crops import PRESETS  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the traditional-operator CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-root", type=Path, required=True, help="Full-capture or slot-crop image root.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="c789_left_top_3x2",
        help="C789 crop preset used for full-capture inputs.",
    )
    parser.add_argument("--side", choices=("top", "bottom"), required=True, help="Inspection side label.")
    parser.add_argument("--view", default="uniform", help="Capture view/light label.")
    parser.add_argument(
        "--input-mode",
        choices=("auto", "full", "slot"),
        default="auto",
        help="Whether input images are full fixture captures or pre-cropped slots.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/traditional/c789.yaml"),
        help="Traditional-operator threshold config.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for branch CSV and evidence.")
    parser.add_argument("--max-images", type=int, help="Optional smoke-test image limit.")
    parser.add_argument(
        "--calibrate-normal-root",
        type=Path,
        help="Trusted normal slot-crop root used to build slot templates and per-slot thresholds.",
    )
    parser.add_argument(
        "--calibration-max-images-per-slot",
        type=int,
        help="Maximum normal images per slot used during calibration; 0 means all available images.",
    )
    parser.add_argument(
        "--template-dir",
        type=Path,
        help="Optional prebuilt slot template directory created by stage 20 calibration or stage 11.",
    )
    parser.add_argument(
        "--geometry-thresholds",
        type=Path,
        help="Optional geometry threshold CSV used with --template-dir.",
    )
    parser.add_argument(
        "--label",
        choices=("auto", "normal", "defect", "invalid", "unknown"),
        default="auto",
        help="Ground-truth label for all inputs, or infer from normal/defect path parts.",
    )
    parser.add_argument(
        "--no-progress",
        dest="show_progress",
        action="store_false",
        help="Disable per-image progress logging.",
    )
    parser.add_argument(
        "--no-evidence",
        dest="save_evidence",
        action="store_false",
        help="Disable evidence overlay image writing.",
    )
    parser.set_defaults(save_evidence=True)
    parser.set_defaults(show_progress=True)
    return parser


def main() -> None:
    """Run the traditional operator branch stage."""
    args = build_parser().parse_args()
    results = traditional.run_traditional_operators(args)
    output_csv = args.output_dir / "traditional_predictions.csv"
    positives = sum(result.pred_label for result in results)
    print(f"Wrote traditional predictions: {output_csv}")
    print(f"Wrote traditional summary: {args.output_dir / 'traditional_summary.md'}")
    print(f"Rows: {len(results)}; positives: {positives}")


if __name__ == "__main__":
    main()
