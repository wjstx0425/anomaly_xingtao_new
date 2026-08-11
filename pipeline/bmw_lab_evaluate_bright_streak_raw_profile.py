#!/usr/bin/env python3
"""Evaluate the offline BMW raw-grayscale bright-streak v2 candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bmw_inspection.lab.bright_streak_raw_profile import (  # noqa: E402
    evaluate_bright_streak_raw_profile,
)


DEFAULT_MANIFEST = (
    REPO_ROOT
    / "dataset/bmw_lab_prepared/bmw_right_batch_20260810_21_v1/manifests/bright_streak.csv"
)
DEFAULT_BASE_CONFIG = (
    REPO_ROOT
    / "results/bmw_lab_one_click/bmw_right_multisource_left_yolo_v1/"
    "bright_streak_right_ridge_v1_bold/calibrated_config.json"
)
DEFAULT_CORRECTED_ROI_XYXY = [1792, 1180, 1873, 1793]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results/bmw_lab_one_click/bmw_right_batch_20260810_21_bright_v2_roi_corrected"
)


def build_parser() -> argparse.ArgumentParser:
    """Build the calibration-only, no-overwrite offline evaluation parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--roi-xyxy",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=DEFAULT_CORRECTED_ROI_XYXY,
        help="21点批次光痕ROI；默认已向左修正80像素并保持81x613。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one immutable raw-profile v2 comparison report."""
    args = build_parser().parse_args(argv)
    try:
        report = evaluate_bright_streak_raw_profile(
            args.manifest,
            args.base_config,
            args.output_dir,
            roi_xyxy=tuple(args.roi_xyxy),
        )
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(
            f"BMW raw-profile bright-streak evaluation failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
