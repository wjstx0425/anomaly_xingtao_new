# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage 30: calibrate versioned ZS32 fusion thresholds offline."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capture_data.fusion_calibration import run_calibration  # noqa: E402


def _probability(value: str) -> float:
    """Parse a finite probability in the interval ``(0, 1]``."""
    try:
        parsed = float(value)
    except ValueError as exc:
        msg = f"expected a number in (0, 1], got {value!r}"
        raise argparse.ArgumentTypeError(msg) from exc
    if not math.isfinite(parsed) or not 0 < parsed <= 1:
        msg = f"expected a number in (0, 1], got {value!r}"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _nonempty(value: str) -> str:
    """Parse a non-empty split or view name."""
    parsed = value.strip()
    if not parsed:
        msg = "expected a non-empty value"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _required_group(value: str) -> tuple[str, str, str, str, str]:
    """Parse ``HAND:VIEW:BRANCH:MODEL_VERSION:ROI_VERSION``."""
    fields = tuple(item.strip() for item in value.split(":"))
    if len(fields) != 5 or any(not item for item in fields):
        msg = "expected HAND:VIEW:BRANCH:MODEL_VERSION:ROI_VERSION"
        raise argparse.ArgumentTypeError(msg)
    hand, view, branch, model_version, roi_version = fields
    return hand, view, branch, model_version, roi_version


def build_parser() -> argparse.ArgumentParser:
    """Build the offline ZS32 fusion calibration parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-csv", type=Path, required=True, help="Calibration branch-score CSV.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Offline calibration report directory.")
    parser.add_argument("--target-recall", type=_probability, default=1.0, help="Observed defect recall target.")
    parser.add_argument("--normal-quantile", type=_probability, default=0.995, help="Normal score quantile for T_high.")
    parser.add_argument("--fit-split", type=_nonempty, default="calibration", help="Split used only to fit thresholds.")
    parser.add_argument(
        "--evaluation-split",
        "--eval-split",
        dest="eval_split",
        type=_nonempty,
        default="test",
        help="Held-out split used only for metrics.",
    )
    parser.add_argument(
        "--required-view",
        type=_nonempty,
        action="append",
        default=None,
        metavar="VIEW",
        help="Required view; repeat to require multiple views.",
    )
    parser.add_argument(
        "--required-group",
        type=_required_group,
        action="append",
        default=[],
        metavar="HAND:VIEW:BRANCH:MODEL_VERSION:ROI_VERSION",
        help="Exact required threshold group; repeat to require multiple groups.",
    )
    return parser


def main() -> None:
    """Run offline calibration without modifying deployment configuration."""
    args = build_parser().parse_args()
    metrics = run_calibration(
        args.input_csv,
        args.output_dir,
        target_recall=args.target_recall,
        normal_quantile=args.normal_quantile,
        fit_split=args.fit_split,
        eval_split=args.eval_split,
        required_views=args.required_view,
        required_groups=args.required_group,
    )
    print(f"Wrote ZS32 calibration reports: {args.output_dir}")
    print(f"Physical parts: {metrics['overall']['part_count']}")


if __name__ == "__main__":
    main()
