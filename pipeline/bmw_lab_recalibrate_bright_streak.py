#!/usr/bin/env python3
"""Recalibrate the BMW bright-streak rule without starting model training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bmw_inspection.lab.bright_streak_recalibration import recalibrate_bright_streak  # noqa: E402


DEFAULT_MANIFEST = REPO_ROOT / "dataset/bmw_lab_prepared/bmw_hdr_eight_view_v1/manifests/bright_streak.csv"
DEFAULT_BASE_CONFIG = (
    REPO_ROOT / "results/bmw_lab_one_click/bmw_lab_eight_view_v1/bright_streak/calibrated_config.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "results/bmw_lab_one_click/bmw_lab_eight_view_v1/bright_streak_ridge_v2"
)


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone bright-streak recalibration CLI.

    Returns:
        argparse.ArgumentParser: Parser with repository-relative input and output defaults.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run split-safe recalibration and print the published report.

    Args:
        argv (list[str] | None): Optional command-line arguments. Defaults to ``sys.argv``.

    Returns:
        int: Zero on success or two when validation, decoding, or artifact publication fails.
    """
    args = build_parser().parse_args(argv)
    try:
        report = recalibrate_bright_streak(
            args.manifest,
            args.base_config,
            args.output_dir,
        )
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"BMW bright-streak recalibration failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
