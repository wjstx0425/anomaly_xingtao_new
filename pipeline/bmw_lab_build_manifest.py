#!/usr/bin/env python3
"""Build explicit BMW physical-part manifests and model-training exports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bmw_inspection.lab.config import load_experiment_config  # noqa: E402
from bmw_inspection.lab.dataset import build_dataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit source-manifest dataset CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True, help="Explicit BMW six-view source CSV.")
    parser.add_argument("--dataset-id", required=True, help="Path-safe dataset release identity.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/bmw/experiments/bmw_lab_v1.json",
        help="Experiment profile supplying the deployment part ROI for every view.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "dataset/bmw_lab",
        help="New dataset release root; existing roots are never overwritten.",
    )
    parser.add_argument(
        "--bright-streak-root",
        type=Path,
        default=REPO_ROOT / "dataset/bmw",
        help="Fixed legacy root containing exactly 6 OK and 7 NG BMP regression images.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic physical-part split seed.")
    parser.add_argument(
        "--experimental-small-data",
        action="store_true",
        help="Allow fewer than 30 complete parts and mark the report experimental-only.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the report without writing outputs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build or dry-run one BMW dataset release."""
    args = build_parser().parse_args(argv)
    config = load_experiment_config(args.config)
    report = build_dataset(
        source_manifest=args.source_manifest,
        output_root=args.output_root,
        dataset_id=args.dataset_id,
        bright_streak_root=args.bright_streak_root,
        part_rois=config.part_rois,
        seed=args.seed,
        experimental_small_data=args.experimental_small_data,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
