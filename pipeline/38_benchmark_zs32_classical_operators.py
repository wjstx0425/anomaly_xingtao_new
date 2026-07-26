#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run the standalone ZS32 classical-operator diagnostic benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capture_data.zs32_classical_benchmark import (  # noqa: E402
    PRIMARY_VIEWS,
    BenchmarkConfig,
    run_benchmark,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the diagnostic-only command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Input ROI crop_manifest.csv.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New immutable output generation.")
    parser.add_argument(
        "--view",
        action="append",
        choices=PRIMARY_VIEWS,
        help="Primary view to include; repeat as needed. Defaults to all six.",
    )
    parser.add_argument("--max-normal-per-view", type=int)
    parser.add_argument("--max-defect-per-view", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resize-scale", type=float, default=1.0)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--timing-rounds", type=int, default=30)
    parser.add_argument("--timing-samples-per-view", type=int, default=1)
    parser.add_argument("--parallel-workers", type=int, default=2)
    parser.add_argument(
        "--save-overlays",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write diagnostic overlays in addition to binary masks.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    """Map parsed CLI arguments to a standalone benchmark configuration."""
    return BenchmarkConfig(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        views=tuple(args.view) if args.view else PRIMARY_VIEWS,
        max_normal_per_view=args.max_normal_per_view,
        max_defect_per_view=args.max_defect_per_view,
        seed=args.seed,
        resize_scale=args.resize_scale,
        warmup_rounds=args.warmup_rounds,
        timing_rounds=args.timing_rounds,
        timing_samples_per_view=args.timing_samples_per_view,
        parallel_workers=args.parallel_workers,
        save_overlays=args.save_overlays,
    )


def main() -> int:
    """Run the benchmark and print its immutable generation path."""
    output = run_benchmark(config_from_args(build_parser().parse_args()))
    print(f"Wrote standalone diagnostic benchmark: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
