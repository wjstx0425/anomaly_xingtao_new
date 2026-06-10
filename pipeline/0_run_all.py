# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run the numbered defect-detection pipeline stages in order."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Stage:
    """One pipeline stage invocation."""

    name: str
    script: str
    args: list[str]


def split_args(value: str) -> list[str]:
    """Split a shell-like argument string without invoking a shell."""
    try:
        return shlex.split(value)
    except ValueError as error:
        msg = f"Could not parse stage arguments {value!r}: {error}"
        raise argparse.ArgumentTypeError(msg) from error


def build_stages(args: argparse.Namespace) -> list[Stage]:
    """Build stage invocations from parsed CLI arguments."""
    stages: list[Stage] = []
    for collect_args in args.collect:
        stages.append(Stage("1_collect_data", "pipeline/1_collect_data.py", split_args(collect_args)))

    if args.process:
        stages.append(Stage("2_process_data", "pipeline/2_process_data.py", split_args(args.process)))
    if args.train:
        stages.append(Stage("3_train_model", "pipeline/3_train_model.py", split_args(args.train)))
    if args.inference:
        stages.append(Stage("4_inference", "pipeline/4_inference.py", split_args(args.inference)))
    return stages


def run_stage(stage: Stage, dry_run: bool) -> int:
    """Run one stage and return its process code."""
    script_path = REPO_ROOT / stage.script
    command = [sys.executable, str(script_path), *stage.args]
    print()
    print(f"=== {stage.name} ===")
    print(" ".join(shlex.quote(part) for part in command))
    if dry_run:
        return 0
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Examples:
  .venv/bin/python pipeline/0_run_all.py \
    --process "auto --data-root dataset/c789 ..." \
    --train "--data-root dataset/c789_left_top_parts --models anomaly_dino ..." \
    --inference "dataset/c789_left_top_parts/left/top --model anomaly_dino ..."

  # Collection can be repeated, for example normal then defect:
  .venv/bin/python pipeline/0_run_all.py \
    --collect "--hand no_hand --position top --label normal ..." \
    --collect "--hand no_hand --position top --label defect ..."
""",
    )
    parser.add_argument(
        "--collect",
        action="append",
        default=[],
        metavar="ARGS",
        help="Arguments for pipeline/1_collect_data.py. Can be repeated.",
    )
    parser.add_argument("--process", metavar="ARGS", help="Arguments for pipeline/2_process_data.py.")
    parser.add_argument("--train", metavar="ARGS", help="Arguments for pipeline/3_train_model.py.")
    parser.add_argument("--inference", metavar="ARGS", help="Arguments for pipeline/4_inference.py.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running later stages after a stage fails.",
    )
    return parser


def main() -> None:
    """Run selected pipeline stages."""
    args = build_parser().parse_args()
    stages = build_stages(args)
    if not stages:
        build_parser().print_help()
        raise SystemExit(2)

    failed: list[tuple[str, int]] = []
    for stage in stages:
        code = run_stage(stage, args.dry_run)
        if code:
            failed.append((stage.name, code))
            if not args.continue_on_error:
                raise SystemExit(code)

    if failed:
        print()
        print("Failed stages:")
        for name, code in failed:
            print(f"  {name}: exit code {code}")
        raise SystemExit(failed[-1][1])


if __name__ == "__main__":
    main()
