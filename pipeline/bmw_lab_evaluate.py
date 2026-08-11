#!/usr/bin/env python3
"""Evaluate one BMW manifest split across one or more resident profiles."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from bmw_inspection.lab.config import LabExperimentConfig, load_experiment_config  # noqa: E402
from bmw_inspection.lab.evaluation import (  # noqa: E402
    OfflineEvaluationRuntime,
    build_offline_runtime,
    evaluate_profiles,
)


RuntimeFactory = Callable[[LabExperimentConfig], OfflineEvaluationRuntime]


def build_parser() -> argparse.ArgumentParser:
    """Build the split-safe offline comparison interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("calibration", "final_test"), required=True)
    profiles = parser.add_mutually_exclusive_group(required=True)
    profiles.add_argument("--profile", type=Path)
    profiles.add_argument("--compare", type=Path, nargs="+")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--write-thresholds", action="store_true")
    return parser


def _default_runtime_factory(config: LabExperimentConfig) -> OfflineEvaluationRuntime:
    return build_offline_runtime(config)


def main(
    argv: Sequence[str] | None = None,
    *,
    runtime_factory: RuntimeFactory | None = None,
) -> int:
    """Load each profile once and publish one atomic batch evaluation."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.split == "final_test" and args.write_thresholds:
        parser.error("final_test is read-only; --write-thresholds is forbidden")
    profile_paths = [args.profile] if args.profile is not None else list(args.compare)
    if args.compare is not None and len(profile_paths) < 2:
        parser.error("--compare requires at least two experiment profiles")
    factory = runtime_factory or _default_runtime_factory
    try:
        runtimes: dict[str, OfflineEvaluationRuntime] = {}
        for profile_path in profile_paths:
            config = load_experiment_config(profile_path)
            if config.experiment_id in runtimes:
                raise ValueError(f"duplicate experiment_id in comparison: {config.experiment_id}")
            runtimes[config.experiment_id] = factory(config)
        summary = evaluate_profiles(
            manifest_path=args.manifest,
            profiles=runtimes,
            split=args.split,
            output_root=args.output_root,
            write_thresholds=args.write_thresholds,
        )
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"BMW evaluation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
