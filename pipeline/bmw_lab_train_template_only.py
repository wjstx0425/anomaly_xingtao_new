#!/usr/bin/env python3
"""Train selected BMW Template views while keeping an existing threshold."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER  # noqa: E402
from bmw_inspection.lab.eight_view_train_all import (  # noqa: E402
    LabTrainingConfig,
    preflight_template_only,
    run_template_only_training,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the selected-view fixed-threshold Template CLI."""
    parser = argparse.ArgumentParser(
        description="只训练指定BMW视角的Template，并把给定旧阈值写入新模型。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--training-root", type=Path, default=REPO_ROOT / "dataset/bmw_lab_training")
    parser.add_argument("--training-id", required=True)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "results/bmw_lab_one_click")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--view",
        action="append",
        choices=VIEW_ORDER,
        required=True,
        help="只训练指定视角；可重复传入。",
    )
    parser.add_argument("--threshold", type=float, required=True, help="沿用的Template部署阈值。")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> LabTrainingConfig:
    """Create the shared path config for a Template-only run."""
    defaults = LabTrainingConfig.defaults(REPO_ROOT)
    return replace(
        defaults,
        training_root=args.training_root.expanduser().resolve(),
        training_id=args.training_id,
        output_root=args.output_root.expanduser().resolve(),
        run_id=args.run_id,
        views=tuple(args.view),
    )


def main(argv: list[str] | None = None) -> int:
    """Validate the selected release and train its sole Template stage."""
    args = build_parser().parse_args(argv)
    config = _config_from_args(args)
    try:
        preflight = preflight_template_only(config)
        training = run_template_only_training(
            config,
            threshold=args.threshold,
            dry_run=args.dry_run,
        )
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {"status": "failed", "error": f"{type(error).__name__}: {error}"},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps({"preflight": preflight, "training": training}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
