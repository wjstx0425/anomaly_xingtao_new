#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Train the BMW 21:00 ROI release with EfficientAD-S only."""

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

from bmw_inspection.lab.eight_view_train_all import (  # noqa: E402
    LabTrainingConfig,
    preflight_efficientad_only,
    run_efficientad_only_training,
)
from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the isolated 21:00 EfficientAD-S training CLI."""
    defaults = LabTrainingConfig.defaults(REPO_ROOT)
    parser = argparse.ArgumentParser(
        description="只训练已发布21点ROI release的8个EfficientAD-S；不materialize、不训练Template/光痕/YOLO。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--training-root", type=Path, default=REPO_ROOT / "dataset/bmw_lab_training")
    parser.add_argument("--training-id", default="bmw_right_batch_20260810_21_roi_v1")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "results/bmw_lab_one_click")
    parser.add_argument("--run-id", default="bmw_right_batch_20260810_21_efficientad_v1")
    parser.add_argument("--imagenette-dir", type=Path, default=defaults.imagenette_dir)
    parser.add_argument("--efficientad-epochs", type=int, default=30)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--view",
        action="append",
        choices=VIEW_ORDER,
        help="只训练指定视角；可重复传入。默认训练完整八视图。",
    )
    parser.add_argument(
        "--efficientad-all-normal-train",
        action="store_true",
        help="全部normal只用于训练checkpoint；跳过normal_test评分与阈值拟合。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只核对已发布release布局并打印单阶段计划；不创建输出目录、不启动GPU训练。",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> LabTrainingConfig:
    """Create the shared training config without any YOLO inputs."""
    defaults = LabTrainingConfig.defaults(REPO_ROOT)
    training_root = args.training_root.expanduser().resolve()
    return replace(
        defaults,
        prepared_root=training_root / args.training_id,
        training_root=training_root,
        training_id=args.training_id,
        output_root=args.output_root.expanduser().resolve(),
        run_id=args.run_id,
        imagenette_dir=args.imagenette_dir.expanduser().resolve(),
        efficientad_epochs=args.efficientad_epochs,
        gpu=args.gpu,
        workers=args.workers,
        seed=args.seed,
        views=tuple(args.view) if args.view else VIEW_ORDER,
        efficientad_all_normal_train=args.efficientad_all_normal_train,
    )


def main(argv: list[str] | None = None) -> int:
    """Validate the released EfficientAD data and run its sole stage."""
    args = build_parser().parse_args(argv)
    config = _config_from_args(args)
    try:
        preflight = preflight_efficientad_only(config)
        training = run_efficientad_only_training(config, dry_run=args.dry_run)
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
