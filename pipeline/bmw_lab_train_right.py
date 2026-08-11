#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""一键训练 BMW 右手八视图实验室模型。"""

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
    preflight_training,
    run_training,
)
from bmw_inspection.lab.right_train_all import prepare_reviewed_yolo_labels  # noqa: E402

LABELING_ROOT = REPO_ROOT / "dataset/bmw_lab_labeling/bmw_right_defects_yolo_labeling_v1"
EXPORT_STEM = "project-1-at-2026-08-10-20-25-8ec6b908"


def build_parser() -> argparse.ArgumentParser:
    """Build the right-hand laboratory training command line.

    Returns:
        argparse.ArgumentParser: Parser with right-hand data and laboratory training defaults.
    """
    defaults = LabTrainingConfig.defaults(REPO_ROOT)
    parser = argparse.ArgumentParser(
        description=("整理已复核右手 YOLO 标签，并依次训练 Template、光痕规则、8 个 EfficientAD-S 和 1 个 YOLO26n。"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=REPO_ROOT / "dataset/bmw_lab_prepared/bmw_right_complete_20260810_v1",
    )
    parser.add_argument(
        "--roi-config",
        type=Path,
        default=REPO_ROOT / "configs/bmw/rois/bmw_right_hdr_eight_view_v1.json",
    )
    parser.add_argument("--label-zip", type=Path, default=LABELING_ROOT / f"{EXPORT_STEM}.zip")
    parser.add_argument("--label-json", type=Path, default=LABELING_ROOT / f"{EXPORT_STEM}.json")
    parser.add_argument(
        "--label-cache",
        type=Path,
        default=(
            REPO_ROOT / "dataset/bmw_lab_labeling/exports/bmw_right_defects_yolo_reviewed_v1/reviewed_yolo_labels"
        ),
    )
    parser.add_argument("--training-root", type=Path, default=REPO_ROOT / "dataset/bmw_lab_training")
    parser.add_argument("--training-id", default="bmw_right_complete_roi_reviewed_v1")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "results/bmw_lab_one_click")
    parser.add_argument("--run-id", default="bmw_right_eight_view_v1")
    parser.add_argument("--yolo-checkpoint", type=Path, default=defaults.yolo_checkpoint)
    parser.add_argument("--imagenette-dir", type=Path, default=defaults.imagenette_dir)
    parser.add_argument("--efficientad-epochs", type=int, default=30)
    parser.add_argument("--yolo-epochs", type=int, default=100)
    parser.add_argument("--yolo-batch", type=int, default=32)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="整理并核对标签，只打印训练计划，不物化 ROI 数据或启动 GPU 训练",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> LabTrainingConfig:
    defaults = LabTrainingConfig.defaults(REPO_ROOT)
    return replace(
        defaults,
        prepared_root=args.prepared_root.expanduser().resolve(),
        roi_config=args.roi_config.expanduser().resolve(),
        reviewed_yolo_labels=args.label_cache.expanduser().resolve(),
        training_root=args.training_root.expanduser().resolve(),
        training_id=args.training_id,
        output_root=args.output_root.expanduser().resolve(),
        run_id=args.run_id,
        yolo_checkpoint=args.yolo_checkpoint.expanduser().resolve(),
        imagenette_dir=args.imagenette_dir.expanduser().resolve(),
        efficientad_epochs=args.efficientad_epochs,
        yolo_epochs=args.yolo_epochs,
        yolo_batch=args.yolo_batch,
        yolo_image_size=args.yolo_imgsz,
        gpu=args.gpu,
        workers=args.workers,
        seed=args.seed,
    )


def main(argv: list[str] | None = None) -> int:
    """Prepare reviewed labels, validate inputs, and run the fail-fast pipeline.

    Args:
        argv (list[str] | None): Optional command-line argument list.

    Returns:
        int: Zero on successful planning/training, otherwise two.
    """
    args = build_parser().parse_args(argv)
    config = _config_from_args(args)
    try:
        labels = prepare_reviewed_yolo_labels(args.label_zip, args.label_json, config.reviewed_yolo_labels)
        preflight = preflight_training(config)
        training = run_training(config, dry_run=args.dry_run)
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
    print(
        json.dumps(
            {"labels": labels, "preflight": preflight, "training": training},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
