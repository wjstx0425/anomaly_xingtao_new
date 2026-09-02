#!/usr/bin/env python3
"""One-click BMW eight-view laboratory training entrypoint."""

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


def build_parser() -> argparse.ArgumentParser:
    """Build the small laboratory-oriented command line interface."""
    defaults = LabTrainingConfig.defaults(REPO_ROOT)
    parser = argparse.ArgumentParser(
        description=(
            "一键训练 BMW 八视图实验室模型：Template、光痕规则、"
            "8 个 EfficientAD-S 和 1 个 YOLO26n。"
        )
    )
    parser.add_argument("--prepared-root", type=Path, default=defaults.prepared_root)
    parser.add_argument("--roi-config", type=Path, default=defaults.roi_config)
    parser.add_argument("--reviewed-yolo-labels", type=Path, default=defaults.reviewed_yolo_labels)
    parser.add_argument("--training-root", type=Path, default=defaults.training_root)
    parser.add_argument("--training-id", default=defaults.training_id)
    parser.add_argument("--output-root", type=Path, default=defaults.output_root)
    parser.add_argument("--run-id", default=defaults.run_id)
    parser.add_argument("--yolo-checkpoint", type=Path, default=defaults.yolo_checkpoint)
    parser.add_argument("--imagenette-dir", type=Path, default=defaults.imagenette_dir)
    parser.add_argument("--efficientad-epochs", type=int, default=defaults.efficientad_epochs)
    parser.add_argument(
        "--efficientad-all-normal-train",
        action="store_true",
        help="将全部 normal 样本用于 EfficientAD 训练；只生成 checkpoint，阈值需用独立验证集标定",
    )
    parser.add_argument("--yolo-epochs", type=int, default=defaults.yolo_epochs)
    parser.add_argument("--yolo-batch", type=int, default=defaults.yolo_batch)
    parser.add_argument("--yolo-imgsz", type=int, default=defaults.yolo_image_size)
    parser.add_argument("--gpu", type=int, default=defaults.gpu)
    parser.add_argument("--workers", type=int, default=defaults.workers)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查输入并打印训练顺序，不创建数据集或启动训练",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> LabTrainingConfig:
    defaults = LabTrainingConfig.defaults(REPO_ROOT)
    return replace(
        defaults,
        prepared_root=args.prepared_root.expanduser().resolve(),
        roi_config=args.roi_config.expanduser().resolve(),
        reviewed_yolo_labels=args.reviewed_yolo_labels.expanduser().resolve(),
        training_root=args.training_root.expanduser().resolve(),
        training_id=args.training_id,
        output_root=args.output_root.expanduser().resolve(),
        run_id=args.run_id,
        yolo_checkpoint=args.yolo_checkpoint.expanduser().resolve(),
        imagenette_dir=args.imagenette_dir.expanduser().resolve(),
        efficientad_epochs=args.efficientad_epochs,
        efficientad_all_normal_train=args.efficientad_all_normal_train,
        yolo_epochs=args.yolo_epochs,
        yolo_batch=args.yolo_batch,
        yolo_image_size=args.yolo_imgsz,
        gpu=args.gpu,
        workers=args.workers,
        seed=args.seed,
    )


def main(argv: list[str] | None = None) -> int:
    """Validate inputs, then execute the fail-fast laboratory pipeline."""
    args = build_parser().parse_args(argv)
    config = _config_from_args(args)
    try:
        preflight = preflight_training(config)
        report = run_training(config, dry_run=args.dry_run)
    except (FileExistsError, OSError, RuntimeError, ValueError) as error:
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
            {"preflight": preflight, "training": report},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
