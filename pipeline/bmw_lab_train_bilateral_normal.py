#!/usr/bin/env python3
"""顺序训练 BMW 0820 右手和左手 Template 与全 normal EfficientAD。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bmw_inspection.lab.bilateral_normal_training import (  # noqa: E402
    BilateralNormalTrainingConfig,
    run_bilateral_normal_training,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the bilateral laboratory training CLI."""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--right-prepared-root",
        type=Path,
        default=REPO_ROOT / "dataset/bmw_lab_prepared/bmw_right_0820_v1",
    )
    parser.add_argument(
        "--right-roi-config",
        type=Path,
        default=REPO_ROOT / "configs/bmw/rois/bmw_right_0820_v1.json",
    )
    parser.add_argument(
        "--left-prepared-root",
        type=Path,
        default=REPO_ROOT / "dataset/bmw_lab_prepared/bmw_left_0820_v1",
    )
    parser.add_argument(
        "--left-roi-config",
        type=Path,
        default=REPO_ROOT / "configs/bmw/rois/bmw_left_0820_v1.json",
    )
    parser.add_argument("--training-root", type=Path, default=REPO_ROOT / "dataset/bmw_lab_training")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "results/bmw_lab_one_click")
    parser.add_argument("--right-training-id", default="bmw_right_0820_template_efficientad_all_normal_v1")
    parser.add_argument("--right-run-id", default="bmw_right_0820_template_efficientad_all_normal_models_v1")
    parser.add_argument("--left-training-id", default="bmw_left_0820_template_efficientad_all_normal_v1")
    parser.add_argument("--left-run-id", default="bmw_left_0820_template_efficientad_all_normal_models_v1")
    parser.add_argument("--efficientad-epochs", type=int, default=30)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="只核对左右手身份和训练计划，不创建目录或启动GPU")
    return parser


def _config(args: argparse.Namespace) -> BilateralNormalTrainingConfig:
    return BilateralNormalTrainingConfig(
        repo_root=REPO_ROOT,
        right_prepared_root=args.right_prepared_root,
        right_roi_config=args.right_roi_config,
        left_prepared_root=args.left_prepared_root,
        left_roi_config=args.left_roi_config,
        training_root=args.training_root,
        output_root=args.output_root,
        right_training_id=args.right_training_id,
        right_run_id=args.right_run_id,
        left_training_id=args.left_training_id,
        left_run_id=args.left_run_id,
        efficientad_epochs=args.efficientad_epochs,
        gpu=args.gpu,
        workers=args.workers,
        seed=args.seed,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the bilateral checkpoint training pipeline."""

    args = build_parser().parse_args(argv)
    try:
        report = run_bilateral_normal_training(_config(args), dry_run=args.dry_run)
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(
            json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
