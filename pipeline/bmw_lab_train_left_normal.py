#!/usr/bin/env python3
"""Plan a fail-closed BMW left-hand normal-only candidate run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bmw_inspection.lab.left_normal_training import LeftNormalTrainingConfig, run_left_normal_training  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the left normal-only planning CLI."""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--roi-config", type=Path, required=True)
    parser.add_argument("--mask-index", type=Path, help="Required only for --stage calibrate or all.")
    parser.add_argument("--component-policy", type=Path, help="Required only for --stage calibrate or all.")
    parser.add_argument("--stage", choices=("train", "calibrate", "all"), default="train")
    parser.add_argument("--training-root", type=Path, default=REPO_ROOT / "dataset/bmw_lab_training")
    parser.add_argument("--training-id", required=True)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "results/bmw_lab_one_click")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--efficientad-epochs", type=int, default=30)
    parser.add_argument(
        "--efficientad-all-normal-train",
        action="store_true",
        help="将全部 normal 样本用于 EfficientAD 训练；只生成 checkpoint，阈值需用独立验证集标定",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="validate left identities and print the plan without GPU work")
    return parser


def _config(args: argparse.Namespace) -> LeftNormalTrainingConfig:
    return LeftNormalTrainingConfig(
        repo_root=REPO_ROOT,
        prepared_root=args.prepared_root,
        roi_config=args.roi_config,
        training_root=args.training_root,
        training_id=args.training_id,
        output_root=args.output_root,
        run_id=args.run_id,
        capture_scope="left",
        mask_index=args.mask_index,
        component_policy=args.component_policy,
        efficientad_epochs=args.efficientad_epochs,
        efficientad_all_normal_train=args.efficientad_all_normal_train,
        gpu=args.gpu,
        workers=args.workers,
        seed=args.seed,
    )


def main(argv: list[str] | None = None) -> int:
    """Execute the default train/calibrate stage or validate it with no GPU work."""

    args = build_parser().parse_args(argv)
    try:
        report = run_left_normal_training(_config(args), dry_run=args.dry_run, stage=args.stage)
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
