#!/usr/bin/env python3
"""Retrain BMW Template, EfficientAD, and legacy bright-streak without YOLO."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bmw_inspection.lab.eight_view_train_all import LabTrainingConfig  # noqa: E402
from bmw_inspection.lab.normal_only_retraining import (  # noqa: E402
    NormalOnlyRetrainingConfig,
    run_normal_only_retraining,
)


def build_parser() -> argparse.ArgumentParser:
    defaults = LabTrainingConfig.defaults(REPO_ROOT)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--reuse-roi", type=Path, required=True)
    parser.add_argument("--old-no-streak-release", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, default=REPO_ROOT / "dataset/bmw_lab_training")
    parser.add_argument("--training-id", required=True)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "results/bmw_lab_one_click")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--derived-roi", type=Path)
    parser.add_argument("--bright-base-config", type=Path, default=REPO_ROOT / "configs/bmw/bright_streak_demo.json")
    parser.add_argument(
        "--baseline-template-root",
        type=Path,
        default=(
            REPO_ROOT
            / "results/bmw_lab_one_click/bmw_right_batch_20260810_21_v3_ng_evidence_demo_v1/template"
        ),
        help="现用右手Template目录；仅复用各视角固定阈值。",
    )
    parser.add_argument("--imagenette-dir", type=Path, default=defaults.imagenette_dir)
    parser.add_argument("--efficientad-epochs", type=int, default=30)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    derived_roi = args.derived_roi or REPO_ROOT / "configs/bmw/rois" / f"{args.training_id}.json"
    config = NormalOnlyRetrainingConfig(
        repo_root=REPO_ROOT,
        prepared_root=args.prepared_root.expanduser().resolve(),
        reuse_roi=args.reuse_roi.expanduser().resolve(),
        old_no_streak_release=args.old_no_streak_release.expanduser().resolve(),
        training_root=args.training_root.expanduser().resolve(),
        training_id=args.training_id,
        output_root=args.output_root.expanduser().resolve(),
        run_id=args.run_id,
        derived_roi=derived_roi.expanduser().resolve(),
        bright_base_config=args.bright_base_config.expanduser().resolve(),
        imagenette_dir=args.imagenette_dir.expanduser().resolve(),
        efficientad_epochs=args.efficientad_epochs,
        gpu=args.gpu,
        workers=args.workers,
        seed=args.seed,
        baseline_template_root=args.baseline_template_root.expanduser().resolve(),
    )
    try:
        report = run_normal_only_retraining(config, dry_run=args.dry_run)
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
