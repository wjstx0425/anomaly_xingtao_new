#!/usr/bin/env python3
"""Train both BMW hands from manually retained Template review crops."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
MAIN_ROOT = REPO_ROOT.parents[1] if REPO_ROOT.parent.name == ".worktrees" else REPO_ROOT

from bmw_inspection.lab.eight_view_demo import load_demo_config  # noqa: E402
from bmw_inspection.lab.template_review_training import (  # noqa: E402
    calibrate_reviewed_template_models,
    write_reviewed_demo_config,
    write_reviewed_template_models,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the bilateral reviewed-Template trainer CLI."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--review-root",
        type=Path,
        default=MAIN_ROOT / "dataset/bmw_lab_labeling/bmw_template_40_review_0823_v1",
    )
    for hand in ("right", "left"):
        parser.add_argument(
            f"--{hand}-roi-config",
            type=Path,
            default=REPO_ROOT / f"configs/bmw/rois/bmw_{hand}_0820_v1.json",
        )
        parser.add_argument(
            f"--{hand}-manifest",
            type=Path,
            default=MAIN_ROOT
            / f"dataset/bmw_lab_prepared/bmw_{hand}_front_right_0823_v1/manifests/dataset_manifest.csv",
        )
        parser.add_argument(
            f"--{hand}-base-config",
            type=Path,
            default=REPO_ROOT / f"configs/bmw/experiments/bmw_eight_view_demo_{hand}_0820_mixed_v1.json",
        )
        parser.add_argument(
            f"--{hand}-output-root",
            type=Path,
            default=MAIN_ROOT / f"results/bmw_lab_one_click/bmw_{hand}_template_40_reviewed_0823_v1",
        )
        parser.add_argument(
            f"--{hand}-output-config",
            type=Path,
            default=REPO_ROOT / f"configs/bmw/experiments/bmw_eight_view_demo_{hand}_0823_template40_v1.json",
        )
        parser.add_argument(
            f"--{hand}-result-root",
            type=Path,
            default=MAIN_ROOT / f"results/bmw_lab_one_click/bmw_eight_view_demo_{hand}_0823_template40_v1",
        )
    return parser


def run_hand(
    *,
    hand: str,
    review_root: Path,
    roi_config: Path,
    prepared_manifest: Path,
    base_config: Path,
    output_root: Path,
    output_config: Path,
    result_root: Path,
) -> dict[str, Any]:
    """Train, calibrate, and write one new hand-specific candidate config."""
    base_path = Path(base_config).expanduser().resolve()
    base_before = base_path.read_bytes()
    models = write_reviewed_template_models(review_root, hand, roi_config, output_root)
    config = load_demo_config(base_path)
    report = calibrate_reviewed_template_models(models, config, prepared_manifest)
    report_path = Path(output_root).expanduser().resolve() / "calibration_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    candidate_config = write_reviewed_demo_config(
        base_path,
        output_config,
        model_paths=models,
        calibration_report=report,
        prepared_manifest=prepared_manifest,
        result_root=result_root,
    )
    if base_path.read_bytes() != base_before:
        raise RuntimeError(f"rollback config changed unexpectedly: {base_path}")
    return {
        "hand": hand,
        "status": "complete",
        "output_root": str(Path(output_root).expanduser().resolve()),
        "candidate_config": str(candidate_config),
        "calibration_report": str(report_path),
        "template_counts": {view: int(report["views"][view]["template_count"]) for view in report["views"]},
        "template_thresholds": report["template_thresholds"],
        "weighted_thresholds": report["weighted_thresholds"],
    }


def main(argv: list[str] | None = None) -> int:
    """Run right then left, without changing either rollback config."""
    args = build_parser().parse_args(argv)
    reports = []
    try:
        for hand in ("right", "left"):
            reports.append(
                run_hand(
                    hand=hand,
                    review_root=args.review_root,
                    roi_config=getattr(args, f"{hand}_roi_config"),
                    prepared_manifest=getattr(args, f"{hand}_manifest"),
                    base_config=getattr(args, f"{hand}_base_config"),
                    output_root=getattr(args, f"{hand}_output_root"),
                    output_config=getattr(args, f"{hand}_output_config"),
                    result_root=getattr(args, f"{hand}_result_root"),
                )
            )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"status": "complete", "hands": reports}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

