#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Prepare left/right BMW 0823 Template candidates for manual review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER  # noqa: E402
from bmw_inspection.lab.template_review import ReviewSource, build_template_review_package  # noqa: E402


MAIN_ROOT = REPO_ROOT.parents[1] if REPO_ROOT.parent.name == ".worktrees" else REPO_ROOT


def _candidate_count(value: str) -> int:
    parsed = int(value)
    if parsed < 3:
        raise argparse.ArgumentTypeError("candidate count must be at least 3")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the focused 0823 Template review-package CLI."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--right-manifest",
        type=Path,
        default=MAIN_ROOT
        / "dataset/bmw_lab_prepared/bmw_right_front_right_0823_v1/manifests/dataset_manifest.csv",
    )
    parser.add_argument(
        "--right-roi-config",
        type=Path,
        default=REPO_ROOT / "configs/bmw/rois/bmw_right_0820_v1.json",
    )
    parser.add_argument(
        "--left-manifest",
        type=Path,
        default=MAIN_ROOT
        / "dataset/bmw_lab_prepared/bmw_left_front_right_0823_v1/manifests/dataset_manifest.csv",
    )
    parser.add_argument(
        "--left-roi-config",
        type=Path,
        default=REPO_ROOT / "configs/bmw/rois/bmw_left_0820_v1.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=MAIN_ROOT / "dataset/bmw_lab_labeling/bmw_template_40_review_0823_v1",
    )
    parser.add_argument("--candidate-count", type=_candidate_count, default=40)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate the review package without training or changing Demo configuration."""
    args = build_parser().parse_args(argv)
    try:
        sources = [
            ReviewSource(hand="right", manifest=args.right_manifest, roi_config=args.right_roi_config),
            ReviewSource(hand="left", manifest=args.left_manifest, roi_config=args.left_roi_config),
        ]
        output_root = build_template_review_package(
            sources,
            args.output_root,
            candidate_count=args.candidate_count,
        )
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as error:
        print(f"BMW Template review package failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "candidate_count_per_view": args.candidate_count,
                "candidate_total": len(sources) * len(VIEW_ORDER) * args.candidate_count,
                "output_root": str(output_root),
                "status": "REVIEW_PACKAGE_READY",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

