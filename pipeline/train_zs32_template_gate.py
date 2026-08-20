# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: EM101, TRY003

"""Train the fail-closed ZS32 whole-view template gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capture_data.zs32_template_gate import DEFAULT_HANDS, TemplateGateError, train_template_gate  # noqa: E402


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _probability(value: str) -> float:
    """Parse a probability in ``(0, 1]``."""
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("expected a number in (0, 1]")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the deterministic offline trainer parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="ZS32 crop or calibration CSV manifest.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination containing model.json/templates.")
    parser.add_argument(
        "--path-root",
        "--data-root",
        dest="path_root",
        type=Path,
        help="Root used to resolve relative output_path/image_path values (for example /home/yunjing/anomalib).",
    )
    parser.add_argument(
        "--required-hand",
        choices=DEFAULT_HANDS,
        action="append",
        help="Hand contract to train; repeat as needed. Defaults to both hands.",
    )
    parser.add_argument("--width", type=_positive_int, default=512, help="Aspect-preserving grayscale resize width.")
    parser.add_argument("--max-shift", type=int, default=12, help="Maximum x/y translation searched in pixels.")
    parser.add_argument("--templates-per-group", type=_positive_int, default=5)
    parser.add_argument("--max-train-per-group", type=_positive_int, default=120)
    parser.add_argument("--normal-quantile", type=_probability, default=0.995)
    parser.add_argument(
        "--normal-only",
        action="store_true",
        help="Build templates and a single PASS/NG threshold exclusively from normal rows; defect rows are ignored.",
    )
    parser.add_argument(
        "--evaluation-fraction",
        type=float,
        default=0.2,
        help="Deterministic held-out part fraction when the manifest has no split column values.",
    )
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--threshold-version", required=True)
    parser.add_argument("--roi-version", required=True)
    parser.add_argument("--template-version", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Train the model and print one machine-readable result."""
    args = build_parser().parse_args(argv)
    try:
        model = train_template_gate(
            args.manifest,
            args.output_dir,
            path_root=args.path_root,
            required_hands=tuple(args.required_hand or DEFAULT_HANDS),
            width=args.width,
            max_shift=args.max_shift,
            templates_per_group=args.templates_per_group,
            max_train_per_group=args.max_train_per_group,
            normal_quantile=args.normal_quantile,
            normal_only=args.normal_only,
            evaluation_fraction=args.evaluation_fraction,
            model_version=args.model_version,
            threshold_version=args.threshold_version,
            roi_version=args.roi_version,
            template_version=args.template_version,
        )
    except TemplateGateError as exc:
        print(json.dumps(exc.to_result(), ensure_ascii=False, allow_nan=False))
        return 2
    result = {
        "status": "TRAINED",
        "reason": "all required template groups have deployable dual thresholds",
        "score": None,
        "threshold": None,
        "model_path": str(args.output_dir / "model.json"),
        "group_count": len(model["groups"]),
        "versions": model["versions"],
    }
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
