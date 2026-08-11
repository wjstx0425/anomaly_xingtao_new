#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Train six independent BMW laboratory Template model groups."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from bmw_inspection.lab.config import load_experiment_config  # noqa: E402
from bmw_inspection.lab.template import load_template_manifest, train_template_groups  # noqa: E402


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _template_count(value: str) -> int:
    parsed = int(value)
    if not 3 <= parsed <= 5:
        raise argparse.ArgumentTypeError("expected an integer from 3 to 5")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the deterministic offline Template trainer CLI."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="Physical-part CSV manifest.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="BMW experiment profile providing the six per-view part ROIs.",
    )
    parser.add_argument("--output-root", type=Path, required=True, help="New root for six immutable model groups.")
    parser.add_argument("--path-root", type=Path, help="Optional base for relative manifest image paths.")
    parser.add_argument("--target-width", type=_positive_int, default=512)
    parser.add_argument("--target-height", type=_positive_int, default=512)
    parser.add_argument("--max-shift", type=int, default=12)
    parser.add_argument("--template-count", type=_template_count, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Train all views or fail without replacing an existing output root."""
    args = build_parser().parse_args(argv)
    try:
        config = load_experiment_config(args.config)
        rows = load_template_manifest(args.manifest, path_root=args.path_root)
        model_paths = train_template_groups(
            rows,
            config.part_rois,
            args.output_root,
            target_size=(args.target_width, args.target_height),
            max_shift=args.max_shift,
            template_count=args.template_count,
        )
    except (FileExistsError, OSError, TypeError, ValueError) as error:
        print(f"BMW Template training failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "TRAINED_SYNTHETIC_OR_LOCAL",
                "model_paths": {view_id.value: str(path) for view_id, path in model_paths.items()},
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
