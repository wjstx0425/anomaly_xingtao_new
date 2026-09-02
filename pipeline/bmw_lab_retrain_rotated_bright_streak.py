#!/usr/bin/env python3
"""Retrain the rotated BMW bright-streak candidate from current normal/no-streak data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bmw_inspection.lab.bright_streak_rotated_retraining import (  # noqa: E402
    retrain_rotated_bright_streak,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-manifest", type=Path, required=True)
    parser.add_argument("--no-streak-image", type=Path, required=True)
    parser.add_argument("--rotated-roi", type=Path, required=True)
    parser.add_argument("--rotated-roi-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = retrain_rotated_bright_streak(
            args.normal_manifest,
            args.no_streak_image,
            args.rotated_roi,
            args.rotated_roi_sha256,
            args.output_dir,
        )
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"BMW旋转光痕重训练失败：{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
