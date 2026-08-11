#!/usr/bin/env python3
"""Select and persist the BMW bright-streak Demo ROI."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from bmw_inspection.roi_selector import select_roi  # noqa: E402

DEFAULT_IMAGE = REPO_ROOT / "dataset/bmw/OK/Image_20260805172921398.bmp"
DEFAULT_CONFIG = REPO_ROOT / "configs/bmw/bright_streak_demo.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-display-width", type=int, default=1280)
    parser.add_argument("--max-display-height", type=int, default=720)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the interactive selector and report the persisted half-open ROI."""
    args = _parser().parse_args(argv)
    try:
        roi = select_roi(
            args.image,
            args.config,
            max_display_width=args.max_display_width,
            max_display_height=args.max_display_height,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"BMW ROI selection failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(f"roi_xyxy={list(roi)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
