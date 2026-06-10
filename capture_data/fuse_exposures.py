"""Fuse already-captured short- and long-exposure images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from exposure_fusion import fuse_exposures, read_bgr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "images",
        nargs="+",
        help=(
            "Exposure images ordered from short to long exposure. For your current "
            "test, pass 7000us first and 40000us second."
        ),
    )
    parser.add_argument("--out", default="./hik_images/fused.png")
    parser.add_argument(
        "--method",
        choices=["selective", "mertens"],
        default="selective",
        help="selective keeps the short exposure as base; mertens performs generic exposure fusion.",
    )
    parser.add_argument("--no-align", action="store_true")
    parser.add_argument("--short-dark-threshold", type=float, default=70.0)
    parser.add_argument("--long-clip-threshold", type=float, default=245.0)
    parser.add_argument("--blend-width", type=float, default=18.0)
    parser.add_argument("--blur-size", type=int, default=31)
    args = parser.parse_args()

    images = [read_bgr(path) for path in args.images]
    fused = fuse_exposures(
        images,
        method=args.method,
        align=not args.no_align,
        short_dark_threshold=args.short_dark_threshold,
        long_clip_threshold=args.long_clip_threshold,
        blend_width=args.blend_width,
        blur_size=args.blur_size,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), fused):
        raise RuntimeError(f"Failed to write image: {out_path}")

    print(f"Saved fused image: {out_path}")


if __name__ == "__main__":
    main()
