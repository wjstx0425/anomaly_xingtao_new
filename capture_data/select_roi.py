"""Interactively select an ROI from an image.

Example:
    python capture_data/select_roi.py dataset/fx11_2/no_hand/top/normal_test/example.png
    python capture_data/select_roi.py image.png --initial-roi 650,220,3300,2860 --save-overlay roi_check.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


WINDOW_NAME = "Select ROI"


def parse_roi(value: str) -> tuple[int, int, int, int]:
    """Parse ROI in x1,y1,x2,y2 format."""
    try:
        x1, y1, x2, y2 = (int(part.strip()) for part in value.split(","))
    except ValueError as error:
        msg = "ROI must be provided as x1,y1,x2,y2."
        raise argparse.ArgumentTypeError(msg) from error

    if x2 <= x1 or y2 <= y1:
        msg = "ROI requires x2 > x1 and y2 > y1."
        raise argparse.ArgumentTypeError(msg)
    return x1, y1, x2, y2


def clamp_roi(roi: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    """Clamp ROI coordinates to image boundaries."""
    x1, y1, x2, y2 = roi
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def draw_roi(image, roi: tuple[int, int, int, int] | None, scale: float):
    """Draw ROI on the display image."""
    display = image.copy()
    if roi is None:
        return display

    x1, y1, x2, y2 = roi
    p1 = (round(x1 * scale), round(y1 * scale))
    p2 = (round(x2 * scale), round(y2 * scale))
    cv2.rectangle(display, p1, p2, (0, 255, 0), 2)
    label = f"{x1},{y1},{x2},{y2}"
    cv2.putText(display, label, (p1[0], max(24, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return display


def save_overlay(path: Path, image, roi: tuple[int, int, int, int]) -> None:
    """Save a full-resolution ROI overlay image."""
    overlay = image.copy()
    x1, y1, x2, y2 = roi
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 4)
    cv2.putText(
        overlay,
        f"ROI {x1},{y1},{x2},{y2}",
        (x1, max(48, y1 - 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), overlay):
        msg = f"Failed to write overlay image: {path}"
        raise RuntimeError(msg)


def select_roi(
    image_path: Path,
    initial_roi: tuple[int, int, int, int] | None,
    max_width: int,
    max_height: int,
) -> tuple[int, int, int, int]:
    """Open a window and return an ROI in original image coordinates."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        msg = f"Failed to read image: {image_path}"
        raise FileNotFoundError(msg)

    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale < 1.0:
        display_image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    else:
        display_image = image.copy()

    initial_roi = clamp_roi(initial_roi, width, height) if initial_roi else None
    preview = draw_roi(display_image, initial_roi, scale)

    print(f"Image: {image_path}")
    print(f"Original size: width={width}, height={height}")
    if scale < 1.0:
        print(f"Display scale: {scale:.4f}")
    if initial_roi:
        print(f"Initial ROI: {','.join(str(value) for value in initial_roi)}")
    print("Mouse: drag a rectangle. Press Enter/Space to accept, c to clear, Esc to quit.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    x, y, roi_width, roi_height = cv2.selectROI(WINDOW_NAME, preview, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(WINDOW_NAME)

    if roi_width <= 0 or roi_height <= 0:
        if initial_roi is not None:
            return initial_roi
        msg = "No ROI selected."
        raise RuntimeError(msg)

    selected = (
        round(x / scale),
        round(y / scale),
        round((x + roi_width) / scale),
        round((y + roi_height) / scale),
    )
    return clamp_roi(selected, width, height)


def build_parser() -> argparse.ArgumentParser:
    """Build command line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Image path used for ROI selection.")
    parser.add_argument("--initial-roi", type=parse_roi, help="Optional existing ROI shown as a green reference box.")
    parser.add_argument("--max-window-width", type=int, default=1600, help="Maximum display window width.")
    parser.add_argument("--max-window-height", type=int, default=1000, help="Maximum display window height.")
    parser.add_argument("--save-overlay", type=Path, help="Optional path for a full-resolution ROI overlay image.")
    return parser


def main() -> None:
    """Run ROI selector."""
    args = build_parser().parse_args()
    roi = select_roi(args.image, args.initial_roi, args.max_window_width, args.max_window_height)

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        msg = f"Failed to read image: {args.image}"
        raise FileNotFoundError(msg)
    height, width = image.shape[:2]
    x1, y1, x2, y2 = clamp_roi(roi, width, height)

    print()
    print(f"ROI: {x1},{y1},{x2},{y2}")
    print(f"ROI width x height: {x2 - x1} x {y2 - y1}")
    print(f"Workflow argument: --roi {x1},{y1},{x2},{y2}")

    if args.save_overlay:
        save_overlay(args.save_overlay, image, (x1, y1, x2, y2))
        print(f"Overlay saved: {args.save_overlay}")


if __name__ == "__main__":
    main()
