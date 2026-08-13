#!/usr/bin/env python3
"""Interactively select a four-point, rotated bright-streak ROI."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bmw_inspection.lab.bright_streak_rotated_roi import (  # noqa: E402
    RotatedBrightStreakRoi,
    rectify_bright_streak_roi,
    write_rotated_bright_streak_roi,
)
from bmw_inspection.lab.eight_view_roi import fit_image_for_display  # noqa: E402


DEFAULT_IMAGE = REPO_ROOT / (
    "results/bmw_eight_view_demo_v5_template_manual_ignore_mask_v1/"
    "bmw_demo_20260813_164043/images/front_left_hdr.png"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "results/bmw_bright_streak_rotated_roi/"
    "bmw_demo_20260813_164043_v1/roi.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_point(
    display_point: tuple[int, int],
    display_shape: tuple[int, int],
    source_shape: tuple[int, int],
) -> tuple[int, int]:
    """Map one displayed-image point to a clamped source-image coordinate."""
    display_height, display_width = display_shape
    source_height, source_width = source_shape
    if display_height <= 0 or display_width <= 0:
        raise ValueError("display shape must be positive")
    source_x = round(display_point[0] * source_width / display_width)
    source_y = round(display_point[1] * source_height / display_height)
    return min(max(source_x, 0), source_width - 1), min(max(source_y, 0), source_height - 1)


def _display_point(
    source_point: tuple[int, int],
    display_shape: tuple[int, int],
    source_shape: tuple[int, int],
) -> tuple[int, int]:
    display_height, display_width = display_shape
    source_height, source_width = source_shape
    return (
        round(source_point[0] * display_width / source_width),
        round(source_point[1] * display_height / source_height),
    )


def _render(
    displayed: np.ndarray,
    *,
    source_shape: tuple[int, int],
    points: list[tuple[int, int]],
) -> np.ndarray:
    canvas = displayed.copy()
    mapped = [
        _display_point(point, canvas.shape[:2], source_shape)
        for point in points
    ]
    if len(mapped) > 1:
        cv2.polylines(canvas, [np.asarray(mapped, dtype=np.int32)], False, (0, 255, 0), 2, cv2.LINE_AA)
    if len(mapped) == 4:
        cv2.polylines(canvas, [np.asarray(mapped, dtype=np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
    for ordinal, point in enumerate(mapped, start=1):
        cv2.circle(canvas, point, 5, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(ordinal),
            (point[0] + 8, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return canvas


def build_parser() -> argparse.ArgumentParser:
    """Build the four-click rotated-ROI selector CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-display-width", type=int, default=1280)
    parser.add_argument("--max-display-height", type=int, default=900)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Select four clockwise points and atomically publish the ROI asset."""
    args = build_parser().parse_args(argv)
    image_path = args.image.expanduser().resolve()
    if args.max_display_width <= 0 or args.max_display_height <= 0:
        raise ValueError("display dimensions must be positive")
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"无法读取图像：{image_path}")
    displayed = fit_image_for_display(
        image,
        max_display_width=args.max_display_width,
        max_display_height=args.max_display_height,
    )
    points: list[tuple[int, int]] = []
    window = "BMW bright-streak rotated ROI"
    preview_window = "BMW bright-streak ROI preview"

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONUP and len(points) < 4:
            points.append(_source_point((x, y), displayed.shape[:2], image.shape[:2]))

    print("请按 左上、右上、右下、左下 的顺序点击四点。Enter 保存；R 清空重选；Esc 或 Q 不保存退出。")
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, displayed.shape[1], displayed.shape[0])
    cv2.setMouseCallback(window, on_mouse)
    preview_shown = False
    selected_asset: RotatedBrightStreakRoi | None = None
    try:
        while True:
            cv2.imshow(window, _render(displayed, source_shape=image.shape[:2], points=points))
            if len(points) == 4 and not preview_shown:
                try:
                    selected_asset = RotatedBrightStreakRoi(
                        points_xy=tuple(points),
                        source_width=image.shape[1],
                        source_height=image.shape[0],
                        output_width=81,
                        output_height=613,
                        source_image=str(image_path),
                        source_image_sha256=_sha256(image_path),
                    )
                except ValueError as error:
                    print(f"点位无效：{error}。请按 R 清空后依次重选。")
                else:
                    cv2.imshow(preview_window, rectify_bright_streak_roi(image, selected_asset))
                    print(f"最终点位：{points}")
                preview_shown = True
            key = cv2.waitKey(30) & 0xFF
            if key == 255:
                continue
            if key in (27, ord("q"), ord("Q")):
                return 0
            if key in (ord("r"), ord("R")):
                points.clear()
                preview_shown = False
                selected_asset = None
                try:
                    cv2.destroyWindow(preview_window)
                except cv2.error:
                    pass
            elif key in (10, 13):
                if len(points) != 4:
                    print("请先按顺序点击四个点。")
                    continue
                if selected_asset is None:
                    print("点位无效，请按 R 清空后依次重选。")
                    continue
                destination = write_rotated_bright_streak_roi(args.output.expanduser().resolve(), selected_asset)
                print(f"已保存 ROI：{destination}")
                print(f"最终点位：{points}")
                print(f"资产 SHA-256：{_sha256(destination)}")
                return 0
    finally:
        try:
            cv2.destroyWindow(window)
            cv2.destroyWindow(preview_window)
        except cv2.error:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
