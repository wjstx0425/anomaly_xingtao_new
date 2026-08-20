#!/usr/bin/env python3
"""Run the BMW DA9625347 bright-streak inspection Demo."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bmw_inspection.camera import SingleCameraSession  # noqa: E402
from bmw_inspection.demo_app import render_dashboard, run_gui, run_image, run_offline  # noqa: E402
from bmw_inspection.contracts import load_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/bmw/bright_streak_demo.json",
    )
    parser.add_argument("--image", type=Path, help="Saved BMP/PNG for offline Demo testing.")
    parser.add_argument("--output-root", type=Path, help="Override the configured result directory.")
    parser.add_argument("--no-gui", action="store_true", help="Inspect once, print the result, and exit.")
    parser.add_argument("--save-screenshot", type=Path, help="Write the rendered 1600x900 Demo screen.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.image is not None:
        if args.no_gui and args.save_screenshot is None:
            result = run_offline(args.image, args.config, output_root=args.output_root)
            print(result.status.value)
            return 0 if result.status.value != "ERROR" else 2
        import cv2

        image_path = args.image.expanduser().resolve()
        capture_started = time.perf_counter()
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        capture_elapsed_ms = (time.perf_counter() - capture_started) * 1000.0
        if image is None:
            raise FileNotFoundError(f"unable to read image: {image_path}")
        result, result_dir = run_image(
            image,
            config,
            output_root=args.output_root,
            source_ref=str(image_path),
            capture_elapsed_ms=capture_elapsed_ms,
        )
        dashboard = render_dashboard(image, result, config, result_dir)
        if args.save_screenshot is not None:
            args.save_screenshot.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.save_screenshot), dashboard):
                raise RuntimeError(f"failed to write screenshot: {args.save_screenshot}")
        if not args.no_gui:
            cv2.imshow("BMW Bright Streak Demo", dashboard)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        print(result.status.value)
        return 0 if result.status.value != "ERROR" else 2

    if not args.no_gui:
        with SingleCameraSession(config) as session:
            result = run_gui(
                config,
                session.capture,
                output_root=args.output_root,
                save_screenshot=args.save_screenshot,
            )
        if result is None:
            return 0
        print(result.status.value)
        return 0 if result.status.value != "ERROR" else 2

    with SingleCameraSession(config) as session:
        capture_started = time.perf_counter()
        image = session.capture()
        capture_elapsed_ms = (time.perf_counter() - capture_started) * 1000.0
    result, result_dir = run_image(
        image,
        config,
        output_root=args.output_root,
        source_ref=f"camera:{config.camera_serial}",
        capture_elapsed_ms=capture_elapsed_ms,
    )
    print(result.status.value)
    if args.save_screenshot is None:
        return 0 if result.status.value != "ERROR" else 2

    import cv2

    dashboard = render_dashboard(image, result, config, result_dir, camera_connected=True)
    if args.save_screenshot is not None:
        args.save_screenshot.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.save_screenshot), dashboard):
            raise RuntimeError(f"failed to write screenshot: {args.save_screenshot}")
    return 0 if result.status.value != "ERROR" else 2


if __name__ == "__main__":
    raise SystemExit(main())
