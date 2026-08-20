# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Display one ZS32 Demo eight-view result in OpenCV."""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from zs32_inspection.dashboard.app import DashboardState, run_dashboard
from zs32_inspection.dashboard.parser import DashboardResultError, load_inspection_result
from zs32_inspection.dashboard.live import Stage35Controller

_DEFAULT_DEMO_CONFIG = Path("configs/zs32/zs32_demo.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--result-dir", type=Path, help="Offline directory containing runtime_manifest.json.")
    source.add_argument("--live", action="store_true", help="Run one dashboard-owned Stage35 inspection.")
    parser.add_argument("--part-id", help="Part identity for the live workflow.")
    parser.add_argument(
        "--demo-config",
        type=Path,
        help="Single Demo configuration for Dashboard, capture, and inference.",
    )
    parser.add_argument("--no-gui", action="store_true", help="Render without any OpenCV HighGUI calls.")
    parser.add_argument("--save-screenshot", type=Path, help="Write the rendered 1600x920 PNG.")
    return parser


def _run(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.live:
            if not args.part_id:
                raise ValueError("--part-id is required with --live")
            demo_config = args.demo_config or _DEFAULT_DEMO_CONFIG
            work_dir = Path(tempfile.mkdtemp(prefix="zs32-dashboard-"))
            run_dashboard(
                DashboardState(result=None, demo_config_path=demo_config),
                no_gui=args.no_gui,
                save_screenshot=args.save_screenshot,
                live_controller=Stage35Controller(
                    work_dir,
                    demo_config=demo_config,
                ),
                part_id=args.part_id,
            )
        else:
            assert args.result_dir is not None
            live_only = {
                "--part-id": args.part_id,
                "--demo-config": args.demo_config,
            }
            invalid = [option for option, value in live_only.items() if value is not None]
            if invalid:
                raise ValueError(f"{', '.join(invalid)} may only be used with --live")
            result = load_inspection_result(args.result_dir)
            run_dashboard(
                DashboardState(result=result),
                no_gui=args.no_gui,
                save_screenshot=args.save_screenshot,
            )
    except (DashboardResultError, OSError, RuntimeError, ValueError) as error:
        print(f"dashboard error: {error}", file=sys.stderr)
        return 2
    return 0


def main() -> None:
    """Run the CLI and translate its status to a process exit code."""
    raise SystemExit(_run())


if __name__ == "__main__":
    main()
