# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Display one strict ZS32 eight-view result in OpenCV."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from zs32_inspection.dashboard.app import DashboardState, run_dashboard
from zs32_inspection.dashboard.parser import DashboardResultError, load_inspection_result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--result-dir", type=Path, help="Offline directory containing runtime_manifest.json.")
    source.add_argument("--live", action="store_true", help="Reserved live mode; connected in Task 6 onward.")
    parser.add_argument("--part-id", help="Part identity for the future live workflow.")
    parser.add_argument("--no-gui", action="store_true", help="Render without any OpenCV HighGUI calls.")
    parser.add_argument("--save-screenshot", type=Path, help="Write the rendered 1600x920 PNG.")
    return parser


def _run(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.live:
        print("错误: live 模式尚未接入；请使用 --result-dir 进行离线验收", file=sys.stderr)
        return 2
    assert args.result_dir is not None
    try:
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
