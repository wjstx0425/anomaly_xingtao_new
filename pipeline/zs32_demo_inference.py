# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run the single hash-free ZS32 Demo inference path."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from capture_data.zs32_demo_config import load_demo_config  # noqa: E402
from capture_data.zs32_demo_runtime import DemoRunResult, ZS32DemoRuntime  # noqa: E402
from zs32_inspection.capture.contracts import utc_now  # noqa: E402
from zs32_inspection.dashboard.contracts import ProgressRecord  # noqa: E402
from zs32_inspection.dashboard.control import write_progress  # noqa: E402
from zs32_inspection.domain.views import VIEW_ORDER  # noqa: E402

DEFAULT_DEMO_CONFIG = REPO_ROOT / "configs/zs32/zs32_demo.json"


def build_parser() -> argparse.ArgumentParser:
    """Build the one supported Demo inference parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--demo-config", type=Path, default=DEFAULT_DEMO_CONFIG)
    parser.add_argument("--part-id", required=True)
    parser.add_argument("--capture-session", required=True)
    parser.add_argument("--group-id", required=True)
    for view in VIEW_ORDER:
        parser.add_argument(f"--{view.replace('_', '-')}-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-json", type=Path)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--yolo-device", default="0")
    return parser


def prepare_demo_runtime(
    config_path: Path,
    *,
    accelerator: str = "gpu",
    devices: int = 1,
    yolo_device: str | None = "0",
) -> ZS32DemoRuntime:
    """Load and retain all eight Template/PatchCore models and one YOLO model."""
    config = load_demo_config(Path(config_path).expanduser().resolve())
    return ZS32DemoRuntime(
        config,
        accelerator=accelerator,
        devices=devices,
        yolo_device=yolo_device,
    )


def _progress(args: argparse.Namespace, state: str, message: str, *, error: str | None = None) -> None:
    if args.progress_json is None:
        return
    write_progress(
        args.progress_json,
        ProgressRecord(
            part_id=args.part_id,
            capture_session=args.capture_session,
            state=state,
            message=message,
            timestamp=utc_now(),
            error=error,
        ),
    )


def run_argv(
    argv: Sequence[str],
    *,
    runtime: ZS32DemoRuntime | None = None,
    startup_config_path: Path | None = None,
) -> DemoRunResult:
    """Parse and execute one request, optionally reusing a resident runtime."""
    args = build_parser().parse_args(list(argv))
    config_path = args.demo_config.expanduser().resolve()
    if startup_config_path is not None and config_path != startup_config_path.expanduser().resolve():
        raise ValueError(
            f"request uses {config_path}, but worker was started with "
            f"{startup_config_path.expanduser().resolve()}",
        )
    _progress(args, "running_models", "running the Template gate and eligible downstream branches")
    active_runtime = runtime or prepare_demo_runtime(
        config_path,
        accelerator=args.accelerator,
        devices=args.devices,
        yolo_device=args.yolo_device,
    )
    images = {
        view: Path(getattr(args, f"{view}_image")).expanduser().resolve()
        for view in VIEW_ORDER
    }
    result = active_runtime.run(
        images,
        part_id=args.part_id,
        capture_session=args.capture_session,
        group_id=args.group_id,
        output_dir=args.output_dir.expanduser().resolve(),
        config_path=config_path,
    )
    state = "completed" if result.inspection_complete else "failed"
    message = f"Demo inspection finished: {result.machine_status}"
    _progress(args, state, message, error=result.errors[0] if result.errors else None)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and convert failures into an explicit nonzero exit."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        run_argv(arguments)
    except Exception as error:  # noqa: BLE001 - CLI must preserve the real first error
        try:
            args = build_parser().parse_args(arguments)
            _progress(args, "failed", "Demo inspection failed", error=f"{type(error).__name__}: {error}")
        except (SystemExit, Exception):
            pass
        print(f"ZS32 Demo failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
