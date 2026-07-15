# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Capture and inspect one right-hand ZS32 part with the eight-view runtime bundle."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capture_data.zs32_live_commissioning import (  # noqa: E402
    LiveCommissioningError,
    LiveRunConfig,
    LiveRunResult,
    run_live_commissioning,
)

DEFAULT_CAPTURE_ROOT = Path("/home/yunjing/anomalib/results/zs32_live_capture")
DEFAULT_OUTPUT_ROOT = Path("/home/yunjing/anomalib/results/zs32_live_runtime")
DEFAULT_RUNTIME_CONFIG = REPO_ROOT / "results/zs32_runtime_bundle_eight_view_v2/runtime_bundle.json"
DEFAULT_TOPOLOGY = REPO_ROOT / "configs/zs32/topology/zs32_4cam_double_side_v1.json"


class _Stage35ArgumentParser(argparse.ArgumentParser):
    """Allow device discovery alone while requiring an identity for live runs."""

    def parse_args(
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        """Parse Stage35 arguments and enforce its two mutually exclusive operating shapes."""
        parsed = super().parse_args(args, namespace)
        if not parsed.list_devices and parsed.part_id is None:
            self.error("the following arguments are required: --part-id")
        return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the right-hand live commissioning command parser."""
    parser = _Stage35ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--part-id", help="Stable operator identity for the part being captured.")
    parser.add_argument("--list-devices", action="store_true", help="List SDK cameras without capturing or inferring.")
    parser.add_argument(
        "--diagnostic-skip-template",
        action="store_true",
        help="Skip template matching and run PatchCore/YOLO infer only; never permits production release.",
    )
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    return parser


def _run_id(part_id: str) -> str:
    """Return one safe timestamped capture-run identity."""
    if part_id in {".", ".."} or re.fullmatch(r"[A-Za-z0-9_.-]+", part_id) is None:
        msg = "part_id must contain only letters, digits, dot, underscore, or hyphen"
        raise LiveCommissioningError(msg)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{timestamp}_{part_id}"


def _list_devices() -> int:
    """Delegate camera enumeration to the existing collector under this interpreter."""
    command = [
        sys.executable,
        str((REPO_ROOT / "capture_data/collect_multicamera_dataset.py").resolve()),
        "--list-devices",
    ]
    try:
        completed = subprocess.run(command, check=False)  # noqa: S603 - fixed local script and current interpreter
    except OSError as error:
        print(f"ERROR: device listing could not run: {error}", file=sys.stderr)
        return 2
    returncode = getattr(completed, "returncode", None)
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        print("ERROR: device listing returned no integer return code", file=sys.stderr)
        return 2
    return returncode


def _config_from_args(args: argparse.Namespace) -> LiveRunConfig:
    """Map reviewed CLI options onto the locked live core configuration."""
    part_id = str(args.part_id)
    return LiveRunConfig(
        repo_root=REPO_ROOT.resolve(),
        part_id=part_id,
        run_id=_run_id(part_id),
        capture_root=args.capture_root,
        output_root=args.output_root,
        runtime_config=args.runtime_config,
        topology_path=args.topology,
        diagnostic_skip_template=args.diagnostic_skip_template,
    )


def _print_result(result: LiveRunResult) -> None:
    """Print the business decision, locked policy, and absolute audit artifacts."""
    if result.diagnostic_skip_template:
        patchcore_csv = result.patchcore_csv
        if patchcore_csv is None:
            msg = "diagnostic result is missing patchcore_csv"
            raise LiveCommissioningError(msg)
        yolo_csv = result.yolo_csv
        if yolo_csv is None:
            msg = "diagnostic result is missing yolo_csv"
            raise LiveCommissioningError(msg)
        print("template: skipped (diagnostic)")
        print("stage18_audit: none (diagnostic infer; fusion was not run)")
        print(f"patchcore_csv: {patchcore_csv.expanduser().resolve()}")
        print(f"yolo_csv: {yolo_csv.expanduser().resolve()}")
    else:
        audit_report = result.audit_report
        print("audit_report:")
        if audit_report:
            print(audit_report)
        else:
            print("none (template short-circuit; Stage18 audit was not generated)")
    print(f"machine_status: {result.machine_status}")
    print(f"inspection_complete: {str(result.inspection_complete).lower()}")
    print(f"commissioning_only: {str(result.commissioning_only).lower()}")
    print(f"production_release_allowed: {str(result.production_release_allowed).lower()}")
    print(f"capture_manifest: {result.sample.manifest_path.expanduser().resolve()}")
    print(f"runtime_summary: {result.runtime_summary_path.expanduser().resolve()}")
    if not result.diagnostic_skip_template:
        audit_path = result.audit_path
        if audit_path is None:
            print("audit: none (template short-circuit; Stage18 audit was not generated)")
        else:
            print(f"audit: {Path(audit_path).expanduser().resolve()}")
    print(f"output_dir: {result.output_dir.expanduser().resolve()}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run device discovery or one complete right-hand live commissioning cycle."""
    args = build_parser().parse_args(argv)
    if args.list_devices:
        return _list_devices()
    try:
        result = run_live_commissioning(_config_from_args(args))
        _print_result(result)
    except LiveCommissioningError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
