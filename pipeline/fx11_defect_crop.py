# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Crop FX11 defect images with fixed top/bottom slot boxes.

This helper is a pipeline-level preset around ``capture_data/manual_part_crop.py``.
It keeps the defect-slot mapping mode at ``from-name`` so filenames such as
``less_2_2.png`` are cropped into ``slot02`` only.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from _common import REPO_ROOT


FX11_TOP_SLOTS = (
    "slot01:1,1,830,40,3230,450",
    "slot02:2,1,830,440,3230,860",
    "slot03:3,1,830,890,3230,1300",
    "slot04:4,1,830,1290,3230,1700",
    "slot05:5,1,830,1700,3230,2110",
    "slot06:6,1,830,2130,3230,2590",
)
FX11_BOTTOM_SLOTS = (
    "slot01:1,1,830,10,3230,420",
    "slot02:2,1,830,420,3230,840",
    "slot03:3,1,830,870,3230,1280",
    "slot04:4,1,830,1280,3230,1700",
    "slot05:5,1,830,1700,3230,2110",
    "slot06:6,1,830,2130,3230,2590",
)
FACE_TO_SLOTS = {
    "top": FX11_TOP_SLOTS,
    "bottom": FX11_BOTTOM_SLOTS,
}


def selected_faces(face: str) -> list[str]:
    """Return face names to process."""
    if face == "both":
        return ["top", "bottom"]
    return [face]


def build_manual_args(face: str, args: argparse.Namespace) -> list[str]:
    """Build arguments for manual_part_crop.py for one FX11 face."""
    manual_args = [
        "--data-root",
        str(args.data_root),
        "--output-root",
        str(args.output_root),
        "--hand",
        args.hand,
        "--position",
        face,
        "--labels",
        "defect",
        "--reference-label",
        "defect",
        "--defect-slot-mode",
        "from-name",
        "--nondefect-slots-from-defect-images",
        args.nondefect_slots_from_defect_images,
        "--hole-mask-method",
        args.hole_mask_method,
    ]
    for slot in FACE_TO_SLOTS[face]:
        manual_args.extend(["--slot", slot])

    if args.results_root is not None:
        manual_args.extend(
            [
                "--save-overlay",
                str(args.results_root / f"{face}_manual_part_crop_preview.png"),
                "--slots-csv",
                str(args.results_root / f"{face}_manual_part_slots.csv"),
            ],
        )
    if args.overwrite:
        manual_args.append("--overwrite")
    return manual_args


def run_manual(face: str, manual_args: list[str], dry_run: bool) -> int:
    """Run manual_part_crop.py for one face."""
    command = [sys.executable, str(REPO_ROOT / "capture_data" / "manual_part_crop.py"), *manual_args]
    print()
    print(f"=== fx11 defect crop: {face} ===")
    print(" ".join(shlex.quote(part) for part in command))
    if dry_run:
        return 0
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--face", choices=("top", "bottom", "both"), default="bottom", help="Face to crop.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "dataset" / "fx11_demo",
        help="Raw FX11 full-image dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "dataset" / "fx11_demo_parts_manual",
        help="Output single-part dataset root.",
    )
    parser.add_argument("--hand", default="no_hand", help="Input/output hand directory.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results" / "fx11_demo",
        help="Directory for overlay and slot CSV files. Use 'none' to disable.",
    )
    parser.add_argument(
        "--hole-mask-method",
        choices=("inpaint", "median", "none"),
        default="none",
        help="Hole masking method.",
    )
    parser.add_argument(
        "--nondefect-slots-from-defect-images",
        choices=("skip", "normal"),
        default="skip",
        help="How to handle non-defective slots from defect-labelled full images.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing crop filenames.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser


def main() -> None:
    """Crop FX11 defect images."""
    args = build_parser().parse_args()
    if str(args.results_root).lower() in {"none", "off", "false"}:
        args.results_root = None

    failed = []
    for face in selected_faces(args.face):
        code = run_manual(face, build_manual_args(face, args), args.dry_run)
        if code:
            failed.append((face, code))
            break
    if failed:
        face, code = failed[-1]
        print(f"FX11 defect crop failed for {face}: exit code {code}")
        raise SystemExit(code)


if __name__ == "__main__":
    main()
