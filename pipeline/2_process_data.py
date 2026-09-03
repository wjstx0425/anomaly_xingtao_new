# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

r"""Pipeline stage 2: crop full images into single-part datasets.

Quick crop commands:

    .venv/bin/python pipeline/2_process_data.py auto \
      --data-root dataset/c789 \
      --output-root dataset/c789_left_top_parts \
      --hand left --position top \
      --preset c789_left_top_3x2 \
      --hole-mask-method inpaint --overwrite

    .venv/bin/python pipeline/2_process_data.py auto \
      --data-root dataset/c789 \
      --output-root dataset/c789_left_bottom_parts \
      --hand left --position bottom --output-position bottom \
      --preset c789_left_bottom_3x2 \
      --hole-mask-method inpaint --overwrite

    .venv/bin/python pipeline/2_process_data.py auto \
      --data-root dataset/fx11_demo \
      --output-root dataset/fx11_demo_parts \
      --hand no_hand --position top \
      --preset fx11_no_hand_top_6x1 \
      --hole-mask-method none --overwrite

    .venv/bin/python pipeline/2_process_data.py auto \
      --data-root dataset/fx11_demo \
      --output-root dataset/fx11_demo_parts \
      --hand no_hand --position bottom \
      --preset fx11_no_hand_bottom_6x1 \
      --hole-mask-method none --overwrite

    .venv/bin/python pipeline/2_process_data.py fx11-defect \
      --face both \
      --data-root dataset/fx11_demo \
      --output-root dataset/fx11_demo_parts_manual \
      --overwrite
"""

from __future__ import annotations

import sys

from _common import run_repo_script


HELP = r"""Pipeline stage 2: process raw data into single-part datasets.

By default, normal full images are split into normal/normal_test at 3:1
when the input dataset does not already contain normal_test images.

Modes:
  auto    Use capture_data/prepare_part_crops.py presets. This is the default.
  manual  Use capture_data/manual_part_crop.py to draw crop boxes with the mouse,
          or pass fixed --slot boxes directly.
  fx11-defect
          Crop FX11 defect images with fixed top/bottom slot boxes.

Examples:
  # C789 left/top, fixed 3x2 crop preset:
  .venv/bin/python pipeline/2_process_data.py auto \
    --data-root dataset/c789 --output-root dataset/c789_left_top_parts \
    --hand left --position top --preset c789_left_top_3x2 \
    --hole-mask-method inpaint \
    --preview-overlay results/c789/left_top_part_crop_preview.png \
    --overwrite

  # C789 left/bottom:
  .venv/bin/python pipeline/2_process_data.py auto \
    --data-root dataset/c789 --output-root dataset/c789_left_bottom_parts \
    --hand left --position bottom --output-position bottom \
    --preset c789_left_bottom_3x2 \
    --preview-overlay results/c789/left_bottom_part_crop_preview.png \
    --overwrite

  # FX11 top/bottom, fixed 6x1 crop presets:
  .venv/bin/python pipeline/2_process_data.py auto \
    --data-root dataset/fx11_demo --output-root dataset/fx11_demo_parts \
    --hand no_hand --position top --preset fx11_no_hand_top_6x1 \
    --hole-mask-method none --overwrite

  .venv/bin/python pipeline/2_process_data.py auto \
    --data-root dataset/fx11_demo --output-root dataset/fx11_demo_parts \
    --hand no_hand --position bottom --preset fx11_no_hand_bottom_6x1 \
    --hole-mask-method none --overwrite

  # Manual crop selection when fixed boxes are not accurate:
  .venv/bin/python pipeline/2_process_data.py manual \
    --data-root dataset/fx11_demo --output-root dataset/fx11_demo_parts_manual \
    --hand no_hand --position bottom --slot-count 6 --order vertical \
    --save-overlay results/fx11_demo/manual_part_crop_preview.png \
    --slots-csv results/fx11_demo/manual_part_slots.csv --overwrite

  .venv/bin/python pipeline/2_process_data.py manual \
    --data-root dataset/fx11_demo --output-root dataset/fx11_demo_parts_manual \
    --hand no_hand --position bottom \
    --slot slot01:1,1,832,43,3200,446 \
    --slot slot02:2,1,835,440,3209,853 \
    --slot slot03:3,1,838,893,3206,1296 \
    --slot slot04:4,1,844,1293,3221,1694 \
    --slot slot05:5,1,844,1700,3221,2107 \
    --slot slot06:6,1,847,2137,3230,2587 \
    --save-overlay results/fx11_demo/manual_part_crop_preview.png \
    --slots-csv results/fx11_demo/manual_part_slots.csv --overwrite

  # FX11 defect-only helper. Filename numbers map to top-to-bottom slots:
  .venv/bin/python pipeline/2_process_data.py fx11-defect \
    --face bottom --overwrite

Use `auto --help`, `manual --help`, or `fx11-defect --help` for the complete argument reference.
"""


MODE_TO_SCRIPT = {
    "auto": "capture_data/prepare_part_crops.py",
    "fx11-defect": "pipeline/fx11_defect_crop.py",
    "manual": "capture_data/manual_part_crop.py",
}


def _resolve_mode(args: list[str]) -> tuple[str, list[str]]:
    """Return the processing mode and arguments for the target script."""
    if not args:
        return "help", []
    if args[0] in {"-h", "--help", "help"}:
        return "help", []
    if args[0] in MODE_TO_SCRIPT:
        return args[0], args[1:]
    if args[0].startswith("-"):
        return "auto", args
    msg = f"Unknown process mode: {args[0]!r}. Use auto or manual."
    raise SystemExit(msg)


def main() -> None:
    """Run the selected processing mode."""
    mode, target_args = _resolve_mode(sys.argv[1:])
    if mode == "help":
        print(HELP)
        return
    run_repo_script(MODE_TO_SCRIPT[mode], target_args)


if __name__ == "__main__":
    main()
