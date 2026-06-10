# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage 1: collect raw camera images."""

from __future__ import annotations

import sys

from _common import run_repo_script


HELP = """\
Pipeline stage 1: collect raw camera images.

This wrapper forwards arguments to capture_data/collect_dataset.py.

Examples:
  .venv/bin/python pipeline/1_collect_data.py \
    --hand no_hand --position top --label normal \
    --part-id part001 --group-count 60 --images-per-group 5 \
    --manual-load --hdr --save-hdr-sources --align-hdr \
    --short-exposure 4000 --long-exposure 35000 \
    --short-dark-threshold 80 --long-clip-threshold 245 \
    --blend-width 50 --blur-size 101 --hdr-settle-frames 8 \
    --gain 0 --root ./dataset/fx11_2

  .venv/bin/python pipeline/1_collect_data.py \\
    --hand left --position bottom --label defect --defect-type surface \\
    --part-id part001 --group-count 20 --images-per-group 1 \\
    --manual-load --root ./dataset/c789

Use capture_data/collect_dataset.py for the complete argument reference.
"""


def main() -> None:
    """Forward collection arguments to the capture script."""
    args = sys.argv[1:]
    if not args or args[0] in {"-h", "--help", "help"}:
        print(HELP)
        return
    run_repo_script("capture_data/collect_dataset.py", args)


if __name__ == "__main__":
    main()
