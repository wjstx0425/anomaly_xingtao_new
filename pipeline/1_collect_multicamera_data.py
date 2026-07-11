# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage 1: collect paired ZS32 multi-camera HDR images."""

from __future__ import annotations

import sys

from _common import run_repo_script


HELP = """\
Pipeline stage 1: collect paired six-view ZS32 HDR images.

This wrapper forwards arguments to capture_data/collect_multicamera_dataset.py.

Examples:
  .venv/bin/python pipeline/1_collect_multicamera_data.py --list-devices
  .venv/bin/python pipeline/1_collect_multicamera_data.py \\
    --devices 0 1 2 --hand left --label normal --hdr

Use capture_data/collect_multicamera_dataset.py --help for the complete argument reference.
"""


def main() -> None:
    """Forward multi-camera collection arguments unchanged."""
    args = sys.argv[1:]
    if not args or args[0] in {"-h", "--help", "help"}:
        print(HELP)
        return
    run_repo_script("capture_data/collect_multicamera_dataset.py", args)


if __name__ == "__main__":
    main()
