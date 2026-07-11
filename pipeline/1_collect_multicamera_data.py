# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage 1: collect paired ZS32 multi-camera images."""

from __future__ import annotations

import sys

from _common import run_repo_script


HELP = """\
Pipeline stage 1: collect paired six-view ZS32 images (single exposure by default).

This wrapper forwards arguments to capture_data/collect_multicamera_dataset.py.

Examples:
  .venv/bin/python pipeline/1_collect_multicamera_data.py --list-devices
  .venv/bin/python pipeline/1_collect_multicamera_data.py \\
    --hand left --label normal --exposure 4000 --capture-interval 0.2

Views are bound by --front-serial/--left-serial/--right-serial. Add --hdr
to use the retained short/long exposure fusion mode.

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
