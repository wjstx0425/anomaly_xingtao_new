# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage 9: split stress-normal crops into train and locked groups."""

from __future__ import annotations

import sys

from _common import run_repo_script


TARGET_SCRIPT = "capture_data/prepare_stress_splits.py"


HELP = """\
Pipeline stage 9: split stress-normal crops by source group.

This wrapper forwards arguments to capture_data/prepare_stress_splits.py.

Example:
  .venv/bin/python pipeline/9_split_stress_normal.py \\
    --input-root dataset/c789_stress_normal_parts \\
    --output-root dataset/c789_stress_normal_group_split \\
    --hand left --position top \\
    --train-ratio 0.7 --link-mode symlink --overwrite
"""


def main() -> None:
    """Forward stress split arguments."""
    args = sys.argv[1:]
    if not args or args[0] in {"-h", "--help", "help"}:
        print(HELP)
        return
    run_repo_script(TARGET_SCRIPT, args)


if __name__ == "__main__":
    main()
