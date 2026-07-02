# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage 10: build a clean-plus-stress-train Folder dataset."""

from __future__ import annotations

import sys

from _common import run_repo_script


TARGET_SCRIPT = "capture_data/build_hardened_dataset.py"


HELP = """\
Pipeline stage 10: build a hardened training dataset.

This wrapper forwards arguments to capture_data/build_hardened_dataset.py.
It merges clean normal/normal_test/defect with stress train normal only.
Locked stress normal stays outside the training root.

Example:
  .venv/bin/python pipeline/10_build_hardened_dataset.py \\
    --clean-root dataset/c789_100_left_top_parts \\
    --stress-split-root dataset/c789_stress_normal_group_split \\
    --output-root dataset/c789_100_left_top_hardened_parts \\
    --hand left --position top \\
    --link-mode symlink --overwrite
"""


def main() -> None:
    """Forward hardened dataset arguments."""
    args = sys.argv[1:]
    if not args or args[0] in {"-h", "--help", "help"}:
        print(HELP)
        return
    run_repo_script(TARGET_SCRIPT, args)


if __name__ == "__main__":
    main()
