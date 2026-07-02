# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage 13: export manual geometry review masks and sheets."""

from __future__ import annotations

import sys

from _common import run_repo_script


TARGET_SCRIPT = "capture_data/export_geometry_review_pack.py"


HELP = """\
Pipeline stage 13: export manual geometry review pack.

This wrapper forwards arguments to capture_data/export_geometry_review_pack.py.

Example:
  .venv/bin/python pipeline/13_export_geometry_review_pack.py \\
    --normal-root dataset/c789_100_left_top_hardened_parts/left/top/normal \\
    --stress-root dataset/c789_stress_normal_group_split/locked/left/top \\
    --defect-root dataset/c789_100_left_top_parts/left/top/defect \\
    --template-dir results/c789_100_hardened/left_top_geometry/templates \\
    --output-dir results/c789_100_hardened/left_top_geometry/manual_review_pack \\
    --preset c789_left_top_3x2
"""


def main() -> None:
    """Forward review-pack export arguments."""
    args = sys.argv[1:]
    if not args or args[0] in {"-h", "--help", "help"}:
        print(HELP)
        return
    run_repo_script(TARGET_SCRIPT, args)


if __name__ == "__main__":
    main()
