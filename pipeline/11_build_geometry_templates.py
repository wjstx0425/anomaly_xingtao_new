# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage 11: build C789 slot geometry templates."""

from __future__ import annotations

import sys

from _common import run_repo_script


TARGET_SCRIPT = "capture_data/build_geometry_templates.py"


HELP = """\
Pipeline stage 11: build slot-aware geometry templates.

This wrapper forwards arguments to capture_data/build_geometry_templates.py.

Example:
  .venv/bin/python pipeline/11_build_geometry_templates.py \\
    --data-root dataset/c789_100_left_top_hardened_parts \\
    --view left_top \\
    --preset c789_left_top_3x2 \\
    --output-dir results/c789_100_hardened/left_top_geometry/templates
"""


def main() -> None:
    """Forward geometry template build arguments."""
    args = sys.argv[1:]
    if not args or args[0] in {"-h", "--help", "help"}:
        print(HELP)
        return
    run_repo_script(TARGET_SCRIPT, args)


if __name__ == "__main__":
    main()
