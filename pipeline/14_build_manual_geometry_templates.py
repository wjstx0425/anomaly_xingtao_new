# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage 14: build geometry templates from manual mask PNGs."""

from __future__ import annotations

import sys

from _common import run_repo_script


TARGET_SCRIPT = "capture_data/build_manual_geometry_templates.py"


HELP = """\
Pipeline stage 14: build manual geometry templates.

This wrapper forwards arguments to capture_data/build_manual_geometry_templates.py.

Example:
  .venv/bin/python pipeline/14_build_manual_geometry_templates.py \\
    --mask-dir results/c789_100_hardened/left_top_geometry/manual_review_pack/manual_masks \\
    --output-dir results/c789_100_hardened/left_top_geometry/manual_templates
"""


def main() -> None:
    """Forward manual geometry template build arguments."""
    args = sys.argv[1:]
    if not args or args[0] in {"-h", "--help", "help"}:
        print(HELP)
        return
    run_repo_script(TARGET_SCRIPT, args)


if __name__ == "__main__":
    main()
