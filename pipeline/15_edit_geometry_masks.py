# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage 15: edit manual C789 geometry masks visually."""

from __future__ import annotations

import sys

from _common import run_repo_script


TARGET_SCRIPT = "capture_data/edit_geometry_masks.py"


HELP = """\
Pipeline stage 15: edit manual C789 geometry masks visually.

This wrapper forwards arguments to capture_data/edit_geometry_masks.py.

Default C789 left_top command:
  .venv/bin/python pipeline/15_edit_geometry_masks.py \\
    --review-pack results/c789_100_hardened/left_top_geometry/manual_review_pack \\
    --template-dir results/c789_100_hardened/left_top_geometry/manual_templates \\
    --stress-root dataset/c789_stress_normal_group_split/locked/left/top \\
    --defect-root dataset/c789_100_left_top_parts/left/top/defect \\
    --anomaly-predictions results/c789_100_hardened/left_top_anomaly_dino/reports/predictions.csv \\
    --stress-output-dir results/c789_100_hardened/left_top_geometry/manual_stress_locked \\
    --defect-output-dir results/c789_100_hardened/left_top_geometry/manual_defect_fused

Common forwarded options:
  --threshold-margin 0.05
  --threshold-mode region
  --tolerance-px 3
  --min-component-area 64
  --hole-dilation 32
  --max-window-width 1900
  --max-window-height 1050
"""


def main() -> None:
    """Forward geometry mask editor arguments."""
    args = sys.argv[1:]
    if not args or args[0] in {"-h", "--help", "help"}:
        print(HELP)
        return
    run_repo_script(TARGET_SCRIPT, args)


if __name__ == "__main__":
    main()
