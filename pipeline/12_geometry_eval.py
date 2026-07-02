# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage 12: evaluate C789 slot geometry templates."""

from __future__ import annotations

import sys

from _common import run_repo_script


TARGET_SCRIPT = "capture_data/evaluate_geometry_shape.py"


HELP = """\
Pipeline stage 12: evaluate slot-aware geometry templates.

This wrapper forwards arguments to capture_data/evaluate_geometry_shape.py.

Examples:
  .venv/bin/python pipeline/12_geometry_eval.py \\
    --data-root dataset/c789_stress_normal_group_split/locked/left/top \\
    --template-dir results/c789_100_hardened/left_top_geometry/templates \\
    --output-dir results/c789_100_hardened/left_top_geometry/stress_locked \\
    --calibrate-thresholds

  .venv/bin/python pipeline/12_geometry_eval.py \\
    --data-root dataset/c789_100_left_top_parts/left/top/defect \\
    --template-dir results/c789_100_hardened/left_top_geometry/templates \\
    --thresholds results/c789_100_hardened/left_top_geometry/stress_locked/geometry_thresholds.csv \\
    --anomaly-predictions results/c789_100_hardened/left_top_anomaly_dino/reports/predictions.csv \\
    --output-dir results/c789_100_hardened/left_top_geometry/defect_fused
"""


def main() -> None:
    """Forward geometry evaluation arguments."""
    args = sys.argv[1:]
    if not args or args[0] in {"-h", "--help", "help"}:
        print(HELP)
        return
    run_repo_script(TARGET_SCRIPT, args)


if __name__ == "__main__":
    main()
