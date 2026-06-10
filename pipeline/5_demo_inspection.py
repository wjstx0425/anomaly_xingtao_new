# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage 5: run the two-sided inspection demo."""

from __future__ import annotations

import sys

from _common import run_repo_script


HELP = """\
Pipeline stage 5: run the two-sided inspection demo.

This wrapper forwards arguments to capture_data/demo_inspection.py.

Examples:
  .venv/bin/python pipeline/5_demo_inspection.py \\
    --device 0 --accelerator gpu \\
    --threshold-profile demo \\
    --predict-batch-size 1 \\
    --quality-gate warn

  .venv/bin/python pipeline/5_demo_inspection.py \\
    --part-profile c789 \\
    --device 0 --accelerator gpu

  .venv/bin/python pipeline/5_demo_inspection.py \\
    --part-profile c789 \\
    --demo-top-image dataset/c789/left/top/normal/example.png \\
    --demo-bottom-image dataset/c789/left/bottom/normal/example.png \\
    --mock-predictions --mock-defects top:slot02 \\
    --auto-run --no-gui \\
    --save-ui-screenshot results/c789/demo_inspection/ui_preview.png

Use capture_data/demo_inspection.py for the complete argument reference.
"""


def main() -> None:
    """Forward demo arguments to the inspection script."""
    args = sys.argv[1:]
    if not args or args[0] in {"-h", "--help", "help"}:
        print(HELP)
        return
    run_repo_script("capture_data/demo_inspection.py", args)


if __name__ == "__main__":
    main()
