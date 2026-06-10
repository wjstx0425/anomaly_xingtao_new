# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage 4: run inference and write review artifacts."""

from __future__ import annotations

import sys

from _common import run_repo_script


HELP = """\
Pipeline stage 4: run inference.

This wrapper forwards arguments to capture_data/inference.py.

Examples:
  .venv/bin/python pipeline/4_inference.py hik_images \\
    --output-root results/c789/left_bottom_parts_anomalydino \\
    --view left_bottom --model anomaly_dino --accelerator gpu

  .venv/bin/python pipeline/4_inference.py dataset/c789_left_bottom_parts/left/bottom_ZS32 \\
    --output-root results/c789/left_bottom_parts_anomalydino \\
    --view left_bottom --model anomaly_dino \\
    --input-is-preprocessed --accelerator gpu

Use capture_data/inference.py for the complete argument reference.
"""


def main() -> None:
    """Forward inference arguments to the inference script."""
    args = sys.argv[1:]
    if not args or args[0] in {"-h", "--help", "help"}:
        print(HELP)
        return
    run_repo_script("capture_data/inference.py", args)


if __name__ == "__main__":
    main()
