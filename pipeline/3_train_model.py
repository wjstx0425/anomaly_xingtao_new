# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage 3: train and evaluate anomaly detection models."""

from __future__ import annotations

import sys

from _common import run_repo_script, with_default_command


HELP = """\
Pipeline stage 3: train and evaluate models.

This wrapper forwards arguments to examples/api/03_models/zs32_defect_workflow.py.
If no workflow subcommand is provided, `all` is inserted automatically.

Workflow subcommands:
  preprocess  Write workflow preprocessed images and manifest.
  train       Train selected models using an existing manifest.
  evaluate    Run prediction and write reports.
  all         Run preprocess, train, and evaluate.

Example:
  .venv/bin/python pipeline/3_train_model.py \
    --data-root dataset/c789_left_bottom_parts \
    --output-root results/c789/left_bottom_parts_anomalydino \
    --views left_bottom --models anomaly_dino \
    --skip-blue-removal --roi full --image-size 392,784 \
    --anomaly-dino-batch-size 1 --eval-batch-size 1 \
    --anomaly-dino-encoder dinov2_vit_small_14 \
    --anomaly-dino-neighbors 1 --accelerator gpu

Use `all --help`, `train --help`, or `evaluate --help` for all options.
"""


WORKFLOW_COMMANDS = {"preprocess", "train", "evaluate", "all"}


def main() -> None:
    """Forward training arguments to the workflow script."""
    args = sys.argv[1:]
    if not args or args[0] in {"-h", "--help", "help"}:
        print(HELP)
        return
    workflow_args = with_default_command(args, WORKFLOW_COMMANDS, "all")
    run_repo_script("examples/api/03_models/zs32_defect_workflow.py", workflow_args)


if __name__ == "__main__":
    main()
