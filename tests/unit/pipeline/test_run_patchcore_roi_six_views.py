# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the strong ROI PatchCore six-view runner."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "pipeline/run_patchcore_roi_six_views.sh"


def test_roi_runner_forwards_strong_configuration_without_training(tmp_path: Path) -> None:
    """The ROI wrapper should select an isolated strong PatchCore experiment."""
    data_root = tmp_path / "cropped"
    data_root.mkdir()
    (data_root / "crop_manifest.csv").touch()
    run_base = tmp_path / "results"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "capture.txt"
    fake_bash = fake_bin / "bash"
    fake_bash.write_text(
        """#!/usr/bin/env sh
set -eu
{
  printf 'args=%s\\n' "$*"
  env | sort
} > "$CAPTURE"
""",
        encoding="utf-8",
    )
    fake_bash.chmod(0o755)
    env = os.environ.copy()
    env.update({"PATH": f"{fake_bin}:{env['PATH']}", "CAPTURE": str(capture)})

    result = subprocess.run(
        ["/usr/bin/bash", str(RUNNER), str(data_root), str(run_base), "0"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    output = capture.read_text(encoding="utf-8")
    assert "run_wrn50_fixed_six_views.sh" in output
    assert f"args={REPO_ROOT}/pipeline/run_wrn50_fixed_six_views.sh {data_root} {run_base} 0" in output
    assert "PATCHCORE_RUN_SUFFIX=wrn_l23_s256_r005_k9_fp32_bs16_fpr005_seed42" in output
    assert "PATCHCORE_BACKBONE=wide_resnet50_2" in output
    assert "PATCHCORE_LAYERS=layer2 layer3" in output
    assert "PATCHCORE_CORESET_RATIO=0.05" in output
    assert "PATCHCORE_NUM_NEIGHBORS=9" in output
    assert "PATCHCORE_PRECISION=float32" in output
    assert "PATCHCORE_BATCH_SIZE=16" in output
    assert "EVAL_BATCH_SIZE=16" in output
