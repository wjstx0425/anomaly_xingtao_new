# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the serial EfficientAD-S/M six-view runner."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "pipeline" / "run_efficientad_s_m_six_views.sh"
VIEWS = (
    "right_front",
    "right_front_left",
    "right_front_right",
    "right_back",
    "right_back_left",
    "right_back_right",
)


def _write_fake_uv(path: Path) -> None:
    """Write a uv shim that materializes deterministic trainer outputs."""
    path.write_text(
        """#!/usr/bin/env bash
set -Eeuo pipefail
if [[ "${1:-}" != "run" || "${2:-}" != "--no-sync" || "${3:-}" != "python" ]]; then
  exit 91
fi
shift 3
if [[ "${1:-}" != */pipeline/8_train_custom_models.py ]]; then
  exec "${REAL_PYTHON}" "$@"
fi
printf '%q ' "$@" >> "${CALL_LOG}"
printf '\n' >> "${CALL_LOG}"
output_root=""
view=""
suffix=""
previous=""
for argument in "$@"; do
  if [[ "$previous" == "--output-root" ]]; then output_root="$argument"; fi
  if [[ "$previous" == "--views" ]]; then view="$argument"; fi
  if [[ "$previous" == "--model-run-suffix" ]]; then suffix="$argument"; fi
  previous="$argument"
done
run_name="efficient_ad_${suffix}"
report_dir="${output_root}/reports/${run_name}"
checkpoint_dir="${output_root}/runs/${view}/${run_name}/version_0"
mkdir -p "$report_dir" "$checkpoint_dir" "${output_root}/preprocessed"
touch "$checkpoint_dir/model.ckpt" "${output_root}/preprocessed/manifest.csv"
cat > "${report_dir}/summary.csv" <<CSV
model,view,deploy_threshold,deploy_target_fpr,image_fpr,image_accuracy,image_recall,image_f1,sample_fpr,sample_accuracy,sample_recall,sample_f1
${run_name},${view},0.42,0.05,0.04,0.90,0.80,0.85,0.03,0.91,0.82,0.86
CSV
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _run(tmp_path: Path) -> tuple[subprocess.CompletedProcess[str], Path, Path, Path]:
    data_root = tmp_path / "data"
    small_root = tmp_path / "small"
    medium_root = tmp_path / "medium"
    call_log = tmp_path / "calls.log"
    fake_uv = tmp_path / "uv"
    data_root.mkdir(exist_ok=True)
    _write_fake_uv(fake_uv)
    env = os.environ.copy()
    env.update({"UV_BIN": str(fake_uv), "REAL_PYTHON": sys.executable, "CALL_LOG": str(call_log)})
    result = subprocess.run(
        ["bash", str(RUNNER), str(data_root), str(small_root), str(medium_root), "0"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result, small_root, medium_root, call_log


def test_runner_trains_six_small_then_six_medium_models(tmp_path: Path) -> None:
    """The runner must launch exactly twelve primary-view jobs in serial order."""
    result, small_root, medium_root, call_log = _run(tmp_path)

    assert result.returncode == 0, result.stdout
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 12
    assert [call.split()[call.split().index("--views") + 1] for call in calls] == [*VIEWS, *VIEWS]
    assert all("secondary" not in call for call in calls)
    assert all("--efficientad-epochs 100" in call for call in calls)
    assert all(r"--image-size 256\,256" in call for call in calls)
    assert all("--max-parallel 1" in call for call in calls)
    assert all("--efficientad-model-size small" in call for call in calls[:6])
    assert all("--efficientad-model-size medium" in call for call in calls[6:])

    for root, size in ((small_root, "small"), (medium_root, "medium")):
        with (root / "six_view_summary.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert [row["view"] for row in rows] == list(VIEWS)
        assert {row["model_size"] for row in rows} == {size}
        assert all(Path(row["checkpoint"]).name == "model.ckpt" for row in rows)


def test_runner_skips_a_complete_model(tmp_path: Path) -> None:
    """A summary plus checkpoint is a durable completion marker on rerun."""
    result, small_root, medium_root, call_log = _run(tmp_path)
    assert result.returncode == 0, result.stdout
    call_log.write_text("", encoding="utf-8")

    result, _, _, call_log = _run(tmp_path)

    assert result.returncode == 0, result.stdout
    assert call_log.read_text(encoding="utf-8") == ""
    assert (small_root / "six_view_summary.csv").is_file()
    assert (medium_root / "six_view_summary.csv").is_file()
