# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the fixed six-view PatchCore runner."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "pipeline" / "run_wrn50_fixed_six_views.sh"
SUFFIX = "wrn_l2_s256_r001_k9_fp16_fpr005_seed42"
VIEWS = (
    "right_front",
    "right_front_left",
    "right_front_right",
    "right_back",
    "right_back_left",
    "right_back_right",
)
EIGHT_VIEWS = (
    "right_front",
    "right_front_left",
    "right_front_right",
    "right_front_secondary",
    "right_back",
    "right_back_left",
    "right_back_right",
    "right_back_secondary",
)


def _write_fake_python(path: Path) -> None:
    """Write a trainer shim that emits deterministic fake artifacts."""
    path.write_text(
        """#!/usr/bin/env bash
set -Eeuo pipefail
if [[ "${1:-}" != */pipeline/8_train_custom_models.py ]]; then
  exec "${REAL_PYTHON}" "$@"
fi
printf '%q ' "$@" >> "${CALL_LOG}"
printf '\n' >> "${CALL_LOG}"
output_root=""
view=""
previous=""
for argument in "$@"; do
  if [[ "$previous" == "--output-root" ]]; then output_root="$argument"; fi
  if [[ "$previous" == "--views" ]]; then view="$argument"; fi
  previous="$argument"
done
if [[ "$view" == "${FAIL_VIEW:-}" ]]; then exit 23; fi
report_dir="${output_root}/reports/patchcore_wrn_l2_s256_r001_k9_fp16_fpr005_seed42"
checkpoint_dir="${output_root}/runs/${view}/patchcore_wrn_l2_s256_r001_k9_fp16_fpr005_seed42/version_0/weights/lightning"
mkdir -p "$report_dir" "$checkpoint_dir" "${output_root}/preprocessed"
touch "$checkpoint_dir/model.ckpt" "${output_root}/preprocessed/manifest.csv"
cat > "${report_dir}/summary.csv" <<CSV
model,view,deploy_threshold,deploy_target_fpr,image_fpr,image_accuracy,image_recall,image_f1,sample_fpr,sample_accuracy,sample_recall,sample_f1
patchcore_wrn_l2_s256_r001_k9_fp16_fpr005_seed42,${view},0.42,0.05,0.04,0.90,0.80,0.85,0.03,0.91,0.82,0.86
CSV
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_completed_artifacts(run_base: Path, view: str, summary: str | None = None) -> None:
    """Write the durable artifacts required for a completed view."""
    report_dir = run_base / view / "reports" / f"patchcore_{SUFFIX}"
    checkpoint = run_base / view / "runs" / view / f"patchcore_{SUFFIX}" / "version_0" / "model.ckpt"
    report_dir.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    if summary is None:
        summary = (
            "model,view,deploy_threshold,deploy_target_fpr,image_fpr,image_accuracy,image_recall,image_f1,"
            "sample_fpr,sample_accuracy,sample_recall,sample_f1\n"
            f"patchcore_{SUFFIX},{view},0.42,0.05,0.04,0.90,0.80,0.85,0.03,0.91,0.82,0.86\n"
        )
    (report_dir / "summary.csv").write_text(summary, encoding="utf-8")


def _run_runner(
    tmp_path: Path,
    *,
    fail_view: str = "",
    env_overrides: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    data_root = tmp_path / "right"
    run_base = tmp_path / "results"
    data_root.mkdir()
    fake_python = tmp_path / "fake-python"
    call_log = tmp_path / "calls.log"
    _write_fake_python(fake_python)
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(fake_python),
            "REAL_PYTHON": sys.executable,
            "CALL_LOG": str(call_log),
            "FAIL_VIEW": fail_view,
            "HF_HUB_OFFLINE": "1",
        },
    )
    env.update(env_overrides or {})
    result = subprocess.run(
        ["bash", str(RUNNER), str(data_root), str(run_base), "0"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result, run_base, call_log


def test_runner_uses_fixed_arguments_and_merges_all_views(tmp_path: Path) -> None:
    """Every fixed view should run serially and contribute one summary row."""
    result, run_base, call_log = _run_runner(tmp_path)

    assert result.returncode == 0, result.stdout
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 6
    assert [call.split()[call.split().index("--views") + 1] for call in calls] == list(VIEWS)
    for call in calls:
        assert "--models patchcore" in call
        assert f"--model-run-suffix {SUFFIX}" in call
        assert "--gpus 0 --max-parallel 1 --accelerator gpu --devices-per-job 1" in call
        assert r"--roi full --image-size 256\,256 --normal-test-ratio 0.2" in call
        assert "--patchcore-backbone wide_resnet50_2 --patchcore-layers layer2" in call
        assert "--patchcore-coreset-ratio 0.01 --patchcore-num-neighbors 9" in call
        assert "--patchcore-precision float16 --patchcore-batch-size 4" in call
        assert "--eval-batch-size 4 --num-workers 2 --deploy-fpr 0.05 --seed 42" in call

    summary_path = run_base / "six_view_summary.csv"
    with summary_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [row["view"] for row in rows] == list(VIEWS)
    assert rows[0]["deploy_threshold"] == "0.42"
    assert rows[0]["checkpoint"].endswith("model.ckpt")
    assert rows[0]["summary_csv"].endswith("summary.csv")


def test_runner_accepts_opt_in_patchcore_parameter_overrides(tmp_path: Path) -> None:
    """An ROI wrapper may strengthen PatchCore without changing fixed defaults."""
    result, _run_base, call_log = _run_runner(
        tmp_path,
        env_overrides={
            "PATCHCORE_LAYERS": "layer2 layer3",
            "PATCHCORE_CORESET_RATIO": "0.05",
            "PATCHCORE_PRECISION": "float32",
            "PATCHCORE_BATCH_SIZE": "16",
            "EVAL_BATCH_SIZE": "16",
        },
    )

    assert result.returncode == 0, result.stdout
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 6
    for call in calls:
        assert "--patchcore-layers layer2 layer3" in call
        assert "--patchcore-coreset-ratio 0.05" in call
        assert "--patchcore-precision float32 --patchcore-batch-size 16" in call
        assert "--eval-batch-size 16" in call


def test_runner_accepts_eight_view_selection_and_summary_name(tmp_path: Path) -> None:
    """The reusable serial runner should train and summarize all eight right-hand views."""
    result, run_base, call_log = _run_runner(
        tmp_path,
        env_overrides={
            "PATCHCORE_VIEWS": " ".join(EIGHT_VIEWS),
            "PATCHCORE_SUMMARY_NAME": "eight_view_summary.csv",
        },
    )

    assert result.returncode == 0, result.stdout
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert [call.split()[call.split().index("--views") + 1] for call in calls] == list(EIGHT_VIEWS)
    with (run_base / "eight_view_summary.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [row["view"] for row in rows] == list(EIGHT_VIEWS)
    assert not (run_base / "six_view_summary.csv").exists()


def test_runner_selects_resume_modes_from_existing_artifacts(tmp_path: Path) -> None:
    """Summary, checkpoint, and manifest artifacts should select distinct resume paths."""
    run_base = tmp_path / "results"
    _write_completed_artifacts(run_base, VIEWS[0])
    checkpoint = run_base / VIEWS[1] / "runs" / VIEWS[1] / f"patchcore_{SUFFIX}" / "model.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    manifest = run_base / VIEWS[2] / "preprocessed" / "manifest.csv"
    manifest.parent.mkdir(parents=True)
    manifest.touch()

    result, _actual_run_base, call_log = _run_runner(tmp_path)

    assert result.returncode == 0, result.stdout
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 5
    assert not any("--views right_front " in call for call in calls)
    assert "--evaluate-only" in next(call for call in calls if "--views right_front_left" in call)
    assert "--skip-preprocess" in next(call for call in calls if "--views right_front_right" in call)


def test_runner_rejects_invalid_completed_summary(tmp_path: Path) -> None:
    """An invalid skipped summary must prevent creation of the merged report."""
    run_base = tmp_path / "results"
    summary_without_target_fpr = (
        "model,view,deploy_threshold,image_fpr,image_accuracy,image_recall,image_f1,"
        "sample_fpr,sample_accuracy,sample_recall,sample_f1\n"
        f"patchcore_{SUFFIX},{VIEWS[0]},0.42,0.04,0.90,0.80,0.85,0.03,0.91,0.82,0.86\n"
    )
    _write_completed_artifacts(run_base, VIEWS[0], summary_without_target_fpr)

    result, _actual_run_base, _call_log = _run_runner(tmp_path)

    assert result.returncode != 0
    assert "Invalid summary" in result.stdout
    assert "deploy_target_fpr" in result.stdout
    assert not (run_base / "six_view_summary.csv").exists()


def test_runner_records_failure_and_continues_later_views(tmp_path: Path) -> None:
    """A failed view should be durable while later views still run."""
    result, run_base, call_log = _run_runner(tmp_path, fail_view=VIEWS[2])

    assert result.returncode != 0
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 6
    assert any(f"--views {VIEWS[-1]}" in call for call in calls)
    failures = (run_base / "failed_views.txt").read_text(encoding="utf-8")
    assert VIEWS[2] in failures
    assert "exit_code=23" in failures
    assert not (run_base / "six_view_summary.csv").exists()
