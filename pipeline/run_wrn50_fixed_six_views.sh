#!/usr/bin/env bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if (( $# < 1 || $# > 3 )); then
    echo "Usage: $0 DATA_ROOT [RUN_BASE] [GPU]" >&2
    exit 2
fi

DATA_ROOT="$1"
RUN_BASE="${2:-${REPO_ROOT}/results/six_view_fixed_seed42}"
GPU="${3:-0}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
TRAINER="${REPO_ROOT}/pipeline/8_train_custom_models.py"
SUFFIX="${PATCHCORE_RUN_SUFFIX:-wrn_l2_s256_r001_k9_fp16_fpr005_seed42}"
PATCHCORE_BACKBONE="${PATCHCORE_BACKBONE:-wide_resnet50_2}"
PATCHCORE_LAYER_NAMES="${PATCHCORE_LAYERS:-layer2}"
read -r -a PATCHCORE_LAYER_ARGS <<< "$PATCHCORE_LAYER_NAMES"
PATCHCORE_CORESET_RATIO="${PATCHCORE_CORESET_RATIO:-0.01}"
PATCHCORE_NUM_NEIGHBORS="${PATCHCORE_NUM_NEIGHBORS:-9}"
PATCHCORE_PRECISION="${PATCHCORE_PRECISION:-float16}"
PATCHCORE_BATCH_SIZE="${PATCHCORE_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
VIEWS=(
    right_front
    right_front_left
    right_front_right
    right_back
    right_back_left
    right_back_right
)

mkdir -p "$RUN_BASE"
failed_views=()

for view in "${VIEWS[@]}"; do
    view_root="${RUN_BASE}/${view}"
    summary_csv="${view_root}/reports/patchcore_${SUFFIX}/summary.csv"
    runner_log="${view_root}/runner.log"
    manifest="${view_root}/preprocessed/manifest.csv"
    mkdir -p "$view_root"

    if [[ -f "$summary_csv" ]]; then
        printf '[skip] %s already has %s\n' "$view" "$summary_csv" | tee -a "$runner_log"
        continue
    fi

    resume_args=()
    checkpoint=""
    checkpoint_root="${view_root}/runs/${view}/patchcore_${SUFFIX}"
    if [[ -d "$checkpoint_root" ]]; then
        checkpoint="$(find "$checkpoint_root" -type f -name '*.ckpt' -print -quit)"
    fi
    if [[ -n "$checkpoint" ]]; then
        resume_args+=(--evaluate-only)
        printf '[resume] %s evaluating checkpoint %s\n' "$view" "$checkpoint" | tee -a "$runner_log"
    elif [[ -f "$manifest" ]]; then
        resume_args+=(--skip-preprocess)
        printf '[resume] %s reusing %s\n' "$view" "$manifest" | tee -a "$runner_log"
    else
        printf '[start] %s full preprocess, train, and evaluate\n' "$view" | tee -a "$runner_log"
    fi

    command=(
        "$PYTHON_BIN" "$TRAINER"
        --data-root "$DATA_ROOT"
        --output-root "$view_root"
        --views "$view"
        --models patchcore
        --model-run-suffix "$SUFFIX"
        --gpus "$GPU" --max-parallel 1 --accelerator gpu --devices-per-job 1
        --roi full --image-size 256,256 --normal-test-ratio 0.2
        --patchcore-backbone "$PATCHCORE_BACKBONE" --patchcore-layers "${PATCHCORE_LAYER_ARGS[@]}"
        --patchcore-coreset-ratio "$PATCHCORE_CORESET_RATIO" --patchcore-num-neighbors "$PATCHCORE_NUM_NEIGHBORS"
        --patchcore-precision "$PATCHCORE_PRECISION" --patchcore-batch-size "$PATCHCORE_BATCH_SIZE"
        --eval-batch-size "$EVAL_BATCH_SIZE" --num-workers 2 --deploy-fpr 0.05 --seed 42
        "${resume_args[@]}"
    )

    printf '[command] ' | tee -a "$runner_log"
    printf '%q ' "${command[@]}" | tee -a "$runner_log"
    printf '\n' | tee -a "$runner_log"
    if "${command[@]}" > >(tee -a "$runner_log") 2>&1; then
        printf '[success] %s\n' "$view" | tee -a "$runner_log"
    else
        exit_code=$?
        failed_views+=("$view")
        printf '%s exit_code=%s\n' "$view" "$exit_code" | tee -a "$RUN_BASE/failed_views.txt" "$runner_log"
    fi
done

all_summaries=true
for view in "${VIEWS[@]}"; do
    if [[ ! -f "${RUN_BASE}/${view}/reports/patchcore_${SUFFIX}/summary.csv" ]]; then
        all_summaries=false
        break
    fi
done

merged_summary="${RUN_BASE}/six_view_summary.csv"
if [[ "$all_summaries" == true && ! -e "$merged_summary" ]]; then
    "$PYTHON_BIN" - "$RUN_BASE" "$SUFFIX" "${VIEWS[@]}" <<'PY'
import csv
import sys
from pathlib import Path

run_base = Path(sys.argv[1])
suffix = sys.argv[2]
views = sys.argv[3:]
rows = []
expected_model = f"patchcore_{suffix}"
required_fields = (
    "deploy_threshold",
    "deploy_target_fpr",
    "image_fpr",
    "image_accuracy",
    "image_recall",
    "image_f1",
    "sample_fpr",
    "sample_accuracy",
    "sample_recall",
    "sample_f1",
)
for view in views:
    summary_csv = run_base / view / "reports" / f"patchcore_{suffix}" / "summary.csv"
    with summary_csv.open(newline="", encoding="utf-8") as file:
        source_rows = list(csv.DictReader(file))
    if len(source_rows) != 1:
        raise SystemExit(f"Invalid summary {summary_csv}: expected exactly one data row")
    row = source_rows[0]
    if row.get("model", "").strip() != expected_model:
        raise SystemExit(f"Invalid summary {summary_csv}: model must be {expected_model}")
    if row.get("view", "").strip() != view:
        raise SystemExit(f"Invalid summary {summary_csv}: view must be {view}")
    missing_fields = [field for field in required_fields if not row.get(field, "").strip()]
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise SystemExit(f"Invalid summary {summary_csv}: missing or empty fields: {missing}")
    checkpoint_root = run_base / view / "runs" / view / f"patchcore_{suffix}"
    checkpoints = sorted(checkpoint_root.rglob("*.ckpt")) if checkpoint_root.is_dir() else []
    if not checkpoints:
        raise SystemExit(f"Invalid completed artifacts for {view}: no checkpoint under {checkpoint_root}")
    row = dict(row)
    row["checkpoint"] = str(checkpoints[0])
    row["summary_csv"] = str(summary_csv)
    rows.append(row)

preferred_fields = [
    "view",
    "model",
    "deploy_threshold",
    "deploy_target_fpr",
    "image_fpr",
    "image_accuracy",
    "image_recall",
    "image_f1",
    "sample_fpr",
    "sample_accuracy",
    "sample_recall",
    "sample_f1",
    "checkpoint",
    "summary_csv",
]
extra_fields = sorted({key for row in rows for key in row} - set(preferred_fields))
output_path = run_base / "six_view_summary.csv"
with output_path.open("x", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=[*preferred_fields, *extra_fields])
    writer.writeheader()
    writer.writerows(rows)
PY
elif [[ "$all_summaries" == true ]]; then
    printf '[skip] merged summary already exists: %s\n' "$merged_summary"
fi

if (( ${#failed_views[@]} > 0 )); then
    printf 'Failed views: %s\n' "${failed_views[*]}" >&2
    exit 1
fi

if [[ "$all_summaries" != true ]]; then
    echo "Not all per-view summaries exist; merged summary was not written." >&2
    exit 1
fi
