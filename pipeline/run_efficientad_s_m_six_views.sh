#!/usr/bin/env bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

if (( $# < 1 || $# > 4 )); then
    echo "Usage: $0 DATA_ROOT [SMALL_OUTPUT_ROOT] [MEDIUM_OUTPUT_ROOT] [GPU]" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="$1"
SMALL_OUTPUT_ROOT="${2:-${REPO_ROOT}/results/zs32_efficientad_s_six_view_seed42_v1}"
MEDIUM_OUTPUT_ROOT="${3:-${REPO_ROOT}/results/zs32_efficientad_m_six_view_seed42_v1}"
GPU="${4:-0}"
UV_BIN="${UV_BIN:-uv}"
TRAINER="${REPO_ROOT}/pipeline/8_train_custom_models.py"
IMAGENET_DIR="${IMAGENET_DIR:-/home/yunjing/anomalib/.cache/anomalib/imagenette/imagenette2}"

VIEWS=(
    right_front
    right_front_left
    right_front_right
    right_back
    right_back_left
    right_back_right
)

checkpoint_for() {
    local output_root="$1"
    local view="$2"
    local run_name="$3"
    find "${output_root}/runs/${view}/${run_name}" -type f -name model.ckpt -print -quit 2>/dev/null || true
}

merge_family_summary() {
    local output_root="$1"
    local model_size="$2"
    local suffix="$3"
    shift 3
    if [[ -e "${output_root}/six_view_summary.csv" ]]; then
        echo "[skip] merged summary already exists: ${output_root}/six_view_summary.csv"
        return
    fi
    "$UV_BIN" run --no-sync python - "$output_root" "$model_size" "$suffix" "$@" <<'PY'
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

output_root = Path(sys.argv[1]).resolve()
model_size = sys.argv[2]
suffix = sys.argv[3]
views = sys.argv[4:]
run_name = f"efficient_ad_{suffix}"
rows: list[dict[str, str]] = []

for view in views:
    view_root = output_root / view
    summary_path = view_root / "reports" / run_name / "summary.csv"
    checkpoints = list((view_root / "runs" / view / run_name).glob("**/model.ckpt"))
    if not summary_path.is_file() or len(checkpoints) != 1:
        raise SystemExit(f"Incomplete output for {model_size}:{view}")
    with summary_path.open(newline="", encoding="utf-8") as stream:
        summary_rows = list(csv.DictReader(stream))
    if len(summary_rows) != 1:
        raise SystemExit(f"Expected one summary row for {model_size}:{view}")
    checkpoint = checkpoints[0].resolve()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    row = dict(summary_rows[0])
    row.update(
        model_size=model_size,
        checkpoint=str(checkpoint),
        checkpoint_sha256=digest,
        summary_csv=str(summary_path.resolve()),
    )
    rows.append(row)

destination = output_root / "six_view_summary.csv"
destination.parent.mkdir(parents=True, exist_ok=True)
fieldnames = list(rows[0])
with destination.open("x", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"[summary] {destination}")
PY
}

run_family() {
    local model_size="$1"
    local output_root="$2"
    local suffix="${model_size}_s256_fp32_e100_seed42"
    local run_name="efficient_ad_${suffix}"
    local failures=0

    mkdir -p "$output_root"
    for view in "${VIEWS[@]}"; do
        local view_root="${output_root}/${view}"
        local summary="${view_root}/reports/${run_name}/summary.csv"
        local checkpoint
        checkpoint="$(checkpoint_for "$view_root" "$view" "$run_name")"

        if [[ -f "$summary" && -n "$checkpoint" ]]; then
            echo "[skip] ${model_size}:${view} already complete"
            continue
        fi

        local resume_args=()
        if [[ -n "$checkpoint" ]]; then
            resume_args+=(--evaluate-only)
            echo "[resume] ${model_size}:${view} checkpoint exists; evaluate only"
        elif [[ -f "${view_root}/preprocessed/manifest.csv" ]]; then
            resume_args+=(--skip-preprocess)
            echo "[resume] ${model_size}:${view} preprocessed data exists"
        else
            echo "[train] ${model_size}:${view}"
        fi

        if ! "$UV_BIN" run --no-sync python "$TRAINER" \
            --data-root "$DATA_ROOT" \
            --output-root "$view_root" \
            --views "$view" \
            --models efficient_ad \
            --model-run-suffix "$suffix" \
            --efficientad-model-size "$model_size" \
            --gpus "$GPU" \
            --max-parallel 1 \
            --accelerator gpu \
            --devices-per-job 1 \
            --roi full \
            --image-size 256,256 \
            --efficientad-batch-size 1 \
            --efficientad-epochs 100 \
            --imagenet-dir "$IMAGENET_DIR" \
            --num-workers 8 \
            --eval-batch-size 1 \
            --normal-test-ratio 0.2 \
            --deploy-fpr 0.05 \
            --seed 42 \
            "${resume_args[@]}"; then
            echo "${model_size}:${view}" | tee -a "${output_root}/failed_models.txt" >&2
            failures=1
        fi
    done

    if (( failures != 0 )); then
        return 1
    fi
    merge_family_summary "$output_root" "$model_size" "$suffix" "${VIEWS[@]}"
}

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "Dataset directory does not exist: $DATA_ROOT" >&2
    exit 2
fi
if [[ ! -d "$IMAGENET_DIR" ]]; then
    echo "ImageNette directory does not exist: $IMAGENET_DIR" >&2
    exit 2
fi

run_family small "$SMALL_OUTPUT_ROOT"
run_family medium "$MEDIUM_OUTPUT_ROOT"

echo "EfficientAD-S: $SMALL_OUTPUT_ROOT"
echo "EfficientAD-M: $MEDIUM_OUTPUT_ROOT"
