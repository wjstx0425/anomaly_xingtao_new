#!/usr/bin/env bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${1:-${REPO_ROOT}/dataset/zs32_patchcore_roi}"
RUN_BASE="${2:-${REPO_ROOT}/results/six_view_roi_wrn_l23_r005_bs16_fp32_seed42}"
GPU="${3:-0}"

if [[ ! -f "${DATA_ROOT}/crop_manifest.csv" ]]; then
    echo "Missing cropped dataset manifest: ${DATA_ROOT}/crop_manifest.csv" >&2
    echo "Run pipeline/30_crop_zs32_patchcore_dataset.py convert first." >&2
    exit 1
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PATCHCORE_RUN_SUFFIX="wrn_l23_s256_r005_k9_fp32_bs16_fpr005_seed42"
export PATCHCORE_BACKBONE="wide_resnet50_2"
export PATCHCORE_LAYERS="layer2 layer3"
export PATCHCORE_CORESET_RATIO="0.05"
export PATCHCORE_NUM_NEIGHBORS="9"
export PATCHCORE_PRECISION="float32"
export PATCHCORE_BATCH_SIZE="16"
export EVAL_BATCH_SIZE="16"

echo "PatchCore ROI strong configuration:"
echo "  backbone=${PATCHCORE_BACKBONE}"
echo "  layers=${PATCHCORE_LAYERS}"
echo "  coreset_sampling_ratio=${PATCHCORE_CORESET_RATIO}"
echo "  precision=${PATCHCORE_PRECISION}"
echo "  train_batch_size=${PATCHCORE_BATCH_SIZE}"
echo "  eval_batch_size=${EVAL_BATCH_SIZE}"
echo "  output=${RUN_BASE}"

exec bash "${REPO_ROOT}/pipeline/run_wrn50_fixed_six_views.sh" \
    "${DATA_ROOT}" \
    "${RUN_BASE}" \
    "${GPU}"
