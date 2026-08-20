#!/usr/bin/env bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PATCHCORE_VIEWS="right_front right_front_left right_front_right right_front_secondary right_back right_back_left right_back_right right_back_secondary"
export PATCHCORE_SUMMARY_NAME="eight_view_summary.csv"

exec bash "${REPO_ROOT}/pipeline/run_patchcore_roi_six_views.sh" "$@"
