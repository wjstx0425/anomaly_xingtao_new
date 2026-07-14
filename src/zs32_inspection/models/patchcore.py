# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PatchCore adapter contract; real Anomalib work is injected on Linux."""

from .base import AnomalyFamily, DelegatingAnomalyAdapter


class PatchCoreAdapter(DelegatingAnomalyAdapter):
    """ZS32 PatchCore adapter with content-addressed load boundaries."""

    family = AnomalyFamily.PATCHCORE
