# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""AnomalyDINO adapter contract; real Anomalib work is injected on Linux."""

from .base import AnomalyFamily, DelegatingAnomalyAdapter


class AnomalyDINOAdapter(DelegatingAnomalyAdapter):
    """ZS32 AnomalyDINO adapter with content-addressed load boundaries."""

    family = AnomalyFamily.ANOMALYDINO
