# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EfficientAD adapter contract; real Anomalib work is injected on Linux."""

from .base import AnomalyFamily, DelegatingAnomalyAdapter


class EfficientADAdapter(DelegatingAnomalyAdapter):
    """ZS32 EfficientAD adapter with content-addressed load boundaries."""

    family = AnomalyFamily.EFFICIENTAD
