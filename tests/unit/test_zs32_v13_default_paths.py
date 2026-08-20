# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""GPU-independent contracts for the default ZS32 runtime bundle."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
V13_RUNTIME_BUNDLE = "results/zs32_runtime_bundle_eight_view_template_0727_v13/runtime_bundle.json"


def test_stage35_defaults_to_template_0727_v13_bundle() -> None:
    source = (REPO_ROOT / "pipeline/35_run_zs32_live_commissioning.py").read_text(encoding="utf-8")
    assert V13_RUNTIME_BUNDLE in source
    assert "zs32_runtime_bundle_eight_view_template_0727_v12/runtime_bundle.json" not in source


def test_dashboard_defaults_to_template_0727_v13_bundle() -> None:
    source = (REPO_ROOT / "src/zs32_inspection/dashboard/live.py").read_text(encoding="utf-8")
    assert V13_RUNTIME_BUNDLE in source
    assert "zs32_runtime_bundle_eight_view_template_0727_v12/runtime_bundle.json" not in source
