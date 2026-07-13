# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unified ZS32 multi-model entrypoint and right-hand profile."""

# The tests intentionally exercise private parser helpers and a harmless temporary path.
# ruff: noqa: S108, SLF001, TC003

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
VIEWS = ("front", "front_left", "front_right", "back", "back_left", "back_right")


def _load_module(name: str, relative_path: str) -> ModuleType:
    """Load one numbered pipeline module without requiring a package."""
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        msg = f"could not load {path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _minimum_args(tmp_path: Path) -> list[str]:
    """Return the identity and six-view arguments shared by parser tests."""
    values = [
        "infer",
        "--part-id",
        "part-001",
        "--capture-session",
        "session-001",
        "--group-id",
        "group-001",
        "--output-dir",
        str(tmp_path / "output"),
    ]
    for view in VIEWS:
        values.extend((f"--{view.replace('_', '-')}-image", str(tmp_path / f"{view}.png")))
    return values


def test_unified_entrypoint_parses_all_six_explicit_images(tmp_path: Path) -> None:
    """The CLI must preserve an explicit source path for every canonical view."""
    stage32 = _load_module("pipeline_zs32_runtime_parser", "pipeline/32_run_zs32_multimodel_inference.py")

    args = stage32.build_parser().parse_args(_minimum_args(tmp_path))
    request = stage32._request_from_args(args)

    assert args.mode == "infer"
    assert tuple(request.images) == VIEWS
    assert request.hand == "right"


def test_fuse_mode_requires_all_non_model_branch_evidence(tmp_path: Path) -> None:
    """Fuse mode must fail before inference when strict evidence is incomplete."""
    stage32 = _load_module("pipeline_zs32_runtime_fuse", "pipeline/32_run_zs32_multimodel_inference.py")
    values = _minimum_args(tmp_path)
    values[0] = "fuse"
    args = stage32.build_parser().parse_args(values)

    with pytest.raises(ValueError, match="--template-model-dir"):
        stage32._validate_mode(args)


def test_right_profile_contains_exactly_thirty_six_versioned_groups() -> None:
    """The right-only deployment contract must not fabricate left-hand groups."""
    path = REPO_ROOT / "config/fusion/zs32_right_six_view.json"
    profile = json.loads(path.read_text(encoding="utf-8"))
    records = profile["expected_versions"]

    assert len(records) == 36
    assert {record["hand"] for record in records} == {"right"}
    assert {record["view"] for record in records} == set(VIEWS)
    assert all(len(profile["required_branches_by_view"][view]) == 6 for view in VIEWS)


def test_stage18_resolves_right_only_named_profile() -> None:
    """Stage 18 must keep the full and right-only strict profiles separate."""
    stage18 = _load_module("pipeline_zs32_right_profile", "pipeline/18_fuse_inspection_results.py")
    args = stage18.build_parser().parse_args(
        ["--profile", "zs32-right", "--output-dir", "/tmp/zs32-right-profile-test"],
    )

    assert stage18._fusion_config_path(args) == REPO_ROOT / "config/fusion/zs32_right_six_view.json"
