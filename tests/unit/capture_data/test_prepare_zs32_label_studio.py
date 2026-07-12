# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for ZS32 Label Studio hard-link staging."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


def load_module() -> ModuleType:
    """Load the ZS32 Label Studio staging script from its file path."""
    script_path = Path(__file__).resolve().parents[3] / "capture_data" / "prepare_zs32_label_studio.py"
    spec = importlib.util.spec_from_file_location("capture_data_prepare_zs32_label_studio", script_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load ZS32 Label Studio staging script from {script_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _create_six_view_group(
    dataset_root: Path,
    module: ModuleType,
    *,
    hand: str = "left",
    defect_type: str = "deform",
    session_id: str = "20260711_175751_318434",
    group_id: str = "group001",
) -> list[Path]:
    """Create six minimal PNG-signature files for one physical group."""
    paths: list[Path] = []
    for view in module.VIEWS:
        path = (
            dataset_root
            / hand
            / view
            / "defect"
            / defect_type
            / session_id
            / "images"
            / f"{hand}_{view}_defect_{defect_type}_zs32_{hand}_defect_{group_id}_000001_fused.png"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x89PNG\r\n\x1a\n")
        paths.append(path)
    return paths


def test_discover_labeling_images_parses_complete_six_view_group(tmp_path: Path) -> None:
    """Discovery should parse all metadata from one complete six-view group."""
    module = load_module()
    dataset_root = tmp_path / "dataset"
    _create_six_view_group(dataset_root, module)

    images = module.discover_labeling_images(dataset_root)

    assert len(images) == 6
    assert {image.view for image in images} == set(module.VIEWS)
    assert {image.group_id for image in images} == {"group001"}
    assert {image.sample_id for image in images} == {
        "left/20260711_175751_318434/deform/group001",
    }


def test_discover_labeling_images_rejects_incomplete_group(tmp_path: Path) -> None:
    """Discovery should reject a physical group that omits canonical views."""
    module = load_module()
    dataset_root = tmp_path / "dataset"
    path = (
        dataset_root
        / "right"
        / "front"
        / "defect"
        / "less"
        / "session-a"
        / "images"
        / "right_front_defect_less_group001_000001_fused.png"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n")

    with pytest.raises(ValueError, match="must contain exactly six views"):
        module.discover_labeling_images(dataset_root)


def test_prepare_label_studio_dataset_writes_hard_links_and_artifacts(tmp_path: Path) -> None:
    """Preparation should hard-link six images and publish labeling artifacts."""
    module = load_module()
    dataset_root = tmp_path / "dataset"
    sources = _create_six_view_group(dataset_root, module)
    source = sources[0]

    summary = module.prepare_label_studio_dataset(
        dataset_root=dataset_root,
        output_root=tmp_path / "labeling",
        repo_root=tmp_path,
    )

    assert summary == {"total": 6, "left": 6, "right": 0, "groups": 1}
    target = tmp_path / "labeling/images/left/front/deform" / source.name
    assert target.stat().st_dev == source.stat().st_dev
    assert target.stat().st_ino == source.stat().st_ino
    with (tmp_path / "labeling/labeling_manifest.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 6
    assert rows[0]["sample_id"] == "left/20260711_175751_318434/deform/group001"
    assert '<Label value="defect"' in (tmp_path / "labeling/label_studio_config.xml").read_text(encoding="utf-8")
    assert "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true" in (tmp_path / "labeling/README.md").read_text(
        encoding="utf-8",
    )


def test_prepare_label_studio_dataset_requires_explicit_overwrite(tmp_path: Path) -> None:
    """Preparation should reject non-empty output unless overwrite is explicit."""
    module = load_module()
    dataset_root = tmp_path / "dataset"
    _create_six_view_group(dataset_root, module)
    output_root = tmp_path / "labeling"
    module.prepare_label_studio_dataset(dataset_root, output_root, tmp_path)

    with pytest.raises(ValueError, match="non-empty"):
        module.prepare_label_studio_dataset(dataset_root, output_root, tmp_path)

    summary = module.prepare_label_studio_dataset(dataset_root, output_root, tmp_path, overwrite=True)
    assert summary == {"total": 6, "left": 6, "right": 0, "groups": 1}


def test_prepare_label_studio_dataset_rejects_unsafe_output_root(tmp_path: Path) -> None:
    """Preparation should never overwrite a source hand directory."""
    module = load_module()
    dataset_root = tmp_path / "dataset"
    sources = _create_six_view_group(dataset_root, module)

    with pytest.raises(ValueError, match="unsafe output root"):
        module.prepare_label_studio_dataset(
            dataset_root,
            dataset_root / "left",
            tmp_path,
            overwrite=True,
        )

    assert all(source.exists() for source in sources)


def test_prepare_label_studio_dataset_rejects_output_inside_source_hand(tmp_path: Path) -> None:
    """Preparation should never overwrite a descendant of a source hand root."""
    module = load_module()
    dataset_root = tmp_path / "dataset"
    sources = _create_six_view_group(dataset_root, module)

    with pytest.raises(ValueError, match="unsafe output root"):
        module.prepare_label_studio_dataset(
            dataset_root,
            dataset_root / "left" / "front",
            tmp_path,
            overwrite=True,
        )

    assert all(source.exists() for source in sources)


def test_prepare_output_root_rejects_ancestor_of_forbidden_root(tmp_path: Path) -> None:
    """Output cleanup should never delete a directory containing a forbidden root."""
    module = load_module()
    output_root = tmp_path / "output"
    forbidden_root = output_root / "repository"
    marker = forbidden_root / "keep.txt"
    marker.parent.mkdir(parents=True)
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe output root"):
        module._prepare_output_root(  # noqa: SLF001
            output_root,
            overwrite=True,
            forbidden_roots=(forbidden_root,),
        )

    assert marker.read_text(encoding="utf-8") == "keep"


def test_prepare_label_studio_dataset_detects_collision_before_manifest(tmp_path: Path) -> None:
    """Duplicate target names should fail without publishing a manifest."""
    module = load_module()
    dataset_root = tmp_path / "dataset"
    _create_six_view_group(dataset_root, module, session_id="session-a")
    _create_six_view_group(dataset_root, module, session_id="session-b")
    output_root = tmp_path / "labeling"

    with pytest.raises(ValueError, match="Duplicate Label Studio target path"):
        module.prepare_label_studio_dataset(dataset_root, output_root, tmp_path)

    assert not (output_root / "labeling_manifest.csv").exists()
