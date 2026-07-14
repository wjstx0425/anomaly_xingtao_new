# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Static filesystem contracts for deterministic candidate registration assembly."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pytest

from zs32_inspection.cli.build_candidate_registration import (
    _global_asset,
    _write_private_no_replace,
)
from zs32_inspection.runtime.publisher import canonical_json_bytes


def test_global_asset_is_rebased_to_the_declared_asset_root(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    publication = asset_root / "anomaly" / "candidate-1"
    checkpoint = publication / "checkpoint" / "model.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    result = _global_asset(
        {
            "role": "checkpoint",
            "relative_path": "checkpoint/model.ckpt",
            "sha256": digest,
            "media_type": "application/x-pytorch-lightning-checkpoint",
        },
        publication_root=publication,
        asset_root=asset_root.resolve(),
        field="test checkpoint",
    )

    assert result["relative_path"] == "anomaly/candidate-1/checkpoint/model.ckpt"


def test_global_asset_rejects_symlinked_content(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    publication = asset_root / "candidate"
    publication.mkdir(parents=True)
    target = tmp_path / "outside.ckpt"
    target.write_bytes(b"outside")
    (publication / "model.ckpt").symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        _global_asset(
            {
                "role": "checkpoint",
                "relative_path": "model.ckpt",
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "media_type": "application/octet-stream",
            },
            publication_root=publication,
            asset_root=asset_root.resolve(),
            field="test checkpoint",
        )


def test_registration_output_is_canonical_private_and_no_replace(tmp_path: Path) -> None:
    output = tmp_path / "descriptors" / "candidate_registration.local.json"
    payload = {"z": [2, 1], "a": "value"}

    _write_private_no_replace(output, payload)

    assert output.read_bytes() == canonical_json_bytes(payload)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        _write_private_no_replace(output, payload)
