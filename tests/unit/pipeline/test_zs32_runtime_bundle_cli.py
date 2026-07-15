# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CLI contract tests for Stage 37 runtime bundle publication."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "pipeline" / "37_publish_zs32_runtime_bundle.py"


def test_help_lists_both_publication_phases() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], check=False, capture_output=True, text=True)
    assert result.returncode == 0
    assert "publish-assets" in result.stdout
    assert "finalize" in result.stdout


def test_malformed_publish_source_exits_nonzero(tmp_path: Path) -> None:
    source = tmp_path / "bad.json"
    source.write_text("{}", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "publish-assets", "--source", str(source), "--output-dir", str(tmp_path / "out")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not (tmp_path / "out").exists()


def test_finalize_hash_mismatch_exits_nonzero(tmp_path: Path) -> None:
    manifest = tmp_path / "assets_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "product": "ZS32",
                "hand": "right",
                "commissioning_only": True,
                "production_release_allowed": False,
                "asset_set": {},
                "asset_set_sha256": "0" * 64,
            },
        ),
        encoding="utf-8",
    )
    threshold = tmp_path / "thresholds.json"
    threshold.write_text("{}", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "finalize",
            "--assets-manifest",
            str(manifest),
            "--threshold-artifact",
            str(threshold),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "SHA-256" in result.stderr
    assert not (tmp_path / "out").exists()
