"""Explicit JSON/YAML safe loader tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from zs32_inspection.config.loaders import load_config_object
from zs32_inspection.config.schemas import parse_release_manifest
from zs32_inspection.domain.errors import ConfigLoadError, SchemaValidationError


@pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml"])
def test_config_loader_supports_release_tree_formats(tmp_path: Path, suffix: str) -> None:
    """Blueprint JSON inputs and YAML release files use the same strict parser path."""
    path = tmp_path / f"config{suffix}"
    if suffix == ".json":
        path.write_text('{"schema_version": 1, "product": "ZS32"}', encoding="utf-8")
    else:
        path.write_text("schema_version: 1\nproduct: ZS32\n", encoding="utf-8")

    assert load_config_object(path) == {"schema_version": 1, "product": "ZS32"}


def test_unknown_config_extension_is_rejected(tmp_path: Path) -> None:
    """The loader never guesses a format from file contents."""
    path = tmp_path / "recipe.txt"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ConfigLoadError, match="must use"):
        load_config_object(path)


def test_release_manifest_excludes_checksum_file_digest_cycle() -> None:
    """checksums.sha256 is verified externally and never self-hashed by manifest.json."""
    manifest = parse_release_manifest(
        {
            "schema_version": 1,
            "release_id": "zs32-release-1",
            "product": "ZS32",
            "contract_sha256": "1" * 64,
            "created_at": "2026-07-14T00:00:00Z",
        },
    )

    assert manifest.contract_sha256 == "1" * 64

    with pytest.raises(SchemaValidationError, match="unknown=.*checksums"):
        parse_release_manifest(
            {
                "schema_version": 1,
                "release_id": "zs32-release-1",
                "product": "ZS32",
                "contract_sha256": "1" * 64,
                "checksums_sha256": "2" * 64,
                "created_at": "2026-07-14T00:00:00Z",
            },
        )
