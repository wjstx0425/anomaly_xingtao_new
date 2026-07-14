"""Byte-level immutable release verification tests for Linux execution."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from zs32_inspection.runtime.publisher import AtomicDirectoryPublisher
from zs32_inspection.runtime.release_loader import ReleaseLoadError, verify_release_files

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="ZS32 verification is Linux-only")
HASH = "a" * 64


def _release(tmp_path: Path) -> Path:
    with AtomicDirectoryPublisher(tmp_path, "release-001") as publisher:
        publisher.write_json(
            "manifest.json",
            {
                "schema_version": 1,
                "release_id": "release-001",
                "product": "ZS32",
                "contract_sha256": HASH,
                "created_at": "2026-07-14T00:00:00Z",
            },
        )
        publisher.write_json(
            "deployment_contract.json",
            {"schema_version": 1, "product": "ZS32", "contract_sha256": HASH},
        )
        publisher.write_bytes("models/yolo/best.pt", b"weights")
        return publisher.finalize(
            validator=lambda _root: None,
            required_paths=frozenset(
                {"manifest.json", "deployment_contract.json", "models/yolo/best.pt"}
            ),
        )


def test_verified_release_covers_every_file(tmp_path: Path) -> None:
    release = verify_release_files(_release(tmp_path))

    assert release.release_id == "release-001"
    assert release.read_bytes("models/yolo/best.pt") == b"weights"


def test_post_publication_mutation_is_rejected(tmp_path: Path) -> None:
    release = _release(tmp_path)
    weights = release / "models/yolo/best.pt"
    weights.chmod(0o640)
    weights.write_bytes(b"changed")

    with pytest.raises(ReleaseLoadError, match="checksum mismatch"):
        verify_release_files(release)


def test_unindexed_file_is_rejected(tmp_path: Path) -> None:
    release = _release(tmp_path)
    release.chmod(0o750)
    (release / "unexpected.txt").write_text("not indexed", encoding="utf-8")

    with pytest.raises(ReleaseLoadError, match="unindexed"):
        verify_release_files(release)


def test_special_files_are_rejected_not_ignored(tmp_path: Path) -> None:
    release = _release(tmp_path)
    release.chmod(0o750)
    os.mkfifo(release / "unexpected.fifo")

    with pytest.raises(ReleaseLoadError, match="special filesystem nodes"):
        verify_release_files(release)


def test_hardlinked_files_are_rejected(tmp_path: Path) -> None:
    release = _release(tmp_path)
    release.chmod(0o750)
    os.link(release / "models/yolo/best.pt", release / "hardlink.pt")

    with pytest.raises(ReleaseLoadError, match="hardlinked files"):
        verify_release_files(release)


def test_use_boundary_rehash_rejects_late_mutation(tmp_path: Path) -> None:
    release = verify_release_files(_release(tmp_path))
    weights = release.path("models/yolo/best.pt")
    weights.chmod(0o640)
    weights.write_bytes(b"changed after initial verification")

    with pytest.raises(ReleaseLoadError, match="changed after verification"):
        release.read_bytes("models/yolo/best.pt")
