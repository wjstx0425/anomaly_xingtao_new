"""Linux-only contract tests for immutable ZS32 directory publication.

These tests are authored on macOS but must only be executed by the Linux
verification workflow described in the refactor blueprint.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest

from zs32_inspection.runtime.publisher import (
    AtomicDirectoryPublisher,
    PublicationError,
    verify_atomic_publication,
)

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="ZS32 verification is Linux-only")


def _accept(_staging: Path) -> None:
    """Accept a publication after the publisher's own completeness checks."""


def test_finalize_publishes_checksummed_tree_once(tmp_path: Path) -> None:
    """A complete tree is visible only after its checksum root exists."""
    with AtomicDirectoryPublisher(tmp_path, "inspection-001") as publisher:
        publisher.write_json("decision.json", {"inspection_status": "OK"})
        published = publisher.finalize(
            validator=_accept,
            required_paths=frozenset({"decision.json"}),
        )

    assert published == tmp_path / "inspection-001"
    assert (published / "checksums.sha256").is_file()
    assert (published / "publication_root.json").is_file()


def test_existing_destination_is_never_overwritten(tmp_path: Path) -> None:
    """A repeated publication id is a hard error and preserves old bytes."""
    destination = tmp_path / "release-001"
    destination.mkdir()
    sentinel = destination / "sentinel"
    sentinel.write_text("original", encoding="utf-8")

    with pytest.raises(PublicationError, match="already exists"):
        with AtomicDirectoryPublisher(tmp_path, "release-001"):
            pass

    assert sentinel.read_text(encoding="utf-8") == "original"


def test_copy_rejects_changed_source(tmp_path: Path) -> None:
    """Expected asset hashes are checked before copying into staging."""
    source = tmp_path / "model.ckpt"
    source.write_bytes(b"actual")
    wrong_digest = hashlib.sha256(b"different").hexdigest()

    with pytest.raises(PublicationError, match="source hash mismatch"):
        with AtomicDirectoryPublisher(tmp_path / "out", "release-002") as publisher:
            publisher.copy_file(source, "models/model.ckpt", expected_sha256=wrong_digest)


def test_validator_mutation_aborts_publication(tmp_path: Path) -> None:
    """Validation cannot modify the content it is certifying."""

    def mutate(staging: Path) -> None:
        (staging / "decision.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(PublicationError, match="mutated staged content"):
        with AtomicDirectoryPublisher(tmp_path, "inspection-002") as publisher:
            publisher.write_json("decision.json", {"inspection_status": "SYSTEM_ERROR"})
            publisher.finalize(validator=mutate)

    assert not (tmp_path / "inspection-002").exists()


def test_unsafe_relative_path_is_rejected(tmp_path: Path) -> None:
    """Publication writes cannot escape staging or introduce portable ambiguity."""
    with AtomicDirectoryPublisher(tmp_path, "inspection-003") as publisher:
        with pytest.raises(PublicationError, match="unsafe publication-relative path"):
            publisher.write_bytes("../escape", b"no")


def test_validator_cannot_add_fifo(tmp_path: Path) -> None:
    """A special node can never disappear from the publication checksum set."""

    def add_fifo(staging: Path) -> None:
        os.mkfifo(staging / "unexpected.fifo")

    with pytest.raises(PublicationError, match="special filesystem nodes"):
        with AtomicDirectoryPublisher(tmp_path, "inspection-fifo") as publisher:
            publisher.write_bytes("decision.json", b"{}\n")
            publisher.finalize(validator=add_fifo)


def test_validator_cannot_add_hardlink(tmp_path: Path) -> None:
    """A staged file must have exactly one link before publication."""

    def add_hardlink(staging: Path) -> None:
        os.link(staging / "decision.json", staging / "alias.json")

    with pytest.raises(PublicationError, match="private regular files"):
        with AtomicDirectoryPublisher(tmp_path, "inspection-hardlink") as publisher:
            publisher.write_bytes("decision.json", b"{}\n")
            publisher.finalize(validator=add_hardlink)


def test_preexisting_lock_symlink_cannot_truncate_its_target(tmp_path: Path) -> None:
    """A hostile lock path must not be followed before flock/ftruncate."""
    target = tmp_path / "do-not-touch.txt"
    target.write_bytes(b"trusted-content")
    (tmp_path / ".inspection-lock.lock").symlink_to(target)

    with pytest.raises((OSError, PublicationError)):
        with AtomicDirectoryPublisher(tmp_path, "inspection-lock"):
            pass

    assert target.read_bytes() == b"trusted-content"


def test_verifier_reopens_only_the_exact_indexed_file_set(tmp_path: Path) -> None:
    """Consumers can require an exact atomic publication schema."""
    with AtomicDirectoryPublisher(tmp_path, "score-run-001") as publisher:
        publisher.write_json("score_run.json", {"schema_version": 1})
        publisher.write_bytes("scores.jsonl", b'{"score":0.1}\n')
        published = publisher.finalize(
            validator=_accept,
            required_paths=frozenset({"score_run.json", "scores.jsonl"}),
        )

    verified = verify_atomic_publication(
        published,
        required_paths=frozenset({"score_run.json", "scores.jsonl"}),
        allowed_paths=frozenset({"score_run.json", "scores.jsonl"}),
    )
    assert verified.publication_id == "score-run-001"
    assert verified.read_bytes("scores.jsonl") == b'{"score":0.1}\n'


def test_verifier_rejects_an_unindexed_file(tmp_path: Path) -> None:
    """A file added after publication cannot cross the consumer boundary."""
    with AtomicDirectoryPublisher(tmp_path, "score-run-extra") as publisher:
        publisher.write_bytes("scores.jsonl", b'{"score":0.1}\n')
        published = publisher.finalize(validator=_accept)

    published.chmod(0o750)
    (published / "unindexed.json").write_bytes(b"{}\n")
    with pytest.raises(PublicationError, match="unindexed"):
        verify_atomic_publication(published)


def test_verifier_rejects_hardlinked_indexed_content(tmp_path: Path) -> None:
    """Matching bytes do not make a shared inode an immutable publication file."""
    with AtomicDirectoryPublisher(tmp_path, "score-run-hardlink") as publisher:
        publisher.write_bytes("scores.jsonl", b'{"score":0.1}\n')
        published = publisher.finalize(validator=_accept)

    source = tmp_path / "shared-scores.jsonl"
    source.write_bytes(b'{"score":0.1}\n')
    published.chmod(0o750)
    (published / "scores.jsonl").unlink()
    os.link(source, published / "scores.jsonl")
    with pytest.raises(PublicationError, match="hardlinked|private regular"):
        verify_atomic_publication(published)


def test_verifier_rejects_symlinked_indexed_content(tmp_path: Path) -> None:
    """A symlink cannot stand in for bytes that were indexed at publication time."""
    with AtomicDirectoryPublisher(tmp_path, "score-run-symlink") as publisher:
        publisher.write_bytes("scores.jsonl", b'{"score":0.1}\n')
        published = publisher.finalize(validator=_accept)

    source = tmp_path / "redirected-scores.jsonl"
    source.write_bytes(b'{"score":0.1}\n')
    published.chmod(0o750)
    (published / "scores.jsonl").unlink()
    (published / "scores.jsonl").symlink_to(source)
    with pytest.raises(PublicationError, match="symlinks|private regular"):
        verify_atomic_publication(published)
