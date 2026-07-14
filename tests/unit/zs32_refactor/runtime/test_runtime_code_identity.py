# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Runtime checkout identity verification for immutable ZS32 releases."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from zs32_inspection.runtime import code_identity
from zs32_inspection.runtime.code_identity import RuntimeCodeIdentityError


COMMIT = "a" * 40
TREE = "b" * 40


def test_default_repository_root_is_derived_from_runtime_module() -> None:
    assert code_identity.default_repository_root() == Path(code_identity.__file__).resolve().parents[3]


def _expected(lock_digest: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "git_commit": COMMIT,
        "git_tree": TREE,
        "dirty": False,
        "dependency_lock_sha256": lock_digest,
        "build_id": "linux-build-v1",
    }


def _mock_git(
    monkeypatch: pytest.MonkeyPatch,
    repository_root: Path,
    *,
    status: str = "",
) -> list[list[str]]:
    calls: list[list[str]] = []
    outputs = {
        ("rev-parse", "--show-toplevel"): str(repository_root),
        ("ls-files", "--error-unmatch", "--", "uv.lock"): "uv.lock",
        ("status", "--porcelain=v1", "--untracked-files=all"): status,
        ("rev-parse", "--verify", "HEAD"): COMMIT,
        ("rev-parse", "HEAD^{tree}"): TREE,
    }

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert kwargs["shell"] is False
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert not any(key.startswith("GIT_") for key in environment)
        arguments = tuple(command[3:])
        return subprocess.CompletedProcess(command, 0, stdout=outputs[arguments] + "\n", stderr="")

    monkeypatch.setattr(code_identity.shutil, "which", lambda executable: f"/usr/bin/{executable}")
    monkeypatch.setattr(code_identity.subprocess, "run", run)
    return calls


def test_runtime_identity_matches_release_commit_tree_clean_state_and_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_content = b"frozen dependency graph\n"
    (tmp_path / "uv.lock").write_bytes(lock_content)
    monkeypatch.setenv("GIT_DIR", "/attacker/controlled/git-dir")
    calls = _mock_git(monkeypatch, tmp_path)

    actual = code_identity.verify_runtime_code_identity(
        _expected(hashlib.sha256(lock_content).hexdigest()),
        repository_root=tmp_path,
    )

    assert actual.git_commit == COMMIT
    assert actual.git_tree == TREE
    assert actual.dirty is False
    assert all(command[1:3] == ["-C", str(tmp_path)] for command in calls)


def test_runtime_identity_rejects_dirty_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_content = b"frozen dependency graph\n"
    (tmp_path / "uv.lock").write_bytes(lock_content)
    _mock_git(monkeypatch, tmp_path, status=" M src/zs32_inspection/runtime/code_identity.py")

    with pytest.raises(RuntimeCodeIdentityError, match="dirty"):
        code_identity.verify_runtime_code_identity(
            _expected(hashlib.sha256(lock_content).hexdigest()),
            repository_root=tmp_path,
        )


@pytest.mark.parametrize("field", ["git_commit", "git_tree", "dependency_lock_sha256"])
def test_runtime_identity_rejects_release_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    lock_content = b"frozen dependency graph\n"
    (tmp_path / "uv.lock").write_bytes(lock_content)
    _mock_git(monkeypatch, tmp_path)
    expected = _expected(hashlib.sha256(lock_content).hexdigest())
    expected[field] = "c" * len(str(expected[field]))

    with pytest.raises(RuntimeCodeIdentityError, match=field):
        code_identity.verify_runtime_code_identity(expected, repository_root=tmp_path)


def test_runtime_identity_rejects_untracked_dependency_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_content = b"frozen dependency graph\n"
    (tmp_path / "uv.lock").write_bytes(lock_content)
    monkeypatch.setattr(code_identity.shutil, "which", lambda executable: f"/usr/bin/{executable}")
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[3:] == ["ls-files", "--error-unmatch", "--", "uv.lock"]:
            raise subprocess.CalledProcessError(1, command)
        assert kwargs["shell"] is False
        original = {
            ("rev-parse", "--show-toplevel"): str(tmp_path),
        }
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=original[tuple(command[3:])] + "\n",
            stderr="",
        )

    monkeypatch.setattr(code_identity.subprocess, "run", run)

    with pytest.raises(RuntimeCodeIdentityError, match="git identity query failed"):
        code_identity.verify_runtime_code_identity(
            _expected(hashlib.sha256(lock_content).hexdigest()),
            repository_root=tmp_path,
        )
    assert any(command[3:5] == ["ls-files", "--error-unmatch"] for command in calls)
