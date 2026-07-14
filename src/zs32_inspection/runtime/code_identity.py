# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Verify that a deployment release is executed by its frozen source checkout."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


class RuntimeCodeIdentityError(RuntimeError):
    """The running checkout does not match release code provenance."""


@dataclass(frozen=True, slots=True)
class RuntimeCodeIdentity:
    """Source identity measured from the checkout running the inspection."""

    git_commit: str
    git_tree: str
    dirty: bool
    dependency_lock_sha256: str


def default_repository_root() -> Path:
    """Resolve the repository root from this module's checked-out source path."""
    return Path(__file__).resolve().parents[3]


def _git(repository_root: Path, *arguments: str) -> str:
    executable = shutil.which("git")
    if executable is None:
        raise RuntimeCodeIdentityError("git executable is unavailable")
    # GIT_DIR/GIT_WORK_TREE and related variables can redirect queries away
    # from the checkout containing this module, so none may cross this trust boundary.
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment["LC_ALL"] = "C"
    try:
        completed = subprocess.run(
            [executable, "-C", str(repository_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
            env=environment,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        command = " ".join(arguments)
        raise RuntimeCodeIdentityError(f"git identity query failed ({command}): {error}") from error
    return completed.stdout.strip()


def _dependency_lock_sha256(repository_root: Path) -> str:
    lock_path = repository_root / "uv.lock"
    if lock_path.is_symlink():
        raise RuntimeCodeIdentityError("runtime uv.lock must not be a symlink")
    try:
        descriptor = os.open(
            lock_path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise RuntimeCodeIdentityError(f"cannot open runtime uv.lock: {error}") from error
    digest = hashlib.sha256()
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeCodeIdentityError("runtime uv.lock is not a regular file")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
    except OSError as error:
        raise RuntimeCodeIdentityError(f"cannot read runtime uv.lock: {error}") from error
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def collect_runtime_code_identity(
    repository_root: Path | None = None,
) -> RuntimeCodeIdentity:
    """Measure the current Git checkout and dependency lock without a shell."""
    root = (repository_root or default_repository_root()).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeCodeIdentityError(f"runtime repository root is not a directory: {root}")
    top_level = _git(root, "rev-parse", "--show-toplevel")
    try:
        actual_top_level = Path(top_level).resolve()
    except (OSError, RuntimeError) as error:
        raise RuntimeCodeIdentityError(
            f"git reported an invalid repository root: {top_level!r}"
        ) from error
    if actual_top_level != root:
        raise RuntimeCodeIdentityError(
            f"runtime module root is not the Git top level: module={root}, git={actual_top_level}"
        )
    tracked_lock = _git(root, "ls-files", "--error-unmatch", "--", "uv.lock")
    if tracked_lock != "uv.lock":
        raise RuntimeCodeIdentityError("runtime uv.lock is not tracked at the repository root")
    git_commit = _git(root, "rev-parse", "--verify", "HEAD")
    git_tree = _git(root, "rev-parse", "HEAD^{tree}")
    dependency_lock_sha256 = _dependency_lock_sha256(root)
    # Measure worktree state last so edits that occur during the preceding reads
    # cannot be mistaken for the release's clean checkout.
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if (
        _git(root, "rev-parse", "--verify", "HEAD") != git_commit
        or _git(root, "rev-parse", "HEAD^{tree}") != git_tree
    ):
        raise RuntimeCodeIdentityError("runtime Git identity changed during verification")
    return RuntimeCodeIdentity(
        git_commit=git_commit,
        git_tree=git_tree,
        dirty=bool(status),
        dependency_lock_sha256=dependency_lock_sha256,
    )


def verify_runtime_code_identity(
    expected: Mapping[str, object],
    *,
    repository_root: Path | None = None,
) -> RuntimeCodeIdentity:
    """Fail closed unless the running checkout equals release code provenance."""
    required = {
        "schema_version",
        "git_commit",
        "git_tree",
        "dirty",
        "dependency_lock_sha256",
        "build_id",
    }
    if set(expected) != required or expected.get("schema_version") != 1:
        raise RuntimeCodeIdentityError("release code identity envelope is malformed")
    if expected.get("dirty") is not False:
        raise RuntimeCodeIdentityError("release code identity must require a clean checkout")
    actual = collect_runtime_code_identity(repository_root)
    comparisons = {
        "git_commit": actual.git_commit,
        "git_tree": actual.git_tree,
        "dirty": actual.dirty,
        "dependency_lock_sha256": actual.dependency_lock_sha256,
    }
    mismatches = [
        field
        for field, actual_value in comparisons.items()
        if expected.get(field) != actual_value
    ]
    if mismatches:
        raise RuntimeCodeIdentityError(
            "runtime source identity differs from release provenance: "
            + ", ".join(sorted(mismatches))
        )
    return actual


__all__ = [
    "RuntimeCodeIdentity",
    "RuntimeCodeIdentityError",
    "collect_runtime_code_identity",
    "default_repository_root",
    "verify_runtime_code_identity",
]
