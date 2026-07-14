"""Fail-closed atomic publication of inspection and release directories."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any


class PublicationError(RuntimeError):
    """A publication could not be completed without weakening its contract."""


class PublicationDurabilityError(PublicationError):
    """Rename succeeded but parent-directory durability could not be confirmed."""

    def __init__(self, published_path: Path, cause: OSError) -> None:
        super().__init__(
            f"publication is visible at {published_path}, but parent fsync failed; "
            "do not retry with the same id until audited: "
            f"{cause}"
        )
        self.published_path = published_path


def canonical_json_bytes(payload: object) -> bytes:
    """Serialize one payload deterministically and reject NaN/Infinity."""
    try:
        return (
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PublicationError(f"payload is not canonical JSON: {error}") from error


def sha256_file(path: Path) -> str:
    """Return a streaming SHA256 for a regular file."""
    if path.is_symlink() or not path.is_file():
        raise PublicationError(f"expected a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: str | Path) -> Path:
    text = str(value)
    pure = PurePosixPath(text)
    if (
        not text
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or any("\x00" in part or "\n" in part or "\r" in part for part in pure.parts)
        or "\\" in text
    ):
        raise PublicationError(f"unsafe publication-relative path: {value!r}")
    return Path(*pure.parts)


def _iter_regular_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise PublicationError(f"symlinks are forbidden in publication trees: {path}")
        if path.is_file():
            metadata = path.stat(follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise PublicationError(f"publication files must be private regular files: {path}")
            files.append(path)
        elif not path.is_dir():
            raise PublicationError(f"special filesystem nodes are forbidden in publication trees: {path}")
    return files


def tree_checksums(root: Path, *, excluded: frozenset[str] = frozenset()) -> dict[str, str]:
    """Hash every regular file below a staging root by portable relative path."""
    checksums: dict[str, str] = {}
    for path in _iter_regular_files(root):
        relative = path.relative_to(root).as_posix()
        if relative not in excluded:
            checksums[relative] = sha256_file(path)
    return checksums


def _checksum_bytes(checksums: Mapping[str, str]) -> bytes:
    lines = [f"{digest}  {relative}" for relative, digest in sorted(checksums.items())]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _read_private_regular_file(path: Path, *, owner_uid: int) -> bytes:
    """Read one immutable publication file while checking its filesystem identity."""
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
    except OSError as error:
        raise PublicationError(f"cannot safely open publication file {path}: {error}") from error
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != owner_uid
    ):
        os.close(descriptor)
        raise PublicationError(
            f"publication file must be a private regular file owned by this user: {path}"
        )
    try:
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        content = b"".join(chunks)
        after = os.fstat(descriptor)
    except OSError as error:
        raise PublicationError(f"cannot safely read publication file {path}: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_nlink,
        before.st_uid,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_nlink,
        after.st_uid,
    )
    if identity_before != identity_after:
        raise PublicationError(f"publication file changed while being read: {path}")
    return content


def _parse_checksum_bytes(content: bytes) -> Mapping[str, str]:
    """Parse exactly the canonical checksum format emitted by this publisher."""
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PublicationError("checksums.sha256 must be UTF-8") from error
    if not text or not text.endswith("\n"):
        raise PublicationError("checksums.sha256 must be non-empty and end with newline")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        try:
            digest, relative_text = line.split("  ", maxsplit=1)
        except ValueError as error:
            raise PublicationError(
                f"malformed checksums.sha256 line {line_number}"
            ) from error
        relative = _safe_relative_path(relative_text).as_posix()
        if relative in {"checksums.sha256", "publication_root.json"}:
            raise PublicationError("checksum index must not include publication metadata")
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise PublicationError(
                f"invalid SHA256 in checksums.sha256 line {line_number}"
            )
        if relative in entries:
            raise PublicationError(f"duplicate checksum path: {relative}")
        entries[relative] = digest
    if not entries or content != _checksum_bytes(entries):
        raise PublicationError("checksums.sha256 is not in canonical sorted form")
    return MappingProxyType(entries)


@dataclass(frozen=True, slots=True)
class VerifiedAtomicPublication:
    """A checksum-verified publication whose files are rehashed at every read."""

    root: Path
    publication_id: str
    root_sha256: str
    checksums: Mapping[str, str]
    owner_uid: int

    def read_bytes(self, relative_path: str) -> bytes:
        """Re-open one indexed file and reject mutations after initial verification."""
        relative = _safe_relative_path(relative_path).as_posix()
        expected = self.checksums.get(relative)
        if expected is None:
            raise PublicationError(f"file is not covered by publication checksums: {relative}")
        content = _read_private_regular_file(
            self.root.joinpath(*PurePosixPath(relative).parts),
            owner_uid=self.owner_uid,
        )
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise PublicationError(
                f"publication file changed after verification: {relative}: {actual} != {expected}"
            )
        return content


def verify_atomic_publication(
    root: Path,
    *,
    required_paths: frozenset[str] = frozenset(),
    allowed_paths: frozenset[str] | None = None,
    expected_publication_id: str | None = None,
) -> VerifiedAtomicPublication:
    """Verify the exact tree and publication-root envelope without trusting paths."""
    unresolved = Path(root).expanduser()
    if unresolved.is_symlink() or not unresolved.is_dir():
        raise PublicationError(f"publication root must be a non-symlink directory: {unresolved}")
    resolved = unresolved.resolve()
    owner_uid = os.geteuid()
    root_metadata = resolved.stat(follow_symlinks=False)
    if not stat.S_ISDIR(root_metadata.st_mode) or root_metadata.st_uid != owner_uid:
        raise PublicationError("publication root must be a directory owned by this user")

    checksum_content = _read_private_regular_file(
        resolved / "checksums.sha256",
        owner_uid=owner_uid,
    )
    checksums = _parse_checksum_bytes(checksum_content)
    indexed_paths = set(checksums)
    normalized_required = {
        _safe_relative_path(path).as_posix() for path in required_paths
    }
    missing = normalized_required - indexed_paths
    if missing:
        raise PublicationError(f"publication is missing required indexed files: {sorted(missing)}")
    if allowed_paths is not None:
        normalized_allowed = {
            _safe_relative_path(path).as_posix() for path in allowed_paths
        }
        unexpected = indexed_paths - normalized_allowed
        if unexpected:
            raise PublicationError(f"publication contains unexpected indexed files: {sorted(unexpected)}")
        if normalized_allowed != indexed_paths:
            raise PublicationError(
                f"publication file set differs from the allowed contract: "
                f"missing={sorted(normalized_allowed - indexed_paths)}"
            )

    expected_directories = {
        PurePosixPath(relative).parent.as_posix()
        for relative in indexed_paths
        if PurePosixPath(relative).parent.as_posix() != "."
    }
    expected_directories |= {
        parent.as_posix()
        for relative in indexed_paths
        for parent in PurePosixPath(relative).parents
        if parent.as_posix() != "."
    }
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for path in resolved.rglob("*"):
        metadata = path.stat(follow_symlinks=False)
        relative = path.relative_to(resolved).as_posix()
        if stat.S_ISLNK(metadata.st_mode):
            raise PublicationError(f"symlinks are forbidden in publications: {path}")
        if metadata.st_uid != owner_uid:
            raise PublicationError(f"publication nodes must be owned by this user: {path}")
        if stat.S_ISREG(metadata.st_mode):
            if metadata.st_nlink != 1:
                raise PublicationError(f"hardlinked files are forbidden in publications: {path}")
            actual_files.add(relative)
        elif stat.S_ISDIR(metadata.st_mode):
            actual_directories.add(relative)
        else:
            raise PublicationError(f"special filesystem nodes are forbidden in publications: {path}")
    expected_files = indexed_paths | {"checksums.sha256", "publication_root.json"}
    if actual_files != expected_files:
        raise PublicationError(
            "publication file set differs from checksum index; "
            f"missing={sorted(expected_files - actual_files)}, "
            f"unindexed={sorted(actual_files - expected_files)}"
        )
    if actual_directories != expected_directories:
        raise PublicationError(
            "publication contains missing or unexpected directories; "
            f"missing={sorted(expected_directories - actual_directories)}, "
            f"unexpected={sorted(actual_directories - expected_directories)}"
        )

    for relative, expected_digest in checksums.items():
        content = _read_private_regular_file(
            resolved.joinpath(*PurePosixPath(relative).parts),
            owner_uid=owner_uid,
        )
        if hashlib.sha256(content).hexdigest() != expected_digest:
            raise PublicationError(f"publication checksum mismatch for {relative}")

    publication_bytes = _read_private_regular_file(
        resolved / "publication_root.json",
        owner_uid=owner_uid,
    )
    try:
        publication = json.loads(publication_bytes)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PublicationError(f"publication_root.json is invalid JSON: {error}") from error
    if (
        not isinstance(publication, dict)
        or set(publication) != {"algorithm", "publication_id", "root_sha256"}
        or canonical_json_bytes(publication) != publication_bytes
    ):
        raise PublicationError("publication_root.json is not the canonical strict schema")
    publication_id = resolved.name if expected_publication_id is None else expected_publication_id
    root_sha256 = hashlib.sha256(checksum_content).hexdigest()
    if (
        publication["algorithm"] != "sha256(checksums.sha256 bytes)"
        or publication["publication_id"] != publication_id
        or publication["root_sha256"] != root_sha256
        or resolved.name != publication_id
    ):
        raise PublicationError("publication root identity or digest mismatch")
    return VerifiedAtomicPublication(
        root=resolved,
        publication_id=publication_id,
        root_sha256=root_sha256,
        checksums=checksums,
        owner_uid=owner_uid,
    )


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without POSIX rename overwrite semantics.

    The production runtime is Linux-only.  A pre-rename ``exists`` check is not
    sufficient because another process can create the destination between that
    check and ``rename``.  Linux ``renameat2(RENAME_NOREPLACE)`` closes that
    race; absence of the syscall is a system error, never permission to weaken
    publication semantics.
    """
    if sys.platform != "linux":
        raise PublicationError("atomic no-replace publication requires the Linux runtime")

    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise PublicationError("Linux libc does not expose renameat2; refusing unsafe publication")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise PublicationError(
            f"atomic publication rename failed: {os.strerror(error_number)}: {destination}"
        )


def _seal_tree(root: Path) -> None:
    """Remove write bits before the last hash and publication rename."""
    for path in _iter_regular_files(root):
        path.chmod(0o440)
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir() and not path.is_symlink()),
        key=lambda item: len(item.parts),
        reverse=True,
    )
    for directory in directories:
        directory.chmod(0o550)
    root.chmod(0o550)


def _make_tree_removable(root: Path) -> None:
    """Restore owner directory write permission for abort cleanup."""
    if not root.exists():
        return
    root.chmod(0o700)
    for path in root.rglob("*"):
        if path.is_dir() and not path.is_symlink():
            path.chmod(0o700)


class AtomicDirectoryPublisher:
    """Build one directory in staging and publish it exactly once.

    The caller may write only through this object. ``finalize`` writes the
    checksum index, runs a read-only validator, rechecks the complete file set,
    fsyncs staged content, and performs the final rename while holding a
    publication lock.
    """

    def __init__(self, output_root: Path, publication_id: str) -> None:
        if not publication_id or publication_id in {".", ".."} or any(
            token in publication_id for token in ("/", "\\", "\x00", "\n", "\r")
        ):
            raise PublicationError(f"unsafe publication_id: {publication_id!r}")
        expanded_root = output_root.expanduser()
        if expanded_root.is_symlink():
            raise PublicationError(f"publication output_root must not be a symlink: {expanded_root}")
        self.output_root = expanded_root.resolve()
        self.publication_id = publication_id
        self.destination = self.output_root / publication_id
        self.lock_path = self.output_root / f".{publication_id}.lock"
        self.staging: Path | None = None
        self._lock_descriptor: int | None = None
        self._published = False

    def __enter__(self) -> AtomicDirectoryPublisher:
        import fcntl

        self.output_root.mkdir(parents=True, exist_ok=True)
        try:
            self._lock_descriptor = os.open(
                self.lock_path,
                os.O_CREAT
                | os.O_RDWR
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
            lock_metadata = os.fstat(self._lock_descriptor)
            if (
                not stat.S_ISREG(lock_metadata.st_mode)
                or lock_metadata.st_nlink != 1
                or lock_metadata.st_uid != os.geteuid()
            ):
                raise PublicationError(
                    f"publication lock must be a private regular file owned by this user: {self.lock_path}"
                )
            try:
                fcntl.flock(self._lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise PublicationError(f"publication lock is held: {self.lock_path}") from error
            os.ftruncate(self._lock_descriptor, 0)
            os.write(self._lock_descriptor, f"pid={os.getpid()}\n".encode("ascii"))
            os.fsync(self._lock_descriptor)
            if self.destination.exists():
                raise PublicationError(f"publication already exists: {self.destination}")
            self.staging = Path(
                tempfile.mkdtemp(prefix=f".{self.publication_id}.staging-", dir=self.output_root)
            )
        except BaseException:
            self.abort()
            raise
        return self

    def _require_staging(self) -> Path:
        if self.staging is None or self._published:
            raise PublicationError("publisher is not in an active staging state")
        return self.staging

    def _destination_for(self, relative_path: str | Path) -> Path:
        staging = self._require_staging()
        relative = _safe_relative_path(relative_path)
        destination = staging / relative
        if destination.exists() or destination.is_symlink():
            raise PublicationError(f"publication path already written: {relative.as_posix()}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    def write_bytes(self, relative_path: str | Path, content: bytes) -> Path:
        """Write one immutable staged file with exclusive creation."""
        destination = self._destination_for(relative_path)
        descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o640)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(descriptor)
        return destination

    def write_json(self, relative_path: str | Path, payload: object) -> Path:
        """Write deterministic JSON into staging."""
        return self.write_bytes(relative_path, canonical_json_bytes(payload))

    def copy_file(
        self,
        source: Path,
        relative_path: str | Path,
        *,
        expected_sha256: str,
    ) -> Path:
        """Copy one approved file and reject source mutation during the copy."""
        source = source.expanduser().resolve()
        before = sha256_file(source)
        if before != expected_sha256:
            raise PublicationError(
                f"source hash mismatch before copy: {source}: {before} != {expected_sha256}"
            )
        destination = self._destination_for(relative_path)
        shutil.copyfile(source, destination, follow_symlinks=False)
        after = sha256_file(destination)
        if after != expected_sha256:
            destination.unlink(missing_ok=True)
            raise PublicationError(
                f"copied hash mismatch: {source}: {after} != {expected_sha256}"
            )
        _fsync_file(destination)
        return destination

    def finalize(
        self,
        *,
        validator: Callable[[Path], None],
        required_paths: frozenset[str] = frozenset(),
    ) -> Path:
        """Validate and atomically publish the complete staged directory."""
        staging = self._require_staging()
        existing = tree_checksums(
            staging,
            excluded=frozenset({"checksums.sha256", "publication_root.json"}),
        )
        missing = required_paths - set(existing)
        if missing:
            raise PublicationError(f"publication is missing required files: {sorted(missing)}")
        checksum_content = _checksum_bytes(existing)
        self.write_bytes("checksums.sha256", checksum_content)
        root_digest = hashlib.sha256(checksum_content).hexdigest()
        self.write_json(
            "publication_root.json",
            {
                "algorithm": "sha256(checksums.sha256 bytes)",
                "publication_id": self.publication_id,
                "root_sha256": root_digest,
            },
        )

        validator(staging)
        after_validation = tree_checksums(
            staging,
            excluded=frozenset({"checksums.sha256", "publication_root.json"}),
        )
        if after_validation != existing:
            raise PublicationError("validator or concurrent process mutated staged content")
        if (staging / "checksums.sha256").read_bytes() != checksum_content:
            raise PublicationError("checksum index changed during validation")

        _seal_tree(staging)
        final_content = tree_checksums(
            staging,
            excluded=frozenset({"checksums.sha256", "publication_root.json"}),
        )
        if final_content != existing or (staging / "checksums.sha256").read_bytes() != checksum_content:
            raise PublicationError("sealed publication content changed before rename")
        for path in _iter_regular_files(staging):
            _fsync_file(path)
        for directory in sorted(
            (path for path in staging.rglob("*") if path.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
        _fsync_directory(staging)
        _rename_directory_noreplace(staging, self.destination)
        self.staging = None
        self._published = True
        self._release_lock()
        try:
            _fsync_directory(self.output_root)
        except OSError as error:
            raise PublicationDurabilityError(self.destination, error) from error
        return self.destination

    def _release_lock(self) -> None:
        if self._lock_descriptor is not None:
            import fcntl

            fcntl.flock(self._lock_descriptor, fcntl.LOCK_UN)
            os.close(self._lock_descriptor)
            self._lock_descriptor = None

    def abort(self) -> None:
        """Remove non-published staging state and release the lock."""
        if self.staging is not None:
            _make_tree_removable(self.staging)
            shutil.rmtree(self.staging, ignore_errors=True)
            self.staging = None
        self._release_lock()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if not self._published:
            self.abort()
