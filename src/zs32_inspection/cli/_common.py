"""Small fail-closed helpers shared by the Linux-only ZS32 commands."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from zs32_inspection.config.loaders import load_config_object
from zs32_inspection.runtime.publisher import canonical_json_bytes


def command_error(command: str, error: Exception) -> int:
    """Emit one machine-readable error without a traceback or false success."""
    payload = {
        "command": command,
        "error_type": type(error).__name__,
        "message": str(error),
        "status": "failed",
    }
    for field in ("failure_kind", "diagnostic_path", "capture_set_id"):
        value = getattr(error, field, None)
        if isinstance(value, str) and value:
            payload[field] = value
    sys.stderr.buffer.write(canonical_json_bytes(payload))
    return 2


def command_result(command: str, payload: Mapping[str, Any]) -> None:
    """Emit one deterministic result document to stdout."""
    sys.stdout.buffer.write(
        canonical_json_bytes({"command": command, "status": "ok", **dict(payload)})
    )


def require_keys(
    value: Mapping[str, Any],
    *,
    required: Sequence[str],
    optional: Sequence[str] = (),
    context: str,
) -> None:
    """Reject unknown and missing descriptor fields."""
    actual = set(value)
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - actual)
    unknown = sorted(actual - allowed)
    if missing or unknown:
        raise ValueError(f"{context} keys invalid; missing={missing}, unknown={unknown}")


def object_value(value: object, context: str) -> dict[str, Any]:
    """Require a string-keyed object and return a defensive copy."""
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be an object with string keys")
    return dict(value)


def array_value(value: object, context: str) -> list[Any]:
    """Require a JSON array, excluding string-like sequences."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{context} must be an array")
    return list(value)


def string_value(value: object, context: str, *, allow_empty: bool = False) -> str:
    """Require a string without silently coercing numbers, booleans, or null."""
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        suffix = "a string" if allow_empty else "a non-empty string"
        raise ValueError(f"{context} must be {suffix}")
    return value


def load_object(path: Path, context: str) -> dict[str, Any]:
    """Load an explicit JSON/YAML object with a contextual error label."""
    try:
        return load_config_object(path)
    except Exception as error:
        raise ValueError(f"cannot load {context} {path}: {error}") from error


def load_jsonl(path: Path, context: str) -> list[dict[str, Any]]:
    """Load strict non-empty JSONL rows from a regular, non-symlink file."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} must be a regular non-symlink file: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid {context} JSON at {path}:{line_number}: {error}") from error
        rows.append(object_value(payload, f"{context}[{line_number}]"))
    if not rows:
        raise ValueError(f"{context} contains no rows: {path}")
    return rows


def resolved_asset(root: Path, relative_path: object, context: str) -> Path:
    """Resolve a descriptor asset strictly below an explicit root."""
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"{context}.relative_path must be a non-empty string")
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{context}.relative_path must be a safe relative POSIX path")
    expanded_root = root.expanduser()
    if expanded_root.is_symlink():
        raise ValueError(f"asset root must not be a symlink: {expanded_root}")
    base = expanded_root.resolve()
    unresolved = base.joinpath(*relative.parts)
    current = base
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"{context} traverses a symlink: {current}")
    candidate = unresolved.resolve()
    try:
        candidate.relative_to(base)
    except ValueError as error:
        raise ValueError(f"{context} escapes asset root: {relative_path!r}") from error
    return candidate
