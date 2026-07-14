"""Atomic JSON transport for dashboard progress and capture confirmation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

from .contracts import ConfirmationCommand, ProgressRecord


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Replace *path* atomically with a complete JSON object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_progress(path: Path, record: ProgressRecord) -> None:
    """Publish the latest runner progress record atomically."""
    _write_json_atomic(path, asdict(record))


def load_progress(path: Path) -> ProgressRecord:
    """Load one runner progress record from JSON."""
    return ProgressRecord(**json.loads(path.read_text(encoding="utf-8")))


def write_confirmation(path: Path, command: ConfirmationCommand) -> None:
    """Publish one operator confirmation command atomically."""
    _write_json_atomic(path, asdict(command))


def consume_confirmation(path: Path, expected: ConfirmationCommand) -> bool:
    """Consume a confirmation once and accept only an exact identity match."""
    if not path.is_file():
        return False
    try:
        actual = ConfirmationCommand(**json.loads(path.read_text(encoding="utf-8")))
    finally:
        path.unlink(missing_ok=True)
    return actual == expected
