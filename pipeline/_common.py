# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for numbered pipeline entrypoints."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from collections.abc import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(relative_path: str) -> Path:
    """Return an absolute path inside the repository."""
    return REPO_ROOT / relative_path


def run_repo_script(relative_path: str, args: Sequence[str]) -> None:
    """Run an existing repository script in a child Python process."""
    script = repo_path(relative_path)
    if not script.is_file():
        msg = f"Missing pipeline target script: {script}"
        raise FileNotFoundError(msg)
    command = [sys.executable, str(script), *args]
    raise SystemExit(subprocess.run(command, cwd=REPO_ROOT, check=False).returncode)


def is_help_request(args: Sequence[str]) -> bool:
    """Return whether arguments ask for help."""
    return not args or args[0] in {"-h", "--help", "help"}


def with_default_command(args: Sequence[str], valid_commands: set[str], default_command: str) -> list[str]:
    """Insert a default subcommand before option-style arguments."""
    if not args or args[0].startswith("-"):
        return [default_command, *args]
    if args[0] in valid_commands:
        return list(args)
    return [default_command, *args]
