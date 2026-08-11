#!/usr/bin/env python3
"""Create a stable receipt for reproducing a BMW lab deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest for a regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_directory(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_dir():
        raise ValueError(f"{label} directory does not exist: {path}")
    return path


def _require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    return path


def _file_row(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _regular_files(root: Path) -> list[dict[str, Any]]:
    rows = [
        _file_row(path, relative_to=root)
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    ]
    return sorted(rows, key=lambda row: row["path"])


def _release_identity(root: Path) -> dict[str, Any]:
    symlinks = [
        {
            "path": path.relative_to(root).as_posix(),
            "target": str(path.resolve()),
        }
        for path in root.rglob("*")
        if path.is_symlink()
    ]
    return {
        "root": str(root),
        "files": _regular_files(root),
        "symlinks": sorted(symlinks, key=lambda row: row["path"]),
    }


def build_snapshot(
    run_dir: Path,
    training_release: Path,
    demo_config: Path,
    roi_config: Path,
    *,
    code_commit: str,
) -> dict[str, Any]:
    """Build a deterministic receipt without dereferencing dataset symlinks."""
    run_dir = _require_directory(Path(run_dir), "run")
    training_release = _require_directory(Path(training_release), "training release")
    demo_config = _require_file(Path(demo_config), "demo config")
    roi_config = _require_file(Path(roi_config), "ROI config")
    return {
        "schema_version": 1,
        "code_commit": code_commit,
        "run_root": str(run_dir),
        "run_files": _regular_files(run_dir),
        "training_release": _release_identity(training_release),
        "configs": {
            "demo": {
                "path": str(demo_config),
                "size_bytes": demo_config.stat().st_size,
                "sha256": _sha256(demo_config),
            },
            "roi": {
                "path": str(roi_config),
                "size_bytes": roi_config.stat().st_size,
                "sha256": _sha256(roi_config),
            },
        },
    }


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--training-release", type=Path, required=True)
    parser.add_argument("--demo-config", type=Path, required=True)
    parser.add_argument("--roi-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Write a reproducibility receipt, refusing to overwrite an existing one."""
    args = _parser().parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f"output already exists: {args.output}")
        snapshot = build_snapshot(
            args.run_dir,
            args.training_release,
            args.demo_config,
            args.roi_config,
            code_commit=args.code_commit or _git_commit(),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (FileExistsError, OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(json.dumps({"status": "failed", "message": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"status": "complete", "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
