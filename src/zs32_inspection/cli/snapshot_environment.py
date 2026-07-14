"""Snapshot the authoritative Linux/NVIDIA inference environment."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from zs32_inspection.runtime.environment_receipt import collect_runtime_environment

from ._common import command_error, command_result
from ._environment import require_linux_nvidia


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write one canonical content-addressed ZS32 runtime environment receipt"
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _run(argv: Sequence[str] | None) -> int:
    args = _parser().parse_args(argv)
    receipt = collect_runtime_environment(args.device)
    output = args.output.expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"runtime environment receipt already exists: {output}")
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise ValueError(
            f"runtime environment receipt parent must be a regular directory: {output.parent}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp-",
        dir=output.parent,
    )
    temporary = Path(temporary_name)
    try:
        try:
            content = receipt.canonical_bytes()
            view = memoryview(content)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short write while persisting runtime environment receipt")
                view = view[written:]
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o444)
        finally:
            os.close(descriptor)
        try:
            os.link(temporary, output)
            directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            directory_descriptor = os.open(output.parent, directory_flags)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except FileExistsError as error:
            raise FileExistsError(
                f"runtime environment receipt already exists: {output}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)
    command_result(
        "zs32-snapshot-runtime-environment",
        {
            "device": args.device,
            "gpu_uuid": receipt.device_mapping.gpu_uuid,
            "output": str(output),
            "file_sha256": hashlib.sha256(receipt.canonical_bytes()).hexdigest(),
            "receipt_sha256": receipt.receipt_sha256,
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        require_linux_nvidia()
        return _run(argv)
    except Exception as error:
        return command_error("zs32-snapshot-runtime-environment", error)


if __name__ == "__main__":
    raise SystemExit(main())
