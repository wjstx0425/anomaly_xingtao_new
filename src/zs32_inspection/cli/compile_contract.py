"""Compile a final threshold-bearing recipe into one deployment contract."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from zs32_inspection.config.compiler import compile_deployment_contract
from zs32_inspection.config.loaders import load_recipe, load_roi_config, load_topology
from zs32_inspection.runtime.publisher import canonical_json_bytes

from ._common import command_error, command_result
from ._environment import require_linux_nvidia


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile one final ZS32 deployment contract")
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--roi", type=Path, required=True)
    parser.add_argument("--fusion-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _write_no_replace(path: Path, payload: bytes) -> None:
    path = path.expanduser()
    if path.suffix.lower() != ".json" or path.is_symlink():
        raise ValueError("contract output must be a non-symlink .json path")
    if path.parent.is_symlink():
        raise ValueError("contract output parent must not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        0o440,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _run(argv: Sequence[str] | None) -> int:
    args = _parser().parse_args(argv)
    recipe = load_recipe(args.recipe)
    topology = load_topology(args.topology)
    roi = load_roi_config(args.roi)
    if args.fusion_policy.is_symlink() or not args.fusion_policy.is_file():
        raise ValueError("fusion policy must be a regular non-symlink file")
    contract = compile_deployment_contract(
        topology,
        roi,
        recipe,
        fusion_policy_bytes=args.fusion_policy.read_bytes(),
    )
    _write_no_replace(args.output, canonical_json_bytes(contract.as_dict()))
    command_result(
        "zs32-compile-contract",
        {
            "contract_sha256": contract.contract_sha256,
            "recipe_sha256": contract.recipe_sha256,
            "output": str(args.output),
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        require_linux_nvidia()
        return _run(argv)
    except Exception as error:
        return command_error("zs32-compile-contract", error)


if __name__ == "__main__":
    raise SystemExit(main())
