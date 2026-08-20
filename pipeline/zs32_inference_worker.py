#!/usr/bin/env python3
"""Serve serial ZS32 Demo jobs with model objects loaded once."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zs32_inspection.dashboard.inference_worker import InferenceWorkerServer  # noqa: E402


def _load_demo_inference() -> ModuleType:
    """Load the repository-local Demo CLI without requiring a package import."""
    path = REPO_ROOT / "pipeline/zs32_demo_inference.py"
    spec = importlib.util.spec_from_file_location("pipeline_zs32_persistent_demo_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Demo inference: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    """Build the private Demo worker CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--demo-config", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> None:
    """Prepare one runtime and reuse it for every serial job."""
    demo_config = args.demo_config.expanduser().resolve()
    demo_inference = _load_demo_inference()
    runtime = demo_inference.prepare_demo_runtime(demo_config)

    def execute(argv: list[str]) -> int:
        demo_inference.run_argv(
            argv,
            runtime=runtime,
            startup_config_path=demo_config,
        )
        return 0

    try:
        InferenceWorkerServer(args.socket, execute).serve_forever()
    finally:
        runtime.close()


def main() -> None:
    """CLI entrypoint."""
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
