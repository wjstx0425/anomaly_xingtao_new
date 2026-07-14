"""Publish bootstrap capture-gate assets as one immutable atomic input."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from pathlib import Path

from zs32_inspection.capture import load_capture_gate_policy_bytes
from zs32_inspection.capture.gate_publication import verify_capture_gate_asset_semantics
from zs32_inspection.config.loaders import load_topology
from zs32_inspection.runtime.publisher import AtomicDirectoryPublisher, canonical_json_bytes

from ._common import command_error, command_result
from ._environment import require_linux_nvidia


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish immutable ZS32 bootstrap capture-gate assets"
    )
    parser.add_argument("--publication-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    return parser


def _run(argv: Sequence[str] | None) -> int:
    args = _parser().parse_args(argv)
    topology = load_topology(args.topology)
    if args.policy.is_symlink() or not args.policy.is_file():
        raise ValueError("capture gate policy must be a regular non-symlink file")
    policy_bytes = args.policy.read_bytes()
    policy = load_capture_gate_policy_bytes(policy_bytes)
    policy_bytes = canonical_json_bytes(policy.as_dict())
    policy.validate_topology(topology, allowed_hands=tuple(policy.hands))
    policy.verify_assets(args.asset_root)
    def read_asset(relative_path: str) -> bytes:
        path = args.asset_root.joinpath(*Path(relative_path).parts)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"capture gate asset is missing: {relative_path}")
        return path.read_bytes()

    verify_capture_gate_asset_semantics(topology, policy, read_asset)
    required = {"topology.json", "capture/gates/policy.json"}
    with AtomicDirectoryPublisher(args.output_root, args.publication_id) as publisher:
        publisher.write_bytes(
            "topology.json",
            canonical_json_bytes(topology.as_dict(include_sha256=False)),
        )
        publisher.write_bytes("capture/gates/policy.json", policy_bytes)
        artifacts = [policy.acquisition_config]
        for hand_policy in policy.hands.values():
            artifacts.extend((
                hand_policy.quality_profile,
                hand_policy.registration_profile,
                *hand_policy.registration_references.values(),
            ))
        for artifact in artifacts:
            publisher.copy_file(
                args.asset_root / artifact.relative_path,
                artifact.relative_path,
                expected_sha256=artifact.sha256,
            )
            required.add(artifact.relative_path)
        published = publisher.finalize(
            validator=lambda _staging: None,
            required_paths=frozenset(required),
        )
    command_result(
        "zs32-publish-gate-policy",
        {
            "publication_id": args.publication_id,
            "published_path": str(published),
            "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        require_linux_nvidia()
        return _run(argv)
    except Exception as error:
        return command_error("zs32-publish-gate-policy", error)


if __name__ == "__main__":
    raise SystemExit(main())
