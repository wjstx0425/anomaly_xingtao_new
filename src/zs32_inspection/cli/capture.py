"""Linux-only entry point for one topology-driven ZS32 capture."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from zs32_inspection.capture import (
    AtomicCaptureStore,
    CapturePlan,
    CaptureRequest,
    CaptureService,
    HikvisionCameraAdapter,
    HikvisionCaptureConfig,
    OpenCvQualityGate,
    OpenCvRegistrationGate,
    QualityGateProfile,
    RegistrationGateProfile,
    CaptureGatePolicy,
    ConsoleRoundCoordinator,
    load_verified_capture_gate_publication,
)
from zs32_inspection.domain.identity import Hand, PartIdentity
from zs32_inspection.domain.topology import CaptureTopology
from zs32_inspection.runtime.publisher import VerifiedAtomicPublication
from zs32_inspection.runtime.release_loader import (
    VerifiedReleaseFiles,
    load_verified_deployment_release,
)

from ._common import command_error, command_result
from ._environment import require_linux_nvidia


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture one complete ZS32 part on Linux + NVIDIA")
    gate_source = parser.add_mutually_exclusive_group(required=True)
    gate_source.add_argument("--release", type=Path)
    gate_source.add_argument("--gate-publication", type=Path)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--capture-session", required=True)
    parser.add_argument("--capture-set-id", required=True)
    parser.add_argument("--part-instance-id", required=True)
    parser.add_argument("--hand", choices=[item.value for item in Hand], required=True)
    parser.add_argument("--operator-id", required=True)
    parser.add_argument("--round-confirmation-timeout", type=float, default=120.0)
    return parser


@dataclass(frozen=True, slots=True)
class _GateInputs:
    topology: CaptureTopology
    policy: CaptureGatePolicy
    policy_sha256: str
    asset_root: Path
    publication: VerifiedAtomicPublication | None = None
    release_files: VerifiedReleaseFiles | None = None

    def read_json(self, relative_path: str) -> dict[str, object]:
        if self.publication is not None:
            import json

            payload = json.loads(self.publication.read_bytes(relative_path))
            if not isinstance(payload, dict):
                raise ValueError(f"gate publication JSON must be an object: {relative_path}")
            return payload
        if self.release_files is not None:
            return self.release_files.read_json(relative_path)
        raise RuntimeError("gate inputs have no verified JSON reader")


def _gate_inputs(args: argparse.Namespace) -> _GateInputs:
    if args.release is not None:
        release = load_verified_deployment_release(args.release)
        return _GateInputs(
            topology=release.contract.topology,
            policy=release.capture_gate_policy,
            policy_sha256=release.capture_gate_policy_file_sha256,
            asset_root=release.files.root,
            release_files=release.files,
        )

    verified = load_verified_capture_gate_publication(args.gate_publication)
    publication = verified.publication
    return _GateInputs(
        topology=verified.topology,
        policy=verified.policy,
        policy_sha256=verified.policy_sha256,
        asset_root=publication.root,
        publication=publication,
    )


def _run(argv: Sequence[str] | None) -> int:
    args = _parser().parse_args(argv)
    gate_inputs = _gate_inputs(args)
    topology = gate_inputs.topology
    plan = CapturePlan.from_topology(topology)
    request = CaptureRequest(
        capture_session=args.capture_session,
        capture_set_id=args.capture_set_id,
        part=PartIdentity(args.part_instance_id, Hand.parse(args.hand)),
    )
    if request.part.hand not in gate_inputs.policy.hands:
        raise ValueError(f"capture hand {request.hand!r} is not enabled by gate policy")
    if args.raw_root.exists() and (args.raw_root.is_symlink() or not args.raw_root.is_dir()):
        raise ValueError(f"raw root must be a directory or absent: {args.raw_root}")
    hand_policy = gate_inputs.policy.hands[request.part.hand]
    quality_profile = QualityGateProfile.from_mapping(
        gate_inputs.read_json(hand_policy.quality_profile.relative_path)
    )
    registration_profile = RegistrationGateProfile.from_mapping(
        gate_inputs.read_json(hand_policy.registration_profile.relative_path)
    )
    quality_profile.validate_plan(plan)
    registration_profile.validate_plan(plan)
    if quality_profile.hand != request.hand or registration_profile.hand != request.hand:
        raise ValueError("capture gate profile hand differs from requested part hand")
    profile_references = {
        view: (spec.reference.relative_path, spec.reference.sha256)
        for view, spec in registration_profile.views.items()
    }
    policy_references = {
        view: (artifact.relative_path, artifact.sha256)
        for view, artifact in hand_policy.registration_references.items()
    }
    if profile_references != policy_references:
        raise ValueError("registration profile references differ from release gate policy")
    acquisition_config = HikvisionCaptureConfig.from_mapping(
        gate_inputs.read_json(gate_inputs.policy.acquisition_config.relative_path)
    )
    with HikvisionCameraAdapter(acquisition_config) as source:
        result = CaptureService(
            source,
            AtomicCaptureStore(args.raw_root),
            gate_provenance=gate_inputs.policy.provenance_for(
                request.part.hand,
                policy_sha256=gate_inputs.policy_sha256,
            ),
            gates=(
                OpenCvQualityGate(quality_profile),
                OpenCvRegistrationGate(
                    registration_profile,
                    asset_root=gate_inputs.asset_root,
                ),
            ),
            round_coordinator=ConsoleRoundCoordinator(
                args.operator_id,
                args.round_confirmation_timeout,
            ),
        ).capture(request, plan)
    command_result(
        "zs32-capture",
        {
            "capture_session": result.capture_session,
            "capture_set_id": result.capture_set_id,
            "part_instance_id": result.part_instance_id,
            "published_path": result.published_path,
            "acquisition_config_sha256": (
                gate_inputs.policy.acquisition_config.sha256
            ),
            "image_sha256_by_view": dict(sorted(result.image_sha256_by_view.items())),
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Enforce the authoritative platform before parsing or touching capture state."""
    try:
        require_linux_nvidia()
        return _run(argv)
    except Exception as error:
        return command_error("zs32-capture", error)


if __name__ == "__main__":
    raise SystemExit(main())
