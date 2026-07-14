"""Run one complete ZS32 inspection from immutable release and capture inputs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

from zs32_inspection.capture.errors import CaptureDataIntegrityError
from zs32_inspection.domain.decisions import InspectionStatus
from zs32_inspection.models.base import DeviceSpec
from zs32_inspection.runtime.code_identity import verify_runtime_code_identity
from zs32_inspection.runtime.cropper import create_opencv_runtime_cropper
from zs32_inspection.runtime.environment_receipt import (
    RuntimeEnvironmentMismatchError,
    verify_runtime_environment,
)
from zs32_inspection.runtime.inspection_sink import FilesystemInspectionSink
from zs32_inspection.runtime.orchestrator import InspectionOrchestrator, InspectionRequest
from zs32_inspection.runtime.preflight import publish_preflight_failure
from zs32_inspection.runtime.release_loader import load_verified_deployment_release
from zs32_inspection.runtime.runtime_factory import build_loaded_runtime

from ._common import command_error, command_result
from ._environment import require_linux_nvidia


def _verify_release_runtime_environment(runtime_release: object, *, device: str) -> object:
    """Keep the live environment check replaceable in isolated CLI tests."""
    return verify_runtime_environment(runtime_release.runtime_environment, device=device)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect one verified complete ZS32 capture")
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--capture-set-root", type=Path, required=True)
    parser.add_argument("--inspection-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="0")
    return parser


def _run(argv: Sequence[str] | None) -> int:
    args = _parser().parse_args(argv)
    if args.output_root.exists() and (args.output_root.is_symlink() or not args.output_root.is_dir()):
        raise ValueError(f"inspection output root must be a directory or absent: {args.output_root}")
    try:
        runtime_release = load_verified_deployment_release(args.release)
    except Exception as error:
        return _publish_preflight(
            args=args,
            status=InspectionStatus.SYSTEM_ERROR,
            stage="release_load",
            error=error,
        )
    try:
        verify_runtime_code_identity(
            runtime_release.files.read_json("provenance/code_version.json")
        )
    except Exception as error:
        return _publish_preflight(
            args=args,
            status=InspectionStatus.SYSTEM_ERROR,
            stage="runtime_code_identity",
            error=error,
        )
    try:
        _verify_release_runtime_environment(runtime_release, device=args.device)
    except RuntimeEnvironmentMismatchError as error:
        return _publish_preflight(
            args=args,
            status=InspectionStatus.SYSTEM_ERROR,
            stage="runtime_environment",
            error=error,
            runtime_environment_audit={
                "differences": list(error.differences),
                "expected": error.expected.as_dict(),
                "observed": error.observed.as_dict(),
            },
        )
    except Exception as error:
        return _publish_preflight(
            args=args,
            status=InspectionStatus.SYSTEM_ERROR,
            stage="runtime_environment",
            error=error,
        )
    try:
        request = InspectionRequest.from_capture_publication(
            inspection_id=args.inspection_id,
            release_id=runtime_release.manifest.release_id,
            capture_set_root=args.capture_set_root,
            topology=runtime_release.contract.topology,
        )
        if request.capture.part.hand not in runtime_release.contract.allowed_hands:
            raise CaptureDataIntegrityError(
                f"capture hand {request.capture.part.hand.value!r} is not enabled by release"
            )
    except CaptureDataIntegrityError as error:
        return _publish_preflight(
            args=args,
            status=InspectionStatus.INVALID_CAPTURE,
            stage="capture_load",
            error=error,
        )
    except Exception as error:
        return _publish_preflight(
            args=args,
            status=InspectionStatus.SYSTEM_ERROR,
            stage="capture_load",
            error=error,
        )
    try:
        device = DeviceSpec(accelerator="gpu", device=args.device)
        runtime = build_loaded_runtime(runtime_release, device=device)
    except Exception as error:
        return _publish_preflight(
            args=args,
            status=InspectionStatus.SYSTEM_ERROR,
            stage="runtime_load",
            error=error,
        )
    with TemporaryDirectory(prefix="zs32-crops-") as temporary_root:
        outcome = InspectionOrchestrator(
            cropper=create_opencv_runtime_cropper(Path(temporary_root)),
            sink=FilesystemInspectionSink(args.output_root),
        ).inspect(request, runtime)
    decision = outcome.run.decision
    command_result(
        "zs32-inspect",
        {
            "inspection_id": args.inspection_id,
            "inspection_status": decision.inspection_status.value,
            "evidence_status": (
                None if decision.evidence_status is None else decision.evidence_status.value
            ),
            "review_status": decision.review_status,
            "released_status": (
                None if decision.released_status is None else decision.released_status.value
            ),
            "reason_codes": list(decision.reason_codes),
            "published_path": str(outcome.published_path),
        },
    )
    if decision.inspection_status in {
        InspectionStatus.RETAKE,
        InspectionStatus.INVALID_CAPTURE,
        InspectionStatus.SYSTEM_ERROR,
    }:
        return 2
    return 0


def _publish_preflight(
    *,
    args: argparse.Namespace,
    status: InspectionStatus,
    stage: str,
    error: Exception,
    runtime_environment_audit: dict[str, object] | None = None,
) -> int:
    reason_code = f"{stage}_failed:{type(error).__name__}:{error}"
    published_path = publish_preflight_failure(
        output_root=args.output_root,
        inspection_id=args.inspection_id,
        status=status,
        reason_code=reason_code,
        release_path=args.release,
        capture_set_root=args.capture_set_root,
        runtime_environment_audit=runtime_environment_audit,
    )
    command_result(
        "zs32-inspect",
        {
            "status": "failed",
            "inspection_id": args.inspection_id,
            "inspection_status": status.value,
            "evidence_status": None,
            "review_status": None,
            "released_status": None,
            "reason_codes": [reason_code],
            "published_path": str(published_path),
        },
    )
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    try:
        require_linux_nvidia()
        return _run(argv)
    except Exception as error:
        return command_error("zs32-inspect", error)


if __name__ == "__main__":
    raise SystemExit(main())
