# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fast isolated multi-camera bootstrap capture command."""

from __future__ import annotations

import argparse
import getpass
import hashlib
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from zs32_inspection.capture import (
    CapturePlan,
    CaptureRequest,
    HikvisionCameraAdapter,
    HikvisionCaptureConfig,
)
from zs32_inspection.capture.bootstrap import (
    AtomicBootstrapCaptureStore,
    BootstrapCaptureMetadata,
    BootstrapCaptureService,
    BootstrapCaptureStore,
    BootstrapRoundCoordinator,
    DashboardRoundCoordinator,
)
from zs32_inspection.capture.legacy_dataset import (
    LegacyCaptureContext,
    LegacyDatasetCaptureStore,
)
from zs32_inspection.config.loaders import load_topology
from zs32_inspection.domain.identity import Hand, PartIdentity
from zs32_inspection.runtime.publisher import canonical_json_bytes

from ._common import command_error, command_result


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _one_image_per_group(value: str) -> int:
    parsed = int(value)
    if parsed != 1:
        raise argparse.ArgumentTypeError(
            "four-camera bootstrap currently requires --images-per-group 1"
        )
    return parsed


def _default_session(*, legacy_layout: bool) -> str:
    now = datetime.now()
    if legacy_layout:
        return now.strftime("%Y%m%d_%H%M%S_%f")
    return now.astimezone(timezone.utc).strftime("bootstrap-%Y%m%d-%H%M%S-%f")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture topology-driven ZS32 bootstrap data without quality/registration gates. "
            "Output is isolated and cannot enter the formal dataset automatically."
        )
    )
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--output-root", "--root", dest="output_root", type=Path, required=True)
    parser.add_argument("--capture-session", default=None)
    parser.add_argument(
        "--legacy-layout",
        action="store_true",
        help="Write the historical per-view dataset tree and CSV manifest.",
    )
    parser.add_argument("--operator-id", default=getpass.getuser())
    parser.add_argument("--round-confirmation-timeout", type=float, default=120.0)
    parser.add_argument("--progress-json", type=Path)
    parser.add_argument("--control-json", type=Path)
    parser.add_argument("--hand", choices=[item.value for item in Hand], required=True)
    parser.add_argument("--label", choices=("normal", "defect"), required=True)
    parser.add_argument("--defect-type")
    parser.add_argument("--part-id", required=True)
    parser.add_argument("--group-count", type=_positive_int, default=1)
    parser.add_argument(
        "--images-per-group",
        type=_one_image_per_group,
        default=1,
        help="Must be 1: each physical part produces one complete front/back 8-view set.",
    )
    parser.add_argument(
        "--manual-load",
        action="store_true",
        help=(
            "Show explicit loading guidance before the front round; "
            "both rounds still wait for Enter."
        ),
    )

    parser.add_argument("--hdr", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--exposure", type=float, default=4000.0)
    parser.add_argument("--short-exposure", type=float, default=4000.0)
    parser.add_argument("--long-exposure", type=float, default=35000.0)
    parser.add_argument("--gain", type=float, default=0.0)
    parser.add_argument("--timeout-ms", type=_positive_int, default=3000)
    parser.add_argument("--capture-interval", type=float, default=0.2)
    parser.add_argument("--hdr-settle-frames", type=_non_negative_int, default=8)
    parser.add_argument("--align-hdr", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--short-dark-threshold", type=float, default=80.0)
    parser.add_argument("--long-clip-threshold", type=float, default=245.0)
    parser.add_argument("--blend-width", type=float, default=50.0)
    parser.add_argument("--blur-size", type=_positive_int, default=101)
    parser.add_argument(
        "--hdr-max-retries",
        type=_non_negative_int,
        default=0,
        help=(
            "Bootstrap default is 0 for fastest capture; "
            "increase only if HDR clipping requires it."
        ),
    )
    parser.add_argument("--hdr-max-clip-pct", type=float, default=5.0)
    parser.add_argument("--png-compression", type=_non_negative_int, default=3)
    parser.add_argument("--frame-buffer-size", type=_positive_int, default=50 * 1024 * 1024)
    return parser


def _acquisition_config(args: argparse.Namespace) -> HikvisionCaptureConfig:
    return HikvisionCaptureConfig(
        exposure=args.exposure,
        gain=args.gain,
        timeout_ms=args.timeout_ms,
        capture_interval=args.capture_interval,
        hdr=args.hdr,
        short_exposure=args.short_exposure,
        long_exposure=args.long_exposure,
        hdr_settle_frames=args.hdr_settle_frames,
        align_hdr=args.align_hdr,
        short_dark_threshold=args.short_dark_threshold,
        long_clip_threshold=args.long_clip_threshold,
        blend_width=args.blend_width,
        blur_size=args.blur_size,
        hdr_max_retries=args.hdr_max_retries,
        hdr_max_clip_pct=args.hdr_max_clip_pct,
        png_compression=args.png_compression,
        frame_buffer_size=args.frame_buffer_size,
    )


def _validate_label(label: str, defect_type: str | None) -> str | None:
    normalized = defect_type.strip() if isinstance(defect_type, str) else None
    normalized = normalized or None
    if label == "defect" and normalized is None:
        raise ValueError("--defect-type is required when --label defect")
    if label == "normal" and normalized is not None:
        raise ValueError("--defect-type is only valid when --label defect")
    return normalized


def _run(argv: Sequence[str] | None) -> int:
    args = _parser().parse_args(argv)
    topology = load_topology(args.topology)
    plan = CapturePlan.from_topology(topology)
    defect_type = _validate_label(args.label, args.defect_type)
    acquisition = _acquisition_config(args)
    acquisition_payload = acquisition.as_dict()
    acquisition_sha256 = hashlib.sha256(canonical_json_bytes(acquisition_payload)).hexdigest()
    metadata = BootstrapCaptureMetadata(
        label=args.label,
        defect_type=defect_type,
        acquisition_config=acquisition_payload,
        acquisition_config_sha256=acquisition_sha256,
    )
    capture_session = args.capture_session or _default_session(
        legacy_layout=args.legacy_layout
    )
    if (args.progress_json is None) != (args.control_json is None):
        raise ValueError("--progress-json and --control-json must be provided together")
    coordinator = (
        DashboardRoundCoordinator(
            args.operator_id,
            args.progress_json,
            args.control_json,
            timeout_seconds=args.round_confirmation_timeout,
        )
        if args.progress_json is not None
        else BootstrapRoundCoordinator(
            args.operator_id,
            args.round_confirmation_timeout,
            manual_load=args.manual_load,
        )
    )
    requests: list[CaptureRequest] = []
    for group_index in range(1, args.group_count + 1):
        part_instance_id = f"{args.part_id}_group{group_index:03d}"
        for image_index in range(1, args.images_per_group + 1):
            requests.append(
                CaptureRequest(
                    capture_session=capture_session,
                    capture_set_id=f"{part_instance_id}_{image_index:06d}",
                    part=PartIdentity(part_instance_id, Hand.parse(args.hand)),
                )
            )

    store: BootstrapCaptureStore
    legacy_store: LegacyDatasetCaptureStore | None = None
    if args.legacy_layout:
        legacy_store = LegacyDatasetCaptureStore(
            args.output_root,
            LegacyCaptureContext(args.label, defect_type, args.part_id),
        )
        legacy_store.preflight(requests, plan)
        store = legacy_store
    else:
        store = AtomicBootstrapCaptureStore(args.output_root)

    published_paths: list[str] = []
    image_hashes_by_set: dict[str, dict[str, str]] = {}

    # Keep all camera handles open for the entire batch. This is substantially
    # faster and less error-prone than enumerating/opening four devices per part.
    with HikvisionCameraAdapter(acquisition) as source:
        service = BootstrapCaptureService(source, store, round_coordinator=coordinator)
        for request in requests:
            result = service.capture(request, plan, metadata)
            published_paths.append(result.published_path)
            image_hashes_by_set[result.capture_set_id] = dict(result.image_sha256_by_view)

    if legacy_store is not None:
        published_paths = [str(legacy_store.manifest_path)]

    command_result(
        "zs32-bootstrap-capture",
        {
            "bootstrap_only": True,
            "eligible_for_dataset": False,
            "capture_session": capture_session,
            "capture_set_count": len(requests),
            "topology_id": plan.topology_id,
            "topology_sha256": plan.topology_sha256,
            "acquisition_config_sha256": acquisition_sha256,
            "published_paths": published_paths,
            "image_sha256_by_set": image_hashes_by_set,
            "output_layout": "legacy" if args.legacy_layout else "bootstrap",
            "manifest_path": str(legacy_store.manifest_path) if legacy_store else None,
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run bootstrap capture without imposing an unrelated NVIDIA requirement."""
    try:
        return _run(argv)
    except Exception as error:
        return command_error("zs32-bootstrap-capture", error)


if __name__ == "__main__":
    raise SystemExit(main())
