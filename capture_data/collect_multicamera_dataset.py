# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pure mapping and command-line schema for ZS32 multi-camera capture.

Hardware SDK imports intentionally live outside this module's import path so
the mapping and CLI contract remain testable on machines without the MVS SDK.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

ROUND_VIEWS: dict[str, tuple[str, str, str]] = {
    "front": ("front", "front_left", "front_right"),
    "back": ("back", "back_left", "back_right"),
}


def validate_devices(devices: Sequence[int]) -> tuple[int, int, int]:
    """Validate and preserve the physical ordering of three camera indices.

    Args:
        devices: Device indices ordered as front, left-side, and right-side camera slots.

    Returns:
        The three device indices with their input order preserved.

    Raises:
        ValueError: If the input does not contain exactly three unique indices.
    """
    if len(devices) != 3 or len(set(devices)) != 3:
        msg = "--devices requires exactly three unique device indices"
        raise ValueError(msg)
    return devices[0], devices[1], devices[2]


def view_for(round_name: str, camera_slot: int) -> str:
    """Return the view assigned to a camera slot for one capture round.

    Args:
        round_name: Capture round name, either ``front`` or ``back``.
        camera_slot: Zero-based physical camera slot.

    Returns:
        The dataset view name assigned to the slot.
    """
    return ROUND_VIEWS[round_name][camera_slot]


class _CaptureArgumentParser(argparse.ArgumentParser):
    """Argument parser with capture-only required options."""

    def parse_args(
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        """Allow discovery alone while requiring label and HDR for capture."""
        parsed = super().parse_args(args, namespace)
        if not parsed.list_devices:
            if parsed.label is None:
                self.error("the following arguments are required: --label")
            if not parsed.hdr:
                self.error("the following arguments are required: --hdr")
        return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the SDK-free CLI parser for six-view ZS32 capture.

    Returns:
        The configured argument parser.
    """
    parser = _CaptureArgumentParser(description="Collect six-view ZS32 HDR images from three cameras.")
    parser.add_argument("--devices", nargs=3, type=int, default=[0, 1, 2])
    parser.add_argument("--hand", choices=("left",), default="left")
    parser.add_argument("--label", choices=("normal", "defect"))
    parser.add_argument("--defect-type", default="")
    parser.add_argument("--part-id", default="part001")
    parser.add_argument("--group-count", type=int, default=1)
    parser.add_argument("--images-per-group", type=int, default=1)
    parser.add_argument("--manual-load", action="store_true")

    parser.add_argument("--hdr", action="store_true")
    parser.add_argument("--short-exposure", type=float, default=7000.0)
    parser.add_argument("--long-exposure", type=float, default=40000.0)
    parser.add_argument("--gain", type=float)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--hdr-settle-frames", type=int, default=5)
    parser.add_argument("--timeout-ms", type=int, default=3000)
    parser.add_argument("--short-dark-threshold", type=float, default=70.0)
    parser.add_argument("--long-clip-threshold", type=float, default=245.0)
    parser.add_argument("--blend-width", type=float, default=18.0)
    parser.add_argument("--blur-size", type=int, default=31)
    parser.add_argument("--align-hdr", action="store_true")
    parser.add_argument("--save-hdr-sources", action="store_true")
    parser.add_argument("--hdr-max-retries", type=int, default=2)
    parser.add_argument("--hdr-max-clip-pct", type=float, default=12.0)

    parser.add_argument("--root", default="./dataset")
    parser.add_argument("--list-devices", action="store_true")
    return parser
