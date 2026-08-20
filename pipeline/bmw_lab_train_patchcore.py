#!/usr/bin/env python3
"""Train and calibrate six BMW laboratory PatchCore models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from bmw_inspection.lab.patchcore import PatchCoreTrainingProfile, train_patchcore_views


def build_parser() -> argparse.ArgumentParser:
    """Build the frozen six-view PatchCore training CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--backbone", default="wide_resnet50_2")
    parser.add_argument("--layers", nargs="+", default=["layer2", "layer3"])
    parser.add_argument("--image-size", nargs=2, type=int, default=[512, 512])
    parser.add_argument("--coreset-sampling-ratio", type=float, default=0.1)
    parser.add_argument("--num-neighbors", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="1")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run serial per-view training and calibration."""
    args = build_parser().parse_args(argv)
    profile = PatchCoreTrainingProfile(
        backbone=args.backbone,
        layers=tuple(args.layers),
        image_size=tuple(args.image_size),
        coreset_sampling_ratio=args.coreset_sampling_ratio,
        num_neighbors=args.num_neighbors,
        seed=args.seed,
        max_epochs=args.max_epochs,
    )
    devices: int | str = int(args.devices) if args.devices.isdecimal() else args.devices
    report = train_patchcore_views(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        profile=profile,
        accelerator=args.accelerator,
        devices=devices,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
