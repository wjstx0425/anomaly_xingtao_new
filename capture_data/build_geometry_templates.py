# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Build slot-aware geometry templates from normal C789 part crops."""

from __future__ import annotations

import argparse
from pathlib import Path

import geometry_shape as geometry


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Folder-style crop dataset root.")
    parser.add_argument("--view", required=True, help="View name such as left_top.")
    parser.add_argument("--preset", choices=geometry.preset_choices(), default="c789_left_top_3x2", help="Slot preset.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for geometry templates.")
    parser.add_argument("--slot", action="append", help="Optional slot name to build. Repeatable.")
    parser.add_argument("--search-radius", type=int, default=25, help="Alignment search radius in pixels.")
    parser.add_argument("--coarse-step", type=int, default=5, help="Coarse alignment step in pixels.")
    parser.add_argument("--expected-occupancy", type=float, default=0.85, help="Stable body occupancy threshold.")
    parser.add_argument("--allowed-occupancy", type=float, default=0.85, help="Allowed body occupancy threshold.")
    parser.add_argument("--allowed-dilation", type=int, default=8, help="Allowed body dilation radius in pixels.")
    parser.add_argument("--edge-band-px", type=int, default=35, help="Boundary band radius for less/more scoring.")
    parser.add_argument("--hole-dilation", type=int, default=32, help="Preset hole mask dilation in pixels.")
    parser.add_argument("--border-margin", type=int, default=8, help="Foreground mask border margin in pixels.")
    parser.add_argument(
        "--foreground-threshold-scale",
        type=float,
        default=0.45,
        help="Scale applied to Otsu threshold for foreground extraction.",
    )
    return parser


def build_geometry_templates(args: argparse.Namespace) -> Path:
    """Build and save geometry templates."""
    normal_root = geometry.resolve_view_label_root(args.data_root, args.view, "normal")
    image_paths = geometry.iter_image_paths(normal_root)
    if not image_paths:
        msg = f"No normal crop images found under {normal_root}"
        raise FileNotFoundError(msg)

    grouped = geometry.group_paths_by_slot(image_paths)
    preset = geometry.get_preset(args.preset)
    requested_slots = args.slot or [slot.name for slot in preset.slots]
    rows: list[dict[str, str | int | float]] = []

    for slot in requested_slots:
        slot_paths = grouped.get(slot, [])
        if not slot_paths:
            print(f"[warn] skip {slot}: no images found")
            continue
        print(f"[build] {slot}: {len(slot_paths)} normal images")
        template = geometry.build_template_for_slot(
            slot_paths,
            slot=slot,
            preset=args.preset,
            search_radius=args.search_radius,
            coarse_step=args.coarse_step,
            expected_occupancy=args.expected_occupancy,
            allowed_occupancy=args.allowed_occupancy,
            allowed_dilation=args.allowed_dilation,
            edge_band_px=args.edge_band_px,
            hole_dilation=args.hole_dilation,
            border_margin=args.border_margin,
            foreground_threshold_scale=args.foreground_threshold_scale,
        )
        template_path = geometry.save_template(template, args.output_dir)
        overlay_path = args.output_dir / "overlays" / f"{slot}_template_overlay.png"
        geometry.write_template_overlay(template, overlay_path)
        rows.append(
            {
                "slot": slot,
                "source_count": template.source_count,
                "template_path": str(template_path),
                "overlay_path": str(overlay_path),
                "height": template.shape[0],
                "width": template.shape[1],
                "expected_body_area": int(template.expected_body.sum()),
                "allowed_body_area": int(template.allowed_body.sum()),
                "edge_band_area": int(template.edge_band.sum()),
                "expected_occupancy": template.expected_occupancy,
                "allowed_occupancy": template.allowed_occupancy,
            },
        )

    if not rows:
        msg = f"No templates were built from {normal_root}"
        raise RuntimeError(msg)

    manifest_path = args.output_dir / "template_manifest.csv"
    geometry.write_csv(
        manifest_path,
        rows,
        [
            "slot",
            "source_count",
            "template_path",
            "overlay_path",
            "height",
            "width",
            "expected_body_area",
            "allowed_body_area",
            "edge_band_area",
            "expected_occupancy",
            "allowed_occupancy",
        ],
    )
    print(f"Wrote {len(rows)} geometry templates to {args.output_dir}")
    print(f"Wrote manifest: {manifest_path}")
    return manifest_path


def main() -> None:
    """Run template building."""
    args = build_parser().parse_args()
    build_geometry_templates(args)


if __name__ == "__main__":
    main()
