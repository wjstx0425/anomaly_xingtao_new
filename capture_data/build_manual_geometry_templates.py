# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Build C789 geometry templates from manually edited mask PNGs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import geometry_shape as geometry


MASK_SUFFIXES = ("expected", "allowed", "ignore", "watch_edge")


def _read_binary_mask(path: Path) -> np.ndarray:
    """Read a binary mask where any non-zero pixel is active."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        msg = f"Could not read mask: {path}"
        raise FileNotFoundError(msg)
    return mask > 0


def _resize_like(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a binary mask to ``shape`` if needed."""
    if mask.shape == shape:
        return mask
    height, width = shape
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0


def _mask_path(mask_dir: Path, slot: str, suffix: str) -> Path:
    """Return the expected path for one manual slot mask."""
    return mask_dir / f"{slot}_{suffix}.png"


def _load_optional_ignore(mask_dir: Path, slot: str, shape: tuple[int, int]) -> np.ndarray:
    """Load the optional ignore mask for one slot."""
    path = _mask_path(mask_dir, slot, "ignore")
    if not path.is_file():
        return np.zeros(shape, dtype=bool)
    return _resize_like(_read_binary_mask(path), shape)


def build_manual_template(mask_dir: Path, slot: str) -> geometry.GeometryTemplate:
    """Build one geometry template from manually edited masks."""
    expected_path = _mask_path(mask_dir, slot, "expected")
    allowed_path = _mask_path(mask_dir, slot, "allowed")
    watch_path = _mask_path(mask_dir, slot, "watch_edge")
    missing = [path for path in (expected_path, allowed_path, watch_path) if not path.is_file()]
    if missing:
        msg = "Missing manual mask(s): " + ", ".join(str(path) for path in missing)
        raise FileNotFoundError(msg)

    expected = _read_binary_mask(expected_path)
    allowed = _resize_like(_read_binary_mask(allowed_path), expected.shape)
    watch_edge = _resize_like(_read_binary_mask(watch_path), expected.shape)
    ignore = _load_optional_ignore(mask_dir, slot, expected.shape)

    expected = expected & ~ignore
    allowed = allowed & ~ignore
    watch_edge = watch_edge & ~ignore
    allowed |= expected

    return geometry.GeometryTemplate(
        slot=slot,
        expected_body=expected,
        allowed_body=allowed,
        edge_band=watch_edge,
        source_count=1,
        expected_occupancy=1.0,
        allowed_occupancy=1.0,
    )


def _slot_names_from_mask_dir(mask_dir: Path) -> list[str]:
    """Infer slot names that have manual expected masks."""
    slots = []
    for path in sorted(mask_dir.glob("slot*_expected.png")):
        slots.append(path.name.removesuffix("_expected.png"))
    return slots


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-dir", type=Path, required=True, help="Directory with slotXX_expected/allowed/ignore/watch_edge PNGs.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for compiled geometry templates.")
    parser.add_argument("--slot", action="append", help="Optional slot name to compile. Repeatable.")
    return parser


def build_manual_templates(args: argparse.Namespace) -> Path:
    """Build all requested manual geometry templates."""
    slots = args.slot or _slot_names_from_mask_dir(args.mask_dir)
    if not slots:
        msg = f"No slot masks found in {args.mask_dir}"
        raise FileNotFoundError(msg)

    rows: list[dict[str, str | int | float]] = []
    for slot in slots:
        template = build_manual_template(args.mask_dir, slot)
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
    print(f"Wrote {len(rows)} manual geometry templates to {args.output_dir}")
    print(f"Wrote manifest: {manifest_path}")
    return manifest_path


def main() -> None:
    """Run manual template building."""
    args = build_parser().parse_args()
    build_manual_templates(args)


if __name__ == "__main__":
    main()
