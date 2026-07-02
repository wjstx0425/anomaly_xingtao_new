# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export editable mask canvases and review sheets for manual geometry templates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import geometry_shape as geometry


def _read_image(path: Path) -> np.ndarray:
    """Read one image as BGR."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        msg = f"Could not read image: {path}"
        raise ValueError(msg)
    return image


def _write_mask(path: Path, mask: np.ndarray) -> None:
    """Write a binary editable mask image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def _resize_like(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a binary mask to ``shape`` if needed."""
    if mask.shape == shape:
        return mask
    height, width = shape
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0


def _choose_reference(paths: list[Path], preset: str, slot: str) -> Path:
    """Choose a representative normal image by median foreground area."""
    if not paths:
        msg = f"No normal images available for {slot}"
        raise FileNotFoundError(msg)
    scored: list[tuple[int, Path]] = []
    for path in paths:
        image = _read_image(path)
        mask = geometry.build_foreground_mask(image, slot_name=slot, preset=preset)
        scored.append((int(mask.sum()), path))
    scored.sort(key=lambda item: item[0])
    return scored[len(scored) // 2][1]


def _paste_tile(canvas: np.ndarray, tile: np.ndarray, x: int, y: int, label: str) -> None:
    """Paste one labelled tile into a review canvas."""
    height, width = tile.shape[:2]
    canvas[y : y + height, x : x + width] = tile
    cv2.putText(canvas, label, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def _overlay_masks(image: np.ndarray, expected: np.ndarray, allowed: np.ndarray, watch: np.ndarray, ignore: np.ndarray) -> np.ndarray:
    """Return a visual overlay for editable masks."""
    overlay = (image * 0.7).astype(np.uint8)
    expected = _resize_like(expected, image.shape[:2])
    allowed = _resize_like(allowed, image.shape[:2])
    watch = _resize_like(watch, image.shape[:2])
    ignore = _resize_like(ignore, image.shape[:2])
    overlay[watch] = (0.45 * overlay[watch] + 0.55 * np.array([0, 220, 255])).astype(np.uint8)
    overlay[ignore] = (0.35 * overlay[ignore] + 0.65 * np.array([180, 80, 220])).astype(np.uint8)
    for mask, color, thickness in (
        (allowed, (0, 200, 0), 1),
        (expected, (0, 255, 255), 2),
    ):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, thickness, cv2.LINE_AA)
    return overlay


def _sample_paths(paths: list[Path], count: int) -> list[Path]:
    """Return a deterministic spread of sample paths."""
    if count <= 0 or len(paths) <= count:
        return paths[:count] if count > 0 else []
    indexes = np.linspace(0, len(paths) - 1, count).round().astype(int)
    return [paths[int(index)] for index in indexes]


def _make_sheet(reference: np.ndarray, overlays: list[tuple[str, np.ndarray]], output_path: Path) -> None:
    """Write a compact visual review sheet."""
    tile_h, tile_w = reference.shape[:2]
    max_tiles = max(1, len(overlays) + 1)
    cols = min(3, max_tiles)
    rows = int(np.ceil(max_tiles / cols))
    canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    _paste_tile(canvas, reference, 0, 0, "reference")
    for index, (label, image) in enumerate(overlays, start=1):
        row = index // cols
        col = index % cols
        _paste_tile(canvas, image, col * tile_w, row * tile_h, label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def _initial_masks(
    reference: np.ndarray,
    *,
    slot: str,
    preset: str,
    template: geometry.GeometryTemplate | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return initial expected, allowed, ignore, and watch masks."""
    shape = reference.shape[:2]
    if template is not None:
        expected = _resize_like(template.expected_body, shape)
        allowed = _resize_like(template.allowed_body, shape)
        watch = _resize_like(template.edge_band, shape)
    else:
        expected = geometry.build_foreground_mask(reference, slot_name=slot, preset=preset)
        allowed = cv2.dilate(expected.astype(np.uint8), np.ones((17, 17), np.uint8), iterations=1) > 0
        watch = geometry.edge_band_for_mask(expected, pixels=35)
    ignore = geometry.preset_hole_mask(shape, slot, preset, dilation=32)
    expected &= ~ignore
    allowed &= ~ignore
    watch &= ~ignore
    return expected, allowed, ignore, watch


def export_review_pack(
    *,
    normal_root: Path,
    output_dir: Path,
    preset: str,
    slots: list[str] | None,
    template_dir: Path | None,
    stress_root: Path | None,
    defect_root: Path | None,
    samples_per_split: int,
) -> Path:
    """Export editable manual masks and visual review sheets."""
    normal_by_slot = geometry.group_paths_by_slot(geometry.iter_image_paths(normal_root))
    stress_by_slot = geometry.group_paths_by_slot(geometry.iter_image_paths(stress_root)) if stress_root else {}
    defect_by_slot = geometry.group_paths_by_slot(geometry.iter_image_paths(defect_root)) if defect_root else {}
    templates = geometry.load_templates(template_dir) if template_dir else {}
    preset_spec = geometry.get_preset(preset)
    requested_slots = slots or [slot.name for slot in preset_spec.slots]
    rows: list[dict[str, str | int]] = []

    for slot in requested_slots:
        normal_paths = normal_by_slot.get(slot, [])
        if not normal_paths:
            print(f"[warn] skip {slot}: no normal images found")
            continue
        reference_path = _choose_reference(normal_paths, preset, slot)
        reference = _read_image(reference_path)
        template = templates.get(slot)
        expected, allowed, ignore, watch = _initial_masks(reference, slot=slot, preset=preset, template=template)

        mask_dir = output_dir / "manual_masks"
        _write_mask(mask_dir / f"{slot}_expected.png", expected)
        _write_mask(mask_dir / f"{slot}_allowed.png", allowed)
        _write_mask(mask_dir / f"{slot}_ignore.png", ignore)
        _write_mask(mask_dir / f"{slot}_watch_edge.png", watch)

        reference_out = output_dir / "references" / f"{slot}_reference.png"
        reference_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(reference_out), reference)

        overlays: list[tuple[str, np.ndarray]] = [
            ("manual masks", _overlay_masks(reference, expected, allowed, watch, ignore)),
        ]
        for label, paths in (
            ("normal", normal_paths),
            ("stress", stress_by_slot.get(slot, [])),
            ("defect", defect_by_slot.get(slot, [])),
        ):
            for index, path in enumerate(_sample_paths(paths, samples_per_split), start=1):
                image = _read_image(path)
                overlays.append((f"{label}{index}", _overlay_masks(image, expected, allowed, watch, ignore)))

        sheet_path = output_dir / "sheets" / f"{slot}_review_sheet.png"
        _make_sheet(reference, overlays, sheet_path)
        rows.append(
            {
                "slot": slot,
                "reference_path": str(reference_path),
                "reference_image": str(reference_out),
                "review_sheet": str(sheet_path),
                "expected_mask": str(mask_dir / f"{slot}_expected.png"),
                "allowed_mask": str(mask_dir / f"{slot}_allowed.png"),
                "ignore_mask": str(mask_dir / f"{slot}_ignore.png"),
                "watch_edge_mask": str(mask_dir / f"{slot}_watch_edge.png"),
                "normal_count": len(normal_paths),
                "stress_count": len(stress_by_slot.get(slot, [])),
                "defect_count": len(defect_by_slot.get(slot, [])),
            },
        )

    if not rows:
        msg = f"No review pack rows were written from {normal_root}"
        raise RuntimeError(msg)
    manifest_path = output_dir / "review_pack_manifest.csv"
    geometry.write_csv(
        manifest_path,
        rows,
        [
            "slot",
            "reference_path",
            "reference_image",
            "review_sheet",
            "expected_mask",
            "allowed_mask",
            "ignore_mask",
            "watch_edge_mask",
            "normal_count",
            "stress_count",
            "defect_count",
        ],
    )
    print(f"Wrote manual geometry review pack to {output_dir}")
    print(f"Wrote manifest: {manifest_path}")
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-root", type=Path, required=True, help="Normal crop directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for review sheets and editable masks.")
    parser.add_argument("--preset", choices=geometry.preset_choices(), default="c789_left_top_3x2", help="Slot preset.")
    parser.add_argument("--slot", action="append", help="Optional slot to export. Repeatable.")
    parser.add_argument("--template-dir", type=Path, help="Optional existing template directory for initial masks.")
    parser.add_argument("--stress-root", type=Path, help="Optional locked stress-normal crop directory.")
    parser.add_argument("--defect-root", type=Path, help="Optional defect crop directory.")
    parser.add_argument("--samples-per-split", type=int, default=2, help="Number of images per split to show in sheets.")
    return parser


def main() -> None:
    """Run review-pack export."""
    args = build_parser().parse_args()
    export_review_pack(
        normal_root=args.normal_root,
        output_dir=args.output_dir,
        preset=args.preset,
        slots=args.slot,
        template_dir=args.template_dir,
        stress_root=args.stress_root,
        defect_root=args.defect_root,
        samples_per_split=args.samples_per_split,
    )


if __name__ == "__main__":
    main()
