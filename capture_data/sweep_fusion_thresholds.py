"""Sweep selective exposure-fusion thresholds and build a comparison sheet."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from exposure_fusion import align_mtb, read_bgr, selective_long_exposure_fusion


def parse_values(spec: str) -> list[float]:
    """Parse comma-separated values and inclusive ranges like 40:120:10."""
    values: list[float] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue

        if ":" not in item:
            values.append(float(item))
            continue

        parts = [float(part) for part in item.split(":")]
        if len(parts) not in {2, 3}:
            raise ValueError(f"Invalid range: {item}")

        start, stop = parts[0], parts[1]
        step = parts[2] if len(parts) == 3 else 1.0
        if step == 0:
            raise ValueError(f"Range step cannot be zero: {item}")

        current = start
        if step > 0:
            while current <= stop + 1e-9:
                values.append(current)
                current += step
        else:
            while current >= stop - 1e-9:
                values.append(current)
                current += step

    if not values:
        raise ValueError("No values parsed")
    return values


def format_value(value: float) -> str:
    """Format a numeric value for labels and filenames."""
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def parse_roi(spec: str | None) -> tuple[int, int, int, int] | None:
    """Parse ROI as x,y,w,h."""
    if spec is None:
        return None

    parts = [int(part.strip()) for part in spec.split(",")]
    if len(parts) != 4:
        raise ValueError("--roi must be x,y,w,h")

    x, y, width, height = parts
    if width <= 0 or height <= 0:
        raise ValueError("ROI width and height must be positive")
    return x, y, width, height


def crop_roi(image: np.ndarray, roi: tuple[int, int, int, int] | None) -> np.ndarray:
    """Crop an image to ROI if provided."""
    if roi is None:
        return image

    x, y, width, height = roi
    return image[y : y + height, x : x + width]


def image_metrics(
    image: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
) -> dict[str, float]:
    """Compute rough quality metrics for sorting candidates."""
    cropped = crop_roi(image, roi)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    dark_pct = float(np.mean(gray <= 8.0))
    clip_pct = float(np.mean(gray >= 250.0))
    midtone_pct = float(np.mean((gray >= 30.0) & (gray <= 230.0)))
    p05 = float(np.percentile(gray, 5))
    p95 = float(np.percentile(gray, 95))
    contrast = p95 - p05
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # This is only a rough ranking helper. The contact sheet is the real judge.
    score = (
        (contrast / 255.0)
        + min(sharpness / 500.0, 2.0)
        + midtone_pct
        - 3.0 * clip_pct
        - 0.4 * dark_pct
    )

    return {
        "score": score,
        "dark_pct": dark_pct,
        "clip_pct": clip_pct,
        "midtone_pct": midtone_pct,
        "p05": p05,
        "p95": p95,
        "contrast": contrast,
        "sharpness": sharpness,
    }


def make_tile(
    image: np.ndarray,
    preview_width: int,
    label_lines: list[str],
) -> np.ndarray:
    """Create a labeled preview tile."""
    scale = preview_width / image.shape[1]
    preview_height = max(1, int(round(image.shape[0] * scale)))
    preview = cv2.resize(
        image,
        (preview_width, preview_height),
        interpolation=cv2.INTER_AREA,
    )

    label_height = 56
    tile = np.zeros((preview_height + label_height, preview_width, 3), dtype=np.uint8)
    tile[:preview_height] = preview
    tile[preview_height:] = (24, 24, 24)

    y = preview_height + 20
    for line in label_lines[:2]:
        cv2.putText(
            tile,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        y += 22

    return tile


def build_contact_sheet(tiles: list[list[np.ndarray]], gap: int = 8) -> np.ndarray:
    """Build a row/column contact sheet from tiles."""
    if not tiles or not tiles[0]:
        raise ValueError("No tiles to compose")

    tile_height, tile_width = tiles[0][0].shape[:2]
    rows = len(tiles)
    cols = len(tiles[0])
    sheet_height = rows * tile_height + (rows + 1) * gap
    sheet_width = cols * tile_width + (cols + 1) * gap
    sheet = np.full((sheet_height, sheet_width, 3), 16, dtype=np.uint8)

    for row_index, row in enumerate(tiles):
        for col_index, tile in enumerate(row):
            y = gap + row_index * (tile_height + gap)
            x = gap + col_index * (tile_width + gap)
            sheet[y : y + tile_height, x : x + tile_width] = tile

    return sheet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("short_image", help="Short-exposure image, e.g. 7000us")
    parser.add_argument("long_image", help="Long-exposure image, e.g. 40000us")
    parser.add_argument("--out-dir", default="./hik_images/fusion_sweep")
    parser.add_argument("--short-dark-values", default="40:120:10")
    parser.add_argument("--long-clip-values", default="230:250:5")
    parser.add_argument("--blend-width", type=float, default=18.0)
    parser.add_argument("--blur-size", type=int, default=31)
    parser.add_argument("--preview-width", type=int, default=360)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--save-all", action="store_true")
    parser.add_argument("--no-align", action="store_true")
    parser.add_argument(
        "--roi",
        help="Optional score ROI as x,y,w,h. Useful for scoring only the part area.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_dir = out_dir / "all"
    top_dir = out_dir / "top"
    if args.save_all:
        all_dir.mkdir(parents=True, exist_ok=True)
    if args.top_k > 0:
        top_dir.mkdir(parents=True, exist_ok=True)

    short_values = parse_values(args.short_dark_values)
    long_values = parse_values(args.long_clip_values)
    roi = parse_roi(args.roi)

    images = [read_bgr(args.short_image), read_bgr(args.long_image)]
    if not args.no_align:
        images = align_mtb(images)

    short_image, long_image = images
    rows: list[list[np.ndarray]] = []
    records: list[dict[str, float | str]] = []
    top_candidates: list[tuple[float, str, np.ndarray]] = []

    for short_dark in short_values:
        row_tiles: list[np.ndarray] = []
        for long_clip in long_values:
            fused = selective_long_exposure_fusion(
                short_image,
                long_image,
                short_dark_threshold=short_dark,
                long_clip_threshold=long_clip,
                blend_width=args.blend_width,
                blur_size=args.blur_size,
            )
            metrics = image_metrics(fused, roi)
            stem = (
                f"sd{format_value(short_dark)}"
                f"_lc{format_value(long_clip)}"
                f"_bw{format_value(args.blend_width)}"
                f"_blur{args.blur_size}"
            )

            if args.save_all:
                cv2.imwrite(str(all_dir / f"{stem}.png"), fused)

            top_candidates.append((float(metrics["score"]), stem, fused.copy()))
            top_candidates.sort(key=lambda item: item[0], reverse=True)
            top_candidates = top_candidates[: max(args.top_k, 0)]

            record: dict[str, float | str] = {
                "file_stem": stem,
                "short_dark_threshold": short_dark,
                "long_clip_threshold": long_clip,
                "blend_width": args.blend_width,
                "blur_size": args.blur_size,
            }
            record.update(metrics)
            records.append(record)

            tile = make_tile(
                fused,
                args.preview_width,
                [
                    f"dark={format_value(short_dark)} clip={format_value(long_clip)}",
                    f"score={metrics['score']:.3f} clip={metrics['clip_pct']:.3%}",
                ],
            )
            row_tiles.append(tile)
        rows.append(row_tiles)

    contact_sheet = build_contact_sheet(rows)
    contact_path = out_dir / "contact_sheet.png"
    cv2.imwrite(str(contact_path), contact_sheet)

    csv_path = out_dir / "metrics.csv"
    fieldnames = [
        "file_stem",
        "short_dark_threshold",
        "long_clip_threshold",
        "blend_width",
        "blur_size",
        "score",
        "dark_pct",
        "clip_pct",
        "midtone_pct",
        "p05",
        "p95",
        "contrast",
        "sharpness",
    ]
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    for rank, (score, stem, image) in enumerate(top_candidates, start=1):
        top_path = top_dir / f"top{rank:02d}_{stem}_score{score:.3f}.png"
        cv2.imwrite(str(top_path), image)

    best = top_candidates[0] if top_candidates else None
    print(f"Saved contact sheet: {contact_path}")
    print(f"Saved metrics: {csv_path}")
    if best is not None:
        print(f"Top rough-score candidate: {best[1]} score={best[0]:.3f}")
        print(f"Saved top candidates: {top_dir}")
    if args.save_all:
        print(f"Saved all full-resolution candidates: {all_dir}")


if __name__ == "__main__":
    main()
