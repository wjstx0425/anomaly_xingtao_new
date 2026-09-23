"""Numerical horizontal motion-blur comparison at original pixel resolution."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def blur_right(image: np.ndarray, displacement: int) -> np.ndarray:
    """Integrate translations from 0 to displacement using linear interpolation."""
    if displacement == 0:
        return image.copy()
    if displacement < 0:
        raise ValueError("displacement must be nonnegative")
    weights = np.ones(displacement + 1, dtype=np.float32) / displacement
    weights[[0, -1]] *= 0.5
    result = cv2.filter2D(
        image.astype(np.float32), -1, weights[None, :],
        anchor=(displacement, 0), borderType=cv2.BORDER_REPLICATE,
    )
    return np.rint(np.clip(result, 0, 255)).astype(np.uint8)


def save_png(path: Path, image: np.ndarray) -> str:
    """Write lossless PNG without overriding existing files."""
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"encoding failed: {path}")
    data = encoded.tobytes()
    with path.open("xb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pixels", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--roi", nargs=4, type=int, action="append", required=True, metavar=("X1", "Y1", "X2", "Y2"))
    args = parser.parse_args()
    if any(value <= 0 for value in args.pixels) or len(set(args.pixels)) != len(args.pixels):
        parser.error("--pixels requires unique positive integers")
    displacements = [0, *sorted(args.pixels)]
    label = " / ".join(str(value) for value in displacements[1:])
    data = args.image.read_bytes()
    source_hash = hashlib.sha256(data).hexdigest()
    original = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if original is None or original.dtype != np.uint8 or original.ndim != 3 or original.shape[2] != 3:
        raise ValueError("source must be uint8 BGR")
    height, width = original.shape[:2]
    for x1, y1, x2, y2 in args.roi:
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("ROI outside image bounds")
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "source": str(args.image.resolve()), "source_sha256": source_hash,
        "shape_hwc": list(original.shape), "direction": "right", "roi_xyxy": args.roi,
        "model": "uniform translation t=0..L of linearly interpolated input pixel values",
        "kernel": "L+1 taps: [0.5, 1, ..., 1, 0.5] / L; L=1 -> [0.5, 0.5]",
        "border": "replicate", "working_space": "stored PNG values; unknown source response; no gamma changes",
        "limitations": "Added blur on already captured image, including its noise/background; not sensor ground truth or object-only motion.",
        "opencv_version": cv2.__version__, "outputs": [],
    }
    crops: list[list[np.ndarray]] = [[] for _ in args.roi]
    cards = []
    for displacement in displacements:
        simulated = blur_right(original, displacement)
        name = f"motion_right_{displacement}px.png"
        checksum = save_png(args.output / name, simulated)
        metadata["outputs"].append({"displacement_px": displacement, "path": name, "sha256": checksum})
        for index, (x1, y1, x2, y2) in enumerate(args.roi):
            crop = simulated[y1:y2, x1:x2]
            enlarged = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
            crops[index].append(enlarged)
            save_png(args.output / f"roi{index + 1}_{displacement}px_4x.png", enlarged)
        thumb = cv2.resize(simulated, (664, round(height * 664 / width)), interpolation=cv2.INTER_AREA)
        save_png(args.output / f"preview_{displacement}px.png", thumb)
        cards.append(f'<figure><a href="{name}"><img src="preview_{displacement}px.png"></a><figcaption>{displacement} px</figcaption></figure>')
    roi_html = []
    for index, images in enumerate(crops):
        h, w = images[0].shape[:2]
        columns = 2 if len(displacements) == 4 else 3
        rows = (len(displacements) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * (w + 12) + 12, rows * (h + 36) + 12), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        for position, (displacement, crop) in enumerate(zip(displacements, images, strict=True)):
            x = 12 + (position % columns) * (w + 12)
            y = 12 + (position // columns) * (h + 36)
            draw.text((x, y), "Original (0 px added)" if displacement == 0 else f"Motion right: {displacement} px", fill="black")
            sheet.paste(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), (x, y + 24))
        sheet.save(args.output / f"roi{index + 1}_comparison_4x.png")
        roi_cards = ''.join(
            f'<figure><a href="motion_right_{d}px.png"><img class="pixels" src="roi{index+1}_{d}px_4x.png"></a><figcaption>{d} px</figcaption></figure>'
            for d in displacements
        )
        roi_html.append(f'<h2>局部 {index+1}：原图坐标 {args.roi[index]}，4倍最近邻放大</h2><div class="grid">{roi_cards}</div>')
    (args.output / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    page = (
        f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{label}像素水平拖影模拟</title>'
        '<style>body{font-family:sans-serif;margin:24px;background:#eee}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}'
        'figure{margin:0}img{width:100%}.pixels{image-rendering:pixelated}figcaption{padding:8px}p{line-height:1.6}</style>'
        f'<h1>向右 {label} 像素拖影</h1>'
        f'<p>源图：{html.escape(str(args.image))}，{width}×{height}。0 px为原图，所有位移均以原始分辨率像素计。</p>'
        '<p>模型：曝光期间匀速向右移动0到L像素，对线性插值后的图像积分。1 px也有模糊；质心同时向右移动L/2像素。'
        '没有提亮、锐化或去噪。PNG本身已有模糊与噪声，结果仅是新增图像域拖影的近似；整帧背景和噪声也参与卷积。</p>'
        f'<p>点击图片打开{width}×{height}完整结果，按100%查看。全图缩小后像素级差异容易被掩盖。</p>'
        + ''.join(roi_html) + '<h2>完整图像概览</h2><div class="grid">' + ''.join(cards) + '</div></html>'
    )
    (args.output / "index.html").write_text(page, encoding="utf-8")
    if hashlib.sha256(args.image.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("source changed during simulation")
    print(args.output / "index.html")


if __name__ == "__main__":
    main()
