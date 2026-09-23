"""Offline image metrics and a local exposure-comparison report."""

from __future__ import annotations

import csv
import html
import json
import tempfile
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np


_METRICS = ("mean", "std", "p01", "p99", "dark_pct", "clip_pct", "laplacian_variance")


def _measure(image: np.ndarray, roi: object) -> dict[str, float]:
    if roi is not None:
        if (
            not isinstance(roi, (list, tuple))
            or len(roi) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) for value in roi)
        ):
            raise ValueError("roi must contain four integer coordinates [x1, y1, x2, y2]")
        x1, y1, x2, y2 = roi
        height, width = image.shape[:2]
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"roi {roi} is outside image bounds {width}x{height} or empty")
        image = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    p01, p99 = np.percentile(gray, (1, 99))
    return {
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "p01": float(p01),
        "p99": float(p99),
        "dark_pct": float(np.mean(gray <= 5) * 100),
        "clip_pct": float(np.mean(np.any(image >= 250, axis=2)) * 100),
        "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }



def _comparison_gallery(records: list[dict]) -> str:
    groups: dict[tuple[str, str, int], list[dict]] = {}
    for record in records:
        key = (record["camera_serial"], record["view"], record["repeat"])
        groups.setdefault(key, []).append(record)
    sections = []
    for (serial, view, repeat), group in groups.items():
        singles = {
            record["exposures_us"][0]: record
            for record in group
            if record["kind"] == "single" and len(record["exposures_us"]) == 1
        }
        pairs: dict[tuple[float, float], dict[str, dict]] = {}
        for record in group:
            if record["kind"] in ("selective", "mertens") and len(record["exposures_us"]) == 2:
                pair = tuple(record["exposures_us"])
                pairs.setdefault(pair, {})[record["kind"]] = record
        for (short, long), fused in pairs.items():
            figures = []
            entries = (
                (f"短曝光 {short:g} μs", singles.get(short)),
                (f"长曝光 {long:g} μs", singles.get(long)),
                (f"selective · {short:g} / {long:g} μs", fused.get("selective")),
                (f"mertens · {short:g} / {long:g} μs", fused.get("mertens")),
            )
            for label, record in entries:
                caption = html.escape(label)
                if record is None:
                    figures.append(f"<figure><p>缺少图像</p><figcaption>{caption}</figcaption></figure>")
                    continue
                original = html.escape(quote(record["path"], safe="/"), quote=True)
                thumbnail = html.escape(quote(record["thumbnail"], safe="/"), quote=True)
                figures.append(
                    f'<figure><a href="{original}"><img src="{thumbnail}" alt="{caption}"></a>'
                    f'<figcaption>{caption}</figcaption></figure>'
                )
            title = html.escape(f"{serial} / {view} / 第 {repeat} 轮 · {short:g} / {long:g} μs")
            sections.append(f'<section><h3>{title}</h3><div class="comparison">{"".join(figures)}</div></section>')
    if not sections:
        return ""
    return "<h2>同组曝光与融合对比</h2>" + "".join(sections)


def write_report(output: Path, records: list[dict], metadata: dict) -> None:
    """Write diagnostic metrics, bounded thumbnails and an escaped HTML report.

    Image paths must be relative to ``output``. ROI coordinates are pixel-based,
    with exclusive right and bottom edges, and apply only to metrics. Images are
    decoded without bit-depth conversion; this report accepts uint8 BGR inputs.
    Original image files and the supplied records are never modified.
    """
    output = Path(output).resolve()
    prepared = []
    reserved = {output / name for name in ("metrics.csv", "report.json", "index.html")}
    for record in records:
        relative = Path(record["path"])
        source = (output / relative).resolve()
        if relative.is_absolute() or not source.is_relative_to(output) or source in reserved:
            raise ValueError("record path must reference an image inside the output directory")
        encoded = np.fromfile(source, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED) if encoded.size else None
        if image is None:
            raise ValueError(f"Cannot read image: {relative}")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Report requires uint8 BGR images: {relative}")
        enriched = {
            **record,
            "path": relative.as_posix(),
            "metrics": _measure(image, metadata.get("roi")),
        }
        height, width = image.shape[:2]
        scale = min(1.0, 480 / max(height, width))
        thumbnail = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        prepared.append((enriched, thumbnail))

    # A unique directory avoids colliding with any supplied original image paths.
    thumbnail_dir = Path(tempfile.mkdtemp(prefix="report_thumbnails_", dir=output))
    enriched_records = []
    rows = []
    for index, (record, thumbnail) in enumerate(prepared):
        thumbnail_path = thumbnail_dir / f"{index:05d}.jpg"
        ok, encoded_thumbnail = cv2.imencode(".jpg", thumbnail)
        if not ok:
            raise OSError(f"Could not encode thumbnail: {thumbnail_path}")
        thumbnail_path.write_bytes(encoded_thumbnail.tobytes())
        record["thumbnail"] = thumbnail_path.relative_to(output).as_posix()
        enriched_records.append(record)
        label = html.escape(f'{record["camera_serial"]} / {record["view"]} / repeat {record["repeat"]} / {record["kind"]}')
        exposure = html.escape(", ".join(str(value) for value in record["exposures_us"]))
        original_url = html.escape(quote(record["path"], safe="/"), quote=True)
        thumbnail_url = html.escape(quote(record["thumbnail"], safe="/"), quote=True)
        cells = "".join(f"<td>{record['metrics'][metric]:.3f}</td>" for metric in _METRICS)
        rows.append(
            f'<tr><td><a href="{original_url}"><img src="{thumbnail_url}" alt="{label}"></a>'
            f'<br>{label}<br>曝光：{exposure} μs</td>{cells}</tr>'
        )
    columns = ["path", "camera_serial", "view", "repeat", "kind", "exposures_us", *_METRICS]
    with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in enriched_records:
            writer.writerow({**{key: record[key] for key in columns[:5]}, "exposures_us": json.dumps(record["exposures_us"]), **record["metrics"]})
    (output / "report.json").write_text(
        json.dumps({"metadata": metadata, "records": enriched_records}, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    headings = "".join(f"<th>{metric}</th>" for metric in _METRICS)
    page = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>曝光与 HDR 对比</title>'
        '<style>body{font-family:sans-serif;margin:24px}table{border-collapse:collapse}'
        'td,th{border:1px solid #ccc;padding:8px}img{max-width:480px;max-height:480px}'
        'pre{white-space:pre-wrap}td{vertical-align:top}'
        '.comparison{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}'
        '.comparison figure{margin:0}.comparison img{width:100%;height:auto}'
        '.comparison figcaption{overflow-wrap:anywhere}section{margin:24px 0}'
        '</style><h1>曝光与 HDR 对比</h1>'
        '<p>这些指标仅用于诊断，不作为自动排名或产品合格判定。请结合原图检查细节、噪声、光晕和运动重影。点击缩略图打开原图。</p>'
        '<p>mean / std：灰度均值与标准差；p01 / p99：灰度第 1 / 99 百分位；'
        'dark_pct：灰度 ≤ 5 的像素百分比；clip_pct：任一 BGR 通道 ≥ 250 的像素百分比；'
        'laplacian_variance：灰度拉普拉斯方差，噪声也可能使其升高。'
        'ROI 使用 [x1,y1,x2,y2]，右下边界不包含；指标应用 ROI，缩略图保留全图。</p>'
        f'<details><summary>采集与处理参数</summary><pre>{html.escape(json.dumps(metadata, ensure_ascii=False, indent=2))}</pre></details>'
        f'{_comparison_gallery(enriched_records)}<h2>逐图指标</h2>'
        f'<table><thead><tr><th>图像与曝光</th>{headings}</tr></thead><tbody>{"".join(rows)}</tbody></table></html>'
    )
    (output / "index.html").write_text(page, encoding="utf-8")
