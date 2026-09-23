# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Replay a reviewed sample manifest through the standalone stamp reader."""

from __future__ import annotations

import argparse
import html
import json
import statistics
from collections import Counter
from pathlib import Path


def _alphanumeric(text: str | None) -> str:
    """Return an evaluation-only comparison key without rewriting recognition."""
    return "".join(char for char in (text or "") if char.isascii() and char.isalnum())


def main() -> None:
    """Save per-image OCR evidence and comparisons without product decisions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New result directory")
    parser.add_argument("--config-version", choices=("v1", "v2"), default="v1")
    args = parser.parse_args()

    import cv2

    from bmw_inspection.checks import StampReader, StampReaderConfig

    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    readers = {
        hand: StampReader(StampReaderConfig.from_json(
            root / f"configs/bmw/checks/stamp_read_{hand}_{args.config_version}.json",
        ))
        for hand in ("left", "right")
    }
    rows = []
    for sample in manifest["samples"]:
        source = root / sample["source"]
        image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not read {source}")
        result = readers[sample["hand"]].read(
            image, source_kind=sample.get("source_kind", "unknown"),
            capture_id=sample.get("capture_id"), inspection_id=sample.get("inspection_id"),
        )
        payload = result.to_dict()
        name = f"sample_{sample['index']:02d}"
        result.save(args.output / name, source=str(source))
        reference = sample["visual_reference"]
        code = payload["raw_code"]
        normalized = payload["normalized_code"]
        assessable = reference["legibility"] == "clear"
        # Metrics only. These transforms never change the reader output or state.
        exact = code == reference["text"] if assessable else None
        alnum_match = _alphanumeric(code) == _alphanumeric(reference["text"]) if assessable else None
        expected = _alphanumeric(reference["text"]) if args.config_version == "v2" else reference["text"]
        normalized_match = normalized == expected if assessable else None
        row = {
            **sample, "output": name, "raw_code": code, "state": payload["state"],
            "reasons": payload["reasons"], "lines": payload["lines"],
            "code_line_indices": payload["code_line_indices"],
            "code_score": payload["code_score"],
            "code_width_fraction": payload["code_width_fraction"],
            "elapsed_ms": payload["elapsed_ms"], "exact_match": exact,
            "alphanumeric_match": alnum_match,
            "normalized_code": normalized, "normalized_match": normalized_match,
            "selected_source": payload.get("selected_source", "detected_code"),
            "removed_characters": payload["removed_characters"],
        }
        rows.append(row)
        print(f"{sample['index']:02d} {row['state']:8} {code!r} {row['reasons']}", flush=True)

    clear = [row for row in rows if row["exact_match"] is not None]
    timings = sorted(row["elapsed_ms"] for row in rows)
    summary = {
        "sample_count": len(rows), "clear_visual_reference_count": len(clear),
        "config_version": args.config_version,
        "uncertain_visual_reference_indices": [row["index"] for row in rows if row["exact_match"] is None],
        "states": dict(Counter(row["state"] for row in rows)),
        "raw_exact_match_count": sum(row["exact_match"] for row in clear),
        "alphanumeric_match_count": sum(row["alphanumeric_match"] for row in clear),
        "normalized_match_count": sum(row["normalized_match"] for row in clear),
        "selected_sources": dict(Counter(row["selected_source"] for row in rows)),
        "no_single_code_count": sum(row["raw_code"] is None for row in rows),
        "readable_raw_mismatch_indices": [
            row["index"] for row in clear if row["state"] == "readable" and not row["exact_match"]
        ],
        "readable_alphanumeric_mismatch_indices": [
            row["index"] for row in clear if row["state"] == "readable" and not row["alphanumeric_match"]
        ],
        "ocr_call_ms": {
            "min": min(timings), "median": statistics.median(timings),
            "p95_nearest_rank": timings[max(0, (95 * len(timings) + 99) // 100 - 1)], "max": max(timings),
        },
        "timing_scope": "OCR call only; excludes engine loading, file I/O, cropping, capture, and saving",
        "reference_source": manifest["reference_source"],
        "business_rules_evaluated": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    cards = []
    for row in rows:
        text = html.escape(str(row["raw_code"]))
        normalized = html.escape(str(row["normalized_code"]))
        reference = html.escape(str(row["visual_reference"]["text"]))
        reason = html.escape(", ".join(row["reasons"]))
        source = html.escape(row["source"])
        cards.append(
            f'<article><h2>{row["index"]:02d} {row["hand"]} / {row["label"]}</h2>'
            f'<p>Code: <strong>{normalized}</strong> | {row["state"]}</p>'
            f'<p>Raw OCR: {text} | {row["selected_source"]}</p><p>{reason}</p>'
            f'<p>Visual reference: {reference} ({row["visual_reference"]["legibility"]})</p>'
            f'<a href="{row["output"]}/result.json">Raw JSON</a>'
            f'<img loading="lazy" src="{row["output"]}/overlay.png" alt="Stamp ROI with OCR boxes">'
            f'<details><summary>Source image</summary><p>{source}</p></details></article>'
        )
    document = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>BMW stamp reading review</title>'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<style>body{font:16px sans-serif;margin:24px;background:#f4f4f4;color:#222}'
        'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:20px}'
        'article{padding:16px;background:white}img{display:block;width:100%;margin-top:12px}'
        'p{overflow-wrap:anywhere}h2{font-size:18px}</style>'
        f'<h1>BMW back stamp reading: {len(rows)}-image review ({args.config_version})</h1>'
        '<p>Raw OCR, without business rules. Visual references are assistant transcriptions, not customer labels.</p>'
        '<p>Blue: configured code region. Red: candidate code boxes. Green: other OCR boxes.</p>'
        '<main>' + "\n".join(cards) + '</main></html>'
    )
    (args.output / "review.html").write_text(document, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
