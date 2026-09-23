# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Measure warmed standalone V2 stamp reading, excluding file I/O and model loading."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from dataclasses import replace
from pathlib import Path


def main() -> None:
    """Benchmark manifest images and preserve per-call outputs for comparison."""
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", help="Repeatable; defaults to reviewed 60+24 sets")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file")
    args = parser.parse_args()
    if args.threads < 1 or args.repetitions < 1:
        parser.error("threads and repetitions must be positive")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    import cv2

    from bmw_inspection.checks import StampReader, StampReaderConfig

    manifests = args.manifest or [
        root / f"docs/bmw/stamp_reading_{name}_20260909_manifest.json" for name in ("validation", "holdout")
    ]
    samples = [
        {**sample, "set": path.stem}
        for path in manifests for sample in json.loads(path.read_text(encoding="utf-8"))["samples"]
    ]
    if not samples:
        parser.error("Manifests contain no images")
    readers = {
        hand: StampReader(replace(
            StampReaderConfig.from_json(root / f"configs/bmw/checks/stamp_read_{hand}_v2.json"),
            intra_op_num_threads=args.threads, inter_op_num_threads=2,
        )) for hand in sorted({sample["hand"] for sample in samples})
    }
    for reader in readers.values():
        reader.initialize()

    def read_image(sample: dict):
        image = cv2.imread(str(root / sample["source"]), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot decode {sample['source']}")
        return image

    # Warm every configured reader twice before measurements.
    warmup_count = 0
    for hand, reader in readers.items():
        sample = next(sample for sample in samples if sample["hand"] == hand)
        for _ in range(2):
            reader.read(read_image(sample))
            warmup_count += 1
    records = []
    for repetition in range(args.repetitions):
        for sample in (samples if repetition % 2 == 0 else reversed(samples)):
            image = read_image(sample)
            start = time.perf_counter()
            result = readers[sample["hand"]].read(
                image, capture_id=sample.get("capture_id"), inspection_id=sample.get("inspection_id"),
                source_kind=sample.get("source_kind", "unknown"),
            )
            wall_ms = (time.perf_counter() - start) * 1000
            records.append({
                "set": sample["set"], "index": sample["index"], "repetition": repetition,
                "reader_wall_ms": wall_ms, "payload": result.to_dict(),
            })
        print(f"Completed pass {repetition + 1}", flush=True)
    summary = {}
    for metric, values in (
        ("ocr_call_ms", [row["payload"]["elapsed_ms"] for row in records]),
        ("reader_wall_ms", [row["reader_wall_ms"] for row in records]),
    ):
        values.sort()
        summary[metric] = {
            "mean": statistics.mean(values), "median": statistics.median(values),
            "p95": values[math.ceil(.95 * len(values)) - 1], "min": values[0], "max": values[-1],
        }
    report = {
        "threads": args.threads, "inter_op_threads": 2, "python": platform.python_version(),
        "images": len(samples), "repetitions": args.repetitions, "warmup_calls": warmup_count,
        "timing_scope": "CPU sequential; excludes loading, image decoding, serialization, saving and capture",
        "summary": summary, "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
