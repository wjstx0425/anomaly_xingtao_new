#!/usr/bin/env python3
"""Compare V3 scores with operator-selected EfficientAD ignore masks on saved maps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bmw_inspection.lab.efficientad_ignore_mask import load_ignore_mask_asset, mask_anomaly_map  # noqa: E402
from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER  # noqa: E402


DEFAULT_SOURCE = REPO_ROOT / "results/bmw_efficientad_mask_ab/bmw_v3_representative_masked_map_ab_v3"
DEFAULT_MASK_INDEX = (
    REPO_ROOT / "results/bmw_efficientad_manual_ignore_masks/bmw_right_manual_ignore_v1/index.json"
)
DEFAULT_ROI_CONFIG = REPO_ROOT / "configs/bmw/rois/bmw_right_hdr_eight_view_v1.json"
DEFAULT_OUTPUT = REPO_ROOT / "results/bmw_efficientad_manual_ignore_ab/bmw_v3_vs_v4_representative_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--mask-index", type=Path, default=DEFAULT_MASK_INDEX)
    parser.add_argument("--roi-config", type=Path, default=DEFAULT_ROI_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = args.source.expanduser().resolve()
    mask_index = args.mask_index.expanduser().resolve()
    roi_config = args.roi_config.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        print(f"输出目录已存在，拒绝覆盖：{output}", file=sys.stderr)
        return 2
    try:
        mask_payload = json.loads(mask_index.read_text(encoding="utf-8"))
        shapes = {
            view: (
                int(mask_payload["views"][view]["roi_height"]),
                int(mask_payload["views"][view]["roi_width"]),
            )
            for view in VIEW_ORDER
        }
        asset = load_ignore_mask_asset(
            mask_index,
            expected_views=VIEW_ORDER,
            expected_roi_config_sha256=_sha256(roi_config),
            expected_shapes=shapes,
        )
        with (source / "per_image_scores.csv").open("r", encoding="utf-8", newline="") as stream:
            source_rows = list(csv.DictReader(stream))
        rows: list[dict[str, object]] = []
        capture_v3_ng: Counter[str] = Counter()
        capture_v4_ng: Counter[str] = Counter()
        for source_row in source_rows:
            capture_id = source_row["capture_id"]
            view = source_row["view"]
            threshold = float(source_row["deployment_threshold"])
            v3_score = float(source_row["rerun_pred_score"])
            manual_mask_active = bool(np.any(asset.masks[view]))
            raw_map_max: float | None = None
            ignored_map_pixels = 0
            if manual_mask_active:
                map_path = source / "evidence" / capture_id / view / "raw_anomaly_map_unmasked.npy"
                masked = mask_anomaly_map(np.load(map_path, allow_pickle=False), asset.masks[view])
                v4_score = masked.score
                raw_map_max = masked.raw_max
                ignored_map_pixels = masked.ignored_map_pixel_count
                score_source = "manual_ignore_masked_anomaly_map_max"
            else:
                v4_score = v3_score
                score_source = "pred_score"
            v3_status = "NG" if v3_score >= threshold else "PASS"
            v4_status = "NG" if v4_score >= threshold else "PASS"
            capture_v3_ng[capture_id] += int(v3_status == "NG")
            capture_v4_ng[capture_id] += int(v4_status == "NG")
            rows.append(
                {
                    "capture_id": capture_id,
                    "view": view,
                    "manual_mask_active": manual_mask_active,
                    "score_source": score_source,
                    "threshold": threshold,
                    "v3_score": v3_score,
                    "v4_score": v4_score,
                    "raw_anomaly_map_max": raw_map_max,
                    "ignored_roi_pixel_count": int(np.count_nonzero(asset.masks[view])),
                    "ignored_map_pixel_count": ignored_map_pixels,
                    "v3_status": v3_status,
                    "v4_status": v4_status,
                    "decision_changed": v3_status != v4_status,
                }
            )
        destination_parent = output.parent
        destination_parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=destination_parent))
        try:
            fields = list(rows[0])
            with (staging / "per_image_scores.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            captures = sorted(capture_v3_ng)
            report = {
                "schema_version": "bmw.efficientad_manual_ignore_ab/1.0",
                "scope": "saved_raw_maps_only_no_new_inference",
                "source_report": str((source / "report.json").resolve()),
                "source_report_sha256": _sha256(source / "report.json"),
                "ignore_mask_index": str(mask_index),
                "ignore_mask_index_sha256": asset.index_sha256,
                "public_roi_config": str(roi_config),
                "public_roi_config_sha256": asset.public_roi_config_sha256,
                "image_count": len(rows),
                "capture_count": len(captures),
                "masked_image_count": sum(bool(row["manual_mask_active"]) for row in rows),
                "decision_change_count": sum(bool(row["decision_changed"]) for row in rows),
                "changes": [row for row in rows if row["decision_changed"]],
                "capture_efficientad_ng_count": {
                    capture: {"v3": capture_v3_ng[capture], "v4": capture_v4_ng[capture]}
                    for capture in captures
                },
                "threshold_note": (
                    "Masked-map max is compared with the existing V3 numeric threshold only as a lab candidate; "
                    "the score domain must be calibrated before production use."
                ),
            }
            (staging / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(staging, output)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"BMW EfficientAD 手动 mask A/B 失败：{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

