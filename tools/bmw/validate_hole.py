# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Exercise the single-hole CLI with synthetic images, never production limits."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def _write_json(path: Path, value: dict | list) -> None:
    """Write readable evidence with Chinese text preserved."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    """Create a new synthetic evidence directory and verify all four verdicts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New directory; existing paths are refused")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    mask = np.zeros((60, 60), dtype=np.uint8)
    cv2.circle(mask, (30, 30), 15, 255, -1)
    if not cv2.imwrite(str(inputs / "expected.png"), mask):
        raise OSError("Could not save synthetic expected mask")
    config = {
        "schema_version": 1, "status": "ready", "part_id": "SYNTHETIC_ONLY", "hand": "left",
        "feature_id": "hole_1", "view_id": "front_left", "source_channel": "fused", "pixel_channel": "gray",
        "reference_image_size": [160, 100], "coordinate_space": "full_image",
        "roi_xyxy": [20, 20, 80, 80], "expected_mask": "expected.png",
        "registration": {"mode": "fixture_fixed", "validated": True, "evidence": "synthetic fixed coordinates"},
        "association": {"exclusive_search_confirmed": True, "evidence": "synthetic isolated ROI"},
        "segmentation": {"threshold": 128, "polarity": "bright", "stability_delta": 10,
                         "max_changed_fraction": 0.05},
        "limits": {"pass_min_open_fraction": 0.95, "ng_max_open_fraction": 0.6,
                   "pass_max_area_difference": 0.05, "ng_min_area_difference": 0.4,
                   "pass_max_shape_difference": 0.05, "ng_min_shape_difference": 0.4,
                   "pass_max_position_px": 1.0, "ng_min_position_px": 6.0},
        "validation_evidence": "Synthetic software test only; no real BMW image or acceptance standard",
    }
    _write_json(inputs / "config.json", config)
    normal = np.full((100, 160, 3), 20, dtype=np.uint8)
    normal[20:80, 20:80][mask > 0] = 230
    absent = np.full_like(normal, 20)
    partial = normal.copy()
    partial[:, :50] = 20
    neighbor = absent.copy()
    cv2.circle(neighbor, (115, 50), 15, (230, 230, 230), -1)
    unstable = normal.copy()
    unstable[unstable == 230] = 130
    cases = [
        ("normal", normal, {}, "PASS", 0),
        ("not_open", absent, {}, "NG", 1),
        ("partial_blockage", partial, {}, "NG", 1),
        ("neighbor_only", neighbor, {}, "NG", 1),
        ("threshold_instability", unstable, {}, "REVIEW", 3),
        ("fixture_occluded", normal, {"observable": False}, "REVIEW", 3),
        ("fixture_unverified", normal, {"fixture_verified": False}, "REVIEW", 3),
        ("masked_input", normal, {"raw_unmasked": False}, "ERROR", 2),
    ]
    rows = []
    for name, image, observation_changes, expected, expected_code in cases:
        image_path = inputs / f"{name}.png"
        if not cv2.imwrite(str(image_path), image):
            raise OSError(f"Could not save {image_path}")
        observation = {"raw_unmasked": True, "fixture_verified": True, "observable": True,
                       "evidence": f"Synthetic scenario: {name}", **observation_changes}
        observation_path = inputs / f"{name}_observation.json"
        _write_json(observation_path, observation)
        command = [
            sys.executable, "-m", "bmw_inspection.cli.check_hole", "--config", str(inputs / "config.json"),
            "--image", str(image_path), "--part-id", "SYNTHETIC_ONLY", "--view-id", "front_left",
            "--source-channel", "fused", "--source-kind", "fused_only", "--observation", str(observation_path),
            "--output", str(output / name), "--synthesized-test-data",
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        (output / f"{name}.stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (output / f"{name}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
        result_path = output / name / "result.json"
        payload = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
        actual = payload.get("status")
        rows.append({"case": name, "expected": expected, "actual": actual, "returncode": completed.returncode,
                     "verified": actual == expected and completed.returncode == expected_code,
                     "evidence": str(result_path.relative_to(output)), "command": command})
        print(f"{name}: {actual}, exit={completed.returncode}, expected={expected}", flush=True)
    summary = {
        "synthesized_test_data": True, "production_acceptance_verified": False,
        "note": "All images, ROI, masks and limits are synthetic software fixtures. Real acceptance remains pending.",
        "cases": rows, "all_verified": all(row["verified"] for row in rows),
    }
    _write_json(output / "summary.json", summary)
    if not summary["all_verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
