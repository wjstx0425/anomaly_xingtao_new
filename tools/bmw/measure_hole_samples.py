"""Measure a small real-image selection without inventing acceptance limits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from bmw_inspection.checks.hole import inspect_hole, load_config


def write_json(path: Path, payload: dict) -> None:
    """Save readable finite JSON."""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_image(path: Path, image: np.ndarray) -> None:
    """Fail explicitly when an evidence image cannot be saved."""
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot save {path}")


def main() -> None:
    """Use the first selected image as an exploratory mask, never a nominal standard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    selection = json.loads(args.manifest.read_text(encoding="utf-8"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    samples = selection["samples"]
    reference = cv2.imread(str(root / samples[selection["reference_index"]]["path"]))
    if reference is None:
        raise ValueError("Unreadable reference image")
    x1, y1, x2, y2 = selection["roi_xyxy"]
    gray = cv2.cvtColor(reference[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    threshold, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    write_image(output / "reference_mask.png", mask)
    config = {
        "schema_version": 1, "status": "diagnostic", "part_id": selection["part_id"], "hand": "left",
        "feature_id": selection["feature_id"], "view_id": "front", "source_channel": "fused",
        "pixel_channel": "gray", "reference_image_size": [reference.shape[1], reference.shape[0]],
        "coordinate_space": "full_image", "roi_xyxy": selection["roi_xyxy"],
        "expected_mask": "reference_mask.png",
        "registration": {"mode": "fixture_fixed", "validated": False,
                         "evidence": "Fixed full-image coordinates only; independent fixture repeatability unverified"},
        "association": {"exclusive_search_confirmed": True,
                        "evidence": "Visual review of six selected upper-front crops: ROI excludes lower hole and background"},
        "segmentation": {"threshold": threshold, "polarity": "dark", "stability_delta": 1,
                         "max_changed_fraction": None},
        "limits": None,
        "validation_evidence": "Exploratory reference Otsu mask from sample 1; not approved nominal geometry. "
                               "Delta=1 probes adjacent uint8 threshold levels; no quality or verdict limits.",
    }
    write_json(output / "diagnostic_config.json", config)
    config = load_config(output / "diagnostic_config.json")
    rows, tiles = [], []
    for index, source in enumerate(samples, 1):
        path = root / source["path"]
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        observation = {"raw_unmasked": True, "fixture_verified": False, "observable": False,
                       "evidence": "Direct clean-raw fused PNG; no model ignore mask applied. "
                                   "Dark region is visible but rear fixture/background and physical openness unconfirmed."}
        result = inspect_hole(image, config, part_id=selection["part_id"], view_id=source["view_id"],
                              source_channel=source["source_channel"], source_kind=source["source_kind"],
                              observation=observation)
        result.payload["lineage"] = source
        result.payload["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        result.payload["reference_sample"] = samples[selection["reference_index"]]
        result.save(output / f"sample_{index:02}", source=str(path))
        rows.append({"index": index, **source, "status": result.payload["status"],
                     "metrics": result.payload["metrics"], "evidence": f"sample_{index:02}/result.json"})
        tile = cv2.hconcat([result.images["roi_original"], result.images["overlay"]])
        banner = np.zeros((36, tile.shape[1], 3), dtype=np.uint8)
        cv2.putText(banner, f"{index}: {source['dataset_label']} {source['session_id'][:8]} {source['group_id']}",
                    (4, 22), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
        tiles.append(cv2.vconcat([banner, tile]))
    write_image(output / "contact_sheet.jpg", cv2.vconcat([cv2.hconcat(tiles[:3]), cv2.hconcat(tiles[3:])]))
    overview = reference.copy()
    cv2.rectangle(overview, (x1, y1), (x2-1, y2-1), (0, 255, 255), 8)
    cv2.putText(overview, "B4 candidate ROI", (x1-180, y1-30), 0, 2, (0, 255, 255), 4)
    write_image(output / "target_overview.jpg", cv2.resize(overview, (1006, 759)))
    write_json(output / "summary.json", {"synthesized_test_data": False, "diagnostic_only": True,
                                         "acceptance_verified": False, "reference_threshold_otsu": threshold,
                                         "selection": selection, "samples": rows})
    for row in rows:
        print(row["index"], row["status"], row["metrics"])


if __name__ == "__main__":
    main()
