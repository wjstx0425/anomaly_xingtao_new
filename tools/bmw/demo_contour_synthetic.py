"""Generate and run a reproducible synthetic contour demonstration (not BMW validation)."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from bmw_inspection.cli.contour import main as contour_main
from bmw_inspection.checks.contour_compare.contracts import read_json, write_json
from bmw_inspection.checks.contour_compare.reference import atomic_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    with atomic_directory(args.output_dir) as out:
        config = read_json(repository / "configs/bmw/checks/contour/left_front.draft.json")
        config["image"].update(width=220, height=220)
        config["part_roi_xyxy"] = [10, 10, 210, 210]
        config["registration"]["anchor_rois_xyxy"] = [[55, 55, 85, 85], [135, 135, 165, 165]]
        config["extraction"].update(search_inward_px=15, search_outward_px=15)
        config["profile_id"] = "synthetic_rectangle_two_holes"
        image = np.full((220, 220, 3), 20, np.uint8)
        mask = np.zeros((220, 220), np.uint8)
        image[30:191, 30:191] = 200
        mask[30:191, 30:191] = 255
        for center in ((70, 70), (150, 150)):
            cv2.circle(image, center, 8, (20, 20, 20), -1)
        cv2.imwrite(str(out / "reference.png"), image)
        cv2.imwrite(str(out / "mask.png"), mask)
        write_json(out / "draft.json", config)
        if contour_main(["teach", "--image", str(out / "reference.png"), "--draft-config", str(out / "draft.json"), "--mask", str(out / "mask.png"), "--confirm-reference", "--output-dir", str(out / "reference")]):
            raise RuntimeError("synthetic teaching failed")
        rows = []
        edits = {"self": None, "left_notch": (slice(85, 115), slice(30, 42)), "right_notch": (slice(85, 115), slice(179, 191)), "top_notch": (slice(30, 42), slice(90, 120)), "bottom_notch": (slice(179, 191), slice(90, 120))}
        for name, slices in edits.items():
            test = image.copy()
            if slices:
                test[slices] = 20
            cv2.imwrite(str(out / f"{name}.png"), test)
            rows.append(f"{name},,,synthetic,left,front,{name}.png,fused,{'normal' if slices is None else 'defect'}")
        (out / "manifest.csv").write_text("sample_id,physical_part_id,placement_group,split,hand,view_id,image_path,channel,ground_truth\n" + "\n".join(rows) + "\n")
        if contour_main(["evaluate", "--config", str(out / "reference/recipe.json"), "--manifest", str(out / "manifest.csv"), "--output-dir", str(out / "evaluation")]):
            raise RuntimeError("synthetic evaluation failed")
        # Atomic staging path is transient; store stable relative source identities.
        for name in edits:
            path = out / "evaluation" / name / "result.json"
            result = read_json(path)
            result["image_source"] = f"../../{name}.png"
            result["data_kind"] = "synthetic"
            write_json(path, result)
    print(f"Synthetic evidence saved to {args.output_dir}")


if __name__ == "__main__":
    main()
