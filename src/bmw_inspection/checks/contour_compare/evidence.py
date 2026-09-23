"""Atomic image and numeric evidence for full-perimeter observations."""
from __future__ import annotations

import csv
import time

import cv2
import numpy as np

from .contracts import json_ready, write_json
from .reference import atomic_directory, save_image


def draw_segments(image, points, valid, color, *, closed=False):
    """Draw only real contiguous observations; no lines across unknown gaps."""
    for i in range(len(points) if closed else max(0, len(points)-1)):
        j = (i+1) % len(points)
        if valid[i] and valid[j] and np.isfinite(points[[i, j]]).all():
            cv2.line(image, tuple(np.rint(points[i]).astype(int)), tuple(np.rint(points[j]).astype(int)), color, 1, cv2.LINE_AA)


def write_evidence(image, reference, observation, result, output_dir):
    """Persist complete evidence before publishing the run directory."""
    from .geometry import transform_points
    save_start = time.perf_counter()
    with atomic_directory(output_dir) as tmp:
        ref_overlay = reference["image"][:, :, :3].copy()
        draw_segments(ref_overlay, reference["dense_xy"], np.ones(len(reference["dense_xy"]), bool), (0, 200, 0), closed=True)
        save_image(tmp / "reference_overlay.png", ref_overlay)
        test_overlay = image[:, :, :3].copy()
        comparison = ref_overlay.copy()
        reg = observation.get("registration", {})
        matrix = reg.get("T_reference_to_test")
        points = observation["sample_xy"]
        valid = np.asarray(result["valid"], dtype=bool)
        dense = observation.get("dense_xy", np.empty((0, 2)))
        dense_valid = observation.get("dense_valid", np.zeros(len(dense), bool))
        if matrix is not None:
            closed = bool(len(dense) and np.isfinite(dense).all())
            draw_segments(test_overlay, transform_points(dense, np.asarray(matrix)), dense_valid, (255, 200, 0), closed=closed)
            draw_segments(comparison, dense, dense_valid, (255, 200, 0), closed=closed)
        for p in reference["sample_xy"][~valid]:
            cv2.circle(comparison, tuple(np.rint(p).astype(int)), 2, (0, 140, 255), -1)
            if matrix is not None:
                p_test = transform_points(np.asarray([p]), np.asarray(matrix))[0]
                cv2.circle(test_overlay, tuple(np.rint(p_test).astype(int)), 2, (0, 140, 255), -1)
        if matrix is None:
            cv2.putText(test_overlay, "POSE UNKNOWN - NO MAPPING", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 140, 255), 1)
        for event in result.get("events", []):
            color = (0, 0, 255) if event["status"] == "NG" else (0, 140, 255)
            for canvas, key in ((comparison, "bbox_reference_xyxy"), (test_overlay, "bbox_test_xyxy")):
                box = event.get(key)
                if box is not None:
                    a, b = np.rint(box).astype(int).reshape(2, 2)
                    a -= 4
                    b += 4
                    cv2.rectangle(canvas, tuple(a), tuple(b), color, 2)
                    cv2.putText(canvas, event["event_id"], tuple(a), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
        save_image(tmp / "test_outline_overlay.png", test_overlay)
        save_image(tmp / "comparison_overlay.png", comparison)
        if "current_image_edge_xy" in observation and matrix is not None:
            # Display actual image-edge observations separately from the dense
            # segmentation and the calibrated comparison coordinate convention.
            measured_overlay = image[:, :, :3].copy()
            raw_points = transform_points(np.asarray(observation["current_image_edge_xy"]), np.asarray(matrix))
            draw_segments(measured_overlay, raw_points, valid, (255, 220, 0), closed=True)
            for point in reference["sample_xy"][~valid]:
                target = transform_points(np.asarray([point]), np.asarray(matrix))[0]
                cv2.circle(measured_overlay, tuple(np.rint(target).astype(int)), 2, (0, 140, 255), -1)
            save_image(tmp / "measured_outline_overlay.png", measured_overlay)
        np.savez_compressed(tmp / "test_segments.npz", dense_xy=dense, dense_valid=dense_valid, sample_xy=points, valid=valid, measurement_starts_xy=result["test_segment_starts_xy"], measurement_ends_xy=result["test_segment_ends_xy"])
        fields = ["arc_id", "sample_id", "s_px", "cell_length_px", "ref_x", "ref_y", "test_x_ref", "test_y_ref", "valid", "invalid_reason", "normal_offset_u_px", "signed_test_to_ref_px", "signed_ref_to_test_px"]
        with (tmp / "contour_samples.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for i, p in enumerate(reference["sample_xy"]):
                row = dict(zip(fields[:8], [reference["arc_id"][i], i, reference["s_px"][i], reference["cell_length_px"][i], *p, *points[i]]))
                row.update(valid=bool(valid[i]), invalid_reason=result["invalid_reason"][i], normal_offset_u_px=result["normal_offset_u_px"][i], signed_test_to_ref_px=result["signed_test_to_ref_px"][i], signed_ref_to_test_px=result["signed_ref_to_test_px"][i])
                writer.writerow(json_ready(row))
        distance_keys = ("signed_test_to_ref_px", "signed_ref_to_test_px", "test_to_ref_px", "ref_to_test_px")
        distances = {k: result[k] for k in distance_keys}
        write_json(tmp / "distance_samples.json", distances)
        write_json(tmp / "events.json", result.get("events", []))
        write_json(tmp / "unconfirmed_coarse_events.json", result.get("unconfirmed_coarse_events", []))
        write_json(tmp / "extraction_diagnostics.json", {k: v for k, v in observation.items() if k not in ("sample_xy", "dense_xy", "dense_valid", "valid", "mask", "registration")})
        profile = np.full((400, 1200, 3), 255, np.uint8)
        offsets = np.asarray(observation.get("normal_offset_u_px", np.full(len(valid), np.nan)))
        good = valid & np.isfinite(offsets)
        max_offset = max(1, float(np.max(np.abs(offsets[good]))) if good.any() else 1)
        x = 40 + reference["s_px"] / np.sum(reference["cell_length_px"]) * 1120
        y = 190 - offsets / max_offset * 140
        cv2.line(profile, (40, 190), (1160, 190), (180, 180, 180), 1)
        for i in np.flatnonzero(good):
            cv2.circle(profile, (int(x[i]), int(y[i])), 1, (220, 80, 20), -1)
        cv2.putText(profile, f"Reference arc length (px); normal offset range +/-{max_offset:.2f} px", (30, 365), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1)
        cv2.putText(profile, "Orange = unknown reference arc; blue = observed offset", (30, 390), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
        for i in np.flatnonzero(~valid):
            cv2.line(profile, (int(x[i]), 320), (int(x[i]), 330), (0, 140, 255), 1)
        save_image(tmp / "displacement_profile.png", profile)
        # Measurement arrays belong in evidence, not mutable/shared result images.
        result = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
        result["timing_ms"] = dict(result.get("timing_ms", {}), evidence_save_before_result=(time.perf_counter()-save_start)*1000)
        result["evidence"] = {p.stem: p.name for p in tmp.iterdir()}
        write_json(tmp / "result.json", result)
    return result
