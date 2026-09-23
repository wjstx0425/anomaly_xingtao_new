"""Bidirectional segment comparison, periodic events, and fixed coverage accounting."""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import cyclic_runs, point_to_segments, transform_points


def _inside(points: np.ndarray, contour: np.ndarray) -> np.ndarray:
    polygon = np.asarray(contour, np.float32).reshape(-1, 1, 2)
    return np.asarray([cv2.pointPolygonTest(polygon, tuple(map(float, p)), False) for p in points])


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values)[np.isfinite(values)]
    return {"max_px": float(np.max(values)) if len(values) else None,
            "p95_px": float(np.percentile(values, 95)) if len(values) else None}


def compare_outline(reference: dict, observation: dict, config: dict) -> dict:
    """Compare actual supported edges without bridging unknown reference cells."""
    ref = np.asarray(reference["sample_xy"], float)
    dense_ref = np.asarray(reference["dense_xy"], float)
    test = np.asarray(observation["sample_xy"], float)
    lengths = np.asarray(reference["cell_length_px"], float)
    required = np.asarray(reference["required_mask"], bool)
    arc_id = np.asarray(reference.get("arc_id", np.zeros(len(ref), int)))
    raw_valid = np.asarray(observation["valid"])
    normals = np.asarray(reference["inward_normal_xy"], float)
    if (ref.ndim != 2 or ref.shape[1] != 2 or len(ref) < 3 or test.shape != ref.shape
            or dense_ref.ndim != 2 or dense_ref.shape[1] != 2 or len(dense_ref) < 3
            or lengths.shape != (len(ref),) or required.shape != lengths.shape
            or raw_valid.shape != lengths.shape or arc_id.shape != lengths.shape
            or normals.shape != ref.shape):
        raise ValueError("Reference and observation sampling contracts do not match")
    if (not np.isfinite(ref).all() or not np.isfinite(dense_ref).all()
            or not np.isfinite(normals).all() or not np.isfinite(lengths).all()
            or np.any(lengths <= 0) or not required.any()
            or np.any(np.linalg.norm(normals, axis=1) < 1e-8)):
        raise ValueError("Finite reference coordinates, normals and positive required arc lengths are required")
    if np.isinf(test).any() or np.any(raw_valid.astype(bool) & ~np.isfinite(test).all(axis=1)):
        raise ValueError("Valid observation coordinates must be finite; unknown points may use NaN")
    valid = raw_valid.astype(bool).copy()
    reference_valid = np.asarray(reference.get("reference_valid", np.ones(len(ref), bool)), dtype=bool)
    if reference_valid.shape != valid.shape:
        raise ValueError("Reference validity shape does not match its arc cells")
    valid &= reference_valid
    settings = config.get("comparison", config)

    def positive(value: object, name: str) -> float:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
            raise ValueError(f"{name} must be a positive finite number")
        number = float(value)
        if not np.isfinite(number) or number <= 0:
            raise ValueError(f"{name} must be a positive finite number")
        return number

    inward = np.full(len(ref), positive(settings.get("default_inward_tolerance_px"), "default_inward_tolerance_px"))
    outward = np.full(len(ref), positive(settings.get("default_outward_tolerance_px"), "default_outward_tolerance_px"))
    minimum = np.full(len(ref), positive(settings.get("min_exceedance_arc_px"), "min_exceedance_arc_px"))
    if settings.get("merge_valid_gap_px", 0.) != 0.:
        raise ValueError("Only merge_valid_gap_px=0 is supported")
    if settings.get("short_large_peak_review_px") is not None:
        positive(settings["short_large_peak_review_px"], "short_large_peak_review_px")
    radius_value = settings.get("correspondence_neighborhood_cells", 8)
    if positive(radius_value, "correspondence_neighborhood_cells") != int(radius_value):
        raise ValueError("correspondence_neighborhood_cells must be an integer")
    seen = set()
    for override in settings.get("arc_overrides", []):
        identifier = override.get("arc_id")
        if identifier in seen or not np.any(arc_id == identifier):
            raise ValueError("Unknown or duplicate arc override ID")
        seen.add(identifier)
        mask = arc_id == identifier
        for key, array in (("inward_tolerance_px", inward), ("outward_tolerance_px", outward),
                           ("min_exceedance_arc_px", minimum)):
            if key in override:
                array[mask] = positive(override[key], key)
    registration = observation.get("registration", {"valid": True})
    if not registration.get("valid", False):
        valid[:] = False
    reason = np.asarray(observation.get("invalid_reason", [""] * len(ref)), dtype=object).copy()
    if reason.shape != lengths.shape:
        raise ValueError("invalid_reason must match the reference sampling")
    reason[~valid & (reason == "")] = "no_valid_edge_or_registration"
    # Only adjacent supported observations create measurement segments.
    segment_valid = valid & np.roll(valid, -1)
    # An isolated edge point has no observed incident curve segment. Comparing
    # it to a distant neighbouring segment manufactures a false displacement.
    incident = segment_valid | np.roll(segment_valid, 1)
    reason[valid & ~incident] = "isolated_observation_without_segment"
    valid &= incident
    segment_valid = valid & np.roll(valid, -1)
    starts, ends = test[segment_valid], np.roll(test, -1, axis=0)[segment_valid]
    distance_rt = np.full(len(ref), np.nan)
    distance_tr = np.full(len(ref), np.nan)
    signed_rt = np.full(len(ref), np.nan)
    signed_tr = np.full(len(ref), np.nan)
    offset = np.full(len(ref), np.nan)
    # Restrict the reference-to-test direction to the corresponding topology.
    # Global nearest-neighbour matching could use the other side of a narrow foot.
    radius = max(1, int(settings.get("correspondence_neighborhood_cells", 8)))
    best = np.full(len(ref), np.inf)
    for shift in range(-radius, radius + 1):
        ids = (np.arange(len(ref)) + shift) % len(ref)
        a, b = test[ids], test[(ids+1) % len(ref)]
        ab = b-a
        t = np.sum((ref-a)*ab, axis=1) / np.maximum(np.sum(ab*ab, axis=1), 1e-30)
        d = np.linalg.norm(ref - (a + np.clip(t, 0, 1)[:, None]*ab), axis=1)
        best = np.minimum(best, np.where(segment_valid[ids] & np.isfinite(d), d, np.inf))
    valid &= np.isfinite(best)
    segment_valid = valid & np.roll(valid, -1)
    starts, ends = test[segment_valid], np.roll(test, -1, axis=0)[segment_valid]
    reason[~valid & (reason == "")] = "no_supported_corresponding_segment"
    distance_rt[valid] = best[valid]
    if valid.any():
        distance_tr[valid], _, _ = point_to_segments(test[valid], dense_ref, np.roll(dense_ref, -1, axis=0))
        signed_tr[valid] = distance_tr[valid] * np.where(_inside(test[valid], dense_ref) >= 0, 1., -1.)
        offset[valid] = np.sum((test[valid] - ref[valid]) * np.asarray(reference["inward_normal_xy"])[valid], axis=1)
    dense_test = np.asarray(observation.get("dense_xy", test), float).reshape(-1, 2)
    dense_valid = np.asarray(observation.get("dense_valid", np.zeros(len(dense_test), bool)), bool)
    if dense_valid.shape != (len(dense_test),) or np.isinf(dense_test).any():
        raise ValueError("Dense observation validity must match finite coordinates or NaN separators")
    if np.any(dense_valid & ~np.isfinite(dense_test).all(axis=1)):
        raise ValueError("Valid dense observation coordinates must be finite")
    dense_valid = dense_valid & np.isfinite(dense_test).all(axis=1)
    closed_reliable = len(dense_test) >= 3 and dense_valid.all() and valid.all()
    if closed_reliable:
        signed_rt[valid] = distance_rt[valid] * np.where(_inside(ref[valid], dense_test) <= 0, 1., -1.)
    # Unknown inside/outside must not receive an invented sign. Use the stricter
    # directional tolerance for conservative candidate detection in that case.
    tolerance_rt = np.where(np.isfinite(signed_rt), np.where(signed_rt >= 0, inward, outward), np.minimum(inward, outward))
    tolerance_tr = np.where(signed_tr >= 0, inward, outward)
    exceed = valid & required & ((distance_rt > tolerance_rt) | (distance_tr > tolerance_tr))
    events = []
    inverse = registration.get("T_reference_to_test")

    def make_event(ids: np.ndarray, source: str, distances: np.ndarray, points: np.ndarray,
                   confirmed: bool, signs: np.ndarray | None = None) -> dict:
        peak_local = int(np.argmax(distances))
        point = points[peak_local]
        back = transform_points(points, inverse) if inverse is not None else None
        direction = "unknown"
        if signs is not None:
            known = signs[np.isfinite(signs) & (np.abs(signs) > 1e-8)]
            if len(known):
                direction = "mixed" if np.any(known > 0) and np.any(known < 0) else ("inward" if known[0] > 0 else "outward")
        return {"event_id": f"contour-{len(events)+1:04d}", "source": source,
                "status": "NG" if confirmed else "REVIEW", "direction": direction,
                "sample_ids": ids.tolist(), "arc_ids": np.unique(arc_id[ids]).tolist(),
                "exceedance_arc_px": float(lengths[np.unique(ids)[valid[np.unique(ids)]]].sum()),
                "event_span_px": float(lengths[np.unique(ids)[valid[np.unique(ids)]]].sum()), "merged_gap_px": 0.,
                "peak_distance_px": float(distances[peak_local]), "peak_xy_reference": point.tolist(),
                "bbox_reference_xyxy": np.r_[points.min(0), points.max(0)].tolist(),
                "bbox_test_xyxy": np.r_[back.min(0), back.max(0)].tolist() if back is not None else None,
                "quality": "supported_current_image_edge" if confirmed else "candidate_requires_review",
                "possible_cause": "visible_outer_contour_change; manufacturing_cause_uncertain"}

    for ids in cyclic_runs(exceed):
        peak = np.maximum(distance_rt[ids], distance_tr[ids])
        confirmed = float(lengths[ids].sum()) >= float(np.max(minimum[ids]))
        event = make_event(ids, "bidirectional", peak, test[ids], confirmed, signed_tr[ids])
        event.update(max_test_to_reference_px=float(distance_tr[ids].max()), max_reference_to_test_px=float(distance_rt[ids].max()))
        events.append(event)
    # Full coarse contour checks catch material far outside all local profiles.
    # Unsupported coarse points are review evidence, never certain defects.
    coarse_distance = np.full(len(dense_test), np.nan)
    finite = np.isfinite(dense_test).all(axis=1)
    coarse_nearest = np.full(len(dense_test), -1, int)
    coarse_ids = np.flatnonzero(finite)
    if len(coarse_ids):
        coarse_distance[finite], _, nearest = point_to_segments(dense_test[finite], ref, np.roll(ref, -1, axis=0))
        coarse_nearest[finite] = nearest
    coarse_exceed = np.zeros(len(dense_test), bool)
    coarse_eligible = np.asarray(observation.get("dense_candidate_eligible", np.ones(len(dense_test), bool)), bool)
    if coarse_eligible.shape != (len(dense_test),):
        raise ValueError("Dense eligibility must match the current dense contour")
    unconfirmed_coarse = []
    if len(coarse_ids):
        ids = coarse_nearest[finite]
        signs = _inside(dense_test[finite], dense_ref)
        limits = np.where(signs >= 0, inward[ids], outward[ids])
        coarse_exceed[finite] = (coarse_distance[finite] > limits) & required[ids]
        uncertain = coarse_exceed & ~coarse_eligible
        for run in (cyclic_runs(uncertain) if finite.all() else [r - 1 for r in cyclic_runs(np.r_[False, uncertain, False])]):
            points = dense_test[run]
            back = transform_points(points, inverse) if inverse is not None else None
            unconfirmed_coarse.append({"status": "REVIEW", "observation_state": "UNKNOWN",
                "reason": "automatic_reference_boundary_inconsistent_with_material_reference",
                "reference_sample_ids": np.unique(coarse_nearest[run]).tolist(),
                "peak_candidate_distance_px": float(coarse_distance[run].max()),
                "bbox_reference_xyxy": np.r_[points.min(0), points.max(0)].tolist(),
                "bbox_test_xyxy": np.r_[back.min(0), back.max(0)].tolist() if back is not None else None})
        coarse_exceed &= coarse_eligible
    # Attach coarse support to an existing event if it addresses the same cells.
    # NaN separators denote separate components, so array endpoints no longer
    # establish a shared periodic boundary between the first and last component.
    coarse_runs = cyclic_runs(coarse_exceed) if finite.all() else [
        run - 1 for run in cyclic_runs(np.r_[False, coarse_exceed, False])
    ]
    for run in coarse_runs:
        ids = np.unique(coarse_nearest[run])
        support = dense_valid[run].all() and reference_valid[ids].all() and registration.get("valid", False)
        confirmed = bool(support and lengths[ids].sum() >= np.max(minimum[ids]))
        event = make_event(ids, "whole_coarse_outline", coarse_distance[run], dense_test[run], confirmed,
                           coarse_distance[run] * np.where(_inside(dense_test[run], dense_ref) >= 0, 1., -1.))
        event["description"] = "Whole-image contour change including possible missing structure or profile-range escape"
        event["coarse_supported_reference_span_px"] = float(lengths[ids].sum()) if support else 0.
        event["confirmation_basis"] = "supported_whole_coarse_shape_change" if confirmed else "uncertain_or_short_coarse_change"
        events.append(event)
    # Fuse overlapping evidence only when the union remains one observed arc.
    # Unknown cells never connect two events, even if a coarse component spans
    # them. Keep each source record so fusion does not erase a stronger peak.
    fused_events = []
    for event in events:
        event["supports"] = [{k: v for k, v in event.items() if k != "supports"}]
        ids = np.asarray(event["sample_ids"], int)
        matched = []
        if valid[ids].all():
            union = set(ids.tolist())
            for index, previous in enumerate(fused_events):
                previous_ids = set(previous["sample_ids"])
                combined = union | previous_ids
                mask = np.zeros(len(ref), bool)
                mask[list(combined)] = True
                if union & previous_ids and valid[list(combined)].all() and len(cyclic_runs(mask)) == 1:
                    matched.append(index)
                    union = combined
            if matched:
                records = [fused_events[index] for index in matched] + [event]
                supports = [item for record in records for item in record["supports"]]
                winner = max(records, key=lambda item: item["peak_distance_px"])
                event = dict(winner)
                event["supports"] = supports
                event["sources"] = sorted({item["source"] for item in supports})
                event["source"] = "bidirectional" if "bidirectional" in event["sources"] else "whole_coarse_outline"
                event["status"] = "NG" if any(item["status"] == "NG" for item in records) else "REVIEW"
                event["quality"] = "supported_current_image_edge" if event["status"] == "NG" else "candidate_requires_review"
                directions = {item["direction"] for item in records} - {"unknown"}
                event["direction"] = next(iter(directions)) if len(directions) == 1 else ("mixed" if directions else "unknown")
                mask = np.zeros(len(ref), bool)
                mask[list(union)] = True
                event["sample_ids"] = cyclic_runs(mask)[0].tolist()
                event["arc_ids"] = np.unique(arc_id[list(union)]).tolist()
                event["exceedance_arc_px"] = float(lengths[list(union)].sum())
                event["event_span_px"] = event["exceedance_arc_px"]
                for key in ("bbox_reference_xyxy", "bbox_test_xyxy"):
                    boxes = np.asarray([item[key] for item in records if item[key] is not None])
                    event[key] = np.r_[boxes[:, :2].min(0), boxes[:, 2:].max(0)].tolist() if len(boxes) else None
                for key in ("max_test_to_reference_px", "max_reference_to_test_px"):
                    values = [item[key] for item in supports if key in item]
                    if values:
                        event[key] = max(values)
                fused_events = [item for index, item in enumerate(fused_events) if index not in matched]
        fused_events.append(event)
    events = fused_events
    for index, event in enumerate(events, 1):
        event["event_id"] = f"contour-{index:04d}"
    unknown = required & ~valid
    unknown_runs = cyclic_runs(unknown)
    unobserved = float(lengths[unknown].sum())
    total, requested = float(lengths.sum()), float(lengths[required].sum())
    full = bool(required.all())
    complete = bool(requested > 0 and not unknown.any())
    reasons = list(observation.get("reason_codes", []))
    if unconfirmed_coarse:
        reasons.append("coarse_reference_boundary_unconfirmed")
    diagnostics = observation.get("diagnostics", {})
    for key in ("multiple_components", "image_clipped", "global_shape_change_candidate", "component_assignment_uncertain", "part_component_missing"):
        if diagnostics.get(key):
            reasons.append(key)
    if diagnostics.get("coarse_reference_inconsistent_point_count", 0) > 0:
        reasons.append("coarse_reference_boundary_unconfirmed")
    if diagnostics.get("dense_unreliable_point_count", 0) > 0:
        reasons.append("coarse_outline_contains_unsupported_edges")
    if not registration.get("valid", False):
        reasons.extend(registration.get("reason_codes", ["registration_invalid"]))
    if unknown.any():
        reasons.append("required_arc_unobserved")
    if not requested:
        reasons.append("no_required_perimeter")
    if events:
        reasons.append("contour_exceedance" if any(e["status"] == "NG" for e in events) else "short_or_uncertain_exceedance")
    status = "NG" if any(e["status"] == "NG" for e in events) else ("REVIEW" if reasons or events or not complete else "PASS")
    return {"status": status, "reason_codes": list(dict.fromkeys(reasons)), "events": events,
            "unconfirmed_coarse_events": unconfirmed_coarse,
            "registration": registration, "whole_part_release": None, "distance_unit": "input_pixel",
            "required_fraction_of_perimeter": requested / total,
            "observed_required_fraction": (requested-unobserved)/requested if requested else 0.,
            "reference_perimeter_px": total, "required_length_px": requested,
            "observed_required_length_px": requested-unobserved,
            "unobserved_required_length_px": unobserved,
            "longest_unobserved_required_run_px": max((float(lengths[r].sum()) for r in unknown_runs), default=0.),
            "full_perimeter_configured": full, "required_coverage_complete": complete,
            "full_perimeter_pass": status == "PASS" and full and complete,
            "unknown_arcs": [{"sample_ids": r.tolist(), "length_px": float(lengths[r].sum()),
                              "reason_codes": np.unique(reason[r]).tolist()} for r in unknown_runs],
            "valid": valid, "invalid_reason": reason, "normal_offset_u_px": offset,
            "signed_test_to_ref_px": signed_tr, "signed_ref_to_test_px": signed_rt,
            "test_to_ref_px": distance_tr, "ref_to_test_px": distance_rt,
            "test_to_reference": _summary(distance_tr), "reference_to_test": _summary(distance_rt),
            "coarse_test_to_reference": _summary(coarse_distance),
            "test_segment_starts_xy": starts, "test_segment_ends_xy": ends}
