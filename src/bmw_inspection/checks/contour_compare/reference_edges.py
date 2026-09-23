"""Conservative reference-image edge calibration with explicit material context."""

from __future__ import annotations

import numpy as np


def select_reference_edge(
    transitions, offsets, intensity, material_profile, *, search_px=6.0,
) -> dict:
    """Select one observed transition supported by material and image context.

    Offsets increase towards material. A transition is the existing image
    detector's ``(u, strength, signed_contrast)`` tuple; this function never
    generates or moves transitions. The teaching mask may have a small offset
    from the actual image edge. Paired mask probes therefore use several
    distances, while contradictory directions remain unconfirmed.

    Image context must sustain the transition's polarity beyond its local
    gradient peak. Multiple supported transitions remain unknown regardless of
    their relative distance or strength. Confidence describes evidence support,
    not a calibrated probability of a correct material boundary.
    """
    offsets, intensity, material = (
        np.asarray(value, dtype=float) for value in (offsets, intensity, material_profile)
    )
    if (
        offsets.ndim != 1 or len(offsets) < 3
        or intensity.shape != offsets.shape or material.shape != offsets.shape
        or not all(np.isfinite(value).all() for value in (offsets, intensity, material))
        or np.any(np.diff(offsets) <= 0)
        or np.any((material < 0) | (material > 1))
        or not np.isfinite(search_px) or search_px <= 0
    ):
        raise ValueError("Reference edge profiles require finite matching arrays and increasing offsets")

    candidates = []
    for transition in transitions:
        u, strength, contrast = map(float, transition)
        if not np.isfinite([u, strength, contrast]).all() or strength <= 0 or contrast == 0:
            raise ValueError("Reference transitions require finite position, positive strength and nonzero contrast")
        candidate = {
            "u_px": u, "strength": strength, "contrast": contrast,
            "supported": False, "confidence": 0.0,
        }
        candidates.append(candidate)
        if abs(u) > search_px:
            candidate["reason"] = "outside_reference_search"
            continue

        # Never extend a short profile by repeating its endpoint values.
        distances = np.array([1.0, 2.0, 3.0, 4.0, 6.0])
        available = (u - distances >= offsets[0]) & (u + distances <= offsets[-1])
        distances = distances[available]
        if not len(distances):
            candidate["reason"] = "insufficient_material_context"
            continue
        inside = np.interp(u + distances, offsets, material)
        outside = np.interp(u - distances, offsets, material)
        forward = (inside >= 0.75) & (outside <= 0.25)
        backward = (outside >= 0.75) & (inside <= 0.25)
        candidate["material_support_distances_px"] = distances[forward].tolist()
        candidate["material_conflict_distances_px"] = distances[backward].tolist()
        if backward.any():
            candidate["reason"] = "conflicting_material_direction"
            continue
        if not forward.any():
            candidate["reason"] = "no_material_boundary_context"
            continue

        # A narrow highlight's two gradients need not be the material boundary.
        # Compare several progressively wider image contexts instead of ranking
        # the nearest/strongest peak. Medians resist an isolated bright pixel.
        context_distances = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        available = (u - context_distances >= offsets[0]) & (u + context_distances <= offsets[-1])
        context_distances = context_distances[available]
        if len(context_distances) < 3:
            candidate["reason"] = "insufficient_image_context"
            continue
        inside_values = np.interp(u + context_distances, offsets, intensity)
        outside_values = np.interp(u - context_distances, offsets, intensity)
        differences = inside_values - outside_values
        context_contrast = float(np.median(differences))
        polarity_agreement = float(np.mean(differences * np.sign(contrast) > 0))
        persistence = abs(context_contrast) / strength
        candidate.update({
            "context_contrast": context_contrast,
            "polarity_agreement": polarity_agreement,
            "context_to_peak_ratio": persistence,
        })
        if context_contrast * contrast <= 0 or polarity_agreement < 0.8:
            candidate["reason"] = "image_context_polarity_conflict"
            continue
        if persistence < 0.35:
            candidate["reason"] = "transient_image_line"
            continue
        candidate.update({
            "supported": True,
            "confidence": float(min(1.0, persistence) * polarity_agreement),
            "reason": "supported_image_and_material_transition",
        })

    supported = [candidate for candidate in candidates if candidate["supported"]]
    if len(supported) != 1:
        return {
            "valid": False, "u_px": None, "confidence": 0.0,
            "reason": "ambiguous_reference_edges" if supported else "no_supported_reference_edge",
            "candidates": candidates,
        }
    return {
        "valid": True, "u_px": supported[0]["u_px"],
        "confidence": supported[0]["confidence"], "reason": "unique_supported_reference_edge",
        "candidates": candidates,
    }
