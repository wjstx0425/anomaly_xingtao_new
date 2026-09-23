"""Conservative whole-outline observations from current image evidence.

Reference geometry initializes probabilities and sampling coordinates only. Missing
or ambiguous image evidence is never replaced by nominal reference points.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import cKDTree


def _channels(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError("Contour extraction requires uint8 gray, BGR or BGRA images")
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] in (3, 4):
        bgr = image[:, :, :3].copy()
    else:
        raise ValueError("Unsupported image channel count")
    valid = np.ones(image.shape[:2], dtype=bool)
    if image.ndim == 3 and image.shape[2] == 4:
        valid = image[:, :, 3] == 255
    return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), valid


def _transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ matrix[:2, :2].T + matrix[:2, 2]


def _matrix(registration: dict | None) -> np.ndarray:
    matrix = np.asarray((registration or {}).get("T_test_to_reference", np.eye(3)), dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Invalid registration matrix")
    if not np.allclose(matrix[2], [0, 0, 1]) or not np.allclose(matrix[:2, :2].T @ matrix[:2, :2], np.eye(2), atol=1e-6) or not np.isclose(np.linalg.det(matrix[:2, :2]), 1):
        raise ValueError("Registration must be a proper rigid transform")
    return matrix


def coarse_segment(image: np.ndarray, reference_mask: np.ndarray | None, config: dict, registration: dict | None = None) -> dict:
    """Segment current material and assign foreground components to the part."""
    from .segmentation import segment_part
    return segment_part(image, reference_mask, config, registration)


def _sample(array: np.ndarray, xy: np.ndarray) -> np.ndarray:
    shape = xy.shape[:-1]
    # OpenCV remap has a signed-short limit on each destination dimension.
    points = xy.reshape(-1, 2).astype(np.float32)
    result = np.empty(len(points), np.float32)
    source = array.astype(np.float32)
    for start in range(0, len(points), 30000):
        batch = points[start:start + 30000]
        result[start:start + len(batch)] = cv2.remap(source, batch[:, 0].reshape(1, -1), batch[:, 1].reshape(1, -1), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0).ravel()
    return result.reshape(shape)


def _candidate_transitions(values, contrasts, accepted, center_offsets, offsets):
    """Locate an observed edge by gradient mass, retaining subpixel motion.

    A centered finite difference has a flat maximum around a sharp step. Taking
    its first argmax introduces a systematic position error. Center the plateau,
    then integrate the signed gradient over a small local window instead.
    """
    indices = np.flatnonzero(accepted)
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    derivative = np.diff(values)
    positions = (offsets[:-1] + offsets[1:]) / 2
    candidates = []
    for group in groups:
        if not len(group):
            continue
        strength = np.abs(contrasts[group])
        plateau = group[np.isclose(strength, strength.max(), rtol=1e-6, atol=1e-6)]
        center = float(np.mean(center_offsets[plateau]))
        peak = int(plateau[len(plateau) // 2])
        polarity = float(np.sign(contrasts[peak]))
        local = np.abs(positions - center) <= 2.0
        weights = np.where(local, np.maximum(polarity * derivative, 0), 0)
        if weights.sum() <= 0:
            continue
        u = float(np.sum(positions * weights) / weights.sum())
        candidates.append((u, float(abs(contrasts[peak])), float(contrasts[peak])))
    return candidates


def extract_full_outline(image: np.ndarray, reference: dict, registration: dict, config: dict) -> dict:
    """Extract image-supported samples and independent global contour evidence."""
    _, gray, domain = _channels(image)
    if "T_test_to_reference" not in registration or registration.get("valid") is False or registration.get("success") is False:
        raise ValueError("Extraction requires successful explicit rigid registration")
    if np.asarray(reference["mask"]).shape != gray.shape:
        raise ValueError("Reference and test image sizes differ")
    settings = config.get("extraction", config)
    mode = settings.get("edge_selection_mode", "independent_v2")
    if mode not in ("independent_v2", "continuous_v3"):
        raise ValueError("Unsupported edge_selection_mode")
    continuous = mode == "continuous_v3"
    if continuous:
        from .reference_edges import select_reference_edge
        from .path_selection import select_candidate_path
        for name, default in (("reference_edge_search_px", 6.), ("path_continuity_weight", .4),
                              ("path_ambiguity_margin", .15), ("path_jump_cap_px", 4.),
                              ("corner_persistence_radius_px", 6.), ("corner_persistence_angle_deg", 25.)):
            value = settings.get(name, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive finite")
    if continuous and settings.get("corner_persistence_angle_deg", 25.) > 180:
        raise ValueError("corner_persistence_angle_deg must not exceed 180")
    if settings.get("min_edge_amplitude") is None or settings.get("min_candidate_margin") is None:
        raise ValueError("Extraction requires explicit edge amplitude and candidate margin")
    amplitude = float(settings["min_edge_amplitude"])
    margin = float(settings["min_candidate_margin"])
    if amplitude <= 0 or not 0 <= margin <= 1:
        raise ValueError("Invalid edge amplitude or candidate margin")
    inward = float(settings.get("search_inward_px", 40))
    outward = float(settings.get("search_outward_px", 40))
    step = float(settings.get("profile_step_px", 0.5))
    if min(inward, outward, step) <= 0:
        raise ValueError("Profile ranges and step must be positive")
    for key, default in (("min_profile_similarity", .90), ("min_profile_similarity_margin", .03)):
        value = settings.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{key} must be finite and in (0, 1]")
    agreement = settings.get("coarse_reference_agreement_px", 2.)
    if isinstance(agreement, bool) or not isinstance(agreement, (int, float)) or not np.isfinite(agreement) or agreement <= 0:
        raise ValueError("coarse_reference_agreement_px must be positive finite")
    matrix = _matrix(registration)
    inverse = np.linalg.inv(matrix)
    coarse = coarse_segment(image, reference["mask"], config, registration)
    baseline = reference.get("automatic_reference_segmentation")
    if baseline is None:
        baseline = coarse_segment(np.asarray(reference["image"]), reference["mask"], config, {"T_test_to_reference": np.eye(3)})
    nominal = np.asarray(reference["sample_xy"], dtype=float)
    normals = np.asarray(reference["inward_normal_xy"], dtype=float)
    offsets = np.arange(-outward, inward + step / 2, step)
    ref_profiles = nominal[:, None, :] + offsets[None, :, None] * normals[:, None, :]
    profiles = _transform(ref_profiles, inverse)
    intensity = _sample(gray, profiles)
    _, reference_gray, reference_domain = _channels(np.asarray(reference["image"]))
    reference_intensity = _sample(reference_gray, ref_profiles)
    reference_foreground = _sample((np.asarray(reference["mask"]) != 0).astype(np.uint8), ref_profiles)
    reference_domains = _sample(reference_domain, ref_profiles) > 0.999
    foreground = _sample(coarse["mask"], profiles)
    baseline_foreground = _sample(baseline["mask"], ref_profiles)
    height, width = gray.shape
    domains = (_sample(domain, profiles) > 0.999) & (profiles[..., 0] >= 1) & (profiles[..., 0] < width - 2) & (profiles[..., 1] >= 1) & (profiles[..., 1] < height - 2)
    if continuous:
        # Reference and current use identical image-domain requirements. The
        # centroid integrates a wider neighbourhood than the contrast operator;
        # exclude candidates whose gradient mass would include padding/alpha gaps.
        rh, rw = reference_gray.shape
        reference_domains &= (ref_profiles[..., 0] >= 1) & (ref_profiles[..., 0] < rw - 2) & (ref_profiles[..., 1] >= 1) & (ref_profiles[..., 1] < rh - 2)
        guard = max(1, int(np.ceil(2. / step)))
        kernel = np.ones((1, 2 * guard + 1), np.uint8)
        domains = cv2.erode(domains.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        reference_domains = cv2.erode(reference_domains.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    # Compare samples two pixels apart, so a 0.5-pixel step does not artificially
    # divide edge amplitude. Candidate runs collapse to their strongest observation.
    radius = max(1, int(round(1 / step)))
    contrasts = intensity[:, 2 * radius:] - intensity[:, :-2 * radius]
    reference_contrasts = reference_intensity[:, 2 * radius:] - reference_intensity[:, :-2 * radius]
    center_offsets = offsets[radius:-radius]
    observed = np.full(nominal.shape, np.nan)
    valid = np.zeros(len(nominal), bool)
    reasons = np.full(len(nominal), "no_supported_edge", dtype=object)
    displacement = np.full(len(nominal), np.nan)
    candidates = []
    path_rows = [[] for _ in nominal]
    path_choice_ids = [[] for _ in nominal]
    reference_raw_u = np.full(len(nominal), np.nan)
    reference_edge_reasons = []
    reference_edge_evidence = []
    material_reference_valid = np.asarray(reference.get("reference_valid", np.ones(len(nominal), bool)), bool)
    if material_reference_valid.shape != (len(nominal),):
        raise ValueError("Reference validity must match samples")
    reference_transitions_all = []
    reference_selected_u = np.full(len(nominal), np.nan)
    reference_path_diagnostics = {}
    if continuous:
        reference_rows = []
        for i in range(len(nominal)):
            good = np.logical_and.reduce([reference_domains[i, j: len(offsets) - 2 * radius + j] for j in range(2 * radius + 1)])
            transitions = _candidate_transitions(reference_intensity[i], reference_contrasts[i],
                                                  (np.abs(reference_contrasts[i]) >= amplitude) & good, center_offsets, offsets)
            reference_transitions_all.append(transitions)
            evidence = select_reference_edge(transitions, offsets, reference_intensity[i], reference_foreground[i],
                                             search_px=float(settings.get("reference_edge_search_px", 6.)))
            reference_edge_evidence.append(evidence)
            supported = [item for item in evidence["candidates"] if item["supported"] and material_reference_valid[i]]
            # The teaching mask supplies material-side evidence, not forced
            # coordinates. Candidate confidence and neighbouring image edges
            # resolve local alternatives; equal paths remain unknown.
            reference_rows.append([{"u_px": item["u_px"], "cost": .2 * (1. - item["confidence"])
                                   + .3 * min(item["material_support_distances_px"]) / 6.} for item in supported])
        ref_path = select_candidate_path(reference_rows, continuity_weight=float(settings.get("path_continuity_weight", .4)),
                                         ambiguity_margin=float(settings.get("path_ambiguity_margin", .15)),
                                         jump_cap_px=float(settings.get("path_jump_cap_px", 4.)), closed=True)
        reference_path_diagnostics = ref_path["diagnostics"]
        for i, j in enumerate(ref_path["selected_indices"]):
            if j >= 0:
                reference_selected_u[i] = reference_rows[i][j]["u_px"]
                reference_edge_reasons.append("supported_reference_image_path")
            else:
                reference_edge_reasons.append("ambiguous_reference_path" if reference_rows[i] else "no_supported_reference_edge")
            reference_edge_evidence[i].update(resolved_valid=bool(j >= 0), path_selected_u_px=float(reference_selected_u[i]), path_margin=float(ref_path["margins"][i]))
    for i in range(len(nominal)):
        good_domain = np.logical_and.reduce([domains[i, j: len(offsets) - 2 * radius + j] for j in range(2 * radius + 1)])
        accepted = (np.abs(contrasts[i]) >= amplitude) & good_domain
        transitions = _candidate_transitions(intensity[i], contrasts[i], accepted, center_offsets, offsets)
        reference_good_domain = np.logical_and.reduce([reference_domains[i, j: len(offsets) - 2 * radius + j] for j in range(2 * radius + 1)])
        reference_accepted = (np.abs(reference_contrasts[i]) >= amplitude) & reference_good_domain
        reference_transitions = _candidate_transitions(reference_intensity[i], reference_contrasts[i], reference_accepted, center_offsets, offsets)
        # The standard mask boundary is a material-pixel contour, while image
        # transitions lie between pixel centers. Calibrate that fixed convention
        # from the standard image, without changing current measured displacement.
        calibration = [item for item in reference_transitions if abs(item[0]) <= 2.0]
        if continuous:
            calibration = [item for item in reference_transitions_all[i] if np.isfinite(reference_selected_u[i]) and np.isclose(item[0], reference_selected_u[i])]
        correction = -calibration[0][0] if len(calibration) == 1 else None
        if correction is not None:
            reference_raw_u[i] = -correction
        choices = []
        for u, strength, contrast in transitions:
            measured = u + correction if correction is not None else u
            choices.append({"candidate_xy": (nominal[i] + measured * normals[i]).tolist(), "normal_offset_u_px": measured,
                            "raw_image_transition_u_px": u, "reference_pixel_boundary_correction_px": correction,
                            "edge_amplitude": strength, "local_contrast": contrast, "polarity": int(np.sign(contrast)),
                            "foreground_background_support": True, "selected_or_rejected_reason": "ambiguous"})
        candidates.append(choices)
        if correction is None:
            reasons[i] = "reference_image_edge_calibration_unconfirmed"
            continue
        if not choices:
            if not domains[i].all():
                reasons[i] = "clipped_or_transparent_search"
            continue
        # Use the same measured reference edge signature, not the strongest
        # unrelated intensity transition. No distance-to-nominal ranking term.
        reference_u, _, reference_contrast = calibration[0]
        stencil = np.arange(-4., 4. + step / 2, step)
        template = np.interp(reference_u + stencil, offsets, reference_intensity[i])
        centered = template - template.mean()
        template_norm = np.linalg.norm(centered)
        for choice in choices:
            u = choice["raw_image_transition_u_px"]
            patch = np.interp(u + stencil, offsets, intensity[i])
            centered_patch = patch - patch.mean()
            norm = np.linalg.norm(centered_patch) * template_norm
            score = float(np.dot(centered_patch, centered) / norm) if norm > 1e-8 else -1.
            if choice["polarity"] != np.sign(reference_contrast) or u + stencil[0] < offsets[0] or u + stencil[-1] > offsets[-1]:
                score = -1.
            material_support = bool(np.interp(u + 3., offsets, foreground[i]) > .75 and np.interp(u - 3., offsets, foreground[i]) < .25)
            reference_material_support = bool(np.interp(reference_u + 3., offsets, baseline_foreground[i]) > .75 and np.interp(reference_u - 3., offsets, baseline_foreground[i]) < .25)
            # Apply material-transition selection symmetrically only where the
            # automatic reference observation actually supports this boundary.
            if reference_material_support and not material_support:
                score = -1.
            choice["reference_profile_similarity"] = score
            choice["foreground_background_support"] = material_support
            choice["reference_material_transition_supported"] = reference_material_support
        if continuous:
            for j, choice in enumerate(choices):
                polarity_matches = choice["polarity"] == np.sign(reference_contrast)
                supported = choice["foreground_background_support"]
                signature = choice["reference_profile_similarity"]
                # A path selects among actual image candidates, never fabricates
                # a point. Material evidence may survive texture changes.
                eligible = polarity_matches and ((reference_material_support and supported) or (not reference_material_support and signature >= float(settings.get("min_profile_similarity", .90))))
                if eligible:
                    cost = (0. if reference_material_support else .5) + .1 * (1. - max(-1., signature))
                    path_rows[i].append({"u_px": choice["normal_offset_u_px"], "cost": cost})
                    path_choice_ids[i].append(j)
                else:
                    choice["selected_or_rejected_reason"] = "no_current_material_or_signature_support"
            continue
        ranking = np.argsort([-item["reference_profile_similarity"] for item in choices])
        best = choices[int(ranking[0])]
        material_choices = [item for item in choices if item["foreground_background_support"] and item["polarity"] == np.sign(reference_contrast)]
        unique_material_transition = bool(reference_material_support and len(material_choices) == 1)
        if unique_material_transition:
            # Surface texture need not match the standard. A unique current
            # material/background transition can be measured on its own evidence.
            best = material_choices[0]
        if not unique_material_transition and best["reference_profile_similarity"] < float(settings.get("min_profile_similarity", .90)):
            reasons[i] = "edge_signature_mismatch"
            continue
        if not unique_material_transition and len(ranking) > 1 and best["reference_profile_similarity"] - choices[int(ranking[1])]["reference_profile_similarity"] < float(settings.get("min_profile_similarity_margin", .03)):
            reasons[i] = "ambiguous_edges"
            continue
        observed[i] = best["candidate_xy"]
        displacement[i] = best["normal_offset_u_px"]
        valid[i] = True
        reasons[i] = ""
        best["selected_or_rejected_reason"] = "unique_current_material_transition" if unique_material_transition else "matching_reference_image_edge_signature"
    path_diagnostics = {}
    if continuous:
        path = select_candidate_path(path_rows, continuity_weight=float(settings.get("path_continuity_weight", .4)),
                                     ambiguity_margin=float(settings.get("path_ambiguity_margin", .15)),
                                     jump_cap_px=float(settings.get("path_jump_cap_px", 4.)), closed=True)
        path_diagnostics = path["diagnostics"]
        for i, j in enumerate(path["selected_indices"]):
            if j < 0:
                if np.isfinite(reference_raw_u[i]):
                    reasons[i] = "continuous_path_ambiguous" if path_rows[i] else "no_supported_path_candidate"
                continue
            choice = candidates[i][path_choice_ids[i][j]]
            observed[i] = choice["candidate_xy"]
            displacement[i] = choice["normal_offset_u_px"]
            valid[i] = True
            reasons[i] = ""
            choice["selected_or_rejected_reason"] = "continuous_supported_image_path"
            choice["path_margin"] = float(path["margins"][i])
    dense_current = coarse["dense_xy"]
    dense = _transform(dense_current, matrix)
    finite = np.isfinite(dense_current).all(axis=1)
    dense_valid = np.zeros(len(dense), bool)
    # A coarse polygon is not automatically measured: require local image contrast,
    # valid alpha and distance from the image boundary for every dense vertex.
    local_range = cv2.dilate(gray, np.ones((3, 3), np.uint8)).astype(float) - cv2.erode(gray, np.ones((3, 3), np.uint8))
    safe_domain = cv2.erode(domain.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    dense_valid[finite] = (_sample(local_range, dense_current[finite]) >= amplitude) & (_sample(safe_domain, dense_current[finite]) > 0.999)
    corners = np.asarray(reference.get("corner_mask", np.zeros(len(nominal), bool)), dtype=bool)
    raw_corners = corners.copy()
    if continuous:
        from .corner_tracking import persistent_corner_mask
        reference_lengths = np.asarray(reference.get("cell_length_px", np.linalg.norm(np.roll(nominal, -1, axis=0) - nominal, axis=1)))
        corners = persistent_corner_mask(nominal, raw_corners, reference_lengths,
                                         radius_px=float(settings.get("corner_persistence_radius_px", 6.)),
                                         min_angle_deg=float(settings.get("corner_persistence_angle_deg", 25.)))
    tree = cKDTree(dense[finite]) if finite.any() else None
    for i in np.flatnonzero(corners if not continuous else np.zeros(len(corners), bool)):
        # Local 2-D contour support comes from current segmentation, never an
        # interpolated nominal edge. Ambiguous/displaced corners remain unknown.
        valid[i] = False
        observed[i] = np.nan
        displacement[i] = np.nan
        reasons[i] = "corner_correspondence_unconfirmed"
        if tree is None:
            continue
        distance, nearest = tree.query(nominal[i])
        index = np.flatnonzero(finite)[nearest]
        if distance <= 1.5 and dense_valid[index]:
            observed[i] = dense[index]
            displacement[i] = np.dot(observed[i] - nominal[i], normals[i])
            valid[i] = True
            reasons[i] = ""
    # A segmentation edge already inconsistent with the material reference
    # cannot become certain missing material merely by being measured again.
    # Observe the reference with the SAME automatic segmenter; retain these
    # unsupported edges as diagnostics, never substitute them into the standard.
    baseline_xy = np.asarray(baseline["dense_xy"])
    baseline_xy = baseline_xy[np.isfinite(baseline_xy).all(axis=1)]
    candidate_eligible = finite.copy()
    if len(baseline_xy) and finite.any():
        nominal_distance = cKDTree(np.asarray(reference["dense_xy"])).query(dense[finite])[0]
        baseline_distance = cKDTree(baseline_xy).query(dense[finite])[0]
        inconsistent = (nominal_distance > float(settings.get("coarse_reference_agreement_px", 2.))) & (baseline_distance <= float(settings.get("coarse_reference_agreement_px", 2.)))
        candidate_eligible[np.flatnonzero(finite)[inconsistent]] = False
    else:
        candidate_eligible[:] = False
    corner_diagnostics = {}
    if continuous:
        from .corner_tracking import track_corner_runs
        raw_ref = nominal + reference_raw_u[:, None] * normals
        raw_test = observed + reference_raw_u[:, None] * normals
        ref_ok = np.isfinite(reference_raw_u)
        reference_valid = np.asarray(reference.get("reference_valid", np.ones(len(nominal), bool)), bool)
        baseline_dense = np.asarray(baseline["dense_xy"], float)
        baseline_finite = np.isfinite(baseline_dense).all(axis=1)
        baseline_valid = np.zeros(len(baseline_dense), bool)
        ref_range = cv2.dilate(reference_gray, np.ones((3, 3), np.uint8)).astype(float) - cv2.erode(reference_gray, np.ones((3, 3), np.uint8))
        ref_domain_safe = cv2.erode(reference_domain.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
        baseline_valid[baseline_finite] = (_sample(ref_range, baseline_dense[baseline_finite]) >= amplitude) & (_sample(ref_domain_safe, baseline_dense[baseline_finite]) > .999)
        ref_track = track_corner_runs(nominal, corners, raw_ref, ref_ok & reference_valid, baseline_dense, baseline_valid,
                                      max_distance_px=max(inward, outward))
        cur_track = track_corner_runs(nominal, corners, raw_test, valid & reference_valid, dense, dense_valid & candidate_eligible,
                                      max_distance_px=max(inward, outward))
        for i in np.flatnonzero(corners):
            for choice in candidates[i]:
                if choice["selected_or_rejected_reason"] == "continuous_supported_image_path":
                    choice["selected_or_rejected_reason"] = "superseded_by_corner_chain_check"
            valid[i] = bool(ref_track["valid"][i] and cur_track["valid"][i] and reference_valid[i])
            if valid[i]:
                delta = cur_track["sample_xy"][i] - ref_track["sample_xy"][i]
                observed[i] = nominal[i] + delta
                displacement[i] = np.dot(delta, normals[i])
                reasons[i] = ""
            else:
                observed[i] = np.nan
                displacement[i] = np.nan
                reasons[i] = "reference_corner_unconfirmed" if not ref_track["valid"][i] else str(cur_track["invalid_reason"][i])
        reasons[~reference_valid] = "reference_material_normal_unconfirmed"
        for i in np.flatnonzero(~reference_valid):
            for choice in candidates[i]:
                if choice["selected_or_rejected_reason"] == "continuous_supported_image_path":
                    choice["selected_or_rejected_reason"] = "rejected_reference_material_normal_unconfirmed"
        valid &= reference_valid
        observed[~valid] = np.nan
        displacement[~valid] = np.nan
        corner_diagnostics = {"reference": ref_track["diagnostics"], "current": cur_track["diagnostics"]}
    diagnostics = dict(coarse["diagnostics"])
    diagnostics["coarse_reference_inconsistent_point_count"] = int((finite & ~candidate_eligible).sum())
    diagnostics.update({"min_edge_amplitude_used": amplitude, "legacy_min_candidate_margin": margin,
                        "edge_selection_method": "continuous_v3" if continuous else "unique_material_transition_then_profile_signature",
                        "min_profile_similarity_used": float(settings.get("min_profile_similarity", .90)),
                        "min_profile_similarity_margin_used": float(settings.get("min_profile_similarity_margin", .03)),
                        "coarse_reference_agreement_px_used": float(agreement),
                        "dense_unreliable_point_count": int((~dense_valid & finite).sum()), "corner_strategy": "local_2d_conservative", "edge_coordinate_convention": "image_gradient_centroid_calibrated_to_reference_material_pixels", "global_shape_change_candidate": False})
    if finite.any():
        distances, _ = cKDTree(np.asarray(reference["dense_xy"])).query(dense[finite])
        diagnostics["global_shape_change_candidate"] = bool(np.any(distances > max(inward, outward)))
        diagnostics["max_coarse_distance_to_reference_vertices_px"] = float(distances.max())
    if continuous:
        diagnostics.update(reference_edge_selection_reasons=reference_edge_reasons, reference_path_selection=reference_path_diagnostics, path_selection=path_diagnostics,
                           corner_tracking=corner_diagnostics, corner_strategy="anchored_observed_dense_chain",
                           raw_corner_sample_count=int(raw_corners.sum()), structural_corner_sample_count=int(corners.sum()),
                           corner_persistence_radius_px=float(settings.get("corner_persistence_radius_px", 6.)),
                           corner_persistence_angle_deg=float(settings.get("corner_persistence_angle_deg", 25.)),
                           reference_edge_search_px=float(settings.get("reference_edge_search_px", 6.)),
                           edge_coordinate_convention="current_minus_reference_image_edges_in_material_pixel_frame")
    result = {"sample_xy": observed, "valid": valid, "invalid_reason": reasons, "normal_offset_u_px": displacement, "dense_xy": dense, "dense_valid": dense_valid, "dense_candidate_eligible": candidate_eligible, "mask": coarse["mask"], "candidates": candidates, "diagnostics": diagnostics, "registration": registration}
    if continuous:
        result.update(reference_image_edge_u_px=reference_raw_u, reference_image_edge_xy=nominal + reference_raw_u[:, None] * normals,
                      reference_edge_evidence=reference_edge_evidence, reference_corner_observed_xy=ref_track["sample_xy"], current_corner_observed_xy=cur_track["sample_xy"], current_image_edge_xy=cur_track["sample_xy"])
    return result
