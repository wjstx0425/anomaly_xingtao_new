"""Select image-supported contour candidates with bounded continuity costs."""

from __future__ import annotations

import numpy as np


def _transition(left: np.ndarray, right: np.ndarray, weight: float, cap: float) -> np.ndarray:
    """Bound even a large geometric jump to one continuity penalty."""
    return weight * np.minimum(np.abs(left[:, None] - right[None, :]) / cap, 1.0)


def _chain_marginals(values: list[np.ndarray], costs: list[np.ndarray], edges: list[np.ndarray], first: int | None = None, closing: np.ndarray | None = None) -> list[np.ndarray]:
    """Return exact min-sum costs conditioned on each row's candidate."""
    forward = [costs[0].copy()]
    if first is not None:
        forward[0][:] = np.inf
        forward[0][first] = costs[0][first]
    for index in range(1, len(values)):
        forward.append(costs[index] + np.min(forward[-1][:, None] + edges[index - 1], axis=0))
    backward = np.zeros(len(values[-1])) if closing is None else closing[:, first].copy()
    result = [np.empty(0)] * len(values)
    result[-1] = forward[-1] + backward
    for index in range(len(values) - 2, -1, -1):
        backward = np.min(edges[index] + costs[index + 1][None, :] + backward[None, :], axis=1)
        result[index] = forward[index] + backward
    return result


def select_candidate_path(candidate_rows, *, continuity_weight: float = 0.4, ambiguity_margin: float = 0.15, jump_cap_px: float = 4.0, closed: bool = False) -> dict:
    """Choose measured candidates, leaving absent or ambiguous rows unknown.

    Each row contains dictionaries with finite ``u_px`` and ``cost`` values;
    lower unary cost means stronger image evidence. Continuity adds
    ``continuity_weight * min(abs(delta_u) / jump_cap_px, 1)`` per adjacent
    pair. Its bounded cost permits genuine sharp defects and does not prefer
    zero displacement. Returned indices always address the original rows.

    ``margins`` are the second-best minus best *whole-path* constrained costs
    for that row, in unary-cost units. Equal alternatives remain unknown even
    when ``ambiguity_margin`` is zero. One candidate has infinite margin;
    no candidates have NaN margin and selected index -1. These margins measure
    candidate-selection ambiguity, not material-edge validity or confidence.

    Empty rows break connectivity and are never interpolated. ``closed=True``
    adds the real last-to-first adjacency. With gaps, only contiguous observed
    runs are solved, including a run across the array seam where appropriate.
    Fully observed loops use exact min-sum inference conditioned on each
    candidate of the smallest row; there is no approximate periodic shortcut.
    """
    parameters = np.asarray([continuity_weight, ambiguity_margin, jump_cap_px], dtype=float)
    if not np.isfinite(parameters).all() or continuity_weight < 0 or ambiguity_margin < 0 or jump_cap_px <= 0:
        raise ValueError("Path weights and margin must be finite and nonnegative; jump cap must be positive")
    rows = list(candidate_rows)
    values = [np.asarray([candidate["u_px"] for candidate in row], dtype=float) for row in rows]
    costs = [np.asarray([candidate["cost"] for candidate in row], dtype=float) for row in rows]
    if any(not np.isfinite(array).all() for array in values + costs):
        raise ValueError("Path candidates require finite u_px and cost")
    count = len(rows)
    selected = np.full(count, -1, dtype=int)
    margins = np.full(count, np.nan)
    populated = np.asarray([len(row) > 0 for row in rows], dtype=bool)
    cycle = bool(closed and count > 1 and populated.all())
    blocks = []
    if cycle:
        pivot = int(np.argmin([len(row) for row in rows]))
        blocks = [np.roll(np.arange(count), -pivot).tolist()]
    elif count:
        order = np.arange(count)
        if closed and not populated.all():
            # Start after a gap: never silently join observations across it.
            order = np.roll(order, -int(np.flatnonzero(~populated)[0]) - 1)
        block = []
        for index in order:
            if populated[index]:
                block.append(int(index))
            elif block:
                blocks.append(block)
                block = []
        if block:
            blocks.append(block)
    energies = []
    for block in blocks:
        offsets = [values[index] for index in block]
        unary = [costs[index] for index in block]
        # Subtract per-row baselines to avoid loss of precision from irrelevant
        # large common unary offsets; this leaves every margin unchanged.
        unary = [cost - cost.min() for cost in unary]
        edges = [_transition(left, right, continuity_weight, jump_cap_px) for left, right in zip(offsets[:-1], offsets[1:])]
        if cycle:
            closing = _transition(offsets[-1], offsets[0], continuity_weight, jump_cap_px)
            marginal = [np.full(len(row), np.inf) for row in offsets]
            for first in range(len(offsets[0])):
                conditioned = _chain_marginals(offsets, unary, edges, first, closing)
                marginal = [np.minimum(best, candidate) for best, candidate in zip(marginal, conditioned)]
        else:
            marginal = _chain_marginals(offsets, unary, edges)
        energies.append(float(marginal[0].min()))
        for index, alternatives in zip(block, marginal):
            ranking = np.argsort(alternatives, kind="stable")
            best = int(ranking[0])
            gap = float(alternatives[ranking[1]] - alternatives[best]) if len(ranking) > 1 else np.inf
            margins[index] = gap
            if gap > max(ambiguity_margin, 1e-12):
                selected[index] = best
    return {"selected_indices": selected, "margins": margins,
            "diagnostics": {"method": "exact_min_sum_candidate_path", "closed_requested": bool(closed),
                            "cycle_solved": cycle, "block_count": len(blocks), "missing_row_count": int((~populated).sum()),
                            "ambiguous_row_count": int(((selected < 0) & populated).sum()),
                            "selected_row_count": int((selected >= 0).sum()), "continuity_weight": float(continuity_weight),
                            "ambiguity_margin": float(ambiguity_margin), "jump_cap_px": float(jump_cap_px),
                            "transition_cost": "weight_times_min_absolute_delta_over_cap_one",
                            "normalized_block_energies": energies}}
