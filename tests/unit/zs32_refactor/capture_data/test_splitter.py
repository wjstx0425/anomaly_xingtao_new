"""Physical-part split isolation tests."""

from __future__ import annotations

import sys

import pytest

from zs32_inspection.data.splitter import (
    SPLIT_ALGORITHM,
    SplitPolicy,
    assign_grouped_splits,
    validate_part_grouped_splits,
)


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="authoritative runtime is Linux only")


def test_split_assignment_is_deterministic_and_grouped() -> None:
    strata = {f"part-{index}": "normal:right" for index in range(12)}
    policy = SplitPolicy(SPLIT_ALGORITHM, 43, 0.2, 0.2)
    first = assign_grouped_splits(
        strata,
        policy=policy,
    )
    second = assign_grouped_splits(
        strata,
        policy=policy,
    )
    assert first == second
    assert set(item.part_instance_id for item in first) == set(strata)
    assert policy.as_dict() == {
        "algorithm": "sha256_seed_stratum_part_v1",
        "seed": 43,
        "calibration_ratio": 0.2,
        "test_ratio": 0.2,
    }


def test_split_policy_rejects_unknown_algorithm_and_non_finite_ratios() -> None:
    with pytest.raises(ValueError, match="unsupported split algorithm"):
        SplitPolicy("random", 43, 0.2, 0.2)
    with pytest.raises(ValueError, match="finite"):
        SplitPolicy(SPLIT_ALGORITHM, 43, float("nan"), 0.2)


def test_one_physical_part_cannot_cross_splits() -> None:
    with pytest.raises(ValueError, match="leaks across splits"):
        validate_part_grouped_splits((("same-part", "train"), ("same-part", "test")))
