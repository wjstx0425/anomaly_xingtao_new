"""Linux-only tests for the production held-out acceptance boundary."""

from __future__ import annotations

import json
import sys

import pytest

from zs32_inspection.calibration.acceptance import (
    HeldOutAcceptancePolicy,
    evaluate_heldout_acceptance,
    load_heldout_acceptance_decision_bytes,
    load_heldout_acceptance_policy_bytes,
)
from zs32_inspection.calibration.reports import HeldOutPartMetrics
from zs32_inspection.runtime.publisher import canonical_json_bytes


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="ZS32 verification is Linux-only")


def _policy_payload(
    *,
    min_normal: int = 10,
    min_defect: int = 10,
) -> dict[str, object]:
    return {
        "schema": "zs32.heldout_acceptance_policy",
        "schema_version": 1,
        "product": "ZS32",
        "policy_id": "zs32-heldout-strict",
        "policy_version": "1.0.0",
        "limits": {
            "max_defect_escape_rate": 0.0,
            "max_normal_reject_rate": 0.0,
            "max_review_rate": 0.0,
            "min_normal_heldout_parts": min_normal,
            "min_defect_heldout_parts": min_defect,
        },
    }


def _metrics(
    *,
    normal: int = 10,
    defect: int = 10,
    defect_escape: int = 0,
    normal_reject: int = 0,
    review: int = 0,
    incomplete: int = 0,
) -> HeldOutPartMetrics:
    part_count = normal + defect
    strong = defect - defect_escape + normal_reject
    clear = part_count - strong - review - incomplete
    return HeldOutPartMetrics(
        dataset_release_id="dataset-v1",
        test_split_id="test-v1",
        part_count=part_count,
        normal_part_count=normal,
        defect_part_count=defect,
        clear_count=clear,
        gray_count=review,
        strong_count=strong,
        template_ng_count=0,
        incomplete_count=incomplete,
        defect_escape_count=defect_escape,
        normal_reject_count=normal_reject,
        defect_escape_rate=defect_escape / defect,
        normal_reject_rate=normal_reject / normal,
        review_rate=review / part_count,
        evaluation_complete=incomplete == 0,
    )


def test_loads_exact_canonical_versioned_policy_and_pass_decision() -> None:
    content = canonical_json_bytes(_policy_payload())
    policy = load_heldout_acceptance_policy_bytes(content)

    decision = evaluate_heldout_acceptance(
        policy,
        _metrics(),
        metrics_sha256="1" * 64,
    )

    assert policy.policy_version == "1.0.0"
    assert decision.policy_sha256 == policy.sha256
    assert decision.normal_heldout_parts == 10
    assert load_heldout_acceptance_decision_bytes(
        decision.canonical_bytes()
    ) == decision


def test_policy_loader_rejects_noncanonical_or_unknown_json() -> None:
    with pytest.raises(ValueError, match="canonical JSON"):
        load_heldout_acceptance_policy_bytes(
            json.dumps(_policy_payload(), indent=2).encode("utf-8")
        )

    payload = _policy_payload()
    payload["override"] = True
    with pytest.raises(ValueError, match="missing or unknown"):
        load_heldout_acceptance_policy_bytes(canonical_json_bytes(payload))


@pytest.mark.parametrize(
    ("policy", "metrics", "reason"),
    [
        (_policy_payload(), _metrics(defect_escape=1), "defect escape rate"),
        (_policy_payload(), _metrics(normal_reject=1), "normal reject rate"),
        (_policy_payload(), _metrics(review=1), "review rate"),
        (_policy_payload(min_normal=11), _metrics(), "normal held-out parts"),
        (_policy_payload(min_defect=11), _metrics(), "defect held-out parts"),
    ],
)
def test_policy_breach_fails_closed_without_a_pass_decision(
    policy: dict[str, object],
    metrics: HeldOutPartMetrics,
    reason: str,
) -> None:
    parsed = HeldOutAcceptancePolicy.from_mapping(policy)

    with pytest.raises(ValueError, match=reason):
        evaluate_heldout_acceptance(parsed, metrics, metrics_sha256="2" * 64)


def test_excluded_branch_incomplete_evaluation_is_not_implicitly_accepted() -> None:
    policy = HeldOutAcceptancePolicy.from_mapping(_policy_payload())

    with pytest.raises(ValueError, match="evaluation is incomplete"):
        evaluate_heldout_acceptance(
            policy,
            _metrics(incomplete=1),
            metrics_sha256="3" * 64,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_defect_escape_rate", -0.01),
        ("max_normal_reject_rate", 1.01),
        ("max_review_rate", float("nan")),
        ("min_normal_heldout_parts", 0),
        ("min_defect_heldout_parts", True),
    ],
)
def test_policy_rejects_invalid_limits(field: str, value: object) -> None:
    payload = _policy_payload()
    limits = payload["limits"]
    assert isinstance(limits, dict)
    limits[field] = value

    with pytest.raises(ValueError, match="held-out acceptance"):
        HeldOutAcceptancePolicy.from_mapping(payload)
