"""Linux-only contract tests for canonical manual promotion receipts."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from zs32_inspection.runtime.promotion_receipt import (
    PromotionReceipt,
    load_promotion_receipt,
)
from zs32_inspection.runtime.publisher import canonical_json_bytes
from zs32_inspection.runtime.release_assembler import (
    ReleaseAssemblySpec,
    _load_and_validate_promotion_receipt,
)


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="ZS32 verification is Linux-only")


def _payload() -> dict[str, object]:
    return {
        "schema": "zs32.promotion_receipt",
        "schema_version": 2,
        "product": "ZS32",
        "release_id": "release-v1",
        "promotion_id": "promotion-v1",
        "approver": "quality-approver",
        "approved_at": "2026-07-14T10:00:00+08:00",
        "candidate_id": "candidate-v1",
        "candidate_digest": "1" * 64,
        "validation_publication_id": "validation-v1",
        "validation_publication_root_sha256": "2" * 64,
        "validation_record_sha256": "3" * 64,
        "contract_sha256": "4" * 64,
        "calibration_artifact_sha256": "5" * 64,
        "dataset_manifest_sha256": "6" * 64,
        "heldout_acceptance_policy_sha256": "a" * 64,
        "heldout_acceptance_decision_sha256": "b" * 64,
        "evidence": {
            "golden_parity": {
                "status": "PASS",
                "evidence_sha256": "7" * 64,
                "reason": None,
            },
            "fat": {
                "status": "PASS",
                "evidence_sha256": "8" * 64,
                "reason": None,
            },
            "sat": {
                "status": "NOT_APPLICABLE",
                "evidence_sha256": None,
                "reason": "SAT is explicitly outside this laboratory release gate",
            },
        },
    }


def _assembly_spec(
    path: Path,
    *,
    created_at: str = "2026-07-14T10:01:00+08:00",
) -> ReleaseAssemblySpec:
    payload = path.read_bytes()
    return cast(
        ReleaseAssemblySpec,
        SimpleNamespace(
            promotion_receipt_path=path,
            promotion_receipt_sha256=hashlib.sha256(payload).hexdigest(),
            release_id="release-v1",
            created_at=created_at,
            validation_publication_id="validation-v1",
            validation_publication_root_sha256="2" * 64,
            validation_record_sha256="3" * 64,
            heldout_acceptance_policy_sha256="a" * 64,
            heldout_acceptance_decision_sha256="b" * 64,
            contract=SimpleNamespace(contract_sha256="4" * 64),
            assets=SimpleNamespace(
                candidate=SimpleNamespace(
                    candidate_id="candidate-v1",
                    digest="1" * 64,
                ),
                calibration_artifact_digest="5" * 64,
            ),
            dataset_release_manifest_sha256="6" * 64,
        ),
    )


def test_loads_exact_canonical_receipt_and_preserves_evidence() -> None:
    payload = _payload()
    receipt = PromotionReceipt.from_mapping(payload)

    assert receipt.canonical_bytes() == canonical_json_bytes(payload)
    assert receipt.evidence["golden_parity"].status == "PASS"
    assert receipt.evidence["sat"].status == "NOT_APPLICABLE"


@pytest.mark.parametrize(
    ("role", "status", "digest", "reason"),
    [
        ("golden_parity", "PASS", None, None),
        ("golden_parity", "NOT_APPLICABLE", None, "not applicable"),
        ("fat", "NOT_APPLICABLE", "8" * 64, "not required"),
        ("sat", "NOT_RUN", None, "not run"),
    ],
)
def test_rejects_ambiguous_or_nonpassing_evidence(
    role: str,
    status: str,
    digest: str | None,
    reason: str | None,
) -> None:
    payload = _payload()
    evidence = payload["evidence"]
    assert isinstance(evidence, dict)
    evidence[role] = {
        "status": status,
        "evidence_sha256": digest,
        "reason": reason,
    }

    with pytest.raises(ValueError, match="promotion"):
        PromotionReceipt.from_mapping(payload)


def test_file_loader_rejects_noncanonical_bytes(tmp_path: Path) -> None:
    path = tmp_path / "promotion_receipt.json"
    path.write_text(json.dumps(_payload(), indent=2), encoding="utf-8")
    path.chmod(0o444)

    with pytest.raises(ValueError, match="canonical JSON"):
        load_promotion_receipt(path)


def test_file_loader_rejects_writable_receipt(tmp_path: Path) -> None:
    path = tmp_path / "promotion_receipt.json"
    path.write_bytes(canonical_json_bytes(_payload()))
    path.chmod(0o644)

    with pytest.raises(ValueError, match="read-only"):
        load_promotion_receipt(path)


def test_rejects_receipt_without_timezone() -> None:
    payload = _payload()
    payload["approved_at"] = "2026-07-14T10:00:00"

    with pytest.raises(ValueError, match="explicit timezone"):
        PromotionReceipt.from_mapping(payload)


def test_rejects_unknown_receipt_field() -> None:
    payload = _payload()
    payload["unreviewed_override"] = True

    with pytest.raises(ValueError, match="missing or unknown fields"):
        PromotionReceipt.from_mapping(payload)


def test_rejects_legacy_receipt_without_acceptance_provenance() -> None:
    payload = _payload()
    payload["schema_version"] = 1
    payload.pop("heldout_acceptance_policy_sha256")
    payload.pop("heldout_acceptance_decision_sha256")

    with pytest.raises(ValueError, match="missing or unknown fields"):
        PromotionReceipt.from_mapping(payload)


def test_assembler_binding_accepts_exact_receipt(tmp_path: Path) -> None:
    path = tmp_path / "promotion_receipt.json"
    path.write_bytes(canonical_json_bytes(_payload()))
    path.chmod(0o444)

    receipt = _load_and_validate_promotion_receipt(_assembly_spec(path))

    assert receipt.promotion_id == "promotion-v1"


@pytest.mark.parametrize(
    ("field", "forged"),
    [
        ("release_id", "other-release"),
        ("candidate_id", "other-candidate"),
        ("candidate_digest", "0" * 64),
        ("validation_publication_id", "other-validation"),
        ("validation_publication_root_sha256", "0" * 64),
        ("validation_record_sha256", "0" * 64),
        ("contract_sha256", "0" * 64),
        ("calibration_artifact_sha256", "0" * 64),
        ("dataset_manifest_sha256", "0" * 64),
        ("heldout_acceptance_policy_sha256", "0" * 64),
        ("heldout_acceptance_decision_sha256", "0" * 64),
    ],
)
def test_assembler_binding_rejects_identity_mismatch(
    tmp_path: Path,
    field: str,
    forged: str,
) -> None:
    payload = _payload()
    payload[field] = forged
    path = tmp_path / "promotion_receipt.json"
    path.write_bytes(canonical_json_bytes(payload))
    path.chmod(0o444)

    with pytest.raises(ValueError, match="differs from release/candidate/validation"):
        _load_and_validate_promotion_receipt(_assembly_spec(path))


def test_assembler_binding_rejects_approval_after_release_time(tmp_path: Path) -> None:
    path = tmp_path / "promotion_receipt.json"
    path.write_bytes(canonical_json_bytes(_payload()))
    path.chmod(0o444)

    with pytest.raises(ValueError, match="cannot approve a release after"):
        _load_and_validate_promotion_receipt(
            _assembly_spec(path, created_at="2026-07-14T09:59:00+08:00")
        )
