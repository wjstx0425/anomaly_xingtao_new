# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Linux test definitions for the ZS32 model adapter contracts."""

from __future__ import annotations

import hashlib

import pytest

from tests.unit.zs32_refactor.execution_fixtures import execution_receipt_mapping
from zs32_inspection.models import (
    AnomalyDINOAdapter,
    AnomalyFamily,
    AnomalyModelArtifact,
    AssetFile,
    BackendUnavailableError,
    DeviceSpec,
    EfficientADAdapter,
    ModelContractError,
    ModelSlot,
    PatchCoreAdapter,
)

HEX = "a" * 64


def _artifact(tmp_path, family: AnomalyFamily) -> AnomalyModelArtifact:
    checkpoint = tmp_path / f"{family.value}.ckpt"
    checkpoint.write_bytes(family.value.encode())
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return AnomalyModelArtifact(
        family=family,
        slot=ModelSlot("right", "front"),
        checkpoint=AssetFile("checkpoint", checkpoint, digest),
        model_digest=digest,
        dataset_release_id="dataset-v1",
        dataset_manifest_digest=HEX,
        train_split_id="train-v1",
        recipe_digest=HEX,
        roi_version="roi-v1",
        roi_digest=HEX,
        framework_version="anomalib-test",
        training_parameters={},
        execution_receipt=execution_receipt_mapping("train_anomaly"),
    )


@pytest.mark.parametrize(
    ("adapter", "family"),
    [
        (PatchCoreAdapter(), AnomalyFamily.PATCHCORE),
        (EfficientADAdapter(), AnomalyFamily.EFFICIENTAD),
        (AnomalyDINOAdapter(), AnomalyFamily.ANOMALYDINO),
    ],
)
def test_three_adapters_share_one_fail_closed_contract(tmp_path, adapter, family) -> None:
    artifact = _artifact(tmp_path, family)
    with pytest.raises(BackendUnavailableError, match="Linux GPU integration"):
        adapter.load({artifact.slot: artifact}, DeviceSpec())


def test_adapter_rejects_mixed_family_before_backend(tmp_path) -> None:
    artifact = _artifact(tmp_path, AnomalyFamily.EFFICIENTAD)
    called = False

    def loader(_artifacts, _device):
        nonlocal called
        called = True
        return object()

    with pytest.raises(ModelContractError, match="cannot contain"):
        PatchCoreAdapter(runtime_loader=loader).load({artifact.slot: artifact}, DeviceSpec())
    assert called is False


def test_adapter_verifies_checkpoint_before_backend(tmp_path) -> None:
    artifact = _artifact(tmp_path, AnomalyFamily.PATCHCORE)
    artifact.checkpoint.path.write_bytes(b"changed")
    called = False

    def loader(_artifacts, _device):
        nonlocal called
        called = True
        return object()

    with pytest.raises(ModelContractError, match="SHA-256 mismatch"):
        PatchCoreAdapter(runtime_loader=loader).load({artifact.slot: artifact}, DeviceSpec())
    assert called is False
