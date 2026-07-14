# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Deployment-metadata boundaries shared by the concrete Anomalib backend."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import zs32_inspection.models.anomalib_backend as backend
from zs32_inspection.domain.contracts import AnomalyFamily
from zs32_inspection.models.base import ModelContractError, ModelSlot


def _efficientad_parameters() -> dict[str, object]:
    return {
        "schema": "zs32.anomalib_backend",
        "schema_version": 2,
        "source_parameters_sha256": "1" * 64,
        "family": "efficientad",
        "slot": {"hand": "right", "view": "front"},
        "image_size": [256, 256],
        "normalization": "none",
        "model": {
            "teacher_out_channels": 384,
            "model_size": "small",
            "lr": 0.0001,
            "weight_decay": 0.00001,
            "padding": False,
            "pad_maps": True,
        },
        "trainer": {
            "max_epochs": 1,
            "precision": "32-true",
            "train_batch_size": 1,
            "num_workers": 0,
            "seed": 43,
            "deterministic": True,
        },
        "dataset": {
            "materialized_manifest_sha256": "2" * 64,
            "selected_train_normal_count": 1,
        },
        "auxiliary": {
            "imagenet_dir": "/DATA/private/operator/imagenette2",
            "imagenet_tree_sha256": "3" * 64,
            "teacher_weights_sha256": "4" * 64,
        },
    }


def test_efficientad_published_parameters_remove_training_host_path() -> None:
    parameters = _efficientad_parameters()

    published = backend._published_parameters(AnomalyFamily.EFFICIENTAD, parameters)

    assert set(published["auxiliary"]) == {
        "imagenet_tree_sha256",
        "teacher_weights_sha256",
    }
    assert published["auxiliary"]["imagenet_tree_sha256"] == "3" * 64
    assert published["auxiliary"]["teacher_weights_sha256"] == "4" * 64
    assert "/DATA/private/operator" not in json.dumps(published, sort_keys=True)
    assert parameters["auxiliary"]["imagenet_dir"] == "/DATA/private/operator/imagenette2"


def test_efficientad_runtime_metadata_accepts_only_content_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published = backend._published_parameters(
        AnomalyFamily.EFFICIENTAD,
        _efficientad_parameters(),
    )
    artifact = SimpleNamespace(
        family=AnomalyFamily.EFFICIENTAD,
        slot=ModelSlot("right", "front"),
        training_parameters=published,
    )
    monkeypatch.setattr(
        backend,
        "_validate_execution_receipt_binding",
        lambda _artifact, _parameters: None,
    )

    validated, image_size = backend._validate_runtime_parameters(artifact)

    assert image_size == (256, 256)
    assert validated["auxiliary"] == {
        "imagenet_tree_sha256": "3" * 64,
        "teacher_weights_sha256": "4" * 64,
    }


def test_efficientad_runtime_metadata_rejects_reintroduced_host_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = _efficientad_parameters()
    artifact = SimpleNamespace(
        family=AnomalyFamily.EFFICIENTAD,
        slot=ModelSlot("right", "front"),
        training_parameters=parameters,
    )
    monkeypatch.setattr(
        backend,
        "_validate_execution_receipt_binding",
        lambda _artifact, _parameters: None,
    )

    with pytest.raises(ModelContractError, match="auxiliary metadata keys"):
        backend._validate_runtime_parameters(artifact)
