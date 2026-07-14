"""Linux/NVIDIA contract tests for explicit PatchCore backbone initialization."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import zs32_inspection.models.anomalib_backend as backend
from tests.unit.zs32_refactor.execution_fixtures import execution_receipt_mapping
from zs32_inspection.domain.contracts import AnomalyFamily
from zs32_inspection.models import ModelContractError
from zs32_inspection.models.base import sha256_file


def _parameters(path: Path, digest: str, *, backbone: str = "wide_resnet50_2") -> dict:
    return {
        "schema": "zs32.anomalib_backend",
        "schema_version": 2,
        "source_parameters_sha256": "7" * 64,
        "family": "patchcore",
        "slot": {"hand": "right", "view": "front_0"},
        "image_size": [256, 256],
        "normalization": "imagenet",
        "model": {
            "backbone": backbone,
            "layers": ["layer2", "layer3"],
            "pre_trained": False,
            "initialization": "explicit_state_dict",
            "coreset_sampling_ratio": 0.1,
            "num_neighbors": 9,
            "precision": "float32",
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
            "materialized_manifest_sha256": "6" * 64,
            "selected_train_normal_count": 1,
        },
        "auxiliary": {
            "backbone_weights_asset_id": (
                "imagenet1k-wide_resnet50_2-anomalib-timm-feature-extractor-v1"
            ),
            "backbone_weights_path": str(path),
            "backbone_weights_sha256": digest,
            "state_dict_scope": "anomalib_timm_feature_extractor",
        },
    }


def _with_layers(parameters: dict, layers: list[str]) -> dict:
    updated = {**parameters, "model": {**parameters["model"], "layers": layers}}
    return updated


def _export_wrapper_state_dict(path: Path, parameters: dict) -> object:
    import torch

    model = backend._build_model(AnomalyFamily.PATCHCORE, parameters, for_training=False)
    torch.save(model.model.feature_extractor.state_dict(), path)
    return model


@pytest.mark.gpu
def test_patchcore_wrapper_state_dict_strict_roundtrip(tmp_path: Path) -> None:
    import torch

    path = tmp_path / "wrapper-state-dict.pt"
    source = _export_wrapper_state_dict(path, _parameters(path, "0" * 64))
    parameters = _parameters(path, sha256_file(path))
    restored = backend._build_model(
        AnomalyFamily.PATCHCORE,
        parameters,
        for_training=True,
    )
    source_state = source.model.feature_extractor.state_dict()
    restored_state = restored.model.feature_extractor.state_dict()
    assert tuple(source_state) == tuple(restored_state)
    assert all(torch.equal(source_state[key], restored_state[key]) for key in source_state)


@pytest.mark.gpu
def test_patchcore_rejects_inner_backbone_key_scope(tmp_path: Path) -> None:
    import torch

    path = tmp_path / "inner-state-dict.pt"
    source = backend._build_model(
        AnomalyFamily.PATCHCORE,
        _parameters(path, "0" * 64),
        for_training=False,
    )
    wrapper = source.model.feature_extractor
    torch.save(wrapper.feature_extractor.state_dict(), path)
    with pytest.raises(ModelContractError, match="does not exactly match"):
        backend._build_model(
            AnomalyFamily.PATCHCORE,
            _parameters(path, sha256_file(path)),
            for_training=True,
        )


@pytest.mark.gpu
def test_patchcore_rejects_state_dict_from_wrong_backbone(tmp_path: Path) -> None:
    path = tmp_path / "wrong-backbone-state-dict.pt"
    _export_wrapper_state_dict(path, _parameters(path, "0" * 64, backbone="resnet18"))
    with pytest.raises(ModelContractError, match="does not exactly match"):
        backend._build_model(
            AnomalyFamily.PATCHCORE,
            _parameters(path, sha256_file(path)),
            for_training=True,
        )


@pytest.mark.gpu
def test_patchcore_rejects_layers_silently_removed_by_anomalib(tmp_path: Path) -> None:
    path = tmp_path / "unused-state-dict.pt"
    parameters = _with_layers(_parameters(path, "0" * 64), ["not_a_real_layer"])
    with pytest.raises(ModelContractError, match="silently changed"):
        backend._build_model(
            AnomalyFamily.PATCHCORE,
            parameters,
            for_training=False,
        )


@pytest.mark.gpu
def test_patchcore_constructor_never_requests_pretrained_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import timm

    observed: list[object] = []
    original = timm.create_model

    def guarded_create_model(*args, **kwargs):
        observed.append(kwargs.get("pretrained"))
        if kwargs.get("pretrained") is not False:
            raise AssertionError("PatchCore attempted implicit pretrained resolution")
        return original(*args, **kwargs)

    monkeypatch.setattr(timm, "create_model", guarded_create_model)
    backend._build_model(
        AnomalyFamily.PATCHCORE,
        _parameters(tmp_path / "unused-state-dict.pt", "0" * 64),
        for_training=False,
    )
    assert observed and all(value is False for value in observed)


def test_patchcore_rehashes_the_exact_bytes_passed_to_torch_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "state-dict.pt"
    path.write_bytes(b"approved bytes")
    parameters = _parameters(path, sha256_file(path))

    def swap_after_first_verification(_parameters):
        path.write_bytes(b"changed after first verification")
        return path

    monkeypatch.setattr(backend, "_verify_patchcore_backbone_asset", swap_after_first_verification)
    with pytest.raises(ModelContractError, match="between verification and load"):
        backend._load_patchcore_backbone_weights(SimpleNamespace(), parameters)


def test_patchcore_training_end_rehash_rejects_changed_asset(tmp_path: Path) -> None:
    path = tmp_path / "state-dict.pt"
    path.write_bytes(b"approved bytes")
    parameters = _parameters(path, sha256_file(path))
    path.write_bytes(b"changed during training")
    with pytest.raises(ModelContractError, match="changed during training"):
        backend._verify_patchcore_backbone_asset(parameters)


def test_patchcore_published_parameters_remove_training_host_path(tmp_path: Path) -> None:
    parameters = _parameters(tmp_path / "private-host-path.pt", "5" * 64)
    published = backend._published_parameters(AnomalyFamily.PATCHCORE, parameters)
    assert "backbone_weights_path" not in published["auxiliary"]
    assert published["auxiliary"]["backbone_weights_sha256"] == "5" * 64
    assert published["model"]["pre_trained"] is False
    assert published["model"]["initialization"] == "explicit_state_dict"


@pytest.mark.parametrize(
    ("receipt_parameter_digest", "receipt_backbone_digest", "message"),
    (
        ("8" * 64, "5" * 64, "parameters_sha256"),
        ("7" * 64, "9" * 64, "explicit backbone"),
    ),
)
def test_patchcore_receipt_must_bind_parameters_and_backbone(
    tmp_path: Path,
    receipt_parameter_digest: str,
    receipt_backbone_digest: str,
    message: str,
) -> None:
    parameters = _parameters(tmp_path / "state-dict.pt", "5" * 64)
    artifact = SimpleNamespace(
        family=AnomalyFamily.PATCHCORE,
        execution_receipt=execution_receipt_mapping(
            "train_anomaly",
            input_sha256_by_role={"patchcore_backbone": receipt_backbone_digest},
            parameters_sha256=receipt_parameter_digest,
        ),
    )
    with pytest.raises(ModelContractError, match=message):
        backend._validate_execution_receipt_binding(artifact, parameters)


def test_patchcore_logical_asset_id_cannot_leak_a_host_path() -> None:
    with pytest.raises(ModelContractError, match="opaque logical ID"):
        backend._logical_asset_id("/home/operator/private/model.pt", "asset_id")
