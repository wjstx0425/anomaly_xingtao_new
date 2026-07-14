# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Linux/GPU construction and strict-state tests for non-PatchCore families."""

from __future__ import annotations

import hashlib
import stat
import sys
from pathlib import Path

import pytest

import zs32_inspection.models.anomalib_backend as backend
from zs32_inspection.domain.contracts import AnomalyFamily
from zs32_inspection.domain.identity import Hand, PartIdentity, RoiSample
from zs32_inspection.models.base import ModelInput


pytestmark = (
    pytest.mark.gpu,
    pytest.mark.skipif(sys.platform != "linux", reason="Anomalib family tests are Linux-only"),
)


def _common(family: str, *, image_size: tuple[int, int]) -> dict[str, object]:
    return {
        "schema": "zs32.anomalib_backend",
        "schema_version": 2,
        "source_parameters_sha256": "1" * 64,
        "family": family,
        "slot": {"hand": "right", "view": "front"},
        "image_size": list(image_size),
        "normalization": "none" if family == "efficientad" else "imagenet",
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
    }


def _efficientad_parameters() -> dict[str, object]:
    return {
        **_common("efficientad", image_size=(256, 256)),
        "model": {
            "teacher_out_channels": 384,
            "model_size": "small",
            "lr": 0.0001,
            "weight_decay": 0.00001,
            "padding": False,
            "pad_maps": True,
        },
        "auxiliary": {
            "imagenet_tree_sha256": "3" * 64,
            "teacher_weights_sha256": "4" * 64,
        },
    }


def _anomalydino_parameters() -> dict[str, object]:
    return {
        **_common("anomalydino", image_size=(252, 252)),
        "model": {
            "num_neighbours": 1,
            "encoder_name": "dinov2_vit_small_14",
            "masking": False,
            "coreset_subsampling": False,
            "sampling_ratio": 0.1,
            "precision": "float32",
        },
        "auxiliary": {"encoder_weights_sha256": "5" * 64},
    }


@pytest.mark.parametrize(
    ("family", "parameters"),
    (
        (AnomalyFamily.EFFICIENTAD, _efficientad_parameters()),
        (AnomalyFamily.ANOMALYDINO, _anomalydino_parameters()),
    ),
)
def test_non_patchcore_family_constructs_and_strictly_restores_state(
    family: AnomalyFamily,
    parameters: dict[str, object],
    tmp_path: Path,
) -> None:
    import torch

    source = backend._build_model(family, parameters, for_training=False)
    checkpoint = tmp_path / f"{family.value}.pt"
    torch.save({"state_dict": source.state_dict()}, checkpoint)
    serialized = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored = backend._build_model(family, parameters, for_training=False)

    incompatibilities = restored.load_state_dict(serialized["state_dict"], strict=True)

    assert not incompatibilities.missing_keys
    assert not incompatibilities.unexpected_keys
    assert tuple(source.state_dict()) == tuple(restored.state_dict())


@pytest.mark.parametrize(
    ("family", "parameters"),
    (
        (AnomalyFamily.EFFICIENTAD, _efficientad_parameters()),
        (AnomalyFamily.ANOMALYDINO, _anomalydino_parameters()),
    ),
)
def test_non_patchcore_preprocessor_has_no_spatial_crop(
    family: AnomalyFamily,
    parameters: dict[str, object],
) -> None:
    model = backend._build_model(family, parameters, for_training=False)
    transform = model.pre_processor.transform
    names = tuple(type(item).__name__ for item in transform.transforms)

    assert names[0] == "Resize"
    assert all("Crop" not in name for name in names)


def _model_input(tmp_path: Path) -> ModelInput:
    import cv2
    import numpy as np

    crop = tmp_path / "crop.png"
    assert cv2.imwrite(str(crop), np.full((20, 30, 3), 64, dtype=np.uint8))
    crop_sha256 = hashlib.sha256(crop.read_bytes()).hexdigest()
    sample = RoiSample(
        part=PartIdentity("part-visual-001", Hand.RIGHT),
        capture_set_id="capture-visual-001",
        view_id="front",
        source_sha256="8" * 64,
        crop_sha256=crop_sha256,
        roi_config_id="roi-visual-v1",
        crop_width=30,
        crop_height=20,
    )
    return ModelInput(sample, crop, "9" * 64)


def test_anomaly_visual_renderer_outputs_distinct_deterministic_private_pngs(
    tmp_path: Path,
) -> None:
    import cv2
    import torch

    sample = _model_input(tmp_path)
    anomaly_map = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    first = backend._render_anomaly_visuals(
        anomaly_map,
        sample,
        expected_size=(4, 5),
        inspection_id="inspection-visual-001",
        scratch_root=tmp_path / "scratch-1",
    )
    second = backend._render_anomaly_visuals(
        anomaly_map,
        sample,
        expected_size=(4, 5),
        inspection_id="inspection-visual-001",
        scratch_root=tmp_path / "scratch-2",
    )

    heatmap_path, heatmap_sha256, overlay_path, overlay_sha256 = first
    assert heatmap_path.read_bytes() == second[0].read_bytes()
    assert overlay_path.read_bytes() == second[2].read_bytes()
    assert heatmap_sha256 == hashlib.sha256(heatmap_path.read_bytes()).hexdigest()
    assert overlay_sha256 == hashlib.sha256(overlay_path.read_bytes()).hexdigest()
    assert heatmap_sha256 != overlay_sha256
    assert stat.S_IMODE(heatmap_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(overlay_path.stat().st_mode) == 0o600
    assert cv2.imread(str(heatmap_path)).shape[:2] == (20, 30)
    assert cv2.imread(str(overlay_path)).shape[:2] == (20, 30)


@pytest.mark.parametrize("failure", ["missing", "shape", "nan"])
def test_anomaly_visual_renderer_fails_closed_on_invalid_map(
    tmp_path: Path,
    failure: str,
) -> None:
    import torch

    sample = _model_input(tmp_path)
    anomaly_map = {
        "missing": None,
        "shape": torch.zeros((3, 5), dtype=torch.float32),
        "nan": torch.full((4, 5), float("nan"), dtype=torch.float32),
    }[failure]

    with pytest.raises(backend.AnomalibBackendError):
        backend._render_anomaly_visuals(
            anomaly_map,
            sample,
            expected_size=(4, 5),
            inspection_id=f"inspection-{failure}",
            scratch_root=tmp_path / "scratch",
        )
