"""Linux-authoritative fault injection for YOLO batch path identity."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from zs32_inspection.domain.identity import Hand, PartIdentity, RoiSample
from zs32_inspection.models.base import DeviceSpec, ModelContractError, ModelInput
from zs32_inspection.models.ultralytics_backend import UltralyticsBatchPredictor
from zs32_inspection.models.yolo import YoloRuntimeSettings

SHA256 = "a" * 64


class _EmptyBoxes:
    xyxy: tuple[object, ...] = ()
    conf: tuple[object, ...] = ()
    cls: tuple[object, ...] = ()


class _Result:
    def __init__(self, path: Path) -> None:
        self.path = str(path)
        self.orig_shape = (32, 48)
        self.boxes = _EmptyBoxes()


class _FaultInjectingModel:
    def __init__(self, returned_paths: tuple[Path, ...]) -> None:
        self.returned_paths = returned_paths
        self.received_sources: tuple[str, ...] = ()

    def predict(self, *, source: list[str], **_kwargs: object) -> tuple[_Result, ...]:
        self.received_sources = tuple(source)
        return tuple(_Result(path) for path in self.returned_paths)


def _inputs(tmp_path: Path) -> tuple[ModelInput, ModelInput]:
    part = PartIdentity(part_instance_id="part-1", hand=Hand.RIGHT)
    rows: list[ModelInput] = []
    for view, payload in (("front", b"front-crop"), ("back", b"back-crop")):
        path = tmp_path / f"{view}.png"
        path.write_bytes(payload)
        rows.append(
            ModelInput(
                sample=RoiSample(
                    part=part,
                    capture_set_id="capture-1",
                    view_id=view,
                    source_sha256=SHA256,
                    crop_sha256=hashlib.sha256(payload).hexdigest(),
                    roi_config_id="roi-v1",
                    crop_width=48,
                    crop_height=32,
                ),
                crop_path=path,
                roi_digest=SHA256,
            )
        )
    return rows[0], rows[1]


def _predictor(model: _FaultInjectingModel) -> UltralyticsBatchPredictor:
    spec = SimpleNamespace(runtime=YoloRuntimeSettings(), model_digest=SHA256)
    return UltralyticsBatchPredictor(model, spec, DeviceSpec())  # type: ignore[arg-type]


def test_ultralytics_batch_accepts_exact_path_identity_and_preserves_view_order(
    tmp_path: Path,
) -> None:
    samples = _inputs(tmp_path)
    paths = tuple(item.crop_path for item in samples)
    model = _FaultInjectingModel(paths)

    scores = _predictor(model).predict_batch(samples, inspection_id="inspection-1")

    assert model.received_sources == tuple(str(path.resolve()) for path in paths)
    assert [score.view for score in scores] == ["front", "back"]


def test_ultralytics_batch_rejects_reversed_result_paths(tmp_path: Path) -> None:
    samples = _inputs(tmp_path)
    paths = tuple(item.crop_path for item in samples)
    model = _FaultInjectingModel(tuple(reversed(paths)))

    with pytest.raises(ModelContractError, match="path order differs"):
        _predictor(model).predict_batch(samples, inspection_id="inspection-1")


def test_ultralytics_batch_rejects_duplicate_result_paths(tmp_path: Path) -> None:
    samples = _inputs(tmp_path)
    duplicate = samples[0].crop_path
    model = _FaultInjectingModel((duplicate, duplicate))

    with pytest.raises(ModelContractError, match="duplicate result paths"):
        _predictor(model).predict_batch(samples, inspection_id="inspection-1")


def test_ultralytics_batch_rejects_missing_result_path(tmp_path: Path) -> None:
    samples = _inputs(tmp_path)
    model = _FaultInjectingModel((samples[0].crop_path,))

    with pytest.raises(ModelContractError, match=r"path set mismatch; missing=\[.+back\.png"):
        _predictor(model).predict_batch(samples, inspection_id="inspection-1")


def test_ultralytics_batch_rejects_extra_result_path(tmp_path: Path) -> None:
    samples = _inputs(tmp_path)
    extra = tmp_path / "extra.png"
    extra.write_bytes(b"extra")
    model = _FaultInjectingModel(
        (samples[0].crop_path, samples[1].crop_path, extra)
    )

    with pytest.raises(ModelContractError, match=r"path set mismatch; missing=\[\], extra=\[.+extra\.png"):
        _predictor(model).predict_batch(samples, inspection_id="inspection-1")
