# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the single-path ZS32 Demo runtime."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import capture_data.zs32_demo_runtime as demo_runtime
from capture_data.zs32_demo_runtime import (
    DemoRuntimeError,
    DemoTemplateMatcher,
    DemoTemplateScore,
    ZS32DemoRuntime,
    fuse_demo_status,
)
from capture_data.zs32_model_runtime import ModelEvidence, PatchcoreArtifacts
from zs32_inspection.dashboard.parser import load_inspection_result
from zs32_inspection.domain.views import VIEW_ORDER


def _config(tmp_path: Path, *, template_threshold: float = 0.5) -> SimpleNamespace:
    checkpoint_paths: dict[str, Path] = {}
    for view in VIEW_ORDER:
        checkpoint = tmp_path / f"{view}.ckpt"
        checkpoint.write_bytes(view.encode())
        checkpoint_paths[view] = checkpoint
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"weights")
    template_dir = tmp_path / "template"
    template_dir.mkdir(exist_ok=True)
    roi_config = tmp_path / "roi.json"
    roi_config.write_text("{}\n", encoding="utf-8")
    topology = tmp_path / "topology.json"
    topology.write_text("{}\n", encoding="utf-8")
    thresholds = SimpleNamespace(
        template={view: template_threshold for view in VIEW_ORDER},
        patchcore={view: 0.5 for view in VIEW_ORDER},
        yolo={view: 0.5 for view in VIEW_ORDER},
    )
    return SimpleNamespace(
        path=tmp_path / "demo.json",
        topology=topology,
        roi_config=roi_config,
        template_dir=template_dir,
        patchcore={view: SimpleNamespace(checkpoint=checkpoint_paths[view]) for view in VIEW_ORDER},
        yolo=SimpleNamespace(
            weights=weights,
            imgsz=1280,
            candidate_conf=0.001,
            iou=0.7,
            max_det=300,
            class_map={0: "defect"},
        ),
        thresholds=thresholds,
        patchcore_process_count=1,
        image_width=12,
        image_height=10,
        rois={view: (0, 0, 6, 5) for view in VIEW_ORDER},
    )


def _images(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for index, view in enumerate(VIEW_ORDER):
        image = np.full((10, 12, 3), index + 1, dtype=np.uint8)
        path = tmp_path / f"{view}.png"
        assert cv2.imwrite(str(path), image)
        paths[view] = path
    return paths


class _Template:
    def __init__(self, template_dir: Path, *, ng_view: str | None = None) -> None:
        self.template_dir = template_dir
        self.ng_view = ng_view
        self.calls: list[str] = []

    def score_array(self, image: np.ndarray, view: str) -> DemoTemplateScore:
        assert image.shape == (5, 6)
        assert image.flags.c_contiguous
        self.calls.append(view)
        evidence = self.template_dir / f"{view}.png"
        assert cv2.imwrite(str(evidence), np.full((5, 6, 3), 10, dtype=np.uint8))
        risk = 0.9 if view == self.ng_view else 0.1
        return DemoTemplateScore(risk, 1.0 - risk, evidence, (0, 0))


class _Patchcore:
    def __init__(self, *, error_view: str | None = None) -> None:
        self.error_view = error_view
        self.calls: list[tuple[str, ...]] = []
        self.close_calls = 0

    def predict_all(
        self,
        crops: dict[str, Path],
        evidence_dir: Path,
        *,
        diagnostic_mask_threshold: float = 0.65,
    ) -> dict[str, ModelEvidence | Exception]:
        del diagnostic_mask_threshold
        self.calls.append(tuple(crops))
        output: dict[str, ModelEvidence | Exception] = {}
        for view in VIEW_ORDER:
            if view == self.error_view:
                output[view] = RuntimeError("patchcore failed")
                continue
            evidence_path = evidence_dir / f"{view}.png"
            raw_path = evidence_dir / "raw_maps" / f"{view}.npy"
            mask_path = evidence_dir / "masks" / f"{view}.png"
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(evidence_path), np.zeros((5, 6, 3), dtype=np.uint8))
            np.save(raw_path, np.zeros((5, 6), dtype=np.float32), allow_pickle=False)
            assert cv2.imwrite(str(mask_path), np.zeros((5, 6), dtype=np.uint8))
            output[view] = ModelEvidence(
                score=0.2,
                evidence_path=evidence_path,
                patchcore_artifacts=PatchcoreArtifacts(
                    raw_anomaly_map_path=raw_path,
                    mask_path=mask_path,
                    mask_source="pred_mask",
                    diagnostic_mask_threshold=None,
                    raw_anomaly_map_shape=(5, 6),
                    mask_shape=(5, 6),
                ),
            )
        return output

    def close(self) -> None:
        self.close_calls += 1


class _Yolo:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def predict(self, crops: dict[str, np.ndarray], evidence_dir: Path) -> dict[str, ModelEvidence]:
        self.calls.append(tuple(crops))
        evidence_dir.mkdir(parents=True, exist_ok=True)
        output: dict[str, ModelEvidence] = {}
        for view in VIEW_ORDER:
            evidence_path = evidence_dir / f"{view}.png"
            assert cv2.imwrite(str(evidence_path), np.zeros((5, 6, 3), dtype=np.uint8))
            output[view] = ModelEvidence(score=0.0, evidence_path=evidence_path, detections=())
        return output


def _runtime(
    config: SimpleNamespace,
    *,
    template: _Template,
    patchcore: _Patchcore,
    yolo: _Yolo,
    loader: object | None = None,
) -> ZS32DemoRuntime:
    return ZS32DemoRuntime(
        config,
        template_matcher=template,
        patchcore_backend=patchcore,
        yolo_backend=yolo,
        config_loader=loader or (lambda _path: config),
    )


def test_template_ng_globally_skips_patchcore_and_yolo(tmp_path: Path) -> None:
    config = _config(tmp_path)
    template = _Template(config.template_dir, ng_view="front")
    patchcore = _Patchcore()
    yolo = _Yolo()
    runtime = _runtime(config, template=template, patchcore=patchcore, yolo=yolo)

    result = runtime.run(
        _images(tmp_path),
        part_id="part-1",
        capture_session="session-1",
        group_id="group001",
        output_dir=tmp_path / "result",
        config_path=config.path,
    )

    assert result.machine_status == "NG_TEMPLATE"
    assert result.inspection_complete is True
    assert template.calls == list(VIEW_ORDER)
    assert patchcore.calls == []
    assert yolo.calls == []
    assert not (result.output_dir / "crops").exists()
    manifest = json.loads((result.output_dir / "runtime_manifest.json").read_text(encoding="utf-8"))
    assert manifest["evaluated_branch_count"] == 8
    assert manifest["planned_branch_count"] == 24
    for view in VIEW_ORDER:
        branches = manifest["views"][view]["branches"]
        expected_fusion_status = "NG_TEMPLATE" if view == "front" else "PASS"
        assert set(branches) == {"template", "patchcore", "yolo", "fusion"}
        assert branches["template"]["state"] == "available"
        assert branches["patchcore"]["state"] == "skipped"
        assert branches["patchcore"]["status"] == "SKIPPED"
        assert branches["patchcore"]["score"] is None
        assert branches["yolo"]["state"] == "skipped"
        assert branches["yolo"]["status"] == "SKIPPED"
        assert branches["yolo"]["score"] is None
        assert branches["fusion"]["state"] == "available"
        assert branches["fusion"]["status"] == expected_fusion_status
        assert branches["fusion"]["score"] is None
        assert branches["fusion"]["threshold"] is None
        assert (result.output_dir / branches["fusion"]["evidence_path"]).is_file()
    parsed = load_inspection_result(result.output_dir)
    assert parsed.machine_status == "NG_TEMPLATE"
    assert parsed.views[0].branches["patchcore"].state.value == "skipped"
    for view in parsed.views:
        expected_fusion_status = "NG_TEMPLATE" if view.view == "front" else "PASS"
        assert view.branches["fusion"].state.value == "available"
        assert view.branches["fusion"].status == expected_fusion_status
    assert "sha256" not in json.dumps(manifest).lower()


@pytest.mark.parametrize("process_count", [2, 4, 8])
def test_explicit_candidate_uses_multiprocess_backend_with_complete_roi_warmup_crops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    process_count: int,
) -> None:
    config = _config(tmp_path)
    constructed: list[dict[str, object]] = []

    class MultiprocessBackend:
        def __init__(
            self,
            specs: object,
            views: object,
            selected_process_count: int,
            *,
            accelerator: str,
            devices: int,
            warmup_crops: dict[str, Path],
        ) -> None:
            constructed.append(
                {
                    "specs": specs,
                    "views": views,
                    "process_count": selected_process_count,
                    "accelerator": accelerator,
                    "devices": devices,
                    "warmup_shapes": {
                        view: cv2.imread(str(path), cv2.IMREAD_COLOR).shape
                        for view, path in warmup_crops.items()
                    },
                },
            )

    def reject_serial_backend(*_args: object, **_kwargs: object) -> object:
        pytest.fail("candidate process count selected the serial PatchCore backend")

    monkeypatch.setattr(demo_runtime, "ResidentMultiprocessPatchcoreBackend", MultiprocessBackend)
    monkeypatch.setattr(demo_runtime, "AnomalibPatchcoreBackend", reject_serial_backend)

    runtime = ZS32DemoRuntime(
        config,
        template_matcher=_Template(config.template_dir),
        yolo_backend=_Yolo(),
        patchcore_process_count=process_count,
    )

    assert runtime.patchcore_backend.__class__ is MultiprocessBackend
    assert len(constructed) == 1
    record = constructed[0]
    assert tuple(record["specs"]) == VIEW_ORDER
    assert tuple(record["views"]) == VIEW_ORDER
    assert record["process_count"] == process_count
    assert record["accelerator"] == "gpu"
    assert record["devices"] == 1
    assert record["warmup_shapes"] == {view: (5, 6, 3) for view in VIEW_ORDER}


def test_default_process_count_keeps_serial_backend_for_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    serial_calls: list[tuple[object, object, str, int]] = []
    serial_backend = object()

    def serial_factory(
        specs: object,
        *,
        views: object,
        accelerator: str,
        devices: int,
    ) -> object:
        serial_calls.append((specs, views, accelerator, devices))
        return serial_backend

    def reject_multiprocess_backend(*_args: object, **_kwargs: object) -> object:
        pytest.fail("production rollback count 1 selected the multiprocess backend")

    monkeypatch.setattr(demo_runtime, "AnomalibPatchcoreBackend", serial_factory)
    monkeypatch.setattr(
        demo_runtime,
        "ResidentMultiprocessPatchcoreBackend",
        reject_multiprocess_backend,
        raising=False,
    )

    runtime = ZS32DemoRuntime(
        config,
        template_matcher=_Template(config.template_dir),
        yolo_backend=_Yolo(),
    )

    assert runtime.patchcore_backend is serial_backend
    assert len(serial_calls) == 1
    assert tuple(serial_calls[0][0]) == VIEW_ORDER
    assert tuple(serial_calls[0][1]) == VIEW_ORDER
    assert serial_calls[0][2:] == ("gpu", 1)


def test_yolo_initialization_failure_closes_started_multiprocess_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    close_calls: list[str] = []

    class MultiprocessBackend:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def close(self) -> None:
            close_calls.append("closed")

    def fail_yolo_initialization(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("YOLO initialization failed")

    monkeypatch.setattr(demo_runtime, "ResidentMultiprocessPatchcoreBackend", MultiprocessBackend)
    monkeypatch.setattr(demo_runtime, "UltralyticsYoloBackend", fail_yolo_initialization)

    with pytest.raises(RuntimeError, match="YOLO initialization failed"):
        ZS32DemoRuntime(
            config,
            template_matcher=_Template(config.template_dir),
            patchcore_process_count=2,
        )

    assert close_calls == ["closed"]


def test_explicit_process_count_distinguishes_effective_startup_signatures(tmp_path: Path) -> None:
    config = _config(tmp_path)
    runtime_two = ZS32DemoRuntime(
        config,
        template_matcher=_Template(config.template_dir),
        patchcore_backend=_Patchcore(),
        yolo_backend=_Yolo(),
        config_loader=lambda _path: config,
        patchcore_process_count=2,
    )
    runtime_four = ZS32DemoRuntime(
        config,
        template_matcher=_Template(config.template_dir),
        patchcore_backend=_Patchcore(),
        yolo_backend=_Yolo(),
        config_loader=lambda _path: config,
        patchcore_process_count=4,
    )

    assert runtime_two.startup_signature != runtime_four.startup_signature
    assert runtime_two._current_config(config.path) is config
    assert runtime_four._current_config(config.path) is config


def test_configured_process_count_drift_requires_restart_with_explicit_override(
    tmp_path: Path,
) -> None:
    startup = _config(tmp_path)
    changed = SimpleNamespace(**{**startup.__dict__, "patchcore_process_count": 8})
    runtime = ZS32DemoRuntime(
        startup,
        template_matcher=_Template(startup.template_dir),
        patchcore_backend=_Patchcore(),
        yolo_backend=_Yolo(),
        config_loader=lambda _path: changed,
        patchcore_process_count=2,
    )

    with pytest.raises(DemoRuntimeError, match="restart Dashboard"):
        runtime._current_config(startup.path)


def test_runtime_close_delegates_once_to_patchcore_backend(tmp_path: Path) -> None:
    config = _config(tmp_path)
    patchcore = _Patchcore()
    runtime = _runtime(
        config,
        template=_Template(config.template_dir),
        patchcore=patchcore,
        yolo=_Yolo(),
    )

    runtime.close()
    runtime.close()

    assert patchcore.close_calls == 1


def test_worker_uses_authoritative_color_and_grayscale_decodes_and_preserves_source_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    template = _Template(config.template_dir)
    patchcore = _Patchcore()
    yolo = _Yolo()
    runtime = _runtime(config, template=template, patchcore=patchcore, yolo=yolo)
    images = _images(tmp_path)
    images["front"].write_bytes(images["front"].read_bytes() + b"capture-metadata")
    source_paths = {path.resolve() for path in images.values()}
    original_imread = cv2.imread
    original_imdecode = cv2.imdecode
    original_copy2 = shutil.copy2
    decode_flags: list[int] = []

    def reject_source_imread(path: str, flags: int) -> np.ndarray | None:
        if Path(path).resolve() in source_paths:
            pytest.fail("worker used imread instead of authoritative encoded-byte decodes")
        return original_imread(path, flags)

    def counted_imdecode(buffer: np.ndarray, flags: int) -> np.ndarray | None:
        decode_flags.append(flags)
        return original_imdecode(buffer, flags)

    def reject_source_copy2(source: str | Path, destination: str | Path) -> str:
        if Path(source).resolve() in source_paths:
            pytest.fail("worker reread a source through copy2 after its authoritative byte read")
        return str(original_copy2(source, destination))

    monkeypatch.setattr(cv2, "imread", reject_source_imread)
    monkeypatch.setattr(cv2, "imdecode", counted_imdecode)
    monkeypatch.setattr(shutil, "copy2", reject_source_copy2)

    result = runtime.run(
        images,
        part_id="part-1",
        capture_session="session-1",
        group_id="group001",
        output_dir=tmp_path / "result",
        config_path=config.path,
    )

    assert result.machine_status == "OK"
    assert set((result.output_dir / "crops").glob("*.png")) == {
        result.output_dir / "crops" / f"{view}.png" for view in VIEW_ORDER
    }
    assert decode_flags.count(cv2.IMREAD_COLOR) == len(VIEW_ORDER)
    assert decode_flags.count(cv2.IMREAD_GRAYSCALE) == len(VIEW_ORDER)
    assert len(decode_flags) == len(VIEW_ORDER) * 2
    for view in VIEW_ORDER:
        archived = result.output_dir / "sources" / f"{view}.png"
        assert hashlib.sha256(archived.read_bytes()).digest() == hashlib.sha256(
            images[view].read_bytes(),
        ).digest()


def test_template_array_scoring_matches_path_scoring(tmp_path: Path) -> None:
    template_dir = tmp_path / "template-model"
    template_dir.mkdir()
    rng = np.random.default_rng(20260727)
    groups: dict[str, dict[str, list[dict[str, str]]]] = {}
    for view in VIEW_ORDER:
        template = rng.integers(0, 256, size=(19, 17), dtype=np.uint8)
        template_path = template_dir / f"{view}.png"
        assert cv2.imwrite(str(template_path), template)
        groups[f"right/{view}"] = {"templates": [{"path": template_path.name}]}
    (template_dir / "model.json").write_text(
        json.dumps(
            {
                "preprocessing": {"width": 17, "max_shift": 3},
                "groups": groups,
            },
        ),
        encoding="utf-8",
    )
    image = rng.integers(0, 256, size=(37, 29, 3), dtype=np.uint8)
    image_path = tmp_path / "source.png"
    assert cv2.imwrite(str(image_path), image)
    authoritative_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    assert authoritative_gray is not None
    matcher = DemoTemplateMatcher(template_dir)

    path_score = matcher.score(image_path, "front")
    array_score = matcher.score_array(authoritative_gray, "front")

    assert array_score.score == path_score.score
    assert array_score.similarity == path_score.similarity
    assert array_score.offset == path_score.offset
    assert array_score.best_template_path == path_score.best_template_path


def test_template_array_scoring_rejects_color_arrays(tmp_path: Path) -> None:
    template_dir = tmp_path / "template-model"
    template_dir.mkdir()
    groups: dict[str, dict[str, list[dict[str, str]]]] = {}
    for view in VIEW_ORDER:
        template = np.arange(30, dtype=np.uint8).reshape(5, 6)
        template_path = template_dir / f"{view}.png"
        assert cv2.imwrite(str(template_path), template)
        groups[f"right/{view}"] = {"templates": [{"path": template_path.name}]}
    (template_dir / "model.json").write_text(
        json.dumps(
            {
                "preprocessing": {"width": 6, "max_shift": 1},
                "groups": groups,
            },
        ),
        encoding="utf-8",
    )
    matcher = DemoTemplateMatcher(template_dir)

    with pytest.raises(DemoRuntimeError, match="2D grayscale"):
        matcher.score_array(np.zeros((5, 6, 3), dtype=np.uint8), "front")


def test_bad_source_png_fails_before_result_publication(tmp_path: Path) -> None:
    config = _config(tmp_path)
    template = _Template(config.template_dir)
    patchcore = _Patchcore()
    yolo = _Yolo()
    runtime = _runtime(config, template=template, patchcore=patchcore, yolo=yolo)
    images = _images(tmp_path)
    images["front"].write_bytes(b"not a png")
    output_dir = tmp_path / "result"

    with pytest.raises(DemoRuntimeError, match="decode source image for front"):
        runtime.run(
            images,
            part_id="part-1",
            capture_session="session-1",
            group_id="group001",
            output_dir=output_dir,
            config_path=config.path,
        )

    assert not output_dir.exists()
    assert template.calls == []
    assert patchcore.calls == []
    assert yolo.calls == []


def test_wrong_source_dimensions_fail_before_result_publication(tmp_path: Path) -> None:
    config = _config(tmp_path)
    template = _Template(config.template_dir)
    patchcore = _Patchcore()
    yolo = _Yolo()
    runtime = _runtime(config, template=template, patchcore=patchcore, yolo=yolo)
    images = _images(tmp_path)
    assert cv2.imwrite(str(images["front"]), np.zeros((9, 12, 3), dtype=np.uint8))
    output_dir = tmp_path / "result"

    with pytest.raises(DemoRuntimeError, match="must be 12x10, found 12x9"):
        runtime.run(
            images,
            part_id="part-1",
            capture_session="session-1",
            group_id="group001",
            output_dir=output_dir,
            config_path=config.path,
        )

    assert not output_dir.exists()
    assert template.calls == []
    assert patchcore.calls == []
    assert yolo.calls == []


def test_missing_model_evidence_fails_closed_after_other_branches_run(tmp_path: Path) -> None:
    config = _config(tmp_path)
    template = _Template(config.template_dir)
    patchcore = _Patchcore(error_view="back_secondary")
    yolo = _Yolo()
    runtime = _runtime(config, template=template, patchcore=patchcore, yolo=yolo)

    result = runtime.run(
        _images(tmp_path),
        part_id="part-1",
        capture_session="session-1",
        group_id="group001",
        output_dir=tmp_path / "result",
        config_path=config.path,
    )

    assert result.machine_status == "ERROR"
    assert result.inspection_complete is False
    assert any("back_secondary" in error and "patchcore failed" in error for error in result.errors)
    assert yolo.calls == [VIEW_ORDER]
    parsed = load_inspection_result(result.output_dir)
    patchcore = parsed.views[-1].branches["patchcore"]
    assert patchcore.status == "ERROR"
    assert patchcore.threshold == 0.5
    assert "patchcore failed" in patchcore.reason


def test_thresholds_reload_without_reloading_models(tmp_path: Path) -> None:
    startup = _config(tmp_path, template_threshold=0.5)
    updated = _config(tmp_path, template_threshold=0.05)
    template = _Template(startup.template_dir)
    patchcore = _Patchcore()
    yolo = _Yolo()
    loaded = iter((startup, updated))
    runtime = _runtime(
        startup,
        template=template,
        patchcore=patchcore,
        yolo=yolo,
        loader=lambda _path: next(loaded),
    )
    images = _images(tmp_path)

    first = runtime.run(
        images,
        part_id="part-1",
        capture_session="session-1",
        group_id="group001",
        output_dir=tmp_path / "first",
        config_path=startup.path,
    )
    second = runtime.run(
        images,
        part_id="part-2",
        capture_session="session-2",
        group_id="group001",
        output_dir=tmp_path / "second",
        config_path=startup.path,
    )

    assert first.machine_status == "OK"
    assert second.machine_status == "NG_TEMPLATE"
    assert len(patchcore.calls) == 1
    assert len(yolo.calls) == 1
    second_manifest = json.loads((second.output_dir / "runtime_manifest.json").read_text(encoding="utf-8"))
    assert second_manifest["views"]["front"]["branches"]["template"]["threshold"] == 0.05


def test_model_path_change_requires_worker_restart(tmp_path: Path) -> None:
    startup = _config(tmp_path)
    changed = _config(tmp_path)
    changed_weights = tmp_path / "changed.pt"
    changed_weights.write_bytes(b"changed")
    changed.yolo = (
        replace(changed.yolo, weights=changed_weights)
        if hasattr(changed.yolo, "__dataclass_fields__")
        else SimpleNamespace(**{**changed.yolo.__dict__, "weights": changed_weights})
    )
    runtime = _runtime(
        startup,
        template=_Template(startup.template_dir),
        patchcore=_Patchcore(),
        yolo=_Yolo(),
        loader=lambda _path: changed,
    )

    with pytest.raises(RuntimeError, match="restart Dashboard"):
        runtime.run(
            _images(tmp_path),
            part_id="part-1",
            capture_session="session-1",
            group_id="group001",
            output_dir=tmp_path / "result",
            config_path=startup.path,
        )


def test_roi_file_change_requires_worker_restart(tmp_path: Path) -> None:
    startup = _config(tmp_path)
    runtime = _runtime(
        startup,
        template=_Template(startup.template_dir),
        patchcore=_Patchcore(),
        yolo=_Yolo(),
        loader=lambda _path: startup,
    )
    startup.roi_config.write_text('{"changed": true}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="restart Dashboard"):
        runtime.run(
            _images(tmp_path),
            part_id="part-1",
            capture_session="session-1",
            group_id="group001",
            output_dir=tmp_path / "result",
            config_path=startup.path,
        )


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ((("template", "PASS"), ("patchcore", "PASS"), ("yolo", "PASS")), "OK"),
        ((("template", "NG_TEMPLATE"), ("patchcore", "NG_ANOMALY"), ("yolo", "NG_YOLO")), "NG_TEMPLATE"),
        ((("template", "PASS"), ("patchcore", "NG_ANOMALY"), ("yolo", "NG_YOLO")), "NG_ANOMALY"),
        ((("template", "PASS"), ("patchcore", "PASS"), ("yolo", "NG_YOLO")), "NG_YOLO"),
        ((("template", "ERROR"), ("patchcore", "PASS"), ("yolo", "PASS")), "ERROR"),
    ],
)
def test_fusion_precedence(statuses: tuple[tuple[str, str], ...], expected: str) -> None:
    assert fuse_demo_status(dict(statuses)) == expected
