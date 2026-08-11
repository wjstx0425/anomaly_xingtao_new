# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Template training and resident-gate tests for BMW six-view inspection."""

from __future__ import annotations

import json
import runpy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from bmw_inspection.lab.config import TemplateGroupConfig
from bmw_inspection.lab.contracts import BranchName, BranchStatus, ViewId
from bmw_inspection.lab.template import (
    TemplateBackend,
    TemplateSample,
    fit_risk_threshold,
    load_template_manifest,
    train_template_group,
    train_template_groups,
)


REPO_ROOT = Path(__file__).parents[4]
ROI = (0, 0, 80, 64)


def _pattern(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(25, 90, size=(64, 80), dtype=np.uint8)
    cv2.rectangle(image, (8 + seed % 5, 9), (40, 38 + seed % 7), 180 + seed % 50, 2)
    cv2.line(image, (4, 54 - seed % 8), (72, 15 + seed % 9), 220, 2)
    cv2.circle(image, (58, 45), 5 + seed % 4, 130 + seed % 80, -1)
    return image


def _shift(image: np.ndarray, dx: int, dy: int = 0) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REFLECT_101)


def _write_image(path: Path, image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)
    return path


def _sample(
    tmp_path: Path,
    *,
    sample_id: str,
    view_id: ViewId,
    split: str,
    label: str,
    image: np.ndarray,
) -> TemplateSample:
    return TemplateSample(
        sample_id=sample_id,
        part_id=f"part-{sample_id}",
        view_id=view_id,
        image_path=_write_image(tmp_path / "images" / f"{sample_id}.png", image),
        split=split,
        label=label,
    )


def _rows(tmp_path: Path, view_id: ViewId = ViewId.FRONT, *, final_seed: int = 91) -> list[TemplateSample]:
    reference = _pattern(1)
    return [
        _sample(tmp_path, sample_id="train-a", view_id=view_id, split="train", label="normal", image=reference),
        _sample(tmp_path, sample_id="train-b", view_id=view_id, split="train", label="normal", image=_pattern(2)),
        _sample(tmp_path, sample_id="train-c", view_id=view_id, split="train", label="normal", image=_pattern(3)),
        _sample(
            tmp_path,
            sample_id="cal-normal",
            view_id=view_id,
            split="calibration",
            label="normal",
            image=_shift(reference, 8),
        ),
        _sample(
            tmp_path,
            sample_id="cal-defect",
            view_id=view_id,
            split="calibration",
            label="defect",
            image=_pattern(71),
        ),
        _sample(
            tmp_path,
            sample_id="final-normal",
            view_id=view_id,
            split="final_test",
            label="normal",
            image=_shift(reference, 2, 1),
        ),
        _sample(
            tmp_path,
            sample_id="final-defect",
            view_id=view_id,
            split="final_test",
            label="defect",
            image=_pattern(final_seed),
        ),
    ]


def _train(tmp_path: Path, *, max_shift: int = 12) -> tuple[Path, list[TemplateSample]]:
    rows = _rows(tmp_path)
    model_path = train_template_group(
        rows,
        ROI,
        tmp_path / "model-front",
        max_shift=max_shift,
        target_size=(80, 64),
        template_count=3,
    )
    return model_path, rows


def test_small_translation_matches_but_large_translation_fails(tmp_path: Path) -> None:
    model_path, rows = _train(tmp_path, max_shift=12)
    backend = TemplateBackend({ViewId.FRONT: model_path}, required_for_ok=True)
    reference = cv2.imread(str(rows[0].image_path), cv2.IMREAD_GRAYSCALE)

    small = backend.predict("front", _shift(reference, 8))
    small_decision = backend.decision_for(ViewId.FRONT)
    large = backend.predict(ViewId.FRONT, _shift(reference, 20))
    large_decision = backend.decision_for(ViewId.FRONT)

    assert small.status is BranchStatus.PASS
    assert small.required_for_ok is True
    assert small_decision is not None
    assert abs(small_decision.offset_x) <= 12
    assert small_decision.risk == pytest.approx(small.score)
    assert large.status is BranchStatus.NG
    assert large_decision is not None
    assert large_decision.risk > large.threshold


def test_best_of_multiple_templates_is_reported_as_structured_evidence(tmp_path: Path) -> None:
    model_path, rows_for_view = _train(tmp_path)
    model = json.loads(model_path.read_text(encoding="utf-8"))
    selected = model["templates"][1]
    selected_row = next(row for row in rows_for_view if row.sample_id == selected["sample_id"])
    query = cv2.imread(str(selected_row.image_path), cv2.IMREAD_GRAYSCALE)
    backend = TemplateBackend({ViewId.FRONT: model_path}, required_for_ok=False, thresholds={ViewId.FRONT: 1.0})

    evidence = backend.predict(ViewId.FRONT, query)
    decision = backend.decision_for(ViewId.FRONT)

    assert evidence.status is BranchStatus.PASS
    assert evidence.required_for_ok is False
    assert decision is not None
    assert decision.best_template_sample_id == selected["sample_id"]
    assert decision.similarity == pytest.approx(1.0)
    assert evidence.artifact_paths["best_template"] == str((model_path.parent / selected["path"]).resolve())


def test_flat_and_wrong_dimension_inputs_fail_closed(tmp_path: Path) -> None:
    model_path, _rows_for_view = _train(tmp_path)
    backend = TemplateBackend({ViewId.FRONT: model_path}, required_for_ok=True)

    flat = backend.predict(ViewId.FRONT, np.full((64, 80), 127, dtype=np.uint8))
    wrong_size = backend.predict(ViewId.FRONT, _pattern(1)[:60, :])

    assert flat.status is BranchStatus.ERROR
    assert flat.score is None
    assert "flat" in flat.reason
    assert wrong_size.status is BranchStatus.ERROR
    assert "dimensions" in wrong_size.reason


def test_threshold_equality_passes(tmp_path: Path) -> None:
    model_path, rows = _train(tmp_path)
    query = cv2.imread(str(rows[0].image_path), cv2.IMREAD_GRAYSCALE)
    probe = TemplateBackend({ViewId.FRONT: model_path}, required_for_ok=True, thresholds={ViewId.FRONT: 1.0})
    risk = probe.predict(ViewId.FRONT, query).score
    assert risk is not None
    backend = TemplateBackend({ViewId.FRONT: model_path}, required_for_ok=True, thresholds={ViewId.FRONT: risk})

    evidence = backend.predict(ViewId.FRONT, query)

    assert evidence.score == pytest.approx(evidence.threshold)
    assert evidence.status is BranchStatus.PASS


def test_missing_or_tampered_template_is_rejected_at_load(tmp_path: Path) -> None:
    missing_model, _rows_for_view = _train(tmp_path / "missing")
    missing_payload = json.loads(missing_model.read_text(encoding="utf-8"))
    (missing_model.parent / missing_payload["templates"][0]["path"]).unlink()
    with pytest.raises(ValueError, match="missing template"):
        TemplateBackend({ViewId.FRONT: missing_model}, required_for_ok=True)

    tampered_model, _rows_for_view = _train(tmp_path / "tampered")
    tampered_payload = json.loads(tampered_model.read_text(encoding="utf-8"))
    template_path = tampered_model.parent / tampered_payload["templates"][0]["path"]
    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    template[0, 0] ^= 0xFF
    assert cv2.imwrite(str(template_path), template)
    with pytest.raises(ValueError, match="template sha256"):
        TemplateBackend({ViewId.FRONT: tampered_model}, required_for_ok=True)


def test_calibration_threshold_fit_is_deterministic_and_prefers_fewer_false_rejects() -> None:
    fit = fit_risk_threshold(
        [
            ("normal", 0.10),
            ("normal", 0.20),
            ("defect", 0.70),
            ("defect", 0.80),
        ]
    )

    assert fit.threshold == pytest.approx(0.20)
    assert fit.balanced_accuracy == pytest.approx(1.0)
    assert fit.normal_false_rejects == 0


def test_final_test_never_changes_templates_or_fitted_threshold(tmp_path: Path) -> None:
    rows_a = _rows(tmp_path / "data-a", final_seed=91)
    rows_b = [
        row
        for row in rows_a
        if row.split != "final_test"
    ] + [
        _sample(
            tmp_path / "data-b",
            sample_id="final-normal-b",
            view_id=ViewId.FRONT,
            split="final_test",
            label="normal",
            image=_pattern(110),
        ),
        _sample(
            tmp_path / "data-b",
            sample_id="final-defect-b",
            view_id=ViewId.FRONT,
            split="final_test",
            label="defect",
            image=_pattern(111),
        ),
    ]
    model_a = train_template_group(rows_a, ROI, tmp_path / "model-a", target_size=(80, 64), template_count=3)
    model_b = train_template_group(rows_b, ROI, tmp_path / "model-b", target_size=(80, 64), template_count=3)

    assert model_a.read_bytes() == model_b.read_bytes()
    assert (model_a.parent / "model.sha256").read_bytes() == (model_b.parent / "model.sha256").read_bytes()
    assert (model_a.parent / "calibration_rows.csv").read_bytes() == (
        model_b.parent / "calibration_rows.csv"
    ).read_bytes()
    assert (model_a.parent / "metrics.json").read_bytes() != (model_b.parent / "metrics.json").read_bytes()


def test_training_is_deterministic_and_refuses_to_overwrite(tmp_path: Path) -> None:
    rows = _rows(tmp_path / "data")
    model_a = train_template_group(rows, ROI, tmp_path / "model-a", target_size=(80, 64), template_count=3)
    model_b = train_template_group(rows, ROI, tmp_path / "model-b", target_size=(80, 64), template_count=3)

    assert model_a.read_bytes() == model_b.read_bytes()
    assert sorted(path.read_bytes() for path in (model_a.parent / "templates").glob("*.png")) == sorted(
        path.read_bytes() for path in (model_b.parent / "templates").glob("*.png")
    )
    before = {
        path.relative_to(model_a.parent): path.read_bytes()
        for path in model_a.parent.rglob("*")
        if path.is_file()
    }
    with pytest.raises(FileExistsError, match="refuse to overwrite"):
        train_template_group(rows, ROI, model_a.parent, target_size=(80, 64), template_count=3)
    after = {
        path.relative_to(model_a.parent): path.read_bytes()
        for path in model_a.parent.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_manifest_rejects_one_part_id_crossing_splits(tmp_path: Path) -> None:
    image_path = _write_image(tmp_path / "shared.png", _pattern(1))
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sample_id,part_id,view_id,image_path,split,label\n"
        f"train,shared-part,front,{image_path},train,normal\n"
        f"calibration,shared-part,front,{image_path},calibration,normal\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="part_id shared-part crosses splits"):
        load_template_manifest(manifest)


def test_direct_group_training_rejects_one_part_id_crossing_splits(tmp_path: Path) -> None:
    rows = _rows(tmp_path / "data")
    rows[3] = replace(rows[3], part_id=rows[0].part_id)

    with pytest.raises(ValueError, match="part_id part-train-a crosses splits"):
        train_template_group(rows, ROI, tmp_path / "model", target_size=(80, 64), template_count=3)


def test_direct_six_view_training_rejects_cross_view_part_split_leakage(tmp_path: Path) -> None:
    rows: list[TemplateSample] = []
    for view_id in ViewId:
        rows.extend(_rows(tmp_path / view_id.value, view_id))
    leaked_index = next(
        index
        for index, row in enumerate(rows)
        if row.view_id is ViewId.BACK and row.split == "calibration"
    )
    rows[leaked_index] = replace(rows[leaked_index], part_id="part-train-a")

    with pytest.raises(ValueError, match="part_id part-train-a crosses splits"):
        train_template_groups(
            rows,
            {view_id: ROI for view_id in ViewId},
            tmp_path / "models",
            target_size=(80, 64),
            template_count=3,
        )


@pytest.mark.parametrize("invalid", [True, 3.0])
@pytest.mark.parametrize("trainer", ["group", "groups"])
def test_public_training_apis_reject_noninteger_template_count(
    tmp_path: Path,
    trainer: str,
    invalid: object,
) -> None:
    with pytest.raises(TypeError, match="template_count must be an integer"):
        if trainer == "group":
            train_template_group([], ROI, tmp_path / "model", template_count=invalid)  # type: ignore[arg-type]
        else:
            train_template_groups([], {}, tmp_path / "models", template_count=invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid", [2, 6])
@pytest.mark.parametrize("trainer", ["group", "groups"])
def test_public_training_apis_reject_template_count_outside_three_to_five(
    tmp_path: Path,
    trainer: str,
    invalid: int,
) -> None:
    with pytest.raises(ValueError, match="template_count must be between 3 and 5"):
        if trainer == "group":
            train_template_group([], ROI, tmp_path / "model", template_count=invalid)
        else:
            train_template_groups([], {}, tmp_path / "models", template_count=invalid)


def test_synthetic_six_view_training_produces_independent_resident_groups(tmp_path: Path) -> None:
    rows: list[TemplateSample] = []
    for index, view_id in enumerate(ViewId):
        view_rows = _rows(tmp_path / view_id.value, view_id, final_seed=90 + index)
        rows.extend(view_rows)
    model_paths = train_template_groups(
        rows,
        {view_id: ROI for view_id in ViewId},
        tmp_path / "six-view-models",
        target_size=(80, 64),
        max_shift=12,
        template_count=3,
    )
    backend = TemplateBackend.from_model_root(tmp_path / "six-view-models", required_for_ok=True)

    assert tuple(model_paths) == tuple(ViewId)
    for view_id in ViewId:
        group_dir = tmp_path / "six-view-models" / view_id.value
        assert model_paths[view_id] == group_dir / "model.json"
        assert {path.name for path in group_dir.iterdir()} == {
            "calibration_rows.csv",
            "metrics.json",
            "model.json",
            "model.sha256",
            "templates",
        }
        query = cv2.imread(str(next((group_dir / "templates").glob("*.png"))), cv2.IMREAD_GRAYSCALE)
        assert backend.predict(view_id, query).status is BranchStatus.PASS


def test_backend_reads_threshold_and_required_flag_from_experiment_profile(tmp_path: Path) -> None:
    model_path, rows = _train(tmp_path)
    profile = SimpleNamespace(
        template=SimpleNamespace(
            enabled=True,
            groups={
                ViewId.FRONT: TemplateGroupConfig(
                    view_id=ViewId.FRONT,
                    model_path=model_path,
                    threshold=0.0,
                )
            },
        ),
        required_for_ok=frozenset({BranchName.TEMPLATE}),
    )
    backend = TemplateBackend.from_experiment_config(profile)
    query = cv2.imread(str(rows[0].image_path), cv2.IMREAD_GRAYSCALE)

    evidence = backend.predict(ViewId.FRONT, query)

    assert evidence.required_for_ok is True
    assert evidence.threshold == 0.0


def test_six_view_roi_selector_maps_scaled_display_coordinates_to_half_open_source() -> None:
    namespace = runpy.run_path(str(REPO_ROOT / "pipeline/bmw_lab_select_rois.py"))
    fit_image_for_display = namespace["fit_image_for_display"]
    source_roi = namespace["source_roi"]
    source = np.zeros((3036, 4024), dtype=np.uint8)

    displayed = fit_image_for_display(source, max_display_width=1280, max_display_height=720)
    roi = source_roi((100, 50, 200, 100), displayed.shape[:2], source.shape[:2])

    assert displayed.shape == (720, 954)
    assert roi == (421, 210, 1266, 633)


def test_template_training_cli_exposes_manifest_config_and_output_contract() -> None:
    namespace = runpy.run_path(str(REPO_ROOT / "pipeline/bmw_lab_train_template.py"))
    parser = namespace["build_parser"]()

    args = parser.parse_args(["--manifest", "rows.csv", "--config", "profile.json", "--output-root", "models"])

    assert args.manifest == Path("rows.csv")
    assert args.config == Path("profile.json")
    assert args.output_root == Path("models")
