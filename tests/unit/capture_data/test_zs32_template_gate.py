# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the fail-closed ZS32 whole-view template gate."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest
from capture_data.fusion_calibration import fit_dual_thresholds
from capture_data.zs32_template_gate import (
    CALIBRATION_FILENAME,
    MODEL_SHA256_FILENAME,
    PreparedTemplateGate,
    ZS32_VIEWS,
    TemplateGateError,
    load_model,
    match_template,
    predict_template_gate,
    train_template_gate,
)

from zs32_inspection.domain.views import VIEW_ORDER

if TYPE_CHECKING:
    from pathlib import Path


EXPECTED_ZS32_VIEWS = (
    "front",
    "front_left",
    "front_right",
    "front_secondary",
    "back",
    "back_left",
    "back_right",
    "back_secondary",
)


def _pattern(offset: int = 0, *, defect: bool = False) -> np.ndarray:
    """Build a textured deterministic test image."""
    image = np.zeros((60, 90), dtype=np.uint8)
    cv2.rectangle(image, (18 + offset, 12), (70 + offset, 48), 120, 2)
    cv2.circle(image, (42 + offset, 30), 9, 220, -1)
    cv2.line(image, (5, 54), (84, 6), 70, 2)
    if defect:
        cv2.rectangle(image, (18, 12), (70, 48), 255, -1)
        cv2.circle(image, (42, 30), 9, 10, -1)
    return image


def _write(path: Path, image: np.ndarray) -> None:
    """Write an image and assert OpenCV accepted it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def _manifest(
    tmp_path: Path,
    *,
    overlap: bool = False,
    omit_defect: bool = False,
    omit_calibration: tuple[str, str] | None = None,
    hand: str = "left",
) -> Path:
    """Create a one-hand, eight-view calibration/test manifest."""
    rows: list[dict[str, str]] = []
    for view in EXPECTED_ZS32_VIEWS:
        for sample, (label, split, image) in enumerate(
            (
                ("normal", "calibration", _pattern()),
                ("normal", "calibration", _pattern(1)),
                ("defect", "calibration", _pattern(defect=True)),
                ("normal", "test", _pattern()),
                ("defect", "test", _pattern(defect=True)),
            ),
        ):
            if (
                split == "calibration"
                and ((omit_defect and view == "back" and label == "defect") or omit_calibration == (view, label))
            ):
                continue
            path = tmp_path / "images" / f"{view}-{sample}.png"
            _write(path, image)
            part_id = f"{view}-{label}-{split}-{sample}"
            if overlap and view == "front" and sample in {0, 3}:
                part_id = "leaked-part"
            rows.append(
                {
                    "part_id": part_id,
                    "hand": hand,
                    "view": view,
                    "label": label,
                    "split": split,
                    "image_path": str(path),
                },
            )
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def _train(tmp_path: Path) -> Path:
    """Train a complete eight-view test model."""
    output = tmp_path / "model"
    train_template_gate(
        _manifest(tmp_path),
        output,
        required_hands=("left",),
        width=90,
        max_shift=3,
        templates_per_group=2,
        normal_quantile=1.0,
        model_version="model-v1",
        threshold_version="threshold-v1",
        roi_version="roi-v1",
        template_version="template-v1",
    )
    return output


def test_zs32_views_use_exact_eight_view_runtime_order() -> None:
    """Template training and inference must share the canonical eight-view order."""
    assert ZS32_VIEWS == EXPECTED_ZS32_VIEWS


def _rewrite_model(model_dir: Path, model: dict[str, object]) -> None:
    """Rewrite a test model and its deployment digest."""
    model_path = model_dir / "model.json"
    model_path.write_text(json.dumps(model, allow_nan=True), encoding="utf-8")
    (model_dir / MODEL_SHA256_FILENAME).write_text(
        hashlib.sha256(model_path.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )


def test_match_template_records_best_offset() -> None:
    """The shift search should report the displacement of the best match."""
    template = _pattern()
    shifted = np.roll(template, 2, axis=1)

    score, offset = match_template(shifted, template, max_shift=4)

    assert score > 0.99
    assert offset == (2, 0)


def test_training_is_deterministic_and_publishes_versioned_schema(tmp_path: Path) -> None:
    """Identical inputs must produce identical model contracts and template bytes."""
    manifest = _manifest(tmp_path)
    outputs = [tmp_path / "first", tmp_path / "second"]
    for output in outputs:
        train_template_gate(
            manifest,
            output,
            required_hands=("left",),
            width=90,
            templates_per_group=2,
            model_version="model-v1",
            threshold_version="threshold-v1",
            roi_version="roi-v1",
            template_version="template-v1",
        )

    first = json.loads((outputs[0] / "model.json").read_text(encoding="utf-8"))
    second = json.loads((outputs[1] / "model.json").read_text(encoding="utf-8"))
    assert first == second
    assert first["schema"] == "anomalib.zs32_template_gate"
    assert first["versions"] == {
        "model": "model-v1",
        "threshold": "threshold-v1",
        "roi": "roi-v1",
        "template": "template-v1",
    }
    assert tuple(first["groups"]) == tuple(f"left/{view}" for view in VIEW_ORDER)
    assert sorted(first["groups"]) == [
        "left/back",
        "left/back_left",
        "left/back_right",
        "left/back_secondary",
        "left/front",
        "left/front_left",
        "left/front_right",
        "left/front_secondary",
    ]
    front_group = first["groups"]["left/front"]
    assert front_group["template_normal_count"] == 1
    assert front_group["threshold_normal_count"] == 1
    assert front_group["templates"][0]["sha256"]
    assert len((outputs[0] / MODEL_SHA256_FILENAME).read_text(encoding="utf-8").strip()) == 64
    assert (outputs[0] / CALIBRATION_FILENAME).is_file()
    first_templates = sorted((outputs[0] / "templates").rglob("*.png"))
    second_templates = sorted((outputs[1] / "templates").rglob("*.png"))
    assert [path.read_bytes() for path in first_templates] == [path.read_bytes() for path in second_templates]


def test_training_publishes_all_right_hand_eight_view_groups(tmp_path: Path) -> None:
    """A right-hand manifest must publish one template group for every view."""
    output = tmp_path / "right-model"

    model = train_template_gate(
        _manifest(tmp_path, hand="right"),
        output,
        required_hands=("right",),
        width=90,
        templates_per_group=2,
        model_version="model-v1",
        threshold_version="threshold-v1",
        roi_version="roi-v1",
        template_version="template-v1",
    )

    assert model["required_hands"] == ["right"]
    assert tuple(model["required_views"]) == VIEW_ORDER
    assert tuple(model["groups"]) == tuple(f"right/{view}" for view in VIEW_ORDER)


def test_normal_only_training_preserves_explicit_four_role_splits(tmp_path: Path) -> None:
    """Train, validation, calibration, and final-test parts must keep independent roles."""
    rows: list[dict[str, str]] = []
    for view in EXPECTED_ZS32_VIEWS:
        for split, count in (("train", 4), ("model_val", 2), ("calibration", 2), ("final_test", 2)):
            for sample in range(count):
                image = tmp_path / "four-role-images" / f"{view}-{split}-{sample}.png"
                _write(image, _pattern(sample))
                rows.append(
                    {
                        "part_id": f"{view}-{split}-{sample}",
                        "hand": "right",
                        "view": view,
                        "label": "normal",
                        "split": split,
                        "image_path": str(image),
                    },
                )
    manifest = tmp_path / "four-role-manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    output = tmp_path / "four-role-model"
    model = train_template_gate(
        manifest,
        output,
        required_hands=("right",),
        width=90,
        max_shift=3,
        templates_per_group=2,
        normal_quantile=1.0,
        normal_only=True,
        model_version="four-role-model-v1",
        threshold_version="four-role-threshold-v1",
        roi_version="roi-v1",
        template_version="four-role-template-v1",
    )

    for group in model["groups"].values():
        assert group["template_normal_count"] == 4
        assert group["threshold_normal_count"] == 2
        assert group["model_val_count"] == 2
        assert group["final_test_count"] == 2
        assert all("-train-" in template["source_part_id"] for template in group["templates"])
    with (output / CALIBRATION_FILENAME).open(encoding="utf-8", newline="") as file:
        calibration_rows = list(csv.DictReader(file))
    assert {row["split"] for row in calibration_rows} == {"calibration", "model_val", "final_test"}


def test_normal_only_training_ignores_all_defect_rows_and_removes_review_band(tmp_path: Path) -> None:
    """Normal-only commissioning must derive templates and thresholds exclusively from normals."""
    output = tmp_path / "normal-only-model"

    model = train_template_gate(
        _manifest(tmp_path, hand="right"),
        output,
        required_hands=("right",),
        width=90,
        templates_per_group=2,
        normal_quantile=1.0,
        normal_only=True,
        model_version="normal-only-model-v1",
        threshold_version="normal-only-threshold-v1",
        roi_version="roi-v1",
        template_version="normal-only-template-v1",
    )

    assert model["threshold_policy"]["calibration_mode"] == "normal_only"
    for group in model["groups"].values():
        assert group["low_threshold"] == group["high_threshold"]
        assert group["calibration_defect_count"] == 0
    with (output / CALIBRATION_FILENAME).open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows
    assert {row["gt_label"] for row in rows} == {"0"}
    by_view = {view: [] for view in EXPECTED_ZS32_VIEWS}
    for row in rows:
        if row["split"] == "calibration":
            by_view[row["view"]].append(float(row["raw_score"]))
    for view, risks in by_view.items():
        threshold = model["groups"][f"right/{view}"]["high_threshold"]
        assert threshold == math.nextafter(max(risks), 2.0)
        assert all(risk < threshold for risk in risks)


@pytest.mark.parametrize(
    ("view", "label"),
    [("front_secondary", "normal"), ("back_secondary", "defect")],
)
def test_training_rejects_missing_secondary_calibration_class(tmp_path: Path, view: str, label: str) -> None:
    """Every secondary view needs both normal and defect calibration rows."""
    with pytest.raises(TemplateGateError, match=rf"{view}.*{label}"):
        train_template_gate(
            _manifest(tmp_path, hand="right", omit_calibration=(view, label)),
            tmp_path / "model",
            required_hands=("right",),
            model_version="model-v1",
            threshold_version="threshold-v1",
            roi_version="roi-v1",
            template_version="template-v1",
        )


def test_load_model_preserves_generic_legacy_contract_for_strict_caller_validation(tmp_path: Path) -> None:
    """Generic loading stays schema-compatible; Stage33 owns exact-right validation."""
    model_dir = _train(tmp_path)
    model = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    model["required_views"].remove("front_secondary")
    del model["groups"]["left/front_secondary"]
    _rewrite_model(model_dir, model)

    assert load_model(model_dir)["required_views"] != list(VIEW_ORDER)


def test_prediction_preserves_continuous_evidence_and_best_template(tmp_path: Path) -> None:
    """Inference should expose the risk, thresholds, template, and shift evidence."""
    model_dir = _train(tmp_path)
    image = tmp_path / "query.png"
    _write(image, _pattern())

    result = predict_template_gate(model_dir, image, hand="left", view="front")

    assert result.status == "PASS"
    assert result.score == result.risk_score
    assert result.threshold == result.high_threshold
    assert result.similarity > 0.99
    assert result.best_template.startswith("templates/left/front/")
    assert len(result.best_template_sha256) == 64
    assert len(result.offset) == 2


def test_prepared_template_gate_is_numerically_identical(tmp_path: Path) -> None:
    """Caching model and template bytes must not alter evidence or decisions."""
    model_dir = _train(tmp_path)
    image = tmp_path / "query.png"
    _write(image, _pattern())

    expected = predict_template_gate(model_dir, image, hand="left", view="front")
    prepared = PreparedTemplateGate(model_dir, hand="left", views=("front",))
    actual = prepared.evaluate(image, "left", "front")

    assert actual == expected


def test_dual_threshold_boundary_statuses(tmp_path: Path) -> None:
    """Risk classification must use inclusive REVIEW and NG lower bounds."""
    model_dir = _train(tmp_path)
    model_path = model_dir / "model.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    group = model["groups"]["left/front"]
    query = tmp_path / "defect.png"
    _write(query, _pattern(defect=True))
    raw = predict_template_gate(model_dir, query, hand="left", view="front")
    group["low_threshold"] = raw.risk_score
    group["high_threshold"] = min(1.0, raw.risk_score + 0.01)
    _rewrite_model(model_dir, model)

    review = predict_template_gate(model_dir, query, hand="left", view="front")
    assert review.status == "REVIEW"

    group["low_threshold"] = max(0.0, raw.risk_score - 0.01)
    group["high_threshold"] = raw.risk_score
    _rewrite_model(model_dir, model)
    rejected = predict_template_gate(model_dir, query, hand="left", view="front")
    assert rejected.status == "NG_TEMPLATE"


@pytest.mark.parametrize(
    ("overlap", "omit_defect", "match"),
    [(True, False, "multiple splits"), (False, True, "defect")],
)
def test_training_rejects_unsafe_calibration_data(
    tmp_path: Path,
    overlap: bool,
    omit_defect: bool,
    match: str,
) -> None:
    """Part leakage and a calibration class gap must reject deployment."""
    with pytest.raises(TemplateGateError, match=match):
        train_template_gate(
            _manifest(tmp_path, overlap=overlap, omit_defect=omit_defect),
            tmp_path / "model",
            required_hands=("left",),
            model_version="model-v1",
            threshold_version="threshold-v1",
            roi_version="roi-v1",
            template_version="template-v1",
        )


def test_training_refuses_to_replace_an_existing_model_generation(tmp_path: Path) -> None:
    """A new training run must not destructively overwrite an existing model directory."""
    output = tmp_path / "model"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("existing", encoding="utf-8")

    with pytest.raises(TemplateGateError) as error:
        train_template_gate(
            _manifest(tmp_path),
            output,
            required_hands=("left",),
            model_version="model-v1",
            threshold_version="threshold-v1",
            roi_version="roi-v1",
            template_version="template-v1",
        )

    assert error.value.code == "OUTPUT_EXISTS"
    assert sentinel.read_text(encoding="utf-8") == "existing"


def test_exported_calibration_rows_reproduce_model_thresholds(tmp_path: Path) -> None:
    """Stage 31 must fit exactly the thresholds embedded in the template model."""
    model_dir = _train(tmp_path)
    model = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    with (model_dir / CALIBRATION_FILENAME).open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert {"model_version", "threshold_version", "roi_version", "template_version"} <= set(rows[0])
    assert all(row["threshold_version"] == "threshold-v1" for row in rows)
    assert all(row["template_version"] == "template-v1" for row in rows)
    thresholds = fit_dual_thresholds(rows, target_recall=1.0, normal_quantile=1.0, required_views=())
    by_view = {record.view: record for record in thresholds}
    for view in EXPECTED_ZS32_VIEWS:
        group = model["groups"][f"left/{view}"]
        assert by_view[view].low_threshold == group["low_threshold"]
        assert by_view[view].high_threshold == group["high_threshold"]


def test_prediction_rejects_tampered_model_contract(tmp_path: Path) -> None:
    """Changing deployed JSON thresholds without the sidecar digest must fail closed."""
    model_dir = _train(tmp_path)
    model_path = model_dir / "model.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    model["groups"]["left/front"]["low_threshold"] = 0.0
    model_path.write_text(json.dumps(model), encoding="utf-8")
    image = tmp_path / "query.png"
    _write(image, _pattern())

    with pytest.raises(TemplateGateError) as error:
        predict_template_gate(model_dir, image, hand="left", view="front")

    assert error.value.code == "MODEL_HASH_MISMATCH"


@pytest.mark.parametrize("path_kind", ["parent", "absolute"])
def test_prediction_rejects_template_paths_outside_model(tmp_path: Path, path_kind: str) -> None:
    """A model record cannot escape its immutable model directory."""
    model_dir = _train(tmp_path)
    model = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    unsafe_path = "../outside.png" if path_kind == "parent" else str(tmp_path.parent / "outside.png")
    model["groups"]["left/front"]["templates"][0]["path"] = unsafe_path
    _rewrite_model(model_dir, model)
    image = tmp_path / "query.png"
    _write(image, _pattern())

    with pytest.raises(TemplateGateError) as error:
        predict_template_gate(model_dir, image, hand="left", view="front")

    assert error.value.code == "MODEL_INVALID"


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("missing_group", "GROUP_NOT_FOUND"),
        ("missing_template", "TEMPLATE_MISSING"),
        ("bad_template", "TEMPLATE_INVALID"),
        ("tampered_template", "TEMPLATE_HASH_MISMATCH"),
        ("nan_threshold", "MODEL_INVALID"),
    ],
)
def test_prediction_fails_closed_with_structured_error(tmp_path: Path, mutation: str, code: str) -> None:
    """Missing or corrupt evidence must never receive a fallback threshold."""
    model_dir = _train(tmp_path)
    model_path = model_dir / "model.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    group = model["groups"]["left/front"]
    if mutation == "missing_group":
        del model["groups"]["left/front"]
    elif mutation == "missing_template":
        group["templates"] = [{"path": "templates/not-there.png", "source_part_id": "missing"}]
    elif mutation == "bad_template":
        template = model_dir / group["templates"][0]["path"]
        template.write_bytes(b"not an image")
    elif mutation == "tampered_template":
        template = model_dir / group["templates"][0]["path"]
        _write(template, _pattern(defect=True))
    else:
        group["low_threshold"] = float("nan")
    _rewrite_model(model_dir, model)
    image = tmp_path / "query.png"
    _write(image, _pattern())

    with pytest.raises(TemplateGateError) as error:
        predict_template_gate(model_dir, image, hand="left", view="front")

    assert error.value.code == code
    payload = error.value.to_result()
    assert payload["status"] == "INVALID_TEMPLATE_GATE"
    assert payload["reason"]
    assert payload["error"]["code"] == code
