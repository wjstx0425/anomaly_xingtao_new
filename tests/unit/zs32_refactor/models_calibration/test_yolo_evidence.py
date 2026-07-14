"""Linux-authoritative tests for strict normalized YOLO evidence."""

from __future__ import annotations

import pytest

from zs32_inspection.models import ModelContractError, RawModelScore

HASH = "a" * 64


def _detection(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "class_id": 0,
        "class_name": "defect",
        "confidence": 0.7,
        "xyxy_norm": [0.1, 0.2, 0.6, 0.8],
        "area_ratio": 0.3,
        "clipped": False,
    }
    payload.update(overrides)
    return payload


def _score(*, score: float, detections: tuple[dict[str, object], ...]) -> RawModelScore:
    return RawModelScore(
        inspection_id="inspection-1",
        part_instance_id="part-1",
        capture_set_id="capture-1",
        hand="right",
        view="front",
        branch="yolo",
        score=score,
        model_family="yolo",
        model_digest=HASH,
        roi_config_id="roi-v1",
        roi_digest=HASH,
        source_sha256=HASH,
        crop_sha256=HASH,
        detections=detections,
    )


def test_yolo_score_must_equal_maximum_detection_confidence() -> None:
    with pytest.raises(ModelContractError, match="maximum detection confidence"):
        _score(score=0.6, detections=(_detection(),))


def test_empty_yolo_detections_require_zero_score() -> None:
    with pytest.raises(ModelContractError, match="maximum detection confidence"):
        _score(score=0.1, detections=())


@pytest.mark.parametrize(
    "detection",
    [
        _detection(class_id=1),
        _detection(class_name="scratch"),
        _detection(confidence=float("nan")),
        _detection(xyxy_norm=[0.6, 0.2, 0.1, 0.8]),
        _detection(xyxy_norm=[-0.1, 0.2, 0.6, 0.8]),
        _detection(area_ratio=0.2),
        _detection(clipped="false"),
    ],
)
def test_malformed_yolo_detection_fails_closed(detection: dict[str, object]) -> None:
    with pytest.raises(ModelContractError):
        _score(score=0.7, detections=(detection,))


def test_valid_yolo_detection_is_copied_into_frozen_normalized_form() -> None:
    row = _score(score=0.7, detections=(_detection(),))

    assert row.detections[0]["class_name"] == "defect"
    assert row.detections[0]["xyxy_norm"] == [0.1, 0.2, 0.6, 0.8]
