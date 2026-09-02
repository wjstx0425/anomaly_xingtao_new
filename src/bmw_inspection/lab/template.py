# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Per-view multi-template training and resident matching for BMW inspection."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Any

import cv2
import numpy as np

from bmw_inspection.lab.config import LabExperimentConfig
from bmw_inspection.lab.contracts import BranchEvidence, BranchName, BranchStatus, ViewId


_SPLITS = frozenset({"train", "calibration", "final_test"})
_LABELS = frozenset({"normal", "defect"})
_SOURCE_SESSION = re.compile(r"(?P<session_id>[0-9]{8}_[0-9]{6}_[0-9]{6})__")
_CALIBRATION_FIELDS = (
    "sample_id",
    "part_id",
    "view_id",
    "label",
    "similarity",
    "risk",
    "threshold",
    "prediction",
    "best_template_sample_id",
    "offset_x",
    "offset_y",
)


def _non_empty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _view_id(value: object) -> ViewId:
    try:
        return ViewId(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unknown BMW view_id: {value!r}") from error


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _validate_template_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("template_count must be an integer")
    if not 3 <= value <= 5:
        raise ValueError("template_count must be between 3 and 5")
    return value


def _validate_part_split_isolation(rows: Sequence[TemplateSample]) -> None:
    part_splits: dict[str, str] = {}
    for row in rows:
        previous = part_splits.setdefault(row.part_id, row.split)
        if previous != row.split:
            raise ValueError(f"part_id {row.part_id} crosses splits: {previous}, {row.split}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True, slots=True)
class TemplateSample:
    """One explicitly split manifest row consumed by Template training."""

    sample_id: str
    part_id: str
    view_id: ViewId
    image_path: Path
    split: str
    label: str

    def __post_init__(self) -> None:
        _non_empty(self.sample_id, "sample_id")
        _non_empty(self.part_id, "part_id")
        object.__setattr__(self, "view_id", _view_id(self.view_id))
        image_path = Path(self.image_path).expanduser().resolve()
        if not image_path.is_file():
            raise ValueError(f"template sample image does not exist: {image_path}")
        object.__setattr__(self, "image_path", image_path)
        if self.split not in _SPLITS:
            raise ValueError(f"split must be one of {sorted(_SPLITS)}")
        if self.label not in _LABELS:
            raise ValueError(f"label must be one of {sorted(_LABELS)}")


@dataclass(frozen=True, slots=True)
class ThresholdFit:
    """Deterministic balanced-accuracy threshold fit summary."""

    threshold: float
    balanced_accuracy: float
    normal_false_rejects: int
    defect_false_accepts: int


@dataclass(frozen=True, slots=True)
class TemplateDecision:
    """Structured diagnostic details for one successful Template comparison."""

    view_id: ViewId
    similarity: float
    risk: float
    threshold: float
    best_template_sample_id: str
    best_template_path: Path
    offset_x: int
    offset_y: int
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class _TemplateImage:
    sample_id: str
    path: Path
    image: np.ndarray


@dataclass(frozen=True, slots=True)
class _ResidentGroup:
    view_id: ViewId
    model_path: Path
    model_sha256: str
    input_width: int
    input_height: int
    target_width: int
    target_height: int
    max_shift: int
    threshold: float
    templates: tuple[_TemplateImage, ...]


def load_template_manifest(path: Path, *, path_root: Path | None = None) -> tuple[TemplateSample, ...]:
    """Load the Task 4 subset of a physical-part CSV manifest."""
    manifest_path = Path(path).expanduser().resolve()
    root = Path(path_root).expanduser().resolve() if path_root is not None else manifest_path.parent
    try:
        with manifest_path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required = {"sample_id", "part_id", "view_id", "image_path", "split", "label"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                missing = required - set(reader.fieldnames or ())
                raise ValueError(f"template manifest missing fields: {', '.join(sorted(missing))}")
            samples: list[TemplateSample] = []
            identities: set[tuple[str, ViewId]] = set()
            for row_number, row in enumerate(reader, start=2):
                raw_path = Path(_non_empty(row.get("image_path"), f"row {row_number} image_path"))
                image_path = raw_path if raw_path.is_absolute() else root / raw_path
                view_id = _view_id(row.get("view_id"))
                identity = (_non_empty(row.get("sample_id"), f"row {row_number} sample_id"), view_id)
                if identity in identities:
                    raise ValueError(f"duplicate template sample identity: {identity[0]}/{view_id.value}")
                identities.add(identity)
                samples.append(
                    TemplateSample(
                        sample_id=identity[0],
                        part_id=_non_empty(row.get("part_id"), f"row {row_number} part_id"),
                        view_id=view_id,
                        image_path=image_path,
                        split=_non_empty(row.get("split"), f"row {row_number} split"),
                        label=_non_empty(row.get("label"), f"row {row_number} label"),
                    )
                )
    except OSError as error:
        raise ValueError(f"cannot read template manifest {manifest_path}: {error}") from error
    if not samples:
        raise ValueError("template manifest must not be empty")
    _validate_part_split_isolation(samples)
    return tuple(samples)


def validate_template_source_sessions(
    rows: Sequence[TemplateSample],
    allowed_session_ids: Sequence[str],
) -> tuple[str, ...]:
    """Fail closed unless every manifest crop is named by an allowed capture session."""
    allowed = tuple(allowed_session_ids)
    if not allowed or len(set(allowed)) != len(allowed) or any(
        not isinstance(session_id, str) or not _SOURCE_SESSION.fullmatch(f"{session_id}__")
        for session_id in allowed
    ):
        raise ValueError("allowed session IDs must be unique capture session IDs")
    sources: set[str] = set()
    for row in rows:
        match = _SOURCE_SESSION.match(row.image_path.name)
        if match is None:
            raise ValueError(f"template crop filename does not encode a source session: {row.image_path}")
        session_id = match.group("session_id")
        if session_id not in allowed:
            raise ValueError(f"template source session {session_id} is not in the allowed session list")
        sources.add(session_id)
    if not sources:
        raise ValueError("template source session validation requires at least one row")
    return tuple(sorted(sources))


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"could not decode template image: {path}")
    if image.dtype != np.uint8 or image.ndim not in {2, 3}:
        raise ValueError(f"template image must be a uint8 grayscale/BGR image: {path}")
    return image


def _crop(image: np.ndarray, roi: tuple[int, int, int, int], name: str) -> np.ndarray:
    if len(roi) != 4 or any(isinstance(value, bool) or not isinstance(value, int) for value in roi):
        raise ValueError(f"{name} ROI must contain four integers")
    x1, y1, x2, y2 = roi
    height, width = image.shape[:2]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"{name} ROI lies outside {width}x{height} image")
    return image[y1:y2, x1:x2]


def _grayscale(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        raise ValueError("template input must use uint8 pixels")
    if image.ndim == 2:
        return image
    if image.ndim != 3:
        raise ValueError("template input must be grayscale or BGR/BGRA")
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError("template input must be grayscale or BGR/BGRA")


def _preprocess(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    gray = _grayscale(image)
    if float(gray.std()) < 1.0:
        raise ValueError("flat template input is not matchable")
    target_width, target_height = target_size
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target_size dimensions must be positive")
    source_height, source_width = gray.shape
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, min(target_width, int(round(source_width * scale))))
    resized_height = max(1, min(target_height, int(round(source_height * scale))))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(gray, (resized_width, resized_height), interpolation=interpolation)
    left = (target_width - resized_width) // 2
    right = target_width - resized_width - left
    top = (target_height - resized_height) // 2
    bottom = target_height - resized_height - top
    fitted = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    return cv2.GaussianBlur(fitted, (3, 3), 0)


def _match(query: np.ndarray, template: np.ndarray, max_shift: int) -> tuple[float, float, int, int]:
    if query.shape != template.shape:
        raise ValueError("preprocessed template dimensions differ")
    padded = cv2.copyMakeBorder(query, max_shift, max_shift, max_shift, max_shift, cv2.BORDER_REFLECT_101)
    response = cv2.matchTemplate(padded, template, cv2.TM_CCOEFF_NORMED)
    _minimum, maximum, _minimum_location, maximum_location = cv2.minMaxLoc(response)
    if not math.isfinite(maximum):
        raise ValueError("template similarity is non-finite")
    similarity = min(1.0, max(-1.0, float(maximum)))
    risk = max(0.0, 1.0 - similarity)
    return similarity, risk, maximum_location[0] - max_shift, maximum_location[1] - max_shift


def _best_match(
    query: np.ndarray,
    templates: Sequence[_TemplateImage],
    max_shift: int,
) -> tuple[_TemplateImage, float, float, int, int]:
    results = [(_match(query, template.image, max_shift), template) for template in templates]
    (similarity, risk, offset_x, offset_y), best = max(results, key=lambda item: item[0][0])
    return best, similarity, risk, offset_x, offset_y


def fit_risk_threshold(rows: Sequence[tuple[str, float]]) -> ThresholdFit:
    """Fit ``risk <= threshold`` using calibration balanced accuracy only."""
    parsed: list[tuple[str, float]] = []
    for label, risk in rows:
        if label not in _LABELS:
            raise ValueError(f"unknown calibration label: {label!r}")
        parsed_risk = _finite(risk, "calibration risk")
        if parsed_risk < 0.0:
            raise ValueError("calibration risk must be non-negative")
        parsed.append((label, parsed_risk))
    normal_count = sum(label == "normal" for label, _risk in parsed)
    defect_count = sum(label == "defect" for label, _risk in parsed)
    if normal_count == 0 or defect_count == 0:
        raise ValueError("calibration requires both normal and defect rows")
    candidates = sorted({risk for _label, risk in parsed})
    candidates.extend((left + right) / 2.0 for left, right in zip(candidates, candidates[1:], strict=False))
    fits: list[ThresholdFit] = []
    for threshold in sorted(set(candidates)):
        normal_false_rejects = sum(label == "normal" and risk > threshold for label, risk in parsed)
        defect_false_accepts = sum(label == "defect" and risk <= threshold for label, risk in parsed)
        true_normal_rate = 1.0 - normal_false_rejects / normal_count
        true_defect_rate = 1.0 - defect_false_accepts / defect_count
        fits.append(
            ThresholdFit(
                threshold=threshold,
                balanced_accuracy=(true_normal_rate + true_defect_rate) / 2.0,
                normal_false_rejects=normal_false_rejects,
                defect_false_accepts=defect_false_accepts,
            )
        )
    return max(fits, key=lambda fit: (fit.balanced_accuracy, -fit.normal_false_rejects, -fit.threshold))


def _select_templates(
    samples: Sequence[TemplateSample],
    images: Sequence[np.ndarray],
    count: int,
) -> tuple[int, ...]:
    count = _validate_template_count(count)
    if len(samples) < 3:
        raise ValueError("at least three train/normal samples are required per view")
    selected_count = min(count, len(samples))
    features = np.stack(
        [cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).reshape(-1) for image in images]
    ).astype(np.float32)
    features -= features.mean(axis=1, keepdims=True)
    features /= features.std(axis=1, keepdims=True) + 1e-6
    distances = np.mean((features[:, None, :] - features[None, :, :]) ** 2, axis=2)
    selected = [int(np.argmin(distances.sum(axis=1)))]
    minimum_distance = distances[selected[0]].copy()
    while len(selected) < selected_count:
        for index in selected:
            minimum_distance[index] = -1.0
        next_index = int(np.argmax(minimum_distance))
        selected.append(next_index)
        minimum_distance = np.minimum(minimum_distance, distances[next_index])
    return tuple(selected)


def _score_samples(
    samples: Sequence[TemplateSample],
    roi: tuple[int, int, int, int],
    target_size: tuple[int, int],
    templates: Sequence[_TemplateImage],
    max_shift: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample in sorted(samples, key=lambda item: item.sample_id):
        query = _preprocess(_crop(_read_image(sample.image_path), roi, sample.sample_id), target_size)
        best, similarity, risk, offset_x, offset_y = _best_match(query, templates, max_shift)
        records.append(
            {
                "sample_id": sample.sample_id,
                "part_id": sample.part_id,
                "view_id": sample.view_id.value,
                "label": sample.label,
                "similarity": similarity,
                "risk": risk,
                "best_template_sample_id": best.sample_id,
                "offset_x": offset_x,
                "offset_y": offset_y,
            }
        )
    return records


def _metrics(records: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    normal_count = sum(record["label"] == "normal" for record in records)
    defect_count = sum(record["label"] == "defect" for record in records)
    normal_false_rejects = sum(
        record["label"] == "normal" and float(record["risk"]) > threshold for record in records
    )
    defect_false_accepts = sum(
        record["label"] == "defect" and float(record["risk"]) <= threshold for record in records
    )
    balanced_accuracy = None
    if normal_count and defect_count:
        balanced_accuracy = (
            1.0
            - normal_false_rejects / normal_count
            + 1.0
            - defect_false_accepts / defect_count
        ) / 2.0
    return {
        "row_count": len(records),
        "normal_count": normal_count,
        "defect_count": defect_count,
        "normal_false_rejects": normal_false_rejects,
        "defect_false_accepts": defect_false_accepts,
        "balanced_accuracy": balanced_accuracy,
    }


def _write_group(
    rows: Sequence[TemplateSample],
    roi: tuple[int, int, int, int],
    output_dir: Path,
    *,
    target_size: tuple[int, int],
    max_shift: int,
    template_count: int,
    fixed_threshold: float | None = None,
) -> None:
    if not rows:
        raise ValueError("template group rows must not be empty")
    _validate_part_split_isolation(rows)
    view_ids = {row.view_id for row in rows}
    if len(view_ids) != 1:
        raise ValueError("one template group may contain only one view_id")
    view_id = next(iter(view_ids))
    if isinstance(max_shift, bool) or not isinstance(max_shift, int) or max_shift < 0:
        raise ValueError("max_shift must be a non-negative integer")
    all_train_rows = [row for row in rows if row.split == "train"]
    if fixed_threshold is None and any(row.label != "normal" for row in all_train_rows):
        raise ValueError("Template train split may contain only normal rows")
    train_rows = sorted(
        (row for row in all_train_rows if row.label == "normal"),
        key=lambda item: item.sample_id,
    )
    calibration_rows = [
        row for row in rows if row.split == "calibration" and (fixed_threshold is None or row.label == "normal")
    ]
    final_rows = [
        row for row in rows if row.split == "final_test" and (fixed_threshold is None or row.label == "normal")
    ]
    if fixed_threshold is not None and (not calibration_rows or not final_rows):
        raise ValueError("fixed-threshold Template scoring requires normal calibration and final_test rows")
    x1, y1, x2, y2 = roi
    target_width, target_height = target_size
    train_images = [
        _preprocess(_crop(_read_image(row.image_path), roi, row.sample_id), target_size) for row in train_rows
    ]
    selected_indices = _select_templates(train_rows, train_images, template_count)
    template_dir = output_dir / "templates"
    template_dir.mkdir(parents=True)
    resident_templates: list[_TemplateImage] = []
    template_metadata: list[dict[str, Any]] = []
    for output_index, sample_index in enumerate(selected_indices):
        sample = train_rows[sample_index]
        template_path = template_dir / f"template_{output_index:02d}.png"
        if not cv2.imwrite(str(template_path), train_images[sample_index]):
            raise OSError(f"could not write template image: {template_path}")
        resident_templates.append(
            _TemplateImage(sample_id=sample.sample_id, path=template_path, image=train_images[sample_index])
        )
        template_metadata.append(
            {
                "part_id": sample.part_id,
                "path": template_path.relative_to(output_dir).as_posix(),
                "sample_id": sample.sample_id,
                "sha256": _sha256(template_path),
            }
        )
    calibration_records = _score_samples(
        calibration_rows,
        roi,
        target_size,
        resident_templates,
        max_shift,
    )
    threshold = _finite(fixed_threshold, "fixed Template threshold") if fixed_threshold is not None else None
    if threshold is not None and threshold < 0.0:
        raise ValueError("fixed Template threshold must be non-negative")
    fit = (
        None
        if threshold is not None
        else fit_risk_threshold([(str(record["label"]), float(record["risk"])) for record in calibration_records])
    )
    resolved_threshold = threshold if threshold is not None else fit.threshold
    for record in calibration_records:
        record["threshold"] = resolved_threshold
        record["prediction"] = "normal" if float(record["risk"]) <= resolved_threshold else "defect"
    final_records = _score_samples(final_rows, roi, target_size, resident_templates, max_shift)
    for record in final_records:
        record["threshold"] = resolved_threshold
        record["prediction"] = "normal" if float(record["risk"]) <= resolved_threshold else "defect"
    model = {
        "schema_version": 1,
        "view_id": view_id.value,
        "input_width": x2 - x1,
        "input_height": y2 - y1,
        "preprocess": {
            "grayscale": True,
            "target_width": target_width,
            "target_height": target_height,
            "blur_kernel": 3,
            "match_method": "TM_CCOEFF_NORMED",
            "padding": "BORDER_REFLECT_101",
            "max_shift": max_shift,
        },
        "threshold": resolved_threshold,
        "threshold_fit": {
            "split": "baseline_fixed" if fit is None else "calibration",
            "metric": "fixed_baseline_threshold" if fit is None else "balanced_accuracy",
            "balanced_accuracy": None if fit is None else fit.balanced_accuracy,
            "normal_false_rejects": _metrics(calibration_records, resolved_threshold)["normal_false_rejects"]
            if fit is None
            else fit.normal_false_rejects,
            "defect_false_accepts": None if fit is None else fit.defect_false_accepts,
            "final_test_used": False,
        },
        "templates": template_metadata,
    }
    model_path = output_dir / "model.json"
    _write_json(model_path, model)
    (output_dir / "model.sha256").write_text(_sha256(model_path) + "\n", encoding="ascii")
    with (output_dir / "calibration_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_CALIBRATION_FIELDS)
        writer.writeheader()
        writer.writerows(calibration_records)
    _write_json(
        output_dir / "metrics.json",
        {
            "schema_version": 1,
            "view_id": view_id.value,
            "threshold": resolved_threshold,
            "calibration": _metrics(calibration_records, resolved_threshold),
            "calibration_rows": calibration_records,
            "final_test": _metrics(final_records, resolved_threshold),
            "final_test_rows": final_records,
        },
    )


def _atomic_group_training(
    rows: Sequence[TemplateSample],
    roi: tuple[int, int, int, int],
    output_dir: Path,
    *,
    target_size: tuple[int, int],
    max_shift: int,
    template_count: int,
) -> Path:
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refuse to overwrite existing template model directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        _write_group(
            rows,
            roi,
            temporary,
            target_size=target_size,
            max_shift=max_shift,
            template_count=template_count,
        )
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination / "model.json"


def train_template_group(
    rows: Sequence[TemplateSample],
    roi: tuple[int, int, int, int],
    output_dir: Path,
    *,
    target_size: tuple[int, int] = (512, 512),
    max_shift: int = 12,
    template_count: int = 5,
) -> Path:
    """Train one immutable per-view model directory."""
    template_count = _validate_template_count(template_count)
    _validate_part_split_isolation(rows)
    return _atomic_group_training(
        rows,
        roi,
        output_dir,
        target_size=target_size,
        max_shift=max_shift,
        template_count=template_count,
    )


def train_template_group_fixed_threshold(
    rows: Sequence[TemplateSample],
    roi: tuple[int, int, int, int],
    output_dir: Path,
    *,
    threshold: float,
    target_size: tuple[int, int] = (512, 512),
    max_shift: int = 12,
    template_count: int = 5,
) -> Path:
    """Train immutable templates from train/normal rows and reuse a fixed threshold."""
    template_count = _validate_template_count(template_count)
    _validate_part_split_isolation(rows)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refuse to overwrite existing template model directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        _write_group(
            rows,
            roi,
            temporary,
            target_size=target_size,
            max_shift=max_shift,
            template_count=template_count,
            fixed_threshold=threshold,
        )
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination / "model.json"


def train_template_groups(
    rows: Sequence[TemplateSample],
    part_rois: Mapping[ViewId, tuple[int, int, int, int]],
    output_root: Path,
    *,
    target_size: tuple[int, int] = (512, 512),
    max_shift: int = 12,
    template_count: int = 5,
) -> Mapping[ViewId, Path]:
    """Train all six independent views and publish the root atomically."""
    template_count = _validate_template_count(template_count)
    _validate_part_split_isolation(rows)
    destination = Path(output_root).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refuse to overwrite existing template model directory: {destination}")
    normalized_rois = {_view_id(view): roi for view, roi in part_rois.items()}
    if set(normalized_rois) != set(ViewId):
        raise ValueError("part_rois must contain exactly the six BMW views")
    rows_by_view = {view_id: [row for row in rows if row.view_id is view_id] for view_id in ViewId}
    if any(not group_rows for group_rows in rows_by_view.values()):
        raise ValueError("template manifest must contain rows for all six BMW views")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for view_id in ViewId:
            _write_group(
                rows_by_view[view_id],
                normalized_rois[view_id],
                temporary / view_id.value,
                target_size=target_size,
                max_shift=max_shift,
                template_count=template_count,
            )
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return MappingProxyType({view_id: destination / view_id.value / "model.json" for view_id in ViewId})


def _load_group(model_path: Path, threshold_override: float | None) -> _ResidentGroup:
    candidate = Path(model_path).expanduser().resolve()
    resolved = candidate / "model.json" if candidate.is_dir() else candidate
    if not resolved.is_file():
        raise ValueError(f"missing Template model.json: {resolved}")
    sha_path = resolved.parent / "model.sha256"
    if not sha_path.is_file():
        raise ValueError(f"missing Template model.sha256: {sha_path}")
    expected_model_sha = sha_path.read_text(encoding="ascii").strip()
    actual_model_sha = _sha256(resolved)
    if expected_model_sha != actual_model_sha:
        raise ValueError(f"Template model sha256 mismatch: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Template model JSON: {resolved}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"unsupported Template model schema: {resolved}")
    preprocess = payload.get("preprocess")
    templates_raw = payload.get("templates")
    if not isinstance(preprocess, dict) or not isinstance(templates_raw, list) or not 3 <= len(templates_raw) <= 5:
        raise ValueError(f"invalid Template model structure: {resolved}")
    target_width = int(preprocess.get("target_width", 0))
    target_height = int(preprocess.get("target_height", 0))
    max_shift = int(preprocess.get("max_shift", -1))
    if target_width <= 0 or target_height <= 0 or max_shift < 0:
        raise ValueError(f"invalid Template preprocessing contract: {resolved}")
    templates: list[_TemplateImage] = []
    for index, item in enumerate(templates_raw):
        if not isinstance(item, dict):
            raise ValueError(f"invalid Template metadata at index {index}: {resolved}")
        relative_path = Path(_non_empty(item.get("path"), f"templates[{index}].path"))
        template_path = (resolved.parent / relative_path).resolve()
        if not template_path.is_file():
            raise ValueError(f"missing template image: {template_path}")
        if _sha256(template_path) != item.get("sha256"):
            raise ValueError(f"template sha256 mismatch: {template_path}")
        image = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (target_height, target_width) or float(image.std()) < 1.0:
            raise ValueError(f"invalid template image: {template_path}")
        image.flags.writeable = False
        templates.append(
            _TemplateImage(
                sample_id=_non_empty(item.get("sample_id"), f"templates[{index}].sample_id"),
                path=template_path,
                image=image,
            )
        )
    threshold = payload.get("threshold") if threshold_override is None else threshold_override
    parsed_threshold = _finite(threshold, "Template threshold")
    if parsed_threshold < 0.0:
        raise ValueError("Template threshold must be non-negative")
    input_width = int(payload.get("input_width", 0))
    input_height = int(payload.get("input_height", 0))
    if input_width <= 0 or input_height <= 0:
        raise ValueError(f"invalid Template input dimensions: {resolved}")
    return _ResidentGroup(
        view_id=_view_id(payload.get("view_id")),
        model_path=resolved,
        model_sha256=actual_model_sha,
        input_width=input_width,
        input_height=input_height,
        target_width=target_width,
        target_height=target_height,
        max_shift=max_shift,
        threshold=parsed_threshold,
        templates=tuple(templates),
    )


class TemplateBackend:
    """Resident per-view Template models with structured diagnostic decisions."""

    def __init__(
        self,
        model_paths: Mapping[ViewId, Path],
        *,
        required_for_ok: bool,
        thresholds: Mapping[ViewId, float] | None = None,
    ) -> None:
        if not isinstance(required_for_ok, bool):
            raise TypeError("required_for_ok must be bool")
        normalized_thresholds = {_view_id(view): value for view, value in (thresholds or {}).items()}
        groups: dict[ViewId, _ResidentGroup] = {}
        for raw_view, model_path in model_paths.items():
            view_id = _view_id(raw_view)
            group = _load_group(model_path, normalized_thresholds.get(view_id))
            if group.view_id is not view_id:
                raise ValueError(
                    f"Template model view_id {group.view_id.value} does not match requested {view_id.value}"
                )
            groups[view_id] = group
        if not groups:
            raise ValueError("TemplateBackend requires at least one model")
        self._groups = MappingProxyType(groups)
        self._required_for_ok = required_for_ok
        self._decisions: dict[ViewId, TemplateDecision] = {}

    @classmethod
    def from_model_root(cls, model_root: Path, *, required_for_ok: bool) -> TemplateBackend:
        """Load the six independently published view directories once."""
        root = Path(model_root).expanduser().resolve()
        return cls(
            {view_id: root / view_id.value / "model.json" for view_id in ViewId},
            required_for_ok=required_for_ok,
        )

    @classmethod
    def from_experiment_config(cls, config: LabExperimentConfig) -> TemplateBackend:
        """Use model paths, editable thresholds, and required status from one profile."""
        if not config.template.enabled:
            raise ValueError("Template branch is disabled in the experiment profile")
        paths: dict[ViewId, Path] = {}
        thresholds: dict[ViewId, float] = {}
        for view_id, group in config.template.groups.items():
            if group.model_path is None:
                raise ValueError(f"Template model_path is missing for {view_id.value}")
            paths[view_id] = group.model_path
            thresholds[view_id] = group.threshold
        return cls(
            paths,
            required_for_ok=BranchName.TEMPLATE in config.required_for_ok,
            thresholds=thresholds,
        )

    def decision_for(self, view_id: ViewId) -> TemplateDecision | None:
        """Return structured evidence from the most recent successful comparison."""
        return self._decisions.get(_view_id(view_id))

    def _error(self, group: _ResidentGroup, reason: str, elapsed_ms: float) -> BranchEvidence:
        return BranchEvidence(
            branch=BranchName.TEMPLATE,
            view_id=group.view_id,
            status=BranchStatus.ERROR,
            required_for_ok=self._required_for_ok,
            score=None,
            threshold=group.threshold,
            elapsed_ms=elapsed_ms,
            reason=reason,
            model_id=group.model_sha256,
            artifact_paths={"model": str(group.model_path)},
        )

    def predict(self, view_id: ViewId | str, crop: np.ndarray) -> BranchEvidence:
        """Score one already-cropped fixed view against its resident templates."""
        parsed_view = _view_id(view_id)
        if parsed_view not in self._groups:
            raise ValueError(f"Template model is not loaded for {parsed_view.value}")
        group = self._groups[parsed_view]
        self._decisions.pop(parsed_view, None)
        started = perf_counter()
        try:
            if not isinstance(crop, np.ndarray) or crop.shape[:2] != (group.input_height, group.input_width):
                found = getattr(crop, "shape", None)
                raise ValueError(
                    f"Template input dimensions must be {group.input_width}x{group.input_height}; found {found}"
                )
            query = _preprocess(crop, (group.target_width, group.target_height))
            best, similarity, risk, offset_x, offset_y = _best_match(query, group.templates, group.max_shift)
        except (TypeError, ValueError, cv2.error) as error:
            return self._error(group, str(error), (perf_counter() - started) * 1000.0)
        elapsed_ms = (perf_counter() - started) * 1000.0
        decision = TemplateDecision(
            view_id=parsed_view,
            similarity=similarity,
            risk=risk,
            threshold=group.threshold,
            best_template_sample_id=best.sample_id,
            best_template_path=best.path,
            offset_x=offset_x,
            offset_y=offset_y,
            elapsed_ms=elapsed_ms,
        )
        self._decisions[parsed_view] = decision
        passed = risk <= group.threshold
        return BranchEvidence(
            branch=BranchName.TEMPLATE,
            view_id=parsed_view,
            status=BranchStatus.PASS if passed else BranchStatus.NG,
            required_for_ok=self._required_for_ok,
            score=risk,
            threshold=group.threshold,
            elapsed_ms=elapsed_ms,
            reason=(
                f"Template {'matched' if passed else 'mismatch'}: similarity={similarity:.6f}, "
                f"risk={risk:.6f}, offset=({offset_x},{offset_y}), template={best.sample_id}"
            ),
            model_id=group.model_sha256,
            artifact_paths={
                "model": str(group.model_path),
                "best_template": str(best.path),
            },
        )
