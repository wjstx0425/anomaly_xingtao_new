# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: EM101, EM102, TRY301

"""Fail-closed whole-view template gate for ZS32 inspection."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from zs32_inspection.domain.views import VIEW_ORDER

SCHEMA = "anomalib.zs32_template_gate"
SCHEMA_VERSION = "1.0"
ZS32_VIEWS = VIEW_ORDER
DEFAULT_HANDS = ("left", "right")
MODEL_FILENAME = "model.json"
MODEL_SHA256_FILENAME = "model.sha256"
CALIBRATION_FILENAME = "calibration_rows.csv"
CALIBRATION_FIELDNAMES = (
    "part_id",
    "hand",
    "view",
    "branch",
    "raw_score",
    "gt_label",
    "split",
    "model_version",
    "threshold_version",
    "roi_version",
    "template_version",
)
_GROUP_PATTERN = re.compile(r"(?:^|[_/\\-])(group\d+)(?:[_/\\.-]|$)", re.IGNORECASE)


class TemplateGateError(ValueError):
    """Structured fail-closed template-gate error."""

    def __init__(self, code: str, message: str, **details: object) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def to_result(self) -> dict[str, Any]:
        """Return a machine-readable invalid gate result."""
        return {
            "status": "INVALID_TEMPLATE_GATE",
            "reason": self.message,
            "score": None,
            "threshold": None,
            "error": {"code": self.code, "message": self.message, "details": self.details},
        }


@dataclass(frozen=True)
class TemplateGateResult:
    """Continuous evidence and decision from one template-gate inference."""

    status: str
    reason: str
    score: float
    threshold: float
    image_path: str
    hand: str
    view: str
    similarity: float
    risk_score: float
    low_threshold: float
    high_threshold: float
    best_template: str
    best_template_path: str
    best_template_sha256: str
    offset: tuple[int, int]
    versions: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result without dropping continuous evidence."""
        payload = asdict(self)
        payload.update(
            model_version=self.versions["model"],
            threshold_version=self.versions["threshold"],
            roi_version=self.versions["roi"],
            template_version=self.versions["template"],
        )
        return payload


@dataclass(frozen=True)
class TemplateGate:
    """Reusable first-stage gate adapter for an inspection orchestrator."""

    model_dir: Path

    def evaluate(self, image_path: Path, hand: str, view: str) -> TemplateGateResult:
        """Evaluate one view using the immutable model directory."""
        return predict_template_gate(self.model_dir, image_path, hand=hand, view=view)


@dataclass(frozen=True)
class _Row:
    """Validated input record used during deterministic training."""

    part_id: str
    hand: str
    view: str
    label: str
    split: str
    image_path: Path


def _error(code: str, message: str, **details: object) -> TemplateGateError:
    """Build a structured error with a concise call site."""
    return TemplateGateError(code, message, **details)


def load_gray(path: str | Path, width: int) -> np.ndarray:
    """Load, aspect-resize, and Gaussian-filter one grayscale image."""
    image_path = Path(path)
    if not image_path.is_file():
        raise _error("IMAGE_MISSING", f"image file does not exist: {image_path}", path=str(image_path))
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.ndim != 2 or not image.size:
        raise _error("IMAGE_INVALID", f"cannot decode grayscale image: {image_path}", path=str(image_path))
    if width <= 0 or image.shape[1] <= 0:
        raise _error("MODEL_INVALID", f"resize width must be positive, got {width}", width=width)
    height = max(1, round(image.shape[0] * width / image.shape[1]))
    interpolation = cv2.INTER_AREA if width <= image.shape[1] else cv2.INTER_LINEAR
    resized = cv2.resize(image, (width, height), interpolation=interpolation)
    return cv2.GaussianBlur(resized, (3, 3), 0)


def _validate_match_image(image: np.ndarray, *, role: str) -> None:
    """Reject unusable arrays before OpenCV can emit misleading scores."""
    if image is None or image.ndim != 2 or not image.size:
        raise _error(f"{role.upper()}_INVALID", f"{role} must be a non-empty grayscale image")
    if not np.isfinite(image).all() or float(np.std(image)) <= 1e-8:
        raise _error(f"{role.upper()}_INVALID", f"{role} has no finite image texture")


def match_template(image: np.ndarray, template: np.ndarray, max_shift: int) -> tuple[float, tuple[int, int]]:
    """Return TM_CCOEFF_NORMED similarity and best ``(x, y)`` shift."""
    if max_shift < 0:
        raise _error("MODEL_INVALID", f"max_shift must be non-negative, got {max_shift}")
    _validate_match_image(image, role="image")
    _validate_match_image(template, role="template")
    if image.shape != template.shape:
        image = cv2.resize(image, (template.shape[1], template.shape[0]), interpolation=cv2.INTER_AREA)
    padded = cv2.copyMakeBorder(image, max_shift, max_shift, max_shift, max_shift, cv2.BORDER_REFLECT_101)
    response = cv2.matchTemplate(padded, template, cv2.TM_CCOEFF_NORMED)
    if not np.isfinite(response).all():
        raise _error("SCORE_INVALID", "template matching produced a non-finite similarity")
    _, similarity, _, location = cv2.minMaxLoc(response)
    if not math.isfinite(similarity):
        raise _error("SCORE_INVALID", "template matching produced a non-finite similarity")
    similarity = min(1.0, max(-1.0, float(similarity)))
    return similarity, (int(location[0] - max_shift), int(location[1] - max_shift))


def _required_text(row: Mapping[str, str], names: Sequence[str], row_number: int) -> str:
    """Read the first non-empty manifest alias."""
    for name in names:
        value = row.get(name, "").strip()
        if value:
            return value
    raise _error("MANIFEST_INVALID", f"row {row_number} is missing one of {tuple(names)}")


def _part_id(row: Mapping[str, str], path_text: str, row_number: int) -> str:
    """Read part identity or recover the stable ``groupNNN`` capture identity."""
    for field in ("part_id", "group_id"):
        value = row.get(field, "").strip()
        if value:
            return value
    for candidate in (path_text, row.get("source_path", ""), row.get("session_id", "")):
        match = _GROUP_PATTERN.search(candidate)
        if match:
            context = ":".join(
                value.strip().lower()
                for value in (row.get("hand", ""), row.get("label", ""), row.get("defect_type", ""))
                if value.strip()
            )
            return f"{context}:{match.group(1).lower()}" if context else match.group(1).lower()
    raise _error(
        "MANIFEST_INVALID",
        f"row {row_number} has no part_id/group_id and no groupNNN identity in its paths",
    )


def _label_and_split(row: Mapping[str, str], row_number: int) -> tuple[str, str]:
    """Normalize explicit labels/splits and legacy crop-manifest labels."""
    raw_label = _required_text(row, ("label", "gt_label"), row_number).lower()
    labels = {"0": "normal", "normal": "normal", "normal_test": "normal", "1": "defect", "defect": "defect"}
    try:
        label = labels[raw_label]
    except KeyError as exc:
        raise _error("MANIFEST_INVALID", f"row {row_number} has unsupported label: {raw_label!r}") from exc
    split = row.get("split", "").strip().lower()
    if not split:
        split = "test" if raw_label == "normal_test" else "calibration"
    if split not in {"train", "model_val", "calibration", "final_test", "test"}:
        raise _error("MANIFEST_INVALID", f"row {row_number} has unsupported split: {split!r}")
    return label, split


def load_training_rows(
    manifest: Path,
    *,
    path_root: Path | None = None,
    evaluation_fraction: float = 0.2,
) -> list[_Row]:
    """Load a generic or ZS32 crop manifest and enforce part-level split isolation.

    Args:
        manifest (Path): CSV containing image paths, identities, labels, and optional explicit roles.
        path_root (Path | None): Optional base directory for relative image paths.
        evaluation_fraction (float): Legacy deterministic test fraction when explicit roles are absent.

    Returns:
        list[_Row]: Validated rows in deterministic group and path order.

    Raises:
        TemplateGateError: If the manifest is missing, malformed, leaking identities, or has invalid roles.
    """
    if not manifest.is_file():
        raise _error("MANIFEST_MISSING", f"manifest does not exist: {manifest}", path=str(manifest))
    if not math.isfinite(evaluation_fraction) or not 0 <= evaluation_fraction < 1:
        raise _error("ARGUMENT_INVALID", "evaluation_fraction must be finite and in [0, 1)")
    root = manifest.parent if path_root is None else path_root
    rows: list[_Row] = []
    split_by_part: dict[str, str] = {}
    with manifest.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise _error("MANIFEST_INVALID", f"manifest has no header: {manifest}")
        for row_number, raw in enumerate(reader, start=2):
            row = {str(key): ("" if value is None else str(value)) for key, value in raw.items()}
            path_text = _required_text(row, ("image_path", "output_path", "source_path"), row_number)
            image_path = Path(path_text)
            if not image_path.is_absolute():
                image_path = root / image_path
            hand = _required_text(row, ("hand",), row_number).lower()
            view = _required_text(row, ("resolved_view", "view", "source_view"), row_number).lower()
            if view not in ZS32_VIEWS:
                raise _error("MANIFEST_INVALID", f"row {row_number} has unsupported ZS32 view: {view!r}")
            label, split = _label_and_split(row, row_number)
            part_id = _part_id(row, path_text, row_number)
            if not row.get("split", "").strip() and row.get("label", "").strip().lower() != "normal_test":
                bucket = int.from_bytes(hashlib.sha256(part_id.encode()).digest()[:8], "big") / 2**64
                split = "test" if bucket < evaluation_fraction else "calibration"
            previous_split = split_by_part.setdefault(part_id, split)
            if previous_split != split:
                raise _error(
                    "SPLIT_LEAKAGE",
                    f"physical part {part_id!r} appears in multiple splits: {previous_split!r}, {split!r}",
                    part_id=part_id,
                )
            rows.append(_Row(part_id, hand, view, label, split, image_path.resolve()))
    if not rows:
        raise _error("MANIFEST_INVALID", f"manifest has no data rows: {manifest}")
    return sorted(
        rows,
        key=lambda item: (item.hand, item.view, item.split, item.label, item.part_id, str(item.image_path)),
    )


def _choose_templates(
    rows: Sequence[_Row],
    *,
    width: int,
    count: int,
    max_samples: int,
) -> list[tuple[_Row, np.ndarray]]:
    """Select deterministic representative and diverse normal templates."""
    candidates = list(rows)
    if len(candidates) > max_samples:
        indexes = np.linspace(0, len(candidates) - 1, max_samples, dtype=int)
        candidates = [candidates[int(index)] for index in indexes]
    images = [load_gray(row.image_path, width) for row in candidates]
    features = np.stack(
        [cv2.resize(image, (64, 32), interpolation=cv2.INTER_AREA).reshape(-1) for image in images],
    ).astype(np.float32)
    features -= features.mean(axis=1, keepdims=True)
    features /= features.std(axis=1, keepdims=True) + 1e-6
    center = np.median(features, axis=0)
    selected = [int(np.argmin(np.mean((features - center) ** 2, axis=1)))]
    minimum_distance = np.mean((features - features[selected[0]]) ** 2, axis=1)
    while len(selected) < min(count, len(candidates)):
        next_index = int(np.argmax(minimum_distance))
        if next_index in selected:
            break
        selected.append(next_index)
        minimum_distance = np.minimum(minimum_distance, np.mean((features - features[next_index]) ** 2, axis=1))
    return [(candidates[index], images[index]) for index in selected]


def _partition_normal_rows(rows: Sequence[_Row]) -> tuple[list[_Row], list[_Row]]:
    """Split normal parts into disjoint template-fit and threshold sets."""
    part_ids = sorted({row.part_id for row in rows})
    if len(part_ids) < 2:
        raise _error(
            "CALIBRATION_INSUFFICIENT",
            "each template group needs at least two normal calibration parts",
            normal_part_count=len(part_ids),
        )
    template_part_ids = set(part_ids[::2])
    threshold_part_ids = set(part_ids[1::2])
    template_rows = [row for row in rows if row.part_id in template_part_ids]
    threshold_rows = [row for row in rows if row.part_id in threshold_part_ids]
    if not template_rows or not threshold_rows:
        raise _error("CALIBRATION_INSUFFICIENT", "normal template and threshold sets must both be non-empty")
    return template_rows, threshold_rows


def _best_match(
    image: np.ndarray,
    templates: Sequence[tuple[str, np.ndarray]],
    max_shift: int,
) -> tuple[float, str, tuple[int, int]]:
    """Return the best finite result across all templates."""
    if not templates:
        raise _error("MODEL_INVALID", "template group is empty")
    matches = [(match_template(image, template, max_shift), path) for path, template in templates]
    (score, offset), path = max(matches, key=lambda item: (item[0][0], item[1]))
    return score, path, offset


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    """Return the same inclusive nearest-rank quantile used by Stage 31."""
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def _validate_versions(versions: Mapping[str, Any]) -> dict[str, str]:
    """Validate the complete deployment version contract."""
    required = ("model", "threshold", "roi", "template")
    result = {field: str(versions.get(field, "")).strip() for field in required}
    if any(not result[field] for field in required):
        raise _error("MODEL_INVALID", f"model versions must contain non-empty {required}")
    return result


def train_template_gate(
    manifest: Path,
    output_dir: Path,
    *,
    path_root: Path | None = None,
    required_hands: Sequence[str] = DEFAULT_HANDS,
    width: int = 512,
    gaussian_kernel: int = 3,
    max_shift: int = 12,
    templates_per_group: int = 5,
    max_train_per_group: int = 120,
    normal_quantile: float = 0.995,
    normal_only: bool = False,
    evaluation_fraction: float = 0.2,
    model_version: str,
    threshold_version: str,
    roi_version: str,
    template_version: str,
) -> dict[str, Any]:
    """Train all required ``(hand, view)`` groups and atomically publish a model.

    Explicit four-role manifests use ``train`` only for template selection, ``calibration`` only for thresholds, and
    reserve ``model_val`` and ``final_test`` for scoring. Legacy ``calibration/test`` manifests retain their original
    deterministic internal partition.

    Args:
        manifest (Path): Template training manifest.
        output_dir (Path): New immutable model directory.
        path_root (Path | None): Optional base directory for relative image paths.
        required_hands (Sequence[str]): Hands for which all eight view groups are required.
        width (int): Aspect-preserving grayscale resize width.
        gaussian_kernel (int): Fixed Gaussian kernel size.
        max_shift (int): Maximum translation searched during matching.
        templates_per_group (int): Maximum templates selected for each hand/view group.
        max_train_per_group (int): Maximum training candidates considered per group.
        normal_quantile (float): Normal-risk quantile used to place the threshold.
        normal_only (bool): Ignore defect rows and publish a binary normal-only gate.
        evaluation_fraction (float): Legacy held-out fraction when explicit roles are absent.
        model_version (str): Model version recorded in the artifact.
        threshold_version (str): Threshold version recorded in the artifact.
        roi_version (str): ROI version recorded in the artifact.
        template_version (str): Template version recorded in the artifact.

    Returns:
        dict[str, Any]: Published model contract.

    Raises:
        TemplateGateError: If arguments, data roles, calibration, or publication are invalid.
    """
    if width <= 0 or gaussian_kernel != 3 or max_shift < 0 or templates_per_group <= 0 or max_train_per_group <= 0:
        raise _error("ARGUMENT_INVALID", "invalid preprocessing or template-selection argument")
    if not math.isfinite(normal_quantile) or not 0 < normal_quantile <= 1:
        raise _error("ARGUMENT_INVALID", "normal_quantile must be finite and in (0, 1]")
    hands = tuple(sorted({str(hand).strip().lower() for hand in required_hands if str(hand).strip()}))
    if not hands:
        raise _error("ARGUMENT_INVALID", "at least one required hand is needed")
    versions = _validate_versions(
        {"model": model_version, "threshold": threshold_version, "roi": roi_version, "template": template_version},
    )
    rows = load_training_rows(manifest, path_root=path_root, evaluation_fraction=evaluation_fraction)
    explicit_four_role = any(row.split in {"train", "model_val", "final_test"} for row in rows)
    if explicit_four_role and any(row.split == "test" for row in rows):
        raise _error("MANIFEST_INVALID", "explicit four-role manifests cannot mix final_test with legacy test")
    grouped: dict[tuple[str, str], list[_Row]] = defaultdict(list)
    for row in rows:
        grouped[row.hand, row.view].append(row)

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise _error("OUTPUT_EXISTS", f"template model output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    model: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "method": "cv2.TM_CCOEFF_NORMED",
        "preprocessing": {
            "color": "grayscale",
            "resize": "aspect_preserving_width",
            "width": width,
            "gaussian_kernel": gaussian_kernel,
            "max_shift": max_shift,
        },
        "versions": versions,
        "threshold_policy": {
            "score": "risk=1-similarity",
            "pass": "risk < low_threshold",
            "review": "low_threshold <= risk < high_threshold",
            "ng_template": "risk >= high_threshold",
            "normal_quantile": normal_quantile,
            "calibration_mode": "normal_only" if normal_only else "normal_and_defect",
            "normal_only_binary_gate": normal_only,
            "evaluation_fraction": evaluation_fraction,
            "role_policy": "explicit_four_role" if explicit_four_role else "legacy_calibration_test",
        },
        "required_hands": list(hands),
        "required_views": list(ZS32_VIEWS),
        "groups": {},
    }
    calibration_rows: list[dict[str, str | float | int]] = []
    try:  # noqa: PLW0717
        for hand in hands:
            for view in ZS32_VIEWS:
                key = (hand, view)
                group_rows = grouped.get(key, [])
                calibration = [row for row in group_rows if row.split == "calibration"]
                normals = [row for row in calibration if row.label == "normal"]
                defects = [] if normal_only else [row for row in calibration if row.label == "defect"]
                if explicit_four_role:
                    template_normals = [
                        row for row in group_rows if row.split == "train" and row.label == "normal"
                    ]
                    threshold_normals = normals
                else:
                    if not normals:
                        raise _error(
                            "CALIBRATION_INSUFFICIENT",
                            f"group {hand}/{view} has no normal calibration data",
                            hand=hand,
                            view=view,
                            missing_class="normal",
                        )
                    template_normals, threshold_normals = _partition_normal_rows(normals)
                if not template_normals or not threshold_normals or (not normal_only and not defects):
                    if not template_normals:
                        missing = "normal training"
                    elif not threshold_normals:
                        missing = "normal calibration"
                    else:
                        missing = "defect calibration"
                    raise _error(
                        "CALIBRATION_INSUFFICIENT",
                        f"group {hand}/{view} has no {missing} data",
                        hand=hand,
                        view=view,
                        missing_class=missing,
                    )
                selected = _choose_templates(
                    template_normals,
                    width=width,
                    count=templates_per_group,
                    max_samples=max_train_per_group,
                )
                template_records: list[dict[str, str]] = []
                loaded_templates: list[tuple[str, np.ndarray]] = []
                template_dir = staging / "templates" / hand / view
                template_dir.mkdir(parents=True, exist_ok=True)
                for index, (source, image) in enumerate(selected, start=1):
                    relative = Path("templates") / hand / view / f"template_{index:02d}.png"
                    destination = staging / relative
                    if not cv2.imwrite(str(destination), image):
                        raise _error("WRITE_FAILED", f"failed to write template: {destination}")
                    relative_text = relative.as_posix()
                    template_records.append(
                        {
                            "path": relative_text,
                            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                            "source_part_id": source.part_id,
                        },
                    )
                    loaded_templates.append((relative_text, image))
                normal_risks = [
                    1.0 - _best_match(load_gray(row.image_path, width), loaded_templates, max_shift)[0]
                    for row in threshold_normals
                ]
                defect_risks = [
                    1.0 - _best_match(load_gray(row.image_path, width), loaded_templates, max_shift)[0]
                    for row in defects
                ]
                normal_threshold = _nearest_rank(normal_risks, normal_quantile)
                if normal_only:
                    if normal_threshold >= 2:
                        raise _error(
                            "CALIBRATION_INVALID",
                            f"group {hand}/{view} normal-only threshold cannot be placed above observed normal risk",
                            normal_threshold=normal_threshold,
                        )
                    low_threshold = math.nextafter(normal_threshold, 2.0)
                    high_threshold = low_threshold
                else:
                    low_threshold = min(defect_risks)
                    high_threshold = max(low_threshold, normal_threshold)
                if not all(math.isfinite(value) for value in (*normal_risks, *defect_risks)):
                    raise _error("CALIBRATION_INVALID", f"group {hand}/{view} produced a non-finite risk")
                if not 0 <= low_threshold <= high_threshold <= 2:
                    raise _error(
                        "CALIBRATION_OVERLAP",
                        f"group {hand}/{view} cannot produce finite ordered dual thresholds",
                        low_threshold=low_threshold,
                        high_threshold=high_threshold,
                    )
                evaluation_splits = {"model_val", "final_test"} if explicit_four_role else {"test"}
                test_rows = [
                    row
                    for row in group_rows
                    if row.split in evaluation_splits and (not normal_only or row.label == "normal")
                ]
                scored_rows = [
                    *((row, risk, 0, "calibration") for row, risk in zip(threshold_normals, normal_risks, strict=True)),
                    *((row, risk, 1, "calibration") for row, risk in zip(defects, defect_risks, strict=True)),
                    *(
                        (
                            row,
                            1.0 - _best_match(load_gray(row.image_path, width), loaded_templates, max_shift)[0],
                            int(row.label == "defect"),
                            row.split,
                        )
                        for row in test_rows
                    ),
                ]
                calibration_rows.extend(
                    {
                        "part_id": row.part_id,
                        "hand": hand,
                        "view": view,
                        "branch": "template_match",
                        "raw_score": risk,
                        "gt_label": gt_label,
                        "split": split,
                        "model_version": versions["model"],
                        "threshold_version": versions["threshold"],
                        "roi_version": versions["roi"],
                        "template_version": versions["template"],
                    }
                    for row, risk, gt_label, split in scored_rows
                )
                model["groups"][f"{hand}/{view}"] = {
                    "low_threshold": low_threshold,
                    "high_threshold": high_threshold,
                    "templates": template_records,
                    "template_normal_count": len(template_normals),
                    "threshold_normal_count": len(threshold_normals),
                    "calibration_normal_count": len(normals),
                    "calibration_defect_count": len(defects),
                    "test_count": len(test_rows),
                    "model_val_count": sum(row.split == "model_val" for row in test_rows),
                    "final_test_count": sum(row.split == "final_test" for row in test_rows),
                }
        model_path = staging / MODEL_FILENAME
        model_path.write_text(
            json.dumps(model, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (staging / MODEL_SHA256_FILENAME).write_text(
            hashlib.sha256(model_path.read_bytes()).hexdigest() + "\n",
            encoding="utf-8",
        )
        with (staging / CALIBRATION_FILENAME).open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=CALIBRATION_FIELDNAMES)
            writer.writeheader()
            writer.writerows(
                sorted(
                    calibration_rows,
                    key=lambda row: (
                        str(row["hand"]),
                        str(row["part_id"]),
                        ZS32_VIEWS.index(str(row["view"])),
                    ),
                ),
            )
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return model


def load_model(model_dir: Path) -> dict[str, Any]:
    """Load and validate the static model-level deployment contract."""
    model_path = model_dir / MODEL_FILENAME
    if not model_path.is_file():
        raise _error("MODEL_MISSING", f"model file does not exist: {model_path}", path=str(model_path))
    digest_path = model_dir / MODEL_SHA256_FILENAME
    if not digest_path.is_file():
        raise _error("MODEL_HASH_MISSING", f"model digest does not exist: {digest_path}", path=str(digest_path))
    expected_digest = digest_path.read_text(encoding="utf-8").strip().lower()
    actual_digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_digest) or expected_digest != actual_digest:
        raise _error("MODEL_HASH_MISMATCH", f"model digest mismatch: {model_path}", path=str(model_path))
    try:
        model = json.loads(
            model_path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise _error("MODEL_INVALID", f"cannot parse finite JSON model: {model_path}") from exc
    if not isinstance(model, dict) or model.get("schema") != SCHEMA or model.get("schema_version") != SCHEMA_VERSION:
        raise _error("MODEL_INVALID", f"unsupported template-gate model schema: {model_path}")
    _validate_versions(model.get("versions", {}))
    preprocessing = model.get("preprocessing")
    groups = model.get("groups")
    if not isinstance(preprocessing, dict) or not isinstance(groups, dict):
        raise _error("MODEL_INVALID", "model preprocessing/groups must be objects")
    return model


def _load_group(model: Mapping[str, Any], hand: str, view: str) -> tuple[Mapping[str, Any], float, float]:
    """Validate one exact group without fallback values."""
    groups = model["groups"]
    key = f"{hand}/{view}"
    group = groups.get(key)
    if not isinstance(group, Mapping):
        raise _error("GROUP_NOT_FOUND", f"model has no exact template group: {key}", hand=hand, view=view)
    try:
        low = float(group["low_threshold"])
        high = float(group["high_threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise _error("MODEL_INVALID", f"group {key} has invalid thresholds") from exc
    if not math.isfinite(low) or not math.isfinite(high) or not 0 <= low <= high <= 2:
        raise _error("MODEL_INVALID", f"group {key} thresholds must satisfy 0 <= low <= high <= 2")
    return group, low, high


def _prepare_template_group(
    model_dir: Path,
    model: Mapping[str, Any],
    hand: str,
    view: str,
) -> tuple[Mapping[str, Any], float, float, list[tuple[str, np.ndarray]], dict[str, str]]:
    """Validate and decode one immutable template group once."""
    group, low, high = _load_group(model, hand, view)
    templates_field = group.get("templates")
    if not isinstance(templates_field, list) or not templates_field:
        raise _error("MODEL_INVALID", f"template group {hand}/{view} has no templates")
    templates: list[tuple[str, np.ndarray]] = []
    template_hashes: dict[str, str] = {}
    for index, record in enumerate(templates_field):
        if not isinstance(record, Mapping) or not str(record.get("path", "")).strip():
            raise _error("MODEL_INVALID", f"template record {index} in {hand}/{view} is invalid")
        relative = str(record["path"])
        relative_path = Path(relative)
        model_root = model_dir.resolve()
        template_path = (model_root / relative_path).resolve()
        if relative_path.is_absolute() or not template_path.is_relative_to(model_root):
            raise _error("MODEL_INVALID", f"template path escapes model directory: {relative}", path=relative)
        if not template_path.is_file():
            raise _error("TEMPLATE_MISSING", f"template file does not exist: {template_path}", path=str(template_path))
        template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise _error("TEMPLATE_INVALID", f"cannot decode template file: {template_path}", path=str(template_path))
        try:
            _validate_match_image(template, role="template")
        except TemplateGateError as exc:
            raise _error(
                "TEMPLATE_INVALID",
                f"invalid template file: {template_path}",
                path=str(template_path),
            ) from exc
        expected_hash = str(record.get("sha256", "")).strip()
        actual_hash = hashlib.sha256(template_path.read_bytes()).hexdigest()
        if not expected_hash or actual_hash != expected_hash:
            raise _error(
                "TEMPLATE_HASH_MISMATCH",
                f"template hash mismatch: {template_path}",
                path=str(template_path),
            )
        templates.append((relative, template))
        template_hashes[relative] = actual_hash
    return group, low, high, templates, template_hashes


def _predict_prepared_template_gate(
    model_dir: Path,
    model: Mapping[str, Any],
    prepared: tuple[Mapping[str, Any], float, float, list[tuple[str, np.ndarray]], dict[str, str]],
    image_path: Path,
    *,
    hand: str,
    view: str,
) -> TemplateGateResult:
    """Evaluate one image against already-validated template arrays."""
    _group, low, high, templates, template_hashes = prepared
    preprocessing = model["preprocessing"]
    try:
        width = int(preprocessing["width"])
        max_shift = int(preprocessing["max_shift"])
    except (KeyError, TypeError, ValueError) as exc:
        raise _error("MODEL_INVALID", "model preprocessing width/max_shift is invalid") from exc
    image = load_gray(image_path, width)
    similarity, best_template, offset = _best_match(image, templates, max_shift)
    risk = 1.0 - similarity
    if not math.isfinite(risk):
        raise _error("SCORE_INVALID", "template risk score is not finite")
    if risk < low:
        status = "PASS"
        reason = "template risk is below the review threshold"
    elif risk < high:
        status = "REVIEW"
        reason = "template risk is inside the manual-review band"
    else:
        status = "NG_TEMPLATE"
        reason = "template risk reached the rejection threshold"
    versions = _validate_versions(model["versions"])
    return TemplateGateResult(
        status=status,
        reason=reason,
        score=risk,
        threshold=high,
        image_path=str(image_path),
        hand=hand,
        view=view,
        similarity=similarity,
        risk_score=risk,
        low_threshold=low,
        high_threshold=high,
        best_template=best_template,
        best_template_path=str((model_dir / best_template).resolve()),
        best_template_sha256=template_hashes[best_template],
        offset=offset,
        versions=versions,
    )


class PreparedTemplateGate:
    """Template gate with model JSON, hashes, and selected groups loaded once."""

    def __init__(self, model_dir: Path, *, hand: str, views: Sequence[str]) -> None:
        self.model_dir = model_dir.expanduser().resolve()
        self.hand = hand.strip().lower()
        self.model = load_model(self.model_dir)
        normalized = tuple(view.strip().lower() for view in views)
        if (
            not normalized
            or any(view not in ZS32_VIEWS for view in normalized)
            or len(set(normalized)) != len(normalized)
        ):
            raise _error("VIEW_INVALID", "prepared template views must be a unique non-empty canonical subset")
        self.groups = {
            view: _prepare_template_group(self.model_dir, self.model, self.hand, view)
            for view in normalized
        }

    def evaluate(self, image_path: Path, hand: str, view: str) -> TemplateGateResult:
        """Evaluate one prepared view without repeating model/template I/O."""
        normalized_hand = hand.strip().lower()
        normalized_view = view.strip().lower()
        if normalized_hand != self.hand or normalized_view not in self.groups:
            raise _error("GROUP_NOT_FOUND", f"template group was not prepared: {normalized_hand}/{normalized_view}")
        return _predict_prepared_template_gate(
            self.model_dir,
            self.model,
            self.groups[normalized_view],
            image_path,
            hand=normalized_hand,
            view=normalized_view,
        )


def predict_template_gate(model_dir: Path, image_path: Path, *, hand: str, view: str) -> TemplateGateResult:
    """Run one whole-view gate and preserve all decision evidence."""
    hand = hand.strip().lower()
    view = view.strip().lower()
    if view not in ZS32_VIEWS:
        raise _error("VIEW_INVALID", f"unsupported ZS32 view: {view!r}", view=view)
    model = load_model(model_dir)
    prepared = _prepare_template_group(model_dir, model, hand, view)
    return _predict_prepared_template_gate(model_dir, model, prepared, image_path, hand=hand, view=view)
