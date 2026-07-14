"""Linux OpenCV backend for the binary ZS32 template gate.

Training selects reference images only.  Threshold fitting remains exclusively
in the calibration workflow.  Runtime preprocessing performs grayscale,
resize, blur, and bounded translation search on the canonical ROI crop; it
never selects a second spatial ROI.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from zs32_inspection.models.base import (
    AssetFile,
    ModelContractError,
    ModelInput,
    ModelSlot,
    require_sha256,
    require_text,
    sha256_file,
)
from zs32_inspection.data.manifests import ADAPTER_BASE_COLUMNS
from zs32_inspection.data.template_export import template_materialized_path
from zs32_inspection.runtime.publisher import (
    AtomicDirectoryPublisher,
    PublicationError,
    canonical_json_bytes,
    tree_checksums,
)

from .artifacts import TemplateAssetSpec
from .predictor import TemplatePredictor, TemplateScore
from .trainer import TemplateTrainSpec

_MODEL_SCHEMA = "zs32.opencv_template"
_MODEL_VERSION = 1
_EXPECTED_PARAMETERS = {
    "width",
    "max_shift",
    "templates_per_group",
    "max_train_samples",
    "template_version",
}


def _safe_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if not relative or pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ModelContractError(f"unsafe materialized template path: {relative!r}")
    path = root.joinpath(*pure.parts)
    if path.is_symlink() or not path.is_file():
        raise ModelContractError(f"materialized template crop is missing: {path}")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ModelContractError(f"materialized template path escapes export: {relative!r}") from error
    return path


def _verify_materialized_export(root: Path, dataset_manifest_digest: str) -> list[dict[str, str]]:
    """Verify the immutable adapter tree and return its strict manifest rows."""
    root = Path(root).resolve()
    checksum_path = root / "checksums.sha256"
    publication_path = root / "publication_root.json"
    if checksum_path.is_symlink() or publication_path.is_symlink():
        raise ModelContractError("materialized template export metadata must not be symlinks")
    try:
        checksum_bytes = checksum_path.read_bytes()
        publication = json.loads(publication_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ModelContractError(f"cannot read materialized template export envelope: {error}") from error
    if (
        not isinstance(publication, dict)
        or set(publication) != {"algorithm", "publication_id", "root_sha256"}
        or publication.get("algorithm") != "sha256(checksums.sha256 bytes)"
        or publication.get("publication_id") != root.name
        or publication.get("root_sha256") != hashlib.sha256(checksum_bytes).hexdigest()
    ):
        raise ModelContractError("materialized template publication root is invalid")
    recorded: dict[str, str] = {}
    try:
        text = checksum_bytes.decode("utf-8")
        if not text.endswith("\n"):
            raise ValueError("checksum index must end with newline")
        for line in text.splitlines():
            digest, relative = line.split("  ", maxsplit=1)
            if relative in recorded or len(digest) != 64:
                raise ValueError("duplicate or malformed checksum entry")
            recorded[relative] = digest
    except (UnicodeError, ValueError) as error:
        raise ModelContractError(f"materialized template checksum index is invalid: {error}") from error
    actual = tree_checksums(
        root,
        excluded=frozenset({"checksums.sha256", "publication_root.json"}),
    )
    if recorded != actual:
        raise ModelContractError("materialized template export checksum tree changed")
    provenance = root / "provenance/dataset_release.json"
    adapter_manifest = root / "provenance/adapter_manifest.csv"
    if sha256_file(provenance) != dataset_manifest_digest:
        raise ModelContractError("template export dataset provenance digest mismatch")
    try:
        dataset = json.loads(provenance.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ModelContractError(f"cannot parse template dataset provenance: {error}") from error
    adapter_digests = dataset.get("adapter_manifest_sha256") if isinstance(dataset, dict) else None
    if (
        not isinstance(adapter_digests, dict)
        or set(adapter_digests) != {"yolo", "anomalib", "template"}
        or sha256_file(adapter_manifest) != adapter_digests["template"]
    ):
        raise ModelContractError("template export adapter provenance is incomplete or inconsistent")
    try:
        with adapter_manifest.open(newline="", encoding="utf-8") as stream:
            adapter_reader = csv.DictReader(stream)
            if tuple(adapter_reader.fieldnames or ()) != ADAPTER_BASE_COLUMNS:
                raise ModelContractError("template source adapter manifest header is invalid")
            adapter_rows = list(adapter_reader)
    except OSError as error:
        raise ModelContractError(f"cannot read template source adapter manifest: {error}") from error
    if not adapter_rows or any(
        set(row) != set(ADAPTER_BASE_COLUMNS) or any(value is None for value in row.values())
        for row in adapter_rows
    ):
        raise ModelContractError("template source adapter manifest rows are malformed")
    manifest = root / "materialized_manifest.csv"
    try:
        with manifest.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            expected_header = (*ADAPTER_BASE_COLUMNS, "materialized_path")
            if tuple(reader.fieldnames or ()) != expected_header:
                raise ModelContractError("template materialized manifest header is invalid")
            rows = list(reader)
    except OSError as error:
        raise ModelContractError(f"cannot read template materialized manifest: {error}") from error
    required = {*ADAPTER_BASE_COLUMNS, "materialized_path"}
    if not rows or any(
        set(row) != required or any(value is None for value in row.values()) for row in rows
    ):
        raise ModelContractError("template materialized manifest schema is invalid")
    expected_rows = [
        {**row, "materialized_path": template_materialized_path(row)}
        for row in adapter_rows
    ]
    if rows != expected_rows:
        raise ModelContractError(
            "template materialized manifest is not the complete canonical adapter export"
        )
    for row in rows:
        relative = PurePosixPath(row["materialized_path"])
        digest = require_sha256(
            row["canonical_crop_sha256"],
            "template.canonical_crop_sha256",
        )
        path = _safe_path(root, relative.as_posix())
        if recorded.get(relative.as_posix()) != digest or sha256_file(path) != digest:
            raise ModelContractError(
                f"template materialized crop is not bound to its canonical digest: {relative}"
            )
    return rows


def _parameters(parameters: Mapping[str, Any]) -> tuple[int, int, int, int, str]:
    if set(parameters) != _EXPECTED_PARAMETERS:
        raise ModelContractError(
            "template training parameter keys invalid; "
            f"missing={sorted(_EXPECTED_PARAMETERS - set(parameters))}, "
            f"unknown={sorted(set(parameters) - _EXPECTED_PARAMETERS)}"
        )
    width = parameters["width"]
    max_shift = parameters["max_shift"]
    count = parameters["templates_per_group"]
    maximum = parameters["max_train_samples"]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (width, max_shift, count, maximum)):
        raise ModelContractError("template width/shift/count parameters must be integers")
    if width <= 0 or max_shift < 0 or count <= 0 or maximum <= 0:
        raise ModelContractError("template width/count must be positive and max_shift non-negative")
    return width, max_shift, count, maximum, require_text(
        parameters["template_version"], "template_version"
    )


def _cv_modules() -> tuple[object, object]:
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError(f"OpenCV template backend dependency is unavailable: {error}") from error
    return cv2, np


def _load_gray(path: Path, width: int) -> object:
    cv2, np = _cv_modules()
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.ndim != 2 or not image.size or not np.isfinite(image).all():
        raise ModelContractError(f"cannot decode finite grayscale template input: {path}")
    if float(np.std(image)) <= 1e-8:
        raise ModelContractError(f"template input has no image texture: {path}")
    height = max(1, round(image.shape[0] * width / image.shape[1]))
    interpolation = cv2.INTER_AREA if width <= image.shape[1] else cv2.INTER_LINEAR
    resized = cv2.resize(image, (width, height), interpolation=interpolation)
    return cv2.GaussianBlur(resized, (3, 3), 0)


def _encode_png(image: object) -> bytes:
    cv2, _ = _cv_modules()
    success, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not success:
        raise RuntimeError("OpenCV failed to encode a reference template")
    return encoded.tobytes()


def _select_templates(paths: Sequence[Path], *, width: int, count: int, maximum: int) -> list[tuple[Path, object]]:
    cv2, np = _cv_modules()
    candidates = list(paths)
    if len(candidates) > maximum:
        indexes = np.linspace(0, len(candidates) - 1, maximum, dtype=int)
        candidates = [candidates[int(index)] for index in indexes]
    images = [_load_gray(path, width) for path in candidates]
    features = np.stack(
        [cv2.resize(image, (64, 32), interpolation=cv2.INTER_AREA).reshape(-1) for image in images]
    ).astype(np.float32)
    features -= features.mean(axis=1, keepdims=True)
    features /= features.std(axis=1, keepdims=True) + 1e-6
    center = np.median(features, axis=0)
    selected = [int(np.argmin(np.mean((features - center) ** 2, axis=1)))]
    distances = np.mean((features - features[selected[0]]) ** 2, axis=1)
    while len(selected) < min(count, len(candidates)):
        index = int(np.argmax(distances))
        if index in selected:
            break
        selected.append(index)
        distances = np.minimum(distances, np.mean((features - features[index]) ** 2, axis=1))
    return [(candidates[index], images[index]) for index in selected]


class OpenCvTemplateTrainingBackend:
    """Select deterministic reference templates and publish one slot candidate."""

    def __call__(self, spec: TemplateTrainSpec) -> TemplateAssetSpec:
        from zs32_inspection.runtime.execution_receipt import (
            ExecutionReceipt,
            verify_execution_receipt_live,
        )

        execution_receipt = ExecutionReceipt.from_mapping(spec.execution_receipt)
        width, max_shift, count, maximum, template_version = _parameters(spec.parameters)
        rows = _verify_materialized_export(spec.dataset_root, spec.dataset_manifest_digest)
        selected_rows = [
            row
            for row in rows
            if row["hand"] == spec.slot.hand
            and row["view"] == spec.slot.view
            and row["split"] == "train"
            and row["label"] == "normal"
        ]
        if not selected_rows:
            raise ModelContractError(f"template training has no train/normal data for {spec.slot.key}")
        part_ids = {row["part_instance_id"] for row in selected_rows}
        if not part_ids:
            raise ModelContractError("template training rows have no physical-part identity")
        paths: list[Path] = []
        source_digest_by_path: dict[Path, str] = {}
        for row in sorted(selected_rows, key=lambda item: (item["part_instance_id"], item["sample_id"])):
            path = _safe_path(spec.dataset_root, row["materialized_path"])
            actual = sha256_file(path)
            if actual != row["canonical_crop_sha256"]:
                raise ModelContractError(f"template source crop digest mismatch: {path}")
            paths.append(path)
            source_digest_by_path[path] = actual
        selected = _select_templates(paths, width=width, count=count, maximum=maximum)
        encoded = [_encode_png(image) for _, image in selected]
        reference_records = [
            {
                "index": index,
                "artifact_sha256": hashlib.sha256(payload).hexdigest(),
                "source_crop_sha256": source_digest_by_path[source],
            }
            for index, ((source, _image), payload) in enumerate(zip(selected, encoded, strict=True), start=1)
        ]
        verify_execution_receipt_live(execution_receipt, device=spec.device)
        framework_version = f"opencv-{execution_receipt.environment.opencv_version}"
        model_payload = {
            "schema": _MODEL_SCHEMA,
            "schema_version": _MODEL_VERSION,
            "hand": spec.slot.hand,
            "view": spec.slot.view,
            "method": "cv2.TM_CCOEFF_NORMED",
            "risk": "1-similarity",
            "preprocessing": {
                "color": "grayscale",
                "resize": "aspect_preserving_width",
                "width": width,
                "gaussian_kernel": 3,
                "max_shift": max_shift,
            },
            "references": reference_records,
            "template_version": template_version,
            "roi_version": spec.roi_version,
            "roi_sha256": spec.roi_digest,
            "dataset_release_id": spec.dataset_release_id,
            "dataset_manifest_sha256": spec.dataset_manifest_digest,
            "train_split_id": spec.train_split_id,
            "recipe_sha256": spec.recipe_digest,
            "train_part_count": len(part_ids),
            "framework_version": framework_version,
            "training_parameters": dict(spec.parameters),
            "execution_receipt": dict(spec.execution_receipt),
        }
        model_bytes = canonical_json_bytes(model_payload)
        model_digest = hashlib.sha256(model_bytes).hexdigest()
        output = spec.output_dir
        if output.name in {"", ".", ".."}:
            raise ModelContractError(f"template output directory is invalid: {output}")
        required = {"model.json", *(f"templates/template_{index:03d}.png" for index in range(1, len(encoded) + 1))}
        with AtomicDirectoryPublisher(output.parent, output.name) as publisher:
            publisher.write_bytes("model.json", model_bytes)
            for index, payload in enumerate(encoded, start=1):
                publisher.write_bytes(f"templates/template_{index:03d}.png", payload)
            published = publisher.finalize(
                validator=lambda staging: _validate_candidate(staging, model_digest, reference_records),
                required_paths=frozenset(required),
            )
        model_asset = AssetFile(
            role="template_model",
            path=published / "model.json",
            sha256=model_digest,
            media_type="application/json",
            logical_path="model.json",
        )
        references = tuple(
            AssetFile(
                role="reference_template",
                path=published / f"templates/template_{index:03d}.png",
                sha256=record["artifact_sha256"],
                media_type="image/png",
                logical_path=f"templates/template_{index:03d}.png",
            )
            for index, record in enumerate(reference_records, start=1)
        )
        return TemplateAssetSpec(
            slot=spec.slot,
            model=model_asset,
            templates=references,
            model_digest=model_digest,
            template_version=template_version,
            roi_version=spec.roi_version,
            roi_digest=spec.roi_digest,
            dataset_release_id=spec.dataset_release_id,
            dataset_manifest_digest=spec.dataset_manifest_digest,
            train_split_id=spec.train_split_id,
            recipe_digest=spec.recipe_digest,
            framework_version=framework_version,
            training_parameters=spec.parameters,
            execution_receipt=spec.execution_receipt,
        )


def _validate_candidate(
    root: Path,
    model_digest: str,
    references: Sequence[Mapping[str, object]],
) -> None:
    if sha256_file(root / "model.json") != model_digest:
        raise PublicationError("template model digest changed in staging")
    for index, record in enumerate(references, start=1):
        if sha256_file(root / f"templates/template_{index:03d}.png") != record["artifact_sha256"]:
            raise PublicationError(f"template reference {index} digest changed in staging")


def _parse_model(asset: TemplateAssetSpec) -> tuple[dict[str, object], tuple[object, ...]]:
    asset.verify()
    try:
        model_bytes = asset.model.path.read_bytes()
        if hashlib.sha256(model_bytes).hexdigest() != asset.model.sha256:
            raise ModelContractError("template model changed at the use boundary")
        payload = json.loads(model_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ModelContractError(f"cannot parse template model {asset.model.path}: {error}") from error
    expected = {
        "schema", "schema_version", "hand", "view", "method", "risk", "preprocessing",
        "references", "template_version", "roi_version", "roi_sha256", "dataset_release_id",
        "dataset_manifest_sha256", "train_split_id", "recipe_sha256", "train_part_count",
        "framework_version", "training_parameters", "execution_receipt",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ModelContractError("template model envelope has missing or unknown fields")
    if (
        payload["schema"] != _MODEL_SCHEMA
        or payload["schema_version"] != _MODEL_VERSION
        or payload["method"] != "cv2.TM_CCOEFF_NORMED"
        or payload["risk"] != "1-similarity"
        or (payload["hand"], payload["view"]) != (asset.slot.hand, asset.slot.view)
        or payload["template_version"] != asset.template_version
        or payload["roi_version"] != asset.roi_version
        or payload["roi_sha256"] != asset.roi_digest
        or payload["dataset_release_id"] != asset.dataset_release_id
        or payload["dataset_manifest_sha256"] != asset.dataset_manifest_digest
        or payload["train_split_id"] != asset.train_split_id
        or payload["recipe_sha256"] != asset.recipe_digest
        or payload["framework_version"] != asset.framework_version
        or payload["training_parameters"] != dict(asset.training_parameters)
        or payload["execution_receipt"] != dict(asset.execution_receipt)
        or isinstance(payload["train_part_count"], bool)
        or not isinstance(payload["train_part_count"], int)
        or payload["train_part_count"] <= 0
    ):
        raise ModelContractError("template model identity differs from its asset descriptor")
    raw_refs = payload["references"]
    if not isinstance(raw_refs, list) or len(raw_refs) != len(asset.templates):
        raise ModelContractError("template model reference set is incomplete")
    expected_ref_keys = {"index", "artifact_sha256", "source_crop_sha256"}
    images: list[object] = []
    for index, (record, reference) in enumerate(zip(raw_refs, asset.templates, strict=True), start=1):
        if (
            not isinstance(record, dict)
            or set(record) != expected_ref_keys
            or record["index"] != index
            or record["artifact_sha256"] != reference.sha256
            or not isinstance(record["source_crop_sha256"], str)
            or len(record["source_crop_sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in record["source_crop_sha256"])
        ):
            raise ModelContractError("template model reference identity mismatch")
        images.append(_load_reference(reference))
    return payload, tuple(images)


def _load_reference(asset: AssetFile) -> object:
    cv2, np = _cv_modules()
    try:
        content = asset.path.read_bytes()
    except OSError as error:
        raise ModelContractError(f"cannot read reference template {asset.path}: {error}") from error
    if hashlib.sha256(content).hexdigest() != asset.sha256:
        raise ModelContractError(f"reference template changed at the use boundary: {asset.path}")
    encoded = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None or image.ndim != 2 or not image.size or not np.isfinite(image).all():
        raise ModelContractError(f"invalid reference template image: {asset.path}")
    if float(np.std(image)) <= 1e-8:
        raise ModelContractError(f"reference template has no image texture: {asset.path}")
    return image


def _best_match(image: object, templates: Sequence[object], max_shift: int) -> tuple[float, int, tuple[int, int]]:
    cv2, np = _cv_modules()
    matches: list[tuple[float, int, tuple[int, int]]] = []
    for index, template in enumerate(templates):
        candidate = image
        if candidate.shape != template.shape:
            candidate = cv2.resize(candidate, (template.shape[1], template.shape[0]), interpolation=cv2.INTER_AREA)
        padded = cv2.copyMakeBorder(candidate, max_shift, max_shift, max_shift, max_shift, cv2.BORDER_REFLECT_101)
        response = cv2.matchTemplate(padded, template, cv2.TM_CCOEFF_NORMED)
        if not np.isfinite(response).all():
            raise ModelContractError("template matching produced non-finite output")
        _minimum, maximum, _minimum_location, location = cv2.minMaxLoc(response)
        similarity = min(1.0, max(-1.0, float(maximum)))
        matches.append((similarity, index, (int(location[0] - max_shift), int(location[1] - max_shift))))
    if not matches:
        raise ModelContractError("template predictor contains no references")
    return max(matches, key=lambda item: (item[0], -item[1]))


class OpenCvTemplatePredictor(TemplatePredictor):
    """Preloaded per-slot OpenCV reference sets."""

    def __init__(self, assets: Mapping[ModelSlot, TemplateAssetSpec]) -> None:
        if not assets:
            raise ModelContractError("template runtime requires at least one slot")
        parsed: dict[ModelSlot, tuple[TemplateAssetSpec, dict[str, object], tuple[object, ...]]] = {}
        for slot, asset in assets.items():
            if slot != asset.slot:
                raise ModelContractError(f"template runtime key differs from asset slot: {slot.key}")
            model, templates = _parse_model(asset)
            parsed[slot] = (asset, model, templates)
        self._assets = MappingProxyType(parsed)

    def score_batch(
        self,
        samples: Sequence[ModelInput],
        *,
        inspection_id: str,
    ) -> Sequence[TemplateScore]:
        inspection_id = require_text(inspection_id, "inspection_id")
        materialized = tuple(samples)
        if not materialized:
            raise ModelContractError("template scoring requires canonical crops")
        if len({item.slot for item in materialized}) != len(materialized):
            raise ModelContractError("template scoring batch contains duplicate hand/view slots")
        output: list[TemplateScore] = []
        for model_input in materialized:
            model_input.verify_crop()
            try:
                asset, model, templates = self._assets[model_input.slot]
            except KeyError as error:
                raise ModelContractError(f"template runtime has no slot {model_input.slot.key}") from error
            if (
                model_input.sample.roi_config_id != asset.roi_version
                or model_input.roi_digest != asset.roi_digest
            ):
                raise ModelContractError(f"template runtime ROI mismatch for {model_input.slot.key}")
            preprocessing = model["preprocessing"]
            if not isinstance(preprocessing, dict) or set(preprocessing) != {
                "color", "resize", "width", "gaussian_kernel", "max_shift"
            }:
                raise ModelContractError("template preprocessing contract is malformed")
            if (
                preprocessing["color"] != "grayscale"
                or preprocessing["resize"] != "aspect_preserving_width"
                or preprocessing["gaussian_kernel"] != 3
                or isinstance(preprocessing["width"], bool)
                or not isinstance(preprocessing["width"], int)
                or isinstance(preprocessing["max_shift"], bool)
                or not isinstance(preprocessing["max_shift"], int)
                or preprocessing["width"] <= 0
                or preprocessing["max_shift"] < 0
            ):
                raise ModelContractError("template preprocessing values are invalid")
            image = _load_gray(model_input.crop_path, preprocessing["width"])
            similarity, reference_index, offset = _best_match(
                image,
                templates,
                preprocessing["max_shift"],
            )
            risk = 1.0 - similarity
            reference = asset.templates[reference_index]
            sample = model_input.sample
            output.append(
                TemplateScore(
                    inspection_id=inspection_id,
                    part_instance_id=sample.part.part_instance_id,
                    capture_set_id=sample.capture_set_id,
                    hand=sample.part.hand.value,
                    view=sample.view_id,
                    risk_score=risk,
                    similarity=similarity,
                    model_digest=asset.model_digest,
                    roi_config_id=sample.roi_config_id,
                    roi_digest=model_input.roi_digest,
                    source_sha256=sample.source_sha256,
                    crop_sha256=sample.crop_sha256,
                    best_template_path=reference.path,
                    best_template_sha256=reference.sha256,
                    offset_xy=offset,
                )
            )
        return tuple(output)


__all__ = ["OpenCvTemplatePredictor", "OpenCvTemplateTrainingBackend"]
