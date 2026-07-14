# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Linux/NVIDIA Anomalib backend for the three ZS32 anomaly families.

The product-image boundary is deliberately narrow: training is allowed to read
only the verified ``<hand>/<view>/train/good`` partition of one immutable,
materialized Anomalib export.  The canonical ROI crop has already happened in
the data layer; this backend only resizes and, where required by the backbone,
normalizes pixels.  It never performs a center crop or any other spatial crop.

Heavy framework imports stay inside Linux-only call paths so importing the ZS32
contracts on a development Mac does not initialize torch, CUDA, or Anomalib.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import stat
import tempfile
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from zs32_inspection.domain.contracts import AnomalyFamily
from zs32_inspection.data.anomalib_export import anomalib_materialized_path
from zs32_inspection.data.manifests import ADAPTER_BASE_COLUMNS
from zs32_inspection.runtime.publisher import (
    AtomicDirectoryPublisher,
    PublicationError,
    canonical_json_bytes,
    sha256_file,
    tree_checksums,
)

from .base import (
    AnomalyModelArtifact,
    AnomalyPredictor,
    AssetFile,
    BackendUnavailableError,
    DeviceSpec,
    ModelContractError,
    ModelInput,
    ModelSlot,
    RawModelScore,
    TrainSpec,
    canonical_sha256,
    require_sha256,
    require_text,
)

BACKEND_SCHEMA = "zs32.anomalib_backend"
BACKEND_SCHEMA_VERSION = 2
MATERIALIZED_COLUMNS = (
    "sample_id",
    "part_instance_id",
    "capture_set_id",
    "hand",
    "view",
    "split",
    "label",
    "defect_type",
    "canonical_crop_path",
    "canonical_crop_sha256",
    "materialized_path",
)
_ANOMALYDINO_BUILD_LOCK = threading.Lock()


class AnomalibBackendError(RuntimeError):
    """A verified job could not safely complete in the Anomalib runtime."""


@dataclass(frozen=True, slots=True)
class _VerifiedTrainingPartition:
    root: Path
    normal_dir: Path
    manifest_sha256: str
    sample_count: int


@dataclass(frozen=True, slots=True)
class _LoadedSlot:
    artifact: AnomalyModelArtifact
    model: Any
    engine: Any
    image_size: tuple[int, int]


def _safe_relative(value: str, field: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in value
        or any("\x00" in part or "\n" in part or "\r" in part for part in path.parts)
    ):
        raise ModelContractError(f"{field} must be a safe relative POSIX path: {value!r}")
    return path


def _require_regular(path: Path, field: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ModelContractError(f"{field} must be a regular non-symlink file: {path}")
    metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ModelContractError(f"{field} must be a private regular file: {path}")
    return path


def _read_json(path: Path, field: str) -> dict[str, Any]:
    _require_regular(path, field)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelContractError(f"cannot read {field}: {path}: {error}") from error
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ModelContractError(f"{field} must contain a JSON object")
    return dict(value)


def _parse_checksum_index(payload: bytes) -> dict[str, str]:
    checksums: dict[str, str] = {}
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ModelContractError("materialized checksums.sha256 is not UTF-8") from error
    if not lines:
        raise ModelContractError("materialized checksums.sha256 is empty")
    for line in lines:
        if len(line) < 67 or line[64:66] != "  ":
            raise ModelContractError(f"malformed materialized checksum row: {line!r}")
        digest = require_sha256(line[:64], "materialized.checksum")
        relative = _safe_relative(line[66:], "materialized.checksum_path").as_posix()
        if relative in {"checksums.sha256", "publication_root.json"} or relative in checksums:
            raise ModelContractError(f"invalid or duplicate materialized checksum path: {relative}")
        checksums[relative] = digest
    return checksums


def _verify_publication_tree(root: Path) -> Mapping[str, str]:
    expanded = root.expanduser()
    if expanded.is_symlink() or not expanded.is_dir():
        raise ModelContractError(f"materialized export must be a non-symlink directory: {expanded}")
    resolved = expanded.resolve()
    for path in sorted(resolved.rglob("*")):
        if path.is_symlink():
            raise ModelContractError(f"symlink is forbidden in materialized export: {path}")
        metadata = path.stat(follow_symlinks=False)
        if stat.S_ISREG(metadata.st_mode):
            if metadata.st_nlink != 1:
                raise ModelContractError(f"hard-linked materialized file is forbidden: {path}")
        elif not stat.S_ISDIR(metadata.st_mode):
            raise ModelContractError(f"special filesystem node is forbidden: {path}")
    checksum_path = _require_regular(resolved / "checksums.sha256", "materialized checksum index")
    publication_path = _require_regular(
        resolved / "publication_root.json",
        "materialized publication root",
    )
    checksum_bytes = checksum_path.read_bytes()
    expected = _parse_checksum_index(checksum_bytes)
    try:
        actual = tree_checksums(
            resolved,
            excluded=frozenset({"checksums.sha256", "publication_root.json"}),
        )
    except PublicationError as error:
        raise ModelContractError(f"cannot verify materialized export tree: {error}") from error
    if dict(actual) != dict(expected):
        raise ModelContractError("materialized export tree differs from checksums.sha256")
    publication = _read_json(publication_path, "materialized publication root")
    if set(publication) != {"algorithm", "publication_id", "root_sha256"}:
        raise ModelContractError("materialized publication_root.json keys are invalid")
    if (
        publication["algorithm"] != "sha256(checksums.sha256 bytes)"
        or publication["publication_id"] != resolved.name
        or publication["root_sha256"] != hashlib.sha256(checksum_bytes).hexdigest()
    ):
        raise ModelContractError("materialized publication root identity or digest is invalid")
    return MappingProxyType(dict(expected))


def _verify_training_partition(spec: TrainSpec) -> _VerifiedTrainingPartition:
    checksums = _verify_publication_tree(spec.materialized_export_root)
    root = spec.materialized_export_root.expanduser().resolve()
    manifest_path = _require_regular(
        root / "materialized_manifest.csv",
        "materialized Anomalib manifest",
    )
    if checksums.get("materialized_manifest.csv") != spec.materialized_manifest_digest:
        raise ModelContractError("materialized manifest digest is not bound by the publication index")
    if sha256_file(manifest_path) != spec.materialized_manifest_digest:
        raise ModelContractError("materialized manifest SHA-256 differs from TrainSpec")
    provenance_path = _require_regular(
        root / "provenance" / "dataset_release.json",
        "materialized dataset provenance",
    )
    if sha256_file(provenance_path) != spec.dataset_manifest_digest:
        raise ModelContractError("materialized export dataset provenance differs from TrainSpec")
    provenance = _read_json(provenance_path, "materialized dataset provenance")
    if provenance.get("dataset_release_id") != spec.dataset_release_id:
        raise ModelContractError("materialized export dataset_release_id differs from TrainSpec")
    adapter_digests = provenance.get("adapter_manifest_sha256")
    adapter_manifest_path = _require_regular(
        root / "provenance" / "adapter_manifest.csv",
        "materialized anomaly source adapter manifest",
    )
    if (
        not isinstance(adapter_digests, Mapping)
        or set(adapter_digests) != {"yolo", "anomalib", "template"}
        or sha256_file(adapter_manifest_path) != adapter_digests["anomalib"]
    ):
        raise ModelContractError("materialized anomaly adapter provenance is inconsistent")
    try:
        with adapter_manifest_path.open(newline="", encoding="utf-8") as stream:
            adapter_reader = csv.DictReader(stream)
            if tuple(adapter_reader.fieldnames or ()) != ADAPTER_BASE_COLUMNS:
                raise ModelContractError("anomaly source adapter manifest header is invalid")
            adapter_rows = list(adapter_reader)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise ModelContractError(f"cannot parse anomaly source adapter manifest: {error}") from error
    if not adapter_rows or any(
        set(row) != set(ADAPTER_BASE_COLUMNS) or any(value is None for value in row.values())
        for row in adapter_rows
    ):
        raise ModelContractError("anomaly source adapter manifest rows are malformed")

    try:
        with manifest_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != MATERIALIZED_COLUMNS:
                raise ModelContractError(
                    "materialized Anomalib manifest columns differ from the frozen schema"
                )
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise ModelContractError(f"cannot parse materialized Anomalib manifest: {error}") from error
    if not rows:
        raise ModelContractError("materialized Anomalib manifest is empty")
    expected_rows = [
        {**row, "materialized_path": anomalib_materialized_path(row)}
        for row in adapter_rows
    ]
    if rows != expected_rows:
        raise ModelContractError(
            "materialized Anomalib manifest is not the complete canonical adapter export"
        )

    selected: list[dict[str, str]] = []
    seen_samples: set[str] = set()
    for row in rows:
        if set(row) != set(MATERIALIZED_COLUMNS) or any(value is None for value in row.values()):
            raise ModelContractError("materialized Anomalib manifest contains a malformed row")
        sample_id = require_text(row["sample_id"], "materialized.sample_id")
        if sample_id in seen_samples:
            raise ModelContractError(f"duplicate materialized sample_id: {sample_id}")
        seen_samples.add(sample_id)
        if row["split"] not in {"train", "calibration", "test"}:
            raise ModelContractError(f"invalid materialized split: {row['split']!r}")
        if row["label"] not in {"normal", "defect"}:
            raise ModelContractError(f"invalid materialized label: {row['label']!r}")
        if (row["label"] == "normal" and row["defect_type"]) or (
            row["label"] == "defect" and not row["defect_type"]
        ):
            raise ModelContractError("materialized label/defect_type semantics are invalid")
        relative = _safe_relative(row["materialized_path"], "materialized_path")
        split_directory = (
            "train_defect_reference"
            if row["split"] == "train" and row["label"] == "defect"
            else row["split"]
        )
        label_directory = (
            "good"
            if row["label"] == "normal"
            else "defect_"
            + hashlib.sha256(row["defect_type"].encode("utf-8")).hexdigest()[:12]
        )
        expected = PurePosixPath(
            row["hand"],
            row["view"],
            split_directory,
            label_directory,
            f"{sample_id}.png",
        )
        if relative != expected:
            raise ModelContractError(
                f"materialized path does not match manifest semantics: {relative.as_posix()}"
            )
        digest = require_sha256(row["canonical_crop_sha256"], "canonical_crop_sha256")
        if checksums.get(relative.as_posix()) != digest:
            raise ModelContractError(
                f"materialized crop is not bound to its canonical digest: {relative.as_posix()}"
            )
        crop = _require_regular(root.joinpath(*relative.parts), "materialized crop")
        if sha256_file(crop) != digest:
            raise ModelContractError(f"materialized crop hash mismatch: {relative.as_posix()}")
        if (
            row["hand"] == spec.slot.hand
            and row["view"] == spec.slot.view
            and row["split"] == "train"
            and row["label"] == "normal"
        ):
            if row["defect_type"]:
                raise ModelContractError("normal training row cannot have a defect_type")
            selected.append(row)

    if not selected:
        raise ModelContractError(f"no verified train/normal samples for slot {spec.slot.key}")
    expected_dir = root / spec.slot.hand / spec.slot.view / "train" / "good"
    if expected_dir.is_symlink() or not expected_dir.is_dir():
        raise ModelContractError(f"verified train/normal directory is missing: {expected_dir}")
    expected_names = {f"{row['sample_id']}.png" for row in selected}
    actual_names = {
        path.name
        for path in expected_dir.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if actual_names != expected_names or any(path.is_dir() or path.is_symlink() for path in expected_dir.iterdir()):
        raise ModelContractError(
            f"train/normal directory contains unmanifested or non-file content: {expected_dir}"
        )
    return _VerifiedTrainingPartition(
        root=root,
        normal_dir=expected_dir,
        manifest_sha256=spec.materialized_manifest_digest,
        sample_count=len(selected),
    )


def _require_exact_keys(value: Mapping[str, Any], allowed: set[str], field: str) -> dict[str, Any]:
    copied = dict(value)
    if set(copied) != allowed:
        raise ModelContractError(
            f"{field} keys invalid; missing={sorted(allowed - set(copied))}, "
            f"unknown={sorted(set(copied) - allowed)}"
        )
    return copied


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ModelContractError(f"{field} must be an integer >= {minimum}")
    return value


def _number(value: object, field: str, *, minimum: float, maximum: float | None = None) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or float(value) < minimum
        or (maximum is not None and float(value) > maximum)
    ):
        suffix = f" and <= {maximum}" if maximum is not None else ""
        raise ModelContractError(f"{field} must be finite and >= {minimum}{suffix}")
    return float(value)


def _logical_asset_id(value: object, field: str) -> str:
    asset_id = require_text(value, field)
    if asset_id in {".", ".."} or any(
        character in asset_id for character in ("/", "\\", "\x00", "\n", "\r")
    ):
        raise ModelContractError(f"{field} must be an opaque logical ID, not a path")
    return asset_id


def _image_size(value: object) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ModelContractError("parameters.image_size must be [height, width]")
    height = _integer(value[0], "parameters.image_size[0]", minimum=1)
    width = _integer(value[1], "parameters.image_size[1]", minimum=1)
    return height, width


def _directory_digest(root: Path, field: str) -> str:
    expanded = root.expanduser()
    if expanded.is_symlink() or not expanded.is_dir():
        raise ModelContractError(f"{field} must be a non-symlink directory: {expanded}")
    base = expanded.resolve()
    digest = hashlib.sha256()
    file_count = 0
    for path in sorted(base.rglob("*"), key=lambda item: item.relative_to(base).as_posix()):
        if path.is_symlink():
            raise ModelContractError(f"{field} contains a symlink: {path}")
        metadata = path.stat(follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ModelContractError(f"{field} contains a special filesystem node: {path}")
        relative = path.relative_to(base).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\x00")
        digest.update(bytes.fromhex(sha256_file(path)))
        digest.update(b"\n")
        file_count += 1
    if file_count == 0:
        raise ModelContractError(f"{field} contains no files: {base}")
    return digest.hexdigest()


def sha256_auxiliary_tree(root: Path) -> str:
    """Hash a local auxiliary tree using the backend's frozen provenance rule."""
    return _directory_digest(Path(root), "anomaly auxiliary tree")


def _resolve_parameters(spec: TrainSpec, sample_count: int) -> dict[str, Any]:
    raw = _require_exact_keys(
        spec.parameters,
        {"schema_version", "image_size", "model", "trainer", "auxiliary"},
        "parameters",
    )
    if _integer(raw["schema_version"], "parameters.schema_version", minimum=1) != BACKEND_SCHEMA_VERSION:
        raise ModelContractError(
            f"parameters.schema_version must be {BACKEND_SCHEMA_VERSION}"
        )
    image_size = _image_size(raw["image_size"])
    if spec.family is AnomalyFamily.ANOMALYDINO and (
        image_size[0] % 14 or image_size[1] % 14
    ):
        raise ModelContractError(
            "anomalydino image_size dimensions must be divisible by patch size 14; "
            "this prevents its implementation from center-cropping resized ROI pixels"
        )
    model = raw["model"]
    trainer = raw["trainer"]
    auxiliary = raw["auxiliary"]
    if not isinstance(model, Mapping) or not isinstance(trainer, Mapping) or not isinstance(auxiliary, Mapping):
        raise ModelContractError("parameters.model/trainer/auxiliary must be objects")

    family_model_keys = {
        AnomalyFamily.PATCHCORE: {
            "backbone", "layers", "pre_trained", "initialization",
            "coreset_sampling_ratio", "num_neighbors", "precision"
        },
        AnomalyFamily.EFFICIENTAD: {
            "teacher_out_channels", "model_size", "lr", "weight_decay", "padding", "pad_maps"
        },
        AnomalyFamily.ANOMALYDINO: {
            "num_neighbours", "encoder_name", "masking", "coreset_subsampling", "sampling_ratio", "precision"
        },
    }
    model_values = _require_exact_keys(model, family_model_keys[spec.family], "parameters.model")
    if spec.family is AnomalyFamily.PATCHCORE:
        require_text(model_values["backbone"], "parameters.model.backbone")
        layers = model_values["layers"]
        if isinstance(layers, (str, bytes)) or not isinstance(layers, Sequence) or not layers:
            raise ModelContractError("parameters.model.layers must be a non-empty array")
        parsed_layers = [require_text(layer, "parameters.model.layers[]") for layer in layers]
        if len(parsed_layers) != len(set(parsed_layers)):
            raise ModelContractError("parameters.model.layers must be unique")
        model_values["layers"] = parsed_layers
        if (
            model_values["pre_trained"] is not False
            or model_values["initialization"] != "explicit_state_dict"
        ):
            raise ModelContractError(
                "PatchCore requires pre_trained=false and initialization=explicit_state_dict"
            )
        _number(model_values["coreset_sampling_ratio"], "coreset_sampling_ratio", minimum=0.0, maximum=1.0)
        if float(model_values["coreset_sampling_ratio"]) == 0:
            raise ModelContractError("coreset_sampling_ratio must be > 0")
        _integer(model_values["num_neighbors"], "num_neighbors", minimum=1)
        if model_values["precision"] not in {"float32", "float16"}:
            raise ModelContractError("PatchCore precision must be float32 or float16")
    elif spec.family is AnomalyFamily.EFFICIENTAD:
        _integer(model_values["teacher_out_channels"], "teacher_out_channels", minimum=1)
        if model_values["model_size"] not in {"small", "medium"}:
            raise ModelContractError("EfficientAD model_size must be small or medium")
        if _number(model_values["lr"], "lr", minimum=0.0) == 0:
            raise ModelContractError("EfficientAD lr must be > 0")
        _number(model_values["weight_decay"], "weight_decay", minimum=0.0)
        if not isinstance(model_values["padding"], bool) or not isinstance(model_values["pad_maps"], bool):
            raise ModelContractError("EfficientAD padding and pad_maps must be booleans")
    else:
        _integer(model_values["num_neighbours"], "num_neighbours", minimum=1)
        require_text(model_values["encoder_name"], "encoder_name")
        if not isinstance(model_values["masking"], bool) or not isinstance(
            model_values["coreset_subsampling"], bool
        ):
            raise ModelContractError("AnomalyDINO masking/coreset_subsampling must be booleans")
        _number(model_values["sampling_ratio"], "sampling_ratio", minimum=0.0, maximum=1.0)
        if model_values["coreset_subsampling"] and float(model_values["sampling_ratio"]) == 0:
            raise ModelContractError("AnomalyDINO sampling_ratio must be > 0 when coreset is enabled")
        if model_values["precision"] not in {"float32", "float16"}:
            raise ModelContractError("AnomalyDINO precision must be float32 or float16")
    trainer_values = _require_exact_keys(
        trainer,
        {"max_epochs", "precision", "train_batch_size", "num_workers", "seed"},
        "parameters.trainer",
    )
    max_epochs = _integer(trainer_values["max_epochs"], "parameters.trainer.max_epochs", minimum=1)
    train_batch_size = _integer(
        trainer_values["train_batch_size"],
        "parameters.trainer.train_batch_size",
        minimum=1,
    )
    num_workers = _integer(trainer_values["num_workers"], "parameters.trainer.num_workers")
    seed = _integer(trainer_values["seed"], "parameters.trainer.seed")
    precision = trainer_values["precision"]
    if precision not in {"32-true", "16-mixed", "bf16-mixed"}:
        raise ModelContractError("trainer precision must be 32-true, 16-mixed, or bf16-mixed")
    if spec.family in {AnomalyFamily.PATCHCORE, AnomalyFamily.ANOMALYDINO} and max_epochs != 1:
        raise ModelContractError(f"{spec.family.value} requires max_epochs=1")
    if spec.family is AnomalyFamily.EFFICIENTAD and train_batch_size != 1:
        raise ModelContractError("efficientad requires train_batch_size=1")

    auxiliary_values: dict[str, Any] = {}
    if spec.family is AnomalyFamily.PATCHCORE:
        auxiliary_values = _require_exact_keys(
            auxiliary,
            {
                "backbone_weights_asset_id", "backbone_weights_path",
                "backbone_weights_sha256", "state_dict_scope",
            },
            "parameters.auxiliary",
        )
        weights_path = _require_regular(
            Path(
                require_text(
                    auxiliary_values["backbone_weights_path"],
                    "auxiliary.backbone_weights_path",
                )
            ).expanduser(),
            "PatchCore backbone weights",
        ).resolve()
        expected_weights = require_sha256(
            auxiliary_values["backbone_weights_sha256"],
            "auxiliary.backbone_weights_sha256",
        )
        if sha256_file(weights_path) != expected_weights:
            raise ModelContractError("PatchCore backbone weights digest mismatch")
        auxiliary_values["backbone_weights_asset_id"] = _logical_asset_id(
            auxiliary_values["backbone_weights_asset_id"],
            "auxiliary.backbone_weights_asset_id",
        )
        if auxiliary_values["state_dict_scope"] != "anomalib_timm_feature_extractor":
            raise ModelContractError(
                "PatchCore state_dict_scope must be anomalib_timm_feature_extractor"
            )
        auxiliary_values["backbone_weights_path"] = str(weights_path)
        auxiliary_values["backbone_weights_sha256"] = expected_weights
    elif spec.family is AnomalyFamily.EFFICIENTAD:
        auxiliary_values = _require_exact_keys(
            auxiliary,
            {"imagenet_dir", "imagenet_tree_sha256", "teacher_weights_sha256"},
            "parameters.auxiliary",
        )
        imagenet_dir = Path(require_text(auxiliary_values["imagenet_dir"], "auxiliary.imagenet_dir"))
        expected_tree = require_sha256(
            auxiliary_values["imagenet_tree_sha256"],
            "auxiliary.imagenet_tree_sha256",
        )
        actual_tree = _directory_digest(imagenet_dir, "EfficientAD ImageNette auxiliary dataset")
        if actual_tree != expected_tree:
            raise ModelContractError("EfficientAD ImageNette auxiliary tree digest mismatch")
        auxiliary_values["imagenet_dir"] = str(imagenet_dir.expanduser().resolve())
        auxiliary_values["imagenet_tree_sha256"] = expected_tree
        auxiliary_values["teacher_weights_sha256"] = require_sha256(
            auxiliary_values["teacher_weights_sha256"],
            "auxiliary.teacher_weights_sha256",
        )
    elif spec.family is AnomalyFamily.ANOMALYDINO:
        auxiliary_values = _require_exact_keys(
            auxiliary,
            {"encoder_weights_sha256"},
            "parameters.auxiliary",
        )
        auxiliary_values["encoder_weights_sha256"] = require_sha256(
            auxiliary_values["encoder_weights_sha256"],
            "auxiliary.encoder_weights_sha256",
        )

    return {
        "schema": BACKEND_SCHEMA,
        "schema_version": BACKEND_SCHEMA_VERSION,
        "source_parameters_sha256": hashlib.sha256(
            canonical_json_bytes(dict(spec.parameters))
        ).hexdigest(),
        "family": spec.family.value,
        "slot": {"hand": spec.slot.hand, "view": spec.slot.view},
        "image_size": list(image_size),
        "normalization": "none" if spec.family is AnomalyFamily.EFFICIENTAD else "imagenet",
        "model": model_values,
        "trainer": {
            "max_epochs": max_epochs,
            "precision": precision,
            "train_batch_size": train_batch_size,
            "num_workers": num_workers,
            "seed": seed,
            "deterministic": True,
        },
        "dataset": {
            "materialized_manifest_sha256": spec.materialized_manifest_digest,
            "selected_train_normal_count": sample_count,
        },
        "auxiliary": auxiliary_values,
    }


def _cuda_index(device: DeviceSpec) -> int:
    if device.accelerator not in {"gpu", "cuda"}:
        raise BackendUnavailableError("Anomalib backend requires accelerator='gpu' or 'cuda'")
    if not device.device.isdecimal():
        raise BackendUnavailableError("Anomalib backend requires one numeric CUDA device index")
    return int(device.device)


def _require_cuda(device: DeviceSpec) -> tuple[Any, int]:
    if os.name != "posix" or not os.uname().sysname.lower().startswith("linux"):
        raise BackendUnavailableError("Anomalib backend is supported only on Linux with NVIDIA CUDA")
    try:
        import torch
    except (ImportError, OSError) as error:
        raise BackendUnavailableError(f"cannot import the Linux torch runtime: {error}") from error
    index = _cuda_index(device)
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise BackendUnavailableError(
            f"requested CUDA device {index} is unavailable; visible device count={torch.cuda.device_count()}"
        )
    try:
        torch.cuda.set_device(index)
        torch.cuda.synchronize(index)
    except Exception as error:
        raise BackendUnavailableError(f"cannot initialize CUDA device {index}: {error}") from error
    return torch, index


def _preprocessor(family: AnomalyFamily, image_size: tuple[int, int]) -> Any:
    from anomalib.pre_processing import PreProcessor
    from torchvision.transforms.v2 import Compose, Normalize, Resize

    transforms: list[Any] = [Resize(image_size, antialias=True)]
    if family is not AnomalyFamily.EFFICIENTAD:
        transforms.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    # Keep this list intentionally limited to Resize/Normalize. CenterCrop and
    # other spatial transforms are forbidden by the canonical ROI contract.
    return PreProcessor(transform=Compose(transforms))


def _verify_patchcore_backbone_asset(parameters: Mapping[str, Any]) -> Path:
    """Rehash the explicit training-only PatchCore feature-extractor state_dict."""
    auxiliary = parameters.get("auxiliary")
    if not isinstance(auxiliary, Mapping):
        raise ModelContractError("PatchCore auxiliary parameters must be an object")
    path = _require_regular(
        Path(
            require_text(
                auxiliary.get("backbone_weights_path"),
                "auxiliary.backbone_weights_path",
            )
        ),
        "PatchCore backbone weights",
    )
    expected = require_sha256(
        auxiliary.get("backbone_weights_sha256"),
        "auxiliary.backbone_weights_sha256",
    )
    if sha256_file(path) != expected:
        raise ModelContractError("PatchCore backbone weights changed during training")
    return path


def _load_patchcore_backbone_weights(model: Any, parameters: Mapping[str, Any]) -> None:
    """Strictly load a raw feature-extractor state_dict without network fallback."""
    path = _verify_patchcore_backbone_asset(parameters)
    try:
        import torch

        content = path.read_bytes()
        expected = parameters["auxiliary"]["backbone_weights_sha256"]
        if hashlib.sha256(content).hexdigest() != expected:
            raise ModelContractError(
                "PatchCore backbone weights changed between verification and load"
            )
        state_dict = torch.load(
            io.BytesIO(content),
            map_location="cpu",
            weights_only=True,
        )
        if (
            not isinstance(state_dict, Mapping)
            or not state_dict
            or any(not isinstance(key, str) for key in state_dict)
        ):
            raise ModelContractError(
                "PatchCore backbone file must contain a non-empty raw state_dict"
            )
        if parameters["auxiliary"]["state_dict_scope"] != "anomalib_timm_feature_extractor":
            raise ModelContractError("PatchCore state_dict scope changed before load")
        feature_extractor = model.model.feature_extractor
        feature_extractor.load_state_dict(dict(state_dict), strict=True)
    except ModelContractError:
        raise
    except Exception as error:
        raise ModelContractError(
            "PatchCore backbone state_dict does not exactly match the selected feature extractor"
        ) from error


def _build_model(
    family: AnomalyFamily,
    parameters: Mapping[str, Any],
    *,
    for_training: bool,
) -> Any:
    from anomalib.models import AnomalyDINO, EfficientAd, Patchcore

    image_size = _image_size(parameters["image_size"])
    model_args = dict(parameters["model"])
    common = {
        "pre_processor": _preprocessor(family, image_size),
        "post_processor": False,
        "evaluator": False,
        "visualizer": False,
    }
    if family is AnomalyFamily.PATCHCORE:
        class ZS32Patchcore(Patchcore):
            @property
            def trainer_arguments(self) -> dict[str, Any]:
                arguments = dict(super().trainer_arguments)
                arguments.pop("devices", None)
                return arguments

        # Network/cache-based pretrained resolution is forbidden.  Training
        # loads one explicit, content-addressed feature-extractor state_dict.
        requested_backbone = model_args["backbone"]
        requested_layers = tuple(model_args["layers"])
        model_args.pop("initialization")
        model_args["pre_trained"] = False
        model = ZS32Patchcore(**model_args, **common)
        feature_extractor = model.model.feature_extractor
        if (
            feature_extractor.backbone != requested_backbone
            or tuple(feature_extractor.layers) != requested_layers
        ):
            raise ModelContractError(
                "Anomalib silently changed the requested PatchCore backbone/layers"
            )
        if for_training:
            _load_patchcore_backbone_weights(model, parameters)
        return model
    if family is AnomalyFamily.ANOMALYDINO:
        class ZS32AnomalyDINO(AnomalyDINO):
            @property
            def trainer_arguments(self) -> dict[str, Any]:
                arguments = dict(super().trainer_arguments)
                arguments.pop("devices", None)
                return arguments

        if for_training:
            return ZS32AnomalyDINO(**model_args, **common)
        from anomalib.models.image.anomaly_dino import torch_model as anomaly_dino_torch_model
        from anomalib.models.components.dinov2 import DinoV2Loader

        class ArchitectureOnlyDinoV2Loader:
            """Construct DINOv2 architecture without reading/downloading base weights."""

            @classmethod
            def from_name(cls, model_name: str) -> Any:
                del cls
                loader = object.__new__(DinoV2Loader)
                loader.vit_factory = None
                model_type, architecture, patch_size = loader._parse_name(model_name)
                return loader.create_model(model_type, architecture, patch_size)

        with _ANOMALYDINO_BUILD_LOCK:
            original = anomaly_dino_torch_model.DinoV2Loader
            anomaly_dino_torch_model.DinoV2Loader = ArchitectureOnlyDinoV2Loader
            try:
                return ZS32AnomalyDINO(**model_args, **common)
            finally:
                anomaly_dino_torch_model.DinoV2Loader = original
    if family is AnomalyFamily.EFFICIENTAD:
        from types import MethodType

        from torch.utils.data import DataLoader
        from torchvision.datasets import ImageFolder
        from torchvision.transforms.v2 import Compose, Resize, ToTensor

        imagenet_dir = parameters.get("auxiliary", {}).get("imagenet_dir", ".") if for_training else "."
        model = EfficientAd(imagenet_dir=imagenet_dir, **model_args, **common)
        if for_training:
            def prepare_resize_only(self: Any, requested_size: tuple[int, int]) -> None:
                """Prepare explicit auxiliary data without spatial recropping."""
                self.data_transforms_imagenet = Compose(
                    [Resize(tuple(requested_size), antialias=True), ToTensor()]
                )
                if not self.imagenet_dir.is_dir():
                    raise FileNotFoundError(
                        f"verified EfficientAD ImageNette directory disappeared: {self.imagenet_dir}"
                    )
                dataset = ImageFolder(self.imagenet_dir, transform=self.data_transforms_imagenet)
                self.imagenet_loader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    pin_memory=True,
                )
                self.imagenet_iterator = iter(self.imagenet_loader)

            model.prepare_imagenette_data = MethodType(prepare_resize_only, model)
        return model
    raise ModelContractError(f"unsupported anomaly family: {family.value}")


def _verify_efficientad_teacher(parameters: Mapping[str, Any]) -> None:
    if parameters["family"] != AnomalyFamily.EFFICIENTAD.value:
        return
    from anomalib.models.image.efficient_ad.lightning_model import get_pretrained_weights_dir

    size = str(parameters["model"]["model_size"])
    teacher = (
        get_pretrained_weights_dir()
        / "efficientad_pretrained_weights"
        / f"pretrained_teacher_{size}.pth"
    )
    _require_regular(teacher, "EfficientAD pretrained teacher")
    expected = parameters["auxiliary"]["teacher_weights_sha256"]
    if sha256_file(teacher) != expected:
        raise ModelContractError("EfficientAD pretrained teacher SHA-256 mismatch")


def _verify_anomalydino_encoder(parameters: Mapping[str, Any]) -> None:
    if parameters["family"] != AnomalyFamily.ANOMALYDINO.value:
        return
    from anomalib.models.components.dinov2 import DinoV2Loader

    loader = DinoV2Loader()
    model_type, architecture, patch_size = loader._parse_name(parameters["model"]["encoder_name"])
    weights = loader._get_weight_path(model_type, architecture, patch_size)
    _require_regular(weights, "AnomalyDINO pretrained encoder")
    expected = parameters["auxiliary"]["encoder_weights_sha256"]
    if sha256_file(weights) != expected:
        raise ModelContractError("AnomalyDINO pretrained encoder SHA-256 mismatch")


def _verify_finite_model(torch: Any, model: Any) -> None:
    for name, value in model.state_dict().items():
        if hasattr(value, "is_floating_point") and value.is_floating_point():
            if not bool(torch.isfinite(value).all().item()):
                raise AnomalibBackendError(
                    f"model state contains NaN/Infinity after training: {name}"
                )


def _published_parameters(
    family: AnomalyFamily,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove training-host paths while retaining exact auxiliary asset identity."""
    payload = json.loads(json.dumps(parameters, allow_nan=False))
    if family is AnomalyFamily.PATCHCORE:
        auxiliary = payload["auxiliary"]
        auxiliary.pop("backbone_weights_path", None)
    elif family is AnomalyFamily.EFFICIENTAD:
        # ImageNette is a training-only auxiliary dataset.  Runtime restore uses
        # the trained checkpoint and the two content digests below; retaining
        # the Linux training path would make an otherwise self-contained
        # candidate host-specific and disclose a path that cannot be verified
        # or consumed on the deployment station.
        auxiliary = payload["auxiliary"]
        auxiliary.pop("imagenet_dir", None)
    return payload


def _validate_execution_receipt_binding(
    spec_or_artifact: TrainSpec | AnomalyModelArtifact,
    parameters: Mapping[str, Any],
) -> None:
    """Bind the raw authoring parameters and explicit assets to the receipt."""
    from zs32_inspection.runtime.execution_receipt import ExecutionReceipt

    receipt = ExecutionReceipt.from_mapping(spec_or_artifact.execution_receipt)
    source_digest = require_sha256(
        parameters.get("source_parameters_sha256"),
        "source_parameters_sha256",
    )
    if receipt.parameters_sha256 != source_digest:
        raise ModelContractError(
            "anomaly execution receipt parameters_sha256 differs from source parameters"
        )
    if spec_or_artifact.family is AnomalyFamily.PATCHCORE:
        expected_backbone = require_sha256(
            parameters["auxiliary"].get("backbone_weights_sha256"),
            "backbone_weights_sha256",
        )
        if receipt.input_sha256_by_role.get("patchcore_backbone") != expected_backbone:
            raise ModelContractError(
                "PatchCore execution receipt does not bind the explicit backbone weights"
            )


def _artifact_payload(
    spec: TrainSpec,
    *,
    checkpoint_digest: str,
    framework_version: str,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "zs32.anomaly_model_artifact",
        "schema_version": 1,
        "family": spec.family.value,
        "hand": spec.slot.hand,
        "view": spec.slot.view,
        "checkpoint": {
            "role": "checkpoint",
            "relative_path": f"checkpoint/{checkpoint_digest}.ckpt",
            "sha256": checkpoint_digest,
            "media_type": "application/x-pytorch-lightning-checkpoint",
        },
        "model_digest": checkpoint_digest,
        "dataset_release_id": spec.dataset_release_id,
        "dataset_manifest_digest": spec.dataset_manifest_digest,
        "train_split_id": spec.train_split_id,
        "recipe_digest": spec.recipe_digest,
        "roi_version": spec.roi_version,
        "roi_digest": spec.roi_digest,
        "framework_version": framework_version,
        "training_parameters": _published_parameters(spec.family, parameters),
        "execution_receipt": dict(spec.execution_receipt),
        "promotion_status": "candidate_only",
    }


def _validate_candidate(
    staging: Path,
    metadata: Mapping[str, Any],
    metadata_relative: str,
) -> None:
    metadata_path = _require_regular(
        staging.joinpath(*PurePosixPath(metadata_relative).parts),
        "candidate artifact metadata",
    )
    if metadata_path.read_bytes() != canonical_json_bytes(metadata):
        raise PublicationError("candidate artifact metadata changed during publication")
    checkpoint = metadata["checkpoint"]
    checkpoint_path = staging.joinpath(*PurePosixPath(checkpoint["relative_path"]).parts)
    if sha256_file(checkpoint_path) != checkpoint["sha256"]:
        raise PublicationError("candidate checkpoint hash differs from artifact metadata")


class AnomalibTrainerBackend:
    """Unified callable that trains one explicitly bound family/hand/view slot."""

    def __call__(self, spec: TrainSpec) -> AnomalyModelArtifact:
        if not isinstance(spec, TrainSpec):
            raise ModelContractError("Anomalib trainer requires a TrainSpec")
        torch, cuda_index = _require_cuda(spec.device)
        partition = _verify_training_partition(spec)
        parameters = _resolve_parameters(spec, partition.sample_count)
        _validate_execution_receipt_binding(spec, parameters)
        _verify_efficientad_teacher(parameters)
        _verify_anomalydino_encoder(parameters)

        output_root = spec.output_dir.expanduser()
        if output_root.is_symlink():
            raise ModelContractError(f"candidate output root must not be a symlink: {output_root}")
        output_root.mkdir(parents=True, exist_ok=True)
        workspace = Path(tempfile.mkdtemp(prefix=".zs32-anomalib-training-", dir=output_root))
        try:
            from anomalib import __version__ as anomalib_version
            from anomalib.data import Folder
            from anomalib.engine import Engine
            from lightning import seed_everything

            seed_everything(parameters["trainer"]["seed"], workers=True)
            model = _build_model(spec.family, parameters, for_training=True)
            datamodule = Folder(
                name=f"zs32_{spec.slot.hand}_{spec.slot.view}",
                root=None,
                normal_dir=partition.normal_dir,
                abnormal_dir=None,
                normal_test_dir=None,
                mask_dir=None,
                train_batch_size=parameters["trainer"]["train_batch_size"],
                eval_batch_size=1,
                num_workers=parameters["trainer"]["num_workers"],
                test_split_mode="none",
                val_split_mode="none",
                seed=parameters["trainer"]["seed"],
            )
            datamodule.setup("fit")
            if len(datamodule.train_data) != partition.sample_count:
                raise ModelContractError(
                    "Anomalib datamodule train sample count differs from verified train/normal manifest"
                )
            engine = Engine(
                default_root_dir=workspace / "engine",
                accelerator="gpu",
                devices=[cuda_index],
                precision=parameters["trainer"]["precision"],
                max_epochs=parameters["trainer"]["max_epochs"],
                deterministic=True,
                logger=False,
                enable_checkpointing=True,
                enable_model_summary=False,
            )
            engine.fit(model=model, datamodule=datamodule)
            torch.cuda.synchronize(cuda_index)
            _verify_finite_model(torch, model)
            for name, value in engine.trainer.callback_metrics.items():
                if hasattr(value, "is_floating_point") and value.is_floating_point():
                    if not bool(torch.isfinite(value).all().item()):
                        raise AnomalibBackendError(f"training metric contains NaN/Infinity: {name}")
            after_training = _verify_training_partition(spec)
            if after_training != partition:
                raise ModelContractError("verified train/normal partition changed during training")
            if spec.family is AnomalyFamily.PATCHCORE:
                _verify_patchcore_backbone_asset(parameters)
            elif spec.family is AnomalyFamily.EFFICIENTAD:
                expected_auxiliary = parameters["auxiliary"]["imagenet_tree_sha256"]
                actual_auxiliary = _directory_digest(
                    Path(parameters["auxiliary"]["imagenet_dir"]),
                    "EfficientAD ImageNette auxiliary dataset",
                )
                if actual_auxiliary != expected_auxiliary:
                    raise ModelContractError("EfficientAD auxiliary dataset changed during training")
                _verify_efficientad_teacher(parameters)
            _verify_anomalydino_encoder(parameters)
            checkpoint_path = workspace / "model.ckpt"
            engine.trainer.save_checkpoint(checkpoint_path, weights_only=False)
            _require_regular(checkpoint_path, "trained Anomalib checkpoint")
            checkpoint_digest = sha256_file(checkpoint_path)
            framework_version = f"anomalib-{anomalib_version}"
            from zs32_inspection.runtime.execution_receipt import (
                ExecutionReceipt,
                verify_execution_receipt_live,
            )

            verify_execution_receipt_live(
                ExecutionReceipt.from_mapping(spec.execution_receipt),
                device=spec.device.device,
            )
            metadata = _artifact_payload(
                spec,
                checkpoint_digest=checkpoint_digest,
                framework_version=framework_version,
                parameters=parameters,
            )
            metadata_digest = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest()
            metadata_relative = f"metadata/{metadata_digest}.json"
            publication_id = (
                f"anomaly-{spec.family.value}-{spec.slot.hand}-"
                f"{metadata_digest[:24]}"
            )
            checkpoint_relative = metadata["checkpoint"]["relative_path"]
            with AtomicDirectoryPublisher(output_root, publication_id) as publisher:
                publisher.copy_file(
                    checkpoint_path,
                    checkpoint_relative,
                    expected_sha256=checkpoint_digest,
                )
                publisher.write_json(metadata_relative, metadata)
                published = publisher.finalize(
                    validator=lambda staging: _validate_candidate(
                        staging,
                        metadata,
                        metadata_relative,
                    ),
                    required_paths=frozenset({metadata_relative, checkpoint_relative}),
                )
            artifact = AnomalyModelArtifact(
                family=spec.family,
                slot=spec.slot,
                checkpoint=AssetFile(
                    role="checkpoint",
                    path=published.joinpath(*PurePosixPath(checkpoint_relative).parts),
                    sha256=checkpoint_digest,
                    media_type="application/x-pytorch-lightning-checkpoint",
                    logical_path=checkpoint_relative,
                ),
                model_digest=checkpoint_digest,
                dataset_release_id=spec.dataset_release_id,
                dataset_manifest_digest=spec.dataset_manifest_digest,
                train_split_id=spec.train_split_id,
                recipe_digest=spec.recipe_digest,
                roi_version=spec.roi_version,
                roi_digest=spec.roi_digest,
                framework_version=framework_version,
                training_parameters=_published_parameters(spec.family, parameters),
                execution_receipt=spec.execution_receipt,
            )
            artifact.verify()
            return artifact
        except (ModelContractError, BackendUnavailableError, PublicationError):
            raise
        except Exception as error:
            raise AnomalibBackendError(
                f"{spec.family.value} training failed for {spec.slot.key}: {error}"
            ) from error
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


def _validate_runtime_parameters(
    artifact: AnomalyModelArtifact,
) -> tuple[dict[str, Any], tuple[int, int]]:
    value = dict(artifact.training_parameters)
    expected = {
        "schema", "schema_version", "family", "slot", "image_size", "normalization",
        "model", "trainer", "dataset", "auxiliary", "source_parameters_sha256",
    }
    if set(value) != expected:
        raise ModelContractError(f"anomaly training metadata keys are invalid for {artifact.slot.key}")
    if (
        value["schema"] != BACKEND_SCHEMA
        or value["schema_version"] != BACKEND_SCHEMA_VERSION
        or value["family"] != artifact.family.value
        or value["slot"] != {"hand": artifact.slot.hand, "view": artifact.slot.view}
    ):
        raise ModelContractError(f"anomaly checkpoint metadata identity mismatch: {artifact.slot.key}")
    require_sha256(value["source_parameters_sha256"], "source_parameters_sha256")
    _validate_execution_receipt_binding(artifact, value)
    dataset = value["dataset"]
    if not isinstance(dataset, Mapping):
        raise ModelContractError("anomaly training dataset metadata must be an object")
    if set(dataset) != {"materialized_manifest_sha256", "selected_train_normal_count"}:
        raise ModelContractError("anomaly training dataset metadata keys are invalid")
    require_sha256(dataset["materialized_manifest_sha256"], "materialized_manifest_sha256")
    _integer(dataset["selected_train_normal_count"], "selected_train_normal_count", minimum=1)
    image_size = _image_size(value["image_size"])
    expected_normalization = "none" if artifact.family is AnomalyFamily.EFFICIENTAD else "imagenet"
    if value["normalization"] != expected_normalization:
        raise ModelContractError(f"unexpected normalization contract for {artifact.slot.key}")
    if not isinstance(value["model"], Mapping) or not isinstance(value["auxiliary"], Mapping):
        raise ModelContractError("anomaly model/auxiliary metadata must be objects")
    model_keys = {
        AnomalyFamily.PATCHCORE: {
            "backbone", "layers", "pre_trained", "initialization",
            "coreset_sampling_ratio", "num_neighbors", "precision"
        },
        AnomalyFamily.EFFICIENTAD: {
            "teacher_out_channels", "model_size", "lr", "weight_decay", "padding", "pad_maps"
        },
        AnomalyFamily.ANOMALYDINO: {
            "num_neighbours", "encoder_name", "masking", "coreset_subsampling", "sampling_ratio", "precision"
        },
    }[artifact.family]
    if set(value["model"]) != model_keys:
        raise ModelContractError(f"anomaly model parameter keys are invalid: {artifact.slot.key}")
    if artifact.family is AnomalyFamily.PATCHCORE and (
        value["model"]["pre_trained"] is not False
        or value["model"]["initialization"] != "explicit_state_dict"
    ):
        raise ModelContractError("PatchCore metadata initialization contract is invalid")
    trainer = value["trainer"]
    if not isinstance(trainer, Mapping) or set(trainer) != {
        "max_epochs", "precision", "train_batch_size", "num_workers", "seed", "deterministic"
    }:
        raise ModelContractError("anomaly trainer metadata keys are invalid")
    if trainer["deterministic"] is not True:
        raise ModelContractError("anomaly training metadata must assert deterministic=True")
    if artifact.family is AnomalyFamily.EFFICIENTAD:
        if set(value["auxiliary"]) != {
            "imagenet_tree_sha256", "teacher_weights_sha256"
        }:
            raise ModelContractError("EfficientAD auxiliary metadata keys are invalid")
        require_sha256(value["auxiliary"]["imagenet_tree_sha256"], "imagenet_tree_sha256")
        require_sha256(value["auxiliary"]["teacher_weights_sha256"], "teacher_weights_sha256")
    elif artifact.family is AnomalyFamily.ANOMALYDINO:
        if set(value["auxiliary"]) != {"encoder_weights_sha256"}:
            raise ModelContractError("AnomalyDINO auxiliary metadata keys are invalid")
        require_sha256(value["auxiliary"]["encoder_weights_sha256"], "encoder_weights_sha256")
        if image_size[0] % 14 or image_size[1] % 14:
            raise ModelContractError("AnomalyDINO runtime image size would trigger an internal center crop")
    else:
        if set(value["auxiliary"]) != {
            "backbone_weights_asset_id", "backbone_weights_sha256", "state_dict_scope"
        }:
            raise ModelContractError("PatchCore auxiliary metadata keys are invalid")
        _logical_asset_id(
            value["auxiliary"]["backbone_weights_asset_id"],
            "backbone_weights_asset_id",
        )
        require_sha256(
            value["auxiliary"]["backbone_weights_sha256"],
            "backbone_weights_sha256",
        )
        if value["auxiliary"]["state_dict_scope"] != "anomalib_timm_feature_extractor":
            raise ModelContractError("PatchCore state_dict scope is invalid")
    return value, image_size


def validate_anomaly_artifact_metadata(artifact: AnomalyModelArtifact) -> None:
    """Validate deployable metadata without importing Anomalib, Torch, or CUDA."""
    if not isinstance(artifact, AnomalyModelArtifact):
        raise TypeError("artifact must be an AnomalyModelArtifact")
    _validate_runtime_parameters(artifact)


def _render_anomaly_visuals(
    anomaly_map: Any,
    sample: ModelInput,
    *,
    expected_size: tuple[int, int],
    inspection_id: str,
    scratch_root: Path,
) -> tuple[Path, str, Path, str]:
    """Render deterministic heatmap and crop overlay without changing scores."""
    if anomaly_map is None or not hasattr(anomaly_map, "detach"):
        raise AnomalibBackendError("Anomalib prediction has no tensor anomaly_map")
    tensor = anomaly_map.detach().float().cpu()
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 2 or tuple(tensor.shape) != tuple(expected_size):
        raise AnomalibBackendError(
            "Anomalib anomaly_map shape differs from the frozen model input size: "
            f"expected={expected_size}, actual={tuple(tensor.shape)}"
        )
    if tensor.numel() == 0 or not bool(tensor.isfinite().all().item()):
        raise AnomalibBackendError("Anomalib anomaly_map contains NaN/Infinity or is empty")
    try:
        import cv2
        import numpy as np

        values = tensor.numpy()
        minimum = float(values.min())
        maximum = float(values.max())
        if not math.isfinite(minimum) or not math.isfinite(maximum):
            raise AnomalibBackendError("Anomalib anomaly_map extrema are not finite")
        if maximum == minimum:
            normalized = np.zeros(values.shape, dtype=np.uint8)
        else:
            normalized = np.rint(
                (values - minimum) * (255.0 / (maximum - minimum))
            ).clip(0, 255).astype(np.uint8)
        image = cv2.imread(str(sample.crop_path), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (
            sample.sample.crop_height,
            sample.sample.crop_width,
        ):
            raise AnomalibBackendError("canonical crop cannot be decoded for anomaly overlay")
        resized = cv2.resize(
            normalized,
            (sample.sample.crop_width, sample.sample.crop_height),
            interpolation=cv2.INTER_LINEAR,
        )
        heatmap = cv2.applyColorMap(resized, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.55, heatmap, 0.45, 0.0)
        heatmap_ok, heatmap_encoded = cv2.imencode(
            ".png",
            heatmap,
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        )
        overlay_ok, overlay_encoded = cv2.imencode(
            ".png",
            overlay,
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        )
        if not heatmap_ok or not overlay_ok:
            raise AnomalibBackendError("OpenCV failed to encode anomaly visual evidence")
        heatmap_content = bytes(heatmap_encoded)
        overlay_content = bytes(overlay_encoded)
    except AnomalibBackendError:
        raise
    except Exception as error:
        raise AnomalibBackendError(f"cannot render anomaly overlay: {error}") from error
    inspection_token = hashlib.sha256(inspection_id.encode("utf-8")).hexdigest()[:24]
    slot_token = hashlib.sha256(sample.slot.key.encode("utf-8")).hexdigest()[:24]
    outputs = (
        (scratch_root / "heatmaps" / inspection_token / f"{slot_token}.png", heatmap_content),
        (scratch_root / "overlays" / inspection_token / f"{slot_token}.png", overlay_content),
    )
    for output, content in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(output, flags, 0o600)
        except FileExistsError as error:
            raise AnomalibBackendError(
                f"anomaly visual already exists for inspection/slot {sample.slot.key}"
            ) from error
        try:
            written = 0
            while written < len(content):
                count = os.write(descriptor, content[written:])
                if count <= 0:
                    raise OSError("anomaly visual write made no progress")
                written += count
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    heatmap_path, overlay_path = (item[0] for item in outputs)
    return (
        heatmap_path,
        hashlib.sha256(heatmap_content).hexdigest(),
        overlay_path,
        hashlib.sha256(overlay_content).hexdigest(),
    )


class AnomalibBatchPredictor(AnomalyPredictor):
    """Loaded, slot-indexed predictor emitting continuous higher-is-riskier scores."""

    def __init__(
        self,
        loaded: Mapping[ModelSlot, _LoadedSlot],
        torch: Any,
        cuda_index: int,
        scratch_root: Path,
    ) -> None:
        if not loaded:
            raise ModelContractError("Anomalib predictor requires at least one loaded slot")
        import weakref

        self._loaded = MappingProxyType(dict(loaded))
        self._torch = torch
        self._cuda_index = cuda_index
        self._scratch_root = scratch_root
        self._scratch_cleanup = weakref.finalize(self, shutil.rmtree, scratch_root, True)

    def predict_batch(
        self,
        samples: Sequence[ModelInput],
        *,
        inspection_id: str,
    ) -> Sequence[RawModelScore]:
        inspection_id = require_text(inspection_id, "inspection_id")
        ordered = tuple(samples)
        if not ordered:
            raise ModelContractError("Anomalib predictor received an empty batch")
        slots = [sample.slot for sample in ordered]
        if len(slots) != len(set(slots)):
            raise ModelContractError("Anomalib predictor accepts exactly one sample per hand/view slot")
        capture_identity = {
            (sample.sample.part.part_instance_id, sample.sample.capture_set_id)
            for sample in ordered
        }
        if len(capture_identity) != 1:
            raise ModelContractError("Anomalib predictor batch mixes part or capture-set identity")

        scores: list[RawModelScore] = []
        for sample in ordered:
            loaded = self._loaded.get(sample.slot)
            if loaded is None:
                raise ModelContractError(f"no loaded anomaly checkpoint for slot {sample.slot.key}")
            artifact = loaded.artifact
            sample.verify_crop()
            if sample.roi_digest != artifact.roi_digest:
                raise ModelContractError(f"ROI digest mismatch for runtime slot {sample.slot.key}")
            if sample.sample.roi_config_id != artifact.roi_version:
                raise ModelContractError(f"ROI version mismatch for runtime slot {sample.slot.key}")
            try:
                from anomalib.data import PredictDataset

                dataset = PredictDataset(path=sample.crop_path)
                predictions = loaded.engine.predict(
                    model=loaded.model,
                    dataset=dataset,
                    return_predictions=True,
                )
                self._torch.cuda.synchronize(self._cuda_index)
            except Exception as error:
                raise AnomalibBackendError(
                    f"Anomalib inference failed for {sample.slot.key}: {error}"
                ) from error
            if predictions is None:
                raise AnomalibBackendError(f"Anomalib returned no predictions for {sample.slot.key}")
            items = [item for batch in predictions for item in batch]
            if len(items) != 1:
                raise AnomalibBackendError(
                    f"Anomalib returned {len(items)} predictions for one input in {sample.slot.key}"
                )
            item = items[0]
            if item.image_path is None or Path(item.image_path).resolve() != sample.crop_path.resolve():
                raise AnomalibBackendError(f"Anomalib prediction path identity mismatch: {sample.slot.key}")
            if item.pred_score is None or getattr(item.pred_score, "numel", lambda: 0)() != 1:
                raise AnomalibBackendError(f"Anomalib returned no scalar pred_score: {sample.slot.key}")
            score = float(item.pred_score.detach().float().cpu().item())
            if not math.isfinite(score):
                raise AnomalibBackendError(f"Anomalib returned NaN/Infinity: {sample.slot.key}")
            heatmap_path, heatmap_sha256, overlay_path, overlay_sha256 = (
                _render_anomaly_visuals(
                    item.anomaly_map,
                    sample,
                    expected_size=loaded.image_size,
                    inspection_id=inspection_id,
                    scratch_root=self._scratch_root,
                )
            )
            scores.append(
                RawModelScore(
                    inspection_id=inspection_id,
                    part_instance_id=sample.sample.part.part_instance_id,
                    capture_set_id=sample.sample.capture_set_id,
                    hand=sample.sample.part.hand.value,
                    view=sample.sample.view_id,
                    branch="anomaly",
                    score=score,
                    model_family=artifact.family.value,
                    model_digest=artifact.model_digest,
                    roi_config_id=sample.sample.roi_config_id,
                    roi_digest=sample.roi_digest,
                    source_sha256=sample.sample.source_sha256,
                    crop_sha256=sample.sample.crop_sha256,
                    heatmap_path=heatmap_path,
                    heatmap_sha256=heatmap_sha256,
                    overlay_path=overlay_path,
                    overlay_sha256=overlay_sha256,
                )
            )
        return tuple(scores)


class AnomalibRuntimeBackend:
    """Callable runtime loader for already verified content-addressed artifacts."""

    def __call__(
        self,
        artifacts: Mapping[ModelSlot, AnomalyModelArtifact],
        device: DeviceSpec,
    ) -> AnomalibBatchPredictor:
        if not artifacts:
            raise ModelContractError("Anomalib runtime requires at least one artifact")
        torch, cuda_index = _require_cuda(device)
        loaded: dict[ModelSlot, _LoadedSlot] = {}
        families = {artifact.family for artifact in artifacts.values()}
        if len(families) != 1:
            raise ModelContractError("one Anomalib runtime cannot mix anomaly families")
        scratch_root = Path(tempfile.mkdtemp(prefix="zs32-anomalib-runtime-"))
        try:
            from anomalib import __version__ as anomalib_version

            actual_framework = f"anomalib-{anomalib_version}"
            for slot, artifact in artifacts.items():
                if slot != artifact.slot:
                    raise ModelContractError(f"artifact mapping key differs from slot: {slot.key}")
                if actual_framework != artifact.framework_version:
                    raise ModelContractError(
                        f"Anomalib runtime version mismatch for {slot.key}: "
                        f"expected {artifact.framework_version!r}, "
                        f"found {actual_framework!r}"
                    )
                artifact.verify()
                parameters, image_size = _validate_runtime_parameters(artifact)
                model = _build_model(artifact.family, parameters, for_training=False)
                checkpoint = torch.load(
                    artifact.checkpoint.path,
                    map_location="cpu",
                    weights_only=False,
                )
                if not isinstance(checkpoint, Mapping) or not isinstance(
                    checkpoint.get("state_dict"), Mapping
                ):
                    raise ModelContractError(
                        f"verified checkpoint has no Lightning state_dict: {slot.key}"
                    )
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                del checkpoint
                model.to(f"cuda:{cuda_index}")
                model.eval()
                model.freeze()
                _verify_finite_model(torch, model)
                from anomalib.engine import Engine

                engine = Engine(
                    default_root_dir=scratch_root / canonical_sha256(slot.key)[:16],
                    accelerator="gpu",
                    devices=[cuda_index],
                    logger=False,
                    enable_checkpointing=False,
                    enable_model_summary=False,
                    inference_mode=True,
                )
                loaded[slot] = _LoadedSlot(artifact, model, engine, image_size)
            torch.cuda.synchronize(cuda_index)
            return AnomalibBatchPredictor(loaded, torch, cuda_index, scratch_root)
        except (ModelContractError, BackendUnavailableError):
            shutil.rmtree(scratch_root, ignore_errors=True)
            raise
        except Exception as error:
            shutil.rmtree(scratch_root, ignore_errors=True)
            current = slot.key if "slot" in locals() else "unknown"
            family = next(iter(families)).value if families else "unknown"
            raise AnomalibBackendError(
                f"cannot restore verified {family} checkpoint for {current}: {error}"
            ) from error


__all__ = [
    "AnomalibBackendError",
    "AnomalibBatchPredictor",
    "AnomalibRuntimeBackend",
    "AnomalibTrainerBackend",
    "sha256_auxiliary_tree",
]
