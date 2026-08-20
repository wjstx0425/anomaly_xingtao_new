# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Single-path, hash-free ZS32 Demo inference runtime."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

import cv2
import numpy as np

from capture_data.zs32_demo_config import DemoConfig, load_demo_config
from capture_data.zs32_model_runtime import (
    AnomalibPatchcoreBackend,
    ModelEvidence,
    UltralyticsYoloBackend,
)
from capture_data.zs32_patchcore_multiprocess import ResidentMultiprocessPatchcoreBackend
from capture_data.zs32_template_gate import load_gray, match_template
from zs32_inspection.domain.views import VIEW_ORDER


class DemoRuntimeError(RuntimeError):
    """Raised when one Demo request cannot safely produce a result."""


@dataclass(frozen=True, slots=True)
class DemoTemplateScore:
    """Continuous Template result independent from any decision threshold."""

    score: float
    similarity: float
    best_template_path: Path
    offset: tuple[int, int]


@dataclass(frozen=True, slots=True)
class DemoRunResult:
    """Published result of one complete eight-view Demo request."""

    output_dir: Path
    machine_status: str
    inspection_complete: bool
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PatchcoreAsset:
    """Minimal adapter expected by the existing PatchCore backend."""

    checkpoint: Path


@dataclass(frozen=True, slots=True)
class _YoloAsset:
    """Minimal adapter expected by the existing YOLO backend."""

    weights: Path
    imgsz: int
    candidate_conf: float
    iou: float = 0.7
    max_det: int = 300
    class_map: Mapping[int, str] = field(default_factory=lambda: {0: "defect"})


@dataclass(frozen=True, slots=True)
class _Geometry:
    """Image geometry resolved from the independently validated ROI file."""

    image_width: int
    image_height: int
    rois: Mapping[str, tuple[int, int, int, int]]


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        output[key] = value
    return output


class DemoTemplateMatcher:
    """Eight-view Template matcher that loads images once and ignores release hashes."""

    def __init__(self, template_dir: Path) -> None:
        self.template_dir = template_dir.expanduser().resolve()
        model_path = self.template_dir / "model.json"
        try:
            payload = json.loads(
                model_path.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_unique_object,
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            raise DemoRuntimeError(f"could not load Template model {model_path}: {error}") from error
        if not isinstance(payload, dict):
            raise DemoRuntimeError("Template model.json must contain one JSON object")
        preprocessing = payload.get("preprocessing")
        groups = payload.get("groups")
        if not isinstance(preprocessing, dict) or not isinstance(groups, dict):
            raise DemoRuntimeError("Template model must contain preprocessing and groups objects")
        try:
            self.width = int(preprocessing["width"])
            self.max_shift = int(preprocessing["max_shift"])
        except (KeyError, TypeError, ValueError) as error:
            raise DemoRuntimeError("Template preprocessing width/max_shift is invalid") from error
        if self.width <= 0 or self.max_shift < 0:
            raise DemoRuntimeError("Template preprocessing width/max_shift is invalid")

        expected_groups = {f"right/{view}" for view in VIEW_ORDER}
        if set(groups) != expected_groups:
            missing = sorted(expected_groups - set(groups))
            extra = sorted(set(groups) - expected_groups)
            raise DemoRuntimeError(
                f"Template model must contain exactly eight groups; missing={missing}, extra={extra}",
            )
        self.templates: dict[str, tuple[tuple[Path, np.ndarray], ...]] = {}
        for view in VIEW_ORDER:
            group = groups[f"right/{view}"]
            records = group.get("templates") if isinstance(group, dict) else None
            if not isinstance(records, list) or not records:
                raise DemoRuntimeError(f"Template group right/{view} has no templates")
            loaded: list[tuple[Path, np.ndarray]] = []
            for index, record in enumerate(records):
                relative_text = record.get("path") if isinstance(record, dict) else None
                if not isinstance(relative_text, str) or not relative_text.strip():
                    raise DemoRuntimeError(f"Template group right/{view} record {index} has no path")
                relative = Path(relative_text)
                path = (self.template_dir / relative).resolve()
                if relative.is_absolute() or not path.is_relative_to(self.template_dir):
                    raise DemoRuntimeError(f"Template path escapes model directory: {relative_text}")
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise DemoRuntimeError(f"could not decode Template image: {path}")
                loaded.append((path, image))
            self.templates[view] = tuple(loaded)

    def score(self, image_path: Path, view: str) -> DemoTemplateScore:
        """Return the best continuous risk score without applying a threshold."""
        return self._score_preprocessed(load_gray(image_path, self.width), view)

    def score_array(self, image: np.ndarray, view: str) -> DemoTemplateScore:
        """Return the path-equivalent score for one authoritative grayscale crop."""
        if not isinstance(image, np.ndarray) or image.ndim != 2 or not image.size:
            raise DemoRuntimeError("Template image must be a non-empty 2D grayscale ndarray")
        gray = image
        if gray.shape[1] <= 0:
            raise DemoRuntimeError("Template image width must be positive")
        height = max(1, round(gray.shape[0] * self.width / gray.shape[1]))
        interpolation = cv2.INTER_AREA if self.width <= gray.shape[1] else cv2.INTER_LINEAR
        resized = cv2.resize(gray, (self.width, height), interpolation=interpolation)
        return self._score_preprocessed(cv2.GaussianBlur(resized, (3, 3), 0), view)

    def _score_preprocessed(self, image: np.ndarray, view: str) -> DemoTemplateScore:
        """Match one image after the model's grayscale preprocessing."""
        if view not in self.templates:
            raise DemoRuntimeError(f"Template view was not prepared: {view}")
        matches: list[tuple[float, Path, tuple[int, int]]] = []
        for template_path, template in self.templates[view]:
            similarity, offset = match_template(image, template, self.max_shift)
            matches.append((similarity, template_path, offset))
        similarity, best_path, offset = max(matches, key=lambda item: (item[0], str(item[1])))
        score = 1.0 - similarity
        if not math.isfinite(score):
            raise DemoRuntimeError(f"Template score is not finite for {view}")
        return DemoTemplateScore(score, similarity, best_path, offset)


def fuse_demo_status(statuses: Mapping[str, str]) -> str:
    """Fuse three branch statuses with the documented fail-closed precedence."""
    required = {"template", "patchcore", "yolo"}
    if set(statuses) != required:
        return "ERROR"
    normalized = {branch: str(status).strip().upper() for branch, status in statuses.items()}
    if any(status in {"ERROR", "FAILED", "EXECUTION_ERROR"} for status in normalized.values()):
        return "ERROR"
    if normalized["template"] == "NG_TEMPLATE":
        return "NG_TEMPLATE"
    if normalized["patchcore"] == "NG_ANOMALY":
        return "NG_ANOMALY"
    if normalized["yolo"] == "NG_YOLO":
        return "NG_YOLO"
    if normalized == {"template": "PASS", "patchcore": "PASS", "yolo": "PASS"}:
        return "OK"
    return "ERROR"


def _asset_identity(path_value: Path | str) -> tuple[str, int, int]:
    """Track an asset without hashing it so in-place changes require restart."""
    path = Path(path_value).expanduser().resolve()
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


def _template_asset_identity(template_dir: Path) -> tuple[tuple[str, int, int], ...]:
    """Track the prepared Template files only when a real model is present."""
    if not (template_dir / "model.json").is_file():
        return ()
    return tuple(
        _asset_identity(path)
        for path in sorted(template_dir.rglob("*"))
        if path.is_file()
    )


def _model_signature(config: DemoConfig | Any) -> tuple[Any, ...]:
    """Return fields that require a worker restart when changed."""
    patchcore = tuple(
        (
            view,
            _asset_identity(
                Path(getattr(config.patchcore[view], "checkpoint", config.patchcore[view])),
            ),
        )
        for view in VIEW_ORDER
    )
    yolo = config.yolo
    class_map = getattr(yolo, "class_map", {0: "defect"})
    template_dir = Path(config.template_dir).expanduser().resolve()
    return (
        int(getattr(config, "patchcore_process_count", 1)),
        _asset_identity(config.topology),
        _asset_identity(config.roi_config),
        str(template_dir),
        _template_asset_identity(template_dir),
        patchcore,
        _asset_identity(yolo.weights),
        int(yolo.imgsz),
        float(yolo.candidate_conf),
        float(getattr(yolo, "iou", 0.7)),
        int(getattr(yolo, "max_det", 300)),
        tuple(sorted((int(key), str(value)) for key, value in class_map.items())),
    )


def _effective_model_signature(config: DemoConfig | Any, patchcore_process_count: int) -> tuple[Any, ...]:
    """Combine configured assets with the process count selected for this runtime."""
    return (*_model_signature(config), ("effective_patchcore_process_count", patchcore_process_count))


def _geometry(config: DemoConfig | Any) -> _Geometry:
    """Use explicit test geometry or resolve it from the validated ROI JSON."""
    if all(hasattr(config, field) for field in ("image_width", "image_height", "rois")):
        return _Geometry(
            image_width=int(config.image_width),
            image_height=int(config.image_height),
            rois=config.rois,
        )
    path = Path(config.roi_config).expanduser().resolve()
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
        image_size = payload["image_size"]
        view_records = payload["views"]
        return _Geometry(
            image_width=int(image_size["width"]),
            image_height=int(image_size["height"]),
            rois={
                view: tuple(int(value) for value in view_records[view]["roi"])
                for view in VIEW_ORDER
            },
        )
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise DemoRuntimeError(f"could not resolve Demo ROI geometry from {path}: {error}") from error


def _status(score: float, threshold: float, ng_status: str) -> str:
    if not math.isfinite(score):
        return "ERROR"
    return ng_status if score >= threshold else "PASS"


def _branch_error(branch: str, error: Exception | str, threshold: float) -> dict[str, Any]:
    detail = str(error)
    return {
        "state": "error",
        "status": "ERROR",
        "score": None,
        "threshold": threshold,
        "reason": f"{branch} failed: {detail}",
    }


def _skipped_branch(branch: str, threshold: float | None) -> dict[str, Any]:
    return {
        "state": "skipped",
        "status": "SKIPPED",
        "score": None,
        "threshold": threshold,
        "reason": f"{branch} skipped because the global Template gate stopped this part",
    }


def _relative(path: Path, root: Path) -> str:
    return str(path.expanduser().resolve().relative_to(root.expanduser().resolve()))


def _validated_patchcore_process_count(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value not in {1, 2, 4, 8}:
        raise ValueError("PatchCore process count must be 1, 2, 4, or 8")
    return value


def _write_warmup_crops(geometry: _Geometry, directory: Path) -> dict[str, Path]:
    """Create one color warmup crop per view with its authoritative ROI geometry."""
    crops: dict[str, Path] = {}
    directory.mkdir(parents=True, exist_ok=True)
    for view in VIEW_ORDER:
        x1, y1, x2, y2 = geometry.rois[view]
        crop = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)
        path = directory / f"{view}.png"
        if crop.size == 0 or not cv2.imwrite(str(path), crop):
            raise DemoRuntimeError(f"could not prepare PatchCore warmup crop for {view}")
        crops[view] = path
    return crops


class ZS32DemoRuntime:
    """Resident eight-PatchCore, one-YOLO and eight-Template Demo runtime."""

    def __init__(
        self,
        config: DemoConfig,
        *,
        template_matcher: DemoTemplateMatcher | Any | None = None,
        patchcore_backend: Any | None = None,
        yolo_backend: Any | None = None,
        accelerator: str = "gpu",
        devices: int = 1,
        yolo_device: str | None = "0",
        config_loader: Callable[[Path], DemoConfig] | None = None,
        patchcore_process_count: int | None = None,
    ) -> None:
        self.startup_config = config
        self.config_loader = config_loader or load_demo_config
        self.template_matcher = template_matcher or DemoTemplateMatcher(config.template_dir)
        self._closed = False
        self.patchcore_process_count = _validated_patchcore_process_count(
            getattr(config, "patchcore_process_count", 1)
            if patchcore_process_count is None
            else patchcore_process_count,
        )
        self.startup_signature = _effective_model_signature(config, self.patchcore_process_count)
        patchcore_specs = {
            view: _PatchcoreAsset(
                Path(getattr(config.patchcore[view], "checkpoint", config.patchcore[view])),
            )
            for view in VIEW_ORDER
        }
        if patchcore_backend is not None:
            self.patchcore_backend = patchcore_backend
        elif self.patchcore_process_count == 1:
            self.patchcore_backend = AnomalibPatchcoreBackend(
                patchcore_specs,  # type: ignore[arg-type]
                views=VIEW_ORDER,
                accelerator=accelerator,
                devices=devices,
            )
        else:
            with tempfile.TemporaryDirectory(prefix="zs32-patchcore-runtime-warmup-") as directory:
                warmup_crops = _write_warmup_crops(_geometry(config), Path(directory))
                self.patchcore_backend = ResidentMultiprocessPatchcoreBackend(
                    patchcore_specs,  # type: ignore[arg-type]
                    VIEW_ORDER,
                    self.patchcore_process_count,
                    accelerator=accelerator,
                    devices=devices,
                    warmup_crops=warmup_crops,
                )
        yolo = config.yolo
        try:
            self.yolo_backend = yolo_backend or UltralyticsYoloBackend(
                _YoloAsset(
                    weights=Path(yolo.weights),
                    imgsz=int(yolo.imgsz),
                    candidate_conf=float(yolo.candidate_conf),
                    iou=float(getattr(yolo, "iou", 0.7)),
                    max_det=int(getattr(yolo, "max_det", 300)),
                    class_map=getattr(yolo, "class_map", {0: "defect"}),
                ),  # type: ignore[arg-type]
                device=yolo_device,
            )
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """Close the resident PatchCore backend once."""
        if self._closed:
            return
        self._closed = True
        close_backend = getattr(self.patchcore_backend, "close", None)
        if callable(close_backend):
            close_backend()

    def _current_config(self, config_path: Path) -> DemoConfig:
        current = self.config_loader(config_path)
        if _effective_model_signature(current, self.patchcore_process_count) != self.startup_signature:
            raise DemoRuntimeError(
                "Demo model, ROI, topology, or inference settings changed; restart Dashboard to load the new assets",
            )
        return current

    @staticmethod
    def _load_sources(
        config: DemoConfig | Any,
        images: Mapping[str, Path],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, bytes]]:
        if set(images) != set(VIEW_ORDER) or len(images) != len(VIEW_ORDER):
            missing = sorted(set(VIEW_ORDER) - set(images))
            extra = sorted(set(images) - set(VIEW_ORDER))
            raise DemoRuntimeError(f"request must contain exactly eight views; missing={missing}, extra={extra}")
        resolved = {view: Path(images[view]).expanduser().resolve() for view in VIEW_ORDER}
        if len(set(resolved.values())) != len(VIEW_ORDER):
            raise DemoRuntimeError("the eight views must use eight distinct source paths")
        color_sources: dict[str, np.ndarray] = {}
        grayscale_sources: dict[str, np.ndarray] = {}
        encoded_sources: dict[str, bytes] = {}
        for view in VIEW_ORDER:
            try:
                encoded = resolved[view].read_bytes()
            except OSError as error:
                raise DemoRuntimeError(f"could not read source image for {view}: {resolved[view]}") from error
            encoded_array = np.frombuffer(encoded, dtype=np.uint8)
            try:
                color = cv2.imdecode(encoded_array, cv2.IMREAD_COLOR)
                grayscale = cv2.imdecode(encoded_array, cv2.IMREAD_GRAYSCALE)
            except cv2.error as error:
                raise DemoRuntimeError(f"could not decode source image for {view}: {resolved[view]}") from error
            if color is None or grayscale is None:
                raise DemoRuntimeError(f"could not decode source image for {view}: {resolved[view]}")
            expected = (int(config.image_height), int(config.image_width))
            if tuple(color.shape[:2]) != expected or tuple(grayscale.shape[:2]) != expected:
                raise DemoRuntimeError(
                    f"source image for {view} must be {expected[1]}x{expected[0]}, "
                    f"found {color.shape[1]}x{color.shape[0]}",
                )
            color_sources[view] = color
            grayscale_sources[view] = grayscale
            encoded_sources[view] = encoded
        return color_sources, grayscale_sources, encoded_sources

    @staticmethod
    def _copy_sources(
        staging: Path,
        encoded_sources: Mapping[str, bytes],
    ) -> dict[str, Path]:
        source_paths: dict[str, Path] = {}
        for view in VIEW_ORDER:
            source_path = staging / "sources" / f"{view}.png"
            source_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                source_path.write_bytes(encoded_sources[view])
            except OSError as error:
                raise DemoRuntimeError(f"could not write source image for {view}: {error}") from error
            source_paths[view] = source_path
        return source_paths

    @staticmethod
    def _crop_arrays(
        config: DemoConfig | Any,
        sources: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        crop_arrays: dict[str, np.ndarray] = {}
        for view in VIEW_ORDER:
            x1, y1, x2, y2 = config.rois[view]
            crop = np.ascontiguousarray(sources[view][y1:y2, x1:x2])
            if crop.size == 0:
                raise DemoRuntimeError(f"ROI produced an empty crop for {view}")
            crop_arrays[view] = crop
        return crop_arrays

    @staticmethod
    def _write_crops(
        staging: Path,
        crop_arrays: Mapping[str, np.ndarray],
    ) -> dict[str, Path]:
        crop_paths: dict[str, Path] = {}
        for view in VIEW_ORDER:
            crop_path = staging / "crops" / f"{view}.png"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(crop_path), crop_arrays[view]):
                raise DemoRuntimeError(f"could not write model crop for {view}: {crop_path}")
            crop_paths[view] = crop_path
        return crop_paths

    @staticmethod
    def _fusion_evidence(path: Path, source: np.ndarray, status: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.array(source, copy=True)
        color = (0, 180, 0) if status in {"OK", "PASS"} else (0, 0, 255)
        cv2.putText(
            image,
            f"DEMO {status}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            color,
            3,
            lineType=cv2.LINE_AA,
        )
        if not cv2.imwrite(str(path), image):
            raise DemoRuntimeError(f"could not write fusion evidence: {path}")

    @staticmethod
    def _publish_result(
        *,
        staging: Path,
        output_dir: Path,
        config: DemoConfig | Any,
        sources: Mapping[str, np.ndarray],
        source_paths: Mapping[str, Path],
        branch_records: Mapping[str, Mapping[str, dict[str, Any]]],
        part_id: str,
        capture_session: str,
        group_id: str,
        machine_status: str,
        inspection_complete: bool,
        errors: list[str],
        evaluated_branch_count: int,
        reason: str,
    ) -> DemoRunResult:
        views = {
            view: {
                "view": view,
                "part_id": part_id,
                "capture_session": capture_session,
                "group_id": group_id,
                "hand": "right",
                "manifest_identity": f"{part_id}:right:{view}",
                "source_path": _relative(source_paths[view], staging),
                "source_shape": list(sources[view].shape[:2]),
                "model_supported": True,
                "branches": branch_records[view],
            }
            for view in VIEW_ORDER
        }
        manifest = {
            "schema_version": 1,
            "mode": "offline",
            "product": "ZS32",
            "manifest_identity": "ZS32/right",
            "part_id": part_id,
            "capture_session": capture_session,
            "group_id": group_id,
            "hand": "right",
            "machine_status": machine_status,
            "inspection_complete": inspection_complete,
            "reason": reason,
            "errors": errors,
            "evaluated_views": list(VIEW_ORDER),
            "evaluated_branch_count": evaluated_branch_count,
            "planned_branch_count": len(VIEW_ORDER) * 3,
            "demo_config": str(Path(config.path).expanduser().resolve()),
            "demo_config_mtime_ns": Path(config.path).stat().st_mtime_ns
            if Path(config.path).is_file()
            else None,
            "views": views,
        }
        (staging / "runtime_manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        staging.replace(output_dir)
        return DemoRunResult(output_dir, machine_status, inspection_complete, tuple(errors))

    def run(
        self,
        images: Mapping[str, Path],
        *,
        part_id: str,
        capture_session: str,
        group_id: str,
        output_dir: Path,
        config_path: Path,
    ) -> DemoRunResult:
        """Apply the global Template gate, then publish one atomic Demo result."""
        for name, value in {
            "part_id": part_id,
            "capture_session": capture_session,
            "group_id": group_id,
        }.items():
            if not isinstance(value, str) or not value.strip():
                raise DemoRuntimeError(f"{name} must be a non-empty string")
        config = self._current_config(Path(config_path).expanduser().resolve())
        geometry = _geometry(config)
        sources, grayscale_sources, encoded_sources = self._load_sources(geometry, images)
        output_dir = Path(output_dir).expanduser().resolve()
        if output_dir.exists():
            raise FileExistsError(f"output directory already exists: {output_dir}")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
        try:
            source_paths = self._copy_sources(staging, encoded_sources)
            crop_arrays = self._crop_arrays(geometry, sources)
            template_crop_arrays = self._crop_arrays(geometry, grayscale_sources)
            errors: list[str] = []
            branch_records: dict[str, dict[str, dict[str, Any]]] = {
                view: {} for view in VIEW_ORDER
            }

            for view in VIEW_ORDER:
                threshold = float(config.thresholds.template[view])
                try:
                    result = self.template_matcher.score_array(template_crop_arrays[view], view)
                    evidence_path = staging / "evidence" / "template" / f"{view}.png"
                    evidence_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(result.best_template_path, evidence_path)
                    status = _status(float(result.score), threshold, "NG_TEMPLATE")
                    branch_records[view]["template"] = {
                        "state": "available",
                        "status": status,
                        "score": float(result.score),
                        "threshold": threshold,
                        "reason": (
                            "template risk reached the Demo threshold"
                            if status == "NG_TEMPLATE"
                            else "template risk is below the Demo threshold"
                        ),
                        "evidence_path": _relative(evidence_path, staging),
                        "roi_xyxy": list(geometry.rois[view]),
                        "similarity": float(result.similarity),
                        "offset": list(result.offset),
                    }
                except Exception as error:  # noqa: BLE001 - one view must not short-circuit other branches
                    message = f"{view}: Template failed: {type(error).__name__}: {error}"
                    errors.append(message)
                    branch_records[view]["template"] = _branch_error("Template", error, threshold)

            template_ng = any(
                branch_records[view]["template"]["status"] == "NG_TEMPLATE"
                for view in VIEW_ORDER
            )
            if errors or template_ng:
                for view in VIEW_ORDER:
                    branch_records[view]["patchcore"] = _skipped_branch(
                        "PatchCore",
                        float(config.thresholds.patchcore[view]),
                    )
                    branch_records[view]["yolo"] = _skipped_branch(
                        "YOLO",
                        float(config.thresholds.yolo[view]),
                    )
                    if errors:
                        branch_records[view]["fusion"] = _skipped_branch("Fusion", None)
                    else:
                        fusion_status = str(branch_records[view]["template"]["status"])
                        evidence_path = staging / "evidence" / "fusion" / f"{view}.png"
                        self._fusion_evidence(evidence_path, sources[view], fusion_status)
                        branch_records[view]["fusion"] = {
                            "state": "available",
                            "status": fusion_status,
                            "score": None,
                            "threshold": None,
                            "reason": (
                                "this view's Template result rejected the part"
                                if fusion_status == "NG_TEMPLATE"
                                else "this view's Template result passed; another view stopped the part"
                            ),
                            "evidence_path": _relative(evidence_path, staging),
                        }
                machine_status = "ERROR" if errors else "NG_TEMPLATE"
                reason = errors[0] if errors else "global Template gate stopped PatchCore and YOLO"
                return self._publish_result(
                    staging=staging,
                    output_dir=output_dir,
                    config=config,
                    sources=sources,
                    source_paths=source_paths,
                    branch_records=branch_records,
                    part_id=part_id,
                    capture_session=capture_session,
                    group_id=group_id,
                    machine_status=machine_status,
                    inspection_complete=not errors,
                    errors=errors,
                    evaluated_branch_count=len(VIEW_ORDER),
                    reason=reason,
                )

            crop_paths = self._write_crops(staging, crop_arrays)
            patchcore_results: Mapping[str, ModelEvidence | Exception]
            try:
                raw_results = self.patchcore_backend.predict_all(
                    crop_paths,
                    staging / "evidence" / "patchcore",
                )
                patchcore_results = raw_results if isinstance(raw_results, Mapping) else {}
            except Exception as error:  # noqa: BLE001 - YOLO must still run
                patchcore_results = {view: error for view in VIEW_ORDER}
            for view in VIEW_ORDER:
                candidate = patchcore_results.get(view, RuntimeError("PatchCore result missing"))
                threshold = float(config.thresholds.patchcore[view])
                if isinstance(candidate, Exception):
                    message = f"{view}: PatchCore failed: {type(candidate).__name__}: {candidate}"
                    errors.append(message)
                    branch_records[view]["patchcore"] = _branch_error("PatchCore", candidate, threshold)
                    continue
                try:
                    score = float(candidate.score)
                    artifacts = candidate.patchcore_artifacts
                    if not math.isfinite(score) or artifacts is None:
                        raise DemoRuntimeError("PatchCore evidence is incomplete or non-finite")
                    for path in (
                        candidate.evidence_path,
                        artifacts.raw_anomaly_map_path,
                        artifacts.mask_path,
                    ):
                        if not path.is_file():
                            raise DemoRuntimeError(f"PatchCore evidence file is missing: {path}")
                    status = _status(score, threshold, "NG_ANOMALY")
                    branch_records[view]["patchcore"] = {
                        "state": "available",
                        "status": status,
                        "score": score,
                        "threshold": threshold,
                        "reason": (
                            "PatchCore score reached the Demo threshold"
                            if status == "NG_ANOMALY"
                            else "PatchCore score is below the Demo threshold"
                        ),
                        "evidence_path": _relative(candidate.evidence_path, staging),
                        "raw_anomaly_map_path": _relative(artifacts.raw_anomaly_map_path, staging),
                        "mask_path": _relative(artifacts.mask_path, staging),
                        "mask_source": artifacts.mask_source,
                        "roi_xyxy": list(geometry.rois[view]),
                    }
                except Exception as error:  # noqa: BLE001 - preserve per-view error
                    message = f"{view}: PatchCore failed: {type(error).__name__}: {error}"
                    errors.append(message)
                    branch_records[view]["patchcore"] = _branch_error("PatchCore", error, threshold)

            try:
                raw_yolo_results = self.yolo_backend.predict(
                    crop_arrays,
                    staging / "evidence" / "yolo",
                )
                yolo_results: Mapping[str, ModelEvidence | Exception] = (
                    raw_yolo_results if isinstance(raw_yolo_results, Mapping) else {}
                )
            except Exception as error:  # noqa: BLE001 - publish all eight error records
                yolo_results = {view: error for view in VIEW_ORDER}
            for view in VIEW_ORDER:
                candidate = yolo_results.get(view, RuntimeError("YOLO result missing"))
                threshold = float(config.thresholds.yolo[view])
                if isinstance(candidate, Exception):
                    message = f"{view}: YOLO failed: {type(candidate).__name__}: {candidate}"
                    errors.append(message)
                    branch_records[view]["yolo"] = _branch_error("YOLO", candidate, threshold)
                    continue
                try:
                    score = float(candidate.score)
                    detections = candidate.detections
                    if not math.isfinite(score) or detections is None or not candidate.evidence_path.is_file():
                        raise DemoRuntimeError("YOLO evidence is incomplete or non-finite")
                    status = _status(score, threshold, "NG_YOLO")
                    branch_records[view]["yolo"] = {
                        "state": "available",
                        "status": status,
                        "score": score,
                        "threshold": threshold,
                        "reason": (
                            "YOLO confidence reached the Demo threshold"
                            if status == "NG_YOLO"
                            else "YOLO confidence is below the Demo threshold"
                        ),
                        "evidence_path": _relative(candidate.evidence_path, staging),
                        "roi_xyxy": list(geometry.rois[view]),
                        "detections": list(detections),
                    }
                except Exception as error:  # noqa: BLE001 - preserve per-view error
                    message = f"{view}: YOLO failed: {type(error).__name__}: {error}"
                    errors.append(message)
                    branch_records[view]["yolo"] = _branch_error("YOLO", error, threshold)

            view_statuses: dict[str, str] = {}
            for view in VIEW_ORDER:
                statuses = {
                    branch: str(branch_records[view][branch]["status"])
                    for branch in ("template", "patchcore", "yolo")
                }
                fusion_status = fuse_demo_status(statuses)
                evidence_path = staging / "evidence" / "fusion" / f"{view}.png"
                self._fusion_evidence(evidence_path, sources[view], fusion_status)
                branch_records[view]["fusion"] = {
                    "state": "error" if fusion_status == "ERROR" else "available",
                    "status": fusion_status,
                    "score": None,
                    "threshold": None,
                    "reason": (
                        "one or more required branches failed"
                        if fusion_status == "ERROR"
                        else "Demo fusion precedence applied"
                    ),
                    "evidence_path": _relative(evidence_path, staging),
                }
                view_statuses[view] = fusion_status

            if any(status == "ERROR" for status in view_statuses.values()):
                machine_status = "ERROR"
            elif any(status == "NG_TEMPLATE" for status in view_statuses.values()):
                machine_status = "NG_TEMPLATE"
            elif any(status == "NG_ANOMALY" for status in view_statuses.values()):
                machine_status = "NG_ANOMALY"
            elif any(status == "NG_YOLO" for status in view_statuses.values()):
                machine_status = "NG_YOLO"
            else:
                machine_status = "OK"
            inspection_complete = machine_status != "ERROR" and not errors

            reason = (
                errors[0]
                if errors
                else {
                    "OK": "all 24 Demo model branches passed",
                    "NG_TEMPLATE": "at least one Template score reached its threshold",
                    "NG_ANOMALY": "at least one PatchCore score reached its threshold",
                    "NG_YOLO": "at least one YOLO confidence reached its threshold",
                }[machine_status]
            )
            return self._publish_result(
                staging=staging,
                output_dir=output_dir,
                config=config,
                sources=sources,
                source_paths=source_paths,
                branch_records=branch_records,
                part_id=part_id,
                capture_session=capture_session,
                group_id=group_id,
                machine_status=machine_status,
                inspection_complete=inspection_complete,
                errors=errors,
                evaluated_branch_count=len(VIEW_ORDER) * 3,
                reason=reason,
            )
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
