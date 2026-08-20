# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Atomic publication of complete BMW inspection evidence bundles."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import cv2
import numpy as np

from bmw_inspection.lab.config import LabExperimentConfig
from bmw_inspection.lab.contracts import BranchName, BranchStatus, InspectionResult, ViewId
from bmw_inspection.lab.yolo import YoloEvidence, render_final_overlay


ArtifactKey = tuple[BranchName, ViewId]


@dataclass(frozen=True, slots=True)
class PublicationArtifacts:
    """Raw branch decisions and exact input crops retained for publication."""

    decisions: Mapping[ArtifactKey, object] = MappingProxyType({})
    crops: Mapping[ArtifactKey, np.ndarray] = MappingProxyType({})

    def __post_init__(self) -> None:
        decisions = {identity: _owned_value(decision) for identity, decision in self.decisions.items()}
        crops: dict[ArtifactKey, np.ndarray] = {}
        for identity, image in self.crops.items():
            if not isinstance(image, np.ndarray):
                raise TypeError("publication crops must be numpy arrays")
            owned = image.copy()
            owned.flags.writeable = False
            crops[identity] = owned
        object.__setattr__(self, "decisions", MappingProxyType(decisions))
        object.__setattr__(self, "crops", MappingProxyType(crops))


def _owned_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        owned = value.copy()
        owned.flags.writeable = False
        return owned
    if is_dataclass(value) and not isinstance(value, type):
        updates = {field.name: _owned_value(getattr(value, field.name)) for field in fields(value)}
        return replace(value, **updates)
    if isinstance(value, Mapping):
        return MappingProxyType({key: _owned_value(item) for key, item in value.items()})
    if isinstance(value, tuple):
        return tuple(_owned_value(item) for item in value)
    if isinstance(value, list):
        return tuple(_owned_value(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class PublishedInspection:
    """Immutable pointer to one fully published evidence directory."""

    result: InspectionResult
    run_dir: Path
    config_digest: str
    triggered_branches: tuple[BranchName, ...]


class EvidencePublisher:
    """Stage a complete result privately, then expose it with one rename."""

    def __init__(self, result_root: Path) -> None:
        self._result_root = Path(result_root).expanduser().resolve()

    def publish(
        self,
        result: InspectionResult,
        *,
        config: LabExperimentConfig,
        triggered_branches: tuple[BranchName, ...],
        artifacts: PublicationArtifacts,
        run_id: str | None = None,
    ) -> PublishedInspection:
        """Publish source pixels, decisions, configuration and timings atomically."""
        if not isinstance(result, InspectionResult):
            raise TypeError("result must be InspectionResult")
        if not isinstance(config, LabExperimentConfig):
            raise TypeError("config must be LabExperimentConfig")
        if not isinstance(artifacts, PublicationArtifacts):
            raise TypeError("artifacts must be PublicationArtifacts")
        _validate_publication(result, triggered_branches, artifacts)
        resolved_run_id = run_id or _default_run_id(result)
        if not resolved_run_id or Path(resolved_run_id).name != resolved_run_id:
            raise ValueError("run_id must be one non-empty path component")

        day = result.capture_set.created_at.strftime("%Y%m%d")
        run_parent = self._result_root / config.experiment_id / "runs" / day
        run_parent.mkdir(parents=True, exist_ok=True)
        final_dir = run_parent / resolved_run_id
        if final_dir.exists():
            raise FileExistsError(f"inspection run already exists: {final_dir}")
        staging = Path(tempfile.mkdtemp(prefix=f".{resolved_run_id}.staging-", dir=run_parent))

        try:
            config_payload = _json_value(config)
            config_bytes = _canonical_json(config_payload)
            config_digest = hashlib.sha256(config_bytes).hexdigest()
            _write_json(
                staging / "config.snapshot.json",
                {"config_digest": config_digest, "config": config_payload},
            )
            _write_result(staging, result, triggered_branches, config_digest)
            _write_model_identities(staging, result, config)
            _write_sources(staging, result)
            _write_crops(staging, artifacts)
            _write_branch_evidence(staging, result, artifacts, config)
            _write_fusion_views(staging, result, triggered_branches)
            _write_timings(staging, result)
            staging.rename(final_dir)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        return PublishedInspection(result, final_dir, config_digest, tuple(triggered_branches))


def _default_run_id(result: InspectionResult) -> str:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S.%f")
    return f"{stamp}-{result.capture_set.capture_set_id}"


def _validate_publication(
    result: InspectionResult,
    triggered: tuple[BranchName, ...],
    artifacts: PublicationArtifacts,
) -> None:
    expected_triggered = tuple(
        branch
        for branch in BranchName
        if any(row.branch is branch and row.status is BranchStatus.NG for row in result.evidence)
    )
    if tuple(triggered) != expected_triggered:
        raise ValueError("triggered_branches must exactly match NG result evidence")
    evidence_by_identity = {(row.branch, row.view_id): row for row in result.evidence}
    artifact_identities = set(artifacts.decisions) | set(artifacts.crops)
    unknown = artifact_identities - set(evidence_by_identity)
    if unknown:
        rendered = ", ".join(f"{branch.value}/{view.value}" for branch, view in sorted(unknown, key=str))
        raise ValueError(f"artifact identity has no corresponding result evidence: {rendered}")
    for identity, decision in artifacts.decisions.items():
        decision_view = getattr(decision, "view_id", None)
        if decision_view is not None and decision_view != identity[1]:
            branch, view_id = identity
            raise ValueError(f"artifact view identity contradicts key: {branch.value}/{view_id.value}")
        decision_threshold = getattr(decision, "threshold", None)
        evidence_threshold = evidence_by_identity[identity].threshold
        if decision_threshold is not None and decision_threshold != evidence_threshold:
            branch, view_id = identity
            raise ValueError(f"artifact threshold contradicts result evidence: {branch.value}/{view_id.value}")
    for identity, row in evidence_by_identity.items():
        if row.status is not BranchStatus.SKIPPED and identity not in artifacts.crops:
            raise ValueError(f"artifact crop is missing for result evidence: {row.branch.value}/{row.view_id.value}")
        decision_required = (
            row.status not in {BranchStatus.SKIPPED, BranchStatus.ERROR}
            and not isinstance(row, YoloEvidence)
        )
        if decision_required and identity not in artifacts.decisions:
            raise ValueError(
                f"artifact decision is missing for result evidence: "
                f"{row.branch.value}/{row.view_id.value}"
            )


def _write_result(
    staging: Path,
    result: InspectionResult,
    triggered: tuple[BranchName, ...],
    config_digest: str,
) -> None:
    payload = {
        "capture_set_id": result.capture_set.capture_set_id,
        "created_at": result.capture_set.created_at,
        "final_status": result.final_status,
        "reason": result.reason,
        "required_complete": result.required_complete,
        "triggered_branches": triggered,
        "config_digest": config_digest,
        "evidence": result.evidence,
    }
    _write_json(staging / "result.json", payload)


def _write_sources(staging: Path, result: InspectionResult) -> None:
    for view_id in ViewId:
        _write_image(
            staging / "source" / f"{view_id.value}.png",
            result.capture_set.views[view_id].image,
            "source image",
        )


def _write_crops(staging: Path, artifacts: PublicationArtifacts) -> None:
    for (branch, view_id), image in artifacts.crops.items():
        _write_image(
            staging / "crops" / branch.value / f"{view_id.value}.png",
            image,
            "branch crop",
        )


def _write_branch_evidence(
    staging: Path,
    result: InspectionResult,
    artifacts: PublicationArtifacts,
    config: LabExperimentConfig,
) -> None:
    for row in result.evidence:
        branch_dir = staging / "evidence" / row.branch.value / row.view_id.value
        _write_json(branch_dir / "result.json", row)
        if isinstance(row, YoloEvidence):
            _write_json(
                branch_dir / "candidates.json",
                {"candidates": row.candidates, "final_boxes": row.final_boxes},
            )
            overlay = render_final_overlay(result.capture_set.views[row.view_id].image, row)
            _write_image(branch_dir / "overlay.png", overlay, "YOLO final overlay")

    for (branch, view_id), decision in artifacts.decisions.items():
        branch_dir = staging / "evidence" / branch.value / view_id.value
        _write_json(branch_dir / "decision.json", decision)
        source = result.capture_set.views[view_id].image
        crop = artifacts.crops.get((branch, view_id))
        raw_map = getattr(decision, "raw_anomaly_map", None)
        if isinstance(raw_map, np.ndarray):
            branch_dir.mkdir(parents=True, exist_ok=True)
            np.save(branch_dir / "anomaly_map.npy", raw_map, allow_pickle=False)
        heatmap = getattr(decision, "display_heatmap", None)
        if isinstance(heatmap, np.ndarray):
            overlay = _patchcore_overlay(source, heatmap, config.part_rois[view_id])
            _write_image(branch_dir / "overlay.png", overlay, "PatchCore source overlay")
        for name in ("roi", "response", "mask", "mask_used_for_metrics"):
            image = getattr(decision, name, None)
            if isinstance(image, np.ndarray):
                display = _display_image(image)
                _write_image(branch_dir / f"{name}.png", display, f"{branch.value} {name}")
                if name == "response":
                    np.save(branch_dir / "response.npy", image, allow_pickle=False)
        if branch is BranchName.TEMPLATE and crop is not None:
            _write_template_visuals(branch_dir, crop, decision)
        if branch is BranchName.BRIGHT_STREAK:
            _write_bright_overlay(branch_dir, source, decision)


def _write_fusion_views(
    staging: Path,
    result: InspectionResult,
    triggered: tuple[BranchName, ...],
) -> None:
    yolo_by_view = {
        row.view_id: row
        for row in result.evidence
        if isinstance(row, YoloEvidence)
    }
    for view_id in ViewId:
        source = result.capture_set.views[view_id].image
        row = yolo_by_view.get(view_id)
        fused = render_final_overlay(source, row) if row is not None else source
        view_findings = tuple(
            f"{evidence.branch.value}={evidence.status.value}"
            for evidence in result.evidence
            if evidence.view_id is view_id and evidence.status is not BranchStatus.PASS
        )
        fused = _annotate_fusion(fused, result.final_status.value, view_findings, triggered)
        _write_image(staging / "evidence" / "fusion" / f"{view_id.value}.png", fused, "fusion view")


def _annotate_fusion(
    image: np.ndarray,
    final_status: str,
    view_findings: tuple[str, ...],
    all_triggered: tuple[BranchName, ...],
) -> np.ndarray:
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    color = (0, 0, 255) if final_status.startswith("NG_") or final_status == "ERROR" else (0, 200, 255)
    if final_status == "OK":
        color = (0, 180, 0)
    height, width = overlay.shape[:2]
    thickness = max(1, min(height, width) // 100)
    cv2.rectangle(overlay, (0, 0), (max(0, width - 1), max(0, height - 1)), color, thickness)
    band_height = min(height, max(2, height // 14))
    overlay[:band_height, :] = color
    view_text = ",".join(view_findings) or "clear"
    all_text = ",".join(branch.value for branch in all_triggered) or "none"
    if width >= 80 and height >= 30:
        cv2.putText(
            overlay,
            f"FINAL={final_status} VIEW={view_text} ALL_NG={all_text}",
            (5, max(14, band_height - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return overlay


def _write_template_visuals(branch_dir: Path, crop: np.ndarray, decision: object) -> None:
    template_path = getattr(decision, "best_template_path", None)
    if not isinstance(template_path, Path):
        return
    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise RuntimeError(f"failed to read Template decision image: {template_path}")
    from bmw_inspection.lab.template import _preprocess

    query = _preprocess(crop, (template.shape[1], template.shape[0]))
    offset_x = int(getattr(decision, "offset_x", 0))
    offset_y = int(getattr(decision, "offset_y", 0))
    padding = max(abs(offset_x), abs(offset_y), 1)
    padded = cv2.copyMakeBorder(query, padding, padding, padding, padding, cv2.BORDER_REFLECT_101)
    start_x = padding + offset_x
    start_y = padding + offset_y
    aligned = padded[start_y : start_y + template.shape[0], start_x : start_x + template.shape[1]]
    difference = cv2.absdiff(aligned, template)
    heatmap = cv2.applyColorMap(difference, cv2.COLORMAP_TURBO)
    base = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base, 0.65, heatmap, 0.35, 0.0)
    _write_image(branch_dir / "difference.png", difference, "Template difference")
    _write_image(branch_dir / "overlay.png", overlay, "Template difference overlay")


def _write_bright_overlay(branch_dir: Path, source: np.ndarray, decision: object) -> None:
    mask = getattr(decision, "mask_used_for_metrics", None)
    if not isinstance(mask, np.ndarray):
        mask = getattr(decision, "mask", None)
    if not isinstance(mask, np.ndarray):
        return
    roi_xyxy = getattr(decision, "roi_xyxy", None)
    if roi_xyxy is None:
        result = getattr(decision, "result", None)
        roi_xyxy = getattr(result, "roi_xyxy", None)
    overlay = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR) if source.ndim == 2 else source.copy()
    height, width = overlay.shape[:2]
    if not isinstance(roi_xyxy, tuple) or len(roi_xyxy) != 4:
        roi_xyxy = (0, 0, width, height)
    x1, y1, x2, y2 = roi_xyxy
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        x1, y1, x2, y2 = 0, 0, width, height
    resized_mask = cv2.resize(_display_image(mask), (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    region = overlay[y1:y2, x1:x2]
    tint = np.zeros_like(region)
    tint[..., 1] = resized_mask
    overlay[y1:y2, x1:x2] = cv2.addWeighted(region, 0.65, tint, 0.35, 0.0)
    cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 1)
    _write_image(branch_dir / "overlay.png", overlay, "bright-streak source overlay")


def _patchcore_overlay(
    source: np.ndarray,
    heatmap: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
) -> np.ndarray:
    overlay = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR) if source.ndim == 2 else source.copy()
    height, width = overlay.shape[:2]
    x1, y1, x2, y2 = roi_xyxy
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        x1, y1, x2, y2 = 0, 0, width, height
    display = _display_image(heatmap)
    if display.ndim == 2:
        display = cv2.applyColorMap(display, cv2.COLORMAP_TURBO)
    display = cv2.resize(display, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
    region = overlay[y1:y2, x1:x2]
    overlay[y1:y2, x1:x2] = cv2.addWeighted(region, 0.55, display, 0.45, 0.0)
    return overlay


def _display_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    finite = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    minimum = float(finite.min()) if finite.size else 0.0
    maximum = float(finite.max()) if finite.size else 0.0
    if maximum <= minimum:
        return np.zeros(finite.shape, dtype=np.uint8)
    return np.rint((finite - minimum) * (255.0 / (maximum - minimum))).astype(np.uint8)


def _write_model_identities(
    staging: Path,
    result: InspectionResult,
    config: LabExperimentConfig,
) -> None:
    evidence_receipts = []
    for row in result.evidence:
        path = _preferred_artifact_path(row.artifact_paths)
        receipt = _model_receipt(row.branch, row.view_id, path)
        receipt["model_id"] = row.model_id
        if (
            isinstance(row.model_id, str)
            and len(row.model_id) == 64
            and receipt["sha256"] is not None
            and row.model_id != receipt["sha256"]
        ):
            raise RuntimeError(f"published model bytes changed after inference: {row.branch.value}/{row.view_id.value}")
        evidence_receipts.append(receipt)

    config_receipts = []
    for view_id, group in config.template.groups.items():
        if group.model_path is not None:
            config_receipts.append(_model_receipt(BranchName.TEMPLATE, view_id, group.model_path))
    for view_id, path in config.bright_streak.config_paths.items():
        config_receipts.append(_model_receipt(BranchName.BRIGHT_STREAK, view_id, path))
    if config.yolo.checkpoint is not None:
        config_receipts.append(_model_receipt(BranchName.YOLO, None, config.yolo.checkpoint))
    for view_id, path in config.patchcore.checkpoints.items():
        if path is not None:
            config_receipts.append(_model_receipt(BranchName.PATCHCORE, view_id, path))

    _write_json(
        staging / "model_identities.json",
        {"evidence": evidence_receipts, "config_assets": config_receipts},
    )


def _preferred_artifact_path(artifact_paths: Mapping[str, str]) -> Path | None:
    for key in ("checkpoint", "model", "config"):
        value = artifact_paths.get(key)
        if value is not None:
            return Path(value).expanduser().resolve()
    return None


def _model_receipt(
    branch: BranchName,
    view_id: ViewId | None,
    path: Path | None,
) -> dict[str, object]:
    version = None
    sha256 = None
    dependencies: list[dict[str, object]] = []
    resolved = Path(path).expanduser().resolve() if path is not None else None
    if resolved is not None and resolved.is_file():
        sha256 = _sha256_file(resolved)
        if resolved.suffix.lower() == ".json":
            try:
                payload = json.loads(resolved.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict):
                version = payload.get("schema_version", payload.get("version"))
                if branch is BranchName.TEMPLATE and isinstance(payload.get("templates"), list):
                    for item in payload["templates"]:
                        if isinstance(item, dict) and isinstance(item.get("path"), str):
                            dependency_path = (resolved.parent / item["path"]).resolve()
                            actual_sha = _sha256_file(dependency_path) if dependency_path.is_file() else None
                            declared_sha = item.get("sha256")
                            if isinstance(declared_sha, str) and actual_sha != declared_sha:
                                raise RuntimeError(f"Template dependency checksum mismatch: {dependency_path}")
                            dependencies.append(
                                {
                                    "path": str(dependency_path),
                                    "sha256": actual_sha,
                                    "declared_sha256": declared_sha,
                                }
                            )
    return {
        "branch": branch.value,
        "view_id": view_id.value if view_id is not None else None,
        "path": str(resolved) if resolved is not None else None,
        "version": version,
        "sha256": sha256,
        "dependencies": dependencies,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_timings(staging: Path, result: InspectionResult) -> None:
    _write_json(
        staging / "timing.json",
        {
            "rows": [
                {"branch": row.branch, "view_id": row.view_id, "elapsed_ms": row.elapsed_ms}
                for row in result.evidence
            ],
            "total_branch_elapsed_ms": sum(row.elapsed_ms for row in result.evidence),
        },
    )


def _write_image(path: Path, image: np.ndarray, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write {label}: {path}")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(_json_value(payload)) + b"\n")


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _json_value(value: object) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "stored_separately": True}
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {_json_key(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((_json_value(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _json_key(value: object) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)
