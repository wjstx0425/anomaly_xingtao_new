# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Template-first orchestration for one six-view BMW laboratory inspection."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from bmw_inspection.lab.config import LabExperimentConfig
from bmw_inspection.lab.contracts import (
    BranchEvidence,
    BranchName,
    BranchStatus,
    CaptureSet,
    InspectionResult,
    ViewId,
)
from bmw_inspection.lab.fusion import fuse_inspection
from bmw_inspection.lab.publisher import EvidencePublisher, PublicationArtifacts
from bmw_inspection.lab.yolo import YoloEvidence


class RuntimeState(str, Enum):
    """Whether an inspection was published or resident assets must be reloaded."""

    PUBLISHED = "PUBLISHED"
    RELOAD_REQUIRED = "RELOAD_REQUIRED"


@dataclass(frozen=True, slots=True)
class QualityDecision:
    """One pre-inference image-quality decision."""

    view_id: ViewId
    passed: bool
    reason: str


@dataclass(frozen=True, slots=True)
class RuntimeOutcome:
    """Immutable result of one runtime attempt."""

    state: RuntimeState
    inspection: InspectionResult | None
    run_dir: Path | None
    triggered_branches: tuple[BranchName, ...]
    reason: str


class LabRuntime:
    """Run quality and all six Template checks before any downstream branch."""

    def __init__(
        self,
        *,
        initial_config: LabExperimentConfig,
        config_loader: Callable[[], LabExperimentConfig],
        template_backend: Any,
        bright_streak_backend: Any,
        yolo_backend: Any,
        patchcore_backend: Any,
        publisher: EvidencePublisher,
        quality_checker: Callable[[ViewId, np.ndarray], QualityDecision] | None = None,
    ) -> None:
        self.initial_config = initial_config
        self.config_loader = config_loader
        self.template_backend = template_backend
        self.bright_streak_backend = bright_streak_backend
        self.yolo_backend = yolo_backend
        self.patchcore_backend = patchcore_backend
        self.publisher = publisher
        self.quality_checker = quality_checker or _quality_pass
        self._cold_digest = _cold_digest(initial_config)

    def inspect(self, capture_set: CaptureSet, *, run_id: str | None = None) -> RuntimeOutcome:
        """Inspect one immutable capture set using a fresh hot-config snapshot."""
        config = self.config_loader()
        if _cold_digest(config) != self._cold_digest:
            return RuntimeOutcome(
                RuntimeState.RELOAD_REQUIRED,
                None,
                None,
                (),
                "checkpoint, ROI, preprocessing, capture, topology, or other cold identity changed",
            )
        _validate_capture(capture_set, config)

        retake_reasons = []
        for view_id in ViewId:
            decision = self.quality_checker(view_id, capture_set.views[view_id].image)
            if decision.view_id is not view_id:
                raise ValueError("quality decision view does not match requested view")
            if not decision.passed:
                retake_reasons.append(f"{view_id.value}: {decision.reason}")
        if retake_reasons:
            fused = fuse_inspection(
                capture_set,
                (),
                expected_required=_expected_required(config),
                retake_reasons=retake_reasons,
            )
            return self._publish(config, fused.result, fused.triggered_branches, PublicationArtifacts(), run_id)

        evidence: list[BranchEvidence] = []
        decisions: dict[tuple[BranchName, ViewId], object] = {}
        crops: dict[tuple[BranchName, ViewId], np.ndarray] = {}

        template_rows = self._run_templates(config, capture_set, decisions, crops)
        evidence.extend(template_rows)
        if any(row.status is not BranchStatus.PASS for row in template_rows):
            evidence.extend(_skipped_downstream(config, "Template global gate did not pass"))
        else:
            evidence.extend(self._run_downstream(config, capture_set, decisions, crops))

        fused = fuse_inspection(
            capture_set,
            evidence,
            expected_required=_expected_required(config),
        )
        artifacts = PublicationArtifacts(decisions=decisions, crops=crops)
        return self._publish(config, fused.result, fused.triggered_branches, artifacts, run_id)

    def _run_templates(
        self,
        config: LabExperimentConfig,
        capture_set: CaptureSet,
        decisions: dict[tuple[BranchName, ViewId], object],
        crops: dict[tuple[BranchName, ViewId], np.ndarray],
    ) -> tuple[BranchEvidence, ...]:
        required = BranchName.TEMPLATE in config.required_for_ok
        if not config.template.enabled or self.template_backend is None:
            return tuple(
                _skipped(BranchName.TEMPLATE, view_id, required, "Template branch is disabled")
                for view_id in ViewId
            )

        rows = []
        for view_id in ViewId:
            image = capture_set.views[view_id].image
            x1, y1, x2, y2 = config.part_rois[view_id]
            crop = image[y1:y2, x1:x2]
            crops[(BranchName.TEMPLATE, view_id)] = crop
            row = self.template_backend.predict(view_id, crop)
            threshold = config.template.groups[view_id].threshold
            initial_threshold = self.initial_config.template.groups[view_id].threshold
            normalized = _normalize_threshold_row(
                row,
                threshold=threshold,
                initial_threshold=initial_threshold,
                required=required,
            )
            rows.append(normalized)
            _retain_decision(
                self.template_backend,
                BranchName.TEMPLATE,
                view_id,
                normalized,
                decisions,
            )
        return tuple(rows)

    def _run_downstream(
        self,
        config: LabExperimentConfig,
        capture_set: CaptureSet,
        decisions: dict[tuple[BranchName, ViewId], object],
        crops: dict[tuple[BranchName, ViewId], np.ndarray],
    ) -> tuple[BranchEvidence, ...]:
        rows: list[BranchEvidence] = []
        bright_required = BranchName.BRIGHT_STREAK in config.required_for_ok
        for view_id in config.bright_streak.enabled_views:
            if self.bright_streak_backend is None:
                rows.append(_skipped(BranchName.BRIGHT_STREAK, view_id, bright_required, "backend is unavailable"))
                continue
            row = self.bright_streak_backend.predict(view_id, capture_set.views[view_id].image)
            normalized = replace(row, required_for_ok=bright_required)
            rows.append(normalized)
            decision = _retain_decision(
                self.bright_streak_backend,
                BranchName.BRIGHT_STREAK,
                view_id,
                normalized,
                decisions,
            )
            decision_roi = getattr(decision, "roi", None)
            crops[(BranchName.BRIGHT_STREAK, view_id)] = (
                decision_roi if isinstance(decision_roi, np.ndarray) else capture_set.views[view_id].image
            )

        yolo_required = BranchName.YOLO in config.required_for_ok
        if not config.yolo.enabled or self.yolo_backend is None:
            rows.extend(
                _skipped(BranchName.YOLO, view_id, yolo_required, "YOLO branch is disabled")
                for view_id in ViewId
            )
        else:
            for view_id in ViewId:
                row = self.yolo_backend.predict(view_id, capture_set.views[view_id].image)
                normalized = self._normalize_yolo(row, config, yolo_required)
                rows.append(normalized)
                _retain_decision(self.yolo_backend, BranchName.YOLO, view_id, normalized, decisions)
                crops[(BranchName.YOLO, view_id)] = _part_crop(config, capture_set, view_id)

        patch_required = BranchName.PATCHCORE in config.required_for_ok
        if not config.patchcore.enabled or self.patchcore_backend is None:
            rows.extend(
                _skipped(BranchName.PATCHCORE, view_id, patch_required, "PatchCore branch is disabled")
                for view_id in ViewId
            )
        else:
            for view_id in ViewId:
                row = self.patchcore_backend.predict(view_id, capture_set.views[view_id].image)
                normalized = _normalize_threshold_row(
                    row,
                    threshold=config.patchcore.thresholds[view_id],
                    initial_threshold=self.initial_config.patchcore.thresholds[view_id],
                    required=patch_required,
                )
                rows.append(normalized)
                _retain_decision(
                    self.patchcore_backend,
                    BranchName.PATCHCORE,
                    view_id,
                    normalized,
                    decisions,
                )
                crops[(BranchName.PATCHCORE, view_id)] = _part_crop(config, capture_set, view_id)
        return tuple(rows)

    def _normalize_yolo(
        self,
        row: BranchEvidence,
        config: LabExperimentConfig,
        required: bool,
    ) -> BranchEvidence:
        threshold = config.yolo.final_threshold
        initial_threshold = self.initial_config.yolo.final_threshold
        if isinstance(row, YoloEvidence):
            boxes = tuple(box for box in row.candidates if box.confidence >= threshold)
            status = row.status
            reason = row.reason
            if threshold != initial_threshold:
                status = BranchStatus.NG if boxes else BranchStatus.PASS
                reason = (
                    f"hot final threshold {threshold:.6g}: candidates={len(row.candidates)}, "
                    f"final_boxes={len(boxes)} => {status.value}"
                )
            return replace(
                row,
                status=status,
                required_for_ok=required,
                score=max((box.confidence for box in row.candidates), default=None),
                threshold=threshold,
                reason=reason,
                final_boxes=boxes,
            )
        return _normalize_threshold_row(
            row,
            threshold=threshold,
            initial_threshold=initial_threshold,
            required=required,
        )

    def _publish(
        self,
        config: LabExperimentConfig,
        result: InspectionResult,
        triggered: tuple[BranchName, ...],
        artifacts: PublicationArtifacts,
        run_id: str | None,
    ) -> RuntimeOutcome:
        published = self.publisher.publish(
            result,
            config=config,
            triggered_branches=triggered,
            artifacts=artifacts,
            run_id=run_id,
        )
        return RuntimeOutcome(
            RuntimeState.PUBLISHED,
            result,
            published.run_dir,
            triggered,
            result.reason,
        )


def build_lab_runtime(
    config: LabExperimentConfig,
    config_loader: Callable[[], LabExperimentConfig] | None = None,
) -> LabRuntime:
    """Build one resident runtime from all branches enabled by an experiment profile."""
    if not isinstance(config, LabExperimentConfig):
        raise TypeError("config must be LabExperimentConfig")
    if config_loader is None:
        from bmw_inspection.lab.config import load_experiment_config

        def config_loader() -> LabExperimentConfig:
            return load_experiment_config(config.path)
    if not callable(config_loader):
        raise TypeError("config_loader must be callable")

    from bmw_inspection.lab.bright_streak import BrightStreakBackend
    from bmw_inspection.lab.patchcore import PatchCoreBackend
    from bmw_inspection.lab.template import TemplateBackend
    from bmw_inspection.lab.yolo import YoloBackend

    return LabRuntime(
        initial_config=config,
        config_loader=config_loader,
        template_backend=TemplateBackend.from_experiment_config(config) if config.template.enabled else None,
        bright_streak_backend=(
            BrightStreakBackend(
                config.bright_streak,
                required_for_ok=BranchName.BRIGHT_STREAK in config.required_for_ok,
            )
            if config.bright_streak.enabled_views
            else None
        ),
        yolo_backend=YoloBackend.from_experiment_config(config) if config.yolo.enabled else None,
        patchcore_backend=PatchCoreBackend.from_experiment_config(config) if config.patchcore.enabled else None,
        publisher=EvidencePublisher(config.result_root),
    )


def _quality_pass(view_id: ViewId, _image: np.ndarray) -> QualityDecision:
    return QualityDecision(view_id, True, "quality checks passed")


def _validate_capture(capture_set: CaptureSet, config: LabExperimentConfig) -> None:
    expected = (config.capture.image_height, config.capture.image_width)
    for view_id in ViewId:
        image = capture_set.views[view_id].image
        if image.dtype != np.uint8 or image.ndim not in {2, 3} or image.shape[:2] != expected:
            raise ValueError(f"{view_id.value} must be a {expected[1]}x{expected[0]} uint8 image")


def _normalize_threshold_row(
    row: BranchEvidence,
    *,
    threshold: float,
    initial_threshold: float,
    required: bool,
) -> BranchEvidence:
    status = row.status
    reason = row.reason
    if threshold != initial_threshold and status in {BranchStatus.PASS, BranchStatus.NG} and row.score is not None:
        status = BranchStatus.NG if row.score > threshold else BranchStatus.PASS
        reason = f"hot threshold {threshold:.6g}: score {row.score:.6g} => {status.value}"
    return replace(row, status=status, required_for_ok=required, threshold=threshold, reason=reason)


def _retain_decision(
    backend: Any,
    branch: BranchName,
    view_id: ViewId,
    evidence: BranchEvidence,
    decisions: dict[tuple[BranchName, ViewId], object],
) -> object | None:
    decision_for = getattr(backend, "decision_for", None)
    if callable(decision_for):
        decision = decision_for(view_id)
        if decision is not None:
            if is_dataclass(decision) and not isinstance(decision, type):
                names = {field.name for field in fields(decision)}
                if "threshold" in names:
                    decision = replace(decision, threshold=evidence.threshold)
            decisions[(branch, view_id)] = decision
            return decision
    return None


def _part_crop(config: LabExperimentConfig, capture_set: CaptureSet, view_id: ViewId) -> np.ndarray:
    x1, y1, x2, y2 = config.part_rois[view_id]
    return capture_set.views[view_id].image[y1:y2, x1:x2]


def _skipped(branch: BranchName, view_id: ViewId, required: bool, reason: str) -> BranchEvidence:
    return BranchEvidence(branch, view_id, BranchStatus.SKIPPED, required, None, None, 0.0, reason, None, {})


def _skipped_downstream(config: LabExperimentConfig, reason: str) -> tuple[BranchEvidence, ...]:
    rows = [
        *(
            _skipped(
                BranchName.BRIGHT_STREAK,
                view_id,
                BranchName.BRIGHT_STREAK in config.required_for_ok,
                reason,
            )
            for view_id in config.bright_streak.enabled_views
        ),
        *(
            _skipped(BranchName.YOLO, view_id, BranchName.YOLO in config.required_for_ok, reason)
            for view_id in ViewId
        ),
        *(
            _skipped(BranchName.PATCHCORE, view_id, BranchName.PATCHCORE in config.required_for_ok, reason)
            for view_id in ViewId
        ),
    ]
    return tuple(rows)


def _expected_required(config: LabExperimentConfig) -> Mapping[BranchName, frozenset[ViewId]]:
    expected: dict[BranchName, frozenset[ViewId]] = {}
    for branch in config.required_for_ok:
        if branch is BranchName.BRIGHT_STREAK:
            expected[branch] = frozenset(config.bright_streak.enabled_views)
        else:
            expected[branch] = frozenset(ViewId)
    return expected


def _cold_digest(config: LabExperimentConfig) -> str:
    payload = {
        "path": str(config.path),
        "experiment_id": config.experiment_id,
        "topology": _plain(config.topology),
        "capture": _plain(config.capture),
        "part_rois": _plain(config.part_rois),
        "template_models": {
            view_id.value: _template_model_identity(group.model_path)
            for view_id, group in config.template.groups.items()
        },
        "bright_configs": {
            view_id.value: _file_identity(path)
            for view_id, path in config.bright_streak.config_paths.items()
        },
        "yolo_checkpoint": _file_identity(config.yolo.checkpoint),
        "yolo_class": config.yolo.class_name,
        "yolo_candidate_conf": config.yolo.candidate_conf,
        "patchcore_checkpoints": {
            view_id.value: _file_identity(path)
            for view_id, path in config.patchcore.checkpoints.items()
        },
        "result_root": str(config.result_root),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_identity(path: Path | None) -> Mapping[str, str | None]:
    if path is None:
        return {"path": None, "sha256": None}
    resolved = Path(path).expanduser().resolve()
    digest = _sha256(resolved) if resolved.is_file() else None
    return {"path": str(resolved), "sha256": digest}


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _template_model_identity(path: Path | None) -> Mapping[str, object]:
    if path is None:
        return {"path": None, "sha256": None, "templates": []}
    resolved = Path(path).expanduser().resolve()
    identity: dict[str, object] = {"path": str(resolved), "sha256": None}
    dependencies = []
    if resolved.is_file():
        try:
            model_bytes = resolved.read_bytes()
            identity["sha256"] = hashlib.sha256(model_bytes).hexdigest()
            payload = json.loads(model_bytes)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            payload = None
        if isinstance(payload, dict):
            identity["schema_version"] = payload.get("schema_version")
            templates = payload.get("templates")
            if isinstance(templates, list):
                for item in templates:
                    if isinstance(item, dict) and isinstance(item.get("path"), str):
                        dependency = (resolved.parent / item["path"]).resolve()
                        dependencies.append(_file_identity(dependency))
    identity["templates"] = dependencies
    return identity


def _plain(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(_plain(key)): _plain(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((_plain(item) for item in value), key=str)
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value
