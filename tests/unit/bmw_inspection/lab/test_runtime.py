# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Ordered Template-first runtime tests for BMW laboratory inspection."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from bmw_inspection.lab.config import TemplateConfig, TemplateGroupConfig, load_experiment_config
from bmw_inspection.lab.contracts import (
    BranchEvidence,
    BranchName,
    BranchStatus,
    CaptureSet,
    CapturedView,
    FinalStatus,
    ViewId,
)
from bmw_inspection.lab.publisher import EvidencePublisher
from bmw_inspection.lab.runtime import LabRuntime, QualityDecision, RuntimeState


REPO_ROOT = Path(__file__).parents[4]


@dataclass(frozen=True, slots=True)
class _ThresholdDecision:
    view_id: ViewId
    threshold: float


class _Backend:
    def __init__(self, branch: BranchName, statuses: dict[ViewId, BranchStatus] | None = None) -> None:
        self.branch = branch
        self.statuses = statuses or {}
        self.calls: list[ViewId] = []

    def predict(self, view_id: ViewId, _image: np.ndarray) -> BranchEvidence:
        self.calls.append(view_id)
        status = self.statuses.get(view_id, BranchStatus.PASS)
        return BranchEvidence(
            branch=self.branch,
            view_id=view_id,
            status=status,
            required_for_ok=True,
            score=0.3,
            threshold=0.5,
            elapsed_ms=1.0,
            reason=f"{self.branch.value}/{status.value}",
            model_id=f"{self.branch.value}-resident",
            artifact_paths={},
        )

    def decision_for(self, view_id: ViewId) -> _ThresholdDecision:
        return _ThresholdDecision(view_id, 0.5)


def _capture_set() -> CaptureSet:
    now = datetime.now()
    return CaptureSet(
        capture_set_id="capture-runtime",
        views={
            view_id: CapturedView(view_id, f"serial-{view_id.value}", np.full((8, 10), 40, np.uint8), now)
            for view_id in ViewId
        },
        created_at=now,
    )


def _config(tmp_path: Path):
    loaded = load_experiment_config(REPO_ROOT / "configs/bmw/experiments/bmw_lab_v1.json")
    groups = {
        view_id: TemplateGroupConfig(view_id, None, 0.5)
        for view_id in ViewId
    }
    return replace(
        loaded,
        capture=replace(loaded.capture, image_width=10, image_height=8),
        part_rois={view_id: (0, 0, 10, 8) for view_id in ViewId},
        template=TemplateConfig(enabled=True, groups=groups),
        yolo=replace(loaded.yolo, enabled=True),
        patchcore=replace(loaded.patchcore, enabled=True),
        result_root=tmp_path / "results",
    )


def _runtime(
    tmp_path: Path,
    *,
    template_statuses: dict[ViewId, BranchStatus] | None = None,
    yolo_statuses: dict[ViewId, BranchStatus] | None = None,
    patch_statuses: dict[ViewId, BranchStatus] | None = None,
    quality_checker=None,
    config_loader=None,
):
    config = _config(tmp_path)
    template = _Backend(BranchName.TEMPLATE, template_statuses)
    bright = _Backend(BranchName.BRIGHT_STREAK)
    bright.configured_views = (ViewId.FRONT_LEFT,)
    yolo = _Backend(BranchName.YOLO, yolo_statuses)
    patch = _Backend(BranchName.PATCHCORE, patch_statuses)
    runtime = LabRuntime(
        initial_config=config,
        config_loader=config_loader or (lambda: config),
        template_backend=template,
        bright_streak_backend=bright,
        yolo_backend=yolo,
        patchcore_backend=patch,
        publisher=EvidencePublisher(config.result_root),
        quality_checker=quality_checker,
    )
    return runtime, template, bright, yolo, patch, config


@pytest.mark.parametrize("blocking", [BranchStatus.NG, BranchStatus.ERROR, BranchStatus.REVIEW])
def test_any_nonpass_template_short_circuits_every_downstream_backend(
    tmp_path: Path,
    blocking: BranchStatus,
) -> None:
    runtime, template, bright, yolo, patch, _config_snapshot = _runtime(
        tmp_path,
        template_statuses={ViewId.BACK_RIGHT: blocking},
    )

    outcome = runtime.inspect(_capture_set(), run_id=f"template-{blocking.value.lower()}")

    assert outcome.state is RuntimeState.PUBLISHED
    assert len(template.calls) == 6
    assert bright.calls == yolo.calls == patch.calls == []
    assert outcome.inspection is not None
    if blocking is BranchStatus.NG:
        assert outcome.inspection.final_status is FinalStatus.NG_TEMPLATE
    elif blocking is BranchStatus.ERROR:
        assert outcome.inspection.final_status is FinalStatus.ERROR
    else:
        assert outcome.inspection.final_status is FinalStatus.REVIEW
    downstream = [row for row in outcome.inspection.evidence if row.branch is not BranchName.TEMPLATE]
    assert downstream and all(row.status is BranchStatus.SKIPPED for row in downstream)


def test_retake_quality_precedes_template_and_all_models(tmp_path: Path) -> None:
    def quality(view_id: ViewId, _image: np.ndarray) -> QualityDecision:
        return QualityDecision(view_id, view_id is not ViewId.FRONT, "front blurred")

    runtime, template, bright, yolo, patch, _config_snapshot = _runtime(tmp_path, quality_checker=quality)

    outcome = runtime.inspect(_capture_set(), run_id="retake")

    assert outcome.inspection is not None
    assert outcome.inspection.final_status is FinalStatus.RETAKE
    assert template.calls == bright.calls == yolo.calls == patch.calls == []


def test_downstream_runs_after_six_template_passes_and_records_multiple_ng(tmp_path: Path) -> None:
    runtime, template, bright, yolo, patch, _config_snapshot = _runtime(
        tmp_path,
        yolo_statuses={ViewId.BACK: BranchStatus.NG},
        patch_statuses={ViewId.BACK_RIGHT: BranchStatus.NG},
    )
    bright.statuses[ViewId.FRONT_LEFT] = BranchStatus.NG

    outcome = runtime.inspect(_capture_set(), run_id="multiple-ng")

    assert len(template.calls) == 6
    assert bright.calls == [ViewId.FRONT_LEFT]
    assert len(yolo.calls) == len(patch.calls) == 6
    assert outcome.inspection is not None
    assert outcome.inspection.final_status is FinalStatus.NG_BRIGHT_STREAK
    assert outcome.triggered_branches == (
        BranchName.BRIGHT_STREAK,
        BranchName.YOLO,
        BranchName.PATCHCORE,
    )
    assert outcome.run_dir is not None
    assert (outcome.run_dir / "crops/bright_streak/front_left.png").is_file()
    assert (outcome.run_dir / "crops/yolo/front.png").is_file()
    assert (outcome.run_dir / "crops/patchcore/front.png").is_file()


def test_threshold_enabled_and_required_changes_are_hot_reloaded(tmp_path: Path) -> None:
    current = [_config(tmp_path)]
    runtime, template, _bright, _yolo, _patch, _initial = _runtime(
        tmp_path,
        config_loader=lambda: current[0],
    )
    changed_groups = {
        view_id: replace(group, threshold=0.2)
        for view_id, group in current[0].template.groups.items()
    }
    current[0] = replace(
        current[0],
        template=TemplateConfig(enabled=True, groups=changed_groups),
        yolo=replace(current[0].yolo, enabled=False, final_threshold=0.7),
        required_for_ok=frozenset({BranchName.TEMPLATE, BranchName.PATCHCORE}),
    )

    outcome = runtime.inspect(_capture_set(), run_id="hot-reload")

    assert outcome.state is RuntimeState.PUBLISHED
    assert len(template.calls) == 6
    assert outcome.inspection is not None
    template_rows = [row for row in outcome.inspection.evidence if row.branch is BranchName.TEMPLATE]
    assert all(row.threshold == 0.2 and row.required_for_ok for row in template_rows)
    assert outcome.run_dir is not None
    decision = json.loads(
        (outcome.run_dir / "evidence/template/front/decision.json").read_text(encoding="utf-8")
    )
    assert decision["threshold"] == 0.2
    yolo_rows = [row for row in outcome.inspection.evidence if row.branch is BranchName.YOLO]
    assert yolo_rows and all(row.status is BranchStatus.SKIPPED and not row.required_for_ok for row in yolo_rows)


@pytest.mark.parametrize("change", ["roi", "checkpoint", "preprocess"])
def test_cold_identity_change_returns_reload_required_without_model_calls(
    tmp_path: Path,
    change: str,
) -> None:
    config = _config(tmp_path)
    model_path = tmp_path / "template-model.json"
    model_path.write_text("model-v1", encoding="utf-8")
    groups = {
        view_id: replace(group, model_path=model_path)
        for view_id, group in config.template.groups.items()
    }
    checkpoint = tmp_path / "yolo-v1.pt"
    checkpoint.write_bytes(b"v1")
    config = replace(config, template=TemplateConfig(True, groups), yolo=replace(config.yolo, checkpoint=checkpoint))
    current = [config]
    runtime, template, bright, yolo, patch, _initial = _runtime(
        tmp_path,
        config_loader=lambda: current[0],
    )
    runtime = replace_runtime_initial(runtime, config)

    if change == "roi":
        rois = dict(config.part_rois)
        rois[ViewId.FRONT] = (1, 0, 10, 8)
        current[0] = replace(config, part_rois=rois)
    elif change == "checkpoint":
        replacement = tmp_path / "yolo-v2.pt"
        replacement.write_bytes(b"v2")
        current[0] = replace(config, yolo=replace(config.yolo, checkpoint=replacement))
    else:
        model_path.write_text("model-v2", encoding="utf-8")

    outcome = runtime.inspect(_capture_set(), run_id=f"reload-{change}")

    assert outcome.state is RuntimeState.RELOAD_REQUIRED
    assert outcome.inspection is None
    assert outcome.run_dir is None
    assert template.calls == bright.calls == yolo.calls == patch.calls == []


def test_candidate_threshold_change_requires_reload_without_model_calls(tmp_path: Path) -> None:
    config = _config(tmp_path)
    current = [config]
    runtime, template, bright, yolo, patch, _initial = _runtime(
        tmp_path,
        config_loader=lambda: current[0],
    )
    current[0] = replace(config, yolo=replace(config.yolo, candidate_conf=config.yolo.candidate_conf + 0.01))

    outcome = runtime.inspect(_capture_set(), run_id="candidate-reload")

    assert outcome.state is RuntimeState.RELOAD_REQUIRED
    assert template.calls == bright.calls == yolo.calls == patch.calls == []


def test_template_dependency_bytes_are_hashed_even_when_stat_identity_is_preserved(tmp_path: Path) -> None:
    config = _config(tmp_path)
    template_image = tmp_path / "template.png"
    template_image.write_bytes(b"AAAA")
    original_stat = template_image.stat()
    model_path = tmp_path / "model.json"
    model_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "templates": [{"path": template_image.name, "sha256": "manifest-value"}],
            }
        ),
        encoding="utf-8",
    )
    groups = {
        view_id: replace(group, model_path=model_path)
        for view_id, group in config.template.groups.items()
    }
    config = replace(config, template=TemplateConfig(True, groups))
    current = [config]
    runtime, template, bright, yolo, patch, _initial = _runtime(
        tmp_path,
        config_loader=lambda: current[0],
    )
    runtime = replace_runtime_initial(runtime, config)

    template_image.write_bytes(b"BBBB")
    os.utime(template_image, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    outcome = runtime.inspect(_capture_set(), run_id="template-dependency-reload")

    assert outcome.state is RuntimeState.RELOAD_REQUIRED
    assert template.calls == bright.calls == yolo.calls == patch.calls == []


def replace_runtime_initial(runtime: LabRuntime, config) -> LabRuntime:
    """Recreate only the runtime identity baseline while retaining its injected fakes."""
    return LabRuntime(
        initial_config=config,
        config_loader=runtime.config_loader,
        template_backend=runtime.template_backend,
        bright_streak_backend=runtime.bright_streak_backend,
        yolo_backend=runtime.yolo_backend,
        patchcore_backend=runtime.patchcore_backend,
        publisher=runtime.publisher,
        quality_checker=runtime.quality_checker,
    )
