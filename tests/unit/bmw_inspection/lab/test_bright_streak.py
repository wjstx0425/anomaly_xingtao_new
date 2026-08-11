"""Bright-streak adapter tests for the BMW six-view laboratory workflow."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from bmw_inspection.contracts import DemoStatus, load_config
from bmw_inspection.detector import detect_bright_streak_evidence
from bmw_inspection.lab.bright_streak import BrightStreakBackend
from bmw_inspection.lab.config import BrightStreakLabConfig
from bmw_inspection.lab.contracts import BranchName, BranchStatus, ViewId


REPO_ROOT = Path(__file__).resolve().parents[4]
DEMO_CONFIG = REPO_ROOT / "configs/bmw/bright_streak_demo.json"
OK_IMAGE = REPO_ROOT / "dataset/bmw/OK/Image_20260805172921398.bmp"


def _image() -> np.ndarray:
    image = cv2.imread(str(OK_IMAGE), cv2.IMREAD_UNCHANGED)
    assert image is not None
    return image


def test_front_left_uses_the_approved_demo_rule_and_exposes_its_decision() -> None:
    image = _image()
    config = load_config(DEMO_CONFIG)
    expected = detect_bright_streak_evidence(image, config)
    backend = BrightStreakBackend(
        BrightStreakLabConfig(config_paths={ViewId.FRONT_LEFT: DEMO_CONFIG}),
    )

    evidence = backend.predict(ViewId.FRONT_LEFT, image)
    decision = backend.decision_for(ViewId.FRONT_LEFT)

    assert expected.result.status is DemoStatus.OK
    assert evidence.branch is BranchName.BRIGHT_STREAK
    assert evidence.view_id is ViewId.FRONT_LEFT
    assert evidence.status is BranchStatus.PASS
    assert evidence.required_for_ok is False
    assert evidence.score == expected.result.metrics.coverage_ratio
    assert evidence.threshold == config.min_coverage_ratio
    assert evidence.model_id == hashlib.sha256(DEMO_CONFIG.read_bytes()).hexdigest()
    assert evidence.artifact_paths["config"] == str(DEMO_CONFIG.resolve())
    assert decision is not None
    assert np.array_equal(decision.response, expected.response)
    assert np.array_equal(decision.mask, expected.mask)
    assert decision.runs == expected.runs
    assert decision.gaps == expected.gaps


def test_unconfigured_view_is_skipped_without_running_the_rule() -> None:
    backend = BrightStreakBackend(
        BrightStreakLabConfig(config_paths={ViewId.FRONT_LEFT: DEMO_CONFIG}),
        required_for_ok=True,
    )

    evidence = backend.predict(ViewId.BACK, _image())

    assert evidence.branch is BranchName.BRIGHT_STREAK
    assert evidence.view_id is ViewId.BACK
    assert evidence.status is BranchStatus.SKIPPED
    assert evidence.required_for_ok is False
    assert evidence.score is None
    assert evidence.threshold is None
    assert evidence.model_id is None
    assert "required_for_ok" not in evidence.reason
    assert backend.decision_for(ViewId.BACK) is None


def test_only_configured_views_can_be_required_for_ok() -> None:
    backend = BrightStreakBackend(
        BrightStreakLabConfig(config_paths={ViewId.FRONT_LEFT: DEMO_CONFIG}),
        required_for_ok=True,
    )

    configured = backend.predict(ViewId.FRONT_LEFT, _image())
    unconfigured = [
        backend.predict(view_id, _image())
        for view_id in ViewId
        if view_id is not ViewId.FRONT_LEFT
    ]

    assert configured.required_for_ok is True
    assert len(unconfigured) == 5
    assert all(evidence.status is BranchStatus.SKIPPED for evidence in unconfigured)
    assert all(evidence.required_for_ok is False for evidence in unconfigured)
