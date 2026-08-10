"""Adapter that reuses the approved BMW bright-streak rule per configured view."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from bmw_inspection.contracts import DemoStatus, load_config
from bmw_inspection.detector import BrightStreakDecision, detect_bright_streak_evidence
from bmw_inspection.lab.config import BrightStreakLabConfig
from bmw_inspection.lab.contracts import BranchEvidence, BranchName, BranchStatus, ViewId


_STATUS_MAP = {
    DemoStatus.OK: BranchStatus.PASS,
    DemoStatus.NG_NO_STREAK: BranchStatus.NG,
    DemoStatus.NG_BROKEN: BranchStatus.NG,
    DemoStatus.ERROR: BranchStatus.ERROR,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class BrightStreakBackend:
    """Run the existing rule only on configured lab views and retain true decision pixels."""

    def __init__(self, config: BrightStreakLabConfig, *, required_for_ok: bool = False) -> None:
        if not isinstance(config, BrightStreakLabConfig):
            raise TypeError("config must be BrightStreakLabConfig")
        if not isinstance(required_for_ok, bool):
            raise TypeError("required_for_ok must be bool")
        self._configs = {
            view_id: load_config(config_path)
            for view_id, config_path in config.config_paths.items()
        }
        self._config_hashes = {
            view_id: _sha256(config_path)
            for view_id, config_path in config.config_paths.items()
        }
        self._required_for_ok = required_for_ok
        self._decisions: dict[ViewId, BrightStreakDecision] = {}

    @property
    def configured_views(self) -> tuple[ViewId, ...]:
        """Return the views whose approved rule configuration is resident."""
        return tuple(self._configs)

    def decision_for(self, view_id: ViewId) -> BrightStreakDecision | None:
        """Return the exact structured decision from the most recent configured prediction."""
        if not isinstance(view_id, ViewId):
            raise TypeError("view_id must be ViewId")
        return self._decisions.get(view_id)

    def predict(self, view_id: ViewId, image: np.ndarray) -> BranchEvidence:
        """Return one truthful branch row; unconfigured views are explicitly skipped."""
        if not isinstance(view_id, ViewId):
            raise TypeError("view_id must be ViewId")
        config = self._configs.get(view_id)
        if config is None:
            return BranchEvidence(
                branch=BranchName.BRIGHT_STREAK,
                view_id=view_id,
                status=BranchStatus.SKIPPED,
                required_for_ok=False,
                score=None,
                threshold=None,
                elapsed_ms=0.0,
                reason="bright-streak is not configured for this view",
                model_id=None,
                artifact_paths={},
            )

        decision = detect_bright_streak_evidence(image, config)
        self._decisions[view_id] = decision
        metrics = decision.metrics
        return BranchEvidence(
            branch=BranchName.BRIGHT_STREAK,
            view_id=view_id,
            status=_STATUS_MAP[decision.status],
            required_for_ok=self._required_for_ok,
            score=metrics.coverage_ratio if metrics is not None else None,
            threshold=config.min_coverage_ratio,
            elapsed_ms=metrics.total_elapsed_ms if metrics is not None else 0.0,
            reason=decision.result.reason,
            model_id=self._config_hashes[view_id],
            artifact_paths={"config": str(config.path)},
        )
