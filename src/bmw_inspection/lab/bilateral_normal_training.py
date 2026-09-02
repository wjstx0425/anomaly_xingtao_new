# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Sequential right/left BMW Template and all-normal EfficientAD training."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bmw_inspection.lab.left_normal_training import LeftNormalTrainingConfig, run_left_normal_training


@dataclass(frozen=True, slots=True)
class BilateralNormalTrainingConfig:
    """Inputs and immutable output identities for one bilateral run."""

    repo_root: Path
    right_prepared_root: Path
    right_roi_config: Path
    left_prepared_root: Path
    left_roi_config: Path
    training_root: Path
    output_root: Path
    right_training_id: str
    right_run_id: str
    left_training_id: str
    left_run_id: str
    efficientad_epochs: int = 30
    gpu: int = 0
    workers: int = 8
    seed: int = 42

    def __post_init__(self) -> None:
        for field in (
            "repo_root",
            "right_prepared_root",
            "right_roi_config",
            "left_prepared_root",
            "left_roi_config",
            "training_root",
            "output_root",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)).expanduser().resolve())


HandRunner = Callable[..., dict[str, Any]]


def build_hand_configs(
    config: BilateralNormalTrainingConfig,
) -> tuple[LeftNormalTrainingConfig, LeftNormalTrainingConfig]:
    """Build right and left checkpoint-only configurations in execution order."""

    common = {
        "repo_root": config.repo_root,
        "training_root": config.training_root,
        "output_root": config.output_root,
        "efficientad_epochs": config.efficientad_epochs,
        "efficientad_all_normal_train": True,
        "gpu": config.gpu,
        "workers": config.workers,
        "seed": config.seed,
    }
    right = LeftNormalTrainingConfig(
        prepared_root=config.right_prepared_root,
        roi_config=config.right_roi_config,
        training_id=config.right_training_id,
        run_id=config.right_run_id,
        capture_scope="right",
        **common,
    )
    left = LeftNormalTrainingConfig(
        prepared_root=config.left_prepared_root,
        roi_config=config.left_roi_config,
        training_id=config.left_training_id,
        run_id=config.left_run_id,
        capture_scope="left",
        **common,
    )
    return right, left


def run_bilateral_normal_training(
    config: BilateralNormalTrainingConfig,
    *,
    dry_run: bool = False,
    runner: HandRunner = run_left_normal_training,
) -> dict[str, Any]:
    """Train right then left and stop immediately if either hand fails."""

    reports: list[dict[str, Any]] = []
    for hand_config in build_hand_configs(config):
        hand_report = runner(hand_config, dry_run=dry_run, stage="train")
        reports.append({"capture_scope": hand_config.capture_scope, "report": hand_report})
    return {
        "status": "dry_run" if dry_run else "complete",
        "efficientad_all_normal_train": True,
        "validation_status": "pending_external_validation",
        "hands": reports,
    }
