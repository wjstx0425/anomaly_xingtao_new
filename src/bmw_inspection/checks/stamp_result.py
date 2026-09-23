# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Stamp reading metadata and optional image evidence persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class StampReadResult:
    """Raw recognition and image evidence in upright ROI coordinates."""

    payload: dict
    roi_original: np.ndarray
    roi_upright: np.ndarray
    overlay: np.ndarray
    line_original: np.ndarray | None = None
    line_clahe: np.ndarray | None = None

    @property
    def normalized_code(self) -> str | None:
        """Return the selected code under the configured normalization policy."""
        return self.payload["normalized_code"]

    @property
    def raw_code(self) -> str | None:
        """Return the selected original reading, including any punctuation."""
        return self.payload["raw_code"]

    @property
    def state(self) -> str:
        """Return readable/review confidence state, never a product verdict."""
        return self.payload["state"]

    @property
    def code_score(self) -> float | None:
        """Return confidence of the selected reading, if available."""
        return self.payload["code_score"]

    @property
    def reasons(self) -> tuple[str, ...]:
        """Return immutable reasons for requesting review."""
        return tuple(self.payload["reasons"])

    def to_dict(self) -> dict:
        """Return independent JSON-ready metadata, excluding evidence images.

        The legacy mutable ``payload`` remains available; consumers should use
        this snapshot when storing or modifying result metadata.
        """
        return json.loads(json.dumps(self.payload, ensure_ascii=False, allow_nan=False))

    def save(self, output: str | Path, *, source: str | None = None) -> None:
        """Write evidence to a new directory, refusing to replace prior results."""
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        images = (("roi_original", self.roi_original), ("roi_upright", self.roi_upright), ("overlay", self.overlay))
        for name, array in images:
            if not cv2.imwrite(str(output / f"{name}.png"), array):
                raise OSError(f"Could not write {name}.png")
        for name, array in (("line_original", self.line_original), ("line_clahe", self.line_clahe)):
            if array is not None and not cv2.imwrite(str(output / f"{name}.png"), array):
                raise OSError(f"Could not write {name}.png")
        payload = self.to_dict()
        payload["source"] = source
        (output / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
