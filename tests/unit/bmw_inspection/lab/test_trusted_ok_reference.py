"""Focused tests for the runtime-only trusted-OK matcher."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER
from bmw_inspection.lab.trusted_ok_reference import TrustedOkMatcher


def _index(tmp_path: Path) -> tuple[Path, np.ndarray]:
    references: list[dict[str, str]] = []
    selected = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.rectangle(selected, (22, 18), (70, 94), (60, 180, 240), -1)
    cv2.circle(selected, (96, 40), 9, (255, 255, 255), -1)
    for view in VIEW_ORDER:
        image_path = tmp_path / f"{view}.png"
        assert cv2.imwrite(str(image_path), selected)
        references.append(
            {
                "view_id": view,
                "physical_part_id": "part-01",
                "sample_id": "sample-01",
                "full_image_path": image_path.name,
                "roi_image_path": image_path.name,
                # Old publisher metadata is deliberately ignored by runtime.
                "source_sha256": "stale",
                "whitelist_sha256": "stale",
            }
        )
    index_path = tmp_path / "reference_index.json"
    index_path.write_text(json.dumps({"references": references}), encoding="utf-8")
    return index_path, selected


def test_matcher_reads_paths_without_sha_binding(tmp_path: Path) -> None:
    index_path, selected = _index(tmp_path)
    matcher = TrustedOkMatcher(index_path)
    matcher.preload()

    match = matcher.match("front", selected, selected, comparison_mode="roi")

    assert match.physical_part_id == "part-01"
    assert match.sample_id == "sample-01"
    assert match.similarity > 0.99
    assert match.difference_overlay.shape == (512, 512, 3)


def test_matcher_keeps_required_file_and_eight_view_checks(tmp_path: Path) -> None:
    index_path, _selected = _index(tmp_path)
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload["references"] = payload["references"][:-1]
    index_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="缺少视角"):
        TrustedOkMatcher(index_path)


def test_match_requires_preload(tmp_path: Path) -> None:
    index_path, selected = _index(tmp_path)
    matcher = TrustedOkMatcher(index_path)

    with pytest.raises(RuntimeError, match="preload"):
        matcher.match("front", selected, selected, comparison_mode="roi")
