from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from zs32_inspection.dashboard.contracts import MODELED_VIEWS, VIEW_ORDER

IDENTITY = {
    "part_id": "ZS32-0001",
    "capture_session": "capture-0001",
    "group_id": "group-0001",
    "hand": "right",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _branch(
    result_dir: Path,
    view: str,
    branch: str,
    *,
    state: str = "available",
    status: str = "REVIEW",
    score: float | None = 0.25,
) -> dict[str, Any]:
    evidence_path = result_dir / "evidence" / branch / f"{view}.png"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(evidence_path), np.zeros((8, 10, 3), dtype=np.uint8))
    record: dict[str, Any] = {
        "state": state,
        "status": status,
        "score": score,
        "reason": "",
        "evidence_path": str(evidence_path.relative_to(result_dir)),
    }
    if branch == "patchcore":
        mask_path = result_dir / "evidence" / branch / "masks" / f"{view}.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(mask_path), np.zeros((10, 20), dtype=np.uint8))
        record.update(
            mask_path=str(mask_path.relative_to(result_dir)),
            mask_source="pred_mask",
            roi_xyxy=[5, 5, 25, 15],
        )
    if branch == "yolo":
        record["detections"] = [{"xyxy": [1, 2, 8, 9], "class_name": "scratch", "confidence": 0.25}]
    return record


def write_manifest(result_dir: Path, manifest: dict[str, Any]) -> None:
    (result_dir / "runtime_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_manifest(result_dir: Path) -> dict[str, Any]:
    return json.loads((result_dir / "runtime_manifest.json").read_text(encoding="utf-8"))


def rewrite_view(result_dir: Path, view: str, **updates: Any) -> None:
    manifest = load_manifest(result_dir)
    manifest["views"][view].update(updates)
    write_manifest(result_dir, manifest)


@pytest.fixture
def eight_view_result_dir(tmp_path: Path) -> Path:
    result_dir = tmp_path / "result"
    source_dir = result_dir / "sources"
    source_dir.mkdir(parents=True)
    views: dict[str, dict[str, Any]] = {}
    for index, view in enumerate(VIEW_ORDER):
        source_path = source_dir / f"{view}.png"
        image = np.full((30, 40, 3), index * 20, dtype=np.uint8)
        assert cv2.imwrite(str(source_path), image)
        modeled = view in MODELED_VIEWS
        branches = (
            {branch: _branch(result_dir, view, branch) for branch in ("template", "patchcore", "yolo", "fusion")}
            if modeled
            else {
                branch: {
                    "state": "unsupported",
                    "status": "UNSUPPORTED",
                    "score": None,
                    "reason": "model assets are not commissioned",
                }
                for branch in ("template", "patchcore", "yolo", "fusion")
            }
        )
        views[view] = {
            "view": view,
            **IDENTITY,
            "manifest_identity": f"{IDENTITY['part_id']}:right:{view}",
            "source_path": str(source_path.relative_to(result_dir)),
            "source_sha256": _sha256(source_path),
            "source_shape": [30, 40],
            "model_supported": modeled,
            "camera": f"camera-{index // 2}",
            "branches": branches,
        }
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "product": "ZS32",
        **IDENTITY,
        "manifest_identity": "ZS32/right",
        "machine_status": "REVIEW",
        "reason": "thresholds are not locked",
        "views": views,
    }
    write_manifest(result_dir, manifest)
    return result_dir
