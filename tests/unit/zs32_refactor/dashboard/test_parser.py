from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from zs32_inspection.dashboard.contracts import VIEW_ORDER, BranchState
from zs32_inspection.dashboard.parser import DashboardResultError, load_inspection_result


def load_manifest(result_dir: Path) -> dict[str, object]:
    return json.loads((result_dir / "runtime_manifest.json").read_text(encoding="utf-8"))


def write_manifest(result_dir: Path, manifest: dict[str, object]) -> None:
    (result_dir / "runtime_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def rewrite_view(result_dir: Path, view: str, **updates: object) -> None:
    manifest = load_manifest(result_dir)
    manifest["views"][view].update(updates)  # type: ignore[index,union-attr]
    write_manifest(result_dir, manifest)


def test_parser_returns_eight_views_in_topology_order(eight_view_result_dir: Path) -> None:
    result = load_inspection_result(eight_view_result_dir)

    assert tuple(view.view for view in result.views) == VIEW_ORDER
    assert all(view.model_supported is True for view in result.views)
    assert all(branch.state is BranchState.AVAILABLE for view in result.views for branch in view.branches.values())
    assert result.identity.hand == "right"


def test_parser_uses_topology_order_not_manifest_mapping_order(eight_view_result_dir: Path) -> None:
    manifest = load_manifest(eight_view_result_dir)
    manifest["views"] = dict(reversed(tuple(manifest["views"].items())))  # type: ignore[index,union-attr]
    write_manifest(eight_view_result_dir, manifest)

    result = load_inspection_result(eight_view_result_dir)

    assert tuple(view.view for view in result.views) == VIEW_ORDER


def test_parser_maps_task2_direct_view_branch_fields(task2_direct_result_dir: Path) -> None:
    result = load_inspection_result(task2_direct_result_dir)

    assert set(result.views[0].branches) == {"template", "patchcore", "yolo", "fusion"}
    assert result.views[0].branches["template"].state is BranchState.AVAILABLE
    assert result.views[0].branches["template"].status == "NG_TEMPLATE"
    assert result.views[1].branches["template"].state is BranchState.AVAILABLE
    assert result.views[1].branches["template"].status == "REVIEW"
    assert result.views[4].branches["patchcore"].state is BranchState.ERROR
    assert result.views[0].branches["fusion"].state is BranchState.SKIPPED
    assert result.views[0].branches["fusion"].score == pytest.approx(0.42)
    assert all(branch.state is BranchState.AVAILABLE for branch in result.views[3].branches.values())


@pytest.mark.parametrize("view", ["front", "front_secondary"])
@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_parser_requires_exact_core_branches(eight_view_result_dir: Path, view: str, mutation: str) -> None:
    manifest = load_manifest(eight_view_result_dir)
    branches = manifest["views"][view]["branches"]
    if mutation == "missing":
        branches.pop("fusion")
    else:
        branches["geometry"] = branches["template"].copy()
    write_manifest(eight_view_result_dir, manifest)

    with pytest.raises(DashboardResultError, match="branches"):
        load_inspection_result(eight_view_result_dir)


def test_parser_preserves_nonzero_skipped_score_after_template_stop(eight_view_result_dir: Path) -> None:
    manifest = load_manifest(eight_view_result_dir)
    branches = manifest["views"]["front_left"]["branches"]
    branches["template"].update(status="NG_TEMPLATE", score=0.91)
    branches["patchcore"].update(state="skipped", status="SKIPPED", score=0.42)
    write_manifest(eight_view_result_dir, manifest)

    result = load_inspection_result(eight_view_result_dir)

    assert result.views[1].branches["template"].status == "NG_TEMPLATE"
    assert result.views[1].branches["patchcore"].state is BranchState.SKIPPED
    assert result.views[1].branches["patchcore"].score == pytest.approx(0.42)


def test_parser_accepts_secondary_available_evidence(eight_view_result_dir: Path) -> None:
    manifest = load_manifest(eight_view_result_dir)
    source = manifest["views"]["front"]["branches"]["yolo"].copy()
    source["evidence_path"] = manifest["views"]["front_secondary"]["source_path"]
    manifest["views"]["front_secondary"]["branches"]["yolo"] = source
    write_manifest(eight_view_result_dir, manifest)

    result = load_inspection_result(eight_view_result_dir)

    branch = result.views[3].branches["yolo"]
    assert result.views[3].model_supported is True
    assert branch.state is BranchState.AVAILABLE
    assert branch.score == pytest.approx(0.25)


@pytest.mark.parametrize("view", VIEW_ORDER)
def test_parser_rejects_unsupported_core_branch(eight_view_result_dir: Path, view: str) -> None:
    manifest = load_manifest(eight_view_result_dir)
    manifest["views"][view]["branches"]["patchcore"].update(
        state="unsupported",
        status="UNSUPPORTED",
        score=None,
        reason="not commissioned",
    )
    write_manifest(eight_view_result_dir, manifest)

    with pytest.raises(DashboardResultError, match="UNSUPPORTED"):
        load_inspection_result(eight_view_result_dir)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing", "views"),
        ("extra", "views"),
        ("duplicate", "view"),
        ("wrong_semantic_view", "view"),
    ],
)
def test_parser_rejects_invalid_view_topology(
    eight_view_result_dir: Path,
    mutation: str,
    match: str,
) -> None:
    manifest = load_manifest(eight_view_result_dir)
    if mutation == "missing":
        manifest["views"].pop("back_secondary")
    elif mutation == "extra":
        manifest["views"]["left"] = manifest["views"]["front"].copy()
    elif mutation == "duplicate":
        manifest["views"]["back_secondary"]["view"] = "back_right"
    else:
        manifest["views"]["front"]["view"] = "back"
    write_manifest(eight_view_result_dir, manifest)

    with pytest.raises(DashboardResultError, match=match):
        load_inspection_result(eight_view_result_dir)


@pytest.mark.parametrize(
    ("scope", "field", "value"),
    [
        ("top", "product", "OTHER"),
        ("top", "hand", "left"),
        ("view", "part_id", "other-part"),
        ("view", "capture_session", "other-session"),
        ("view", "group_id", "other-group"),
        ("view", "hand", "left"),
        ("view", "manifest_identity", "wrong:right:front"),
    ],
)
def test_parser_rejects_mixed_or_wrong_identity(
    eight_view_result_dir: Path,
    scope: str,
    field: str,
    value: str,
) -> None:
    manifest = load_manifest(eight_view_result_dir)
    target = manifest if scope == "top" else manifest["views"]["front"]
    target[field] = value
    write_manifest(eight_view_result_dir, manifest)

    with pytest.raises(DashboardResultError, match=field):
        load_inspection_result(eight_view_result_dir)


@pytest.mark.parametrize("escaped_path", ["../outside.png", "/tmp/outside.png"])
def test_parser_rejects_source_path_escape(eight_view_result_dir: Path, escaped_path: str) -> None:
    rewrite_view(eight_view_result_dir, "front", source_path=escaped_path)

    with pytest.raises(DashboardResultError, match="source_path"):
        load_inspection_result(eight_view_result_dir)


def test_parser_rejects_source_symlink_escape(eight_view_result_dir: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.png"
    assert cv2.imwrite(str(outside), np.zeros((30, 40, 3), dtype=np.uint8))
    link = eight_view_result_dir / "sources" / "escaped.png"
    link.symlink_to(outside)
    rewrite_view(eight_view_result_dir, "front", source_path="sources/escaped.png")

    with pytest.raises(DashboardResultError, match="source_path"):
        load_inspection_result(eight_view_result_dir)


def test_parser_wraps_source_path_resolve_failure(
    eight_view_result_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_resolve = Path.resolve

    def fail_source(path: Path, *args: object, **kwargs: object) -> Path:
        if path.name == "front.png" and path.parent.name == "sources":
            raise OSError("source resolve failed")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", fail_source)

    with pytest.raises(DashboardResultError, match="source_path.*resolve failed"):
        load_inspection_result(eight_view_result_dir)


def test_parser_wraps_source_read_failure(
    eight_view_result_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_read_bytes = Path.read_bytes

    def fail_source_read(path: Path) -> bytes:
        if path.name == "front.png" and path.parent.name == "sources":
            raise PermissionError("source became unreadable")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_source_read)

    with pytest.raises(DashboardResultError, match="source_path.*read.*source became unreadable"):
        load_inspection_result(eight_view_result_dir)


def test_parser_rejects_source_hash_mismatch(eight_view_result_dir: Path) -> None:
    rewrite_view(eight_view_result_dir, "front", source_sha256="0" * 64)

    with pytest.raises(DashboardResultError, match="source_sha256"):
        load_inspection_result(eight_view_result_dir)


def test_parser_rejects_invalid_source_image(eight_view_result_dir: Path) -> None:
    source = eight_view_result_dir / "sources" / "front.png"
    source.write_bytes(b"not an image")
    rewrite_view(eight_view_result_dir, "front", source_sha256=_sha256(source))

    with pytest.raises(DashboardResultError, match="decode"):
        load_inspection_result(eight_view_result_dir)


def test_parser_rejects_source_shape_mismatch(eight_view_result_dir: Path) -> None:
    rewrite_view(eight_view_result_dir, "front", source_shape=[31, 40])

    with pytest.raises(DashboardResultError, match="source_shape"):
        load_inspection_result(eight_view_result_dir)


def test_parser_rejects_one_source_reused_for_two_views(eight_view_result_dir: Path) -> None:
    manifest = load_manifest(eight_view_result_dir)
    front = manifest["views"]["front"]  # type: ignore[index]
    back = manifest["views"]["back"]  # type: ignore[index]
    back["source_path"] = front["source_path"]
    back["source_sha256"] = front["source_sha256"]
    write_manifest(eight_view_result_dir, manifest)

    with pytest.raises(DashboardResultError, match="distinct source"):
        load_inspection_result(eight_view_result_dir)


@pytest.mark.parametrize("failure", ["missing", "geometry", "nonbinary", "escape", "raw_escape"])
def test_parser_turns_mask_artifact_failure_into_branch_error(
    eight_view_result_dir: Path,
    failure: str,
) -> None:
    manifest = load_manifest(eight_view_result_dir)
    branch = manifest["views"]["front"]["branches"]["patchcore"]
    mask_path = eight_view_result_dir / branch["mask_path"]
    if failure == "missing":
        mask_path.unlink()
    elif failure == "geometry":
        assert cv2.imwrite(str(mask_path), np.zeros((9, 20), dtype=np.uint8))
    elif failure == "nonbinary":
        assert cv2.imwrite(str(mask_path), np.full((10, 20), 127, dtype=np.uint8))
    elif failure == "raw_escape":
        branch["raw_anomaly_map_path"] = "../../escaped-map.npy"
    else:
        branch["mask_path"] = "../../escaped-mask.png"
    write_manifest(eight_view_result_dir, manifest)

    result = load_inspection_result(eight_view_result_dir)

    view = result.views[0]
    assert view.source_path.is_file()
    assert view.branches["patchcore"].state is BranchState.ERROR
    assert view.branches["patchcore"].score == pytest.approx(0.25)
    assert view.branches["patchcore"].reason


@pytest.mark.parametrize("path_field", ["evidence_path", "mask_path", "raw_anomaly_map_path"])
def test_parser_turns_branch_path_resolve_failure_into_branch_error(
    eight_view_result_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    path_field: str,
) -> None:
    manifest = load_manifest(eight_view_result_dir)
    branch = manifest["views"]["front"]["branches"]["patchcore"]
    if path_field == "raw_anomaly_map_path":
        raw_path = eight_view_result_dir / "evidence" / "patchcore" / "raw_maps" / "front.npy"
        raw_path.parent.mkdir(parents=True)
        raw_path.write_bytes(b"raw")
        branch[path_field] = str(raw_path.relative_to(eight_view_result_dir))
    target = eight_view_result_dir / branch[path_field]
    write_manifest(eight_view_result_dir, manifest)
    original_resolve = Path.resolve

    def fail_branch(path: Path, *args: object, **kwargs: object) -> Path:
        if path == target:
            raise RuntimeError("symlink loop")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", fail_branch)

    result = load_inspection_result(eight_view_result_dir)

    assert result.views[0].branches["patchcore"].state is BranchState.ERROR
    assert "symlink loop" in result.views[0].branches["patchcore"].reason


@pytest.mark.parametrize("duplicate", ["top", "view", "branch"])
def test_parser_rejects_duplicate_json_object_keys(eight_view_result_dir: Path, duplicate: str) -> None:
    path = eight_view_result_dir / "runtime_manifest.json"
    text = path.read_text(encoding="utf-8")
    if duplicate == "top":
        text = text.replace("{", '{\n  "product": "ZS32",', 1)
    elif duplicate == "view":
        text = text.replace('"view": "front",', '"view": "front",\n      "view": "front",', 1)
    else:
        text = text.replace('"score": 0.25,', '"score": 0.25,\n          "score": 0.25,', 1)
    path.write_text(text, encoding="utf-8")

    with pytest.raises(DashboardResultError, match="duplicate.*key"):
        load_inspection_result(eight_view_result_dir)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_parser_rejects_non_finite_json_scores(eight_view_result_dir: Path, constant: str) -> None:
    path = eight_view_result_dir / "runtime_manifest.json"
    text = path.read_text(encoding="utf-8").replace('"score": 0.25', f'"score": {constant}', 1)
    path.write_text(text, encoding="utf-8")

    with pytest.raises(DashboardResultError, match="finite|JSON"):
        load_inspection_result(eight_view_result_dir)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
