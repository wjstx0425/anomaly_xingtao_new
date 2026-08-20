# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ZS32 PatchCore ROI dataset helpers."""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import ModuleType

import capture_data.zs32_patchcore_roi_dataset as roi_dataset
import cv2
import numpy as np
import pytest
from capture_data.zs32_patchcore_roi_dataset import (
    HANDS,
    VIEWS,
    SourceImage,
    discover_patchcore_images,
    load_patchcore_roi_config,
)
from zs32_inspection.domain.views import CANONICAL_VIEWS

LEGACY_VIEWS = ("front", "front_left", "front_right", "back", "back_left", "back_right")


def _load_stage30() -> ModuleType:
    """Load the numbered stage-30 wrapper from its file path."""
    script_path = Path(__file__).resolve().parents[3] / "pipeline/30_crop_zs32_patchcore_dataset.py"
    spec = importlib.util.spec_from_file_location("pipeline_stage30_patchcore_roi", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _complete_config() -> dict[str, object]:
    """Return a valid two-hand, eight-view configuration payload."""
    return {
        "schema_version": 1,
        "coordinate_system": "pixel_xyxy_half_open",
        "image_size": {"width": 40, "height": 30},
        "hands": {
            hand: {
                "views": {
                    view: {
                        "roi": [1, 2, 20, 25],
                        "reference_image": f"dataset/{hand}/{view}/normal/session/images/{hand}_{view}_001.png",
                    }
                    for view in VIEWS
                },
            }
            for hand in HANDS
        },
    }


def _write_config(path: Path, payload: dict[str, object]) -> None:
    """Write a JSON configuration fixture."""
    path.write_text(json.dumps(payload), encoding="utf-8")


def _add_image(
    dataset_root: Path,
    *,
    hand: str,
    view: str,
    label: str = "normal",
    defect_type: str = "",
    session_id: str = "session",
    filename_view: str | None = None,
    suffix: str = ".png",
    stem_tail: str = "001",
) -> Path:
    """Add one discoverable image fixture and return its path."""
    label_tail = (label, defect_type, session_id, "images") if defect_type else (label, session_id, "images")
    image_dir = dataset_root / hand / view
    for part in label_tail:
        image_dir /= part
    image_dir.mkdir(parents=True, exist_ok=True)
    resolved_filename_view = filename_view or view
    image_path = image_dir / f"{hand}_{resolved_filename_view}_{label}_{stem_tail}{suffix}"
    image_path.write_bytes(b"fixture")
    return image_path


def _complete_discovery_tree(dataset_root: Path) -> None:
    """Create one normal image for every required hand and view."""
    for hand in HANDS:
        for view in VIEWS:
            _add_image(dataset_root, hand=hand, view=view)


def _write_test_image(path: Path, *, width: int = 40, height: int = 30) -> None:
    """Write a small readable image fixture at an existing source path."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _complete_readable_tree(dataset_root: Path) -> dict[tuple[str, str], Path]:
    """Create one readable normal image for every required hand/view pair."""
    paths: dict[tuple[str, str], Path] = {}
    for hand in HANDS:
        for view in VIEWS:
            image_path = _add_image(dataset_root, hand=hand, view=view)
            _write_test_image(image_path)
            paths[(hand, view)] = image_path
    return paths


def test_config_loads_complete_roi_payload_without_mutating_it(tmp_path: Path) -> None:
    """A complete two-hand configuration returns typed ROIs and the original payload."""
    config_path = tmp_path / "rois.json"
    expected_payload = _complete_config()
    _write_config(config_path, expected_payload)

    width, height, rois, payload = load_patchcore_roi_config(config_path)

    assert (width, height) == (40, 30)
    assert rois["right"]["front"] == (1, 2, 20, 25)
    assert set(rois) == {"right", "left"}
    assert all(set(rois[hand]) == set(VIEWS) for hand in HANDS)
    assert payload == expected_payload


def test_config_loads_stage29_top_level_views_for_right_hand(tmp_path: Path) -> None:
    """The eight-view Stage 29 config is reusable for a right-only Stage 30 conversion."""
    payload = _complete_config()
    hands = payload.pop("hands")
    assert isinstance(hands, dict)
    right = hands["right"]
    assert isinstance(right, dict)
    payload["views"] = right["views"]
    config_path = tmp_path / "rois.json"
    _write_config(config_path, payload)

    width, height, rois, loaded = load_patchcore_roi_config(config_path, hands=("right",))

    assert (width, height) == (40, 30)
    assert set(rois) == {"right"}
    assert set(rois["right"]) == set(VIEWS)
    assert loaded == payload


@pytest.mark.parametrize(
    ("hands", "match"),
    [
        ({"right"}, "hands"),
        ({"right", "left", "unknown"}, "hands"),
    ],
)
def test_config_rejects_missing_or_extra_hands(tmp_path: Path, hands: set[str], match: str) -> None:
    """The hand mapping must contain exactly right and left."""
    payload = _complete_config()
    configured_hands = payload["hands"]
    assert isinstance(configured_hands, dict)
    configured_hands.clear()
    configured_hands.update({hand: {"views": {view: {"roi": [1, 2, 20, 25]} for view in VIEWS}} for hand in hands})
    config_path = tmp_path / "rois.json"
    _write_config(config_path, payload)

    with pytest.raises(ValueError, match=match):
        load_patchcore_roi_config(config_path)


@pytest.mark.parametrize("extra", [False, True])
def test_config_rejects_missing_or_extra_views(tmp_path: Path, extra: bool) -> None:
    """Each hand must contain exactly the eight canonical views."""
    payload = _complete_config()
    hands = payload["hands"]
    assert isinstance(hands, dict)
    right = hands["right"]
    assert isinstance(right, dict)
    views = right["views"]
    assert isinstance(views, dict)
    if extra:
        views["side"] = {"roi": [1, 2, 20, 25]}
    else:
        views.pop("back_right")
    config_path = tmp_path / "rois.json"
    _write_config(config_path, payload)

    with pytest.raises(ValueError, match="views"):
        load_patchcore_roi_config(config_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("coordinate_system", "normalized_xyxy"),
    ],
)
def test_config_rejects_bad_schema_or_coordinate_system(tmp_path: Path, field: str, value: object) -> None:
    """Only schema version one with pixel half-open coordinates is supported."""
    payload = _complete_config()
    payload[field] = value
    config_path = tmp_path / "rois.json"
    _write_config(config_path, payload)

    with pytest.raises(ValueError, match=field):
        load_patchcore_roi_config(config_path)


@pytest.mark.parametrize("value", [True, 1.5, "1"])
def test_config_rejects_non_integer_image_dimensions(tmp_path: Path, value: object) -> None:
    """Image dimensions must be true integers rather than coercible values."""
    payload = _complete_config()
    image_size = payload["image_size"]
    assert isinstance(image_size, dict)
    image_size["width"] = value
    config_path = tmp_path / "rois.json"
    _write_config(config_path, payload)

    with pytest.raises(ValueError, match="image_size"):
        load_patchcore_roi_config(config_path)


@pytest.mark.parametrize("value", [True, 1.5, "1"])
def test_config_rejects_non_integer_roi_values(tmp_path: Path, value: object) -> None:
    """ROI coordinates must be true integers rather than coercible values."""
    payload = _complete_config()
    hands = payload["hands"]
    assert isinstance(hands, dict)
    mutated = copy.deepcopy(hands["right"])
    assert isinstance(mutated, dict)
    views = mutated["views"]
    assert isinstance(views, dict)
    front = views["front"]
    assert isinstance(front, dict)
    front["roi"] = [value, 2, 20, 25]
    hands["right"] = mutated
    config_path = tmp_path / "rois.json"
    _write_config(config_path, payload)

    with pytest.raises(ValueError, match="ROI"):
        load_patchcore_roi_config(config_path)


@pytest.mark.parametrize(
    "roi",
    [
        [1, 2, 20],
        [-1, 2, 20, 25],
        [1, -1, 20, 25],
        [20, 2, 20, 25],
        [1, 25, 20, 25],
        [1, 2, 41, 25],
        [1, 2, 20, 31],
    ],
)
def test_config_rejects_malformed_or_out_of_bounds_rois(tmp_path: Path, roi: list[int]) -> None:
    """Every ROI must be a non-empty half-open rectangle inside the configured image."""
    payload = _complete_config()
    hands = payload["hands"]
    assert isinstance(hands, dict)
    right = hands["right"]
    assert isinstance(right, dict)
    views = right["views"]
    assert isinstance(views, dict)
    front = views["front"]
    assert isinstance(front, dict)
    front["roi"] = roi
    config_path = tmp_path / "rois.json"
    _write_config(config_path, payload)

    with pytest.raises(ValueError, match="ROI"):
        load_patchcore_roi_config(config_path)


def test_discovery_returns_frozen_records_in_stable_order_and_accepts_uppercase_extensions(tmp_path: Path) -> None:
    """Discovery is deterministic and treats supported filename extensions case-insensitively."""
    dataset_root = tmp_path / "dataset"
    _complete_discovery_tree(dataset_root)
    uppercase_path = _add_image(
        dataset_root,
        hand="right",
        view="front_left",
        label="normal_test",
        session_id="later",
        suffix=".JPG",
    )
    ignored_path = uppercase_path.with_suffix(".txt")
    ignored_path.write_text("ignored", encoding="utf-8")

    records = discover_patchcore_images(dataset_root)

    assert [record.source_path for record in records] == sorted(record.source_path for record in records)
    assert uppercase_path in {record.source_path for record in records}
    assert ignored_path not in {record.source_path for record in records}
    with pytest.raises(FrozenInstanceError):
        records[0].hand = "left"  # type: ignore[misc]


def test_discovery_supports_right_only_and_excludes_complete_sessions(tmp_path: Path) -> None:
    """Right-only discovery neither requires left normals nor emits an excluded session."""
    dataset_root = tmp_path / "dataset"
    for view in VIEWS:
        _add_image(dataset_root, hand="right", view=view, session_id="keep")
        _add_image(dataset_root, hand="right", view=view, session_id="duplicate", stem_tail="duplicate")

    records = discover_patchcore_images(
        dataset_root,
        hands=("right",),
        excluded_session_ids=("duplicate",),
    )

    assert len(records) == len(VIEWS)
    assert {record.hand for record in records} == {"right"}
    assert {record.session_id for record in records} == {"keep"}


def test_discovery_uses_longest_view_prefix_and_extracts_identity(tmp_path: Path) -> None:
    """Compound views, labels, defect types, sessions, and relative tails are preserved."""
    dataset_root = tmp_path / "dataset"
    _complete_discovery_tree(dataset_root)
    source_path = _add_image(
        dataset_root,
        hand="right",
        view="front_left",
        label="defect",
        defect_type="deform",
        session_id="capture_001",
        stem_tail="compound",
    )

    record = next(item for item in discover_patchcore_images(dataset_root) if item.source_path == source_path)

    assert record == SourceImage(
        source_path=source_path,
        hand="right",
        source_view="front_left",
        resolved_view="front_left",
        label="defect",
        defect_type="deform",
        session_id="capture_001",
        relative_tail=Path("defect/deform/capture_001/images/right_front_left_defect_compound.png"),
        view_corrected=False,
    )


def test_discovery_preserves_parent_view_when_filename_view_differs(tmp_path: Path) -> None:
    """The source directory wins over a different, but still valid, filename view."""
    dataset_root = tmp_path / "dataset"
    _complete_discovery_tree(dataset_root)
    source_path = _add_image(
        dataset_root,
        hand="right",
        view="back",
        filename_view="front",
        label="defect",
        defect_type="deform",
        session_id="capture_002",
        stem_tail="corrected",
    )

    record = next(item for item in discover_patchcore_images(dataset_root) if item.source_path == source_path)

    assert (record.source_view, record.resolved_view, record.view_corrected) == ("back", "back", False)
    assert record.relative_tail == Path("defect/deform/capture_002/images/right_front_defect_corrected.png")


def test_discovery_uses_parent_view_for_normal_completeness(tmp_path: Path) -> None:
    """A valid filename mismatch does not move normal data out of its source view."""
    dataset_root = tmp_path / "dataset"
    _complete_discovery_tree(dataset_root)
    back_normal = dataset_root / "right/back/normal/session/images/right_back_normal_001.png"
    back_normal.unlink()
    _add_image(
        dataset_root,
        hand="right",
        view="back",
        filename_view="front",
        session_id="misplaced",
        stem_tail="misplaced",
    )

    records = discover_patchcore_images(dataset_root)

    record = next(item for item in records if item.session_id == "misplaced")
    assert (record.source_view, record.resolved_view, record.view_corrected) == ("back", "back", False)


def test_discovery_rejects_unknown_filename_view_prefix(tmp_path: Path) -> None:
    """Every image filename must identify one of the eight canonical views."""
    dataset_root = tmp_path / "dataset"
    _complete_discovery_tree(dataset_root)
    source_path = dataset_root / "right/front/normal/session/images/right_side_normal_001.png"
    source_path.write_bytes(b"fixture")

    with pytest.raises(ValueError, match=r"filename.*view|view.*filename"):
        discover_patchcore_images(dataset_root)


def test_discovery_rejects_filename_hand_mismatch(tmp_path: Path) -> None:
    """A filename cannot silently move an image from one hand to the other."""
    dataset_root = tmp_path / "dataset"
    _complete_discovery_tree(dataset_root)
    source_path = dataset_root / "right/front/normal/session/images/left_front_normal_001.png"
    source_path.write_bytes(b"fixture")

    with pytest.raises(ValueError, match="hand mismatch"):
        discover_patchcore_images(dataset_root)


@pytest.mark.parametrize("missing", ["hand", "view", "normal"])
def test_discovery_rejects_missing_required_data(tmp_path: Path, missing: str) -> None:
    """Both hands, all views, and a normal image per hand/view are mandatory."""
    dataset_root = tmp_path / "dataset"
    _complete_discovery_tree(dataset_root)
    if missing == "hand":
        target = dataset_root / "left"
    elif missing == "view":
        target = dataset_root / "right" / "front"
    else:
        target = dataset_root / "right/front/normal/session/images/right_front_normal_001.png"
    if target.is_file():
        target.unlink()
    else:
        for path in sorted(target.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            else:
                path.rmdir()
        target.rmdir()

    with pytest.raises((FileNotFoundError, ValueError), match=missing):
        discover_patchcore_images(dataset_root)


def test_discovery_keeps_same_filename_in_different_source_views_separate(tmp_path: Path) -> None:
    """Matching filenames remain distinct when their source view directories differ."""
    dataset_root = tmp_path / "dataset"
    _complete_discovery_tree(dataset_root)
    collision_name = "right_front_defect_collision.png"
    for source_view in ("front", "back"):
        image_dir = dataset_root / "right" / source_view / "defect/deform/capture/images"
        image_dir.mkdir(parents=True, exist_ok=True)
        (image_dir / collision_name).write_bytes(b"fixture")

    records = discover_patchcore_images(dataset_root)

    matching = [record for record in records if record.source_path.name == collision_name]
    assert [(record.source_view, record.resolved_view) for record in matching] == [
        ("back", "back"),
        ("front", "front"),
    ]


def test_selection_reuses_existing_rois_and_atomically_writes_all_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selection uses stable normal references and publishes only one complete configuration."""
    repo_root = tmp_path
    dataset_root = repo_root / "dataset"
    selected_images: dict[tuple[str, str], Path] = {}
    for hand in HANDS:
        for view in VIEWS:
            image_path = _add_image(dataset_root, hand=hand, view=view)
            _write_test_image(image_path)
            selected_images[(hand, view)] = image_path
    unreadable_first = _add_image(dataset_root, hand="right", view="front", stem_tail="000")

    config_path = repo_root / "config/rois.json"
    config_path.parent.mkdir(parents=True)
    existing_payload = _complete_config()
    _write_config(config_path, existing_payload)
    existing_text = config_path.read_text(encoding="utf-8")
    preview_dir = repo_root / "previews"
    selection_calls: list[tuple[Path, tuple[int, int, int, int] | None, int, int]] = []
    overlay_calls: list[tuple[Path, tuple[int, ...], tuple[int, int, int, int]]] = []
    replace_calls: list[tuple[Path, Path]] = []
    original_replace = Path.replace

    def fake_select_roi(
        image_path: Path,
        initial_roi: tuple[int, int, int, int] | None,
        max_width: int,
        max_height: int,
    ) -> tuple[int, int, int, int]:
        assert config_path.read_text(encoding="utf-8") == existing_text
        assert not config_path.with_suffix(config_path.suffix + ".tmp").exists()
        selection_calls.append((image_path, initial_roi, max_width, max_height))
        index = len(selection_calls) - 1
        return index, 1, index + 10, 20

    def fake_save_overlay(path: Path, image: np.ndarray, roi: tuple[int, int, int, int]) -> None:
        overlay_calls.append((path, image.shape, roi))

    def track_replace(source: Path, target: Path) -> Path:
        replace_calls.append((source, target))
        return original_replace(source, target)

    monkeypatch.setattr(roi_dataset, "select_roi", fake_select_roi, raising=False)
    monkeypatch.setattr(roi_dataset, "save_overlay", fake_save_overlay, raising=False)
    monkeypatch.setattr(Path, "replace", track_replace)

    payload = roi_dataset.select_patchcore_rois(
        repo_root=repo_root,
        dataset_root=dataset_root,
        config_path=config_path,
        preview_dir=preview_dir,
        max_window_width=800,
        max_window_height=600,
    )

    expected_order = [(hand, view) for hand in HANDS for view in VIEWS]
    assert unreadable_first < selected_images[("right", "front")]
    assert [call[0] for call in selection_calls] == [selected_images[pair] for pair in expected_order]
    pair_count = len(HANDS) * len(VIEWS)
    assert [call[1] for call in selection_calls] == [(1, 2, 20, 25)] * pair_count
    assert [call[2:] for call in selection_calls] == [(800, 600)] * pair_count
    assert [call[0].name for call in overlay_calls] == [f"{hand}_{view}_roi.png" for hand, view in expected_order]
    assert [call[1] for call in overlay_calls] == [(30, 40, 3)] * pair_count
    assert [call[2] for call in overlay_calls] == [(index, 1, index + 10, 20) for index in range(pair_count)]
    assert replace_calls == [(config_path.with_suffix(".json.tmp"), config_path)]
    assert not config_path.with_suffix(".json.tmp").exists()
    assert json.loads(config_path.read_text(encoding="utf-8")) == payload
    assert payload["image_size"] == {"width": 40, "height": 30}
    hands_payload = payload["hands"]
    assert isinstance(hands_payload, dict)
    assert sum(len(hand_payload["views"]) for hand_payload in hands_payload.values()) == pair_count
    for index, (hand, view) in enumerate(expected_order):
        view_payload = hands_payload[hand]["views"][view]
        assert view_payload == {
            "roi": [index, 1, index + 10, 20],
            "reference_image": selected_images[(hand, view)].relative_to(repo_root).as_posix(),
        }


def test_selection_rejects_reference_size_mismatch_before_opening_selector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All normal reference images must share one source size before selection starts."""
    dataset_root = tmp_path / "dataset"
    for hand in HANDS:
        for view in VIEWS:
            image_path = _add_image(dataset_root, hand=hand, view=view)
            width = 41 if (hand, view) == ("left", "back_right") else 40
            _write_test_image(image_path, width=width)
    config_path = tmp_path / "rois.json"
    original_payload = _complete_config()
    _write_config(config_path, original_payload)
    monkeypatch.setattr(
        roi_dataset,
        "select_roi",
        lambda *_args, **_kwargs: pytest.fail("selector must not open before size preflight"),
        raising=False,
    )

    with pytest.raises(ValueError, match="same size|size mismatch|dimensions"):
        roi_dataset.select_patchcore_rois(tmp_path, dataset_root, config_path, tmp_path / "previews")

    assert json.loads(config_path.read_text(encoding="utf-8")) == original_payload
    assert not config_path.with_suffix(".json.tmp").exists()


def test_preflight_rejects_unreadable_image_without_creating_output(tmp_path: Path) -> None:
    """An unreadable source fails before the converter creates any output artifact."""
    repo_root = tmp_path
    dataset_root = repo_root / "dataset"
    paths = _complete_readable_tree(dataset_root)
    paths[("right", "front")].write_bytes(b"not an image")
    config_path = repo_root / "rois.json"
    _write_config(config_path, _complete_config())
    output_root = dataset_root / "cropped"

    with pytest.raises(FileNotFoundError, match="read image|unreadable"):
        roi_dataset.crop_patchcore_dataset(repo_root, dataset_root, output_root, config_path)

    assert not output_root.exists()
    assert not list(dataset_root.glob(".cropped.*"))


def test_right_only_conversion_accepts_nested_dataset_and_stage29_config(tmp_path: Path) -> None:
    """Stage 30 can crop nested right-only data with the eight-view Stage 29 ROI file."""
    repo_root = tmp_path
    dataset_root = repo_root / "dataset/zs32_new"
    for view in CANONICAL_VIEWS:
        retained = _add_image(dataset_root, hand="right", view=view, session_id="keep")
        excluded = _add_image(dataset_root, hand="right", view=view, session_id="duplicate", stem_tail="duplicate")
        _write_test_image(retained)
        _write_test_image(excluded)
    payload = {
        "schema_version": 1,
        "coordinate_system": "pixel_xyxy_half_open",
        "image_size": {"width": 40, "height": 30},
        "views": {view: {"roi": [1, 2, 20, 25]} for view in CANONICAL_VIEWS},
    }
    config_path = repo_root / "dataset/zs32_eight_view_roi_config.json"
    _write_config(config_path, payload)
    output_root = repo_root / "dataset/zs32_eight_view_patchcore_roi"

    result = roi_dataset.crop_patchcore_dataset(
        repo_root,
        dataset_root,
        output_root,
        config_path,
        hands=("right",),
        expected_views=CANONICAL_VIEWS,
        excluded_session_ids=("duplicate",),
    )

    assert result["total_images"] == len(CANONICAL_VIEWS)
    assert len(list((output_root / "right").rglob("*.png"))) == len(CANONICAL_VIEWS)
    manifest = (output_root / "crop_manifest.csv").read_text(encoding="utf-8")
    assert "duplicate" not in manifest


def test_preflight_rejects_image_size_mismatch_without_creating_output(tmp_path: Path) -> None:
    """Every source must match the configured size before conversion begins."""
    repo_root = tmp_path
    dataset_root = repo_root / "dataset"
    paths = _complete_readable_tree(dataset_root)
    _write_test_image(paths[("left", "back_right")], width=41)
    config_path = repo_root / "rois.json"
    _write_config(config_path, _complete_config())
    output_root = dataset_root / "cropped"

    with pytest.raises(ValueError, match="size mismatch|dimensions"):
        roi_dataset.crop_patchcore_dataset(repo_root, dataset_root, output_root, config_path)

    assert not output_root.exists()


@pytest.mark.parametrize(
    "output_relative",
    [
        Path("outside"),
        Path("dataset"),
        Path("dataset/right"),
        Path("dataset/right/cropped"),
        Path("dataset/left"),
        Path("dataset/left/cropped"),
    ],
)
def test_protection_rejects_unsafe_output_paths(tmp_path: Path, output_relative: Path) -> None:
    """Output must be a separate strict child of repo_root/dataset and cannot overlap either input."""
    repo_root = tmp_path
    dataset_root = repo_root / "dataset"
    _complete_readable_tree(dataset_root)
    config_path = repo_root / "rois.json"
    _write_config(config_path, _complete_config())

    with pytest.raises(ValueError, match="safe|separate|overlap"):
        roi_dataset.crop_patchcore_dataset(
            repo_root,
            dataset_root,
            repo_root / output_relative,
            config_path,
        )


def test_protection_rejects_existing_output_without_overwrite(tmp_path: Path) -> None:
    """Default conversion refuses to alter an existing output directory."""
    repo_root = tmp_path
    dataset_root = repo_root / "dataset"
    _complete_readable_tree(dataset_root)
    config_path = repo_root / "rois.json"
    _write_config(config_path, _complete_config())
    output_root = dataset_root / "cropped"
    output_root.mkdir()
    sentinel = output_root / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="overwrite"):
        roi_dataset.crop_patchcore_dataset(repo_root, dataset_root, output_root, config_path)

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_overwrite_keeps_same_filename_in_different_source_views_separate(tmp_path: Path) -> None:
    """Overwrite publishes both files under their original source view directories."""
    repo_root = tmp_path
    dataset_root = repo_root / "dataset"
    _complete_readable_tree(dataset_root)
    collision_name = "right_front_defect_collision.png"
    for source_view in ("front", "back"):
        image_dir = dataset_root / "right" / source_view / "defect/deform/capture/images"
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / collision_name
        image_path.touch()
        _write_test_image(image_path)
    config_path = repo_root / "rois.json"
    _write_config(config_path, _complete_config())
    output_root = dataset_root / "cropped"
    output_root.mkdir()
    sentinel = output_root / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    roi_dataset.crop_patchcore_dataset(
        repo_root,
        dataset_root,
        output_root,
        config_path,
        overwrite=True,
    )

    assert not sentinel.exists()
    assert (output_root / f"right/front/defect/deform/capture/images/{collision_name}").is_file()
    assert (output_root / f"right/back/defect/deform/capture/images/{collision_name}").is_file()
    assert not list(dataset_root.glob(".cropped.*"))


def test_preflight_invalid_source_preserves_existing_output_with_overwrite(tmp_path: Path) -> None:
    """Overwrite does not move or delete old output until every source has passed preflight."""
    repo_root = tmp_path
    dataset_root = repo_root / "dataset"
    paths = _complete_readable_tree(dataset_root)
    paths[("left", "front_left")].write_bytes(b"unreadable")
    config_path = repo_root / "rois.json"
    _write_config(config_path, _complete_config())
    output_root = dataset_root / "cropped"
    output_root.mkdir()
    sentinel = output_root / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="read image|unreadable"):
        roi_dataset.crop_patchcore_dataset(
            repo_root,
            dataset_root,
            output_root,
            config_path,
            overwrite=True,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not list(dataset_root.glob(".cropped.*"))


def _varied_config() -> tuple[dict[str, object], dict[tuple[str, str], tuple[int, int, int, int]]]:
    """Return a valid config with a distinct ROI for every hand/view pair."""
    payload = _complete_config()
    hands_payload = payload["hands"]
    assert isinstance(hands_payload, dict)
    rois: dict[tuple[str, str], tuple[int, int, int, int]] = {}
    for index, (hand, view) in enumerate((hand, view) for hand in HANDS for view in VIEWS):
        x1, y1 = index % 5, index % 4
        roi = (x1, y1, x1 + 2 + index % 3, y1 + 3 + index % 2)
        hand_payload = hands_payload[hand]
        assert isinstance(hand_payload, dict)
        views_payload = hand_payload["views"]
        assert isinstance(views_payload, dict)
        view_payload = views_payload[view]
        assert isinstance(view_payload, dict)
        view_payload["roi"] = list(roi)
        rois[(hand, view)] = roi
    return payload, rois


def _pattern_image(width: int, height: int, offset: int) -> np.ndarray:
    """Create an exactly reproducible color image with position-dependent pixels."""
    y, x = np.indices((height, width), dtype=np.uint16)
    return np.stack(
        (
            (x + offset) % 256,
            (y * 3 + offset) % 256,
            (x + y * 5 + offset) % 256,
        ),
        axis=2,
    ).astype(np.uint8)


def test_conversion_shows_visible_progress_for_all_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Conversion wraps every prepared image in one visible progress iterator."""
    repo_root = tmp_path
    dataset_root = repo_root / "dataset"
    _complete_readable_tree(dataset_root)
    config_path = repo_root / "rois.json"
    _write_config(config_path, _complete_config())
    calls: list[tuple[int, str]] = []

    def fake_track(items: list[object], *, description: str) -> list[object]:
        calls.append((len(items), description))
        return items

    monkeypatch.setattr(roi_dataset, "track", fake_track, raising=False)

    roi_dataset.crop_patchcore_dataset(
        repo_root,
        dataset_root,
        dataset_root / "cropped",
        config_path,
    )

    image_count = len(HANDS) * len(VIEWS)
    assert calls == [(image_count, "Checking images"), (image_count, "Cropping images")]


def test_conversion_preserves_pixels_hierarchy_extensions_and_writes_complete_metadata(tmp_path: Path) -> None:
    """Conversion uses each hand/view ROI and publishes a complete reproducible dataset."""
    repo_root = tmp_path
    dataset_root = repo_root / "dataset"
    payload, rois = _varied_config()
    sources: dict[Path, np.ndarray] = {}
    for index, (hand, view) in enumerate((hand, view) for hand in HANDS for view in VIEWS):
        source_path = _add_image(dataset_root, hand=hand, view=view)
        image = _pattern_image(40, 30, index * 13)
        assert cv2.imwrite(str(source_path), image)
        sources[source_path] = image

    normal_test_path = _add_image(
        dataset_root,
        hand="left",
        view="front_left",
        label="normal_test",
        session_id="validation_session",
        suffix=".bmp",
        stem_tail="validation",
    )
    normal_test_image = _pattern_image(40, 30, 201)
    assert cv2.imwrite(str(normal_test_path), normal_test_image)
    sources[normal_test_path] = normal_test_image
    corrected_path = _add_image(
        dataset_root,
        hand="right",
        view="back",
        filename_view="front",
        label="defect",
        defect_type="deform",
        session_id="defect_session",
        suffix=".tiff",
        stem_tail="corrected",
    )
    corrected_image = _pattern_image(40, 30, 233)
    assert cv2.imwrite(str(corrected_path), corrected_image)
    sources[corrected_path] = corrected_image
    source_bytes = {path: path.read_bytes() for path in sources}

    config_path = repo_root / "rois.json"
    _write_config(config_path, payload)
    output_root = dataset_root / "cropped"

    result = roi_dataset.crop_patchcore_dataset(repo_root, dataset_root, output_root, config_path)

    assert result["output_root"] == output_root
    assert result["manifest_path"] == output_root / "crop_manifest.csv"
    expected_count = len(HANDS) * len(VIEWS) + 2
    assert result["total_images"] == expected_count
    assert result["output_images"] == expected_count
    assert result["corrected_views"] == 0
    assert result["cleanup_warning"] == ""
    assert result["backup_path"] == ""
    assert {path: path.read_bytes() for path in sources} == source_bytes

    for source_path, source_image in sources.items():
        record = next(record for record in discover_patchcore_images(dataset_root) if record.source_path == source_path)
        output_path = output_root / record.hand / record.resolved_view / record.relative_tail
        assert output_path.suffix == source_path.suffix
        actual = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
        x1, y1, x2, y2 = rois[(record.hand, record.resolved_view)]
        np.testing.assert_array_equal(actual, source_image[y1:y2, x1:x2])

    with (output_root / "crop_manifest.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == expected_count
    assert set(rows[0]) == {
        "source_path",
        "output_path",
        "hand",
        "source_view",
        "resolved_view",
        "view_corrected",
        "label",
        "defect_type",
        "session_id",
        "roi_x1",
        "roi_y1",
        "roi_x2",
        "roi_y2",
        "source_width",
        "source_height",
        "crop_width",
        "crop_height",
    }
    corrected_row = next(row for row in rows if row["source_path"] == corrected_path.relative_to(repo_root).as_posix())
    corrected_roi = rois[("right", "back")]
    assert corrected_row == {
        "source_path": corrected_path.relative_to(repo_root).as_posix(),
        "output_path": (
            output_root / "right/back/defect/deform/defect_session/images/right_front_defect_corrected.tiff"
        )
        .relative_to(repo_root)
        .as_posix(),
        "hand": "right",
        "source_view": "back",
        "resolved_view": "back",
        "view_corrected": "false",
        "label": "defect",
        "defect_type": "deform",
        "session_id": "defect_session",
        "roi_x1": str(corrected_roi[0]),
        "roi_y1": str(corrected_roi[1]),
        "roi_x2": str(corrected_roi[2]),
        "roi_y2": str(corrected_roi[3]),
        "source_width": "40",
        "source_height": "30",
        "crop_width": str(corrected_roi[2] - corrected_roi[0]),
        "crop_height": str(corrected_roi[3] - corrected_roi[1]),
    }
    assert json.loads((output_root / "roi_config.json").read_text(encoding="utf-8")) == payload

    expected_summary = {
        "total_images": expected_count,
        "output_images": expected_count,
        "corrected_views": 0,
        "by_hand_view": {
            hand: {
                view: {
                    "input": 1 + int((hand, view) == ("left", "front_left")) + int((hand, view) == ("right", "back")),
                    "output": 1 + int((hand, view) == ("left", "front_left")) + int((hand, view) == ("right", "back")),
                    "corrected_views": 0,
                    "labels": {
                        "normal": 1,
                        **({"normal_test": 1} if (hand, view) == ("left", "front_left") else {}),
                        **({"defect": 1} if (hand, view) == ("right", "back") else {}),
                    },
                }
                for view in VIEWS
            }
            for hand in HANDS
        },
    }
    assert json.loads((output_root / "summary.json").read_text(encoding="utf-8")) == expected_summary
    assert result["by_hand_view"] == expected_summary["by_hand_view"]


def test_conversion_uses_png_compression_one_only_for_png(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PNG outputs use fast compression one while other formats use OpenCV defaults."""
    repo_root = tmp_path
    dataset_root = repo_root / "dataset"
    _complete_readable_tree(dataset_root)
    bmp_path = _add_image(
        dataset_root,
        hand="left",
        view="front",
        label="normal_test",
        suffix=".bmp",
    )
    _write_test_image(bmp_path)
    config_path = repo_root / "rois.json"
    _write_config(config_path, _complete_config())
    calls: list[tuple[Path, list[int] | None]] = []
    original_imwrite = cv2.imwrite

    def track_imwrite(path: str, image: np.ndarray, params: list[int] | None = None) -> bool:
        calls.append((Path(path), params))
        return original_imwrite(path, image, params) if params is not None else original_imwrite(path, image)

    monkeypatch.setattr(roi_dataset.cv2, "imwrite", track_imwrite)

    roi_dataset.crop_patchcore_dataset(
        repo_root,
        dataset_root,
        dataset_root / "cropped",
        config_path,
    )

    assert calls
    assert all(params == [cv2.IMWRITE_PNG_COMPRESSION, 1] for path, params in calls if path.suffix == ".png")
    assert all(params is None for path, params in calls if path.suffix != ".png")


def test_conversion_restores_backup_when_atomic_replacement_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed publish restores the old output and removes transaction artifacts."""
    repo_root = tmp_path
    dataset_root = repo_root / "dataset"
    _complete_readable_tree(dataset_root)
    config_path = repo_root / "rois.json"
    _write_config(config_path, _complete_config())
    output_root = dataset_root / "cropped"
    output_root.mkdir()
    sentinel = output_root / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    original_replace = Path.replace
    failed = False

    def fail_new_output_publish(source: Path, target: Path) -> Path:
        nonlocal failed
        if target == output_root and source.name.startswith(".cropped.tmp-") and not failed:
            failed = True
            raise OSError("simulated atomic publish failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_new_output_publish)

    with pytest.raises(OSError, match="publish failure"):
        roi_dataset.crop_patchcore_dataset(
            repo_root,
            dataset_root,
            output_root,
            config_path,
            overwrite=True,
        )

    assert failed
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not list(dataset_root.glob(".cropped.*"))


def test_backup_cleanup_failure_keeps_committed_output_and_reports_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure to delete an old backup is a warning after the new output is committed."""
    repo_root = tmp_path
    dataset_root = repo_root / "dataset"
    _complete_readable_tree(dataset_root)
    config_path = repo_root / "rois.json"
    _write_config(config_path, _complete_config())
    output_root = dataset_root / "cropped"
    output_root.mkdir()
    sentinel = output_root / "sentinel.txt"
    sentinel.write_text("old output", encoding="utf-8")
    original_rmtree = roi_dataset.shutil.rmtree

    def fail_backup_cleanup(path: Path) -> None:
        if path.name.startswith(".cropped.backup-"):
            raise OSError("simulated backup cleanup failure")
        original_rmtree(path)

    monkeypatch.setattr(roi_dataset.shutil, "rmtree", fail_backup_cleanup)

    result = roi_dataset.crop_patchcore_dataset(
        repo_root,
        dataset_root,
        output_root,
        config_path,
        overwrite=True,
    )

    expected_count = len(HANDS) * len(VIEWS)
    assert result["total_images"] == expected_count
    assert result["output_images"] == expected_count
    assert "simulated backup cleanup failure" in result["cleanup_warning"]
    backup_path = Path(result["backup_path"])
    assert backup_path.is_dir()
    assert (backup_path / "sentinel.txt").read_text(encoding="utf-8") == "old output"
    assert not sentinel.exists()
    assert (output_root / "crop_manifest.csv").is_file()
    assert (output_root / "roi_config.json").is_file()
    assert (output_root / "summary.json").is_file()
    assert len(list(output_root.rglob("*.png"))) == expected_count


def test_stage30_parser_uses_repo_local_defaults() -> None:
    """Stage 30 defaults to the repository's source, config, preview, and output paths."""
    stage30 = _load_stage30()
    dataset_root = stage30.REPO_ROOT / "dataset"

    select_args = stage30.build_parser().parse_args(["select"])
    convert_args = stage30.build_parser().parse_args(["convert"])

    assert select_args.dataset_root == dataset_root
    assert select_args.config == dataset_root / "zs32_eight_view_patchcore_roi_config.json"
    assert select_args.preview_dir == dataset_root / "zs32_eight_view_patchcore_roi_previews"
    assert convert_args.dataset_root == dataset_root
    assert convert_args.output_root == dataset_root / "zs32_eight_view_patchcore_roi"
    assert convert_args.overwrite is False


def test_stage30_select_and_convert_delegate_and_print_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Stage 30 delegates both commands and reports ROI and conversion results."""
    stage30 = _load_stage30()
    payload = {
        "hands": {
            hand: {"views": {view: {"roi": [1, 2, 20, 25]} for view in CANONICAL_VIEWS}} for hand in HANDS
        },
    }
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_select(**kwargs: object) -> dict[str, object]:
        calls.append(("select", kwargs))
        return payload

    def fake_convert(**kwargs: object) -> dict[str, object]:
        calls.append(("convert", kwargs))
        return {
            "output_root": tmp_path / "output",
            "manifest_path": tmp_path / "output/crop_manifest.csv",
            "total_images": 12,
            "output_images": 12,
            "corrected_views": 2,
            "cleanup_warning": "old backup kept",
        }

    monkeypatch.setattr(stage30, "select_patchcore_rois", fake_select)
    monkeypatch.setattr(stage30, "crop_patchcore_dataset", fake_convert)

    stage30.main(["select", "--dataset-root", str(tmp_path / "dataset"), "--hand", "right"])
    select_output = capsys.readouterr().out
    stage30.main(
        [
            "convert",
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--hand",
            "right",
            "--exclude-session",
            "duplicate-session",
            "--overwrite",
        ],
    )
    convert_output = capsys.readouterr().out

    assert len([line for line in select_output.splitlines() if "[1, 2, 20, 25]" in line]) == len(CANONICAL_VIEWS)
    assert "Images: 12 -> 12" in convert_output
    assert "Corrected views: 2" in convert_output
    assert "WARNING: old backup kept" in convert_output
    assert calls[0][0] == "select"
    assert calls[1][0] == "convert"
    assert calls[1][1]["overwrite"] is True
    assert calls[1][1]["hands"] == ("right",)
    assert calls[0][1]["expected_views"] == CANONICAL_VIEWS
    assert calls[1][1]["expected_views"] == CANONICAL_VIEWS
    assert calls[1][1]["excluded_session_ids"] == ("duplicate-session",)
def test_legacy_six_view_roi_config_replays_by_default(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "coordinate_system": "pixel_xyxy_half_open",
        "image_size": {"width": 100, "height": 80},
        "hands": {
            hand: {"views": {view: {"roi": [0, 0, 100, 80]} for view in LEGACY_VIEWS}} for hand in HANDS
        },
    }
    path = tmp_path / "legacy-six.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    _, _, rois, _ = load_patchcore_roi_config(path)

    assert VIEWS == LEGACY_VIEWS
    assert all(tuple(rois[hand]) == LEGACY_VIEWS for hand in HANDS)


def test_patchcore_strict_eight_mode_requires_exact_canonical_set(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "coordinate_system": "pixel_xyxy_half_open",
        "image_size": {"width": 100, "height": 80},
        "views": {view: {"roi": [0, 0, 100, 80]} for view in CANONICAL_VIEWS},
    }
    path = tmp_path / "strict-eight.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    _, _, rois, _ = load_patchcore_roi_config(path, hands=("right",), expected_views=CANONICAL_VIEWS)

    assert tuple(rois["right"]) == CANONICAL_VIEWS
