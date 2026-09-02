from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from bmw_inspection.lab.efficientad_ignore_mask import (
    mask_anomaly_map,
    load_ignore_mask_asset,
    polygons_to_ignore_mask,
    save_ignore_mask_asset,
)
from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER


def _images() -> dict[str, np.ndarray]:
    return {view: np.zeros((8, 10, 3), dtype=np.uint8) for view in VIEW_ORDER}


def test_polygons_to_ignore_mask_allows_empty_and_multiple_polygons() -> None:
    empty = polygons_to_ignore_mask((8, 10), [])
    mask = polygons_to_ignore_mask(
        (8, 10),
        [[(1, 1), (4, 1), (4, 4), (1, 4)], [(7, 5), (9, 5), (9, 7), (7, 7)]],
    )

    assert empty.dtype == np.uint8
    assert not np.any(empty)
    assert mask[2, 2] == 255
    assert mask[6, 8] == 255
    assert mask[0, 0] == 0


@pytest.mark.parametrize(
    "polygons",
    [
        [[(1, 1), (2, 2)]],
        [[(-1, 1), (2, 1), (2, 2)]],
        [[(1, 1), (10, 1), (2, 2)]],
    ],
)
def test_polygons_to_ignore_mask_rejects_invalid_polygon(polygons: object) -> None:
    with pytest.raises(ValueError):
        polygons_to_ignore_mask((8, 10), polygons)  # type: ignore[arg-type]


def test_save_and_load_ignore_mask_asset_is_sha_bound_and_keeps_empty_views(tmp_path: Path) -> None:
    roi_config = tmp_path / "roi.json"
    roi_config.write_text('{"roi":"v1"}\n', encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    images = _images()
    source_paths: dict[str, Path] = {}
    polygons = {view: [] for view in VIEW_ORDER}
    polygons["front"] = [[(1, 1), (4, 1), (4, 4), (1, 4)]]
    for view, image in images.items():
        path = source / f"{view}.png"
        assert cv2.imwrite(str(path), image)
        source_paths[view] = path

    index_path = save_ignore_mask_asset(
        tmp_path / "asset",
        images=images,
        polygons_by_view=polygons,
        source_paths=source_paths,
        source_capture_id="trusted-ok-1",
        public_roi_config=roi_config,
    )
    asset = load_ignore_mask_asset(
        index_path,
        expected_views=VIEW_ORDER,
        expected_roi_config_sha256=hashlib.sha256(roi_config.read_bytes()).hexdigest(),
        expected_shapes={view: (8, 10) for view in VIEW_ORDER},
    )

    assert tuple(asset.masks) == VIEW_ORDER
    assert asset.masks["front"][2, 2] == 255
    assert not np.any(asset.masks["back"])
    assert asset.source_capture_id == "trusted-ok-1"
    assert asset.index_sha256 == hashlib.sha256(index_path.read_bytes()).hexdigest()
    assert not asset.masks["front"].flags.writeable
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["mask_semantics"] == {"0": "inspect", "255": "ignore"}
    assert payload["views"]["front"]["polygon_count"] == 1
    assert payload["views"]["back"]["polygon_count"] == 0


def test_load_ignore_mask_asset_rejects_tampered_mask(tmp_path: Path) -> None:
    roi_config = tmp_path / "roi.json"
    roi_config.write_text("{}\n", encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    images = _images()
    source_paths = {}
    for view, image in images.items():
        path = source / f"{view}.png"
        assert cv2.imwrite(str(path), image)
        source_paths[view] = path
    index_path = save_ignore_mask_asset(
        tmp_path / "asset",
        images=images,
        polygons_by_view={view: [] for view in VIEW_ORDER},
        source_paths=source_paths,
        source_capture_id="trusted-ok-1",
        public_roi_config=roi_config,
    )
    (index_path.parent / "masks/front.png").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="SHA"):
        load_ignore_mask_asset(
            index_path,
            expected_views=VIEW_ORDER,
            expected_roi_config_sha256=hashlib.sha256(roi_config.read_bytes()).hexdigest(),
            expected_shapes={view: (8, 10) for view in VIEW_ORDER},
        )


def test_mask_anomaly_map_excludes_ignored_peak_and_reports_valid_hotspot() -> None:
    anomaly_map = np.array([[0.2, 1.0], [0.7, 0.4]], dtype=np.float32)
    ignore_mask = np.zeros((4, 4), dtype=np.uint8)
    ignore_mask[:2, 2:] = 255

    result = mask_anomaly_map(anomaly_map, ignore_mask)

    assert result.score == pytest.approx(0.7)
    assert (result.hotspot_x, result.hotspot_y) == (0, 1)
    assert result.raw_max == pytest.approx(1.0)
    assert result.ignored_map_pixel_count == 1
    assert result.masked_map[0, 1] == 0
    assert not result.masked_map.flags.writeable


def test_mask_anomaly_map_rejects_mask_that_ignores_every_pixel() -> None:
    with pytest.raises(ValueError, match="every anomaly-map pixel"):
        mask_anomaly_map(np.ones((2, 2), dtype=np.float32), np.full((4, 4), 255, dtype=np.uint8))
