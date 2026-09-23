"""Verify exposure-report metrics, escaping and preservation of source images."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from bmw_inspection.capture.exposure_report import write_report


def _record(path: str = "image.png") -> dict:
    return {"path": path, "camera_serial": "camera", "view": "front", "repeat": 0, "kind": "single", "exposures_us": [1500.0]}


def test_metrics_and_original_preservation(tmp_path: Path) -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[5:] = 255
    source = tmp_path / "image.png"
    assert cv2.imwrite(str(source), image)
    original = source.read_bytes()
    record = _record()
    write_report(tmp_path, [record], {"roi": None})
    payload = json.loads((tmp_path / "report.json").read_text())
    metrics = payload["records"][0]["metrics"]
    assert metrics["mean"] == 127.5
    assert metrics["std"] == 127.5
    assert metrics["p01"] == 0
    assert metrics["p99"] == 255
    assert metrics["dark_pct"] == 50
    assert metrics["clip_pct"] == 50
    assert metrics["laplacian_variance"] > 0
    assert source.read_bytes() == original
    assert "metrics" not in record
    assert "dark_pct" in (tmp_path / "metrics.csv").read_text(encoding="utf-8-sig")


def test_roi_and_channel_clipping(tmp_path: Path) -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[5:, :, 0] = 255
    assert cv2.imwrite(str(tmp_path / "image.png"), image)
    write_report(tmp_path, [_record()], {"roi": [0, 5, 10, 10]})
    metrics = json.loads((tmp_path / "report.json").read_text())["records"][0]["metrics"]
    assert metrics["clip_pct"] == 100
    assert metrics["dark_pct"] == 0
    assert metrics["mean"] == 29
    assert metrics["laplacian_variance"] == 0


@pytest.mark.parametrize("roi", [[0, 0, 11, 10], [2, 0, 1, 5], [-1, 0, 2, 2], [0, 0, 2], [0, 0, 1.5, 2], [False, 0, 2, 2]])
def test_invalid_roi(tmp_path: Path, roi: list) -> None:
    assert cv2.imwrite(str(tmp_path / "image.png"), np.zeros((10, 10, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="roi"):
        write_report(tmp_path, [_record()], {"roi": roi})
    assert not (tmp_path / "index.html").exists()


def test_html_escaping_and_thumbnail_bounds(tmp_path: Path) -> None:
    filename = 'image & "quote".png'
    assert cv2.imwrite(str(tmp_path / filename), np.zeros((100, 1000, 3), dtype=np.uint8))
    record = {**_record(filename), "view": '<script>alert("x")</script>'}
    write_report(tmp_path, [record], {"note": "<script>bad()</script>"})
    page = (tmp_path / "index.html").read_text()
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
    assert "image%20%26%20%22quote%22.png" in page
    payload = json.loads((tmp_path / "report.json").read_text())
    thumbnail = cv2.imread(str(tmp_path / payload["records"][0]["thumbnail"]))
    assert thumbnail.shape[:2] == (48, 480)


def test_rejects_16_bit_image(tmp_path: Path) -> None:
    assert cv2.imwrite(str(tmp_path / "image.png"), np.zeros((10, 10, 3), dtype=np.uint16))
    with pytest.raises(ValueError, match="uint8 BGR"):
        write_report(tmp_path, [_record()], {})


def test_rejects_path_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inside the output"):
        write_report(tmp_path, [_record("../image.png")], {})


def test_unicode_paths_and_pair_gallery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "曝光测试"
    output.mkdir()
    records = []
    for repeat in (1, 2):
        for kind, exposures in (("single", [1000.0]), ("single", [4000.0]), ("selective", [1000.0, 4000.0]), ("mertens", [1000.0, 4000.0])):
            filename = f"原图_{repeat}_{kind}_{exposures[0]:g}.png"
            ok, encoded = cv2.imencode(".png", np.full((10, 10, 3), repeat * 40, dtype=np.uint8))
            assert ok
            (output / filename).write_bytes(encoded.tobytes())
            records.append({**_record(filename), "repeat": repeat, "kind": kind, "exposures_us": exposures})

    def reject_path_api(*args: object, **kwargs: object) -> None:
        raise AssertionError("Use Unicode-safe encoded image IO")

    monkeypatch.setattr(cv2, "imread", reject_path_api)
    monkeypatch.setattr(cv2, "imwrite", reject_path_api)
    write_report(output, records, {})
    page = (output / "index.html").read_text()
    assert page.index("同组曝光与融合对比") < page.index("逐图指标")
    sections = page.split("<section>")[1:]
    assert len(sections) == 2
    for repeat, section in enumerate(sections, start=1):
        gallery = section.split("</section>")[0]
        assert gallery.count("<figure>") == 4
        assert f"第 {repeat} 轮" in gallery
        assert "短曝光 1000 μs" in gallery
        assert "长曝光 4000 μs" in gallery
        assert "selective · 1000 / 4000 μs" in gallery
        assert "mertens · 1000 / 4000 μs" in gallery
        assert f"_{repeat}_selective_1000.png" in gallery
        assert f"_{3 - repeat}_selective_1000.png" not in gallery
    payload = json.loads((output / "report.json").read_text())
    assert len(list(output.glob("report_thumbnails_*/*.jpg"))) == len(records)
    assert all((output / record["thumbnail"]).is_file() for record in payload["records"])
