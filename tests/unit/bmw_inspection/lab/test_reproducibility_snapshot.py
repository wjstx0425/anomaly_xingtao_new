"""Tests for the BMW deployment reproducibility receipt."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[4] / "pipeline/bmw_lab_snapshot_reproducibility.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("bmw_lab_snapshot_reproducibility", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load reproducibility snapshot module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    run_dir = tmp_path / "results/current"
    (run_dir / "efficientad/front").mkdir(parents=True)
    (run_dir / "efficientad/front/model.ckpt").write_bytes(b"checkpoint")
    (run_dir / "run_report.json").write_text('{"status":"complete"}\n', encoding="utf-8")
    release = tmp_path / "training/release"
    release.mkdir(parents=True)
    (release / "report.json").write_text('{"release_status":"published"}\n', encoding="utf-8")
    image = tmp_path / "dataset/image.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"customer-image")
    (release / "image.png").symlink_to(image)
    demo = tmp_path / "demo.json"
    roi = tmp_path / "roi.json"
    demo.write_text('{"demo_id":"bmw"}\n', encoding="utf-8")
    roi.write_text('{"profile_id":"bmw-right"}\n', encoding="utf-8")
    return run_dir, release, demo, roi


def test_snapshot_is_stable_and_does_not_hash_linked_dataset_images(tmp_path: Path) -> None:
    module = _load_module()
    run_dir, release, demo, roi = _fixture(tmp_path)

    first = module.build_snapshot(run_dir, release, demo, roi, code_commit="abc123")
    second = module.build_snapshot(run_dir, release, demo, roi, code_commit="abc123")

    assert first == second
    assert first["code_commit"] == "abc123"
    assert first["run_files"][0]["path"] == "efficientad/front/model.ckpt"
    assert {row["path"] for row in first["run_files"]} == {
        "efficientad/front/model.ckpt",
        "run_report.json",
    }
    assert first["training_release"]["symlinks"] == [
        {"path": "image.png", "target": str((tmp_path / "dataset/image.png").resolve())}
    ]
    assert all("customer-image" not in json.dumps(row) for row in first["training_release"]["files"])


def test_snapshot_requires_complete_inputs(tmp_path: Path) -> None:
    module = _load_module()
    run_dir, release, demo, roi = _fixture(tmp_path)
    demo.unlink()

    with pytest.raises(ValueError, match="demo config"):
        module.build_snapshot(run_dir, release, demo, roi, code_commit="abc123")


def test_main_refuses_to_overwrite_receipt(tmp_path: Path) -> None:
    module = _load_module()
    run_dir, release, demo, roi = _fixture(tmp_path)
    output = tmp_path / "snapshot.json"
    output.write_text("existing", encoding="utf-8")

    assert module.main(
        [
            "--run-dir",
            str(run_dir),
            "--training-release",
            str(release),
            "--demo-config",
            str(demo),
            "--roi-config",
            str(roi),
            "--output",
            str(output),
        ]
    ) == 2
    assert output.read_text(encoding="utf-8") == "existing"
