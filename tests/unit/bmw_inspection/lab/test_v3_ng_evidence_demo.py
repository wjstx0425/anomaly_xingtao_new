"""Contracts for the BMW third-version NG-evidence Demo assets."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "pipeline/bmw_lab_prepare_v3_ng_evidence_demo.py"
CONFIG = REPO_ROOT / "configs/bmw/experiments/bmw_eight_view_demo_v3_ng_evidence.json"
BASE_CONFIG = REPO_ROOT / "configs/bmw/experiments/bmw_eight_view_demo_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("bmw_lab_prepare_v3_ng_evidence_demo", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _threshold_artifact(path: Path, checkpoints: dict[str, Path]) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "bmw.efficientad_normal_only_thresholds/1.0",
        "calibration_source": "normal_test_only",
        "candidate_only": True,
        "default_demo_config_updated": False,
        "demo_only": True,
        "test_used_for_selection": True,
        "source_csv": "/fixed/calibration.csv",
        "source_csv_sha256": "a" * 64,
        "checkpoint_sha256_by_view": {
            view: hashlib.sha256(checkpoints[view].read_bytes()).hexdigest() for view in VIEW_ORDER
        },
        "thresholds": {view: 0.40 + index / 100 for index, view in enumerate(VIEW_ORDER)},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _assets(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    template_root = tmp_path / "template_only/template"
    efficientad_run = tmp_path / "efficientad_candidate"
    yolo_baseline = tmp_path / "morning_baseline"
    checkpoints: dict[str, Path] = {}
    for view in VIEW_ORDER:
        model = template_root / view / "model.json"
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_text("{}", encoding="utf-8")
        checkpoint = efficientad_run / "efficientad" / view / "model.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(view.encode())
        checkpoints[view] = checkpoint
    yolo = yolo_baseline / "yolo/train/weights/best.pt"
    yolo.parent.mkdir(parents=True)
    yolo.write_bytes(b"morning-yolo")
    base_thresholds = tmp_path / "part_thresholds.json"
    _threshold_artifact(base_thresholds, checkpoints)
    return template_root, efficientad_run, yolo_baseline, base_thresholds


def test_prepare_v3_composes_requested_models_and_margin_threshold_asset(tmp_path: Path) -> None:
    template_root, efficientad_run, yolo_baseline, base_thresholds = _assets(tmp_path)
    output = tmp_path / "v3-composite"
    original_config_digest = hashlib.sha256(BASE_CONFIG.read_bytes()).hexdigest()

    report = _module().prepare_v3_ng_evidence_run(
        template_root=template_root,
        efficientad_candidate_run=efficientad_run,
        yolo_baseline_run=yolo_baseline,
        base_threshold_artifact=base_thresholds,
        output_run=output,
    )

    assert (output / "template").resolve() == template_root.resolve()
    assert (output / "efficientad").resolve() == (efficientad_run / "efficientad").resolve()
    assert (output / "yolo").resolve() == (yolo_baseline / "yolo").resolve()
    artifact = output / "efficientad_thresholds_deployment_v3.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    base = json.loads(base_thresholds.read_text(encoding="utf-8"))
    assert payload["base_thresholds"] == base["thresholds"]
    assert payload["threshold_margin"] == pytest.approx(0.05)
    assert payload["thresholds"] == {
        view: pytest.approx(base["thresholds"][view] + 0.05) for view in VIEW_ORDER
    }
    assert payload["checkpoint_sha256_by_view"] == base["checkpoint_sha256_by_view"]
    assert payload["source_csv"] == base["source_csv"]
    assert payload["source_csv_sha256"] == base["source_csv_sha256"]
    assert report["threshold_artifact_sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert hashlib.sha256(BASE_CONFIG.read_bytes()).hexdigest() == original_config_digest
    with pytest.raises(FileExistsError, match="refuse to overwrite"):
        _module().prepare_v3_ng_evidence_run(
            template_root,
            efficientad_run,
            yolo_baseline,
            base_thresholds,
            output,
        )


def test_v3_config_pins_tracked_profile_v3_and_generated_threshold_asset() -> None:
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    candidate = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert candidate["training_run"].endswith("bmw_right_batch_20260810_21_v3_ng_evidence_demo_v1")
    assert candidate["bright_streak"] == {
        "engine": "tracked_profile_v3",
        "config": (
            "../../../results/bmw_lab_one_click/"
            "bmw_right_batch_20260810_21_bright_v3_tracked_v8/report.json"
        ),
        "config_sha256": "6b43690af67702333646fa7a88a2a5053a0afae6d19cb343bf6d3abb193c17d4",
    }
    report = (CONFIG.parent / candidate["bright_streak"]["config"]).resolve()
    assert hashlib.sha256(report.read_bytes()).hexdigest() == candidate["bright_streak"]["config_sha256"]
    assert candidate["yolo"] == base["yolo"]
    assert candidate["trusted_ok_reference"] == {
        "index": (
            "../../../dataset/bmw_trusted_ok_reference/"
            "bmw_right_20260810_21_train_normal_approved_v2/reference_index.json"
        ),
        "index_sha256": "ae7833ab35cbc76cbfef6cfa5163f77ef345d879a6e8e4834fa8cbfcf6023acc",
    }
    artifact = (CONFIG.parent / candidate["efficientad"]["threshold_artifact"]).resolve()
    assert artifact.name == "efficientad_thresholds_deployment_v3.json"
    assert candidate["efficientad"]["threshold_artifact_sha256"] == hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["threshold_margin"] == pytest.approx(0.05)
    assert all(
        payload["thresholds"][view] == pytest.approx(payload["base_thresholds"][view] + 0.05)
        for view in VIEW_ORDER
    )
