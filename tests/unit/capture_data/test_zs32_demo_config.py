# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the editable ZS32 Demo configuration."""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

VIEW_ORDER = (
    "front",
    "front_left",
    "front_right",
    "front_secondary",
    "back",
    "back_left",
    "back_right",
    "back_secondary",
)


def _load_demo_config(path: Path, *, repo_root: Path | None = None) -> Any:
    module = importlib.import_module("capture_data.zs32_demo_config")
    return module.load_demo_config(path, repo_root=repo_root)


@pytest.fixture
def demo_repository(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repository"
    topology = root / "configs/zs32/topology.json"
    roi_config = root / "dataset/roi.json"
    template_dir = root / "results/template"
    yolo_weights = root / "results/yolo/best.pt"
    config_path = root / "configs/zs32/zs32_demo.json"

    topology.parent.mkdir(parents=True)
    topology.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "camera_slots": [
                    {"serial": "1", "views": {"front": "front", "back": "back"}},
                    {"serial": "2", "views": {"front": "front_left", "back": "back_left"}},
                    {"serial": "3", "views": {"front": "front_right", "back": "back_right"}},
                    {
                        "serial": "4",
                        "views": {"front": "front_secondary", "back": "back_secondary"},
                    },
                ],
                "required_views": list(VIEW_ORDER),
            },
        ),
        encoding="utf-8",
    )
    roi_config.parent.mkdir(parents=True)
    roi_config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "image_size": {"width": 100, "height": 80},
                "views": {view: {"roi": [1, 2, 90, 70]} for view in VIEW_ORDER},
            },
        ),
        encoding="utf-8",
    )
    template_dir.mkdir(parents=True)
    yolo_weights.parent.mkdir(parents=True)
    yolo_weights.write_bytes(b"weights")
    checkpoints: dict[str, str] = {}
    for view in VIEW_ORDER:
        checkpoint = root / "results/patchcore" / f"{view}.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(view.encode())
        checkpoints[view] = str(checkpoint.relative_to(root))

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "demo",
                "patchcore_process_count": 1,
                "topology": str(topology.relative_to(root)),
                "roi_config": str(roi_config.relative_to(root)),
                "models": {
                    "template_dir": str(template_dir.relative_to(root)),
                    "patchcore": checkpoints,
                    "yolo": {
                        "weights": str(yolo_weights.relative_to(root)),
                        "imgsz": 1280,
                        "candidate_conf": 0.001,
                    },
                },
                "thresholds": {
                    branch: {view: 0.5 for view in VIEW_ORDER}
                    for branch in ("template", "patchcore", "yolo")
                },
            },
        ),
        encoding="utf-8",
    )
    return root, config_path


def _mutate_json(path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_valid_config_loads_explicit_dataclasses_and_repo_relative_paths(
    demo_repository: tuple[Path, Path],
) -> None:
    root, config_path = demo_repository

    config = _load_demo_config(config_path.relative_to(root), repo_root=root)

    assert config.path == config_path.resolve()
    assert config.topology == (root / "configs/zs32/topology.json").resolve()
    assert config.roi_config == (root / "dataset/roi.json").resolve()
    assert config.template_dir == (root / "results/template").resolve()
    assert tuple(config.patchcore) == VIEW_ORDER
    assert config.patchcore["front_secondary"].is_file()
    assert config.patchcore["back_secondary"].is_file()
    assert config.yolo.weights == (root / "results/yolo/best.pt").resolve()
    assert config.yolo.imgsz == 1280
    assert config.yolo.candidate_conf == pytest.approx(0.001)
    assert config.patchcore_process_count == 1
    assert tuple(config.thresholds.template) == VIEW_ORDER
    assert tuple(config.thresholds.patchcore) == VIEW_ORDER
    assert tuple(config.thresholds.yolo) == VIEW_ORDER
    assert type(config).__name__ == "DemoConfig"
    assert type(config.yolo).__name__ == "YoloConfig"
    assert type(config.thresholds).__name__ == "DemoThresholds"


def test_repository_config_loads_all_real_assets() -> None:
    config = _load_demo_config(Path("configs/zs32/zs32_demo.json"))
    repo_root = Path.cwd().resolve()
    template_root = repo_root / "results/zs32_template_gate_right_0727_plus_defect_eight_view_v14"
    patchcore_root = repo_root / "results/zs32_patchcore_eight_view_0727_plus_defect_seed42_v14"
    expected_template_thresholds = {
        "front": 0.012784421443939209,
        "front_left": 0.17422431707382202,
        "front_right": 0.273634672164917,
        "front_secondary": 0.10163009166717529,
        "back": 0.017854928970336914,
        "back_left": 0.16238605976104736,
        "back_right": 0.21661347150802612,
        "back_secondary": 0.1012694239616394,
    }
    expected_patchcore_thresholds = {
        "front": 0.4700116217136383,
        "front_left": 0.1211932897567749,
        "front_right": 0.025172829627990723,
        "front_secondary": 0.1191740334033966,
        "back": 0.5475995540618896,
        "back_left": 0.11428269743919373,
        "back_right": 0.08657169342041016,
        "back_secondary": 0.230668842792511,
    }

    assert tuple(config.patchcore) == VIEW_ORDER
    assert all(path.is_file() for path in config.patchcore.values())
    assert config.template_dir == template_root
    assert all(path.is_relative_to(patchcore_root) for path in config.patchcore.values())
    assert config.yolo.weights.is_file()
    assert config.patchcore_process_count == 8
    assert config.thresholds.template == expected_template_thresholds
    assert config.thresholds.patchcore == expected_patchcore_thresholds


@pytest.mark.parametrize("process_count", [1, 2, 4, 8])
def test_patchcore_process_count_accepts_supported_candidates(
    demo_repository: tuple[Path, Path],
    process_count: int,
) -> None:
    root, config_path = demo_repository
    _mutate_json(
        config_path,
        lambda payload: payload.__setitem__("patchcore_process_count", process_count),
    )

    config = _load_demo_config(config_path, repo_root=root)

    assert config.patchcore_process_count == process_count


@pytest.mark.parametrize("process_count", [0, 3, 5, 9, True, 2.0, "2", None])
def test_patchcore_process_count_rejects_unsupported_values(
    demo_repository: tuple[Path, Path],
    process_count: object,
) -> None:
    root, config_path = demo_repository
    _mutate_json(
        config_path,
        lambda payload: payload.__setitem__("patchcore_process_count", process_count),
    )

    with pytest.raises((TypeError, ValueError), match="patchcore_process_count"):
        _load_demo_config(config_path, repo_root=root)


@pytest.mark.parametrize("extra", [False, True], ids=["missing", "extra"])
def test_patchcore_requires_exact_canonical_views(
    demo_repository: tuple[Path, Path],
    extra: bool,
) -> None:
    root, config_path = demo_repository

    def mutate(payload: dict[str, Any]) -> None:
        patchcore = payload["models"]["patchcore"]
        if extra:
            patchcore["unsupported"] = patchcore["front"]
        else:
            del patchcore["front_left"]

    _mutate_json(config_path, mutate)

    with pytest.raises(ValueError, match="patchcore.*canonical eight views"):
        _load_demo_config(config_path, repo_root=root)


def test_secondary_views_are_required_by_topology(demo_repository: tuple[Path, Path]) -> None:
    root, config_path = demo_repository
    topology = root / "configs/zs32/topology.json"

    def mutate(payload: dict[str, Any]) -> None:
        payload["camera_slots"][-1]["views"]["front"] = "front"

    _mutate_json(topology, mutate)

    with pytest.raises(ValueError, match="topology.*secondary"):
        _load_demo_config(config_path, repo_root=root)


@pytest.mark.parametrize(
    ("branch", "value"),
    [
        ("template", -0.01),
        ("template", 2.01),
        ("patchcore", -0.01),
        ("patchcore", 2.01),
        ("yolo", -0.01),
        ("yolo", 1.01),
        ("template", float("nan")),
        ("patchcore", float("inf")),
    ],
)
def test_thresholds_must_be_finite_and_in_branch_range(
    demo_repository: tuple[Path, Path],
    branch: str,
    value: float,
) -> None:
    root, config_path = demo_repository
    _mutate_json(config_path, lambda payload: payload["thresholds"][branch].__setitem__("front", value))

    expected = branch if math.isfinite(value) else "non-finite JSON constant"
    with pytest.raises(ValueError, match=expected):
        _load_demo_config(config_path, repo_root=root)


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf")])
def test_yolo_candidate_conf_must_be_finite_probability(
    demo_repository: tuple[Path, Path],
    value: float,
) -> None:
    root, config_path = demo_repository
    _mutate_json(config_path, lambda payload: payload["models"]["yolo"].__setitem__("candidate_conf", value))

    expected = "candidate_conf" if math.isfinite(value) else "non-finite JSON constant"
    with pytest.raises(ValueError, match=expected):
        _load_demo_config(config_path, repo_root=root)


@pytest.mark.parametrize("target", ["roi", "template", "patchcore", "yolo"])
def test_declared_asset_paths_must_exist(
    demo_repository: tuple[Path, Path],
    target: str,
) -> None:
    root, config_path = demo_repository

    def mutate(payload: dict[str, Any]) -> None:
        if target == "roi":
            payload["roi_config"] = "missing/roi.json"
        elif target == "template":
            payload["models"]["template_dir"] = "missing/template"
        elif target == "patchcore":
            payload["models"]["patchcore"]["back_secondary"] = "missing/model.ckpt"
        else:
            payload["models"]["yolo"]["weights"] = "missing/best.pt"

    _mutate_json(config_path, mutate)

    with pytest.raises(FileNotFoundError, match=target):
        _load_demo_config(config_path, repo_root=root)


def test_roi_requires_exact_views_and_valid_bounded_coordinates(
    demo_repository: tuple[Path, Path],
) -> None:
    root, config_path = demo_repository
    roi_config = root / "dataset/roi.json"
    _mutate_json(roi_config, lambda payload: payload["views"]["back_secondary"].__setitem__("roi", [1, 2, 101, 70]))

    with pytest.raises(ValueError, match="ROI.*back_secondary"):
        _load_demo_config(config_path, repo_root=root)


def test_duplicate_json_keys_are_rejected(demo_repository: tuple[Path, Path]) -> None:
    root, config_path = demo_repository
    text = config_path.read_text(encoding="utf-8")
    config_path.write_text(text.replace('"mode": "demo"', '"mode": "demo", "mode": "demo"', 1), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key.*mode"):
        _load_demo_config(config_path, repo_root=root)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_standard_json_constants_are_rejected(
    demo_repository: tuple[Path, Path],
    constant: str,
) -> None:
    root, config_path = demo_repository
    text = config_path.read_text(encoding="utf-8")
    mutated = text.replace('"candidate_conf": 0.001', f'"candidate_conf": {constant}', 1)
    config_path.write_text(mutated, encoding="utf-8")

    with pytest.raises(ValueError, match="non-finite JSON constant"):
        _load_demo_config(config_path, repo_root=root)
