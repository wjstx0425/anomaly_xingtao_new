# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the standalone ZS32 classical-operator benchmark."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest
from capture_data.zs32_classical_benchmark import (
    PRIMARY_VIEWS,
    BenchmarkConfig,
    load_crop_manifest,
    measure_timing,
    publish_directory_noreplace,
    run_benchmark,
    timing_statistics,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from capture_data.zs32_classical_benchmark import BenchmarkCase


def _write_image(path: Path, value: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    image[8:56, 12:84] = value
    cv2.line(image, (24, 32), (70, 32), (30, 30, 30), 2)
    assert cv2.imwrite(str(path), image)


def _write_manifest(root: Path, *, relative_paths: bool = False) -> Path:
    manifest = root / "dataset" / "roi" / "crop_manifest.csv"
    manifest.parent.mkdir(parents=True)
    rows: list[dict[str, str]] = []
    for index, view in enumerate((*PRIMARY_VIEWS, "front_secondary", "back_secondary")):
        image = root / "dataset" / "roi" / view / "normal" / f"session_group{index:03d}.png"
        _write_image(image, 140 + index)
        output_path = image.relative_to(root) if relative_paths else image
        rows.append(
            {
                "output_path": str(output_path),
                "resolved_view": view,
                "label": "normal",
                "defect_type": "",
                "session_id": f"session_{index}",
            },
        )
    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def _minimal_config(manifest: Path, output: Path, **overrides: object) -> BenchmarkConfig:
    values: dict[str, object] = {
        "manifest_path": manifest,
        "output_dir": output,
        "views": ("front",),
        "max_normal_per_view": 1,
        "max_defect_per_view": 0,
        "resize_scale": 0.5,
        "warmup_rounds": 0,
        "timing_rounds": 1,
        "timing_samples_per_view": 1,
        "parallel_workers": 2,
        "save_overlays": True,
    }
    values.update(overrides)
    return BenchmarkConfig(**values)


def test_load_crop_manifest_uses_only_primary_views(tmp_path: Path) -> None:
    """Only canonical primary views are loaded from a mixed manifest."""
    manifest = _write_manifest(tmp_path)

    cases = load_crop_manifest(manifest, PRIMARY_VIEWS)

    assert {case.view for case in cases} == set(PRIMARY_VIEWS)


def test_absolute_manifest_resolves_repo_root_relative_output_path(tmp_path: Path) -> None:
    """Repo-relative paths resolve even when the manifest path is absolute."""
    manifest = _write_manifest(tmp_path, relative_paths=True).resolve()

    cases = load_crop_manifest(manifest, ("front",))

    assert cases[0].image_path == (tmp_path / "dataset" / "roi" / "front" / "normal" / "session_group000.png")
    assert cases[0].image_path.is_file()


def test_load_crop_manifest_rejects_bad_contract(tmp_path: Path) -> None:
    """Invalid manifest labels fail validation."""
    manifest = tmp_path / "crop_manifest.csv"
    manifest.write_text(
        "output_path,resolved_view,label,session_id\nmissing.png,front,unknown,s1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="label"):
        load_crop_manifest(manifest, ("front",))


def test_load_crop_manifest_rejects_duplicate_case_identity(tmp_path: Path) -> None:
    """A repeated session-and-filename identity must fail instead of aliasing outputs."""
    manifest = _write_manifest(tmp_path)
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    duplicate = dict(rows[0])
    duplicate["output_path"] = rows[1]["output_path"]
    duplicate["resolved_view"] = "front"
    duplicate["session_id"] = rows[0]["session_id"]
    Path(duplicate["output_path"]).rename(Path(duplicate["output_path"]).with_name(Path(rows[0]["output_path"]).name))
    duplicate["output_path"] = str(Path(duplicate["output_path"]).with_name(Path(rows[0]["output_path"]).name))
    with manifest.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writerow(duplicate)

    with pytest.raises(ValueError, match="Duplicate case identity"):
        load_crop_manifest(manifest, ("front",))


def test_run_benchmark_rejects_duplicate_config_views(tmp_path: Path) -> None:
    """Repeated configured views must not duplicate cases or timing samples."""
    manifest = _write_manifest(tmp_path)

    with pytest.raises(ValueError, match="duplicate"):
        run_benchmark(_minimal_config(manifest, tmp_path / "out", views=("front", "front")))


def test_run_benchmark_refuses_existing_output(tmp_path: Path) -> None:
    """Existing immutable generations are never overwritten."""
    manifest = _write_manifest(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError):
        run_benchmark(_minimal_config(manifest, output))


def test_run_benchmark_publishes_diagnostic_generation(tmp_path: Path) -> None:
    """A successful run publishes the complete diagnostic artifact contract."""
    manifest = _write_manifest(tmp_path)
    output = tmp_path / "generation"

    published = run_benchmark(_minimal_config(manifest, output))

    assert published == output
    expected = {
        "predictions.csv",
        "parameters.json",
        "benchmark.json",
        "run_manifest.json",
        "masks",
        "overlays",
    }
    assert expected <= {path.name for path in output.iterdir()}
    run_manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["diagnostic_only"] is True
    assert run_manifest["integrated_with_stage32"] is False
    assert run_manifest["integrated_with_stage18"] is False
    assert run_manifest["integrated_with_dashboard"] is False

    with (output / "predictions.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert {row["operator"] for row in rows} == {"thin_line", "pit_spot"}
    assert "final_status" not in rows[0]
    assert "pred_label" not in rows[0]
    for row in rows:
        assert len(row["source_sha256"]) == 64
        assert row["original_shape"] == "64x96x3"
        assert row["processed_shape"] == "32x48x3"
        assert float(row["resize_scale"]) == pytest.approx(0.5)
        assert float(row["score"]) >= 0.0
        assert int(row["component_count"]) >= 0
        assert (output / row["mask_path"]).is_file()
        assert (output / row["overlay_path"]).is_file()
        assert float(row["decode_resize_seconds"]) >= 0.0
        assert float(row["operator_seconds"]) >= 0.0
        assert float(row["evidence_render_write_seconds"]) >= 0.0

    benchmark = json.loads((output / "benchmark.json").read_text(encoding="utf-8"))
    assert benchmark["formal_rounds"] == 1
    assert benchmark["warmup_rounds"] == 0
    assert benchmark["pure_operator"]["thin_line"]["count"] > 0
    assert benchmark["pure_operator"]["pit_spot"]["count"] > 0
    assert benchmark["serial_wall"]["count"] == 1
    assert benchmark["bounded_parallel_wall"]["count"] == 1
    assert benchmark["process_cpu_seconds"] >= 0.0
    assert benchmark["peak_rss_bytes"] > 0


def test_run_benchmark_cleans_staging_when_operator_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operator failures remove staging data and leave no partial generation."""
    manifest = _write_manifest(tmp_path)
    output = tmp_path / "failed"

    def fail_operator(*_args: object, **_kwargs: object) -> object:
        msg = "operator failed"
        raise RuntimeError(msg)

    monkeypatch.setattr("capture_data.zs32_classical_benchmark.detect_thin_lines", fail_operator)

    with pytest.raises(RuntimeError, match="operator failed"):
        run_benchmark(_minimal_config(manifest, output))

    assert not output.exists()
    assert not list(tmp_path.glob(".failed.staging-*"))


def test_run_benchmark_streams_cases_and_retains_only_bounded_timing_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming should retain no more than the configured timing subset."""
    manifest = _write_manifest(tmp_path)
    output = tmp_path / "streamed"
    observed: dict[str, int] = {}

    def fake_measure(images: Iterable[np.ndarray], _config: BenchmarkConfig) -> dict[str, object]:
        materialized = tuple(images)
        observed["count"] = len(materialized)
        return {
            "formal_rounds": 1,
            "warmup_rounds": 0,
            "pure_operator": {
                "thin_line": timing_statistics([0.1]),
                "pit_spot": timing_statistics([0.2]),
            },
            "serial_wall": timing_statistics([0.3]),
            "bounded_parallel_wall": timing_statistics([0.2]),
            "process_cpu_seconds": 0.1,
            "peak_rss_bytes": 1,
            "parallel_workers": 2,
            "opencv_threads": 1,
        }

    monkeypatch.setattr("capture_data.zs32_classical_benchmark.measure_timing", fake_measure)

    run_benchmark(
        _minimal_config(
            manifest,
            output,
            views=PRIMARY_VIEWS,
            max_normal_per_view=1,
            timing_samples_per_view=1,
        ),
    )

    assert observed["count"] == len(PRIMARY_VIEWS)


def test_publish_directory_noreplace_preserves_racing_target(tmp_path: Path) -> None:
    """Atomic publication must preserve an output owned by a competing process."""
    staging = tmp_path / ".out.staging"
    target = tmp_path / "out"
    staging.mkdir()
    (staging / "new.txt").write_text("new", encoding="utf-8")
    target.mkdir()
    (target / "owner.txt").write_text("racer", encoding="utf-8")

    with pytest.raises(FileExistsError):
        publish_directory_noreplace(staging, target)

    assert (target / "owner.txt").read_text(encoding="utf-8") == "racer"
    assert staging.is_dir()


def test_run_benchmark_never_replaces_target_created_at_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A target appearing after generation build must win the publication race."""
    from capture_data import zs32_classical_benchmark as benchmark_module

    manifest = _write_manifest(tmp_path)
    output = tmp_path / "racing-output"
    real_build = benchmark_module._build_generation  # noqa: SLF001

    def build_then_race(
        staging: Path,
        cases: Sequence[BenchmarkCase],
        manifest_path: Path,
        config: BenchmarkConfig,
    ) -> None:
        real_build(staging, cases, manifest_path, config)
        output.mkdir()
        (output / "owner.txt").write_text("racer", encoding="utf-8")

    monkeypatch.setattr(benchmark_module, "_build_generation", build_then_race)

    with pytest.raises(FileExistsError):
        run_benchmark(_minimal_config(manifest, output))

    assert (output / "owner.txt").read_text(encoding="utf-8") == "racer"
    assert not list(tmp_path.glob(".racing-output.staging-*"))


def test_timing_statistics_returns_expected_percentiles() -> None:
    """Timing summaries expose stable count and percentile fields."""
    statistics = timing_statistics([1.0, 2.0, 3.0, 4.0])

    assert statistics == {
        "count": 4,
        "p50_seconds": 2.5,
        "p95_seconds": pytest.approx(3.85),
        "p99_seconds": pytest.approx(3.97),
        "max_seconds": 4.0,
    }


def test_timing_statistics_requires_a_formal_sample() -> None:
    """Timing summaries reject an empty formal sample."""
    with pytest.raises(ValueError, match="formal"):
        timing_statistics([])


def test_formal_timing_excludes_warmup_and_supports_injected_clocks(tmp_path: Path) -> None:
    """Formal timing excludes warmup calls and accepts deterministic fakes."""
    class InlineExecutor:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> InlineExecutor:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        @staticmethod
        def map(
            function: Callable[[np.ndarray], dict[str, float]],
            images: Iterable[np.ndarray],
        ) -> Iterator[dict[str, float]]:
            return map(function, images)

    calls = 0

    def runner(_image: np.ndarray) -> dict[str, float]:
        nonlocal calls
        calls += 1
        return {"thin_line": 0.1, "pit_spot": 0.2}

    wall_values = iter([10.0, 12.0, 20.0, 21.5])
    cpu_values = iter([5.0, 5.25])
    manifest = _write_manifest(tmp_path)
    config = _minimal_config(
        manifest,
        tmp_path / "unused",
        warmup_rounds=2,
        timing_rounds=1,
    )

    result = measure_timing(
        (np.zeros((4, 4, 3), dtype=np.uint8),),
        config,
        operator_runner=runner,
        clock=lambda: next(wall_values),
        cpu_clock=lambda: next(cpu_values),
        executor_factory=InlineExecutor,
        peak_rss_reader=lambda: 1234,
    )

    assert calls == 4  # two warmups, one serial formal run, one parallel formal run
    assert result["pure_operator"]["thin_line"]["count"] == 1
    assert result["pure_operator"]["pit_spot"]["count"] == 1
    assert result["serial_wall"]["p50_seconds"] == pytest.approx(2.0)
    assert result["bounded_parallel_wall"]["p50_seconds"] == pytest.approx(1.5)
    assert result["process_cpu_seconds"] == pytest.approx(0.25)
    assert result["peak_rss_bytes"] == 1234
