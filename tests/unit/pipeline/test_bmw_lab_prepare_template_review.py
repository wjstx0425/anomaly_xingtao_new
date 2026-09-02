# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CLI tests for the BMW Template review-package generator."""

from __future__ import annotations

import json
import runpy
from pathlib import Path


SCRIPT = Path(__file__).parents[3] / "pipeline" / "bmw_lab_prepare_template_review.py"


def test_parser_defaults_to_40_candidates() -> None:
    module = runpy.run_path(str(SCRIPT))

    args = module["build_parser"]().parse_args([])

    assert args.candidate_count == 40
    assert args.output_root.name == "bmw_template_40_review_0823_v1"


def test_main_passes_both_hands_to_package_builder(tmp_path: Path, capsys) -> None:
    module = runpy.run_path(str(SCRIPT))
    paths = {}
    for hand in ("right", "left"):
        paths[f"{hand}_manifest"] = tmp_path / f"{hand}.csv"
        paths[f"{hand}_roi"] = tmp_path / f"{hand}.json"
        paths[f"{hand}_manifest"].touch()
        paths[f"{hand}_roi"].touch()
    output = tmp_path / "review"
    captured = {}

    def fake_build(sources, output_root, *, candidate_count):
        captured["sources"] = sources
        captured["output_root"] = output_root
        captured["candidate_count"] = candidate_count
        return Path(output_root).resolve()

    module["main"].__globals__["build_template_review_package"] = fake_build
    status = module["main"](
        [
            "--right-manifest",
            str(paths["right_manifest"]),
            "--right-roi-config",
            str(paths["right_roi"]),
            "--left-manifest",
            str(paths["left_manifest"]),
            "--left-roi-config",
            str(paths["left_roi"]),
            "--output-root",
            str(output),
        ]
    )

    assert status == 0
    assert [source.hand for source in captured["sources"]] == ["right", "left"]
    assert captured["candidate_count"] == 40
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "candidate_count_per_view": 40,
        "candidate_total": 640,
        "output_root": str(output.resolve()),
        "status": "REVIEW_PACKAGE_READY",
    }
