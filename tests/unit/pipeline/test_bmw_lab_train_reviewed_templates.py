"""CLI tests for bilateral reviewed Template training."""

from __future__ import annotations

import runpy
from pathlib import Path


SCRIPT = Path(__file__).parents[3] / "pipeline" / "bmw_lab_train_reviewed_templates.py"


def test_defaults_target_new_template40_assets() -> None:
    module = runpy.run_path(str(SCRIPT))

    args = module["build_parser"]().parse_args([])

    assert args.review_root.name == "bmw_template_40_review_0823_v1"
    assert args.right_output_root.name == "bmw_right_template_40_reviewed_0823_v1"
    assert args.left_output_root.name == "bmw_left_template_40_reviewed_0823_v1"
    assert args.right_output_config.name == "bmw_eight_view_demo_right_0823_template40_v1.json"
    assert args.left_output_config.name == "bmw_eight_view_demo_left_0823_template40_v1.json"


def test_main_runs_right_then_left(tmp_path: Path, capsys) -> None:
    module = runpy.run_path(str(SCRIPT))
    calls = []

    def fake_run_hand(**kwargs):
        calls.append(kwargs)
        return {"hand": kwargs["hand"], "status": "complete"}

    module["main"].__globals__["run_hand"] = fake_run_hand
    status = module["main"](
        [
            "--review-root",
            str(tmp_path / "review"),
            "--right-output-root",
            str(tmp_path / "right-models"),
            "--left-output-root",
            str(tmp_path / "left-models"),
            "--right-output-config",
            str(tmp_path / "right.json"),
            "--left-output-config",
            str(tmp_path / "left.json"),
        ]
    )

    assert status == 0
    assert [call["hand"] for call in calls] == ["right", "left"]
    assert "\"status\": \"complete\"" in capsys.readouterr().out
