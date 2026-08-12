"""Contract tests for the immutable tracked-profile v3 evaluator."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pytest


def _load_evaluator_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "pipeline/bmw_lab_evaluate_bright_streak_tracked_profile.py"
    spec = importlib.util.spec_from_file_location(
        "bmw_lab_evaluate_bright_streak_tracked_profile",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_roi(path: Path, *, streak: bool) -> None:
    image = np.full((613, 81), 40, dtype=np.uint8)
    if streak:
        centres = 20 + np.arange(613, dtype=np.int64) // 16
        for row, centre in enumerate(centres):
            image[row, centre - 3 : centre + 4] = 220
    assert cv2.imwrite(str(path), image)


def _write_manifest(root: Path) -> Path:
    root.mkdir(parents=True)
    rows = (
        ("cal-normal", "calibration", "OK", True),
        ("cal-no-streak", "calibration", "NG_NO_STREAK", False),
        ("final-normal", "final_test", "OK", True),
        ("final-no-streak", "final_test", "NG_NO_STREAK", False),
    )
    manifest = root / "bright_streak.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("sample_id", "split", "expected_status", "source_path"),
        )
        writer.writeheader()
        for index, (sample_id, split, expected_status, streak) in enumerate(rows):
            image_path = root / f"source-{index}.png"
            _write_roi(image_path, streak=streak)
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "expected_status": expected_status,
                    "source_path": image_path,
                }
            )
    return manifest


def _write_live_record(root: Path, capture_id: str, *, streak: bool = True) -> Path:
    record = root / capture_id
    images = record / "images"
    images.mkdir(parents=True)
    for suffix in ("hdr", "short", "long"):
        _write_roi(images / f"front_left_{suffix}.png", streak=streak)
    (record / "inspection.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "capture_id": capture_id,
                "views": {
                    "front_left": {
                        "source": {
                            "hdr": "images/front_left_hdr.png",
                            "short": "images/front_left_short.png",
                            "long": "images/front_left_long.png",
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return record


def test_tracked_profile_cli_exposes_immutable_evaluation_inputs() -> None:
    module = _load_evaluator_module()

    help_text = module.build_parser().format_help()
    args = module.build_parser().parse_args(
        [
            "--accepted-normal-record",
            "/tmp/accepted-one",
            "--accepted-normal-record",
            "/tmp/accepted-two",
        ]
    )

    assert "--manifest" in help_text
    assert "--output-dir" in help_text
    assert "--accepted-normal-record" in help_text
    assert "--roi-xyxy X1 Y1 X2 Y2" in help_text
    assert args.roi_xyxy == [1792, 1180, 1873, 1793]
    assert args.accepted_normal_record == [Path("/tmp/accepted-one"), Path("/tmp/accepted-two")]
    assert not hasattr(args, "overwrite")


@pytest.mark.parametrize("existing_kind", ["file", "directory", "symlink"])
def test_tracked_profile_evaluator_rejects_any_existing_output_before_inputs(
    tmp_path: Path,
    existing_kind: str,
) -> None:
    module = _load_evaluator_module()
    output = tmp_path / "already-exists"
    if existing_kind == "file":
        output.write_text("existing", encoding="utf-8")
    elif existing_kind == "directory":
        output.mkdir()
    else:
        output.symlink_to(tmp_path / "missing-target", target_is_directory=True)

    with pytest.raises(FileExistsError, match="output path already exists"):
        module.evaluate_bright_streak_tracked_profile(
            tmp_path / "missing-manifest.csv",
            output,
            (),
            (0, 0, 81, 613),
        )


@pytest.mark.parametrize(
    "missing_relative_path",
    [
        "inspection.json",
        "images/front_left_hdr.png",
        "images/front_left_short.png",
        "images/front_left_long.png",
    ],
)
def test_tracked_profile_evaluator_requires_complete_accepted_record(
    tmp_path: Path,
    missing_relative_path: str,
) -> None:
    module = _load_evaluator_module()
    manifest = _write_manifest(tmp_path / "inputs")
    record = _write_live_record(tmp_path / "live", "capture-accepted")
    (record / missing_relative_path).unlink()

    with pytest.raises(ValueError, match="accepted normal record is missing required file"):
        module.evaluate_bright_streak_tracked_profile(
            manifest,
            tmp_path / "output",
            (record,),
            (0, 0, 81, 613),
        )


def test_tracked_profile_evaluator_binds_capture_id_to_record_directory(tmp_path: Path) -> None:
    module = _load_evaluator_module()
    manifest = _write_manifest(tmp_path / "inputs")
    record = _write_live_record(tmp_path / "live", "capture-accepted")
    inspection = json.loads((record / "inspection.json").read_text(encoding="utf-8"))
    inspection["capture_id"] = "different-capture"
    (record / "inspection.json").write_text(json.dumps(inspection), encoding="utf-8")

    with pytest.raises(ValueError, match="inspection capture_id must match record directory"):
        module.evaluate_bright_streak_tracked_profile(
            manifest,
            tmp_path / "output",
            (record,),
            (0, 0, 81, 613),
        )


def test_tracked_profile_report_binds_fit_provenance_and_npz_evidence(tmp_path: Path) -> None:
    module = _load_evaluator_module()
    manifest = _write_manifest(tmp_path / "inputs")
    accepted = _write_live_record(tmp_path / "live", "capture-accepted")
    output = tmp_path / "tracked-v3"

    report = module.evaluate_bright_streak_tracked_profile(
        manifest,
        output,
        (accepted,),
        (0, 0, 81, 613),
    )

    assert report["schema_version"] == 1
    assert report["status"] == "complete"
    assert report["algorithm"] == "tracked_profile_v3"
    assert report["fit_split"] == "calibration"
    assert report["final_test_used_for_fit"] is False
    assert report["real_broken_samples"] == 0
    assert report["roi_xyxy"] == [0, 0, 81, 613]
    assert report["geometry"]["candidate_width"] in {5, 7, 9}
    assert report["geometry_selection"]["final_test_used_for_selection"] is False
    assert report["geometry_selection"]["candidate_widths"] == [5, 7, 9]
    assert len(report["geometry_selection"]["candidates"]) == 3
    assert all(
        set(candidate) >= {
            "candidate_width",
            "calibration_normal_false_rejects",
            "calibration_no_streak_errors",
            "accepted_live_false_rejects",
            "weakest_normal_coverage_ratio",
            "normal_max_gap_ratio",
            "normal_max_gap_count",
        }
        for candidate in report["geometry_selection"]["candidates"]
    )
    assert len(report["geometry_selection"]["selection_input_sha256"]) == 64
    assert len(report["geometry_selection"]["selection_evidence_sha256"]) == 64
    assert report["comparison_to_v2"]["available"] is False
    assert set(report["thresholds"]) == {
        "strong_row_score",
        "weak_row_score",
        "min_presence_coverage_ratio",
        "min_longest_run_ratio",
        "max_gap_ratio",
        "max_gap_count",
    }
    assert report["calibration_counts"] == {
        "manifest_normal": 1,
        "manifest_no_streak": 1,
        "accepted_live_normal": 1,
        "fit_total": 3,
    }
    assert report["final_test"]["count"] == 2
    assert len(report["final_test"]["outcomes"]) == 2
    assert report["final_test"]["normal_false_rejects"] == 0
    assert report["final_test"]["no_streak_false_accepts"] == 0
    accepted_report = report["accepted_live_normals"][0]
    assert accepted_report["capture_id"] == "capture-accepted"
    assert accepted_report["provenance_kind"] == "user_confirmed_live_normal"
    assert accepted_report["predicted_status"] == "OK"
    assert set(accepted_report["file_sha256"]) == {
        "inspection.json",
        "front_left_hdr.png",
        "front_left_short.png",
        "front_left_long.png",
    }
    assert set(report["identities"]) == {
        "manifest_sha256",
        "roi_config_sha256",
        "algorithm_source_sha256",
        "evaluator_source_sha256",
    }
    assert report["acceptance_gate"]["passed"] is True
    assert set(report["artifact_identities"]) == {
        "metrics_csv_sha256",
        "replay_summary_sha256",
        "profile_npz_sha256",
    }
    assert len(report["artifact_identities"]["profile_npz_sha256"]) == 5
    assert report["cpu_per_image_ms"]["count"] == 5
    assert report["cpu_per_image_ms"]["p50"] >= 0.0
    assert report["cpu_per_image_ms"]["max"] >= report["cpu_per_image_ms"]["p50"]
    assert Path(report["metrics_csv"]).is_file()
    assert Path(report["replay_summary_json"]).is_file()
    assert json.loads(Path(report["report_json"]).read_text(encoding="utf-8")) == report

    metric_rows = list(csv.DictReader(Path(report["metrics_csv"]).open(encoding="utf-8")))
    assert len(metric_rows) == 5
    with np.load(metric_rows[0]["profile_npz"]) as profile:
        assert set(profile.files) == {
            "response_map",
            "path_x",
            "path_scores",
            "strong_mask",
            "accepted_mask",
            "bridged_mask",
        }
        edge = report["geometry"]["candidate_width"] // 2 + 3 + 10
        assert profile["response_map"].shape == (613, 81 - 2 * edge)
        assert profile["path_x"].shape == (613,)


@pytest.mark.parametrize(
    ("accepted", "no_streak", "comparison", "message"),
    [
        ([{"capture_id": "bad", "predicted_status": "NG_BROKEN"}], [], {}, "confirmed live normal"),
        ([], [{"sample_id": "bad", "predicted_status": "OK"}], {}, "no-streak"),
        (
            [],
            [],
            {
                "calibration": {
                    "normal_count": 1,
                    "v2_normal_false_rejects": 0,
                    "v3_normal_false_rejects": 1,
                }
            },
            "normal false rejects",
        ),
    ],
)
def test_tracked_profile_acceptance_gate_fails_closed_before_publication(
    accepted: list[dict[str, object]],
    no_streak: list[dict[str, object]],
    comparison: dict[str, object],
    message: str,
) -> None:
    module = _load_evaluator_module()

    with pytest.raises(RuntimeError, match=message):
        module._enforce_acceptance_gate(accepted, no_streak, comparison)


def test_tracked_profile_replays_unconfirmed_recent_records_as_unknown_truth(
    tmp_path: Path,
) -> None:
    module = _load_evaluator_module()
    manifest = _write_manifest(tmp_path / "inputs")
    accepted = _write_live_record(tmp_path / "live", "capture-001")
    unknown = _write_live_record(tmp_path / "live", "capture-002")

    report = module.evaluate_bright_streak_tracked_profile(
        manifest,
        tmp_path / "tracked-v3",
        (accepted,),
        (0, 0, 81, 613),
    )

    replay = report["replay"]
    unknown_outcome = next(
        outcome for outcome in replay["outcomes"] if outcome["capture_id"] == unknown.name
    )
    assert unknown_outcome["truth"] == "unknown"
    assert unknown_outcome["included_in_accuracy"] is False
    assert "correct" not in unknown_outcome
    assert replay["known_truth_count"] == 1
    assert replay["unknown_truth_count"] == 1
    assert replay["accuracy_denominator"] == 1


def test_tracked_profile_keeps_confirmed_truth_when_record_arrives_through_symlink(
    tmp_path: Path,
) -> None:
    module = _load_evaluator_module()
    manifest = _write_manifest(tmp_path / "inputs")
    real_live = tmp_path / "real-live"
    _write_live_record(real_live, "capture-001")
    linked_live = tmp_path / "linked-live"
    linked_live.symlink_to(real_live, target_is_directory=True)

    report = module.evaluate_bright_streak_tracked_profile(
        manifest,
        tmp_path / "tracked-v3",
        (linked_live / "capture-001",),
        (0, 0, 81, 613),
    )

    assert report["replay"]["known_truth_count"] == 1
    assert report["replay"]["unknown_truth_count"] == 0
    assert report["replay"]["outcomes"][0]["truth"] == "normal"
