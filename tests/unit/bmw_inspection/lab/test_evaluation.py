"""Batch evaluation and profile-comparison tests for BMW laboratory inspection."""

from __future__ import annotations

import csv
import hashlib
import json
import runpy
from dataclasses import replace
from pathlib import Path

import pytest

from bmw_inspection.lab.contracts import BranchEvidence, BranchName, BranchStatus, FinalStatus, ViewId
from bmw_inspection.lab.dataset import MANIFEST_FIELDS
from bmw_inspection.lab.evaluation import EvaluationOutcome, EvaluationPart, evaluate_profiles
from bmw_inspection.lab.publisher import _canonical_json, _json_value


def _manifest(root: Path, parts: list[tuple[str, str, str]]) -> Path:
    """Write six rows per ``(part_id, split, label)`` tuple."""
    rows: list[dict[str, str]] = []
    for part_id, split, label in parts:
        for view_id in ViewId:
            image_path = root / "images" / f"{part_id}__{view_id.value}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(f"{part_id}:{view_id.value}".encode())
            is_defect = label == "defect" and view_id is ViewId.FRONT
            rows.append(
                {
                    "sample_id": part_id,
                    "part_id": part_id,
                    "session_id": f"session-{part_id}",
                    "view_id": view_id.value,
                    "image_path": str(image_path),
                    "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    "label": "defect" if is_defect else "normal",
                    "defect_type": "scratch" if is_defect else "",
                    "x1": "0" if is_defect else "",
                    "y1": "0" if is_defect else "",
                    "x2": "1" if is_defect else "",
                    "y2": "1" if is_defect else "",
                    "split": split,
                },
            )
    path = root / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


class _FakeRuntime:
    def __init__(
        self,
        root: Path,
        predictions: dict[str, FinalStatus],
        *,
        profile_id: str = "profile-a",
        available_branches: frozenset[BranchName] = frozenset(BranchName),
    ) -> None:
        self.root = root
        self.profile_id = profile_id
        self.predictions = predictions
        self.evidence_root = root / "evidence"
        self.required_evidence = {
            branch: (
                frozenset({ViewId.FRONT_LEFT})
                if branch is BranchName.BRIGHT_STREAK
                else frozenset(ViewId)
            )
            for branch in available_branches
        }
        self.inspected: list[str] = []
        self.calibration_calls = 0
        self.written_thresholds: list[dict[str, float]] = []

    def _config_payload(self) -> dict[str, object]:
        enabled = set(self.required_evidence)
        return {
            "path": str(self.root / f"{self.profile_id}.json"),
            "experiment_id": self.profile_id,
            "topology": {"topology_id": "test-topology", "camera_slots": []},
            "capture": {
                "image_width": 1,
                "image_height": 1,
                "exposure": 1.0,
                "gain": 0.0,
                "timeout_ms": 1,
                "warmup_frames": 0,
            },
            "part_rois": {view.value: [0, 0, 1, 1] for view in ViewId},
            "template": {
                "enabled": BranchName.TEMPLATE in enabled,
                "groups": {},
            },
            "bright_streak": {
                "config_paths": {
                    view.value: f"/{view.value}.json"
                    for view in self.required_evidence.get(BranchName.BRIGHT_STREAK, ())
                },
            },
            "yolo": {
                "enabled": BranchName.YOLO in enabled,
                "checkpoint": None,
                "class_name": "defect",
                "candidate_conf": 0.25,
                "final_threshold": 0.5,
            },
            "patchcore": {
                "enabled": BranchName.PATCHCORE in enabled,
                "checkpoints": {},
                "thresholds": {},
            },
            "required_for_ok": [branch.value for branch in enabled],
            "result_root": str(self.evidence_root),
        }

    def inspect(self, part: EvaluationPart) -> EvaluationOutcome:
        self.inspected.append(part.part_id)
        final_status = self.predictions[part.part_id]
        branch_status = BranchStatus.PASS if final_status is FinalStatus.OK else BranchStatus.NG
        evidence = tuple(
            BranchEvidence(
                branch=branch,
                view_id=view_id,
                status=branch_status,
                required_for_ok=True,
                score=0.1 if branch_status is BranchStatus.PASS else 0.9,
                threshold=0.5,
                elapsed_ms=float(index + 1),
                reason="fake evaluation evidence",
                model_id=f"{branch.value}-model",
                artifact_paths={},
            )
            for index, branch in enumerate(sorted(self.required_evidence, key=lambda item: item.value))
            for view_id in sorted(self.required_evidence[branch], key=lambda item: item.value)
        )
        evidence_path = self.evidence_root / self.profile_id / part.part_id
        evidence_path.mkdir(parents=True, exist_ok=True)
        config_payload = self._config_payload()
        config_digest = hashlib.sha256(_canonical_json(config_payload)).hexdigest()
        (evidence_path / "result.json").write_text(
            json.dumps(
                {
                    "capture_set_id": f"evaluation-{part.part_id}",
                    "created_at": "2026-08-05T12:00:00+00:00",
                    "final_status": final_status.value,
                    "reason": "fake evaluation result",
                    "required_complete": True,
                    "triggered_branches": sorted(
                        {
                            item.branch.value
                            for item in evidence
                            if item.status is BranchStatus.NG
                        },
                    ),
                    "config_digest": config_digest,
                    "evidence": _json_value(evidence),
                },
            ),
            encoding="utf-8",
        )
        (evidence_path / "config.snapshot.json").write_text(
            json.dumps({"config_digest": config_digest, "config": config_payload}),
            encoding="utf-8",
        )
        return EvaluationOutcome(final_status=final_status, evidence=evidence, evidence_path=evidence_path)

    def calibration_thresholds(self, parts: tuple[EvaluationPart, ...]) -> dict[str, float]:
        self.calibration_calls += 1
        assert parts
        return {"yolo.final_threshold": 0.42}

    def write_thresholds(self, thresholds: dict[str, float]) -> None:
        self.written_thresholds.append(dict(thresholds))

    def snapshot_thresholds(self) -> list[dict[str, float]]:
        return list(self.written_thresholds)

    def restore_thresholds(self, snapshot: list[dict[str, float]]) -> None:
        self.written_thresholds = list(snapshot)


def test_evaluation_reports_one_prediction_per_physical_part_and_complete_artifacts(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        [
            ("normal-ok", "final_test", "normal"),
            ("normal-ng", "final_test", "normal"),
            ("defect-ng", "final_test", "defect"),
            ("defect-ok", "final_test", "defect"),
        ],
    )
    runtime = _FakeRuntime(
        tmp_path,
        {
            "normal-ok": FinalStatus.OK,
            "normal-ng": FinalStatus.NG_ANOMALY,
            "defect-ng": FinalStatus.NG_YOLO,
            "defect-ok": FinalStatus.OK,
        },
    )

    summary = evaluate_profiles(
        manifest_path=manifest,
        profiles={"profile-a": runtime},
        split="final_test",
        output_root=tmp_path / "evaluation",
    )

    prediction_rows = list(csv.DictReader((tmp_path / "evaluation/predictions.csv").open()))
    assert len(prediction_rows) == 4
    assert {row["part_id"] for row in prediction_rows} == {
        "normal-ok",
        "normal-ng",
        "defect-ng",
        "defect-ok",
    }
    profile = summary["profiles"]["profile-a"]
    assert profile["partial_system_metrics"] == {
        "part_count": 4,
        "evaluated_binary_count": 4,
        "accuracy": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
    assert profile["experimental_only"] is True
    assert profile["trusted_profile_contract"] is False
    assert profile["missing_branches"] == []
    assert set(profile["per_branch_metrics"]) == {branch.value for branch in BranchName}
    assert profile["per_branch_metrics"]["yolo"]["classification_metrics"]["accuracy"] == 0.5
    assert set(profile["latency_ms"]) == {"p50", "p95", "p99"}
    assert (tmp_path / "evaluation/confusion.csv").is_file()
    assert (tmp_path / "evaluation/branch_metrics.csv").is_file()
    false_ok = json.loads((tmp_path / "evaluation/false_ok/profile-a/defect-ok.json").read_text())
    false_ng = json.loads((tmp_path / "evaluation/false_ng/profile-a/normal-ng.json").read_text())
    assert Path(false_ok["evidence_path"]).name == "defect-ok"
    assert Path(false_ng["evidence_path"]).name == "normal-ng"
    defect_ok = next(row for row in prediction_rows if row["part_id"] == "defect-ok")
    result_path = Path(defect_ok["evidence_path"]) / "result.json"
    assert defect_ok["evidence_result_sha256"] == hashlib.sha256(result_path.read_bytes()).hexdigest()


def test_final_test_is_read_only_and_never_requests_threshold_fitting(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [("part-1", "final_test", "normal")])
    runtime = _FakeRuntime(tmp_path, {"part-1": FinalStatus.OK})

    evaluate_profiles(
        manifest_path=manifest,
        profiles={"profile-a": runtime},
        split="final_test",
        output_root=tmp_path / "read-only",
    )

    assert runtime.calibration_calls == 0
    assert runtime.written_thresholds == []
    with pytest.raises(ValueError, match="final_test.*read-only"):
        evaluate_profiles(
            manifest_path=manifest,
            profiles={"profile-a": runtime},
            split="final_test",
            output_root=tmp_path / "forbidden",
            write_thresholds=True,
        )
    assert not (tmp_path / "forbidden").exists()


def test_calibration_may_report_or_explicitly_write_candidate_thresholds(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [("part-1", "calibration", "normal")])
    report_only = _FakeRuntime(tmp_path, {"part-1": FinalStatus.OK}, profile_id="report")
    writer = _FakeRuntime(tmp_path, {"part-1": FinalStatus.OK}, profile_id="writer")

    first = evaluate_profiles(
        manifest_path=manifest,
        profiles={"report": report_only},
        split="calibration",
        output_root=tmp_path / "report",
    )
    second = evaluate_profiles(
        manifest_path=manifest,
        profiles={"writer": writer},
        split="calibration",
        output_root=tmp_path / "write",
        write_thresholds=True,
    )

    assert first["profiles"]["report"]["candidate_thresholds"] == {"yolo.final_threshold": 0.42}
    assert report_only.written_thresholds == []
    assert writer.written_thresholds == [{"yolo.final_threshold": 0.42}]
    assert second["thresholds_written"] is True


def test_compare_profiles_keeps_outputs_and_metrics_separate(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [("part-1", "final_test", "normal")])
    first = _FakeRuntime(tmp_path, {"part-1": FinalStatus.OK})
    second = _FakeRuntime(tmp_path, {"part-1": FinalStatus.NG_TEMPLATE}, profile_id="profile-b")

    summary = evaluate_profiles(
        manifest_path=manifest,
        profiles={"profile-a": first, "profile-b": second},
        split="final_test",
        output_root=tmp_path / "comparison",
    )

    assert summary["comparison"] is True
    assert tuple(summary["profiles"]) == ("profile-a", "profile-b")
    rows = list(csv.DictReader((tmp_path / "comparison/predictions.csv").open()))
    assert [(row["profile_id"], row["predicted_label"]) for row in rows] == [
        ("profile-a", "normal"),
        ("profile-b", "defect"),
    ]


def test_partial_profile_is_explicitly_experimental_and_has_no_fused_metrics_claim(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [("part-1", "final_test", "normal")])
    runtime = _FakeRuntime(
        tmp_path,
        {"part-1": FinalStatus.OK},
        profile_id="partial",
        available_branches=frozenset({BranchName.TEMPLATE, BranchName.BRIGHT_STREAK}),
    )

    summary = evaluate_profiles(
        manifest_path=manifest,
        profiles={"partial": runtime},
        split="final_test",
        output_root=tmp_path / "partial",
    )

    profile = summary["profiles"]["partial"]
    assert summary["experimental_only"] is True
    assert profile["experimental_only"] is True
    assert profile["missing_branches"] == ["patchcore", "yolo"]
    assert "fused_metrics" not in profile
    assert "partial_system_metrics" in profile


def test_same_part_split_leakage_is_rejected_before_runtime_calls(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        [("part-1", "calibration", "normal"), ("part-1", "final_test", "normal")],
    )
    runtime = _FakeRuntime(tmp_path, {"part-1": FinalStatus.OK})

    with pytest.raises(ValueError, match="split leakage.*part-1"):
        evaluate_profiles(
            manifest_path=manifest,
            profiles={"profile-a": runtime},
            split="final_test",
            output_root=tmp_path / "leaked",
        )

    assert runtime.inspected == []
    assert not (tmp_path / "leaked").exists()


def test_claimed_full_profile_rejects_empty_evidence_and_false_ok(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [("part-1", "final_test", "normal")])

    class EmptyEvidenceRuntime(_FakeRuntime):
        def inspect(self, part: EvaluationPart) -> EvaluationOutcome:
            outcome = super().inspect(part)
            return replace(outcome, evidence=())

    runtime = EmptyEvidenceRuntime(tmp_path, {"part-1": FinalStatus.OK})

    with pytest.raises(ValueError, match="evidence coverage.*template"):
        evaluate_profiles(
            manifest_path=manifest,
            profiles={"profile-a": runtime},
            split="final_test",
            output_root=tmp_path / "empty-evidence",
        )

    assert not (tmp_path / "empty-evidence").exists()


def test_untrusted_runtime_cannot_self_report_a_fused_profile(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [("part-1", "final_test", "normal")])
    runtime = _FakeRuntime(tmp_path, {"part-1": FinalStatus.OK})

    summary = evaluate_profiles(
        manifest_path=manifest,
        profiles={"profile-a": runtime},
        split="final_test",
        output_root=tmp_path / "untrusted",
    )

    profile = summary["profiles"]["profile-a"]
    assert profile["experimental_only"] is True
    assert profile["trusted_profile_contract"] is False
    assert "fused_metrics" not in profile


def test_evidence_pointer_must_be_controlled_and_bound_to_part_profile_and_status(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [("part-1", "final_test", "normal")])

    class WrongIdentityRuntime(_FakeRuntime):
        def inspect(self, part: EvaluationPart) -> EvaluationOutcome:
            outcome = super().inspect(part)
            result_path = outcome.evidence_path / "result.json"
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            payload["capture_set_id"] = "another-part"
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            return outcome

    runtime = WrongIdentityRuntime(tmp_path, {"part-1": FinalStatus.OK})

    with pytest.raises(ValueError, match="capture_set_id.*part-1"):
        evaluate_profiles(
            manifest_path=manifest,
            profiles={"profile-a": runtime},
            split="final_test",
            output_root=tmp_path / "wrong-evidence",
        )


@pytest.mark.parametrize("corruption", ("simplified", "evidence", "config_digest"))
def test_archived_result_and_config_must_fully_match_memory(
    tmp_path: Path,
    corruption: str,
) -> None:
    manifest = _manifest(tmp_path, [("part-1", "final_test", "normal")])

    class CorruptArchiveRuntime(_FakeRuntime):
        def inspect(self, part: EvaluationPart) -> EvaluationOutcome:
            outcome = super().inspect(part)
            result_path = outcome.evidence_path / "result.json"
            snapshot_path = outcome.evidence_path / "config.snapshot.json"
            if corruption == "simplified":
                result_path.write_text(
                    json.dumps(
                        {
                            "capture_set_id": f"evaluation-{part.part_id}",
                            "final_status": outcome.final_status.value,
                        },
                    ),
                    encoding="utf-8",
                )
            elif corruption == "evidence":
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                payload["evidence"][0]["status"] = "NG"
                result_path.write_text(json.dumps(payload), encoding="utf-8")
            else:
                payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
                payload["config"]["capture"]["image_width"] = 999
                snapshot_path.write_text(json.dumps(payload), encoding="utf-8")
            return outcome

    runtime = CorruptArchiveRuntime(tmp_path, {"part-1": FinalStatus.OK})

    with pytest.raises(ValueError, match="archive|evidence|config_digest|config snapshot"):
        evaluate_profiles(
            manifest_path=manifest,
            profiles={"profile-a": runtime},
            split="final_test",
            output_root=tmp_path / f"corrupt-archive-{corruption}",
        )


def test_branch_classification_aggregates_all_views_once_per_part(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [("part-1", "final_test", "defect")])

    class MixedViewRuntime(_FakeRuntime):
        def inspect(self, part: EvaluationPart) -> EvaluationOutcome:
            outcome = super().inspect(part)
            evidence = tuple(
                replace(
                    item,
                    status=(
                        BranchStatus.NG
                        if item.branch is BranchName.YOLO and item.view_id is ViewId.FRONT
                        else BranchStatus.PASS
                    ),
                )
                for item in outcome.evidence
            )
            result_path = outcome.evidence_path / "result.json"
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            payload["evidence"] = _json_value(evidence)
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            return replace(outcome, evidence=evidence)

    runtime = MixedViewRuntime(tmp_path, {"part-1": FinalStatus.NG_YOLO})

    summary = evaluate_profiles(
        manifest_path=manifest,
        profiles={"profile-a": runtime},
        split="final_test",
        output_root=tmp_path / "part-branch",
    )

    metrics = summary["profiles"]["profile-a"]["per_branch_metrics"]["yolo"]["classification_metrics"]
    assert metrics["part_count"] == 1
    assert metrics["evaluated_binary_count"] == 1
    assert metrics["accuracy"] == 1.0
    branch = summary["profiles"]["profile-a"]["per_branch_metrics"]["yolo"]
    assert branch["part_count"] == 1
    assert branch["diagnostic_view_evidence_count"] == 6
    assert branch["status_counts"] == {
        "PASS": 0,
        "NG": 1,
        "SKIPPED": 0,
        "REVIEW": 0,
        "ERROR": 0,
    }
    assert branch["part_latency_total_ms"] == {"p50": 24.0, "p95": 24.0, "p99": 24.0}
    assert branch["part_latency_max_view_ms"] == {"p50": 4.0, "p95": 4.0, "p99": 4.0}
    assert "evidence_count" not in branch


@pytest.mark.parametrize("corruption", ("sample_id", "image_sha256"))
def test_manifest_capture_identity_and_image_digest_are_verified_before_runtime(
    tmp_path: Path,
    corruption: str,
) -> None:
    manifest = _manifest(tmp_path, [("part-1", "final_test", "normal")])
    if corruption == "sample_id":
        rows = list(csv.DictReader(manifest.open()))
        rows[0]["sample_id"] = "different-capture"
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    else:
        image_path = tmp_path / "images/part-1__front.png"
        image_path.write_bytes(b"tampered")
    runtime = _FakeRuntime(tmp_path, {"part-1": FinalStatus.OK})

    with pytest.raises(ValueError, match="sample_id|sha256"):
        evaluate_profiles(
            manifest_path=manifest,
            profiles={"profile-a": runtime},
            split="final_test",
            output_root=tmp_path / f"corrupt-{corruption}",
        )

    assert runtime.inspected == []


def test_sample_id_is_globally_bound_to_one_physical_part(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        [("part-1", "calibration", "normal"), ("part-2", "final_test", "normal")],
    )
    rows = list(csv.DictReader(manifest.open()))
    for row in rows:
        row["sample_id"] = "shared-capture"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    runtime = _FakeRuntime(tmp_path, {"part-2": FinalStatus.OK})

    with pytest.raises(ValueError, match="sample_id.*multiple physical parts"):
        evaluate_profiles(
            manifest_path=manifest,
            profiles={"profile-a": runtime},
            split="final_test",
            output_root=tmp_path / "shared-sample",
        )

    assert runtime.inspected == []


@pytest.mark.parametrize("corruption", ("image_path", "image_sha256"))
def test_each_six_view_sample_requires_six_distinct_images(
    tmp_path: Path,
    corruption: str,
) -> None:
    manifest = _manifest(tmp_path, [("part-1", "final_test", "normal")])
    rows = list(csv.DictReader(manifest.open()))
    if corruption == "image_path":
        for row in rows[1:]:
            row["image_path"] = rows[0]["image_path"]
            row["image_sha256"] = rows[0]["image_sha256"]
    else:
        shared = b"same-pixels-for-all-views"
        shared_digest = hashlib.sha256(shared).hexdigest()
        for row in rows:
            Path(row["image_path"]).write_bytes(shared)
            row["image_sha256"] = shared_digest
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    runtime = _FakeRuntime(tmp_path, {"part-1": FinalStatus.OK})

    with pytest.raises(ValueError, match=f"six distinct {corruption}"):
        evaluate_profiles(
            manifest_path=manifest,
            profiles={"profile-a": runtime},
            split="final_test",
            output_root=tmp_path / f"duplicate-{corruption}",
        )

    assert runtime.inspected == []


def test_threshold_writes_roll_back_if_any_profile_or_publish_step_fails(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [("part-1", "calibration", "normal")])
    first = _FakeRuntime(tmp_path, {"part-1": FinalStatus.OK}, profile_id="first")

    class FailingWriter(_FakeRuntime):
        def write_thresholds(self, thresholds: dict[str, float]) -> None:
            super().write_thresholds(thresholds)
            raise RuntimeError("threshold write failed")

    second = FailingWriter(tmp_path, {"part-1": FinalStatus.OK}, profile_id="second")

    with pytest.raises(RuntimeError, match="threshold write failed"):
        evaluate_profiles(
            manifest_path=manifest,
            profiles={"first": first, "second": second},
            split="calibration",
            output_root=tmp_path / "rollback",
            write_thresholds=True,
        )

    assert first.written_thresholds == []
    assert second.written_thresholds == []
    assert not (tmp_path / "rollback").exists()


def test_cli_supports_compare_and_rejects_final_test_threshold_writes() -> None:
    project_root = Path(__file__).resolve().parents[4]
    namespace = runpy.run_path(str(project_root / "pipeline/bmw_lab_evaluate.py"))
    parser = namespace["build_parser"]()

    args = parser.parse_args(
        [
            "--manifest",
            "manifest.csv",
            "--split",
            "calibration",
            "--compare",
            "profile-a.json",
            "profile-b.json",
            "--output-root",
            "evaluation",
        ],
    )

    assert args.split == "calibration"
    assert args.compare == [Path("profile-a.json"), Path("profile-b.json")]
    config = namespace["load_experiment_config"](
        project_root / "configs/bmw/experiments/bmw_lab_v1.json",
    )
    runtime = namespace["_default_runtime_factory"](config)
    assert runtime.profile_id == config.experiment_id
    assert runtime.required_evidence == {
        BranchName.BRIGHT_STREAK: frozenset({ViewId.FRONT_LEFT}),
    }
    assert runtime.profile_contract.profile_id == config.experiment_id
    assert runtime.profile_contract.required_evidence == runtime.required_evidence
    assert len(runtime.profile_contract.config_digest) == 64
    with pytest.raises(SystemExit):
        namespace["main"](
            [
                "--manifest",
                "missing.csv",
                "--split",
                "final_test",
                "--profile",
                "profile.json",
                "--output-root",
                "evaluation",
                "--write-thresholds",
            ],
        )
