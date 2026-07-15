# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unified ZS32 multi-model entrypoint and right-hand profile."""

# The tests intentionally exercise private parser helpers and a harmless temporary path.
# ruff: noqa: S108, SLF001

from __future__ import annotations

import importlib.util
import json
import sys
from hashlib import sha256
from pathlib import Path
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
import pytest
from zs32_inspection.dashboard.contracts import BranchState
from zs32_inspection.dashboard.parser import load_inspection_result

REPO_ROOT = Path(__file__).resolve().parents[3]
VIEWS = (
    "front",
    "front_left",
    "front_right",
    "front_secondary",
    "back",
    "back_left",
    "back_right",
    "back_secondary",
)
LEGACY_VIEWS = ("front", "front_left", "front_right", "back", "back_left", "back_right")


def _load_module(name: str, relative_path: str) -> ModuleType:
    """Load one numbered pipeline module without requiring a package."""
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        msg = f"could not load {path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _minimum_args(tmp_path: Path) -> list[str]:
    """Return the identity and eight-view arguments shared by parser tests."""
    values = [
        "infer",
        "--part-id",
        "part-001",
        "--capture-session",
        "session-001",
        "--group-id",
        "group-001",
        "--output-dir",
        str(tmp_path / "output"),
    ]
    for view in VIEWS:
        values.extend((f"--{view.replace('_', '-')}-image", str(tmp_path / f"{view}.png")))
    return values


def _write_images(tmp_path: Path) -> dict[str, Path]:
    images = {}
    for index, view in enumerate(VIEWS):
        path = tmp_path / f"{view}.png"
        assert cv2.imwrite(str(path), np.full((3, 4, 3), index, dtype=np.uint8))
        images[view] = path
    return images


def test_unified_entrypoint_parses_all_eight_explicit_images(tmp_path: Path) -> None:
    """The CLI must preserve an explicit source path for every canonical view."""
    stage32 = _load_module("pipeline_zs32_runtime_parser", "pipeline/32_run_zs32_multimodel_inference.py")

    args = stage32.build_parser().parse_args(_minimum_args(tmp_path))
    request = stage32._request_from_args(args)

    assert args.mode == "infer"
    assert tuple(request.images) == VIEWS
    assert request.hand == "right"
    assert args.fusion_profile == "zs32-right-24-commissioning"
    assert args.runtime_config == REPO_ROOT / "config/fusion/zs32_runtime_models_eight_view.json"


@pytest.mark.parametrize(
    "forbidden",
    [
        ("--hand", "left"),
        ("--fusion-profile", "zs32-right"),
        ("--fusion-profile", "zs32-right-18-commissioning"),
    ],
)
def test_stage32_rejects_non_right_or_non_eight_view_commissioning_contract(
    tmp_path: Path,
    forbidden: tuple[str, str],
) -> None:
    stage32 = _load_module("pipeline_zs32_reject_legacy_contract", "pipeline/32_run_zs32_multimodel_inference.py")

    with pytest.raises(SystemExit):
        stage32.build_parser().parse_args([*_minimum_args(tmp_path), *forbidden])


def test_stage32_parser_requires_all_eight_images() -> None:
    stage32 = _load_module("pipeline_zs32_required_images", "pipeline/32_run_zs32_multimodel_inference.py")

    required = {action.dest for action in stage32.build_parser()._actions if action.required}

    assert {f"{view}_image" for view in VIEWS} <= required


def test_template_gate_evaluates_all_eight_before_aggregate_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage32 = _load_module("pipeline_zs32_all_template_views", "pipeline/32_run_zs32_multimodel_inference.py")
    images = _write_images(tmp_path)
    calls: list[str] = []

    class FakeGate:
        def __init__(self, _model_dir: Path) -> None:
            pass

        def evaluate(self, crop_path: Path, _hand: str, view: str) -> SimpleNamespace:
            calls.append(view)
            status = "NG_TEMPLATE" if view == "front_left" else "PASS"
            return SimpleNamespace(
                to_dict=lambda: {
                    "view": view,
                    "status": status,
                    "reason": "mismatch" if status == "NG_TEMPLATE" else "",
                    "score": 0.9 if status == "NG_TEMPLATE" else 0.01,
                    "risk_score": 0.9 if status == "NG_TEMPLATE" else 0.01,
                    "low_threshold": 0.1,
                    "high_threshold": 0.5,
                    "best_template_path": str(crop_path),
                },
            )

    monkeypatch.setattr(stage32, "TemplateGate", FakeGate)
    request = stage32.InspectionRequest("part-001", "session-001", "group-001", "right", images)
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            image_width=4,
            image_height=3,
            patchcore_rois={"right": {view: (0, 0, 4, 3) for view in VIEWS}},
        ),
    )

    results, _ = stage32._template_gate(
        request,
        runtime,
        tmp_path / "model",
        tmp_path / "workspace",
        "zs32-right-24-commissioning",
    )

    assert calls == list(VIEWS)
    assert tuple(result["view"] for result in results) == VIEWS


def test_diagnostic_mask_threshold_parses_with_display_only_default(tmp_path: Path) -> None:
    """The CLI must expose the documented display-only default and custom override."""
    stage32 = _load_module("pipeline_zs32_runtime_mask_threshold", "pipeline/32_run_zs32_multimodel_inference.py")

    default_args = stage32.build_parser().parse_args(_minimum_args(tmp_path))
    custom_args = stage32.build_parser().parse_args(
        [*_minimum_args(tmp_path), "--diagnostic-mask-threshold", "0.4"],
    )

    assert default_args.diagnostic_mask_threshold == pytest.approx(0.65)
    assert custom_args.diagnostic_mask_threshold == pytest.approx(0.4)


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -0.01, 1.01])
def test_validate_mode_rejects_invalid_diagnostic_mask_threshold(tmp_path: Path, threshold: float) -> None:
    """The CLI must reject non-finite values and values outside the probability range."""
    stage32 = _load_module(
        "pipeline_zs32_runtime_invalid_mask_threshold",
        "pipeline/32_run_zs32_multimodel_inference.py",
    )
    args = stage32.build_parser().parse_args(
        [*_minimum_args(tmp_path), "--diagnostic-mask-threshold", str(threshold)],
    )

    with pytest.raises(ValueError, match="--diagnostic-mask-threshold"):
        stage32._validate_mode(args)


def test_diagnostic_skip_template_flag_parses(tmp_path: Path) -> None:
    """The diagnostic opt-in must be explicit on the Stage32 CLI."""
    stage32 = _load_module("pipeline_zs32_runtime_diagnostic_parser", "pipeline/32_run_zs32_multimodel_inference.py")

    args = stage32.build_parser().parse_args([*_minimum_args(tmp_path), "--diagnostic-skip-template"])

    assert args.diagnostic_skip_template is True


def test_diagnostic_skip_template_is_rejected_in_fuse_mode(tmp_path: Path) -> None:
    """Strict fusion must never accept the template-skipping diagnostic path."""
    stage32 = _load_module("pipeline_zs32_runtime_diagnostic_fuse", "pipeline/32_run_zs32_multimodel_inference.py")
    values = _minimum_args(tmp_path)
    values[0] = "fuse"
    args = stage32.build_parser().parse_args([*values, "--diagnostic-skip-template"])

    with pytest.raises(ValueError, match="--diagnostic-skip-template"):
        stage32._validate_mode(args)


@pytest.mark.parametrize("conflicting_option", ["--template-model-dir", "--threshold-artifact"])
def test_diagnostic_skip_template_rejects_template_assets(tmp_path: Path, conflicting_option: str) -> None:
    """Diagnostic inference must not silently consume template or threshold assets."""
    stage32 = _load_module(
        f"pipeline_zs32_runtime_diagnostic_conflict_{conflicting_option[2:]}",
        "pipeline/32_run_zs32_multimodel_inference.py",
    )
    args = stage32.build_parser().parse_args(
        [
            *_minimum_args(tmp_path),
            "--diagnostic-skip-template",
            conflicting_option,
            str(tmp_path / "forbidden-asset"),
        ],
    )

    with pytest.raises(ValueError, match="--diagnostic-skip-template"):
        stage32._validate_mode(args)


def test_diagnostic_skip_template_publication_is_explicit_and_non_production(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Diagnostic inference must publish identity and a fail-closed policy on both JSON surfaces."""
    stage32 = _load_module(
        "pipeline_zs32_runtime_diagnostic_publication",
        "pipeline/32_run_zs32_multimodel_inference.py",
    )
    output = tmp_path / "output"

    class FakeRuntime:
        """Publish the minimal ordinary runtime artifacts consumed by Stage32."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        @staticmethod
        def run(_request: object, output_dir: Path, **_kwargs: object) -> SimpleNamespace:
            output_dir.mkdir()
            patchcore_csv = output_dir / "patchcore.csv"
            yolo_csv = output_dir / "yolo.csv"
            patchcore_csv.write_text("branch\n", encoding="utf-8")
            yolo_csv.write_text("branch\n", encoding="utf-8")
            (output_dir / "runtime_manifest.json").write_text(
                json.dumps({"runtime_config_sha256": "a" * 64, "note": "continuous evidence only"}),
                encoding="utf-8",
            )
            return SimpleNamespace(
                machine_status="REVIEW",
                inspection_complete=False,
                patchcore_csv=patchcore_csv,
                yolo_csv=yolo_csv,
                calibration_csv=None,
                errors=(),
                missing_required_evidence=("template_match", "strict_fusion_not_run"),
            )

    monkeypatch.setattr(stage32, "load_runtime_config", lambda _path: SimpleNamespace(supported_hands=("right",)))
    monkeypatch.setattr(stage32, "ZS32ModelRuntime", FakeRuntime)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage32",
            *_minimum_args(tmp_path),
            "--part-id",
            "diagnostic-part",
            "--group-id",
            "group001",
            "--diagnostic-skip-template",
        ],
    )

    stage32.main()

    expected = {
        "part_id": "diagnostic-part",
        "capture_session": "session-001",
        "group_id": "group001",
        "hand": "right",
        "diagnostic_skip_template": True,
        "inspection_complete": False,
        "strict_fusion": False,
        "commissioning_only": True,
        "production_release_allowed": False,
    }
    summary = json.loads((output / "runtime_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "runtime_manifest.json").read_text(encoding="utf-8"))
    assert {field: summary[field] for field in expected} == expected
    assert {field: manifest[field] for field in expected} == expected
    assert Path(summary["patchcore_csv"]).is_absolute()
    assert Path(summary["yolo_csv"]).is_absolute()
    assert summary["patchcore_csv"] == str((output / "patchcore.csv").resolve())
    assert summary["yolo_csv"] == str((output / "yolo.csv").resolve())
    assert "template" in manifest["note"].lower()
    assert "Stage18" in manifest["note"]
    assert "intentionally skipped" in manifest["note"]
    assert not (output / "template_match.csv").exists()
    assert not (output / "crops" / "template").exists()
    assert not (output / "fusion").exists()
    assert not any("audit" in path.name.lower() for path in output.rglob("*"))


def test_fuse_mode_requires_all_non_model_branch_evidence(tmp_path: Path) -> None:
    """Fuse mode must fail before inference when strict evidence is incomplete."""
    stage32 = _load_module("pipeline_zs32_runtime_fuse", "pipeline/32_run_zs32_multimodel_inference.py")
    values = _minimum_args(tmp_path)
    values[0] = "fuse"
    args = stage32.build_parser().parse_args(values)

    with pytest.raises(ValueError, match="--template-model-dir"):
        stage32._validate_mode(args)


def test_commissioning_fuse_requires_only_template_and_threshold_assets(tmp_path: Path) -> None:
    """The explicit 18-group profile must not require unavailable future-stage CSVs."""
    stage32 = _load_module("pipeline_zs32_runtime_commissioning", "pipeline/32_run_zs32_multimodel_inference.py")
    values = _minimum_args(tmp_path)
    values[0] = "fuse"
    values.extend(
        (
            "--fusion-profile",
            "zs32-right-24-commissioning",
            "--threshold-artifact",
            str(tmp_path / "thresholds.json"),
            "--template-model-dir",
            str(tmp_path / "template-model"),
        ),
    )
    args = stage32.build_parser().parse_args(values)

    stage32._validate_mode(args)


def test_validate_mode_rejects_production_namespace_even_if_constructed_directly(tmp_path: Path) -> None:
    """The eight-view Stage32 implementation must never execute a production profile."""
    stage32 = _load_module("pipeline_zs32_runtime_production", "pipeline/32_run_zs32_multimodel_inference.py")
    values = _minimum_args(tmp_path)
    values[0] = "fuse"
    args = stage32.build_parser().parse_args(values)
    args.fusion_profile = "zs32-right"

    with pytest.raises(ValueError, match="24-group commissioning"):
        stage32._validate_mode(args)


@pytest.mark.parametrize(
    ("profile", "status", "expected"),
    [
        ("zs32-right-24-commissioning", "PASS", True),
        ("zs32-right-24-commissioning", "REVIEW", True),
        ("zs32-right-24-commissioning", "NG_TEMPLATE", False),
    ],
)
def test_template_review_continues_only_in_complementary_commissioning(
    profile: str,
    status: str,
    expected: bool,
) -> None:
    """Template GRAY may be covered downstream only in the explicit 24-group policy."""
    stage32 = _load_module("pipeline_zs32_runtime_template_policy", "pipeline/32_run_zs32_multimodel_inference.py")

    assert stage32._template_status_allows_downstream(status, profile) is expected


def _valid_template_review() -> dict[str, object]:
    return {
        "status": "REVIEW",
        "score": 0.15,
        "risk_score": 0.15,
        "low_threshold": 0.1,
        "high_threshold": 0.2,
        "model_version": "model-v1",
        "threshold_version": "threshold-v1",
        "roi_version": "roi-v1",
        "template_version": "template-v1",
        "best_template_sha256": sha256(b"template").hexdigest(),
    }


def test_only_complete_semantic_template_review_continues_commissioning() -> None:
    """A real finite gray-band result may continue to its complementary branches."""
    stage32 = _load_module("pipeline_zs32_runtime_valid_review", "pipeline/32_run_zs32_multimodel_inference.py")

    assert stage32._template_result_allows_downstream(
        _valid_template_review(),
        "zs32-right-24-commissioning",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("score", None),
        ("risk_score", float("nan")),
        ("model_version", ""),
        ("best_template_sha256", "not-a-sha"),
        ("low_threshold", 0.16),
    ],
)
def test_incomplete_or_nonsemantic_template_review_stops_commissioning(field: str, value: object) -> None:
    """An exception-shaped or malformed REVIEW must remain fail-closed."""
    stage32 = _load_module("pipeline_zs32_runtime_invalid_review", "pipeline/32_run_zs32_multimodel_inference.py")
    result = _valid_template_review()
    result[field] = value

    assert not stage32._template_result_allows_downstream(result, "zs32-right-24-commissioning")


def test_template_exception_review_stops_commissioning() -> None:
    """The REVIEW wrapper emitted for TemplateGate exceptions must not launch GPU inference."""
    stage32 = _load_module("pipeline_zs32_runtime_exception_review", "pipeline/32_run_zs32_multimodel_inference.py")
    result = {"view": "front", "status": "REVIEW", "reason": "template gate exception: ValueError: broken"}

    assert not stage32._template_result_allows_downstream(result, "zs32-right-24-commissioning")


def test_template_short_circuit_retains_commissioning_policy(tmp_path: Path) -> None:
    """Even an NG short circuit must remain visibly non-production in both runtime records."""
    stage32 = _load_module("pipeline_zs32_runtime_stop_policy", "pipeline/32_run_zs32_multimodel_inference.py")
    images = _write_images(tmp_path)
    request = stage32.InspectionRequest(
        "part-001",
        "session-001",
        "group-001",
        "right",
        images,
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    output = tmp_path / "output"

    stage32._publish_template_stop(
        request,
        output,
        tuple(
            {
                "view": view,
                "status": "NG_TEMPLATE" if view == "front" else "PASS",
                "reason": "risk reached threshold" if view == "front" else "",
                "score": 0.9 if view == "front" else 0.01,
                "best_template_path": str(images[view]),
            }
            for view in VIEWS
        ),
        workspace,
        "zs32-right-24-commissioning",
    )

    for name in ("runtime_summary.json", "runtime_manifest.json"):
        payload = json.loads((output / name).read_text(encoding="utf-8"))
        assert {field: payload[field] for field in ("part_id", "capture_session", "group_id", "hand")} == {
            "part_id": "part-001",
            "capture_session": "session-001",
            "group_id": "group-001",
            "hand": "right",
        }
        assert payload["fusion_profile"] == "zs32-right-24-commissioning"
        assert payload["commissioning_only"] is True
        assert payload["production_release_allowed"] is False

    manifest = json.loads((output / "runtime_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["views"]) == set(VIEWS)
    for view in VIEWS:
        branches = manifest["views"][view]["branches"]
        assert branches["template"]["status"] == ("NG_TEMPLATE" if view == "front" else "PASS")
        for branch in ("patchcore", "yolo", "fusion"):
            assert branches[branch] == {
                "state": "skipped",
                "status": "SKIPPED",
                "score": None,
                "reason": "template gate did not allow downstream inference",
            }
    parsed = load_inspection_result(output)
    assert all(
        view.branches[branch].state is BranchState.SKIPPED
        for view in parsed.views
        for branch in ("patchcore", "yolo", "fusion")
    )


def test_successful_strict_fusion_summary_retains_inspection_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The successful strict summary must identify the exact captured physical part."""
    stage32 = _load_module("pipeline_zs32_runtime_strict_identity", "pipeline/32_run_zs32_multimodel_inference.py")
    stage18 = SimpleNamespace(
        build_parser=lambda: SimpleNamespace(parse_args=lambda _values: SimpleNamespace()),
        run_fusion=lambda _args: [SimpleNamespace(part_id="part-001", final_status="OK", reason="all clear")],
    )
    monkeypatch.setattr(stage32, "_load_stage18", lambda: stage18)
    args = SimpleNamespace(
        part_id="part-001",
        capture_session="session-001",
        group_id="group-001",
        hand="right",
        fusion_profile="zs32-right-24-commissioning",
        threshold_artifact=tmp_path / "thresholds.json",
    )
    output = tmp_path / "output"
    audit = output / "fusion" / "audit" / "part-001.json"
    audit.parent.mkdir(parents=True)
    audit.write_text(json.dumps({"inspection_complete": True}), encoding="utf-8")

    summary = stage32._run_strict_fusion(args, output, tmp_path / "template.csv")

    assert summary["inspection_complete"] is True
    assert {field: summary[field] for field in ("part_id", "capture_session", "group_id", "hand")} == {
        "part_id": "part-001",
        "capture_session": "session-001",
        "group_id": "group-001",
        "hand": "right",
    }


def test_right_profile_contains_exactly_thirty_six_versioned_groups() -> None:
    """The right-only deployment contract must not fabricate left-hand groups."""
    path = REPO_ROOT / "config/fusion/zs32_right_six_view.json"
    profile = json.loads(path.read_text(encoding="utf-8"))
    records = profile["expected_versions"]

    assert len(records) == 36
    assert {record["hand"] for record in records} == {"right"}
    assert {record["view"] for record in records} == set(LEGACY_VIEWS)
    assert all(len(profile["required_branches_by_view"][view]) == 6 for view in LEGACY_VIEWS)


def test_eight_view_commissioning_profile_contains_exactly_twenty_four_groups() -> None:
    path = REPO_ROOT / "config/fusion/zs32_right_eight_view_24_group_commissioning.json"
    profile = json.loads(path.read_text(encoding="utf-8"))
    records = profile["expected_versions"]

    assert len(records) == 24
    assert {record["hand"] for record in records} == {"right"}
    assert {record["view"] for record in records} == set(VIEWS)
    assert tuple(profile["required_branches_by_view"]) == VIEWS
    for view in VIEWS:
        assert set(profile["required_branches_by_view"][view]) == {
            "template_match",
            f"anomaly_{view}",
            "yolo",
        }


def test_stage18_resolves_right_only_named_profile() -> None:
    """Stage 18 must keep the full and right-only strict profiles separate."""
    stage18 = _load_module("pipeline_zs32_right_profile", "pipeline/18_fuse_inspection_results.py")
    args = stage18.build_parser().parse_args(
        ["--profile", "zs32-right", "--output-dir", "/tmp/zs32-right-profile-test"],
    )

    assert stage18._fusion_config_path(args) == REPO_ROOT / "config/fusion/zs32_right_six_view.json"


def test_stage18_resolves_explicit_twenty_four_group_commissioning_profile() -> None:
    stage18 = _load_module("pipeline_zs32_right_24_profile", "pipeline/18_fuse_inspection_results.py")
    args = stage18.build_parser().parse_args(
        ["--profile", "zs32-right-24-commissioning", "--output-dir", "/tmp/zs32-right-24-profile-test"],
    )

    assert stage18._fusion_config_path(args) == (
        REPO_ROOT / "config/fusion/zs32_right_eight_view_24_group_commissioning.json"
    )


def test_strict_fusion_updates_runtime_manifest_without_losing_provenance(tmp_path: Path) -> None:
    """The final manifest must not retain the pre-fusion REVIEW as its current result."""
    stage32 = _load_module("pipeline_zs32_runtime_manifest_update", "pipeline/32_run_zs32_multimodel_inference.py")
    output = tmp_path / "output"
    output.mkdir()
    manifest = {
        "runtime_config_sha256": "a" * 64,
        "machine_status": "REVIEW",
        "inspection_complete": False,
        "missing_required_evidence": ["strict_fusion_not_run", "quality_gate"],
        "note": "continuous evidence only",
        "views": {
            view: {"branches": {"fusion": {"state": "skipped", "status": "SKIPPED", "score": None}}}
            for view in VIEWS
        },
    }
    (output / "runtime_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    fusion = output / "fusion"
    fusion.mkdir()
    (fusion / "fused_predictions.csv").write_text("part_id,final_status\npart-001,OK\n", encoding="utf-8")
    summary = {
        "machine_status": "OK",
        "inspection_complete": True,
        "strict_fusion": True,
        "fusion_profile": "zs32-right-24-commissioning",
        "commissioning_only": True,
        "production_release_allowed": False,
        "fusion_output": str(output / "fusion"),
    }

    stage32._update_runtime_manifest_after_fusion(output, summary)
    updated = json.loads((output / "runtime_manifest.json").read_text(encoding="utf-8"))

    assert updated["runtime_config_sha256"] == "a" * 64
    assert updated["machine_status"] == "OK"
    assert updated["inspection_complete"] is True
    assert updated["missing_required_evidence"] == []
    assert updated["production_release_allowed"] is False
    assert updated["pre_fusion_result"]["machine_status"] == "REVIEW"
