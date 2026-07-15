# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run template-first ZS32 eight-view PatchCore, YOLO, and strict fusion."""

# The CLI converts model, file-system, and strict-fusion failures into explicit
# fail-closed diagnostics, so exception messages retain their original context.
# ruff: noqa: EM101, EM102, PLW0717, TRY003, TRY301

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2

if TYPE_CHECKING:
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capture_data.zs32_inspection_orchestrator import (  # noqa: E402
    InspectionRequest,
    write_template_match_csv,
)
from capture_data.zs32_model_runtime import ZS32ModelRuntime, load_runtime_config  # noqa: E402
from capture_data.zs32_template_gate import TemplateGate  # noqa: E402
from zs32_inspection.capture.contracts import utc_now  # noqa: E402
from zs32_inspection.dashboard.contracts import ProgressRecord  # noqa: E402
from zs32_inspection.dashboard.control import write_progress  # noqa: E402
from zs32_inspection.domain.views import CANONICAL_VIEWS  # noqa: E402

DEFAULT_RUNTIME_CONFIG = REPO_ROOT / "config/fusion/zs32_runtime_models_eight_view.json"
EIGHT_VIEW_COMMISSIONING_FUSION_PROFILE = "zs32-right-24-commissioning"


def build_parser() -> argparse.ArgumentParser:
    """Build the unified ZS32 runtime parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("mode", nargs="?", choices=("infer", "fuse"), default="infer")
    parser.add_argument("--part-id", required=True)
    parser.add_argument("--capture-session", required=True)
    parser.add_argument("--group-id", required=True)
    parser.add_argument(
        "--hand",
        choices=("right",),
        default="right",
        help="Strict eight-view commissioning supports the right hand only.",
    )
    for view in CANONICAL_VIEWS:
        parser.add_argument(f"--{view.replace('_', '-')}-image", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-json", type=Path)
    parser.add_argument("--template-model-dir", type=Path, help="Optional first-stage template gate model.")
    parser.add_argument(
        "--diagnostic-skip-template",
        action="store_true",
        help="Run infer-only diagnostics without template or Stage18 evidence.",
    )
    parser.add_argument("--quality-csv", type=Path, help="Strict eight-view quality-gate evidence CSV.")
    parser.add_argument("--registration-csv", type=Path, help="Strict eight-view registration evidence CSV.")
    parser.add_argument("--geometry-csv", type=Path, help="Strict eight-view geometry evidence CSV.")
    parser.add_argument("--threshold-artifact", type=Path, help="Locked Stage-31 right-hand dual thresholds.")
    parser.add_argument(
        "--fusion-profile",
        choices=(EIGHT_VIEW_COMMISSIONING_FUSION_PROFILE,),
        default=EIGHT_VIEW_COMMISSIONING_FUSION_PROFILE,
        help="Explicit Stage-18 right-hand fusion contract.",
    )
    parser.add_argument(
        "--fusion-config",
        type=Path,
        help="Exact bundle-generated Stage-18 fusion profile; overrides the matching named profile path.",
    )
    parser.add_argument("--gt-label", type=int, choices=(0, 1), help="Optional label for calibration-row export.")
    parser.add_argument("--split", help="Optional physical-part split paired with --gt-label.")
    parser.add_argument("--accelerator", default="auto", help="Anomalib accelerator.")
    parser.add_argument("--devices", type=int, default=1, help="Anomalib device count.")
    parser.add_argument("--yolo-device", help="Optional Ultralytics device, e.g. 0 or cpu.")
    parser.add_argument(
        "--diagnostic-mask-threshold",
        type=float,
        default=0.65,
        help="Display-only normalized anomaly-map fallback threshold.",
    )
    return parser


def _request_from_args(args: argparse.Namespace) -> InspectionRequest:
    images = {view: getattr(args, f"{view}_image") for view in CANONICAL_VIEWS}
    return InspectionRequest(args.part_id, args.capture_session, args.group_id, args.hand, images)


def _template_status_allows_downstream(status: str, fusion_profile: str) -> bool:
    """Allow template GRAY evidence to be covered only in complementary commissioning."""
    normalized = status.strip().upper()
    return normalized == "PASS" or (
        fusion_profile == EIGHT_VIEW_COMMISSIONING_FUSION_PROFILE and normalized == "REVIEW"
    )


def _template_result_allows_downstream(result: dict[str, Any], fusion_profile: str) -> bool:
    """Accept only complete PASS evidence or a genuine commissioning gray-band result."""
    status = str(result.get("status", "")).strip().upper()
    if not _template_status_allows_downstream(status, fusion_profile):
        return False
    score = _finite_template_score(result.get("score"))
    risk = _finite_template_score(result.get("risk_score"))
    low = _finite_template_score(result.get("low_threshold"))
    high = _finite_template_score(result.get("high_threshold"))
    if any(value is None for value in (score, risk, low, high)):
        return False
    numbers_valid = score == risk and low <= high
    decision_valid = (status == "PASS" and risk < low) or (status == "REVIEW" and low <= risk < high)
    versions_valid = all(
        isinstance(result.get(field), str) and bool(str(result[field]).strip())
        for field in ("model_version", "threshold_version", "roi_version", "template_version")
    )
    evidence_sha256 = result.get("best_template_sha256")
    evidence_valid = (
        isinstance(evidence_sha256, str)
        and len(evidence_sha256) == 64
        and all(character in "0123456789abcdef" for character in evidence_sha256)
    )
    return numbers_valid and decision_valid and versions_valid and evidence_valid


def _finite_template_score(value: object) -> float | None:
    """Return only strict JSON-safe numeric Template scores."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    score = float(value)
    return score if math.isfinite(score) else None


_TEMPLATE_NUMERIC_FIELDS = frozenset(
    {
        "score",
        "risk_score",
        "raw_score",
        "similarity",
        "threshold",
        "low_threshold",
        "high_threshold",
    },
)


def _json_safe_template_value(value: object, *, field: str | None = None) -> Any:
    """Recursively normalize Template evidence to strict JSON values."""
    if field in _TEMPLATE_NUMERIC_FIELDS:
        return _finite_template_score(value)
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("template result JSON object keys must be strings")
        return {key: _json_safe_template_value(item, field=key) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_template_value(item) for item in value]
    raise TypeError(f"unsupported template result JSON value: {type(value).__name__}")


def _normalize_template_result_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return one complete Template result with invalid numeric evidence nulled."""
    return {key: _json_safe_template_value(value, field=key) for key, value in result.items()}


def _template_gate(
    request: InspectionRequest,
    runtime: ZS32ModelRuntime,
    model_dir: Path,
    workspace_parent: Path,
    fusion_profile: str,
) -> tuple[tuple[dict[str, Any], ...], Path]:
    """Crop and evaluate all eight template inputs before aggregate gating."""
    if fusion_profile != EIGHT_VIEW_COMMISSIONING_FUSION_PROFILE:
        raise ValueError("template gate requires the 24-group commissioning profile")
    workspace_parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix=".zs32-template-first-", dir=workspace_parent))
    gate = TemplateGate(model_dir.resolve())
    results: list[dict[str, Any]] = []
    crop_dir = workspace / "crops" / "template"
    crop_dir.mkdir(parents=True)
    try:
        for view in CANONICAL_VIEWS:
            source_path = Path(request.images[view]).resolve()
            image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"could not read template source image: {source_path}")
            if (image.shape[1], image.shape[0]) != (runtime.config.image_width, runtime.config.image_height):
                raise ValueError(
                    f"source image dimensions for {view} must be "
                    f"{runtime.config.image_width}x{runtime.config.image_height}",
                )
            x1, y1, x2, y2 = runtime.config.patchcore_rois[request.hand][view]
            crop_path = crop_dir / f"{view}.png"
            if not cv2.imwrite(str(crop_path), image[y1:y2, x1:x2]):
                raise OSError(f"failed to write template crop: {crop_path}")
            try:
                raw_result = gate.evaluate(crop_path, request.hand, view)
                result = raw_result.to_dict()
            except Exception as error:  # noqa: BLE001 - gate errors always fail closed
                result = {
                    "view": view,
                    "status": "REVIEW",
                    "reason": f"template gate exception: {type(error).__name__}: {error}",
                }
            results.append(result)
        return tuple(results), workspace
    except Exception:
        shutil.rmtree(workspace)
        raise


def _publish_template_stop(
    request: InspectionRequest,
    output_dir: Path,
    results: tuple[dict[str, Any], ...],
    workspace: Path,
    fusion_profile: str,
) -> None:
    """Publish an eight-view template result with every downstream branch skipped."""
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    results_by_view = {str(result.get("view", "")): result for result in results}
    if len(results) != len(CANONICAL_VIEWS) or set(results_by_view) != set(CANONICAL_VIEWS):
        raise ValueError("template stop publication requires exactly one result for all eight views")
    blocked = [
        results_by_view[view]
        for view in CANONICAL_VIEWS
        if not _template_result_allows_downstream(results_by_view[view], fusion_profile)
    ]
    if not blocked:
        raise ValueError("template stop publication requires at least one blocked template result")
    normalized_results = tuple(_normalize_template_result_payload(results_by_view[view]) for view in CANONICAL_VIEWS)
    normalized_by_view = {str(result["view"]): result for result in normalized_results}
    decisive_raw = next(
        (result for result in blocked if str(result.get("status", "")).upper() == "NG_TEMPLATE"),
        blocked[0],
    )
    decisive = normalized_by_view[str(decisive_raw["view"])]
    status = str(decisive.get("status", "REVIEW")).upper()
    summary = {
        "machine_status": status if status in {"NG_TEMPLATE", "INVALID_CAPTURE"} else "REVIEW",
        "inspection_complete": False,
        "short_circuited": True,
        "stopped_after": "template_match",
        "part_id": request.part_id,
        "capture_session": request.capture_session,
        "group_id": request.group_id,
        "hand": request.hand,
        "evaluated_views": [item.get("view") for item in results],
        "template_results": list(normalized_results),
        "reason": decisive.get("reason") or "template gate did not allow downstream inference",
        "strict_fusion": False,
        "fusion_profile": fusion_profile,
        "commissioning_only": True,
        "production_release_allowed": False,
    }
    (workspace / "runtime_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    source_dir = workspace / "sources"
    template_evidence_dir = workspace / "evidence" / "template"
    source_dir.mkdir(parents=True, exist_ok=True)
    template_evidence_dir.mkdir(parents=True, exist_ok=True)
    views: dict[str, dict[str, Any]] = {}
    for view in CANONICAL_VIEWS:
        result = normalized_by_view[view]
        source = Path(request.images[view]).resolve()
        source_copy = source_dir / f"{view}{source.suffix or '.png'}"
        shutil.copy2(source, source_copy)
        image = cv2.imread(str(source_copy), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not decode template-stop source image: {source}")
        template_source = Path(str(result.get("best_template_path", ""))).expanduser()
        template_branch: dict[str, Any] = {
            "state": "error",
            "status": str(result.get("status", "REVIEW")).upper(),
            "score": result.get("score"),
            "reason": str(result.get("reason") or "template evidence is incomplete"),
        }
        if template_source.is_file():
            evidence_copy = template_evidence_dir / f"{view}{template_source.suffix or '.bin'}"
            shutil.copy2(template_source, evidence_copy)
            template_branch.update(
                state="available",
                evidence_path=str(evidence_copy.relative_to(workspace)),
            )
        skipped = {
            "state": "skipped",
            "status": "SKIPPED",
            "score": None,
            "reason": "template gate did not allow downstream inference",
        }
        views[view] = {
            "view": view,
            "part_id": request.part_id,
            "capture_session": request.capture_session,
            "group_id": request.group_id,
            "hand": request.hand,
            "manifest_identity": f"{request.part_id}:{request.hand}:{view}",
            "source_path": str(source_copy.relative_to(workspace)),
            "source_sha256": hashlib.sha256(source_copy.read_bytes()).hexdigest(),
            "source_shape": list(image.shape[:2]),
            "model_supported": True,
            "branches": {
                "template": template_branch,
                "patchcore": dict(skipped),
                "yolo": dict(skipped),
                "fusion": dict(skipped),
            },
        }
    manifest = {
        **summary,
        "schema_version": "1.0",
        "product": "ZS32",
        "manifest_identity": "ZS32/right",
        "views": views,
        "missing_required_evidence": ["PatchCore", "YOLO", "fusion"],
        "note": "Template-first aggregate stop; all model and fusion branches were skipped.",
    }
    (workspace / "runtime_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    workspace.replace(output_dir)


def _load_stage18() -> ModuleType:
    script_path = REPO_ROOT / "pipeline/18_fuse_inspection_results.py"
    spec = importlib.util.spec_from_file_location("pipeline_fuse_inspection_results_runtime", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Stage 18: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_strict_fusion(args: argparse.Namespace, output_dir: Path, template_csv: Path) -> dict[str, Any]:
    """Call Stage 18 in-process so one profile remains authoritative."""
    stage18 = _load_stage18()
    fusion_config = getattr(args, "fusion_config", None)
    profile_args = (
        ["--fusion-config", str(fusion_config)]
        if fusion_config is not None
        else ["--profile", args.fusion_profile]
    )
    values = [
        *profile_args,
        "--threshold-artifact",
        str(args.threshold_artifact),
        "--template-match-csv",
        str(template_csv),
        "--branch-csv",
        f"patchcore={output_dir / 'patchcore.csv'}",
        "--branch-csv",
        f"yolo={output_dir / 'yolo.csv'}",
        "--output-dir",
        str(output_dir / "fusion"),
    ]
    fusion_args = stage18.build_parser().parse_args(values)
    decisions = stage18.run_fusion(fusion_args)
    if len(decisions) != 1 or decisions[0].part_id != args.part_id:
        raise RuntimeError("strict fusion did not return exactly the requested physical part")
    decision = decisions[0]
    audit_path = output_dir / "fusion" / "audit" / f"{args.part_id}.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
    return {
        "part_id": args.part_id,
        "capture_session": args.capture_session,
        "group_id": args.group_id,
        "hand": args.hand,
        "machine_status": decision.final_status,
        "inspection_complete": bool(audit.get("inspection_complete", False)),
        "strict_fusion": True,
        "fusion_output": str(output_dir / "fusion"),
        "reason": decision.reason,
        "fusion_profile": args.fusion_profile,
        "commissioning_only": True,
        "production_release_allowed": False,
    }


def _validate_mode(args: argparse.Namespace) -> None:
    if args.hand != "right":
        raise ValueError("strict eight-view Stage32 supports only hand='right'")
    if args.fusion_profile != EIGHT_VIEW_COMMISSIONING_FUSION_PROFILE:
        raise ValueError("strict eight-view Stage32 requires the 24-group commissioning profile")
    if not math.isfinite(args.diagnostic_mask_threshold) or not 0 <= args.diagnostic_mask_threshold <= 1:
        raise ValueError("--diagnostic-mask-threshold must be finite and within [0, 1]")
    if (args.gt_label is None) != (args.split is None):
        raise ValueError("--gt-label and --split must be provided together")
    if args.diagnostic_skip_template and (
        args.mode != "infer" or args.template_model_dir is not None or args.threshold_artifact is not None
    ):
        raise ValueError(
            "--diagnostic-skip-template requires infer mode without --template-model-dir or --threshold-artifact",
        )
    if args.mode == "fuse":
        required = {
            "--threshold-artifact": args.threshold_artifact,
            "--template-model-dir": args.template_model_dir,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"fuse mode requires: {', '.join(missing)}")


def _validate_commissioning_source_assets(args: argparse.Namespace) -> None:
    """Fail before model loading when commissioning assets drift from locked thresholds."""
    if args.mode != "fuse":
        return
    payload = json.loads(args.threshold_artifact.read_text(encoding="utf-8"))
    unsigned = dict(payload)
    artifact_sha256 = unsigned.pop("artifact_sha256", None)
    canonical = hashlib.sha256(json.dumps(unsigned, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
    if artifact_sha256 != canonical:
        raise ValueError("commissioning threshold artifact immutable SHA256 mismatch")
    if payload.get("commissioning_only") is not True or payload.get("production_release_allowed") is not False:
        raise ValueError("strict eight-view thresholds must be commissioning-only and non-production")
    sources = payload.get("source_artifacts")
    if not isinstance(sources, dict):
        raise TypeError("commissioning artifact source_artifacts must be an object")
    for name, path in (
        ("runtime_config", args.runtime_config),
        ("template_model", args.template_model_dir / "model.json"),
    ):
        binding = sources.get(name)
        expected = binding.get("sha256") if isinstance(binding, dict) else None
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected != actual:
            raise ValueError(f"{name} SHA256 differs from the locked commissioning threshold artifact")


def _update_runtime_manifest_after_fusion(output_dir: Path, summary: dict[str, Any]) -> None:
    """Promote the Stage18 result while retaining the pre-fusion runtime state."""
    path = output_dir / "runtime_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["pre_fusion_result"] = {
        field: payload.get(field)
        for field in ("machine_status", "inspection_complete", "missing_required_evidence", "note")
    }
    for field in (
        "machine_status",
        "inspection_complete",
        "strict_fusion",
        "fusion_profile",
        "commissioning_only",
        "production_release_allowed",
        "fusion_output",
    ):
        payload[field] = summary.get(field)
    payload["reason"] = str(summary.get("reason") or "Stage18 fusion completed.")
    output_root = output_dir.resolve(strict=True)
    fusion_output = Path(str(summary.get("fusion_output", ""))).expanduser().resolve()
    fusion_evidence = fusion_output / "fused_predictions.csv"
    try:
        resolved_fusion_evidence = fusion_evidence.resolve(strict=True)
    except OSError:
        resolved_fusion_evidence = None
    evidence_is_local = (
        resolved_fusion_evidence is not None
        and resolved_fusion_evidence.is_file()
        and resolved_fusion_evidence.is_relative_to(output_root)
    )
    if not evidence_is_local:
        summary["machine_status"] = "REVIEW"
        summary["inspection_complete"] = False
        summary["missing_required_evidence"] = sorted(
            {*summary.get("missing_required_evidence", []), "fusion_evidence"},
        )
        summary["reason"] = "Stage18 fusion evidence is missing or outside the result generation."
        payload["machine_status"] = summary["machine_status"]
        payload["inspection_complete"] = summary["inspection_complete"]
        payload["reason"] = summary["reason"]
    fusion_branch = {
        "state": "available" if evidence_is_local else "error",
        "status": str(summary.get("machine_status") or "ERROR") if evidence_is_local else "ERROR",
        "score": None,
        "reason": str(summary.get("reason") or "") if evidence_is_local else "Stage18 fusion evidence is missing",
    }
    if evidence_is_local:
        fusion_branch["evidence_path"] = str(resolved_fusion_evidence)
    for view in CANONICAL_VIEWS:
        payload["views"][view]["branches"]["fusion"] = dict(fusion_branch)
    if evidence_is_local and summary.get("inspection_complete") is True:
        payload["missing_required_evidence"] = []
    elif not evidence_is_local:
        payload["missing_required_evidence"] = list(summary["missing_required_evidence"])
    else:
        payload["missing_required_evidence"] = payload.get("missing_required_evidence", [])
    payload["note"] = (
        "Stage18 fusion completed for the named profile; production release remains governed by "
        "production_release_allowed."
    )
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _merge_template_results_into_runtime_manifest(
    output_dir: Path,
    results: tuple[dict[str, Any], ...],
) -> None:
    """Replace skipped Template branches with the real eight-view gate evidence."""
    by_view = {str(result.get("view", "")): result for result in results}
    if len(results) != len(CANONICAL_VIEWS) or set(by_view) != set(CANONICAL_VIEWS):
        raise ValueError("template manifest merge requires exactly one result for all eight views")
    evidence_dir = output_dir / "evidence" / "template"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "runtime_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    all_available = True
    for view in CANONICAL_VIEWS:
        result = by_view[view]
        raw_score = result.get("score")
        score = (
            float(raw_score)
            if not isinstance(raw_score, bool) and isinstance(raw_score, int | float) and math.isfinite(float(raw_score))
            else None
        )
        source = Path(str(result.get("best_template_path", ""))).expanduser()
        branch: dict[str, Any] = {
            "state": "error",
            "status": str(result.get("status") or "ERROR").upper(),
            "score": score,
            "reason": str(result.get("reason") or "Template evidence is incomplete"),
        }
        if source.is_file():
            destination = evidence_dir / f"{view}{source.suffix or '.bin'}"
            shutil.copy2(source, destination)
            branch.update(state="available", evidence_path=str(destination.resolve()))
        else:
            all_available = False
        payload["views"][view]["branches"]["template"] = branch
    if all_available:
        payload["missing_required_evidence"] = [
            item for item in payload.get("missing_required_evidence", []) if item != "template_match"
        ]
    payload["template_results"] = list(results)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _publish_diagnostic_contract(
    request: InspectionRequest,
    output_dir: Path,
    summary: dict[str, Any],
    patchcore_csv: Path,
    yolo_csv: Path,
) -> None:
    """Publish an explicit non-production contract for template-skipping diagnostics."""
    fields = {
        "part_id": request.part_id,
        "capture_session": request.capture_session,
        "group_id": request.group_id,
        "hand": request.hand,
        "diagnostic_skip_template": True,
        "inspection_complete": False,
        "strict_fusion": False,
        "commissioning_only": True,
        "production_release_allowed": False,
    }
    summary.update(
        fields,
        patchcore_csv=str(patchcore_csv.expanduser().resolve()),
        yolo_csv=str(yolo_csv.expanduser().resolve()),
    )
    path = output_dir / "runtime_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(fields)
    manifest["note"] = "Template and Stage18 were intentionally skipped for diagnostic inference."
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    """Run the unified template-first inference and optional strict fusion workflow."""
    args = build_parser().parse_args()
    _validate_mode(args)
    _validate_commissioning_source_assets(args)
    config = load_runtime_config(args.runtime_config)
    request = _request_from_args(args)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        msg = f"output directory already exists: {output_dir}"
        raise FileExistsError(msg)
    if request.hand not in config.supported_hands:
        msg = f"unsupported hand {request.hand!r}; available weights: {config.supported_hands}"
        raise ValueError(msg)
    runtime = ZS32ModelRuntime(
        config,
        accelerator=args.accelerator,
        devices=args.devices,
        yolo_device=args.yolo_device,
    )

    template_results: tuple[dict[str, Any], ...] | None = None
    template_workspace: Path | None = None
    if args.template_model_dir is not None:
        if args.progress_json is not None:
            write_progress(args.progress_json, ProgressRecord(request.part_id, request.capture_session, "running_template", "running template", utc_now()))
        template_results, template_workspace = _template_gate(
            request,
            runtime,
            args.template_model_dir,
            output_dir.parent,
            args.fusion_profile,
        )
        if len(template_results) != len(CANONICAL_VIEWS) or any(
            not _template_result_allows_downstream(result, args.fusion_profile) for result in template_results
        ):
            _publish_template_stop(request, output_dir, template_results, template_workspace, args.fusion_profile)
            print(f"machine_status: {json.loads((output_dir / 'runtime_summary.json').read_text())['machine_status']}")
            print("inspection_complete: false")
            print(f"output_dir: {output_dir}")
            return

    if args.progress_json is not None:
        write_progress(args.progress_json, ProgressRecord(request.part_id, request.capture_session, "running_patchcore_yolo", "running PatchCore and YOLO", utc_now()))
    result = runtime.run(
        request,
        output_dir,
        threshold_artifact=args.threshold_artifact,
        gt_label=args.gt_label,
        split=args.split,
        diagnostic_mask_threshold=args.diagnostic_mask_threshold,
    )
    template_csv: Path | None = None
    if template_results is not None and template_workspace is not None:
        final_template_crops = output_dir / "crops" / "template"
        final_template_crops.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(template_workspace / "crops" / "template", final_template_crops)
        shutil.rmtree(template_workspace)
        template_request = InspectionRequest(
            request.part_id,
            request.capture_session,
            request.group_id,
            request.hand,
            {view: final_template_crops / f"{view}.png" for view in CANONICAL_VIEWS},
        )
        template_csv = write_template_match_csv(
            template_request,
            template_results,
            output_dir / "template_match.csv",
            profile=config.profile,
        )
        _merge_template_results_into_runtime_manifest(output_dir, template_results)

    summary: dict[str, Any] = {
        "machine_status": result.machine_status,
        "inspection_complete": result.inspection_complete,
        "strict_fusion": False,
        "missing_required_evidence": list(result.missing_required_evidence),
        "errors": list(result.errors),
        "commissioning_only": True,
        "production_release_allowed": False,
    }
    if args.diagnostic_skip_template:
        _publish_diagnostic_contract(
            request,
            output_dir,
            summary,
            result.patchcore_csv,
            result.yolo_csv,
        )
    if args.mode == "fuse":
        if args.progress_json is not None:
            write_progress(args.progress_json, ProgressRecord(request.part_id, request.capture_session, "running_fusion", "running fusion", utc_now()))
        if template_csv is None:
            raise RuntimeError("template evidence unexpectedly missing")
        summary = _run_strict_fusion(args, output_dir, template_csv)
        _update_runtime_manifest_after_fusion(output_dir, summary)
    (output_dir / "runtime_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"machine_status: {summary['machine_status']}")
    print(f"inspection_complete: {str(summary['inspection_complete']).lower()}")
    print(f"patchcore_csv: {result.patchcore_csv}")
    print(f"yolo_csv: {result.yolo_csv}")
    if result.calibration_csv is not None:
        print(f"calibration_csv: {result.calibration_csv}")
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
