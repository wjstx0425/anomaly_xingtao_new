# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run template-first ZS32 six-view PatchCore, YOLO, and strict fusion."""

# The CLI converts model, file-system, and strict-fusion failures into explicit
# fail-closed diagnostics, so exception messages retain their original context.
# ruff: noqa: EM101, EM102, PLW0717, TRY003, TRY301

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2

if TYPE_CHECKING:
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capture_data.zs32_inspection_orchestrator import (  # noqa: E402
    CANONICAL_VIEWS,
    InspectionRequest,
    write_template_match_csv,
)
from capture_data.zs32_model_runtime import ZS32ModelRuntime, load_runtime_config  # noqa: E402
from capture_data.zs32_template_gate import TemplateGate  # noqa: E402

DEFAULT_RUNTIME_CONFIG = REPO_ROOT / "config/fusion/zs32_runtime_models.json"


def build_parser() -> argparse.ArgumentParser:
    """Build the unified ZS32 runtime parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("mode", nargs="?", choices=("infer", "fuse"), default="infer")
    parser.add_argument("--part-id", required=True)
    parser.add_argument("--capture-session", required=True)
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--hand", choices=("right", "left"), default="right")
    for view in CANONICAL_VIEWS:
        parser.add_argument(f"--{view.replace('_', '-')}-image", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--template-model-dir", type=Path, help="Optional first-stage template gate model.")
    parser.add_argument("--quality-csv", type=Path, help="Strict six-view quality-gate evidence CSV.")
    parser.add_argument("--registration-csv", type=Path, help="Strict six-view registration evidence CSV.")
    parser.add_argument("--geometry-csv", type=Path, help="Strict six-view geometry evidence CSV.")
    parser.add_argument("--threshold-artifact", type=Path, help="Locked Stage-31 right-hand dual thresholds.")
    parser.add_argument("--gt-label", type=int, choices=(0, 1), help="Optional label for calibration-row export.")
    parser.add_argument("--split", help="Optional physical-part split paired with --gt-label.")
    parser.add_argument("--accelerator", default="auto", help="Anomalib accelerator.")
    parser.add_argument("--devices", type=int, default=1, help="Anomalib device count.")
    parser.add_argument("--yolo-device", help="Optional Ultralytics device, e.g. 0 or cpu.")
    return parser


def _request_from_args(args: argparse.Namespace) -> InspectionRequest:
    images = {view: getattr(args, f"{view}_image") for view in CANONICAL_VIEWS}
    return InspectionRequest(args.part_id, args.capture_session, args.group_id, args.hand, images)


def _template_gate(
    request: InspectionRequest,
    runtime: ZS32ModelRuntime,
    model_dir: Path,
    workspace_parent: Path,
) -> tuple[tuple[dict[str, Any], ...], Path]:
    """Crop template inputs, then stop at the first non-PASS result."""
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
            if str(result.get("status", "")).upper() != "PASS":
                break
        return tuple(results), workspace
    except Exception:
        shutil.rmtree(workspace)
        raise


def _publish_template_stop(
    request: InspectionRequest,
    output_dir: Path,
    results: tuple[dict[str, Any], ...],
    workspace: Path,
) -> None:
    """Publish a template short-circuit without starting PatchCore or YOLO."""
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    result = results[-1]
    status = str(result.get("status", "REVIEW")).upper()
    summary = {
        "machine_status": status if status in {"NG_TEMPLATE", "INVALID_CAPTURE"} else "REVIEW",
        "inspection_complete": False,
        "short_circuited": True,
        "stopped_after": "template_match",
        "part_id": request.part_id,
        "evaluated_views": [item.get("view") for item in results],
        "template_results": list(results),
        "reason": result.get("reason"),
    }
    (workspace / "runtime_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
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
    values = [
        "--profile",
        "zs32-right",
        "--threshold-artifact",
        str(args.threshold_artifact),
        "--template-match-csv",
        str(template_csv),
        "--quality-csv",
        str(args.quality_csv),
        "--registration-csv",
        str(args.registration_csv),
        "--geometry-csv",
        str(args.geometry_csv),
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
        "machine_status": decision.final_status,
        "inspection_complete": bool(audit.get("inspection_complete", False)),
        "strict_fusion": True,
        "fusion_output": str(output_dir / "fusion"),
        "reason": decision.reason,
    }


def _validate_mode(args: argparse.Namespace) -> None:
    if (args.gt_label is None) != (args.split is None):
        raise ValueError("--gt-label and --split must be provided together")
    if args.mode == "fuse":
        required = {
            "--threshold-artifact": args.threshold_artifact,
            "--template-model-dir": args.template_model_dir,
            "--quality-csv": args.quality_csv,
            "--registration-csv": args.registration_csv,
            "--geometry-csv": args.geometry_csv,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"fuse mode requires: {', '.join(missing)}")


def main() -> None:
    """Run the unified template-first inference and optional strict fusion workflow."""
    args = build_parser().parse_args()
    _validate_mode(args)
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
        template_results, template_workspace = _template_gate(
            request,
            runtime,
            args.template_model_dir,
            output_dir.parent,
        )
        if len(template_results) != len(CANONICAL_VIEWS) or any(
            str(result.get("status", "")).upper() != "PASS" for result in template_results
        ):
            _publish_template_stop(request, output_dir, template_results, template_workspace)
            print(f"machine_status: {json.loads((output_dir / 'runtime_summary.json').read_text())['machine_status']}")
            print("inspection_complete: false")
            print(f"output_dir: {output_dir}")
            return

    result = runtime.run(
        request,
        output_dir,
        threshold_artifact=args.threshold_artifact,
        gt_label=args.gt_label,
        split=args.split,
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

    summary: dict[str, Any] = {
        "machine_status": result.machine_status,
        "inspection_complete": result.inspection_complete,
        "strict_fusion": False,
        "missing_required_evidence": list(result.missing_required_evidence),
        "errors": list(result.errors),
    }
    if args.mode == "fuse":
        if template_csv is None:
            raise RuntimeError("template evidence unexpectedly missing")
        summary = _run_strict_fusion(args, output_dir, template_csv)
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
