# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run a two-sided inspection demo with an OpenCV operator UI.

The demo runs in a fixed order: top side first, then bottom side. Press ``s``
to inspect the current side, ``r`` to reset the part, number keys to switch
part profiles, and ``q`` to quit.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import json
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capture_data.exposure_fusion import fuse_exposures
from capture_data.prepare_part_crops import PRESETS, CropPreset, SlotSpec, crop_slot, mask_holes


MODEL_NAME = "anomaly_dino"
FACE_ORDER = ("top", "bottom")
PART_ORDER = ("fx11", "c789")
DEFAULT_PART_PROFILE = "fx11"
WINDOW_NAME = "Dual-Side Defect Inspection Demo"
THRESHOLD_PROFILES = ("demo", "report")
FX11_TOP_CKPT_PATH = (
    REPO_ROOT
    / "results"
    / "fx11_demo"
    / "no_hand_top_parts_anomalydino_182x1008"
    / "runs"
    / "no_hand_top"
    / "anomaly_dino"
    / "AnomalyDINO"
    / "zs32_no_hand_top"
    / "no_hand_top"
    / "v0"
    / "weights"
    / "lightning"
    / "model.ckpt"
)
FX11_BOTTOM_CKPT_PATH = (
    REPO_ROOT
    / "results"
    / "fx11_demo"
    / "no_hand_bottom_parts_anomalydino_182x1008"
    / "runs"
    / "no_hand_bottom"
    / "anomaly_dino"
    / "AnomalyDINO"
    / "zs32_no_hand_bottom"
    / "no_hand_bottom"
    / "v1"
    / "weights"
    / "lightning"
    / "model.ckpt"
)
PROFILE_DEMO_THRESHOLDS = {
    "fx11": {
        "top": 0.5,
        "bottom": 0.49,
    },
    "c789": {
        "top": 0.6,
        "bottom": 0.48961880803108215,
    },
}
TIMING_LABELS = {
    "capture_hdr": "采集HDR",
    "crop_mask": "裁剪mask",
    "quality_gate": "质量门控",
    "model_inference": "模型推理",
    "save_results": "保存结果",
    "gui_render": "GUI渲染",
    "total_face": "单面总计",
}


@dataclass(frozen=True)
class PartFaceDefaults:
    """Default settings for one face of a supported part profile."""

    title: str
    view: str
    preset_name: str
    output_root: Path
    ckpt_path: Path | None = None
    demo_threshold: float | None = None
    image_size: tuple[int, int] | None = None


@dataclass(frozen=True)
class PartProfile:
    """A selectable inspection profile for one physical part."""

    key: str
    title: str
    description: str
    faces: dict[str, PartFaceDefaults]


PART_PROFILES = {
    "fx11": PartProfile(
        key="fx11",
        title="FX11",
        description="no_hand top/bottom 6x1",
        faces={
            "top": PartFaceDefaults(
                title="正面",
                view="no_hand_top",
                preset_name="fx11_no_hand_top_6x1",
                output_root=REPO_ROOT / "results" / "fx11_demo" / "no_hand_top_parts_anomalydino_182x1008",
                ckpt_path=FX11_TOP_CKPT_PATH,
                demo_threshold=PROFILE_DEMO_THRESHOLDS["fx11"]["top"],
                image_size=(182, 1008),
            ),
            "bottom": PartFaceDefaults(
                title="背面",
                view="no_hand_bottom",
                preset_name="fx11_no_hand_bottom_6x1",
                output_root=REPO_ROOT / "results" / "fx11_demo" / "no_hand_bottom_parts_anomalydino_182x1008",
                ckpt_path=FX11_BOTTOM_CKPT_PATH,
                demo_threshold=PROFILE_DEMO_THRESHOLDS["fx11"]["bottom"],
                image_size=(182, 1008),
            ),
        },
    ),
    "c789": PartProfile(
        key="c789",
        title="C789",
        description="left top/bottom 3x2",
        faces={
            "top": PartFaceDefaults(
                title="正面",
                view="left_top",
                preset_name="c789_left_top_3x2",
                output_root=REPO_ROOT / "results" / "c789" / "left_top_parts_anomalydino",
                demo_threshold=PROFILE_DEMO_THRESHOLDS["c789"]["top"],
            ),
            "bottom": PartFaceDefaults(
                title="底面",
                view="left_bottom",
                preset_name="c789_left_bottom_3x2",
                output_root=REPO_ROOT / "results" / "c789" / "left_bottom_parts_anomalydino",
                demo_threshold=PROFILE_DEMO_THRESHOLDS["c789"]["bottom"],
            ),
        },
    ),
}


@dataclass(frozen=True)
class FaceConfig:
    """Static settings for one inspected face."""

    part_key: str
    part_title: str
    key: str
    title: str
    view: str
    preset_name: str
    output_root: Path
    ckpt_path: Path | None
    threshold: float | None
    threshold_source: str
    image_size: tuple[int, int] | None
    short_exposure: float
    long_exposure: float
    gain: float

    @property
    def preset(self) -> CropPreset:
        """Return the crop preset for this face."""
        return PRESETS[self.preset_name]


@dataclass(frozen=True)
class SlotResult:
    """Prediction result for one slot."""

    slot: str
    score: float | None
    threshold: float | None
    pred_label: int
    source_path: Path

    @property
    def status(self) -> str:
        """Return OK/NG text for display."""
        return "NG" if self.pred_label else "OK"


@dataclass(frozen=True)
class QualityReport:
    """Image quality gate output for one inspected face."""

    status: str
    issues: list[str]
    image_mean: float
    image_std: float
    image_dark_pct: float
    image_clip_pct: float
    image_sharpness: float
    crop_mean_min: float
    crop_mean_max: float
    crop_mean_avg: float
    crop_std_avg: float
    reference_mean_avg: float | None = None
    reference_mean_delta: float | None = None


@dataclass(frozen=True)
class ImageInput:
    """Image and source metadata for one inspection request."""

    image: np.ndarray
    source: str
    source_path: Path | None = None


class QualityGateError(RuntimeError):
    """Raised when image quality gate is configured to block prediction."""

    def __init__(self, face: FaceConfig, quality: QualityReport) -> None:
        self.face = face
        self.quality = quality
        super().__init__(f"{face.title}质量门控失败: {'; '.join(quality.issues)}")


@dataclass
class FaceResult:
    """All output artifacts and slot predictions for one face."""

    face_key: str
    title: str
    image_path: Path
    crops_dir: Path
    predictions_csv: Path
    slots: list[SlotResult]
    quality: QualityReport | None = None
    crop_metrics: list[dict[str, Any]] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    trace_path: Path | None = None
    checkpoint_path: Path | None = None
    predict_batch_size: int | None = None

    @property
    def defect_slots(self) -> list[str]:
        """Return slots predicted as defective."""
        return [slot.slot for slot in self.slots if slot.pred_label == 1]

    @property
    def status(self) -> str:
        """Return OK/NG for this face."""
        return "NG" if self.defect_slots else "OK"


@dataclass
class DemoState:
    """Mutable UI state for the two-sided inspection flow."""

    part_key: str = DEFAULT_PART_PROFILE
    part_title: str = PART_PROFILES[DEFAULT_PART_PROFILE].title
    active_index: int = 0
    status_message: str = "等待正面上料，按 s 开始检测"
    current_face: str | None = None
    current_image: np.ndarray | None = None
    results: dict[str, FaceResult] = field(default_factory=dict)

    @property
    def active_face(self) -> str | None:
        """Return the next face to inspect."""
        if self.active_index >= len(FACE_ORDER):
            return None
        return FACE_ORDER[self.active_index]

    @property
    def finished(self) -> bool:
        """Return whether both sides have been inspected."""
        return self.active_index >= len(FACE_ORDER)


@dataclass
class TimingRecorder:
    """Accumulate stage timings in milliseconds."""

    stages_ms: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Any:
        """Record elapsed time for one named stage."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter() - start) * 1000.0)

    def add(self, name: str, elapsed_ms: float) -> None:
        """Add elapsed milliseconds to a named stage."""
        self.stages_ms[name] = self.stages_ms.get(name, 0.0) + elapsed_ms


class HdrCamera:
    """Small lazy wrapper around the Hikvision camera SDK."""

    def __init__(
        self,
        *,
        device: int,
        fps: float,
        timeout_ms: int,
        settle_frames: int,
        align_hdr: bool,
        short_dark_threshold: float,
        long_clip_threshold: float,
        blend_width: float,
        blur_size: int,
        hdr_max_retries: int,
        hdr_max_clip_pct: float,
    ) -> None:
        self.device = device
        self.fps = fps
        self.timeout_ms = timeout_ms
        self.settle_frames = settle_frames
        self.align_hdr = align_hdr
        self.short_dark_threshold = short_dark_threshold
        self.long_clip_threshold = long_clip_threshold
        self.blend_width = blend_width
        self.blur_size = blur_size
        self.hdr_max_retries = hdr_max_retries
        self.hdr_max_clip_pct = hdr_max_clip_pct
        self.camera_utils: ModuleType | None = None
        self.cam: Any | None = None
        self.frame: Any | None = None
        self.buf: Any | None = None

    def __enter__(self) -> HdrCamera:
        """Open and start the camera."""
        from capture_data import collect_dataset as camera_utils

        self.camera_utils = camera_utils
        self.cam = camera_utils.open_camera(self.device)
        camera_utils.setup_camera(self.cam, 4000.0, 0.0, self.fps)
        camera_utils.check_ret(self.cam.MV_CC_StartGrabbing(), "StartGrabbing")
        self.frame = camera_utils.MV_FRAME_OUT_INFO_EX()
        self.buf = (ctypes.c_ubyte * (50 * 1024 * 1024))()
        return self

    def __exit__(self, *_exc: object) -> None:
        """Stop and close the camera."""
        if self.cam is None:
            return
        try:
            self.cam.MV_CC_StopGrabbing()
        except Exception:
            pass
        self.cam.MV_CC_CloseDevice()
        self.cam.MV_CC_DestroyHandle()

    def capture(self, face: FaceConfig) -> np.ndarray:
        """Capture one HDR-fused image for a face without saving source exposures."""
        if self.camera_utils is None or self.cam is None or self.frame is None or self.buf is None:
            msg = "Camera has not been opened."
            raise RuntimeError(msg)

        camera_args = SimpleNamespace(
            hdr_settle_frames=self.settle_frames,
            timeout_ms=self.timeout_ms,
        )
        self.camera_utils.set_float(self.cam, "Gain", face.gain)

        fused: np.ndarray | None = None
        fused_clip_pct = 100.0
        for attempt in range(self.hdr_max_retries + 1):
            short_img = self.camera_utils.capture_at_exposure(
                self.cam,
                face.short_exposure,
                self.buf,
                self.frame,
                camera_args,
            )
            long_img = self.camera_utils.capture_at_exposure(
                self.cam,
                face.long_exposure,
                self.buf,
                self.frame,
                camera_args,
            )
            fused = fuse_exposures(
                [short_img, long_img],
                method="selective",
                align=self.align_hdr,
                short_dark_threshold=self.short_dark_threshold,
                long_clip_threshold=self.long_clip_threshold,
                blend_width=self.blend_width,
                blur_size=self.blur_size,
            )
            fused_clip_pct = self.camera_utils.image_clip_pct(fused)
            if fused_clip_pct <= self.hdr_max_clip_pct:
                break
            if attempt < self.hdr_max_retries:
                print(f"HDR clip={fused_clip_pct:.2f}%, retrying...")

        if fused is None:
            msg = "HDR capture did not return an image."
            raise RuntimeError(msg)
        return fused


class AnomalibPredictor:
    """Cached Anomalib predictor for one face."""

    def __init__(self, face: FaceConfig, args: argparse.Namespace) -> None:
        from capture_data import inference as inference_helpers
        from anomalib.data import PredictDataset
        from anomalib.engine import Engine
        from torch.utils.data import DataLoader

        self.face = face
        self.inference = inference_helpers
        self.workflow = inference_helpers._load_workflow_module()
        self.predict_batch_size = args.predict_batch_size
        self.predict_num_workers = args.predict_num_workers
        self.last_effective_batch_size = args.predict_batch_size
        self.predict_dataset_class = PredictDataset
        self.data_loader_class = DataLoader
        self.args = SimpleNamespace(
            output_root=face.output_root,
            view=face.view,
            model=MODEL_NAME,
            ckpt_path=face.ckpt_path,
            threshold=face.threshold,
            visualize=False,
            rebuild_model=False,
            image_size=face.image_size,
            imagenet_dir=args.imagenet_dir,
            anomaly_dino_neighbors=args.anomaly_dino_neighbors,
            anomaly_dino_encoder=args.anomaly_dino_encoder,
            anomaly_dino_masking=args.anomaly_dino_masking,
            anomaly_dino_coreset_subsampling=args.anomaly_dino_coreset_subsampling,
            anomaly_dino_sampling_ratio=args.anomaly_dino_sampling_ratio,
        )
        self.checkpoint_path = inference_helpers._resolve_checkpoint(self.args, self.workflow)
        self.threshold = inference_helpers._resolve_threshold(self.args, self.workflow)
        image_size = face.image_size or self.workflow.DEFAULT_IMAGE_SIZE
        self.model = inference_helpers._load_model(
            self.args,
            self.checkpoint_path,
            image_size,
            args.output_dir / "visualizations" / face.part_key / face.key,
        )
        self.engine = Engine(
            accelerator=args.accelerator,
            devices=args.devices,
            default_root_dir=args.output_dir / "engine" / face.part_key / face.key,
            logger=False,
            enable_progress_bar=args.show_progress_bar,
        )

    def _predict_with_batch_size(self, crop_paths: list[Path], batch_size: int) -> Any:
        """Run Engine.predict with an explicit batch size."""
        dataset = self.predict_dataset_class(crop_paths[0].parent)
        dataloader = self.data_loader_class(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            num_workers=self.predict_num_workers,
            pin_memory=True,
        )
        return self.engine.predict(
            model=self.model,
            dataloaders=dataloader,
            ckpt_path=None,
            return_predictions=True,
        )

    @staticmethod
    def _clear_cuda_after_oom() -> None:
        """Best-effort cleanup after CUDA OOM."""
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def predict(
        self,
        crop_paths: list[Path],
        predictions_csv: Path,
        timings: TimingRecorder | None = None,
    ) -> list[SlotResult]:
        """Predict all slot crops in one batch and save a CSV report."""
        recorder = timings or TimingRecorder()
        with recorder.stage("model_inference"):
            try:
                predictions = self._predict_with_batch_size(crop_paths, self.predict_batch_size)
                self.last_effective_batch_size = self.predict_batch_size
            except RuntimeError as error:
                if "out of memory" not in str(error).lower() or self.predict_batch_size <= 1:
                    raise
                print(
                    f"[runtime] {self.face.title} batch={self.predict_batch_size} 显存不足，"
                    "清理缓存后自动回退到 batch=1 重试。"
                )
                self._clear_cuda_after_oom()
                predictions = self._predict_with_batch_size(crop_paths, 1)
                self.last_effective_batch_size = 1
        with recorder.stage("save_results"):
            source_map = {str(path.resolve()): str(path.resolve()) for path in crop_paths}
            frame = self.inference._prediction_frame(
                predictions,
                self.args,
                self.checkpoint_path,
                source_map,
                self.threshold,
                self.workflow,
            )
            frame["face"] = self.face.key
            frame["slot"] = frame["source_path"].map(slot_from_path)
            frame["pred_label"] = [
                label_for_prediction(row, self.threshold, self.inference, self.workflow)
                for row in frame.itertuples(index=False)
            ]
            predictions_csv.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(predictions_csv, index=False)
        return slot_results_from_frame(frame, self.threshold)


class MockPredictor:
    """Deterministic predictor used for UI screenshots and offline checks."""

    def __init__(self, face: FaceConfig, defect_slots: set[str], predict_batch_size: int) -> None:
        self.face = face
        self.defect_slots = defect_slots
        self.threshold = face.threshold if face.threshold is not None else 0.5
        self.last_effective_batch_size = predict_batch_size

    def predict(
        self,
        crop_paths: list[Path],
        predictions_csv: Path,
        timings: TimingRecorder | None = None,
    ) -> list[SlotResult]:
        """Return mock predictions and save the same CSV shape used by the demo."""
        recorder = timings or TimingRecorder()
        with recorder.stage("model_inference"):
            rows = []
            results = []
            for path in sorted(crop_paths):
                slot = slot_from_path(str(path))
                is_defect = slot in self.defect_slots
                score = self.threshold + 0.18 if is_defect else max(0.0, self.threshold - 0.18)
                pred_label = int(is_defect)
                results.append(SlotResult(slot, score, self.threshold, pred_label, path))
                rows.append(
                    {
                        "face": self.face.key,
                        "slot": slot,
                        "source_path": str(path.resolve()),
                        "pred_score": score,
                        "deploy_threshold": self.threshold,
                        "pred_label": pred_label,
                    },
                )
        with recorder.stage("save_results"):
            predictions_csv.parent.mkdir(parents=True, exist_ok=True)
            with predictions_csv.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        return results


def slot_from_path(value: Any) -> str:
    """Extract ``slotNN`` from a file path-like value."""
    match = re.search(r"slot\d{2}", str(value))
    return match.group(0) if match else "slot??"


def label_for_prediction(row: Any, threshold: float | None, inference: ModuleType, workflow: ModuleType) -> int:
    """Return the deployment prediction label for a prediction row."""
    value = getattr(row, "deploy_pred_label", None) if threshold is not None else getattr(row, "anomalib_pred_label", None)
    return int(inference._coerce_optional_label(value, workflow))


def slot_results_from_frame(frame: Any, threshold: float | None) -> list[SlotResult]:
    """Convert a prediction DataFrame into sorted slot results."""
    results = []
    for row in frame.sort_values("slot").itertuples(index=False):
        score = None if getattr(row, "pred_score", None) is None else float(row.pred_score)
        results.append(
            SlotResult(
                slot=row.slot,
                score=score,
                threshold=threshold,
                pred_label=int(row.pred_label),
                source_path=Path(row.source_path),
            ),
        )
    return results


def image_metrics(
    image: np.ndarray,
    *,
    dark_pixel_threshold: float = 10.0,
    clip_pixel_threshold: float = 245.0,
) -> dict[str, float | int]:
    """Return simple image quality metrics."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    dark_pct = float((gray <= dark_pixel_threshold).mean() * 100.0)
    clip_pct = float((gray >= clip_pixel_threshold).mean() * 100.0)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "channels": int(image.shape[2]) if image.ndim == 3 else 1,
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "dark_pct": dark_pct,
        "clip_pct": clip_pct,
        "sharpness": sharpness,
    }


@lru_cache(maxsize=16)
def reference_normal_mean(output_root: str, view: str, limit: int = 60) -> float | None:
    """Return cached normal-reference crop mean for quality checks."""
    normal_dir = Path(output_root) / "preprocessed" / view / "normal"
    if not normal_dir.is_dir():
        return None
    paths = sorted(normal_dir.glob("*/images/*"))[:limit]
    means = [mean for path in paths if (mean := image_gray_mean(path)) is not None]
    if not means:
        return None
    return float(sum(means) / len(means))


def evaluate_quality(
    face: FaceConfig,
    image: np.ndarray,
    crop_metrics: list[dict[str, Any]],
    args: argparse.Namespace,
) -> QualityReport:
    """Evaluate simple image and crop quality gates."""
    full_metrics = image_metrics(
        image,
        dark_pixel_threshold=args.quality_dark_pixel_threshold,
        clip_pixel_threshold=args.quality_clip_pixel_threshold,
    )
    crop_means = [float(metric["mean"]) for metric in crop_metrics]
    crop_stds = [float(metric["std"]) for metric in crop_metrics]
    crop_mean_min = min(crop_means) if crop_means else 0.0
    crop_mean_max = max(crop_means) if crop_means else 0.0
    crop_mean_avg = float(sum(crop_means) / len(crop_means)) if crop_means else 0.0
    crop_std_avg = float(sum(crop_stds) / len(crop_stds)) if crop_stds else 0.0
    reference_mean = None
    if args.quality_reference_mean_tolerance > 0:
        reference_mean = reference_normal_mean(str(face.output_root.resolve()), face.view)
    reference_delta = None if reference_mean is None else abs(crop_mean_avg - reference_mean)

    issues = []
    if len(crop_metrics) != len(face.preset.slots):
        issues.append(f"slot数量异常: {len(crop_metrics)}/{len(face.preset.slots)}")
    if float(full_metrics["mean"]) < args.quality_min_mean:
        issues.append(f"图像过暗: mean={float(full_metrics['mean']):.1f}")
    if float(full_metrics["mean"]) > args.quality_max_mean:
        issues.append(f"图像过亮: mean={float(full_metrics['mean']):.1f}")
    if float(full_metrics["dark_pct"]) > args.quality_max_dark_pct:
        issues.append(f"暗像素过多: {float(full_metrics['dark_pct']):.1f}%")
    if float(full_metrics["clip_pct"]) > args.quality_max_clip_pct:
        issues.append(f"饱和像素过多: {float(full_metrics['clip_pct']):.1f}%")
    if float(full_metrics["sharpness"]) < args.quality_min_sharpness:
        issues.append(f"图像清晰度偏低: {float(full_metrics['sharpness']):.1f}")
    if reference_delta is not None and reference_delta > args.quality_reference_mean_tolerance:
        issues.append(f"crop均值偏离训练normal: delta={reference_delta:.1f}")

    if args.quality_gate == "off":
        status = "OFF"
        issues = []
    elif issues:
        status = "FAIL" if args.quality_gate == "fail" else "WARN"
    else:
        status = "OK"

    return QualityReport(
        status=status,
        issues=issues,
        image_mean=float(full_metrics["mean"]),
        image_std=float(full_metrics["std"]),
        image_dark_pct=float(full_metrics["dark_pct"]),
        image_clip_pct=float(full_metrics["clip_pct"]),
        image_sharpness=float(full_metrics["sharpness"]),
        crop_mean_min=crop_mean_min,
        crop_mean_max=crop_mean_max,
        crop_mean_avg=crop_mean_avg,
        crop_std_avg=crop_std_avg,
        reference_mean_avg=reference_mean,
        reference_mean_delta=reference_delta,
    )


def print_quality_report(face: FaceConfig, quality: QualityReport) -> None:
    """Print one compact quality line for operators and logs."""
    issue_text = "; ".join(quality.issues) if quality.issues else "无"
    print(
        f"[quality] {face.title}: {quality.status}, "
        f"image_mean={quality.image_mean:.1f}, image_clip={quality.image_clip_pct:.2f}%, "
        f"sharpness={quality.image_sharpness:.1f}, crop_mean_avg={quality.crop_mean_avg:.1f}, "
        f"issues={issue_text}"
    )


def _quality_to_dict(quality: QualityReport | None) -> dict[str, Any] | None:
    """Convert quality report to JSON-serializable data."""
    if quality is None:
        return None
    return {
        "status": quality.status,
        "issues": quality.issues,
        "image_mean": quality.image_mean,
        "image_std": quality.image_std,
        "image_dark_pct": quality.image_dark_pct,
        "image_clip_pct": quality.image_clip_pct,
        "image_sharpness": quality.image_sharpness,
        "crop_mean_min": quality.crop_mean_min,
        "crop_mean_max": quality.crop_mean_max,
        "crop_mean_avg": quality.crop_mean_avg,
        "crop_std_avg": quality.crop_std_avg,
        "reference_mean_avg": quality.reference_mean_avg,
        "reference_mean_delta": quality.reference_mean_delta,
    }


def _slot_result_to_dict(slot: SlotResult) -> dict[str, Any]:
    """Convert a slot result to JSON-serializable data."""
    return {
        "slot": slot.slot,
        "score": slot.score,
        "threshold": slot.threshold,
        "pred_label": slot.pred_label,
        "status": slot.status,
        "source_path": str(slot.source_path),
    }


def _compact_timings(timings_ms: dict[str, float]) -> dict[str, float]:
    """Round timing values for logs and trace records."""
    return {name: round(value, 2) for name, value in sorted(timings_ms.items())}


def write_trace_record(
    result: FaceResult,
    face: FaceConfig,
    image_input: ImageInput,
    args: argparse.Namespace,
) -> Path:
    """Write a per-face JSON trace record for auditability."""
    trace_path = result.image_path.parent / f"{face.key}_trace.json"
    record = {
        "created_at": datetime.now().isoformat(timespec="milliseconds"),
        "part": face.part_key,
        "part_title": face.part_title,
        "face": face.key,
        "title": face.title,
        "view": face.view,
        "status": result.status,
        "defect_slots": result.defect_slots,
        "source": {
            "type": image_input.source,
            "path": None if image_input.source_path is None else str(image_input.source_path),
            "device": args.device if image_input.source == "camera" else None,
        },
        "hdr": {
            "short_exposure": face.short_exposure,
            "long_exposure": face.long_exposure,
            "gain": face.gain,
            "fps": args.fps,
            "timeout_ms": args.timeout_ms,
            "settle_frames": args.hdr_settle_frames,
            "align_hdr": args.align_hdr,
            "short_dark_threshold": args.short_dark_threshold,
            "long_clip_threshold": args.long_clip_threshold,
            "blend_width": args.blend_width,
            "blur_size": args.blur_size,
            "max_retries": args.hdr_max_retries,
            "max_clip_pct": args.hdr_max_clip_pct,
        },
        "crop": {
            "preset": face.preset_name,
            "crops_dir": str(result.crops_dir),
            "slot_metrics": result.crop_metrics,
        },
        "model": {
            "name": MODEL_NAME,
            "output_root": str(face.output_root),
            "checkpoint_path": None if result.checkpoint_path is None else str(result.checkpoint_path),
            "threshold": face.threshold,
            "threshold_source": face.threshold_source,
            "threshold_profile": args.threshold_profile,
            "image_size": face.image_size,
            "predict_batch_size": args.predict_batch_size,
            "effective_predict_batch_size": result.predict_batch_size,
            "predict_num_workers": args.predict_num_workers,
            "show_progress_bar": args.show_progress_bar,
            "matmul_precision": args.matmul_precision,
            "accelerator": args.accelerator,
            "devices": args.devices,
            "anomaly_dino_encoder": args.anomaly_dino_encoder,
            "anomaly_dino_neighbors": args.anomaly_dino_neighbors,
            "anomaly_dino_masking": args.anomaly_dino_masking,
            "anomaly_dino_coreset_subsampling": args.anomaly_dino_coreset_subsampling,
            "anomaly_dino_sampling_ratio": args.anomaly_dino_sampling_ratio,
        },
        "quality": _quality_to_dict(result.quality),
        "quality_gate": {
            "mode": args.quality_gate,
            "min_mean": args.quality_min_mean,
            "max_mean": args.quality_max_mean,
            "max_dark_pct": args.quality_max_dark_pct,
            "max_clip_pct": args.quality_max_clip_pct,
            "min_sharpness": args.quality_min_sharpness,
            "reference_mean_tolerance": args.quality_reference_mean_tolerance,
        },
        "slots": [_slot_result_to_dict(slot) for slot in result.slots],
        "artifacts": {
            "fused_image": str(result.image_path),
            "predictions_csv": str(result.predictions_csv),
            "trace": str(trace_path),
            "ui_screenshot": None if args.save_ui_screenshot is None else str(args.save_ui_screenshot),
        },
        "timings_ms": _compact_timings(result.timings_ms),
    }
    trace_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result.trace_path = trace_path
    return trace_path


def parse_mock_defects(values: list[str] | None) -> dict[str, set[str]]:
    """Parse mock defect arguments as ``top:slot02,slot05`` values."""
    defects = {face: set() for face in FACE_ORDER}
    for value in values or []:
        if ":" not in value:
            msg = f"Mock defect value must be face:slots, got {value!r}."
            raise argparse.ArgumentTypeError(msg)
        face, slots_text = value.split(":", maxsplit=1)
        face = face.strip()
        if face not in defects:
            msg = f"Mock defect face must be one of {FACE_ORDER}, got {face!r}."
            raise argparse.ArgumentTypeError(msg)
        slots = {slot.strip() for slot in re.split(r"[,;\s]+", slots_text) if slot.strip()}
        defects[face] = {slot for slot in slots if slot.lower() not in {"none", "ok", "normal"}}
    return defects


def _optional_float(value: str | None) -> float | None:
    """Parse a CSV value as an optional float."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _summary_threshold(output_root: Path, view: str) -> float | None:
    """Return the report summary threshold for one view."""
    summary_path = output_root / "reports" / "summary.csv"
    if not summary_path.is_file():
        return None
    with summary_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row.get("model") == MODEL_NAME and row.get("view") == view:
                return _optional_float(row.get("deploy_threshold"))
    return None


def _report_scores(output_root: Path, view: str) -> dict[str, list[float]]:
    """Return prediction scores grouped by report label for one view."""
    predictions_path = output_root / "reports" / "predictions.csv"
    scores = {"normal_test": [], "defect": []}
    if not predictions_path.is_file():
        return scores

    with predictions_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row.get("model") != MODEL_NAME or row.get("view") != view:
                continue
            label = row.get("label")
            if label not in scores:
                continue
            score = _optional_float(row.get("pred_score"))
            if score is not None:
                scores[label].append(score)
    return scores


def _resolve_face_threshold(
    args: argparse.Namespace,
    *,
    part_key: str,
    face_key: str,
    view: str,
    output_root: Path,
    explicit_threshold: float | None,
    default_threshold: float | None,
) -> tuple[float | None, str]:
    """Resolve the deployment threshold and a human-readable source."""
    if explicit_threshold is not None:
        return explicit_threshold, "命令行显式指定"

    summary_threshold = _summary_threshold(output_root, view)
    if args.threshold_profile == "report":
        if summary_threshold is None:
            return None, "report profile: summary.csv 未找到阈值"
        return summary_threshold, "report profile: reports/summary.csv"

    if default_threshold is not None and (part_key == "fx11" or face_key == "top"):
        return default_threshold, f"demo profile: {PART_PROFILES[part_key].title}现场固定阈值"

    scores = _report_scores(output_root, view)
    normal_test_scores = scores["normal_test"]
    defect_scores = scores["defect"]
    if normal_test_scores and defect_scores:
        normal_test_max = max(normal_test_scores)
        defect_min = min(defect_scores)
        if defect_min > normal_test_max:
            threshold = (normal_test_max + defect_min) / 2.0
            source = (
                "demo profile: normal_test最大值 "
                f"{normal_test_max:.6f} 与 defect最小值 {defect_min:.6f} 的中点"
            )
            return threshold, source
        source = (
            "demo profile: normal_test最大值 "
            f"{normal_test_max:.6f}; defect最小值 {defect_min:.6f} 与正常测试有重叠"
        )
        return normal_test_max, source

    if summary_threshold is not None:
        return summary_threshold, "demo profile fallback: reports/summary.csv"
    if default_threshold is not None:
        return default_threshold, f"demo profile fallback: 内置{PART_PROFILES[part_key].title}阈值"
    return None, "demo profile: 未找到阈值"


def _profile_override(args: argparse.Namespace, part_key: str, face_key: str, option: str) -> Any | None:
    """Return a side override only for the initially selected part profile."""
    if part_key != args.part_profile:
        return None
    return getattr(args, f"{face_key}_{option}")


def resolve_face_configs(args: argparse.Namespace, part_key: str) -> dict[str, FaceConfig]:
    """Build face configs for a selectable part profile."""
    profile = PART_PROFILES[part_key]
    top_gain = args.top_gain if args.top_gain is not None else args.gain
    bottom_gain = args.bottom_gain if args.bottom_gain is not None else args.gain
    gains = {
        "top": 0.0 if top_gain is None else top_gain,
        "bottom": 0.0 if bottom_gain is None else bottom_gain,
    }
    configs = {}
    for face_key in FACE_ORDER:
        defaults = profile.faces[face_key]
        output_root = _profile_override(args, part_key, face_key, "output_root") or defaults.output_root
        ckpt_path = _profile_override(args, part_key, face_key, "ckpt_path") or defaults.ckpt_path
        explicit_threshold = _profile_override(args, part_key, face_key, "threshold")
        threshold, threshold_source = _resolve_face_threshold(
            args,
            part_key=part_key,
            face_key=face_key,
            view=defaults.view,
            output_root=output_root,
            explicit_threshold=explicit_threshold,
            default_threshold=defaults.demo_threshold,
        )
        configs[face_key] = FaceConfig(
            part_key=profile.key,
            part_title=profile.title,
            key=face_key,
            title=defaults.title,
            view=defaults.view,
            preset_name=defaults.preset_name,
            output_root=output_root,
            ckpt_path=ckpt_path,
            threshold=threshold,
            threshold_source=threshold_source,
            image_size=defaults.image_size,
            short_exposure=getattr(args, f"{face_key}_short_exposure") or args.short_exposure,
            long_exposure=getattr(args, f"{face_key}_long_exposure") or args.long_exposure,
            gain=gains[face_key],
        )
    return configs


def crop_face_image(
    image: np.ndarray,
    face: FaceConfig,
    crops_dir: Path,
    timings: TimingRecorder | None = None,
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Crop and save all slots for a face."""
    crops_dir.mkdir(parents=True, exist_ok=True)
    crop_paths = []
    metrics = []
    preset = face.preset
    recorder = timings or TimingRecorder()
    for slot in preset.slots:
        with recorder.stage("crop_mask"):
            crop = crop_slot(image, preset.roi, slot)
            crop = mask_holes(crop, preset.slot_hole_masks.get(slot.name, ()), "inpaint", 9)
            metric = image_metrics(crop)
            metric["slot"] = slot.name
            metrics.append(metric)
        output_path = crops_dir / f"{face.key}_{slot.name}.png"
        with recorder.stage("save_results"):
            if not cv2.imwrite(str(output_path), crop):
                msg = f"Could not write crop: {output_path}"
                raise RuntimeError(msg)
        crop_paths.append(output_path)
    return crop_paths, metrics


def read_demo_image(path: Path) -> ImageInput:
    """Read a demo image."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        msg = f"Could not read demo image: {path}"
        raise RuntimeError(msg)
    return ImageInput(image=image, source="demo", source_path=path)


def inspect_face(
    face: FaceConfig,
    image_input: ImageInput,
    predictor: Any,
    args: argparse.Namespace,
    timings: TimingRecorder,
) -> FaceResult:
    """Crop, predict, save artifacts, and return one face result."""
    image = image_input.image
    output_dir = args.output_dir
    face_dir = output_dir / face.part_key / face.key
    face_dir.mkdir(parents=True, exist_ok=True)
    image_path = face_dir / f"{face.key}_fused.png"
    with timings.stage("save_results"):
        if not cv2.imwrite(str(image_path), image):
            msg = f"Could not write fused image: {image_path}"
            raise RuntimeError(msg)

    crops_dir = face_dir / "crops"
    crop_paths, crop_metrics = crop_face_image(image, face, crops_dir, timings)
    with timings.stage("quality_gate"):
        quality = evaluate_quality(face, image, crop_metrics, args)
    print_quality_report(face, quality)
    if quality.status == "FAIL":
        raise QualityGateError(face, quality)
    if args.diagnostics:
        print_diagnostics(face, crop_paths)
    predictions_csv = face_dir / f"{face.key}_predictions.csv"
    slots = predictor.predict(crop_paths, predictions_csv, timings)
    return FaceResult(
        face.key,
        face.title,
        image_path,
        crops_dir,
        predictions_csv,
        slots,
        quality=quality,
        crop_metrics=crop_metrics,
        checkpoint_path=getattr(predictor, "checkpoint_path", None),
        predict_batch_size=getattr(predictor, "last_effective_batch_size", args.predict_batch_size),
    )


def image_gray_mean(path: Path) -> float | None:
    """Return the grayscale mean of an image."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def describe_means(paths: list[Path]) -> str:
    """Return a compact mean-intensity summary."""
    means = [mean for path in paths if (mean := image_gray_mean(path)) is not None]
    if not means:
        return "无可读图片"
    return f"min={min(means):.1f}, max={max(means):.1f}, avg={sum(means) / len(means):.1f}, n={len(means)}"


def reference_normal_paths(face: FaceConfig, limit: int = 60) -> list[Path]:
    """Return normal reference crops from the workflow preprocessed directory."""
    normal_dir = face.output_root / "preprocessed" / face.view / "normal"
    if not normal_dir.is_dir():
        return []
    return sorted(normal_dir.glob("*/images/*"))[:limit]


def print_diagnostics(face: FaceConfig, crop_paths: list[Path]) -> None:
    """Print simple input-distribution diagnostics for one face."""
    current = describe_means(crop_paths)
    reference_paths = reference_normal_paths(face)
    reference = describe_means(reference_paths)
    print(f"[diagnostics] {face.title} 当前crop灰度均值: {current}")
    print(f"[diagnostics] {face.title} 训练normal参考灰度均值: {reference}")


def format_face_result(result: FaceResult) -> str:
    """Return the required per-face console line."""
    defects = ", ".join(result.defect_slots) if result.defect_slots else "无"
    return f"{result.title}: {result.status}, 缺陷位置: {defects}"


def format_timing_report(result: FaceResult) -> str:
    """Return a compact timing report for one face."""
    parts = []
    for key in (
        "capture_hdr",
        "crop_mask",
        "quality_gate",
        "model_inference",
        "save_results",
        "gui_render",
        "total_face",
    ):
        value = result.timings_ms.get(key)
        if value is not None:
            parts.append(f"{TIMING_LABELS.get(key, key)}={value:.1f}ms")
    return f"[timing] {result.title}: " + ", ".join(parts)


def part_status(results: dict[str, FaceResult]) -> str:
    """Return the whole-part OK/NG status."""
    if any(result.defect_slots for result in results.values()):
        return "NG"
    if len(results) == len(FACE_ORDER):
        return "OK"
    return "待检测"


def all_defect_positions(results: dict[str, FaceResult]) -> list[str]:
    """Return all defective face/slot positions."""
    positions = []
    for face in FACE_ORDER:
        result = results.get(face)
        if result is None:
            continue
        positions.extend(f"{result.title} {slot}" for slot in result.defect_slots)
    return positions


def _fit_image(image: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float, int, int]:
    """Fit an image into a target rectangle."""
    src_h, src_w = image.shape[:2]
    scale = min(width / src_w, height / src_h)
    dst_w = max(1, round(src_w * scale))
    dst_h = max(1, round(src_h * scale))
    resized = cv2.resize(image, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
    offset_x = (width - dst_w) // 2
    offset_y = (height - dst_h) // 2
    return resized, scale, offset_x, offset_y


def _load_font(size: int, *, bold: bool = False) -> Any | None:
    """Load a font that can draw Chinese text when Pillow is available."""
    try:
        from PIL import ImageFont
    except Exception:
        return None

    font_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in font_paths:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _draw_text(
    image: np.ndarray,
    text: str,
    xy: tuple[int, int],
    *,
    size: int,
    color: tuple[int, int, int],
    bold: bool = False,
) -> None:
    """Draw text on a BGR image."""
    font = _load_font(size, bold=bold)
    if font is None:
        cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, size / 32.0, color, 2, cv2.LINE_AA)
        return

    from PIL import Image, ImageDraw

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_image)
    draw.text(xy, text, font=font, fill=(color[2], color[1], color[0]))
    image[:, :] = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)


def _draw_panel(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    title: str,
    *,
    subtitle: str | None = None,
) -> None:
    """Draw a clean panel with a title."""
    x1, y1, x2, y2 = rect
    cv2.rectangle(image, (x1 + 3, y1 + 4), (x2 + 3, y2 + 4), (218, 218, 218), -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), (224, 224, 224), 1)
    cv2.rectangle(image, (x1, y1), (x2, y1 + 58), (248, 248, 248), -1)
    cv2.line(image, (x1, y1 + 58), (x2, y1 + 58), (232, 232, 232), 1)
    _draw_text(image, title, (x1 + 20, y1 + 16), size=24, color=(36, 44, 56), bold=True)
    if subtitle:
        _draw_text(image, subtitle, (x1 + 190, y1 + 19), size=17, color=(112, 122, 138))


def _draw_badge(
    image: np.ndarray,
    text: str,
    rect: tuple[int, int, int, int],
    *,
    color: tuple[int, int, int],
    text_color: tuple[int, int, int] = (255, 255, 255),
    size: int = 22,
) -> None:
    """Draw a compact status badge."""
    x1, y1, x2, y2 = rect
    cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
    _draw_text(image, text, (x1 + 14, y1 + max(5, (y2 - y1 - size) // 2)), size=size, color=text_color, bold=True)


def _status_color(status: str) -> tuple[int, int, int]:
    """Return BGR color for a status."""
    if status == "OK":
        return (84, 160, 84)
    if status == "NG":
        return (70, 84, 220)
    if status in {"WARN", "FAIL"}:
        return (42, 142, 221)
    return (150, 155, 165)


def _quality_status(result: FaceResult | None) -> str:
    """Return quality status text for display."""
    if result is None or result.quality is None:
        return "待检"
    return result.quality.status


def _face_timing_text(result: FaceResult | None) -> str:
    """Return compact timing text for one face."""
    if result is None or not result.timings_ms:
        return "耗时 --"
    total = result.timings_ms.get("total_face")
    inference = result.timings_ms.get("model_inference")
    if total is None:
        return "耗时 --"
    if inference is None:
        return f"总计 {total / 1000.0:.2f}s"
    return f"总计 {total / 1000.0:.2f}s / 推理 {inference / 1000.0:.2f}s"


def _face_summary_meta(title: str, result: FaceResult | None) -> str:
    """Return compact face status text for the summary panel."""
    if result is None:
        return f"{title} 结果待检 质量待检 耗时--"
    total = result.timings_ms.get("total_face")
    total_text = "--" if total is None else f"{total / 1000.0:.2f}s"
    return f"{title} 结果{result.status} 质量{_quality_status(result)} 总计{total_text}"


def _slot_status(result: FaceResult | None, slot: str) -> tuple[str, float | None]:
    """Return display status and score for one slot."""
    if result is None:
        return "待检", None
    found = next((item for item in result.slots if item.slot == slot), None)
    if found is None:
        return "待检", None
    return found.status, found.score


def _draw_face_result_panel(
    canvas: np.ndarray,
    face: FaceConfig,
    result: FaceResult | None,
    rect: tuple[int, int, int, int],
) -> None:
    """Draw one face result panel."""
    status = result.status if result is not None else "待检测"
    quality = _quality_status(result)
    _draw_panel(canvas, rect, f"{face.title}检测结果", subtitle=_face_timing_text(result))
    x1, y1, x2, _y2 = rect
    status = result.status if result is not None else "待检测"
    color = _status_color(status)
    _draw_badge(canvas, status, (x2 - 112, y1 + 14, x2 - 22, y1 + 46), color=color, size=21)
    quality_color = _status_color(quality)
    _draw_badge(canvas, f"质量 {quality}", (x1 + 20, y1 + 70, x1 + 130, y1 + 100), color=quality_color, size=16)
    defect_text = "缺陷 " + (", ".join(result.defect_slots) if result and result.defect_slots else "无")
    _draw_text(canvas, defect_text, (x1 + 150, y1 + 72), size=18, color=(74, 85, 101), bold=bool(result and result.defect_slots))

    slots = sorted(slot.name for slot in face.preset.slots)
    for index, slot in enumerate(slots):
        row = index // 2
        col = index % 2
        tile_x = x1 + 20 + col * 214
        tile_y = y1 + 114 + row * 58
        slot_status, score = _slot_status(result, slot)
        tile_color = _status_color("OK" if slot_status == "OK" else "NG" if slot_status == "NG" else "")
        cv2.rectangle(canvas, (tile_x, tile_y), (tile_x + 196, tile_y + 48), (248, 250, 252), -1)
        cv2.rectangle(canvas, (tile_x, tile_y), (tile_x + 196, tile_y + 48), (226, 231, 238), 1)
        cv2.rectangle(canvas, (tile_x, tile_y), (tile_x + 6, tile_y + 48), tile_color, -1)
        _draw_text(canvas, slot, (tile_x + 16, tile_y + 7), size=18, color=(35, 45, 58), bold=True)
        _draw_text(canvas, slot_status, (tile_x + 126, tile_y + 7), size=18, color=tile_color, bold=True)
        score_text = "--" if score is None else f"{score:.3f}"
        _draw_text(canvas, score_text, (tile_x + 16, tile_y + 27), size=14, color=(96, 106, 120))
        threshold = None
        if result is not None:
            found = next((item for item in result.slots if item.slot == slot), None)
            threshold = None if found is None else found.threshold
        if score is not None and threshold is not None and threshold > 0:
            bar_x, bar_y = tile_x + 76, tile_y + 34
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + 102, bar_y + 6), (225, 230, 236), -1)
            ratio = max(0.0, min(float(score) / max(float(threshold) * 1.4, 1e-6), 1.0))
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + int(102 * ratio), bar_y + 6), tile_color, -1)


def _slot_box_in_display(
    slot: SlotSpec,
    face: FaceConfig,
    scale: float,
    offset_x: int,
    offset_y: int,
    origin: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return a slot box scaled to the current image display."""
    roi = face.preset.roi
    ox, oy = origin
    x1 = ox + offset_x + round((roi.x1 + slot.box.x1) * scale)
    y1 = oy + offset_y + round((roi.y1 + slot.box.y1) * scale)
    x2 = ox + offset_x + round((roi.x1 + slot.box.x2) * scale)
    y2 = oy + offset_y + round((roi.y1 + slot.box.y2) * scale)
    return x1, y1, x2, y2


def _draw_part_switcher(canvas: np.ndarray, state: DemoState, x: int, y: int) -> None:
    """Draw keyboard-selectable part profile buttons."""
    for index, part_key in enumerate(PART_ORDER, start=1):
        profile = PART_PROFILES[part_key]
        active = part_key == state.part_key
        button_x = x + (index - 1) * 154
        color = (126, 139, 45) if active else (104, 96, 88)
        cv2.rectangle(canvas, (button_x, y), (button_x + 138, y + 34), color, -1)
        cv2.rectangle(canvas, (button_x, y), (button_x + 138, y + 34), (184, 194, 204), 1)
        _draw_text(
            canvas,
            f"{index}  {profile.title}",
            (button_x + 14, y + 7),
            size=18,
            color=(255, 255, 255),
            bold=active,
        )


def render_dashboard(state: DemoState, face_configs: dict[str, FaceConfig], width: int = 1600, height: int = 920) -> np.ndarray:
    """Render the operator dashboard as a BGR image."""
    canvas = np.full((height, width, 3), (247, 247, 247), dtype=np.uint8)

    profile = PART_PROFILES[state.part_key]
    cv2.rectangle(canvas, (0, 0), (width, 100), (52, 49, 46), -1)
    cv2.rectangle(canvas, (0, 96), (width, 100), (126, 139, 45), -1)
    _draw_text(canvas, f"{profile.title} 双面缺陷检测", (34, 16), size=34, color=(255, 255, 255), bold=True)
    _draw_text(canvas, profile.description, (38, 58), size=17, color=(204, 215, 226), bold=True)
    _draw_badge(canvas, state.status_message, (380, 24, 1018, 66), color=(82, 75, 70), size=20)
    _draw_part_switcher(canvas, state, 1070, 16)

    top_title = face_configs["top"].title
    bottom_title = face_configs["bottom"].title
    top_stage = "OK" if "top" in state.results and state.results["top"].status == "OK" else "NG" if "top" in state.results else "待检"
    bottom_stage = "OK" if "bottom" in state.results and state.results["bottom"].status == "OK" else "NG" if "bottom" in state.results else "待检"
    _draw_badge(canvas, f"{top_title} {top_stage}", (1070, 56, 1302, 88), color=_status_color(top_stage), size=17)
    _draw_badge(canvas, f"{bottom_title} {bottom_stage}", (1320, 56, 1552, 88), color=_status_color(bottom_stage), size=17)

    image_rect = (34, 112, 1068, 742)
    active_face = state.current_face or state.active_face or "top"
    active_config = face_configs[active_face]
    active_title = active_config.title
    _draw_panel(canvas, image_rect, "当前检测画面", subtitle=f"{state.part_title} / 当前面: {active_title}")
    image_x, image_y = 54, 176
    image_w, image_h = 988, 522
    cv2.rectangle(canvas, (image_x, image_y), (image_x + image_w, image_y + image_h), (234, 234, 234), -1)
    cv2.rectangle(canvas, (image_x, image_y), (image_x + image_w, image_y + image_h), (214, 214, 214), 1)

    if state.current_image is not None:
        display, scale, off_x, off_y = _fit_image(state.current_image, image_w, image_h)
        canvas[image_y + off_y : image_y + off_y + display.shape[0], image_x + off_x : image_x + off_x + display.shape[1]] = display
        active_result = state.results.get(active_face)
        for slot in active_config.preset.slots:
            status, _score = _slot_status(active_result, slot.name)
            color = _status_color("OK" if status == "OK" else "NG" if status == "NG" else "")
            thickness = 4 if status == "NG" else 2
            box = _slot_box_in_display(slot, active_config, scale, off_x, off_y, (image_x, image_y))
            cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), color, thickness)
            _draw_text(canvas, slot.name, (box[0] + 8, box[1] + 8), size=18, color=color, bold=True)
        if active_result is not None:
            result_color = _status_color(active_result.status)
            _draw_badge(canvas, f"{active_title} {active_result.status}", (image_x + 18, image_y + 18, image_x + 170, image_y + 54), color=result_color, size=20)
    else:
        cv2.circle(canvas, (image_x + 390, image_y + 252), 34, (210, 218, 228), -1)
        _draw_text(canvas, "等待上料并按 s", (image_x + 442, image_y + 226), size=31, color=(92, 104, 120), bold=True)
        _draw_text(canvas, f"下一步: {active_title}", (image_x + 442, image_y + 268), size=22, color=(122, 134, 150))

    _draw_face_result_panel(canvas, face_configs["top"], state.results.get("top"), (1100, 112, 1566, 412))
    _draw_face_result_panel(canvas, face_configs["bottom"], state.results.get("bottom"), (1100, 432, 1566, 732))

    summary_rect = (34, 768, 1566, 892)
    _draw_panel(canvas, summary_rect, "整件汇总", subtitle="双面任一 NG 则整件 NG")
    status = part_status(state.results)
    color = _status_color(status)
    _draw_badge(canvas, status, (200, 808, 342, 858), color=color, size=30)
    defects = all_defect_positions(state.results)
    defect_text = "缺陷位置: " + (", ".join(defects) if defects else "无")
    _draw_text(canvas, defect_text, (390, 806), size=24, color=(40, 49, 63), bold=bool(defects))
    top_result = state.results.get("top")
    bottom_result = state.results.get("bottom")
    top_meta = _face_summary_meta("正面", top_result)
    bottom_meta = _face_summary_meta("底面", bottom_result)
    _draw_text(canvas, top_meta, (390, 844), size=17, color=(92, 104, 120))
    _draw_text(canvas, bottom_meta, (790, 844), size=17, color=(92, 104, 120))
    _draw_text(canvas, "1/2 切换零件   s 连续检测   r 重置   q 退出", (1120, 844), size=18, color=(92, 104, 120), bold=True)
    return canvas


def save_dashboard(
    path: Path,
    state: DemoState,
    face_configs: dict[str, FaceConfig],
    timings: TimingRecorder | None = None,
) -> None:
    """Save the current dashboard canvas."""
    path.parent.mkdir(parents=True, exist_ok=True)
    recorder = timings or TimingRecorder()
    with recorder.stage("gui_render"):
        canvas = render_dashboard(state, face_configs)
    with recorder.stage("save_results"):
        cv2.imwrite(str(path), canvas)


def next_status_after_result(
    state: DemoState,
    face: FaceConfig,
    result: FaceResult,
    face_configs: dict[str, FaceConfig],
) -> str:
    """Return the next status message after a face result."""
    if state.finished:
        first_title = face_configs[FACE_ORDER[0]].title
        return f"{face.title}检测完成: {result.status}。双面检测结束，再按 s 检测下一件{first_title}"
    next_face_key = state.active_face
    next_title = face_configs[next_face_key].title if next_face_key else FACE_ORDER[-1]
    return f"{face.title}检测完成: {result.status}。请切换到{next_title}后按 s 检测"


def run_face(
    state: DemoState,
    face: FaceConfig,
    image_input: ImageInput,
    predictor: Any,
    face_configs: dict[str, FaceConfig],
    args: argparse.Namespace,
    timings: TimingRecorder,
) -> FaceResult:
    """Run one face and update UI state."""
    state.current_face = face.key
    state.current_image = image_input.image
    state.status_message = f"{face.title}裁剪与推理中..."
    result = inspect_face(face, image_input, predictor, args, timings)
    state.results[face.key] = result
    state.active_index += 1
    state.status_message = next_status_after_result(state, face, result, face_configs)
    print(format_face_result(result))
    if state.finished:
        defects = all_defect_positions(state.results)
        print(f"整件: {part_status(state.results)}, 缺陷位置: {', '.join(defects) if defects else '无'}")
    return result


def finalize_face_result(
    result: FaceResult,
    face: FaceConfig,
    image_input: ImageInput,
    args: argparse.Namespace,
    timings: TimingRecorder,
) -> None:
    """Attach timings, write trace, and print timing report."""
    result.timings_ms = dict(timings.stages_ms)
    write_trace_record(result, face, image_input, args)
    print(format_timing_report(result))


def build_predictors(
    face_configs: dict[str, FaceConfig],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build real or mock predictors for all faces."""
    mock_defects = parse_mock_defects(args.mock_defects)
    predictors = {}
    for face_key in FACE_ORDER:
        face = face_configs[face_key]
        if args.mock_predictions:
            predictors[face_key] = MockPredictor(face, mock_defects[face_key], args.predict_batch_size)
        else:
            predictors[face_key] = AnomalibPredictor(face, args)
    return predictors


def clear_runtime_cache() -> None:
    """Release Python and CUDA memory after replacing model predictors."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def release_predictors(predictors: dict[str, Any]) -> None:
    """Drop model predictors before loading another part profile."""
    predictors.clear()
    clear_runtime_cache()


def initial_status_message(face_configs: dict[str, FaceConfig]) -> str:
    """Return the waiting message for the first face of the active part."""
    first_face = face_configs[FACE_ORDER[0]]
    return f"等待{first_face.title}上料，按 s 开始检测"


def configure_state_for_part(state: DemoState, part_key: str, face_configs: dict[str, FaceConfig]) -> None:
    """Reset UI state for a newly selected part profile."""
    profile = PART_PROFILES[part_key]
    state.part_key = part_key
    state.part_title = profile.title
    state.active_index = 0
    state.status_message = initial_status_message(face_configs)
    state.current_face = None
    state.current_image = None
    state.results.clear()


def load_part_runtime(
    state: DemoState,
    args: argparse.Namespace,
    part_key: str,
) -> tuple[dict[str, FaceConfig], dict[str, Any]]:
    """Resolve configs and load predictors for a part profile."""
    face_configs = resolve_face_configs(args, part_key)
    validate_checkpoint_sources(face_configs, args)
    predictors = build_predictors(face_configs, args)
    configure_state_for_part(state, part_key, face_configs)
    return face_configs, predictors


def validate_checkpoint_sources(face_configs: dict[str, FaceConfig], args: argparse.Namespace) -> None:
    """Fail early if a selected profile has no visible checkpoint source."""
    if args.mock_predictions:
        return
    for face in face_configs.values():
        if face.ckpt_path is not None:
            if not face.ckpt_path.is_file():
                msg = f"{face.part_title} {face.title} checkpoint does not exist: {face.ckpt_path}"
                raise FileNotFoundError(msg)
            continue
        if not face.output_root.is_dir() or not any(face.output_root.rglob("*.ckpt")):
            msg = f"{face.part_title} {face.title} has no checkpoint under {face.output_root}"
            raise FileNotFoundError(msg)


def part_key_from_keyboard(key: int) -> str | None:
    """Map number keys to part profile keys."""
    for index, part_key in enumerate(PART_ORDER, start=1):
        if key == ord(str(index)):
            return part_key
    return None


def image_for_face(face_key: str, args: argparse.Namespace, camera: HdrCamera | None, face: FaceConfig) -> ImageInput:
    """Return an image from demo input or camera capture."""
    demo_path = args.demo_top_image if face_key == "top" else args.demo_bottom_image
    if demo_path is not None:
        return read_demo_image(demo_path)
    if camera is None:
        msg = f"No demo image was provided for {face_key}, and camera is not available."
        raise RuntimeError(msg)
    return ImageInput(image=camera.capture(face), source="camera")


def reset_state(state: DemoState, face_configs: dict[str, FaceConfig]) -> None:
    """Reset the current part inspection flow."""
    state.active_index = 0
    state.status_message = initial_status_message(face_configs)
    state.current_face = None
    state.current_image = None
    state.results.clear()


def run_auto_demo(
    state: DemoState,
    face_configs: dict[str, FaceConfig],
    predictors: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """Run both sides without opening a GUI window."""
    for face_key in FACE_ORDER:
        face = face_configs[face_key]
        timings = TimingRecorder()
        with timings.stage("total_face"):
            with timings.stage("capture_hdr"):
                image_input = image_for_face(face_key, args, None, face)
            result = run_face(state, face, image_input, predictors[face_key], face_configs, args, timings)
            with timings.stage("gui_render"):
                render_dashboard(state, face_configs)
        finalize_face_result(result, face, image_input, args, timings)
    if args.save_ui_screenshot is not None:
        save_dashboard(args.save_ui_screenshot, state, face_configs)


def run_gui(
    state: DemoState,
    face_configs: dict[str, FaceConfig],
    predictors: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """Run the OpenCV keyboard-driven UI."""
    needs_camera = args.demo_top_image is None or args.demo_bottom_image is None
    camera_context = HdrCamera(
        device=args.device,
        fps=args.fps,
        timeout_ms=args.timeout_ms,
        settle_frames=args.hdr_settle_frames,
        align_hdr=args.align_hdr,
        short_dark_threshold=args.short_dark_threshold,
        long_clip_threshold=args.long_clip_threshold,
        blend_width=args.blend_width,
        blur_size=args.blur_size,
        hdr_max_retries=args.hdr_max_retries,
        hdr_max_clip_pct=args.hdr_max_clip_pct,
    )
    camera = camera_context.__enter__() if needs_camera else None
    try:
        if args.save_ui_screenshot is not None:
            args.save_ui_screenshot.parent.mkdir(parents=True, exist_ok=True)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        while True:
            canvas = render_dashboard(state, face_configs)
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(50) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                reset_state(state, face_configs)
                continue
            selected_part = part_key_from_keyboard(key)
            if selected_part is not None:
                if selected_part == state.part_key:
                    state.status_message = f"当前已是 {state.part_title}，按 s 开始检测"
                    continue
                profile = PART_PROFILES[selected_part]
                state.status_message = f"切换到 {profile.title}，模型加载中..."
                canvas = render_dashboard(state, face_configs)
                cv2.imshow(WINDOW_NAME, canvas)
                cv2.waitKey(1)
                try:
                    new_face_configs = resolve_face_configs(args, selected_part)
                    validate_checkpoint_sources(new_face_configs, args)
                    release_predictors(predictors)
                    new_predictors = build_predictors(new_face_configs, args)
                except Exception as error:
                    state.status_message = f"{profile.title}模型加载失败: {error}"
                    print(f"[runtime] {state.status_message}")
                    continue
                face_configs = new_face_configs
                predictors = new_predictors
                configure_state_for_part(state, selected_part, face_configs)
                print_startup_settings(face_configs, args)
                continue
            if key != ord("s"):
                continue
            if state.finished:
                reset_state(state, face_configs)

            face_key = state.active_face
            if face_key is None:
                continue
            if face_key not in predictors:
                state.status_message = f"{state.part_title}模型未加载，请重新切换零件"
                continue
            face = face_configs[face_key]
            timings = TimingRecorder()
            state.status_message = f"{face.title}采集中..."
            with timings.stage("gui_render"):
                canvas = render_dashboard(state, face_configs)
                cv2.imshow(WINDOW_NAME, canvas)
            if args.save_ui_screenshot is not None:
                with timings.stage("save_results"):
                    cv2.imwrite(str(args.save_ui_screenshot), canvas)
            cv2.waitKey(1)
            try:
                with timings.stage("total_face"):
                    with timings.stage("capture_hdr"):
                        image_input = image_for_face(face_key, args, camera, face)
                    result = run_face(state, face, image_input, predictors[face_key], face_configs, args, timings)
                    with timings.stage("gui_render"):
                        canvas = render_dashboard(state, face_configs)
                        cv2.imshow(WINDOW_NAME, canvas)
                    if args.save_ui_screenshot is not None:
                        with timings.stage("save_results"):
                            cv2.imwrite(str(args.save_ui_screenshot), canvas)
                finalize_face_result(result, face, image_input, args, timings)
            except QualityGateError as error:
                state.status_message = str(error)
                print(f"[quality] {error}")
    finally:
        cv2.destroyWindow(WINDOW_NAME)
        if camera is not None:
            camera_context.__exit__(None, None, None)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, default=0, help="Camera device index.")
    parser.add_argument("--output-dir", type=Path, help="Demo run output directory.")
    parser.add_argument(
        "--part-profile",
        choices=PART_ORDER,
        default=DEFAULT_PART_PROFILE,
        help="Initial part profile. In GUI mode, press 1/2 to switch profiles and reload models.",
    )
    parser.add_argument(
        "--top-output-root",
        type=Path,
        help="Optional top output-root override for the initially selected part profile.",
    )
    parser.add_argument(
        "--bottom-output-root",
        type=Path,
        help="Optional bottom output-root override for the initially selected part profile.",
    )
    parser.add_argument("--top-ckpt-path", type=Path, help="Optional explicit top checkpoint for the initial profile.")
    parser.add_argument("--bottom-ckpt-path", type=Path, help="Optional explicit bottom checkpoint for the initial profile.")
    parser.add_argument("--top-threshold", type=float, help="Optional explicit top threshold for the initial profile.")
    parser.add_argument("--bottom-threshold", type=float, help="Optional explicit bottom threshold for the initial profile.")
    parser.add_argument(
        "--threshold-profile",
        choices=THRESHOLD_PROFILES,
        default="demo",
        help=(
            "Threshold source. 'demo' calibrates from reports/predictions.csv; "
            "'report' uses reports/summary.csv. Explicit side thresholds override this."
        ),
    )

    parser.add_argument("--short-exposure", type=float, default=4000.0, help="Default HDR short exposure.")
    parser.add_argument("--long-exposure", type=float, default=35000.0, help="Default HDR long exposure.")
    parser.add_argument("--top-short-exposure", type=float, help="Top-side HDR short exposure override.")
    parser.add_argument("--top-long-exposure", type=float, help="Top-side HDR long exposure override.")
    parser.add_argument("--bottom-short-exposure", type=float, help="Bottom-side HDR short exposure override.")
    parser.add_argument("--bottom-long-exposure", type=float, help="Bottom-side HDR long exposure override.")
    parser.add_argument("--gain", type=float, help="Camera gain used for both sides unless side-specific gain is set.")
    parser.add_argument("--top-gain", type=float, help="Top-side camera gain override.")
    parser.add_argument("--bottom-gain", type=float, help="Bottom-side camera gain override.")
    parser.add_argument("--fps", type=float, default=10.0, help="Camera frame rate.")
    parser.add_argument("--timeout-ms", type=int, default=3000, help="Frame timeout.")
    parser.add_argument("--hdr-settle-frames", type=int, default=8, help="Frames discarded after exposure changes.")
    parser.add_argument("--align-hdr", action="store_true", help="Align HDR source frames before fusion.")
    parser.add_argument("--short-dark-threshold", type=float, default=80.0, help="Selective HDR dark threshold.")
    parser.add_argument("--long-clip-threshold", type=float, default=245.0, help="Selective HDR clip threshold.")
    parser.add_argument("--blend-width", type=float, default=50.0, help="Selective HDR blend width.")
    parser.add_argument("--blur-size", type=int, default=101, help="Selective HDR blur size.")
    parser.add_argument("--hdr-max-retries", type=int, default=1, help="Retries when fused image is clipped.")
    parser.add_argument("--hdr-max-clip-pct", type=float, default=12.0, help="Max acceptable clipped pixel percentage.")

    parser.add_argument("--demo-top-image", type=Path, help="Use a local full-size top image instead of camera capture.")
    parser.add_argument("--demo-bottom-image", type=Path, help="Use a local full-size bottom image instead of camera capture.")
    parser.add_argument("--mock-predictions", action="store_true", help="Use deterministic mock predictions.")
    parser.add_argument(
        "--mock-defects",
        action="append",
        help="Mock defects as face:slots, e.g. top:slot02,slot05 or bottom:none.",
    )
    parser.add_argument("--auto-run", action="store_true", help="Run top and bottom once without keyboard input.")
    parser.add_argument("--no-gui", action="store_true", help="Do not open the OpenCV window.")
    parser.add_argument("--save-ui-screenshot", type=Path, help="Save the rendered UI canvas.")
    parser.add_argument("--diagnostics", action="store_true", help="Print crop brightness diagnostics.")
    parser.add_argument(
        "--quality-gate",
        choices=("warn", "fail", "off"),
        default="warn",
        help="Image quality gate behavior before prediction.",
    )
    parser.add_argument("--quality-min-mean", type=float, default=20.0, help="Minimum full-image grayscale mean.")
    parser.add_argument("--quality-max-mean", type=float, default=210.0, help="Maximum full-image grayscale mean.")
    parser.add_argument("--quality-max-dark-pct", type=float, default=75.0, help="Maximum dark-pixel percentage.")
    parser.add_argument("--quality-max-clip-pct", type=float, default=10.0, help="Maximum clipped-pixel percentage.")
    parser.add_argument("--quality-min-sharpness", type=float, default=5.0, help="Minimum Laplacian sharpness.")
    parser.add_argument(
        "--quality-reference-mean-tolerance",
        type=float,
        default=0.0,
        help="Allowed crop mean delta from training normal reference. Use <=0 to disable this slower check.",
    )
    parser.add_argument(
        "--quality-dark-pixel-threshold",
        type=float,
        default=10.0,
        help="Pixel value used for dark-pixel quality percentage.",
    )
    parser.add_argument(
        "--quality-clip-pixel-threshold",
        type=float,
        default=245.0,
        help="Pixel value used for clipped-pixel quality percentage.",
    )

    parser.add_argument("--accelerator", choices=("gpu", "cpu", "auto"), default="gpu", help="Lightning accelerator.")
    parser.add_argument("--devices", type=int, default=1, help="Number of accelerator devices.")
    parser.add_argument("--predict-batch-size", type=int, default=1, help="Batch size for slot crop prediction.")
    parser.add_argument("--predict-num-workers", type=int, default=0, help="DataLoader workers for slot prediction.")
    parser.add_argument("--show-progress-bar", action="store_true", help="Show Lightning prediction progress bars.")
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium", "none"),
        default="high",
        help="Torch float32 matmul precision used before model loading.",
    )
    parser.add_argument(
        "--imagenet-dir",
        type=Path,
        default=REPO_ROOT / "datasets" / "imagenette",
        help="EfficientAd ImageNette dir; kept for inference helper compatibility.",
    )
    parser.add_argument("--anomaly-dino-neighbors", type=int, default=1, help="AnomalyDINO nearest-neighbor count.")
    parser.add_argument("--anomaly-dino-encoder", default="dinov2_vit_small_14", help="AnomalyDINO encoder name.")
    parser.add_argument("--anomaly-dino-masking", action="store_true", help="Enable AnomalyDINO foreground masking.")
    parser.add_argument("--anomaly-dino-coreset-subsampling", action="store_true", help="Enable coreset subsampling.")
    parser.add_argument("--anomaly-dino-sampling-ratio", type=float, default=0.1, help="AnomalyDINO sampling ratio.")
    return parser


def _format_threshold(value: float | None) -> str:
    """Return a compact threshold string."""
    return "none" if value is None else f"{value:.6f}"


def print_startup_settings(face_configs: dict[str, FaceConfig], args: argparse.Namespace) -> None:
    """Print the deployment settings that affect the live demo."""
    first_face = face_configs[FACE_ORDER[0]]
    print(f"part profile: {first_face.part_title} ({first_face.part_key})")
    print(f"threshold profile: {args.threshold_profile}")
    for face_key in FACE_ORDER:
        face = face_configs[face_key]
        print(f"{face.title} threshold: {_format_threshold(face.threshold)} ({face.threshold_source})")
        print(f"{face.title} checkpoint: {face.ckpt_path or 'auto'}")
        if face.image_size is not None:
            print(f"{face.title} image_size: {face.image_size[0]},{face.image_size[1]}")
    print(
        "HDR capture: "
        f"top gain={face_configs['top'].gain:g}, bottom gain={face_configs['bottom'].gain:g}, "
        f"short={args.short_exposure:g}, long={args.long_exposure:g}, "
        f"settle={args.hdr_settle_frames}, dark={args.short_dark_threshold:g}, "
        f"clip={args.long_clip_threshold:g}, blend={args.blend_width:g}, blur={args.blur_size}"
    )
    print(
        "Runtime: "
        f"predict_batch_size={args.predict_batch_size}, workers={args.predict_num_workers}, "
        f"progress_bar={args.show_progress_bar}, matmul_precision={args.matmul_precision}, "
        f"quality_gate={args.quality_gate}"
    )


def configure_runtime(args: argparse.Namespace) -> None:
    """Configure low-risk runtime settings for the live demo."""
    if args.predict_batch_size <= 0:
        msg = "--predict-batch-size must be positive."
        raise SystemExit(msg)
    if args.predict_num_workers < 0:
        msg = "--predict-num-workers must be non-negative."
        raise SystemExit(msg)
    if args.matmul_precision == "none":
        return
    try:
        import torch

        torch.set_float32_matmul_precision(args.matmul_precision)
    except Exception as error:
        print(f"[runtime] Could not set torch matmul precision: {error}")


def main() -> None:
    """Run the two-sided inspection demo."""
    args = build_parser().parse_args()
    configure_runtime(args)
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_group = "fx11_demo" if args.part_profile == "fx11" else args.part_profile
        args.output_dir = REPO_ROOT / "results" / output_group / "demo_inspection" / timestamp
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.no_gui and not args.auto_run:
        args.auto_run = True
    if args.auto_run and (args.demo_top_image is None or args.demo_bottom_image is None):
        msg = "--auto-run requires --demo-top-image and --demo-bottom-image."
        raise SystemExit(msg)

    state = DemoState(part_key=args.part_profile, part_title=PART_PROFILES[args.part_profile].title)
    face_configs, predictors = load_part_runtime(state, args, args.part_profile)
    print_startup_settings(face_configs, args)

    if args.auto_run:
        run_auto_demo(state, face_configs, predictors, args)
    else:
        run_gui(state, face_configs, predictors, args)


if __name__ == "__main__":
    main()
