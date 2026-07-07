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
from dataclasses import dataclass, field, replace
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
        "bottom": 0.48,
    },
    "c789": {
        "top": 0.55,
        "bottom": 0.55,
    },
}
TIMING_LABELS = {
    "capture_hdr": "采集HDR",
    "crop_mask": "裁剪mask",
    "stamp_ocr": "钢印识别",
    "quality_gate": "质量门控",
    "model_inference": "模型推理",
    "save_results": "保存结果",
    "gui_render": "GUI渲染",
    "total_face": "单面总计",
}
DEFAULT_DEFECT_MAP_THRESHOLD = 0.65
DEFAULT_DEFECT_MIN_AREA = 64
STAMP_ROI_DEFAULTS = {
    ("c789", "bottom"): {
        "x1": 620,
        "y1": 60,
        "x2": 1040,
        "y2": 285,
        "rotate_180": True,
        "mirror_horizontal": True,
    },
    ("fx11", "top"): {
        "x1": 820,
        "y1": 115,
        "x2": 1260,
        "y2": 285,
        "rotate_180": False,
        "mirror_horizontal": True,
    },
}
ARCHIVE_SLOT_FIELDS = (
    "created_at",
    "inspection_id",
    "part",
    "part_title",
    "face",
    "face_title",
    "slot",
    "status",
    "score",
    "threshold",
    "quality_status",
    "quality_reasons",
    "defect_regions_json",
    "part_number",
    "stamp_date",
    "ocr_status",
    "ocr_confidence",
    "ocr_text",
    "fused_image",
    "annotated_image",
    "crop_image",
    "stamp_roi_image",
    "stamp_enhanced_image",
    "predictions_csv",
    "trace_json",
)
ARCHIVE_PART_FIELDS = (
    "created_at",
    "inspection_id",
    "part",
    "part_title",
    "slot",
    "status",
    "defect_faces",
    "part_number",
    "stamp_date",
    "ocr_status",
    "top_score",
    "bottom_score",
    "top_status",
    "bottom_status",
    "top_quality_status",
    "top_quality_reasons",
    "bottom_quality_status",
    "bottom_quality_reasons",
    "top_crop_image",
    "bottom_crop_image",
)


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
class StampOcrResult:
    """Recognized U-stamp data for one slot."""

    status: str
    part_number: str | None = None
    stamp_date: str | None = None
    raw_text: str = ""
    confidence: float = 0.0
    roi_path: Path | None = None
    enhanced_path: Path | None = None
    message: str = ""


@dataclass(frozen=True)
class DefectRegion:
    """One predicted defect region in crop and full-image coordinates."""

    slot: str
    bbox: tuple[int, int, int, int]
    full_bbox: tuple[int, int, int, int]
    area: float
    contour: tuple[tuple[int, int], ...]
    full_contour: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class SlotResult:
    """Prediction result for one slot."""

    slot: str
    score: float | None
    threshold: float | None
    pred_label: int
    source_path: Path
    defect_regions: tuple[DefectRegion, ...] = ()
    stamp_ocr: StampOcrResult | None = None

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
    annotated_image_path: Path | None = None
    archive_slot_rows: list[dict[str, Any]] = field(default_factory=list)
    archive_part_rows: list[dict[str, Any]] = field(default_factory=list)

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
        self.defect_map_threshold = args.defect_map_threshold
        self.defect_min_area = args.defect_min_area
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
            artifacts = self.inference._prediction_artifacts(predictions, self.workflow)
            predictions_csv.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(predictions_csv, index=False)
        return slot_results_from_frame_with_artifacts(
            frame,
            self.threshold,
            self.face,
            artifacts,
            map_threshold=self.defect_map_threshold,
            min_area=self.defect_min_area,
        )


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
                slot_spec = next((item for item in self.face.preset.slots if item.name == slot), None)
                regions = (
                    (synthetic_defect_region(self.face, slot_spec),)
                    if is_defect and slot_spec is not None
                    else ()
                )
                results.append(SlotResult(slot, score, self.threshold, pred_label, path, defect_regions=regions))
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
    value = (
        getattr(row, "deploy_pred_label", None)
        if threshold is not None
        else getattr(row, "anomalib_pred_label", None)
    )
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


def _to_numpy_array(value: Any) -> np.ndarray | None:
    """Convert tensor-like prediction output to a numpy array."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        array = np.asarray(value)
    except Exception:
        return None
    if array.size == 0:
        return None
    return np.squeeze(array)


def _normalize_float_map(array: np.ndarray) -> np.ndarray:
    """Normalize a numeric map to 0..1."""
    data = array.astype("float32", copy=False)
    minimum = float(np.nanmin(data))
    maximum = float(np.nanmax(data))
    if maximum <= minimum:
        return np.zeros_like(data, dtype="float32")
    return (data - minimum) / (maximum - minimum)


def _mask_from_artifact(
    artifact: dict[str, Any] | None,
    *,
    crop_size: tuple[int, int],
    threshold: float,
) -> np.ndarray | None:
    """Return a crop-size binary defect mask from a prediction artifact."""
    if artifact is None:
        return None
    width, height = crop_size
    mask = _to_numpy_array(artifact.get("pred_mask"))
    if mask is not None:
        if mask.ndim > 2:
            mask = np.squeeze(mask)
        mask = mask.astype("float32")
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return mask > 0.5

    anomaly_map = _to_numpy_array(artifact.get("anomaly_map"))
    if anomaly_map is None:
        return None
    if anomaly_map.ndim > 2:
        anomaly_map = np.squeeze(anomaly_map)
    anomaly_map = _normalize_float_map(anomaly_map)
    if anomaly_map.shape[:2] != (height, width):
        anomaly_map = cv2.resize(anomaly_map, (width, height), interpolation=cv2.INTER_LINEAR)
    return anomaly_map > threshold


def _tuple_points(contour: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Convert an OpenCV contour to plain integer points."""
    points = contour.reshape(-1, 2)
    return tuple((int(x), int(y)) for x, y in points)


def defect_regions_from_artifact(
    artifact: dict[str, Any] | None,
    face: FaceConfig,
    slot: SlotSpec,
    *,
    map_threshold: float = DEFAULT_DEFECT_MAP_THRESHOLD,
    min_area: float = DEFAULT_DEFECT_MIN_AREA,
) -> tuple[DefectRegion, ...]:
    """Extract full-image defect regions for one slot prediction artifact."""
    mask = _mask_from_artifact(
        artifact,
        crop_size=(slot.box.width, slot.box.height),
        threshold=map_threshold,
    )
    if mask is None:
        return ()

    contours, _hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    offset_x = face.preset.roi.x1 + slot.box.x1
    offset_y = face.preset.roi.y1 + slot.box.y1
    regions = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        points = _tuple_points(contour)
        full_points = tuple((x_point + offset_x, y_point + offset_y) for x_point, y_point in points)
        regions.append(
            DefectRegion(
                slot=slot.name,
                bbox=(int(x), int(y), int(x + width), int(y + height)),
                full_bbox=(int(x + offset_x), int(y + offset_y), int(x + width + offset_x), int(y + height + offset_y)),
                area=area,
                contour=points,
                full_contour=full_points,
            ),
        )
    return tuple(sorted(regions, key=lambda item: item.area, reverse=True))


def defect_region_to_dict(region: DefectRegion) -> dict[str, Any]:
    """Convert a defect region to JSON-serializable data."""
    return {
        "slot": region.slot,
        "bbox": list(region.bbox),
        "full_bbox": list(region.full_bbox),
        "area": region.area,
        "contour": [list(point) for point in region.contour],
        "full_contour": [list(point) for point in region.full_contour],
    }


def synthetic_defect_region(face: FaceConfig, slot: SlotSpec) -> DefectRegion:
    """Return a deterministic mock defect region inside one slot."""
    width, height = slot.box.width, slot.box.height
    x1 = max(0, int(width * 0.42))
    y1 = max(0, int(height * 0.38))
    x2 = min(width, max(x1 + 1, int(width * 0.58)))
    y2 = min(height, max(y1 + 1, int(height * 0.62)))
    contour = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    offset_x = face.preset.roi.x1 + slot.box.x1
    offset_y = face.preset.roi.y1 + slot.box.y1
    full_contour = tuple((x + offset_x, y + offset_y) for x, y in contour)
    return DefectRegion(
        slot=slot.name,
        bbox=(x1, y1, x2, y2),
        full_bbox=(x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
        area=float((x2 - x1) * (y2 - y1)),
        contour=contour,
        full_contour=full_contour,
    )


def _artifact_for_row(row: Any, artifacts: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Return the prediction artifact for a frame row."""
    for attribute in ("processed_path", "source_path"):
        value = getattr(row, attribute, None)
        if value is None:
            continue
        artifact = artifacts.get(str(Path(value).resolve()))
        if artifact is not None:
            return artifact
    return None


def slot_results_from_frame_with_artifacts(
    frame: Any,
    threshold: float | None,
    face: FaceConfig,
    artifacts: dict[str, dict[str, Any]],
    *,
    map_threshold: float,
    min_area: float,
) -> list[SlotResult]:
    """Convert predictions to slot results with optional defect regions."""
    slots_by_name = {slot.name: slot for slot in face.preset.slots}
    results = []
    for row in frame.sort_values("slot").itertuples(index=False):
        slot_name = str(row.slot)
        score = None if getattr(row, "pred_score", None) is None else float(row.pred_score)
        pred_label = int(row.pred_label)
        regions = ()
        slot_spec = slots_by_name.get(slot_name)
        if pred_label and slot_spec is not None:
            regions = defect_regions_from_artifact(
                _artifact_for_row(row, artifacts),
                face,
                slot_spec,
                map_threshold=map_threshold,
                min_area=min_area,
            )
        results.append(
            SlotResult(
                slot=slot_name,
                score=score,
                threshold=threshold,
                pred_label=pred_label,
                source_path=Path(row.source_path),
                defect_regions=regions,
            ),
        )
    return results


def stamp_ocr_to_dict(result: StampOcrResult | None) -> dict[str, Any] | None:
    """Convert OCR result to JSON-serializable data."""
    if result is None:
        return None
    return {
        "status": result.status,
        "part_number": result.part_number,
        "stamp_date": result.stamp_date,
        "raw_text": result.raw_text,
        "confidence": result.confidence,
        "roi_path": None if result.roi_path is None else str(result.roi_path),
        "enhanced_path": None if result.enhanced_path is None else str(result.enhanced_path),
        "message": result.message,
    }


def load_stamp_roi_config(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    """Load optional stamp ROI overrides from JSON."""
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    overrides: dict[tuple[str, str], dict[str, Any]] = {}
    if not isinstance(data, dict):
        return overrides
    for key, value in data.items():
        if isinstance(value, dict) and "." in key:
            part_key, face_key = key.split(".", maxsplit=1)
            overrides[(part_key, face_key)] = value
        elif isinstance(value, dict):
            for face_key, face_value in value.items():
                if isinstance(face_value, dict):
                    overrides[(str(key), str(face_key))] = face_value
    return overrides


def _stamp_roi_for_face(face: FaceConfig, overrides: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any] | None:
    """Return crop-relative stamp ROI config for one face."""
    config = STAMP_ROI_DEFAULTS.get((face.part_key, face.key))
    override = overrides.get((face.part_key, face.key))
    if config is None and override is None:
        return None
    merged = dict(config or {})
    merged.update(override or {})
    try:
        roi = {
            "x1": int(merged["x1"]),
            "y1": int(merged["y1"]),
            "x2": int(merged["x2"]),
            "y2": int(merged["y2"]),
            "rotate_180": bool(merged.get("rotate_180", False)),
            "mirror_horizontal": bool(merged.get("mirror_horizontal", merged.get("flip_horizontal", False))),
            "mirror_vertical": bool(merged.get("mirror_vertical", merged.get("flip_vertical", False))),
        }
    except (KeyError, TypeError, ValueError):
        return None
    if roi["x2"] <= roi["x1"] or roi["y2"] <= roi["y1"]:
        return None
    return roi


def _clamp_roi(roi: dict[str, Any], image: np.ndarray) -> tuple[int, int, int, int] | None:
    """Clamp an OCR ROI to the crop image bounds."""
    height, width = image.shape[:2]
    x1 = max(0, min(width, int(roi["x1"])))
    y1 = max(0, min(height, int(roi["y1"])))
    x2 = max(0, min(width, int(roi["x2"])))
    y2 = max(0, min(height, int(roi["y2"])))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _crop_stamp_roi(crop: np.ndarray, roi: dict[str, Any]) -> np.ndarray | None:
    """Crop and orient a U-stamp ROI from one slot crop."""
    clamped = _clamp_roi(roi, crop)
    if clamped is None:
        return None
    x1, y1, x2, y2 = clamped
    stamp = crop[y1:y2, x1:x2].copy()
    if roi.get("rotate_180"):
        stamp = cv2.rotate(stamp, cv2.ROTATE_180)
    if roi.get("mirror_horizontal"):
        stamp = cv2.flip(stamp, 1)
    if roi.get("mirror_vertical"):
        stamp = cv2.flip(stamp, 0)
    return stamp


def enhance_stamp_roi(stamp: np.ndarray) -> np.ndarray:
    """Enhance an embossed stamp ROI for OpenCV-only OCR."""
    gray = cv2.cvtColor(stamp, cv2.COLOR_BGR2GRAY) if stamp.ndim == 3 else stamp.copy()
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
    enhanced = clahe.apply(gray.astype(np.uint8))
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    enhanced = cv2.addWeighted(enhanced, 1.7, blurred, -0.7, 0)
    return cv2.resize(enhanced, (enhanced.shape[1] * 2, enhanced.shape[0] * 2), interpolation=cv2.INTER_CUBIC)


@lru_cache(maxsize=1)
def _digit_templates() -> dict[str, list[np.ndarray]]:
    """Build simple OpenCV-rendered digit templates."""
    templates: dict[str, list[np.ndarray]] = {str(value): [] for value in range(10)}
    for digit in templates:
        for scale, thickness in ((1.15, 2), (1.3, 2), (1.45, 3)):
            image = np.zeros((56, 42), dtype=np.uint8)
            (text_width, text_height), baseline = cv2.getTextSize(digit, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            x = max(0, (image.shape[1] - text_width) // 2)
            y = max(text_height + 2, (image.shape[0] + text_height - baseline) // 2)
            cv2.putText(image, digit, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, thickness, cv2.LINE_AA)
            templates[digit].append(image)
    return templates


def _binary_stamp_line(line: np.ndarray) -> np.ndarray:
    """Return a binary image emphasizing stamp digits."""
    line = cv2.normalize(line, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blackhat = cv2.morphologyEx(line, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 9)))
    tophat = cv2.morphologyEx(line, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 9)))
    contrast = cv2.addWeighted(blackhat, 1.0, tophat, 1.0, 0)
    if float(contrast.std()) < 8.0:
        contrast = cv2.absdiff(line, cv2.GaussianBlur(line, (0, 0), 5))
    _threshold, binary = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return binary


def _candidate_digit_boxes(binary: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return candidate digit boxes sorted left to right."""
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    image_area = binary.shape[0] * binary.shape[1]
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = width * height
        if area < max(18, image_area * 0.0008):
            continue
        if height < binary.shape[0] * 0.12 or width < 3:
            continue
        if width > binary.shape[1] * 0.25 or height > binary.shape[0] * 0.95:
            continue
        boxes.append((x, y, x + width, y + height))
    boxes.sort(key=lambda box: box[0])

    merged: list[tuple[int, int, int, int]] = []
    for box in boxes:
        if not merged or box[0] - merged[-1][2] > 4:
            merged.append(box)
            continue
        previous = merged[-1]
        merged[-1] = (previous[0], min(previous[1], box[1]), max(previous[2], box[2]), max(previous[3], box[3]))
    return merged


def _prepare_digit_patch(binary: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Normalize one candidate digit patch for template matching."""
    x1, y1, x2, y2 = box
    patch = binary[y1:y2, x1:x2]
    if patch.size == 0:
        return np.zeros((56, 42), dtype=np.uint8)
    padded = cv2.copyMakeBorder(patch, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(padded, (42, 56), interpolation=cv2.INTER_AREA)


def _match_digit(patch: np.ndarray) -> tuple[str, float]:
    """Match a normalized digit patch against generated templates."""
    best_digit = ""
    best_score = -1.0
    patch_float = patch.astype("float32")
    for digit, templates in _digit_templates().items():
        for template in templates:
            score = float(cv2.matchTemplate(patch_float, template.astype("float32"), cv2.TM_CCOEFF_NORMED)[0, 0])
            if score > best_score:
                best_digit = digit
                best_score = score
    return best_digit, max(0.0, best_score)


def _recognize_digit_line(line: np.ndarray) -> tuple[str, float]:
    """Recognize one numeric stamp line with simple template matching."""
    binary = _binary_stamp_line(line)
    boxes = _candidate_digit_boxes(binary)
    if not boxes:
        return "", 0.0
    digits = []
    scores = []
    for box in boxes:
        digit, score = _match_digit(_prepare_digit_patch(binary, box))
        if score < 0.08:
            continue
        digits.append(digit)
        scores.append(score)
    if not digits:
        return "", 0.0
    return "".join(digits), float(sum(scores) / len(scores))


def _split_stamp_lines(enhanced: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split an enhanced ROI into top and bottom text lines."""
    height = enhanced.shape[0]
    top = enhanced[int(height * 0.12) : int(height * 0.52), :]
    bottom = enhanced[int(height * 0.45) : int(height * 0.88), :]
    return top, bottom


def _valid_stamp_date(value: str) -> bool:
    """Return whether a recognized value is a plausible YYYYMMDD date."""
    if not re.fullmatch(r"\d{8}", value):
        return False
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError:
        return False
    return True


def parse_stamp_lines(top_line: str, bottom_line: str) -> tuple[str | None, str | None, str]:
    """Parse fixed two-line U-stamp text into part number and date."""
    first = re.sub(r"\D", "", top_line)
    second = re.sub(r"\D", "", bottom_line)
    if _valid_stamp_date(second):
        return first or None, second, "OK" if first else "FAIL"
    if _valid_stamp_date(first) and second:
        return second, first, "OK"
    return (first or None), (second if _valid_stamp_date(second) else None), "FAIL"


def _stamp_orientation_candidates(stamp: np.ndarray) -> list[np.ndarray]:
    """Return OCR candidates for mirrored or rotated stamp images."""
    candidates = [
        stamp,
        cv2.flip(stamp, 1),
        cv2.flip(stamp, 0),
        cv2.rotate(stamp, cv2.ROTATE_180),
    ]
    unique = []
    signatures = set()
    for candidate in candidates:
        signature = (candidate.shape, candidate.tobytes()[:128])
        if signature in signatures:
            continue
        signatures.add(signature)
        unique.append(candidate)
    return unique


def _recognize_oriented_stamp(stamp: np.ndarray) -> tuple[StampOcrResult, np.ndarray]:
    """Recognize one already-oriented stamp ROI and return enhanced image."""
    enhanced = enhance_stamp_roi(stamp)
    top, bottom = _split_stamp_lines(enhanced)
    top_text, top_confidence = _recognize_digit_line(top)
    bottom_text, bottom_confidence = _recognize_digit_line(bottom)
    part_number, stamp_date, status = parse_stamp_lines(top_text, bottom_text)
    confidence = float((top_confidence + bottom_confidence) / 2.0)
    if confidence < 0.16 or status != "OK":
        status = "FAIL"
    raw_text = "\n".join(text for text in (top_text, bottom_text) if text)
    return (
        StampOcrResult(
            status=status,
            part_number=part_number if status == "OK" else None,
            stamp_date=stamp_date if status == "OK" else None,
            raw_text=raw_text,
            confidence=confidence,
            message="" if status == "OK" else "钢印识别置信度低或格式不合法",
        ),
        enhanced,
    )


def recognize_stamp_roi(stamp: np.ndarray, roi_path: Path, enhanced_path: Path) -> StampOcrResult:
    """Recognize a two-line numeric U stamp from one cropped ROI."""
    best_result: StampOcrResult | None = None
    best_stamp = stamp
    best_enhanced: np.ndarray | None = None
    for candidate in _stamp_orientation_candidates(stamp):
        result, enhanced = _recognize_oriented_stamp(candidate)
        if best_result is None:
            best_result, best_stamp, best_enhanced = result, candidate, enhanced
            continue
        if result.status == "OK" and best_result.status != "OK":
            best_result, best_stamp, best_enhanced = result, candidate, enhanced
            continue
        if result.status == best_result.status and result.confidence > best_result.confidence:
            best_result, best_stamp, best_enhanced = result, candidate, enhanced

    if best_result is None or best_enhanced is None:
        best_result = StampOcrResult(status="FAIL", message="钢印识别失败")
        best_enhanced = enhance_stamp_roi(stamp)
    roi_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(roi_path), best_stamp)
    cv2.imwrite(str(enhanced_path), best_enhanced)
    return replace(
        best_result,
        roi_path=roi_path,
        enhanced_path=enhanced_path,
    )


def recognize_stamp_ocr(
    face: FaceConfig,
    crop_paths: list[Path],
    stamp_dir: Path,
    args: argparse.Namespace,
    timings: TimingRecorder | None = None,
) -> dict[str, StampOcrResult]:
    """Recognize supported U stamps for all slot crops in one face."""
    roi_config = _stamp_roi_for_face(face, load_stamp_roi_config(args.stamp_roi_config))
    if roi_config is None:
        return {}

    results: dict[str, StampOcrResult] = {}
    recorder = timings or TimingRecorder()
    with recorder.stage("stamp_ocr"):
        for crop_path in sorted(crop_paths):
            slot = slot_from_path(str(crop_path))
            crop = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
            if crop is None:
                results[slot] = StampOcrResult(status="FAIL", message=f"无法读取crop: {crop_path}")
                continue
            stamp = _crop_stamp_roi(crop, roi_config)
            if stamp is None:
                results[slot] = StampOcrResult(status="FAIL", message="钢印ROI超出crop范围")
                continue
            roi_path = stamp_dir / f"{face.key}_{slot}_stamp.png"
            enhanced_path = stamp_dir / f"{face.key}_{slot}_stamp_enhanced.png"
            results[slot] = recognize_stamp_roi(stamp, roi_path, enhanced_path)
    return results


def attach_stamp_ocr(slots: list[SlotResult], stamp_results: dict[str, StampOcrResult]) -> list[SlotResult]:
    """Attach OCR results to slot predictions."""
    if not stamp_results:
        return slots
    return [replace(slot, stamp_ocr=stamp_results.get(slot.slot)) for slot in slots]


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
        "reason": _quality_reasons_text(quality),
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


def _quality_reasons_text(quality: QualityReport | None) -> str:
    """Return compact quality issue text."""
    if quality is None or not quality.issues:
        return ""
    return "; ".join(quality.issues)


def _quality_summary_text(result: FaceResult | None) -> str:
    """Return quality status plus warning reasons for UI display."""
    if result is None or result.quality is None:
        return "质量 待检"
    reason = _quality_reasons_text(result.quality)
    if not reason:
        return f"质量 {result.quality.status}"
    return f"质量 {result.quality.status}: {reason}"


def _slot_result_to_dict(slot: SlotResult) -> dict[str, Any]:
    """Convert a slot result to JSON-serializable data."""
    return {
        "slot": slot.slot,
        "score": slot.score,
        "threshold": slot.threshold,
        "pred_label": slot.pred_label,
        "status": slot.status,
        "source_path": str(slot.source_path),
        "defect_regions": [defect_region_to_dict(region) for region in slot.defect_regions],
        "stamp_ocr": stamp_ocr_to_dict(slot.stamp_ocr),
    }


def _compact_timings(timings_ms: dict[str, float]) -> dict[str, float]:
    """Round timing values for logs and trace records."""
    return {name: round(value, 2) for name, value in sorted(timings_ms.items())}


def _box_to_dict(box: Any) -> dict[str, int]:
    """Return a serializable crop box with its size."""
    return {
        "x1": box.x1,
        "y1": box.y1,
        "x2": box.x2,
        "y2": box.y2,
        "width": box.width,
        "height": box.height,
    }


def crop_preset_layout(face: FaceConfig) -> dict[str, Any]:
    """Return the exact crop layout used by one face."""
    preset = face.preset
    return {
        "preset": face.preset_name,
        "roi": _box_to_dict(preset.roi),
        "slots": [
            {
                "name": slot.name,
                "row": slot.row,
                "col": slot.col,
                "box": _box_to_dict(slot.box),
            }
            for slot in preset.slots
        ],
    }


def _format_crop_layout(face: FaceConfig) -> str:
    """Return a compact crop-layout summary for startup logs."""
    layout = crop_preset_layout(face)
    roi = layout["roi"]
    slots = ", ".join(f"{slot['name']}={slot['box']['width']}x{slot['box']['height']}" for slot in layout["slots"])
    return f"{layout['preset']} roi={roi['width']}x{roi['height']} slots: {slots}"


def write_trace_record(
    result: FaceResult,
    face: FaceConfig,
    image_input: ImageInput,
    args: argparse.Namespace,
) -> Path:
    """Write a per-face JSON trace record for auditability."""
    trace_path = result.trace_path or result.image_path.parent / f"{face.key}_trace.json"
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
            "layout": crop_preset_layout(face),
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
        "defect_regions": [
            defect_region_to_dict(region)
            for slot in result.slots
            for region in slot.defect_regions
        ],
        "stamp_ocr": {
            slot.slot: stamp_ocr_to_dict(slot.stamp_ocr)
            for slot in result.slots
            if slot.stamp_ocr is not None
        },
        "archive_rows": {
            "slots": result.archive_slot_rows,
            "parts": result.archive_part_rows,
        },
        "artifacts": {
            "fused_image": str(result.image_path),
            "annotated_image": None if result.annotated_image_path is None else str(result.annotated_image_path),
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
    """Return profile-specific overrides, falling back to initial-profile side overrides."""
    profile_value = getattr(args, f"{part_key}_{face_key}_{option}", None)
    if profile_value is not None:
        return profile_value
    if part_key != args.part_profile:
        return None
    return getattr(args, f"{face_key}_{option}", None)


def resolve_face_configs(args: argparse.Namespace, part_key: str | None = None) -> dict[str, FaceConfig]:
    """Build face configs for a selectable part profile."""
    part_key = args.part_profile if part_key is None else part_key
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
    stamp_results = recognize_stamp_ocr(face, crop_paths, face_dir / "stamp_rois", args, timings)
    slots = attach_stamp_ocr(slots, stamp_results)
    annotated_image_path = save_annotated_image(image, face, slots, face_dir / f"{face.key}_annotated.png")
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
        annotated_image_path=annotated_image_path,
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
        "stamp_ocr",
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


def _contour_array(points: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Return an OpenCV contour array from plain points."""
    return np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)


def draw_defect_annotations(image: np.ndarray, face: FaceConfig, slots: list[SlotResult]) -> np.ndarray:
    """Draw slot boxes and predicted defect regions on a full fused image."""
    annotated = image.copy()
    slots_by_name = {slot.name: slot for slot in face.preset.slots}
    for result in slots:
        if not result.pred_label:
            continue
        slot = slots_by_name.get(result.slot)
        if slot is None:
            continue
        offset_x = face.preset.roi.x1 + slot.box.x1
        offset_y = face.preset.roi.y1 + slot.box.y1
        cv2.rectangle(
            annotated,
            (offset_x, offset_y),
            (offset_x + slot.box.width, offset_y + slot.box.height),
            (0, 0, 255),
            4,
        )
        if result.defect_regions:
            for region in result.defect_regions:
                cv2.polylines(
                    annotated,
                    [_contour_array(region.full_contour)],
                    isClosed=True,
                    color=(0, 0, 255),
                    thickness=5,
                )
                x1, y1, x2, y2 = region.full_bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            annotated,
            result.slot,
            (offset_x + 12, offset_y + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
    return annotated


def save_annotated_image(image: np.ndarray, face: FaceConfig, slots: list[SlotResult], path: Path) -> Path:
    """Save a full-image annotation for predicted defects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    annotated = draw_defect_annotations(image, face, slots)
    if not cv2.imwrite(str(path), annotated):
        msg = f"Could not write annotated image: {path}"
        raise RuntimeError(msg)
    return path


def _json_compact(value: Any) -> str:
    """Return compact JSON for CSV cells."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _archive_root(args: argparse.Namespace) -> Path:
    """Return the archive root for this run."""
    return args.archive_root if args.archive_root is not None else args.output_dir / "archive"


def _write_archive_rows(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    """Append archive rows, writing the header once."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _float_cell(value: float | None) -> str:
    """Return a stable CSV float cell."""
    return "" if value is None else f"{value:.6f}"


def _slot_archive_rows(
    result: FaceResult,
    face: FaceConfig,
    args: argparse.Namespace,
    *,
    trace_path: Path,
) -> list[dict[str, Any]]:
    """Build per-slot archive rows for one face result."""
    created_at = datetime.now().isoformat(timespec="seconds")
    inspection_id = args.output_dir.name
    quality_status = "" if result.quality is None else result.quality.status
    quality_reasons = _quality_reasons_text(result.quality)
    rows = []
    for slot in result.slots:
        ocr = slot.stamp_ocr
        rows.append(
            {
                "created_at": created_at,
                "inspection_id": inspection_id,
                "part": face.part_key,
                "part_title": face.part_title,
                "face": face.key,
                "face_title": face.title,
                "slot": slot.slot,
                "status": slot.status,
                "score": _float_cell(slot.score),
                "threshold": _float_cell(slot.threshold),
                "quality_status": quality_status,
                "quality_reasons": quality_reasons,
                "defect_regions_json": _json_compact([defect_region_to_dict(region) for region in slot.defect_regions]),
                "part_number": "" if ocr is None or ocr.part_number is None else ocr.part_number,
                "stamp_date": "" if ocr is None or ocr.stamp_date is None else ocr.stamp_date,
                "ocr_status": "" if ocr is None else ocr.status,
                "ocr_confidence": "" if ocr is None else _float_cell(ocr.confidence),
                "ocr_text": "" if ocr is None else ocr.raw_text.replace("\n", " / "),
                "fused_image": str(result.image_path),
                "annotated_image": "" if result.annotated_image_path is None else str(result.annotated_image_path),
                "crop_image": str(slot.source_path),
                "stamp_roi_image": "" if ocr is None or ocr.roi_path is None else str(ocr.roi_path),
                "stamp_enhanced_image": "" if ocr is None or ocr.enhanced_path is None else str(ocr.enhanced_path),
                "predictions_csv": str(result.predictions_csv),
                "trace_json": str(trace_path),
            },
        )
    return rows


def _slot_result_map(result: FaceResult | None) -> dict[str, SlotResult]:
    """Return slot results keyed by slot name."""
    if result is None:
        return {}
    return {slot.slot: slot for slot in result.slots}


def _preferred_stamp_result(slots: list[SlotResult]) -> StampOcrResult | None:
    """Return the best available OCR result for one physical slot."""
    for slot in slots:
        if slot.stamp_ocr is not None and slot.stamp_ocr.status == "OK":
            return slot.stamp_ocr
    return next((slot.stamp_ocr for slot in slots if slot.stamp_ocr is not None), None)


def _part_archive_rows(
    state: DemoState,
    face_configs: dict[str, FaceConfig],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Build per-physical-slot archive rows after both faces are inspected."""
    created_at = datetime.now().isoformat(timespec="seconds")
    inspection_id = args.output_dir.name
    top_result = state.results.get("top")
    bottom_result = state.results.get("bottom")
    top_slots = _slot_result_map(top_result)
    bottom_slots = _slot_result_map(bottom_result)
    top_quality_status = "" if top_result is None or top_result.quality is None else top_result.quality.status
    bottom_quality_status = (
        "" if bottom_result is None or bottom_result.quality is None else bottom_result.quality.status
    )
    top_quality_reasons = _quality_reasons_text(None if top_result is None else top_result.quality)
    bottom_quality_reasons = _quality_reasons_text(None if bottom_result is None else bottom_result.quality)
    slot_names = sorted(set(top_slots) | set(bottom_slots))
    rows = []
    for slot_name in slot_names:
        top_slot = top_slots.get(slot_name)
        bottom_slot = bottom_slots.get(slot_name)
        candidates = [slot for slot in (top_slot, bottom_slot) if slot is not None]
        status = "NG" if any(slot.pred_label for slot in candidates) else "OK"
        defect_faces = []
        if top_slot is not None and top_slot.pred_label:
            defect_faces.append(face_configs["top"].title)
        if bottom_slot is not None and bottom_slot.pred_label:
            defect_faces.append(face_configs["bottom"].title)
        stamp = _preferred_stamp_result(candidates)
        rows.append(
            {
                "created_at": created_at,
                "inspection_id": inspection_id,
                "part": state.part_key,
                "part_title": state.part_title,
                "slot": slot_name,
                "status": status,
                "defect_faces": ";".join(defect_faces),
                "part_number": "" if stamp is None or stamp.part_number is None else stamp.part_number,
                "stamp_date": "" if stamp is None or stamp.stamp_date is None else stamp.stamp_date,
                "ocr_status": "" if stamp is None else stamp.status,
                "top_score": "" if top_slot is None else _float_cell(top_slot.score),
                "bottom_score": "" if bottom_slot is None else _float_cell(bottom_slot.score),
                "top_status": "" if top_slot is None else top_slot.status,
                "bottom_status": "" if bottom_slot is None else bottom_slot.status,
                "top_quality_status": top_quality_status,
                "top_quality_reasons": top_quality_reasons,
                "bottom_quality_status": bottom_quality_status,
                "bottom_quality_reasons": bottom_quality_reasons,
                "top_crop_image": "" if top_slot is None else str(top_slot.source_path),
                "bottom_crop_image": "" if bottom_slot is None else str(bottom_slot.source_path),
            },
        )
    return rows


def archive_face_result(
    result: FaceResult,
    face: FaceConfig,
    args: argparse.Namespace,
    *,
    trace_path: Path,
) -> list[dict[str, Any]]:
    """Append per-slot archive rows for one face."""
    rows = _slot_archive_rows(result, face, args, trace_path=trace_path)
    _write_archive_rows(_archive_root(args) / "inspection_slots.csv", ARCHIVE_SLOT_FIELDS, rows)
    return rows


def archive_finished_part(
    state: DemoState,
    face_configs: dict[str, FaceConfig],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Append per-part archive rows after a two-face inspection finishes."""
    if not state.finished:
        return []
    rows = _part_archive_rows(state, face_configs, args)
    _write_archive_rows(_archive_root(args) / "inspection_parts.csv", ARCHIVE_PART_FIELDS, rows)
    return rows


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


@lru_cache(maxsize=64)
def _load_font(size: int, *, bold: bool = False) -> Any | None:
    """Load a font that can draw Chinese text when Pillow is available."""
    try:
        from PIL import ImageFont
    except Exception:
        return None

    font_paths = [
        (
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
            if bold
            else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        ),
        (
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc"
            if bold
            else "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"
        ),
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
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


def _text_bbox(text: str, *, size: int, bold: bool = False) -> tuple[int, int, int, int] | None:
    """Return rendered Pillow text bounds when available."""
    font = _load_font(size, bold=bold)
    if font is None:
        return None
    return font.getbbox(text)


def _text_size(text: str, *, size: int, bold: bool = False) -> tuple[int, int]:
    """Return rendered text size in pixels."""
    bbox = _text_bbox(text, size=size, bold=bold)
    if bbox is None:
        (width, height), _baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size / 32.0, 2)
        return width, height
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _fit_text_to_width(text: str, max_width: int, *, size: int, bold: bool = False) -> str:
    """Return text truncated to fit the available width."""
    if max_width <= 0:
        return ""
    if _text_size(text, size=size, bold=bold)[0] <= max_width:
        return text
    suffix = "..."
    suffix_width = _text_size(suffix, size=size, bold=bold)[0]
    if suffix_width > max_width:
        return ""

    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        candidate = text[:mid] + suffix
        if _text_size(candidate, size=size, bold=bold)[0] <= max_width:
            low = mid
        else:
            high = mid - 1
    return text[:low] + suffix


def _draw_text_fit(
    image: np.ndarray,
    text: str,
    rect: tuple[int, int, int, int],
    *,
    size: int,
    color: tuple[int, int, int],
    bold: bool = False,
    align: str = "left",
    valign: str = "top",
) -> None:
    """Draw text constrained to a rectangle."""
    x1, y1, x2, y2 = rect
    text = _fit_text_to_width(text, x2 - x1, size=size, bold=bold)
    if not text:
        return
    width, height = _text_size(text, size=size, bold=bold)
    bbox = _text_bbox(text, size=size, bold=bold)
    text_left = 0 if bbox is None else bbox[0]
    text_top = 0 if bbox is None else bbox[1]
    if align == "center":
        x = x1 + max(0, (x2 - x1 - width) // 2) - text_left
    elif align == "right":
        x = x2 - width - text_left
    else:
        x = x1 - text_left
    if valign == "center":
        y = y1 + max(0, (y2 - y1 - height) // 2) - text_top
    elif valign == "bottom":
        y = y2 - height - text_top
    else:
        y = y1 - text_top
    _draw_text(image, text, (x, y), size=size, color=color, bold=bold)


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
    _draw_text_fit(image, title, (x1 + 20, y1 + 14, x2 - 20, y1 + 48), size=24, color=(36, 44, 56), bold=True)
    if subtitle:
        _draw_text_fit(image, subtitle, (x1 + 190, y1 + 18, x2 - 20, y1 + 45), size=17, color=(112, 122, 138))


def _draw_badge(
    image: np.ndarray,
    text: str,
    rect: tuple[int, int, int, int],
    *,
    color: tuple[int, int, int],
    text_color: tuple[int, int, int] = (255, 255, 255),
    size: int = 22,
    align: str = "left",
) -> None:
    """Draw a compact status badge."""
    x1, y1, x2, y2 = rect
    cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
    _draw_text_fit(
        image,
        text,
        (x1 + 12, y1 + 2, x2 - 12, y2 - 2),
        size=size,
        color=text_color,
        bold=True,
        align=align,
        valign="center",
    )


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
        return f"总计{total / 1000.0:.2f}s"
    return f"总计{total / 1000.0:.2f}s / 推理{inference / 1000.0:.2f}s"


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


def _slot_ocr_text(result: FaceResult | None, slot: str) -> str:
    """Return compact OCR display text for one slot."""
    if result is None:
        return ""
    found = next((item for item in result.slots if item.slot == slot), None)
    if found is None or found.stamp_ocr is None:
        return ""
    ocr = found.stamp_ocr
    if ocr.status == "OK" and ocr.part_number and ocr.stamp_date:
        return f"{ocr.part_number} / {ocr.stamp_date}"
    if ocr.raw_text.strip():
        return "钢印? " + " / ".join(line for line in ocr.raw_text.splitlines() if line.strip())
    return "钢印--"


def _draw_face_result_panel(
    canvas: np.ndarray,
    face: FaceConfig,
    result: FaceResult | None,
    rect: tuple[int, int, int, int],
) -> None:
    """Draw one face result panel."""
    status = result.status if result is not None else "待检测"
    quality = _quality_status(result)
    _draw_panel(canvas, rect, f"{face.title}检测结果")
    x1, y1, x2, _y2 = rect
    status = result.status if result is not None else "待检测"
    color = _status_color(status)
    status_rect = (x2 - 112, y1 + 14, x2 - 22, y1 + 46)
    _draw_text_fit(
        canvas,
        _face_timing_text(result),
        (x1 + 188, y1 + 17, status_rect[0] - 14, y1 + 46),
        size=15,
        color=(112, 122, 138),
        align="right",
        valign="center",
    )
    _draw_badge(canvas, status, status_rect, color=color, size=21, align="center")
    quality_color = _status_color(quality)
    _draw_badge(canvas, f"质量 {quality}", (x1 + 20, y1 + 70, x1 + 130, y1 + 100), color=quality_color, size=16)
    defect_text = "缺陷 " + (", ".join(result.defect_slots) if result and result.defect_slots else "无")
    _draw_text_fit(
        canvas,
        defect_text,
        (x1 + 150, y1 + 70, x2 - 24, y1 + 100),
        size=18,
        color=(74, 85, 101),
        bold=bool(result and result.defect_slots),
        valign="center",
    )
    if result is not None and result.quality is not None and result.quality.issues:
        _draw_text_fit(
            canvas,
            _quality_summary_text(result),
            (x1 + 20, y1 + 100, x2 - 24, y1 + 114),
            size=12,
            color=(120, 83, 28),
            bold=True,
            valign="center",
        )

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
        _draw_text_fit(
            canvas,
            slot_status,
            (tile_x + 118, tile_y + 6, tile_x + 184, tile_y + 27),
            size=18,
            color=tile_color,
            bold=True,
            align="center",
        )
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
        ocr_text = _slot_ocr_text(result, slot)
        if ocr_text:
            _draw_text_fit(
                canvas,
                ocr_text,
                (tile_x + 74, tile_y + 25, tile_x + 184, tile_y + 42),
                size=11,
                color=(92, 104, 120),
                align="right",
            )


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


def _draw_defect_regions_in_display(
    canvas: np.ndarray,
    result: FaceResult | None,
    scale: float,
    offset_x: int,
    offset_y: int,
    origin: tuple[int, int],
) -> None:
    """Draw full-image defect regions scaled into the current display."""
    if result is None:
        return
    ox, oy = origin
    for slot in result.slots:
        for region in slot.defect_regions:
            points = np.asarray(
                [
                    [ox + offset_x + round(x * scale), oy + offset_y + round(y * scale)]
                    for x, y in region.full_contour
                ],
                dtype=np.int32,
            )
            if len(points) >= 2:
                cv2.polylines(canvas, [points.reshape(-1, 1, 2)], isClosed=True, color=(0, 0, 255), thickness=4)


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


def render_dashboard(
    state: DemoState,
    face_configs: dict[str, FaceConfig],
    width: int = 1600,
    height: int = 920,
) -> np.ndarray:
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
    if "top" not in state.results:
        top_stage = "待检"
    else:
        top_stage = "OK" if state.results["top"].status == "OK" else "NG"
    bottom_stage = (
        "OK"
        if "bottom" in state.results and state.results["bottom"].status == "OK"
        else "NG"
        if "bottom" in state.results
        else "待检"
    )
    _draw_badge(
        canvas,
        f"{top_title} {top_stage}",
        (1070, 56, 1302, 88),
        color=_status_color(top_stage),
        size=17,
        align="center",
    )
    _draw_badge(
        canvas,
        f"{bottom_title} {bottom_stage}",
        (1320, 56, 1552, 88),
        color=_status_color(bottom_stage),
        size=17,
        align="center",
    )

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
        display_y = slice(image_y + off_y, image_y + off_y + display.shape[0])
        display_x = slice(image_x + off_x, image_x + off_x + display.shape[1])
        canvas[display_y, display_x] = display
        active_result = state.results.get(active_face)
        for slot in active_config.preset.slots:
            status, _score = _slot_status(active_result, slot.name)
            color = _status_color(
                "OK" if status == "OK" else "NG" if status == "NG" else "",
            )
            thickness = 4 if status == "NG" else 2
            box = _slot_box_in_display(slot, active_config, scale, off_x, off_y, (image_x, image_y))
            cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), color, thickness)
            _draw_text(
                canvas,
                slot.name,
                (box[0] + 8, box[1] + 8),
                size=18,
                color=color,
                bold=True,
            )
        _draw_defect_regions_in_display(canvas, active_result, scale, off_x, off_y, (image_x, image_y))
        if active_result is not None:
            result_color = _status_color(active_result.status)
            _draw_badge(
                canvas,
                f"{active_title} {active_result.status}",
                (image_x + 18, image_y + 18, image_x + 170, image_y + 54),
                color=result_color,
                size=20,
            )
    else:
        cv2.circle(canvas, (image_x + 390, image_y + 252), 34, (210, 218, 228), -1)
        _draw_text(
            canvas,
            "等待上料并按 s",
            (image_x + 442, image_y + 226),
            size=31,
            color=(92, 104, 120),
            bold=True,
        )
        _draw_text(
            canvas,
            f"下一步: {active_title}",
            (image_x + 442, image_y + 268),
            size=22,
            color=(122, 134, 150),
        )

    _draw_face_result_panel(
        canvas,
        face_configs["top"],
        state.results.get("top"),
        (1100, 112, 1566, 412),
    )
    _draw_face_result_panel(canvas, face_configs["bottom"], state.results.get("bottom"), (1100, 432, 1566, 732))

    summary_rect = (34, 768, 1566, 892)
    _draw_panel(canvas, summary_rect, "整件汇总", subtitle="双面任一 NG 则整件 NG")
    status = part_status(state.results)
    color = _status_color(status)
    _draw_badge(canvas, status, (200, 834, 342, 884), color=color, size=30, align="center")
    defects = all_defect_positions(state.results)
    defect_text = "缺陷位置: " + (", ".join(defects) if defects else "无")
    _draw_text_fit(
        canvas,
        defect_text,
        (390, 828, 1088, 858),
        size=24,
        color=(40, 49, 63),
        bold=bool(defects),
        valign="center",
    )
    top_result = state.results.get("top")
    bottom_result = state.results.get("bottom")
    top_meta = _face_summary_meta(top_title, top_result)
    bottom_meta = _face_summary_meta(bottom_title, bottom_result)
    _draw_text_fit(canvas, top_meta, (390, 860, 760, 886), size=17, color=(92, 104, 120), valign="center")
    _draw_text_fit(canvas, bottom_meta, (790, 860, 1090, 886), size=17, color=(92, 104, 120), valign="center")
    _draw_text_fit(
        canvas,
        "1/2 切换零件   s 连续检测   r 重置   q 退出",
        (1120, 860, 1544, 886),
        size=18,
        color=(92, 104, 120),
        bold=True,
        valign="center",
    )
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
    quality_note = ""
    if result.quality is not None and result.quality.status == "WARN":
        quality_note = f"；{_quality_summary_text(result)}"
    if state.finished:
        first_title = face_configs[FACE_ORDER[0]].title
        return (
            f"{face.title}检测完成: {result.status}{quality_note}。"
            f"双面检测结束，再按 s 检测下一件{first_title}"
        )
    next_face_key = state.active_face
    next_title = face_configs[next_face_key].title if next_face_key else FACE_ORDER[-1]
    return f"{face.title}检测完成: {result.status}{quality_note}。请切换到{next_title}后按 s 检测"


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
    state: DemoState | None = None,
    face_configs: dict[str, FaceConfig] | None = None,
) -> None:
    """Attach timings, write trace, and print timing report."""
    result.timings_ms = dict(timings.stages_ms)
    result.trace_path = result.image_path.parent / f"{face.key}_trace.json"
    result.archive_slot_rows = archive_face_result(result, face, args, trace_path=result.trace_path)
    if state is not None and face_configs is not None and state.finished:
        result.archive_part_rows = archive_finished_part(state, face_configs, args)
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
        finalize_face_result(result, face, image_input, args, timings, state, face_configs)
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
                finalize_face_result(result, face, image_input, args, timings, state, face_configs)
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
    parser.add_argument(
        "--top-ckpt-path",
        type=Path,
        help="Optional explicit top checkpoint for the initial profile.",
    )
    parser.add_argument(
        "--bottom-ckpt-path",
        type=Path,
        help="Optional explicit bottom checkpoint for the initial profile.",
    )
    parser.add_argument("--top-threshold", type=float, help="Optional explicit top threshold for the initial profile.")
    parser.add_argument(
        "--bottom-threshold",
        type=float,
        help="Optional explicit bottom threshold for the initial profile.",
    )
    for part_key in PART_ORDER:
        for face_key in FACE_ORDER:
            parser.add_argument(
                f"--{part_key}-{face_key}-output-root",
                type=Path,
                help=f"Optional {part_key} {face_key} output-root override.",
            )
            parser.add_argument(
                f"--{part_key}-{face_key}-ckpt-path",
                type=Path,
                help=f"Optional explicit {part_key} {face_key} checkpoint.",
            )
            parser.add_argument(
                f"--{part_key}-{face_key}-threshold",
                type=float,
                help=f"Optional explicit {part_key} {face_key} threshold.",
            )
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

    parser.add_argument(
        "--demo-top-image",
        type=Path,
        help="Use a local full-size top image instead of camera capture.",
    )
    parser.add_argument(
        "--demo-bottom-image",
        type=Path,
        help="Use a local full-size bottom image instead of camera capture.",
    )
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
    parser.add_argument("--archive-root", type=Path, help="Archive directory for inspection CSVs and linked artifacts.")
    parser.add_argument(
        "--defect-map-threshold",
        type=float,
        default=DEFAULT_DEFECT_MAP_THRESHOLD,
        help="Normalized anomaly-map threshold used when pred_mask is unavailable.",
    )
    parser.add_argument(
        "--defect-min-area",
        type=float,
        default=DEFAULT_DEFECT_MIN_AREA,
        help="Minimum defect region area in crop pixels.",
    )
    parser.add_argument("--stamp-roi-config", type=Path, help="Optional JSON file overriding stamp OCR ROIs.")
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
        print(f"{face.title} crop: {_format_crop_layout(face)}")
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
        f"quality_gate={args.quality_gate}, archive={_archive_root(args)}, "
        f"defect_map_threshold={args.defect_map_threshold:g}, defect_min_area={args.defect_min_area:g}"
    )


def configure_runtime(args: argparse.Namespace) -> None:
    """Configure low-risk runtime settings for the live demo."""
    if args.predict_batch_size <= 0:
        msg = "--predict-batch-size must be positive."
        raise SystemExit(msg)
    if args.predict_num_workers < 0:
        msg = "--predict-num-workers must be non-negative."
        raise SystemExit(msg)
    if not 0.0 <= args.defect_map_threshold <= 1.0:
        msg = "--defect-map-threshold must be between 0 and 1."
        raise SystemExit(msg)
    if args.defect_min_area < 0:
        msg = "--defect-min-area must be non-negative."
        raise SystemExit(msg)
    if args.stamp_roi_config is not None and not args.stamp_roi_config.is_file():
        msg = f"--stamp-roi-config does not exist: {args.stamp_roi_config}"
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
