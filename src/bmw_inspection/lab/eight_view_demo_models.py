"""Resident model adapters for the BMW eight-view laboratory Demo."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import tempfile
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Any

import cv2
import numpy as np

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER
from bmw_inspection.lab.efficientad_analysis import fixed_scale_heatmap
from bmw_inspection.lab.eight_view_demo import (
    BranchStatus,
    DemoBranch,
    DemoBranchResult,
    EightViewInspection,
    EightViewDemoConfig,
    fuse_demo_status,
)
from bmw_inspection.lab.eight_view_roi import load_roi_config
from bmw_inspection.lab.trusted_ok_reference import (
    TrustedOkMatch,
    TrustedOkMatcher,
)


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """Small model-independent output consumed by the Demo runtime."""

    status: BranchStatus
    score: float | None
    threshold: float | None
    reason: str
    overlay: np.ndarray | None
    raw_pred_label: bool | None = None
    details: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not isinstance(self.status, BranchStatus):
            raise TypeError("status must be BranchStatus")
        for name in ("score", "threshold"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite or None")
        if not self.reason.strip():
            raise ValueError("reason must not be empty")
        if self.raw_pred_label is not None and not isinstance(self.raw_pred_label, bool):
            raise TypeError("raw_pred_label must be bool or None")
        if not isinstance(self.details, Mapping):
            raise TypeError("details must be a mapping")


ViewPredictor = Callable[[str, np.ndarray], ModelOutput]
BrightPredictor = Callable[[np.ndarray], ModelOutput]


@dataclass(frozen=True, slots=True)
class _TemplateModel:
    input_width: int
    input_height: int
    target_width: int
    target_height: int
    max_shift: int
    threshold: float
    templates: tuple[np.ndarray, ...]


def load_part_rois(path: Path) -> Mapping[str, tuple[int, int, int, int]]:
    """Load the manifest-bound eight-view ROI asset in canonical order."""
    config = load_roi_config(Path(path).expanduser().resolve())
    return MappingProxyType({view: config.part_rois[view] for view in VIEW_ORDER})


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError("Template输入必须是灰度、BGR或BGRA图像")


def _prepare_template_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    gray = _gray(image)
    target_width, target_height = size
    scale = min(target_width / gray.shape[1], target_height / gray.shape[0])
    width = max(1, min(target_width, int(round(gray.shape[1] * scale))))
    height = max(1, min(target_height, int(round(gray.shape[0] * scale))))
    resized = cv2.resize(
        gray,
        (width, height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    left = (target_width - width) // 2
    top = (target_height - height) // 2
    fitted = cv2.copyMakeBorder(
        resized,
        top,
        target_height - height - top,
        left,
        target_width - width - left,
        cv2.BORDER_REFLECT_101,
    )
    return cv2.GaussianBlur(fitted, (3, 3), 0)


class EightViewTemplatePredictor:
    """Load and score all eight independently trained Template models."""

    def __init__(
        self,
        model_paths: Mapping[str, Path],
        *,
        ignore_masks: Mapping[str, np.ndarray] | None = None,
        ignore_mask_index_sha256: str | None = None,
        masked_thresholds: Mapping[str, float] | None = None,
        masked_threshold_artifact_sha256: str | None = None,
    ) -> None:
        if tuple(model_paths) != VIEW_ORDER:
            raise ValueError("Template模型必须按标准顺序覆盖八个视角")
        self._models = MappingProxyType(
            {view: self._load(view, Path(model_paths[view]).expanduser().resolve()) for view in VIEW_ORDER}
        )
        optional_values = (
            ignore_mask_index_sha256,
            masked_thresholds,
            masked_threshold_artifact_sha256,
        )
        if ignore_masks is None:
            if any(value is not None for value in optional_values):
                raise ValueError("Template手动忽略区配置必须同时提供mask、SHA和独立阈值")
            self._ignore_masks: Mapping[str, np.ndarray] | None = None
            self._ignore_mask_index_sha256 = None
            self._masked_thresholds: Mapping[str, float] | None = None
            self._masked_threshold_artifact_sha256 = None
            return
        if tuple(ignore_masks) != VIEW_ORDER or masked_thresholds is None or tuple(masked_thresholds) != VIEW_ORDER:
            raise ValueError("Template手动忽略区和独立阈值必须按标准顺序覆盖八个视角")
        for label, value in (
            ("ignore mask index", ignore_mask_index_sha256),
            ("masked threshold artifact", masked_threshold_artifact_sha256),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"Template {label} SHA256格式不正确")
        owned_masks: dict[str, np.ndarray] = {}
        parsed_thresholds: dict[str, float] = {}
        for view in VIEW_ORDER:
            mask = np.asarray(ignore_masks[view])
            model = self._models[view]
            expected_shape = (model.input_height, model.input_width)
            if mask.dtype != np.uint8 or mask.ndim != 2 or mask.shape != expected_shape:
                raise ValueError(f"{view} Template ignore mask必须是{expected_shape}的uint8二维图")
            if not set(np.unique(mask).tolist()).issubset({0, 255}):
                raise ValueError(f"{view} Template ignore mask只能包含0和255")
            owned = mask.copy()
            owned.flags.writeable = False
            owned_masks[view] = owned
            threshold = masked_thresholds[view]
            if isinstance(threshold, bool) or not isinstance(threshold, Real) or not math.isfinite(float(threshold)):
                raise ValueError(f"{view} Template masked threshold必须是有限数值")
            parsed_thresholds[view] = float(threshold)
        self._ignore_masks = MappingProxyType(owned_masks)
        self._ignore_mask_index_sha256 = ignore_mask_index_sha256
        self._masked_thresholds = MappingProxyType(parsed_thresholds)
        self._masked_threshold_artifact_sha256 = masked_threshold_artifact_sha256

    @staticmethod
    def _load(view: str, path: Path) -> _TemplateModel:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("view_id") != view:
            raise ValueError(f"Template模型视角不匹配：{path}")
        preprocess = payload.get("preprocess")
        items = payload.get("templates")
        if not isinstance(preprocess, dict) or not isinstance(items, list) or not items:
            raise ValueError(f"Template模型结构不正确：{path}")
        target_width = int(preprocess["target_width"])
        target_height = int(preprocess["target_height"])
        templates: list[np.ndarray] = []
        for item in items:
            template_path = (path.parent / item["path"]).resolve()
            image = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (target_height, target_width):
                raise ValueError(f"Template图片不可用：{template_path}")
            templates.append(image)
        return _TemplateModel(
            input_width=int(payload["input_width"]),
            input_height=int(payload["input_height"]),
            target_width=target_width,
            target_height=target_height,
            max_shift=int(preprocess["max_shift"]),
            threshold=float(payload["threshold"]),
            templates=tuple(templates),
        )

    def predict(self, view: str, image: np.ndarray) -> ModelOutput:
        """Return Template risk and a display-only difference heatmap."""
        model = self._models[view]
        if image.shape[:2] != (model.input_height, model.input_width):
            raise ValueError(
                f"{view} Template要求{model.input_width}x{model.input_height}，实际{image.shape[1]}x{image.shape[0]}"
            )
        query = _prepare_template_image(image, (model.target_width, model.target_height))
        padded = cv2.copyMakeBorder(
            query,
            model.max_shift,
            model.max_shift,
            model.max_shift,
            model.max_shift,
            cv2.BORDER_REFLECT_101,
        )
        matches: list[tuple[float, np.ndarray, tuple[int, int]]] = []
        for template in model.templates:
            response = cv2.matchTemplate(padded, template, cv2.TM_CCOEFF_NORMED)
            _minimum, maximum, _minimum_location, maximum_location = cv2.minMaxLoc(response)
            matches.append((float(maximum), template, maximum_location))
        raw_similarity, raw_best, raw_best_location = max(matches, key=lambda item: item[0])
        raw_risk = max(0.0, 1.0 - raw_similarity)
        similarity, best, best_location = raw_similarity, raw_best, raw_best_location
        risk = raw_risk
        threshold = model.threshold
        score_source = "unmasked_ccoeff_normed"
        aligned_inspect_mask: np.ndarray | None = None
        if self._ignore_masks is not None and np.any(self._ignore_masks[view]):
            from bmw_inspection.lab.template_ignore_mask import (
                prepare_template_inspect_mask,
                select_masked_template_match,
            )

            inspect_mask = prepare_template_inspect_mask(
                self._ignore_masks[view],
                (model.target_width, model.target_height),
            )
            padded_inspect_mask = cv2.copyMakeBorder(
                inspect_mask,
                model.max_shift,
                model.max_shift,
                model.max_shift,
                model.max_shift,
                cv2.BORDER_REFLECT_101,
            )
            minimum_valid_pixels = max(
                256,
                math.ceil(0.10 * model.target_width * model.target_height),
            )
            masked_match = select_masked_template_match(
                padded,
                model.templates,
                padded_inspect_mask,
                minimum_valid_pixels=minimum_valid_pixels,
            )
            similarity = masked_match.similarity
            best = masked_match.template
            best_location = masked_match.location
            aligned_inspect_mask = masked_match.aligned_inspect_mask
            risk = max(0.0, 1.0 - similarity)
            assert self._masked_thresholds is not None
            threshold = self._masked_thresholds[view]
            score_source = "manual_ignore_masked_ccoeff_normed"
        best_x, best_y = best_location
        aligned_query = padded[
            best_y : best_y + model.target_height,
            best_x : best_x + model.target_width,
        ]
        difference = cv2.absdiff(aligned_query, best)
        if aligned_inspect_mask is None:
            mean_absolute_difference = float(difference.mean())
            valid_target_pixel_count = int(difference.size)
        else:
            valid = aligned_inspect_mask.astype(bool)
            if not np.any(valid):
                raise ValueError(f"{view} Template ignore mask排除了全部对齐像素")
            mean_absolute_difference = float(difference[valid].mean())
            valid_target_pixel_count = int(np.count_nonzero(valid))
            difference = difference.copy()
            difference[~valid] = 0
        heatmap = cv2.applyColorMap(difference, cv2.COLORMAP_TURBO)
        base = cv2.cvtColor(aligned_query, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(base, 0.65, heatmap, 0.35, 0.0)
        if aligned_inspect_mask is not None:
            overlay[~aligned_inspect_mask.astype(bool)] = base[~aligned_inspect_mask.astype(bool)]
        passed = risk <= threshold
        masked = score_source == "manual_ignore_masked_ccoeff_normed"
        reason_prefix = "手动忽略区外 Template 诊断热区" if masked else "Template 诊断热区"
        return ModelOutput(
            BranchStatus.PASS if passed else BranchStatus.NG,
            risk,
            threshold,
            (
                f"{reason_prefix}：模板风险 {risk:.4f}，部署阈值 {threshold:.4f}；"
                f"最佳平移 ({best_x - model.max_shift}, {best_y - model.max_shift})，"
                f"对齐后平均绝对差 {mean_absolute_difference:.3f}"
            ),
            overlay,
            details={
                "evidence_type": "诊断热区",
                "score_source": score_source,
                "similarity": similarity,
                "risk": risk,
                "deployment_threshold": threshold,
                "threshold_exceedance": risk - threshold,
                "best_shift_x": best_x - model.max_shift,
                "best_shift_y": best_y - model.max_shift,
                "aligned_mean_absolute_difference": mean_absolute_difference,
                "raw_unmasked_similarity": raw_similarity,
                "raw_unmasked_risk": raw_risk,
                "raw_unmasked_best_shift_x": raw_best_location[0] - model.max_shift,
                "raw_unmasked_best_shift_y": raw_best_location[1] - model.max_shift,
                "masked_similarity": similarity if masked else None,
                "masked_risk": risk if masked else None,
                "ignored_input_pixel_count": (
                    int(np.count_nonzero(self._ignore_masks[view])) if masked and self._ignore_masks is not None else 0
                ),
                "valid_target_pixel_count": valid_target_pixel_count,
                "valid_target_pixel_fraction": valid_target_pixel_count / difference.size,
                "ignore_mask_index_sha256": self._ignore_mask_index_sha256 if masked else None,
                "masked_threshold_artifact_sha256": (
                    self._masked_threshold_artifact_sha256 if masked else None
                ),
            },
        )


class EightViewBrightStreakPredictor:
    """Use the calibrated full-image `front_left` rule and render its evidence."""

    def __init__(self, config_path: Path) -> None:
        from bmw_inspection.contracts import load_config

        self._config = load_config(Path(config_path).expanduser().resolve())

    def predict(self, image: np.ndarray) -> ModelOutput:
        from bmw_inspection.contracts import DemoStatus
        from bmw_inspection.detector import detect_bright_streak_evidence, render_evidence

        decision = detect_bright_streak_evidence(image, self._config)
        metrics = decision.metrics
        if decision.status is DemoStatus.ERROR:
            status = BranchStatus.ERROR
        elif decision.status is DemoStatus.OK:
            status = BranchStatus.PASS
        else:
            status = BranchStatus.NG
        return ModelOutput(
            status,
            None if metrics is None else metrics.coverage_ratio,
            self._config.min_coverage_ratio,
            {
                DemoStatus.OK: "亮痕存在且连续",
                DemoStatus.NG_NO_STREAK: "未检测到有效亮痕",
                DemoStatus.NG_BROKEN: "亮痕存在但不连续",
                DemoStatus.ERROR: "光痕检测输入异常",
            }[decision.status],
            render_evidence(image, decision),
            details={
                "evidence_type": "规则 ROI 证据",
                "coverage_ratio": None if metrics is None else metrics.coverage_ratio,
                "deployment_threshold": self._config.min_coverage_ratio,
            },
        )


class EightViewRawProfileBrightStreakPredictor:
    """Apply the calibrated raw-grayscale row profile to one fixed full-image ROI."""

    def __init__(self, report_path: Path) -> None:
        from bmw_inspection.lab.bright_streak_raw_profile import RawProfileThresholds

        path = Path(report_path).expanduser().resolve()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            roi = payload["roi_xyxy"]
            values = payload["raw_profile_v2"]["thresholds"]
            thresholds = RawProfileThresholds(
                min_row_score=float(values["min_row_score"]),
                min_presence_coverage_ratio=float(values["min_presence_coverage_ratio"]),
                min_longest_run_ratio=float(values["min_longest_run_ratio"]),
                max_gap_ratio=float(values["max_gap_ratio"]),
                max_gap_count=int(values["max_gap_count"]),
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"原灰度光痕v2配置不可用：{path}: {error}") from error
        if (
            not isinstance(roi, list)
            or len(roi) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) for value in roi)
        ):
            raise ValueError("原灰度光痕v2 ROI必须包含四个整数")
        x1, y1, x2, y2 = roi
        if not (0 <= x1 < x2 and 0 <= y1 < y2) or (y2 - y1, x2 - x1) != (613, 81):
            raise ValueError("原灰度光痕v2 ROI必须是81x613")
        self._roi = (x1, y1, x2, y2)
        self._thresholds = thresholds

    def predict(self, image: np.ndarray) -> ModelOutput:
        from bmw_inspection.lab.bright_streak_raw_profile import analyze_raw_profile, classify_raw_profile

        x1, y1, x2, y2 = self._roi
        height, width = image.shape[:2]
        if x2 > width or y2 > height:
            raise ValueError(f"原灰度光痕v2 ROI超出输入图像{width}x{height}")
        gray = _gray(image)
        metrics = analyze_raw_profile(
            gray[y1:y2, x1:x2],
            min_row_score=self._thresholds.min_row_score,
        )
        decision = classify_raw_profile(metrics, self._thresholds)
        status = BranchStatus.PASS if decision == "OK" else BranchStatus.NG
        roi_gray = gray[y1:y2, x1:x2]
        overlay = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
        color = (0, 200, 0) if status is BranchStatus.PASS else (0, 0, 255)
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1] - 1, overlay.shape[0] - 1), color, 2)
        centre_x = overlay.shape[1] // 2
        active_rows = np.flatnonzero(metrics.mask)
        overlay[active_rows, max(0, centre_x - 2) : min(overlay.shape[1], centre_x + 3)] = color
        if decision == "OK":
            label = "原灰度光痕存在且连续"
        elif decision == "NG_NO_STREAK":
            label = "原灰度未检测到有效光痕"
        else:
            label = "原灰度光痕存在但不连续"
        reason = (
            f"{label}；覆盖率 {metrics.coverage_ratio:.3f}，"
            f"覆盖率阈值 {self._thresholds.min_presence_coverage_ratio:.3f}；"
            f"最长连续段 {metrics.longest_run_ratio:.3f}，"
            f"最长连续段阈值 {self._thresholds.min_longest_run_ratio:.3f}；"
            f"最大断点 {metrics.max_gap_ratio:.3f}（上限 {self._thresholds.max_gap_ratio:.3f}），"
            f"断点数 {metrics.gap_count}（上限 {self._thresholds.max_gap_count}）"
        )
        return ModelOutput(
            status,
            metrics.coverage_ratio,
            self._thresholds.min_presence_coverage_ratio,
            reason,
            overlay,
            details={
                "evidence_type": "规则 ROI 证据",
                "decision": decision,
                "roi_xyxy": self._roi,
                "row_count": int(len(metrics.mask)),
                "coverage_ratio": metrics.coverage_ratio,
                "longest_run_ratio": metrics.longest_run_ratio,
                "max_gap_ratio": metrics.max_gap_ratio,
                "gap_count": metrics.gap_count,
                "min_row_score": self._thresholds.min_row_score,
                "min_presence_coverage_ratio": self._thresholds.min_presence_coverage_ratio,
                "min_longest_run_ratio": self._thresholds.min_longest_run_ratio,
                "allowed_max_gap_ratio": self._thresholds.max_gap_ratio,
                "allowed_max_gap_count": self._thresholds.max_gap_count,
            },
        )


class EightViewTrackedProfileBrightStreakPredictor:
    """Apply one immutable tracked-profile v3 report to its fixed full-image ROI."""

    _FIXED_ROI = (1792, 1180, 1873, 1793)
    _CONFIRMED_LIVE_CAPTURE_ID = "bmw_demo_20260812_211302"
    _CONFIRMED_LIVE_FILES = {
        "front_left_hdr.png",
        "front_left_long.png",
        "front_left_short.png",
        "inspection.json",
    }

    _REPORT_FIELDS = {
        "schema_version",
        "status",
        "algorithm",
        "fit_split",
        "final_test_used_for_fit",
        "real_broken_samples",
        "manifest",
        "roi_xyxy",
        "geometry",
        "geometry_selection",
        "thresholds",
        "calibration_counts",
        "accepted_live_normals",
        "acceptance_gate",
        "comparison_to_v2",
        "final_test",
        "replay",
        "artifact_identities",
        "cpu_per_image_ms",
        "metrics_csv",
        "profiles_dir",
        "replay_summary_json",
        "report_json",
        "identities",
    }
    _GEOMETRY_FIELDS = {
        "candidate_width",
        "background_width",
        "background_gap",
        "smooth_window",
        "max_step",
        "step_penalty",
    }
    _THRESHOLD_FIELDS = {
        "strong_row_score",
        "weak_row_score",
        "min_presence_coverage_ratio",
        "min_longest_run_ratio",
        "max_gap_ratio",
        "max_gap_count",
    }
    _IDENTITY_FIELDS = {
        "algorithm_source_sha256",
        "evaluator_source_sha256",
        "manifest_sha256",
        "roi_config_sha256",
    }
    _ARTIFACT_FIELDS = {
        "metrics_csv_sha256",
        "replay_summary_sha256",
        "profile_npz_sha256",
    }

    def __init__(
        self,
        report_path: Path,
        *,
        rotated_roi_path: Path | None = None,
        rotated_roi_sha256: str | None = None,
        weak_row_score_override: float | None = None,
    ) -> None:
        from bmw_inspection.lab.bright_streak_tracked_profile import (
            TrackedProfileGeometry,
            TrackedProfileThresholds,
        )

        path = Path(report_path).expanduser().resolve()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"追踪光痕v3报告不可用：{path}: {error}") from error
        if not isinstance(payload, dict) or set(payload) != self._REPORT_FIELDS:
            raise ValueError("追踪光痕v3 report schema不正确")
        if (
            payload["schema_version"] != 1
            or payload["status"] != "complete"
            or payload["algorithm"] != "tracked_profile_v3"
            or payload["fit_split"] != "calibration"
            or payload["final_test_used_for_fit"] is not False
            or payload["real_broken_samples"] != 0
            or not isinstance(payload["acceptance_gate"], dict)
            or payload["acceptance_gate"].get("passed") is not True
        ):
            raise ValueError("追踪光痕v3 report identity不正确")
        roi = payload["roi_xyxy"]
        if (
            not isinstance(roi, list)
            or len(roi) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) for value in roi)
        ):
            raise ValueError("追踪光痕v3 ROI必须包含四个整数")
        x1, y1, x2, y2 = roi
        if (x1, y1, x2, y2) != self._FIXED_ROI:
            raise ValueError("追踪光痕v3必须使用固定ROI [1792,1180,1873,1793]")
        geometry_values = payload["geometry"]
        threshold_values = payload["thresholds"]
        if not isinstance(geometry_values, dict) or set(geometry_values) != self._GEOMETRY_FIELDS:
            raise ValueError("追踪光痕v3 geometry schema不正确")
        if not isinstance(threshold_values, dict) or set(threshold_values) != self._THRESHOLD_FIELDS:
            raise ValueError("追踪光痕v3 thresholds schema不正确")
        try:
            geometry = TrackedProfileGeometry(**geometry_values)
            thresholds = TrackedProfileThresholds(**threshold_values)
        except (TypeError, ValueError) as error:
            raise ValueError(f"追踪光痕v3参数不正确：{error}") from error
        required_width = geometry.candidate_width + 2 * (
            geometry.background_gap + geometry.background_width
        )
        if required_width > self._FIXED_ROI[2] - self._FIXED_ROI[0]:
            raise ValueError("追踪光痕v3 geometry超出固定ROI宽度")
        identities = payload["identities"]
        if (
            not isinstance(identities, dict)
            or set(identities) != self._IDENTITY_FIELDS
            or any(not self._is_sha256(value) for value in identities.values())
        ):
            raise ValueError("追踪光痕v3 source identity不正确")
        repository_root = Path(__file__).resolve().parents[3]
        bound_sources = {
            "algorithm_source_sha256": repository_root
            / "src/bmw_inspection/lab/bright_streak_tracked_profile.py",
            "evaluator_source_sha256": repository_root
            / "pipeline/bmw_lab_evaluate_bright_streak_tracked_profile.py",
            "manifest_sha256": Path(payload["manifest"]).expanduser().resolve(),
        }
        if any(
            not source.is_file() or _file_sha256(source) != identities[field]
            for field, source in bound_sources.items()
        ):
            raise ValueError("追踪光痕v3 source identity SHA256不匹配")
        roi_identity = hashlib.sha256(
            json.dumps(
                {"roi_xyxy": roi},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if roi_identity != identities["roi_config_sha256"]:
            raise ValueError("追踪光痕v3 source identity SHA256不匹配")
        self._validate_confirmed_live_normal(payload["accepted_live_normals"])
        self._validate_acceptance_evidence(payload)
        self._validate_artifact_inventory(path.parent, payload["artifact_identities"])
        self._validate_semantic_inventory(path.parent, payload)
        self._roi = (x1, y1, x2, y2)
        self._geometry = geometry
        self._report_weak_row_score = thresholds.weak_row_score
        if weak_row_score_override is not None:
            if rotated_roi_path is None:
                raise ValueError("弱响应覆盖阈值只允许用于手动倾斜光痕ROI")
            if (
                isinstance(weak_row_score_override, bool)
                or not isinstance(weak_row_score_override, Real)
                or not math.isfinite(float(weak_row_score_override))
                or float(weak_row_score_override) < 0
                or float(weak_row_score_override) > thresholds.strong_row_score
            ):
                raise ValueError("弱响应覆盖阈值必须是有限非负数且不高于强响应阈值")
            threshold_values = dict(threshold_values)
            threshold_values["weak_row_score"] = float(weak_row_score_override)
            thresholds = TrackedProfileThresholds(**threshold_values)
        self._thresholds = thresholds
        self._weak_row_score_overridden = weak_row_score_override is not None
        if (rotated_roi_path is None) != (rotated_roi_sha256 is None):
            raise ValueError("倾斜光痕ROI路径和SHA256必须同时提供")
        self._rotated_roi = None
        self._rotated_roi_sha256 = None
        if rotated_roi_path is not None and rotated_roi_sha256 is not None:
            from bmw_inspection.lab.bright_streak_rotated_roi import (
                load_rotated_bright_streak_roi,
            )

            try:
                self._rotated_roi = load_rotated_bright_streak_roi(
                    rotated_roi_path,
                    expected_sha256=rotated_roi_sha256,
                )
            except ValueError as error:
                raise ValueError(f"倾斜光痕ROI SHA256或资产校验失败：{error}") from error
            self._rotated_roi_sha256 = rotated_roi_sha256

    @staticmethod
    def _is_sha256(value: object) -> bool:
        return isinstance(value, str) and len(value) == 64 and all(
            character in "0123456789abcdef" for character in value
        )

    @classmethod
    def _validate_confirmed_live_normal(cls, records: object) -> None:
        expected_fields = {
            "capture_id",
            "file_sha256",
            "predicted_status",
            "provenance_kind",
            "record_path",
        }
        if not isinstance(records, list) or len(records) != 1:
            raise ValueError("追踪光痕v3缺少唯一确认的现场正常样本")
        record = records[0]
        if (
            not isinstance(record, dict)
            or set(record) != expected_fields
            or record["capture_id"] != cls._CONFIRMED_LIVE_CAPTURE_ID
            or record["predicted_status"] != "OK"
            or record["provenance_kind"] != "user_confirmed_live_normal"
            or not isinstance(record["record_path"], str)
        ):
            raise ValueError("追踪光痕v3现场正常样本身份不正确")
        file_sha256 = record["file_sha256"]
        if (
            not isinstance(file_sha256, dict)
            or set(file_sha256) != cls._CONFIRMED_LIVE_FILES
            or any(not cls._is_sha256(digest) for digest in file_sha256.values())
        ):
            raise ValueError("追踪光痕v3现场正常样本文件身份不正确")
        root = Path(record["record_path"]).expanduser().resolve()
        files = {
            "front_left_hdr.png": root / "images/front_left_hdr.png",
            "front_left_long.png": root / "images/front_left_long.png",
            "front_left_short.png": root / "images/front_left_short.png",
            "inspection.json": root / "inspection.json",
        }
        if any(
            not source.is_file() or _file_sha256(source) != file_sha256[name]
            for name, source in files.items()
        ):
            raise ValueError("追踪光痕v3现场正常样本文件SHA256不匹配")

    @staticmethod
    def _validate_acceptance_evidence(payload: Mapping[str, object]) -> None:
        gate = payload["acceptance_gate"]
        calibration = payload["calibration_counts"]
        final_test = payload["final_test"]
        comparison = payload["comparison_to_v2"]
        required_gate = {
            "confirmed_live_normal_count": 1,
            "no_streak_count": 8,
            "normal_not_worse_than_v2_splits": ["calibration", "final_test"],
            "passed": True,
        }
        required_calibration = {
            "accepted_live_normal": 1,
            "fit_total": 22,
            "manifest_no_streak": 4,
            "manifest_normal": 17,
        }
        if gate != required_gate or calibration != required_calibration:
            raise ValueError("追踪光痕v3验收证据不完整")
        if (
            not isinstance(final_test, dict)
            or final_test.get("count") != 20
            or final_test.get("normal_count") != 16
            or final_test.get("no_streak_count") != 4
            or final_test.get("no_streak_false_accepts") != 0
            or final_test.get("normal_false_rejects") != 1
            or not isinstance(comparison, dict)
        ):
            raise ValueError("追踪光痕v3验收证据不完整")
        for split in ("calibration", "final_test"):
            split_evidence = comparison.get(split)
            if (
                not isinstance(split_evidence, dict)
                or split_evidence.get("v3_normal_false_rejects")
                > split_evidence.get("v2_normal_false_rejects", -1)
            ):
                raise ValueError("追踪光痕v3验收证据不完整")

    @classmethod
    def _validate_artifact_inventory(cls, root: Path, inventory: object) -> None:
        if not isinstance(inventory, dict) or set(inventory) != cls._ARTIFACT_FIELDS:
            raise ValueError("追踪光痕v3 artifact inventory不正确")
        profiles = inventory["profile_npz_sha256"]
        if (
            not cls._is_sha256(inventory["metrics_csv_sha256"])
            or not cls._is_sha256(inventory["replay_summary_sha256"])
            or not isinstance(profiles, dict)
            or not profiles
            or any(
                not isinstance(name, str)
                or Path(name).name != name
                or not name.endswith(".npz")
                or not cls._is_sha256(digest)
                for name, digest in profiles.items()
            )
        ):
            raise ValueError("追踪光痕v3 artifact inventory不正确")
        expected = {
            root / "metrics.csv": inventory["metrics_csv_sha256"],
            root / "replay_summary.json": inventory["replay_summary_sha256"],
            **{root / "profiles" / name: digest for name, digest in profiles.items()},
        }
        if any(not file.is_file() or _file_sha256(file) != digest for file, digest in expected.items()):
            raise ValueError("追踪光痕v3 artifact inventory SHA256不匹配")
        actual_profiles = {path.name for path in (root / "profiles").glob("*.npz")}
        if actual_profiles != set(profiles):
            raise ValueError("追踪光痕v3 artifact inventory不完整")

    @classmethod
    def _validate_semantic_inventory(cls, root: Path, payload: Mapping[str, object]) -> None:
        metrics_path = root / "metrics.csv"
        try:
            with metrics_path.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                required = {
                    "sample_id",
                    "split",
                    "truth",
                    "provenance_kind",
                    "profile_npz",
                    "tracked_profile_v3_predicted_status",
                }
                if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                    raise ValueError("追踪光痕v3 metrics schema不正确")
                rows = list(reader)
        except OSError as error:
            raise ValueError(f"追踪光痕v3 metrics不可用：{error}") from error
        inventory = payload["artifact_identities"]["profile_npz_sha256"]
        profile_names = [Path(row["profile_npz"]).name for row in rows]
        if (
            len(rows) != 61
            or len(profile_names) != len(set(profile_names))
            or set(profile_names) != set(inventory)
        ):
            raise ValueError("追踪光痕v3 metrics与profile inventory不一致")
        expected_counts = {
            "split": {"calibration": 22, "final_test": 20, "live_replay": 19},
            "truth": {"normal": 34, "no_streak": 8, "unknown": 19},
            "provenance_kind": {
                "manifest": 41,
                "user_confirmed_live_normal": 1,
                "unconfirmed_live_replay": 19,
            },
        }
        for field, expected in expected_counts.items():
            actual = {value: sum(row[field] == value for row in rows) for value in expected}
            if actual != expected or sum(actual.values()) != len(rows):
                raise ValueError("追踪光痕v3 metrics验收计数不一致")
        no_streak = [row for row in rows if row["truth"] == "no_streak"]
        accepted = [row for row in rows if row["provenance_kind"] == "user_confirmed_live_normal"]
        if (
            any(row["tracked_profile_v3_predicted_status"] != "NG_NO_STREAK" for row in no_streak)
            or len(accepted) != 1
            or accepted[0]["sample_id"] != cls._CONFIRMED_LIVE_CAPTURE_ID
            or accepted[0]["tracked_profile_v3_predicted_status"] != "OK"
        ):
            raise ValueError("追踪光痕v3逐样本验收结果不一致")
        manifest_metrics = {
            row["sample_id"]: row for row in rows if row["provenance_kind"] == "manifest"
        }
        try:
            with Path(payload["manifest"]).expanduser().resolve().open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                reader = csv.DictReader(stream)
                required = {"sample_id", "source_class", "split", "expected_status"}
                if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                    raise ValueError("追踪光痕v3 manifest schema不正确")
                manifest_rows = [
                    row
                    for row in reader
                    if row["source_class"] in {"normal", "no_streak"}
                    and row["split"] in {"calibration", "final_test"}
                ]
        except OSError as error:
            raise ValueError(f"追踪光痕v3 manifest不可用：{error}") from error
        manifest_ids = [row["sample_id"] for row in manifest_rows]
        if len(manifest_ids) != 41 or len(set(manifest_ids)) != 41 or set(manifest_ids) != set(manifest_metrics):
            raise ValueError("追踪光痕v3 manifest逐样本身份不一致")
        for manifest_row in manifest_rows:
            metric = manifest_metrics[manifest_row["sample_id"]]
            expected_truth = manifest_row["source_class"]
            if (
                metric["split"] != manifest_row["split"]
                or metric["truth"] != expected_truth
                or manifest_row["expected_status"]
                != ("NG_NO_STREAK" if expected_truth == "no_streak" else "OK")
            ):
                raise ValueError("追踪光痕v3 manifest逐样本身份不一致")
        final_rows = {row["sample_id"]: row for row in rows if row["split"] == "final_test"}
        outcomes = payload["final_test"].get("outcomes")
        if not isinstance(outcomes, list) or len(outcomes) != 20 or len(final_rows) != 20:
            raise ValueError("追踪光痕v3 final-test逐样本证据不完整")
        outcome_ids = [outcome.get("sample_id") for outcome in outcomes if isinstance(outcome, dict)]
        if len(outcome_ids) != 20 or len(set(outcome_ids)) != 20 or set(outcome_ids) != set(final_rows):
            raise ValueError("追踪光痕v3 final-test逐样本身份不一致")
        for outcome in outcomes:
            if not isinstance(outcome, dict) or outcome.get("sample_id") not in final_rows:
                raise ValueError("追踪光痕v3 final-test逐样本证据不一致")
            row = final_rows[outcome["sample_id"]]
            expected_status = "NG_NO_STREAK" if row["truth"] == "no_streak" else "OK"
            predicted = row["tracked_profile_v3_predicted_status"]
            if (
                outcome.get("expected_status") != expected_status
                or outcome.get("predicted_status") != predicted
                or outcome.get("correct") is not (predicted == expected_status)
            ):
                raise ValueError("追踪光痕v3 final-test逐样本证据不一致")
        replay_path = root / "replay_summary.json"
        try:
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"追踪光痕v3 replay证据不可用：{error}") from error
        if replay != payload["replay"] or not isinstance(replay, dict):
            raise ValueError("追踪光痕v3 replay证据不一致")
        if (
            replay.get("count") != 20
            or replay.get("known_truth_count") != 1
            or replay.get("unknown_truth_count") != 19
            or not isinstance(replay.get("outcomes"), list)
            or len(replay["outcomes"]) != 20
        ):
            raise ValueError("追踪光痕v3 replay证据不完整")
        replay_ids = [
            outcome.get("capture_id")
            for outcome in replay["outcomes"]
            if isinstance(outcome, dict)
        ]
        metric_replay_ids = {
            row["sample_id"] for row in rows if row["split"] == "live_replay"
        } | {cls._CONFIRMED_LIVE_CAPTURE_ID}
        if (
            len(replay_ids) != 20
            or len(set(replay_ids)) != 20
            or set(replay_ids) != metric_replay_ids
        ):
            raise ValueError("追踪光痕v3 replay逐样本身份不一致")

    def predict(self, image: np.ndarray) -> ModelOutput:
        from bmw_inspection.lab.bright_streak_tracked_profile import (
            analyze_tracked_profile,
            classify_tracked_profile,
        )

        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim not in {2, 3}:
            raise ValueError("追踪光痕v3输入必须是uint8灰度或BGR图像")
        if self._rotated_roi is None:
            x1, y1, x2, y2 = self._roi
            height, width = image.shape[:2]
            if x2 > width or y2 > height:
                raise ValueError(f"追踪光痕v3 ROI超出输入图像{width}x{height}")
            roi_gray = _gray(image)[y1:y2, x1:x2]
        else:
            from bmw_inspection.lab.bright_streak_rotated_roi import rectify_bright_streak_roi

            roi_gray = _gray(rectify_bright_streak_roi(image, self._rotated_roi))
        metrics = analyze_tracked_profile(roi_gray, self._geometry, self._thresholds)
        decision = classify_tracked_profile(metrics, self._thresholds)
        status = BranchStatus.PASS if decision == "OK" else BranchStatus.NG
        overlay = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
        for row, column in enumerate(metrics.path_x):
            if metrics.strong_mask[row]:
                color = (0, 255, 0)
            elif metrics.bridged_mask[row]:
                color = (0, 200, 255)
            elif (
                metrics.active_start_row is not None
                and metrics.active_stop_row is not None
                and metrics.active_start_row <= row < metrics.active_stop_row
            ):
                color = (0, 0, 255)
            else:
                color = (128, 128, 128)
            overlay[row, int(column)] = color
        present = bool(metrics.strong_mask.any())
        reason = (
            f"追踪光痕判定 {decision}；是否存在 {'是' if present else '否'}；"
            f"覆盖率 {metrics.coverage_ratio:.3f}（阈值 "
            f"{self._thresholds.min_presence_coverage_ratio:.3f}）；"
            f"最长连续段 {metrics.longest_run_ratio:.3f}（阈值 "
            f"{self._thresholds.min_longest_run_ratio:.3f}）；"
            f"最大断点 {metrics.max_gap_ratio:.3f}（上限 {self._thresholds.max_gap_ratio:.3f}）；"
            f"断点数 {metrics.gap_count}（上限 {self._thresholds.max_gap_count}）；"
            f"强阈值 {self._thresholds.strong_row_score:.3f}，"
            f"弱阈值 {self._thresholds.weak_row_score:.3f}，"
            f"桥接行 {int(np.count_nonzero(metrics.bridged_mask))}"
        )
        roi_details: dict[str, object] = {"roi_xyxy": self._roi}
        if self._rotated_roi is not None:
            if self._weak_row_score_overridden:
                reason = (
                    "手动倾斜ROI，倾斜HDR现场重标定弱阈值 "
                    f"{self._report_weak_row_score:.3f}→{self._thresholds.weak_row_score:.3f}；{reason}"
                )
                threshold_calibration = "rotated_hdr_field_recalibration_v1"
            else:
                reason = f"手动倾斜ROI，沿用V3阈值（未重标定）；{reason}"
                threshold_calibration = "existing_v3_not_recalibrated"
            roi_details.update(
                {
                    "roi_points_xy": self._rotated_roi.points_xy,
                    "roi_mode": "manual_rotated_perspective",
                    "threshold_calibration": threshold_calibration,
                    "rotated_roi_sha256": self._rotated_roi_sha256,
                    "report_weak_row_score": self._report_weak_row_score,
                }
            )
        return ModelOutput(
            status,
            metrics.coverage_ratio,
            self._thresholds.min_presence_coverage_ratio,
            reason,
            overlay,
            details={
                "evidence_type": "追踪中心线诊断证据",
                "decision": decision,
                **roi_details,
                "presence": present,
                "coverage_ratio": metrics.coverage_ratio,
                "longest_run_px": metrics.longest_run_px,
                "longest_run_ratio": metrics.longest_run_ratio,
                "max_gap_px": metrics.max_gap_px,
                "max_gap_ratio": metrics.max_gap_ratio,
                "gap_count": metrics.gap_count,
                "strong_row_score": self._thresholds.strong_row_score,
                "weak_row_score": self._thresholds.weak_row_score,
                "bridged_rows": int(np.count_nonzero(metrics.bridged_mask)),
                "active_start_row": metrics.active_start_row,
                "active_stop_row": metrics.active_stop_row,
                "max_step": self._geometry.max_step,
                "tracked_centerline_x": tuple(int(value) for value in metrics.path_x),
            },
        )


def _default_yolo_factory(path: Path) -> Any:
    from ultralytics import YOLO

    return YOLO(str(path))


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class EightViewYoloPredictor:
    """One resident one-class YOLO model shared by all ROI crops."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        candidate_conf: float,
        final_threshold: float,
        imgsz: int,
        model_factory: Callable[[Path], Any] = _default_yolo_factory,
    ) -> None:
        resolved = Path(checkpoint).expanduser().resolve()
        if not resolved.is_file():
            raise ValueError(f"YOLO模型不存在：{resolved}")
        if not 0 <= candidate_conf <= final_threshold <= 1:
            raise ValueError("YOLO阈值必须满足0 <= candidate_conf <= final_threshold <= 1")
        self._candidate_conf = float(candidate_conf)
        self._final_threshold = float(final_threshold)
        self._imgsz = int(imgsz)
        self._model = model_factory(resolved)
        names = getattr(self._model, "names", None)
        parsed = {index: name for index, name in enumerate(names)} if isinstance(names, list) else dict(names or {})
        if parsed != {0: "defect"}:
            raise ValueError("YOLO模型必须只包含类别defect")
        self._lock = threading.Lock()

    def predict(self, view: str, image: np.ndarray) -> ModelOutput:
        del view
        with self._lock:
            predictions = self._model.predict(
                source=image,
                conf=self._candidate_conf,
                imgsz=self._imgsz,
                verbose=False,
            )
        if len(predictions) != 1:
            raise ValueError("YOLO必须返回一张图的结果")
        boxes = getattr(predictions[0], "boxes", None)
        if boxes is None:
            coordinates = np.empty((0, 4), dtype=np.float32)
            confidences = np.empty((0,), dtype=np.float32)
            classes = np.empty((0,), dtype=np.float32)
        else:
            coordinates = _array(boxes.xyxy).reshape((-1, 4))
            confidences = _array(boxes.conf).reshape(-1)
            classes = _array(boxes.cls).reshape(-1)
        if not (len(coordinates) == len(confidences) == len(classes)):
            raise ValueError("YOLO检测框字段长度不一致")
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
        final_count = 0
        box_details: list[dict[str, Any]] = []
        for xyxy, confidence, class_id in zip(coordinates, confidences, classes, strict=True):
            if int(class_id) != 0:
                raise ValueError("YOLO输出了非defect类别")
            final = float(confidence) >= self._final_threshold
            final_count += int(final)
            color = (0, 0, 255) if final else (0, 170, 255)
            x1, y1, x2, y2 = (int(round(float(value))) for value in xyxy)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3 if final else 2)
            cv2.putText(
                overlay,
                f"defect {float(confidence):.2f}",
                (max(0, x1), max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
            box_details.append(
                {
                    "xyxy": (x1, y1, x2, y2),
                    "confidence": float(confidence),
                    "class_name": "defect",
                    "is_final": final,
                }
            )
        score = max((float(value) for value in confidences), default=0.0)
        return ModelOutput(
            BranchStatus.NG if final_count else BranchStatus.PASS,
            score,
            self._final_threshold,
            (
                f"YOLO 真实检测框：最终缺陷框 {final_count} 个，候选框 {len(confidences)} 个；"
                f"最高置信度 {score:.4f}，部署阈值 {self._final_threshold:.4f}，"
                f"候选阈值 {self._candidate_conf:.4f}"
            ),
            overlay,
            details={
                "evidence_type": "真实检测框",
                "candidate_threshold": self._candidate_conf,
                "deployment_threshold": self._final_threshold,
                "threshold_exceedance": score - self._final_threshold,
                "candidate_box_count": len(confidences),
                "final_box_count": final_count,
                "boxes": box_details,
            },
        )


EfficientPrediction = tuple[float, bool, np.ndarray]
EfficientPredictorFactory = Callable[[Path], Callable[[np.ndarray], EfficientPrediction]]


class _AnomalibEfficientPredictor:
    """One resident EfficientAD checkpoint with Anomalib preprocessing."""

    def __init__(self, checkpoint: Path) -> None:
        from anomalib.engine import Engine
        from anomalib.models import EfficientAd

        self._model = EfficientAd.load_from_checkpoint(
            checkpoint,
            map_location="cpu",
            weights_only=False,
            visualizer=False,
        )
        self._engine = Engine(logger=False)

    def __call__(self, image: np.ndarray) -> EfficientPrediction:
        with tempfile.TemporaryDirectory(prefix="bmw-efficientad-demo-") as directory:
            path = Path(directory) / "roi.png"
            if not cv2.imwrite(str(path), image):
                raise RuntimeError("无法编码EfficientAD临时ROI")
            predictions = self._engine.predict(
                model=self._model,
                data_path=path,
                ckpt_path=None,
                return_predictions=True,
            )
        items = [] if predictions is None else [item for batch in predictions for item in batch]
        if len(items) != 1:
            raise RuntimeError("EfficientAD必须返回一个预测")
        item = items[0]
        score_values = _array(item.pred_score).reshape(-1)
        label_values = _array(item.pred_label).reshape(-1)
        anomaly_map = _array(item.anomaly_map)
        while anomaly_map.ndim > 2 and anomaly_map.shape[0] == 1:
            anomaly_map = anomaly_map[0]
        if score_values.size != 1 or label_values.size != 1 or anomaly_map.ndim != 2:
            raise RuntimeError("EfficientAD预测字段形状不正确")
        return float(score_values[0]), bool(label_values[0]), anomaly_map.astype(np.float32, copy=False)


class EightViewEfficientAdPredictor:
    """Keep eight EfficientAD-S predictors resident and expose comparable normalized scores."""

    def __init__(
        self,
        checkpoints: Mapping[str, Path],
        *,
        thresholds: Mapping[str, float],
        base_thresholds: Mapping[str, float] | None = None,
        threshold_margin: float = 0.0,
        predictor_factory: EfficientPredictorFactory = _AnomalibEfficientPredictor,
        ignore_masks: Mapping[str, np.ndarray] | None = None,
        ignore_mask_index_sha256: str | None = None,
    ) -> None:
        if tuple(checkpoints) != VIEW_ORDER:
            raise ValueError("EfficientAD模型必须按标准顺序覆盖八个视角")
        if tuple(thresholds) != VIEW_ORDER:
            raise ValueError("EfficientAD阈值必须按标准顺序覆盖八个视角")
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in thresholds.values()):
            raise ValueError("EfficientAD阈值必须是有限数值")
        from bmw_inspection.lab.efficientad_thresholds import _validated_thresholds

        try:
            _views, threshold_values = _validated_thresholds(thresholds)
        except ValueError as error:
            raise ValueError("EfficientAD阈值必须是有限数值") from error
        self._thresholds = MappingProxyType(threshold_values)
        base_input = thresholds if base_thresholds is None else base_thresholds
        if tuple(base_input) != VIEW_ORDER:
            raise ValueError("EfficientAD基础阈值必须按标准顺序覆盖八个视角")
        try:
            _base_views, base_values = _validated_thresholds(base_input)
        except ValueError as error:
            raise ValueError("EfficientAD基础阈值必须是有限数值") from error
        if isinstance(threshold_margin, bool) or not isinstance(threshold_margin, Real):
            raise ValueError("EfficientAD阈值余量必须是有限非负数值")
        margin = float(threshold_margin)
        if not math.isfinite(margin) or margin < 0:
            raise ValueError("EfficientAD阈值余量必须是有限非负数值")
        if any(
            not math.isclose(threshold_values[view], base_values[view] + margin, abs_tol=1e-12)
            for view in VIEW_ORDER
        ):
            raise ValueError("EfficientAD基础阈值加余量必须等于部署阈值")
        self._base_thresholds = MappingProxyType(base_values)
        self._threshold_margin = margin
        if ignore_masks is None:
            if ignore_mask_index_sha256 is not None:
                raise ValueError("EfficientAD ignore-mask SHA requires ignore masks")
            self._ignore_masks: Mapping[str, np.ndarray] | None = None
            self._ignore_mask_index_sha256 = None
        else:
            if tuple(ignore_masks) != VIEW_ORDER:
                raise ValueError("EfficientAD ignore masks must use the canonical eight-view order")
            if not isinstance(ignore_mask_index_sha256, str) or len(ignore_mask_index_sha256) != 64:
                raise ValueError("EfficientAD ignore-mask index SHA must contain 64 characters")
            owned_masks: dict[str, np.ndarray] = {}
            for view, mask in ignore_masks.items():
                array = np.asarray(mask)
                if array.dtype != np.uint8 or array.ndim != 2 or array.size == 0:
                    raise ValueError(f"EfficientAD ignore mask is invalid: {view}")
                if not set(np.unique(array).tolist()).issubset({0, 255}):
                    raise ValueError(f"EfficientAD ignore mask is not binary: {view}")
                copied = array.copy()
                copied.flags.writeable = False
                owned_masks[view] = copied
            self._ignore_masks = MappingProxyType(owned_masks)
            self._ignore_mask_index_sha256 = ignore_mask_index_sha256
        predictors: dict[str, Callable[[np.ndarray], EfficientPrediction]] = {}
        for view in VIEW_ORDER:
            checkpoint = Path(checkpoints[view]).expanduser().resolve()
            if not checkpoint.is_file():
                raise ValueError(f"EfficientAD模型不存在：{checkpoint}")
            predictors[view] = predictor_factory(checkpoint)
        self._predictors = MappingProxyType(predictors)
        self._lock = threading.Lock()

    def predict(self, view: str, image: np.ndarray) -> ModelOutput:
        with self._lock:
            score, raw_pred_label, anomaly_map = self._predictors[view](image)
        if not math.isfinite(score) or not np.isfinite(anomaly_map).all():
            raise ValueError("EfficientAD输出包含非有限数值")
        raw_pred_score = score
        scoring_map = anomaly_map
        score_source = "pred_score"
        ignored_roi_pixel_count = 0
        ignored_map_pixel_count = 0
        raw_map_max = float(np.max(anomaly_map))
        if self._ignore_masks is not None and np.any(self._ignore_masks[view]):
            from bmw_inspection.lab.efficientad_ignore_mask import mask_anomaly_map

            masked = mask_anomaly_map(anomaly_map, self._ignore_masks[view])
            score = masked.score
            scoring_map = masked.masked_map
            score_source = "manual_ignore_masked_anomaly_map_max"
            ignored_roi_pixel_count = int(np.count_nonzero(self._ignore_masks[view]))
            ignored_map_pixel_count = masked.ignored_map_pixel_count
            raw_map_max = masked.raw_max
        threshold = self._thresholds[view]
        base_threshold = self._base_thresholds[view]
        heatmap = fixed_scale_heatmap(scoring_map)
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
        heatmap = cv2.resize(heatmap, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
        overlay = cv2.addWeighted(base, 0.6, heatmap, 0.4, 0.0)
        hotspot_y, hotspot_x = np.unravel_index(int(np.argmax(scoring_map)), scoring_map.shape)
        display_x = int(round(hotspot_x * max(0, base.shape[1] - 1) / max(1, anomaly_map.shape[1] - 1)))
        display_y = int(round(hotspot_y * max(0, base.shape[0] - 1) / max(1, anomaly_map.shape[0] - 1)))
        cv2.drawMarker(overlay, (display_x, display_y), (0, 0, 255), cv2.MARKER_CROSS, 13, 2)
        score_label = (
            "手动忽略区外异常图最大值"
            if score_source == "manual_ignore_masked_anomaly_map_max"
            else "异常分数"
        )
        return ModelOutput(
            BranchStatus.NG if score >= threshold else BranchStatus.PASS,
            score,
            threshold,
            (
                f"EfficientAD{score_label} {score:.4f}，基础阈值 {base_threshold:.4f}，"
                f"部署阈值 {threshold:.4f}，余量 {self._threshold_margin:.4f}"
            ),
            overlay,
            raw_pred_label=raw_pred_label,
            details={
                "evidence_type": "诊断热区",
                "score": score,
                "raw_pred_score": raw_pred_score,
                "raw_anomaly_map_max": raw_map_max,
                "score_source": score_source,
                "base_threshold": base_threshold,
                "deployment_threshold": threshold,
                "threshold_margin": self._threshold_margin,
                "threshold_exceedance": score - threshold,
                "hotspot_x": int(hotspot_x),
                "hotspot_y": int(hotspot_y),
                "hotspot_value": float(scoring_map[hotspot_y, hotspot_x]),
                "ignore_mask_index_sha256": self._ignore_mask_index_sha256,
                "ignored_roi_pixel_count": ignored_roi_pixel_count,
                "ignored_map_pixel_count": ignored_map_pixel_count,
                "raw_pred_label": raw_pred_label,
            },
        )


class EightViewModelSuite:
    """Run every configured branch and retain evidence even after an earlier NG."""

    def __init__(
        self,
        *,
        rois: Mapping[str, tuple[int, int, int, int]],
        template_predictor: ViewPredictor,
        bright_streak_predictor: BrightPredictor,
        yolo_predictor: ViewPredictor,
        efficientad_predictor: ViewPredictor,
        trusted_ok_matcher: TrustedOkMatcher | None = None,
        trusted_ok_matcher_error: str | None = None,
    ) -> None:
        if tuple(rois) != VIEW_ORDER:
            raise ValueError("rois must use the canonical BMW eight-view order")
        parsed: dict[str, tuple[int, int, int, int]] = {}
        for view, roi in rois.items():
            if len(roi) != 4 or any(isinstance(value, bool) or not isinstance(value, int) for value in roi):
                raise ValueError(f"ROI for {view} must contain four integers")
            x1, y1, x2, y2 = roi
            if not (0 <= x1 < x2 and 0 <= y1 < y2):
                raise ValueError(f"ROI for {view} must have positive area")
            parsed[view] = roi
        self._rois = parsed
        self._template = template_predictor
        self._bright_streak = bright_streak_predictor
        self._yolo = yolo_predictor
        self._efficientad = efficientad_predictor
        self._trusted_ok_matcher = trusted_ok_matcher
        self._trusted_ok_matcher_error = trusted_ok_matcher_error

    def inspect(self, images: Mapping[str, np.ndarray], *, capture_id: str) -> EightViewInspection:
        """Run 8 Template + 1 light streak + 8 YOLO + 8 EfficientAD checks."""
        if tuple(images) != VIEW_ORDER:
            raise ValueError("images must use the canonical BMW eight-view order")
        started = perf_counter()
        crops = {view: self._crop(view, images[view]) for view in VIEW_ORDER}
        results: list[DemoBranchResult] = []
        for view in VIEW_ORDER:
            results.append(self._call(DemoBranch.TEMPLATE, view, self._template, view, crops[view]))
        results.append(
            self._call(
                DemoBranch.BRIGHT_STREAK,
                "front_left",
                self._bright_streak,
                images["front_left"],
            )
        )
        for view in VIEW_ORDER:
            results.append(self._call(DemoBranch.YOLO, view, self._yolo, view, crops[view]))
        for view in VIEW_ORDER:
            results.append(self._call(DemoBranch.EFFICIENTAD, view, self._efficientad, view, crops[view]))
        rows = tuple(results)
        final_status = fuse_demo_status(rows)
        actionable_comparisons = tuple(
            dict.fromkeys(
                (
                    row.view_id,
                    "full" if row.branch is DemoBranch.BRIGHT_STREAK else "roi",
                )
                for row in rows
                if row.status in {BranchStatus.NG, BranchStatus.ERROR}
            )
        )
        trusted_ok_by_comparison: dict[tuple[str, str], TrustedOkMatch] = {}
        match_errors: dict[str, str] = {}
        for view, comparison_mode in actionable_comparisons:
            comparison_id = f"{view}/{comparison_mode}"
            if self._trusted_ok_matcher is None:
                if self._trusted_ok_matcher_error is not None:
                    match_errors[comparison_id] = self._trusted_ok_matcher_error
                continue
            try:
                trusted_ok_by_comparison[(view, comparison_mode)] = self._trusted_ok_matcher.match(
                    view,
                    images[view],
                    crops[view],
                    comparison_mode=comparison_mode,
                )
            except Exception as error:
                match_errors[comparison_id] = str(error)
        diagnostic_metadata = (
            {"trusted_ok_match_errors": match_errors} if match_errors else {}
        )
        return EightViewInspection(
            capture_id=capture_id,
            images=images,
            results=rows,
            final_status=final_status,
            elapsed_ms=(perf_counter() - started) * 1000.0,
            roi_images=crops,
            trusted_ok_by_comparison=trusted_ok_by_comparison,
            diagnostic_metadata=diagnostic_metadata,
        )

    def _crop(self, view: str, image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim not in {2, 3}:
            raise ValueError(f"{view} must be a uint8 grayscale/BGR image")
        x1, y1, x2, y2 = self._rois[view]
        height, width = image.shape[:2]
        if x2 > width or y2 > height:
            raise ValueError(f"ROI for {view} lies outside {width}x{height}")
        return image[y1:y2, x1:x2].copy()

    @staticmethod
    def _call(branch: DemoBranch, view: str, predictor: Callable[..., ModelOutput], *args: object) -> DemoBranchResult:
        started = perf_counter()
        try:
            output = predictor(*args)
            if not isinstance(output, ModelOutput):
                raise TypeError("模型后端必须返回ModelOutput")
            return DemoBranchResult(
                branch=branch,
                view_id=view,
                status=output.status,
                score=output.score,
                threshold=output.threshold,
                elapsed_ms=(perf_counter() - started) * 1000.0,
                reason=output.reason,
                overlay=output.overlay,
                raw_pred_label=output.raw_pred_label,
                details=output.details,
            )
        except Exception as error:
            return DemoBranchResult(
                branch=branch,
                view_id=view,
                status=BranchStatus.ERROR,
                score=None,
                threshold=None,
                elapsed_ms=(perf_counter() - started) * 1000.0,
                reason=f"{branch.value}推理失败：{error}",
                overlay=None,
                details={"evidence_type": "不可用", "error": str(error)},
            )


def build_model_suite(
    config: EightViewDemoConfig,
    *,
    status_callback: Callable[[str], None] | None = None,
) -> EightViewModelSuite:
    """Build the four resident model branches from one resolved Demo profile."""
    if config.bright_streak_engine == "raw_profile_v2":
        bright_streak = EightViewRawProfileBrightStreakPredictor(config.bright_streak_config)
    elif config.bright_streak_engine == "tracked_profile_v3":
        bright_streak = EightViewTrackedProfileBrightStreakPredictor(config.bright_streak_config)
    elif config.bright_streak_engine == "tracked_profile_v3_manual_rotated_roi":
        bright_streak = EightViewTrackedProfileBrightStreakPredictor(
            config.bright_streak_config,
            rotated_roi_path=getattr(config, "bright_streak_rotated_roi", None),
            rotated_roi_sha256=getattr(config, "bright_streak_rotated_roi_sha256", None),
            weak_row_score_override=getattr(
                config,
                "bright_streak_weak_row_score_override",
                None,
            ),
        )
    else:
        bright_streak = EightViewBrightStreakPredictor(config.bright_streak_config)
    yolo = EightViewYoloPredictor(
        config.yolo_checkpoint,
        candidate_conf=config.yolo_candidate_conf,
        final_threshold=config.yolo_final_threshold,
        imgsz=config.yolo_imgsz,
    )
    rois = load_part_rois(config.roi_config)
    ignore_masks: Mapping[str, np.ndarray] | None = None
    ignore_mask_index_sha256: str | None = None
    configured_ignore_mask_index = getattr(config, "efficientad_ignore_mask_index", None)
    configured_ignore_mask_sha256 = getattr(config, "efficientad_ignore_mask_index_sha256", None)
    if configured_ignore_mask_index is not None:
        from bmw_inspection.lab.efficientad_ignore_mask import load_ignore_mask_asset

        asset = load_ignore_mask_asset(
            configured_ignore_mask_index,
            expected_views=VIEW_ORDER,
            expected_roi_config_sha256=_file_sha256(config.roi_config),
            expected_shapes={
                view: (rois[view][3] - rois[view][1], rois[view][2] - rois[view][0])
                for view in VIEW_ORDER
            },
        )
        if asset.index_sha256 != configured_ignore_mask_sha256:
            raise ValueError("EfficientAD ignore mask index SHA256不匹配")
        ignore_masks = asset.masks
        ignore_mask_index_sha256 = asset.index_sha256
    configured_template_mask_index = getattr(config, "template_ignore_mask_index", None)
    if configured_template_mask_index is None:
        template = EightViewTemplatePredictor(config.template_models)
    else:
        configured_template_mask_sha256 = getattr(config, "template_ignore_mask_index_sha256", None)
        if (
            ignore_masks is None
            or ignore_mask_index_sha256 is None
            or configured_template_mask_sha256 != ignore_mask_index_sha256
        ):
            raise ValueError("Template与EfficientAD没有加载同一手动ignore mask资产")
        template = EightViewTemplatePredictor(
            config.template_models,
            ignore_masks=ignore_masks,
            ignore_mask_index_sha256=ignore_mask_index_sha256,
            masked_thresholds=getattr(config, "template_masked_thresholds", None),
            masked_threshold_artifact_sha256=getattr(
                config,
                "template_masked_threshold_artifact_sha256",
                None,
            ),
        )
    efficientad = EightViewEfficientAdPredictor(
        config.efficientad_checkpoints,
        thresholds=config.efficientad_thresholds,
        base_thresholds=config.efficientad_base_thresholds,
        threshold_margin=config.efficientad_threshold_margin,
        ignore_masks=ignore_masks,
        ignore_mask_index_sha256=ignore_mask_index_sha256,
    )
    trusted_ok_matcher: TrustedOkMatcher | None = None
    trusted_ok_matcher_error: str | None = None
    index_path = getattr(config, "trusted_ok_reference_index", None)
    index_sha256 = getattr(config, "trusted_ok_reference_index_sha256", None)
    prevalidated_error = getattr(config, "trusted_ok_reference_error", None)
    if index_path is not None:
        announce = status_callback if status_callback is not None else (lambda _message: None)
        announce("正在校验并预热可信OK参考库，首次启动约需27秒……")
        try:
            if prevalidated_error is not None:
                raise ValueError(prevalidated_error)
            if index_sha256 is None:
                raise ValueError("配置缺少可信OK索引SHA256")
            candidate = TrustedOkMatcher(
                Path(index_path).parent,
                expected_index_sha256=index_sha256,
            )
            candidate.preload()
            current_roi_sha256 = _file_sha256(Path(config.roi_config).expanduser().resolve())
            if candidate.roi_config_sha256 != current_roi_sha256:
                raise ValueError("可信参考ROI配置与当前Demo ROI配置SHA256不匹配")
        except Exception as error:
            trusted_ok_matcher_error = f"可信OK参考不可用：{error}"
            announce(f"可信OK参考库不可用，已仅禁用参考诊断：{error}")
        else:
            trusted_ok_matcher = candidate
            announce("可信OK参考库预热完成。")
    return EightViewModelSuite(
        rois=rois,
        template_predictor=template.predict,
        bright_streak_predictor=bright_streak.predict,
        yolo_predictor=yolo.predict,
        efficientad_predictor=efficientad.predict,
        trusted_ok_matcher=trusted_ok_matcher,
        trusted_ok_matcher_error=trusted_ok_matcher_error,
    )


def _file_sha256(path: Path) -> str:
    """Hash one resolved runtime asset for cross-contract binding."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "EightViewBrightStreakPredictor",
    "EightViewRawProfileBrightStreakPredictor",
    "EightViewTrackedProfileBrightStreakPredictor",
    "EightViewEfficientAdPredictor",
    "EightViewModelSuite",
    "EightViewTemplatePredictor",
    "EightViewYoloPredictor",
    "ModelOutput",
    "build_model_suite",
    "load_part_rois",
]
