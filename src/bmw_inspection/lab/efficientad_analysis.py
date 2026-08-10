"""Offline score and heatmap helpers for BMW EfficientAD evaluation."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class EfficientAdScore:
    """One image-level EfficientAD result used by the offline report."""

    view_id: str
    label: str
    image_path: Path
    score: float
    predicted_anomalous: bool

    def __post_init__(self) -> None:
        """Reject incomplete report rows before publication."""
        if not self.view_id:
            raise ValueError("view_id must be non-empty")
        if self.label not in {"normal", "defect"}:
            raise ValueError("label must be normal or defect")
        if not isinstance(self.image_path, Path):
            raise TypeError("image_path must be a Path")
        if not math.isfinite(float(self.score)):
            raise ValueError("score must be finite")


def fixed_scale_heatmap(anomaly_map: np.ndarray) -> np.ndarray:
    """Colorize one anomaly map on a shared zero-to-one scale."""
    values = np.asarray(anomaly_map, dtype=np.float32)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("anomaly_map must be a non-empty 2D array")
    if not np.isfinite(values).all():
        raise ValueError("anomaly_map must contain only finite values")
    normalized = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def select_hard_examples(
    records: Sequence[EfficientAdScore],
    *,
    per_label: int,
) -> dict[str, tuple[EfficientAdScore, ...]]:
    """Select high-score normal and low-score defect examples."""
    if isinstance(per_label, bool) or not isinstance(per_label, int) or per_label <= 0:
        raise ValueError("per_label must be a positive integer")
    normals = sorted(
        (item for item in records if item.label == "normal"),
        key=lambda item: (-item.score, item.image_path.name),
    )
    defects = sorted(
        (item for item in records if item.label == "defect"),
        key=lambda item: (item.score, item.image_path.name),
    )
    return {
        "normal": tuple(normals[:per_label]),
        "defect": tuple(defects[:per_label]),
    }


def score_metrics(records: Sequence[EfficientAdScore]) -> dict[str, float | int]:
    """Recompute image metrics from the exact runtime scores and labels."""
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

    labels = [int(item.label == "defect") for item in records]
    if set(labels) != {0, 1}:
        raise ValueError("score metrics require both normal and defect records")
    scores = [item.score for item in records]
    predictions = [int(item.predicted_anomalous) for item in records]
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "f1": float(f1_score(labels, predictions, zero_division=0.0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "normal_predicted_ng": sum(
            label == 0 and prediction == 1 for label, prediction in zip(labels, predictions, strict=True)
        ),
        "defect_predicted_ok": sum(
            label == 1 and prediction == 0 for label, prediction in zip(labels, predictions, strict=True)
        ),
    }


def _validated_threshold(threshold: float) -> float:
    value = float(threshold)
    if not math.isfinite(value):
        raise ValueError("threshold must be finite")
    return value


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import font_manager, pyplot

    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        pyplot.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    pyplot.rcParams["axes.unicode_minus"] = False
    return pyplot


def write_score_csv(
    output_path: Path,
    records: Sequence[EfficientAdScore],
    *,
    threshold: float,
) -> Path:
    """Write deterministic per-image scores for later threshold analysis."""
    threshold = _validated_threshold(threshold)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("view_id", "label", "prediction", "score", "threshold", "image_path"))
        for item in sorted(records, key=lambda row: (row.view_id, row.label, row.image_path.name)):
            writer.writerow(
                (
                    item.view_id,
                    item.label,
                    "NG" if item.predicted_anomalous else "OK",
                    f"{item.score:.12g}",
                    f"{threshold:.12g}",
                    str(item.image_path),
                )
            )
    return output


def render_score_distributions(
    output_path: Path,
    records: Sequence[EfficientAdScore],
    *,
    views: Sequence[str],
    threshold: float,
) -> Path:
    """Render comparable normal/defect score panels for every requested view."""
    threshold = _validated_threshold(threshold)
    view_order = tuple(views)
    if not view_order or len(set(view_order)) != len(view_order):
        raise ValueError("views must be non-empty and unique")
    pyplot = _pyplot()
    columns = min(2, len(view_order))
    rows = math.ceil(len(view_order) / columns)
    figure, raw_axes = pyplot.subplots(rows, columns, figsize=(7.2 * columns, 4.2 * rows), squeeze=False)
    axes = raw_axes.ravel()
    score_values = [item.score for item in records]
    lower = min(0.0, min(score_values, default=0.0))
    upper = max(1.0, max(score_values, default=1.0))
    padding = max(0.03, (upper - lower) * 0.05)
    for axis, view in zip(axes, view_order, strict=False):
        view_records = [item for item in records if item.view_id == view]
        for x_position, label, color, chinese in (
            (0.0, "normal", "#2E8B57", "正常"),
            (1.0, "defect", "#D94841", "缺陷"),
        ):
            values = [item.score for item in view_records if item.label == label]
            offsets = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.zeros(len(values))
            axis.scatter(
                x_position + offsets,
                values,
                s=30,
                alpha=0.8,
                color=color,
                label=f"{chinese} n={len(values)}",
            )
        axis.axhline(threshold, color="#202124", linestyle="--", linewidth=1.4, label=f"阈值 {threshold:.2f}")
        axis.set_title(view)
        axis.set_xticks((0, 1), ("正常", "缺陷"))
        axis.set_ylabel("EfficientAD异常分数")
        axis.set_ylim(lower - padding, upper + padding)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(loc="best", fontsize=9)
    for axis in axes[len(view_order) :]:
        axis.set_visible(False)
    figure.suptitle("BMW八视角 EfficientAD 正常/缺陷分数分布", fontsize=16)
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    pyplot.close(figure)
    return output


def render_example_sheet(
    output_path: Path,
    records: Sequence[EfficientAdScore],
    anomaly_maps: Mapping[Path, np.ndarray],
    *,
    threshold: float,
    per_label: int,
) -> Path:
    """Render the hardest normal and defect overlays with one fixed color scale."""
    threshold = _validated_threshold(threshold)
    selected = select_hard_examples(records, per_label=per_label)
    pyplot = _pyplot()
    figure, axes = pyplot.subplots(2, per_label, figsize=(5.0 * per_label, 8.0), squeeze=False)
    for row_index, (label, chinese) in enumerate((("normal", "高分正常"), ("defect", "低分缺陷"))):
        examples = selected[label]
        for column_index in range(per_label):
            axis = axes[row_index, column_index]
            if column_index >= len(examples):
                axis.set_visible(False)
                continue
            item = examples[column_index]
            image = cv2.imread(str(item.image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"cannot decode report image: {item.image_path}")
            if item.image_path not in anomaly_maps:
                raise ValueError(f"anomaly map is missing: {item.image_path}")
            heatmap = fixed_scale_heatmap(anomaly_maps[item.image_path])
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
            overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0.0)
            prediction = "NG" if item.predicted_anomalous else "OK"
            axis.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            axis.set_title(f"{chinese} | 分数 {item.score:.4f} | {prediction}")
            axis.set_xlabel(item.image_path.name, fontsize=8)
            axis.set_xticks(())
            axis.set_yticks(())
    view_id = records[0].view_id if records else "unknown"
    figure.suptitle(f"{view_id}：最难样本（统一0–1色标，阈值{threshold:.2f}）", fontsize=15)
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    pyplot.close(figure)
    return output


__all__ = [
    "EfficientAdScore",
    "fixed_scale_heatmap",
    "render_example_sheet",
    "render_score_distributions",
    "score_metrics",
    "select_hard_examples",
    "write_score_csv",
]
