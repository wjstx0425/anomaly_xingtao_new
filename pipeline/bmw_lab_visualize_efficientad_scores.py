#!/usr/bin/env python3
"""导出BMW八视图EfficientAD逐图分数、分布图和统一色标热力图。"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bmw_inspection.lab.efficientad_analysis import (  # noqa: E402
    EfficientAdScore,
    render_example_sheet,
    render_score_distributions,
    score_metrics,
    write_score_csv,
)
from bmw_inspection.lab.efficientad_thresholds import part_id_from_image_path  # noqa: E402
from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER  # noqa: E402

DEFAULT_TRAINING_RELEASE = REPO_ROOT / "dataset/bmw_lab_training/bmw_hdr_roi_training_reviewed_v1"
DEFAULT_MODEL_ROOT = REPO_ROOT / "results/bmw_lab_one_click/bmw_lab_eight_view_v1/efficientad"
DEFAULT_OUTPUT_DIR = DEFAULT_MODEL_ROOT / "score_analysis"


def build_parser() -> argparse.ArgumentParser:
    """Build the offline score-analysis command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-release", type=Path, default=DEFAULT_TRAINING_RELEASE)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--examples-per-label", type=int, default=3)
    parser.add_argument("--views", nargs="+", choices=VIEW_ORDER, default=list(VIEW_ORDER))
    return parser


def _array(value: object) -> np.ndarray:
    current = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "numpy"):
        current = current.numpy()
    return np.asarray(current)


def _predictions_for_directory(
    engine: object,
    model: object,
    directory: Path,
    *,
    view: str,
    label: str,
) -> tuple[list[EfficientAdScore], dict[Path, np.ndarray]]:
    predictions = engine.predict(
        model=model,
        data_path=directory,
        ckpt_path=None,
        return_predictions=True,
    )
    return _prediction_records(predictions, view=view, label=label)


def _prediction_records(
    predictions: object,
    *,
    view: str,
    label: str,
) -> tuple[list[EfficientAdScore], dict[Path, np.ndarray]]:
    """Normalize Anomalib prediction batches into the score-report contract."""
    records: list[EfficientAdScore] = []
    anomaly_maps: dict[Path, np.ndarray] = {}
    for batch in predictions or []:
        paths = tuple(Path(path) for path in batch.image_path)
        scores = _array(batch.pred_score).reshape(-1)
        predicted_labels = _array(batch.pred_label).reshape(-1)
        maps = _array(batch.anomaly_map)
        while maps.ndim > 3 and maps.shape[1] == 1:
            maps = maps[:, 0]
        if maps.ndim == 2:
            maps = maps[np.newaxis, ...]
        if len(paths) != len(scores) or len(paths) != len(predicted_labels) or len(paths) != len(maps):
            raise RuntimeError(f"EfficientAD批量输出数量不一致：{view}/{label}")
        for path, score, predicted, anomaly_map in zip(
            paths,
            scores,
            predicted_labels,
            maps,
            strict=True,
        ):
            value = float(score)
            map_value = np.asarray(anomaly_map, dtype=np.float32)
            if not math.isfinite(value) or not np.isfinite(map_value).all():
                raise ValueError(f"EfficientAD输出包含非有限数值：{path}")
            records.append(EfficientAdScore(view, label, path, value, bool(predicted)))
            anomaly_maps[path] = map_value
    return records, anomaly_maps


def _predictions_for_paths(
    engine: object,
    model: object,
    image_paths: tuple[Path, ...],
    *,
    view: str,
    label: str,
) -> tuple[list[EfficientAdScore], dict[Path, np.ndarray]]:
    """Predict one exact, auditable image set without including directory siblings."""
    from anomalib.data import PredictDataset

    paths = tuple(Path(path) for path in image_paths)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("EfficientAD精确推理路径必须非空且唯一")
    dataset = PredictDataset(paths[0])
    dataset.image_filenames = list(paths)
    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=None,
        return_predictions=True,
    )
    records, maps = _prediction_records(predictions, view=view, label=label)
    if len(records) != len(paths) or {record.image_path for record in records} != set(paths):
        raise RuntimeError(f"EfficientAD精确推理返回了非请求图像：{view}/{label}")
    return records, maps


def _complete_defect_crop_paths(training_release: Path, views: tuple[str, ...]) -> dict[str, tuple[Path, ...]]:
    """Select only confirmed defect parts, then bind each to one crop per view."""
    release = Path(training_release)
    view_order = tuple(views)
    if not view_order or len(set(view_order)) != len(view_order):
        raise ValueError("缺陷补全视图必须非空且唯一")
    selected_parts: set[str] = set()
    for view in view_order:
        defect_root = release / "efficientad" / view / "defect"
        if not defect_root.is_dir():
            raise ValueError(f"EfficientAD缺陷目录不存在：{defect_root}")
        for image_path in sorted(defect_root.rglob("*.png")):
            if image_path.parent.name != "images" or not image_path.is_file():
                raise ValueError(f"EfficientAD可见缺陷路径契约错误：{image_path}")
            declared_part_id = image_path.parent.parent.name
            parsed_part_id = part_id_from_image_path(image_path, view)
            if parsed_part_id != declared_part_id:
                raise ValueError(
                    f"visible defect identity mismatch: directory={declared_part_id}, "
                    f"filename={parsed_part_id}, path={image_path}"
                )
            selected_parts.add(declared_part_id)
    if not selected_parts:
        raise ValueError("EfficientAD可见缺陷目录没有确认缺陷物理件")

    selected_order = tuple(sorted(selected_parts))
    paths_by_view: dict[str, tuple[Path, ...]] = {}
    for view in view_order:
        crop_root = release / "crops" / view
        if not crop_root.is_dir():
            raise ValueError(f"EfficientAD完整ROI目录不存在：{crop_root}")
        matching: dict[str, list[Path]] = {part_id: [] for part_id in selected_order}
        for image_path in sorted(crop_root.glob("*.png")):
            if not image_path.is_file():
                continue
            part_id = part_id_from_image_path(image_path, view)
            if part_id in matching:
                matching[part_id].append(image_path)
        invalid = {part_id: len(paths) for part_id, paths in matching.items() if len(paths) != 1}
        if invalid:
            raise ValueError(
                f"每个视图必须为每个确认缺陷件提供exactly one crop for every selected defect part：{view} {invalid}"
            )
        paths_by_view[view] = tuple(matching[part_id][0] for part_id in selected_order)
    return paths_by_view


def analyze(args: argparse.Namespace) -> dict[str, object]:
    """Run the eight-view held-out score analysis and publish report files."""
    if not math.isfinite(float(args.threshold)):
        raise ValueError("--threshold必须是有限数值")
    if args.examples_per_label <= 0:
        raise ValueError("--examples-per-label必须为正整数")
    views = tuple(args.views)
    if views != VIEW_ORDER:
        raise ValueError("--views必须是完整且有序的BMW八视图")

    import torch

    from anomalib.engine import Engine
    from anomalib.models import EfficientAd

    training_release = Path(args.training_release).expanduser().resolve()
    defect_paths_by_view = _complete_defect_crop_paths(training_release, views)
    defect_part_ids = tuple(part_id_from_image_path(path, views[0]) for path in defect_paths_by_view[views[0]])
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")

    all_records: list[EfficientAdScore] = []
    counts: dict[str, dict[str, int]] = {}
    for view in views:
        checkpoint = Path(args.model_root).expanduser().resolve() / view / "model.ckpt"
        view_data = training_release / "efficientad" / view
        if not checkpoint.is_file():
            raise ValueError(f"EfficientAD模型不存在：{checkpoint}")
        normal_directory = view_data / "normal_test"
        if not normal_directory.is_dir() or not tuple(normal_directory.rglob("*.png")):
            raise ValueError(f"EfficientAD测试目录为空：{normal_directory}")

        model = EfficientAd.load_from_checkpoint(
            checkpoint,
            map_location="cpu",
            weights_only=False,
            visualizer=False,
        )
        engine = Engine(logger=False)
        view_records: list[EfficientAdScore] = []
        view_maps: dict[Path, np.ndarray] = {}
        records, maps = _predictions_for_directory(
            engine,
            model,
            normal_directory,
            view=view,
            label="normal",
        )
        view_records.extend(records)
        view_maps.update(maps)
        records, maps = _predictions_for_paths(
            engine,
            model,
            defect_paths_by_view[view],
            view=view,
            label="defect",
        )
        view_records.extend(records)
        view_maps.update(maps)
        render_example_sheet(
            output_dir / "examples" / f"{view}.png",
            view_records,
            view_maps,
            threshold=args.threshold,
            per_label=args.examples_per_label,
        )
        metrics = score_metrics(view_records)
        counts[view] = {
            "normal": sum(item.label == "normal" for item in view_records),
            "defect": sum(item.label == "defect" for item in view_records),
            **metrics,
        }
        all_records.extend(view_records)
        del engine, model
        gc.collect()
        torch.cuda.empty_cache()

    csv_path = write_score_csv(output_dir / "efficientad_scores.csv", all_records, threshold=args.threshold)
    distribution_path = render_score_distributions(
        output_dir / "efficientad_score_distributions.png",
        all_records,
        views=views,
        threshold=args.threshold,
    )
    report = {
        "status": "complete",
        "threshold": args.threshold,
        "score_count": len(all_records),
        "views": list(views),
        "complete_eight_view": True,
        "defect_part_ids": list(defect_part_ids),
        "defect_part_count": len(defect_part_ids),
        "defect_input_policy": "visible_defect_part_union_completed_from_crops",
        "defect_identity_source": "union_of_per_view_visible_defect_directories",
        "defect_completion_source": "crops_per_view_exact_part_id_match",
        "counts": counts,
        "scores_csv": str(csv_path),
        "distribution_plot": str(distribution_path),
        "example_dir": str(output_dir / "examples"),
        "heatmap_scale": [0.0, 1.0],
        "metric_source": "recomputed_from_runtime_predict_outputs",
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**report, "report": str(report_path)}


def main(argv: list[str] | None = None) -> int:
    """Run score analysis and print the published artifact map."""
    args = build_parser().parse_args(argv)
    try:
        report = analyze(args)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"BMW EfficientAD分数可视化失败：{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
