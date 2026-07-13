# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fit versioned dual thresholds and report physical-part fusion metrics."""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from io import StringIO
from math import ceil, isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

GROUP_FIELDS = ("hand", "view", "branch", "model_version", "roi_version")
ZS32_REQUIRED_VIEWS = ("front", "front_left", "front_right", "back", "back_left", "back_right")
THRESHOLD_FIELDS = (
    *GROUP_FIELDS,
    "low_threshold",
    "high_threshold",
    "normal_count",
    "defect_count",
    "status",
)
GroupKey = tuple[str, str, str, str, str]


@dataclass(frozen=True)
class ThresholdRecord:
    """Immutable dual thresholds for one versioned branch and view group."""

    hand: str
    view: str
    branch: str
    model_version: str
    roi_version: str
    low_threshold: float | None
    high_threshold: float | None
    normal_count: int
    defect_count: int
    status: str


@dataclass(frozen=True)
class _CalibrationRow:
    """Validated internal representation of one input CSV row."""

    part_id: str
    hand: str
    view: str
    branch: str
    raw_score: float
    gt_label: int
    split: str
    model_version: str
    roi_version: str

    @property
    def group_key(self) -> GroupKey:
        """Versioned threshold group key."""
        return (self.hand, self.view, self.branch, self.model_version, self.roi_version)


def _required_text(row: Mapping[str, Any], field: str, row_number: int) -> str:
    """Return a required, stripped field value."""
    value = row.get(field)
    text = "" if value is None else str(value).strip()
    if not text:
        msg = f"row {row_number} has empty required field: {field}"
        raise ValueError(msg)
    return text


def _parse_rows(rows: Iterable[Mapping[str, Any]]) -> list[_CalibrationRow]:
    """Validate calibration rows and enforce physical-part split consistency."""
    parsed: list[_CalibrationRow] = []
    split_by_part: dict[str, str] = {}
    label_by_part: dict[str, int] = {}
    hand_by_part: dict[str, str] = {}
    for row_number, row in enumerate(rows, start=2):
        part_id = _required_text(row, "part_id", row_number)
        split = _required_text(row, "split", row_number)
        previous_split = split_by_part.setdefault(part_id, split)
        if previous_split != split:
            msg = f"physical part {part_id!r} appears in multiple split values: {previous_split!r}, {split!r}"
            raise ValueError(msg)
        label_text = _required_text(row, "gt_label", row_number)
        if label_text not in {"0", "1"}:
            msg = f"row {row_number} gt_label must be 0 or 1, got {label_text!r}"
            raise ValueError(msg)
        gt_label = int(label_text)
        previous_label = label_by_part.setdefault(part_id, gt_label)
        if previous_label != gt_label:
            msg = f"physical part {part_id!r} has inconsistent gt_label values"
            raise ValueError(msg)
        hand = _required_text(row, "hand", row_number)
        previous_hand = hand_by_part.setdefault(part_id, hand)
        if previous_hand != hand:
            msg = f"physical part {part_id!r} has inconsistent hand values"
            raise ValueError(msg)
        try:
            raw_score = float(_required_text(row, "raw_score", row_number))
        except ValueError as exc:
            msg = f"row {row_number} raw_score must be numeric"
            raise ValueError(msg) from exc
        if not isfinite(raw_score):
            msg = f"row {row_number} raw_score must be finite"
            raise ValueError(msg)
        parsed.append(
            _CalibrationRow(
                part_id=part_id,
                hand=hand,
                view=_required_text(row, "view", row_number),
                branch=_required_text(row, "branch", row_number),
                raw_score=raw_score,
                gt_label=gt_label,
                split=split,
                model_version=_required_text(row, "model_version", row_number),
                roi_version=_required_text(row, "roi_version", row_number),
            ),
        )
    if not parsed:
        msg = "calibration rows must not be empty"
        raise ValueError(msg)
    return parsed


def _validate_rate(value: float, name: str) -> None:
    """Require a finite probability in the interval ``(0, 1]``."""
    if not isfinite(value) or not 0 < value <= 1:
        msg = f"{name} must be in (0, 1], got {value!r}"
        raise ValueError(msg)


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    """Return an inclusive nearest-rank quantile without numerical dependencies."""
    ordered = sorted(values)
    rank = max(1, ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def _low_threshold(defect_scores: Sequence[float], target_recall: float) -> float:
    """Return the largest observed threshold whose inclusive recall meets the target."""
    ordered = sorted(defect_scores)
    required_hits = ceil(target_recall * len(ordered))
    return ordered[len(ordered) - required_hits]


def _normalize_group_key(value: Sequence[str]) -> GroupKey:
    """Validate and normalize one exact versioned required-group key."""
    if len(value) != len(GROUP_FIELDS):
        msg = f"required group must have {len(GROUP_FIELDS)} values: {value!r}"
        raise ValueError(msg)
    items = tuple(str(item).strip() for item in value)
    if any(not item for item in items):
        msg = f"required group values must not be empty: {value!r}"
        raise ValueError(msg)
    hand, view, branch, model_version, roi_version = items
    return hand, view, branch, model_version, roi_version


def fit_dual_thresholds(
    rows: Iterable[Mapping[str, Any]],
    *,
    target_recall: float,
    normal_quantile: float,
    fit_split: str = "calibration",
    required_views: Sequence[str] = (),
    required_groups: Sequence[GroupKey] = (),
) -> list[ThresholdRecord]:
    """Fit versioned grouped thresholds from independent normal and defect calibration rows.

    Args:
        rows (Iterable[Mapping[str, Any]]): Calibration records using the documented CSV schema.
        target_recall (float): Minimum observed defect recall required for ``T_low``.
        normal_quantile (float): Inclusive nearest-rank normal quantile used for ``T_high``.
        fit_split (str): Split whose rows are allowed to influence fitted thresholds.
        required_views (Sequence[str]): Views that must exist in the selected fit split.
        required_groups (Sequence[GroupKey]): Exact versioned groups that require explicit threshold records.

    Returns:
        list[ThresholdRecord]: Deterministically sorted threshold or insufficient-data records.

    Raises:
        ValueError: If probability arguments or calibration rows are invalid.
    """
    _validate_rate(target_recall, "target_recall")
    _validate_rate(normal_quantile, "normal_quantile")
    all_rows = _parse_rows(rows)
    parsed = [row for row in all_rows if row.split == fit_split]
    if not parsed:
        msg = f"fit split {fit_split!r} is missing from calibration rows"
        raise ValueError(msg)
    grouped: dict[GroupKey, list[_CalibrationRow]] = defaultdict(list)
    for row in parsed:
        grouped[row.group_key].append(row)
    required = {view.strip() for view in required_views if view.strip()}
    missing_views = sorted(required - {row.view for row in parsed})
    if missing_views:
        msg = f"required view is missing from fit split {fit_split!r}: {', '.join(missing_views)}"
        raise ValueError(msg)
    for key in (_normalize_group_key(value) for value in required_groups):
        grouped.setdefault(key, [])

    thresholds: list[ThresholdRecord] = []
    for key in sorted(grouped):
        group_rows = grouped[key]
        normal_scores = [row.raw_score for row in group_rows if row.gt_label == 0]
        defect_scores = [row.raw_score for row in group_rows if row.gt_label == 1]
        if not normal_scores or not defect_scores:
            low_threshold = None
            high_threshold = None
            status = "insufficient_data"
        else:
            low_threshold = _low_threshold(defect_scores, target_recall)
            high_threshold = max(low_threshold, _nearest_rank(normal_scores, normal_quantile))
            status = "ok"
        thresholds.append(
            ThresholdRecord(
                hand=key[0],
                view=key[1],
                branch=key[2],
                model_version=key[3],
                roi_version=key[4],
                low_threshold=low_threshold,
                high_threshold=high_threshold,
                normal_count=len(normal_scores),
                defect_count=len(defect_scores),
                status=status,
            ),
        )
    return thresholds


def _threshold_record(value: ThresholdRecord | Mapping[str, Any]) -> ThresholdRecord:
    """Normalize a threshold record or serialized mapping."""
    record = (
        value
        if isinstance(value, ThresholdRecord)
        else ThresholdRecord(
            hand=str(value["hand"]),
            view=str(value["view"]),
            branch=str(value["branch"]),
            model_version=str(value["model_version"]),
            roi_version=str(value["roi_version"]),
            low_threshold=None if value.get("low_threshold") is None else float(value["low_threshold"]),
            high_threshold=None if value.get("high_threshold") is None else float(value["high_threshold"]),
            normal_count=int(value["normal_count"]),
            defect_count=int(value["defect_count"]),
            status=str(value["status"]),
        )
    )
    thresholds = (record.low_threshold, record.high_threshold)
    low, high = thresholds
    valid_ok = record.status == "ok" and low is not None and high is not None
    valid_ok = valid_ok and isfinite(low) and isfinite(high) and low <= high
    valid_insufficient = record.status == "insufficient_data" and thresholds == (None, None)
    if not valid_ok and not valid_insufficient:
        msg = f"invalid threshold record for {(record.hand, record.view, record.branch)}"
        raise ValueError(msg)
    if record.normal_count < 0 or record.defect_count < 0:
        msg = "threshold sample counts must not be negative"
        raise ValueError(msg)
    return record


def _metric_block(
    outcomes: Mapping[str, tuple[int, str]],
    *,
    calibration_valid: bool,
) -> dict[str, int | float | bool | None]:
    """Compute physical-part counts and rates for a set of classified outcomes."""
    defect_outcomes = [outcome for label, outcome in outcomes.values() if label == 1]
    normal_outcomes = [outcome for label, outcome in outcomes.values() if label == 0]
    escape_count = defect_outcomes.count("CLEAR")
    review_count = sum(outcome == "GRAY" for _label, outcome in outcomes.values())
    reject_count = normal_outcomes.count("STRONG")
    defect_count = len(defect_outcomes)
    normal_count = len(normal_outcomes)
    part_count = len(outcomes)
    escape_rate = escape_count / defect_count if defect_count else None
    non_clear_recall = None if escape_rate is None or not calibration_valid else 1 - escape_rate
    return {
        "calibration_valid": calibration_valid,
        "part_count": part_count,
        "defect_part_count": defect_count,
        "normal_part_count": normal_count,
        "escape_count": escape_count,
        "escape_rate": escape_rate,
        "non_clear_recall": non_clear_recall,
        "recall": non_clear_recall,
        "normal_reject_count": reject_count,
        "normal_reject_rate": reject_count / normal_count if normal_count else None,
        "review_count": review_count,
        "review_rate": review_count / part_count if part_count else None,
        "escape_rate_95_upper": (
            1 - 0.05 ** (1 / defect_count) if calibration_valid and defect_count and escape_count == 0 else None
        ),
    }


def _row_level(row: _CalibrationRow, threshold: ThresholdRecord | None) -> tuple[str, bool]:
    """Classify one score and report whether its threshold contract is valid."""
    if (
        threshold is None
        or threshold.status != "ok"
        or threshold.low_threshold is None
        or threshold.high_threshold is None
    ):
        return "GRAY", False
    if row.raw_score >= threshold.high_threshold:
        return "STRONG", True
    if row.raw_score >= threshold.low_threshold:
        return "GRAY", True
    return "CLEAR", True


def _classify_rows(
    rows: Sequence[_CalibrationRow],
    threshold_by_key: Mapping[GroupKey, ThresholdRecord],
    required_groups: Sequence[GroupKey],
) -> tuple[dict[str, tuple[int, str]], bool]:
    """Apply fail-closed OR/GRAY semantics and return one outcome per part."""
    levels_by_part: dict[str, list[str]] = defaultdict(list)
    labels: dict[str, int] = {}
    hands: dict[str, str] = {}
    observed_keys_by_part: dict[str, set[GroupKey]] = defaultdict(set)
    calibration_valid = True
    for row in rows:
        labels[row.part_id] = row.gt_label
        hands[row.part_id] = row.hand
        observed_keys_by_part[row.part_id].add(row.group_key)
        level, row_valid = _row_level(row, threshold_by_key.get(row.group_key))
        levels_by_part[row.part_id].append(level)
        calibration_valid &= row_valid
    for part_id, hand in hands.items():
        for key in required_groups:
            if key[0] != hand:
                continue
            threshold = threshold_by_key.get(key)
            if threshold is None or threshold.status != "ok" or key not in observed_keys_by_part[part_id]:
                levels_by_part[part_id].append("GRAY")
                calibration_valid = False
    outcomes: dict[str, tuple[int, str]] = {}
    for part_id, levels in levels_by_part.items():
        outcome = "STRONG" if "STRONG" in levels else "GRAY" if "GRAY" in levels else "CLEAR"
        outcomes[part_id] = (labels[part_id], outcome)
    return outcomes, calibration_valid


def part_level_metrics(
    rows: Iterable[Mapping[str, Any]],
    thresholds: Iterable[ThresholdRecord | Mapping[str, Any]],
    *,
    eval_split: str = "test",
    required_groups: Sequence[GroupKey] = (),
) -> dict[str, Any]:
    """Evaluate dual thresholds with physical-part OR/GRAY semantics.

    Args:
        rows (Iterable[Mapping[str, Any]]): Evaluation records using the calibration CSV schema.
        thresholds (Iterable[ThresholdRecord | Mapping[str, Any]]): Versioned thresholds to apply.
        eval_split (str): Held-out split whose physical parts are evaluated.
        required_groups (Sequence[GroupKey]): Exact groups required for each matching-hand evaluation part.

    Returns:
        dict[str, Any]: Overall and group-level physical-part metrics.

    Raises:
        ValueError: If rows are invalid, the evaluation split is absent, or threshold keys are duplicated.
    """
    all_rows = _parse_rows(rows)
    parsed = [row for row in all_rows if row.split == eval_split]
    if not parsed:
        msg = f"evaluation split {eval_split!r} is missing from calibration rows"
        raise ValueError(msg)
    required_keys = tuple(sorted({_normalize_group_key(value) for value in required_groups}))
    threshold_records = [_threshold_record(value) for value in thresholds]
    threshold_by_key: dict[GroupKey, ThresholdRecord] = {}
    for threshold in threshold_records:
        key = (threshold.hand, threshold.view, threshold.branch, threshold.model_version, threshold.roi_version)
        if key in threshold_by_key:
            msg = f"duplicate threshold group: {key}"
            raise ValueError(msg)
        threshold_by_key[key] = threshold
    overall_outcomes, overall_valid = _classify_rows(parsed, threshold_by_key, required_keys)
    overall = _metric_block(overall_outcomes, calibration_valid=overall_valid)
    grouped = []
    all_part_ids_by_hand: dict[str, set[str]] = defaultdict(set)
    labels = {row.part_id: row.gt_label for row in parsed}
    for row in parsed:
        all_part_ids_by_hand[row.hand].add(row.part_id)
    for key in sorted({row.group_key for row in parsed} | set(threshold_by_key) | set(required_keys)):
        rows_by_part: dict[str, list[_CalibrationRow]] = defaultdict(list)
        for row in parsed:
            if row.group_key == key:
                rows_by_part[row.part_id].append(row)
        expected_parts = all_part_ids_by_hand[key[0]] if key in required_keys else set(rows_by_part)
        outcomes: dict[str, tuple[int, str]] = {}
        group_valid = True
        threshold = threshold_by_key.get(key)
        for part_id in expected_parts:
            part_rows = rows_by_part.get(part_id, [])
            if not part_rows:
                outcomes[part_id] = (labels[part_id], "GRAY")
                group_valid = False
                continue
            levels = []
            for row in part_rows:
                level, row_valid = _row_level(row, threshold)
                levels.append(level)
                group_valid &= row_valid
            outcome = "STRONG" if "STRONG" in levels else "GRAY" if "GRAY" in levels else "CLEAR"
            outcomes[part_id] = (labels[part_id], outcome)
        if key in required_keys and (threshold is None or threshold.status != "ok"):
            group_valid = False
        metrics = _metric_block(outcomes, calibration_valid=group_valid)
        grouped.append({**dict(zip(GROUP_FIELDS, key, strict=True)), **metrics})
    return {"overall": overall, "groups": grouped}


def load_calibration_rows(input_csv: Path) -> list[dict[str, str]]:
    """Load calibration CSV rows without changing their declared schema."""
    with input_csv.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _publish_report_directory(output_dir: Path, reports: Mapping[str, str]) -> None:
    """Stage a complete immutable report snapshot and publish it with one directory rename."""
    if output_dir.exists():
        msg = f"output directory already exists: {output_dir}"
        raise FileExistsError(msg)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        for name, content in reports.items():
            (temporary_dir / name).write_text(content, encoding="utf-8")
        temporary_dir.replace(output_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def _threshold_csv(thresholds: Sequence[ThresholdRecord]) -> str:
    """Serialize threshold records into deterministic CSV text."""
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=THRESHOLD_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(asdict(threshold) for threshold in thresholds)
    return output.getvalue()


def _summary(
    rows: Sequence[Mapping[str, Any]],
    thresholds: Sequence[ThresholdRecord],
    metrics: Mapping[str, Any],
    *,
    target_recall: float,
    normal_quantile: float,
    fit_split: str,
    eval_split: str,
    missing_fit_views: Sequence[str],
    missing_evaluation_views: Sequence[str],
    invalid_threshold_groups: Sequence[GroupKey],
) -> str:
    """Build the human-readable calibration summary."""
    part_ids = {str(row["part_id"]).strip() for row in rows}
    normal_parts = {str(row["part_id"]).strip() for row in rows if str(row["gt_label"]).strip() == "0"}
    defect_parts = part_ids - normal_parts
    insufficient_count = sum(threshold.status == "insufficient_data" for threshold in thresholds)
    overall = metrics["overall"]
    return "\n".join(
        [
            "# ZS32 Fusion Calibration Summary",
            "",
            f"- CSV rows: `{len(rows)}`",
            f"- physical parts: `{len(part_ids)}`",
            f"- normal parts: `{len(normal_parts)}`",
            f"- defect parts: `{len(defect_parts)}`",
            f"- threshold groups: `{len(thresholds)}`",
            f"- insufficient-data groups: `{insufficient_count}`",
            f"- target recall: `{target_recall:g}`",
            f"- normal quantile: `{normal_quantile:g}`",
            f"- fit split: `{fit_split}`",
            f"- evaluation split: `{eval_split}`",
            f"- missing fit views: `{list(missing_fit_views)}`",
            f"- missing evaluation views: `{list(missing_evaluation_views)}`",
            f"- invalid threshold groups: `{[list(key) for key in invalid_threshold_groups]}`",
            f"- calibration valid: `{overall['calibration_valid']}`",
            f"- observed part escape count: `{overall['escape_count']}`",
            f"- observed part non-CLEAR recall: `{overall['non_clear_recall']}`",
            f"- zero-escape 95% upper bound: `{overall['escape_rate_95_upper']}`",
            "",
            "Observed 100% recall is not proof of zero production escapes; always interpret it with the ",
            "defect-part count and the one-sided 95% upper bound.",
            "",
        ],
    )


def run_calibration(
    input_csv: Path,
    output_dir: Path,
    *,
    target_recall: float = 1.0,
    normal_quantile: float = 0.995,
    fit_split: str = "calibration",
    eval_split: str = "test",
    required_views: Sequence[str] | None = None,
    required_groups: Sequence[GroupKey] = (),
) -> dict[str, Any]:
    """Fit thresholds, evaluate parts, and atomically publish offline reports.

    Args:
        input_csv (Path): Calibration CSV path.
        output_dir (Path): Directory for immutable calibration artifacts.
        target_recall (float): Minimum observed recall used to select ``T_low``.
        normal_quantile (float): Normal-score quantile used to select ``T_high``.
        fit_split (str): Split used exclusively to fit thresholds.
        eval_split (str): Held-out split used exclusively for reported metrics.
        required_views (Sequence[str] | None): Views required in both splits; defaults to all six ZS32 views.
        required_groups (Sequence[GroupKey]): Exact versioned groups required for fit and evaluation.

    Returns:
        dict[str, Any]: Metrics payload written to ``calibration_metrics.json``.
    """
    if fit_split == eval_split:
        msg = "fit split and evaluation split must be different"
        raise ValueError(msg)
    rows = load_calibration_rows(input_csv)
    effective_required_views = tuple(required_views) if required_views is not None else ZS32_REQUIRED_VIEWS
    fit_views = {str(row.get("view", "")).strip() for row in rows if str(row.get("split", "")).strip() == fit_split}
    evaluation_views = {
        str(row.get("view", "")).strip() for row in rows if str(row.get("split", "")).strip() == eval_split
    }
    missing_fit_views = sorted(set(effective_required_views) - fit_views)
    missing_evaluation_views = sorted(set(effective_required_views) - evaluation_views)
    thresholds = fit_dual_thresholds(
        rows,
        target_recall=target_recall,
        normal_quantile=normal_quantile,
        fit_split=fit_split,
        required_views=(),
        required_groups=required_groups,
    )
    threshold_groups = tuple(
        (threshold.hand, threshold.view, threshold.branch, threshold.model_version, threshold.roi_version)
        for threshold in thresholds
    )
    effective_required_groups = tuple(sorted(set(required_groups) | set(threshold_groups)))
    threshold_by_key = {
        (threshold.hand, threshold.view, threshold.branch, threshold.model_version, threshold.roi_version): threshold
        for threshold in thresholds
    }
    invalid_threshold_groups = [
        key
        for key in effective_required_groups
        if (threshold := threshold_by_key.get(key)) is None
        or threshold.status != "ok"
        or threshold.low_threshold is None
        or threshold.high_threshold is None
    ]
    metrics = part_level_metrics(
        rows,
        thresholds,
        eval_split=eval_split,
        required_groups=effective_required_groups,
    )
    coverage_valid = not missing_fit_views and not missing_evaluation_views and not invalid_threshold_groups
    if not coverage_valid:
        metrics["overall"].update(
            {
                "calibration_valid": False,
                "non_clear_recall": None,
                "recall": None,
                "escape_rate_95_upper": None,
            },
        )
    fit_rows = [row for row in rows if str(row.get("split", "")).strip() == fit_split]
    eval_rows = [row for row in rows if str(row.get("split", "")).strip() == eval_split]
    metrics = {
        "fit_split": fit_split,
        "evaluation_split": eval_split,
        "fit_row_count": len(fit_rows),
        "evaluation_row_count": len(eval_rows),
        "fit_part_count": len({str(row["part_id"]).strip() for row in fit_rows}),
        "evaluation_part_count": len({str(row["part_id"]).strip() for row in eval_rows}),
        "missing_fit_views": missing_fit_views,
        "missing_evaluation_views": missing_evaluation_views,
        "invalid_threshold_groups": [list(key) for key in invalid_threshold_groups],
        **metrics,
    }
    threshold_payload = {
        "target_recall": target_recall,
        "normal_quantile": normal_quantile,
        "fit_split": fit_split,
        "evaluation_split": eval_split,
        "required_views": list(effective_required_views),
        "required_groups": [list(_normalize_group_key(value)) for value in effective_required_groups],
        "thresholds": [asdict(threshold) for threshold in thresholds],
    }
    reports = {
        "thresholds.json": json.dumps(threshold_payload, indent=2, sort_keys=True) + "\n",
        "thresholds.csv": _threshold_csv(thresholds),
        "calibration_metrics.json": json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        "calibration_summary.md": _summary(
            rows,
            thresholds,
            metrics,
            target_recall=target_recall,
            normal_quantile=normal_quantile,
            fit_split=fit_split,
            eval_split=eval_split,
            missing_fit_views=missing_fit_views,
            missing_evaluation_views=missing_evaluation_views,
            invalid_threshold_groups=invalid_threshold_groups,
        ),
    }
    _publish_report_directory(output_dir, reports)
    return metrics
