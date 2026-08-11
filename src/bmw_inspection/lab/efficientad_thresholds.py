"""Deterministic whole-part threshold fitting for BMW EfficientAD scores."""

from __future__ import annotations

import csv
import hashlib
import io
import itertools
import math
import re
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

_SAMPLE_INDEX = re.compile(r"^(?P<part_id>.+)_[0-9]{6}$")
_MIN_ACTIVE_THRESHOLD = 1e-3


@dataclass(frozen=True, slots=True)
class PartScore:
    """One view score bound to a physical BMW part."""

    part_id: str
    view_id: str
    label: str
    score: float
    image_path: Path

    def __post_init__(self) -> None:
        """Reject incomplete or unsafe score rows."""
        if not isinstance(self.part_id, str) or not self.part_id.strip():
            raise ValueError("part_id must be non-empty")
        if not isinstance(self.view_id, str) or not self.view_id.strip():
            raise ValueError("view_id must be non-empty")
        if self.label not in {"normal", "defect"}:
            raise ValueError("label must be normal or defect")
        if isinstance(self.score, bool) or not isinstance(self.score, Real):
            raise TypeError("score must be a real number")
        if not math.isfinite(float(self.score)):
            raise ValueError("score must be finite")
        object.__setattr__(self, "score", float(self.score))
        if not isinstance(self.image_path, Path):
            raise TypeError("image_path must be a Path")


@dataclass(frozen=True, slots=True)
class PartThresholdEvaluation:
    """Whole-part results for one per-view threshold mapping."""

    normal_part_count: int
    normal_false_positive_count: int
    observed_normal_part_fpr: float
    defect_part_count: int
    defect_detected_count: int
    defect_image_hits: int


@dataclass(frozen=True, slots=True)
class PartThresholdFit:
    """Selected thresholds and their score-set evaluation."""

    thresholds: Mapping[str, float]
    target_part_fpr: float
    allowed_normal_false_positive_count: int
    normal_part_count: int
    normal_false_positive_count: int
    observed_normal_part_fpr: float
    defect_part_count: int
    defect_detected_count: int
    defect_image_hits: int
    demo_only: bool = True
    test_used_for_selection: bool = True

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-safe calibration contract."""
        return {
            "thresholds": dict(self.thresholds),
            "target_part_fpr": self.target_part_fpr,
            "allowed_normal_false_positive_count": self.allowed_normal_false_positive_count,
            "normal_part_count": self.normal_part_count,
            "normal_false_positive_count": self.normal_false_positive_count,
            "observed_normal_part_fpr": self.observed_normal_part_fpr,
            "defect_part_count": self.defect_part_count,
            "defect_detected_count": self.defect_detected_count,
            "defect_image_hits": self.defect_image_hits,
            "demo_only": self.demo_only,
            "test_used_for_selection": self.test_used_for_selection,
        }


def part_id_from_image_path(image_path: Path, view_id: str) -> str:
    """Recover the physical-part identity from a frozen score-image filename."""
    pieces = image_path.stem.rsplit("__", 2)
    if len(pieces) == 3 and pieces[-1] == view_id:
        match = _SAMPLE_INDEX.fullmatch(pieces[-2])
        if match:
            return match.group("part_id")
    raise ValueError(f"cannot derive physical part identity from image_path: {image_path}")


def _source_id_from_explicit_score_path(image_path: Path, view_id: str) -> str:
    """Extract source release only from a supported generated score-path schema."""
    path = Path(image_path)
    if len(path.parents) >= 3 and path.parent.name == view_id and path.parents[1].name == "crops":
        return path.parents[2].name
    if (
        len(path.parents) >= 6
        and path.parent.name == "images"
        and path.parents[3].name == "normal_test"
        and path.parents[4].name == view_id
        and path.parents[5].name == "efficientad"
    ):
        return path.parents[2].name
    if (
        len(path.parents) >= 6
        and path.parent.name == "images"
        and path.parents[2].name == "normal_test"
        and path.parents[3].name == view_id
        and path.parents[4].name == "efficientad"
    ):
        return path.parents[5].name
    raise ValueError(f"cannot derive source release from explicit score image_path: {image_path}")


def _part_scores_from_bytes(content: bytes) -> tuple[PartScore, ...]:
    """Parse an immutable score CSV snapshot."""
    required = {"view_id", "label", "score", "image_path"}
    with io.StringIO(content.decode("utf-8"), newline="") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"score CSV must contain fields: {sorted(required)}")
        rows: list[PartScore] = []
        has_explicit_identity = "part_id" in (reader.fieldnames or ())
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"score CSV row {row_number} contains extra cells")
            view_id = (row.get("view_id") or "").strip()
            label = (row.get("label") or "").strip()
            raw_path = (row.get("image_path") or "").strip()
            explicit_part_id = (row.get("part_id") or "").strip()
            if not raw_path:
                raise ValueError(f"score CSV row {row_number} has empty image_path")
            image_path = Path(raw_path)
            try:
                score = float(row.get("score") or "")
            except ValueError as error:
                raise ValueError(f"score CSV row {row_number} has invalid score") from error
            derived_part_id = part_id_from_image_path(image_path, view_id)
            if has_explicit_identity:
                pieces = explicit_part_id.split("::")
                try:
                    source_id = _source_id_from_explicit_score_path(image_path, view_id)
                except ValueError as error:
                    raise ValueError(f"score CSV row {row_number} has invalid explicit part_id") from error
                if (
                    len(pieces) != 2
                    or not all(pieces)
                    or pieces[1] != derived_part_id
                    or pieces[0] != source_id
                ):
                    raise ValueError(f"score CSV row {row_number} has invalid explicit part_id")
            rows.append(
                PartScore(
                    part_id=explicit_part_id or derived_part_id,
                    view_id=view_id,
                    label=label,
                    score=score,
                    image_path=image_path,
                )
            )
    if not rows:
        raise ValueError("score CSV must contain at least one row")
    return tuple(rows)


def read_part_scores_csv_snapshot(path: Path) -> tuple[tuple[PartScore, ...], str]:
    """Read score rows and SHA256 from one immutable byte snapshot."""
    content = Path(path).read_bytes()
    return _part_scores_from_bytes(content), hashlib.sha256(content).hexdigest()


def read_part_scores_csv(path: Path) -> tuple[PartScore, ...]:
    """Read score analysis CSV rows and recover their physical-part identities."""
    rows, _sha256 = read_part_scores_csv_snapshot(path)
    return rows


def _validated_views(views: Sequence[str]) -> tuple[str, ...]:
    view_order = tuple(views)
    if not view_order or any(not isinstance(view, str) or not view.strip() for view in view_order):
        raise ValueError("views must be non-empty strings")
    if len(set(view_order)) != len(view_order):
        raise ValueError("views must be unique")
    return view_order


def _group_parts(rows: Sequence[PartScore], views: Sequence[str]) -> dict[str, tuple[PartScore, ...]]:
    view_order = _validated_views(views)
    expected_views = set(view_order)
    grouped: dict[str, list[PartScore]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, PartScore):
            raise TypeError("rows must contain PartScore values")
        if row.view_id not in expected_views:
            raise ValueError(f"part {row.part_id} contains unexpected view: {row.view_id}")
        grouped[row.part_id].append(row)
    if not grouped:
        raise ValueError("rows must contain at least one physical part")
    ordered: dict[str, tuple[PartScore, ...]] = {}
    view_index = {view: index for index, view in enumerate(view_order)}
    for part_id in sorted(grouped):
        part_rows = grouped[part_id]
        if len({row.label for row in part_rows}) != 1:
            raise ValueError(f"physical part {part_id} must have one label")
        actual_views = {row.view_id for row in part_rows}
        if len(actual_views) != len(part_rows):
            raise ValueError(f"physical part {part_id} must not repeat a requested view")
        if actual_views != expected_views:
            raise ValueError(f"physical part {part_id} must contain exactly one row for each requested view")
        if len({row.image_path for row in part_rows}) != len(part_rows):
            raise ValueError(f"physical part {part_id} must have distinct image paths")
        ordered[part_id] = tuple(sorted(part_rows, key=lambda row: view_index[row.view_id]))
    return ordered


def _validated_thresholds(thresholds: Mapping[str, float]) -> tuple[tuple[str, ...], dict[str, float]]:
    views = _validated_views(tuple(thresholds))
    values: dict[str, float] = {}
    for view in views:
        value = float(thresholds[view])
        if not math.isfinite(value):
            raise ValueError(f"threshold for {view} must be finite")
        if 0 <= value < _MIN_ACTIVE_THRESHOLD:
            raise ValueError(f"threshold for {view} is below the minimum active threshold")
        values[view] = value
    return views, values


def evaluate_part_thresholds(
    rows: Sequence[PartScore],
    thresholds: Mapping[str, float],
) -> PartThresholdEvaluation:
    """Evaluate the any-view-NG rule at physical-part granularity."""
    views, threshold_values = _validated_thresholds(thresholds)
    parts = _group_parts(rows, views)
    normal_count = 0
    normal_false_positives = 0
    defect_count = 0
    defect_detected = 0
    defect_image_hits = 0
    for part_rows in parts.values():
        hits = sum(row.score >= threshold_values[row.view_id] for row in part_rows)
        if part_rows[0].label == "normal":
            normal_count += 1
            normal_false_positives += hits > 0
        else:
            defect_count += 1
            defect_detected += hits > 0
            defect_image_hits += hits
    if normal_count == 0:
        raise ValueError("threshold evaluation requires at least one normal part")
    return PartThresholdEvaluation(
        normal_part_count=normal_count,
        normal_false_positive_count=normal_false_positives,
        observed_normal_part_fpr=normal_false_positives / normal_count,
        defect_part_count=defect_count,
        defect_detected_count=defect_detected,
        defect_image_hits=defect_image_hits,
    )


def _allowed_part_sets(normal_part_ids: Sequence[str], maximum_size: int) -> Iterator[tuple[str, ...]]:
    for size in range(maximum_size + 1):
        yield from itertools.combinations(normal_part_ids, size)


def fit_part_thresholds(
    rows: Sequence[PartScore],
    views: Sequence[str],
    target_part_fpr: float,
) -> PartThresholdFit:
    """Jointly fit per-view thresholds under a whole-part normal-FPR budget."""
    view_order = _validated_views(views)
    target = float(target_part_fpr)
    if not math.isfinite(target) or not 0.0 <= target <= 1.0:
        raise ValueError("target_part_fpr must be finite and between zero and one")
    parts = _group_parts(rows, view_order)
    normal_part_ids = tuple(part_id for part_id, part_rows in parts.items() if part_rows[0].label == "normal")
    if not normal_part_ids:
        raise ValueError("threshold fitting requires at least one normal part")
    normal_rows = tuple(row for part_id in normal_part_ids for row in parts[part_id])
    allowed_count = math.floor(target * len(normal_part_ids) + 1e-12)

    candidates: list[tuple[PartThresholdEvaluation, dict[str, float], tuple[str, ...]]] = []
    for allowed_tuple in _allowed_part_sets(normal_part_ids, allowed_count):
        allowed = set(allowed_tuple)
        thresholds: dict[str, float] = {}
        for view in view_order:
            protected_scores = [row.score for row in normal_rows if row.view_id == view and row.part_id not in allowed]
            if protected_scores:
                threshold = math.nextafter(max(protected_scores), math.inf)
            else:
                all_view_scores = [row.score for row in normal_rows if row.view_id == view]
                threshold = math.nextafter(min(all_view_scores), -math.inf)
            if not math.isfinite(threshold):
                raise ValueError(f"cannot derive a finite threshold for view {view}")
            if 0 <= threshold < _MIN_ACTIVE_THRESHOLD:
                threshold = _MIN_ACTIVE_THRESHOLD
            thresholds[view] = threshold
        evaluation = evaluate_part_thresholds(rows, thresholds)
        if evaluation.normal_false_positive_count <= allowed_count:
            candidates.append((evaluation, thresholds, allowed_tuple))
    if not candidates:
        raise RuntimeError("no threshold candidate satisfies the requested normal part FPR")

    evaluation, thresholds, _allowed = max(
        candidates,
        key=lambda candidate: (
            candidate[0].defect_detected_count,
            candidate[0].defect_image_hits,
            -candidate[0].normal_false_positive_count,
            tuple(candidate[1][view] for view in view_order),
            candidate[2],
        ),
    )
    return PartThresholdFit(
        thresholds=thresholds,
        target_part_fpr=target,
        allowed_normal_false_positive_count=allowed_count,
        normal_part_count=evaluation.normal_part_count,
        normal_false_positive_count=evaluation.normal_false_positive_count,
        observed_normal_part_fpr=evaluation.observed_normal_part_fpr,
        defect_part_count=evaluation.defect_part_count,
        defect_detected_count=evaluation.defect_detected_count,
        defect_image_hits=evaluation.defect_image_hits,
    )


__all__ = [
    "PartScore",
    "PartThresholdEvaluation",
    "PartThresholdFit",
    "evaluate_part_thresholds",
    "fit_part_thresholds",
    "part_id_from_image_path",
    "read_part_scores_csv",
    "read_part_scores_csv_snapshot",
]
