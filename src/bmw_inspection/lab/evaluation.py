"""Part-level offline evaluation and experiment-profile comparison."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol

import cv2

from bmw_inspection.lab.config import LabExperimentConfig, load_experiment_config
from bmw_inspection.lab.contracts import (
    BranchEvidence,
    BranchName,
    BranchStatus,
    CapturedView,
    CaptureSet,
    FinalStatus,
    ViewId,
)
from bmw_inspection.lab.dataset import ManifestRow, find_part_split_leakage, read_manifest
from bmw_inspection.lab.publisher import _canonical_json, _json_value
from bmw_inspection.lab.runtime import LabRuntime, RuntimeState, build_lab_runtime


_EVALUATION_SPLITS = frozenset({"calibration", "final_test"})
_SAFE_PROFILE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PREDICTION_FIELDS = (
    "profile_id",
    "part_id",
    "session_id",
    "split",
    "expected_label",
    "predicted_label",
    "final_status",
    "total_latency_ms",
    "evidence_path",
    "evidence_result_sha256",
)
_RESULT_ARCHIVE_FIELDS = frozenset(
    {
        "capture_set_id",
        "created_at",
        "final_status",
        "reason",
        "required_complete",
        "triggered_branches",
        "config_digest",
        "evidence",
    }
)
_CONFIG_SNAPSHOT_FIELDS = frozenset({"config_digest", "config"})
_CONFIG_ARCHIVE_FIELDS = frozenset(
    {
        "path",
        "experiment_id",
        "topology",
        "capture",
        "part_rois",
        "template",
        "bright_streak",
        "yolo",
        "patchcore",
        "required_for_ok",
        "result_root",
    }
)


@dataclass(frozen=True, slots=True)
class EvaluationProfileContract:
    """Config-derived identity that may authorize non-experimental system metrics."""

    profile_id: str
    config_digest: str
    required_evidence: Mapping[BranchName, frozenset[ViewId]]

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or _SAFE_PROFILE_ID.fullmatch(self.profile_id) is None:
            raise ValueError("profile contract requires a path-safe profile_id")
        if (
            not isinstance(self.config_digest, str)
            or len(self.config_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.config_digest)
        ):
            raise ValueError("profile contract requires a lowercase SHA256 config_digest")
        object.__setattr__(
            self,
            "required_evidence",
            MappingProxyType(_normalize_required_evidence(self.required_evidence, self.profile_id)),
        )


@dataclass(frozen=True, slots=True)
class EvaluationPart:
    """One physical part and its exact six manifest rows."""

    part_id: str
    session_id: str
    split: str
    expected_label: str
    rows: Mapping[ViewId, ManifestRow]

    def __post_init__(self) -> None:
        if not self.part_id or not self.session_id:
            raise ValueError("evaluation part and session identities must be non-empty")
        if self.split not in _EVALUATION_SPLITS:
            raise ValueError("evaluation split must be calibration or final_test")
        if self.expected_label not in {"normal", "defect"}:
            raise ValueError("expected_label must be normal or defect")
        if set(self.rows) != set(ViewId):
            raise ValueError("evaluation part must contain exactly the six BMW views")
        object.__setattr__(self, "rows", MappingProxyType(dict(self.rows)))


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    """Normalized boundary returned by an injected resident offline runtime."""

    final_status: FinalStatus
    evidence: tuple[BranchEvidence, ...]
    evidence_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.final_status, FinalStatus):
            raise TypeError("final_status must be FinalStatus")
        if not isinstance(self.evidence, tuple) or not all(
            isinstance(item, BranchEvidence) for item in self.evidence
        ):
            raise TypeError("evidence must be a tuple of BranchEvidence")
        evidence_path = Path(self.evidence_path).expanduser().resolve()
        if not evidence_path.is_dir():
            raise ValueError(f"evaluation evidence directory does not exist: {evidence_path}")
        object.__setattr__(self, "evidence_path", evidence_path)


class OfflineEvaluationRuntime(Protocol):
    """Resident runtime contract required by batch evaluation."""

    profile_id: str
    evidence_root: Path
    required_evidence: Mapping[BranchName, frozenset[ViewId]]

    def inspect(self, part: EvaluationPart) -> EvaluationOutcome:
        """Inspect one complete physical part without reloading models."""


def _normalize_required_evidence(
    required_evidence: Mapping[BranchName, frozenset[ViewId]],
    profile_id: str,
) -> dict[BranchName, frozenset[ViewId]]:
    required: dict[BranchName, frozenset[ViewId]] = {}
    for branch, views in required_evidence.items():
        if not isinstance(branch, BranchName):
            raise TypeError(f"required evidence for {profile_id} contains a non-BranchName key")
        normalized = frozenset(views)
        if not normalized or any(not isinstance(view, ViewId) for view in normalized):
            raise TypeError(
                f"required evidence for {profile_id}/{branch.value} must contain ViewId values"
            )
        if branch is not BranchName.BRIGHT_STREAK and normalized != frozenset(ViewId):
            raise ValueError(f"{branch.value} must declare evidence coverage for all six views")
        required[branch] = normalized
    return required


def _config_required_evidence(config: LabExperimentConfig) -> dict[BranchName, frozenset[ViewId]]:
    required: dict[BranchName, frozenset[ViewId]] = {}
    if config.template.enabled:
        required[BranchName.TEMPLATE] = frozenset(ViewId)
    if config.bright_streak.enabled_views:
        required[BranchName.BRIGHT_STREAK] = frozenset(config.bright_streak.enabled_views)
    if config.yolo.enabled:
        required[BranchName.YOLO] = frozenset(ViewId)
    if config.patchcore.enabled:
        required[BranchName.PATCHCORE] = frozenset(ViewId)
    return required


def _profile_contract(config: LabExperimentConfig) -> EvaluationProfileContract:
    payload = _json_value(config)
    digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return EvaluationProfileContract(
        config.experiment_id,
        digest,
        _config_required_evidence(config),
    )


class LabRuntimeEvaluationAdapter:
    """Load manifest images and adapt the Task 8 runtime to offline evaluation."""

    def __init__(self, config: LabExperimentConfig, runtime: LabRuntime) -> None:
        self.config = config
        self.runtime = runtime
        self.profile_id = config.experiment_id
        self.evidence_root = (config.result_root / config.experiment_id / "runs").resolve()
        self.profile_contract = _profile_contract(config)
        self.required_evidence = self.profile_contract.required_evidence
        self._serial_by_view = {
            view_id: slot.serial
            for slot in config.topology.camera_slots
            for view_id in (slot.front, slot.back)
        }

    def inspect(self, part: EvaluationPart) -> EvaluationOutcome:
        captured_at = datetime.now().astimezone()
        views: dict[ViewId, CapturedView] = {}
        for view_id, row in part.rows.items():
            image = cv2.imread(str(row.image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"cannot decode manifest image: {row.image_path}")
            views[view_id] = CapturedView(view_id, self._serial_by_view[view_id], image, captured_at)
        capture_set = CaptureSet(f"evaluation-{part.part_id}", views, captured_at)
        result = self.runtime.inspect(capture_set)
        if result.state is not RuntimeState.PUBLISHED or result.inspection is None or result.run_dir is None:
            raise RuntimeError(
                f"profile {self.profile_id} could not publish {part.part_id}: {result.reason}"
            )
        return EvaluationOutcome(result.inspection.final_status, result.inspection.evidence, result.run_dir)


def build_offline_runtime(config: LabExperimentConfig) -> LabRuntimeEvaluationAdapter:
    """Build the production offline adapter while loading every enabled model once."""
    return LabRuntimeEvaluationAdapter(
        config,
        build_lab_runtime(config, config_loader=lambda: load_experiment_config(config.path)),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_manifest_samples(rows: Sequence[ManifestRow]) -> None:
    by_sample: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in rows:
        by_sample[row.sample_id].append(row)
    for sample_id, sample_rows in sorted(by_sample.items()):
        part_ids = {row.part_id for row in sample_rows}
        if len(part_ids) != 1:
            raise ValueError(f"sample_id {sample_id} is bound to multiple physical parts")
        session_ids = {row.session_id for row in sample_rows}
        views = {row.view_id for row in sample_rows}
        if len(session_ids) != 1 or len(sample_rows) != len(ViewId) or views != set(ViewId):
            raise ValueError(f"sample_id {sample_id} must be one complete six-view capture")
        if len({row.image_path for row in sample_rows}) != len(ViewId):
            raise ValueError(f"sample_id {sample_id} requires six distinct image_path values")
        if len({row.image_sha256 for row in sample_rows}) != len(ViewId):
            raise ValueError(f"sample_id {sample_id} requires six distinct image_sha256 values")


def _parts(rows: Sequence[ManifestRow], split: str) -> tuple[EvaluationPart, ...]:
    selected = [row for row in rows if row.split == split]
    if not selected:
        raise ValueError(f"manifest contains no {split} rows")
    grouped: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in selected:
        if not row.image_path.is_file():
            raise ValueError(f"manifest image does not exist: {row.image_path}")
        if _sha256(row.image_path) != row.image_sha256:
            raise ValueError(f"manifest image_sha256 mismatch: {row.image_path}")
        grouped[row.part_id].append(row)
    parts: list[EvaluationPart] = []
    for part_id, part_rows in sorted(grouped.items()):
        sessions = {row.session_id for row in part_rows}
        if len(sessions) != 1:
            raise ValueError(f"physical part {part_id} spans multiple sessions")
        sample_ids = {row.sample_id for row in part_rows}
        if len(sample_ids) != 1:
            raise ValueError(f"physical part {part_id} rows must share one sample_id/capture identity")
        by_view = {row.view_id: row for row in part_rows}
        if len(part_rows) != len(ViewId) or set(by_view) != set(ViewId):
            raise ValueError(f"physical part {part_id} must contain exactly one row for each BMW view")
        parts.append(
            EvaluationPart(
                part_id,
                next(iter(sessions)),
                split,
                "defect" if any(row.label == "defect" for row in part_rows) else "normal",
                by_view,
            ),
        )
    return tuple(parts)


def _predicted_label(status: FinalStatus) -> str:
    if status is FinalStatus.OK:
        return "normal"
    if status in {
        FinalStatus.NG_TEMPLATE,
        FinalStatus.NG_BRIGHT_STREAK,
        FinalStatus.NG_YOLO,
        FinalStatus.NG_ANOMALY,
    }:
        return "defect"
    if status in {FinalStatus.REVIEW, FinalStatus.RETAKE}:
        return "review"
    return "error"


def _binary_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, int | float | None]:
    binary = [row for row in rows if row["predicted_label"] in {"normal", "defect"}]
    tp = sum(row["expected_label"] == "defect" and row["predicted_label"] == "defect" for row in binary)
    fp = sum(row["expected_label"] == "normal" and row["predicted_label"] == "defect" for row in binary)
    fn = sum(row["expected_label"] == "defect" and row["predicted_label"] == "normal" for row in binary)
    correct = sum(row["expected_label"] == row["predicted_label"] for row in binary)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "part_count": len(rows),
        "evaluated_binary_count": len(binary),
        "accuracy": correct / len(binary) if binary else None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))


def _latencies(values: Sequence[float]) -> dict[str, float | None]:
    return {
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
    }


def _branch_metrics(
    evidence: Sequence[tuple[str, str, BranchEvidence]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[BranchName, dict[str, list[tuple[str, BranchEvidence]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for part_id, expected_label, item in evidence:
        grouped[item.branch][part_id].append((expected_label, item))
    result: dict[str, dict[str, Any]] = {}
    for branch in sorted(grouped, key=lambda item: item.value):
        by_part = grouped[branch]
        items = [item for part_items in by_part.values() for _expected, item in part_items]
        classification_rows = []
        part_statuses: list[BranchStatus] = []
        part_latency_totals: list[float] = []
        part_latency_maxima: list[float] = []
        for part_items in by_part.values():
            expected = part_items[0][0]
            statuses = {item.status for _expected, item in part_items}
            if BranchStatus.ERROR in statuses:
                part_status = BranchStatus.ERROR
                predicted = "error"
            elif BranchStatus.NG in statuses:
                part_status = BranchStatus.NG
                predicted = "defect"
            elif BranchStatus.REVIEW in statuses:
                part_status = BranchStatus.REVIEW
                predicted = "review"
            elif BranchStatus.SKIPPED in statuses:
                part_status = BranchStatus.SKIPPED
                predicted = "review"
            else:
                part_status = BranchStatus.PASS
                predicted = "normal"
            part_statuses.append(part_status)
            elapsed = [item.elapsed_ms for _expected, item in part_items]
            part_latency_totals.append(sum(elapsed))
            part_latency_maxima.append(max(elapsed))
            classification_rows.append({"expected_label": expected, "predicted_label": predicted})
        counts = Counter(status.value for status in part_statuses)
        result[branch.value] = {
            "part_count": len(by_part),
            "diagnostic_view_evidence_count": len(items),
            "status_counts": {status.value: counts.get(status.value, 0) for status in BranchStatus},
            "classification_metrics": _binary_metrics(classification_rows),
            "part_latency_total_ms": _latencies(part_latency_totals),
            "part_latency_max_view_ms": _latencies(part_latency_maxima),
        }
    return result


def _runtime_profile_contract(
    profile_id: str,
    runtime: OfflineEvaluationRuntime,
) -> EvaluationProfileContract | None:
    if runtime.profile_id != profile_id:
        raise ValueError(f"runtime profile_id {runtime.profile_id!r} does not match {profile_id!r}")
    contract = getattr(runtime, "profile_contract", None)
    if contract is None:
        return None
    if not isinstance(contract, EvaluationProfileContract):
        raise TypeError(f"profile_contract for {profile_id} has an untrusted type")
    if contract.profile_id != profile_id:
        raise ValueError(f"profile contract identity does not match {profile_id}")
    return contract


def _archived_required_evidence(
    config: object,
    profile_id: str,
) -> dict[BranchName, frozenset[ViewId]]:
    if not isinstance(config, dict) or set(config) != _CONFIG_ARCHIVE_FIELDS:
        raise ValueError(f"config snapshot archive for {profile_id} is incomplete")
    if config.get("experiment_id") != profile_id:
        raise ValueError(f"config snapshot profile identity does not match {profile_id}")
    template = config.get("template")
    bright_streak = config.get("bright_streak")
    yolo = config.get("yolo")
    patchcore = config.get("patchcore")
    if not all(isinstance(section, dict) for section in (template, bright_streak, yolo, patchcore)):
        raise ValueError(f"config snapshot archive for {profile_id} has invalid branch sections")
    assert isinstance(template, dict)
    assert isinstance(bright_streak, dict)
    assert isinstance(yolo, dict)
    assert isinstance(patchcore, dict)
    if not isinstance(template.get("enabled"), bool):
        raise ValueError(f"config snapshot archive for {profile_id} has invalid Template enabled flag")
    if not isinstance(yolo.get("enabled"), bool):
        raise ValueError(f"config snapshot archive for {profile_id} has invalid YOLO enabled flag")
    if not isinstance(patchcore.get("enabled"), bool):
        raise ValueError(f"config snapshot archive for {profile_id} has invalid PatchCore enabled flag")
    config_paths = bright_streak.get("config_paths")
    if not isinstance(config_paths, dict):
        raise ValueError(f"config snapshot archive for {profile_id} has invalid bright-streak paths")
    try:
        bright_views = frozenset(ViewId(value) for value in config_paths)
    except ValueError as error:
        raise ValueError(f"config snapshot archive for {profile_id} has an unknown bright-streak view") from error
    required: dict[BranchName, frozenset[ViewId]] = {}
    if template["enabled"]:
        required[BranchName.TEMPLATE] = frozenset(ViewId)
    if bright_views:
        required[BranchName.BRIGHT_STREAK] = bright_views
    if yolo["enabled"]:
        required[BranchName.YOLO] = frozenset(ViewId)
    if patchcore["enabled"]:
        required[BranchName.PATCHCORE] = frozenset(ViewId)
    return required


def _validate_coverage(
    profile_id: str,
    part: EvaluationPart,
    outcome: EvaluationOutcome,
    required: Mapping[BranchName, frozenset[ViewId]],
) -> None:
    identities = [(item.branch, item.view_id) for item in outcome.evidence]
    if len(identities) != len(set(identities)):
        raise ValueError(f"duplicate branch/view evidence for profile {profile_id}, part {part.part_id}")
    actual = set(identities)
    for branch in BranchName:
        if branch not in required:
            continue
        views = required[branch]
        missing = {(branch, view) for view in views} - actual
        if missing:
            missing_views = ", ".join(sorted(view.value for _branch, view in missing))
            raise ValueError(
                f"evidence coverage for {branch.value} is incomplete on part "
                f"{part.part_id}: {missing_views}"
            )
    if outcome.final_status is FinalStatus.OK:
        for item in outcome.evidence:
            if (
                item.branch in required
                and item.view_id in required[item.branch]
                and item.status is not BranchStatus.PASS
            ):
                raise ValueError(
                    f"OK result has non-PASS evidence for {item.branch.value}/{item.view_id.value}"
                )


def _evidence_identity(
    profile_id: str,
    part: EvaluationPart,
    outcome: EvaluationOutcome,
    runtime: OfflineEvaluationRuntime,
    contract: EvaluationProfileContract | None,
) -> tuple[str, str, dict[BranchName, frozenset[ViewId]]]:
    root = Path(runtime.evidence_root).expanduser().resolve()
    try:
        outcome.evidence_path.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"evidence_path for {profile_id}/{part.part_id} is outside controlled evidence_root"
        ) from error
    result_path = outcome.evidence_path / "result.json"
    snapshot_path = outcome.evidence_path / "config.snapshot.json"
    if not result_path.is_file() or not snapshot_path.is_file():
        raise ValueError(f"evidence_path for {profile_id}/{part.part_id} lacks result.json or config.snapshot.json")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if not isinstance(result, dict) or not isinstance(snapshot, dict):
            raise TypeError
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"invalid evidence archive JSON for {profile_id}/{part.part_id}") from error
    if set(result) != _RESULT_ARCHIVE_FIELDS:
        raise ValueError(f"result archive for {profile_id}/{part.part_id} is incomplete")
    if set(snapshot) != _CONFIG_SNAPSHOT_FIELDS:
        raise ValueError(f"config snapshot archive for {profile_id}/{part.part_id} is incomplete")
    config = snapshot["config"]
    required = _archived_required_evidence(config, profile_id)
    recomputed_config_digest = hashlib.sha256(_canonical_json(config)).hexdigest()
    snapshot_digest = snapshot["config_digest"]
    result_config_digest = result["config_digest"]
    if snapshot_digest != recomputed_config_digest or result_config_digest != recomputed_config_digest:
        raise ValueError(f"config_digest mismatch in archive for {profile_id}/{part.part_id}")
    if contract is not None:
        if contract.config_digest != recomputed_config_digest:
            raise ValueError(f"trusted config_digest mismatch for {profile_id}/{part.part_id}")
        if dict(contract.required_evidence) != required:
            raise ValueError(f"trusted branch contract mismatch for {profile_id}/{part.part_id}")
    expected_capture = f"evaluation-{part.part_id}"
    if result.get("capture_set_id") != expected_capture:
        raise ValueError(f"result capture_set_id does not match part {part.part_id}: expected {expected_capture}")
    if result.get("final_status") != outcome.final_status.value:
        raise ValueError(f"result final_status does not match runtime outcome for {part.part_id}")
    _validate_coverage(profile_id, part, outcome, required)
    if result.get("evidence") != _json_value(outcome.evidence):
        raise ValueError(f"archived evidence does not match runtime outcome for {part.part_id}")
    if not isinstance(result.get("created_at"), str) or not result["created_at"]:
        raise ValueError(f"result archive created_at is invalid for {part.part_id}")
    if not isinstance(result.get("reason"), str) or not result["reason"]:
        raise ValueError(f"result archive reason is invalid for {part.part_id}")
    if not isinstance(result.get("required_complete"), bool):
        raise ValueError(f"result archive required_complete is invalid for {part.part_id}")
    if not isinstance(result.get("triggered_branches"), list):
        raise ValueError(f"result archive triggered_branches is invalid for {part.part_id}")
    return _sha256(result_path), recomputed_config_digest, required


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _candidate_thresholds(
    runtime: OfflineEvaluationRuntime,
    parts: tuple[EvaluationPart, ...],
) -> dict[str, float]:
    fit: Callable[[tuple[EvaluationPart, ...]], Mapping[str, float]] | None = getattr(
        runtime,
        "calibration_thresholds",
        None,
    )
    if fit is None:
        return {}
    candidates = dict(fit(parts))
    for name, value in candidates.items():
        if not isinstance(name, str) or not name or isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError("candidate thresholds must map non-empty names to finite numbers")
    return {name: float(value) for name, value in sorted(candidates.items())}


def evaluate_profiles(
    *,
    manifest_path: Path,
    profiles: Mapping[str, OfflineEvaluationRuntime],
    split: str,
    output_root: Path,
    write_thresholds: bool = False,
) -> dict[str, object]:
    """Evaluate one manifest split across one or more resident profiles."""
    if split not in _EVALUATION_SPLITS:
        raise ValueError("split must be calibration or final_test")
    if split == "final_test" and write_thresholds:
        raise ValueError("final_test evaluation is read-only; --write-thresholds is forbidden")
    if not profiles:
        raise ValueError("at least one evaluation profile is required")
    for profile_id in profiles:
        if not isinstance(profile_id, str) or _SAFE_PROFILE_ID.fullmatch(profile_id) is None:
            raise ValueError(f"unsafe or empty profile id: {profile_id!r}")

    rows = read_manifest(manifest_path)
    leakage = find_part_split_leakage(rows)
    if leakage:
        raise ValueError(f"physical-part split leakage detected: {leakage}")
    _validate_manifest_samples(rows)
    parts = _parts(rows, split)
    destination = Path(output_root).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"evaluation output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    prediction_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    branch_csv_rows: list[dict[str, object]] = []
    summary_profiles: dict[str, Any] = {}
    threshold_transactions: list[tuple[OfflineEvaluationRuntime, dict[str, float]]] = []
    snapshots: list[tuple[Callable[[object], None], object]] = []
    try:
        for profile_id, runtime in profiles.items():
            contract = _runtime_profile_contract(profile_id, runtime)
            profile_required: dict[BranchName, frozenset[ViewId]] | None = None
            profile_config_digest: str | None = None
            profile_rows: list[dict[str, object]] = []
            all_evidence: list[tuple[str, str, BranchEvidence]] = []
            total_latencies: list[float] = []
            for part in parts:
                outcome = runtime.inspect(part)
                if not isinstance(outcome, EvaluationOutcome):
                    raise TypeError(f"runtime {profile_id} must return EvaluationOutcome")
                result_digest, config_digest, archived_required = _evidence_identity(
                    profile_id,
                    part,
                    outcome,
                    runtime,
                    contract,
                )
                if profile_required is None:
                    profile_required = archived_required
                    profile_config_digest = config_digest
                elif profile_required != archived_required or profile_config_digest != config_digest:
                    raise ValueError(f"profile config archive changed between parts for {profile_id}")
                predicted = _predicted_label(outcome.final_status)
                total_latency = sum(item.elapsed_ms for item in outcome.evidence)
                row: dict[str, object] = {
                    "profile_id": profile_id,
                    "part_id": part.part_id,
                    "session_id": part.session_id,
                    "split": split,
                    "expected_label": part.expected_label,
                    "predicted_label": predicted,
                    "final_status": outcome.final_status.value,
                    "total_latency_ms": total_latency,
                    "evidence_path": str(outcome.evidence_path),
                    "evidence_result_sha256": result_digest,
                }
                profile_rows.append(row)
                prediction_rows.append(row)
                all_evidence.extend((part.part_id, part.expected_label, item) for item in outcome.evidence)
                total_latencies.append(total_latency)
                error_kind = (
                    "false_ok" if part.expected_label == "defect" and predicted == "normal"
                    else "false_ng" if part.expected_label == "normal" and predicted == "defect"
                    else None
                )
                if error_kind is not None:
                    link_dir = staging / error_kind / profile_id
                    link_dir.mkdir(parents=True, exist_ok=True)
                    _write_json(
                        link_dir / f"{part.part_id}.json",
                        {
                            "part_id": part.part_id,
                            "profile_id": profile_id,
                            "evidence_path": str(outcome.evidence_path),
                            "evidence_result_sha256": result_digest,
                        },
                    )

            metrics = _binary_metrics(profile_rows)
            per_branch = _branch_metrics(all_evidence)
            assert profile_required is not None
            missing = sorted(set(BranchName) - set(profile_required), key=lambda item: item.value)
            experimental_only = contract is None or bool(missing)
            candidates = _candidate_thresholds(runtime, parts) if split == "calibration" else {}
            if write_thresholds:
                threshold_transactions.append((runtime, candidates))
            profile_summary: dict[str, Any] = {
                "experimental_only": experimental_only,
                "trusted_profile_contract": contract is not None,
                "config_digest": profile_config_digest,
                "missing_branches": [branch.value for branch in missing],
                "candidate_thresholds": candidates,
                "per_branch_metrics": per_branch,
                "latency_ms": _latencies(total_latencies),
            }
            profile_summary[
                "partial_system_metrics" if experimental_only else "fused_metrics"
            ] = metrics
            summary_profiles[profile_id] = profile_summary
            confusion = Counter(
                (str(row["expected_label"]), str(row["predicted_label"])) for row in profile_rows
            )
            for actual in ("normal", "defect"):
                for predicted in ("normal", "defect", "review", "error"):
                    confusion_rows.append(
                        {
                            "profile_id": profile_id,
                            "actual_label": actual,
                            "predicted_label": predicted,
                            "count": confusion.get((actual, predicted), 0),
                        }
                    )
            for branch_name, branch in per_branch.items():
                branch_csv_rows.append(
                    {
                        "profile_id": profile_id,
                        "branch": branch_name,
                        "part_count": branch["part_count"],
                        "diagnostic_view_evidence_count": branch[
                            "diagnostic_view_evidence_count"
                        ],
                        "pass_count": branch["status_counts"][BranchStatus.PASS.value],
                        "ng_count": branch["status_counts"][BranchStatus.NG.value],
                        "review_count": branch["status_counts"][BranchStatus.REVIEW.value],
                        "error_count": branch["status_counts"][BranchStatus.ERROR.value],
                        "skipped_count": branch["status_counts"][BranchStatus.SKIPPED.value],
                        "latency_total_p50_ms": branch["part_latency_total_ms"]["p50"],
                        "latency_total_p95_ms": branch["part_latency_total_ms"]["p95"],
                        "latency_total_p99_ms": branch["part_latency_total_ms"]["p99"],
                        "latency_max_view_p50_ms": branch["part_latency_max_view_ms"]["p50"],
                        "latency_max_view_p95_ms": branch["part_latency_max_view_ms"]["p95"],
                        "latency_max_view_p99_ms": branch["part_latency_max_view_ms"]["p99"],
                    }
                )

        summary: dict[str, Any] = {
            "schema_version": 1,
            "split": split,
            "comparison": len(profiles) > 1,
            "experimental_only": any(profile["experimental_only"] for profile in summary_profiles.values()),
            "thresholds_written": write_thresholds,
            "manifest_path": str(Path(manifest_path).expanduser().resolve()),
            "part_count": len(parts),
            "profiles": summary_profiles,
        }
        _write_csv(staging / "predictions.csv", _PREDICTION_FIELDS, prediction_rows)
        _write_csv(
            staging / "confusion.csv",
            ("profile_id", "actual_label", "predicted_label", "count"),
            confusion_rows,
        )
        branch_fields = (
            "profile_id",
            "branch",
            "part_count",
            "diagnostic_view_evidence_count",
            "pass_count",
            "ng_count",
            "review_count",
            "error_count",
            "skipped_count",
            "latency_total_p50_ms",
            "latency_total_p95_ms",
            "latency_total_p99_ms",
            "latency_max_view_p50_ms",
            "latency_max_view_p95_ms",
            "latency_max_view_p99_ms",
        )
        _write_csv(staging / "branch_metrics.csv", branch_fields, branch_csv_rows)
        _write_json(staging / "summary.json", summary)

        if write_thresholds:
            for runtime, _candidates in threshold_transactions:
                snapshotter = getattr(runtime, "snapshot_thresholds", None)
                restorer = getattr(runtime, "restore_thresholds", None)
                writer = getattr(runtime, "write_thresholds", None)
                if not callable(snapshotter) or not callable(restorer) or not callable(writer):
                    raise ValueError(
                        "threshold writing requires snapshot_thresholds, write_thresholds, "
                        "and restore_thresholds"
                    )
                snapshots.append((restorer, snapshotter()))
            for runtime, candidates in threshold_transactions:
                runtime.write_thresholds(candidates)  # type: ignore[attr-defined]
        if destination.exists():
            raise FileExistsError(f"evaluation output already exists: {destination}")
        staging.rename(destination)
        return summary
    except BaseException:
        for restore, snapshot in reversed(snapshots):
            try:
                restore(snapshot)
            except Exception:
                pass
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "EvaluationOutcome",
    "EvaluationPart",
    "EvaluationProfileContract",
    "LabRuntimeEvaluationAdapter",
    "OfflineEvaluationRuntime",
    "build_offline_runtime",
    "evaluate_profiles",
]
