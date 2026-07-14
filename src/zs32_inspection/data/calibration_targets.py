"""Human-approved, dataset-bound calibration targets for every crop branch."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Sequence

from zs32_inspection.domain.identity import Hand

from .manifests import CanonicalSampleRow


CALIBRATION_TARGET_BRANCHES = ("template", "anomaly", "yolo")


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"calibration targets contains duplicate JSON key: {key!r}")
        output[key] = value
    return output


def _canonical_text(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character in value for character in ("\x00", "\n", "\r"))
    ):
        raise ValueError(f"{field} must be a canonical non-empty single-line string")
    return value


class CalibrationTargetValue(str, Enum):
    """Whether one branch should see a defect, see normal, or be excluded."""

    NORMAL = "normal"
    DEFECT = "defect"
    EXCLUDE = "exclude"


@dataclass(frozen=True, slots=True)
class CalibrationTargetApproval:
    """Human approval identity bound to the exact target list."""

    reviewed_by: str
    reviewed_at: str
    approved: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reviewed_by",
            _canonical_text(self.reviewed_by, "calibration target reviewed_by"),
        )
        if (
            not isinstance(self.reviewed_at, str)
            or self.reviewed_at != self.reviewed_at.strip()
            or re.fullmatch(
                r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})",
                self.reviewed_at,
            )
            is None
        ):
            raise ValueError("calibration target reviewed_at must be canonical RFC3339")
        try:
            parsed = datetime.fromisoformat(self.reviewed_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("calibration target reviewed_at is invalid") from error
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("calibration target reviewed_at must include a timezone")
        if self.approved is not True:
            raise ValueError("calibration target approval must be exactly true")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "CalibrationTargetApproval":
        expected = {"reviewed_by", "reviewed_at", "approved"}
        if set(payload) != expected:
            raise ValueError("calibration target approval fields differ from strict schema")
        return cls(
            reviewed_by=payload["reviewed_by"],  # type: ignore[arg-type]
            reviewed_at=payload["reviewed_at"],  # type: ignore[arg-type]
            approved=payload["approved"],  # type: ignore[arg-type]
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
            "approved": self.approved,
        }


@dataclass(frozen=True, slots=True, order=True)
class CalibrationTargetRecord:
    """One explicit target for one capture/view/detection branch."""

    capture_set_id: str
    part_instance_id: str
    hand: str
    view: str
    branch: str
    part_ground_truth: str
    target: CalibrationTargetValue
    reason: str | None

    def __post_init__(self) -> None:
        for field in ("capture_set_id", "part_instance_id", "view"):
            object.__setattr__(
                self,
                field,
                _canonical_text(getattr(self, field), f"calibration target {field}"),
            )
        object.__setattr__(self, "hand", Hand.parse(self.hand).value)
        if self.branch not in CALIBRATION_TARGET_BRANCHES:
            raise ValueError(f"invalid calibration target branch: {self.branch!r}")
        if self.part_ground_truth not in {"normal", "defect"}:
            raise ValueError("part_ground_truth must be normal or defect")
        object.__setattr__(self, "target", CalibrationTargetValue(self.target))
        if self.reason is not None:
            object.__setattr__(
                self,
                "reason",
                _canonical_text(self.reason, "calibration target reason"),
            )
        if self.part_ground_truth == "normal" and self.target is CalibrationTargetValue.DEFECT:
            raise ValueError("a normal part cannot have a defect calibration target")
        if (
            self.target is CalibrationTargetValue.EXCLUDE
            or (
                self.part_ground_truth == "defect"
                and self.target is CalibrationTargetValue.NORMAL
            )
        ) and self.reason is None:
            raise ValueError("exclude and defect-to-normal targets require an audited reason")

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        return (
            self.capture_set_id,
            self.part_instance_id,
            self.hand,
            self.view,
            self.branch,
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "CalibrationTargetRecord":
        expected = {
            "capture_set_id", "part_instance_id", "hand", "view", "branch",
            "part_ground_truth", "target", "reason",
        }
        if set(payload) != expected:
            raise ValueError("calibration target record fields differ from strict schema")
        return cls(
            capture_set_id=payload["capture_set_id"],  # type: ignore[arg-type]
            part_instance_id=payload["part_instance_id"],  # type: ignore[arg-type]
            hand=payload["hand"],  # type: ignore[arg-type]
            view=payload["view"],  # type: ignore[arg-type]
            branch=payload["branch"],  # type: ignore[arg-type]
            part_ground_truth=payload["part_ground_truth"],  # type: ignore[arg-type]
            target=payload["target"],  # type: ignore[arg-type]
            reason=payload["reason"],  # type: ignore[arg-type]
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "capture_set_id": self.capture_set_id,
            "part_instance_id": self.part_instance_id,
            "hand": self.hand,
            "view": self.view,
            "branch": self.branch,
            "part_ground_truth": self.part_ground_truth,
            "target": self.target.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class CalibrationTargetsSnapshot:
    """Canonical approved target document embedded in one dataset release."""

    dataset_release_id: str
    topology_id: str
    roi_version: str
    approval: CalibrationTargetApproval
    targets: tuple[CalibrationTargetRecord, ...]

    def __post_init__(self) -> None:
        for field in ("dataset_release_id", "topology_id", "roi_version"):
            object.__setattr__(
                self,
                field,
                _canonical_text(getattr(self, field), f"calibration targets {field}"),
            )
        if not isinstance(self.approval, CalibrationTargetApproval):
            raise TypeError("calibration targets approval must be structured")
        materialized = tuple(self.targets)
        if not materialized or any(
            not isinstance(item, CalibrationTargetRecord) for item in materialized
        ):
            raise ValueError("calibration targets must contain structured records")
        ordered = tuple(sorted(materialized, key=lambda item: item.key))
        if materialized != ordered:
            raise ValueError("calibration target records must use canonical key order")
        keys = tuple(item.key for item in materialized)
        if len(keys) != len(set(keys)):
            raise ValueError("calibration target contract contains duplicate keys")
        object.__setattr__(self, "targets", materialized)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "CalibrationTargetsSnapshot":
        expected = {
            "schema", "schema_version", "product", "dataset_release_id",
            "topology_id", "roi_version", "approval", "targets",
        }
        if set(payload) != expected:
            raise ValueError("calibration targets fields differ from strict schema")
        if (
            payload["schema"] != "zs32.calibration_targets"
            or payload["schema_version"] != 1
            or payload["product"] != "ZS32"
        ):
            raise ValueError("calibration targets must use zs32.calibration_targets v1 for ZS32")
        approval = payload["approval"]
        targets = payload["targets"]
        if not isinstance(approval, Mapping) or not isinstance(targets, list):
            raise ValueError("calibration targets approval/targets types are invalid")
        parsed_targets: list[CalibrationTargetRecord] = []
        for index, item in enumerate(targets):
            if not isinstance(item, Mapping):
                raise ValueError(f"calibration targets[{index}] must be an object")
            parsed_targets.append(CalibrationTargetRecord.from_mapping(item))
        return cls(
            dataset_release_id=payload["dataset_release_id"],  # type: ignore[arg-type]
            topology_id=payload["topology_id"],  # type: ignore[arg-type]
            roi_version=payload["roi_version"],  # type: ignore[arg-type]
            approval=CalibrationTargetApproval.from_mapping(approval),
            targets=tuple(parsed_targets),
        )

    @property
    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "zs32.calibration_targets",
            "schema_version": 1,
            "product": "ZS32",
            "dataset_release_id": self.dataset_release_id,
            "topology_id": self.topology_id,
            "roi_version": self.roi_version,
            "approval": self.approval.as_dict(),
            "targets": [item.as_dict() for item in self.targets],
        }

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.as_dict)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @property
    def by_key(self) -> Mapping[tuple[str, str, str, str, str], CalibrationTargetRecord]:
        return MappingProxyType({item.key: item for item in self.targets})


def validate_calibration_targets(
    snapshot: CalibrationTargetsSnapshot,
    rows: Sequence[CanonicalSampleRow],
    *,
    dataset_release_id: str,
    topology_id: str,
    roi_version: str,
) -> None:
    """Require an exact three-branch target join for every canonical crop."""
    if (
        snapshot.dataset_release_id != dataset_release_id
        or snapshot.topology_id != topology_id
        or snapshot.roi_version != roi_version
    ):
        raise ValueError("calibration targets identity differs from dataset contracts")
    expected: dict[tuple[str, str, str, str, str], str] = {}
    for row in rows:
        for branch in CALIBRATION_TARGET_BRANCHES:
            key = (
                row.capture_set_id,
                row.part_instance_id,
                row.hand,
                row.view,
                branch,
            )
            if key in expected:
                raise ValueError(f"canonical dataset contains duplicate calibration target key: {key}")
            expected[key] = row.label
    actual = snapshot.by_key
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise ValueError(
            "calibration targets do not exactly cover canonical crops; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for key, part_ground_truth in expected.items():
        if actual[key].part_ground_truth != part_ground_truth:
            raise ValueError(f"calibration target part_ground_truth differs from canonical label: {key}")


def load_calibration_targets(path) -> CalibrationTargetsSnapshot:
    """Load one strict JSON target contract without accepting symlinks."""
    from pathlib import Path

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"calibration targets must be a regular non-symlink file: {source}")
    try:
        payload = json.loads(
            source.read_bytes().decode("utf-8"),
            object_pairs_hook=_strict_json_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load calibration targets {source}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("calibration targets root must be an object")
    return CalibrationTargetsSnapshot.from_mapping(payload)
