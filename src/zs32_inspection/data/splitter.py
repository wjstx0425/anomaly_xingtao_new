"""Deterministic physical-part grouped split assignment."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


SPLIT_ALGORITHM = "sha256_seed_stratum_part_v1"


@dataclass(frozen=True, slots=True)
class SplitPolicy:
    """Frozen algorithm and parameters used for physical-part split assignment."""

    algorithm: str
    seed: int
    calibration_ratio: float
    test_ratio: float

    def __post_init__(self) -> None:
        if self.algorithm != SPLIT_ALGORITHM:
            raise ValueError(
                f"unsupported split algorithm {self.algorithm!r}; "
                f"expected {SPLIT_ALGORITHM!r}"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("split seed must be an integer")
        for field in ("calibration_ratio", "test_ratio"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"split {field} must be a finite number")
            value = float(value)
            if not math.isfinite(value) or value < 0 or value >= 1:
                raise ValueError(f"split {field} must be finite and in [0, 1)")
            object.__setattr__(self, field, value)
        if self.calibration_ratio + self.test_ratio >= 1:
            raise ValueError("calibration/test ratios must sum to less than one")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "SplitPolicy":
        """Load a policy only when its envelope exactly matches the frozen schema."""
        expected = {"algorithm", "seed", "calibration_ratio", "test_ratio"}
        if set(payload) != expected:
            raise ValueError(
                "split policy fields differ from strict schema; "
                f"missing={sorted(expected - set(payload))}, "
                f"unknown={sorted(set(payload) - expected)}"
            )
        return cls(
            algorithm=payload["algorithm"],  # type: ignore[arg-type]
            seed=payload["seed"],  # type: ignore[arg-type]
            calibration_ratio=payload["calibration_ratio"],  # type: ignore[arg-type]
            test_ratio=payload["test_ratio"],  # type: ignore[arg-type]
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "seed": self.seed,
            "calibration_ratio": self.calibration_ratio,
            "test_ratio": self.test_ratio,
        }

    @property
    def sha256(self) -> str:
        payload = (
            json.dumps(
                self.as_dict(),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    """One physical part assigned wholly to one data split."""

    part_instance_id: str
    split: str
    stratum: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.part_instance_id, str)
            or not isinstance(self.stratum, str)
            or not self.part_instance_id.strip()
            or not self.stratum.strip()
        ):
            raise ValueError("split assignment identity and stratum must not be empty")
        if not isinstance(self.split, str):
            raise ValueError("split assignment split must be a string")
        if self.split not in {"train", "calibration", "test"}:
            raise ValueError(f"unsupported split: {self.split!r}")


def assign_grouped_splits(
    strata_by_part: Mapping[str, str],
    *,
    policy: SplitPolicy,
) -> tuple[SplitAssignment, ...]:
    """Assign whole parts deterministically inside each stratum.

    The SHA256 ordering is stable across Python versions and platforms; unlike
    ``random`` it does not depend on an interpreter implementation detail.
    """
    if not isinstance(policy, SplitPolicy):
        raise TypeError("split assignment requires a structured SplitPolicy")
    if not strata_by_part:
        raise ValueError("split assignment requires at least one physical part")
    grouped: dict[str, list[str]] = defaultdict(list)
    for part_instance_id, stratum in strata_by_part.items():
        if not part_instance_id.strip() or not stratum.strip():
            raise ValueError("part_instance_id and split stratum must not be empty")
        grouped[stratum].append(part_instance_id)
    assignments: list[SplitAssignment] = []
    for stratum, part_ids in sorted(grouped.items()):
        ordered = sorted(
            set(part_ids),
            key=lambda item: hashlib.sha256(
                f"{policy.seed}:{stratum}:{item}".encode("utf-8")
            ).hexdigest(),
        )
        size = len(ordered)
        calibration_count = round(size * policy.calibration_ratio)
        test_count = round(size * policy.test_ratio)
        if policy.calibration_ratio > 0 and size >= 3:
            calibration_count = max(1, calibration_count)
        if policy.test_ratio > 0 and size >= 3:
            test_count = max(1, test_count)
        while calibration_count + test_count >= size:
            if calibration_count >= test_count and calibration_count:
                calibration_count -= 1
            elif test_count:
                test_count -= 1
        split_by_id = {
            item: (
                "test"
                if index < test_count
                else "calibration"
                if index < test_count + calibration_count
                else "train"
            )
            for index, item in enumerate(ordered)
        }
        assignments.extend(
            SplitAssignment(item, split_by_id[item], stratum) for item in ordered
        )
    return tuple(sorted(assignments, key=lambda item: item.part_instance_id))


def validate_part_grouped_splits(
    rows: Iterable[tuple[str, str]],
    *,
    required_part_ids: Sequence[str] | None = None,
) -> dict[str, str]:
    """Reject leakage of one physical part across multiple splits."""
    by_part: dict[str, str] = {}
    for part_instance_id, split in rows:
        if split not in {"train", "calibration", "test"}:
            raise ValueError(f"unsupported split for {part_instance_id!r}: {split!r}")
        previous = by_part.setdefault(part_instance_id, split)
        if previous != split:
            raise ValueError(
                f"physical part {part_instance_id!r} leaks across splits: {previous!r}, {split!r}"
            )
    if required_part_ids is not None:
        missing = sorted(set(required_part_ids) - set(by_part))
        extra = sorted(set(by_part) - set(required_part_ids))
        if missing or extra:
            raise ValueError(f"split assignment identity mismatch; missing={missing}, extra={extra}")
    return by_part
