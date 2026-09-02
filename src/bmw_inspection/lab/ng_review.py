"""Data model and persistence for BMW Template / EfficientAD NG review."""

from __future__ import annotations

import csv
import os
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path


DECISIONS = ("", "误判", "真实缺陷", "不确定")
REQUIRED_FIELDS = (
    "case_id",
    "decision",
    "review_note",
    "capture_id",
    "branch",
    "view_id",
    "panel_path",
)


@dataclass(slots=True)
class ReviewCase:
    """One detector problem awaiting a human decision."""

    row: dict[str, str]

    @property
    def case_id(self) -> str:
        """Return the stable review identifier."""
        return self.row["case_id"]

    @property
    def capture_id(self) -> str:
        """Return the inspected part identifier."""
        return self.row["capture_id"]

    @property
    def decision(self) -> str:
        """Return the current human decision."""
        return self.row["decision"]

    @property
    def review_note(self) -> str:
        """Return the current human note."""
        return self.row["review_note"]


@dataclass(frozen=True, slots=True)
class ReviewGroup:
    """All detector problems belonging to one captured part."""

    capture_id: str
    cases: tuple[ReviewCase, ...]


@dataclass(frozen=True, slots=True)
class ReviewProgress:
    """Current review completion counters."""

    case_count: int
    reviewed_case_count: int
    capture_count: int
    completed_capture_count: int


class ReviewDataset:
    """Validated, mutable review rows backed by one CSV file."""

    def __init__(self, path: Path, fieldnames: tuple[str, ...], cases: tuple[ReviewCase, ...]) -> None:
        self.path = path
        self.fieldnames = fieldnames
        self.cases = cases
        by_capture: OrderedDict[str, list[ReviewCase]] = OrderedDict()
        self._by_case_id: dict[str, ReviewCase] = {}
        for case in cases:
            if case.case_id in self._by_case_id:
                raise ValueError(f"duplicate case_id: {case.case_id}")
            self._by_case_id[case.case_id] = case
            by_capture.setdefault(case.capture_id, []).append(case)
        self.groups = tuple(
            ReviewGroup(capture_id=capture_id, cases=tuple(group_cases))
            for capture_id, group_cases in by_capture.items()
        )
        self._by_capture = {group.capture_id: group for group in self.groups}

    @classmethod
    def load(cls, path: Path | str) -> ReviewDataset:
        """Load and validate a review CSV without changing it."""
        resolved = Path(path).expanduser().resolve()
        with resolved.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = tuple(reader.fieldnames or ())
            missing = [field for field in REQUIRED_FIELDS if field not in fieldnames]
            if missing:
                raise ValueError(f"missing required columns: {', '.join(missing)}")
            cases = tuple(ReviewCase(dict(row)) for row in reader)
        for case in cases:
            _validate_decision(case.decision)
        return cls(resolved, fieldnames, cases)

    def find_case(self, case_id: str) -> ReviewCase:
        """Return one case by its stable identifier."""
        try:
            return self._by_case_id[case_id]
        except KeyError as error:
            raise KeyError(f"unknown case_id: {case_id}") from error

    def set_decision(self, case_id: str, decision: str) -> None:
        """Set one validated human decision."""
        _validate_decision(decision)
        self.find_case(case_id).row["decision"] = decision

    def set_note(self, case_id: str, note: str) -> None:
        """Set one free-form human note."""
        self.find_case(case_id).row["review_note"] = note

    def set_remaining_for_capture(self, capture_id: str, decision: str) -> int:
        """Set undecided cases for one part and return the number changed."""
        _validate_decision(decision)
        if not decision:
            raise ValueError("batch decision cannot be empty")
        try:
            group = self._by_capture[capture_id]
        except KeyError as error:
            raise KeyError(f"unknown capture_id: {capture_id}") from error
        changed = 0
        for case in group.cases:
            if not case.decision:
                case.row["decision"] = decision
                changed += 1
        return changed

    @property
    def progress(self) -> ReviewProgress:
        """Calculate problem-level and part-level completion counters."""
        reviewed = sum(bool(case.decision) for case in self.cases)
        completed = sum(all(case.decision for case in group.cases) for group in self.groups)
        return ReviewProgress(
            case_count=len(self.cases),
            reviewed_case_count=reviewed,
            capture_count=len(self.groups),
            completed_capture_count=completed,
        )

    def save(self) -> None:
        """Atomically persist all review rows while preserving the CSV schema."""
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8-sig",
                newline="",
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                dir=self.path.parent,
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                writer = csv.DictWriter(stream, fieldnames=self.fieldnames, extrasaction="raise")
                writer.writeheader()
                writer.writerows(case.row for case in self.cases)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, self.path)
        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise


def _validate_decision(decision: str) -> None:
    if decision not in DECISIONS:
        raise ValueError(f"invalid decision: {decision!r}")
