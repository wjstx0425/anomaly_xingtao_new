"""Small dependency-free timing recorder for ZS32 pipeline generations."""

from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class TimingRecorder:
    """Accumulate monotonic stage durations and publish one atomic JSON sidecar."""

    component: str
    started_at: str = field(default_factory=_utc_now)
    _stages: dict[str, dict[str, float | int]] = field(default_factory=dict)

    def add(self, name: str, seconds: float) -> None:
        """Add one finite non-negative duration to *name*."""
        value = float(seconds)
        if not name or not math.isfinite(value) or value < 0:
            raise ValueError("timing stage requires a name and a finite non-negative duration")
        stage = self._stages.setdefault(name, {"count": 0, "seconds": 0.0})
        stage["count"] = int(stage["count"]) + 1
        stage["seconds"] = float(stage["seconds"]) + value

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        """Measure a stage with ``time.perf_counter`` even when it raises."""
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - started)

    def payload(self, *, total_seconds: float | None = None) -> dict[str, Any]:
        """Return the stable JSON payload."""
        payload: dict[str, Any] = {
            "schema_version": 1,
            "component": self.component,
            "started_at": self.started_at,
            "completed_at": _utc_now(),
            "stages": {name: dict(value) for name, value in sorted(self._stages.items())},
        }
        if total_seconds is not None:
            value = float(total_seconds)
            if not math.isfinite(value) or value < 0:
                raise ValueError("total_seconds must be finite and non-negative")
            payload["total_seconds"] = value
        return payload

    def write(self, path: Path, *, total_seconds: float | None = None) -> Path:
        """Atomically replace *path* with the current timing payload."""
        path = path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(self.payload(total_seconds=total_seconds), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return path


def merge_timing_payloads(
    *payloads: Mapping[str, Any],
    prefixes: Sequence[str] | None = None,
    total_seconds: float | None = None,
) -> dict[str, Any]:
    """Merge child timing payloads without losing duplicate stage durations."""
    if prefixes is not None and len(prefixes) != len(payloads):
        raise ValueError("prefixes must match the number of timing payloads")
    recorder = TimingRecorder("zs32_end_to_end")
    for index, payload in enumerate(payloads):
        prefix = prefixes[index].strip("/") if prefixes is not None else ""
        stages = payload.get("stages", {})
        if not isinstance(stages, Mapping):
            continue
        for name, value in stages.items():
            if not isinstance(name, str) or not isinstance(value, Mapping):
                continue
            seconds = value.get("seconds")
            count = value.get("count", 1)
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                continue
            if isinstance(seconds, bool) or not isinstance(seconds, int | float):
                continue
            merged_name = f"{prefix}/{name}" if prefix else name
            recorder.add(merged_name, float(seconds))
            recorder._stages[merged_name]["count"] = count
    if total_seconds is None:
        for payload in payloads:
            candidate = payload.get("total_seconds")
            if not isinstance(candidate, bool) and isinstance(candidate, int | float):
                value = float(candidate)
                if math.isfinite(value) and value >= 0:
                    total_seconds = value
                    break
    return recorder.payload(total_seconds=total_seconds)


def print_timing_summary(payload: Mapping[str, Any], *, limit: int = 12) -> None:
    """Print one compact, slowest-first terminal summary."""
    stages = payload.get("stages", {})
    if not isinstance(stages, Mapping):
        return
    ranked: list[tuple[str, float]] = []
    for name, value in stages.items():
        if isinstance(name, str) and isinstance(value, Mapping):
            seconds = value.get("seconds")
            if not isinstance(seconds, bool) and isinstance(seconds, int | float):
                ranked.append((name, float(seconds)))
    print("timing:")
    for name, seconds in sorted(ranked, key=lambda item: item[1], reverse=True)[:limit]:
        print(f"  {name}: {seconds:.3f}s")
    total = payload.get("total_seconds")
    if not isinstance(total, bool) and isinstance(total, int | float):
        print(f"  total: {float(total):.3f}s")
