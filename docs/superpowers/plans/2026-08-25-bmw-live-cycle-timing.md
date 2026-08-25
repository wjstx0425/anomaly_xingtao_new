# BMW Live Cycle Timing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accurately measure the live GUI interval from the first accepted Space key to the first submitted RESULT frame, display the timing, and save the phase breakdown beside each inspection.

**Architecture:** Add one small timing value module containing the final immutable record and JSON writer. Keep mutable timestamp marks private to the live GUI pipeline, wrap the existing single-worker round/finalize/persist calls without changing detector code, and attach the completed timing record to UI state only after the first RESULT `imshow` returns.

**Tech Stack:** Python 3.13, `time.perf_counter_ns`, OpenCV HighGUI, dataclasses, JSON, pytest, uv.

## Global Constraints

- Do not change camera/HDR parameters, Template, bright-streak, YOLO, EfficientAD, thresholds, ROI, masks, or final fusion.
- Keep the current one-worker inference ordering and front-inference overlap with flip/back capture.
- Do not add SHA, receipt, provenance, publisher, or schema-version machinery.
- Preserve all unrelated dirty-worktree changes and do not commit shared modified runtime/test files.
- Offline `--no-gui` and offline GUI replay must keep their current behavior and must not invent capture or flip timings.

---

### Task 1: Final timing record and JSON persistence

**Files:**
- Create: `src/bmw_inspection/lab/live_cycle_timing.py`
- Create: `tests/unit/bmw_inspection/lab/test_live_cycle_timing.py`

**Interfaces:**
- Produces: `LiveCycleTiming`, `persist_cycle_timing(result_root: Path, timing: LiveCycleTiming) -> Path`.
- Consumes: only standard-library dataclasses, JSON, math, datetime strings, and pathlib.

- [ ] **Step 1: Write the failing tests**

```python
def test_live_cycle_timing_exposes_exact_total_and_overlap() -> None:
    timing = LiveCycleTiming(
        capture_id="bmw_demo_20260825_120000",
        started_at="2026-08-25T12:00:00.000+08:00",
        displayed_at="2026-08-25T12:00:12.000+08:00",
        front_capture_ms=1200.0,
        flip_wait_ms=2500.0,
        back_capture_ms=1300.0,
        front_inference_ms=3900.0,
        back_inference_ms=3800.0,
        finalize_ms=200.0,
        persist_ms=5100.0,
        result_display_ms=30.0,
        front_overlap_ms=3800.0,
        total_cycle_ms=12000.0,
    )
    assert timing.total_cycle_ms == 12000.0
    assert timing.front_overlap_ms == 3800.0


def test_persist_cycle_timing_writes_simple_json(tmp_path: Path) -> None:
    timing = _timing_fixture()
    (tmp_path / timing.capture_id).mkdir()

    path = persist_cycle_timing(tmp_path, timing)

    assert path == (tmp_path / timing.capture_id / "cycle_timing.json").resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["capture_id"] == timing.capture_id
    assert payload["boundary"] == "first_space_accepted_to_first_result_imshow_returned"
    assert payload["total_cycle_ms"] == 12000.0
    assert "sha256" not in path.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run RED**

Run:

```bash
UV_CACHE_DIR=/tmp/bmw-cycle-timing-uv-cache uv run --project /home/yunjing/anomaly_xingtao_new --no-sync python -m pytest -q tests/unit/bmw_inspection/lab/test_live_cycle_timing.py
```

Expected: collection fails because `bmw_inspection.lab.live_cycle_timing` does not exist.

- [ ] **Step 3: Implement the minimum timing module**

```python
@dataclass(frozen=True, slots=True)
class LiveCycleTiming:
    capture_id: str
    started_at: str
    displayed_at: str
    front_capture_ms: float
    flip_wait_ms: float
    back_capture_ms: float
    front_inference_ms: float
    back_inference_ms: float
    finalize_ms: float
    persist_ms: float
    result_display_ms: float
    front_overlap_ms: float
    total_cycle_ms: float

    def __post_init__(self) -> None:
        if not self.capture_id.strip() or not self.started_at.strip() or not self.displayed_at.strip():
            raise ValueError("完整周期计时身份和墙钟时间不能为空")
        for field in fields(self)[3:]:
            value = float(getattr(self, field.name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"完整周期计时字段无效：{field.name}")

    def to_payload(self) -> dict[str, str | float]:
        return {
            "capture_id": self.capture_id,
            "boundary": "first_space_accepted_to_first_result_imshow_returned",
            "started_at": self.started_at,
            "displayed_at": self.displayed_at,
            **{field.name: float(getattr(self, field.name)) for field in fields(self)[3:]},
        }


def persist_cycle_timing(result_root: Path, timing: LiveCycleTiming) -> Path:
    path = Path(result_root).expanduser().resolve() / timing.capture_id / "cycle_timing.json"
    path.write_text(json.dumps(timing.to_payload(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path.resolve()
```

- [ ] **Step 4: Run GREEN**

Run the Step 2 command. Expected: all tests pass.

---

### Task 2: Instrument the live single-worker state machine

**Files:**
- Modify: `pipeline/bmw_lab_eight_view_demo.py`
- Modify: `tests/unit/pipeline/test_bmw_lab_eight_view_demo.py`

**Interfaces:**
- Consumes: `LiveCycleTiming`, `persist_cycle_timing` from Task 1.
- Produces: `_TimedRoundResult`, `_TimedFinalResult`, exact live timestamps, one persisted timing record per displayed live result.

- [ ] **Step 1: Add a deterministic failing orchestration test**

Use a thread-safe controlled nanosecond clock and the existing fake camera/model pattern. The assertions must prove:

```python
assert timing.front_capture_ms == 100.0
assert timing.flip_wait_ms == 200.0
assert timing.back_capture_ms == 100.0
assert timing.front_inference_ms == 500.0
assert timing.back_inference_ms == 300.0
assert timing.finalize_ms == 50.0
assert timing.persist_ms == 150.0
assert timing.result_display_ms >= 0.0
assert timing.total_cycle_ms == displayed_ns_delta_ms
assert timing.front_overlap_ms == expected_interval_intersection_ms
assert timing_write_calls == [(config.result_root, timing)]
```

The test must also assert that both camera calls and every HighGUI call remain on the main thread, while both inference rounds, finalize, and normal inspection persistence remain on the same single worker thread.

- [ ] **Step 2: Add failing Reset coverage**

Extend the existing generation test so an old completed future cannot call `persist_cycle_timing` and cannot attach a timing record to the new UI state.

- [ ] **Step 3: Run RED**

```bash
UV_CACHE_DIR=/tmp/bmw-cycle-timing-uv-cache uv run --project /home/yunjing/anomaly_xingtao_new --no-sync python -m pytest -q tests/unit/pipeline/test_bmw_lab_eight_view_demo.py
```

Expected: new assertions fail because the pipeline has no timed worker wrappers or cycle timing writer.

- [ ] **Step 4: Add private timed worker results**

```python
@dataclass(frozen=True, slots=True)
class _TimedRoundResult:
    value: object
    started_ns: int
    finished_ns: int


@dataclass(frozen=True, slots=True)
class _TimedFinalResult:
    inspection: object
    finalize_started_ns: int
    finalize_finished_ns: int
    persist_started_ns: int
    persist_finished_ns: int


def _inspect_round_timed(models, images, round_name, clock_ns) -> _TimedRoundResult:
    started_ns = clock_ns()
    value = models.inspect_round(images, round_name)
    return _TimedRoundResult(value, started_ns, clock_ns())
```

Change `_finalize_and_persist_live` to measure `models.finalize_rounds` and `persist_inspection` separately and return `_TimedFinalResult`. Pass only `.value` from each timed round into `finalize_rounds`.

- [ ] **Step 5: Record main-thread marks**

Extend `_LiveGuiWork` with:

```python
cycle_started_ns: int
cycle_started_at: str
front_capture_finished_ns: int
back_capture_started_ns: int | None = None
back_capture_finished_ns: int | None = None
```

Add injectable `_run_gui(..., clock_ns=perf_counter_ns, wall_clock=None)` parameters. Record the cycle start immediately before front `capture_round`, front completion immediately after it returns, back start immediately before back `capture_round`, and back completion immediately after it returns.

- [ ] **Step 6: Build timing after first RESULT frame**

When the final future completes, transition to RESULT as before and retain a pending completed work item. After `cv2.imshow()` returns for that first RESULT frame:

```python
displayed_ns = clock_ns()
timing = LiveCycleTiming(
    capture_id=completed.capture_id,
    started_at=completed.cycle_started_at,
    displayed_at=wall_clock().astimezone().isoformat(timespec="milliseconds"),
    front_capture_ms=_ms(completed.front_capture_finished_ns - completed.cycle_started_ns),
    flip_wait_ms=_ms(completed.back_capture_started_ns - completed.front_capture_finished_ns),
    back_capture_ms=_ms(completed.back_capture_finished_ns - completed.back_capture_started_ns),
    front_inference_ms=_ms(front.finished_ns - front.started_ns),
    back_inference_ms=_ms(back.finished_ns - back.started_ns),
    finalize_ms=_ms(final.finalize_finished_ns - final.finalize_started_ns),
    persist_ms=_ms(final.persist_finished_ns - final.persist_started_ns),
    result_display_ms=_ms(displayed_ns - final.persist_finished_ns),
    front_overlap_ms=_overlap_ms(
        front.started_ns,
        front.finished_ns,
        completed.front_capture_finished_ns,
        completed.back_capture_finished_ns,
    ),
    total_cycle_ms=_ms(displayed_ns - completed.cycle_started_ns),
)
persist_cycle_timing(config.result_root, timing)
state = replace(state, cycle_timing=timing)
```

If the timing write raises, move the next frame to `DemoUiPhase.ERROR` with the write error. Clear the pending work after exactly one write.

- [ ] **Step 7: Run GREEN**

Run the Step 3 command. Expected: all pipeline tests pass without camera/GPU access.

---

### Task 3: Display exact timing without altering inspection semantics

**Files:**
- Modify: `src/bmw_inspection/lab/eight_view_demo_ui.py`
- Modify: `tests/unit/bmw_inspection/lab/test_eight_view_demo_ui.py`

**Interfaces:**
- Consumes: optional `LiveCycleTiming` on `EightViewUiState`.
- Produces: RESULT status card with complete-cycle time and a compact phase breakdown.

- [ ] **Step 1: Write the failing renderer test**

Create a RESULT state with a `LiveCycleTiming` fixture, render it, and monkeypatch the text helper/font drawing boundary to capture strings. Assert that rendered text contains:

```python
assert "完整周期 12.000 s" in rendered_text
assert "采集 1.200/1.300" in rendered_text
assert "翻面 2.500" in rendered_text
assert "推理 3.900/3.800" in rendered_text
assert "融合 0.200" in rendered_text
assert "保存 5.100" in rendered_text
```

Also render a legacy RESULT state with `cycle_timing=None` and assert its existing status text remains unchanged.

- [ ] **Step 2: Run RED**

```bash
UV_CACHE_DIR=/tmp/bmw-cycle-timing-uv-cache uv run --project /home/yunjing/anomaly_xingtao_new --no-sync python -m pytest -q tests/unit/bmw_inspection/lab/test_eight_view_demo_ui.py
```

Expected: the timing field/text does not exist.

- [ ] **Step 3: Implement the minimal UI field and two timing lines**

```python
@dataclass(frozen=True, slots=True)
class EightViewUiState:
    # existing fields unchanged
    cycle_timing: LiveCycleTiming | None = None
```

For RESULT with timing, replace the ordinary detail with `完整周期 X.XXX s` and render one smaller line inside the status card containing capture F/B, flip, inference F/B, finalize, and persist seconds. Do not modify branch cards, evidence panels, scores, thresholds, or final status.

- [ ] **Step 4: Run GREEN**

Run the Step 2 command. Expected: all UI tests pass.

---

### Task 4: Focused verification and handoff memory

**Files:**
- Modify: `AGENTS_MEMORY.md`

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: verified headless timing implementation and a concise local handoff record.

- [ ] **Step 1: Run the focused suite**

```bash
UV_CACHE_DIR=/tmp/bmw-cycle-timing-uv-cache PYTHONPATH=src uv run --project /home/yunjing/anomaly_xingtao_new --no-sync python -m pytest -q \
  tests/unit/bmw_inspection/lab/test_live_cycle_timing.py \
  tests/unit/pipeline/test_bmw_lab_eight_view_demo.py \
  tests/unit/bmw_inspection/lab/test_eight_view_demo_ui.py \
  tests/unit/bmw_inspection/lab/test_eight_view_demo_models.py \
  tests/unit/bmw_inspection/lab/test_eight_view_demo_persistence.py
```

Expected: all selected tests pass. Do not run camera/GPU inference in this verification step.

- [ ] **Step 2: Run syntax and whitespace checks**

```bash
UV_CACHE_DIR=/tmp/bmw-cycle-timing-uv-cache uv run --project /home/yunjing/anomaly_xingtao_new --no-sync python -m py_compile \
  pipeline/bmw_lab_eight_view_demo.py \
  src/bmw_inspection/lab/live_cycle_timing.py \
  src/bmw_inspection/lab/eight_view_demo_ui.py
git diff --check
```

Expected: both commands exit 0.

- [ ] **Step 3: Update local memory**

Append a dated `BMW 完整周期计时` section to `AGENTS_MEMORY.md` documenting the exact boundary, fields, `cycle_timing.json` location, verification command/result, and that one new live camera cycle is still required for real measured values.

- [ ] **Step 4: Field measurement handoff**

Start the requested left or right live config only when explicitly requested. After the operator completes one part, read `<result_root>/<capture_id>/cycle_timing.json` and report every phase plus `total_cycle_ms`; do not substitute offline replay timings for camera measurements.

## Plan self-review

- Coverage: includes all approved phase fields, first-Space/first-RESULT boundary, UI, persistence, Reset isolation, offline compatibility, and focused verification.
- Placeholder scan: complete; no unresolved markers remain.
- Type consistency: `LiveCycleTiming` and `persist_cycle_timing` names and fields are identical across all tasks.
- Scope: detector/model/config code is untouched; only timing orchestration, timing presentation, tests, and local memory are in scope.
