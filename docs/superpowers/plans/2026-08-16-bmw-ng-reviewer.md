# BMW NG Interactive Reviewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Tkinter reviewer that groups the existing Template and EfficientAD NG cases by part and atomically persists a per-problem human decision.

**Architecture:** A small domain module owns CSV validation, grouping, decision updates, batch updates, progress calculation, and atomic persistence. A separate pipeline entrypoint owns Tkinter widgets, image rendering, navigation, keyboard shortcuts, and Chinese error dialogs. The detector outputs and model configuration remain read-only.

**Tech Stack:** Python 3.11+, standard-library `csv`/`tkinter`, Pillow `Image`/`ImageTk`, pytest, uv.

## Global Constraints

- Decisions are per problem, not per part.
- Allowed values are empty, `误判`, `真实缺陷`, and `不确定`.
- Every click auto-saves to `review_cases.csv`; unknown columns and row order are preserved.
- Batch actions affect only undecided problems in the current part.
- No model, threshold, ROI, `inspection.json`, or evidence image may be modified.

---

### Task 1: Review data model and atomic CSV persistence

**Files:**
- Create: `src/bmw_inspection/lab/ng_review.py`
- Create: `tests/unit/bmw_inspection/lab/test_ng_review.py`

**Interfaces:**
- Produces: `ReviewDataset.load(path: Path) -> ReviewDataset`, `groups`, `set_decision(case_id, decision)`, `set_note(case_id, note)`, `set_remaining_for_capture(capture_id, decision)`, `progress`, and `save()`.
- Consumes: the existing review CSV with `case_id`, `capture_id`, `branch`, `view_id`, `decision`, `review_note`, and `panel_path` fields.

- [x] **Step 1: Write failing tests**

Create fixtures with two captures and unknown columns. Assert grouping order, allowed decision validation, batch updates preserving existing decisions, progress counts, UTF-8 CSV round-trip, row order, and unknown-column preservation.

- [x] **Step 2: Verify RED**

Run: `env UV_CACHE_DIR=/tmp/bmw-uv-cache uv run --no-sync pytest -q tests/unit/bmw_inspection/lab/test_ng_review.py`

Expected: collection fails because `bmw_inspection.lab.ng_review` does not exist.

- [x] **Step 3: Implement the minimal domain module**

Use immutable CSV field names, mutable row dictionaries, stable capture grouping, strict decision validation, and `tempfile.NamedTemporaryFile(dir=csv_path.parent)` followed by `os.replace()`.

- [x] **Step 4: Verify GREEN**

Run the Task 1 pytest command. Expected: all tests pass.

### Task 2: Tkinter reviewer and launcher

**Files:**
- Create: `pipeline/bmw_lab_review_ng_cases.py`
- Modify: `AGENTS_MEMORY.md`
- Test: `tests/unit/bmw_inspection/lab/test_ng_review.py`

**Interfaces:**
- Consumes: `ReviewDataset` from Task 1 and a `--review-csv PATH` CLI option.
- Produces: `main(argv: Sequence[str] | None = None) -> int`, a 46-part sidebar, per-problem decision controls, batch actions, navigation shortcuts, image zoom, and autosave status.

- [x] **Step 1: Add failing CLI/default-path tests**

Assert the parser resolves the existing left review package by default and accepts an explicit CSV path without constructing a Tk root.

- [x] **Step 2: Verify RED**

Run the focused pytest command. Expected: import or parser assertion fails because the launcher does not exist.

- [x] **Step 3: Implement the GUI**

Build a fixed sidebar plus scrollable main canvas. Create one problem card per current capture using the saved four-panel JPEG, radio-style decision buttons, note entry, and current-problem focus. Bind `1/2/3`, arrow keys, `Ctrl+S`, double-click zoom, and `Esc`; call `ReviewDataset.save()` after every decision or note focus-out.

- [x] **Step 4: Verify using real data without opening a window**

Run a `--check` mode that loads the real CSV and prints JSON with `capture_count=46`, `case_count=111`, `template_count=32`, and `efficientad_count=79`.

- [x] **Step 5: Run focused tests and compile checks**

Run:

```bash
env UV_CACHE_DIR=/tmp/bmw-uv-cache uv run --no-sync pytest -q tests/unit/bmw_inspection/lab/test_ng_review.py
env UV_CACHE_DIR=/tmp/bmw-uv-cache uv run --no-sync python -m compileall -q src/bmw_inspection/lab/ng_review.py pipeline/bmw_lab_review_ng_cases.py
```

Expected: tests pass and compileall exits 0.

- [x] **Step 6: Record handoff and commit only task files**

Append the launcher command and package contract to `AGENTS_MEMORY.md`, check the exact staged file list, and commit only the spec correction, plan, domain module, launcher, tests, and memory hunk.
