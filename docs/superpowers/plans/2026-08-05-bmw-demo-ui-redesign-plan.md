# BMW Chinese Industrial Demo UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the debug-like BMW OpenCV dashboard with a polished 1600x900 fully Chinese industrial presentation UI while preserving inspection behavior.

**Architecture:** Keep `render_dashboard` as a pure NumPy/OpenCV renderer, add small palette/geometry/copy helpers, and use Pillow only for antialiased CJK text drawn back onto the BGR canvas. Preserve the existing detector, camera session, publisher, action enum, keyboard mapping, and click regions.

**Tech Stack:** Python, NumPy, OpenCV, Pillow, Noto Sans CJK SC, pytest, uv.

**Implementation status (2026-08-05):** Implemented. Chinese OK and no-streak NG screenshots were visually checked;
the dashboard is 1600x900, hover rendering is covered, and detector/camera/publication behavior is unchanged.

## Global Constraints

- Render exactly 1600x900.
- All visible copy is Chinese; do not display raw English reasons or filesystem paths.
- Keep `OK`, `NG_NO_STREAK`, `NG_BROKEN`, and `ERROR` business semantics unchanged.
- Preserve Space/R/Q/Escape and existing click actions.
- Add no web, Qt, deep-learning, or network dependency.
- Do not modify detector thresholds or camera behavior.

---

### Task 1: Add testable Chinese presentation and font primitives

**Files:**
- Modify: `src/bmw_inspection/demo_app.py`
- Modify: `tests/unit/bmw_inspection/test_demo_app.py`

**Interfaces:**
- Produces: `StatusPresentation`, `status_presentation(status)`, `format_metric_rows(result, config)`, and an internal antialiased `_draw_text` helper.

- [ ] **Step 1: Write failing presentation tests**

Assert exact Chinese title/status/reason mappings for all four statuses, Chinese metric labels, and that ERROR contrast displays `不可用`.

- [ ] **Step 2: Run the focused tests and confirm missing presentation helpers fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest -q tests/unit/bmw_inspection/test_demo_app.py`

- [ ] **Step 3: Implement immutable copy and metric helpers**

Map OK to `检测通过 / 亮痕存在且连续`, no-streak to `检测不通过 / 未检测到有效亮痕`, broken to `检测不通过 / 亮痕存在但不连续`, and ERROR to `设备或图像异常 / 请检查相机、光源与工件位置`.

- [ ] **Step 4: Implement CJK font loading and antialiased text**

Prefer `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`, allow `BMW_DEMO_FONT` override, render with Pillow, and raise a clear startup error if neither path exists.

- [ ] **Step 5: Run focused tests**

Expected: all Demo presentation tests pass.

### Task 2: Rebuild the 1600x900 dashboard and interactions

**Files:**
- Modify: `src/bmw_inspection/demo_app.py`
- Modify: `tests/unit/bmw_inspection/test_demo_app.py`

**Interfaces:**
- Consumes: existing `BrightStreakResult`, `BrightStreakConfig`, `render_evidence`, and `Action`.
- Produces: `render_dashboard(..., hovered_action: Action | None = None) -> np.ndarray` and mouse hover state in `run_gui`.

- [ ] **Step 1: Write failing layout/render tests**

Assert output shape `(900, 1600, 3)`, deterministic repeated rendering, unchanged click hit regions, and a different button surface when `hovered_action=Action.CAPTURE`.

- [ ] **Step 2: Implement palette, cards, header, and footer**

Use dark navy background, blue-gray cards, cyan dividers, a connection pill, current time, and a footer status. Do not render result paths.

- [ ] **Step 3: Implement source and ROI evidence cards**

Show `实时采集图像` with the yellow ROI and `亮痕检测区域` with a large fitted ROI evidence crop.

- [ ] **Step 4: Implement business result and simplified metric cards**

Render the Chinese status card with semantic colors, progress bars for coverage/continuity, gap pixels with pass/fail text, and SNR with `良好`, `偏低`, or `不可用`.

- [ ] **Step 5: Implement polished buttons and hover feedback**

Keep the same button rectangles and actions, add cyan hover borders/fills, and update hover state through `cv2.EVENT_MOUSEMOVE` without changing click behavior.

- [ ] **Step 6: Run focused Demo tests**

Expected: all Demo controller and rendering tests pass.

### Task 3: Screenshot acceptance and project memory

**Files:**
- Update: `AGENTS_MEMORY.md`

**Interfaces:**
- Consumes: existing `--image`, `--no-gui`, and `--save-screenshot` CLI options.
- Produces: one inspectable 1600x900 screenshot and updated operational memory.

- [ ] **Step 1: Render a headless screenshot from a labeled OK BMP**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python pipeline/bmw_bright_streak_demo.py \
  --image dataset/bmw/OK/Image_20260805172921398.bmp \
  --no-gui --output-root /tmp/bmw-ui-redesign-smoke \
  --save-screenshot /tmp/bmw-ui-redesign.png
```

Expected: CLI prints `OK`; screenshot is exactly 1600x900.

- [ ] **Step 2: Visually inspect the screenshot**

Confirm Chinese glyphs render correctly, no text overlaps, ROI evidence is legible, OK dominates the hierarchy, and buttons are aligned.

- [ ] **Step 3: Run the focused BMW acceptance suite**

Run the five test files under `tests/unit/bmw_inspection`; expect all tests to pass.

- [ ] **Step 4: Run syntax and whitespace checks**

Compile `demo_app.py` and `pipeline/bmw_bright_streak_demo.py`, then run `git diff --check` only on BMW UI files, the plan, and memory.

- [ ] **Step 5: Update project memory**

Record the Chinese title, visual semantics, font path/override, hover behavior, screenshot command, and unchanged detector/runtime boundary.

### Task 4: Rotate the ROI evidence view clockwise without distortion

**Files:**
- Modify: `src/bmw_inspection/demo_app.py`
- Modify: `tests/unit/bmw_inspection/test_demo_app.py`
- Update: `AGENTS_MEMORY.md`

**Interfaces:**
- Produces: `_prepare_roi_evidence(zoom: np.ndarray, width: int, height: int) -> np.ndarray` for display-only clockwise rotation and proportional fitting.

- [ ] **Step 1: Write a failing orientation and aspect-ratio test**

Use a synthetic tall ROI with distinct corner colors. Assert the prepared evidence is exactly the card size, the
top-left source marker moves to the top-right after clockwise rotation, and non-background content retains the
rotated width-to-height ratio within rounding tolerance.

- [ ] **Step 2: Run the focused test and confirm the missing helper fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest -q tests/unit/bmw_inspection/test_demo_app.py`

- [ ] **Step 3: Implement clockwise rotation and proportional fitting**

Call `cv2.rotate(zoom, cv2.ROTATE_90_CLOCKWISE)` and then reuse `_fit` for the fixed evidence viewport. Remove the
existing non-uniform `cv2.resize(zoom, (250, 192))` path.

- [ ] **Step 4: Regenerate OK and NG screenshots**

Expected: the bright streak runs left-to-right, the ROI is not stretched, and all Chinese layout elements remain aligned.

- [ ] **Step 5: Run focused BMW tests and update memory**

Expected: all BMW tests pass; record that rotation is presentation-only and saved artifacts retain camera orientation.

### Task 5: Start live Demo in an idle state

**Files:**
- Modify: `src/bmw_inspection/demo_app.py`
- Modify: `pipeline/bmw_bright_streak_demo.py`
- Modify: `tests/unit/bmw_inspection/test_demo_app.py`
- Update: `AGENTS_MEMORY.md`

**Interfaces:**
- Produces: `render_waiting_dashboard(config, hovered_action=None, camera_connected=True, now=None) -> np.ndarray`, `action_enabled(action, has_result) -> bool`, and `run_gui(...) -> BrightStreakResult | None`.

- [ ] **Step 1: Write failing idle-state and acquisition-timing tests**

Assert retry is disabled before a result, the waiting dashboard is deterministic and 1600x900, and a mocked GUI that
receives Q first returns `None` with zero acquisition calls. Assert Space followed by Q performs exactly one acquisition.

- [ ] **Step 2: Run the focused Demo tests and confirm missing idle behavior fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest -q tests/unit/bmw_inspection/test_demo_app.py`

- [ ] **Step 3: Implement the waiting dashboard and action gate**

Render `等待检测 / 点击“开始检测”或按空格键开始` with empty source/evidence cards, no business result, no metrics,
and a disabled `重新检测` button. Keep Start and Quit enabled.

- [ ] **Step 4: Refactor the live event loop**

Show the waiting dashboard immediately after window creation. Call `acquire()` only after enabled Capture/Retry input;
return `None` when quitting before the first inspection.

- [ ] **Step 5: Update the pipeline exit contract**

Treat a `None` live result as a successful operator exit and do not print a fabricated status.

- [ ] **Step 6: Run focused BMW tests and update memory**

Expected: all BMW tests pass and the startup path has a regression proving zero automatic captures.
