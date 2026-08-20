# BMW Bright-Streak Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fast local Demo that captures one frame from Hikvision serial `DA9625347` and reports `OK`, `NG_NO_STREAK`, `NG_BROKEN`, or `ERROR` using explainable OpenCV rules.

**Architecture:** Keep the BMW code in a new `src/bmw_inspection` package, separate the pure detector from camera and UI side effects, and expose two thin pipeline commands for ROI selection and live/offline inspection. Reuse the existing serial-bound MVS adapter behaviors without importing ZS32 product contracts.

**Tech Stack:** Python 3.10+, OpenCV, NumPy, Hikvision MVS Python SDK, pytest, uv.

**Implementation status (2026-08-05):** Implemented. The focused BMW suite and both approved offline image smokes
pass; the remaining site-only acceptance item is a real USB/MVS capture from camera `DA9625347`.

## Global Constraints

- Bind only by serial `DA9625347`; never fall back to device index.
- Use one software-triggered Mono8 frame per inspection and no live preview.
- The fixture, part, camera, and light are fixed.
- Use traditional vision only; do not load deep-learning models.
- `ERROR` must never be reported as a part-level NG.
- Preserve all unrelated staged, unstaged, and untracked worktree changes.
- Keep verification focused on the new BMW package and two approved images.

---

### Task 1: Configuration, contracts, and ROI selector

**Files:**
- Create: `src/bmw_inspection/__init__.py`
- Create: `src/bmw_inspection/contracts.py`
- Create: `src/bmw_inspection/roi_selector.py`
- Create: `configs/bmw/bright_streak_demo.json`
- Create: `pipeline/bmw_select_bright_streak_roi.py`
- Create: `tests/unit/bmw_inspection/test_contracts.py`
- Create: `tests/unit/bmw_inspection/test_roi_selector.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `DemoStatus`, `BrightStreakConfig`, `BrightStreakMetrics`, `BrightStreakResult`, `load_config(path: Path)`, `save_roi(path: Path, roi_xyxy: tuple[int, int, int, int])`, and `select_roi(image_path: Path, config_path: Path) -> tuple[int, int, int, int]`.

- [ ] **Step 1: Write failing config and ROI persistence tests**

Test exact status values, finite threshold validation, half-open ROI bounds, null ROI rejection at detection time, and atomic preservation of all non-ROI JSON fields.

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest -q tests/unit/bmw_inspection/test_contracts.py tests/unit/bmw_inspection/test_roi_selector.py`

Expected: import failure for `bmw_inspection`.

- [ ] **Step 3: Implement the minimal contracts and ROI writer**

Use frozen dataclasses and strict JSON validation. `save_roi` must write a temporary sibling file and replace the configuration only after valid JSON is complete. The selector uses `cv2.selectROI`, converts `(x, y, w, h)` to half-open XYXY, and requires Enter confirmation.

- [ ] **Step 4: Add the initial Demo configuration**

Set `camera_serial=DA9625347`, `image_width=4024`, `image_height=3036`, `gain=0`, `timeout_ms=3000`, `warmup_frames=1`, result root `results/bmw_bright_streak_demo`, and seed ROI `[1825, 1290, 1870, 1405]` around the user-confirmed central mark. Keep all rule thresholds editable so the ROI tool can replace the seed coordinates after the user draws the exact region.

- [ ] **Step 5: Run the focused tests**

Expected: all Task 1 tests pass.

### Task 2: Pure offline bright-streak detector

**Files:**
- Create: `src/bmw_inspection/detector.py`
- Create: `tests/unit/bmw_inspection/test_detector.py`

**Interfaces:**
- Consumes: `BrightStreakConfig`, `BrightStreakMetrics`, `BrightStreakResult`, and `DemoStatus`.
- Produces: `detect_bright_streak(image: np.ndarray, config: BrightStreakConfig) -> BrightStreakResult` and `render_evidence(image: np.ndarray, result: BrightStreakResult) -> np.ndarray`.

- [ ] **Step 1: Write failing tests for both approved BMP images**

Require `Image_20260805160156065.bmp -> OK` and `Image_20260805160207779.bmp -> NG_NO_STREAK`. Add deterministic synthetic arrays for `NG_BROKEN`, wrong dimensions, clipped image, and repeated-run equality.

- [ ] **Step 2: Run detector tests and confirm they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest -q tests/unit/bmw_inspection/test_detector.py`

Expected: import failure for `bmw_inspection.detector`.

- [ ] **Step 3: Implement quality gates and ROI response**

Convert BGR input to gray when needed; validate dimensions; calculate mean, clipping ratios, and Laplacian variance; crop the configured ROI; remove low-frequency background; calculate white top-hat response and a median/MAD threshold.

- [ ] **Step 4: Implement component filtering and continuity metrics**

Filter by area, width, center corridor, and vertical aspect; project the mask along the streak axis; fill only configured micro-gaps; calculate contrast SNR, coverage, longest run, maximum gap, gap count, mean width, and lateral offset.

- [ ] **Step 5: Implement fail-closed decision precedence and evidence**

Return `ERROR` for invalid input/quality, then `NG_NO_STREAK` for insufficient contrast/coverage, then `NG_BROKEN` for excessive gaps/insufficient longest run, otherwise `OK`. Evidence must draw the ROI, accepted streak in green, and gaps in red.

- [ ] **Step 6: Tune only the Demo JSON against the two approved images**

Do not branch on filenames or reference-image identity. Adjust ROI and thresholds until the two approved labels pass while the synthetic broken-streak test remains `NG_BROKEN`.

- [ ] **Step 7: Run detector tests**

Expected: all Task 2 tests pass.

### Task 3: Serial-bound single-camera capture

**Files:**
- Create: `src/bmw_inspection/camera.py`
- Create: `tests/unit/bmw_inspection/test_camera.py`

**Interfaces:**
- Consumes: `BrightStreakConfig`.
- Produces: `select_unique_device(devices: Sequence[DeviceDescription], serial: str) -> DeviceDescription` and context manager `SingleCameraSession(config)` with `capture() -> np.ndarray`.

- [ ] **Step 1: Write failing adapter-driven tests**

Cover unique serial selection, missing/duplicate serial, one warm-up plus one accepted frame, trigger/read timeout propagation, and reverse-order cleanup after partial failure.

- [ ] **Step 2: Run camera tests and confirm they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest -q tests/unit/bmw_inspection/test_camera.py`

Expected: import failure for `bmw_inspection.camera`.

- [ ] **Step 3: Implement the camera session**

Reuse `HikvisionAdapter`, `open_cameras`, and `capture_single_round` from `capture_data.collect_multicamera_dataset`, but select exactly one enumerated device by serial. Open once, lock exposure/gain, discard configured warm-up frames, and return exactly one image per `capture()`.

- [ ] **Step 4: Run camera tests**

Expected: all Task 3 tests pass without requiring the real SDK or USB hardware.

### Task 4: Demo controller, evidence publication, and entrypoints

**Files:**
- Create: `src/bmw_inspection/demo_app.py`
- Create: `pipeline/bmw_bright_streak_demo.py`
- Create: `tests/unit/bmw_inspection/test_demo_app.py`
- Update: `AGENTS_MEMORY.md`

**Interfaces:**
- Consumes: `load_config`, `SingleCameraSession`, `detect_bright_streak`, and `render_evidence`.
- Produces: `run_offline(image_path: Path, config_path: Path, output_root: Path | None = None) -> BrightStreakResult`, atomic result publication, OpenCV local UI, and CLI `--image` offline fallback.

- [ ] **Step 1: Write failing controller and publication tests**

Cover offline OK/NG execution, required evidence filenames, valid finite JSON, atomic publication, explicit error result, and keyboard mapping for Space/R/Q.

- [ ] **Step 2: Run controller tests and confirm they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest -q tests/unit/bmw_inspection/test_demo_app.py`

Expected: import failure for `bmw_inspection.demo_app`.

- [ ] **Step 3: Implement atomic result publication and offline CLI**

Write `source.png`, `roi.png`, `response.png`, `mask.png`, `evidence.png`, and `result.json` into a temporary directory, then rename it to `results/bmw_bright_streak_demo/YYYYMMDD/<timestamp>`.

- [ ] **Step 4: Implement the minimal 1600 x 900 OpenCV UI**

Show camera status, source with yellow ROI, magnified evidence, coverage/longest-run/max-gap/contrast metrics, a large colored status, and drawn click regions for Capture/Retry/Quit. No live preview or additional settings screen.

- [ ] **Step 5: Add the live/offline pipeline wrapper**

Support `--config`, `--image` for offline testing, `--no-gui`, and `--save-screenshot`. Without `--image`, open `DA9625347` and reuse the same detector and publisher.

- [ ] **Step 6: Run the minimal acceptance suite**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest -q \
  tests/unit/bmw_inspection/test_contracts.py \
  tests/unit/bmw_inspection/test_roi_selector.py \
  tests/unit/bmw_inspection/test_detector.py \
  tests/unit/bmw_inspection/test_camera.py \
  tests/unit/bmw_inspection/test_demo_app.py
```

Expected: all BMW tests pass.

- [ ] **Step 7: Run two offline smoke commands**

Run the pipeline once for each approved BMP with `--no-gui`; expect `OK` for `160156065` and `NG_NO_STREAK` for `160207779`, each with a result directory and evidence image.

- [ ] **Step 8: Run syntax and whitespace checks**

Run `uv run --no-sync python -m py_compile` on only the new Python files and `git diff --check` on only BMW files plus the scoped `pyproject.toml`/memory edits.
