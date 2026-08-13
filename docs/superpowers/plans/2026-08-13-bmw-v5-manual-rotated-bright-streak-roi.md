# BMW V5 Manual Rotated Bright-Streak ROI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace V5's accidental raw-profile V2 light-streak branch with tracked-profile V3 operating on an operator-selected inclined quadrilateral, rectified to `81x613`, while retaining the existing V3 thresholds unchanged.

**Architecture:** A new focused module owns the immutable rotated-ROI asset and perspective rectification. The existing V3 predictor gains an optional manual-ROI adapter but keeps its report, geometry, thresholds, and fixed-ROI rollback path unchanged. V5 alone binds the manual ROI and its SHA-256; other experiment configs keep their current behavior.

**Tech Stack:** Python 3.11+, OpenCV, NumPy, dataclasses, JSON/SHA-256 asset binding, pytest, uv.

## Global Constraints

- Work only in `/home/yunjing/anomaly_xingtao_new/.worktrees/bmw-eight-view-handoff`.
- Preserve all unrelated dirty EfficientAD and Template changes; never reset or stage them with broad commands.
- The operator selects four source-image points in order: upper-left, upper-right, lower-right, lower-left.
- Rectified output is exactly width `81`, height `613`.
- Reuse V3 geometry and thresholds from `bmw_right_batch_20260810_21_bright_v3_tracked_v8/report.json`; do not fit or edit thresholds.
- The rotated candidate must explicitly say that thresholds were not recalibrated.
- Do not modify Template, YOLO, EfficientAD, HDR capture, common part ROIs, or final fusion semantics.
- The existing fixed-ROI `tracked_profile_v3` path remains valid as rollback.

---

### Task 1: Rotated ROI asset and perspective rectification

**Files:**
- Create: `src/bmw_inspection/lab/bright_streak_rotated_roi.py`
- Create: `tests/unit/bmw_inspection/lab/test_bright_streak_rotated_roi.py`

**Interfaces:**
- Produces: `RotatedBrightStreakRoi(points_xy, source_width, source_height, output_width, output_height, source_image, source_image_sha256)`.
- Produces: `rectify_bright_streak_roi(image, asset) -> np.ndarray`.
- Produces: `load_rotated_bright_streak_roi(path, expected_sha256=...) -> RotatedBrightStreakRoi`.
- Produces: `write_rotated_bright_streak_roi(path, asset) -> Path` with no-overwrite publication.

- [ ] **Step 1: Write failing validation and rectification tests**

```python
def test_rectify_manual_quadrilateral_to_v3_shape() -> None:
    image = np.zeros((100, 120), dtype=np.uint8)
    image[10:90, 40:70] = 180
    asset = RotatedBrightStreakRoi(
        points_xy=((40, 10), (70, 15), (65, 90), (35, 85)),
        source_width=120,
        source_height=100,
        output_width=81,
        output_height=613,
        source_image="sample.png",
        source_image_sha256="a" * 64,
    )
    rectified = rectify_bright_streak_roi(image, asset)
    assert rectified.shape == (613, 81)
    assert rectified.dtype == np.uint8


def test_rejects_non_convex_or_out_of_bounds_points() -> None:
    with pytest.raises(ValueError, match="凸四边形"):
        RotatedBrightStreakRoi(
            points_xy=((10, 10), (50, 50), (50, 10), (10, 50)),
            source_width=100,
            source_height=100,
            output_width=81,
            output_height=613,
            source_image="sample.png",
            source_image_sha256="a" * 64,
        )
```

- [ ] **Step 2: Run the focused test and verify it fails because the module is absent**

```bash
env UV_CACHE_DIR=/tmp/bmw-uv-cache uv run --no-sync pytest -q tests/unit/bmw_inspection/lab/test_bright_streak_rotated_roi.py
```

Expected: collection fails with `ModuleNotFoundError: bmw_inspection.lab.bright_streak_rotated_roi`.

- [ ] **Step 3: Implement the asset contract and rectifier**

```python
@dataclass(frozen=True, slots=True)
class RotatedBrightStreakRoi:
    points_xy: tuple[tuple[int, int], ...]
    source_width: int
    source_height: int
    output_width: int
    output_height: int
    source_image: str
    source_image_sha256: str


def rectify_bright_streak_roi(
    image: np.ndarray,
    asset: RotatedBrightStreakRoi,
) -> np.ndarray:
    destination = np.array(
        [[0, 0], [80, 0], [80, 612], [0, 612]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(
        np.asarray(asset.points_xy, dtype=np.float32),
        destination,
    )
    return cv2.warpPerspective(
        image,
        matrix,
        (asset.output_width, asset.output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
```

Validation must require four in-bounds clockwise points forming a convex non-degenerate quadrilateral, output `81x613`, and lowercase 64-character SHA-256. The loader must require schema `bmw.bright_streak_rotated_roi/1.0`, exact JSON fields, matching asset/source hashes, and matching image dimensions. The writer must reject an existing destination and atomically publish one JSON object.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run the Step 2 command. Expected: all tests in `test_bright_streak_rotated_roi.py` pass.

- [ ] **Step 5: Commit only Task 1 files**

```bash
git add src/bmw_inspection/lab/bright_streak_rotated_roi.py tests/unit/bmw_inspection/lab/test_bright_streak_rotated_roi.py
git commit -m "feat: add rotated bright-streak ROI asset"
```

---

### Task 2: Four-click manual selector

**Files:**
- Create: `pipeline/bmw_lab_select_bright_streak_rotated_roi.py`
- Create: `tests/unit/pipeline/test_bmw_lab_select_bright_streak_rotated_roi.py`

**Interfaces:**
- Consumes the Task 1 asset, rectifier, and writer.
- Produces CLI options `--image`, `--output`, `--max-display-width`, and `--max-display-height`.
- Produces helper `_source_point(display_point, display_shape, source_shape) -> tuple[int, int]`.

- [ ] **Step 1: Write failing CLI helper tests**

```python
def test_source_point_maps_scaled_display_to_original() -> None:
    assert _source_point((320, 180), (720, 1280), (3036, 4024)) == (1006, 759)


def test_parser_defaults_to_the_confirmed_broken_capture() -> None:
    args = build_parser().parse_args([])
    assert args.image.name == "front_left_hdr.png"
    assert "bmw_demo_20260813_164043" in str(args.image)
```

- [ ] **Step 2: Run tests and verify they fail because the selector is absent**

```bash
env UV_CACHE_DIR=/tmp/bmw-uv-cache uv run --no-sync pytest -q tests/unit/pipeline/test_bmw_lab_select_bright_streak_rotated_roi.py
```

Expected: collection fails with `ModuleNotFoundError` for the new pipeline module.

- [ ] **Step 3: Implement the selector loop**

```python
DEFAULT_IMAGE = REPO_ROOT / (
    "results/bmw_eight_view_demo_v5_template_manual_ignore_mask_v1/"
    "bmw_demo_20260813_164043/images/front_left_hdr.png"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "results/bmw_bright_streak_rotated_roi/"
    "bmw_demo_20260813_164043_v1/roi.json"
)
```

The mouse callback appends at most four displayed-bitmap points and maps them directly to source coordinates. Draw lines and labels `1-4`; after point four, show the rectified preview. `Enter` writes the asset, `R` clears points, and `Esc` or `Q` exits without writing. Print Chinese instructions, final points, and asset SHA-256.

- [ ] **Step 4: Run selector unit tests**

Run the Step 2 command. Expected: all selector tests pass without opening a GUI.

- [ ] **Step 5: Commit only Task 2 files**

```bash
git add pipeline/bmw_lab_select_bright_streak_rotated_roi.py tests/unit/pipeline/test_bmw_lab_select_bright_streak_rotated_roi.py
git commit -m "feat: select manual rotated bright-streak ROI"
```

---

### Task 3: V3 runtime adapter and V5 configuration contract

**Files:**
- Modify: `src/bmw_inspection/lab/eight_view_demo.py`
- Modify: `src/bmw_inspection/lab/eight_view_demo_models.py`
- Modify: `src/bmw_inspection/lab/eight_view_demo_ui.py`
- Modify: `tests/unit/bmw_inspection/lab/test_eight_view_demo.py`
- Modify: `tests/unit/bmw_inspection/lab/test_eight_view_demo_models.py`
- Modify: `tests/unit/bmw_inspection/lab/test_eight_view_demo_ui.py`

**Interfaces:**
- Extends `EightViewDemoConfig` with optional rotated-ROI path and SHA fields.
- Adds engine `tracked_profile_v3_manual_rotated_roi`.
- Extends `EightViewTrackedProfileBrightStreakPredictor(report_path, *, rotated_roi_path=None, rotated_roi_sha256=None)`.
- Extends UI reference extraction to rectify the same quadrilateral from a trusted full reference.

- [ ] **Step 1: Add failing config and predictor tests**

```python
def test_rotated_v3_reuses_report_thresholds() -> None:
    predictor = EightViewTrackedProfileBrightStreakPredictor(
        v3_report,
        rotated_roi_path=roi_asset,
        rotated_roi_sha256=roi_sha256,
    )
    result = predictor.predict(front_left_hdr)
    assert result.details["roi_mode"] == "manual_rotated_perspective"
    assert result.details["threshold_calibration"] == "existing_v3_not_recalibrated"
    assert result.overlay.shape[:2] == (613, 81)


def test_rotated_v3_rejects_roi_sha_mismatch() -> None:
    with pytest.raises(ValueError, match="倾斜光痕ROI.*SHA256"):
        EightViewTrackedProfileBrightStreakPredictor(
            v3_report,
            rotated_roi_path=roi_asset,
            rotated_roi_sha256="0" * 64,
        )
```

Add a config-loader case whose `bright_streak` block contains exactly `engine`, `config`, `config_sha256`, `rotated_roi`, and `rotated_roi_sha256`. Assert legacy V3 remains accepted and partial rotated fields are rejected.

- [ ] **Step 2: Run focused tests and confirm the new cases fail**

```bash
env UV_CACHE_DIR=/tmp/bmw-uv-cache uv run --no-sync pytest -q   tests/unit/bmw_inspection/lab/test_eight_view_demo.py   tests/unit/bmw_inspection/lab/test_eight_view_demo_models.py   tests/unit/bmw_inspection/lab/test_eight_view_demo_ui.py
```

Expected: only new rotated-ROI cases fail.

- [ ] **Step 3: Add the minimal runtime connection**

For the new engine, resolve and SHA-check both assets in `load_demo_config`. Keep existing V3 report validation, geometry, and thresholds. Replace only the fixed crop:

```python
if self._rotated_roi is None:
    roi_gray = _gray(image)[y1:y2, x1:x2]
else:
    rectified = rectify_bright_streak_roi(image, self._rotated_roi)
    roi_gray = _gray(rectified)
```

For the rotated case, prefix the reason with `手动倾斜ROI，沿用V3阈值（未重标定）` and publish `roi_points_xy`, `roi_mode`, `threshold_calibration`, and `rotated_roi_sha256`. In the UI, rectify the trusted full reference when `roi_points_xy` exists; retain fixed `roi_xyxy` slicing otherwise.

- [ ] **Step 4: Run focused tests and verify they pass**

Run the Step 2 command. Expected: all selected tests pass.

- [ ] **Step 5: Commit intended Task 3 hunks only**

Review `git diff` first. Stage only intended hunks because these files contain unrelated EfficientAD/Template work. Commit message:

```text
feat: run V3 on manual rotated bright-streak ROI
```

---

### Task 4: Select ROI, bind V5, and replay the broken sample

**Files:**
- Create at runtime: `results/bmw_bright_streak_rotated_roi/bmw_demo_20260813_164043_v1/roi.json`
- Modify: `configs/bmw/experiments/bmw_eight_view_demo_v5_template_manual_ignore_mask_v1.json`
- Modify: `AGENTS_MEMORY.md`

**Interfaces:**
- Consumes Task 2 selector and Task 3 engine.
- Produces a SHA-bound V5 candidate and offline replay evidence.

- [ ] **Step 1: Launch selector**

```bash
env UV_CACHE_DIR=/tmp/bmw-uv-cache uv run --no-sync python   pipeline/bmw_lab_select_bright_streak_rotated_roi.py
```

Expected: operator clicks upper-left, upper-right, lower-right, lower-left and presses `Enter`. Selector prints the no-overwrite ROI JSON path and its SHA-256.

- [ ] **Step 2: Bind V5 to the actual selector output**

Replace only V5's `bright_streak` block with engine `tracked_profile_v3_manual_rotated_roi`, V3 report path/SHA `6b43690af67702333646fa7a88a2a5053a0afae6d19cb343bf6d3abb193c17d4`, rotated ROI path, and the exact 64-character digest printed by the selector. The JSON must not be written until that real digest is available.

- [ ] **Step 3: Run light-streak-only replay**

Run the configured predictor on `bmw_demo_20260813_164043/images/front_left_hdr.png`. Expected: branch status `NG`, detail decision `NG_BROKEN`, and reason includes the unrecalibrated-threshold label. If not `NG_BROKEN`, do not alter thresholds; publish a new versioned ROI selection.

- [ ] **Step 4: Run V5 offline Demo and focused regression**

```bash
env UV_CACHE_DIR=/tmp/bmw-uv-cache MPLCONFIGDIR=/tmp/bmw-mpl-cache   uv run --no-sync python pipeline/bmw_lab_eight_view_demo.py   --config configs/bmw/experiments/bmw_eight_view_demo_v5_template_manual_ignore_mask_v1.json   --offline-dir results/bmw_eight_view_demo_v5_template_manual_ignore_mask_v1/bmw_demo_20260813_164043   --no-gui
```

A business-NG exit is acceptable. The persisted light-streak row must be `NG_BROKEN`, and the inspection must contain exactly 25 result rows.

```bash
env UV_CACHE_DIR=/tmp/bmw-uv-cache uv run --no-sync pytest -q   tests/unit/bmw_inspection/lab/test_bright_streak_rotated_roi.py   tests/unit/pipeline/test_bmw_lab_select_bright_streak_rotated_roi.py   tests/unit/bmw_inspection/lab/test_eight_view_demo.py   tests/unit/bmw_inspection/lab/test_eight_view_demo_models.py   tests/unit/bmw_inspection/lab/test_eight_view_demo_ui.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Update project memory and commit the V5 binding**

Append actual ROI points, ROI SHA-256, V3 report SHA-256, replay outcome, and the explicit `not recalibrated` limitation to `AGENTS_MEMORY.md`. Stage only V5 config and that exact memory hunk. Commit message:

```text
feat: bind V5 to manual rotated V3 light streak
```
