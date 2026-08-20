# BMW Eight-View Demo Typography Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve Chinese text clarity in the BMW eight-view Demo without changing its accepted layout.

**Architecture:** Keep typography helpers local to `eight_view_demo_ui.py` so the legacy six-view UI remains unchanged. Resolve installed Noto Sans CJK Medium/Bold assets with cached helpers, then apply semantic font weights and slightly larger small text while retaining all panel coordinates.

**Tech Stack:** Python, Pillow, OpenCV, pytest, Noto Sans CJK

## Global Constraints

- Keep the dashboard at exactly 1600×900.
- Do not change panel coordinates, image regions, grid structure, semantic colours, controls, or inspection logic.
- Use Noto Sans CJK Medium for body text and Noto Sans CJK Bold for titles and statuses.

---

### Task 1: Add semantic font resources and apply them

**Files:**
- Modify: `src/bmw_inspection/lab/eight_view_demo_ui.py`
- Modify: `tests/unit/bmw_inspection/lab/test_eight_view_demo_ui.py`
- Modify: `AGENTS_MEMORY.md`

**Interfaces:**
- Produces: `_demo_font(size: int, weight: str = "medium") -> ImageFont.FreeTypeFont`
- Preserves: `render_eight_view_dashboard(state: EightViewUiState) -> np.ndarray`

- [ ] **Step 1: Write a failing font-resource test**

Assert that the local font resolver returns installed, distinct Medium and Bold Noto CJK paths and the renderer remains 1600×900.

- [ ] **Step 2: Run the focused test and verify RED**

Run `UV_CACHE_DIR=/tmp/bmw_demo_uv_cache MPLCONFIGDIR=/tmp/bmw_demo_mpl uv run --no-sync pytest -q tests/unit/bmw_inspection/lab/test_eight_view_demo_ui.py` and expect an import failure for the new font resolver.

- [ ] **Step 3: Implement semantic typography**

Add cached Medium/Bold path and font helpers, use bold for titles/results/statuses, medium for body text, raise small text to 16–18 px, and darken muted text without changing layout coordinates.

- [ ] **Step 4: Verify and regenerate the real screenshot**

Run the focused BMW eight-view tests, compile the modified module, and use `bmw_normal_group072_000001` with `--no-gui --experiment-mode --save-screenshot` to overwrite the comparison screenshot.

- [ ] **Step 5: Update project memory**

Record the font assets, override variables, unchanged-layout boundary, and screenshot verification in `AGENTS_MEMORY.md`.
