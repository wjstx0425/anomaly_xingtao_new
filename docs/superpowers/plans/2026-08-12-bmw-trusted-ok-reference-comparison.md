# BMW Trusted OK Reference Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a manually certified late-evening OK reference bank, automatically select the closest same-view OK image for every NG/ERROR result, and expose the comparison through the `O` shortcut without changing any detector decision.

**Architecture:** A preparation CLI filters the immutable 21:00 manifest and creates human-review contact sheets. A separate publisher accepts only explicit `APPROVED` decisions and produces a SHA-bound reference index. The Demo loads that index through an optional config block, computes one aligned reference match per actionable view after all 25 checks finish, stores the diagnostic comparison in the inspection record, and lets the UI toggle between HDR-source evidence and trusted-OK comparison.

**Tech Stack:** Python 3.13, uv, OpenCV, NumPy, CSV/JSON, Pillow, pytest, existing BMW eight-view Demo contracts.

## Global Constraints

- Candidate source is only session `20260810_210030_527506`, `split=train`, `source_class=normal`, `business_label=OK`, with all eight canonical views and valid source SHA256.
- `no_streak`, `edge`, `deform`, `others`, every NG row, calibration rows, and final-test rows are ineligible.
- Only explicit human `APPROVED` decisions enter the trusted bank; `PENDING`, blank, unknown, and `REJECTED` never enter.
- A missing or invalid trusted reference must display `无可信OK参考`; never fall back to an unapproved image.
- Reference comparison is diagnostic only and must not modify Template, bright-streak, YOLO, EfficientAD, or final fusion status.
- Existing v1/v2 Demo configs without a trusted-reference block remain loadable with the comparison feature disabled.
- Generated releases are immutable and reject overwrite.

---

### Task 1: Human-review candidate package

**Files:**
- Create: `src/bmw_inspection/lab/trusted_ok_reference.py`
- Create: `pipeline/bmw_lab_prepare_trusted_ok_review.py`
- Create: `tests/unit/bmw_inspection/lab/test_trusted_ok_reference.py`
- Modify: `AGENTS_MEMORY.md`
- Modify: `pipeline/AGENTS_MEMORY.md`

**Interfaces:**
- Consumes: `dataset_manifest.csv` with the current BMW fields and canonical `VIEW_ORDER`.
- Produces: `prepare_review_package(manifest: Path, output_dir: Path, *, session_id: str) -> ReviewPackageSummary` and files `candidate_manifest.csv`, `review_decisions.csv`, `review/contact_sheets/<physical_part_id>.png`, `review_package.json`.

- [ ] **Step 1: Write filtering and immutability tests**

Create fixtures containing one valid eight-view training normal, one `no_streak`, one calibration normal, one incomplete normal, and one source-hash mismatch. Assert that only the valid part produces eight manifest rows, one `PENDING` decision, and one contact sheet. Assert a pre-existing output directory raises `FileExistsError`.

```python
summary = prepare_review_package(manifest, output, session_id="20260810_210030_527506")
assert summary.candidate_part_count == 1
assert summary.candidate_image_count == 8
assert read_decisions(output) == [("normal-train-001", "PENDING")]
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
UV_CACHE_DIR=/tmp/bmw_ok_reference_uv uv run --no-sync pytest \
  tests/unit/bmw_inspection/lab/test_trusted_ok_reference.py -q
```

Expected: collection fails because `trusted_ok_reference.py` and the preparation API do not exist.

- [ ] **Step 3: Implement strict candidate parsing and review rendering**

Implement frozen row/summary dataclasses, canonical-view grouping, file SHA verification, deterministic physical-part ordering, and a 4x2 Pillow contact sheet containing the eight fused source images with Chinese view labels. Write all outputs into a sibling staging directory and publish with `_atomic_publish_noreplace`.

`review_decisions.csv` must contain exactly:

```text
physical_part_id,sample_id,decision,reviewer,review_note
...
```

Every generated decision starts as `PENDING`; the script never approves a part automatically.

- [ ] **Step 4: Add the CLI and generate the real review package**

Default command:

```bash
uv run --no-sync python pipeline/bmw_lab_prepare_trusted_ok_review.py \
  --manifest dataset/bmw_lab_prepared/bmw_right_batch_20260810_21_v1/manifests/dataset_manifest.csv \
  --session-id 20260810_210030_527506 \
  --output dataset/bmw_trusted_ok_review/bmw_right_20260810_21_train_normal_v1
```

Expected summary: `50` candidate physical parts, `400` candidate images, `50` contact sheets, and all decisions `PENDING`.

- [ ] **Step 5: Verify and commit Task 1**

Run the focused test, validate the JSON/CSV counts, run `python -m compileall` on both production files, and run `git diff --check`.

```bash
git add src/bmw_inspection/lab/trusted_ok_reference.py \
  pipeline/bmw_lab_prepare_trusted_ok_review.py \
  tests/unit/bmw_inspection/lab/test_trusted_ok_reference.py \
  AGENTS_MEMORY.md pipeline/AGENTS_MEMORY.md
git commit -m "feat: prepare BMW trusted OK review package"
```

---

### Task 2: Approved whitelist and reference index publisher

**Files:**
- Modify: `src/bmw_inspection/lab/trusted_ok_reference.py`
- Create: `pipeline/bmw_lab_publish_trusted_ok_reference.py`
- Modify: `tests/unit/bmw_inspection/lab/test_trusted_ok_reference.py`

**Interfaces:**
- Consumes: Task 1 package plus human-edited `review_decisions.csv`.
- Produces: `publish_trusted_reference_index(review_dir: Path, output_dir: Path, roi_config: Path) -> TrustedIndexSummary`, `trusted_ok_whitelist.json`, `reference_index.json`, and immutable copied ROI images under `references/<view>/`.

- [ ] **Step 1: Write rejection and publication tests**

Assert that publication fails when any selected row is blank/unknown, an approved source SHA no longer matches, an approved part lacks one view, or no part is approved. Assert `PENDING` and `REJECTED` remain excluded and do not block publication when at least one complete `APPROVED` part exists.

```python
summary = publish_trusted_reference_index(review, release, roi_config)
assert summary.approved_part_count == 1
assert summary.reference_count_by_view == {view: 1 for view in VIEW_ORDER}
```

- [ ] **Step 2: Run the publisher tests and verify RED**

Run the Task 2 test selection and expect failure because the publisher is absent.

- [ ] **Step 3: Implement explicit approval and SHA-bound publication**

Require the exact decisions `APPROVED`, `REJECTED`, or `PENDING`. Crop each approved source using the current eight-view ROI asset, copy the ROI into the immutable release, and record copied-image SHA, source-image SHA, physical part, sample, view, split, session, whitelist SHA, ROI-config SHA, and preprocessing identity.

- [ ] **Step 4: Add the publishing CLI**

The CLI must refuse to run while all decisions remain `PENDING` and must never edit the review package. Its output path is versioned and no-overwrite.

- [ ] **Step 5: Verify and commit Task 2**

Run all trusted-reference tests, compile the module and CLI, validate JSON, and commit only Task 2 files.

---

### Task 3: Runtime trusted-reference matcher

**Files:**
- Modify: `src/bmw_inspection/lab/eight_view_demo.py`
- Modify: `src/bmw_inspection/lab/eight_view_demo_models.py`
- Modify: `src/bmw_inspection/lab/trusted_ok_reference.py`
- Modify: `tests/unit/bmw_inspection/lab/test_eight_view_demo.py`
- Modify: `tests/unit/bmw_inspection/lab/test_eight_view_demo_models.py`
- Modify: `tests/unit/bmw_inspection/lab/test_trusted_ok_reference.py`

**Interfaces:**
- Produces core contract `TrustedOkMatch(view_id, physical_part_id, sample_id, similarity, shift_x, shift_y, current_roi, reference_roi, aligned_reference_roi, difference_overlay, source_sha256, reference_sha256, index_sha256)`.
- Produces `TrustedOkMatcher.match(view_id: str, current_roi: np.ndarray) -> TrustedOkMatch`.
- Extends `EightViewInspection.trusted_ok_by_view: Mapping[str, TrustedOkMatch]` with an empty immutable default.

- [ ] **Step 1: Write matcher and decision-invariance tests**

Use three synthetic approved references and assert the aligned normalized-correlation winner is selected deterministically. Assert only unique NG/ERROR views are matched, PASS-only inspections do no matching, and `final_status/results` are byte-for-byte equivalent with the matcher enabled or disabled.

- [ ] **Step 2: Run focused tests and verify RED**

Expected failures: missing `TrustedOkMatch`, matcher, and inspection field.

- [ ] **Step 3: Implement finite-shift matching and diagnostic overlays**

Reuse the Template preparation contract: grayscale, aspect-preserving 512x512 fit, 3x3 blur, reflected padding, and `TM_CCOEFF_NORMED` within `max_shift=12`. Create a high-contrast absolute-difference overlay with a red contour around the strongest connected difference region, but label it as diagnostic.

- [ ] **Step 4: Integrate after all model decisions**

After the existing 25 results have been constructed and fused, match each unique actionable view against the trusted bank. Matcher errors attach no reference and do not turn PASS/NG into ERROR; record the diagnostic problem separately in inspection metadata.

- [ ] **Step 5: Verify and commit Task 3**

Run the three focused test files plus the existing Template/EfficientAD tests, compile changed modules, and commit.

---

### Task 4: Config, `O` shortcut, comparison UI, and persistence

**Files:**
- Modify: `configs/bmw/experiments/bmw_eight_view_demo_v3_ng_evidence.json`
- Modify: `pipeline/bmw_lab_eight_view_demo.py`
- Modify: `src/bmw_inspection/lab/eight_view_demo.py`
- Modify: `src/bmw_inspection/lab/eight_view_demo_models.py`
- Modify: `src/bmw_inspection/lab/eight_view_demo_ui.py`
- Modify: `src/bmw_inspection/lab/eight_view_demo_persistence.py`
- Modify: `tests/unit/bmw_inspection/lab/test_eight_view_demo.py`
- Modify: `tests/unit/bmw_inspection/lab/test_eight_view_demo_ui.py`
- Modify: `tests/unit/bmw_inspection/lab/test_eight_view_demo_persistence.py`
- Modify: `tests/unit/pipeline/test_bmw_lab_eight_view_demo.py`

**Interfaces:**
- Optional config object: `trusted_ok_reference: {"index": "...", "index_sha256": "<64 hex>"}`.
- Extends `EightViewUiState.trusted_ok_mode: bool = False`.
- `O` toggles the field only when an inspection exists; `N/P` keeps selecting actionable rows.

- [ ] **Step 1: Write backward-compatibility, shortcut, layout, and persistence tests**

Assert old configs load with reference comparison disabled. Assert the SHA-bound v3 config loads the published index. Assert `O` leaves model results unchanged and changes the lower panels to `现场NG`, `可信OK`, `对齐差异`, and the selected evidence type. Assert absent reference shows `无可信OK参考`. Assert `inspection.json` and `references/<view>.png` preserve all reference identities and hashes.

- [ ] **Step 2: Run focused tests and verify RED**

Expected failures: missing config fields, UI toggle, panels, and persisted reference payload.

- [ ] **Step 3: Implement optional config and matcher construction**

Keep the strict existing config schema but permit exactly one optional `trusted_ok_reference` object. Validate its SHA before model construction. If absent, pass `None` to `build_model_suite`; if present, build one resident matcher.

- [ ] **Step 4: Implement the comparison UI**

In trusted mode display four equal panels: current NG ROI, aligned trusted OK ROI, diagnostic aligned difference, and existing model evidence. Show the trusted part/sample ID, similarity, and alignment shift in the right column. Keep the current HDR source comparison when trusted mode is off.

- [ ] **Step 5: Persist exact chosen reference evidence**

Copy the matched reference ROI and difference overlay into the capture record. Add `reference_is_diagnostic_only=true`, index/whitelist/reference/source hashes, IDs, similarity, and shift to `inspection.json`; keep the existing no-overwrite capture publication.

- [ ] **Step 6: Verify and commit Task 4**

Run all affected UI/config/persistence/entrypoint tests, compile production modules, run `git diff --check`, and commit.

---

### Task 5: End-to-end laboratory verification and handoff

**Files:**
- Modify: `AGENTS_MEMORY.md`
- Modify: `pipeline/AGENTS_MEMORY.md`
- Create: `artifacts/bmw_trusted_ok_reference_smoke/` only as ignored local verification output.

**Interfaces:**
- Consumes: manually approved reference release and updated v3 Demo config.
- Produces: an offline screenshot/inspection record and a live launch command.

- [ ] **Step 1: Run the full focused regression gate**

Run all BMW v3, trusted-reference, UI, capture, persistence, Template, bright-streak, YOLO-adapter, and EfficientAD-adapter unit tests. Expected result is zero failures.

- [ ] **Step 2: Run one saved现场NG record offline**

Load an eight-view capture directory, run all 25 checks, toggle trusted mode in renderer tests, and save a 1600x900 screenshot. Verify every actionable view either has a SHA-verified approved reference or explicitly says `无可信OK参考`.

- [ ] **Step 3: Compare detector results with the feature off and on**

Serialize all 25 `(branch, view, status, score, threshold, reason)` tuples from both runs and assert exact equality. Reference similarity must appear only in diagnostic fields.

- [ ] **Step 4: Update folder memories and commit**

Record the final reference release ID, whitelist/index SHA, candidate/approved counts, shortcut, evidence meanings, verification command, and live/hardware boundary in both memory files. Commit the scoped documentation changes.

- [ ] **Step 5: Launch the v3 Demo**

```bash
cd /home/yunjing/anomaly_xingtao_new
uv run --no-sync python \
  .worktrees/bmw-eight-view-handoff/pipeline/bmw_lab_eight_view_demo.py \
  --config .worktrees/bmw-eight-view-handoff/configs/bmw/experiments/bmw_eight_view_demo_v3_ng_evidence.json \
  --experiment-mode
```

Verify the GUI remains running and four cameras open successfully before claiming live readiness.

