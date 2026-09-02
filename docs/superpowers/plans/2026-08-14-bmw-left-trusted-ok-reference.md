# BMW Left Trusted OK Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish and bind a 52-part left-hand trusted-OK reference bank for diagnostic comparisons.

**Architecture:** Add a schema-v2 manifest publisher that recomputes missing source hashes, freezes user approval metadata, and emits full/ROI reference images. Extend `TrustedOkMatcher` to load schema v2 using index-declared counts while retaining strict schema-v1 behavior.

**Tech Stack:** Python 3.13, csv/json/hashlib, Pillow, OpenCV, pytest, uv.

## Global Constraints

- Input release: `dataset/bmw_lab_prepared/bmw_left_normal_20260814_v2`.
- Include only complete eight-view `normal/OK` parts.
- Recompute every missing source SHA-256 from the current source file.
- Do not alter model outputs, thresholds, ROI coordinates or fusion.
- Do not weaken the legacy right-hand schema-v1 contract.

---

### Task 1: Generic approved-normal publisher and schema-v2 matcher

**Files:**
- Modify: `src/bmw_inspection/lab/trusted_ok_reference.py`
- Create: `pipeline/bmw_lab_publish_manifest_trusted_ok.py`
- Test: `tests/unit/bmw_inspection/lab/test_trusted_ok_reference.py`

**Interfaces:**
- Produces: `publish_manifest_trusted_reference(manifest, output_dir, roi_config, reviewer, review_note)`.
- Consumes: frozen dataset manifest and fixed-setup ROI config.

- [ ] Write a failing test with two complete parts and empty source hashes.
- [ ] Verify RED because the publisher and schema-v2 matcher do not exist.
- [ ] Implement no-overwrite publication, v2 count validation and legacy-v1 preservation.
- [ ] Verify the focused trusted-reference tests pass.

### Task 2: Publish and bind the real left reference bank

**Files:**
- Modify: `configs/bmw/experiments/bmw_eight_view_demo_left_normal_20260814_v1.json`
- Modify: `AGENTS_MEMORY.md`

**Interfaces:**
- Produces: `dataset/bmw_trusted_ok_reference/bmw_left_normal_20260814_approved_v1/reference_index.json` and its SHA-bound Demo configuration.

- [ ] Run the publisher against all52 left normal parts.
- [ ] Add `trusted_ok_reference.index` and `index_sha256` to the left Demo config.
- [ ] Load the config and preload the matcher.
- [ ] Run focused tests, compilation and diff checks, then record the exact release identity in project memory.
