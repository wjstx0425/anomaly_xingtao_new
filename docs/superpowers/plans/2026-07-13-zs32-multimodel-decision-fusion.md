# ZS32 Multimodel Decision Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing industrial fusion pipeline into a fail-closed ZS32 six-view system that preserves continuous scores, applies per-branch dual thresholds, supports front/back staged decisions, and writes auditable evidence for human review.

**Architecture:** Keep CSV normalization and the final rule engine in `capture_data/fusion_engine.py`, but move nested evidence serialization into a focused `capture_data/inspection_audit.py` module and threshold fitting/reporting into `capture_data/fusion_calibration.py`. Extend stage 18 for production fusion and add stage 30 for offline threshold calibration; do not build the review GUI in this plan.

**Tech Stack:** Python 3.10+, standard-library `csv`, `json`, `hashlib`, `dataclasses`, `pathlib`, existing optional PyYAML config loading, pytest, uv.

## Global Constraints

- Preventing false negatives is the first priority; no uncertain or incomplete inspection may become `OK`.
- Required views are exactly `front`, `front_left`, `front_right`, `back`, `back_left`, and `back_right` for each physical `part_id`.
- A strong positive may not be cancelled by clear results from other views or branches.
- Preserve raw scores, both thresholds, threshold margins, source/evidence paths, and model/config versions.
- Keep `machine_status`, `review_status`, and `released_status` separate; human review never overwrites machine evidence.
- Split and evaluate by physical `part_id`, never by independent images.
- Preserve backward compatibility for current C789 stage-18 CSVs and single-threshold configurations.
- Use `uv` for environment-managed commands. If pytest dependencies are unavailable, run the exact direct module harness plus `py_compile` and report the limitation.
- Do not modify or commit unrelated stage-29 ROI work already present in the worktree.

---

## File Structure

- Modify `capture_data/fusion_engine.py`: normalized branch contract, dual-threshold classification, per-view requirements, staged decisions, and all-trigger summaries.
- Create `capture_data/inspection_audit.py`: immutable JSON/CSV evidence assembly, image hashing, and atomic audit writes.
- Create `capture_data/fusion_calibration.py`: grouped score loading, `T_low/T_high` fitting, part-level metrics, and confidence-bound helpers.
- Modify `pipeline/18_fuse_inspection_results.py`: strict ZS32 profile inputs and audit output integration.
- Create `pipeline/30_calibrate_zs32_fusion.py`: offline calibration CLI; it never changes deployed thresholds in place.
- Create `config/fusion/zs32_six_view.json`: explicit production profile and required branches/views.
- Modify `tests/unit/capture_data/test_fusion_engine.py`: compatibility, dual-threshold, all-trigger, and staged-state tests.
- Create `tests/unit/capture_data/test_inspection_audit.py`: evidence schema, hashes, and atomic-write tests.
- Create `tests/unit/capture_data/test_fusion_calibration.py`: grouped calibration and part-level metric tests.
- Modify `tests/unit/pipeline/test_pipeline_wrappers.py`: stage-18 and stage-30 parser contracts.
- Modify `pipeline/README.md`: commands, CSV contract, output semantics, and calibration workflow.
- Modify `AGENTS_MEMORY.md`: implementation entrypoints, verified behavior, and exact validation results.

### Task 1: Add a Backward-Compatible Dual-Threshold Branch Contract

**Files:**
- Modify: `capture_data/fusion_engine.py:33-82, 348-466, 588-639`
- Modify: `tests/unit/capture_data/test_fusion_engine.py`

**Interfaces:**
- Consumes: existing branch CSV aliases and `BranchPrediction` constructor calls.
- Produces: `EvidenceLevel(str, Enum)`, `classify_evidence(prediction: BranchPrediction) -> EvidenceLevel`, and new optional `BranchPrediction` fields `low_threshold`, `high_threshold`, `evidence_level`, `model_version`, `threshold_version`, `roi_version`, `template_version`.

- [ ] **Step 1: Write failing CSV and classification tests**

Append tests that load `raw_score,low_threshold,high_threshold` rows and assert the three exact bands:

```python
def test_dual_thresholds_classify_clear_gray_and_strong(tmp_path: Path) -> None:
    fusion = load_fusion_module()
    csv_path = tmp_path / "predictions.csv"
    csv_path.write_text(
        "part_id,side,view,raw_score,low_threshold,high_threshold,model_version,threshold_version\n"
        "p1,zs32,front,0.20,0.30,0.50,m1,t1\n"
        "p2,zs32,front,0.40,0.30,0.50,m1,t1\n"
        "p3,zs32,front,0.60,0.30,0.50,m1,t1\n",
        encoding="utf-8",
    )
    rows = fusion.load_branch_predictions_csv(csv_path, branch="anomaly_front")
    assert [fusion.classify_evidence(row).value for row in rows] == ["CLEAR", "GRAY", "STRONG"]
    assert rows[0].model_version == "m1"
    assert rows[0].threshold_version == "t1"
```

Add compatibility tests asserting a legacy `pred_label=1` row remains STRONG and a legacy `pred_label=0` row remains CLEAR when dual thresholds are absent. Add validation tests for `T_low > T_high`, non-finite scores, and a row containing only one dual threshold; each must raise `ValueError` with the `part_id` and branch in the message.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
uv run pytest tests/unit/capture_data/test_fusion_engine.py -k "dual_thresholds or legacy_evidence or invalid_dual" -v
```

Expected: FAIL because the new fields and `classify_evidence` do not exist.

- [ ] **Step 3: Implement the enum, optional fields, parsing, and validation**

Add:

```python
from enum import Enum
from math import isfinite

class EvidenceLevel(str, Enum):
    CLEAR = "CLEAR"
    GRAY = "GRAY"
    STRONG = "STRONG"

def classify_evidence(prediction: BranchPrediction) -> EvidenceLevel:
    if prediction.evidence_level is not None:
        return EvidenceLevel(prediction.evidence_level.upper())
    low, high, score = prediction.low_threshold, prediction.high_threshold, prediction.score
    if low is None and high is None:
        return EvidenceLevel.STRONG if prediction.pred_label == 1 else EvidenceLevel.CLEAR
    if score is None or low is None or high is None:
        raise ValueError(f"incomplete dual thresholds for {prediction.part_id}:{prediction.branch}")
    if not all(isfinite(value) for value in (score, low, high)) or low > high:
        raise ValueError(f"invalid dual thresholds for {prediction.part_id}:{prediction.branch}")
    if score >= high:
        return EvidenceLevel.STRONG
    if score >= low:
        return EvidenceLevel.GRAY
    return EvidenceLevel.CLEAR
```

Append optional dataclass fields after existing defaulted fields so current keyword and positional construction remains valid. Parse `raw_score` before legacy score aliases, `low_threshold`, `high_threshold`, and the four version fields. Extend `BRANCH_FIELDNAMES` so normalized CSV output retains every field.

- [ ] **Step 4: Run focused and full fusion tests**

Run:

```bash
uv run pytest tests/unit/capture_data/test_fusion_engine.py -v
```

Expected: all fusion-engine tests PASS, including unchanged legacy tests.

- [ ] **Step 5: Commit Task 1**

```bash
git add capture_data/fusion_engine.py tests/unit/capture_data/test_fusion_engine.py
git commit -m "feat: add dual-threshold fusion evidence"
```

### Task 2: Enforce Strict Per-View Gates and Preserve Every Trigger

**Files:**
- Modify: `capture_data/fusion_engine.py:83-104, 289-466, 547-585, 673-705`
- Modify: `tests/unit/capture_data/test_fusion_engine.py`
- Create: `config/fusion/zs32_six_view.json`

**Interfaces:**
- Consumes: `classify_evidence()` from Task 1 and config keys `ok_requires.required_view_keys` plus `required_branches_by_view`.
- Produces: `TriggerEvidence`, `FusedDecision.triggered_evidence`, `missing_required_branch_keys(...) -> list[str]`, and strict `REVIEW/NG/OK` behavior.

- [ ] **Step 1: Write failing strict-fusion tests**

Add tests covering these independent cases:

```python
def test_gray_evidence_returns_review_and_strong_wins() -> None:
    # p1 has one GRAY anomaly and otherwise CLEAR -> REVIEW.
    # p2 has one GRAY anomaly plus one STRONG geometry row -> NG_GEOMETRY.
    # Both decisions retain every GRAY/STRONG TriggerEvidence item.

def test_ok_requires_every_configured_branch_per_view() -> None:
    # Six views exist, but back_right lacks anomaly_back_right -> REVIEW.
    # reason contains "missing required branch: back_right:anomaly_back_right".

def test_all_six_views_and_required_branches_clear_returns_ok() -> None:
    # Exactly six view rows and all configured branches are CLEAR -> OK.
```

Use `side="zs32"` consistently. Assert `triggered_evidence` contains stable IDs in `view:branch` form and is ordered by configured branch order, then view name.

- [ ] **Step 2: Run tests and confirm RED**

```bash
uv run pytest tests/unit/capture_data/test_fusion_engine.py -k "gray_evidence or every_configured_branch or all_six_views" -v
```

Expected: FAIL because strict per-view branches and all-trigger evidence are not implemented.

- [ ] **Step 3: Implement strict branch/view requirements and all-trigger decisions**

Add immutable evidence details:

```python
@dataclass(frozen=True)
class TriggerEvidence:
    evidence_id: str
    branch: str
    side: str
    view: str | None
    level: str
    score: float | None
    low_threshold: float | None
    high_threshold: float | None
    reason: str
```

Change `FusedDecision` to retain the current primary fields plus `triggered_evidence: tuple[TriggerEvidence, ...] = ()`. Implement `missing_required_branch_keys` using config shaped as:

```json
{
  "required_branches_by_view": {
    "front": ["quality_gate", "registration", "anomaly_front", "yolo", "geometry"],
    "front_left": ["quality_gate", "registration", "anomaly_front_left", "yolo", "geometry"]
  }
}
```

Evaluate all non-gate predictions once. Any STRONG row contributes a trigger, any GRAY row contributes a trigger, and the configured first STRONG determines the primary `NG_*` status. If no STRONG exists but any GRAY or required branch is missing, return `REVIEW`. Only return OK when missing-view and missing-branch lists are empty and all required predictions classify CLEAR.

Keep reading old `suspect_policy` configs, mapping the old near-threshold path to `REVIEW` while accepting existing `SUSPECT` input status as a GRAY trigger. Keep writing legacy `triggered_branch` for downstream compatibility.

- [ ] **Step 4: Add the production ZS32 profile**

Create `config/fusion/zs32_six_view.json` with explicit keys:

```json
{
  "profile": "zs32_six_view_v1",
  "ok_requires": {
    "quality_gate": "PASS",
    "registration": "PASS",
    "required_view_keys": [
      "zs32:front", "zs32:front_left", "zs32:front_right",
      "zs32:back", "zs32:back_left", "zs32:back_right"
    ]
  },
  "required_branches_by_view": {
    "front": ["quality_gate", "registration", "anomaly_front", "yolo", "geometry"],
    "front_left": ["quality_gate", "registration", "anomaly_front_left", "yolo", "geometry"],
    "front_right": ["quality_gate", "registration", "anomaly_front_right", "yolo", "geometry"],
    "back": ["quality_gate", "registration", "anomaly_back", "yolo", "geometry"],
    "back_left": ["quality_gate", "registration", "anomaly_back_left", "yolo", "geometry"],
    "back_right": ["quality_gate", "registration", "anomaly_back_right", "yolo", "geometry"]
  },
  "branch_order": ["geometry", "yolo", "anomaly_front", "anomaly_front_left", "anomaly_front_right", "anomaly_back", "anomaly_back_left", "anomaly_back_right"],
  "rules": {
    "geometry": {"status_on_positive": "NG_GEOMETRY"},
    "yolo": {"status_on_positive": "NG_YOLO"},
    "anomaly_front": {"status_on_positive": "NG_ANOMALY"},
    "anomaly_front_left": {"status_on_positive": "NG_ANOMALY"},
    "anomaly_front_right": {"status_on_positive": "NG_ANOMALY"},
    "anomaly_back": {"status_on_positive": "NG_ANOMALY"},
    "anomaly_back_left": {"status_on_positive": "NG_ANOMALY"},
    "anomaly_back_right": {"status_on_positive": "NG_ANOMALY"}
  }
}
```

Update `required_view_keys()` to honor explicit `ok_requires.required_view_keys` without removing the existing sides-by-views behavior.

- [ ] **Step 5: Run tests and validate the JSON profile**

```bash
uv run pytest tests/unit/capture_data/test_fusion_engine.py -v
uv run python -m json.tool config/fusion/zs32_six_view.json
```

Expected: tests PASS and JSON prints without error.

- [ ] **Step 6: Commit Task 2**

```bash
git add capture_data/fusion_engine.py tests/unit/capture_data/test_fusion_engine.py config/fusion/zs32_six_view.json
git commit -m "feat: enforce strict ZS32 fusion gates"
```

### Task 3: Add Front/Back Stage Decisions and Atomic Audit Evidence

**Files:**
- Create: `capture_data/inspection_audit.py`
- Create: `tests/unit/capture_data/test_inspection_audit.py`
- Modify: `capture_data/fusion_engine.py`
- Modify: `tests/unit/capture_data/test_fusion_engine.py`

**Interfaces:**
- Consumes: `BranchPrediction`, `FusedDecision`, and `TriggerEvidence` from Tasks 1-2.
- Produces: `FaceDecision`, `fuse_face_predictions(...) -> FaceDecision`, `build_part_audit(...) -> dict[str, Any]`, and `write_part_audit(...) -> Path`.

- [ ] **Step 1: Write failing staged-state tests**

Add exact cases:

```python
def test_front_face_never_returns_final_ok() -> None:
    result = fusion.fuse_face_predictions("p1", clear_front_rows(), face="front", config=zs32_config())
    assert result.stage_status == "FRONT_CLEAR"
    assert result.final_status is None

def test_front_strong_is_front_ng_but_marks_inspection_incomplete() -> None:
    result = fusion.fuse_face_predictions("p1", strong_front_rows(), face="front", config=zs32_config())
    assert result.stage_status == "FRONT_NG"
    assert result.inspection_complete is False
```

Also test `FRONT_REVIEW`, back-stage equivalents, identity mismatch, and final combination `CLEAR+CLEAR -> OK`, `REVIEW+CLEAR -> REVIEW`, `NG+anything -> NG`.

- [ ] **Step 2: Write failing audit tests**

Create a 1-byte source image and evidence file under `tmp_path`. Assert the output JSON includes `schema_version="1.0"`, image SHA-256, all six view keys, raw scores, both thresholds, margins, version fields, all triggers, and separate review fields:

```python
assert audit["machine_status"] == "REVIEW"
assert audit["review"]["status"] == "PENDING"
assert audit["review_status"] is None
assert audit["released_status"] is None
```

Monkeypatch `Path.replace` to fail and assert no final partial JSON is published.

- [ ] **Step 3: Run the staged and audit tests to confirm RED**

```bash
uv run pytest tests/unit/capture_data/test_fusion_engine.py -k "front_face or front_strong or final_combination" -v
uv run pytest tests/unit/capture_data/test_inspection_audit.py -v
```

Expected: FAIL because neither interface exists.

- [ ] **Step 4: Implement the face state machine**

Add:

```python
@dataclass(frozen=True)
class FaceDecision:
    part_id: str
    face: str
    stage_status: str
    final_status: str | None
    inspection_complete: bool
    triggered_evidence: tuple[TriggerEvidence, ...]
    reason: str
```

`fuse_face_predictions` accepts only `face in {"front", "back"}`, requires the corresponding three views, and maps the strict fusion result to `{FACE}_CLEAR`, `{FACE}_REVIEW`, or `{FACE}_NG`. Add `combine_face_decisions(front, back)` that rejects differing `part_id`, requires both faces, and never returns OK unless both are CLEAR.

- [ ] **Step 5: Implement focused atomic audit serialization**

In `inspection_audit.py`, implement:

```python
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def write_part_audit(audit: Mapping[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return output_path
```

`build_part_audit` groups predictions by the six canonical views, computes margins `score-low_threshold` and `score-high_threshold` when values exist, hashes every present source image, retains evidence paths without silently ignoring missing evidence, and initializes review fields without overwriting machine fields.

- [ ] **Step 6: Run all focused tests**

```bash
uv run pytest tests/unit/capture_data/test_fusion_engine.py tests/unit/capture_data/test_inspection_audit.py -v
uv run python -m py_compile capture_data/fusion_engine.py capture_data/inspection_audit.py
```

Expected: PASS and compilation succeeds.

- [ ] **Step 7: Commit Task 3**

```bash
git add capture_data/fusion_engine.py capture_data/inspection_audit.py tests/unit/capture_data/test_fusion_engine.py tests/unit/capture_data/test_inspection_audit.py
git commit -m "feat: record staged inspection evidence"
```

### Task 4: Integrate Strict ZS32 Fusion and Audit Outputs into Stage 18

**Files:**
- Modify: `pipeline/18_fuse_inspection_results.py:17-123`
- Modify: `tests/unit/pipeline/test_pipeline_wrappers.py`
- Create: `tests/unit/pipeline/test_fuse_inspection_results.py`

**Interfaces:**
- Consumes: strict fusion config, normalized predictions, `build_part_audit`, and `write_part_audit`.
- Produces: CLI options `--profile`, `--audit-dir`, and `--require-complete-evidence`; outputs `branch_predictions.csv`, `fused_predictions.csv`, `summary.md`, and `<audit-dir>/<part_id>.json`.

- [ ] **Step 1: Write failing parser and end-to-end tests**

Add parser assertions:

```python
args = module.build_parser().parse_args([
    "--profile", "zs32",
    "--audit-dir", str(tmp_path / "audit"),
    "--require-complete-evidence",
    "--output-dir", str(tmp_path / "out"),
])
assert args.profile == "zs32"
assert args.require_complete_evidence is True
```

The end-to-end test writes six view rows with all configured branch rows, verifies a GRAY row produces REVIEW, and opens the generated JSON to assert the raw score and trigger ID. Add failure tests for missing source image/evidence path under `--require-complete-evidence`; expected result is REVIEW with no OK release.

- [ ] **Step 2: Run tests and confirm RED**

```bash
uv run pytest tests/unit/pipeline/test_pipeline_wrappers.py -k "fuse_inspection" -v
uv run pytest tests/unit/pipeline/test_fuse_inspection_results.py -v
```

Expected: FAIL because the new CLI and audit integration do not exist.

- [ ] **Step 3: Implement CLI integration**

Resolve `--profile zs32` to `config/fusion/zs32_six_view.json`; reject simultaneous incompatible profile/config values rather than merging silently. If `--audit-dir` is omitted, default to `<output-dir>/audit`. Under `--require-complete-evidence`, missing source/evidence artifacts add a system trigger and force REVIEW.

Write one audit JSON per part only after fusion completes. Add summary counts for `review`, `retake`, `invalid_capture`, `complete_inspections`, and `incomplete_inspections`. Keep accepting all existing CSV options and current non-ZS32 usage.

- [ ] **Step 4: Run stage-18 tests and help smoke**

```bash
uv run pytest tests/unit/pipeline/test_pipeline_wrappers.py tests/unit/pipeline/test_fuse_inspection_results.py -v
uv run python pipeline/18_fuse_inspection_results.py --help
```

Expected: PASS; help lists the three new options.

- [ ] **Step 5: Commit Task 4**

```bash
git add pipeline/18_fuse_inspection_results.py tests/unit/pipeline/test_pipeline_wrappers.py tests/unit/pipeline/test_fuse_inspection_results.py
git commit -m "feat: emit ZS32 fusion audit records"
```

### Task 5: Add Offline Dual-Threshold Calibration and Part-Level Metrics

**Files:**
- Create: `capture_data/fusion_calibration.py`
- Create: `pipeline/30_calibrate_zs32_fusion.py`
- Create: `tests/unit/capture_data/test_fusion_calibration.py`
- Modify: `tests/unit/pipeline/test_pipeline_wrappers.py`

**Interfaces:**
- Consumes: calibration CSV rows with `part_id,hand,view,branch,raw_score,gt_label,split`.
- Produces: `fit_dual_thresholds(rows, *, target_recall: float, normal_quantile: float) -> list[ThresholdRecord]`, `part_level_metrics(rows, thresholds) -> dict[str, Any]`, and immutable JSON/CSV reports.

- [ ] **Step 1: Write failing grouped-calibration tests**

Create fixtures where six views share one `part_id`. Assert:

- all views of a part must have the same `split`, otherwise `ValueError`;
- thresholds group by `(hand, view, branch, model_version, roi_version)`;
- `T_low <= T_high` for every group;
- a defect score below `T_low` counts as a part-level escape;
- six image rows for one escaped part count as one escape, not six;
- missing required groups produce an explicit `insufficient_data` record, not a fabricated threshold;
- zero observed escapes still reports sample count and an exact one-sided 95% upper bound `1 - 0.05 ** (1 / n)`.

- [ ] **Step 2: Run calibration tests and confirm RED**

```bash
uv run pytest tests/unit/capture_data/test_fusion_calibration.py -v
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement deterministic calibration helpers**

Use no new numerical dependency. Implement an inclusive nearest-rank quantile and immutable records:

```python
@dataclass(frozen=True)
class ThresholdRecord:
    hand: str
    view: str
    branch: str
    model_version: str
    roi_version: str
    low_threshold: float | None
    high_threshold: float | None
    normal_count: int
    defect_count: int
    status: str
```

Choose `T_low` from defect calibration scores to meet the configured recall constraint whenever defect data exists. Choose `T_high` from the maximum of `T_low` and the configured normal quantile, so the review band is never inverted. If data cannot support the target, set `status="insufficient_data"` and leave deployment thresholds null. Do not mutate an existing deployment config.

Compute part-level outcomes with the same OR/GRAY semantics as the fusion engine. Report overall and grouped escape count/rate, recall, normal reject rate, review rate, counts, and the zero-escape upper bound.

- [ ] **Step 4: Add the stage-30 CLI and parser test**

Implement parser arguments:

```text
--input-csv PATH              required
--output-dir PATH             required
--target-recall FLOAT         default 1.0, range (0, 1]
--normal-quantile FLOAT       default 0.995, range (0, 1]
--required-view VIEW          repeatable
```

Write `thresholds.json`, `thresholds.csv`, `calibration_metrics.json`, and `calibration_summary.md` atomically. Include dataset counts and warn that observed 100% recall is not proof of zero production escapes.

- [ ] **Step 5: Run unit, parser, and CLI smoke tests**

```bash
uv run pytest tests/unit/capture_data/test_fusion_calibration.py tests/unit/pipeline/test_pipeline_wrappers.py -k "fusion_calibration or calibrate_zs32" -v
uv run python pipeline/30_calibrate_zs32_fusion.py --help
uv run python -m py_compile capture_data/fusion_calibration.py pipeline/30_calibrate_zs32_fusion.py
```

Expected: PASS; help shows all five options.

- [ ] **Step 6: Commit Task 5**

```bash
git add capture_data/fusion_calibration.py pipeline/30_calibrate_zs32_fusion.py tests/unit/capture_data/test_fusion_calibration.py tests/unit/pipeline/test_pipeline_wrappers.py
git commit -m "feat: calibrate ZS32 fusion thresholds"
```

### Task 6: Document, Fault-Test, and Verify the Complete Workflow

**Files:**
- Modify: `tests/unit/pipeline/test_fuse_inspection_results.py`
- Modify: `pipeline/README.md:1594-1636`
- Modify: `CHANGELOG.md`
- Modify: `AGENTS_MEMORY.md`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: runnable operational commands, fault-injection regression coverage, changelog entry, and durable project memory.

- [ ] **Step 1: Add table-driven fail-closed fault tests**

Parameterize these injected failures: missing view, duplicate view identity, mismatched `part_id`, quality FAIL, registration WARN, non-finite score, model version mismatch, ROI version mismatch, missing source image, missing evidence artifact, and an empty YOLO detection row alongside a GRAY anomaly row.

For every case assert:

```python
assert decision.final_status != "OK"
assert audit["released_status"] is None
assert audit["machine_status"] in {"REVIEW", "RETAKE", "INVALID_CAPTURE", "NG_ANOMALY", "NG_GEOMETRY", "NG_YOLO"}
```

- [ ] **Step 2: Run fault tests and fix only implementation defects revealed by them**

```bash
uv run pytest tests/unit/pipeline/test_fuse_inspection_results.py -k "fault" -v
```

Expected: all injected failures are fail-closed. If a case reaches OK, add the smallest validation in its owning module and a focused regression assertion.

- [ ] **Step 3: Document exact calibration and fusion commands**

Add a `ZS32 六视角严格融合` subsection to `pipeline/README.md` with these copy-pastable forms:

```bash
uv run python pipeline/30_calibrate_zs32_fusion.py \
  --input-csv results/zs32_fusion/calibration_rows.csv \
  --output-dir results/zs32_fusion/calibration_v1 \
  --target-recall 1.0 \
  --normal-quantile 0.995

uv run python pipeline/18_fuse_inspection_results.py \
  --profile zs32 \
  --manifest results/zs32_fusion/inspection_manifest.csv \
  --quality-csv results/zs32_fusion/quality_gate.csv \
  --registration-csv results/zs32_fusion/registration.csv \
  --branch-csv anomaly_front=results/zs32_fusion/anomaly_front.csv \
  --branch-csv anomaly_front_left=results/zs32_fusion/anomaly_front_left.csv \
  --branch-csv anomaly_front_right=results/zs32_fusion/anomaly_front_right.csv \
  --branch-csv anomaly_back=results/zs32_fusion/anomaly_back.csv \
  --branch-csv anomaly_back_left=results/zs32_fusion/anomaly_back_left.csv \
  --branch-csv anomaly_back_right=results/zs32_fusion/anomaly_back_right.csv \
  --branch-csv yolo=results/zs32_fusion/yolo.csv \
  --branch-csv geometry=results/zs32_fusion/geometry.csv \
  --require-complete-evidence \
  --output-dir results/zs32_fusion/fused_v1
```

Document every output field, the five final statuses, staged front/back semantics, the rule that YOLO no-box does not cancel other evidence, and the human review fields.

- [ ] **Step 4: Add changelog and memory entries**

Under `CHANGELOG.md` `## [Unreleased]` → `Added`, record the strict ZS32 dual-threshold fusion, staged decisions, audit JSON, and calibration CLI. Append `AGENTS_MEMORY.md` with exact entrypoints, config path, output files, commands run, test counts, and any environment limitations. Preserve all pre-existing uncommitted ROI memory content.

- [ ] **Step 5: Run the complete verification gate**

```bash
uv run pytest tests/unit/capture_data/test_fusion_engine.py tests/unit/capture_data/test_inspection_audit.py tests/unit/capture_data/test_fusion_calibration.py tests/unit/pipeline/test_fuse_inspection_results.py tests/unit/pipeline/test_pipeline_wrappers.py -v
uv run ruff check capture_data/fusion_engine.py capture_data/inspection_audit.py capture_data/fusion_calibration.py pipeline/18_fuse_inspection_results.py pipeline/30_calibrate_zs32_fusion.py tests/unit/capture_data/test_fusion_engine.py tests/unit/capture_data/test_inspection_audit.py tests/unit/capture_data/test_fusion_calibration.py tests/unit/pipeline/test_fuse_inspection_results.py tests/unit/pipeline/test_pipeline_wrappers.py
uv run ruff format --check capture_data/fusion_engine.py capture_data/inspection_audit.py capture_data/fusion_calibration.py pipeline/18_fuse_inspection_results.py pipeline/30_calibrate_zs32_fusion.py tests/unit/capture_data/test_fusion_engine.py tests/unit/capture_data/test_inspection_audit.py tests/unit/capture_data/test_fusion_calibration.py tests/unit/pipeline/test_fuse_inspection_results.py
uv run python -m py_compile capture_data/fusion_engine.py capture_data/inspection_audit.py capture_data/fusion_calibration.py pipeline/18_fuse_inspection_results.py pipeline/30_calibrate_zs32_fusion.py
uv run python -m json.tool config/fusion/zs32_six_view.json
git diff --check
```

Expected: all tests and static checks PASS. If full pytest cannot start because the environment lacks a dependency, record the exact error and run direct imports plus the target test-function harness; do not claim the unavailable suite passed.

- [ ] **Step 6: Review the final diff for scope and evidence preservation**

```bash
git status --short
git diff --stat
git diff -- capture_data/fusion_engine.py capture_data/inspection_audit.py capture_data/fusion_calibration.py pipeline/18_fuse_inspection_results.py pipeline/30_calibrate_zs32_fusion.py config/fusion/zs32_six_view.json pipeline/README.md CHANGELOG.md AGENTS_MEMORY.md
```

Expected: only fusion, audit, calibration, tests, docs, changelog, and the appended memory entry are in scope; pre-existing stage-29 files remain untouched.

- [ ] **Step 7: Commit Task 6**

```bash
git add tests/unit/pipeline/test_fuse_inspection_results.py pipeline/README.md CHANGELOG.md AGENTS_MEMORY.md
git commit -m "docs: document ZS32 strict fusion workflow"
```

## Final Acceptance Criteria

- One STRONG result from any required branch produces NG and is never cancelled by other CLEAR rows.
- One GRAY result, missing required branch, version mismatch, or incomplete evidence produces REVIEW/RETAKE/INVALID_CAPTURE, never OK.
- Final OK requires the same physical `part_id`, all six views, required quality and registration PASS rows, and all configured branch rows classified CLEAR.
- The front face alone never emits final OK.
- Every audit JSON retains raw continuous scores, both thresholds, margins, hashes, versions, paths, all triggers, and separate machine/review/release statuses.
- Calibration and evaluation operate at physical-part level and report sample counts plus the zero-escape confidence upper bound.
- Existing C789 and legacy single-threshold stage-18 tests remain green.
- Operational docs and `AGENTS_MEMORY.md` match the verified implementation rather than anticipated behavior.
