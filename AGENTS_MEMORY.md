# AGENTS Memory

## ZS32 serial-bound three-camera capture

- The runnable entrypoint is `pipeline/1_collect_multicamera_data.py`; its default mode is single exposure, while `--hdr` explicitly enables the retained HDR path.
- Camera roles are bound by USB serial, not enumeration index: front `DA9805574`, left `DA9625347`, right `DB0998274`.
- The collector no longer writes `AcquisitionFrameRate` or accepts `--fps`; `--capture-interval` is application-side pacing only.
- Every grouped pass keeps `trigger x3 -> read x3`. Single images use `_single.png`; HDR keeps fused/source outputs.
- Cleanup independently attempts stop, `TriggerMode=Off`, close, and destroy so one failure does not skip later cleanup.
- Codex performed offline parser/direct assertions, compile, help, and diff checks after commit `c9f1e38f`; per user request, post-fix hardware acceptance remains user-owned.
- Right-hand capture uses the same serial-bound cameras and six views with `--hand right`. The confirmed normal HDR batch is 100 groups, one image per view, short/long exposures `1500/6000 us`, and root `/home/yunjing/anomalib/dataset`; source exposure images are off unless `--save-hdr-sources` is passed.

## ZS32 three-camera grouped HDR capture (2026-07-11)

### Serial-bound safe camera lifecycle follow-up (Task 2, 2026-07-11)

- Reviewer follow-up: `_CaptureArgumentParser.parse_args()` now normalizes an omitted `--fps` to `10.0` and rejects
  non-finite or non-positive `--exposure`, `--short-exposure`, `--long-exposure`, and `--fps` values before
  `HikvisionAdapter.load()` can run. Keep this validation in the SDK-free parser rather than moving it back into
  `main()`.
- `HikvisionAdapter.open(device, gain)` must not configure `AcquisitionFrameRateEnable` or
  `AcquisitionFrameRate`; application-level pacing remains separate from camera-node setup.
- After `CreateHandle` succeeds, every setup failure including `KeyboardInterrupt` must independently attempt
  `TriggerMode=0`, `CloseDevice`, and `DestroyHandle`. Normal/context cleanup uses reverse camera order and attempts
  stop (when started), restore, close, and destroy even if an earlier cleanup operation fails.
- Float camera writes query their SDK range through `self.sdk.MVCC_FLOATVALUE` and `MV_CC_GetFloatValue`, keeping the
  module importable without the Hikvision SDK. Gain and every exposure write reject out-of-range and non-finite values.
- Cleanup failures are aggregated. With no primary error they raise after all handles are processed; with a capture or
  setup error they are reported as an exception note so the original exception identity is preserved.
- This task is unit/static only and must not connect to cameras; hardware validation remains a separate explicit task.

- Core module: `capture_data/collect_multicamera_dataset.py`; thin numbered wrapper:
  `pipeline/1_collect_multicamera_data.py`.
- The six canonical view names are `front`, `front_left`, `front_right`, `back`, `back_left`, and `back_right`.
- Default `--devices 0 1 2` mapping: device 0 is the central camera (`front`/`back`), device 1 is the left-side
  camera (`front_left`/`back_left`), and device 2 is the right-side camera (`front_right`/`back_right`). Always run
  `pipeline/1_collect_multicamera_data.py --list-devices` before capture because SDK enumeration indices may change.
- Each group has two placement prompts: capture every front image index first, flip the same static part, then
  capture every back image index. Matching front/back image indices retain one paired `sample_id`.
- `capture_exposure_pass()` sets one exposure on all handles, discards configured settle passes using grouped
  trigger/read ordering, then returns one grouped final pass. `capture_hdr_round()` captures the full three-camera
  short/long pair, fuses by physical camera slot, and retries the complete pair when any fused view exceeds
  `hdr_max_clip_pct`. Any camera read exception propagates without returning a partial `HdrViewResult` list.
- `TriggerPassPacer` must live for the full open-camera session: `main()` creates it once and passes it through every
  `capture_group()`, while compatibility `capture_sample()` creates one shared across its front/back rounds. Do not
  recreate it at HDR-round or group boundaries, because that clears `_last_pass_at` and can allow adjacent software
  triggers inside the camera frame interval, causing `MV_E_NODATA`.
- Software triggering sends all three triggers before reading frames. It is suitable for static parts but is not
  hardware synchronization; moving parts or strict simultaneous exposure require shared hardware trigger wiring.
- A sample is `complete` only when all six distinct canonical views were stored. Failures remain explicit as an
  `incomplete` sample row with round/view/device/error diagnostics in `<root>/manifests/<session_id>.csv`.
- Static verification commands for this implementation:
  - `.venv/bin/python -m pytest tests/unit/capture_data/test_collect_multicamera_dataset.py tests/unit/pipeline/test_pipeline_wrappers.py -v`
  - `.venv/bin/python -m compileall capture_data/collect_multicamera_dataset.py pipeline/1_collect_multicamera_data.py`
- Hardware smoke-test verification (Task 7): **pending; not run as part of Task 6**.
  - Smoke-test output path: **pending Task 7**.
  - Connected camera serials: **pending Task 7 device discovery**.
  - Measured capture results: **pending Task 7; do not infer success from unit/static tests**.

## GitHub upload guardrails

- On 2026-07-02, before uploading local code to GitHub, the checkout had large local artifacts under `results/` (~70G), `dataset/` (~79G), and `c789_bottom/` (~3G).
- `.gitignore` already ignored `results`, `dataset/`, `datasets`, and training logs such as `wandb/`, `lightning_logs/`, and `mlruns`.
- Added `c789_bottom/` and `*.ckpt` to `.gitignore` so local model checkpoints are not accidentally staged by `git add .`.
- Push local code changes to the `mygithub` remote (`git@github.com:wjstx0425/anomaly_xingtao.git`) rather than the upstream `origin` remote unless the user explicitly wants to contribute to upstream anomalib.
- On 2026-07-02, a follow-up review found documentation ambiguity around stages 7-15, upload boundaries, EfficientAD asset defaults, C789 geometry template workflow, demo archive/OCR outputs, and threshold sources. `README.md`, `pipeline/README.md`, and `CHANGELOG.md` were updated to make these boundaries explicit.
- The same review fixed a small data-split bug: negative `--normal-test-ratio` values now raise `ValueError` instead of silently disabling the split.

## C789 left_top geometry + AnomalyDINO fusion

- User override for this workflow: use `.venv/bin/python`, not `uv`, even though `AGENTS.md` mentions uv.
- Do not overwrite `results/c789_100_hardened/left_top_geometry/manual_review_pack/manual_masks` unless the user explicitly asks.
- Pytest is not available in the current `.venv`; use `compileall` and direct Python/CLI harnesses for validation.
- Current manual-mask workflow paths:
  - Mask input: `results/c789_100_hardened/left_top_geometry/manual_review_pack/manual_masks`
  - Manual templates: `results/c789_100_hardened/left_top_geometry/manual_templates`
  - Stress locked eval: `results/c789_100_hardened/left_top_geometry/manual_stress_locked`
  - Defect fused eval: `results/c789_100_hardened/left_top_geometry/manual_defect_fused`
- Manual masks are complete for `slot01` through `slot06`: each has `expected`, `allowed`, `ignore`, and `watch_edge` PNGs. They appeared generated around the same time as the review sheets on 2026-06-20, so treat them as seed masks unless the user confirms hand edits.
- Fresh commands run on 2026-06-20:
  - `.venv/bin/python pipeline/15_edit_geometry_masks.py --review-pack results/c789_100_hardened/left_top_geometry/manual_review_pack --template-dir results/c789_100_hardened/left_top_geometry/manual_templates --stress-root dataset/c789_stress_normal_group_split/locked/left/top --defect-root dataset/c789_100_left_top_parts/left/top/defect --anomaly-predictions results/c789_100_hardened/left_top_anomaly_dino/reports/predictions.csv --stress-output-dir results/c789_100_hardened/left_top_geometry/manual_stress_locked --defect-output-dir results/c789_100_hardened/left_top_geometry/manual_defect_fused`
  - `.venv/bin/python pipeline/14_build_manual_geometry_templates.py --mask-dir results/c789_100_hardened/left_top_geometry/manual_review_pack/manual_masks --output-dir results/c789_100_hardened/left_top_geometry/manual_templates`
  - `.venv/bin/python pipeline/12_geometry_eval.py --data-root dataset/c789_stress_normal_group_split/locked/left/top --template-dir results/c789_100_hardened/left_top_geometry/manual_templates --output-dir results/c789_100_hardened/left_top_geometry/manual_stress_locked --calibrate-thresholds`
  - `.venv/bin/python pipeline/12_geometry_eval.py --data-root dataset/c789_100_left_top_parts/left/top/defect --template-dir results/c789_100_hardened/left_top_geometry/manual_templates --thresholds results/c789_100_hardened/left_top_geometry/manual_stress_locked/geometry_thresholds.csv --anomaly-predictions results/c789_100_hardened/left_top_anomaly_dino/reports/predictions.csv --output-dir results/c789_100_hardened/left_top_geometry/manual_defect_fused`
- Visual editor validation note: in `pipeline/15_edit_geometry_masks.py`, `v` should rebuild `manual_templates`, recalibrate locked stress normal thresholds, and rerun defect fusion against `manual_defect_fused`; keep masks pure black/white, and rely on editor backups before save overwrites.
- Fresh metrics from those outputs:
  - Stress locked geometry FP: `0/90`.
  - AnomalyDINO defect recall: `8/17`.
  - Geometry defect positives: `13/17`.
  - Fused defect recall: `15/17`.
  - Fused misses: `less_1_2_slot02`, `corner_3_1_slot05`.
  - Key samples:
    - `less_1_2_slot02`: anomaly `0`, geometry `0`, final `0`, type `more`, region `r00_c05`, score `2722.0`, threshold `5817.0`, missing `186`, extra `2722`.
    - `less_2_1_slot03`: anomaly `0`, geometry `1`, final `1`, type `more`, region `r03_c06`, score `986.0`, threshold `954.45`, missing `150`, extra `24449`.
    - `more_2_2_slot04`: anomaly `0`, geometry `1`, final `1`, type `more`, region `r02_c00`, score `3993.0`, threshold `127.05`, missing `0`, extra `11379`.
- Implementation caveat: default geometry threshold calibration is per slot/type/coarse region. A defect in a region/type not seen in stress-normal calibration can be ignored unless a slot or wildcard fallback threshold exists.
- MVP-2 geometry fallback implemented on 2026-07-03:
  - Threshold lookup order in `capture_data/geometry_shape.py` is `exact -> slot_type -> slot_region -> slot -> defect_type -> global`.
  - `defect_type` in geometry threshold CSVs maps to the current geometry branch `geometry_type` (`less`/`more`); geometry eval still does not read external manifests for true defect categories.
  - `load_thresholds()` accepts both old columns (`slot`, `geometry_type`, `geometry_region`, `geometry_threshold`) and MVP-2 columns (`slot_id`, `defect_type`, `region_id`, `threshold`).
  - Old exact threshold CSVs are completed with conservative wildcard fallback rows using max existing threshold values, favoring stress-normal FP safety.
  - If locked-normal region calibration has no nonzero region scores, it still emits per-slot `slot/*/* = 0.0` rows plus global fallback instead of an empty threshold CSV.
  - Samples with no region scores are explicitly marked `geometry_pred_label=0` via slot/global fallback diagnostics instead of leaving threshold fields blank.
  - `geometry_predictions.csv` now includes `threshold_source` and `threshold_lookup_level`.
  - Recalibrated `geometry_thresholds.csv` keeps old columns and adds `slot_id`, `defect_type`, `region_id`, `threshold`, `threshold_source`, `n_normal`, `max_normal`, `p99_normal`, and `p999_normal`.
- Review-pack visual observations:
  - `slot02` currently has a broad full-outline `watch_edge`; rescuing `less_1_2_slot02` likely needs a more precise watch edge around the actual missing boundary and less tolerance around unrelated bright/fixture areas.
  - `slot03` and `slot04` seed masks include visible non-part fixture/background structures in `allowed`/`watch_edge`; remove these from part geometry or put them into `ignore` before treating the masks as final.
- Fresh verification: `.venv/bin/python -m compileall` on the geometry scripts and pipeline wrappers exited with code 0.

## C789 left_top high-exposure geometry seed

- On 2026-06-21, generated a separate high-exposure seed flow from:
  - `dataset/c789_100/left/top/normal/part001_20260610_105404/raw_exposures/no_hand_top_normal_part001_g001_000000_exp35000.png`
- This flow intentionally does not overwrite the active manual masks under:
  - `results/c789_100_hardened/left_top_geometry/manual_review_pack/manual_masks`
- High-exposure seed artifacts:
  - Slot crops: `results/c789_100_hardened/left_top_geometry/high_exp_seed_crops`
  - Review pack and editable seed masks: `results/c789_100_hardened/left_top_geometry/high_exp_review_pack`
  - Compiled templates: `results/c789_100_hardened/left_top_geometry/high_exp_manual_templates`
  - Stress calibration: `results/c789_100_hardened/left_top_geometry/high_exp_stress_locked`
  - Defect fusion: `results/c789_100_hardened/left_top_geometry/high_exp_defect_fused`
- Important high-exposure seed rule: when exporting the review pack, do not pass `--template-dir`; otherwise the editable masks are copied from the old templates instead of being generated from the high-exposure reference crop.
- Commands used:
  - `.venv/bin/python pipeline/13_export_geometry_review_pack.py --normal-root results/c789_100_hardened/left_top_geometry/high_exp_seed_crops --stress-root dataset/c789_stress_normal_group_split/locked/left/top --defect-root dataset/c789_100_left_top_parts/left/top/defect --output-dir results/c789_100_hardened/left_top_geometry/high_exp_review_pack --preset c789_left_top_3x2 --samples-per-split 2`
  - `.venv/bin/python pipeline/14_build_manual_geometry_templates.py --mask-dir results/c789_100_hardened/left_top_geometry/high_exp_review_pack/manual_masks --output-dir results/c789_100_hardened/left_top_geometry/high_exp_manual_templates`
  - `.venv/bin/python pipeline/12_geometry_eval.py --data-root dataset/c789_stress_normal_group_split/locked/left/top --template-dir results/c789_100_hardened/left_top_geometry/high_exp_manual_templates --output-dir results/c789_100_hardened/left_top_geometry/high_exp_stress_locked --calibrate-thresholds`
  - `.venv/bin/python pipeline/12_geometry_eval.py --data-root dataset/c789_100_left_top_parts/left/top/defect --template-dir results/c789_100_hardened/left_top_geometry/high_exp_manual_templates --thresholds results/c789_100_hardened/left_top_geometry/high_exp_stress_locked/geometry_thresholds.csv --anomaly-predictions results/c789_100_hardened/left_top_anomaly_dino/reports/predictions.csv --output-dir results/c789_100_hardened/left_top_geometry/high_exp_defect_fused`
- Fresh high-exposure seed metrics:
  - Stress locked geometry FP: `0/90`.
  - AnomalyDINO defect recall: `8/17`.
  - Geometry defect positives: `10/17`.
  - Fused defect recall: `13/17`.
  - Fused misses: `less_1_2_slot02`, `less_2_1_slot03`, `corner_3_1_slot05`, `surface_3_1_slot05`.
  - Key samples:
    - `less_1_2_slot02`: anomaly `0`, geometry `0`, final `0`, type `more`, region `r00_c06`, score `435.0`, threshold `2197.65`, missing `161`, extra `821`.
    - `less_2_1_slot03`: anomaly `0`, geometry `0`, final `0`, type `more`, region `r02_c02`, score `957.0`, threshold `1004.85`, missing `90`, extra `28888`.
    - `more_2_2_slot04`: anomaly `0`, geometry `1`, final `1`, type `more`, region `r00_c06`, score `512.0`, threshold `385.35`, missing `1415`, extra `19756`.
- Interpretation: high-exposure seed is useful as a cleaner visual/manual-edit starting point, but this unedited seed is not better than the previous best manual output because fused recall is `13/17` instead of `15/17`.

## C789 left_top high-exposure corner-only strategy

- On 2026-06-21, applied the corner-only strategy to the active high-exposure masks:
  - Active mask dir overwritten by request: `results/c789_100_hardened/left_top_geometry/high_exp_review_pack/manual_masks`
  - Backup before overwrite: `results/c789_100_hardened/left_top_geometry/high_exp_review_pack/manual_masks_backup/20260621_170113_corner_strategy_before_apply`
  - Final candidate source: `results/c789_100_hardened/left_top_geometry/high_exp_review_pack/corner_strategy_candidate_v5/manual_masks`
- The non-high-exp manual masks were not targeted:
  - `results/c789_100_hardened/left_top_geometry/manual_review_pack/manual_masks`
- Final mask semantics after applying the strategy:
  - `expected`: unchanged high-exp body masks, still full enough for alignment.
  - `watch_edge`: constrained to four corner windows, no longer full long-edge contour.
  - `allowed`: `expected` plus previous high-exp allowed tolerance only near the new four-corner watch area.
  - `ignore`: unchanged high-exp ignore masks.
- Final high-exp corner-only mask areas:
  - `slot01`: expected `369978`, allowed `400939`, ignore `39738`, watch_edge `234698`.
  - `slot02`: expected `428527`, allowed `457282`, ignore `39738`, watch_edge `225007`.
  - `slot03`: expected `484736`, allowed `515733`, ignore `39738`, watch_edge `237915`.
  - `slot04`: expected `491914`, allowed `520151`, ignore `39738`, watch_edge `230774`.
  - `slot05`: expected `559056`, allowed `584887`, ignore `39738`, watch_edge `195492`.
  - `slot06`: expected `570722`, allowed `597418`, ignore `39738`, watch_edge `199231`.
- Commands rerun after applying masks:
  - `.venv/bin/python pipeline/14_build_manual_geometry_templates.py --mask-dir results/c789_100_hardened/left_top_geometry/high_exp_review_pack/manual_masks --output-dir results/c789_100_hardened/left_top_geometry/high_exp_manual_templates`
  - `.venv/bin/python pipeline/12_geometry_eval.py --data-root dataset/c789_stress_normal_group_split/locked/left/top --template-dir results/c789_100_hardened/left_top_geometry/high_exp_manual_templates --output-dir results/c789_100_hardened/left_top_geometry/high_exp_stress_locked --calibrate-thresholds`
  - `.venv/bin/python pipeline/12_geometry_eval.py --data-root dataset/c789_100_left_top_parts/left/top/defect --template-dir results/c789_100_hardened/left_top_geometry/high_exp_manual_templates --thresholds results/c789_100_hardened/left_top_geometry/high_exp_stress_locked/geometry_thresholds.csv --anomaly-predictions results/c789_100_hardened/left_top_anomaly_dino/reports/predictions.csv --output-dir results/c789_100_hardened/left_top_geometry/high_exp_defect_fused`
- Fresh metrics after applying corner-only high-exp masks:
  - Stress locked geometry FP: `0/90`.
  - Geometry defect positives: `12/17`.
  - AnomalyDINO defect recall: `8/17`.
  - Fused defect recall: `14/17`.
  - Fused misses: `less_2_1_slot03`, `more_2_2_slot04`, `crack_3_2_slot06`.
  - Key samples:
    - `less_1_2_slot02`: anomaly `0`, geometry `1`, final `1`, type `more`, region `r00_c02`, score `821.0`, threshold `168.0`.
    - `less_2_1_slot03`: anomaly `0`, geometry `0`, final `0`, type `more`, region `r02_c02`, score `301.0`, threshold `316.05`.
    - `more_2_2_slot04`: anomaly `0`, geometry `0`, final `0`, type `more`, region `r00_c01`, score `8582.0`, threshold `9059.4`.
    - `corner_3_1_slot05`: anomaly `0`, geometry `1`, final `1`, type `more`, region `r00_c06`, score `11062.0`, threshold `4590.6`.
- Interpretation: corner-only high-exp masks are more aligned with the intended edge/corner inspection target and improved fused recall from `13/17` to `14/17` while keeping stress FP at `0/90`, but they no longer rescue `more_2_2_slot04` and still miss `less_2_1_slot03`.

## Industrial fusion robustness MVP-1

- On 2026-07-03, implemented the first slice from `CODEX_SOFTWARE_IMPLEMENTATION_PLAN.md`: CSV-based fail-closed fusion plus benchmark summary generation.
- New core module: `capture_data/fusion_engine.py`.
  - Normalizes existing geometry/anomaly/quality/registration CSV rows into `BranchPrediction`.
  - Emits `FusedDecision` states: `OK`, `NG_GEOMETRY`, `NG_ANOMALY`, `NG_CRACK`, `NG_GLOBAL`, `SUSPECT`, `RETAKE`, `INVALID_CAPTURE`.
  - Does not overwrite the older geometry `fused_predictions.csv` schema consumed by the manual geometry workflow.
- New pipeline entries:
  - `pipeline/18_fuse_inspection_results.py`: reads branch CSVs and writes `branch_predictions.csv`, `fused_predictions.csv`, `summary.md`.
  - `pipeline/19_run_robustness_benchmark.py`: summarizes fused decisions into `robustness_summary.md`, `robustness_summary.csv`, `by_defect_type.csv`, `by_slot.csv`, `misses.csv`, `false_positives.csv`, `retake_cases.csv`.
- MVP-1 benchmark wrapper is CSV-only. It accepts `--geometry-template-dir` and `--geometry-thresholds` for command compatibility, but does not rerun geometry/model inference yet; pass `--geometry-csv` and `--anomaly-predictions`.
- Code review follow-up on 2026-07-03 fixed two MVP-1 correctness risks:
  - Branch CSV rows now merge by strong aliases (`part_id`/`sample_id`/`id`, exact path, resolved path); basename/stem are weak aliases and only merge when they resolve to a single non-conflicting candidate.
  - `pipeline/19_run_robustness_benchmark.py` now enumerates input roots; defect/normal missing predictions stay in recall/FP denominators, while invalid-root missing predictions are tracked as `not_evaluated/missing_prediction`.
- Fusion config `ok_requires.quality_gate: PASS` and `ok_requires.registration: PASS` now require corresponding passing branch rows before a part can become `OK`.
- Follow-up double-check on 2026-07-03 tightened MVP-1 behavior:
  - Fixed CSV normalization so geometry/anomaly rows no longer hit an uninitialized `status` value.
  - `ok_requires` now rejects explicit `WARN` quality/registration rows when `PASS` is required.
  - `required_sides`/`required_views` are enforced from branch CSV `side/view` even without a manifest.
  - Robustness summaries include `fused_recall`, `geometry_recall`, and `anomaly_dino_recall`.
  - Missing benchmark inputs are written as `benchmark_input` branch trace rows, with filename-derived `slot_id` and known defect type when possible.
  - Scalar config values such as `branch_order: anomaly_dino` are treated as one value, not split into characters.
- User-confirmed MVP-1 policy on 2026-07-03:
  - No-config CSV mode remains permissive and may output `OK` for all-negative available branches.
  - `basename`/`stem` matching is allowed only as a unique weak match; conflicting same-name inputs must stay separate.
  - Invalid-root images with no branch prediction are tracked as `not_evaluated/missing_prediction`, not as successful invalid rejects.
- Validation run on 2026-07-03:
  - `.venv/bin/python -m compileall capture_data pipeline`
  - `.venv/bin/python pipeline/18_fuse_inspection_results.py --help`
  - `.venv/bin/python pipeline/19_run_robustness_benchmark.py --help`
  - Direct Python harness for `tests/unit/capture_data/test_fusion_engine.py` because `.venv` does not have pytest.
  - Direct Python harness for `tests/unit/pipeline/test_pipeline_wrappers.py`, including stage 18 CSV smoke and stage 19 missing-input benchmark checks.
  - Synthetic CLI smoke produced `OK`, `NG_ANOMALY`, and `RETAKE`; benchmark metrics included `defect_total=2`, `defect_detected=1`, and `missing_prediction_count=1`, with `benchmark_input` and `missing_prediction` trace rows.
  - `git diff --check` and 120-column checks on touched Python files passed.

## Industrial inspection MVP-3/4/5

- User-confirmed scope on 2026-07-04:
  - Use `/home/yunjing/anomalib/dataset` normal images as calibration references.
  - No invalid images have been collected yet; do not fake invalid calibration.
  - Quality gate online mode should start as `warn`, and whole-image quality metrics are enough for MVP-3.
  - Demo command uses `pipeline/5_demo_inspection.py` with `--quality-gate warn`; quality failures should display as WARN and show reasons in the UI.
  - Multi-view missing side/view should be treated as invalid capture.
- MVP-3 implementation:
  - `capture_data/quality_gate.py` computes whole-image `brightness_mean`, `brightness_std`, `saturation_ratio`, `dark_ratio`, `blur_laplacian_var`, `highlight_ratio`, and optional `foreground_coverage`.
  - `pipeline/17_calibrate_quality_gate.py` calibrates warn-mode thresholds from normal plus optional stress-normal roots, writes `quality_metrics.csv`, calibrated YAML, and a markdown report.
  - `quality_gate.csv` rows include `branch=quality_gate`, `status`, `fail_label`, `reason`, and `source_path`, so they can feed stage 18 fusion.
- MVP-4 implementation:
  - The existing demo `--quality-gate warn` path still continues model prediction.
  - WARN reasons are shown via the dashboard quality summary, added to post-face status messages, and written into trace/archive outputs.
  - Archive CSVs now include per-slot `quality_status`/`quality_reasons` and per-part top/bottom quality status/reasons.
- MVP-5 implementation:
  - `capture_data/multiview_manifest.py` supports explicit CSV schema and configurable filename regex parsing.
  - `pipeline/16_build_multiview_manifest.py` writes `manifest.csv` and prints `[invalid_capture]` when required side/view pairs are missing.
  - Until real file naming is fixed, prefer explicit CSV or pass `--filename-regex`; later move the regex into an inspection profile.

## C789 traditional operators and YOLO branch reservation

- On 2026-07-05, added the first offline C789 traditional-operator branch slice.
- New core module: `capture_data/traditional_operators.py`.
  - Reuses existing C789 full-capture presets instead of relocalizing parts from scratch.
  - Top preset: `c789_left_top_3x2`, ROI `460,30,3480,2600`.
  - Bottom preset: `c789_left_bottom_3x2`, ROI `350,320,3600,3030`.
  - Emits fusion-compatible rows with `part_id,side,view,slot_id,branch,pred_label,score,threshold,defect_type,reason,source_path,evidence_path,status`.
  - Branches: `registration`, `geometry`, `crack`, `surface_texture`, and `feature_presence`.
  - `registration` in WARN mode writes `status=WARN` with `pred_label=0`; only FAIL-mode registration rows should force fusion `RETAKE`.
  - Evidence overlays are written under `<output-dir>/evidence`.
- New pipeline entry:
  - `pipeline/20_run_traditional_operators.py`: runs traditional operators on full C789 captures or pre-cropped slot images and writes `traditional_predictions.csv`.
  - Stage 20 now prints per-image progress by default; use `--no-progress` to silence it.
  - Stage 20 also writes `traditional_cases.csv`, `traditional_summary.csv`, and `traditional_summary.md`.
  - Case-level FP/FN reporting groups by `source_path + slot_id`; `normal`/`normal_test`/`stress_normal` path parts count as normal, `defect` path parts count as defect, and `--label normal|defect|invalid|unknown` can override path inference.
- Traditional-operator enhancement decision on 2026-07-05:
  - Stage 20 should calibrate per-slot templates from normal images instead of relying only on global coarse rules.
  - New CLI surface: `--calibrate-normal-root PATH`, `--template-dir PATH`, and `--geometry-thresholds PATH`.
  - Calibration roots must be slot-crop datasets with filenames/paths containing `slotNN`; raw full-frame roots do not provide enough slot identity for template calibration.
  - Current C789 defaults use `calibration.max_images_per_slot=96`, `geometry.threshold_mode=region`, and `geometry.threshold_margin=0.25`; use `--calibration-max-images-per-slot 0` only for slower full-normal offline calibration.
  - Default calibration artifacts live under `<output-dir>/calibration/`; downstream defect/stress runs should pass the saved template directory and geometry threshold CSV explicitly.
  - `geometry` should prefer slot-template difference scoring; if a slot template is unavailable, keep a no-template fallback instead of failing the whole run.
  - `crack`, `surface_texture`, and `deform` scoring should be restricted to the material ROI so fixture/background texture does not drive decisions.
  - First-version `surface_texture` and `deform` positives should surface as `SUSPECT`, not hard `NG`, until calibrated with enough normal/stress-normal coverage.
  - Smoke on C789 top slots after slot-template calibration: `top_defect_slots_calibrated_region_margin` detected 11/17 defect slot crops; `top_normal_slots_calibrated_region_margin_smoke --max-images 180` had 8/180 normal positives. Treat this as an offline starting point, not a final production threshold.
- New config:
  - `config/traditional/c789.yaml`: conservative defaults; `registration` starts as WARN, and surface texture thresholds are intentionally conservative until calibrated from normal/stress normal.
- Fusion extension:
  - `pipeline/18_fuse_inspection_results.py` accepts repeated `--branch-csv BRANCH=PATH` values.
  - `capture_data/fusion_engine.py` preserves `evidence_path` in normalized `branch_predictions.csv`.
  - Default custom statuses include `feature_presence -> NG_GEOMETRY`, `surface_texture -> SUSPECT`, and `yolo -> NG_YOLO`.
- YOLO policy:
  - YOLO is reserved as a future branch only; no model training was added because bbox/seg labels are not present.
  - Future YOLO outputs should be slot-crop based and written as the same fusion-compatible CSV, then passed as `--branch-csv yolo=...`.
- C789 top slot-level traditional tuning on 2026-07-05:
  - Keep crack/surface as weak evidence for now; the real top defect smoke is still driven almost entirely by `geometry`.
  - Recommended tuned geometry thresholds for the current C789 top slot dataset:
    `results/c789_traditional/top_calibrated/calibration/geometry_thresholds_slot_tuned_v2.csv`.
  - Validation outputs:
    `results/c789_traditional/top_normal_tuned_v2/` and `results/c789_traditional/top_defect_tuned_v2/`.
  - Compared with the original calibrated thresholds, normal case FP changed from `100/1752` to `77/1752`.
  - Slot03 normal FP changed from `31/292` to `4/292`, while slot03 defect recall stayed `3/3`.
  - Defect recall changed from `11/17` to `13/17`; slot05 defect recall changed from `0/3` to `2/3`.
  - Slot05 normal FP increased from `9/292` to `13/292` because lowering `slot05/more/r00_c01` to catch `corner_3_1_slot05.png` costs several normal positives.
  - Do not blindly raise `slot03/more/r00_c04` or `slot03/more/r00_c05`: they overlap with existing slot03 defect scores, so clearing their remaining normal positives would lose current defect detections.

## C789 YOLO supervised detection fork

- On 2026-07-06, the YOLO plan moved from "future branch only" to an explicit supervised detection workflow.
- Ultralytics source lives outside this repo at `/home/yunjing/ultralytics-c789` on branch `c789-defect-yolo`.
  - Keep Ultralytics changes additive under `examples/c789/`.
  - Do not vendor Ultralytics source into anomalib; Ultralytics is AGPL-3.0 and commercial deployments may need the upstream enterprise license.
  - The fork is installed editable into the local uv environment with:
    `UV_CACHE_DIR=/tmp/uv-cache uv pip install -e /home/yunjing/ultralytics-c789`.
  - `examples/c789/train.py --batch` parses numeric strings into `int`/`float`; this avoids Ultralytics rejecting `--batch 32` as a string.
- New anomalib export entry:
  - `capture_data/prepare_yolo_dataset.py`
  - `pipeline/21_prepare_yolo_dataset.py`
  - Input is `part_crop_manifest.csv` plus bbox annotation CSV.
  - Output is Ultralytics detect format: `images/{train,val,test}`, `labels/{train,val,test}`, `data.yaml`, and `export_manifest.csv`.
  - First class schema is single-class `0: defect`; keep true `defect_type` as metadata for per-type reporting.
  - Annotate bbox on slot crop images, not raw 4024x3036 full captures.
  - Bbox annotation rows must use crop-level identifiers such as `processed_path`, `image_path`, or `sample_id`; do not match annotations through raw `source_path` or raw-frame `frame_id`.
  - `slot_box` from `part_crop_manifest.csv` is a crop location, not a defect bbox.
  - Missing bbox labels for `defect` rows fail closed by default; do not silently export defect crops as empty negative labels.
  - `output_root` reuse is fail-closed: pass `--overwrite` to intentionally clean a non-empty YOLO export directory.
  - `normal` and `normal_test` rows become empty-label negative examples; default maps `normal_test` to YOLO `val`.
  - Positive defect rows are deterministically assigned by source/sample id using `--positive-val-ratio`, `--positive-test-ratio`, and `--seed`.
  - Optional bbox overlay previews go to `--preview-dir` and should be manually checked before training.
- New defect collection helper:
  - `pipeline/22_collect_c789_yolo_defects.py` reuses `capture_data/collect_dataset.py` plus `capture_data/prepare_part_crops.py`; it does not add a new crop implementation.
  - Intended use: fill all six C789 fixture slots with defect parts for each manual load, then crop every slot as `defect` via `--defect-slot-mode all`.
  - YOLO collection crops must not hide/paint holes; stage 22 hardcodes `--hole-mask-method none` and does not expose a hole-mask CLI option.
  - Top uses `c789_left_top_3x2`; bottom uses `c789_left_bottom_3x2` and writes cropped workflow data under `bottom_ZS32`.
  - Final annotation candidates are flattened into `--defect-output-dir` with `defect_image_manifest.csv`; full crop metadata remains in `--parts-root/part_crop_manifest.csv`.
  - Stage 22 only prepares defect crop images for bbox labeling. Stage 21 still performs YOLO detect-format export after bbox CSVs exist.
  - If manual collection is interrupted with `KeyboardInterrupt`, already saved groups are still usable. Re-run stage 22 with `--skip-collect` and the same roots/hand/position to crop and flatten the existing raw session.
  - On 2026-07-06, interrupted C789 YOLO top defect collection after 10 groups under `dataset/c789_yolo_raw/left/top/defect/scratch/yolo_batch001_20260706_161959`; processed with:
    `.venv/bin/python pipeline/22_collect_c789_yolo_defects.py --skip-collect --hand left --position top --defect-type scratch --part-id yolo_batch001 --raw-root dataset/c789_yolo_raw --parts-root dataset/c789_yolo_parts --defect-output-dir dataset/c789_yolo_defect_images --overwrite`.
  - That run produced 60 defect crop images: 10 each for `slot01` through `slot06`; manifests are `dataset/c789_yolo_parts/part_crop_manifest.csv` and `dataset/c789_yolo_defect_images/defect_image_manifest.csv`.
  - Later on 2026-07-06, the user decided YOLO training images should keep the real holes. Stage 22 was changed to remove the hole-mask option and always crop with `hole_mask_method=none`; the same 10-group batch was reprocessed, producing 60 unpainted defect crops with manifest `hole_mask_method` all `none`.
- Slot policy:
  - Train one shared YOLO model across `slot01`-`slot06`; defect appearance can transfer between slots.
  - Validation must still report per-slot recall because slot pose/light/background differences can produce slot-specific misses.
  - Do not copy one slot's bbox coordinates into another slot; cross-slot reuse should happen through model training or carefully reviewed copy-paste synthesis only.
- Fusion policy:
  - YOLO predictions should be written as `yolo_predictions.csv` with `branch=yolo`, `pred_label`, `score`, `threshold`, `slot_id`, `source_path`, and optional `evidence_path`.
  - Stage 18 accepts this through `--branch-csv yolo=...`; default fusion branch order includes `yolo`, mapping positives to `NG_YOLO`.
- C789 YOLO offline augmentation:
  - Stage 23 (`pipeline/23_augment_yolo_dataset.py`) augments a flat YOLO `images/` + `labels/` defect dataset into `images/train`, `labels/train`, `data.yaml`, `augmentation_manifest.csv`, and bbox overlay previews.
  - On 2026-07-06, `/home/yunjing/ultralytics-c789/dataset/c789_all` was augmented into `/home/yunjing/ultralytics-c789/dataset/c789_all_augmented`.
  - The augmentation set is `orig`, `hflip`, `rot_m5`, `rot_p5`, `bright_gamma`, `blur`, `noise`, and `scale_translate`; all YOLO bbox labels are transformed and clipped with the image.
  - Verification for this output: 59 source images, 472 output train images, 472 output label files, 560 total boxes, 0 invalid labels, and 120 preview images.
  - Balanced train/val/test dataset for the first real YOLO retry lives at `/home/yunjing/ultralytics-c789/dataset/c789_all_balanced_yolo`.
  - It uses 39 real defect images augmented into 312 train defect images, 600 train normal images, 10 real val defect + 150 val normal, and 10 real test defect + 150 test normal.
  - Verification for the balanced dataset: train 912 image/label pairs with 392 boxes, val 160 pairs with 10 boxes, test 160 pairs with 11 boxes, and 0 invalid labels.
  - First balanced training run `defect_balanced_v1` lives at `/home/yunjing/ultralytics-c789/runs/detect/runs/c789/defect_balanced_v1`.
  - Diagnosis: metrics peaked early at epoch 13 (`mAP50=0.33846`, `recall=0.4`) and collapsed to all-zero validation metrics by epoch 100 while train loss kept falling (`train/cls_loss=0.04486`, `val/cls_loss=3.46323`), so treat it as small-data overfitting/generalization failure rather than a basic YOLO-format failure.
  - The run used `batch=64`; online detection augment still included default `hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4`, while geometric online augment was disabled. For the next retry, use smaller batch/patience and conservative HSV because offline augmentation is already present.
  - Second balanced training run `defect_balanced_v2` lives at `/home/yunjing/ultralytics-c789/runs/detect/runs/c789/defect_balanced_v2`.
  - `defect_balanced_v2` used `batch=16`, `patience=20`, no online geometry augmentation, and conservative HSV (`hsv_h=0.0`, `hsv_s=0.15`, `hsv_v=0.15`).
  - Diagnosis: v2 improved stability over v1 but still generalizes poorly. Best validation metrics were `mAP50=0.375` at epoch 30 and `mAP50-95=0.17779` at epoch 42; final epoch stayed nonzero (`precision=0.4`, `recall=0.2`, `mAP50=0.195`) but recall remained too low.
  - Low-confidence validation with `best.pt` at `conf=0.001` on CPU gave `precision=0.2308`, `recall=0.3`, `mAP50=0.295`, `mAP50-95=0.1535`; only 3 of 10 validation defect images produced any prediction and max confidence was only `0.0326`.
  - Train-split validation with the same `best.pt` at `conf=0.001` gave `precision=0.9974`, `recall=0.9896`, `mAP50=0.9948`, `mAP50-95=0.9531`, proving the model memorizes the augmented training set but does not generalize to held-out real defects.
  - Third balanced training run `defect_balanced_v2-3` lives at `/home/yunjing/ultralytics-c789/runs/detect/runs/c789/defect_balanced_v2-3`.
  - `defect_balanced_v2-3` changed to `yolo26m.pt` and `imgsz=1536` while keeping the same balanced dataset and conservative HSV settings.
  - Diagnosis: larger model plus higher resolution did not fix the problem and performed worse than v2. Best validation metrics were `mAP50=0.22227` at epoch 23 and `mAP50-95=0.05833` at epoch 36; final epoch was `precision=0.30852`, `recall=0.1`, `mAP50=0.03167`, `mAP50-95=0.00317`.
  - Low-confidence validation with `best.pt` at `conf=0.001` on CPU gave `precision=0.3822`, `recall=0.1`, `mAP50=0.0975`, `mAP50-95=0.04375`. Only a few validation defects produced weak predictions, while one normal image had a much higher false-positive confidence (`0.2156`).
  - Treat this as confirmation that full-slot supervised YOLO is bottlenecked by real defect diversity/problem framing, not model size alone. Prefer ROI-level YOLO or fusion-oriented evidence branches before further full-slot model scaling.
- Same-distribution YOLO validation dataset on 2026-07-06:
  - New builder: `capture_data/yolo_same_dist_dataset.py`; wrapper: `pipeline/25_prepare_yolo_same_dist_dataset.py`.
  - Output: `/home/yunjing/ultralytics-c789/dataset/c789_same_dist_yolo`.
  - Split rule uses filename `gNNN`; `g002,g008` are validation groups, and train defect images are augmented after split.
  - Counts: train has 47 real defect sources plus 600 empty-label normal sources, exported as 976 image/label pairs; val has 12 real defect images plus 150 empty-label normals, exported as 162 pairs.
  - Verification: image/label pairing matched, 469 total boxes, 750 empty labels, 0 invalid labels, 0 leakage rows.
- ROI-level YOLO datasets on 2026-07-06:
  - New builder: `capture_data/roi_yolo_dataset.py`; wrapper: `pipeline/26_prepare_yolo_roi_dataset.py`.
  - GT-centered diagnostic output: `/home/yunjing/ultralytics-c789/dataset/c789_roi_gt_center_yolo`.
  - GT-centered counts after clipping-safety fix: train `55 defect_roi + 110 normal_roi`; val `13 defect_roi + 150 normal_roi`.
  - Stage 26 writes only boxes fully contained in a ROI. Source boxes that cannot fit inside a 512x512 GT-centered ROI are skipped and written to `roi_review_report.csv`; current review rows are `g006 slot02 bbox0` and `g007 slot05 bbox0`.
  - Tiled deployable output: `/home/yunjing/ultralytics-c789/dataset/c789_roi_tile_yolo`, using 512x512 tiles with stride 256; counts are train `732 defect_roi`, val `140 defect_roi`, with empty tile labels retained as background examples.
  - Verification: same-dist and both ROI datasets have matching image/label counts, no label values outside `[0,1]`, no split leakage, and ROI label round-trip back to slot coordinates checked 229 boxes with 0 failures.
  - Pytest is still unavailable in `.venv`; verification used `py_compile`, direct target test-function harnesses, actual dataset label checks, and `git diff --check`.
- Traditional-operator semantic correction on 2026-07-06:
  - Treat traditional operators as fusion evidence branches, not complete defect-type classifiers.
  - `geometry` now writes `defect_type=geometry_delta` and uses `evidence_type=missing_mask|extra_mask|corner_delta|shape_delta`; internal `less/more` appears only as `raw_delta` in `reason`.
  - `gt_defect_type` is parsed from file names/paths for offline evaluation and reports; benchmark type recall should group by `gt_defect_type`, not geometry evidence labels.
  - Stage 20 now writes `traditional_defect_evidence_confusion.csv` and appends a `GT Defect Type vs Primary Evidence` table to `traditional_summary.md`.
  - Stage 24 (`pipeline/24_visualize_traditional_results.py`) renders contact sheets that display `GT` separately from `Evidence`, avoiding misleading labels such as `geometry less` as a defect class.
  - Stage 24 also writes per-case localization images under `visual_reports*/localization/`: it uses precise stage-20 `evidence_path` overlays when available and falls back to 4x8 `region=rXX_cYY` heat boxes from the geometry reason. Red means `missing_mask`, blue means `extra_mask`.
  - Example semantic smoke output: `results/c789_traditional/top_defect_semantic_v2/`; visual report: `results/c789_traditional/visual_reports_semantic/top_defect_semantic_v2_defect-all_page01.jpg`.

## ZS32 six-view dataset labeling decision (2026-07-12)

- Dataset roots: `dataset/right` and `dataset/left`; all `1,338` files are valid `4024x3036` RGB PNG images, with no YOLO txt, JSON/XML annotation, or masks yet.
- Treat the data as `223` independent physical sample groups expanded into six views, not as `1,338` independent samples. Every split must keep all six views from one `(hand, session, group_id)` together.
- Right hand: `113` normal groups and `29` defect groups (`deform=17`, `less=4`, `others=8`). Left hand: no real normal groups and `81` defect groups (`deform=35`, `less=42`, `others=4`). Combined defect diversity is `deform=52`, `less=46`, `others=12` physical groups.
- First YOLO baseline remains single-class detection: every visible defect bbox uses class `0: defect`; preserve `defect_type=deform|less|others` in the manifest/file metadata for per-type recall reports rather than training a multi-class head now.
- YOLO annotations are image-local: a defect part view with no visible defect gets an empty label file. Never draw a speculative box merely because another view shows that the physical part is defective.
- Split YOLO data by physical source group before augmentation. Mirrors, all six views, crops, and other derivatives of one source group must remain in the same split.
- For anomalib, train only on normal images and use one model per camera view as the baseline. Do not mix the six view distributions into one model.
- If right normal images are mirrored to synthesize left normal training data, swap view semantics after horizontal flip: `front_left <-> front_right`, `back_left <-> back_right`, while `front` and `back` remain unchanged. Keep each original and all mirrored derivatives in the same split.
- Synthetic left normal data cannot validate left-hand false-positive rate. Collect real left normal samples for validation/test before claiming left-hand deployment performance.
- Current six-view capture layout is not directly accepted by the existing stage-3 workflow: its registered view names do not include the new six names, and its defect glob is one directory level shallower than `defect/<defect_type>/<session>/images`.
- Label Studio Local Files staging was implemented on 2026-07-12:
  - Builder: `capture_data/prepare_zs32_label_studio.py`; wrapper: `pipeline/27_prepare_zs32_label_studio.py`.
  - Output: `dataset/zs32_yolo_labeling`; verified `660` PNG hard links, `110` physical groups, `left=486`, `right=174`, with source/output inode identity for every manifest row.
  - Label Studio document root: `/home/yunjing/anomalib/dataset/zs32_yolo_labeling`; Source Storage path: `/home/yunjing/anomalib/dataset/zs32_yolo_labeling/images`.
  - Use Import method `Files`, filter `.*\.png$`, and the generated single-class `label_studio_config.xml` with rectangle label `defect`.
  - Completed Label Studio YOLO export: `dataset/zs32_yolo_labeling/project-10-at-2026-07-12-12-29-b424b36b`; it contains `414` labeled images and `488` valid class-0 boxes. Missing label files were confirmed as intentional no-visible-defect views.
  - Stage 28 (`pipeline/28_prepare_zs32_yolo_dataset.py`) builds `dataset/zs32_six_view_yolo` and excludes `left/20260711_181850_552955/less/group027` entirely.
  - Verified stage-28 output: `2010` image/label pairs (`train=1398`, `val=306`, `test=306`), `222` source groups, `654` defect views, `678` real right normals, `678` horizontally mirrored left normals, `488` boxes, `0` invalid labels, and `0` group leakage. Real images are hard-linked; only mirrored normals consume new image storage (about `21G` total output tree accounting for hard links).
  - Stage 29 fixed-view ROI tool: `capture_data/zs32_view_roi_dataset.py`; wrapper: `pipeline/29_zs32_fixed_roi.py`.
  - Run `uv run --no-sync python pipeline/29_zs32_fixed_roi.py select` to choose one ROI for each of the six manifest `view` values. The selector shows exactly one clean right-hand normal reference image per view, with no bbox aggregation; left/right images share the corresponding view ROI. It saves `dataset/zs32_six_view_roi_config.json` plus six ROI overlay previews.
  - Run `uv run --no-sync python pipeline/29_zs32_fixed_roi.py convert` to create `dataset/zs32_six_view_roi_yolo`. It preserves split/sample semantics and empty labels, clips boxes crossing an ROI boundary, drops boxes completely outside the ROI, and reports both counts in the CLI and manifest.
  - Stage 29 code/test verification was completed before interactive selection; actual ROI coordinates and the 2010-image cropped output remain pending the user's GUI selections.
  - Current stage-29 ROIs verified on 2026-07-12: `front=(50,1020,3810,2520)`, `front_left=(0,0,3910,2420)`, `front_right=(570,260,3390,2150)`, `back=(390,930,4020,2510)`, `back_left=(400,0,4024,2540)`, `back_right=(410,120,3500,2430)`.
  - The final ROI config is `dataset/zs32_six_view_roi_config.json`. After the mirror correction, stage 29 regenerated `dataset/zs32_six_view_roi_yolo`: 2010 image/label pairs (`train=1398`, `val=306`, `test=306`), 488 retained boxes, 19 boundary-clipped boxes, 0 fully outside dropped boxes, 1596 empty labels, and 0 invalid labels.
  - Stage 29 convert now shows two Rich progress bars without changing the command: `Preflight` covers full image/label validation and `Cropping` covers crop/label writes; each total equals the split manifest row count.
  - Stage 29 mirror ROI correction: rows with `kind=normal_mirror` must use the configured `source_view` ROI transformed as `(image_width - x2, y1, image_width - x1, y2)`. Do not crop a mirrored full image with the target `view` ROI. `_preflight()` resolves this effective ROI once and passes it through label conversion, image cropping, and output manifest fields.
  - The current `dataset/zs32_six_view_roi_yolo` is the post-correction rebuild: all 678 `normal_mirror` rows match the source-view horizontal ROI formula, and the two named front-right/back-right mirror samples match their expected crops pixel-for-pixel with no obvious missing part edges.
  - Training smoke on 2026-07-12 used `/home/yunjing/miniconda3/envs/yolo/bin/python` with the local `/home/yunjing/ultralytics-c789` checkout. `yolo26n.pt`, `imgsz=1536`, `batch=16`, and 3 epochs completed train/val with peak 19.9 GiB on the RTX 4090; the run is only a pipeline smoke and its near-zero metrics are not a usable detector baseline.

## ZS32 multimodel decision fusion design (2026-07-13)

- Approved design: `docs/superpowers/specs/2026-07-13-zs32-multimodel-decision-fusion-design.md`.
- Production priority is preventing false negatives; the workflow supports human review and must retain raw continuous scores and visual evidence.
- Do not use majority voting across six views. A defect visible in one view cannot be cancelled by five clear views.
- Use fail-closed layered fusion: identity/completeness/quality gates, per-branch `T_low/T_high`, any `STRONG` evidence to `NG_*`, any `GRAY` or uncertainty to `REVIEW`, and `OK` only when every required view and branch is valid and CLEAR.
- Front-side inspection produces only `FRONT_CLEAR/FRONT_REVIEW/FRONT_NG`; final OK is forbidden until the same `part_id` completes all back-side views.
- Preserve `machine_status`, `review_status`, and `released_status` separately. Human review must not overwrite model evidence.
- Evidence records must keep original/ROI image identity, SHA-256, raw score, both thresholds, margins, model/threshold/ROI/template versions, heatmaps or detection/geometry overlays, reasons, and all triggering branches.
- Calibrate thresholds per product, hand, view, model/branch, and version. Split calibration and tests by physical `part_id`; report part-level escape rate and worst-group recall, not image-level accuracy alone.
- Reuse `capture_data/fusion_engine.py` and `pipeline/18_fuse_inspection_results.py`, extending them for dual thresholds, per-view PASS gates, all-trigger evidence, front/back state, JSON audit output, calibration, and fault-injection tests.
- Approved implementation plan: `docs/superpowers/plans/2026-07-13-zs32-multimodel-decision-fusion.md`; it splits delivery into backward-compatible dual-threshold evidence, strict per-view gates, staged audit records, stage-18 integration, offline stage-30 calibration, and final fault-injection/docs verification.
- Task 5 offline calibration implementation (2026-07-13):
  - Core module: `capture_data/fusion_calibration.py`; CLI: `pipeline/30_calibrate_zs32_fusion.py`.
  - Threshold groups are keyed by `(hand, view, branch, model_version, roi_version)`. `T_low` is the largest observed defect-score threshold satisfying inclusive `score >= T_low` target recall; `T_high` is `max(T_low, normal nearest-rank quantile)`.
  - `--fit-split` (default `calibration`) is the only split allowed to influence thresholds; `--eval-split` (default `test`) is held out for metrics. Missing selected splits fail closed, and a physical `part_id` still cannot cross splits.
  - `--required-view` is only a fit-data existence gate. Use repeatable exact `--required-group HAND:VIEW:BRANCH:MODEL_VERSION:ROI_VERSION` values for heterogeneous view-specific branch/version contracts; the implementation never invents a view/branch Cartesian product.
  - Groups missing either normal or defect calibration evidence write `status=insufficient_data` with both thresholds set to JSON/CSV null rather than inventing deployment values. Missing required evaluation evidence injects GRAY/REVIEW, sets `calibration_valid=false`, and suppresses recall/confidence-upper claims.
  - Metrics use physical `part_id` OR/GRAY semantics, so six CLEAR rows from one defect part count as one escape. A valid zero-escape result reports `1 - 0.05 ** (1 / defect_part_count)` as the exact one-sided 95% upper bound; `non_clear_recall` makes the GRAY-inclusive safety meaning explicit.
  - Stage 30 writes `thresholds.json`, `thresholds.csv`, `calibration_metrics.json`, and `calibration_summary.md` via same-directory atomic replace. It never edits `config/fusion/zs32_six_view.json`.
  - Verification environment: use `/home/yunjing/anomalib/.venv/bin/python -m pytest` from this worktree; the worktree-local `.venv` has no pytest and default `uv` cache is read-only. Task-5 focused calibration and wrapper verification reached `63 passed`, with new-file Ruff, `py_compile`, `--help`, and `git diff --check` passing.
- Task 6 strict-fusion documentation and fault verification (2026-07-13):
  - Implementation commits through Task 5 are `db2ea06d`, `623182a7`, `8a74bb67`, `6e90f273`, `73a64d31`, `fe7584c9`, `745530cc`, `70e412e3`, and `21d52822`. Operational entrypoints are stage 18 (`pipeline/18_fuse_inspection_results.py`) and stage 30 (`pipeline/30_calibrate_zs32_fusion.py`); deployed profile is `config/fusion/zs32_six_view.json`.
  - Stage 18 writes `branch_predictions.csv`, `fused_predictions.csv`, `summary.md`, and `audit/<part_id>.json`. Stage 30 writes `thresholds.json`, `thresholds.csv`, `calibration_metrics.json`, and `calibration_summary.md` without modifying the deployed profile.
  - The table-driven stage-18 fault matrix covers missing view/branch, duplicate view/branch identity, mismatched part identity, quality FAIL, registration WARN, non-finite score, model/ROI version mismatch, missing source/evidence artifacts, and YOLO no-box plus a GRAY anomaly. Initial RED exposed four gaps: duplicate and version mismatch could return OK, while non-finite input aborted before audit; the final focused command `uv run --no-sync python -m pytest tests/unit/pipeline/test_fuse_inspection_results.py -k fault -q` reports `13 passed, 6 deselected` and every emitted audit keeps `released_status=null`.
  - Complete target verification command uses the five planned test files and reports `140 passed`. `py_compile` for all five production modules, JSON validation, stage-18/stage-30 `--help`, task-file Ruff, task-file format check, and `git diff --check` pass.
  - Environment update: `uv 0.11.16` and the worktree `.venv` are currently usable; both `uv run` and `uv run --no-sync` resolve to the worktree environment. The plan-wide Ruff check still reports 60 pre-existing style findings, and the plan-wide format check reports two previously unformatted Task1-3 files (`capture_data/fusion_engine.py`, `tests/unit/capture_data/test_fusion_engine.py`); Task6-owned changed test/audit files are clean and formatted, and existing debt was left untouched.
