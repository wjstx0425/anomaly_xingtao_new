# AGENTS Memory

## BMW right multisource Demo integration lives on the BMW worktree (2026-08-11)

- The newly trained assets under `results/bmw_lab_one_click/bmw_right_multisource_left_yolo_v1` are integrated in
  `.worktrees/bmw-eight-view-handoff` rather than merged into this dirty ZS32 root branch. The Demo profile there
  selects the right-hand ROI, eight new Template/EfficientAD models, new YOLO, the newly calibrated 5% whole-part
  EfficientAD thresholds, and a detector-version-compatible right bright-streak config.
- EfficientAD threshold evidence separates 41 branch-negative parts (33 true business normals plus 8 no-streak
  business NG) from the business-normal metric. The tightened asset gives 1/41 branch-negative false NG (2.44%),
  1/33 business-normal false NG (3.03%), and 7/7 visible-defect detection on the same demo-only selection data.
  It pins all eight checkpoint SHA-256 values, the Demo pins the threshold artifact SHA, and `back_left` is guarded
  at `0.001` rather than the former subnormal value. The complete score artifact is
  `results/bmw_lab_one_click/bmw_right_multisource_left_yolo_v1/efficientad/score_analysis/part_thresholds.json`.
- The compatible right bright-streak recalibration remains weak (final-test balanced accuracy 0.453); do not claim
  that branch is accepted until a new right-hand bright ROI/rule is validated. The end-to-end offline smoke sample
  `bmw_right_normal_group039_000001` passed all 25 checks and produced
  `results/bmw_eight_view_demo/right_multisource_group039_smoke.png`.

## ZS32 EfficientAD six-view benchmark audit blocker (2026-07-21)

- A post-training same-image runtime candidate benchmark now exists at `results/zs32_efficientad_s_six_view_seed42_v1/runtime_benchmark_same_image_20260721.json`. Six resident EfficientAD-S models on RTX 4090, one warmup plus five formal canonical rounds over `zs32_timing_worker_parallel_v1/run_02/crops/patchcore`, measured six `Engine.predict` calls at 1.717-1.787 s, median 1.748 s. The matching current PatchCore stage median is 3.301 s, so the measured Engine segment saves 1.553 s / 47.0%; holding every other Stage32 cost fixed projects 7.006 -> 5.454 s (-22.2%), but this is not yet an integrated end-to-end EfficientAD result.
- EfficientAD checkpoint restoration totaled 0.762 s and the first six-view warmup totaled 2.077 s. Do not compare the 0.762 s directly with the full PatchCore worker startup because imports, Engine setup, GPU transfer, and runtime publication boundaries differ.
- The six artifacts are EfficientAD-small, 256x256, FP32 checkpoints for primary views only. Current Stage32 hardcodes PatchCore loading/manifest names, v10 binds PatchCore model and profile hashes, and EfficientAD reports only one deploy threshold per view rather than the required low/high pair. Never replace v10 paths directly; first build a separate candidate runtime/evidence adapter and benchmark score/map/mask/evidence plus Stage18 under a new bundle.
- Historical context: the earlier requested EfficientAD-S/M training benchmark stopped after its mandatory read-only audit. The later user-supplied `zs32_efficientad_s_six_view_seed42_v1` checkpoints supersede that no-checkpoint state for speed testing only; they do not resolve the documented split or runtime-publication limitations.
- Authoritative source/ROI lineage is `dataset/zs32_all_right_patchcore_roi/crop_manifest.csv`: 2456 unique source and output paths. Each primary view has 306 normal and 28 defect crops; model input remains `256x256`. Its `roi_config.json` is byte-identical to `/home/yunjing/anomalib/dataset/zs32_eight_view_roi_config.json`, SHA-256 `9412b2838cdb96f722db356714cf9bbbb5ea01810671ccf92b67323de77ebe65`.
- Active v10 still binds the historical `results/zs32_patchcore_eight_view_all_seed42` checkpoints. Per primary view that old preprocessing manifest has 334 rows but only 279 unique output paths/files because session identities collided. The six measured checkpoint hashes still match v10 exactly; preserving hash integrity does not make that training lineage valid.
- `results/zs32_patchcore_eight_view_all_seed42_v2` fixes path identity (334 rows/files per primary view: 245 train normal, 61 `normal_test`, 28 defect), but the legacy datamodule uses `val_split_mode=same_as_test` and its threshold/FPR reporting reuses `normal_test`; it has no independent calibration/final-test partition.
- The raw ROI manifest has no independently verified physical-part identifier. `session_id__groupNNN` is only a capture-identity proxy, so the required physical `part_id` train/calibration/test non-overlap cannot be proved. Do not launch the 12-model benchmark until an operator supplies or approves a canonical physical-part mapping and immutable three-way split artifact.
- EfficientAD resources are complete: small teacher SHA-256 `a16ded54719674435576aee641152616a640dfc6dc2b83115dab6e226610ae7d`, medium teacher SHA-256 `f7356663c8e00ada12ae01fb8c8aad0a1de2f800f8eadf252a46d29bbdfdf718`, and ImageNette tree `/home/yunjing/anomalib/.cache/anomalib/imagenette/imagenette2` SHA-256 `820188d384d211ab438e2fd1e6200fe3fea2685ac0756bc53af3b2dcf49d1dc3` (13,394 verified images). The checked-in example config remains unusable because its paths/hashes are placeholders and seed is 43.
- The strict `zs32-train-anomaly` backend supports small/medium and immutable hash-bound candidates but currently lacks a real EfficientAD recipe plus dataset-release/materialized-export publication. Legacy Stage 8 is not an acceptable substitute without adding model-size selection, explicit three-way physical split, held-out evaluation, resume-safe per-view outputs, requested metrics, and segmented benchmark timing.
- v10 thresholds remain commissioning-only and explicitly declare `data_leakage=true`, `test_used_for_selection=true`, `fit_split=calibration+test`, `evaluation_split=test_reused_for_selection`, PatchCore model rebinding, and four manual PatchCore overrides. Never use them as EfficientAD thresholds or as clean held-out accuracy evidence.

## ZS32 demo timing and resident inference worker (2026-07-17)

- The 2026-07-21 capture CPU pass runs four independent in-memory HDR fusions and four PNG encodes concurrently after SDK acquisition. Camera/result order and the first canonical error remain topology-ordered; workers return outcomes/timings and only the main thread mutates `TimingRecorder`. Existing grouped SDK reads remain the separately authorized prior optimization.
- New critical-path fields are `hdr_fusion_parallel_wall` and `image_encoding_parallel_wall`. The old `hdr_fusion`/`image_encoding` fields now sum per-worker CPU time and must not be interpreted as elapsed wall time. A real-size offline four-front-view benchmark using the latest saved 4024x3036 images measured HDR 1.742 -> 0.683 s (2.55x) and PNG 2.290 -> 0.978 s (2.34x), with exact fused pixels and PNG bytes; peak process RSS was about 2.88 GiB. This is not a live-camera benchmark.
- Capture-parallel verification: 253 directly related adapter/bootstrap/service/live/collector tests passed; targeted `py_compile` and `git diff --check` passed; independent review found no remaining Critical/Important issues. The broader capture-data directory still has 19 unrelated existing failures from topology prompt fixtures and private-file permission expectations.
- The 2026-07-19 lossless parallel pass decodes the eight independent source PNGs with a bounded four-thread pool and evaluates the six primary prepared Template groups with six threads. Decode results and Template results are still collected in canonical view order; all eight Template crop PNGs are written before evaluation, and `front_secondary/back_secondary` are never submitted to Template in the 20/22-group profiles.
- The authoritative parallel benchmark is `results/zs32_timing_worker_parallel_v1/run_01..05` after `warmup`: all five runs are `OK` and complete. Median Stage32+Stage18 is 7.006 s (range 6.826-7.296), versus io-v3 median 8.155 s (-14.1%). Source decode falls from 0.896 to 0.299 s (-66.6%); Template aggregate falls from 1.339 to 0.906 s (-32.3%), with the six-view parallel evaluate wall at 0.123 s. CSV semantic projections and every source/crop/evidence PNG byte plus all six raw NPY arrays exactly match io-v3 run_02.
- Second lossless I/O pass removes eight temporary YOLO ROI PNGs by passing canonical contiguous BGR ndarrays, overlaps PatchCore raw/mask/overlay writers with later view inference, writes YOLO evidence with two bounded workers, reuses the in-memory PatchCore crop for overlays, and computes each archived source SHA once. `sources/` remains an independent `copy2` archive; do not replace it with hardlinks.
- The authoritative post-I/O benchmark is `results/zs32_timing_worker_io_v3/run_01..03`: complete Stage32+Stage18 totals 8.171/8.155/8.061 s (median 8.155 s), all `OK` and complete. The old 3.798 s I/O critical path is 1.276 s median (0.710 source/crop preparation + 0.136 PatchCore writer residual wait + 0.430 YOLO evidence), a 66.4% reduction; total worker time improves 30.7% from 11.765 s. Per-view async writer seconds are summed CPU work, not wall time.
- Worker v2 versus all three io-v3 runs match projected manifest fields, PatchCore/YOLO/Template/fusion CSV values, 20-group decisions, source/evidence hashes, six raw NPY arrays, PatchCore mask/overlay pixels, and eight YOLO evidence pixels. `zs32_timing_worker_io_v1` used the wrong default 22-group profile and failed closed; `io_v2` was infer-only; neither is acceptance evidence.
- The active bundle remains immutable `results/zs32_runtime_bundle_eight_view_20group_patchcore_tuned_v10/runtime_bundle.json`; no v10 model, threshold artifact, or bundle file was overwritten.
- Real same-image cold Stage32+Stage18 timing was 24.260 s internally (25.49 s external; the pre-instrumentation external baseline was 27.89 s). The top bottleneck was six PatchCore loads at 10.801 s, followed by Template at 2.412 s and YOLO batch at 1.457 s.
- Live Dashboard now owns an independent serial Unix-socket inference worker. It preloads exactly six primary PatchCore models, one eight-view YOLO, and six primary Template groups. Stage35 keeps capture/result validation/final complete publication and retains the legacy Stage32 subprocess fallback.
- Worker startup preload measured 10.888 s and runs while the Dashboard/operator capture flow proceeds. The final prepared worker same-image Stage32+Stage18 run measured 11.765 s, a 51.5% reduction from the instrumented 24.260 s cold run.
- Stage32 decodes the eight originals once and shares them between Template and model ROI generation. Hikvision grouped capture still triggers all four cameras first, then performs one blocking read per distinct camera handle in parallel and restores topology order.
- Per-generation `timing.json` records front/back capture, HDR short/long, HDR fusion, image encoding/write, Stage35 child/worker wall time, runtime config, Template, PatchCore load/per-view inference, YOLO load/batch inference, evidence writes, Stage18, Dashboard parse/render/display, and click-to-first-frame total.
- Historical v10, cold instrumented, two initial worker generations, and final prepared-worker generation have identical projected final status, scores, locked thresholds, YOLO detections/boxes, 20 Stage18 identities, and secondary Template/PatchCore SKIPPED states on the saved live group001 images.
- Precision-sensitive A/B options were not adopted: YOLO 960/640 changed detections; PatchCore autocast FP16 changed scores/maps for only 1.8% aggregate inference gain; reducing Template count changed back_right to NG_TEMPLATE. TensorRT export was blocked by absent ONNX/TensorRT packages. Camera interval/timeout/settle A/B still requires a real operator/hardware run.
- Detailed report and command: `docs/ZS32_DEMO_SPEED_OPTIMIZATION_20260717.md`. Final headless screenshot: `artifacts/zs32_timing_worker_v2_dashboard.png` (1600x920).
- Verification at this checkpoint: the second I/O pass has 103 targeted model-runtime/Stage32 tests passing, and the final expanded directly related pipeline/model/live regression is 354 passed. Earlier Dashboard/capture/worker groups remain 135/28/5 passed. Broader unrelated coverage exposed 2 old runtime-bundle CLI import failures, 7 old strict-fusion fixture ROI-id failures, and 1 old compositor pixel-equality failure; these were not modified as part of the timing/worker optimization.

## ZS32 Stage35 live HDR alignment (2026-07-15)

- The operator's authoritative historical Stage1 HDR command uses short/long exposure `1500/5500 us`, gain `0`, capture interval `0.5 s`, one HDR settle frame, and timeout `5000 ms`.
- Stage35 now locks those explicit values in `build_capture_command` and also locks the referenced Stage1 fusion defaults: dark/clip thresholds `70/245`, blend width `18`, blur size `31`, two HDR retries, max clip `12%`, and no MTB alignment.
- This fixes acquisition-domain drift only. The previous live run used different bootstrap defaults (`4000/35000`, blend `50`, blur `101`, no retry); its existing fused images are not rewritten because short/long source frames were not saved.
- Minimal verification: the exact capture-command unit test passed, targeted `py_compile` and `git diff --check` passed. No camera or GPU run was performed.

## ZS32 dashboard minimal UI blockers (2026-07-15)

- Offline `--result-dir` dashboards now render a disabled `离线结果` CTA; keyboard and mouse inspection actions cannot enqueue live work without a live result/controller context.
- Live `complete` progress no longer reloads, hashes, and decodes the same eight-view result every 50 ms. It reloads only on the first complete record or when the result path changes.
- The dashboard test fixture now gives available YOLO detections a valid ROI, and the stale secondary-view unsupported assertion was removed because `MODELED_VIEWS == VIEW_ORDER`.
- Minimal verification: 79 parser/render/app tests passed with the UI-only CUDA gate bypassed via `--confcutdir=tests/unit/zs32_refactor/dashboard`; real group064 headless rendering, targeted `py_compile`, and `git diff --check` passed. In this sandbox, set `UV_CACHE_DIR=/tmp/uv-cache` because the default uv cache is read-only.
- Current debug screenshot: `artifacts/zs32_dashboard_eight_view_commissioning/debug_dashboard.png`. It remains commissioning evidence only: overall `REVIEW`, Fusion `SKIPPED`, not production acceptance.

## ZS32 live dashboard progress/control Tasks 6+7 (2026-07-15)

- Bootstrap now supports paired `--progress-json/--control-json` operation without a TTY. Each front/back wait publishes a fresh confirmation token, consumes one exact command, and publishes `capturing_front/back` immediately before camera capture.
- Stage32 publishes `running_template`, `running_patchcore_yolo`, and `running_fusion` at the real operation boundaries. Stage35 owns the final strict-contract `complete` publication and the fail-closed `failed` fallback.
- `Stage35Controller` owns one new-session subprocess group, rejects duplicate starts, writes each waiting token once, translates nonzero exits to failed progress, and closes with SIGTERM then SIGKILL on timeout.
- Live dashboard mode uses one contextual Start/Confirm CTA. Before a result exists it renders an empty eight-view shell; only `complete` loads result evidence through the strict parser. `--no-gui` performs no HighGUI calls.
- Focused verification at the implementation checkpoint: Task6 `63 passed`, Task7 `24 passed` (`87 passed` combined); CLI help, targeted `py_compile`, `git diff --check`, and a decodable `1600 x 920` headless fake screenshot passed. No real camera/GPU smoke was run.

## ZS32 topology-driven Stage35 live capture (2026-07-15)

- Stage35 consumes `results/zs32_runtime_bundle_eight_view_v2/runtime_bundle.json` plus `configs/zs32/topology/zs32_4cam_double_side_v1.json`; serial overrides and separate runtime/template/threshold CLI assets are removed.
- Capture delegates to `pipeline/zs32_bootstrap_capture.py` with one right-hand normal group/image and publishes an exact eight-view legacy manifest. Manifest loading verifies view, round, camera serial, session, sample, and group identity.
- Before any camera child runs, Stage35 validates the finalized bundle and its runtime-assets, Template model, generated fusion profile, and threshold bindings. Stage32 receives those inner assets and all eight image arguments.
- Device discovery on 2026-07-15 found all four topology serials: `DA9805574`, `DA9625347`, `DB0998274`, and `DB0968108`. Real capture remains a foreground two-round operator action requiring the part and Enter confirmations.

## ZS32 24-group commissioning bundle Task 4 (2026-07-15)

- Task4 code/test commit is `f4efec44` (`feat: publish ZS32 eight-view commissioning fusion`).
- The immutable `results/zs32_runtime_assets_eight_view_v1` publication uses the pre-hardening manifest shape and cannot be finalized by current Stage37 because it lacks the exact `fusion_profile_template` path+SHA binding. Keep it unchanged; do not add a legacy compatibility bypass.
- The current versioned chain is `results/zs32_runtime_assets_eight_view_v2` -> `results/zs32_24group_commissioning_v2` -> `results/zs32_runtime_bundle_eight_view_v2`. The final bundle loads in a fresh process with exact 24 identities and finite thresholds.
- The v2 thresholds intentionally use the user-authorized Stage33 v4 YOLO test-leakage auxiliary source. They declare `data_leakage=true`, `test_used_for_selection=true`, `commissioning_only=true`, and `production_release_allowed=false`; never describe this bundle as production.
- Stage32 accepts `--fusion-config PATH` and passes that exact bundle-generated profile to Stage18. The no-GPU Stage18 replay smoke is `results/zs32_stage18_24group_bundle_profile_smoke_v1`; it consumed 24 rows from the existing same-identity group064 CSVs and returned the already-known `NG_YOLO` back-view trigger.
- Task4 verification: 122 focused tests passed, py_compile/help/diff-check passed, and Ruff E/F/W excluding the unrelated pre-existing E501 at Stage32 line 537 passed. Detailed evidence is in `.git/sdd/task-4-report.md`.

## ZS32 strict eight-view Stage33 Task 3 (2026-07-15)

- Template, Stage33 case construction, threshold preflight, per-view YOLO calibration, and missing-score validation now reuse `zs32_inspection.domain.views.VIEW_ORDER` and require exact right-hand eight-view coverage.
- Template `load_model` rejects any digest-valid model whose declared view order or hand-by-view groups are not exact. Stage33 rejects missing per-view calibration normal/defect rows, physical-part fit/test reuse, runtime/Template ROI or template-generation mismatch, and missing/extra YOLO score identities.
- Focused RED was 4 failed/34 passed for the four missing contracts; focused GREEN is 38 passed. The existing real `results/zs32_template_gate_right_eight_view_v1` passes the exact `load_model` eight-group command, so it was not retrained.
- Code/test commit is `a263ca2f`; its clean-HEAD export passed 73 focused/related tests, compile, CLI help, and full targeted Ruff.
- Real Stage33 is blocked before GPU: `results/zs32_runtime_assets_eight_view_v1` is an older immutable publication. Its runtime/fusion bytes match the current source, but its asset-set SHA is `ea1756c...` rather than current publisher output `dda14d6...` because it uses the obsolete scalar `fusion_profile_template_sha256` instead of the current path+SHA binding. Do not delete or overwrite it; choose a new immutable runtime-assets directory before rerunning publish and Stage33.
- Evidence and exact continuation commands are in `.git/sdd/task-3-report.md`. Generated runtime/template/calibration artifacts remain uncommitted.

## ZS32 versioned runtime bundle publisher Task 2 (2026-07-15)

- Commit `50bbc22a` adds the immutable two-phase publisher, Stage 37 CLI, and checked-in replaceable source declaration.
- Publication enforces exact-eight identity, content-distinct checkpoints, mandatory ROI/template generation equality, Template containment, canonical SHA bindings, and atomic `renameat2(RENAME_NOREPLACE)` no-clobber publication.
- Finalization accepts only the complete Task1 exact-24 commissioning threshold contract bound to the generated fusion profile plus exact runtime-assets and Template hashes. `reported_deploy_threshold` remains provenance only.
- Clean-HEAD verification: 74 focused tests, py_compile, both CLI subcommands, and Ruff F/E/W/I passed. Evidence is in `.git/sdd/task-2-report.md`.
- The existing PatchCore overlay change remains user-owned and was excluded from the Task 2 commit.
- Main-review follow-up: real Stage34 records intentionally omit per-record `threshold_version` and `template_version`; validate those versions through top-level `threshold_versions` and `deployment_contract.expected_versions`, while records remain exact ordered five-field identities. Threshold runtime/template bindings and the fusion profile template require exact canonical paths plus hashes. Profile branch/rule/order contracts are exact.
- Follow-up commit `b7701918`; clean HEAD passed 85 Task2/CLI/runtime tests plus compile/help/Ruff. Dirty worktree tests call the real uncommitted Stage34 `_compose_records`; clean HEAD uses its exact field-for-field contract fallback because that Task1 helper is not yet committed.
- Final schema follow-up commit `b7898122`: manifest, asset set, runtime assets, and bundle accept only schema version 1; all published/loaded asset-set, runtime, and bundle view orders must exactly equal canonical `VIEW_ORDER`. Clean HEAD passed 94 Task2/CLI/runtime tests plus compile/help/Ruff.

## ZS32 eight-view PatchCore heatmap fix (2026-07-15)

- User-reported case: known-normal `group064`, old fused output `results/zs32_eight_view_24group_test_leakage_smoke/group064`.
- Initial evidence: all eight Stage32 PatchCore crops are byte-identical to the corresponding training/evaluation preprocessed inputs (matching SHA-256), and all eight Stage32 `pred_score` values exactly match the training workflow `predictions.csv` rows for group064.
- The eight configured checkpoint files exist and their measured SHA-256 values match `config/fusion/zs32_runtime_models_eight_view.json`.
- Root cause: the old `capture_data/zs32_model_runtime.py::_write_patchcore_overlay` min-max normalized each image independently, while the training `ImageVisualizer` uses the checkpoint post-processor's absolute `[0, 1]` scale with `normalize=False`. Normal group064 maps with maxima only `0.278317` to `0.397034` were therefore stretched to red and appeared spatially dispersed.
- Fix: `_write_patchcore_overlay` now reuses Anomalib `visualize_anomaly_map(..., normalize=False)` and the training visualizer's 0.5 overlay alpha. Raw maps, scores, masks, thresholds, and decisions are unchanged. Regression: `test_patchcore_overlay_preserves_absolute_anomaly_scale` first failed because proportional low/high maps rendered identically, then passed after the fix.
- Same-image/same-checkpoint front comparison is `results/zs32_patchcore_group064_heatmap_diagnosis/front_before_fix/comparison.json`: native `Engine.predict(ckpt_path=...)` and runtime `load_from_checkpoint` plus `predict(ckpt_path=None)` produced exactly equal raw maps and scores. The warning only means Engine did not reload a second time.
- Complete eight-model GPU smoke: `results/zs32_eight_view_runtime_heatmap_fixed_smoke_complete/group064`; it has 8 PatchCore rows, 8 YOLO rows, 16 overlays, `errors=[]`, and intentionally remains diagnostic `REVIEW` without locked thresholds/template/fusion. `diagnostics/patchcore_old_new_numeric_comparison.json` proves all eight old/new raw maps are array-equal, all score diffs are zero, all crop bytes match, and only overlay rendering changed.
- Verification: focused Stage32/runtime suite `87 passed`; `py_compile`, `git diff --check`, and Ruff F/E/W passed. Full Ruff/format still reports unrelated pre-existing import/style and indentation findings elsewhere in the already-modified files; do not reformat those unrelated regions as part of this fix.

## ZS32 strict eight-view application contract Task 1 (2026-07-15)

- The authoritative dependency-free view tuple is `src/zs32_inspection/domain/views.py`: `VIEW_ORDER`, `MODELED_VIEWS`, and `CANONICAL_VIEWS` are the same ordered eight-view tuple (`front`, `front_left`, `front_right`, `front_secondary`, `back`, `back_left`, `back_right`, `back_secondary`).
- Stage32 now requires eight explicit image arguments; runtime requires eight distinct PatchCore checkpoints, exact eight-view PatchCore/YOLO ROIs, and preserves canonical order for one eight-image YOLO batch.
- Dashboard manifests require `model_supported=true` for every view. Compositor notices are determined by each branch state; there is no secondary-view unsupported shortcut.
- Review correction: both orchestrator and Stage32 always evaluate all eight template views before aggregating the gate. A template stop publishes all eight actual Template states and explicit score-free `SKIPPED` PatchCore/YOLO/Fusion records; strict dashboard parsing rejects `UNSUPPORTED` for every core branch.
- Review correction: the Stage32 eight-view CLI is right-only and accepts/defaults only `zs32-right-24-commissioning`; it uses `config/fusion/zs32_runtime_models_eight_view.json` and always publishes `commissioning_only=true`, `production_release_allowed=false`. Legacy six-view production remains confined to historical Stage18 profiles.
- Review correction: malformed or non-positive YOLO boxes fail closed; they are never silently discarded. Runtime configs require exact product `ZS32` and supported hands `['right']`.
- Legacy right-hand production remains the historical six-view, 36-group profile. Eight-view commissioning uses 24 groups (eight views x template/PatchCore/YOLO) and remains non-production.
- Task 1 focused verification is 190 passed; Stage32 `--help` lists both `--front-secondary-image` and `--back-secondary-image`. Full dirty-worktree/TDD evidence is in `.git/sdd/task-1-report.md`.

## ZS32 offline dashboard Task 5 checkpoint (2026-07-14)

- Offline dashboard commits are `e71e18fc` (renderer, reducer, headless/GUI app,
  CLI, Stage 36 wrapper and tests) and `e79f4f0c` (only the
  `zs32-dashboard` console-script registration).
- The stable acceptance result is
  `artifacts/zs32_dashboard_offline_acceptance/result`; its part ID and reason
  explicitly mark it as an offline fixture. The corresponding OpenCV-decoded
  `1600 x 920` uint8 screenshot is
  `artifacts/zs32_dashboard_offline_acceptance/zs32_dashboard_1600x920.png`.
- Launch the accepted offline GUI with
  `uv run --no-sync python pipeline/36_zs32_inspection_dashboard.py --result-dir /home/yunjing/anomaly_xingtao_new/artifacts/zs32_dashboard_offline_acceptance/result`.
- Task 5 stops at the mandatory offline acceptance checkpoint. Do not implement
  Tasks 6-9 or claim live lifecycle support until the user explicitly accepts
  this display. `--live` currently fails with a clear not-connected error.
- Final Task 5 verification: dashboard `141 passed`, CLI `5 passed`, compile,
  help, focused Ruff/format, diff check, and real Stage 36 screenshot generation
  passed. Full evidence is in `.git/sdd/task-5-report.md`.
- Review follow-up `de5d29ef` makes disabled inspection actions return the
  exact existing state (preserving any pending request), changes active-layer
  and CTA outlines from 2 px to the frozen 1 px style, and adds a real
  subprocess regression for the Stage 36 wrapper. Updated counts are dashboard
  `143 passed` and CLI `6 passed`; the same acceptance screenshot path was
  regenerated and OpenCV-validated after the border fix.

## ZS32 eight-view inspection dashboard design (2026-07-14)

- User approved the complete design for a pure OpenCV result dashboard; design document: `docs/superpowers/specs/2026-07-14-zs32-eight-view-inspection-dashboard-design.md`.
- Detailed TDD implementation plan: `docs/superpowers/plans/2026-07-14-zs32-eight-view-inspection-dashboard.md`. Tasks 1-5 end at a mandatory offline acceptance checkpoint; Tasks 6-9 cover four-camera Stage35, structured progress/control, live process-group integration, and the real one-part smoke.
- The display order is `front`, `front_left`, `front_right`, `front_secondary`, `back`, `back_left`, `back_right`, `back_secondary`. Four-camera Stage35 captures all eight images from one sample; the current six primary views enter the existing models while the two secondary views show real originals and `model unsupported` until their weights and calibration assets are trained.
- The UI is an independent process with a 4 x 2 result grid, Fusion/Original/PatchCore/YOLO/Template layers, one Start button, mouse plus keyboard control, card enlargement, and Quit. It deliberately has no history browser, open-directory action, in-window file picker, product selector, or part-ID editor.
- Live mode starts Stage35 as an owned child process and polls an atomically written progress JSON instead of parsing terminal output. Offline mode loads one `--result-dir`; both modes use the same strict identity-validating parser.
- The single detection button is stateful: Start while idle, Confirm front and capture during `waiting_front`, and Confirm back and capture during `waiting_back`. Progress publishes a unique `confirmation_id`; the GUI atomically writes a matching control JSON, so four-camera capture needs no terminal Enter or TTY. Stale, duplicate, wrong-round, or wrong-part confirmations are rejected.
- The GUI starts Stage35 in its own process group and terminates the owned group on exit so capture or Stage32 grandchildren cannot be orphaned.
- PatchCore output must preserve the raw anomaly map and binary mask. Prefer real `pred_mask`; otherwise use a configurable normalized anomaly-map display threshold defaulting to `0.65` and label it `DIAGNOSTIC MASK`. The fallback never changes model or fusion decisions.
- Existing YOLO output is box detection, so the dashboard may draw only true boxes and confidence; template score/offset alone cannot be presented as a pixel mask. `NG_TEMPLATE` is a valid template-first short circuit, while missing/mismatched artifacts remain execution errors rather than NG.
- Delivery is gated: implement and obtain user acceptance of the offline parser/compositor/dashboard first, then extend four-camera Stage35 and connect live Start/progress, then run one real eight-image/six-model hardware smoke test. The two secondary model rows must fit later without changing the GUI result contract.

## ZS32 PatchCore dashboard artifacts (2026-07-14)

- Stage32 persists each modeled view's untouched finite 2D anomaly output as float32 `.npy`, a binary uint8 mask PNG, and the existing heatmap overlay below `evidence/patchcore/{raw_maps,masks}` plus the overlay root. `runtime_manifest.json.views` contains only the six `MODELED_VIEWS`; secondary views must never receive fabricated model records.
- A valid Anomalib `pred_mask` takes priority and records no diagnostic threshold. If absent, the runtime min-max normalizes the raw map and applies `--diagnostic-mask-threshold` (default `0.65`) only to the display mask. This value never changes the score, locked thresholds, Stage18 input, or OK/NG decision.
- Non-finite raw maps, invalid thresholds, or present-but-malformed `pred_mask`/artifact geometry fail closed. The manifest may retain a finite diagnostic score for display, but the affected PatchCore branch CSV row has no score and cannot enter Stage18.

## ZS32 four-camera legacy-layout capture design (2026-07-14)

- User approved and implementation completed for direct legacy-format four-camera output. Design: `docs/designs/2026-07-14-zs32-four-camera-legacy-layout-capture-design.md`; operator commands: `configs/zs32/topology/README.md`.
- Final CLI is explicit `--legacy-layout` on `pipeline/zs32_bootstrap_capture.py`. It writes only the historical dataset tree plus one `manifests/<session>.csv`, not duplicate `_bootstrap` images. Without the flag, isolated bootstrap behavior is unchanged.
- Historical paths remain `<root>/<hand>/<view>/normal/<session>/images` or `defect/<defect_type>/<session>/images`; filenames retain `{hand}_{view}_{label_token}_{part_id}_groupNNN_000001_{fused|single}.png`.
- Four-camera legacy samples extend the unchanged 25-column CSV from six image rows to eight image rows plus one sample row. New views are `front_secondary/back_secondary`, both bound to `DB0968108` by topology.
- Hardware handoff uses root `/home/yunjing/anomalib/dataset/test`, topology `configs/zs32/topology/zs32_4cam_double_side_v1.json`, `images-per-group=1`, HDR `1500/5500 us`, gain `0`, interval `0.2 s`, settle frames `1`, and timeout `2000 ms`. Run `group-count=1` first and require 8 PNGs plus 9 data rows; only after operator confirmation run 120 groups and require 960 image plus 120 complete sample rows, 1080 data rows total.
- Stage 2 discovers arbitrary view directories and can see secondary images. Stage 3, YOLO/Label Studio, and parts of training remain six-view constrained; downstream eight-view expansion is explicitly out of this capture-format task.
- Final offline gate on 2026-07-14: capture-focused suite `57 passed`, topology suite `9 passed`; `py_compile`, CLI `--help`, topology JSON/legacy CSV contract checks, and `git diff --check` passed. This is not evidence of a real one-group or 120-group hardware capture.

## ZS32 formal four-camera topology (2026-07-14)

- Checked-in topology: `configs/zs32/topology/zs32_4cam_double_side_v1.json`; operator guide: `configs/zs32/topology/README.md`. The fourth camera is the front-mounted secondary camera, serial `DB0968108`, slot `front_secondary`, producing `front_secondary` in the front round and `back_secondary` after flipping the same part.
- The four-camera topology has eight required views in this order: `front`, `front_left`, `front_right`, `front_secondary`, `back`, `back_left`, `back_right`, `back_secondary`. Its current canonical topology SHA256 is `e516aa4fc95d31800c67720926b0855ceba881eed99a4cc4d57d0509e026ae39`.
- The legacy `pipeline/1_collect_multicamera_data.py` remains a fixed three-camera/six-view collector and must not be described as four-camera capable. Its `--list-devices` command is still useful for confirming all four serials. Fast four-camera data collection now uses `pipeline/zs32_bootstrap_capture.py` or the `zs32-bootstrap-capture` console script; do not modify the legacy collector or replace its right serial to emulate four cameras.
- The bootstrap command keeps the existing `group-count`, `images-per-group=1`, `manual-load`, HDR and exposure surface. It deliberately rejects `images-per-group>1` rather than changing the old collector's flip semantics, does not support `--save-hdr-sources`, and leaves HDR alignment off unless `--align-hdr` is explicit. It opens all topology cameras once for the entire batch, asks for Enter before front/back grouped rounds, and atomically publishes eight-view sets below `<root>/_bootstrap/<session>/<set>`. Failures and KeyboardInterrupt go to `_bootstrap_incomplete`. Bootstrap manifests always mark `bootstrap_only=true`, `eligible_for_dataset=false`, and `production_release_allowed=false`; they deliberately omit formal gate/canonical raw manifests.
- The current `.venv` cannot import `MvCameraControl_class` without an SDK path. Prefix hardware commands with `PYTHONPATH=/opt/MVS/Samples/64/Python/MvImport:${PYTHONPATH:-}`; that exact path exists on this host and a direct import succeeds. The adapter intentionally does not mutate `sys.path`.
- Formal four-camera capture must use `zs32-capture` with a new immutable gate publication bound to the four-camera topology. Before publishing it, prepare real eight-view quality/registration profiles and eight approved full-frame reference PNGs. Do not copy old thresholds or references into `front_secondary/back_secondary`.
- ROI does not block the first formal raw capture. It does block dataset publication: the current `configs/zs32/roi/zs32_roi_v2.json` and `configs/zs32/recipes/zs32_right_patchcore_training_v1.json` are three-camera assets and cannot be combined with four-camera captures. A later four-camera dataset needs eight measured ROIs, eight-view annotations/targets, eight templates, eight PatchCore candidates, a retrained global YOLO, and new calibration artifacts.
- TDD evidence: the checked-in-config regression first failed because `zs32_4cam_double_side_v1.json` did not exist, then passed after adding the config. The bootstrap CLI suite first failed because its module was absent, then passed seven behavior cases covering 8-image binding, one adapter open per batch, early rejection of ambiguous multi-image groups, front/back partial failure isolation, Ctrl-C retention/cleanup, and hardware-free help. Use `--confcutdir` on the focused test folders to bypass the repository CUDA skip hook.

## ZS32 Stage35 live 18-group commissioning (2026-07-14)

- Operator entrypoint: `pipeline/35_run_zs32_live_commissioning.py`; reusable orchestration: `capture_data/zs32_live_commissioning.py`. Preflight with `/home/yunjing/anomalib/.venv/bin/python pipeline/35_run_zs32_live_commissioning.py --list-devices`, then run one right-hand part with the same interpreter and `--part-id live_part_001`.
- Template-operator diagnostic command: `/home/yunjing/anomalib/.venv/bin/python pipeline/35_run_zs32_live_commissioning.py --part-id live_part_001 --diagnostic-skip-template`. It captures the same six views, runs Stage32 `infer` with all six PatchCore branches plus YOLO, and prints absolute `<output_dir>/patchcore.csv` and `<output_dir>/yolo.csv` paths. It intentionally creates no `template_match.csv`, fusion output, or Stage18 audit and remains `REVIEW`, `inspection_complete=false`, `strict_fusion=false`, `commissioning_only=true`, `production_release_allowed=false`; never use it as an OK/NG or release result.
- Stage35 locks one manual-load HDR sample (`1500/6000 us`, gain `0`, settle frames `1`) and serial-bound cameras front/centre `DA9805574`, left `DA9625347`, right `DB0998274`. The operator confirms the front three views, flips the same right-hand part, then confirms the back three views.
- Default capture root is `/home/yunjing/anomalib/results/zs32_live_capture`; default runtime root is `/home/yunjing/anomalib/results/zs32_live_runtime/<capture_session>/<sample_id>`. Runtime assets are the local commissioning config, unified-ROI template model, and `/home/yunjing/anomalib/results/zs32_18group_commissioning_v1/thresholds.json`.
- The default Stage35 flow invokes `fuse --fusion-profile zs32-right-18-commissioning` and does not expose quality, registration, geometry, GUI, hand, profile, or group-count switches. Every business result remains `commissioning_only=true`, `production_release_allowed=false`.
- A complete fusion result has a Stage18 audit and 18 canonical template/PatchCore/YOLO rows. A validated Stage32 template short circuit (`NG_TEMPLATE` or exception-shaped `REVIEW`) is also a legal business result: it has `inspection_complete=false`, prints verified `template_results`, and deliberately returns `audit_path=None` because Stage18 did not run. Do not classify missing audit as a contract error in this short-circuit case. Source-image decode/dimension failures currently make Stage32 nonzero and are execution errors, not `INVALID_CAPTURE` summaries.
- Focused tests are `tests/unit/pipeline/test_zs32_multimodel_inference.py`, `tests/unit/capture_data/test_zs32_live_commissioning.py`, and `tests/unit/pipeline/test_zs32_live_commissioning_cli.py`; right/left prompt regressions remain in `tests/unit/capture_data/test_collect_multicamera_dataset.py`.
- Final skip-template diagnostic gate on 2026-07-14 was `233 passed`; Ruff/format, compilation, CLI help, diff check, and the Stage35 real-artifact validator passed. Real `--list-devices` returned all three configured serials with exit code 0.
- The saved correct-right-hand capture `20260714_145508_515286` was reused without recapturing for a real GPU diagnostic at `/home/yunjing/anomalib/results/zs32_live_runtime_diagnostic/20260714_145508_515286/live_part_001_group001_000001`. It produced 6 PatchCore plus 6 YOLO rows, 12 evidence files, no runtime errors, no template/fusion/audit artifacts, and the required REVIEW/incomplete/non-production flags. PatchCore scores were front `0.313845`, front_left `0.888506`, front_right `1.0`, back `0.364892`, back_left `0.796209`, back_right `0.919070`; all YOLO scores were `0.0` on this normal part.
- Hardware preflight saw one RTX 4090 with PyTorch `2.8.0+cu128` and CUDA available. All six PatchCore checkpoints plus YOLO `best.pt` existed and matched the SHA-256 values in `runtime_models.local.json`; both ROI configs existed and `validate_18_group_source_assets` accepted the threshold/runtime/template binding.

## ZS32 unified-ROI offline commissioning (2026-07-14)

- The authoritative reference/training dataset for this commissioning pass is `/home/yunjing/anomalib/dataset/zs32_patchcore_roi_yolo`; do not substitute the older `zs32_patchcore_roi` tree.
- The right-hand six-view template gate was trained from that tree's `crop_manifest.csv` and published at `/home/yunjing/anomalib/results/zs32_template_gate_right_unified_roi_v1`. It contains six groups, five templates per group, and a `model.json` SHA-256 of `db3730a5e6559ef734b77c67c8f82b2b06c0165719533ebf929dd8f0e3ad330d`.
- Fresh serial PatchCore training from the same dataset completed under `/home/yunjing/anomalib/results/six_view_roi_yolo_fixed_seed42`; `six_view_summary.csv` has exactly six rows, every checkpoint exists, and `failed_views.txt` is absent. This is the commissioning model set; do not mix it with checkpoints from `six_view_roi_fixed_seed42` or `six_view_roi_wrn_l23_r005_bs16_fp32_seed42`.
- Local ignored runtime config: `/home/yunjing/anomalib/results/zs32_offline_commissioning/runtime_models.local.json`. It pins those six fresh checkpoints, PatchCore ROI `/home/yunjing/anomalib/dataset/zs32_patchcore_stage29_roi_config.json`, YOLO ROI `/home/yunjing/anomalib/dataset/zs32_six_view_roi_config.json`, and YOLO `/home/yunjing/ultralytics-c789/final_n640_p1_seed42/weights/best.pt` (`8c9495d150a008db2c9b0a464848d92c06088d24a33960404d549c0412ca15e2`).
- A real GPU `infer` smoke on right normal `group038` succeeded at `/home/yunjing/anomalib/results/zs32_offline_commissioning/group038`: six template rows all PASS, six PatchCore rows with nonzero scores, six YOLO rows with zero detections, 12 evidence PNGs, 18 crop PNGs, and no runtime errors.
- The expected terminal state is `REVIEW` with `inspection_complete=false`: Stage32 `infer` intentionally has no locked dual-threshold artifact or quality/registration/geometry evidence. PatchCore/YOLO CSV row status is therefore `ERROR` with reason `locked dual thresholds unavailable` even though continuous inference and evidence generation succeeded. Do not call this a completed production inspection.
- Fresh PatchCore deployment recall was weakest on `right_front_left` (`0.689655`) and `right_back_right` (`0.724138`); inspect these views during calibration instead of accepting the smoke result as an accuracy sign-off.

## ZS32 right-hand unified PatchCore + YOLO runtime (2026-07-13)

- Use `pipeline/32_run_zs32_multimodel_inference.py` as the single entrypoint. `infer` preserves continuous evidence and remains REVIEW; `fuse` calls strict Stage 18 only when all required external branches and locked thresholds are supplied.
- Runtime code is `capture_data/zs32_model_runtime.py`; pinned local assets are declared in `config/fusion/zs32_runtime_models.json` with SHA-256. It loads six right-hand PatchCore checkpoints from `results/six_view_roi_fixed_seed42` and YOLO `best.pt` from `/home/yunjing/ultralytics-c789/final_n640_p1_seed42/weights`.
- PatchCore and YOLO must keep separate ROI configs. YOLO `candidate_conf=0.001` is an evidence collection floor; no legacy single deploy threshold is authoritative for final fusion.
- Template matching, when configured, runs on PatchCore ROI crops before model backends and short-circuits on any non-PASS result.
- `config/fusion/zs32_right_six_view.json` is the right-only 36-group strict contract exposed as `--profile zs32-right`. The original `config/fusion/zs32_six_view.json` remains the two-hand 72-group contract.
- No final OK is allowed without a valid Stage-31 artifact and complete template/quality/registration/PatchCore/YOLO/geometry evidence. Missing inputs fail closed to REVIEW or the appropriate non-release state.
- Both full and right-only strict profiles recompute evidence bands from locked continuous thresholds; never trust a CSV `evidence_level` over `score/low/high`. Stage 32 `fuse` requires the online template model and does not accept an old template CSV as a gate replacement.
- Real end-to-end CPU smoke output `/tmp/zs32-stage32-real-smoke` confirmed 6 PatchCore + 6 YOLO evidence rows, 12 calibration rows, all 12 overlays, no runtime errors, and the expected REVIEW/incomplete result without locked fusion inputs.

## ZS32 PatchCore left/right per-view ROI dataset (2026-07-13, design approved)

- Add a standalone stage 30 tool for cropping `dataset/right` and `dataset/left` into a reusable PatchCore tree under `dataset/zs32_patchcore_roi`; do not extend stage 29 YOLO behavior or crop dynamically inside training.
- Use 12 independent ROIs: six canonical views for each of `right` and `left`. Store pixel half-open `xyxy` coordinates in `dataset/zs32_patchcore_roi_config.json` and previews under `dataset/zs32_patchcore_roi_previews`.
- Preserve the hand-relative `view/{normal,normal_test,defect}/<defect_type>/<session>/images` tree. Write `crop_manifest.csv`, `roi_config.json`, and `summary.json`; original datasets remain unchanged.
- Validate filename hand/view identity, but always use the original parent view directory for ROI selection and output routing. Filename-view mismatches are never moved; `resolved_view` equals `source_view` and `view_corrected` stays false.
- Default to no overwrite. `--overwrite` may rebuild only after a complete preflight validates all inputs and target mappings.
- Approved design: `docs/superpowers/specs/2026-07-13-zs32-patchcore-roi-dataset-design.md`. Stage 30 is implemented, the 12-ROI config/previews exist, and the full cropped dataset was generated under `dataset/zs32_patchcore_roi`.
- Implementation plan: `docs/superpowers/plans/2026-07-13-zs32-patchcore-roi-dataset.md`; it separates strict config/discovery, interactive selection, transactional conversion, and CLI/docs/final verification into four TDD tasks.
- Task 1 routing was revised after user review: filename hand/view validation remains, while normal completeness, ROI selection, output paths, manifest, and summary all follow the original parent view directory. The 24 filename/directory mismatches remain in their original folders.
- Task 2 complete and reviewed: `select_patchcore_rois()` preloads 12 readable normal references, checks one common source size, then opens right-six followed by left-six selectors, reuses an existing config as initial ROIs, writes `<hand>_<view>_roi.png` previews, and atomically replaces the config only after all selections/previews succeed. Tests reached `32` passing cases; reviewer noted only a Minor missing injected mid-selection exception test, not a functional defect.
- Minimal stage-30 entrypoint: `pipeline/30_crop_zs32_patchcore_dataset.py`. `convert` shows `Checking images` and `Cropping images` progress bars. The directory-preserving dataset was regenerated and verified with `2010` images (`right=852`, `left=1158`), `routed_changes=0`, `view_corrected=0`, complete manifest paths, and untouched source files. The focused suite passes `50` tests plus Ruff F/I, `py_compile`, and `git diff --check`.
- Historical ROI training results based on the earlier routing were deleted before this directory-preserving recrop and must not be reused. Retraining should create fresh `results/six_view_roi_fixed_seed42`.
- Strong ROI PatchCore runner prepared (not executed): `pipeline/run_patchcore_roi_six_views.sh`. It opts into configurable defaults exposed by `run_wrn50_fixed_six_views.sh` and fixes `wide_resnet50_2`, layers `layer2 layer3`, coreset `0.05`, float32, train/eval batch `16`, workers `2`, k `9`, image size `256x256`, deploy FPR `0.05`, and seed `42`. Its isolated default output is `results/six_view_roi_wrn_l23_r005_bs16_fp32_seed42`, suffix `wrn_l23_s256_r005_k9_fp32_bs16_fpr005_seed42`. The user will run training manually.

## ZS32 six-view fixed PatchCore training (2026-07-13, complete)

- Requested `/home/ljl/anomaly_xingtao` and `/DATA/ljl/right` do not exist on this machine. The adjusted checkout is `/home/yunjing/anomalib` (the historical `mygithub` remote pointed to `git@github.com:wjstx0425/anomaly_xingtao.git`) and the adjusted data root is `/home/yunjing/anomalib/dataset/right`.
- Fixed output root: `/home/yunjing/anomalib/results/six_view_fixed_seed42`; GPU: `0`; suffix: `wrn_l2_s256_r001_k9_fp16_fpr005_seed42`.
- Fixed PatchCore settings: `wide_resnet50_2`, `layer2`, `256,256`, ROI `full`, coreset `0.01`, neighbors `9`, feature precision `float16`, train/eval batch `4`, workers `2`, normal holdout `0.2`, deploy FPR `0.05`, seed `42`.
- Views are `right_front`, `right_front_left`, `right_front_right`, `right_back`, `right_back_left`, and `right_back_right`, trained serially with independent output roots/checkpoints/thresholds.
- The strict offline timm preload check passed: `HF_HUB_OFFLINE=1 .venv/bin/python -c 'import timm; timm.create_model("wide_resnet50_2", pretrained=True, features_only=True, out_indices=(2,)); print("wide_resnet50_2 cached")'` printed `wide_resnet50_2 cached`.
- `examples/api/03_models/zs32_defect_workflow.py` now calls `seed_everything(args.seed, workers=True)` at the start of every `(view, model)` training iteration, before datamodule/model construction.
- Serial/resumable runner: `pipeline/run_wrn50_fixed_six_views.sh`. Per-view outputs are `<output-root>/<view>`; per-view orchestration logs append to `<output-root>/<view>/runner.log`; durable failures append to `<output-root>/failed_views.txt`.
- TDD/static verification before real training: the runner tests first failed because the script was absent, then the combined related suite passed `43` tests; `bash -n`, `py_compile`, focused Ruff `F/I`, and `git diff --check` passed.
- The real host exposes one RTX 4090 on GPU 0. At the pre-training check, another user-owned YOLO sweep (`examples/c789/sweep_zs32.py --stage all --experiment-id zs32_full_v2 --resume`) occupied about 24.6 GiB; do not terminate it.
- Real execution command: `HF_HUB_OFFLINE=1 bash pipeline/run_wrn50_fixed_six_views.sh /home/yunjing/anomalib/dataset/right /home/yunjing/anomalib/results/six_view_fixed_seed42 0`.
- All six independent train and evaluate jobs succeeded. The first merge attempt correctly failed closed because its model-schema check expected `patchcore` while the workflow writes the suffixed run name; after a TDD regression fix to require `patchcore_<suffix>`, rerunning the identical command skipped all six completed views and generated the aggregate without retraining or overwriting results.
- Final aggregate: `/home/yunjing/anomalib/results/six_view_fixed_seed42/six_view_summary.csv`; verified exactly six rows, and every referenced checkpoint and per-view summary exists. `failed_views.txt` is absent.
- Final image/sample deployment metrics are identical at this dataset grouping level:
  - `right_front`: threshold `0.3961148560`, FPR `0.0434782609`, accuracy `0.9807692308`, recall `1.0`, F1 `0.9830508475`.
  - `right_front_left`: threshold `0.6611694098`, FPR `0.0434782609`, accuracy `0.8076923077`, recall `0.6896551724`, F1 `0.8`. This is the expected weak view and was not tuned differently.
  - `right_front_right`: threshold `0.4201563299`, FPR `0.0434782609`, accuracy `0.9807692308`, recall `1.0`, F1 `0.9830508475`.
  - `right_back`: threshold `0.3071553111`, FPR `0.0434782609`, accuracy `0.9807692308`, recall `1.0`, F1 `0.9830508475`.
  - `right_back_left`: threshold `0.4350364804`, FPR `0.0434782609`, accuracy `0.9807692308`, recall `1.0`, F1 `0.9830508475`.
  - `right_back_right`: threshold `0.4216867387`, FPR `0.0434782609`, accuracy `0.9807692308`, recall `1.0`, F1 `0.9830508475`.

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
- Push current code changes to the new repository (`git@github.com:wjstx0425/anomaly_xingtao_new.git`); keep the historical repository untouched unless the user explicitly requests otherwise.
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
- Approved implementation plan: `docs/superpowers/plans/2026-07-13-zs32-multimodel-decision-fusion.md`; it splits delivery into backward-compatible dual-threshold evidence, strict per-view gates, staged audit records, stage-18 integration, offline stage-31 calibration, and final fault-injection/docs verification.
- Task 5 offline calibration implementation (2026-07-13):
  - Core module: `capture_data/fusion_calibration.py`; CLI: `pipeline/31_calibrate_zs32_fusion.py`.
  - Threshold groups are keyed by `(hand, view, branch, model_version, roi_version)`. `T_low` is the largest observed defect-score threshold satisfying inclusive `score >= T_low` target recall; `T_high` is `max(T_low, normal nearest-rank quantile)`.
  - `--fit-split` (default `calibration`) is the only split allowed to influence thresholds; `--eval-split` (default `test`) is held out for metrics. Missing selected splits fail closed, and a physical `part_id` still cannot cross splits.
  - `--required-view` is only a fit-data existence gate. Use repeatable exact `--required-group HAND:VIEW:BRANCH:MODEL_VERSION:ROI_VERSION` values for heterogeneous view-specific branch/version contracts; the implementation never invents a view/branch Cartesian product.
  - Groups missing either normal or defect calibration evidence write `status=insufficient_data` with both thresholds set to JSON/CSV null rather than inventing deployment values. Missing required evaluation evidence injects GRAY/REVIEW, sets `calibration_valid=false`, and suppresses recall/confidence-upper claims.
  - Metrics use physical `part_id` OR/GRAY semantics, so six CLEAR rows from one defect part count as one escape. A valid zero-escape result reports `1 - 0.05 ** (1 / defect_part_count)` as the exact one-sided 95% upper bound; `non_clear_recall` makes the GRAY-inclusive safety meaning explicit.
  - Stage 31 writes `thresholds.json`, `thresholds.csv`, `calibration_metrics.json`, and `calibration_summary.md` via same-directory atomic replace. It never edits `config/fusion/zs32_six_view.json`.
  - Verification environment: use `/home/yunjing/anomalib/.venv/bin/python -m pytest` from this worktree; the worktree-local `.venv` has no pytest and default `uv` cache is read-only. Task-5 focused calibration and wrapper verification reached `63 passed`, with new-file Ruff, `py_compile`, `--help`, and `git diff --check` passing.
- Task 6 strict-fusion documentation and fault verification (2026-07-13):
  - Implementation commits through Task 5 are `db2ea06d`, `623182a7`, `8a74bb67`, `6e90f273`, `73a64d31`, `fe7584c9`, `745530cc`, `70e412e3`, and `21d52822`. Operational entrypoints are stage 18 (`pipeline/18_fuse_inspection_results.py`) and stage 31 (`pipeline/31_calibrate_zs32_fusion.py`); deployed profile is `config/fusion/zs32_six_view.json`.
  - Stage 18 writes `branch_predictions.csv`, `fused_predictions.csv`, `summary.md`, and `audit/<part_id>.json`. Stage 31 writes `thresholds.json`, `thresholds.csv`, `calibration_metrics.json`, and `calibration_summary.md` without modifying the deployed profile.
  - Twelve table-driven stage-18 matrix cases cover missing view/branch, duplicate view/branch identity, split part IDs, quality FAIL, registration WARN, non-finite score, model/ROI version mismatch, missing source/evidence artifacts, and YOLO no-box plus a GRAY anomaly. `split_part_ids` specifically proves that rows split across two physical IDs become two incomplete groups and neither group is OK; a separate thirteenth fault test constructs front/back face decisions with different IDs and proves `combine_face_decisions` rejects the identity mismatch. The face-identity test was GREEN on its first run because the staged combiner already had the required guard; no RED was fabricated.
  - Initial matrix RED exposed four gaps: duplicate and version mismatch could return OK, while non-finite input aborted before audit. The final focused command `uv run pytest tests/unit/pipeline/test_fuse_inspection_results.py -k fault -v` reports exactly `13 passed, 7 deselected`; every matrix audit keeps `released_status=null`.
  - Complete target verification command uses the five planned test files and reports `141 passed`. `py_compile` for all five production modules, JSON validation, stage-18/stage-31 `--help`, task-file Ruff, task-file format check, and `git diff --check` pass.
  - Environment update: `uv 0.11.16` and the worktree `.venv` are currently usable; both `uv run` and `uv run --no-sync` resolve to the worktree environment. The plan-wide Ruff check still reports 60 pre-existing style findings, and the plan-wide format check reports two previously unformatted Task1-3 files (`capture_data/fusion_engine.py`, `tests/unit/capture_data/test_fusion_engine.py`); Task6-owned changed test/audit files are clean and formatted, and existing debt was left untouched.

### ZS32 final review closure (2026-07-13)

- This section supersedes the earlier Task 5/6 verification counts and incomplete strict-contract description.
- Strict identity is now `product=ZS32`, `profile=zs32_six_view_v1`, one consistent `hand` selected from `left/right`, `side=zs32`, exact `view`, and exact `branch`. Normalized CSV and audit records retain `product`, `profile`, `hand`, `source_hash`, and `manifest_identity`.
- `config/fusion/zs32_six_view.json` contains 72 non-wildcard `expected_versions` records: 2 hands x 6 views x 6 required branches, including `template_match`. Every `(hand,side,view,branch)` requires exact model, threshold, ROI, and template versions; missing or mismatched values fail closed.
- Reusing a source path, declared source hash, manifest identity, or computed image hash across required views marks the inspection incomplete and blocks release. A valid STRONG `NG_*` remains the immutable machine result; system/capture/version/evidence faults are retained as additional triggers and keep `inspection_complete=false` rather than downgrading machine NG.
- Non-strict legacy C789 `SUSPECT` remains exactly `SUSPECT`; only the strict ZS32 profile maps ambiguous evidence to `REVIEW`.
- Stage 18 always requires strict evidence artifacts, rejects an existing final output directory, stages CSV/summary/all audits in one sibling directory, and publishes the complete generation with one rename. Hash, CSV, audit, or rename failure leaves neither final nor staging output. Malformed strict thresholds/evidence publish a non-releasable diagnostic generation before the CLI exits nonzero.
- Audit schema `2.0` retains product/profile/hand/session/timestamp, `inspection_complete`, computed evidence class, normalized risk, source/evidence SHA-256, both margins, all versions and triggers, while keeping `machine_status`, `review_status`, and `released_status` separate.
- Stage 31 reads the same strict profile as stage 18, automatically requires all 72 exact calibration groups, and embeds the profile SHA-256, full expected-version contract, threshold-version set, and canonical threshold-record SHA-256 in `thresholds.json`. Invalid calibration nulls all safety performance rates; raw observed rates are available only as `diagnostic_observed_*`.
- Final-fix commits before integration are `5b38807c` (invalid calibration rates), `c78b24d0` (strict identity/version/source contract), and `a0e4810d` (atomic diagnostic audit generations).
- Fresh final-fix verification before the integration commit: the five directly related files report `178 passed`; legacy traditional-operator and multiview-manifest compatibility adds `11 passed`. Stage-18 and stage-31 help, changed-module `py_compile`, profile JSON validation, formatting, and `git diff --check` are required final gates.

### ZS32 locked threshold and structured evidence closure (2026-07-13, round 2)

- This section supersedes the preceding final-review closure where threshold artifacts, capture session/group identity, gate triggers, or YOLO detections are concerned.
- Strict stage 18 now requires `--threshold-artifact <stage31-output>/thresholds.json`. The bundle must have schema/version `anomalib.zs32_fusion_thresholds/1.0`, `calibration_valid=true`, the exact current profile/config SHA-256, all 72 required groups, 72 unique deployable `status=ok` threshold records, a valid canonical record hash, and a valid artifact hash calculated over the canonical payload without its own hash field.
- Every strict branch CSV `low_threshold/high_threshold` must exactly match the unique locked record selected by `(hand,view,branch,model_version,roi_version)`. Missing, tampered, incomplete, invalid-calibration, profile-mismatched, or numerically inconsistent bundles publish a non-releasable diagnostic generation and then exit nonzero.
- Generation-level `threshold_artifact.json` and every per-part audit retain the artifact path, source-file SHA-256, immutable artifact SHA-256 and threshold-record SHA-256.
- `capture_session` and `group_id` are normalized CSV/audit identity fields. Every strict row requires both values and all rows for one physical part must agree; missing or mixed identities are `INVALID_CAPTURE`. An absent acquisition timestamp remains `null` and is never replaced with the audit write time.
- All quality/registration non-PASS rows are retained as separate triggers. Valid STRONG evidence remains the machine `NG_*`; gate/system evidence forces `inspection_complete=false` and keeps release null.
- YOLO uses one summary branch row per part/view with a JSON `detections` list. Each box preserves class, confidence, `xyxy`, area and supplied ROI/border flags. Multiple boxes never count as duplicate required branch identities; no-box evidence is the explicit empty list `[]`.
- Calibration coverage invalidation uses one helper to null escape, recall, normal-reject, review and confidence-bound safety rates while keeping raw observations only as `diagnostic_observed_*`.
- Round-2 implementation commits before integration are `478adf79` (capture identity, gates and YOLO evidence) and `72a64d79` (locked threshold artifact).
- Fresh round-2 directly related gate after cross-layer integration: `199 passed, 3 warnings`; legacy traditional-operator/multiview-manifest gate is `11 passed, 3 warnings`. A broader `tests/unit/capture_data` diagnostic run reports `288 passed, 2 failed`; the two failures are pre-existing unrelated stale expectations in multicamera right-hand parser policy and demo face-name console wording, not fusion files or behavior. Format, core Ruff `E4/E7/E9/F/I`, production `py_compile`, profile JSON, both CLI help commands, and `git diff --check` pass; the broad project Ruff policy retains its existing 58 whole-file findings.

### ZS32 explicit gate and YOLO evidence closure (2026-07-13, round 3)

- Strict ZS32 quality/registration rows require an explicit nonblank `status=PASS`. Missing, blank, WARN, FAIL, or any other value is a gate fault; without STRONG it blocks OK, and with STRONG it preserves the machine NG while adding the gate trigger and forcing `inspection_complete=false`. Legacy non-strict rows may still use `pred_label=0` without an explicit status.
- `BranchPrediction.detections=None` means the YOLO summary is missing; an explicit empty tuple/list means a present no-box summary. CSV missing/blank cells remain `None`, while JSON `[]` remains an explicit CLEAR summary. Strict ZS32 rejects missing summaries.
- Every strict YOLO box requires class, confidence, `xyxy`, and area. Confidence must be finite in `[0,1]`, coordinates must be four finite nonnegative increasing values, area must be positive and finite, and supplied ROI/border flags must be boolean. Missing or malformed evidence is published only as a fail-closed diagnostic generation.
- Legacy near-threshold policy and `surface_texture status=SUSPECT` both retain `final_status=SUSPECT` with `final_label=None`; only the strict ZS32 profile maps GRAY evidence to REVIEW.
- Fresh round-3 unified gate is `222 passed, 4 warnings`: 211 directly related fusion/audit/calibration/stage-18/wrapper tests plus 11 traditional-operator/multiview-manifest compatibility tests. All 10 files pass formatting and core Ruff `E4/E7/E9/F/I`; production `py_compile`, profile JSON validation, both CLI help commands, and `git diff --check` pass. Broad project Ruff remains at the prior 58 whole-file findings after cleaning three new policy findings from this round.

### ZS32 template-first inspection gate (2026-07-13)

- `template_match` is now the first product-detection branch for every left/right ZS32 view. It uses whole-view fixed-ROI grayscale matching, not the C789 slot prototype.
- Execution order is identity/image prerequisites, six-view template gate, then expensive PatchCore/YOLO/geometry only after six explicit PASS results. REVIEW, NG_TEMPLATE, malformed continuous evidence, or adapter exceptions short-circuit all downstream calls.
- Template risk is `1-similarity`; `risk<T_low` is PASS, `T_low<=risk<T_high` is REVIEW, and `risk>=T_high` is NG_TEMPLATE. Template-fit normal parts are disjoint from threshold-normal parts.
- Template evidence retains the raw similarity/risk, both thresholds, best template path/hash, offset, image hash, and model/threshold/ROI/template versions. Template files are hash-verified at inference.
- Template training locks `model.json` with `model.sha256`, rejects template paths outside the model directory, and exports same-source `calibration_rows.csv` using Stage 31's nearest-rank policy. Stage 18 retains and verifies `evidence_hash` for the selected template.
- A template result cannot PASS without all four versions, an existing best-template path, and a matching template SHA-256. Downstream `OK` is accepted only with `inspection_complete=true`; missing/unknown downstream state becomes REVIEW.
- Strict Stage 18 requires valid 64-hex declared source and evidence hashes for every required branch row and verifies both against the files; an omitted hash cannot bypass the orchestrator's template evidence contract.
- Template short-circuit audits initialize `review.status=PENDING`, keep `review_status=null`, and keep `released_status=null`, matching the full Stage 18 audit lifecycle even when Stage 18 is intentionally skipped.
- Strict profile required branches increased from five to six per view and the exact versioned calibration contract increased from 60 to 72 groups. Stage 18 derives the expected count from the profile instead of hard-coding it.
- `ZS32InspectionOrchestrator` passes all six normalized template results to its downstream runner. It records early-stop/skipped state without inventing missing branch CLEAR evidence.
- Local integration on 2026-07-13 merged `feat/zs32/multimodel-fusion` into `main` after first committing the completed PatchCore ROI workflow and C789 matcher prototype. Backup branch: `backup/main-before-zs32-fusion-20260713`; no GitHub push was performed.
- The merged PatchCore + fusion target suite reports `333 passed, 1 warning`; production/core-import Ruff, Python compilation, shell syntax, profile JSON, and all five relevant CLI help commands pass.

## BMW eight-view bright-streak and EfficientAD Demo thresholds (2026-08-10)

- Branch/worktree: `agent/bmw-eight-view-handoff` in `.worktrees/bmw-eight-view-handoff`; the public Draft PR is `wjstx0425/anomaly_xingtao_new#1`. Never commit `dataset/`, `results/`, customer images, checkpoints, or the local asset symlinks.
- Bright-streak detection in `src/bmw_inspection/detector.py` now follows a row-wise center ridge, allowing gradual local drift and width changes instead of requiring one whole connected component. It keeps the existing fixed ROI, presence, continuity, and evidence contracts.
- Standalone recalibration is `pipeline/bmw_lab_recalibrate_bright_streak.py`. Default mode fits only `split=calibration`; explicit `--bold-continuity` keeps SNR/coverage presence fitting calibration-only but uses calibration plus final-test normals for the three continuity envelopes. It is strictly boolean and records `demo_only=true` and detailed split/leakage flags.
- Real bold artifact: `results/bmw_lab_one_click/bmw_lab_eight_view_v1/bright_streak_ridge_v3_bold/`. Thresholds are SNR `3.289698`, coverage `0.257749`, longest-run `0.220228`, max-gap ratio `0.393148`, max-gap count `12`. Current-data result is 20/20 calibration and 20/20 final rows correct, including 17/17 normal streaks OK and 3/3 no-streak samples `NG_NO_STREAK`; this is demo tuning, not independent validation. There are still no real broken-but-present streak negatives.
- Complete EfficientAD scoring uses 20 normal physical parts and 6 defect physical parts, each with all eight views: 208 rows in `efficientad/score_analysis/efficientad_scores.csv`. The defect set is the union of per-view visible-defect part IDs, completed from the canonical per-view ROI crops.
- Whole-part threshold CLI is `pipeline/bmw_lab_calibrate_efficientad_thresholds.py`. Current thresholds are front `0.680344`, front_left `0.009403`, front_right `0.486333`, front_secondary `0.111191`, back `1.0000000000000002`, back_left `0.457497`, back_right `0.230414`, back_secondary `0.575206`. The `back > 1` threshold intentionally disables a noisy view.
- Current selected-data EfficientAD result is 1/20 normal whole-part false NG (`5%`, `bmw_normal_group051/back_left`) and 6/6 defect whole-parts detected with 17 defect-view hits. The artifact is explicitly `demo_only=true` and `test_used_for_selection=true`; it must be validated on new untouched physical parts before any acceptance claim.
- Demo configuration requires explicit `bright_streak.config` and `efficientad.threshold_artifact` assets and fails closed on missing/invalid files. Runtime EfficientAD status is solely `score >= per-view threshold`; checkpoint `pred_label` is diagnostic only. Heatmaps use a fixed 0-1 scale.
- Fresh final offline smoke `bmw_normal_group072_000001` produced 25 PASS, final OK, 3430.967 ms with RTX 4090 CUDA inference. Screenshot: `artifacts/bmw_eight_view_threshold_demo/group072_bold_efficientad_5pct.png`; this single-sample smoke is not a throughput benchmark.

### ZS32 right-hand offline calibration commissioning (2026-07-14)

- Stage 33 is `pipeline/33_run_zs32_offline_calibration.py`; its core is `capture_data/zs32_offline_calibration.py`. It is explicitly commissioning-only, keeps six PatchCore models plus YOLO resident in one process, supports case-level resume, and recomputes all six template scores without applying the online short-circuit.
- The source contract is `/home/yunjing/anomalib/dataset/zs32_patchcore_roi_yolo/crop_manifest.csv`, but semantic views come from original source basenames rather than legacy `resolved_view`. This repairs 24 known routed-view mismatches among deform groups 009-013. Template score/split input is `/home/yunjing/anomalib/results/zs32_template_gate_right_unified_roi_v1/calibration_rows.csv`.
- The completed GPU run is `/home/yunjing/anomalib/results/zs32_offline_calibration_unified_roi_v1`: 99 physical parts, 594 views, 1782 three-model rows, zero runtime errors. Fit has 42 normal plus 25 defect parts; held-out test has 28 normal plus only 4 defect parts.
- `template_patchcore_threshold_calibration/` is a 12-group diagnostic fit: held-out defect non-clear recall is 4/4, normal strong-reject rate is 5/28, and review rate is 16/32. It is not a production-independent validation because model/data splits were not jointly locked.
- YOLO labels come from `/home/yunjing/anomalib/dataset/zs32_six_view_roi_yolo/labels/{train,val,test}`. Stage 33 excludes train, maps val to calibration and test to test, and uses per-view identities. `yolo_annotation_thresholds_by_view/summary.json` reports that only front/front_right avoid zero thresholds; four views are degenerate because labeled positives have score zero at the candidate floor. Do not inject these thresholds.
- The naive 18-group root threshold fit is explicitly rejected by `threshold_quality_report.json`: it lacks the remaining 18 right-profile quality/registration/geometry groups, uses physical-part labels for its YOLO branch, has only four held-out defect parts, and strongly rejects all 28 held-out normal parts because of zero YOLO thresholds.
- Post-review hardening adds `commissioning_run_contract.json` plus one `stage33_publication.json` per threshold directory. Resume now binds runtime config/checkpoints, template model, all selected source-image SHA-256 values, physical labels/splits, YOLO label-tree hash, fit parameters, aggregate CSV and threshold/metric hashes. It also rejects one physical part split across YOLO train/val/test and validates paired images plus normalized bbox syntax.
- The strengthened per-view acceptance contract requires distinct positive thresholds, at least five fit and five test positives, held-out recall at the requested target, and normal strong-reject rate at most 0.2. Under that contract all six current YOLO views are `operationally_usable=false`; front/front_right are also rejected for low sample counts and 2/3 held-out positive recall.
- The hardened preflight validates YOLO label syntax and physical-part split isolation before GPU work and hashes every paired ROI image. The real resume verification decoded and compared all 594 selected runtime YOLO crops against their labeled dataset ROI pixels; all matched, recorded in `/home/yunjing/anomalib/results/zs32_offline_calibration_unified_roi_v1/yolo_runtime_crop_identity.json`.
- Raw `/home/yunjing/anomalib/dataset/zs32_six_view_yolo` and cropped `/home/yunjing/anomalib/dataset/zs32_six_view_roi_yolo` label stems/splits/box-presence align exactly: train 1398, val 306, test 306, with zero missing, extra, or presence mismatches. Calibration uses the ROI labels because runtime YOLO scores are produced from those ROI pixels.
- Stage 33 now has an explicit complementary YOLO policy via `--yolo-aux-min-image-precision`. It preserves the old recall-first `yolo_annotation_thresholds_by_view/` diagnostic and separately publishes `yolo_high_precision_auxiliary/`; fit uses only val/calibration, test is evaluation-only, zero-score/no-box rows can never become a threshold, and every physical part is revalidated as split-isolated with exactly six canonical views.
- The 2026-07-14 commissioning run at minimum fit image-presence precision 0.90 selected front `0.38068947196006775`, front_left `0.8800162672996521`, front_right `0.8765060305595398`, back `0.025683363899588585`, back_left `0.5230337381362915`, and back_right `0.008334489539265633`. Physical-part six-view OR is val TP/FP/FN/TN `4/0/1/8` and test `5/0/0/13`; test GT-aligned trigger recall is also 5/5, so no positive part was counted solely from a trigger on an unlabeled view.
- Interpret this as report-only direct-NG candidates: a future locked profile would map `low=high=T`, so `score>=T` is YOLO STRONG and may directly produce `NG_YOLO`, while a YOLO miss remains CLEAR for that branch and can be caught by PatchCore/template/another view. The artifact uses image-level bbox-presence precision, not IoU-matched box precision, and is marked `runtime_injection_supported=false`; do not pass it directly to Stage 32/18 until the full 36-group right profile is locked.

### ZS32 unified-ROI 18-group fusion commissioning (2026-07-14)

- The explicit reduced profile is `config/fusion/zs32_right_unified_roi_18_group_commissioning.json`, Stage18/32 alias `zs32-right-18-commissioning`. It requires exactly six right-hand views times `template_match`, the matching `anomaly_<view>` PatchCore branch, and `yolo`; it never replaces or relaxes production `zs32-right` and carries `commissioning_only=true`, `production_release_allowed=false`.
- Stage 34 is `pipeline/34_publish_zs32_18_group_commissioning.py`, backed by `capture_data/zs32_18_group_commissioning.py`. It verifies the live runtime/template version contract, template `model.sha256`, both Stage33 publication sidecars, exact 12+6 group coverage, and atomically signs a Stage18-compatible artifact with profile SHA, canonical records SHA and immutable artifact SHA. It rejects invalid calibration, non-ok records, YOLO candidate incompleteness, test-selection leakage and candidates below their declared fit-precision floor. Output: `/home/yunjing/anomalib/results/zs32_18group_commissioning_v1/thresholds.json`.
- Template thresholds come from the online template `model.json`, not the Stage33 refit; front_left low is exactly `0.011378765106201172`. PatchCore uses the six Stage33 template/PatchCore records. YOLO maps the six auxiliary thresholds to binary `low=high=T`.
- Stage32 now has explicit `--fusion-profile`. Production remains the default and still requires quality/registration/geometry CSVs. Only `zs32-right-18-commissioning` omits them. Stage32 validates the current runtime-config and template `model.json` byte hashes against the artifact before model/GPU loading. Under this complementary commissioning profile, only a complete finite REVIEW genuinely inside the template gray band continues so PatchCore/YOLO may cover it; exception-shaped REVIEW and explicit NG_TEMPLATE short-circuit while retaining non-production policy metadata.
- Real GPU smoke output is `/home/yunjing/anomalib/results/zs32_18group_smoke/group038`: 18 required rows, all CLEAR, Stage18 `OK`, `inspection_complete=true`, no runtime errors. The post-hardening rerun is `/home/yunjing/anomalib/results/zs32_18group_smoke_v2/group038` with the same terminal result. The runtime summary/manifest, `fusion/fusion_policy.json`, and audit all retain `production_release_allowed=false`; the manifest also preserves its pre-fusion REVIEW under `pre_fusion_result`.
- Final regression gate after hardening: 216 relevant fusion/runtime tests passed; the focused publisher/Stage18/Stage32 subset reports 37 passed. Ruff, legacy core E/F/I, formatting, compilation, JSON and diff checks also pass.

### ZS32 live template-score geometry drift diagnosis (2026-07-14)

- The confirmed normal right-hand live capture is `/home/yunjing/anomalib/results/zs32_live_capture/20260714_145508_291649_live_part_001`, manifest session `20260714_145508_515286`. The historical same-domain control is right normal `group038` from session `20260711_165347_078120`; the wrong-left-hand capture remains a useful negative control.
- The live and historical manifests bind the same serials to the same camera-slot views: `DA9805574` -> `front/back`, `DA9625347` -> `front_left/back_left`, and `DB0998274` -> `front_right/back_right`. SDK `device_index` changed between sessions, but collection reorders handles by serial; cross-view and left/right swap scoring became worse, so the anomaly is not a view-routing or flip-naming error.
- The runtime uses the exact configured PatchCore ROI for template crops. All source images are `4024x3036`; live, group038, training crops, and saved templates have matching per-view dimensions. All 30 template PNGs hash-match `model.json`, trace to normal right-hand same-view samples, and equal the declared grayscale/width-512/Gaussian preprocessing output. No template contamination or version/hash mismatch was found.
- Same-code replay on group038 gives all-zero/one-pixel offsets and risks `0.0046-0.0073`. On the live normal right part, `front=0.01217@(1,0)` and `back=0.01279@(0,0)` remain near the normal domain, while `front_left=0.05847@(9,-12)`, `front_right=0.70670@(-12,-12)`, `back_left=0.07591@(9,-12)`, and `back_right=0.69402@(-12,-12)` hit the configured translation boundary.
- SIFT/RANSAC comparison with group038 shows camera-specific, front/back-consistent drift: the centre camera is effectively stable; the left camera is approximately `(+48,-129)` raw pixels; the right camera is approximately `(-848,+39)` raw pixels with about `+5%` scale and `-5.8 deg` rotation. The current template gate searches only translation (`+-12` pixels after resize), with no rotation or scale search.
- Expanding translation only in memory recovers the left views near the threshold range but cannot recover the right views. Warping the live images back toward group038 geometry before applying the same ROI and matcher reduces left risks into PASS (`front_left 0.01016`, `back_left 0.01185`) and reduces right risks by roughly an order of magnitude, confirming geometry drift as causal. The right fixed ROIs also clip content after the roughly 848-pixel framing shift, so translation-only matching cannot reconstruct the missing region.
- Root-cause priority is: side-camera acquisition geometry changed (right camera severe, left camera mild) > fixed-ROI clipping/background participation and insufficient translation-only alignment > smaller HDR photometric shift (`1500/6000 us` live versus `1500/5500 us` historical). Threshold direction, score formula, template aggregation, hashes, crop dimensions, hand identity, and serial/view routing were consistent.
- Do not repair this by raising production template thresholds. First restore/lock the physical camera and fixture geometry against approved full-frame references, recapture one six-view normal sample, and require current `+-12` matching to stop hitting boundaries. If the new camera pose is intentional, create a new measured ROI/reference/model/calibration generation from the new geometry rather than mixing it with the July 11 model.

### ZS32 eight-view YOLO ROI preparation (2026-07-15)

- Stage 29 now selects and converts all eight four-camera views in topology order: `front`, `front_left`, `front_right`, `front_secondary`, `back`, `back_left`, `back_right`, `back_secondary`.
- Real eight-view YOLO data lives under `/home/yunjing/anomalib/dataset`, so Stage 29 exposes `--repo-root`; use `/home/yunjing/anomalib` together with explicit absolute input/config/output paths so manifest-relative paths resolve correctly.
- Secondary mirrored normals reuse their own source-view ROI and horizontally flip it, matching the existing normal-mirror geometry contract. The focused Stage 29 suite reports `11 passed`.

### ZS32 right-hand eight-view PatchCore ROI preparation (2026-07-15)

- Stage 30 accepts `--repo-root`, repeated `--hand`, and repeated `--exclude-session`; the current commissioning path is right-only and uses all eight views including `front_secondary/back_secondary`.
- A right-only conversion accepts the Stage 29 top-level `views` config directly. The current source is `/home/yunjing/anomalib/dataset/zs32_new`, and the ROI config is `/home/yunjing/anomalib/dataset/zs32_eight_view_roi_config.json`.
- Exclude complete duplicate sessions `zs32_right_deform_20260714_204302_593214015` and `zs32_right_deform_20260714_201404_438451409`; never delete or rewrite the raw source tree.
- Expected clean output is 1048 images: each of eight right-hand views has 108 normal and 23 defect images. Focused Stage 30 tests report `53 passed`.

### ZS32 right-hand eight-view PatchCore training (2026-07-15)

- `pipeline/run_patchcore_roi_eight_views.sh` is the eight-view strong-parameter entrypoint. It reuses the serial/resumable runner while selecting `right_front`, `right_front_left`, `right_front_right`, `right_front_secondary`, `right_back`, `right_back_left`, `right_back_right`, and `right_back_secondary`.
- The eight-view run writes `eight_view_summary.csv`; the legacy six-view entrypoint and `six_view_summary.csv` remain compatible and unchanged by default.
- Both `pipeline/8_train_custom_models.py` and `examples/api/03_models/zs32_defect_workflow.py` accept the two secondary views. Focused runner/workflow verification reports `26 passed`.

### ZS32 right-hand eight-view 24-group fusion commissioning (2026-07-15)

- `config/fusion/zs32_right_eight_view_24_group_commissioning.json` is the exact right-only `8×3=24` profile for `template_match`, `anomaly_<view>`, and `yolo`; alias: `zs32-right-24-commissioning`.
- The Stage 34 publisher is profile-driven and supports both the legacy 18-group profile and the new 24-group profile. Both remain `commissioning_only=true` and `production_release_allowed=false`.
- `capture_data/inspection_audit.py` preserves the legacy six-view audit shape for six-view inputs and adds the secondary keys only when those views are present. Focused publisher, Stage 18, and audit verification reports `25 passed`.

### ZS32 Stage 32 eight-view runtime (2026-07-15)

- `capture_data/zs32_inspection_orchestrator.py` and `capture_data/zs32_model_runtime.py` use the exact eight-view order `front`, `front_left`, `front_right`, `front_secondary`, `back`, `back_left`, `back_right`, `back_secondary`.
- Runtime config loading requests `load_patchcore_roi_config(..., hands=("right",))`, so the right-only Stage 29 top-level `views` ROI config is valid without fabricating a left-hand section.
- PatchCore loads and runs eight distinct checkpoints; YOLO receives one ordered eight-image batch. Both secondary views are included in crops, evidence rows, manifests, and request validation.
- Focused orchestrator/runtime/Stage 32 tests report `75 passed` before the separate fusion-profile assertions are included.

### ZS32 eight-view trained assets and calibration result (2026-07-15)

- The real eight-view template model is `results/zs32_template_gate_right_eight_view_v1/model.json`: eight groups, 40 template PNGs, 720 calibration rows, and a verified `model.sha256` of `bc5fa70965e756c30030a9e523df927c396f99629cfd3cbe4e6e4bae0a86ebfd`.
- The pinned runtime bundle is `config/fusion/zs32_runtime_models_eight_view.json`. It binds all eight PatchCore checkpoints from `eight_view_summary.csv`, YOLO `results/yolo/zs32_eight_view_roi_n640_seed42/weights/best.pt`, and the right-only eight-view ROI config.
- Real GPU Stage 32 smoke output is `results/zs32_eight_view_runtime_smoke/group064_v2`: eight template PASS rows, eight PatchCore rows, eight YOLO rows, all expected crops/overlays/maps, and no runtime errors. It intentionally remains REVIEW until a valid locked threshold artifact exists.
- Real Stage 33 output is `results/zs32_offline_calibration_eight_view_v1`: 90 complete cases, 720 pixel-identical runtime YOLO crop checks, 2160 calibration rows, and 16 valid template/PatchCore thresholds.
- The 24-group profile is implemented, but Stage 34 must currently reject publication because YOLO lacks two valid calibration candidates. `front` has only 2 calibration positives and its best threshold has precision `2/3`, below `0.90`; `front_secondary` has zero calibration positives and its only positive is held-out test data. Do not hand-fill, move test data after inspection, or disable these YOLO branches to pretend the requested 24-group artifact is calibrated.
- Final focused eight-view template/runtime/offline-calibration/fusion regression reports `133 passed`.

### ZS32 temporary test-leaked 24-group smoke (2026-07-15)

- The user explicitly authorized a temporary fake 24-group commissioning result before collecting more labels. Stage 33 therefore has opt-in `--yolo-aux-use-test-for-selection` plus `--reuse-existing-inference`; Stage 34 requires the separate `--allow-test-leakage` opt-in. Default strict behavior remains unchanged.
- The leaked Stage 33 source is `results/zs32_offline_calibration_eight_view_v1/yolo_high_precision_auxiliary_test_leakage/thresholds.json`. It reuses all 90 completed cases and marks `test_used_for_selection=true`, `data_leakage=true`, `fit_split=calibration+test`, and `evaluation_split=test_reused_for_selection`. All eight YOLO records are `ok`; leaked `front=0.7289669513702393` and `front_secondary=0.8694137334823608`.
- The fake 24-group artifact is `results/zs32_24group_commissioning_test_leakage_v1/thresholds.json`: 24 exact records, `commissioning_only=true`, `production_release_allowed=false`, and the same explicit leakage markers. Never use it as production validation.
- The real GPU fused smoke is `results/zs32_eight_view_24group_test_leakage_smoke/group064`: 24 branch rows across eight views and three branches per view, `inspection_complete=true`, final `NG_YOLO`. The known-normal group was rejected by back YOLO (`score=0.00245661 >= leaked threshold=0.00126685`), demonstrating the expected false-positive risk of the fake thresholds rather than a runtime failure.
- Focused leakage/Stage33/Stage34/Stage32/Stage18 verification after the run reports `64 passed`; compilation and `git diff --check` pass.

### ZS32 strict eight-view commissioning decision (2026-07-15)

- User approved `docs/superpowers/specs/2026-07-15-zs32-strict-eight-view-commissioning-design.md`: all eight views are modeled; `MODELED_VIEWS == VIEW_ORDER`; secondary may not be published as unsupported or with empty branches.
- The accepted target is commissioning completeness, not production release. Every view must have Template, PatchCore, YOLO, and Fusion dashboard records, while Stage18 consumes exact `8×3=24` commissioning evidence groups.
- Model paths must live in a versioned runtime bundle accepted through `--runtime-config`. Replacing Template, any per-view PatchCore, or the shared YOLO must publish a new hash-bound bundle and matching 24-group threshold artifact without Python edits.
- Existing eight PatchCore checkpoints, eight-view ROI, and shared eight-view YOLO are real assets. A same-ROI eight-view Template and unified locked bundle still must be published; the current `front_secondary` PatchCore is commissioning-only because its sample F1 is about `0.286`.
- The approved execution plan is `docs/superpowers/plans/2026-07-15-zs32-strict-eight-view-commissioning-plan.md`. It supersedes only Tasks 6-9 of the 2026-07-14 dashboard plan and preserves the accepted offline Tasks 1-5.
- The bundle implementation is two-phase to avoid circular hashes: publish immutable inner runtime assets plus a generated 24-group profile, publish Stage33/34 thresholds bound to those assets and the Template, then finalize the outer Stage35 bundle.

### ZS32 strict eight-view Task 1 implementation boundary (2026-07-15)

- Commit `a6980413` freezes the right-only eight-view Stage32 contract: all eight Template gates are evaluated before aggregate stop, stopped views retain real Template status, and PatchCore/YOLO/Fusion are explicit `SKIPPED` with null scores.
- Stage32 exposes only `zs32-right-24-commissioning`; it always publishes `commissioning_only=true` and `production_release_allowed=false`. The legacy six-view production contract stays outside this path.
- Runtime bundles require exact `product=ZS32`, `supported_hands=[right]`, eight canonical PatchCore views, and fail closed on non-positive YOLO boxes. Dashboard parsing rejects `UNSUPPORTED` in every core branch.
- The clean commit snapshot passes 139 focused tests; the dirty checkout with later user tests passes 206. User-staged 18-group Stage18/test work was deliberately preserved outside the Task 1 commit.

### ZS32 strict runtime manifest and ROI compatibility correction (2026-07-15)

- Follow-up commit: `e283fec6` (`fix: publish strict ZS32 runtime manifests`); its clean archive passes 219 focused plus ROI tests.
- Model-only runtime output is itself a strict dashboard generation: eight copied/distinct source images with real SHA-256 and shapes, exact per-view identity, and Template/PatchCore/YOLO/Fusion branches. Unexecuted branches use `SKIPPED` plus null score.
- Stage32 ordinary infer merges copied Template evidence into those records; successful fuse replaces every Fusion branch with the real local `fusion/fused_predictions.csv`, still without inventing a per-view score.
- `zs32_patchcore_roi_dataset.VIEWS` and `zs32_view_roi_dataset.VIEWS` remain historical six-view defaults. Strict Stage32 passes `expected_views=CANONICAL_VIEWS` explicitly to both loaders.
- The eight-view runtime config must declare `commissioning_only=true` and `production_release_allowed=false`; runtime loading rejects missing or changed flags, and all manifest/summary modes expose them.
- YOLO ROI JSON image sizes and coordinates accept only positive/non-bool integers; bool, float, and string values fail closed rather than being converted.
- Template-stop scores are JSON-safe: only finite non-bool int/float values survive as floats; NaN, infinity, bool, and strings become null, with `allow_nan=false` serialization.
- Successful Stage18 status is conditional on a real `fusion/fused_predictions.csv` inside the generation. Missing, non-file, or escaped evidence forces REVIEW/incomplete, adds `fusion_evidence` to missing evidence, and marks all eight Fusion branches ERROR without paths.
- Final fail-closed follow-up commit is `db1e23e6`; its clean archive passes 226 complete Task 1 focused plus ROI tests.

### ZS32 eight-view YOLO label reconciliation (2026-07-15)

- Source data is read-only at `/home/yunjing/anomalib/dataset/zs32_new`; the fixed eight-view ROI is read-only at `/home/yunjing/anomalib/dataset/zs32_eight_view_roi_config.json`.
- Historical annotation evidence comes from `/home/yunjing/anomalib/dataset/zs32_new_yolo_labeling` and the existing cropped dataset `/home/yunjing/anomalib/dataset/zs32_eight_view_roi_yolo`; never infer that an unlabeled defect image is negative merely because the generated YOLO txt is empty.
- Reconciliation must use all eight canonical views, content hashes before filenames, group/session-isolated splits, explicit conflict/review flags, and a new independent output under this checkout. Stage 27 local-file conventions and Stage 29 ROI/label transforms are the compatibility anchors.
- Stage 36 is `pipeline/36_reconcile_zs32_yolo_labels.py`, backed by `capture_data/reconcile_zs32_yolo_labels.py`. `prepare` uses encoded SHA256, decoded-pixel SHA256, a recorded horizontal-flip pixel digest, ROI dHash candidates, Stage 29 label migration, alias-bound deterministic splits, Label Studio predictions, and strict YOLO validation; `finalize` consumes a completed Label Studio JSON task export and creates a separate final dataset. The current raw set produced no horizontal-flip auto-reuse rows.
- Prepared output is `dataset/zs32_eight_view_yolo_reconciled_20260715_v3`: 2104 source images -> 1664 canonical images, 440 duplicate images across 55 renamed group aliases, 248 historical positive labels reused, 864 normal-directory empty labels confirmed, 526 missing labels, 26 exact-duplicate label conflicts, and 552 Label Studio review tasks. Perceptual candidates are audit-only and never auto-reused; the current data has zero proven horizontal-flip reuse rows. The earlier unsuffixed target existed as an empty directory, so it was not overwritten.
- The draft is deliberately blocked from training: it has `yolo_draft/DO_NOT_TRAIN_UNTIL_LABEL_STUDIO_REVIEW_COMPLETE.txt` and intentionally has no `data.yaml`. Import `label_studio/tasks_all.json` to see every reused prediction and filter on `needs_human_annotation`; use `tasks_review.json` only for the 552 unresolved/conflicting tasks. After review, export Label Studio JSON and run Stage 36 `finalize` to produce the only standard train/val/test dataset and a refreshed final mapping.
- The operator later narrowed labeling scope to all 840 raw defect images only. Use `pipeline/36_reconcile_zs32_yolo_labels.py prepare-defect-840` and `dataset/zs32_eight_view_yolo_reconciled_20260715/defect_label_studio_840/tasks_defect_840.json`; it contains exactly 105 tasks per view, no normal images, 253 safe preannotations, 52 conflict-review tasks across 26 exact-duplicate clusters, and 535 missing-label tasks. The 40 duplicate aliases remain independent tasks while retaining alias-bound splits.
- On 2026-07-15 the operator explicitly confirmed that all 439 Label Studio tasks marked skipped/cancelled had been manually inspected and contained no visible defect. Stage 36 now exposes `finalize-defect-840-yolo`, which requires `--confirm-skipped-empty`, converts those omissions to audited empty YOLO txt files, rejects orphan/invalid labels and exact-pixel leakage, and uses a fixed whole-session split that binds the 40 cross-session duplicate clusters. The generated independent dataset is `dataset/zs32_defect_840_yolo_final_20260715`: 840 images/labels, 400 positive images, 440 empty images, 448 boxes; train 616/275 positive, val 88/51 positive, test 136/74 positive. The source Label Studio YOLO export remains unchanged.

### ZS32 strict eight-view Stage33 real commissioning attempt (2026-07-15)

- Immutable runtime assets are `results/zs32_runtime_assets_eight_view_v1`; asset-set hash `ea1756c26b45126d86c9d2396b19985efd7073dc6c7e8d13a0744c08ab4b7647` binds the eight PatchCore checkpoints, shared YOLO, exact-eight Template, ROI, and commissioning profile.
- Real Stage33 inference completed all 90 cases under `results/zs32_stage33_eight_view_commissioning_v4`; its run contract binds runtime SHA `27d9fcf9ce69b5037ab917694d27936be856c83842c45877680c7e308e1abd86`, Template SHA `bc5fa70965e756c30030a9e523df927c396f99629cfd3cbe4e6e4bae0a86ebfd`, 90 source records, and the YOLO dataset hash. The artifact is commissioning-only.
- Ultralytics can emit a finite four-coordinate box collapsed to zero area after boundary clipping. Such candidates are now warning diagnostics and are excluded from detection/score; malformed/nonfinite boxes still fail closed. The observed real candidate was `right:normal:group019/back_right`, confidence about `0.00445`, box `[0,2429,2763,2429]`.
- `threshold_calibration/thresholds.json` has 24 finite part-level thresholds and `template_patchcore_threshold_calibration/thresholds.json` has 16 finite thresholds, but strict Stage33 publication remains blocked: `front_secondary` YOLO annotation calibration has 13 rows all labeled `0` and no positive calibration row. The single positive is in test and must not be used for selection. Complete Label Studio review/finalization before rerunning to a new immutable output directory.
- For the explicitly approved quick commissioning fallback, v4 also contains `yolo_high_precision_auxiliary_test_leakage/thresholds.json`: exact eight views, finite thresholds, `test_used_for_selection=true`, `data_leakage=true`, and a sidecar bound to the v4 run contract. Stage34 may consume it only with `--allow-test-leakage`; it is never production evidence. Threshold SHA is `7f2c45b4bcdb730e1579d6c3c9d405e0b6f8921b3cce0c64b514a71172812edb`.

### ZS32 Stage33 review hardening (2026-07-15)

- Stage33 reuse validates every case against the active runtime, ROI, model, source, and evidence contract; preflight performs the same available read-only checks before GPU initialization.
- Stage31's four reports and Stage33 sidecar are one atomic no-replace generation. Task2 is consumed through its public manifest loader and binds the asset-set SHA, ROI bytes, and Template bytes; regenerate stale immutable generations instead of editing them.

## ZS32 Task 8 minimal eight-view commissioning acceptance (2026-07-15)

- Reused real same-identity Stage33 v4 case `results/zs32_stage33_eight_view_commissioning_v4/cases/right_normal_group064-b1959d15`; no GPU model rerun. Strict dashboard parsing returns canonical eight-view order, all `model_supported=true`, and exact Template/PatchCore/YOLO/Fusion branches. The case stays `REVIEW` because Fusion is `SKIPPED`.
- Headless Stage36 screenshot: `artifacts/zs32_dashboard_eight_view_commissioning/group064_stage33_v4_dashboard_1600x920.png` (1600x920, SHA-256 `2356905b70ac0859864764f600412fdb536890f6455ac421b642567b2f615f43`). No HighGUI mode was used.
- Fresh v2 bundle/topology preflight passed for `results/zs32_runtime_bundle_eight_view_v2/runtime_bundle.json`; bundle ID `zs32-right-eight-view-commissioning-v1`, asset-set SHA `dda14d6b3379a186bf791ad5873bee8019cb0e407ba89d8eaae75784e9fcdcfa`.
- Stage35 device-list preflight found all four topology serials: `DA9805574`, `DA9625347`, `DB0998274`, `DB0968108`. No operator capture was run; hardware capture remains incomplete.
- Acceptance remains commissioning-only/non-production. v2 thresholds contain authorized Stage33 v4 test leakage; existing Stage18 group064 replay is known `NG_YOLO` from back; finite zero-area post-clip YOLO candidates are warning/excluded while malformed or nonfinite candidates fail closed.

### ZS32 positive-defect + trusted PatchCore-normal YOLO rebuild (2026-07-15)

- Use `dataset/zs32_defect_positive_patchcore_normal_yolo_20260715_v2` for the minimal clean retraining route. The unsuffixed directory is an incomplete failed build and must not be trained.
- The v2 dataset contains 779 images: 379 exact-pixel-unique Label Studio positive defect images with 425 boxes, plus 400 empty-label normal ROI images selected as 50 complete eight-view groups from `/home/yunjing/anomalib/dataset/zs32_eight_view_patchcore_roi/right`.
- All 440 defect tasks without a non-empty exported YOLO label are excluded rather than treated as negatives. The audit is `excluded_uncertain_defects.csv`. Another 21 exact-pixel positive aliases are excluded and recorded in `deduplicated_positive_aliases.csv`.
- The split is physical-group bound with exact-pixel alias binding and seed 42: train 539 (267 positive/272 normal), val 113 (49/64), test 127 (63/64). Every split contains both hands, all three defect types, all eight positive views, and all eight normal views; no sample group or exact pixel SHA crosses splits.
- Whole-session isolation is intentionally relaxed because the available `less` and `others` positive sessions cannot populate train/val/test simultaneously. Do not describe this dataset as session-isolated; it is group-isolated.
- Stage 36 `build-balanced-defect-normal` now accepts the PatchCore `crop_manifest.csv` directly and `--max-normal-groups`. It resolves the existing ROI paths, computes decoded-pixel SHA256, creates empty txt files, and refuses to overwrite an existing output.

### ZS32 Stage 37 small-defect tiled YOLO workflow (2026-07-15)

- Stage 37 entrypoint is `pipeline/37_tile_zs32_yolo_dataset.py`; its implementation is `capture_data/tile_zs32_yolo_dataset.py`. Focused regression coverage is in `tests/unit/capture_data/test_tile_zs32_yolo_dataset.py`.
- The generated independent dataset is `dataset/zs32_defect_positive_patchcore_normal_tile1280_s960_20260715`: 817 tiles and 425 boxes. Train has 559 tiles (287 positive, 272 trusted-normal empty, 292 boxes), val has 125 (61 positive, 64 empty, 64 boxes), and test has 133 (69 positive, 64 empty, 69 boxes).
- The fixed geometry is `1280 x 1280`, stride `960`, overlap `320`, and padding value `114`. Every source box is assigned once to the center-containing tile with maximum visibility; the measured minimum retained visibility is `0.954962`.
- Tiles inherit the v2 grouped split. Validation found no cross-split leakage across 139 physical groups or all 779 exact pixel SHA identities; the 440 uncertain defect images remain excluded.
- Real tiled inference smoke output is `results/yolo/zs32_stage37_tiled_infer_smoke_20260715`: one real ROI image produced 12 tiles, 27 raw tile detections, and 14 source-coordinate detections after global merge.

### ZS32 strict 22-group live commissioning v3 (2026-07-15)

- New immutable commissioning bundle: `results/zs32_runtime_bundle_eight_view_22group_v3/runtime_bundle.json`; do not overwrite or reinterpret the existing eight-view v2 bundle.
- The display/runtime contract remains exact eight-view. `front_secondary` and `back_secondary` Template are fixed `SKIPPED` with null numeric evidence, while PatchCore and YOLO remain required for both views. Stage18 therefore requires exactly 22 groups: six Template, eight PatchCore, and eight YOLO.
- Six primary Template highs are temporary fixture-commissioning values: front `.034`, front_left `.037`, front_right `.036`, back `.036`, back_left `.035`, back_right `.036`; all lows are unchanged. The independent model is `results/zs32_template_gate_right_eight_view_22group_v3`.
- Stage35 now derives the 22/24 alias from the bundle, validates Template stop status by aggregate priority instead of the obsolete final-row rule, and accepts SKIPPED only for the two secondary views. Stage18 treats bundle-generated ZS32 commissioning configs as strict and validates the bound threshold artifact.
- CPU Template smoke on fixed-fixture generation `20260715_193411_917807_live_part_001` produced REVIEW on all six primary views and no NG_TEMPLATE; secondary Template is covered by the SKIPPED contract. The v3 bundle remains `commissioning_only=true`, `production_release_allowed=false`, and retains the authorized YOLO test-leakage warning.
- Before freezing production thresholds, collect 10-20 known-normal fixture groups and evaluate the six primary score distributions. Do not promote the temporary highs or current UI result as production acceptance.

### ZS32 Stage 30 eight-view ROI crop compatibility fix (2026-07-15)

- Stage 30 must pass `CANONICAL_VIEWS` explicitly through ROI loading, source discovery, selection, conversion, and summary generation. The lower-level `VIEWS` default remains the historical six-view tuple for compatibility with legacy callers.
- The right-hand source root is `dataset/zs32_all` (the command adds `/right` itself). The reused ROI is `/home/yunjing/anomalib/dataset/zs32_eight_view_roi_config.json`, image size `4024x3036`, SHA-256 `9412b2838cdb96f722db356714cf9bbbb5ea01810671ccf92b67323de77ebe65`.
- Real read-only preflight after the fix finds all eight canonical views and 2456 images: 334 each for the six primary views and 226 each for `front_secondary` and `back_secondary`. The focused ROI dataset suite passes 55 tests.

### ZS32 multi-session PatchCore preprocessing identity fix (2026-07-16)

- Nested `.../<session>/images/*_groupNNN_*` PatchCore sources must use `sample_id=<session>__groupNNN`; the same value is the normal split key. Flat legacy inputs and nested non-group inputs keep their historical identities.
- This prevents different capture sessions that reuse `group001` and the same basename from overwriting one another. Preprocessing also keeps an in-memory `processed_path -> source_path` registry and fails before a second source can overwrite the first; the final manifest asserts `processed_path` uniqueness.
- The old `results/zs32_patchcore_eight_view_all_seed42` checkpoints were trained after 55 collisions per view and must not be reused: primary views had 334 manifest rows but only 279 files, secondary views 226 rows but only 171 files. Use a new run root and retrain.
- Real identity preflight on `dataset/zs32_all_right_patchcore_roi` now produces 2456 unique targets from 2456 sources: 334 per primary view and 226 per secondary view. The workflow and runner regression set passes 29 tests.
- Separate known limitation: because each view independently samples 20% of its own normal-key population, the current primary and secondary populations produce 67 shared session/group keys with mixed train/normal_test assignments. This does not reintroduce path collisions, but strict fusion-level group isolation requires a separate stable cross-population split change.

### ZS32 two-session normal-only Template model v7 (2026-07-15)

- The useful Template source is restricted to the right-hand ROI data from sessions `20260715_204149_641124` and `20260715_214625_447032`. The filtered manifest is `dataset/zs32_all_right_patchcore_roi/crop_manifest_normal_20260715_two_sessions.csv`: 968 rows, all `normal`, with 148 images for each primary view and 40 for each secondary view.
- `pipeline/train_zs32_template_gate.py --normal-only` now explicitly ignores defect rows and calibrates a binary PASS/NG gate using only held-out normal risks. It publishes `low_threshold == high_threshold == nextafter(normal_quantile cutoff)`, so the demo has no REVIEW interval while preserving the old normal+defect behavior unless the flag is used.
- The trained model is `results/zs32_template_gate_right_20260715_two_sessions_normal_only_v7/model.json`, using `normal_quantile=1.0` and `evaluation_fraction=0`. Every group records `calibration_defect_count=0`.
- Latest fixed-fixture smoke input `20260715_231954_656011_live_part_001` passes all six primary Template views. The immutable 22-group commissioning bundle is `results/zs32_runtime_bundle_eight_view_20260715_template_normal_only_v7/runtime_bundle.json`; secondary Template remains `SKIPPED`, while PatchCore and YOLO remain required.
- This bundle is demo/commissioning only: `production_release_allowed=false`. Its PatchCore and YOLO thresholds retain the explicitly acknowledged demo model-rebind/test-leakage warnings.

### ZS32 20-group YOLO 0.07 and secondary PatchCore skip v8 (2026-07-16)

- The immutable demo bundle is `results/zs32_runtime_bundle_eight_view_20group_yolo007_secondary_skip_v8/runtime_bundle.json`; it does not replace v7. Its exact contract is 20 groups: six primary Template, six primary PatchCore, and eight-view YOLO.
- `front_secondary` and `back_secondary` now require only `yolo`. Their Template and PatchCore branches are both published as `SKIPPED`; the model runtime does not crop, load, execute, or emit PatchCore CSV rows for those two views. YOLO still executes all eight views.
- Every YOLO record uses `low_threshold=high_threshold=0.07`, so score `<0.07` is CLEAR/OK evidence and score `>=0.07` is STRONG/NG_YOLO. Runtime `candidate_conf` remains `0.001` so low-score candidates are retained for audit rather than discarded before thresholding.
- Dashboard display contract: the standalone YOLO evidence page retains every `candidate_conf` candidate for diagnostics, while the Fusion page draws only boxes whose individual `confidence >=` that view's final YOLO threshold. If the branch score is below the threshold, Fusion draws no YOLO box, preventing sub-threshold candidates from being presented as defects.
- The reusable profile is `config/fusion/zs32_right_eight_view_20_group_commissioning.json`. Stage18, Stage32, Stage35, runtime bundle validation, audit formatting, and dashboard manifests accept the exact 20-group identity; unsupported group counts remain fail-closed.
- Stage34 exposes the explicit demo-only `--yolo-threshold-override` option. The v8 threshold artifact is `results/zs32_20group_yolo007_secondary_skip_commissioning_v8/thresholds.json` and records `commissioning_source=manual_demo_override` for YOLO.
- Focused regression across runtime, bundle, Stage18/32/35, orchestrator, and dashboard passed 389 tests. The bundle remains `commissioning_only=true`, `production_release_allowed=false`, with the existing YOLO test-leakage and model-rebind warnings.

### Stage18 PatchCore decision writeback for dashboard (2026-07-16)

- After successful Stage18 fusion, `pipeline/32_run_zs32_multimodel_inference.py` now reads the per-part audit and atomically rewrites each executed PatchCore dashboard branch status from `computed_evidence_level`: `CLEAR -> CLEAR`, `GRAY -> REVIEW`, and `STRONG -> NG_ANOMALY`.
- PatchCore `state=available` continues to mean its heatmap/mask evidence exists; the user-facing `status` now carries the actual decision. Existing score and evidence/mask/ROI paths are preserved. The two secondary PatchCore branches in the 20-group profile remain `state=skipped,status=SKIPPED` and are never overwritten from the global result.
- The writeback validates all active views, exact audit score equality, finite ordered thresholds, and score/level boundary consistency before modifying any branch, preventing partial status updates.
- Read-only verification against real generation `20260716_082357_361244_live_part_001` maps `back_right` score `0.499255895614624` over high `0.4606882333755493` to `NG_ANOMALY/STRONG`; the other primary views map to `CLEAR`, and both secondary views stay `SKIPPED`.
- Focused Stage32/runtime/dashboard regression passed 161 tests. Ruff was not available in the current uv environment; `py_compile` and `git diff --check` are the fallback source checks.

### ZS32 per-view PatchCore threshold tuning v9 (2026-07-16)

- The immutable tuned bundle is `results/zs32_runtime_bundle_eight_view_20group_patchcore_tuned_v9/runtime_bundle.json`; v8 is unchanged.
- `front_right` PatchCore is now `low=0.5`, `high=0.6661418676376343` (high preserved). `back_right` PatchCore is now `low=0.5`, `high=0.55`. All other Template/PatchCore records are unchanged, all eight YOLO records remain `low=high=0.07`, and both secondary PatchCore branches remain skipped by the 20-group profile.
- Stage34 now supports repeatable demo-only `--patchcore-threshold-override VIEW:LOW:HIGH`, records `commissioning_source=manual_demo_override`, validates required PatchCore views and finite ordered bounds, and includes the canonical override map in the signed artifact.
- The v9 threshold artifact is `results/zs32_20group_patchcore_tuned_commissioning_v9/thresholds.json`. Bundle loading, exact record comparison, `py_compile`, `git diff --check`, and 76 Stage34/bundle-related tests passed.

### ZS32 front PatchCore threshold tuning v10 (2026-07-16)

- The immutable bundle is `results/zs32_runtime_bundle_eight_view_20group_patchcore_tuned_v10/runtime_bundle.json`; v9 remains unchanged.
- Relative to v9, only two PatchCore records changed: `front` from `0.4493589401245117/0.5248668193817139` to `low=0.5/high=0.55`, and `front_left` from `0.3332592248916626/0.492424875497818` to `low=0.5/high=0.55`.
- Prior overrides are retained: `front_right=0.5/0.6661418676376343` and `back_right=0.5/0.55`. `back` and `back_left` remain calibrated, both secondary PatchCore branches remain skipped, and all eight YOLO records remain `0.07/0.07`.

### ZS32 EfficientAD-S/M six-primary-view training runner (2026-07-21)

- `pipeline/run_efficientad_s_m_six_views.sh` serially trains 12 independent models: EfficientAD small then medium for `right_front`, `right_front_left`, `right_front_right`, `right_back`, `right_back_left`, and `right_back_right`. It never includes either secondary view.
- The runner uses `uv run --no-sync python pipeline/8_train_custom_models.py`, 256-square input, batch size 1, 100 epochs, FP32 workflow defaults, seed 42, GPU 0 by default, and the verified local ImageNette tree. Its default roots are new `results/zs32_efficientad_{s,m}_six_view_seed42_v1` directories.
- `--efficientad-model-size small|medium` now passes from Stage 8 into `zs32_defect_workflow.py`; the default stays `medium` for backward compatibility. Completed summary+checkpoint pairs are skipped, checkpoint-only runs evaluate, and manifest-only runs reuse preprocessing.
- Each family writes `six_view_summary.csv` with checkpoint path and SHA256. This runner is a training/evaluation candidate workflow only; it does not alter the v10 runtime bundle or integrate EfficientAD into Stage32.

### ZS32 0723 eight-view retraining preparation v2 (2026-07-26)

- Use `dataset/zs32_all_plus_0723_retraining_release_v2`; never train from `release_v1`, which predates strict Template
  role separation.
- The source session has 106 user-confirmed distinct parts and 848 right-normal images. Seed 42 partitions physical parts
  as train/model_val/calibration/final_test = 64/16/13/13.
- PatchCore primary views use legacy + 0723, while both secondary views use only 0723. Template uses only 0723 for all
  eight views. YOLO preserves legacy double-hand bbox labels and adds 0723 normal images with strict empty labels.
- Template training now natively supports the four explicit roles: train selects templates, calibration determines
  thresholds, and model_val/final_test are scoring-only. Legacy calibration/test manifests retain old behavior.
- Commands and evidence are in `docs/ZS32_0723_RETRAINING_PREPARATION_20260726.md`. Current environment has no usable
  NVIDIA device, so no formal Template/PatchCore/YOLO training was started.

### ZS32 0723 normal-only secondary PatchCore resume fix (2026-07-26)

- The first v11 run completed all six primary views but both secondary views failed before training because their
  preprocessed data correctly had no `defect/` directory while `_build_datamodule` always passed
  `abnormal_dir="defect"`.
- `examples/api/03_models/zs32_defect_workflow.py` now passes `abnormal_dir=None` when a view has no real defect
  directory. This keeps secondary detection enabled without fabricating defect samples.
- Rerun the same eight-view runner command against
  `results/zs32_patchcore_eight_view_all_plus_0723_seed42_v11`; six completed summaries are skipped and both secondary
  views reuse their preprocessing manifests.

### ZS32 0723 v11 24-group platform bundle (2026-07-26)

- The default commissioning bundle is now
  `results/zs32_runtime_bundle_eight_view_v11/runtime_bundle.json`; it binds the v11 0723 Template, eight PatchCore
  checkpoints, n1280 YOLO weights, and release-v2 ROI into exactly 24 required groups.
- Both `front_secondary` and `back_secondary` require Template, PatchCore, and YOLO; neither secondary branch may be
  published or displayed as skipped. The v10 20-group bundle remains available as an explicit rollback path.
- Stage34 supports an explicit ROI-version rebind only with `--allow-roi-version-rebind --source-roi-config PATH`.
  It verifies the signed Stage33 run contract, its source runtime path/SHA and ROI versions, both source-runtime ROI
  files, and both new-runtime ROI files before rebind; all bindings are recorded in the signed threshold artifact.
- v11 commissioning uses online Template normal-only thresholds, YOLO `0.07/0.07`, and each PatchCore summary deploy
  threshold as equal low/high. It remains non-production and records YOLO test leakage plus model rebind warnings.
- Template threshold records use the v11 model calibration counts. Model-rebound PatchCore/YOLO records never inherit
  old Stage33 counts; they use `0/0` with `count_provenance=unavailable_for_rebound_model`.
- There is no GT mask: pixel AUROC/AUPRO/F1/IoU are N/A. Secondary PatchCore has normal-only evidence, and current YOLO
  validation recall is low, so hardware smoke acceptance must emphasize missed defects.

### ZS32 0727 Template v12 candidate bundle (2026-07-27)

- `dataset/zs32_0727_template_release_v1` contains only the 19 user-confirmed distinct normal parts: 152 unique
  eight-view images, seed 42, and fixed train/model_val/calibration/final_test roles 11/3/3/2.
- The model is `results/zs32_template_gate_right_0727_eight_view_v12`; model SHA256 is
  `5b071bcfb8a603f5bbf20ecf284f590e88e3e65e114f434fbde1eaf81692ff6e`. It has eight groups and 40 templates.
- Full scoring is in `evaluation.csv/json`: train 88 PASS, model_val 24 PASS, calibration 24 PASS, final_test
  12 PASS/4 NG_TEMPLATE. The false rejects are group001 back/back_left/back_right and group004 back. Do not tune from
  final_test without explicit user approval.
- The candidate bundle is
  `results/zs32_runtime_bundle_eight_view_template_0727_v12/runtime_bundle.json`, canonical SHA
  `1b3341ca05ac7e786733690534c8a75ee6c5a493927145cc39860c4448c29133`. It is strict 24-group and preserves v11
  PatchCore, YOLO, and ROI bytes; both secondary views remain required.
- The user reviewed the final-test overview and current/template/difference visualization, accepted the measured
  4/16 final-test normal false rejects, and authorized the default switch. Stage35 and Dashboard now default to v12;
  pass the v11 runtime bundle explicitly for rollback. Commands and limits are in
  `docs/ZS32_0727_TEMPLATE_V12_REPLACEMENT_20260727.md`.

### ZS32 0727 Template v13 manual-threshold release (2026-07-27)

- v13 isolates the three confirmed Template threshold changes from the restored immutable v12 model:
  `front=0.024885842800140383`, `front_left=0.029833445549011234`, and
  `front_secondary=0.216357362270355228` (low=high). Its model SHA256 is
  `eac211cab4ee1d27adda9933c3cb9c6fde71c4a7eb50fae138201a600fa17ab5`.
- The finalized strict 24-group bundle is
  `results/zs32_runtime_bundle_eight_view_template_0727_v13/runtime_bundle.json`, with canonical identity
  `94b025f85bb2f529fc36a0f62a941eafe30cbd71b1c2298431c24b539a29e2bf`.
  Both secondary views require Template/PatchCore/YOLO; PatchCore, YOLO, and ROI bindings remain the v12 values.
- Stage35 and Dashboard now default to v13. v13, v12, v11, and v10 were separately fresh-loaded successfully; select
  any rollback bundle explicitly through `--runtime-config`. This remains commissioning-only/non-production because
  the existing YOLO test-leakage and model/ROI rebind warnings are unchanged.
- Exact hashes, all manual Template/PatchCore/YOLO thresholds, the Hikvision-SDK live command, and rollback commands
  are appended to `docs/ZS32_0727_TEMPLATE_V12_REPLACEMENT_20260727.md`.

### ZS32 fast Demo platform clean-runtime handoff (2026-07-27)

- The user no longer wants the online platform to retain a strict/runtime-bundle mode. The target is one clean Demo
  path driven by a single editable JSON, with Template/PatchCore/YOLO on all eight views and threshold changes taking
  effect on the next part without SHA regeneration or GPU model reload.
- Keep only low-cost operational checks: exact eight views and four-camera topology, decodable inputs, valid ROI/model
  paths, finite thresholds, missing evidence never yielding OK, unique outputs, and visible Dashboard errors/DEMO label.
- The copy-ready new-session task brief is
  `docs/ZS32_FAST_DEMO_PLATFORM_NEW_SESSION_PROMPT_20260727.md`. It defines the target architecture, removals,
  phased implementation, minimal tests, and final command. No runtime refactor was performed when this brief was added.

### ZS32 fast Demo platform implementation (2026-07-27)

- The clean Demo path is now implemented. The only operator config is `configs/zs32/zs32_demo.json`; the only online
  entry is `pipeline/36_zs32_inspection_dashboard.py --live --demo-config ... --part-id ...`.
- `capture_data/zs32_demo_config.py` validates exact four-camera/eight-view topology, ROI bounds, all real model paths,
  and finite branch thresholds without loading or comparing any SHA. Thresholds reload before every part.
- `capture_data/zs32_demo_runtime.py` directly retains eight Template groups, eight PatchCore models, and one YOLO
  model. Its initial implementation ran all 24 branches even after Template NG; this behavior was later superseded by
  the global Template gate recorded below. Model/ROI/topology/inference-setting changes require a Dashboard restart.
- Fusion is local and fail-closed: ERROR, then NG_TEMPLATE, NG_ANOMALY, NG_YOLO, otherwise OK. The unified
  `runtime_manifest.json` contains source/evidence paths, score, threshold, status, and no bundle/SHA fields.
- Dashboard -> Stage35 -> resident worker -> Demo runtime no longer imports or calls runtime bundle, Stage18,
  threshold artifact, fusion profile, or SHA validation. Dashboard shows `DEMO / 非生产`, config path/mtime, branch
  score/threshold, worker log path, and the real first error.
- Saved eight-view `group004` completed an actual CPU inference smoke: 24 model branches, 65 files, zero execution
  errors, `NG_TEMPLATE` only on `back_secondary`. Dashboard parser loaded the emitted manifest successfully.
- Focused validation passed: 38 Demo config/runtime/CLI tests, 28 live-core tests, 6 Stage35 CLI tests, 123 Dashboard
  tests, 11 Dashboard CLI tests, and all 7 Unix-socket worker tests. Real four-camera capture is still pending.
- Historical Stage18/34/37 scripts, bundle configs, and results were not physically deleted. They are retained only
  for history/replay; do not use the earlier v13 `--runtime-config` guidance for online operation.

### ZS32 Demo global Template gate restored (2026-07-27)

- The user explicitly replaced the earlier all-24-branches-on-NG behavior with a global Template gate.
- Every part first evaluates all eight Template views. If any Template is `NG_TEMPLATE`, the part ends conclusively as
  `NG_TEMPLATE`; PatchCore and YOLO are not executed and remain score-free `SKIPPED` branches. Fusion is published
  as available per view, mirroring that view's Template `NG_TEMPLATE` or `PASS`, with a real evidence image.
- A Template-gated NG remains `inspection_complete=true` because it is a valid business decision, not an execution
  error. Template execution errors remain `ERROR/inspection_complete=false` and also skip downstream models.
- Only when all eight Template views PASS may the runtime execute all eight PatchCore and eight YOLO branches and apply
  normal fusion. Do not restore unconditional downstream execution without fresh user approval.
- Dashboard main title is exactly `ZS32 八视角检测`; `DEMO / 非生产` remains a separate visible badge rather than part
  of the title.

### ZS32 bootstrap operator confirmation has no default deadline (2026-07-27)

- `pipeline/zs32_bootstrap_capture.py` now waits indefinitely by default for both terminal Enter confirmation and
  Dashboard round confirmation. The former 120-second default was removed.
- `--round-confirmation-timeout SECONDS` remains available when a positive explicit deadline is wanted.
- `--timeout-ms` is unchanged and still controls individual camera-frame retrieval only; it does not limit manual
  loading or front/back positioning time.
- Regression evidence: all 13 bootstrap CLI tests and both live progress/coordinator tests passed on CPU. The bootstrap
  suite needed `--confcutdir=tests/unit/zs32_refactor/capture_data` because its parent conftest otherwise skips every
  test when CUDA is unavailable.

### ZS32 0727 two-session right-normal merged manifest (2026-07-27)

- The two raw sessions under `dataset/zs32_0727` remain unchanged. Their training-ready combined inventory is
  `dataset/zs32_0727/manifests/zs32_0727_right_normal_merged.csv`; use this manifest instead of copying the 5.6 GB
  image tree.
- The merged manifest preserves the original session identities and contains 104 complete physical samples and 832
  unique images: 20 samples from `zs32_4cam_accept_20260727_183107_567838700` and 84 from
  `zs32_4cam_accept_20260727_184011_699976351`.
- The cancelled second-session `group085` is excluded. Verification found 104/104 exact eight-view sets, 0 duplicate
  `(session_id, sample_id, view)` keys, 0 duplicate paths, 0 missing files, and 0 invalid PNG signatures.

### ZS32 0727 normal plus unique defect Template/PatchCore v14 preparation (2026-07-27)

- The user confirmed the two 0727 sessions contain 104 different physical parts. Their 832 raw images were cropped
  with the established eight-view ROI into `dataset/zs32_0727_right_normal_roi_104part_v1`; ROI SHA256 remains
  `9412b2838cdb96f722db356714cf9bbbb5ea01810671ccf92b67323de77ebe65`.
- The training release is `dataset/zs32_0727_plus_zs32_all_unique_defect_release_v1`. It combines 104 normal parts
  with 23 unique `zs32_all/right` defect parts by file-level symlink and preserves all eight views including both
  secondary views.
- Exclude `zs32_right_deform_20260714_201404_438451409`: its five parts are byte-identical duplicates of another
  defect session. Retained defect identities are deform 12, less 2, others 9; no source files were deleted.
- Seed-42 normal roles are 62 train, 16 model_val, 13 calibration, 13 final_test. Defect roles are 11 calibration,
  6 model_val, 6 final_test, stratified by type. No defect enters Template selection or the PatchCore Memory Bank.
- `template_manifest.csv` has 1016 rows. PatchCore's active trainer surface has 688 rows: per view 62 normal train,
  13 normal_test, and 11 calibration defects. All model_val/final_test parts remain in release holdout paths.
- Reusable builder: `pipeline/prepare_zs32_0727_template_patchcore.py`; commands and limitations:
  `docs/ZS32_0727_TEMPLATE_PATCHCORE_V14_TRAINING.md`. Candidate results must use new v14 roots and are not
  automatically integrated into `configs/zs32/zs32_demo.json`.
- The builder rejects unsafe session path segments, removes a newly published release if post-publish validation
  fails, and cross-checks every split, Template route, PatchCore route, active trainer row, symlink, and content hash.
  It also proves Template and PatchCore use the same source image for every physical-part/view key and recomputes the
  release summary from the manifests. The release uses absolute symlinks, so its two source ROI directories must
  remain in place and unchanged through training.

### ZS32 top front/back-only incremental PatchCore v15 (2026-07-27)

- The user confirmed `dataset/zs32_top` groups 001-028 are 28 distinct normal parts; cancelled group029 is excluded.
  Only `front` and `back` are used. The other six captured views remain untouched and are never routed into training.
- Reproducible builder: `pipeline/prepare_zs32_top_patchcore.py`; release:
  `dataset/zs32_top_front_back_patchcore_release_v1`. Seed 42 assigns new parts 20 train, 4 calibration, 4 final_test.
  Per view the active surface is 82 normal train, 17 normal calibration, and 11 reused v14 calibration defects.
- Candidate result: `results/zs32_patchcore_front_back_top_incremental_seed42_v15`. Exactly two checkpoints exist:
  front SHA256 `7c005651ee3756416a9deeb37c23089d5bf6f062e3fc53113f5186401d503e03`, back SHA256
  `d187724a5d889e696644e5f703d302be6935f62c307a92c08238b191a1b54e13`.
- Candidate thresholds are front `0.3681967556476593`, back `0.4843376576900482`. Independent holdout detects all
  24 old defects, but normal part-level front-or-back false rejects are 4/33. New-normal holdout is front 3/4 and
  back 4/4 PASS.
- The prior saved live normal probe remains front `1.0` and back `0.9134062` under v15, so this candidate is not
  integrated into `configs/zs32/zs32_demo.json`. Keep active v14 weights until the user explicitly approves otherwise.

### ZS32 live HDR matches top-data acquisition (2026-07-27)

- Stage35 online capture now matches the actual `dataset/zs32_top` acquisition command: 1500/5500 us exposures,
  gain 0, interval 0.2 s, settle 1, timeout 2000 ms, short-dark threshold 80, long-clip threshold 245,
  blend width 50, blur size 101, zero HDR retries, max clip 5%, and HDR alignment disabled.
- Keep these image-formation parameters aligned when collecting PatchCore training data; changing the HDR thresholds,
  blend width, or blur size changes the fused PNG distribution even when exposure and lighting remain fixed.
## BMW DA9625347 bright-streak customer Demo design (2026-08-05)

- The approved Demo is isolated from ZS32 and uses exactly one Hikvision camera bound by serial `DA9625347`.
- The fixture, light, camera, and part position are fixed. The UI performs one software-triggered Mono8 capture per
  button click and intentionally has no live preview.
- `dataset/bmw/Image_20260805160156065.bmp` is the approved `OK` reference with a continuous central bright streak;
  `dataset/bmw/Image_20260805160207779.bmp` is `NG_NO_STREAK` with no streak.
- The pure OpenCV detector uses a user-selected fixed ROI, local background suppression/white top-hat response,
  robust thresholding, component filtering, and an along-streak one-dimensional presence sequence.
- Stable decisions are `OK`, `NG_NO_STREAK`, `NG_BROKEN`, and `ERROR`; capture/image-quality/configuration failures
  must never be presented as part NG.
- The approved design is `docs/designs/2026-08-05-bmw-bright-streak-demo-design.md`. Implementation order is ROI
  selector, offline detector, then serial-bound live camera and local OpenCV UI.

## BMW DA9625347 bright-streak Demo implementation (2026-08-05)

- The implementation lives in `src/bmw_inspection` and is intentionally independent of ZS32 product decisions.
  Editable runtime settings and the seed half-open ROI `[1825, 1290, 1870, 1405]` are in
  `configs/bmw/bright_streak_demo.json`.
- Select an exact ROI with `uv run --no-sync python pipeline/bmw_select_bright_streak_roi.py --image <bmp>`; Enter
  atomically updates only `roi_xyxy` in the JSON. Start the live customer Demo with
  `uv run --no-sync python pipeline/bmw_bright_streak_demo.py`; it binds only serial `DA9625347`, discards one
  warm-up frame, and captures one software-triggered frame for each Capture/Retry action.
- Offline replay uses the same detector and publisher: add `--image <bmp> --no-gui`. Every run is atomically
  published below `results/bmw_bright_streak_demo/YYYYMMDD/<timestamp>` with `source.png`, `roi.png`,
  `response.png`, `mask.png`, `evidence.png`, and finite `result.json`.
- The focused acceptance suite is `tests/unit/bmw_inspection`. At implementation time it passed 32 tests; offline
  smoke classified `Image_20260805160156065.bmp` as `OK`, `Image_20260805160207779.bmp` as `NG_NO_STREAK`, and
  the deterministic synthetic interrupted line as `NG_BROKEN`. Real-camera USB/MVS smoke remains a site step.
- The ROI selector must not pass the 4024x3036 source directly to `cv2.selectROI`. It now fits the selection image
  inside 1280x720 by default and maps display-space XYWH back to half-open source coordinates using independent X/Y
  ratios. Smaller screens can override `--max-display-width` and `--max-display-height`.
- The original two root-level BMW BMP references were replaced on 2026-08-05 by 6 OK images under
  `dataset/bmw/OK` and 7 no-bright-streak NG images under `dataset/bmw/NG`; all are 4024x3036 Mono8. The selector
  default is now `OK/Image_20260805172921398.bmp`.
- The operator-confirmed shared ROI for the new dataset is half-open `[1872, 1180, 1953, 1793]`. The fixed-light
  OK streak is saturated across nearly the full ROI height, while every labeled no-streak NG has zero saturated
  support. The detector therefore accepts a centered, narrow saturated-line mask before falling back to the original
  white-top-hat/component path, and coverage/longest-run ratios are normalized by the full ROI height so a short NG
  reflection cannot appear 100% continuous.
- Batch acceptance on the new dataset passed exactly 6/6 OK as `OK` and 7/7 NG as `NG_NO_STREAK`; the synthetic
  interrupted-line regression remains `NG_BROKEN`. The focused BMW suite passed 34 tests. The shared config uses
  `max_dark_clip_ratio=0.55` because the confirmed ROI intentionally includes a fixed dark background corridor.
- Live `DA9625347` evidence later showed valid continuous streaks with 97.6%-98.4% full-ROI coverage and zero gaps,
  but contrast SNR varied from 3.52 to 4.31 and the original `min_contrast_snr=4.0` caused unstable false NG. The
  threshold is now 3.0, leaving at least 0.5 observed normal margin; labeled no-streak NG remains separated by
  coverage (0 saturated support in the 7-image dataset, and at most 1.2% in recorded live no-streak results).
- The customer Demo UI is a 1600x900 fully Chinese dark industrial dashboard titled `BMW 零件亮痕检测演示系统`.
  It uses `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc` (override with `BMW_DEMO_FONT`), flat bordered
  cards, a semantic result rail, Chinese metric rows, and hover feedback on the existing buttons. Offline screenshots
  truthfully show `离线图像回放`; only a successful live session shows `相机已连接`. Raw English reasons and result
  paths remain in artifacts but are intentionally absent from the presentation surface. For no-streak NG, maximum
  gap displays `不可用` rather than the misleading `0 像素 · 合格`.
- The narrow ROI evidence is rotated 90 degrees clockwise only in the dashboard, then proportionally fitted into the
  566x200 evidence viewport. Do not restore the former non-uniform 250x192 stretch. Detector coordinates, masks,
  `roi.png`, `evidence.png`, and decision metrics remain in the original camera orientation.
- The live Demo opens in a connected `等待检测` state and must not acquire, detect, publish, or reuse a prior result
  until the operator clicks `开始检测` or presses Space. Before the first result, `重新检测`/R is disabled and Quit
  returns successfully without a result; after a result, Start/Retry each acquire exactly one new frame. The focused
  GUI regression explicitly proves zero startup acquisitions and one acquisition for Space followed by Quit.

## BMW six-view laboratory inspection plan (2026-08-05)

- The confirmed lab target uses three fixed Hikvision cameras and two manual rounds on the same physical part:
  centre `DA9805574`, left `DA9625347`, and right `DB0968108`; front then operator flip then back produces canonical
  views `front`, `front_left`, `front_right`, `back`, `back_left`, and `back_right`.
- Keep the customer bright-streak Demo intact. Add an isolated `bmw_inspection.lab` package rather than adapting the
  ZS32 eight-view release/runtime contracts. Models remain resident; thresholds reload between parts.
- Template is a six-view wrong-part/pose/large-structure gate. Any Template NG stops bright-streak, YOLO, and
  PatchCore. If Template passes, final OK requires every configured required downstream branch to execute and pass.
- YOLO is one global single-class (`defect`) model with bounding-box labels and candidate/final thresholds separated.
  PatchCore uses one model and threshold per view. Bright-streak routing is config-driven; the initial mapping reuses
  the approved ROI `[1872, 1180, 1953, 1793]` on `front_left` only until other view ROIs are explicitly selected.
- The existing 6 OK and 7 no-streak NG images represent 13 distinct physical parts but remain single-view regression
  evidence. Complete six-view data and YOLO boxes still need collection before training/evaluating the full system.
- Final statuses distinguish product NG, RETAKE, REVIEW, and ERROR. Disabled/missing required branches never produce
  OK, and Template/image/model failures remain truthful. Experiment mode and Presentation mode share one Chinese UI.
- The executable task plan is `docs/superpowers/plans/2026-08-05-bmw-six-view-lab-inspection-plan.md`; it covers
  contracts/config, six-view capture, structured bright-streak evidence, Template, manifests, YOLO, PatchCore,
  Template-first fusion, publishing, batch evaluation, UI, and minimum acceptance checks.

## BMW six-view laboratory implementation progress (2026-08-05)

- Tasks 1-3 are implemented under `src/bmw_inspection/lab`: strict six-view contracts/config, fixed three-camera
  two-round capture, strict offline six-view loading, and a structured adapter for the existing bright-streak rule.
- The shipped topology enforces the exact binding centre `DA9805574`, left `DA9625347`, right `DB0968108`; a unique
  but incorrect serial, swapped slot, wrong view mapping, or wrong topology ID is rejected.
- Learned branches are initially disabled because BMW-specific assets do not exist. The profile still marks all four
  branches required, so a partial profile cannot yield OK. YOLO candidate/final thresholds are separate and validated.
- Template and PatchCore settings are per view. Bright-streak is initially configured only on `front_left`; the other
  five views emit non-required `SKIPPED` evidence rather than incorrectly forcing REVIEW.
- `CaptureSet` copies all six uint8 images and marks them read-only; it rejects incomplete, mixed-size, or mutable
  aliasing inputs. The CLI remains idle until an explicit front capture and explicit back capture after the flip.
- Bright-streak decisions now carry the actual response, decision mask, runs, gaps, and measured acquisition/processing
  timings. Published evidence is taken from the exact decision object and is never recomputed independently.
- Independent review caught and fixed a 0/255-mask width inflation bug, contradictory publisher fallback, missing
  Python 3.10 enum compatibility, weak topology identity, invalid YOLO threshold relationships, and mutable captures.
- Root verification after these fixes passed all 93 BMW tests with `--import-mode=importlib`; the only warning is the
  environment's Torch NVML initialization warning. Template and dataset-manifest Tasks 4-5 are now in progress.
- Tasks 4-5 are now implemented and independently reviewed. Template publishes six independent groups, selects 3-5
  medoid/farthest-point references, searches bounded XY translation, fits thresholds from calibration only, reports
  final-test metrics without feeding them into selection, verifies model/template hashes, and refuses overwrite.
- Template manifest, one-view training, and six-view training all reject one physical `part_id` crossing splits.
  The ROI selector maps scaled display coordinates back to half-open 4024x3036 source coordinates.
- The BMW dataset builder requires complete six-view samples and explicit part/session IDs, assigns deterministic
  60/20/20 part splits, rejects content/view/split leakage, and needs at least 30 parts unless explicitly experimental.
- The current 6 OK/7 NG files are emitted only in a separate bright-streak regression manifest with expected statuses
  `OK` and `NG_NO_STREAK`; they never enter YOLO or PatchCore exports.
- YOLO export is one class `defect`, requires valid in-bounds boxes, and its `data.yaml` omits `path` so Ultralytics
  resolves train/val/test relative to the YAML directory from any working directory. PatchCore exports only normal
  train images to each per-view memory bank and keeps calibration/final-test good/defect directories separate.
- Root verification after Task 4-5 review fixes passed all 78 BMW lab tests. YOLO and PatchCore Tasks 6-7 are next.

## BMW six-view laboratory implementation completion boundary (2026-08-05)

- Tasks 6-10 are implemented under `src/bmw_inspection/lab` and `pipeline/bmw_lab_*.py`. The current local unit
  acceptance is `217 passed` for `tests/unit/bmw_inspection`; the only warning is Torch NVML initialization in this
  environment. `compileall`, `git diff --check`, CLI `--help`, `uv lock --check`, and an offline 1600x900 screenshot
  smoke also passed. Ruff is not installed in the current environment, so no lint-pass claim is allowed.
- `pyproject.toml` declares `bmw-lab = ["ultralytics==8.4.89"]`; `uv.lock` contains the same package and extra, and
  `uv sync --extra bmw-lab --frozen --dry-run` recognizes it. Do not rely on the separate
  `/home/yunjing/ultralytics-c789` checkout as the dependency contract.
- YOLO training and runtime now share the same spatial contract: Task 5 exports each view's configured part-ROI crop,
  clips/re-normalizes `defect` boxes to that crop, and calibration reads the original full frame then applies the ROI
  exactly once. The resident single model is inference-locked. Candidate boxes remain diagnostic; only boxes meeting
  the final threshold are red/business NG. Training reads calibration only, uses `trainer.best`, persists actual
  Ultralytics args/version, validates in staging, and atomically publishes without overwrite.
- PatchCore trains six independent view models and verifies one identical calibration image through the training and
  independently loaded runtime paths before publication. The receipt binds the checkpoint SHA and raw score/map at
  `atol=1e-6`. Raw anomaly-map dtype is preserved; display normalization is separate. A view without calibration
  defect data remains REVIEW and cannot form an OK profile.
- Runtime order is quality -> all six Template -> global Template gate -> configured bright-streak -> six-view YOLO ->
  six-view PatchCore -> fusion -> atomic evidence publication. Template non-PASS invokes no downstream model. Fusion
  priority is `RETAKE > ERROR > confirmed NG > REVIEW`; an explicit product NG is never hidden by simultaneous
  incomplete evidence. Cold changes (checkpoint/config content, Template image dependencies, ROI, preprocessing,
  candidate floor, capture/topology) require reload; final business thresholds and enabled/required flags are hot.
- Publications live under `<result_root>/<experiment_id>/runs/YYYYMMDD/<run_id>`. They bind the config digest, model
  path/version/SHA, source images, branch crops, Template difference/overlay, bright-streak ROI/response/masks/overlay,
  YOLO candidates plus final-only overlay, raw PatchCore map plus true overlay, timings, and per-view fusion overlays.
- Batch evaluation is physical-part-level and accepts only calibration/final_test. Final-test threshold writes are
  rejected before manifest reads. Complete fused metrics require a trusted config-derived profile contract plus exact
  archived result/config/evidence agreement; generic self-reported runtimes remain experimental-only. Sample IDs bind
  one part, six paths and six hashes must be distinct, current image SHA values must match, and all public branch
  metrics aggregate once per part. Threshold writes use snapshot/restore rollback around publication.
- The new 1600x900 laboratory UI uses a light Swiss grid and fully Chinese business labels. Presentation mode selects
  only actual triggered evidence; missing artifacts display `当前视角没有该项目证据` instead of substituting an original
  or PASS view. Experiment mode adds scores, thresholds, timings, model identity, view/branch keys, amber diagnostic
  YOLO candidates, red final boxes, and explicit model reload. Only the published bright-streak ROI bitmap is rotated
  90 degrees clockwise; detector coordinates and archived evidence remain unchanged. Startup performs zero captures,
  processing is painted before synchronous inference, and live camera sessions are context-owned and reopened after
  cold reload.
- Operator and training commands are documented in `pipeline/README.md`. The final offline UI smoke used one existing
  OK bright-streak image copied into six temporary view names only to exercise control/render/publication paths; it
  truthfully returned REVIEW because Template/YOLO/PatchCore assets are disabled. It is not accuracy or six-view data
  evidence.
- Remaining site/data work is explicit: collect at least 30 complete physical parts with six distinct real views and
  actual `defect` boxes, train/publish six Template groups, one YOLO model, and six PatchCore models, activate a new
  experiment profile, run independent final-test evaluation, then perform the foreground three-camera/two-round live
  smoke with `DA9805574`, `DA9625347`, and `DB0968108`. No such real training or hardware acceptance was performed in
  this implementation session, so the four-branch profile must not yet be described as performance-complete.

## BMW four-camera eight-view minimal HDR collection (2026-08-06)

- The user replaced the BMW acquisition topology with four fixed cameras: centre `DA9805574`, left `DA9625347`,
  right `DB0998274`, and secondary/front-mounted `DB0968108`. Front/back rounds produce canonical views
  `front/front_left/front_right/front_secondary` and `back/back_left/back_right/back_secondary`.
- The user explicitly requested the smallest fast implementation. The immediate entrypoint is
  `pipeline/bmw_lab_collect_data.py`; its versioned profile is
  `configs/bmw/capture/bmw_4cam_eight_view_hdr_v1.json`. It is a thin BMW wrapper over the already-tested
  topology-driven four-camera bootstrap collector and does not modify the legacy three-camera collector.
- The fixed acquisition identity is HDR short/long `1500/6000 us`, gain `0`, interval `0.2 s`, settle `1`, timeout
  `3000 ms`, no alignment, dark/clip `70/245`, blend width `18`, blur size `31`, zero retries, and max clip `12%`.
  The minimal path saves fused PNGs only and writes the legacy eight-image plus one-sample session manifest.
- The existing `bmw_inspection.lab` runtime, model configs, dataset builder, evaluation, and UI remain the older
  three-camera/six-view/single-exposure contract. Do not feed the new eight-view HDR set into that runtime or claim
  model readiness until the next explicit full-system upgrade.
- The first live wrapper run reached the front round but failed before enumeration with
  `MvCameraControl_class is unavailable`. Root cause: the reused topology-driven adapter intentionally does not
  mutate `sys.path`, while the older collector automatically exposes `/opt/MVS/Samples/64/Python/MvImport`.
  `pipeline/bmw_lab_collect_data.py` now validates that SDK file and prepends the directory before device listing or
  capture. A subprocess import probe resolves the module to the installed vendor file after the fix.

## BMW eight-view HDR canonical data preparer (2026-08-06)

- Entrypoint: `pipeline/bmw_lab_prepare_eight_view_data.py`; core implementation:
  `src/bmw_inspection/lab/eight_view_dataset.py`. It is deliberately independent from the legacy six-view BMW
  `ViewId` and runtime.
- It reads the frozen 25-column append-only capture manifests, keeps only explicit complete samples with exact eight
  HDR fused views, validates round/view/camera bindings and image dimensions, and excludes incomplete samples.
- Physical identity follows the capture-native part instance: strip the final six-digit image index from `sample_id`
  (for example `bmw_edge_group003_000001 -> bmw_edge_group003`). Splits are deterministic and stratified by capture
  class; all views and repeat samples for one identity remain in one split.
- Branch labels remain truthful: `no_streak` is business NG but EfficientAD/Template normal, YOLO negative, and
  `front_left` bright-streak `NG_NO_STREAK`. `deform/edge/others` are `review_required` per view until visibility and
  boxes are manually confirmed.
- The immutable reference-only release contains `dataset_manifest.csv`, `part_splits.csv`, four branch manifests, and
  `report.json`; it does not copy images, crop ROI, invent boxes, train models, or feed the old six-view runtime.
- Focused implementation acceptance: 19 tests passed across the new preparer and legacy BMW dataset builder; CLI help
  and compileall passed.
- The hashed real-data release is `dataset/bmw_lab_prepared/bmw_hdr_eight_view_v1`: 5 input manifests, 132 complete
  physical-part captures, 5 incomplete samples excluded, 1056 images (132 per view), and exact class counts normal 85,
  deform 10, edge 17, others 4, no_streak 16. Seed 42 produced 79 train, 26 calibration, and 27 final-test parts.
  `report.json` records full source-image SHA-256 verification and hashes for all six published CSV manifests.

## BMW eight-view ROI selector and branch data materializer (2026-08-06)

- ROI entrypoint: `pipeline/bmw_lab_select_eight_view_rois.py`; typed contract:
  `src/bmw_inspection/lab/eight_view_roi.py`. It selects one complete `normal/train` sample, preflights all eight source
  hashes/dimensions before opening GUI, maps scaled display XYWH to source half-open XYXY, and writes a manifest-bound
  eight-view ROI JSON. Existing ROI configs are refused unless `--force` is explicit.
- Materializer entrypoint: `pipeline/bmw_lab_materialize_training_data.py`; implementation:
  `src/bmw_inspection/lab/eight_view_training_data.py`. It writes one lossless canonical PNG crop per source image,
  validates round-trip pixels and hashes, then uses relative symlinks for EfficientAD, Template, and YOLO adapters.
- EfficientAD treats `normal/no_streak` as branch-good and keeps final_test under `efficientad/held_out`. Template train
  references use source normal only; no_streak remains normal in calibration/final_test. Review-required defect views
  never become anomaly/Template defects unless a reviewed non-empty YOLO label proves visible local defect evidence.
- Current release has 808 confirmed YOLO negatives and 248 review-required defect views. Without a reviewed txt for
  every one of those 248 rows, materialization publishes `yolo/annotation_queue.csv` but deliberately omits
  `yolo/data.yaml`. Empty reviewed txt means human-confirmed invisible/negative; non-empty labels must use class 0
  `defect` and valid normalized crop coordinates.
- The selector/materializer preserve the old six-view runtime and do not train models. Synthetic focused tests cover
  exact pixels, labels, symlinks, manifest binding, and no-overwrite behavior.
- The operator-selected ROI asset is `configs/bmw/rois/bmw_hdr_eight_view_v1.json`, bound to prepared manifest SHA256
  `6391b456653867bca413cd6f826ddd7b96f3b55cbaf85a585c4e7625189d77c6`; its own SHA256 is
  `414957f7a8d1d7156cd25ebdd01e16d7c3568dc1e697b42fec20ac0645fba1ca`.
- The real immutable ROI release is `dataset/bmw_lab_training/bmw_hdr_roi_training_v1`: 1056 canonical PNG crops,
  132 per view, 728 Template manifest rows, 808 YOLO negative labels, and 248 pending defect-view annotations.
  `yolo_training_ready=false` and no `data.yaml` is intentionally published until all pending rows have reviewed txt.

## BMW portable YOLO labeling handoff (2026-08-07)

- Entrypoint: `pipeline/bmw_lab_prepare_labeling_package.py`; implementation:
  `src/bmw_inspection/lab/labeling_package.py`. It consumes only the immutable training release's
  `yolo/annotation_queue.csv`, refuses normal/no_streak rows, and publishes without overwriting an existing package.
- Current handoff is `dataset/bmw_lab_labeling/bmw_hdr_roi_yolo_248_v1`: 248 real copied ROI PNGs (not symlinks), one
  flat collision-safe image namespace, eight views with 31 tasks each, and source classes deform 80, edge 136,
  others 32. Total image payload is 821,622,396 bytes.
- The handoff includes `annotation_tasks.csv`, `SHA256SUMS`, Chinese `标注说明.md`, Label Studio `tasks.json`, and a
  single-class `defect` interface with required visible/invisible review choice. The annotator must return both JSON
  and YOLO exports so completed empty views remain distinguishable from unfinished tasks.
- Superseding handoff `dataset/bmw_lab_labeling/bmw_hdr_roi_yolo_248_refs8_v2` keeps the same 248 annotation tasks and
  adds 64 real normal-train reference PNGs under `normal_references/<view>/`: eight complete normal physical parts and
  exactly eight references per canonical view. References have their own manifest and never enter Label Studio tasks.

## BMW Label Studio API review project (2026-08-10)

- The original-account Label Studio database is `/home/yunjing/.local/share/label-studio/label_studio.sqlite3`.
  Project `BMW八视图标签复核_248` was created through the local API as project ID `14`; its review page is
  `http://127.0.0.1:8080/projects/14/data`.
- The imported source is `dataset/bmw_lab_labeling/bmw_label_review_248_linux.json`: 248 tasks, 247 existing
  annotations, and one intentionally unfinished task. API verification returned the same 248/247/247
  task/annotation/finished counts.
- Local image storage ID `8` points at
  `dataset/bmw_lab_labeling/bmw_hdr_roi_yolo_248_refs8/images`. It is deliberately non-synchronizable so it only
  authorizes image reads and cannot duplicate the JSON-imported tasks. Local PNG access was verified with HTTP 200.
- The unfinished task is
  `20260806_180848_928560__bmw_others_group003_000001__back_left.png` (task ID `2314`). The known choice/box review
  target is `20260806_175121_541680__bmw_deform_group008_000001__back_right.png` (task ID `2139`).
- Start the service with the original database and the labeling package as document root:
  `DEBUG=false LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/home/yunjing/anomaly_xingtao_new/dataset/bmw_lab_labeling/bmw_hdr_roi_yolo_248_refs8 conda run --no-capture-output -n label-studio label-studio start --data-dir /home/yunjing/.local/share/label-studio --port 8080 --no-browser`.

## BMW Label Studio reviewed export attempt (2026-08-10)

- Project 14 reached 248 tasks, 248 annotations, and 248 finished tasks. The API export is preserved under
  `dataset/bmw_lab_labeling/exports/bmw_label_review_project14_20260810_030838` as raw Label Studio JSON plus raw
  YOLO ZIP; the ZIP passes integrity testing and contains all 248 expected label files.
- This export is not yet final training truth. Validation found one remaining semantic contradiction on task ID 2139,
  `20260806_175121_541680__bmw_deform_group008_000001__back_right.png`: the choice is `无可见缺陷` while one
  `defect` rectangle remains. Do not silently choose which side is correct; fix it in Label Studio and re-export.
- The built-in Label Studio YOLO exporter lists the two review-choice strings in `classes.txt` in addition to
  `defect`, although every emitted box line currently uses valid class ID 0. Preserve the raw ZIP for audit and use
  a BMW-specific normalized label directory for the immutable training-data materializer only after the semantic
  contradiction is resolved.

## BMW Label Studio final reviewed export (2026-08-10)

- The user confirmed task ID 2139 is genuinely `无可见缺陷`. Annotation ID 1740 was updated through the local API by
  deleting its single stale `defect` rectangle while preserving the `无可见缺陷` choice.
- The immutable final export is
  `dataset/bmw_lab_labeling/exports/bmw_label_review_project14_final_20260810_032057`. It contains the raw Label
  Studio JSON, raw Label Studio YOLO ZIP, extracted raw export, a normalized `reviewed_yolo_labels` directory, an
  export manifest, and a validation report. Do not use the earlier `...030838` review-required export for training.
- Final validation is clean: 248 tasks and annotations, 248 expected/actual label files, 131 images with boxes, 117
  empty labels, 170 total class-0 boxes, no invalid YOLO rows, and no review-choice/box contradictions. The normalized
  label root contains exactly the 248 expected per-image txt files, with no extra `classes.txt`, and is the direct
  input for the strict `--yolo-label-root` materializer.
- Final raw artifact identities: Label Studio JSON SHA256
  `7758a3e8ef1b29190fa4c73d89704e68446e2614802ebf7442c0e52b8a3c50ea`; YOLO ZIP SHA256
  `712884ec289c9b2cb31155fe46df3e9dbfabb7e8b90dcd7ea9406bce0dce67a1`.

## BMW eight-view laboratory one-click training (2026-08-10)

- Entrypoint: `pipeline/bmw_lab_train_all.py`; implementation:
  `src/bmw_inspection/lab/eight_view_train_all.py`. It is an experimental-only, fail-fast sequence of final-label
  materialization, eight Template groups, `front_left` bright-streak calibration, eight serial EfficientAD-S models,
  and one global YOLO26n model. It does not activate assets in the legacy six-view runtime.
- Confirmed defaults are EfficientAD-S `256x256`, 30 epochs and framework-required train batch 1; YOLO26n `640`,
  100 epochs and batch 32 on GPU 0. The canonical reviewed labels are the 248-file normalized directory under the
  final project-14 export; the immutable training target is `bmw_hdr_roi_training_reviewed_v1`.
- Bright-streak fitting uses calibration only. Current HDR data cleanly separates calibration normal/no_streak;
  fitted presence thresholds are contrast SNR `3.289477...` and coverage `0.190048...`. Because no interrupted-streak
  defect examples exist, continuity remains an explicit rule fitted as the strict calibration-normal envelope
  (`min_longest_run_ratio=0.331158...`, `max_gap_ratio=0.070146...`, `max_gap_count=2`). The held-out result is reported,
  not used to tune.
- Recommended run: `uv run --no-sync python pipeline/bmw_lab_train_all.py --run-id bmw_lab_eight_view_v1`.
  Preflight-only command: append `--dry-run`. Focused acceptance is 7 unit tests plus CLI compile/help and real-data
  dry-run; no GPU training was started during implementation.

## BMW eight-view trained-model assessment and independent Demo (2026-08-10)

- Trained run `results/bmw_lab_one_click/bmw_lab_eight_view_v1` is complete. Held-out Template balanced accuracy is
  0.925-1.000 across the eight views with no defect false accepts; the calibrated bright-streak rule has final-test
  balanced accuracy 0.8824 with 4 false rejects and 0 false accepts. YOLO's independent test is the weak branch:
  precision 0.557, recall 0.279, mAP50 0.219, mAP50-95 0.091. EfficientAD held-out AUROC ranges 0.7375-1.0, with
  `back_right` and `back_secondary` needing the most future threshold/data work. This is suitable for a lab Demo and
  rapid iteration, not an industrial acceptance claim.
- Independent entrypoint: `pipeline/bmw_lab_eight_view_demo.py`; config:
  `configs/bmw/experiments/bmw_eight_view_demo_v1.json`. It leaves the legacy six-view UI untouched and supports
  default live four-camera/two-round HDR capture, `--sample-id`, exact-name `--capture-set`, `--experiment-mode`,
  `--no-gui`, and `--save-screenshot`.
- Runtime modules are `src/bmw_inspection/lab/eight_view_demo.py`, `eight_view_demo_capture.py`,
  `eight_view_demo_models.py`, and `eight_view_demo_ui.py`. Every completed part runs all 25 checks: 8 Template,
  `front_left` bright streak, 8 YOLO, and 8 resident EfficientAD. Fusion is ERROR on any model error, otherwise NG on
  any NG, otherwise OK. Startup is a blank waiting state; the Chinese 4x2 dashboard has presentation/experiment
  modes, switchable evidence, and a clockwise-rotated bright-streak evidence image.
- Real GPU smoke used held-out normal sample `bmw_normal_group072_000001`: 25 PASS, final OK, 3478.956 ms total on
  RTX 4090. Screenshot: `results/bmw_eight_view_demo/bmw_normal_group072_demo.png`. Focused verification passed 13
  tests plus compileall. Hardware camera acquisition is wired to the approved HDR profile but was not live-triggered
  during this implementation turn.
- The eight-view Demo typography was sharpened without changing layout: body text uses installed
  `/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc`, while titles, final status, branch names, and branch states
  use `NotoSansCJK-Bold.ttc`. Small copy is 16-18 px and muted text uses a darker neutral grey. Overrides are
  `BMW_DEMO_FONT_MEDIUM`, `BMW_DEMO_FONT_BOLD`, or the shared `BMW_DEMO_FONT`. The real normal-sample screenshot was
  regenerated at the same path and visually checked for clipping; the smoke remained 25 PASS and final OK.

## BMW eight-view GitHub handoff (2026-08-10)

- The public handoff branch is `agent/bmw-eight-view-handoff` in
  `wjstx0425/anomaly_xingtao_new`; published commit `6ed45cbb34a91064c2cdc57b9fef66bba9a98ccd` is reviewed through Draft PR
  `https://github.com/wjstx0425/anomaly_xingtao_new/pull/1` against `main`.
- `BMW_EIGHT_VIEW_HANDOFF_README.md` is the primary Chinese transfer document. It covers the four-camera HDR capture,
  eight-view preparation and ROI, Label Studio review, final materialization, one-click Template/bright-streak/
  EfficientAD/YOLO training, model interpretation, offline/live Demo, asset migration, troubleshooting, and takeover
  checklist. The documented extra YOLO dependency is pinned to `ultralytics==8.4.89`.
- The branch intentionally includes the current eight-view entrypoints, required BMW and four-camera capture code,
  canonical `bmw_hdr_eight_view_v1` ROI/configs, and focused tests. It intentionally excludes datasets, results,
  checkpoints, customer images, Label Studio state, credentials, MVS SDK, the later unreferenced `bmw_right` ROI, and
  legacy standalone BMW commands.
- Fresh handoff verification passed 48 core eight-view tests plus 5 asset-independent Demo-model tests, seven CLI
  help checks, compileall, and a real-data one-click `--dry-run` preflight with eight views, 248 reviewed labels, and
  YOLO batch 32. Two Demo-model tests remain asset-dependent by design; the live camera was not retriggered during
  publication, and low-level camera tests were skipped in the sandbox because CUDA/NVML was unavailable.
- Follow-up commit `a08b18278fb70f9a258e6aa86d8f2cfc15ad2d7b` expands README section 12 with per-view Template
  thresholds/confusion counts, bright-streak calibration populations and continuity limits, per-view EfficientAD
  AUROC/F1 interpretation, YOLO split/box counts and metric meanings, plus the 25-check fusion-level false-reject
  caveat. The key handoff conclusion is that Template is conservative, no-streak detection has no final-test false
  accepts, interrupted-streak behavior lacks real negative samples, EfficientAD `back_right` has F1 0 at its current
  threshold, and YOLO recall 0.279 is unsuitable as a reliable defect backstop.

## BMW model-score visualization and decision-boundary audit (2026-08-10)

- Published commit `45e3909f56ae015002e3af004b70570cef0e9d70` on `agent/bmw-eight-view-handoff` adds
  `pipeline/bmw_lab_visualize_efficientad_scores.py`, dataset-level EfficientAD score analysis utilities/tests, and
  the corresponding README instructions. GitHub Draft PR #1 points to that exact commit. Generated customer-data
  reports remain local and ignored under
  `results/bmw_lab_one_click/bmw_lab_eight_view_v1/efficientad/score_analysis`.
- The real held-out report contains 183 image rows, one fixed-scale 0-1 distribution chart, eight hard-example sheets,
  and `report.json`. Metrics are recomputed from the exact runtime `predict` outputs used by the Demo. The original
  Anomalib `image_F1Score` is not authoritative because rerunning `Engine.test` emitted TorchMetrics warnings that
  `compute` was called before `update`; per-view AUROC was independently reproduced, but runtime F1/confusion counts
  from the new report supersede the logged F1 values.
- Runtime EfficientAD audit by view (AUROC/F1/normal FP/defect FN): front `.9625/.8000/2/0`, front_left
  `.8250/.8000/0/1`, front_right `.8333/.8000/0/1`, front_secondary `1.0000/.8000/0/1`, back
  `.9125/.5714/3/0`, back_left `.9750/.8000/1/0`, back_right `.7375/.6667/0/1`, and back_secondary
  `.8875/.6667/2/1`. Scores are Anomalib-normalized 0-1 and comparable within one view/model, not as a universal
  cross-view defect-severity scale.
- Template accepted none of the 31 visible-defect image rows in final test, but those rows represent only six distinct
  defective physical parts and three coarse types (`deform`, `edge`, `others`). This does not prove Template can detect
  every defect; it remains sensitive to alignment/illumination and can miss small local or texture anomalies.
- Bright-streak final test rejected all three `no_streak` parts but also rejected 4/17 normal parts (23.5% normal false
  reject). Two failures were continuity-related (`NG_BROKEN`), while two normal samples had presence metrics that
  overlap the no-streak population (`NG_NO_STREAK`). Therefore the current rule is conservative, but blindly lowering
  thresholds can admit real no-streak defects; improve the ROI/segmentation and collect real interrupted-streak
  negatives before retuning continuity thresholds.

## BMW right-side incremental ROI batch (2026-08-10)

- `pipeline/bmw_lab_prepare_eight_view_data.py` now accepts repeatable `--session-id` arguments. The filter is applied
  before manifest/image validation, rejects duplicate, malformed, or missing session IDs, and records the selected IDs
  in `report.json`. Use it for incremental releases instead of reprocessing older right-side sessions.
- The five new right-side sessions are `20260810_210030_527506`, `20260810_213407_600611`,
  `20260810_213852_756399`, `20260810_214505_826120`, and `20260810_214747_085800`. Their immutable prepared release
  is `dataset/bmw_lab_prepared/bmw_right_batch_20260810_21_v1`: 133 complete parts, 5 incomplete attempts excluded,
  and 1064 SHA-256-verified HDR images. Part counts are normal 83, deform 9, edge 18, others 3, and no_streak 20.
- The corresponding ROI release is `dataset/bmw_lab_training/bmw_right_batch_20260810_21_roi_v1`. It was generated
  from the exact user-selected config
  `dataset/bmw_lab_training/bmw_right_complete_roi_v1/roi_config.json` (SHA-256
  `0dab057714cd51eee937297550afdcfd8c3290440dbfe40086a9c320382c6a53`, fixed_setup/right). It contains 1064
  canonical PNG crops, 133 per view, no broken symlinks, and 240 visible-defect ROI images still pending YOLO labels.
- Focused verification after publication: all 9 tests in
  `tests/unit/bmw_inspection/lab/test_eight_view_dataset.py` passed. Both releases are no-overwrite versioned outputs;
  do not append later captures into them—select new session IDs and publish a new dataset/training ID.

## BMW worktree 合并前整理（2026-09-02）

- `agent/bmw-21only-diagnostics` 的 2026-08-20 至 2026-08-25 未提交 BMW 改动已作为待合并快照整理；`*.orig` 临时备份不纳入版本控制。
- 八视图运行时精简已删除 `pipeline/bmw_lab_snapshot_reproducibility.py`，因此同步删除其孤儿单元测试；旧 receipt/SHA/publisher 机制不恢复。
- `template40_outside05` 左右手候选配置已同步当前 rollback 配置的普通 Template 阈值及四个后视角 weighted 阈值，继续保证候选只改变四个前视角的 `outside_weight=0.5` 权重与其校准阈值。
- BMW 聚焦回归结果为 `560 passed`；另有 3 个仅由独立 worktree 路径和旧 worktree `pyproject.toml`/`uv.lock` 引起的合并前失败，需在合入主工作区后用主分支依赖文件复验。

## BMW 实时 GUI 完整检测周期计时（2026-08-25）

- 实时 GUI 现在用 `time.perf_counter_ns()` 记录从第一次空格键被接受、即将采集正面，到第一帧 RESULT 画面的 `cv2.imshow()` 返回的完整软件周期。该边界不声称包含显示器扫描或人眼感知时间。
- 每次实时检测在 `<result_root>/<capture_id>/cycle_timing.json` 保存 `front_capture_ms`、`flip_wait_ms`、`back_capture_ms`、正/反面推理、finalize/可信OK、原有结果保存、结果显示延迟、正面推理重叠时间与 `total_cycle_ms`。`total_cycle_ms` 只到第一帧 RESULT 返回；后台保存耗时另记 `persist_ms`。文件不含 SHA、receipt、provenance 或发布 schema。
- RESULT 状态卡显示完整周期总时间，并紧凑显示正/反采集、翻面、正/反推理、融合对比和保存时间。该计时只在真实相机 GUI 路径生成；离线 `--no-gui`/回放不伪造采集或翻面数值。
- 单 worker 时序已优化为：正面推理可与翻面/反面采集重叠，正/反推理仍串行；finalize 完成后先显示 RESULT，再由同一 worker 保存完整图片、证据、JSON和索引。保存期间禁止启动下一件，保存失败会明确进入 ERROR。未修改 Template、光痕、YOLO、EfficientAD、阈值、ROI、mask、HDR 参数或融合。
- 优化前左手现场 `bmw_demo_20260825_103344` 实测完整可见周期 `17701.495 ms`，其中保存 `5911.947 ms`。因此类似工况下本次调度优化预计把结果显示提前约 `5.9 s`，但不提高连续件吞吐；必须再拍一件读取新 `cycle_timing.json` 才是优化后现场实测。
- TDD 受控时钟回归验证了结果帧先于保存、可见周期由 `1120.0 ms` 降为 `970.0 ms`，并保留 `300.0 ms` 正面推理重叠；最终聚焦回归为 `50 passed`。

## BMW 右手0820可信OK图库更新（2026-08-21）

- 右手可信OK库已从旧的20260810图库切换为 `/home/yunjing/anomaly_xingtao_new/dataset/bmw_trusted_ok_reference/bmw_right_0820_train_normal_v1/reference_index.json`。新库使用右手0820 train/normal/OK 的 `group002/003/006/008/009/010/011/013` 八个完整零件，共64条八视图参考；全图直接引用 prepared manifest 原图，并按 `configs/bmw/rois/bmw_right_0820_v1.json` 生成64张 ROI。
- `configs/bmw/experiments/bmw_eight_view_demo_right_0820_mixed_v1.json` 已指向上述绝对路径。新库无SHA、receipt、publisher或provenance绑定；仍保留八视图完整、文件存在、图片可读、uint8、尺寸和ROI边界检查。`TrustedOkMatcher.preload()` 实测成功，相关聚焦回归12项通过。
- 右手 EfficientAD 阈值保持恢复后的原值 `0.5502086162570001/0.5500145435330002/0.549995529652`，本次没有修改任何阈值、模型、mask、ROI或融合规则。
- 加载旧右手可信库的 PID `666937/666950` 已TERM退出；加载新右手0820可信库的GUI于21:59启动，PID `849653/849666`，控制台确认可信OK预热完成和模型加载完成。

## BMW 右手0820 EfficientAD比例调整已回退（2026-08-21）

- 用户曾要求把左手相对放宽比例应用到右手，短暂得到 `front/front_left/front_right=0.6502465464855457/0.6111272705922224/0.639183993919892`；随后明确要求恢复原值。当前 `configs/bmw/experiments/bmw_eight_view_demo_right_0820_mixed_v1.json` 已恢复为 `0.5502086162570001/0.5500145435330002/0.549995529652`，阈值来源恢复 `legacy_reuse_for_0820_candidate`，验证状态恢复 `pending_independent_validation`。
- 其他五个 EfficientAD 阈值、Template、YOLO、光痕、ROI、mask和融合规则在调整及回退过程中均未修改。比例调整期的 `bmw_right_normal_group002_000001` 25项全PASS结果只作为历史实验记录，不能代表当前原阈值进程的现场检测。
- 比例阈值右手GUI PID `606915/606928` 已TERM退出；恢复原值后的右手GUI于21:27启动，PID `666937/666950`，控制台确认可信OK预热完成和模型加载完成。配置契约聚焦测试为5项通过；Qt字体目录提示不影响窗口启动。

## BMW 左手0820可信OK图库与EfficientAD现场误拒修正（2026-08-21）

- 新可信OK图库为 `/home/yunjing/anomaly_xingtao_new/dataset/bmw_trusted_ok_reference/bmw_left_0820_train_normal_v1/reference_index.json`。它从 left 0820 manifest 的 train/normal/OK 中固定选择 `group002/004/005/006/008/011/012/013` 八个完整零件，共64条八视图参考；全图直接引用原始 fused 图，按当前 `configs/bmw/rois/bmw_left_0820_v1.json` 生成64张 ROI。运行时 index 只保留 `physical_part_id/sample_id/view_id/full_image_path/roi_image_path`，没有 SHA、receipt、provenance 或 publisher。
- 生成器为 `pipeline/bmw_lab_build_trusted_ok_reference.py`，默认输出上述新目录。它保留文件存在、图片可读、uint8/尺寸、ROI边界、train normal OK及八视图完整检查。生成器与 matcher 聚焦回归为6项通过；真实图库64 references已通过 `TrustedOkMatcher.preload()`。
- 当前左手配置 `configs/bmw/experiments/bmw_eight_view_demo_left_0820_mixed_v1.json` 已切换到新图库。前三视角 EfficientAD 部署阈值从 `0.33/0.36/0.37` 放宽为 `front=0.39`、`front_left=0.40`、`front_right=0.43`；其余五视角不变。依据是8个训练正常件的最终组件分数上界 `0.318326/0.309631/0.342541`，以及4次“Template/YOLO/光痕均通过、仅EA误拒”的现场件上界 `0.362178/0.376671/0.406091`，再保留约0.02现场余量。标记为 `field_ok_envelope_20260821`，仍是实验室阈值而非独立验证结论。
- 已知小脚形变件的 EfficientAD 分数与现场正常分数重叠，不能同时靠单一EA阈值区分；该形变继续由已收紧的 Template 与新YOLO兜底。EfficientAD overlay 不改计算，但最终NG的 accepted component 现在用红框/红轮廓，rejected维持橙色，并直接写出 `P95/threshold/exceedance/area/reason`，避免把绿色宽连通域误解为真实缺陷框。
- 保存记录 `bmw_demo_20260821_201034` 原来只有 `front_right EfficientAD=0.403831/0.37` 为NG；用新配置CPU离线回放得到25 PASS / 0 NG / 0 ERROR，模型耗时 `4368.194 ms`。结果目录为 `results/bmw_lab_one_click/bmw_eight_view_demo_left_0820_mixed_v1/bmw-left-201034-new-ok-replay.n5ofHy/`，截图为同一结果根的 `replay_201034_new_ok_bank.png`。
- 19:54启动的旧进程 PID `135207/135210` 已TERM退出。新左手GUI于20:34启动，PID `367133/367146`，控制台已确认“可信OK参考库预热完成”“模型加载完成”；Qt字体目录提示仍不影响窗口启动。启动命令保持显式 left 0820 mixed config，不要使用默认右手配置。

## BMW 0820 左右手混合模型接入（2026-08-20）

- 新增互不覆盖的实验配置：`configs/bmw/experiments/bmw_eight_view_demo_right_0820_mixed_v1.json` 与 `bmw_eight_view_demo_left_0820_mixed_v1.json`。两者分别使用 `configs/bmw/rois/bmw_right_0820_v1.json`、`bmw_left_0820_v1.json`，直接加载各手 0820 run 下八个 `template/<view>/model.json` 和八个顶层 `efficientad/<view>/model.ckpt`；共享旧 YOLO `bmw_right_multisource_left_yolo_v1/yolo/train/weights/best.pt`，并保留各手旧版全图光痕规则与旋转四点 ROI。
- Template 判定继续使用配置中的旧部署阈值，但结果同时记录模型 JSON 内的 `model_threshold`、`deployment_threshold`、`risk` 和 `threshold_exceedance`。EfficientAD 结果显式记录 `threshold_source=legacy_reuse_for_0820_candidate` 与 `validation_status=pending_independent_validation`，不可将一次离线可运行解释为阈值已适配新 checkpoint。
- 同一视角仅裁剪一次 0820 公共 ROI，Template、YOLO、EfficientAD 收到同一个 NumPy 数组对象；光痕仍收到完整 `front_left` HDR 图。左右旧光痕四点均位于对应 0820 `front_left` 公共 ROI 范围内。左手 YOLO `front_secondary` 固定忽略框确认是旧 ROI 局部坐标，按全图位置不变从 `[1580,450,1756,800]` 转为新 ROI 局部 `[1505,628,1681,978]`，转换前后坐标均写入左配置说明。
- 操作者随后直接在 0820 ROI 上重画 EfficientAD mask：右手 `results/bmw_efficientad_manual_ignore_masks/bmw_right_0820_manual_ignore_v1/index.json`，左手 `.../bmw_left_0820_manual_ignore_v1/index.json`。两套混合配置的 `efficientad.ignore_mask_index` 已切换到对应新 index；八张 mask 均可读、仅含 `0/255`，尺寸逐视角精确匹配 0820 ROI，因此 EfficientAD 不再需要运行时缩放。右手 Template 仍独立使用旧 `bmw_right_manual_ignore_v3` 并按 `INTER_NEAREST` 适配，左手 Template 不使用 mask；左手 EfficientAD 组件过滤策略保持不变。此前记录的两次离线分数产生于切换前，尚未用新 mask 重放。
- 右手离线样本 `bmw_right_normal_group001_000001`：25 项为 20 PASS / 5 NG / 0 ERROR，最终 NG，模型耗时 `6142.605 ms`。8 个 EfficientAD 全 PASS，分数范围 `0.314678-0.409941`，旧阈值范围 `0.549995-0.550209`。结果目录为 `results/bmw_lab_one_click/bmw_eight_view_demo_right_0820_mixed_v1/bmw_right_normal_group001_000001/`，截图为根目录 `replay_right_group001.png`。
- 左手离线样本 `bmw_left_normal_group001_000001`：25 项为 22 PASS / 3 NG / 0 ERROR，最终 NG，模型耗时 `4994.287 ms`。8 个 EfficientAD 全 PASS，分数范围 `0-0.405828`，旧阈值范围 `0.507070-0.601900`。结果目录为 `results/bmw_lab_one_click/bmw_eight_view_demo_left_0820_mixed_v1/bmw_left_normal_group001_000001/`，截图为根目录 `replay_left_group001.png`。两侧各保存 24 张输入图、8 张 ROI、25 张分支证据；结果包含全部分数和阈值。
- 0820 checkpoint 的教师 mean/std 已写入，但 map quantiles 因 all-normal 训练未运行内部验证而仍为零；一次正常样本上分数与旧阈值处于同一 0.x 数量级且全部低于阈值，只能作为接入 smoke，仍待独立正常/缺陷集验证。当前环境 NVML/GPU 不可用，未验证 GPU 推理或四相机实时采集。聚焦回归为 `45 passed`。
- `pipeline/bmw_lab_select_bright_streak_rotated_roi.py` 已同步精简后的 `RotatedBrightStreakRoi` 接口：四点选择器只写 `points_xy/source_width/source_height/output_width/output_height`，不再传已删除的 `source_image/source_image_sha256`，也不再计算或打印 ROI SHA。修复回归为 `5 passed`；Qt 字体目录提示不影响 OpenCV 点击和保存。
- 操作者重新选择了左右手光痕四点 ROI，并已切入对应 0820 mixed 配置。右手 `results/bmw_bright_streak_rotated_roi/bmw_right_0820_manual_v1/roi.json` 点位为 `[(1838,1444),(1899,1437),(2010,2000),(1956,2017)]`；左手 `.../bmw_left_0820_manual_v1/roi.json` 点位为 `[(1929,1113),(1973,1117),(1956,1710),(1909,1707)]`。两者均为完整 4024×3036 `front_left` HDR 坐标，可矫正为 81×613，且四点位于对应 0820 `front_left` 公共 ROI 内。光痕 geometry 和判定阈值未改；此前离线光痕结果产生于旧 ROI，尚未用新 ROI 重放。切换聚焦验证为 `7 passed`。

## BMW 八视图实验室运行时精简（2026-08-20）

- 当前唯一活跃入口是 `pipeline/bmw_lab_eight_view_demo.py`。默认右手配置为 `configs/bmw/experiments/bmw_eight_view_demo_v6_right_normal_20260814_v1.json`；左手显式使用 `configs/bmw/experiments/bmw_eight_view_demo_left_normal_20260814_v1.json`。两套配置直接列出 Template 模型与阈值、光痕 geometry/thresholds/旋转 ROI、8 个 EfficientAD checkpoint 与阈值/mask/组件过滤、YOLO checkpoint/阈值、ROI、可信 OK 索引和结果目录。
- Demo 活跃链路不再计算或校验 SHA-256，不再读取 training-run receipt、composition、manifest identity、source release identity 或 publisher rebind 产物。修改阈值只改当前配置中的 `template.thresholds`、`bright_streak.thresholds`、`efficientad.thresholds`、`yolo.candidate_conf/final_threshold`；替换模型、ROI、mask 或可信 OK 索引只改对应路径。
- 检测计算仍是 8 Template + `front_left` 光痕 + 8 YOLO + 8 EfficientAD，共 25 项；所有分支都会执行，融合保持 `ERROR > NG > OK`。EfficientAD 仍保留手工 ignore mask 和左手 component policy，中文 UI、真实 YOLO 框、Template/EfficientAD 热力图、光痕中心线、可信 OK 对比及每次图片/结果保存均保留。
- 运行时仍检查配置/JSON/文件可读、八视角完整、ROI 边界、二值 mask 尺寸、模型加载、相机采集和推理异常转 ERROR。旧 V1-V5 Demo 配置、composition/rebind publisher、可信 OK 发布 CLI 和不可覆盖保存层已删除；历史结果与模型/数据资产未清理。
- uv 离线回放 `bmw_normal_group001_000001`（右手 V6）实测为 `OK`，25 PASS / 0 NG / 0 ERROR，模型检测耗时 `4832.444 ms`。结果保存在 `results/bmw_lab_one_click/bmw_eight_view_demo_v6_right_normal_20260814_v1/bmw_normal_group001_000001/`，截图为同一根目录下 `lab_simplify_group001.png`。本次环境无可用 GPU，未验证实时四相机采集。

## BMW 四算法卡片点击切换修复（2026-08-13）

- 修复提交 `c8f56ab0`。OpenCV 4.13 Qt 的 `setMouseCallback` 已经把窗口事件映射到原始 `1600x900` 图像坐标，入口旧逻辑又按 `getWindowImageRect` 做了一次缩放，导致右侧四张算法卡片的点击落到错误区域。现在直接使用 Qt 回调的图像像素坐标，同时修复八视角卡片和证据面板的同源偏移。
- TDD 回归覆盖 Template、光痕、YOLO、EfficientAD 四个卡片中心点；干净提交快照的 UI/V3 聚焦回归为 `183 passed`。
- 四相机重启时 `DA9805574` 返回独占占用 `0x80000203`，因此未声称真机运行成功。已改用保存记录 `211302` 启动离线 GUI 会话 `16688` 供现场点击验证。

## BMW 光痕 V3 与可点击 UI 真机整合（2026-08-12）

- 整合提交为 `37172148` 和 Qt 中文窗口兼容修复 `642a37c6`：四算法卡片、八视角卡片、证据详情页与光痕 V3 现在位于同一运行入口。
- OpenCV 4.13 Qt 后端无法用中文窗口名查找 `setMouseCallback` 句柄；运行时改用 ASCII 内部句柄 `BMW_EIGHT_VIEW_DEMO`，再用 `setWindowTitle` 显示全中文标题。
- 干净提交快照的 UI/V3 聚焦回归为 `182 passed`。真实保存记录 `bmw_demo_20260812_211302` 离线重放仍为光痕 `PASS/OK`，其余 24 项逐字段完全一致，SHA256 为 `03c50c877762f9bc1943ab7a0b4d23616bcbc858bd2f3886244cf2674d97e8e0`。
- 四相机 Demo 已在会话 `8809` 真实启动：模型和可信 OK 库预热完成，越过原鼠标回调崩溃点并保持在四相机主循环。未自动触发拍摄，因此不将本次启动声称为新零件检测结果。

## BMW 光痕 tracked-profile V3 最终验收（2026-08-12）

- 光痕 V3 已完成到 `1ca4f471`：固定全图 ROI `[1792,1180,1873,1793]`，在 `81x613` 区域内追踪斜向中心线，并用强/弱阈值、覆盖率、最长连续段、内部最大断点和断点数判定；旧 `raw_profile_v2` 仍保留用于回退。
- 正式 V3 产物是 `results/bmw_lab_one_click/bmw_right_batch_20260810_21_bright_v3_tracked_v8/report.json`，SHA256 为 `6b43690af67702333646fa7a88a2a5053a0afae6d19cb343bf6d3abb193c17d4`。Demo 配置只把光痕 engine/config/SHA 切到该产物；Template、YOLO、EfficientAD 与融合不属于 V3 变更。
- 干净提交快照的最终聚焦回归是 `181 passed`。保存的真实正常记录 `bmw_demo_20260812_211302` 离线结果为光痕 `PASS/OK`；24 个非光痕结果逐字段完全一致，序列 SHA256 为 `03c50c877762f9bc1943ab7a0b4d23616bcbc858bd2f3886244cf2674d97e8e0`。忽略的证据位于 `artifacts/bmw_bright_streak_tracked_v3_smoke/final_1ca4f471/`。
- V3 启动时失败关闭：重算算法、评估器、manifest、ROI 与全部评估产物哈希；强制固定 ROI、可容纳的 geometry、唯一确认现场样本 `211302=OK` 及其四个源文件哈希，并逐条核对 metrics、61 个 NPZ、8 个 no-streak、20 个 final-test 和 20 条 replay 的身份及语义一致性。
- 证据色彩：绿色是强响应，橙色是弱阈值桥接，红色只表示有效光痕范围内的内部断点，灰色是前后背景。不要把灰色背景解释成断续。
- 当前没有真实“有光痕但断续”的 NG 样本（`real_broken_samples=0`）；断续能力只有合成测试保障，不能声称真实断续召回率。补充真实断续样本后需要重新做独立验证。

## BMW clickable evidence UI handoff (2026-08-12)

- The completed UI is readable in `/home/yunjing/anomaly_xingtao_new/.worktrees/bmw-eight-view-handoff` on branch
  `agent/bmw-21only-diagnostics`. It adds clickable four-algorithm cards, clickable eight-view cards, evidence-panel
  detail navigation, and an in-window trusted-OK/current-evidence comparison page. `Esc` returns from detail to the
  dashboard, dashboard `Esc` is a no-op, and `Q` exits.
- UI-only commits, in cherry-pick order, are `31da1aab`, `8ae921ee`, `548c2439`, `bda9c434`, and `ee4ece91`. Do not
  cherry-pick the continuous range because algorithm commit `70ace398` is interleaved between the fourth and fifth UI
  commits. Every listed UI commit changes only the renderer/entrypoint and their two unit-test files.
- Template detail compares an aligned trusted reference against the selected Template model overlay; YOLO and
  EfficientAD compare the trusted ROI against the selected ROI overlay; bright-streak crops the same `roi_xyxy` from
  the trusted full image and rotates both sides 90 degrees clockwise. PASS details never expose a reference selected
  for another NG branch, and missing references never fall back to unapproved normal images.
- Fresh final UI verification was `25 passed` for the two focused test files; both implementation files compiled and
  `git diff --check` passed. This is code/headless verification only: no real OpenCV display, camera, or GPU run was
  claimed. The target environment uses OpenCV 4.13 Qt, whose mouse callback coordinates are local to the image viewport.

## BMW EfficientAD manual ignore masks (2026-08-12)

- User rejected automatic foreground segmentation and selected the minimum operator workflow: each of the eight saved
  ROI images may have zero or multiple polygons, and only those polygons are excluded from EfficientAD anomaly-map
  scoring. EfficientAD still receives the unchanged ROI image. Template, bright-streak, YOLO, HDR, and the public
  rectangular ROI remain unchanged.
- Selector: `pipeline/bmw_lab_select_efficientad_ignore_masks.py`. Controls are left-click add point, right-click/Enter
  close polygon, `S` save view, `N` no mask, `U` undo, `R` reset, and Esc cancel the whole no-overwrite publication.
  The asset contract in `src/bmw_inspection/lab/efficientad_ignore_mask.py` uses `0=inspect`, `255=ignore`, permits empty
  masks, and verifies source ROI, individual mask, index, and public ROI SHA values.
- Human-selected asset:
  `results/bmw_efficientad_manual_ignore_masks/bmw_right_manual_ignore_v1/index.json`, SHA-256
  `6c338107770030b550374cd66fbe21ac57b218d8e24f808ecf63756a5d263d83`. Active polygons are `back=1`
  (`2.3254%` ROI ignored), `back_left=1` (`2.2217%`), and `back_secondary=1` (`18.9240%`); the other five views have
  explicit empty masks and retain their V3 `pred_score` behavior.
- V4 lab config is `configs/bmw/experiments/bmw_eight_view_demo_v4_manual_ignore_mask.json`, SHA-256
  `88644f1dc40a7ccb36d201ca781c7fbc849f09e30fc6180060723d07896e0b09`. For the three active views, status uses the
  maximum returned anomaly-map value outside the manual mask; evidence retains the original `pred_score`, raw map max,
  score source, ignored pixel counts, and mask-index SHA. The heatmap and hotspot are also restricted to the inspect
  region. Empty-mask views remain byte-for-byte on the original scoring path.
- Saved-map A/B report:
  `results/bmw_efficientad_manual_ignore_ab/bmw_v3_vs_v4_representative_v1/report.json`. On 11 captures / 88 rows,
  33 rows used an active mask and only three decisions changed, all `back` NG to PASS: `203036` (`0.577622 ->
  0.337733`), `204352` (`0.573523 -> 0.381002`), and `204447` (`0.773533 -> 0.351794`). Trusted OK group002 stayed
  zero NG; strong multi-view samples `171815`, `173050`, and `205326` kept the same EfficientAD NG counts.
- V4 deliberately reuses the V3 numeric thresholds as a laboratory candidate. `pred_score` and returned map values have
  distinct Anomalib normalization contracts, so this is not calibrated production evidence. No new training or GPU
  live V4 run was performed. Focused verification is `79 passed`; V3 config loading remains compatible.

### Manual ignore-mask expansion v2 (2026-08-13)

- The selector now defaults to `--from-index .../bmw_right_manual_ignore_v1/index.json`, reconstructs every saved
  polygon, verifies exact equality with the SHA-bound v1 mask and source ROI, then preloads it for editing. The default
  no-overwrite output is `bmw_right_manual_ignore_v2`; `S` preserves seeded polygons, additions expand the union, and
  `R`/`N` deliberately clear the current view.
- Human-expanded asset:
  `results/bmw_efficientad_manual_ignore_masks/bmw_right_manual_ignore_v2/index.json`, SHA-256
  `50860b70316fad79e9ca2e19b555036fd2f882abce684a2fa53b543c53bc10bb`. All eight views now have masks, with
  polygon counts `front=3`, `front_left=2`, `front_right=3`, `front_secondary=2`, `back=4`, `back_left=4`,
  `back_right=3`, `back_secondary=2`. Ignored ROI fractions are respectively `25.0648%`, `42.9707%`, `30.5420%`,
  `24.9138%`, `26.7288%`, `42.8829%`, `26.7384%`, and `30.7070%`. A pixel-wise check proved v2 is a superset of
  v1 in every view with zero removed v1 pixels.
- Independent config:
  `configs/bmw/experiments/bmw_eight_view_demo_v4_manual_ignore_mask_v2.json`, SHA-256
  `b5afed34c289a99705753fd6f3be325a582f01578590725e5b5d1d6ddea814af`. It writes to the separate result root
  `results/bmw_eight_view_demo_v4_manual_ignore_mask_v2` and leaves v1/V3/V4-mask-v1 unchanged.
- Saved-map A/B:
  `results/bmw_efficientad_manual_ignore_ab/bmw_v3_vs_v4_manual_ignore_v2_representative_v1/report.json`, SHA-256
  `8119df96226fdfc4ccfb24907e4fcc931e233f7d18f5f73fbcbead2422276f2b`. On 11 captures / 88 rows, all 88 rows
  used manual masks and five EfficientAD decisions changed from NG to PASS: the three earlier `back` rows plus
  `170451/front_right` and `205326/front_secondary`. Fail-close remains for those two additions because `170451` has
  Template/front_right NG, while `205326` has all eight Template views NG and retains three EfficientAD NG views.
  Trusted OK group002 remains zero EfficientAD NG. This is still an uncalibrated saved-map laboratory candidate.
- Fresh focused verification after v2 expansion: `98 passed`; v2 config load, mask SHA binding, A/B count assertions,
  `py_compile`, and `git diff --check` passed. No new inference, training, camera run, commit, merge, or push occurred.

### Manual ignore-mask expansion v3 (2026-08-13)

- The operator continued from the SHA-verified v2 asset and published the no-overwrite v3 asset at
  `results/bmw_efficientad_manual_ignore_masks/bmw_right_manual_ignore_v3/index.json`, SHA-256
  `fa5cf8eb6ff9cfa9a9c6d187a9aff58c8101b51dd5d3cef3a2576a7745b1bc58`. Polygon counts are `front=5`,
  `front_left=4`, `front_right=5`, `front_secondary=4`, `back=6`, `back_left=5`, `back_right=5`, and
  `back_secondary=4`. Ignored ROI fractions are `25.4079%`, `43.8898%`, `30.8862%`, `25.2620%`, `27.0462%`,
  `43.2217%`, `27.2266%`, and `31.2031%`. Pixel-wise verification proved v3 is a strict superset of v2 in every
  view, with zero removed v2 pixels.
- Independent config is `configs/bmw/experiments/bmw_eight_view_demo_v4_manual_ignore_mask_v3.json`, SHA-256
  `71bfcc5ab31ae906af5dd804321d8ccb6430346a7c1fcee96882484fb61dc305`, with independent result root
  `results/bmw_eight_view_demo_v4_manual_ignore_mask_v3`.
- Saved-map A/B report is
  `results/bmw_efficientad_manual_ignore_ab/bmw_v3_vs_v4_manual_ignore_v3_representative_v1/report.json`, SHA-256
  `0e39e48e5959e80e7a040c30e008c22ea582a74593efbe8ae09d4414df063d7d`. It remains 11 captures / 88 rows / all
  88 rows masked / five EfficientAD NG-to-PASS changes, exactly the same changed identities as v2. The additional v3
  area produced no additional decision changes on saved maps. Focused verification is `98 passed`; config load, mask
  and report SHA binding, and `git diff --check` passed. This remains an uncalibrated, no-new-inference lab candidate.

## BMW trusted-OK reference comparison Task-5 handoff (2026-08-12)

- The only runtime reference release is ignored local data at `dataset/bmw_trusted_ok_reference/bmw_right_20260810_21_train_normal_approved_v2`: 50 user-confirmed complete OK parts, 400 indexed images (50 per canonical view), and 802 regular files in the release. Root checkout and the isolated BMW worktree both verified the same `reference_index.json` SHA-256 `ae7833ab35cbc76cbfef6cfa5163f77ef345d879a6e8e4834fa8cbfcf6023acc` and whitelist SHA-256 `15d9d86d8ffc7a28b55706b4cce2bd84e2ddd60ebdef7bd7b67d2725281745f5`.
- The v3 Demo preloads the SHA-bound reference bank once at startup; the operator message budgets about 27 seconds for this CPU-heavy one-time step. `O` toggles diagnostic comparison without rerunning models, while `N` and `P` cycle actionable NG/ERROR rows. Bright-streak evidence uses `front_left/full`; Template, YOLO, and EfficientAD use per-view `roi`. The reference score never changes any detector threshold, branch status, or final fusion.
- Task-5 real offline smoke reused saved HDR-pair record `results/bmw_eight_view_demo_v3_ng_evidence_v1/bmw_demo_20260812_170451`. The current v3 stack produced 25 checks and final `NG`: `template/front_right` and `efficientad/front_right` were NG. Both resolved to one SHA-verified `front_right/roi` reference (`bmw_right_normal_group077`, similarity `0.9715221524`, shift `(0, 0)`); there were no missing-reference diagnostics. Two real inference passes with the matcher disabled/enabled had exact equality for all 25 `(branch, view, status, score, threshold, reason)` tuples.
- Ignored smoke artifacts are under `artifacts/bmw_trusted_ok_reference_smoke/`: `bmw_demo_20260812_170451_trusted_ok_1600x900.png`, `verification_report.json`, and `inspection_records/bmw_demo_20260812_170451_trusted_on/inspection.json`. Persistence rechecked every saved reference image SHA and retained `reference_is_diagnostic_only=true`; do not commit these images or customer capture records.
- Focused BMW v3/trusted/UI/capture/persistence/Template/bright/YOLO-runtime/EfficientAD regression: 212 passed after excluding three legacy YOLO training-packaging assertions that require the optional installed/locked `ultralytics` distribution. Direct YOLO runtime/evidence coverage was 15 passed. Real smoke required `PYTHONPATH=/home/yunjing/ultralytics-c789`, whose source reports Ultralytics 8.4.89. NVML was unavailable and Lightning reported `GPU available: False`, so this proves offline CPU behavior only; it does not prove live four-camera opening, GPU speed, or camera hardware readiness.

## BMW trusted-OK approved reference release (2026-08-12)

- `pipeline/bmw_lab_publish_trusted_ok_reference.py` atomically publishes a no-overwrite reference release only from exact human `APPROVED` decisions. It binds the whitelist and every reference entry to the review decision, source, copied full-image, ROI-crop, and fixed ROI-config SHA-256 values.
- The user-confirmed 50-part release is `dataset/bmw_trusted_ok_reference/bmw_right_20260810_21_train_normal_approved_v1`: 400 full frames, 400 ROI PNGs, eight views with 50 references each. It is trusted-OK provenance only, not model-quality or production acceptance.
- P1 follow-up preserves v1 but retires it from future publisher input because its historical review package does not bind a candidate-manifest SHA. Use v2 only: `bmw_right_20260810_21_train_normal_v2` review and `bmw_right_20260810_21_train_normal_approved_v2` reference release. v2 is frozen to candidate SHA `9ff29f52bf0bb63636558832b8da0ffc7244809b2d5f056aa124ba444559ce8e` and was independently verified as 50 parts, 400 full images, 400 ROI images, and all source/copy/crop SHA matches.

## BMW v3 NG evidence laboratory Demo (2026-08-12)

- The independent v3 profile is `configs/bmw/experiments/bmw_eight_view_demo_v3_ng_evidence.json`; it does not replace
  the default Demo. Its immutable composition uses the 21:00 Template candidate, corrected `raw_profile_v2`
  bright-streak report, unchanged morning YOLO, and the 21:00 EfficientAD-v2 checkpoints.
- `pipeline/bmw_lab_prepare_v3_ng_evidence_demo.py` generates the no-overwrite composite run and a SHA-bound
  EfficientAD deployment asset. The base per-view thresholds stay recorded, while v3 adds a fixed experimental
  `threshold_margin=0.05`, producing deployment thresholds around `0.55`. This reduces false NG but has no independent
  defect-validation evidence, so it remains a lab-only recall-risk experiment.
- Every model result carries structured `details`. Template records similarity/risk, best translation and aligned mean
  absolute difference; bright-streak records presence/continuity/gap metrics; YOLO records real defect boxes; EfficientAD
  records base/deployment thresholds, margin, exceedance and anomaly hotspot. Template/EfficientAD heatmaps are
  diagnostic localization, not classified defect boxes.
- The Chinese 1600x900 experiment UI exposes short exposure, long exposure, fused HDR and selected evidence side by
  side. `N/P` cycles only NG/ERROR rows; `1-8` selects views and `T/L/Y/E` selects branches. Full untruncated reasons and
  threshold fields are shown in the right panel.
- Live capture persistence writes physical short/long/HDR frames. Offline prepared samples cannot recover historical
  source exposures and are explicitly saved as `source_kind=fused_only`, with the same fused image occupying the three
  storage slots; the UI hides the unavailable short/long panels instead of presenting duplicates as real exposures.
- Every `inspection.json` snapshots the capture-profile path/SHA, camera slot/serial/view mapping, configured HDR
  exposures, gain and fusion parameters. These are marked `configured_not_camera_readback`; ISO is not applicable to
  this industrial-camera contract, and aperture/focus/lamp output remain explicitly unrecorded physical controls.
- The selected bright-streak ROI evidence is rotated 90 degrees clockwise in the four-panel comparison so the narrow
  vertical ROI remains legible, matching the previously approved Demo convention.
- Fresh focused verification is `66 passed`; an offline CPU smoke on `bmw_right_normal_group001_000001` completed all
  25 checks with 21 PASS, 4 NG and 0 ERROR, wrote the evidence directory/JSON/index, and rendered the v3 dashboard.

## BMW v3 HDR source and inspection persistence (2026-08-12)

- `FourCameraHdrSession.capture_round()` still returns the existing fused-image mapping. It now also retains immutable
  `last_sources` entries for every captured semantic view: short image, long image, fused HDR, clip percentage, and
  HDR attempt. After front and back rounds the mapping follows canonical `VIEW_ORDER`.
- `bmw_inspection.lab.eight_view_demo_persistence.persist_inspection(config, inspection, source_images)` publishes one
  no-overwrite `<result_root>/<capture_id>` directory by staging then atomically renaming. It writes the three source
  images per view, ROI images and their statistics/SHA-256, non-null model overlays, `inspection.json`, and appends
  `inspection_index.csv` only after the directory is published.
- Live callers pass `camera.last_sources`. Offline callers must use `fused_only_sources(images)`, which deliberately
  records `source_kind="fused_only"` and stores the fused sample in all three image slots without presenting it as a
  physical HDR short/long pair.
- Persistence is covered by the current v3 focused verification gate; its earlier transient cross-task collection
  concern was resolved when the UI and entrypoint integration landed.

## BMW normal-only fast retraining entrypoint (2026-08-14)

- The operator corrected the capture identity to right-hand. The authoritative prepared release is
  `dataset/bmw_lab_prepared/bmw_right_normal_20260814_v1`; the earlier left-named release is obsolete and must not be
  used. It contains session
  `20260814_091058_791487`: 50 complete normal parts, 400 HDR fused images, split 30 train / 10 calibration /
  10 final-test; the incomplete group051 attempt is excluded.
- Use `pipeline/bmw_lab_retrain_normal_only.py` with `bmw_right_hdr_eight_view_v1.json` to derive a right fixed-setup
  ROI from existing coordinates, materialize a
  fresh ROI release, retrain eight Template and eight EfficientAD models, then calibrate EfficientAD thresholds from
  the actual normal-test part count. The former exactly-21-part calibration restriction is removed while the target
  whole-part FPR remains `1/21`.
- Historical data contributes only `no_streak` rows to the legacy bright-streak recalibration manifest. It never enters
  the new Template/EfficientAD training release. The stage order contains no YOLO stage and reports
  `yolo_trained=false`.
- This is an experimental candidate generator. It does not modify a Demo config and its legacy calibrated-rule
  bright-streak output is not a replacement for the V5 rotated tracked-profile V3 asset.
- The selective retrainer is resumable after a failed run: an identical derived ROI, matching published ROI training
  release, and identical combined bright-streak manifest are reused. A run directory is resumable only when its
  `run_report.json` status is `failed`; completed or unknown directories remain protected.

## BMW generic defect capture compatibility (2026-08-20)

- The 2026-08-20 left/right generic-defect sessions were captured under the legacy layout `defect/defect`.
- `bmw_inspection.lab.eight_view_dataset` maps that exact layout to the canonical prepared `source_class=others`.
  Existing `normal`, `deform`, `edge`, `others`, and `no_streak` meanings are unchanged, and source images/manifests are
  not moved or rewritten.
- Prepared releases created with `--skip-image-hash` intentionally contain blank `source_sha256` fields. The ROI
  representative selector skips source-content comparison only for those blank fields while retaining strict mismatch
  rejection whenever a hash is populated. The real `bmw_right_0820_v1` representative preflight selected eight
  4024x3036 views successfully after this compatibility fix.

## BMW right multisource models integrated into eight-view Demo (2026-08-11)

- The active BMW Demo profile in this worktree now binds the right-hand fixed ROI
  `configs/bmw/rois/bmw_right_hdr_eight_view_v1.json` and the complete training run
  `results/bmw_lab_one_click/bmw_right_multisource_left_yolo_v1`. This switches all eight Template models,
  all eight EfficientAD checkpoints, and the shared YOLO `best.pt` as one `training_run` contract.
- EfficientAD score analysis now supports the symlink-based multisource release: visible defect parts are completed
  from each named source release's `crops/<view>` directory, and the score CSV publishes an explicit
  `<source_release>::<part_id>` identity to prevent same-numbered right batches from being merged.
- The deployed EfficientAD threshold asset is
  `results/bmw_lab_one_click/bmw_right_multisource_left_yolo_v1/efficientad/score_analysis/part_thresholds.json`.
  The 41 branch-negative parts contain 33 true business-normal parts plus 8 no-streak business-NG parts. The
  tightened asset produces 1/41 branch-negative false NG (2.44%), 1/33 business-normal false NG (3.03%), and
  detects 7/7 visible-defect parts on the same selection data. It is explicitly demo-only, not independent
  acceptance evidence. The former `back_left=5e-324` threshold is now guarded at `0.001`.
- The threshold asset binds all eight EfficientAD checkpoint SHA-256 values, and the Demo profile also pins the
  threshold artifact SHA-256. Startup fails closed if a checkpoint or threshold file changes. Multisource score
  parsing now validates `<source_release>::<part_id>` against both the image path and filename and rejects mixed
  legacy/multisource defect layouts.
- Do not bind the raw new-training bright config directly to the ridge Demo detector. The root training run used an
  older detector implementation, while this worktree's Demo uses detector SHA-256
  `9726bd5cb6adb0a89d5dfc09ed51619d97a4d2bcdf6b355a40c15ed8af2ae2c0`. The compatible asset is
  `bright_streak_right_ridge_v1_bold/calibrated_config.json`; its current right-hand data performance remains weak
  (calibration balanced accuracy 0.682, final-test balanced accuracy 0.453), so the right-hand bright ROI/rule still
  needs a dedicated follow-up before acceptance.
- Offline integration smoke used `bmw_right_normal_group039_000001`: all 25 checks passed, final status `OK`,
  CPU-only inference took 13.314 s after strict asset verification, and the screenshot is
  `results/bmw_eight_view_demo/right_multisource_group039_smoke.png`. Use the root `.venv` from this worktree because
  the worktree-local minimal environment does not contain Ultralytics.

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
- Approved implementation plan: `docs/superpowers/plans/2026-07-13-zs32-multimodel-decision-fusion.md`; it splits delivery into backward-compatible dual-threshold evidence, strict per-view gates, staged audit records, stage-18 integration, offline stage-31 calibration, and final fault-injection/docs verification.
- Task 5 offline calibration implementation (2026-07-13):
  - Core module: `capture_data/fusion_calibration.py`; CLI: `pipeline/31_calibrate_zs32_fusion.py`.
  - Threshold groups are keyed by `(hand, view, branch, model_version, roi_version)`. `T_low` is the largest observed defect-score threshold satisfying inclusive `score >= T_low` target recall; `T_high` is `max(T_low, normal nearest-rank quantile)`.
  - `--fit-split` (default `calibration`) is the only split allowed to influence thresholds; `--eval-split` (default `test`) is held out for metrics. Missing selected splits fail closed, and a physical `part_id` still cannot cross splits.
  - `--required-view` is only a fit-data existence gate. Use repeatable exact `--required-group HAND:VIEW:BRANCH:MODEL_VERSION:ROI_VERSION` values for heterogeneous view-specific branch/version contracts; the implementation never invents a view/branch Cartesian product.
  - Groups missing either normal or defect calibration evidence write `status=insufficient_data` with both thresholds set to JSON/CSV null rather than inventing deployment values. Missing required evaluation evidence injects GRAY/REVIEW, sets `calibration_valid=false`, and suppresses recall/confidence-upper claims.
  - Metrics use physical `part_id` OR/GRAY semantics, so six CLEAR rows from one defect part count as one escape. A valid zero-escape result reports `1 - 0.05 ** (1 / defect_part_count)` as the exact one-sided 95% upper bound; `non_clear_recall` makes the GRAY-inclusive safety meaning explicit.
  - Stage 31 writes `thresholds.json`, `thresholds.csv`, `calibration_metrics.json`, and `calibration_summary.md` via same-directory atomic replace. It never edits `config/fusion/zs32_six_view.json`.
  - Verification environment: use `/home/yunjing/anomalib/.venv/bin/python -m pytest` from this worktree; the worktree-local `.venv` has no pytest and default `uv` cache is read-only. Task-5 focused calibration and wrapper verification reached `63 passed`, with new-file Ruff, `py_compile`, `--help`, and `git diff --check` passing.
- Task 6 strict-fusion documentation and fault verification (2026-07-13):
  - Implementation commits through Task 5 are `db2ea06d`, `623182a7`, `8a74bb67`, `6e90f273`, `73a64d31`, `fe7584c9`, `745530cc`, `70e412e3`, and `21d52822`. Operational entrypoints are stage 18 (`pipeline/18_fuse_inspection_results.py`) and stage 31 (`pipeline/31_calibrate_zs32_fusion.py`); deployed profile is `config/fusion/zs32_six_view.json`.
  - Stage 18 writes `branch_predictions.csv`, `fused_predictions.csv`, `summary.md`, and `audit/<part_id>.json`. Stage 31 writes `thresholds.json`, `thresholds.csv`, `calibration_metrics.json`, and `calibration_summary.md` without modifying the deployed profile.
  - Twelve table-driven stage-18 matrix cases cover missing view/branch, duplicate view/branch identity, split part IDs, quality FAIL, registration WARN, non-finite score, model/ROI version mismatch, missing source/evidence artifacts, and YOLO no-box plus a GRAY anomaly. `split_part_ids` specifically proves that rows split across two physical IDs become two incomplete groups and neither group is OK; a separate thirteenth fault test constructs front/back face decisions with different IDs and proves `combine_face_decisions` rejects the identity mismatch. The face-identity test was GREEN on its first run because the staged combiner already had the required guard; no RED was fabricated.
  - Initial matrix RED exposed four gaps: duplicate and version mismatch could return OK, while non-finite input aborted before audit. The final focused command `uv run pytest tests/unit/pipeline/test_fuse_inspection_results.py -k fault -v` reports exactly `13 passed, 7 deselected`; every matrix audit keeps `released_status=null`.
  - Complete target verification command uses the five planned test files and reports `141 passed`. `py_compile` for all five production modules, JSON validation, stage-18/stage-31 `--help`, task-file Ruff, task-file format check, and `git diff --check` pass.
  - Environment update: `uv 0.11.16` and the worktree `.venv` are currently usable; both `uv run` and `uv run --no-sync` resolve to the worktree environment. The plan-wide Ruff check still reports 60 pre-existing style findings, and the plan-wide format check reports two previously unformatted Task1-3 files (`capture_data/fusion_engine.py`, `tests/unit/capture_data/test_fusion_engine.py`); Task6-owned changed test/audit files are clean and formatted, and existing debt was left untouched.

### ZS32 final review closure (2026-07-13)

- This section supersedes the earlier Task 5/6 verification counts and incomplete strict-contract description.
- Strict identity is now `product=ZS32`, `profile=zs32_six_view_v1`, one consistent `hand` selected from `left/right`, `side=zs32`, exact `view`, and exact `branch`. Normalized CSV and audit records retain `product`, `profile`, `hand`, `source_hash`, and `manifest_identity`.
- `config/fusion/zs32_six_view.json` contains 72 non-wildcard `expected_versions` records: 2 hands x 6 views x 6 required branches, including `template_match`. Every `(hand,side,view,branch)` requires exact model, threshold, ROI, and template versions; missing or mismatched values fail closed.
- Reusing a source path, declared source hash, manifest identity, or computed image hash across required views marks the inspection incomplete and blocks release. A valid STRONG `NG_*` remains the immutable machine result; system/capture/version/evidence faults are retained as additional triggers and keep `inspection_complete=false` rather than downgrading machine NG.
- Non-strict legacy C789 `SUSPECT` remains exactly `SUSPECT`; only the strict ZS32 profile maps ambiguous evidence to `REVIEW`.
- Stage 18 always requires strict evidence artifacts, rejects an existing final output directory, stages CSV/summary/all audits in one sibling directory, and publishes the complete generation with one rename. Hash, CSV, audit, or rename failure leaves neither final nor staging output. Malformed strict thresholds/evidence publish a non-releasable diagnostic generation before the CLI exits nonzero.
- Audit schema `2.0` retains product/profile/hand/session/timestamp, `inspection_complete`, computed evidence class, normalized risk, source/evidence SHA-256, both margins, all versions and triggers, while keeping `machine_status`, `review_status`, and `released_status` separate.
- Stage 31 reads the same strict profile as stage 18, automatically requires all 72 exact calibration groups, and embeds the profile SHA-256, full expected-version contract, threshold-version set, and canonical threshold-record SHA-256 in `thresholds.json`. Invalid calibration nulls all safety performance rates; raw observed rates are available only as `diagnostic_observed_*`.
- Final-fix commits before integration are `5b38807c` (invalid calibration rates), `c78b24d0` (strict identity/version/source contract), and `a0e4810d` (atomic diagnostic audit generations).
- Fresh final-fix verification before the integration commit: the five directly related files report `178 passed`; legacy traditional-operator and multiview-manifest compatibility adds `11 passed`. Stage-18 and stage-31 help, changed-module `py_compile`, profile JSON validation, formatting, and `git diff --check` are required final gates.

### ZS32 locked threshold and structured evidence closure (2026-07-13, round 2)

- This section supersedes the preceding final-review closure where threshold artifacts, capture session/group identity, gate triggers, or YOLO detections are concerned.
- Strict stage 18 now requires `--threshold-artifact <stage31-output>/thresholds.json`. The bundle must have schema/version `anomalib.zs32_fusion_thresholds/1.0`, `calibration_valid=true`, the exact current profile/config SHA-256, all 72 required groups, 72 unique deployable `status=ok` threshold records, a valid canonical record hash, and a valid artifact hash calculated over the canonical payload without its own hash field.
- Every strict branch CSV `low_threshold/high_threshold` must exactly match the unique locked record selected by `(hand,view,branch,model_version,roi_version)`. Missing, tampered, incomplete, invalid-calibration, profile-mismatched, or numerically inconsistent bundles publish a non-releasable diagnostic generation and then exit nonzero.
- Generation-level `threshold_artifact.json` and every per-part audit retain the artifact path, source-file SHA-256, immutable artifact SHA-256 and threshold-record SHA-256.
- `capture_session` and `group_id` are normalized CSV/audit identity fields. Every strict row requires both values and all rows for one physical part must agree; missing or mixed identities are `INVALID_CAPTURE`. An absent acquisition timestamp remains `null` and is never replaced with the audit write time.
- All quality/registration non-PASS rows are retained as separate triggers. Valid STRONG evidence remains the machine `NG_*`; gate/system evidence forces `inspection_complete=false` and keeps release null.
- YOLO uses one summary branch row per part/view with a JSON `detections` list. Each box preserves class, confidence, `xyxy`, area and supplied ROI/border flags. Multiple boxes never count as duplicate required branch identities; no-box evidence is the explicit empty list `[]`.
- Calibration coverage invalidation uses one helper to null escape, recall, normal-reject, review and confidence-bound safety rates while keeping raw observations only as `diagnostic_observed_*`.
- Round-2 implementation commits before integration are `478adf79` (capture identity, gates and YOLO evidence) and `72a64d79` (locked threshold artifact).
- Fresh round-2 directly related gate after cross-layer integration: `199 passed, 3 warnings`; legacy traditional-operator/multiview-manifest gate is `11 passed, 3 warnings`. A broader `tests/unit/capture_data` diagnostic run reports `288 passed, 2 failed`; the two failures are pre-existing unrelated stale expectations in multicamera right-hand parser policy and demo face-name console wording, not fusion files or behavior. Format, core Ruff `E4/E7/E9/F/I`, production `py_compile`, profile JSON, both CLI help commands, and `git diff --check` pass; the broad project Ruff policy retains its existing 58 whole-file findings.

### ZS32 explicit gate and YOLO evidence closure (2026-07-13, round 3)

- Strict ZS32 quality/registration rows require an explicit nonblank `status=PASS`. Missing, blank, WARN, FAIL, or any other value is a gate fault; without STRONG it blocks OK, and with STRONG it preserves the machine NG while adding the gate trigger and forcing `inspection_complete=false`. Legacy non-strict rows may still use `pred_label=0` without an explicit status.
- `BranchPrediction.detections=None` means the YOLO summary is missing; an explicit empty tuple/list means a present no-box summary. CSV missing/blank cells remain `None`, while JSON `[]` remains an explicit CLEAR summary. Strict ZS32 rejects missing summaries.
- Every strict YOLO box requires class, confidence, `xyxy`, and area. Confidence must be finite in `[0,1]`, coordinates must be four finite nonnegative increasing values, area must be positive and finite, and supplied ROI/border flags must be boolean. Missing or malformed evidence is published only as a fail-closed diagnostic generation.
- Legacy near-threshold policy and `surface_texture status=SUSPECT` both retain `final_status=SUSPECT` with `final_label=None`; only the strict ZS32 profile maps GRAY evidence to REVIEW.
- Fresh round-3 unified gate is `222 passed, 4 warnings`: 211 directly related fusion/audit/calibration/stage-18/wrapper tests plus 11 traditional-operator/multiview-manifest compatibility tests. All 10 files pass formatting and core Ruff `E4/E7/E9/F/I`; production `py_compile`, profile JSON validation, both CLI help commands, and `git diff --check` pass. Broad project Ruff remains at the prior 58 whole-file findings after cleaning three new policy findings from this round.

### ZS32 template-first inspection gate (2026-07-13)

- `template_match` is now the first product-detection branch for every left/right ZS32 view. It uses whole-view fixed-ROI grayscale matching, not the C789 slot prototype.
- Execution order is identity/image prerequisites, six-view template gate, then expensive PatchCore/YOLO/geometry only after six explicit PASS results. REVIEW, NG_TEMPLATE, malformed continuous evidence, or adapter exceptions short-circuit all downstream calls.
- Template risk is `1-similarity`; `risk<T_low` is PASS, `T_low<=risk<T_high` is REVIEW, and `risk>=T_high` is NG_TEMPLATE. Template-fit normal parts are disjoint from threshold-normal parts.
- Template evidence retains the raw similarity/risk, both thresholds, best template path/hash, offset, image hash, and model/threshold/ROI/template versions. Template files are hash-verified at inference.
- Template training locks `model.json` with `model.sha256`, rejects template paths outside the model directory, and exports same-source `calibration_rows.csv` using Stage 31's nearest-rank policy. Stage 18 retains and verifies `evidence_hash` for the selected template.
- A template result cannot PASS without all four versions, an existing best-template path, and a matching template SHA-256. Downstream `OK` is accepted only with `inspection_complete=true`; missing/unknown downstream state becomes REVIEW.
- Strict Stage 18 requires valid 64-hex declared source and evidence hashes for every required branch row and verifies both against the files; an omitted hash cannot bypass the orchestrator's template evidence contract.
- Template short-circuit audits initialize `review.status=PENDING`, keep `review_status=null`, and keep `released_status=null`, matching the full Stage 18 audit lifecycle even when Stage 18 is intentionally skipped.
- Strict profile required branches increased from five to six per view and the exact versioned calibration contract increased from 60 to 72 groups. Stage 18 derives the expected count from the profile instead of hard-coding it.
- `ZS32InspectionOrchestrator` passes all six normalized template results to its downstream runner. It records early-stop/skipped state without inventing missing branch CLEAR evidence.
- Local integration on 2026-07-13 merged `feat/zs32/multimodel-fusion` into `main` after first committing the completed PatchCore ROI workflow and C789 matcher prototype. Backup branch: `backup/main-before-zs32-fusion-20260713`; no GitHub push was performed.
- The merged PatchCore + fusion target suite reports `333 passed, 1 warning`; production/core-import Ruff, Python compilation, shell syntax, profile JSON, and all five relevant CLI help commands pass.

## BMW eight-view bright-streak and EfficientAD Demo thresholds (2026-08-10)

- Branch/worktree: `agent/bmw-eight-view-handoff` in `.worktrees/bmw-eight-view-handoff`; the public Draft PR is `wjstx0425/anomaly_xingtao_new#1`. Never commit `dataset/`, `results/`, customer images, checkpoints, or the local asset symlinks.
- Bright-streak detection in `src/bmw_inspection/detector.py` follows a row-wise center ridge, allowing gradual local drift and width changes instead of requiring one whole connected component. The latest audit tracks candidate-row adjacency separately from the last accepted ridge: rejected lateral-jump rows keep the candidate run active, while only a real candidate gap permits reacquisition and continuity-gap accounting. It keeps the existing fixed ROI, presence, continuity, and evidence contracts.
- Standalone recalibration is `pipeline/bmw_lab_recalibrate_bright_streak.py`. Default mode fits only `split=calibration`; explicit `--bold-continuity` keeps SNR/coverage presence fitting calibration-only but uses calibration plus final-test normals for the three continuity envelopes. It is strictly boolean and records `demo_only=true` and detailed split/leakage flags.
- Real bold artifact: `results/bmw_lab_one_click/bmw_lab_eight_view_v1/bright_streak_ridge_v4_bold/`. Detector SHA-256 is `9726bd5cb6adb0a89d5dfc09ed51619d97a4d2bcdf6b355a40c15ed8af2ae2c0`. Thresholds are SNR `3.2675675643`, coverage `0.2536704731`, longest-run `0.2202283850`, max-gap ratio `0.3931484502`, max-gap count `12`. Current-data result is 20/20 calibration and 20/20 final rows correct, including 17/17 normal streaks OK and 3/3 no-streak samples `NG_NO_STREAK`; this is demo tuning, not independent validation. There are still no real broken-but-present streak negatives.
- Complete EfficientAD scoring uses 20 normal physical parts and 6 defect physical parts, each with all eight views: 208 rows in `efficientad/score_analysis/efficientad_scores.csv`. The defect set is the union of per-view visible-defect part IDs, completed from the canonical per-view ROI crops.
- Whole-part threshold CLI is `pipeline/bmw_lab_calibrate_efficientad_thresholds.py`. Current thresholds are front `0.680344`, front_left `0.009403`, front_right `0.486333`, front_secondary `0.111191`, back `1.0000000000000002`, back_left `0.457497`, back_right `0.230414`, back_secondary `0.575206`. The `back > 1` threshold intentionally disables a noisy view.
- Current selected-data EfficientAD result is 1/20 normal whole-part false NG (`5%`, `bmw_normal_group051/back_left`) and 6/6 defect whole-parts detected with 17 defect-view hits. The artifact is explicitly `demo_only=true` and `test_used_for_selection=true`; it must be validated on new untouched physical parts before any acceptance claim.
- Demo configuration requires explicit `bright_streak.config` and `efficientad.threshold_artifact` assets and fails closed on missing/invalid files. Runtime EfficientAD status is solely `score >= per-view threshold`; checkpoint `pred_label` is diagnostic only. Heatmaps use a fixed 0-1 scale.
- Fresh final offline smoke `bmw_normal_group072_000001` produced 25 PASS, final OK, 3385.422 ms with RTX 4090 CUDA inference. Screenshot: `artifacts/bmw_eight_view_threshold_demo/group072_bold_efficientad_5pct.png`; this single-sample smoke is not a throughput benchmark.

## BMW current algorithm backup (2026-08-11)

- GitHub backup target is branch `agent/bmw-eight-view-handoff`; the snapshot includes BMW code, configs, pipeline entrypoints, and tests but excludes `dataset/`, `results/`, images, and model checkpoints.
- Keep the latest right-hand training entrypoints in the backup: `pipeline/bmw_lab_train_right.py`, `pipeline/bmw_lab_train_right_multisource.py`, `src/bmw_inspection/lab/right_train_all.py`, and `src/bmw_inspection/lab/multisource_training_data.py`, together with their unit tests.
- Local recoverable backup is a complete Git bundle under `/home/yunjing/anomaly_xingtao_new/local_backups/`; verify it with `git bundle verify` and use its recorded SHA-256 before restoration.

## BMW 21:00 EfficientAD-only diagnostic entrypoint (2026-08-11)

- Use `pipeline/bmw_lab_train_efficientad_only.py` for the published `bmw_right_batch_20260810_21_roi_v1` ROI release. It runs only EfficientAD-S with 30 epochs, batch 1, 256x256, GPU 0, and seed 42; it then scores the 21 `normal_test` physical parts and fits a normal-only whole-part threshold asset with at most 1/21 false NG. It neither materializes data nor invokes Template, bright-streak, or YOLO stages.
- The release has pending YOLO labels (`yolo_training_ready=false`), which must not block this diagnostic path. The isolated preflight validates only all eight EfficientAD `normal` and `normal_test` directories.
- The candidate threshold asset is `efficientad/score_analysis/part_thresholds.json`, explicitly marked `normal_test_only`, `candidate_only`, and `defect_metrics=not_evaluated`; it does not update the default Demo. The default candidate output is `results/bmw_lab_one_click/bmw_right_batch_20260810_21_efficientad_v1`; a non-dry run rejects any existing directory or symlink, so choose a new `--run-id` for every retry. `--dry-run` is safe for layout and plan verification and does not start GPU training.
- The real 2026-08-11 run completed all eight 30-epoch checkpoints and 168 `normal_test` scores. `part_thresholds.json` reports 21 normal parts, 0 false NG, observed whole-part FPR 0.0, and an allowed budget of 1/21. Per-view thresholds are approximately 0.5 after each checkpoint's EfficientAD post-processing. There are no defect rows, so defect recall/AUROC remain unverified and this candidate must not replace the Demo merely from the normal-FPR result.

## BMW 21:00 raw-profile bright-streak v2 diagnostic (2026-08-11)

- `src/bmw_inspection/lab/bright_streak_raw_profile.py` is an offline-only v2 candidate. It calculates a median-smoothed, per-row centre-band minus left/right-background contrast profile, then publishes coverage, longest run, internal max gap, and gap count.
- Calibration-only class-mean inspection proved the morning base ROI `[1872,1180,1953,1793]` is shifted about 80 pixels right of the 21:00 streak. The corrected candidate ROI is `[1792,1180,1873,1793]`; the base/current detector still uses its unchanged ROI for the A/B comparison.
- `pipeline/bmw_lab_evaluate_bright_streak_raw_profile.py` defaults to the corrected ROI and immutable output `results/bmw_lab_one_click/bmw_right_batch_20260810_21_bright_v2_roi_corrected`. It fits thresholds only from calibration rows, compares v2 and the unchanged current detector only on final_test, and rejects an already-existing output directory.
- Each evaluated sample writes `profiles/profile_<manifest-row>.npz` with `row_scores` and `mask`; `metrics.csv` links that artifact and records all continuity statistics. The code does not alter `src/bmw_inspection/detector.py` or the default Demo configuration.
- Real final-test A/B on 20 held-out rows: corrected raw-profile v2 is 19/20 (95%), with 15/16 normal accepted and 4/4 no-streak detected; the unchanged current algorithm is 10/20 (50%). No real broken-but-present samples exist, so interrupted-streak recall remains unverified even though synthetic continuity tests pass.

## BMW tracked-profile bright-streak v3 design (2026-08-12)

- The accepted v3 design is `docs/superpowers/specs/2026-08-12-bmw-bright-streak-tracked-profile-v3-design.md`. It replaces v2's fixed central vertical band with a smooth slanted-path tracker inside the same `81x613` ROI, then applies strong/weak hysteresis along that path while retaining coverage/run/gap evidence.
- Execute it from `docs/superpowers/plans/2026-08-12-bmw-bright-streak-tracked-profile-v3.md`: core tracked profile, immutable evaluator/replay, Demo integration, then offline acceptance and live restart. Each stage is TDD and independently reviewed.
- The immediate accepted现场 normal `bmw_demo_20260812_211302` has sufficient presence and run evidence; v2 rejects it only because five rows scoring roughly `88.3-90.2` fall just below the single `92.43` row threshold. This is treated as a geometry/segmentation false break, not solved by globally relaxing the final gap limit.
- Scope is bright streak only. Template, YOLO, EfficientAD, 25-row result structure, and whole-part fusion must stay unchanged. Existing v2 artifacts remain rollback assets.
- Current labeled defects include complete `no_streak` only; no real broken-but-present samples exist. v3 must retain all no-streak rejects and synthetic broken-path checks, but interrupted-streak recall remains explicitly unverified until real samples are added.

## BMW 21:00 Template fixed-threshold diagnostic result (2026-08-11)

- The isolated candidate changed only the five selected templates per view and reused the morning numeric thresholds, 512x512 preprocessing, and max shift 12. It used only 21:00 `train/normal` rows and did not update the Demo.
- Whole-part normal pass rate is 13/21 (61.9%) on calibration and 8/20 (40.0%) on final_test. The worst final-test false-reject views are `front_right` (8/20) and `front_secondary` (5/20). Therefore replacing afternoon templates with 21:00 templates alone does not solve Template instability; threshold/registration sensitivity remains the main follow-up.

## BMW 21:00 Template-only live Demo (2026-08-12)

- Use `pipeline/bmw_lab_prepare_template_21only_demo.py` to create the immutable composite run `results/bmw_lab_one_click/bmw_right_batch_20260810_21_template_demo_v1`. Its `template` link targets the 21:00 candidate; `efficientad` and `yolo` target the morning baseline run. Existing output is always rejected.
- Start the single-variable live test with `configs/bmw/experiments/bmw_eight_view_demo_template_21only_v1.json`. It keeps the morning bright-streak config, EfficientAD threshold asset/checkpoints, YOLO parameters, ROI, and HDR capture config; only `demo_id`, `training_run`, and `result_root` differ from the default config.
- The worktree virtual environment lacks `ultralytics`; run the worktree script through the root workspace uv environment. Offline normal `bmw_right_normal_group001_000001` loaded all models and completed 25 checks: 21 PASS and 4 NG. Candidate Template rejected `back` and `back_left`; unchanged morning EfficientAD also rejected those two views. Exit 1 is the business NG result, not a runtime failure.

## BMW 21:00 EfficientAD + raw-profile bright-streak v2 live Demo (2026-08-12)

- Use `pipeline/bmw_lab_prepare_efficientad_bright_v2_demo.py` to create the immutable composite run `results/bmw_lab_one_click/bmw_right_batch_20260810_21_efficientad_bright_v2_demo_v1`: baseline Template/YOLO and candidate `bmw_right_batch_20260810_21_efficientad_v1/efficientad`.
- Start the second live experiment with `configs/bmw/experiments/bmw_eight_view_demo_efficientad_bright_v2_v1.json`. It keeps baseline HDR, part ROI, Template, YOLO and YOLO thresholds, while switching EfficientAD checkpoints/threshold asset and bright-streak to SHA-bound `raw_profile_v2` report.
- `EightViewRawProfileBrightStreakPredictor` consumes the full `front_left` image, crops corrected ROI `[1792,1180,1873,1793]`, calculates raw grayscale local row contrast, and overlays coverage/continuity evidence. The default Demo remains on `calibrated_rule_v1`.
- EfficientAD threshold assets may serialize view keys alphabetically; the Demo now checks the exact eight-view key set and normalizes to canonical `VIEW_ORDER`, while retaining threshold-asset and per-checkpoint SHA checks.
- GPU offline check on candidate-calibration part `bmw_right_no_streak_group002_000001` completed all 25 calls with zero EfficientAD NG; bright-streak correctly returned no-streak. `bmw_right_normal_group001_000001`, which was not in the 21-part EfficientAD threshold set, produced EfficientAD NG on back/back_left and must not be used to claim a runtime score-scale mismatch.

## BMW trusted-OK human review candidates (2026-08-12)

- `src/bmw_inspection/lab/trusted_ok_reference.py` is the only Task-1 producer for candidate reference parts. It accepts only one requested session's `source_class=normal`, `business_label=OK`, `split=train` rows, requires an exact complete canonical eight-view group, rehashes every selected regular source file, and rejects malformed candidate identities or hashes before any publication.
- `pipeline/bmw_lab_prepare_trusted_ok_review.py --session-id <capture-session>` defaults to the ignored, no-overwrite frozen v2 review package. It creates `candidate_manifest.csv`, exactly one `PENDING` decision per physical part, Chinese-labelled 4x2 contact sheets, and `review_package.json` with `automatic_approvals: 0`; it can never approve a candidate automatically.
- The future candidate is `dataset/bmw_trusted_ok_review/bmw_right_20260810_21_train_normal_v2`, generated from session `20260810_210030_527506`: 50 candidate parts / 400 images / 50 contact sheets. Historical v1 is preserved only for audit and must not be regenerated or published; neither review package itself is model acceptance.

## BMW trusted-OK runtime matching (2026-08-12)

- `TrustedOkMatcher` accepts only the exact ignored release identity `bmw_right_20260810_21_train_normal_approved_v2` and requires a caller-supplied expected index SHA-256. It strictly validates the frozen index/whitelist schemas, whitelist digest, candidate/ROI provenance, fixed session/train/normal/OK semantics, 50 approved complete parts, 50 references per view, 400 rows total, and `source_sha256 == full_image_sha256` before use.
- `build_model_suite()` deliberately leaves trusted-OK matching disabled and preserves legacy config behavior/empty metadata. A later explicitly SHA-bound configuration task must construct, preload, and inject the matcher. When injected, `EightViewModelSuite` matches only after all 25 model rows are fused, once per unique NG/ERROR view; `comparison_mode="full"` is explicit for actionable `front_left` bright streak and `"roi"` for the other model evidence.
- `TrustedOkMatcher.preload()` prepares all eight ROI banks plus the `front_left` full-image bank; `match()` refuses an unpreloaded mode rather than doing first-use bank I/O during inspection. Real v2 serial preload prepared 450 references/9 keys in about 27.571 seconds, and the idempotent second preload took about 9 microseconds with no file loads.
- Trusted matches and errors are immutable diagnostics on `EightViewInspection`; matcher failure never edits a branch result or changes `final_status`. The strongest-difference red contour is diagnostic only, not validated defect segmentation.

## BMW trusted-OK Demo integration (2026-08-12)

- `bmw_eight_view_demo_v3_ng_evidence.json` explicitly pins the approved-v2 `reference_index.json` with SHA-256 `ae7833ab35cbc76cbfef6cfa5163f77ef345d879a6e8e4834fa8cbfcf6023acc`. Older v1/v2 configs omit the optional block and keep trusted-reference diagnostics disabled.
- `build_model_suite()` constructs and preloads the matcher only for an explicit trusted-reference config. Startup prints a Chinese approximately-27-second preload status; any index, whitelist, image, or preload failure is retained as a diagnostic error and disables only trusted comparison, never the four detector branches.
- Matcher initialization also cross-checks the SHA-256 of the resolved current Demo `roi_config` file against the digest validated from the trusted index and whitelist. ROI drift disables only trusted comparison with a Chinese diagnostic initialization error; detector models, 25 rows, and final fusion remain unchanged.
- Trusted matches are keyed by `(view, comparison_mode)` rather than view alone. `front_left/full` belongs only to bright-streak evidence, while Template/YOLO/EfficientAD use `<view>/roi`; simultaneous front-left light-streak and appearance NG therefore retain two independent matches.
- After an actionable inspection, `O` toggles trusted-reference comparison without rerunning models; `N`/`P` still navigate the immutable NG/ERROR rows. The lower area shows `现场NG`, `可信OK`, `对齐差异`, and current model evidence only for the selected NG/ERROR row. A selected PASS row explicitly says `当前项目通过，无需NG对比`; a missing ROI match uses the retained part ROI, never the full image as an ROI substitute.
- Capture persistence writes each selected reference full image, ROI, aligned region, and difference under `references/<view>/<mode>/`, plus source/full/ROI/index/whitelist and saved-file hashes with `reference_is_diagnostic_only=true`. Legacy records with no reference diagnostics retain their former metadata shape.

## BMW tracked-profile bright-streak v3 Demo integration (2026-08-12)

- The isolated v3 Demo now selects `tracked_profile_v3` through exactly the three `bright_streak` fields in
  `configs/bmw/experiments/bmw_eight_view_demo_v3_ng_evidence.json`. Its ignored immutable report is
  `results/bmw_lab_one_click/bmw_right_batch_20260810_21_bright_v3_tracked_v8/report.json`, pinned by SHA-256
  `6b43690af67702333646fa7a88a2a5053a0afae6d19cb343bf6d3abb193c17d4`. Roll back only those three fields to
  engine `raw_profile_v2`, report `bmw_right_batch_20260810_21_bright_v2_roi_corrected/report.json`, SHA-256
  `24250c00b8518e3c886673815b696f0a4b6cfd5ffdb0fb1dc7d2fd8a5799268d`.
- The fixed full-image ROI remains `[1792,1180,1873,1793]` (`81x613`). Selected geometry is candidate width 5,
  background width 10, gap 3, median window 5, max step 2, and step penalty 1.0. Thresholds are strong
  `136.85`, weak `128.20`, minimum coverage `0.0848287113`, minimum longest run `0.0440456770`, maximum gap
  `0.1060358891`, and maximum gap count 2.
- The predictor validates the report identity, current algorithm/evaluator source hashes, and every metrics/replay/NPZ
  inventory hash before use. Evidence colors the diagnostic tracked centreline green for strong rows, orange for weak
  bridged rows, and red for gaps; this is path evidence, not defect segmentation truth.
- The user-confirmed normal `bmw_demo_20260812_211302` replays as `OK` (`PASS`) with coverage `0.1076672104`,
  longest run `0.0554649266`, maximum gap `0.0032626427`, one gap, and 20 bridged rows. The report retains all
  8/8 labeled no-streak rows as NG; calibration normal false rejects are 0 and final-test normal false rejects remain
  1, equal to v2. Recent replay has 20 records: one confirmed normal and 19 truth-unknown records.
- Only the bright-streak result may change. The invariant remains exactly 25 rows total and exact equality of all 24
  Template, YOLO, and EfficientAD `(branch, view, status, score, threshold, reason)` tuples. There are zero real
  broken-but-present samples, so broken-path handling is synthetic regression evidence only and real broken-streak
  recall remains unverified.

## BMW Template manual ignore-mask V5 candidate (2026-08-13)

- `configs/bmw/experiments/bmw_eight_view_demo_v5_template_manual_ignore_mask_v1.json` is an independent lab candidate. It reuses the exact EfficientAD manual-mask v3 index SHA-256 `fa5cf8eb6ff9cfa9a9c6d187a9aff58c8101b51dd5d3cef3a2576a7745b1bc58`; public ROI, Template model images, YOLO, bright-streak, HDR, EfficientAD checkpoints, and V1-V4 configs are not overwritten.
- Template masks follow the same aspect-fit and `BORDER_REFLECT_101` translation geometry as the query; masked CCOEFF_NORMED excludes selected pixels at every shift. Runtime details retain raw unmasked risk/shift plus masked risk/shift, valid fraction, mask SHA, threshold-asset SHA, and a diagnostic-only masked heatmap. Empty masks retain the exact legacy path.
- Calibration-only thresholds are in `results/bmw_template_manual_ignore_ab/bmw_right_manual_ignore_v3_template_ab_v1/template_masked_thresholds.json`, SHA-256 `ea7a57fcc27741793e8176aff21a8f9457eaf20405a0e932b240ebf94d5a3027`. `final_test_used_for_selection=false`; model JSON hashes and mask SHA are fail-closed in config loading.
- Offline A/B is pinned to the first 41 records (328 views) of `results/bmw_eight_view_demo_v3_ng_evidence_v1`, index-prefix SHA-256 `c4707c81b1bd145dd8b108b297394face234b7e5f5298af9faff663438deaf63`. View decisions: 38 old NG, 36 masked NG, 4 NG-to-PASS, 2 PASS-to-NG. Template capture decisions: 17 old NG, 14 masked NG, 3 NG-to-PASS, 0 PASS-to-NG.
- Calibration balanced accuracy improves `0.916319 -> 0.951781` (normal false rejects `29 -> 23`, defect false accepts `3 -> 1`). Untouched final-test balanced accuracy is slightly worse `0.836742 -> 0.832055` (normal false rejects `56 -> 59`, defect false accepts unchanged `5`). This remains a lab candidate, not production acceptance.
- The pose/absence pressure record `bmw_demo_20260812_205326` remains Template NG in all eight views. `bmw_demo_20260812_170451/front_right` becomes Template PASS and also becomes EfficientAD PASS under manual mask v3; its business truth is still uncertain, so V5 can make that capture OK and must not be called a verified defect-recall success.
- The worktree `.venv` lacks `ultralytics`; the root workspace environment has `ultralytics 8.4.89`. From an activated root environment, use `uv run --active --no-sync ...` so uv does not switch to the worktree environment.

## BMW V5 manual rotated bright-streak ROI (2026-08-13)

- V5 now selects `tracked_profile_v3_manual_rotated_roi` and keeps the existing tracked-profile v3 report and all geometry/threshold values unchanged. The report SHA-256 remains `6b43690af67702333646fa7a88a2a5053a0afae6d19cb343bf6d3abb193c17d4`.
- The accepted manual ROI asset is `results/bmw_bright_streak_rotated_roi/bmw_demo_20260813_164043_v3/roi.json`, SHA-256 `6ea49dacfae8d7d090bded6f8d67187d13fe00246c502a35813b77ce28aba4c8`. Its source-image points are LT `(1761,1548)`, RT `(1818,1528)`, RB `(1943,2108)`, LB `(1875,2135)` and it perspective-rectifies the full HDR image directly to the v3 contract `81x613` without Template alignment.
- The first two narrow manual selections were preserved as rejected diagnostic assets. Their roughly 27-pixel source widths stretched the white streak across the 81-pixel output and reduced the five-pixel local-contrast response below the unchanged strong threshold, yielding `NG_NO_STREAK`; do not bind V1 or V2.
- Full offline V5 replay is `results/bmw_eight_view_demo_v5_template_manual_ignore_mask_v1/bmw_demo_20260813_164043_rotated_v3`. The user identified the source as a real present-but-broken streak; runtime returned `NG_BROKEN`, presence true, coverage `0.4143556281`, longest run `108` pixels (`0.1761827080`), maximum gap `184` pixels (`0.3001631321`), and 7 gaps. The unchanged v3 limits are coverage `0.0848287113`, longest-run ratio `0.0440456770`, maximum-gap ratio `0.1060358891`, and maximum gap count 2.
- The full replay completed all 25 checks with 20 PASS, 5 NG, and 0 ERROR in about 6.47 seconds of model execution on CPU. Process exit 1 is the expected business NG result. Only the bright-streak row is acceptance evidence for this task; the other four NG rows belong to unchanged YOLO/EfficientAD branches.

### Rotated HDR weak-response recalibration v1 (2026-08-13)

- V5 keeps the SHA-bound original tracked-profile v3 report and the accepted rotated ROI, but explicitly deploys `weak_row_score_override=95.0` for `tracked_profile_v3_manual_rotated_roi`. The report value remains `128.2`; the strong threshold remains `136.85`; coverage, longest-run, maximum-gap and gap-count limits are unchanged.
- Root cause: the old weak threshold was fitted on fixed rectangular HDR crops, while the perspective-rectified ROI changes the row-response distribution. `bmw_demo_20260813_184658` had five tiny gaps of 9/8/3/4/4 pixels, and `184751` had a visually continuous but weak HDR segment; neither ROI was misplaced.
- Acceptance through the actual V5 config/predictor: `184658` and `184751` are now `PASS/OK`; all 33 manifest normal rows reprocessed through the rotated ROI are `OK`; all 8 no-streak rows remain `NG_NO_STREAK`; confirmed broken sample `164043` remains `NG_BROKEN` with maximum-gap ratio `0.1419249592` and gap count 3.
- Runtime evidence records `threshold_calibration=rotated_hdr_field_recalibration_v1`, `report_weak_row_score=128.2`, and deployed `weak_row_score=95.0`. This is an HDR field recalibration, not model retraining and not a change to Template, YOLO, EfficientAD, or whole-part fusion.

## BMW right-hand rotated bright-streak fast retraining (2026-08-14)

- The current right-hand normal release is `dataset/bmw_lab_prepared/bmw_right_normal_20260814_v1`: session `20260814_091058_791487`, 50 complete normal parts / 400 fused HDR images. The obsolete `bmw_left_normal_20260814_v1` name must not be used.
- The new no-streak capture is session `20260814_094431_343851`. Only `front_left` participates in bright-streak fitting: `dataset/bmw_lab_raw_clean/right/front_left/defect/no_streak/20260814_094431_343851/images/right_front_left_defect_no_streak_bmw_no_streak_group001_000001_fused.png`. It is one complete HDR-fused sample; no short/long source images were saved.
- `pipeline/bmw_lab_retrain_rotated_bright_streak.py` and `src/bmw_inspection/lab/bright_streak_rotated_retraining.py` implement the minimal tracked-profile V3 retraining path. They reuse the accepted SHA-bound rotated ROI and do not train or modify Template, EfficientAD, or YOLO.
- Candidate output is `results/bmw_bright_streak_rotated_retrain/bmw_right_normal50_no_streak1_20260814_v2`; report SHA-256 is `9d7b52a10726db45fd5f89776b5311be3e223916d9bd9a89939844efec7a5265`. The newly fitted strong threshold is `124.025`; the accepted rotated-HDR field weak threshold remains `95.0`. Normal-derived continuity limits are minimum coverage `0.3735725938`, minimum longest run `0.2365415987`, maximum gap ratio `0.5513866232`, and maximum gap count `4`.
- In-sample replay is 50/50 normal PASS and 1/1 no-streak rejected. The previously user-confirmed broken capture `bmw_demo_20260813_164043_rotated_v3` independently replays as `NG_BROKEN` because gap count is 6 above the new maximum 4. The sole no-streak image was used for fitting, so `no_streak_independent_test_count=0`; this is a lab candidate, not independent no-streak generalization evidence. It is not automatically wired into V5 because the existing Demo loader is pinned to the older report contract and runtime weak override.

## BMW right-hand normal-only selective retraining (2026-08-14)

- `pipeline/bmw_lab_retrain_normal_only.py` consumes the 50-part right-hand prepared release, reuses the existing fixed-setup ROI coordinates, trains Template and EfficientAD, recalibrates the legacy bright-streak rule with historical `no_streak`, and never invokes YOLO. Failed runs can be resumed only when completed ROI, release, Template, and bright-streak artifacts match their recorded paths and SHA-256 identities.
- A normal-only Template release cannot use the legacy risk-threshold fitter because that fitter requires both normal and defect calibration rows. The selective trainer now rebuilds all eight template banks from the new normal images while retaining the numeric per-view thresholds from `bmw_right_batch_20260810_21_v3_ng_evidence_demo_v1/template`.
- The current run `results/bmw_lab_one_click/bmw_right_normal_20260814_models_v1` already contains all eight rebuilt Template models. Its legacy bright-streak candidate has final-test balanced accuracy `0.45`, with 6 normal false rejects and 2 no-streak false accepts across 14 final-test rows; do not deploy that legacy candidate into V5. The tracked-profile V3 rotated retraining remains a separate contract.

## BMW V6 one-command handoff (2026-08-14)

- `pipeline/bmw_lab_prepare_normal_20260814_v6_demo.py` invokes `publish_v6_demo` with no-overwrite V6 output defaults, emits a Chinese JSON error on publication failure, and prints the exact experiment-mode launch command after the receipt. It delegates source validation and rebinding to `src/bmw_inspection/lab/v6_demo_publisher.py`; no training or live inference is performed by this CLI.

## BMW left-hand minimal retraining workflow (2026-08-14)

- The runnable handoff is `docs/bmw/LEFT_HAND_QUICKSTART.md`. Work from `.worktrees/bmw-eight-view-handoff`, but use absolute dataset/result paths under `/home/yunjing/anomaly_xingtao_new`.
- Select a new left `fixed_setup` eight-view ROI from raw normal data, then run `pipeline/bmw_lab_train_left_normal.py`: `train` materializes data and trains only Template plus EfficientAD; `calibrate` scores normal calibration images with the deployment component policy. This flow has no YOLO or bright-streak training stage.
- EfficientAD deployment score is `accepted_component_max_p95`, not the anomaly-map single-pixel maximum. The experimental policy filters small and shallow isolated components while keeping hard peaks, long thin regions, and larger regions; calibration and runtime must use the same mask, policy, checkpoint and score domain.
- `pipeline/bmw_lab_select_efficientad_ignore_masks.py --training-release ...` supports a blank left-hand mask without seeding from the right-hand asset. The ROI, mask, policy, checkpoint and threshold artifacts remain linked by their recorded identities.
- Reuse the existing joint left/right YOLO checkpoint at `results/bmw_lab_one_click/bmw_right_multisource_left_yolo_v1/yolo/train/weights/best.pt` (SHA-256 `0e9591f2fa2487ad12000847f1d80137901ed69989e0e95cb5c8907b95ba3913`); do not retrain YOLO for this normal-only experiment.
- Bright-streak retraining remains separate until left-hand normal and no-streak captures exist. No real left GPU training, live camera run, or model-quality acceptance was performed while preparing this workflow.
- Left normal session `20260814_124313_743121` contains 52 complete parts; group017 and group018 are byte-identical in all eight views. Its `bmw_left_normal_20260814_v2` release therefore uses `--skip-image-hash`. Materialization now treats an empty `source_sha256` as the explicit unverified experimental mode while continuing to verify every non-empty digest; the real left `--stage train --dry-run` passed after this compatibility fix.

## BMW left-hand Demo composition v1 (2026-08-14)

- `pipeline/bmw_lab_prepare_left_20260814_demo.py` publishes `results/bmw_lab_one_click/bmw_left_normal_20260814_demo_v1` and `configs/bmw/experiments/bmw_eight_view_demo_left_normal_20260814_v1.json` without changing the common Demo schema or the completed source training run.
- The composite run links `template` and `efficientad` from `bmw_left_normal_20260814_models_v3` and links `yolo` from `bmw_right_multisource_left_yolo_v1`; shared YOLO checkpoint SHA-256 remains `0e9591f2fa2487ad12000847f1d80137901ed69989e0e95cb5c8907b95ba3913`.
- The config binds left ROI v2 SHA `2119e90a14074206ff9484b4e43c1b581ef4a8a88abd46eb73f85d8fed214a13`, left EA mask/policy/component thresholds, and bright engine `tracked_profile_v3_manual_rotated_candidate` with the left candidate report and rotated ROI. It omits the Template block because thresholds are embedded in each left model.json, and omits trusted-OK because no approved left reference bank exists.
- The generated config passed `load_demo_config()` and composition tests, but GPU inference, live four-camera capture, and model-quality acceptance remain separate field tests.

## BMW left-hand YOLO fixed fixture ignore region (2026-08-14)

- `configs/bmw/experiments/bmw_eight_view_demo_left_normal_20260814_v1.json` configures a YOLO-only ignore rectangle for `front_secondary`: `[1580, 450, 1756, 800]` in ROI coordinates. It does not change the shared ROI or affect Template, EfficientAD, or bright-streak processing.
- The trigger was six repeatable live detections of the same fixture at approximately `[1641, 521, 1756, 724]`, confidence `0.293` to `0.377`. A box is ignored only when its center lies inside a configured rectangle for the current view; identical coordinates in another view remain eligible.
- Ignored boxes remain in YOLO diagnostics with `is_ignored=true` and `ignore_reason=fixed_yolo_ignore_region`, but do not contribute to decision score or final NG count. Saved-image replay of `bmw_demo_20260814_143530/front_secondary` changed the YOLO result to PASS with one candidate, one ignored box and zero final boxes. The two focused Demo test files pass with `122 passed`.

## BMW left-hand trusted-OK reference bank (2026-08-14)

- The user approved the current left normal release for trusted evidence. `pipeline/bmw_lab_publish_manifest_trusted_ok.py` published all 52 complete parts / 416 views from `bmw_left_normal_20260814_v2`, recomputing the intentionally omitted source hashes and binding ROI SHA `2119e90a14074206ff9484b4e43c1b581ef4a8a88abd46eb73f85d8fed214a13`.
- The immutable reference bank is `/home/yunjing/anomaly_xingtao_new/dataset/bmw_trusted_ok_reference/bmw_left_normal_20260814_approved_v1`; `reference_index.json` SHA-256 is `9e38b125d0ed1ab1207c6eb7f4951f214090824568608f99a1b4788113f4865d`, whitelist SHA-256 is `ee820269d37530a96bfca25fdc57a05e55a122d89009a84e9f6f3a0a6df2b449`, and every view has 52 references.
- Schema v1 retains the frozen right-hand identity/session/exact-50 contract. Schema v2 permits an index-declared positive part count while still requiring equal view counts, complete eight-view parts, consistent per-part session/split, source/full/ROI SHA validation and index-SHA config binding.
- `bmw_eight_view_demo_left_normal_20260814_v1.json` uses the absolute main-workspace index path because the worktree dataset only symlinks selected dataset branches. Real preload completed for all eight ROI banks plus front-left full-image bank; it took about 28 seconds and did not run detectors or cameras.

## BMW EfficientAD all-normal checkpoint training (2026-08-20)

- `--efficientad-all-normal-train` is available on the materialization, full-training, and left-normal training CLIs. It sends every current `normal` row to each view's EfficientAD `normal` directory while leaving Template and YOLO on the prepared release's public train/calibration/final-test split.
- This mode is deliberately checkpoint-only: EfficientAD gets no internal validation/test directory, `engine.test` is skipped, and reports record `validation_status=pending_external_validation`. The left workflow accepts only `--stage train` in this mode.
- Do not deploy or compare the new checkpoints with old numeric thresholds. Collect an independent normal/defect validation release, apply the same ROI, ignore mask, and component-score policy, then fit new per-view and whole-part thresholds.

## BMW 0820 bilateral Template and all-normal EfficientAD command (2026-08-20)

- `pipeline/bmw_lab_train_bilateral_normal.py` sequentially runs right then left with `stage=train`. Each hand materializes its own ROI release, trains Template, and trains eight EfficientAD checkpoints with `efficientad_all_normal_train=true`; any right-hand failure prevents left-hand training.
- Defaults bind `bmw_right_0820_v1` to `configs/bmw/rois/bmw_right_0820_v1.json` and `bmw_left_0820_v1` to `configs/bmw/rois/bmw_left_0820_v1.json`. The prepared-manifest ROI contract is verified through dataset identity, source-manifest path, and manifest SHA rather than a fixed-setup `capture_scope` field.
- Current manifests contain 89 right normal parts and 90 left normal parts, so each of the eight right checkpoints receives 89 images and each left checkpoint receives 90 images. Template retains its train/calibration split. YOLO, bright-streak, internal EfficientAD validation, and threshold fitting are not invoked.
- Preflight with `UV_CACHE_DIR=/tmp/bmw-uv-cache uv run --no-sync python pipeline/bmw_lab_train_bilateral_normal.py --dry-run`; remove `--dry-run` to train. The generated checkpoints remain pending independent validation and must not reuse old thresholds.

## BMW left-hand Template / EfficientAD NG review package (2026-08-16)

- The no-overwrite review package is `results/bmw_template_efficientad_ng_review_left_20260816_v1`. It scans 96 valid inspection records from `results/bmw_lab_one_click/bmw_eight_view_demo_left_normal_20260814_v1` and contains 111 review rows from 46 captures: 32 Template NG and 79 EfficientAD NG.
- Every row has the current ROI, detector evidence overlay, a newly matched same-view reference from `bmw_left_normal_20260814_approved_v1`, and an aligned current-versus-OK difference image. Review identities are `T001..T032` and `E001..E079`; human decisions belong in `review_cases.csv` and no detector threshold or model was changed while generating this diagnostic package.
- `pipeline/bmw_lab_review_ng_cases.py` is the local Tkinter reviewer for this package. It groups all problems by `capture_id`, records per-problem decisions (`误判`, `真实缺陷`, `不确定`), supports undecided-only per-part batch actions and keyboard navigation, and atomically updates the original CSV while preserving its schema and row order. Run it from the handoff worktree with `env UV_CACHE_DIR=/tmp/bmw-uv-cache uv run --no-sync python pipeline/bmw_lab_review_ng_cases.py`; use `--check` for a non-GUI package preflight.
- The reviewer must use Tk's actually available `song ti` family for both classic Tk and ttk widgets; requesting `Noto Sans CJK SC` silently falls back to unreadable `fixed` on this workstation. EfficientAD keeps its same-size anomaly overlay. Template is a whole-ROI similarity detector rather than a pixel-localizer, and its saved absolute-difference composite contains both current and template edges plus reflected padding; the reviewer therefore hides that misleading composite and shows the original ROI under the explicit label `Template整体匹配异常（原始ROI）`. This changes no score, threshold, or detector decision.

## BMW live front-round inference overlap (2026-08-20)

- Live GUI inference is split without changing detector calculations: the front round runs 4 Template + 1 full-image `front_left` bright-streak + 4 YOLO + 4 EfficientAD checks (13 total), and the back round runs 4 Template + 4 YOLO + 4 EfficientAD checks (12 total).
- `pipeline/bmw_lab_eight_view_demo.py` keeps all camera capture and OpenCV HighGUI work on the main thread. A single `ThreadPoolExecutor(max_workers=1)` owns both model rounds, final fusion, trusted-OK matching, and persistence. Back HDR capture may begin while front inference is still running, but front/back model calls never run concurrently.
- `EightViewModelSuite.finalize_rounds()` restores the original canonical order (8 Template, 1 bright-streak, 8 YOLO, 8 EfficientAD), then performs the unchanged `ERROR > NG > OK` fusion and trusted-OK comparisons. Only the complete 25-result inspection is persisted. Space is ignored in `PROCESSING`; reset advances a generation token so a stale completed result cannot replace the current UI state.
- Focused verification passed: 51 unit tests across the active pipeline/models/config/persistence/UI chain, including the worker-before-camera shutdown order, plus Python compilation and `git diff --check`. A real CPU single-worker replay of right sample `bmw_right_normal_group001_000001` produced 13 + 12 = 25 results, 0 ERROR, final NG, and about 5.88 seconds accumulated model time. Its saved capture is `results/bmw_lab_one_click/bmw_eight_view_demo_right_0820_mixed_v1/bmw_right_normal_group001_000001_front_overlap_smoke_v1`.
- The replay used CPU because NVML/CUDA was unavailable. Actual CUDA worker-thread behavior, four-camera timing, and the amount of field-cycle-time improvement remain unverified and require one live smoke test.

## BMW 0820 EfficientAD reduced ignore masks v2 (2026-08-20)

- The operator revised both hands' EfficientAD masks to ignore less of every ROI. The active assets are `results/bmw_efficientad_manual_ignore_masks/bmw_right_0820_manual_ignore_v2/index.json` and the corresponding `bmw_left_0820_manual_ignore_v2/index.json`; each contains eight readable binary masks whose dimensions exactly match the 0820 public ROI crops.
- `bmw_eight_view_demo_right_0820_mixed_v1.json` and `bmw_eight_view_demo_left_0820_mixed_v1.json` now reference these v2 EfficientAD indexes directly. The right Template branch still uses its separate legacy Template mask and was not changed.
- Both configs and all 16 masks loaded through the active Demo loader. The two focused config test files pass with 5 tests. Detector replay, score changes, and live camera behavior with v2 remain unverified.

## BMW 0820 combined left/right YOLO deployment (2026-08-21)

- The shared one-class detector is now `results/bmw_lab_one_click/bmw_right_multisource_left_yolo_0820_v2/yolo/train/weights/best.pt`. It is an Ultralytics detect checkpoint with the single class `0: defect`; the training YAML combines the previous shared release with the reviewed right and left 0820 ROI crops.
- Both active mixed configs reference that checkpoint directly: `bmw_eight_view_demo_right_0820_mixed_v1.json` keeps `configs/bmw/rois/bmw_right_0820_v1.json`, while `bmw_eight_view_demo_left_0820_mixed_v1.json` keeps `configs/bmw/rois/bmw_left_0820_v1.json`. The left `front_secondary` ROI-local ignore region remains `[1505, 628, 1681, 978]`. YOLO confidence thresholds remain candidate `0.1` and final `0.25`.
- The completed training validation reported 816 images / 198 instances, precision `0.578`, recall `0.373`, mAP50 `0.396`, and mAP50-95 `0.180`. These are training-run validation metrics, not live-camera acceptance.
- Focused regression verification passed with 24 tests. CPU offline replay loaded the complete four-branch suite for both hands: right `bmw_right_0820_new_yolo_replay_DYkNdn` produced 25 checks, 0 ERROR, 8 YOLO PASS and final NG only because `front_left` Template exceeded its unchanged threshold; left `bmw_left_0820_new_yolo_replay_V8hxPM` produced 25 PASS, 0 ERROR. CUDA inference and live four-camera capture remain unverified.

## BMW left front-view deformation thresholds (2026-08-21)

- The operator confirmed visible small-foot deformation in the latest left现场 capture `bmw_demo_20260821_192534` and requested stricter Template plus EfficientAD decisions only for `front`, `front_left`, and `front_right`.
- Active left Template thresholds are now `0.015`, `0.011`, and `0.008` in that view order. Active left EfficientAD `accepted_component_max_p95` thresholds are now `0.33`, `0.36`, and `0.37`. The remaining five views, all models, ROI, masks, YOLO, bright-streak, HDR, and fusion are unchanged.
- Threshold metadata is `field_deformation_anchor_20260821`; EfficientAD validation status is `lab_field_tuned_single_defect_anchor_pending_normal_recheck`. These values are a single confirmed defect-anchor adjustment, not independent normal/defect calibration.
- CPU replay `bmw_left_192534_strict_front_replay_LrHWyw` produced 25 checks, 19 PASS, 6 NG, and 0 ERROR. All three targeted Template rows changed PASS→NG with identical scores; all three targeted EfficientAD rows changed PASS→NG. The other 19 rows had no status or threshold changes. Focused active-chain verification passed with 28 tests; CUDA and new live-camera captures remain unverified.

## BMW 0823 front_right-only EfficientAD retraining (2026-08-23)

- New clean normal captures are right session `20260823_144248_477990` with 87 complete parts and left sessions `20260823_161606_428291` plus `20260823_164432_522052` with 88 complete samples. Incomplete trailing samples are excluded. Sampled first/middle/last `front_right` images no longer show the reported foreign object.
- New prepared releases are `dataset/bmw_lab_prepared/bmw_right_front_right_0823_v1` and `bmw_left_front_right_0823_v1`. The ROI training releases are `dataset/bmw_lab_training/bmw_right_front_right_0823_v1` and the corresponding left release. They contain only `crops/front_right` and `efficientad/front_right`; no Template, YOLO, or other-view crops/models were generated.
- `pipeline/bmw_lab_materialize_training_data.py` now accepts repeatable `--view` and `--efficientad-only`. The fast EfficientAD-only path reuses the explicitly supplied ROI geometry on a new prepared release without requiring the old prepared-manifest identity/SHA binding, while image readability, dimensions, and ROI crop checks remain active.
- `pipeline/bmw_lab_train_efficientad_only.py` now accepts repeatable `--view` and `--efficientad-all-normal-train`. A selected all-normal run trains only the requested checkpoint and skips unavailable `normal_test` scoring/threshold fitting; the default full-view validated workflow remains unchanged.
- Completed CUDA runs are `results/bmw_lab_one_click/bmw_right_front_right_efficientad_0823_v2` and `bmw_left_front_right_efficientad_0823_v2`. Each contains exactly one `efficientad/front_right/model.ckpt`, trained for 30 epochs with batch 1 and image size 256. Checkpoints are 76,869,373 bytes and each loaded for a real one-image GPU prediction with finite score and a finite 256x256 anomaly map.
- These are all-normal checkpoint candidates with `pending_external_validation`; no threshold was fitted and neither active Demo config was changed. Do not reuse the old `front_right` threshold without a separate score replay. The failed right `...0823_v1` run only records the initial sandbox GPU-detection failure and has no accepted checkpoint.
- Focused verification: 29 unit tests passed across materialization, EfficientAD-only orchestration, and shared training contracts; Python compilation and `git diff --check` passed.

## BMW 0823 front_right EfficientAD Demo integration (2026-08-23)

- The active mixed configs now replace only `efficientad.checkpoints.front_right`: right uses `results/bmw_lab_one_click/bmw_right_front_right_efficientad_0823_v2/efficientad/front_right/model.ckpt`; left uses the corresponding `bmw_left_front_right_efficientad_0823_v2` checkpoint. The other seven EfficientAD checkpoints, all Template/YOLO/bright-streak assets, ROI, masks, HDR settings, and fusion are unchanged.
- Per the operator's explicit request, the old numeric deployment thresholds are retained without recalibration: right `front_right=0.549995529652`, left `front_right=0.43`. Existing global threshold-source/validation metadata is also unchanged; these values remain lab candidates rather than independently validated thresholds.
- Focused config contracts passed with 9 tests. Full GPU offline replay on clean 0823 normal group001 loaded all four branches and the trusted-OK bank for both hands. Right `bmw-right-0823-new-ea-replay-v1` was 25 PASS / 0 NG / 0 ERROR, with new front-right EA score `0.3124246299 < 0.549995529652`. Left `bmw-left-0823-new-ea-replay-v1` was also 25 PASS / 0 NG / 0 ERROR, with score `0.2961165607 < 0.43`.
- Replay inputs are fused-only saved captures, not a live camera run. The live GUI must be restarted after this config change because model paths and thresholds are loaded only once at process startup.

## BMW 0823 front_right-only Template retraining and integration (2026-08-23)

- `pipeline/bmw_lab_materialize_training_data.py` now also supports `--template-only` with repeatable `--view`. For this mode it writes only the selected canonical crops and `template/trainer_manifest.csv`; it does not create EfficientAD or YOLO branch data. The new `pipeline/bmw_lab_train_template_only.py` trains only the selected Template view using the existing fixed-threshold trainer, so an all-normal retake does not attempt invalid normal-versus-defect threshold fitting.
- The right release `dataset/bmw_lab_training/bmw_right_front_right_template_0823_v1` contains 87 `front_right` crops; the left counterpart contains 88. Completed model runs are `results/bmw_lab_one_click/bmw_right_front_right_template_0823_v1` and `bmw_left_front_right_template_0823_v1`; each contains only `template/front_right`, five 512x512 grayscale templates, and one model.json whose input size matches the hand-specific 0820 ROI.
- Only `template.models.front_right` changed in the two active mixed configs. Right points to `bmw_right_front_right_template_0823_v1/template/front_right/model.json` and keeps the exact old threshold `0.012455999851226807`; left points to the corresponding left run and keeps `0.008`. The other seven Template models/thresholds, all EfficientAD/YOLO/bright-streak assets, ROI, masks, HDR, and fusion remain unchanged.
- Full GPU replay loaded all 25 checks with no ERROR. Left `bmw-left-0823-new-template-replay-v1` was 25 PASS and final OK; its new Template score was `0.0038328767 < 0.008`. Right `bmw-right-0823-new-template-replay-v1` was 24 PASS / 1 NG and final NG solely because the new masked Template score `0.0126539469` was slightly above the explicitly retained old threshold `0.0124559999` by `0.0001979470`. Do not silently loosen it: the operator explicitly requested the old threshold.
- Focused verification passed with 31 tests across Template-only materialization/training, existing EfficientAD-only behavior, and active mixed-config contracts; Python compilation and `git diff --check` passed. The replay used saved fused images; live four-camera capture remains to be tested after restarting the GUI.

## BMW key-region Template secondary checks approved design (2026-08-23)

- Approved design: `docs/bmw/BMW_KEY_TEMPLATE_SUBROI_DESIGN_20260823.md` (commit `c9bb71c4`). It keeps the existing whole-ROI Template and adds an optional `key_template` aggregate result only for views with at least one configured key ROI.
- Right and left maintain separate ROI selections, models, and thresholds. Every canonical view may contain zero or multiple public-ROI-local rectangles named `roi_01`, `roi_02`, etc. Empty views produce no SKIPPED row, so checks are `25 + enabled view count`.
- Each sub-ROI is scored independently; per-view aggregation and final fusion use `ERROR > NG > PASS`. Any key ROI NG makes the part NG, while all remaining checks continue and persist evidence.
- Initial per-ROI thresholds use the calibration-normal maximum risk; final_test normals are reporting only. Thresholds remain directly editable JSON values with no SHA/publisher/rebind layer. Implementation has not started and requires user review of the written design first.

## BMW key-region Template secondary checks implementation (2026-08-23)

- The optional infrastructure is implemented without activating either hand. `pipeline/bmw_lab_select_key_template_rois.py` selects zero or multiple public-ROI-local rectangles for every canonical view; `pipeline/bmw_lab_train_key_templates.py` trains only those regions and writes direct `model.json`, `metrics.json`, and `runtime.json` paths without SHA, receipt, publisher, or rebind metadata.
- `src/bmw_inspection/lab/key_template.py` accepts missing/empty views, validates only the necessary JSON/path/image/ROI/model contracts, selects up to five normal-train templates, fits each deployment threshold as the next float above the calibration-normal maximum risk, and reports final-test false rejects without using final-test rows for selection.
- A Demo profile may optionally add top-level `key_template_config`. When absent, the original 25 checks and front 13/back 12 scheduling remain unchanged. When present, each enabled view contributes one aggregate `key_template` row after the whole-ROI Template rows; every configured sub-ROI runs, and per-view plus final fusion use `ERROR > NG > PASS`.
- The Chinese UI adds a conditional `关键区模板` card and K shortcut only when key results exist. Evidence is one public-ROI overlay with each key rectangle/status, while per-region scores, thresholds, reasons, and Template diagnostics remain in `inspection.json`. Existing persistence and trusted-OK ROI comparison are reused.
- Focused verification passed: 56 tests across key config/training, Demo config/models/UI/persistence/pipeline; Python compilation, CLI help, and `git diff --check` passed. A no-key left offline replay of `bmw_left_normal_group002_000001` produced 25 PASS / 0 NG / 0 ERROR in 4307 ms on CPU. Actual key ROI selection, model training, active-config wiring, GPU key inference, and live cameras remain pending operator-selected ROI files.

## BMW 0823 key-region Template activation (2026-08-23)

- Operator selections are `configs/bmw/key_template/bmw_right_0823_key_rois_v1.json` and `bmw_left_0823_key_rois_v1.json`. Both hands enable two regions in each of `front`, `front_left`, `front_right`, and `front_secondary` (8 regions per hand); all four back views are empty. An exact duplicate left `front_left` box was removed before training.
- Trained direct assets are `results/bmw_lab_one_click/bmw_right_key_template_0823_v1/runtime.json` and the corresponding left directory. Each contains eight region `model.json` files, five 512x512 templates per region, metrics, and editable thresholds fitted as the next float above calibration-normal maximum risk. No SHA, receipt, publisher, or rebind metadata was generated.
- The two active mixed configs now set top-level `key_template_config` to their hand-specific runtime JSON. Enabled runs produce 29 rows: original 25 plus four aggregate key-template view rows. Each aggregate evidence image shows all public-ROI-local rectangles and the worst sub-ROI Template heatmap; full per-region scores remain in `inspection.json`.
- Final-test normal reporting shows some expected false rejects under the deliberately strict calibration-max thresholds: right region totals by final-test false rejects are `1,0,0,0,0,2,1,0`; left are `1,1,0,3,0,2,2,0` in canonical enabled-view/region order. Thresholds were not loosened automatically.
- CPU full-suite replay produced 29 rows and 0 ERROR for both hands. Right 0823 final group007 had one key NG (`front/roi_01`, 0.03303 > 0.03013) plus two pre-existing-module NGs; left 0823 final group005 session1 had one key NG (`front_right/roi_02`, 0.02713 > 0.02607). This validates wiring and strict behavior, not live-camera acceptance. CUDA and cameras remain unverified.

## BMW key-region Template feature removed (2026-08-23)

- The operator judged the sub-ROI Template branch too sensitive and explicitly requested removal. Both active mixed configs no longer contain `key_template_config`; the Demo enum/config loader, model predictor/round ordering/fusion, UI card/K shortcut, selector/trainer CLIs, implementation tests, and design/plan docs were removed.
- The active detector is restored to the established four modules and fixed 25 checks: front 13 plus back 12, ordered as 8 whole-ROI Template, 1 bright-streak, 8 YOLO, and 8 EfficientAD. No key-template symbols or active config references remain under `src`, `pipeline`, active experiment configs, or focused tests.
- Previously generated selection JSON, model directories under `results/bmw_lab_one_click/bmw_{left,right}_key_template_0823_v1`, and historical 29-row replay results were intentionally retained because project boundaries prohibit deleting models/results/data. They are inactive and are not loaded by the Demo.
- Focused verification after removal passed with 53 tests, Python compilation, `git diff --check`, and an active-source residual search. CPU left replay `bmw_left_normal_group004_000001` produced 25 PASS / 0 NG / 0 ERROR in 4313 ms. CUDA and live cameras remain unverified.

## BMW weighted Template regions approved design (2026-08-23)

- After removing the independent key-template branch, the operator approved reusing the saved hand-specific ROI selections only as pixel weights inside the existing whole-ROI Template result. The detector must remain fixed at 25 checks with no new branch or one-vote veto.
- Approved initial weight is `3.0` inside selected regions and `1.0` elsewhere; Template ignore-mask pixels override both with weight `0`. The existing whole-ROI Template still selects the template and shift first, then a weighted normalized correlation recomputes the single active risk.
- Existing `template.thresholds` remain unchanged for rollback. A new `template.weighted_regions` block will hold `enabled`, ROI path, weight, and separately recalibrated thresholds. Disabling it restores exact legacy scoring after restart.
- Weighted thresholds will use each view's 0823 calibration-normal maximum weighted risk plus 10% margin; final-test normal only reports false rejects. Formal spec is `docs/superpowers/specs/2026-08-23-bmw-weighted-template-regions-design.md`, committed as `8dc28aff`. No production code or active threshold was changed during design.

## BMW weighted Template regions implementation and calibration (2026-08-23)

- The saved hand-specific selections are now weights inside the existing whole-ROI Template branch, not an independent detector. `src/bmw_inspection/lab/template_region_weighting.py` maps public-ROI-local rectangles through the same aspect-fit/alignment geometry, applies weight `3.0` versus base `1.0`, uses max in overlaps, and gives Template ignore-mask pixels final weight `0.0`. The active chain remains front 13 + back 12 = 25 rows.
- Right uses `configs/bmw/key_template/bmw_right_0823_key_rois_v1.json`; left uses `configs/bmw/key_template/bmw_left_0823_key_rois_v1.json`. Both contain two regions in each of the four front views and no back-view regions. Empty back views continue the exact legacy score and threshold.
- Legacy rollback thresholds remain unchanged under each active config's `template.thresholds`. Weighted thresholds are under `template.weighted_regions.thresholds`; rollback is only `enabled=false` plus a full Demo restart. No SHA, receipt, provenance, publisher, or rebind path was added.
- Right weighted thresholds in `front, front_left, front_right, front_secondary` order are `0.012498484389187015`, `0.005421768437248353`, `0.01583567259539418`, and `0.010992674875391907`. Its 18 final-test normals had one false reject in each weighted view under the fixed calibration-max-plus-10% rule.
- Left weighted thresholds in the same order are `0.011145261314028676`, `0.005135343811192717`, `0.0054486672193632575`, and `0.005552518444693267`. Its 18 final-test normals had zero false rejects in all four weighted views.
- Calibration reports are `results/bmw_lab_one_click/bmw_weighted_template_calibration_right_0823_v1/report.json` and the corresponding left report. The CLI is `pipeline/bmw_lab_calibrate_weighted_template.py`; `--write-config` modifies only the weighted threshold object and never the legacy threshold object.
- Final focused regression passed with 47 tests. Fresh CPU full-suite replays on 0823 group001 were both 25 PASS / 0 NG / 0 ERROR: `bmw-right-weighted-replay-final-20260823` in 4864 ms and `bmw-left-weighted-replay-final-20260823` in 4169 ms. Four-camera capture and CUDA inference for this weighting change remain unverified.

## BMW 20-template manual review workflow approved design (2026-08-23)

- The operator requested expanding every Template view from five stored templates to up to twenty: right and left remain independent, all eight views are included, and the initial review package therefore contains 320 ROI crops.
- Candidates come only from each hand's 0823 normal train split and prioritize distinct physical parts. Left duplicate sample IDs are disambiguated by capture session. The review package defaults every candidate to approved; deleting a copied candidate rejects it, and rejected candidates are not automatically replaced.
- Each view must retain 3–20 reviewed images. Training consumes exactly the surviving reviewed images without a second selection pass, writes new versioned model roots, and leaves the current active configs/models untouched until review, retraining, threshold calibration, and offline replay are complete.
- Both ordinary and weight-3.0 Template thresholds must be recalibrated from calibration normal with a 10% margin; final_test remains reporting-only. The approved specification is `docs/superpowers/specs/2026-08-23-bmw-template-20-review-design.md`, commit `4bec9962`.

## BMW 40-template manual review package (2026-08-23)

- The operator superseded the 20-candidate plan with 40 candidates for every canonical view of both hands. The current design is `docs/superpowers/specs/2026-08-23-bmw-template-40-review-design.md`; implementation steps are in `docs/superpowers/plans/2026-08-23-bmw-template-40-review.md`.
- `pipeline/bmw_lab_prepare_template_review.py` uses the right and left 0823 prepared manifests plus their hand-specific 0820 public ROI configs. It selects only `train/normal/OK`, prioritizes previously unseen physical parts, then uses a deterministic farthest-point diversity order. It does not train or modify active Demo configs.
- The generated review package is `/home/yunjing/anomaly_xingtao_new/dataset/bmw_lab_labeling/bmw_template_40_review_0823_v1`: 640 readable color ROI PNGs, 16 contact sheets, `candidate_manifest.csv`, and `review_instructions.txt`. Right has 40 unique physical parts per view from session `20260823_144248_477990`; left has 28 unique physical parts per view and uses both sessions `20260823_161606_428291` and `20260823_164432_522052` to reach 40.
- Manual review is delete-to-reject: delete only unsuitable `candidate_*.png` copies. Do not edit/delete source images or `candidate_manifest.csv`. Deleted candidates are not refilled and number gaps are valid. Training must wait for the operator to confirm review completion and must consume exactly the surviving copies, with at least 3 per view.
- Current 5-template models, legacy thresholds, weight-3.0 thresholds, and active left/right mixed configs remain unchanged in this phase. Adding the reviewed templates later requires recalibrating both ordinary and weighted Template thresholds before any candidate config is activated.

## BMW reviewed Template-40 training and candidate configs (2026-08-23)

- The operator completed delete-to-reject review. Ten of 640 candidate copies were removed and were not refilled. Right retained counts in canonical view order are `39,39,40,40,40,40,40,40` (319 total); left retained `38,37,39,38,40,40,40,40` (311 total).
- `pipeline/bmw_lab_train_reviewed_templates.py` and `src/bmw_inspection/lab/template_review_training.py` consume exactly the surviving review copies and perform no second selection. New roots are `results/bmw_lab_one_click/bmw_right_template_40_reviewed_0823_v1` and the corresponding left root. Every retained crop became one 512x512 resident template; the new model JSON files and directories contain no SHA sidecars or SHA fields.
- Both ordinary runtime Template risk and weight-3.0 risk were scored in one pass over each hand's 0823 calibration/final-test normal rows. Thresholds use calibration-normal maximum plus 10%; final_test is reporting only. Exact values and per-view false rejects are in each model root's `calibration_report.json`.
- Rollback configs `bmw_eight_view_demo_{right,left}_0820_mixed_v1.json` remained byte-identical. New integrated candidates are `configs/bmw/experiments/bmw_eight_view_demo_right_0823_template40_v1.json` and the corresponding left config. They switch only all eight Template model paths, ordinary Template thresholds, weighted Template thresholds, the prepared manifest, and an independent result root; EfficientAD, YOLO, bright-streak, ROI, masks, trusted OK, HDR, and fusion are inherited unchanged.
- CPU full-suite replay with the right candidate on `bmw_right_normal_retake_group007_000001` produced 25 rows / 0 ERROR / 24 PASS / 1 NG; the sole NG was unchanged EfficientAD `back_left`, while all eight new Template rows passed. Left session `20260823_161606_428291` group005 replay produced 25 PASS / 0 NG / 0 ERROR. These are saved-image replays; CUDA and live four-camera operation remain unverified.

## BMW Template ROI-outside 0.5 weighting candidate (2026-08-24)

- The existing rollback configs `configs/bmw/experiments/bmw_eight_view_demo_{left,right}_0823_template40_v1.json` remain unchanged. New candidates are `bmw_eight_view_demo_{left,right}_0823_template40_outside05_v1.json`, with independent result roots. Switching versions requires selecting the desired config at startup; a running GUI does not reload it.
- The existing weighted Template branch now accepts optional `template.weighted_regions.outside_weight`, defaulting to `1.0` for backward compatibility. The candidate uses ROI weight `3.0`, outside weight `0.5`, and ignore-mask weight `0.0`, so the effective ROI-to-outside ratio is `6:1`. Only the four front views contain regions; the four back views keep their exact legacy scores and thresholds.
- Left candidate weighted thresholds in `front, front_left, front_right, front_secondary` order are `0.010475272808328773`, `0.003943851201103199`, `0.005322204567186717`, and `0.006334702633248857`. Right values are `0.011022074315514753`, `0.0045888184310348586`, `0.01314845384400194`, and `0.006763783384760403`.
- Thresholds use calibration-normal maximum plus 10%. Final-test is reporting-only: each hand has one false reject in `front` and one in `front_left`, with zero in `front_right` and `front_secondary`. Reports are under each new result root at `calibration/weighted_template_outside05_report.json`.
- Focused regression passed with 40 tests. CPU replay `bmw_right_normal_retake_group001_000001` using the right candidate produced 25 PASS / 0 NG / 0 ERROR in 9172 ms; its front Template evidence records weights `3.0/0.5` and effective ratio `6.0`. This is saved-image verification; CUDA behavior for the new weighting and live four-camera capture remain unverified.

## ZS32 standalone classical surface-operator prototype (2026-07-26, merged 2026-09-02)

- The prototype code is merged into the current repository, but remains runtime-isolated: it is not connected to Stage 32, Stage 18, Dashboard, v10, model configs, or threshold artifacts.
- Core code is `capture_data/zs32_classical_operators.py`; offline benchmarking is in `capture_data/zs32_classical_benchmark.py` and `pipeline/38_benchmark_zs32_classical_operators.py`. The operators publish uncalibrated continuous scores, masks, components, and overlays only, never final inspection labels.
- Historical benchmark artifact `results/zs32_classical_operator_benchmark_v1` used 30 full-resolution ROIs and reported p50 `0.1691 s` for `thin_line`, `0.2429 s` for `pit_spot`, `2.8204 s` for a six-image serial round, and `1.5489 s` for a two-worker round.
- Visual quality was not accepted: fixed geometry, normal texture, high-contrast hole edges, and angled-view backgrounds produced false evidence. Add per-view material/ignore masks and region-level target labels before considering runtime integration.
- Detailed limitations and commands are in `docs/ZS32_CLASSICAL_OPERATORS_PROTOTYPE.md`.

## Worktree consolidation (2026-09-02)

- BMW branch `agent/bmw-21only-diagnostics` was snapshotted at `fedeb4ba` and merged into `feat/zs32-strict-eight-view-commissioning` as `6b619bba`. The BMW subsystem uses the worktree's latest simplified runtime/config/tests; current ZS32 memory was retained and BMW-only memory sections were appended.
- ZS32 branch `feat/zs32-classical-operators-prototype` was merged as `40ce26b1`. The classical operators remain offline diagnostic tools and are not wired into Stage 32, Stage 18, Dashboard, v10, model configs, or threshold artifacts.
- Post-merge verification was `563 passed` for BMW-focused tests and `74 passed` for the classical-operator/wrapper tests. Both runs emitted the existing NVML initialization warning; no CUDA, camera, live GUI, or production acceptance was performed.
- Unique ignored BMW assets were preserved before cleanup: `dataset/bmw_trusted_ok_review`, `dataset/bmw_trusted_ok_reference/bmw_right_20260810_21_train_normal_approved_v1`, and three `artifacts/bmw_*` directories were moved from the worktree into the root checkout. The duplicate approved-v2 bank had identical index bytes and exact path/size inventory in both locations.
- Two unstaged `.orig` backups were moved to `/tmp/anomaly-worktree-orig-backup-20260902/`. Worktree removal was not executed because the safety gate rejected deleting about 11 GB without a separate explicit deletion authorization; both worktrees and their branch refs remain available.

## Repository A/B cleanup (2026-09-03)

- With explicit operator authorization, both merged clean worktrees were removed and stale `.git/worktrees` metadata was pruned. Their branch refs were retained. The BMW worktree released about 4 GiB of unique data; its roughly 7 GiB environment content had been hard-linked to the root environment.
- Removed rebuildable caches/logs, the empty `dataset/zs32_0723` tree, the unused root `yolo26n.pt`, the orphan V6 publisher test, the tracked ZS32 `.orig` test backup, inactive key-template model roots, and BMW/ZS32 files under `docs/superpowers/plans` and `docs/superpowers/specs`.
- Removed B-class derived assets: the `artifacts` smoke tree, rejected `results/zs32_patchcore_front_back_top_incremental_seed42_v15`, six unreferenced ZS32 YOLO experiment roots, two verified BMW labeling ZIP duplicates, and two local Git bundle backups. Current ZS32 YOLO v11 and legacy strict/commissioning replay roots were retained.
- Removed the duplicate BMW right trusted-OK `approved_v1` image bank after retaining its two JSON files under `dataset/bmw_trusted_ok_reference/bmw_right_20260810_21_train_normal_approved_v1_metadata`. Current configs continue to use `approved_v2`.
- C-class BMW Demo session outputs were intentionally untouched pending operator review. Raw captures, training datasets, manifests, Label Studio state, active models/configs/ROIs, and the root `.venv` were also preserved.
- Post-cleanup focused tests were `53 passed, 1 deselected`; the deselected repository-asset contract still fails because the pre-existing `configs/zs32/zs32_demo.json` ROI path `dataset/zs32_all_plus_0723_retraining_release_v2/roi_config.json` is absent. This cleanup did not remove that ROI or the also-missing v14 PatchCore assets, and the checkout must not be described as a runnable full ZS32 Demo until those dependencies are restored or the config is corrected.

## BMW-only four-camera runtime consolidation (2026-09-03)

- BMW is now the only product-specific delivery runtime. The supported topology is four serial-bound cameras, a front round plus a back round after flipping the same part, and eight semantic views. Left and right profiles must be selected explicitly; there is no default hand.
- Supported profiles are only `configs/bmw/experiments/bmw_eight_view_demo_left_0820_mixed_v1.json` and the corresponding right profile. The 0814/V6 and 0823 Template40/outside05 candidate configs were removed as historical iterations. Current 0820 profiles still intentionally use the accepted 0823 `front_right` replacements.
- Camera/HDR/pixel conversion now lives under `src/bmw_inspection/capture`; the topology is `configs/bmw/topology/bmw_4cam_eight_view_v1.json`. BMW no longer imports `capture_data.collect_multicamera_dataset`, `zs32_inspection`, a ZS32 topology, the old three-camera chain, or the old single-camera Demo.
- Packaged CLIs are `bmw-inspect` and `bmw-collect`; the two `pipeline/bmw_lab_*` files are compatibility wrappers. `bmw_runtime/pyproject.toml` provides a separate uv environment using the local checkout and the `bmw-lab,cu126` extras. Set `BMW_MVS_SDK_PATH` when MVS is not in its default installation path.
- The delivery tree is runtime-only. Historical BMW training, calibration, A/B, publication/receipt, six-view, and one-off diagnosis modules and their tests were removed. Raw BMW data, current model/checkpoint assets, current masks, trusted-OK banks, and current result directories were not deleted.
- Active BMW runtime code/config contains no SHA, receipt, or provenance gate. ROI configs no longer bind source manifests, and both active profiles plus trusted-OK indexes use relative paths. Missing files, invalid config values, unavailable serials, and incomplete eight-view inputs remain hard errors.
- Focused delivery verification passed: `146 passed` across the remaining BMW tests and the packaged Demo entrypoint integration. Python compilation, root `uv lock --check`, and both package-module `--help` commands passed. `ruff` was unavailable in the current environment. NVML still cannot initialize, so CUDA, real MVS cameras, live GUI, and left/right hardware cycle acceptance remain unverified.
- GitHub publication preflight restored `.label-studio/zs32_defect_840` so raw annotation history is not part of the code deletion. It also removed orphaned numbered training/fusion/inference/Demo wrappers, strict fusion/audit libraries, and tests whose generic filenames still depended on the retired ZS32 workflow. Remaining Python/Shell code has zero `ZS32` or six-view references. The final combined BMW/capture/pipeline suite passed with `212 passed`.
- `prepared_manifest` path resolution is now lazy: a profile can load for live cameras or `--capture-set` when the training manifest is absent, while `--sample-id` still fails at the manifest read boundary. The external BMW asset directory contract is documented in `bmw_runtime/README.md`.
- The dedicated `bmw_runtime/uv.lock` was generated online and `uv lock --project bmw_runtime --check` resolved 121 packages successfully. The GitHub code checkout still requires the documented external model, ROI, mask, and trusted-OK asset trees before real inspection can start.
