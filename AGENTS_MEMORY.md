# AGENTS Memory

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

### ZS32 strict eight-view Stage33 real commissioning attempt (2026-07-15)

- Immutable runtime assets are `results/zs32_runtime_assets_eight_view_v1`; asset-set hash `ea1756c26b45126d86c9d2396b19985efd7073dc6c7e8d13a0744c08ab4b7647` binds the eight PatchCore checkpoints, shared YOLO, exact-eight Template, ROI, and commissioning profile.
- Real Stage33 inference completed all 90 cases under `results/zs32_stage33_eight_view_commissioning_v4`; its run contract binds runtime SHA `27d9fcf9ce69b5037ab917694d27936be856c83842c45877680c7e308e1abd86`, Template SHA `bc5fa70965e756c30030a9e523df927c396f99629cfd3cbe4e6e4bae0a86ebfd`, 90 source records, and the YOLO dataset hash. The artifact is commissioning-only.
- Ultralytics can emit a finite four-coordinate box collapsed to zero area after boundary clipping. Such candidates are now warning diagnostics and are excluded from detection/score; malformed/nonfinite boxes still fail closed. The observed real candidate was `right:normal:group019/back_right`, confidence about `0.00445`, box `[0,2429,2763,2429]`.
- `threshold_calibration/thresholds.json` has 24 finite part-level thresholds and `template_patchcore_threshold_calibration/thresholds.json` has 16 finite thresholds, but strict Stage33 publication remains blocked: `front_secondary` YOLO annotation calibration has 13 rows all labeled `0` and no positive calibration row. The single positive is in test and must not be used for selection. Complete Label Studio review/finalization before rerunning to a new immutable output directory.
- For the explicitly approved quick commissioning fallback, v4 also contains `yolo_high_precision_auxiliary_test_leakage/thresholds.json`: exact eight views, finite thresholds, `test_used_for_selection=true`, `data_leakage=true`, and a sidecar bound to the v4 run contract. Stage34 may consume it only with `--allow-test-leakage`; it is never production evidence. Threshold SHA is `7f2c45b4bcdb730e1579d6c3c9d405e0b6f8921b3cce0c64b514a71172812edb`.

### ZS32 Stage33 review hardening (2026-07-15)

- Stage33 reuse validates every case against the active runtime, ROI, model, source, and evidence contract; preflight performs the same available read-only checks before GPU initialization.
- Stage31's four reports and Stage33 sidecar are one atomic no-replace generation. Task2 is consumed through its public manifest loader and binds the asset-set SHA, ROI bytes, and Template bytes; regenerate stale immutable generations instead of editing them.

### ZS32 standalone classical surface-operator prototype (2026-07-26)

- Keep this work isolated on branch `feat/zs32-classical-operators-prototype` and worktree `/home/yunjing/anomaly_xingtao_new/.worktrees/zs32-classical-operators`. It is not merged into Stage 32, Stage 18, Dashboard, v10, model configs, or threshold artifacts.
- Core code is `capture_data/zs32_classical_operators.py`: `thin_line` returns Scharr/Laplacian elongated-component evidence; `pit_spot` returns multi-scale top-hat/black-hat compact-component evidence. Both produce uncalibrated continuous scores, masks, components, and overlays only—never final inspection labels.
- Offline runner is `capture_data/zs32_classical_benchmark.py`, exposed by `pipeline/38_benchmark_zs32_classical_operators.py`. It consumes an explicit crop manifest, streams cases one at a time, retains bounded timing samples, rejects duplicate identities/views, and atomically publishes a no-clobber diagnostic generation with Linux `RENAME_NOREPLACE`.
- The measured artifact is `/home/yunjing/anomaly_xingtao_new/results/zs32_classical_operator_benchmark_v1`: 30 full-resolution real ROIs, 60 masks, 60 overlays, 30 formal rounds. Timing p50 is 0.1691 s for `thin_line`, 0.2429 s for `pit_spot`, 2.8204 s for a six-image serial round, and 1.5489 s for a six-image two-worker round.
- Current visual effect is not approved: fixed geometry, normal texture, high-contrast hole edges, and angled-view background create substantial false evidence. Six-view normal `thin_line` scores exceeded their same-view sampled defect medians, and labels do not distinguish scratch/crack/pit/spot. Add exact per-view material/ignore masks and region-level target-defect labels before any manual merge.
- Detailed command, timing table, label limitations, and observations are in `docs/ZS32_CLASSICAL_OPERATORS_PROTOTYPE.md`.
