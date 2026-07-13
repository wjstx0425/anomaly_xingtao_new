# Pipeline Memory

## Cropped six-view PatchCore retraining

- `run_patchcore_roi_six_views.sh` is the minimal user-run entrypoint.
- Defaults: data `dataset/zs32_patchcore_roi`, output `results/six_view_roi_fixed_seed42`, GPU `0`, and `HF_HUB_OFFLINE=1`.
- It verifies `crop_manifest.csv`, then reuses `run_wrn50_fixed_six_views.sh` for serial, resumable right-side six-view PatchCore training with the existing fixed seed-42 configuration.
- Run from the repository root with `bash pipeline/run_patchcore_roi_six_views.sh`.
- Before the directory-preserving recrop, all old training processes/services and old ROI results were stopped/removed. The regenerated dataset contains 2010 images, preserves every original hand/view directory, and records zero corrected views. Start retraining fresh with the runner.

## C789 grayscale multi-template matcher prototype

- `train_template_matcher.py` and `predict.py` implement a standalone OpenCV image-level baseline; they are not part of the numbered pipeline and do not use anomalib `Engine` or checkpoints.
- Training groups already-cropped images by `left_top|left_bottom` and `slotNN`. It selects representative normal templates using a median prototype plus farthest-first thumbnail sampling.
- Matching uses grayscale resize, a `3x3` Gaussian blur, and `cv2.TM_CCOEFF_NORMED` over a reflected `+-shift` translation window. The score is the maximum similarity across a slot's templates; lower than the group threshold means `defect`.
- The default `normal_quantile=0.995` produces a lower-tail threshold at quantile `0.005` from the same `normal_test` images later included in metrics. Those normal metrics are calibration-set results, not an independent test estimate; a missing `normal_test` group silently falls back to `0.8`.
- Outputs are `dataset_index.csv`, template PNGs, `model.json`, `predictions.csv`, and `metrics.json`. `predict.py` requires an already-cropped image plus explicit `--view` and `--slot` and prints one JSON result.
- Keep this baseline distinct from the canonical stage 11-15 geometry workflow in `capture_data/geometry_shape.py`: this prototype compares whole-image grayscale appearance and produces one binary score, while geometry templates compare aligned masks and produce interpretable `missing_mask` / `extra_mask` evidence.
- Known prototype boundaries: no ROI/slot detection, no defect localization/type, no dedicated README/tests/config, no friendly checks for missing groups/templates, direct aspect-ratio coercion on shape mismatch, and direct-script-only import style for `predict.py`.

## ZS32 template-first inspection gate (2026-07-13)

- Production ZS32 gate code is now merged into main and remains independent from the separate C789 prototype scripts.
- `train_zs32_template_gate.py` trains 12 whole-view `(hand, view)` groups from the stage-30 ROI `crop_manifest.csv`; use `--path-root /home/yunjing/anomalib` when the data remains in the main workspace.
- Template-fit normals and threshold normals are disjoint by physical `part_id`. Risk is `1 - similarity`; `T_low` is the minimum calibration defect risk and `T_high=max(T_low, normal quantile)`.
- Training writes `model.sha256` and same-source `calibration_rows.csv`; Stage 31 must combine those template rows with the other five branch families so its locked template thresholds exactly reproduce the online model values.
- `predict_zs32_template_gate.py` returns exit code 0/10/20/2 for PASS/REVIEW/NG_TEMPLATE/invalid gate. It retains similarity, risk, both thresholds, best offset, best-template path and SHA-256, and all deployment versions.
- `capture_data/zs32_inspection_orchestrator.py` is the execution short-circuit. It validates all six inputs, runs `template_match` in canonical order, and calls the injected downstream runner only after six explicit, numerically consistent PASS results.
- The downstream runner receives both `InspectionRequest` and the six normalized template results. Use `template_results_to_branch_rows` or `write_template_match_csv` to preserve them for Stage 18.
- PASS requires four deployment versions plus a readable best-template path whose SHA-256 matches the declared hash. Stage 18 preserves that value as `evidence_hash` and verifies it against the evidence file.
- A downstream `OK` is valid only when `inspection_complete=true`; unknown status or incomplete OK becomes REVIEW.
- Strict Stage 18 requires nonblank 64-hex source/evidence hashes on every required row and recomputes both before release.
- Early-stop audits retain the human-review lifecycle fields (`PENDING` review, null review/release statuses) because Stage 18 is not called after a template non-PASS.
- A non-PASS template result never fabricates downstream CLEAR evidence: it records skipped views/branches and returns REVIEW or NG_TEMPLATE with `inspection_complete=false`.
- Strict fusion now requires six branches per view, including `template_match`; the left/right contract is 72 exact groups. `template_match` is ordinary anomaly evidence, not a quality/registration RETAKE gate.
- `/tmp/zs32-template-gate-smoke-*` models are reduced-width smoke artifacts only and must not be deployed.
- Local merge verification after combining Stage 30 PatchCore and template-first fusion: `333 passed, 1 warning`; the backup branch is `backup/main-before-zs32-fusion-20260713`, and nothing was pushed to GitHub.
