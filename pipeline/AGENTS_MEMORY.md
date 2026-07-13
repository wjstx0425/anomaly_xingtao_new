# Pipeline Memory

## ZS32 template-first inspection gate (2026-07-13)

- Production code lives in the fusion worktree and does not import the untracked C789 prototypes from the main workspace.
- `train_zs32_template_gate.py` trains 12 whole-view `(hand, view)` groups from the stage-30 ROI `crop_manifest.csv`; use `--path-root /home/yunjing/anomalib` when the data remains in the main workspace.
- Template-fit normals and threshold normals are disjoint by physical `part_id`. Risk is `1 - similarity`; `T_low` is the minimum calibration defect risk and `T_high=max(T_low, normal quantile)`.
- Training writes `model.sha256` and same-source `calibration_rows.csv`; Stage 31 must combine those template rows with the other five branch families so its locked template thresholds exactly reproduce the online model values.
- `predict_zs32_template_gate.py` returns exit code 0/10/20/2 for PASS/REVIEW/NG_TEMPLATE/invalid gate. It retains similarity, risk, both thresholds, best offset, best-template path and SHA-256, and all deployment versions.
- `capture_data/zs32_inspection_orchestrator.py` is the execution short-circuit. It validates all six inputs, runs `template_match` in canonical order, and calls the injected downstream runner only after six explicit, numerically consistent PASS results.
- The downstream runner receives both `InspectionRequest` and the six normalized template results. Use `template_results_to_branch_rows` or `write_template_match_csv` to preserve them for Stage 18.
- PASS requires four deployment versions plus a readable best-template path whose SHA-256 matches the declared hash. Stage 18 preserves that value as `evidence_hash` and verifies it against the evidence file.
- A downstream `OK` is valid only when `inspection_complete=true`; unknown status or incomplete OK becomes REVIEW.
- A non-PASS template result never fabricates downstream CLEAR evidence: it records skipped views/branches and returns REVIEW or NG_TEMPLATE with `inspection_complete=false`.
- Strict fusion now requires six branches per view, including `template_match`; the left/right contract is 72 exact groups. `template_match` is ordinary anomaly evidence, not a quality/registration RETAKE gate.
- `/tmp/zs32-template-gate-smoke-*` models are reduced-width smoke artifacts only and must not be deployed.
