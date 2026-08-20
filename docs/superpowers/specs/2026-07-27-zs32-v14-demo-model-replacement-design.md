# ZS32 v14 Demo Model Replacement Design

## Goal

Replace the active ZS32 Demo Template and eight-view PatchCore artifacts with the
user-selected `0727_plus_defect` v14 results while keeping the existing ROI,
YOLO model, global Template gate, and eight-process PatchCore runtime unchanged.

## Artifact Contract

- Template root:
  `results/zs32_template_gate_right_0727_plus_defect_eight_view_v14`
- PatchCore root:
  `results/zs32_patchcore_eight_view_0727_plus_defect_seed42_v14`
- All eight canonical views must have exactly one configured checkpoint.
- Template runtime thresholds come from each `model.json` group's
  `high_threshold`, because the Demo has a binary PASS/NG Template gate.
- PatchCore runtime thresholds come from `eight_view_summary.csv` field
  `deploy_threshold`.
- The YOLO model and thresholds, ROI configuration, topology, and
  `patchcore_process_count=8` remain unchanged.

## Validation

Load `configs/zs32/zs32_demo.json` through the production parser, verify that all
eight configured checkpoints exist under the selected v14 root, verify all 40
Template PNG SHA-256 values against `model.json`, and run the focused Demo
configuration tests. Update the folder memory with the deployed artifact roots
and thresholds.
