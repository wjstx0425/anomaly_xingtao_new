# ZS32 calibration policy

`fit_strict_v1.json` is a versioned starting policy, not an automatically
approved production threshold.  The Linux run must report the actual number of
independent physical parts in calibration and held-out test.  If any required
template/anomaly/YOLO hand-view group lacks the configured normal or defect
part count, calibration is non-deployable.

The held-out `test` split is evaluated only after fitting and is never used to
choose thresholds.  Physical-part identity must remain isolated across
`train`, `calibration`, and `test`.

`heldout_acceptance_strict_v1.json` is the separate production-candidate gate.
It is strict canonical JSON: do not reformat it.  `zs32-validate-candidate`
hashes its exact bytes, checks minimum held-out normal/defect physical-part
counts, and rejects any defect escape, normal reject, or review outcome in this
starting policy.  A policy breach produces no validated publication.  Changing
any limit requires a new `policy_version`; the exact policy SHA256 and resulting
PASS-decision SHA256 are carried through validation, manual promotion, and the
assembled release provenance.
`calibration_targets.example.json` 是数据集 schema v4 的人工审批目标合同示例。
正式文件必须为每个 canonical crop 精确列出 `template`、`anomaly`、`yolo`
三条记录，并按 `(capture_set_id, part_instance_id, hand, view, branch)` 排序。
`part_ground_truth` 来自零件级人工真值；`target` 是该分支在该视角应使用的
`normal`、`defect` 或 `exclude`。YOLO bbox 只能辅助审核 YOLO target，禁止据此
推断 template/anomaly target。`exclude` 以及 defect part 被标为 branch-normal 时
必须填写 `reason`。
