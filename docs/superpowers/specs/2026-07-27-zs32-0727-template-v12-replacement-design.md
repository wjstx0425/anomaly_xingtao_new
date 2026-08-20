# ZS32 0727 Template v12 替换设计

日期：2026-07-27

## 目标

只使用 `dataset/zs32_0727` 中 19 个不同且人工确认正常的右手零件，训练新的八视角
Template 模型，并在验证通过后替换检测平台默认 Template。PatchCore、YOLO 和 ROI
保持 v11 不变；原 v11/v10 bundle 保留用于回滚。

## 已确认的数据事实

- `group001` 至 `group019` 对应 19 个不同实体零件，且均为合格正常件。
- 每个零件有 8 个 canonical views，共 152 张 PNG。
- `group020` 是取消采集的不完整记录且没有图片，必须排除。
- 所有图片均为 RGB `4024x3036`、可解码、无精确文件重复。
- 数据只有 normal，没有 defect，也没有 ground-truth mask。
- 当前 v11 Template 在新数据上通过 118/152；`back_secondary` 19/19 均为
  `NG_TEMPLATE`，说明新采集域与旧 Template 存在显著偏移。

## 数据准备

原始 `dataset/zs32_0727` 保持只读。使用现有 v11 ROI：

`dataset/zs32_all_plus_0723_retraining_release_v2/roi_config.json`

其 SHA256 和版本必须继续绑定为：

- SHA256：`9412b2838cdb96f722db356714cf9bbbb5ea01810671ccf92b67323de77ebe65`
- ROI version：`zs32-eight-view-roi-9412b2838cdb`

按固定 seed 42，以实体零件为单位划分：

| Role | Parts | Images per view |
|---|---:|---:|
| train | 11 | 11 |
| model_val | 3 | 3 |
| calibration | 3 | 3 |
| final_test | 2 | 2 |

同一实体的八个视角必须处于同一 role。输出显式 `template_manifest.csv`、
`physical_part_splits.csv`、数据摘要和 SHA256 receipts；不得使用隐式图片级随机划分。

## Template 训练

复用平台当前消费的 `pipeline/train_zs32_template_gate.py`：

- 输出：`results/zs32_template_gate_right_0727_eight_view_v12`
- required hand：`right`
- views：八个 canonical views，包含两个 secondary
- width：512
- max shift：12
- templates per group：5
- normal quantile：1.0
- explicit manifest roles，`evaluation_fraction=0`
- model version：`zs32-right-0727-eight-view-template-v12`
- threshold version：`zs32-right-0727-temporary-threshold-v12`
- template version：`zs32-right-0727-eight-view-template-v12`

每个视角从 11 个 train 零件中选出 5 张 Template，共 40 张。3 个 calibration
零件只生成初始 normal-only 阈值，2 个 final-test 零件不参与模板或阈值选择。

## 模型验收

- `model.sha256` 必须与 `model.json` 的真实 SHA256 一致。
- 新进程 `load_model()` 必须成功。
- 模型必须精确包含 right/eight-view 8 组及 40 个可验证模板。
- 每视角输出 model-val、calibration、final-test 的 risk min/median/max 和 PASS/NG。
- final-test 若存在正常件 NG，必须如实报告，不允许使用 final-test 反向调阈值。
- 因为没有 defect，模型和 bundle 必须保持 `commissioning_only=true`、
  `production_release_allowed=false`，不能声明缺陷 recall、F1 或 AUROC 已验证。

## 平台替换

不覆盖 v11，发布独立 v12 链：

```text
Template v12
  -> runtime assets template-0727-v12
  -> 24-group thresholds template-0727-v12
  -> runtime bundle template-0727-v12
```

v12 中只改变八组 Template 资产及其阈值：

- PatchCore checkpoint、summary 和阈值保持 v11。
- YOLO `best.pt`、标签语义、`imgsz=1280` 和阈值保持 v11。
- ROI 文件字节及版本保持 v11。
- 八个视角均要求 Template、PatchCore 和 YOLO，共 24 个 required groups。
- 两个 secondary 不得 skipped。

先通过 Stage35 显式 `--runtime-config` 使用 v12 做双面现场 smoke。通过后再把以下默认
路径从 v11 切到 v12：

- `pipeline/35_run_zs32_live_commissioning.py`
- `src/zs32_inspection/dashboard/live.py`

## 回滚与完成条件

- v12 final bundle 在 fresh process 中加载成功，精确包含 24 个 threshold records。
- 八条 Template threshold 均绑定新 Template 模型 SHA。
- PatchCore、YOLO、ROI SHA 与 v11 相同。
- v11 和 v10 bundle 均可继续 fresh-load。
- 默认 CLI 和 Dashboard 指向 v12，显式传入 v11 可立即回滚。
- 更新根目录和 pipeline 的 `AGENTS_MEMORY.md`，记录数据契约、版本路径、限制和回滚方式。
