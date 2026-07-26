# ZS32 v11 模型接入检测平台设计

日期：2026-07-26

## 目标

将以下已训练模型接入 ZS32 八视角检测平台，并将新 bundle 设为 Stage35 与 Dashboard 的默认运行配置：

- Template：`results/zs32_template_gate_right_0723_eight_view_v11`
- PatchCore：`results/zs32_patchcore_eight_view_all_plus_0723_seed42_v11`
- YOLO：`results/yolo/zs32_all_plus_0723_normal_n1280_seed42_v11/weights/best.pt`

接入后采用严格 24-group 合约：八个视角均执行 Template、PatchCore 和 YOLO。现有 v10 20-group bundle
及其引用资产必须保持原样，以便显式回滚。

## 已确认的模型证据

### Template

- `model.json` SHA256：
  `ef0d99139c1294db0681b99e6ef277d9acf41dc2240fd12ddfc551e90407fc6d`
- 8 个视角、每视角 5 个模板，共 40 个模板；文件及内嵌 SHA 均有效。
- normal-only binary gate，阈值来自 13 个 calibration 零件。
- 无缺陷校准样本，因此不能报告 Template defect recall/F1。

### PatchCore

- `eight_view_summary.csv` SHA256：
  `fb6965310efe17cc9671af4dbf0ffc5b894fa5e9c542bf05063c60ec7b9770bb`
- 8 个视角均有独立 checkpoint，八个 checkpoint SHA256 不同。
- 临时 deploy threshold：

| View | Threshold |
|---|---:|
| front | 0.4711672067642212 |
| front_left | 0.5397635102272034 |
| front_right | 0.814327597618103 |
| front_secondary | 0.5006473064422607 |
| back | 0.4759141802787781 |
| back_left | 0.6454828977584839 |
| back_right | 0.5977694988250732 |
| back_secondary | 0.49985870718955994 |

两个 secondary 只有 normal/normal_test，其 recall、F1、缺陷 AUROC 不可解释。旧 `failed_views.txt`
是首次训练失败的陈旧记录；最终 runner log、checkpoint 和 summary 证明两个视角已恢复完成。

### YOLO

- `best.pt` SHA256：
  `1375ee156d10093afd689cc70daf6b243264a4dc49c7cb1c39293cb780434528`
- 最佳 fitness epoch 57，precision 约 0.5008、recall 约 0.1875、mAP50 约 0.1653。
- 产物完整，但 recall 偏低，必须作为 commissioning 限制显示。

## 版本化资产设计

新增以下文件和不可变输出，不覆盖旧版本：

1. `config/fusion/zs32_right_eight_view_24_group_commissioning_v11.json`
   - 八视角均要求 `template_match`、`anomaly_<view>`、`yolo`。
   - 不携带旧 `expected_versions`；Stage37 根据 v11 文件 SHA 自动生成。
2. `config/fusion/zs32_eight_view_24group_v11_bundle_source.json`
   - `bundle_id` 使用独立 v11 ID。
   - ROI 使用
     `dataset/zs32_all_plus_0723_retraining_release_v2/roi_config.json`。
   - ROI version 与 Template 模型一致：
     `zs32-eight-view-roi-9412b2838cdb`。
   - 绑定 v11 Template、PatchCore summary、YOLO best.pt 和 `imgsz=1280`。
3. `results/zs32_runtime_assets_eight_view_v11`
   - 由 Stage37 `publish-assets` 原子发布。
4. `results/zs32_24group_0723_v11_commissioning`
   - 由 Stage34 发布 24 条临时 threshold records。
5. `results/zs32_runtime_bundle_eight_view_v11`
   - 由 Stage37 `finalize` 将 thresholds 与 runtime assets 精确哈希绑定。

所有新产物必须保留：

```text
commissioning_only=true
production_release_allowed=false
```

## 临时阈值策略

当前没有满足 Stage33 严格正负 calibration 要求的 v11 八视角缺陷数据，因此不得把新 bundle 描述为正式校准。

- Template：使用 v11 `model.json` 中每视角 normal-only low/high。
- PatchCore：使用上表中的每视角 deploy threshold，并设置 `low=high`。
- YOLO：沿用用户已确认的临时默认值 `low=high=0.07`。
- Stage34 显式启用 model rebind、YOLO threshold override，并保留旧 YOLO threshold source 的
  test-leakage 标记。
- summary/threshold artifact 必须记录 model rebound、manual override、test leakage 以及 production 禁止。

以后手工调整阈值时发布新的不可变 threshold/bundle 版本，不修改 v11 已发布文件。

## 平台默认值和回滚

完成新 bundle 验证后修改两个默认入口：

- `pipeline/35_run_zs32_live_commissioning.py`
- `src/zs32_inspection/dashboard/live.py`

默认指向：

`results/zs32_runtime_bundle_eight_view_v11/runtime_bundle.json`

回滚不修改代码，显式传入：

`results/zs32_runtime_bundle_eight_view_20group_patchcore_tuned_v10/runtime_bundle.json`

旧 v10 source、profile、assets、thresholds、bundle 和全部引用模型均不得修改或删除。

## 平台数据流

```text
四相机八视角采集
  -> v11 runtime bundle hash preflight
  -> Template v11（八视角）
  -> PatchCore v11（八视角）
  -> YOLO v11（八视角 batch）
  -> 24-group Stage18 fusion
  -> Dashboard 展示 commissioning/non-production 状态
```

任一模型、ROI、profile、threshold 或 SHA 不匹配时 fail closed，不回退到旧模型，也不静默跳过 secondary。

## 指标语义

- 当前没有 ground-truth mask。
- pixel AUROC、AUPRO、pixel F1、pixel IoU 必须标记为 `NOT_APPLICABLE`/`null`，不能写 0。
- 推理产生的可视化 mask 不等于 ground-truth mask，不能据此计算像素级评估指标。
- 现有严格 threshold/release schema 不增加未知字段；N/A 说明写入 commissioning 文档或独立报告。

## 验证

1. Source/profile JSON schema 与 canonical 八视角顺序验证。
2. Stage37 publish-assets 后核对 8 个 PatchCore checkpoint、40 个 Template 文件、YOLO、ROI 的 SHA。
3. Stage34 输出严格 24 条 thresholds，检查所有风险标记。
4. Stage37 finalize 后在全新 Python 进程调用 `load_runtime_bundle`。
5. 回归 Stage34、Stage37、Stage35、Stage32、Dashboard live-controller 测试。
6. 验证默认入口指向 v11，同时显式加载 v10 仍成功。
7. 使用已有离线八视角案例运行 Stage32/Stage18 和无 GUI Dashboard screenshot smoke。
8. 当前 Codex 环境无 CUDA/NVML；GPU 推理 smoke 必须在用户可见 GPU 的终端运行，不能以 skipped 测试替代。

## 验收标准

- 新 v11 bundle 可在 fresh process 中加载。
- bundle 精确包含 8 Template、8 PatchCore、8 YOLO，共 24 个 required groups。
- Stage35 和 Dashboard 默认解析到 v11 bundle。
- 两个 secondary 的 Template/PatchCore 均为 required/available，不是 `SKIPPED`。
- v10 bundle 仍能通过原 SHA 绑定加载。
- 平台明确显示 commissioning-only/non-production。
- 无 GT mask 的像素指标保持 N/A。
