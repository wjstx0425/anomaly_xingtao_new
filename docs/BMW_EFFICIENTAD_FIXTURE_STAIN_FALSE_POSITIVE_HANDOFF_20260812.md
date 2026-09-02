# BMW EfficientAD 夹具与小污渍误报优化交接

日期：2026-08-12
适用目录：`/home/yunjing/anomaly_xingtao_new/.worktrees/bmw-eight-view-handoff`
当前分支：`agent/bmw-21only-diagnostics`
交接基线提交：`268a9b67eb5ecf0040bedf1a971793924c3635b9`

## 1. 任务目标

当前实验 Demo 的 EfficientAD 有两类明显误报：

1. 很小、业务上允许的表面污渍会被当作缺陷；
2. 当前矩形零件 ROI 包含夹具和零件外空白区域，模型会在这些非零件区域产生较高异常分数。

目标是在不破坏 Template、光痕、YOLO 和现有 Demo 的前提下，制作一个独立的 EfficientAD V4 候选：

- EfficientAD 的最终分数只由有效零件表面贡献；
- 用户确认允许的小污渍作为 EfficientAD 分支的正常变化处理；
- 不通过继续盲目提高阈值掩盖误报；
- 缺件、错位仍由 Template/零件存在性检查失败关闭，不能因为屏蔽夹具而得到 OK；
- 所有新数据、掩膜、模型、阈值和 Demo 配置采用新版本目录，不覆盖 V1/V2/V3。

这是实验室快速迭代任务，不是工业验收或生产发布。

## 2. 当前真实基线

### 2.1 当前建议复现的 Demo

- V3 配置：`configs/bmw/experiments/bmw_eight_view_demo_v3_ng_evidence.json`
- 配置 SHA-256：`1b854790f51d0de7bb7d2c73c670e38c04a8dd0e2bb362c0e287375d97c33d96`
- 运行结果：`results/bmw_eight_view_demo_v3_ng_evidence_v1`
- 当前结果索引包含 40 次检测：16 次 OK、24 次 NG。现场样本没有完整、独立的业务标签，因此这个比例不能当作假阳性率或模型准确率。

复现命令：

```bash
cd /home/yunjing/anomaly_xingtao_new/.worktrees/bmw-eight-view-handoff

MPLCONFIGDIR=/tmp/bmw-mpl-cache \
uv run --no-sync python pipeline/bmw_lab_eight_view_demo.py \
  --config configs/bmw/experiments/bmw_eight_view_demo_v3_ng_evidence.json \
  --experiment-mode
```

### 2.2 当前 EfficientAD

- 八个 checkpoint：`results/bmw_lab_one_click/bmw_right_batch_20260810_21_efficientad_v1/efficientad/<view>/model.ckpt`
- 训练报告：`results/bmw_lab_one_click/bmw_right_batch_20260810_21_efficientad_v1/efficientad/training_report.json`
- V3 阈值资产：`results/bmw_lab_one_click/bmw_right_batch_20260810_21_v3_ng_evidence_demo_v1/efficientad_thresholds_deployment_v3.json`
- 阈值资产 SHA-256：`1c7e9f9e0f1750aabba62846f816d0cd273f23047e2ce6a6100e3c43fde2417e`

V3 阈值约为每个视角 `0.55`，来源是 21 个 `normal_test` 零件的候选阈值再统一增加 `0.05`。该资产明确记录：

- `test_used_for_selection=true`；
- `defect_metrics=not_evaluated`；
- `defect_part_count=0`；
- `candidate_only=true`；
- `demo_only=true`。

因此，不能继续依据这 21 个正常件上调阈值并宣称问题已经解决，也不能从该资产得出缺陷召回率结论。

当前运行时在 `src/bmw_inspection/lab/eight_view_demo_models.py` 中直接使用 Anomalib 返回的整张矩形 ROI `pred_score` 判定：

```text
score >= per_view_threshold => NG
```

`anomaly_map` 目前只用于诊断热图和最大热点标记，没有前景掩膜，也没有仅在零件表面进行分数聚合。现有 `inspection.json` 保存了分数和热点坐标，但没有保存原始 anomaly map；需要基于已保存的 `rois/<view>.png` 重新推理，或者先增加原始 map 的 `.npy` 持久化，才能做可靠的离线重算。

### 2.3 当前 ROI 和可信 OK 参考

- 公共矩形 ROI：`configs/bmw/rois/bmw_right_hdr_eight_view_v1.json`
- ROI SHA-256：`0dab057714cd51eee937297550afdcfd8c3290440dbfe40086a9c320382c6a53`
- ROI 被 Template、YOLO 和 EfficientAD 共同使用，不应直接缩小或覆盖。
- 可信 OK 参考：`dataset/bmw_trusted_ok_reference/bmw_right_20260810_21_train_normal_approved_v2`
- 参考索引：上述目录的 `reference_index.json`，SHA-256 为 `ae7833ab35cbc76cbfef6cfa5163f77ef345d879a6e8e4834fa8cbfcf6023acc`。
- 可信 OK 库有 50 个完整零件、八视角各 50 张，可用于制作/检查固定前景掩膜和诊断对比；它来自训练正常件，不能同时当作独立最终验收集。

## 3. 根因假设与验证顺序

按优先级验证，不要一开始就重训：

1. **夹具/背景进入整图分数**：当前最明确的结构性问题。矩形 ROI 内的非零件像素没有被屏蔽，且 `pred_score` 针对整张输入。
2. **允许污渍没有进入正常分布**：如果训练正常图中缺少这类表面状态，EfficientAD 将其视为异常是符合模型行为的。
3. **单点/极小区域过度主导分数**：当前分数可能被少量高响应像素主导；需要比较 masked max、高分位数、top-k mean 和最小连通区域规则。
4. **姿态或光照漂移**：固定工装不等于完全没有亚像素位移或 HDR 分布变化。先对热点与可信 OK 做配准/亮度差检查。
5. **单纯阈值过低**：只能在完成上述验证并有独立缺陷集后判断。V3 已经人为增加 `0.05`，继续加阈值有明显漏检风险。

必须先把用户指出的误报样本按以下原因分类并保存清单：

- `fixture_or_background`：热点不在零件表面；
- `acceptable_stain`：热点在零件表面，但用户明确允许；
- `visible_defect`：用户确认是真缺陷；
- `pose_or_light_drift`：整体位置或灰度分布漂移；
- `uncertain`：不能仅凭热图判断。

热图只能叫“诊断热区”，不能当作精确缺陷分割结果。

## 4. 方案比较

### 方案 A：只缩小八个矩形 ROI

优点是实现最快。缺点是零件轮廓不规则，矩形仍会包含夹具；同时公共 ROI 绑定 Template/YOLO，直接修改会造成多分支资产失配，也可能裁掉边缘缺陷。只适合做快速对照，不建议作为最终方案。

### 方案 B：只加入允许污渍图并重新训练

能改善污渍容忍，但夹具仍参与训练与推理，背景高分问题不会根治。样本量不足时还可能只是记住几种污渍外观。

### 方案 C：EfficientAD 专用前景掩膜 + 允许污渍正常样本 + 重新标定分数（推荐）

保留公共矩形 ROI，另建八视角 EfficientAD 专用二值前景掩膜。训练和推理使用完全相同的掩膜预处理；推理时只在有效零件表面聚合 anomaly map。然后将用户确认允许的污渍作为 EfficientAD 分支正常样本，并保留独立污渍验证件和真实缺陷件。

该方案同时解决两个问题，且不会影响 Template、YOLO 和光痕。代价是需要制作八张掩膜、重训八个模型并重新标定阈值。

## 5. 推荐实施路径

### 阶段 1：无重训快速 A/B，先证明问题

1. 从 `results/bmw_eight_view_demo_v3_ng_evidence_v1/*/inspection.json` 和对应 `rois/*.png` 中选出用户确认的误报案例。
2. 记录每个案例的 capture ID、视角、原分数/阈值、热点坐标和业务类别。
3. 基于 50 个可信 OK 参考制作八视角初版零件前景掩膜；掩膜必须人工预览，不能只依赖一次自动分割。
4. 用当前 checkpoint 对保存的 ROI 重新推理，导出原始 anomaly map `.npy`。
5. 同一张 map 同时计算：
   - 当前 `pred_score`；
   - 前景内最大值；
   - 前景内高分位数；
   - 前景内 top-k mean；
   - 满足最小面积后的最大连通区域分数。
6. 输出基线与候选的逐视角、逐零件 A/B 表，并单独统计热点落在夹具区域的次数。

如果掩膜后夹具误报没有明显下降，先检查配准、掩膜坐标和模型归一化，不要直接进入重训。

### 阶段 2：建立 EfficientAD 专用掩膜契约

建议新增独立资产，不修改公共 ROI：

- 八张原始 ROI 尺寸的单通道 PNG，`255=有效零件表面`、`0=忽略区域`；
- 一个 JSON 索引，记录视角、ROI 尺寸、mask 路径/SHA-256、前景比例、来源参考和版本；
- 掩膜必须精确覆盖八个标准视角，尺寸不符、缺失、非二值或 SHA 不符时启动失败；
- 零件边界保留可配置容差带。夹具必须排除，但不能未经验证就排除需要检测的零件边缘。

训练与推理必须使用同一预处理：公共 ROI 裁剪后，将掩膜外像素填充为固定背景值，再送入 EfficientAD。不能只在推理时遮挡，否则训练/推理分布不一致。

### 阶段 3：正常语义和数据拆分

为用户允许的小污渍建立明确类别，例如 `acceptable_stain`，但在 EfficientAD 分支映射为 normal。不要把“面积小”自动等同于“允许污渍”。

按物理零件拆分，避免同一件进入训练和验证：

- `train_normal_clean`；
- `train_normal_acceptable_stain`；
- `calibration_normal_clean`；
- `calibration_normal_acceptable_stain`；
- `final_normal_clean`；
- `final_normal_acceptable_stain`；
- `final_visible_defect`；
- `uncertain_review`。

`no_streak` 对 EfficientAD 表面分支可以是正常外观，但它在业务融合中仍由光痕分支判为 NG，不能改成整件 OK。

### 阶段 4：V4 重训和阈值

1. 发布新的 masked training release，保留源图、公共 ROI、mask 版本、物理零件拆分和 SHA。
2. 训练八个新的 EfficientAD-S checkpoint，不覆盖 21:00 V2 checkpoint。
3. 只用 calibration 选择每视角分数聚合方式和阈值；final test 只报告，不参与选择。
4. 同时报告：
   - 清洁正常件整件误拒率；
   - 允许污渍整件误拒率；
   - 真实缺陷整件/逐视角召回；
   - 夹具区域热点率；
   - 每视角分数分布和最难样本；
   - 与当前 V3 的同图 A/B。
5. 现有约 5% 的整件正常误拒目标可以作为实验参考，但只有独立标注集才能形成结论。

### 阶段 5：独立 Demo 集成

建议建立 V4 配置，例如：

```text
configs/bmw/experiments/bmw_eight_view_demo_v4_efficientad_masked.json
```

V4 只替换 EfficientAD checkpoint、前景掩膜、分数契约和阈值资产。Template、光痕、YOLO、HDR 参数和公共 ROI 保持 V3 身份不变。配置必须固定：

- 八个 checkpoint SHA；
- mask 索引 SHA 和八张 mask SHA；
- 分数聚合算法与参数；
- 阈值资产 SHA；
- 训练数据 release 和代码身份。

证据目录同时保存原 ROI、前景掩膜、masked ROI、原始 anomaly map、masked heatmap、分数构成和最终判定。旧 V1/V2/V3 配置与结果不得修改。

## 6. 最小验收条件

在没有用户提供最终数值标准前，不虚构工业指标。最小技术验收为：

1. 八视角掩膜都经过可视化检查，夹具不参与 EfficientAD 分数；
2. 用户确认的误报样本完成 V3/V4 同图 A/B；
3. 允许污渍验证集的通过率明显改善；
4. 已标注可见缺陷集不低于 V3 同图结果；若 V3 对该集合没有有效结果，必须明确写成“新基线”，不能声称无回退；
5. final test 没有参与阈值或聚合器选择；
6. 缺件/错位仍由 Template 或单独存在性检查失败关闭；
7. V4 为独立、无覆盖、SHA 绑定的实验配置；
8. 聚焦测试、离线同图 A/B 和至少一次现场正常/污渍/缺陷烟雾测试均留下报告。

## 7. 禁止事项

- 不要在脏的根分支 `/home/yunjing/anomaly_xingtao_new` 直接实现；使用 BMW 独立 worktree。
- 不要覆盖 V1/V2/V3 模型、阈值、ROI、配置或结果。
- 不要直接修改公共 `bmw_right_hdr_eight_view_v1.json` 来影响所有分支。
- 不要把阈值继续加 `0.05` 当成根治方案。
- 不要仅凭热图判断某块是污渍或真缺陷。
- 不要把训练正常件、可信 OK 参考同时用作独立最终验收。
- 不要按图片随机拆分；必须按物理零件拆分。
- 不要为了让 EfficientAD PASS 而允许缺件最终 OK。
- 未经用户要求，不合并根分支、不推送、不启动长时间全量训练。

## 8. 新会话可直接使用的 Prompt

```text
请在 `/home/yunjing/anomaly_xingtao_new/.worktrees/bmw-eight-view-handoff` 中继续 BMW 八视图实验系统工作。请先完整阅读：

1. `AGENTS.md`
2. `AGENTS_MEMORY.md`
3. `docs/BMW_EFFICIENTAD_FIXTURE_STAIN_FALSE_POSITIVE_HANDOFF_20260812.md`

当前目标：解决 EfficientAD 对业务允许的小表面污渍过度敏感，以及公共矩形 ROI 把夹具/零件外区域纳入异常分数的问题。我要的是快速可迭代的实验室版本，不是工业平台。

重要基线：

- 工作分支应从 `agent/bmw-21only-diagnostics`、提交 `268a9b67eb5ecf0040bedf1a971793924c3635b9` 开始核对；如果已经变化，先报告实际 HEAD 和差异，不要重置用户修改。
- 当前问题主要在 V3：`configs/bmw/experiments/bmw_eight_view_demo_v3_ng_evidence.json`。
- 当前公共 ROI 是 `configs/bmw/rois/bmw_right_hdr_eight_view_v1.json`，SHA-256 为 `0dab057714cd51eee937297550afdcfd8c3290440dbfe40086a9c320382c6a53`。
- 当前 V3 阈值资产是 `results/bmw_lab_one_click/bmw_right_batch_20260810_21_v3_ng_evidence_demo_v1/efficientad_thresholds_deployment_v3.json`，它只在21个 normal_test 件上选过阈值，`test_used_for_selection=true`、`defect_metrics=not_evaluated`，不能继续靠抬阈值宣称修复。
- 当前可信 OK 参考是 `dataset/bmw_trusted_ok_reference/bmw_right_20260810_21_train_normal_approved_v2`，可用于掩膜制作和诊断，不能当独立最终测试集。
- 当前 V3 保存结果在 `results/bmw_eight_view_demo_v3_ng_evidence_v1`。`inspection.json` 没有原始 anomaly map，但保留了 ROI 图，可以重新推理。

请按以下顺序工作：

1. 只读审计实际 HEAD、当前配置、checkpoint/阈值/ROI SHA、最近 V3 结果和 EfficientAD 运行时。不要先改代码或训练。
2. 从用户指出的误报样本中区分 `fixture_or_background`、`acceptable_stain`、`visible_defect`、`pose_or_light_drift`、`uncertain`。如果无法从现有记录确定用户说的是哪些 capture ID，只汇总问我一次，不要逐个追问。
3. 先做最小离线 A/B：制作八视角 EfficientAD 专用前景掩膜候选，重新推理保存的 ROI，导出原始 anomaly map，并比较当前 pred_score、前景 masked max、高分位数、top-k mean 和最小连通区域方案。输出逐视角、逐零件报告，证明夹具和小污渍各自贡献了什么。
4. 推荐路线是：保留公共矩形 ROI；新增 EfficientAD 专用、SHA 绑定的八视角前景掩膜；训练和推理使用一致的 mask 外固定填充值；允许污渍经人工确认后作为 EfficientAD normal，同时保留独立污渍验证件和真实缺陷件。
5. 只有离线 A/B 证明掩膜有效后，才发布新的 masked training release、训练八个 V4 EfficientAD-S，并仅用 calibration 选分数聚合器与阈值。final test 只能报告。
6. 建立独立 V4 Demo 配置，只替换 EfficientAD 相关资产；Template、光痕、YOLO、HDR 和公共 ROI 保持 V3 不变。所有新目录无覆盖、记录 SHA，启动失败应关闭。
7. 缺件/错位必须继续由 Template 或存在性检查拦截；前景掩膜不能让缺件最终 OK。
8. 保留原 ROI、mask、masked ROI、raw anomaly map、masked heatmap、分数构成、阈值和判定证据。EfficientAD 热图只称诊断热区。
9. 使用 `uv`，先写聚焦测试再实现；只运行与 BMW EfficientAD/mask/Demo 直接相关的测试和离线样本，不做无关大范围检查。
10. 不修改或覆盖 V1/V2/V3，不改脏的根分支，不合并、不推送。完成后更新这个 worktree 的 `AGENTS_MEMORY.md`，给出具体产物路径、复现命令、A/B 指标和仍未验证的边界。

优先做最小可验证实现，不要另起工业级平台，也不要只写计划后停下。只有涉及用户必须定义的污渍容忍边界或需要点名具体误报 capture ID 时，汇总一次问题再问我。
```
