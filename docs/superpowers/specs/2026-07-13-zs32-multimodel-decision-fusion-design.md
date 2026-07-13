# ZS32 六视角多模型决策融合与人工复检设计

## 目标与决策边界

ZS32 检测系统包含六个按视角独立训练的无监督模型、一个覆盖六视角的单类 YOLO 模型，以及若干传统视觉算子。系统先检测正面三个视角，再翻面检测背面三个视角。设计目标是以防止漏检为第一优先级，并在不确定时进入人工复检，而不是通过多数投票强行输出 OK/NG。

系统内部保留 `OK`、`NG_*`、`REVIEW`、`RETAKE` 和 `INVALID_CAPTURE`。只有六个必需视角及所有必需分支均有效且明确正常时才允许输出 `OK`。任一强异常证据直接输出对应的 `NG_*`；任一灰区、冲突或系统不确定性至少输出 `REVIEW`。

本设计复用 `capture_data/fusion_engine.py` 和 `pipeline/18_fuse_inspection_results.py` 的现有 fail-closed 骨架，但需要扩展双阈值、逐视角完整性、证据链和正背面状态机。第一版不采用多数投票、跨模型原始分数求和或黑箱 stacking。

## 方案选择

### 不采用多数投票

六个无监督模型观察的是六个不同视角，不是对同一证据进行六次独立测量。真实缺陷可能只在一个视角可见，因此一个视角 NG 不能被其他五个视角 OK 抵消。YOLO、无监督模型和传统算子检测的对象与失效模式也不同，不能作为等权票处理。

### 不采用纯二态 OR

任一阳性直接 NG 能提高召回率，但接近阈值的正常波动会导致大量误杀。系统既然支持人工复检，应把明确异常与不确定证据分开。

### 采用分层门控、双阈值和三态决策

推荐顺序为：

1. 身份、输入完整性和图像质量门；
2. 每个模型或算子的独立双阈值判定；
3. 任一强阳性直接 NG；
4. 任一灰区或冲突进入 REVIEW；
5. 所有必需证据明确正常后才输出 OK。

这一方案具有单调性：新增异常证据只会使结果保持或变得更严格，不会把 NG/REVIEW 降级为 OK。

## 身份、完整性与质量门

每个物理工件必须有稳定的 `part_id`，并绑定 `hand`、采集 session 和 group。正面与背面翻转前后必须保持同一身份。六个标准视角为：

- `front`
- `front_left`
- `front_right`
- `back`
- `back_left`
- `back_right`

进入缺陷融合前必须满足：

- 六个视角全部存在且没有重复或错配；
- 每张图通过模糊、曝光、遮挡和 ROI/配准检查；
- 对应模型完成推理，没有超时或加载失败；
- 模型版本、阈值版本和 ROI 配置版本与当前产品 profile 匹配；
- 原始图、实际推理输入和输出记录能够通过 `part_id` 与哈希相互追溯。

状态规则为：

- 缺少必需视角或身份错配：`INVALID_CAPTURE`；
- 图像质量不合格：`RETAKE`；
- 推理失败、证据写入失败或配置版本不匹配：`REVIEW`；
- 全部通过后才进入缺陷证据融合。

这些状态都不得静默转换为 OK。

## 分支双阈值

每个分支保留原始连续分数 `s`，并配置独立的 `T_low` 和 `T_high`：

```text
s < T_low                -> CLEAR
T_low <= s < T_high      -> GRAY
s >= T_high              -> STRONG
```

阈值必须绑定到具体的产品型号、左右手、视角、分支、模型版本和 ROI 版本。不同模型或不同视角的原始分数不可直接比较或相加。

所有分支至少输出：

```text
part_id, hand, side, view, branch,
raw_score, normalized_risk,
low_threshold, high_threshold, decision,
model_version, threshold_version, roi_version,
source_path, evidence_path, reason
```

第一版允许 `normalized_risk` 为空，直接使用每个分支独立标定的原始分数和阈值。后续只有在积累充分标注数据后，才考虑以 isotonic、Platt scaling 或正常分布尾部概率生成可比较风险值。

## 各类算法语义

### 六个无监督模型

- 任一视角为 `STRONG`：输出 `NG_ANOMALY`；
- 任一视角为 `GRAY`：至少输出 `REVIEW`；
- 六个视角全部为 `CLEAR` 时，无监督分支才整体通过；
- 其他视角的低分不能抵消一个视角的高分；
- 保存 anomaly heatmap、叠加图、原始分数、双阈值和阈值余量。

### YOLO

YOLO 不能只输出是否存在检测框。每个检测框保存类别、置信度、坐标、面积、视角以及可视化证据。分支判定为：

- 置信度达到高阈值，且位置、面积和 ROI 约束合理：`STRONG`，输出 `NG_YOLO`；
- 置信度位于灰区，或框靠近 ROI 边界、尺寸异常：`GRAY`；
- 无框或全部低于低阈值：仅代表 YOLO 分支 `CLEAR`，不能抵消其他分支的阳性证据。

YOLO 阈值按缺陷类别、视角和必要的尺寸区间分别标定，不直接沿用通用的 `0.25` 或 `0.5`。

### 传统算子

- 明确的缺件、孔位缺失、轮廓严重变形或确定性公差越界：`STRONG`，输出 `NG_GEOMETRY` 或对应规则状态；
- 接近公差边界或配准不稳定：`GRAY`；
- 安全范围内：`CLEAR`；
- 保存测量值、允许范围、超限余量、规则名、模板版本和可视化证据。

传统几何算子输出的是 `geometry_delta`、`missing_mask`、`extra_mask`、`corner_delta` 或 `shape_delta` 等几何证据，不应把几何命中冒充为裂纹或表面缺陷类型识别正确。

## 零件级融合规则

零件级规则保持简单、可审计和 fail-closed：

```python
if missing_view_or_identity_mismatch:
    return INVALID_CAPTURE
if quality_failed:
    return RETAKE
if inference_failed_or_version_mismatch_or_evidence_failed:
    return REVIEW
if any_required_branch_is_strong:
    return NG
if any_required_branch_is_gray_or_conflicting:
    return REVIEW
if all_required_views_and_branches_are_valid_and_clear:
    return OK
return REVIEW
```

当多个强阳性同时出现时，最终状态可以按配置优先级选择一个主要 `NG_*`，但必须在证据链中保留全部触发分支，不能像当前 first-trigger 输出一样丢失次要证据。

## 正面与背面状态机

正面三个视角完成后只输出阶段状态：

- `FRONT_CLEAR`
- `FRONT_REVIEW`
- `FRONT_NG`

正面阶段不允许输出最终 OK。翻面后验证身份一致，再检测背面三个视角并执行六视角最终融合。

最终状态组合为：

| 正面 | 背面 | 最终状态 |
|---|---|---|
| NG | 任意 | NG |
| 任意 | NG | NG |
| REVIEW | CLEAR | REVIEW |
| CLEAR | REVIEW | REVIEW |
| REVIEW | REVIEW | REVIEW |
| CLEAR | CLEAR | OK |

正面强 NG 后默认仍完成背面检测，以保存完整缺陷画像和支持各视角召回统计。如果节拍要求提前停止，记录必须包含 `inspection_complete=false` 和 `early_stop_reason`，且该记录不能作为六视角完整检查样本参与背面召回统计。

## 人工复检证据链

复检界面以一个物理工件为单位展示两行六视角：正面三视角和背面三视角。每张图使用 CLEAR、GRAY、STRONG 和无效四类状态，并优先展开触发最终结果的视角。

无监督证据包括原图、实际 ROI、heatmap、叠加图、原始分数、双阈值、阈值余量和版本。YOLO 证据包括原图、带框图以及每个框的类别、置信度、坐标和面积。传统算子证据包括测量值、公差、越界余量、配准信息、规则名和轮廓或关键点可视化。

机器可读审计记录使用按零件嵌套的 JSON，统计分析使用一行一个分支的 CSV。JSON 至少包含：

- `schema_version`、`part_id`、`hand`、session 和时间戳；
- `inspection_complete`、机器最终状态和全部触发证据 ID；
- 六个视角各自的源图路径、图像 SHA-256、质量状态和分支结果；
- 模型、阈值、ROI 与模板版本；
- 所有原始分数、双阈值、判定、原因和证据路径；
- 人工复检状态、复检人、时间、结果和备注。

人工复检结果限定为：

- `CONFIRMED_NG`
- `CONFIRMED_OK`
- `RETAKE_REQUIRED`
- `UNRESOLVED`

必须分别保存 `machine_status`、`review_status` 和 `released_status`。人工结果不能覆盖原始机器结论和证据。

## 阈值标定

训练集、阈值标定集和最终锁定测试集必须按物理 `part_id` 分组。一个工件的六个视角、镜像和其他衍生样本不得跨集合泄漏。左右手、视角、缺陷类型、日期和批次应分层覆盖。

`T_low` 控制自动放行，应首先满足漏检约束：真实缺陷即使不够强，也应尽可能落入 GRAY 而不是 CLEAR。`T_high` 控制自动拒绝，应使用明确缺陷和正常压力集标定，减少正常波动直接进入强 NG。两者之间的不确定区交由人工复检。

正常样本不足时可先基于独立正常标定集的尾部分位数建立阈值；缺陷标注充分后，再根据零件级召回约束调整。左手若只有合成正常训练数据，不能据此宣称左手正常误报率达到上线要求，必须使用真实左手正常验证和测试样本。

不得为了降低复检率在线自动放宽 `T_low`。检测到正常分数漂移、相机工况变化或校准失效时，应增加 REVIEW/RETAKE，而不是提高自动放行概率。

## 验证与上线门槛

主指标按物理零件统计，而不是按图片统计：

- 零件级漏检率和召回率；
- 各缺陷类型、左右手、正背面和六视角的最差分组召回率；
- 正常件误杀率；
- 人工复检率、重拍率和自动直通率；
- NG precision；
- 平均与 P95 检测时延。

不能只报告测试集 100% 召回。必须同时给出缺陷样本数量和二项分布单侧置信界限；样本不足时结论应为证据不足，而不是已证明零漏检。

上线前必须进行故障注入，至少覆盖缺视角、错 `part_id`、重复图片、模糊、过曝、ROI 失败、模型超时、YOLO 空框、版本不匹配、单相机漂移和证据文件写入失败，并验证这些情况均不会误放为 OK。

第一版上线后收集所有 REVIEW 的人工结果和正常误报样本。只有当数据充分时，才考虑在 GRAY 区增加受约束的单调聚合器；其权重必须非负，并按同相机、相邻视角和同类算法等相关组限制累计贡献。强阳性和严格 OK 门不由学习型聚合器覆盖。

## 与现有代码的衔接

现有 `capture_data/fusion_engine.py` 已支持标准化分支记录、必需视角、质量/配准门、分支优先级、近阈值 `SUSPECT` 和 `OK/NG/RETAKE/INVALID_CAPTURE` 输出。`pipeline/18_fuse_inspection_results.py` 已支持 manifest、各类分支 CSV 和额外 `BRANCH=PATH` 输入。

后续实现应在此基础上：

1. 扩展分支结构以保存低/高阈值、分支三级状态和版本信息；
2. 将必需视角和质量 PASS 校验落实到逐视角，而不是任意一行 PASS；
3. 保存全部触发证据，而不只保存第一个优先分支；
4. 增加正面阶段状态和六视角最终封单状态机；
5. 写出零件级 JSON 审计记录与分支级 CSV；
6. 为人工复检提供六视角证据索引和稳定的复检结果字段；
7. 增加阈值标定、零件级评估和故障注入测试。

本设计不包含模型训练本身、人工复检前端的具体视觉样式，也不在第一版引入学习型 stacking。具体文件修改和测试任务将在用户审核本设计后写入独立实现计划。
