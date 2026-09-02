# BMW 八视图 Template 关键区域加权设计

日期：2026-08-23

## 目标

在不恢复独立关键区检测分支的前提下，让操作员已经划分的关键 ROI 对现有整块 Template 总分产生更大影响。

- 总检测项继续固定为 25 项。
- 每个视角仍然只有一条 `template` 结果。
- 普通区域权重为 `1.0`，关键 ROI 初始统一权重为 `3.0`。
- 右手和左手继续使用各自独立的 ROI 文件、模型和阈值。
- 当前原始 Template 算法和阈值完整保留，可通过一个配置开关回滚。
- 加权模式使用 0823 正常数据重新标定阈值，不复用原阈值。

本设计不修改 Template 模板图片、EfficientAD、YOLO、光痕、公共 ROI、HDR、相机参数或最终融合优先级。

## 方案选择

采用“整块对齐后加权归一化相关系数”。不使用独立子 ROI 匹配，也不把多个局部分数相加。

原因：

1. 保留现有整块 Template 的模板选择和平移对齐能力，避免小 ROI 独立对齐引起过敏。
2. 加权相关系数仍对整体亮度平移较稳定，比加权绝对差更适合当前 HDR 现场图。
3. 关键区域会影响唯一的 Template 总分，但不会单独一票否决。

## 配置合同

现有 `template.thresholds` 原样保留，明确作为旧算法回滚阈值。活动左右配置的 `template` 段新增一个轻量对象：

```json
{
  "template": {
    "models": {"...": "..."},
    "thresholds": {"...": 0.01},
    "weighted_regions": {
      "enabled": true,
      "weight": 3.0,
      "roi_config": "../../key_template/bmw_left_0823_key_rois_v1.json",
      "thresholds": {"...": 0.02}
    }
  }
}
```

字段语义：

- `template.thresholds`：当前原始阈值，永远保留且不被标定工具覆盖。
- `weighted_regions.enabled=false` 或整个对象缺失：完全执行当前原始 Template 分数和阈值。
- `weighted_regions.enabled=true`：对有关键 ROI 的视角计算加权分数并使用 `weighted_regions.thresholds`。
- `weighted_regions.weight`：关键 ROI 像素权重，初始为 `3.0`。
- `weighted_regions.roi_config`：此前左右手独立选择文件，坐标继续使用公共 ROI 局部半开区间 `xyxy`。
- 没有关键 ROI 的视角继续使用原始分数和 `template.thresholds`，即使总开关已开启。

不增加 schema 版本、SHA、receipt、publisher、provenance 或 rebind。

## 加权分数算法

### 1. 保留原始匹配

每个视角首先执行当前实现：

1. 将公共 ROI 灰度化、等比例缩放并反射填充至模型目标尺寸。
2. 对模板执行 `TM_CCOEFF_NORMED` 和 `max_shift` 平移搜索。
3. 若存在 Template ignore mask，继续使用当前 masked match 选择模板和位置。
4. 得到最终选择的模板、平移位置以及原始 `legacy_similarity` / `legacy_risk`。

该步骤保持现有行为，作为回滚结果和诊断基线。

### 2. 构造权重图

在公共 ROI 原始坐标域中：

- 普通有效像素权重为 `1.0`。
- 所有关键 ROI 内像素权重为 `3.0`。
- ROI 重叠时取最大值 `3.0`，不叠加成 `9.0`。
- ignore mask 排除的像素最终权重为 `0.0`，ignore 优先级高于关键 ROI。

权重图采用与 Template 图一致的等比例缩放和 padding 几何，矩形起点向下取整、终点向上取整，保持半开区间覆盖。平移对齐后，从 padding 权重图中切出与 `aligned_query` 相同的区域。

### 3. 计算加权相关系数

令对齐后的查询图为 `Q`，选中模板为 `T`，最终权重图为 `W`：

```text
mu_q = sum(W * Q) / sum(W)
mu_t = sum(W * T) / sum(W)

weighted_similarity =
    sum(W * (Q - mu_q) * (T - mu_t))
    / sqrt(sum(W * (Q - mu_q)^2) * sum(W * (T - mu_t)^2))

weighted_risk = max(0, 1 - weighted_similarity)
```

相关系数限制在 `[-1, 1]`。若有效权重为空或分母为零，该 Template 分支返回明确 `ERROR`。

加权只重算最终选中模板和位置的分数，不重新执行局部 ROI 模板搜索。因此关键 ROI 会提高总分影响，但不会重新引入独立局部对齐的敏感性。

## 阈值重新标定

新增一个只处理 Template 加权分数的轻量标定命令。它直接读取：

- 左右活动 Demo 配置中的公共 ROI、Template 模型和 ignore mask。
- 对应 0823 prepared manifest。
- 左右各自关键 ROI 文件。

数据使用规则：

- `train normal` 不重新训练模板，仅用于记录覆盖情况。
- `calibration normal` 用于拟合加权阈值。
- `final_test normal` 只报告误拒，不参与阈值选择。

每个有关键 ROI 的视角：

```text
weighted_threshold = max(calibration_normal_weighted_risk) * 1.10
```

即在 calibration normal 最大值上增加 10% 实验室波动余量。若最大值为零，则使用下一个大于零的浮点数。没有关键 ROI 的视角直接沿用原始阈值。

标定输出：

- 每视角 calibration 分数、最大值和新阈值。
- final-test 分数、最大值和正常误拒数。
- 建议写入活动配置的 `weighted_regions.thresholds` 数值。

标定工具不修改 `template.thresholds`，不生成发布包或摘要绑定。

## 运行时结果与界面

Template 仍输出一条 `DemoBranchResult`。加权模式的 `details` 增加：

- `score_source=weighted_region_ccoeff_normed`
- `legacy_similarity` / `legacy_risk`
- `weighted_similarity` / `weighted_risk`
- `region_weight=3.0`
- `weighted_region_count`
- `weighted_regions_xyxy`
- `deployment_threshold`
- `threshold_exceedance`

顶层 `score` 使用 `weighted_risk`，顶层 `threshold` 使用新的加权阈值。最终状态仍是 `risk <= threshold` 为 PASS，否则 NG。

Template 证据图继续使用当前对齐差异热力图，并在图上叠加关键 ROI 矩形。界面仍显示原来的“模板匹配”卡，不新增分支、卡片或快捷键。`inspection.json` 和现有 evidence 路径保持不变。

## 错误处理

只保留实验室版本必要检查：

- ROI JSON 可解析。
- ROI 坐标为公共 ROI 局部坐标且不越界。
- 权重为有限且不小于 `1.0`。
- 加权阈值覆盖八个标准视角。
- 有效权重非空，相关系数分母非零。

配置或模型加载失败时启动命令在终端明确报错；检测期计算异常按现有 `_call()` 转为界面 `ERROR`。不增加额外身份、重复资产或完整性检查。

## 回滚

最快回滚只需将：

```json
"enabled": false
```

然后完全重启 Demo。程序立即恢复现有原始 Template 分数和 `template.thresholds`，无需重新训练、移动模型或重新计算任何 SHA。

## 测试与验收

聚焦测试：

1. `weighted_regions` 缺失或关闭时，25 项顺序、Template 分数、阈值和状态与当前版本完全一致。
2. 无关键 ROI 的视角在加权模式下仍走原始结果。
3. 合成差异仅位于关键 ROI 时，权重 `3.0` 得分高于权重 `1.0`。
4. ROI 重叠取最大权重，不叠加；ignore mask 覆盖区域最终权重为零。
5. 左右 ROI 文件严格独立，坐标按各自公共 ROI 验证。
6. 标定仅使用 calibration normal；改变 final-test 数据不会改变阈值。
7. 原始阈值字段在标定和接入后逐值保持不变。
8. 持久化仍是 25 条结果，并保存 legacy/weighted 两类诊断字段。

动态验证：

- 对左右各运行一次 0823 normal 离线回放，要求 25 项、0 ERROR，并报告 Template 加权 PASS/NG。
- 对现有左手确认形变样本做 A/B 回放，比较原始风险和加权风险，不自动把单样本结果当作正式验收。
- CUDA 和四相机现场行为在最终说明中单独列为尚未验证，除非实际完成现场测试。
