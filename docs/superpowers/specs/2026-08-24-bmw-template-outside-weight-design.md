# BMW Template ROI 外权重设计

## 目标

在现有整块 Template 匹配中，正面四视角保持关键 ROI 权重 `3.0`，将关键 ROI 外权重从 `1.0` 降为 `0.5`。加权相关系数归一化后，关键 ROI 相对普通区域的有效权重由 `3:1` 提高为 `6:1`。

## 运行逻辑

- 配置字段为 `template.weighted_regions.outside_weight`。
- 权重图初始填充 `outside_weight=0.5`，人工关键 ROI 覆盖为 `weight=3.0`，ignore mask 区域仍为 `0.0`。
- 只有存在关键 ROI 的视角进入加权评分。当前左右关键 ROI 文件都仅在 `front`、`front_left`、`front_right`、`front_secondary` 各包含两个区域，因此反面四视角行为和阈值保持不变。
- 证据详情记录 ROI 权重、ROI 外权重和有效相对比例，不改变现有 Template 证据图和四模块融合。

## 阈值与回退

- 不修改现有 `bmw_eight_view_demo_{left,right}_0823_template40_v1.json`，它们是回退版本。
- 新增 `bmw_eight_view_demo_{left,right}_0823_template40_outside05_v1.json`，只在新版本中加入 `outside_weight=0.5` 并写入重新标定的阈值。
- 使用现有 calibration normal 按每视角最大风险加 10% 标定正面四视角；final_test 只报告，不参与阈值选择。反面阈值沿用现有版本。

## 验证

- 单元测试证明权重图是 ROI 内 `3.0`、ROI 外 `0.5`、ignore 区 `0.0`。
- 配置测试证明缺省 `outside_weight` 仍等价于 `1.0`，旧配置可直接回退。
- 运行左右标定并保存报告；再对一个已有八视图样本做一次离线回放。

