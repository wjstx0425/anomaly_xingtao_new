# BMW 21点Template单变量Demo设计

## 目标

通过真实四相机八视图Demo，只测试21点批次重新训练的Template模型。除Template模型文件外，EfficientAD checkpoint与阈值、YOLO、光痕、ROI、HDR采集参数全部保持上午部署版本不变。

## 方案比较

1. 推荐：建立独立组合run目录。`template`指向21点候选，`efficientad`和`yolo`指向上午run；新增独立Demo配置和结果目录。优点是改动最小、归因清楚、可立即回退。
2. 修改Demo配置schema，增加每个分支独立模型根目录。长期更灵活，但本次会扩大代码和测试范围。
3. 覆盖上午run中的Template。最快但会破坏基线，无法接受。

采用方案1。

## 资产与数据流

- 候选Template：`results/bmw_lab_one_click/bmw_right_batch_20260810_21_template_only_v1/template`
- 上午基线：`results/bmw_lab_one_click/bmw_right_multisource_left_yolo_v1`
- 组合run：`results/bmw_lab_one_click/bmw_right_batch_20260810_21_template_demo_v1`
- 测试配置：`configs/bmw/experiments/bmw_eight_view_demo_template_21only_v1.json`
- 测试结果：`results/bmw_eight_view_demo_template_21only_v1`

准备脚本创建三个明确的目录链接并拒绝覆盖既有组合run。Demo仍由现有 `load_demo_config` 验证所有模型、EfficientAD阈值资产和checkpoint哈希，然后由现有相机、推理和中文界面执行。

## 验收边界

- 原 `bmw_eight_view_demo_v1.json` 字节与哈希不变。
- 新配置加载后，8个Template模型来自21点候选，8个EfficientAD checkpoint和YOLO来自上午run。
- 光痕配置与EfficientAD阈值资产保持上午路径和哈希。
- 首先启动GUI联动相机测试正常件；本轮结果用于判断Template单变量表现，不代表其他模型重新验证。
