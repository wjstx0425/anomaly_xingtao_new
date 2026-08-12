# BMW EfficientAD 与光痕 v2 Demo 设计

## 目标

建立第二个独立实验配置：Template、YOLO、HDR 采集和零件 ROI 保持上午基线，仅将 EfficientAD 切换到 21:00 模型及其逐视角阈值，将光痕切换到修正 ROI 上的原灰度逐行局部对比算法。

## 设计

- 用组合运行目录把基线 Template/YOLO 与候选 EfficientAD 连接在一起，不复制或覆盖模型。
- Demo 配置新增可选 `raw_profile_v2` 光痕引擎；候选报告路径必须带 SHA-256，加载时校验。
- 光痕 v2 从报告读取 `[1792,1180,1873,1793]` ROI 和五个阈值，在 `front_left` 全图原始灰度上计算 coverage、最长连续段和内部断点，并在证据图标出 ROI。
- EfficientAD 阈值资产继续按现有契约校验每个 checkpoint 的 SHA，防止模型与阈值错配。
- 默认 Demo 与 Template 单变量 Demo 不变；第二版结果写入独立目录。

## 验证边界

先跑配置、组合资产和光痕预测器单测，再用一组离线八视图完成 25 项推理，最后启动四相机实验界面。光痕 v2 的历史留出结果为 19/20，但断续缺陷尚无真实样本；EfficientAD 候选仅验证了正常件误判率，缺陷召回尚未验证。
