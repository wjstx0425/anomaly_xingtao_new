# BMW Right 八视图一键训练设计

## 目标

为右手 BMW 数据提供独立的一键训练入口，消费已完成的八视图 ROI 数据及 Label Studio 导出的 ZIP/JSON，
训练 Template、光痕规则、EfficientAD-S 和 YOLO26n。产物只用于实验室快速迭代。

## 方案

新增轻量右手入口并复用现有八视图训练编排器。入口先核对 JSON 中的任务文件名与 ZIP 中的 240 个 YOLO
标签文件，再生成稳定标签缓存；随后使用右手 prepared release 和固定 ROI 配置物化新的不可变训练 release。
现有训练器移除旧数据专属的 248 标签/每视图 132 张硬编码，改为从当前 prepared manifest 推导数量。

## 固定参数与数据

- Prepared 数据：`dataset/bmw_lab_prepared/bmw_right_complete_20260810_v1`。
- ROI：`configs/bmw/rois/bmw_right_hdr_eight_view_v1.json`。
- 标签：用户提供的 `project-1-at-2026-08-10-20-25-8ec6b908.zip` 和同名 JSON。
- 新训练 release：`dataset/bmw_lab_training/bmw_right_complete_roi_reviewed_v1`。
- 结果目录：`results/bmw_lab_one_click/bmw_right_eight_view_v1`。
- EfficientAD-S：八视图、batch 1、30 轮；`normal` 与 `no_streak` 均作为正常类。
- YOLO26n：共享模型、单类别 `defect`、batch 32、100 轮、640 像素。
- Template：每视图 5 张模板；光痕规则：`front_left` 标定。

## 边界与验证

标签缓存已存在且内容不一致时直接失败，不覆盖旧训练数据或结果。仅运行聚焦单元测试、CLI 帮助和真实数据
`--dry-run`；不在实现阶段启动 GPU 训练。
