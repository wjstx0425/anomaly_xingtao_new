# BMW 左右手 Template 与 EfficientAD 一键训练设计

## 目标

新增一个实验室一键脚本，顺序训练 0820 右手和左手数据的 Template 与八视角 EfficientAD。两侧 EfficientAD 都使用各自 prepared release 中的全部 `normal` 数据。

## 固定输入

- 右手 prepared release：`dataset/bmw_lab_prepared/bmw_right_0820_v1`
- 右手 ROI：`configs/bmw/rois/bmw_right_0820_v1.json`
- 左手 prepared release：`dataset/bmw_lab_prepared/bmw_left_0820_v1`
- 左手 ROI：`configs/bmw/rois/bmw_left_0820_v1.json`

命令行允许覆盖这些路径以及两侧的 `training-id`、`run-id`、训练轮数、GPU、workers 和随机种子。

## 执行流程

脚本按右手、左手顺序执行，每侧固定执行：

1. 根据对应 prepared release 和同手 ROI 物化训练数据。
2. Template 继续使用公开的 train/calibration 划分训练与拟合候选阈值。
3. EfficientAD 将全部 `source_class=normal` 图像放入训练目录，并训练八个视角 checkpoint。

任一步失败后立即停止，不启动另一侧。`--dry-run` 只验证两侧身份并打印计划，不创建训练目录、不启动 GPU。

## 边界

- 不训练或修改 YOLO。
- 不训练或修改光痕模型。
- 不并行运行两个手型，避免争抢单张 GPU。
- 全 normal EfficientAD 模式没有内部 validation/test，不运行 `engine.test()`。
- 输出 checkpoint 标记为等待独立验证；不得沿用旧阈值或直接发布到 Demo。
- 输出目录保持不可覆盖语义，重复运行必须更换 ID。

## 验收

- 一键脚本默认选择正确的左右手 prepared release 和 ROI。
- 两侧配置都强制 `efficientad_all_normal_train=True` 和 `stage=train`。
- 两侧计划都只有 materialize、Template、EfficientAD，不出现 YOLO、光痕或内部 calibration 阶段。
- dry-run 和失败即停行为有聚焦单元测试。
