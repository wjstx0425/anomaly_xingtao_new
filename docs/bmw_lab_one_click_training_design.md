# BMW八视图实验室一键训练设计

## 目标

提供一个Python入口，一次运行完成最终标签数据物化、八视图EfficientAD-S、全局YOLO26n、八视图模板匹配和
`front_left`光痕规则标定。产物仅用于实验室快速测试迭代，不修改旧BMW六视图运行时，也不声明工业发布。

## 方案选择

1. 推荐：新增八视图训练编排器，直接复用Anomalib、Ultralytics、现有模板算法和光痕检测器。
2. 串联旧脚本：改动少，但旧BMW脚本固定六视图且会对已裁ROI重复裁剪，不能正确消费当前数据。
3. 重构整个BMW运行时：长期更完整，但明显超出当前一键训练需求。

采用方案1。

## 固定训练合同

- 视图顺序：`front/front_left/front_right/front_secondary/back/back_left/back_right/back_secondary`。
- EfficientAD-S：每视图独立模型，`image_size=256x256`、`batch_size=1`、30轮，八个模型串行训练。
- YOLO26n：八视图共享一个模型，`imgsz=640`、`batch=32`、100轮、单类别`defect`。
- Template：每视图5张模板，输入为已裁ROI全图，使用当前OpenCV匹配与校准逻辑。
- Bright streak：只使用`front_left`，从calibration中的normal/no_streak拟合亮度/覆盖率阈值；当前没有断续缺陷样本，
  因此连续性仍保留为显式规则门槛，并取能覆盖calibration正常件的最严格包络。最终测试集只评估、不参与拟合。
- 单卡GPU 0；任一步失败立即停止；已有完整步骤可跳过，以便实验中断后继续。

## 数据与输出

- 输入最终标签：`dataset/bmw_lab_labeling/exports/bmw_label_review_project14_final_20260810_032057/reviewed_yolo_labels`。
- 训练数据版本：`dataset/bmw_lab_training/bmw_hdr_roi_training_reviewed_v1`。
- 默认结果根：`results/bmw_lab_one_click`，由`--run-id`区分实验。
- 每个步骤写独立报告；总入口写`run_report.json`，明确标记`experimental_only=true`。

## 验证边界

只增加参数/编排单元测试和`--dry-run`冒烟，不启动耗时GPU训练，不做全仓测试。
