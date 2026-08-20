# BMW 右手多数据源一键训练设计

## 目标

新增一个实验室训练入口：EfficientAD、Template 和光痕规则同时使用原右手 ROI release 与
`bmw_right_batch_20260810_21_roi_v1`；YOLO26n 同时使用左手完整 YOLO release 与原右手已复核数据。

## 数据合同

- 右手分支源：`bmw_right_complete_roi_v1`、`bmw_right_batch_20260810_21_roi_v1`。
- 左手 YOLO 源：`bmw_hdr_roi_training_reviewed_v1`。
- 右手 YOLO：使用 `bmw_right_complete_roi_v1` 的 ROI 图片和现有 240 个复核标签。
- 新右手批次不进入 YOLO。
- YOLO 合并的是各自正确配对的图片和标签，不把一只手的坐标标签套到另一只手图片上。
- EfficientAD 中 normal/no_streak 都是正常类；Template 和光痕保留各 release 的既有 split。

## 实现

生成不可变组合 release，使用相对符号链接复用图片。Template 的 `sample_id` 和 `part_id` 添加源 release
前缀，避免两次采集重复编号造成身份碰撞；光痕 manifest 同样加前缀。YOLO 保留左右手原 split，统一类别
`0: defect`。组合 release 完成后复用现有五阶段训练器。

## 默认输出与验证

- 组合数据：`dataset/bmw_lab_training/bmw_right_multisource_left_yolo_v1`。
- 模型结果：`results/bmw_lab_one_click/bmw_right_multisource_left_yolo_v1`。
- YOLO26n 默认 batch 32、100 轮、imgsz 640；EfficientAD-S 默认 30 轮。
- dry-run 只核对源数据和打印计划，不创建组合 release 或启动 GPU 训练。
