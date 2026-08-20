# BMW 八视图数据准备器设计

## 目标

从 `dataset/bmw_lab_raw` 的四相机、正反面八视图 HDR 采集清单中，生成一个不可覆盖、可追溯的实验室数据发布。发布负责过滤不完整采集、按物理件切分，并为 EfficientAD、Template、YOLO 和光痕规则给出彼此独立的标签清单。

## 边界

- 保留现有六视图 BMW runtime 与 `ViewId`，新准备器使用独立八视图合同。
- 原始数据只读；输出写入新的版本目录，已存在目录拒绝覆盖。
- 只接受 sample 记录明确为 `complete`、且对应八张 image 记录完整存在的采集组。
- 物理件身份使用 `sample_id` 去掉末尾的六位图序号，例如 `bmw_edge_group003_000001` 映射为 `bmw_edge_group003`。这复用采集器原生的 part-instance 合同，也允许同一物理件跨 session 复拍仍保持同一身份；报告显式记录该身份依据。
- 先按源类别分层、再按物理件确定性切分为 `train/calibration/final_test`，同一物理件的八视图永不跨 split。
- `no_streak` 的业务标签仍为 NG；EfficientAD/Template 为分支 normal；YOLO 为已确认负样本；只有 `front_left` 光痕标签为 `NG_NO_STREAK`。
- `deform/edge/others` 的逐视图可见性无法从目录名推断，因此 EfficientAD、Template、YOLO 均发布为 `review_required`，不伪造异常标签或框。
- 本任务不裁 ROI、不生成 YOLO 框、不训练模型。后续 ROI 与人工标注完成后，由下一阶段导出训练目录。

## 输出

发布根目录 `<output-root>/<dataset-id>/` 包含：

- `manifests/dataset_manifest.csv`：1056 张完整图的统一事实表。
- `manifests/part_splits.csv`：每个物理件的源类别和 split。
- `manifests/efficientad.csv`、`template.csv`、`yolo_annotation.csv`：分支标签与人工复核状态。
- `manifests/bright_streak.csv`：仅 `front_left` 的预期光痕结果。
- `report.json`：输入清单、完整/不完整计数、类别/视图/split 计数、身份依据、发布状态与文件哈希。

CSV 中的源图路径保存为绝对路径，图片内容不复制，避免额外占用空间。默认计算 SHA-256；`--skip-image-hash` 仅允许快速实验并在报告中标记弱完整性。

## 错误处理与验收

下列情况 fail closed：manifest 表头漂移、sample 状态重复或矛盾、完整 sample 缺少/重复视图、路径不存在、视图/相机/round 不符合固定拓扑、图片尺寸不一致、同一图被多个物理件复用、dataset ID 不安全、输出已存在。

验收以单元测试、真实数据 dry-run、真实数据不可覆盖发布、CLI help、compileall 和 diff 检查为准；不把训练或模型准确率计入本任务完成标准。
