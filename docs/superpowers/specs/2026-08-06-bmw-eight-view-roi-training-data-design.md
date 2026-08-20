# BMW 八视图 ROI 与训练数据设计

## 目标

在已发布的 `bmw_hdr_eight_view_v1` canonical manifests 上选择八个固定零件 ROI，并把1056张全图裁成一次性 canonical crops，随后组织为 EfficientAD、Template 和 YOLO 可消费的数据布局。

## ROI 选择器

- 从 `dataset_manifest.csv` 选择同一个 `normal/train` 完整八视图 sample 作为代表。
- 逐视图缩放到屏幕内调用 OpenCV ROI 选择，保存半开区间 `[x1,y1,x2,y2]`。
- ROI JSON 固定包含 dataset ID、源 manifest 路径/SHA-256、图像尺寸、代表 sample、八视图顺序和八个 ROI。
- 默认拒绝覆盖；显式 `--force` 才允许原子替换同一实验 ROI 文件。

## 训练数据物化

- 输出目录不可覆盖；所有裁剪先写 staging，完成校验后原子发布。
- 每张源图只写一份 `crops/<view>/<sample>__<view>.png`，分支目录使用相对软链接。
- EfficientAD：`normal/no_streak` 均为 branch-good；train 写入 `normal`，calibration 写入 `normal_test`，final_test 独立保存。缺陷视图只有在提供的 YOLO 标签非空、且 split 为 calibration/final_test 时才作为 defect 评估样本。
- Template：train references 只使用源 `normal`；`no_streak` 在 calibration/final_test 中仍是 normal。生成八视图目录与 manifest，不调用旧六视图 trainer。
- YOLO：`normal/no_streak` 自动生成空标签。`deform/edge/others` 默认进入待标注队列，不生成 `data.yaml`；当 `--yolo-label-root` 为每个待标注图提供一个同名标准 YOLO txt（允许空文件表示该视图确认不可见）后，才发布完整 images/labels 和 `data.yaml`。
- YOLO 类别固定为 `0: defect`，标签坐标相对 ROI crop，所有数值必须有限且在 `[0,1]`，宽高必须大于0。

## 边界

本任务生成训练数据，不训练模型、不改旧六视图 runtime、不伪造缺陷可见性。ROI 必须由用户在固定工装真实图上选择；在 ROI 文件存在之前，真实数据物化命令不得执行。
