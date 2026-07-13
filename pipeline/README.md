# 缺陷检测 Pipeline 快速指南

所有命令都在仓库根目录执行：

```bash
cd /home/yunjing/anomalib
uv sync
```

下面示例统一使用 `.venv/bin/python`。如果已经激活虚拟环境，也可以直接用
`python`。

## 流程总览

| 脚本 | 用途 | 适用场景 |
| --- | --- | --- |
| `0_run_all.py` | 串联多个阶段 | 已经确认单步命令正确后再使用 |
| `1_collect_data.py` | 采集原始大图 | normal/defect 原始数据采集 |
| `1_collect_multicamera_data.py` | 三相机采集 ZS32 正反面六视图 HDR | 静止工件人工翻面、单终端成组采集 |
| `2_process_data.py` | 裁成单零件图片，可 mask 孔 | 自动预设裁剪或手动画框 |
| `3_train_model.py` | 预处理、训练、评估 | 单模型或指定模型训练的主入口 |
| `4_inference.py` | 离线推理并输出复核图 | 复核新图片、stress normal、real defects |
| `5_demo_inspection.py` | 现场双面演示 | 相机在线检测、C789/FX11 切换 |
| `6_compare_models.py` | 比较 PatchCore、EfficientAD、AnomalyDINO | 单个数据集的常规模型对比 |
| `7_train_c789_100_compare_4gpu.py` | C789-100 top/bottom 四卡批量实验 | 专用脚本，默认路径绑定 `/DATA/ljl/...` |
| `8_train_custom_models.py` | 自定义数据集多模型训练 | 通用批量训练入口，路径和参数都显式传入 |
| `9_split_stress_normal.py` | stress normal 按 group 拆 train/locked | 鲁棒性验证，要求文件名能解析 `_g编号_` |
| `10_build_hardened_dataset.py` | clean + stress train 合成 hardened 数据集 | 用 stress train 扩充正常分布 |
| `11_build_geometry_templates.py` | 从正常图自动建几何模板 | 初始几何模板或快速 baseline |
| `12_geometry_eval.py` | 评估几何模板并可融合 AnomalyDINO | locked stress 校准、defect 验收 |
| `13_export_geometry_review_pack.py` | 导出人工 review pack 和可编辑 mask | 人工检查、生成手工模板种子 |
| `14_build_manual_geometry_templates.py` | 从手工 mask 编译模板 | 编辑 mask 后必须运行 |
| `15_edit_geometry_masks.py` | 可视化编辑手工 mask | 交互式修 expected/allowed/ignore/watch_edge |
| `16_build_multiview_manifest.py` | 生成多视角 manifest | 正反面、多光照图片按 part/side/view 分组 |
| `17_calibrate_quality_gate.py` | 校准整图质量门控 | 从 normal/stress normal 统计亮度、过曝、欠曝、模糊阈值 |
| `18_fuse_inspection_results.py` | 融合 quality/registration/geometry/anomaly CSV | 输出统一 OK/NG/RETAKE/INVALID_CAPTURE/SUSPECT |
| `19_run_robustness_benchmark.py` | 汇总融合后的鲁棒性 benchmark | clean/stress normal FP、invalid reject、defect recall |
| `20_run_traditional_operators.py` | 运行 C789 传统可解释算子 | registration、geometry、crack、surface、feature presence 分支 |
| `21_prepare_yolo_dataset.py` | 导出 Ultralytics YOLO 检测数据集 | 从 slot crop manifest + bbox 标注生成 images/labels/data.yaml |
| `22_collect_c789_yolo_defects.py` | 采集并裁剪 C789 YOLO 缺陷样本 | 6 个槽位都放缺陷件，输出待标注 defect crop |
| `23_augment_yolo_dataset.py` | 对 YOLO bbox 数据做离线增强 | 训练集增强，bbox 同步变换 |
| `25_prepare_yolo_same_dist_dataset.py` | 构建同分布 YOLO 验证集 | 按 `gNNN` 采集组拆分，train 增强、val 不增强 |
| `26_prepare_yolo_roi_dataset.py` | 构建 ROI-level YOLO 数据集 | GT-centered 诊断 ROI 或 tiled 部署 ROI |
| `27_prepare_zs32_label_studio.py` | 准备 ZS32 Label Studio 本地文件目录 | 汇总六视图缺陷图并生成 manifest 和标注界面配置 |
| `28_prepare_zs32_yolo_dataset.py` | 构建 ZS32 六视角 YOLO 数据集 | 合并 Label Studio 框、空视角、真实/镜像正常负样本并按工件拆分 |
| `29_zs32_fixed_roi.py` | 选择六视角固定 ROI 并裁剪 YOLO 数据集 | OpenCV 可视化框选，保持原 split 并转换 bbox |
| `31_calibrate_zs32_fusion.py` | 离线锁定 ZS32 72 组双阈值 | 输出不可原地覆盖的阈值 bundle 和校准报告 |
| `train_zs32_template_gate.py` | 训练左右手六视角整图模板门禁 | 输出带双阈值、版本和模板哈希的 12 组模型 |
| `predict_zs32_template_gate.py` | 执行第一个模板检测项目 | 非 PASS 返回非零退出码并阻止后续检测 |

最常用流程：

```text
采 normal/defect -> 裁剪 -> 训练评估 -> 推理复核或现场演示
```

如果要上传到 GitHub，只提交代码和文档，不提交数据、checkpoint 或训练输出。
当前 `.gitignore` 已排除 `dataset/`、`datasets`、`results`、`c789_bottom/`、
`*.ckpt`、`wandb/`、`lightning_logs/`、`mlruns` 等本地产物。提交前建议检查：

```bash
git status --short --ignored dataset results c789_bottom
git diff --cached --stat
```

`git status --short --ignored` 中这些目录应显示为 `!!`，不应显示为 `A` 或 `M`。

## 1. 采集数据

正常样本：

```bash
.venv/bin/python pipeline/1_collect_data.py \
  --hand no_hand \
  --position top \
  --label normal \
  --part-id part001 \
  --group-count 60 \
  --images-per-group 5 \
  --manual-load \
  --hdr \
  --save-hdr-sources \
  --align-hdr \
  --short-exposure 4000 \
  --long-exposure 35000 \
  --short-dark-threshold 80 \
  --long-clip-threshold 245 \
  --blend-width 50 \
  --blur-size 101 \
  --hdr-settle-frames 8 \
  --gain 0 \
  --root ./dataset/fx11_2
```

缺陷样本：

```bash
.venv/bin/python pipeline/1_collect_data.py \
  --hand no_hand \
  --position top \
  --label defect \
  --defect-type scratch \
  --part-id part001 \
  --group-count 20 \
  --images-per-group 1 \
  --manual-load \
  --hdr \
  --save-hdr-sources \
  --align-hdr \
  --short-exposure 4000 \
  --long-exposure 35000 \
  --short-dark-threshold 80 \
  --long-clip-threshold 245 \
  --blend-width 50 \
  --blur-size 101 \
  --hdr-settle-frames 8 \
  --gain 0 \
  --root ./dataset/fx11_2
```

`--manual-load` 会在每组采集前等待人工上料，按 `Enter` 或 `s` 都会开始采集。

输出目录：

```text
<root>/<hand>/<position>/normal/<part>_<time>/images
<root>/<hand>/<position>/defect/<defect-type>/<part>_<time>/images
```

关键参数：

| 参数 | 说明 |
| --- | --- |
| `--hand` | 采集视角，例如 `left`、`right`、`no_hand` |
| `--position` | 相机位置，例如 `top`、`bottom`、`side`、`bottom_ZS32` |
| `--label` | `normal` 或 `defect` |
| `--defect-type` | 缺陷类型，采 `defect` 时建议填写 |
| `--group-count` | 上料次数 |
| `--images-per-group` | 每次上料保存几张图 |
| `--root` | 原始数据根目录 |
| `--hdr` | 开启短曝光/长曝光融合 |
| `--save-hdr-sources` | 同时保存短曝光和长曝光原图 |

### ZS32 三相机六视图采集

先确认 SDK 当前枚举到的设备索引、型号和序列号：

```bash
.venv/bin/python pipeline/1_collect_multicamera_data.py --list-devices
```

三个视角按相机序列号固定绑定：正面 `DA9805574`、左侧 `DA9625347`、
右侧 `DB0998274`，不依赖可能变化的 SDK 设备编号。先用默认单曝光跑通流程：

```bash
.venv/bin/python pipeline/1_collect_multicamera_data.py \
  --hand left \
  --label normal \
  --part-id zs32_test \
  --group-count 1 \
  --images-per-group 1 \
  --manual-load \
  --exposure 4000 \
  --gain 0 \
  --capture-interval 0.2 \
  --timeout-ms 3000 \
  --root /tmp/zs32_three_camera_test
```

采集缺陷样本时将 `--label normal` 改为 `--label defect` 并增加非空的
`--defect-type <缺陷类型>`。单曝光图像文件名以 `_single.png` 结尾，manifest 的
`capture_mode=single`，并且 `source_short`、`source_long` 为空。

物理相机槽位和六视图的关系如下：

| 相机槽位 | 固定序列号 | 正面轮视图 | 翻面后视图 |
| --- | --- | --- | --- |
| 中央 | `DA9805574` | `front` | `back` |
| 左侧 | `DA9625347` | `front_left` | `back_left` |
| 右侧 | `DB0998274` | `front_right` | `back_right` |

HDR 功能没有删除。需要时在上述命令中显式增加 `--hdr`，并按需加入
`--save-hdr-sources --short-exposure 7000 --long-exposure 40000`。程序不再向相机写入
硬件帧率；`--capture-interval` 仅表示相邻软件触发批次之间的最小等待秒数。

右手件 100 组正常 HDR 批量采集（每个视角 1 张、只保存融合图）：

```bash
.venv/bin/python pipeline/1_collect_multicamera_data.py \
  --hand right \
  --label normal \
  --part-id zs32_right_normal \
  --group-count 100 \
  --images-per-group 1 \
  --manual-load \
  --hdr \
  --short-exposure 1500 \
  --long-exposure 6000 \
  --gain 0 \
  --capture-interval 0.2 \
  --hdr-settle-frames 1 \
  --timeout-ms 3000 \
  --root /home/yunjing/anomalib/dataset
```

输出位于 `/home/yunjing/anomalib/dataset/right/<view>/normal/<session_id>/images`，
100 个工件共生成 600 张 HDR 融合图。需要同时保存 1200 张短、长曝光源图时，再增加
`--save-hdr-sources`。

每个 group 只提示两次：首先按提示放好正面，连续采完该 group 的所有
`images-per-group`；然后将同一工件翻到背面，再连续采完所有图像。每个
image index 的正面三图与背面三图共用一个 `sample_id`。

对于 normal，六个最终图像目录是：

```text
<root>/left/front/normal/<session_id>/images
<root>/left/front_left/normal/<session_id>/images
<root>/left/front_right/normal/<session_id>/images
<root>/left/back/normal/<session_id>/images
<root>/left/back_left/normal/<session_id>/images
<root>/left/back_right/normal/<session_id>/images
```

缺陷样本在视图目录下使用 `defect/<defect-type>/<session_id>/images`。会话 manifest
写入 `<root>/manifests/<session_id>.csv`。传入 `--save-hdr-sources` 时，短、长曝光原图
与 fused 图保存在同一视图目录，manifest 的 `source_short` 和 `source_long`
列会记录对应路径。

现场先以 `--list-devices` 确认三个目标序列号都在线后再采集。单曝光图像过暗或
过曝时调整 `--exposure`；HDR 模式再调整 `--short-exposure` 和 `--long-exposure`，
还可按需调整 `--short-dark-threshold`、
`--long-clip-threshold`、`--blend-width`、`--blur-size`、`--hdr-max-clip-pct` 和
`--hdr-max-retries`。

manifest 只有在同一 `sample_id` 的六个标准视图都成功保存后才标记
`sample_status=complete`。任一轮采集、读图或写盘失败都保留可用行，同时写入
`sample_status=incomplete`、`failed_round`、`failed_view`、`failed_device_index` 和
`error`；下游不应将 incomplete 样本当成完整六视图。

该实现使用软件触发：每次先向三台相机发送触发，再依次读图。这适合静止工件，
但不是硬件同步；对运动件或要求严格同时曝光的场景，需改用共享硬件触发/同步线。

## 2. 处理数据

处理阶段把一张大图裁成多个单零件图。裁剪后的数据训练时必须配合：

```bash
--roi full
```

### 自动裁剪

C789 left/top：

```bash
.venv/bin/python pipeline/2_process_data.py auto \
  --data-root dataset/c789 \
  --output-root dataset/c789_left_top_parts \
  --hand left \
  --position top \
  --preset c789_left_top_3x2 \
  --hole-mask-method inpaint \
  --preview-overlay results/c789/left_top_part_crop_preview.png \
  --overwrite
```

C789 left/bottom：

```bash
.venv/bin/python pipeline/2_process_data.py auto \
  --data-root dataset/c789 \
  --output-root dataset/c789_left_bottom_parts \
  --hand left \
  --position bottom \
  --output-position bottom_ZS32 \
  --preset c789_left_bottom_3x2 \
  --hole-mask-method inpaint \
  --preview-overlay results/c789/left_bottom_part_crop_preview.png \
  --overwrite
```

FX11 no_hand/top：

```bash
.venv/bin/python pipeline/2_process_data.py auto \
  --data-root dataset/fx11_demo \
  --output-root dataset/fx11_demo_parts \
  --hand no_hand \
  --position top \
  --preset fx11_no_hand_top_6x1 \
  --hole-mask-method none \
  --preview-overlay results/fx11_demo/top_part_crop_preview.png \
  --overwrite
```

FX11 no_hand/bottom：

```bash
.venv/bin/python pipeline/2_process_data.py auto \
  --data-root dataset/fx11_demo \
  --output-root dataset/fx11_demo_parts \
  --hand no_hand \
  --position bottom \
  --preset fx11_no_hand_bottom_6x1 \
  --hole-mask-method none \
  --preview-overlay results/fx11_demo/bottom_part_crop_preview.png \
  --overwrite
```

### 手动裁剪

固定框不准时，用 `manual` 模式手动画框：

```bash
.venv/bin/python pipeline/2_process_data.py manual \
  --data-root dataset/fx11_2 \
  --output-root dataset/fx11_2_parts_manual \
  --hand no_hand \
  --position top \
  --slot-count 6 \
  --order vertical \
  --save-overlay results/fx11_2/manual_part_crop_preview.png \
  --slots-csv results/fx11_2/manual_part_slots.csv \
  --hole-mask-method none \
  --overwrite
```

如果已有坐标，可以直接传 `--slot` 跳过 GUI：

```bash
.venv/bin/python pipeline/2_process_data.py manual \
  --data-root dataset/fx11_2 \
  --output-root dataset/fx11_2_parts_manual \
  --hand no_hand \
  --position top \
  --slot slot01:1,1,832,43,3200,446 \
  --slot slot02:2,1,835,440,3209,853 \
  --slot slot03:3,1,838,893,3206,1296 \
  --slot slot04:4,1,844,1293,3221,1694 \
  --slot slot05:5,1,844,1700,3221,2107 \
  --slot slot06:6,1,847,2137,3230,2587 \
  --save-overlay results/fx11_2/manual_part_crop_preview.png \
  --slots-csv results/fx11_2/manual_part_slots.csv \
  --hole-mask-method none \
  --overwrite
```

`--slot` 格式：

```text
slot名:行,列,x1,y1,x2,y2
```

### FX11 缺陷专用裁剪

缺陷文件名带 slot 编号时，可以用固定规则只裁缺陷图：

```bash
.venv/bin/python pipeline/2_process_data.py fx11-defect \
  --face both \
  --data-root dataset/fx11_demo \
  --output-root dataset/fx11_demo_parts_manual \
  --overwrite
```

`--face` 可选 `top`、`bottom`、`both`。

处理阶段关键参数：

| 参数 | 说明 |
| --- | --- |
| `auto` | 使用预设裁剪框 |
| `manual` | 手动画框或传入固定 `--slot` |
| `fx11-defect` | FX11 缺陷文件专用裁剪 |
| `--data-root` | 原始大图根目录 |
| `--output-root` | 裁剪后单零件数据根目录 |
| `--preset` | 裁剪预设，例如 `c789_left_top_3x2` |
| `--hole-mask-method` | `none` 或 `inpaint` |
| `--preview-overlay` | 保存裁剪预览图，训练前必须检查 |
| `--overwrite` | 覆盖已有输出 |

normal 拆分参数：

```bash
--normal-split-mode auto      # 默认：没有 normal_test 时自动拆
--normal-split-mode always    # 即使已有 normal_test，也重新拆 normal
--normal-split-mode never     # normal 全部用于训练
--normal-test-ratio 0.25      # 默认 0.25
--normal-split-seed 0         # 固定随机种子
```

注意：workflow 的 `left_bottom` 读取目录是 `left/bottom_ZS32`。如果输入是
`left/bottom`，处理时要加：

```bash
--output-position bottom_ZS32
```

## 3. 训练与评估

`3_train_model.py` 默认执行完整流程：

```text
preprocess -> train -> evaluate
```

也可以只跑某一步：

```bash
.venv/bin/python pipeline/3_train_model.py preprocess ...
.venv/bin/python pipeline/3_train_model.py train ...
.venv/bin/python pipeline/3_train_model.py evaluate ...
```

### AnomalyDINO 推荐命令

C789 left/top：

```bash
.venv/bin/python pipeline/3_train_model.py \
  --data-root dataset/c789_left_top_parts \
  --output-root results/c789/left_top_parts_anomalydino \
  --views left_top \
  --models anomaly_dino \
  --skip-blue-removal \
  --roi full \
  --image-size 392,784 \
  --anomaly-dino-batch-size 1 \
  --eval-batch-size 1 \
  --anomaly-dino-encoder dinov2_vit_small_14 \
  --anomaly-dino-neighbors 1 \
  --accelerator gpu
```

C789 left/bottom：

```bash
.venv/bin/python pipeline/3_train_model.py \
  --data-root dataset/c789_left_bottom_parts \
  --output-root results/c789/left_bottom_parts_anomalydino \
  --views left_bottom \
  --models anomaly_dino \
  --skip-blue-removal \
  --roi full \
  --image-size 392,784 \
  --anomaly-dino-batch-size 1 \
  --eval-batch-size 1 \
  --anomaly-dino-encoder dinov2_vit_small_14 \
  --anomaly-dino-neighbors 1 \
  --accelerator gpu
```

FX11 no_hand/top：

```bash
.venv/bin/python pipeline/3_train_model.py \
  --data-root dataset/fx11_demo_parts \
  --output-root results/fx11_demo/no_hand_top_parts_anomalydino \
  --views no_hand_top \
  --models anomaly_dino \
  --skip-blue-removal \
  --roi full \
  --image-size 224,1008 \
  --anomaly-dino-batch-size 1 \
  --eval-batch-size 1 \
  --anomaly-dino-encoder dinov2_vit_small_14 \
  --anomaly-dino-neighbors 1 \
  --anomaly-dino-coreset-subsampling \
  --anomaly-dino-sampling-ratio 0.1 \
  --accelerator gpu
```

FX11 no_hand/bottom：

```bash
.venv/bin/python pipeline/3_train_model.py \
  --data-root dataset/fx11_demo_parts \
  --output-root results/fx11_demo/no_hand_bottom_parts_anomalydino \
  --views no_hand_bottom \
  --models anomaly_dino \
  --skip-blue-removal \
  --roi full \
  --image-size 224,1008 \
  --anomaly-dino-batch-size 1 \
  --eval-batch-size 1 \
  --anomaly-dino-encoder dinov2_vit_small_14 \
  --anomaly-dino-neighbors 1 \
  --anomaly-dino-coreset-subsampling \
  --anomaly-dino-sampling-ratio 0.1 \
  --accelerator gpu
```

训练关键参数：

| 参数 | 说明 |
| --- | --- |
| `--data-root` | 裁剪后的单零件数据根目录 |
| `--output-root` | 训练结果输出目录 |
| `--views` | 数据视角，例如 `left_top`、`left_bottom`、`no_hand_top` |
| `--models` | 模型名，例如 `anomaly_dino`、`patchcore`、`efficient_ad` |
| `--roi full` | 单零件 crop 必须使用 |
| `--image-size` | 输入尺寸，格式 `高,宽` |
| `--accelerator gpu` | 使用 GPU |
| `--eval-batch-size` | 评估 batch size |
| `--deploy-fpr` | 用 `normal_test` 定部署阈值时允许的误报率，如 `0.05` 表示约 5% |

默认 `--deploy-fpr 0.0`，阈值会取 `normal_test` 的最高分，尽量不误报。
如果想提高缺陷召回率，可以在训练评估或单独 `evaluate` 时加：

```bash
--deploy-fpr 0.05
```

报表里的分数直方图只画 `normal_test` 和 `defect`，训练集 `normal`
不参与阈值判断，也不会再画到直方图里。

注意：`3_train_model.py` 的默认部署目标是 `--deploy-fpr 0.0`；用于快速比较的
`6_compare_models.py` 和 `8_train_custom_models.py` 默认示例偏向召回率，通常使用
`--deploy-fpr 0.05`。对比不同结果时先确认 `reports/summary.csv` 里的
`deploy_target_fpr` 一致。

显存紧张时，AnomalyDINO 可以保留：

```bash
--anomaly-dino-coreset-subsampling
--anomaly-dino-sampling-ratio 0.1
```

PatchCore 最小替换参数：

```bash
--models patchcore
--patchcore-batch-size 1
--patchcore-layers layer2
--patchcore-precision float16
--patchcore-coreset-ratio 0.1
--patchcore-num-neighbors 1
```

EfficientAD 需要本地 teacher 权重和 ImageNette 数据；资源不全时可加：

```bash
--skip-missing-efficientad-assets
```

## 3.1 自动比较多个模型

想一次性跑 PatchCore、EfficientAD、AnomalyDINO，并生成统一对比表，可以用：

```bash
.venv/bin/python pipeline/6_compare_models.py \
  --data-root dataset/fx11_100_parts \
  --output-root results/fx11_100/no_hand_top_parts_compare \
  --views no_hand_top \
  --skip-blue-removal \
  --roi full \
  --image-size 256,1152 \
  --eval-batch-size 1 \
  --deploy-fpr 0.05 \
  --accelerator gpu
```

脚本默认使用比较保守的显存参数：

```text
PatchCore: layer2, float16, coreset=0.1
EfficientAD: batch=1, epochs=20
AnomalyDINO: batch=1, coreset=0.03
```

EfficientAD 要求特征图不能小于内部 `8x8` 卷积核。FX11 的 `224,1008`
高度太小，脚本会自动等比例放大到 `256,1152`；你也可以直接显式传
`--image-size 256,1152`。

如果 EfficientAD 的 teacher weights 或 ImageNette 数据不存在，脚本默认会跳过
EfficientAD，继续比较另外两个模型。想让资源缺失时直接报错，加：

```bash
--require-efficientad-assets
```

如果已经 preprocess 好了，想复用预处理结果：

```bash
.venv/bin/python pipeline/6_compare_models.py \
  --data-root dataset/fx11_100_parts \
  --output-root results/fx11_100/no_hand_top_parts_compare \
  --views no_hand_top \
  --skip-preprocess \
  --skip-blue-removal \
  --roi full \
  --image-size 256,1152 \
  --eval-batch-size 1 \
  --deploy-fpr 0.05 \
  --accelerator gpu
```

如果模型已经训练完，只想按新的阈值重新生成报表：

```bash
.venv/bin/python pipeline/6_compare_models.py \
  --data-root dataset/fx11_100_parts \
  --output-root results/fx11_100/no_hand_top_parts_compare \
  --views no_hand_top \
  --evaluate-only \
  --deploy-fpr 0.05 \
  --accelerator gpu
```

对比表会写到：

```text
<output-root>/reports/model_comparison.md
```

默认缺少 EfficientAD teacher weights 或 ImageNette 数据时会跳过 EfficientAD。
如果必须确认 EfficientAD 资源齐全，加 `--require-efficientad-assets`。

`6_compare_models.py` 是单数据集比较入口；如果要同时跑 C789 top/bottom
并扫多组采样比例，再用下面的 `7_train_c789_100_compare_4gpu.py`。如果只是换
自己的数据路径和超参数，优先用 `8_train_custom_models.py`，不要改硬编码脚本。

### 专用四卡 C789-100 批量实验

`7_train_c789_100_compare_4gpu.py` 是实验复现脚本，不是通用入口。默认读取：

```text
/DATA/ljl/c789_100_left_top_parts
/DATA/ljl/c789_100_left_bottom_parts
```

默认使用 GPU `0 1 2 3`，并对 PatchCore/EfficientAD/AnomalyDINO 扫多组采样比例。
在其他机器上必须显式传数据路径和 GPU：

```bash
.venv/bin/python pipeline/7_train_c789_100_compare_4gpu.py \
  --top-data-root dataset/c789_100_left_top_parts \
  --bottom-data-root dataset/c789_100_left_bottom_parts \
  --output-root results/c789_100_compare_4gpu \
  --gpus 0 1 2 3 \
  --skip-missing-efficientad-assets \
  --dry-run
```

确认 dry-run 打印的命令和路径都正确后，再去掉 `--dry-run`。

### 通用自定义模型训练

`8_train_custom_models.py` 适合把路径、模型和显存参数都显式写出来。它会先
preprocess 一次，再按模型并行训练，最后生成汇总表：

```bash
.venv/bin/python pipeline/8_train_custom_models.py \
  --data-root dataset/c789_100_left_top_parts \
  --output-root results/custom_left_top_compare \
  --views left_top \
  --models patchcore anomaly_dino \
  --gpus 0 \
  --roi full \
  --image-size 392,784 \
  --deploy-fpr 0.05 \
  --skip-missing-efficientad-assets \
  --dry-run
```

输出表在：

```text
<output-root>/reports/model_comparison.md
<output-root>/reports/model_comparison.csv
```

### ZS32 right 六视角数据

`right/` 数据集支持以下六个独立视角：

```text
right_front right_front_left right_front_right
right_back  right_back_left  right_back_right
```

原始目录可以包含缺陷类型和采集批次，例如：

```text
right/front/normal/<session>/images/*.png
right/front/defect/<defect_type>/<session>/images/*.png
```

如果没有 `normal_test`，工作流默认按文件名中的 `groupNNN` 固定留出 20% 正常组，
并让六个视角的同一 group 保持相同 train/test 划分。可用
`--normal-test-ratio` 调整；显式存在 `normal_test` 时不会重新划分。

源图约为 4:3 时，首轮 PatchCore 可执行：

```bash
.venv/bin/python pipeline/8_train_custom_models.py \
  --data-root /path/to/right \
  --output-root results/zs32_right_patchcore \
  --views right_front right_front_left right_front_right \
          right_back right_back_left right_back_right \
  --models patchcore \
  --gpus 0 \
  --roi full \
  --image-size 504,672 \
  --normal-test-ratio 0.2 \
  --patchcore-batch-size 1 \
  --patchcore-layers layer2 \
  --patchcore-precision float16 \
  --patchcore-coreset-ratio 0.1 \
  --deploy-fpr 0.0 \
  --dry-run
```

确认 dry-run 的路径后去掉 `--dry-run`。六个视角会各自训练模型，不会混成
一个视觉分布。

## 4. 推理

对原始图片或目录推理：

```bash
.venv/bin/python pipeline/4_inference.py hik_images \
  --output-root results/c789/left_bottom_parts_anomalydino \
  --view left_bottom \
  --model anomaly_dino \
  --accelerator gpu
```

对已经裁好的单零件目录推理：

```bash
.venv/bin/python pipeline/4_inference.py dataset/c789_left_bottom_parts/left/bottom_ZS32 \
  --output-root results/c789/left_bottom_parts_anomalydino \
  --view left_bottom \
  --model anomaly_dino \
  --input-is-preprocessed \
  --accelerator gpu
```

推理输出：

```text
predictions.csv
review/TP
review/TN
review/FP
review/FN
```

部署阈值优先级：

```text
显式 --threshold > <output-root>/reports/summary.csv 的 deploy_threshold > anomalib 原始 pred_label
```

如果 `--output-root`、`--view` 或 `--model` 指错，脚本可能找不到匹配的
`reports/summary.csv`，控制台会提示 `threshold: not found`，这时复核分类会退回
模型原始标签，不再等价于部署阈值。上线或对比报告时，建议显式传
`--threshold`，或先确认对应 `summary.csv` 存在且 view/model 匹配。

关键参数：

| 参数 | 说明 |
| --- | --- |
| 第一个位置参数 | 图片文件或图片目录 |
| `--output-root` | 对应训练结果目录 |
| `--view` | 视角，必须和训练一致 |
| `--model` | 模型名，必须和训练一致 |
| `--input-is-preprocessed` | 输入已经是单零件 crop 时使用 |
| `--visualize` | 额外保存可视化图 |

### Valid-region mask 计分

C789 stress normal 误报较高时，可以在推理阶段只让有效零件区域参与
部署判定，排除背景、夹具、孔洞和 inpaint 区域：

```bash
.venv/bin/python pipeline/4_inference.py dataset/c789_stress_normal_parts/left/top \
  --output-root results/c789_stress_normal/left_top_anomaly_dino \
  --output-dir results/c789_stress_normal/left_top_anomaly_dino/stress_valid_region \
  --view left_top \
  --model anomaly_dino \
  --ckpt-path results/c789_100/left_top_parts_anomalydino/runs/left_top/anomaly_dino/AnomalyDINO/zs32_left_top/left_top/v3/weights/lightning/model.ckpt \
  --input-is-preprocessed \
  --threshold 0.5 \
  --valid-region-mask-preset c789_left_top_3x2 \
  --valid-region-score-mode deploy \
  --valid-region-score-method masked_max \
  --accelerator gpu
```

`pred_score` 会保留模型原始分数；CSV 额外写入
`valid_region_score`、`valid_region_coverage` 和 `deploy_score_source`。
上线前不要只看 stress normal，还必须同时检查 locked real defects。

## 4.1 Stress normal 鲁棒性数据

stress normal 需要按 group 固定拆成 train/locked。locked 部分只用于验收，
不能混进训练或调阈值。

```bash
.venv/bin/python pipeline/9_split_stress_normal.py \
  --input-root dataset/c789_stress_normal_parts \
  --output-root dataset/c789_stress_normal_group_split \
  --hand left \
  --position top \
  --train-ratio 0.7 \
  --link-mode symlink \
  --overwrite
```

把 clean 数据和 stress train normal 合成训练数据集：

```bash
.venv/bin/python pipeline/10_build_hardened_dataset.py \
  --clean-root dataset/c789_100_left_top_parts \
  --stress-split-root dataset/c789_stress_normal_group_split \
  --output-root dataset/c789_100_left_top_hardened_parts \
  --hand left \
  --position top \
  --link-mode symlink \
  --overwrite
```

然后用 hardened 数据重建正常分布：

```bash
.venv/bin/python pipeline/3_train_model.py \
  --data-root dataset/c789_100_left_top_hardened_parts \
  --output-root results/c789_100_hardened/left_top_anomaly_dino \
  --views left_top \
  --models anomaly_dino \
  --skip-blue-removal \
  --roi full \
  --image-size 392,784 \
  --anomaly-dino-batch-size 1 \
  --eval-batch-size 1 \
  --anomaly-dino-encoder dinov2_vit_small_14 \
  --anomaly-dino-neighbors 1 \
  --accelerator gpu
```

locked stress normal 验收用 `pipeline/4_inference.py` 单独跑
`dataset/c789_stress_normal_group_split/locked/left/top`；locked real defects
也用同一个 ckpt 单独跑，两个结果都通过才算稳。

## 5. 现场双面演示

启动现场演示：

```bash
.venv/bin/python pipeline/5_demo_inspection.py \
  --device 0 \
  --accelerator gpu \
  --threshold-profile demo \
  --predict-batch-size 1 \
  --quality-gate warn
```

指定 C789：

```bash
.venv/bin/python pipeline/5_demo_inspection.py \
  --part-profile c789 \
  --device 0 \
  --accelerator gpu
```

同时指定 C789 和 FX11 的权重，界面里按 `1`/`2` 切换时会重新加载对应模型：

```bash
.venv/bin/python pipeline/5_demo_inspection.py \
  --part-profile c789 \
  --c789-top-output-root results/c789_100/left_top_parts_anomalydino \
  --c789-top-ckpt-path results/c789_100/left_top_parts_anomalydino/runs/left_top/anomaly_dino/AnomalyDINO/zs32_left_top/left_top/v3/weights/lightning/model.ckpt \
  --c789-bottom-output-root c789_bottom \
  --c789-bottom-ckpt-path c789_bottom/ckpt_015/model.ckpt \
  --fx11-top-output-root results/fx11_100/no_hand_top_parts_anomalydino \
  --fx11-top-ckpt-path results/fx11_100/no_hand_top_parts_anomalydino/runs/no_hand_top/anomaly_dino/AnomalyDINO/zs32_no_hand_top/no_hand_top/v1/weights/lightning/model.ckpt \
  --fx11-bottom-output-root results/fx11_100/no_hand_bottom_parts_anomalydino \
  --fx11-bottom-ckpt-path results/fx11_100/no_hand_bottom_parts_anomalydino/runs/no_hand_bottom/anomaly_dino/AnomalyDINO/zs32_no_hand_bottom/no_hand_bottom/v1/weights/lightning/model.ckpt \
  --device 0 \
  --accelerator gpu \
  --predict-batch-size 1 \
  --quality-gate warn
```

无相机调界面：

```bash
.venv/bin/python pipeline/5_demo_inspection.py \
  --part-profile c789 \
  --demo-top-image dataset/c789/left/top/normal/example.png \
  --demo-bottom-image dataset/c789/left/bottom/normal/example.png \
  --mock-predictions \
  --mock-defects top:slot02 \
  --auto-run \
  --no-gui \
  --save-ui-screenshot results/c789/demo_inspection/ui_preview.png
```

带归档和钢印 OCR 的离线演示：

```bash
.venv/bin/python pipeline/5_demo_inspection.py \
  --part-profile c789 \
  --demo-top-image dataset/c789/left/top/normal/example.png \
  --demo-bottom-image dataset/c789/left/bottom/normal/example.png \
  --archive-root results/c789/demo_inspection/archive \
  --stamp-roi-config config/stamp_roi.json \
  --diagnostics \
  --auto-run \
  --no-gui
```

归档会追加写入：

```text
<archive-root>/inspection_slots.csv
<archive-root>/inspection_parts.csv
```

每一面还会写 trace JSON、annotated image、slot crops、`predictions.csv`，
OCR 会额外保存 `stamp_rois/*_stamp.png` 和 `*_stamp_enhanced.png`。

`--stamp-roi-config` 是 crop 内坐标，不是原始大图坐标。支持两种 JSON 写法：

```json
{
  "c789.top": {
    "x1": 100,
    "y1": 40,
    "x2": 360,
    "y2": 160,
    "mirror_horizontal": true
  },
  "c789": {
    "bottom": {
      "x1": 120,
      "y1": 50,
      "x2": 380,
      "y2": 170,
      "rotate_180": true
    }
  }
}
```

`x2` 必须大于 `x1`，`y2` 必须大于 `y1`。如果 ROI 配置缺字段或越界，当前版本会跳过
对应面的 OCR 或给该 slot 写 `FAIL`，所以调试时建议加 `--diagnostics` 并检查
trace JSON 里的 `stamp_ocr`。

界面操作：

```text
s      检测当前面
r      重置当前零件流程
1/2    切换 FX11/C789
q      退出
```

演示默认按 `top -> bottom` 顺序检测。第一下 `s` 检测正面，第二下 `s`
检测背面/底面；双面完成后再按 `s` 会进入下一件。

关键参数：

| 参数 | 说明 |
| --- | --- |
| `--part-profile` | `fx11` 或 `c789` |
| `--device` | 相机编号 |
| `--threshold-profile` | `demo` 使用现场阈值，`report` 使用训练报告阈值 |
| `--top-threshold` / `--bottom-threshold` | 临时覆盖阈值 |
| `--c789-top-ckpt-path` / `--fx11-top-ckpt-path` | 指定某个零件某一面的 ckpt |
| `--c789-top-threshold` / `--fx11-top-threshold` | 指定某个零件某一面的阈值 |
| `--quality-gate` | `warn` 告警继续，`fail` 异常阻断 |
| `--predict-batch-size` | 推理 batch size，现场建议 `1` |
| `--diagnostics` | 打印亮度、清晰度、crop 分布诊断信息 |

## 0. 串联运行

需要把多个阶段串起来时，用 `0_run_all.py`。每个阶段参数写成一整段字符串：

```bash
PROCESS_ARGS='auto
  --data-root dataset/c789
  --output-root dataset/c789_left_top_parts
  --hand left
  --position top
  --preset c789_left_top_3x2
  --hole-mask-method inpaint
  --preview-overlay results/c789/left_top_part_crop_preview.png
  --overwrite'

TRAIN_ARGS='--data-root dataset/c789_left_top_parts
  --output-root results/c789/left_top_parts_anomalydino
  --views left_top
  --models anomaly_dino
  --skip-blue-removal
  --roi full
  --image-size 392,784
  --anomaly-dino-batch-size 1
  --eval-batch-size 1
  --accelerator gpu'

INFER_ARGS='dataset/c789_left_top_parts/left/top
  --output-root results/c789/left_top_parts_anomalydino
  --view left_top
  --model anomaly_dino
  --input-is-preprocessed
  --accelerator gpu'

.venv/bin/python pipeline/0_run_all.py \
  --process "$PROCESS_ARGS" \
  --train "$TRAIN_ARGS" \
  --inference "$INFER_ARGS"
```

只想看最终会执行什么命令：

```bash
.venv/bin/python pipeline/0_run_all.py --dry-run --process "auto --help"
```

## 必看注意事项

- 训练前先打开 `--preview-overlay` 的图片，确认裁剪框和孔 mask 正确。
- 单零件 crop 训练必须使用 `--roi full`。
- `left_bottom` 的 workflow 目录是 `left/bottom_ZS32`。
- `9_split_stress_normal.py` 依赖文件名里的 `_g编号_`，例如 `..._g001_...png`。
- `--link-mode symlink` 会写绝对源路径的符号链接，适合本机省空间；需要拷到别的机器时用 `--link-mode copy`。
- 指标好不代表能上线，建议再采一批新正常件和新缺陷件做独立验证。
- EfficientAD 可能需要额外本地资源；没有资源时先用 AnomalyDINO 或 PatchCore。

## C789 手工几何模板

C789 top 的 less/more/corner 缺陷建议走人工核验的几何模板流程。可以先用正常图
自动建一版模板：

```bash
.venv/bin/python pipeline/11_build_geometry_templates.py \
  --data-root dataset/c789_100_left_top_hardened_parts \
  --view left_top \
  --preset c789_left_top_3x2 \
  --output-dir results/c789_100_hardened/left_top_geometry/templates
```

如果要人工修模板，再导出 review pack：

```bash
.venv/bin/python pipeline/13_export_geometry_review_pack.py \
  --normal-root dataset/c789_100_left_top_hardened_parts/left/top/normal \
  --stress-root dataset/c789_stress_normal_group_split/locked/left/top \
  --defect-root dataset/c789_100_left_top_parts/left/top/defect \
  --output-dir results/c789_100_hardened/left_top_geometry/manual_review_pack \
  --preset c789_left_top_3x2 \
  --samples-per-split 2
```

默认不传 `--template-dir`，这样会从当前 normal reference 重新生成可编辑 mask。
只有在明确想基于已有自动模板继续修时，才额外加：

```bash
--template-dir results/c789_100_hardened/left_top_geometry/templates
```

人工检查 `manual_review_pack/sheets/slotXX_review_sheet.png`，然后编辑
`manual_review_pack/manual_masks/` 里的四类 PNG：

```text
slotXX_expected.png    零件必须存在的标准区域
slotXX_allowed.png     正常扰动允许出现的区域
slotXX_ignore.png      孔洞、inpaint、夹具、强反光等忽略区
slotXX_watch_edge.png  真正用于 less/more 判断的边界区域
```

### 可视化编辑器

也可以用可视化编辑器直接打开同一个 review pack：

```bash
.venv/bin/python pipeline/15_edit_geometry_masks.py \
  --review-pack results/c789_100_hardened/left_top_geometry/manual_review_pack \
  --template-dir results/c789_100_hardened/left_top_geometry/manual_templates \
  --stress-root dataset/c789_stress_normal_group_split/locked/left/top \
  --defect-root dataset/c789_100_left_top_parts/left/top/defect \
  --anomaly-predictions results/c789_100_hardened/left_top_anomaly_dino/reports/predictions.csv \
  --stress-output-dir results/c789_100_hardened/left_top_geometry/manual_stress_locked \
  --defect-output-dir results/c789_100_hardened/left_top_geometry/manual_defect_fused
```

常用快捷键：`1`-`6` 切换 slot，`e`/`a`/`i`/`w` 切换
`expected`/`allowed`/`ignore`/`watch_edge`，`[`/`]` 调笔刷大小，`u`/`r`
撤销/重做，`s` 保存，`v` 编译模板并运行 locked stress + defect 验证，`q`
退出。必须使用 `.venv/bin/python` 启动；mask 只能是纯黑/纯白，
不要使用灰阶、半透明或抗锯齿笔刷。保存会先备份原 PNG，再写入新 mask。

编辑完成后编译手工模板：

```bash
.venv/bin/python pipeline/14_build_manual_geometry_templates.py \
  --mask-dir results/c789_100_hardened/left_top_geometry/manual_review_pack/manual_masks \
  --output-dir results/c789_100_hardened/left_top_geometry/manual_templates
```

再用 locked stress normal 校准阈值：

```bash
.venv/bin/python pipeline/12_geometry_eval.py \
  --data-root dataset/c789_stress_normal_group_split/locked/left/top \
  --template-dir results/c789_100_hardened/left_top_geometry/manual_templates \
  --output-dir results/c789_100_hardened/left_top_geometry/manual_stress_locked \
  --calibrate-thresholds
```

最后跑 defect + AnomalyDINO 融合验收：

```bash
.venv/bin/python pipeline/12_geometry_eval.py \
  --data-root dataset/c789_100_left_top_parts/left/top/defect \
  --template-dir results/c789_100_hardened/left_top_geometry/manual_templates \
  --thresholds results/c789_100_hardened/left_top_geometry/manual_stress_locked/geometry_thresholds.csv \
  --anomaly-predictions results/c789_100_hardened/left_top_anomaly_dino/reports/predictions.csv \
  --output-dir results/c789_100_hardened/left_top_geometry/manual_defect_fused
```

融合时 `--anomaly-predictions` 会按 `source_path`、`processed_path`、
`image_path` 或文件名去匹配几何样本。路径来自不同输出目录时，可能匹配不上并退化成
geometry-only 结果；因此跑完后要检查 `manual_defect_fused/fused_predictions.csv`
里的 `anomaly_score`、`anomaly_threshold`、`anomaly_deploy_pred_label` 是否按预期非空。

### 几何阈值 fallback

`pipeline/12_geometry_eval.py` 现在支持 MVP-2 的几何阈值 fallback。阈值查找顺序是：

```text
exact:       slot_id + defect_type + region_id
slot_type:   slot_id + defect_type + *
slot_region: slot_id + * + region_id
slot:        slot_id + * + *
defect_type: * + defect_type + *
global:      * + * + *
```

这里的 `defect_type` 对应当前几何分支内部的 `geometry_type`，也就是 `less` / `more`
这类几何异常类型；当前阶段不从外部 manifest 读取真实缺陷大类。

旧版 `geometry_thresholds.csv` 仍可读取。加载旧 CSV 时会从已有阈值自动补齐 per-slot、
per-type 和 global fallback 行；重新 `--calibrate-thresholds` 时也会写出兼容新旧字段：

```text
slot,geometry_type,geometry_region,geometry_threshold
slot_id,defect_type,region_id,threshold,threshold_source,n_normal,max_normal,p99_normal,p999_normal
```

`geometry_predictions.csv` 额外包含：

```text
threshold_source
threshold_lookup_level
```

这两个字段用于解释某个样本命中了 exact 阈值还是 fallback 阈值。自动 fallback 使用已有
locked normal 阈值/统计中的最大值作为保守阈值，优先避免 stress normal FP 变差。
如果 locked normal 与模板完全一致、没有任何区域偏差，校准仍会写出每个 slot 的
`slot_id + * + * = 0.0` 和 global fallback，避免生成空阈值表。
单个预测样本如果没有任何区域偏差，也会通过 slot/global fallback 标记为
`geometry_pred_label=0`，而不是留下空预测。

## 质量门控 MVP-3

`capture_data/quality_gate.py` 提供不依赖深度学习的整图质量指标：

```text
brightness_mean
brightness_std
saturation_ratio
dark_ratio
blur_laplacian_var
highlight_ratio
foreground_coverage
```

当前上线策略先使用 `mode: warn`，质量异常不会阻断模型推理。没有 invalid 图时可以只用
normal / stress normal 校准阈值：

```bash
.venv/bin/python pipeline/17_calibrate_quality_gate.py \
  --normal-root dataset/c789_100_left_top_parts/left/top/normal \
  --stress-root dataset/c789_stress_normal_group_split/locked/left/top \
  --output-yaml config/quality_gate/c789_calibrated.yaml \
  --output-report results/c789_quality_gate/calibration_report.md
```

输出包括：

```text
quality_metrics.csv
c789_calibrated.yaml
calibration_report.md
```

`quality_gate.csv` 写出 `branch=quality_gate`、`status`、`fail_label`、`reason` 和
`source_path`，可以直接传给 `pipeline/18_fuse_inspection_results.py --quality-csv`。
WARN 行的 `fail_label=0`，只有 `mode=fail` 下的 FAIL 才会作为 retake gate 阻断 OK。

## Demo 质量告警 MVP-4

现场 demo 的旧命令保持可运行。推荐先用：

```bash
--quality-gate warn
```

WARN 模式下，质量门控只提示不阻断推理；UI 会显示 `质量 WARN` 和原因，trace JSON 的
`quality.issues` / `quality.reason`、以及 archive CSV 的 `quality_status` /
`quality_reasons` 会同步记录。只有显式使用 `--quality-gate fail` 时，质量失败才会阻断
模型推理。

## 多视角 Manifest MVP-5

`pipeline/16_build_multiview_manifest.py` 用于把正反面、多光照图片按
`part_id / side / view` 分组。优先推荐显式 CSV：

```text
part_id,side,view,image_path,label,defect_type,slot_id,group_id,notes
```

也可以通过文件名正则自动解析：

```bash
.venv/bin/python pipeline/16_build_multiview_manifest.py \
  --input-root dataset/c789_multiview_raw \
  --filename-regex '(?P<part_id>part[0-9]+)_(?P<side>top|bottom)_(?P<view>uniform)_slot(?P<slot_id>[0-9]+)\.png' \
  --required-side top \
  --required-side bottom \
  --required-view uniform \
  --output-csv results/c789_multiview/manifest.csv
```

如果某个 `part_id` 缺少 required side/view，stage 16 会在控制台标记
`[invalid_capture]`；stage 18 读取 manifest 后会把对应零件输出为 `INVALID_CAPTURE`。
真实文件名规则确定后，只需要把 `filename_regex` 放进 inspection profile，不需要改代码。

## C789 传统算子分支

`capture_data/traditional_operators.py` 提供不依赖训练的 C789 可解释检查。它复用现有
slot preset，不重新定位零件：

```text
c789_left_top_3x2     ROI 460,30,3480,2600
c789_left_bottom_3x2  ROI 350,320,3600,3030
```

默认配置在 `config/traditional/c789.yaml`。stage 20 已从全局粗规则升级为
“按 slot 自动标定模板 + 检测时复用模板”的流程。传统算子在这里是融合检测的一组
证据分支，不是全类别缺陷分类器；它只负责输出自己擅长的几何、暗线、纹理、姿态等
证据，再交给 stage 18 和深度模型一起融合。

第一版传统分支包括：

```text
registration       前景面积、bbox 中心偏移，默认 WARN
geometry           slot template 差分证据；输出 missing_mask / extra_mask，不宣称真实 less/more 类别
crack              暗细线增强 + 连通域长线证据，只在 material ROI 内评分
surface_texture    CLAHE/Laplacian 纹理残差证据，只在 material ROI 内评分，第一版只输出 SUSPECT
feature_presence   孔洞/暗特征数量证据，当前 C789 配置默认关闭
```

经验上，`geometry` 更适合支持 `less / more / corner / deform` 这类结构或轮廓相关缺陷；
`crack` 更适合支持细长暗裂纹候选；`surface_texture` 更适合做 `surface` 类复核提示。
表面污渍、反光、轻微划痕、裂纹 vs 划痕等语义强的类别，不应只靠单个传统分支硬分类。

先用 normal slot crop 自动标定 slot 模板和几何阈值。默认每个 slot 抽样 96
张 normal 做标定；需要全量离线精标定时传 `--calibration-max-images-per-slot 0`。
默认标定产物写到 `<output-dir>/calibration/`，后续 defect/stress normal 检测可直接复用：

```bash
.venv/bin/python pipeline/20_run_traditional_operators.py \
  --input-root dataset/c789_100_left_top_parts/left/top/normal \
  --calibrate-normal-root dataset/c789_100_left_top_parts/left/top/normal \
  --preset c789_left_top_3x2 \
  --side top \
  --view uniform \
  --input-mode slot \
  --config config/traditional/c789.yaml \
  --output-dir results/c789_traditional/top_calibrated
```

对 C789 top defect 或 stress normal slot crop 运行时，显式传入标定产物：

```bash
.venv/bin/python pipeline/20_run_traditional_operators.py \
  --input-root dataset/c789_100_left_top_parts/left/top/defect \
  --preset c789_left_top_3x2 \
  --side top \
  --view uniform \
  --input-mode slot \
  --config config/traditional/c789.yaml \
  --template-dir results/c789_traditional/top_calibrated/calibration/templates \
  --geometry-thresholds results/c789_traditional/top_calibrated/calibration/geometry_thresholds.csv \
  --output-dir results/c789_traditional/top_defect
```

对 C789 bottom 原始大图同样先标定，再复用模板：

```bash
.venv/bin/python pipeline/20_run_traditional_operators.py \
  --input-root dataset/c789_100_left_bottom_parts/left/bottom/normal \
  --calibrate-normal-root dataset/c789_100_left_bottom_parts/left/bottom/normal \
  --preset c789_left_bottom_3x2 \
  --side bottom \
  --view uniform \
  --input-mode slot \
  --config config/traditional/c789.yaml \
  --output-dir results/c789_traditional/bottom_calibrated
```

```bash
.venv/bin/python pipeline/20_run_traditional_operators.py \
  --input-root dataset/c789_100_left_bottom_parts/left/bottom/defect \
  --preset c789_left_bottom_3x2 \
  --side bottom \
  --view uniform \
  --input-mode slot \
  --config config/traditional/c789.yaml \
  --template-dir results/c789_traditional/bottom_calibrated/calibration/templates \
  --geometry-thresholds results/c789_traditional/bottom_calibrated/calibration/geometry_thresholds.csv \
  --output-dir results/c789_traditional/bottom_defect
```

输出：

```text
calibration/
traditional_predictions.csv
traditional_cases.csv
traditional_summary.csv
traditional_summary.md
traditional_defect_evidence_confusion.csv
evidence/*.png
```

`--calibrate-normal-root PATH` 会从 normal 样本生成每个 slot 的模板和阈值；
`--template-dir PATH` 指向已标定的 slot template；`--geometry-thresholds PATH` 指向几何阈值
CSV。上线验收时建议把 normal/stress normal 和 defect 都跑成独立输出目录，避免新一轮
标定覆盖旧的 `calibration/` 产物。

`traditional_predictions.csv` 是 fusion-compatible CSV，字段包括：

```text
part_id,side,view,slot_id,branch,pred_label,score,threshold,
defect_type,evidence_type,gt_defect_type,reason,source_path,evidence_path,status
```

这里需要区分三件事：

- `defect_type`：该传统分支能声称的候选类型，例如 `crack`、`surface`、`geometry_delta`。
- `evidence_type`：证据形态，例如 `missing_mask`、`extra_mask`、`dark_line`、`texture_residual`。
- `gt_defect_type`：从人工文件名/路径解析出的真实标签类型，只用于离线评估和报告。

例如一个文件名为 `surface_*.png` 的样本被 `geometry` 命中时，正确解释是：
“该 surface 样本有 geometry_delta 证据”，而不是“传统算法把它分类成 less/more”。
`evidence_path` 指向每个分支的 overlay 图，便于解释为什么被判为
`NG`/`SUSPECT`/`WARN`。如果输入已经是 slot crop，可以加 `--input-mode slot`。

运行时默认逐张图片打印进度：

```text
[traditional] 1/100 dataset/...
```

`traditional_cases.csv` 会把多个 branch 行聚合到 `source_path + slot_id` 级别。
`traditional_summary.csv` / `traditional_summary.md` 会报告：

```text
normal_total
normal_false_positive
false_positive_rate
defect_total
defect_false_negative
false_negative_rate
defect_recall
unknown_total
```

`traditional_defect_evidence_confusion.csv` 和 summary markdown 里的
`GT Defect Type vs Primary Evidence` 表会显示真实缺陷类型和传统主证据的对应关系，用于检查
传统算子到底是在检测它擅长的结构/纹理/暗线证据，还是只是把样本判成了泛化 NG。

标签默认从路径推断：`normal` / `normal_test` / `stress_normal` 计为 normal，
`defect` 计为 defect。路径无法判断时会进入 `unknown_total`，不参与误报/漏报分母。
如果一次输入目录标签很明确，也可以手动指定：

```bash
--label normal
--label defect
```

如果不想打印逐图进度，可以加 `--no-progress`。

图片复核可以用 stage 24，把 dense CSV 转成 contact sheet 和单张定位图。卡片会分开显示
`GT` 和 `Evidence`，避免把 `geometry` 的 missing/extra 证据误读为真实缺陷类别。
stage 24 会优先使用 stage 20 的 `evidence_path` 精细 mask；如果没有 evidence 图，则从
`reason` 里的 `region=rXX_cYY` 生成 4x8 coarse region 热力框：

```text
red   missing_mask / raw_delta=less
blue  extra_mask / raw_delta=more
yellow dark_line
orange texture_residual
```

```bash
.venv/bin/python pipeline/24_visualize_traditional_results.py \
  --predictions-csv results/c789_traditional/top_defect_semantic_v2_evidence/traditional_predictions.csv \
  --cases-csv results/c789_traditional/top_defect_semantic_v2_evidence/traditional_cases.csv \
  --output-dir results/c789_traditional/visual_reports_semantic_evidence \
  --report-name top_defect_semantic_v2_evidence \
  --mode defect-all
```

输出包括：

```text
top_defect_semantic_v2_evidence_defect-all_page01.jpg
localization/*_localization.jpg
```

YOLO 第一版采用独立 Ultralytics fork，不把 Ultralytics 源码 vendoring 到 anomalib。
anomalib 只负责把 C789 slot crop 和人工 bbox 标注导出为 YOLO 检测数据集，并把 YOLO
推理结果接入 stage 18 融合。当前推荐先训练单类：

```text
defect
```

如果要专门采集一批 C789 YOLO 缺陷样本，可以每次在 6 个槽位都放缺陷件，然后运行：

```bash
.venv/bin/python pipeline/22_collect_c789_yolo_defects.py \
  --hand left \
  --position top \
  --defect-type scratch \
  --part-id yolo_batch001 \
  --group-count 20 \
  --images-per-group 1 \
  --raw-root dataset/c789_yolo_raw \
  --parts-root dataset/c789_yolo_parts \
  --defect-output-dir dataset/c789_yolo_defect_images \
  --overwrite
```

脚本会复用 stage 1 的相机采集和 stage 2 的 C789 预设裁剪，并用
`--defect-slot-mode all` 把同一张满盘缺陷图的 `slot01`-`slot06` 全部输出为 defect。
YOLO 待标注图不会做孔洞 inpaint/paint，stage 22 固定使用 `--hole-mask-method none`，
保留 crop 中的真实孔洞外观。
`--position bottom` 会自动使用 `c789_left_bottom_3x2`，裁剪数据仍按工作流写成
`bottom_ZS32`。最终待标注图片在 `--defect-output-dir`，同目录会写
`defect_image_manifest.csv`；完整 crop manifest 在 `--parts-root/part_crop_manifest.csv`。

准备 YOLO 数据集前，必须先对 slot crop 图像人工标 bbox。不要在原始 `4024x3036`
大图上标注，也不要把 `part_crop_manifest.csv` 里的 `slot_box` 当作缺陷框。缺少 bbox
的 defect crop 默认会 fail-closed 报错，避免把缺陷图误导出为空负样本。bbox CSV
应使用 crop 级 `processed_path` / `image_path` / `sample_id`；原图 `source_path` 会被拒绝。
导出目录非空时默认拒绝重写，需要明确传 `--overwrite` 清理旧 `images/labels`：

```bash
.venv/bin/python pipeline/21_prepare_yolo_dataset.py \
  --manifest dataset/c789_100_left_top_parts/part_crop_manifest.csv \
  --annotations dataset/c789_100_left_top_parts/bbox_annotations.csv \
  --output-root dataset/c789_yolo/left_top \
  --positive-val-ratio 0.2 \
  --normal-test-split val \
  --preview-dir results/c789_yolo/left_top_bbox_previews \
  --overwrite
```

导出结果包含：

```text
images/train, images/val, images/test
labels/train, labels/val, labels/test
data.yaml
export_manifest.csv
```

`labels/*.txt` 使用 Ultralytics 检测格式：

```text
class_id x_center y_center width height
```

坐标全部归一化到 `[0,1]`。`normal` / `normal_test` crop 会导出为空 label 负样本；
带 bbox 的 defect crop 会按 `--positive-val-ratio` 和 `--positive-test-ratio`
以 source/sample 为单位确定性拆分，只增强训练集，不增强 val/test。

Ultralytics fork 位于：

```bash
/home/yunjing/ultralytics-c789
```

最小训练命令：

```bash
cd /home/yunjing/ultralytics-c789
python examples/c789/train.py \
  --data-yaml /home/yunjing/anomalib/dataset/c789_yolo/left_top/data.yaml \
  --model yolo26n.pt \
  --epochs 100 \
  --imgsz 1024 \
  --batch 8 \
  --device 0 \
  --project /home/yunjing/anomalib/results/c789_yolo \
  --name left_top_defect
```

6 个 slot 共用一个 YOLO 模型；最终验收必须按 `slot01`-`slot06` 分别统计召回率，
确认没有某个 slot 系统性漏检。YOLO 推理结果写成 `yolo_predictions.csv` 后，通过
stage 18 参与融合：

```bash
.venv/bin/python pipeline/18_fuse_inspection_results.py \
  --branch-csv yolo=results/c789_yolo/yolo_predictions.csv \
  --output-dir results/c789_yolo/fused
```

如果 full-slot YOLO 在少量缺陷上泛化差，先做同分布验证，再做 ROI-level YOLO。stage 25
会按文件名里的 `gNNN` 采集组拆分：默认 `g002,g008` 作为验证组，训练组只增强 defect，
验证组保持真实未增强图；normal 来源只采空 label 负样本，不做离线增强：

```bash
.venv/bin/python pipeline/25_prepare_yolo_same_dist_dataset.py \
  --input-root /home/yunjing/ultralytics-c789/dataset/c789_all \
  --normal-source-root /home/yunjing/ultralytics-c789/dataset/c789_all_balanced_yolo \
  --output-root /home/yunjing/ultralytics-c789/dataset/c789_same_dist_yolo \
  --val-groups g002 g008 \
  --train-normal-limit 600 \
  --val-normal-limit 150 \
  --overwrite
```

stage 26 生成 ROI-level YOLO 数据集。`gt-center` 是诊断集：用标注 bbox 中心切 512x512
ROI，用来判断“小图 + 少背景”是否提升 YOLO 能力；`tile` 是部署集：推理时不知道 bbox，
因此用 512x512、stride 256 的网格覆盖 slot crop：

```bash
.venv/bin/python pipeline/26_prepare_yolo_roi_dataset.py \
  --input-root /home/yunjing/ultralytics-c789/dataset/c789_all \
  --normal-source-root /home/yunjing/ultralytics-c789/dataset/c789_all_balanced_yolo \
  --output-root /home/yunjing/ultralytics-c789/dataset/c789_roi_gt_center_yolo \
  --mode gt-center \
  --roi-size 512 \
  --stride 256 \
  --val-groups g002 g008 \
  --train-normal-ratio 2 \
  --val-normal-limit 150 \
  --overwrite

.venv/bin/python pipeline/26_prepare_yolo_roi_dataset.py \
  --input-root /home/yunjing/ultralytics-c789/dataset/c789_all \
  --output-root /home/yunjing/ultralytics-c789/dataset/c789_roi_tile_yolo \
  --mode tile \
  --roi-size 512 \
  --stride 256 \
  --val-groups g002 g008 \
  --overwrite
```

ROI 输出会写 `roi_manifest.csv`，其中包含 `roi_x1_in_slot` 等字段。为了避免训练到被
ROI 边界截断的错误 bbox，stage 26 只把完整落入 ROI 的框写入 label；GT-centered 模式下
源 bbox 太大、无法完整放入 512x512 ROI 的样本会写入 `roi_review_report.csv`，需要人工复核
标注是否过宽或是否应该改用更大的 ROI。ROI 预测框映射回 slot 坐标时使用：

```text
slot_x = roi_x1_in_slot + pred_x_in_roi
slot_y = roi_y1_in_slot + pred_y_in_roi
```

### ZS32 Label Studio Local Files 标注

先在 anomalib 仓库根目录准备 Label Studio staging 目录：

```bash
uv run python pipeline/27_prepare_zs32_label_studio.py
```

然后启动独立的 Label Studio 环境，并将生成目录设为 Local Files 文档根目录：

```bash
conda activate label-studio
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/home/yunjing/anomalib/dataset/zs32_yolo_labeling
label-studio start
```

在项目的 Source Storage 中新增本地文件存储，使用以下精确配置：

```text
Storage type: Local Files
Absolute local path: /home/yunjing/anomalib/dataset/zs32_yolo_labeling/images
Import method: Files
File Filter Regex: .*\.png$
Expected tasks: 660
```

把 `dataset/zs32_yolo_labeling/label_studio_config.xml` 的完整内容粘贴到 Labeling Interface。
每个任务对应一个特定视图；如果该视图中看不到缺陷，也要提交空 annotation，不要跳过任务。

标注完成并导出 YOLO 格式后，使用 stage 28 构建可直接训练的数据集：

```bash
.venv/bin/python pipeline/28_prepare_zs32_yolo_dataset.py
```

默认读取 `dataset/zs32_yolo_labeling/project-10-at-2026-07-12-12-29-b424b36b`，排除六视角均无框的
`left/20260711_181850_552955/less/group027`，并输出到 `dataset/zs32_six_view_yolo`。同一物理工件的
六视角以及真实/镜像正常对保持在同一 split；有框视角使用 Label Studio 标签，无可见缺陷的视角和正常图
使用空标签。生成的训练入口为 `dataset/zs32_six_view_yolo/data.yaml`。

完整图中缺陷框较小时，先用 stage 29 为六个视角各选择一个固定 ROI。程序依次为每个视角只显示一张
干净的右手正常参考图，对应视角的左右手件共用这个 ROI。鼠标拖框后按 Enter/Space 接受当前视角，
随后自动显示下一视角；六个视角全部完成后才写配置：

```bash
uv run --no-sync python pipeline/29_zs32_fixed_roi.py select
```

配置和检查图分别写到 `dataset/zs32_six_view_roi_config.json` 与
`dataset/zs32_six_view_roi_previews/`。然后转换全部 2010 张图及其标签：

```bash
uv run --no-sync python pipeline/29_zs32_fixed_roi.py convert
```

默认输出为 `dataset/zs32_six_view_roi_yolo`，训练入口是其中的 `data.yaml`。转换会保留原来的
train/val/test、`sample_id` 和空标签；跨越 ROI 边界的框会裁到边缘，完全位于 ROI 外的框会丢弃，
命令行与 manifest 分别报告裁框和丢框数量。重新生成已有输出时显式添加 `--overwrite`。

## 工业融合检测 MVP-1

### ZS32 六视角严格融合

ZS32 上线配置固定为 `config/fusion/zs32_six_view.json`。传统整视角模板匹配是第一个产品检测项目：
质量、身份和 ROI 可用性预检通过后，按 `front/front_left/front_right/back/back_left/back_right` 顺序执行
`template_match`。只有六个视角都明确 `PASS` 才调用 PatchCore、YOLO 和 geometry；任一 `REVIEW`、
`NG_TEMPLATE` 或模板资产异常都会立即停止并把未运行分支记录为 `SKIPPED`。模板 PASS 只允许继续，不能
单独形成最终 OK。

模板训练直接读取 stage 30 裁剪生成的 `crop_manifest.csv`。当代码运行在 fusion worktree、数据仍位于主目录时：

```bash
MAIN_ROOT=/home/yunjing/anomalib

uv run python pipeline/train_zs32_template_gate.py \
  --manifest "$MAIN_ROOT/dataset/zs32_patchcore_roi/crop_manifest.csv" \
  --path-root "$MAIN_ROOT" \
  --output-dir "$MAIN_ROOT/results/zs32_template_gate/v1" \
  --model-version zs32-models-2026.07.13 \
  --threshold-version zs32-thresholds-2026.07.13 \
  --roi-version zs32-roi-2026.07.12 \
  --template-version zs32-templates-2026.07.13
```

训练以 `(hand, view)` 建立 12 个组，正常模板图、阈值正常图和 test 按物理 `part_id` 隔离；不会使用默认
`0.8` 阈值，也不会覆盖已存在的输出目录。风险分数统一为 `risk=1-similarity`：`risk<T_low` 为 PASS，
`T_low<=risk<T_high` 为 REVIEW，`risk>=T_high` 为 NG_TEMPLATE。模型 JSON 和单图预测会保留连续分数、
双阈值、最佳模板、位移、模板 SHA-256 和四类版本。`model.sha256` 锁定包含在线阈值和模板索引的
`model.json`，模板相对路径也不得逃出模型目录。训练目录还会输出 `calibration_rows.csv`，其模板分数与
模型阈值同源，供 Stage 31 合并到其它五类分支的总校准 CSV。单图门禁可独立检查：

```bash
uv run python pipeline/predict_zs32_template_gate.py \
  --model-dir "$MAIN_ROOT/results/zs32_template_gate/v1" \
  --image /absolute/path/to/front_roi.png \
  --hand right \
  --view front \
  --output-json /tmp/front_template_gate.json
```

退出码 `0/10/20/2` 分别表示 PASS/REVIEW/NG_TEMPLATE/模板资产或输入无效。在线 Python 编排接口位于
`capture_data/zs32_inspection_orchestrator.py`；下游 runner 同时接收 `InspectionRequest` 和六条模板结果，
可用 `template_results_to_branch_rows()`/`write_template_match_csv()` 生成 Stage 18 输入。短路时不会伪造
后续分支 CLEAR，也不会调用 Stage 18。

完成模板门禁后，再用按物理 `part_id` 划分的
calibration/test 分数离线拟合各 `hand/view/branch/model_version/roi_version` 组的双阈值；Stage 31
默认读取这一份 strict profile，自动要求其中 72 个精确的左/右手分支组，但不会原地修改上线配置：

```bash
uv run python pipeline/31_calibrate_zs32_fusion.py \
  --input-csv results/zs32_fusion/calibration_rows.csv \
  --output-dir results/zs32_fusion/calibration_v1 \
  --target-recall 1.0 \
  --normal-quantile 0.995
```

校准输入字段为 `part_id,hand,view,branch,raw_score,gt_label,split,model_version,roi_version`。
输出目录包含 `thresholds.json`、`thresholds.csv`、`calibration_metrics.json` 和
`calibration_summary.md`；阈值记录同时给出 `low_threshold`、`high_threshold`、normal/defect 样本数和
`status`。`thresholds.json` 还写入 stage 18 共享的 product/profile/hand/side 身份、全部
`expected_versions`、profile SHA-256、`threshold_versions` 与阈值记录 SHA-256；任一必需组缺失两类
标定数据都会保持 `insufficient_data`/`calibration_valid=false`。标定无效时，`escape_rate`、recall、
normal reject rate 和 review rate 等安全率全部为 `null`；未经合同验证的原始观测率只保留为
`diagnostic_observed_*`，不可当作上线指标。

`thresholds.json` 是不可绕过的上线 bundle：包含 `artifact_schema/version`、`calibration_valid`、
profile/config SHA-256、72 个 exact group、canonical `threshold_records_sha256` 以及排除自身哈希字段后重算的
`artifact_sha256`。不要手工修改该文件，也不要从 branch CSV 反向生成阈值 bundle。

生产融合命令为：

```bash
uv run python pipeline/18_fuse_inspection_results.py \
  --profile zs32 \
  --manifest results/zs32_fusion/inspection_manifest.csv \
  --quality-csv results/zs32_fusion/quality_gate.csv \
  --registration-csv results/zs32_fusion/registration.csv \
  --template-match-csv results/zs32_fusion/template_match.csv \
  --branch-csv anomaly_front=results/zs32_fusion/anomaly_front.csv \
  --branch-csv anomaly_front_left=results/zs32_fusion/anomaly_front_left.csv \
  --branch-csv anomaly_front_right=results/zs32_fusion/anomaly_front_right.csv \
  --branch-csv anomaly_back=results/zs32_fusion/anomaly_back.csv \
  --branch-csv anomaly_back_left=results/zs32_fusion/anomaly_back_left.csv \
  --branch-csv anomaly_back_right=results/zs32_fusion/anomaly_back_right.csv \
  --branch-csv yolo=results/zs32_fusion/yolo.csv \
  --branch-csv geometry=results/zs32_fusion/geometry.csv \
  --threshold-artifact results/zs32_fusion/calibration_v1/thresholds.json \
  --output-dir results/zs32_fusion/fused_v1
```

严格 profile 不需要额外开关就会强制源图和证据文件完整；`--require-complete-evidence` 仅保留给兼容流程。
各 branch CSV 会标准化为 `branch_predictions.csv`，字段含 `part_id`、`product`、`profile`、`hand`、
`capture_session`、`group_id`、`side`、`view`、`slot_id`、`source_hash`、`evidence_hash`、`manifest_identity`、
`branch`、`pred_label`、连续 `score`、旧单阈值 `threshold`、`low_threshold`、`high_threshold`、
`evidence_level`、`defect_type`、`evidence_type`、`gt_defect_type`、`reason`、`source_path`、
`evidence_path`、`status` 以及 `model_version`、`threshold_version`、`roi_version`、
`template_version`、`detections`。严格 ZS32 的每一行必须携带非空且工件内完全一致的
`capture_session/group_id`；缺失或混用会进入 `INVALID_CAPTURE`。`fused_predictions.csv` 的字段为
`part_id`、`final_status`、`final_label`、
`defect_side`、`defect_view`、`defect_slot`、`defect_type`、`triggered_branch` 和 `reason`；
`summary.md` 汇总 parts/branches、OK、NG、REVIEW、RETAKE、INVALID_CAPTURE、兼容模式的
`suspect` 及完整/不完整检查数。

最终状态只有五类语义：`OK`、`REVIEW`、`RETAKE`、`INVALID_CAPTURE` 和具体的 `NG_*`
（`NG_TEMPLATE`/`NG_ANOMALY`/`NG_GEOMETRY`/`NG_YOLO`）。只有六视角、每视角全部必需 branch、质量与配准 PASS、
版本一致、证据文件完整且全部为 CLEAR 才能自动 `OK`；任一 STRONG 直接进入对应 `NG_*`，任一 GRAY、
缺 branch 或版本不一致进入 `REVIEW`，采集身份/完整性错误进入 `INVALID_CAPTURE`，质量或配准失败进入
`RETAKE`。YOLO 无框只是该 YOLO 行的 CLEAR 证据，不能抵消其它视角或分支的 GRAY/STRONG。
一旦已有合法 STRONG，缺证据、版本/运行故障或采集错误只会追加 system trigger、使
`inspection_complete=false` 并阻止发布，不会把不可变的机器 `NG_*` 降为 REVIEW/INVALID_CAPTURE。
严格 ZS32 的质量与配准行必须在 `status` 中显式写 `PASS`；字段缺失、空白或任意其它值都属于 gate
fault，不能发布 OK。每个 non-PASS 行都会作为独立 trigger 保留，不会被第一个 gate 或 STRONG 证据覆盖；
已有 STRONG 时机器 NG 保持不变，但检查仍为不完整且不可放行。非严格兼容流程仍可沿用 `pred_label=0`。

YOLO 每个视角只提交一条 summary branch identity，`detections` 是 JSON list。缺字段或空白表示 evidence
缺失，与显式无框 `[]` 不同；严格 ZS32 只接受后者作为 CLEAR。每个 box 必须包含合法的 class、有限且位于
`[0,1]` 的 confidence、四个递增有限坐标 `xyxy`、正有限 area，所带 ROI/border flags 必须为 boolean；
多框不会被当成重复 required branch。缺失或畸形 summary 会进入不可放行诊断代际并非零退出。

正面三个视角只能产生 `FRONT_CLEAR`、`FRONT_REVIEW` 或 `FRONT_NG`，此时 `final_status` 仍为空；
翻面后，同一 `part_id` 的背面三个视角产生对应 `BACK_*`，仅 `FRONT_CLEAR + BACK_CLEAR` 能组合为
最终 `OK`。其它组合保持 REVIEW 或 NG，不能以多数投票覆盖强阳性。

严格 profile 默认把逐工件审计写入 `results/zs32_fusion/fused_v1/audit/<part_id>.json`。JSON 保存
`schema_version`、`capture_session/group_id`、六视角的每一行原始分数/双阈值/阈值 margin、路径与 SHA-256、
模型/阈值/ROI/模板版本、structured YOLO boxes、locked threshold artifact path/file/artifact/record hashes、
全部触发证据，以及彼此独立的 `machine_status`、`review.status`、`review_status`、`released_status`。
输入没有真实采集 timestamp 时审计字段保持 `null`，不会把当前时间冒充为采集证据。
机器结果不会被人工结论覆盖；初始 `review.status=PENDING`，`review_status=null` 且
`released_status=null`。人工复检必须查看原图、热图/检测框/几何 overlay 和所有触发原因，再由独立发布
流程填写复检与放行状态。

stage 18 会先核对 locked bundle 的 schema、有效标定状态、profile hash、72 组完整性、两层哈希，
再逐行要求 CSV `low/high` 与 bundle 的权威数值完全一致。缺失、篡改、不完整、标定无效或数值不符都会
先发布不可放行的诊断代际，再非零退出。然后 stage 18 把 `branch_predictions.csv`、`fused_predictions.csv`、
`summary.md`、`threshold_artifact.json` 和所有 audit JSON 先写入唯一
sibling staging 目录，只在哈希、序列化和所有写入完成后执行一次目录 rename。已存在的
`--output-dir` 会直接拒绝；任一发布失败都不会留下可消费的 OK 代际。严格输入的双阈值或
`evidence_level` 畸形时，CLI 会先发布不可放行的诊断代际，然后以非零状态退出。

`pipeline/18_fuse_inspection_results.py` 是新的 fail-closed 融合入口。它不会覆盖
`pipeline/12_geometry_eval.py` 生成的旧版 `fused_predictions.csv`，而是把现有
`geometry_predictions.csv`、AnomalyDINO `predictions.csv`、可选 `quality_gate.csv` 和
`registration_results.csv` 统一转换成：

```text
branch_predictions.csv
fused_predictions.csv
summary.md
```

最小用法：

```bash
.venv/bin/python pipeline/18_fuse_inspection_results.py \
  --geometry-csv results/c789_100_hardened/left_top_geometry/manual_defect_fused/geometry_predictions.csv \
  --anomaly-csv results/c789_100_hardened/left_top_anomaly_dino/reports/predictions.csv \
  --output-dir results/c789_robustness_v2/fused
```

如果已经有多视角 manifest 或 fusion config，可以额外传：

```bash
--manifest results/c789_multiview/manifest.csv
--fusion-config config/fusion/c789.yaml
--required-view top:uniform
--branch-csv traditional=results/c789_traditional/top_normal/traditional_predictions.csv
--branch-csv yolo=results/c789_yolo/yolo_predictions.csv
```

当 required side/view 缺失时，融合结果会输出 `INVALID_CAPTURE`；quality 或 registration
失败时输出 `RETAKE`；geometry、crack、AnomalyDINO、EfficientAD、YOLO 阳性时分别输出
`NG_GEOMETRY`、`NG_CRACK`、`NG_ANOMALY`、`NG_GLOBAL`、`NG_YOLO`。启用
`suspect_policy.near_threshold_ratio` 后，接近阈值但未阳性的分支会输出 `SUSPECT`。
如果 `fusion_config.ok_requires.required_sides` 和 `required_views` 已配置，stage 18 会在
没有 manifest 的情况下也根据 branch CSV 里的 `side/view` 检查缺失视角。

注意：MVP-1 的最小 CSV 模式为了复用现有 geometry/anomaly 报告，不会默认强制要求
quality gate 与 registration CSV。用于上线式严格 OK 判定时，应传入包含
`ok_requires.quality_gate: PASS`、`ok_requires.registration: PASS`、`required_sides` 和
`required_views` 的 `--fusion-config`。

`pipeline/19_run_robustness_benchmark.py` 在 MVP-1 阶段不重跑模型或几何评估，
因此不会生成大规模训练结果。它会枚举传入的 normal/defect/invalid root；如果某个输入图片
没有任何 branch prediction，会计入 `missing_prediction_count`，避免 recall 被虚高。
defect/normal/stress 输入会以 `benchmark_input` 写入 `branch_predictions.csv`；
invalid 输入会以 `missing_prediction` 写入，并在详情里标记
`not_evaluated/missing_prediction`，不计入 invalid reject 成功数。

路径匹配优先使用 `part_id/sample_id/id`、完整路径和 resolved 路径；`basename/stem`
只在唯一匹配时作为弱匹配使用，避免 clean/defect/stress/invalid 目录下同名图片互相污染。
对无预测 defect 输入，benchmark 会尽量从文件名解析 `defect_type` 和 `slot_id`，让
`by_defect_type.csv`、`by_slot.csv` 的分母包含漏检样本。示例：

```bash
.venv/bin/python pipeline/19_run_robustness_benchmark.py \
  --clean-normal-root dataset/c789_100_left_top_parts/left/top/normal_test \
  --stress-normal-root dataset/c789_stress_normal_group_split/locked/left/top \
  --defect-root dataset/c789_100_left_top_parts/left/top/defect \
  --geometry-csv results/c789_100_hardened/left_top_geometry/manual_defect_fused/geometry_predictions.csv \
  --anomaly-predictions results/c789_100_hardened/left_top_anomaly_dino/reports/predictions.csv \
  --output-dir results/c789_robustness_v2
```

输出包括 `robustness_summary.md`、`robustness_summary.csv`、`by_defect_type.csv`、
`by_slot.csv`、`misses.csv`、`false_positives.csv`、`retake_cases.csv`。
`robustness_summary.csv` 会报告 `defect_recall`/`fused_recall`、
`geometry_recall`、`anomaly_dino_recall`、normal FP、stress normal FP 和
invalid reject rate。这里的 `invalid_reject_rate` 只统计已评估并被拒绝的 invalid 样本；
无预测 invalid 样本会进入 `not_evaluated_missing_prediction_count`。
