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
