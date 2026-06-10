# 缺陷检测 Pipeline 快速指南

所有命令都在仓库根目录执行：

```bash
cd /home/yunjing/anomalib
uv sync
```

下面示例统一使用 `.venv/bin/python`。如果已经激活虚拟环境，也可以直接用
`python`。

## 流程总览

```text
1_collect_data.py   采集原始大图
2_process_data.py   裁成单零件图片，可 mask 孔
3_train_model.py    预处理、训练、评估
4_inference.py      离线推理并输出复核图
5_demo_inspection.py 现场双面演示
```

最常用流程：

```text
采 normal/defect -> 裁剪 -> 训练评估 -> 推理复核或现场演示
```

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

关键参数：

| 参数 | 说明 |
| --- | --- |
| 第一个位置参数 | 图片文件或图片目录 |
| `--output-root` | 对应训练结果目录 |
| `--view` | 视角，必须和训练一致 |
| `--model` | 模型名，必须和训练一致 |
| `--input-is-preprocessed` | 输入已经是单零件 crop 时使用 |
| `--visualize` | 额外保存可视化图 |

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
- 指标好不代表能上线，建议再采一批新正常件和新缺陷件做独立验证。
- EfficientAD 可能需要额外本地资源；没有资源时先用 AnomalyDINO 或 PatchCore。
