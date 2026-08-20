# ZS32 0727 + 唯一缺陷：Template / PatchCore v14 训练

日期：2026-07-27

## 已完成的数据准备

本次使用：

- normal：`dataset/zs32_0727`，104 个不同实物零件，832 张八视角图片。
- defect：`dataset/zs32_all/right` 中的缺陷数据。
- 已排除重复 session：
  `zs32_right_deform_20260714_201404_438451409`。
- 过滤后 defect：23 个唯一实物零件，184 张八视角图片：
  deform 12、less 2、others 9。
- 两个 secondary 视角均参与 Template 和 PatchCore。
- 没有 mask，只能使用 image/sample-level 指标。

新 normal ROI：

`dataset/zs32_0727_right_normal_roi_104part_v1`

正式训练 release：

`dataset/zs32_0727_plus_zs32_all_unique_defect_release_v1`

关键文件：

- 实体划分：
  `physical_part_splits.csv`
- Template：
  `template_manifest.csv`
- PatchCore：
  `patchcore/crop_manifest.csv`
- 完整 PatchCore/holdout 路由：
  `patchcore_manifest.csv`
- 汇总：
  `release_summary.json`

release 使用指向当前工作区源 ROI 图片的绝对文件级软链接，目录本身约 14 MB，不复制旧缺陷数据。
训练完成前不要移动、改名或修改以下两个源目录，否则软链接或内容哈希校验会失效：

- `dataset/zs32_0727_right_normal_roi_104part_v1`
- `dataset/zs32_all_right_patchcore_roi`

## 实体级划分

| label | train | model_val | calibration | final_test |
|---|---:|---:|---:|---:|
| normal | 62 | 16 | 13 | 13 |
| defect | 0 | 6 | 11 | 6 |

缺陷分层结果：

| defect type | model_val | calibration | final_test |
|---|---:|---:|---:|
| deform | 3 | 6 | 3 |
| less | 0 | 1 | 1 |
| others | 3 | 4 | 2 |

每个实体的八个视角始终属于同一个 role。Template 只从 normal/train 选择模板；defect/calibration
参与阈值放置，defect/model_val 和 defect/final_test 只评分。

PatchCore 的 Memory Bank 只使用 62 个 normal/train；13 个 normal/calibration 进入 `normal_test`，
11 个 defect/calibration 进入 `defect`。model_val/final_test 保留在 release 的 holdout 目录，不进入本次
Memory Bank 或部署阈值计算。

## 训练前检查

```bash
cd /home/yunjing/anomaly_xingtao_new

export UV_CACHE_DIR=/tmp/uv-cache
export MPLCONFIGDIR=/tmp/matplotlib-cache

UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run --no-sync python \
  pipeline/prepare_zs32_0727_template_patchcore.py validate \
  --output-root dataset/zs32_0727_plus_zs32_all_unique_defect_release_v1

nvidia-smi

UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -c \
  'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

Template 本身使用 OpenCV，不依赖 GPU；PatchCore 命令必须在第二条 GPU 检查成功后运行。
`validate` 返回 `"status": "VALID"` 后再开始训练；它会检查实体划分、八视角、缺陷类型分层、
Template/PatchCore 路由、软链接和图片内容哈希。

## Template v14 训练命令

注意：本次不能添加 `--normal-only`，否则 defect 会被全部忽略。

```bash
cd /home/yunjing/anomaly_xingtao_new

UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run --no-sync python \
  pipeline/train_zs32_template_gate.py \
  --manifest dataset/zs32_0727_plus_zs32_all_unique_defect_release_v1/template_manifest.csv \
  --output-dir results/zs32_template_gate_right_0727_plus_defect_eight_view_v14 \
  --required-hand right \
  --width 512 \
  --max-shift 12 \
  --templates-per-group 5 \
  --max-train-per-group 120 \
  --normal-quantile 1.0 \
  --evaluation-fraction 0 \
  --model-version zs32-right-0727-plus-defect-eight-view-template-v14 \
  --threshold-version zs32-right-0727-plus-defect-threshold-v14 \
  --roi-version zs32-eight-view-roi-9412b2838cdb \
  --template-version zs32-right-0727-plus-defect-eight-view-template-v14
```

输出目录必须不存在。该命令会生成八组 Template、`model.json`、`model.sha256` 和
`calibration_rows.csv`。

## PatchCore v14 训练命令

本地 Hugging Face 缓存已经存在
`timm/wide_resnet50_2.racm_in1k`，因此可以继续使用离线模式：

```bash
cd /home/yunjing/anomaly_xingtao_new

export UV_CACHE_DIR=/tmp/uv-cache
export MPLCONFIGDIR=/tmp/matplotlib-cache

HF_HUB_OFFLINE=1 bash pipeline/run_patchcore_roi_eight_views.sh \
  dataset/zs32_0727_plus_zs32_all_unique_defect_release_v1/patchcore \
  results/zs32_patchcore_eight_view_0727_plus_defect_seed42_v14 \
  0
```

该 runner 会按以下顺序串行训练八个模型：

```text
right_front
right_front_left
right_front_right
right_front_secondary
right_back
right_back_left
right_back_right
right_back_secondary
```

固定参数为 WRN50、layer2+layer3、image size 256、coreset ratio 0.05、k=9、FP32、
batch size 16、deploy FPR 0.05、seed 42。命令可恢复：已有完整 summary 的视角会跳过。

## 训练完成后的最小检查

```bash
test -f results/zs32_template_gate_right_0727_plus_defect_eight_view_v14/model.json
test -f results/zs32_patchcore_eight_view_0727_plus_defect_seed42_v14/eight_view_summary.csv

sha256sum \
  results/zs32_template_gate_right_0727_plus_defect_eight_view_v14/model.json \
  results/zs32_patchcore_eight_view_0727_plus_defect_seed42_v14/eight_view_summary.csv
```

训练完成不代表自动接入当前 Demo 平台。应先检查 Template 各视角 normal/defect 分布、
PatchCore 的八视角 summary、false positives/false negatives 和 holdout 结果，再单独更新
`configs/zs32/zs32_demo.json`。
