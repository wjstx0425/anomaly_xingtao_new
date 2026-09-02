# BMW 左手件最短训练流程

代码运行目录：

```bash
cd /home/yunjing/anomaly_xingtao_new/.worktrees/bmw-eight-view-handoff
```

先设置本次会话和发布名称：

```bash
SESSION_ID=替换为左手normal采集会话
DATASET_ID=bmw_left_normal_20260814_v1
ROI_ID=bmw_left_normal_20260814_roi_v1
TRAINING_ID=bmw_left_normal_20260814_training_v1
RUN_ID=bmw_left_normal_20260814_models_v1
```

## 1. 准备左手 normal 数据

```bash
uv run --no-sync python pipeline/bmw_lab_prepare_eight_view_data.py \
  --raw-root /home/yunjing/anomaly_xingtao_new/dataset/bmw_lab_raw_clean \
  --output-root /home/yunjing/anomaly_xingtao_new/dataset/bmw_lab_prepared \
  --hand left \
  --session-id "$SESSION_ID" \
  --dataset-id "$DATASET_ID" \
  --skip-image-hash
```

## 2. 选择左手八视图公共 ROI

从已经采完的 raw 会话选择一件完整 normal。八个 ROI 供 Template、EfficientAD 和 YOLO 现场裁剪共同使用。

```bash
uv run --no-sync python pipeline/bmw_lab_select_eight_view_rois.py \
  --raw-root /home/yunjing/anomaly_xingtao_new/dataset/bmw_lab_raw_clean \
  --hand left \
  --source-class normal \
  --sample-id bmw_normal_group001_000001 \
  --session-id "$SESSION_ID" \
  --profile-id "$ROI_ID" \
  --output "configs/bmw/rois/${ROI_ID}.json" \
  --max-display-width 1280 \
  --max-display-height 720
```

如果 `group001` 不存在，将 `--sample-id` 换成该会话中任意一个完整 normal sample。

## 3. Dry-run 后训练 Template 和 EfficientAD

```bash
uv run --no-sync python pipeline/bmw_lab_train_left_normal.py \
  --prepared-root "/home/yunjing/anomaly_xingtao_new/dataset/bmw_lab_prepared/${DATASET_ID}" \
  --roi-config "configs/bmw/rois/${ROI_ID}.json" \
  --training-root /home/yunjing/anomaly_xingtao_new/dataset/bmw_lab_training \
  --training-id "$TRAINING_ID" \
  --output-root /home/yunjing/anomaly_xingtao_new/results/bmw_lab_one_click \
  --run-id "$RUN_ID" \
  --stage train \
  --efficientad-epochs 30 \
  --gpu 0 \
  --workers 8 \
  --dry-run
```

dry-run 成功后去掉最后一行 `--dry-run`，执行真实训练。此流程不会训练 YOLO。

## 4. 从训练 ROI 空白选择 EfficientAD mask

```bash
uv run --no-sync python pipeline/bmw_lab_select_efficientad_ignore_masks.py \
  --training-release "/home/yunjing/anomaly_xingtao_new/dataset/bmw_lab_training/${TRAINING_ID}" \
  --roi-config "configs/bmw/rois/${ROI_ID}.json" \
  --output "/home/yunjing/anomaly_xingtao_new/results/bmw_efficientad_manual_ignore_masks/${RUN_ID}_mask_v1" \
  --max-display-width 1280 \
  --max-display-height 900
```

不传 `--from-index` 即从空白 mask 开始。mask 只画固定工装、背景和确定不检测的区域。

## 5. 使用连通域评分重新标定 EfficientAD

```bash
uv run --no-sync python pipeline/bmw_lab_train_left_normal.py \
  --prepared-root "/home/yunjing/anomaly_xingtao_new/dataset/bmw_lab_prepared/${DATASET_ID}" \
  --roi-config "configs/bmw/rois/${ROI_ID}.json" \
  --training-root /home/yunjing/anomaly_xingtao_new/dataset/bmw_lab_training \
  --training-id "$TRAINING_ID" \
  --output-root /home/yunjing/anomaly_xingtao_new/results/bmw_lab_one_click \
  --run-id "$RUN_ID" \
  --stage calibrate \
  --mask-index "/home/yunjing/anomaly_xingtao_new/results/bmw_efficientad_manual_ignore_masks/${RUN_ID}_mask_v1/index.json" \
  --component-policy configs/bmw/efficientad_component_filter_lab_v1.json \
  --gpu 0 \
  --workers 8
```

该评分会过滤小且浅的孤立热点，同时保留小而强、细而长和较大成片异常。阈值使用左手 normal calibration 数据重新生成，不复用右手阈值。

## 6. 后续采集 no_streak 后训练光痕

先用一张左手合格 `front_left` HDR 原图选择四点倾斜 ROI：

```bash
uv run --no-sync python pipeline/bmw_lab_select_bright_streak_rotated_roi.py \
  --image /绝对路径/left_front_left_normal_xxx_fused.png \
  --output /home/yunjing/anomaly_xingtao_new/results/bmw_bright_streak_rotated_roi/bmw_left_20260814_v1/roi.json
```

取得 ROI SHA：

```bash
sha256sum /home/yunjing/anomaly_xingtao_new/results/bmw_bright_streak_rotated_roi/bmw_left_20260814_v1/roi.json
```

然后重训练 V3：

```bash
uv run --no-sync python pipeline/bmw_lab_retrain_rotated_bright_streak.py \
  --normal-manifest "/home/yunjing/anomaly_xingtao_new/dataset/bmw_lab_prepared/${DATASET_ID}/manifests/bright_streak.csv" \
  --no-streak-image /绝对路径/left_front_left_defect_no_streak_xxx_fused.png \
  --rotated-roi /home/yunjing/anomaly_xingtao_new/results/bmw_bright_streak_rotated_roi/bmw_left_20260814_v1/roi.json \
  --rotated-roi-sha256 替换为上一步SHA \
  --output-dir /home/yunjing/anomaly_xingtao_new/results/bmw_bright_streak_rotated_retrain/bmw_left_normal_no_streak_20260814_v1
```

## YOLO

不重新训练。继续复用左右手联合模型：

```text
/home/yunjing/anomaly_xingtao_new/results/bmw_lab_one_click/bmw_right_multisource_left_yolo_v1/yolo/train/weights/best.pt
SHA-256: 0e9591f2fa2487ad12000847f1d80137901ed69989e0e95cb5c8907b95ba3913
```
