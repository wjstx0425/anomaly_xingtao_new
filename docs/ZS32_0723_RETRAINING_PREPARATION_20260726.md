# ZS32 0723 八视角重训练准备记录

日期：2026-07-26

## 当前结论

- 可用新数据固定为 `zs32_0723` 会话 `zs32_4cam_accept_20260723_163955_301282113`。
- 数据包含 106 个不同实物零件，每个零件八个视角，共 848 张右手正常图。
- 固定 seed 42，按实物零件隔离为：train 64、model_val 16、calibration 13、final_test 13。
- 正式训练输入为 `dataset/zs32_all_plus_0723_retraining_release_v2`。
- `release_v1` 是发现 Template 角色混用后保留的旧准备产物，不得用于训练。
- 当前 Codex 沙箱看不到可用 NVIDIA GPU；用户已在可用 GPU 的终端完成下述 Template、PatchCore、YOLO
  正式训练，产物及接入结果见文末。

## 输入与审计

- 新 ROI crop manifest：`dataset/zs32_0723_patchcore_roi_v1/crop_manifest.csv`
  - 848 行，八视角各 106 行。
  - SHA256：`c5fe1e3af8f958d909bf4030d200200ea1d9e9c5b5dcc602cf84ae1b66ae0707`
- 旧 PatchCore manifest：`dataset/zs32_all_right_patchcore_roi/crop_manifest.csv`
  - SHA256：`071bcc6d9ee61667699d931f98eeb47abfed8381629fa65460b41642c12e232b`
- 旧 YOLO mapping：`dataset/zs32_defect_positive_patchcore_normal_yolo_20260715_v2/mapping.csv`
  - SHA256：`3c95c139c033498f9192b17ed3fed304464fea7c75ffdeddd6f3cdf4c5f6b09f`
- 八视角 ROI SHA256：
  `9412b2838cdb96f722db356714cf9bbbb5ea01810671ccf92b67323de77ebe65`
- ROI 抽查图：
  - `artifacts/zs32_0723_roi_audit_v1/roi_overlay_contact_sheet.jpg`
  - `artifacts/zs32_0723_roi_audit_v1/roi_crop_contact_sheet.jpg`

正式 release 验证结果：

```text
physical parts: 106
release rows: 5327
PatchCore rows: 2852
Template rows: 848
YOLO rows: 1627
status: VALID
```

数据角色：

- PatchCore：六个主视角使用旧数据 + 0723；两个 secondary 只使用 0723。新数据 train 进入 normal，
  calibration 进入 normal_test，model_val/final_test 保持 holdout。
- Template：只使用 0723 八视角。train 只选模板，calibration 只确定临时阈值，model_val 和 final_test
  只评分，不参与模板或阈值选择。
- YOLO：保留旧双手 train/val bbox；旧 test 只作为 retrospective_test；0723 train/model_val
  分别加入 train/val 且标签严格为空，calibration/final_test 不进入训练 YAML。

## 训练前 GPU 门

```bash
cd /home/yunjing/anomaly_xingtao_new
export UV_CACHE_DIR=/tmp/uv-cache
export MPLCONFIGDIR=/tmp/matplotlib-cache

nvidia-smi
uv run --no-sync python -c \
  'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

只有第二条命令成功并打印 GPU 名称后，才运行 PatchCore 和 YOLO。

## Template：0723 八视角

```bash
uv run --no-sync python pipeline/train_zs32_template_gate.py \
  --manifest dataset/zs32_all_plus_0723_retraining_release_v2/template_manifest.csv \
  --output-dir results/zs32_template_gate_right_0723_eight_view_v11 \
  --required-hand right \
  --width 512 \
  --max-shift 12 \
  --templates-per-group 5 \
  --max-train-per-group 120 \
  --normal-quantile 1.0 \
  --normal-only \
  --evaluation-fraction 0 \
  --model-version zs32-right-0723-eight-view-template-v11 \
  --threshold-version zs32-right-0723-temporary-threshold-v11 \
  --roi-version zs32-eight-view-roi-9412b2838cdb \
  --template-version zs32-right-0723-eight-view-template-v11
```

## PatchCore：八视角

```bash
HF_HUB_OFFLINE=1 bash pipeline/run_patchcore_roi_eight_views.sh \
  dataset/zs32_all_plus_0723_retraining_release_v2/patchcore \
  results/zs32_patchcore_eight_view_all_plus_0723_seed42_v11 \
  0
```

该 runner 按八视角串行运行且可恢复。两个 secondary 没有缺陷正样本，其正常阈值可用于 commissioning，
但缺陷 recall、缺陷 AUROC 等指标没有有效真值，不能作为缺陷能力结论。

2026-07-26 首次运行时六个主视角完成，两个 secondary 因 Folder datamodule 无条件要求 `defect/`
目录而失败。工作流已修正为：仅在真实 defect 目录存在时传入 `abnormal_dir`。重新运行同一条命令即可；
六个已有 summary 的主视角会跳过，两个 secondary 复用已有 preprocess manifest 继续训练。

## YOLO：旧双手标签 + 0723 右手正常图

```bash
uv run --no-sync yolo detect train \
  model=/home/yunjing/ultralytics-c789/yolo26n.pt \
  data=/home/yunjing/anomaly_xingtao_new/dataset/zs32_all_plus_0723_retraining_release_v2/yolo/data.yaml \
  project=/home/yunjing/anomaly_xingtao_new/results/yolo \
  name=zs32_all_plus_0723_normal_n1280_seed42_v11 \
  epochs=200 patience=40 batch=16 imgsz=1280 device=0 workers=8 \
  save=True save_period=-1 cache=False exist_ok=False pretrained=True optimizer=auto \
  seed=42 deterministic=True single_cls=True rect=False cos_lr=False \
  close_mosaic=10 resume=False amp=True fraction=1.0 val=True split=val \
  lr0=0.01 lrf=0.01 momentum=0.937 weight_decay=0.0005 \
  warmup_epochs=3.0 warmup_momentum=0.8 warmup_bias_lr=0.1 \
  box=7.5 cls=0.5 dfl=1.5 \
  hsv_h=0.005 hsv_s=0.2 hsv_v=0.15 \
  degrees=0.0 translate=0.03 scale=0.1 shear=0.0 perspective=0.0 \
  flipud=0.0 fliplr=0.0 mosaic=0.0 mixup=0.0 cutmix=0.0
```

本地初始权重 SHA256：
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`。

## 训练完成后的最小检查

```bash
test -f results/zs32_template_gate_right_0723_eight_view_v11/model.json
test -f results/zs32_patchcore_eight_view_all_plus_0723_seed42_v11/eight_view_summary.csv
test -f results/yolo/zs32_all_plus_0723_normal_n1280_seed42_v11/weights/best.pt

sha256sum \
  results/zs32_template_gate_right_0723_eight_view_v11/model.json \
  results/zs32_patchcore_eight_view_all_plus_0723_seed42_v11/eight_view_summary.csv \
  results/yolo/zs32_all_plus_0723_normal_n1280_seed42_v11/weights/best.pt
```

训练完成后再基于这三个实际 SHA 生成新的 24-group commissioning bundle。阈值先采用模型生成的默认值，
之后可手工发布新的不可变阈值版本。由于没有 mask，pixel AUROC/AUPRO 固定记为 N/A。

## v11 检测平台接入结果

三类训练产物已完成校验并接入：

- Template：`results/zs32_template_gate_right_0723_eight_view_v11/model.json`
  - SHA256：`ef0d99139c1294db0681b99e6ef277d9acf41dc2240fd12ddfc551e90407fc6d`
- PatchCore：`results/zs32_patchcore_eight_view_all_plus_0723_seed42_v11/eight_view_summary.csv`
  - SHA256：`fb6965310efe17cc9671af4dbf0ffc5b894fa5e9c542bf05063c60ec7b9770bb`
- YOLO：`results/yolo/zs32_all_plus_0723_normal_n1280_seed42_v11/weights/best.pt`
  - SHA256：`1375ee156d10093afd689cc70daf6b243264a4dc49c7cb1c39293cb780434528`

新的不可变接入链：

- runtime assets：`results/zs32_runtime_assets_eight_view_v11`
- 24-group thresholds：`results/zs32_24group_0723_v11_commissioning/thresholds.json`
- final bundle：`results/zs32_runtime_bundle_eight_view_v11/runtime_bundle.json`
- bundle SHA256：`d5b6f5cf336a8b0153e11f56a55dc11c000d979bdddee92080b2135bf08df86c`

该 bundle 要求八个视角各执行 Template、PatchCore、YOLO，共 24 组；`front_secondary` 和
`back_secondary` 均不得 skipped。YOLO 阈值暂定为 `0.07`；八个 PatchCore 视角分别使用各自 v11
summary 的 deploy threshold，且 low/high 相同；Template 使用 v11 模型内 normal-only 阈值。

旧 Stage33 阈值使用日期式 ROI version，新 v11 使用内容 SHA 派生的 ROI version。两份 ROI 文件的
SHA256 均为 `9412b2838cdb96f722db356714cf9bbbb5ea01810671ccf92b67323de77ebe65`。Stage34 增加了显式
ROI version rebind 门：只有提供旧 ROI 文件且 source/runtime 的 PatchCore、YOLO ROI 字节 SHA
全部一致时才允许发布。旧阈值 sidecar 还必须签名绑定 Stage33 run contract，run contract 再绑定旧
runtime config；旧 runtime 中的两份 ROI 也必须与显式 source ROI 字节相同。完整授权链与 SHA 均写入签名阈值产物。

当前边界：

- 仍是 `commissioning_only=true`、`production_release_allowed=false`。
- YOLO 阈值源保留 test leakage 与 model rebind 警告；YOLO 当前验证 recall 偏低，现场必须重点检查漏检。
- 两个 secondary PatchCore 只有正常数据，不能据此宣称缺陷 recall/AUROC。
- Template 记录使用 v11 模型自身的 calibration count；模型重绑定后的 PatchCore/YOLO 不继承旧 Stage33
  count，记录为 `0/0` 并标记 `count_provenance=unavailable_for_rebound_model`。
- 没有 GT mask，pixel AUROC、AUPRO、pixel F1、pixel IoU 均为 N/A。
- v10 回滚 bundle 保留为
  `results/zs32_runtime_bundle_eight_view_20group_patchcore_tuned_v10/runtime_bundle.json`。
