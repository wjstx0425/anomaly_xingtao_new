# BMW 四相机八视图实验室检测系统交接 README

本文档是当前 BMW 零件视觉检测工作的主交接入口：四相机 HDR 采集 → 八视图数据整理 → ROI → YOLO 标注 → 训练目录 → Template / 光痕 / EfficientAD / YOLO 训练 → 模型检查 → 离线与实时 Demo。

这是用于快速验证和迭代的实验室版本，不是已经完成节拍、PLC、MES、权限、冗余和长期稳定性验证的工业发布版本。

## 1. 当前状态和发布边界

- 代码分支：`agent/bmw-eight-view-handoff`
- 当前训练运行：`results/bmw_lab_one_click/bmw_lab_eight_view_v1`
- 当前八视图 Demo：`pipeline/bmw_lab_eight_view_demo.py`
- 当前 ROI：`configs/bmw/rois/bmw_hdr_eight_view_v1.json`
- 四类检测：Template、光痕规则、YOLO26n、EfficientAD-S
- 每个样本：正面4张、翻面后反面4张，共8张HDR融合图
- 每次Demo执行25项：8 Template + 1 光痕 + 8 YOLO + 8 EfficientAD

融合规则：任一模型错误为 `ERROR`；否则任一检查为 `NG` 时整件为 `NG`；全部通过才是 `OK`。不会因前面已出现NG而跳过后续模型。

GitHub只包含代码、配置、测试和本文档，不包含 `dataset/`、`results/`、模型权重、Label Studio数据库、账号或海康MVS SDK。数据和模型必须从原工作站线下复制，禁止把客户图片和账号提交到公开仓库。

## 2. 流程和代码地图

```text
bmw_lab_collect_data.py -> dataset/bmw_lab_raw
  -> bmw_lab_prepare_eight_view_data.py -> dataset/bmw_lab_prepared
  -> bmw_lab_select_eight_view_rois.py -> configs/bmw/rois
  -> bmw_lab_materialize_training_data.py -> dataset/bmw_lab_training
  -> bmw_lab_prepare_labeling_package.py -> Label Studio标注/复核
  -> bmw_lab_train_all.py -> results/bmw_lab_one_click
  -> bmw_lab_eight_view_demo.py -> 离线/四相机实时测试
```

| 路径 | 职责 |
|---|---|
| `src/bmw_inspection/capture/config.py` | 四相机HDR配置校验 |
| `src/bmw_inspection/lab/eight_view_dataset.py` | 完整八视图检查、物理件分组和切分 |
| `src/bmw_inspection/lab/eight_view_roi.py` | ROI选择和清单绑定 |
| `src/bmw_inspection/lab/eight_view_training_data.py` | 生成三个训练分支目录 |
| `src/bmw_inspection/lab/labeling_package.py` | Label Studio标注包和正常参考图 |
| `src/bmw_inspection/lab/eight_view_train_all.py` | 一键训练编排 |
| `src/bmw_inspection/lab/eight_view_demo_models.py` | 四类模型推理适配器 |
| `src/bmw_inspection/lab/eight_view_demo_capture.py` | Demo四相机两轮HDR采集 |
| `src/bmw_inspection/lab/eight_view_demo_ui.py` | 全中文1600×900界面 |

## 3. 固定硬件、视角与HDR

工装、零件、相机和光源必须固定：

| 位置 | 相机序列号 | 正面 | 反面 |
|---|---|---|---|
| 中间 | `DA9805574` | `front` | `back` |
| 左侧 | `DA9625347` | `front_left` | `back_left` |
| 右侧 | `DB0998274` | `front_right` | `back_right` |
| 辅助 | `DB0968108` | `front_secondary` | `back_secondary` |

标准顺序：`front, front_left, front_right, front_secondary, back, back_left, back_right, back_secondary`。

`configs/bmw/capture/bmw_4cam_eight_view_hdr_v1.json` 固定短曝光1500 μs、长曝光6000 μs、增益0、分组触发间隔0.2 s、稳定帧1、超时3000 ms、关闭HDR对齐。不要临时改回单曝光4000 μs，否则训练与部署的图像形成过程不一致。

## 4. 新电脑准备

```bash
git clone -b agent/bmw-eight-view-handoff \
  git@github.com:wjstx0425/anomaly_xingtao_new.git
cd anomaly_xingtao_new
uv sync
uv pip install ultralytics==8.4.89
```

`ultralytics` 是 BMW 实验室 YOLO 分支的额外运行依赖，当前代码按训练时版本固定为
`8.4.89`；不要直接升级后复用旧指标。其余依赖由仓库的 `uv.lock` 管理。

项目统一使用：

```bash
uv run --no-sync python <脚本> <参数>
```

Linux采集前确认：

```bash
test -f /opt/MVS/Samples/64/Python/MvImport/MvCameraControl_class.py
```

训练还需要：

```text
yolo26n.pt
~/.cache/anomalib/pre_trained/efficientad_pretrained_weights/pretrained_teacher_small.pth
ImageNette目录，例如 /path/to/imagenette/imagenette2
```

新电脑训练时必须显式传 `--imagenette-dir`，不要沿用原工作站的绝对路径。

## 5. 采集八视图HDR数据

先检查四台相机：

```bash
uv run --no-sync python pipeline/bmw_lab_collect_data.py --list-devices
```

正常件冒烟：

```bash
uv run --no-sync python pipeline/bmw_lab_collect_data.py \
  --label normal \
  --part-id bmw_normal_smoke \
  --group-count 1 \
  --root dataset/bmw_lab_raw
```

程序先提示固定正面并采集四张HDR融合图，然后提示翻转同一零件并采集反面四张。确认冒烟完整后再提高 `--group-count`。

无光痕件：

```bash
uv run --no-sync python pipeline/bmw_lab_collect_data.py \
  --label defect \
  --defect-type no_streak \
  --part-id bmw_no_streak \
  --group-count 1 \
  --root dataset/bmw_lab_raw
```

`no_streak`只对光痕分支是NG；对EfficientAD和Template是正常外观；YOLO不能把缺失亮痕框成 `defect`。

其他缺陷示例：

```bash
uv run --no-sync python pipeline/bmw_lab_collect_data.py \
  --label defect \
  --defect-type scratch \
  --part-id bmw_scratch \
  --group-count 1 \
  --root dataset/bmw_lab_raw
```

原始图片位于 `dataset/bmw_lab_raw/left/<view>/<label>/...`，会话清单位于 `dataset/bmw_lab_raw/manifests/`。`left`是旧目录兼容字段，不表示只使用左侧相机。

## 6. 准备标准八视图数据

先预检，再正式发布（正式执行时删掉 `--dry-run`）：

```bash
uv run --no-sync python pipeline/bmw_lab_prepare_eight_view_data.py \
  --raw-root dataset/bmw_lab_raw \
  --output-root dataset/bmw_lab_prepared \
  --dataset-id bmw_hdr_eight_view_v1 \
  --hand left \
  --dry-run
```

输出 `dataset/bmw_lab_prepared/bmw_hdr_eight_view_v1/`，其中有 `dataset_manifest.csv`、各分支清单、物理件切分和 `report.json`。

当前发布记录：132个完整物理件、1056张图、每视角132张、4024×3036；normal 85件、no_streak 16件、deform 10件、edge 17件、others 4件。

注意：prepared清单中的 `source_path` 是绝对路径。换电脑或目录后，应从复制过去的raw重新运行本阶段，不要直接使用旧路径。

## 7. 选择八视图ROI

```bash
uv run --no-sync python pipeline/bmw_lab_select_eight_view_rois.py \
  --prepared-root dataset/bmw_lab_prepared/bmw_hdr_eight_view_v1 \
  --output configs/bmw/rois/bmw_hdr_eight_view_v1.json
```

每个视角框住固定零件区域，尽量排除工装和背景。ROI与prepared清单哈希绑定；prepared重建后应重新检查ROI。覆盖已有配置必须显式加 `--force`。

## 8. 生成待标注训练目录

```bash
uv run --no-sync python pipeline/bmw_lab_materialize_training_data.py \
  --prepared-root dataset/bmw_lab_prepared/bmw_hdr_eight_view_v1 \
  --roi-config configs/bmw/rois/bmw_hdr_eight_view_v1.json \
  --output-root dataset/bmw_lab_training \
  --training-id bmw_hdr_roi_training_v1
```

该步骤生成统一PNG裁剪和三个分支目录。没有完整人工标签时，YOLO只生成 `annotation_queue.csv`，不会发布可训练的 `data.yaml`。

## 9. YOLO标注与复核

生成带每视角8张正常参考图的便携标注包：

```bash
uv run --no-sync python pipeline/bmw_lab_prepare_labeling_package.py \
  --training-root dataset/bmw_lab_training/bmw_hdr_roi_training_v1 \
  --output-root dataset/bmw_lab_labeling \
  --package-id bmw_hdr_roi_yolo_handoff \
  --normal-references-per-view 8
```

启动Label Studio：

```bash
conda activate label-studio
export DEBUG=false
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$(pwd)/dataset/bmw_lab_labeling/bmw_hdr_roi_yolo_handoff"
label-studio start --data-dir "$HOME/.local/share/label-studio" --port 8080 --no-browser
```

浏览器打开 `http://127.0.0.1:8080`，新建项目，使用包内 `label_studio/label_config.xml`，导入 `label_studio/tasks.json`。每张图选择“有可见缺陷”或“无可见缺陷”；只框当前视图实际可见的 `defect`，不框正常孔洞、边缘、固定高光、亮痕或背景。

一个零件整体NG不等于八个视角都要有框。确认无可见缺陷的视图必须保留空txt，不能当作未标注。最终标签目录必须满足：

- 与 `annotation_queue.csv` 一一对应
- 类别只有0号 `defect`
- 无缺陷为存在但内容为空的txt
- 不含 `classes.txt`
- 不存在“无可见缺陷但仍有框”的矛盾

当前本机复核结果为248个任务/标签，131张有框、117张空标签、170个框。该标签目录不在GitHub，需要线下复制。

## 10. 生成最终训练目录

```bash
uv run --no-sync python pipeline/bmw_lab_materialize_training_data.py \
  --prepared-root dataset/bmw_lab_prepared/bmw_hdr_eight_view_v1 \
  --roi-config configs/bmw/rois/bmw_hdr_eight_view_v1.json \
  --output-root dataset/bmw_lab_training \
  --training-id bmw_hdr_roi_training_reviewed_v1 \
  --yolo-label-root /path/to/reviewed_yolo_labels
```

检查 `dataset/bmw_lab_training/bmw_hdr_roi_training_reviewed_v1/report.json`：当前应为 `crop_count=1056`、`yolo_label_count=1056`、`yolo_pending_count=0`、`yolo_training_ready=true`。

## 11. 一键训练

先用同一命令加 `--dry-run` 做预检，确认后删除该参数正式训练：

```bash
uv run --no-sync python pipeline/bmw_lab_train_all.py \
  --prepared-root dataset/bmw_lab_prepared/bmw_hdr_eight_view_v1 \
  --roi-config configs/bmw/rois/bmw_hdr_eight_view_v1.json \
  --reviewed-yolo-labels /path/to/reviewed_yolo_labels \
  --training-root dataset/bmw_lab_training \
  --training-id bmw_hdr_roi_training_reviewed_v1 \
  --output-root results/bmw_lab_one_click \
  --run-id bmw_lab_eight_view_v1 \
  --yolo-checkpoint yolo26n.pt \
  --imagenette-dir /path/to/imagenette/imagenette2 \
  --yolo-batch 32 \
  --dry-run
```

训练顺序：最终数据目录 → 八视图Template → `front_left`光痕标定 → 八个EfficientAD-S → 一个全视角YOLO26n。

当前默认：Template每视角5张模板、512×512、最大偏移12；EfficientAD-S为256×256、30 epochs、batch固定1；YOLO26n为640、100 epochs、batch 32、GPU 0；seed 42。EfficientAD batch 1是框架实现约束，不是显存不足。重新训练应使用新 `--run-id`，不要覆盖旧运行。

### 11.1 只重新标定亮痕规则，不启动训练

更新亮痕 detector 后，不需要重跑 Template、EfficientAD 或 YOLO。使用独立命令只读取 `bright_streak.csv` 和现有亮痕配置：

```bash
uv run --no-sync python pipeline/bmw_lab_recalibrate_bright_streak.py
```

默认输入是：

- `dataset/bmw_lab_prepared/bmw_hdr_eight_view_v1/manifests/bright_streak.csv`
- `results/bmw_lab_one_click/bmw_lab_eight_view_v1/bright_streak/calibrated_config.json`

默认输出到 `results/bmw_lab_one_click/bmw_lab_eight_view_v1/bright_streak_ridge_v2/`。命令只使用 `split=calibration` 拟合阈值，`split=final_test` 始终只用于最终报告；`sample_id` 只作为追溯元数据，不参与判定。输出包含 `calibrated_config.json`、逐图 `metrics.csv`、`report.json`，以及每个 final-test 错误对应的 `final_test_errors/*.png` 证据图。需要改路径时使用 `--manifest`、`--base-config` 和 `--output-dir`；输出目录必须是新目录，避免覆盖旧标定结果。

实验室演示若需要显式放宽断续性，可另选新输出目录并启用：

```bash
uv run --no-sync python pipeline/bmw_lab_recalibrate_bright_streak.py \
  --bold-continuity \
  --output-dir results/bmw_lab_one_click/bmw_lab_eight_view_v1/bright_streak_ridge_v2_bold
```

`--bold-continuity` 是 **demo-only** 模式：`min_contrast_snr` 与 `min_coverage_ratio` 仍只由 calibration 拟合，但三个 continuity 阈值会使用 calibration normal 与 final-test normal 的包络，因此 final test 不再是完全隔离的无泄漏评估集。报告会分别记录 presence 与 continuity 的拟合 split。该开关默认关闭，不应用于正式验收指标。

## 12. 模型目录与当前效果

```text
results/bmw_lab_one_click/bmw_lab_eight_view_v1/
├── run_report.json
├── template/<view>/model.json
├── bright_streak/calibrated_config.json
├── efficientad/<view>/model.ckpt
└── yolo/train/weights/best.pt
```

`run_report.json`的总状态和 materialize、template、bright_streak、efficientad、yolo 五阶段都必须是 `complete`。

### 12.1 先理解这些指标

- `balanced accuracy` 是正常件正确率与缺陷件正确率的平均值。当前缺陷样本少，用它比普通准确率更合理。
- `false accept` 是缺陷件被判成OK，属于漏检；`false reject` 是正常件被判成NG，属于过检。
- `AUROC` 衡量异常分数能否把正常与异常排序分开，不代表当前固定阈值已经合适。
- `F1` 同时考虑精确率和召回率，并受当前判定阈值直接影响。AUROC较高但F1较低通常说明分数有区分度，但阈值或样本分布仍需重新标定。
- YOLO的 `Precision` 表示模型画出的框中有多少与标注匹配，`Recall` 表示标注缺陷中有多少被模型找出；`mAP50-95` 对框位置要求比 `mAP50` 更严格。

所有表格中的final-test都没有参与阈值拟合，但样本量仍然很小。因此这些数字只能用于当前实验室数据上的模型比较，不能直接作为甲方量产验收指标。

### 12.2 Template：当前最稳定的结构基线

Template不是判断具体缺陷类别，而是把ROI与5张正常模板做归一化相关匹配。当前风险值为 `1 - 最大相似度`；风险超过各视角自己的阈值就判NG。它适合固定工装、固定姿态和固定光照下检查轮廓、孔位、边缘及较明显的整体外观变化。

| 视角 | 风险阈值 | final-test正常/缺陷 | 正常误拒 | 缺陷误放行 | 平衡准确率 |
|---|---:|---:|---:|---:|---:|
| `front` | 0.005995 | 20 / 5 | 2 | 0 | 0.950 |
| `front_left` | 0.010074 | 20 / 4 | 0 | 0 | 1.000 |
| `front_right` | 0.006055 | 20 / 4 | 2 | 0 | 0.950 |
| `front_secondary` | 0.002983 | 20 / 6 | 3 | 0 | 0.925 |
| `back` | 0.005478 | 20 / 4 | 2 | 0 | 0.950 |
| `back_left` | 0.009188 | 20 / 1 | 2 | 0 | 0.950 |
| `back_right` | 0.003331 | 20 / 4 | 3 | 0 | 0.925 |
| `back_secondary` | 0.004705 | 20 / 3 | 3 | 0 | 0.925 |

当前final-test里没有出现缺陷误放行，说明这组阈值偏保守；代价是多个视角会把2–3张正常图判成NG。`front_left`看起来达到1.000，但测试集只有4张可见缺陷，不能据此认为它已经解决所有缺陷。`back_left`更只有1张缺陷图，统计可信度尤其有限。

Template对位置和成像变化敏感。后续如果出现相机轻微移动、零件批次纹理变化或光源衰减，正常误拒通常会先增加。新采一批正常件后，应先查看风险分数分布，再决定是扩充正常模板还是调整阈值，不能只把阈值整体放宽。

### 12.3 光痕规则：无光痕检测可靠，断续验证尚不完整

光痕只检查 `front_left`。阈值由calibration中的17张正常有光痕图和3张无光痕图拟合；final-test也包含17张正常有光痕图和3张无光痕图。

| 证据 | 当前阈值 | 含义 |
|---|---:|---|
| 对比度信噪比 | `>= 3.2895` | 光痕必须明显高于局部背景 |
| 覆盖率 | `>= 0.1900` | 有效光痕必须覆盖足够长度 |
| 最长连续段比例 | `>= 0.3312` | 至少存在一段足够长的连续亮区 |
| 最大断口比例 | `<= 0.07015` | 单个断口不能过长 |
| 断口数量 | `<= 2` | 不能出现过多断续段 |

calibration平衡准确率为1.000，0正常误拒、0无光痕误放行。独立final-test平衡准确率为0.8824：3张无光痕图全部被拒绝，没有漏检；17张正常有光痕图中有4张被误判NG。因此当前规则的特点是宁可过检，也不放过完全无光痕件。

当前最重要的限制是：数据中只有“连续正常光痕”和“完全无光痕”，没有足够的“光痕存在但中间断续”实物样本。因此最长连续段、断口比例和断口数量阈值只是按正常光痕的严格包络确定，尚未证明能稳定区分各种真实断续形态。演示时可以显示这些连续性证据，但不能宣称断续缺陷已经完成独立验证。

### 12.4 EfficientAD：部分视角有排序能力，固定阈值仍不稳定

每个视角各训练一个EfficientAD-S，仅用该视角的正常外观学习分布；`no_streak`在这个分支中按正常件处理，因为它没有其他外观缺陷。模型输入为256×256，训练30 epochs，框架训练batch固定为1。

原训练报告中的AUROC可以由逐图运行时分数完整复现；但重新执行 `engine.test` 时，TorchMetrics明确警告F1在更新状态前被计算，因此原 `image_F1Score` 不应继续用于判断部署阈值。下表的运行时F1、平衡准确率、正常误拒和缺陷漏检均由Demo同路径的逐图 `predict` 输出重新计算。

| 视角 | AUROC | 运行时F1 | 平衡准确率 | 正常误拒 | 缺陷漏检 |
|---|---:|---:|---:|---:|---:|
| `front` | 0.9625 | 0.8000 | 0.9500 | 2 / 20 | 0 / 4 |
| `front_left` | 0.8250 | 0.8000 | 0.8333 | 0 / 20 | 1 / 3 |
| `front_right` | 0.8333 | 0.8000 | 0.8333 | 0 / 20 | 1 / 3 |
| `front_secondary` | 1.0000 | 0.8000 | 0.8333 | 0 / 20 | 1 / 3 |
| `back` | 0.9125 | 0.5714 | 0.9250 | 3 / 20 | 0 / 2 |
| `back_left` | 0.9750 | 0.8000 | 0.9750 | 1 / 20 | 0 / 2 |
| `back_right` | 0.7375 | 0.6667 | 0.7500 | 0 / 20 | 1 / 2 |
| `back_secondary` | 0.8875 | 0.6667 | 0.8250 | 2 / 20 | 1 / 4 |

这里不能只看AUROC。例如 `back_right` 的AUROC为0.7375，表示异常分数并非完全随机，但2张缺陷只检出1张。`front_secondary` 的AUROC为1.0也只代表当前3张缺陷与20张正常的分数排序完全分开；其中仍有1张缺陷落在运行判定阈值下，不能理解为量产100%。

逐图分数和热力图可重新导出：

```bash
uv run --no-sync python pipeline/bmw_lab_visualize_efficientad_scores.py
```

默认输出到 `results/bmw_lab_one_click/bmw_lab_eight_view_v1/efficientad/score_analysis/`：

- `efficientad_scores.csv`：183张测试图的视角、真实标签、运行判定、异常分数、阈值和图片路径。
- `efficientad_score_distributions.png`：八视角正常/缺陷散点分布及0.5阈值线。
- `examples/<view>.png`：每视角高分正常和低分缺陷样本的热力图。
- `report.json`：从逐图运行时输出重新计算的AUROC、F1、平衡准确率、误拒和漏检。

分数是Anomalib后处理后的0–1归一化分数，同一视角内比较最有意义；不同视角由不同模型独立标定，不能把两个视角相同的0.6理解为缺陷严重度完全相同。热力图固定使用0–1色标，不再对每张图单独拉伸颜色，因此正常图和缺陷图可以直观对比。

当前热力图复查还显示：部分 `front` 正常误拒主要高亮零件外侧亮斑、孔位或ROI边缘，而不是稳定的缺陷区域；`back_right` 有一张真实缺陷的分数接近0且几乎没有有效热点。这说明当前EfficientAD既受背景/反光正常波动影响，也会漏掉训练中覆盖不足的局部缺陷。后续应先收紧或遮罩无关背景，并增加对应视角的正常反光变化和漏检缺陷，不应只靠降低0.5阈值。

因此当前EfficientAD适合在Demo中提供第二种异常证据，不适合作为单独放行依据。优先工作应是补充 `back_right`、`back_secondary`、`front_left` 的正常波动和真实缺陷，然后只用calibration重新选阈值，再在未参与调参的物理件上报告F1、误放行和误拒。

### 12.5 YOLO26n：可以画框，但漏检是当前主要问题

YOLO训练集共632张图，其中77张有框、99个缺陷框；验证集208张图，其中23张有框、28个框；独立测试集216张图，其中31张有框、43个框。其余图片都是经过人工确认的负样本。类别只有一个：`defect`。

| 指标 | 独立测试结果 | 实际含义 |
|---|---:|---|
| Precision | 0.557 | 预测出的缺陷框约一半能与人工框正确匹配，仍存在较多误框 |
| Recall | 0.279 | 只能找到约27.9%的标注缺陷，约72.1%的缺陷框没有被检出 |
| mAP50 | 0.219 | 在IoU 0.5标准下的综合检测能力偏低 |
| mAP50-95 | 0.091 | 框的位置和尺度精度较弱，严格定位能力不足 |

这说明当前模型可以用于演示“检测后画框”和验证整条推理链，但不能承担可靠的缺陷兜底。主要原因不是batch设置，而是正样本和框数量少、不同视角外观差异大、所有缺陷都合并成一个 `defect` 类，并且部分缺陷只在少数视角可见。Demo使用候选置信度0.1保留证据、最终NG阈值0.25；继续降低最终阈值会增加召回，但也会明显增加误框，不能代替补数据。

下一轮应优先增加当前漏检样本及小缺陷样本，保持物理件级train/val/test隔离，并分别统计八个视角的Precision与Recall。若某些缺陷只在固定视角出现，可以考虑按视角训练或至少按视角设置阈值，而不是只看一个全局mAP。

### 12.6 四分支融合后的实际含义

Demo对一个零件执行8个Template、1个光痕、8个YOLO和8个EfficientAD，共25项检查。只要任一项为NG，整件就是NG。因此单模型“没有缺陷误放行”不等于整套系统良率高：多个偏保守分支叠加后，任一视角的正常误拒都会把整件拦下。

当前正常样本 `bmw_normal_group072_000001` 的25项全部PASS，只能证明模型、配置、GPU推理和界面融合链路可以跑通，不能代表正常件误拒率。下一步真正有价值的系统指标应以物理件为单位，在全新批次上统计：整件正常放行率、整件缺陷拦截率、各分支触发次数、重复拍摄一致性和单件耗时分位数。

## 13. 离线部署测试

Demo默认寻找当前训练运行、prepared清单和ROI配置。

prepared里的绝对源图路径仍有效时：

```bash
uv run --no-sync python pipeline/bmw_lab_eight_view_demo.py \
  --sample-id bmw_normal_group072_000001 \
  --experiment-mode
```

换电脑后推荐使用便携目录，目录中必须恰好有八个标准视角同名图片：

```text
/path/to/capture_set/
├── front.png
├── front_left.png
├── front_right.png
├── front_secondary.png
├── back.png
├── back_left.png
├── back_right.png
└── back_secondary.png
```

```bash
uv run --no-sync python pipeline/bmw_lab_eight_view_demo.py \
  --capture-set /path/to/capture_set \
  --experiment-mode
```

无界面冒烟并保存截图：

```bash
uv run --no-sync python pipeline/bmw_lab_eight_view_demo.py \
  --capture-set /path/to/capture_set \
  --no-gui \
  --experiment-mode \
  --save-screenshot results/bmw_eight_view_demo/smoke.png
```

当前RTX 4090真实样本 `bmw_normal_group072_000001` 为25 PASS、最终OK，总推理约3.4秒。

## 14. 四相机实时Demo

连接四台相机、固定工装并确认模型完整后：

```bash
uv run --no-sync python pipeline/bmw_lab_eight_view_demo.py
```

操作：

- 空格：拍正面；翻面后再次按空格拍反面并检测
- `R`：清空并重置
- `Q`或`Esc`：退出
- 实验模式下 `1–8`：切换视角
- 实验模式下 `T/L/Y/E`：切换Template、光痕、YOLO、EfficientAD证据

启动时必须是空白等待态。光痕证据顺时针旋转90°只影响显示，不改变检测坐标。

交接发布前已验证离线真实模型和界面；实时四相机入口已经接好，但本轮没有再次触发实体相机。接班后的第一项硬件任务应是完成一组正常件正反面实时冒烟。

## 15. 数据和模型线下交接

只运行Demo至少复制：

```text
results/bmw_lab_one_click/bmw_lab_eight_view_v1/
configs/bmw/rois/bmw_hdr_eight_view_v1.json
一个便携八视图capture_set（离线测试时）
```

继续训练还需复制：

```text
dataset/bmw_lab_raw/
最终reviewed_yolo_labels/
yolo26n.pt
EfficientAD teacher权重
ImageNette目录
```

新电脑应从raw重建prepared、ROI和training release，消除绝对路径问题。复制后核对目录数量、关键 `report.json` 和模型SHA256，不要只比较文件夹名。

## 16. 常见问题

### MVS SDK不可用

若提示 `MvCameraControl_class is unavailable`，检查 `/opt/MVS/Samples/64/Python/MvImport/MvCameraControl_class.py`；MVS图形工具能打开不代表Python模块可用。

### 少一台相机或序列号交换

运行 `--list-devices` 并对照第3节，不要按枚举顺序猜视角。

### prepared源图不存在

清单保存了旧机器绝对路径，从raw重新运行数据准备器。

### ROI哈希不匹配

重新检查/选择ROI，不要手工删除哈希绕过约束。

### YOLO没有data.yaml

说明待复核视图的txt不完整，或存在缺失、多余、非法标签。

### EfficientAD预检失败

检查teacher权重和ImageNette路径，用 `--imagenette-dir` 覆盖默认值。

### 正常件被判NG

在实验模式定位具体视角和分支。Template、光痕和EfficientAD都有正常误拒，不能不看证据就整体放宽所有阈值。

## 17. 接班后的优先工作

1. 完成真实四相机正常件实时采集和Demo冒烟。
2. 补充“有亮痕但断续”的真实样本，单独校准连续性规则。
3. 增加YOLO可见缺陷和难负样本，解决0.279召回率。
4. 补充 `back_right`、`back_secondary` 正常变化与缺陷样本，重新评估EfficientAD。
5. 建立按物理件隔离的固定回归集，不能用训练图片证明部署效果。
6. 每次采集保持相机、曝光、光源、工装和HDR参数一致。

## 18. 接班验收清单

- [ ] 能克隆交接分支并完成 `uv sync` 和 `uv pip install ultralytics==8.4.89`
- [ ] 能解释四相机和八视图映射
- [ ] 能运行 `--list-devices`
- [ ] 能采集一个完整八视图HDR样本
- [ ] 能从raw生成prepared release
- [ ] 能选择八视图ROI
- [ ] 能生成Label Studio包并解释空txt
- [ ] 能执行一键训练 `--dry-run`
- [ ] 能找到四类模型和 `run_report.json`
- [ ] 能用 `--capture-set --no-gui` 跑离线模型
- [ ] 能启动全中文实时Demo
- [ ] 明确当前YOLO、EfficientAD和连续性样本短板
- [ ] 明确GitHub不包含数据、权重、SDK和账号

## 19. 聚焦代码验证

不需要图片或权重的八视图聚焦测试：

```bash
uv run --no-sync pytest -q \
  tests/unit/bmw_inspection/capture/test_config.py \
  tests/unit/bmw_inspection/capture/test_cli.py \
  tests/unit/bmw_inspection/lab/test_eight_view_dataset.py \
  tests/unit/bmw_inspection/lab/test_eight_view_roi.py \
  tests/unit/bmw_inspection/lab/test_eight_view_training_data.py \
  tests/unit/bmw_inspection/lab/test_labeling_package.py \
  tests/unit/bmw_inspection/lab/test_eight_view_train_all.py \
  tests/unit/bmw_inspection/lab/test_eight_view_demo.py \
  tests/unit/bmw_inspection/lab/test_eight_view_demo_ui.py
```

`test_eight_view_demo_models.py` 中的
`test_generic_template_predictor_loads_secondary_view_model` 和
`test_bright_streak_predictor_uses_trained_config` 会读取本地训练产物与真实图片；GitHub克隆后没有线下资产时，应先运行其余5个纯运行时用例，复制模型和数据后再跑完整文件。

检查全部交接入口：

```bash
for script in \
  pipeline/bmw_lab_collect_data.py \
  pipeline/bmw_lab_prepare_eight_view_data.py \
  pipeline/bmw_lab_select_eight_view_rois.py \
  pipeline/bmw_lab_prepare_labeling_package.py \
  pipeline/bmw_lab_materialize_training_data.py \
  pipeline/bmw_lab_train_all.py \
  pipeline/bmw_lab_visualize_efficientad_scores.py \
  pipeline/bmw_lab_eight_view_demo.py
do
  uv run --no-sync python "$script" --help >/dev/null || exit 1
done
```

不要用全仓库无关测试的失败代替BMW聚焦验证结论，也不要用CLI能启动代替真实GPU或实体相机验证。
