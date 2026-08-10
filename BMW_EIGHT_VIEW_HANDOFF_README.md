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

| 分支 | 当前复查结果 | 交接结论 |
|---|---|---|
| Template | 八视图final-test平衡准确率0.925–1.000；没有缺陷误放行 | 当前最稳定，但部分正常件会误拒 |
| 光痕 | final-test平衡准确率0.8824；4个正常误拒、0个无光痕误放行 | 保守可用；仍缺少真实“有亮痕但断续”样本 |
| YOLO | Precision 0.557、Recall 0.279、mAP50 0.219、mAP50-95 0.091 | 能演示画框，但召回率偏低，是最需要补数据的分支 |
| EfficientAD | 独立八视图AUROC约0.7375–1.000 | `back_right`、`back_secondary`优先补数据和重新标定 |

这些指标只表示当前数据上的实验室效果，不能作为甲方量产验收指标。

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
  pipeline/bmw_lab_eight_view_demo.py
do
  uv run --no-sync python "$script" --help >/dev/null || exit 1
done
```

不要用全仓库无关测试的失败代替BMW聚焦验证结论，也不要用CLI能启动代替真实GPU或实体相机验证。
