# ZS32 四相机采集到离线测试完整操作手册（方案 1）

本文用于在 `/home/yunjing/anomaly_xingtao_new` 中从零完成一次新的 ZS32 数据闭环：

```text
四相机采集
  -> 原始数据验收
  -> Label Studio 标注
  -> YOLO 数据集和 ROI
  -> PatchCore 数据集和 ROI
  -> PatchCore 训练
  -> YOLO 训练
  -> 模板匹配训练
  -> 运行配置绑定
  -> Stage 33 离线联调和阈值诊断
```

## 0. 当前能力边界（开始前必须阅读）

四相机采集会为每个物理零件保存 8 张图：

```text
front, front_left, front_right, front_secondary,
back,  back_left,  back_right,  back_secondary
```

本文后续第 1～11 节保留的是历史六视图重建流程。当前现场检测使用下一小节的右手八视图
Demo 链；不要从后续历史章节复制 Stage 18、runtime bundle 或旧 Stage 32 启动参数。

### 0.1 八视图 Demo 唯一在线入口（2026-07-27）

唯一操作员配置：

```text
configs/zs32/zs32_demo.json
```

它直接声明八个 PatchCore checkpoint、共享 YOLO、八视图 Template、ROI、topology 和三类阈值。
该 Demo 不读取 runtime bundle、threshold artifact、fusion profile 或 SHA 发布信息，也不调用
Stage 18/34/37。

现场命令：

```bash
cd /home/yunjing/anomaly_xingtao_new

UV_CACHE_DIR=/tmp/uv-cache \
PYTHONPATH=/opt/MVS/Samples/64/Python/MvImport:. \
uv run --no-sync python \
  pipeline/36_zs32_inspection_dashboard.py \
  --live \
  --demo-config configs/zs32/zs32_demo.json \
  --part-id zs32_demo_001
```

Dashboard 会先常驻加载八个 PatchCore、一个 YOLO 和八视图 Template，随后等待正面/背面两次人工确认。
每件开始前重新读取 Demo JSON：只改阈值会在下一件生效且不重载模型；修改模型路径、ROI、topology 或
推理设置会明确要求重启 Dashboard。八个视角先全部运行 Template；任一 Template NG 时整件立即输出
`NG_TEMPLATE`，PatchCore/YOLO 均明确为 `SKIPPED`；Fusion 逐视角显示对应 Template 的
`NG_TEMPLATE` 或 `PASS`。只有八个 Template 全部 PASS，才继续运行
八个 PatchCore 和八个 YOLO。缺图、缺证据、配置错误、相机错误或模型异常都必须显示真实错误并输出
`ERROR`，绝不能判为 OK。

当前实现和阈值仅用于 `DEMO / 非生产`。本轮已用保存的八视角图片完成离线 smoke，但尚未等待摆件并完成
真实四相机的两轮采集，因此 hardware capture 验收仍未完成。旧 bundle/Stage 18 结果目录和脚本仅为历史
回放保留，不是在线入口，也未在本次改造中物理删除。

历史六视图流程只消费：

```text
front, front_left, front_right, back, back_left, back_right
```

因此，本手册现在能够真正跑通的是：

- 用四台相机采集并永久保留 8 个视图；
- 使用原来的 6 个主视图完成处理、标注、PatchCore、YOLO、模板和离线联调；
- `front_secondary/back_secondary` 先作为原始数据归档，不进入本轮模型；
- 最终 Stage 33/34 是右手六视图 `commissioning_only`，不能解释为生产放行。

这不是命令参数问题。下游源码的视图集合目前就是六个。要让第四台相机进入模型，必须另外扩展 Stage 27～33、
准备 8-view ROI/标签/模型/阈值和新的融合 profile。本文末尾列出了具体缺项。

## 1. 统一工作目录和环境

所有命令默认从新仓库执行，数据也放在新仓库的 `dataset/` 下。Stage 27/28 会记录相对仓库路径，不能直接把
`/home/yunjing/anomalib/dataset/test` 当作这次新流程的正式输入。

```bash
cd /home/yunjing/anomaly_xingtao_new

export REPO=/home/yunjing/anomaly_xingtao_new
export DATASET_ROOT="$REPO/dataset"
export ACCEPT_ROOT="$REPO/dataset_acceptance"
export LABEL_ROOT="$DATASET_ROOT/zs32_yolo_labeling"
export YOLO_FULL="$DATASET_ROOT/zs32_six_view_yolo"
export YOLO_ROI="$DATASET_ROOT/zs32_six_view_roi_yolo"
export PATCHCORE_ROI="$DATASET_ROOT/zs32_patchcore_roi"
export RUN_ROOT="$REPO/results/zs32_rebuild_v1"
export PATCHCORE_RUN="$RUN_ROOT/patchcore_six_view"
export YOLO_RUN="$RUN_ROOT/yolo/zs32_roi_n640_seed42"
export TEMPLATE_RUN="$RUN_ROOT/template_right"
export RUNTIME_CONFIG="$RUN_ROOT/runtime_models.local.json"
export OFFLINE_RUN="$RUN_ROOT/offline_calibration"
export ULTRA_ROOT=/home/yunjing/ultralytics-c789

mkdir -p "$DATASET_ROOT" "$ACCEPT_ROOT" "$RUN_ROOT"
```

检查 Python、GPU、相机和 CLI：

```bash
.venv/bin/python -V
.venv/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'

PYTHONPATH=/opt/MVS/Samples/64/Python/MvImport:${PYTHONPATH:-} \
  .venv/bin/python pipeline/1_collect_multicamera_data.py --list-devices

.venv/bin/python pipeline/zs32_bootstrap_capture.py --help
```

相机枚举必须包含：

```text
DA9805574
DA9625347
DB0998274
DB0968108
```

## 2. 四相机采集

### 2.1 先做一组验收

```bash
export CAPTURE_SESSION="zs32_4cam_accept_$(date +%Y%m%d_%H%M%S_%N)"

PYTHONPATH=/opt/MVS/Samples/64/Python/MvImport:${PYTHONPATH:-} \
.venv/bin/python pipeline/zs32_bootstrap_capture.py \
  --topology configs/zs32/topology/zs32_4cam_double_side_v1.json \
  --capture-session "$CAPTURE_SESSION" \
  --legacy-layout \
  --hand right \
  --label normal \
  --part-id zs32_right_normal_accept \
  --group-count 20 \
  --images-per-group 1 \
  --manual-load \
  --hdr \
  --short-exposure 1500 \
  --long-exposure 5500 \
  --gain 0 \
  --capture-interval 0.2 \
  --hdr-settle-frames 1 \
  --timeout-ms 2000 \
  --root "$ACCEPT_ROOT"
```

验收这一个 session：

```bash
.venv/bin/python - <<'PY'
import csv
import os
from pathlib import Path

root = Path(os.environ["ACCEPT_ROOT"])
manifest = root / "manifests" / f"{os.environ['CAPTURE_SESSION']}.csv"
with manifest.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
images = [row for row in rows if row["record_type"] == "image"]
complete = [row for row in rows if row["record_type"] == "sample" and row["sample_status"] == "complete"]
assert len(images) == 8, len(images)
assert len(complete) == 1, len(complete)
assert all(Path(row["file"]).is_file() for row in images)
assert {row["view"] for row in images} == {
    "front", "front_left", "front_right", "front_secondary",
    "back", "back_left", "back_right", "back_secondary",
}
print("四相机一组验收通过:", manifest)
PY
```

### 2.2 采集左右手 normal

Stage 30 目前要求左右手六个主视图都有正常参考图，所以从零开始时左右手 normal 都要采。下面每只手采 120 组：

当前旧下游用 `hand + label + defect_type + groupNNN` 恢复物理件身份，没有把 session 加入身份。因此正式数据中，
同一 `hand/label/defect_type` 只能保留一个从 `group001` 开始的采集 session。上面的单组验收必须留在
`dataset_acceptance/`，不能混入正式 `dataset/`。批次中断后重新从 `group001` 采集前，也要先把失败 session
整体移出正式 `left/right` 数据树，不能让两个 session 的 `group001` 同时进入 Stage 30。

```bash
for HAND in right left; do
  SESSION="zs32_${HAND}_normal_$(date +%Y%m%d_%H%M%S_%N)"
  PYTHONPATH=/opt/MVS/Samples/64/Python/MvImport:${PYTHONPATH:-} \
  .venv/bin/python pipeline/zs32_bootstrap_capture.py \
    --topology configs/zs32/topology/zs32_4cam_double_side_v1.json \
    --capture-session "$SESSION" \
    --legacy-layout \
    --hand "$HAND" \
    --label normal \
    --part-id "zs32_${HAND}_normal_rebuild_v1" \
    --group-count 120 \
    --images-per-group 1 \
    --manual-load \
    --hdr \
    --short-exposure 1500 \
    --long-exposure 5500 \
    --gain 0 \
    --capture-interval 0.2 \
    --hdr-settle-frames 1 \
    --timeout-ms 2000 \
    --root "$DATASET_ROOT" || break
done
```

每一组的操作顺序都是：放正面并按 Enter，四机采 4 张；翻转同一零件并按 Enter，再采 4 张。

### 2.3 采集 defect

现有标注流程只接受缺陷类型 `deform`、`less`、`others`。下面以右手每类 30 组为例；`30` 是本轮采集参数，
应根据实际缺陷件数量调整。不要用同一物理件的大量重复摆拍替代独立缺陷件。

```bash
for DEFECT in deform less others; do
  SESSION="zs32_right_${DEFECT}_$(date +%Y%m%d_%H%M%S_%N)"
  PYTHONPATH=/opt/MVS/Samples/64/Python/MvImport:${PYTHONPATH:-} \
  .venv/bin/python pipeline/zs32_bootstrap_capture.py \
    --topology configs/zs32/topology/zs32_4cam_double_side_v1.json \
    --capture-session "$SESSION" \
    --legacy-layout \
    --hand right \
    --label defect \
    --defect-type "$DEFECT" \
    --part-id "zs32_right_${DEFECT}_rebuild_v1" \
    --group-count 30 \
    --images-per-group 1 \
    --manual-load \
    --hdr \
    --short-exposure 1500 \
    --long-exposure 5500 \
    --gain 0 \
    --capture-interval 0.2 \
    --hdr-settle-frames 1 \
    --timeout-ms 2000 \
    --root "$DATASET_ROOT" || break
done
```

如果还要训练左手缺陷数据，使用相同命令把 `right` 换成 `left`。当前 Stage 32/33 最终离线融合仍只支持右手。

### 2.4 原始数据目录

主六视图和 secondary 视图都必须保留：

```text
dataset/<hand>/<view>/normal/<session>/images/*.png
dataset/<hand>/<view>/defect/<defect_type>/<session>/images/*.png
dataset/manifests/<session>.csv
```

不要删除 `front_secondary/back_secondary`。它们只是暂时没有进入当前六视图训练代码。

## 3. 数据处理：选择和裁剪 ROI

四相机 legacy 采集已经是一件零件对应一张完整视图图像，不需要再运行通用的
`pipeline/2_process_data.py`。本流程的数据处理是分别为 YOLO 和 PatchCore 选择固定 ROI。

### 3.1 先准备 Label Studio 数据并标注

```bash
.venv/bin/python pipeline/27_prepare_zs32_label_studio.py \
  --dataset-root "$DATASET_ROOT" \
  --output-root "$LABEL_ROOT"
```

输出：

```text
dataset/zs32_yolo_labeling/images/
dataset/zs32_yolo_labeling/labeling_manifest.csv
dataset/zs32_yolo_labeling/label_studio_config.xml
```

启动 Label Studio：

```bash
conda activate label-studio
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$LABEL_ROOT"
label-studio start
```

在 Label Studio 中：

1. Source Storage 选择 `Local Files`；
2. Absolute local path 使用 `$LABEL_ROOT/images` 的展开后绝对路径；
3. File Filter Regex 使用 `.*\.png$`；
4. 把 `$LABEL_ROOT/label_studio_config.xml` 的内容粘贴到 Labeling Interface；
5. 只使用类别 `defect` 画矩形框；
6. 当前视图看不到缺陷时也要提交空 annotation，不能跳过任务；
7. 全部完成后导出 YOLO 格式并解压，确认导出目录中存在 `labels/*.txt`。

记录导出目录：

```bash
export LABEL_EXPORT="$LABEL_ROOT/REPLACE_WITH_YOUR_EXPORTED_YOLO_DIRECTORY"
test -d "$LABEL_EXPORT/labels"
```

### 3.2 构建完整六视图 YOLO 数据集

```bash
.venv/bin/python pipeline/28_prepare_zs32_yolo_dataset.py \
  --label-export-root "$LABEL_EXPORT" \
  --output-root "$YOLO_FULL" \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42
```

同一物理零件的六个主视图会进入同一个 split。检查：

```bash
test -f "$YOLO_FULL/data.yaml"
test -f "$YOLO_FULL/split_manifest.csv"
find "$YOLO_FULL/images" -type f | wc -l
find "$YOLO_FULL/labels" -type f | wc -l
```

### 3.3 选择 YOLO 六视图 ROI

```bash
.venv/bin/python pipeline/29_zs32_fixed_roi.py select \
  --input-root "$YOLO_FULL" \
  --config "$DATASET_ROOT/zs32_six_view_roi_config.json" \
  --preview-dir "$DATASET_ROOT/zs32_six_view_roi_previews"
```

程序会依次显示六个主视图。每个视图拖框后按 Enter/Space 接受。完成后检查 preview，再转换：

```bash
.venv/bin/python pipeline/29_zs32_fixed_roi.py convert \
  --input-root "$YOLO_FULL" \
  --config "$DATASET_ROOT/zs32_six_view_roi_config.json" \
  --output-root "$YOLO_ROI"
```

重点检查终端输出的 `Boxes clipped` 和 `Boxes dropped`。出现 dropped box 时必须确认 ROI 是否选错，不能直接训练。

### 3.4 选择 PatchCore/模板六视图 ROI

```bash
.venv/bin/python pipeline/30_crop_zs32_patchcore_dataset.py select \
  --dataset-root "$DATASET_ROOT" \
  --config "$DATASET_ROOT/zs32_patchcore_roi_config.json" \
  --preview-dir "$DATASET_ROOT/zs32_patchcore_roi_previews"

.venv/bin/python pipeline/30_crop_zs32_patchcore_dataset.py convert \
  --dataset-root "$DATASET_ROOT" \
  --config "$DATASET_ROOT/zs32_patchcore_roi_config.json" \
  --output-root "$PATCHCORE_ROI"
```

验收：

```bash
test -f "$PATCHCORE_ROI/crop_manifest.csv"
test -f "$DATASET_ROOT/zs32_patchcore_roi_config.json"
```

PatchCore 和模板使用这一份 crop manifest；YOLO 使用上一节转换 bbox 后的数据集。

## 4. 训练 PatchCore

当前可恢复训练脚本固定训练右手六个主视图，每个视图一个独立 PatchCore：

```bash
HF_HUB_OFFLINE=1 bash pipeline/run_patchcore_roi_six_views.sh \
  "$PATCHCORE_ROI" \
  "$PATCHCORE_RUN" \
  0
```

默认强配置为 `wide_resnet50_2`、`layer2 layer3`、`256x256`、coreset `0.05`、float32、batch `16`。
显存不足时不要修改已生成的同名 run；使用新的输出目录并降低 batch。

完成条件：

```bash
test -f "$PATCHCORE_RUN/six_view_summary.csv"
find "$PATCHCORE_RUN" -name model.ckpt -type f
```

应当找到 6 个不同视图的 checkpoint。重新执行同一命令会跳过已完成视图并继续失败视图。

## 5. 训练 YOLO

YOLO trainer 位于独立仓库 `/home/yunjing/ultralytics-c789`。本仓库负责数据准备和运行时调用，不包含 trainer。

下面是一套可执行的 640 baseline。`yolo26n.pt` 只是预训练初始化，最终运行必须使用训练生成的 `best.pt`。

```bash
cd "$ULTRA_ROOT"

/home/yunjing/miniconda3/envs/yolo/bin/python examples/c789/train.py \
  --data-yaml "$YOLO_ROI/data.yaml" \
  --model yolo26n.pt \
  --epochs 150 \
  --imgsz 640 \
  --batch 64 \
  --device 0 \
  --workers 8 \
  --project "$RUN_ROOT/yolo" \
  --name zs32_roi_n640_seed42 \
  --seed 42 \
  --patience 30 \
  --degrees 0 \
  --translate 0.03 \
  --scale 0.10 \
  --fliplr 0 \
  --flipud 0 \
  --mosaic 0 \
  --override hsv_h=.005 \
  --override hsv_s=.2 \
  --override hsv_v=.15

cd "$REPO"
```

如果显存不足，使用新的 run name 并降低 `--batch`。不要给正式训练加 `--exist-ok` 覆盖已有结果。

验收：

```bash
test -f "$YOLO_RUN/weights/best.pt"
test -f "$YOLO_RUN/args.yaml"
test -f "$YOLO_RUN/results.csv"
sha256sum "$YOLO_RUN/weights/best.pt"
```

参数和 seed 以新 run 的 `args.yaml` 为准，不要根据目录名猜测。

## 6. 训练模板匹配

模板和 PatchCore 使用同一份 Stage 30 ROI crop。当前完整离线运行只支持右手，所以本轮只训练右手六组：

```bash
.venv/bin/python pipeline/train_zs32_template_gate.py \
  --manifest "$PATCHCORE_ROI/crop_manifest.csv" \
  --path-root "$REPO" \
  --output-dir "$TEMPLATE_RUN" \
  --required-hand right \
  --model-version zs32-rebuild-v1 \
  --threshold-version zs32-rebuild-v1 \
  --roi-version zs32-patchcore-roi-rebuild-v1 \
  --template-version zs32-template-rebuild-v1
```

验收：

```bash
test -f "$TEMPLATE_RUN/model.json"
test -f "$TEMPLATE_RUN/calibration_rows.csv"
find "$TEMPLATE_RUN/templates" -type f | head
```

不要复用旧模板的阈值，因为 ROI 和新数据已经变化。

## 7. 生成本轮运行配置

Stage 32/33 不会自动猜 checkpoint。运行配置必须精确绑定六个 PatchCore checkpoint、YOLO `best.pt`、两份 ROI
配置和它们的 SHA-256。下面的脚本从新训练的 `six_view_summary.csv` 生成配置：

```bash
export PATCHCORE_RUN YOLO_RUN RUNTIME_CONFIG DATASET_ROOT

.venv/bin/python - <<'PY'
import csv
import hashlib
import json
import os
from pathlib import Path

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

run = Path(os.environ["PATCHCORE_RUN"]).resolve()
with (run / "six_view_summary.csv").open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))

by_view = {row["view"].removeprefix("right_"): row for row in rows}
views = ("front", "front_left", "front_right", "back", "back_left", "back_right")
assert set(by_view) == set(views), sorted(by_view)

patchcore = {}
for view in views:
    checkpoint = Path(by_view[view]["checkpoint"]).resolve()
    assert checkpoint.is_file(), checkpoint
    digest = sha256(checkpoint)
    patchcore[view] = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "model_version": f"patchcore-rebuild-{view.replace('_', '-')}-{digest[:12]}",
    }

yolo = Path(os.environ["YOLO_RUN"]).resolve() / "weights" / "best.pt"
assert yolo.is_file(), yolo
yolo_digest = sha256(yolo)
dataset = Path(os.environ["DATASET_ROOT"]).resolve()

payload = {
    "schema_version": 1,
    "product": "ZS32",
    "profile": "zs32_right_six_view_rebuild_v1",
    "supported_hands": ["right"],
    "patchcore_roi_config": str(dataset / "zs32_patchcore_roi_config.json"),
    "yolo_roi_config": str(dataset / "zs32_six_view_roi_config.json"),
    "versions": {
        "threshold": "zs32-rebuild-v1",
        "patchcore_roi": "zs32-patchcore-roi-rebuild-v1",
        "yolo_roi": "zs32-yolo-roi-rebuild-v1",
        "template": "zs32-template-rebuild-v1",
    },
    "patchcore": patchcore,
    "yolo": {
        "weights": str(yolo),
        "weights_sha256": yolo_digest,
        "model_version": f"yolo-best-{yolo_digest[:12]}",
        "imgsz": 640,
        "candidate_conf": 0.001,
        "iou": 0.7,
        "max_det": 300,
        "class_map": {"0": "defect"},
    },
}

output = Path(os.environ["RUNTIME_CONFIG"]).resolve()
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(output)
PY
```

只读验证配置和所有 hash：

```bash
.venv/bin/python - <<'PY'
import os
from pathlib import Path
from capture_data.zs32_model_runtime import load_runtime_config

config = load_runtime_config(Path(os.environ["RUNTIME_CONFIG"]))
print("runtime profile:", config.profile)
print("patchcore models:", len(config.patchcore))
print("yolo:", config.yolo.weights)
PY
```

## 8. Stage 33 离线联调和阈值诊断

Stage 33 会常驻加载六个 PatchCore 和一个 YOLO，按物理零件运行模板、PatchCore、YOLO，使用 YOLO 的真实 bbox
标签做阈值诊断。它只接受右手六视图，并且必须显式声明 `--commissioning-only`。

### 8.1 先只做 preflight

```bash
.venv/bin/python pipeline/33_run_zs32_offline_calibration.py \
  --crop-manifest "$PATCHCORE_ROI/crop_manifest.csv" \
  --template-calibration-csv "$TEMPLATE_RUN/calibration_rows.csv" \
  --template-model-dir "$TEMPLATE_RUN" \
  --runtime-config "$RUNTIME_CONFIG" \
  --yolo-dataset-root "$YOLO_ROI" \
  --path-root "$REPO" \
  --output-dir "$OFFLINE_RUN" \
  --commissioning-only \
  --yolo-aux-min-image-precision 0.90 \
  --accelerator gpu \
  --devices 1 \
  --yolo-device 0 \
  --preflight-only
```

必须看到 `preflight: ok`。如果这里失败，先修复数据 split、图片路径、标签、模型 hash 或 ROI，不要开始长时间推理。

### 8.2 执行完整离线联调

```bash
.venv/bin/python pipeline/33_run_zs32_offline_calibration.py \
  --crop-manifest "$PATCHCORE_ROI/crop_manifest.csv" \
  --template-calibration-csv "$TEMPLATE_RUN/calibration_rows.csv" \
  --template-model-dir "$TEMPLATE_RUN" \
  --runtime-config "$RUNTIME_CONFIG" \
  --yolo-dataset-root "$YOLO_ROI" \
  --path-root "$REPO" \
  --output-dir "$OFFLINE_RUN" \
  --commissioning-only \
  --yolo-aux-min-image-precision 0.90 \
  --accelerator gpu \
  --devices 1 \
  --yolo-device 0
```

中断后使用完全相同的命令并增加 `--resume`。输入文件、hash 或参数变化时，resume 会拒绝复用旧结果。

### 8.3 结果验收

```bash
test -f "$OFFLINE_RUN/commissioning_run_contract.json"
test -f "$OFFLINE_RUN/calibration_rows.csv"
test -f "$OFFLINE_RUN/template_patchcore_threshold_calibration/thresholds.json"
test -f "$OFFLINE_RUN/yolo_annotation_thresholds_by_view/summary.json"
test -f "$OFFLINE_RUN/yolo_high_precision_auxiliary/summary.json"
test -f "$OFFLINE_RUN/yolo_runtime_crop_identity.json"
test -f "$OFFLINE_RUN/threshold_quality_report.json"
```

重点查看：

```bash
.venv/bin/python -m json.tool "$OFFLINE_RUN/threshold_quality_report.json"
.venv/bin/python -m json.tool "$OFFLINE_RUN/yolo_annotation_thresholds_by_view/summary.json"
.venv/bin/python -m json.tool "$OFFLINE_RUN/yolo_high_precision_auxiliary/summary.json"
```

验收原则：

- `yolo_runtime_crop_identity.json` 必须显示像素身份校验通过；
- 任何视图出现零阈值、样本不足、召回不达标或 `operationally_usable=false`，都应补数据/修模型后重跑；
- `cases/` 中要人工抽查正常误报、缺陷漏检、PatchCore 热图和 YOLO 框图；
- test split 只能评估，不能回头参与选择阈值；
- 本阶段输出仍是 commissioning，不是生产 release。

## 9. 单个零件离线冒烟测试

Stage 33 完成前，可以使用 Stage 32 `infer` 检查新模型是否能对一件右手零件生成三类连续分数。传入的是六张
原始 4024x3036 主视图，不是 ROI crop：

```bash
.venv/bin/python pipeline/32_run_zs32_multimodel_inference.py infer \
  --part-id smoke_part_001 \
  --capture-session REPLACE_SESSION \
  --group-id group001 \
  --hand right \
  --front-image /absolute/path/front.png \
  --front-left-image /absolute/path/front_left.png \
  --front-right-image /absolute/path/front_right.png \
  --back-image /absolute/path/back.png \
  --back-left-image /absolute/path/back_left.png \
  --back-right-image /absolute/path/back_right.png \
  --runtime-config "$RUNTIME_CONFIG" \
  --template-model-dir "$TEMPLATE_RUN" \
  --accelerator gpu \
  --devices 1 \
  --yolo-device 0 \
  --output-dir "$RUN_ROOT/smoke_part_001"
```

输出应包含 `template_match.csv`、`patchcore.csv`、`yolo.csv`、`runtime_manifest.json` 和证据图片。若模板明确
`NG_TEMPLATE` 或 REVIEW 短路，下游模型不运行是当前 fail-closed 策略，不是程序崩溃。

## 10. 本轮完成定义

只有以下项目全部完成，才能说“方案 1 的现有六视图闭环跑完”：

- [ ] 四机一组验收为 8 张图和 1 条 complete sample；
- [ ] 左右手 normal 已采集，右手三类 defect 已采集；
- [ ] 所有原始 manifest 无 incomplete sample；
- [ ] Label Studio 每个主视图都提交了标注或空 annotation；
- [ ] YOLO train/val/test 按物理 group 隔离；
- [ ] YOLO ROI 没有未经复核的 dropped bbox；
- [ ] PatchCore 六个 checkpoint 和 `six_view_summary.csv` 存在；
- [ ] YOLO `best.pt`、`args.yaml`、`results.csv` 存在；
- [ ] 右手模板 `model.json` 和 `calibration_rows.csv` 存在；
- [ ] runtime config 的模型和 ROI hash 全部验证通过；
- [ ] Stage 33 preflight 和完整批跑成功；
- [ ] 阈值质量报告、YOLO 分视图报告和人工证据抽查通过；
- [ ] 结果明确标记为 `commissioning_only`。

## 11. 要实现真正四相机 8-view 全流程还缺什么

`front_secondary/back_secondary` 要正式进入模型，至少需要完成：

1. Stage 27 Label Studio discovery 从 6 view 扩展到 topology-driven 8 view；
2. Stage 28 YOLO group、镜像规则和 split manifest 扩展到 8 view；
3. Stage 29/30 ROI 配置和选择器增加两个 secondary view；
4. PatchCore runner 训练 8 个右手 checkpoint，而不是固定 6 个；
5. 模板训练合同增加两个 view；
6. 为 secondary view 完成人工 bbox 标注和正常/缺陷分支目标；
7. runtime config、Stage 32/33 和融合 profile 从 18 组扩展为 24 组三模型证据；
8. 使用独立 calibration/test 物理件重新锁定阈值；
9. 新版正式链还要准备四机 gate publication、8-view ROI、semantics v2、calibration targets、recipe 和
   candidate/release 资产。

在这些代码和资产完成前，不能把 secondary 两个目录简单改名混进旧六视图模型，也不能复制其他视图的 ROI 或阈值。

正式不可变发布链另见 [ZS32 Linux refactor runbook](ZS32_LINUX_REFACTOR_RUNBOOK.md)。当前本文描述的是可操作的
历史六视图 commissioning 闭环，不替代正式 release 审批。
