# ZS32 相机拓扑配置

本目录保存正式 `zs32_inspection` 主链使用的相机、轮次和视图身份合同。拓扑只描述
“哪台物理相机在每轮产生哪个 canonical view”，不包含曝光、ROI、质量阈值或配准阈值。

## 已跟踪拓扑

| 配置 | 相机数 | 每件视图数 | 用途 |
| --- | ---: | ---: | --- |
| `zs32_3cam_double_side_v1.json` | 3 | 6 | 现有三机正式主链 |
| `zs32_4cam_double_side_v1.json` | 4 | 8 | 增加正面第二相机 `DB0968108` 的正式主链 |

四相机映射为：

| 物理槽位 | 序列号 | 正面轮 | 翻面后 |
| --- | --- | --- | --- |
| `center` | `DA9805574` | `front` | `back` |
| `left_oblique` | `DA9625347` | `front_left` | `back_left` |
| `right_oblique` | `DB0998274` | `front_right` | `back_right` |
| `front_secondary` | `DB0968108` | `front_secondary` | `back_secondary` |

`front_secondary` 表示第四台相机安装在工位正面、作为第二个正面视角。翻转同一零件后，
同一物理相机产生 `back_secondary`。四台相机仍执行 `front`、`back` 两轮，每件完整样本
必须恰好包含 8 个唯一视图。

## 1. 验证拓扑配置

在仓库根目录执行：

```bash
.venv/bin/python -c '
from zs32_inspection.config.loaders import load_topology
t = load_topology("configs/zs32/topology/zs32_4cam_double_side_v1.json")
print("topology_id:", t.topology_id)
print("topology_sha256:", t.topology_sha256)
print("required_views:", t.required_views)
'
```

预期 `topology_id=zs32-4cam-double-side-v1`，并输出 8 个 `required_views`。拓扑内容发生
任何变化时 SHA256 都会变化，旧 gate publication、ROI、recipe、模型或阈值不能继续复用。

## 2. 确认四台相机在线

设备枚举仍可借用旧采集器的只读入口：

```bash
.venv/bin/python pipeline/1_collect_multicamera_data.py --list-devices
```

输出必须同时包含：

```text
DA9805574
DA9625347
DB0998274
DB0968108
```

SDK 枚举编号可能变化，正式身份只按 serial 绑定。不要把
`--right-serial DB0968108` 当成四机模式；旧 Stage 1 只会用它替换原右侧相机，仍然只采三台。

## 3. 采集四机 legacy 数据

`--legacy-layout` 直接写入历史数据集目录和 25 列 CSV manifest，不复制 `_bootstrap`
图片。必须先完成 1 组硬件验收，确认 8 张 PNG 和 9 条数据行后，才可执行 120 组命令。

### 3.1 先采 1 组

在本仓库根目录直接复制执行：

```bash
export CAPTURE_SESSION="zs32_left_legacy_accept_$(date +%Y%m%d_%H%M%S_%N)"
UV_CACHE_DIR=/tmp/uv-cache \
PYTHONPATH=/opt/MVS/Samples/64/Python/MvImport:${PYTHONPATH:-} \
uv run --no-sync python pipeline/zs32_bootstrap_capture.py \
  --topology configs/zs32/topology/zs32_4cam_double_side_v1.json \
  --capture-session "$CAPTURE_SESSION" \
  --legacy-layout \
  --hand left \
  --label normal \
  --part-id zs32_left_normal_part2 \
  --group-count 1 \
  --images-per-group 1 \
  --manual-load \
  --hdr \
  --short-exposure 1500 \
  --long-exposure 5500 \
  --gain 0 \
  --capture-interval 0.2 \
  --hdr-settle-frames 1 \
  --timeout-ms 2000 \
  --root /home/yunjing/anomalib/dataset/test
```

采集成功后执行：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python - <<'PY'
import csv
import os
from pathlib import Path

manifest_dir = Path("/home/yunjing/anomalib/dataset/test/manifests")
manifest = manifest_dir / f"{os.environ['CAPTURE_SESSION']}.csv"
with manifest.open(newline="", encoding="utf-8") as stream:
    reader = csv.DictReader(stream)
    rows = list(reader)
assert tuple(reader.fieldnames or ()) == (
    "record_type", "session_id", "sample_id", "group_id", "image_index",
    "round", "view", "device_index", "camera_serial", "capture_mode",
    "exposure", "gain", "file", "source_short", "source_long",
    "short_exposure", "long_exposure", "hdr_attempt", "fused_clip_pct",
    "captured_at", "sample_status", "failed_round", "failed_view",
    "failed_device_index", "error",
)
image_rows = [row for row in rows if row["record_type"] == "image"]
complete_rows = [
    row
    for row in rows
    if row["record_type"] == "sample" and row["sample_status"] == "complete"
]
assert len(rows) == 9
assert len(image_rows) == 8
assert len(complete_rows) == 1
assert {row["view"] for row in image_rows} == {
    "front", "front_left", "front_right", "front_secondary",
    "back", "back_left", "back_right", "back_secondary",
}
assert all(Path(row["file"]).is_file() for row in image_rows)
print("legacy four-camera one-group verified")
PY
```

通过条件是八个 view 目录共 8 张 PNG，且该 capture session 的 manifest 恰好有 8 条 `image` 和 1 条
`complete sample`，合计 9 条数据行（不计 CSV 表头）。

### 3.2 验收 1 组后再采 120 组

```bash
export CAPTURE_SESSION="zs32_left_legacy_batch_$(date +%Y%m%d_%H%M%S_%N)"
UV_CACHE_DIR=/tmp/uv-cache \
PYTHONPATH=/opt/MVS/Samples/64/Python/MvImport:${PYTHONPATH:-} \
uv run --no-sync python pipeline/zs32_bootstrap_capture.py \
  --topology configs/zs32/topology/zs32_4cam_double_side_v1.json \
  --capture-session "$CAPTURE_SESSION" \
  --legacy-layout \
  --hand left \
  --label normal \
  --part-id zs32_left_normal_part2 \
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
  --root /home/yunjing/anomalib/dataset/test
```

120 组完成后执行：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python - <<'PY'
import csv
import os
from pathlib import Path

manifest_dir = Path("/home/yunjing/anomalib/dataset/test/manifests")
manifest = manifest_dir / f"{os.environ['CAPTURE_SESSION']}.csv"
with manifest.open(newline="", encoding="utf-8") as stream:
    reader = csv.DictReader(stream)
    rows = list(reader)
assert len(reader.fieldnames or ()) == 25
assert sum(row["record_type"] == "image" for row in rows) == 960
assert sum(
    row["record_type"] == "sample" and row["sample_status"] == "complete"
    for row in rows
) == 120
assert len(rows) == 1080
print("legacy four-camera batch verified")
PY
```

通过条件是 960 条 `image` 加 120 条 `complete sample`，合计 1080 条数据行。

本机海康 SDK 通过命令中的 `PYTHONPATH=/opt/MVS/Samples/64/Python/MvImport:...` 加载。每组操作顺序是：

1. 放好同一个零件的正面，按 `Enter`，四台相机 grouped trigger/read 得到 4 张正面图；
2. 将同一个零件翻到背面，按 `Enter`，再采 4 张图；
3. 程序发布 8 张 PNG 和对应 CSV 行，然后进入下一组。输入 `q` 可取消。

四台相机在整个批次中只打开一次。命令明确要求 `--images-per-group 1`，不保存 HDR
short/long source，默认不做 HDR 对齐。成功 legacy 输出位于：

```text
/home/yunjing/anomalib/dataset/test/left/<view>/normal/<session_id>/images/*.png
/home/yunjing/anomalib/dataset/test/manifests/<session_id>.csv
```

不加 `--legacy-layout` 时，现有隔离 bootstrap 模式仍保持不变：成功结果写到 `_bootstrap/`，
失败或取消写到 `_bootstrap_incomplete/`，并保持 `bootstrap_only=true`、
`eligible_for_dataset=false` 和 `production_release_allowed=false`。

## 4. 第一次正式四机采集前的资产

`zs32-capture` 不接受松散的 `--topology`、serial 或曝光参数。它只接受一个已经验证的
`--gate-publication`，或正式上线后的 `--release`。因此必须先针对四机拓扑准备并审核：

- 完整的 `capture/acquisition/hikvision.json`；
- 当前 hand 的 8-view quality profile；
- 当前 hand 的 8-view registration profile；
- 8 张真实、全分辨率、人工批准的 registration reference PNG；
- schema v2 capture gate policy，其中 ArtifactRef 绑定上述文件的精确 SHA256。

这些阈值和参考图必须来自现场实拍，不能把旧视图的 ROI、阈值或图片复制给
`front_secondary/back_secondary`。详细字段合同见
[`../capture/README.md`](../capture/README.md)。

## 5. 发布四机 gate publication

准备好真实资产后执行：

```bash
.venv/bin/zs32-publish-gate-policy \
  --publication-id gate-zs32-left-4cam-r1 \
  --output-root /path/to/zs32/gate-publications \
  --topology configs/zs32/topology/zs32_4cam_double_side_v1.json \
  --policy /path/to/zs32/capture_gate_policy.local.json \
  --asset-root /path/to/zs32/gate_assets
```

发布器会在打开相机前所需的信任边界上验证 topology、8-view profile 完整性、文件 hash 和
reference PNG 尺寸。任何视图缺失或 hash 不一致都会拒绝发布。

## 6. 执行正式四机采集

```bash
.venv/bin/zs32-capture \
  --gate-publication /path/to/zs32/gate-publications/gate-zs32-left-4cam-r1 \
  --raw-root /path/to/zs32/raw \
  --capture-session session-20260714-4cam-a \
  --capture-set-id capture-left-part0001-a \
  --part-instance-id zs32-left-part0001 \
  --hand left \
  --operator-id operator001 \
  --round-confirmation-timeout 120
```

操作员必须在前台 TTY 中依次输入：

```text
CONFIRM front
CONFIRM back
```

程序对四台相机执行 grouped software trigger：每轮先依次触发四台，再依次读取四帧。它适合
静止工件，但不等同于共享硬件触发线的严格同时曝光。完整 capture set 才能进入后续数据集。

## 7. 后续数据集与训练边界

ROI 不阻塞第一次 raw capture，但会阻塞 `zs32-build-dataset`。四机数据集必须另外准备：

- 绑定 `zs32-4cam-double-side-v1` 的新 ROI 配置；
- 当前启用 hand 的 8 个实测 ROI；
- 每个物理件精确覆盖 8 个视图的 semantics、YOLO 标签和 calibration targets；
- 8 个 template 模型和 8 个 PatchCore 模型；
- 基于四机 YOLO export 重新训练并回导的全局 YOLO；
- 覆盖 8 个 view 的 template/anomaly/YOLO 标定结果。

现有 `zs32_roi_v2.json`、`zs32_right_patchcore_training_v1.json`、六视图 checkpoint、阈值和
Stage 35 commissioning 均绑定三机拓扑，不能与四机 capture 混用。Stage 2 会扫描 view
目录，因此可发现全部 8 个 view；但 Stage 3、YOLO、Label Studio 和部分训练入口仍只有
历史六视图合同。本次 legacy 直写不等于整条下游训练链已支持
`front_secondary/back_secondary`；八视图训练和发布必须在后续任务中独立扩展。
