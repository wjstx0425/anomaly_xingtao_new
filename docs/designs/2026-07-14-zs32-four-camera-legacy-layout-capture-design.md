# ZS32 四相机旧数据格式直写设计

日期：2026-07-14  
状态：已实现

## 1. 目标

让现有四相机 topology-driven 采集入口直接生成历史
`pipeline/1_collect_multicamera_data.py` 使用的数据目录、文件名和 CSV manifest，不再要求把
`_bootstrap/<session>/<capture_set>/images` 结果另行转换。

四相机拓扑保持不变：前三台继续产生六个历史视图，第四台 `DB0968108` 产生
`front_secondary` 和 `back_secondary`。每个物理零件必须包含八个唯一视图。

## 2. 用户入口

在 `pipeline/zs32_bootstrap_capture.py` 增加显式参数：

```text
--legacy-layout
```

未指定时保留当前隔离 `_bootstrap` 输出，避免静默改变已有命令。指定后只发布历史数据集格式，
不再复制一份 `_bootstrap` 图片。

legacy 模式继续支持现有参数：`hand`、`label`、`defect-type`、`part-id`、`group-count`、
`images-per-group=1`、HDR/曝光、manual-load 和 topology。四台相机在整个批次中只打开一次。

legacy 模式默认生成历史 session ID：

```text
YYYYMMDD_HHMMSS_microseconds
```

若显式传入 `--capture-session`，必须满足安全标识符规则，并直接作为历史 session ID 使用。

## 3. 成功输出合同

normal 数据写入：

```text
<root>/<hand>/<view>/normal/<session_id>/images/<filename>.png
```

defect 数据写入：

```text
<root>/<hand>/<view>/defect/<defect_type>/<session_id>/images/<filename>.png
```

八个 view 为：

```text
front
front_left
front_right
front_secondary
back
back_left
back_right
back_secondary
```

文件名严格沿用历史规则：

```text
{hand}_{view}_{label_token}_{part_id}_groupNNN_{image_index:06d}_{kind}.png
```

其中：

- normal 的 `label_token=normal`；
- defect 的 `label_token=defect_{defect_type}`；
- HDR 融合图的 `kind=fused`；
- 单曝光图的 `kind=single`；
- 当前四机入口不保存 HDR short/long source，CSV 中相应路径留空。

示例：

```text
left_front_secondary_normal_zs32_left_normal_part2_group001_000001_fused.png
```

## 4. CSV manifest 合同

manifest 路径保持为：

```text
<root>/manifests/<session_id>.csv
```

列顺序严格保持历史 25 列：

```text
record_type,session_id,sample_id,group_id,image_index,round,view,
device_index,camera_serial,capture_mode,exposure,gain,file,source_short,
source_long,short_exposure,long_exposure,hdr_attempt,fused_clip_pct,
captured_at,sample_status,failed_round,failed_view,failed_device_index,error
```

完整四机样本恰好写九行：

- 八行 `record_type=image`，分别绑定 topology round/view/serial；
- 一行 `record_type=sample`；
- 所有行 `sample_status=complete`；
- `sample_id={part_id}_groupNNN_000001`；
- `device_index` 保留当次 SDK 枚举值，身份判断仍以 `camera_serial` 为准。

## 5. 发布与失败处理

每个样本先在八个目标 view 目录分别写临时 PNG。只有八张图全部写入成功后才依次原子 rename
到最终文件名。manifest 更新采用“读取已有行、写完整临时 CSV、原子替换”的方式加入该样本九行，
不直接对正式 CSV 做多次 append。若图片或 CSV 发布失败，清理本次已经发布的图片，不得留下没有
complete manifest 的半成品。

相机采集失败时：

- 不写 `complete` sample；
- 已形成完整 `CaptureFrame` 的视图可以按历史文件名保留；
- manifest 写对应 image 行和一行 `sample_status=incomplete` 汇总行；
- 填写 `failed_round`、`failed_view`、`failed_device_index` 和 `error`；
- 结束当前批次并释放全部相机。

同一 session/文件名已存在时必须在打开相机前失败，禁止覆盖历史数据。

## 6. 下游边界

`pipeline/2_process_data.py` 不读取采集 manifest，而是扫描
`<hand>/<position>/<label>/**/images/*`，因此上述八个 view 目录都能被 Stage 2 发现。

本次范围只解决采集格式。现有 Stage 3、YOLO/Label Studio 和部分训练入口仍硬编码历史六视图；
它们能继续处理原六个 view，但要正式训练 `front_secondary/back_secondary`，需要后续独立扩展这些
下游 view contracts。本次实现不得宣称整条训练链已经支持八视图。

## 7. 测试与验收

离线测试至少覆盖：

1. normal 四机样本生成八个历史 view 路径和九行 CSV；
2. defect 数据正确插入 `defect/<defect_type>`；
3. 文件名、session、sample/group/index 与历史规则完全一致；
4. CSV 列顺序与历史 25 列完全一致；
5. serial/view/round/device index 来源于真实 topology 和 frame；
6. 失败样本不能出现 complete 汇总行；
7. 目标文件冲突在相机初始化前失败；
8. legacy 模式不生成 `_bootstrap` 图片副本；
9. Stage 2 discovery 能发现新增 secondary view 图片。

真实硬件验收先运行 `group-count=1`，确认八张图片、九行 CSV 和目录命名，再运行120组；120组
结束后应有960条 image行、120条 complete sample行，共1080条数据行。

实现完成后的离线验证结果：

- capture focused pytest：`57 passed in 0.18s`；
- domain topology pytest：`9 passed in 0.02s`；
- `py_compile` 和 `pipeline/zs32_bootstrap_capture.py --help` 均以退出码 0 完成；
- 四机 topology JSON 和 25 列 legacy CSV 合同通过静态检查；
- `git diff --check` 无输出。

上述结果仅证明离线实现和合同，不代表已执行真实 1 组或 120 组硬件采集。操作员必须先按
`configs/zs32/topology/README.md` 执行 1 组验收，确认 8 张 PNG 和 9 条数据行，再执行
120 组命令。Stage 2 可发现 8 个 view，但 Stage 3、YOLO、Label Studio 和部分训练仍只支持
历史六视图。
