# ZS32 左手件六视角三相机单曝光与 HDR 采集设计

## 目标

新增一个单终端采集入口，在一个进程中控制三台海康相机。每个 ZS32 左手件先采集正面、正面左侧和正面右侧，再提示用户翻面并采集背面、背面左侧和背面右侧。六个视角共享同一个采集组标识。默认使用单曝光以先稳定走通流程，显式传入 `--hdr` 时保留可调 HDR 采集。

该入口服务于静止零件。三台相机采用软件触发，目标是让每个曝光档先快速触发全部相机，再读取全部图像；它不承诺硬件级同时曝光。

## 用户流程

1. 启动命令后，程序枚举设备并打印设备编号、序列号和视角映射。
2. 程序按 USB 序列号解析并一次性打开正面、左侧面和右侧面相机，不依赖可能变化的枚举编号。
3. 每个采集组提示用户放置零件正面；按 `Enter` 或 `s` 后采集三个正面视角。
4. 程序提示用户把同一个零件翻到背面；再次按 `Enter` 或 `s` 后采集三个背面视角。
5. 六个视角全部成功保存后，该组才记为成功；然后进入下一组。
6. 用户按 `Ctrl+C` 时，程序停止并关闭全部已打开的相机。

默认相机与视角映射如下：

| USB 序列号 | 物理位置 | 正面轮视角 | 背面轮视角 |
| --- | --- | --- | --- |
| `DA9805574` | 正面 | `front` | `back` |
| `DA9625347` | 左侧面 | `front_left` | `back_left` |
| `DB0998274` | 右侧面 | `front_right` | `back_right` |

零件目录固定使用 `hand=left`；命令行仍显式保留 `--hand left`，并拒绝其他值，避免数据写入错误的手性目录。

## 入口与模块边界

新增两个文件：

- `capture_data/collect_multicamera_dataset.py`：设备枚举、映射校验、相机生命周期、软件触发、HDR 采集、融合、原图保存、manifest 和错误处理。
- `pipeline/1_collect_multicamera_data.py`：与其他 pipeline stage 一致的薄包装，只把参数转发给核心脚本。

核心模块复用 `capture_data/collect_dataset.py` 中已经验证的帧格式转换、曝光融合参数、人工按键等待和元数据约定，但不调用其单相机 `capture()` 循环。多相机采集为每台相机持有独立 frame info 和 50 MiB buffer。

SDK 相关逻辑与纯编排逻辑分离，使离线单元测试无需连接真实相机，也无需依赖可用的 MVS 设备。

## 单曝光采集顺序

单曝光是默认模式，不需要 `--hdr`。每个正面轮或背面轮、每个图像序号按以下顺序执行：

1. 将三台相机都设置为 `--exposure` 指定的曝光时间。
2. 快速依次向三台相机发送软件触发命令。
3. 依次取回三台相机的图像。
4. 三路图像全部有效后再保存该轮结果。

单曝光模式不设置相机的 `AcquisitionFrameRate`，也不接受硬件 `--fps` 作为采集速度控制。静止件、人工翻面的流程天然提供足够间隔；当 `images-per-group > 1` 时，程序使用应用层采集间隔，避免把低 FPS 强行写入不同型号相机而触发 GenICam 范围错误。

单曝光支持：

- `--exposure`
- `--gain`
- `--timeout-ms`
- 应用层采集间隔参数

## HDR 采集顺序

每个正面轮或背面轮、每个图像序号按以下顺序执行：

1. 将三台相机都设置为短曝光。
2. 如配置了稳定帧，按三路成组的方式完成稳定帧采集。
3. 快速依次向三台相机发送软件触发命令。
4. 依次取回三台相机的短曝光图像。
5. 将三台相机都设置为长曝光。
6. 重复成组触发与读取，取得三张长曝光图像。
7. 分别对同一设备的短、长曝光图像做融合。
8. 所有三路结果均有效后，再保存该轮图像。

这保证触发顺序是 `trigger x3 -> read x3`，避免把完整的单相机 HDR 流程串行执行三次。静止零件不需要公共硬件触发线。

支持以下参数并允许用户调整：

- `--short-exposure`
- `--long-exposure`
- `--gain`
- 应用层采集间隔参数，不写入相机 `AcquisitionFrameRate`
- `--hdr-settle-frames`
- `--timeout-ms`
- `--short-dark-threshold`
- `--long-clip-threshold`
- `--blend-width`
- `--blur-size`
- `--align-hdr`
- `--save-hdr-sources`
- `--hdr-max-retries`
- `--hdr-max-clip-pct`

HDR 功能完整保留，但只在显式传入 `--hdr` 时启用。未传入 `--hdr` 时走上述单曝光路径，不切换短、长曝光，不做曝光融合，也不生成 HDR source 文件。

## 数据与命名

默认输出目录：

```text
<root>/left/front/<label>/<session>/images
<root>/left/front_left/<label>/<session>/images
<root>/left/front_right/<label>/<session>/images
<root>/left/back/<label>/<session>/images
<root>/left/back_left/<label>/<session>/images
<root>/left/back_right/<label>/<session>/images
<root>/manifests/<session>.csv
```

缺陷数据继续在 `<label>/` 下使用缺陷类型层级，与现有 stage 1 保持一致。一次程序运行只生成一个共享 `session_id`。文件名包含 `hand`、视角、标签、零件编号、三位组号和六位图像序号。

session manifest 每行表示一张最终输出图（单曝光原图或 HDR 融合图），至少记录：

- `session_id`
- `sample_id`
- `group_id`
- `image_index`
- `round` (`front` 或 `back`)
- `view`
- `device_index`
- `camera_serial`
- `file`
- `source_short`
- `source_long`
- 采集模式、单曝光或 HDR 参数
- 采集时间
- 状态和错误信息

同一零件的六张图共享 `sample_id`。完成六张图后才将该 sample 标记为完整。

## 设备绑定

默认按以下三个 USB 序列号绑定物理位置：

- `--front-serial DA9805574`
- `--left-serial DA9625347`
- `--right-serial DB0998274`

启动时验证：

- 恰好解析到三个不同的设备；
- 三个序列号互不重复；
- 三个序列号均存在；
- 三台设备均可打开；
- 打印物理位置、当前枚举编号、型号和 USB 序列号的最终映射。

枚举编号只作为本次进程打开 SDK 设备的内部索引，不作为视角身份。重新插拔导致 `device 0/1/2` 顺序变化时，视角映射仍保持正确。`--list-devices` 用于查看当前设备，但正式采集按序列号选择。

相机使用独占模式打开。正常退出、异常退出、部分初始化失败或 `Ctrl+C` 时，都必须关闭和销毁所有已创建句柄，并将已经配置过的相机恢复到连续采集可用状态，至少把 `TriggerMode` 恢复为 `Off`。配置参数应先验证可写范围；不得在设置中途失败后把相机永久留在软件触发状态。

## 完整性与错误处理

- 打开或启动过程中失败时，逆序关闭所有已经成功打开的相机。
- 每轮先把三路最终结果都保存在内存中，确认齐全后再写盘。
- 检查 `cv2.imwrite()` 返回值，写盘失败视为该轮失败。
- 任意相机超时、单曝光失败或 HDR 失败时，manifest 记录失败原因，不把该组六视角标记为完整。
- 正面轮成功但背面轮失败时，保留已写入的正面图作为失败组证据，但 manifest 明确标记 sample 不完整，后续不得当作完整六视角样本使用。
- 错误信息必须包含 group、round、device 和 view，便于现场定位。
- 程序退出时，对每台已启动相机执行 stop、close 和 destroy；一台清理失败不得阻止其他相机清理。

## 单曝光命令行示例

```bash
.venv/bin/python pipeline/1_collect_multicamera_data.py \
  --front-serial DA9805574 \
  --left-serial DA9625347 \
  --right-serial DB0998274 \
  --hand left \
  --label normal \
  --part-id zs32 \
  --group-count 60 \
  --images-per-group 1 \
  --manual-load \
  --exposure 4000 \
  --gain 0 \
  --timeout-ms 3000 \
  --root dataset/zs32
```

## HDR 命令行示例

```bash
.venv/bin/python pipeline/1_collect_multicamera_data.py \
  --front-serial DA9805574 \
  --left-serial DA9625347 \
  --right-serial DB0998274 \
  --hand left \
  --label normal \
  --part-id zs32 \
  --group-count 60 \
  --images-per-group 1 \
  --manual-load \
  --hdr \
  --save-hdr-sources \
  --align-hdr \
  --short-exposure 4000 \
  --long-exposure 35000 \
  --gain 0 \
  --root dataset/zs32
```

`--manual-load` 在六视角流程中分别控制正面和背面两次确认。若未提供该参数，程序仍按正面轮、背面轮执行，但不等待按键，主要用于自动化测试。

## 测试与验收

离线测试放在 `tests/unit/capture_data/test_collect_multicamera_dataset.py`，覆盖：

- 三个相机序列号的解析、缺失和重复校验；
- 正面、左侧面、右侧面序列号到六个视角的稳定映射；
- 单曝光模式不写入 `AcquisitionFrameRate`，并严格执行 `trigger x3 -> read x3`；
- 配置失败、正常退出和异常退出均恢复 `TriggerMode=Off`；
- 每个曝光档严格执行 `trigger x3 -> read x3`；
- 每台相机使用独立 buffer 和 frame info；
- 六视角共享 session、group 和 sample 标识；
- 任意一路读取失败时 sample 不完整；
- `imwrite` 返回 false 时报告保存失败；
- 部分设备打开或启动失败时仍完整清理资源；
- pipeline wrapper 正确转发参数。

真实硬件 smoke test 使用一个临时输出目录，验收顺序为：

1. 执行 `--list-devices`，确认三个目标序列号存在并核对最终物理位置映射。
2. 先运行一次正面和背面单曝光采集。
3. 验证六张单曝光图均能由 OpenCV 读取，尺寸和通道合理。
4. 验证 manifest 中六个视角共享同一个 sample，并标记完整。
5. 验证程序退出后 MVS 客户端能够以连续采集方式重新打开三台相机。
6. 单曝光流程稳定后，再单独运行 HDR smoke test；启用 `--save-hdr-sources` 时验证六张融合图和十二张曝光原图。
7. 记录软件触发间隔和端到端耗时；报告时明确区分曝光时间与程序总耗时。

smoke test 只写入临时目录，不污染正式 `dataset/`。硬件测试若发现设备编号与物理视角不符，停止正式采集并要求重新核对映射，不根据画面内容自动猜测。

## 文档和仓库记忆

`pipeline/README.md` 增加单终端六视角采集章节，包括设备映射、HDR 参数、翻面流程、输出结构、完整性语义和软件触发边界。

完成实现与验证后更新仓库根目录的 `AGENTS_MEMORY.md`，记录新入口、六视角命名、默认设备映射、smoke test 命令和真实验证结果。
