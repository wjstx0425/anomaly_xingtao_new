# ZS32 左手件六视角三相机 HDR 采集设计

## 目标

新增一个单终端采集入口，在一个进程中控制三台海康相机。每个 ZS32 左手件先采集正面、正面左侧和正面右侧，再提示用户翻面并采集背面、背面左侧和背面右侧。六个视角共享同一个采集组标识，并支持可调 HDR 参数。

该入口服务于静止零件。三台相机采用软件触发，目标是让每个曝光档先快速触发全部相机，再读取全部图像；它不承诺硬件级同时曝光。

## 用户流程

1. 启动命令后，程序枚举设备并打印设备编号、序列号和视角映射。
2. 程序一次性打开并配置 `device 0`、`device 1`、`device 2`。
3. 每个采集组提示用户放置零件正面；按 `Enter` 或 `s` 后采集三个正面视角。
4. 程序提示用户把同一个零件翻到背面；再次按 `Enter` 或 `s` 后采集三个背面视角。
5. 六个视角全部成功保存后，该组才记为成功；然后进入下一组。
6. 用户按 `Ctrl+C` 时，程序停止并关闭全部已打开的相机。

默认相机与视角映射如下：

| 设备 | 正面轮视角 | 背面轮视角 |
| --- | --- | --- |
| `device 0` | `front` | `back` |
| `device 1` | `front_left` | `back_left` |
| `device 2` | `front_right` | `back_right` |

零件目录固定使用 `hand=left`；命令行仍显式保留 `--hand left`，并拒绝其他值，避免数据写入错误的手性目录。

## 入口与模块边界

新增两个文件：

- `capture_data/collect_multicamera_dataset.py`：设备枚举、映射校验、相机生命周期、软件触发、HDR 采集、融合、原图保存、manifest 和错误处理。
- `pipeline/1_collect_multicamera_data.py`：与其他 pipeline stage 一致的薄包装，只把参数转发给核心脚本。

核心模块复用 `capture_data/collect_dataset.py` 中已经验证的帧格式转换、曝光融合参数、人工按键等待和元数据约定，但不调用其单相机 `capture()` 循环。多相机采集为每台相机持有独立 frame info 和 50 MiB buffer。

SDK 相关逻辑与纯编排逻辑分离，使离线单元测试无需连接真实相机，也无需依赖可用的 MVS 设备。

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
- `--fps`
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

首版要求使用 HDR，不提供额外的自由运行单曝光分支，以保持六视角采集路径单一清晰。

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

session manifest 每行表示一张最终融合图，至少记录：

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
- HDR 参数
- 采集时间
- 状态和错误信息

同一零件的六张图共享 `sample_id`。完成六张图后才将该 sample 标记为完整。

## 设备绑定

默认接受 `--devices 0 1 2`，其顺序固定代表正面、左侧面和右侧面。启动时验证：

- 恰好提供三个设备；
- 设备编号互不重复；
- 三个编号均存在；
- 三台设备均可打开；
- 打印每台相机的型号和 USB 序列号。

同时提供 `--list-devices`，让用户在正式采集前核对映射。实现预留按 USB 序列号绑定的能力；编号映射是本次交付的默认操作方式。

## 完整性与错误处理

- 打开或启动过程中失败时，逆序关闭所有已经成功打开的相机。
- 每轮先把三路融合结果都保存在内存中，确认齐全后再写盘。
- 检查 `cv2.imwrite()` 返回值，写盘失败视为该轮失败。
- 任意相机超时或 HDR 失败时，manifest 记录失败原因，不把该组六视角标记为完整。
- 正面轮成功但背面轮失败时，保留已写入的正面图作为失败组证据，但 manifest 明确标记 sample 不完整，后续不得当作完整六视角样本使用。
- 错误信息必须包含 group、round、device 和 view，便于现场定位。
- 程序退出时，对每台已启动相机执行 stop、close 和 destroy；一台清理失败不得阻止其他相机清理。

## 命令行示例

```bash
.venv/bin/python pipeline/1_collect_multicamera_data.py \
  --devices 0 1 2 \
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

- 三个设备编号的解析、顺序和重复校验；
- 正面轮与背面轮的设备到视角映射；
- 每个曝光档严格执行 `trigger x3 -> read x3`；
- 每台相机使用独立 buffer 和 frame info；
- 六视角共享 session、group 和 sample 标识；
- 任意一路读取失败时 sample 不完整；
- `imwrite` 返回 false 时报告保存失败；
- 部分设备打开或启动失败时仍完整清理资源；
- pipeline wrapper 正确转发参数。

真实硬件 smoke test 使用一个临时输出目录，验收顺序为：

1. 执行 `--list-devices`，确认三台预期海康相机存在。
2. 使用较低组数运行一次正面和背面 HDR 采集。
3. 验证六个融合图均能由 OpenCV 读取，尺寸和通道合理。
4. 启用 `--save-hdr-sources` 时，验证十二张曝光原图存在且可读。
5. 验证 manifest 中六个视角共享同一个 sample，并标记完整。
6. 记录软件触发间隔和端到端耗时；报告时明确区分曝光时间与程序总耗时。

smoke test 只写入临时目录，不污染正式 `dataset/`。硬件测试若发现设备编号与物理视角不符，停止正式采集并要求重新核对映射，不根据画面内容自动猜测。

## 文档和仓库记忆

`pipeline/README.md` 增加单终端六视角采集章节，包括设备映射、HDR 参数、翻面流程、输出结构、完整性语义和软件触发边界。

完成实现与验证后更新仓库根目录的 `AGENTS_MEMORY.md`，记录新入口、六视角命名、默认设备映射、smoke test 命令和真实验证结果。
