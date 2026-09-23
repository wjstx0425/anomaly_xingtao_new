# 曝光扫描与 HDR 融合测试

独立入口 `python -m bmw_inspection.cli.exposure_test`（重新安装后也可用 `bmw-exposure-test`）。
复用 `bmw-collect` 的海康 SDK、相机序列号配置、固定增益、软件触发、曝光切换丢帧和现有融合算法。
无需加载检测模型。采集同一固定姿态的多档曝光，再用同一组原图比较融合效果。

## 开始测试

在仓库根目录执行，使用已经配置好的 uv 环境。新相机主机先按 `bmw_runtime/README.md` 安装运行环境和海康 MVS，
下列命令中的 `uv run --no-sync` 可替换为 `uv run --project bmw_runtime`。

```bash
# 枚举设备，不开始采集
uv run --no-sync python -m bmw_inspection.cli.exposure_test --list-devices

# 默认使用现有配置中的 1500、6000 μs，以及全部四台相机
uv run --no-sync python -m bmw_inspection.cli.exposure_test

# 单台相机，五档曝光，重复三轮：以下只是实验扫描值，可自行调整
uv run --no-sync python -m bmw_inspection.cli.exposure_test \
  --serial DA9805574 \
  --exposures-us 750 1500 3000 6000 12000 \
  --pairs 750:3000 1500:6000 3000:12000 \
  --repeat 3 \
  --output results/bmw_exposure/front_trial_01
```

默认每一轮先等回车。固定好零件、光源和相机后开始，整个扫描期间不要移动。
`--no-prompt` 用于已准备好的自动采集。`--round back` 记录背面语义视图；一次命令只拍一个姿态，
正反面分两次运行并使用不同输出目录。`--serial` 可选择一台或多台在线相机，包括配置外的新序列号；不填则要求配置中的四台都在线。
已配置相机沿用原视图名。新相机按照命令参数位置保存为 `camera_01_front` 等测试视图，并在会话中记录序列号映射，
不自动认定为正式 `front` 相机，也不修改正式采集配置。例如：

```bash
uv run --no-sync python -m bmw_inspection.cli.exposure_test \
  --serial DB1624062 --exposures-us 600 750 --repeat 1
```

曝光单位为 **μs**：1000 μs = 1 ms。曝光列表必须严格递增、无重复、2–12 档。
不填 `--pairs` 时对所有短/长组合各做两种融合；档位越多，融合耗时和磁盘占用越大。
`--gain` 可设置固定增益；默认沿用配置，自动曝光与自动增益由现有适配器关闭。
曝光和增益的合法范围由设备 SDK 检查，不会静默截断为别的值。

## 看哪些结果

每次自动创建 `results/bmw_exposure/时间戳/`，或使用 `--output` 指定的**新目录**；已有目录拒绝覆盖。

- `raw/`：每档曝光的无损 PNG 原图，每台相机、每轮分别保存。
- `fusion/`：每组曝光的 `selective` 与 `mertens` 融合 PNG。
- `index.html`：用浏览器打开，比较短曝光、长曝光与融合效果；点击缩略图打开原尺寸图像。
- `metrics.csv`：Excel 可读的亮度、暗区比例、高亮比例、对比度与拉普拉斯方差。
- `report.json`：统计、图像路径、源图对应关系、SHA256 和运行参数。
- `session.json`：原图清单、请求曝光、曝光回读值、主机收帧完成时间、相机和配置快照、运行状态。

`selective` 完全复用当前生产采集代码的暗区选择性融合，参数来自 `--config` 中的 `hdr`。
`mertens` 使用 OpenCV `createMergeMertens(1,1,1)` 的曝光融合，输出裁剪到 [0,1] 后转 8 位；
不做每张图分别归一化或额外 gamma 调整。两者用于比较显示/检测输入效果，不输出经标定的线性辐射 HDR。
算法说明：[OpenCV MergeMertens 官方文档](https://docs.opencv.org/4.x/d7/dd6/classcv_1_1MergeMertens.html)。

默认沿用配置的 `align=false`。`--align` 使用 OpenCV AlignMTB，对同一对源图对齐一次后交给两种融合；
原始 PNG 保持不变。静态工装优先保持不对齐，以免图像位移影响位置比较。

`--roi X1 Y1 X2 Y2` 可仅统计指定区域，右下边界不包含，缩略图仍为全图。
同一 ROI 应用于本次所有相机；不同相机需要不同 ROI 时分别运行。默认统计全图，背景也会计入。
指标仅供诊断：更亮、对比度或拉普拉斯方差更高不等于更好；噪声也会提高这些数值。
应重点检查目标区域细节、过曝、暗区噪声、光晕和重影，不自动推荐“最优曝光”。
高亮比例采用任一 BGR 通道 ≥250，与原采集器基于灰度的 `fused_clip_pct` 口径不同。

曝光回读是 SDK 的 `ExposureTime` 当前设置，不是帧内 chunk 元数据；不能据此声称已验证每帧实际积分时间。
多相机沿用逐台发送软件触发，不能视为硬件同步。需要现场确认切换丢帧数足够，且光源无频闪。
测试不按过曝比例重拍，保留差曝光结果用于比较。

## 无相机离线重放

```bash
uv run --no-sync python -m bmw_inspection.cli.exposure_test \
  --replay results/bmw_exposure/front_trial_01/session.json \
  --pairs 1500:6000 \
  --output results/bmw_exposure/front_trial_01_replay
```

重放无需 SDK、相机或原配置文件。只接受本程序完成的 `session.json`；每张原图先核对 SHA256。
沿用保存的全部原图和 HDR 参数，可另设 `--pairs`、`--align/--no-align`、`--roi`；
不允许覆盖采集曝光、增益、相机或重复次数。文件与参数不修改原会话。
任意已融合图片不能伪装成不同曝光原图进入此流程。

发生错误或 Ctrl-C 时，已有原图和 `status=failed` 的会话清单保留，相机按现有清理流程关闭。
不自动覆盖、续写或重放失败会话。相机曝光/增益设置不保证恢复运行前数值；退出恢复连续触发模式，
再次启动正常采集时由现有采集器重新设置曝光和增益。

## 验证范围

包含无硬件测试：曝光切换丢帧、原图/回读对应、相机异常清理、离线完整性校验、重复输出拒绝、
现有融合像素一致性、ROI 指标与 HTML 输出。尚未在真实相机上验证成像、曝光生效延迟、光源稳定性，
也未完成 Windows 现场验收。测试通过不等于现场 HDR 效果已确认。
