# ZS32 Demo 全链路计时与无损提速报告（2026-07-17）

## 范围与契约

- 工作 bundle 保持为 `results/zs32_runtime_bundle_eight_view_20group_patchcore_tuned_v10/runtime_bundle.json`，未覆盖或修改 v10。
- 六个主视角运行 Template、PatchCore、YOLO；`front_secondary/back_secondary` 的 Template/PatchCore 仍为 `SKIPPED`；YOLO 仍为一次八图 batch。
- Stage18 的 CSV、JSON、audit 和 `runtime_manifest.json` 布局不变。
- 同图基准输入为 `results/zs32_stage35_live_runtime_v1/20260716_092656_638662_live_part_001/live_part_001_group001_000001/sources`。

## 基线与瓶颈

未修改代码的完整冷启动外部墙钟为 27.89 s。加入只读计时后，同图 Stage32+Stage18 内部总耗时为 24.260 s，外部墙钟为 25.49 s；结果均为 `OK`、`inspection_complete=true`。

冷启动前三大瓶颈：

1. 六个 PatchCore 模型加载：10.801 s。
2. Template 六主视角加载与推理：2.412 s。
3. YOLO 八图 batch：1.457 s；ROI/图片写盘另为 1.456 s。

## 已实施的无损优化

- Dashboard 启动独立常驻 inference worker；worker 一次加载六个主视角 PatchCore、一个 YOLO 和六组主视角 Template。Stage35 默认仍保留原 Stage32 subprocess 回退路径。
- worker 使用私有 Unix socket、单任务队列、job ID、超时、STOP；Dashboard 退出时先正常停止，超时再终止进程组。worker 异常退出后，下一次检测会重新创建。
- Stage32 复用同一组已加载 backend，并保持每次输出目录独占和 Stage18 原子发布。
- Template 的 model JSON、模板 SHA 和模板图在 worker 启动时只加载一次。
- 八张原图只解码一次，Template 和 ROI 生成复用同一内存图像。
- 四相机保持“先全部触发”，之后按独立 camera handle 并行阻塞取帧；结果仍按 topology 顺序组装。
- 采集、推理、Stage18、Dashboard 解析/首帧均写入结果目录的 `timing.json`；Dashboard 的 `total_seconds` 是点击开始到首帧显示（headless 时到首帧渲染）。

worker 启动预加载为 10.888 s，可在 Dashboard 展示、操作员放件和确认前台/后台期间并行发生。worker 已就绪后，同图完整 Stage32+Stage18 为 11.765 s，相对 24.260 s 减少 51.5%。早期 worker 两次连续运行分别为 12.578 s 和 12.803 s；加入 Template 缓存与共享解码后为 11.765 s。

Dashboard 对该结果的真实 headless 解析为 1.483 s、首帧渲染为 1.585 s；GUI `imshow` 返回计时已由代码和测试覆盖，但本次无显示服务器，因此未伪造真实 GUI 显示耗时。

## 第二轮无损 I/O 优化

针对 prepared worker 中原有 3.798 s 的 ROI/编码与 evidence 写盘关键路径，继续实施：

- YOLO 八视角 ROI 直接以 canonical 顺序的连续 BGR ndarray batch 输入，不再生成或回读八张临时 PNG；
- PatchCore 仍保留 Anomalib `Engine.predict(data_path=...)` 必需的六张主视角 PNG，secondary 不生成；
- PatchCore 每个视角完成推理后立即把 raw NPY、mask PNG、overlay PNG 提交给有界双线程 writer，CPU 写盘与后续视角 GPU 推理重叠；
- YOLO 八张 `result.plot()` 与 PNG 写盘使用有界双线程并行，返回前显式等待所有 future；
- PatchCore overlay 复用已读入的 crop ndarray，不再二次 `cv2.imread`；
- `sources/` 仍使用独立 `copy2` 归档，避免原 capture 文件后续变化破坏已发布结果；每张归档原图 SHA 每轮只计算一次并复用于 manifest、PatchCore CSV 和 YOLO CSV。

最终权威基准为 `results/zs32_timing_worker_io_v3/run_01..03`。三轮完整 Stage32+Stage18 均为
`OK`、`inspection_complete=true`，总耗时分别为 8.171 / 8.155 / 8.061 s，中位数 8.155 s；
相对 worker v2 的 11.765 s 再减少 30.7%。原 3.798 s 关键路径降为：

| I/O部分 | worker v2 | io v3中位数 | 变化 |
|---|---:|---:|---:|
| source归档、六张PatchCore crop、ROI准备 | 1.454 s | 0.710 s | -51.2% |
| PatchCore evidence 串行关键路径 / 并发残余等待 | 1.556 s | 0.136 s | -91.3% |
| YOLO evidence PNG | 0.788 s | 0.430 s | -45.4% |
| 合计关键路径 | 3.798 s | 1.276 s | -66.4% |

并行后的逐视角 PatchCore writer `seconds` 是多个线程的 CPU 工作量之和，不能再作为墙钟占比直接相加；
关键路径应读取 `evidence_write/patchcore_wait`。`zs32_timing_worker_io_v1` 因手工 benchmark
误用默认 22 组 profile 而 fail-closed，`io_v2` 只运行 infer，不作为验收结果；`io_v3` 才是正确
20 组 `fuse` 基准。

旧 worker v2 与 io v3 三轮逐字段比较均通过：runtime manifest 语义、三类 CSV、20 组 Stage18、
六视角 PatchCore raw NPY、mask/overlay PNG、八视角 YOLO evidence PNG 以及声明的 source/evidence hash
均一致。

## 同图结果一致性

历史 v10、计时冷启动、worker 第一次、worker 第二次和最终 worker v2 的以下投影逐字段相同：

- 最终状态和 `inspection_complete`；
- 六条 PatchCore 分数、low/high threshold、pred label；
- 八条 YOLO 分数、检测框和置信度；
- Template 状态、分数和 evidence；
- 20 组 Stage18 分支身份和决策；
- 两个 secondary 的 Template/PatchCore `SKIPPED`。

## 八图并行解码与 Template 六视角并行（2026-07-19）

Stage32 继续只解码每张原图一次，但八个相互独立的 `cv2.imread` 改为有界四线程并行，结果和首个错误仍按
canonical 视角顺序收集。Template 仍先完整生成八张 crop，随后仅把六个主视角提交到六线程池；20/22 组的
`front_secondary/back_secondary` 没有进入 Template 推理，结果仍在原位置标记 `SKIPPED`。

同一组保存图像、同一 v10 bundle、常驻 worker 预热后五轮正式结果位于
`results/zs32_timing_worker_parallel_v1/run_01..05`：

| 指标 | io v3 中位数 | 并行版中位数（范围） | 变化 |
|---|---:|---:|---:|
| Stage32 + Stage18 总耗时 | 8.155 s | 7.006 s（6.826-7.296） | -14.1% |
| 八图共享解码 | 0.896 s | 0.299 s（0.280-0.329） | -66.6% |
| Template 整体阶段 | 1.339 s | 0.906 s（0.897-0.951） | -32.3% |
| Template 六视角 evaluate 墙钟 | 原流程串行 | 0.123 s（0.121-0.131） | 新增分项 |

五轮均为 `OK`、`inspection_complete=true`。并行版 run_02 与 io-v3 run_02 的三类模型 CSV 和 Stage18
branch CSV 在去除样本身份/输出绝对路径后逐字段相同；全部 source/crop/evidence PNG 字节完全相同，六个
PatchCore raw NPY 逐数组相同。此次没有修改或覆盖 v10 bundle、模型或 threshold artifact。

## 四相机 HDR 融合与 PNG 编码并行（2026-07-21）

相机曝光、trigger/read 完成后，四个互相独立的内存图像进入有界四线程CPU池：HDR融合和clip统计为一组，
PNG编码为另一组。future仍按topology顺序读取；并行完成顺序不会改变CaptureFrame顺序或首个错误，PNG失败
只保留失败视角之前的canonical成功前缀。worker不写`TimingRecorder`，CaptureFrame及`captured_at`仍由
主线程构造。

并行后，`hdr_fusion`和`image_encoding`表示重叠的逐任务CPU时间之和；关键路径必须读取新增的
`hdr_fusion_parallel_wall`和`image_encoding_parallel_wall`。

使用最近现场保存的四张4024x3036前侧图像进行离线真实尺寸微基准：

| 阶段 | 串行 | 四路并行 | 加速 |
|---|---:|---:|---:|
| 四张HDR融合 | 1.742 s | 0.683 s | 2.55x |
| 四张PNG编码 | 2.290 s | 0.978 s | 2.34x |

并行与串行的融合图逐像素一致，PNG逐字节一致；进程峰值RSS约2.88 GiB，本机总内存62 GiB。该结果不包含
相机SDK和操作员翻面/确认，不能替代一次真实前后两轮硬件benchmark。

最终直接相关的adapter、bootstrap、capture service、Stage35 live和collector测试为253 passed；目标文件
`py_compile`与`git diff --check`通过，独立复审无剩余Critical/Important问题。

## 可能影响精度的独立测试

这些选项均未写回 runtime config 或 bundle。

| 选项 | 同图速度 | 同图结果差异 | 结论 |
|---|---:|---|---|
| YOLO FP32, 1280 | 0.831 s | 基线 | 保持 |
| YOLO FP16, 1280 | 0.792 s（快 4.7%） | 该组框/置信度完全相同；Ultralytics 已提示 `half` 弃用 | 不启用 |
| YOLO FP32, 960 | 0.769 s（快 7.5%） | 新增框，最大置信度差 0.00309 | 不启用 |
| YOLO FP32, 640 | 0.746 s（快 10.2%） | 新增框，最大置信度差 0.01295 | 不启用 |
| YOLO TensorRT | 未完成 | 本机缺少 `onnx/onnxruntime/onnxslim/tensorrt`；`/tmp` 导出在 19.47 s 后失败 | 未安装依赖，不启用 |
| PatchCore autocast FP16 | 六视角 3.086 s，对比 FP32 3.141 s（快 1.8%） | 最大分数差 0.00620，map 最大差 0.00870 | 不启用 |
| Template width 384/256 | 0.544/0.536 s，对比 0.554 s | 状态相同，但最大 risk 差 0.00161/0.00381 | 不启用 |
| Template 3/1 张 | 0.533/0.515 s | `back_right` 从 PASS 变 NG_TEMPLATE | 禁止替换 |
| Template max_shift 8/4 | 0.551/0.548 s | 该组状态相同，risk 最大差约 2.98e-7 | 收益过小，不启用 |

`capture-interval 0.5/0.2/0.1`、`timeout 5000/2000` 和 `hdr-settle-frames 1/0` 会改变真实取像时序或图像内容，必须由操作员放置同一工件后做相机 A/B。当前代码已经为 short/long、HDR fusion、front/back 和编码/写盘提供真实计时；本次离线同图 benchmark 不对这些三项给出伪造的速度或精度结论。

## 启动命令

启动命令保持不变，Dashboard 会自动创建并管理 worker：

```bash
PYTHONPATH=/opt/MVS/Samples/64/Python/MvImport:${PYTHONPATH:-} \
UV_CACHE_DIR=/tmp/uv-cache \
uv run --no-sync python pipeline/36_zs32_inspection_dashboard.py \
  --live \
  --part-id live_part_001 \
  --runtime-config results/zs32_runtime_bundle_eight_view_20group_patchcore_tuned_v10/runtime_bundle.json \
  --capture-root results/zs32_stage35_live_capture_v1 \
  --output-root results/zs32_stage35_live_runtime_v1
```

worker 日志和 startup timing 位于本次 Dashboard 临时 work dir；每次检测的完整 `timing.json` 位于该次结果目录。

## 验证证据

- 本次直接相关的 pipeline/model/runtime/live/template/timing：354 passed。
- 本次直接相关的 Dashboard parser/app/render/control/live：135 passed。
- capture bootstrap/Hikvision：28 passed。
- Unix socket worker：5 passed。
- 第二轮 I/O 优化定向 model runtime + Stage32：103 passed（包含在上述354项中）。
- 目标 Python 文件 `py_compile` 通过。
- `git diff --check` 通过。
- 最终 Dashboard headless 截图：`artifacts/zs32_timing_worker_v2_dashboard.png`，OpenCV 解码为 `920 x 1600 x 3`。

额外扩大测试范围时得到 351 passed、9 failed：2 个失败来自旧 runtime bundle CLI 子进程缺少
`PYTHONPATH`，7 个来自旧 strict-fusion 测试夹具的 ROI id 与当前 contract 不一致。Dashboard 扩展组另有
178 passed、1 failed，失败项是未修改的 compositor 旧测试对 unsupported secondary 原图做逐像素相等
断言。本次没有顺手修改这些与计时/常驻 worker 无关的既有问题。
