# BMW 八视图完整检测周期计时设计

## 目标

在不改变相机参数、四个检测模块、阈值、ROI、mask或融合规则的前提下，准确记录实时 GUI 中“第一次按下空格”到“结果画面提交给 OpenCV 窗口”的完整软件周期时间，并保存可追溯的阶段明细。

## 时间边界

- 起点：GUI 在 `IDLE`/`RESULT`/`ERROR` 状态接受第一次空格键，即将调用正面 `capture_round()` 之前。
- 终点：主线程已取得完整检测和保存结果，并调用 `cv2.imshow()` 提交第一帧 `RESULT` 画面之后。
- 时钟：所有耗时使用 `time.perf_counter_ns()`，不受系统时间调整影响。另记录起止本地墙钟 ISO 时间，只用于人工定位。
- “显示完成”指软件已将画面交给 HighGUI；不声称测量了显示器扫描、像素响应或人眼感知时间。

## 记录的阶段

1. `front_capture_ms`：正面四相机 HDR 采集与融合。
2. `flip_wait_ms`：正面采集返回后，到第二次空格键被接受。
3. `back_capture_ms`：反面四相机 HDR 采集与融合。
4. `front_inference_ms`：现有 `RoundInspectionResult.elapsed_ms`。
5. `back_inference_ms`：现有 `RoundInspectionResult.elapsed_ms`。
6. `finalize_ms`：结果重排、四模块融合、可信 OK 匹配及最终对象构建。
7. `persist_ms`：保存 short/long/HDR、ROI、证据、可信 OK、`inspection.json` 和索引。
8. `result_display_ms`：保存完成到结果画面提交给 `cv2.imshow()`。
9. `total_cycle_ms`：第一次空格键接受到结果画面提交。

由于正面推理与人工翻面/反面采集并行，阶段耗时之和可能大于 `total_cycle_ms`。文件同时保存 `front_overlap_ms`，表示正面推理与翻面等待/反面采集的重叠时间，避免用户误把所有阶段相加。

## 数据流与文件

- 在 `pipeline/bmw_lab_eight_view_demo.py` 内维护一个单次作业计时状态，它与现有 `generation` 绑定，防止 Reset 后把旧任务时间显示到新零件。
- worker 返回检测结果、正/反面推理时间、finalize 时间和 persist 时间；不让 worker 访问 HighGUI。
- 结果帧第一次 `imshow()` 后，在当次 capture 目录写入 `cycle_timing.json`。
- `inspection.json` 保持原有结构，避免破坏已有回放、对比和分析脚本。
- `cycle_timing.json` 只包含 capture ID、起止墙钟、上述毫秒字段和边界说明；不包含 SHA、receipt、schema 版本或发布身份。

## UI 呈现

- RESULT 页面顶部显示 `完整周期 X.XXX s`。
- 同一行显示简短拆分：`采集 F/B、翻面、推理 F/B、融合对比、保存`。
- ERROR 发生在结果显示前时，仍在终端显示已完成的阶段；不伪造一个完整 `total_cycle_ms`。

## 兼容性与错误处理

- 仅实时 GUI 路径生成完整周期时间。离线 `--no-gui` 和 GUI 回放继续使用现有 `inspection.elapsed_ms`，不写虚假的采集/翻面数值。
- `cycle_timing.json` 写入失败时，界面明确进入 ERROR，因为用户要求每次可信的完整计时记录。
- Reset 丢弃旧 generation 的显示与 timing 文件；已开始的模型 worker 按现有行为完成，但不得污染新周期。

## 测试范围

- 用可控单调时钟测试每个阶段的起止点与毫秒值。
- 验证正面推理与翻面/反面采集重叠时，`total_cycle_ms` 不是各阶段简单相加。
- 验证 RESULT 第一帧后写入的 `cycle_timing.json` 数值与 UI 一致。
- 验证 Reset/generation 不会写入陈旧 timing。
- 验证旧离线检测、25 项结果、融合和现有保存路径不变。

## 验收标准

- 现场完成一次新检测后，界面显示从第一次空格到 RESULT 帧的完整时间。
- 同一 capture 目录存在 `cycle_timing.json`，包含所有可分阶段和并行重叠时间。
- 时间单调、非负，且 `total_cycle_ms` 等于定义的起止差。
- Template、光痕、YOLO、EfficientAD 的分数、阈值、状态和最终融合与改动前一致。
