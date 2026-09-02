# BMW 正面采集后提前检测设计

## 目标

正面四视角 HDR 采集完成后立即在唯一后台模型线程运行正面 13 项检测，同时主线程继续提示翻面并允许采集背面，以人工翻面和背面采集时间覆盖正面推理耗时。

## 约束

- 不改变 Template、光痕、YOLO、EfficientAD 的计算、模型、阈值、ROI、mask 或最终融合。
- 相机生命周期、正反面 `capture_round()` 和全部 OpenCV HighGUI 操作留在主线程。
- 所有模型调用只在一个 `ThreadPoolExecutor(max_workers=1)` 中串行执行，正反面模型不并发。
- 正面阶段不构造最终 OK/NG、不匹配可信 OK、不持久化。
- 最终结果顺序保持 `8 Template → 1 光痕 → 8 YOLO → 8 EfficientAD`。
- 离线/no-GUI 的完整八视图入口保持兼容。

## 数据流

1. 主线程采集四张正面 HDR，将图像副本提交给后台线程。
2. 后台运行正面 4 Template、1 光痕、4 YOLO、4 EfficientAD，共 13 项。
3. UI 立即显示翻面提示；正面检测未完成也允许按空格采集背面。
4. 背面采集完成后，把背面 4 Template、4 YOLO、4 EfficientAD 共 12 项排入同一个后台线程。
5. 两阶段完成后验证 25 个唯一 `(branch, view)` 键，按旧顺序重排，执行融合、可信 OK 匹配并构造完整 `EightViewInspection`。
6. 完整结果只持久化一次，随后进入 RESULT；编排/保存异常进入 UI ERROR。

## 状态和取消

- `WAITING_FLIP` 允许一次背面采集；`PROCESSING` 的空格被忽略。
- `R` 在后台任务运行时标记当前检测作废；旧结果完成后丢弃，不与新零件组合。
- `Q` 不强杀正在执行的 Torch/YOLO 任务，等待唯一 worker 安全结束后释放相机和窗口。
- 分支推理异常仍由现有 `_call()` 转为结果行 ERROR，并继续完成其余分支。

## 验证

- 正面精确 13 项、背面精确 12 项，光痕只接收完整 `front_left`。
- 同视角 Template/YOLO/EfficientAD 共用一次 ROI crop。
- 最终 25 项顺序、融合、可信 OK 和同步入口保持一致。
- 正面 future 未结束时仍可触发背面采集；模型调用仍在同一 worker 串行。
- 持久化仅在最终完成后调用一次；陈旧 future 不改变当前 UI 状态。

