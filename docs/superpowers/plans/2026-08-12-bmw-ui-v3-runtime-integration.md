# BMW UI 与光痕 V3 运行时整合实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将已开发的可点击证据 UI 稳定接入光痕 V3 Demo，并完成离线与四相机启动验证。

**Architecture:** 保留现有 UI 状态、命中测试和详情页，不复制或重写界面。只在 OpenCV 主循环入口保证 Qt 窗口先通过首帧 `imshow` 和一次事件处理完成原生创建，再注册鼠标回调；光痕 V3、其他三类算法和 25 项融合结果保持不变。

**Tech Stack:** Python 3.13、OpenCV Qt HighGUI、NumPy、pytest、uv。

## Global Constraints

- 复用提交 `31da1aab`、`8ae921ee`、`548c2439`、`bda9c434`、`ee4ece91` 的 UI，不重新设计交互。
- 使用 `configs/bmw/experiments/bmw_eight_view_demo_v3_ng_evidence.json` 与光痕 V3 正式报告。
- 不纳入当前未提交的 EfficientAD manual-ignore 代码或 V4 配置。
- 不改变 Template、光痕、YOLO、EfficientAD 判定、阈值、25 项结果或融合。
- 保留主页面点击、详情页、Esc 返回、Q 退出、R 重置及原快捷键。

---

### Task 1: 修复 Qt 窗口与鼠标回调初始化顺序

**Files:**
- Modify: `pipeline/bmw_lab_eight_view_demo.py`
- Test: `tests/unit/pipeline/test_bmw_lab_eight_view_demo.py`

**Interfaces:**
- Consumes: `render_eight_view_screen(state) -> np.ndarray`
- Produces: `_initialize_gui_window(title: str, state: EightViewUiState, releases: list[tuple[int, int]]) -> None`

- [ ] 新增失败测试，记录 `namedWindow -> resizeWindow -> imshow -> waitKey -> setMouseCallback` 的精确调用顺序，并断言回调携带同一事件队列。
- [ ] 运行该测试，确认旧实现因为在 `imshow` 前注册回调而失败。
- [ ] 提取最小窗口初始化函数：先显示初始 1600x900 画面，调用 `waitKey(1)` 完成 Qt 原生窗口创建，再注册鼠标回调。
- [ ] 重新运行 pipeline UI 测试及 renderer UI 测试，确认点击、键盘和页面状态不变。
- [ ] 仅提交入口、入口测试和已批准的 UI spec/plan，不纳入算法或 EfficientAD 文件。

### Task 2: 整合验收与现场启动

**Files:**
- Modify: `AGENTS_MEMORY.md`
- Modify: `pipeline/AGENTS_MEMORY.md`

**Interfaces:**
- Consumes: 光痕 V3 配置、保存记录 `bmw_demo_20260812_211302`、四相机 `FourCameraHdrSession`。
- Produces: 可运行的可点击 V3 Demo 进程和验收记录。

- [ ] 在干净提交快照运行 UI、光痕 V3、配置、持久化、捕获与入口测试。
- [ ] 离线运行 `211302`，要求光痕 PASS、25 项齐全、24 个非光痕结果不变。
- [ ] 启动 V3 GUI，要求进程持续运行、模型全部加载、窗口回调成功、四相机成功打开。
- [ ] 若未实际拍摄，不声称现场检测周期或新记录成功；只报告已验证到的层级。
- [ ] 更新两份项目记忆，记录整合提交、测试、启动结果和剩余硬件边界。
