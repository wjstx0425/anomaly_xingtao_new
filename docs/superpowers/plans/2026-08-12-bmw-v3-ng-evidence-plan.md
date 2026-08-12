# BMW 第三版 NG 证据 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建第三版模型组合，并让每次现场检测可逐条解释、保存和比较全部 NG 证据。

**Architecture:** 新配置和组合目录固定四模块身份；捕获层保留短/长/HDR；结果层以结构化指标和原子目录持久化；UI 通过 NG 队列浏览证据。旧配置和旧结果保持不变。

**Tech Stack:** Python, OpenCV, NumPy, Pillow, anomalib, ultralytics, pytest, uv

## Global Constraints

- Template 必须来自晚九点候选，光痕必须使用 raw-profile v2，YOLO 必须保持上午基线。
- EfficientAD checkpoint 保持第二版，每个部署阈值必须等于第二版基础阈值加 `0.05`。
- 保存短曝光、长曝光、融合 HDR、ROI、模型证据和完整结构化指标。
- YOLO 可称真实检测框；光痕称规则证据；Template/EfficientAD 只能称诊断热区。
- 不修改或覆盖前两版配置、模型和结果。

---

### Task 1: 第三版模型组合与阈值资产

**Files:**
- Create: `pipeline/bmw_lab_prepare_v3_ng_evidence_demo.py`
- Create: `configs/bmw/experiments/bmw_eight_view_demo_v3_ng_evidence.json`
- Test: `tests/unit/bmw_inspection/lab/test_v3_ng_evidence_demo.py`

**Interfaces:**
- Produces: immutable composite `template/efficientad/yolo` links and a SHA-bound EfficientAD threshold artifact with `base_thresholds`, `threshold_margin=0.05`, and deployment `thresholds`.

- [ ] 写入失败测试，验证三条模型链接、阈值逐视角增加 `0.05`、旧配置未变化。
- [ ] 运行测试并确认因准备器和配置缺失而失败。
- [ ] 实现原子无覆盖准备器和第三版配置。
- [ ] 运行定向测试并确认通过。

### Task 2: HDR 源图与检测证据持久化

**Files:**
- Modify: `src/bmw_inspection/lab/eight_view_demo_capture.py`
- Create: `src/bmw_inspection/lab/eight_view_demo_persistence.py`
- Modify: `pipeline/bmw_lab_eight_view_demo.py`
- Test: `tests/unit/bmw_inspection/lab/test_eight_view_demo_capture.py`
- Create: `tests/unit/bmw_inspection/lab/test_eight_view_demo_persistence.py`

**Interfaces:**
- Produces: `FourCameraHdrSession.last_sources`; `persist_inspection(config, inspection, source_images)` atomic result writer.

- [ ] 写失败测试，覆盖八视角短/长/HDR映射、图像统计、无覆盖原子发布和索引。
- [ ] 运行测试并确认失败原因正确。
- [ ] 实现缓存与持久化，并在离线/现场检测完成后调用。
- [ ] 运行定向测试并确认通过。

### Task 3: 结构化 NG 证据与浏览界面

**Files:**
- Modify: `src/bmw_inspection/lab/eight_view_demo.py`
- Modify: `src/bmw_inspection/lab/eight_view_demo_models.py`
- Modify: `src/bmw_inspection/lab/eight_view_demo_ui.py`
- Modify: `pipeline/bmw_lab_eight_view_demo.py`
- Test: `tests/unit/bmw_inspection/lab/test_eight_view_demo_models.py`
- Test: `tests/unit/bmw_inspection/lab/test_eight_view_demo_ui.py`

**Interfaces:**
- Produces: immutable structured result details; `actionable_results()` and `step_actionable_selection()`; `N/P` navigation and four-panel evidence comparison.

- [ ] 写失败测试，覆盖完整原因、阈值余量、证据类型、NG顺序和N/P循环。
- [ ] 运行测试并确认因新契约缺失而失败。
- [ ] 为四模块生成结构化指标，修正 Template 最佳平移后的差分图，光痕直接放大 ROI，并生成 EfficientAD 热点诊断。
- [ ] 实现 NG 队列与短/长/HDR/证据四图对比。
- [ ] 运行模型和 UI 定向测试并确认通过。

### Task 4: 集成验证与交付

**Files:**
- Modify: `AGENTS_MEMORY.md`
- Modify: `pipeline/AGENTS_MEMORY.md`

- [ ] 生成第三版组合资产并校验全部 SHA。
- [ ] 运行 BMW Demo、模型、捕获、持久化和 UI 相关测试。
- [ ] 用离线八视图完成 25 项推理并检查保存目录和截图。
- [ ] 更新项目记忆并提交可复现代码。
