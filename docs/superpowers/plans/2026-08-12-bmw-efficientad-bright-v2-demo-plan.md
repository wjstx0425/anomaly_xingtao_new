# BMW EfficientAD 与光痕 v2 Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 启动只替换 21:00 EfficientAD 和原灰度光痕 v2 的第二个四相机实验 Demo。

**Architecture:** 保留既有八视图运行器，增加一种经 SHA 校验的光痕资产类型，并创建不覆盖现有模型的组合运行目录。默认配置保持不变。

**Tech Stack:** Python, OpenCV, anomalib, ultralytics, pytest, uv

## Global Constraints

- Template、YOLO、HDR 和零件 ROI 使用上午基线。
- EfficientAD checkpoint 与逐视角阈值必须匹配。
- 光痕 v2 只处理 `front_left` 全图的固定 ROI。
- 所有新增输出不覆盖既有目录。

---

### Task 1: 接入原灰度光痕 v2

**Files:**
- Modify: `src/bmw_inspection/lab/eight_view_demo.py`
- Modify: `src/bmw_inspection/lab/eight_view_demo_models.py`
- Test: `tests/unit/bmw_inspection/lab/test_eight_view_demo.py`
- Test: `tests/unit/bmw_inspection/lab/test_eight_view_demo_models.py`

- [ ] 先写配置 SHA 校验、ROI 读取和预测状态/证据图的失败测试。
- [ ] 运行定向测试并确认因接口缺失而失败。
- [ ] 实现 `raw_profile_v2` 配置解析和预测器。
- [ ] 运行定向测试并确认通过。

### Task 2: 创建第二版组合运行并启动

**Files:**
- Create: `pipeline/bmw_lab_prepare_efficientad_bright_v2_demo.py`
- Create: `configs/bmw/experiments/bmw_eight_view_demo_efficientad_bright_v2_v1.json`
- Test: `tests/unit/bmw_inspection/lab/test_efficientad_bright_v2_demo.py`
- Modify: `AGENTS_MEMORY.md`
- Modify: `pipeline/AGENTS_MEMORY.md`

- [ ] 先写组合目录分支身份和候选配置差异测试。
- [ ] 运行测试并确认因准备器/配置缺失而失败。
- [ ] 实现无覆盖组合目录与独立配置。
- [ ] 运行相关单测与离线 25 项推理。
- [ ] 启动四相机实验界面。
