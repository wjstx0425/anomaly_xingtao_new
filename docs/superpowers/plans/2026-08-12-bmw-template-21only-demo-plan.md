# BMW Template 21点单变量Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建可实际联动四相机运行、且只替换21点Template模型的独立BMW Demo入口。

**Architecture:** 用一个no-overwrite准备器生成组合run的三个目录链接；新增Demo JSON指向该组合run，其余算法资产继续锁定上午版本。复用现有Demo加载、模型校验、相机采集和UI代码。

**Tech Stack:** Python 3.13、uv、pathlib、pytest、现有BMW OpenCV/Anomalib Demo。

## Global Constraints

- 不修改 `configs/bmw/experiments/bmw_eight_view_demo_v1.json`。
- 只替换Template模型；EfficientAD、YOLO、光痕、ROI与采集配置保持上午版本。
- 组合run和测试结果使用新目录，拒绝覆盖。
- 不训练任何模型。

---

### Task 1: 独立组合run准备器与配置

**Files:**
- Create: `pipeline/bmw_lab_prepare_template_21only_demo.py`
- Create: `configs/bmw/experiments/bmw_eight_view_demo_template_21only_v1.json`
- Create: `tests/unit/bmw_inspection/lab/test_template_21only_demo.py`
- Modify: `AGENTS_MEMORY.md`

**Interfaces:**
- Consumes: 21点Template候选目录与上午基线run。
- Produces: `prepare_template_only_run(candidate_template_root: Path, baseline_run: Path, output_run: Path) -> dict[str, object]`，以及可由 `load_demo_config` 读取的新配置。

- [ ] 写失败测试：验证组合run只替换Template、拒绝覆盖、新配置保持其他资产不变。
- [ ] 运行定向pytest，确认因准备器缺失而RED。
- [ ] 实现原子no-overwrite目录链接准备器和CLI。
- [ ] 新增独立Demo配置，result_root与原版隔离。
- [ ] 运行定向pytest至GREEN。
- [ ] 生成真实组合run并调用 `load_demo_config` 验证模型身份。
- [ ] 运行离线正常样本smoke，再启动相机GUI。
- [ ] 更新项目记忆并提交代码。
