# BMW 21点批次诊断实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development to implement this plan task-by-task.

**Goal:** 在不覆盖上午部署资产的前提下，产生Template单变量候选、光痕v2离线候选和21点EfficientAD阈值候选。

**Architecture:** 三个候选相互独立；共同读取21点发布数据和固定ROI，分别写入新结果目录。默认Demo配置不改变。

**Tech Stack:** Python 3.11+、uv、OpenCV、Anomalib EfficientAD、pytest。

## Global Constraints

- 只使用session `20260810_210030_527506`、`20260810_213407_600611`、`20260810_213852_756399`、`20260810_214505_826120`、`20260810_214747_085800`。
- 不修改或覆盖 `results/bmw_lab_one_click/bmw_right_multisource_left_yolo_v1`。
- 不修改 `configs/bmw/experiments/bmw_eight_view_demo_v1.json`。
- YOLO不训练、不替换。
- 每个产物记录输入清单、ROI、基线模型和代码身份。

---

### Task 1: 上午部署资产复现凭据

**Files:**
- Create: `pipeline/bmw_lab_snapshot_reproducibility.py`
- Test: `tests/unit/bmw_inspection/lab/test_reproducibility_snapshot.py`

**Interfaces:**
- Consumes: 上午run、组合release、Demo配置和ROI配置路径。
- Produces: `snapshot.json`，包含相对路径、大小、SHA256和代码提交。

- [ ] 先写测试，验证缺失必需资产时失败、输出不包含dataset图像内容、哈希稳定。
- [ ] 运行测试并观察功能缺失导致的RED。
- [ ] 实现只读快照CLI，拒绝覆盖既有输出。
- [ ] 运行测试至GREEN，并对上午资产生成快照。

### Task 2: 21点Template单变量候选

**Files:**
- Create: `pipeline/bmw_lab_train_template_fixed_thresholds.py`
- Modify: `src/bmw_inspection/lab/template.py`
- Test: `tests/unit/bmw_inspection/lab/test_template_fixed_thresholds.py`

**Interfaces:**
- Consumes: 21点 `template/trainer_manifest.csv`、上午8个`model.json`中的阈值。
- Produces: 新的8视角Template模型、normal-only calibration/final_test评分和整件通过率报告。

- [ ] 写测试：只允许train/normal选择模板、阈值逐视角复用、来源session必须属于白名单。
- [ ] 运行测试确认RED。
- [ ] 实现固定阈值训练与正常件评分，保持512、shift12、5模板。
- [ ] 运行测试至GREEN。
- [ ] 在21点release上训练并生成报告。

### Task 3: 光痕v2原灰度逐行局部对比

**Files:**
- Create: `src/bmw_inspection/lab/bright_streak_raw_profile.py`
- Create: `pipeline/bmw_lab_evaluate_bright_streak_raw_profile.py`
- Test: `tests/unit/bmw_inspection/lab/test_bright_streak_raw_profile.py`

**Interfaces:**
- Consumes: 81x613灰度ROI、21点bright_streak.csv、现有光痕配置。
- Produces: row-score、mask、coverage、longest-run、gap指标及calibration/final_test对比报告。

- [ ] 写合成连续、断续、无光痕测试并确认RED。
- [ ] 实现逐行中心减左右背景的局部对比与连续性统计。
- [ ] 运行单元测试至GREEN。
- [ ] 只用calibration拟合阈值，在final_test比较当前算法与v2。

### Task 4: 21点EfficientAD训练与整件阈值

**Files:**
- Create: `pipeline/bmw_lab_train_efficientad_only.py`
- Modify: `src/bmw_inspection/lab/eight_view_train_all.py`
- Test: `tests/unit/bmw_inspection/lab/test_efficientad_only.py`

**Interfaces:**
- Consumes: 21点release的8个EfficientAD normal目录和Imagenette。
- Produces: 8个checkpoint、21个校准分支正常件分数、整件最多1/21误拒的阈值资产。

- [ ] 写测试，证明stage选择不会进入materialize/Template/光痕/YOLO。
- [ ] 运行测试确认RED。
- [ ] 实现EfficientAD-only入口并复用现有训练函数。
- [ ] 运行测试至GREEN。
- [ ] 训练8视图、评分并选择整件阈值。

### Task 5: 汇总与候选边界

**Files:**
- Create: `results/bmw_lab_one_click/bmw_right_batch_20260810_21_diagnostics_summary/report.json`
- Create: `results/bmw_lab_one_click/bmw_right_batch_20260810_21_diagnostics_summary/summary.md`
- Modify: `AGENTS_MEMORY.md`

**Interfaces:**
- Consumes: Tasks 1-4的报告。
- Produces: 明确列出Template变化、光痕A/B结果、EfficientAD正常误拒和未验证缺陷边界。

- [ ] 校验所有输入路径和SHA256。
- [ ] 生成JSON和中文Markdown汇总。
- [ ] 运行相关单元测试、`py_compile`和`git diff --check`。
- [ ] 不修改默认Demo配置，提交候选代码和复现实验身份。
