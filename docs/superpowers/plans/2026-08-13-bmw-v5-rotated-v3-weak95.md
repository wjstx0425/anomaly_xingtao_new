# BMW V5 Rotated V3 Weak-95 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变其他算法和 V3 最终连续性规则的前提下，为 V5 倾斜 HDR ROI 部署 `weak_row_score=95.0`。

**Architecture:** 原始 SHA 绑定 V3 报告继续作为参数来源；V5 配置只覆盖弱响应阈值。配置加载、模型接线和证据元数据都显式保留报告值与部署值。

**Tech Stack:** Python、OpenCV、pytest、uv。

## Global Constraints

- 只允许 `tracked_profile_v3_manual_rotated_roi` 使用弱阈值覆盖。
- 原始报告、ROI、HDR 输入和其余五项阈值不变。
- 不修改 Template、YOLO、EfficientAD 或融合规则。

---

### Task 1: 配置与运行时覆盖

**Files:**
- Modify: `src/bmw_inspection/lab/eight_view_demo.py`
- Modify: `src/bmw_inspection/lab/eight_view_demo_models.py`
- Modify: `configs/bmw/experiments/bmw_eight_view_demo_v5_template_manual_ignore_mask_v1.json`
- Test: `tests/unit/bmw_inspection/lab/test_eight_view_demo.py`
- Test: `tests/unit/bmw_inspection/lab/test_eight_view_demo_models.py`

**Interfaces:**
- Consumes: 原始 V3 report、倾斜 ROI 和可选 `weak_row_score_override: float`。
- Produces: 使用部署弱阈值的 `EightViewTrackedProfileBrightStreakPredictor`，并在 details 中记录原始值和部署值。

- [ ] **Step 1:** 先写配置解析、suite 接线和 predictor 证据的失败测试。
- [ ] **Step 2:** 运行聚焦测试，确认因缺少 override 支持而失败。
- [ ] **Step 3:** 最小实现可选 override、严格校验和证据元数据；V5 写入 `95.0`。
- [ ] **Step 4:** 运行聚焦单元测试，确认通过。
- [ ] **Step 5:** 重放两件正常、8 件无光痕、33 件旧正常和 `164043` 断续样本，核对验收矩阵。
- [ ] **Step 6:** 更新 `AGENTS_MEMORY.md`，精确提交本任务文件。
