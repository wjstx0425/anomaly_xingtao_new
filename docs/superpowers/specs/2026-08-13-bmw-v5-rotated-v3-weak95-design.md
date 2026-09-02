# BMW V5 倾斜 ROI 光痕 V3 弱响应重标定设计

## 目标

只在 V5 倾斜 ROI 光痕分支中，把部署弱响应阈值从报告原值 `128.2` 覆盖为 `95.0`，修复 `bmw_demo_20260813_184658` 和 `bmw_demo_20260813_184751` 的假断续。HDR 输入、倾斜 ROI、强响应阈值和最终覆盖率/连续段/断口规则保持不变；Template、YOLO、EfficientAD 不变。

## 方案

保留 SHA 绑定的原始 V3 报告，在 V5 `bright_streak` 配置中新增唯一可选字段 `weak_row_score_override`。加载器只允许它用于 `tracked_profile_v3_manual_rotated_roi`，并校验为有限非负数且不高于报告强阈值；模型构建时把它传给 V3 predictor。运行证据同时记录报告原值、部署值及 `rotated_hdr_field_recalibration_v1`，避免把配置覆盖误认为重新训练。

## 验收

- `184658`、`184751` 均为 `OK`。
- 33 张旧正常图在当前倾斜 ROI 下全部为 `OK`。
- 8 张完全无光痕图仍为 `NG_NO_STREAK`。
- 已确认断续样本 `164043` 仍为 `NG_BROKEN`。
- 其余五项 V3 阈值保持报告原值。
