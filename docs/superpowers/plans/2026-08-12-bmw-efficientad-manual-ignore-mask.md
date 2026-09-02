# BMW EfficientAD 手动忽略区域最小实施计划

目标：操作员逐视角选择是否需要 mask，并可绘制多个多边形；多边形区域只从 EfficientAD 异常图评分与热点中排除。Template、YOLO、光痕、公共矩形 ROI 和 EfficientAD 输入图保持不变。

1. 为多边形栅格化、空 mask、资产 SHA/尺寸校验写聚焦单元测试。
2. 实现八视角 OpenCV 选择器：左键加点，右键/Enter 闭合当前多边形，U 撤销，R 重置，N 表示本视角无需 mask，S 保存本视角，Esc 取消整次发布。
3. 一次性发布八张原 ROI 尺寸的二值 ignore mask、预览图和 SHA 绑定索引；`255=忽略`，`0=检测`，禁止覆盖已有发布目录。
4. 用户完成圈选后，用这组 mask 对 calibration 重推异常图，单独确定 masked-map 阈值，再接入独立 V4；不沿用 V3 pred_score 阈值。

聚焦验证：

```bash
uv run --no-sync pytest -q tests/unit/bmw_inspection/lab/test_efficientad_ignore_mask.py tests/unit/pipeline/test_bmw_lab_select_efficientad_ignore_masks.py
```

