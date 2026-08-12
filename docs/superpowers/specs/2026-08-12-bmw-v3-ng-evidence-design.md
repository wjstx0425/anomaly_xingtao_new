# BMW 第三版 NG 证据与现场留存设计

## 目标

建立一个不覆盖前两版的实验配置：使用晚九点 Template、光痕 v2、既有 YOLO，以及在第二版 EfficientAD 逐视角阈值上增加 `0.05` 部署余量；现场每轮保留 HDR 短曝光、长曝光、融合图、ROI、模型结果和图像统计，并能逐条查看所有 NG。

## 模型身份

- Template：`bmw_right_batch_20260810_21_template_only_v1/template`。
- Bright streak：修正 ROI `[1792,1180,1873,1793]` 的 `raw_profile_v2`。
- YOLO：`bmw_right_multisource_left_yolo_v1/yolo`，候选阈值 `0.10`、最终阈值 `0.25`。
- EfficientAD：`bmw_right_batch_20260810_21_efficientad_v1/efficientad`；部署阈值为第二版每个视角阈值加 `0.05`，同时保留基础阈值、调整量和 checkpoint SHA。

## 数据留存

相机接口在保持现有融合图返回契约的同时，缓存本轮每个视角的短曝光、长曝光、融合 HDR 和融合过曝率。完成八视图检测后，结果以独立 capture ID 写入结果目录：

- `images/<view>_short.png`
- `images/<view>_long.png`
- `images/<view>_hdr.png`
- `rois/<view>.png`
- `evidence/<branch>_<view>.png`
- `inspection.json`
- 追加式 `inspection_index.csv`

`inspection.json` 保存每条结果的分数、基础阈值、部署阈值、超限量、完整原因，以及 ROI 的均值、标准差、P1、P99、暗像素率、亮饱和率和 SHA-256。写入采用临时目录后原子发布，不覆盖已有 capture ID。

## 前端

结果页从所有 `NG/ERROR` 结果生成队列；`N/P` 循环切换下一条/上一条，`1-8` 与 `T/L/Y/E` 仍可手动查看。右侧显示 NG 序号、视角、项目、完整原因、分数、基础阈值、部署阈值和超限量。证据区同时呈现当前 NG 的短曝光、长曝光、融合 HDR 和模型证据。

证据名称必须准确：YOLO 为“真实检测框”，光痕为“规则 ROI 证据”，Template 与 EfficientAD 为“诊断热区”，不声称后两者是精确缺陷框。

## 边界

`+0.05` 是现场实验余量，不是缺陷召回已经验证的最终阈值。任何因阈值放宽而产生的漏检，都必须通过保存的现场 NG/OK 数据和后续缺陷样本重新评价。
