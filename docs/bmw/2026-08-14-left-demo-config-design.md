# BMW 左手 Demo 配置设计

## 目标

将已完成的左手 Template、EfficientAD、手工 mask、组件阈值和旋转光痕 V3，与现有左右手联合 YOLO 组合为一个可由现有八视图 Demo 直接加载的实验配置。

## 方案

建立独立部署组合目录，不修改训练源目录，也不修改通用 Demo schema：

- `template` 链接到 `bmw_left_normal_20260814_models_v3/template`。
- `efficientad` 链接到 `bmw_left_normal_20260814_models_v3/efficientad`。
- `yolo` 链接到 `bmw_right_multisource_left_yolo_v1/yolo`。
- 左手配置绑定 ROI v2、prepared manifest、EfficientAD mask/policy/threshold 和左手旋转光痕候选报告。
- Template 使用各视角 `model.json` 内置阈值，不配置右手专用 Template mask 阈值。
- 不配置右手可信 OK 参考库。

发布器使用临时目录写入三个目录链接和 `composition.json`，完成后改名为最终组合目录；目标已存在时拒绝覆盖。输出配置通过现有 `load_demo_config()` 验证后发布。

## 验收

- 组合目录能解析到八个左手 Template、八个左手 EfficientAD checkpoint 和共享 YOLO checkpoint。
- 配置使用 `tracked_profile_v3_manual_rotated_candidate`，且不带 weak override。
- `load_demo_config()` 能完整加载配置并验证所有已有 SHA 绑定。
- 不修改旧 V5/V6 配置、训练源目录或模型数值。
