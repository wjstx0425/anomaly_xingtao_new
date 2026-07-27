# ZS32 Template v13 手动阈值发布设计

日期：2026-07-27

## 目标

把用户在 Template v12 `model.json` 中确认的三个手动阈值发布为独立、可校验的 v13，
不覆盖 v12，并将 Stage35 与 Dashboard 的默认运行包切换至 v13。

## 阈值

- `right/front`: `0.024885842800140383`
- `right/front_left`: `0.029833445549011232`
- `right/front_secondary`: `0.21635736227035524`
- 其余五个视角保持 v12 原值。
- 每个视角继续使用 `low_threshold == high_threshold` 的二值门限。

`front_secondary` 阈值明显宽松于 v12，但这是用户检查后明确确认的手动值。

## 不可变发布边界

1. 先把当前已修改的 v12 模型快照到独立 v13 目录。
2. v13 的 model、template、threshold 版本身份统一更新为 v13，ROI 身份保持不变。
3. v13 仅复制 `model.json`、`model.sha256` 和 40 张模板；旧 v12 评估报告不作为 v13 评估结果。
4. 恢复 v12 三个原始阈值，并验证其 `model.json` SHA256 回到
   `5b071bcfb8a603f5bbf20ecf284f590e88e3e65e114f434fbde1eaf81692ff6e`。
5. 从 v12 配置派生独立 v13 source/profile，保留严格八视角、24-group 和两个 secondary required。
6. 复用现有发布链：
   `Stage37 publish-assets -> Stage34 thresholds -> Stage37 finalize`。
7. PatchCore、YOLO、ROI 资产及其阈值保持 v12/v11 当前绑定不变。

## 输出

- `results/zs32_template_gate_right_0727_eight_view_v13`
- `config/fusion/zs32_right_eight_view_24_group_commissioning_template_0727_v13.json`
- `config/fusion/zs32_eight_view_24group_template_0727_v13_bundle_source.json`
- `results/zs32_runtime_assets_eight_view_template_0727_v13`
- `results/zs32_24group_template_0727_v13_commissioning`
- `results/zs32_runtime_bundle_eight_view_template_0727_v13/runtime_bundle.json`

## 验证

- v13 模型 sidecar 与实际 `model.json` SHA256 一致。
- v13 runtime bundle 在 fresh process 中成功加载。
- v13 恰有 24 条阈值记录和 8 条 Template 记录。
- v13 Template 记录与模型八视角 low/high 完全一致。
- 两个 secondary 视角均要求 Template、PatchCore、YOLO。
- v13 的 PatchCore、YOLO、ROI 与 v12 内容绑定保持一致。
- v12、v11、v10 分别在 fresh process 中继续可加载。
- Stage35 和 Dashboard 默认路径解析到 v13。

## 限制

- v13 是用户手动阈值版本，不重新声称 calibration/final-test 指标。
- 保持 `commissioning_only=true` 和 `production_release_allowed=false`。
- v12 的历史评估文件不复制为 v13 评估证据。
