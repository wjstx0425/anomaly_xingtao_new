# BMW 整圈轮廓误报修复与复测（2026-09-11）

本轮修复了参考自检、背景组件归属和不连续观测的比较错误；跨正常件的误报和未知边仍未解决，不能用于整圈自动放行。最终结果使用 `artifacts/bmw_contour_fix_20260911/run_v5` 与 `extra_normals_v5`，此前各版本仅供调试追溯。

## 已定位的原因与修改

1. 原实现将背景螺钉等前景组件混入工件外圈。现在复用 OpenCV GrabCut，通过参考内部核心的重叠判断组件归属；保留与工件相连的大凸起、所有原始组件及诊断。靠近工件但归属不明的组件触发 REVIEW，不直接消失。
2. 参考与待测剖面的候选生成规则不一致，同图也可能选到不同边。现在使用一致规则、极性和材料过渡支持；有多个候选时要求局部灰度形状匹配及足够区分度，无法确认时保留 UNKNOWN。没有相同文件特殊返回或强制贴合参考。
3. 孤立有效点没有相邻有效测量线段，却被用于与远处线段比较，产生虚假大偏差。现在不将孤立点算作有效曲线，不连接未知段。
4. 参考法向只在一个像素距离取证，窄脚和像素锯齿会被误判。现在联合 1/2 px 材料内外证据，矛盾证据仍未知；无效参考法向从 313/9787 降至 50/9787。参考 mask、整圈必检分母和开发公差未改。
5. 自动分割在参考图上就不符合材料标准的位置，不能在待测图上直接认定为确定缺陷。相应粗轮廓候选单独保存为 `unconfirmed_coarse_events.json`，包含位置和偏差，仍阻止 PASS。原始粗轮廓及距离保留。
6. 公共 `inspect_image` 补充规格和完整参考有效性检查；草稿回放通过独立诊断脚本运行，明确强制 REVIEW，不能绕过正式参考确认。

## 最终实图结果

NG/REVIEW 数为候选事件数，不是独立缺陷数；还有单列的未确认粗轮廓区域。所有图片正式状态均为 REVIEW，`full_perimeter_pass=false`，参考仍是 draft。

| 样本 | 有效覆盖 | NG 候选 | 测量 REVIEW 候选 | 未确认粗轮廓区域 |
|---|---:|---:|---:|---:|
| 用户正常参考 002 自检 | 83.31% | 0 | 0 | 6 |
| 缺陷 002 | 66.83% | 36 | 156 | 13 |
| 缺陷 005 | 68.13% | 40 | 146 | 18 |
| 缺陷 007 | 65.11% | 73 | 362 | 15 |
| 缺陷 015 | 68.33% | 42 | 176 | 19 |
| 缺陷 020 | 61.09% | 58 | 348 | 13 |
| 额外正常目录 001 | 75.47% | 4 | 72 | 17 |
| 额外正常目录 003 | 75.52% | 0 | 55 | 12 |

参考自检从原来的 6 NG + 58 REVIEW 测量候选降到 0，但仍有约 1633 px 未观测和 6 个未确认粗轮廓区域。覆盖低于原来的 87.87%，不能把变为 UNKNOWN 当成检测质量提升。额外两张来自相同正常 session 的目录标签，尚未逐张获得用户合格确认；没有独立验证集结论。

5 张用户缺陷图中原先复核的 8 个局部区域仍有对应 NG 候选，包括 group020 顶脚约 230 px 的大变化。与此同时错误/不确定候选仍多，不能声称 5/5 准确率。具体见 `posthoc_v5.json`。

额外正常 001 的 4 个 NG 候选集中于底脚末端（约 x2519–2609、y2865–2868），偏差约 5–7 px。对齐后 5 个横坐标探针均显示可见下缘比正常参考向上约 5 px，上方颈部折线基本对齐，因此不像内部折线误选。现有 4 px 开发阈值会触发；二维图无法判断这是正常件间差异、姿态还是实际形变，未为通过样本而放宽阈值。

## 需要人工提供的信息

- 参考图上、下小脚横向暗线是否属于折弯/阴影、材料连续；底部暗色侧壁是否全部属于要求检测的外轮廓。仅需确认这些局部的物理边界，不需要人工描待测图。
- 额外 normal001 是否确为合格，以及底脚允许差异。若无既定公差，需要后续用确认合格件重复取放数据建立正常波动范围，不能从这一张反推合格阈值。

局部对照：[上脚](../../artifacts/bmw_contour_fix_20260911/reference_analysis/top_foot.png)、[底脚暗边](../../artifacts/bmw_contour_fix_20260911/reference_analysis/bottom_dark_edge.png)、[两张正常图底脚对齐](../../artifacts/bmw_contour_fix_20260911/normal001_bottom_comparison.jpg)。对齐图左为参考、中央为 normal001、右为绿/紫叠图。

## 复现与证据

六张用户样本：

```bash
UV_CACHE_DIR=/tmp/bmw-uv-cache uv run --no-sync python tools/bmw/validate_contour_real_samples.py \
  --manifest docs/bmw/contour_real_samples_20260911.json \
  --draft-config configs/bmw/checks/contour/left_front_4024_development_v2.json \
  --reference-mask artifacts/bmw_contour_real_20260911/reference_mask_candidate_v2.png \
  --output-dir artifacts/bmw_contour_fix_20260911/replay_new
```

额外正常图使用 `docs/bmw/contour_extra_normals_20260911.json`，增加 `--sample-id normal_extra_001 --sample-id normal_extra_003`，使用新的输出目录。标签只用于报告；输入 SHA 必须匹配。mask 仍是上一轮的正常参考候选，没有手工修改任何待测轮廓。

[完整对照网页](../../artifacts/bmw_contour_fix_20260911/review.html)、[8 图总览](../../artifacts/bmw_contour_fix_20260911/contact_sheet_v5.jpg)、`comparison_summary.json`、`artifact_checks.json` 均在同目录。8 张输入 SHA 不变、9787 个必检采样分母不变、JSON 严格有效、证据路径齐全。结果保存算法文件 SHA。

软件回归 `UV_CACHE_DIR=/tmp/bmw-uv-cache uv run --no-sync pytest -q tests/unit/bmw_inspection tests/unit/pipeline/test_bmw_lab_eight_view_demo.py`：**436 passed，1 warning，3.29 s**。该总数包含工作区其他 BMW 功能；warning 为既有 NVML 初始化警告。覆盖新增组件、参考支持、候选参数、孤立点和未知粗边证据等回归。`git diff --check` 通过。未验证相机、GPU、Windows、交互教学 GUI、生产准确率及节拍。
