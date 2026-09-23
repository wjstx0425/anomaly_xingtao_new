# BMW 参考折弯修正与正常标签确认（2026-09-12）

> 后续已确认：用户指B（亮脚下方黑色区域），不是A。B已在mask之外，无需再删材料；下文A/B待确认描述为历史过程。后续见 [B定位与阈值敏感性复核](contour_fixture_B_20260912.md)。

用户确认：暗线是折弯，最下方暗色侧壁是工装而非材料；normal001 可以认为正常。已将 normal001 标签来源更新为本次用户确认，normal003 仍只有正常目录标签。

本轮完成确定的四处折弯参考修补并复测八张图。工装在截图中的精确位置仍需 A/B 定位，未猜测删除。因此这次结果明确命名为 folds_only，不能称作已完成工装排除。

## 参考修改

- 仅在正常参考四个局部填补被误分成背景的折弯细缝，共新增 548 个材料像素、删除 0 个像素；没有全局平滑、缩放或待测图人工描边。
- 新 mask：`artifacts/bmw_contour_reference_20260912/reference_mask_folds_only.png`。编辑 ROI、规则、原因、源 SHA 和复现脚本在同目录 `reference_edits.json`、`edit_reference_mask.py`。
- 参考采样从 9787 变为 9485：错误伸入材料内部的细缝边界不再作为外轮廓。新标准仍整圈必检，未排除任何弧段。由于参考几何变化，不应把覆盖比例直接当成同一标准下准确率的提升。
- 材料法向仍有 49 个不确定点，参考保持 draft。配置、公差和提取算法不变；回放脚本仅将参考版本元数据改为实际 mask 文件名，避免沿用旧硬编码版本。

## 实测结果

事件数并非独立缺陷数；所有正式输出均 REVIEW，未知段仍阻止整圈通过。

| 样本 | 覆盖 | 未知长度 px | NG 候选 | REVIEW 候选 | 未确认粗边区域 |
|---|---:|---:|---:|---:|---:|
| reference | 84.84% | 1437.9 | 0 | 0 | 9 |
| normal_extra_001 | 75.35% | 2337.9 | 6 | 58 | 7 |
| normal_extra_003 | 75.15% | 2356.9 | 0 | 50 | 8 |
| defect_002 | 68.79% | 2959.8 | 34 | 137 | 7 |
| defect_005 | 66.45% | 3181.8 | 34 | 118 | 3 |
| defect_007 | 64.66% | 3351.8 | 69 | 375 | 5 |
| defect_015 | 68.87% | 2952.8 | 42 | 150 | 5 |
| defect_020 | 64.29% | 3386.8 | 52 | 389 | 5 |

正常参考仍为 0 NG，未知长度从约 1633 px 减到 1438 px。但用户确认正常的001仍有6个NG候选（之前4个），均集中底脚下缘，约5–7 px，不能声称误报解决。折弯修补改变了曲线采样及连续有效支持，事件数也会改变。该样本目前体现“已确认正常与开发规则冲突”，并非6处物理缺陷。没有因此放宽全局4 px阈值。

上一轮两正常图均有908个参考边校准未知单元；角点对应失败另有602/506个。说明主要瓶颈还包括参考图像边界校准和角点支持，单独改阈值不能解决未知覆盖。可复现原因统计保存在 `normal_diagnostics.json`（明确是上一轮v5结果，不能当本轮统计）。

## 工装位置

[标框图](../../artifacts/bmw_contour_reference_20260912/fixture_location_confirmation.png)：A为左下大块暗色凸出部，B为右下小脚亮边以下黑色区域。上一张提问图没有标号，现补充位置确认，不能把区域名直接转成像素范围。B目前已在mask之外；若A是工装，需要进一步在亮材料/工装接触边上建立正确参考。

历史v2曾将A附近暗区用参考前景笔刷补入。v1则会沿材料内部暗筋形成长假开口，因此不能直接把v2-v1差集全部删除，见 `dark_region_candidate_v1_vs_v2.png`。待位置确认后继续准确修正。

## 验证与复现

八图源SHA未变；新mask仅在声明的四个ROI内增加548像素，外连通组件为1、源参考像素与用户图片一致。检查见 `mask_checks.json`、`summary.json`。未改核心检测算法，本轮未重复运行上轮436项软件回归；回放脚本帮助命令及实际八图运行完成。无生产准确率、相机或节拍验收。

```bash
UV_CACHE_DIR=/tmp/bmw-uv-cache uv run --no-sync python tools/bmw/validate_contour_real_samples.py \
  --manifest docs/bmw/contour_real_samples_20260911.json \
  --draft-config configs/bmw/checks/contour/left_front_4024_development_v2.json \
  --reference-mask artifacts/bmw_contour_reference_20260912/reference_mask_folds_only.png \
  --output-dir artifacts/bmw_contour_reference_20260912/replay_new
```

额外正常图采用 `docs/bmw/contour_extra_normals_20260911.json`；每次使用新的输出目录。已完成输出分别是 `folds_only_normals`（参考及001/003）与 `folds_only_defects`（五张缺陷），逐图目录包含轮廓叠图、原始测量、事件和参数快照。
