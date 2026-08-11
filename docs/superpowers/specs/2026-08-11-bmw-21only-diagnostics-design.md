# BMW 21点批次单变量诊断设计

## 目标

验证8月10日21点批次与下午批次之间的工装位置变化，是否是Template正常件通过率低的主要原因；随后独立验证光痕新特征和EfficientAD阈值，避免多变量同时变化。

## 固定条件

- 上午部署结果 `results/bmw_lab_one_click/bmw_right_multisource_left_yolo_v1` 不覆盖、不修改。
- 严格数据源为 `dataset/bmw_lab_training/bmw_right_batch_20260810_21_roi_v1`，prepared manifest SHA256 为 `a5ce0526628afa16616341c6660e2e6e7453c833f285fe64dd3625e696744459`。
- 保持八视图ROI、Template预处理、512x512尺寸、最大平移12像素、每视角5模板不变。
- YOLO保持上午模型不变；21点批次240张缺陷视图尚未复核，不用于YOLO训练。
- 所有候选使用新目录，默认Demo配置不自动切换。

## 阶段一：Template单变量候选

每视角只从21点批次 `split=train,label=normal` 的50张图中按现有算法选择5张模板。由于21点缺陷视图尚未复核，不能重新拟合平衡准确率阈值；候选模型沿用上午对应视角的数值阈值。这样唯一变化是模板训练样本，可直接检验工装位置假设。

在21点 `calibration+final_test` 正常件上报告每视角误拒数和八视图整件通过率。缺陷召回只作诊断，不作验收结论。

## 阶段二：光痕v2并行诊断

保留 `front_left` 专用81x613 ROI。对每一行计算中心候选带相对左右背景带的原灰度局部对比，平滑成一维row-score，再计算存在覆盖率、最长连续段、最大断口和断口数。v2只作为离线并行候选，与现有Top-hat在相同calibration/final_test数据上比较，不修改默认光痕配置。

## 阶段三：EfficientAD 21点候选

每视角仍使用EfficientAD-S、30 epochs、batch 1、256x256、seed 42，只将正常训练集替换为21点的50张normal和12张no_streak。21个校准分支正常件用于整件阈值选择，允许最多1件被判NG，即4.76%。阈值从新checkpoint分数重新计算，不直接给旧阈值加常数。

21点可见缺陷尚未逐视角复核，因此缺陷分数和整件命中只作探索性报告；不能宣称完整缺陷验收。

## 产物

- `results/bmw_lab_one_click/bmw_right_batch_20260810_21_template_only_v1`
- `results/bmw_lab_one_click/bmw_right_batch_20260810_21_bright_v2`
- `results/bmw_lab_one_click/bmw_right_batch_20260810_21_efficientad_v1`
- 一份汇总JSON/Markdown，记录代码提交、manifest/ROI/model SHA256、参数和对比结果。

## 成功标准

- Template候选确实只引用21点session，且算法参数和数值阈值保持不变。
- Template报告能回答八视图正常件通过率是否提高。
- 光痕v2必须在final_test上优于当前规则，且证据像素落在肉眼光痕位置，才允许后续接入。
- EfficientAD阈值候选在21个校准分支正常件上最多误拒1件；缺陷召回明确标为未完整验证。
