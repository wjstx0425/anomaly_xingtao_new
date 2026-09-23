# 三张新增划痕图回放：2026-09-13

用户提供017、018、031并确认三图都有划痕。本次先看原图，再复用上一轮冻结模型/阈值/ROI回放，没有用目视位置裁推理输入，没有训练或调阈值。三图加两张既有确认正常图，共5图×5分支=25条结果，0执行错误。

**结论：可以看到明显痕迹，但现有检测还不能可靠覆盖这三张图的划痕。**

| 图 | Template | EfficientAD | YOLO640 | YOLO1280 | 实际定位检查 |
|---|---|---|---|---|---|
| 017 | NG | PASS | NG | NG | YOLO最终框主要是右缘缺口，未覆盖长斜划痕 |
| 018 | NG | NG | NG | NG | 640主要报左侧凹点；右上擦划区0.2147未过0.25。1280在右上擦划区出现0.2819/0.2571最终框，覆盖一部分 |
| 031 | NG | PASS | PASS | PASS | YOLO只有上脚低分候选，没有框定位到用户确认的右上圆孔上方痕迹；Template全图NG不能作为划痕定位证明 |

两张确认正常图在四个模型上均PASS，分数与前轮完全一致。三张图均有用户图像级划痕标签，但目视框尚未逐实例确认；不能把本表的图像NG直接称为划痕召回率。独立零件数量和训练来源重合仍未核实。

## 目视位置

017在中右部下平面内有明显从左下到右上的斜长沟痕，约 `[2390,1550,2865,1905]`。沟痕伴有粗糙擦伤，靠外缘另有缺口。现有YOLO640最终框 `[2795,1717,2847,1792]`，只约52×75px，不能解释为覆盖475×355px目视区域中的斜痕。

018在右上圆孔下方、靠外缘有明显成簇沟痕，约 `[3100,900,3255,1080]`，伴挤压/撕裂外观。图中左侧还存在凹点和其他细痕。640最高分框在左侧凹点，不能替右上区域加分；1280在右上有最终框，但不是完整实例分割。EfficientAD为无类别热区，不能单凭NG声称识别划痕类型。

031最初目视提出中右部上下两个平面内三条细暗曲线A/B/C，属于历史未确认假设：

- A：`[2535,1195,2620,1360]`，上平面偏下竖向弯曲细线。
- B：`[2495,1810,2615,1895]`，下平面靠下弧形细线。
- C：`[2695,1745,2810,1845]`，下平面靠右斜向细线。

[031全图位置](../../results/bmw_benchmark/scratch_review_20260913/031_locations.jpg) · [A/B/C局部2倍显示](../../results/bmw_benchmark/scratch_review_20260913/031_details.jpg)。**用户随后明确：指的是零件原图右上角圆孔/小脚附近，不是按放大图C作确认。** A/B/C保留为未确认假设，不作为划痕真值，也不能自动改成正常。

根据该指示，已查看圆孔上方靠外缘的短横向暗沟，以及邻近外缘斜向痕迹。复核窗口为 `[3040,580,3270,730]`，这是助手定位窗口，不是用户逐像素确认的完整实例框。[最新原图位置D](../../results/bmw_benchmark/scratch_review_20260913/031_confirmed_location.jpg) · [右上局部3倍显示](../../results/bmw_benchmark/scratch_review_20260913/031_confirmed_upper_right_crop.png)。

两个YOLO尺寸都没有在该窗口输出候选框；031的唯一候选在更上方小脚，640/1280分数为0.1084/0.1991，不能通过降低阈值就声称识别了这条划痕。EfficientAD仍PASS0.3453；该窗口内现有ignore mask覆盖率为0%，所以此处不能归因为被ignore直接屏蔽。R02a在短横向暗沟上有较清晰响应，但也响应邻近纹理/外缘，[规则局部图](../../results/bmw_benchmark/scratch_review_20260913/031_confirmed_r02_crop.png)显示这种混杂，仍不等于可靠独立判缺陷。用户定位原话和派生观测保存在 [user_location_confirmation.json](../../results/bmw_benchmark/scratch_review_20260913/user_location_confirmation.json)。

独立目视审查没有查看模型结果，全部假设记录于 [visual_review.json](../../results/bmw_benchmark/scratch_review_20260913/visual_review.json)，不是正式GT；未圈出区域也不当正常。

## 细线规则

R02a在017/018/031分别产生1606/1700/1546个候选连通域，两张正常图仍有1313/1211个。上述目视区域内有响应，但材料粗糙纹理、加强筋和边界也大量响应；不能把区域内亮起来算作划痕已正确提取。全部保持REVIEW，下一步应在确认的目标线与相近正常纹理之间检验区分能力，而不是把本组三张图调到NG。

## 证据及复现

- [所有模型同图对照](../../results/bmw_benchmark/scratch_review_20260913/replay/comparison.html)
- [原始逐图预测](../../results/bmw_benchmark/scratch_review_20260913/replay/predictions.jsonl)
- [5图清单与SHA](../../results/bmw_benchmark/scratch_review_20260913/replay/image_manifest.json)
- [运行状态](../../results/bmw_benchmark/scratch_review_20260913/replay/run_manifest.json)

复现脚本 `results/bmw_benchmark/scratch_review_20260913/run_replay.py` 使用现有 `run_algorithm_comparison.py` 的像素加载、固定预测器、证据导出和评价接口，输出独立 `replay` 目录并拒绝覆盖。若重跑应复制到新的实验目录（保持项目层级）或修改输出目录为新路径。命令为：

```bash
UV_CACHE_DIR=/tmp/bmw-uv-cache MPLCONFIGDIR=/tmp/bmw-mpl-cache \
uv run --no-sync python results/bmw_benchmark/scratch_review_20260913/run_replay.py
```

CPU、4计算线程、2interop线程、每分支一次预热。既有模型/配置/算法源码未修改，所有原图只读，未读取locked_test。验证以本轮25条真实回放和资产SHA检查为准；没有因仅增加样本而重复宣称上一轮546项测试是本轮新测结果。
