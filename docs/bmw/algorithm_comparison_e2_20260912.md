# BMW 第1—10项算法对照：2026-09-12 首轮开发回放

已完成当前连续轮廓修改，并按 V1.1 交接包执行 E0—E2 的首轮有限对照。**8 张同一开发清单、6 个分支、48 条结果，执行错误为 0；尚无证据支持替换现场基线。**本轮只有第5项“变形”的图像级确认标签，不是十项验收或独立测试集。

先看[同图并列结果](../../results/bmw_benchmark/left_front_e2_20260912/comparison.html)、[机器汇总](../../results/bmw_benchmark/left_front_e2_20260912/summary.md)。轮廓前后对照独立记录在[连续轮廓实图报告](contour_continuous_20260912.md)。

## 实际结果

两张用户确认正常图为 reference、normal_extra_001；normal_extra_003 未经逐图确认，单独记 unknown。五张 defect_002/005/007/015/020 由用户确认存在轮廓变形。以下分母是图片数，不是五件独立实体或完整缺陷实例数。

| 分支 | 变形图 NG / 5 | 变形图 PASS / 5 | 变形图 REVIEW / 5 | 确认正常 NG / 2 | 确认正常 REVIEW / 2 |
|---|---:|---:|---:|---:|---:|
| 现有 Template | 5 | 0 | 0 | 0 | 0 |
| 冻结 EfficientAD | 1 | 4 | 0 | 0 | 0 |
| 现有 YOLO，640 输入 | 5 | 0 | 0 | 0 | 0 |
| R01 连续轮廓 | 0 | 0 | 5 | 0 | 2 |
| R02a 黑帽/顶帽细线 | 0 | 0 | 5 | 0 | 2 |
| 同一 YOLO 权重，1280 输入 | 5 | 0 | 0 | 0 | 0 |

EfficientAD 只对 defect_007 自动 NG，其余四张变形图 PASS。其他分支始终独立运行，没有让 EfficientAD 的 PASS 跳过后续检查。Template 和 YOLO640 已经拦截全部五张图，因此1280相对它们的新增自动 NG 图像为0；相对单独 EfficientAD 为4。这个差集不能宣称为相对现有完整系统补检4个缺陷。

使用原 `fuse_demo_status` 对这三个 front 基线结果进行有限融合，得到五张变形图 NG、两张确认正常图 PASS、unknown 图 PASS。记录在 [partial_front_fusion.json](../../results/bmw_benchmark/left_front_e2_20260912/partial_front_fusion.json)。**这只是 front 的三分支融合**；缺同件八视图，未运行全部25项检查，`baseline_system` 和完整工件耗时均不可用。

R01 保留各图 NG 候选及未知周长，但草稿参考和覆盖缺口使正式结果均为 REVIEW。不能把这个表里的0正常NG理解为轮廓没有误报：normal001仍有5条底脚NG候选，两个确认正常图都不能自动放行。

## 1280 没有显示稳定的定位收益

图像级5/5不足以说明变形位置找对。逐框检查发现：

- defect_007：640的最终框覆盖先前目视左侧区域；1280改为覆盖右侧区域，左侧没有最终框。
- defect_015：1280保留左侧、底脚框，但上侧目视区域没有最终框。
- defect_020：1280仍然NG，却没有框落在明显展开的上脚区域；最高分框为中下部 `[1797,2005,1850,2112]`，置信度0.7217。640有上脚附近的最终框。

[1280 的020实图](../../results/bmw_benchmark/left_front_e2_20260912/evidence/D01_yolo_high/defect_020/overlay.png)显示了为什么“某处报NG”不能算正确定位。沿用已有八个事后目视区域的复核数据见 [yolo_posthoc_regions.json](../../results/bmw_benchmark/left_front_e2_20260912/yolo_posthoc_regions.json)：区域并非完整审核标注，只用于指出框位置改变，不报告AP或定位召回率。

YOLO运行ROI仍为现用 `[1218,166,3353,2980]`；ROI之外没有检测覆盖，包括020上脚高于y166的部分。没有根据缺陷框裁图或静默扩大ROI。对reference增加独立前向探针，实际模型输入tensor分别为 `[1,3,640,512]` 和 `[1,3,1280,992]`，原ROI为2814×2135；记录见 [yolo_input_shapes.json](../../results/bmw_benchmark/left_front_e2_20260912/yolo_input_shapes.json)，不是假定两者都为正方形输入。这次探针不计入性能或准确率结果。本轮只改变推理输入尺寸；旧权重在更高尺寸下未证明收益，也不能由此断言匹配训练或分块方案无效。

## 细线规则目前不能作为裂纹判定

新增 `R02a` 使用现有 OpenCV 与 scikit-image：低频背景扣除、多尺度多方向黑帽/顶帽，独立保留暗/亮响应，连通域输出面积、骨架长度、宽度和方向。阈值20灰度响应、最小面积20px固定，未逐图调整，也没有以大长宽比规则删除全部短粗候选。

确认正常的 reference 和 normal001 分别有 **1313、1211个候选连通域**。从[正常图叠图](../../results/bmw_benchmark/left_front_e2_20260912/evidence/R02a_morph_lines/reference/overlay.png)可见，规则也大量响应材料纹理、折弯和亮暗边界。原生输入坐标响应图、候选mask和全部框均保存；ROI外明确为未观察，响应为0不代表正常。

因此8图全部 REVIEW，候选数不是裂纹数。当前没有确认开裂/划痕的定位真值，不能通过提高阈值把正常图“调干净”后就宣布有裂纹检出能力。

## 运行与证据边界

沿用左件现用配置、加权Template阈值、EfficientAD ignore/component后处理、YOLO候选0.1与判定0.25。EfficientAD最终分数是现有连通域规则得分，阈值0.39，没有用原始 `pred_score` 替换它。三套模型、模板图片、ROI、mask、训练元数据和后处理实现共80项依赖已冻结SHA，运行结束复核未变。

EfficientAD保存的256²图是**模型内部上采样、Engine归一化后、runtime ignore之前**的异常图；不是原生特征分辨率，也未逐测试图min-max后参与评分。Template保留原512²配准空间叠图，并记录含padding、配准平移和像素中心的回映射；全图展示仅作可视化，没有用放大图片冒充更高测量精度。Template青色矩形是预设加权区域，不是缺陷框或真值框。

实机 CUDA 不可用，所有新推理使用CPU、4计算线程、2 interop线程，模型顺序运行。每个实跑分支预热1次，再测8图。中位数如下：

| 分支 | 算法记录 / ms | 含读图、SHA校验及证据导出的wall / ms |
|---|---:|---:|
| Template runtime | 20.63 | 712.64 |
| EfficientAD runtime | 708.46 | 1258.77 |
| YOLO640 runtime | 26.85 | 571.85 |
| YOLO1280 runtime | 54.21 | 596.39 |
| R02a 预处理+规则+测量 | 约648 | 2490.39 |

R01复用本轮刚完成的8图完整轮廓结果；导入时检查原图SHA、推理参数、实际参考recipe及参考资产SHA。参考教学元数据与基础配置的差异逐项记录，不放过公差变化。原会话算法时间中位数约37.05秒；本次约161ms只是缓存读取和证据复制，不能算作轮廓推理提速，也不能与表内新会话公平排名。详细阶段时间和预热状态见 [latency.json](../../results/bmw_benchmark/left_front_e2_20260912/latency.json)。没有相机、生产节拍或完整工件性能验收。

## 数据审计与下一批条件

[数据审计](../../results/bmw_benchmark/left_front_e2_20260912/data_audit.md)、[十项覆盖表](../../results/bmw_benchmark/left_front_e2_20260912/requirement_coverage.csv)、[待解决项](../../results/bmw_benchmark/left_front_e2_20260912/blocked_items.md)已生成。

物理零件ID、谱系及完整定位标注缺失，全部样本留在dev。reference已经用于建立参考。历史训练元数据显示部分缺陷图来自训练/校准来源，旧清单又没有可完整核验的图像SHA；不能称为干净泛化评估。unknown不作正常负例，第1—4、6—10项无确认类别正样本，指标保持不可评估。没有读取locked_test图像、下载/训练新网络或升级环境。

当前保留 R01 作几何诊断、R02a作细线证据，继续保留现有640输入基线；1280只作为已完成的消融记录，不替换部署配置。下轮最需要：

1. 一批确认开裂/划痕的原图及缺陷位置，配正常折弯、油膜边缘、高光负例；这样才能评价R02a是否有补检价值。
2. 正常001底脚的人工材料边界和同件重复摆放图，用于区分投影变化、配准残差与错边；不再重复确认已明确的工装B。
3. 实体零件与重拍关联，以及完整框/mask，才能建立独立划分、统计真实新增定位；有mask后再讨论后续监督分割。

## 复现

新入口复用现有runtime预测器，没有重新搭建训练平台：

```bash
UV_CACHE_DIR=/tmp/bmw-uv-cache MPLCONFIGDIR=/tmp/bmw-mpl-cache \
uv run --no-sync python tools/bmw/run_algorithm_comparison.py \
  --plan configs/bmw/benchmark/left_front_e2_development.json \
  --output results/bmw_benchmark/left_front_e2_replay_new
```

输出目录必须不存在，已有结果不会被覆盖。计划和清单位于 [配置](../../configs/bmw/benchmark/left_front_e2_development.json)、[8图清单](benchmark_left_front_manifest_20260912.json)。预测入口不接收类别真值，评价在预测后关联标签；缺失/异常预测计ERROR保留分母。正式能力验收仍受上述数据缺口限制。


## 验证

`uv run --no-sync python -m pytest tests/unit/bmw_inspection tests/unit/pipeline -q`：**546 passed**，仅已有NVML初始化警告。入口help、Python编译及diff空白检查通过；没有GPU或硬件测试。独立产物审查确认48条结果唯一且完整、5次预热成功、8原图/80依赖/22源码SHA一致、48叠图尺寸正确、HTML链接均存在，详见 [verification_review.json](../../results/bmw_benchmark/left_front_e2_20260912/verification_review.json)。
