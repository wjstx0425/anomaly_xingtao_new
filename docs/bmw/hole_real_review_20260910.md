# B4 六张真实图单孔诊断（2026-09-10）

已从用户指定 `dataset/bmw_lab_raw_clean_0820` 抽取6张全幅融合图，检查一个左件front视角右上圆孔候选。真实图诊断与证据保存已完成，真实缺孔/堵孔检出验证及验收边界仍未完成。

## 样本与目标

完整路径和来源见 `hole_real_samples_20260910.json`。选择4张normal、2张defect/defect，跨0820、0823采集；视角front，相机DA9805574。样本唯一键是session_id/sample_id/view；不同session可能复用sample_id，不能证明是六件不同实物。没有企业零件号和逐孔真值，型号暂用明确数据集占位标识 `dataset_bmw_left_unconfirmed_part_number`。

全幅4024×3036、uint8 BGR；使用fused_only源的fused通道，再BGR转gray。清单记录hdr_fused、1500/6000曝光，short/long文件列为空。直接读取原PNG，不加公共ROI ignore mask；不是单曝光传感器RAW。

候选孔原图ROI `[2940,660,3200,940]`，半开xyxy，宽260×高280，见 `results/bmw_hole_b4_real_20260910/target_overview.jpg` 黄色框。候选特征ID `front_upper_round_candidate`，需用户确认它属于本轮首检孔。六图检查区未包含下方孔，不扩大搜索或进行孔驱动配准。

孔内可见同心纹理，可能是后方夹具或背景，当前不能据暗色断言实际通孔畅通；也不能叫堵孔。此观察是待确认问题，不是已识别夹具或确定缺陷。

## 实测方法与边界

首张normal的ROI采用OpenCV Otsu自动得到阈值72，dark极性，输出参考mask面积14664px²。这是从图像提取的探索参考，不是批准的名义孔mask。其余五图固定使用同一阈值和原图坐标；没有按待检孔平移对齐，没有形态学修补。阈值扰动±1仅探查相邻uint8灰度级，未设置通过边界。

新增 `status=diagnostic`，允许显式 `limits=null`、`max_changed_fraction=null`、`registration.validated=false`。保留原图/通道/ROI/分割参数校验；输出全部几何证据但始终REVIEW，不设判废fact。ready模式依旧要求完整已验证参数，draft仍拒绝。per-capture fixture_verified与observable为false，未伪造确认。

| 编号 | 标签 / session / group | 暗区与参考重合比例 | 面积相对差 | 形状差1-IoU | 质心偏差px |
|---|---|---:|---:|---:|---:|
| 1（参考） | normal / 20260820_162040_682498 / 001 | 100.00% | 0.000% | 0.000% | 0.00 |
| 2 | normal / 同上 / 020 | 98.03% | 0.327% | 3.556% | 1.56 |
| 3 | normal / 20260823_161606_428291 / 001 | 95.87% | 0.375% | 7.592% | 3.99 |
| 4 | normal / 20260823_164432_522052 / 020 | 96.86% | 0.552% | 5.590% | 2.94 |
| 5 | generic defect / 20260820_155511_803522 / 001 | 97.85% | 0.334% | 3.891% | 1.96 |
| 6 | generic defect / 同上 / 020 | 97.76% | 0.075% | 4.459% | 1.85 |

字段open_fraction在本实验只是**分割暗区与参考内部的重合比例**，不能解释为已证实的实际开放比例。blocked_fraction也不是已确认堵塞比例。红色叠加是相对参考未重合的像素，可能包含位置变化和分割差异，不等同于堵塞。孔外小暗点未被去噪去掉，也计入几何；样本6的组件数为7，其中多数是小暗点，不能当作7个孔。

样本1参考自比较自然为零差异，不是独立验证。两张通用缺陷图的目标区域仍接近参考，未找到确定缺孔/堵孔正例。不能根据这六张设置或验收缺陷阈值。

## 复现与交付

```bash
UV_CACHE_DIR=/tmp/b4-uv-cache uv run --no-sync python tools/bmw/measure_hole_samples.py \
  --manifest docs/bmw/hole_real_samples_20260910.json \
  --output results/bmw_hole_b4_real_20260910
UV_CACHE_DIR=/tmp/b4-uv-cache uv run --no-sync python -m pytest tests/unit/bmw_inspection -q
```

输出目录需是新目录，已存在时换一个新名称。当前六图运行完成：6 REVIEW、0 ERROR，synthesized_test_data=false、diagnostic_only=true、acceptance_verified=false。270个BMW测试通过，1条既存NVML警告；未测试GPU或相机。

结果目录包含诊断配置、参考mask、summary.json、目标全图、六图原ROI/叠加对照图，以及每图独立result.json、原ROI、灰度图、mask、ROI/全图overlay。每图JSON包含源文件SHA256和采集谱系；代码没有改源图。主入口 `results/bmw_hole_b4_real_20260910/summary.json`，对照图 `contact_sheet.jpg`。

## 汇总待用户确认

1. 黄色框的左件front右上圆孔是否就是首个要检查的孔？若是，对应企业型号/零件号是什么？
2. 孔内同心纹理是否来自孔后方夹具或背景？正常件应看到什么，哪些现有文件是这个孔明确漏冲/卡料/堵孔的正例？可只提供样本文件名或group；本轮六图没有这种真值。
3. 孔内多少残留/边缘缺口需要判废，允许多少几何偏差？有图纸公差或好坏边界图则优先使用；若暂无标准，继续保持诊断并收集边界样本。

无须重新提供整批数据。独立夹具定位重复性还需后续验证；当前不利用目标孔自身消除偏差。未扩展多个孔、未接主流程、未改钢印、未提交推送。
