# B4 单孔离线检查（2026-09-10）

本轮交付可执行单孔模块、四态规则、离线证据和合成回归；**真实单孔验证未完成**。不扩展多个孔，不接主流程/GUI，不修改钢印、25项检测、公共ROI或阈值。没有生产用孔参数。

## 本地核对与复用

- `configs/bmw/rois/bmw_{left,right}_0820_v1.json`：4024×3036原图上的整件公共ROI，非指定孔ROI。`key_template/*key_rois*.json` 是Template加权区域，也没有孔身份或内部mask。
- 可读真实图：`dataset/bmw_lab_raw_clean_0820/left/front/normal/20260820_162040_682498/images/left_front_normal_bmw_left_normal_group001_000001_fused.png`，BGR、3036高×4024宽；对应采集清单为hdr_fused，short/long路径为空。不能宣称独立曝光可用。
- 数据集通用 normal/OK、others/NG 没有孔漏冲/堵孔专属标注，不能据此设边界或归因。
- 复用 `checks/stamp_config.py::_rectangle` 半开区间xyxy校验、现有OpenCV/numpy依赖，以及钢印 `StampReadResult.save` 的新目录独立存证模式。未改动这些接口。
- 主流程 `DemoBranchResult/BranchStatus` 没有REVIEW，`persist_inspection` 保存25项及曝光源图；其按分支/视角的证据命名不能直接容纳多特征。本轮采用独立 `HoleResult` 和含feature_id的文件名，B7再做统一融合，避免改变现有枚举或输出合同。

## 输入与配置

`configs/bmw/checks/hole_single_draft.json` 是待填写草稿，所有现场未知项为null；命令拒绝运行draft并保存ERROR。不得直接将合成样例参数抄成生产配置。

配置只能描述一个型号、一个视角、一个实体孔。`part_id`是工单期望标识，不是自动型号识别结果；`hand`声明配置左右件。调用方必须独立确认输入工件身份。

- `reference_image_size=[width,height]`，`coordinate_space=full_image`，`roi_xyxy=[x1,y1,x2,y2]`右下边界不包含。拒绝尺寸变化和自动缩放。
- `source_channel=fused/short/long`是曝光来源；`pixel_channel=gray/blue/green/red`是像素通道，两者不可混用。灰度转换遵循OpenCV的BGR输入。仅接受uint8灰度或三通道BGR全图。
- `source_kind=fused_only`仅允许fused；`hdr`表示调用方已有真实独立曝光来源记录，支持三种源。CLI不会通过文件名证明来源，调用方须核实采集清单。
- `expected_mask`是相对检查JSON目录的ROI尺寸二值PNG（0/255，非空，孔内部不接触ROI边界），定义应当开放区域H。不能用异常模型ignore mask。
- ROI同时是目标孔的唯一搜索区；需确认不含邻孔，并在association.evidence记录依据。算法不平移、旋转、扩大搜索，不用待检孔或最近邻孔配准。当前只支持独立固定夹具坐标，registration.evidence需指向重复取放验证记录。
- `segmentation`必须明确固定阈值、bright/dark极性、阈值扰动stability_delta及允许改变比例。阈值变化比例在H与两次扰动分割前景的并集中计算，含孔外影响几何判断的像素。没有去噪/形态学修复，不抹除小堵塞；形态学仅用于计算参考边缘指标。
- `limits`为人工验证的PASS和确定NG边界，两者之间必须保留REVIEW区间。所有长度为px，面积为px²，比例无单位；无物理标定不输出mm。

每次输入另需observation JSON：

```json
{
  "raw_unmasked": true,
  "fixture_verified": true,
  "observable": true,
  "evidence": "填写本次采集记录与独立定位、无夹具遮挡、反光可区分的核验依据"
}
```

这是**本次人工核验输入，不是自动遮挡或夹具识别能力**。未确认fixture_verified/observable/evidence则REVIEW；raw_unmasked未确认则ERROR。不能为批量通过固定填true。严重反光也可能稳定地越过阈值，阈值稳定性检测不能替代观测有效性确认。

## 指标与判定

B为专属ROI内阈值分割结果，H为预期内部mask：

- `open_fraction=|B∩H|/|H|`，`blocked_fraction=1-open_fraction`。
- `area_difference=abs(|B|-|H|)/|H|`；`shape_difference=1-IoU(B,H)`，位置不做补偿，保留实际偏差。
- `position_deviation_px`是B与H质心欧氏距离，B为空时null；保存面积、连通域数量及各域面积。
- `edge_coverage`为一像素参考内部边缘上分割开放像素比例，仅为图像指标，不是测得的真实孔壁覆盖率。
- `opening_observed`仅表示H内存在阈值开放像素；`physical_hole_presence=UNDETERMINED`，不能从无通光推断真实冲孔工艺。

ERROR：配置/源图合同错误。REVIEW：定位或观察条件不明、阈值扰动不稳定、分割接触搜索边界，或落在判定灰区。质量门控先于NG，遮挡不会被直接叫堵孔。

有效条件下，开放比例≤确定NG边界返回NG与 `HOLE_NOT_OPEN`、**“孔未正常开放，原因待确认”**。面积/形状/位置达到确定NG边界则报告开放几何异常。全部通过明确PASS边界才PASS；其余REVIEW。

`cause_uncertain=true`，第11/15项工艺原因均为NOT_DETERMINED：这是一个孔的共享几何事实，不伪造漏冲和堵孔两项确定原因，也不代表整个B4或第一阶段合格。

## 运行与证据

准备经确认的配置和observation后执行（以下路径是占位，不是本地已确认的真实参数）：

```bash
UV_CACHE_DIR=/tmp/b4-uv-cache uv run --no-sync python -m bmw_inspection.cli.check_hole \
  --config /path/to/confirmed_single_hole.json --image /path/to/raw_fused.png \
  --part-id CONFIRMED_PART --view-id CONFIRMED_VIEW \
  --source-channel fused --source-kind fused_only \
  --observation /path/to/observation.json --output /path/to/new_evidence_directory
```

退出码0=PASS、1=NG、2=ERROR、3=REVIEW。输出目录必须不存在；保存失败返回ERROR退出码，可能留下部分文件，不能当作完整证据。

输出result.json、summary.md，以及feature_id前缀的roi_original、selected_channel、expected_mask、open_mask、blocked_mask、overlay、full_overlay PNG。蓝线为预期边界，绿线为分割轮廓，红色为H内未开放像素，青框为原图ROI。JSON包含配置快照、来源路径、源/像素通道、本次观测声明、四态和指标；输入错误可能没有可生成的图像证据。

实际执行：

```bash
UV_CACHE_DIR=/tmp/b4-uv-cache uv run --no-sync python -m pytest tests/unit/bmw_inspection -q
UV_CACHE_DIR=/tmp/b4-uv-cache uv run --no-sync python tools/bmw/validate_hole.py \
  --output results/bmw_hole_b4_synthetic_20260910_v2
```

回归256 passed（包含钢印和原BMW测试），有既存NVML警告，无GPU/相机验证。合成CLI八场景全部符合预期：正常PASS；未开放、局部堵塞、只有邻孔NG；阈值不稳定、夹具遮挡、定位未确认REVIEW；遮蔽输入ERROR。结果入口 `results/bmw_hole_b4_synthetic_20260910_v2/summary.json`。所有样例标记synthesized_test_data=true，只证明软件合同，不是BMW真实缺陷检出率。

## 待提供与下一步

1. 首个真实型号/左右件映射、视角、孔feature_id，确认选用融合图或真实曝光源。
2. 该孔原图ROI、预期开放mask、排除邻孔的搜索边界及独立夹具定位验证。
3. 对应正常、确定未开放、边界、反光、夹具遮挡样件，保留物理工件/采集关联。
4. 分割极性和阈值、质量门控、开放比例/面积/形状/位置的PASS和NG边界。

取得上述资料后，先完成这个孔的真实离线验证与证据复核，再决定多个孔扩展。没有自动提交或推送。

## 后续真实图诊断（同日）

现已选取六张真实左件front图，完成右上圆孔候选的测量和存证，详见 `hole_real_review_20260910.md`。新增 `status=diagnostic`，在定位/验收标准未确认时允许显式null边界，只输出REVIEW。真实验收仍未完成；上文ready与draft合同保持。最新BMW回归270 passed。
