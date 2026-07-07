# Codex 软件改造任务书：正反面多视角工业缺陷检测融合框架

适用仓库：`https://github.com/wjstx0425/anomaly_xingtao`

目标：在现有 AnomalyDINO / PatchCore / EfficientAD / 几何模板 workflow 基础上，把项目升级为“质量门控 + 配准 + ROI 几何分支 + 外观异常分支 + 多视角/正反面融合 + 鲁棒性 benchmark”的工业检测软件框架。

> 这份文档是给 Codex 的实现指令。优先复用现有 `pipeline/` 和 `capture_data/` 代码，不要重写整个项目。不要提交数据、结果、checkpoint、训练日志。

---

## 0. 当前仓库事实与约束

### 0.1 当前已有能力

仓库已经有以下主要入口：

- `pipeline/1_collect_data.py`：采集原始大图。
- `pipeline/2_process_data.py`：裁剪单零件图。
- `pipeline/3_train_model.py`：训练与评估单模型。
- `pipeline/4_inference.py`：离线推理。
- `pipeline/5_demo_inspection.py`：现场双面 demo。
- `pipeline/6_compare_models.py`：比较 PatchCore / EfficientAD / AnomalyDINO。
- `pipeline/9_split_stress_normal.py`：stress normal 按 group 拆分 train / locked。
- `pipeline/10_build_hardened_dataset.py`：clean normal + stress_train normal 生成 hardened dataset。
- `pipeline/11_build_geometry_templates.py` 到 `15_edit_geometry_masks.py`：几何模板、manual mask、stress calibration、defect fusion workflow。

`pipeline/` 里的很多文件只是 wrapper，实际实现主要在 `capture_data/` 下。修改时优先查找并复用：

- `capture_data/inference.py`
- `capture_data/demo_inspection.py`
- `capture_data/geometry_shape.py`
- `capture_data/evaluate_geometry_shape.py`
- `capture_data/build_geometry_templates.py`
- `capture_data/export_geometry_review_pack.py`
- `capture_data/edit_geometry_masks.py`
- `capture_data/prepare_part_crops.py`
- `examples/api/03_models/zs32_defect_workflow.py`

### 0.2 关键实验现状

历史记录中已有以下结果，应作为 regression baseline：

- AnomalyDINO defect recall：`8/17`。
- active manual geometry + AnomalyDINO fused recall：`15/17`。
- active manual geometry stress locked FP：`0/90`。
- active fused misses：`less_1_2_slot02`、`corner_3_1_slot05`。
- high-exposure corner-only fused recall：`14/17`，stress locked FP：`0/90`，但漏 `less_2_1_slot03`、`more_2_2_slot04`、`crack_3_2_slot06`。

因此新代码的首要目标不是只重跑 AnomalyDINO，而是：

1. 保持 stress normal FP 尽量为 `0/90` 或不劣化。
2. 尽量超过 fused recall `15/17`。
3. 优先救回：
   - `less_1_2_slot02`
   - `corner_3_1_slot05`
4. 不要让已经被 geometry 救回的样本重新漏掉：
   - `less_2_1_slot03`
   - `more_2_2_slot04`
5. 允许新增 `RETAKE / INVALID_CAPTURE / SUSPECT` 状态，不要把不可检输入强行判 OK/NG。

### 0.3 操作约束

- 当前 workflow 优先使用 `.venv/bin/python`。
- 不要 overwrite 现有 manual masks，尤其不要覆盖：
  - `results/c789_100_hardened/left_top_geometry/manual_review_pack/manual_masks`
- 如果要生成新模板或新实验输出，写到新的目录，例如：
  - `results/c789_100_hardened/left_top_geometry/v2_*`
  - `results/c789_100_hardened/left_top_multiview/*`
- Pytest 可能不可用。必须至少支持：
  - `.venv/bin/python -m compileall capture_data pipeline`
  - 用临时目录和合成小图跑 CLI dry-run / smoke test。
- 不要提交：
  - `dataset/`
  - `datasets/`
  - `results/`
  - `c789_bottom/`
  - `*.ckpt`
  - `wandb/`
  - `lightning_logs/`
  - `mlruns/`

---

## 1. 最终软件架构

目标流水线：

```text
raw capture images
  ↓
image completeness check
  ↓
quality gate: brightness / saturation / blur / highlight / coverage / pose
  ↓
part registration to reference coordinate
  ↓
ROI crop by side / slot / view
  ↓
branch detectors:
  A. geometry branch for less / more / corner
  B. AnomalyDINO ROI or full-crop branch for surface / local boundary
  C. EfficientAD / PatchCore optional branch for appearance/global anomaly
  D. crack branch using dark-field or line-enhanced view, optional v1
  ↓
fusion engine: OK / NG_* / SUSPECT / RETAKE
  ↓
review artifacts + CSV + JSON trace + robustness report
```

输出状态必须至少支持：

```text
OK
NG_GEOMETRY
NG_ANOMALY
NG_CRACK
NG_GLOBAL
SUSPECT
RETAKE
INVALID_CAPTURE
```

`OK` 必须严格：图像完整、质量通过、配准通过、正反面关键 ROI 均通过、所有分支低于阈值。

`NG` 可以敏感：任一可靠分支明确异常即可 NG。

`RETAKE / INVALID_CAPTURE` 用于硬件采集异常、光照异常、偏位、失焦、缺图等情况。

---

## 2. 新增配置文件

新增目录：

```text
config/
  inspection_profiles/
    c789_multiview.yaml
    fx11_multiview.yaml          # optional
  quality_gate/
    c789.yaml
  fusion/
    c789.yaml
  registration/
    c789.yaml
```

### 2.1 `config/inspection_profiles/c789_multiview.yaml`

示例：

```yaml
part_profile: c789
sides:
  top:
    aliases: [top, A, front]
    required_views: [uniform, left_bar, right_bar]
    optional_views: [high_exp, low_exp, darkfield]
    slots: [slot01, slot02, slot03, slot04, slot05, slot06]
  bottom:
    aliases: [bottom, B, back]
    required_views: [uniform, left_bar, right_bar]
    optional_views: [high_exp, low_exp, darkfield]
    slots: [slot01, slot02, slot03, slot04, slot05, slot06]

naming:
  # 新采集建议命名：part0001_top_uniform_g001_000001.png
  regex: "(?P<part_id>part\\d+)_(?P<side>top|bottom|A|B|front|back)_(?P<view>uniform|left_bar|right_bar|high_exp|low_exp|darkfield).*\\.(png|jpg|jpeg|bmp)$"

branch_policy:
  geometry:
    enabled: true
    views: [uniform, high_exp]
    target_defects: [less, more, corner]
  anomaly_dino:
    enabled: true
    views: [uniform]
    target_defects: [surface, crack, corner]
  efficient_ad:
    enabled: false
    views: [uniform]
  crack:
    enabled: false
    views: [left_bar, right_bar, darkfield]
```

### 2.2 `config/quality_gate/c789.yaml`

示例：

```yaml
version: 1
mode: fail
metrics:
  brightness_mean:
    min: 30
    max: 230
  brightness_std:
    min: 5
    max: 90
  saturation_ratio:
    max: 0.02
  dark_ratio:
    max: 0.05
  blur_laplacian_var:
    min: 80
  highlight_ratio:
    max: 0.03
  foreground_coverage:
    min: 0.20
    max: 0.95
  registration:
    max_abs_dx_px: 8
    max_abs_dy_px: 8
    max_abs_angle_deg: 1.0
    min_score: 0.5
```

这些默认值只是占位。后续由 `pipeline/17_calibrate_quality_gate.py` 从 normal / stress normal 统计生成。

### 2.3 `config/fusion/c789.yaml`

示例：

```yaml
version: 1
ok_requires:
  quality_gate: PASS
  registration: PASS
  required_sides: [top, bottom]
  required_views: [uniform]

branch_order:
  - geometry
  - crack
  - anomaly_dino
  - efficient_ad

rules:
  geometry:
    status_on_positive: NG_GEOMETRY
    min_confidence: 0.0
  anomaly_dino:
    status_on_positive: NG_ANOMALY
    min_confidence: 0.0
  crack:
    status_on_positive: NG_CRACK
    min_confidence: 0.0
  efficient_ad:
    status_on_positive: NG_GLOBAL
    min_confidence: 0.0

suspect_policy:
  enable: true
  near_threshold_ratio: 0.90
  conflicting_branches: SUSPECT
```

---

## 3. 新增核心模块

### 3.1 `capture_data/quality_gate.py`

实现图像质量门控。不要依赖深度学习。

必须提供 dataclass：

```python
@dataclass(frozen=True)
class ImageQualityMetrics:
    image_path: str
    side: str | None
    view: str | None
    brightness_mean: float
    brightness_std: float
    saturation_ratio: float
    dark_ratio: float
    blur_laplacian_var: float
    highlight_ratio: float
    foreground_coverage: float | None = None

@dataclass(frozen=True)
class QualityGateResult:
    status: str  # PASS / WARN / FAIL
    reasons: list[str]
    metrics: ImageQualityMetrics
```

必须实现函数：

```python
def compute_quality_metrics(image_path: Path, *, foreground_mask: np.ndarray | None = None) -> ImageQualityMetrics: ...

def evaluate_quality_gate(metrics: ImageQualityMetrics, config: Mapping[str, Any]) -> QualityGateResult: ...

def write_quality_gate_csv(results: Sequence[QualityGateResult], output_csv: Path) -> None: ...
```

质量指标定义：

- `brightness_mean`：灰度均值。
- `brightness_std`：灰度标准差。
- `saturation_ratio`：任一通道接近 255 的像素比例，默认阈值像素值 `>=250`。
- `dark_ratio`：灰度 `<=5` 的像素比例。
- `blur_laplacian_var`：灰度图 Laplacian variance。
- `highlight_ratio`：灰度高于 `mean + 2.5*std` 且亮度高的高光比例，或使用配置阈值。
- `foreground_coverage`：可选，若有 foreground mask 则为 mask 占比。

注意：

- 缺图、读图失败、尺寸异常直接 `FAIL`。
- `status=FAIL` 时，不应进入模型推理。
- 支持 `mode=warn` 时返回 `WARN` 但允许继续，用于实验阶段。

### 3.2 `capture_data/part_registration.py`

实现轻量配准，不追求一次到位，但必须输出诊断信息。

必须提供 dataclass：

```python
@dataclass(frozen=True)
class RegistrationResult:
    status: str  # PASS / WARN / FAIL
    image_path: str
    reference_path: str | None
    transform: list[list[float]] | None
    dx_px: float | None
    dy_px: float | None
    angle_deg: float | None
    scale: float | None
    score: float | None
    reason: str | None
```

优先实现 OpenCV ECC 或 ORB/homography 的简化版本：

```python
def register_to_reference(image: np.ndarray, reference: np.ndarray, config: Mapping[str, Any]) -> tuple[np.ndarray, RegistrationResult]: ...
```

最低要求：

- 支持 identity fallback。
- 支持 `--registration off`。
- 支持把配准后图像保存到 `registered/`。
- 记录 `registration_results.csv`。
- 配准失败不要崩溃，返回 `FAIL` 并交给 fusion 判 `RETAKE`。

建议先做 translation/rotation small-motion 配准，不要一开始做复杂 3D 或深度模型。

### 3.3 `capture_data/multiview_manifest.py`

用于把多张图片按 `part_id / side / view` 分组。

必须支持两种输入：

1. 按文件名 regex 自动解析。
2. 显式 CSV manifest。

CSV schema：

```text
part_id,side,view,image_path,label,defect_type,slot_id,group_id,notes
```

必须提供：

```python
@dataclass(frozen=True)
class MultiViewImageRecord:
    part_id: str
    side: str
    view: str
    image_path: Path
    label: str | None = None
    defect_type: str | None = None
    slot_id: str | None = None
    group_id: str | None = None

@dataclass(frozen=True)
class MultiViewPartRecord:
    part_id: str
    images: tuple[MultiViewImageRecord, ...]
```

函数：

```python
def load_multiview_manifest(input_root: Path, config: Mapping[str, Any], *, manifest_csv: Path | None = None) -> list[MultiViewPartRecord]: ...

def validate_required_views(part_record: MultiViewPartRecord, config: Mapping[str, Any]) -> tuple[bool, list[str]]: ...
```

缺少 required side/view 时，fusion 输出 `INVALID_CAPTURE`。

### 3.4 `capture_data/fusion_engine.py`

实现规则融合。不要训练融合网络。

必须提供 dataclass：

```python
@dataclass(frozen=True)
class BranchPrediction:
    part_id: str
    side: str
    view: str | None
    slot_id: str | None
    branch: str  # geometry / anomaly_dino / efficient_ad / crack / quality / registration
    pred_label: int
    score: float | None
    threshold: float | None
    defect_type: str | None
    reason: str | None
    source_path: str | None

@dataclass(frozen=True)
class FusedDecision:
    part_id: str
    final_status: str
    final_label: int | None
    defect_side: str | None
    defect_view: str | None
    defect_slot: str | None
    defect_type: str | None
    triggered_branch: str | None
    reason: str
```

融合逻辑：

```python
if any required image missing:
    INVALID_CAPTURE
elif quality gate FAIL:
    RETAKE
elif registration FAIL:
    RETAKE
elif any geometry pred_label == 1:
    NG_GEOMETRY
elif any crack pred_label == 1:
    NG_CRACK
elif any anomaly_dino pred_label == 1:
    NG_ANOMALY
elif any efficient_ad pred_label == 1:
    NG_GLOBAL
elif any branch near threshold and suspect_policy enabled:
    SUSPECT
else:
    OK
```

必须保留所有分支的原始分数。不要只输出最终 label。

输出 CSV：

```text
fused_predictions.csv
part_id,final_status,final_label,defect_side,defect_view,defect_slot,defect_type,triggered_branch,reason

branch_predictions.csv
part_id,side,view,slot_id,branch,pred_label,score,threshold,defect_type,reason,source_path
```

### 3.5 `capture_data/crack_branch.py`，可选 v1

如果时间有限，先实现为可关闭的简单传统视觉分支。

输入：暗场 / 条形光图或普通图。

最小实现：

- 灰度化。
- Top-hat 或 black-hat。
- Gabor 或 Frangi 可选。
- Canny / adaptive threshold。
- 连通域筛选：面积、长度、细长比、skeleton length。

输出 `BranchPrediction`。

默认 `enabled=false`，确保不影响已有 pipeline。

---

## 4. 修改现有几何分支

现有 geometry workflow 已经很有价值。不要推倒重来。重点修复两个问题。

### 4.1 增加几何阈值 fallback

已知 caveat：当前 geometry threshold calibration 是按 `slot/type/coarse region` 做的。如果 defect 落在 stress-normal calibration 没见过的 region/type，可能被忽略或阈值过宽。

在 `capture_data/evaluate_geometry_shape.py` 或 `capture_data/geometry_shape.py` 里实现阈值 fallback：

阈值查找顺序：

```text
1. exact: slot_id + defect_type + region_id
2. slot_type: slot_id + defect_type + region='*'
3. slot_region: slot_id + defect_type='*' + region_id
4. slot: slot_id + defect_type='*' + region='*'
5. defect_type: slot_id='*' + defect_type + region='*'
6. global: slot_id='*' + defect_type='*' + region='*'
```

`geometry_thresholds.csv` 需要新增或兼容以下列：

```text
slot_id,defect_type,region_id,threshold,threshold_source,n_normal,max_normal,p99_normal,p999_normal
```

如果旧 CSV 没有 fallback 行，加载时自动补：

- per slot fallback：取该 slot 所有 region/type 正常分数的高分位或最大值。
- global fallback：取所有正常分数的高分位或最大值。

在输出 `geometry_predictions.csv` 里新增：

```text
threshold_source
threshold_lookup_level
```

这样可以解释为什么某个缺陷被判 NG 或漏掉。

### 4.2 增加 less / more 专用 score

当前 some less 被识别为 type `more`，说明 scoring 语义不够清晰。不要强制改历史输出，但新增更可解释的指标：

```text
missing_area
extra_area
missing_in_watch_edge
extra_in_watch_edge
edge_offset_max
edge_offset_mean
contour_chamfer_distance
```

建议：

- `less_score = missing_in_watch_edge + edge_inward_offset_weighted`
- `more_score = extra_in_watch_edge + edge_outward_offset_weighted`
- `geometry_score = max(less_score / less_threshold, more_score / more_threshold, corner_score / corner_threshold)`

短期实现可以先把 `missing_in_watch_edge`、`extra_in_watch_edge` 输出出来，不必马上改所有模板。

### 4.3 less_1_2_slot02 专项 rescue hook

不要硬编码文件名判 NG。允许通过配置对特定 slot 增强 watch edge 或降低 fallback 阈值。

新增配置示例：

```yaml
geometry_overrides:
  slot02:
    less:
      watch_edge_dilation: 3
      threshold_scale: 0.75
      min_missing_area: 120
  slot03:
    less:
      threshold_scale: 0.90
  slot04:
    more:
      threshold_scale: 0.90
```

原则：

- 不能只针对一个 defect 文件名。
- 必须用 stress locked normal 验证 FP 不劣化。
- override 必须写入输出 CSV 的 `threshold_source` 或 `reason`。

---

## 5. 新增 pipeline wrapper

### 5.1 `pipeline/16_build_multiview_manifest.py`

功能：扫描多视角采集目录，生成 manifest。

CLI：

```bash
.venv/bin/python pipeline/16_build_multiview_manifest.py \
  --input-root dataset/c789_multiview_raw \
  --profile config/inspection_profiles/c789_multiview.yaml \
  --output-csv results/c789_multiview/manifest.csv
```

实际实现可以在 `capture_data/multiview_manifest.py`，pipeline 文件只做 wrapper。

### 5.2 `pipeline/17_calibrate_quality_gate.py`

功能：从 clean normal / stress normal / invalid input 统计质量门控阈值。

CLI：

```bash
.venv/bin/python pipeline/17_calibrate_quality_gate.py \
  --normal-root dataset/c789_100_left_top_hardened_parts \
  --stress-root dataset/c789_stress_normal_group_split/locked/left/top \
  --invalid-root dataset/c789_invalid_input_optional \
  --profile config/inspection_profiles/c789_multiview.yaml \
  --output-yaml config/quality_gate/c789_calibrated.yaml \
  --output-report results/c789_quality_gate/calibration_report.md
```

输出：

- `quality_metrics.csv`
- `c789_calibrated.yaml`
- `calibration_report.md`

阈值建议：

- 对正常/stress normal：取 `P0.5` 到 `P99.5` 或 `min/max` 加 margin。
- invalid input 用于检查阈值能否拒绝明显过曝、偏位、模糊，不参与 OK training。

### 5.3 `pipeline/18_fuse_inspection_results.py`

功能：融合 geometry/anomaly/quality/registration 结果。

CLI：

```bash
.venv/bin/python pipeline/18_fuse_inspection_results.py \
  --manifest results/c789_multiview/manifest.csv \
  --quality-csv results/c789_multiview/quality_gate.csv \
  --registration-csv results/c789_multiview/registration_results.csv \
  --geometry-csv results/c789_100_hardened/left_top_geometry/manual_defect_fused/geometry_predictions.csv \
  --anomaly-csv results/c789_100_hardened/left_top_anomaly_dino/reports/predictions.csv \
  --fusion-config config/fusion/c789.yaml \
  --output-dir results/c789_multiview/fused
```

要求：

- 支持缺少某个 branch CSV 时自动跳过该分支，但记录 warning。
- 支持 path 匹配：`source_path`、`processed_path`、`image_path`、basename。
- 输出 `branch_predictions.csv`、`fused_predictions.csv`、`summary.md`。

### 5.4 `pipeline/19_run_robustness_benchmark.py`

功能：统一跑 clean normal、stress normal、defect、invalid input benchmark。

CLI：

```bash
.venv/bin/python pipeline/19_run_robustness_benchmark.py \
  --clean-normal-root dataset/c789_100_left_top_parts/left/top/normal_test \
  --stress-normal-root dataset/c789_stress_normal_group_split/locked/left/top \
  --defect-root dataset/c789_100_left_top_parts/left/top/defect \
  --invalid-root dataset/c789_invalid_input_optional \
  --geometry-template-dir results/c789_100_hardened/left_top_geometry/manual_templates \
  --geometry-thresholds results/c789_100_hardened/left_top_geometry/manual_stress_locked/geometry_thresholds.csv \
  --anomaly-predictions results/c789_100_hardened/left_top_anomaly_dino/reports/predictions.csv \
  --fusion-config config/fusion/c789.yaml \
  --output-dir results/c789_robustness_v2
```

输出：

```text
robustness_summary.md
robustness_summary.csv
by_defect_type.csv
by_slot.csv
misses.csv
false_positives.csv
retake_cases.csv
```

必须报告：

- clean normal FP
- stress normal FP
- invalid input reject rate
- defect recall
- less recall
- more recall
- corner recall
- surface recall
- crack recall
- per-slot recall
- fused recall vs AnomalyDINO-only vs geometry-only

---

## 6. 修改现场 demo

修改 `capture_data/demo_inspection.py`，不要破坏旧参数。

新增参数：

```bash
--inspection-profile config/inspection_profiles/c789_multiview.yaml
--quality-config config/quality_gate/c789.yaml
--fusion-config config/fusion/c789.yaml
--registration-config config/registration/c789.yaml
--quality-gate fail|warn|off
--registration on|off
--multi-view-mode sequential|single
--view-sequence uniform,left_bar,right_bar
--save-branch-traces
```

现场状态机改成：

```text
WAIT_LOAD
  ↓
CAPTURE_TOP_UNIFORM / TOP_LEFT_BAR / TOP_RIGHT_BAR
  ↓
QUALITY_TOP
  ↓
INFER_TOP_BRANCHES
  ↓
WAIT_FLIP_OR_AUTO_FLIP
  ↓
CAPTURE_BOTTOM_UNIFORM / BOTTOM_LEFT_BAR / BOTTOM_RIGHT_BAR
  ↓
QUALITY_BOTTOM
  ↓
INFER_BOTTOM_BRANCHES
  ↓
FUSE_PART_DECISION
  ↓
SHOW_OK_NG_RETAKE
```

对于当前硬件仍是单相机 + 人工/半自动翻面，软件只需要识别当前 side：

- 用户按键确认 `top` / `bottom`；或者
- 后续由翻面机构的传感器串口给出 `SIDE_TOP_READY` / `SIDE_BOTTOM_READY`。

新增可选串口接口不要强依赖硬件：

```bash
--flip-controller-serial COM3
--flip-controller-baudrate 115200
--mock-flip-controller
```

串口协议 v1：

```text
PC -> controller: GET_STATE\n
controller -> PC: STATE TOP_LOCKED\n
controller -> PC: STATE BOTTOM_LOCKED\n
controller -> PC: STATE MOVING\n
controller -> PC: ERROR NOT_LOCKED\n
```

如果没有串口，旧版手动流程继续可用。

归档输出新增：

```text
trace.json
quality_gate.csv
registration_results.csv
branch_predictions.csv
fused_predictions.csv
annotated_top.png
annotated_bottom.png
```

UI 上必须显示：

```text
OK / NG_GEOMETRY / NG_ANOMALY / NG_CRACK / RETAKE / INVALID_CAPTURE / SUSPECT
side
slot
branch
score / threshold
quality reason
registration reason
```

---

## 7. ROI AnomalyDINO 改造建议

短期不要重训复杂模型。先做结果融合和 ROI 后处理。

新增或修改推理后处理，使 AnomalyDINO 输出支持：

```text
image_score
roi_score
roi_p99_score
roi_topk_mean_score
roi_max_score
connected_component_area_max
connected_component_length_max
```

建议在 `capture_data/inference.py` 中新增参数：

```bash
--roi-config config/roi/c789_slots.yaml
--roi-score-mode full|max|p99|topk|masked_max
--save-roi-scores
```

ROI 配置示例：

```yaml
rois:
  slot02:
    x1: 100
    y1: 200
    x2: 500
    y2: 360
  slot03:
    x1: 100
    y1: 360
    x2: 500
    y2: 520
```

如果当前 inference heatmap 不容易取出，先只实现 path-level CSV 融合，不阻塞主线。

---

## 8. 数据组织与命名建议

新采集数据建议采用：

```text
dataset/c789_multiview_raw/
  part0001/
    part0001_top_uniform_g001_000001.png
    part0001_top_left_bar_g001_000001.png
    part0001_top_right_bar_g001_000001.png
    part0001_bottom_uniform_g001_000001.png
    part0001_bottom_left_bar_g001_000001.png
    part0001_bottom_right_bar_g001_000001.png
    meta.json
```

`meta.json` 示例：

```json
{
  "part_id": "part0001",
  "label": "normal",
  "defect_type": null,
  "slot_id": null,
  "operator": "",
  "fixture_version": "flip_jig_v1",
  "camera": "top_cam_0",
  "lights": ["uniform", "left_bar", "right_bar"],
  "exposure_us": 12000,
  "gain": 0,
  "notes": ""
}
```

原则：

- 按 `part_id` 切 train/val/test，不按图片切。
- stress normal 只能包含上线允许范围内的扰动。
- 过曝、严重偏位、失焦、遮挡应该进 `invalid_input`，不要当 normal 扩充。

---

## 9. 最小可交付版本排序

Codex 按下面顺序实现，不要一次性改太多。

### MVP-1：融合引擎 + benchmark

新增：

- `capture_data/fusion_engine.py`
- `pipeline/18_fuse_inspection_results.py`
- `pipeline/19_run_robustness_benchmark.py`

目标：能读取现有 geometry/anomaly CSV，输出统一 fused 结果和 benchmark summary。

验收：

```bash
.venv/bin/python -m compileall capture_data pipeline
.venv/bin/python pipeline/18_fuse_inspection_results.py --help
.venv/bin/python pipeline/19_run_robustness_benchmark.py --help
```

并用 synthetic CSV 测试：

- quality fail -> RETAKE
- geometry positive -> NG_GEOMETRY
- anomaly positive -> NG_ANOMALY
- no positive -> OK
- missing required view -> INVALID_CAPTURE

### MVP-2：几何阈值 fallback

修改现有 geometry eval。

验收：用 toy thresholds CSV 验证 exact 缺失时能 fallback 到 slot/global，不会静默忽略异常。

必须在输出 CSV 里能看到：

```text
threshold_source
threshold_lookup_level
```

### MVP-3：quality gate

新增：

- `capture_data/quality_gate.py`
- `pipeline/17_calibrate_quality_gate.py`

验收：用 synthetic images 生成：

- 正常灰度图 -> PASS
- 全白过曝图 -> FAIL saturation
- 全黑欠曝图 -> FAIL dark
- 强模糊图 -> FAIL blur

### MVP-4：demo 接入

修改 `capture_data/demo_inspection.py`，把 quality/fusion 状态接入 UI 和 trace。必须保持旧 demo 命令可运行。

### MVP-5：multiview manifest + sequential capture

新增：

- `capture_data/multiview_manifest.py`
- `pipeline/16_build_multiview_manifest.py`

支持正反面、多光照命名和 grouping。

### MVP-6：registration

新增 `capture_data/part_registration.py` 并接入 quality/fusion。先允许 `--registration off`，不要阻塞前面功能。

---

## 10. 推荐 commit 切分

```text
commit 1: add fusion_engine and fuse wrapper
commit 2: add robustness benchmark summary
commit 3: add geometry threshold fallback and CSV diagnostics
commit 4: add quality_gate module and calibration wrapper
commit 5: wire quality/fusion into demo_inspection without breaking old CLI
commit 6: add multiview manifest parser and docs
commit 7: add registration module behind off-by-default flag
commit 8: update pipeline/README.md with new stages 16-19
```

---

## 11. 文档更新

更新：

- `pipeline/README.md`
- `README.md` 的 Custom Industrial Defect Pipeline 部分
- `CHANGELOG.md`

新增 section：

```text
## 6. 工业融合检测 v2
- Quality gate
- Registration
- Multi-view manifest
- Geometry + AnomalyDINO fusion
- Robustness benchmark
- Demo states: OK / NG / RETAKE / SUSPECT
```

文档中明确：

- `RETAKE` 不是误报，而是采集不可判。
- stress normal locked 不能进入训练或阈值调参。
- defect_test 不能反复调阈值。
- 数据、结果、checkpoint 不提交 Git。

---

## 12. 关键验收指标

最终 PR 至少要让以下命令可执行或可 dry-run：

```bash
.venv/bin/python -m compileall capture_data pipeline

.venv/bin/python pipeline/16_build_multiview_manifest.py --help
.venv/bin/python pipeline/17_calibrate_quality_gate.py --help
.venv/bin/python pipeline/18_fuse_inspection_results.py --help
.venv/bin/python pipeline/19_run_robustness_benchmark.py --help

.venv/bin/python pipeline/5_demo_inspection.py \
  --part-profile c789 \
  --demo-top-image dataset/c789/left/top/normal/example.png \
  --demo-bottom-image dataset/c789/left/bottom/normal/example.png \
  --mock-predictions \
  --auto-run \
  --no-gui
```

如果本地没有数据，必须提供 synthetic smoke test 或 `--dry-run`。不要因为缺少真实数据让基本 CLI 崩溃。

最终 benchmark 报告必须包含：

```text
AnomalyDINO only recall
Geometry only recall
Fused recall
Clean normal FP
Stress locked normal FP
Invalid input reject rate, if invalid data exists
Missed defects list
False positives list
Per-defect-type recall
Per-slot recall
```

---

## 13. 不要做的事

- 不要把手工 mask 直接覆盖掉。
- 不要把 defect test 用来反复调阈值后再声称是测试结果。
- 不要为了 recall 直接降低全局阈值导致 stress normal FP 大量上升。
- 不要把过曝、失焦、严重偏位当作 normal 加入 hardened dataset。
- 不要训练一个复杂融合网络；当前缺陷样本太少，先用规则融合。
- 不要把 PatchCore、EfficientAD、AnomalyDINO 分数简单平均。应按 branch OR 规则融合。
- 不要破坏旧命令行参数和旧 pipeline 行为。

---

## 14. 给 Codex 的首个具体任务

先完成 MVP-1 和 MVP-2：

1. 新建 `capture_data/fusion_engine.py`。
2. 新建 `pipeline/18_fuse_inspection_results.py` wrapper。
3. 新建 `pipeline/19_run_robustness_benchmark.py` 的基础版本。
4. 给 geometry threshold lookup 增加 fallback，不改变原有 CSV 也能兼容。
5. 加 synthetic CSV/image-free smoke test 或 CLI dry-run。
6. 更新 `pipeline/README.md`，只写新增命令和最小示例。
7. 跑：

```bash
.venv/bin/python -m compileall capture_data pipeline
.venv/bin/python pipeline/18_fuse_inspection_results.py --help
.venv/bin/python pipeline/19_run_robustness_benchmark.py --help
```

优先保证：现有 workflow 不坏；新融合结果可解释；每个 final decision 都能追溯到 branch、side、slot、score、threshold、reason。
