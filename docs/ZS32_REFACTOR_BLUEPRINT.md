# ZS32 多算法多视角视觉检测系统重构蓝图

状态：设计冻结稿（不包含代码重构）
适用产品：ZS32
目标：将当前多代实验脚本重构为可扩展、可校准、可审计、fail-closed 的多视角检测系统。

## 1. 已冻结的产品决策

本蓝图以下列约束为准，后续实现不得自行改变：

1. 系统只支持 ZS32，不再保留 C789、FX11 兼容性。
2. 同时支持左手件和右手件；左右手分别维护 ROI、模板和无监督模型。
3. 当前为三个相机、正反两轮采集，共六个必需视角；未来可能扩为四个或五个相机，对应八个或十个必需视角。
4. 所有相机视角都是必需视角。任一视角缺失、重复、身份不一致或来源冲突时，不允许继续检测。
5. Phase 0/parity 前 `pipeline/1_collect_multicamera_data.py` 保持字节不变作为 legacy
   replay 入口；正式新链路使用 `pipeline/zs32_capture.py`。只有 Linux golden parity
   通过后，旧文件才允许改成调用同一配置驱动 capture service 的兼容薄包装。
6. 每个物理零件具有唯一 `part_instance_id`；一次拍摄每个相机只保存一张图。
7. 每个 `(hand, topology, view)` 只维护一份权威 ROI，坐标以 YOLO ROI 为准。
8. 模板匹配、YOLO、PatchCore、EfficientAD、AnomalyDINO 全部消费同一份 ROI crop；模型只允许 resize/normalize，不允许再次做空间裁剪。
9. 每个部署版本只允许启用一种无监督算法：PatchCore、EfficientAD、AnomalyDINO 三选一，同一部署版本禁止按视角混搭算法族。
10. YOLO 使用一套全局模型覆盖左右手和全部视角；训练位于外部工程，本仓库只导入部署产物。
11. 第一层当前只保留模板匹配。任一视角模板不匹配，直接输出 `NG_TEMPLATE` 并短路后续模型；模板没有 REVIEW 区间。
12. 图像质量异常或配准失败输出 `RETAKE`，要求重新采集。
13. 第二层为 `YOLO + 已选择的无监督算法`。只有所有必需视角的两个分支都为 CLEAR，零件才可输出 OK。
14. 任一 STRONG 证据直接 NG；任一 GRAY 证据进入 REVIEW；采集身份错误进入 INVALID_CAPTURE。
15. 模板、权重、阈值、配置或运行时损坏新增明确的 `SYSTEM_ERROR`，不得误判为零件 NG，也不得要求无意义的重新采集。
16. 允许删除旧接口、旧脚本和旧产品实现；重构不承担向后兼容义务。
17. Linux + NVIDIA GPU 主机是唯一权威执行环境；Mac 只用于编辑代码和文档，不在 Mac 上运行测试、数据处理、ROI 生成、推理、标定、相机验证或部署。

### 1.1 开发与执行环境边界

| 环境 | 允许 | 禁止作为验收证据 |
| --- | --- | --- |
| Mac | 编辑代码、配置和文档；查看静态文件；通过 Git 传递变更 | 任何测试结果、数据集生成结果、ROI 处理结果、模型推理结果、标定结果、性能结果、相机结果和部署结论 |
| Linux + NVIDIA GPU | Phase 0 冻结、数据治理、ROI 生成、全部测试、相机验证、模型训练、批量标定、正式推理、性能测试和部署 | 无；这是本项目唯一权威运行与验收环境 |

Mac 上不得因为缺少 GPU、Hikvision SDK、Linux 路径或模型资产而加入 mock fallback、降低校验强度或改变生产逻辑。所有运行命令、验收报告和 golden outputs 必须由 Linux 主机生成，并携带 Linux 环境、GPU、驱动、CUDA、Python 和依赖锁信息。

## 2. 当前 ROI 资产状态

当前权威 ROI 输入来自用户提供的 `roi_config.json`；仓库内只保存其经审核的
规范化配置 `configs/zs32/roi/zs32_roi_v2.json`，不依赖任一开发机的下载目录。

- 原图尺寸：`4024 x 3036`
- 坐标语义：`pixel_xyxy_half_open`
- 当前文件实际只包含右手六视角；参考图路径全部为 `right_*`
- 左手 ROI 尚未确定，因此第一版重构可以支持左手 schema、训练和运行接口，但在左手 ROI 补齐前必须禁止生成左手部署 release

当前右手 ROI：

| view | ROI `[x1,y1,x2,y2]` | crop size |
| --- | --- | --- |
| front | `[150,1020,3910,2520]` | `3760 x 1500` |
| front_left | `[0,0,3910,2420]` | `3910 x 2420` |
| front_right | `[570,260,3390,2150]` | `2820 x 1890` |
| back | `[390,930,4020,2510]` | `3630 x 1580` |
| back_left | `[400,0,4024,2540]` | `3624 x 2540` |
| back_right | `[410,20,3500,2330]` | `3090 x 2310` |

目标 ROI 文件应升级为 hand-aware schema：

```yaml
schema_version: 2
roi_version: zs32-roi-v2
product: ZS32
topology_id: zs32-3cam-double-side-v1
coordinate_system: pixel_xyxy_half_open
source_image_size: {width: 4024, height: 3036}
hands:
  right:
    status: ready
    views:
      front: {xyxy: [150, 1020, 3910, 2520]}
      front_left: {xyxy: [0, 0, 3910, 2420]}
      front_right: {xyxy: [570, 260, 3390, 2150]}
      back: {xyxy: [390, 930, 4020, 2510]}
      back_left: {xyxy: [400, 0, 4024, 2540]}
      back_right: {xyxy: [410, 20, 3500, 2330]}
  left:
    status: pending
    views: {}
```

发布检查必须保证 recipe 中启用的每个 `(hand, view)` 均存在合法 ROI。`status: pending` 不允许被默认值、镜像坐标或右手坐标静默替代。

## 3. 目标系统分层

```text
CLI / Service
    ↓
Application Workflows
    ↓
Domain Contracts + State Machine
    ↓
Capture / ROI / Gates / Model Adapters / Calibration / Fusion
    ↓
Hikvision SDK / Anomalib / Ultralytics / OpenCV / Filesystem
```

### 3.1 Domain 层

Domain 层只定义稳定业务语义，不依赖 Anomalib、Ultralytics、OpenCV 或相机 SDK：

- `PartIdentity`
- `CaptureTopology`
- `CaptureSet`
- `ViewImage`
- `RoiSample`
- `ModelEvidence`
- `TemplateEvidence`
- `ThresholdRecord`
- `InspectionDecision`
- `DeploymentContract`
- `ReleaseManifest`

### 3.2 Application 层

Application 层编排完整用例：

- 采集一个物理零件
- 从 raw data 构建 canonical ROI dataset release
- 训练一种无监督算法
- 导入外部 YOLO 模型
- 训练模板门禁
- 收集连续分数并校准阈值
- 组装不可变部署 release
- 执行一次在线多视角检测

### 3.3 Adapter 层

外部依赖均通过 Adapter 隔离：

- `HikvisionCameraAdapter`
- `PatchCoreAdapter`
- `EfficientADAdapter`
- `AnomalyDINOAdapter`
- `UltralyticsYoloAdapter`
- `OpenCVTemplateAdapter`
- `FilesystemArtifactStore`

## 4. 推荐目录结构

```text
src/zs32_inspection/
  domain/
    identity.py
    topology.py
    evidence.py
    decisions.py
    contracts.py
    errors.py
  config/
    schemas.py
    loaders.py
    compiler.py
  capture/
    service.py
    hikvision.py
    storage.py
    quality.py
    registration.py
  data/
    calibration_targets.py
    manifests.py
    splitter.py
    roi.py
    dataset_release.py
    yolo_export.py
    anomalib_export.py
  template/
    trainer.py
    predictor.py
    artifacts.py
  models/
    base.py
    patchcore.py
    efficientad.py
    anomalydino.py
    yolo.py
    registry.py
  calibration/
    acceptance.py
    scoring.py
    thresholds.py
    reports.py
  fusion/
    policy.py
    engine.py
    completeness.py
  runtime/
    orchestrator.py
    release_loader.py
    audit.py
    publisher.py
  cli/
    publish_gate_policy.py
    snapshot_environment.py
    capture.py
    build_dataset.py
    train_anomaly.py
    train_template.py
    import_yolo.py
    build_candidate_registration.py
    register_candidate.py
    score_calibration.py
    calibrate.py
    finalize_recipe.py
    validate_candidate.py
    assemble_release.py
    compile_contract.py
    inspect.py

configs/zs32/
  topology/
  roi/
  recipes/
  policies/

data/zs32/
  raw/
  manifests/
  dataset_releases/

artifacts/zs32/
  candidates/
  releases/

pipeline/
  1_collect_multicamera_data.py
  zs32_build_dataset.py
  zs32_train_anomaly.py
  zs32_train_template.py
  zs32_import_yolo.py
  zs32_build_candidate_registration.py
  zs32_register_candidate.py
  zs32_score_calibration.py
  zs32_calibrate.py
  zs32_finalize_recipe.py
  zs32_validate_candidate.py
  zs32_assemble_release.py
  zs32_inspect.py
```

`pipeline/` 只保留薄 CLI，不再放业务实现、动态 import、路径猜测或模型逻辑。

## 5. 动态相机拓扑

禁止继续使用固定 `CANONICAL_VIEWS`、固定三个 serial 或“恰好六视角”的代码常量。

三相机配置示例：

```yaml
schema_version: 1
topology_id: zs32-3cam-double-side-v1
product: ZS32
rounds:
  - round_id: front
    prompt: capture_front
  - round_id: back
    prompt: flip_then_capture_back
camera_slots:
  - slot_id: center
    serial: DA9805574
    views: {front: front, back: back}
  - slot_id: left_oblique
    serial: DA9625347
    views: {front: front_left, back: back_left}
  - slot_id: right_oblique
    serial: DB0998274
    views: {front: front_right, back: back_right}
required_views:
  - front
  - front_left
  - front_right
  - back
  - back_left
  - back_right
```

扩为四个或五个相机时，只新增 `camera_slots` 和 view 映射；采集、ROI、模型槽位、标定组和融合必需分支均由 topology/recipe 编译生成，不允许手工同步多处常量。

稳定相机身份为 `slot_id + serial`。SDK `device_index` 只能作为诊断信息，不能进入部署身份合同。

每个 round 都是独立的物理触发边界。触发前必须展示 topology 中的精确
`prompt`，并由前台操作员显式确认；尤其是 back round，未确认翻面不得调用
相机 SDK。operator identity、prompted/confirmed UTC 时间和 round identity
必须写入不可变采集清单。取消、超时或错误确认按 `RETAKE` 隔离，不能自动
继续下一轮。

## 6. 数据身份与不可变数据集

### 6.1 原始数据

原始图不按 normal/defect 目录分流，也不允许被覆盖：

```text
data/zs32/raw/<capture_session>/images/<capture_set_id>/<view_id>.png
```

一次完整双面采集形成一个 `capture_set_id`，对应唯一 `part_instance_id`。

### 6.2 采集清单

`capture_sets.csv` 一件一行：

```text
capture_set_id,part_instance_id,product,hand,topology_id,
topology_sha256,expected_view_count,captured_view_count,status,
started_at,completed_at,failure_reason
```

`images.csv` 一张图一行：

```text
capture_set_id,part_instance_id,round_id,view_id,camera_slot_id,
camera_serial,device_index,relative_path,image_sha256,width,height,
capture_mode,exposure,gain,captured_at,status,error
```

采集进程只有在一个 capture set 的全部必需视角原子发布后才能返回成功。不完整采集必须返回非零状态，并保留诊断记录。
`capture_manifest.json` schema v2 还必须包含与 topology 顺序完全一致的
`round_confirmations`；该文件与图片、CSV 和 gate evidence 一起进入 publication
root hash，dataset 与 inspection 加载时重新验证。

### 6.3 Canonical ROI dataset release

```text
raw immutable images
  → topology/identity validation
  → crop once with authoritative ROI
  → canonical crop + crop SHA256
  → immutable dataset release
  → YOLO/Anomalib/template format adapters
```

canonical manifest 至少包含：

```text
dataset_release_id,part_instance_id,capture_set_id,hand,view,
source_path,source_sha256,crop_path,crop_sha256,roi_version,
label,defect_type,split,is_synthetic
```

release 同时必须包含独立 canonical `capture_provenance.jsonl`：每个
`capture_set_id` 恰好一行，记录 hand、policy SHA256、acquisition config
SHA256、quality profile SHA256、registration profile SHA256 与逐 view reference
SHA256。dataset
manifest 绑定该文件 hash；同一 hand 内混用任一 gate 身份必须
fail closed。dataset 仅绑定 capture gate policy，不绑定 PatchCore、
EfficientAD 或 AnomalyDINO，以便同一 canonical release 在三选一训练中复用。

train/calibration/test 必须按 `part_instance_id` 分组，禁止同一物理件的不同视角、不同重复图进入不同 split。

`semantics.local.json` 使用 schema v2，顶层必须且只能包含
`schema_version`、`approval`、`captures`。`approval` 必须且只能包含
`reviewed_by`、带时区的 `reviewed_at` 和布尔值 `approved:true`；未审批、
字段缺失或多出、无时区时间均 fail closed。dataset build 将输入文档
规范化为 `canonical_semantics.json`，保留完整快照并把 SHA256 写入
dataset manifest 和 `dataset_provenance.json`。

split 不只保留最终分配。`split_assignments.json` schema v2 同时写入
`SplitPolicy(algorithm,seed,calibration_ratio,test_ratio)` 及 policy SHA256；验证器使用
canonical rows 的 part/hand/label/defect stratum 重算分配。policy 也写入
dataset manifest 和 provenance，任一处不一致均拒绝。

canonical crop 的 PNG 编码器、实现版本、无损格式、compression 和“仅执行一次
half-open ROI crop、不 resize”的空间操作必须写入 manifest/provenance。实际
cropper 声明的 codec 与构建规格不一致时 fail closed，从而可以解释不同
PNG 字节产生的 crop SHA256。

capture 级 `label/defect_type` 只表示该物理件的 part-level 真值，不得默认
为每个视角、每个分支都可见的缺陷。生产标定必须另行发布并由 dataset
schema v4 manifest 哈希绑定的人工审批 calibration-target contract，对每个
`(capture_set_id, part_instance_id, hand, view, branch)` 显式给出 part ground
truth 和 `normal/defect/exclude` 标定目标。YOLO bbox 可用来校验 YOLO
target，但不能推断 anomaly/template target。该合同现作为
`calibration_targets.json` 内嵌于 immutable dataset release；每个 canonical
crop 必须精确覆盖三个 branch。`exclude` 仍保留原始打分审计但不参与拟合，
held-out test 因此不完整时不得通过候选验收。

现有 4K YOLO 标签迁移到 canonical crop 时必须生成：

- 原始 bbox
- ROI 后 bbox
- `clipped` 标记
- `outside_roi` 标记
- `dropped` 原因
- 迁移前后标签文件 hash

任何框被裁掉都必须出现在审计报告，不能静默变成空标签。

## 7. 模型插件与“三选一”无监督算法

统一接口示意：

```python
class AnomalyAdapter(Protocol):
    family: Literal["patchcore", "efficientad", "anomalydino"]

    def train(self, spec: TrainSpec) -> TrainedArtifact: ...

    def load(
        self,
        artifacts: Mapping[tuple[Hand, ViewId], ModelArtifact],
        device: DeviceSpec,
    ) -> AnomalyPredictor: ...


class AnomalyPredictor(Protocol):
    def predict_batch(self, samples: Sequence[RoiSample]) -> Sequence[ModelEvidence]: ...
```

约束：

- recipe 只保存一个 `anomaly_family`
- 每个 `(hand, view)` 独立训练一个无监督 checkpoint
- 同一 release 所有视角必须属于同一 family
- 融合分支统一叫 `anomaly`，算法类型写入 `model_family/model_digest`
- 各 Adapter 只负责模型构建、恢复、预处理、连续 score、heatmap 和 anomaly
  overlay 输出
- 每条 anomaly raw score 必须同时绑定由同次推理生成的 heatmap、canonical crop
  overlay 及各自 SHA256；YOLO raw score 禁止复用这些字段，检测框保留在结构化
  detections 中
- Adapter 不负责阈值判定、融合、目录扫描或部署晋升

## 8. YOLO 外部训练边界

本仓库不包含 YOLO trainer，只负责以下两件事：

1. 从 canonical dataset release 导出 YOLO 数据集。
2. 导入外部训练完成的部署包。

外部 trainer 不得接触 canonical `calibration` 或 held-out `test` 图像/标签。
YOLO training export 只能消费 canonical `train` rows，并按
`part_instance_id` 整体、按 `hand/label/defect_type` 分层，用冻结的
`seed/model_val_ratio` 从 train parts 中确定性派生 `model_val`。export 必须包含
canonical `training_export_policy.json`；`data.yaml` 只能暴露
`images/train` 与 `images/model_val`，禁止出现 calibration/test 路径。

用户提供的配置中：

- `yolo26n.pt` 是训练初始化权重
- 正式部署权重是训练产生的 `runs/.../weights/best.pt`
- 使用一套全局 YOLO 权重覆盖左右手和所有视角
- 单类别检测，业务类名固定为 `defect`
- 当前输入尺寸 `640`
- 当前训练轮数 `250`

导入命令应要求：

```text
best.pt
args.yaml
data.yaml
class_names.yaml
training_export_policy.json SHA256
ultralytics version or commit
dataset_release_id and manifest hash
```

导入后生成：

```text
models/yolo/
  best.pt
  metadata.json
  train_args.yaml
  data.yaml
  class_names.yaml
  weights.sha256
```

训练参数与推理参数必须拆开。`device`、`project`、`save_dir` 等训练机字段只进入 provenance，不进入生产运行配置。

生产推理参数至少冻结：

```yaml
imgsz: 640
candidate_conf: 0.001
iou: 0.7
max_det: 300
single_class_name: defect
```

`candidate_conf` 只是候选采集下限，最终 CLEAR/GRAY/STRONG 由独立 calibration artifact 决定。

特别注意：当前训练快照中 `seed: 43`，但 run name 为 `final_n640_p1_seed42`。模型导入前必须统一真实 seed 与命名，禁止错误 provenance 进入 release。

## 9. 模板门禁

当前第一层只保留模板匹配：

- 每个 `(hand, view)` 一组模板资产
- 模板输入必须是 canonical ROI crop
- 模板资产、参考图和阈值全部带 SHA256 和版本
- 在线输出只有 `PASS` 或 `NG_TEMPLATE`
- 不再保留模板 GRAY/REVIEW 区间
- 任一视角 `NG_TEMPLATE` 立即短路，无需加载 YOLO 或无监督模型
- 模板权重/模板文件/阈值损坏属于 `SYSTEM_ERROR`，不是 `NG_TEMPLATE`

模板二值阈值必须通过独立 calibration split 拟合并锁定；没有充足正常/缺陷样本时禁止发布模板模型。

## 10. 标定、模型注册与发布

### 10.1 Offline 流程

```text
capture raw
  → build immutable dataset release
  → train template
  → train selected anomaly family per hand/view
  → external YOLO training
  → external trainer canonical observation receipt
  → import YOLO best.pt
  → register frozen candidate (REGISTERED, never production)
  → score calibration split with frozen models
  → fit thresholds
  → evaluate held-out test split
  → validate candidate (VALIDATED)
  → manual promote
  → assemble immutable deployment release
```

训练成功绝不能自动修改生产配置。

外部 YOLO trainer 不在本仓库的信任边界内。因此 importer 必须要求严格 canonical
`training_receipt.json`，并同时验证实际 dataset release 与实际 YOLO export
原子发布。receipt 绑定四个训练产物 hash、export publication ID/root/manifest/data
hash、training-export policy hash、dataset ID/manifest hash、`args.yaml` 数据路径、run ID/name/seed，以及 clean
Ultralytics commit 或内容寻址源码包。它的语义只是外部 trainer attestation，不得表述为
“该数据必然生成该权重”的因果证明。receipt 必须连同权重原子打包到 candidate 和
deployment release，在模型加载前再次校验。

### 10.2 双阈值组

第二层阈值 key：

```text
hand, view, branch, model_digest, roi_version
```

其中 branch 只有：

- `anomaly`
- `yolo`

模板是独立的单阈值二值门禁；quality/registration 是采集 gate，不应再被伪装成连续模型阈值组。

### 10.3 Calibration artifact

必须包含：

- recipe/profile digest
- topology digest
- ROI digest
- model digests
- dataset release/split IDs
- calibration 参数
- 每组 low/high、样本数和状态
- held-out test 的 part-level 指标
- required group 完整集合
- canonical artifact SHA256

任一必需组缺少正常或缺陷标定数据时，`calibration_valid=false`，禁止发布。

### 10.4 Production held-out acceptance policy

阈值拟合参数与生产候选验收标准是两个独立合同。后者使用版本化、strict canonical
JSON；其精确 bytes 的 SHA256 是内容身份。`validate_candidate` 必须在 candidate
变为 `VALIDATED` 前，对 calibration artifact 的 held-out part metrics fail closed：

- `normal_part_count`、`defect_part_count` 分别达到策略最小物理件数；
- `defect_escape_rate`、`normal_reject_rate`、`review_rate` 均不超过策略上限；
- `evaluation_complete=true`，任何因缺 view、缺 branch 或 `exclude` 造成的 incomplete
  都不能被默认豁免。

validation publication 同时封存原始 policy 和确定性 PASS decision，并在
`validation.json`、promotion receipt、release candidate provenance 中绑定 policy/
decision SHA256。assembler 与 release loader 必须用 release 中的 calibration metrics
重新计算 decision，不能只信任一个可编辑的 `PASS` 字段。

为避免内容哈希自引用，`recipe_sha256` 定义为训练前、标定前的稳定
profile 身份，只覆盖 product、recipe id、enabled hands、topology id、ROI
version、anomaly family、capture gate policy digest 和 fusion profile digest。
capture gate policy 在首次数据采集前已由 atomic publication 冻结，因此
必须进入训练稳定身份。它不覆盖训练后才能知道的
template/anomaly/YOLO 资产引用，也不覆盖拟合后的阈值和
`calibration_sha256`：模板 `model.json` 和 anomaly metadata 本身记录该
profile digest，如果又把产物 hash 放回 profile digest 会形成不可解的模型
自引用；阈值与 calibration artifact 之间也有同类自引用。Calibration
artifact 另行记录全部实际 model digests。最终
`DeploymentContract.contract_sha256` 覆盖稳定 profile 身份、全部具体资产、
全部阈值和全部 calibration SHA256，因此正式 release 内容仍然不可
替换。

## 11. 不可变部署 Release

```text
artifacts/zs32/releases/<release_id>/
  manifest.json
  checksums.sha256
  recipe.yaml
  topology.yaml
  roi.yaml
  models/
    anomaly/<family>/<hand>/<view>/
      model.ckpt
      metadata.json
    yolo/
      best.pt
      metadata.json
      train_args.yaml
      class_names.yaml
  template/<hand>/<view>/
    model.json
    templates/
    checksums.sha256
  calibration/
    calibration_artifact.json
    template_thresholds.json
    model_thresholds.json
    metrics.json
    input_provenance.json
  fusion/
    policy.yaml
  provenance/
    dataset_release.json
    dataset_capture_provenance.jsonl
    dataset_canonical_semantics.json
    dataset_calibration_targets.json
    dataset_governance.json
    dataset_split_assignments.json
    model_candidate.json
    model_candidate_full.json
    promotion_receipt.json
    code_version.json
    runtime_environment.json
```

release loader 在加载任何大模型前完成：

- schema 校验
- checksums 校验
- product/hand/topology/ROI 一致性校验
- required views 完整性校验
- template/anomaly/YOLO/calibration group 完整性校验
- 左手 ROI readiness 校验
- runtime environment receipt 的 canonical schema、自校验 digest 和 release 文件 digest 校验
- promotion receipt 的 canonical schema、文件 digest 和完整身份链校验
- held-out acceptance policy/decision 的 canonical bytes、内容哈希和重新计算校验
- template/anomaly/score-calibration execution receipt 的 clean code、Linux/NVIDIA、wheel
  installed-tree/import-origin、输入与参数身份校验
- dataset v4 的人工 semantics/calibration-target 审批、SplitPolicy 可重放性和实际 PNG codec 校验

release 不接受裸的 command-line promotion ID。人工批准必须以一个
no-replace 发布的 canonical receipt 输入，至少绑定 `release_id`、
`promotion_id`、approver/time、candidate ID/digest、validation publication
ID/root/record SHA256、contract SHA256、calibration artifact SHA256、dataset manifest
SHA256、held-out acceptance policy/decision SHA256，以及 golden parity/FAT/SAT
的明确证据状态。golden parity 必须为
`PASS + evidence SHA256`；FAT/SAT 可为 `PASS + evidence SHA256` 或带理由的
`NOT_APPLICABLE`。`NOT_RUN`/`FAIL`、缺失或未知字段一律 fail closed。
release 内封存 receipt 原始 bytes，`model_candidate.json` 引用其精确 SHA256；
loader 必须在加载模型前逐字段重建这条身份链。

`runtime_environment.json` 必须在最终部署 Linux + NVIDIA 环境生成，冻结 kernel、
`/etc/os-release` ID/VERSION_ID、GPU UUID、NVIDIA driver、`CUDA_VISIBLE_DEVICES`/
CLI device/physical GPU 映射，以及 Python、torch、torchvision、anomalib、
ultralytics、OpenCV、NumPy、torch CUDA build 和 cuDNN 版本。上述六个 Python
distribution 还必须记录原始 `METADATA`/`RECORD` SHA256 与 RECORD 条目数，不能
仅以 `__version__` 代替依赖内容身份。
inspect 必须在读取 capture 和加载任何模型前采集现场 receipt 并精确比较；任一漂移
为 `SYSTEM_ERROR`。receipt 文件由 release checksum root 内容寻址，内部另带不含
自引用字段的 canonical semantic digest。

## 12. 在线检测 DAG

```text
1. Load and validate immutable release
2. Verify live Linux/NVIDIA runtime receipt against the release before reading capture
3. Validate capture identity, required views and persisted gate evidence
4. Reject any capture that did not pass capture-time quality and registration gates
5. Crop every source image exactly once with authoritative ROI
6. Run template gate for all required views
7. If any template mismatch: NG_TEMPLATE and short-circuit
8. Run global YOLO batch and selected anomaly predictors
9. Validate evidence identity/version/hash/completeness
10. Apply strict multi-view fusion
11. At the sink boundary, independently revalidate the run against the verified
    release contract and rederive the strict-fusion decision
12. Atomically publish evidence, decision and audit
```

YOLO 和无监督算法必须作为两个隔离分支并行执行，但必须在 release 校验、采集校验、持久化的
质量/配准通过证据、统一 ROI 和模板门禁完成后启动。质量或配准失败由采集服务
直接返回 `RETAKE` 并隔离到 `_incomplete`；失败采集不得伪装成 complete
publication 进入 inspect。
两个分支必须全部启动并等待结束；任一分支失败不得取消另一分支。协调器必须
按 `anomaly -> yolo`、再按视角 ID 的固定顺序收集原始与标定证据，而不是依赖线程
完成顺序。一边故障、另一边 STRONG 时，必须同时保留系统错误和 STRONG 证据。

Ultralytics 的全局 batch 结果不得仅按列表位置与视角绑定。backend 必须把输入 crop
转换为唯一、可解析的规范路径，并在生成 evidence 前回验每个 `result.path`。返回路径重复、
缺失、多出或顺序变化均必须 fail closed 为 `SYSTEM_ERROR`，不允许对同尺寸多视角结果
盲目 `zip`。

## 13. 状态机与优先级

### 13.1 最终业务状态

| status | 含义 | 是否允许放行 |
| --- | --- | --- |
| `OK` | 全部必需视角 template PASS，anomaly/yolo 均 CLEAR | 是 |
| `NG_TEMPLATE` | 任一视角模板明确不匹配 | 否 |
| `NG_ANOMALY` | 任一视角无监督分支 STRONG | 否 |
| `NG_YOLO` | 任一视角 YOLO 分支 STRONG | 否 |
| `REVIEW` | 无 STRONG，但存在 anomaly/yolo GRAY | 否 |
| `RETAKE` | 图像质量或配准失败；由采集阶段直接返回 | 否，重新采集 |
| `INVALID_CAPTURE` | 缺视角、重复视角、身份或来源冲突 | 否 |
| `SYSTEM_ERROR` | 模型、阈值、配置、资产或运行时故障 | 否 |

### 13.2 判定顺序

```text
release invalid                         -> SYSTEM_ERROR
capture identity/topology invalid       -> INVALID_CAPTURE
capture-time quality/registration failed -> RETAKE (do not publish complete capture)
persisted gate evidence invalid           -> INVALID_CAPTURE
template asset/runtime failed            -> SYSTEM_ERROR
template mismatch                        -> NG_TEMPLATE
anomaly/yolo runtime or evidence missing -> SYSTEM_ERROR
any anomaly/yolo STRONG                  -> NG_ANOMALY / NG_YOLO
any anomaly/yolo GRAY                    -> REVIEW
all required evidence CLEAR              -> OK
```

如果某个分支已发现强阳性，随后另一个分支发生系统故障：

- `evidence_status` 保留已发现的 `NG_*`
- `inspection_status=SYSTEM_ERROR`
- `released_status=null`
- 审计中同时保留强阳性和系统故障

这样既不丢失真实缺陷证据，也不会发布一次不完整检测的正式 NG/OK。

## 14. 严格融合合同

DeploymentContract 由 recipe 编译产生，必须精确列出：

- product = ZS32
- allowed hands
- topology/profile hash
- canonical capture gate policy ArtifactRef/hash，及其 quality/registration
  profiles 和逐 view registration reference hashes
- required views
- 每个视角唯一 ROI
- template asset/version
- 当前 anomaly family
- 每个 `(hand,view)` anomaly checkpoint/hash/version
- 全局 YOLO checkpoint/hash/version
- 每个 `(hand,view,branch)` low/high threshold
- required evidence group
- fusion policy hash

capture evidence 必须结构化记录 policy/profile/reference digests。gate
`reason` 只是诊断文本，不参与 provenance。inspect 必须在任何模型加载前
将 capture provenance 与 verified release policy 完全比较；不一致不得进入 OK。

orchestrator 的检查不能代替 publication boundary 的检查。inspection sink
必须把 `InspectionRun` 当作不可信输入：按 verified `DeploymentContract`
重新校验 gate/crop/template/raw+calibrated model 的 identity、hash、唯一性和
必需组完整性，然后用同一个 strict fusion policy 重新推导 decision。
`OK`/`NG_*`/`REVIEW` 的内存 decision 与重新推导结果不完全相同时，
或 terminal status 缺少对应的 gate/system-error 证据时，必须在创建
可消费目录前拒绝发布。

三相机时，第二层必需证据为：

```text
6 views × (anomaly + yolo) = 12 rows
```

四相机和五相机自动扩展为 16、20 行。禁止在 Stage 18、31、32 分别维护三份 group 常量。

## 15. 审计与原子发布

每次检测生成：

```text
inspection/<inspection_id>/
  request.json
  capture_manifest.json
  crops/<view>.png
  template_evidence.jsonl
  anomaly_evidence.jsonl
  yolo_evidence.jsonl
  heatmaps/anomaly/<view>.png
  overlays/anomaly/<view>.png
  decision.json
  audit.json
  checksums.sha256
```

发布流程：

1. 写入唯一 staging 目录。
2. 完成所有序列化与 SHA256。
3. 运行 release/evidence/completeness 校验。
4. 最后一次原子 rename。
5. 已存在目标目录时拒绝覆盖。

审计必须独立保存：

- `evidence_status`
- `inspection_status`
- `review_status`
- `released_status`
- 全部触发证据
- source/crop/evidence/model/config hashes

`audit.json` schema v3 必须对全部持久化 evidence 文件（包括 PASS/CLEAR 行与空文件）
独立记录 SHA256，不得只记录触发 NG/REVIEW 的行，也不得只依赖最终
`checksums.sha256`。审计中还必须显式保留 capture gate policy 和 acquisition
config digest。

## 16. 现有代码迁移策略

### 16.1 保留并迁移

- `pipeline/1_collect_multicamera_data.py`：保留为采集入口，重写为 profile-driven CLI。
- `capture_data/collect_multicamera_dataset.py`：保留 SDK adapter、序列号绑定、group trigger、HDR 和资源清理。
- `capture_data/exposure_fusion.py`：保留曝光融合能力。
- `capture_data/zs32_view_roi_dataset.py`：迁移 bbox 坐标变换和越界报告。
- `capture_data/zs32_patchcore_roi_dataset.py`：迁移 preflight、manifest 和事务发布思路，不保留独立 PatchCore ROI。
- `capture_data/zs32_template_gate.py`：迁移模板 hash、score 和 overlay，改成二值 PASS/NG。
- `capture_data/fusion_calibration.py`：迁移双阈值、required groups、fit/test 隔离和 artifact hash。
- `capture_data/fusion_engine.py`：迁移 fail-closed、证据等级、trigger 和严格完整性语义。
- `capture_data/inspection_audit.py`：迁移审计结构。
- `capture_data/zs32_model_runtime.py`：迁移 asset hash、YOLO boxes 规范化、overlay 和原子 staging。

### 16.2 重写

- Stage 29 与 Stage 30 合并为一个统一 ROI/dataset release builder。
- `pipeline/8_train_custom_models.py` 与 `zs32_defect_workflow.py` 拆为三种 Adapter、Trainer、Evaluator、CLI。
- Stage 31 改为只校准 template 单阈值和 anomaly/yolo 双阈值。
- Stage 18 改为消费唯一 DeploymentContract，不再兼容 C789/FX11 CSV。
- Stage 32 改为 package-driven runtime，不再动态 import Stage 18。
- 三份 `zs32_six_view/right/runtime_models` 配置合并为 recipe + immutable release。

### 16.3 删除或归档

- C789/FX11 所有采集、裁剪、stress、geometry、traditional、YOLO one-off 和 demo。
- `0_run_all.py`。
- `2_process_data.py`、`4_inference.py`、`5_demo_inspection.py`。
- 专用 6/7/8 多模型子进程拼装器。
- C789 template matcher 原型 `train_template_matcher.py`、`predict.py`。
- 固定六视角 `CANONICAL_VIEWS` 与 right-only profile。
- PatchCore/YOLO 两套 ROI loader 和两套 ROI version。
- `sys.path.insert`、动态 import pipeline 脚本和以目录名猜业务身份的逻辑。

旧脚本只有在新链路完成 parity 验证后才删除；尤其 Stage 3/8/workflow 必须等新训练链能够复现当前 PatchCore checkpoint 后再移除。

## 17. 分阶段实施计划

### Phase 0：冻结现状

- 本阶段只在 Linux + NVIDIA GPU 主机执行；Mac 只允许编辑 Phase 0 所需脚本和文档，不产生任何冻结或验收产物。
- 执行入口为 `tools/zs32_phase0.py`，逐步操作见 `docs/PHASE0_LINUX_RUNBOOK.md`；所有大资产和 golden 数据保留在 Linux bundle，Git 只提交配置、代码和最终 bundle pointer。
- 记录当前 commit、右手 ROI、现场相机采集参数、PatchCore checkpoint、YOLO best.pt、模板和阈值。
- 保存一批真实六视角 golden samples。
- 固化当前可接受检测结果和所有失败用例。
- `replay-bindings` 先将 clean Git、`uv.lock`、topology、ROI、全部
  normative assets、capture identity 和逐 view golden image hash 固化到
  每个 case；两次 replay 必须使用同一命令并与该 binding 一致。
- required asset roles、golden scenarios 和 known blocker IDs 在工具中固化，
  local spec 删空它们不能降低 Phase 0 门槛。
- 人工 approval 必须绑定最终不可覆盖 inventory 的文件 hash 与
  review-subject hash；freeze 重新采集后的稳定审核 payload 必须完全一致。
- legacy replay 已固定为 Stage 32 `fuse` 入口，并由 Linux-only runner
  原子捕获真实输出。每个 case 的质量、配准、几何 CSV 必须作为
  内容寻址 asset 登记；缺少它们时禁止伪造 PASS 来完成 replay。
- 记录 Linux 发行版、Python、GPU、NVIDIA driver、CUDA、PyTorch、Anomalib、Ultralytics 和相机 SDK 版本。
- 禁止在重构期间继续向旧 pipeline 添加功能。

验收：在 Linux + NVIDIA GPU 主机上可从同一输入重现基线输出及资产 hashes；Mac 结果一律不计入验收。

### Phase 1：Domain、schema 与 recipe compiler

- 建立新 Python package。
- 定义 topology、ROI、dataset、evidence、release schema。
- 实现 DeploymentContract 编译与 hash。
- 加入左手 ROI pending 的 fail-closed 校验。

验收：3/4/5 相机配置能分别编译出 6/8/10 必需视角与 12/16/20 第二层证据组。

### Phase 2：采集与数据身份

- 将三相机硬编码改为 topology-driven。
- 建立 `part_instance_id/capture_set_id`。
- 将曝光、增益、HDR/融合、超时和编码参数固化为版本化 acquisition asset，
  capture provenance 必须记录其 SHA256，禁止正式采集接受未绑定的 loose config。
- 修复 incomplete 仍返回成功的问题。
- 原始图和 manifests 原子发布。

验收：缺任一视角时非零退出；视角、serial、hash 可追溯。

### Phase 3：统一 ROI 与 dataset release

- 导入当前右手 ROI。
- 合并 Stage 29/30。
- 迁移 4K YOLO 标签并生成 clipped/dropped 报告。
- 建立不可变 canonical crop release。
- 输出 YOLO、Anomalib 和 template 数据适配视图。

验收：三类模型读取完全相同的 crop SHA256；不存在第二套空间 ROI。

### Phase 4：模型 Adapter 与训练

- 拆出 PatchCore Adapter 并先复现当前 checkpoint。
- PatchCore 预训练 backbone 必须使用显式本地 raw feature-extractor state_dict，
  使用 backend schema v2，以 logical asset ID、SHA256 和固定
  `anomalib_timm_feature_extractor` scope 绑定并 strict load；训练 receipt 还必须绑定
  原始参数摘要与 backbone input digest，禁止训练期隐式下载或未绑定缓存解析。
- 加入 EfficientAD、AnomalyDINO Adapter。
- 实现外部 YOLO model import。
- 建立 candidate registry 和 provenance。

验收：同一 recipe 只能激活一个 anomaly family；训练成功不会修改 production release。

### Phase 5：模板、标定与部署包

- 模板改成二值 PASS/NG。
- 重构标定组为 template + anomaly/yolo。
- test split 仅评估，不参与阈值拟合。
- 实现 release assemble/promote/validate。

验收：任一缺组、坏 hash、pending ROI 或版本不一致都无法发布。

### Phase 6：新运行时与严格融合

- 实现统一 orchestrator。
- crop once。
- 模板短路。
- YOLO + selected anomaly 并行。
- 动态视角严格融合。
- SYSTEM_ERROR 和审计/原子发布。
- inspection sink 独立进行 release/contract-aware evidence completeness 校验，
  并重新推导 strict-fusion decision。

验收：任意故障注入均不能产生可放行 OK。

### Phase 7：清理旧代码

- 对 golden samples 做新旧 parity。
- 输出删除清单和迁移说明。
- 删除 C789/FX11、旧 pipeline 和重复配置。
- 更新 README、安装和部署文档。

验收：仓库只剩 ZS32 主链；clone + 外部资产导入后可以复现。

## 18. 测试矩阵

### 18.1 Schema 与配置

- 3/4/5 相机 topology 编译。
- serial 重复、view 重复、缺 round、缺 ROI。
- 左手 ROI pending 阻止 release。
- recipe anomaly family 混搭被拒绝。

### 18.2 数据

- 同一物理件绝不跨 split。
- 原图、crop 和标签 hash 稳定。
- bbox crop/clip/drop 数学正确。
- 缺标注与确认空标注明确区分。
- 不完整 capture set 不能进入 dataset release。

### 18.3 模型

- 三个 anomaly Adapter 的统一 contract tests。
- checkpoint 恢复与版本校验。
- YOLO 全局 batch 对左右手/所有视角保持 identity。
- YOLO batch 的 reversed/duplicate/missing/extra `result.path` 故障注入全部 fail closed。
- score direction、NaN/Inf、空 detection、畸形 box。

### 18.4 状态机

- 模板 mismatch 立即 NG_TEMPLATE，第二层未调用。
- 模板资产损坏为 SYSTEM_ERROR。
- capture quality/registration fail 由采集 CLI 返回 RETAKE，且不发布 complete capture。
- 缺视角为 INVALID_CAPTURE。
- anomaly/yolo STRONG 为对应 NG。
- GRAY 为 REVIEW。
- 所有必需证据 CLEAR 才 OK。
- 强阳性后系统故障保留 evidence NG，但不发布 released status。

### 18.5 安全性

- 任意缺证据、坏 hash、错版本、错 ROI、错 hand、错 topology 不得产生 OK。
- publication failure 不留下可消费目录。
- 已有 release/inspection 目录禁止覆盖。
- anomaly heatmap/overlay 任一缺失、被替换、非 canonical regular file 或 hash
  不一致时拒绝发布；YOLO 行携带 anomaly visual 字段同样拒绝发布。
- fault-injection property test：`system fault ⇒ released_status != OK`。
- 直接伪造内部 `InspectionRun` 的缺 view、缺 branch、raw/calibrated 不一致、
  REVIEW 伪装 OK、NG_TEMPLATE 证据不完整、无失败 gate 的 RETAKE 和无错误证据的
  SYSTEM_ERROR，inspection sink 均必须在原子 rename 前拒绝。

### 18.6 硬件与性能

- 以下测试全部只允许在 Linux + NVIDIA GPU 主机运行。
- 三相机同步 trigger/read。
- 单相机掉线、超时、错序列号、翻面中断。
- 缺 round confirmation、错误 prompt/operator、确认超时后下一轮绝不触发。
- GPU/CPU 模型加载失败。
- 模型常驻内存和六/八/十视角延迟。
- 长时间运行无资源泄漏。

## 19. 第一版明确不做

- 不在本仓库实现 YOLO 训练器。
- 不支持 ZS32 之外产品。
- 不保留 geometry、crack、surface、feature-presence 等传统检测分支。
- 不允许在线自动训练或自动晋升模型。
- 不允许左手 ROI 缺失时复用右手 ROI。
- 不让多数投票覆盖强阳性、缺证据或系统故障。
- 不让模型直接控制 PLC、安全链路或机械动作。

## 20. 尚未冻结但不阻塞架构的参数

以下参数必须在生产发布前通过独立数据冻结，但不影响本次代码重构：

- 左手各视角 ROI。
- 最终 4/5 相机 serial 与 view 名称。
- 模板二值阈值的 calibration 目标。
- 缺陷在各 view/branch 的显式 calibration target，以及 `exclude` 审核规则。
- anomaly/yolo low/high 的 target recall、normal quantile 和最小物理件数量。
- 最终 YOLO 训练超参数和真实 seed。
- 生产 GPU、最大延迟与吞吐要求。

这些参数属于版本化 recipe/release 数据，不应再次写死进 Python 代码。

## 21. 重构完成定义

只有同时满足以下条件，重构才算完成：

1. 仓库中只有一条 ZS32 主链。
2. 采集 topology 可通过配置扩展到 3/4/5 相机。
3. 一个视角只有一个空间 ROI 真相。
4. 一个 release 只激活一种无监督算法。
5. YOLO 外部权重可被内容寻址地导入和审计。
6. 模板不匹配可靠短路为 NG_TEMPLATE。
7. 所有必需 anomaly/yolo evidence CLEAR 才可能 OK。
8. 缺视角、质量问题、系统故障具有互不混淆的状态。
9. 数据、模型、阈值、ROI、代码和检测证据形成完整 hash 链。
10. 任何 fault injection 都无法生成可放行 OK。
11. 左手 ROI 未就绪时无法发布左手 release。
12. 新训练链复现基线后，旧 C789/FX11 与重复 pipeline 被删除。
