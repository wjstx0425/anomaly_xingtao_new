# ZS32 重构主链 Linux + NVIDIA 执行手册

状态：代码实现已在 Mac 上静态编辑，所有运行、测试、训练、推理、相机验证和发布验收均等待 Linux + NVIDIA 主机执行。Mac 结果不计入任何阶段验收。

当前代码合同已升级为 dataset schema v4：每个 canonical crop 必须绑定经人工审批的
逐 view、逐 `template/anomaly/yolo` 标定目标。旧 dataset schema v3 只有
capture-level defect label，不能自动升级或直接用于生产标定；必须先在 Linux 侧按本手册
建立 `calibration_targets.local.json` 并重新发布数据集。任何 test target 为 `exclude`
都会使完整系统评估不成立并 fail closed。

本文从已完成的代码同步开始，Phase 0 的冻结细节见 `PHASE0_LINUX_RUNBOOK.md`。任何命令失败都应停在当前阶段；禁止跳过校验、手工把 `registered` 改成 `validated`，或复用已有 publication id 覆盖目录。

## 1. GitHub → Linux 交接合同

代码由人工在 Mac 审核、提交并 push；本重构过程本身不自动
commit/push。每次交接必须先在 Mac 产生三个不可模糊的源码指针，并通过
可审计渠道交给 Linux 操作人员：

```bash
git status --porcelain=v1 --untracked-files=all
git rev-parse HEAD
git rev-parse HEAD^{tree}
shasum -a 256 uv.lock
```

分别记为 `HANDOFF_COMMIT`、`HANDOFF_TREE` 和 `HANDOFF_UV_LOCK_SHA256`。仅提供
branch 名不算交接；branch 可继续移动。Mac 上的 `git status` 必须为空，
`uv.lock` 必须已跟踪且与该 commit 一起 push。提交前必须检查 staged 文件，
不得包含 `*.local.*`、图像、权重、bundle、数据集、训练输出或运行证据。

Linux 必须从 GitHub 拉取后核对上述三值，而不是接受“当前 main”：

```bash
cd /path/to/anomaly_xingtao
git fetch --prune origin
git checkout main
test -z "$(git status --porcelain=v1 --untracked-files=all)"
git pull --ff-only origin main
test "$(git rev-parse HEAD)" = "$HANDOFF_COMMIT"
test "$(git rev-parse HEAD^{tree})" = "$HANDOFF_TREE"
test "$(sha256sum uv.lock | awk '{print $1}')" = "$HANDOFF_UV_LOCK_SHA256"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

任一 `test` 失败都必须停止，不得改用本地旧代码继续。正式 Phase 0
freeze、训练、标定、推理和 release assembly 必须全程使用这个 clean
worktree。模型、真实图像、golden bundle、标定输出、score run、release
实体和 Linux 日志留在资产盘，不进入 GitHub。GitHub 中唯一允许指向
外部大资产的文件是经审核的轻量 pointer，例如
`baselines/zs32/<freeze_id>.pointer.json`；pointer 只含 ID、digest、Git commit 和不可变
storage reference，不含凭证或资产内容。

## 2. 环境与目录约定

示例统一使用：

```bash
export REPO=/path/to/anomaly_xingtao
export ZS32=/DATA/ljl/zs32_refactor
export ASSET_ROOT=$ZS32/artifacts
export TOPOLOGY=$REPO/configs/zs32/topology/zs32_3cam_double_side_v1.json
export ROI=$REPO/configs/zs32/roi/zs32_roi_v2.json
export POLICY=$REPO/configs/zs32/policies/zs32_strict_fusion_v1.json
export HELDOUT_ACCEPTANCE_POLICY=$REPO/configs/zs32/calibration/heldout_acceptance_strict_v1.json
export RECIPE_TEMPLATE=$REPO/configs/zs32/recipes/zs32_right_patchcore_training_v1.json
export BASE_RECIPE=$ZS32/zs32_right_patchcore_training_v1.local.json
mkdir -p "$ZS32" "$ASSET_ROOT"
```

以上变量是现有三相机右手 release 示例。第四台正面相机 `DB0968108` 使用
`configs/zs32/topology/zs32_4cam_double_side_v1.json`，产生
`front_secondary/back_secondary`，整件为 8 个视图。四机不能继续使用上面的三机
`ROI`、`RECIPE_TEMPLATE` 或 gate publication；首次 gate 发布、正式采集以及后续 8-view
ROI/训练资产的顺序见
[`configs/zs32/topology/README.md`](../configs/zs32/topology/README.md)。

`RECIPE_TEMPLATE` 不可执行：它的 zero SHA256 是 unbound sentinel。第 4 节
发布 gate publication 后，必须把命令输出的真实 `policy_sha256`
写入 `capture_gate_policy` 后生成 `$BASE_RECIPE`；从首次训练到正式
release 始终使用这一份 bound local recipe。不得用 zero 值运行训练。

根据实际 CUDA 选择唯一 extra，例如：

```bash
uv lock --check
uv sync --frozen --extra cu126 --extra test
.venv/bin/python -c 'import torch; assert torch.cuda.is_available()'
```

`uv lock --check` 必须先证明 `pyproject.toml` 与 `uv.lock` 一致；`--frozen` 禁止 Linux 端重写依赖解析。安装后再次执行第 1 节的 commit/tree/
lock/worktree 四项校验；环境问题应通过新的受审核 commit 修复，不得在 Linux
上直接改 `pyproject.toml` 或 `uv.lock`。

`ultralytics` 是外部 YOLO26 fork，但 template/anomaly 的执行收据也会冻结
完整生产环境，所以它必须在任何 `zs32-*` 命令之前安装，不能等到 YOLO 导入
阶段。先验证受审核 checkout 或内容寻址源码包，再装入刚才创建的同一个
`.venv`：

```bash
test -z "$(git -C /path/to/audited/ultralytics status --porcelain=v1 --untracked-files=all)"
uv pip install --python .venv/bin/python --no-deps /path/to/audited/ultralytics
.venv/bin/python -c 'from pathlib import Path; import ultralytics; from zs32_inspection.models.yolo_receipt import runtime_package_tree_sha256; print(runtime_package_tree_sha256(Path(ultralytics.__file__).resolve().parent))'
```

输出必须与外部 trainer receipt 记录的 package tree 一致。此后本文全部使用
`.venv/bin/...`，避免 `uv run` 的隐式同步删除这个受审核但未写入公共 lock 的 fork。
若 fork 版本需要变化，必须重新训练/attest YOLO，并重新生成后续全部 execution
receipt 和 deployment release。

当前左手 ROI 为 `pending`，因此这份示例只允许右手 release。左手 ROI 未测量、审核并改为 `ready` 前，任何左手 candidate/contract/release 都必须失败。

## 3. Phase 0 必须先冻结

严格执行 `docs/PHASE0_LINUX_RUNBOOK.md`。在以下证据回传前，不得声称 Phase 0 完成：

- trusted bundle root；
- Git pointer；
- 两次 legacy replay；
- 右手六视角 golden cases；
- 旧 PatchCore、YOLO `best.pt`、模板、阈值、ROI 和环境 hashes；
- Phase 0 Linux tests 与故障注入结果。

新主链的开发可以继续，但 Phase 7 删除旧代码必须等待 golden parity。

## 4. 采集与不可变数据集

一次 `zs32-capture` 对应同一物理件的一个完整 `capture_set_id`。双轮采集必须得到当前
topology 的完整 view 集：三机为 6 个、四机为 8 个、五机为 10 个；缺图、serial 错误、
quality/registration 失败均非成功采集。

首个模型 release 之前，先把 topology、canonical
`capture/gates/policy.json`、版本化 `capture/acquisition/hikvision.json`、按 hand
独立的 quality/registration profile 和每个 view 的 registration reference
发布为 no-replace atomic gate publication。gate policy schema v2 通过
`acquisition_config` ArtifactRef 绑定该文件的精确 SHA256。左手 profile
未批准时 policy 不得包含 left：

发布命令会将 authoring policy 规范化为 sorted-key compact JSON，输出的
`policy_sha256` 以 publication 内的规范化字节为准，不得对本地排版文件
自行计算后替代该输出。

```bash
.venv/bin/zs32-publish-gate-policy \
  --publication-id gate-zs32-right-r1 \
  --output-root "$ASSET_ROOT/gate-publications" \
  --topology "$TOPOLOGY" \
  --policy "$ZS32/capture_gate_policy.local.json" \
  --asset-root "$ZS32/gate_assets"
```

记录输出的 `policy_sha256`，然后生成唯一 bound training recipe
（`artifact_id/version` 也必须是本次批准值）：

```bash
export GATE_POLICY_SHA256=<zs32-publish-gate-policy-output-policy_sha256>
jq --arg sha "$GATE_POLICY_SHA256" \
  '.capture_gate_policy = {
    artifact_id:"zs32-capture-gate-policy-right-r1",
    version:"gate-zs32-right-r1",
    sha256:$sha,
    relative_path:"capture/gates/policy.json"
  }' "$RECIPE_TEMPLATE" > "$BASE_RECIPE"
```

bootstrap/dataset 采集只接受这份完整 publication，不接受 loose profile：

```bash
.venv/bin/zs32-capture \
  --gate-publication "$ASSET_ROOT/gate-publications/gate-zs32-right-r1" \
  --raw-root "$ZS32/raw" \
  --capture-session session-20260714-a \
  --capture-set-id capture-part0001-a \
  --part-instance-id part0001 \
  --hand right \
  --operator-id operator001 \
  --round-confirmation-timeout 120
```

`zs32-capture` 不提供 `--camera-config`；采集参数只能从已验证 gate
publication 或 deployment release 中的 acquisition asset 读取。
每一轮触发前终端会显示 topology 中冻结的定位/翻面提示。操作员必须在前台
TTY 内输入精确的 `CONFIRM <round_id>`；取消、错误输入或超时均按 `RETAKE`
写入 `_incomplete`，下一轮绝不会触发。成功确认的 operator、prompt 和 UTC
时间会写入已哈希的 `capture_manifest.json`。

正式 release 产生后，采集命令改用 `--release <immutable-release-root>`，
它与 `--gate-publication` 严格二选一。capture evidence 结构化记录
policy、acquisition config、quality profile、registration profile 和逐 view
reference SHA256；
`reason` 只作诊断。后续 recipe/DeploymentContract/release 必须绑定同一
policy SHA256。inspect 发现 capture 不一致则 `INVALID_CAPTURE`，release
策略资产损坏则 `SYSTEM_ERROR`，都不可能产出 OK。

数据治理文件 `semantics.local.json` 必须由人工绑定物理件、normal/defect、defect type 和标注状态。它必须使用 schema v2，其 `approval` 对象严格为：

```json
{
  "reviewed_by": "qa-user-id",
  "reviewed_at": "2026-07-14T12:00:00+08:00",
  "approved": true
}
```

顶层必须且只能包含 `schema_version:2`、上述 `approval` 和非空
`captures`；每个 capture 的 annotations 必须精确覆盖 topology 全部 view。
`capture_set_path` 必须是 `<session>/images/<capture_set_id>` 相对路径，并与
capture manifest 和每个 canonical row 的 source path 相互印证。
`approval` 缺失、多字段、`approved` 不是布尔 `true`、时间无时区均会
fail closed。4K YOLO 标注只在 dataset build 中按唯一 ROI 做坐标迁移；
三类模型读取相同 canonical crop SHA256。

数据集 schema v4 还强制绑定人工审批的
`calibration_targets.local.json`。参考
`configs/zs32/calibration/calibration_targets.example.json`，对每个 canonical
`(capture_set_id,part_instance_id,hand,view)` 精确写三条 `template/anomaly/yolo`
记录，并按完整 key 排序。`part_ground_truth` 必须与 semantics 的零件真值一致；
`target` 可为 `normal/defect/exclude`。正常件不得标为 branch defect；`exclude`
以及“缺陷件在该 branch/view 不可见而标为 normal”都必须写人工理由。YOLO bbox
只能辅助审核 YOLO target，禁止据此推断 template/anomaly target。审批对象同样
必须含审核人、带时区时间和布尔 `approved:true`。合同缺项、多项、错 hand/view、
错零件真值或审批失败时，dataset 不会发布。

```bash
.venv/bin/zs32-build-dataset \
  --topology "$TOPOLOGY" \
  --roi "$ROI" \
  --recipe "$BASE_RECIPE" \
  --gate-publication "$ASSET_ROOT/gate-publications/gate-zs32-right-r1" \
  --raw-root "$ZS32/raw" \
  --semantics "$ZS32/semantics.local.json" \
  --calibration-targets "$ZS32/calibration_targets.local.json" \
  --dataset-release-id dataset-zs32-r1 \
  --output-root "$ASSET_ROOT/datasets" \
  --created-at 2026-07-14T12:00:00+08:00 \
  --hand right \
  --split-seed 43 \
  --calibration-ratio 0.20 \
  --test-ratio 0.20 \
  --png-compression 1 \
  --export-output-root "$ASSET_ROOT/exports" \
  --yolo-export-id yolo-r1 \
  --yolo-model-val-seed 43 \
  --yolo-model-val-ratio 0.20 \
  --anomalib-export-id anomalib-r1 \
  --template-export-id template-r1
```

`$BASE_RECIPE` 授权采集时绑定的 capture gate policy；`--gate-publication`
还会重新打开实际原子 publication，证明每个 capture 中的 policy ID、
acquisition config、quality/registration profile SHA256 和逐 view reference
SHA256 都确实源自该
policy，而不只是外层 policy SHA256 相同。dataset release 不绑定具体
anomaly family，因此同一 canonical crop release 可用于 PatchCore、EfficientAD
或 AnomalyDINO。所有 capture provenance 必须与 recipe 和已验证 publication
精确一致，否则 fail closed。

记录 stdout 返回的 `dataset_release.json`、`capture_provenance.jsonl`、
`split_assignments_sha256` 和三份 adapter manifest hashes。manifest 会绑定
capture provenance SHA256；该 JSONL 每个 capture set 恰好一行，并与
canonical rows 的 capture/part/hand 和动态 3/4/5 相机 required views 一一对应。
release 还会保留 `labels/source/<sample_id>.txt` 的原始 4K YOLO 标签字节；
验证器使用原始标签、权威 ROI 和原图尺寸重新执行坐标迁移，并逐字节比较裁剪
标签、逐框 clip/drop 审计和两侧 SHA256，禁止只保留不可重放的统计摘要。
物理件不能跨 train/calibration/test。

YOLO export 是额外的 training-safe publication：它物理排除 canonical
`calibration/test` rows，只在 canonical `train` parts 内按
`hand/label/defect_type` 分层并确定性切出 `model_val`。输出的
`training_export_policy.json` 冻结算法、seed、比例、assignment unit 与排除列表；
`export_manifest.csv` 中原始 `split` 必须全部为 `train`，同一
`part_instance_id` 只能属于 train/model_val 之一，`data.yaml` 不得出现 test 或
calibration。stdout 的 `export_policy_sha256.yolo` 必须与 publication 内 policy
文件一致。

release 额外保留经规范化的 `canonical_semantics.json` 和
`dataset_provenance.json`。后者绑定人工审批身份/时间、semantics SHA256、
`SplitPolicy` 及 SHA256、`split_assignments.json` SHA256，以及 canonical PNG 的
`opencv.imencode`/OpenCV 实现版本/lossless PNG/compression/单次 half-open ROI crop 编码契约。
验证器必须从 canonical rows 重算 split，并确认实际 cropper 声明的
codec 与记录完全一致。这是必须字段的 dataset manifest schema v4，
旧 schema v1/v2/v3 不会被隐式升级。

## 5. 训练模板与唯一无监督算法

本 release 选择 PatchCore 后，不得混入 EfficientAD/AnomalyDINO。对动态 `required_slots` 中每个 `(hand, view)` 分别执行模板与 anomaly 训练；训练 split id 必须等于数据集的 `split_assignments_sha256`。

PatchCore backend 参数必须使用 schema v2；禁止在训练时隐式联网或从未绑定缓存选择
`pre_trained` 权重。
`patchcore.example.json` 的 `auxiliary` 必须指向 Linux 主机上预先准备的、与
所选 backbone/layers 完全匹配的 raw feature-extractor `state_dict`，并填写
opaque `backbone_weights_asset_id`、文件 SHA256 和固定
`state_dict_scope=anomalib_timm_feature_extractor`。后端以 `pre_trained=false` 构建结构后执行 strict load，且训练前后
重复 rehash；路径缺失、hash 漂移、wrapped checkpoint、缺键或多键都立即失败。
训练收据的 `parameters_sha256` 必须等于原始参数文件的 canonical digest，并以
`patchcore_backbone` input role 再绑定同一权重 SHA256。发布 metadata 删除训练主机路径，
只保留 asset ID、digest 和 scope；完整 checkpoint 自包含 backbone，因此该原始文件不是
正式推理依赖。

```bash
.venv/bin/zs32-train-template \
  --recipe "$BASE_RECIPE" --topology "$TOPOLOGY" --roi "$ROI" \
  --dataset-release "$ASSET_ROOT/datasets/dataset-zs32-r1" \
  --template-export "$ASSET_ROOT/exports/template-r1" \
  --template-export-manifest-sha256 <TEMPLATE_EXPORT_MANIFEST_SHA256> \
  --hand right --view front \
  --train-split-id <SPLIT_ASSIGNMENTS_SHA256> \
  --parameters configs/zs32/training/template_opencv.example.json \
  --output-dir "$ASSET_ROOT/templates/right/front" --device 0

.venv/bin/zs32-train-anomaly \
  --recipe "$BASE_RECIPE" --topology "$TOPOLOGY" --roi "$ROI" \
  --dataset-release "$ASSET_ROOT/datasets/dataset-zs32-r1" \
  --anomalib-export "$ASSET_ROOT/exports/anomalib-r1" \
  --anomalib-export-manifest-sha256 <ANOMALIB_EXPORT_MANIFEST_SHA256> \
  --hand right --view front \
  --train-split-id <SPLIT_ASSIGNMENTS_SHA256> \
  --parameters configs/zs32/training/patchcore.example.json \
  --output-dir "$ASSET_ROOT/anomaly" --device 0
```

依次替换 view 为 `front/front_left/front_right/back/back_left/back_right`。每次保存 stdout 的完整 `artifact`，不可只记 checkpoint 路径。

两条训练命令都会在 Linux 上生成 `zs32.execution_receipt` schema v1：要求
clean Git commit/tree，绑定 `uv.lock` SHA256、实际 NVIDIA 设备、Python/CUDA/cuDNN/
driver，并通过 wheel `RECORD` 重算 torch/torchvision/anomalib/ultralytics/OpenCV/
NumPy 的安装文件树和实际 import origin。receipt 同时绑定 dataset、materialized
manifest、recipe、ROI、topology 及训练参数。训练结束、原子发布前必须重新观测；
中途 code/environment 变化则不得发布。template/anomaly metadata 与最终 release 都保留
完整 receipt，不接受只记版本号的训练记录。

## 6. 导入外部 YOLO `best.pt`

外部训练仍使用经审核的 ZS32 YOLO 工程。部署权重只能是训练输出 `best.pt`，`yolo26n.pt` 只是预训练初始化。bundle 必须包含 `best.pt/args.yaml/data.yaml/class_names.yaml`，其中类别只能是 `0: defect`。还必须由外部 trainer 生成 canonical `training_receipt.json`；它只是“外部 trainer 观测到这些输入/输出”的 attestation，不是训练因果证明。

receipt v3 必须使用排序 key、无空白的 canonical JSON 并以换行结尾，且严格包含：

- `best.pt/args.yaml/data.yaml/class_names.yaml` 四个 SHA256；
- 真实 `run_id/run_name/actual_seed`，且 `actual_seed` 与 `args.yaml` 一致；
- 实际 YOLO export 原子发布的 publication ID/root SHA256、`export_manifest.csv` SHA256 和 `data.yaml` SHA256；
- training-safe `training_export_policy.json` SHA256；旧 receipt v2 因未绑定训练数据隔离策略而拒绝；
- dataset release ID 和 `dataset_release.json` SHA256；
- `args.yaml` 中真实的 `data:` reference；
- Ultralytics 源码二选一：精确 clean git commit，或带内容寻址引用的 source bundle SHA256；
- 训练进程实际导入的 `ultralytics` package tree 确定性
  `runtime_package_tree_sha256`（排除 `__pycache__/*.pyc/*.pyo`）；
- RFC3339 带时区的 `attested_at` 与明确 `attested_by`。

`bundle/data.yaml` 必须与原子 YOLO export 里的 `data.yaml` 字节完全一致。importer 会打开并重新验证实际 dataset release 和 YOLO export，重算 train/model_val 物理件分配，并拒绝 calibration/test 行、隐藏文件、跨 split part 或弱化后的 policy；不接受只由 receipt 自我声明的 ID/hash。

```bash
.venv/bin/zs32-import-yolo \
  --bundle-dir /path/to/audited/yolo/run/bundle \
  --output-root "$ASSET_ROOT/yolo" \
  --import-id yolo-zs32-r1 \
  --dataset-release "$ASSET_ROOT/datasets/dataset-zs32-r1" \
  --yolo-export "$ASSET_ROOT/exports/yolo-r1" \
  --training-receipt /path/to/audited/yolo/run/training_receipt.json \
  --imgsz 640 --candidate-conf 0.001 --iou 0.7 --max-det 300
```

导入发布会原子打包 receipt，并在 `metadata.json` 中保留 receipt digest、export root/manifest/data digest、dataset 绑定、trainer source、run ID/name 和 seed。任何额外字段、非 canonical JSON、dirty source、hash/path/seed 矛盾都会 fail closed。

`ultralytics` 是外部 YOLO26 fork，不能用不明 PyPI 版本补位。这里重新验证
第 2 节已安装的实际 package tree，禁止在训练收据生成后换包：

```bash
test -z "$(git -C /path/to/audited/ultralytics status --porcelain=v1 --untracked-files=all)"
.venv/bin/python -c 'from pathlib import Path; import ultralytics; from zs32_inspection.models.yolo_receipt import runtime_package_tree_sha256; print(runtime_package_tree_sha256(Path(ultralytics.__file__).resolve().parent))'
```

输出必须与 receipt 一致。正式推理会再次重算，不接受环境变量自报的
commit/version。

## 7. 注册冻结 candidate

不要人工拼接六份 anomaly、六份 template 和 YOLO 输出。对当前
三相机右手 recipe，显式传入六个 anomaly `metadata/*.json`、六个
template `model.json` 和全局 YOLO `metadata.json`；四/五相机时按
topology 增加到八/十组。生成器会重新打开每个文件、验证 hash，把局部
artifact path 确定性改写为相对同一 `$ASSET_ROOT` 的路径，并从
recipe/topology 派生 `required_slots`：

```bash
.venv/bin/zs32-build-candidate-registration \
  --asset-root "$ASSET_ROOT" \
  --base-recipe "$BASE_RECIPE" --topology "$TOPOLOGY" --roi "$ROI" \
  --dataset-release "$ASSET_ROOT/datasets/dataset-zs32-r1" \
  --anomaly-metadata <RIGHT_FRONT_ANOMALY_METADATA_JSON> \
  --anomaly-metadata <RIGHT_FRONT_LEFT_ANOMALY_METADATA_JSON> \
  --anomaly-metadata <RIGHT_FRONT_RIGHT_ANOMALY_METADATA_JSON> \
  --anomaly-metadata <RIGHT_BACK_ANOMALY_METADATA_JSON> \
  --anomaly-metadata <RIGHT_BACK_LEFT_ANOMALY_METADATA_JSON> \
  --anomaly-metadata <RIGHT_BACK_RIGHT_ANOMALY_METADATA_JSON> \
  --template-model "$ASSET_ROOT/templates/right/front/model.json" \
  --template-model "$ASSET_ROOT/templates/right/front_left/model.json" \
  --template-model "$ASSET_ROOT/templates/right/front_right/model.json" \
  --template-model "$ASSET_ROOT/templates/right/back/model.json" \
  --template-model "$ASSET_ROOT/templates/right/back_left/model.json" \
  --template-model "$ASSET_ROOT/templates/right/back_right/model.json" \
  --yolo-metadata "$ASSET_ROOT/yolo/yolo-zs32-r1/metadata.json" \
  --output "$ZS32/candidate_registration.local.json"
```

输出是权限 `0600`、canonical JSON、no-replace 的
`zs32.candidate_registration`。所有 `relative_path` 都相对同一个
`$ASSET_ROOT`；绝对路径、`..`、symlink、hash 漂移、缺/重复 slot 都会在写出
前被拒绝。根字段为：

```text
schema, schema_version, product, anomaly_family,
anomaly_artifacts, yolo, recipe_digest, topology_digest,
roi_digest, roi_version, dataset_release_id, dataset_manifest_digest,
required_slots, templates
```

`required_slots` 的顺序必须由 topology/allowed hands 得到，不得手工删 view。注册命令会复核实际文件、训练 split、ROI、数据集和模型族，并原子生成 `candidate.json`、`template_assets.json`、`bound_recipe.json`、`calibration_spec.json`、`registration.json`：

```bash
.venv/bin/zs32-register-candidate \
  --candidate-id candidate-zs32-patchcore-r1 \
  --registration-id registration-zs32-r1 \
  --registration "$ZS32/candidate_registration.local.json" \
  --asset-root "$ASSET_ROOT" \
  --base-recipe "$BASE_RECIPE" \
  --topology "$TOPOLOGY" --roi "$ROI" \
  --dataset-release "$ASSET_ROOT/datasets/dataset-zs32-r1" \
  --calibration-split-id calibration-r1 \
  --test-split-id test-r1 \
  --fit-parameters configs/zs32/calibration/fit_strict_v1.json \
  --output-root "$ASSET_ROOT/registrations"
```

注册只产生 `registered` candidate，不改变生产指针。

## 8. 冻结模型打分与阈值标定

`zs32-score-calibration` 只接受 dataset schema v4 内嵌并哈希绑定的人工审批
calibration-target contract。模型仍会对每个 crop 的三个分支都执行并留下
`score_audit.jsonl`；`exclude` 分支不进入 `scores.jsonl` 或阈值拟合。held-out
test 不允许 `exclude`，score/calibrate/validate 三个边界都会按 dataset 内的原始
target 文件逐 key 重放并 fail closed；不能靠重封装 score publication 或
`exclude` 绕过端到端质量门槛。

```bash
export REG=$ASSET_ROOT/registrations/registration-zs32-r1

.venv/bin/zs32-score-calibration \
  --dataset-release "$ASSET_ROOT/datasets/dataset-zs32-r1" \
  --candidate "$REG/candidate.json" \
  --template-assets "$REG/template_assets.json" \
  --asset-root "$ASSET_ROOT" \
  --calibration-split-id calibration-r1 \
  --test-split-id test-r1 \
  --output-root "$ASSET_ROOT/score_runs" \
  --score-run-id score-zs32-r1 --device 0

.venv/bin/zs32-calibrate \
  --spec "$REG/calibration_spec.json" \
  --score-run "$ASSET_ROOT/score_runs/score-zs32-r1" \
  --candidate "$REG/candidate.json" \
  --asset-root "$ASSET_ROOT" \
  --recipe "$REG/bound_recipe.json" \
  --topology "$TOPOLOGY" --roi "$ROI" \
  --dataset-release "$ASSET_ROOT/datasets/dataset-zs32-r1" \
  --output-root "$ASSET_ROOT/calibrations" \
  --calibration-id calibration-zs32-r1
```

任一必需组样本不足、不可分、test 不完整时命令返回非部署状态，禁止继续。不要编辑 threshold JSON 伪造通过。

## 9. 最终 recipe、contract 与 candidate validation

```bash
export CAL=$ASSET_ROOT/calibrations/calibration-zs32-r1

.venv/bin/zs32-finalize-recipe \
  --base-recipe "$REG/bound_recipe.json" \
  --calibration "$CAL" \
  --output "$ZS32/final_recipe_zs32_r1.json"

.venv/bin/zs32-compile-contract \
  --recipe "$ZS32/final_recipe_zs32_r1.json" \
  --topology "$TOPOLOGY" --roi "$ROI" --fusion-policy "$POLICY" \
  --output "$ZS32/deployment_contract_zs32_r1.json"

.venv/bin/zs32-validate-candidate \
  --candidate "$REG/candidate.json" \
  --template-assets "$REG/template_assets.json" \
  --asset-root "$ASSET_ROOT" \
  --recipe "$ZS32/final_recipe_zs32_r1.json" \
  --topology "$TOPOLOGY" --roi "$ROI" --fusion-policy "$POLICY" \
  --dataset-release "$ASSET_ROOT/datasets/dataset-zs32-r1" \
  --score-run "$ASSET_ROOT/score_runs/score-zs32-r1" \
  --calibration "$CAL" \
  --heldout-acceptance-policy "$HELDOUT_ACCEPTANCE_POLICY" \
  --output-root "$ASSET_ROOT/validations" \
  --validation-id validation-zs32-r1
```

validation 会重新验证原始 immutable score-run publication，并逐字段绑定
calibration 记录中的 publication root、scores/audit hash、candidate、recipe、
dataset 以及 calibration/test split。`validation.json` 会显式记录
registration、score-run、calibration 三个 publication 的 ID/root，以及
scores/audit hash，禁止 downstream 通过可变路径反推本次验证输入。通过后会
生成一个不可覆盖的 atomic validation publication。validation 在转换 candidate
状态之前，用独立、版本化、canonical 且内容寻址的 held-out acceptance policy
检查最小正常/缺陷物理件数，以及 defect escape、normal reject、review 三项上限；
test split 存在 `exclude` 导致的不可评估件仍按 incomplete fail closed，不会被策略绕过。
任一越界均不发布 validation。其校验索引文件集必须恰好为
`candidate.json`（状态 `validated`）、`deployment_assets.json`、`validation.json`、
`heldout_acceptance_policy.json` 和 `heldout_acceptance.json`。后两者分别保留原始策略
字节和确定性 PASS 决策；后续不得抽取或手工重建，assembler 只接受同一完整
publication 内的原始路径。

## 10. 人工 promotion 与不可变 release

人工审核 golden parity、held-out 指标、模型/数据 provenance 和现场签字后，不能只
在命令行输入一个没有证据链的 promotion 字符串。批准人必须先生成一个唯一、canonical、
不可覆盖的 `promotion_receipt.local.json`。receipt 精确绑定 release、candidate、
validation publication、contract、calibration artifact、dataset manifest、held-out
acceptance policy/PASS decision，以及
golden parity、FAT、SAT 三项批准证据：

```json
{
  "schema": "zs32.promotion_receipt",
  "schema_version": 2,
  "product": "ZS32",
  "release_id": "release-zs32-r1",
  "promotion_id": "promotion-zs32-r1-approved-by-<NAME>",
  "approver": "<APPROVER>",
  "approved_at": "2026-07-14T17:55:00+08:00",
  "candidate_id": "<CANDIDATE_ID>",
  "candidate_digest": "<CANDIDATE_SHA256>",
  "validation_publication_id": "validation-zs32-r1",
  "validation_publication_root_sha256": "<VALIDATION_ROOT_SHA256>",
  "validation_record_sha256": "<VALIDATION_JSON_SHA256>",
  "contract_sha256": "<CONTRACT_SHA256>",
  "calibration_artifact_sha256": "<CALIBRATION_ARTIFACT_SHA256>",
  "dataset_manifest_sha256": "<DATASET_RELEASE_JSON_SHA256>",
  "heldout_acceptance_policy_sha256": "<HELDOUT_ACCEPTANCE_POLICY_SHA256>",
  "heldout_acceptance_decision_sha256": "<HELDOUT_ACCEPTANCE_DECISION_SHA256>",
  "evidence": {
    "golden_parity": {
      "status": "PASS",
      "evidence_sha256": "<GOLDEN_PARITY_REPORT_SHA256>",
      "reason": null
    },
    "fat": {
      "status": "PASS",
      "evidence_sha256": "<FAT_SIGNED_RECORD_SHA256>",
      "reason": null
    },
    "sat": {
      "status": "NOT_APPLICABLE",
      "evidence_sha256": null,
      "reason": "<WHY_SAT_IS_NOT_APPLICABLE_TO_THIS_RELEASE>"
    }
  }
}
```

三项 evidence 的 key 必须恰好为 `golden_parity`、`fat`、`sat`。golden parity
是 release 前置门，必须为 `PASS`。`PASS` 必须携带证据文件 SHA256 且
`reason:null`；FAT/SAT 尚不适用时只能使用 `NOT_APPLICABLE`，必须给出
单行原因且 digest 为 `null`。`NOT_RUN`、`FAIL`、缺字段、未知字段、无时区时间戳均
fail closed。上面的可读 JSON 只是 draft，必须用 `jq -cS` 转为与程序一致的
canonical bytes，再用 no-replace hard link 发布；目标存在时禁止覆盖：

```bash
export PROMOTION_DRAFT="$ZS32/promotion_receipt.draft.local.json"
export PROMOTION_RECEIPT="$ZS32/promotion_receipt.local.json"
(
  set -euo pipefail
  test ! -e "$PROMOTION_RECEIPT"
  PROMOTION_TMP=$(mktemp "${PROMOTION_RECEIPT}.tmp.XXXXXX")
  trap 'chmod 0600 "$PROMOTION_TMP" 2>/dev/null || true; rm -f "$PROMOTION_TMP"' EXIT
  jq -cS . "$PROMOTION_DRAFT" > "$PROMOTION_TMP"
  sync -f "$PROMOTION_TMP"
  chmod 0444 "$PROMOTION_TMP"
  ln "$PROMOTION_TMP" "$PROMOTION_RECEIPT"
  rm -f "$PROMOTION_TMP"
  trap - EXIT
  sync -f "$(dirname "$PROMOTION_RECEIPT")"
)
```

receipt 的 `approved_at` 不得晚于 assemble 的 `created_at`。receipt 与 draft 都留在
Linux，均不得 push。assembler 会在复制前后复验 receipt，并把原始 canonical bytes
封存为 `provenance/promotion_receipt.json`；candidate provenance 写入其精确 SHA 并镜像
promotion ID、approver 和 approved time，loader 必须确认两份记录完全一致。

`code_version.local.json` 必须只包含下列六个字段，其中 commit/tree/lock 必须与第 1
节的已核对交接值完全一致，且 `dirty:false`：

```json
{
  "schema_version": 1,
  "git_commit": "<HANDOFF_COMMIT>",
  "git_tree": "<HANDOFF_TREE>",
  "dirty": false,
  "dependency_lock_sha256": "<HANDOFF_UV_LOCK_SHA256>",
  "build_id": "<UNIQUE_LINUX_BUILD_ID>"
}
```

该 `.local.json` 留在 Linux，已由 `.gitignore` 排除，不得 push。在生成它之后、
assemble 之前再次检查 clean worktree 和三个 handoff 值。

在最终部署机、最终 Python 环境和正式 `CUDA_VISIBLE_DEVICES` 映射下生成 runtime
environment receipt。receipt 严格冻结 Linux kernel、`/etc/os-release` 的
ID/VERSION_ID、GPU UUID、NVIDIA driver、CLI device 到 physical GPU 的映射，
以及 Python、torch、torchvision、anomalib、ultralytics、OpenCV、NumPy、torch
CUDA build 和 cuDNN 版本。六个关键 Python distribution 还必须记录其原始
`METADATA`/`RECORD` SHA256，逐项重算 RECORD 中每个文件的 SHA256/size，生成
installed-tree digest，并证明实际 import origin 由该 RECORD 唯一拥有；不能只用
可伪装的 `__version__` 或未核验的 RECORD 文本作为依赖锁。editable、缺 hash、
symlink 或 prefix 外文件均 fail closed。训练机与部署机不同时，禁止用训练机
receipt 代替部署机 receipt。

```bash
export CUDA_VISIBLE_DEVICES=3
export RUNTIME_ENV_RECEIPT="$ZS32/runtime_environment.local.json"

.venv/bin/zs32-snapshot-runtime-environment \
  --device 0 \
  --output "$RUNTIME_ENV_RECEIPT"

export RUNTIME_ENV_FILE_SHA256=$(sha256sum "$RUNTIME_ENV_RECEIPT" | awk '{print $1}')
```

snapshot 先写同目录临时文件并 `fsync`，再以 no-replace hard link 原子发布为
`0444` 文件；目标已存在时必须拒绝，不得覆盖旧 receipt。

该 `.local.json` 同样不得 push。若 GPU、driver、device mapping 或上述任一包版本
变化，必须重新完成受影响的 Linux 验证并组装新 release；禁止覆盖旧 receipt 或旧
release。

```bash
export VAL=$ASSET_ROOT/validations/validation-zs32-r1

.venv/bin/zs32-assemble-release \
  --release-id release-zs32-r1 \
  --output-root "$ASSET_ROOT/releases" \
  --created-at 2026-07-14T18:00:00+08:00 \
  --promotion-receipt "$PROMOTION_RECEIPT" \
  --contract "$ZS32/deployment_contract_zs32_r1.json" \
  --asset-root "$ASSET_ROOT" \
  --candidate "$VAL/candidate.json" \
  --deployment-assets "$VAL/deployment_assets.json" \
  --fusion-policy "$POLICY" \
  --capture-gate-policy "$ASSET_ROOT/gate-publications/gate-zs32-right-r1/capture/gates/policy.json" \
  --capture-gate-asset-root "$ASSET_ROOT/gate-publications/gate-zs32-right-r1" \
  --dataset-release-manifest "$ASSET_ROOT/datasets/dataset-zs32-r1/dataset_release.json" \
  --dataset-manifest-sha256 <DATASET_RELEASE_JSON_SHA256> \
  --code-version "$ZS32/code_version.local.json" \
  --runtime-environment-receipt "$RUNTIME_ENV_RECEIPT" \
  --runtime-environment-sha256 "$RUNTIME_ENV_FILE_SHA256"
```

assembler 会先要求两个 descriptor 来自同一个、文件集精确的 atomic
validation publication，再逐字节确认五个 JSON 都为 canonical JSON。
`validation.json` 必须同时绑定 candidate、contract/recipe、dataset manifest、
calibration/registration/score-run publication roots、calibration artifact、
calibration/test split，并且 `manual_promotion_required` 必须严格为 `true`。
策略和决策 SHA256 必须同时与 validation publication、promotion receipt 和
release candidate provenance 一致；assembler 与 loader 都会用 release 内的
calibration metrics 重新执行同一 acceptance policy，不信任单独的 `PASS` 字符串。

validation publication 会在调用组装器紧前和 release 发布后各复验一次。
validation publication ID/root SHA256 和 `validation.json` SHA256 会写入
release provenance，完整 validation publication 也会封存在 release 内，并由
release loader 重建其 checksum root 后复验语义绑定。loader 会重新解析 canonical
promotion receipt，逐字段核对 candidate/validation/contract/calibration/dataset/
release 身份，并要求 candidate provenance 引用 receipt 的精确 SHA。loader 还会用 release 中
五个 calibration 文件的原始 bytes，按 atomic publisher 的 canonical checksum
index 格式独立重建 calibration publication root。registration 和 score-run 的
完整 publication 没有封存在 release 内，因此 loader 只把它们的 ID/root/hash 作为
已由 validation publication 认证的不可变断言，不声称可独立重建。任何可重建的
root digest、checksum index 或文件变化都使命令失败。组装器最后会用正式 release loader
在 staging 内做完整复验后才 no-replace rename。不要建立可变的 `current`
symlink 代替显式 release id。

score-run schema v3 同样内置 execution receipt、逐分支完整 audit 证据和 dataset-bound
calibration targets；calibration input provenance schema v3
保留其 receipt SHA256，validation schema v3 继续镜像该 SHA256，并增加 acceptance
policy/decision 内容哈希。release 另外封存
`dataset_canonical_semantics.json`、`dataset_governance.json` 和
`dataset_split_assignments.json`，loader 会重建 semantics 审批、SplitPolicy 和 PNG codec
身份；不只依赖 `dataset_release.json` 里的自我声明。

`provenance/runtime_environment.json` 使用 canonical JSON，并同时由内部语义 digest
和 release `checksums.sha256` 的文件 digest 约束。release loader 会严格拒绝缺失、
未知字段、非 canonical bytes 或自校验 digest 不一致，不能把可变的系统查询结果留到
模型加载后再解释。

## 11. 正式推理与结果语义

```bash
.venv/bin/zs32-inspect \
  --release "$ASSET_ROOT/releases/release-zs32-r1" \
  --capture-set-root "$ZS32/raw/<capture_session>/images/<capture_set_id>" \
  --inspection-id inspection-zs32-000001 \
  --output-root "$ASSET_ROOT/inspections" \
  --device 0
```

`zs32-inspect` 必须从 release 记录的同一个 clean Git checkout 启动。它会在
读取 capture 或加载模型前，实测当前 Git top-level、HEAD commit、tree、完整
worktree 状态和已跟踪 `uv.lock` 的 SHA256，并与
`provenance/code_version.json` 精确比较；任一不一致直接发布
`SYSTEM_ERROR`。因此正式机不得在 release 组装后修改源码或 lock，也不得从只
包含 wheel、没有对应 Git checkout 的目录启动当前部署形态。

随后 inspect 会在读取 capture 和加载任何模型前重新采集当前 runtime receipt，并与
release 精确逐字段比较。kernel/OS、GPU UUID、driver、`CUDA_VISIBLE_DEVICES`、
`--device` 映射、Python/torch/torchvision/anomalib/ultralytics/OpenCV/NumPy、
CUDA/cuDNN、distribution `METADATA`/`RECORD`、installed-tree 或实际 import
origin 任一漂移，均
发布 `runtime_environment` 预检审计和 `SYSTEM_ERROR`，绝不进入 template 或二层模型。

端到端顺序固定为：采集时执行 quality/registration → 仅发布 gate 全通过的
complete capture → inspect 复验 capture identity 与持久化 gate evidence →
canonical crop once → template → anomaly + YOLO 并行分支 → strict fusion。
模板全部 PASS 后才能同时启动 anomaly 与 YOLO；协调器必须等待两边结束，
一边故障不得取消另一边。持久化顺序固定为 `anomaly -> yolo`、各分支内按
view ID 排序，不得依赖线程完成顺序。

- 模板任一 NG：立即 `NG_TEMPLATE`，不加载二层模型；
- quality/registration 失败：由 `zs32-capture` 直接返回 `RETAKE`，失败采集只进
  `_incomplete` 隔离区，不能作为 `zs32-inspect` 输入；
- capture 身份错误：`INVALID_CAPTURE`；
- 资产、配置、模型运行错误：`SYSTEM_ERROR`；
- anomaly/YOLO 任一 STRONG：保留对应 NG 证据；
- 任一 GRAY 且无 STRONG：`REVIEW`；
- 所有必需 anomaly 与 YOLO evidence 均 CLEAR：才可 `OK`；
- `SYSTEM_ERROR` 即使保留了强 NG evidence，也绝不产生 released status。

因此 `RETAKE` 是采集阶段的业务终态，而不是合法 complete capture 在 inspect
阶段通常能够产生的状态。inspect 中仍保留 gate-failure 防御分支，用于拒绝未来
错误接线或不可信输入，但正式存储合同不会发布 gate 失败的 complete capture。

## 12. Linux 验收顺序

先执行纯 contract/unit tests，再执行 GPU/真实资产/相机集成测试和 golden replay。Mac 不执行这些命令。

```bash
.venv/bin/pytest -q tests/unit/zs32_refactor tests/unit/tools/test_zs32_phase0.py
```

三种 anomaly backend 的真实 Anomalib 构建/恢复测试必须在 Linux + NVIDIA
环境单独确认；它们不是 Mac 或无框架 mock 结论：

```bash
.venv/bin/pytest -q -m gpu \
  tests/unit/zs32_refactor/models_calibration/test_patchcore_explicit_backbone.py \
  tests/unit/zs32_refactor/models_calibration/test_anomalib_family_construction.py
```

其中 release 输入边界可单独重跑：

```bash
.venv/bin/pytest -q \
  tests/unit/zs32_refactor/calibration/test_heldout_acceptance_policy.py \
  tests/unit/zs32_refactor/calibration/test_validate_candidate_boundary.py \
  tests/unit/zs32_refactor/runtime/test_promotion_receipt.py \
  tests/unit/zs32_refactor/runtime/test_assemble_release_validation_boundary.py
```

至少注入：缺 view、错 serial、坏图 hash、pending left ROI、模板文件损坏、
anomaly checkpoint 损坏、YOLO metadata/weights 损坏、阈值缺组、score-run 多文件/
少文件、validation descriptor 跨 publication 混用、validation 非 canonical JSON、
validation 记录错绑 candidate/contract/calibration/dataset/split/publication root、
held-out 正常/缺陷件数不足、defect escape/normal reject/review 越界、incomplete
evaluation、acceptance policy/decision 被替换或与 promotion/release provenance 错绑、
`manual_promotion_required:false`、promotion receipt 非 canonical/缺时区/错绑
candidate 或 validation/evidence 状态非法、发布包新增文件、symlink/hardlink、anomaly 故障后
YOLO STRONG、YOLO 故障前 anomaly STRONG、anomaly heatmap/overlay 缺失、改字节或
错 hash、YOLO raw row 非法携带 anomaly visual。成功 inspection 的每个必需视角
必须同时发布 `heatmaps/anomaly/<view>.png` 与 `overlays/anomaly/<view>.png`，其
SHA256 必须与 raw/calibrated evidence 中的引用一致。

只有 Linux 全量验证与 Phase 0 golden parity 都通过，才允许进入 Phase 7 删除 legacy pipeline/C789/FX11。当前仓库中的旧代码仍应保留。

## 13. Linux 证据回传合同

回传给 Mac/审核人的是小型证据，不是任何原图、权重或 release 目录。
建议在仓库外的 `$ZS32/linux_evidence/<HANDOFF_COMMIT>/` 保存并回传：

- `source_identity.txt`：`origin/main`、HEAD commit、tree、`uv.lock` SHA256、clean
  `git status` 结果；
- `environment.txt`：Linux/driver/GPU/CUDA/Python/PyTorch/uv 版本，以及
  `uv sync --frozen` 命令和 exit code；
- `phase0.txt`：inventory/freeze/strict verify 的命令、exit code、trusted bundle
  root 和已提交 pointer 的 Git commit；
- `tests.txt`：完整测试命令、exit code、passed/failed/skipped 统计与故障注入
  结果；
- `publications.txt`：dataset/candidate/score/calibration/validation/release ID、根
  digest、人工 promotion ID 和对应命令 exit code；
- `acceptance.txt`：golden parity 结论、held-out 结论、现场审核人与带时区
  时间。

每个证据文件同时回传 SHA256；日志中的 token、凭证、内网地址和不必要
的绝对路径必须先脱敏。这些证据不得 push 到 GitHub；GitHub 只接收审核后的
Phase 0 pointer 或后续明确定义的轻量签名 pointer。任何证据缺失都只能记为
“Linux 验证未完成”，不得由 Mac 静态检查替代。
