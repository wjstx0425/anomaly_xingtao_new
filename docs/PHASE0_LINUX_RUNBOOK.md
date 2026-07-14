# ZS32 Phase 0 Linux 冻结运行手册

状态：代码已准备，等待在 Linux + NVIDIA GPU 主机执行。
适用范围：当前只冻结右手、三相机、双面六视角的旧系统基线。
禁止事项：不得在 Mac 上执行本文的发现、测试、推理、冻结或校验命令。

## 1. Phase 0 的交付物

Phase 0 不重构模型逻辑，只形成一份可追溯的旧系统基线：

- 明确的 Git commit 和 clean worktree；
- Linux、GPU、CUDA、Python、依赖和 MVS SDK 快照；
- 当前右手权威 YOLO ROI，以及现场实际使用的相机曝光、增益、HDR 和采集参数配置；
- 六视角 PatchCore checkpoint、YOLO `best.pt`、模板、阈值和运行配置的 SHA256；
- YOLO/PatchCore 训练数据 manifest、split、ROI/标签变换和训练 provenance 的 SHA256；
- 经人工确认的真实六视角 golden cases；
- 每个 case 至少两次旧链路 replay 输出，以及一份绑定这些不可变 replay 的人工审核 receipt；
- 可发现任何缺失、额外或被修改文件的 bundle root；
- Git 中保存的 bundle pointer，作为 bundle root 的外部信任锚。

现有旧数据若缺少采集 manifest，只能用于 `refactor_parity_only`。它不能被包装成独立准确率验收集；正反面属于同一物理件和目录标签可信性必须由人签署。

## 2. Mac 上只做代码同步

在 Mac 审阅本次新增文件后，由你自行选择文件、提交并推送。不要
复制一份过期的固定 `git add` 列表；先检查当前完整 diff，再显式确认
staged 边界：

```bash
git status --short --untracked-files=all
git diff --check
git add -A
git diff --cached --stat
git diff --cached --name-only
```

`git add -A` 之后必须人工逐项审核；如有不应提交的文件，只取消该文件的
staging 后重新审核。提交前可用下列守卫检查常见大资产和本地配置：

```bash
test -z "$(git diff --cached --name-only | grep -E '\.(pt|pth|ckpt|onnx|engine|safetensors|bundle)$' || true)"
test -z "$(git diff --cached --name-only | grep -E '(^|/)raw/|(^|/)artifacts/|(^|/)runs/|\.local\.(json|ya?ml)$' || true)"
```

守卫通过后由你决定 commit message 并 push。push 完成后记录下列三个值，
作为发给 Linux 的 handoff 记录：

```bash
git status --porcelain=v1 --untracked-files=all
git rev-parse HEAD
git rev-parse HEAD^{tree}
shasum -a 256 uv.lock
```

`git status` 必须为空。分别保存 commit、tree 和 `uv.lock` SHA256，仅说“已推到
main”不足以启动 Linux freeze。

不要提交以下内容：

- golden PNG；
- PatchCore/Yolo 权重；
- 模板和阈值实体；
- `*.local.json`；
- replay 输出；
- Phase 0 bundle；
- Linux 环境/测试/推理日志和证据归档。

GitHub 可以接收 source、docs、reviewed small config、`uv.lock` 和后续审核过的
`baselines/zs32/*.pointer.json`。pointer 是 digest 信任锚，不是 bundle 的替代品，
不得包含访问凭证或可变网络目录。

## 3. Linux 拉取同一提交

以下所有命令都在 Linux + NVIDIA 主机执行：

```bash
export HANDOFF_COMMIT=<COMMIT_FROM_MAC>
export HANDOFF_TREE=<TREE_FROM_MAC>
export HANDOFF_UV_LOCK_SHA256=<UV_LOCK_SHA256_FROM_MAC>

cd /path/to/anomaly_xingtao
git fetch --prune origin
git checkout main
test -z "$(git status --porcelain=v1 --untracked-files=all)"
git pull --ff-only origin main
test "$(git rev-parse HEAD)" = "$HANDOFF_COMMIT"
test "$(git rev-parse HEAD^{tree})" = "$HANDOFF_TREE"
test "$(sha256sum uv.lock | awk '{print $1}')" = "$HANDOFF_UV_LOCK_SHA256"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
nvidia-smi
```

任一 handoff 校验失败都立即停止，禁止改用 Linux 上的旧 checkout。正式
`freeze` 前，`git status` 必须没有任何输出。工具不会提供 dirty-tree
绕过参数。

按照 Linux 当前 CUDA 版本选择一个且仅一个 CUDA extra，例如 CUDA 12.6；
`test` 是独立的测试依赖 extra，可以与所选 CUDA extra 同时启用：

```bash
uv lock --check
uv sync --frozen --extra cu126 --extra test
.venv/bin/python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda)"
```

`uv lock --check` 必须先证明 `pyproject.toml` 与 `uv.lock` 一致；`--frozen` 是依赖锁的执行边界。禁止在 Linux 现场改写 `pyproject.toml` 或
`uv.lock` 来解决环境问题。安装后再次检查 commit/tree/lock 和 clean
worktree。Phase 0 bundle 会同时冻结 `pyproject.toml` 与 `uv.lock` 的 SHA256。

Ultralytics 和现场所用 MVS SDK 也必须安装在同一运行环境，并能被 Phase 0 工具记录。
Ultralytics 必须按 `ZS32_LINUX_REFACTOR_RUNBOOK.md` 第 2 节先验证受审核 fork、再安装到
同一 `.venv`；之后只用 `.venv/bin/...`，避免隐式 sync 删除它。不要为了通过守卫安装 CPU 版 PyTorch。

## 4. 发现 legacy golden 候选

候选发现只建立待人工审核清单，不会自动批准样本：

```bash
mkdir -p /DATA/ljl/zs32_phase0_work

.venv/bin/python tools/zs32_phase0.py discover-legacy \
  --data-root /DATA/ljl/zs32_legacy \
  --topology configs/zs32/topology/zs32_3cam_double_side_v1.json \
  --hand right \
  --output /DATA/ljl/zs32_phase0_work/golden_candidates-01.json
```

发现器按以下合同分组：

- 根目录是 `<data-root>/right/<view>/...`；
- 文件名由当前采集脚本产生，包含 `groupNNN`、六位 image index 和 `single/fused`；
- 同一候选必须正好覆盖 `front/front_left/front_right/back/back_left/back_right`；
- 重复 view 立即失败；不完整组只计数，不进入候选 cases。

当前 Phase 0 命令只接受 `--hand right`。左手虽然是最终产品必须支持的独立配置，但当前 ROI 仍为 `pending`，所以左手候选、baseline 和 release 均不得混入本次冻结；左手 ROI 审核完成后应使用新的 freeze ID 和独立 golden selection 建立后续基线。

## 5. 人工选择并签署 golden cases

从候选文件复制少量代表性 cases 到：

```text
configs/zs32/phase0/golden_selection.local.json
```

该文件已被 `.gitignore` 排除。每个保留 case 必须人工填写：

- `expected_target_status`；
- 真实 `part_instance_id`；
- `physical_part_identity_verified: true`；
- `label_verified: true`；
- `verified_by` 和带时区的 `verified_at`；
- `parity.mode` 与原因。

任何 `REPLACE`、`PENDING`、`TBD`、空签名、缺视角、重复图片、非 PNG 或非 `4024×3036` 图像都会阻止冻结。

建议至少覆盖：

- 2 个旧系统全 clear 的正常件；
- 模板 mismatch；
- anomaly 强阳性；
- YOLO 强阳性；
- 灰区 REVIEW；
- 质量失败 RETAKE；
- 配准失败 RETAKE。

场景名不是人工标签豁免：inventory 会从 Stage 32 原始产物重新推导 evidence。
`normal_clear` 必须实际为三分支全 CLEAR；模板 mismatch 必须产生 template STRONG
并短路二层；anomaly/YOLO strong 必须由对应分支 STRONG 且另一二层分支真实运行；
gray review 必须存在 GRAY 且没有 STRONG；两类 RETAKE 还分别要求对应 gate CSV
中出现真实 hard failure。`semantic` 或 `must_change` 不会绕过这些覆盖条件。

除六张原图外，每个 case 还必须有当次旧链真实使用的
`quality_gate.csv`、`registration_results.csv` 和
`geometry_predictions.csv`。三份 CSV 必须各有六行，`part_id`
等于已签署的 `part_instance_id`，每个 view 只出现一次。这些是旧系统
的真实输入，不得为了跑通 replay 手写全 PASS CSV。每个 case 还必须从
`gate_producer_receipt.example.json` 复制并填写一份真实 producer receipt，
把生成命令、producer commit、已登记 source/config/threshold 资产、六张输入图
hash、三份输出 CSV hash、capture identity 和带时区人工复核绑定在一起。

缺少真实样本时保持 freeze spec 为 `draft`，不要用伪造案例宣称 Phase 0 完成。缺视角、坏权重等系统故障可在后续测试中由真实 case 派生，但必须标为 fault injection。

## 6. 汇集外部资产

仓库不包含正式模型和现场资产。Linux 上至少要准备：

- 六个右手 PatchCore checkpoint 及各自阈值/训练摘要；
- 旧链路现场实际使用的相机 acquisition config；不能用示例默认值冒充；
- YOLO 训练输出 `best.pt`，不能把 `yolo26n.pt` 登记为部署权重；
- YOLO `args.yaml`、`data.yaml`、类别映射和 Ultralytics 版本；
- YOLO 数据集 manifest：固定 train/val/test split、原始 4K 图、ROI crop、调整后标签及其逐文件 hash，并绑定 ROI/topology ID；
- 实际训练/推理所用 Ultralytics fork 的 clean commit，并用 `git bundle` 固定源码；`cls_remap` 等自定义行为不能只靠 PyPI 版本描述；
- 六视角模板模型目录，以及 Stage 31 真实产生的单个 `thresholds.json`；
- PatchCore 训练 provenance 与数据集 manifest：固定每视角训练数据、split、训练命令/参数、checkpoint 对应关系及逐文件 hash；
- Stage 31/32 当前使用的融合、模型、ROI 配置；
- 资产对应的数据集/manifest hash。

复制并修改 freeze spec：

```bash
cp configs/zs32/capture/hikvision.example.json \
  configs/zs32/capture/hikvision.local.json
cp configs/zs32/phase0/phase0_freeze.example.json \
  configs/zs32/phase0/phase0_freeze.local.json
```

先把 `hikvision.local.json` 改成旧链路在现场真实使用的曝光、增益、HDR、超时、
融合和 PNG 参数；若旧链路由多个文件或命令行共同决定这些值，应生成一份经人工
签署的等价快照并在 evidence 中说明来源。示例默认值不能作为现场事实。
该文件在 Phase 0 中作为 legacy 基线资产冻结；进入新采集主链时，必须再作为
`capture/acquisition/hikvision.json` 被 capture gate policy schema v2 以 ArtifactRef
绑定并原子发布，不得在 `zs32-capture` 命令行上松散传入。

先在真实 Ultralytics 源码仓库确认 clean commit，再生成源码 bundle（路径按现场修改）：

```bash
git -C /path/to/ultralytics status --porcelain=v1 --untracked-files=all
git -C /path/to/ultralytics rev-parse HEAD
git -C /path/to/ultralytics bundle create \
  /DATA/ljl/zs32_assets/source/ultralytics-training-source.bundle HEAD
```

先把路径、MVS 版本和其他 `REPLACE_ME` 改为真实值；`camera_sdk.fingerprint_paths`
必须同时列出实际 import 的 Python wrapper 和至少一个现场加载的 native `.so`，不能
只记录人工版本字符串。`expected_sha256: null` 可以保留到首次预检查。审核工具计算
出的 digest 后，必须把每一个 required asset 的 digest 回填为真实
`expected_sha256`，否则 freeze 会拒绝。当前右手 Phase 0 的全部 normative asset
都必须 `required:true` 且 `copy_into_bundle:true`：只有这样 bundle 在脱离原 Linux
可变路径后才能重新验证六个 PatchCore checkpoint、YOLO `best.pt`、模板、阈值和
源码 bundle。非复制的 `immutable_storage_reference` 仅保留为未来合同能力，不得
用于本次 freeze。

当前 spec 还显式记录了 YOLO provenance blocker：训练快照写的是 `seed: 43`，而运行目录名含 `seed42`。必须读取 Linux 上该 run 的真实 `args.yaml`，确认实际 seed，并把结论写入 `evidence` 后才能将 `resolved` 改为 `true`。仅修改目录名或凭记忆选择一个 seed 都不算解决。

`data.yaml` 不是数据集内容证明，checkpoint 目录也不是训练 provenance。`training-dataset-manifest-completeness` blocker 只有在 YOLO 与 PatchCore manifest 都包含逐文件 hash、split、ROI/topology 绑定、标签变换和训练配置绑定并经人工审核后才能置为 `resolved: true`。大数据本体不列为本次 normative bundle asset，可以留在 Linux 数据盘；但它的内容寻址 manifest 必须作为 normative asset 复制进 bundle 并固定 hash。当前 spec 中任一 normative 项设置 `copy_into_bundle:false` 都会失败。

`legacy-gate-evidence-provenance` 是当前硬阻塞：仓库内没有已证明可为 ZS32 生成 case-local quality、registration、geometry CSV 的完整旧链 producer。在 Linux 上找到真实历史 receipt 后，必须同时固定生成命令、代码 commit、配置、阈值、capture identity 以及 source/evidence hash；不得手写六行全 PASS CSV。如果这些历史产物从未存在，必须改为冻结更早、真正可运行的旧入口，或把补建 gate 明确标为 `must_change` commissioning，不能宣称 legacy parity。

新重构链路还要求外部 YOLO trainer 交付 canonical v3
`training_receipt.json`，并在 Linux 导入时实际打开、验证 dataset release
与 YOLO export 原子发布。receipt 必须绑定四个训练输出 hash、export
root/manifest/data hash、training-safe `training_export_policy.json` SHA256、
dataset ID/manifest hash、`args.yaml` 的 data reference、
run ID/name/seed、clean commit 或内容寻址 source bundle，以及训练进程实际
导入的 `ultralytics` package tree SHA256。该 receipt 是外部 trainer 观测
attestation，不得当作训练因果证明。

目录资产使用确定性 tree hash：

```text
relative_path NUL file_sha256 NUL size LF
```

任何 symlink、空目录、缺失资产或 hash 不一致都会失败。
工具还会打开 legacy runtime registry，要求 `patchcore` 恰好覆盖六个
required views、六个 registry checkpoint hash 唯一且与 checkpoint bundle
中恰好六个 `.ckpt` 一一对应，并要求 registry 的 YOLO hash 与
实际 `best.pt` 一致。模板目录必须覆盖六个 view，阈值资产必须是
Stage 31 产生且覆盖六视角的精确 `thresholds.json`。
Ultralytics 源码必须是能通过 `git bundle verify` 的 `.bundle`。因此
工具还会定位 `import ultralytics` 的实际 package root，拒绝该 package 下任何
未跟踪/缺失源码，记录 package-tree SHA256 与 clean Git commit/tree，并要求该
commit 正是已登记 source bundle 的 head；只对上 distribution 版本号不算绑定。
因此
`/home/yunjing` registry 与 `/home/ljl` 实际权重的冲突必须在 replay
前解决，不能只在 blocker 文字中写“已处理”。

再从 `configs/zs32/phase0/legacy_replay_plan.example.json` 复制一份
`legacy_replay_plan.local.json`。该 plan 的入口固定为
`pipeline/32_run_zs32_multimodel_inference.py`，只允许填现场 GPU 设备参数
和 case 到三份 gate CSV 及 producer receipt asset ID 的映射。每份 gate CSV、
每份 receipt 及 plan 本身
都要在 freeze spec `assets` 中登记为 `required:true`、
`copy_into_bundle:true`；对应 kind/role 分别是：

```text
replay_plan / legacy_replay_plan
branch_evidence_csv / legacy_quality_evidence
branch_evidence_csv / legacy_registration_evidence
branch_evidence_csv / legacy_geometry_evidence
producer_receipt / legacy_gate_producer_receipt
git_source_bundle / legacy_gate_producer_source
```

工具会将 plan、所有 gate CSV、producer receipt 及其引用资产的 hash 通过
`source_binding.asset_sha256_by_id` 绑定到每次 replay，并将实体复制进最终
bundle。

`legacy_replay_plan.cases` 必须与 golden selection 中的 cases 一一对应，
不多不少。每个 case 必须使用自己独立的 quality、registration、geometry
三个 asset ID，任何 gate CSV 都不得跨 case 复用。freeze spec 的
producer receipt 同样必须每 case 独立、不可复用，并且其中引用的 source、
config、threshold 也必须是 freeze spec 中 `required:true` 且
`copy_into_bundle:true` 的内容寻址资产。freeze spec 的
`baseline_outputs` 还必须为每个 case 登记至少两个不同的 `run_id`；示例中的
两条只是单 case 模板，实际填写时要为全部 golden cases 展开。

## 7. 在旧链路上执行两次 replay

在第一次 replay 前，先从 clean Git checkout、当前资产和已签署
golden selection 生成不可覆盖的 source-binding 模板：

```bash
.venv/bin/python tools/zs32_phase0.py replay-bindings \
  --spec configs/zs32/phase0/phase0_freeze.local.json \
  --output /DATA/ljl/zs32_phase0_work/replay-bindings-01.json
```

该命令不要求 `baseline_outputs` 已存在，因此打破“生成 inventory
前就必须先填 replay hash”的循环；但它仍要求 Linux + NVIDIA、
clean HEAD==origin branch、完整 normative assets/scenarios 和右手六视角。
runner 会从 bindings 文件读取对应 case 的
`source_bindings.<case_id>`，并原样写入两次 replay 的
`phase0_case_result.json.source_binding`，无需也不得人工编辑结果文件。
任一 Git、环境、资产、ROI、topology 或 golden image 在绑定后改变，最终
inventory 都会拒绝这两次 replay；这种情况必须用递增文件名重新生成一份
bindings 文件，并使用新的 `run_id` 和输出目录重跑。schema 中不存在需要
人工填写的“binding ID”。

现在使用 Stage 32 专用 runner，不再手工组装 evidence：

```bash
.venv/bin/python tools/zs32_phase0.py replay \
  --spec configs/zs32/phase0/phase0_freeze.local.json \
  --bindings /DATA/ljl/zs32_phase0_work/replay-bindings-01.json \
  --case-id right-normal-001 \
  --run-id replay-01 \
  --output /DATA/ljl/zs32_phase0_runs/replay-01/right-normal-001

.venv/bin/python tools/zs32_phase0.py replay \
  --spec configs/zs32/phase0/phase0_freeze.local.json \
  --bindings /DATA/ljl/zs32_phase0_work/replay-bindings-01.json \
  --case-id right-normal-001 \
  --run-id replay-02 \
  --output /DATA/ljl/zs32_phase0_runs/replay-02/right-normal-001
```

runner 使用 `sys.executable` 调用固定 Stage 32 `fuse` 入口，不经过
shell；三份 gate CSV 先按 hash 复制到隔离工作目录，内层 Stage 32
`command_argv` 使用固定相对输入/输出路径，所以同一 case 两次
记录的真实命令完全一致。外层 `--run-id/--output` 只负责隔离 Phase 0
证据目录。输出目录不可覆盖，成功时通过 Linux no-replace rename
原子发布。

对每个 golden case 在同一 Linux 环境至少运行两次，输出到彼此隔离、不可覆盖的目录：

```text
/DATA/ljl/zs32_phase0_runs/replay-01/<case_id>/
/DATA/ljl/zs32_phase0_runs/replay-02/<case_id>/
```

每个目录至少保留：

- 精确的 `command_argv` 字符串数组，不用一条存在 shell 歧义的命令字符串；
- stdout、stderr、exit code；
- runtime summary；
- template、PatchCore、YOLO evidence；
- 最终融合状态；
- audit 和必要 overlay；
- runner 生成的 `phase0_case_result.json`，将 `observed_legacy` 与 `target_contract_expected` 分开；
- runner 最后生成的 `replay_root.json`，固定该 replay 目录的完整证据树。

`configs/zs32/phase0/phase0_case_result.example.json` 仅用于理解严格 schema。
该文件不能复制成运行结果，也不能在 Mac 上填写；正式结果只允许由 Linux
runner 生成。

工具要求以下文件使用固定文件名存在，避免只写一份主观汇总便宣称 replay 完成：

```text
stdout.log
stderr.log
runtime_summary.json
template_evidence.json
anomaly_evidence.json
yolo_evidence.json
fusion_result.json
audit.json
phase0_case_result.json
replay_root.json
```

`runtime_summary.json`、三个分支 evidence、`fusion_result.json` 和
`audit.json` 必须使用下列严格 envelope：

```json
{
  "schema_version": 1,
  "case_id": "right-normal-001",
  "run_id": "replay-01",
  "execution_id": "right-normal-001-replay-01-20260714T120000",
  "captured_at": "2026-07-14T12:00:00+08:00",
  "payload": {"REPLACE_WITH_FILE_SPECIFIC_EVIDENCE": true}
}
```

`case_id/run_id/execution_id/captured_at` 必须与 `phase0_case_result.json`
完全一致；`payload` 必须是非空对象，不得包含占位文本。
`observed_legacy.final_status` 只接受冻结状态枚举，
`evidence_groups` 必须恰好包含 `template/anomaly/yolo`；前置 gate
短路时应显式记录未运行原因，不得删掉分支。
`phase0_case_result.json.source_binding` 还必须精确绑定 replay
时的 Git commit/tree、`uv.lock`、topology、ROI、全部冻结资产 hash、
稳定环境 hash、capture identity 和逐 view golden image hash。inventory 会与当前真实
输入逐字段比较，两次 replay 也必须使用完全相同的执行命令。

旧链路 replay 必须成功退出，因此 `exit_code` 必须为 `0`。
`observed_legacy.final_status` 不能是占位文本，`evidence_groups` 必须是非空
结构；`parity` 只含 `mode`，并且必须与 golden selection 完全一致。
`phase0_case_result.json` 是机器证据，不承载人工审核状态，禁止为签署容差、
补充理由或修改结论而编辑它。

runner 会为每次 replay 记录全局唯一的 `execution_id` 和带时区的 RFC3339
`executed_at`；同一个 case 的两次 replay 不得复用执行时间。不得人工填写
或修正这些字段。工具会解析上述 JSON evidence，并拒绝空对象和占位文本。

当前 runner 是 Stage 32 专用 adapter：它从 `runtime_summary.json`、
`template_match.csv`、`patchcore.csv`、`yolo.csv`、
`fused_predictions.csv` 和 audit JSON 解析稳定类别证据，不从 overlay
或文件名猜测结果。模板短路时会明确记录 anomaly/YOLO
`NOT_RUN`，不伪造 CLEAR。Stage 32 非零退出时会保留 stdout/stderr 和
`execution_failure.json`，但该目录不是合格 baseline，必须解决后用新的
run/output 重跑。runner 还有固定 1800 秒超时；超时同样只发布诊断目录，不生成可审核 replay root。

`observed_legacy.evidence_groups` 只放用于语义稳定性比较的分支状态/等级，不放连续 score；连续 score 单独保存并按报告中预先声明的容差审核。

runner 成功发布前会写入 `replay_root.json`。其
`replay_root_sha256` 按目录内除 `replay_root.json` 自身之外的全部普通文件，
使用以下确定性算法计算：

```text
sha256(relative_path NUL file_sha256 NUL size LF)
```

runner 发布后，整个 replay 目录视为不可变。增加、删除或修改任意证据文件，
包括编辑 `phase0_case_result.json`，都会使 root 校验失败。需要修正时必须使用
新的 `run_id` 和新目录重新执行，不能原地修补。

两次或更多 replay 全部完成后，人工比较同一 case 的连续分数和其他需要审核
的证据。审核结论另存为 replay 目录之外的一份 receipt，例如：

```text
/DATA/ljl/zs32_phase0_reviews/<case_id>.baseline-review.json
```

从 `configs/zs32/phase0/baseline_review_receipt.example.json` 复制结构并填写。
每个 case 恰好一份 receipt；其中 `replays` 必须一次性列出 freeze spec 为该
case 登记的全部 replay，且每项精确绑定 `run_id`、对应
`replay_root.json.replay_root_sha256` 和语义签名。语义签名是
`phase0_case_result.json.observed_legacy` 的 canonical JSON SHA256。可在 Linux
上对每个 replay 计算并核对：

```bash
export REPLAY_DIR=/DATA/ljl/zs32_phase0_runs/replay-01/right-normal-001
.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["replay_root_sha256"])' "$REPLAY_DIR/replay_root.json"
.venv/bin/python - "$REPLAY_DIR/phase0_case_result.json" <<'PY'
import hashlib
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    observed = json.load(stream)["observed_legacy"]
encoded = json.dumps(
    observed,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
print(hashlib.sha256(encoded).hexdigest())
PY
```

若需要比较连续分数，`score_comparison.decision` 填
`WITHIN_TOLERANCE`，并在 `basis` 中写明预先约定的指标、容差和结论；确实
没有连续分数时填 `NOT_APPLICABLE` 并说明原因。`reviewer`、带时区的
`reviewed_at` 和 `decision: "APPROVED"` 都必须由实际审核人填写。receipt
生成后不得修改；如 replay 集合或审核结论改变，应生成新的 receipt 路径并
重新审核。

已知必须标记为 `must_change` 的差异：

- PatchCore ROI 改为权威 YOLO ROI；
- 旧模板 REVIEW 改为模板二值 NG；
- 资产/运行时故障改为 `SYSTEM_ERROR`；
- geometry 等非模板传统分支删除；
- 缺失证据不得产生可发布 OK。

freeze spec 中为每个 case 登记至少两个不同 `run_id`，并在
`baseline_reviews` 中登记该 case 唯一的外部 receipt。工具强制
`minimum_replays_per_case >= 2`，要求各次
`observed_legacy.final_status + evidence_groups` 完全一致，并验证 receipt
绑定的全部 run/root/signature 与真实不可变 replay 精确相等。连续分数不能
只比较 overlay PNG 字节，人工容差结论必须写入 receipt。

## 8. 先 inventory，再 freeze

在 freeze spec 仍为 `draft` 时可以做 inventory，提前发现问题：

```bash
.venv/bin/python tools/zs32_phase0.py inventory \
  --spec configs/zs32/phase0/phase0_freeze.local.json \
  --output /DATA/ljl/zs32_phase0_work/inventory-01.json
```

inventory 不是生成 review receipt 的工具。运行它之前，全部 replay、每个
case 的外部 receipt，以及 spec 中的 `baseline_outputs`/`baseline_reviews`
映射都必须已经填写；`draft` 仅表示尚未完成最终 freeze approval。

`discover-legacy`、`replay-bindings`、`inventory` 和 `fault-injection` 的 JSON
输出均是不可覆盖的。二次执行必须使用递增文件名（例如
`golden_candidates-02.json`、`replay-bindings-02.json`、`inventory-02.json`），
禁止先覆盖旧证据来掩盖差异。

`required_asset_roles`、`required_golden_scenarios` 和 `known_blockers`
不是 local spec 可以缩减的参数；工具中已固化与示例一致的 normative
集合，inventory、freeze 和 bundle verify 都会重新检查。删空列表或
删除 blocker 不会降低完成门槛。

审核 inventory 后，将 spec 的：

```json
"status": "ready"
```

最终 inventory 必须在所有 blocker 证据、资产 hash、golden selection 和
replay 目录都已完成后重新生成。记录其路径、文件 SHA256 和内部
`review_subject_sha256`：

```bash
export REVIEWED_INVENTORY=/DATA/ljl/zs32_phase0_work/inventory-final.json
sha256sum "$REVIEWED_INVENTORY"
.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["review_subject_sha256"])' "$REVIEWED_INVENTORY"
```

然后只允许修改 spec 的 `status` 和 `approval`：将 `status` 改为
`ready`、`approved` 改为 `true`，填写审核人、带时区 RFC3339
时间、非占位 `basis`，并精确填写 `reviewed_inventory_path`、
`reviewed_inventory_sha256` 和 `review_subject_sha256`。freeze 会重新
打开该 inventory，校验文件 hash、审批时间不早于 inventory，并将
当前 Git/environment/assets/golden/replay 的稳定审核 payload 与已审文件
逐字段比较。`status: ready` 不能替代人工签署；任一 blocker
未解决或审核后输入改变时都会拒绝冻结。

然后确认 Git 仍然 clean，再冻结：

```bash
git status --porcelain=v1 --untracked-files=all

.venv/bin/python tools/zs32_phase0.py freeze \
  --spec configs/zs32/phase0/phase0_freeze.local.json
```

freeze 通过 staging 目录原子发布；目标 `freeze_id` 已存在时拒绝覆盖，中途失败不会留下正式 bundle。

## 9. 严格校验 bundle

```bash
.venv/bin/python tools/zs32_phase0.py verify \
  --bundle /DATA/ljl/anomaly_xingtao_phase0/<freeze_id> \
  --expected-root <TRUSTED_ROOT_FROM_FREEZE_OR_GIT_POINTER>
```

`verify` 会拒绝：

- 缺失文件；
- 未登记的额外文件；
- 任意文件 hash 改变；
- checksum 行被篡改；
- bundle root 与 checksum 集不一致。

外部信任锚是必填项：首次冻结使用 `--expected-root`，后续使用 `--pointer`。这可以防止攻击者同时重写业务文件、`checksums.sha256` 和 `bundle_root.json`。第一次使用 freeze 命令原子发布后直接返回的 root；把该 root 写入仓库 pointer 并 commit/push 后，后续验证只能使用 Git pointer 中的 root。

把 root digest 写入仓库中的 pointer 文件并单独 commit/push：

```text
baselines/zs32/<freeze_id>.pointer.json
```

pointer 必须与工具 schema 完全一致，顶层恰好包含 `schema_version:1`、`freeze_id`、`bundle_root_sha256`、冻结时的 `git_commit`、`scope` 和非占位的不可变 `storage_reference`；`scope` 恰好包含 `product:"ZS32"`、`hands:["right"]`、`topology_id:"zs32-3cam-double-side-v1"`、`left_roi_status:"pending"`。格式以 `baselines/zs32/README.md` 为准。缺少、多余或与 bundle manifest 不一致的字段都会被拒绝。如部署要求更强防篡改，再对 pointer 或 attestation 使用 GPG/minisign/cosign 签名。
其 `storage_reference` 必须是不可变、内容寻址的引用，引用文本中必须包含完整 `bundle_root_sha256`，不得是 `current`
symlink、可覆盖路径、含 token 的 URL 或其他凭证。

pointer 提交后改用：

```bash
# Mac：人工审核 pointer 后 commit/push，记录新的 commit/tree
git status --porcelain=v1 --untracked-files=all
git rev-parse HEAD
git rev-parse HEAD^{tree}

# Linux：必须再次从 GitHub 拉取这个 pointer commit
git fetch --prune origin
git checkout main
test -z "$(git status --porcelain=v1 --untracked-files=all)"
git pull --ff-only origin main
test "$(git rev-parse HEAD)" = "<POINTER_COMMIT_FROM_MAC>"
test "$(git rev-parse HEAD^{tree})" = "<POINTER_TREE_FROM_MAC>"
test -z "$(git status --porcelain=v1 --untracked-files=all)"

.venv/bin/python tools/zs32_phase0.py verify \
  --bundle /DATA/ljl/anomaly_xingtao_phase0/<freeze_id> \
  --pointer baselines/zs32/<freeze_id>.pointer.json
```

不得在 Linux worktree 内现场生成并提交 pointer 后直接验证；第二次
handoff 与源码 handoff 一样，必须经过 Mac 审核、GitHub 传递、Linux
clean pull 和精确 commit/tree 校验。

## 10. 只在 Linux 执行测试

```bash
.venv/bin/pytest -q tests/unit/tools/test_zs32_phase0.py
```

随后运行内置故障注入。它只操作临时副本，不修改正式 bundle；前三项验证 checksum 集，后两项会重新计算副本 root，以继续验证资产 manifest 和右手 scope 的语义守卫：

```bash
mkdir -p /DATA/ljl/zs32_phase0_fault_work

.venv/bin/python tools/zs32_phase0.py fault-injection \
  --bundle /DATA/ljl/anomaly_xingtao_phase0/<freeze_id> \
  --expected-root <TRUSTED_ROOT_FROM_FREEZE> \
  --work-root /DATA/ljl/zs32_phase0_fault_work \
  --output /DATA/ljl/zs32_phase0_work/<freeze_id>.fault-injection.json
```

`--work-root` 必须是 Linux 大容量数据盘上已经创建的真实目录，不使用容量有限的 `/tmp`。工具每次只保留一个 bundle 副本并在该项检测结束后立即删除。输出是不可覆盖的 JSON 报告，必须出现五项 `detected: true`：修改一个字节、删除文件、新增文件、替换 deployment weight、将冻结 scope 改成 left。把报告 SHA256 和保存位置写入 Phase 0 验收记录；任何一项未 fail closed，工具都会非零退出且不得进入 Phase 1。

## 11. Phase 0 完成门槛

只有下面条件全部满足，才能从 Phase 0 进入 Phase 1：

1. Linux + NVIDIA 守卫和 Linux 单测通过；
2. clean commit、环境和相机 SDK 已记录；
3. 右手 ROI、相机采集参数、六个 anomaly 权重、YOLO `best.pt`、模板、阈值全部 hash 固定；
4. golden cases 六视角完整并经人工签署；
5. 每个 case 至少两次不可变 replay，语义结果稳定，且唯一的人工 review receipt 精确绑定全部 replay；
6. bundle strict verify 通过，root digest 已由 Git pointer 锚定；
7. 五项 fault injection 全部 fail closed，报告已保存并固定 hash；
8. 左手仍为 `ROI_PENDING`，不存在左手 baseline 或 release。

在这些结果从 Linux 回传前，当前状态只能写成“Phase 0 工具与配置已准备”，不能写成“Phase 0 已完成”。

回传证据至少包含：已核对的 commit/tree/lock SHA256、clean status、Linux/GPU/
CUDA/Python 环境、inventory/freeze/verify 的完整命令与 exit code、trusted root、
pointer commit、测试与故障注入结果、人工签署人与带时区时间。回传文件应
附 SHA256 并先移除 token、凭证、内网地址和不必要的绝对路径；它们不提交到
GitHub。
