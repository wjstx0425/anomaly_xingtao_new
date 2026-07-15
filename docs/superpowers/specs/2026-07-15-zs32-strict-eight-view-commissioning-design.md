# ZS32 严格八视角 Commissioning 设计

**日期：** 2026-07-15  
**状态：** 已由用户逐节确认，等待书面规范复核  
**范围：** ZS32、right hand、四相机两轮采集、八视角全模型 commissioning

## 1. 背景与决策

本设计把 ZS32 新链路的基准从“六个 modeled view 加两个 unsupported secondary”改为严格八视角。稳定顺序为：

```text
front, front_left, front_right, front_secondary,
back, back_left, back_right, back_secondary
```

八个视角都必须进入 Template、PatchCore、YOLO 和 Fusion。`front_secondary`、`back_secondary` 不再是特殊展示位，也不允许发布 `model_supported=false`、`unsupported` 或空 branches。

本设计覆盖 `2026-07-14-zs32-eight-view-inspection-dashboard-design.md` 中“仅六个主视角建模、secondary 暂未接入模型”的部分。旧六视角配置和产物继续保留用于历史回放，但不再作为新 Stage32、Stage35 或看板链路的默认值。

当前验收目标是 **八视角 commissioning 链路完整跑通**，不是生产放行。尤其 `front_secondary` 的现有 PatchCore sample F1 约为 `0.286`，模型虽可接线，但性能不足以支持生产就绪声明。

## 2. 已确认资产与缺口

### 2.1 已存在的真实资产

- 八视角 ROI：`/home/yunjing/anomalib/dataset/zs32_eight_view_roi_config.json`。
- 八个独立 PatchCore checkpoint 及阈值汇总：`results/zs32_patchcore_eight_view_seed42/eight_view_summary.csv`。
- 八视角共享 YOLO checkpoint：`results/yolo/zs32_eight_view_roi_n640_seed42/weights/best.pt`。
- 四相机 topology：`configs/zs32/topology/zs32_4cam_double_side_v1.json`。
- 四相机 legacy publisher 已能表达八个 image rows 加一个 complete sample row。

### 2.2 必须补齐的发布资产

- 使用同一八视角 ROI 代际重新生成的八个 Template groups。
- 将八个 PatchCore checkpoint、各自 deploy threshold、共享 YOLO 和 ROI 绑定在一起的 runtime bundle。
- 与 bundle hash 绑定的 24-group commissioning threshold artifact。
- 严格 `8 × 3` Stage18 commissioning profile 与 expected versions。
- Stage35 四相机采集、八图 Stage32 和八视角看板之间一致的 identity 与 manifest 合同。

旧六视角 Template 属于另一 ROI 代际，不得被包装成八视角新资产。

## 3. 总体架构

```text
四相机两轮采集
  -> 同一 identity 的八张原图
  -> 八视角 Template gate
  -> 八个 PatchCore + 一次八图 YOLO
  -> 24-group commissioning Fusion
  -> 八视角 runtime manifest
  -> OpenCV 看板
```

核心合同：

- `MODELED_VIEWS == VIEW_ORDER`。
- 八张图必须来自同一个 complete sample，并共享 `part_id`、`capture_session`、`group_id` 和 `hand=right`。
- 每个 view 必须有 `template`、`patchcore`、`yolo`、`fusion` 四个 dashboard branch 记录。
- Stage18 commissioning 输入严格为 24 个证据组：每个 view 的 `template_match`、`anomaly_<view>` 和 `yolo`。
- progress/control sidecar 和 OpenCV 主线程/Stage35 子进程边界保持不变。
- 所有新产物明确记录 `commissioning_only=true`、`production_release_allowed=false`。

## 4. 可替换 Runtime Bundle

模型路径不得硬编码到 Stage32、Stage35 或看板。新链路以一个版本化 runtime bundle 作为唯一资产入口：

```text
runtime bundle
├── schema_version
├── bundle_id
├── asset_set_sha256
├── product / hand / view_order
├── roi
│   ├── path
│   ├── version
│   └── sha256
├── template
│   ├── model_dir
│   ├── version
│   └── sha256
├── patchcore
│   └── <view>
│       ├── checkpoint
│       ├── checkpoint_sha256
│       └── deploy_threshold
├── yolo
│   ├── checkpoint
│   ├── checkpoint_sha256
│   └── inference settings
└── fusion
    ├── profile
    ├── threshold_artifact
    └── source hashes
```

Stage32 和 Stage35 只接收 `--runtime-config <path>`。更换任一 Template、PatchCore 或 YOLO 权重时，通过发布命令生成新 bundle，而不是修改 Python 代码或覆盖旧文件。

发布器必须：

1. 校验 product、hand、固定八视角顺序和资产完整性；
2. 计算所有文件 hash；
3. 从八视角 PatchCore summary 提取有限且有效的 deploy thresholds；
4. 生成 24 个 expected-version 记录和对应 threshold artifact；
5. 使用原子写入发布不可变 bundle；
6. 拒绝缺 view、重复 checkpoint、空阈值、hash 漂移和 ROI 代际混用。

`asset_set_sha256` 只覆盖 ROI、Template、八个 PatchCore、YOLO 及其推理设置。24-group threshold artifact 绑定这个资产集合 hash；bundle 最后再记录 threshold artifact 的路径和 hash。这样避免“bundle hash 包含 threshold artifact、threshold artifact 又反向绑定 bundle hash”的循环依赖。

旧 bundle 保持不变，从而保证历史 runtime manifest 可以追溯其实际模型版本。YOLO 当前为八视角共享模型；未来若改为分视角 YOLO，只扩展 bundle schema，不改变 dashboard view/branch 合同。

## 5. 八视角执行语义

### 5.1 Template gate

Template 使用八视角 ROI 正常样本重新训练并发布八个 group。Stage32 必须先获得八个 Template 结果，再计算整体 gate，不允许在第四个或更早视角失败后留下结构不完整的 manifest。

若任一 Template 触发 NG：

- 整件短路；
- 八个 PatchCore 和 YOLO branch 均记录 `SKIPPED`；
- 不生成伪造 score、mask 或 detections；
- runtime manifest 仍包含八个 view 和四个核心 branch；
- `running_patchcore_yolo` 与 `running_fusion` progress 状态不得伪造。

### 5.2 PatchCore 与 YOLO

Template 允许下游后：

- PatchCore 按固定顺序加载并执行八个独立 checkpoint；
- raw anomaly map、真实 `pred_mask` 或 display-only diagnostic mask 继续按已冻结合同保存；
- YOLO 对八个 ROI crop 运行同一个共享 checkpoint，并返回八条有序结果；
- 所有 score 必须有限，路径、ROI、mask shape 和身份冲突均 fail-closed。

### 5.3 Fusion

Stage18 使用版本化的八视角 24-group commissioning profile。任何缺组、额外组、重复组、expected version 不一致或 threshold artifact hash 漂移都属于执行错误。

Fusion 不把 commissioning 结果提升为生产结论。若保留分面辅助函数，每面视角必须从三项扩展为四项，不能丢弃 secondary。

## 6. Stage35 与进度控制

Stage35 必须使用 topology-driven 四相机采集，不再调用三 serial collector。流程为：

```text
校验 runtime bundle
-> waiting_front
-> 确认正面并拍摄四张
-> waiting_back
-> 确认背面并拍摄四张
-> running_template
-> running_patchcore_yolo
-> running_fusion
-> complete 或 failed
```

要求：

- 打开相机前完成 bundle、topology、Template groups、八个 PatchCore、YOLO、ROI、阈值和 hash 预检。
- 每轮 waiting 使用新的 `confirmation_id`；错误、过期或重复 control 不触发拍摄。
- complete sample 恰好包含八个 image rows 和一个 sample row。
- Stage32 command 传入八个 image arguments，不允许漏 secondary。
- Stage35 只管理自己启动的 process group；退出看板时依次 SIGTERM、超时后 SIGKILL，不能遗留采集器或推理进程。
- progress 写入失败本身是执行错误；看板不解析终端文本。

## 7. Runtime Manifest 与看板

最终 manifest 的八个 view 均满足：

- `model_supported=true`；
- source path、SHA256、shape 和 capture identity 完整；
- exact core branches 为 `template`、`patchcore`、`yolo`、`fusion`；
- 记录 bundle ID、实际 model versions、threshold source 和 ROI version；
- 短路或失败使用 `SKIPPED`/`ERROR`，不使用 `UNSUPPORTED`。

看板继续保留 Original、Template、PatchCore、YOLO、Fusion 五层和单一上下文检测按钮。两个 secondary 与其他六个视角完全相同，可显示真实 status、score、mask、bbox 和 fusion notice。

## 8. 错误边界

以下情况必须在模型执行前或对应 branch 边界 fail-closed，不能显示为普通 NG：

- 缺失、重复或混合 identity 的八张图；
- topology serial/view/round 不一致；
- bundle 缺 view、路径越界、文件缺失、hash 漂移；
- Template group 或 ROI 代际不一致；
- 非有限阈值或 score；
- mask、raw map、bbox 几何错误；
- 24-group 融合缺组、额外组或 expected version 不一致；
- progress/control JSON 损坏或写入失败；
- Stage35 子进程非零退出。

错误 manifest 仍应尽可能保留已验证的 source identity 和八视角结构，但不得伪造未执行分支的证据。

## 9. 测试与验收

### 9.1 自动化测试

- `MODELED_VIEWS == VIEW_ORDER`，八个 view 均拒绝 `model_supported=false`。
- bundle 发布、hash、原子写、路径边界、八 checkpoint 唯一性和 24 expected versions。
- 八视角 Template 训练/加载/推理与缺 group fail-closed。
- Stage32 八图参数、八次 PatchCore、八图 YOLO、Template 短路和结构完整 manifest。
- Stage18 正常 24-group、缺组、额外组、重复组、版本/阈值漂移。
- Stage35 四相机命令、八图 legacy manifest、identity、round、serial 和 path 校验。
- progress/control 状态序列、一次性 confirmation 和无 TTY 行为。
- dashboard secondary 的五层显示、键鼠交互、进程组清理和 headless CLI。

### 9.2 本地真实资产 smoke

使用同 identity 的已有八图，绑定新八视角 Template、八个现有 PatchCore 和现有八视角 YOLO，运行完整 Stage32/Stage18。必须得到八 view、24 输入组及可解析看板 manifest。

### 9.3 单件真机 smoke

验收条件全部满足才算通过：

1. 一个 right-hand part；
2. 正背面 confirmation 各正确消费一次；
3. legacy manifest 恰好八个 image rows 加一个 complete sample row；
4. 八张 PNG 共享唯一 identity；
5. Stage32 command 包含八个 image arguments；
6. Template、PatchCore、YOLO 均有八视角记录；
7. Stage18 接收严格 24 groups；
8. runtime manifest 有八个 modeled views 和四个核心 branches；
9. 看板显示最终真实状态和八张图；
10. 退出后无 Stage35、采集器或 Stage32 遗留进程。

验收报告只能写“八视角 commissioning 链路跑通”。生产放行必须等待独立的模型性能、标定、重复性和现场稳定性验收，特别是 `front_secondary` 模型改进。

## 10. 实施边界

- 保留旧六视角配置、模型和历史结果，不改写其语义。
- 不把训练 profile、全零 hash 或旧 ROI Template 当作部署 bundle。
- 不在 OpenCV 主线程运行 GPU 推理。
- 不新增历史浏览、目录打开或 Web UI。
- 不在本设计中宣称 production-ready。
