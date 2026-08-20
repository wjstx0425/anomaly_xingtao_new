# ZS32 快速 Demo 视觉检测平台重构：新对话任务说明

日期：2026-07-27

> 将本文完整交给新的 Codex 对话。工作目录固定为
> `/home/yunjing/anomaly_xingtao_new`。

## 1. 用户目标

当前仍处于 Demo 验证阶段。用户要的是能够快速更换模型、手动调整阈值并立即运行的八视角视觉检测
平台，不需要工业发布级的不可变资产、整条 SHA256 绑定、runtime bundle、版本回滚或反复多轮验证。

目标是把在线检测重构为一条简洁、单一的 Demo 路径：

```text
Dashboard
  -> 四相机正面/背面采集
  -> 八视角完整性检查
  -> 八视角 Template 全局门禁
  -> 全部 PASS 时运行 PatchCore + YOLO
  -> 读取一个可编辑 Demo JSON 中的阈值
  -> 简单融合
  -> 八视角可视化
```

用户已经明确：

- 不保留现有严格哈希链作为第二运行模式。
- 不保留 runtime bundle 作为在线检测依赖。
- 不要在旧严格代码中堆叠 `skip_hash`、`unsafe` 等条件分支。
- 最终代码应当简洁、干净，只有一条在线运行路径。
- 当前这份文档只定义后续任务；新对话开始后先检查真实代码，再实施。

## 2. 当前真实背景

当前 Dashboard 入口：

```text
pipeline/36_zs32_inspection_dashboard.py
```

当前在线链大致为：

```text
Dashboard
  -> src/zs32_inspection/dashboard/live.py
  -> pipeline/35_run_zs32_live_commissioning.py
  -> capture_data/zs32_live_commissioning.py
  -> pipeline/zs32_inference_worker.py
  -> pipeline/32_run_zs32_multimodel_inference.py
  -> Template / PatchCore / YOLO
  -> pipeline/18_fuse_inspection_results.py
```

现有严格链会重复验证：

- `runtime_bundle.json`
- `assets_manifest.json`
- runtime assets 和 fusion profile
- `model.sha256`
- Template 图片 SHA
- PatchCore checkpoint SHA
- YOLO weights SHA
- threshold artifact canonical SHA
- source artifact/profile/version 精确绑定
- Stage18 的 24-group、排序、版本和证据哈希

因此只改一个 Template 阈值也会导致 model、assets、thresholds、bundle 整条链失效，需要重新执行
Stage34/Stage37 和多轮 fresh-load。这个机制不适合当前 Demo 快速迭代目标。

当前模型资产仍可作为 Demo 初始配置来源：

- Template：
  `results/zs32_template_gate_right_0727_eight_view_v13`
- PatchCore：
  `results/zs32_patchcore_eight_view_all_plus_0723_seed42_v11`
- YOLO：
  `results/yolo/zs32_all_plus_0723_normal_n1280_seed42_v11/weights/best.pt`
- ROI：
  `dataset/zs32_all_plus_0723_retraining_release_v2/roi_config.json`
- Topology：
  `configs/zs32/topology/zs32_4cam_double_side_v1.json`

规范八视角顺序必须保持：

```text
front
front_left
front_right
front_secondary
back
back_left
back_right
back_secondary
```

两个 secondary 视角必须真实运行 Template；全局 Template 门禁全部 PASS 后，它们的 PatchCore 和 YOLO
也必须真实运行。若任一 Template NG，所有 PatchCore、YOLO 必须明确标记为 `SKIPPED`；Fusion
逐视角显示对应 Template 的 `NG_TEMPLATE` 或 `PASS`。

## 3. 非目标

首版不要实现：

- strict/demo 双模式
- runtime bundle 自动生成
- SHA256 发布或签名
- v10/v11/v12/v13 在线回滚
- Stage34/Stage37 发布流程
- 网页阈值编辑器
- 配置历史、版本管理或回滚
- 运行过程中热切换 checkpoint/YOLO weights
- 新的模型训练逻辑
- 工业生产放行或合规审计

只修改阈值应当热更新；修改模型路径后允许重启 Dashboard。

## 4. 目标配置

新增唯一需要操作员编辑的文件：

```text
configs/zs32/zs32_demo.json
```

建议结构：

```json
{
  "schema_version": 1,
  "mode": "demo",
  "topology": "configs/zs32/topology/zs32_4cam_double_side_v1.json",
  "roi_config": "dataset/zs32_all_plus_0723_retraining_release_v2/roi_config.json",
  "models": {
    "template_dir": "results/zs32_template_gate_right_0727_eight_view_v13",
    "patchcore": {
      "front": "results/zs32_patchcore_eight_view_all_plus_0723_seed42_v11/right_front/runs/right_front/patchcore_wrn_l23_s256_r005_k9_fp32_bs16_fpr005_seed42/Patchcore/zs32_right_front/right_front/v0/weights/lightning/model.ckpt",
      "front_left": "results/zs32_patchcore_eight_view_all_plus_0723_seed42_v11/right_front_left/runs/right_front_left/patchcore_wrn_l23_s256_r005_k9_fp32_bs16_fpr005_seed42/Patchcore/zs32_right_front_left/right_front_left/v0/weights/lightning/model.ckpt",
      "front_right": "results/zs32_patchcore_eight_view_all_plus_0723_seed42_v11/right_front_right/runs/right_front_right/patchcore_wrn_l23_s256_r005_k9_fp32_bs16_fpr005_seed42/Patchcore/zs32_right_front_right/right_front_right/v0/weights/lightning/model.ckpt",
      "front_secondary": "results/zs32_patchcore_eight_view_all_plus_0723_seed42_v11/right_front_secondary/runs/right_front_secondary/patchcore_wrn_l23_s256_r005_k9_fp32_bs16_fpr005_seed42/Patchcore/zs32_right_front_secondary/right_front_secondary/v0/weights/lightning/model.ckpt",
      "back": "results/zs32_patchcore_eight_view_all_plus_0723_seed42_v11/right_back/runs/right_back/patchcore_wrn_l23_s256_r005_k9_fp32_bs16_fpr005_seed42/Patchcore/zs32_right_back/right_back/v0/weights/lightning/model.ckpt",
      "back_left": "results/zs32_patchcore_eight_view_all_plus_0723_seed42_v11/right_back_left/runs/right_back_left/patchcore_wrn_l23_s256_r005_k9_fp32_bs16_fpr005_seed42/Patchcore/zs32_right_back_left/right_back_left/v0/weights/lightning/model.ckpt",
      "back_right": "results/zs32_patchcore_eight_view_all_plus_0723_seed42_v11/right_back_right/runs/right_back_right/patchcore_wrn_l23_s256_r005_k9_fp32_bs16_fpr005_seed42/Patchcore/zs32_right_back_right/right_back_right/v0/weights/lightning/model.ckpt",
      "back_secondary": "results/zs32_patchcore_eight_view_all_plus_0723_seed42_v11/right_back_secondary/runs/right_back_secondary/patchcore_wrn_l23_s256_r005_k9_fp32_bs16_fpr005_seed42/Patchcore/zs32_right_back_secondary/right_back_secondary/v0/weights/lightning/model.ckpt"
    },
    "yolo": {
      "weights": "results/yolo/zs32_all_plus_0723_normal_n1280_seed42_v11/weights/best.pt",
      "imgsz": 1280,
      "candidate_conf": 0.001
    }
  },
  "thresholds": {
    "template": {
      "front": 0.024885842800140383,
      "front_left": 0.029833445549011232,
      "front_right": 0.02277636528015137,
      "front_secondary": 0.21635736227035524,
      "back": 0.010035157203674318,
      "back_left": 0.015659272670745853,
      "back_right": 0.019793987274169925,
      "back_secondary": 0.026354730129241947
    },
    "patchcore": {
      "front": 0.4711672067642212,
      "front_left": 0.5397635102272034,
      "front_right": 0.814327597618103,
      "front_secondary": 0.5006473064422607,
      "back": 0.4759141802787781,
      "back_left": 0.6454828977584839,
      "back_right": 0.5977694988250732,
      "back_secondary": 0.49985870718955994
    },
    "yolo": {
      "front": 0.07,
      "front_left": 0.07,
      "front_right": 0.07,
      "front_secondary": 0.07,
      "back": 0.07,
      "back_left": 0.07,
      "back_right": 0.07,
      "back_secondary": 0.07
    }
  }
}
```

以上八个 PatchCore checkpoint 路径已从当前 v13 runtime assets 中解析。新对话实施前仍需确认这些文件
存在，但不要重新通过 bundle 加载它们。

`candidate_conf` 是 YOLO 产生候选框的低门槛；`thresholds.yolo` 是最终业务判定门槛。两者不能混用。
YOLO 独立证据页可以保留低于业务阈值的候选框用于诊断，但 Fusion 页面只能显示单框
`confidence >= thresholds.yolo[view]` 的红框；如果该视角的 YOLO 分支总分低于业务阈值，则 Fusion
不得显示任何 YOLO 红框。

## 5. 目标模块边界

### 5.1 Demo 配置

新增：

```text
capture_data/zs32_demo_config.py
```

职责：

- 读取一个 JSON。
- 解析为明确的 dataclass。
- 解析相对仓库根目录的路径。
- 校验恰好八个规范视角。
- 校验两个 secondary 存在。
- 校验路径存在、阈值有限且范围有效。
- 不计算或比较任何 SHA。

阈值约束：

- Template 和 PatchCore：`0 <= threshold <= 2`
- YOLO 和 `candidate_conf`：`0 <= threshold <= 1`
- 禁止 NaN 和 Infinity。

### 5.2 Demo 推理

新增：

```text
capture_data/zs32_demo_runtime.py
pipeline/zs32_demo_inference.py
```

职责：

- 直接从 DemoConfig 构建 Template、八个 PatchCore 和 YOLO。
- 复用当前真实推理实现，不复制模型算法。
- 不调用 `load_runtime_bundle()`。
- 不读取 threshold artifact 或 fusion profile。
- 不执行 SHA 或版本身份验证。
- 输出 Dashboard 当前需要的原图、overlay、score、threshold 和 status。

Template 分数计算与阈值判定必须解耦。Template 图片和模型可在 worker 启动时加载；每个新零件开始前
重新读取 Demo JSON 的阈值，修改阈值不应重载 GPU 模型。

### 5.3 简单融合

先运行所有八视角 Template。任一 Template NG 时整件立即输出 `NG_TEMPLATE`，不运行 PatchCore 和
YOLO，并把所有未执行的 PatchCore、YOLO 分支明确记录为 `SKIPPED`；Fusion 逐视角发布对应
Template 的 `NG_TEMPLATE` 或 `PASS`。只有八个 Template 全部
PASS 时，才运行八个 PatchCore 和八个 YOLO。

建议融合顺序：

```text
任一输入缺失、模型异常或分支缺失 -> ERROR
否则任一 Template 超阈值          -> NG_TEMPLATE
否则任一 PatchCore 超阈值         -> NG_ANOMALY
否则任一 YOLO 超阈值              -> NG_YOLO
否则                               -> OK
```

融合代码应放在 Demo runtime 内的一个小函数中，不调用 Stage18。

### 5.4 Dashboard 与采集

修改：

```text
src/zs32_inspection/cli/dashboard.py
src/zs32_inspection/dashboard/live.py
src/zs32_inspection/dashboard/render.py
pipeline/35_run_zs32_live_commissioning.py
capture_data/zs32_live_commissioning.py
pipeline/zs32_inference_worker.py
```

要求：

- 删除在线入口的 `--runtime-config`。
- 改为唯一的 `--demo-config`。
- Dashboard、采集进程、worker 和推理进程传递同一个 DemoConfig 路径。
- 保留正面/背面两次人工确认。
- 保留常驻模型 worker，避免每件重新加载模型。
- 每件检测前重新读取阈值。
- Dashboard 标题醒目标记 `DEMO / 非生产`。
- JSON错误、相机错误和模型加载错误必须显示真实错误。
- 子进程失败时不能再错误显示成“开始检测”。

## 6. 最低限度的实用检查

移除工业发布保护，但保留这些低成本检查：

1. 四相机 serial 和八视角 topology 映射完整。
2. 正面与背面各四张图片齐全。
3. 八视角路径存在且图片可解码。
4. capture manifest 的 part/session/group 身份一致。
5. ROI 坐标合法且覆盖八视角。
6. checkpoint、Template 和 YOLO 权重存在且能够加载。
7. 阈值是合法有限数。
8. 缺视角、缺模型证据或推理异常绝不能判为 OK。
9. 输出目录使用唯一时间戳，不覆盖历史结果。
10. Dashboard 显示实际错误和 `DEMO / 非生产`。

这些是防止普通配置错误产生假结果，不是工业发布保护。

## 7. 明确移除的运行依赖

完成新路径并验收后，在线检测不得再 import 或调用：

```text
capture_data/zs32_runtime_bundle.py
capture_data/zs32_18_group_commissioning.py
pipeline/34_publish_zs32_18_group_commissioning.py
pipeline/37_publish_zs32_runtime_bundle.py
pipeline/18_fuse_inspection_results.py
```

还要移除在线路径中的：

- runtime bundle 参数和默认路径
- assets manifest
- model/checkpoint/weights SHA 校验
- Template sidecar 和 Template 图片 SHA 校验
- threshold artifact
- fusion profile
- profile/version 精确绑定
- commissioning publication identity
- v10/v11/v12/v13 rollback 运行测试
- Stage34/Stage37 发布测试

在真实 Demo 路径验收前不要删除历史 `results/` 产物。验收后先用只读搜索确认无引用，再向用户列出拟删除
目录，由用户单独确认物理删除；不要在重构时顺手删除大量历史结果。

训练、数据准备和独立模型评估脚本继续保留，只移除严格发布与在线消费链。

## 8. 分阶段实施计划

### 阶段 1：建立 DemoConfig

1. 检查当前 v13 runtime assets，解析真实八视角模型和 ROI 路径。
2. 先写 DemoConfig 失败测试。
3. 新增 JSON loader 和 dataclass。
4. 生成完整、无占位符的 `configs/zs32/zs32_demo.json`。
5. 验证缺视角、缺 secondary、非法阈值和不存在路径都会给出清晰错误。

验收：一份真实配置可独立加载，不读取任何 bundle 或 SHA 文件。

### 阶段 2：建立无 bundle 的离线推理

1. 新增 Demo runtime。
2. 接入 Template 连续 risk score。
3. 接入八个 PatchCore。
4. 接入 YOLO。
5. 确保 Template 全局门禁 PASS 时产生所有 24 个 `(8 view × 3 branch)` 结果；NG 时只执行八个
   Template，并保留 16 个明确的模型 `SKIPPED` 记录。
6. 新增简单融合。
7. 使用已有 final-test 八视角图片完成离线 smoke。

验收：不调用 Stage18、Stage34、Stage37 或 runtime bundle，也能输出完整八视角结果。

### 阶段 3：接入实时采集和 Dashboard

1. 把 `--demo-config` 从 Dashboard 传到采集与 worker。
2. 删除在线 CLI 的 `--runtime-config`。
3. worker 启动时只加载模型。
4. 每件检测前重新读取阈值。
5. 修复失败状态显示。
6. Dashboard 显示 Demo 标签、当前配置路径和配置修改时间。

验收：手工修改一个阈值后，下一件检测使用新值，不生成新版本、不更新 SHA、不重载 GPU 模型。

### 阶段 4：清理旧严格链

1. 用 `rg` 列出 bundle、Stage18、Stage34、Stage37 在在线路径中的所有引用。
2. 删除已无调用者的严格在线模块、CLI、配置和测试。
3. 保留训练和独立评估需要的代码。
4. 删除 README/运行说明中旧 bundle 启动命令，改为唯一 Demo 命令。
5. 运行 scoped tests 和 py_compile。

验收：Dashboard 到模型推理的依赖图中不再包含 bundle、SHA、Stage18、Stage34 或 Stage37。

### 阶段 5：真实设备验收

1. 使用现有四相机完成一次正面和背面采集。
2. 检查八张原图。
3. 检查 Template PASS 件产生 24 条模型结果。
4. 检查两个 secondary 在 Template PASS 件中确实运行三条分支。
5. 修改一个 Template 阈值，再检测一件，确认新阈值立即生效。
6. 检查相机或配置错误能在 Dashboard 正确展示。

验收后再讨论是否物理删除历史 bundle 结果目录。

## 9. 最小测试集

不要再运行数百项严格发布测试。新增并保留约 10～15 个高价值测试：

1. 有效 DemoConfig 可加载。
2. 缺少或多出视角会失败。
3. 任一 secondary 缺失会失败。
4. NaN、Infinity 或越界阈值会失败。
5. 模型或 ROI 路径不存在会失败。
6. 在线链只传 `--demo-config`，没有 `--runtime-config`。
7. Template 全部 PASS 时推理产生 24 条完整结果。
8. Template PASS 件的两个 secondary 均有三条真实分支。
9. Template NG 时 PatchCore/YOLO 不被调用，未执行分支明确为 `SKIPPED`。
10. 缺证据或模型异常不能输出 OK。
11. 修改阈值后下一件生效。
12. Dashboard 显示真实失败原因。
13. 一次已有图片离线 smoke。
14. 一次真实四相机 smoke。

## 10. 最终运行命令

目标命令：

```bash
cd /home/yunjing/anomaly_xingtao_new

UV_CACHE_DIR=/tmp/uv-cache \
PYTHONPATH=/opt/MVS/Samples/64/Python/MvImport:. \
uv run --no-sync python \
  pipeline/36_zs32_inspection_dashboard.py \
  --live \
  --demo-config configs/zs32/zs32_demo.json \
  --part-id zs32_demo_001
```

## 11. 最终验收标准

- 用户只编辑一个 JSON。
- 修改阈值后下一件立即生效。
- 调阈值不创建新版本目录。
- 调阈值不更新任何 SHA。
- 在线检测不读取 runtime bundle。
- 在线检测不执行 Stage18、Stage34 或 Stage37。
- 调阈值不重载 PatchCore、YOLO 或 Template 图片。
- Template 全部 PASS 时，三个模型分支在八个视角上全部运行。
- 任一 Template NG 时，整件输出 `NG_TEMPLATE`，PatchCore/YOLO 全部 `SKIPPED`，Fusion 逐视角显示
  对应 Template 的 `NG_TEMPLATE` 或 `PASS`。
- 两个 secondary 只允许因全局 Template 门禁 NG 而 skipped。
- 缺视角或模型异常不能输出 OK。
- Dashboard 只有一条正式入口命令。
- 代码中没有 strict/demo 双模式和散落的安全绕过开关。

## 12. 新对话执行要求

新对话开始后：

1. 先完整阅读本文。
2. 检查当前 dirty worktree，不得覆盖用户已有修改。
3. 用 `rg` 核对本文列出的真实入口、imports 和路径；本文的路径可能因后续会话而漂移。
4. 先汇总不一致和必须确认的问题。
5. 用户确认后再实施。
6. 对复杂实现使用多 agent，但不同 agent 不得同时修改同一文件。
7. 使用 `uv` 环境。
8. 用 `apply_patch` 修改代码。
9. 优先复用当前 Template、PatchCore、YOLO 和 Dashboard 实现，不重写模型算法。
10. 首先交付最小可运行链，再清理旧严格代码；不要同时进行无关重构。
11. 不进行大量历史 `results/` 删除，除非用户明确批准确切目录。
12. 完成时给出修改文件、实际测试结果、DemoConfig 路径和唯一运行命令。
