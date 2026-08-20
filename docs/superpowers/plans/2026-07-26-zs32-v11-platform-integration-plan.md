# ZS32 v11 检测平台接入实施计划

**目标：** 将已训练完成的 YOLO、八视角 PatchCore 和八视角 Template 模型封装为新的、不可变的 v11 24-group commissioning runtime bundle，并将 Stage35 与 Dashboard 的默认入口切换到该 bundle，同时保留 v10 回滚路径。

**设计依据：** `docs/superpowers/specs/2026-07-26-zs32-v11-platform-integration-design.md`

**环境约束：**

- 使用 `uv` 管理 Python 环境。
- 不覆盖任何 v10 或旧 24-group 产物。
- 所有 Stage37 命令显式设置 `PYTHONPATH=.`。
- 当前 Codex 环境无 CUDA/NVML；GPU 推理和真实相机 smoke test 交由用户终端执行。

## Task 1：新增 v11 24-group source/profile 配置

**文件：**

- 新增：`config/fusion/zs32_eight_view_24group_v11_bundle_source.json`
- 新增：`config/fusion/zs32_right_eight_view_24_group_commissioning_v11.json`
- 测试：`tests/unit/capture_data/test_zs32_runtime_bundle.py`

**步骤：**

1. 先增加配置契约测试，断言：
   - profile 恰好包含 24 个 group；
   - 八个视角各包含 Template、PatchCore、YOLO；
   - source 绑定用户确认的 v11 三类模型目录；
   - ROI 固定为 `dataset/zs32_all_plus_0723_retraining_release_v2/roi_config.json`；
   - profile 不携带旧 bundle 的 `expected_versions`。
2. 运行目标测试并确认因新配置不存在而失败。
3. 参照现有 24-group profile 和 bundle source，创建两个 v11 配置。
4. 重新运行目标测试并确认通过。
5. 对 JSON 做解析检查和 `git diff --check`。

**验证命令：**

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run --no-sync pytest \
  tests/unit/capture_data/test_zs32_runtime_bundle.py -q

UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -m json.tool \
  config/fusion/zs32_eight_view_24group_v11_bundle_source.json >/dev/null

UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -m json.tool \
  config/fusion/zs32_right_eight_view_24_group_commissioning_v11.json >/dev/null
```

## Task 2：发布 v11 runtime assets

**输入：**

- source：`config/fusion/zs32_eight_view_24group_v11_bundle_source.json`
- profile：`config/fusion/zs32_right_eight_view_24_group_commissioning_v11.json`
- ROI：`dataset/zs32_all_plus_0723_retraining_release_v2/roi_config.json`

**输出：**

- `results/zs32_runtime_assets_eight_view_v11/`

**步骤：**

1. 运行 Stage37 `publish-assets`。
2. 校验生成的 `runtime_assets.json`：
   - 三类模型路径均指向 v11；
   - Template/PatchCore 都有八个视角；
   - ROI version 为 `zs32-eight-view-roi-9412b2838cdb`；
   - 文件 SHA256 与实际文件一致；
   - 没有引用 v10 模型目录。

**命令：**

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run --no-sync python \
  pipeline/37_publish_zs32_runtime_bundle.py publish-assets \
  --source config/fusion/zs32_eight_view_24group_v11_bundle_source.json \
  --output-dir results/zs32_runtime_assets_eight_view_v11
```

## Task 3：生成 v11 commissioning thresholds

**输入：**

- profile：`config/fusion/zs32_right_eight_view_24_group_commissioning_v11.json`
- runtime assets：`results/zs32_runtime_assets_eight_view_v11/runtime_assets.json`
- Template model：`results/zs32_template_gate_right_0723_eight_view_v11/model.json`
- 基础阈值：
  - `results/zs32_stage33_eight_view_commissioning_v4/template_patchcore_threshold_calibration/thresholds.json`
  - `results/zs32_stage33_eight_view_commissioning_v4/yolo_high_precision_auxiliary_test_leakage/thresholds.json`

**输出：**

- `results/zs32_24group_0723_v11_commissioning/`

**步骤：**

1. 运行 Stage34，并显式允许 commissioning-only 的测试集阈值来源和模型重绑定。
2. 固定 YOLO 阈值为 `0.07`。
3. 使用各 PatchCore v11 summary 中的 deploy threshold，且 low/high 相同。
4. 旧 Stage33 ROI 与 v11 ROI 的文件 SHA256 相同、版本名不同，因此必须同时提供
   `--allow-roi-version-rebind` 和旧 ROI 文件；Stage34 必须验证签名 Stage33 run contract、其绑定的旧
   runtime config、旧 runtime 两份 ROI 以及新 runtime 两份 ROI，所有 ROI 字节 SHA256 完全相同后才允许重绑定。
5. 检查 `summary.json` 和 `thresholds.json`：
   - 24 个 group 全部存在；
   - 两个 secondary Template/PatchCore group 均未 skipped；
   - `commissioning_only=true`；
   - YOLO、PatchCore 模型重绑定和 test leakage 均被显式记录；
   - Template 使用当前模型的 calibration count；重绑定 PatchCore/YOLO 不继承旧 count；
   - 不新增无 GT mask 的 pixel metric 字段。

**命令：**

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run --no-sync python \
  pipeline/34_publish_zs32_18_group_commissioning.py \
  --profile results/zs32_runtime_assets_eight_view_v11/fusion_profile.json \
  --runtime-config results/zs32_runtime_assets_eight_view_v11/runtime_assets.json \
  --template-model results/zs32_template_gate_right_0723_eight_view_v11/model.json \
  --template-patchcore-thresholds results/zs32_stage33_eight_view_commissioning_v4/template_patchcore_threshold_calibration/thresholds.json \
  --yolo-auxiliary-thresholds results/zs32_stage33_eight_view_commissioning_v4/yolo_high_precision_auxiliary_test_leakage/thresholds.json \
  --output-dir results/zs32_24group_0723_v11_commissioning \
  --commissioning-only \
  --allow-test-leakage \
  --allow-yolo-model-rebind \
  --allow-patchcore-model-rebind \
  --allow-roi-version-rebind \
  --source-roi-config /home/yunjing/anomalib/dataset/zs32_eight_view_roi_config.json \
  --yolo-threshold-override 0.07 \
  --patchcore-threshold-override front:0.4711672067642212:0.4711672067642212 \
  --patchcore-threshold-override front_left:0.5397635102272034:0.5397635102272034 \
  --patchcore-threshold-override front_right:0.814327597618103:0.814327597618103 \
  --patchcore-threshold-override front_secondary:0.5006473064422607:0.5006473064422607 \
  --patchcore-threshold-override back:0.4759141802787781:0.4759141802787781 \
  --patchcore-threshold-override back_left:0.6454828977584839:0.6454828977584839 \
  --patchcore-threshold-override back_right:0.5977694988250732:0.5977694988250732 \
  --patchcore-threshold-override back_secondary:0.49985870718955994:0.49985870718955994
```

## Task 4：finalize v11 immutable runtime bundle

**输出：**

- `results/zs32_runtime_bundle_eight_view_v11/runtime_bundle.json`

**步骤：**

1. 用 Stage37 `finalize` 组合 runtime assets、profile 和 thresholds。
2. fresh-load 新 bundle，确认 24 个 group 和版本绑定完整。
3. fresh-load v10 bundle，确认回滚路径仍可用。
4. 检查最终 bundle 的 `commissioning_only=true`、`production=false` 和所有 SHA256。

**命令：**

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run --no-sync python \
  pipeline/37_publish_zs32_runtime_bundle.py finalize \
  --assets-manifest results/zs32_runtime_assets_eight_view_v11/assets_manifest.json \
  --threshold-artifact results/zs32_24group_0723_v11_commissioning/thresholds.json \
  --output-dir results/zs32_runtime_bundle_eight_view_v11

UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run --no-sync python -c \
  "from pathlib import Path; from capture_data.zs32_runtime_bundle import load_runtime_bundle; print(load_runtime_bundle(Path('results/zs32_runtime_bundle_eight_view_v11/runtime_bundle.json')))"

UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run --no-sync python -c \
  "from pathlib import Path; from capture_data.zs32_runtime_bundle import load_runtime_bundle; print(load_runtime_bundle(Path('results/zs32_runtime_bundle_eight_view_20group_patchcore_tuned_v10/runtime_bundle.json')))"
```

## Task 5：切换平台默认 bundle

**文件：**

- 修改：`pipeline/35_run_zs32_live_commissioning.py`
- 修改：`src/zs32_inspection/dashboard/live.py`
- 修改：`tests/unit/pipeline/test_zs32_live_commissioning_cli.py`
- 修改：`tests/unit/zs32_refactor/dashboard/test_live.py`

**步骤：**

1. 先修改测试，使 Stage35 与 Dashboard 默认路径都期望：
   `results/zs32_runtime_bundle_eight_view_v11/runtime_bundle.json`。
2. 运行两个目标测试并确认旧默认值导致失败。
3. 仅替换两个生产入口的默认路径，不改变显式 `--runtime-config` 覆盖行为。
4. 重新运行目标测试并确认通过。

**验证命令：**

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run --no-sync pytest \
  tests/unit/pipeline/test_zs32_live_commissioning_cli.py \
  tests/unit/zs32_refactor/dashboard/test_live.py -q
```

## Task 6：文档、项目记忆和完整回归

**文件：**

- 修改：`docs/ZS32_0723_RETRAINING_PREPARATION_20260726.md`
- 修改：`AGENTS_MEMORY.md`
- 修改：`pipeline/AGENTS_MEMORY.md`

**步骤：**

1. 记录 v11 三类模型路径、24-group bundle 路径、commissioning-only 边界和 v10 回滚路径。
2. 记录无 GT mask，因此 pixel AUROC/AUPRO/F1/IoU 为 N/A。
3. 记录 YOLO 当前验证召回偏低，现场 smoke test 必须关注漏检。
4. 运行与 runtime bundle、Stage35、Dashboard、多模型推理相关的回归测试。
5. 运行 `git diff --check`，并核对没有修改或覆盖任何 v10 产物。
6. 向用户提供 GPU/相机终端上的最终启动与 smoke test 命令。

**回归命令：**

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run --no-sync pytest \
  tests/unit/capture_data/test_zs32_runtime_bundle.py \
  tests/unit/pipeline/test_zs32_runtime_bundle_cli.py \
  tests/unit/capture_data/test_zs32_live_commissioning.py \
  tests/unit/pipeline/test_zs32_live_commissioning_cli.py \
  tests/unit/pipeline/test_zs32_multimodel_inference.py \
  tests/unit/zs32_refactor/dashboard/test_live.py \
  tests/unit/zs32_refactor/runtime/test_dashboard_cli.py -q

git diff --check
```

**现场启动命令：**

```bash
HF_HUB_OFFLINE=1 UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run --no-sync python \
  pipeline/35_run_zs32_live_commissioning.py
```

如需临时回滚到 v10：

```bash
HF_HUB_OFFLINE=1 UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run --no-sync python \
  pipeline/35_run_zs32_live_commissioning.py \
  --runtime-config results/zs32_runtime_bundle_eight_view_20group_patchcore_tuned_v10/runtime_bundle.json
```
