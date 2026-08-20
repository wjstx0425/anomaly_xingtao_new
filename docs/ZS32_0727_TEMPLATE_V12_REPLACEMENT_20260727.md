# ZS32 0727 Template v12 训练与平台候选发布

日期：2026-07-27

## 当前状态

已仅使用 `dataset/zs32_0727` 的 19 个不同正常零件完成八视角 Template v12 训练，
并发布独立的 24-group commissioning 候选 bundle。v11/v10 未被覆盖。

独立 `final_test` 的 16 张正常图中有 4 张被判为 `NG_TEMPLATE`。用户查看八视角总览及
“当前图/最佳模板/差异热图”后接受该结果，Stage35 和 Dashboard 默认路径已切换到 v12。

## 数据与划分

- 原始数据：`dataset/zs32_0727`
- ROI 数据：`dataset/zs32_0727_template_roi_v1`
- 显式 release：`dataset/zs32_0727_template_release_v1`
- 完整组：`group001` 至 `group019`
- 图片：19 parts × 8 views = 152
- 固定 seed：42
- 实体级划分：train/model_val/calibration/final_test = `11/3/3/2`
- ROI SHA256：
  `9412b2838cdb96f722db356714cf9bbbb5ea01810671ccf92b67323de77ebe65`

`group020` 是取消采集且没有图片的记录，不进入任何 manifest。

## Template v12

- 模型目录：`results/zs32_template_gate_right_0727_eight_view_v12`
- 模型 SHA256：
  `5b071bcfb8a603f5bbf20ecf284f590e88e3e65e114f434fbde1eaf81692ff6e`
- 八个视角各 5 张 Template，共 40 张。
- model-val、calibration 和 final-test 均不参与 Template 选择。
- calibration 仅包含 3 个正常零件，因此阈值仍属于 normal-only 临时 commissioning 阈值。

| View | Low/High |
|---|---:|
| front | 0.014885842800140383 |
| front_left | 0.019833445549011234 |
| front_right | 0.022776365280151370 |
| front_secondary | 0.016357362270355228 |
| back | 0.010035157203674318 |
| back_left | 0.015659272670745853 |
| back_right | 0.019793987274169925 |
| back_secondary | 0.026354730129241947 |

阈值源文件：

`results/zs32_template_gate_right_0727_eight_view_v12/model.json`

手工调整时不要原地修改已发布 v12。复制模型到新版本目录、修改
`groups["right/<view>"].low_threshold/high_threshold`，重新计算 `model.sha256`，然后重新发布新的
assets、thresholds 和 final bundle。

## 独立评估

完整的 152 行评分位于：

- `results/zs32_template_gate_right_0727_eight_view_v12/evaluation.csv`
- `results/zs32_template_gate_right_0727_eight_view_v12/evaluation.json`

| Role | PASS | NG_TEMPLATE |
|---|---:|---:|
| train | 88 | 0 |
| model_val | 24 | 0 |
| calibration | 24 | 0 |
| final_test | 12 | 4 |

4 个正常误拒：

- `group001/back`：0.022214174271 > 0.010035157204
- `group004/back`：0.010666966438 > 0.010035157204
- `group001/back_left`：0.022505164146 > 0.015659272671
- `group001/back_right`：0.031112849712 > 0.019793987274

`back_secondary` 的 final-test 为 2/2 PASS；旧 v11 在全部 19 张新 `back_secondary` 正常图上均为
NG，因此新模型已修正该视角的主要域偏移。

## v12 候选平台链

- Source：
  `config/fusion/zs32_eight_view_24group_template_0727_v12_bundle_source.json`
- Profile：
  `config/fusion/zs32_right_eight_view_24_group_commissioning_template_0727_v12.json`
- Runtime assets：
  `results/zs32_runtime_assets_eight_view_template_0727_v12`
- Thresholds：
  `results/zs32_24group_template_0727_v12_commissioning`
- Final bundle：
  `results/zs32_runtime_bundle_eight_view_template_0727_v12/runtime_bundle.json`

关键哈希：

- assets manifest：
  `a7c602b469f90b9c4fc09cd9d689b6f08f5dc779c2604a136d4e06a00b03070b`
- thresholds：
  `f1d06c1c1a3d7345eec1b0efa30f15ffec2aa587a1c6fed27167aaf476cb5b69`
- bundle canonical SHA：
  `1b3341ca05ac7e786733690534c8a75ee6c5a493927145cc39860c4448c29133`

v12 精确要求八视角的 Template、PatchCore、YOLO，共 24 组；两个 secondary 均 required。
PatchCore、YOLO、ROI 的路径、内容 SHA 和运行参数与 v11 保持一致。

## 显式候选 smoke

在相机/GPU 终端运行：

```bash
cd /home/yunjing/anomaly_xingtao_new

UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run --no-sync python \
  pipeline/35_run_zs32_live_commissioning.py \
  --runtime-config results/zs32_runtime_bundle_eight_view_template_0727_v12/runtime_bundle.json \
  --part-id zs32_template_v12_smoke_001
```

显式回滚到 v11：

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run --no-sync python \
  pipeline/35_run_zs32_live_commissioning.py \
  --runtime-config results/zs32_runtime_bundle_eight_view_v11/runtime_bundle.json \
  --part-id zs32_template_v11_rollback_001
```

当前默认入口：

- `pipeline/35_run_zs32_live_commissioning.py`
- `src/zs32_inspection/dashboard/live.py`

两者均解析到：

`results/zs32_runtime_bundle_eight_view_template_0727_v12/runtime_bundle.json`

## 限制

- 全部 19 个零件都是 normal，没有 defect，不能报告缺陷 recall、F1 或 AUROC。
- 没有 ground-truth mask，pixel AUROC/AUPRO/F1/IoU 均为 N/A。
- bundle 保持 `commissioning_only=true`、`production_release_allowed=false`。
- v11 YOLO 阈值来源仍带 test-leakage/model-rebind 警告；v12 没有改变该限制。

## v13 追加说明：手工阈值版本化与默认切换

本节是 v12 历史记录之后的增量说明，不改变上文关于 v12 训练、评估和当时默认路径的结论。
三项经确认的 Template 手工阈值已复制到独立 v13；v12 已恢复为原始模型，继续作为不可变回滚版本。

- v13 Template 模型：
  `results/zs32_template_gate_right_0727_eight_view_v13/model.json`
- 模型 SHA256：
  `eac211cab4ee1d27adda9933c3cb9c6fde71c4a7eb50fae138201a600fa17ab5`
- 手工 Template 阈值（low=high）：
  `front=0.024885842800140383`、
  `front_left=0.029833445549011234`、
  `front_secondary=0.216357362270355228`
- 其余五个 Template 视角保持 v12 原值。

v13 平台链：

- Source：
  `config/fusion/zs32_eight_view_24group_template_0727_v13_bundle_source.json`
- Profile：
  `config/fusion/zs32_right_eight_view_24_group_commissioning_template_0727_v13.json`
- Runtime assets：
  `results/zs32_runtime_assets_eight_view_template_0727_v13`
- Thresholds：
  `results/zs32_24group_template_0727_v13_commissioning`
- Final bundle：
  `results/zs32_runtime_bundle_eight_view_template_0727_v13/runtime_bundle.json`

继续沿用的手工 commissioning 阈值：

- YOLO：八视角均为 `low=high=0.07`。
- PatchCore：
  `front=0.4711672067642212`、
  `front_left=0.5397635102272034`、
  `front_right=0.814327597618103`、
  `front_secondary=0.5006473064422607`、
  `back=0.4759141802787781`、
  `back_left=0.6454828977584839`、
  `back_right=0.5977694988250732`、
  `back_secondary=0.49985870718955994`；每项均为 low=high。

关键哈希：

| Artifact | SHA256 |
|---|---|
| v13 Template `model.json` | `eac211cab4ee1d27adda9933c3cb9c6fde71c4a7eb50fae138201a600fa17ab5` |
| `runtime_assets.json` | `d485865927db250008fb2c9fb95e533d1147b55a04f4cd7bc2898e07520561f1` |
| `fusion_profile.json` | `4f2de0ffcb7b5f7862b6c7e2f86b68688ce5a871bc890c876b0a852de8abec64` |
| `assets_manifest.json` | `bee5847b6a99825150b5417dab69b2fd117f28a6f58818d6b91a833bd622bf7c` |
| asset-set identity | `f24c2e402e3711269fe728b3c884ff3c5d4c095a5955c967086b8a4b99e2721a` |
| `thresholds.json` 文件 | `7b820d05af224771883e133d0da038d5fa3be4aa73386cfe8d8a5f605d1cd268` |
| thresholds canonical artifact | `10825d9c686ceab9699f660c938a9510c36954c2f7feb25c0ac2313e4a8bb479` |
| `runtime_bundle.json` 文件 | `c35474fe1e8bdaa448967a62208a94624029927afd355deb31de4dd1a15560ea` |
| runtime bundle canonical identity | `94b025f85bb2f529fc36a0f62a941eafe30cbd71b1c2298431c24b539a29e2bf` |

发布合同验证通过：24 条唯一 `(hand, view, branch)` 记录、八个 Template 组、Template
模型版本及 low/high 阈值与 v13 `model.json` 精确相等；`front_secondary` 和
`back_secondary` 均要求 Template/PatchCore/YOLO。PatchCore checkpoint、YOLO weights/class map
和 PatchCore/YOLO ROI 绑定与 v12 相同。

当前 Stage35 和 Dashboard 默认解析到 v13。带本机海康 MVS SDK 路径的显式 live 命令：

```bash
cd /home/yunjing/anomaly_xingtao_new

UV_CACHE_DIR=/tmp/uv-cache \
PYTHONPATH=/opt/MVS/Samples/64/Python/MvImport:. \
uv run --no-sync python pipeline/35_run_zs32_live_commissioning.py \
  --runtime-config results/zs32_runtime_bundle_eight_view_template_0727_v13/runtime_bundle.json \
  --part-id zs32_template_v13_smoke_001
```

逐级回滚命令：

```bash
# v12：保留 0727 原始自动阈值
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run --no-sync python \
  pipeline/35_run_zs32_live_commissioning.py \
  --runtime-config results/zs32_runtime_bundle_eight_view_template_0727_v12/runtime_bundle.json \
  --part-id zs32_template_v12_rollback_001

# v11：回到 0723 strict 24-group
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run --no-sync python \
  pipeline/35_run_zs32_live_commissioning.py \
  --runtime-config results/zs32_runtime_bundle_eight_view_v11/runtime_bundle.json \
  --part-id zs32_template_v11_rollback_001

# v10：回到历史 20-group，secondary Template/PatchCore 为 SKIPPED
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run --no-sync python \
  pipeline/35_run_zs32_live_commissioning.py \
  --runtime-config results/zs32_runtime_bundle_eight_view_20group_patchcore_tuned_v10/runtime_bundle.json \
  --part-id zs32_template_v10_rollback_001
```

2026-07-27 的独立新进程验证中，v13、v12、v11、v10 四个 bundle 均 fresh-load PASS。
v13 仍保留既有 YOLO test-leakage、YOLO/PatchCore model-rebind、ROI identity-rebind 标记，
因此仍是 `commissioning_only=true`、`production_release_allowed=false`，不得描述为生产发布。
