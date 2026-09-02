# ZS32 划痕/裂纹与凹坑/斑点传统算子原型

## 结论

这是一条独立、离线、仅用于诊断的实验分支。当前效果**尚未达到人工合并进检测平台主线的条件**：

- 细线算子能标出部分可见线状痕迹，但也大量响应冲压筋、孔边、零件边缘、正常表面纹理和侧视图背景。
- 凹坑/斑点算子能标出局部明暗 blob，但同样会响应圆孔边缘、高光和固定几何细节。
- 当前数据只提供 `normal/defect` 以及 `deform/less/others` 目录标签，没有“划痕 vs 裂纹”或“凹坑 vs 斑点/污渍”的区域级真值，因此不能报告这四类缺陷的准确率、召回率或阈值。

该实现没有导入或调用 Stage 32、Stage 18、Dashboard、v10，也不输出 `pred_label`、`final_status` 或任何融合分支行。

## 独立代码

- 核心算子：`capture_data/zs32_classical_operators.py`
- 离线 benchmark：`capture_data/zs32_classical_benchmark.py`
- 独立入口：`pipeline/38_benchmark_zs32_classical_operators.py`
- 实现分支：`feat/zs32-classical-operators-prototype`
- 隔离 worktree：`/home/yunjing/anomaly_xingtao_new/.worktrees/zs32-classical-operators`

核心输出只有连续、未标定的 diagnostic score、二值 mask、连通域几何证据和 overlay：

- `thin_line`：CLAHE 灰度归一化、Scharr、Laplacian、鲁棒阈值、细长连通域过滤。
- `pit_spot`：多尺度椭圆 top-hat/black-hat、鲁棒阈值、圆度和面积过滤。

材料 mask 会去除粗略轮廓边缘，但它不是精确零件分割，也不替代现有 ROI/CAD/模板几何。

## 实测命令

```bash
/home/yunjing/anomaly_xingtao_new/.venv/bin/python \
  pipeline/38_benchmark_zs32_classical_operators.py \
  --manifest /home/yunjing/anomaly_xingtao_new/dataset/zs32_all_right_patchcore_roi/crop_manifest.csv \
  --output-dir /home/yunjing/anomaly_xingtao_new/results/zs32_classical_operator_benchmark_v1 \
  --max-normal-per-view 1 \
  --max-defect-per-view 4 \
  --seed 0 \
  --resize-scale 1.0 \
  --warmup-rounds 1 \
  --timing-rounds 30 \
  --timing-samples-per-view 1 \
  --parallel-workers 2 \
  --save-overlays
```

产物：

`/home/yunjing/anomaly_xingtao_new/results/zs32_classical_operator_benchmark_v1`

- 6 个主视角，每视角 1 张 normal + 4 张 defect，共 30 张真实 ROI。
- 60 行连续证据，60 张 mask，60 张 overlay。
- 输入保持真实 ROI 分辨率，`resize_scale=1.0`。
- 抽样覆盖目录标签 `deform`、`less`、`others`；这些标签不能解释为表面缺陷类型真值。
- manifest SHA256：`071bcc6d9ee61667699d931f98eeb47abfed8381629fa65460b41642c12e232b`。

## 全分辨率耗时

下列数值只来自 `benchmark.json`。每个正式 round 包含 6 张 timing ROI；OpenCV 报告 20 个内部线程，外层并发为 2 workers。

| 测量项 | 样本数 | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|
| 单张 `thin_line` | 180 | 0.1691 s | 0.3142 s | 0.3201 s | 0.3285 s |
| 单张 `pit_spot` | 180 | 0.2429 s | 0.3771 s | 0.3847 s | 0.3970 s |
| 6 张串行整轮 | 30 | 2.8204 s | 2.8993 s | 2.9759 s | 3.0036 s |
| 6 张、2 workers 整轮 | 30 | 1.5489 s | 1.6116 s | 1.6223 s | 1.6230 s |

本次进程峰值 RSS 为 `1,184,870,400 bytes`（约 1.10 GiB）。benchmark 已改成逐张流式生成证据，只为 timing 保留每视角指定数量的图像；默认 2 workers 是为避免全分辨率并发内存继续放大。

这些时间只代表当前 CPU/OpenCV 环境和当前参数，不包含现有检测平台模型耗时，也不能直接外推为整个平台节拍。

## 实图观察

- `front` normal 已存在肉眼可见的细线、压痕和表面纹理。细线 overlay 同时标出了固定筋条、孔边和下沿，normal 的 `thin_line` score 为 `0.01242`。
- 本次抽样中，6 个视角的 normal `thin_line` score 都高于各自 4 张 defect 的中位数；normal 总体中位数为 `0.00382`，defect 总体中位数为 `0.00252`。这说明当前分数主要受视角、固定几何、表面纹理和采集条件影响，不能直接设一个全局阈值。
- `front_left` 的斜视样本中，粗略材料 mask 把一部分纹理背景纳入了候选，形成明显背景误响应。
- `pit_spot` 在 normal 与 defect 上高度重叠：normal 中位数 `0.000358`，defect 中位数 `0.000403`。圆孔边、高光和固定凸起均会形成橙色 blob 证据。
- `others` 样本中某些可见线状痕迹会被标出，但同一图中仍有大量固定纹理和背景候选，无法据此判断是划痕还是裂纹。
- `deform` 和 `less` 是几何目录标签，本就不应期待表面算子将其稳定分开。

因此，本轮只证明了算子链路、证据可视化和实测速度可运行；没有证明传统算子对目标缺陷有效。

## 下一轮效果验证前置条件

1. 用现有 ROI/CAD/模板几何生成每视角精确材料 mask，并建立固定孔边、筋条和轮廓 ignore mask。
2. 每个主视角单独标定参数和正常纹理基线，不跨视角比较绝对 score。
3. 补一份区域级小评测集，至少明确标注：真实划痕、真实裂纹、真实凹坑、真实斑点/污渍、正常划痕状纹理和高光。
4. 先以 overlay 的候选覆盖率和正常误响应评审效果；通过后再讨论阈值，仍不直接接入 Stage 18/32。

## 复验

```bash
/home/yunjing/anomaly_xingtao_new/.venv/bin/python -m pytest \
  tests/unit/capture_data/test_zs32_classical_operators.py \
  tests/unit/capture_data/test_zs32_classical_benchmark.py \
  tests/unit/pipeline/test_pipeline_wrappers.py -q
```

当前结果：`72 passed`；唯一警告是当前环境无法初始化 NVML，与 CPU/OpenCV 原型无关。
