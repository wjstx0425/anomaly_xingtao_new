# ZS32 真机 18 组融合联调设计

## 目标

新增一个终端入口，让操作员在同一终端中完成一件右手 ZS32 零件的正面三相机拍摄、人工翻面、背面三相机拍摄，并立即调用现有 Stage32/Stage18 模板、PatchCore、YOLO 共 18 组 commissioning 融合。

第一版以最快跑通真机为目标。它不加入 quality gate、registration、Geometry、GUI、自动上下料或模型常驻服务。`OK` 只代表 18 组联调合同完整，始终保留 `commissioning_only=true` 和 `production_release_allowed=false`。

## 方案

新增薄入口 `pipeline/35_run_zs32_live_commissioning.py`，不新增模型推理或融合算法；Stage32 仅补齐结果身份字段，供 Stage35 做跨产物一致性校验。入口使用当前 Python 解释器依次启动：

1. `capture_data/collect_multicamera_dataset.py`，采集一件零件的六视角 HDR 图像；
2. 解析本次独立采集目录中的唯一 manifest；
3. `pipeline/32_run_zs32_multimodel_inference.py fuse --fusion-profile zs32-right-18-commissioning`；
4. 读取 `runtime_summary.json`：完整融合时再读取 Stage18 audit 并输出每视角三分支分数；合法模板短路时
   输出 `template_results` 和最终状态，明确 Stage18 audit 不存在。

每件零件重新启动 Stage32 并加载模型。该开销在首次真机验收阶段可以接受，模型常驻作为后续性能优化。

## 命令接口

最小命令为：

```bash
/home/yunjing/anomalib/.venv/bin/python pipeline/35_run_zs32_live_commissioning.py \
  --part-id live_part_001
```

默认资产和参数：

- hand：`right`，不可切换到当前没有模型合同的左手；
- camera serial：中央 `DA9805574`、左侧 `DA9625347`、右侧 `DB0998274`；
- 一件零件：`group-count=1`、`images-per-group=1`、人工翻面；
- HDR：短曝光 `1500 us`、长曝光 `6000 us`、gain `0`、settle frames `1`；
- capture root：`/home/yunjing/anomalib/results/zs32_live_capture`；
- output root：`/home/yunjing/anomalib/results/zs32_live_runtime`；
- runtime config：`/home/yunjing/anomalib/results/zs32_offline_commissioning/runtime_models.local.json`；
- template model：`/home/yunjing/anomalib/results/zs32_template_gate_right_unified_roi_v1`；
- threshold artifact：`/home/yunjing/anomalib/results/zs32_18group_commissioning_v1/thresholds.json`；
- GPU：anomalib `gpu/1`，YOLO device `0`。

入口提供 `--list-devices`，复用现有相机枚举且不启动推理。资产路径、采集根目录、结果根目录、曝光和超时允许通过显式参数覆盖；18 组 profile 和右手限制不可被普通参数放宽。

## 数据和身份映射

每次启动创建唯一的 capture run 子目录，避免通过文件修改时间猜测 manifest。采集器在其中只允许生成一个 manifest。

manifest 必须包含同一个 complete sample：

- 一条 `record_type=sample` 且 `sample_status=complete`；
- 六条 `record_type=image` 且 `sample_status=complete`；
- `front`、`front_left`、`front_right`、`back`、`back_left`、`back_right` 各一条；
- 六个 `file` 均存在且是普通文件；
- image 行的 `session_id`、`sample_id`、`group_id` 一致。

Stage32 身份映射固定为：

- `part_id = sample_id`；
- `capture_session = session_id`；
- `group_id = group_id`；
- 六个 view 的 `file` 分别传给对应的 `--<view>-image`。

采集子进程退出码为零不代表样本完整，因此只有 manifest 校验通过才能启动 GPU 推理。

## 终端交互与输出

操作员看到两次明确的右手件提示：

1. 放好右手件正面，按 Enter 拍摄三个正面视角；
2. 将同一零件翻到背面，按 Enter 拍摄三个背面视角。

同时修正现有采集器把右手件错误显示为“左手件”的文案。

完整进入 Stage18 后，终端按六视角打印：

```text
view         branch           score       low         high        level
front        template_match   ...         ...         ...         CLEAR
front        anomaly_front    ...         ...         ...         CLEAR
front        yolo             ...         ...         ...         CLEAR
```

最后打印 `machine_status`、`inspection_complete`、`commissioning_only`、`production_release_allowed`、capture manifest、runtime summary、audit 和输出目录的绝对路径。完整融合的分支数据以 Stage18 audit 的 `computed_evidence_level` 为准，不信任 CSV 自报等级。模板明确 NG 或模板算子异常可以在 Stage32 内合法短路；此时打印已评估的模板结果和 `audit: none`，`audit_path=None`，不伪造 Stage18 audit。源图解码/尺寸异常仍按 Stage32 非零执行错误处理。

## 失败处理

- 相机枚举、采集或 Stage32 子进程非零：终止本件检测并返回非零；
- manifest 缺失、多个 manifest、sample incomplete、缺失/重复视角或图片不存在：不启动 GPU；
- Stage32 输出目录已存在：拒绝覆盖；
- runtime summary 缺失、JSON 无效或身份不一致：返回非零并打印诊断路径；非短路结果还必须存在且校验
  Stage18 audit，合法模板短路则要求 `short_circuited=true`、受支持的状态和完整模板结果，不要求 audit；
- `inspection_complete=false` 或 `production_release_allowed=true`：终端醒目标记，不能作为生产放行；
- `NG_*`、`REVIEW` 是有效检测结果，入口正常打印并保留 Stage32 产物，不把业务判定误报为程序崩溃；
  当前 Stage32 模板短路只支持 `NG_TEMPLATE` 和异常型 `REVIEW`，二者均为
  `inspection_complete=false` 且没有 Stage18 audit。源图无法解码或尺寸错误会使 Stage32 非零，按执行错误处理。

## 测试与验收

单元测试覆盖：完整 manifest 映射、缺失/重复视角、incomplete sample、图片缺失、多 manifest、采集失败、Stage32 失败、输出目录冲突、终端 audit 表格和右手提示。

不连接相机的验收先用历史 complete manifest 验证参数映射和终端报告。真机验收依次执行 `--list-devices` 和一个真实零件的完整命令，要求生成六张可读 `4024x3036` 图像和 runtime summary。若模板没有短路，还要求 18 条融合证据和 Stage18 audit；若模板合法短路，则要求受校验的 `NG_TEMPLATE`/`REVIEW` 模板结果、`audit=None` 和明确的短路原因。最终状态可以是 `OK`、`NG_*` 或 `REVIEW`，但身份与非生产策略字段必须一致。
