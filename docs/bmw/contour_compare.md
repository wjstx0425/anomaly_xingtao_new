# BMW 单视角整圈外轮廓：离线开发版

实现位置：`src/bmw_inspection/checks/contour_compare/`；命令模块：`src/bmw_inspection/cli/contour.py`。对应交接包 C0—C3 的开发框架；C4 真实重复取放与公差验证、C5 八视图集成尚未进行。原钢印、孔检测、25 条模型与相机流程保留。

用户已选定 4024×3036 正常参考和五张缺陷图，已另建全分辨率配置并完成实图复测。最新结果见 [9月12日连续轮廓八图对照](contour_continuous_20260912.md)：固定参考和公差下自检覆盖89.05%，但正常001仍有5个NG候选，全部正式结果仍为REVIEW。人工定位与局部阈值实验见 [B定位复核](contour_fixture_B_20260912.md)，参考修订见 [9月12日复测](contour_reference_20260912.md)，前轮算法修改见 [误报修复报告](contour_fix_20260911.md)。文档原定 2047×1545 图仍未找到，不能混用规格；历史核查见 [C0 审计](contour_c0_audit_20260911.md)。

## 已实现

- `teach`：参考阶段交互 GrabCut，ROI、前/背景笔刷、缩放、平移、撤销和预览；也支持无 GUI 的已确认 mask。保存原图、完整密集曲线、按弧长采样、内法向、结构弧段、mask、SHA 和版本化 recipe。
- 标准上未知边段或无法确认内外法向时保持 draft。`inspect` 拒绝 draft、无效规格/ROI/必要参数、非法容差和错误参考资产，不自动缩放。
- 两个或更多独立暗孔锚点拟合二维正向刚性矩阵；检查椭圆、孔间距、移动、角度和拟合残差。失败保持 REVIEW，不回退单位矩阵。两个锚点不能独立验证三维姿态，结果明确记录该限制。
- 整图 GrabCut 使用可改变的参考概率初值；原始组件保留诊断，按参考内部核心的重叠确定工件归属；近边不明组件触发 REVIEW，不把名义轮廓强制为前景。全周灰度剖面、局部二维角点和独立粗轮廓检查相互提供证据。
- 双向点到线段距离、向内/向外符号、按参考弧长聚合事件；未知段不造连接线。短超差仍输出 REVIEW，明确 NG 与未知覆盖同时保留。
- `inspect` 输出 PNG/NPZ/CSV/JSON；临时目录完整写完后发布，不覆盖旧目录。`evaluate` 单独消费标签并输出状态、覆盖、耗时和拆分泄漏信息。

当前只支持整圈必检、零允许未知长度和 `merge_valid_gap_px=0`。非空独立检查区域暂拒绝；可以使用三个或更多拟合锚点，但不等价于独立姿态验证。默认 `independent_v2` 模式中角点移动超过 1.5 px 保守标 UNKNOWN；显式启用 `continuous_v3` 后，参考与待测采用连续候选路径和两侧锚点之间的受支持转角边链，缺失或歧义仍为 UNKNOWN。粗边全局候选独立保留；复杂窄脚/暗边尚需实图开发。没有工业检出率、毫米精度或节拍承诺。

## 使用已有 uv 环境

仓库 `pyproject.toml` 已注册 `bmw-contour`。本机现有根目录环境尚未重新安装脚本入口，也没有 `bmw_runtime/.venv`；本轮实际使用下面的模块命令，不执行依赖升级或重新建立 GPU 环境。正常安装项目后可用交接文档的 `bmw-contour` 入口。

```bash
UV_CACHE_DIR=/tmp/bmw-uv-cache uv run --no-sync python -m bmw_inspection.cli.contour --help
```

建立参考（图片必须与 draft 的尺寸/孔窗口一致）：

```bash
UV_CACHE_DIR=/tmp/bmw-uv-cache uv run --no-sync python -m bmw_inspection.cli.contour teach \
  --image /path/to/reference.png \
  --draft-config configs/bmw/checks/contour/left_front.draft.json \
  --output-dir dataset/bmw_contour/reference/left_front_v001
```

交互窗口：`R` 框整件 ROI，`F/B` 前景/背景笔刷，`G` 运行分割，`U` 撤销，滚轮或 `+/-` 缩放，中键拖动平移，`S` 确认标准并保存，Esc 取消。锚点和参数通过 draft 或 `--parameters overrides.json` 设置；不要将待测图人工描边结果输入检测。

无 GUI：增加 `--mask confirmed_mask.png --confirm-reference --roi x1 y1 x2 y2`。mask 必须是与输入同尺寸的单通道 uint8，非零表示前景。`--confirm-reference` 仅表示操作者已经确认该参考边界；不表示现场阈值验收。单独 `--headless` 自动分割只生成 draft，不能直接 inspect。

`teach` 会为必要 null 参数生成明确标为 development 的候选值，记录参考灰度统计与参数来源，可用 `--parameters` 调整。正式公差不能从单张标准图推断。参考 unknown 标记和结构弧段可由 `--annotations annotations.json` 导入：

```json
{
  "unknown_sample_ids": [],
  "arc_ranges": [
    {"arc_id": 0, "start_sample": 0, "end_sample": 200},
    {"arc_id": 1, "start_sample": 200, "end_sample": 640}
  ]
}
```

上述 640 仅为合成矩形示例。真实模型必须使用 `contour.npz` 的实际采样数量；区间半开、不得重叠，必须覆盖全部采样点。省略 `arc_ranges` 则全部归于 0。`comparison.arc_overrides` 以实际 `arc_id` 覆盖 `inward_tolerance_px`、`outward_tolerance_px`、`min_exceedance_arc_px`，拒绝未知 ID、重复 ID 和 null/非正值。

单张检查：

```bash
UV_CACHE_DIR=/tmp/bmw-uv-cache uv run --no-sync python -m bmw_inspection.cli.contour inspect \
  --config dataset/bmw_contour/reference/left_front_v001/recipe.json \
  --image /path/to/test.png --capture-id capture001 --source-kind fused_only \
  --output-dir results/bmw_contour/capture001
```

退出码：PASS=0、NG=10、REVIEW=20、ERROR=2。PASS 仅指本视角轮廓；`whole_part_release` 永远为 null。RGBA 中 alpha 非 255 位置无效；融合图只计一个来源。

批量检查：

```bash
UV_CACHE_DIR=/tmp/bmw-uv-cache uv run --no-sync python -m bmw_inspection.cli.contour evaluate \
  --config dataset/bmw_contour/reference/left_front_v001/recipe.json \
  --manifest /path/to/validation.csv --output-dir results/bmw_contour/validation_v001
```

CSV 必需字段为 `sample_id,hand,view_id,image_path,channel`；另存 `physical_part_id,placement_group,split,ground_truth`。图片路径相对 manifest；缺失物理身份保留 null。`normal/OK/PASS` 和 `defect/NG` 标签用于汇总，其他标签保留但不纳入这两类统计；标签不进入预测。批量处理完成返回 0，即使样本含 NG/REVIEW/ERROR；报告分别记录，不把 ERROR 当成功检测。

## API 与输出

```python
from bmw_inspection.checks.contour_compare import load_reference
from bmw_inspection.cli.contour import inspect_image

reference = load_reference("path/to/recipe.json")
observation, result = inspect_image(bgr_or_bgra_uint8, reference,
                                    capture_id="capture001", source_kind="fused_only")
```

预加载 reference 可复用；图像数组只读使用，画图另复制。`inspect_image` 返回算法数值和 observation，`write_evidence` 将大数组分离到证据文件，最终 `result.json` 不持有图像。各方向数值分别命名，法向偏移不是最短欧氏距离。精细边缘使用灰度梯度质心；参考图上同法测得固定的材料像素边界坐标校准量，避免整数 mask 与灰度边缘定义产生约 1 px 自检偏移。候选记录原始灰度边位置与校准量；0.25 px 位移保留测试通过，不能据此声称真实成像具备相同量测精度。

证据包括未确认粗边位置 `unconfirmed_coarse_events.json`、`reference_overlay.png`、`test_outline_overlay.png`、`comparison_overlay.png`、`displacement_profile.png`、`contour_samples.csv`、`test_segments.npz`、`distance_samples.json`、`events.json`、`extraction_diagnostics.json`、`result.json`。绿=参考，青=当前观测，橙=未知/候选，红框=NG 事件。测量线段单独存储，缺失段不插值。

## 本次软件验证与待验证事项

合成示例生成命令：

```bash
UV_CACHE_DIR=/tmp/bmw-uv-cache uv run --no-sync python tools/bmw/demo_contour_synthetic.py \
  --output-dir artifacts/bmw_contour_synthetic_20260911_final
```

每次使用新的输出目录。合成参考自检和上下左右四个缺口经过完整 teach→inspect/evaluate 流程；这不是真实 BMW 图，也不是工业准确率验证。最终运行记录见同目录的 `evaluation/report.json`。5 张合成图结果为 1 PASS / 4 NG / 0 REVIEW / 0 ERROR；自检最大偏差约 7.1e-14 px，四个 12 px 缺口均为单事件，全部覆盖完成、未知长度 0。V1/V2 是历史调试产物，使用 final。

初版历史软件回归：**385 passed，1 warning，2.40 s**，其中新增轮廓测试 102 项、保留旧功能测试 283 项。warning 为既有 NVML 初始化警告；没有 GPU 验证。模块 `--help` 和 `compileall` 已通过。交互 GUI 尚未人工操作验收。

软件回归命令：

```bash
UV_CACHE_DIR=/tmp/bmw-uv-cache uv run --no-sync pytest -q \
  tests/unit/bmw_inspection tests/unit/pipeline/test_bmw_lab_eight_view_demo.py
```

此前误报修复阶段软件回归为 **436 passed**，这是历史测试记录。连续轮廓最终实图对照见[独立报告](contour_continuous_20260912.md)；正常参考自检无 NG 不代表完整观测通过。

仍需要：人工确认全圈标准（特别是暗边、小脚、夹具接触）；C4 的重复取放、多件合格及真实缺陷数据、最小目标缺陷和公差。未运行相机、GPU、Windows、交互窗口人工验收或生产放行集成。
