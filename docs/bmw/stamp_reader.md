# BMW 钢印读取接口

独立读取 back 图像中的字母和数字。当前采用 V2：保留原始读数，去除空白和标点，困难样本保留复核状态。工单比对、字符业务规则及主检测流程接入留给后续工作。

## Python 接口

从项目根目录运行，设置 `PYTHONPATH=src`。OCR 依赖安装在独立环境：

```bash
uv venv --python 3.13 /tmp/bmw-stamp-ocr-env
uv pip install --python /tmp/bmw-stamp-ocr-env/bin/python -r bmw_runtime/requirements-stamp-ocr.txt
```

已有该环境时直接复用。部署时可把环境建在持久目录；当前固定依赖在本机 CPU 验证过，Windows 和现场并行运行尚未验证。

```python
from bmw_inspection.checks import StampReader, StampReaderConfig, StampReadResult

config = StampReaderConfig.from_json("configs/bmw/checks/stamp_read_left_v2.json")
reader = StampReader(config)
reader.initialize()  # 只加载模型，不执行预测；也可由首次 read 自动加载

# back_bgr 由调用方提供，必须是完整的 uint8 BGR 图像，shape=(3036, 4024, 3)。
result: StampReadResult = reader.read(
    back_bgr,
    capture_id="capture-001",
    inspection_id="inspection-001",
    source_kind="fused_only",
)
print(result.normalized_code, result.state, result.code_score, result.reasons)
payload = result.to_dict()  # 独立、可 JSON 序列化的元数据快照
# 可选保存证据，目录必须不存在：
result.save("artifacts/my_stamp_read_001")
```

右件改用 `stamp_read_right_v2.json`，左右件由调用方明确选择。读取器不采集图片，不猜测编号，也不修改输入图像。可选编号必须为非空字符串；`source_kind` 可选 `unknown`（默认）或 `fused_only`，只记录调用方声明，不验证曝光来源。

同一实例连续复用模型；每个工作线程使用自己的实例，避免同时调用同一个实例。`initialize()` 不等于推理预热。V2 默认 ONNX intra-op=4、inter-op=2，可在 JSON 中配置，或用 `dataclasses.replace(config, intra_op_num_threads=2)` 创建新配置后构造新读取器。旧版 V1 配置显式保留 2/2 线程以便对照。

## 返回约定

| 字段或属性 | 含义 |
|---|---|
| `normalized_code` | V2 去除空白/标点后的文本；可能为 `None`，review 时也可能有候选文本 |
| `raw_code` | 当前选中读数的原文；全部主流程 OCR 观察保存在 `lines` |
| `state` | `readable` 或 `review`，是读取状态，不是产品 OK/NG |
| `code_score` | 当前选中读数的置信度；补读成功时为两次补读的最低分，可能为 `None` |
| `reasons` | 只读 tuple，说明需要复核的原因 |
| `to_dict()` | 包含配置、编号、原始框、归一化审计、补读证据和耗时的独立字典，不含图像数组 |
| `save()` | 保存元数据、ROI、叠加图及存在时的补读图像 |

现有 `.payload` 和 `bmw_inspection.checks.stamp_reader` 导入保持兼容；新调用方优先使用公共导入、属性及 `to_dict()`。图像证据数组保留在结果对象中。主流程框使用旋转后的 ROI 坐标；补读选中框查看 `effective_code_geometry`，不要把主流程框当成补读框。

配置、尺寸、dtype 或编号错误抛出 `ValueError`；缺少 OCR 依赖抛出 `ModuleNotFoundError`；文件保存错误抛出相应文件系统异常。运行异常由集成方处理，与正常返回的 `review` 分开。没有识别出唯一代码时不得把空值当成功。

## 单图、复核与测速

以下命令在项目根目录执行，输出目录或 JSON 文件必须是新路径。

```bash
PYTHONPATH=src /tmp/bmw-stamp-ocr-env/bin/python -m bmw_inspection.cli.read_stamp \
  --image /path/to/back.png --config configs/bmw/checks/stamp_read_left_v2.json \
  --capture-id capture-001 --inspection-id inspection-001 --source-kind fused_only \
  --output artifacts/stamp_cli_001

PYTHONPATH=src /tmp/bmw-stamp-ocr-env/bin/python tools/bmw/validate_stamp_reading.py \
  --manifest docs/bmw/stamp_reading_validation_20260909_manifest.json \
  --config-version v2 --output artifacts/stamp_replay_001

PYTHONPATH=src /tmp/bmw-stamp-ocr-env/bin/python tools/bmw/benchmark_stamp_reading.py \
  --threads 4 --repetitions 2 --output artifacts/stamp_timing_001.json
```

单图命令 readable 退出码 0、review 退出码 3，参数错误为 2。复核工具生成逐图证据及 HTML；测速工具默认使用既有 60+24 张清单，可重复指定 `--manifest` 替换输入集。模型加载、图像解码和落盘不计入测速；`reader_wall_ms` 包含裁剪与结果构造，`elapsed_ms` 只累计 OCR 调用。

## 已有证据与边界

- [V2 识别与复核结果](stamp_reading_alphanumeric_v2_20260909.md)：原 60 张中 55 张 readable、5 张 review；新增 24 张全部 readable。样本有限且可能重复拍摄相同工件。
- [CPU 线程对照](stamp_reading_speed_20260910.md)：本机 4 线程 OCR 中位数约 78 ms，P95 约 91 ms；168 次成对输出除耗时外完全一致。这是独立 CPU 测试，不代表主检测流程并行运行的节拍。
- 已有低置信度、B/8、D/0、缺失或重叠字符处理保留。没有根据预期钢印补字符，也没有加入工单规则。

代码分工：`stamp_config.py` 管配置，`stamp_reader.py` 管推理和补读，`stamp_result.py` 管结果及证据保存；CLI 和批量工具调用同一公共接口。OCR 保持可选依赖，未加入 BMW 主运行环境的依赖锁定文件。

本次整理验证（2026-09-10）：237 项 BMW/钢印测试通过；84 张 CPU 重放与整理前的所有非耗时、非新增身份、非线程配置字段完全一致。证据位于 `artifacts/bmw_stamp_handoff_20260910/parity.json`，复核网页分别在 `original60/review.html` 和 `new24/review.html`。单图 CLI 身份字段保存及 24 张测速工具冒烟验证通过。测试环境有已有 NVML 警告，未据此声称 GPU 可用。
