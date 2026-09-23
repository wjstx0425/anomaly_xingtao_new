# BMW back 钢印读取：60张扩展验证与独立命令

> 后续用户已确认编码只含字母数字。新使用方式与改进结果见 [V2说明](stamp_reading_alphanumeric_v2_20260909.md)，请选择`stamp_read_*_v2.json`。本文保留V1实验结果用于对照。

2026-09-09。用户确认的范围：扩大样本、核对固定ROI、整理识别错误、提供单图读取命令。只读编码，不实施工单、日期、批次或合格判定，不接入原25项检测和相机流程。

## 本轮结果

从 `dataset/bmw_lab_raw_clean_0820` 选60张新路径，排除上轮8张：左右各30张，每侧18张normal、12张defect。左normal覆盖3个会话，各6张；右normal覆盖2个会话，各9张；左右defect各1个会话，各12张，共7个会话。各组按排序后的路径均匀取样，未按OCR输出挑图。

采集时间覆盖2026-08-20与2026-08-23。60是图像数量，不是已确认的独立工件数量；不同会话可能重复拍摄同件。normal/defect是整件目录分类，不是钢印正误标签。

完整输入和逐图参考转录见 [manifest](stamp_reading_validation_20260909_manifest.json)。目视参考由助手在读取算法结果前查看原图裁剪形成，右30张由独立助手逐图查看；不是甲方标注或工单真值。28号损伤跨过末尾字符，56号暗区和破损穿过开头字符，均标为参考不确定，不纳入完整字符串准确度分母，也没有删掉其推理结果。

| 项目 | 左件 | 右件 | 合计 |
|---|---:|---:|---:|
| 图像数 | 30 | 30 | 60 |
| 通过当前读取筛选 readable | 25 | 25 | 50 |
| 提示复核 review | 5 | 5 | 10 |
| 可明确目视转录数 | 29 | 29 | 58 |
| 原始字符串完全一致 | 25/29 | 26/29 | 51/58（87.9%） |
| 仅字母数字一致，忽略空白/标点 | 29/29 | 26/29 | 55/58（94.8%） |

“忽略空白/标点”仅用于第二个评估指标，**不会改写程序的原始读数**。分成两框而没有唯一编码的样本也计入未匹配，不从分母中移除。

在50张readable中，字母数字与目视参考一致；其中0、21、26号有额外冒号/点号，原始字符串仍不完全一致。readable只是通过当前置信分数和几何筛选，不保证字符串正确，更不是工件合格。

最终60张回放的OCR调用中位数122.6ms，P95 137.9ms，范围112.6—181.5ms。仅含CPU OCR调用，排除模型加载、文件读写、ROI处理、相机采集与保存；本次是功能验证时的观测值，不是专门的节拍基准。

## 发现与处理

- 60张主编码均完整落在前轮固定ROI内，未发现需要逐图移动框才能读到编码的情况，因此本轮没有增加定位算法。
- 28、31、46号检测成多个编码候选框，保留所有框和读数、输出review；不自动把候选拼成确定答案。
- 42号漏掉尾部`02`，56号漏掉开头`5A`，引擎分数仍分别约0.999和0.991。新增可配置的文字框横向覆盖比例检查：低于编码搜索区域宽度的0.70时提示`incomplete_code_span`，原始残缺字符串照常保存。
- 10、13、14、27、51号分数低于0.95，保留读数并提示复核。46号也同时存在低分。
- 0、21、26号仍存在高分的标点误识别；14号有额外前导空格。不根据工单或已知样本字符串纠正。

0.95与0.70是实验筛选参数，没有经过独立验收标定；几何覆盖检查不能证明所有字符完整。覆盖比例提示是在查看本轮漏读后增加，并在同60张回放统计，因此不能将该复核效果称为独立测试集泛化性能。现有样本主要只有两种编码，不足以验证新数字排列和所有字符混淆。

## 独立命令

实现：`src/bmw_inspection/checks/stamp_reader.py`；入口：`src/bmw_inspection/cli/read_stamp.py`。继续使用本轮实测的RapidOCR ONNX Runtime 1.4.4，CPU双线程、模型实例复用，原图裁剪后顺时针90度，无训练、生成式补字或期望字符串提示。

左右配置独立位于 `configs/bmw/checks/stamp_read_{left,right}_v1.json`。输入限定4024×3036、uint8 BGR三通道原始back图片；尺寸不符、ROI越界、灰度/16位图片均明确拒绝，不静默缩放或降位深。

配置中`roi_xyxy`是原图半开坐标，`code_region_xyxy`是顺时针90度后的钢印ROI坐标，OCR框也在该旋转ROI中。只从编码区域选候选，未找到编码时不能退回BMW或LH/RH文字。CLI不从像素或文件名推断曝光来源，输出`source_kind=unknown`；本次批量回放的`fused_only`来自明确记录的样本manifest。

当前临时环境已可用。仓库根目录直接运行以下命令，输出目录必须尚不存在：

```bash
PYTHONPATH=src /tmp/bmw-stamp-ocr-env/bin/python -m bmw_inspection.cli.read_stamp \
  --config configs/bmw/checks/stamp_read_left_v1.json \
  --image dataset/bmw_lab_raw_clean_0820/left/back/normal/20260820_162040_682498/images/left_back_normal_bmw_left_normal_group002_000001_fused.png \
  --output artifacts/stamp_single_result
```

该示例预期读出`5A9D6B3 02`，原始模型分数约0.976。右件需显式改用右配置和右back图片。

输出：`result.json`、`roi_original.png`、`roi_upright.png`、`overlay.png`。JSON包含原始文字、所有框、单编码分数、筛选原因、坐标与配置快照、引擎版本和调用耗时；分数不是正确率。叠加图蓝框为配置编码区域、红框为候选编码、绿框为其他文字。

退出码0=通过当前读取筛选，3=需要复核，2=输入/配置/缺少OCR依赖。其他后台执行或保存故障明确报错并返回非零，不将异常伪装成成功。单图命令没有工件放行语义。

临时环境丢失后，可用uv重新准备独立环境；不要将这些OCR/OpenCV依赖安装进现有BMW GUI环境：

```bash
UV_CACHE_DIR=/tmp/bmw-uv-cache uv venv --python 3.13 /tmp/bmw-stamp-ocr-env
UV_CACHE_DIR=/tmp/bmw-uv-cache uv pip install \
  --python /tmp/bmw-stamp-ocr-env/bin/python \
  -r bmw_runtime/requirements-stamp-ocr.txt
```

这些版本来自已执行的独立环境快照，没有修改根或BMW运行环境的锁文件。安装是明确的联网准备步骤；运行只使用随包模型和本机图像。ONNX Runtime导入时出现遥测标识持久化警告，推理仍成功；运行时事件已禁用，生成的`:memory:.ses`文件归入本地实验目录。

## 证据与复现

最终证据在 `artifacts/bmw_stamp_validation_20260909/reader_final/`（本地Git忽略）：

- `review.html`：60张可逐项打开的对照页，含原始文字、参考转录、复核原因和各项JSON链接。
- `summary.json` / `results.json`：汇总与逐图记录。
- `sample_00`—`sample_59`：各图的原始ROI、转正ROI、框图和完整结果。

可重复回放同一清单，输出新目录：

```bash
PYTHONPATH=src /tmp/bmw-stamp-ocr-env/bin/python tools/bmw/validate_stamp_reading.py \
  --manifest docs/bmw/stamp_reading_validation_20260909_manifest.json \
  --output artifacts/bmw_stamp_validation_replay
```

新增25项离线测试覆盖空间选码、缺码不回退、模糊候选、低分、触边、高分残缺框、坐标/尺寸/位深校验、输入数组保护、原文保留、拒绝覆盖、CLI帮助和复核退出码。与原BMW测试及兼容入口一起实际运行：

```bash
PYTHONPATH=src UV_CACHE_DIR=/tmp/bmw-uv-cache MPLCONFIGDIR=/tmp/bmw-mpl-cache \
  uv run --no-sync python -m pytest -q \
  tests/unit/bmw_inspection tests/unit/pipeline/test_bmw_lab_eight_view_demo.py
```

结果：**172 passed，1条既有NVML警告**。原始日志为`artifacts/bmw_stamp_validation_20260909/tests_final.log`。真实CLI另测1号返回0、56号返回3，并保存原始证据，日志为`cli_smoke.json`。以上不包含GPU、实机相机、在线界面或钢印业务规则验收。
