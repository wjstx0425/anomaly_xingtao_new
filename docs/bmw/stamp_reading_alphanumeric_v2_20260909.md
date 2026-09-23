# BMW钢印读取V2：仅字母数字与整行补读

2026-09-09。用户明确确认编码只含字母和数字、没有标点，并要求修改。此版本仍只读取，不包含工单、日期、批次或产品合格规则。

## 修改内容

新配置：`configs/bmw/checks/stamp_read_left_v2.json`、`stamp_read_right_v2.json`。V1配置保留用于对照，未启用新功能时兼容原读取方式。

- `normalization_policy=ascii_alphanumeric`：接受A—Z、a—z、0—9。清除空白和标点，包括Unicode标点/空白，并记录各字符原位置和清理原因。
- `raw_code`和原始OCR行不被清理覆盖；正式供读取端使用的字段为`normalized_code`。例如`5A9D6B3:02`→`5A9D6B302`。
- 不做B/8、O/0、D/0替换，不改变大小写，不向模型提供目标编码。全角字母数字、其他文字和不支持的符号保留在原始证据中并提示复核，不能通过简单删除伪装成完整编码。
- 仅初次需要复核的图片执行补读：配置好的编码四边形校正为450×104整行图；原图与固定CLAHE增强图各做一次纯识别，跳过文字检测与方向分类。
- 两种处理结果规范化后必须一致，分数各至少0.90，并与初次检测的文字证据兼容，才能解除部分复核。原初筛0.95未被全局降低。
- 初次低分但完整的读数，必须与补读一致；初次缺少首尾的读数，补读必须提供严格更长且包含原读数的结果；多框必须不重叠、纵向对齐、合并跨度足够，且拼接文字与补读一致。重复相同残缺读数不能解除复核。
- 两种处理来自同一图像、同一识别器，不是两次独立曝光或独立模型证据。仍需以额外样本检验识别质量。

没有全局替换成直接整行识别：60张探索对照发现它会在部分原本读对的图上新增B/8、D/0混淆。证据为`artifacts/bmw_stamp_alphanumeric_20260909/direct_line_ab.json`，未把任何探索读数伪装成最终输出。

## 实测对照

| 样本 | V1需复核 | V2需复核 | 字母数字一致数（V1→V2） |
|---|---:|---:|---:|
| 原60张 | 10/60（16.7%） | 5/60（8.3%） | 55/58→57/58 |
| 额外24张新路径 | 3/24（12.5%） | 0/24 | 24/24→24/24 |

原60张有2张目视参考不确定（损伤/暗区），不纳入58张明确参考的分母，但保留全部推理和复核记录。V2初筛加补读共55张readable、5张review；readable中的规范化编码均与目视参考一致。57/58包括仍需复核但编码实际一致的个别样本，不等于57张自动通过筛选。

原60张解除复核的5张：13、14、27（低分但补读一致）、31（拆框）、42（漏尾部02）。仍复核：10（B/8分歧）、28（物理损伤）、46（重叠候选框）、51（D/0分歧）、56（遮挡及低分）。全部原始候选仍可查看，没有补写成预期字符。

额外24张排除此前全部68张路径，在代码方案确定后比较V1/V2；左右各12张，覆盖7个采集会话。助手独立逐图目视转录在运行模型前完成，24张均可读。输入清单为`docs/bmw/stamp_reading_holdout_20260909_manifest.json`。参考不是客户真值；不同路径仍可能是同一工件重复拍摄，且仍主要只有两种编码，不能推断新型号或现场准确率为100%。

最终60张OCR调用中位123.9ms、P95 149.1ms；额外24张中位129.1ms。包含触发的补读调用，不包含模型加载、ROI校正/增强、读写、采集和保存；功能回放时还存在其他任务，非正式节拍基准。

## 单图使用

当前独立临时环境已准备，使用V2配置：

```bash
PYTHONPATH=src /tmp/bmw-stamp-ocr-env/bin/python -m bmw_inspection.cli.read_stamp \
  --config configs/bmw/checks/stamp_read_left_v2.json \
  --image dataset/bmw_lab_raw_clean_0820/left/back/normal/20260820_162040_682498/images/left_back_normal_bmw_left_normal_group002_000001_fused.png \
  --output artifacts/stamp_single_v2
```

示例规范化编码为`5A9D6B302`。右件改用右V2配置和右back原图。输出目录须不存在，避免覆盖先前证据。环境重建步骤和固定依赖见`bmw_runtime/requirements-stamp-ocr.txt`及[V1使用说明](stamp_reading_validation_20260909.md)。未改根/BMW运行环境依赖或模型资产。

主要输出字段：

- `normalized_code`、`raw_code`、`removed_characters`、`unsupported_characters`。
- `state`、`reasons`、`selected_source`、`code_score_source`、`effective_code_geometry`。
- 触发补读时，`primary_reading`保留初次判定，`rectified_reading`记录两次补读、变换矩阵、分数和清理日志。

常规ROI与框图照常保存；补读另存`line_original.png`和`line_clahe.png`。`lines`和`code_width_fraction`始终属于初次检测，最终选中字形区域看`effective_code_geometry`，最终分数来源看`code_score_source`。CLI同时打印规范化与原始读数，退出码仍为0/readable、3/review、2/输入配置错误；不代表工件放行。

## 本地证据与验证

- 原60张：`artifacts/bmw_stamp_alphanumeric_20260909/final_original60/review.html`、`results.json`、`summary.json`和各样本证据。
- 新24张：同目录下`final_new24/`；V1对照为`v1_new24/`。
- 真实CLI覆盖标点清理、拆框补读、遮挡继续复核，记录在`cli_smoke.json`。

复跑命令（输出目录选择新路径）：

```bash
PYTHONPATH=src /tmp/bmw-stamp-ocr-env/bin/python tools/bmw/validate_stamp_reading.py \
  --config-version v2 \
  --manifest docs/bmw/stamp_reading_holdout_20260909_manifest.json \
  --output artifacts/bmw_stamp_v2_replay
```

新增/现有读取专项测试65项，覆盖原文清理日志、Unicode标点、字母数字不替换、空编码、不支持字符、补读一致/冲突、缺字严格扩展、多框重叠和跨度、原始证据保留等。与BMW全部单元测试及兼容入口一起实跑 **212 passed，1条既有NVML警告**。日志`artifacts/bmw_stamp_alphanumeric_20260909/tests_final.log`。本轮无GPU/相机或业务规则验证。
