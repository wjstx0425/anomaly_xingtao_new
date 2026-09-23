# BMW钢印读取CPU线程速度对照（2026-09-10）

用户询问能否在保持准确率的情况下提速。本轮只进行同模型实验，没有修改读取器默认2线程、V2图像处理、阈值或模型文件。

本机CPU：Intel Core Ultra 7 265K，20逻辑CPU。使用既有独立环境`/tmp/bmw-stamp-ocr-env`、RapidOCR ONNX Runtime 1.4.4及V2配置。ONNX Runtime支持调整算子内部并行线程，见[官方说明](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)。

历史84张的阶段中位耗时：检测92.15ms、方向分类1.81ms、识别28.26ms。模型已常驻；仅关闭方向分类最多针对约2ms，不能作为主要提速手段。全量整行直读此前出现B/8、D/0误读，不作为无损优化。

## 测试方法与结果

先用10张含难例的小样探索1/2/4/8线程；4线程最好。随后对原60张及额外24张共84张进行完整对照：每个配置先预热4张，再按正序/逆序各跑一遍，共168次计时。两个配置分别在独立进程中串行执行，没有并行运行其他性能测试；不是生产全模型混跑。

最终只改变`intra_op_num_threads`从2到4，`inter_op_num_threads`保持2。模型、精度、配置、输入尺寸及全部处理保持相同。引擎通过现有backend注入接口调用，未修改运行代码。通用Anomalib模型训练基准不适用于该独立OCR读取器，因此使用本实验专用脚本。

| 计时范围 | 2线程中位数 | 4线程中位数 | 2线程P95 | 4线程P95 |
|---|---:|---:|---:|---:|
| OCR调用（含触发的补读） | 123.61ms | 78.02ms | 143.81ms | 91.38ms |
| 完整`StampReader.read()` | 124.09ms | 78.51ms | 144.56ms | 92.06ms |

OCR中位延迟下降36.88%。完整read计时包含裁剪、校正/增强、推理和内存结果构建；两种计时均排除模型加载、输入文件读写、相机采集及结果保存。

逐项比较两配置168次输出，递归去除`elapsed_ms`和`backend_times`后，**其余payload完全一致**：包括原始/规范化文字、框、分数、状态、原因、补读选择和变换信息。没有观察到样本结果变化；这不是对任意未来图片的准确率保证，也不是整机含25项检测同时执行的节拍验收。

建议优先采用4线程作为本机候选；其他机器与整机并发场景应重新测资源竞争。默认读取器仍为2线程，本轮没有上线此参数。

## 证据与复现

- CSV：`results/bmw_stamp_speed_20260910/cpu_threads.csv`。
- 完整输出和比对：`artifacts/bmw_stamp_speed_20260910/threads2.json`、`threads4.json`、`comparison.json`。
- 小样探索：`thread_probe.json`，其样本/计时口径不同，不与完整84张均值混合。
- 专用脚本：`artifacts/bmw_stamp_speed_20260910/benchmark_threads.py`，输入清单复用现有两份manifest。输出目录应使用新文件名。

```bash
PYTHONPATH=src /tmp/bmw-stamp-ocr-env/bin/python artifacts/bmw_stamp_speed_20260910/benchmark_threads.py \
  --threads 4 --output artifacts/bmw_stamp_speed_20260910/threads4_repeat.json
```

本轮无生产代码变更，因此没有重新运行全套功能测试。ONNX Runtime既有遥测标识持久化警告未阻断CPU推理；生成的sidecar归入实验目录。没有GPU、相机或主检测流程集成验证。

## 后续整理

用户确认后，V2 配置及读取器默认采用已对照的 intra4/inter2。稳定接口及可复用测速入口见 [钢印读取接口](stamp_reader.md)；上文保留参数实验当时的测量记录。
