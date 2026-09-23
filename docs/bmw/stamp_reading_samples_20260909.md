# BMW back 钢印读取小样实验（2026-09-09）

本轮按用户最新范围只读取钢印，不实现19—22项规则，不接主运行时。

## 输入与方法

- 数据：`dataset/bmw_lab_raw_clean_0820/{left,right}/back`；各侧normal与defect分别按路径排序选前两张，共8张，均为group001/group002。完整路径见 `artifacts/bmw_stamp_read_20260909/selected.json`。
- 输入均为4024×3036融合图；不是三个独立曝光。原始数据不变。
- 人工查看后配置原图半开xyxy ROI：左 `[1260,1450,1900,1980]`，右 `[1300,1100,1980,1830]`；裁剪后顺时针旋转90度。
- 使用成熟项目 [RapidOCR](https://github.com/RapidAI/RapidOCR)，本次固定 `rapidocr-onnxruntime==1.4.4`，CPU推理，模型随包提供。独立临时环境 `/tmp/bmw-stamp-ocr-env`，没有修改BMW依赖或锁文件，也没有上传图片。
- 仅原图裁剪和旋转，没有增强候选搜索、训练、工单提示、预期字符串纠正。按文字框纵坐标选择最上方编码行，保留所有原始OCR行。

## 原始算法输出

| index | 图像类别 | 编码原始输出 |
|---|---|---|
| 0 | left normal group001 | `5A9D6B3.02` |
| 1 | left normal group002 | `5A9D6B3 02` |
| 2 | left defect group001 | `5A9D6B3 02` |
| 3 | left defect group002 | `5A9D6B3 02` |
| 4 | right normal group001 | `5A9D6B4 02` |
| 5 | right normal group002 | `5A9D6B4 02` |
| 6 | right defect group001 | `5A9D6B4 02` |
| 7 | right defect group002 | `5A9D6B4 02` |

独立人工视觉转录：左4张为 `5A9D6B3 02`，右4张为 `5A9D6B4 02`，并非企业工单真值。8张编码的字母和数字与该人工查看一致，但第0张间隔被读成点号，完整字符串不能报8/8一致。旁边BMW标识在第4张被读成BMAN，其余7张为BMW；LH/RH读取与画面一致。未修改这些错误。

复跑 `run_01` 的每张裁剪图OCR调用实测约111—168 ms，不含模型加载、文件读写、相机采集或完整检测周期；不是稳定节拍基准。引擎行级得分不是正确率。

这些normal/defect目录是整件类别，不能据此认定钢印存在缺陷。只测试了8张、两种编码，固定ROI尚未验证跨会话取放偏移，不代表全数据集或现场可靠率。

## 结果与复现

本地证据目录（Git忽略）：`artifacts/bmw_stamp_read_20260909/`。

- `read_samples.py`：独立可复跑实验脚本；`selected.json`：精确输入清单。
- `requirements.txt`：本次临时环境完整版本。
- `run_01/result.json`：原始文字、框、分数、坐标系、来源及耗时。
- `run_01/comparison.jpg`：8张结果对照；每张另存原始ROI、转正ROI、识别框叠加图。

从仓库根目录执行，输出目录必须尚不存在：

```bash
UV_CACHE_DIR=/tmp/bmw-uv-cache uv venv --python 3.13 /tmp/bmw-stamp-ocr-env
UV_CACHE_DIR=/tmp/bmw-uv-cache uv pip install --python /tmp/bmw-stamp-ocr-env/bin/python -r artifacts/bmw_stamp_read_20260909/requirements.txt
/tmp/bmw-stamp-ocr-env/bin/python artifacts/bmw_stamp_read_20260909/read_samples.py --output artifacts/bmw_stamp_read_20260909/run_02
```

现有临时环境还在时只需最后一条命令。脚本已实际跑完8张并打开检查对照图；没有运行BMW回归或相机，因为本次没有改动BMW运行代码。ONNX Runtime启动产生了遥测标识持久化警告，推理仍成功；意外生成的本地`:memory:.ses`移入证据目录保留，脚本同时禁用了运行时遥测事件。
