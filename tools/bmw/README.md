# BMW 离线工具

统一调用 `bmw_inspection.checks` 公共接口，使用独立 OCR 环境。

- `validate_stamp_reading.py`：按已有目视参考清单重放，保存逐图证据和复核 HTML。使用 `--config-version v2`。
- `benchmark_stamp_reading.py`：模型加载并预热后测量 CPU 延迟，支持线程数、重复次数和输入清单配置。

安装、完整命令和接口约定见 [钢印读取接口](../../docs/bmw/stamp_reader.md)。历史样本及性能实验保留在 `artifacts/` 中，不作为代码入口。


## 第1—10项离线算法对照

`run_algorithm_comparison.py` 复用冻结的 Template、EfficientAD、YOLO 以及 R01/R02a；仅运行已解析的 `development_only` 计划，不训练。使用 `uv run --no-sync python tools/bmw/run_algorithm_comparison.py --help` 查看入口；输出目录必须不存在。

首轮8图/6分支结果、复现命令、数据缺口及计时边界见 [E2开发回放报告](../../docs/bmw/algorithm_comparison_e2_20260912.md)。

## 轮廓与几何

- `measure_front_geometry.py`：通过公共轮廓 API 生成正面几何诊断报告，见 [接口说明](../../docs/bmw/front_contour_api.md)。
- `validate_contour_real_samples.py`：重放真实轮廓样本清单。
- `demo_contour_synthetic.py`：合成轮廓演示，不能替代真实零件验证。

## 孔位与成像诊断

- `validate_hole.py`、`measure_hole_samples.py`：单孔检查与样本测量，见 [单孔说明](../../docs/bmw/hole_single_b4.md)。
- `simulate_motion_blur.py`：运动模糊模拟。
- `subtract_reflection_template.py`：反光模板诊断实验。

各脚本参数以 `uv run --no-sync python tools/bmw/<脚本名> --help` 为准；历史实验所需图片、模型及报告留在本地资产目录。
