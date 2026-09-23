# BMW 开发导航

本仓库在 Anomalib 基础上维护 BMW 四相机、两轮八视图检测流程，并保留独立的离线检查和算法实验工具。

## 代码目录

| 目录 | 用途 |
| --- | --- |
| `src/bmw_inspection/capture/` | 相机采集、曝光实验与报告 |
| `src/bmw_inspection/checks/` | 钢印、孔位及轮廓检查 |
| `src/bmw_inspection/benchmark/` | 冻结基线和离线算法对照 |
| `src/bmw_inspection/cli/` | 命令行入口 |
| `configs/bmw/` | 运行、检查和实验配置 |
| `tools/bmw/` | 离线验证、计时和诊断脚本 |
| `tests/unit/bmw_inspection/` | BMW 单元测试和回归测试 |
| `bmw_runtime/` | 独立 uv 运行环境与 OCR 依赖清单 |

## 使用入口

- [项目背景与需求范围](project_background_20260915.md)
- [运行环境与外部资产](../../bmw_runtime/README.md)
- [正面轮廓 Python API](front_contour_api.md)
- [整圈轮廓检查](contour_compare.md)
- [单孔检查](hole_single_b4.md)
- [钢印读取接口](stamp_reader.md)
- [曝光与 HDR 对比](exposure_test.md)
- [离线工具目录](../../tools/bmw/README.md)
- [算法对照开发回放](algorithm_comparison_e2_20260912.md)
- [划痕泛化改进计划](scratch_generalization_plan_20260913.md)

在已安装测试依赖的开发环境中验证 BMW 模块：

```bash
uv run --no-sync python -m pytest tests/unit/bmw_inspection -q
```

新环境先按运行说明或对应模块文档安装依赖。OCR 使用独立依赖；相机采集需要 Hikvision MVS。测试通过不等于相机、真实零件或生产验收通过；各离线模块的 REVIEW 和适用范围以接口文档为准。

## Git 与本地资产

Git 保存源码、测试、小型配置、文档及样本清单。原始图片、权重、数据集和生成报告位于 `.gitignore` 排除的本地目录；文档中的 `artifacts/`、`results/` 等链接需要站点资产才能访问。新克隆不能直接复现所有历史实验。

根目录 `AGENTS_MEMORY.md` 记录仓库历史；模块内同名文件记录各自上下文。修改前参考对应目录记忆，修改后同步更新，避免将不同模块的结论混用。
