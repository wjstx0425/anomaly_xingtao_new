# ZS32 ROI 数据集转换进度条设计

## 目标

为 `pipeline/29_zs32_fixed_roi.py convert` 增加可见进度，让 2010 张 4K 图片的预检和裁剪写盘不再长时间无输出，同时保持现有命令、输出目录、裁框策略和统计结果不变。

## 方案

复用项目直接依赖的 `Rich`：

1. `_preflight()` 遍历 manifest 时显示 `Preflight` 进度条，总数为 manifest 行数。
2. `crop_zs32_yolo_dataset()` 写图片和标签时显示 `Cropping` 进度条，总数同样为 manifest 行数。
3. 进度条使用 Rich 标准展示，显示进度条、百分比和预计剩余时间；完成后保留最终状态。
4. 原命令不增加参数：

   ```bash
   uv run --no-sync python pipeline/29_zs32_fixed_roi.py convert --overwrite
   ```

## 错误处理

图片、标签、ROI 或写盘发生异常时，Rich 正常结束当前进度显示，随后保留原异常行为。进度条不吞掉错误，也不改变 `clipped_boxes`、`dropped_boxes` 等统计。

## 验证

- 单元测试替换进度迭代器，确认 `Preflight` 和 `Cropping` 各被调用一次且 `total` 等于 manifest 行数。
- 运行现有 stage-29 测试、Ruff 和编译检查。
- 使用小型测试数据集完成一次真实转换，避免为验证再次重写 11GB 正式数据集。
