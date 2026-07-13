# ZS32 PatchCore 左右手六视角 ROI 数据集设计

## 目标与范围

新增 stage 30 工具，从 `dataset/right` 和 `dataset/left` 裁剪 PatchCore 使用的分类图片，降低零件外环境对异常分数的影响。左右手各自保存六个独立固定 ROI，共 12 个 ROI。工具不修改原始数据、不读取或变换 YOLO 标签，也不改变 stage 29 的 YOLO 职责。

默认输入为：

```text
dataset/right
dataset/left
```

默认输出为：

```text
dataset/zs32_patchcore_roi/
├── right/<view>/{normal,normal_test,defect}/...
└── left/<view>/{normal,normal_test,defect}/...
```

输出保留源数据在 hand 根目录下的标签、缺陷类型、采集 session 和 `images` 相对层级，供现有 `zs32_defect_workflow.py` 通过手侧输出根目录直接消费。

## 方案选择

采用独立 stage 30，而不是扩展 stage 29 或修改 PatchCore 预处理器：

- stage 29 继续只负责 manifest 驱动的 YOLO 图片和 bbox 标签同步变换；
- stage 30 只负责 PatchCore 分类图片树，不引入 YOLO 依赖；
- 裁剪结果一次生成、重复训练复用，避免每次 preprocess 重复裁图；
- 现有 PatchCore `--roi` 只能给一次运行中的所有视角使用同一个矩形，不能表达左右手 12 个独立 ROI，因此不在训练入口中硬塞映射逻辑。

## 文件与职责

新增核心模块：

```text
capture_data/zs32_patchcore_roi_dataset.py
```

职责包括：

- 枚举并验证 right/left 六视角图片；
- 选择每个 hand/view 的正常参考图；
- 加载和严格校验 12-ROI 配置；
- 完整预检源图片、尺寸、视角身份、目标冲突与输出安全性；
- 执行裁剪、写出 manifest 和配置快照；
- 生成统计信息与错误诊断。

新增薄 CLI 包装：

```text
pipeline/30_crop_zs32_patchcore_dataset.py
```

CLI 只负责解析参数、调用核心函数并打印结果，不重复实现裁剪逻辑。

## CLI 设计

### 选择 ROI

```bash
uv run --no-sync python pipeline/30_crop_zs32_patchcore_dataset.py select \
  --dataset-root dataset \
  --config dataset/zs32_patchcore_roi_config.json \
  --preview-dir dataset/zs32_patchcore_roi_previews
```

`select` 固定按 `right` 六视角、再 `left` 六视角的顺序打开 OpenCV 窗口。每个 hand/view 使用第一张可读取的正常图片作为参考；已有合法配置时使用原 ROI 作为初始框。每次选择都写一张带框预览图，但只有 12 个 ROI 全部成功后才通过临时文件原子替换正式配置。

### 转换数据

```bash
uv run --no-sync python pipeline/30_crop_zs32_patchcore_dataset.py convert \
  --dataset-root dataset \
  --config dataset/zs32_patchcore_roi_config.json \
  --output-root dataset/zs32_patchcore_roi
```

默认不允许输出目录已存在。只有显式传入 `--overwrite` 才允许在完整预检成功后重建目标目录。输入与输出必须是仓库 `dataset/` 下彼此分离的安全子目录，禁止输出等于输入、包含输入或落在输入内部。

## ROI 配置

配置使用像素半开区间 `xyxy`：

```json
{
  "schema_version": 1,
  "coordinate_system": "pixel_xyxy_half_open",
  "image_size": {"width": 4024, "height": 3036},
  "hands": {
    "right": {
      "views": {
        "front": {"roi": [150, 1020, 3910, 2520], "reference_image": "..."}
      }
    },
    "left": {
      "views": {
        "front": {"roi": [0, 0, 100, 100], "reference_image": "..."}
      }
    }
  }
}
```

配置必须恰好包含 `right`、`left`，每只手必须包含：

```text
front, front_left, front_right, back, back_left, back_right
```

每个 ROI 必须是四个整数，并满足 `0 <= x1 < x2 <= width`、`0 <= y1 < y2 <= height`。所有源图片必须与配置的原图尺寸一致；不同尺寸时 fail closed，不进行隐式缩放。

## 数据发现与视角纠正

支持扩展名 `.bmp`、`.jpeg`、`.jpg`、`.png`、`.tif`、`.tiff`，按稳定排序递归发现：

```text
<dataset-root>/<hand>/<view>/normal/**/*
<dataset-root>/<hand>/<view>/normal_test/**/*
<dataset-root>/<hand>/<view>/defect/**/*
```

文件名必须以 `<hand>_<view>_` 开头，其中 view 使用最长匹配，避免把 `front_left` 误判为 `front`。当文件名前缀的 hand 与输入 hand 不一致时直接报错；当文件名 view 与父目录 view 不一致时，以文件名 view 为准写入对应输出目录，并在 manifest 中记录 `source_view`、`resolved_view`、`view_corrected=true`。

该规则用于纠正 `dataset/right` 已发现的 24 张 deform 图片 front/back 错目录问题，不修改源文件。若两个源文件纠正后映射到同一个目标路径，预检报错，不覆盖任何文件。

## 输出与 manifest

每张图片按对应 hand/view ROI 裁剪并保留原格式和源 hand 根目录下的剩余相对层级。输出 manifest 位于：

```text
dataset/zs32_patchcore_roi/crop_manifest.csv
```

至少包含：

```text
source_path, output_path, hand, source_view, resolved_view,
view_corrected, label, defect_type, session_id,
roi_x1, roi_y1, roi_x2, roi_y2,
source_width, source_height, crop_width, crop_height
```

输出根目录同时保存实际使用的 `roi_config.json` 快照和 `summary.json`。summary 按 hand/view/label 统计输入数、输出数和纠正视角数。转换成功条件是预检图片数与实际写出数一致；任何写入失败均以非零状态退出，不宣称数据集完成。

## 错误处理与覆盖语义

- 缺少 hand、视角或正常参考图：报错；
- 图片不可读、尺寸不同、ROI 越界：报错；
- 文件名无法解析 hand/view 或 hand 冲突：报错；
- 目标路径冲突：报错；
- 输出已存在且未指定 `--overwrite`：报错；
- `--overwrite` 只在所有源图片和目标映射预检成功后生效；
- 不修改 `dataset/right`、`dataset/left`、stage 29 配置或任何 YOLO 数据。

## 测试与验收

新增 `tests/unit/capture_data/test_zs32_patchcore_roi_dataset.py`，覆盖：

- stage 30 parser 默认值和两个子命令；
- 左右手各六个独立 ROI；
- 正常、normal_test 和多级 defect 目录保持；
- 不同 hand/view 使用不同裁剪尺寸和内容；
- 文件名前缀最长匹配与 24 张同类错目录纠正规则；
- hand 前缀冲突、未知视角、图片尺寸不一致和目标冲突 fail closed；
- 配置缺 hand/view、非整数或越界 ROI；
- 输出保护与显式 overwrite；
- manifest、配置快照、summary 计数完整；
- 转换前失败不会创建或删除输出。

实现后运行定向 pytest、`py_compile`、Ruff F/I、CLI `--help` 和一个 `/tmp` 合成数据烟雾测试。真实 `dataset/right`、`dataset/left` 的裁剪需要先由用户完成 12 个 ROI 的交互选择；在 ROI 配置未完成前不猜测左手坐标。

## 最小可见进度（2026-07-13 补充）

`convert` 复用仓库已有的 `rich.progress.track` 包裹图片裁剪循环，显示 `Cropping images`、完成数量和进度条。不增加依赖、不新增命令行参数，也不改变输出结构或裁剪语义。
