# ZS32 六视角 YOLO Label Studio 本地标注目录设计

## 目标

为 `dataset/left` 和 `dataset/right` 中现有的 ZS32 六视角缺陷图片准备一个可由 Label Studio Local Files 直接读取的标注目录。标注项目只使用一个矩形框类别 `defect`，保留左右手、视角、缺陷类型和物理工件组元数据。不得复制、移动或修改原始图片。

当前输入共包含 `660` 张缺陷 PNG，来自 `110` 个物理缺陷工件组，每组有六个完整视角。正常图片不需要人工画框，因此不进入待标注目录。

## 方案选择

采用 Label Studio Local Files Source Storage，不通过浏览器上传图片。待标注图片使用硬链接引用原始 PNG：硬链接对 Label Studio 表现为普通文件，不额外占用一份图片数据块，也不依赖 Label Studio 是否跟随符号链接。

硬链接只用于只读标注输入。准备工具不得编辑图片内容；删除输出目录中的链接不会删除原始路径。若输出目录与原始数据不在同一文件系统、无法创建硬链接，工具应明确失败，不得静默退化成完整图片复制。

## 输出结构

默认输出根目录：

```text
dataset/zs32_yolo_labeling/
├── images/
│   ├── left/
│   │   ├── front/{deform,less,others}/*.png
│   │   ├── front_left/{deform,less,others}/*.png
│   │   ├── front_right/{deform,less,others}/*.png
│   │   ├── back/{deform,less,others}/*.png
│   │   ├── back_left/{deform,less,others}/*.png
│   │   └── back_right/{deform,less,others}/*.png
│   └── right/
│       └── <相同六视角结构>
├── labeling_manifest.csv
├── label_studio_config.xml
└── README.md
```

目录不保留采集 session 层级，避免 Label Studio 中路径过深；session 和 group 信息写入 manifest。现有文件名已经包含 hand、view、缺陷类型和 group，输出路径仍保持全局唯一。

## 准备工具

新增一个可重复运行的数据准备工具，从 `dataset/left` 和 `dataset/right` 发现路径结构为：

```text
<hand>/<view>/defect/<defect_type>/<session>/images/*.png
```

工具只接受以下值：

- hand：`left`、`right`；
- view：`front`、`front_left`、`front_right`、`back`、`back_left`、`back_right`；
- defect type：`deform`、`less`、`others`。

默认拒绝覆盖非空输出目录。显式要求重建时，只清理该工具自己的输出根目录，不接触输入目录。创建完成后验证每个链接可读、指向预期源文件、文件名无冲突，并检查总数和分组完整性。

## Manifest

`labeling_manifest.csv` 每张图一行，至少包含：

```text
labeling_path,source_path,hand,view,defect_type,session_id,group_id,sample_id
```

`source_path` 和 `labeling_path` 使用仓库根目录下的相对路径。`sample_id` 使用 `hand/session/defect_type/group_id` 的组合，保证不同 session 中重复的 `group001` 不会混淆。

manifest 用于后续将 Label Studio 导出的结果映射回原始数据，并确保 train、val、test 按物理工件组拆分，而不是按六视角图片随机拆分。

## Label Studio 配置

`label_studio_config.xml` 使用一个矩形框标签：

```xml
<View>
  <Image name="image" value="$image" zoom="true" rotateControl="true"/>
  <RectangleLabels name="label" toName="image">
    <Label value="defect" background="#E53935"/>
  </RectangleLabels>
</View>
```

启动 Label Studio 前设置：

```bash
conda activate label-studio
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/home/yunjing/anomalib/dataset/zs32_yolo_labeling
label-studio start
```

在项目中选择 `Settings -> Cloud Storage -> Add Source Storage -> Local Files`：

- Absolute local path：`/home/yunjing/anomalib/dataset/zs32_yolo_labeling/images`
- Import method：`Files`
- File Filter Regex：`.*\\.png$`

测试连接后同步，预期创建 `660` 个图片任务。项目标注界面粘贴 `label_studio_config.xml` 的内容。

## 标注语义

- 所有可见缺陷统一标为 `defect`。
- 每张图片独立判断；某个缺陷工件在某视角看不到缺陷时，该任务提交为空标注。
- 框住实际缺陷影响区域，不框整个零件。
- `deform`、`less`、`others` 只作为元数据和后续分类型统计，不作为 YOLO 类别。
- 不在标注阶段生成左右镜像；左右手真实缺陷图均保留并分别标注。

## 验证与验收

准备完成必须满足：

1. 输出共 `660` 个 PNG 硬链接，原始数据仍为 `660` 张缺陷图；
2. left 为 `486` 张，right 为 `174` 张；
3. 每个 hand、view、defect type 的计数与输入一致；
4. 每个物理缺陷工件组恰好包含六个视角；
5. manifest 恰好 `660` 行且路径全部存在；
6. 所有输出文件与对应源文件具有相同 inode，证明没有复制图片；
7. Label Studio 本地存储连接测试通过，同步后出现 `660` 个任务；
8. 随机打开左右手、六视角和三种缺陷类型的任务，图片均能显示并可画 `defect` 矩形框。

## 非目标

- 不修改 `capture_data/prepare_yolo_dataset.py` 的 YOLO 导出逻辑；
- 不自动生成缺陷框；
- 不把正常图片导入人工标注项目；
- 不在本阶段执行 train、val、test 拆分；
- 不启动或修改用户现有的 Label Studio 数据库。
