# BMW Right 可复用八视图 ROI 设计

## 目标

从 `dataset/bmw_lab_raw/right` 已采集的完整八视图缺陷件中选择一组固定零件 ROI，并把 ROI 保存为与
固定相机、光源、工装及 `4024×3036` 分辨率绑定的可复用配置。以后新增正常件和无光痕件时，不重新框选
ROI，只重新准备和物化数据。

## 选择方案

采用向后兼容的 ROI schema-v2 `fixed_setup` 配置。schema-v1 继续严格绑定某个 prepared manifest；
schema-v2 记录参考采集 manifest、参考 sample、采集范围和图像尺寸，但允许应用到同一固定成像条件下的
后续 prepared release。

不采用每次新增数据后重写 manifest 绑定，因为这会制造不必要的配置修改；也不等待正常件采集完成，因为
当前缺陷数据已经足以框选覆盖零件的几何区域。

## 数据流

1. ROI 选择器读取公共 `dataset/bmw_lab_raw/manifests`，仅保留源图位于 `right` 范围的完整八视图 sample。
2. 操作者可用 `--source-class` 或 `--sample-id` 指定参考件，逐视图在缩放窗口中框选 ROI。
3. 选择器原子写入独立的 `configs/bmw/rois/bmw_right_hdr_eight_view_v1.json`。
4. 后续数据准备器使用 `--hand right`，排除原有 `left` 零件数据。
5. 训练数据物化器允许 schema-v2 ROI 用于新的 prepared dataset，但仍逐张校验源图 SHA-256、尺寸和 ROI 边界。

## 兼容与安全边界

- 旧 schema-v1 ROI 的 prepared dataset 路径及 manifest SHA-256 校验保持不变。
- schema-v2 只在 `binding_mode=fixed_setup`、八视图齐全、分辨率完全相同时复用，不做坐标缩放。
- 默认不覆盖 ROI 配置；只有显式 `--force` 才能替换。
- 本次不训练模型、不生成 YOLO 框、不覆盖旧 prepared/training release。

## 验证

- 单元测试覆盖 right 范围过滤、raw 参考 sample 选择、schema-v2 往返读写、旧 schema-v1 严格绑定，以及
  schema-v2 跨 prepared release 物化。
- 对真实 raw 数据执行只读选择预检，确认得到 30 个完整 right sample、八视图尺寸一致。
- CLI `--help` 和聚焦 BMW 测试通过后，交付一次 GUI 框选命令。
