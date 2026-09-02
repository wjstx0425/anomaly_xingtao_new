# BMW 左右手八视图 40 张 Template 候选审核设计

日期：2026-08-23

本设计取代同目录的 20 张版本。目标是在不修改活动检测配置的前提下，从左右手 0823 正常训练数据中为每个标准视角选择 40 张 Template 候选，先由操作者人工审核，再进入后续训练。

## 数据与数量

- 右手 manifest：`dataset/bmw_lab_prepared/bmw_right_front_right_0823_v1/manifests/dataset_manifest.csv`
- 左手 manifest：`dataset/bmw_lab_prepared/bmw_left_front_right_0823_v1/manifests/dataset_manifest.csv`
- ROI：分别使用 `configs/bmw/rois/bmw_right_0820_v1.json` 与 `configs/bmw/rois/bmw_left_0820_v1.json`
- 只允许 `split=train`、`source_class=normal`、`business_label=OK`。
- 左右手分别维护，全部八视角，每视角 40 张，共 640 张。
- 候选身份使用 `session_id + sample_id + view_id`，避免左手两个 session 的同名样本混淆。

右手 train 每视角有 52 个不同物理件。左手 train 每视角有 52 张图、28 个不同物理件，来自两个 session。左手先覆盖不同物理件，再允许第二 session 的重复物理件进入候选，从而选满 40 张。

## 选择和审核

候选选择沿用 Template 的确定性多样性思想：先选接近中心的样本，再做最远点选择；在仍有未覆盖物理件时，只从新物理件中选。所有候选均裁为当前公共 ROI 的彩色 PNG。

输出到新目录：

```text
dataset/bmw_lab_labeling/bmw_template_40_review_0823_v1/
├── right/<view>/candidate_01__<session>__<sample>.png
├── right/<view>/...candidate_40...
├── right/<view>/contact_sheet.jpg
├── left/<view>/...
├── candidate_manifest.csv
└── review_instructions.txt
```

每视角 contact sheet 使用 5×8 布局并标出候选编号。审核方式是直接删除不合格的 `candidate_*.png` 副本。删除后不自动补齐，编号空缺允许存在；原图不删除、不修改。

## 后续训练边界

本轮只生成审核包，不训练、不替换模型、不修改左右活动配置。人工审核完成后，训练入口只读取 manifest 中登记且候选副本仍存在的图片；每视角至少保留 3 张、最多 40 张。新模型必须使用新目录，旧 5 张模板模型和旧阈值继续保留以便回滚。

模板数量改变后，普通 Template 阈值与权重 3.0 的加权 Template 阈值都必须用 calibration normal 重新标定；final_test 只做报告。审核、训练和接入均不增加 SHA、receipt、publisher、provenance、rebind 或不可变发布逻辑。

## 本轮验收

1. 每手每视角恰好 40 张可读候选，共 640 张。
2. 候选均来自对应 0823 manifest 的 train/normal/OK 行。
3. ROI 坐标合法，候选尺寸与对应公共 ROI 一致。
4. 左手候选能够区分两个 session，并优先覆盖不同物理件。
5. 重复输入得到相同候选顺序。
6. 活动配置、现有模型和阈值不变。

