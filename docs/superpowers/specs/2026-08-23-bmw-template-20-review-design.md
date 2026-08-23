# BMW 左右手八视图 20 张 Template 候选审核与重训设计

日期：2026-08-23

## 目标

把当前每视角 5 张 Template 扩展为“最多 20 张、先人工审核、再训练”的实验室流程。

- 右手和左手分别维护。
- 每只手覆盖全部 8 个标准视角。
- 每个视角先选择 20 张候选，共 320 张。
- 操作者审核候选副本并删除问题图片。
- 被删除的候选不自动补齐。
- 审核完成前不训练、不替换模型、不修改活动配置。
- 审核后用各视角剩余的候选训练新 Template 模型。
- 当前 5 张模板、当前阈值和活动配置完整保留以便回滚。

本设计不修改 EfficientAD、YOLO、光痕、公共 ROI、关键区域权重、HDR、相机参数或最终融合规则。

## 当前约束

当前 src/bmw_inspection/lab/template.py 将 template_count 限制在 3 到 5，八视图训练入口也固定传入 5。因此需要把 Template 内核和专用训练入口的上限扩展至 20，但不能让其他算法或 Demo 分支发生变化。

当前活动模型中，左右手 front_right 使用 0823 重拍模型，其他七个视角仍使用 0820 模型。本次候选审核统一从 0823 重拍 prepared manifest 选择全部八视角，使候选与调整后的相机机位一致。

## 方案选择

采用“审核目录中默认保留，删除副本表示拒绝”的最小流程。

不采用：

1. CSV 逐行填写 approved/rejected：状态明确，但 320 行人工填写较慢。
2. 新建图形审核器：交互更好，但会增加本次不必要的 UI 和状态管理代码。

## 数据来源

右手：
/home/yunjing/anomaly_xingtao_new/dataset/bmw_lab_prepared/bmw_right_front_right_0823_v1/manifests/dataset_manifest.csv

左手：
/home/yunjing/anomaly_xingtao_new/dataset/bmw_lab_prepared/bmw_left_front_right_0823_v1/manifests/dataset_manifest.csv

候选选择只允许：

- source_class=normal
- business_label=OK
- split=train
- 八个标准 view_id
- 文件存在且 OpenCV 可读取
- 公共 ROI 裁剪不越界

候选选择不读取 calibration 或 final_test 图像。calibration 只用于训练后的阈值标定，final_test 只用于报告。

## 候选选择

沿用当前 Template 的多样性选择思路，在每个视角的正常 train crop 中选择 20 张代表图，而不是简单取前 20 张。

左手 0823 数据包含两个采集 session 中重复的 group/sample ID。候选身份使用 session_id + sample_id + view_id，并优先保证不同 physical_part_id 的覆盖：

1. 首轮每个 physical part 最多选择一张。
2. 只有某视角独立 physical part 少于 20 时，才允许同一 physical part 的第二个 session 图进入候选。
3. 本次左手 train 中有足够的不同 physical part，因此目标是 20 个不同 physical part。

相同 manifest、ROI 和参数必须得到相同候选顺序。

## 审核包布局

生成新目录，不覆盖现有模型、训练数据或历史审核包：

    dataset/bmw_lab_labeling/bmw_template_20_review_0823_v1/
    ├── right/<view>/candidate_01__<session>__<sample>.png
    ├── right/<view>/...candidate_20...
    ├── right/<view>/contact_sheet.jpg
    ├── left/<view>/...
    ├── candidate_manifest.csv
    └── review_instructions.txt

每张 candidate PNG 是公共 ROI 彩色裁图，便于检查姿态、杂物、污渍和曝光。文件名包含候选编号、session 和 sample，避免左手重复 sample ID 混淆。

每个视角生成一张 4×5 总览拼图并标注编号。拼图只用于快速总览，训练不读取。

candidate_manifest.csv 至少包含 hand、view_id、candidate_index、session_id、sample_id、physical_part_id、source_path、review_image_path 和 selected_rank。不包含 SHA、receipt、publisher、provenance 或不可变发布字段。

## 人工审核合同

初始 320 张 candidate PNG 默认均为保留。

- 操作者直接删除有问题的 candidate PNG 副本。
- 不删除 contact_sheet、CSV 或原始数据。
- 不自动补充候选。
- 删除后的空号允许存在。
- 每个视角至少保留 3 张；少于 3 张时训练明确报错并停止。

审核完成后，训练入口以“CSV 中登记且审核图片仍存在”为批准集合。CSV 存在但审核图片已删除表示拒绝。

## 训练

审核完成后创建新模型目录，不覆盖当前模型：

    results/bmw_lab_one_click/bmw_right_template_20_reviewed_0823_v1/
    results/bmw_lab_one_click/bmw_left_template_20_reviewed_0823_v1/

每个视角使用审核后剩余的 3–20 张候选作为最终模板集合。训练不再从更大的 train pool 二次选择，避免审核集合与模型实际模板不一致。

每个新 model.json 的 templates 逐项记录审核候选路径、session_id、sample_id、physical_part_id 和模型内模板路径。

Template 预处理继续使用灰度、等比例缩放、反射 padding、3×3 Gaussian blur、512×512 目标尺寸和 max_shift=12。

## 阈值标定

增加模板数量会改变最大相似度和风险分布，因此不能直接套用当前数值阈值。

1. 使用对应 0823 calibration normal 计算普通 Template risk。
2. 使用相同 calibration 图像计算当前关键区域权重 3.0 下的 weighted risk。
3. 普通和加权阈值分别按 calibration-normal 最大值增加 10% 实验室余量。
4. final_test normal 只报告正常误拒，不参与阈值选择。
5. 报告每视角保留模板数、calibration 最大值、新阈值、final-test 最大值和误拒数。

当前活动配置中的旧模型路径、template.thresholds 和 template.weighted_regions.thresholds 在候选审核和训练期间不修改。

## 接入

训练和离线回放完成后，先创建新的左右候选 Demo 配置；只有操作者明确同意才修改活动配置。接入时必须同时切换八个 model.json、新普通阈值和新加权阈值，不得只换模型而沿用旧阈值。

接入前至少执行：

- 左右各一次 0823 normal 离线回放
- 固定 25 项、0 ERROR 检查
- Template 普通与加权 details 检查
- final-test normal 误拒汇总
- 当前确认形变样本的 A/B 比较

可信 OK、EfficientAD、YOLO、光痕和融合不变。

## 实现边界

仅实现：

1. Template 数量校验从 3–5 扩展为 3–20。
2. 候选审核包生成命令。
3. 从审核后剩余候选直接写模型的专用训练路径。
4. 普通及加权阈值标定。
5. 聚焦测试和 AGENTS_MEMORY.md 更新。

不增加自动补齐、审核 GUI、schema 版本、SHA、receipt/provenance、publisher/rebind/composition、模型身份绑定、数据删除或旧模型覆盖。

## 测试与验收

1. 每只手、每视角生成 20 张候选，合计 320 张。
2. 候选全部来自 0823 normal train，公共 ROI 坐标正确。
3. 左手优先选择 20 个不同 physical part，不因重复 sample ID 混淆 session。
4. 相同输入重复执行时候选顺序一致。
5. 删除候选后不自动补齐。
6. 保留 3–20 张时，model.json 模板数与实际保留数完全一致。
7. 少于 3 张时明确拒绝训练。
8. calibration/final_test 不参与模板候选选择。
9. final_test 变化不影响普通和加权阈值。
10. 当前活动配置和模型在人工审核完成前完全不变。
11. 新模型接入后仍固定 25 项，其他三算法及光痕结果不变。

