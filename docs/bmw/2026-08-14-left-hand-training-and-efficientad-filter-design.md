# BMW 左手训练与 EfficientAD 微小误报抑制设计

## 目标

在左手 normal 数据采集结束后，提供可立即运行的实验室工具链，完成数据准备、八视图 ROI、训练数据物化、Template/EfficientAD 训练、EfficientAD mask、阈值标定，以及后续光痕倾斜 ROI 与 V3 重训练。YOLO 复用现有左右手联合检查点，不重新训练。

## 范围

- 左手 normal 数据按 capture scope 和会话生成独立 prepared release。
- Template、EfficientAD 和 YOLO 现场输入共用一套左手八视图零件 ROI。
- EfficientAD 使用左手独立的八视图 ignore mask。
- Template 与 EfficientAD 使用左手 normal 数据训练；阈值不得复制右手数值。
- 光痕独立使用 `front_left` HDR 原图和四点倾斜 ROI，等待 no_streak 数据后训练。
- 不训练 YOLO；最终候选记录复用检查点路径和 SHA-256。

## 最短工作流

1. `prepare`：显式接收左手 normal session，生成不可覆盖 prepared release。
2. `select-roi`：从 prepared release 的完整代表件选择八视图 ROI，资产绑定 manifest SHA 和 `capture_scope=left`。
3. `materialize`：生成 Template、EfficientAD 和运行所需 ROI crops；拒绝跨手 ROI。
4. `train`：训练八视图 Template 和 EfficientAD，不包含 YOLO、光痕。
5. `select-mask`：直接从 training release 的代表件 ROI 空白创建八视图 mask；允许显式加载同一 ROI 资产绑定的旧 mask 续画。
6. `calibrate-efficientad`：用左手 normal calibration 数据、左手 mask 和最终连通域评分重打分并生成阈值资产。
7. no_streak 采集完成后，运行倾斜光痕 ROI 选择器和 V3 重训练器。

## Template normal-only 候选

从左手 train split 选择模板，使用左手 calibration normal 风险分数拟合逐视角阈值。产物必须标记 `candidate_only=true`、`defect_metrics=not_evaluated`，并绑定八个 `model.json` SHA。不得使用右手 Template 阈值。

## EfficientAD 连通域评分

当前运行时的 mask 外 anomaly-map 单像素最大值会被孤立热点触发，并且与训练后的 `pred_score` 标定域不一致。新评分在标定和运行时共用同一个纯函数：

1. 将 ignore mask 以最近邻方式缩放到 anomaly map；mask 区域不参与评分。
2. 使用低阈值生成 8 连通域，并要求连通域内部至少包含一个高阈值种子。
3. 对每个连通域计算面积、峰值、均值、P95 和包围框长边。
4. 满足任一条件的连通域有效：
   - 面积达到下限且 P95 达到下限；
   - 峰值达到强点阈值，保留小而深的缺陷；
   - 长边和面积达到细线门槛，保留细长划痕。
5. 最终分数是有效连通域的最大 P95；无有效连通域时为 0。

过滤参数按视角写入 SHA 绑定资产。第一版用左手 normal calibration map 做小规模确定性候选搜索，以整件任一视角 NG 的规则拟合不超过 5% 的目标误报率；样本数不足以量化 5% 时明确记录分辨率和 `candidate_only`，不伪称独立验收。

## 证据与界面

EfficientAD 仍保存原始最大值作为诊断，但不参与判定。结果新增有效/拒绝连通域数量、面积、峰值、均值、P95、包围框和过滤原因；overlay 画有效连通域轮廓和框。小且浅的孤立区域应为 PASS，小而强、细而长或成片异常应为 NG。

## 契约与失败处理

- prepared capture scope、ROI capture scope 必须均为 `left`。
- ROI、mask、Template 模型、EfficientAD checkpoint、阈值资产和共享 YOLO 均记录 SHA-256。
- 输出目录存在时拒绝覆盖；允许验证完全相同的已完成产物后复用。
- mask 标定与 Demo 必须使用完全相同的评分策略和参数。
- 联合 YOLO 检查点 SHA 必须保持 `0e9591f2fa2487ad12000847f1d80137901ed69989e0e95cb5c8907b95ba3913`。

## 验收

- 跨手 ROI/mask 被拒绝。
- mask 可从左手 training ROI 空白新建。
- 单个浅热点 PASS；小而强的点、细长划痕和较大区域 NG。
- calibration 与 runtime 对同一 anomaly map 得到相同分数和连通域。
- Template 与 EfficientAD 阈值资产绑定左手模型 SHA。
- 左手训练计划和执行记录中没有 YOLO 训练阶段，共享 YOLO SHA 不变。
- 聚焦测试、编译检查、diff check 通过；无真实完整左手数据时只做 dry-run，不声称模型已训练。
