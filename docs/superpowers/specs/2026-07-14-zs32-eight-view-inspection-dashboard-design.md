# ZS32 八视角检测结果看板设计

## 目标

为单次右手 ZS32 检测开发一个最小 OpenCV 桌面看板。界面可以启动一次真机检测，也可以从命令行加载一个离线结果目录；同屏以 4×2 网格展示同一次四相机采集得到的八张图片，并且只叠加运行链路真实产生的缺陷证据。

第一阶段保持现有六视角模板、PatchCore、YOLO 和 Stage18 检测链路。`front_secondary` 与 `back_secondary` 参与采集和展示，但在对应权重及标定资产准备好之前，界面明确显示为“暂未接入模型”。

## 已确认范围

- 产品固定为 `ZS32`，hand 固定为 `right`。
- 八视角显示顺序固定为：
  1. `front`
  2. `front_left`
  3. `front_right`
  4. `front_secondary`
  5. `back`
  6. `back_left`
  7. `back_right`
  8. `back_secondary`
- 使用纯 OpenCV 窗口，不引入 Tkinter、Qt、浏览器界面或 Web 服务。
- 同时支持鼠标按钮和对应键盘快捷键。
- 只保留一个上下文相关的检测按钮、证据层切换和“退出”，不提供历史结果浏览或打开结果目录功能。该按钮依次承担“开始检测”“确认正面并拍摄”“确认背面并拍摄”，不增加其他采集按钮。
- 离线模式仅通过 `--result-dir` 接收一个结果目录，不在窗口内实现文件选择器。
- 真机模式通过 `--live --part-id <id>` 启动，Stage35 在后台子进程中运行。
- 该界面用于 commissioning 和调试展示，不承担生产放行职责。

## 现有系统结论

旧入口 `pipeline/5_demo_inspection.py` 调用的 OpenCV 看板一次显示一个 active face 和六个零件 crop 槽，并未定义 ZS32 八相机视角。可复用的部分是图片缩放、坐标映射、OpenCV 绘制，以及 Anomalib artifact 到缺陷区域的数据流。

当前 Stage35 真实样例只有六个 canonical source views。diagnostic 结果包含六张 PatchCore heatmap overlay 和六张 YOLO detection overlay。现有 PatchCore evidence 已经是 JET 着色并与 crop 混合后的图片，无法还原原始 anomaly map 或可靠二值 mask。完整 runtime 样例在 `front_left` 得到 `NG_TEMPLATE` 后短路，因此没有运行 PatchCore、YOLO 和 Stage18。

已提交的四相机 topology 将新增视角定义为 `front_secondary` 和 `back_secondary`，二者分别来自相机 `DB0968108` 在翻面前后的图像。本机已有真实八视角采集样例，但 secondary 暂无 ROI、模板、PatchCore checkpoint、阈值或经过验证的 YOLO 资产。

## 架构

新增独立 OpenCV 入口：

```text
pipeline/36_zs32_inspection_dashboard.py
```

窗口事件循环不直接执行相机采集或 GPU 推理。

```text
OpenCV 看板
  |-- 离线：解析一个 --result-dir
  `-- 真机：启动四相机 Stage35 子进程
                 |
                 |-- 同一 sample 采集八视角
                 |-- 六个主视角运行现有模型
                 |-- 两个 secondary 保留原图
                 `-- 原子发布进度和结果产物
                                  |
                                  `-- 看板解析并绘制
```

实现拆成四个边界清晰的单元：

1. **结果解析器**：校验身份，将 JSON、CSV 和图像产物转换为一个八视角结果对象。
2. **证据合成器**：把 crop 坐标中的 mask 和 detection 映射回源图，生成 Original、PatchCore、YOLO、Template 与 Fusion 层。
3. **OpenCV 看板**：绘制顶部信息、八张卡片、控制区、单卡放大和错误状态，并负责鼠标命中与快捷键。
4. **真机控制器**：只管理自己启动的 Stage35 进程组，通过结构化 progress/control 文件获取状态并发送正背面确认，不解析终端文本，也不依赖 TTY。

## 四相机 Stage35 合同

Stage35 使用 `configs/zs32/topology/zs32_4cam_double_side_v1.json`，把八个必需视角采集为同一个 complete sample。第一阶段只将以下六个主视角送入现有模型：

```text
front, front_left, front_right,
back, back_left, back_right
```

最终 runtime manifest 同时保留 `front_secondary` 和 `back_secondary` 的 source path、采集元数据及明确的 unsupported model 状态。第一阶段缺少 secondary 模型资产属于预期状态，不是执行错误。

结果 schema 按 view 组织，不把 GUI 固定成“只有六个模型结果”。每个 view 至少记录：

- view 名称与稳定显示顺序；
- source path 与 capture identity；
- `model_supported` 和可用分支；
- 可用分支的 score、status、evidence path 与 reason；
- mask 类型与阈值来源；
- 采集或运行错误文本。

以后 secondary 权重准备好时，只需新增这两个 view 的模型证据，不改变 GUI 合同。

## 结构化进度与采集确认

Stage35 接收 progress 输出路径，并在状态变化时原子替换 JSON sidecar。至少支持：

- `waiting_front`
- `capturing_front`
- `waiting_back`
- `capturing_back`
- `running_template`
- `running_patchcore_yolo`
- `running_fusion`
- `complete`
- `failed`

每条记录至少包含 part ID、已知时的 capture session、时间戳、state、message 和 error details。看板轮询该文件，不能根据 stdout 或 stderr 推断状态。

四相机采集不得继续依赖终端 `Enter`。当 state 为 `waiting_front` 或 `waiting_back` 时，progress JSON 同时发布本次等待唯一的 `confirmation_id`。看板把同一个检测按钮分别显示为“确认正面并拍摄”或“确认背面并拍摄”；操作员点击按钮或按 `S` 后，看板原子写入 control JSON：

```json
{
  "action": "confirm_round",
  "round": "front",
  "confirmation_id": "本次 waiting 状态提供的唯一值",
  "part_id": "当前 part ID"
}
```

采集进程只接受 round、part ID 和 `confirmation_id` 均与当前等待状态一致的确认。旧文件、重复点击、错误 round 或其他检测的确认全部拒绝。收到有效确认后才进入对应 `capturing_*` 状态。control 文件与 progress 文件使用同目录临时文件加 `Path.replace()` 原子发布，不通过 stdin/stdout 传递控制指令。

## PatchCore mask 产物

Stage32 需要额外持久化足以生成真实红色缺陷 mask 的证据：

1. 无损保存原始浮点 anomaly map。
2. Anomalib 提供 `pred_mask` 时优先保存并使用它。
3. 缺少 `pred_mask` 时，对该视角 anomaly map 归一化，并使用默认 `0.65` 的可配置显示阈值。
4. 保存最终二值 mask，并记录来源是 `pred_mask` 还是 diagnostic fallback。

fallback 只用于显示。界面必须标注 `DIAGNOSTIC MASK`，并且不得改变 PatchCore、融合、OK/NG、ROI 或部署阈值。

mask 根据记录的 ROI 从 PatchCore crop 坐标映射到源图。mask resize 使用 nearest-neighbor。源图显示缩放和 letterbox 必须把同一个几何变换应用于图片及所有 overlay。

## 其他证据类型

### YOLO

当前 YOLO 是 object detection 模型。真实 `xyxy` detection 显示为浅红色半透明矩形、红色边框、类别和置信度。空 detections 不产生红色区域，不能把 bbox 宣称为像素级 segmentation mask。

### 模板匹配

显示 template score 和 status。只有 runtime 以后产生经过验证的像素级差分产物时，才允许显示 template mask。当前 score、best template 和 offset 不足以构造缺陷 mask。

### Fusion

Fusion 只合并同一 view 中实际可用的证据层，并保留证据来源。被跳过、不支持或缺失的分支必须明确标记，不能静默当作 CLEAR。

## 看板布局

```text
┌─────────────────────────────────────────────────────────────────────┐
│ ZS32 | Part ID | 检测时间 | LIVE/OFFLINE | 总状态及原因             │
├──────────────┬──────────────┬──────────────┬────────────────────────┤
│ front        │ front_left   │ front_right  │ front_secondary        │
│ 图片/红mask  │ 图片/红mask  │ 图片/红mask  │ 原图·暂未接入模型      │
│ 分数和状态   │ 分数和状态   │ 分数和状态   │ 采集状态               │
├──────────────┼──────────────┼──────────────┼────────────────────────┤
│ back         │ back_left    │ back_right   │ back_secondary         │
│ 图片/红mask  │ 图片/红mask  │ 图片/红mask  │ 原图·暂未接入模型      │
│ 分数和状态   │ 分数和状态   │ 分数和状态   │ 采集状态               │
├─────────────────────────────────────────────────────────────────────┤
│ Fusion | Original | PatchCore | YOLO | Template                    │
│ 检测操作[S] | 退出[Q] | 当前阶段、进度或错误信息                   │
└─────────────────────────────────────────────────────────────────────┘
```

- 默认显示 Fusion。
- 红色 mask 默认透明度约为 45%。
- secondary 在模型接入前显示原图和“暂未接入模型”。
- 点击卡片进入单卡放大；再次点击或按 Escape 返回八图。
- 鼠标按钮与键盘快捷键调用同一动作。
- 按钮标签随状态变化：idle 为“开始检测”，`waiting_front` 为“确认正面并拍摄”，`waiting_back` 为“确认背面并拍摄”，其他运行状态禁用。重复点击和过期确认不得触发采集。
- `--part-id` 提供真机 part ID；第一版不在 OpenCV 画布中实现文本输入。

状态颜色：OK 为绿色，NG 为红色，REVIEW 为橙色，运行中为蓝灰色，执行错误为紫红色。

## 判定与异常语义

- 合法 `NG_TEMPLATE` 保留真实模板状态并短路后续模型；未执行的 view 和 branch 显示“未执行”，不能显示零分或伪造分数。
- 没有锁定阈值的 diagnostic inference 保持 REVIEW；diagnostic mask 不得把它升级为 OK 或 NG。
- secondary 模型不支持是第一阶段预期中性状态。
- 文件缺失、图片无效、采集不完整、身份冲突或 mask 几何错误属于执行错误，不是 NG。
- mask 错误不应隐藏仍可读取的原图和有效分数，受影响卡片显示证据错误。
- Stage35 非零退出时显示退出码和结构化失败原因，保留已经落盘的文件。
- 看板使用独立 process group 启动 Stage35。关闭时终止并等待仍在运行且由看板拥有的整个进程组，确保 Stage35、采集器或 Stage32 孙进程都被回收；不删除 capture 或 runtime 产物。

## 身份校验

解析器拒绝混合不同检测。所有可用记录必须在以下字段上一致：

- `part_id`
- `capture_session`
- `group_id`
- `hand`
- `view`
- `manifest_identity`

八张源图必须来自同一个 complete sample，并且 topology 的每个 required view 恰好一张。第一阶段 branch CSV 可以只包含六个受支持视角，但每一行仍必须属于同一个 inspection identity。

## 测试策略

增加聚焦测试覆盖：

- 八视角 manifest 映射、稳定顺序、唯一性和 complete sample 校验；
- manifest、JSON、CSV 与图片路径之间的 identity 冲突拒绝；
- 六个主视角进入推理、两个 secondary 透传；
- 同一 view-oriented parser 以后可以接收 secondary 模型结果；
- `pred_mask` 优先级和 anomaly map `0.65` diagnostic fallback；
- ROI 到源图的 mask 映射、nearest-neighbor resize、显示缩放与 letterbox 对齐；
- 真实 YOLO box、空 detections 以及禁止伪造 segmentation；
- template short circuit、REVIEW、NG、执行错误和 unsupported secondary 绘制；
- OpenCV 卡片绘制、鼠标热区、快捷键、证据层切换与单卡放大；
- 子进程启动、重复启动拦截、进度轮询、非零退出和关闭清理；
- front/back `confirmation_id` 绑定、过期或重复 control 拒绝、按钮状态切换，以及无 TTY 采集；
- CLI `--help` 和基于本地真实产物的离线 smoke test。

离线验收后执行最小真机四相机 smoke test：一个右手件、八张源图、六个模型视角、两个 unsupported secondary 卡片和一个界面结果。该测试仅证明 commissioning 链路可运行，不是生产放行证据。

## 交付顺序

1. 定义 view-oriented result 和 progress 合同。
2. 持久化 PatchCore raw map 与真实二值 mask。
3. 实现并验证离线结果解析器、证据合成器和 OpenCV 看板。
4. 使用现有真实产物与八视角 fixture 验证布局和证据行为，且不得混合 capture identity。
5. 请用户验收离线显示。
6. 扩展 Stage35 四相机采集、六视角推理和 secondary 透传。
7. 接入“开始检测”和结构化真机进度。
8. 执行单件真机 hardware smoke test。

实现到第 4 步后必须暂停，等待用户确认离线显示正确，再进入真机集成。
