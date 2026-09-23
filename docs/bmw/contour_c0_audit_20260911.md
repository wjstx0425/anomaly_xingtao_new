# 整圈轮廓 C0 输入及兼容性核查（2026-09-11）

本记录是本机静态核查与已有软件测试结果，不是整圈实图验证或现场验收。

## 本地代码

- 分支：`feat/bmw-four-camera-runtime-cleanup`。
- HEAD：`c59c33ceae8a1e394995ee426c8b06e8eb78610b`。未执行 reset、checkout、pull、提交或推送。
- 本地钢印公共接口在 `src/bmw_inspection/checks/stamp_reader.py`、`stamp_result.py`，`StampReadResult` 表示 OCR 读取质量，不是产品判定。其 `to_dict()` 返回独立 JSON 快照，`save()` 拒绝覆盖；结构含 OCR 特定图像，不适合作为轮廓结果基类。
- `EightViewInspection.__post_init__` 仍要求 `final_status` 与原模型结果融合一致；`DemoBranch` 仍只有 Template、bright_streak、YOLO、EfficientAD。未发现钢印已接入的通用 `requirement_results` 集合。C0—C3 保持独立离线，不能把轮廓伪装成第 26 个模型或改 UI 状态绕开合同。
- `HdrSourceImages.source_kind` 明确区分 `hdr_pair` 与 `fused_only`。融合输入不能声称独立多曝光验证。

## 目标图片

交接包 README 明确不含两张原图。对 `docs`、`dataset`、`artifacts`、`results` 使用包含忽略文件的文件名检索，只找到以下同名原始采集图，全部是 **4024×3036 RGB**，全部不匹配文档的 **2047×1545 RGBA** 摘要：

| 用途与 session | SHA-256 |
|---|---|
| normal / 20260823_161606_428291 | `00d5e69ee1b082c4fd7b0b4a589a2bc80919372ebdd707f32905b26d10f5db9c` |
| normal / 20260823_164432_522052 | `1f2fedfb03d804a125ace991be328489f40032c4a991de76b3a55e6378b74fc1` |
| defect / 20260820_155511_803522 | `04a0834ad594408be5bc0dd89a313cb1e1742a0f33cecd4c5131f8846da2dde4` |

正常路径：`dataset/bmw_lab_raw_clean_0820/left/front/normal/<session>/images/left_front_normal_bmw_left_normal_retake_group001_000001_fused.png`。

缺陷路径：`dataset/bmw_lab_raw_clean_0820/left/front/defect/defect/20260820_155511_803522/images/left_front_defect_defect_bmw_left_defect_group010_000001_fused.png`。

对应 `dataset/bmw_lab_raw_clean_0820/manifests/<session>.csv` 的 image 行确认 `view=front`、相机 `DA9805574`、来源 `hdr_fused`；不是语义视图 `front_left`。两个正常 session 内容不同，不应仅按 basename 选标准。

文档预期参考 SHA：`0cb49973a910dfc582fa846da92dd5ed73b18f2c72ac96e84e1abfa0424bdd7b`；待测 SHA：`89ce4a9ce13bdfc61032a0937bcc5efc535497c051d35264e074174ea9b70265`。未发现这两份同名输入，因此本轮不能声称复现文档 11.52 px / 11.05 px 局部结果。没有擅自缩放本地图来伪造上述输入，也没有移用文档孔 ROI。

检索未见旧 `BMW_轮廓缺口_实图验证与Codex接续.zip`；已有 `artifacts/bmw_contour_review_20260908/` 是不同视图的可行性材料，不能代替该回归包。本核查没有按 SHA 扫描全盘其他名称大图。

## 已运行旧功能软件基线

在已有根目录 `.venv` 上运行（不下载或升级依赖）：

```bash
UV_CACHE_DIR=/tmp/bmw-contour-audit-uv uv run --no-sync python -m pytest tests/unit/bmw_inspection tests/unit/pipeline/test_bmw_lab_eight_view_demo.py -q
```

结果：**283 passed，1 warning，1.03 s**。warning 为 PyTorch `Can't initialize NVML`。这是 BMW/钢印/现有八视图逻辑测试，不包含 GPU、相机、现场检测验证，不等于新增轮廓模块测试通过。

## 实现审核关注项

- required 覆盖分母绑定参考弧长；未知和忽略不得通过缩小分母消失。跨起点事件合并，不跨未知间隙连接曲线。
- 正负距离约定：向内为正；不完整待测轮廓不能人为闭合后推导符号，未知符号用 null。
- 短但严重的超差至少 REVIEW 候选，不能被连续长度过滤后 PASS。
- 法向剖面不是唯一通路；角点、窄脚、多值结构需要二维当前图像证据。保留远离参考的实际粗轮廓以检查大余料和结构消失。
- 参考 mask 仅为可改变的在线概率初值，不能强制正常轮廓存在。确定主体种子应由当前图像验证。
- 两孔刚性配准只去除 SE(2) 放置变化，不能独立证明没有三维姿态变化；第三检查基准或真实重复取放验证仍需要。
- 非 8 位拒绝；alpha 无效域不得视作黑背景；错误规格不 resize；图像只读。
- 独立事件与未知段同时保留；所有输出需相对路径、严格 JSON、原子发布和拒绝覆盖。evaluate 标签不传入 inspect。

## 需用户补充的信息

1. 提供文档所列两张 2047×1545 RGBA 原图所在路径，或确认改用上述哪一个正常 session 与本地全分辨率缺陷图重新建模；两种方案的 ROI 和数值基线不可混用。
2. 全圈材料边界（尤其暗边、小脚、夹具接触处）及独立定位基准的人工确认。开发阈值可先实现，但不可当验收公差。
3. 后续 C4 所需真实重复取放、质量确认正常件和缺陷样件，以及最小目标缺陷/边段公差。该数据缺口不阻止离线软件框架开发。
