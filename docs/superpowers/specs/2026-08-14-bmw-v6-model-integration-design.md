# BMW 2026-08-14 新模型接入 V6 设计

## 目标

训练完成后，用一条命令生成可由现有八视图中文 Demo 直接加载的独立 V6 候选。V5 不修改，随时可以回退。

## 固定输入

- 新训练 run：`results/bmw_lab_one_click/bmw_right_normal_20260814_models_v1`
- V5 配置：`configs/bmw/experiments/bmw_eight_view_demo_v5_template_manual_ignore_mask_v1.json`
- 新旋转光痕候选：`results/bmw_bright_streak_rotated_retrain/bmw_right_normal50_no_streak1_20260814_v2/report.json`
- V5 手动忽略 mask、可信 OK、YOLO、相机 HDR 和公共 ROI 继续复用。

## 发布产物

新增 `pipeline/bmw_lab_prepare_normal_20260814_v6_demo.py`。它只发布新资产，不训练模型：

1. 组合 run `results/bmw_lab_one_click/bmw_right_normal_20260814_v6_demo_v1`
   - `template` 指向新 run 的 Template；
   - `efficientad` 指向新 run 的 EfficientAD；
   - `yolo` 指向 V5 使用的 YOLO；
   - `composition.json` 记录三类来源路径和 SHA-256。
2. Template 阈值资产
   - 八个数值阈值逐值沿用 V5；
   - 手动 ignore-mask SHA 不变；
   - 只把八个 `model_json_sha256` 重新绑定到新 Template；
   - 明确记录 `thresholds_recalibrated=false` 和 V5 来源资产 SHA。
3. EfficientAD 阈值资产
   - `thresholds`、`base_thresholds` 和 `threshold_margin` 逐值沿用 V5；
   - 只把 `checkpoint_sha256_by_view` 重新绑定到新 checkpoint；
   - 明确记录 `thresholds_recalibrated=false` 和 V5 来源资产 SHA。
4. V6 experiment JSON
   - 新 `demo_id`、`training_run` 和 `result_root`；
   - 保留 V5 的手动 mask、相机、HDR、旧公共 ROI、可信 OK 和 YOLO 参数；
   - Template/EfficientAD 指向两份新重绑阈值资产；
   - 光痕使用单独的 rotated-V3-candidate 引擎、现有倾斜 ROI 和新候选报告。

## ROI 与 mask 合同

新训练 ROI 和 V5 公共 ROI 的八组坐标相同，但文件 SHA 不同。手动 mask 绑定的是 V5 公共 ROI SHA，所以 V6 运行配置必须继续引用 V5 公共 ROI。准备脚本发布前逐视角验证两份 ROI 坐标完全相同；不相同则停止。

## 光痕合同

新增独立的 `tracked_profile_v3_manual_rotated_candidate` 配置分支，不放宽 V5 原有严格报告加载器。候选加载器验证：

- 顶层报告 SHA；
- candidate schema、算法名、状态和阈值字段；
- normal manifest、单张 no-streak 原图和旋转 ROI 的路径及 SHA；
- 50 个 normal、1 个 no-streak、训练内 0 false reject / 0 false accept；
- `no_streak_independent_test_count=0`，并在运行证据中标记为实验候选。

这张无光痕图是 `20260814_094431_343851/front_left` HDR 图。它参与了拟合，因此不能作为独立测试证据。

## 完整性和失败处理

准备命令在以下任一条件下拒绝发布：

- 新训练 `run_report.json` 不是 `complete`；
- Template 或 EfficientAD 不是八视角完整模型；
- EfficientAD 新阈值分析产物缺失；
- 任一 V5 阈值、mask、ROI、YOLO、光痕或可信 OK 资产 SHA/路径不匹配；
- 输出组合 run 或 V6 配置已经存在。

脚本不修改 V5，不复制大模型，不重新训练 YOLO，不使用新 run 中 final-test balanced accuracy 仅 0.45 的 legacy bright-streak 结果。

## 验证

聚焦测试覆盖：半套训练拒绝、阈值数值逐值不变、模型 SHA 更新、mask/ROI 保持、YOLO SHA receipt、光痕候选严格加载、V5 文件字节不变、V6 配置可由现有 loader 和 model suite 加载。训练结束后先运行准备命令，再进行一次离线已保存正常件 smoke，最后启动四相机实验模式。

## 当前训练状态快照

2026-08-14 当前核查时只有 `front`、`front_left`、`front_right`、`front_secondary`、`back` 五个 EfficientAD checkpoint，进程列表中未发现训练进程。V6 准备器必须保持拒绝，直到八个视角及最终阈值资产全部生成。
