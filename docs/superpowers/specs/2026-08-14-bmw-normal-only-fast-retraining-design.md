# BMW正常件快速重训练设计

## 目标

新增一个实验室入口，用新发布的八视角正常数据重训 Template 和 EfficientAD，并用旧数据中的 `no_streak` 与新正常数据重新标定传统光痕规则。YOLO 不进入计划、不执行训练，现有模型和 Demo 配置不自动替换。

## 数据流

1. 输入已经由 `bmw_lab_prepare_eight_view_data.py` 发布的 normal-only prepared release。
2. 从已有右手八视角 ROI 复制坐标，发布一个绑定新 prepared release 的 fixed-setup ROI 配置。
3. 物化新的 ROI training release；只消费其 Template 与 EfficientAD 分支。
4. 新正常光痕清单与指定旧 training release 所指向清单中的 `no_streak` 行合并，再调用现有传统光痕重标定器。
5. 八视角 Template、八个 EfficientAD 和光痕报告写入新的 `run-id`。YOLO 不创建输出。

## 最小接口

入口为 `pipeline/bmw_lab_retrain_normal_only.py`。必须传入 `--prepared-root`、`--reuse-roi`、`--old-no-streak-release`、`--training-id` 和 `--run-id`；GPU、epoch、输出根目录保持现有实验室默认值。`--dry-run` 只打印计划，不裁图、不训练。

## 约束

- prepared release 只能包含 normal；否则拒绝。
- 旧 release 必须至少提供 calibration 和 final_test 的 `no_streak`。
- EfficientAD 正常校准件数从实际 `normal_test` 动态读取，不再要求恰好21件；整件目标误报率仍为 `1/21`，10个校准件时允许0个误报。
- 不修改当前 Demo 配置，不覆盖任何现有 release/run，不训练或复制 YOLO 模型。
