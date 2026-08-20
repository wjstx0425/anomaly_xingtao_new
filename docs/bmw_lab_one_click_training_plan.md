# BMW八视图实验室一键训练实施计划

## 交付目标

新增可恢复、失败即停的一键训练入口，默认YOLO batch为32，并确保旧六视图代码不被改变。

## 实施步骤

1. 先写最小测试，覆盖八视图顺序、YOLO batch=32、五个步骤编排和dry-run不启动训练。
2. 新增`src/bmw_inspection/lab/eight_view_train_all.py`，实现数据物化、EfficientAD、YOLO、Template、光痕标定和报告。
3. 新增`pipeline/bmw_lab_train_all.py`，提供中文命令行入口和实验室默认参数。
4. 运行目标测试、CLI帮助和真实数据dry-run；不运行GPU训练。
5. 更新`pipeline/README.md`和`AGENTS_MEMORY.md`，记录唯一推荐命令及实验边界。
