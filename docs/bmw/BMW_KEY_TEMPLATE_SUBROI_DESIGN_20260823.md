# BMW八视图关键区Template二次检测设计

日期：2026-08-23

## 1. 目标

在现有左右手BMW八视图检测链路中保留整块Template检测，并增加可选的关键区Template二次检测：

- 左右手分别维护ROI、模板模型和阈值。
- 八个视角都允许配置关键区，每个视角允许0到多个矩形ROI。
- 每个关键ROI独立执行Template匹配并产生分数、阈值、状态和证据。
- 同一视角的关键ROI聚合成一项 `key_template` 结果。
- 任一关键ROI为NG时，该视角关键区结果为NG，最终整件为NG。
- 不短路其他检查，继续完成并保存全部证据。
- 阈值在JSON中直接修改，不使用SHA、receipt、publisher或rebind流程。

## 2. 非目标

- 不替换现有整块Template检测。
- 不修改YOLO、EfficientAD、光痕、HDR采集、公共ROI或现有融合优先级。
- 不把Template诊断热图描述为缺陷分割结果。
- 不共用左右手关键ROI坐标或模板。
- 不在本功能中训练其他视角的现有四模块模型。

## 3. 方案选择

采用“每个启用视角聚合一项关键区结果”的方案。

每个ROI仍独立判定，但顶层结果身份保持为 `(key_template, view_id)`，与现有按视角和分支组织UI、保存结果及可信OK对比的方式一致。没有关键ROI的视角不执行该分支，也不生成SKIPPED结果。

总检查数动态计算：

```text
总检查数 = 25 + 配置了至少一个关键ROI的视角数量
```

## 4. 文件与资产布局

左右手分别使用以下文件：

```text
configs/bmw/key_template_rois/bmw_right_v1.json
configs/bmw/key_template_rois/bmw_left_v1.json

configs/bmw/key_templates/bmw_right_v1.json
configs/bmw/key_templates/bmw_left_v1.json

results/bmw_lab_one_click/bmw_right_key_template_v1/template/<view>/<roi_id>/
results/bmw_lab_one_click/bmw_left_key_template_v1/template/<view>/<roi_id>/
```

ROI选择文件只保存人工选择的矩形。运行配置由训练成功后生成，保存实际模型路径和直接可编辑的阈值。活动Demo配置通过可选字段 `key_template_config` 引用对应运行配置。没有该字段时，Demo维持当前25项行为。

## 5. 坐标合同

关键ROI坐标使用当前手对应公共ROI裁图内的局部半开区间坐标：

```text
[x1, y1, x2, y2], 0 <= x1 < x2 <= public_roi_width
                      0 <= y1 < y2 <= public_roi_height
```

选择工具直接展示公共ROI裁图，用户不需要换算4024×3036全图坐标。训练和运行时均从同一公共ROI裁图提取关键ROI，避免左右手或全图/局部坐标混用。

## 6. ROI选择工具

新增独立CLI，左右手分别运行。工具读取：

- 对应0823 prepared manifest。
- 当前手公共ROI配置。
- 用户指定的代表正常样本；未指定时选择确定性的首个完整train normal样本。

工具依次展示八个公共ROI裁图，支持：

- 鼠标拖动：新增矩形。
- `Z`或Backspace：撤销最后一个矩形。
- `R`：清空当前视角。
- `Enter`：确认当前视角并进入下一个；零个矩形表示该视角不启用。
- `Esc`或`Q`：退出且不保存。

矩形按视角自动命名为 `roi_01`、`roi_02`。工具拒绝越界、非正面积、过小或完全重复的矩形，并在最终保存前显示每个视角的ROI数量摘要。

## 7. 训练与阈值

新增关键区Template训练CLI。输入为：

- 一只手的0823 prepared release。
- 该手公共ROI配置。
- 该手关键ROI选择文件。
- 新模型输出根和运行配置输出路径。

首次训练默认处理全部已选ROI；后续可用 `--view` 和 `--region` 只重训一个或一组关键区，未选择的运行配置条目和模型路径保持不变。运行中的Demo仍使用内存模型，重训完成后需要重启才会加载新资产。

每个关键ROI独立处理：

1. 从train normal样本裁出关键区。
2. 沿用当前Template预处理：灰度、3×3模糊、512×512、`TM_CCOEFF_NORMED`、最大平移12。
3. 从train normal中确定性选择5个代表模板。
4. 对calibration normal计算风险 `1 - similarity`。
5. 初始部署阈值取该ROI calibration normal风险最大值。
6. final_test normal只报告误报数、最大风险和分位数，不参与阈值选择。

训练输出记录样本数量、模板数量、阈值和final_test报告，但不生成或校验SHA。训练全部成功后才写完整运行配置；失败时不修改活动Demo配置。

由于初始阈值只使用正常件，它只定义当前正常分布上界，不证明真实缺陷召回率。后续现场确认的缺陷件可用于收紧阈值，但不得静默改变本次初值。

## 8. 运行时数据流

`EightViewModelSuite`增加可选关键区Template预测器：

1. 每轮采集后仍只生成一次每视角公共ROI裁图。
2. 现有整块Template、YOLO和EfficientAD继续复用该裁图。
3. 对本轮中配置非空的视角，从公共ROI裁图提取各关键ROI并逐一匹配。
4. 同一视角结果按 `ERROR > NG > PASS` 聚合。
5. front和back阶段分别追加已启用视角的关键区结果。
6. 最终合并时按固定分支顺序组织：整块Template、关键区Template、光痕、YOLO、EfficientAD。
7. 最终融合仍为任一ERROR则ERROR，否则任一NG则NG，否则OK。

关键区模型推理继续由现有单模型worker串行执行，不引入新的模型并发或相机线程。

## 9. 结果与证据

新增 `DemoBranch.KEY_TEMPLATE = "key_template"`。每个启用视角产生一条顶层结果，`details`包含所有子ROI：

```json
{
  "region_count": 2,
  "pass_count": 1,
  "ng_count": 1,
  "error_count": 0,
  "worst_region_id": "roi_02",
  "regions": [
    {
      "id": "roi_01",
      "roi_xyxy": [120, 300, 360, 620],
      "status": "PASS",
      "score": 0.006,
      "threshold": 0.012
    }
  ]
}
```

聚合分数使用“最大阈值超限量”对应ROI的分数与阈值，便于现有列表显示。每个视角只生成一张关键区证据联系图：左侧为公共ROI，PASS矩形为绿色、NG为红色、ERROR为黄色，并标注ROI编号、分数和阈值；右侧依次排列各关键ROI的局部诊断差异图。这样现有持久化结构仍只需保存 `evidence/key_template_<view>.png`，每个ROI的数值详情同时写入 `inspection.json`。

中文UI增加“关键区模板”分支入口。没有配置关键ROI的视角不显示该分支。可信OK诊断继续使用该视角公共ROI参考，仅用于对比，不改变关键区判定。

## 10. 配置合同

运行配置允许八个规范视角的子集；缺失视角与空数组等价。示例：

```json
{
  "views": {
    "front": [
      {
        "id": "roi_01",
        "roi_xyxy": [120, 300, 360, 620],
        "model": "../../../results/bmw_lab_one_click/bmw_right_key_template_v1/template/front/roi_01/model.json",
        "threshold": 0.0123
      }
    ],
    "back_left": []
  }
}
```

加载时只保留必要检查：

- JSON可解析。
- 视角名称属于规范八视角。
- 同一视角ROI ID唯一。
- 坐标为四个整数、正面积且位于公共ROI内。
- 模型文件和模板图片存在且可读。
- 模型输入尺寸与关键ROI尺寸一致。
- 阈值为有限非负数。

不检查SHA、schema版本、manifest身份、source release身份或重复发布身份。

## 11. ERROR行为

- 运行配置、模型或ROI合同错误：启动失败并在GUI显示明确ERROR。
- 单个关键ROI推理异常：该ROI为ERROR，该视角聚合结果为ERROR，最终整件ERROR。
- 一个ROI为NG、另一个为ERROR：ERROR优先。
- 一个视角关键区失败不阻止其余视角和四个现有模块继续执行。
- 保存失败沿用现有Demo错误处理，不把未保存结果显示为普通成功。

## 12. 测试与验证

只做聚焦验证：

- ROI选择文件解析、空视角、自动编号和坐标边界测试。
- 单ROI与多ROI训练、calibration最大风险阈值、final_test不参与选择测试。
- 左右手配置和模型路径隔离测试。
- 运行时0个、1个和多个ROI的PASS/NG/ERROR聚合测试。
- front/back分阶段结果数量和顺序测试。
- 任一关键ROI NG导致最终NG且其他模块仍执行的测试。
- UI中文分支、红绿黄证据框和保存JSON/证据测试。
- Python语法检查和 `git diff --check`。
- 每只手一次完整GPU离线八视图回放。

相机采集和现场关键缺陷召回率属于实施后的未验证边界，需要操作员选择ROI、完成训练并用现场件确认。

## 13. 验收标准

- 左右手关键ROI完全独立。
- 八个视角均允许0到多个ROI。
- 未启用关键区时原25项结果不变。
- 任一关键ROI NG使整件NG，且检测不短路。
- 每个ROI的分数、阈值、状态和证据可查看并保存。
- 阈值可在JSON中直接修改，无需重新计算SHA。
- 重新划分某个ROI后只需重训对应关键区，不影响其他ROI和四个现有模块。
- 当前活动调用链仍保持直接、可读，零基础人员可以找到ROI、模型和阈值位置。
