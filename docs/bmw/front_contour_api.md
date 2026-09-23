# 正面轮廓与几何测量模块接入

更新：2026-09-23。Python 内存接口已从批量报告脚本中提取；当前能力仍是开发级观测，不是已验收的整圈轮廓判定。

## 1. 代码位置

仓库根目录：`/home/yunjing/anomaly_xingtao_new`。

| 路径 | 职责 |
|---|---|
| `src/bmw_inspection/checks/contour_compare/api.py` | 初始化参考、单张处理、结构化结果；后续系统调用入口 |
| 同目录 `registration.py` | 双孔自动搜索与刚性定位 |
| 同目录 `segmentation.py`、`acquisition.py` | 当前材料候选分割、图像支持的轮廓采集 |
| 同目录 `geometry_features.py` | 直边、小脚边段与孔心到拟合直线的距离 |
| `tools/bmw/measure_front_geometry.py` | 清单校验、调用 API、保存 JSON/CSV/NPZ 和图片报告 |
| `configs/bmw/checks/contour/left_front_4024_automatic_geometry_v1.json` | 当前左件 front、4024×3036、fused 开发配方 |
| `tests/unit/bmw_inspection/checks/test_front_contour_api.py` | 接口约束与缓存隔离测试 |

`/home/yunjing/MVI_xingtao_PL/V` 是需求和交接资料位置，代码没有写入该目录。

## 2. 最小调用示例

在仓库已有 uv 环境中运行。其他 Python 工程可以将本仓库 `src` 加入模块搜索路径，或使用既有项目安装方式；本轮未制作独立 wheel。核心依赖为 NumPy、OpenCV、SciPy，不需要相机 SDK 或神经网络权重。

```python
import json
from pathlib import Path

import cv2
from bmw_inspection.checks.contour_compare import FrontContourInspector

root = Path('/home/yunjing/anomaly_xingtao_new')
manifest = json.loads((root / 'artifacts/bmw_front_contour_20260915/manifest.json').read_text())
row = next(r for r in manifest['samples']
           if r['sample_id'] == manifest['reference_sample_id'])
reference_path = root / row['image_path']

# 初始化一次，可连续处理多帧；参考图与输入图必须属于同一成像配方。
inspector = FrontContourInspector.from_files(
    root / 'configs/bmw/checks/contour/left_front_4024_automatic_geometry_v1.json',
    reference_path,
    root / 'artifacts/bmw_contour_reference_20260912/reference_mask_folds_only.png',
)

# 示例使用参考自身。接入时替换成采集程序传来的 uint8 BGR/BGRA 数组。
frame = cv2.imread(str(reference_path), cv2.IMREAD_UNCHANGED)
result = inspector.process(frame, hand='left', view_id='front', channel='fused')
print(result.record['pose']['valid'], result.record['hole_spacing_px'])

if result.contour is not None:
    points = result.contour['observed_xy']
    # 只能连接明确有效的线段，不能把过滤后所有点直接连成闭合轮廓。
    starts = result.contour['segment_valid'].nonzero()[0]
    ends = result.contour['segment_end_index'][starts]
    segments = [(points[i], points[j]) for i, j in zip(starts, ends)]

payload = result.to_dict()  # 不含大体积 mask/轮廓数组，仍包含几何特征采样数据
json_text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
# result.to_dict(include_arrays=True) 可导出完整数组；未知坐标转换为 null。
```

也可直接用 `FrontContourInspector(config_dict, reference_image, reference_mask)` 初始化，完全不读文件。`reference_features` 返回独立副本。参考资产与配置在初始化时复制，输入帧不修改；每次结果属于该次调用，不会修改内部基线。

## 3. 输入与输出契约

- 图像：完整图、uint8、OpenCV BGR 或 BGRA，尺寸必须严格匹配配方；BGRA 中只有 alpha=255 的区域有效。
- 掩膜：同尺寸二维二值数组，接受 bool、0/1、0/255；必须同时有前景和背景。
- `hand/view_id/channel` 必填，与配方一致。channel 是调用方的来源声明，程序无法仅凭像素证实图像经过 HDR。
- 不隐式缩放、不做 HDR、不触发相机、不写文件。`acquire_contour=False` 可以跳过轮廓采集，保留定位和几何测量。
- `record`：`pose`、`features`、`hole_spacing_px`、独立边段一致性、轮廓摘要、计算耗时及来源声明。
- `current_mask`：0/1 材料分割候选，不是已确认真实边界。
- `contour`：候选点 `candidate_xy`、观测点 `observed_xy`、点有效性 `valid`、失效原因 `invalid_reason`、线段终点索引及线段有效性。坐标均为输入原图 x/y 像素。
- `record['status']` 始终 `REVIEW`、`diagnostic_only=True`、`whole_part_release=None`。独立边段一致不代表整件合格。
- 孔心到直线距离是二维投影到无限拟合直线的垂距；垂足是否超出观测区间另有标记。观测支持跨度不等于完整物理边长。
- 当前候选弧长支持率不是固定参考周长覆盖率，不可与旧严格比较的 coverage 混用。

## 4. 失败和运行方式

错误图像尺寸、数据类型、来源声明以及无效参考/几何配置抛 `ContourInputError`。参考孔定位失败会阻止初始化；当前帧孔定位失败则返回 `REVIEW`、空 features、`current_mask=None`、`contour=None`，不继续分割。

文件读取错误和底层 OpenCV 运行异常可继续抛出，调用方应记录为处理错误，不能转为合格。默认接口不保存采集ID、文件哈希或证据；这些由宿主程序添加，原批量脚本仍校验清单 SHA。

一个实例串行处理。并行时使用独立进程，各自初始化，避免 OpenCV 随机种子及线程资源互相影响。API `elapsed_ms` 是本次算法调用时间；批量脚本的同名字段还包括读取和部分报告处理时间，不能混作节拍测试。

换 500 万像素相机后应重新准备配方、参考图、参考掩膜和孔/边段区域；当前 4024×3036 配方会拒绝新尺寸。模块封装没有解决暗面误分割，也没有实现此前讨论的取消掩膜硬排除方案。

## 5. 与严格轮廓比较的区别

`python -m bmw_inspection.cli.contour {teach,inspect,evaluate}` 是原有固定参考周长比较流程；本接口封装的是本次会话的自动几何测量和独立轮廓采集。两者没有合并判定，也没有接入八视角主流程、PLC 或生产放行逻辑。

验证结果：240 项轮廓测试通过，其中17项为新接口测试；仅有既存 NVML 警告。参考图和 normal_extra_001 两张真实图的源 SHA 已核验，几何结果与旧结果在1e-10容差内一致，mask及所有轮廓NPZ数组完全一致。

真实图封装回归结果位于 `artifacts/bmw_front_contour_api_20260923/`。此前实际轮廓仍存在暗面/小脚误分割，完整结果参见 `docs/bmw/front_automatic_geometry_20260915.md`。
