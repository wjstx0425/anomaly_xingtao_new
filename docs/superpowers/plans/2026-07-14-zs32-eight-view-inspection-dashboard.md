# ZS32 八视角检测结果看板实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个纯 OpenCV ZS32 看板，离线展示同一检测的八视角和真实模型证据，并在离线验收后接入四相机 Stage35、无 TTY 正背面确认与单次真机检测。

**Architecture:** 在 `src/zs32_inspection/dashboard/` 中建立独立的合同、解析、证据合成、绘制和真机控制模块，由 `pipeline/36_zs32_inspection_dashboard.py` 提供薄入口。Stage32 保持六主视角模型合同，但新增 raw anomaly map、二值 mask 和 view-oriented manifest；Stage35 使用现有 topology 驱动的 legacy-layout 四相机采集，将两个 secondary 作为中性 unsupported view 透传。

**Tech Stack:** Python 3.10+、OpenCV、NumPy、Pillow、dataclasses、JSON/CSV、`subprocess` process groups、pytest、uv。

## Global Constraints

- 依赖与测试统一使用 `uv`；不新增 Tkinter、Qt、浏览器 UI、Web 服务或大型 GUI 框架。
- 产品固定 `ZS32`、hand 固定 `right`；八视角顺序固定为 `front/front_left/front_right/front_secondary/back/back_left/back_right/back_secondary`。
- 当前只有六个主视角进入模板、PatchCore、YOLO 和 Stage18；secondary 显示真实原图和 `unsupported`，不得伪造分数或 mask。
- PatchCore 优先真实 `pred_mask`；缺失时才使用归一化 anomaly map 和默认 `0.65` 的 display-only 阈值，并标记 `DIAGNOSTIC MASK`。
- YOLO 只画真实 bbox，Template 没有像素差分产物时只显示 score/status。
- 任何身份冲突、文件缺失、图片无效或 mask 几何错误都是执行错误，不得伪装成 NG。
- 看板不得解析终端文本；进度和正背面确认只通过原子 JSON 文件传递。
- `pipeline/32_run_zs32_multimodel_inference.py`、其测试和多个 Stage35 文件已有未提交改动；所有编辑必须增量进行，禁止 checkout/reset 覆盖。
- 完成 Task 5 后必须暂停并让用户验收离线截图/界面；未得到明确确认不得执行 Task 6-9。

---

## 文件结构

```text
src/zs32_inspection/dashboard/
├── __init__.py          # 稳定公开入口
├── contracts.py         # 八视角、结果、进度、确认和 UI state dataclass
├── control.py           # progress/control JSON 原子传输
├── parser.py            # runtime_manifest 到 InspectionResult 的严格解析
├── compositor.py        # ROI mask、YOLO bbox、Fusion 和 letterbox 纯函数
├── render.py            # 4×2 canvas、按钮、状态色和 hit regions
├── live.py              # Stage35 process-group 生命周期
└── app.py               # OpenCV HighGUI 事件循环和 reducer
src/zs32_inspection/cli/dashboard.py
pipeline/36_zs32_inspection_dashboard.py
```

旧 `capture_data/demo_inspection.py` 不被新模块 import；只迁移其中无副作用的 letterbox、中文字体和绘制思路。

---

### Task 1: 建立八视角结果与 progress/control 合同

**Files:**
- Create: `src/zs32_inspection/dashboard/__init__.py`
- Create: `src/zs32_inspection/dashboard/contracts.py`
- Create: `src/zs32_inspection/dashboard/control.py`
- Create: `tests/unit/zs32_refactor/dashboard/test_contracts.py`
- Create: `tests/unit/zs32_refactor/dashboard/test_control.py`

**Interfaces:**
- Produces: `VIEW_ORDER`, `MODELED_VIEWS`, `EvidenceLayer`, `BranchState`, `InspectionIdentity`, `BranchEvidence`, `ViewResult`, `InspectionResult`, `ProgressRecord`, `ConfirmationCommand`。
- Produces: `write_progress(path, record)`, `load_progress(path)`, `write_confirmation(path, command)`, `consume_confirmation(path, expected) -> bool`。

- [ ] **Step 1: 写合同和原子传输的失败测试**

```python
def test_view_order_and_modeled_views_are_closed_sets() -> None:
    assert VIEW_ORDER == (
        "front", "front_left", "front_right", "front_secondary",
        "back", "back_left", "back_right", "back_secondary",
    )
    assert MODELED_VIEWS == (
        "front", "front_left", "front_right", "back", "back_left", "back_right",
    )


def test_confirmation_rejects_stale_wrong_round_and_wrong_part(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    expected = ConfirmationCommand("confirm_round", "front", "token-1", "part-1")
    for command in (
        ConfirmationCommand("confirm_round", "front", "old-token", "part-1"),
        ConfirmationCommand("confirm_round", "back", "token-1", "part-1"),
        ConfirmationCommand("confirm_round", "front", "token-1", "part-2"),
    ):
        write_confirmation(path, command)
        assert consume_confirmation(path, expected) is False
        assert not path.exists()
```

- [ ] **Step 2: 运行测试并确认红灯**

Run:

```bash
uv run --no-sync pytest --confcutdir=tests/unit \
  tests/unit/zs32_refactor/dashboard/test_contracts.py \
  tests/unit/zs32_refactor/dashboard/test_control.py -q
```

Expected: FAIL，提示 `zs32_inspection.dashboard` 或合同类型尚不存在。

- [ ] **Step 3: 实现最小冻结合同和原子 JSON**

```python
VIEW_ORDER = (
    "front", "front_left", "front_right", "front_secondary",
    "back", "back_left", "back_right", "back_secondary",
)
MODELED_VIEWS = ("front", "front_left", "front_right", "back", "back_left", "back_right")

@dataclass(frozen=True, slots=True)
class ProgressRecord:
    part_id: str
    capture_session: str | None
    state: str
    message: str
    timestamp: str
    confirmation_id: str | None = None
    error: str | None = None

@dataclass(frozen=True, slots=True)
class ConfirmationCommand:
    action: Literal["confirm_round"]
    round: Literal["front", "back"]
    confirmation_id: str
    part_id: str

class EvidenceLayer(StrEnum):
    FUSION = "fusion"
    ORIGINAL = "original"
    PATCHCORE = "patchcore"
    YOLO = "yolo"
    TEMPLATE = "template"

class BranchState(StrEnum):
    AVAILABLE = "available"
    SKIPPED = "skipped"
    UNSUPPORTED = "unsupported"
    ERROR = "error"

@dataclass(frozen=True, slots=True)
class InspectionIdentity:
    part_id: str
    capture_session: str
    group_id: str
    hand: Literal["right"]

@dataclass(frozen=True, slots=True)
class BranchEvidence:
    branch: str
    state: BranchState
    status: str
    score: float | None
    reason: str
    evidence_path: Path | None = None
    mask_path: Path | None = None
    mask_source: str | None = None
    roi_xyxy: tuple[int, int, int, int] | None = None
    detections: tuple[dict[str, Any], ...] = ()

@dataclass(frozen=True, slots=True)
class ViewResult:
    view: str
    source_path: Path
    source_sha256: str
    source_shape: tuple[int, int]
    model_supported: bool
    branches: Mapping[str, BranchEvidence]
    capture: Mapping[str, str]

@dataclass(frozen=True, slots=True)
class InspectionResult:
    identity: InspectionIdentity
    views: tuple[ViewResult, ...]
    machine_status: str
    reason: str
    mode: Literal["offline", "live"] = "offline"

def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)

def consume_confirmation(path: Path, expected: ConfirmationCommand) -> bool:
    if not path.is_file():
        return False
    try:
        actual = ConfirmationCommand(**json.loads(path.read_text(encoding="utf-8")))
    finally:
        path.unlink(missing_ok=True)
    return actual == expected
```

`InspectionResult` 必须包含唯一 `InspectionIdentity`、按 `VIEW_ORDER` 排列的八个 `ViewResult`、总状态、原因和模式；`BranchEvidence` 使用 `float | None`，不能用 `0.0` 表示未执行。

- [ ] **Step 4: 运行测试并确认绿灯**

Run: 与 Step 2 相同。  
Expected: PASS。

- [ ] **Step 5: 只提交本任务文件**

```bash
git add src/zs32_inspection/dashboard tests/unit/zs32_refactor/dashboard/test_contracts.py tests/unit/zs32_refactor/dashboard/test_control.py
git commit --only -m "feat: add ZS32 dashboard contracts" -- \
  src/zs32_inspection/dashboard \
  tests/unit/zs32_refactor/dashboard/test_contracts.py \
  tests/unit/zs32_refactor/dashboard/test_control.py
```

---

### Task 2: 持久化 PatchCore raw map、mask 与六主视角 manifest

**Files:**
- Modify: `capture_data/zs32_model_runtime.py:97-124,296-365,617-835`
- Modify: `pipeline/32_run_zs32_multimodel_inference.py:45-78,380-470`
- Modify: `tests/unit/capture_data/test_zs32_model_runtime.py:99-185`
- Modify: `tests/unit/pipeline/test_zs32_multimodel_inference.py`

**Interfaces:**
- Consumes: `MODELED_VIEWS` 和 Task 1 的 view-oriented 字段名。
- Produces: `PatchcoreArtifacts`、`_build_patchcore_mask(anomaly_map, pred_mask, *, output_shape, diagnostic_mask_threshold)`、`--diagnostic-mask-threshold`，以及 `runtime_manifest.json.views` 中六个 modeled view 的完整证据记录。

- [ ] **Step 1: 写 mask 优先级、阈值和持久化失败测试**

```python
def test_patchcore_mask_prefers_pred_mask() -> None:
    anomaly = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    pred_mask = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    mask, source, threshold = _build_patchcore_mask(
        anomaly, pred_mask, output_shape=(4, 4), diagnostic_mask_threshold=0.65,
    )
    assert source == "pred_mask"
    assert threshold is None
    assert set(np.unique(mask)) <= {0, 255}


def test_patchcore_mask_uses_normalized_diagnostic_fallback() -> None:
    anomaly = np.array([[10.0, 16.4], [16.5, 20.0]], dtype=np.float32)
    mask, source, threshold = _build_patchcore_mask(
        anomaly, None, output_shape=(2, 2), diagnostic_mask_threshold=0.65,
    )
    assert source == "diagnostic_anomaly_map"
    assert threshold == pytest.approx(0.65)
    assert mask.tolist() == [[0, 0], [255, 255]]
```

同时新增参数化测试，拒绝 NaN/Inf raw map、畸形但存在的 `pred_mask`、NaN 阈值和不在 `[0,1]` 的阈值。

- [ ] **Step 2: 运行 mask 测试并确认红灯**

```bash
uv run --no-sync pytest --confcutdir=tests/unit \
  tests/unit/capture_data/test_zs32_model_runtime.py -k "patchcore_mask or patchcore_artifact" -q
```

Expected: FAIL，提示 `_build_patchcore_mask`/`PatchcoreArtifacts` 不存在。

- [ ] **Step 3: 实现 artifact 合同和写盘逻辑**

```python
@dataclass(frozen=True, slots=True)
class PatchcoreArtifacts:
    raw_anomaly_map_path: Path
    mask_path: Path
    mask_source: Literal["pred_mask", "diagnostic_anomaly_map"]
    diagnostic_mask_threshold: float | None
    raw_anomaly_map_shape: tuple[int, int]
    mask_shape: tuple[int, int]

@dataclass(frozen=True, slots=True)
class ModelEvidence:
    score: float
    evidence_path: Path
    detections: tuple[dict[str, Any], ...] | None = None
    patchcore_artifacts: PatchcoreArtifacts | None = None
```

`AnomalibPatchcoreBackend.predict(view, crop_path, evidence_path, *, diagnostic_mask_threshold=0.65)` 必须在 resize/normalize 前使用 `np.save(path, raw.astype(np.float32), allow_pickle=False)` 保存原始 2D finite map；mask PNG 保存为 `uint8` 0/255。路径固定为：

```text
evidence/patchcore/<view>.png
evidence/patchcore/raw_maps/<view>.npy
evidence/patchcore/masks/<view>.png
```

存在但无效的 `pred_mask` 直接令该 view 进入现有 fail-closed error path，不允许静默 fallback。

- [ ] **Step 4: 扩展 runtime/CLI 集成测试**

```python
assert manifest["views"]["front"]["patchcore"]["mask_source"] == "pred_mask"
assert manifest["views"]["front"]["patchcore"]["display_only"] is True
assert manifest["views"]["front"]["patchcore"]["roi_xyxy"] == [0, 0, 6, 5]
assert manifest["views"]["front"]["patchcore"]["mask_shape"] == [5, 6]
assert Path(manifest["views"]["front"]["patchcore"]["raw_anomaly_map_path"]).is_file()
```

CLI 新增：

```python
parser.add_argument("--diagnostic-mask-threshold", type=float, default=0.65)
```

在 `_validate_mode()` 中要求 finite 且 `0 <= value <= 1`，再传给 `runtime.run(...)`。该值不得进入 BranchPrediction score、locked threshold 或 Stage18 决策。

- [ ] **Step 5: 运行 Stage32 聚焦回归**

```bash
uv run --no-sync pytest --confcutdir=tests/unit \
  tests/unit/capture_data/test_zs32_model_runtime.py \
  tests/unit/pipeline/test_zs32_multimodel_inference.py -q
uv run --no-sync python pipeline/32_run_zs32_multimodel_inference.py --help
```

Expected: 全部 PASS；help 显示默认 `0.65` 参数。

- [ ] **Step 6: 提交本任务**

```bash
git add capture_data/zs32_model_runtime.py pipeline/32_run_zs32_multimodel_inference.py \
  tests/unit/capture_data/test_zs32_model_runtime.py tests/unit/pipeline/test_zs32_multimodel_inference.py
git commit --only -m "feat: persist ZS32 PatchCore mask artifacts" -- \
  capture_data/zs32_model_runtime.py pipeline/32_run_zs32_multimodel_inference.py \
  tests/unit/capture_data/test_zs32_model_runtime.py tests/unit/pipeline/test_zs32_multimodel_inference.py
```

---

### Task 3: 实现严格八视角结果解析器

**Files:**
- Create: `src/zs32_inspection/dashboard/parser.py`
- Create: `tests/unit/zs32_refactor/dashboard/conftest.py`
- Create: `tests/unit/zs32_refactor/dashboard/test_parser.py`

**Interfaces:**
- Consumes: Task 1 contracts 和 Task 2 `runtime_manifest.json.views`。
- Produces: `load_inspection_result(result_dir: Path) -> InspectionResult`、`DashboardResultError`。

- [ ] **Step 1: 建立同 identity 的八视角 fixture 与失败测试**

fixture 生成八张 40×30 PNG、六个 modeled view 的 branch artifacts、两个 unsupported secondary view，并使全部记录共享 `part_id/capture_session/group_id/hand`。测试必须覆盖稳定顺序、`NG_TEMPLATE` 后 SKIPPED 非零分、未来 secondary AVAILABLE、缺 view、重复 view、identity 冲突、path escape、hash 错误和图像尺寸错误。

```python
def test_parser_returns_eight_views_in_topology_order(eight_view_result_dir: Path) -> None:
    result = load_inspection_result(eight_view_result_dir)
    assert tuple(view.view for view in result.views) == VIEW_ORDER
    assert result.views[3].model_supported is False
    assert result.views[7].model_supported is False


def test_parser_rejects_mixed_capture_identity(eight_view_result_dir: Path) -> None:
    rewrite_view(eight_view_result_dir, "back_secondary", capture_session="other-session")
    with pytest.raises(DashboardResultError, match="capture_session"):
        load_inspection_result(eight_view_result_dir)
```

- [ ] **Step 2: 运行 parser 测试并确认红灯**

```bash
uv run --no-sync pytest --confcutdir=tests/unit \
  tests/unit/zs32_refactor/dashboard/test_parser.py -q
```

Expected: FAIL，提示 parser 不存在。

- [ ] **Step 3: 实现显式 manifest 解析**

解析器只读取 `<result-dir>/runtime_manifest.json` 明确列出的路径，不使用 glob 猜文件。先校验顶层 identity，再按 `VIEW_ORDER` 读取 view；source path 必须存在、hash/尺寸匹配。单个 branch mask 几何异常时保留 source image，并把该 branch 转为 `BranchState.ERROR`；顶层 identity/view 集合错误则拒绝整个结果。

```python
def load_inspection_result(result_dir: Path) -> InspectionResult:
    manifest = _load_json(result_dir / "runtime_manifest.json")
    identity = _parse_identity(manifest)
    records = _require_exact_views(manifest["views"], VIEW_ORDER)
    views = tuple(_parse_view(records[name], identity, result_dir) for name in VIEW_ORDER)
    return InspectionResult(identity, views, str(manifest["machine_status"]), str(manifest.get("reason", "")))
```

- [ ] **Step 4: 运行 parser 测试并确认绿灯**

Run: 与 Step 2 相同。  
Expected: PASS。

- [ ] **Step 5: 提交 parser**

```bash
git add src/zs32_inspection/dashboard/parser.py tests/unit/zs32_refactor/dashboard
git commit --only -m "feat: parse ZS32 eight-view results" -- \
  src/zs32_inspection/dashboard/parser.py tests/unit/zs32_refactor/dashboard
```

---

### Task 4: 实现 mask、YOLO 和 Fusion 证据合成

**Files:**
- Create: `src/zs32_inspection/dashboard/compositor.py`
- Create: `tests/unit/zs32_refactor/dashboard/test_compositor.py`

**Interfaces:**
- Consumes: `ViewResult`, `EvidenceLayer`。
- Produces: `FittedImage`, `ComposedView`, `fit_letterbox`, `place_crop_mask`, `draw_yolo_detections`, `compose_view`。

- [ ] **Step 1: 写几何和像素级失败测试**

```python
def test_place_crop_mask_uses_half_open_roi_and_nearest() -> None:
    mask = np.array([[255, 0], [0, 255]], dtype=np.uint8)
    full = place_crop_mask(mask, (2, 1, 6, 5), source_shape=(6, 8))
    assert full.shape == (6, 8)
    assert full[1, 2]
    assert not full[1, 5]


def test_empty_yolo_detections_do_not_change_pixels(source_view: ViewResult) -> None:
    original = cv2.imread(str(source_view.source_path))
    rendered = draw_yolo_detections(original.copy(), (), source_view.yolo_roi)
    assert np.array_equal(rendered, original)
```

同时加入以下明确断言：

```python
def test_overlay_red_mask_uses_045_alpha() -> None:
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    output = overlay_red_mask(image, np.ones((1, 1), dtype=bool), alpha=0.45)
    assert output[0, 0].tolist() == [0, 0, 115]

def test_secondary_non_original_layer_is_unsupported(secondary_view: ViewResult) -> None:
    composed = compose_view(secondary_view, EvidenceLayer.FUSION)
    assert composed.notice == "暂未接入模型"
    assert np.array_equal(composed.image, cv2.imread(str(secondary_view.source_path)))
```

另外分别断言 letterbox scale/offset、YOLO bbox 只修改框区域，以及 Template 无 mask 时原图像素保持不变。

- [ ] **Step 2: 运行 compositor 测试并确认红灯**

```bash
uv run --no-sync pytest --confcutdir=tests/unit \
  tests/unit/zs32_refactor/dashboard/test_compositor.py -q
```

Expected: FAIL，提示 compositor 函数不存在。

- [ ] **Step 3: 实现纯 NumPy/OpenCV 合成器**

```python
def place_crop_mask(mask: np.ndarray, roi_xyxy: tuple[int, int, int, int], source_shape: tuple[int, int]) -> np.ndarray:
    x1, y1, x2, y2 = roi_xyxy
    if not (0 <= x1 < x2 <= source_shape[1] and 0 <= y1 < y2 <= source_shape[0]):
        raise ValueError("ROI is outside source image")
    resized = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST) > 0
    output = np.zeros(source_shape, dtype=bool)
    output[y1:y2, x1:x2] = resized
    return output

def overlay_red_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    output = image.copy()
    red = np.zeros_like(output)
    red[..., 2] = 255
    output[mask] = cv2.addWeighted(output, 1.0 - alpha, red, alpha, 0)[mask]
    return output
```

Fusion 只叠加 AVAILABLE PatchCore mask 和真实 YOLO bbox。Template 当前只保留原图及文本状态。

- [ ] **Step 4: 运行 compositor 测试并确认绿灯**

Run: 与 Step 2 相同。  
Expected: PASS。

- [ ] **Step 5: 提交 compositor**

```bash
git add src/zs32_inspection/dashboard/compositor.py tests/unit/zs32_refactor/dashboard/test_compositor.py
git commit --only -m "feat: compose ZS32 dashboard evidence" -- \
  src/zs32_inspection/dashboard/compositor.py tests/unit/zs32_refactor/dashboard/test_compositor.py
```

---

### Task 5: 完成离线 OpenCV 看板、reducer 和 headless CLI

**Files:**
- Create: `src/zs32_inspection/dashboard/render.py`
- Create: `src/zs32_inspection/dashboard/app.py`
- Create: `src/zs32_inspection/cli/dashboard.py`
- Create: `pipeline/36_zs32_inspection_dashboard.py`
- Modify: `pyproject.toml:120-155`
- Create: `tests/unit/zs32_refactor/dashboard/test_render.py`
- Create: `tests/unit/zs32_refactor/dashboard/test_app.py`
- Create: `tests/unit/zs32_refactor/runtime/test_dashboard_cli.py`

**Interfaces:**
- Consumes: Tasks 1、3、4。
- Produces: `render_dashboard(state, width=1600, height=920) -> RenderFrame`、`key_to_action`、`mouse_to_action`、`reduce_state`、`run_dashboard`、CLI `zs32-dashboard`。

- [ ] **Step 1: 写 canvas、hit region、reducer 与 CLI 失败测试**

```python
def test_render_has_eight_cards_and_single_detection_action(state: DashboardState) -> None:
    frame = render_dashboard(state)
    assert frame.canvas.shape == (920, 1600, 3)
    assert [hit.action for hit in frame.hit_regions].count("select_view") == 8
    assert [hit.action for hit in frame.hit_regions].count("inspection_action") == 1


def test_s_key_and_detection_button_share_one_action(frame: RenderFrame) -> None:
    assert key_to_action(ord("s")) == Action.INSPECTION
    assert mouse_to_action(frame.inspection_button.center, frame.hit_regions) == Action.INSPECTION
```

CLI 测试要求 `--result-dir` 与 `--live` 互斥，并支持：

```text
--no-gui --save-screenshot /tmp/zs32-dashboard.png
```

- [ ] **Step 2: 运行 UI 测试并确认红灯**

```bash
uv run --no-sync pytest --confcutdir=tests/unit \
  tests/unit/zs32_refactor/dashboard/test_render.py \
  tests/unit/zs32_refactor/dashboard/test_app.py \
  tests/unit/zs32_refactor/runtime/test_dashboard_cli.py -q
```

Expected: FAIL，提示 render/app/CLI 尚不存在。

- [ ] **Step 3: 实现纯 canvas renderer 和 reducer**

迁移旧 demo 的 `_fit_image` 与 Pillow CJK fallback 思路，但不 import 旧模块。renderer 固定绘制 4×2 卡片、顶部 identity/status、五个 layer、一个上下文检测按钮和 Quit；点击卡片进入大图，Escape 返回网格。所有 mouse/key action 都经同一个 reducer。

```python
def inspection_button_label(progress: ProgressRecord | None, running: bool) -> tuple[str, bool]:
    if not running:
        return "开始检测 [S]", True
    if progress and progress.state == "waiting_front":
        return "确认正面并拍摄 [S]", True
    if progress and progress.state == "waiting_back":
        return "确认背面并拍摄 [S]", True
    return "检测运行中", False
```

- [ ] **Step 4: 实现 offline/no-gui CLI**

```python
parser.add_argument("--result-dir", type=Path)
parser.add_argument("--live", action="store_true")
parser.add_argument("--part-id")
parser.add_argument("--no-gui", action="store_true")
parser.add_argument("--save-screenshot", type=Path)
```

离线 no-gui 路径只执行 parser、compositor、renderer 和 `cv2.imwrite`，不得调用 `namedWindow/imshow`。HighGUI 循环使用 `cv2.waitKey(50)`。

- [ ] **Step 5: 运行离线测试、编译和 screenshot smoke**

```bash
uv run --no-sync pytest --confcutdir=tests/unit tests/unit/zs32_refactor/dashboard -q
uv run --no-sync pytest --confcutdir=tests/unit tests/unit/zs32_refactor/runtime/test_dashboard_cli.py -q
uv run --no-sync python -m compileall -q src/zs32_inspection/dashboard \
  src/zs32_inspection/cli/dashboard.py pipeline/36_zs32_inspection_dashboard.py
uv run --no-sync python pipeline/36_zs32_inspection_dashboard.py --help
```

Expected: 全部 PASS，help 显示 offline/live/no-gui 参数。使用同 identity 的测试 fixture 运行 `--no-gui --save-screenshot` 后，PNG 存在且可由 OpenCV 解码。

- [ ] **Step 6: 提交离线看板**

```bash
git add src/zs32_inspection/dashboard src/zs32_inspection/cli/dashboard.py \
  pipeline/36_zs32_inspection_dashboard.py pyproject.toml tests/unit/zs32_refactor
git commit --only -m "feat: add offline ZS32 inspection dashboard" -- \
  src/zs32_inspection/dashboard src/zs32_inspection/cli/dashboard.py \
  pipeline/36_zs32_inspection_dashboard.py pyproject.toml tests/unit/zs32_refactor
```

- [ ] **Step 7: 强制离线验收门**

把 screenshot 的绝对路径和启动命令交给用户；停止实现并等待用户明确确认。真实六视角 diagnostic 和真实八视角 capture 的 identity 不同，严禁拼接。完整八图布局使用同 identity fixture 验证；两个真实产物只能分别验证六路证据和八张原图的解析边界。

---

## STOP CHECKPOINT

Task 1-5 完成后必须停止。只有用户确认离线结果正确，才允许继续以下真机任务。

---

### Task 6: 将 Stage35 采集切换为四相机八视角并透传 secondary

**Files:**
- Modify: `capture_data/zs32_live_commissioning.py:22-180,519-725`
- Modify: `pipeline/35_run_zs32_live_commissioning.py:55-190`
- Modify: `tests/unit/capture_data/test_zs32_live_commissioning.py`
- Modify: `tests/unit/pipeline/test_zs32_live_commissioning_cli.py`

**Interfaces:**
- Consumes: checked-in topology、`VIEW_ORDER/MODELED_VIEWS`、Task 2 manifest schema。
- Produces: 八视角 `CapturedSample`，四相机 capture command，六图 Stage32 command，八视角 final runtime manifest。

- [ ] **Step 1: 把 fixture 改为真实九行 legacy manifest 并写失败测试**

```python
assert tuple(sample.views) == VIEW_ORDER
assert set(build_stage32_command(config, sample, output_dir)) >= {"--front-image", "--back-right-image"}
assert "--front-secondary-image" not in command
assert "--back-secondary-image" not in command
```

覆盖缺 secondary、重复 view、错误 serial、错误 round、混合 session 和 path 越界。

- [ ] **Step 2: 运行 Stage35 测试并确认红灯**

```bash
uv run --no-sync pytest --confcutdir=tests/unit \
  tests/unit/capture_data/test_zs32_live_commissioning.py \
  tests/unit/pipeline/test_zs32_live_commissioning_cli.py -q
```

Expected: FAIL，当前命令仍调用三相机 collector，loader 仍要求六图。

- [ ] **Step 3: 实现四相机 capture command 和八视角 loader**

```python
command = [
    sys.executable, str(repo_root / "pipeline/zs32_bootstrap_capture.py"),
    "--topology", str(config.topology_path),
    "--root", str(capture_run_root),
    "--capture-session", config.run_id,
    "--legacy-layout", "--hand", "right", "--label", "normal",
    "--part-id", config.part_id, "--group-count", "1", "--images-per-group", "1",
    "--manual-load", "--hdr",
]
```

serial 只能来自 topology，不再接受 front/left/right serial override。`build_stage32_command()` 只循环 `MODELED_VIEWS`。

- [ ] **Step 4: 原子补全八视角 runtime manifest**

新增：

```python
def publish_eight_view_runtime_manifest(output_dir: Path, sample: CapturedSample) -> Path:
    path = output_dir / "runtime_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_runtime_identity(payload, sample)
    payload["views"] = {
        view: build_runtime_view_record(sample.views[view], payload.get("views", {}).get(view))
        for view in VIEW_ORDER
    }
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
```

函数先校验现有 manifest 的 part/session/group/hand，再加入八个 view。两个 secondary 写 `model_supported=false`、`model_status="unsupported"`、`branches={}`，且不得改变 `machine_status/errors/missing_required_evidence`。模板短路结果也必须补齐八视角。

- [ ] **Step 5: 运行四相机/Stage35 回归并提交**

```bash
uv run --no-sync pytest --confcutdir=tests/unit \
  tests/unit/zs32_refactor/domain/test_topology.py \
  tests/unit/zs32_refactor/capture_data/test_legacy_dataset_store.py \
  tests/unit/zs32_refactor/capture_data/test_bootstrap_capture_cli.py \
  tests/unit/capture_data/test_zs32_live_commissioning.py \
  tests/unit/pipeline/test_zs32_live_commissioning_cli.py -q
```

Expected: PASS。随后只提交上述四个 Stage35 文件及相关测试。

---

### Task 7: 把真实采集/推理阶段接入 progress/control 协议

**Files:**
- Modify: `src/zs32_inspection/capture/bootstrap.py`
- Modify: `src/zs32_inspection/cli/bootstrap_capture.py:69-255`
- Modify: `pipeline/32_run_zs32_multimodel_inference.py:45-470`
- Modify: `capture_data/zs32_live_commissioning.py:180-620`
- Create: `tests/unit/capture_data/test_zs32_live_progress.py`
- Modify: `tests/unit/zs32_refactor/capture_data/test_bootstrap_capture_cli.py`
- Modify: `tests/unit/pipeline/test_zs32_multimodel_inference.py`

**Interfaces:**
- Consumes: Task 1 progress/control transport。
- Produces: `--progress-json`、`--control-json`，真实阶段状态和无 TTY round confirmation。

- [ ] **Step 1: 写状态序列和 confirmation_id 失败测试**

```python
assert states == [
    "waiting_front", "capturing_front", "waiting_back", "capturing_back",
    "running_template", "running_patchcore_yolo", "running_fusion", "complete",
]
```

模板短路序列只允许到 `running_template, complete`；不得伪造后续阶段。测试还要证明 wrong/stale/duplicate confirmation 不触发 capture，正确 token 在无 TTY stream 下触发一次。

- [ ] **Step 2: 运行 progress 测试并确认红灯**

```bash
uv run --no-sync pytest --confcutdir=tests/unit \
  tests/unit/capture_data/test_zs32_live_progress.py \
  tests/unit/zs32_refactor/capture_data/test_bootstrap_capture_cli.py \
  tests/unit/pipeline/test_zs32_multimodel_inference.py -q
```

Expected: FAIL，CLI 和 observer 尚无 progress/control 参数。

- [ ] **Step 3: 在真实边界发送状态**

`BootstrapCaptureService` 在每轮确认前写 `waiting_*`，确认成功后、采集开始前写 `capturing_*`。每次 waiting 使用新的 `uuid4().hex`。bootstrap CLI 在指定 `--control-json` 时使用文件确认器，不检查 stdin/stdout TTY。

Stage32 分别在 `_template_gate()`、`runtime.run()`、`_run_strict_fusion()` 之前写 `running_template/running_patchcore_yolo/running_fusion`。Stage35 验证最终合同后写 `complete`；任何异常先写 `failed`，progress 写失败本身按执行错误处理。

- [ ] **Step 4: 运行回归并提交**

Run: 与 Step 2 相同。  
Expected: PASS。提交本任务列出的实现和测试文件。

---

### Task 8: 接入 Stage35 process group 和上下文检测按钮

**Files:**
- Create: `src/zs32_inspection/dashboard/live.py`
- Modify: `src/zs32_inspection/dashboard/app.py`
- Modify: `src/zs32_inspection/cli/dashboard.py`
- Create: `tests/unit/zs32_refactor/dashboard/test_live.py`
- Modify: `tests/unit/zs32_refactor/dashboard/test_app.py`
- Modify: `tests/unit/zs32_refactor/runtime/test_dashboard_cli.py`

**Interfaces:**
- Consumes: Task 1 control、Task 5 renderer/reducer、Task 7 Stage35 CLI。
- Produces: `Stage35Controller.start/poll/confirm/close` 和 live Start/Confirm 状态机。

- [ ] **Step 1: 写 process-group 和按钮状态失败测试**

```python
controller.start("part-1", progress_path, control_path)
with pytest.raises(RuntimeError, match="already running"):
    controller.start("part-1", progress_path, control_path)
controller.confirm()
assert read_control(control_path).round == "front"
controller.close()
fake_killpg.assert_called_once_with(fake_process.pid, signal.SIGTERM)
```

覆盖非零退出码、failed progress、wait timeout 后 SIGKILL、非 owned PID 不操作、waiting 外 confirm no-op。

- [ ] **Step 2: 运行 live 测试并确认红灯**

```bash
uv run --no-sync pytest --confcutdir=tests/unit \
  tests/unit/zs32_refactor/dashboard/test_live.py \
  tests/unit/zs32_refactor/dashboard/test_app.py \
  tests/unit/zs32_refactor/runtime/test_dashboard_cli.py -q
```

Expected: FAIL，`Stage35Controller` 不存在。

- [ ] **Step 3: 实现 owned process group**

```python
self.process = subprocess.Popen(
    command,
    start_new_session=True,
    stdout=log_file,
    stderr=subprocess.STDOUT,
)

def close(self, timeout: float = 5.0) -> None:
    if self.process is None or self.process.poll() is not None:
        return
    os.killpg(self.process.pid, signal.SIGTERM)
    try:
        self.process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(self.process.pid, signal.SIGKILL)
        self.process.wait(timeout=timeout)
```

`confirm()` 只在 progress state 为 `waiting_front/back` 且存在 `confirmation_id` 时写 control JSON。

- [ ] **Step 4: 运行 live 回归并提交**

Run: 与 Step 2 相同。  
Expected: PASS。提交 live/app/CLI 及对应测试。

---

### Task 9: 文档、总回归与单件真机 smoke

**Files:**
- Modify: `pipeline/README.md`
- Modify: `docs/ZS32_FOUR_CAMERA_END_TO_END_README.md`
- Modify: `AGENTS_MEMORY.md`
- Modify: `pipeline/AGENTS_MEMORY.md`

**Interfaces:**
- Consumes: Tasks 1-8 全部命令和产物合同。
- Produces: 可复制的 offline/live 命令、真机验收记录和更新后的目录记忆。

- [ ] **Step 1: 写准确运行命令和边界说明**

```bash
uv run --no-sync python pipeline/36_zs32_inspection_dashboard.py \
  --result-dir /absolute/eight-view-result

PYTHONPATH=/opt/MVS/Samples/64/Python/MvImport:${PYTHONPATH:-} \
uv run --no-sync python pipeline/36_zs32_inspection_dashboard.py \
  --live --part-id live_part_001
```

文档必须写明 secondary 暂无模型、diagnostic mask 不参与判定、按钮三种标签、无 TTY control 协议、`NG_TEMPLATE` 会短路，以及 commissioning 不等于生产放行。

- [ ] **Step 2: 运行总回归和静态检查**

```bash
uv run --no-sync pytest --confcutdir=tests/unit \
  tests/unit/zs32_refactor/dashboard \
  tests/unit/zs32_refactor/runtime/test_dashboard_cli.py \
  tests/unit/zs32_refactor/domain/test_topology.py \
  tests/unit/zs32_refactor/capture_data/test_legacy_dataset_store.py \
  tests/unit/zs32_refactor/capture_data/test_bootstrap_capture_cli.py \
  tests/unit/capture_data/test_zs32_model_runtime.py \
  tests/unit/capture_data/test_zs32_live_commissioning.py \
  tests/unit/capture_data/test_zs32_live_progress.py \
  tests/unit/pipeline/test_zs32_multimodel_inference.py \
  tests/unit/pipeline/test_zs32_live_commissioning_cli.py -q
uv run --no-sync python -m compileall -q src/zs32_inspection/dashboard \
  capture_data/zs32_model_runtime.py capture_data/zs32_live_commissioning.py \
  pipeline/32_run_zs32_multimodel_inference.py \
  pipeline/35_run_zs32_live_commissioning.py pipeline/36_zs32_inspection_dashboard.py
uv run --no-sync python pipeline/36_zs32_inspection_dashboard.py --help
git diff --check
```

Expected: 全部测试 PASS、compileall 无输出、help 成功、diff check 无输出。

- [ ] **Step 3: 执行一次真机验收**

验收条件必须全部满足：一个 right-hand part；legacy manifest 恰好八个 image rows 加一个 complete sample row；八张 PNG 同一 identity；Stage32 command 只有六主视角；runtime manifest 有八个 view；两个 secondary 均为 unsupported；正背面按钮确认各消费一次正确 `confirmation_id`；看板显示最终真实状态和八张图。

- [ ] **Step 4: 更新记忆并提交文档**

把实际测试数量、产物绝对路径、manifest identity、运行命令、真机是否执行成功写入 `AGENTS_MEMORY.md` 和 `pipeline/AGENTS_MEMORY.md`，不得把未执行的硬件测试写成通过。只提交本任务实际修改的文档与记忆文件。

---

## 实施完成标准

- 离线阶段在 Task 5 获得用户明确验收。
- 真机阶段形成一个同 identity 的八图、六主视角模型证据结果。
- secondary 没有权重时始终中性显示；将来加入两组模型记录无需修改 GUI 合同。
- raw anomaly map、mask、ROI、hash 和来源均可追溯。
- OpenCV 主线程不运行 GPU 推理；退出不会遗留 Stage35、采集器或 Stage32 进程。
- OK、NG、REVIEW、运行中和执行错误的显示语义与 runtime 一致。
