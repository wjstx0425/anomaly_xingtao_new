# BMW Left Demo Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use test-driven-development to implement this task.

**Goal:** 发布一个现有 BMW 八视图 Demo 可直接加载的左手实验配置。

**Architecture:** 新增专用发布器，在独立组合 run 中链接左手 Template、左手 EfficientAD 和共享 YOLO，再原子写入 SHA 绑定的左手配置。保持现有 Demo loader 与旧配置不变。

**Tech Stack:** Python 3.13、JSON、pathlib、symlink、pytest、uv。

## Global Constraints

- 不修改 `bmw_left_normal_20260814_models_v3` 训练源目录。
- 不修改旧 V5/V6 配置或通用 Demo schema。
- 光痕 engine 必须为 `tracked_profile_v3_manual_rotated_candidate`，不得设置 `weak_row_score_override`。
- Template 配置段省略；EfficientAD 必须绑定左手 mask、component policy 和 component threshold。
- YOLO 必须复用 SHA-256 `0e9591f2fa2487ad12000847f1d80137901ed69989e0e95cb5c8907b95ba3913` 的共享 checkpoint。

---

### Task 1: 发布左手组合 run 与 Demo 配置

**Files:**
- Create: `src/bmw_inspection/lab/left_demo_publisher.py`
- Create: `pipeline/bmw_lab_prepare_left_20260814_demo.py`
- Create: `tests/unit/bmw_inspection/lab/test_left_demo_publisher.py`
- Generate: `configs/bmw/experiments/bmw_eight_view_demo_left_normal_20260814_v1.json`
- Generate: `results/bmw_lab_one_click/bmw_left_normal_20260814_demo_v1/`

**Interfaces:**
- Produces: `publish_left_demo(repo_root: Path, *, output_run: Path, output_config: Path) -> dict[str, object]`。
- CLI 无参数时使用上述固定实验资产和输出路径。

- [ ] 写失败测试：断言发布器创建三个目录链接、composition receipt，并生成能被 `load_demo_config()` 加载的配置；断言不覆盖已有目标。
- [ ] 运行 `uv run --no-sync pytest -q tests/unit/bmw_inspection/lab/test_left_demo_publisher.py`，确认因模块缺失失败。
- [ ] 实现最小发布器和 CLI，不扩展通用 loader。
- [ ] 重新运行该测试并确认通过。
- [ ] 运行 CLI 生成真实组合 run 与配置。
- [ ] 用 `load_demo_config()` 加载真实配置，确认左手 ROI/模型/光痕与共享 YOLO 路径正确。
- [ ] 仅提交本任务文件，保留现有无关脏改。
