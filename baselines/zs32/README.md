# ZS32 baseline pointers

本目录只保存由 Linux Phase 0 bundle 生成的轻量 pointer JSON，不保存图片、模型、模板、阈值或运行输出。

pointer 示例：

```json
{
  "schema_version": 1,
  "freeze_id": "zs32-right-3cam-legacy-YYYYMMDD-r1",
  "bundle_root_sha256": "REPLACE_WITH_LINUX_FREEZE_ROOT",
  "git_commit": "REPLACE_WITH_CLEAN_LINUX_COMMIT",
  "scope": {
    "product": "ZS32",
    "hands": ["right"],
    "topology_id": "zs32-3cam-double-side-v1",
    "left_roi_status": "pending"
  },
  "storage_reference": "s3://REPLACE_BUCKET/zs32/sha256/REPLACE_WITH_BUNDLE_ROOT_SHA256"
}
```

pointer 必须在 bundle 通过 `tools/zs32_phase0.py verify` 后生成，并在独立 Git commit 中提交。`storage_reference` 必须是内容寻址且不可覆盖的引用，并且引用文本中必须包含完整 `bundle_root_sha256`；不得填写 `current` symlink、可变目录、带 token 的 URL、SSH 私钥位置或任何访问凭证。pointer 顶层和 `scope` 字段必须与示例恰好一致，多余字段也会被拒绝。

Phase 0 的 Git 交互分两步：首先 push 源码 commit 供 Linux 执行 freeze；
Linux 返回 trusted root 后，再单独提交 pointer。之后 Linux 必须 pull 该
pointer commit，并使用 `--pointer` 再次 strict verify。bundle 本体、原图、权重、
模板、阈值和验证日志均不进入 GitHub。
