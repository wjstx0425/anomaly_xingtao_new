# ZS32 Label Studio Local Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible tool that exposes the 660 ZS32 six-view defect PNGs to Label Studio Local Files through hard links, with a traceable manifest and a one-class `defect` labeling interface.

**Architecture:** A focused `capture_data` module discovers and validates the fixed ZS32 capture hierarchy, writes a safe hard-link staging tree plus CSV/XML/README artifacts, and fails closed on malformed or incomplete physical groups. A thin numbered pipeline wrapper forwards CLI arguments. Unit tests exercise discovery, group completeness, hard-link identity, metadata output, collision handling, and overwrite safety before the tool is run against the real ignored dataset.

**Tech Stack:** Python 3.10+, standard library (`argparse`, `csv`, `dataclasses`, `os`, `pathlib`, `re`, `shutil`), pytest, Label Studio Local Files.

## Global Constraints

- Do not copy, move, edit, or delete any source image under `dataset/left` or `dataset/right`.
- Stage only defect PNGs; normal images do not enter the manual annotation project.
- Use one RectangleLabels value, exactly `defect`.
- The only supported hands are `left` and `right`.
- The only supported views are `front`, `front_left`, `front_right`, `back`, `back_left`, and `back_right`.
- The only supported defect types in this batch are `deform`, `less`, and `others`.
- Every `(hand, session_id, defect_type, group_id)` must contain exactly the six supported views.
- Hard-link creation failures must fail explicitly; never fall back to copying image bytes.
- A non-empty output root must fail unless `--overwrite` is explicit, and overwrite may only remove the configured staging root.
- Keep `dataset/zs32_yolo_labeling` ignored by Git; commit source, tests, documentation, and memory only.

---

## File Structure

- Create `capture_data/prepare_zs32_label_studio.py`: discovery, path parsing, physical-group validation, safe output preparation, hard-link creation, and manifest/config/README generation.
- Create `pipeline/27_prepare_zs32_label_studio.py`: thin numbered wrapper around the core module.
- Create `tests/unit/capture_data/test_prepare_zs32_label_studio.py`: focused unit and failure-mode coverage.
- Modify `tests/unit/pipeline/test_pipeline_wrappers.py`: verify stage 27 argument forwarding surface.
- Modify `pipeline/README.md`: document stage 27, exact build command, exact Label Studio start command, Source Storage settings, and visible-only bbox semantics.
- Modify `AGENTS_MEMORY.md`: record the final artifact paths, counts, commands, and verification results after the real build succeeds.
- Generate ignored artifacts under `dataset/zs32_yolo_labeling/`: hard-linked images, `labeling_manifest.csv`, `label_studio_config.xml`, and `README.md`.

### Task 1: Build and test the hard-link staging core

**Files:**
- Create: `capture_data/prepare_zs32_label_studio.py`
- Create: `tests/unit/capture_data/test_prepare_zs32_label_studio.py`

**Interfaces:**
- Produces: `LabelingImage`, `discover_labeling_images(dataset_root: Path) -> list[LabelingImage]`, and `prepare_label_studio_dataset(dataset_root: Path, output_root: Path, repo_root: Path, *, overwrite: bool = False) -> dict[str, int]`.
- Consumes: the fixed source layout `<dataset_root>/<hand>/<view>/defect/<defect_type>/<session>/images/*.png`.

- [ ] **Step 1: Write failing discovery and completeness tests**

Create `tests/unit/capture_data/test_prepare_zs32_label_studio.py` with a file-path loader matching the existing capture-data tests, a helper that creates six minimal PNG-signature files, and these assertions:

```python
def test_discover_labeling_images_parses_complete_six_view_group(tmp_path: Path) -> None:
    module = load_module()
    dataset_root = tmp_path / "dataset"
    for view in module.VIEWS:
        path = (
            dataset_root
            / "left"
            / view
            / "defect"
            / "deform"
            / "20260711_175751_318434"
            / "images"
            / f"left_{view}_defect_deform_zs32_left_defect_group001_000001_fused.png"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x89PNG\r\n\x1a\n")

    images = module.discover_labeling_images(dataset_root)

    assert len(images) == 6
    assert {image.view for image in images} == set(module.VIEWS)
    assert {image.group_id for image in images} == {"group001"}
    assert {image.sample_id for image in images} == {
        "left/20260711_175751_318434/deform/group001"
    }


def test_discover_labeling_images_rejects_incomplete_group(tmp_path: Path) -> None:
    module = load_module()
    dataset_root = tmp_path / "dataset"
    path = (
        dataset_root
        / "right"
        / "front"
        / "defect"
        / "less"
        / "session-a"
        / "images"
        / "right_front_defect_less_group001_000001_fused.png"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n")

    with pytest.raises(ValueError, match="must contain exactly six views"):
        module.discover_labeling_images(dataset_root)
```

- [ ] **Step 2: Run discovery tests and confirm the expected failure**

Run:

```bash
uv run pytest tests/unit/capture_data/test_prepare_zs32_label_studio.py -v
```

Expected: FAIL because `capture_data/prepare_zs32_label_studio.py` does not exist.

- [ ] **Step 3: Implement strict discovery and metadata parsing**

Implement these exact public definitions in `capture_data/prepare_zs32_label_studio.py`:

```python
HANDS = ("left", "right")
VIEWS = ("front", "front_left", "front_right", "back", "back_left", "back_right")
DEFECT_TYPES = ("deform", "less", "others")
GROUP_PATTERN = re.compile(r"_(group\d{3})_")


@dataclass(frozen=True)
class LabelingImage:
    source_path: Path
    hand: str
    view: str
    defect_type: str
    session_id: str
    group_id: str

    @property
    def sample_id(self) -> str:
        return f"{self.hand}/{self.session_id}/{self.defect_type}/{self.group_id}"


def discover_labeling_images(dataset_root: Path) -> list[LabelingImage]:
    images: list[LabelingImage] = []
    pattern = "*/" + "*/defect/*/*/images/*.png"
    for source_path in sorted(dataset_root.glob(pattern)):
        relative = source_path.relative_to(dataset_root)
        hand, view, label, defect_type, session_id, images_dir, _ = relative.parts
        if hand not in HANDS or view not in VIEWS or label != "defect":
            raise ValueError(f"Unsupported ZS32 labeling path: {source_path}")
        if defect_type not in DEFECT_TYPES or images_dir != "images":
            raise ValueError(f"Unsupported ZS32 defect path: {source_path}")
        match = GROUP_PATTERN.search(source_path.name)
        if match is None:
            raise ValueError(f"Could not parse group_id from: {source_path.name}")
        images.append(
            LabelingImage(source_path, hand, view, defect_type, session_id, match.group(1)),
        )
    if not images:
        raise RuntimeError(f"No ZS32 defect PNGs found under: {dataset_root}")
    grouped: dict[str, set[str]] = {}
    for image in images:
        grouped.setdefault(image.sample_id, set()).add(image.view)
    expected = set(VIEWS)
    incomplete = {sample: sorted(views) for sample, views in grouped.items() if views != expected}
    if incomplete:
        raise ValueError(f"Every physical group must contain exactly six views: {incomplete}")
    return images
```

- [ ] **Step 4: Run discovery tests and confirm they pass**

Run the same focused pytest command. Expected: `2 passed`.

- [ ] **Step 5: Write failing output, manifest, collision, and overwrite tests**

Add tests that call `prepare_label_studio_dataset()` and assert:

```python
summary = module.prepare_label_studio_dataset(
    dataset_root=dataset_root,
    output_root=tmp_path / "labeling",
    repo_root=tmp_path,
)
assert summary == {"total": 6, "left": 6, "right": 0, "groups": 1}
target = tmp_path / "labeling/images/left/front/deform" / source.name
assert target.stat().st_dev == source.stat().st_dev
assert target.stat().st_ino == source.stat().st_ino
with (tmp_path / "labeling/labeling_manifest.csv").open(newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))
assert len(rows) == 6
assert rows[0]["sample_id"] == "left/20260711_175751_318434/deform/group001"
assert "<Label value=\"defect\"" in (
    tmp_path / "labeling/label_studio_config.xml"
).read_text(encoding="utf-8")
assert "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true" in (
    tmp_path / "labeling/README.md"
).read_text(encoding="utf-8")
```

Also assert that a second run without `overwrite=True` raises `ValueError`, an unsafe output root is rejected, and duplicate target names raise a collision error before any manifest is published.

- [ ] **Step 6: Run the expanded tests and confirm the output API is missing**

Run the focused test file. Expected: FAIL with `AttributeError` for `prepare_label_studio_dataset`.

- [ ] **Step 7: Implement safe preparation and artifact writers**

Implement private helpers with these exact signatures:

```python
def _prepare_output_root(
    output_root: Path,
    *,
    overwrite: bool,
    forbidden_roots: Sequence[Path],
) -> None: ...

def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None: ...

def _write_label_config(path: Path) -> None: ...

def _write_readme(path: Path, output_root: Path) -> None: ...
```

Then implement:

```python
def prepare_label_studio_dataset(
    dataset_root: Path,
    output_root: Path,
    repo_root: Path,
    *,
    overwrite: bool = False,
) -> dict[str, int]:
    images = discover_labeling_images(dataset_root)
    _prepare_output_root(
        output_root,
        overwrite=overwrite,
        forbidden_roots=(repo_root, dataset_root, dataset_root / "left", dataset_root / "right"),
    )
    manifest_rows: list[dict[str, str]] = []
    used_targets: set[Path] = set()
    for image in images:
        target = output_root / "images" / image.hand / image.view / image.defect_type / image.source_path.name
        if target in used_targets:
            raise ValueError(f"Duplicate Label Studio target path: {target}")
        used_targets.add(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(image.source_path, target)
        if target.stat().st_dev != image.source_path.stat().st_dev or target.stat().st_ino != image.source_path.stat().st_ino:
            raise RuntimeError(f"Output is not a hard link to source: {target}")
        manifest_rows.append(
            {
                "labeling_path": str(target.relative_to(repo_root)),
                "source_path": str(image.source_path.relative_to(repo_root)),
                "hand": image.hand,
                "view": image.view,
                "defect_type": image.defect_type,
                "session_id": image.session_id,
                "group_id": image.group_id,
                "sample_id": image.sample_id,
            },
        )
    _write_manifest(output_root / "labeling_manifest.csv", manifest_rows)
    _write_label_config(output_root / "label_studio_config.xml")
    _write_readme(output_root / "README.md", output_root)
    return {
        "total": len(images),
        "left": sum(image.hand == "left" for image in images),
        "right": sum(image.hand == "right" for image in images),
        "groups": len({image.sample_id for image in images}),
    }
```

Use the same forbidden overwrite targets as `capture_data/prepare_yolo_dataset.py`: `/`, the current user home, the current working directory, `dataset_root`, and either source hand directory. Write the manifest atomically through a temporary sibling followed by `Path.replace()` so a failed build does not publish a partial CSV.

- [ ] **Step 8: Run the complete core test file**

Run:

```bash
uv run pytest tests/unit/capture_data/test_prepare_zs32_label_studio.py -v
```

Expected: all tests PASS.

- [ ] **Step 9: Run style checks for the new module and test**

Run:

```bash
uv run ruff check capture_data/prepare_zs32_label_studio.py tests/unit/capture_data/test_prepare_zs32_label_studio.py
uv run ruff format --check capture_data/prepare_zs32_label_studio.py tests/unit/capture_data/test_prepare_zs32_label_studio.py
```

Expected: both commands exit `0`.

- [ ] **Step 10: Commit the tested core**

```bash
git add capture_data/prepare_zs32_label_studio.py tests/unit/capture_data/test_prepare_zs32_label_studio.py
git commit -m "feat: prepare ZS32 Label Studio staging data"
```

### Task 2: Add the pipeline entrypoint and operational documentation

**Files:**
- Create: `pipeline/27_prepare_zs32_label_studio.py`
- Modify: `tests/unit/pipeline/test_pipeline_wrappers.py`
- Modify: `pipeline/README.md`

**Interfaces:**
- Consumes: `prepare_label_studio_dataset()` from Task 1.
- Produces: `build_parser() -> argparse.ArgumentParser`, a stable command, `uv run python pipeline/27_prepare_zs32_label_studio.py`, and exact Label Studio Local Files setup instructions.

- [ ] **Step 1: Write the failing stage-27 parser test**

Append:

```python
def test_prepare_zs32_label_studio_parser_accepts_paths_and_overwrite(tmp_path: Path) -> None:
    wrapper = _load_module(
        "pipeline_prepare_zs32_label_studio",
        "pipeline/27_prepare_zs32_label_studio.py",
    )
    args = wrapper.build_parser().parse_args(
        [
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-root",
            str(tmp_path / "labeling"),
            "--overwrite",
        ],
    )
    assert args.dataset_root == tmp_path / "dataset"
    assert args.output_root == tmp_path / "labeling"
    assert args.overwrite is True
```

- [ ] **Step 2: Run the parser test and confirm stage 27 is missing**

Run:

```bash
uv run pytest tests/unit/pipeline/test_pipeline_wrappers.py::test_prepare_zs32_label_studio_parser_accepts_paths_and_overwrite -v
```

Expected: FAIL because `pipeline/27_prepare_zs32_label_studio.py` does not exist.

- [ ] **Step 3: Implement the thin stage-27 wrapper**

Follow the stage-21 import-path pattern. Import `argparse`, then define:

```python
def build_parser() -> argparse.ArgumentParser:
    """Build the ZS32 Label Studio preparation parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "dataset" / "zs32_yolo_labeling",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser
```

Use this `main()`:

```python
def main() -> None:
    """Prepare the ZS32 Label Studio Local Files staging directory."""
    args = build_parser().parse_args()
    summary = prepare_label_studio_dataset(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        repo_root=REPO_ROOT,
        overwrite=args.overwrite,
    )
    print(f"Label Studio directory: {args.output_root}")
    print(f"Manifest: {args.output_root / 'labeling_manifest.csv'}")
    print(f"Summary: {summary}")
```

Expose only `--dataset-root`, `--output-root`, and `--overwrite`.

- [ ] **Step 4: Run parser and core tests**

Run:

```bash
uv run pytest tests/unit/pipeline/test_pipeline_wrappers.py::test_prepare_zs32_label_studio_parser_accepts_paths_and_overwrite tests/unit/capture_data/test_prepare_zs32_label_studio.py -v
```

Expected: all selected tests PASS.

- [ ] **Step 5: Document stage 27 and the exact workflow**

Add stage 27 to the pipeline table and a section containing these commands:

```bash
uv run python pipeline/27_prepare_zs32_label_studio.py

conda activate label-studio
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/home/yunjing/anomalib/dataset/zs32_yolo_labeling
label-studio start
```

Document Source Storage as:

```text
Storage type: Local Files
Absolute local path: /home/yunjing/anomalib/dataset/zs32_yolo_labeling/images
Import method: Files
File Filter Regex: .*\.png$
Expected tasks: 660
```

Also state that the user pastes `dataset/zs32_yolo_labeling/label_studio_config.xml` into the labeling interface and submits an empty annotation when a defect is not visible in that particular view.

- [ ] **Step 6: Run wrapper/style/document checks**

Run:

```bash
uv run ruff check capture_data/prepare_zs32_label_studio.py pipeline/27_prepare_zs32_label_studio.py tests/unit/capture_data/test_prepare_zs32_label_studio.py tests/unit/pipeline/test_pipeline_wrappers.py
uv run ruff format --check capture_data/prepare_zs32_label_studio.py pipeline/27_prepare_zs32_label_studio.py tests/unit/capture_data/test_prepare_zs32_label_studio.py
git diff --check
```

Expected: every command exits `0`.

- [ ] **Step 7: Commit wrapper and documentation**

```bash
git add pipeline/27_prepare_zs32_label_studio.py tests/unit/pipeline/test_pipeline_wrappers.py pipeline/README.md
git commit -m "docs: add ZS32 Label Studio workflow"
```

### Task 3: Build and verify the real 660-image labeling directory

**Files:**
- Generate ignored: `dataset/zs32_yolo_labeling/**`
- Modify: `AGENTS_MEMORY.md`

**Interfaces:**
- Consumes: the stage-27 command from Task 2 and the real `dataset/left`, `dataset/right` trees.
- Produces: the ready-to-sync Label Studio directory and durable verified operating notes.

- [ ] **Step 1: Dry-run discovery against real data without writing output**

Use a direct read-only check:

```bash
uv run python -c "from pathlib import Path; from capture_data.prepare_zs32_label_studio import discover_labeling_images; x=discover_labeling_images(Path('dataset')); print(len(x), len({i.sample_id for i in x}))"
```

Expected output:

```text
660 110
```

- [ ] **Step 2: Build the real staging directory**

Run once without `--overwrite`:

```bash
uv run python pipeline/27_prepare_zs32_label_studio.py
```

Expected summary:

```text
{'total': 660, 'left': 486, 'right': 174, 'groups': 110}
```

If the output already exists from a failed or obsolete run, inspect it first and only then rerun with `--overwrite`.

- [ ] **Step 3: Verify counts, manifest identity, and hard links**

Run a read-only verification command that loads `labeling_manifest.csv` and asserts:

```python
assert len(rows) == 660
assert Counter(row["hand"] for row in rows) == {"left": 486, "right": 174}
assert len({row["sample_id"] for row in rows}) == 110
for row in rows:
    source = repo_root / row["source_path"]
    target = repo_root / row["labeling_path"]
    assert source.is_file() and target.is_file()
    assert (source.stat().st_dev, source.stat().st_ino) == (
        target.stat().st_dev,
        target.stat().st_ino,
    )
```

Expected: command exits `0` and prints the same four summary counts.

- [ ] **Step 4: Check Label Studio artifacts and ignored-data boundary**

Run:

```bash
test -f dataset/zs32_yolo_labeling/label_studio_config.xml
test -f dataset/zs32_yolo_labeling/README.md
git status --short --ignored dataset/zs32_yolo_labeling
```

Expected: both file tests exit `0`; Git reports the generated root only as ignored (`!!`), never staged.

- [ ] **Step 5: Run the full relevant test and syntax suite**

Run:

```bash
uv run pytest tests/unit/capture_data/test_prepare_zs32_label_studio.py tests/unit/pipeline/test_pipeline_wrappers.py -v
uv run python -m py_compile capture_data/prepare_zs32_label_studio.py pipeline/27_prepare_zs32_label_studio.py
git diff --check
```

Expected: pytest passes, compilation exits `0`, and diff check is clean.

- [ ] **Step 6: Update the repository memory with measured results**

Append a dated `ZS32 Label Studio local storage` entry to `AGENTS_MEMORY.md` containing:

```text
- Builder: capture_data/prepare_zs32_label_studio.py
- Wrapper: pipeline/27_prepare_zs32_label_studio.py
- Output: dataset/zs32_yolo_labeling
- Verified: 660 hard-linked PNGs, 110 physical groups, left=486, right=174
- Label Studio document root: /home/yunjing/anomalib/dataset/zs32_yolo_labeling
- Source storage path: /home/yunjing/anomalib/dataset/zs32_yolo_labeling/images
- Import method: Files; filter: .*\.png$
- Label schema: one rectangle class, defect
```

Record actual test counts and commands rather than anticipated results.

- [ ] **Step 7: Commit the memory update without generated data**

```bash
git add AGENTS_MEMORY.md
git diff --cached --stat
git commit -m "docs: record ZS32 Label Studio dataset"
```

Expected staged files: only `AGENTS_MEMORY.md`; nothing under `dataset/`.

- [ ] **Step 8: Hand off the exact Label Studio setup**

Report the generated paths and these UI settings to the user:

```text
Absolute local path: /home/yunjing/anomalib/dataset/zs32_yolo_labeling/images
Import method: Files
File Filter Regex: .*\.png$
Expected tasks after sync: 660
```

Do not start Label Studio or alter its database automatically; the user already manages the `label-studio` Conda environment and will perform the UI sync.
