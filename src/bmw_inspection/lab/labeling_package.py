"""Build a portable BMW ROI labeling handoff package."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER, _atomic_publish_noreplace


_QUEUE_FIELDS = (
    "sample_id",
    "physical_part_id",
    "session_id",
    "group_id",
    "view_id",
    "camera_serial",
    "source_path",
    "source_sha256",
    "source_class",
    "business_label",
    "split",
    "crop_path",
    "expected_label_filename",
)
_PACKAGE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_TASK_FIELDS = (
    "task_index",
    "image_filename",
    "image_sha256",
    *_QUEUE_FIELDS,
)
_CROP_FIELDS = _QUEUE_FIELDS[:-2] + (
    "roi_x1",
    "roi_y1",
    "roi_x2",
    "roi_y2",
    "crop_path",
    "crop_sha256",
    "crop_width",
    "crop_height",
)
_REFERENCE_FIELDS = (
    "reference_index",
    "image_filename",
    "image_sha256",
    *_CROP_FIELDS,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_queue(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != _QUEUE_FIELDS:
            raise ValueError("annotation queue header differs from the frozen BMW schema")
        rows = list(reader)
    if not rows:
        raise ValueError("annotation queue must not be empty")
    return rows


def _resolve_crop(training_root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"crop path must stay inside the training release: {relative_path}")
    source = (training_root / candidate).resolve()
    try:
        source.relative_to(training_root)
    except ValueError as error:
        raise ValueError(f"crop path escapes the training release: {relative_path}") from error
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"missing crop image: {source}")
    return source


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _select_normal_references(
    training_root: Path,
    count_per_view: int,
) -> tuple[list[dict[str, str]], list[str]]:
    if count_per_view == 0:
        return [], []
    manifest_path = training_root / "manifests/crop_manifest.csv"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"crop manifest does not exist: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != _CROP_FIELDS:
            raise ValueError("crop manifest header differs from the frozen BMW schema")
        normal_rows = [row for row in reader if row["source_class"] == "normal" and row["split"] == "train"]

    captures_by_part: dict[str, dict[tuple[str, str], dict[str, dict[str, str]]]] = {}
    for row in normal_rows:
        part_captures = captures_by_part.setdefault(row["physical_part_id"], {})
        capture = part_captures.setdefault((row["session_id"], row["sample_id"]), {})
        if row["view_id"] in capture:
            raise ValueError(f"duplicate normal reference view: {row['physical_part_id']} {row['view_id']}")
        capture[row["view_id"]] = row

    complete_by_part: dict[str, dict[str, dict[str, str]]] = {}
    for part_id, captures in captures_by_part.items():
        complete_captures = [
            capture for _identity, capture in sorted(captures.items()) if set(capture) == set(VIEW_ORDER)
        ]
        if complete_captures:
            complete_by_part[part_id] = complete_captures[0]
    selected_part_ids = sorted(complete_by_part)[:count_per_view]
    if len(selected_part_ids) != count_per_view:
        raise ValueError(
            f"need {count_per_view} complete normal train parts, found {len(complete_by_part)}",
        )
    selected_rows = [complete_by_part[part_id][view] for part_id in selected_part_ids for view in VIEW_ORDER]
    return selected_rows, selected_part_ids


def _label_config() -> str:
    return """<View>
  <Header value="BMW八视图ROI | $source_class | $view_id | $sample_id"/>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="defect" background="#E53935"/>
  </RectangleLabels>
  <Choices name="review_result" toName="image" choice="single" required="true">
    <Choice value="有可见缺陷"/>
    <Choice value="无可见缺陷"/>
  </Choices>
</View>
"""


def _guide(package_id: str, task_count: int, normal_references_per_view: int) -> str:
    reference_note = (
        f"- 正常参考图：每个视角：{normal_references_per_view}张，仅供对照，不需要标注。\n"
        if normal_references_per_view
        else ""
    )
    return f"""# BMW八视图YOLO标注说明

## 数据范围

- 标注包：`{package_id}`
- 待复核图片：{task_count}张ROI裁剪图
- 类别只有一个：`defect`
- 248张待标任务不包含 normal 和 no_streak；亮痕缺失由专用规则检测，不要标为 defect。
{reference_note}

## 标注规则

1. 当前ROI中确实能看到缺陷：选择“有可见缺陷”，用 `defect` 矩形框紧贴缺陷；多个独立缺陷分别画框。
2. 当前视图没有可见缺陷：选择“无可见缺陷”，不画框，然后提交任务。
3. 零件整体为NG不代表八个视图都要有框，只标当前图片实际可见的缺陷。
4. 不要框正常轮廓、孔洞、固定高光、中心亮痕或背景。
5. 必须完成全部任务，不能用“尚未标注”代替“无可见缺陷”。

## Label Studio使用方法

在标注包根目录对应的机器上启动：

```bash
conda activate label-studio
export DEBUG=false
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/标注包的绝对路径/{package_id}
label-studio start --port 8080
```

新建项目后：

1. 导入 `label_studio/tasks.json`。
2. 将 `label_studio/label_config.xml` 内容粘贴到标注界面配置。
3. 完成全部任务后，同时导出 **JSON** 和 **YOLO** 两种格式。
4. 将两个完整导出包一起交回，不要自行修改或重命名图片和标签文件。

## 文件说明

- `images/`：需要标注的实体图片，可复制到其他电脑，不依赖原始数据目录。
- `normal_references/`：按视角存放的正常零件参考图，只用于比较，不导入标注任务。
- `normal_reference_manifest.csv`：正常参考图身份和哈希清单。
- `annotation_tasks.csv`：图片、视图、缺陷类别、样本身份和期望标签文件名。
- `label_studio/tasks.json`：Label Studio导入任务。
- `label_studio/label_config.xml`：标注界面配置。
- `SHA256SUMS`：交付图片完整性校验。
- `report.json`：标注包统计和来源信息。
"""


def prepare_labeling_package(
    *,
    training_root: Path,
    output_root: Path,
    package_id: str,
    normal_references_per_view: int = 0,
) -> dict[str, object]:
    """Publish one immutable, portable Label Studio handoff package."""
    if not isinstance(package_id, str) or not _PACKAGE_ID.fullmatch(package_id):
        raise ValueError("package_id contains unsupported characters")
    if not isinstance(normal_references_per_view, int) or not 0 <= normal_references_per_view <= 10:
        raise ValueError("normal_references_per_view must be an integer from 0 to 10")
    training = Path(training_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    queue_path = training / "yolo/annotation_queue.csv"
    if not queue_path.is_file() or queue_path.is_symlink():
        raise ValueError(f"annotation queue does not exist: {queue_path}")
    destination = output / package_id
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"labeling package already exists: {destination}")

    queue_rows = _read_queue(queue_path)
    reference_source_rows, reference_part_ids = _select_normal_references(
        training,
        normal_references_per_view,
    )
    output.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{package_id}.", dir=output))
    task_rows: list[dict[str, str]] = []
    tasks: list[dict[str, object]] = []
    seen_images: set[str] = set()
    seen_labels: set[str] = set()
    checksums: list[str] = []
    image_bytes = 0
    reference_bytes = 0
    reference_rows: list[dict[str, str]] = []
    try:
        images_root = staging / "images"
        images_root.mkdir()
        label_studio_root = staging / "label_studio"
        label_studio_root.mkdir()
        for index, row in enumerate(queue_rows, start=1):
            if row["source_class"] in {"normal", "no_streak"}:
                raise ValueError("annotation queue must not contain normal or no_streak rows")
            source = _resolve_crop(training, row["crop_path"])
            image_name = source.name
            expected_label = row["expected_label_filename"]
            if Path(expected_label).name != expected_label or Path(expected_label).suffix != ".txt":
                raise ValueError(f"invalid expected label filename: {expected_label}")
            if Path(expected_label).stem != Path(image_name).stem:
                raise ValueError(f"image and expected label stems differ: {image_name}")
            if image_name in seen_images or expected_label in seen_labels:
                raise ValueError(f"duplicate portable image or label name: {image_name}")
            seen_images.add(image_name)
            seen_labels.add(expected_label)

            target = images_root / image_name
            shutil.copy2(source, target)
            digest = _sha256(target)
            image_bytes += target.stat().st_size
            checksums.append(f"{digest}  images/{image_name}")
            task_row = {
                "task_index": str(index),
                "image_filename": image_name,
                "image_sha256": digest,
                **row,
            }
            task_rows.append(task_row)
            tasks.append(
                {
                    "data": {
                        "image": f"/data/local-files/?d=images/{image_name}",
                        "task_index": index,
                        "image_filename": image_name,
                        "expected_label_filename": expected_label,
                        "sample_id": row["sample_id"],
                        "physical_part_id": row["physical_part_id"],
                        "session_id": row["session_id"],
                        "group_id": row["group_id"],
                        "view_id": row["view_id"],
                        "source_class": row["source_class"],
                        "split": row["split"],
                    }
                }
            )

        for reference_index, row in enumerate(reference_source_rows, start=1):
            source = _resolve_crop(training, row["crop_path"])
            if row["crop_sha256"] and _sha256(source) != row["crop_sha256"]:
                raise ValueError(f"normal reference crop SHA-256 mismatch: {source}")
            target = staging / "normal_references" / row["view_id"] / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            digest = _sha256(target)
            reference_bytes += target.stat().st_size
            checksums.append(f"{digest}  normal_references/{row['view_id']}/{source.name}")
            reference_rows.append(
                {
                    "reference_index": str(reference_index),
                    "image_filename": source.name,
                    "image_sha256": digest,
                    **row,
                }
            )

        _write_csv(staging / "annotation_tasks.csv", _TASK_FIELDS, task_rows)
        _write_csv(staging / "normal_reference_manifest.csv", _REFERENCE_FIELDS, reference_rows)
        (label_studio_root / "tasks.json").write_text(
            json.dumps(tasks, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (label_studio_root / "label_config.xml").write_text(_label_config(), encoding="utf-8")
        (staging / "标注说明.md").write_text(
            _guide(package_id, len(task_rows), normal_references_per_view),
            encoding="utf-8",
        )
        (staging / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
        report: dict[str, object] = {
            "schema_version": 1,
            "package_id": package_id,
            "release_status": "published",
            "training_root": str(training),
            "annotation_queue": str(queue_path),
            "annotation_queue_sha256": _sha256(queue_path),
            "task_count": len(task_rows),
            "image_bytes": image_bytes,
            "normal_reference_count": len(reference_rows),
            "normal_reference_bytes": reference_bytes,
            "normal_references_per_view": normal_references_per_view,
            "normal_reference_part_ids": reference_part_ids,
            "class_names": ["defect"],
            "source_class_counts": dict(sorted(Counter(row["source_class"] for row in queue_rows).items())),
            "view_counts": dict(sorted(Counter(row["view_id"] for row in queue_rows).items())),
            "portable_images": True,
            "contains_normal_or_no_streak": False,
        }
        (staging / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _atomic_publish_noreplace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report
