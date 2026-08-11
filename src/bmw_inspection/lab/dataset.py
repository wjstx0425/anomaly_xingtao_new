"""Physical-part manifests and training exports for BMW laboratory inspection."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import cv2

from bmw_inspection.lab.contracts import ViewId


MANIFEST_FIELDS = (
    "sample_id",
    "part_id",
    "session_id",
    "view_id",
    "image_path",
    "image_sha256",
    "label",
    "defect_type",
    "x1",
    "y1",
    "x2",
    "y2",
    "split",
)
REGRESSION_FIELDS = (
    "sample_id",
    "part_id",
    "session_id",
    "view_id",
    "image_path",
    "image_sha256",
    "expected_bright_streak_status",
    "purpose",
)
SPLITS = ("train", "calibration", "final_test")
MIN_FINAL_PARTS = 30
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_YOLO_SPLIT = {"train": "train", "calibration": "val", "final_test": "test"}
_PATCHCORE_SPLIT = {"calibration": "calibration", "final_test": "final_test"}


@dataclass(frozen=True, slots=True)
class ManifestRow:
    """One explicitly identified physical-view image and its ground truth."""

    sample_id: str
    part_id: str
    session_id: str
    view_id: ViewId
    image_path: Path
    image_sha256: str
    label: str
    defect_type: str
    x1: float | None
    y1: float | None
    x2: float | None
    y2: float | None
    split: str


def _text(row: Mapping[str, object], field: str) -> str:
    value = row.get(field)
    return "" if value is None else str(value).strip()


def _required_text(row: Mapping[str, object], field: str) -> str:
    value = _text(row, field)
    if not value:
        raise ValueError(f"{field} must be explicit and non-empty")
    return value


def _identifier(row: Mapping[str, object], field: str) -> str:
    value = _required_text(row, field)
    if _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{field} must use only path-safe letters, numbers, '.', '_', or '-'")
    return value


def _optional_float(row: Mapping[str, object], field: str) -> float | None:
    text = _text(row, field)
    if not text:
        return None
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(f"bounding box field {field} must be numeric") from error
    if not math.isfinite(value):
        raise ValueError(f"bounding box field {field} must be finite")
    return value


def validate_row(row: Mapping[str, object]) -> ManifestRow:
    """Validate one strict main-manifest row without consulting the filesystem."""
    missing = set(MANIFEST_FIELDS) - set(row)
    unknown = set(row) - set(MANIFEST_FIELDS)
    if missing or unknown:
        raise ValueError(
            f"manifest fields differ from the explicit schema; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    sample_id = _identifier(row, "sample_id")
    part_id = _identifier(row, "part_id")
    session_id = _identifier(row, "session_id")
    try:
        view_id = ViewId(_required_text(row, "view_id"))
    except ValueError as error:
        raise ValueError("view_id must be one of the six canonical BMW views") from error
    image_path = Path(_required_text(row, "image_path")).expanduser()
    image_sha256 = _required_text(row, "image_sha256").lower()
    if len(image_sha256) != 64 or any(character not in "0123456789abcdef" for character in image_sha256):
        raise ValueError("image_sha256 must be a 64-character hexadecimal digest")
    label = _required_text(row, "label")
    if label not in {"normal", "defect"}:
        raise ValueError("label must be normal or defect")
    defect_type = _text(row, "defect_type")
    box = tuple(_optional_float(row, field) for field in ("x1", "y1", "x2", "y2"))
    if label == "defect":
        if not defect_type:
            raise ValueError("defect rows require an explicit defect_type")
        if any(value is None for value in box):
            raise ValueError("every YOLO-positive defect row requires a complete bounding box")
        x1, y1, x2, y2 = box
        assert x1 is not None and y1 is not None and x2 is not None and y2 is not None
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("defect bounding box must have positive area and non-negative coordinates")
    elif defect_type or any(value is not None for value in box):
        raise ValueError("normal rows must not contain defect_type or bounding box values")
    split = _text(row, "split")
    if split and split not in SPLITS:
        raise ValueError(f"split must be empty or one of {SPLITS}")
    return ManifestRow(
        sample_id=sample_id,
        part_id=part_id,
        session_id=session_id,
        view_id=view_id,
        image_path=image_path,
        image_sha256=image_sha256,
        label=label,
        defect_type=defect_type,
        x1=box[0],
        y1=box[1],
        x2=box[2],
        y2=box[3],
        split=split,
    )


def find_part_split_leakage(rows: Sequence[Mapping[str, object] | ManifestRow]) -> dict[str, tuple[str, ...]]:
    """Return physical parts observed in more than one non-empty split."""
    observed: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if isinstance(row, ManifestRow):
            part_id, split = row.part_id, row.split
        else:
            part_id, split = _text(row, "part_id"), _text(row, "split")
        if part_id and split:
            observed[part_id].add(split)
    return {part_id: tuple(sorted(splits)) for part_id, splits in sorted(observed.items()) if len(splits) > 1}


def assign_part_splits(part_ids: Sequence[str], *, seed: int = 42) -> dict[str, str]:
    """Assign unique physical parts deterministically to a 60/20/20 partition."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    unique = sorted(set(part_ids))
    if not unique or len(unique) != len(part_ids):
        raise ValueError("part_ids must be a non-empty sequence of unique physical identities")
    ordered = sorted(
        unique,
        key=lambda part_id: hashlib.sha256(f"{seed}:{part_id}".encode()).hexdigest(),
    )
    train_count = int(len(ordered) * 0.60)
    calibration_count = int(len(ordered) * 0.20)
    split_by_part: dict[str, str] = {}
    for index, part_id in enumerate(ordered):
        if index < train_count:
            split = "train"
        elif index < train_count + calibration_count:
            split = "calibration"
        else:
            split = "final_test"
        split_by_part[part_id] = split
    return {part_id: split_by_part[part_id] for part_id in unique}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _float_text(value: float | None) -> str:
    return "" if value is None else f"{value:g}"


def _row_dict(row: ManifestRow) -> dict[str, str]:
    return {
        "sample_id": row.sample_id,
        "part_id": row.part_id,
        "session_id": row.session_id,
        "view_id": row.view_id.value,
        "image_path": str(row.image_path),
        "image_sha256": row.image_sha256,
        "label": row.label,
        "defect_type": row.defect_type,
        "x1": _float_text(row.x1),
        "y1": _float_text(row.y1),
        "x2": _float_text(row.x2),
        "y2": _float_text(row.y2),
        "split": row.split,
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: Path) -> tuple[ManifestRow, ...]:
    """Read and validate the exact BMW main-manifest CSV schema."""
    with Path(path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ValueError("manifest header differs from the explicit BMW schema")
        parsed = [validate_row(row) for row in reader]
    base = Path(path).resolve().parent
    return tuple(
        replace(
            row,
            image_path=(base / row.image_path).resolve()
            if not row.image_path.is_absolute()
            else row.image_path.resolve(),
        )
        for row in parsed
    )


def _validate_source(rows: Sequence[ManifestRow]) -> tuple[str, ...]:
    if not rows:
        raise ValueError("source manifest must not be empty")
    if any(row.split for row in rows):
        raise ValueError("source manifest split values must be empty; the builder owns deterministic assignment")
    by_sample: dict[str, list[ManifestRow]] = defaultdict(list)
    parts_by_digest: dict[str, set[str]] = defaultdict(set)
    seen_sample_views: set[tuple[str, ViewId]] = set()
    for row in rows:
        if not row.image_path.is_file():
            raise ValueError(f"manifest image does not exist: {row.image_path}")
        if _sha256(row.image_path) != row.image_sha256:
            raise ValueError(f"manifest image_sha256 mismatch: {row.image_path}")
        parts_by_digest[row.image_sha256].add(row.part_id)
        identity = (row.sample_id, row.view_id)
        if identity in seen_sample_views:
            raise ValueError(f"duplicate view {row.view_id.value} for sample {row.sample_id}")
        seen_sample_views.add(identity)
        by_sample[row.sample_id].append(row)
    duplicate_content = {
        digest: tuple(sorted(part_ids))
        for digest, part_ids in parts_by_digest.items()
        if len(part_ids) > 1
    }
    if duplicate_content:
        raise ValueError(f"same image content is assigned to multiple physical parts: {duplicate_content}")
    required_views = set(ViewId)
    for sample_id, sample_rows in sorted(by_sample.items()):
        part_ids = {row.part_id for row in sample_rows}
        session_ids = {row.session_id for row in sample_rows}
        views = {row.view_id for row in sample_rows}
        if len(part_ids) != 1 or len(session_ids) != 1:
            raise ValueError(f"sample {sample_id} mixes part_id or session_id identities")
        if views != required_views or len(sample_rows) != len(required_views):
            missing = sorted(view.value for view in required_views - views)
            raise ValueError(f"sample {sample_id} is not a complete six-view capture; missing={missing}")
        if len({row.image_path for row in sample_rows}) != len(required_views) or len(
            {row.image_sha256 for row in sample_rows}
        ) != len(required_views):
            raise ValueError(f"sample {sample_id} must use six distinct source images")
    return tuple(sorted({row.part_id for row in rows}))


def _image_dimensions(row: ManifestRow) -> tuple[int, int]:
    image = cv2.imread(str(row.image_path), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        raise ValueError(f"cannot decode manifest image: {row.image_path}")
    height, width = image.shape[:2]
    if row.label == "defect":
        assert row.x2 is not None and row.y2 is not None
        if row.x2 > width or row.y2 > height:
            raise ValueError(f"defect bounding box exceeds image dimensions for {row.image_path}")
    return width, height


def _export_name(row: ManifestRow) -> str:
    suffix = row.image_path.suffix.lower() or ".png"
    return f"{row.sample_id}__{row.view_id.value}{suffix}"


def _copy_source(row: ManifestRow, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError(f"dataset export destination collision: {target}")
    shutil.copy2(row.image_path, target)
    if _sha256(target) != row.image_sha256:
        raise RuntimeError(f"exported image digest differs from source: {target}")


def _validated_part_rois(
    part_rois: Mapping[ViewId, tuple[int, int, int, int]],
) -> dict[ViewId, tuple[int, int, int, int]]:
    normalized: dict[ViewId, tuple[int, int, int, int]] = {}
    for raw_view, raw_roi in part_rois.items():
        try:
            view_id = ViewId(raw_view)
        except (TypeError, ValueError) as error:
            raise ValueError(f"part_rois contains unknown view: {raw_view!r}") from error
        if (
            not isinstance(raw_roi, tuple)
            or len(raw_roi) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_roi)
        ):
            raise ValueError(f"part ROI for {view_id.value} must contain four integers")
        x1, y1, x2, y2 = raw_roi
        if not (0 <= x1 < x2 and 0 <= y1 < y2):
            raise ValueError(f"part ROI for {view_id.value} must have positive area")
        normalized[view_id] = raw_roi
    if set(normalized) != set(ViewId):
        raise ValueError("part_rois must contain exactly the six BMW views")
    return normalized


def _validate_yolo_spatial_contract(
    rows: Sequence[ManifestRow],
    part_rois: Mapping[ViewId, tuple[int, int, int, int]],
) -> None:
    for row in rows:
        width, height = _image_dimensions(row)
        roi_x1, roi_y1, roi_x2, roi_y2 = part_rois[row.view_id]
        if roi_x2 > width or roi_y2 > height:
            raise ValueError(f"part ROI for {row.view_id.value} exceeds image dimensions for {row.image_path}")
        if row.label == "defect":
            assert row.x1 is not None and row.y1 is not None and row.x2 is not None and row.y2 is not None
            clipped_x1 = max(row.x1, roi_x1)
            clipped_y1 = max(row.y1, roi_y1)
            clipped_x2 = min(row.x2, roi_x2)
            clipped_y2 = min(row.y2, roi_y2)
            if clipped_x1 >= clipped_x2 or clipped_y1 >= clipped_y2:
                raise ValueError(
                    f"defect bounding box does not intersect configured part ROI for {row.image_path}"
                )


def _export_yolo(
    rows: Sequence[ManifestRow],
    output_root: Path,
    part_rois: Mapping[ViewId, tuple[int, int, int, int]],
) -> None:
    export_rows: list[dict[str, str]] = []
    for split in _YOLO_SPLIT.values():
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    for row in rows:
        yolo_split = _YOLO_SPLIT[row.split]
        name = _export_name(row)
        image_target = output_root / "images" / yolo_split / name
        label_target = output_root / "labels" / yolo_split / f"{Path(name).stem}.txt"
        image = cv2.imread(str(row.image_path), cv2.IMREAD_UNCHANGED)
        if image is None or image.size == 0:
            raise ValueError(f"cannot decode manifest image: {row.image_path}")
        roi_x1, roi_y1, roi_x2, roi_y2 = part_rois[row.view_id]
        crop = image[roi_y1:roi_y2, roi_x1:roi_x2]
        image_target.parent.mkdir(parents=True, exist_ok=True)
        if image_target.exists():
            raise ValueError(f"dataset export destination collision: {image_target}")
        if not cv2.imwrite(str(image_target), crop):
            raise RuntimeError(f"cannot write YOLO ROI crop: {image_target}")
        width, height = roi_x2 - roi_x1, roi_y2 - roi_y1
        label_text = ""
        if row.label == "defect":
            assert row.x1 is not None and row.y1 is not None and row.x2 is not None and row.y2 is not None
            clipped_x1 = max(row.x1, roi_x1) - roi_x1
            clipped_y1 = max(row.y1, roi_y1) - roi_y1
            clipped_x2 = min(row.x2, roi_x2) - roi_x1
            clipped_y2 = min(row.y2, roi_y2) - roi_y1
            x_center = ((clipped_x1 + clipped_x2) / 2.0) / width
            y_center = ((clipped_y1 + clipped_y2) / 2.0) / height
            box_width = (clipped_x2 - clipped_x1) / width
            box_height = (clipped_y2 - clipped_y1) / height
            label_text = f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n"
        label_target.write_text(label_text, encoding="utf-8")
        export_rows.append(
            {
                **_row_dict(row),
                "yolo_split": yolo_split,
                "roi_x1": str(roi_x1),
                "roi_y1": str(roi_y1),
                "roi_x2": str(roi_x2),
                "roi_y2": str(roi_y2),
                "exported_image_path": str(Path("images") / yolo_split / name),
                "exported_label_path": str(Path("labels") / yolo_split / f"{Path(name).stem}.txt"),
            }
        )
    _write_csv(
        output_root / "export_manifest.csv",
        (
            *MANIFEST_FIELDS,
            "yolo_split",
            "roi_x1",
            "roi_y1",
            "roi_x2",
            "roi_y2",
            "exported_image_path",
            "exported_label_path",
        ),
        export_rows,
    )
    (output_root / "data.yaml").write_text(
        "train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: defect\n",
        encoding="utf-8",
    )


def _export_patchcore(rows: Sequence[ManifestRow], output_root: Path) -> None:
    by_view: dict[ViewId, list[dict[str, str]]] = {view_id: [] for view_id in ViewId}
    for view_id in ViewId:
        view_root = output_root / view_id.value
        for relative in (
            "train/good",
            "calibration/good",
            "calibration/defect",
            "final_test/good",
            "final_test/defect",
        ):
            (view_root / relative).mkdir(parents=True, exist_ok=True)
    for row in rows:
        if row.split == "train":
            if row.label != "normal":
                continue
            partition, category = "train", "good"
        else:
            partition = _PATCHCORE_SPLIT[row.split]
            category = "good" if row.label == "normal" else "defect"
        target = output_root / row.view_id.value / partition / category / _export_name(row)
        _copy_source(row, target)
        by_view[row.view_id].append(
            {
                **_row_dict(row),
                "patchcore_partition": partition,
                "exported_image_path": str(Path(row.view_id.value) / partition / category / _export_name(row)),
            }
        )
    for view_id, export_rows in by_view.items():
        _write_csv(
            output_root / view_id.value / "manifest.csv",
            (*MANIFEST_FIELDS, "patchcore_partition", "exported_image_path"),
            export_rows,
        )


def _bright_streak_rows(root: Path) -> list[dict[str, str]]:
    counts = {status: len(list((root / status).glob("*.bmp"))) for status in ("OK", "NG")}
    if counts != {"OK": 6, "NG": 7}:
        raise ValueError(f"bright-streak regression root must contain the fixed 6 OK / 7 NG images, got {counts}")
    rows: list[dict[str, str]] = []
    for status in ("OK", "NG"):
        for image_path in sorted((root / status).glob("*.bmp")):
            part_id = f"legacy-bright-streak-{image_path.stem}"
            rows.append(
                {
                    "sample_id": part_id,
                    "part_id": part_id,
                    "session_id": "legacy-bright-streak-20260805",
                    "view_id": ViewId.FRONT_LEFT.value,
                    "image_path": str(image_path.resolve()),
                    "image_sha256": _sha256(image_path),
                    "expected_bright_streak_status": "OK" if status == "OK" else "NG_NO_STREAK",
                    "purpose": "bright_streak_regression_only",
                }
            )
    return rows


def build_dataset(
    *,
    source_manifest: Path,
    output_root: Path,
    dataset_id: str,
    bright_streak_root: Path,
    part_rois: Mapping[ViewId, tuple[int, int, int, int]],
    seed: int = 42,
    experimental_small_data: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    """Build one immutable-by-convention manifest release and its training exports."""
    if _SAFE_ID.fullmatch(dataset_id) is None:
        raise ValueError("dataset_id must be a non-empty path-safe identifier")
    source_rows = read_manifest(source_manifest)
    part_ids = _validate_source(source_rows)
    validated_rois = _validated_part_rois(part_rois)
    if len(part_ids) < MIN_FINAL_PARTS and not experimental_small_data:
        raise ValueError(
            f"need at least 30 complete six-view physical parts for a final dataset; "
            "pass --experimental-small-data to publish an experimental-only release"
        )
    split_by_part = assign_part_splits(part_ids, seed=seed)
    rows = tuple(replace(row, split=split_by_part[row.part_id]) for row in source_rows)
    _validate_yolo_spatial_contract(rows, validated_rois)
    leakage = find_part_split_leakage(rows)
    if leakage:
        raise ValueError(f"physical-part split leakage detected: {leakage}")
    regression_rows = _bright_streak_rows(Path(bright_streak_root))
    split_part_counts = Counter(split_by_part.values())
    split_row_counts = Counter(row.split for row in rows)
    defect_bbox_counts = Counter(row.split for row in rows if row.label == "defect")
    if not experimental_small_data and any(defect_bbox_counts.get(split, 0) == 0 for split in SPLITS):
        raise ValueError("final evaluation requires actual defect bounding boxes in every split")
    report: dict[str, object] = {
        "dataset_id": dataset_id,
        "source_manifest": str(Path(source_manifest).resolve()),
        "source_manifest_sha256": _sha256(Path(source_manifest)),
        "seed": seed,
        "manifest_rows": len(rows),
        "complete_capture_count": len({row.sample_id for row in rows}),
        "physical_part_count": len(part_ids),
        "split_part_counts": {split: split_part_counts.get(split, 0) for split in SPLITS},
        "split_row_counts": {split: split_row_counts.get(split, 0) for split in SPLITS},
        "defect_bbox_rows": sum(defect_bbox_counts.values()),
        "defect_bbox_rows_by_split": {split: defect_bbox_counts.get(split, 0) for split in SPLITS},
        "split_leakage_count": len(leakage),
        "bright_streak_regression_rows": len(regression_rows),
        "experimental_only": bool(experimental_small_data),
        "final_evaluation_allowed": len(part_ids) >= MIN_FINAL_PARTS and not experimental_small_data,
        "yolo_part_rois": {
            view_id.value: list(validated_rois[view_id])
            for view_id in ViewId
        },
    }
    if dry_run:
        return report
    output_root = Path(output_root)
    if output_root.exists():
        raise ValueError(f"output_root already exists; refusing to overwrite: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dataset_id}-", dir=output_root.parent))
    try:
        _write_csv(staging / "manifests" / f"{dataset_id}.csv", MANIFEST_FIELDS, [_row_dict(row) for row in rows])
        if regression_rows:
            _write_csv(staging / "manifests/bright_streak_regression.csv", REGRESSION_FIELDS, regression_rows)
        _export_yolo(rows, staging / "exports" / dataset_id / "yolo", validated_rois)
        _export_patchcore(rows, staging / "exports" / dataset_id / "patchcore")
        (staging / "manifests" / f"{dataset_id}.report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


__all__ = [
    "MANIFEST_FIELDS",
    "MIN_FINAL_PARTS",
    "REGRESSION_FIELDS",
    "ManifestRow",
    "assign_part_splits",
    "build_dataset",
    "find_part_split_leakage",
    "read_manifest",
    "validate_row",
]
