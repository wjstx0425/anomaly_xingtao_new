# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: EM101, EM102, TRY003

"""Resumable three-model offline calibration support for legacy ZS32 assets."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
from capture_data.zs32_inspection_orchestrator import InspectionRequest
from capture_data.zs32_template_gate import TemplateGate

from zs32_inspection.domain.views import VIEW_ORDER

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from capture_data.zs32_model_runtime import ZS32ModelRuntime

ZS32_VIEWS = VIEW_ORDER
CALIBRATION_FIELDS = (
    "part_id",
    "hand",
    "view",
    "branch",
    "raw_score",
    "gt_label",
    "split",
    "model_version",
    "roi_version",
)
_SEMANTIC_VIEW = re.compile(
    rf"^(?:left|right)_({'|'.join(re.escape(view) for view in sorted(ZS32_VIEWS, key=len, reverse=True))})_",
    re.IGNORECASE,
)
_GROUP_ID = re.compile(r"_(group\d+)_\d+_fused\.png$", re.IGNORECASE)


@dataclass(frozen=True)
class OfflineCalibrationCase:
    """One labeled physical part with all canonical semantic-view source images."""

    part_id: str
    hand: str
    gt_label: int
    split: str
    session_id: str
    group_id: str
    defect_type: str
    images: Mapping[str, Path]

    @property
    def slug(self) -> str:
        """Readable collision-resistant output directory name."""
        readable = re.sub(r"[^a-zA-Z0-9._-]+", "_", self.part_id).strip("_")
        digest = hashlib.sha256(self.part_id.encode()).hexdigest()[:8]
        return f"{readable}-{digest}"

    def request(self) -> InspectionRequest:
        """Build the runtime request without changing source image identities."""
        return InspectionRequest(
            self.part_id,
            self.session_id,
            self.group_id,
            self.hand,
            self.images,
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV into normalized string mappings."""
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return [{str(key): "" if value is None else str(value).strip() for key, value in row.items()} for row in reader]


def _required(row: Mapping[str, str], field: str, *, source: Path, row_number: int) -> str:
    value = row.get(field, "").strip()
    if not value:
        raise ValueError(f"{source} row {row_number} has empty required field: {field}")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _template_split_contract(path: Path, hand: str) -> dict[str, tuple[int, str]]:
    """Load the frozen physical-part labels and splits from template scoring."""
    records: dict[str, tuple[int, str]] = {}
    row_count_by_part: dict[str, int] = {}
    for row_number, row in enumerate(_read_csv(path), start=2):
        if _required(row, "hand", source=path, row_number=row_number).lower() != hand:
            continue
        if _required(row, "branch", source=path, row_number=row_number) != "template_match":
            raise ValueError(f"{path} row {row_number} must use branch=template_match")
        part_id = _required(row, "part_id", source=path, row_number=row_number)
        label_text = _required(row, "gt_label", source=path, row_number=row_number)
        if label_text not in {"0", "1"}:
            raise ValueError(f"{path} row {row_number} gt_label must be 0 or 1")
        split = _required(row, "split", source=path, row_number=row_number)
        if split not in {"calibration", "test"}:
            raise ValueError(f"{path} row {row_number} has unsupported split: {split}")
        identity = (int(label_text), split)
        if part_id in records and records[part_id] != identity:
            raise ValueError(f"physical part {part_id!r} has inconsistent label/split values")
        records[part_id] = identity
        view = _required(row, "view", source=path, row_number=row_number)
        if view not in ZS32_VIEWS:
            raise ValueError(f"{path} row {row_number} has unsupported view: {view}")
        row_count_by_part[part_id] = row_count_by_part.get(part_id, 0) + 1
    if not records:
        raise ValueError(f"no {hand!r} template calibration rows found: {path}")
    for part_id, row_count in row_count_by_part.items():
        if row_count != len(ZS32_VIEWS):
            raise ValueError(
                f"template split part {part_id!r} must contain exactly {len(ZS32_VIEWS)} score rows",
            )
    return records


def _part_id(row: Mapping[str, str], group_id: str, hand: str) -> str:
    label = row.get("label", "").strip().lower()
    if label == "normal":
        return f"{hand}:normal:{group_id}"
    if label != "defect":
        raise ValueError(f"unsupported crop-manifest label: {label!r}")
    defect_type = row.get("defect_type", "").strip().lower()
    if not defect_type:
        raise ValueError("defect crop-manifest row has empty defect_type")
    return f"{hand}:defect:{defect_type}:{group_id}"


def load_offline_cases(
    crop_manifest: Path,
    template_calibration_csv: Path,
    *,
    path_root: Path,
    hand: str = "right",
) -> tuple[OfflineCalibrationCase, ...]:
    """Build canonical-view cases from raw sources using filename semantic views.

    The crop manifest remains the authoritative sample index, but its routed
    ``resolved_view`` and already-cropped ``output_path`` are not trusted for
    inference. This repairs known legacy front/back routing mismatches by
    binding each original source image to the view encoded in its basename.
    """
    hand = hand.strip().lower()
    if hand != "right":
        raise ValueError("current commissioning assets support exactly hand=right")
    root = path_root.expanduser().resolve()
    contract = _template_split_contract(template_calibration_csv, hand)
    grouped: dict[str, dict[str, Path]] = {}
    metadata: dict[str, tuple[str, str, str]] = {}
    source_owners: dict[Path, tuple[str, str]] = {}
    for row_number, row in enumerate(_read_csv(crop_manifest), start=2):
        if row.get("hand", "").strip().lower() != hand:
            continue
        source_text = _required(row, "source_path", source=crop_manifest, row_number=row_number)
        source_path = _resolve(root, source_text)
        view_match = _SEMANTIC_VIEW.search(source_path.name)
        group_match = _GROUP_ID.search(source_path.name)
        if view_match is None or group_match is None:
            raise ValueError(f"cannot parse semantic view/group from source basename: {source_path.name}")
        view = view_match.group(1).lower()
        group_id = group_match.group(1).lower()
        part_id = _part_id(row, group_id, hand)
        if part_id not in contract:
            continue
        if not source_path.is_file():
            raise FileNotFoundError(f"source image does not exist: {source_path}")
        owner = (part_id, view)
        if source_path in source_owners and source_owners[source_path] != owner:
            raise ValueError(
                f"source image is reused across physical identities: {source_path} "
                f"({source_owners[source_path]} and {owner})",
            )
        source_owners[source_path] = owner
        images = grouped.setdefault(part_id, {})
        if view in images:
            raise ValueError(f"physical part {part_id!r} has duplicate semantic view {view!r}")
        images[view] = source_path
        current = (
            _required(row, "session_id", source=crop_manifest, row_number=row_number),
            group_id,
            row.get("defect_type", "").strip().lower(),
        )
        if part_id in metadata and metadata[part_id] != current:
            raise ValueError(f"physical part {part_id!r} has inconsistent session/group metadata")
        metadata[part_id] = current
    missing_parts = sorted(set(contract) - set(grouped))
    if missing_parts:
        raise ValueError(f"template split parts are missing from crop manifest: {missing_parts}")
    cases: list[OfflineCalibrationCase] = []
    for part_id in sorted(contract):
        images = grouped[part_id]
        if set(images) != set(ZS32_VIEWS) or len(images) != len(ZS32_VIEWS):
            raise ValueError(
                f"physical part {part_id!r} must contain exactly the {len(ZS32_VIEWS)} canonical views",
            )
        gt_label, split = contract[part_id]
        session_id, group_id, defect_type = metadata[part_id]
        cases.append(
            OfflineCalibrationCase(
                part_id=part_id,
                hand=hand,
                gt_label=gt_label,
                split=split,
                session_id=session_id,
                group_id=group_id,
                defect_type=defect_type,
                images={view: images[view] for view in ZS32_VIEWS},
            ),
        )
    return tuple(cases)


def _validate_calibration_row(row: Mapping[str, str], case: OfflineCalibrationCase, source: Path) -> None:
    for field in CALIBRATION_FIELDS:
        if not str(row.get(field, "")).strip():
            raise ValueError(f"{source} contains an empty calibration field: {field}")
    if row["part_id"] != case.part_id or row["hand"] != case.hand:
        raise ValueError(f"{source} row identity does not match {case.part_id!r}")
    if int(row["gt_label"]) != case.gt_label or row["split"] != case.split:
        raise ValueError(f"{source} row label/split does not match {case.part_id!r}")


def _write_rows_atomic(output_csv: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if output_csv.exists():
        expected = [{field: str(row.get(field, "")) for field in CALIBRATION_FIELDS} for row in rows]
        if _read_csv(output_csv) != expected:
            raise ValueError(f"existing calibration rows differ from current inputs: {output_csv}")
        return
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_name(f".{output_csv.name}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=CALIBRATION_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output_csv)
    finally:
        temporary.unlink(missing_ok=True)


def merge_calibration_rows(
    cases: Sequence[OfflineCalibrationCase],
    case_root: Path,
    output_csv: Path,
) -> int:
    """Merge complete template/PatchCore/YOLO rows into one Stage31 CSV."""
    view_order = {view: index for index, view in enumerate(ZS32_VIEWS)}
    merged: list[dict[str, str]] = []
    for case in cases:
        case_dir = case_root / case.slug
        rows: list[dict[str, str]] = []
        for name in ("template_calibration_rows.csv", "calibration_rows.csv"):
            source = case_dir / name
            if not source.is_file():
                raise FileNotFoundError(f"missing case calibration CSV: {source}")
            source_rows = _read_csv(source)
            for row in source_rows:
                _validate_calibration_row(row, case, source)
            rows.extend(source_rows)
        keys = {(row["view"], row["branch"]) for row in rows}
        expected = {(view, "template_match") for view in ZS32_VIEWS}
        expected |= {(view, f"anomaly_{view}") for view in ZS32_VIEWS}
        expected |= {(view, "yolo") for view in ZS32_VIEWS}
        if len(rows) != len(expected) or keys != expected:
            raise ValueError(f"physical part {case.part_id!r} must contain 18 unique calibration groups")
        rows.sort(
            key=lambda row: (
                view_order[row["view"]],
                {"template_match": 0, f"anomaly_{row['view']}": 1, "yolo": 2}[row["branch"]],
            ),
        )
        merged.extend(rows)
    _write_rows_atomic(output_csv, merged)
    return len(merged)


def write_yolo_annotation_calibration_rows(
    cases: Sequence[OfflineCalibrationCase],
    aggregate_csv: Path,
    yolo_dataset_root: Path,
    output_csv: Path,
) -> int:
    """Build leakage-aware per-view YOLO rows from actual box-label files.

    YOLO ``val`` becomes the fit split, ``test`` remains held out, and
    ``train`` is excluded. Each source physical part/view is assigned its own
    score identity because box presence is a per-view target.
    """
    score_rows = [row for row in _read_csv(aggregate_csv) if row.get("branch") == "yolo"]
    scores: dict[tuple[str, str], dict[str, str]] = {}
    for row in score_rows:
        key = (row.get("part_id", ""), row.get("view", ""))
        if key in scores:
            raise ValueError(f"duplicate YOLO score row: {key}")
        scores[key] = row
    expected_scores = {(case.part_id, view) for case in cases for view in ZS32_VIEWS}
    missing_scores = sorted(expected_scores - set(scores))
    extra_scores = sorted(set(scores) - expected_scores)
    if missing_scores:
        part_id, view = missing_scores[0]
        raise ValueError(f"missing YOLO score for physical part {part_id!r}, view {view!r}")
    if extra_scores:
        raise ValueError(f"unexpected YOLO score identities: {extra_scores}")
    case_label_index = validate_yolo_annotation_dataset(cases, yolo_dataset_root)
    rows: list[dict[str, Any]] = []
    for case in cases:
        case_labels = case_label_index[case.part_id]
        for view in ZS32_VIEWS:
            score_row = scores[case.part_id, view]
            yolo_split, label_path, _ = case_labels[view]
            if yolo_split == "train":
                continue
            rows.append(
                {
                    "part_id": f"{case.part_id}::{view}",
                    "hand": case.hand,
                    "view": view,
                    "branch": "yolo",
                    "raw_score": score_row["raw_score"],
                    "gt_label": int(bool(label_path.read_text(encoding="utf-8").strip())),
                    "split": "calibration" if yolo_split == "val" else "test",
                    "model_version": score_row["model_version"],
                    "roi_version": score_row["roi_version"],
                },
            )
    if not rows:
        raise ValueError("YOLO val/test annotation calibration rows are empty")
    _write_rows_atomic(output_csv, rows)
    return len(rows)


def validate_yolo_annotation_dataset(
    cases: Sequence[OfflineCalibrationCase],
    yolo_dataset_root: Path,
) -> dict[str, dict[str, tuple[str, Path, Path]]]:
    """Validate YOLO pairing, syntax, and physical-part split isolation."""
    label_index: dict[str, tuple[str, Path, Path]] = {}
    root = yolo_dataset_root.expanduser().resolve()
    for split in ("train", "val", "test"):
        label_dir = root / "labels" / split
        image_dir = root / "images" / split
        if not label_dir.is_dir():
            raise FileNotFoundError(f"YOLO label split does not exist: {label_dir}")
        if not image_dir.is_dir():
            raise FileNotFoundError(f"YOLO image split does not exist: {image_dir}")
        for path in sorted(label_dir.glob("*.txt")):
            if path.stem in label_index:
                raise ValueError(f"YOLO label basename appears in multiple splits: {path.stem}")
            image_matches = [candidate for candidate in image_dir.glob(f"{path.stem}.*") if candidate.is_file()]
            if len(image_matches) != 1:
                raise ValueError(f"YOLO label must match exactly one split image: {path}")
            _validate_yolo_label(path)
            label_index[path.stem] = (split, path, image_matches[0])
    case_label_index = {}
    for case in cases:
        case_labels: dict[str, tuple[str, Path, Path]] = {}
        for view in ZS32_VIEWS:
            stem = case.images[view].stem
            try:
                case_labels[view] = label_index[stem]
            except KeyError as exc:
                raise ValueError(f"missing YOLO label for source basename: {stem}") from exc
        case_splits = {split for split, _, _ in case_labels.values()}
        if len(case_splits) != 1:
            raise ValueError(f"physical part {case.part_id!r} spans YOLO splits: {sorted(case_splits)}")
        case_label_index[case.part_id] = case_labels
    return case_label_index


def validate_yolo_runtime_crop_identity(
    cases: Sequence[OfflineCalibrationCase],
    case_root: Path,
    yolo_dataset_root: Path,
) -> int:
    """Require every runtime YOLO crop to equal its labeled ROI image pixel-for-pixel."""
    case_label_index = validate_yolo_annotation_dataset(cases, yolo_dataset_root)
    matched = 0
    for case in cases:
        for view in ZS32_VIEWS:
            _, _, dataset_image = case_label_index[case.part_id][view]
            runtime_image = case_root / case.slug / "crops" / "yolo" / f"{view}.png"
            expected = cv2.imread(str(dataset_image), cv2.IMREAD_UNCHANGED)
            actual = cv2.imread(str(runtime_image), cv2.IMREAD_UNCHANGED)
            if expected is None or actual is None:
                raise ValueError(f"cannot decode YOLO crop identity pair: {dataset_image}, {runtime_image}")
            if expected.shape != actual.shape or not (expected == actual).all():
                raise ValueError(
                    f"runtime YOLO crop does not match labeled dataset pixels: {case.part_id}/{view}",
                )
            matched += 1
    return matched


def _validate_yolo_label(path: Path) -> None:
    """Validate the subset of Ultralytics detection-label syntax used here."""
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number} must contain class and four normalized bbox values")
        try:
            class_id = int(fields[0])
            center_x, center_y, width, height = (float(value) for value in fields[1:])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number} contains a non-numeric YOLO label") from exc
        if class_id < 0:
            raise ValueError(f"{path}:{line_number} has a negative class id")
        if not (0 <= center_x <= 1 and 0 <= center_y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"{path}:{line_number} has an invalid normalized bbox")


def write_template_calibration_rows(
    case: OfflineCalibrationCase,
    gate: TemplateGate,
    patchcore_crop_dir: Path,
    output_csv: Path,
) -> None:
    """Score all canonical views without applying the online short-circuit."""
    rows = []
    for view in ZS32_VIEWS:
        result = gate.evaluate(patchcore_crop_dir / f"{view}.png", case.hand, view)
        rows.append(
            {
                "part_id": case.part_id,
                "hand": case.hand,
                "view": view,
                "branch": "template_match",
                "raw_score": result.risk_score,
                "gt_label": case.gt_label,
                "split": case.split,
                "model_version": result.versions["model"],
                "roi_version": result.versions["roi"],
            },
        )
    _write_rows_atomic(output_csv, rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _completed_runtime_case(  # noqa: PLR0911
    case: OfflineCalibrationCase,
    case_dir: Path,
    *,
    runtime_config_sha256: str,
) -> bool:
    manifest_path = case_dir / "runtime_manifest.json"
    model_rows_path = case_dir / "calibration_rows.csv"
    if not manifest_path.is_file() or not model_rows_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest = {
        "part_id": case.part_id,
        "capture_session": case.session_id,
        "group_id": case.group_id,
        "hand": case.hand,
        "errors": [],
        "runtime_config_sha256": runtime_config_sha256,
        "source_images": {view: str(case.images[view]) for view in ZS32_VIEWS},
    }
    if any(manifest.get(key) != value for key, value in expected_manifest.items()):
        return False
    rows = _read_csv(model_rows_path)
    expected_keys = {(view, f"anomaly_{view}") for view in ZS32_VIEWS}
    expected_keys |= {(view, "yolo") for view in ZS32_VIEWS}
    if len(rows) != len(expected_keys) or {(row.get("view"), row.get("branch")) for row in rows} != expected_keys:
        return False
    if any(
        row.get("part_id") != case.part_id
        or row.get("hand") != case.hand
        or row.get("gt_label") != str(case.gt_label)
        or row.get("split") != case.split
        for row in rows
    ):
        return False
    source_hashes = {view: _sha256_file(case.images[view]) for view in ZS32_VIEWS}
    for name in ("patchcore.csv", "yolo.csv"):
        evidence_path = case_dir / name
        if not evidence_path.is_file():
            return False
        evidence_rows = _read_csv(evidence_path)
        if len(evidence_rows) != len(ZS32_VIEWS):
            return False
        if any(
            row.get("source_path") != str(case.images.get(row.get("view", ""), ""))
            or row.get("source_hash") != source_hashes.get(row.get("view", ""))
            for row in evidence_rows
        ):
            return False
    return True


def run_offline_batch(
    cases: Sequence[OfflineCalibrationCase],
    runtime: ZS32ModelRuntime,
    template_model_dir: Path,
    output_dir: Path,
    *,
    resume: bool = False,
) -> dict[str, int]:
    """Score cases with persistent model backends and publish resumable state."""
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and not resume:
        raise FileExistsError(f"output directory already exists; pass --resume: {output_dir}")
    case_root = output_dir / "cases"
    case_root.mkdir(parents=True, exist_ok=True)
    gate = TemplateGate(template_model_dir.expanduser().resolve())
    runtime_config_sha256 = _sha256_file(runtime.config.path)
    completed = 0
    skipped = 0
    for index, case in enumerate(cases, start=1):
        case_dir = case_root / case.slug
        if case_dir.exists():
            if not resume or not _completed_runtime_case(
                case,
                case_dir,
                runtime_config_sha256=runtime_config_sha256,
            ):
                raise ValueError(f"existing case publication is incomplete or invalid: {case_dir}")
            skipped += 1
        else:
            result = runtime.run(
                case.request(),
                case_dir,
                gt_label=case.gt_label,
                split=case.split,
            )
            if result.errors:
                raise RuntimeError(f"model inference failed for {case.part_id}: {result.errors}")
            completed += 1
        write_template_calibration_rows(
            case,
            gate,
            case_dir / "crops" / "patchcore",
            case_dir / "template_calibration_rows.csv",
        )
        state = {
            "schema_version": "1.0",
            "commissioning_only": True,
            "data_leakage_risk": True,
            "case_count": len(cases),
            "last_completed_index": index,
            "last_completed_part_id": case.part_id,
            "completed_this_run": completed,
            "resumed_cases": skipped,
        }
        temporary = output_dir / ".batch_state.json.tmp"
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output_dir / "batch_state.json")
        print(f"[{index}/{len(cases)}] {case.part_id}", flush=True)
    return {"case_count": len(cases), "completed": completed, "resumed": skipped}


def write_case_index(cases: Iterable[OfflineCalibrationCase], output_csv: Path) -> None:
    """Publish the exact physical-part inputs used by the commissioning run."""
    fields = (
        "part_id",
        "hand",
        "gt_label",
        "split",
        "session_id",
        "group_id",
        "defect_type",
        *(f"{view}_source_path" for view in ZS32_VIEWS),
    )
    rows = []
    for case in cases:
        row: dict[str, Any] = {
            "part_id": case.part_id,
            "hand": case.hand,
            "gt_label": case.gt_label,
            "split": case.split,
            "session_id": case.session_id,
            "group_id": case.group_id,
            "defect_type": case.defect_type,
        }
        row.update({f"{view}_source_path": case.images[view] for view in ZS32_VIEWS})
        rows.append(row)
    if output_csv.exists():
        return
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
