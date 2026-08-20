"""Prepare truthful branch manifests from BMW eight-view HDR captures."""

from __future__ import annotations

import csv
import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2


VIEW_ORDER = (
    "front",
    "front_left",
    "front_right",
    "front_secondary",
    "back",
    "back_left",
    "back_right",
    "back_secondary",
)
CAPTURE_FIELDS = (
    "record_type",
    "session_id",
    "sample_id",
    "group_id",
    "image_index",
    "round",
    "view",
    "device_index",
    "camera_serial",
    "capture_mode",
    "exposure",
    "gain",
    "file",
    "source_short",
    "source_long",
    "short_exposure",
    "long_exposure",
    "hdr_attempt",
    "fused_clip_pct",
    "captured_at",
    "sample_status",
    "failed_round",
    "failed_view",
    "failed_device_index",
    "error",
)
SPLITS = ("train", "calibration", "final_test")
SOURCE_CLASSES = ("normal", "deform", "edge", "others", "no_streak")
CAPTURE_SCOPES = ("left", "right")
_SERIAL_BY_VIEW = {
    "front": "DA9805574",
    "front_left": "DA9625347",
    "front_right": "DB0998274",
    "front_secondary": "DB0968108",
    "back": "DA9805574",
    "back_left": "DA9625347",
    "back_right": "DB0998274",
    "back_secondary": "DB0968108",
}
_DATASET_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SAMPLE_ID = re.compile(r"(?P<physical_part_id>.+)_(?P<image_index>[0-9]{6})\Z")

_DATASET_FIELDS = (
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
)
_BRANCH_FIELDS = _DATASET_FIELDS + ("branch_label", "review_reason")
_YOLO_FIELDS = _DATASET_FIELDS + ("branch_label", "annotation_status", "class_name", "review_reason")
_STREAK_FIELDS = _DATASET_FIELDS + ("expected_status", "review_reason")
_PART_FIELDS = ("physical_part_id", "session_id", "group_id", "source_class", "split")


@dataclass(frozen=True, slots=True)
class PreparedImage:
    """One validated image belonging to a complete physical-part capture."""

    sample_id: str
    physical_part_id: str
    session_id: str
    group_id: str
    view_id: str
    camera_serial: str
    source_path: Path
    source_class: str
    source_sha256: str


@dataclass(frozen=True, slots=True)
class CaptureAudit:
    """Read-only audit counts from the raw append-only capture manifests."""

    manifest_count: int
    complete_sample_count: int
    incomplete_sample_count: int
    excluded_incomplete_image_count: int
    image_width: int
    image_height: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(row: Mapping[str, str], field: str, *, context: str) -> str:
    value = row.get(field, "").strip()
    if not value:
        raise ValueError(f"{context} has empty {field}")
    return value


def _source_class(raw_root: Path, image_path: Path, view_id: str) -> str:
    try:
        relative = image_path.relative_to(raw_root)
    except ValueError as error:
        raise ValueError(f"capture image escapes raw root: {image_path}") from error
    parts = relative.parts
    try:
        view_index = parts.index(view_id)
    except ValueError as error:
        raise ValueError(f"capture path does not contain view {view_id}: {image_path}") from error
    tail = parts[view_index + 1 :]
    if tail and tail[0] == "normal":
        return "normal"
    if len(tail) >= 2 and tail[0] == "defect" and tail[1] in SOURCE_CLASSES[1:]:
        return tail[1]
    raise ValueError(f"capture path has unsupported label layout: {image_path}")


def _capture_scope(raw_root: Path, image_path: Path) -> str:
    """Return the top-level legacy capture scope for one source image."""
    try:
        relative = image_path.relative_to(raw_root)
    except ValueError as error:
        raise ValueError(f"capture image escapes raw root: {image_path}") from error
    if not relative.parts or relative.parts[0] not in CAPTURE_SCOPES:
        raise ValueError(f"capture path has unsupported hand scope: {image_path}")
    return relative.parts[0]


def _read_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CAPTURE_FIELDS:
            raise ValueError(f"capture manifest header differs from the frozen schema: {path}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"capture manifest contains malformed or extra cells: {path}")
    return rows


def _capture_manifest_paths(root: Path, session_ids: Sequence[str] | None) -> tuple[Path, ...]:
    """Resolve all or an explicit subset of immutable capture manifests."""
    manifest_root = root / "manifests"
    if session_ids is None:
        paths = tuple(sorted(manifest_root.glob("*.csv")))
        if not paths:
            raise ValueError(f"no capture manifests found below {manifest_root}")
        return paths
    if isinstance(session_ids, (str, bytes)):
        raise TypeError("session_ids must be a sequence of session ID strings")
    requested = tuple(session_ids)
    if not requested:
        raise ValueError("session_ids must not be empty")
    if len(set(requested)) != len(requested):
        raise ValueError("session_ids must not contain duplicates")
    paths: list[Path] = []
    for session_id in requested:
        if not isinstance(session_id, str) or not _DATASET_ID.fullmatch(session_id):
            raise ValueError(f"invalid capture session ID: {session_id!r}")
        path = manifest_root / f"{session_id}.csv"
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"capture session manifest does not exist: {path}")
        paths.append(path)
    return tuple(sorted(paths))


def read_complete_capture_rows(
    raw_root: Path,
    *,
    verify_image_hash: bool = True,
    capture_scope: str | None = None,
    session_ids: Sequence[str] | None = None,
) -> tuple[tuple[PreparedImage, ...], CaptureAudit]:
    """Read complete BMW capture samples and reject topology or identity drift."""
    if capture_scope is not None and capture_scope not in CAPTURE_SCOPES:
        raise ValueError(f"capture_scope must be one of {CAPTURE_SCOPES}")
    root = Path(raw_root).expanduser().resolve()
    manifest_paths = _capture_manifest_paths(root, session_ids)

    sample_rows: dict[tuple[str, str], dict[str, str]] = {}
    images_by_sample: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    scopes_by_session: dict[str, set[str]] = defaultdict(set)
    for manifest_path in manifest_paths:
        for row in _read_manifest_rows(manifest_path):
            record_type = row["record_type"].strip()
            session_id = _required_text(row, "session_id", context=str(manifest_path))
            if session_id != manifest_path.stem:
                raise ValueError(f"capture row session_id differs from manifest filename: {manifest_path}")
            sample_id = _required_text(row, "sample_id", context=str(manifest_path))
            identity = (session_id, sample_id)
            if record_type == "sample":
                if identity in sample_rows:
                    raise ValueError(f"duplicate sample status row for {session_id}/{sample_id}")
                sample_rows[identity] = row
            elif record_type == "image":
                images_by_sample[identity].append(row)
                image_path = Path(_required_text(row, "file", context=str(manifest_path))).expanduser().resolve()
                scopes_by_session[session_id].add(_capture_scope(root, image_path))
            else:
                raise ValueError(f"unsupported capture record_type {record_type!r} in {manifest_path}")

    orphan_images = sorted(set(images_by_sample) - set(sample_rows))
    if orphan_images:
        raise ValueError(f"capture image rows have no sample status row: {orphan_images[:3]}")

    prepared: list[PreparedImage] = []
    complete_count = 0
    incomplete_count = 0
    excluded_incomplete_images = 0
    observed_dimensions: set[tuple[int, int]] = set()
    content_part: dict[str, str] = {}
    part_classes: dict[str, str] = {}
    view_index = {view: index for index, view in enumerate(VIEW_ORDER)}
    included_sessions: set[str] = set()

    for identity, sample_row in sorted(sample_rows.items()):
        session_id, sample_id = identity
        status = sample_row["sample_status"].strip()
        image_rows = images_by_sample.get(identity, [])
        sample_scopes = {
            _capture_scope(
                root,
                Path(_required_text(row, "file", context=f"sample {session_id}/{sample_id}")).expanduser().resolve(),
            )
            for row in image_rows
        }
        if len(sample_scopes) > 1:
            raise ValueError(f"sample {session_id}/{sample_id} mixes capture hand scopes")
        effective_scopes = sample_scopes or scopes_by_session.get(session_id, set())
        if capture_scope is not None and effective_scopes != {capture_scope}:
            continue
        included_sessions.add(session_id)
        if status == "incomplete":
            incomplete_count += 1
            excluded_incomplete_images += len(image_rows)
            continue
        if status != "complete":
            raise ValueError(f"sample {session_id}/{sample_id} has unsupported status {status!r}")
        if any(sample_row[field].strip() for field in ("failed_round", "failed_view", "failed_device_index", "error")):
            raise ValueError(f"complete sample {session_id}/{sample_id} contains failure metadata")
        complete_count += 1
        group_id = _required_text(sample_row, "group_id", context=f"sample {session_id}/{sample_id}")
        sample_match = _SAMPLE_ID.fullmatch(sample_id)
        if sample_match is None:
            raise ValueError(f"sample_id does not contain a six-digit image index: {sample_id}")
        physical_part_id = sample_match.group("physical_part_id")
        if not physical_part_id.endswith(f"_{group_id}"):
            raise ValueError(f"sample_id does not bind group_id for {session_id}/{sample_id}")
        expected_image_index = str(int(sample_match.group("image_index")))
        if str(int(_required_text(sample_row, "image_index", context=f"sample {session_id}/{sample_id}"))) != expected_image_index:
            raise ValueError(f"sample_id image index differs from sample row for {session_id}/{sample_id}")
        if len(image_rows) != len(VIEW_ORDER):
            raise ValueError(f"complete sample {session_id}/{sample_id} must contain exactly eight views")
        rows_by_view: dict[str, dict[str, str]] = {}
        sample_classes: set[str] = set()
        for image_row in image_rows:
            if image_row["sample_status"].strip() != "complete":
                raise ValueError(f"complete sample {session_id}/{sample_id} contains a non-complete image row")
            if image_row["group_id"].strip() != group_id:
                raise ValueError(f"sample {session_id}/{sample_id} mixes group identities")
            if str(int(_required_text(image_row, "image_index", context=f"sample {session_id}/{sample_id}"))) != expected_image_index:
                raise ValueError(f"sample {session_id}/{sample_id} mixes image indices")
            view_id = _required_text(image_row, "view", context=f"sample {session_id}/{sample_id}")
            if view_id not in view_index or view_id in rows_by_view:
                raise ValueError(f"complete sample {session_id}/{sample_id} must contain exactly eight views")
            expected_round = "front" if view_id.startswith("front") else "back"
            if image_row["round"].strip() != expected_round:
                raise ValueError(f"view {view_id} has wrong capture round in sample {session_id}/{sample_id}")
            serial = _required_text(image_row, "camera_serial", context=f"sample {session_id}/{sample_id}")
            if serial != _SERIAL_BY_VIEW[view_id]:
                raise ValueError(f"view {view_id} has unexpected camera serial {serial}")
            if image_row["capture_mode"].strip() != "hdr_fused":
                raise ValueError(f"view {view_id} is not an HDR fused capture")
            image_path = Path(_required_text(image_row, "file", context=f"sample {session_id}/{sample_id}"))
            image_path = image_path.expanduser().resolve()
            if not image_path.is_file():
                raise ValueError(f"capture image does not exist: {image_path}")
            if image_path.is_symlink():
                raise ValueError(f"capture image must be a regular non-symlink file: {image_path}")
            source_class = _source_class(root, image_path, view_id)
            sample_classes.add(source_class)
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None or image.size == 0:
                raise ValueError(f"cannot decode capture image: {image_path}")
            height, width = image.shape[:2]
            observed_dimensions.add((width, height))
            digest = _sha256(image_path) if verify_image_hash else ""
            if digest:
                previous_part = content_part.setdefault(digest, physical_part_id)
                if previous_part != physical_part_id:
                    raise ValueError("same image content is assigned to multiple physical parts")
            rows_by_view[view_id] = image_row
            prepared.append(
                PreparedImage(
                    sample_id=sample_id,
                    physical_part_id=physical_part_id,
                    session_id=session_id,
                    group_id=group_id,
                    view_id=view_id,
                    camera_serial=serial,
                    source_path=image_path,
                    source_class=source_class,
                    source_sha256=digest,
                )
            )
        if set(rows_by_view) != set(VIEW_ORDER) or len(sample_classes) != 1:
            raise ValueError(f"complete sample {session_id}/{sample_id} has inconsistent views or labels")
        source_class = next(iter(sample_classes))
        previous_class = part_classes.setdefault(physical_part_id, source_class)
        if previous_class != source_class:
            raise ValueError(f"physical part {physical_part_id} crosses source classes")

    if not prepared:
        raise ValueError("capture root contains no complete eight-view samples")
    if len(observed_dimensions) != 1:
        raise ValueError(f"complete capture images have inconsistent dimensions: {sorted(observed_dimensions)}")
    width, height = next(iter(observed_dimensions))
    prepared.sort(key=lambda row: (row.physical_part_id, row.sample_id, view_index[row.view_id]))
    return (
        tuple(prepared),
        CaptureAudit(
            manifest_count=len(manifest_paths) if capture_scope is None else len(included_sessions),
            complete_sample_count=complete_count,
            incomplete_sample_count=incomplete_count,
            excluded_incomplete_image_count=excluded_incomplete_images,
            image_width=width,
            image_height=height,
        ),
    )


def _split_counts(count: int) -> tuple[int, int, int]:
    if count <= 0:
        raise ValueError("split count must be positive")
    if count == 1:
        return 1, 0, 0
    if count == 2:
        return 1, 0, 1
    train = max(1, round(count * 0.60))
    calibration = max(1, round(count * 0.20))
    final_test = count - train - calibration
    if final_test < 1:
        train -= 1
        final_test += 1
    return train, calibration, final_test


def assign_stratified_part_splits(
    rows: Sequence[PreparedImage],
    *,
    seed: int = 42,
) -> dict[str, str]:
    """Assign deterministic 60/20/20 splits within each source class."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    class_by_part: dict[str, str] = {}
    for row in rows:
        previous = class_by_part.setdefault(row.physical_part_id, row.source_class)
        if previous != row.source_class:
            raise ValueError(f"physical part {row.physical_part_id} crosses source classes")
    if not class_by_part:
        raise ValueError("rows must contain at least one physical part")
    parts_by_class: dict[str, list[str]] = defaultdict(list)
    for part_id, source_class in class_by_part.items():
        parts_by_class[source_class].append(part_id)
    split_by_part: dict[str, str] = {}
    for source_class, part_ids in sorted(parts_by_class.items()):
        ordered = sorted(
            part_ids,
            key=lambda part_id: hashlib.sha256(f"{seed}:{source_class}:{part_id}".encode()).hexdigest(),
        )
        train_count, calibration_count, _final_count = _split_counts(len(ordered))
        for index, part_id in enumerate(ordered):
            if index < train_count:
                split = "train"
            elif index < train_count + calibration_count:
                split = "calibration"
            else:
                split = "final_test"
            split_by_part[part_id] = split
    return dict(sorted(split_by_part.items()))


def _dataset_row(row: PreparedImage, split: str) -> dict[str, str]:
    return {
        "sample_id": row.sample_id,
        "physical_part_id": row.physical_part_id,
        "session_id": row.session_id,
        "group_id": row.group_id,
        "view_id": row.view_id,
        "camera_serial": row.camera_serial,
        "source_path": str(row.source_path),
        "source_sha256": row.source_sha256,
        "source_class": row.source_class,
        "business_label": "OK" if row.source_class == "normal" else "NG",
        "split": split,
    }


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _atomic_publish_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a staged directory without replacing another writer."""
    if os.name == "posix":
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is not None:
            renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
            renameat2.restype = ctypes.c_int
            result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
            if result == 0:
                return
            error_number = ctypes.get_errno()
            if error_number == errno.EEXIST:
                raise FileExistsError(f"dataset release already exists: {destination}")
            if error_number not in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
                raise OSError(error_number, os.strerror(error_number), destination)
    reservation = destination.parent / f".{destination.name}.publish-reservation"
    try:
        reservation.mkdir()
    except FileExistsError as error:
        raise FileExistsError(f"another dataset publisher is active: {destination}") from error
    try:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"dataset release already exists: {destination}")
        source.rename(destination)
    finally:
        reservation.rmdir()


def _branch_rows(
    rows: Sequence[PreparedImage],
    split_by_part: Mapping[str, str],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    efficientad: list[dict[str, str]] = []
    template: list[dict[str, str]] = []
    yolo: list[dict[str, str]] = []
    streak: list[dict[str, str]] = []
    for row in rows:
        base = _dataset_row(row, split_by_part[row.physical_part_id])
        is_branch_normal = row.source_class in {"normal", "no_streak"}
        branch_label = "normal" if is_branch_normal else "review_required"
        review_reason = "" if is_branch_normal else "defect_visibility_requires_review"
        efficientad.append({**base, "branch_label": branch_label, "review_reason": review_reason})
        template.append({**base, "branch_label": branch_label, "review_reason": review_reason})
        if is_branch_normal:
            yolo.append(
                {
                    **base,
                    "branch_label": "normal",
                    "annotation_status": "negative_confirmed",
                    "class_name": "",
                    "review_reason": "",
                }
            )
        else:
            yolo.append(
                {
                    **base,
                    "branch_label": "review_required",
                    "annotation_status": "review_required",
                    "class_name": "defect",
                    "review_reason": "bbox_and_visibility_require_review",
                }
            )
        if row.view_id == "front_left":
            expected = {"normal": "OK", "no_streak": "NG_NO_STREAK"}.get(row.source_class, "REVIEW")
            streak.append(
                {
                    **base,
                    "expected_status": expected,
                    "review_reason": "" if expected != "REVIEW" else "streak_state_requires_review",
                }
            )
    return efficientad, template, yolo, streak


def _report(
    *,
    dataset_id: str,
    raw_root: Path,
    rows: Sequence[PreparedImage],
    audit: CaptureAudit,
    split_by_part: Mapping[str, str],
    seed: int,
    verify_image_hash: bool,
    release_status: str,
    capture_scope: str | None,
    session_ids: Sequence[str] | None,
) -> dict[str, object]:
    source_by_part = {row.physical_part_id: row.source_class for row in rows}
    return {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "release_status": release_status,
        "raw_root": str(raw_root),
        "capture_scope": capture_scope or "all",
        "session_ids": sorted(session_ids) if session_ids is not None else "all",
        "identity_basis": "sample_id_without_six_digit_image_index",
        "physical_part_count": len(split_by_part),
        "complete_sample_count": audit.complete_sample_count,
        "incomplete_sample_count": audit.incomplete_sample_count,
        "excluded_incomplete_image_count": audit.excluded_incomplete_image_count,
        "manifest_count": audit.manifest_count,
        "image_count": len(rows),
        "image_width": audit.image_width,
        "image_height": audit.image_height,
        "source_class_part_counts": dict(sorted(Counter(source_by_part.values()).items())),
        "view_image_counts": dict(sorted(Counter(row.view_id for row in rows).items())),
        "split_part_counts": {split: list(split_by_part.values()).count(split) for split in SPLITS},
        "split_seed": seed,
        "image_sha256_verified": verify_image_hash,
        "experimental_only": True,
        "remaining_requirements": [
            "select eight per-view part ROIs",
            "review defect visibility for EfficientAD and Template",
            "annotate visible YOLO defect boxes",
            "calibrate the HDR front_left bright-streak rule",
        ],
    }


def prepare_eight_view_dataset(
    *,
    raw_root: Path,
    output_root: Path,
    dataset_id: str,
    seed: int = 42,
    verify_image_hash: bool = True,
    dry_run: bool = False,
    capture_scope: str | None = None,
    session_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """Validate raw captures and atomically publish branch-specific manifests."""
    if not isinstance(dataset_id, str) or not _DATASET_ID.fullmatch(dataset_id):
        raise ValueError("dataset_id must contain only letters, digits, dot, underscore, or hyphen")
    raw = Path(raw_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    release = output / dataset_id
    if release.exists() or release.is_symlink():
        raise FileExistsError(f"dataset release already exists: {release}")
    rows, audit = read_complete_capture_rows(
        raw,
        verify_image_hash=verify_image_hash,
        capture_scope=capture_scope,
        session_ids=session_ids,
    )
    split_by_part = assign_stratified_part_splits(rows, seed=seed)
    report = _report(
        dataset_id=dataset_id,
        raw_root=raw,
        rows=rows,
        audit=audit,
        split_by_part=split_by_part,
        seed=seed,
        verify_image_hash=verify_image_hash,
        release_status="dry_run" if dry_run else "published",
        capture_scope=capture_scope,
        session_ids=session_ids,
    )
    if dry_run:
        return report

    output.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dataset_id}.", dir=output))
    try:
        dataset_rows = [_dataset_row(row, split_by_part[row.physical_part_id]) for row in rows]
        efficientad, template, yolo, streak = _branch_rows(rows, split_by_part)
        part_identity: dict[str, PreparedImage] = {}
        for row in rows:
            part_identity.setdefault(row.physical_part_id, row)
        part_rows = [
            {
                "physical_part_id": part_id,
                "session_id": part_identity[part_id].session_id,
                "group_id": part_identity[part_id].group_id,
                "source_class": part_identity[part_id].source_class,
                "split": split_by_part[part_id],
            }
            for part_id in sorted(part_identity)
        ]
        manifest_specs = {
            "dataset_manifest.csv": (_DATASET_FIELDS, dataset_rows),
            "part_splits.csv": (_PART_FIELDS, part_rows),
            "efficientad.csv": (_BRANCH_FIELDS, efficientad),
            "template.csv": (_BRANCH_FIELDS, template),
            "yolo_annotation.csv": (_YOLO_FIELDS, yolo),
            "bright_streak.csv": (_STREAK_FIELDS, streak),
        }
        for name, (fields, manifest_rows) in manifest_specs.items():
            _write_csv(staging / "manifests" / name, fields, manifest_rows)
        report["manifest_sha256"] = {
            name: _sha256(staging / "manifests" / name) for name in sorted(manifest_specs)
        }
        (staging / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _atomic_publish_noreplace(staging, release)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report
