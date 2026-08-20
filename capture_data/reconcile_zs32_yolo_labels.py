# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: C901, EM101, EM102, TRY003, TRY004

"""Reconcile eight-view ZS32 YOLO labels by image content and physical group."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import cv2
import numpy as np
from capture_data.zs32_view_roi_dataset import load_roi_config
from rich.progress import track

from zs32_inspection.data.roi import LabelDocument, migrate_yolo_labels, parse_yolo_labels
from zs32_inspection.data.splitter import SPLIT_ALGORITHM, SplitPolicy, assign_grouped_splits
from zs32_inspection.domain.contracts import RoiBox
from zs32_inspection.domain.views import CANONICAL_VIEWS

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


GROUP_PATTERN = re.compile(r"_(group\d{3})_")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MAPPING_COLUMNS = (
    "original_image_path",
    "new_image_path",
    "view",
    "session_group",
    "sample_id",
    "split",
    "match_method",
    "original_label_path",
    "new_label_path",
    "is_reused",
    "is_duplicate",
    "is_conflict",
    "needs_human_annotation",
    "file_sha256",
    "pixel_sha256",
    "perceptual_hash",
    "canonical_original_path",
)

DEFECT_840_SESSION_SPLIT = {
    "zs32_right_deform_20260714_202519_099232811": "train",
    "zs32_right_deform_20260714_210638_381004531": "train",
    "zs32_right_deform_20260714_212547_690787547": "train",
    "zs32_right_deform_20260714_201750_036166226": "val",
    "zs32_right_deform_20260714_201953_436127442": "val",
    "zs32_right_deform_20260714_200750_268230020": "test",
    "zs32_right_deform_20260714_201404_438451409": "test",
}


@dataclass(frozen=True, slots=True)
class ImageFingerprint:
    """Content identity for encoded bytes, decoded pixels, mirrors, and near matches."""

    file_sha256: str
    pixel_sha256: str
    flipped_pixel_sha256: str
    perceptual_hash: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class SourceImage:
    """One source image with path-derived physical identity."""

    path: Path
    hand: str
    view: str
    label: str
    defect_type: str
    session_id: str
    group_id: str

    @property
    def sample_id(self) -> str:
        """Physical group identity that cannot collide across sessions."""
        defect = self.defect_type if self.label == "defect" else "none"
        return f"{self.hand}/{self.label}/{defect}/{self.session_id}/{self.group_id}"

    @property
    def stratum(self) -> str:
        """Group split stratum."""
        return f"{self.hand}:{self.label}:{self.defect_type or 'none'}"


@dataclass(frozen=True, slots=True)
class LabelCandidate:
    """One trusted crop-space annotation and its provenance."""

    text: str
    source_label_path: str
    source_image_path: str
    method: str
    clipped_boxes: int = 0
    dropped_boxes: int = 0


@dataclass(frozen=True, slots=True)
class CanonicalRecord:
    """One de-duplicated image and its reconciled annotation decision."""

    source: SourceImage
    aliases: tuple[SourceImage, ...]
    fingerprint: ImageFingerprint
    crop_fingerprint: ImageFingerprint
    split: str
    image_path: Path
    label_path: Path
    label_text: str
    match_method: str
    source_label_path: str
    is_reused: bool
    is_conflict: bool
    needs_human_annotation: bool
    candidate_labels: tuple[LabelCandidate, ...]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _pixel_sha256(image: np.ndarray) -> str:
    header = f"{image.dtype.str}:{image.shape}".encode()
    return _sha256_bytes(header + image.tobytes(order="C"))


def _dhash(image: np.ndarray) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def _fingerprint_array(image: np.ndarray, *, encoded_sha256: str = "") -> ImageFingerprint:
    height, width = image.shape[:2]
    return ImageFingerprint(
        file_sha256=encoded_sha256,
        pixel_sha256=_pixel_sha256(image),
        flipped_pixel_sha256=_pixel_sha256(cv2.flip(image, 1)),
        perceptual_hash=_dhash(image),
        width=width,
        height=height,
    )


def fingerprint_image(path: Path) -> ImageFingerprint:
    """Hash encoded bytes, decoded pixels, a horizontal mirror, and a 64-bit dHash.

    Args:
        path (Path): Image to decode without color conversion beyond OpenCV's normal image loading.

    Returns:
        ImageFingerprint: Stable exact and approximate image identities.

    Raises:
        ValueError: If the image cannot be decoded.
    """
    payload = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return _fingerprint_array(image, encoded_sha256=_sha256_bytes(payload))


def _canonical_boxes(text: str) -> tuple[tuple[int, float, float, float, float], ...]:
    boxes = []
    for _line_number, _original, box in parse_yolo_labels(text, source="label comparison"):
        boxes.append((box.class_id, box.cx, box.cy, box.width, box.height))
    return tuple(sorted(boxes))


def labels_equivalent(left: str, right: str, *, tolerance: float = 1e-6) -> bool:
    """Compare normalized YOLO boxes independent of line ordering and harmless rounding."""
    left_boxes = _canonical_boxes(left)
    right_boxes = _canonical_boxes(right)
    if len(left_boxes) != len(right_boxes):
        return False
    return all(
        left_box[0] == right_box[0]
        and all(
            math.isclose(a, b, abs_tol=tolerance, rel_tol=0) for a, b in zip(left_box[1:], right_box[1:], strict=True)
        )
        for left_box, right_box in zip(left_boxes, right_boxes, strict=True)
    )


def horizontal_flip_yolo(text: str, *, source: str) -> str:
    """Flip crop-space normalized YOLO boxes after a proven horizontal image mirror."""
    lines = []
    for _line_number, _original, box in parse_yolo_labels(text, source=source):
        lines.append(
            f"{box.class_id} {1 - box.cx:.8f} {box.cy:.8f} {box.width:.8f} {box.height:.8f}",
        )
    return "\n".join(lines) + ("\n" if lines else "")


class _DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def split_alias_groups(
    strata_by_group: Mapping[str, str],
    *,
    alias_pairs: Sequence[tuple[str, str]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    """Assign physical groups while binding renamed duplicate groups to one split."""
    aliases = _DisjointSet(strata_by_group)
    for left, right in alias_pairs:
        if left not in strata_by_group or right not in strata_by_group:
            raise ValueError(f"Alias references an unknown group: {(left, right)!r}")
        aliases.union(left, right)
    members: dict[str, list[str]] = defaultdict(list)
    for group_id in strata_by_group:
        members[aliases.find(group_id)].append(group_id)
    component_strata: dict[str, str] = {}
    component_for_group: dict[str, str] = {}
    for group_ids in members.values():
        strata = {strata_by_group[group_id] for group_id in group_ids}
        if len(strata) != 1:
            raise ValueError(
                f"Duplicate images cross semantic strata: groups={sorted(group_ids)}, strata={sorted(strata)}",
            )
        component_id = "alias:" + _sha256_bytes("\n".join(sorted(group_ids)).encode())
        component_strata[component_id] = strata.pop()
        for group_id in group_ids:
            component_for_group[group_id] = component_id
    policy = SplitPolicy(
        algorithm=SPLIT_ALGORITHM,
        seed=seed,
        calibration_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    component_split = {
        assignment.part_instance_id: ("val" if assignment.split == "calibration" else assignment.split)
        for assignment in assign_grouped_splits(component_strata, policy=policy)
    }
    return {group_id: component_split[component_for_group[group_id]] for group_id in strata_by_group}


def build_label_studio_task(
    *,
    task_id: int,
    image_url: str,
    label_text: str | None,
    fingerprint: ImageFingerprint,
    metadata: Mapping[str, Any],
    model_version: str,
) -> dict[str, Any]:
    """Build one Label Studio local-file task with optional rectangle preannotations."""
    task: dict[str, Any] = {
        "id": task_id,
        "data": {"image": image_url, **metadata},
        "meta": dict(metadata),
    }
    if label_text is None:
        return task
    results = []
    for index, (_line_number, _original, box) in enumerate(parse_yolo_labels(label_text, source=image_url), start=1):
        results.append(
            {
                "id": f"pred-{task_id}-{index}",
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "original_width": fingerprint.width,
                "original_height": fingerprint.height,
                "image_rotation": 0,
                "value": {
                    "x": (box.cx - box.width / 2) * 100,
                    "y": (box.cy - box.height / 2) * 100,
                    "width": box.width * 100,
                    "height": box.height * 100,
                    "rotation": 0,
                    "rectanglelabels": ["defect"],
                },
            },
        )
    task["predictions"] = [{"model_version": model_version, "score": 1.0, "result": results}]
    return task


def discover_source_images(dataset_root: Path) -> list[SourceImage]:
    """Discover complete eight-view raw groups from the path contract."""
    images: list[SourceImage] = []
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(dataset_root)
        parts = relative.parts
        if len(parts) == 6 and parts[2] == "normal" and parts[4] == "images":
            hand, view, label, session_id, _images, _name = parts
            defect_type = ""
        elif len(parts) == 7 and parts[2] == "defect" and parts[5] == "images":
            hand, view, label, defect_type, session_id, _images, _name = parts
        else:
            continue
        if hand not in {"left", "right"} or view not in CANONICAL_VIEWS:
            raise ValueError(f"Unsupported ZS32 image path: {path}")
        match = GROUP_PATTERN.search(path.name)
        if match is None:
            raise ValueError(f"Could not parse group_id from: {path}")
        images.append(SourceImage(path, hand, view, label, defect_type, session_id, match.group(1)))
    if not images:
        raise ValueError(f"No source images found under: {dataset_root}")
    views_by_group: dict[str, set[str]] = defaultdict(set)
    for image in images:
        if image.view in views_by_group[image.sample_id]:
            raise ValueError(f"Duplicate view inside physical group: {image.sample_id}/{image.view}")
        views_by_group[image.sample_id].add(image.view)
    incomplete = {
        group_id: sorted(views) for group_id, views in views_by_group.items() if views != set(CANONICAL_VIEWS)
    }
    if incomplete:
        raise ValueError(f"Every source group must contain exactly eight views: {incomplete}")
    return images


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _resolve_legacy_path(value: str, legacy_repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else legacy_repo_root / path


def _load_label_studio_candidates(
    *,
    labeling_root: Path,
    legacy_repo_root: Path,
    rois: Mapping[str, tuple[int, int, int, int]],
    image_width: int,
    image_height: int,
) -> dict[Path, list[LabelCandidate]]:
    manifest = _read_csv(labeling_root / "labeling_manifest.csv")
    rows_by_stem = {Path(row["labeling_path"]).stem: row for row in manifest}
    candidates: dict[Path, list[LabelCandidate]] = defaultdict(list)
    export_labels = [path for path in labeling_root.glob("*/labels/*.txt") if path.name != "classes.txt"]
    for label_path in sorted(export_labels):
        row = rows_by_stem.get(label_path.stem)
        if row is None:
            raise ValueError(f"Label Studio export cannot be mapped to its manifest: {label_path}")
        source_path = _resolve_legacy_path(row["source_path"], legacy_repo_root).resolve()
        text = label_path.read_text(encoding="utf-8")
        document = LabelDocument("annotated", text, str(label_path))
        migrated = migrate_yolo_labels(
            document,
            source_width=image_width,
            source_height=image_height,
            roi=RoiBox(*rois[row["view"]]),
        )
        candidates[source_path].append(
            LabelCandidate(
                migrated.text,
                str(label_path),
                str(source_path),
                "label_studio_export+stage29_roi",
                migrated.audit.clipped_box_count,
                migrated.audit.dropped_box_count,
            ),
        )
    return candidates


def _load_existing_rows(existing_yolo_root: Path, legacy_repo_root: Path) -> list[dict[str, str]]:
    rows = _read_csv(existing_yolo_root / "split_manifest.csv")
    output = []
    for row in rows:
        if row.get("kind") == "normal_mirror":
            continue
        resolved = dict(row)
        resolved["source_path"] = str(_resolve_legacy_path(row["source_path"], legacy_repo_root).resolve())
        resolved["output_image"] = str(_resolve_legacy_path(row["output_image"], legacy_repo_root).resolve())
        resolved["output_label"] = str(_resolve_legacy_path(row["output_label"], legacy_repo_root).resolve())
        output.append(resolved)
    return output


def _unique_candidates(candidates: Iterable[LabelCandidate]) -> tuple[LabelCandidate, ...]:
    unique: list[LabelCandidate] = []
    for candidate in candidates:
        if not any(labels_equivalent(candidate.text, existing.text) for existing in unique):
            unique.append(candidate)
    return tuple(unique)


def _select_defect_preannotation(
    own_candidates: Sequence[LabelCandidate],
    duplicate_candidates: Sequence[LabelCandidate],
) -> tuple[LabelCandidate | None, str, bool, bool]:
    """Choose one source-specific preannotation without hiding duplicate conflicts."""
    unique = _unique_candidates(duplicate_candidates)
    conflict = len(unique) > 1 or any(candidate.dropped_boxes for candidate in duplicate_candidates)
    if own_candidates:
        selected = own_candidates[0]
        method = "source_label_studio_preannotation_conflict" if conflict else "source_label_studio_preannotation"
        return selected, method, conflict, conflict
    if len(unique) == 1 and not conflict:
        return unique[0], "sha256_duplicate_label_reuse", False, False
    if conflict:
        return None, "exact_duplicate_label_conflict", True, True
    return None, "missing_annotation", False, True


def _safe_link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _safe_name(source: SourceImage) -> str:
    defect = source.defect_type or "normal"
    return f"{source.hand}__{source.view}__{source.label}__{defect}__{source.session_id}__{source.group_id}.png"


def _hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _write_label_studio_config(path: Path) -> None:
    path.write_text(
        """<View>
  <Header value="ZS32 $view | $sample_id | match=$match_method"/>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="defect" background="#E53935"/>
  </RectangleLabels>
</View>
""",
        encoding="utf-8",
    )


def _validate_materialized_yolo(root: Path, records: Sequence[CanonicalRecord]) -> None:
    observed_groups: dict[str, str] = {}
    image_paths: set[Path] = set()
    label_paths: set[Path] = set()
    for record in records:
        previous = observed_groups.setdefault(record.source.sample_id, record.split)
        if previous != record.split:
            raise ValueError(f"Physical group leaks across splits: {record.source.sample_id}")
        image_paths.add(record.image_path)
        label_paths.add(record.label_path)
        image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode materialized image: {record.image_path}")
        parse_yolo_labels(record.label_path.read_text(encoding="utf-8"), source=str(record.label_path))
    discovered_images = {path for path in (root / "images").glob("*/*") if path.is_file()}
    discovered_labels = {path for path in (root / "labels").glob("*/*.txt") if path.is_file()}
    if image_paths != discovered_images or label_paths != discovered_labels:
        raise ValueError("YOLO images and labels are not one-to-one with the reconciliation manifest")


def reconcile_zs32_yolo_dataset(
    *,
    dataset_root: Path,
    roi_config: Path,
    labeling_root: Path,
    existing_yolo_root: Path,
    output_root: Path,
    legacy_repo_root: Path,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    perceptual_distance: int = 6,
    workers: int = 4,
) -> dict[str, Any]:
    """Create a de-duplicated draft YOLO dataset and Label Studio review tasks.

    The output is intentionally marked non-trainable while any label is missing or
    conflicted. A later Label Studio JSON export can be finalized with
    :func:`finalize_reconciled_dataset`.
    """
    dataset_root = dataset_root.resolve()
    output_root = output_root.resolve()
    protected = (dataset_root, labeling_root.resolve(), existing_yolo_root.resolve())
    if any(
        output_root == path or output_root.is_relative_to(path) or path.is_relative_to(output_root)
        for path in protected
    ):
        raise ValueError(f"Output must be independent from every source root: {output_root}")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")
    output_root.mkdir(parents=True)

    image_width, image_height, rois, roi_payload = load_roi_config(roi_config, expected_views=CANONICAL_VIEWS)
    sources = discover_source_images(dataset_root)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        fingerprints = dict(
            zip(
                (source.path.resolve() for source in sources),
                track(
                    executor.map(fingerprint_image, (source.path for source in sources)),
                    total=len(sources),
                    description="Hashing source images",
                ),
                strict=True,
            ),
        )

    existing_rows = _load_existing_rows(existing_yolo_root, legacy_repo_root)
    existing_sources = {Path(row["source_path"]).resolve() for row in existing_rows}
    by_exact: dict[tuple[str, str], list[SourceImage]] = defaultdict(list)
    for source in sources:
        by_exact[source.view, fingerprints[source.path.resolve()].pixel_sha256].append(source)
    aliases_by_canonical: dict[Path, tuple[SourceImage, ...]] = {}
    canonical_sources: list[SourceImage] = []
    alias_pairs: set[tuple[str, str]] = set()
    for aliases in by_exact.values():
        ordered = sorted(aliases, key=lambda item: (item.path.resolve() not in existing_sources, str(item.path)))
        canonical = ordered[0]
        canonical_sources.append(canonical)
        aliases_by_canonical[canonical.path.resolve()] = tuple(ordered)
        for alias in ordered[1:]:
            alias_pairs.add(tuple(sorted((canonical.sample_id, alias.sample_id))))
    for source in sources:
        fingerprint = fingerprints[source.path.resolve()]
        for mirrored in by_exact.get((source.view, fingerprint.flipped_pixel_sha256), ()):
            if mirrored.path.resolve() != source.path.resolve():
                alias_pairs.add(tuple(sorted((source.sample_id, mirrored.sample_id))))

    strata_by_group = {source.sample_id: source.stratum for source in sources}
    split_by_group = split_alias_groups(
        strata_by_group,
        alias_pairs=tuple(sorted(alias_pairs)),
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    candidates_by_source = _load_label_studio_candidates(
        labeling_root=labeling_root,
        legacy_repo_root=legacy_repo_root,
        rois=rois,
        image_width=image_width,
        image_height=image_height,
    )
    existing_by_source = {Path(row["source_path"]).resolve(): row for row in existing_rows}
    source_by_path = {source.path.resolve(): source for source in sources}
    flipped_candidates_by_crop: dict[tuple[str, str], list[LabelCandidate]] = defaultdict(list)
    for source_path, direct_candidates in candidates_by_source.items():
        source = source_by_path.get(source_path)
        if source is None:
            continue
        image = cv2.imread(str(source.path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode labeled source image: {source.path}")
        roi = RoiBox(*rois[source.view])
        crop_fingerprint = _fingerprint_array(image[roi.y1 : roi.y2, roi.x1 : roi.x2])
        flipped_candidates_by_crop[source.view, crop_fingerprint.flipped_pixel_sha256].extend(
            LabelCandidate(
                horizontal_flip_yolo(candidate.text, source=candidate.source_label_path),
                candidate.source_label_path,
                candidate.source_image_path,
                "horizontal_flip_pixel_sha256",
                candidate.clipped_boxes,
                candidate.dropped_boxes,
            )
            for candidate in direct_candidates
        )

    ordered_canonical = sorted(canonical_sources, key=lambda item: str(item.path))

    def prepare_crop(canonical: SourceImage) -> tuple[SourceImage, ImageFingerprint, Path | None, np.ndarray | None]:
        """Decode and verify one canonical ROI crop in a worker thread."""
        aliases = aliases_by_canonical[canonical.path.resolve()]
        image = cv2.imread(str(canonical.path), cv2.IMREAD_COLOR)
        if image is None or (image.shape[1], image.shape[0]) != (image_width, image_height):
            raise ValueError(f"Source image size mismatch: {canonical.path}")
        roi = RoiBox(*rois[canonical.view])
        crop = image[roi.y1 : roi.y2, roi.x1 : roi.x2]
        crop_fingerprint = _fingerprint_array(crop)
        reuse_row = next(
            (
                existing_by_source.get(alias.path.resolve())
                for alias in aliases
                if alias.path.resolve() in existing_by_source
            ),
            None,
        )
        if reuse_row is not None:
            existing_image = Path(reuse_row["output_image"])
            if fingerprint_image(existing_image).pixel_sha256 == crop_fingerprint.pixel_sha256:
                return canonical, crop_fingerprint, existing_image, None
        return canonical, crop_fingerprint, None, crop

    records: list[CanonicalRecord] = []
    conflicts: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        prepared_crops = executor.map(prepare_crop, ordered_canonical)
        for canonical, crop_fingerprint, reuse_image, crop in track(
            prepared_crops,
            total=len(ordered_canonical),
            description="Verifying and materializing ROI crops",
        ):
            aliases = aliases_by_canonical[canonical.path.resolve()]
            source_fingerprint = fingerprints[canonical.path.resolve()]

            candidates: list[LabelCandidate] = []
            for alias in aliases:
                candidates.extend(candidates_by_source.get(alias.path.resolve(), ()))
            if canonical.label == "defect" and not candidates:
                candidates.extend(
                    flipped_candidates_by_crop.get((canonical.view, crop_fingerprint.pixel_sha256), ()),
                )
            if canonical.label == "normal":
                candidates.append(
                    LabelCandidate("", "", str(canonical.path), "normal_directory_confirmed_empty"),
                )
            for alias in aliases:
                existing = existing_by_source.get(alias.path.resolve())
                if existing is None:
                    continue
                existing_label = Path(existing["output_label"]).read_text(encoding="utf-8")
                trusted = canonical.label == "normal" or alias.path.resolve() in candidates_by_source
                if trusted:
                    candidates.append(
                        LabelCandidate(
                            existing_label,
                            existing["output_label"],
                            existing["source_path"],
                            "existing_stage29_roi",
                        ),
                    )
            unique_candidates = _unique_candidates(candidates)
            conflict = len(unique_candidates) > 1 or any(candidate.dropped_boxes for candidate in candidates)
            if conflict:
                label_text = ""
                match_method = "label_conflict"
                source_label_path = ""
                reused = False
                needs_human = True
                conflicts.append(
                    {
                        "canonical_source": str(canonical.path),
                        "aliases": [str(alias.path) for alias in aliases],
                        "candidates": [asdict(candidate) for candidate in unique_candidates],
                    },
                )
            elif unique_candidates:
                selected = unique_candidates[0]
                label_text = selected.text
                source_label_path = selected.source_label_path
                reused = canonical.label == "defect"
                needs_human = False
                if any(
                    alias.path.resolve() != canonical.path.resolve() and alias.path.resolve() in candidates_by_source
                    for alias in aliases
                ):
                    match_method = "sha256_duplicate_label_reuse"
                else:
                    match_method = selected.method
            else:
                label_text = ""
                match_method = "missing_annotation"
                source_label_path = ""
                reused = False
                needs_human = True

            split = split_by_group[canonical.sample_id]
            name = _safe_name(canonical)
            ls_image = output_root / "label_studio" / "images" / canonical.view / name
            if reuse_image is not None:
                _safe_link(reuse_image, ls_image)
            else:
                assert crop is not None
                ls_image.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(ls_image), crop, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                    raise RuntimeError(f"Could not write crop: {ls_image}")
            yolo_image = output_root / "yolo_draft" / "images" / split / name
            yolo_label = output_root / "yolo_draft" / "labels" / split / f"{Path(name).stem}.txt"
            _safe_link(ls_image, yolo_image)
            yolo_label.parent.mkdir(parents=True, exist_ok=True)
            yolo_label.write_text(label_text, encoding="utf-8")
            records.append(
                CanonicalRecord(
                    canonical,
                    aliases,
                    source_fingerprint,
                    crop_fingerprint,
                    split,
                    yolo_image,
                    yolo_label,
                    label_text,
                    match_method,
                    source_label_path,
                    reused,
                    conflict,
                    needs_human,
                    unique_candidates,
                ),
            )

    record_by_path = {alias.path.resolve(): record for record in records for alias in record.aliases}
    approximate_rows = []
    annotated_by_view: dict[str, list[CanonicalRecord]] = defaultdict(list)
    for record in records:
        if record.is_reused and record.label_text:
            annotated_by_view[record.source.view].append(record)
    for record in records:
        if not record.needs_human_annotation or record.is_conflict:
            continue
        choices = [
            (_hamming(record.crop_fingerprint.perceptual_hash, candidate.crop_fingerprint.perceptual_hash), candidate)
            for candidate in annotated_by_view[record.source.view]
            if candidate.crop_fingerprint.pixel_sha256 != record.crop_fingerprint.pixel_sha256
        ]
        if not choices:
            continue
        distance, candidate = min(choices, key=lambda item: (item[0], str(item[1].source.path)))
        if distance <= perceptual_distance:
            approximate_rows.append(
                {
                    "unlabeled_image": str(record.source.path),
                    "candidate_image": str(candidate.source.path),
                    "view": record.source.view,
                    "hamming_distance": str(distance),
                    "auto_reused": "false",
                    "reason": "perceptual matches require human confirmation",
                },
            )

    mapping_rows = []
    for source in sorted(sources, key=lambda item: str(item.path)):
        record = record_by_path[source.path.resolve()]
        is_duplicate = source.path.resolve() != record.source.path.resolve()
        match_method = record.match_method
        if is_duplicate and record.is_reused and not record.is_conflict:
            match_method = "sha256_duplicate_label_reuse"
        mapping_rows.append(
            {
                "original_image_path": str(source.path),
                "new_image_path": str(record.image_path),
                "view": source.view,
                "session_group": f"{source.session_id}/{source.group_id}",
                "sample_id": source.sample_id,
                "split": record.split,
                "match_method": match_method,
                "original_label_path": record.source_label_path,
                "new_label_path": str(record.label_path),
                "is_reused": str(record.is_reused).lower(),
                "is_duplicate": str(is_duplicate).lower(),
                "is_conflict": str(record.is_conflict).lower(),
                "needs_human_annotation": str(record.needs_human_annotation).lower(),
                "file_sha256": fingerprints[source.path.resolve()].file_sha256,
                "pixel_sha256": fingerprints[source.path.resolve()].pixel_sha256,
                "perceptual_hash": fingerprints[source.path.resolve()].perceptual_hash,
                "canonical_original_path": str(record.source.path),
            },
        )
    _write_csv(output_root / "mapping.csv", MAPPING_COLUMNS, mapping_rows)
    if approximate_rows:
        _write_csv(output_root / "approximate_duplicates.csv", tuple(approximate_rows[0]), approximate_rows)
    else:
        _write_csv(
            output_root / "approximate_duplicates.csv",
            ("unlabeled_image", "candidate_image", "view", "hamming_distance", "auto_reused", "reason"),
            (),
        )
    (output_root / "conflicts.json").write_text(
        json.dumps(conflicts, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    tasks_all = []
    tasks_review = []
    annotation_rows = []
    for task_id, record in enumerate(sorted(records, key=lambda item: str(item.image_path)), start=1):
        relative_ls = record.image_path.relative_to(output_root / "yolo_draft" / "images" / record.split)
        ls_relative = Path("images") / record.source.view / relative_ls.name
        image_url = "/data/local-files/?d=" + quote(ls_relative.as_posix())
        metadata = {
            "sample_key": record.image_path.stem,
            "sample_id": record.source.sample_id,
            "view": record.source.view,
            "hand": record.source.hand,
            "session_id": record.source.session_id,
            "group_id": record.source.group_id,
            "split": record.split,
            "match_method": record.match_method,
            "is_conflict": record.is_conflict,
            "needs_human_annotation": record.needs_human_annotation,
        }
        preannotation = None if record.is_conflict or not record.is_reused else record.label_text
        task = build_label_studio_task(
            task_id=task_id,
            image_url=image_url,
            label_text=preannotation,
            fingerprint=record.crop_fingerprint,
            metadata=metadata,
            model_version=record.match_method,
        )
        tasks_all.append(task)
        if record.needs_human_annotation:
            tasks_review.append(task)
        annotation_rows.append(
            {
                **metadata,
                "image_path": str(record.image_path),
                "label_path": str(record.label_path),
                "box_count": str(len(record.label_text.splitlines())),
                "source_label_path": record.source_label_path,
                "crop_pixel_sha256": record.crop_fingerprint.pixel_sha256,
            },
        )
    (output_root / "label_studio" / "tasks_all.json").write_text(
        json.dumps(tasks_all, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "label_studio" / "tasks_review.json").write_text(
        json.dumps(tasks_review, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_label_studio_config(output_root / "label_studio" / "label_studio_config.xml")
    _write_csv(output_root / "annotation_manifest.csv", tuple(annotation_rows[0]), annotation_rows)
    (output_root / "yolo_draft" / "README.md").write_text(
        "# Incomplete ZS32 YOLO draft\n\n"
        "This directory intentionally has no `data.yaml`. Missing and conflicting annotations are placeholder empty "
        "txt files until Label Studio review is complete. Run Stage 36 `finalize` to create a trainable dataset.\n",
        encoding="utf-8",
    )
    (output_root / "roi_config.json").write_text(json.dumps(roi_payload, indent=2) + "\n", encoding="utf-8")

    view_stats: dict[str, dict[str, int]] = {}
    split_view_positive: dict[str, dict[str, int]] = {split: {} for split in ("train", "val", "test")}
    for view in CANONICAL_VIEWS:
        view_records = [record for record in records if record.source.view == view]
        view_stats[view] = {
            "normal": sum(record.source.label == "normal" for record in view_records),
            "defect": sum(record.source.label == "defect" for record in view_records),
            "annotated_positive": sum(bool(record.label_text.strip()) for record in view_records),
            "needs_human_annotation": sum(record.needs_human_annotation for record in view_records),
        }
        for split in split_view_positive:
            split_view_positive[split][view] = sum(
                record.split == split and bool(record.label_text.strip()) for record in view_records
            )
    summary: dict[str, Any] = {
        "source_images": len(sources),
        "canonical_images": len(records),
        "physical_groups_before_deduplication": len(strata_by_group),
        "duplicate_images": len(sources) - len(records),
        "duplicate_group_aliases": len(alias_pairs),
        "reused_labels": sum(record.is_reused for record in records),
        "reused_positive_labels": sum(record.is_reused and bool(record.label_text.strip()) for record in records),
        "horizontal_flip_reused_labels": sum(
            record.match_method == "horizontal_flip_pixel_sha256" for record in records
        ),
        "confirmed_normal_empty_labels": sum(
            record.source.label == "normal" and not record.needs_human_annotation for record in records
        ),
        "resolved_without_human": sum(not record.needs_human_annotation for record in records),
        "new_or_missing_to_label": sum(record.needs_human_annotation and not record.is_conflict for record in records),
        "label_conflicts": sum(record.is_conflict for record in records),
        "perceptual_candidates": len(approximate_rows),
        "review_tasks": len(tasks_review),
        "views": view_stats,
        "positive_by_split_and_view": split_view_positive,
        "dataset_trainable": not tasks_review,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if tasks_review:
        (output_root / "yolo_draft" / "DO_NOT_TRAIN_UNTIL_LABEL_STUDIO_REVIEW_COMPLETE.txt").write_text(
            f"{len(tasks_review)} images still require completed Label Studio review.\n",
            encoding="utf-8",
        )
    _validate_materialized_yolo(output_root / "yolo_draft", records)
    return summary


def export_defect_label_studio_tasks(
    *,
    prepared_root: Path,
    dataset_root: Path,
    roi_config: Path,
    labeling_root: Path,
    legacy_repo_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Export all 840 raw defect images as independent Label Studio tasks."""
    prepared_root = prepared_root.resolve()
    dataset_root = dataset_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing defect task output: {output_root}")
    image_width, image_height, rois, _payload = load_roi_config(
        roi_config,
        expected_views=CANONICAL_VIEWS,
    )
    sources = [source for source in discover_source_images(dataset_root) if source.label == "defect"]
    if len(sources) != 840:
        raise ValueError(f"Expected exactly 840 raw defect images, found {len(sources)}")
    prepared_rows = {
        Path(row["original_image_path"]).resolve(): row for row in _read_csv(prepared_root / "mapping.csv")
    }
    if any(source.path.resolve() not in prepared_rows for source in sources):
        raise ValueError("Prepared mapping does not cover every raw defect image")
    candidates_by_source = _load_label_studio_candidates(
        labeling_root=labeling_root,
        legacy_repo_root=legacy_repo_root,
        rois=rois,
        image_width=image_width,
        image_height=image_height,
    )
    sources_by_exact: dict[tuple[str, str], list[SourceImage]] = defaultdict(list)
    for source in sources:
        row = prepared_rows[source.path.resolve()]
        sources_by_exact[source.view, row["pixel_sha256"]].append(source)

    tasks_all: list[dict[str, Any]] = []
    tasks_review: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    method_counts: dict[str, int] = defaultdict(int)
    view_counts: dict[str, dict[str, int]] = {
        view: {"total": 0, "safe_preannotated": 0, "conflict": 0, "missing": 0} for view in CANONICAL_VIEWS
    }
    conflict_clusters: set[str] = set()
    for task_id, source in enumerate(sorted(sources, key=lambda item: str(item.path)), start=1):
        prepared = prepared_rows[source.path.resolve()]
        exact_sources = sources_by_exact[source.view, prepared["pixel_sha256"]]
        own_candidates = tuple(candidates_by_source.get(source.path.resolve(), ()))
        duplicate_candidates = tuple(
            candidate
            for exact_source in exact_sources
            for candidate in candidates_by_source.get(exact_source.path.resolve(), ())
        )
        selected, method, conflict, needs_human = _select_defect_preannotation(
            own_candidates,
            duplicate_candidates,
        )
        cluster_id = ""
        if conflict:
            cluster_id = _sha256_bytes("\n".join(sorted(str(item.path) for item in exact_sources)).encode())[:16]
            conflict_clusters.add(cluster_id)
        name = _safe_name(source)
        image_path = output_root / "images" / source.view / name
        _safe_link(Path(prepared["new_image_path"]), image_path)
        roi = RoiBox(*rois[source.view])
        fingerprint = ImageFingerprint(
            prepared["file_sha256"],
            prepared["pixel_sha256"],
            "",
            prepared["perceptual_hash"],
            roi.width,
            roi.height,
        )
        metadata = {
            "sample_key": image_path.stem,
            "sample_id": source.sample_id,
            "view": source.view,
            "hand": source.hand,
            "session_id": source.session_id,
            "group_id": source.group_id,
            "split": prepared["split"],
            "match_method": method,
            "is_duplicate": len(exact_sources) > 1,
            "is_conflict": conflict,
            "conflict_cluster_id": cluster_id,
            "needs_human_annotation": needs_human,
        }
        task = build_label_studio_task(
            task_id=task_id,
            image_url="/data/local-files/?d=" + quote((Path("images") / source.view / name).as_posix()),
            label_text=selected.text if selected is not None else None,
            fingerprint=fingerprint,
            metadata=metadata,
            model_version=method,
        )
        tasks_all.append(task)
        if needs_human:
            tasks_review.append(task)
        method_counts[method] += 1
        view_counts[source.view]["total"] += 1
        if conflict:
            view_counts[source.view]["conflict"] += 1
        elif selected is None:
            view_counts[source.view]["missing"] += 1
        else:
            view_counts[source.view]["safe_preannotated"] += 1
        manifest_rows.append(
            {
                "original_image_path": str(source.path),
                "task_image_path": str(image_path),
                "view": source.view,
                "session_id": source.session_id,
                "group_id": source.group_id,
                "sample_id": source.sample_id,
                "split": prepared["split"],
                "match_method": method,
                "source_label_path": selected.source_label_path if selected is not None else "",
                "has_preannotation": str(selected is not None).lower(),
                "is_duplicate": str(len(exact_sources) > 1).lower(),
                "is_conflict": str(conflict).lower(),
                "conflict_cluster_id": cluster_id,
                "needs_human_annotation": str(needs_human).lower(),
                "pixel_sha256": prepared["pixel_sha256"],
            },
        )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "tasks_defect_840.json").write_text(
        json.dumps(tasks_all, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "tasks_defect_review.json").write_text(
        json.dumps(tasks_review, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_label_studio_config(output_root / "label_studio_config.xml")
    _write_csv(output_root / "defect_mapping.csv", tuple(manifest_rows[0]), manifest_rows)
    summary: dict[str, Any] = {
        "defect_tasks": len(tasks_all),
        "safe_preannotated": sum(not task["data"]["needs_human_annotation"] for task in tasks_all),
        "source_preannotations": method_counts["source_label_studio_preannotation"],
        "duplicate_label_reuse": method_counts["sha256_duplicate_label_reuse"],
        "label_conflict_clusters": len(conflict_clusters),
        "label_conflict_tasks": method_counts["source_label_studio_preannotation_conflict"],
        "missing_tasks": method_counts["missing_annotation"],
        "review_tasks": len(tasks_review),
        "views": view_counts,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _label_studio_results_to_yolo(task: Mapping[str, Any]) -> str:
    annotations = task.get("annotations")
    if not isinstance(annotations, list) or not annotations:
        raise ValueError(f"Label Studio task has no completed annotation: {task.get('id')!r}")
    completed = [
        annotation
        for annotation in annotations
        if isinstance(annotation, Mapping) and annotation.get("was_cancelled") is not True
    ]
    if not completed:
        raise ValueError(f"Label Studio task annotation is cancelled: {task.get('id')!r}")
    annotation = completed[-1]
    results = annotation.get("result")
    if not isinstance(results, list):
        raise ValueError(f"Label Studio task result is malformed: {task.get('id')!r}")
    lines = []
    for result in results:
        if not isinstance(result, Mapping) or result.get("type") != "rectanglelabels":
            continue
        value = result.get("value")
        if not isinstance(value, Mapping):
            raise ValueError("Label Studio rectangle value is malformed")
        x, y, width, height = (float(value[key]) / 100 for key in ("x", "y", "width", "height"))
        line = f"0 {x + width / 2:.8f} {y + height / 2:.8f} {width:.8f} {height:.8f}"
        parse_yolo_labels(line, source="Label Studio JSON export")
        lines.append(line)
    return "\n".join(lines) + ("\n" if lines else "")


def finalize_reconciled_dataset(
    *,
    prepared_root: Path,
    label_studio_json: Path,
    output_root: Path,
) -> dict[str, int]:
    """Merge completed Label Studio JSON review into a new trainable YOLO dataset."""
    prepared_root, output_root = prepared_root.resolve(), output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing final output: {output_root}")
    rows = _read_csv(prepared_root / "annotation_manifest.csv")
    exported = json.loads(label_studio_json.read_text(encoding="utf-8"))
    if not isinstance(exported, list):
        raise ValueError("Label Studio JSON export must be a task list")
    reviewed: dict[str, str] = {}
    for task in exported:
        if not isinstance(task, Mapping):
            raise ValueError("Label Studio JSON export contains a non-object task")
        data = task.get("data")
        annotations = task.get("annotations")
        has_completed_annotation = isinstance(annotations, list) and any(
            isinstance(annotation, Mapping) and annotation.get("was_cancelled") is not True
            for annotation in annotations
        )
        if isinstance(data, Mapping) and isinstance(data.get("sample_key"), str) and has_completed_annotation:
            reviewed[data["sample_key"]] = _label_studio_results_to_yolo(task)
    required = {row["sample_key"] for row in rows if row["needs_human_annotation"].lower() == "true"}
    missing = sorted(required - set(reviewed))
    if missing:
        raise ValueError(f"Label Studio export is missing completed review tasks: {missing[:10]}")
    output_root.mkdir(parents=True)
    box_count = 0
    for row in rows:
        split, sample_key = row["split"], row["sample_key"]
        source_image = Path(row["image_path"])
        source_label = Path(row["label_path"])
        output_image = output_root / "images" / split / source_image.name
        output_label = output_root / "labels" / split / source_label.name
        _safe_link(source_image, output_image)
        output_label.parent.mkdir(parents=True, exist_ok=True)
        label_text = reviewed.get(sample_key, source_label.read_text(encoding="utf-8"))
        parse_yolo_labels(label_text, source=sample_key)
        output_label.write_text(label_text, encoding="utf-8")
        box_count += len(label_text.splitlines())
    (output_root / "data.yaml").write_text(
        "train: images/train\nval: images/val\ntest: images/test\n\nnames:\n  0: defect\n",
        encoding="utf-8",
    )
    mapping_rows = _read_csv(prepared_root / "mapping.csv")
    final_mapping = []
    for row in mapping_rows:
        split = row["split"]
        image_name = Path(row["new_image_path"]).name
        label_name = f"{Path(image_name).stem}.txt"
        was_reviewed = row["needs_human_annotation"].lower() == "true"
        updated = dict(row)
        updated.update(
            {
                "new_image_path": str((output_root / "images" / split / image_name).resolve()),
                "new_label_path": str((output_root / "labels" / split / label_name).resolve()),
                "match_method": "label_studio_review" if was_reviewed else row["match_method"],
                "is_conflict": "false",
                "needs_human_annotation": "false",
            },
        )
        final_mapping.append(updated)
    _write_csv(output_root / "mapping.csv", MAPPING_COLUMNS, final_mapping)
    shutil.copy2(prepared_root / "roi_config.json", output_root / "roi_config.json")
    return {"images": len(rows), "labels": len(rows), "boxes": box_count, "reviewed": len(required)}


def finalize_defect_yolo_export(
    *,
    prepared_root: Path,
    yolo_export_root: Path,
    output_root: Path,
    confirm_skipped_empty: bool,
    session_split: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a session-isolated YOLO dataset from the defect-840 Label Studio export.

    Missing YOLO files are interpreted as confirmed empty annotations only when
    ``confirm_skipped_empty`` records an explicit operator decision.
    """
    prepared_root = prepared_root.resolve()
    yolo_export_root = yolo_export_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing final output: {output_root}")
    if not confirm_skipped_empty:
        raise ValueError("Missing exported labels require explicit confirmation that skipped tasks are empty")
    if (yolo_export_root / "classes.txt").read_text(encoding="utf-8").splitlines() != ["defect"]:
        raise ValueError("Defect-840 YOLO export must contain exactly one class named 'defect'")

    rows = _read_csv(prepared_root / "defect_mapping.csv")
    if not rows:
        raise ValueError("Defect-840 mapping is empty")
    split_by_session = dict(session_split or DEFECT_840_SESSION_SPLIT)
    observed_sessions = {row["session_id"] for row in rows}
    if set(split_by_session) != observed_sessions:
        raise ValueError(
            "Session split does not exactly cover defect mapping; "
            f"missing={sorted(observed_sessions - set(split_by_session))}, "
            f"extra={sorted(set(split_by_session) - observed_sessions)}",
        )
    if set(split_by_session.values()) != {"train", "val", "test"}:
        raise ValueError("Session split must use train, val, and test")

    labels_by_name = {path.name: path for path in (yolo_export_root / "labels").glob("*.txt")}
    expected_names = {f"{Path(row['task_image_path']).stem}.txt" for row in rows}
    orphan_labels = sorted(set(labels_by_name) - expected_names)
    if orphan_labels:
        raise ValueError(f"YOLO export contains labels outside defect mapping: {orphan_labels[:10]}")

    pixel_splits: dict[str, str] = {}
    final_rows: list[dict[str, str]] = []
    split_stats: dict[str, dict[str, Any]] = {
        split: {"images": 0, "positive_images": 0, "empty_images": 0, "boxes": 0, "positive_by_view": {}}
        for split in ("train", "val", "test")
    }
    confirmed_empty = 0
    exported_empty = 0
    output_root.mkdir(parents=True)
    for row in rows:
        split = split_by_session[row["session_id"]]
        previous_split = pixel_splits.setdefault(row["pixel_sha256"], split)
        if previous_split != split:
            raise ValueError(
                f"Exact duplicate pixel hash leaks across splits: {row['pixel_sha256']}",
            )
        source_image = Path(row["task_image_path"])
        label_name = f"{source_image.stem}.txt"
        exported_label = labels_by_name.get(label_name)
        if exported_label is None:
            label_text = ""
            annotation_status = "operator_confirmed_skipped_empty"
            confirmed_empty += 1
        else:
            label_text = exported_label.read_text(encoding="utf-8")
            annotation_status = "label_studio_yolo_export" if label_text.strip() else "label_studio_confirmed_empty"
            exported_empty += int(not label_text.strip())
        parsed = parse_yolo_labels(label_text, source=str(exported_label or label_name))
        image_output = output_root / "images" / split / source_image.name
        label_output = output_root / "labels" / split / label_name
        _safe_link(source_image, image_output)
        label_output.parent.mkdir(parents=True, exist_ok=True)
        label_output.write_text(label_text, encoding="utf-8")
        stats = split_stats[split]
        stats["images"] += 1
        stats["boxes"] += len(parsed)
        if label_text.strip():
            stats["positive_images"] += 1
            by_view = stats["positive_by_view"]
            by_view[row["view"]] = by_view.get(row["view"], 0) + 1
        else:
            stats["empty_images"] += 1
        updated = dict(row)
        updated.update(
            {
                "split": split,
                "new_image_path": str(image_output),
                "new_label_path": str(label_output),
                "annotation_status": annotation_status,
            },
        )
        final_rows.append(updated)

    (output_root / "data.yaml").write_text(
        "train: images/train\nval: images/val\ntest: images/test\n\nnames:\n  0: defect\n",
        encoding="utf-8",
    )
    _write_csv(output_root / "mapping.csv", tuple(final_rows[0]), final_rows)
    summary: dict[str, Any] = {
        "images": len(rows),
        "labels": len(rows),
        "positive_images": sum(stats["positive_images"] for stats in split_stats.values()),
        "empty_images": sum(stats["empty_images"] for stats in split_stats.values()),
        "boxes": sum(stats["boxes"] for stats in split_stats.values()),
        "confirmed_empty_images": confirmed_empty,
        "exported_empty_images": exported_empty,
        "split_by_session": split_by_session,
        "splits": split_stats,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "SKIPPED_AS_EMPTY_CONFIRMATION.md").write_text(
        "# Skipped-task empty-label confirmation\n\n"
        "The operator explicitly confirmed on 2026-07-15 that all Label Studio tasks omitted from the YOLO export "
        "were manually inspected and contained no visible defect. They are represented by empty YOLO label files.\n",
        encoding="utf-8",
    )
    return summary


def build_balanced_defect_normal_yolo_dataset(
    *,
    defect_prepared_root: Path,
    yolo_export_root: Path,
    normal_manifest: Path,
    output_root: Path,
    normal_session_id: str | None = None,
    max_normal_groups: int | None = None,
    seed: int = 42,
    group_split: Mapping[str, str] | None = None,
    require_full_coverage: bool = True,
) -> dict[str, Any]:
    """Build a clean group-stratified dataset from reviewed positives and trusted normal images."""
    defect_prepared_root = defect_prepared_root.resolve()
    yolo_export_root = yolo_export_root.resolve()
    normal_manifest = normal_manifest.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing balanced output: {output_root}")
    if (yolo_export_root / "classes.txt").read_text(encoding="utf-8").splitlines() != ["defect"]:
        raise ValueError("Balanced YOLO export must contain exactly one class named 'defect'")

    defect_rows = _read_csv(defect_prepared_root / "defect_mapping.csv")
    labels_by_name = {path.name: path for path in (yolo_export_root / "labels").glob("*.txt")}
    expected_names = {f"{Path(row['task_image_path']).stem}.txt" for row in defect_rows}
    orphan_labels = sorted(set(labels_by_name) - expected_names)
    if orphan_labels:
        raise ValueError(f"YOLO export contains labels outside defect mapping: {orphan_labels[:10]}")

    positive_candidates: dict[str, list[dict[str, str]]] = defaultdict(list)
    uncertain_rows: list[dict[str, str]] = []
    for row in defect_rows:
        label_name = f"{Path(row['task_image_path']).stem}.txt"
        label_path = labels_by_name.get(label_name)
        label_text = label_path.read_text(encoding="utf-8") if label_path is not None else ""
        if not label_text.strip():
            uncertain_rows.append(dict(row))
            continue
        parse_yolo_labels(label_text, source=str(label_path))
        candidate = dict(row)
        candidate.update(
            {
                "image_path": row["task_image_path"],
                "label_text": label_text,
                "annotation_status": "label_studio_positive",
                "source_kind": "defect_positive",
                "hand": row["sample_id"].split("/")[0],
                "defect_type": row["sample_id"].split("/")[2],
            },
        )
        positive_candidates[row["pixel_sha256"]].append(candidate)

    selected_defects: list[dict[str, str]] = []
    duplicate_alias_rows: list[dict[str, str]] = []
    for pixel_sha256, candidates in sorted(positive_candidates.items()):
        ordered = sorted(candidates, key=lambda row: row["task_image_path"])
        selected = ordered[0]
        selected_defects.append(selected)
        duplicate_alias_rows.extend(
            {
                    "pixel_sha256": pixel_sha256,
                    "selected_image_path": selected["task_image_path"],
                    "excluded_alias_image_path": alias["task_image_path"],
                    "labels_equivalent": str(
                        labels_equivalent(selected["label_text"], alias["label_text"]),
                    ).lower(),
            }
            for alias in ordered[1:]
        )

    normal_rows = []
    for row in _read_csv(normal_manifest):
        normalized_row = row
        if "sample_id" in row:
            if row.get("match_method") != "normal_directory_confirmed_empty":
                continue
            parts = row["sample_id"].split("/")
            if len(parts) != 5 or parts[1:3] != ["normal", "none"]:
                raise ValueError(f"Malformed normal sample_id: {row['sample_id']}")
            hand, session_id, group_id = parts[0], parts[3], parts[4]
            view = row["view"]
            image_path = row.get("new_image_path") or row.get("image_path")
            pixel_sha256 = row.get("pixel_sha256") or row.get("crop_pixel_sha256")
            source_label = row.get("new_label_path") or row.get("label_path")
        else:
            if row.get("label") != "normal" or row.get("hand") != "right":
                continue
            hand = row["hand"]
            session_id = row["session_id"]
            view = row.get("resolved_view") or row["source_view"]
            match = GROUP_PATTERN.search(Path(row["output_path"]).name)
            if match is None:
                raise ValueError(f"Could not parse normal group from: {row['output_path']}")
            group_id = match.group(1)
            parts = [hand, "normal", "none", session_id, group_id]
            image_path = row["output_path"]
            if not Path(image_path).is_absolute():
                candidates = [parent / image_path for parent in normal_manifest.parents]
                image_path = str(next((path for path in candidates if path.is_file()), candidates[0]))
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not decode normal ROI image: {image_path}")
            pixel_sha256 = _pixel_sha256(image)
            source_label = ""
            normalized_row = {
                **row,
                "sample_id": "/".join(parts),
                "view": view,
                "match_method": "patchcore_roi_normal_directory",
            }
        if normal_session_id is not None and session_id != normal_session_id:
            continue
        if not image_path or not pixel_sha256:
            raise ValueError("Normal mapping must provide a prepared image path and pixel SHA256")
        if source_label and Path(source_label).read_text(encoding="utf-8").strip():
            raise ValueError(f"Normal sample has a non-empty label: {source_label}")
        safe_name = f"{hand}__{view}__normal__normal__{session_id}__{group_id}.png"
        normal_rows.append(
            {
                **normalized_row,
                "image_path": image_path,
                "output_name": safe_name,
                "label_text": "",
                "annotation_status": "normal_directory_confirmed_empty",
                "source_kind": "trusted_normal",
                "hand": hand,
                "defect_type": "none",
                "session_id": session_id,
                "group_id": group_id,
                "pixel_sha256": pixel_sha256,
            },
        )
    if not normal_rows:
        raise ValueError("No trusted normal rows matched the requested normal session")
    if max_normal_groups is not None:
        if max_normal_groups <= 0:
            raise ValueError("max_normal_groups must be positive")
        normal_group_ids = sorted(
            {row["sample_id"] for row in normal_rows},
            key=lambda sample_id: _sha256_bytes(f"{seed}:{sample_id}".encode()),
        )[:max_normal_groups]
        selected_normal_groups = set(normal_group_ids)
        normal_rows = [row for row in normal_rows if row["sample_id"] in selected_normal_groups]
    if len({row["pixel_sha256"] for row in normal_rows}) != len(normal_rows):
        raise ValueError("Selected normal subset contains exact duplicate images")

    entries = selected_defects + normal_rows
    strata_by_group: dict[str, str] = {}
    for row in entries:
        group_key = row["sample_id"]
        stratum = "normal" if row["source_kind"] == "trusted_normal" else f"defect:{row['defect_type']}"
        previous = strata_by_group.setdefault(group_key, stratum)
        if previous != stratum:
            raise ValueError(f"Physical group crosses semantic strata: {group_key}")

    pixel_groups: dict[str, set[str]] = defaultdict(set)
    for row in entries:
        pixel_groups[row["pixel_sha256"]].add(row["sample_id"])
    alias_pairs = tuple(
        (group_ids[0], group_id)
        for groups in pixel_groups.values()
        if len(groups) > 1
        for group_ids in [sorted(groups)]
        for group_id in group_ids[1:]
    )

    def coverage_ok(assignments: Mapping[str, str]) -> bool:
        for split in ("train", "val", "test"):
            positives = [
                row for row in selected_defects if assignments[row["sample_id"]] == split
            ]
            normals = [row for row in normal_rows if assignments[row["sample_id"]] == split]
            if {row["hand"] for row in positives} != {"left", "right"}:
                return False
            if {row["defect_type"] for row in positives} != {"deform", "less", "others"}:
                return False
            if {row["view"] for row in positives} != set(CANONICAL_VIEWS):
                return False
            if {row["view"] for row in normals} != set(CANONICAL_VIEWS):
                return False
        return True

    if group_split is None:
        assignments = None
        effective_seed = seed
        for candidate_seed in range(seed, seed + 10_000):
            candidate = split_alias_groups(
                strata_by_group,
                alias_pairs=alias_pairs,
                val_ratio=0.15,
                test_ratio=0.15,
                seed=candidate_seed,
            )
            if not require_full_coverage or coverage_ok(candidate):
                assignments = candidate
                effective_seed = candidate_seed
                break
        if assignments is None:
            raise ValueError("Could not find a leakage-safe balanced group split")
    else:
        assignments = dict(group_split)
        effective_seed = seed
        if set(assignments) != set(strata_by_group):
            raise ValueError("Provided group split does not exactly cover the selected physical groups")
        if require_full_coverage and not coverage_ok(assignments):
            raise ValueError("Provided group split does not satisfy full hand/type/view coverage")

    output_root.mkdir(parents=True)
    split_stats: dict[str, dict[str, Any]] = {
        split: {
            "images": 0,
            "positive_images": 0,
            "trusted_normal_images": 0,
            "boxes": 0,
            "positive_by_view": dict.fromkeys(CANONICAL_VIEWS, 0),
        }
        for split in ("train", "val", "test")
    }
    final_rows: list[dict[str, str]] = []
    pixel_splits: dict[str, str] = {}
    for row in entries:
        split = assignments[row["sample_id"]]
        previous_split = pixel_splits.setdefault(row["pixel_sha256"], split)
        if previous_split != split:
            raise ValueError(f"Exact pixel duplicate leaks across splits: {row['pixel_sha256']}")
        source_image = Path(row["image_path"])
        output_name = row.get("output_name") or source_image.name
        output_image = output_root / "images" / split / output_name
        output_label = output_root / "labels" / split / f"{Path(output_name).stem}.txt"
        _safe_link(source_image, output_image)
        output_label.parent.mkdir(parents=True, exist_ok=True)
        output_label.write_text(row["label_text"], encoding="utf-8")
        boxes = parse_yolo_labels(row["label_text"], source=str(output_label))
        stats = split_stats[split]
        stats["images"] += 1
        stats["boxes"] += len(boxes)
        if row["source_kind"] == "trusted_normal":
            stats["trusted_normal_images"] += 1
        else:
            stats["positive_images"] += 1
            stats["positive_by_view"][row["view"]] += 1
        final_rows.append(
            {
                "source_image_path": str(source_image),
                "new_image_path": str(output_image),
                "new_label_path": str(output_label),
                "sample_id": row["sample_id"],
                "view": row["view"],
                "hand": row["hand"],
                "defect_type": row["defect_type"],
                "session_id": row["session_id"],
                "group_id": row["group_id"],
                "split": split,
                "source_kind": row["source_kind"],
                "annotation_status": row["annotation_status"],
                "pixel_sha256": row["pixel_sha256"],
            },
        )

    (output_root / "data.yaml").write_text(
        "train: images/train\nval: images/val\ntest: images/test\n\nnames:\n  0: defect\n",
        encoding="utf-8",
    )
    _write_csv(output_root / "mapping.csv", tuple(final_rows[0]), final_rows)
    if uncertain_rows:
        _write_csv(output_root / "excluded_uncertain_defects.csv", tuple(uncertain_rows[0]), uncertain_rows)
    if duplicate_alias_rows:
        _write_csv(
            output_root / "deduplicated_positive_aliases.csv",
            tuple(duplicate_alias_rows[0]),
            duplicate_alias_rows,
        )
    summary: dict[str, Any] = {
        "images": len(entries),
        "positive_images": len(selected_defects),
        "trusted_normal_images": len(normal_rows),
        "boxes": sum(stats["boxes"] for stats in split_stats.values()),
        "excluded_uncertain_defect_images": len(uncertain_rows),
        "deduplicated_positive_aliases": len(duplicate_alias_rows),
        "split_unit": "physical_group_with_exact_pixel_alias_binding",
        "session_isolation": False,
        "requested_seed": seed,
        "effective_seed": effective_seed,
        "splits": split_stats,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
