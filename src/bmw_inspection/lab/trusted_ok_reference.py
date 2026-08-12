"""Prepare immutable, human-reviewed BMW trusted-OK reference candidates."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER, _atomic_publish_noreplace


DATASET_FIELDS = (
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
DECISION_FIELDS = ("physical_part_id", "sample_id", "decision", "reviewer", "review_note")
TRUSTED_OK_SESSION_ID = "20260810_210030_527506"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_PART_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_VIEW_LABELS = {
    "front": "前正面",
    "front_left": "前左侧",
    "front_right": "前右侧",
    "front_secondary": "前辅助",
    "back": "后正面",
    "back_left": "后左侧",
    "back_right": "后右侧",
    "back_secondary": "后辅助",
}
_CJK_FONTS = (
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
)


@dataclass(frozen=True, slots=True)
class CandidateImage:
    """One source image eligible for a human trusted-OK decision."""

    sample_id: str
    physical_part_id: str
    session_id: str
    group_id: str
    view_id: str
    camera_serial: str
    source_path: Path
    source_sha256: str
    source_class: str
    business_label: str
    split: str

    def manifest_row(self) -> dict[str, str]:
        """Return this image in the frozen prepared-manifest schema."""
        return {
            "sample_id": self.sample_id,
            "physical_part_id": self.physical_part_id,
            "session_id": self.session_id,
            "group_id": self.group_id,
            "view_id": self.view_id,
            "camera_serial": self.camera_serial,
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "source_class": self.source_class,
            "business_label": self.business_label,
            "split": self.split,
        }


@dataclass(frozen=True, slots=True)
class ReviewPackageSummary:
    """Counts and identity of one immutable manual-review package."""

    manifest_path: Path
    output_dir: Path
    session_id: str
    candidate_part_count: int
    candidate_image_count: int
    pending_decision_count: int
    contact_sheet_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _required(row: Mapping[str, str], field: str, *, context: str) -> str:
    value = row.get(field, "").strip()
    if not value:
        raise ValueError(f"{context} has empty {field}")
    return value


def _read_manifest(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != DATASET_FIELDS:
            raise ValueError(f"dataset manifest header differs from the frozen BMW schema: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"dataset manifest is empty: {path}")
    if any(None in row for row in rows):
        raise ValueError(f"dataset manifest contains malformed or extra cells: {path}")
    return tuple(rows)


def _candidate_rows(rows: Iterable[Mapping[str, str]], session_id: str) -> tuple[CandidateImage, ...]:
    grouped: dict[str, list[CandidateImage]] = defaultdict(list)
    for row_number, raw in enumerate(rows, start=2):
        context = f"dataset manifest row {row_number}"
        if raw.get("session_id", "").strip() != session_id:
            continue
        if (
            raw.get("source_class", "").strip() != "normal"
            or raw.get("business_label", "").strip() != "OK"
            or raw.get("split", "").strip() != "train"
        ):
            continue
        source_sha256 = _required(raw, "source_sha256", context=context).lower()
        if not _SHA256.fullmatch(source_sha256):
            raise ValueError(f"{context} has invalid source_sha256")
        physical_part_id = _required(raw, "physical_part_id", context=context)
        if not _SAFE_PART_ID.fullmatch(physical_part_id):
            raise ValueError(f"{context} has unsafe physical_part_id")
        grouped[physical_part_id].append(
            CandidateImage(
                sample_id=_required(raw, "sample_id", context=context),
                physical_part_id=physical_part_id,
                session_id=_required(raw, "session_id", context=context),
                group_id=_required(raw, "group_id", context=context),
                view_id=_required(raw, "view_id", context=context),
                camera_serial=_required(raw, "camera_serial", context=context),
                source_path=Path(_required(raw, "source_path", context=context)).expanduser().absolute(),
                source_sha256=source_sha256,
                source_class=_required(raw, "source_class", context=context),
                business_label=_required(raw, "business_label", context=context),
                split=_required(raw, "split", context=context),
            )
        )

    candidates: list[CandidateImage] = []
    view_index = {view: index for index, view in enumerate(VIEW_ORDER)}
    for part_id, part_rows in sorted(grouped.items()):
        identities = {(row.sample_id, row.session_id, row.group_id) for row in part_rows}
        if len(identities) != 1:
            raise ValueError(f"candidate physical part has inconsistent identity: {part_id}")
        view_ids = [row.view_id for row in part_rows]
        if any(view_id not in view_index for view_id in view_ids):
            raise ValueError(f"candidate physical part has unsupported view: {part_id}")
        if len(view_ids) != len(set(view_ids)):
            raise ValueError(f"candidate physical part has duplicate views: {part_id}")
        if set(view_ids) != set(VIEW_ORDER):
            continue
        candidates.extend(sorted(part_rows, key=lambda row: view_index[row.view_id]))
    if not candidates:
        raise ValueError("no complete training-normal eight-view candidates found for session")
    return tuple(candidates)


def _verify_sources(rows: Iterable[CandidateImage]) -> None:
    for row in rows:
        if not row.source_path.is_file() or row.source_path.is_symlink():
            raise ValueError(f"candidate source image is not a regular file: {row.source_path}")
        actual = _sha256(row.source_path)
        if actual != row.source_sha256:
            raise ValueError(f"candidate source_sha256 mismatch: {row.source_path}")


def _write_csv(path: Path, fields: tuple[str, ...], rows: Iterable[Mapping[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _CJK_FONTS:
        if path.is_file():
            return ImageFont.truetype(path, size=22)
    return ImageFont.load_default()


def _contact_sheet(rows: tuple[CandidateImage, ...], destination: Path) -> None:
    cell_width, image_height, label_height = 320, 240, 42
    sheet = Image.new("RGB", (cell_width * 4, (image_height + label_height) * 2), "white")
    draw = ImageDraw.Draw(sheet)
    font = _font()
    for index, row in enumerate(rows):
        column, line = index % 4, index // 4
        origin_x = column * cell_width
        origin_y = line * (image_height + label_height)
        with Image.open(row.source_path) as source:
            rendered = ImageOps.contain(source.convert("RGB"), (cell_width, image_height), Image.Resampling.LANCZOS)
        image_x = origin_x + (cell_width - rendered.width) // 2
        image_y = origin_y + (image_height - rendered.height) // 2
        sheet.paste(rendered, (image_x, image_y))
        label = f"{_VIEW_LABELS[row.view_id]} ({row.view_id})"
        draw.text((origin_x + 8, origin_y + image_height + 8), label, fill="black", font=font)
    sheet.save(destination, format="PNG")


def prepare_review_package(manifest: Path, output_dir: Path, *, session_id: str) -> ReviewPackageSummary:
    """Publish PENDING-only human-review candidates from one BMW capture session."""
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string")
    normalized_session_id = session_id.strip()
    if normalized_session_id != TRUSTED_OK_SESSION_ID:
        raise ValueError(f"trusted-OK review package only supports session_id {TRUSTED_OK_SESSION_ID}")
    manifest_path = Path(manifest).expanduser().resolve()
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"dataset manifest is not a regular file: {manifest_path}")
    output = Path(output_dir).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"review package already exists: {output}")

    candidates = _candidate_rows(_read_manifest(manifest_path), normalized_session_id)
    _verify_sources(candidates)
    part_rows = {
        part_id: tuple(rows)
        for part_id, rows in _group_by_part(candidates).items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        _write_csv(staging / "candidate_manifest.csv", DATASET_FIELDS, (row.manifest_row() for row in candidates))
        decisions = [
            {
                "physical_part_id": part_id,
                "sample_id": rows[0].sample_id,
                "decision": "PENDING",
                "reviewer": "",
                "review_note": "",
            }
            for part_id, rows in part_rows.items()
        ]
        _write_csv(staging / "review_decisions.csv", DECISION_FIELDS, decisions)
        contact_sheets = staging / "review" / "contact_sheets"
        contact_sheets.mkdir(parents=True)
        for part_id, rows in part_rows.items():
            _contact_sheet(rows, contact_sheets / f"{part_id}.png")
        payload = {
            "schema_version": 1,
            "status": "awaiting_human_review",
            "automatic_approvals": 0,
            "session_id": normalized_session_id,
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "candidate_part_count": len(part_rows),
            "candidate_image_count": len(candidates),
            "pending_decision_count": len(decisions),
            "contact_sheet_count": len(part_rows),
            "required_decision": "PENDING",
        }
        (staging / "review_package.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _atomic_publish_noreplace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return ReviewPackageSummary(
        manifest_path=manifest_path,
        output_dir=output,
        session_id=normalized_session_id,
        candidate_part_count=len(part_rows),
        candidate_image_count=len(candidates),
        pending_decision_count=len(part_rows),
        contact_sheet_count=len(part_rows),
    )


def _group_by_part(rows: Iterable[CandidateImage]) -> dict[str, list[CandidateImage]]:
    grouped: dict[str, list[CandidateImage]] = defaultdict(list)
    for row in rows:
        grouped[row.physical_part_id].append(row)
    return {part_id: grouped[part_id] for part_id in sorted(grouped)}
