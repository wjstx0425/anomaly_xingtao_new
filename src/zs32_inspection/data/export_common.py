"""Shared immutable materialization for canonical dataset adapter views."""

from __future__ import annotations

import csv
import io
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping, Sequence

from zs32_inspection.runtime.publisher import (
    AtomicDirectoryPublisher,
    PublicationError,
    sha256_file,
)

from .dataset_release import verify_dataset_release
from .manifests import ADAPTER_BASE_COLUMNS, YOLO_ADAPTER_COLUMNS


def read_adapter_rows(release_root: Path, adapter: str) -> list[dict[str, str]]:
    """Read an adapter manifest only after verifying the entire source release."""
    verify_dataset_release(release_root)
    path = release_root / "adapters" / adapter / "manifest.csv"
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        expected = YOLO_ADAPTER_COLUMNS if adapter == "yolo" else ADAPTER_BASE_COLUMNS
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(f"{adapter} adapter manifest header differs from frozen schema")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{adapter} adapter manifest is empty")
    if any(set(row) != set(expected) or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{adapter} adapter manifest contains malformed or extra cells")
    return rows


def csv_bytes(columns: Sequence[str], rows: Sequence[Mapping[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def safe_source(release_root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(item in {"", ".", ".."} for item in pure.parts):
        raise PublicationError(f"unsafe adapter source path: {relative!r}")
    source = release_root.joinpath(*pure.parts)
    if source.is_symlink() or not source.is_file():
        raise PublicationError(f"adapter source must be a regular release file: {source}")
    return source


def materialize_adapter(
    *,
    release_root: Path,
    output_root: Path,
    export_id: str,
    adapter: str,
    destination_for: Callable[[Mapping[str, str]], str],
    additional_files: Mapping[str, bytes] | None = None,
) -> Path:
    """Copy canonical crops into one no-replace adapter-specific filesystem view."""
    rows = read_adapter_rows(release_root, adapter)
    destinations: set[str] = set()
    materialized_rows: list[dict[str, str]] = []
    with AtomicDirectoryPublisher(output_root, export_id) as publisher:
        for row in rows:
            destination = destination_for(row)
            if destination in destinations:
                raise ValueError(f"adapter destination collision: {destination}")
            destinations.add(destination)
            publisher.copy_file(
                safe_source(release_root, row["canonical_crop_path"]),
                destination,
                expected_sha256=row["canonical_crop_sha256"],
            )
            materialized_rows.append({**row, "materialized_path": destination})
        columns = (*rows[0].keys(), "materialized_path")
        publisher.write_bytes("materialized_manifest.csv", csv_bytes(columns, materialized_rows))
        release_manifest_path = safe_source(release_root, "dataset_release.json")
        adapter_manifest_path = safe_source(
            release_root,
            f"adapters/{adapter}/manifest.csv",
        )
        publisher.copy_file(
            release_manifest_path,
            "provenance/dataset_release.json",
            expected_sha256=sha256_file(release_manifest_path),
        )
        publisher.copy_file(
            adapter_manifest_path,
            "provenance/adapter_manifest.csv",
            expected_sha256=sha256_file(adapter_manifest_path),
        )
        for path, payload in (additional_files or {}).items():
            publisher.write_bytes(path, payload)
        required = {
            "materialized_manifest.csv",
            "provenance/dataset_release.json",
            "provenance/adapter_manifest.csv",
            *destinations,
            *(additional_files or {}),
        }
        return publisher.finalize(
            validator=lambda staging: _validate_materialized(staging, materialized_rows),
            required_paths=frozenset(required),
        )


def _validate_materialized(staging: Path, rows: Sequence[Mapping[str, str]]) -> None:
    for row in rows:
        if sha256_file(staging / row["materialized_path"]) != row["canonical_crop_sha256"]:
            raise PublicationError(
                f"materialized adapter crop hash mismatch: {row['materialized_path']}"
            )
