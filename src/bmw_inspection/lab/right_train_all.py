# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Prepare reviewed YOLO labels for BMW right-hand one-click training."""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from bmw_inspection.lab.eight_view_dataset import _atomic_publish_noreplace
from bmw_inspection.lab.eight_view_training_data import _validate_yolo_text


def _task_label_names(json_path: Path) -> tuple[str, ...]:
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Label Studio JSON is invalid: {json_path}") from error
    if not isinstance(payload, list) or not payload:
        raise ValueError("Label Studio JSON must contain a non-empty task list")
    names: list[str] = []
    for index, task in enumerate(payload, start=1):
        if not isinstance(task, dict) or not isinstance(task.get("data"), dict):
            raise ValueError(f"Label Studio task {index} has no data object")
        name = task["data"].get("expected_label_filename")
        if (
            not isinstance(name, str)
            or not name.endswith(".txt")
            or Path(name).name != name
            or "/" in name
            or "\\" in name
        ):
            raise ValueError(f"Label Studio task {index} has an invalid expected label filename")
        if not isinstance(task.get("annotations"), list) or not task["annotations"]:
            raise ValueError(f"Label Studio task {index} has not been reviewed")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("Label Studio JSON contains duplicate expected label filenames")
    return tuple(sorted(names))


def _zip_label_texts(zip_path: Path, expected_names: tuple[str, ...]) -> dict[str, str]:
    try:
        with zipfile.ZipFile(zip_path) as archive:
            try:
                classes = archive.read("classes.txt").decode("utf-8-sig").splitlines()
            except (KeyError, UnicodeDecodeError) as error:
                raise ValueError("Label Studio ZIP has no valid classes.txt") from error
            if not classes or classes[0].strip() != "defect":
                raise ValueError("Label Studio ZIP class 0 must be defect")
            members: dict[str, str] = {}
            for member in archive.namelist():
                parts = PurePosixPath(member).parts
                if len(parts) == 2 and parts[0] == "labels" and parts[1].endswith(".txt"):
                    if parts[1] in members:
                        raise ValueError(f"Label Studio ZIP contains duplicate label: {parts[1]}")
                    members[parts[1]] = member
            if set(members) != set(expected_names):
                missing = len(set(expected_names) - set(members))
                extra = len(set(members) - set(expected_names))
                raise ValueError(
                    "Label Studio ZIP labels do not match JSON tasks: "
                    f"expected={len(expected_names)}, actual={len(members)}, missing={missing}, extra={extra}"
                )
            labels: dict[str, str] = {}
            for name in expected_names:
                try:
                    text = archive.read(members[name]).decode("utf-8")
                except UnicodeDecodeError as error:
                    raise ValueError(f"YOLO label is not UTF-8: {name}") from error
                labels[name] = _validate_yolo_text(text, path=zip_path / f"labels/{name}")
            return labels
    except zipfile.BadZipFile as error:
        raise ValueError(f"Label Studio ZIP is invalid: {zip_path}") from error


def _cache_matches(destination: Path, labels: dict[str, str]) -> bool:
    if not destination.is_dir() or destination.is_symlink():
        return False
    actual = {path.name for path in destination.glob("*.txt") if path.is_file() and not path.is_symlink()}
    if actual != set(labels):
        return False
    return all((destination / name).read_text(encoding="utf-8") == text for name, text in labels.items())


def prepare_reviewed_yolo_labels(
    zip_path: Path,
    json_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Validate one Label Studio export and publish a stable YOLO label cache.

    Args:
        zip_path (Path): Label Studio YOLO export ZIP.
        json_path (Path): Label Studio project export JSON.
        output_root (Path): Destination directory for normalized YOLO text labels.

    Returns:
        dict[str, Any]: Report containing cache status, label counts, and destination path.

    Raises:
        FileExistsError: If an existing cache differs from the selected export.
        OSError: If input files cannot be read or the cache cannot be published.
        ValueError: If the ZIP, JSON, task mapping, or YOLO labels are invalid.
    """
    archive = Path(zip_path).expanduser().resolve()
    project_json = Path(json_path).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    if not archive.is_file():
        raise ValueError(f"Label Studio ZIP does not exist: {archive}")
    if not project_json.is_file():
        raise ValueError(f"Label Studio JSON does not exist: {project_json}")
    expected_names = _task_label_names(project_json)
    labels = _zip_label_texts(archive, expected_names)
    status = "created"
    if destination.exists() or destination.is_symlink():
        if not _cache_matches(destination, labels):
            raise FileExistsError(f"reviewed YOLO label cache differs from this export: {destination}")
        status = "reused"
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
        try:
            for name, text in labels.items():
                (staging / name).write_text(text, encoding="utf-8")
            _atomic_publish_noreplace(staging, destination)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return {
        "status": status,
        "label_count": len(labels),
        "positive_label_count": sum(bool(text) for text in labels.values()),
        "label_root": str(destination),
    }
