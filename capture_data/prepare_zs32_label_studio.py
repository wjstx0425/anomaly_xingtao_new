# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Prepare ZS32 defect images for Label Studio local-file labeling."""

from __future__ import annotations

import csv
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


HANDS = ("left", "right")
VIEWS = ("front", "front_left", "front_right", "back", "back_left", "back_right")
DEFECT_TYPES = ("deform", "less", "others")
GROUP_PATTERN = re.compile(r"_(group\d{3})_")


@dataclass(frozen=True)
class LabelingImage:
    """Metadata for one source image included in Label Studio staging."""

    source_path: Path
    hand: str
    view: str
    defect_type: str
    session_id: str
    group_id: str

    @property
    def sample_id(self) -> str:
        """The physical six-view group identifier."""
        return f"{self.hand}/{self.session_id}/{self.defect_type}/{self.group_id}"


def discover_labeling_images(dataset_root: Path) -> list[LabelingImage]:
    """Discover ZS32 defect PNGs and require complete six-view groups.

    Args:
        dataset_root (Path): Root containing the fixed ZS32 hand/view dataset layout.

    Returns:
        list[LabelingImage]: Sorted image metadata for complete physical groups.

    Raises:
        RuntimeError: If the dataset contains no defect PNGs.
        ValueError: If a path is unsupported or a physical group is incomplete.
    """
    images: list[LabelingImage] = []
    pattern = "*/" + "*/defect/*/*/images/*.png"
    for source_path in sorted(dataset_root.glob(pattern)):
        relative = source_path.relative_to(dataset_root)
        hand, view, label, defect_type, session_id, images_dir, _ = relative.parts
        if hand not in HANDS or view not in VIEWS or label != "defect":
            msg = f"Unsupported ZS32 labeling path: {source_path}"
            raise ValueError(msg)
        if defect_type not in DEFECT_TYPES or images_dir != "images":
            msg = f"Unsupported ZS32 defect path: {source_path}"
            raise ValueError(msg)
        match = GROUP_PATTERN.search(source_path.name)
        if match is None:
            msg = f"Could not parse group_id from: {source_path.name}"
            raise ValueError(msg)
        images.append(
            LabelingImage(source_path, hand, view, defect_type, session_id, match.group(1)),
        )
    if not images:
        msg = f"No ZS32 defect PNGs found under: {dataset_root}"
        raise RuntimeError(msg)
    grouped: dict[str, set[str]] = {}
    for image in images:
        grouped.setdefault(image.sample_id, set()).add(image.view)
    expected = set(VIEWS)
    incomplete = {sample: sorted(views) for sample, views in grouped.items() if views != expected}
    if incomplete:
        msg = f"Every physical group must contain exactly six views: {incomplete}"
        raise ValueError(msg)
    return images


def _prepare_output_root(
    output_root: Path,
    *,
    overwrite: bool,
    forbidden_roots: Sequence[Path],
) -> None:
    """Create or safely clean the Label Studio output root."""
    if output_root.exists() and not output_root.is_dir():
        msg = f"Output root exists but is not a directory: {output_root}"
        raise ValueError(msg)
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            msg = f"Output root is non-empty: {output_root}. Pass --overwrite to replace it."
            raise ValueError(msg)
        resolved = output_root.resolve(strict=False)
        forbidden = {
            Path("/").resolve(),
            Path.home().resolve(),
            Path.cwd().resolve(),
            *(path.resolve(strict=False) for path in forbidden_roots),
        }
        if any(resolved == path or path.is_relative_to(resolved) for path in forbidden):
            msg = f"Refusing to overwrite unsafe output root: {output_root}"
            raise ValueError(msg)
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    """Atomically write the Label Studio source-to-staging manifest."""
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def _write_label_config(path: Path) -> None:
    """Write a Label Studio rectangle-labeling configuration."""
    path.write_text(
        """<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="defect" background="#E53935"/>
  </RectangleLabels>
</View>
""",
        encoding="utf-8",
    )


def _write_readme(path: Path, output_root: Path) -> None:
    """Write local-file serving setup instructions for Label Studio."""
    resolved_output = output_root.resolve(strict=False)
    path.write_text(
        f"""# ZS32 Label Studio staging data

The images in this directory are hard links to the source dataset. Do not edit or delete the source files.

Configure Label Studio local-file serving before starting the service:

```bash
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT={resolved_output}
label-studio start
```

Use `{resolved_output / "images"}` as the local storage path and `label_studio_config.xml` as the labeling interface.
`labeling_manifest.csv` records the staged path, immutable source path, and six-view group metadata.
""",
        encoding="utf-8",
    )


def prepare_label_studio_dataset(
    dataset_root: Path,
    output_root: Path,
    repo_root: Path,
    *,
    overwrite: bool = False,
) -> dict[str, int]:
    """Build a hard-link staging tree and metadata artifacts for Label Studio.

    Args:
        dataset_root (Path): Root containing the fixed ZS32 hand/view dataset layout.
        output_root (Path): Destination for hard-linked images and Label Studio artifacts.
        repo_root (Path): Repository root used to record relative manifest paths.
        overwrite (bool): Whether to replace an existing non-empty safe output root. Defaults to ``False``.

    Returns:
        dict[str, int]: Counts for total images, images per hand, and physical groups.

    Raises:
        RuntimeError: If no images are found or a staged image is not a hard link.
        ValueError: If source metadata, output safety, or target uniqueness validation fails.
    """
    images = discover_labeling_images(dataset_root)
    resolved_output = output_root.resolve(strict=False)
    source_hand_roots = ((dataset_root / hand).resolve(strict=False) for hand in HANDS)
    if any(resolved_output.is_relative_to(source_root) for source_root in source_hand_roots):
        msg = f"Refusing to overwrite unsafe output root: {output_root}"
        raise ValueError(msg)
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
            msg = f"Duplicate Label Studio target path: {target}"
            raise ValueError(msg)
        used_targets.add(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(image.source_path, target)
        if (
            target.stat().st_dev != image.source_path.stat().st_dev
            or target.stat().st_ino != image.source_path.stat().st_ino
        ):
            msg = f"Output is not a hard link to source: {target}"
            raise RuntimeError(msg)
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
