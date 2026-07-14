"""Materialize canonical crops for per-hand/per-view template training."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

from .export_common import materialize_adapter


def template_materialized_path(row: Mapping[str, str]) -> str:
    category = (
        "normal"
        if row["label"] == "normal"
        else "defect/defect_"
        + hashlib.sha256(row["defect_type"].encode("utf-8")).hexdigest()[:12]
    )
    return (
        f"{row['hand']}/{row['view']}/{row['split']}/{category}/{row['sample_id']}.png"
    )


def export_template_dataset(
    *,
    release_root: Path,
    output_root: Path,
    export_id: str,
) -> Path:
    """Publish template data without crop, resize, or mirror transformations."""
    return materialize_adapter(
        release_root=release_root,
        output_root=output_root,
        export_id=export_id,
        adapter="template",
        destination_for=template_materialized_path,
    )
