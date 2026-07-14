"""Materialize canonical crops for PatchCore/EfficientAD/AnomalyDINO adapters."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

from .export_common import materialize_adapter


def anomalib_materialized_path(row: Mapping[str, str]) -> str:
    label = row["label"]
    if row["split"] == "train" and label == "defect":
        partition = "train_defect_reference"
    else:
        partition = row["split"]
    category = (
        "good"
        if label == "normal"
        else "defect_" + hashlib.sha256(row["defect_type"].encode("utf-8")).hexdigest()[:12]
    )
    return (
        f"{row['hand']}/{row['view']}/{partition}/{category}/{row['sample_id']}.png"
    )


def export_anomalib_dataset(
    *,
    release_root: Path,
    output_root: Path,
    export_id: str,
) -> Path:
    """Publish an anomaly-family-neutral view; recipe selects exactly one family later."""
    return materialize_adapter(
        release_root=release_root,
        output_root=output_root,
        export_id=export_id,
        adapter="anomalib",
        destination_for=anomalib_materialized_path,
    )
