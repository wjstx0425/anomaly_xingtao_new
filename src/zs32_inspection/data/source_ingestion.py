"""Bridge verified raw CaptureBundle objects into canonical dataset sources."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from zs32_inspection.capture.manifests import CaptureBundle
from zs32_inspection.domain.topology import CaptureTopology

from .dataset_release import SourceSample
from .roi import LabelDocument


@dataclass(frozen=True, slots=True)
class CaptureSemantics:
    """Human-approved part-level label and per-view 4K annotations."""

    label: str
    defect_type: str
    annotations: Mapping[str, LabelDocument]

    def __post_init__(self) -> None:
        if self.label not in {"normal", "defect"}:
            raise ValueError(f"capture label must be normal or defect, got {self.label!r}")
        if self.label == "normal" and self.defect_type:
            raise ValueError("normal capture semantics cannot include defect_type")
        if self.label == "defect" and not self.defect_type.strip():
            raise ValueError("defect capture semantics require defect_type")
        if any(not isinstance(item, LabelDocument) for item in self.annotations.values()):
            raise TypeError("capture annotations must contain only LabelDocument values")
        object.__setattr__(
            self,
            "annotations",
            MappingProxyType(dict(self.annotations)),
        )


def sources_from_capture_bundle(
    bundle: CaptureBundle,
    topology: CaptureTopology,
    semantics: CaptureSemantics,
) -> tuple[SourceSample, ...]:
    """Create one source per required view without filename identity inference."""
    bundle.domain_capture_set.validate_required_views(topology.required_views)
    if set(semantics.annotations) != set(topology.required_views):
        raise ValueError(
            "annotation view set must exactly match topology; "
            f"missing={sorted(set(topology.required_views) - set(semantics.annotations))}, "
            f"extra={sorted(set(semantics.annotations) - set(topology.required_views))}"
        )
    row_by_view = {row.view_id: row for row in bundle.images}
    return tuple(
        SourceSample(
            capture_set=bundle.domain_capture_set,
            view_id=view_id,
            source_png=bundle.payload_by_view[view_id],
            source_path=(
                f"raw/{bundle.capture_session}/{row_by_view[view_id].relative_path}"
            ),
            label=semantics.label,
            defect_type=semantics.defect_type,
            annotation=semantics.annotations[view_id],
            gate_provenance=bundle.gate_provenance,
        )
        for view_id in topology.required_views
    )
