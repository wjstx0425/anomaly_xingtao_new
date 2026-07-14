"""Filesystem implementation of the raw-capture atomic publication protocol."""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

from zs32_inspection.runtime.publisher import AtomicDirectoryPublisher, PublicationError

from .contracts import CaptureFrame, CapturePlan, CaptureRequest, RoundConfirmation
from .gate_policy import CAPTURE_GATE_EVIDENCE_SCHEMA_VERSION, CaptureGateProvenance
from .manifests import (
    CAPTURE_SET_COLUMNS,
    IMAGE_COLUMNS,
    CaptureBundle,
    CaptureImageRow,
    build_incomplete_rows,
)


class CapturePublicationError(RuntimeError):
    """A complete or incomplete capture could not be published atomically."""


def _csv_bytes(columns: Sequence[str], rows: Sequence[Mapping[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


class AtomicCaptureStore:
    """Store complete capture sets and quarantine incomplete ones.

    Each complete set is a no-replace atomic directory at
    ``<raw>/<session>/images/<capture_set_id>``.  This preserves the blueprint's
    image path, supports multiple sets in one session, and keeps both CSV tables
    beside the exact image generation they describe.  Session-wide indexes can
    be derived from these immutable per-set tables and are never authoritative.
    """

    def __init__(self, raw_root: Path) -> None:
        self._raw_root = raw_root

    def publish_complete(self, plan: CapturePlan, bundle: CaptureBundle) -> Path:
        """Publish one complete set with Linux ``RENAME_NOREPLACE`` semantics."""
        bundle.validate(plan)
        output_root = self._raw_root / bundle.capture_session / "images"
        capture_csv = _csv_bytes(
            CAPTURE_SET_COLUMNS,
            (bundle.capture_set.to_csv_row(),),
        )
        images_csv = _csv_bytes(
            IMAGE_COLUMNS,
            tuple(row.to_csv_row() for row in bundle.images),
        )
        try:
            with AtomicDirectoryPublisher(
                output_root,
                bundle.capture_set.capture_set_id,
            ) as publisher:
                for row in bundle.images:
                    publisher.write_bytes(f"{row.view_id}.png", bundle.payload_by_view[row.view_id])
                publisher.write_bytes("capture_sets.csv", capture_csv)
                publisher.write_bytes("images.csv", images_csv)
                publisher.write_json(
                    "capture_gates.json",
                    {
                        "schema_version": CAPTURE_GATE_EVIDENCE_SCHEMA_VERSION,
                        "capture_set_id": bundle.capture_set.capture_set_id,
                        "gate_provenance": bundle.gate_provenance.as_dict(),
                        "results": [asdict(result) for result in bundle.gate_results],
                    },
                )
                publisher.write_json(
                    "capture_manifest.json",
                    {
                        "schema_version": 2,
                        "capture_session": bundle.capture_session,
                        "capture_set": asdict(bundle.capture_set),
                        "domain_capture_set": {
                            "capture_set_id": bundle.domain_capture_set.capture_set_id,
                            "capture_session_id": bundle.domain_capture_set.capture_session_id,
                            "part_instance_id": bundle.domain_capture_set.part.part_instance_id,
                            "hand": bundle.domain_capture_set.part.hand.value,
                            "product": bundle.domain_capture_set.part.product,
                            "topology_id": bundle.domain_capture_set.topology_id,
                            "topology_sha256": bundle.domain_capture_set.topology_sha256,
                        },
                        "images": [asdict(row) for row in bundle.images],
                        "round_confirmations": [
                            asdict(item) for item in bundle.round_confirmations
                        ],
                    },
                )
                required = {
                    "capture_sets.csv",
                    "images.csv",
                    "capture_gates.json",
                    "capture_manifest.json",
                    *(f"{view}.png" for view in plan.required_views),
                }
                return publisher.finalize(
                    validator=lambda staging: self._verify_complete_staging(
                        staging,
                        bundle.images,
                    ),
                    required_paths=frozenset(required),
                )
        except PublicationError as error:
            raise CapturePublicationError(str(error)) from error

    def publish_incomplete(
        self,
        request: CaptureRequest,
        plan: CapturePlan,
        frames: Sequence[CaptureFrame],
        *,
        started_at: str,
        failure_reason: str,
        gate_provenance: CaptureGateProvenance,
        round_confirmations: Sequence[RoundConfirmation],
    ) -> Path:
        """Atomically retain partial frames outside complete raw discovery roots."""
        capture_set, image_rows = build_incomplete_rows(
            request,
            plan,
            frames,
            started_at=started_at,
            failure_reason=failure_reason,
        )
        output_root = self._raw_root / "_incomplete" / request.capture_session
        try:
            with AtomicDirectoryPublisher(output_root, request.capture_set_id) as publisher:
                for row, frame in zip(image_rows, frames, strict=True):
                    publisher.write_bytes(
                        row.relative_path,
                        frame.image_bytes,
                    )
                publisher.write_bytes(
                    "capture_sets.csv",
                    _csv_bytes(CAPTURE_SET_COLUMNS, (capture_set.to_csv_row(),)),
                )
                publisher.write_bytes(
                    "images.csv",
                    _csv_bytes(
                        IMAGE_COLUMNS,
                        tuple(row.to_csv_row() for row in image_rows),
                    ),
                )
                publisher.write_json(
                    "failure.json",
                    {
                        "schema_version": 2,
                        "capture_session": request.capture_session,
                        "capture_set": asdict(capture_set),
                        "images": [asdict(row) for row in image_rows],
                        "gate_provenance": gate_provenance.as_dict(),
                        "round_confirmations": [
                            asdict(item) for item in round_confirmations
                        ],
                    },
                )
                required = {"capture_sets.csv", "images.csv", "failure.json"}
                required.update(
                    row.relative_path for row in image_rows
                )
                return publisher.finalize(
                    validator=lambda staging: self._verify_incomplete_staging(
                        staging,
                        image_rows,
                    ),
                    required_paths=frozenset(required),
                )
        except PublicationError as error:
            raise CapturePublicationError(str(error)) from error

    @staticmethod
    def _verify_complete_staging(staging: Path, rows: Sequence[CaptureImageRow]) -> None:
        for row in rows:
            path = staging / f"{row.view_id}.png"
            if not path.is_file():
                raise PublicationError(f"staged capture image is missing: {path}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != row.image_sha256:
                raise PublicationError(
                    f"staged image hash mismatch for {row.view_id}: "
                    f"{actual} != {row.image_sha256}"
                )

    @staticmethod
    def _verify_incomplete_staging(
        staging: Path,
        rows: Sequence[CaptureImageRow],
    ) -> None:
        for row in rows:
            path = staging / row.relative_path
            if not path.is_file():
                raise PublicationError(f"staged incomplete image is missing: {path}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != row.image_sha256:
                raise PublicationError(f"staged incomplete image hash mismatch: {row.view_id}")
