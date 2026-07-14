"""Strict loader for atomically published complete raw capture sets."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import stat

from zs32_inspection.domain.identity import CaptureSet, Hand, PartIdentity, ViewImage
from zs32_inspection.domain.topology import CaptureTopology
from zs32_inspection.runtime.publisher import tree_checksums

from .contracts import CapturePlan, RoundConfirmation
from .errors import CaptureDataIntegrityError
from .gates import CaptureGateResult
from .gate_policy import CAPTURE_GATE_EVIDENCE_SCHEMA_VERSION, CaptureGateProvenance
from .manifests import (
    CAPTURE_SET_COLUMNS,
    IMAGE_COLUMNS,
    CaptureBundle,
    CaptureImageRow,
    CaptureSetRow,
)


def _require_regular_capture_file(path: Path, description: str) -> None:
    """Classify absence/symlinks as bad data while allowing I/O errors to propagate."""
    if path.is_symlink():
        raise CaptureDataIntegrityError(f"{description} must not be a symlink: {path}")
    try:
        mode = path.stat().st_mode
    except FileNotFoundError as error:
        raise CaptureDataIntegrityError(f"{description} is missing: {path}") from error
    if not stat.S_ISREG(mode):
        raise CaptureDataIntegrityError(f"{description} is not a regular file: {path}")


def _read_csv(path: Path, expected_columns: tuple[str, ...]) -> list[dict[str, str]]:
    _require_regular_capture_file(path, "capture manifest table")
    try:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if tuple(reader.fieldnames or ()) != expected_columns:
                raise CaptureDataIntegrityError(
                    f"capture manifest header differs from frozen schema: {path.name}"
                )
            rows = list(reader)
    except OSError:
        raise
    except (UnicodeError, csv.Error) as error:
        raise CaptureDataIntegrityError(
            f"capture manifest cannot be parsed: {path.name}: {error}"
        ) from error
    if any(set(row) != set(expected_columns) or any(value is None for value in row.values()) for row in rows):
        raise CaptureDataIntegrityError(
            f"capture manifest contains malformed or extra cells: {path.name}"
        )
    return rows


def _optional_int(value: str) -> int | None:
    return None if value == "" else int(value)


def _optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def _load_capture_gate_results(
    path: Path,
    capture_set_id: str,
) -> tuple[tuple[CaptureGateResult, ...], CaptureGateProvenance]:
    """Strictly parse the immutable per-view quality/registration evidence."""
    _require_regular_capture_file(path, "capture gate evidence")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CaptureDataIntegrityError(
            f"capture gate evidence cannot be parsed: {error}"
        ) from error
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "capture_set_id",
        "gate_provenance",
        "results",
    }:
        raise CaptureDataIntegrityError("capture gate evidence envelope is malformed")
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != CAPTURE_GATE_EVIDENCE_SCHEMA_VERSION
        or not isinstance(document["capture_set_id"], str)
        or document["capture_set_id"] != capture_set_id
    ):
        raise CaptureDataIntegrityError(
            "capture gate evidence schema or capture identity is invalid"
        )
    raw_results = document["results"]
    if not isinstance(raw_results, list):
        raise CaptureDataIntegrityError("capture gate evidence results must be an array")
    parsed: list[CaptureGateResult] = []
    for index, result in enumerate(raw_results):
        if not isinstance(result, dict) or set(result) != {
            "gate",
            "passed",
            "reason",
            "view_id",
        }:
            raise CaptureDataIntegrityError(f"capture gate result {index} is malformed")
        try:
            parsed.append(
                CaptureGateResult(
                    gate=result["gate"],
                    passed=result["passed"],
                    reason=result["reason"],
                    view_id=result["view_id"],
                )
            )
        except (TypeError, ValueError) as error:
            raise CaptureDataIntegrityError(
                f"capture gate result {index} is invalid: {error}"
            ) from error
    raw_provenance = document["gate_provenance"]
    if not isinstance(raw_provenance, dict):
        raise CaptureDataIntegrityError("capture gate provenance must be an object")
    try:
        provenance = CaptureGateProvenance.from_mapping(raw_provenance)
    except (TypeError, ValueError) as error:
        raise CaptureDataIntegrityError(
            f"capture gate provenance is invalid: {error}"
        ) from error
    return tuple(parsed), provenance


def _load_round_confirmations(
    path: Path,
    capture_set_id: str,
) -> tuple[RoundConfirmation, ...]:
    """Read the per-round operator acknowledgements from the canonical manifest."""
    _require_regular_capture_file(path, "capture manifest")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CaptureDataIntegrityError(f"capture manifest cannot be parsed: {error}") from error
    expected = {
        "schema_version",
        "capture_session",
        "capture_set",
        "domain_capture_set",
        "images",
        "round_confirmations",
    }
    if not isinstance(document, dict) or set(document) != expected:
        raise CaptureDataIntegrityError("capture manifest envelope is malformed")
    raw_capture_set = document.get("capture_set")
    if (
        document.get("schema_version") != 2
        or not isinstance(raw_capture_set, dict)
        or raw_capture_set.get("capture_set_id") != capture_set_id
        or not isinstance(document.get("round_confirmations"), list)
    ):
        raise CaptureDataIntegrityError("capture manifest identity is invalid")
    confirmations: list[RoundConfirmation] = []
    fields = {"round_id", "prompt", "confirmed_by", "prompted_at", "confirmed_at"}
    for index, item in enumerate(document["round_confirmations"]):
        if not isinstance(item, dict) or set(item) != fields:
            raise CaptureDataIntegrityError(
                f"round confirmation {index} fields differ from the strict schema"
            )
        try:
            confirmations.append(RoundConfirmation(**item))
        except (TypeError, ValueError) as error:
            raise CaptureDataIntegrityError(
                f"round confirmation {index} is invalid: {error}"
            ) from error
    return tuple(confirmations)


def _verify_tree(root: Path) -> None:
    checksums_path = root / "checksums.sha256"
    publication_path = root / "publication_root.json"
    _require_regular_capture_file(checksums_path, "capture checksums")
    _require_regular_capture_file(publication_path, "capture publication root")
    checksum_bytes = checksums_path.read_bytes()
    try:
        publication_root = json.loads(publication_path.read_text(encoding="utf-8"))
    except OSError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CaptureDataIntegrityError(
            f"capture publication root cannot be parsed: {error}"
        ) from error
    if not isinstance(publication_root, dict):
        raise CaptureDataIntegrityError("capture publication root must be an object")
    if (
        publication_root.get("publication_id") != root.name
        or publication_root.get("algorithm") != "sha256(checksums.sha256 bytes)"
        or publication_root.get("root_sha256")
        != hashlib.sha256(checksum_bytes).hexdigest()
    ):
        raise CaptureDataIntegrityError(
            f"raw capture publication root digest mismatch: {root}"
        )
    recorded: dict[str, str] = {}
    try:
        for line in checksum_bytes.decode("utf-8").splitlines():
            digest, relative = line.split("  ", maxsplit=1)
            if relative in recorded:
                raise ValueError(f"duplicate checksum path: {relative}")
            recorded[relative] = digest
    except (UnicodeError, ValueError) as error:
        raise CaptureDataIntegrityError(
            f"capture checksum index cannot be parsed: {error}"
        ) from error
    actual = tree_checksums(
        root,
        excluded=frozenset({"checksums.sha256", "publication_root.json"}),
    )
    if recorded != actual:
        raise CaptureDataIntegrityError(f"raw capture tree checksum mismatch: {root}")


def load_capture_bundle(set_root: Path, topology: CaptureTopology) -> CaptureBundle:
    """Load one complete capture and prove all hashes and identities."""
    if set_root.is_symlink():
        raise CaptureDataIntegrityError(
            f"capture set root must not be a symlink: {set_root}"
        )
    set_root = set_root.resolve()
    if set_root.parent.name != "images":
        raise CaptureDataIntegrityError(
            "complete capture set must use <raw>/<session>/images/<capture_set_id> layout"
        )
    _verify_tree(set_root)
    capture_rows = _read_csv(set_root / "capture_sets.csv", CAPTURE_SET_COLUMNS)
    image_rows = _read_csv(set_root / "images.csv", IMAGE_COLUMNS)
    if len(capture_rows) != 1 or not image_rows:
        raise CaptureDataIntegrityError(
            "complete capture set requires one capture row and non-empty image rows"
        )
    raw_capture = capture_rows[0]
    try:
        capture_values = {
            "capture_set_id": raw_capture["capture_set_id"],
            "part_instance_id": raw_capture["part_instance_id"],
            "product": raw_capture["product"],
            "hand": raw_capture["hand"],
            "topology_id": raw_capture["topology_id"],
            "topology_sha256": raw_capture["topology_sha256"],
            "expected_view_count": int(raw_capture["expected_view_count"]),
            "captured_view_count": int(raw_capture["captured_view_count"]),
            "status": raw_capture["status"],
            "started_at": raw_capture["started_at"],
            "completed_at": raw_capture["completed_at"],
            "failure_reason": raw_capture["failure_reason"],
        }
    except (KeyError, TypeError, ValueError) as error:
        raise CaptureDataIntegrityError(
            f"capture set table cannot be parsed: {error}"
        ) from error
    try:
        capture_row = CaptureSetRow(
            **capture_values,
        )
    except ValueError as error:
        raise CaptureDataIntegrityError(f"capture set row is invalid: {error}") from error
    if set_root.name != capture_row.capture_set_id:
        raise CaptureDataIntegrityError(
            f"capture directory identity {set_root.name!r} conflicts with manifest "
            f"{capture_row.capture_set_id!r}"
        )
    parsed_image_items: list[CaptureImageRow] = []
    for index, row in enumerate(image_rows):
        try:
            json.loads(row["capture_parameters_json"])
            image_values = {
                "capture_set_id": row["capture_set_id"],
                "part_instance_id": row["part_instance_id"],
                "round_id": row["round_id"],
                "view_id": row["view_id"],
                "camera_slot_id": row["camera_slot_id"],
                "camera_serial": row["camera_serial"],
                "device_index": _optional_int(row["device_index"]),
                "relative_path": row["relative_path"],
                "image_sha256": row["image_sha256"],
                "width": int(row["width"]),
                "height": int(row["height"]),
                "capture_mode": row["capture_mode"],
                "exposure": _optional_float(row["exposure"]),
                "gain": _optional_float(row["gain"]),
                "capture_parameters_json": row["capture_parameters_json"],
                "captured_at": row["captured_at"],
                "status": row["status"],
                "error": row["error"],
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise CaptureDataIntegrityError(
                f"capture image row {index} cannot be parsed: {error}"
            ) from error
        try:
            parsed_image_items.append(CaptureImageRow(**image_values))
        except ValueError as error:
            raise CaptureDataIntegrityError(
                f"capture image row {index} is invalid: {error}"
            ) from error
    parsed_images = tuple(parsed_image_items)
    gate_results, gate_provenance = _load_capture_gate_results(
        set_root / "capture_gates.json",
        capture_row.capture_set_id,
    )
    round_confirmations = _load_round_confirmations(
        set_root / "capture_manifest.json",
        capture_row.capture_set_id,
    )
    payload_by_view: dict[str, bytes] = {}
    view_images: dict[str, ViewImage] = {}
    for row in parsed_images:
        image_path = set_root / f"{row.view_id}.png"
        _require_regular_capture_file(image_path, "raw capture image")
        payload = image_path.read_bytes()
        if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise CaptureDataIntegrityError(
                f"raw capture image is not a PNG: {row.view_id}"
            )
        if hashlib.sha256(payload).hexdigest() != row.image_sha256:
            raise CaptureDataIntegrityError(
                f"raw capture image hash mismatch: {row.view_id}"
            )
        payload_by_view[row.view_id] = payload
        try:
            view_images[row.view_id] = ViewImage(
                view_id=row.view_id,
                round_id=row.round_id,
                camera_slot_id=row.camera_slot_id,
                camera_serial=row.camera_serial,
                relative_path=row.relative_path,
                image_sha256=row.image_sha256,
                width=row.width,
                height=row.height,
            )
        except ValueError as error:
            raise CaptureDataIntegrityError(
                f"capture image identity is invalid for view {row.view_id!r}: {error}"
            ) from error
    try:
        capture_set = CaptureSet(
            capture_set_id=capture_row.capture_set_id,
            capture_session_id=set_root.parent.parent.name,
            part=PartIdentity(capture_row.part_instance_id, Hand.parse(capture_row.hand)),
            topology_id=capture_row.topology_id,
            topology_sha256=capture_row.topology_sha256,
            images=view_images,
        )
    except ValueError as error:
        raise CaptureDataIntegrityError(f"capture set identity is invalid: {error}") from error
    bundle = CaptureBundle(
        capture_session=set_root.parent.parent.name,
        domain_capture_set=capture_set,
        capture_set=capture_row,
        images=parsed_images,
        gate_results=gate_results,
        gate_provenance=gate_provenance,
        round_confirmations=round_confirmations,
        payload_by_view=payload_by_view,
    )
    try:
        bundle.validate(CapturePlan.from_topology(topology))
    except ValueError as error:
        raise CaptureDataIntegrityError(
            f"capture set completeness or topology identity is invalid: {error}"
        ) from error
    return bundle


def discover_complete_capture_sets(raw_root: Path) -> tuple[Path, ...]:
    """Discover only complete-store set roots; quarantine is excluded by construction."""
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw capture root is missing: {raw_root}")
    return tuple(
        sorted(
            (
                path.parent
                for path in raw_root.glob("*/images/*/capture_manifest.json")
                if path.is_file() and path.parent.parent.parent.name != "_incomplete"
            ),
            key=lambda item: item.as_posix(),
        )
    )
