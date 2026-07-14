"""Linux-only contracts for the single authoritative online ROI crop."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from zs32_inspection.data.roi import CroppedImage
from zs32_inspection.domain.errors import RoiValidationError
from zs32_inspection.domain.identity import CaptureSet, Hand, PartIdentity, ViewImage
from zs32_inspection.runtime.cropper import AuthoritativeRuntimeCropper, RuntimeCropError
from zs32_inspection.runtime.orchestrator import InspectionRequest


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="ZS32 verification is Linux-only")


class RecordingCodec:
    """Return lossless-looking bytes while recording the only permitted ROI."""

    def __init__(self) -> None:
        self.calls: list[tuple[bytes, int, int, object]] = []

    def crop_png(
        self,
        source_png: bytes,
        *,
        source_width: int,
        source_height: int,
        roi: object,
    ) -> CroppedImage:
        self.calls.append((source_png, source_width, source_height, roi))
        payload = b"\x89PNG\r\n\x1a\n" + hashlib.sha256(source_png).digest()
        return CroppedImage(payload, roi.width, roi.height)


def _request(tmp_path: Path, contract: object, *, hand: Hand = Hand.RIGHT) -> InspectionRequest:
    images: dict[str, ViewImage] = {}
    for view in contract.topology.required_views:
        round_id, slot_id, serial = contract.topology.binding_for_view(view)
        content = f"source:{view}:{hand.value}".encode()
        relative = f"images/{view}.png"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        images[view] = ViewImage(
            view_id=view,
            round_id=round_id,
            camera_slot_id=slot_id,
            camera_serial=serial,
            relative_path=relative,
            image_sha256=hashlib.sha256(content).hexdigest(),
            width=contract.roi.source_width,
            height=contract.roi.source_height,
        )
    capture = CaptureSet(
        capture_set_id=f"capture-{hand.value}",
        capture_session_id="session-runtime-crop",
        part=PartIdentity("part-runtime-crop", hand),
        topology_id=contract.topology.topology_id,
        topology_sha256=contract.topology.topology_sha256,
        images=images,
    )
    request = object.__new__(InspectionRequest)
    object.__setattr__(request, "inspection_id", f"inspection-{hand.value}")
    object.__setattr__(request, "release_id", "release-runtime-v1")
    object.__setattr__(request, "capture", capture)
    object.__setattr__(request, "capture_root", tmp_path)
    return request


def test_every_required_view_is_cropped_once_with_only_contract_roi(
    tmp_path: Path,
    compiled_contract_factory: object,
) -> None:
    contract = compiled_contract_factory()
    request = _request(tmp_path / "capture", contract)
    codec = RecordingCodec()
    cropper = AuthoritativeRuntimeCropper(codec=codec, work_root=tmp_path / "work")

    outputs = cropper.crop_batch(request, contract)

    assert tuple(outputs) == contract.topology.required_views
    assert len(codec.calls) == len(contract.topology.required_views)
    for view, call in zip(contract.topology.required_views, codec.calls, strict=True):
        source_bytes, width, height, roi = call
        assert source_bytes == (request.capture_root / request.capture.images[view].relative_path).read_bytes()
        assert (width, height) == (contract.roi.source_width, contract.roi.source_height)
        assert roi is contract.roi.hands[Hand.RIGHT].views[view]
        assert outputs[view].sample.source_sha256 == request.capture.images[view].image_sha256
        assert outputs[view].sample.roi_config_id == contract.roi.roi_config_id
        assert outputs[view].roi_digest == contract.roi.roi_sha256
        assert outputs[view].sample.crop_sha256 == hashlib.sha256(
            outputs[view].crop_path.read_bytes()
        ).hexdigest()


def test_same_inspection_crop_directory_is_never_overwritten(
    tmp_path: Path,
    compiled_contract_factory: object,
) -> None:
    contract = compiled_contract_factory()
    request = _request(tmp_path / "capture", contract)
    codec = RecordingCodec()
    cropper = AuthoritativeRuntimeCropper(codec=codec, work_root=tmp_path / "work")
    first = cropper.crop_batch(request, contract)
    original = {view: item.crop_path.read_bytes() for view, item in first.items()}

    with pytest.raises(RuntimeCropError, match="refusing overwrite"):
        cropper.crop_batch(request, contract)

    assert len(codec.calls) == len(contract.topology.required_views)
    assert {view: item.crop_path.read_bytes() for view, item in first.items()} == original


def test_source_hash_is_rechecked_before_the_only_crop(
    tmp_path: Path,
    compiled_contract_factory: object,
) -> None:
    contract = compiled_contract_factory()
    request = _request(tmp_path / "capture", contract)
    first_view = contract.topology.required_views[0]
    (request.capture_root / request.capture.images[first_view].relative_path).write_bytes(b"changed")
    codec = RecordingCodec()

    with pytest.raises(RuntimeCropError, match="source SHA-256 mismatch"):
        AuthoritativeRuntimeCropper(codec=codec, work_root=tmp_path / "work").crop_batch(
            request,
            contract,
        )

    assert codec.calls == []
    assert not any((tmp_path / "work").iterdir())


def test_left_pending_roi_fails_closed_without_reusing_right_coordinates(
    tmp_path: Path,
    compiled_contract_factory: object,
) -> None:
    contract = compiled_contract_factory(left_ready=False, hands=("right",))
    request = _request(tmp_path / "capture", contract, hand=Hand.LEFT)
    codec = RecordingCodec()

    with pytest.raises(RoiValidationError, match="pending"):
        AuthoritativeRuntimeCropper(codec=codec, work_root=tmp_path / "work").crop_batch(
            request,
            contract,
        )

    assert codec.calls == []
