"""Runtime implementation of the one authoritative ZS32 spatial crop."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from types import MappingProxyType

from zs32_inspection.data.roi import ImageCropper, validate_crop_result
from zs32_inspection.domain.contracts import DeploymentContract
from zs32_inspection.domain.identity import RoiSample
from zs32_inspection.models.base import ModelInput

from .orchestrator import InspectionRequest


class RuntimeCropError(RuntimeError):
    """Canonical online crop creation failed closed."""


def _read_private_regular(path: Path, *, root: Path, expected_sha256: str) -> bytes:
    """Read a capture source once while rejecting links and path escapes."""
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
        metadata = path.stat(follow_symlinks=False)
    except (OSError, ValueError) as error:
        raise RuntimeCropError(f"capture source path is invalid: {path}: {error}") from error
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise RuntimeCropError(f"capture source must be a private regular file: {path}")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            before = os.fstat(descriptor)
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise RuntimeCropError(f"capture source cannot be read safely: {path}: {error}") from error
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or not stat.S_ISREG(after.st_mode) or after.st_nlink != 1:
        raise RuntimeCropError(f"capture source changed while being read: {path}")
    payload = b"".join(chunks)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise RuntimeCropError(
            f"capture source SHA-256 mismatch: expected {expected_sha256}, found {actual}: {path}"
        )
    return payload


def _write_exclusive(path: Path, payload: bytes) -> None:
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o440)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(descriptor)
    except OSError as error:
        raise RuntimeCropError(f"canonical crop cannot be written exclusively: {path}: {error}") from error


class AuthoritativeRuntimeCropper:
    """Create exactly one canonical crop per required view in an ephemeral root.

    The injected codec receives only the ROI from the verified deployment
    contract.  No model, template, or CLI option can provide alternate crop
    geometry.
    """

    def __init__(self, *, codec: ImageCropper, work_root: Path) -> None:
        if not callable(getattr(codec, "crop_png", None)):
            raise TypeError("runtime crop codec must implement data.roi.ImageCropper")
        root = Path(work_root).expanduser()
        if root.is_symlink():
            raise RuntimeCropError(f"runtime crop work root must not be a symlink: {root}")
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not root.is_dir():
            raise RuntimeCropError(f"runtime crop work root is not a directory: {root}")
        self._codec = codec
        self._work_root = root.resolve()

    def crop_batch(
        self,
        request: InspectionRequest,
        contract: DeploymentContract,
    ) -> MappingProxyType[str, ModelInput]:
        """Crop the exact verified capture once using the release ROI boxes."""
        if not isinstance(request, InspectionRequest) or not isinstance(contract, DeploymentContract):
            raise TypeError("runtime crop requires verified request and deployment contract")
        hand_roi = contract.roi.require_ready(
            request.capture.part.hand,
            contract.topology.required_views,
        )
        directory_name = hashlib.sha256(
            f"{request.inspection_id}\0{request.capture.capture_set_id}".encode("utf-8")
        ).hexdigest()
        output_root = self._work_root / directory_name
        try:
            output_root.mkdir(mode=0o700)
        except FileExistsError as error:
            raise RuntimeCropError(
                f"runtime crop directory already exists; refusing overwrite: {output_root}"
            ) from error
        output: dict[str, ModelInput] = {}
        try:
            for view_index, view_id in enumerate(contract.topology.required_views, start=1):
                source = request.capture.images[view_id]
                source_path = request.capture_root / source.relative_path
                source_bytes = _read_private_regular(
                    source_path,
                    root=request.capture_root,
                    expected_sha256=source.image_sha256,
                )
                roi = hand_roi.views[view_id]
                crop = self._codec.crop_png(
                    source_bytes,
                    source_width=source.width,
                    source_height=source.height,
                    roi=roi,
                )
                validate_crop_result(crop, roi)
                crop_path = output_root / f"view_{view_index:03d}.png"
                _write_exclusive(crop_path, crop.image_bytes)
                persisted_digest = hashlib.sha256(crop_path.read_bytes()).hexdigest()
                if persisted_digest != crop.sha256:
                    raise RuntimeCropError(f"persisted canonical crop changed: {view_id}")
                sample = RoiSample(
                    part=request.capture.part,
                    capture_set_id=request.capture.capture_set_id,
                    view_id=view_id,
                    source_sha256=source.image_sha256,
                    crop_sha256=crop.sha256,
                    roi_config_id=contract.roi.roi_config_id,
                    crop_width=crop.width,
                    crop_height=crop.height,
                )
                output[view_id] = ModelInput(
                    sample=sample,
                    crop_path=crop_path,
                    roi_digest=contract.roi.roi_sha256,
                )
        except BaseException:
            for path in output_root.glob("*"):
                if path.is_file() and not path.is_symlink():
                    path.chmod(0o600)
                    path.unlink(missing_ok=True)
            output_root.rmdir()
            raise
        return MappingProxyType(output)


def create_opencv_runtime_cropper(work_root: Path) -> AuthoritativeRuntimeCropper:
    """Construct the Linux OpenCV codec lazily at the executable boundary."""
    from zs32_inspection.data.image_codec import OpenCvPngCropper

    return AuthoritativeRuntimeCropper(codec=OpenCvPngCropper(), work_root=work_root)


__all__ = [
    "AuthoritativeRuntimeCropper",
    "RuntimeCropError",
    "create_opencv_runtime_cropper",
]
