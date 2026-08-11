"""Serial-bound two-round acquisition for the BMW laboratory workflow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from uuid import uuid4

import cv2
import numpy as np

from capture_data.collect_multicamera_dataset import (
    CameraAdapter,
    CameraHandle,
    DeviceDescription,
    GroupedTriggerPacer,
    HikvisionAdapter,
    capture_single_round,
    open_cameras,
)

from .config import LabExperimentConfig
from .contracts import CapturedView, CaptureSet, ViewId


_ROUND_VIEWS: dict[str, tuple[ViewId, ViewId, ViewId]] = {
    "front": (ViewId.FRONT, ViewId.FRONT_LEFT, ViewId.FRONT_RIGHT),
    "back": (ViewId.BACK, ViewId.BACK_LEFT, ViewId.BACK_RIGHT),
}
_VIEW_SERIALS: dict[ViewId, str] = {
    ViewId.FRONT: "DA9805574",
    ViewId.FRONT_LEFT: "DA9625347",
    ViewId.FRONT_RIGHT: "DB0968108",
    ViewId.BACK: "DA9805574",
    ViewId.BACK_LEFT: "DA9625347",
    ViewId.BACK_RIGHT: "DB0968108",
}


def select_serial_bound_devices(
    devices: Sequence[DeviceDescription],
    serials: Sequence[str],
) -> tuple[DeviceDescription, ...]:
    """Select each declared camera once, preserving declared physical-slot order."""
    selected: list[DeviceDescription] = []
    for serial in serials:
        matches = [device for device in devices if device.serial == serial]
        if not matches:
            raise ValueError(f"camera serial {serial!r} is unavailable")
        if len(matches) != 1:
            raise ValueError(f"camera serial {serial!r} appears multiple times")
        selected.append(matches[0])
    if len({device.serial for device in selected}) != len(selected):
        raise ValueError("declared camera serials must be unique")
    return tuple(selected)


def map_round_images(
    round_id: str,
    images: Sequence[np.ndarray],
) -> Mapping[ViewId, np.ndarray]:
    """Map one ordered three-camera trigger pass to its declared side's view IDs."""
    try:
        view_ids = _ROUND_VIEWS[round_id]
    except KeyError as error:
        raise ValueError(f"unknown capture round {round_id!r}") from error
    if len(images) != len(view_ids):
        raise ValueError("a capture round must contain exactly three images")
    dimensions: tuple[int, ...] | None = None
    mapped: dict[ViewId, np.ndarray] = {}
    for view_id, image in zip(view_ids, images, strict=True):
        if not isinstance(image, np.ndarray) or image.ndim not in {2, 3} or image.size == 0:
            raise ValueError(f"capture image for {view_id.value} must be a non-empty 2D or 3D array")
        if dimensions is None:
            dimensions = image.shape[:2]
        elif image.shape[:2] != dimensions:
            raise ValueError("all images in one capture round must have the same dimensions")
        mapped[view_id] = image
    return mapped


def _validate_image_dimensions(image: np.ndarray, width: int, height: int, view_id: ViewId) -> None:
    if image.shape[:2] != (height, width):
        raise ValueError(
            f"capture image for {view_id.value} has dimensions {image.shape[1]}x{image.shape[0]}; "
            f"expected {width}x{height}"
        )


def build_capture_set(
    front: Mapping[ViewId, np.ndarray],
    back: Mapping[ViewId, np.ndarray],
) -> CaptureSet:
    """Create one immutable six-view capture set from two completed operator rounds."""
    overlap = set(front) & set(back)
    if overlap:
        raise ValueError(f"front and back rounds overlap on views: {', '.join(sorted(view.value for view in overlap))}")
    for round_id, supplied, expected in (
        ("front", front, set(_ROUND_VIEWS["front"])),
        ("back", back, set(_ROUND_VIEWS["back"])),
    ):
        missing = expected - set(supplied)
        extra = set(supplied) - expected
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing {', '.join(sorted(view.value for view in missing))}")
            if extra:
                details.append(f"unexpected {', '.join(sorted(view.value for view in extra))}")
            raise ValueError(f"{round_id} round views are incomplete: {'; '.join(details)}")
    created_at = datetime.now(timezone.utc)
    images = {**front, **back}
    dimensions: tuple[int, int] | None = None
    owned_images: dict[ViewId, np.ndarray] = {}
    for view_id in ViewId:
        image = images[view_id]
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise ValueError(f"capture image for {view_id.value} must be a uint8 numpy array")
        if image.ndim not in {2, 3} or image.size == 0:
            raise ValueError(f"capture image for {view_id.value} must be a non-empty 2D or 3D array")
        if dimensions is None:
            dimensions = image.shape[:2]
        elif image.shape[:2] != dimensions:
            raise ValueError("all six capture images must have the same dimensions")
        owned_image = image.copy()
        owned_image.flags.writeable = False
        owned_images[view_id] = owned_image
    return CaptureSet(
        capture_set_id=f"capture-{uuid4().hex}",
        views={
            view_id: CapturedView(
                view_id=view_id,
                camera_serial=_VIEW_SERIALS[view_id],
                image=owned_images[view_id],
                captured_at=created_at,
            )
            for view_id in ViewId
        },
        created_at=created_at,
    )


def load_capture_set(path: Path) -> CaptureSet:
    """Load exactly one BMP or PNG image for each canonical BMW laboratory view."""
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"capture-set path is not a directory: {root}")
    images: dict[ViewId, np.ndarray] = {}
    for candidate in sorted(root.iterdir()):
        if not candidate.is_file():
            continue
        if candidate.suffix not in {".bmp", ".png"}:
            raise ValueError(f"capture-set contains unsupported file: {candidate.name}")
        try:
            view_id = ViewId(candidate.stem)
        except ValueError as error:
            raise ValueError(f"capture-set contains unknown view filename: {candidate.name}") from error
        if view_id in images:
            raise ValueError(f"capture-set contains duplicate view: {view_id.value}")
        image = cv2.imread(str(candidate), cv2.IMREAD_UNCHANGED)
        if image is None or image.size == 0:
            raise ValueError(f"cannot read capture-set image: {candidate}")
        images[view_id] = image
    missing = set(ViewId) - set(images)
    if missing:
        raise ValueError(f"capture-set is missing views: {', '.join(sorted(view.value for view in missing))}")
    dimensions = {image.shape[:2] for image in images.values()}
    if len(dimensions) != 1:
        raise ValueError("capture-set images must all have the same dimensions")
    front = {view_id: images[view_id] for view_id in _ROUND_VIEWS["front"]}
    back = {view_id: images[view_id] for view_id in _ROUND_VIEWS["back"]}
    return build_capture_set(front, back)


class LabCameraSession:
    """Keep the three declared BMW cameras open across explicit front/back rounds."""

    def __init__(
        self,
        config: LabExperimentConfig,
        *,
        adapter: CameraAdapter | None = None,
    ) -> None:
        self.config = config
        self._adapter = HikvisionAdapter.load() if adapter is None else adapter
        self._camera_context: AbstractContextManager[list[CameraHandle]] | None = None
        self._handles: list[CameraHandle] | None = None
        self._pacer = GroupedTriggerPacer(0.0)

    def __enter__(self) -> LabCameraSession:
        if self._camera_context is not None:
            raise RuntimeError("three-camera session is already open")
        serials = tuple(slot.serial for slot in self.config.topology.camera_slots)
        selected = select_serial_bound_devices(self._adapter.list_devices(), serials)
        camera_context = open_cameras(selected, self._adapter, self.config.capture.gain)
        self._camera_context = camera_context
        try:
            handles = camera_context.__enter__()
        except BaseException:
            self._camera_context = None
            raise
        if len(handles) != len(self.config.topology.camera_slots):
            self._camera_context = None
            camera_context.__exit__(None, None, None)
            raise RuntimeError(f"three-camera session opened {len(handles)} handles")
        self._handles = handles
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        camera_context = self._camera_context
        self._camera_context = None
        self._handles = None
        if camera_context is None:
            return False
        return bool(camera_context.__exit__(exc_type, exc_value, traceback))

    def capture_round(self, round_id: str) -> Mapping[ViewId, np.ndarray]:
        """Capture one requested physical side after any configured grouped warm-up passes."""
        if round_id not in _ROUND_VIEWS:
            raise ValueError(f"unknown capture round {round_id!r}")
        if self._handles is None:
            raise RuntimeError("three-camera session is not open")
        for _ in range(self.config.capture.warmup_frames):
            self._capture_and_validate(round_id)
        return self._capture_and_validate(round_id)

    def _capture_and_validate(self, round_id: str) -> Mapping[ViewId, np.ndarray]:
        assert self._handles is not None
        results = capture_single_round(
            self._handles,
            self._adapter,
            self.config.capture.exposure,
            self.config.capture.timeout_ms,
            self._pacer,
        )
        images = map_round_images(round_id, tuple(result.final_image for result in results))
        expected_views = tuple(
            slot.front if round_id == "front" else slot.back for slot in self.config.topology.camera_slots
        )
        if tuple(images) != expected_views:
            raise RuntimeError(f"topology round mapping differs from canonical {round_id} views")
        for view_id, image in images.items():
            _validate_image_dimensions(
                image,
                self.config.capture.image_width,
                self.config.capture.image_height,
                view_id,
            )
        return images


__all__ = [
    "LabCameraSession",
    "build_capture_set",
    "load_capture_set",
    "map_round_images",
    "select_serial_bound_devices",
]
