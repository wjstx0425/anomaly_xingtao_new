"""Configuration-bound OpenCV affine registration gate.

Registration compares each full capture frame with an explicitly content-
addressed reference for the same topology view.  ECC non-convergence or a pose
outside the configured limits is a recapturable gate failure.  Missing/corrupt
assets, dependency failures, malformed images, and non-finite results raise so
the capture service records ``SYSTEM_ERROR`` instead of mislabelling a part.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from zs32_inspection.domain.contracts import canonical_sha256

from .contracts import (
    CaptureFrame,
    CapturePlan,
    CaptureRequest,
    require_identifier,
    require_sha256,
)
from .gates import CaptureGateResult
from .opencv_quality import _require_exact_frames


DependencyLoader = Callable[[], tuple[Any, Any]]


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _range(value: object, field: str, minimum: float, maximum: float) -> float:
    result = _finite_number(value, field)
    if not minimum <= result <= maximum:
        raise ValueError(f"{field} must be in [{minimum}, {maximum}]")
    return result


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _safe_relative_path(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} must be a safe relative POSIX path")
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class RegistrationReference:
    """Content-addressed full-frame reference owned by one topology view."""

    asset_id: str
    relative_path: str
    sha256: str
    width: int
    height: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "asset_id", require_identifier(self.asset_id, "asset_id"))
        object.__setattr__(
            self,
            "relative_path",
            _safe_relative_path(self.relative_path, "reference relative_path"),
        )
        object.__setattr__(self, "sha256", require_sha256(self.sha256, "reference sha256"))
        object.__setattr__(self, "width", _positive_int(self.width, "reference width"))
        object.__setattr__(self, "height", _positive_int(self.height, "reference height"))

    def as_dict(self) -> dict[str, object]:
        return {
            "asset_id": self.asset_id,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "RegistrationReference":
        expected = {"asset_id", "relative_path", "sha256", "width", "height"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            actual = set(payload) if isinstance(payload, Mapping) else set()
            raise ValueError(
                "registration reference fields differ from the strict schema; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        return cls(
            asset_id=payload["asset_id"],  # type: ignore[arg-type]
            relative_path=payload["relative_path"],  # type: ignore[arg-type]
            sha256=payload["sha256"],  # type: ignore[arg-type]
            width=payload["width"],  # type: ignore[arg-type]
            height=payload["height"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class RegistrationThresholds:
    """Explicit affine/ECC acceptance envelope for one view."""

    analysis_scale: float
    min_ecc_correlation: float
    max_translation_x_px: float
    max_translation_y_px: float
    max_rotation_deg: float
    max_scale_deviation: float
    max_shear: float
    max_iterations: int
    epsilon: float
    gaussian_filter_size: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "analysis_scale",
            _range(self.analysis_scale, "analysis_scale", 0.01, 1.0),
        )
        object.__setattr__(
            self,
            "min_ecc_correlation",
            _range(self.min_ecc_correlation, "min_ecc_correlation", 0.0, 1.0),
        )
        for field in (
            "max_translation_x_px",
            "max_translation_y_px",
            "max_rotation_deg",
            "max_scale_deviation",
            "max_shear",
        ):
            value = _finite_number(getattr(self, field), field)
            if value < 0.0:
                raise ValueError(f"{field} must be non-negative")
            object.__setattr__(self, field, value)
        object.__setattr__(
            self,
            "max_iterations",
            _positive_int(self.max_iterations, "max_iterations"),
        )
        epsilon = _finite_number(self.epsilon, "epsilon")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be greater than zero")
        object.__setattr__(self, "epsilon", epsilon)
        gaussian_size = _positive_int(self.gaussian_filter_size, "gaussian_filter_size")
        if gaussian_size % 2 == 0:
            raise ValueError("gaussian_filter_size must be odd")
        object.__setattr__(self, "gaussian_filter_size", gaussian_size)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "analysis_scale": self.analysis_scale,
            "min_ecc_correlation": self.min_ecc_correlation,
            "max_translation_x_px": self.max_translation_x_px,
            "max_translation_y_px": self.max_translation_y_px,
            "max_rotation_deg": self.max_rotation_deg,
            "max_scale_deviation": self.max_scale_deviation,
            "max_shear": self.max_shear,
            "max_iterations": self.max_iterations,
            "epsilon": self.epsilon,
            "gaussian_filter_size": self.gaussian_filter_size,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "RegistrationThresholds":
        expected = {
            "analysis_scale",
            "min_ecc_correlation",
            "max_translation_x_px",
            "max_translation_y_px",
            "max_rotation_deg",
            "max_scale_deviation",
            "max_shear",
            "max_iterations",
            "epsilon",
            "gaussian_filter_size",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            actual = set(payload) if isinstance(payload, Mapping) else set()
            raise ValueError(
                "registration threshold fields differ from the strict schema; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        return cls(**{key: payload[key] for key in expected})  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class RegistrationViewSpec:
    """Reference plus acceptance thresholds for exactly one view."""

    reference: RegistrationReference
    thresholds: RegistrationThresholds

    def __post_init__(self) -> None:
        if not isinstance(self.reference, RegistrationReference):
            raise TypeError("registration reference has the wrong type")
        if not isinstance(self.thresholds, RegistrationThresholds):
            raise TypeError("registration thresholds have the wrong type")

    def as_dict(self) -> dict[str, object]:
        return {
            "reference": self.reference.as_dict(),
            "thresholds": self.thresholds.as_dict(),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "RegistrationViewSpec":
        expected = {"reference", "thresholds"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            actual = set(payload) if isinstance(payload, Mapping) else set()
            raise ValueError(
                "registration view fields differ from the strict schema; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        return cls(
            RegistrationReference.from_mapping(payload["reference"]),  # type: ignore[arg-type]
            RegistrationThresholds.from_mapping(payload["thresholds"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class RegistrationGateProfile:
    """Immutable topology binding for every registration reference and limit."""

    schema_version: int
    profile_id: str
    product: str
    hand: str
    topology_id: str
    topology_sha256: str
    views: Mapping[str, RegistrationViewSpec]
    profile_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != 1
        ):
            raise ValueError("registration gate schema_version must be 1")
        if self.product != "ZS32":
            raise ValueError("registration gate product must be ZS32")
        if not isinstance(self.hand, str) or self.hand.strip().lower() not in {"left", "right"}:
            raise ValueError("registration gate hand must be explicitly left or right")
        object.__setattr__(self, "hand", self.hand.strip().lower())
        object.__setattr__(self, "profile_id", require_identifier(self.profile_id, "profile_id"))
        object.__setattr__(
            self,
            "topology_id",
            require_identifier(self.topology_id, "topology_id"),
        )
        object.__setattr__(
            self,
            "topology_sha256",
            require_sha256(self.topology_sha256, "topology_sha256"),
        )
        object.__setattr__(
            self,
            "profile_sha256",
            require_sha256(self.profile_sha256, "registration profile_sha256"),
        )
        if not isinstance(self.views, Mapping) or not self.views:
            raise ValueError("registration profile must contain explicit per-view records")
        frozen: dict[str, RegistrationViewSpec] = {}
        asset_ids: set[str] = set()
        asset_paths: set[str] = set()
        for view_id, spec in self.views.items():
            checked_view = require_identifier(view_id, "registration view_id")
            if not isinstance(spec, RegistrationViewSpec):
                raise TypeError(f"registration spec for {checked_view!r} has the wrong type")
            if spec.reference.asset_id in asset_ids:
                raise ValueError(f"duplicate registration asset_id: {spec.reference.asset_id!r}")
            if spec.reference.relative_path in asset_paths:
                raise ValueError(
                    f"registration reference path is reused: {spec.reference.relative_path!r}"
                )
            asset_ids.add(spec.reference.asset_id)
            asset_paths.add(spec.reference.relative_path)
            frozen[checked_view] = spec
        object.__setattr__(self, "views", MappingProxyType(frozen))
        if canonical_sha256(self.as_dict(include_sha256=False)) != self.profile_sha256:
            raise ValueError(
                "registration profile_sha256 does not match the canonical profile payload"
            )

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "product": self.product,
            "hand": self.hand,
            "topology_id": self.topology_id,
            "topology_sha256": self.topology_sha256,
            "views": {view: spec.as_dict() for view, spec in sorted(self.views.items())},
        }
        if include_sha256:
            payload["profile_sha256"] = self.profile_sha256
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "RegistrationGateProfile":
        expected = {
            "schema_version",
            "profile_id",
            "product",
            "hand",
            "topology_id",
            "topology_sha256",
            "views",
            "profile_sha256",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            actual = set(payload) if isinstance(payload, Mapping) else set()
            raise ValueError(
                "registration profile fields differ from the strict schema; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        raw_views = payload["views"]
        if not isinstance(raw_views, Mapping) or any(
            not isinstance(key, str) for key in raw_views
        ):
            raise TypeError("registration gate views must be an object keyed by view_id")
        return cls(
            schema_version=payload["schema_version"],  # type: ignore[arg-type]
            profile_id=payload["profile_id"],  # type: ignore[arg-type]
            product=payload["product"],  # type: ignore[arg-type]
            hand=payload["hand"],  # type: ignore[arg-type]
            topology_id=payload["topology_id"],  # type: ignore[arg-type]
            topology_sha256=payload["topology_sha256"],  # type: ignore[arg-type]
            views={
                view: RegistrationViewSpec.from_mapping(item)  # type: ignore[arg-type]
                for view, item in raw_views.items()
            },
            profile_sha256=payload["profile_sha256"],  # type: ignore[arg-type]
        )

    def validate_plan(self, plan: CapturePlan) -> None:
        if plan.product != self.product:
            raise ValueError("registration gate product conflicts with capture plan")
        if plan.topology_id != self.topology_id or plan.topology_sha256 != self.topology_sha256:
            raise ValueError("registration gate topology identity conflicts with capture plan")
        expected = set(plan.required_views)
        actual = set(self.views)
        if actual != expected:
            raise ValueError(
                "registration gate view set differs from required topology views; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )


def _load_opencv_dependencies() -> tuple[Any, Any]:
    return importlib.import_module("cv2"), importlib.import_module("numpy")


class OpenCvRegistrationGate:
    """Register every source frame against its exact view-specific reference."""

    name = "registration"

    def __init__(
        self,
        profile: RegistrationGateProfile,
        *,
        asset_root: str | Path,
        dependency_loader: DependencyLoader | None = None,
    ) -> None:
        if not isinstance(profile, RegistrationGateProfile):
            raise TypeError("profile must be a RegistrationGateProfile")
        if isinstance(asset_root, str) and not asset_root.strip():
            raise ValueError("registration asset_root must not be empty")
        root = Path(asset_root)
        self.profile = profile
        self.asset_root = root
        self._dependency_loader = dependency_loader or _load_opencv_dependencies

    def evaluate(
        self,
        request: CaptureRequest,
        plan: CapturePlan,
        frames: Sequence[CaptureFrame],
    ) -> Sequence[CaptureGateResult]:
        self.profile.validate_plan(plan)
        if request.hand != self.profile.hand:
            raise ValueError(
                f"registration gate hand {self.profile.hand!r} conflicts with capture hand "
                f"{request.hand!r}"
            )
        frame_by_view = _require_exact_frames(plan, frames, "registration")
        cv2, np = self._dependency_loader()
        results: list[CaptureGateResult] = []
        for view_id in plan.required_views:
            frame = frame_by_view[view_id]
            spec = self.profile.views[view_id]
            reference_bytes = _read_verified_asset(
                self.asset_root,
                spec.reference.relative_path,
                spec.reference.sha256,
            )
            reference = _decode_gray(
                reference_bytes,
                spec.reference.width,
                spec.reference.height,
                view_id,
                "reference",
                cv2,
                np,
            )
            current = _decode_gray(
                frame.image_bytes,
                frame.width,
                frame.height,
                view_id,
                "capture",
                cv2,
                np,
            )
            if (frame.width, frame.height) != (spec.reference.width, spec.reference.height):
                raise ValueError(
                    f"registration reference dimensions conflict for view {view_id!r}: "
                    f"reference={spec.reference.width}x{spec.reference.height}, "
                    f"capture={frame.width}x{frame.height}"
                )
            results.append(self._evaluate_view(view_id, reference, current, spec, cv2, np))
        return tuple(results)

    def _evaluate_view(
        self,
        view_id: str,
        reference: Any,
        current: Any,
        spec: RegistrationViewSpec,
        cv2: Any,
        np: Any,
    ) -> CaptureGateResult:
        thresholds = spec.thresholds
        horizontal_analysis_scale = 1.0
        vertical_analysis_scale = 1.0
        if thresholds.analysis_scale < 1.0:
            target_width = max(2, round(reference.shape[1] * thresholds.analysis_scale))
            target_height = max(2, round(reference.shape[0] * thresholds.analysis_scale))
            horizontal_analysis_scale = target_width / reference.shape[1]
            vertical_analysis_scale = target_height / reference.shape[0]
            size = (target_width, target_height)
            reference = cv2.resize(reference, size, interpolation=cv2.INTER_AREA)
            current = cv2.resize(current, size, interpolation=cv2.INTER_AREA)
        reference_float = reference.astype(np.float32) / 255.0
        current_float = current.astype(np.float32) / 255.0
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            thresholds.max_iterations,
            thresholds.epsilon,
        )
        try:
            correlation, warp = cv2.findTransformECC(
                reference_float,
                current_float,
                warp,
                cv2.MOTION_AFFINE,
                criteria,
                None,
                thresholds.gaussian_filter_size,
            )
        except cv2.error:
            reason = (
                "ecc_non_convergence; "
                f"profile_sha256={self.profile.profile_sha256}; "
                f"reference_sha256={spec.reference.sha256}"
            )
            return CaptureGateResult("registration", False, reason, view_id)
        matrix = tuple(tuple(float(value) for value in row) for row in warp.tolist())
        values = [float(correlation), *(value for row in matrix for value in row)]
        if any(not math.isfinite(value) for value in values):
            raise RuntimeError(f"registration produced non-finite results for view {view_id!r}")
        a, b, tx_scaled = matrix[0]
        c, d, ty_scaled = matrix[1]
        scale_x = math.hypot(a, c)
        scale_y = math.hypot(b, d)
        if scale_x <= 0.0 or scale_y <= 0.0:
            raise RuntimeError(f"registration produced a singular transform for view {view_id!r}")
        translation_x = tx_scaled / horizontal_analysis_scale
        translation_y = ty_scaled / vertical_analysis_scale
        rotation_deg = math.degrees(math.atan2(c, a))
        scale_deviation = max(abs(scale_x - 1.0), abs(scale_y - 1.0))
        shear = abs((a * b + c * d) / (scale_x * scale_y))
        metrics = {
            "correlation": float(correlation),
            "translation_x_px": translation_x,
            "translation_y_px": translation_y,
            "rotation_deg": rotation_deg,
            "scale_deviation": scale_deviation,
            "shear": shear,
        }
        failures: list[str] = []
        if metrics["correlation"] < thresholds.min_ecc_correlation:
            failures.append(f"correlation below {thresholds.min_ecc_correlation:.6g}")
        if abs(translation_x) > thresholds.max_translation_x_px:
            failures.append(
                f"abs(translation_x_px) above {thresholds.max_translation_x_px:.6g}"
            )
        if abs(translation_y) > thresholds.max_translation_y_px:
            failures.append(
                f"abs(translation_y_px) above {thresholds.max_translation_y_px:.6g}"
            )
        if abs(rotation_deg) > thresholds.max_rotation_deg:
            failures.append(f"abs(rotation_deg) above {thresholds.max_rotation_deg:.6g}")
        if scale_deviation > thresholds.max_scale_deviation:
            failures.append(
                f"scale_deviation above {thresholds.max_scale_deviation:.6g}"
            )
        if shear > thresholds.max_shear:
            failures.append(f"shear above {thresholds.max_shear:.6g}")
        metric_text = ",".join(
            f"{key}={value:.6g}" for key, value in sorted(metrics.items())
        )
        provenance = (
            f"profile_sha256={self.profile.profile_sha256}; "
            f"reference_sha256={spec.reference.sha256}; {metric_text}"
        )
        reason = provenance if not failures else f"{'; '.join(failures)}; {provenance}"
        return CaptureGateResult("registration", not failures, reason, view_id)


def _decode_gray(
    payload: bytes,
    width: int,
    height: int,
    view_id: str,
    source: str,
    cv2: Any,
    np: Any,
) -> Any:
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None or getattr(image, "ndim", 0) != 2:
        raise ValueError(f"cannot decode registration {source} PNG for view {view_id!r}")
    actual_height, actual_width = image.shape
    if (actual_width, actual_height) != (width, height):
        raise ValueError(
            f"registration {source} dimensions conflict for view {view_id!r}: "
            f"decoded={actual_width}x{actual_height}, declared={width}x{height}"
        )
    return image


def _read_verified_asset(root: Path, relative_path: str, expected_sha256: str) -> bytes:
    """Read one private regular file beneath ``root`` without following symlinks."""
    parts = PurePosixPath(relative_path).parts
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    opened_directories: list[int] = []
    file_descriptor: int | None = None
    try:
        current = os.open(root, directory_flags)
        opened_directories.append(current)
        for component in parts[:-1]:
            current = os.open(component, directory_flags, dir_fd=current)
            opened_directories.append(current)
        file_descriptor = os.open(parts[-1], file_flags, dir_fd=current)
        before = os.fstat(file_descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            raise ValueError(
                "registration reference must be a private regular file without group/world "
                f"write access: {relative_path}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError(f"registration reference changed while reading: {relative_path}")
        payload = b"".join(chunks)
    except OSError as error:
        raise ValueError(f"cannot securely read registration reference {relative_path!r}: {error}") from error
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        for descriptor in reversed(opened_directories):
            os.close(descriptor)
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"registration reference hash mismatch for {relative_path!r}: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    return payload
