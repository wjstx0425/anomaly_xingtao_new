"""Strict JSON configuration loading for BMW six-view laboratory experiments."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, NoReturn

from bmw_inspection.lab.contracts import BranchName, ViewId


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        payload[key] = value
    return payload


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(
            resolved.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    except OSError as error:
        raise ValueError(f"cannot read {label} {resolved}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON {label} {resolved}: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _require_fields(payload: Mapping[str, Any], required: set[str], label: str) -> None:
    missing = required - set(payload)
    unknown = set(payload) - required
    if missing:
        raise ValueError(f"{label} missing required fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {', '.join(sorted(unknown))}")


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _integer(value: object, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    parsed = float(value)
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def _probability(value: object, name: str) -> float:
    parsed = _finite(value, name)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return parsed


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    return value


def _view(value: object, name: str) -> ViewId:
    try:
        return ViewId(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} contains unknown view: {value!r}") from error


def _model_path(value: object, base_path: Path, name: str, *, enabled: bool) -> Path | None:
    if value is None:
        if enabled:
            raise ValueError(f"{name} is required when enabled")
        return None
    candidate = Path(_string(value, name))
    resolved = candidate if candidate.is_absolute() else (base_path / candidate).resolve()
    if enabled and not resolved.is_file():
        raise ValueError(f"{name} does not exist: {resolved}")
    return resolved


def _roi(value: object, name: str, width: int, height: int) -> tuple[int, int, int, int]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{name} must be four integer half-open coordinates")
    x1, y1, x2, y2 = (_integer(item, name) for item in value)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"{name} must be within image bounds")
    return x1, y1, x2, y2


@dataclass(frozen=True, slots=True)
class CameraSlot:
    """One deterministic serial-to-view binding for both capture rounds."""

    slot_id: str
    serial: str
    front: ViewId
    back: ViewId


@dataclass(frozen=True, slots=True)
class LabTopology:
    """Validated three-camera topology for a six-view experiment."""

    topology_id: str
    camera_slots: tuple[CameraSlot, ...]

    @property
    def view_ids(self) -> tuple[ViewId, ...]:
        """Canonical view order, independent of capture-slot ordering."""
        return tuple(ViewId)


@dataclass(frozen=True, slots=True)
class CaptureSettings:
    """Fixed camera settings shared by the three laboratory cameras."""

    image_width: int
    image_height: int
    exposure: float
    gain: float
    timeout_ms: int
    warmup_frames: int


@dataclass(frozen=True, slots=True)
class TemplateGroupConfig:
    """The Template model and threshold for one fixed view."""

    view_id: ViewId
    model_path: Path | None
    threshold: float


@dataclass(frozen=True, slots=True)
class TemplateConfig:
    enabled: bool
    groups: Mapping[ViewId, TemplateGroupConfig]

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", MappingProxyType(dict(self.groups)))


@dataclass(frozen=True, slots=True)
class BrightStreakLabConfig:
    config_paths: Mapping[ViewId, Path]

    def __post_init__(self) -> None:
        object.__setattr__(self, "config_paths", MappingProxyType(dict(self.config_paths)))

    @property
    def enabled_views(self) -> tuple[ViewId, ...]:
        """Views whose existing bright-streak rule is enabled for this profile."""
        return tuple(self.config_paths)


@dataclass(frozen=True, slots=True)
class YoloConfig:
    enabled: bool
    checkpoint: Path | None
    class_name: str
    candidate_conf: float
    final_threshold: float


@dataclass(frozen=True, slots=True)
class PatchCoreConfig:
    enabled: bool
    checkpoints: Mapping[ViewId, Path | None]
    thresholds: Mapping[ViewId, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoints", MappingProxyType(dict(self.checkpoints)))
        object.__setattr__(self, "thresholds", MappingProxyType(dict(self.thresholds)))


@dataclass(frozen=True, slots=True)
class LabExperimentConfig:
    """Immutable, strictly validated BMW laboratory experiment profile."""

    path: Path
    experiment_id: str
    topology: LabTopology
    capture: CaptureSettings
    part_rois: Mapping[ViewId, tuple[int, int, int, int]]
    template: TemplateConfig
    bright_streak: BrightStreakLabConfig
    yolo: YoloConfig
    patchcore: PatchCoreConfig
    required_for_ok: frozenset[BranchName]
    result_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "part_rois", MappingProxyType(dict(self.part_rois)))


def _load_topology(path: Path) -> LabTopology:
    payload = _read_json_object(path, "BMW topology")
    _require_fields(payload, {"topology_id", "camera_slots"}, "BMW topology")
    topology_id = _string(payload["topology_id"], "topology_id")
    slots_raw = payload["camera_slots"]
    if not isinstance(slots_raw, list) or len(slots_raw) != 3:
        raise ValueError("camera_slots must contain exactly three camera slots")
    slots: list[CameraSlot] = []
    serials: set[str] = set()
    slot_ids: set[str] = set()
    for index, raw in enumerate(slots_raw):
        slot = _mapping(raw, f"camera_slots[{index}]")
        _require_fields(slot, {"slot_id", "serial", "front", "back"}, f"camera_slots[{index}]")
        slot_id = _string(slot["slot_id"], f"camera_slots[{index}].slot_id")
        serial = _string(slot["serial"], f"camera_slots[{index}].serial")
        if slot_id in slot_ids:
            raise ValueError(f"duplicate camera slot_id: {slot_id}")
        if serial in serials:
            raise ValueError(f"duplicate camera serial: {serial}")
        slot_ids.add(slot_id)
        serials.add(serial)
        slots.append(
            CameraSlot(
                slot_id=slot_id,
                serial=serial,
                front=_view(slot["front"], f"camera_slots[{index}].front"),
                back=_view(slot["back"], f"camera_slots[{index}].back"),
            )
        )
    mapped_views = tuple(view for slot in slots for view in (slot.front, slot.back))
    if set(mapped_views) != set(ViewId) or len(mapped_views) != len(ViewId):
        raise ValueError("camera_slots must map exactly the six required views")
    expected_slots = {
        "center": ("DA9805574", ViewId.FRONT, ViewId.BACK),
        "left": ("DA9625347", ViewId.FRONT_LEFT, ViewId.BACK_LEFT),
        "right": ("DB0968108", ViewId.FRONT_RIGHT, ViewId.BACK_RIGHT),
    }
    actual_slots = {slot.slot_id: (slot.serial, slot.front, slot.back) for slot in slots}
    if topology_id != "bmw-3cam-double-side-v1" or actual_slots != expected_slots:
        raise ValueError("topology must match the fixed BMW fixture topology")
    return LabTopology(topology_id=topology_id, camera_slots=tuple(slots))


def _parse_capture(value: object) -> CaptureSettings:
    payload = _mapping(value, "capture")
    _require_fields(
        payload,
        {"image_width", "image_height", "exposure", "gain", "timeout_ms", "warmup_frames"},
        "capture",
    )
    return CaptureSettings(
        image_width=_integer(payload["image_width"], "capture.image_width", positive=True),
        image_height=_integer(payload["image_height"], "capture.image_height", positive=True),
        exposure=_finite(payload["exposure"], "capture.exposure", minimum=0.0),
        gain=_finite(payload["gain"], "capture.gain", minimum=0.0),
        timeout_ms=_integer(payload["timeout_ms"], "capture.timeout_ms", positive=True),
        warmup_frames=_integer(payload["warmup_frames"], "capture.warmup_frames", positive=True),
    )


def _parse_part_rois(value: object, capture: CaptureSettings) -> Mapping[ViewId, tuple[int, int, int, int]]:
    payload = _mapping(value, "part_rois")
    try:
        supplied_views = {ViewId(key) for key in payload}
    except ValueError as error:
        raise ValueError("part_rois contains unknown view") from error
    if supplied_views != set(ViewId) or len(payload) != len(ViewId):
        raise ValueError("part_rois must contain exactly the six required views")
    return {
        ViewId(key): _roi(raw, f"part_rois.{key}", capture.image_width, capture.image_height)
        for key, raw in payload.items()
    }


def _parse_template(value: object, base_path: Path) -> TemplateConfig:
    payload = _mapping(value, "template")
    _require_fields(payload, {"enabled", "groups"}, "template")
    enabled = _bool(payload["enabled"], "template.enabled")
    groups_raw = _mapping(payload["groups"], "template.groups")
    if not groups_raw:
        raise ValueError("template.groups must not be empty")
    if set(groups_raw) != {view.value for view in ViewId}:
        raise ValueError("template groups must cover exactly the six required views")
    groups: dict[ViewId, TemplateGroupConfig] = {}
    for view in ViewId:
        group = _mapping(groups_raw[view.value], f"template.groups.{view.value}")
        _require_fields(group, {"model_path", "threshold"}, f"template.groups.{view.value}")
        groups[view] = TemplateGroupConfig(
            view_id=view,
            model_path=_model_path(
                group["model_path"],
                base_path,
                f"template.groups.{view.value}.model_path",
                enabled=enabled,
            ),
            threshold=_finite(group["threshold"], f"template.groups.{view.value}.threshold", minimum=0.0),
        )
    return TemplateConfig(enabled=enabled, groups=groups)


def _parse_bright_streak(value: object, base_path: Path) -> BrightStreakLabConfig:
    payload = _mapping(value, "bright_streak")
    _require_fields(payload, {"config_paths"}, "bright_streak")
    paths = _mapping(payload["config_paths"], "bright_streak.config_paths")
    config_paths: dict[ViewId, Path] = {}
    for raw_view, raw_path in paths.items():
        view = _view(raw_view, "bright_streak.config_paths")
        path = Path(_string(raw_path, f"bright_streak.config_paths.{view.value}"))
        config_paths[view] = path if path.is_absolute() else (base_path / path).resolve()
    return BrightStreakLabConfig(config_paths=config_paths)


def _parse_yolo(value: object, base_path: Path) -> YoloConfig:
    payload = _mapping(value, "yolo")
    _require_fields(payload, {"enabled", "checkpoint", "class_name", "candidate_conf", "final_threshold"}, "yolo")
    enabled = _bool(payload["enabled"], "yolo.enabled")
    class_name = _string(payload["class_name"], "yolo.class_name")
    if class_name != "defect":
        raise ValueError("yolo.class_name must be exactly 'defect'")
    candidate_conf = _probability(payload["candidate_conf"], "yolo.candidate_conf")
    final_threshold = _probability(payload["final_threshold"], "yolo.final_threshold")
    if candidate_conf > final_threshold:
        raise ValueError("yolo.candidate_conf must not exceed yolo.final_threshold")
    return YoloConfig(
        enabled=enabled,
        checkpoint=_model_path(payload["checkpoint"], base_path, "yolo.checkpoint", enabled=enabled),
        class_name=class_name,
        candidate_conf=candidate_conf,
        final_threshold=final_threshold,
    )


def _parse_patchcore(value: object, base_path: Path) -> PatchCoreConfig:
    payload = _mapping(value, "patchcore")
    _require_fields(payload, {"enabled", "checkpoints", "thresholds"}, "patchcore")
    enabled = _bool(payload["enabled"], "patchcore.enabled")
    checkpoints_raw = _mapping(payload["checkpoints"], "patchcore.checkpoints")
    thresholds_raw = _mapping(payload["thresholds"], "patchcore.thresholds")
    if set(checkpoints_raw) != {view.value for view in ViewId}:
        raise ValueError("patchcore.checkpoints must contain exactly the six required views")
    if set(thresholds_raw) != {view.value for view in ViewId}:
        raise ValueError("patchcore.thresholds must contain exactly the six required views")
    return PatchCoreConfig(
        enabled=enabled,
        checkpoints={
            view: _model_path(
                checkpoints_raw[view.value],
                base_path,
                f"patchcore.checkpoints.{view.value}",
                enabled=enabled,
            )
            for view in ViewId
        },
        thresholds={
            view: _finite(thresholds_raw[view.value], f"patchcore.thresholds.{view.value}", minimum=0.0)
            for view in ViewId
        },
    )


def _parse_required(value: object) -> frozenset[BranchName]:
    if not isinstance(value, list) or not value:
        raise ValueError("required_for_ok must be a non-empty list")
    branches: list[BranchName] = []
    for raw in value:
        try:
            branches.append(BranchName(raw))
        except (TypeError, ValueError) as error:
            raise ValueError(f"required_for_ok contains unknown branch: {raw!r}") from error
    if len(set(branches)) != len(branches):
        raise ValueError("required_for_ok must not contain duplicate branches")
    return frozenset(branches)


def load_experiment_config(path: Path) -> LabExperimentConfig:
    """Load a strict BMW laboratory experiment profile and its topology file."""
    config_path = Path(path).expanduser().resolve()
    payload = _read_json_object(config_path, "BMW laboratory experiment configuration")
    _require_fields(
        payload,
        {
            "schema_version",
            "experiment_id",
            "topology_path",
            "capture",
            "part_rois",
            "template",
            "bright_streak",
            "yolo",
            "patchcore",
            "required_for_ok",
            "result_root",
        },
        "BMW laboratory experiment configuration",
    )
    if _integer(payload["schema_version"], "schema_version") != 1:
        raise ValueError("schema_version must be 1")
    base_path = config_path.parent
    topology_path = Path(_string(payload["topology_path"], "topology_path"))
    topology = _load_topology(topology_path if topology_path.is_absolute() else base_path / topology_path)
    capture = _parse_capture(payload["capture"])
    result_root_path = Path(_string(payload["result_root"], "result_root"))
    result_root = result_root_path if result_root_path.is_absolute() else (base_path / result_root_path).resolve()
    return LabExperimentConfig(
        path=config_path,
        experiment_id=_string(payload["experiment_id"], "experiment_id"),
        topology=topology,
        capture=capture,
        part_rois=_parse_part_rois(payload["part_rois"], capture),
        template=_parse_template(payload["template"], base_path),
        bright_streak=_parse_bright_streak(payload["bright_streak"], base_path),
        yolo=_parse_yolo(payload["yolo"], base_path),
        patchcore=_parse_patchcore(payload["patchcore"], base_path),
        required_for_ok=_parse_required(payload["required_for_ok"]),
        result_root=result_root,
    )
