"""ROI, recipe, release, and immutable deployment contracts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

from .errors import DeploymentContractError, RecipeValidationError, RoiValidationError
from .evidence import EvidenceBranch, EvidenceLevel, ModelEvidence, TemplateEvidence, TemplateOutcome
from .identity import Hand, PRODUCT, require_non_empty, require_sha256
from .topology import CaptureTopology


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Hash one canonical JSON object without platform-dependent formatting."""
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class RoiReadiness(str, Enum):
    """Explicit ROI readiness; pending is never interpreted as usable."""

    READY = "ready"
    PENDING = "pending"


class AnomalyFamily(str, Enum):
    """Exactly one unsupervised family active in one deployment."""

    PATCHCORE = "patchcore"
    EFFICIENTAD = "efficientad"
    ANOMALYDINO = "anomalydino"


@dataclass(frozen=True, slots=True)
class RoiBox:
    """Half-open pixel ROI ``[x1, y1, x2, y2]``."""

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        """Require integer, positive-area coordinates."""
        values = (self.x1, self.y1, self.x2, self.y2)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            msg = f"ROI coordinates must be integers, got {values!r}"
            raise RoiValidationError(msg)
        if self.x1 < 0 or self.y1 < 0:
            msg = f"ROI origin must be non-negative, got {values!r}"
            raise RoiValidationError(msg)
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            msg = f"ROI must have positive half-open area, got {values!r}"
            raise RoiValidationError(msg)

    @property
    def width(self) -> int:
        """Return crop width."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Return crop height."""
        return self.y2 - self.y1

    def validate_image_bounds(self, width: int, height: int) -> None:
        """Reject coordinates outside the declared source image."""
        if self.x2 > width or self.y2 > height:
            msg = f"ROI {(self.x1, self.y1, self.x2, self.y2)!r} exceeds image {width}x{height}"
            raise RoiValidationError(msg)

    def as_list(self) -> list[int]:
        """Return canonical JSON coordinates."""
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True, slots=True)
class RoiHandConfig:
    """ROI readiness and unique per-view boxes for one hand."""

    hand: Hand
    status: RoiReadiness
    views: Mapping[str, RoiBox]
    reason: str | None = None

    def __post_init__(self) -> None:
        """Freeze view boxes and forbid data hidden behind pending status."""
        try:
            object.__setattr__(self, "hand", Hand.parse(self.hand))
            object.__setattr__(self, "status", RoiReadiness(self.status))
        except ValueError as error:
            raise RoiValidationError(str(error)) from error
        if not isinstance(self.views, Mapping):
            msg = "ROI views must be a mapping"
            raise RoiValidationError(msg)
        frozen: dict[str, RoiBox] = {}
        for view_id, roi in self.views.items():
            try:
                canonical_view = require_non_empty(view_id, "ROI view_id")
            except ValueError as error:
                raise RoiValidationError(str(error)) from error
            if not isinstance(roi, RoiBox):
                msg = f"ROI for {self.hand.value}/{canonical_view} must be a RoiBox"
                raise RoiValidationError(msg)
            if canonical_view in frozen:
                msg = f"duplicate ROI view {self.hand.value}/{canonical_view}"
                raise RoiValidationError(msg)
            frozen[canonical_view] = roi
        if self.status is RoiReadiness.PENDING:
            if frozen:
                msg = f"pending ROI hand {self.hand.value!r} must not contain unapproved view coordinates"
                raise RoiValidationError(msg)
            if self.reason is None or not self.reason.strip():
                msg = f"pending ROI hand {self.hand.value!r} must include a reason"
                raise RoiValidationError(msg)
        elif not frozen:
            msg = f"ready ROI hand {self.hand.value!r} must contain view coordinates"
            raise RoiValidationError(msg)
        object.__setattr__(self, "views", MappingProxyType(frozen))


@dataclass(frozen=True, slots=True)
class RoiConfig:
    """The single authoritative YOLO-derived spatial crop contract."""

    schema_version: int
    roi_config_id: str
    product: str
    topology_id: str
    coordinate_system: str
    source_width: int
    source_height: int
    hands: Mapping[Hand, RoiHandConfig]
    roi_sha256: str

    def __post_init__(self) -> None:
        """Require the hand-aware v2 schema and bounded coordinates."""
        if self.schema_version != 2:
            msg = f"unsupported ROI schema_version {self.schema_version!r}; expected hand-aware version 2"
            raise RoiValidationError(msg)
        if self.product != PRODUCT:
            msg = f"ROI product must be {PRODUCT!r}, got {self.product!r}"
            raise RoiValidationError(msg)
        try:
            object.__setattr__(self, "roi_config_id", require_non_empty(self.roi_config_id, "roi_config_id"))
            object.__setattr__(self, "topology_id", require_non_empty(self.topology_id, "topology_id"))
            object.__setattr__(self, "roi_sha256", require_sha256(self.roi_sha256, "roi_sha256"))
        except ValueError as error:
            raise RoiValidationError(str(error)) from error
        if self.coordinate_system != "pixel_xyxy_half_open":
            msg = "coordinate_system must be 'pixel_xyxy_half_open'"
            raise RoiValidationError(msg)
        for field_name in ("source_width", "source_height"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                msg = f"{field_name} must be a positive integer, got {value!r}"
                raise RoiValidationError(msg)
        if not isinstance(self.hands, Mapping):
            msg = "hands must be a mapping with explicit left and right entries"
            raise RoiValidationError(msg)
        frozen: dict[Hand, RoiHandConfig] = {}
        for hand_key, hand_config in self.hands.items():
            try:
                hand = Hand.parse(hand_key)
            except ValueError as error:
                raise RoiValidationError(str(error)) from error
            if not isinstance(hand_config, RoiHandConfig) or hand_config.hand is not hand:
                msg = f"ROI hands key {hand.value!r} conflicts with its RoiHandConfig"
                raise RoiValidationError(msg)
            if hand in frozen:
                msg = f"duplicate ROI hand {hand.value!r}"
                raise RoiValidationError(msg)
            for roi in hand_config.views.values():
                roi.validate_image_bounds(self.source_width, self.source_height)
            frozen[hand] = hand_config
        if set(frozen) != set(Hand):
            missing = sorted(hand.value for hand in set(Hand) - set(frozen))
            msg = f"ROI config must explicitly declare both hands; missing={missing}"
            raise RoiValidationError(msg)
        object.__setattr__(self, "hands", MappingProxyType(frozen))
        if canonical_sha256(self.as_dict(include_sha256=False)) != self.roi_sha256:
            msg = "roi_sha256 does not match the canonical ROI payload"
            raise RoiValidationError(msg)

    def require_ready(self, hand: Hand, required_views: tuple[str, ...]) -> RoiHandConfig:
        """Return a complete hand ROI or fail closed."""
        canonical_hand = Hand.parse(hand)
        hand_config = self.hands[canonical_hand]
        if hand_config.status is not RoiReadiness.READY:
            msg = f"ROI for hand {canonical_hand.value!r} is pending and cannot enter a deployment contract"
            raise RoiValidationError(msg)
        expected = set(required_views)
        actual = set(hand_config.views)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            msg = f"ROI view set mismatch for {canonical_hand.value}; missing={missing}, extra={extra}"
            raise RoiValidationError(msg)
        return hand_config

    def as_dict(self, *, include_sha256: bool = False) -> dict[str, Any]:
        """Return the single canonical external ROI v2 schema."""
        hands: dict[str, Any] = {}
        for hand in sorted(self.hands, key=lambda item: item.value):
            hand_config = self.hands[hand]
            item: dict[str, Any] = {
                "status": hand_config.status.value,
                "views": {
                    view_id: {"xyxy": roi.as_list()}
                    for view_id, roi in sorted(hand_config.views.items())
                },
            }
            if hand_config.reason is not None:
                item["reason"] = hand_config.reason
            hands[hand.value] = item
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "roi_version": self.roi_config_id,
            "product": self.product,
            "topology_id": self.topology_id,
            "coordinate_system": self.coordinate_system,
            "source_image_size": {"width": self.source_width, "height": self.source_height},
            "hands": hands,
        }
        if include_sha256:
            payload["roi_sha256"] = self.roi_sha256
        return payload


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Content-addressed deployable asset reference."""

    artifact_id: str
    version: str
    sha256: str
    relative_path: str

    def __post_init__(self) -> None:
        """Reject floating identities, invalid digests, and unsafe paths."""
        try:
            object.__setattr__(self, "artifact_id", require_non_empty(self.artifact_id, "artifact_id"))
            object.__setattr__(self, "version", require_non_empty(self.version, "version"))
            object.__setattr__(self, "sha256", require_sha256(self.sha256, "sha256"))
            relative_path = require_non_empty(self.relative_path, "relative_path")
        except ValueError as error:
            raise RecipeValidationError(str(error)) from error
        path = PurePosixPath(relative_path)
        if path.is_absolute() or ".." in path.parts:
            msg = f"artifact relative_path must remain inside a release: {relative_path!r}"
            raise RecipeValidationError(msg)
        object.__setattr__(self, "relative_path", path.as_posix())

    def as_dict(self) -> dict[str, str]:
        """Return canonical JSON representation."""
        return {
            "artifact_id": self.artifact_id,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class TemplateBinding:
    """Template asset bound to exactly one hand/view."""

    hand: Hand
    view_id: str
    asset: ArtifactRef

    def __post_init__(self) -> None:
        """Validate binding identity."""
        try:
            object.__setattr__(self, "hand", Hand.parse(self.hand))
            object.__setattr__(self, "view_id", require_non_empty(self.view_id, "view_id"))
        except ValueError as error:
            raise RecipeValidationError(str(error)) from error
        if not isinstance(self.asset, ArtifactRef):
            msg = "template binding asset must be an ArtifactRef"
            raise RecipeValidationError(msg)


@dataclass(frozen=True, slots=True)
class AnomalyBinding:
    """One anomaly checkpoint bound to one hand/view and one family."""

    hand: Hand
    view_id: str
    family: AnomalyFamily
    asset: ArtifactRef

    def __post_init__(self) -> None:
        """Validate binding identity and family."""
        try:
            object.__setattr__(self, "hand", Hand.parse(self.hand))
            object.__setattr__(self, "view_id", require_non_empty(self.view_id, "view_id"))
            object.__setattr__(self, "family", AnomalyFamily(self.family))
        except ValueError as error:
            raise RecipeValidationError(str(error)) from error
        if not isinstance(self.asset, ArtifactRef):
            msg = "anomaly binding asset must be an ArtifactRef"
            raise RecipeValidationError(msg)


@dataclass(frozen=True, slots=True)
class TemplateThresholdRecord:
    """Calibrated binary template threshold for one hand/view."""

    hand: Hand
    view_id: str
    threshold: float
    template_sha256: str
    calibration_sha256: str

    def __post_init__(self) -> None:
        """Validate finite threshold and asset/calibration identity."""
        try:
            object.__setattr__(self, "hand", Hand.parse(self.hand))
            object.__setattr__(self, "view_id", require_non_empty(self.view_id, "view_id"))
            object.__setattr__(self, "template_sha256", require_sha256(self.template_sha256, "template_sha256"))
            object.__setattr__(
                self,
                "calibration_sha256",
                require_sha256(self.calibration_sha256, "calibration_sha256"),
            )
        except ValueError as error:
            raise RecipeValidationError(str(error)) from error
        if isinstance(self.threshold, bool) or not isinstance(self.threshold, (int, float)) or not math.isfinite(
            self.threshold,
        ):
            msg = f"template threshold must be finite, got {self.threshold!r}"
            raise RecipeValidationError(msg)
        object.__setattr__(self, "threshold", float(self.threshold))

    def classify(self, score: float) -> TemplateOutcome:
        """Apply the single binary template boundary to a risk score."""
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            msg = f"template score must be finite, got {score!r}"
            raise RecipeValidationError(msg)
        return TemplateOutcome.PASS if score < self.threshold else TemplateOutcome.NG_TEMPLATE


@dataclass(frozen=True, slots=True)
class ThresholdRecord:
    """Calibrated CLEAR/GRAY/STRONG thresholds for one second-layer group."""

    hand: Hand
    view_id: str
    branch: EvidenceBranch
    low: float
    high: float
    model_sha256: str
    roi_config_id: str
    calibration_sha256: str

    def __post_init__(self) -> None:
        """Require finite ordered thresholds bound to model and ROI identity."""
        try:
            object.__setattr__(self, "hand", Hand.parse(self.hand))
            object.__setattr__(self, "view_id", require_non_empty(self.view_id, "view_id"))
            object.__setattr__(self, "branch", EvidenceBranch(self.branch))
            object.__setattr__(self, "model_sha256", require_sha256(self.model_sha256, "model_sha256"))
            object.__setattr__(self, "roi_config_id", require_non_empty(self.roi_config_id, "roi_config_id"))
            object.__setattr__(
                self,
                "calibration_sha256",
                require_sha256(self.calibration_sha256, "calibration_sha256"),
            )
        except ValueError as error:
            raise RecipeValidationError(str(error)) from error
        for field_name in ("low", "high"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                msg = f"{field_name} must be finite, got {value!r}"
                raise RecipeValidationError(msg)
            object.__setattr__(self, field_name, float(value))
        if self.low >= self.high:
            msg = f"thresholds require low < high, got low={self.low}, high={self.high}"
            raise RecipeValidationError(msg)

    def classify(self, score: float) -> EvidenceLevel:
        """Apply deterministic low/high evidence semantics."""
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            msg = f"model score must be finite, got {score!r}"
            raise RecipeValidationError(msg)
        if score < self.low:
            return EvidenceLevel.CLEAR
        if score < self.high:
            return EvidenceLevel.GRAY
        return EvidenceLevel.STRONG


@dataclass(frozen=True, slots=True)
class RequiredEvidenceGroup:
    """One mandatory second-layer evidence row."""

    hand: Hand
    view_id: str
    branch: EvidenceBranch

    def __post_init__(self) -> None:
        """Normalize strict group identity."""
        try:
            object.__setattr__(self, "hand", Hand.parse(self.hand))
            object.__setattr__(self, "view_id", require_non_empty(self.view_id, "view_id"))
            object.__setattr__(self, "branch", EvidenceBranch(self.branch))
        except ValueError as error:
            raise RecipeValidationError(str(error)) from error


@dataclass(frozen=True, slots=True)
class InspectionRecipe:
    """Versioned inputs from which one deployment contract is compiled.

    ``recipe_sha256`` is the stable pre-training/pre-calibration profile
    identity.  It intentionally excludes generated asset references and fitted
    thresholds.  A template model embeds this identity in ``model.json`` and
    anomaly checkpoints bind it in metadata, so including their future hashes
    here would create a model self-reference.  Likewise, fitted thresholds
    refer to the calibration artifact that records this identity.  The final
    DeploymentContract hash covers this stable identity plus every concrete
    asset reference and every fitted threshold field.
    """

    schema_version: int
    recipe_id: str
    product: str
    topology_id: str
    roi_config_id: str
    allowed_hands: tuple[Hand, ...]
    anomaly_family: AnomalyFamily
    capture_gate_policy: ArtifactRef
    template_bindings: tuple[TemplateBinding, ...]
    anomaly_bindings: tuple[AnomalyBinding, ...]
    yolo_model: ArtifactRef
    template_thresholds: tuple[TemplateThresholdRecord, ...]
    model_thresholds: tuple[ThresholdRecord, ...]
    fusion_policy_sha256: str
    recipe_sha256: str

    def __post_init__(self) -> None:
        """Validate top-level recipe identity and immutable collection types."""
        if self.schema_version != 1:
            msg = f"unsupported recipe schema_version {self.schema_version!r}; expected 1"
            raise RecipeValidationError(msg)
        if self.product != PRODUCT:
            msg = f"recipe product must be {PRODUCT!r}, got {self.product!r}"
            raise RecipeValidationError(msg)
        try:
            object.__setattr__(self, "recipe_id", require_non_empty(self.recipe_id, "recipe_id"))
            object.__setattr__(self, "topology_id", require_non_empty(self.topology_id, "topology_id"))
            object.__setattr__(self, "roi_config_id", require_non_empty(self.roi_config_id, "roi_config_id"))
            object.__setattr__(self, "anomaly_family", AnomalyFamily(self.anomaly_family))
            object.__setattr__(
                self,
                "fusion_policy_sha256",
                require_sha256(self.fusion_policy_sha256, "fusion_policy_sha256"),
            )
            object.__setattr__(self, "recipe_sha256", require_sha256(self.recipe_sha256, "recipe_sha256"))
        except ValueError as error:
            raise RecipeValidationError(str(error)) from error
        hands = tuple(Hand.parse(hand) for hand in self.allowed_hands)
        if not hands or len(set(hands)) != len(hands):
            msg = "allowed_hands must contain one or two unique hands"
            raise RecipeValidationError(msg)
        object.__setattr__(self, "allowed_hands", hands)
        for field_name, expected_type in (
            ("template_bindings", TemplateBinding),
            ("anomaly_bindings", AnomalyBinding),
            ("template_thresholds", TemplateThresholdRecord),
            ("model_thresholds", ThresholdRecord),
        ):
            values = tuple(getattr(self, field_name))
            if not all(isinstance(value, expected_type) for value in values):
                msg = f"{field_name} contains an invalid record"
                raise RecipeValidationError(msg)
            object.__setattr__(self, field_name, values)
        if not isinstance(self.capture_gate_policy, ArtifactRef):
            msg = "capture_gate_policy must be one content-addressed ArtifactRef"
            raise RecipeValidationError(msg)
        if not isinstance(self.yolo_model, ArtifactRef):
            msg = "yolo_model must be one global ArtifactRef"
            raise RecipeValidationError(msg)
        if canonical_sha256(self.identity_dict()) != self.recipe_sha256:
            msg = "recipe_sha256 does not match the canonical recipe payload"
            raise RecipeValidationError(msg)

    def identity_dict(self) -> dict[str, Any]:
        """Return the stable profile shared by training and final binding."""
        return {
            "schema_version": self.schema_version,
            "recipe_id": self.recipe_id,
            "product": self.product,
            "topology_id": self.topology_id,
            "roi_version": self.roi_config_id,
            "allowed_hands": [hand.value for hand in self.allowed_hands],
            "anomaly_family": self.anomaly_family.value,
            "capture_gate_policy_sha256": self.capture_gate_policy.sha256,
            "fusion_policy_sha256": self.fusion_policy_sha256,
        }

    def as_dict(self, *, include_sha256: bool = False) -> dict[str, Any]:
        """Return the canonical external recipe schema."""
        template_assets: dict[str, dict[str, Any]] = {}
        for binding in self.template_bindings:
            template_assets.setdefault(binding.hand.value, {})[binding.view_id] = binding.asset.as_dict()
        anomaly_models: dict[str, dict[str, Any]] = {}
        for binding in self.anomaly_bindings:
            anomaly_models.setdefault(binding.hand.value, {})[binding.view_id] = {
                "family": binding.family.value,
                "artifact": binding.asset.as_dict(),
            }
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "recipe_id": self.recipe_id,
            "product": self.product,
            "topology_id": self.topology_id,
            "roi_version": self.roi_config_id,
            "allowed_hands": [hand.value for hand in self.allowed_hands],
            "anomaly_family": self.anomaly_family.value,
            "capture_gate_policy": self.capture_gate_policy.as_dict(),
            "template_assets": template_assets,
            "anomaly_models": anomaly_models,
            "yolo_model": self.yolo_model.as_dict(),
            "template_thresholds": [
                {
                    "hand": threshold.hand.value,
                    "view": threshold.view_id,
                    "threshold": threshold.threshold,
                    "template_sha256": threshold.template_sha256,
                    "calibration_sha256": threshold.calibration_sha256,
                }
                for threshold in self.template_thresholds
            ],
            "model_thresholds": [
                {
                    "hand": threshold.hand.value,
                    "view": threshold.view_id,
                    "branch": threshold.branch.value,
                    "low": threshold.low,
                    "high": threshold.high,
                    "model_sha256": threshold.model_sha256,
                    "roi_version": threshold.roi_config_id,
                    "calibration_sha256": threshold.calibration_sha256,
                }
                for threshold in self.model_thresholds
            ],
            "fusion_policy_sha256": self.fusion_policy_sha256,
        }
        if include_sha256:
            payload["recipe_sha256"] = self.recipe_sha256
        return payload


@dataclass(frozen=True, slots=True)
class DeploymentContract:
    """Complete content-addressed runtime contract compiled before model loading."""

    schema_version: int
    product: str
    recipe_id: str
    recipe_sha256: str
    topology: CaptureTopology
    roi: RoiConfig
    allowed_hands: tuple[Hand, ...]
    anomaly_family: AnomalyFamily
    capture_gate_policy: ArtifactRef
    template_bindings: tuple[TemplateBinding, ...]
    anomaly_bindings: tuple[AnomalyBinding, ...]
    yolo_model: ArtifactRef
    template_thresholds: tuple[TemplateThresholdRecord, ...]
    model_thresholds: tuple[ThresholdRecord, ...]
    required_evidence_groups: tuple[RequiredEvidenceGroup, ...]
    fusion_policy_sha256: str
    contract_sha256: str

    def __post_init__(self) -> None:
        """Validate the compiled envelope and its canonical digest."""
        if self.schema_version != 1 or self.product != PRODUCT:
            msg = "deployment contract must be schema_version=1 and product='ZS32'"
            raise DeploymentContractError(msg)
        if not isinstance(self.topology, CaptureTopology) or not isinstance(self.roi, RoiConfig):
            msg = "deployment contract requires validated topology and ROI objects"
            raise DeploymentContractError(msg)
        try:
            object.__setattr__(self, "recipe_id", require_non_empty(self.recipe_id, "recipe_id"))
            object.__setattr__(self, "recipe_sha256", require_sha256(self.recipe_sha256, "recipe_sha256"))
            object.__setattr__(
                self,
                "fusion_policy_sha256",
                require_sha256(self.fusion_policy_sha256, "fusion_policy_sha256"),
            )
            object.__setattr__(self, "anomaly_family", AnomalyFamily(self.anomaly_family))
        except ValueError as error:
            raise DeploymentContractError(str(error)) from error
        try:
            object.__setattr__(self, "allowed_hands", tuple(Hand.parse(hand) for hand in self.allowed_hands))
        except ValueError as error:
            raise DeploymentContractError(str(error)) from error
        for field_name, expected_type in (
            ("template_bindings", TemplateBinding),
            ("anomaly_bindings", AnomalyBinding),
            ("template_thresholds", TemplateThresholdRecord),
            ("model_thresholds", ThresholdRecord),
            ("required_evidence_groups", RequiredEvidenceGroup),
        ):
            values = tuple(getattr(self, field_name))
            if not all(isinstance(value, expected_type) for value in values):
                msg = f"deployment contract {field_name} contains an invalid record"
                raise DeploymentContractError(msg)
            object.__setattr__(self, field_name, values)
        if not isinstance(self.capture_gate_policy, ArtifactRef):
            msg = "deployment contract capture_gate_policy must be an ArtifactRef"
            raise DeploymentContractError(msg)
        if not isinstance(self.yolo_model, ArtifactRef):
            msg = "deployment contract yolo_model must be an ArtifactRef"
            raise DeploymentContractError(msg)
        expected_digest = canonical_sha256(self.as_dict(include_contract_sha256=False))
        if self.contract_sha256 == "":
            object.__setattr__(self, "contract_sha256", expected_digest)
        else:
            try:
                object.__setattr__(self, "contract_sha256", require_sha256(self.contract_sha256, "contract_sha256"))
            except ValueError as error:
                raise DeploymentContractError(str(error)) from error
        if expected_digest != self.contract_sha256:
            msg = "contract_sha256 does not match the canonical deployment contract payload"
            raise DeploymentContractError(msg)

    def groups_for_hand(self, hand: Hand) -> tuple[RequiredEvidenceGroup, ...]:
        """Return the 12/16/20 mandatory groups for one enabled hand."""
        canonical_hand = Hand.parse(hand)
        if canonical_hand not in self.allowed_hands:
            msg = f"hand {canonical_hand.value!r} is not enabled by this contract"
            raise DeploymentContractError(msg)
        return tuple(group for group in self.required_evidence_groups if group.hand is canonical_hand)

    def validate_template_evidence(self, evidence: TemplateEvidence) -> None:
        """Verify template evidence against its release asset and threshold."""
        if not isinstance(evidence, TemplateEvidence):
            msg = "template evidence must be a TemplateEvidence"
            raise DeploymentContractError(msg)
        if evidence.hand not in self.allowed_hands or evidence.view_id not in self.topology.required_views:
            msg = f"template evidence identity is not required: {evidence.hand.value}/{evidence.view_id}"
            raise DeploymentContractError(msg)
        binding = next(
            item
            for item in self.template_bindings
            if item.hand is evidence.hand and item.view_id == evidence.view_id
        )
        threshold = next(
            item
            for item in self.template_thresholds
            if item.hand is evidence.hand and item.view_id == evidence.view_id
        )
        if evidence.template_sha256 != binding.asset.sha256:
            msg = "template evidence uses a model outside the deployment contract"
            raise DeploymentContractError(msg)
        if evidence.threshold_sha256 != threshold.calibration_sha256 or evidence.threshold != threshold.threshold:
            msg = "template evidence uses a threshold outside the deployment contract"
            raise DeploymentContractError(msg)
        if evidence.outcome is not threshold.classify(evidence.score):
            msg = "template evidence outcome contradicts its release threshold"
            raise DeploymentContractError(msg)
        if evidence.roi_config_id != self.roi.roi_config_id:
            msg = "template evidence uses a different ROI version"
            raise DeploymentContractError(msg)

    def validate_model_evidence(self, evidence: ModelEvidence) -> None:
        """Verify one anomaly/YOLO row against its required release group."""
        if not isinstance(evidence, ModelEvidence):
            msg = "model evidence must be a ModelEvidence"
            raise DeploymentContractError(msg)
        group = RequiredEvidenceGroup(evidence.hand, evidence.view_id, evidence.branch)
        if group not in self.required_evidence_groups:
            identity = f"{evidence.hand.value}/{evidence.view_id}/{evidence.branch.value}"
            msg = f"model evidence group is not required: {identity}"
            raise DeploymentContractError(msg)
        threshold = next(
            item
            for item in self.model_thresholds
            if item.hand is evidence.hand and item.view_id == evidence.view_id and item.branch is evidence.branch
        )
        expected_model_sha256 = self.yolo_model.sha256
        if evidence.branch is EvidenceBranch.ANOMALY:
            binding = next(
                item
                for item in self.anomaly_bindings
                if item.hand is evidence.hand and item.view_id == evidence.view_id
            )
            expected_model_sha256 = binding.asset.sha256
            if evidence.model_family != self.anomaly_family.value:
                msg = "anomaly evidence family differs from the selected deployment family"
                raise DeploymentContractError(msg)
        if evidence.model_sha256 != expected_model_sha256:
            msg = "model evidence uses a model outside the deployment contract"
            raise DeploymentContractError(msg)
        if evidence.threshold_sha256 != threshold.calibration_sha256:
            msg = "model evidence uses thresholds outside the deployment contract"
            raise DeploymentContractError(msg)
        if evidence.level is not threshold.classify(evidence.score):
            msg = "model evidence level contradicts its release thresholds"
            raise DeploymentContractError(msg)
        if evidence.roi_config_id != self.roi.roi_config_id:
            msg = "model evidence uses a different ROI version"
            raise DeploymentContractError(msg)

    def as_dict(self, *, include_contract_sha256: bool = True) -> dict[str, Any]:
        """Return canonical, JSON-serializable release data."""
        payload: dict[str, Any] = {
            "allowed_hands": [hand.value for hand in self.allowed_hands],
            "anomaly_bindings": [
                {
                    "asset": binding.asset.as_dict(),
                    "family": binding.family.value,
                    "hand": binding.hand.value,
                    "view_id": binding.view_id,
                }
                for binding in self.anomaly_bindings
            ],
            "anomaly_family": self.anomaly_family.value,
            "capture_gate_policy": self.capture_gate_policy.as_dict(),
            "fusion_policy_sha256": self.fusion_policy_sha256,
            "model_thresholds": [
                {
                    "branch": threshold.branch.value,
                    "calibration_sha256": threshold.calibration_sha256,
                    "hand": threshold.hand.value,
                    "high": threshold.high,
                    "low": threshold.low,
                    "model_sha256": threshold.model_sha256,
                    "roi_version": threshold.roi_config_id,
                    "view_id": threshold.view_id,
                }
                for threshold in self.model_thresholds
            ],
            "product": self.product,
            "recipe_id": self.recipe_id,
            "recipe_sha256": self.recipe_sha256,
            "required_evidence_groups": [
                {"branch": group.branch.value, "hand": group.hand.value, "view_id": group.view_id}
                for group in self.required_evidence_groups
            ],
            "roi": self.roi.as_dict(include_sha256=True),
            "schema_version": self.schema_version,
            "template_bindings": [
                {"asset": binding.asset.as_dict(), "hand": binding.hand.value, "view_id": binding.view_id}
                for binding in self.template_bindings
            ],
            "template_thresholds": [
                {
                    "calibration_sha256": threshold.calibration_sha256,
                    "hand": threshold.hand.value,
                    "template_sha256": threshold.template_sha256,
                    "threshold": threshold.threshold,
                    "view_id": threshold.view_id,
                }
                for threshold in self.template_thresholds
            ],
            "topology": self.topology.as_dict(include_sha256=True),
            "yolo_model": self.yolo_model.as_dict(),
        }
        if include_contract_sha256:
            payload["contract_sha256"] = self.contract_sha256
        return payload


@dataclass(frozen=True, slots=True)
class ReleaseManifest:
    """Minimal immutable release identity checked before runtime model loading.

    The release-root ``checksums.sha256`` covers this manifest, so its own
    digest must not be embedded here (that would create a hash cycle).  A
    publisher may content-address the checksum file in an external pointer.
    """

    schema_version: int
    release_id: str
    product: str
    contract_sha256: str
    created_at: str

    def __post_init__(self) -> None:
        """Validate content-addressed release identity."""
        if self.schema_version != 1 or self.product != PRODUCT:
            msg = "release manifest must be schema_version=1 and product='ZS32'"
            raise DeploymentContractError(msg)
        try:
            object.__setattr__(self, "release_id", require_non_empty(self.release_id, "release_id"))
            object.__setattr__(self, "contract_sha256", require_sha256(self.contract_sha256, "contract_sha256"))
            object.__setattr__(self, "created_at", require_non_empty(self.created_at, "created_at"))
        except ValueError as error:
            raise DeploymentContractError(str(error)) from error

    def as_dict(self) -> dict[str, object]:
        """Return the strict non-cyclic manifest schema."""
        return {
            "schema_version": self.schema_version,
            "release_id": self.release_id,
            "product": self.product,
            "contract_sha256": self.contract_sha256,
            "created_at": self.created_at,
        }
