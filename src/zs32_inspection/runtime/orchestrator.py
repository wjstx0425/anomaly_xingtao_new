"""ZS32 online inspection DAG with explicit fail-closed stage boundaries."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

from zs32_inspection.calibration.scoring import to_domain_model_evidence
from zs32_inspection.capture.errors import CaptureDataIntegrityError
from zs32_inspection.capture.gates import CaptureGateResult
from zs32_inspection.capture.gate_policy import (
    CAPTURE_GATE_EVIDENCE_SCHEMA_VERSION,
    CaptureGatePolicy,
    CaptureGateProvenance,
)
from zs32_inspection.domain.contracts import DeploymentContract, ThresholdRecord
from zs32_inspection.domain.decisions import InspectionDecision, InspectionStatus
from zs32_inspection.domain.evidence import (
    EvidenceBranch,
    EvidenceLevel,
    ModelEvidence,
    TemplateEvidence,
)
from zs32_inspection.domain.identity import CaptureSet, Hand, require_non_empty, require_sha256
from zs32_inspection.fusion.completeness import EvidenceContext
from zs32_inspection.fusion.engine import StrictFusionEngine
from zs32_inspection.fusion.policy import FusionPolicy
from zs32_inspection.models.base import ModelInput, RawModelScore, sha256_file
from zs32_inspection.template.predictor import TemplatePredictor, TemplateScore, decide_template
from zs32_inspection.domain.topology import CaptureTopology

from .audit import system_error_decision

if TYPE_CHECKING:
    from .release_loader import VerifiedDeploymentRelease


@dataclass(frozen=True, slots=True, init=False)
class InspectionRequest:
    """One byte-verified immutable capture submitted for inspection."""

    inspection_id: str
    release_id: str
    capture: CaptureSet
    capture_root: Path
    capture_manifest_path: Path
    capture_manifest_sha256: str
    capture_gates_path: Path
    capture_gates_sha256: str
    capture_publication_root_sha256: str
    capture_gate_results: tuple[CaptureGateResult, ...]
    capture_gate_provenance: CaptureGateProvenance

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("InspectionRequest must be created from a verified capture publication")

    @classmethod
    def from_capture_publication(
        cls,
        *,
        inspection_id: str,
        release_id: str,
        capture_set_root: Path,
        topology: CaptureTopology,
    ) -> InspectionRequest:
        """Load and bind the exact capture manifest, gate evidence, and checksum root."""
        from zs32_inspection.capture.reader import load_capture_bundle

        set_root = Path(capture_set_root).expanduser()
        try:
            bundle = load_capture_bundle(set_root, topology)
        except CaptureDataIntegrityError:
            raise
        except (TypeError, ValueError, UnicodeError) as error:
            raise CaptureDataIntegrityError(
                f"capture publication contains malformed persisted data: {error}"
            ) from error
        set_root = set_root.resolve()
        manifest_path = set_root / "capture_manifest.json"
        gates_path = set_root / "capture_gates.json"
        checksum_bytes = (set_root / "checksums.sha256").read_bytes()
        instance = object.__new__(cls)
        object.__setattr__(instance, "inspection_id", inspection_id)
        object.__setattr__(instance, "release_id", release_id)
        object.__setattr__(instance, "capture", bundle.domain_capture_set)
        object.__setattr__(instance, "capture_root", set_root.parent.parent)
        object.__setattr__(instance, "capture_manifest_path", manifest_path)
        object.__setattr__(instance, "capture_manifest_sha256", sha256_file(manifest_path))
        object.__setattr__(instance, "capture_gates_path", gates_path)
        object.__setattr__(instance, "capture_gates_sha256", sha256_file(gates_path))
        object.__setattr__(
            instance,
            "capture_publication_root_sha256",
            hashlib.sha256(checksum_bytes).hexdigest(),
        )
        object.__setattr__(instance, "capture_gate_results", tuple(bundle.gate_results))
        object.__setattr__(instance, "capture_gate_provenance", bundle.gate_provenance)
        instance.__post_init__()
        return instance

    def __post_init__(self) -> None:
        object.__setattr__(self, "inspection_id", require_non_empty(self.inspection_id, "inspection_id"))
        object.__setattr__(self, "release_id", require_non_empty(self.release_id, "release_id"))
        if not isinstance(self.capture, CaptureSet):
            raise TypeError("inspection request capture must be a domain CaptureSet")
        root = Path(self.capture_root).expanduser()
        if root.is_symlink() or not root.is_dir():
            raise CaptureDataIntegrityError(
                f"capture_root must be a regular directory: {root}"
            )
        object.__setattr__(self, "capture_root", root.resolve())
        manifest_path = Path(self.capture_manifest_path).expanduser()
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise CaptureDataIntegrityError(
                f"capture_manifest_path must be a regular file: {manifest_path}"
            )
        manifest_path = manifest_path.resolve()
        try:
            manifest_path.relative_to(root.resolve())
        except ValueError as error:
            raise CaptureDataIntegrityError(
                "capture_manifest_path must remain inside capture_root"
            ) from error
        object.__setattr__(self, "capture_manifest_path", manifest_path)
        object.__setattr__(
            self,
            "capture_manifest_sha256",
            require_sha256(self.capture_manifest_sha256, "capture_manifest_sha256"),
        )
        if sha256_file(manifest_path) != self.capture_manifest_sha256:
            raise CaptureDataIntegrityError(
                "capture manifest hash does not match the inspection request"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"capture manifest cannot be parsed: {error}") from error
        self._validate_manifest_identity(manifest)
        gates_path = Path(self.capture_gates_path).expanduser()
        if gates_path.is_symlink() or not gates_path.is_file():
            raise CaptureDataIntegrityError(
                f"capture_gates_path must be a regular file: {gates_path}"
            )
        gates_path = gates_path.resolve()
        try:
            gates_path.relative_to(root.resolve())
        except ValueError as error:
            raise CaptureDataIntegrityError(
                "capture_gates_path must remain inside capture_root"
            ) from error
        object.__setattr__(self, "capture_gates_path", gates_path)
        object.__setattr__(
            self,
            "capture_gates_sha256",
            require_sha256(self.capture_gates_sha256, "capture_gates_sha256"),
        )
        if sha256_file(gates_path) != self.capture_gates_sha256:
            raise CaptureDataIntegrityError(
                "capture gate evidence hash does not match the inspection request"
            )
        self._validate_gate_file_identity(gates_path)
        object.__setattr__(
            self,
            "capture_publication_root_sha256",
            require_sha256(
                self.capture_publication_root_sha256,
                "capture_publication_root_sha256",
            ),
        )
        object.__setattr__(self, "capture_gate_results", tuple(self.capture_gate_results))
        if not isinstance(self.capture_gate_provenance, CaptureGateProvenance):
            raise CaptureDataIntegrityError(
                "inspection request requires structured capture gate provenance"
            )

    def _validate_gate_file_identity(self, path: Path) -> None:
        """Bind parsed gate results to the exact persisted capture_gates.json bytes."""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"capture gate evidence cannot be parsed: {error}") from error
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version", "capture_set_id", "gate_provenance", "results"
        }:
            raise CaptureDataIntegrityError("capture gate evidence envelope is malformed")
        if (
            payload.get("schema_version") != CAPTURE_GATE_EVIDENCE_SCHEMA_VERSION
            or payload.get("capture_set_id") != self.capture.capture_set_id
            or not isinstance(payload.get("results"), list)
        ):
            raise CaptureDataIntegrityError("capture gate evidence identity is invalid")
        persisted = []
        for item in payload["results"]:
            if not isinstance(item, dict) or set(item) != {"gate", "passed", "reason", "view_id"}:
                raise CaptureDataIntegrityError(
                    "capture gate evidence contains a malformed result"
                )
            persisted.append(
                CaptureGateResult(
                    gate=item["gate"],
                    passed=item["passed"],
                    reason=item["reason"],
                    view_id=item["view_id"],
                )
            )
        if tuple(persisted) != tuple(self.capture_gate_results):
            raise CaptureDataIntegrityError(
                "capture gate results differ from persisted capture_gates.json"
            )
        raw_provenance = payload["gate_provenance"]
        if not isinstance(raw_provenance, dict):
            raise CaptureDataIntegrityError("capture gate provenance must be an object")
        if CaptureGateProvenance.from_mapping(raw_provenance) != self.capture_gate_provenance:
            raise CaptureDataIntegrityError(
                "capture gate provenance differs from persisted capture_gates.json"
            )

    def _validate_manifest_identity(self, manifest: object) -> None:
        """Prove that the exact manifest describes the supplied CaptureSet."""
        if not isinstance(manifest, dict):
            raise CaptureDataIntegrityError("capture manifest root must be an object")
        expected_keys = {
            "schema_version",
            "capture_session",
            "capture_set",
            "domain_capture_set",
            "images",
            "round_confirmations",
        }
        if set(manifest) != expected_keys or manifest.get("schema_version") != 2:
            raise CaptureDataIntegrityError("capture manifest envelope is malformed")
        domain = manifest.get("domain_capture_set")
        if not isinstance(domain, dict):
            raise CaptureDataIntegrityError(
                "capture manifest domain_capture_set must be an object"
            )
        expected_domain = (
            self.capture.capture_set_id,
            self.capture.capture_session_id,
            self.capture.part.part_instance_id,
            self.capture.part.hand.value,
            self.capture.part.product,
            self.capture.topology_id,
            self.capture.topology_sha256,
        )
        actual_domain = (
            domain.get("capture_set_id"),
            domain.get("capture_session_id"),
            domain.get("part_instance_id"),
            domain.get("hand"),
            domain.get("product"),
            domain.get("topology_id"),
            domain.get("topology_sha256"),
        )
        if actual_domain != expected_domain or manifest.get("capture_session") != self.capture.capture_session_id:
            raise CaptureDataIntegrityError(
                "capture manifest identity differs from CaptureSet"
            )
        rows = manifest.get("images")
        if not isinstance(rows, list):
            raise CaptureDataIntegrityError("capture manifest images must be an array")
        by_view = {
            row.get("view_id"): row
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("view_id"), str)
        }
        if len(by_view) != len(rows) or set(by_view) != set(self.capture.images):
            raise CaptureDataIntegrityError(
                "capture manifest image identities are incomplete or duplicated"
            )
        for view, image in self.capture.images.items():
            row = by_view[view]
            actual = (
                row.get("round_id"),
                row.get("camera_slot_id"),
                row.get("camera_serial"),
                row.get("relative_path"),
                row.get("image_sha256"),
                row.get("width"),
                row.get("height"),
                row.get("status"),
            )
            expected = (
                image.round_id,
                image.camera_slot_id,
                image.camera_serial,
                image.relative_path,
                image.image_sha256,
                image.width,
                image.height,
                "complete",
            )
            if actual != expected:
                raise CaptureDataIntegrityError(
                    f"capture manifest row differs from CaptureSet for view {view}"
                )


class CanonicalCropper(Protocol):
    """Crop every source exactly once using only the authoritative release ROI."""

    def crop_batch(
        self,
        request: InspectionRequest,
        contract: DeploymentContract,
    ) -> Mapping[str, ModelInput]: ...


class RawPredictor(Protocol):
    """Anomaly or global YOLO raw-score batch port."""

    def predict_batch(
        self,
        samples: Sequence[ModelInput],
        *,
        inspection_id: str,
    ) -> Sequence[RawModelScore]: ...


class RuntimePredictorLoader(Protocol):
    """Load each model stage only when the decision DAG reaches that stage."""

    def load_template_predictor(self) -> TemplatePredictor: ...

    def load_anomaly_predictor(self) -> RawPredictor: ...

    def load_yolo_predictor(self) -> RawPredictor: ...


@dataclass(frozen=True, slots=True)
class RuntimeTemplateReference:
    """One release-indexed template reference usable by the online matcher."""

    relative_path: str
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        relative = PurePosixPath(require_non_empty(self.relative_path, "template reference path"))
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("runtime template reference path must stay inside the release")
        path = Path(self.path).expanduser()
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"runtime template reference must be a regular file: {path}")
        object.__setattr__(self, "relative_path", relative.as_posix())
        object.__setattr__(self, "path", path.resolve())
        object.__setattr__(
            self,
            "sha256",
            require_sha256(self.sha256, "runtime_template_reference.sha256"),
        )


@dataclass(frozen=True, slots=True)
class RuntimeTemplateSlotContract:
    """Release-bound reference allowlist and displacement limit for one slot."""

    hand: Hand
    view_id: str
    max_shift: int
    references: tuple[RuntimeTemplateReference, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "hand", Hand.parse(self.hand))
        object.__setattr__(self, "view_id", require_non_empty(self.view_id, "template slot view_id"))
        if (
            isinstance(self.max_shift, bool)
            or not isinstance(self.max_shift, int)
            or self.max_shift < 0
        ):
            raise ValueError("runtime template max_shift must be a non-negative integer")
        references = tuple(self.references)
        if not references or not all(isinstance(item, RuntimeTemplateReference) for item in references):
            raise ValueError("runtime template slot requires release-bound references")
        paths = tuple(item.path for item in references)
        relative_paths = tuple(item.relative_path for item in references)
        if len(set(paths)) != len(paths) or len(set(relative_paths)) != len(relative_paths):
            raise ValueError("runtime template reference paths must be unique within one slot")
        object.__setattr__(self, "references", references)


@dataclass(frozen=True, slots=True, init=False)
class LoadedRuntime:
    """Verified release identity plus stage-lazy predictor construction."""

    release_id: str
    release_root_sha256: str
    contract: DeploymentContract
    fusion_policy: FusionPolicy
    fusion_policy_file_sha256: str
    capture_gate_policy: CaptureGatePolicy
    capture_gate_policy_file_sha256: str
    runtime_environment_receipt_sha256: str
    runtime_environment_file_sha256: str
    template_slot_contracts: Mapping[tuple[Hand, str], RuntimeTemplateSlotContract]
    predictor_loader: RuntimePredictorLoader

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("LoadedRuntime must be created from VerifiedDeploymentRelease")

    @classmethod
    def from_verified_release(
        cls,
        release: VerifiedDeploymentRelease,
        *,
        predictor_loader: RuntimePredictorLoader,
    ) -> LoadedRuntime:
        """Bind already-created predictors to one byte/contract-verified release."""
        instance = object.__new__(cls)
        object.__setattr__(instance, "release_id", release.manifest.release_id)
        object.__setattr__(instance, "release_root_sha256", release.files.root_sha256)
        object.__setattr__(instance, "contract", release.contract)
        object.__setattr__(instance, "fusion_policy", release.fusion_policy)
        object.__setattr__(
            instance,
            "fusion_policy_file_sha256",
            release.fusion_policy_file_sha256,
        )
        object.__setattr__(instance, "capture_gate_policy", release.capture_gate_policy)
        object.__setattr__(
            instance,
            "capture_gate_policy_file_sha256",
            release.capture_gate_policy_file_sha256,
        )
        object.__setattr__(
            instance,
            "runtime_environment_receipt_sha256",
            release.runtime_environment.receipt_sha256,
        )
        object.__setattr__(
            instance,
            "runtime_environment_file_sha256",
            release.runtime_environment_file_sha256,
        )
        template_slot_contracts: dict[tuple[Hand, str], RuntimeTemplateSlotContract] = {}
        for binding in release.contract.template_bindings:
            group = PurePosixPath(binding.asset.relative_path).parent
            metadata = release.files.read_json((group / "metadata.json").as_posix())
            model = release.files.read_json(binding.asset.relative_path)
            preprocessing = model.get("preprocessing")
            raw_references = metadata.get("reference_templates")
            if not isinstance(preprocessing, Mapping) or not isinstance(raw_references, list):
                raise ValueError(
                    f"verified template runtime metadata is malformed: {binding.hand.value}/{binding.view_id}"
                )
            references: list[RuntimeTemplateReference] = []
            for reference in raw_references:
                if not isinstance(reference, Mapping):
                    raise ValueError("verified template reference metadata must be an object")
                relative = reference.get("relative_path")
                digest = reference.get("sha256")
                if not isinstance(relative, str) or not isinstance(digest, str):
                    raise ValueError("verified template reference metadata has invalid path/hash")
                if release.files.checksums.get(relative) != digest:
                    raise ValueError("verified template reference digest differs from metadata")
                references.append(
                    RuntimeTemplateReference(
                        relative_path=relative,
                        path=release.files.path(relative),
                        sha256=digest,
                    )
                )
            key = (binding.hand, binding.view_id)
            if key in template_slot_contracts:
                raise ValueError(f"duplicate runtime template slot contract: {key!r}")
            template_slot_contracts[key] = RuntimeTemplateSlotContract(
                hand=binding.hand,
                view_id=binding.view_id,
                max_shift=preprocessing.get("max_shift"),
                references=tuple(references),
            )
        object.__setattr__(
            instance,
            "template_slot_contracts",
            MappingProxyType(template_slot_contracts),
        )
        object.__setattr__(instance, "predictor_loader", predictor_loader)
        instance.__post_init__()
        return instance

    def __post_init__(self) -> None:
        object.__setattr__(self, "release_id", require_non_empty(self.release_id, "release_id"))
        object.__setattr__(
            self,
            "release_root_sha256",
            require_sha256(self.release_root_sha256, "release_root_sha256"),
        )
        if not isinstance(self.contract, DeploymentContract):
            raise TypeError("loaded runtime requires a DeploymentContract")
        if not isinstance(self.fusion_policy, FusionPolicy):
            raise TypeError("loaded runtime requires a parsed FusionPolicy")
        object.__setattr__(
            self,
            "fusion_policy_file_sha256",
            require_sha256(self.fusion_policy_file_sha256, "fusion_policy_file_sha256"),
        )
        if self.fusion_policy_file_sha256 != self.contract.fusion_policy_sha256:
            raise ValueError("loaded fusion policy bytes differ from deployment contract")
        if not isinstance(self.capture_gate_policy, CaptureGatePolicy):
            raise TypeError("loaded runtime requires a parsed capture gate policy")
        object.__setattr__(
            self,
            "capture_gate_policy_file_sha256",
            require_sha256(
                self.capture_gate_policy_file_sha256,
                "capture_gate_policy_file_sha256",
            ),
        )
        if self.capture_gate_policy_file_sha256 != self.contract.capture_gate_policy.sha256:
            raise ValueError("loaded capture gate policy bytes differ from deployment contract")
        object.__setattr__(
            self,
            "runtime_environment_receipt_sha256",
            require_sha256(
                self.runtime_environment_receipt_sha256,
                "runtime_environment_receipt_sha256",
            ),
        )
        object.__setattr__(
            self,
            "runtime_environment_file_sha256",
            require_sha256(
                self.runtime_environment_file_sha256,
                "runtime_environment_file_sha256",
            ),
        )
        self.capture_gate_policy.validate_topology(
            self.contract.topology,
            allowed_hands=self.contract.allowed_hands,
        )
        if not isinstance(self.template_slot_contracts, Mapping):
            raise TypeError("loaded runtime requires template slot contracts")
        template_slot_contracts = dict(self.template_slot_contracts)
        expected_template_slots = {
            (binding.hand, binding.view_id) for binding in self.contract.template_bindings
        }
        if set(template_slot_contracts) != expected_template_slots:
            raise ValueError("loaded runtime template slot contracts are incomplete or unexpected")
        for key, slot_contract in template_slot_contracts.items():
            if not isinstance(slot_contract, RuntimeTemplateSlotContract) or key != (
                slot_contract.hand,
                slot_contract.view_id,
            ):
                raise TypeError(f"loaded runtime template slot contract is invalid: {key!r}")
        object.__setattr__(
            self,
            "template_slot_contracts",
            MappingProxyType(template_slot_contracts),
        )
        for method in (
            "load_template_predictor",
            "load_anomaly_predictor",
            "load_yolo_predictor",
        ):
            if not callable(getattr(self.predictor_loader, method, None)):
                raise TypeError(f"runtime predictor_loader is missing {method}")

    def load_template_predictor(self) -> TemplatePredictor:
        return self.predictor_loader.load_template_predictor()

    def load_anomaly_predictor(self) -> RawPredictor:
        return self.predictor_loader.load_anomaly_predictor()

    def load_yolo_predictor(self) -> RawPredictor:
        return self.predictor_loader.load_yolo_predictor()


@dataclass(frozen=True, slots=True)
class InspectionRun:
    """Complete in-memory result passed to an atomic inspection sink."""

    request: InspectionRequest
    decision: InspectionDecision
    gate_results: tuple[CaptureGateResult, ...]
    crops: Mapping[str, ModelInput]
    template_scores: tuple[TemplateScore, ...]
    template_evidence: tuple[TemplateEvidence, ...]
    raw_model_scores: tuple[RawModelScore, ...]
    model_evidence: tuple[ModelEvidence, ...]
    system_errors: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.decision.inspection_id != self.request.inspection_id:
            raise ValueError("inspection run decision identity mismatch")
        object.__setattr__(self, "gate_results", tuple(self.gate_results))
        object.__setattr__(self, "crops", MappingProxyType(dict(self.crops)))
        for field in (
            "template_scores",
            "template_evidence",
            "raw_model_scores",
            "model_evidence",
            "system_errors",
        ):
            object.__setattr__(self, field, tuple(getattr(self, field)))


class InspectionSink(Protocol):
    """Atomically persist the full run and return its immutable directory."""

    def publish(self, runtime: LoadedRuntime, run: InspectionRun) -> Path: ...


@dataclass(frozen=True, slots=True)
class InspectionOutcome:
    """Published result; no outcome exists if atomic publication fails."""

    run: InspectionRun
    published_path: Path


@dataclass(frozen=True, slots=True)
class _ModelBranchResult:
    """One isolated second-layer branch result collected by the coordinator."""

    branch: EvidenceBranch
    raw_scores: tuple[RawModelScore, ...]
    calibrated: tuple[ModelEvidence, ...]
    error: str | None
    resource_guard: object | None


class InspectionOrchestrator:
    """Execute release → identity → gates → crop → template → models → fusion."""

    def __init__(
        self,
        *,
        cropper: CanonicalCropper,
        sink: InspectionSink,
        fusion: StrictFusionEngine | None = None,
    ) -> None:
        self._cropper = cropper
        self._expected_fusion = fusion
        self._sink = sink

    def inspect(self, request: InspectionRequest, runtime: LoadedRuntime) -> InspectionOutcome:
        """Run one inspection without converting system faults into part defects."""
        if request.release_id != runtime.release_id:
            return self._finish(
                runtime,
                self._terminal_run(
                    request,
                    InspectionStatus.SYSTEM_ERROR,
                    "release_id_mismatch",
                ),
            )
        contract = runtime.contract
        if self._expected_fusion is not None and self._expected_fusion.policy != runtime.fusion_policy:
            return self._finish(
                runtime,
                self._terminal_run(request, InspectionStatus.SYSTEM_ERROR, "fusion_policy_semantics_mismatch"),
            )
        fusion = StrictFusionEngine(runtime.fusion_policy)
        capture_error = self._capture_error(request, contract)
        if capture_error is not None:
            return self._finish(
                runtime,
                self._terminal_run(request, InspectionStatus.INVALID_CAPTURE, capture_error),
            )
        try:
            expected_gate_provenance = runtime.capture_gate_policy.provenance_for(
                request.capture.part.hand,
                policy_sha256=runtime.capture_gate_policy_file_sha256,
            )
        except Exception as error:
            return self._finish(
                runtime,
                self._terminal_run(
                    request,
                    InspectionStatus.SYSTEM_ERROR,
                    f"release_gate_policy_error:{error}",
                ),
            )
        if request.capture_gate_provenance != expected_gate_provenance:
            return self._finish(
                runtime,
                self._terminal_run(
                    request,
                    InspectionStatus.INVALID_CAPTURE,
                    "capture_gate_policy_mismatch",
                ),
            )

        try:
            gate_results = self._validate_gate_results(
                contract,
                request.capture_gate_results,
            )
        except Exception as error:
            return self._finish(
                runtime,
                self._terminal_run(request, InspectionStatus.SYSTEM_ERROR, f"gate_runtime_error:{error}"),
            )
        failures = tuple(result for result in gate_results if not result.passed)
        if failures:
            reasons = tuple(
                f"retake:{result.gate}:{result.view_id or 'all'}:{result.reason}" for result in failures
            )
            decision = InspectionDecision(
                inspection_id=request.inspection_id,
                evidence_status=None,
                inspection_status=InspectionStatus.RETAKE,
                review_status=None,
                released_status=None,
                reason_codes=reasons,
            )
            return self._finish(runtime, InspectionRun(request, decision, gate_results, {}, (), (), (), (), ()))

        crops: dict[str, ModelInput] = {}
        template_scores: tuple[TemplateScore, ...] = ()
        template_evidence: tuple[TemplateEvidence, ...] = ()
        try:
            crops = dict(self._cropper.crop_batch(request, contract))
            ordered_inputs = self._validate_crops(request, contract, crops)
            inputs_by_view = {item.sample.view_id: item for item in ordered_inputs}
            template_predictor = runtime.load_template_predictor()
            template_scores = tuple(
                template_predictor.score_batch(
                    ordered_inputs,
                    inspection_id=request.inspection_id,
                )
            )
            template_evidence = self._template_evidence(
                request,
                contract,
                template_scores,
                inputs_by_view,
            )
            context = EvidenceContext.from_contract(
                inspection_id=request.inspection_id,
                capture_set_id=request.capture.capture_set_id,
                part_instance_id=request.capture.part.part_instance_id,
                hand=request.capture.part.hand,
                contract=contract,
            )
            template_decision = fusion.fuse_templates(
                context=context,
                evidence=template_evidence,
            )
        except Exception as error:
            decision = system_error_decision(
                request.inspection_id,
                reason_code=f"template_or_crop_runtime_error:{error}",
            )
            run = InspectionRun(
                request,
                decision,
                gate_results,
                crops,
                template_scores,
                template_evidence,
                (),
                (),
                (str(error),),
            )
            return self._finish(runtime, run)

        if template_decision is not None:
            return self._finish(
                runtime,
                InspectionRun(
                    request,
                    template_decision,
                    gate_results,
                    crops,
                    template_scores,
                    template_evidence,
                    (),
                    (),
                    (),
                ),
            )

        branch_specs = (
            (EvidenceBranch.ANOMALY, runtime.load_anomaly_predictor),
            (EvidenceBranch.YOLO, runtime.load_yolo_predictor),
        )
        with ThreadPoolExecutor(
            max_workers=len(branch_specs),
            thread_name_prefix="zs32-second-layer",
        ) as executor:
            futures = {
                branch: executor.submit(
                    self._run_model_branch,
                    request,
                    contract,
                    branch,
                    load_predictor,
                    ordered_inputs,
                    inputs_by_view,
                )
                for branch, load_predictor in branch_specs
            }
            # Both futures are submitted before either result is observed. Collect in
            # contract order so one branch fault never cancels its peer and persisted
            # evidence/error ordering is independent of thread scheduling.
            collected: list[_ModelBranchResult] = []
            for branch, _load_predictor in branch_specs:
                try:
                    result = futures[branch].result()
                except Exception as error:
                    result = _ModelBranchResult(
                        branch=branch,
                        raw_scores=(),
                        calibrated=(),
                        error=f"{branch.value}_runtime_error:{error}",
                        resource_guard=None,
                    )
                collected.append(result)
            branch_results = tuple(collected)

        raw_scores: list[RawModelScore] = []
        calibrated: list[ModelEvidence] = []
        model_errors: list[str] = []
        for result in branch_results:
            raw_scores.extend(sorted(result.raw_scores, key=lambda row: row.view))
            calibrated.extend(sorted(result.calibrated, key=lambda row: row.view_id))
            if result.error is not None:
                model_errors.append(result.error)

        context = EvidenceContext.from_contract(
            inspection_id=request.inspection_id,
            capture_set_id=request.capture.capture_set_id,
            part_instance_id=request.capture.part.part_instance_id,
            hand=request.capture.part.hand,
            contract=contract,
        )
        if model_errors:
            evidence_status = self._partial_strong_status(calibrated, fusion)
            decision = system_error_decision(
                request.inspection_id,
                reason_code=";".join(model_errors),
                evidence_status=evidence_status,
            )
        else:
            try:
                decision = fusion.fuse_models(context=context, evidence=calibrated)
            except Exception as error:
                model_errors.append(f"model_evidence_contract_error:{error}")
                decision = system_error_decision(
                    request.inspection_id,
                    reason_code=model_errors[-1],
                    evidence_status=self._partial_strong_status(calibrated, fusion),
                )
        return self._finish(
            runtime,
            InspectionRun(
                request,
                decision,
                gate_results,
                crops,
                template_scores,
                template_evidence,
                tuple(raw_scores),
                tuple(calibrated),
                tuple(model_errors),
            ),
        )

    def _run_model_branch(
        self,
        request: InspectionRequest,
        contract: DeploymentContract,
        branch: EvidenceBranch,
        load_predictor: Callable[[], RawPredictor],
        ordered_inputs: tuple[ModelInput, ...],
        inputs_by_view: Mapping[str, ModelInput],
    ) -> _ModelBranchResult:
        """Run one model branch without affecting or cancelling its peer branch."""
        try:
            predictor = load_predictor()
            raw_scores = tuple(
                predictor.predict_batch(
                    ordered_inputs,
                    inspection_id=request.inspection_id,
                )
            )
        except Exception as error:
            return _ModelBranchResult(
                branch=branch,
                raw_scores=(),
                calibrated=(),
                error=f"{branch.value}_runtime_error:{error}",
                resource_guard=None,
            )

        try:
            calibrated = self._calibrate_branch(
                request,
                contract,
                branch,
                raw_scores,
                inputs_by_view,
            )
        except Exception as error:
            return _ModelBranchResult(
                branch=branch,
                raw_scores=raw_scores,
                calibrated=(),
                error=f"{branch.value}_runtime_error:{error}",
                resource_guard=predictor,
            )
        return _ModelBranchResult(
            branch=branch,
            raw_scores=raw_scores,
            calibrated=calibrated,
            error=None,
            resource_guard=predictor,
        )

    def _finish(self, runtime: LoadedRuntime, run: InspectionRun) -> InspectionOutcome:
        published = self._sink.publish(runtime, run)
        return InspectionOutcome(run=run, published_path=published)

    @staticmethod
    def _terminal_run(
        request: InspectionRequest,
        status: InspectionStatus,
        reason: str,
    ) -> InspectionRun:
        if status is InspectionStatus.SYSTEM_ERROR:
            decision = system_error_decision(request.inspection_id, reason_code=reason)
            errors = (reason,)
        else:
            decision = InspectionDecision(
                request.inspection_id,
                None,
                status,
                None,
                None,
                (reason,),
            )
            errors = ()
        return InspectionRun(request, decision, (), {}, (), (), (), (), errors)

    @staticmethod
    def _capture_error(request: InspectionRequest, contract: DeploymentContract) -> str | None:
        capture = request.capture
        if capture.part.product != contract.product:
            return "capture_product_mismatch"
        if capture.part.hand not in contract.allowed_hands:
            return "capture_hand_not_enabled"
        if capture.topology_id != contract.topology.topology_id:
            return "capture_topology_id_mismatch"
        if capture.topology_sha256 != contract.topology.topology_sha256:
            return "capture_topology_digest_mismatch"
        try:
            capture.validate_required_views(contract.topology.required_views)
        except ValueError as error:
            return f"capture_view_identity_error:{error}"
        for view_id, image in capture.images.items():
            if (image.width, image.height) != (
                contract.roi.source_width,
                contract.roi.source_height,
            ):
                return f"capture_source_dimensions_mismatch:{view_id}"
            expected_round, expected_slot, expected_serial = contract.topology.binding_for_view(view_id)
            if (image.round_id, image.camera_slot_id, image.camera_serial) != (
                expected_round,
                expected_slot,
                expected_serial,
            ):
                return f"capture_camera_identity_mismatch:{view_id}"
            path = request.capture_root / image.relative_path
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(request.capture_root)
            except (OSError, ValueError):
                return f"capture_source_path_invalid:{view_id}"
            try:
                invalid_file = path.is_symlink() or not path.is_file() or sha256_file(path) != image.image_sha256
            except OSError:
                invalid_file = True
            if invalid_file:
                return f"capture_source_hash_mismatch:{view_id}"
        return None

    @staticmethod
    def _validate_gate_results(
        contract: DeploymentContract,
        results: Sequence[CaptureGateResult],
    ) -> tuple[CaptureGateResult, ...]:
        """Require one quality and one registration result for every view."""
        indexed: dict[tuple[str, str], CaptureGateResult] = {}
        for result in results:
            if not isinstance(result, CaptureGateResult):
                raise ValueError(f"inspection gate returned invalid type: {type(result).__name__}")
            if result.view_id is None:
                raise ValueError(f"{result.gate} gate result must identify one required view")
            key = (result.gate, result.view_id)
            if key in indexed:
                raise ValueError(f"duplicate gate result: {key}")
            indexed[key] = result
        expected = {
            (gate, view)
            for gate in ("quality", "registration")
            for view in contract.topology.required_views
        }
        if set(indexed) != expected:
            missing = sorted(expected - set(indexed))
            extra = sorted(set(indexed) - expected)
            raise ValueError(f"gate result group mismatch; missing={missing}, extra={extra}")
        return tuple(
            indexed[(gate, view)]
            for gate in ("quality", "registration")
            for view in contract.topology.required_views
        )

    @staticmethod
    def _validate_crops(
        request: InspectionRequest,
        contract: DeploymentContract,
        crops: Mapping[str, ModelInput],
    ) -> tuple[ModelInput, ...]:
        expected = set(contract.topology.required_views)
        if set(crops) != expected:
            raise ValueError(
                f"crop view mismatch; missing={sorted(expected - set(crops))}, extra={sorted(set(crops) - expected)}"
            )
        ordered: list[ModelInput] = []
        for view_id in contract.topology.required_views:
            model_input = crops[view_id]
            sample = model_input.sample
            source = request.capture.images[view_id]
            roi = contract.roi.hands[request.capture.part.hand].views[view_id]
            if (
                sample.part != request.capture.part
                or sample.capture_set_id != request.capture.capture_set_id
                or sample.view_id != view_id
                or sample.source_sha256 != source.image_sha256
                or sample.roi_config_id != contract.roi.roi_config_id
                or (sample.crop_width, sample.crop_height) != (roi.width, roi.height)
                or model_input.roi_digest != contract.roi.roi_sha256
            ):
                raise ValueError(f"canonical crop identity mismatch for view {view_id}")
            path = model_input.crop_path
            if path.is_symlink() or not path.is_file() or sha256_file(path) != sample.crop_sha256:
                raise ValueError(f"canonical crop file/hash mismatch for view {view_id}")
            ordered.append(model_input)
        return tuple(ordered)

    @staticmethod
    def _template_evidence(
        request: InspectionRequest,
        contract: DeploymentContract,
        scores: Sequence[TemplateScore],
        inputs: Mapping[str, ModelInput],
    ) -> tuple[TemplateEvidence, ...]:
        indexed: dict[str, TemplateScore] = {}
        for score in scores:
            if score.view in indexed:
                raise ValueError(f"duplicate template score for view {score.view}")
            indexed[score.view] = score
        if set(indexed) != set(contract.topology.required_views):
            raise ValueError("template score views are incomplete or unexpected")
        for view, score in indexed.items():
            model_input = inputs[view]
            sample = model_input.sample
            if (
                score.inspection_id != request.inspection_id
                or score.capture_set_id != request.capture.capture_set_id
                or score.part_instance_id != request.capture.part.part_instance_id
                or score.hand != request.capture.part.hand.value
                or score.roi_config_id != contract.roi.roi_config_id
                or score.roi_digest != contract.roi.roi_sha256
                or score.source_sha256 != sample.source_sha256
                or score.crop_sha256 != sample.crop_sha256
            ):
                raise ValueError(f"template score identity/hash mismatch for view {view}")
        thresholds = {
            (item.hand, item.view_id): item for item in contract.template_thresholds
        }
        output = tuple(
            decide_template(indexed[view], thresholds[(request.capture.part.hand, view)])
            for view in contract.topology.required_views
        )
        for evidence in output:
            contract.validate_template_evidence(evidence)
        return output

    @staticmethod
    def _calibrate_branch(
        request: InspectionRequest,
        contract: DeploymentContract,
        branch: EvidenceBranch,
        rows: Sequence[RawModelScore],
        inputs: Mapping[str, ModelInput],
    ) -> tuple[ModelEvidence, ...]:
        indexed: dict[str, RawModelScore] = {}
        for row in rows:
            if row.branch != branch.value:
                raise ValueError(f"{branch.value} predictor emitted branch {row.branch!r}")
            if row.view in indexed:
                raise ValueError(f"duplicate {branch.value} score for view {row.view}")
            indexed[row.view] = row
        if set(indexed) != set(contract.topology.required_views):
            raise ValueError(f"{branch.value} score views are incomplete or unexpected")
        for view, row in indexed.items():
            model_input = inputs[view]
            sample = model_input.sample
            if (
                row.inspection_id != request.inspection_id
                or row.capture_set_id != request.capture.capture_set_id
                or row.part_instance_id != request.capture.part.part_instance_id
                or row.hand != request.capture.part.hand.value
                or row.roi_config_id != contract.roi.roi_config_id
                or row.roi_digest != contract.roi.roi_sha256
                or row.source_sha256 != sample.source_sha256
                or row.crop_sha256 != sample.crop_sha256
            ):
                raise ValueError(f"{branch.value} score identity/hash mismatch for view {view}")
        threshold_by_view: dict[str, ThresholdRecord] = {
            item.view_id: item
            for item in contract.model_thresholds
            if item.hand is request.capture.part.hand and item.branch is branch
        }
        output = tuple(
            to_domain_model_evidence(
                indexed[view],
                threshold=threshold_by_view[view],
            )
            for view in contract.topology.required_views
        )
        for evidence in output:
            contract.validate_model_evidence(evidence)
        return output

    @staticmethod
    def _partial_strong_status(
        evidence: Sequence[ModelEvidence],
        fusion: StrictFusionEngine,
    ) -> InspectionStatus | None:
        branches = {row.branch for row in evidence if row.level is EvidenceLevel.STRONG}
        for branch in fusion.policy.strong_priority:
            if branch in branches:
                return (
                    InspectionStatus.NG_ANOMALY
                    if branch is EvidenceBranch.ANOMALY
                    else InspectionStatus.NG_YOLO
                )
        return None
