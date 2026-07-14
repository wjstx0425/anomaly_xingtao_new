"""Linux-only strict short-circuit tests for the online inspection DAG."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import replace
from pathlib import Path
from threading import Barrier

import pytest
import zs32_inspection.runtime.inspection_sink as inspection_sink_module

from zs32_inspection.capture.gates import CaptureGateResult
from zs32_inspection.capture.gate_policy import (
    CaptureGatePolicy,
    HandCaptureGatePolicy,
)
from zs32_inspection.domain.contracts import ArtifactRef
from zs32_inspection.domain.decisions import InspectionDecision, InspectionStatus
from zs32_inspection.domain.evidence import EvidenceBranch, EvidenceLevel
from zs32_inspection.domain.identity import (
    CaptureSet,
    Hand,
    PartIdentity,
    RoiSample,
    ViewImage,
)
from zs32_inspection.fusion.policy import DEFAULT_POLICY
from zs32_inspection.models.base import ModelInput, RawModelScore
from zs32_inspection.runtime.inspection_sink import (
    FilesystemInspectionSink,
    _validate_template_score_references,
    validate_inspection_run_for_publication,
)
from zs32_inspection.runtime.orchestrator import (
    InspectionOrchestrator,
    InspectionRequest,
    RuntimeTemplateReference,
    RuntimeTemplateSlotContract,
)
from zs32_inspection.runtime.publisher import PublicationError
from zs32_inspection.template.predictor import TemplateScore


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="ZS32 verification is Linux-only")


class RecordingSink:
    """Keep the complete run in memory; publication semantics are tested elsewhere."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.runs: list[object] = []

    def publish(self, _runtime: object, run: object) -> Path:
        self.runs.append(run)
        return self.output


class RecordingCropper:
    """Create contract-bound crop files while exposing whether cropping ran."""

    def __init__(self, root: Path, events: list[str]) -> None:
        self.root = root
        self.events = events
        self.calls = 0

    def crop_batch(self, request: InspectionRequest, contract: object) -> dict[str, ModelInput]:
        self.calls += 1
        self.events.append("crop")
        outputs: dict[str, ModelInput] = {}
        for view in contract.topology.required_views:
            payload = f"crop:{request.inspection_id}:{view}".encode()
            path = self.root / f"{view}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            source = request.capture.images[view]
            roi = contract.roi.hands[request.capture.part.hand].views[view]
            sample = RoiSample(
                part=request.capture.part,
                capture_set_id=request.capture.capture_set_id,
                view_id=view,
                source_sha256=source.image_sha256,
                crop_sha256=hashlib.sha256(payload).hexdigest(),
                roi_config_id=contract.roi.roi_config_id,
                crop_width=roi.width,
                crop_height=roi.height,
            )
            outputs[view] = ModelInput(sample, path, contract.roi.roi_sha256)
        return outputs


class RecordingTemplatePredictor:
    def __init__(self, runtime: "RecordingRuntime") -> None:
        self.runtime = runtime

    def score_batch(
        self,
        samples: object,
        *,
        inspection_id: str,
    ) -> tuple[TemplateScore, ...]:
        self.runtime.events.append("predict:template")
        rows: list[TemplateScore] = []
        for index, model_input in enumerate(samples):
            sample = model_input.sample
            binding = next(
                item
                for item in self.runtime.contract.template_bindings
                if item.hand is sample.part.hand and item.view_id == sample.view_id
            )
            risk = self.runtime.template_risk if index == 0 else 0.1
            rows.append(
                TemplateScore(
                    inspection_id=inspection_id,
                    part_instance_id=sample.part.part_instance_id,
                    capture_set_id=sample.capture_set_id,
                    hand=sample.part.hand.value,
                    view=sample.view_id,
                    risk_score=risk,
                    similarity=1.0 - risk,
                    model_digest=binding.asset.sha256,
                    roi_config_id=sample.roi_config_id,
                    roi_digest=model_input.roi_digest,
                    source_sha256=sample.source_sha256,
                    crop_sha256=sample.crop_sha256,
                    best_template_path=Path("unused-reference.png"),
                    best_template_sha256="9" * 64,
                    offset_xy=(0, 0),
                )
            )
        return tuple(rows)


class RecordingRawPredictor:
    def __init__(self, runtime: "RecordingRuntime", branch: EvidenceBranch, score: float) -> None:
        self.runtime = runtime
        self.branch = branch
        self.score = score

    def predict_batch(
        self,
        samples: object,
        *,
        inspection_id: str,
    ) -> tuple[RawModelScore, ...]:
        self.runtime.events.append(f"predict:{self.branch.value}")
        if self.runtime.second_layer_barrier is not None:
            self.runtime.second_layer_barrier.wait(timeout=5)
        predict_error = (
            self.runtime.anomaly_predict_error
            if self.branch is EvidenceBranch.ANOMALY
            else self.runtime.yolo_predict_error
        )
        if predict_error is not None:
            raise predict_error
        rows: list[RawModelScore] = []
        for model_input in samples:
            sample = model_input.sample
            if self.branch is EvidenceBranch.ANOMALY:
                binding = next(
                    item
                    for item in self.runtime.contract.anomaly_bindings
                    if item.hand is sample.part.hand and item.view_id == sample.view_id
                )
                digest = binding.asset.sha256
                family = self.runtime.contract.anomaly_family.value
                detections = ()
                heatmap_payload = (
                    f"heatmap:{inspection_id}:{sample.part.hand.value}:{sample.view_id}"
                ).encode()
                heatmap_path = model_input.crop_path.parent / (
                    f"{sample.view_id}.{hashlib.sha256(inspection_id.encode()).hexdigest()[:12]}.heatmap.png"
                )
                heatmap_path.write_bytes(heatmap_payload)
                heatmap_sha256 = hashlib.sha256(heatmap_payload).hexdigest()
                overlay_payload = (
                    f"overlay:{inspection_id}:{sample.part.hand.value}:{sample.view_id}"
                ).encode()
                overlay_path = model_input.crop_path.parent / (
                    f"{sample.view_id}.{hashlib.sha256(inspection_id.encode()).hexdigest()[:12]}.overlay.png"
                )
                overlay_path.write_bytes(overlay_payload)
                overlay_sha256 = hashlib.sha256(overlay_payload).hexdigest()
            else:
                digest = self.runtime.contract.yolo_model.sha256
                family = "yolo"
                detections = (
                    {
                        "class_id": 0,
                        "class_name": "defect",
                        "confidence": self.score,
                        "xyxy_norm": [0.1, 0.1, 0.2, 0.2],
                        "area_ratio": 0.01,
                        "clipped": False,
                    },
                ) if self.score else ()
                heatmap_path = None
                heatmap_sha256 = None
                overlay_path = None
                overlay_sha256 = None
            rows.append(
                RawModelScore(
                    inspection_id=inspection_id,
                    part_instance_id=sample.part.part_instance_id,
                    capture_set_id=sample.capture_set_id,
                    hand=sample.part.hand.value,
                    view=sample.view_id,
                    branch=self.branch.value,
                    score=self.score,
                    model_family=family,
                    model_digest=digest,
                    roi_config_id=sample.roi_config_id,
                    roi_digest=model_input.roi_digest,
                    source_sha256=sample.source_sha256,
                    crop_sha256=sample.crop_sha256,
                    heatmap_path=heatmap_path,
                    heatmap_sha256=heatmap_sha256,
                    overlay_path=overlay_path,
                    overlay_sha256=overlay_sha256,
                    detections=detections,
                )
            )
        result = tuple(rows)
        return tuple(reversed(result)) if self.runtime.reverse_model_rows else result


class DeletingRawPredictor(RecordingRawPredictor):
    """Model predictor whose destructor reproduces temporary visual cleanup."""

    def __init__(self, runtime: "RecordingRuntime", branch: EvidenceBranch, score: float) -> None:
        super().__init__(runtime, branch, score)
        self.visuals: list[Path] = []

    def predict_batch(
        self,
        samples: object,
        *,
        inspection_id: str,
    ) -> tuple[RawModelScore, ...]:
        rows = super().predict_batch(samples, inspection_id=inspection_id)
        self.visuals = [
            path
            for row in rows
            for path in (row.heatmap_path, row.overlay_path)
            if path is not None
        ]
        return rows

    def __del__(self) -> None:
        for path in self.visuals:
            path.unlink(missing_ok=True)


class VisualPresenceSink:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.saw_complete_visuals = False

    def publish(self, _runtime: object, run: object) -> Path:
        anomaly = [
            row
            for row in run.raw_model_scores
            if row.branch == EvidenceBranch.ANOMALY.value
        ]
        self.saw_complete_visuals = bool(anomaly) and all(
            row.heatmap_path is not None
            and row.heatmap_path.is_file()
            and row.overlay_path is not None
            and row.overlay_path.is_file()
            for row in anomaly
        )
        if not self.saw_complete_visuals:
            raise AssertionError("anomaly visuals were cleaned before publication")
        return self.output


class RecordingRuntime:
    """Expose lazy load events without constructing any third-party backend."""

    release_id = "release-runtime-order-v1"
    fusion_policy = DEFAULT_POLICY

    def __init__(
        self,
        contract: object,
        events: list[str],
        *,
        template_risk: float = 0.1,
        anomaly_score: float = 0.1,
        anomaly_load_error: Exception | None = None,
        anomaly_predict_error: Exception | None = None,
        yolo_score: float = 0.1,
        yolo_load_error: Exception | None = None,
        yolo_predict_error: Exception | None = None,
        second_layer_barrier: Barrier | None = None,
        reverse_model_rows: bool = False,
    ) -> None:
        self.contract = contract
        profile = HandCaptureGatePolicy(
            hand=Hand.RIGHT,
            quality_profile=ArtifactRef("quality", "v1", "2" * 64, "capture/gates/right/quality.json"),
            registration_profile=ArtifactRef("registration", "v1", "3" * 64, "capture/gates/right/registration.json"),
            registration_references={
                view: ArtifactRef(
                    f"reference-{view}", "v1", "4" * 64,
                    f"capture/gates/right/references/{view}.png",
                )
                for view in contract.topology.required_views
            },
        )
        self.capture_gate_policy = CaptureGatePolicy(
            2,
            "test-policy",
            "ZS32",
            contract.topology.topology_id,
            contract.topology.topology_sha256,
            ArtifactRef(
                "hikvision-acquisition",
                "v1",
                "5" * 64,
                "capture/acquisition/hikvision.json",
            ),
            {Hand.RIGHT: profile},
        )
        self.capture_gate_policy_file_sha256 = contract.capture_gate_policy.sha256
        self.runtime_environment_receipt_sha256 = "6" * 64
        self.runtime_environment_file_sha256 = "7" * 64
        self.events = events
        self.template_risk = template_risk
        self.anomaly_score = anomaly_score
        self.anomaly_load_error = anomaly_load_error
        self.anomaly_predict_error = anomaly_predict_error
        self.yolo_score = yolo_score
        self.yolo_load_error = yolo_load_error
        self.yolo_predict_error = yolo_predict_error
        self.second_layer_barrier = second_layer_barrier
        self.reverse_model_rows = reverse_model_rows

    def load_template_predictor(self) -> RecordingTemplatePredictor:
        self.events.append("load:template")
        return RecordingTemplatePredictor(self)

    def load_anomaly_predictor(self) -> RecordingRawPredictor:
        self.events.append("load:anomaly")
        if self.anomaly_load_error is not None:
            raise self.anomaly_load_error
        return RecordingRawPredictor(self, EvidenceBranch.ANOMALY, self.anomaly_score)

    def load_yolo_predictor(self) -> RecordingRawPredictor:
        self.events.append("load:yolo")
        if self.yolo_load_error is not None:
            raise self.yolo_load_error
        return RecordingRawPredictor(self, EvidenceBranch.YOLO, self.yolo_score)


def _gate_results(contract: object, *, failed_gate: str | None = None) -> tuple[CaptureGateResult, ...]:
    return tuple(
        CaptureGateResult(
            gate=gate,
            passed=gate != failed_gate,
            reason="blur" if gate == failed_gate else "",
            view_id=view,
        )
        for gate in ("quality", "registration")
        for view in contract.topology.required_views
    )


def _request(
    root: Path,
    contract: object,
    *,
    hand: Hand = Hand.RIGHT,
    failed_gate: str | None = None,
) -> InspectionRequest:
    images: dict[str, ViewImage] = {}
    for view in contract.topology.required_views:
        payload = f"source:{hand.value}:{view}".encode()
        relative = f"images/{view}.png"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        round_id, slot_id, serial = contract.topology.binding_for_view(view)
        images[view] = ViewImage(
            view_id=view,
            round_id=round_id,
            camera_slot_id=slot_id,
            camera_serial=serial,
            relative_path=relative,
            image_sha256=hashlib.sha256(payload).hexdigest(),
            width=contract.roi.source_width,
            height=contract.roi.source_height,
        )
    request = object.__new__(InspectionRequest)
    object.__setattr__(request, "inspection_id", f"inspection-order-{hand.value}")
    object.__setattr__(request, "release_id", RecordingRuntime.release_id)
    object.__setattr__(
        request,
        "capture",
        CaptureSet(
            capture_set_id=f"capture-order-{hand.value}",
            capture_session_id="session-order-v1",
            part=PartIdentity(f"part-order-{hand.value}", hand),
            topology_id=contract.topology.topology_id,
            topology_sha256=contract.topology.topology_sha256,
            images=images,
        ),
    )
    object.__setattr__(request, "capture_root", root)
    object.__setattr__(request, "capture_gate_results", _gate_results(contract, failed_gate=failed_gate))
    runtime_policy = RecordingRuntime(contract, []).capture_gate_policy
    object.__setattr__(
        request,
        "capture_gate_provenance",
        runtime_policy.provenance_for(
            Hand.RIGHT,
            policy_sha256=contract.capture_gate_policy.sha256,
        ),
    )
    return request


def _orchestrator(tmp_path: Path, events: list[str]) -> tuple[InspectionOrchestrator, RecordingCropper]:
    cropper = RecordingCropper(tmp_path / "crops", events)
    return (
        InspectionOrchestrator(
            cropper=cropper,
            sink=RecordingSink(tmp_path / "published"),
        ),
        cropper,
    )


def _bind_release_template_references(
    root: Path,
    runtime: RecordingRuntime,
    run: object,
    *,
    max_shift: int = 2,
):
    """Replace synthetic matcher references with an exact release-style allowlist."""
    contracts = {}
    scores = []
    for score in run.template_scores:
        payload = f"reference:{score.hand}:{score.view}".encode()
        path = root / score.hand / score.view / "templates" / "reference-001.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        reference = RuntimeTemplateReference(
            relative_path=f"template/{score.hand}/{score.view}/templates/reference-001.png",
            path=path,
            sha256=digest,
        )
        contracts[(Hand.parse(score.hand), score.view)] = RuntimeTemplateSlotContract(
            hand=Hand.parse(score.hand),
            view_id=score.view,
            max_shift=max_shift,
            references=(reference,),
        )
        scores.append(
            replace(
                score,
                best_template_path=path,
                best_template_sha256=digest,
                offset_xy=(0, 0),
            )
        )
    runtime.template_slot_contracts = contracts
    return replace(run, template_scores=tuple(scores))


def _assert_second_layer_event_contract(
    events: list[str],
    *,
    expected: set[str],
) -> None:
    """Assert DAG partial order without depending on thread scheduling order."""
    assert events[:3] == ["crop", "load:template", "predict:template"]
    second_layer = events[3:]
    assert len(second_layer) == len(expected)
    assert set(second_layer) == expected
    for branch in ("anomaly", "yolo"):
        load = f"load:{branch}"
        predict = f"predict:{branch}"
        if predict in expected:
            assert second_layer.index(load) < second_layer.index(predict)


def test_capture_identity_failure_never_crops_or_loads_models(
    tmp_path: Path,
    compiled_contract_factory: object,
) -> None:
    contract = compiled_contract_factory(hands=("right",))
    events: list[str] = []
    orchestrator, cropper = _orchestrator(tmp_path, events)
    runtime = RecordingRuntime(contract, events)

    outcome = orchestrator.inspect(
        _request(tmp_path / "capture", contract, hand=Hand.LEFT),
        runtime,
    )

    assert outcome.run.decision.inspection_status is InspectionStatus.INVALID_CAPTURE
    assert outcome.run.decision.released_status is None
    assert outcome.run.decision.reason_codes == ("capture_hand_not_enabled",)
    assert cropper.calls == 0
    assert events == []
    validate_inspection_run_for_publication(runtime, outcome.run)


@pytest.mark.parametrize("failed_gate", ["quality", "registration"])
def test_failed_capture_gate_returns_retake_before_crop_or_model_load(
    tmp_path: Path,
    compiled_contract_factory: object,
    failed_gate: str,
) -> None:
    contract = compiled_contract_factory()
    events: list[str] = []
    orchestrator, cropper = _orchestrator(tmp_path, events)
    runtime = RecordingRuntime(contract, events)

    outcome = orchestrator.inspect(
        _request(tmp_path / "capture", contract, failed_gate=failed_gate),
        runtime,
    )

    assert outcome.run.decision.inspection_status is InspectionStatus.RETAKE
    assert outcome.run.decision.released_status is None
    assert cropper.calls == 0
    assert outcome.run.crops == {}
    assert events == []
    validate_inspection_run_for_publication(runtime, outcome.run)


def test_template_ng_short_circuits_both_second_layer_model_loads(
    tmp_path: Path,
    compiled_contract_factory: object,
) -> None:
    contract = compiled_contract_factory()
    events: list[str] = []
    orchestrator, cropper = _orchestrator(tmp_path, events)
    runtime = RecordingRuntime(contract, events, template_risk=0.9)

    outcome = orchestrator.inspect(_request(tmp_path / "capture", contract), runtime)

    assert outcome.run.decision.inspection_status is InspectionStatus.NG_TEMPLATE
    assert outcome.run.decision.evidence_status is InspectionStatus.NG_TEMPLATE
    assert outcome.run.decision.released_status is InspectionStatus.NG_TEMPLATE
    assert cropper.calls == 1
    assert events == ["crop", "load:template", "predict:template"]
    assert outcome.run.raw_model_scores == ()
    assert outcome.run.model_evidence == ()


def test_strong_anomaly_evidence_is_preserved_when_yolo_load_fails(
    tmp_path: Path,
    compiled_contract_factory: object,
) -> None:
    contract = compiled_contract_factory()
    events: list[str] = []
    orchestrator, cropper = _orchestrator(tmp_path, events)
    runtime = RecordingRuntime(
        contract,
        events,
        template_risk=0.1,
        anomaly_score=0.9,
        yolo_load_error=RuntimeError("yolo unavailable"),
    )

    outcome = orchestrator.inspect(_request(tmp_path / "capture", contract), runtime)

    assert cropper.calls == 1
    _assert_second_layer_event_contract(
        events,
        expected={"load:anomaly", "predict:anomaly", "load:yolo"},
    )
    assert outcome.run.decision.inspection_status is InspectionStatus.SYSTEM_ERROR
    assert outcome.run.decision.evidence_status is InspectionStatus.NG_ANOMALY
    assert outcome.run.decision.released_status is None
    assert outcome.run.decision.reason_codes == ("yolo_runtime_error:yolo unavailable",)
    assert {row.branch for row in outcome.run.model_evidence} == {EvidenceBranch.ANOMALY}
    assert all(row.level is EvidenceLevel.STRONG for row in outcome.run.model_evidence)
    assert outcome.run.system_errors == ("yolo_runtime_error:yolo unavailable",)
    validate_inspection_run_for_publication(runtime, outcome.run)


def test_yolo_still_runs_and_preserves_strong_evidence_after_anomaly_load_fails(
    tmp_path: Path,
    compiled_contract_factory: object,
) -> None:
    contract = compiled_contract_factory()
    events: list[str] = []
    orchestrator, cropper = _orchestrator(tmp_path, events)
    runtime = RecordingRuntime(
        contract,
        events,
        anomaly_load_error=RuntimeError("anomaly unavailable"),
        yolo_score=0.9,
    )

    outcome = orchestrator.inspect(_request(tmp_path / "capture", contract), runtime)

    assert cropper.calls == 1
    _assert_second_layer_event_contract(
        events,
        expected={"load:anomaly", "load:yolo", "predict:yolo"},
    )
    assert outcome.run.decision.inspection_status is InspectionStatus.SYSTEM_ERROR
    assert outcome.run.decision.evidence_status is InspectionStatus.NG_YOLO
    assert outcome.run.decision.released_status is None
    assert outcome.run.decision.reason_codes == (
        "anomaly_runtime_error:anomaly unavailable",
    )
    assert {row.branch for row in outcome.run.model_evidence} == {EvidenceBranch.YOLO}
    assert all(row.level is EvidenceLevel.STRONG for row in outcome.run.model_evidence)
    assert outcome.run.system_errors == ("anomaly_runtime_error:anomaly unavailable",)
    validate_inspection_run_for_publication(runtime, outcome.run)


def test_parallel_second_layer_waits_for_healthy_branch_when_peer_prediction_fails(
    tmp_path: Path,
    compiled_contract_factory: object,
) -> None:
    contract = compiled_contract_factory()
    events: list[str] = []
    runtime = RecordingRuntime(
        contract,
        events,
        anomaly_predict_error=RuntimeError("anomaly prediction failed"),
        yolo_score=0.9,
        second_layer_barrier=Barrier(2),
    )
    orchestrator, cropper = _orchestrator(tmp_path, events)

    outcome = orchestrator.inspect(_request(tmp_path / "capture", contract), runtime)

    assert cropper.calls == 1
    _assert_second_layer_event_contract(
        events,
        expected={
            "load:anomaly",
            "predict:anomaly",
            "load:yolo",
            "predict:yolo",
        },
    )
    assert outcome.run.decision.inspection_status is InspectionStatus.SYSTEM_ERROR
    assert outcome.run.decision.evidence_status is InspectionStatus.NG_YOLO
    assert outcome.run.decision.released_status is None
    assert outcome.run.decision.reason_codes == (
        "anomaly_runtime_error:anomaly prediction failed",
    )
    assert {row.branch for row in outcome.run.model_evidence} == {EvidenceBranch.YOLO}
    assert all(row.level is EvidenceLevel.STRONG for row in outcome.run.model_evidence)
    assert outcome.run.system_errors == (
        "anomaly_runtime_error:anomaly prediction failed",
    )
    validate_inspection_run_for_publication(runtime, outcome.run)


def test_parallel_second_layer_output_order_is_independent_of_predictor_order(
    tmp_path: Path,
    compiled_contract_factory: object,
) -> None:
    contract = compiled_contract_factory()
    events: list[str] = []
    runtime = RecordingRuntime(contract, events, reverse_model_rows=True)
    orchestrator, _cropper = _orchestrator(tmp_path, events)

    outcome = orchestrator.inspect(_request(tmp_path / "capture", contract), runtime)

    expected = [
        (branch, view)
        for branch in (EvidenceBranch.ANOMALY.value, EvidenceBranch.YOLO.value)
        for view in sorted(contract.topology.required_views)
    ]
    assert [(row.branch, row.view) for row in outcome.run.raw_model_scores] == expected
    assert [
        (row.branch.value, row.view_id)
        for row in outcome.run.model_evidence
    ] == expected
    assert outcome.run.decision.inspection_status is InspectionStatus.OK
    validate_inspection_run_for_publication(runtime, outcome.run)


def test_anomaly_predictor_lifetime_extends_through_sink_publication(
    tmp_path: Path,
    compiled_contract_factory: object,
) -> None:
    contract = compiled_contract_factory()
    runtime = RecordingRuntime(contract, [])

    def load_deleting_predictor() -> DeletingRawPredictor:
        runtime.events.append("load:anomaly")
        return DeletingRawPredictor(
            runtime,
            EvidenceBranch.ANOMALY,
            runtime.anomaly_score,
        )

    runtime.load_anomaly_predictor = load_deleting_predictor
    sink = VisualPresenceSink(tmp_path / "published")
    orchestrator = InspectionOrchestrator(
        cropper=RecordingCropper(tmp_path / "crops", runtime.events),
        sink=sink,
    )

    outcome = orchestrator.inspect(_request(tmp_path / "capture", contract), runtime)

    assert outcome.run.decision.inspection_status is InspectionStatus.OK
    assert sink.saw_complete_visuals is True


@pytest.mark.parametrize(
    ("template_risk", "anomaly_score", "yolo_score", "expected_status"),
    [
        (0.1, 0.1, 0.1, InspectionStatus.OK),
        (0.1, 0.5, 0.1, InspectionStatus.REVIEW),
        (0.1, 0.9, 0.1, InspectionStatus.NG_ANOMALY),
        (0.1, 0.1, 0.9, InspectionStatus.NG_YOLO),
        (0.9, 0.1, 0.1, InspectionStatus.NG_TEMPLATE),
    ],
)
def test_sink_contract_accepts_only_complete_strict_fusion_runs(
    tmp_path: Path,
    compiled_contract_factory: object,
    template_risk: float,
    anomaly_score: float,
    yolo_score: float,
    expected_status: InspectionStatus,
) -> None:
    contract = compiled_contract_factory()
    events: list[str] = []
    runtime = RecordingRuntime(
        contract,
        events,
        template_risk=template_risk,
        anomaly_score=anomaly_score,
        yolo_score=yolo_score,
    )
    orchestrator, _cropper = _orchestrator(tmp_path, events)
    outcome = orchestrator.inspect(_request(tmp_path / "capture", contract), runtime)

    assert outcome.run.decision.inspection_status is expected_status
    validate_inspection_run_for_publication(runtime, outcome.run)


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("missing_heatmap", "missing its heatmap"),
        ("missing_overlay", "missing its overlay"),
        ("bytes", "overlay hash differs"),
        ("yolo_field", "YOLO raw score must not carry anomaly visuals"),
    ],
)
def test_sink_rejects_missing_tampered_or_misassigned_anomaly_overlay(
    tmp_path: Path,
    compiled_contract_factory: object,
    tamper: str,
    message: str,
) -> None:
    contract = compiled_contract_factory()
    runtime = RecordingRuntime(contract, [])
    orchestrator, _cropper = _orchestrator(tmp_path, [])
    outcome = orchestrator.inspect(_request(tmp_path / "capture", contract), runtime)
    rows = list(outcome.run.raw_model_scores)
    anomaly_index = next(
        index
        for index, row in enumerate(rows)
        if row.branch == EvidenceBranch.ANOMALY.value
    )
    if tamper == "missing_heatmap":
        rows[anomaly_index] = replace(
            rows[anomaly_index],
            heatmap_path=None,
            heatmap_sha256=None,
        )
    elif tamper == "missing_overlay":
        rows[anomaly_index] = replace(
            rows[anomaly_index],
            overlay_path=None,
            overlay_sha256=None,
        )
    elif tamper == "bytes":
        overlay = rows[anomaly_index].overlay_path
        assert overlay is not None
        overlay.write_bytes(b"changed-after-anomaly-prediction")
    else:
        yolo_index = next(
            index
            for index, row in enumerate(rows)
            if row.branch == EvidenceBranch.YOLO.value
        )
        anomaly = rows[anomaly_index]
        rows[yolo_index] = replace(
            rows[yolo_index],
            heatmap_path=anomaly.heatmap_path,
            heatmap_sha256=anomaly.heatmap_sha256,
        )
    forged = replace(outcome.run, raw_model_scores=tuple(rows))

    with pytest.raises(PublicationError, match=message):
        validate_inspection_run_for_publication(runtime, forged)


def test_sink_template_reference_contract_accepts_exact_release_reference(
    tmp_path: Path,
    compiled_contract_factory: object,
) -> None:
    contract = compiled_contract_factory()
    runtime = RecordingRuntime(contract, [])
    orchestrator, _cropper = _orchestrator(tmp_path, [])
    outcome = orchestrator.inspect(_request(tmp_path / "capture", contract), runtime)
    run = _bind_release_template_references(
        tmp_path / "release-references",
        runtime,
        outcome.run,
    )

    _validate_template_score_references(runtime, run)


def test_filesystem_sink_publishes_both_anomaly_visuals_with_checksums(
    tmp_path: Path,
    compiled_contract_factory: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = compiled_contract_factory()
    runtime = RecordingRuntime(contract, [])
    runtime.release_root_sha256 = "8" * 64
    runtime.fusion_policy_file_sha256 = "9" * 64
    orchestrator, _cropper = _orchestrator(tmp_path, [])
    outcome = orchestrator.inspect(_request(tmp_path / "capture", contract), runtime)
    run = _bind_release_template_references(
        tmp_path / "release-references",
        runtime,
        outcome.run,
    )
    manifest = run.request.capture_root / "capture_manifest.json"
    gates = run.request.capture_root / "capture_gates.json"
    manifest.write_bytes(b'{"schema_version":1}\n')
    gates.write_bytes(b'{"schema_version":1}\n')
    object.__setattr__(run.request, "capture_manifest_path", manifest)
    object.__setattr__(run.request, "capture_manifest_sha256", hashlib.sha256(manifest.read_bytes()).hexdigest())
    object.__setattr__(run.request, "capture_gates_path", gates)
    object.__setattr__(run.request, "capture_gates_sha256", hashlib.sha256(gates.read_bytes()).hexdigest())
    object.__setattr__(run.request, "capture_publication_root_sha256", "a" * 64)
    monkeypatch.setattr(
        inspection_sink_module,
        "_revalidate_capture_publication",
        lambda _runtime, _run: None,
    )

    published = FilesystemInspectionSink(tmp_path / "inspection-output").publish(runtime, run)

    checksums = (published / "checksums.sha256").read_text(encoding="utf-8")
    for view in contract.topology.required_views:
        for kind in ("heatmaps", "overlays"):
            relative = f"{kind}/anomaly/{view}.png"
            assert (published / relative).is_file()
            assert relative in checksums


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("path", "outside the release allowlist"),
        ("hash", "hash differs from release"),
        ("bytes", "hash differs from release"),
        ("offset", "offset exceeds release max_shift"),
    ],
)
def test_sink_template_reference_contract_rejects_path_hash_and_offset_tampering(
    tmp_path: Path,
    compiled_contract_factory: object,
    tamper: str,
    message: str,
) -> None:
    contract = compiled_contract_factory()
    runtime = RecordingRuntime(contract, [])
    orchestrator, _cropper = _orchestrator(tmp_path, [])
    outcome = orchestrator.inspect(_request(tmp_path / "capture", contract), runtime)
    run = _bind_release_template_references(
        tmp_path / "release-references",
        runtime,
        outcome.run,
        max_shift=2,
    )
    first = run.template_scores[0]
    if tamper == "path":
        outside = tmp_path / "outside-reference.png"
        outside.write_bytes(first.best_template_path.read_bytes())
        forged_score = replace(first, best_template_path=outside)
    elif tamper == "hash":
        forged_score = replace(first, best_template_sha256="0" * 64)
    elif tamper == "bytes":
        first.best_template_path.write_bytes(b"changed-after-template-scoring")
        forged_score = first
    else:
        forged_score = replace(first, offset_xy=(3, 0))
    forged = replace(
        run,
        template_scores=(forged_score, *run.template_scores[1:]),
    )

    with pytest.raises(PublicationError, match=message):
        _validate_template_score_references(runtime, forged)


def test_filesystem_sink_rejects_ok_with_one_missing_yolo_group_before_writing(
    tmp_path: Path,
    compiled_contract_factory: object,
) -> None:
    contract = compiled_contract_factory()
    events: list[str] = []
    runtime = RecordingRuntime(contract, events)
    orchestrator, _cropper = _orchestrator(tmp_path, events)
    outcome = orchestrator.inspect(_request(tmp_path / "capture", contract), runtime)
    missing_view = contract.topology.required_views[0]
    forged = replace(
        outcome.run,
        raw_model_scores=tuple(
            row
            for row in outcome.run.raw_model_scores
            if not (row.view == missing_view and row.branch == EvidenceBranch.YOLO.value)
        ),
        model_evidence=tuple(
            row
            for row in outcome.run.model_evidence
            if not (row.view_id == missing_view and row.branch is EvidenceBranch.YOLO)
        ),
    )
    output_root = tmp_path / "forged-output"

    with pytest.raises(PublicationError, match="raw model evidence is incomplete"):
        FilesystemInspectionSink(output_root).publish(runtime, forged)

    assert not (output_root / forged.request.inspection_id).exists()


def test_filesystem_sink_rederives_review_instead_of_trusting_forged_ok(
    tmp_path: Path,
    compiled_contract_factory: object,
) -> None:
    contract = compiled_contract_factory()
    events: list[str] = []
    runtime = RecordingRuntime(contract, events, anomaly_score=0.5)
    orchestrator, _cropper = _orchestrator(tmp_path, events)
    outcome = orchestrator.inspect(_request(tmp_path / "capture", contract), runtime)
    forged = replace(
        outcome.run,
        decision=InspectionDecision(
            inspection_id=outcome.run.request.inspection_id,
            evidence_status=None,
            inspection_status=InspectionStatus.OK,
            review_status=None,
            released_status=InspectionStatus.OK,
            reason_codes=("all_required_model_evidence_clear",),
        ),
    )

    with pytest.raises(PublicationError, match="decision differs from strict evidence-derived decision"):
        FilesystemInspectionSink(tmp_path / "forged-output").publish(runtime, forged)


def test_filesystem_sink_rejects_incomplete_ng_template_evidence(
    tmp_path: Path,
    compiled_contract_factory: object,
) -> None:
    contract = compiled_contract_factory()
    events: list[str] = []
    runtime = RecordingRuntime(contract, events, template_risk=0.9)
    orchestrator, _cropper = _orchestrator(tmp_path, events)
    outcome = orchestrator.inspect(_request(tmp_path / "capture", contract), runtime)
    forged = replace(outcome.run, template_evidence=outcome.run.template_evidence[:-1])

    with pytest.raises(PublicationError, match="template evidence is incomplete"):
        FilesystemInspectionSink(tmp_path / "forged-output").publish(runtime, forged)


def test_filesystem_sink_rejects_retake_when_all_persisted_gates_pass(
    tmp_path: Path,
    compiled_contract_factory: object,
) -> None:
    contract = compiled_contract_factory()
    events: list[str] = []
    runtime = RecordingRuntime(contract, events)
    orchestrator, _cropper = _orchestrator(tmp_path, events)
    failed = orchestrator.inspect(
        _request(tmp_path / "failed-capture", contract, failed_gate="quality"),
        runtime,
    )
    passing_request = _request(tmp_path / "passing-capture", contract)
    forged = replace(
        failed.run,
        request=passing_request,
        gate_results=passing_request.capture_gate_results,
    )

    with pytest.raises(PublicationError, match="RETAKE requires at least one failed capture gate"):
        FilesystemInspectionSink(tmp_path / "forged-output").publish(runtime, forged)


def test_filesystem_sink_rejects_system_error_without_retained_error_evidence(
    tmp_path: Path,
    compiled_contract_factory: object,
) -> None:
    contract = compiled_contract_factory()
    events: list[str] = []
    runtime = RecordingRuntime(
        contract,
        events,
        yolo_load_error=RuntimeError("yolo unavailable"),
    )
    orchestrator, _cropper = _orchestrator(tmp_path, events)
    outcome = orchestrator.inspect(_request(tmp_path / "capture", contract), runtime)
    forged = replace(outcome.run, system_errors=())

    with pytest.raises(PublicationError, match="requires non-empty retained system error evidence"):
        FilesystemInspectionSink(tmp_path / "forged-output").publish(runtime, forged)


def test_filesystem_sink_rejects_invalid_capture_with_downstream_evidence(
    tmp_path: Path,
    compiled_contract_factory: object,
) -> None:
    contract = compiled_contract_factory()
    events: list[str] = []
    runtime = RecordingRuntime(contract, events)
    orchestrator, _cropper = _orchestrator(tmp_path, events)
    outcome = orchestrator.inspect(_request(tmp_path / "capture", contract), runtime)
    forged = replace(
        outcome.run,
        decision=InspectionDecision(
            inspection_id=outcome.run.request.inspection_id,
            evidence_status=None,
            inspection_status=InspectionStatus.INVALID_CAPTURE,
            review_status=None,
            released_status=None,
            reason_codes=("capture_identity_invalid",),
        ),
    )

    with pytest.raises(PublicationError, match="must not carry downstream stage evidence"):
        FilesystemInspectionSink(tmp_path / "forged-output").publish(runtime, forged)
