"""Linux-only atomic raw capture publication tests."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from zs32_inspection.capture import (
    AtomicCaptureStore,
    CaptureFrame,
    CapturePlan,
    CaptureRequest,
    RoundConfirmation,
    load_capture_bundle,
)
from zs32_inspection.capture.gates import CaptureGateResult
from zs32_inspection.capture.gate_policy import CaptureGateProvenance
from zs32_inspection.capture.manifests import build_capture_rows
from zs32_inspection.domain.contracts import canonical_sha256
from zs32_inspection.domain.identity import Hand, PartIdentity
from zs32_inspection.domain.topology import CameraSlot, CaptureRound, CaptureTopology
from zs32_inspection.runtime.orchestrator import InspectionRequest
from zs32_inspection.runtime.inspection_sink import _revalidate_capture_publication
from zs32_inspection.runtime.publisher import PublicationError


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="authoritative runtime is Linux only")


def _topology() -> CaptureTopology:
    rounds = (CaptureRound("front", "front prompt"), CaptureRound("back", "back prompt"))
    slots = tuple(
        CameraSlot(
            f"slot{index}",
            f"SERIAL{index}",
            {"front": f"front_{index}", "back": f"back_{index}"},
        )
        for index in range(3)
    )
    required_views = tuple(
        [f"front_{index}" for index in range(3)]
        + [f"back_{index}" for index in range(3)]
    )
    topology_sha256 = canonical_sha256(
        {
            "schema_version": 1,
            "topology_id": "topology",
            "product": "ZS32",
            "rounds": [
                {"round_id": item.round_id, "prompt": item.prompt}
                for item in rounds
            ],
            "camera_slots": [
                {
                    "slot_id": item.slot_id,
                    "serial": item.serial,
                    "views": dict(sorted(item.views.items())),
                }
                for item in slots
            ],
            "required_views": list(required_views),
        }
    )
    return CaptureTopology(
        1,
        "topology",
        "ZS32",
        rounds,
        slots,
        required_views,
        topology_sha256,
    )


def _bundle(capture_set_id: str, part_id: str):
    topology = _topology()
    plan = CapturePlan.from_topology(topology)
    request = CaptureRequest("shared-session", capture_set_id, PartIdentity(part_id, Hand.RIGHT))
    frames = tuple(
        CaptureFrame(
            round_id=capture_round.round_id,
            view_id=slot.views[capture_round.round_id],
            camera_slot_id=slot.slot_id,
            camera_serial=slot.serial,
            device_index=index,
            image_bytes=(
                b"\x89PNG\r\n\x1a\n"
                + f"{capture_set_id}:{capture_round.round_id}:{slot.slot_id}".encode()
            ),
            width=4024,
            height=3036,
            capture_mode="single",
            exposure=4000.0,
            gain=0.0,
            captured_at="2026-07-14T00:00:00Z",
        )
        for capture_round in topology.rounds
        for index, slot in enumerate(topology.camera_slots)
    )
    gate_results = tuple(
        CaptureGateResult(gate, True, "", view_id)
        for gate in ("quality", "registration")
        for view_id in plan.required_views
    )
    return plan, build_capture_rows(
        request,
        plan,
        frames,
        gate_results,
        CaptureGateProvenance(
            policy_id="test-policy",
            policy_sha256="1" * 64,
            hand=Hand.RIGHT,
            topology_sha256=plan.topology_sha256,
            acquisition_config_sha256="5" * 64,
            quality_profile_sha256="2" * 64,
            registration_profile_sha256="3" * 64,
            registration_reference_sha256_by_view={
                view: "4" * 64 for view in plan.required_views
            },
        ),
        tuple(
            RoundConfirmation(
                item.round_id,
                item.prompt,
                "operator-test",
                "2026-07-14T00:00:00Z",
                "2026-07-14T00:00:01Z",
            )
            for item in plan.rounds
        ),
        started_at="2026-07-14T00:00:00Z",
    )


def test_same_session_accepts_multiple_unique_capture_sets(tmp_path) -> None:
    store = AtomicCaptureStore(tmp_path / "raw")
    first_plan, first = _bundle("set-1", "part-1")
    second_plan, second = _bundle("set-2", "part-2")
    first_path = store.publish_complete(first_plan, first)
    second_path = store.publish_complete(second_plan, second)
    assert first_path == tmp_path / "raw/shared-session/images/set-1"
    assert second_path == tmp_path / "raw/shared-session/images/set-2"
    assert (first_path / "front_0.png").is_file()
    assert (second_path / "back_2.png").is_file()
    assert (first_path / "capture_gates.json").is_file()
    reloaded = load_capture_bundle(first_path, _topology())
    assert reloaded.gate_results == first.gate_results
    assert reloaded.round_confirmations == first.round_confirmations
    inspection = InspectionRequest.from_capture_publication(
        inspection_id="inspection-set-1",
        release_id="release-test",
        capture_set_root=first_path,
        topology=_topology(),
    )
    assert inspection.capture.capture_set_id == "set-1"


@pytest.mark.parametrize("changed_role", ["source", "manifest", "gates", "checksum_root"])
def test_inspection_sink_revalidates_complete_capture_publication_at_final_boundary(
    tmp_path: Path,
    changed_role: str,
) -> None:
    topology = _topology()
    store = AtomicCaptureStore(tmp_path / "raw")
    plan, bundle = _bundle("set-revalidate", "part-revalidate")
    published = store.publish_complete(plan, bundle)
    request = InspectionRequest.from_capture_publication(
        inspection_id="inspection-revalidate",
        release_id="release-test",
        capture_set_root=published,
        topology=topology,
    )
    runtime = SimpleNamespace(contract=SimpleNamespace(topology=topology))
    run = SimpleNamespace(request=request)

    _revalidate_capture_publication(runtime, run)

    changed = {
        "source": published / f"{topology.required_views[0]}.png",
        "manifest": published / "capture_manifest.json",
        "gates": published / "capture_gates.json",
        "checksum_root": published / "checksums.sha256",
    }[changed_role]
    changed.chmod(0o600)
    changed.write_bytes(b"changed-after-request-verification")
    with pytest.raises(PublicationError, match="capture publication revalidation failed"):
        _revalidate_capture_publication(runtime, run)


def test_inspection_sink_rejects_capture_publication_that_differs_from_memory(
    tmp_path: Path,
) -> None:
    topology = _topology()
    store = AtomicCaptureStore(tmp_path / "raw")
    plan, bundle = _bundle("set-memory", "part-memory")
    published = store.publish_complete(plan, bundle)
    request = InspectionRequest.from_capture_publication(
        inspection_id="inspection-memory",
        release_id="release-test",
        capture_set_root=published,
        topology=topology,
    )
    forged_request = SimpleNamespace(
        capture=replace(
            request.capture,
            part=PartIdentity("different-part", Hand.RIGHT),
        ),
        capture_root=request.capture_root,
        capture_manifest_path=request.capture_manifest_path,
        capture_manifest_sha256=request.capture_manifest_sha256,
        capture_gates_path=request.capture_gates_path,
        capture_gates_sha256=request.capture_gates_sha256,
        capture_publication_root_sha256=request.capture_publication_root_sha256,
        capture_gate_results=request.capture_gate_results,
        capture_gate_provenance=request.capture_gate_provenance,
    )
    runtime = SimpleNamespace(contract=SimpleNamespace(topology=topology))

    with pytest.raises(PublicationError, match="differs from the in-memory CaptureSet"):
        _revalidate_capture_publication(runtime, SimpleNamespace(request=forged_request))


def test_inspection_sink_rejects_in_memory_source_hash_not_in_manifest(
    tmp_path: Path,
) -> None:
    topology = _topology()
    store = AtomicCaptureStore(tmp_path / "raw")
    plan, bundle = _bundle("set-source-memory", "part-source-memory")
    published = store.publish_complete(plan, bundle)
    request = InspectionRequest.from_capture_publication(
        inspection_id="inspection-source-memory",
        release_id="release-test",
        capture_set_root=published,
        topology=topology,
    )
    view = topology.required_views[0]
    forged_images = dict(request.capture.images)
    forged_images[view] = replace(forged_images[view], image_sha256="0" * 64)
    forged_request = SimpleNamespace(
        capture=replace(request.capture, images=forged_images),
        capture_root=request.capture_root,
        capture_manifest_path=request.capture_manifest_path,
        capture_manifest_sha256=request.capture_manifest_sha256,
        capture_gates_path=request.capture_gates_path,
        capture_gates_sha256=request.capture_gates_sha256,
        capture_publication_root_sha256=request.capture_publication_root_sha256,
        capture_gate_results=request.capture_gate_results,
        capture_gate_provenance=request.capture_gate_provenance,
    )
    runtime = SimpleNamespace(contract=SimpleNamespace(topology=topology))

    with pytest.raises(PublicationError, match="differs from the in-memory CaptureSet"):
        _revalidate_capture_publication(runtime, SimpleNamespace(request=forged_request))


def test_existing_capture_set_is_never_overwritten(tmp_path) -> None:
    store = AtomicCaptureStore(tmp_path / "raw")
    plan, bundle = _bundle("same-set", "part-1")
    store.publish_complete(plan, bundle)
    with pytest.raises(Exception, match="publication already exists"):
        store.publish_complete(plan, bundle)
