"""Linux-only production contract tests for OpenCV capture gates."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from zs32_inspection.capture import (
    CameraBinding,
    CaptureFrame,
    CapturePlan,
    CaptureRequest,
    CaptureRoundPlan,
    CaptureService,
    IncompleteCaptureError,
    OpenCvQualityGate,
    OpenCvRegistrationGate,
    QualityGateProfile,
    RegistrationGateProfile,
    RoundConfirmation,
)
from zs32_inspection.capture.gates import CaptureGateResult
from zs32_inspection.capture.gate_policy import CaptureGateProvenance
from zs32_inspection.domain.contracts import canonical_sha256
from zs32_inspection.domain.identity import Hand, PartIdentity


pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="OpenCV capture-gate verification is authoritative on Linux only",
)


def _gate_provenance(plan: CapturePlan) -> CaptureGateProvenance:
    return CaptureGateProvenance(
        policy_id="test-policy",
        policy_sha256="1" * 64,
        hand=Hand.RIGHT,
        topology_sha256=plan.topology_sha256,
        acquisition_config_sha256="5" * 64,
        quality_profile_sha256="2" * 64,
        registration_profile_sha256="3" * 64,
        registration_reference_sha256_by_view={view: "4" * 64 for view in plan.required_views},
    )


def _plan(camera_count: int) -> CapturePlan:
    rounds = (
        CaptureRoundPlan("front", "capture front"),
        CaptureRoundPlan("back", "capture back"),
    )
    cameras = tuple(
        CameraBinding(
            f"slot{index}",
            f"SERIAL{index}",
            {"front": f"front_{index}", "back": f"back_{index}"},
        )
        for index in range(camera_count)
    )
    return CapturePlan(
        "ZS32",
        f"topology-{camera_count}",
        f"{camera_count}" * 64,
        rounds,
        cameras,
        tuple(
            [f"front_{index}" for index in range(camera_count)]
            + [f"back_{index}" for index in range(camera_count)]
        ),
    )


def _request() -> CaptureRequest:
    return CaptureRequest("session", "capture", PartIdentity("part", Hand.RIGHT))


def _encode_png(image) -> bytes:
    import cv2

    written, encoded = cv2.imencode(".png", image)
    assert written
    return encoded.tobytes()


def _checkerboard(width: int = 128, height: int = 96):
    import numpy as np

    yy, xx = np.indices((height, width))
    gray = (((xx // 8 + yy // 8) % 2) * 180 + 35).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def _frames(plan: CapturePlan, payload_by_view: dict[str, bytes]) -> tuple[CaptureFrame, ...]:
    frames = []
    for round_plan in plan.rounds:
        for device_index, camera in enumerate(plan.cameras):
            view_id = camera.views[round_plan.round_id]
            frames.append(
                CaptureFrame(
                    round_id=round_plan.round_id,
                    view_id=view_id,
                    camera_slot_id=camera.slot_id,
                    camera_serial=camera.serial,
                    device_index=device_index,
                    image_bytes=payload_by_view[view_id],
                    width=128,
                    height=96,
                    capture_mode="single",
                    exposure=4000.0,
                    gain=0.0,
                    captured_at="2026-07-14T00:00:00Z",
                )
            )
    return tuple(frames)


def _quality_thresholds() -> dict[str, float | int]:
    return {
        "brightness_mean_min": 20.0,
        "brightness_mean_max": 235.0,
        "brightness_std_min": 20.0,
        "dark_pixel_level": 5,
        "dark_ratio_max": 0.1,
        "saturated_pixel_level": 250,
        "saturated_ratio_max": 0.1,
        "laplacian_variance_min": 50.0,
    }


def _quality_profile(
    plan: CapturePlan,
    *,
    views: tuple[str, ...] | None = None,
) -> QualityGateProfile:
    payload: dict[str, object] = {
        "schema_version": 1,
        "profile_id": "quality-v1",
        "product": "ZS32",
        "hand": "right",
        "topology_id": plan.topology_id,
        "topology_sha256": plan.topology_sha256,
        "views": {
            view: _quality_thresholds()
            for view in (plan.required_views if views is None else views)
        },
    }
    payload["profile_sha256"] = canonical_sha256(payload)
    return QualityGateProfile.from_mapping(payload)


@pytest.mark.parametrize("camera_count", [3, 4, 5])
def test_quality_gate_is_topology_driven_for_six_eight_ten_views(camera_count: int) -> None:
    plan = _plan(camera_count)
    source = _encode_png(_checkerboard())
    frames = _frames(plan, {view: source for view in plan.required_views})
    results = OpenCvQualityGate(_quality_profile(plan)).evaluate(_request(), plan, frames)
    assert len(results) == camera_count * 2
    assert tuple(item.view_id for item in results) == plan.required_views
    assert all(item.passed for item in results)
    assert all("profile_sha256=" in item.reason for item in results)


def test_quality_gate_has_no_default_pass_for_blurred_view() -> None:
    import numpy as np

    plan = _plan(1)
    textured = _encode_png(_checkerboard())
    flat = _encode_png(np.full((96, 128, 3), 120, dtype=np.uint8))
    frames = _frames(
        plan,
        {plan.required_views[0]: flat, plan.required_views[1]: textured},
    )
    results = OpenCvQualityGate(_quality_profile(plan)).evaluate(_request(), plan, frames)
    failed = [item for item in results if not item.passed]
    assert len(failed) == 1
    assert failed[0].view_id == plan.required_views[0]
    assert "laplacian_variance below" in failed[0].reason


def test_quality_profile_missing_one_required_view_fails_before_dependency_load() -> None:
    plan = _plan(3)
    called = False

    def forbidden_loader():
        nonlocal called
        called = True
        raise AssertionError("OpenCV must not load for an invalid profile")

    gate = OpenCvQualityGate(
        _quality_profile(plan, views=plan.required_views[:-1]),
        dependency_loader=forbidden_loader,
    )
    with pytest.raises(ValueError, match="view set differs"):
        gate.evaluate(_request(), plan, ())
    assert not called


def test_gate_profile_cannot_be_reused_for_the_other_hand() -> None:
    plan = _plan(1)
    left_request = CaptureRequest(
        "session",
        "capture-left",
        PartIdentity("part-left", Hand.LEFT),
    )
    called = False

    def forbidden_loader():
        nonlocal called
        called = True
        raise AssertionError("OpenCV must not load for a hand mismatch")

    gate = OpenCvQualityGate(
        _quality_profile(plan),
        dependency_loader=forbidden_loader,
    )
    with pytest.raises(ValueError, match="conflicts with capture hand"):
        gate.evaluate(left_request, plan, ())
    assert not called


def _registration_thresholds(
    *,
    max_translation_x_px: float = 4.0,
) -> dict[str, float | int]:
    return {
        "analysis_scale": 1.0,
        "min_ecc_correlation": 0.8,
        "max_translation_x_px": max_translation_x_px,
        "max_translation_y_px": 4.0,
        "max_rotation_deg": 2.0,
        "max_scale_deviation": 0.03,
        "max_shear": 0.03,
        "max_iterations": 100,
        "epsilon": 1e-6,
        "gaussian_filter_size": 5,
    }


def _registration_profile(
    plan: CapturePlan,
    reference_by_view: dict[str, bytes],
    *,
    max_translation_x_px: float = 4.0,
) -> RegistrationGateProfile:
    payload: dict[str, object] = {
        "schema_version": 1,
        "profile_id": "registration-v1",
        "product": "ZS32",
        "hand": "right",
        "topology_id": plan.topology_id,
        "topology_sha256": plan.topology_sha256,
        "views": {
            view: {
                "reference": {
                    "asset_id": f"reference-{view}",
                    "relative_path": f"references/{view}.png",
                    "sha256": hashlib.sha256(reference_by_view[view]).hexdigest(),
                    "width": 128,
                    "height": 96,
                },
                "thresholds": _registration_thresholds(
                    max_translation_x_px=max_translation_x_px
                ),
            }
            for view in plan.required_views
        },
    }
    payload["profile_sha256"] = canonical_sha256(payload)
    return RegistrationGateProfile.from_mapping(payload)


def _write_references(root: Path, reference_by_view: dict[str, bytes]) -> None:
    reference_dir = root / "references"
    reference_dir.mkdir(parents=True)
    for view, payload in reference_by_view.items():
        (reference_dir / f"{view}.png").write_bytes(payload)


def test_registration_gate_passes_identity_and_rejects_excess_translation(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    plan = _plan(1)
    random = np.random.default_rng(43)
    gray = random.integers(20, 235, size=(96, 128), dtype=np.uint8)
    reference_image = np.repeat(gray[:, :, None], 3, axis=2)
    reference = _encode_png(reference_image)
    reference_by_view = {view: reference for view in plan.required_views}
    _write_references(tmp_path, reference_by_view)
    shifted_image = cv2.warpAffine(
        reference_image,
        np.array([[1.0, 0.0, 12.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        (128, 96),
    )
    frames = _frames(
        plan,
        {
            plan.required_views[0]: reference,
            plan.required_views[1]: _encode_png(shifted_image),
        },
    )
    results = OpenCvRegistrationGate(
        _registration_profile(plan, reference_by_view),
        asset_root=tmp_path,
    ).evaluate(_request(), plan, frames)
    assert results[0].passed is True
    assert results[1].passed is False
    assert "translation_x_px" in results[1].reason
    assert "reference_sha256=" in results[1].reason


def test_registration_reference_hash_mismatch_raises_instead_of_returning_gate_failure(
    tmp_path: Path,
) -> None:
    plan = _plan(1)
    reference = _encode_png(_checkerboard())
    reference_by_view = {view: reference for view in plan.required_views}
    profile = _registration_profile(plan, reference_by_view)
    _write_references(tmp_path, reference_by_view)
    (tmp_path / "references" / f"{plan.required_views[0]}.png").write_bytes(
        reference + b"tampered"
    )
    frames = _frames(plan, reference_by_view)
    with pytest.raises(ValueError, match="hash mismatch"):
        OpenCvRegistrationGate(profile, asset_root=tmp_path).evaluate(
            _request(),
            plan,
            frames,
        )


def test_gate_construction_never_loads_opencv(tmp_path: Path) -> None:
    plan = _plan(1)
    reference = b"\x89PNG\r\n\x1a\nnot-decoded-during-construction"
    references = {view: reference for view in plan.required_views}
    called = False

    def forbidden_loader():
        nonlocal called
        called = True
        raise AssertionError("dependency loading must be lazy")

    OpenCvQualityGate(_quality_profile(plan), dependency_loader=forbidden_loader)
    OpenCvRegistrationGate(
        _registration_profile(plan, references),
        asset_root=tmp_path,
        dependency_loader=forbidden_loader,
    )
    assert not called


class _RoundSource:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def capture_round(self, request, round_plan, cameras):
        del request
        return tuple(
            CaptureFrame(
                round_id=round_plan.round_id,
                view_id=camera.views[round_plan.round_id],
                camera_slot_id=camera.slot_id,
                camera_serial=camera.serial,
                device_index=index,
                image_bytes=self.payload,
                width=128,
                height=96,
                capture_mode="single",
                exposure=4000.0,
                gain=0.0,
                captured_at="2026-07-14T00:00:00Z",
            )
            for index, camera in enumerate(cameras)
        )


class _ViewMapSource:
    def __init__(self, payload_by_view: dict[str, bytes]) -> None:
        self.payload_by_view = payload_by_view

    def capture_round(self, request, round_plan, cameras):
        del request
        return tuple(
            CaptureFrame(
                round_id=round_plan.round_id,
                view_id=camera.views[round_plan.round_id],
                camera_slot_id=camera.slot_id,
                camera_serial=camera.serial,
                device_index=index,
                image_bytes=self.payload_by_view[camera.views[round_plan.round_id]],
                width=128,
                height=96,
                capture_mode="single",
                exposure=4000.0,
                gain=0.0,
                captured_at="2026-07-14T00:00:00Z",
            )
            for index, camera in enumerate(cameras)
        )


class _DiagnosticStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.complete = False
        self.failure_reason = ""

    def publish_complete(self, plan, bundle):
        del plan, bundle
        self.complete = True
        return self.root / "complete"

    def publish_incomplete(
        self,
        request,
        plan,
        frames,
        *,
        started_at,
        failure_reason,
        gate_provenance,
        round_confirmations,
    ):
        del request, plan, frames, started_at, round_confirmations
        self.failure_reason = failure_reason
        self.gate_provenance = gate_provenance
        return self.root / "quarantine"


class _PassRegistration:
    name = "registration"

    def evaluate(self, request, plan, frames):
        del request, frames
        return tuple(
            CaptureGateResult("registration", True, "configured pass", view)
            for view in plan.required_views
        )


class _PassQuality:
    name = "quality"

    def evaluate(self, request, plan, frames):
        del request, frames
        return tuple(
            CaptureGateResult("quality", True, "configured pass", view)
            for view in plan.required_views
        )


class _ConfirmRounds:
    def confirm_round(self, request, round_plan, *, round_index, round_count):
        del request, round_index, round_count
        return RoundConfirmation(
            round_plan.round_id,
            round_plan.prompt,
            "operator-test",
            "2026-07-14T00:00:00Z",
            "2026-07-14T00:00:01Z",
        )


def test_opencv_quality_failure_is_published_as_retake(tmp_path: Path) -> None:
    import numpy as np

    plan = _plan(1)
    flat = _encode_png(np.full((96, 128, 3), 120, dtype=np.uint8))
    store = _DiagnosticStore(tmp_path)
    service = CaptureService(
        _RoundSource(flat),
        store,
        gate_provenance=_gate_provenance(plan),
        gates=(OpenCvQualityGate(_quality_profile(plan)), _PassRegistration()),
        round_coordinator=_ConfirmRounds(),
    )
    with pytest.raises(IncompleteCaptureError) as raised:
        service.capture(_request(), plan)
    assert raised.value.failure_kind == "RETAKE"
    assert store.complete is False
    assert store.failure_reason.startswith("RETAKE: capture gate failed")


def test_opencv_registration_failure_is_published_as_retake(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    plan = _plan(1)
    random = np.random.default_rng(43)
    gray = random.integers(20, 235, size=(96, 128), dtype=np.uint8)
    reference_image = np.repeat(gray[:, :, None], 3, axis=2)
    reference = _encode_png(reference_image)
    references = {view: reference for view in plan.required_views}
    _write_references(tmp_path, references)
    shifted = _encode_png(
        cv2.warpAffine(
            reference_image,
            np.array([[1.0, 0.0, 12.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            (128, 96),
        )
    )
    payloads = {
        plan.required_views[0]: reference,
        plan.required_views[1]: shifted,
    }
    store = _DiagnosticStore(tmp_path)
    service = CaptureService(
        _ViewMapSource(payloads),
        store,
        gate_provenance=_gate_provenance(plan),
        gates=(
            _PassQuality(),
            OpenCvRegistrationGate(
                _registration_profile(plan, references),
                asset_root=tmp_path,
            ),
        ),
        round_coordinator=_ConfirmRounds(),
    )
    with pytest.raises(IncompleteCaptureError) as raised:
        service.capture(_request(), plan)
    assert raised.value.failure_kind == "RETAKE"
    assert store.complete is False
    assert store.failure_reason.startswith("RETAKE: capture gate failed")


def test_corrupt_registration_reference_is_published_as_system_error(tmp_path: Path) -> None:
    plan = _plan(1)
    reference = _encode_png(_checkerboard())
    references = {view: reference for view in plan.required_views}
    profile = _registration_profile(plan, references)
    _write_references(tmp_path, references)
    (tmp_path / "references" / f"{plan.required_views[0]}.png").write_bytes(
        reference + b"tampered"
    )
    store = _DiagnosticStore(tmp_path)
    service = CaptureService(
        _ViewMapSource(references),
        store,
        gate_provenance=_gate_provenance(plan),
        gates=(
            _PassQuality(),
            OpenCvRegistrationGate(profile, asset_root=tmp_path),
        ),
        round_coordinator=_ConfirmRounds(),
    )
    with pytest.raises(IncompleteCaptureError) as raised:
        service.capture(_request(), plan)
    assert raised.value.failure_kind == "SYSTEM_ERROR"
    assert store.complete is False
    assert store.failure_reason.startswith("SYSTEM_ERROR: capture gate 'registration' runtime failed")
