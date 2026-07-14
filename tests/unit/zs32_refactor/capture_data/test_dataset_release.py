"""Linux-only canonical release and crop identity tests."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace

import pytest

from zs32_inspection.capture.gate_policy import CaptureGatePolicy
from zs32_inspection.capture.gate_publication import (
    VerifiedCaptureGatePublication,
    load_verified_capture_gate_publication,
)
from zs32_inspection.config.schemas import parse_recipe
from zs32_inspection.data.codec_contract import CanonicalPngCodec
from zs32_inspection.data.calibration_targets import (
    CalibrationTargetApproval,
    CalibrationTargetRecord,
    CalibrationTargetsSnapshot,
)
from zs32_inspection.data.dataset_release import (
    CanonicalDatasetBuilder,
    DatasetBuildSpec,
    SourceSample,
    verify_dataset_release,
)
from zs32_inspection.data.manifests import CanonicalSemanticsSnapshot
from zs32_inspection.data.roi import CroppedImage, LabelDocument
from zs32_inspection.data.splitter import SPLIT_ALGORITHM, SplitAssignment, SplitPolicy
from zs32_inspection.domain.contracts import (
    RoiBox,
    RoiConfig,
    RoiHandConfig,
    RoiReadiness,
    canonical_sha256,
)
from zs32_inspection.domain.identity import CaptureSet, Hand, PartIdentity, ViewImage
from zs32_inspection.domain.topology import CameraSlot, CaptureRound, CaptureTopology
from zs32_inspection.runtime.publisher import AtomicDirectoryPublisher, canonical_json_bytes


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="authoritative runtime is Linux only")


class _IdentityCropper:
    canonical_png_codec = CanonicalPngCodec.opencv(
        1,
        implementation_version="test-fixture",
    )

    def crop_png(self, source_png, *, source_width, source_height, roi):
        return CroppedImage(source_png, roi.width, roi.height)


def _split_policy() -> SplitPolicy:
    return SplitPolicy(SPLIT_ALGORITHM, 43, 0.0, 0.0)


def _png_codec(compression: int = 1) -> CanonicalPngCodec:
    return CanonicalPngCodec.opencv(
        compression,
        implementation_version="test-fixture",
    )


def _semantics_snapshot() -> CanonicalSemanticsSnapshot:
    views = tuple(
        [f"front_{index}" for index in range(3)]
        + [f"back_{index}" for index in range(3)]
    )
    return CanonicalSemanticsSnapshot.from_mapping(
        {
            "schema_version": 2,
            "approval": {
                "reviewed_by": "qa-user-1",
                "reviewed_at": "2026-07-14T00:00:00+08:00",
                "approved": True,
            },
            "captures": [
                {
                    "capture_set_path": "session-1/images/set-1",
                    "label": "normal",
                    "defect_type": "",
                    "annotations": {
                        view: {
                            "state": "confirmed_empty",
                            "source_path": f"labels/set-1/{view}.txt",
                            "text": "",
                        }
                        for view in views
                    },
                }
            ],
        }
    )


def _calibration_targets(
    topology: CaptureTopology,
    roi: RoiConfig,
    dataset_release_id: str,
) -> CalibrationTargetsSnapshot:
    targets = tuple(
        sorted(
            (
                CalibrationTargetRecord(
                    capture_set_id="set-1",
                    part_instance_id="part-1",
                    hand="right",
                    view=view,
                    branch=branch,
                    part_ground_truth="normal",
                    target="normal",
                    reason=None,
                )
                for view in topology.required_views
                for branch in ("template", "anomaly", "yolo")
            ),
            key=lambda item: item.key,
        )
    )
    return CalibrationTargetsSnapshot(
        dataset_release_id=dataset_release_id,
        topology_id=topology.topology_id,
        roi_version=roi.roi_config_id,
        approval=CalibrationTargetApproval(
            reviewed_by="quality-engineer-1",
            reviewed_at="2026-07-14T00:00:00+08:00",
            approved=True,
        ),
        targets=targets,
    )


@pytest.mark.parametrize(
    "approval",
    [
        {"reviewed_by": "qa-user-1", "reviewed_at": "2026-07-14T00:00:00Z", "approved": False},
        {"reviewed_by": "qa-user-1", "reviewed_at": "2026-07-14T00:00:00Z"},
        {
            "reviewed_by": "qa-user-1",
            "reviewed_at": "2026-07-14T00:00:00Z",
            "approved": True,
            "comment": "unbound field",
        },
        {"reviewed_by": "qa-user-1", "reviewed_at": "2026-07-14T00:00:00", "approved": True},
    ],
)
def test_semantics_approval_envelope_fails_closed(approval) -> None:
    with pytest.raises(ValueError, match="semantics"):
        CanonicalSemanticsSnapshot.from_mapping(
            {"schema_version": 2, "approval": approval, "captures": [{}]}
        )


def _contracts():
    rounds = (CaptureRound("front", "front prompt"), CaptureRound("back", "back prompt"))
    slots = tuple(
        CameraSlot(
            f"slot{index}",
            f"SERIAL{index}",
            {"front": f"front_{index}", "back": f"back_{index}"},
        )
        for index in range(3)
    )
    views = tuple([f"front_{index}" for index in range(3)] + [f"back_{index}" for index in range(3)])
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
            "required_views": list(views),
        }
    )
    topology = CaptureTopology(
        1,
        "topology",
        "ZS32",
        rounds,
        slots,
        views,
        topology_sha256,
    )
    roi_sha256 = canonical_sha256(
        {
            "schema_version": 2,
            "roi_version": "roi-v1",
            "product": "ZS32",
            "topology_id": topology.topology_id,
            "coordinate_system": "pixel_xyxy_half_open",
            "source_image_size": {"width": 100, "height": 100},
            "hands": {
                "left": {
                    "status": "pending",
                    "views": {},
                    "reason": "not measured",
                },
                "right": {
                    "status": "ready",
                    "views": {
                        view: {"xyxy": [0, 0, 100, 100]}
                        for view in views
                    },
                },
            },
        }
    )
    roi = RoiConfig(
        2,
        "roi-v1",
        "ZS32",
        topology.topology_id,
        "pixel_xyxy_half_open",
        100,
        100,
        {
            Hand.RIGHT: RoiHandConfig(
                Hand.RIGHT,
                RoiReadiness.READY,
                {view: RoiBox(0, 0, 100, 100) for view in views},
            ),
            Hand.LEFT: RoiHandConfig(
                Hand.LEFT,
                RoiReadiness.PENDING,
                {},
                "not measured",
            ),
        },
        roi_sha256,
    )
    return topology, roi


def _sources(
    topology: CaptureTopology,
    gate_publication: VerifiedCaptureGatePublication,
):
    part = PartIdentity("part-1", Hand.RIGHT)
    payloads = {
        view: b"\x89PNG\r\n\x1a\n" + f"png:{view}".encode()
        for view in topology.required_views
    }
    images = {}
    for view, payload in payloads.items():
        round_id, slot_id, serial = topology.binding_for_view(view)
        images[view] = ViewImage(
            view,
            round_id,
            slot_id,
            serial,
            f"images/set-1/{view}.png",
            hashlib.sha256(payload).hexdigest(),
            100,
            100,
        )
    capture = CaptureSet("set-1", "session-1", part, topology.topology_id, topology.topology_sha256, images)
    provenance = gate_publication.policy.provenance_for(
        Hand.RIGHT,
        policy_sha256=gate_publication.policy_sha256,
    )
    return tuple(
        SourceSample(
            capture,
            view,
            payloads[view],
            f"raw/session-1/images/set-1/{view}.png",
            "normal",
            "",
            LabelDocument("confirmed_empty", "", f"labels/set-1/{view}.txt"),
            provenance,
        )
        for view in topology.required_views
    )


def _recipe(topology, roi, policy_sha256: str):
    return parse_recipe(
        {
            "schema_version": 1,
            "recipe_id": "dataset-gate-authorization-v1",
            "product": "ZS32",
            "topology_id": topology.topology_id,
            "roi_version": roi.roi_config_id,
            "allowed_hands": ["right"],
            "anomaly_family": "patchcore",
            "capture_gate_policy": {
                "artifact_id": "zs32-right-gate-policy-v1",
                "version": "gate-policy-v1",
                "sha256": policy_sha256,
                "relative_path": "capture/gates/policy.json",
            },
            "template_assets": {},
            "anomaly_models": {},
            "yolo_model": {
                "artifact_id": "unbound-yolo",
                "version": "training-profile",
                "sha256": "0" * 64,
                "relative_path": "models/yolo/best.pt",
            },
            "template_thresholds": [],
            "model_thresholds": [],
            "fusion_policy_sha256": "9" * 64,
        }
    )


_QUALITY_PROFILE_BYTES = b'{"profile":"quality"}\n'
_REGISTRATION_PROFILE_BYTES = b'{"profile":"registration"}\n'
_ACQUISITION_CONFIG_BYTES = canonical_json_bytes(
    {
        "exposure": 4000.0, "gain": 0.0, "timeout_ms": 3000,
        "capture_interval": 0.2, "hdr": False,
        "short_exposure": 4000.0, "long_exposure": 35000.0,
        "hdr_settle_frames": 8, "align_hdr": True,
        "short_dark_threshold": 80.0, "long_clip_threshold": 245.0,
        "blend_width": 50.0, "blur_size": 101,
        "hdr_max_retries": 2, "hdr_max_clip_pct": 5.0,
        "png_compression": 3, "frame_buffer_size": 52428800,
    }
)


def _reference_bytes(view: str) -> bytes:
    return f"reference:{view}".encode()


def _gate_policy(topology: CaptureTopology) -> CaptureGatePolicy:
    references = {
        view: {
            "artifact_id": f"reference-right-{view}",
            "version": "gate-policy-v1",
            "sha256": hashlib.sha256(_reference_bytes(view)).hexdigest(),
            "relative_path": f"capture/gates/right/references/{view}.png",
        }
        for view in topology.required_views
    }
    return CaptureGatePolicy.from_mapping(
        {
            "schema_version": 2,
            "policy_id": "gate-policy-v1",
            "product": "ZS32",
            "topology_id": topology.topology_id,
            "topology_sha256": topology.topology_sha256,
            "acquisition_config": {
                "artifact_id": "hikvision-acquisition",
                "version": "gate-policy-v1",
                "sha256": hashlib.sha256(_ACQUISITION_CONFIG_BYTES).hexdigest(),
                "relative_path": "capture/acquisition/hikvision.json",
            },
            "hands": {
                "right": {
                    "quality_profile": {
                        "artifact_id": "quality-right",
                        "version": "gate-policy-v1",
                        "sha256": hashlib.sha256(_QUALITY_PROFILE_BYTES).hexdigest(),
                        "relative_path": "capture/gates/right/quality.json",
                    },
                    "registration_profile": {
                        "artifact_id": "registration-right",
                        "version": "gate-policy-v1",
                        "sha256": hashlib.sha256(_REGISTRATION_PROFILE_BYTES).hexdigest(),
                        "relative_path": "capture/gates/right/registration.json",
                    },
                    "registration_references": references,
                }
            },
        }
    )


def _gate_publication(
    tmp_path,
    topology: CaptureTopology,
) -> VerifiedCaptureGatePublication:
    policy = _gate_policy(topology)
    publication_id = "gate-publication-v1"
    output_root = tmp_path / "gate-publications"
    required = {"topology.json", "capture/gates/policy.json"}
    with AtomicDirectoryPublisher(output_root, publication_id) as publisher:
        publisher.write_bytes(
            "topology.json",
            canonical_json_bytes(topology.as_dict(include_sha256=False)),
        )
        publisher.write_bytes(
            "capture/gates/policy.json",
            canonical_json_bytes(policy.as_dict()),
        )
        publisher.write_bytes(
            policy.acquisition_config.relative_path,
            _ACQUISITION_CONFIG_BYTES,
        )
        required.add(policy.acquisition_config.relative_path)
        for hand_policy in policy.hands.values():
            publisher.write_bytes(
                hand_policy.quality_profile.relative_path,
                _QUALITY_PROFILE_BYTES,
            )
            publisher.write_bytes(
                hand_policy.registration_profile.relative_path,
                _REGISTRATION_PROFILE_BYTES,
            )
            required.update(
                {
                    hand_policy.quality_profile.relative_path,
                    hand_policy.registration_profile.relative_path,
                }
            )
            for view, reference in hand_policy.registration_references.items():
                publisher.write_bytes(reference.relative_path, _reference_bytes(view))
                required.add(reference.relative_path)
        publication_root = publisher.finalize(
            validator=lambda _staging: None,
            required_paths=frozenset(required),
        )
    return load_verified_capture_gate_publication(publication_root)


def test_all_three_adapter_manifests_reference_same_crop_hash(tmp_path) -> None:
    topology, roi = _contracts()
    gate_publication = _gate_publication(tmp_path, topology)
    spec = DatasetBuildSpec(
        "release-1",
        tmp_path / "releases",
        topology,
        roi,
        _recipe(topology, roi, gate_publication.policy_sha256),
        (Hand.RIGHT,),
        (SplitAssignment("part-1", "train", "right:normal:none"),),
        _split_policy(),
        _semantics_snapshot(),
        _png_codec(),
        "2026-07-14T00:00:00Z",
        gate_publication,
        _calibration_targets(topology, roi, "release-1"),
    )
    release = CanonicalDatasetBuilder(_IdentityCropper()).build(
        spec,
        _sources(topology, gate_publication),
    )
    verified = verify_dataset_release(release.path)
    assert verified.sample_count == 6
    assert len(release.samples) == 6
    assert set(verified.adapter_manifest_sha256) == {"yolo", "anomalib", "template"}
    assert verified.capture_gate_policy_sha256 == gate_publication.policy_sha256
    assert verified.schema_version == 4
    assert verified.split_policy == _split_policy().as_dict()
    assert verified.split_policy_sha256 == _split_policy().sha256
    assert verified.canonical_png_codec == _png_codec().as_dict()
    assert verified.canonical_png_codec_sha256 == _png_codec().sha256
    assert verified.canonical_semantics_sha256 == _semantics_snapshot().sha256
    targets = _calibration_targets(topology, roi, "release-1")
    assert verified.calibration_targets_sha256 == targets.sha256
    assert verified.calibration_target_count == 18
    for field, invalid in (
        ("sample_count", True),
        ("part_count", 1.0),
        ("capture_set_count", False),
        ("calibration_target_count", 18.0),
    ):
        with pytest.raises(ValueError, match="strict integers"):
            replace(verified, **{field: invalid})
    assert (release.path / "calibration_targets.json").read_bytes() == targets.canonical_bytes
    semantics = json.loads(
        (release.path / "canonical_semantics.json").read_text(encoding="utf-8")
    )
    assert semantics["approval"] == {
        "reviewed_by": "qa-user-1",
        "reviewed_at": "2026-07-14T00:00:00+08:00",
        "approved": True,
    }
    provenance = json.loads(
        (release.path / "dataset_provenance.json").read_text(encoding="utf-8")
    )
    split_artifact = json.loads(
        (release.path / "split_assignments.json").read_text(encoding="utf-8")
    )
    assert split_artifact["schema_version"] == 2
    assert split_artifact["policy"] == _split_policy().as_dict()
    assert split_artifact["policy_sha256"] == _split_policy().sha256
    assert provenance["split"]["policy"] == _split_policy().as_dict()
    assert provenance["canonical_png"]["codec"]["compression"] == 1
    assert provenance["canonical_semantics"]["sha256"] == _semantics_snapshot().sha256
    assert provenance["schema_version"] == 2
    assert provenance["calibration_targets"]["sha256"] == targets.sha256
    assert provenance["calibration_targets"]["approval"]["approved"] is True
    assert (release.path / "capture_provenance.jsonl").is_file()
    provenance_row = json.loads(
        (release.path / "capture_provenance.jsonl").read_text(encoding="utf-8")
    )
    assert provenance_row["acquisition_config_sha256"] == hashlib.sha256(
        _ACQUISITION_CONFIG_BYTES
    ).hexdigest()


def test_declared_png_codec_must_match_cropper(tmp_path) -> None:
    topology, roi = _contracts()
    gate_publication = _gate_publication(tmp_path, topology)
    spec = DatasetBuildSpec(
        "release-codec-mismatch",
        tmp_path / "releases",
        topology,
        roi,
        _recipe(topology, roi, gate_publication.policy_sha256),
        (Hand.RIGHT,),
        (SplitAssignment("part-1", "train", "right:normal:none"),),
        _split_policy(),
        _semantics_snapshot(),
        _png_codec(9),
        "2026-07-14T00:00:00Z",
        gate_publication,
        _calibration_targets(topology, roi, "release-codec-mismatch"),
    )
    with pytest.raises(ValueError, match="codec differs"):
        CanonicalDatasetBuilder(_IdentityCropper()).build(
            spec,
            _sources(topology, gate_publication),
        )


def test_semantics_snapshot_must_match_dataset_sources(tmp_path) -> None:
    topology, roi = _contracts()
    gate_publication = _gate_publication(tmp_path, topology)
    semantics_payload = json.loads(_semantics_snapshot().canonical_bytes)
    semantics_payload["captures"][0]["label"] = "defect"
    semantics_payload["captures"][0]["defect_type"] = "scratch"
    spec = DatasetBuildSpec(
        "release-semantics-mismatch",
        tmp_path / "releases",
        topology,
        roi,
        _recipe(topology, roi, gate_publication.policy_sha256),
        (Hand.RIGHT,),
        (SplitAssignment("part-1", "train", "right:normal:none"),),
        _split_policy(),
        CanonicalSemanticsSnapshot.from_mapping(semantics_payload),
        _png_codec(),
        "2026-07-14T00:00:00Z",
        gate_publication,
        _calibration_targets(topology, roi, "release-semantics-mismatch"),
    )
    with pytest.raises(ValueError, match="semantics differs"):
        CanonicalDatasetBuilder(_IdentityCropper()).build(
            spec,
            _sources(topology, gate_publication),
        )


def test_split_assignments_must_be_reproducible_from_policy(tmp_path) -> None:
    topology, roi = _contracts()
    gate_publication = _gate_publication(tmp_path, topology)
    spec = DatasetBuildSpec(
        "release-split-mismatch",
        tmp_path / "releases",
        topology,
        roi,
        _recipe(topology, roi, gate_publication.policy_sha256),
        (Hand.RIGHT,),
        (SplitAssignment("part-1", "train", "wrong-stratum"),),
        _split_policy(),
        _semantics_snapshot(),
        _png_codec(),
        "2026-07-14T00:00:00Z",
        gate_publication,
        _calibration_targets(topology, roi, "release-split-mismatch"),
    )
    with pytest.raises(ValueError, match="deterministic result"):
        CanonicalDatasetBuilder(_IdentityCropper()).build(
            spec,
            _sources(topology, gate_publication),
        )


@pytest.mark.parametrize(
    "field",
    ["acquisition_config_sha256", "quality_profile_sha256"],
)
def test_gate_provenance_not_derived_from_verified_policy_is_rejected(
    tmp_path, field: str
) -> None:
    topology, roi = _contracts()
    gate_publication = _gate_publication(tmp_path, topology)
    spec = DatasetBuildSpec(
        "release-mixed-gates",
        tmp_path / "releases",
        topology,
        roi,
        _recipe(topology, roi, gate_publication.policy_sha256),
        (Hand.RIGHT,),
        (SplitAssignment("part-1", "train", "right:normal:none"),),
        _split_policy(),
        _semantics_snapshot(),
        _png_codec(),
        "2026-07-14T00:00:00Z",
        gate_publication,
        _calibration_targets(topology, roi, "release-mixed-gates"),
    )
    sources = list(_sources(topology, gate_publication))
    sources[0] = replace(
        sources[0],
        gate_provenance=replace(
            sources[0].gate_provenance,
            **{field: "a" * 64},
        ),
    )
    with pytest.raises(ValueError, match="verified policy publication"):
        CanonicalDatasetBuilder(_IdentityCropper()).build(spec, sources)


def test_recipe_gate_policy_must_match_every_capture(tmp_path) -> None:
    topology, roi = _contracts()
    gate_publication = _gate_publication(tmp_path, topology)
    recipe_payload = _recipe(topology, roi, gate_publication.policy_sha256).as_dict()
    recipe_payload["capture_gate_policy"]["sha256"] = "b" * 64
    recipe = parse_recipe(recipe_payload)
    with pytest.raises(ValueError, match="verified capture gate policy differs"):
        DatasetBuildSpec(
            "release-wrong-policy",
            tmp_path / "releases",
            topology,
            roi,
            recipe,
            (Hand.RIGHT,),
            (SplitAssignment("part-1", "train", "right:normal:none"),),
            _split_policy(),
            _semantics_snapshot(),
            _png_codec(),
            "2026-07-14T00:00:00Z",
            gate_publication,
            _calibration_targets(topology, roi, "release-wrong-policy"),
        )


def test_left_pending_roi_blocks_left_dataset_release(tmp_path) -> None:
    topology, roi = _contracts()
    gate_publication = _gate_publication(tmp_path, topology)
    with pytest.raises(Exception, match="pending"):
        DatasetBuildSpec(
            "release-left",
            tmp_path / "releases",
            topology,
            roi,
            _recipe(topology, roi, gate_publication.policy_sha256),
            (Hand.LEFT,),
            (SplitAssignment("part-left", "train", "left:normal:none"),),
            _split_policy(),
            _semantics_snapshot(),
            _png_codec(),
            "2026-07-14T00:00:00Z",
            gate_publication,
            _calibration_targets(topology, roi, "release-left"),
        )
