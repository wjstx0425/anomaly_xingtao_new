"""Linux-only unit tests for the ZS32 Phase 0 freezer."""

from __future__ import annotations

import binascii
import json
import platform
import shutil
import struct
import zlib
from pathlib import Path

import pytest

from tools import zs32_phase0 as phase0


def _has_linux_nvidia() -> bool:
    if platform.system() != "Linux" or shutil.which("nvidia-smi") is None:
        return False
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)


pytestmark = pytest.mark.skipif(
    not _has_linux_nvidia(),
    reason="ZS32 Phase 0 tests run only on Linux + NVIDIA",
)


REQUIRED_VIEWS = [
    "front",
    "front_left",
    "front_right",
    "back",
    "back_left",
    "back_right",
]


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_png_header(path: Path, width: int, height: int, marker: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    pixel = bytes([marker, marker, marker])
    scanlines = b"".join(b"\x00" + pixel * width for _ in range(height))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )
    return path


def _topology() -> dict[str, object]:
    return {
        "schema_version": 1,
        "topology_id": "test-3cam",
        "product": "ZS32",
        "rounds": [{"round_id": "front"}, {"round_id": "back"}],
        "camera_slots": [
            {
                "slot_id": "center",
                "serial": "A",
                "views": {"front": "front", "back": "back"},
            },
            {
                "slot_id": "left",
                "serial": "B",
                "views": {"front": "front_left", "back": "back_left"},
            },
            {
                "slot_id": "right",
                "serial": "C",
                "views": {"front": "front_right", "back": "back_right"},
            },
        ],
        "required_views": REQUIRED_VIEWS,
    }


def _roi(left_status: str = "pending") -> dict[str, object]:
    right_views = {view: {"xyxy": [0, 0, 10, 10]} for view in REQUIRED_VIEWS}
    left_views = {} if left_status == "pending" else right_views
    return {
        "schema_version": 2,
        "roi_version": "test-roi",
        "product": "ZS32",
        "topology_id": "test-3cam",
        "coordinate_system": "pixel_xyxy_half_open",
        "source_image_size": {"width": 10, "height": 10},
        "hands": {
            "right": {"status": "ready", "views": right_views},
            "left": {"status": left_status, "views": left_views},
        },
    }


def test_platform_guard_rejects_macos_without_bypass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(phase0.platform, "system", lambda: "Darwin")

    with pytest.raises(phase0.Phase0Error, match=r"Linux \+ NVIDIA only"):
        phase0.require_linux_nvidia()


def test_topology_mapping_must_equal_required_views(tmp_path: Path) -> None:
    topology = _topology()
    topology["required_views"] = [*REQUIRED_VIEWS, "unexpected"]
    path = _write_json(tmp_path / "topology.json", topology)

    with pytest.raises(phase0.Phase0Error, match="required_views"):
        phase0._validate_topology(path)


def test_right_roi_is_ready_while_left_remains_pending(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "roi.json", _roi())

    phase0._validate_roi(path, ["right"], REQUIRED_VIEWS)
    with pytest.raises(phase0.Phase0Error, match="ROI_PENDING"):
        phase0._validate_roi(path, ["left"], REQUIRED_VIEWS)


def test_pending_roi_cannot_contain_fallback_coordinates(tmp_path: Path) -> None:
    roi = _roi()
    roi["hands"]["left"]["views"] = roi["hands"]["right"]["views"]
    path = _write_json(tmp_path / "roi.json", roi)

    with pytest.raises(phase0.Phase0Error, match="pending ROI hand left"):
        phase0._validate_roi(path, ["right"], REQUIRED_VIEWS)


def test_roi_requires_explicit_left_pending_entry(tmp_path: Path) -> None:
    roi = _roi()
    del roi["hands"]["left"]
    path = _write_json(tmp_path / "roi.json", roi)

    with pytest.raises(phase0.Phase0Error, match="exactly left and right"):
        phase0._validate_roi(path, ["right"], REQUIRED_VIEWS)


def test_roi_is_half_open_and_bounded(tmp_path: Path) -> None:
    roi = _roi()
    roi["hands"]["right"]["views"]["front"] = {"xyxy": [5, 5, 5, 10]}
    path = _write_json(tmp_path / "roi.json", roi)

    with pytest.raises(phase0.Phase0Error, match="empty or outside"):
        phase0._validate_roi(path, ["right"], REQUIRED_VIEWS)


def test_legacy_golden_requires_signed_manual_attestation(tmp_path: Path) -> None:
    images = {
        view: str(_write_png_header(tmp_path / f"{view}.png", 10, 10, index))
        for index, view in enumerate(REQUIRED_VIEWS, start=1)
    }
    selection = {
        "schema_version": 1,
        "selection_id": "test-selection",
        "product": "ZS32",
        "usage": "refactor_parity_only",
        "cases": [
            {
                "case_id": "case-1",
                "hand": "right",
                "label": "normal",
                "scenario": "normal_clear",
                "expected_target_status": "OK",
                "capture_identity": {
                    "session_id": "session",
                    "sample_id": "sample",
                    "group_id": "group001",
                    "part_instance_id": "part",
                },
                "provenance": {
                    "mode": "legacy_directory_without_capture_manifest",
                    "physical_part_identity_verified": False,
                    "label_verified": True,
                    "verified_by": "operator",
                    "verified_at": "2020-01-01T00:00:00+08:00",
                },
                "parity": {"mode": "semantic", "reason": "normal parity"},
                "images": images,
            }
        ],
    }
    path = _write_json(tmp_path / "selection.json", selection)

    with pytest.raises(phase0.Phase0Error, match="physical part identity"):
        phase0._golden_inventory(path, ["right"], REQUIRED_VIEWS, (10, 10))


def test_golden_requires_exact_view_set(tmp_path: Path) -> None:
    images = {
        view: str(_write_png_header(tmp_path / f"{view}.png", 10, 10, index))
        for index, view in enumerate(REQUIRED_VIEWS[:-1], start=1)
    }
    selection = {
        "schema_version": 1,
        "selection_id": "test-selection",
        "product": "ZS32",
        "usage": "refactor_parity_only",
        "cases": [
            {
                "case_id": "case-1",
                "hand": "right",
                "label": "normal",
                "scenario": "normal_clear",
                "expected_target_status": "OK",
                "capture_identity": {
                    "session_id": "session",
                    "sample_id": "sample",
                    "group_id": "group001",
                    "part_instance_id": "part",
                },
                "provenance": {
                    "mode": "legacy_directory_without_capture_manifest",
                    "physical_part_identity_verified": True,
                    "label_verified": True,
                    "verified_by": "operator",
                    "verified_at": "2020-01-01T00:00:00+08:00",
                },
                "parity": {"mode": "semantic", "reason": "normal parity"},
                "images": images,
            }
        ],
    }
    path = _write_json(tmp_path / "selection.json", selection)

    with pytest.raises(phase0.Phase0Error, match="exactly every required view"):
        phase0._golden_inventory(path, ["right"], REQUIRED_VIEWS, (10, 10))


def test_yolo_deployment_asset_must_be_best_pt(tmp_path: Path) -> None:
    init_weight = tmp_path / "yolo26n.pt"
    init_weight.write_bytes(b"init")
    spec = {
        "assets": [
            {
                "asset_id": "yolo",
                "kind": "yolo_checkpoint",
                "role": "deployment_weight",
                "path": str(init_weight),
                "required": True,
            }
        ]
    }

    with pytest.raises(phase0.Phase0Error, match="trained best.pt"):
        phase0._asset_inventory(spec, tmp_path)


def test_directory_digest_rejects_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "asset"
    root.mkdir()
    target = root / "target.bin"
    target.write_bytes(b"asset")
    (root / "link.bin").symlink_to(target)

    with pytest.raises(phase0.Phase0Error, match="symlinks are forbidden"):
        phase0._path_digest(root)


def test_copy_and_verify_rejects_unexpected_content(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"current")

    with pytest.raises(phase0.Phase0Error, match="copied input hash mismatch"):
        phase0._copy_and_verify(source, destination, "0" * 64)


def test_freeze_publication_rename_is_atomic_and_never_replaces(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "frozen"
    destination.mkdir()
    marker = destination / "existing.txt"
    marker.write_text("trusted\n", encoding="utf-8")
    source = tmp_path / "staging"
    source.mkdir()
    (source / "replacement.txt").write_text("untrusted\n", encoding="utf-8")

    with pytest.raises(phase0.Phase0Error, match="no-replace freeze publication failed"):
        phase0._rename_directory_noreplace(source, destination)

    assert marker.read_text(encoding="utf-8") == "trusted\n"
    assert source.is_dir()


def _trusted_pointer(root: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "freeze_id": "freeze-1",
        "bundle_root_sha256": root,
        "git_commit": "commit-1",
        "scope": {
            "product": "ZS32",
            "hands": ["right"],
            "topology_id": "topology-1",
            "left_roi_status": "pending",
        },
        "storage_reference": f"s3://bucket/zs32/sha256/{root}",
    }


def test_trusted_pointer_requires_exact_schema_and_content_address() -> None:
    root = "a" * 64
    pointer = _trusted_pointer(root)
    phase0._validate_trusted_pointer(
        pointer,
        freeze_id="freeze-1",
        expected_root=root,
        git_commit="commit-1",
        topology_id="topology-1",
    )

    pointer["unexpected"] = True
    with pytest.raises(phase0.Phase0Error, match="does not match"):
        phase0._validate_trusted_pointer(
            pointer,
            freeze_id="freeze-1",
            expected_root=root,
            git_commit="commit-1",
            topology_id="topology-1",
        )
    pointer = _trusted_pointer(root)
    pointer["storage_reference"] = "s3://bucket/zs32/current"
    with pytest.raises(phase0.Phase0Error, match="content address"):
        phase0._validate_trusted_pointer(
            pointer,
            freeze_id="freeze-1",
            expected_root=root,
            git_commit="commit-1",
            topology_id="topology-1",
        )


def test_arbitrary_checksum_directory_cannot_impersonate_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    artifact = bundle / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")
    root = phase0._write_checksums(bundle)
    monkeypatch.setattr(phase0, "require_linux_nvidia", lambda: {"platform": "Linux"})

    with pytest.raises(phase0.Phase0Error, match="freeze_manifest"):
        phase0.verify(bundle, expected_root=root)


def test_joint_rehash_still_fails_trusted_external_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    artifact = bundle / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")
    trusted_root = phase0._write_checksums(bundle)
    artifact.write_text('{"tampered": true}\n', encoding="utf-8")
    (bundle / "checksums.sha256").unlink()
    (bundle / "bundle_root.json").unlink()
    phase0._write_checksums(bundle)
    monkeypatch.setattr(phase0, "require_linux_nvidia", lambda: {"platform": "Linux"})

    with pytest.raises(phase0.Phase0Error, match="trusted external root"):
        phase0.verify(bundle, expected_root=trusted_root)


def test_freeze_refuses_unreviewed_draft_before_any_runtime_probe(tmp_path: Path) -> None:
    spec = {
        "schema_version": 1,
        "freeze_id": "draft-freeze",
        "product": "ZS32",
        "status": "draft",
        "expected_git": {
            "remote_name": "origin",
            "remote_url": "git@github.com:wjstx0425/anomaly_xingtao_new.git",
            "branch": "main",
        },
        "output_root": str(tmp_path / "output"),
        "scope": {
            "hands": ["right"],
            "usage": "refactor_parity_only",
        },
    }
    path = _write_json(tmp_path / "spec.json", spec)

    with pytest.raises(phase0.Phase0Error, match="status must be changed"):
        phase0.freeze(path)


def _write_baseline_output(
    root: Path,
    *,
    run_id: str,
    parity_mode: str = "semantic",
    observed_status: str = "NG_TEMPLATE",
    omitted_evidence: str | None = None,
) -> Path:
    root.mkdir(parents=True)
    execution_id = f"case-1-{run_id}"
    executed_at = f"2020-01-01T00:00:{int(run_id[-2:]):02d}+08:00"
    summary_path = _write_json(
        root / "legacy_output" / "runtime_summary.json",
        {
            "machine_status": "NG_TEMPLATE",
            "inspection_complete": False,
            "short_circuited": True,
            "part_id": "part",
            "evaluated_views": ["front"],
            "template_results": [{"view": "front", "status": "NG_TEMPLATE"}],
        },
    )
    template_rows = [{"view": "front", "status": "NG_TEMPLATE", "evidence_level": "STRONG"}]
    not_run = {"state": "NOT_RUN", "reason": "template gate short-circuited Stage 32"}
    evidence_groups = {
        "template": {"front": {"status": "NG_TEMPLATE", "evidence_level": "STRONG"}},
        "anomaly": not_run,
        "yolo": not_run,
    }
    payloads = {
        "runtime_summary.json": phase0._replay_artifact_payload(
            summary_path,
            root,
            machine_status="NG_TEMPLATE",
            inspection_complete=False,
            execution_state="SHORT_CIRCUITED",
        ),
        "template_evidence.json": phase0._replay_artifact_payload(
            summary_path,
            root,
            rows=template_rows,
        ),
        "anomaly_evidence.json": not_run,
        "yolo_evidence.json": not_run,
        "fusion_result.json": phase0._replay_artifact_payload(
            summary_path,
            root,
            final_status="NG_TEMPLATE",
            source="template_short_circuit",
        ),
        "audit.json": {
            "state": "NOT_PRODUCED",
            "reason": "template gate short-circuited before strict fusion audit",
        },
    }
    for name, payload in payloads.items():
        if name != omitted_evidence:
            phase0._write_replay_envelope(
                root / name,
                case_id="case-1",
                run_id=run_id,
                execution_id=execution_id,
                executed_at=executed_at,
                payload=payload,
            )
    (root / "stdout.log").write_text("", encoding="utf-8")
    (root / "stderr.log").write_text("", encoding="utf-8")
    if omitted_evidence != "phase0_case_result.json":
        _write_json(
            root / "phase0_case_result.json",
            {
                "schema_version": 1,
                "case_id": "case-1",
                "run_id": run_id,
                "execution_id": execution_id,
                "executed_at": executed_at,
                "command_argv": phase0._legacy_replay_command(
                    capture_identity=_source_binding()["capture_identity"],
                    hand="right",
                    required_views=["front"],
                    images={"front": "golden_inputs/front.png"},
                    runtime_inputs={
                        "runtime_config": "runtime_inputs/runtime_models.json",
                        "template_model_dir": "runtime_inputs/templates",
                        "threshold_artifact": "runtime_inputs/thresholds.json",
                    },
                    accelerator="gpu",
                    devices=1,
                    yolo_device="0",
                ),
                "exit_code": 0,
                "observed_legacy": {
                    "final_status": observed_status,
                    "evidence_groups": evidence_groups,
                },
                "target_contract_expected": {
                    "final_status": "NG_TEMPLATE",
                    "reason": "template mismatch must stop before layer two",
                },
                "source_binding": _source_binding(),
                "parity": {"mode": parity_mode},
            },
        )
    if omitted_evidence != "replay_root.json":
        phase0._write_replay_root(root)
    return root


def _baseline_spec(*paths: Path, review_decision: str = "APPROVED") -> dict[str, object]:
    semantic_payload = {
        "final_status": "NG_TEMPLATE",
        "evidence_groups": {
            "template": {"front": {"status": "NG_TEMPLATE", "evidence_level": "STRONG"}},
            "anomaly": {"state": "NOT_RUN", "reason": "template gate short-circuited Stage 32"},
            "yolo": {"state": "NOT_RUN", "reason": "template gate short-circuited Stage 32"},
        },
    }
    review_path = paths[0].parent / "baseline-review.json"
    _write_json(
        review_path,
        {
            "schema_version": 1,
            "product": "ZS32",
            "case_id": "case-1",
            "reviewed_at": "2020-01-01T01:00:00+08:00",
            "reviewer": "phase0-reviewer",
            "decision": review_decision,
            "replays": [
                {
                    "run_id": f"replay-{index:02d}",
                    "replay_root_sha256": phase0._load_json(path / "replay_root.json")[
                        "replay_root_sha256"
                    ],
                    "semantic_signature": phase0._canonical_sha256(semantic_payload),
                }
                for index, path in enumerate(paths, start=1)
            ],
            "score_comparison": {
                "decision": "NOT_APPLICABLE",
                "basis": "template short circuit has no layer-two continuous score comparison",
            },
        },
    )
    return {
        "minimum_replays_per_case": 2,
        "baseline_outputs": [
            {
                "case_id": "case-1",
                "run_id": f"replay-{index:02d}",
                "path": str(path),
                "required": True,
            }
            for index, path in enumerate(paths, start=1)
        ],
        "baseline_reviews": [{"case_id": "case-1", "path": str(review_path)}],
    }


def _expected_case() -> dict[str, dict[str, object]]:
    return {
        "case-1": {
            "target_status": "NG_TEMPLATE",
            "parity_mode": "semantic",
            "source_binding": _source_binding(),
            "required_views": ["front"],
            "scenario": "template_mismatch",
            "replay_runtime": {
                "accelerator": "gpu",
                "devices": 1,
                "yolo_device": "0",
            },
        }
    }


def _source_binding() -> dict[str, object]:
    return {
        "git_commit": "a" * 40,
        "git_tree": "b" * 40,
        "uv_lock_sha256": "c" * 64,
        "topology_sha256": "d" * 64,
        "roi_sha256": "e" * 64,
        "environment_sha256": "2" * 64,
        "asset_sha256_by_id": {"weight": "f" * 64},
        "capture_identity": {
            "session_id": "session",
            "sample_id": "sample",
            "group_id": "group001",
            "part_instance_id": "part",
        },
        "golden_image_sha256_by_view": {"front": "1" * 64},
    }


def test_baseline_requires_all_named_evidence_files(tmp_path: Path) -> None:
    replay = _write_baseline_output(
        tmp_path / "replay-01",
        run_id="replay-01",
        omitted_evidence="audit.json",
    )

    with pytest.raises(phase0.Phase0Error, match="lacks required evidence"):
        phase0._baseline_inventory(_baseline_spec(replay), tmp_path, _expected_case())


def test_baseline_parity_mode_must_match_golden_selection(tmp_path: Path) -> None:
    first = _write_baseline_output(
        tmp_path / "replay-01",
        run_id="replay-01",
        parity_mode="exact",
    )
    second = _write_baseline_output(
        tmp_path / "replay-02",
        run_id="replay-02",
        parity_mode="exact",
    )

    with pytest.raises(phase0.Phase0Error, match="parity mode disagrees"):
        phase0._baseline_inventory(_baseline_spec(first, second), tmp_path, _expected_case())


def test_baseline_requires_external_approved_review_receipt(tmp_path: Path) -> None:
    first = _write_baseline_output(tmp_path / "replay-01", run_id="replay-01")
    second = _write_baseline_output(tmp_path / "replay-02", run_id="replay-02")

    with pytest.raises(phase0.Phase0Error, match="not approved"):
        phase0._baseline_inventory(
            _baseline_spec(first, second, review_decision="REJECTED"),
            tmp_path,
            _expected_case(),
        )


def test_baseline_rejects_placeholder_observed_status(tmp_path: Path) -> None:
    first = _write_baseline_output(
        tmp_path / "replay-01",
        run_id="replay-01",
        observed_status="REPLACE_WITH_OBSERVED_STATUS",
    )
    second = _write_baseline_output(tmp_path / "replay-02", run_id="replay-02")

    with pytest.raises(phase0.Phase0Error, match="observed status is unsupported"):
        phase0._baseline_inventory(_baseline_spec(first, second), tmp_path, _expected_case())


def test_baseline_rejects_unknown_observed_status(tmp_path: Path) -> None:
    first = _write_baseline_output(
        tmp_path / "replay-01",
        run_id="replay-01",
        observed_status="BANANA",
    )
    second = _write_baseline_output(tmp_path / "replay-02", run_id="replay-02")

    with pytest.raises(phase0.Phase0Error, match="observed status is unsupported"):
        phase0._baseline_inventory(_baseline_spec(first, second), tmp_path, _expected_case())


def test_baseline_evidence_envelope_must_bind_execution_identity(tmp_path: Path) -> None:
    first = _write_baseline_output(tmp_path / "replay-01", run_id="replay-01")
    second = _write_baseline_output(tmp_path / "replay-02", run_id="replay-02")
    evidence = json.loads((first / "audit.json").read_text(encoding="utf-8"))
    evidence["execution_id"] = "another-execution"
    (first / "audit.json").write_text(json.dumps(evidence), encoding="utf-8")
    (first / "replay_root.json").unlink()
    phase0._write_replay_root(first)

    with pytest.raises(phase0.Phase0Error, match="evidence identity differs"):
        phase0._baseline_inventory(_baseline_spec(first, second), tmp_path, _expected_case())


def test_baseline_result_must_bind_frozen_sources(tmp_path: Path) -> None:
    first = _write_baseline_output(tmp_path / "replay-01", run_id="replay-01")
    second = _write_baseline_output(tmp_path / "replay-02", run_id="replay-02")
    result = json.loads((first / "phase0_case_result.json").read_text(encoding="utf-8"))
    result["source_binding"]["git_commit"] = "9" * 40
    (first / "phase0_case_result.json").write_text(json.dumps(result), encoding="utf-8")
    (first / "replay_root.json").unlink()
    phase0._write_replay_root(first)

    with pytest.raises(phase0.Phase0Error, match="source binding differs"):
        phase0._baseline_inventory(_baseline_spec(first, second), tmp_path, _expected_case())


def test_baseline_replay_root_detects_post_publication_edit(tmp_path: Path) -> None:
    first = _write_baseline_output(tmp_path / "replay-01", run_id="replay-01")
    second = _write_baseline_output(tmp_path / "replay-02", run_id="replay-02")
    spec = _baseline_spec(first, second)
    (first / "stdout.log").write_text("post-publication edit", encoding="utf-8")

    with pytest.raises(phase0.Phase0Error, match="changed after publication"):
        phase0._baseline_inventory(spec, tmp_path, _expected_case())


def test_baseline_review_must_bind_exact_replay_roots(tmp_path: Path) -> None:
    first = _write_baseline_output(tmp_path / "replay-01", run_id="replay-01")
    second = _write_baseline_output(tmp_path / "replay-02", run_id="replay-02")
    spec = _baseline_spec(first, second)
    review_path = Path(spec["baseline_reviews"][0]["path"])
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["replays"][0]["replay_root_sha256"] = "0" * 64
    review_path.write_text(json.dumps(review), encoding="utf-8")

    with pytest.raises(phase0.Phase0Error, match="does not bind the exact immutable replays"):
        phase0._baseline_inventory(spec, tmp_path, _expected_case())


def test_normative_phase0_requirements_cannot_be_deleted() -> None:
    spec = {
        "required_asset_roles": [],
        "required_golden_scenarios": [],
        "known_blockers": [],
    }

    with pytest.raises(phase0.Phase0Error, match="normative Phase 0 role set"):
        phase0._validate_normative_spec_requirements(
            spec,
            require_resolved_blockers=False,
        )


def test_review_subject_allows_only_status_and_approval_changes() -> None:
    draft = {
        "schema_version": 1,
        "freeze_id": "freeze-1",
        "status": "draft",
        "approval": {"approved": False},
        "assets": [{"asset_id": "weight", "expected_sha256": "a" * 64}],
    }
    ready = json.loads(json.dumps(draft))
    ready["status"] = "ready"
    ready["approval"] = {"approved": True, "approved_by": "operator"}

    assert phase0._freeze_review_subject_sha256(draft) == phase0._freeze_review_subject_sha256(
        ready
    )

    ready["assets"][0]["expected_sha256"] = "b" * 64
    assert phase0._freeze_review_subject_sha256(draft) != phase0._freeze_review_subject_sha256(
        ready
    )


def test_manual_timestamp_requires_timezone() -> None:
    with pytest.raises(phase0.Phase0Error, match="explicit timezone"):
        phase0._signed_timestamp("2026-07-14T00:00:00", "verified_at")


def test_phase0_json_outputs_are_not_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "inventory.json"
    phase0._write_json(output, {"version": 1})

    with pytest.raises(phase0.Phase0Error, match="refusing to overwrite"):
        phase0._write_json(output, {"version": 2})


def test_replay_plan_binds_real_case_local_gate_csvs(tmp_path: Path) -> None:
    case_id = "right-normal-001"
    part_id = "physical-part-001"
    gate_assets: list[dict[str, object]] = []
    role_by_gate = {
        "quality": "legacy_quality_evidence",
        "registration": "legacy_registration_evidence",
        "geometry": "legacy_geometry_evidence",
    }
    for gate, role in role_by_gate.items():
        path = tmp_path / f"{gate}.csv"
        path.write_text(
            "part_id,view,branch,status\n"
            + "".join(
                f"{part_id},{view},{'quality_gate' if gate == 'quality' else gate},PASS\n"
                for view in REQUIRED_VIEWS
            ),
            encoding="utf-8",
        )
        gate_assets.append(
            {
                "asset_id": f"{case_id}-{gate}",
                "kind": "branch_evidence_csv",
                "role": role,
                "source_path": str(path),
                "required": True,
                "present": True,
                "copy_into_bundle": True,
                "sha256": phase0._sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    supporting_assets: list[dict[str, object]] = []
    producer_repository = tmp_path / "gate-producer-source"
    producer_repository.mkdir()
    phase0._run(["git", "init"], cwd=producer_repository)
    (producer_repository / "legacy_gate_producer.py").write_text("# frozen producer\n", encoding="utf-8")
    phase0._run(["git", "add", "legacy_gate_producer.py"], cwd=producer_repository)
    phase0._run(
        [
            "git",
            "-c",
            "user.name=Phase0 Test",
            "-c",
            "user.email=phase0@example.invalid",
            "commit",
            "-m",
            "freeze producer",
        ],
        cwd=producer_repository,
    )
    producer_commit = phase0._run(["git", "rev-parse", "HEAD"], cwd=producer_repository)
    producer_bundle = tmp_path / "gate-producer-source.bundle"
    phase0._run(
        ["git", "bundle", "create", str(producer_bundle), "HEAD"],
        cwd=producer_repository,
    )
    supporting_assets.append(
        {
            "asset_id": "gate-producer-source",
            "kind": "git_source_bundle",
            "role": "legacy_gate_producer_source",
            "source_path": str(producer_bundle),
            "required": True,
            "present": True,
            "copy_into_bundle": True,
            "sha256": phase0._sha256_file(producer_bundle),
        }
    )
    for asset_id, role in (
        ("gate-producer-config", "capture_acquisition_config"),
        ("gate-producer-threshold", "legacy_thresholds"),
    ):
        path = tmp_path / f"{asset_id}.bin"
        path.write_bytes(asset_id.encode())
        supporting_assets.append(
            {
                "asset_id": asset_id,
                "kind": "producer_input",
                "role": role,
                "source_path": str(path),
                "required": True,
                "present": True,
                "copy_into_bundle": True,
                "sha256": phase0._sha256_file(path),
            }
        )
    capture_identity = {
        "session_id": "session-001",
        "sample_id": "sample-001",
        "group_id": "group-001",
        "part_instance_id": part_id,
    }
    source_hashes = {view: f"{index:064x}" for index, view in enumerate(REQUIRED_VIEWS, start=1)}
    receipt_path = _write_json(
        tmp_path / "gate-producer-receipt.json",
        {
            "schema_version": 1,
            "product": "ZS32",
            "case_id": case_id,
            "generated_at": "2020-01-01T00:00:00Z",
            "capture_identity": capture_identity,
            "producer": {
                "command_argv": ["python", "legacy_gate_producer.py"],
                "git_commit": producer_commit,
                "source_asset_id": "gate-producer-source",
                "source_asset_sha256": supporting_assets[0]["sha256"],
                "config_assets": [
                    {
                        "asset_id": "gate-producer-config",
                        "sha256": supporting_assets[1]["sha256"],
                    }
                ],
                "threshold_assets": [
                    {
                        "asset_id": "gate-producer-threshold",
                        "sha256": supporting_assets[2]["sha256"],
                    }
                ],
            },
            "source_image_sha256_by_view": source_hashes,
            "outputs": {
                gate: {
                    "asset_id": asset["asset_id"],
                    "sha256": asset["sha256"],
                }
                for gate, asset in zip(role_by_gate, gate_assets, strict=True)
            },
            "attestation": {
                "reviewed_by": "reviewer",
                "reviewed_at": "2020-01-02T00:00:00Z",
                "basis": "bound historical producer logs",
            },
        },
    )
    receipt_asset = {
        "asset_id": f"{case_id}-gate-receipt",
        "kind": "producer_receipt",
        "role": "legacy_gate_producer_receipt",
        "source_path": str(receipt_path),
        "required": True,
        "present": True,
        "copy_into_bundle": True,
        "sha256": phase0._sha256_file(receipt_path),
    }
    plan_path = _write_json(
        tmp_path / "replay-plan.json",
        {
            "schema_version": 1,
            "product": "ZS32",
            "entrypoint": phase0.LEGACY_REPLAY_ENTRYPOINT,
            "accelerator": "gpu",
            "devices": 1,
            "yolo_device": "0",
            "cases": [
                {
                    "case_id": case_id,
                    "quality_asset_id": f"{case_id}-quality",
                    "registration_asset_id": f"{case_id}-registration",
                    "geometry_asset_id": f"{case_id}-geometry",
                    "gate_producer_receipt_asset_id": receipt_asset["asset_id"],
                }
            ],
        },
    )
    assets = [
        {
            "asset_id": "replay-plan",
            "kind": "replay_plan",
            "role": "legacy_replay_plan",
            "source_path": str(plan_path),
            "required": True,
            "present": True,
            "copy_into_bundle": True,
            "sha256": phase0._sha256_file(plan_path),
        },
        *gate_assets,
        *supporting_assets,
        receipt_asset,
    ]
    golden = [
        {
            "case_id": case_id,
            "scenario": "normal_clear",
            "capture_identity": capture_identity,
            "images": [
                {"view": view, "sha256": source_hashes[view]}
                for view in REQUIRED_VIEWS
            ],
        }
    ]

    inventory = phase0._validate_legacy_replay_plan(assets, golden, REQUIRED_VIEWS)

    assert inventory["entrypoint"] == phase0.LEGACY_REPLAY_ENTRYPOINT
    assert set(inventory["cases"][0]["inputs"]) == {"quality", "registration", "geometry"}


def test_replay_gate_csv_rejects_missing_view(tmp_path: Path) -> None:
    path = tmp_path / "quality.csv"
    path.write_text(
        "part_id,view,branch,status\n"
        + "".join(f"part-1,{view},quality_gate,PASS\n" for view in REQUIRED_VIEWS[:-1]),
        encoding="utf-8",
    )

    with pytest.raises(phase0.Phase0Error, match="exactly 6 rows"):
        phase0._validate_legacy_gate_csv(
            path,
            case_id="case-1",
            part_id="part-1",
            gate="quality",
            required_views=REQUIRED_VIEWS,
        )


def test_normative_anomaly_scenario_cannot_be_claimed_after_template_short_circuit() -> None:
    observed = {
        "final_status": "NG_TEMPLATE",
        "evidence_groups": {
            "template": {
                "front": {"status": "NG_TEMPLATE", "evidence_level": "STRONG"}
            },
            "anomaly": {"state": "NOT_RUN", "reason": "template short circuit"},
            "yolo": {"state": "NOT_RUN", "reason": "template short circuit"},
        },
    }

    with pytest.raises(phase0.Phase0Error, match="did not actually exercise"):
        phase0._validate_normative_scenario_observed(
            "anomaly_strong",
            observed,
            case_id="case-false-anomaly",
        )


def test_replay_adapter_derives_levels_from_locked_dual_thresholds(tmp_path: Path) -> None:
    path = tmp_path / "patchcore.csv"
    scores = ("0.1", "0.3", "0.7", "0.1", "0.3", "0.7")
    path.write_text(
        "view,branch,status,pred_label,score,low_threshold,high_threshold,evidence_level\n"
        + "".join(
            f"{view},anomaly_{view},,0,{score},0.3,0.7,\n"
            for view, score in zip(REQUIRED_VIEWS, scores, strict=True)
        ),
        encoding="utf-8",
    )

    rows, semantic = phase0._read_replay_csv_summary(
        path,
        required_views=REQUIRED_VIEWS,
        branch_group="anomaly",
    )

    assert [row["evidence_level"] for row in rows] == [
        "CLEAR",
        "GRAY",
        "STRONG",
        "CLEAR",
        "GRAY",
        "STRONG",
    ]
    assert semantic["front_right"] == {"evidence_level": "STRONG"}


def test_replay_adapter_rejects_declared_level_that_disagrees_with_thresholds(tmp_path: Path) -> None:
    path = tmp_path / "yolo.csv"
    path.write_text(
        "view,branch,score,low_threshold,high_threshold,evidence_level\n"
        + "".join(f"{view},yolo,0.1,0.3,0.7,STRONG\n" for view in REQUIRED_VIEWS),
        encoding="utf-8",
    )

    with pytest.raises(phase0.Phase0Error, match="disagrees with locked dual thresholds"):
        phase0._read_replay_csv_summary(
            path,
            required_views=REQUIRED_VIEWS,
            branch_group="yolo",
        )
