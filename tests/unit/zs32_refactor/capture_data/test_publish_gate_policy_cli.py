"""Linux-only bootstrap gate-publication CLI contract."""

from __future__ import annotations

import base64
import hashlib
import json
import sys
from pathlib import Path

import pytest

from tests.unit.zs32_refactor.domain._fixtures import topology_mapping
from zs32_inspection.cli import publish_gate_policy
from zs32_inspection.capture import load_verified_capture_gate_publication
from zs32_inspection.config.schemas import parse_topology
from zs32_inspection.domain.contracts import canonical_sha256
from zs32_inspection.runtime.publisher import canonical_json_bytes, verify_atomic_publication


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="ZS32 verification is Linux-only")


def test_publish_gate_policy_creates_exact_atomic_bootstrap_input(tmp_path: Path) -> None:
    topology_payload = topology_mapping(3)
    topology = parse_topology(topology_payload)
    topology_path = tmp_path / "topology.json"
    topology_path.write_bytes(canonical_json_bytes(topology_payload))
    asset_root = tmp_path / "assets"
    acquisition = asset_root / "capture/acquisition/hikvision.json"
    quality = asset_root / "capture/gates/right/quality.json"
    registration = asset_root / "capture/gates/right/registration.json"
    acquisition.parent.mkdir(parents=True)
    acquisition.write_bytes(
        canonical_json_bytes(
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
    )
    quality.parent.mkdir(parents=True)
    quality_payload: dict[str, object] = {
        "schema_version": 1,
        "profile_id": "quality-right-v1",
        "product": "ZS32",
        "hand": "right",
        "topology_id": topology.topology_id,
        "topology_sha256": topology.topology_sha256,
        "views": {
            view: {
                "brightness_mean_min": 10.0,
                "brightness_mean_max": 245.0,
                "brightness_std_min": 1.0,
                "dark_pixel_level": 5,
                "dark_ratio_max": 0.5,
                "saturated_pixel_level": 250,
                "saturated_ratio_max": 0.5,
                "laplacian_variance_min": 1.0,
            }
            for view in topology.required_views
        },
    }
    quality_payload["profile_sha256"] = canonical_sha256(quality_payload)
    quality.write_bytes(canonical_json_bytes(quality_payload))
    references: dict[str, dict[str, str]] = {}
    registration_views: dict[str, object] = {}
    reference_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwC"
        "AAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    for view in topology.required_views:
        path = asset_root / f"capture/gates/right/references/{view}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(reference_png)
        references[view] = {
            "artifact_id": f"reference-{view}",
            "version": "v1",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "relative_path": path.relative_to(asset_root).as_posix(),
        }
        registration_views[view] = {
            "reference": {
                "asset_id": references[view]["artifact_id"],
                "relative_path": references[view]["relative_path"],
                "sha256": references[view]["sha256"],
                "width": 1,
                "height": 1,
            },
            "thresholds": {
                "analysis_scale": 0.5,
                "min_ecc_correlation": 0.5,
                "max_translation_x_px": 10.0,
                "max_translation_y_px": 10.0,
                "max_rotation_deg": 5.0,
                "max_scale_deviation": 0.1,
                "max_shear": 0.1,
                "max_iterations": 50,
                "epsilon": 0.0001,
                "gaussian_filter_size": 5,
            },
        }
    registration_payload: dict[str, object] = {
        "schema_version": 1,
        "profile_id": "registration-right-v1",
        "product": "ZS32",
        "hand": "right",
        "topology_id": topology.topology_id,
        "topology_sha256": topology.topology_sha256,
        "views": registration_views,
    }
    registration_payload["profile_sha256"] = canonical_sha256(registration_payload)
    registration.write_bytes(canonical_json_bytes(registration_payload))
    policy_path = tmp_path / "policy.json"
    policy_payload = {
        "schema_version": 2,
        "policy_id": "gate-right-v1",
        "product": "ZS32",
        "topology_id": topology.topology_id,
        "topology_sha256": topology.topology_sha256,
        "acquisition_config": {
            "artifact_id": "hikvision-acquisition",
            "version": "v1",
            "sha256": hashlib.sha256(acquisition.read_bytes()).hexdigest(),
            "relative_path": acquisition.relative_to(asset_root).as_posix(),
        },
        "hands": {
            "right": {
                "quality_profile": {
                    "artifact_id": "quality-right",
                    "version": "v1",
                    "sha256": hashlib.sha256(quality.read_bytes()).hexdigest(),
                    "relative_path": quality.relative_to(asset_root).as_posix(),
                },
                "registration_profile": {
                    "artifact_id": "registration-right",
                    "version": "v1",
                    "sha256": hashlib.sha256(registration.read_bytes()).hexdigest(),
                    "relative_path": registration.relative_to(asset_root).as_posix(),
                },
                "registration_references": references,
            }
        },
    }
    policy_path.write_text(
        json.dumps(policy_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    assert policy_path.read_bytes() != canonical_json_bytes(policy_payload)

    assert publish_gate_policy._run(
        [
            "--publication-id", "gate-right-v1",
            "--output-root", str(tmp_path / "publications"),
            "--topology", str(topology_path),
            "--policy", str(policy_path),
            "--asset-root", str(asset_root),
        ]
    ) == 0
    publication = verify_atomic_publication(tmp_path / "publications/gate-right-v1")
    verified = load_verified_capture_gate_publication(publication.root)
    assert verified.topology.topology_sha256 == topology.topology_sha256
    assert publication.checksums["capture/gates/policy.json"] == hashlib.sha256(
        canonical_json_bytes(policy_payload)
    ).hexdigest()
    assert publication.read_bytes("capture/gates/policy.json") == canonical_json_bytes(
        policy_payload
    )
    assert set(publication.checksums) == {
        "topology.json",
        "capture/gates/policy.json",
        acquisition.relative_to(asset_root).as_posix(),
        quality.relative_to(asset_root).as_posix(),
        registration.relative_to(asset_root).as_posix(),
        *(item["relative_path"] for item in references.values()),
    }
