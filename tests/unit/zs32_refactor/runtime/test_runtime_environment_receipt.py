"""Strict contracts for Linux/NVIDIA inference-environment receipts."""

from __future__ import annotations

import base64
import hashlib
import json
import stat
from dataclasses import replace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from zs32_inspection.cli import snapshot_environment
from zs32_inspection.runtime import environment_receipt as subject


def _receipt() -> subject.RuntimeEnvironmentReceipt:
    return subject.RuntimeEnvironmentReceipt(
        schema="zs32.runtime_environment_receipt",
        schema_version=2,
        platform="linux",
        architecture="x86_64",
        kernel_release="6.8.0-test",
        os_release_id="ubuntu",
        os_release_version_id="24.04",
        python_version="3.12.9",
        torch_version="2.7.1+cu128",
        torchvision_version="0.22.1+cu128",
        anomalib_version="2.4.3-dev.0",
        ultralytics_version="8.3.0",
        opencv_version="4.11.0",
        numpy_version="2.2.0",
        torch_cuda_version="12.8",
        cudnn_version=90701,
        nvidia_driver_version="570.124.06",
        device_mapping=subject.GpuDeviceMapping(
            requested_device="0",
            cuda_visible_devices="3",
            torch_device_index=0,
            nvidia_smi_index=3,
            gpu_uuid="GPU-test-uuid",
            gpu_name="NVIDIA RTX test",
        ),
        package_installations={
            role: subject.PackageInstallation(
                distribution_name=("opencv-python" if role == "opencv" else role),
                distribution_version="1.0",
                metadata_sha256=character * 64,
                record_sha256=character * 64,
                record_entry_count=1,
                installed_tree_sha256=character * 64,
                import_origin_relative_path=f"{role}/__init__.py",
                import_origin_sha256=character * 64,
            )
            for role, character in zip(
                ("torch", "torchvision", "anomalib", "ultralytics", "opencv", "numpy"),
                "abcdef",
                strict=True,
            )
        },
    )


def test_canonical_receipt_round_trip_and_digest(tmp_path: Path) -> None:
    receipt = _receipt()
    path = tmp_path / "runtime_environment.json"
    path.write_bytes(receipt.canonical_bytes())

    restored = subject.load_runtime_environment_receipt(path)

    assert restored == receipt
    assert restored.receipt_sha256 == hashlib.sha256(
        subject._canonical_json_bytes(restored.as_dict(include_receipt_sha256=False))
    ).hexdigest()


def test_receipt_rejects_unknown_fields_and_noncanonical_bytes(tmp_path: Path) -> None:
    receipt = _receipt()
    payload = receipt.as_dict()
    payload["unexpected"] = True
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(subject.RuntimeEnvironmentError, match="unknown fields"):
        subject.load_runtime_environment_receipt(path)

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(json.dumps(receipt.as_dict(), indent=2), encoding="utf-8")
    with pytest.raises(subject.RuntimeEnvironmentError, match="canonical JSON"):
        subject.load_runtime_environment_receipt(noncanonical)


def test_live_version_or_gpu_drift_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = _receipt()
    observed = replace(expected, nvidia_driver_version="571.0", receipt_sha256="")
    monkeypatch.setattr(subject, "collect_runtime_environment", lambda _device: observed)

    with pytest.raises(subject.RuntimeEnvironmentMismatchError, match="driver") as caught:
        subject.verify_runtime_environment(expected, device="0")

    assert caught.value.expected is expected
    assert caught.value.observed is observed
    assert any("nvidia_driver_version" in item for item in caught.value.differences)


def test_snapshot_publication_is_canonical_read_only_and_no_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt()
    monkeypatch.setattr(
        snapshot_environment,
        "collect_runtime_environment",
        lambda _device: receipt,
    )
    output = tmp_path / "runtime_environment.json"

    assert snapshot_environment._run(["--device", "0", "--output", str(output)]) == 0
    assert output.read_bytes() == receipt.canonical_bytes()
    assert stat.S_IMODE(output.stat().st_mode) == 0o444

    with pytest.raises(FileExistsError, match="already exists"):
        snapshot_environment._run(["--device", "0", "--output", str(output)])

    assert output.read_bytes() == receipt.canonical_bytes()


def test_distribution_record_drift_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = _receipt()
    packages = dict(expected.package_installations)
    packages["torch"] = replace(
        packages["torch"],
        record_sha256="1" * 64,
    )
    observed = replace(
        expected,
        package_installations=packages,
        receipt_sha256="",
    )
    monkeypatch.setattr(subject, "collect_runtime_environment", lambda _device: observed)

    with pytest.raises(subject.RuntimeEnvironmentMismatchError) as caught:
        subject.verify_runtime_environment(expected, device="0")

    assert any("package_installations" in item for item in caught.value.differences)


def test_installed_distribution_file_is_verified_against_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "fakepkg/__init__.py"
    metadata = tmp_path / "fakepkg-1.0.dist-info/METADATA"
    record = tmp_path / "fakepkg-1.0.dist-info/RECORD"
    package.parent.mkdir(parents=True)
    metadata.parent.mkdir(parents=True)
    package.write_bytes(b"__version__ = '1.0'\n")
    metadata.write_bytes(b"Name: fakepkg\nVersion: 1.0\n")

    def record_row(relative: str, content: bytes) -> str:
        encoded = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=")
        return f"{relative},sha256={encoded.decode('ascii')},{len(content)}\n"

    record.write_text(
        record_row("fakepkg/__init__.py", package.read_bytes())
        + record_row("fakepkg-1.0.dist-info/METADATA", metadata.read_bytes())
        + "fakepkg-1.0.dist-info/RECORD,,\n",
        encoding="utf-8",
    )

    class FakeDistribution:
        files = (
            PurePosixPath("fakepkg/__init__.py"),
            PurePosixPath("fakepkg-1.0.dist-info/METADATA"),
            PurePosixPath("fakepkg-1.0.dist-info/RECORD"),
        )
        metadata = {"Name": "fakepkg"}
        version = "1.0"

        @staticmethod
        def locate_file(relative):
            return tmp_path / str(relative)

    monkeypatch.setattr(subject.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(
        subject.importlib_metadata,
        "packages_distributions",
        lambda: {"fakepkg": ["fakepkg"]},
    )
    monkeypatch.setattr(
        subject.importlib_metadata,
        "distribution",
        lambda _name: FakeDistribution(),
    )
    installation = subject._package_installation(
        "fakepkg",
        SimpleNamespace(__file__=str(package)),
    )
    assert installation.import_origin_relative_path == "fakepkg/__init__.py"
    assert installation.import_origin_sha256 == hashlib.sha256(package.read_bytes()).hexdigest()

    package.write_bytes(b"tampered\n")
    with pytest.raises(subject.RuntimeEnvironmentError, match="differs from RECORD"):
        subject._package_installation(
            "fakepkg",
            SimpleNamespace(__file__=str(package)),
        )
