"""Deterministic execution provenance used by Linux-authoritative unit contracts."""

from __future__ import annotations

from collections.abc import Mapping

from zs32_inspection.runtime.environment_receipt import (
    GpuDeviceMapping,
    PackageInstallation,
    RuntimeEnvironmentReceipt,
)
from zs32_inspection.runtime.execution_receipt import ExecutionReceipt


def execution_receipt_mapping(
    operation: str,
    *,
    input_sha256_by_role: Mapping[str, str] | None = None,
    parameters_sha256: str = "5" * 64,
    requested_device: str = "0",
) -> dict[str, object]:
    environment = RuntimeEnvironmentReceipt(
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
        device_mapping=GpuDeviceMapping(
            requested_device=requested_device,
            cuda_visible_devices="3",
            torch_device_index=0,
            nvidia_smi_index=3,
            gpu_uuid="GPU-execution-fixture",
            gpu_name="NVIDIA RTX fixture",
        ),
        package_installations={
            role: PackageInstallation(
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
    return ExecutionReceipt(
        schema="zs32.execution_receipt",
        schema_version=1,
        operation=operation,
        captured_at="2026-07-14T00:00:00Z",
        code_identity={
            "git_commit": "1" * 40,
            "git_tree": "2" * 40,
            "dirty": False,
            "dependency_lock_sha256": "3" * 64,
        },
        environment=environment,
        input_sha256_by_role=(
            {"fixture": "4" * 64}
            if input_sha256_by_role is None
            else input_sha256_by_role
        ),
        parameters_sha256=parameters_sha256,
    ).as_dict()


__all__ = ["execution_receipt_mapping"]
