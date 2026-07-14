"""Content-addressed Linux/NVIDIA runtime-environment receipts."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import importlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import MappingProxyType
from typing import Any

from zs32_inspection.domain.identity import require_non_empty, require_sha256


_SCHEMA = "zs32.runtime_environment_receipt"
_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "platform",
        "architecture",
        "kernel_release",
        "os_release_id",
        "os_release_version_id",
        "python_version",
        "torch_version",
        "torchvision_version",
        "anomalib_version",
        "ultralytics_version",
        "opencv_version",
        "numpy_version",
        "torch_cuda_version",
        "cudnn_version",
        "nvidia_driver_version",
        "device_mapping",
        "package_installations",
        "receipt_sha256",
    }
)
_PACKAGE_KEYS = frozenset(
    {
        "distribution_name",
        "distribution_version",
        "metadata_sha256",
        "record_sha256",
        "record_entry_count",
        "installed_tree_sha256",
        "import_origin_relative_path",
        "import_origin_sha256",
    }
)
_PACKAGE_ROLES = frozenset(
    {"torch", "torchvision", "anomalib", "ultralytics", "opencv", "numpy"}
)
_DEVICE_KEYS = frozenset(
    {
        "requested_device",
        "cuda_visible_devices",
        "torch_device_index",
        "nvidia_smi_index",
        "gpu_uuid",
        "gpu_name",
    }
)


class RuntimeEnvironmentError(RuntimeError):
    """Runtime environment could not be captured or did not match a release."""


class RuntimeEnvironmentMismatchError(RuntimeEnvironmentError):
    """The live Linux/NVIDIA environment differs from the release receipt."""

    def __init__(
        self,
        *,
        expected: RuntimeEnvironmentReceipt,
        observed: RuntimeEnvironmentReceipt,
        differences: tuple[str, ...],
    ) -> None:
        self.expected = expected
        self.observed = observed
        self.differences = differences
        super().__init__("runtime environment differs from release: " + "; ".join(differences))


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _version(module: object, label: str) -> str:
    value = getattr(module, "__version__", None)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeEnvironmentError(f"{label} does not expose a non-empty __version__")
    return value.strip()


def _non_empty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


@dataclass(frozen=True, slots=True)
class GpuDeviceMapping:
    """Mapping from the inspect CLI device token to one physical NVIDIA GPU."""

    requested_device: str
    cuda_visible_devices: str | None
    torch_device_index: int
    nvidia_smi_index: int
    gpu_uuid: str
    gpu_name: str

    def __post_init__(self) -> None:
        requested = require_non_empty(self.requested_device, "requested_device")
        if re.fullmatch(r"[0-9]+", requested) is None:
            raise ValueError("requested_device must be one non-negative CUDA device index")
        object.__setattr__(self, "requested_device", requested)
        if self.cuda_visible_devices is not None:
            visible = require_non_empty(self.cuda_visible_devices, "cuda_visible_devices")
            object.__setattr__(self, "cuda_visible_devices", visible)
        for field in ("torch_device_index", "nvidia_smi_index"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        object.__setattr__(self, "gpu_uuid", require_non_empty(self.gpu_uuid, "gpu_uuid"))
        object.__setattr__(self, "gpu_name", _non_empty_text(self.gpu_name, "gpu_name"))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> GpuDeviceMapping:
        if set(payload) != _DEVICE_KEYS:
            raise ValueError("runtime environment device_mapping has missing or unknown fields")
        return cls(
            requested_device=payload["requested_device"],
            cuda_visible_devices=payload["cuda_visible_devices"],
            torch_device_index=payload["torch_device_index"],
            nvidia_smi_index=payload["nvidia_smi_index"],
            gpu_uuid=payload["gpu_uuid"],
            gpu_name=payload["gpu_name"],
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_device": self.requested_device,
            "cuda_visible_devices": self.cuda_visible_devices,
            "torch_device_index": self.torch_device_index,
            "nvidia_smi_index": self.nvidia_smi_index,
            "gpu_uuid": self.gpu_uuid,
            "gpu_name": self.gpu_name,
        }


@dataclass(frozen=True, slots=True)
class PackageInstallation:
    """Exact installed distribution metadata without exposing host paths."""

    distribution_name: str
    distribution_version: str
    metadata_sha256: str
    record_sha256: str
    record_entry_count: int
    installed_tree_sha256: str
    import_origin_relative_path: str
    import_origin_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "distribution_name",
            require_non_empty(self.distribution_name, "distribution_name"),
        )
        object.__setattr__(
            self,
            "distribution_version",
            require_non_empty(self.distribution_version, "distribution_version"),
        )
        for field in (
            "metadata_sha256",
            "record_sha256",
            "installed_tree_sha256",
            "import_origin_sha256",
        ):
            object.__setattr__(self, field, require_sha256(getattr(self, field), field))
        if (
            isinstance(self.record_entry_count, bool)
            or not isinstance(self.record_entry_count, int)
            or self.record_entry_count <= 0
        ):
            raise ValueError("record_entry_count must be a positive integer")
        origin = require_non_empty(
            self.import_origin_relative_path,
            "import_origin_relative_path",
        )
        if (
            origin.startswith("/")
            or "\\" in origin
            or any(part in {"", ".", ".."} for part in origin.split("/"))
        ):
            raise ValueError("import_origin_relative_path must be a safe RECORD path")
        object.__setattr__(self, "import_origin_relative_path", origin)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> PackageInstallation:
        if set(payload) != _PACKAGE_KEYS:
            raise ValueError("package installation has missing or unknown fields")
        return cls(
            distribution_name=payload["distribution_name"],
            distribution_version=payload["distribution_version"],
            metadata_sha256=payload["metadata_sha256"],
            record_sha256=payload["record_sha256"],
            record_entry_count=payload["record_entry_count"],
            installed_tree_sha256=payload["installed_tree_sha256"],
            import_origin_relative_path=payload["import_origin_relative_path"],
            import_origin_sha256=payload["import_origin_sha256"],
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "distribution_name": self.distribution_name,
            "distribution_version": self.distribution_version,
            "metadata_sha256": self.metadata_sha256,
            "record_sha256": self.record_sha256,
            "record_entry_count": self.record_entry_count,
            "installed_tree_sha256": self.installed_tree_sha256,
            "import_origin_relative_path": self.import_origin_relative_path,
            "import_origin_sha256": self.import_origin_sha256,
        }


@dataclass(frozen=True, slots=True)
class RuntimeEnvironmentReceipt:
    """Exact inference environment frozen into an immutable release."""

    schema: str
    schema_version: int
    platform: str
    architecture: str
    kernel_release: str
    os_release_id: str
    os_release_version_id: str
    python_version: str
    torch_version: str
    torchvision_version: str
    anomalib_version: str
    ultralytics_version: str
    opencv_version: str
    numpy_version: str
    torch_cuda_version: str
    cudnn_version: int
    nvidia_driver_version: str
    device_mapping: GpuDeviceMapping
    package_installations: Mapping[str, PackageInstallation]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            self.schema != _SCHEMA
            or isinstance(self.schema_version, bool)
            or self.schema_version != 2
            or self.platform != "linux"
        ):
            raise ValueError(
                "runtime environment receipt must use schema v2 on platform='linux'"
            )
        for field in (
            "architecture",
            "kernel_release",
            "os_release_id",
            "os_release_version_id",
            "python_version",
            "torch_version",
            "torchvision_version",
            "anomalib_version",
            "ultralytics_version",
            "opencv_version",
            "numpy_version",
            "torch_cuda_version",
            "nvidia_driver_version",
        ):
            object.__setattr__(self, field, require_non_empty(getattr(self, field), field))
        if (
            isinstance(self.cudnn_version, bool)
            or not isinstance(self.cudnn_version, int)
            or self.cudnn_version <= 0
        ):
            raise ValueError("cudnn_version must be a positive integer")
        if not isinstance(self.device_mapping, GpuDeviceMapping):
            raise TypeError("runtime environment receipt requires a GpuDeviceMapping")
        installations = dict(self.package_installations)
        if set(installations) != _PACKAGE_ROLES or any(
            not isinstance(item, PackageInstallation) for item in installations.values()
        ):
            raise ValueError(
                "runtime environment receipt requires exact package installation roles"
            )
        object.__setattr__(
            self,
            "package_installations",
            MappingProxyType(dict(sorted(installations.items()))),
        )
        expected = hashlib.sha256(
            _canonical_json_bytes(self.as_dict(include_receipt_sha256=False))
        ).hexdigest()
        if not self.receipt_sha256:
            object.__setattr__(self, "receipt_sha256", expected)
        else:
            actual = require_sha256(self.receipt_sha256, "receipt_sha256")
            if actual != expected:
                raise ValueError("runtime environment receipt_sha256 is not canonical")
            object.__setattr__(self, "receipt_sha256", actual)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> RuntimeEnvironmentReceipt:
        if set(payload) != _RECEIPT_KEYS:
            raise ValueError("runtime environment receipt has missing or unknown fields")
        raw_mapping = payload["device_mapping"]
        if not isinstance(raw_mapping, Mapping):
            raise ValueError("runtime environment device_mapping must be an object")
        raw_installations = payload["package_installations"]
        if not isinstance(raw_installations, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, Mapping)
            for key, value in raw_installations.items()
        ):
            raise ValueError("runtime environment package_installations must be an object")
        return cls(
            schema=payload["schema"],
            schema_version=payload["schema_version"],
            platform=payload["platform"],
            architecture=payload["architecture"],
            kernel_release=payload["kernel_release"],
            os_release_id=payload["os_release_id"],
            os_release_version_id=payload["os_release_version_id"],
            python_version=payload["python_version"],
            torch_version=payload["torch_version"],
            torchvision_version=payload["torchvision_version"],
            anomalib_version=payload["anomalib_version"],
            ultralytics_version=payload["ultralytics_version"],
            opencv_version=payload["opencv_version"],
            numpy_version=payload["numpy_version"],
            torch_cuda_version=payload["torch_cuda_version"],
            cudnn_version=payload["cudnn_version"],
            nvidia_driver_version=payload["nvidia_driver_version"],
            device_mapping=GpuDeviceMapping.from_mapping(raw_mapping),
            package_installations={
                key: PackageInstallation.from_mapping(value)
                for key, value in raw_installations.items()
            },
            receipt_sha256=payload["receipt_sha256"],
        )

    def as_dict(self, *, include_receipt_sha256: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "platform": self.platform,
            "architecture": self.architecture,
            "kernel_release": self.kernel_release,
            "os_release_id": self.os_release_id,
            "os_release_version_id": self.os_release_version_id,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "torchvision_version": self.torchvision_version,
            "anomalib_version": self.anomalib_version,
            "ultralytics_version": self.ultralytics_version,
            "opencv_version": self.opencv_version,
            "numpy_version": self.numpy_version,
            "torch_cuda_version": self.torch_cuda_version,
            "cudnn_version": self.cudnn_version,
            "nvidia_driver_version": self.nvidia_driver_version,
            "device_mapping": self.device_mapping.as_dict(),
            "package_installations": {
                key: value.as_dict()
                for key, value in self.package_installations.items()
            },
        }
        if include_receipt_sha256:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.as_dict())


def load_runtime_environment_receipt(path: Path) -> RuntimeEnvironmentReceipt:
    """Load one canonical regular receipt and verify its self digest."""
    source = Path(path).expanduser()
    if source.is_symlink() or not source.is_file():
        raise RuntimeEnvironmentError(
            f"runtime environment receipt must be a regular non-symlink file: {source}"
        )
    try:
        content = source.read_bytes()
        payload = json.loads(content)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeEnvironmentError(f"cannot load runtime environment receipt: {error}") from error
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise RuntimeEnvironmentError("runtime environment receipt root must be an object")
    try:
        receipt = RuntimeEnvironmentReceipt.from_mapping(payload)
    except (TypeError, ValueError) as error:
        raise RuntimeEnvironmentError(f"invalid runtime environment receipt: {error}") from error
    if content != receipt.canonical_bytes():
        raise RuntimeEnvironmentError("runtime environment receipt must use canonical JSON bytes")
    return receipt


def _nvidia_rows() -> tuple[tuple[int, str, str, str], ...]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeEnvironmentError("nvidia-smi is unavailable")
    environment = dict(os.environ)
    environment.setdefault("LC_ALL", "C")
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=index,uuid,name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeEnvironmentError(f"NVIDIA driver query failed: {error}") from error
    rows: list[tuple[int, str, str, str]] = []
    for line in result.stdout.splitlines():
        fields = tuple(field.strip() for field in line.split(",", maxsplit=3))
        if len(fields) != 4 or not fields[0].isdigit() or not all(fields[1:]):
            raise RuntimeEnvironmentError(f"malformed nvidia-smi inventory row: {line!r}")
        rows.append((int(fields[0]), fields[1], fields[2], fields[3]))
    if not rows:
        raise RuntimeEnvironmentError("nvidia-smi reported no NVIDIA GPU")
    return tuple(rows)


def _regular_file_bytes(path: Path, label: str) -> bytes:
    """Read one installed file without accepting symlinks or special nodes."""
    if path.is_symlink():
        raise RuntimeEnvironmentError(f"{label} must not be a symlink")
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as error:
        raise RuntimeEnvironmentError(f"cannot stat {label}: {error}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeEnvironmentError(f"{label} must be a regular file")
    try:
        return path.read_bytes()
    except OSError as error:
        raise RuntimeEnvironmentError(f"cannot read {label}: {error}") from error


def _record_rows(record_bytes: bytes) -> tuple[tuple[str, str, str], ...]:
    try:
        reader = csv.reader(io.StringIO(record_bytes.decode("utf-8"), newline=""))
        rows = tuple(tuple(row) for row in reader)
    except (UnicodeError, csv.Error) as error:
        raise RuntimeEnvironmentError(f"cannot parse distribution RECORD: {error}") from error
    if not rows or any(len(row) != 3 or not row[0] for row in rows):
        raise RuntimeEnvironmentError("distribution RECORD contains malformed rows")
    paths = tuple(row[0] for row in rows)
    if len(paths) != len(set(paths)):
        raise RuntimeEnvironmentError("distribution RECORD contains duplicate paths")
    return rows  # type: ignore[return-value]


def _package_installation(import_name: str, module: object) -> PackageInstallation:
    """Verify every installed wheel file and bind the module actually imported."""
    distributions = tuple(importlib_metadata.packages_distributions().get(import_name, ()))
    if len(distributions) != 1:
        raise RuntimeEnvironmentError(
            f"import package {import_name!r} must map to exactly one installed distribution"
        )
    try:
        distribution = importlib_metadata.distribution(distributions[0])
    except importlib_metadata.PackageNotFoundError as error:
        raise RuntimeEnvironmentError(
            f"installed distribution is unavailable for import package {import_name!r}"
        ) from error
    files = tuple(distribution.files or ())
    metadata_files = tuple(
        item for item in files if item.name == "METADATA" and item.parent.name.endswith(".dist-info")
    )
    record_files = tuple(
        item for item in files if item.name == "RECORD" and item.parent.name.endswith(".dist-info")
    )
    if len(metadata_files) != 1 or len(record_files) != 1:
        raise RuntimeEnvironmentError(
            f"distribution {distribution.metadata.get('Name')!r} lacks unique METADATA/RECORD"
        )
    metadata_path = Path(distribution.locate_file(metadata_files[0]))
    record_path = Path(distribution.locate_file(record_files[0]))
    metadata_bytes = _regular_file_bytes(metadata_path, "distribution METADATA")
    record_bytes = _regular_file_bytes(record_path, "distribution RECORD")
    rows = _record_rows(record_bytes)
    record_relative = record_files[0].as_posix()
    verified: dict[str, tuple[Path, str, int]] = {}
    tree_lines: list[bytes] = []
    for relative, hash_field, size_field in rows:
        if (
            relative.startswith("/")
            or "\\" in relative
            or any(part in {"", "."} for part in relative.split("/"))
        ):
            raise RuntimeEnvironmentError(
                f"distribution RECORD contains unsafe path: {relative!r}"
            )
        if relative == record_relative:
            if hash_field or size_field:
                raise RuntimeEnvironmentError("distribution RECORD self-row must be unhashed")
            continue
        if not hash_field or not size_field:
            raise RuntimeEnvironmentError(
                f"distribution RECORD entry is not content-addressed: {relative}"
            )
        try:
            algorithm, encoded_digest = hash_field.split("=", maxsplit=1)
        except ValueError as error:
            raise RuntimeEnvironmentError(
                f"distribution RECORD hash is malformed: {relative}"
            ) from error
        if algorithm != "sha256" or not encoded_digest:
            raise RuntimeEnvironmentError(
                f"distribution RECORD requires sha256 for every file: {relative}"
            )
        try:
            expected_size = int(size_field)
        except ValueError as error:
            raise RuntimeEnvironmentError(
                f"distribution RECORD size is malformed: {relative}"
            ) from error
        if expected_size < 0:
            raise RuntimeEnvironmentError(
                f"distribution RECORD size is negative: {relative}"
            )
        installed_path = Path(distribution.locate_file(relative))
        try:
            installed_path.resolve(strict=True).relative_to(Path(sys.prefix).resolve())
        except (OSError, ValueError) as error:
            raise RuntimeEnvironmentError(
                f"distribution RECORD path escapes the active Python prefix: {relative}"
            ) from error
        content = _regular_file_bytes(installed_path, f"installed distribution file {relative}")
        actual_digest = hashlib.sha256(content).digest()
        actual_encoded = base64.urlsafe_b64encode(actual_digest).rstrip(b"=").decode("ascii")
        if actual_encoded != encoded_digest or len(content) != expected_size:
            raise RuntimeEnvironmentError(
                f"installed distribution file differs from RECORD: {relative}"
            )
        hex_digest = actual_digest.hex()
        verified[relative] = (installed_path.resolve(), hex_digest, len(content))
        tree_lines.append(
            f"{relative}\0{hex_digest}\0{len(content)}\n".encode("utf-8")
        )
    if len(rows) != len(verified) + 1 or record_relative not in {row[0] for row in rows}:
        raise RuntimeEnvironmentError("distribution RECORD must contain one self-row")
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file.strip():
        raise RuntimeEnvironmentError(
            f"import package {import_name!r} does not expose a module file"
        )
    origin_path = Path(module_file)
    if origin_path.is_symlink():
        raise RuntimeEnvironmentError(f"import origin for {import_name!r} must not be a symlink")
    try:
        origin = origin_path.resolve(strict=True)
    except OSError as error:
        raise RuntimeEnvironmentError(
            f"cannot resolve import origin for {import_name!r}: {error}"
        ) from error
    origin_matches = [
        (relative, digest)
        for relative, (path, digest, _size) in verified.items()
        if path == origin
    ]
    if len(origin_matches) != 1:
        raise RuntimeEnvironmentError(
            f"import origin for {import_name!r} is not uniquely owned by its distribution RECORD"
        )
    origin_relative, origin_digest = origin_matches[0]
    return PackageInstallation(
        distribution_name=distribution.metadata["Name"],
        distribution_version=distribution.version,
        metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
        record_sha256=hashlib.sha256(record_bytes).hexdigest(),
        record_entry_count=len(rows),
        installed_tree_sha256=hashlib.sha256(b"".join(sorted(tree_lines))).hexdigest(),
        import_origin_relative_path=origin_relative,
        import_origin_sha256=origin_digest,
    )


def collect_runtime_environment(device: str) -> RuntimeEnvironmentReceipt:
    """Observe the exact Linux inference stack without loading model weights."""
    if platform.system().lower() != "linux":
        raise RuntimeEnvironmentError("runtime environment receipts may be collected only on Linux")
    device_token = require_non_empty(device, "device")
    if re.fullmatch(r"[0-9]+", device_token) is None:
        raise RuntimeEnvironmentError("inspect device must be one non-negative CUDA device index")
    logical_index = int(device_token)
    torch = importlib.import_module("torch")
    torchvision = importlib.import_module("torchvision")
    anomalib = importlib.import_module("anomalib")
    ultralytics = importlib.import_module("ultralytics")
    cv2 = importlib.import_module("cv2")
    numpy = importlib.import_module("numpy")
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        raise RuntimeEnvironmentError("torch CUDA runtime is unavailable")
    count = cuda.device_count()
    if logical_index >= count:
        raise RuntimeEnvironmentError(
            f"requested CUDA device {logical_index} is outside torch device_count={count}"
        )
    properties = cuda.get_device_properties(logical_index)
    gpu_name = str(getattr(properties, "name", "")).strip()
    if not gpu_name:
        raise RuntimeEnvironmentError("torch CUDA properties do not expose a GPU name")
    nvidia_rows = _nvidia_rows()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        matching = [row for row in nvidia_rows if row[0] == logical_index]
    else:
        tokens = tuple(token.strip() for token in visible.split(","))
        if any(not token for token in tokens) or logical_index >= len(tokens):
            raise RuntimeEnvironmentError(
                "CUDA_VISIBLE_DEVICES does not map the requested torch device"
            )
        selected = tokens[logical_index]
        matching = (
            [row for row in nvidia_rows if row[0] == int(selected)]
            if selected.isdigit()
            else [row for row in nvidia_rows if row[1] == selected]
        )
    if len(matching) != 1:
        raise RuntimeEnvironmentError(
            "requested torch device does not map uniquely to nvidia-smi inventory"
        )
    physical_index, gpu_uuid, smi_name, driver_version = matching[0]
    if smi_name != gpu_name:
        raise RuntimeEnvironmentError("torch and nvidia-smi GPU names disagree")
    torch_cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    if not isinstance(torch_cuda_version, str) or not torch_cuda_version.strip():
        raise RuntimeEnvironmentError("torch does not expose a CUDA build version")
    cudnn_version = getattr(getattr(getattr(torch, "backends", None), "cudnn", None), "version", None)
    cudnn_version = None if not callable(cudnn_version) else cudnn_version()
    if isinstance(cudnn_version, bool) or not isinstance(cudnn_version, int) or cudnn_version <= 0:
        raise RuntimeEnvironmentError("torch cuDNN runtime version is unavailable")
    os_release = platform.freedesktop_os_release()
    if not os_release.get("ID") or not os_release.get("VERSION_ID"):
        raise RuntimeEnvironmentError("/etc/os-release must provide ID and VERSION_ID")
    return RuntimeEnvironmentReceipt(
        schema=_SCHEMA,
        schema_version=2,
        platform="linux",
        architecture=platform.machine(),
        kernel_release=platform.release(),
        os_release_id=os_release["ID"],
        os_release_version_id=os_release["VERSION_ID"],
        python_version=platform.python_version(),
        torch_version=_version(torch, "torch"),
        torchvision_version=_version(torchvision, "torchvision"),
        anomalib_version=_version(anomalib, "anomalib"),
        ultralytics_version=_version(ultralytics, "ultralytics"),
        opencv_version=_version(cv2, "opencv"),
        numpy_version=_version(numpy, "numpy"),
        torch_cuda_version=torch_cuda_version.strip(),
        cudnn_version=cudnn_version,
        nvidia_driver_version=driver_version,
        device_mapping=GpuDeviceMapping(
            requested_device=device_token,
            cuda_visible_devices=visible,
            torch_device_index=logical_index,
            nvidia_smi_index=physical_index,
            gpu_uuid=gpu_uuid,
            gpu_name=gpu_name,
        ),
        package_installations={
            "torch": _package_installation("torch", torch),
            "torchvision": _package_installation("torchvision", torchvision),
            "anomalib": _package_installation("anomalib", anomalib),
            "ultralytics": _package_installation("ultralytics", ultralytics),
            "opencv": _package_installation("cv2", cv2),
            "numpy": _package_installation("numpy", numpy),
        },
    )


def verify_runtime_environment(
    expected: RuntimeEnvironmentReceipt,
    *,
    device: str,
) -> RuntimeEnvironmentReceipt:
    """Require a byte-identical live receipt before capture or model loading."""
    if not isinstance(expected, RuntimeEnvironmentReceipt):
        raise TypeError("expected runtime environment must be a parsed receipt")
    observed = collect_runtime_environment(device)
    expected_payload = expected.as_dict(include_receipt_sha256=False)
    observed_payload = observed.as_dict(include_receipt_sha256=False)
    differences = tuple(
        f"{field}: expected={expected_payload[field]!r}, observed={observed_payload[field]!r}"
        for field in sorted(expected_payload)
        if expected_payload[field] != observed_payload[field]
    )
    if differences:
        raise RuntimeEnvironmentMismatchError(
            expected=expected,
            observed=observed,
            differences=differences,
        )
    return observed


__all__ = [
    "GpuDeviceMapping",
    "PackageInstallation",
    "RuntimeEnvironmentError",
    "RuntimeEnvironmentMismatchError",
    "RuntimeEnvironmentReceipt",
    "collect_runtime_environment",
    "load_runtime_environment_receipt",
    "verify_runtime_environment",
]
