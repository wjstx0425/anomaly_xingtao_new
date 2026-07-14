#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Freeze and verify the legacy ZS32 baseline on Linux + NVIDIA only.

This tool is intentionally independent from the legacy pipeline.  It records
the repository, runtime environment, immutable asset hashes, manually approved
golden captures, and already-produced baseline outputs without attempting to
reinterpret their model decisions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PRODUCT = "ZS32"
ALLOWED_TARGET_STATUSES = {
    "OK",
    "NG_TEMPLATE",
    "NG_ANOMALY",
    "NG_YOLO",
    "REVIEW",
    "RETAKE",
    "INVALID_CAPTURE",
    "SYSTEM_ERROR",
}
ALLOWED_LEGACY_STATUSES = ALLOWED_TARGET_STATUSES | {
    "NG_GEOMETRY",
    "NG_CRACK",
    "NG_GLOBAL",
    "SUSPECT",
}
REQUIRED_SCENARIO_TARGET_STATUSES = {
    "normal_clear": "OK",
    "template_mismatch": "NG_TEMPLATE",
    "anomaly_strong": "NG_ANOMALY",
    "yolo_strong": "NG_YOLO",
    "gray_review": "REVIEW",
    "quality_retake": "RETAKE",
    "registration_retake": "RETAKE",
}
NORMATIVE_REQUIRED_ASSET_ROLES = frozenset(
    {
        "capture_acquisition_config",
        "authoritative_yolo_roi",
        "legacy_patchcore_roi",
        "legacy_yolo_roi",
        "legacy_fusion_contract",
        "legacy_model_registry",
        "deployment_weight",
        "training_init_weight",
        "yolo_training_provenance",
        "yolo_source_snapshot",
        "yolo_dataset_provenance",
        "yolo_dataset_manifest",
        "template_match",
        "legacy_thresholds",
        "selected_anomaly_family",
        "anomaly_training_provenance",
        "anomaly_dataset_manifest",
    }
)
NORMATIVE_ASSET_KIND_BY_ROLE = {
    "capture_acquisition_config": "camera_config",
    "authoritative_yolo_roi": "roi_config",
    "legacy_patchcore_roi": "roi_config",
    "legacy_yolo_roi": "roi_config",
    "legacy_fusion_contract": "runtime_config",
    "legacy_model_registry": "runtime_config",
    "deployment_weight": "yolo_checkpoint",
    "training_init_weight": "yolo_checkpoint",
    "yolo_training_provenance": "training_config",
    "yolo_source_snapshot": "git_source_bundle",
    "yolo_dataset_provenance": "dataset_config",
    "yolo_dataset_manifest": "dataset_manifest",
    "template_match": "template_bundle",
    "legacy_thresholds": "threshold_bundle",
    "selected_anomaly_family": "anomaly_checkpoint_bundle",
    "anomaly_training_provenance": "training_provenance_bundle",
    "anomaly_dataset_manifest": "dataset_manifest",
}
NORMATIVE_REQUIRED_GOLDEN_SCENARIOS = frozenset(REQUIRED_SCENARIO_TARGET_STATUSES)
NORMATIVE_BLOCKER_IDS = frozenset(
    {
        "yolo-run-name-seed-mismatch",
        "runtime-registry-asset-path-mismatch",
        "training-dataset-manifest-completeness",
        "legacy-gate-evidence-provenance",
    }
)
LEGACY_REPLAY_ENTRYPOINT = "pipeline/32_run_zs32_multimodel_inference.py"
LEGACY_REPLAY_TIMEOUT_SECONDS = 1800
LEGACY_REPLAY_PLAN_ROLE = "legacy_replay_plan"
LEGACY_GATE_RECEIPT_ROLE = "legacy_gate_producer_receipt"
LEGACY_GATE_ROLE_BY_NAME = {
    "quality": "legacy_quality_evidence",
    "registration": "legacy_registration_evidence",
    "geometry": "legacy_geometry_evidence",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
PLACEHOLDER_PATTERN = re.compile(r"REPLACE|PENDING|TBD|YYYY", re.IGNORECASE)
RFC3339_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)
REQUIRED_BASELINE_EVIDENCE_FILES = frozenset(
    {
        "anomaly_evidence.json",
        "audit.json",
        "fusion_result.json",
        "phase0_case_result.json",
        "runtime_summary.json",
        "replay_root.json",
        "stderr.log",
        "stdout.log",
        "template_evidence.json",
        "yolo_evidence.json",
    }
)
REQUIRED_BASELINE_JSON_EVIDENCE_FILES = frozenset(
    name
    for name in REQUIRED_BASELINE_EVIDENCE_FILES
    if name.endswith(".json") and name not in {"phase0_case_result.json", "replay_root.json"}
)
REPLAY_ROOT_ALGORITHM = "sha256(relative_path\\0file_sha256\\0size\\n)"


class Phase0Error(RuntimeError):
    """The requested Phase 0 operation is unsafe or incomplete."""


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
        raise Phase0Error(f"command failed ({' '.join(command)}): {detail}")
    return result.stdout.strip()


def require_linux_nvidia() -> dict[str, Any]:
    """Reject every operational command outside a CUDA-capable Linux host."""
    system = platform.system()
    if system != "Linux":
        raise Phase0Error(
            f"Phase 0 operations are Linux + NVIDIA only; current platform is {system}"
        )
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise Phase0Error("nvidia-smi was not found on PATH")
    gpu_csv = _run(
        [
            nvidia_smi,
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    gpu_rows = [row.strip() for row in gpu_csv.splitlines() if row.strip()]
    if not gpu_rows:
        raise Phase0Error("nvidia-smi reported no NVIDIA GPUs")
    try:
        import torch
    except Exception as error:  # pragma: no cover - depends on Linux runtime
        raise Phase0Error(f"PyTorch cannot be imported: {error}") from error
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise Phase0Error("PyTorch reports no available CUDA device")
    return {
        "platform": system,
        "nvidia_smi": nvidia_smi,
        "gpus": gpu_rows,
        "torch_cuda_available": True,
        "torch_cuda_device_count": torch.cuda.device_count(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }


def _load_json(path: Path) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError(f"duplicate JSON key {key!r}")
            payload[key] = value
        return payload

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise Phase0Error(f"cannot read JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise Phase0Error(f"JSON root must be an object: {path}")
    return payload


def _write_text_no_replace(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise Phase0Error(f"refusing to overwrite existing output: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    _write_text_no_replace(path, f"{encoded}\n")


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _freeze_review_subject_sha256(spec: dict[str, Any]) -> str:
    """Hash the freeze intent while excluding the post-review approval envelope."""
    subject = {key: value for key, value in spec.items() if key not in {"status", "approval"}}
    return _canonical_sha256(subject)


def _reviewable_inventory_sha256(inventory: dict[str, Any]) -> str:
    """Hash the stable evidence an operator must review before approving freeze."""
    required = {
        "review_subject_sha256",
        "scope",
        "repository",
        "environment",
        "topology",
        "roi",
        "golden_selection",
        "legacy_replay_plan",
        "assets",
        "golden_cases",
        "baseline_outputs",
        "requirements",
        "repository_contracts",
    }
    if not required.issubset(inventory):
        raise Phase0Error("reviewed inventory is missing required evidence sections")
    payload = {key: inventory[key] for key in required}
    environment = payload["environment"]
    if not isinstance(environment, dict):
        raise Phase0Error("reviewed inventory environment is malformed")
    stable_environment = dict(environment)
    stable_environment.pop("captured_at", None)
    payload["environment"] = stable_environment
    return _canonical_sha256(payload)


def _resolve(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _safe_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or SAFE_ID_PATTERN.fullmatch(value) is None:
        raise Phase0Error(f"{field} must match {SAFE_ID_PATTERN.pattern}")
    return value


def _signed_value(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or PLACEHOLDER_PATTERN.search(value):
        raise Phase0Error(f"{field} is empty or still contains a placeholder")
    return value


def _signed_timestamp(value: Any, field: str) -> str:
    timestamp = _signed_value(value, field)
    if RFC3339_PATTERN.fullmatch(timestamp) is None:
        raise Phase0Error(f"{field} must be an RFC3339 timestamp with an explicit timezone")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise Phase0Error(f"{field} must be an RFC3339 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise Phase0Error(f"{field} must include an explicit timezone offset")
    return timestamp


def _reject_placeholders(value: Any, field: str) -> None:
    """Reject unfinished strings recursively in manually authored evidence."""
    if isinstance(value, str):
        _signed_value(value, field)
    elif isinstance(value, dict):
        if not value:
            raise Phase0Error(f"{field} must not be empty")
        for key, nested in value.items():
            if not isinstance(key, str) or not key:
                raise Phase0Error(f"{field} keys must be non-empty strings")
            _reject_placeholders(nested, f"{field}.{key}")
    elif isinstance(value, list):
        if not value:
            raise Phase0Error(f"{field} must not be empty")
        for index, nested in enumerate(value):
            _reject_placeholders(nested, f"{field}[{index}]")
    elif value is None:
        raise Phase0Error(f"{field} must not contain null evidence")


def _validate_normative_spec_requirements(
    spec: dict[str, Any],
    *,
    require_resolved_blockers: bool,
) -> None:
    """Prevent a local freeze spec from weakening the checked-in Phase 0 scope."""
    roles = spec.get("required_asset_roles")
    if (
        not isinstance(roles, list)
        or any(not isinstance(role, str) for role in roles)
        or len(roles) != len(set(roles))
        or set(roles) != NORMATIVE_REQUIRED_ASSET_ROLES
    ):
        raise Phase0Error("required_asset_roles must exactly match the normative Phase 0 role set")
    scenarios = spec.get("required_golden_scenarios")
    if (
        not isinstance(scenarios, list)
        or any(not isinstance(scenario, str) for scenario in scenarios)
        or len(scenarios) != len(set(scenarios))
        or set(scenarios) != NORMATIVE_REQUIRED_GOLDEN_SCENARIOS
    ):
        raise Phase0Error(
            "required_golden_scenarios must exactly match the normative Phase 0 scenario set"
        )
    blockers = spec.get("known_blockers")
    if not isinstance(blockers, list) or any(not isinstance(item, dict) for item in blockers):
        raise Phase0Error("known_blockers must contain the normative Phase 0 blocker records")
    blocker_ids = [item.get("blocker_id") for item in blockers]
    if (
        any(not isinstance(blocker_id, str) for blocker_id in blocker_ids)
        or len(blocker_ids) != len(set(blocker_ids))
        or set(blocker_ids) != NORMATIVE_BLOCKER_IDS
    ):
        raise Phase0Error("known_blockers must exactly match the normative Phase 0 blocker IDs")
    for item in blockers:
        if set(item) != {"blocker_id", "resolved", "evidence"}:
            raise Phase0Error(f"blocker {item.get('blocker_id')!r} fields differ from strict schema")
        if not isinstance(item.get("resolved"), bool):
            raise Phase0Error(f"blocker {item.get('blocker_id')!r} resolved must be boolean")
        _signed_value(item.get("evidence"), f"blocker {item.get('blocker_id')} evidence")
        if require_resolved_blockers and item["resolved"] is not True:
            raise Phase0Error(f"freeze spec still has unresolved blocker: {item['blocker_id']}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_files(root: Path) -> list[Path]:
    if root.is_symlink():
        raise Phase0Error(f"symlinks are forbidden in Phase 0 inputs: {root}")
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise Phase0Error(f"path is neither a regular file nor directory: {root}")
    files: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if any("\n" in part or "\r" in part for part in path.relative_to(root).parts):
            raise Phase0Error(f"control characters are forbidden in Phase 0 paths: {path}")
        if path.is_symlink():
            raise Phase0Error(f"symlinks are forbidden in Phase 0 inputs: {path}")
        if path.is_file():
            if path.stat().st_nlink != 1:
                raise Phase0Error(f"hard-linked files are forbidden in Phase 0 inputs: {path}")
            files.append(path)
        elif not path.is_dir():
            raise Phase0Error(f"special filesystem nodes are forbidden in Phase 0 inputs: {path}")
    if not files:
        raise Phase0Error(f"directory asset is empty: {root}")
    return files


def _path_digest(path: Path) -> tuple[str, int, int]:
    files = _regular_files(path)
    if path.is_file():
        return _sha256_file(path), path.stat().st_size, 1
    digest = hashlib.sha256()
    total_size = 0
    for file_path in files:
        relative = file_path.relative_to(path).as_posix()
        file_digest = _sha256_file(file_path)
        size = file_path.stat().st_size
        total_size += size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), total_size, len(files)


def _directory_digest_excluding(root: Path, excluded: set[str]) -> tuple[str, int, int]:
    """Hash a directory tree while excluding exact POSIX-relative control files."""
    if not root.is_dir() or root.is_symlink():
        raise Phase0Error(f"replay root must be a regular directory: {root}")
    digest = hashlib.sha256()
    total_size = 0
    file_count = 0
    for file_path in _regular_files(root):
        relative = file_path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        file_digest = _sha256_file(file_path)
        size = file_path.stat().st_size
        total_size += size
        file_count += 1
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\n")
    if file_count == 0:
        raise Phase0Error(f"replay root contains no evidence files: {root}")
    return digest.hexdigest(), total_size, file_count


def _write_replay_root(root: Path) -> dict[str, Any]:
    digest, size, file_count = _directory_digest_excluding(root, {"replay_root.json"})
    payload = {
        "schema_version": 1,
        "algorithm": REPLAY_ROOT_ALGORITHM,
        "replay_root_sha256": digest,
        "size_bytes_excluding_root": size,
        "file_count_excluding_root": file_count,
    }
    _write_json(root / "replay_root.json", payload)
    return payload


def _verify_replay_root(root: Path) -> dict[str, Any]:
    root_path = root / "replay_root.json"
    payload = _load_json(root_path)
    if (
        set(payload)
        != {
            "schema_version",
            "algorithm",
            "replay_root_sha256",
            "size_bytes_excluding_root",
            "file_count_excluding_root",
        }
        or payload.get("schema_version") != 1
        or payload.get("algorithm") != REPLAY_ROOT_ALGORITHM
        or not isinstance(payload.get("replay_root_sha256"), str)
        or SHA256_PATTERN.fullmatch(payload["replay_root_sha256"]) is None
    ):
        raise Phase0Error(f"replay root receipt is malformed: {root_path}")
    digest, size, file_count = _directory_digest_excluding(root, {"replay_root.json"})
    if (
        payload["replay_root_sha256"] != digest
        or payload.get("size_bytes_excluding_root") != size
        or payload.get("file_count_excluding_root") != file_count
    ):
        raise Phase0Error(f"replay evidence tree changed after publication: {root}")
    return payload


def _copy_input(source: Path, destination: Path) -> None:
    _regular_files(source)
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return
    shutil.copytree(source, destination, symlinks=False)


def _copy_and_verify(source: Path, destination: Path, expected_sha256: str) -> None:
    """Copy one input and reject a concurrent or partial source mutation."""
    _copy_input(source, destination)
    actual, _, _ = _path_digest(destination)
    if actual != expected_sha256:
        raise Phase0Error(
            f"copied input hash mismatch for {source}: expected {expected_sha256}, got {actual}"
        )


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise Phase0Error(f"golden image is not a valid PNG header: {path}")
    width, height = struct.unpack(">II", header[16:24])
    try:
        import cv2
    except Exception as error:  # pragma: no cover - checked by Linux environment guard
        raise Phase0Error(f"OpenCV cannot be imported while validating {path}: {error}") from error
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise Phase0Error(f"golden PNG cannot be decoded by OpenCV: {path}")
    decoded_height, decoded_width = image.shape[:2]
    if (decoded_width, decoded_height) != (width, height):
        raise Phase0Error(f"golden PNG header/decoded dimensions disagree: {path}")
    return width, height


def _legacy_filename_identity(path: Path, hand: str, view: str) -> tuple[str, str]:
    prefix = f"{hand}_{view}_"
    if not path.name.startswith(prefix):
        raise Phase0Error(f"legacy golden filename does not match hand/view {hand}/{view}: {path}")
    tail = path.name[len(prefix) :]
    match = re.search(r"_(group\d+)_(\d{6})_(?:single|fused)\.png$", tail)
    if match is None:
        raise Phase0Error(f"legacy golden filename has no capture group identity: {path}")
    sample_token = tail[: match.start()]
    return f"{sample_token}_{match.group(1)}_{match.group(2)}", match.group(1)


def _git_snapshot(*, require_clean: bool, expected: dict[str, Any]) -> dict[str, Any]:
    status = _run(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=REPO_ROOT)
    if require_clean and status:
        raise Phase0Error("git worktree must be completely clean before freeze")
    remote_name = _signed_value(expected.get("remote_name"), "expected_git.remote_name")
    remote_url = _signed_value(expected.get("remote_url"), "expected_git.remote_url")
    expected_branch = _signed_value(expected.get("branch"), "expected_git.branch")
    actual_remote_url = _run(["git", "remote", "get-url", remote_name], cwd=REPO_ROOT)
    if actual_remote_url != remote_url:
        raise Phase0Error(
            f"git remote URL mismatch: expected {remote_url!r}, got {actual_remote_url!r}"
        )
    branch = _run(["git", "branch", "--show-current"], cwd=REPO_ROOT)
    if branch != expected_branch:
        raise Phase0Error(f"git branch mismatch: expected {expected_branch!r}, got {branch!r}")
    commit = _run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    upstream_commit = _run(
        ["git", "rev-parse", f"refs/remotes/{remote_name}/{expected_branch}"],
        cwd=REPO_ROOT,
    )
    if commit != upstream_commit:
        raise Phase0Error(
            f"HEAD {commit} is not the pulled {remote_name}/{expected_branch} commit {upstream_commit}"
        )
    remotes = _run(["git", "remote", "-v"], cwd=REPO_ROOT)
    return {
        "commit": commit,
        "tree": _run(["git", "rev-parse", "HEAD^{tree}"], cwd=REPO_ROOT),
        "branch": branch,
        "upstream_commit": upstream_commit,
        "status_porcelain": status.splitlines(),
        "remotes": remotes.splitlines(),
    }


def _os_release() -> dict[str, str]:
    path = Path("/etc/os-release")
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def _environment_snapshot(spec: dict[str, Any], gpu: dict[str, Any]) -> dict[str, Any]:
    package_versions: dict[str, str] = {}
    missing: list[str] = []
    required_packages = spec.get("required_python_packages")
    if not isinstance(required_packages, list) or not required_packages or not all(
        isinstance(name, str) and name for name in required_packages
    ):
        raise Phase0Error("required_python_packages must be a non-empty string array")
    for name in required_packages:
        try:
            package_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
    if missing:
        raise Phase0Error(f"required Python packages are missing: {', '.join(missing)}")

    all_packages = sorted(
        f"{distribution.metadata.get('Name', 'unknown')}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
    )
    sdk = spec.get("camera_sdk", {})
    if not isinstance(sdk, dict):
        raise Phase0Error("camera_sdk must be an object")
    sdk_path_value = sdk.get("required_path")
    if not isinstance(sdk_path_value, str) or not Path(sdk_path_value).is_dir():
        raise Phase0Error(f"required camera SDK path is missing: {sdk_path_value!r}")
    sdk_version = _signed_value(sdk.get("version"), "camera_sdk.version")
    fingerprint_values = sdk.get("fingerprint_paths")
    if (
        not isinstance(fingerprint_values, list)
        or len(fingerprint_values) < 2
        or len(fingerprint_values) != len(set(fingerprint_values))
        or any(not isinstance(value, str) or not value for value in fingerprint_values)
    ):
        raise Phase0Error(
            "camera_sdk.fingerprint_paths must contain unique Python and native SDK files"
        )
    fingerprint_paths: list[Path] = []
    for value in fingerprint_values:
        candidate = Path(value).expanduser()
        if candidate.is_symlink() or not candidate.is_file():
            raise Phase0Error(f"camera SDK fingerprint file is missing or unsafe: {candidate}")
        fingerprint_paths.append(candidate.resolve())
    if not any(".so" in path.name for path in fingerprint_paths):
        raise Phase0Error("camera SDK fingerprints must include at least one native shared library")

    try:
        ultralytics = importlib.import_module("ultralytics")
        package_file = getattr(ultralytics, "__file__", None)
        if not isinstance(package_file, str) or not package_file:
            raise Phase0Error("Ultralytics import does not expose a source file")
        package_root = Path(package_file).resolve().parent
        repository_root = Path(
            _run(["git", "-C", str(package_root), "rev-parse", "--show-toplevel"])
        ).resolve()
        package_relative = package_root.relative_to(repository_root)
        status = _run(
            [
                "git",
                "-C",
                str(repository_root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                package_relative.as_posix(),
            ]
        )
        if status:
            raise Phase0Error("the imported Ultralytics package tree is not clean")
        tracked = set(
            _run(
                [
                    "git",
                    "-C",
                    str(repository_root),
                    "ls-files",
                    "--",
                    package_relative.as_posix(),
                ]
            ).splitlines()
        )
        actual = {
            path.relative_to(repository_root).as_posix()
            for path in _regular_files(package_root)
            if "__pycache__" not in path.relative_to(package_root).parts
            and path.suffix not in {".pyc", ".pyo"}
        }
        if not tracked or actual != tracked:
            raise Phase0Error(
                "the imported Ultralytics package contains untracked/ignored or missing source files"
            )
        from zs32_inspection.models.yolo_receipt import runtime_package_tree_sha256

        ultralytics_runtime = {
            "package_root": str(package_root),
            "package_tree_sha256": runtime_package_tree_sha256(package_root),
            "repository_root": str(repository_root),
            "git_commit": _run(["git", "-C", str(repository_root), "rev-parse", "HEAD"]),
            "git_tree": _run(["git", "-C", str(repository_root), "rev-parse", "HEAD^{tree}"]),
            "worktree_state": "clean",
        }
    except (ImportError, OSError, ValueError) as error:
        raise Phase0Error(f"cannot bind the imported Ultralytics source tree: {error}") from error

    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "os_release": _os_release(),
        "uname": platform.uname()._asdict(),
        "python": sys.version,
        "python_executable": sys.executable,
        "packages": package_versions,
        "all_installed_packages": all_packages,
        "gpu": gpu,
        "camera_sdk": {
            "path": str(Path(sdk_path_value).resolve()),
            "version": sdk_version,
            "fingerprints": [
                {
                    "path": str(path),
                    "sha256": _sha256_file(path),
                }
                for path in fingerprint_paths
            ],
        },
        "ultralytics_runtime": ultralytics_runtime,
        "environment_variables": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "PYTHONHASHSEED",
                "CUBLAS_WORKSPACE_CONFIG",
                "LANG",
                "LC_ALL",
                "TZ",
            )
        },
    }


def _validate_ultralytics_source_binding(
    assets: list[dict[str, Any]],
    environment: dict[str, Any],
) -> None:
    """Prove the imported Stage 32 package commit is present in the frozen bundle."""
    runtime = environment.get("ultralytics_runtime")
    if not isinstance(runtime, dict) or set(runtime) != {
        "package_root",
        "package_tree_sha256",
        "repository_root",
        "git_commit",
        "git_tree",
        "worktree_state",
    }:
        raise Phase0Error("Ultralytics runtime identity is missing or malformed")
    for field in ("package_tree_sha256", "git_commit", "git_tree"):
        value = runtime.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value) is None:
            raise Phase0Error(f"Ultralytics runtime {field} is invalid")
    if runtime.get("worktree_state") != "clean":
        raise Phase0Error("Ultralytics runtime source tree is not clean")
    source = _asset_by_role(assets, "yolo_source_snapshot")
    bundle = Path(source["source_path"])
    heads = _run(["git", "bundle", "list-heads", str(bundle)], cwd=REPO_ROOT)
    commits = {line.split(maxsplit=1)[0] for line in heads.splitlines() if line.strip()}
    if runtime["git_commit"] not in commits:
        raise Phase0Error(
            "the imported Ultralytics commit is not a head in yolo_source_snapshot"
        )


def _stable_environment_snapshot(environment: dict[str, Any]) -> dict[str, Any]:
    """Remove only the observation timestamp from a replay environment receipt."""
    stable = dict(environment)
    stable.pop("captured_at", None)
    return stable


def _validate_topology(path: Path) -> tuple[dict[str, Any], list[str]]:
    topology = _load_json(path)
    if topology.get("schema_version") != 1 or topology.get("product") != EXPECTED_PRODUCT:
        raise Phase0Error(f"unsupported topology identity: {path}")
    _safe_id(topology.get("topology_id"), "topology.topology_id")
    rounds = topology.get("rounds")
    slots = topology.get("camera_slots")
    required_views = topology.get("required_views")
    if not isinstance(rounds, list) or not rounds or not isinstance(slots, list) or not slots:
        raise Phase0Error("topology requires non-empty rounds and camera_slots")
    if (
        not isinstance(required_views, list)
        or not required_views
        or not all(isinstance(view, str) and view for view in required_views)
        or len(required_views) != len(set(required_views))
    ):
        raise Phase0Error("topology.required_views must be a string array")
    if not all(isinstance(item, dict) for item in rounds):
        raise Phase0Error("topology rounds must be objects")
    round_ids = [item.get("round_id") for item in rounds]
    if len(round_ids) != len(set(round_ids)) or not all(
        isinstance(item, str) and item for item in round_ids
    ):
        raise Phase0Error("topology round IDs must be unique strings")
    slot_ids: list[str] = []
    serials: list[str] = []
    mapped_views: list[str] = []
    for slot in slots:
        if not isinstance(slot, dict):
            raise Phase0Error("topology camera slots must be objects")
        slot_ids.append(slot.get("slot_id"))
        serials.append(slot.get("serial"))
        views = slot.get("views")
        if not isinstance(views, dict) or set(views) != set(round_ids):
            raise Phase0Error("every camera slot must map exactly one view for every round")
        mapped_views.extend(views.values())
    for values, label in ((slot_ids, "slot IDs"), (serials, "camera serials")):
        if not all(isinstance(item, str) and item for item in values) or len(values) != len(set(values)):
            raise Phase0Error(f"topology {label} must be non-empty and unique")
    if len(mapped_views) != len(set(mapped_views)) or set(mapped_views) != set(required_views):
        raise Phase0Error("required_views must exactly match the unique round/slot view mapping")
    return topology, required_views


def _validate_roi(path: Path, hands: list[str], required_views: list[str]) -> dict[str, Any]:
    roi = _load_json(path)
    if (
        roi.get("schema_version") != 2
        or roi.get("product") != EXPECTED_PRODUCT
        or roi.get("coordinate_system") != "pixel_xyxy_half_open"
    ):
        raise Phase0Error(f"unsupported ROI contract: {path}")
    _safe_id(roi.get("roi_version"), "roi.roi_version")
    _safe_id(roi.get("topology_id"), "roi.topology_id")
    source_image = roi.get("source_image_size", {})
    width, height = source_image.get("width"), source_image.get("height")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise Phase0Error("ROI source image dimensions must be positive integers")
    configured_hands = roi.get("hands")
    if not isinstance(configured_hands, dict):
        raise Phase0Error("ROI hands must be an object")
    if set(configured_hands) != {"left", "right"}:
        raise Phase0Error("ROI hands must explicitly contain exactly left and right")
    for hand, config in configured_hands.items():
        if not isinstance(config, dict):
            raise Phase0Error(f"ROI hand {hand} must be an object")
        status = config.get("status")
        views = config.get("views")
        if status == "pending":
            if views != {}:
                raise Phase0Error(f"pending ROI hand {hand} must have an empty views object")
            continue
        if status != "ready" or not isinstance(views, dict):
            raise Phase0Error(f"ROI hand {hand} has invalid status/views")
        if set(views) != set(required_views):
            raise Phase0Error(f"ROI hand {hand} does not cover exactly the required topology views")
        for view, record in views.items():
            if not isinstance(record, dict):
                raise Phase0Error(f"ROI {hand}/{view} must be an object containing xyxy")
            box = record.get("xyxy")
            if not isinstance(box, list) or len(box) != 4 or not all(isinstance(v, int) for v in box):
                raise Phase0Error(f"ROI {hand}/{view} must be four integer xyxy values")
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise Phase0Error(f"ROI {hand}/{view} is empty or outside {width}x{height}")
    for hand in hands:
        config = configured_hands.get(hand)
        if not isinstance(config, dict):
            raise Phase0Error(f"ROI does not define requested hand {hand}")
        if config.get("status") != "ready":
            raise Phase0Error(f"ROI_PENDING: requested hand {hand} is not release-ready")
    return roi


def _asset_inventory(spec: dict[str, Any], base: Path) -> list[dict[str, Any]]:
    assets = spec.get("assets")
    if not isinstance(assets, list) or not assets:
        raise Phase0Error("freeze spec requires a non-empty assets array")
    seen: set[str] = set()
    inventory: list[dict[str, Any]] = []
    for item in assets:
        if not isinstance(item, dict):
            raise Phase0Error("every asset entry must be an object")
        asset_id = _safe_id(item.get("asset_id"), "asset_id")
        if asset_id in seen:
            raise Phase0Error(f"duplicate asset_id: {asset_id}")
        seen.add(asset_id)
        source = _resolve(item.get("path", ""), base)
        required = item.get("required") is True
        if not source.exists():
            if required:
                raise Phase0Error(f"required asset is missing: {asset_id} ({source})")
            inventory.append({"asset_id": asset_id, "present": False, "required": False})
            continue
        if item.get("kind") == "yolo_checkpoint" and item.get("role") == "deployment_weight":
            if source.name != "best.pt":
                raise Phase0Error("YOLO deployment weight must be a trained best.pt, not an init weight")
        digest, size, file_count = _path_digest(source)
        minimum_size = item.get("minimum_size_bytes", 0)
        if not isinstance(minimum_size, int) or minimum_size < 0:
            raise Phase0Error(f"asset {asset_id} minimum_size_bytes must be a non-negative integer")
        if size < minimum_size:
            raise Phase0Error(
                f"asset {asset_id} is smaller than its minimum size: {size} < {minimum_size}"
            )
        minimum_file_count = item.get("minimum_file_count", 1)
        if not isinstance(minimum_file_count, int) or minimum_file_count < 1:
            raise Phase0Error(f"asset {asset_id} minimum_file_count must be a positive integer")
        if file_count < minimum_file_count:
            raise Phase0Error(
                f"asset {asset_id} has too few files: {file_count} < {minimum_file_count}"
            )
        expected = item.get("expected_sha256")
        if expected is not None:
            if not isinstance(expected, str) or SHA256_PATTERN.fullmatch(expected) is None:
                raise Phase0Error(f"asset {asset_id} expected_sha256 is not a real SHA256")
            if digest != expected:
                raise Phase0Error(f"asset {asset_id} SHA256 mismatch: expected {expected}, got {digest}")
        copy_into_bundle = item.get("copy_into_bundle") is True
        immutable_reference = item.get("immutable_storage_reference")
        if not copy_into_bundle:
            if expected is None or not isinstance(immutable_reference, str) or not immutable_reference:
                raise Phase0Error(
                    f"non-copied asset {asset_id} requires expected_sha256 and "
                    "immutable_storage_reference"
                )
        inventory.append(
            {
                key: value
                for key, value in {
                    "asset_id": asset_id,
                    "kind": item.get("kind"),
                    "role": item.get("role"),
                    "model_family": item.get("model_family"),
                    "source_path": str(source),
                    "required": required,
                    "present": True,
                    "copy_into_bundle": copy_into_bundle,
                    "immutable_storage_reference": immutable_reference,
                    "source_type": "file" if source.is_file() else "directory",
                    "bundle_relative_path": (
                        f"assets/{asset_id}{source.suffix}" if source.is_file() else f"assets/{asset_id}"
                    ),
                    "sha256": digest,
                    "size_bytes": size,
                    "file_count": file_count,
                }.items()
                if value is not None
            }
        )
    return inventory


def _validate_legacy_model_assets(
    assets: list[dict[str, Any]],
    required_views: list[str],
    *,
    expected_roi_sha256: str,
) -> None:
    """Bind the legacy registry to the actual six checkpoints and YOLO weight."""
    by_role: dict[str, dict[str, Any]] = {}
    for role in NORMATIVE_REQUIRED_ASSET_ROLES:
        matching = [asset for asset in assets if asset.get("role") == role and asset.get("present")]
        if len(matching) != 1:
            raise Phase0Error(f"legacy asset role {role!r} must resolve exactly once")
        asset = matching[0]
        if asset.get("required") is not True or asset.get("copy_into_bundle") is not True:
            raise Phase0Error(
                f"legacy normative asset role {role!r} must be required and copied into the bundle"
            )
        if asset.get("kind") != NORMATIVE_ASSET_KIND_BY_ROLE[role]:
            raise Phase0Error(
                f"legacy asset role {role!r} must use kind "
                f"{NORMATIVE_ASSET_KIND_BY_ROLE[role]!r}"
            )
        by_role[role] = asset

    roi_asset = by_role["authoritative_yolo_roi"]
    if roi_asset["sha256"] != expected_roi_sha256:
        raise Phase0Error("authoritative YOLO ROI asset differs from the frozen scope ROI")
    authoritative_roi = _validate_roi(Path(roi_asset["source_path"]), ["right"], required_views)

    acquisition_payload = _load_json(Path(by_role["capture_acquisition_config"]["source_path"]))
    try:
        from zs32_inspection.capture.hikvision import HikvisionCaptureConfig

        HikvisionCaptureConfig.from_mapping(acquisition_payload)
    except (ImportError, TypeError, ValueError) as error:
        raise Phase0Error(f"legacy acquisition config violates the strict Hikvision schema: {error}") from error

    fusion_asset = by_role["legacy_fusion_contract"]
    fixed_fusion_reference = "config/fusion/zs32_right_six_view.json"
    fixed_fusion_path = (REPO_ROOT / fixed_fusion_reference).resolve()
    fusion_source = Path(fusion_asset["source_path"]).resolve()
    if fusion_source == fixed_fusion_path:
        fusion_asset["runtime_reference"] = fixed_fusion_reference
    elif fusion_asset.get("runtime_reference") != fixed_fusion_reference:
        raise Phase0Error("legacy fusion asset is not the exact profile hard-coded by Stage 32")
    fusion = _load_json(fusion_source)
    identity = fusion.get("identity")
    ok_requires = fusion.get("ok_requires")
    branches = fusion.get("required_branches_by_view")
    required_view_keys = (
        ok_requires.get("required_view_keys") if isinstance(ok_requires, dict) else None
    )
    if (
        not isinstance(identity, dict)
        or identity.get("product") != EXPECTED_PRODUCT
        or identity.get("allowed_hands") != ["right"]
        or not isinstance(ok_requires, dict)
        or not isinstance(required_view_keys, list)
        or any(not isinstance(value, str) for value in required_view_keys)
        or set(required_view_keys)
        != {f"zs32:{view}" for view in required_views}
        or not isinstance(branches, dict)
        or set(branches) != set(required_views)
        or any(not isinstance(value, list) or not value for value in branches.values())
    ):
        raise Phase0Error("legacy fusion contract does not cover the exact right-hand six-view scope")

    registry_path = Path(by_role["legacy_model_registry"]["source_path"])
    registry = _load_json(registry_path)
    patchcore = registry.get("patchcore")
    yolo = registry.get("yolo")
    if (
        registry.get("schema_version") != 1
        or registry.get("product") != EXPECTED_PRODUCT
        or not isinstance(patchcore, dict)
        or set(patchcore) != set(required_views)
        or not isinstance(yolo, dict)
    ):
        raise Phase0Error("legacy runtime model registry does not define exact ZS32 six-view assets")
    for field, role in (
        ("patchcore_roi_config", "legacy_patchcore_roi"),
        ("yolo_roi_config", "legacy_yolo_roi"),
    ):
        reference = _signed_value(registry.get(field), f"legacy registry {field}")
        resolved_reference = _resolve(reference, REPO_ROOT)
        roi_input_asset = by_role[role]
        original_source = Path(roi_input_asset["source_path"]).resolve()
        recorded_reference = roi_input_asset.get("registry_reference")
        if original_source == resolved_reference:
            if roi_input_asset["sha256"] != _sha256_file(resolved_reference):
                raise Phase0Error(f"legacy registry {field} changed while assets were inventoried")
            roi_input_asset["registry_reference"] = reference
        elif recorded_reference != reference:
            raise Phase0Error(f"legacy registry {field} is not bound to the registered {role} asset")
    try:
        from capture_data.zs32_patchcore_roi_dataset import load_patchcore_roi_config
        from capture_data.zs32_view_roi_dataset import load_roi_config

        pc_width, pc_height, patchcore_rois, _ = load_patchcore_roi_config(
            Path(by_role["legacy_patchcore_roi"]["source_path"])
        )
        yolo_width, yolo_height, legacy_yolo_rois, _ = load_roi_config(
            Path(by_role["legacy_yolo_roi"]["source_path"])
        )
    except (ImportError, OSError, TypeError, ValueError) as error:
        raise Phase0Error(f"legacy runtime ROI assets violate their deployed schemas: {error}") from error
    if (
        (pc_width, pc_height) != (yolo_width, yolo_height)
        or set(patchcore_rois) != {"right"}
        or set(patchcore_rois["right"]) != set(required_views)
        or set(legacy_yolo_rois) != set(required_views)
    ):
        raise Phase0Error("legacy PatchCore/YOLO ROI assets do not cover the exact right six-view scope")
    authoritative_boxes = {
        view: tuple(authoritative_roi["hands"]["right"]["views"][view]["xyxy"])
        for view in required_views
    }
    if {view: tuple(legacy_yolo_rois[view]) for view in required_views} != authoritative_boxes:
        raise Phase0Error("legacy Stage 32 YOLO ROI differs from the authoritative frozen YOLO ROI")
    expected_checkpoint_sha256_by_view: dict[str, str] = {}
    for view in required_views:
        entry = patchcore[view]
        if not isinstance(entry, dict) or set(entry) != {
            "checkpoint",
            "checkpoint_sha256",
            "model_version",
        }:
            raise Phase0Error(f"legacy registry PatchCore entry is malformed for view {view!r}")
        _signed_value(entry.get("checkpoint"), f"legacy registry patchcore.{view}.checkpoint")
        _signed_value(entry.get("model_version"), f"legacy registry patchcore.{view}.model_version")
        digest = entry.get("checkpoint_sha256")
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            raise Phase0Error(f"legacy registry PatchCore hash is invalid for view {view!r}")
        expected_checkpoint_sha256_by_view[view] = digest
    if len(set(expected_checkpoint_sha256_by_view.values())) != len(required_views):
        raise Phase0Error("legacy registry reuses one PatchCore checkpoint across multiple views")

    checkpoint_root = Path(by_role["selected_anomaly_family"]["source_path"])
    if by_role["selected_anomaly_family"].get("model_family") != "patchcore":
        raise Phase0Error("current Phase 0 selected anomaly family must be patchcore")
    checkpoint_files = [path for path in _regular_files(checkpoint_root) if path.suffix == ".ckpt"]
    actual_checkpoint_paths_by_sha256: dict[str, list[Path]] = {}
    for path in checkpoint_files:
        actual_checkpoint_paths_by_sha256.setdefault(_sha256_file(path), []).append(path)
    expected_checkpoint_hashes = set(expected_checkpoint_sha256_by_view.values())
    if (
        len(checkpoint_files) != len(required_views)
        or set(actual_checkpoint_paths_by_sha256) != expected_checkpoint_hashes
        or any(len(paths) != 1 for paths in actual_checkpoint_paths_by_sha256.values())
    ):
        raise Phase0Error(
            "PatchCore checkpoint bundle must contain exactly the six unique registry checkpoints"
        )

    yolo_expected = yolo.get("weights_sha256")
    if not isinstance(yolo_expected, str) or SHA256_PATTERN.fullmatch(yolo_expected) is None:
        raise Phase0Error("legacy registry YOLO weights_sha256 is invalid")
    if by_role["deployment_weight"]["sha256"] != yolo_expected:
        raise Phase0Error(
            "YOLO deployment best.pt hash differs from the legacy runtime registry; "
            "resolve the runtime-registry-asset-path-mismatch blocker"
        )
    if Path(by_role["training_init_weight"]["source_path"]).name != "yolo26n.pt":
        raise Phase0Error("YOLO training init role must point to yolo26n.pt")

    template_root = Path(by_role["template_match"]["source_path"])
    if not template_root.is_dir():
        raise Phase0Error("legacy template_match asset must be one template model directory")
    try:
        import cv2
        from capture_data.zs32_template_gate import _load_group, _validate_match_image, load_model

        template_model = load_model(template_root)
        groups = template_model.get("groups")
        if not isinstance(groups, dict) or set(groups) != {f"right/{view}" for view in required_views}:
            raise Phase0Error("legacy template model must contain exactly the right-hand six-view groups")
        for view in required_views:
            group, _low, _high = _load_group(template_model, "right", view)
            records = group.get("templates")
            if not isinstance(records, list) or not records:
                raise Phase0Error(f"legacy template group right/{view} has no templates")
            for record in records:
                if not isinstance(record, dict):
                    raise Phase0Error(f"legacy template group right/{view} has a malformed record")
                relative = Path(str(record.get("path", "")))
                template_path = (template_root / relative).resolve()
                if relative.is_absolute() or not template_path.is_relative_to(template_root.resolve()):
                    raise Phase0Error(f"legacy template path escapes its model directory: {relative}")
                if not template_path.is_file() or _sha256_file(template_path) != record.get("sha256"):
                    raise Phase0Error(f"legacy template file/hash mismatch: {template_path}")
                image = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise Phase0Error(f"legacy template cannot be decoded: {template_path}")
                _validate_match_image(image, role="template")
    except Phase0Error:
        raise
    except Exception as error:
        raise Phase0Error(f"legacy template deployment contract is invalid: {error}") from error

    threshold_path = Path(by_role["legacy_thresholds"]["source_path"])
    if not threshold_path.is_file() or threshold_path.suffix.lower() != ".json":
        raise Phase0Error("legacy_thresholds must be the exact Stage 31 thresholds.json file")
    stage18_path = REPO_ROOT / "pipeline/18_fuse_inspection_results.py"
    module_spec = importlib.util.spec_from_file_location("zs32_phase0_stage18_validator", stage18_path)
    if module_spec is None or module_spec.loader is None:
        raise Phase0Error("cannot load the authoritative Stage 18 threshold validator")
    stage18 = importlib.util.module_from_spec(module_spec)
    try:
        module_spec.loader.exec_module(stage18)
        stage18._load_threshold_artifact(threshold_path, profile_path=fusion_source)
    except Exception as error:
        raise Phase0Error(f"legacy threshold deployment contract is invalid: {error}") from error

    source_bundle = Path(by_role["yolo_source_snapshot"]["source_path"])
    if source_bundle.suffix != ".bundle" or not source_bundle.is_file():
        raise Phase0Error("YOLO source snapshot must be one Git bundle file")
    _run(["git", "bundle", "verify", str(source_bundle)], cwd=REPO_ROOT)


def _golden_inventory(
    selection_path: Path,
    hands: list[str],
    required_views: list[str],
    image_size: tuple[int, int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selection = _load_json(selection_path)
    if selection.get("schema_version") != 1 or selection.get("product") != EXPECTED_PRODUCT:
        raise Phase0Error("unsupported golden selection identity")
    if selection.get("usage") != "refactor_parity_only":
        raise Phase0Error("legacy golden selection must be explicitly parity-only")
    _safe_id(
        _signed_value(selection.get("selection_id"), "golden selection_id"),
        "golden selection_id",
    )
    cases = selection.get("cases")
    if not isinstance(cases, list) or not cases:
        raise Phase0Error("golden selection requires at least one case")
    seen: set[str] = set()
    seen_image_digests: set[str] = set()
    seen_part_instances: set[str] = set()
    inventory: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            raise Phase0Error("every golden case must be an object")
        case_id = _safe_id(_signed_value(case.get("case_id"), "case_id"), "case_id")
        if case_id in seen:
            raise Phase0Error(f"duplicate golden case_id: {case_id}")
        seen.add(case_id)
        hand = case.get("hand")
        if hand not in hands:
            raise Phase0Error(f"golden case {case_id} hand {hand!r} is outside freeze scope")
        expected = case.get("expected_target_status")
        if expected not in ALLOWED_TARGET_STATUSES:
            raise Phase0Error(f"golden case {case_id} has unsupported expected status {expected!r}")
        label = case.get("label")
        if label not in {"normal", "defect"}:
            raise Phase0Error(f"golden case {case_id} label must be normal or defect")
        if label == "defect":
            _signed_value(case.get("defect_type"), f"golden defect case {case_id} defect_type")
        elif case.get("defect_type") is not None:
            raise Phase0Error(f"golden normal case {case_id} must use defect_type=null")
        scenario = _signed_value(case.get("scenario"), f"golden case {case_id} scenario")
        required_target = REQUIRED_SCENARIO_TARGET_STATUSES.get(scenario)
        if required_target is not None and expected != required_target:
            raise Phase0Error(
                f"golden case {case_id} scenario {scenario!r} requires target "
                f"status {required_target!r}, got {expected!r}"
            )
        identity = case.get("capture_identity")
        if not isinstance(identity, dict):
            raise Phase0Error(f"golden case {case_id} has incomplete capture identity")
        for key in ("session_id", "sample_id", "group_id"):
            _signed_value(identity.get(key), f"golden case {case_id} capture_identity.{key}")
        part_instance_id = _safe_id(
            identity.get("part_instance_id"),
            f"golden case {case_id} capture_identity.part_instance_id",
        )
        if part_instance_id in seen_part_instances:
            raise Phase0Error(f"multiple golden cases reuse part_instance_id {part_instance_id}")
        seen_part_instances.add(part_instance_id)
        provenance = case.get("provenance")
        if not isinstance(provenance, dict):
            raise Phase0Error(f"golden case {case_id} has no provenance object")
        if provenance.get("mode") != "legacy_directory_without_capture_manifest":
            raise Phase0Error(
                f"golden case {case_id} uses an unsupported provenance mode; "
                "this Phase 0 tool only accepts explicitly attested legacy directories"
            )
        if provenance.get("physical_part_identity_verified") is not True:
            raise Phase0Error(f"golden case {case_id} physical part identity is not attested")
        if provenance.get("label_verified") is not True:
            raise Phase0Error(f"golden case {case_id} label is not attested")
        _signed_value(provenance.get("verified_by"), f"golden case {case_id} verified_by")
        verified_at = _signed_timestamp(
            provenance.get("verified_at"),
            f"golden case {case_id} verified_at",
        )
        if datetime.fromisoformat(verified_at.replace("Z", "+00:00")) > datetime.now(timezone.utc):
            raise Phase0Error(f"golden case {case_id} verification time is in the future")
        parity = case.get("parity")
        if not isinstance(parity, dict) or parity.get("mode") not in {
            "exact",
            "semantic",
            "must_change",
            "not_applicable",
        }:
            raise Phase0Error(f"golden case {case_id} has no valid parity mode")
        _signed_value(parity.get("reason"), f"golden case {case_id} parity.reason")
        images = case.get("images")
        if not isinstance(images, dict) or set(images) != set(required_views):
            raise Phase0Error(f"golden case {case_id} must contain exactly every required view")
        image_rows: list[dict[str, Any]] = []
        digests: set[str] = set()
        for view in required_views:
            source = _resolve(images[view], selection_path.parent)
            if not source.is_file() or source.is_symlink():
                raise Phase0Error(f"golden image is missing, non-file, or symlink: {source}")
            dimensions = _png_dimensions(source)
            if dimensions != image_size:
                raise Phase0Error(
                    f"golden image {source} is {dimensions[0]}x{dimensions[1]}, expected "
                    f"{image_size[0]}x{image_size[1]}"
                )
            filename_identity, filename_group_id = _legacy_filename_identity(source, hand, view)
            if filename_identity != identity["sample_id"]:
                raise Phase0Error(
                    f"golden image identity disagrees with case {case_id}: "
                    f"{filename_identity!r} != {identity['sample_id']!r}"
                )
            if filename_group_id != identity["group_id"]:
                raise Phase0Error(
                    f"golden image group disagrees with case {case_id}: "
                    f"{filename_group_id!r} != {identity['group_id']!r}"
                )
            image_session = source.parent.parent.name if source.parent.name == "images" else "legacy-flat"
            if image_session != identity["session_id"]:
                raise Phase0Error(
                    f"golden image session disagrees with case {case_id}: "
                    f"{image_session!r} != {identity['session_id']!r}"
                )
            digest = _sha256_file(source)
            if digest in digests:
                raise Phase0Error(f"golden case {case_id} reuses the same image in multiple views")
            if digest in seen_image_digests:
                raise Phase0Error(f"multiple golden cases reuse image content {source}")
            digests.add(digest)
            seen_image_digests.add(digest)
            image_rows.append(
                {
                    "view": view,
                    "source_path": str(source),
                    "sha256": digest,
                    "size_bytes": source.stat().st_size,
                    "width": dimensions[0],
                    "height": dimensions[1],
                }
            )
        inventory.append(
            {
                "case_id": case_id,
                "hand": hand,
                "label": label,
                "defect_type": case.get("defect_type"),
                "scenario": scenario,
                "expected_target_status": expected,
                "capture_identity": identity,
                "provenance": provenance,
                "parity": parity,
                "images": image_rows,
            }
        )
    return selection, inventory


def _validate_legacy_gate_csv(
    path: Path,
    *,
    case_id: str,
    part_id: str,
    gate: str,
    required_views: list[str],
) -> list[dict[str, str]]:
    """Require one real, case-local six-view Stage 18 gate input."""
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fieldnames = reader.fieldnames
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise Phase0Error(f"cannot read replay {gate} CSV for {case_id}: {error}") from error
    if fieldnames is None or not {"part_id", "view"}.issubset(fieldnames):
        raise Phase0Error(f"replay {gate} CSV for {case_id} must contain part_id and view columns")
    if len(rows) != len(required_views):
        raise Phase0Error(
            f"replay {gate} CSV for {case_id} must contain exactly {len(required_views)} rows"
        )
    if {row.get("part_id", "").strip() for row in rows} != {part_id}:
        raise Phase0Error(f"replay {gate} CSV part_id differs from golden identity for {case_id}")
    views = [row.get("view", "").strip() for row in rows]
    if len(views) != len(set(views)) or set(views) != set(required_views):
        raise Phase0Error(f"replay {gate} CSV for {case_id} must cover each required view once")
    allowed_branches = {
        "quality": {"quality", "quality_gate"},
        "registration": {"registration"},
        "geometry": {"geometry"},
    }[gate]
    declared = {row.get("branch", "").strip() for row in rows if row.get("branch", "").strip()}
    if declared and not declared.issubset(allowed_branches):
        raise Phase0Error(f"replay {gate} CSV for {case_id} declares another branch: {sorted(declared)}")
    if gate in {"quality", "registration"} and any(
        not row.get("status", "").strip() for row in rows
    ):
        raise Phase0Error(f"replay {gate} CSV for {case_id} requires a status for every view")
    return rows


def _gate_has_hard_failure(rows: list[dict[str, str]]) -> bool:
    for row in rows:
        if row.get("status", "").strip().upper() in {
            "FAIL",
            "RETAKE",
            "INVALID_CAPTURE",
        }:
            return True
        for field in ("fail_label", "pred_label"):
            if row.get(field, "").strip().lower() in {"1", "true", "yes"}:
                return True
    return False


def _validate_gate_producer_receipt(
    path: Path,
    *,
    case: dict[str, Any],
    receipt_asset: dict[str, Any],
    gate_assets: dict[str, dict[str, Any]],
    assets_by_id: dict[str, dict[str, Any]],
    required_views: list[str],
) -> dict[str, Any]:
    """Bind real legacy gate outputs to their producer, inputs, and attestation."""
    receipt = _load_json(path)
    expected_fields = {
        "schema_version",
        "product",
        "case_id",
        "generated_at",
        "capture_identity",
        "producer",
        "source_image_sha256_by_view",
        "outputs",
        "attestation",
    }
    if (
        set(receipt) != expected_fields
        or receipt.get("schema_version") != 1
        or receipt.get("product") != EXPECTED_PRODUCT
        or receipt.get("case_id") != case["case_id"]
    ):
        raise Phase0Error(f"legacy gate producer receipt is malformed: {path}")
    generated_at = _signed_timestamp(receipt.get("generated_at"), f"gate receipt generated_at {path}")
    generated_datetime = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    if generated_datetime > datetime.now(timezone.utc):
        raise Phase0Error(f"legacy gate producer receipt is dated in the future: {path}")
    if receipt.get("capture_identity") != case["capture_identity"]:
        raise Phase0Error(f"legacy gate producer receipt capture identity differs: {path}")
    source_hashes = receipt.get("source_image_sha256_by_view")
    expected_source_hashes = {item["view"]: item["sha256"] for item in case["images"]}
    if source_hashes != expected_source_hashes or set(source_hashes or {}) != set(required_views):
        raise Phase0Error(f"legacy gate producer receipt source images differ: {path}")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(LEGACY_GATE_ROLE_BY_NAME):
        raise Phase0Error(f"legacy gate producer receipt outputs are incomplete: {path}")
    for gate, asset in gate_assets.items():
        if outputs.get(gate) != {
            "asset_id": asset["asset_id"],
            "sha256": asset["sha256"],
        }:
            raise Phase0Error(f"legacy gate producer receipt output differs for {gate}: {path}")
    producer = receipt.get("producer")
    if not isinstance(producer, dict) or set(producer) != {
        "command_argv",
        "git_commit",
        "source_asset_id",
        "source_asset_sha256",
        "config_assets",
        "threshold_assets",
    }:
        raise Phase0Error(f"legacy gate producer identity is malformed: {path}")
    command = producer.get("command_argv")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(value, str) or not value for value in command)
    ):
        raise Phase0Error(f"legacy gate producer command is malformed: {path}")
    _reject_placeholders(command, f"legacy gate producer command {path}")
    commit = producer.get("git_commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) is None:
        raise Phase0Error(f"legacy gate producer git commit is invalid: {path}")
    source_id = _safe_id(producer.get("source_asset_id"), f"gate receipt source asset {path}")
    source_asset = assets_by_id.get(source_id)
    if (
        source_asset is None
        or source_asset.get("kind") != "git_source_bundle"
        or source_asset.get("required") is not True
        or source_asset.get("copy_into_bundle") is not True
        or producer.get("source_asset_sha256") != source_asset.get("sha256")
    ):
        raise Phase0Error(f"legacy gate producer source asset is not content-bound: {path}")
    source_bundle = Path(source_asset["source_path"])
    if source_bundle.suffix != ".bundle" or not source_bundle.is_file():
        raise Phase0Error(f"legacy gate producer source must be one Git bundle: {path}")
    _run(["git", "bundle", "verify", str(source_bundle)], cwd=REPO_ROOT)
    heads = _run(["git", "bundle", "list-heads", str(source_bundle)], cwd=REPO_ROOT)
    commits = {line.split(maxsplit=1)[0] for line in heads.splitlines() if line.strip()}
    if commit not in commits:
        raise Phase0Error(f"legacy gate producer commit is not present in its source bundle: {path}")
    for field in ("config_assets", "threshold_assets"):
        references = producer.get(field)
        if not isinstance(references, list) or not references:
            raise Phase0Error(f"legacy gate producer {field} must be non-empty: {path}")
        seen: set[str] = set()
        for reference in references:
            if not isinstance(reference, dict) or set(reference) != {"asset_id", "sha256"}:
                raise Phase0Error(f"legacy gate producer {field} reference is malformed: {path}")
            asset_id = _safe_id(reference.get("asset_id"), f"gate receipt {field} asset")
            if asset_id in seen:
                raise Phase0Error(f"legacy gate producer {field} repeats {asset_id}: {path}")
            seen.add(asset_id)
            asset = assets_by_id.get(asset_id)
            allowed_roles = (
                {
                    "capture_acquisition_config",
                    "authoritative_yolo_roi",
                    "legacy_patchcore_roi",
                    "legacy_yolo_roi",
                    "legacy_fusion_contract",
                    "legacy_model_registry",
                }
                if field == "config_assets"
                else {"legacy_thresholds"}
            )
            if (
                asset is None
                or asset.get("required") is not True
                or asset.get("copy_into_bundle") is not True
                or asset.get("role") not in allowed_roles
                or reference.get("sha256") != asset.get("sha256")
            ):
                raise Phase0Error(f"legacy gate producer {field} is not content-bound: {path}")
    attestation = receipt.get("attestation")
    if not isinstance(attestation, dict) or set(attestation) != {
        "reviewed_by",
        "reviewed_at",
        "basis",
    }:
        raise Phase0Error(f"legacy gate producer attestation is malformed: {path}")
    _signed_value(attestation.get("reviewed_by"), f"gate receipt reviewer {path}")
    _signed_value(attestation.get("basis"), f"gate receipt basis {path}")
    reviewed_at = _signed_timestamp(attestation.get("reviewed_at"), f"gate receipt review time {path}")
    reviewed_datetime = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    if reviewed_datetime < generated_datetime or reviewed_datetime > datetime.now(timezone.utc):
        raise Phase0Error(f"legacy gate producer attestation time is invalid: {path}")
    return {
        "asset_id": receipt_asset["asset_id"],
        "source_path": str(path),
        "sha256": receipt_asset["sha256"],
        "generated_at": generated_at,
        "reviewed_at": reviewed_at,
    }


def _validate_legacy_replay_plan(
    assets: list[dict[str, Any]],
    golden: list[dict[str, Any]],
    required_views: list[str],
) -> dict[str, Any]:
    """Bind Stage 32 replay controls to registered, immutable input assets."""
    plan_assets = [
        asset
        for asset in assets
        if asset.get("present") is True and asset.get("role") == LEGACY_REPLAY_PLAN_ROLE
    ]
    if len(plan_assets) != 1:
        raise Phase0Error("Phase 0 requires exactly one registered legacy_replay_plan asset")
    plan_asset = plan_assets[0]
    if (
        plan_asset.get("kind") != "replay_plan"
        or plan_asset.get("required") is not True
        or plan_asset.get("copy_into_bundle") is not True
    ):
        raise Phase0Error("legacy_replay_plan must be a required replay_plan copied into the bundle")
    plan_path = Path(plan_asset["source_path"])
    if not plan_path.is_file():
        raise Phase0Error("legacy_replay_plan must be one JSON file")
    plan = _load_json(plan_path)
    expected_fields = {
        "schema_version",
        "product",
        "entrypoint",
        "accelerator",
        "devices",
        "yolo_device",
        "cases",
    }
    if (
        set(plan) != expected_fields
        or plan.get("schema_version") != 1
        or plan.get("product") != EXPECTED_PRODUCT
        or plan.get("entrypoint") != LEGACY_REPLAY_ENTRYPOINT
    ):
        raise Phase0Error("legacy replay plan identity or fields differ from the Stage 32 contract")
    accelerator = _signed_value(plan.get("accelerator"), "replay plan accelerator")
    if accelerator != "gpu":
        raise Phase0Error("legacy replay plan accelerator must be explicitly gpu")
    devices = plan.get("devices")
    if not isinstance(devices, int) or isinstance(devices, bool) or devices < 1:
        raise Phase0Error("replay plan devices must be a positive integer")
    yolo_device = _signed_value(plan.get("yolo_device"), "replay plan yolo_device")
    if not yolo_device.isdigit():
        raise Phase0Error("legacy replay plan yolo_device must be one explicit numeric GPU index")
    cases = plan.get("cases")
    if not isinstance(cases, list) or any(not isinstance(item, dict) for item in cases):
        raise Phase0Error("legacy replay plan cases must be an object array")
    golden_by_id = {case["case_id"]: case for case in golden}
    case_ids = [item.get("case_id") for item in cases]
    if (
        any(not isinstance(case_id, str) for case_id in case_ids)
        or len(case_ids) != len(set(case_ids))
        or set(case_ids) != set(golden_by_id)
    ):
        raise Phase0Error("legacy replay plan must cover every golden case exactly once")
    assets_by_id = {asset["asset_id"]: asset for asset in assets if asset.get("present") is True}
    used_gate_assets: set[str] = set()
    used_receipt_assets: set[str] = set()
    case_inventory: list[dict[str, Any]] = []
    for item in cases:
        if set(item) != {
            "case_id",
            "quality_asset_id",
            "registration_asset_id",
            "geometry_asset_id",
            "gate_producer_receipt_asset_id",
        }:
            raise Phase0Error(f"legacy replay case fields differ from strict schema: {item.get('case_id')!r}")
        case_id = item["case_id"]
        golden_case = golden_by_id[case_id]
        inputs: dict[str, dict[str, Any]] = {}
        gate_rows: dict[str, list[dict[str, str]]] = {}
        for gate, expected_role in LEGACY_GATE_ROLE_BY_NAME.items():
            asset_id = _safe_id(item.get(f"{gate}_asset_id"), f"replay {case_id} {gate}_asset_id")
            if asset_id in used_gate_assets:
                raise Phase0Error(f"legacy replay gate asset is reused across cases: {asset_id}")
            used_gate_assets.add(asset_id)
            asset = assets_by_id.get(asset_id)
            if asset is None:
                raise Phase0Error(f"legacy replay plan references missing asset {asset_id}")
            if (
                asset.get("kind") != "branch_evidence_csv"
                or asset.get("role") != expected_role
                or asset.get("required") is not True
                or asset.get("copy_into_bundle") is not True
            ):
                raise Phase0Error(
                    f"legacy replay asset {asset_id} must be a required copied {expected_role} CSV"
                )
            source = Path(asset["source_path"])
            if not source.is_file() or source.suffix.lower() != ".csv":
                raise Phase0Error(f"legacy replay asset {asset_id} must be one CSV file")
            gate_rows[gate] = _validate_legacy_gate_csv(
                source,
                case_id=case_id,
                part_id=golden_case["capture_identity"]["part_instance_id"],
                gate=gate,
                required_views=required_views,
            )
            inputs[gate] = {
                "asset_id": asset_id,
                "source_path": str(source),
                "sha256": asset["sha256"],
                "size_bytes": asset["size_bytes"],
            }
        quality_failed = _gate_has_hard_failure(gate_rows["quality"])
        registration_failed = _gate_has_hard_failure(gate_rows["registration"])
        scenario = golden_case["scenario"]
        if scenario == "quality_retake" and not quality_failed:
            raise Phase0Error(f"golden case {case_id} does not contain a real quality failure")
        if scenario == "registration_retake" and (
            quality_failed or not registration_failed
        ):
            raise Phase0Error(
                f"golden case {case_id} must isolate a registration failure after quality passes"
            )
        if scenario not in {"quality_retake", "registration_retake"} and (
            quality_failed or registration_failed
        ):
            raise Phase0Error(
                f"golden model/template scenario {case_id} is contaminated by a capture gate failure"
            )
        receipt_id = _safe_id(
            item.get("gate_producer_receipt_asset_id"),
            f"replay {case_id} gate_producer_receipt_asset_id",
        )
        if receipt_id in used_receipt_assets:
            raise Phase0Error(f"legacy gate producer receipt is reused across cases: {receipt_id}")
        used_receipt_assets.add(receipt_id)
        receipt_asset = assets_by_id.get(receipt_id)
        if (
            receipt_asset is None
            or receipt_asset.get("kind") != "producer_receipt"
            or receipt_asset.get("role") != LEGACY_GATE_RECEIPT_ROLE
            or receipt_asset.get("required") is not True
            or receipt_asset.get("copy_into_bundle") is not True
        ):
            raise Phase0Error(
                f"legacy replay case {case_id} requires one copied gate producer receipt"
            )
        receipt_path = Path(receipt_asset["source_path"])
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise Phase0Error(f"legacy gate producer receipt is missing or unsafe: {receipt_path}")
        receipt = _validate_gate_producer_receipt(
            receipt_path,
            case=golden_case,
            receipt_asset=receipt_asset,
            gate_assets={gate: assets_by_id[record["asset_id"]] for gate, record in inputs.items()},
            assets_by_id=assets_by_id,
            required_views=required_views,
        )
        case_inventory.append({"case_id": case_id, "inputs": inputs, "producer_receipt": receipt})
    declared_gate_assets = {
        asset["asset_id"]
        for asset in assets
        if asset.get("present") is True and asset.get("role") in set(LEGACY_GATE_ROLE_BY_NAME.values())
    }
    if declared_gate_assets != used_gate_assets:
        raise Phase0Error(
            "legacy replay gate assets must be referenced exactly once; "
            f"unused={sorted(declared_gate_assets - used_gate_assets)}, "
            f"unregistered={sorted(used_gate_assets - declared_gate_assets)}"
        )
    declared_receipt_assets = {
        asset["asset_id"]
        for asset in assets
        if asset.get("present") is True and asset.get("role") == LEGACY_GATE_RECEIPT_ROLE
    }
    if declared_receipt_assets != used_receipt_assets:
        raise Phase0Error(
            "legacy gate producer receipts must be referenced exactly once; "
            f"unused={sorted(declared_receipt_assets - used_receipt_assets)}, "
            f"unregistered={sorted(used_receipt_assets - declared_receipt_assets)}"
        )
    return {
        "asset_id": plan_asset["asset_id"],
        "path": str(plan_path),
        "sha256": plan_asset["sha256"],
        "entrypoint": LEGACY_REPLAY_ENTRYPOINT,
        "accelerator": accelerator,
        "devices": devices,
        "yolo_device": yolo_device,
        "cases": sorted(case_inventory, key=lambda item: item["case_id"]),
    }


def _validate_replay_evidence_envelope(
    path: Path,
    *,
    replay_root: Path,
    case_id: str,
    run_id: str,
    execution_id: str,
    executed_at: str,
) -> dict[str, Any]:
    payload = _load_json(path)
    expected_fields = {
        "schema_version",
        "case_id",
        "run_id",
        "execution_id",
        "captured_at",
        "payload",
    }
    if set(payload) != expected_fields or payload.get("schema_version") != 1:
        raise Phase0Error(f"baseline evidence envelope fields differ from strict schema: {path}")
    if (
        payload.get("case_id") != case_id
        or payload.get("run_id") != run_id
        or payload.get("execution_id") != execution_id
        or payload.get("captured_at") != executed_at
    ):
        raise Phase0Error(f"baseline evidence identity differs from case result: {path}")
    evidence = payload.get("payload")
    if not isinstance(evidence, dict) or not evidence:
        raise Phase0Error(f"baseline evidence payload must be a non-empty object: {path}")
    _reject_placeholders(evidence, f"baseline evidence {path}.payload")
    relative = evidence.get("artifact_relative_path")
    digest = evidence.get("artifact_sha256")
    if (relative is None) != (digest is None):
        raise Phase0Error(f"baseline artifact path/hash must be present together: {path}")
    if relative is not None:
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or "\\" in relative
            or ".." in Path(relative).parts
            or not isinstance(digest, str)
            or SHA256_PATTERN.fullmatch(digest) is None
        ):
            raise Phase0Error(f"baseline artifact reference is unsafe: {path}")
        candidate = (replay_root / relative).resolve()
        try:
            candidate.relative_to(replay_root.resolve())
        except ValueError as error:
            raise Phase0Error(f"baseline artifact escapes replay root: {path}") from error
        if candidate.is_symlink() or not candidate.is_file():
            raise Phase0Error(f"baseline artifact is missing or not a regular file: {candidate}")
        if _sha256_file(candidate) != digest:
            raise Phase0Error(f"baseline artifact SHA256 mismatch: {candidate}")
    return evidence


def _validate_normative_scenario_observed(
    scenario: str,
    observed: dict[str, Any],
    *,
    case_id: str,
) -> None:
    """Require the named golden scenario to occur in raw-derived branch evidence."""
    groups = observed.get("evidence_groups")
    if not isinstance(groups, dict):
        raise Phase0Error(f"golden scenario {case_id} has no observed evidence groups")

    def levels(branch: str) -> list[str]:
        group = groups.get(branch)
        if not isinstance(group, dict) or group.get("state") == "NOT_RUN":
            return []
        values: list[str] = []
        for row in group.values():
            if not isinstance(row, dict):
                raise Phase0Error(f"golden scenario {case_id} has malformed {branch} evidence")
            level = row.get("evidence_level")
            if not isinstance(level, str):
                raise Phase0Error(f"golden scenario {case_id} lacks {branch} evidence level")
            values.append(level.upper())
        return values

    status = observed.get("final_status")
    template = levels("template")
    anomaly = levels("anomaly")
    yolo = levels("yolo")
    all_levels = (*template, *anomaly, *yolo)
    valid = False
    if scenario == "normal_clear":
        valid = status == "OK" and bool(anomaly) and bool(yolo) and set(all_levels) == {"CLEAR"}
    elif scenario == "template_mismatch":
        valid = (
            status == "NG_TEMPLATE"
            and "STRONG" in template
            and not anomaly
            and not yolo
        )
    elif scenario == "anomaly_strong":
        valid = (
            status == "NG_ANOMALY"
            and "STRONG" in anomaly
            and bool(yolo)
            and set(template) == {"CLEAR"}
        )
    elif scenario == "yolo_strong":
        valid = (
            status == "NG_YOLO"
            and "STRONG" in yolo
            and bool(anomaly)
            and set(template) == {"CLEAR"}
        )
    elif scenario == "gray_review":
        valid = (
            status == "REVIEW"
            and bool(anomaly)
            and bool(yolo)
            and "GRAY" in all_levels
            and "STRONG" not in all_levels
        )
    elif scenario in {"quality_retake", "registration_retake"}:
        valid = status in {"RETAKE", "INVALID_CAPTURE", "REVIEW"}
    if not valid:
        raise Phase0Error(
            f"golden case {case_id} did not actually exercise normative scenario {scenario!r}"
        )


def _legacy_replay_command(
    *,
    capture_identity: dict[str, str],
    hand: str,
    required_views: list[str],
    images: dict[str, str],
    runtime_inputs: dict[str, str],
    accelerator: str,
    devices: int,
    yolo_device: str,
) -> list[str]:
    command = [
        sys.executable,
        str((REPO_ROOT / LEGACY_REPLAY_ENTRYPOINT).resolve()),
        "fuse",
        "--part-id",
        capture_identity["part_instance_id"],
        "--capture-session",
        capture_identity["session_id"],
        "--group-id",
        capture_identity["group_id"],
        "--hand",
        hand,
    ]
    for view in required_views:
        command.extend([f"--{view.replace('_', '-')}-image", images[view]])
    command.extend(
        [
            "--runtime-config",
            runtime_inputs["runtime_config"],
            "--output-dir",
            "legacy_output",
            "--template-model-dir",
            runtime_inputs["template_model_dir"],
            "--quality-csv",
            "legacy_inputs/quality.csv",
            "--registration-csv",
            "legacy_inputs/registration.csv",
            "--geometry-csv",
            "legacy_inputs/geometry.csv",
            "--threshold-artifact",
            runtime_inputs["threshold_artifact"],
            "--accelerator",
            accelerator,
            "--devices",
            str(devices),
            "--yolo-device",
            yolo_device,
        ]
    )
    return command


def _baseline_inventory(
    spec: dict[str, Any],
    base: Path,
    expected_cases: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    entries = spec.get("baseline_outputs")
    if not isinstance(entries, list) or not entries:
        raise Phase0Error("freeze spec requires baseline_outputs")
    inventory: list[dict[str, Any]] = []
    minimum_replays = spec.get("minimum_replays_per_case")
    if not isinstance(minimum_replays, int) or minimum_replays < 2:
        raise Phase0Error("minimum_replays_per_case must be an integer greater than or equal to 2")
    case_ids = set(expected_cases)
    replay_counts = dict.fromkeys(case_ids, 0)
    semantic_signatures: dict[str, str] = {}
    command_by_case: dict[str, tuple[str, ...]] = {}
    replay_review_subjects: dict[str, list[dict[str, str]]] = {
        case_id: [] for case_id in case_ids
    }
    inventory_by_case: dict[str, list[dict[str, Any]]] = {
        case_id: [] for case_id in case_ids
    }
    latest_execution_by_case: dict[str, datetime] = {}
    score_comparison_required_by_case: dict[str, bool] = {}
    seen_execution_ids: set[str] = set()
    seen_execution_times: set[tuple[str, str]] = set()
    seen: set[tuple[str, str]] = set()
    for item in entries:
        if not isinstance(item, dict):
            raise Phase0Error("every baseline output entry must be an object")
        if set(item) != {"case_id", "run_id", "path", "required"} or item.get("required") is not True:
            raise Phase0Error(
                "every baseline output must contain exactly case_id/run_id/path/required=true"
            )
        case_id = _safe_id(item.get("case_id"), "baseline case_id")
        run_id = _safe_id(item.get("run_id"), "baseline run_id")
        if case_id not in case_ids:
            raise Phase0Error(f"baseline output references unknown golden case {case_id}")
        identity = (case_id, run_id)
        if identity in seen:
            raise Phase0Error(f"duplicate baseline output for case/run {case_id}/{run_id}")
        source = _resolve(item.get("path", ""), base)
        if not source.exists():
            raise Phase0Error(f"baseline output is missing for {case_id}/{run_id}: {source}")
        if not source.is_dir():
            raise Phase0Error(f"baseline output must be a directory: {source}")
        missing_evidence = sorted(
            name for name in REQUIRED_BASELINE_EVIDENCE_FILES if not (source / name).is_file()
        )
        if missing_evidence:
            raise Phase0Error(
                f"baseline output lacks required evidence for {case_id}/{run_id}: "
                f"{missing_evidence}"
            )
        replay_root = _verify_replay_root(source)
        result_path = source / "phase0_case_result.json"
        result = _load_json(result_path)
        expected_result_fields = {
            "schema_version",
            "case_id",
            "run_id",
            "execution_id",
            "executed_at",
            "command_argv",
            "exit_code",
            "observed_legacy",
            "target_contract_expected",
            "parity",
            "source_binding",
        }
        if (
            set(result) != expected_result_fields
            or result.get("schema_version") != 1
            or result.get("case_id") != case_id
            or result.get("run_id") != run_id
        ):
            raise Phase0Error(f"case result fields or identity mismatch: {result_path}")
        command_argv = result.get("command_argv")
        if (
            not isinstance(command_argv, list)
            or not command_argv
            or any(not isinstance(value, str) for value in command_argv)
        ):
            raise Phase0Error(f"case result command_argv must be a non-empty string array: {result_path}")
        _reject_placeholders(command_argv, f"case result command_argv {result_path}")
        command_identity = tuple(command_argv)
        previous_command = command_by_case.setdefault(case_id, command_identity)
        if command_identity != previous_command:
            raise Phase0Error(f"baseline replays for {case_id} use different commands")
        execution_id = _safe_id(result.get("execution_id"), "case result execution_id")
        executed_at = _signed_timestamp(
            result.get("executed_at"),
            f"case result executed_at {result_path}",
        )
        if execution_id in seen_execution_ids:
            raise Phase0Error(f"baseline replays reuse execution_id {execution_id}")
        execution_time_identity = (case_id, executed_at)
        if execution_time_identity in seen_execution_times:
            raise Phase0Error(f"baseline replays for {case_id} reuse executed_at {executed_at}")
        seen_execution_ids.add(execution_id)
        seen_execution_times.add(execution_time_identity)
        executed_datetime = datetime.fromisoformat(executed_at.replace("Z", "+00:00"))
        if executed_datetime > datetime.now(timezone.utc):
            raise Phase0Error(f"baseline replay execution time is in the future: {result_path}")
        previous_latest = latest_execution_by_case.get(case_id)
        if previous_latest is None or executed_datetime > previous_latest:
            latest_execution_by_case[case_id] = executed_datetime
        envelope_payloads: dict[str, dict[str, Any]] = {}
        for name in sorted(REQUIRED_BASELINE_JSON_EVIDENCE_FILES):
            envelope_payloads[name] = _validate_replay_evidence_envelope(
                source / name,
                replay_root=source,
                case_id=case_id,
                run_id=run_id,
                execution_id=execution_id,
                executed_at=executed_at,
            )
        if result.get("exit_code") != 0:
            raise Phase0Error(f"successful legacy replay must record exit_code=0: {result_path}")
        observed = result.get("observed_legacy")
        target = result.get("target_contract_expected")
        parity = result.get("parity")
        if (
            not isinstance(observed, dict)
            or set(observed) != {"final_status", "evidence_groups"}
            or not isinstance(target, dict)
            or set(target) != {"final_status", "reason"}
            or not isinstance(parity, dict)
            or set(parity) != {"mode"}
        ):
            raise Phase0Error(
                f"case result must separate observed_legacy, target_contract_expected and parity: {result_path}"
            )
        claimed_status = observed.get("final_status")
        if claimed_status not in ALLOWED_LEGACY_STATUSES:
            raise Phase0Error(f"case result observed status is unsupported: {claimed_status!r}")
        claimed_groups = observed.get("evidence_groups")
        if not isinstance(claimed_groups, dict) or set(claimed_groups) != {
            "template",
            "anomaly",
            "yolo",
        }:
            raise Phase0Error(
                f"case result observed legacy evidence must contain template/anomaly/yolo: {result_path}"
            )
        expected_case = expected_cases[case_id]
        required_views = expected_case.get("required_views")
        if (
            not isinstance(required_views, list)
            or not required_views
            or any(not isinstance(view, str) for view in required_views)
        ):
            raise Phase0Error(f"expected replay views are malformed for case {case_id}")
        source_binding = expected_case["source_binding"]
        expected_command = _legacy_replay_command(
            capture_identity=source_binding["capture_identity"],
            hand="right",
            required_views=required_views,
            images={view: f"golden_inputs/{view}.png" for view in required_views},
            runtime_inputs={
                "runtime_config": "runtime_inputs/runtime_models.json",
                "template_model_dir": "runtime_inputs/templates",
                "threshold_artifact": "runtime_inputs/thresholds.json",
            },
            accelerator=expected_case["replay_runtime"]["accelerator"],
            devices=expected_case["replay_runtime"]["devices"],
            yolo_device=expected_case["replay_runtime"]["yolo_device"],
        )
        if command_argv != expected_command:
            raise Phase0Error(
                f"baseline replay command differs from the fixed Stage 32 runner for {case_id}"
            )
        observed = _derive_replay_observed(
            source,
            result=result,
            evidence=envelope_payloads,
            required_views=required_views,
        )
        _validate_normative_scenario_observed(
            expected_case["scenario"],
            observed,
            case_id=case_id,
        )
        observed_status = observed["final_status"]
        groups_for_review = observed["evidence_groups"]
        scores_required = not (
            groups_for_review.get("anomaly", {}).get("state") == "NOT_RUN"
            and groups_for_review.get("yolo", {}).get("state") == "NOT_RUN"
        )
        previous_scores_required = score_comparison_required_by_case.setdefault(
            case_id,
            scores_required,
        )
        if previous_scores_required != scores_required:
            raise Phase0Error(f"legacy replay execution path changed for case {case_id}")
        if observed_status not in ALLOWED_LEGACY_STATUSES:
            raise Phase0Error(f"case result observed status is unsupported: {observed_status!r}")
        evidence_groups = observed.get("evidence_groups")
        if not isinstance(evidence_groups, dict) or set(evidence_groups) != {
            "template",
            "anomaly",
            "yolo",
        }:
            raise Phase0Error(
                f"case result observed legacy evidence must contain template/anomaly/yolo: {result_path}"
            )
        _reject_placeholders(
            evidence_groups,
            f"case result observed_legacy.evidence_groups {result_path}",
        )
        source_binding = result.get("source_binding")
        if not isinstance(source_binding, dict) or source_binding != expected_case["source_binding"]:
            raise Phase0Error(
                f"case result source binding differs from frozen Git/assets/golden inputs: {result_path}"
            )
        if target.get("final_status") != expected_case["target_status"]:
            raise Phase0Error(
                f"case result target status disagrees with golden selection for {case_id}: "
                f"{target.get('final_status')!r} != {expected_case['target_status']!r}"
            )
        _signed_value(target.get("reason"), f"case result target reason {result_path}")
        if parity.get("mode") != expected_case["parity_mode"]:
            raise Phase0Error(
                f"case result parity mode disagrees with golden selection for {case_id}: "
                f"{parity.get('mode')!r} != {expected_case['parity_mode']!r}"
            )
        if parity["mode"] == "exact" and observed_status != expected_case["target_status"]:
            raise Phase0Error(
                f"exact parity case {case_id} observed {observed_status!r}, "
                f"expected {expected_case['target_status']!r}"
            )
        semantic_payload = {
            "final_status": observed["final_status"],
            "evidence_groups": observed["evidence_groups"],
        }
        semantic_signature = _canonical_sha256(semantic_payload)
        previous_signature = semantic_signatures.setdefault(case_id, semantic_signature)
        if previous_signature != semantic_signature:
            raise Phase0Error(f"legacy replay semantic result changed for case {case_id}")
        digest, size, file_count = _path_digest(source)
        seen.add(identity)
        replay_counts[case_id] += 1
        record = {
            "case_id": case_id,
            "run_id": run_id,
            "source_path": str(source),
            "sha256": digest,
            "size_bytes": size,
            "file_count": file_count,
            "replay_root_sha256": replay_root["replay_root_sha256"],
            "semantic_signature": semantic_signature,
        }
        inventory.append(record)
        inventory_by_case[case_id].append(record)
        replay_review_subjects[case_id].append(
            {
                "run_id": run_id,
                "replay_root_sha256": replay_root["replay_root_sha256"],
                "semantic_signature": semantic_signature,
            }
        )
    insufficient = {
        case_id: count for case_id, count in replay_counts.items() if count < minimum_replays
    }
    if insufficient:
        raise Phase0Error(
            f"baseline outputs have fewer than {minimum_replays} replays: {insufficient}"
        )
    reviews = spec.get("baseline_reviews")
    if not isinstance(reviews, list) or any(not isinstance(item, dict) for item in reviews):
        raise Phase0Error("freeze spec requires one external baseline review receipt per golden case")
    review_case_ids = [item.get("case_id") for item in reviews]
    if (
        any(not isinstance(value, str) for value in review_case_ids)
        or len(review_case_ids) != len(set(review_case_ids))
        or set(review_case_ids) != case_ids
    ):
        raise Phase0Error("baseline review receipts must cover every golden case exactly once")
    for item in reviews:
        if set(item) != {"case_id", "path"}:
            raise Phase0Error("baseline review spec entries must contain exactly case_id and path")
        case_id = item["case_id"]
        receipt_path = _resolve(item.get("path", ""), base)
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise Phase0Error(f"baseline review receipt must be a regular non-symlink file: {receipt_path}")
        receipt = _load_json(receipt_path)
        if (
            set(receipt)
            != {
                "schema_version",
                "product",
                "case_id",
                "reviewed_at",
                "reviewer",
                "decision",
                "replays",
                "score_comparison",
            }
            or receipt.get("schema_version") != 1
            or receipt.get("product") != EXPECTED_PRODUCT
            or receipt.get("case_id") != case_id
            or receipt.get("decision") != "APPROVED"
        ):
            raise Phase0Error(f"baseline review receipt is malformed or not approved: {receipt_path}")
        reviewed_at = _signed_timestamp(
            receipt.get("reviewed_at"),
            f"baseline review reviewed_at {receipt_path}",
        )
        reviewed_datetime = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
        if reviewed_datetime < latest_execution_by_case[case_id]:
            raise Phase0Error(f"baseline review predates a bound replay: {receipt_path}")
        if reviewed_datetime > datetime.now(timezone.utc):
            raise Phase0Error(f"baseline review timestamp is in the future: {receipt_path}")
        _signed_value(receipt.get("reviewer"), f"baseline review reviewer {receipt_path}")
        score_comparison = receipt.get("score_comparison")
        if (
            not isinstance(score_comparison, dict)
            or set(score_comparison) != {"decision", "basis"}
            or score_comparison.get("decision") not in {"WITHIN_TOLERANCE", "NOT_APPLICABLE"}
        ):
            raise Phase0Error(f"baseline review score comparison is malformed: {receipt_path}")
        required_decision = (
            "WITHIN_TOLERANCE"
            if score_comparison_required_by_case[case_id]
            else "NOT_APPLICABLE"
        )
        if score_comparison.get("decision") != required_decision:
            raise Phase0Error(
                f"baseline review score decision must be {required_decision}: {receipt_path}"
            )
        _signed_value(score_comparison.get("basis"), f"baseline review score basis {receipt_path}")
        expected_subjects = sorted(replay_review_subjects[case_id], key=lambda value: value["run_id"])
        actual_subjects = receipt.get("replays")
        if (
            not isinstance(actual_subjects, list)
            or any(not isinstance(value, dict) for value in actual_subjects)
            or sorted(actual_subjects, key=lambda value: str(value.get("run_id")))
            != expected_subjects
        ):
            raise Phase0Error(f"baseline review does not bind the exact immutable replays: {receipt_path}")
        receipt_sha256 = _sha256_file(receipt_path)
        for record in inventory_by_case[case_id]:
            record["review_receipt_path"] = str(receipt_path)
            record["review_receipt_sha256"] = receipt_sha256
            record["review_receipt_reviewed_at"] = reviewed_at
    return inventory


def _load_and_validate_spec(spec_path: Path) -> dict[str, Any]:
    spec = _load_json(spec_path)
    if spec.get("schema_version") != 1 or spec.get("product") != EXPECTED_PRODUCT:
        raise Phase0Error("unsupported Phase 0 freeze spec")
    _safe_id(spec.get("freeze_id"), "freeze_id")
    expected_git = spec.get("expected_git")
    if not isinstance(expected_git, dict):
        raise Phase0Error("expected_git must be an object")
    scope = spec.get("scope")
    if not isinstance(scope, dict) or scope.get("usage") != "refactor_parity_only":
        raise Phase0Error("freeze scope must be explicitly refactor_parity_only")
    hands = scope.get("hands")
    if hands != ["right"]:
        raise Phase0Error("the current Phase 0 freeze scope must be exactly hands=['right']")
    if not isinstance(spec.get("output_root"), str) or not spec["output_root"]:
        raise Phase0Error("output_root must be a non-empty path string")
    return spec


def discover_legacy_candidates(
    *,
    data_root: Path,
    topology_path: Path,
    hand: str,
    output: Path,
) -> dict[str, Any]:
    """Build a manual-review list from the legacy hand/view directory layout."""
    require_linux_nvidia()
    _, required_views = _validate_topology(topology_path)
    if hand != "right":
        raise Phase0Error("current Phase 0 discovery is right-only; left ROI remains pending")
    hand_root = data_root.resolve() / hand
    if not hand_root.is_dir():
        raise Phase0Error(f"legacy hand directory does not exist: {hand_root}")
    grouped: dict[str, dict[str, Path]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    ignored = 0
    for view in required_views:
        view_root = hand_root / view
        if not view_root.is_dir():
            raise Phase0Error(f"legacy view directory does not exist: {view_root}")
        prefix = f"{hand}_{view}_"
        for image in sorted(view_root.rglob("*.png")):
            if image.is_symlink() or not image.is_file() or not image.name.startswith(prefix):
                ignored += 1
                continue
            relative_parts = image.relative_to(view_root).parts
            label = relative_parts[0] if relative_parts else "unknown"
            defect_type = relative_parts[1] if label == "defect" and len(relative_parts) > 1 else None
            session_id = image.parent.parent.name if image.parent.name == "images" else "legacy-flat"
            filename_tail = image.name[len(prefix) :]
            suffix_match = re.search(r"_(group\d+)_(\d{6})_(?:single|fused)\.png$", filename_tail)
            if suffix_match is None:
                ignored += 1
                continue
            group_id, image_index = suffix_match.groups()
            sample_token = filename_tail[: suffix_match.start()]
            key = "|".join((session_id, label, defect_type or "", sample_token, group_id, image_index))
            view_map = grouped.setdefault(key, {})
            if view in view_map:
                raise Phase0Error(f"duplicate legacy image for candidate {key} view {view}")
            view_map[view] = image.resolve()
            metadata[key] = {
                "session_id": session_id,
                "sample_id": f"{sample_token}_{group_id}_{image_index}",
                "group_id": group_id,
                "label": label,
                "defect_type": defect_type,
            }
    complete_keys = sorted(key for key, images in grouped.items() if set(images) == set(required_views))
    incomplete_keys = sorted(set(grouped) - set(complete_keys))
    cases: list[dict[str, Any]] = []
    for index, key in enumerate(complete_keys, start=1):
        info = metadata[key]
        cases.append(
            {
                "case_id": f"candidate-{hand}-{index:04d}",
                "hand": hand,
                "label": info["label"],
                "defect_type": info["defect_type"],
                "scenario": None,
                "expected_target_status": None,
                "capture_identity": {
                    "session_id": info["session_id"],
                    "sample_id": info["sample_id"],
                    "group_id": info["group_id"],
                    "part_instance_id": "PENDING_MANUAL_ATTESTATION",
                },
                "provenance": {
                    "mode": "legacy_directory_without_capture_manifest",
                    "source_manifest": None,
                    "physical_part_identity_verified": False,
                    "label_verified": False,
                    "verified_by": "",
                    "verified_at": "",
                    "notes": "Manual review is mandatory; parity-only, not an independent accuracy test.",
                },
                "images": {view: str(grouped[key][view]) for view in required_views},
                "parity": {"mode": None, "reason": ""},
            }
        )
    payload = {
        "schema_version": 1,
        "selection_id": f"zs32-{hand}-legacy-candidates",
        "product": EXPECTED_PRODUCT,
        "usage": "refactor_parity_only",
        "discovery": {
            "data_root": str(data_root.resolve()),
            "topology": str(topology_path.resolve()),
            "complete_candidate_count": len(complete_keys),
            "incomplete_group_count": len(incomplete_keys),
            "ignored_file_count": ignored,
            "instructions": "Copy selected cases to a reviewed selection file; do not approve all candidates blindly.",
        },
        "cases": cases,
    }
    _write_json(output.resolve(), payload)
    return payload


def _replay_source_bindings(
    *,
    repository: dict[str, Any],
    repository_contracts: dict[str, dict[str, str]],
    topology_sha256: str,
    roi_sha256: str,
    assets: list[dict[str, Any]],
    golden: list[dict[str, Any]],
    environment_sha256: str,
) -> dict[str, dict[str, Any]]:
    common = {
        "git_commit": repository["commit"],
        "git_tree": repository["tree"],
        "uv_lock_sha256": repository_contracts["uv.lock"]["sha256"],
        "topology_sha256": topology_sha256,
        "roi_sha256": roi_sha256,
        "environment_sha256": environment_sha256,
        "asset_sha256_by_id": {
            asset["asset_id"]: asset["sha256"]
            for asset in sorted(assets, key=lambda value: value["asset_id"])
            if asset.get("present") is True
        },
    }
    return {
        case["case_id"]: {
            **common,
            "capture_identity": case["capture_identity"],
            "golden_image_sha256_by_view": {
                image["view"]: image["sha256"] for image in case["images"]
            },
        }
        for case in golden
    }


def build_replay_bindings(spec_path: Path, *, output: Path) -> dict[str, Any]:
    """Publish exact source bindings before the first real legacy replay is run."""
    spec_sha256 = _sha256_file(spec_path)
    gpu = require_linux_nvidia()
    spec = _load_and_validate_spec(spec_path)
    _validate_normative_spec_requirements(spec, require_resolved_blockers=False)
    base = spec_path.parent
    scope = spec["scope"]
    topology_path = _resolve(scope["topology_config"], base)
    roi_path = _resolve(scope["roi_config"], base)
    selection_path = _resolve(scope["golden_selection"], base)
    topology, required_views = _validate_topology(topology_path)
    if (
        len(topology["rounds"]) != 2
        or len(topology["camera_slots"]) != 3
        or required_views
        != ["front", "front_left", "front_right", "back", "back_left", "back_right"]
    ):
        raise Phase0Error("replay bindings require the frozen 3-camera/6-view topology")
    roi = _validate_roi(roi_path, ["right"], required_views)
    if (
        roi.get("topology_id") != topology.get("topology_id")
        or roi.get("source_image_size") != {"width": 4024, "height": 3036}
        or roi["hands"]["left"].get("status") != "pending"
        or roi["hands"]["left"].get("views") != {}
    ):
        raise Phase0Error("replay binding ROI/topology scope differs from frozen Phase 0")
    _selection, golden = _golden_inventory(
        selection_path,
        ["right"],
        required_views,
        (4024, 3036),
    )
    assets = _asset_inventory(spec, base)
    for role in NORMATIVE_REQUIRED_ASSET_ROLES:
        matching = [
            asset
            for asset in assets
            if asset.get("role") == role
            and asset.get("present") is True
            and asset.get("required") is True
        ]
        if len(matching) != 1:
            raise Phase0Error(f"replay binding requires exactly one required asset role {role!r}")
    _validate_legacy_model_assets(
        assets,
        required_views,
        expected_roi_sha256=_sha256_file(roi_path),
    )
    _validate_legacy_replay_plan(assets, golden, required_views)
    scenarios = {case["scenario"] for case in golden}
    if not NORMATIVE_REQUIRED_GOLDEN_SCENARIOS.issubset(scenarios):
        raise Phase0Error("replay binding golden selection lacks normative scenario coverage")
    repository = _git_snapshot(require_clean=True, expected=spec["expected_git"])
    environment = _stable_environment_snapshot(_environment_snapshot(spec, gpu))
    _validate_ultralytics_source_binding(assets, environment)
    environment_sha256 = _canonical_sha256(environment)
    repository_contracts = {
        name: {
            "path": str(REPO_ROOT / name),
            "sha256": _sha256_file(REPO_ROOT / name),
        }
        for name in ("pyproject.toml", "uv.lock")
    }
    if _sha256_file(spec_path) != spec_sha256:
        raise Phase0Error("freeze spec changed while replay bindings were built")
    payload = {
        "schema_version": 1,
        "freeze_id": spec["freeze_id"],
        "product": EXPECTED_PRODUCT,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "freeze_spec_sha256": spec_sha256,
        "environment": environment,
        "environment_sha256": environment_sha256,
        "source_bindings": _replay_source_bindings(
            repository=repository,
            repository_contracts=repository_contracts,
            topology_sha256=_sha256_file(topology_path),
            roi_sha256=_sha256_file(roi_path),
            assets=assets,
            golden=golden,
            environment_sha256=environment_sha256,
        ),
    }
    _write_json(output.resolve(), payload)
    return payload


def _asset_by_role(assets: list[dict[str, Any]], role: str) -> dict[str, Any]:
    matches = [asset for asset in assets if asset.get("present") is True and asset.get("role") == role]
    if len(matches) != 1:
        raise Phase0Error(f"legacy replay requires exactly one asset with role {role!r}")
    return matches[0]


def _read_replay_csv_summary(
    path: Path,
    *,
    required_views: list[str],
    branch_group: str,
    expected_identity: dict[str, str] | None = None,
) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    """Read only stable categorical Stage 32 evidence, excluding continuous scores."""
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise Phase0Error(f"cannot read Stage 32 {branch_group} evidence: {error}") from error
    if len(rows) != len(required_views):
        raise Phase0Error(f"Stage 32 {branch_group} evidence must contain exactly six rows")
    summary: list[dict[str, str]] = []
    semantic: dict[str, dict[str, str]] = {}
    for row in rows:
        view = str(row.get("view", "")).strip()
        declared_level = str(row.get("evidence_level", "")).strip().upper()
        try:
            score = float(str(row.get("score", "")).strip())
            low = float(str(row.get("low_threshold", "")).strip())
            high = float(str(row.get("high_threshold", "")).strip())
        except ValueError as error:
            raise Phase0Error(
                f"Stage 32 {branch_group}/{view or 'unknown'} cannot derive categorical evidence"
            ) from error
        if not all(math.isfinite(value) for value in (score, low, high)) or low > high:
            raise Phase0Error(
                f"Stage 32 {branch_group}/{view or 'unknown'} has invalid dual-threshold evidence"
            )
        if branch_group == "yolo" and not all(0 <= value <= 1 for value in (score, low, high)):
            raise Phase0Error(
                f"Stage 32 yolo/{view or 'unknown'} score/threshold is outside [0, 1]"
            )
        level = "STRONG" if score >= high else "GRAY" if score >= low else "CLEAR"
        if declared_level and declared_level != level:
            raise Phase0Error(
                f"Stage 32 {branch_group}/{view or 'unknown'} declared evidence_level "
                "disagrees with locked dual thresholds"
            )
        if view not in required_views or view in semantic:
            raise Phase0Error(f"Stage 32 {branch_group} evidence has an invalid or duplicate view")
        if expected_identity is not None:
            for field, expected in expected_identity.items():
                if str(row.get(field, "")).strip() != expected:
                    raise Phase0Error(
                        f"Stage 32 {branch_group}/{view} {field} disagrees with the golden case"
                    )
        expected_branch = {
            "template": "template_match",
            "anomaly": f"anomaly_{view}",
            "yolo": "yolo",
        }[branch_group]
        if str(row.get("branch", "")).strip() != expected_branch:
            raise Phase0Error(
                f"Stage 32 {branch_group}/{view} branch identity disagrees with the deployed contract"
            )
        if level not in {"CLEAR", "GRAY", "STRONG"}:
            raise Phase0Error(f"Stage 32 {branch_group}/{view} lacks a valid evidence_level")
        stable = {"view": view, "evidence_level": level}
        for name in ("branch", "status", "pred_label"):
            value = str(row.get(name, "")).strip()
            if value:
                stable[name] = value
        summary.append(stable)
        semantic[view] = {
            key: stable[key] for key in ("status", "evidence_level") if key in stable
        }
        if not semantic[view]:
            raise Phase0Error(f"Stage 32 {branch_group}/{view} has no stable categorical evidence")
        if branch_group == "template":
            expected_level = {
                "PASS": "CLEAR",
                "REVIEW": "GRAY",
                "NG_TEMPLATE": "STRONG",
            }.get(stable.get("status", "").upper())
            if expected_level != level:
                raise Phase0Error(
                    f"Stage 32 template/{view} status disagrees with dual-threshold evidence"
                )
    if set(semantic) != set(required_views):
        raise Phase0Error(f"Stage 32 {branch_group} evidence does not cover all required views")
    return summary, semantic


def _write_replay_envelope(
    path: Path,
    *,
    case_id: str,
    run_id: str,
    execution_id: str,
    executed_at: str,
    payload: dict[str, Any],
) -> None:
    _reject_placeholders(payload, f"generated replay evidence {path.name}")
    _write_json(
        path,
        {
            "schema_version": 1,
            "case_id": case_id,
            "run_id": run_id,
            "execution_id": execution_id,
            "captured_at": executed_at,
            "payload": payload,
        },
    )


def _replay_artifact_payload(path: Path, relative_to: Path, **values: Any) -> dict[str, Any]:
    if not path.is_file():
        raise Phase0Error(f"Stage 32 replay artifact is missing: {path}")
    return {
        "artifact_relative_path": path.relative_to(relative_to).as_posix(),
        "artifact_sha256": _sha256_file(path),
        **values,
    }


def _derive_replay_observed(
    replay_root: Path,
    *,
    result: dict[str, Any],
    evidence: dict[str, dict[str, Any]],
    required_views: list[str],
) -> dict[str, Any]:
    """Rebuild categorical replay semantics from immutable raw Stage 32 artifacts."""
    legacy_output = replay_root / "legacy_output"
    summary_path = legacy_output / "runtime_summary.json"
    summary = _load_json(summary_path)
    status = summary.get("machine_status")
    if status not in ALLOWED_LEGACY_STATUSES or not isinstance(summary.get("inspection_complete"), bool):
        raise Phase0Error("Stage 32 runtime summary has an unsupported status/completion contract")
    source_binding = result.get("source_binding")
    capture_identity = source_binding.get("capture_identity") if isinstance(source_binding, dict) else None
    if not isinstance(capture_identity, dict):
        raise Phase0Error("phase0_case_result source binding lacks capture identity")
    identity_values: dict[str, str] = {}
    for source_field, target_field in (
        ("part_instance_id", "part_id"),
        ("session_id", "capture_session"),
        ("group_id", "group_id"),
    ):
        value = capture_identity.get(source_field)
        if not isinstance(value, str) or not value:
            raise Phase0Error(f"phase0_case_result capture identity lacks {source_field}")
        identity_values[target_field] = value
    expected_identity = {
        **identity_values,
        "hand": "right",
        "product": EXPECTED_PRODUCT,
        "side": "zs32",
        "profile": "zs32_right_six_view_v1",
    }
    short_circuited = summary.get("short_circuited") is True
    if short_circuited:
        template_results = summary.get("template_results")
        evaluated_views = summary.get("evaluated_views")
        if (
            summary.get("inspection_complete") is not False
            or summary.get("part_id") != expected_identity["part_id"]
            or not isinstance(template_results, list)
            or not template_results
            or evaluated_views != required_views[: len(template_results)]
            or [item.get("view") for item in template_results if isinstance(item, dict)]
            != evaluated_views
        ):
            raise Phase0Error("template short circuit violates the canonical prefix/completion contract")
        template_rows: list[dict[str, str]] = []
        template_semantic: dict[str, dict[str, str]] = {}
        level_by_status = {
            "PASS": "CLEAR",
            "REVIEW": "GRAY",
            "NG_TEMPLATE": "STRONG",
            "INVALID_CAPTURE": "STRONG",
        }
        statuses: list[str] = []
        for item in template_results:
            if not isinstance(item, dict):
                raise Phase0Error("template short-circuit evidence is malformed")
            view = str(item.get("view", "")).strip()
            item_status = str(item.get("status", "")).strip().upper()
            if view not in required_views or view in template_semantic or item_status not in level_by_status:
                raise Phase0Error("template short-circuit evidence has invalid view/status")
            statuses.append(item_status)
            stable = {
                "view": view,
                "status": item_status,
                "evidence_level": level_by_status[item_status],
            }
            template_rows.append(stable)
            template_semantic[view] = {
                "status": item_status,
                "evidence_level": level_by_status[item_status],
            }
        if any(item != "PASS" for item in statuses[:-1]) or statuses[-1] == "PASS":
            raise Phase0Error("template short circuit must stop at the first non-PASS view")
        expected_status = statuses[-1] if statuses[-1] in {"NG_TEMPLATE", "INVALID_CAPTURE"} else "REVIEW"
        if status != expected_status:
            raise Phase0Error("template short-circuit status disagrees with the stopping result")
        forbidden = [
            legacy_output / "patchcore.csv",
            legacy_output / "yolo.csv",
            legacy_output / "fusion",
        ]
        if any(path.exists() for path in forbidden):
            raise Phase0Error("template short circuit produced forbidden layer-2/fusion artifacts")
        not_run = {"state": "NOT_RUN", "reason": "template gate short-circuited Stage 32"}
        expected_payloads = {
            "runtime_summary.json": _replay_artifact_payload(
                summary_path,
                replay_root,
                machine_status=status,
                inspection_complete=False,
                execution_state="SHORT_CIRCUITED",
            ),
            "template_evidence.json": _replay_artifact_payload(
                summary_path,
                replay_root,
                rows=template_rows,
            ),
            "anomaly_evidence.json": not_run,
            "yolo_evidence.json": not_run,
            "fusion_result.json": _replay_artifact_payload(
                summary_path,
                replay_root,
                final_status=status,
                source="template_short_circuit",
            ),
            "audit.json": {
                "state": "NOT_PRODUCED",
                "reason": "template gate short-circuited before strict fusion audit",
            },
        }
        observed = {
            "final_status": status,
            "evidence_groups": {
                "template": template_semantic,
                "anomaly": not_run,
                "yolo": not_run,
            },
        }
    else:
        if summary.get("strict_fusion") is not True:
            raise Phase0Error("non-short-circuit replay must record strict_fusion=true")
        paths = {
            "template": legacy_output / "template_match.csv",
            "anomaly": legacy_output / "patchcore.csv",
            "yolo": legacy_output / "yolo.csv",
        }
        rows: dict[str, list[dict[str, str]]] = {}
        semantics: dict[str, dict[str, dict[str, str]]] = {}
        for branch_group, path in paths.items():
            rows[branch_group], semantics[branch_group] = _read_replay_csv_summary(
                path,
                required_views=required_views,
                branch_group=branch_group,
                expected_identity=expected_identity,
            )
        part_id = expected_identity["part_id"]
        fused_path = legacy_output / "fusion" / "fused_predictions.csv"
        with fused_path.open(newline="", encoding="utf-8") as stream:
            fused_rows = list(csv.DictReader(stream))
        if (
            len(fused_rows) != 1
            or fused_rows[0].get("part_id") != part_id
            or fused_rows[0].get("final_status") != status
        ):
            raise Phase0Error("Stage 32 fused prediction disagrees with runtime summary")
        raw_audit_path = legacy_output / "fusion" / "audit" / f"{part_id}.json"
        raw_audit = _load_json(raw_audit_path)
        if (
            raw_audit.get("machine_status") != status
            or not isinstance(raw_audit.get("inspection_complete"), bool)
            or raw_audit.get("inspection_complete") != summary["inspection_complete"]
            or raw_audit.get("part_id") != part_id
            or raw_audit.get("product") != EXPECTED_PRODUCT
            or raw_audit.get("profile") != expected_identity["profile"]
            or raw_audit.get("hand") != "right"
            or raw_audit.get("capture_session") != expected_identity["capture_session"]
            or raw_audit.get("group_id") != expected_identity["group_id"]
        ):
            raise Phase0Error("Stage 32 audit disagrees with runtime summary")
        expected_payloads = {
            "runtime_summary.json": _replay_artifact_payload(
                summary_path,
                replay_root,
                machine_status=status,
                inspection_complete=summary["inspection_complete"],
                execution_state="STRICT_FUSION_COMPLETED",
            ),
            "template_evidence.json": _replay_artifact_payload(
                paths["template"], replay_root, rows=rows["template"]
            ),
            "anomaly_evidence.json": _replay_artifact_payload(
                paths["anomaly"], replay_root, rows=rows["anomaly"]
            ),
            "yolo_evidence.json": _replay_artifact_payload(
                paths["yolo"], replay_root, rows=rows["yolo"]
            ),
            "fusion_result.json": _replay_artifact_payload(
                fused_path,
                replay_root,
                part_id=part_id,
                final_status=status,
            ),
            "audit.json": _replay_artifact_payload(
                raw_audit_path,
                replay_root,
                part_id=part_id,
                machine_status=status,
                inspection_complete=summary["inspection_complete"],
            ),
        }
        observed = {
            "final_status": status,
            "evidence_groups": semantics,
        }
    for name, expected_payload in expected_payloads.items():
        if evidence.get(name) != expected_payload:
            raise Phase0Error(f"replay evidence envelope disagrees with raw Stage 32 artifact: {name}")
    if result.get("observed_legacy") != observed:
        raise Phase0Error("phase0_case_result observed_legacy disagrees with raw Stage 32 artifacts")
    return observed


def _materialize_replay_runtime(
    assets: list[dict[str, Any]],
    staging: Path,
    required_views: list[str],
) -> dict[str, str]:
    """Snapshot every transitive Stage 32 model/ROI input and rewrite one local registry."""
    runtime_root = staging / "runtime_inputs"
    runtime_root.mkdir()
    registry_asset = _asset_by_role(assets, "legacy_model_registry")
    registry = _load_json(Path(registry_asset["source_path"]))
    patchcore = registry.get("patchcore")
    if not isinstance(patchcore, dict):
        raise Phase0Error("legacy runtime registry patchcore section is malformed")

    checkpoint_root = Path(_asset_by_role(assets, "selected_anomaly_family")["source_path"])
    checkpoints_by_sha = {
        _sha256_file(path): path
        for path in _regular_files(checkpoint_root)
        if path.suffix == ".ckpt"
    }
    checkpoint_dir = runtime_root / "checkpoints"
    checkpoint_dir.mkdir()
    for view in required_views:
        record = patchcore.get(view)
        if not isinstance(record, dict):
            raise Phase0Error(f"legacy runtime registry lacks PatchCore view {view}")
        digest = record.get("checkpoint_sha256")
        source = checkpoints_by_sha.get(digest)
        if source is None:
            raise Phase0Error(f"legacy runtime checkpoint asset is missing for {view}")
        destination = checkpoint_dir / f"{view}.ckpt"
        _copy_and_verify(source, destination, digest)
        record["checkpoint"] = str(destination)

    yolo_asset = _asset_by_role(assets, "deployment_weight")
    yolo_destination = runtime_root / "yolo" / "best.pt"
    _copy_and_verify(Path(yolo_asset["source_path"]), yolo_destination, yolo_asset["sha256"])
    yolo = registry.get("yolo")
    if not isinstance(yolo, dict):
        raise Phase0Error("legacy runtime registry yolo section is malformed")
    yolo["weights"] = str(yolo_destination)

    for field, role, filename in (
        ("patchcore_roi_config", "legacy_patchcore_roi", "patchcore_roi.json"),
        ("yolo_roi_config", "legacy_yolo_roi", "yolo_roi.json"),
    ):
        asset = _asset_by_role(assets, role)
        destination = runtime_root / filename
        _copy_and_verify(Path(asset["source_path"]), destination, asset["sha256"])
        registry[field] = str(destination)
    registry_path = runtime_root / "runtime_models.json"
    _write_json(registry_path, registry)

    template_asset = _asset_by_role(assets, "template_match")
    template_destination = runtime_root / "templates"
    _copy_and_verify(
        Path(template_asset["source_path"]),
        template_destination,
        template_asset["sha256"],
    )
    threshold_asset = _asset_by_role(assets, "legacy_thresholds")
    threshold_destination = runtime_root / "thresholds.json"
    _copy_and_verify(
        Path(threshold_asset["source_path"]),
        threshold_destination,
        threshold_asset["sha256"],
    )
    return {
        "runtime_config": registry_path.relative_to(staging).as_posix(),
        "template_model_dir": template_destination.relative_to(staging).as_posix(),
        "threshold_artifact": threshold_destination.relative_to(staging).as_posix(),
    }


def replay_legacy_case(
    spec_path: Path,
    *,
    bindings_path: Path,
    case_id: str,
    run_id: str,
    output: Path,
) -> Path:
    """Run one immutable Stage 32 replay and adapt its real outputs to Phase 0 evidence."""
    require_linux_nvidia()
    case_id = _safe_id(case_id, "replay case_id")
    run_id = _safe_id(run_id, "replay run_id")
    spec_path = spec_path.resolve()
    output = output.resolve()
    if output.exists():
        raise Phase0Error(f"replay output already exists and cannot be overwritten: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    bindings = _load_json(bindings_path.resolve())
    bound_environment = bindings.get("environment")
    bound_environment_sha256 = bindings.get("environment_sha256")
    if (
        set(bindings)
        != {
            "schema_version",
            "freeze_id",
            "product",
            "generated_at",
            "freeze_spec_sha256",
            "environment",
            "environment_sha256",
            "source_bindings",
        }
        or bindings.get("schema_version") != 1
        or bindings.get("product") != EXPECTED_PRODUCT
        or bindings.get("freeze_id") != _load_json(spec_path).get("freeze_id")
        or RFC3339_PATTERN.fullmatch(str(bindings.get("generated_at", ""))) is None
        or bindings.get("freeze_spec_sha256") != _sha256_file(spec_path)
        or not isinstance(bound_environment, dict)
        or not isinstance(bound_environment_sha256, str)
        or SHA256_PATTERN.fullmatch(bound_environment_sha256) is None
        or bound_environment_sha256 != _canonical_sha256(bound_environment)
    ):
        raise Phase0Error("replay bindings do not match the current freeze spec")
    bound_cases = bindings.get("source_bindings")
    if not isinstance(bound_cases, dict) or not isinstance(bound_cases.get(case_id), dict):
        raise Phase0Error(f"replay bindings do not contain case {case_id}")

    with tempfile.TemporaryDirectory(prefix=".zs32-phase0-binding-check-", dir=output.parent) as temporary:
        current_path = Path(temporary) / "current-bindings.json"
        current = build_replay_bindings(spec_path, output=current_path)
    if current["source_bindings"].get(case_id) != bound_cases[case_id]:
        raise Phase0Error("replay bindings differ from current Git/assets/golden inputs")

    spec = _load_and_validate_spec(spec_path)
    base = spec_path.parent
    scope = spec["scope"]
    topology_path = _resolve(scope["topology_config"], base)
    roi_path = _resolve(scope["roi_config"], base)
    selection_path = _resolve(scope["golden_selection"], base)
    _topology, required_views = _validate_topology(topology_path)
    roi = _validate_roi(roi_path, ["right"], required_views)
    image_size = (roi["source_image_size"]["width"], roi["source_image_size"]["height"])
    _selection, golden = _golden_inventory(selection_path, ["right"], required_views, image_size)
    golden_by_id = {case["case_id"]: case for case in golden}
    if case_id not in golden_by_id:
        raise Phase0Error(f"legacy replay references unknown golden case {case_id}")
    golden_case = golden_by_id[case_id]
    assets = _asset_inventory(spec, base)
    plan = _validate_legacy_replay_plan(assets, golden, required_views)
    plan_case = next(item for item in plan["cases"] if item["case_id"] == case_id)

    entrypoint = (REPO_ROOT / plan["entrypoint"]).resolve()
    if not entrypoint.is_file():
        raise Phase0Error(f"legacy Stage 32 entrypoint is missing: {entrypoint}")

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    published = False
    execution_id: str | None = None
    executed_at: str | None = None
    command: list[str] = []
    try:
        input_dir = staging / "legacy_inputs"
        input_dir.mkdir()
        for gate, record in plan_case["inputs"].items():
            _copy_and_verify(
                Path(record["source_path"]),
                input_dir / f"{gate}.csv",
                record["sha256"],
            )
        runtime_inputs = _materialize_replay_runtime(assets, staging, required_views)
        image_dir = staging / "golden_inputs"
        image_dir.mkdir()
        images: dict[str, str] = {}
        for item in golden_case["images"]:
            destination = image_dir / f"{item['view']}.png"
            _copy_and_verify(Path(item["source_path"]), destination, item["sha256"])
            images[item["view"]] = destination.relative_to(staging).as_posix()
        part_id = golden_case["capture_identity"]["part_instance_id"]
        command = _legacy_replay_command(
            capture_identity=golden_case["capture_identity"],
            hand=golden_case["hand"],
            required_views=required_views,
            images=images,
            runtime_inputs=runtime_inputs,
            accelerator=plan["accelerator"],
            devices=plan["devices"],
            yolo_device=plan["yolo_device"],
        )
        executed_at = datetime.now(timezone.utc).isoformat()
        execution_id = f"{case_id}-{run_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
        try:
            result = subprocess.run(
                command,
                cwd=staging,
                check=False,
                capture_output=True,
                text=True,
                timeout=LEGACY_REPLAY_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout if isinstance(error.stdout, str) else ""
            stderr = error.stderr if isinstance(error.stderr, str) else ""
            _write_text_no_replace(staging / "stdout.log", stdout)
            _write_text_no_replace(staging / "stderr.log", stderr)
            _write_json(
                staging / "execution_failure.json",
                {
                    "schema_version": 1,
                    "case_id": case_id,
                    "run_id": run_id,
                    "execution_id": execution_id,
                    "executed_at": executed_at,
                    "command_argv": command,
                    "exit_code": None,
                    "reason": (
                        "legacy Stage 32 exceeded the fixed "
                        f"{LEGACY_REPLAY_TIMEOUT_SECONDS}-second timeout"
                    ),
                },
            )
            _rename_directory_noreplace(staging, output)
            published = True
            _fsync_directory(output.parent)
            raise Phase0Error(f"legacy replay timed out; diagnostics were published at {output}") from error
        except OSError as error:
            _write_text_no_replace(staging / "stdout.log", "")
            _write_text_no_replace(staging / "stderr.log", f"{type(error).__name__}: {error}\n")
            _write_json(
                staging / "execution_failure.json",
                {
                    "schema_version": 1,
                    "case_id": case_id,
                    "run_id": run_id,
                    "execution_id": execution_id,
                    "executed_at": executed_at,
                    "command_argv": command,
                    "exit_code": None,
                    "reason": "legacy Stage 32 process could not be started",
                },
            )
            _rename_directory_noreplace(staging, output)
            published = True
            _fsync_directory(output.parent)
            raise Phase0Error(f"legacy replay could not start; diagnostics were published at {output}") from error
        _write_text_no_replace(staging / "stdout.log", result.stdout)
        _write_text_no_replace(staging / "stderr.log", result.stderr)
        if result.returncode != 0:
            _write_json(
                staging / "execution_failure.json",
                {
                    "schema_version": 1,
                    "case_id": case_id,
                    "run_id": run_id,
                    "execution_id": execution_id,
                    "executed_at": executed_at,
                    "command_argv": command,
                    "exit_code": result.returncode,
                    "reason": "legacy Stage 32 returned a non-zero exit code",
                },
            )
            _rename_directory_noreplace(staging, output)
            published = True
            _fsync_directory(output.parent)
            raise Phase0Error(f"legacy replay failed; diagnostics were published at {output}")

        with tempfile.TemporaryDirectory(
            prefix=".zs32-phase0-post-binding-check-",
            dir=output.parent,
        ) as temporary:
            current_path = Path(temporary) / "current-bindings.json"
            current_after = build_replay_bindings(spec_path, output=current_path)
        if current_after["source_bindings"].get(case_id) != bound_cases[case_id]:
            raise Phase0Error("Git/environment/assets/golden inputs changed during legacy replay")

        legacy_output = staging / "legacy_output"
        summary_path = legacy_output / "runtime_summary.json"
        summary = _load_json(summary_path)
        status = summary.get("machine_status")
        if status not in ALLOWED_LEGACY_STATUSES or not isinstance(summary.get("inspection_complete"), bool):
            raise Phase0Error("Stage 32 runtime summary has an unsupported status/completion contract")
        short_circuited = summary.get("short_circuited") is True

        if short_circuited:
            template_results = summary.get("template_results")
            if not isinstance(template_results, list) or not template_results:
                raise Phase0Error("template short circuit lacks template_results")
            template_rows: list[dict[str, str]] = []
            template_semantic: dict[str, dict[str, str]] = {}
            level_by_status = {
                "PASS": "CLEAR",
                "REVIEW": "GRAY",
                "NG_TEMPLATE": "STRONG",
                "INVALID_CAPTURE": "STRONG",
            }
            for item in template_results:
                if not isinstance(item, dict):
                    raise Phase0Error("template short-circuit evidence is malformed")
                view = str(item.get("view", "")).strip()
                item_status = str(item.get("status", "")).strip().upper()
                if view not in required_views or view in template_semantic or item_status not in level_by_status:
                    raise Phase0Error("template short-circuit evidence has invalid view/status")
                stable = {
                    "view": view,
                    "status": item_status,
                    "evidence_level": level_by_status[item_status],
                }
                template_rows.append(stable)
                template_semantic[view] = {
                    "status": item_status,
                    "evidence_level": level_by_status[item_status],
                }
            not_run = {"state": "NOT_RUN", "reason": "template gate short-circuited Stage 32"}
            anomaly_rows: list[dict[str, str]] | dict[str, str] = not_run
            yolo_rows: list[dict[str, str]] | dict[str, str] = not_run
            anomaly_semantic: dict[str, Any] = not_run
            yolo_semantic: dict[str, Any] = not_run
            fusion_payload = _replay_artifact_payload(
                summary_path,
                staging,
                final_status=status,
                source="template_short_circuit",
            )
            audit_payload = {
                "state": "NOT_PRODUCED",
                "reason": "template gate short-circuited before strict fusion audit",
            }
            template_payload = _replay_artifact_payload(
                summary_path,
                staging,
                rows=template_rows,
            )
            anomaly_payload = not_run
            yolo_payload = not_run
        else:
            template_path = legacy_output / "template_match.csv"
            anomaly_path = legacy_output / "patchcore.csv"
            yolo_path = legacy_output / "yolo.csv"
            template_rows, template_semantic = _read_replay_csv_summary(
                template_path,
                required_views=required_views,
                branch_group="template",
            )
            anomaly_rows, anomaly_semantic = _read_replay_csv_summary(
                anomaly_path,
                required_views=required_views,
                branch_group="anomaly",
            )
            yolo_rows, yolo_semantic = _read_replay_csv_summary(
                yolo_path,
                required_views=required_views,
                branch_group="yolo",
            )
            template_payload = _replay_artifact_payload(template_path, staging, rows=template_rows)
            anomaly_payload = _replay_artifact_payload(anomaly_path, staging, rows=anomaly_rows)
            yolo_payload = _replay_artifact_payload(yolo_path, staging, rows=yolo_rows)
            fused_path = legacy_output / "fusion" / "fused_predictions.csv"
            with fused_path.open(newline="", encoding="utf-8") as stream:
                fused_rows = list(csv.DictReader(stream))
            if (
                len(fused_rows) != 1
                or fused_rows[0].get("part_id") != part_id
                or fused_rows[0].get("final_status") != status
            ):
                raise Phase0Error("Stage 32 fused prediction disagrees with runtime summary")
            fusion_payload = _replay_artifact_payload(
                fused_path,
                staging,
                part_id=part_id,
                final_status=status,
            )
            raw_audit_path = legacy_output / "fusion" / "audit" / f"{part_id}.json"
            raw_audit = _load_json(raw_audit_path)
            if raw_audit.get("machine_status") != status:
                raise Phase0Error("Stage 32 audit disagrees with runtime summary")
            audit_payload = _replay_artifact_payload(
                raw_audit_path,
                staging,
                part_id=part_id,
                machine_status=status,
                inspection_complete=raw_audit.get("inspection_complete") is True,
            )

        runtime_payload = _replay_artifact_payload(
            summary_path,
            staging,
            machine_status=status,
            inspection_complete=summary["inspection_complete"],
            execution_state="SHORT_CIRCUITED" if short_circuited else "STRICT_FUSION_COMPLETED",
        )
        generated_evidence = {
            "runtime_summary.json": runtime_payload,
            "template_evidence.json": template_payload,
            "anomaly_evidence.json": anomaly_payload,
            "yolo_evidence.json": yolo_payload,
            "fusion_result.json": fusion_payload,
            "audit.json": audit_payload,
        }
        for name, payload in generated_evidence.items():
            _write_replay_envelope(
                staging / name,
                case_id=case_id,
                run_id=run_id,
                execution_id=execution_id,
                executed_at=executed_at,
                payload=payload,
            )
        case_result = {
            "schema_version": 1,
            "case_id": case_id,
            "run_id": run_id,
            "execution_id": execution_id,
            "executed_at": executed_at,
            "command_argv": command,
            "exit_code": 0,
            "observed_legacy": {
                "final_status": status,
                "evidence_groups": {
                    "template": template_semantic,
                    "anomaly": anomaly_semantic,
                    "yolo": yolo_semantic,
                },
            },
            "target_contract_expected": {
                "final_status": golden_case["expected_target_status"],
                "reason": golden_case["parity"]["reason"],
            },
            "source_binding": bound_cases[case_id],
            "parity": {
                "mode": golden_case["parity"]["mode"],
            },
        }
        _write_json(staging / "phase0_case_result.json", case_result)
        _derive_replay_observed(
            staging,
            result=case_result,
            evidence=generated_evidence,
            required_views=required_views,
        )
        _write_replay_root(staging)
        _verify_replay_root(staging)
        _rename_directory_noreplace(staging, output)
        published = True
        _fsync_directory(output.parent)
    except Phase0Error as error:
        if (
            not published
            and execution_id is not None
            and executed_at is not None
            and (staging / "stdout.log").is_file()
            and (staging / "stderr.log").is_file()
        ):
            _write_json(
                staging / "adapter_failure.json",
                {
                    "schema_version": 1,
                    "case_id": case_id,
                    "run_id": run_id,
                    "execution_id": execution_id,
                    "executed_at": executed_at,
                    "command_argv": command,
                    "reason": str(error),
                },
            )
            _rename_directory_noreplace(staging, output)
            published = True
            _fsync_directory(output.parent)
            raise Phase0Error(f"legacy replay adapter failed; diagnostics were published at {output}") from error
        raise
    except Exception as error:  # noqa: BLE001 - preserve unexpected adapter diagnostics fail closed
        if (
            execution_id is not None
            and executed_at is not None
            and (staging / "stdout.log").is_file()
            and (staging / "stderr.log").is_file()
        ):
            _write_json(
                staging / "adapter_failure.json",
                {
                    "schema_version": 1,
                    "case_id": case_id,
                    "run_id": run_id,
                    "execution_id": execution_id,
                    "executed_at": executed_at,
                    "command_argv": command,
                    "error_type": type(error).__name__,
                    "reason": str(error) or "unexpected empty adapter diagnostic",
                },
            )
            _rename_directory_noreplace(staging, output)
            published = True
            _fsync_directory(output.parent)
            raise Phase0Error(f"legacy replay adapter failed; diagnostics were published at {output}") from error
        raise Phase0Error(f"legacy replay could not start safely: {type(error).__name__}: {error}") from error
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return output


def build_inventory(spec_path: Path, *, require_clean: bool) -> dict[str, Any]:
    """Validate every freeze input and return a serializable inventory."""
    spec_sha256 = _sha256_file(spec_path)
    gpu = require_linux_nvidia()
    spec = _load_and_validate_spec(spec_path)
    _validate_normative_spec_requirements(spec, require_resolved_blockers=False)
    if _sha256_file(spec_path) != spec_sha256:
        raise Phase0Error("freeze spec changed while it was being read")
    base = spec_path.parent
    scope = spec["scope"]
    hands = scope["hands"]
    topology_path = _resolve(scope["topology_config"], base)
    roi_path = _resolve(scope["roi_config"], base)
    selection_path = _resolve(scope["golden_selection"], base)
    topology, required_views = _validate_topology(topology_path)
    if (
        len(topology["rounds"]) != 2
        or {item["round_id"] for item in topology["rounds"]} != {"front", "back"}
        or len(topology["camera_slots"]) != 3
        or required_views
        != ["front", "front_left", "front_right", "back", "back_left", "back_right"]
    ):
        raise Phase0Error("the current Phase 0 topology must be the frozen 3-camera/2-round/6-view setup")
    roi = _validate_roi(roi_path, hands, required_views)
    if roi.get("topology_id") != topology.get("topology_id"):
        raise Phase0Error("ROI topology_id does not match the frozen camera topology")
    width = roi["source_image_size"]["width"]
    height = roi["source_image_size"]["height"]
    if (width, height) != (4024, 3036):
        raise Phase0Error("the current Phase 0 source image contract must be 4024x3036")
    if roi["hands"]["left"].get("status") != "pending" or roi["hands"]["left"].get("views") != {}:
        raise Phase0Error("the current Phase 0 requires explicit left ROI pending with empty views")
    selection, golden = _golden_inventory(
        selection_path,
        hands,
        required_views,
        (width, height),
    )
    assets = _asset_inventory(spec, base)
    required_roles = spec["required_asset_roles"]
    present_roles = {asset.get("role") for asset in assets if asset.get("present")}
    missing_roles = set(required_roles) - present_roles
    if missing_roles:
        raise Phase0Error(f"required Phase 0 asset roles are missing: {sorted(missing_roles)}")
    duplicate_roles = {
        role
        for role in required_roles
        if sum(asset.get("role") == role and bool(asset.get("present")) for asset in assets) != 1
    }
    if duplicate_roles:
        raise Phase0Error(
            f"required Phase 0 asset roles must occur exactly once: {sorted(duplicate_roles)}"
        )
    optional_required_roles = sorted(
        role
        for role in required_roles
        if not next(asset for asset in assets if asset.get("role") == role).get("required")
    )
    if optional_required_roles:
        raise Phase0Error(
            "normative Phase 0 assets must be marked required=true: "
            f"{optional_required_roles}"
        )
    _validate_legacy_model_assets(
        assets,
        required_views,
        expected_roi_sha256=_sha256_file(roi_path),
    )
    replay_plan = _validate_legacy_replay_plan(assets, golden, required_views)
    required_scenarios = spec["required_golden_scenarios"]
    present_scenarios = {case["scenario"] for case in golden}
    missing_scenarios = set(required_scenarios) - present_scenarios
    if missing_scenarios:
        raise Phase0Error(f"required golden scenarios are missing: {sorted(missing_scenarios)}")
    repository = _git_snapshot(require_clean=require_clean, expected=spec.get("expected_git", {}))
    environment = _environment_snapshot(spec, gpu)
    _validate_ultralytics_source_binding(assets, environment)
    repository_contracts = {
        name: {
            "path": str(REPO_ROOT / name),
            "sha256": _sha256_file(REPO_ROOT / name),
        }
        for name in ("pyproject.toml", "uv.lock")
    }
    source_bindings = _replay_source_bindings(
        repository=repository,
        repository_contracts=repository_contracts,
        topology_sha256=_sha256_file(topology_path),
        roi_sha256=_sha256_file(roi_path),
        assets=assets,
        golden=golden,
        environment_sha256=_canonical_sha256(_stable_environment_snapshot(environment)),
    )
    baselines = _baseline_inventory(
        spec,
        base,
        {
            case["case_id"]: {
                "target_status": case["expected_target_status"],
                "parity_mode": case["parity"]["mode"],
                "scenario": case["scenario"],
                "source_binding": source_bindings[case["case_id"]],
                "required_views": required_views,
                "replay_runtime": {
                    "accelerator": replay_plan["accelerator"],
                    "devices": replay_plan["devices"],
                    "yolo_device": replay_plan["yolo_device"],
                },
            }
            for case in golden
        },
    )
    if _sha256_file(spec_path) != spec_sha256:
        raise Phase0Error("freeze spec changed during inventory")
    return {
        "schema_version": 1,
        "freeze_id": spec["freeze_id"],
        "product": EXPECTED_PRODUCT,
        "review_subject_sha256": _freeze_review_subject_sha256(spec),
        "freeze_spec": {
            "path": str(spec_path),
            "sha256": spec_sha256,
        },
        "scope": scope,
        "repository": repository,
        "environment": environment,
        "topology": {
            "path": str(topology_path),
            "sha256": _sha256_file(topology_path),
            "topology_id": topology["topology_id"],
            "required_views": required_views,
        },
        "roi": {
            "path": str(roi_path),
            "sha256": _sha256_file(roi_path),
            "roi_version": roi["roi_version"],
            "source_image_size": roi["source_image_size"],
        },
        "golden_selection": {
            "path": str(selection_path),
            "sha256": _sha256_file(selection_path),
            "selection_id": selection.get("selection_id"),
        },
        "legacy_replay_plan": replay_plan,
        "assets": assets,
        "golden_cases": golden,
        "baseline_outputs": baselines,
        "requirements": {
            "required_asset_roles": required_roles,
            "required_golden_scenarios": required_scenarios,
            "minimum_replays_per_case": spec["minimum_replays_per_case"],
        },
        "repository_contracts": repository_contracts,
    }


def _write_checksums(bundle: Path) -> str:
    excluded = {"checksums.sha256", "bundle_root.json"}
    checksum_path = bundle / "checksums.sha256"
    root_path = bundle / "bundle_root.json"
    if checksum_path.exists() or root_path.exists():
        raise Phase0Error("refusing to overwrite existing bundle checksum metadata")
    rows: list[str] = []
    for path in _regular_files(bundle):
        relative = path.relative_to(bundle).as_posix()
        if relative in excluded:
            continue
        rows.append(f"{_sha256_file(path)}  {relative}")
    content = "\n".join(rows) + "\n"
    _write_text_no_replace(checksum_path, content)
    root_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    _write_json(
        root_path,
        {
            "algorithm": "sha256(checksums.sha256 bytes)",
            "bundle_root_sha256": root_digest,
            "external_trust_required": True,
        },
    )
    return root_digest


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a freeze without an overwrite race.

    A preceding ``destination.exists()`` check is useful for diagnostics but
    cannot enforce the Phase 0 no-replace contract: another process may create
    the destination before a normal POSIX ``rename``.  The authoritative
    runtime is Linux, so fail closed unless libc exposes
    ``renameat2(RENAME_NOREPLACE)``.
    """
    if sys.platform != "linux":
        raise Phase0Error("atomic no-replace freeze publication requires Linux")

    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise Phase0Error(
            "Linux libc does not expose renameat2; refusing unsafe freeze publication"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise Phase0Error(
            "atomic no-replace freeze publication failed: "
            f"{os.strerror(error_number)}: {destination}"
        )


def _fsync_directory(path: Path) -> None:
    """Confirm directory-entry durability after the publication rename."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    except OSError as error:
        raise Phase0Error(
            f"freeze may be visible at {path}, but directory fsync failed; audit before retry: {error}"
        ) from error
    finally:
        os.close(descriptor)


def _validate_trusted_pointer(
    pointer: dict[str, Any],
    *,
    freeze_id: str,
    expected_root: str,
    git_commit: str,
    topology_id: str,
) -> None:
    """Validate the exact lightweight Git trust-anchor schema."""
    pointer_scope = pointer.get("scope")
    if (
        set(pointer)
        != {
            "schema_version",
            "freeze_id",
            "bundle_root_sha256",
            "git_commit",
            "scope",
            "storage_reference",
        }
        or pointer.get("schema_version") != 1
        or pointer.get("freeze_id") != freeze_id
        or pointer.get("bundle_root_sha256") != expected_root
        or pointer.get("git_commit") != git_commit
        or not isinstance(pointer_scope, dict)
        or set(pointer_scope)
        != {"product", "hands", "topology_id", "left_roi_status"}
        or pointer_scope.get("product") != EXPECTED_PRODUCT
        or pointer_scope.get("hands") != ["right"]
        or pointer_scope.get("topology_id") != topology_id
        or pointer_scope.get("left_roi_status") != "pending"
    ):
        raise Phase0Error("trusted Git pointer does not match the frozen bundle identity")
    storage_reference = _signed_value(
        pointer.get("storage_reference"),
        "trusted pointer storage_reference",
    )
    if expected_root not in storage_reference:
        raise Phase0Error(
            "trusted pointer storage_reference must contain the bundle root SHA256 "
            "as its content address"
        )


def freeze(spec_path: Path) -> Path:
    """Create an atomic, non-overwritable Phase 0 bundle."""
    spec = _load_and_validate_spec(spec_path)
    _signed_value(spec["freeze_id"], "freeze_id")
    if spec.get("status") != "ready":
        raise Phase0Error("freeze spec status must be changed from draft to ready after review")
    approval = spec.get("approval")
    expected_approval_fields = {
        "approved",
        "approved_by",
        "approved_at",
        "basis",
        "reviewed_inventory_path",
        "reviewed_inventory_sha256",
        "review_subject_sha256",
    }
    if (
        not isinstance(approval, dict)
        or set(approval) != expected_approval_fields
        or approval.get("approved") is not True
    ):
        raise Phase0Error("freeze spec requires explicit manual approval before publication")
    _signed_value(approval.get("approved_by"), "approval.approved_by")
    approved_at = _signed_timestamp(approval.get("approved_at"), "approval.approved_at")
    if datetime.fromisoformat(approved_at.replace("Z", "+00:00")) > datetime.now(timezone.utc):
        raise Phase0Error("freeze approval timestamp is in the future")
    _signed_value(approval.get("basis"), "approval.basis")
    reviewed_inventory_value = _signed_value(
        approval.get("reviewed_inventory_path"),
        "approval.reviewed_inventory_path",
    )
    reviewed_inventory_path = Path(reviewed_inventory_value).expanduser()
    if not reviewed_inventory_path.is_absolute():
        reviewed_inventory_path = spec_path.parent / reviewed_inventory_path
    if reviewed_inventory_path.is_symlink() or not reviewed_inventory_path.is_file():
        raise Phase0Error("approval reviewed inventory must be a regular non-symlink file")
    reviewed_inventory_path = reviewed_inventory_path.resolve()
    reviewed_inventory_sha256 = approval.get("reviewed_inventory_sha256")
    if (
        not isinstance(reviewed_inventory_sha256, str)
        or SHA256_PATTERN.fullmatch(reviewed_inventory_sha256) is None
        or _sha256_file(reviewed_inventory_path) != reviewed_inventory_sha256
    ):
        raise Phase0Error("approval reviewed inventory SHA256 is invalid or changed")
    reviewed_inventory = _load_json(reviewed_inventory_path)
    review_subject_sha256 = _freeze_review_subject_sha256(spec)
    if (
        approval.get("review_subject_sha256") != review_subject_sha256
        or reviewed_inventory.get("review_subject_sha256") != review_subject_sha256
        or reviewed_inventory.get("freeze_id") != spec["freeze_id"]
        or reviewed_inventory.get("schema_version") != 1
        or reviewed_inventory.get("product") != EXPECTED_PRODUCT
    ):
        raise Phase0Error("approval does not bind the final freeze review subject")
    reviewed_environment = reviewed_inventory.get("environment")
    if not isinstance(reviewed_environment, dict):
        raise Phase0Error("reviewed inventory environment is malformed")
    captured_at = reviewed_environment.get("captured_at")
    reviewed_at = _signed_timestamp(captured_at, "reviewed inventory environment.captured_at")
    if datetime.fromisoformat(approved_at.replace("Z", "+00:00")) < datetime.fromisoformat(
        reviewed_at.replace("Z", "+00:00")
    ):
        raise Phase0Error("approval timestamp precedes the reviewed inventory")
    reviewed_baselines = reviewed_inventory.get("baseline_outputs")
    if not isinstance(reviewed_baselines, list) or not reviewed_baselines:
        raise Phase0Error("reviewed inventory baseline evidence is missing")
    receipt_review_times = [
        datetime.fromisoformat(
            _signed_timestamp(
                item.get("review_receipt_reviewed_at") if isinstance(item, dict) else None,
                "reviewed inventory baseline review_receipt_reviewed_at",
            ).replace("Z", "+00:00")
        )
        for item in reviewed_baselines
    ]
    if datetime.fromisoformat(approved_at.replace("Z", "+00:00")) < max(receipt_review_times):
        raise Phase0Error("approval timestamp precedes an external baseline review receipt")
    reviewed_payload_sha256 = _reviewable_inventory_sha256(reviewed_inventory)
    _validate_normative_spec_requirements(spec, require_resolved_blockers=True)
    missing_expected_hashes = [
        item.get("asset_id", "unnamed")
        for item in spec.get("assets", [])
        if isinstance(item, dict)
        and item.get("required") is True
        and SHA256_PATTERN.fullmatch(str(item.get("expected_sha256", ""))) is None
    ]
    if missing_expected_hashes:
        raise Phase0Error(
            f"required assets still lack reviewed expected SHA256 values: {missing_expected_hashes}"
        )
    inventory = build_inventory(spec_path, require_clean=True)
    if _reviewable_inventory_sha256(inventory) != reviewed_payload_sha256:
        raise Phase0Error("freeze inputs differ from the inventory bound by manual approval")
    if _sha256_file(spec_path) != inventory["freeze_spec"]["sha256"]:
        raise Phase0Error("freeze spec changed between approval and inventory")
    current_spec = _load_and_validate_spec(spec_path)
    if current_spec != spec:
        raise Phase0Error("freeze spec content changed during freeze")
    output_root = _resolve(spec["output_root"], spec_path.parent)
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / inventory["freeze_id"]
    lock_path = output_root / f".{inventory['freeze_id']}.lock"
    try:
        lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise Phase0Error(f"another freeze owns the publication lock: {lock_path}") from error
    os.close(lock_descriptor)
    staging: Path | None = None
    try:
        if destination.exists():
            raise Phase0Error(f"freeze bundle already exists and cannot be overwritten: {destination}")
        staging = Path(tempfile.mkdtemp(prefix=f".{inventory['freeze_id']}.tmp-", dir=output_root))
        _write_json(staging / "freeze_manifest.json", inventory)
        for label, source, expected in (
            ("freeze_spec.json", spec_path, inventory["freeze_spec"]["sha256"]),
            (
                "topology.json",
                Path(inventory["topology"]["path"]),
                inventory["topology"]["sha256"],
            ),
            ("roi.json", Path(inventory["roi"]["path"]), inventory["roi"]["sha256"]),
            (
                "golden_selection.json",
                Path(inventory["golden_selection"]["path"]),
                inventory["golden_selection"]["sha256"],
            ),
        ):
            _copy_and_verify(source, staging / "contracts" / label, expected)
        for name, contract in inventory["repository_contracts"].items():
            _copy_and_verify(
                Path(contract["path"]),
                staging / "contracts" / "repository" / name,
                contract["sha256"],
            )
        for asset in inventory["assets"]:
            if asset.get("present") and asset.get("copy_into_bundle"):
                source = Path(asset["source_path"])
                _copy_and_verify(
                    source,
                    staging / asset["bundle_relative_path"],
                    asset["sha256"],
                )
        for case in inventory["golden_cases"]:
            for image in case["images"]:
                source = Path(image["source_path"])
                _copy_and_verify(
                    source,
                    staging / "golden" / case["case_id"] / f"{image['view']}.png",
                    image["sha256"],
                )
        copied_review_cases: set[str] = set()
        for baseline in inventory["baseline_outputs"]:
            _copy_and_verify(
                Path(baseline["source_path"]),
                staging / "legacy_results" / baseline["case_id"] / baseline["run_id"],
                baseline["sha256"],
            )
            if baseline["case_id"] not in copied_review_cases:
                _copy_and_verify(
                    Path(baseline["review_receipt_path"]),
                    staging / "legacy_results" / baseline["case_id"] / "review_receipt.json",
                    baseline["review_receipt_sha256"],
                )
                copied_review_cases.add(baseline["case_id"])
        current_repository = _git_snapshot(require_clean=True, expected=spec["expected_git"])
        if current_repository != inventory["repository"]:
            raise Phase0Error("repository changed during freeze")
        root_digest = _write_checksums(staging)
        verify(staging, expected_root=root_digest, expected_freeze_id=inventory["freeze_id"])
        if destination.exists():
            raise Phase0Error(f"freeze destination appeared during publication: {destination}")
        _rename_directory_noreplace(staging, destination)
        staging = None
        _fsync_directory(output_root)
    except BaseException:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        lock_path.unlink(missing_ok=True)
    return destination


def verify(
    bundle: Path,
    *,
    expected_root: str,
    expected_freeze_id: str | None = None,
    trusted_pointer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify bundle contents against a trusted external root digest."""
    require_linux_nvidia()
    if not isinstance(expected_root, str) or SHA256_PATTERN.fullmatch(expected_root) is None:
        raise Phase0Error("verify requires a real external expected root SHA256")
    bundle = bundle.resolve()
    checksums_path = bundle / "checksums.sha256"
    root_path = bundle / "bundle_root.json"
    if not checksums_path.is_file() or not root_path.is_file():
        raise Phase0Error("bundle is missing checksums.sha256 or bundle_root.json")
    try:
        checksum_bytes = checksums_path.read_bytes()
        checksum_text = checksum_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise Phase0Error(f"bundle checksum index is unreadable: {error}") from error
    if not checksum_text or not checksum_text.endswith("\n"):
        raise Phase0Error("bundle checksum index must be non-empty canonical UTF-8 ending in newline")
    expected_rows: dict[str, str] = {}
    checksum_lines = checksum_text.splitlines()
    for line in checksum_lines:
        digest, separator, relative = line.partition("  ")
        if separator != "  " or SHA256_PATTERN.fullmatch(digest) is None:
            raise Phase0Error(f"malformed checksum row: {line!r}")
        if (
            relative in expected_rows
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or "\\" in relative
            or "\x00" in relative
            or "\r" in relative
            or "\n" in relative
        ):
            raise Phase0Error(f"unsafe or duplicate checksum path: {relative}")
        expected_rows[relative] = digest
    canonical_lines = [
        f"{expected_rows[relative]}  {relative}"
        for relative in sorted(expected_rows)
    ]
    if checksum_lines != canonical_lines:
        raise Phase0Error("bundle checksum index is not in canonical path order")
    actual_paths = {
        relative
        for path in _regular_files(bundle)
        if (relative := path.relative_to(bundle).as_posix())
        not in {"checksums.sha256", "bundle_root.json"}
    }
    if actual_paths != set(expected_rows):
        raise Phase0Error(
            f"bundle file set mismatch; missing={sorted(set(expected_rows) - actual_paths)}, "
            f"extra={sorted(actual_paths - set(expected_rows))}"
        )
    for relative, expected in expected_rows.items():
        actual = _sha256_file(bundle / relative)
        if actual != expected:
            raise Phase0Error(f"bundle file hash mismatch: {relative}")
    checksums_digest = hashlib.sha256(checksum_bytes).hexdigest()
    root = _load_json(root_path)
    if (
        set(root) != {"algorithm", "bundle_root_sha256", "external_trust_required"}
        or root.get("algorithm") != "sha256(checksums.sha256 bytes)"
        or root.get("bundle_root_sha256") != checksums_digest
        or root.get("external_trust_required") is not True
    ):
        raise Phase0Error("bundle root digest does not match checksums.sha256")
    if checksums_digest != expected_root:
        raise Phase0Error(
            f"bundle root does not match trusted external root: {checksums_digest} != {expected_root}"
        )

    manifest_path = bundle / "freeze_manifest.json"
    if not manifest_path.is_file():
        raise Phase0Error("bundle is missing freeze_manifest.json")
    manifest = _load_json(manifest_path)
    freeze_id = manifest.get("freeze_id")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("product") != EXPECTED_PRODUCT
        or not isinstance(freeze_id, str)
    ):
        raise Phase0Error("bundle freeze manifest identity is invalid")
    expected_name = freeze_id if expected_freeze_id is None else expected_freeze_id
    if freeze_id != expected_name:
        raise Phase0Error(f"bundle freeze ID mismatch: {freeze_id!r} != {expected_name!r}")
    if expected_freeze_id is None and bundle.name != freeze_id:
        raise Phase0Error(f"bundle directory name must equal freeze_id {freeze_id!r}")
    scope = manifest.get("scope")
    repository = manifest.get("repository")
    if not isinstance(scope, dict) or scope.get("hands") != ["right"]:
        raise Phase0Error("bundle scope is not the approved right-only Phase 0 scope")
    if (
        not isinstance(repository, dict)
        or repository.get("status_porcelain") != []
        or repository.get("commit") != repository.get("upstream_commit")
    ):
        raise Phase0Error("bundle repository snapshot is dirty or not synchronized to upstream")
    if trusted_pointer is not None:
        _validate_trusted_pointer(
            trusted_pointer,
            freeze_id=freeze_id,
            expected_root=expected_root,
            git_commit=repository["commit"],
            topology_id=manifest.get("topology", {}).get("topology_id"),
        )

    topology_path = bundle / "contracts" / "topology.json"
    roi_path = bundle / "contracts" / "roi.json"
    topology, required_views = _validate_topology(topology_path)
    roi = _validate_roi(roi_path, ["right"], required_views)
    if (
        len(topology["camera_slots"]) != 3
        or len(topology["rounds"]) != 2
        or required_views
        != ["front", "front_left", "front_right", "back", "back_left", "back_right"]
        or roi.get("source_image_size") != {"width": 4024, "height": 3036}
    ):
        raise Phase0Error("bundle is not the approved 3-camera/6-view/4024x3036 Phase 0 contract")
    if roi.get("topology_id") != topology.get("topology_id"):
        raise Phase0Error("bundled ROI topology_id does not match the bundled camera topology")
    if topology.get("topology_id") != manifest.get("topology", {}).get("topology_id"):
        raise Phase0Error("bundled topology does not match freeze manifest")
    if roi["hands"]["left"].get("status") != "pending" or roi["hands"]["left"].get("views") != {}:
        raise Phase0Error("bundled ROI does not preserve explicit left pending state")

    contract_checks = (
        (bundle / "contracts" / "freeze_spec.json", manifest.get("freeze_spec", {}).get("sha256")),
        (topology_path, manifest.get("topology", {}).get("sha256")),
        (roi_path, manifest.get("roi", {}).get("sha256")),
        (
            bundle / "contracts" / "golden_selection.json",
            manifest.get("golden_selection", {}).get("sha256"),
        ),
    )
    for path, expected in contract_checks:
        if not isinstance(expected, str) or _sha256_file(path) != expected:
            raise Phase0Error(f"bundled contract does not match freeze manifest: {path}")
    bundled_freeze_spec = _load_json(bundle / "contracts" / "freeze_spec.json")
    _validate_normative_spec_requirements(
        bundled_freeze_spec,
        require_resolved_blockers=True,
    )
    repository_contracts = manifest.get("repository_contracts")
    if not isinstance(repository_contracts, dict) or set(repository_contracts) != {
        "pyproject.toml",
        "uv.lock",
    }:
        raise Phase0Error("bundle repository contracts are missing or malformed")
    for name, contract in repository_contracts.items():
        path = bundle / "contracts" / "repository" / name
        if not isinstance(contract, dict) or _sha256_file(path) != contract.get("sha256"):
            raise Phase0Error(f"bundled repository contract mismatch: {name}")
    assets = manifest.get("assets")
    if not isinstance(assets, list) or not assets:
        raise Phase0Error("bundle asset manifest is missing")
    for asset in assets:
        if not isinstance(asset, dict) or asset.get("present") is not True:
            raise Phase0Error("bundle asset manifest contains a missing or malformed asset")
        if asset.get("copy_into_bundle") is True:
            relative = asset.get("bundle_relative_path")
            if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise Phase0Error(f"bundle asset path is unsafe: {relative!r}")
            bundled = bundle / relative
            digest, _, _ = _path_digest(bundled)
            if digest != asset.get("sha256"):
                raise Phase0Error(f"bundled asset does not match freeze manifest: {asset['asset_id']}")
    requirements = manifest.get("requirements")
    if not isinstance(requirements, dict):
        raise Phase0Error("bundle requirements are missing")
    required_roles = requirements.get("required_asset_roles")
    if (
        not isinstance(required_roles, list)
        or len(required_roles) != len(set(required_roles))
        or set(required_roles) != NORMATIVE_REQUIRED_ASSET_ROLES
    ):
        raise Phase0Error("bundle required asset role declaration is not normative")
    if any(
        sum(
            asset.get("role") == role and asset.get("required") is True
            for asset in assets
        )
        != 1
        for role in NORMATIVE_REQUIRED_ASSET_ROLES
    ):
        raise Phase0Error("bundle does not satisfy required asset roles")
    bundled_assets: list[dict[str, Any]] = []
    for asset in assets:
        if not isinstance(asset, dict) or asset.get("copy_into_bundle") is not True:
            raise Phase0Error("normative Phase 0 assets must be copied into the verified bundle")
        relative = asset.get("bundle_relative_path")
        if not isinstance(relative, str):
            raise Phase0Error("bundled asset lacks a safe relative path")
        bound = dict(asset)
        bound["source_path"] = str(bundle / relative)
        bundled_assets.append(bound)
    _validate_legacy_model_assets(
        bundled_assets,
        required_views,
        expected_roi_sha256=_sha256_file(roi_path),
    )
    golden_cases = manifest.get("golden_cases")
    if not isinstance(golden_cases, list) or not golden_cases:
        raise Phase0Error("bundle golden case manifest is missing")
    _validate_legacy_replay_plan(bundled_assets, golden_cases, required_views)
    for case in golden_cases:
        if not isinstance(case, dict):
            raise Phase0Error("bundle golden case manifest is malformed")
        for image in case.get("images", []):
            bundled = bundle / "golden" / case["case_id"] / f"{image['view']}.png"
            if _sha256_file(bundled) != image.get("sha256"):
                raise Phase0Error(f"bundled golden image mismatch: {case['case_id']}/{image['view']}")
    required_scenarios = requirements.get("required_golden_scenarios")
    if (
        not isinstance(required_scenarios, list)
        or len(required_scenarios) != len(set(required_scenarios))
        or set(required_scenarios) != NORMATIVE_REQUIRED_GOLDEN_SCENARIOS
        or not NORMATIVE_REQUIRED_GOLDEN_SCENARIOS.issubset(
            {case.get("scenario") for case in golden_cases}
        )
    ):
        raise Phase0Error("bundle does not satisfy required golden scenario coverage")
    baseline_outputs = manifest.get("baseline_outputs")
    if not isinstance(baseline_outputs, list) or not baseline_outputs:
        raise Phase0Error("bundle baseline manifest is missing")
    for baseline in baseline_outputs:
        if not isinstance(baseline, dict):
            raise Phase0Error("bundle baseline manifest is malformed")
        bundled = bundle / "legacy_results" / baseline["case_id"] / baseline["run_id"]
        digest, _, _ = _path_digest(bundled)
        if digest != baseline.get("sha256"):
            raise Phase0Error(
                f"bundled baseline mismatch: {baseline['case_id']}/{baseline['run_id']}"
            )
        review_receipt = bundle / "legacy_results" / baseline["case_id"] / "review_receipt.json"
        if _sha256_file(review_receipt) != baseline.get("review_receipt_sha256"):
            raise Phase0Error(f"bundled baseline review mismatch: {baseline['case_id']}")
    environment = manifest.get("environment")
    if not isinstance(environment, dict):
        raise Phase0Error("bundle environment receipt is missing")
    _validate_ultralytics_source_binding(bundled_assets, environment)
    source_bindings = _replay_source_bindings(
        repository=repository,
        repository_contracts=repository_contracts,
        topology_sha256=manifest["topology"]["sha256"],
        roi_sha256=manifest["roi"]["sha256"],
        assets=assets,
        golden=golden_cases,
        environment_sha256=_canonical_sha256(_stable_environment_snapshot(environment)),
    )
    bundled_baseline_spec = {
        "minimum_replays_per_case": requirements.get("minimum_replays_per_case"),
        "baseline_outputs": [
            {
                "case_id": baseline["case_id"],
                "run_id": baseline["run_id"],
                "path": str(bundle / "legacy_results" / baseline["case_id"] / baseline["run_id"]),
                "required": True,
            }
            for baseline in baseline_outputs
        ],
        "baseline_reviews": [
            {
                "case_id": case_id,
                "path": str(bundle / "legacy_results" / case_id / "review_receipt.json"),
            }
            for case_id in sorted({baseline["case_id"] for baseline in baseline_outputs})
        ],
    }
    expected_cases = {
        case["case_id"]: {
            "target_status": case["expected_target_status"],
            "parity_mode": case["parity"]["mode"],
            "source_binding": source_bindings[case["case_id"]],
            "required_views": required_views,
        }
        for case in golden_cases
    }
    verified_baselines = _baseline_inventory(bundled_baseline_spec, bundle, expected_cases)
    stable_fields = {
        "case_id",
        "run_id",
        "sha256",
        "size_bytes",
        "file_count",
        "replay_root_sha256",
        "semantic_signature",
        "review_receipt_sha256",
        "review_receipt_reviewed_at",
    }
    expected_stable = sorted(
        ({key: value for key, value in record.items() if key in stable_fields} for record in baseline_outputs),
        key=lambda value: (value["case_id"], value["run_id"]),
    )
    actual_stable = sorted(
        ({key: value for key, value in record.items() if key in stable_fields} for record in verified_baselines),
        key=lambda value: (value["case_id"], value["run_id"]),
    )
    if actual_stable != expected_stable:
        raise Phase0Error("bundled baseline semantic inventory differs from the freeze manifest")
    return {
        "bundle": str(bundle),
        "file_count": len(expected_rows),
        "bundle_root_sha256": checksums_digest,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }


def fault_injection(
    bundle: Path,
    *,
    expected_root: str,
    work_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Prove that integrity and semantic mutations fail closed on Linux."""
    require_linux_nvidia()
    bundle = bundle.resolve()
    work_root = work_root.resolve()
    if not work_root.is_dir() or work_root.is_symlink():
        raise Phase0Error(f"fault injection work root is missing or unsafe: {work_root}")
    verify(bundle, expected_root=expected_root)
    manifest = _load_json(bundle / "freeze_manifest.json")
    checksum_rows = (bundle / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    covered_paths = [line.partition("  ")[2] for line in checksum_rows if line]
    if not covered_paths:
        raise Phase0Error("cannot fault-inject an empty bundle")

    deployment_assets = [
        asset
        for asset in manifest.get("assets", [])
        if isinstance(asset, dict)
        and asset.get("role") == "deployment_weight"
        and asset.get("copy_into_bundle") is True
    ]
    if len(deployment_assets) != 1:
        raise Phase0Error("fault injection requires exactly one copied deployment weight")
    deployment_relative = deployment_assets[0].get("bundle_relative_path")
    if not isinstance(deployment_relative, str):
        raise Phase0Error("deployment weight has no bundle_relative_path")

    results: list[dict[str, str | bool]] = []
    with tempfile.TemporaryDirectory(
        prefix="zs32-phase0-faults-",
        dir=work_root,
    ) as temporary_root:
        root = Path(temporary_root)

        def expect_failure(
            fault_id: str,
            mutate: Any,
            *,
            rehash: bool = False,
        ) -> None:
            candidate = root / fault_id
            try:
                shutil.copytree(bundle, candidate, symlinks=False)
                mutate(candidate)
                candidate_root = expected_root
                if rehash:
                    (candidate / "checksums.sha256").unlink()
                    (candidate / "bundle_root.json").unlink()
                    candidate_root = _write_checksums(candidate)
                try:
                    verify(
                        candidate,
                        expected_root=candidate_root,
                        expected_freeze_id=manifest["freeze_id"],
                    )
                except Phase0Error as error:
                    results.append(
                        {"fault_id": fault_id, "detected": True, "diagnostic": str(error)}
                    )
                    return
                raise Phase0Error(f"fault injection was not detected: {fault_id}")
            finally:
                shutil.rmtree(candidate, ignore_errors=True)

        target_relative = covered_paths[0]

        def append_bytes(path: Path, content: bytes) -> None:
            with path.open("ab") as stream:
                stream.write(content)

        expect_failure(
            "modified-byte",
            lambda candidate: append_bytes(candidate / target_relative, b"\0"),
        )
        expect_failure("missing-file", lambda candidate: (candidate / target_relative).unlink())
        expect_failure(
            "extra-file",
            lambda candidate: (candidate / "UNEXPECTED_FILE").write_bytes(b"fault"),
        )
        expect_failure(
            "replaced-deployment-weight",
            lambda candidate: append_bytes(candidate / deployment_relative, b"fault"),
            rehash=True,
        )

        def change_hand(candidate: Path) -> None:
            path = candidate / "freeze_manifest.json"
            payload = _load_json(path)
            payload["scope"]["hands"] = ["left"]
            path.unlink()
            _write_json(path, payload)

        expect_failure("left-hand-scope", change_hand, rehash=True)

    report = {
        "schema_version": 1,
        "freeze_id": manifest.get("freeze_id"),
        "bundle_root_sha256": expected_root,
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "results": results,
        "all_detected": len(results) == 5 and all(item["detected"] for item in results),
    }
    _write_json(output.resolve(), report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    discover_parser = subparsers.add_parser(
        "discover-legacy",
        help="Create a manual-review candidate list from legacy directories.",
    )
    discover_parser.add_argument("--data-root", required=True, type=Path)
    discover_parser.add_argument("--topology", required=True, type=Path)
    discover_parser.add_argument("--hand", required=True, choices=("right",))
    discover_parser.add_argument("--output", required=True, type=Path)
    binding_parser = subparsers.add_parser(
        "replay-bindings",
        help="Freeze Git/assets/golden identities for real legacy replay evidence.",
    )
    binding_parser.add_argument("--spec", required=True, type=Path)
    binding_parser.add_argument("--output", required=True, type=Path)
    replay_parser = subparsers.add_parser(
        "replay",
        help="Run one real Stage 32 case and publish Phase 0 evidence envelopes.",
    )
    replay_parser.add_argument("--spec", required=True, type=Path)
    replay_parser.add_argument("--bindings", required=True, type=Path)
    replay_parser.add_argument("--case-id", required=True)
    replay_parser.add_argument("--run-id", required=True)
    replay_parser.add_argument("--output", required=True, type=Path)
    inventory_parser = subparsers.add_parser("inventory", help="Validate inputs and write an inventory JSON.")
    inventory_parser.add_argument("--spec", required=True, type=Path)
    inventory_parser.add_argument("--output", required=True, type=Path)
    freeze_parser = subparsers.add_parser("freeze", help="Create an immutable Phase 0 bundle.")
    freeze_parser.add_argument("--spec", required=True, type=Path)
    verify_parser = subparsers.add_parser("verify", help="Verify all files in a Phase 0 bundle.")
    verify_parser.add_argument("--bundle", required=True, type=Path)
    trust_group = verify_parser.add_mutually_exclusive_group(required=True)
    trust_group.add_argument("--expected-root")
    trust_group.add_argument("--pointer", type=Path)
    fault_parser = subparsers.add_parser(
        "fault-injection",
        help="Run fail-closed mutations against a frozen bundle without changing the original.",
    )
    fault_parser.add_argument("--bundle", required=True, type=Path)
    fault_parser.add_argument("--expected-root", required=True)
    fault_parser.add_argument("--work-root", required=True, type=Path)
    fault_parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "discover-legacy":
            payload = discover_legacy_candidates(
                data_root=args.data_root,
                topology_path=args.topology.resolve(),
                hand=args.hand,
                output=args.output,
            )
            print(
                f"{args.output.resolve()} "
                f"({payload['discovery']['complete_candidate_count']} complete candidates)"
            )
        elif args.command == "replay-bindings":
            bindings = build_replay_bindings(args.spec.resolve(), output=args.output)
            print(
                f"{args.output.resolve()} "
                f"({len(bindings['source_bindings'])} replay source bindings)"
            )
        elif args.command == "replay":
            destination = replay_legacy_case(
                args.spec.resolve(),
                bindings_path=args.bindings.resolve(),
                case_id=args.case_id,
                run_id=args.run_id,
                output=args.output,
            )
            print(destination)
        elif args.command == "inventory":
            inventory = build_inventory(args.spec.resolve(), require_clean=False)
            _write_json(args.output.resolve(), inventory)
            print(args.output.resolve())
        elif args.command == "freeze":
            destination = freeze(args.spec.resolve())
            root = _load_json(destination / "bundle_root.json")["bundle_root_sha256"]
            print(json.dumps({"bundle": str(destination), "bundle_root_sha256": root}, sort_keys=True))
        elif args.command == "verify":
            pointer = _load_json(args.pointer.resolve()) if args.pointer is not None else None
            expected_root = (
                pointer.get("bundle_root_sha256") if pointer is not None else args.expected_root
            )
            print(
                json.dumps(
                    verify(
                        args.bundle,
                        expected_root=expected_root,
                        trusted_pointer=pointer,
                    ),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        elif args.command == "fault-injection":
            report = fault_injection(
                args.bundle,
                expected_root=args.expected_root,
                work_root=args.work_root,
                output=args.output,
            )
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        else:  # pragma: no cover - argparse owns the command choices
            raise Phase0Error(f"unsupported Phase 0 command: {args.command}")
    except Phase0Error as error:
        print(f"PHASE0_ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
