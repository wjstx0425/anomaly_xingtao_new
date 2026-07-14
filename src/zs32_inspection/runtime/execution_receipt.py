"""Code, environment, and input identity for training and calibration executions."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from zs32_inspection.capture.contracts import utc_now
from zs32_inspection.domain.identity import require_non_empty, require_sha256

from .code_identity import RuntimeCodeIdentity, collect_runtime_code_identity
from .environment_receipt import (
    RuntimeEnvironmentReceipt,
    collect_runtime_environment,
)
from .publisher import canonical_json_bytes


_SCHEMA = "zs32.execution_receipt"
_OPERATIONS = frozenset({"train_anomaly", "train_template", "score_calibration"})


@dataclass(frozen=True, slots=True)
class ExecutionReceipt:
    """Self-addressed receipt for one model-producing or score-producing process."""

    schema: str
    schema_version: int
    operation: str
    captured_at: str
    code_identity: Mapping[str, object]
    environment: RuntimeEnvironmentReceipt
    input_sha256_by_role: Mapping[str, str]
    parameters_sha256: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.schema != _SCHEMA or self.schema_version != 1:
            raise ValueError("execution receipt must use zs32.execution_receipt schema v1")
        if self.operation not in _OPERATIONS:
            raise ValueError(f"unsupported execution receipt operation: {self.operation!r}")
        timestamp = require_non_empty(self.captured_at, "captured_at")
        if not timestamp.endswith("Z"):
            raise ValueError("execution receipt captured_at must be UTC and end in Z")
        code = dict(self.code_identity)
        expected_code = {
            "git_commit",
            "git_tree",
            "dirty",
            "dependency_lock_sha256",
        }
        if set(code) != expected_code or code.get("dirty") is not False:
            raise ValueError("execution receipt requires an exact clean code identity")
        for field in ("git_commit", "git_tree"):
            value = code.get(field)
            if (
                not isinstance(value, str)
                or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", value) is None
            ):
                raise ValueError(f"execution receipt has invalid {field}")
        dependency = code["dependency_lock_sha256"]
        if not isinstance(dependency, str):
            raise TypeError("execution receipt dependency_lock_sha256 must be text")
        code["dependency_lock_sha256"] = require_sha256(
            dependency,
            "dependency_lock_sha256",
        )
        object.__setattr__(self, "code_identity", MappingProxyType(code))
        if not isinstance(self.environment, RuntimeEnvironmentReceipt):
            raise TypeError("execution receipt requires a runtime environment receipt")
        inputs: dict[str, str] = {}
        for role, digest in self.input_sha256_by_role.items():
            checked_role = require_non_empty(role, "execution input role")
            if re.fullmatch(r"[a-z][a-z0-9_.-]{0,127}", checked_role) is None:
                raise ValueError(f"invalid execution input role: {role!r}")
            inputs[checked_role] = require_sha256(digest, f"execution input {checked_role}")
        if not inputs:
            raise ValueError("execution receipt requires content-addressed inputs")
        object.__setattr__(
            self,
            "input_sha256_by_role",
            MappingProxyType(dict(sorted(inputs.items()))),
        )
        object.__setattr__(
            self,
            "parameters_sha256",
            require_sha256(self.parameters_sha256, "parameters_sha256"),
        )
        expected = hashlib.sha256(
            canonical_json_bytes(self.as_dict(include_receipt_sha256=False))
        ).hexdigest()
        if not self.receipt_sha256:
            object.__setattr__(self, "receipt_sha256", expected)
        elif require_sha256(self.receipt_sha256, "receipt_sha256") != expected:
            raise ValueError("execution receipt_sha256 is not canonical")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "ExecutionReceipt":
        expected = {
            "schema",
            "schema_version",
            "operation",
            "captured_at",
            "code_identity",
            "environment",
            "input_sha256_by_role",
            "parameters_sha256",
            "receipt_sha256",
        }
        if set(payload) != expected:
            raise ValueError("execution receipt has missing or unknown fields")
        if not isinstance(payload["code_identity"], Mapping):
            raise TypeError("execution receipt code_identity must be an object")
        if not isinstance(payload["environment"], Mapping):
            raise TypeError("execution receipt environment must be an object")
        if not isinstance(payload["input_sha256_by_role"], Mapping):
            raise TypeError("execution receipt inputs must be an object")
        return cls(
            schema=payload["schema"],  # type: ignore[arg-type]
            schema_version=payload["schema_version"],  # type: ignore[arg-type]
            operation=payload["operation"],  # type: ignore[arg-type]
            captured_at=payload["captured_at"],  # type: ignore[arg-type]
            code_identity=payload["code_identity"],
            environment=RuntimeEnvironmentReceipt.from_mapping(payload["environment"]),
            input_sha256_by_role=payload["input_sha256_by_role"],  # type: ignore[arg-type]
            parameters_sha256=payload["parameters_sha256"],  # type: ignore[arg-type]
            receipt_sha256=payload["receipt_sha256"],  # type: ignore[arg-type]
        )

    def as_dict(self, *, include_receipt_sha256: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "operation": self.operation,
            "captured_at": self.captured_at,
            "code_identity": dict(self.code_identity),
            "environment": self.environment.as_dict(),
            "input_sha256_by_role": dict(self.input_sha256_by_role),
            "parameters_sha256": self.parameters_sha256,
        }
        if include_receipt_sha256:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload


def _code_mapping(identity: RuntimeCodeIdentity) -> dict[str, object]:
    if identity.dirty:
        raise ValueError("model-producing executions require a clean Git worktree")
    return {
        "git_commit": identity.git_commit,
        "git_tree": identity.git_tree,
        "dirty": identity.dirty,
        "dependency_lock_sha256": identity.dependency_lock_sha256,
    }


def collect_execution_receipt(
    *,
    operation: str,
    device: str,
    input_sha256_by_role: Mapping[str, str],
    parameters: Mapping[str, object],
) -> ExecutionReceipt:
    """Collect the live clean checkout and exact Linux/NVIDIA package tree."""
    return ExecutionReceipt(
        schema=_SCHEMA,
        schema_version=1,
        operation=operation,
        captured_at=utc_now(),
        code_identity=_code_mapping(collect_runtime_code_identity()),
        environment=collect_runtime_environment(device),
        input_sha256_by_role=input_sha256_by_role,
        parameters_sha256=hashlib.sha256(canonical_json_bytes(parameters)).hexdigest(),
    )


def verify_execution_receipt_live(
    receipt: ExecutionReceipt,
    *,
    device: str,
) -> None:
    """Re-observe code/environment immediately before publishing generated bytes."""
    code = _code_mapping(collect_runtime_code_identity())
    environment = collect_runtime_environment(device)
    if code != dict(receipt.code_identity):
        raise ValueError("execution code identity changed while the operation was running")
    if environment.as_dict() != receipt.environment.as_dict():
        raise ValueError("execution environment changed while the operation was running")


__all__ = [
    "ExecutionReceipt",
    "collect_execution_receipt",
    "verify_execution_receipt_live",
]
