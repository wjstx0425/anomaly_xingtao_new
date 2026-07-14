"""Authoritative execution-environment guard shared by every ZS32 CLI."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass


class EnvironmentContractError(RuntimeError):
    """The process is not running on the frozen Linux + NVIDIA platform."""


@dataclass(frozen=True, slots=True)
class NvidiaEnvironment:
    """Minimal GPU identity captured before a workflow starts."""

    nvidia_smi_path: str
    driver_query: str


def require_linux_nvidia() -> NvidiaEnvironment:
    """Reject macOS and hosts without a functioning NVIDIA driver.

    This is a production guard, not a test skip.  A bypass environment
    variable is intentionally not provided because the blueprint forbids Mac
    outputs from becoming workflow artifacts or acceptance evidence.
    """
    if sys.platform != "linux":
        raise EnvironmentContractError(
            "ZS32 workflows may execute only on the authoritative Linux + NVIDIA host; "
            "macOS is edit-only"
        )
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise EnvironmentContractError("nvidia-smi is unavailable on the Linux host")
    environment = dict(os.environ)
    environment.setdefault("LC_ALL", "C")
    try:
        completed = subprocess.run(
            [executable, "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise EnvironmentContractError(f"NVIDIA driver query failed: {error}") from error
    output = completed.stdout.strip()
    if not output:
        raise EnvironmentContractError("nvidia-smi reported no NVIDIA GPU")
    return NvidiaEnvironment(nvidia_smi_path=executable, driver_query=output)
