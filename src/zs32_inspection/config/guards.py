"""Shared fail-closed guards for runnable recipe inputs."""

from __future__ import annotations

from pathlib import PurePosixPath

from zs32_inspection.domain.contracts import InspectionRecipe


def require_bound_capture_gate_policy(recipe: InspectionRecipe) -> None:
    """Reject the checked-in training template's visible unbound sentinel."""
    if not isinstance(recipe, InspectionRecipe):
        raise TypeError("capture gate recipe guard requires an InspectionRecipe")
    artifact = recipe.capture_gate_policy
    if (
        artifact.sha256 == "0" * 64
        or "unbound" in artifact.artifact_id.casefold()
        or artifact.version.casefold() == "training-profile"
        or PurePosixPath(artifact.relative_path)
        != PurePosixPath("capture/gates/policy.json")
    ):
        raise ValueError(
            "recipe capture_gate_policy is unbound; use the Linux gate publication "
            "policy_sha256/artifact identity, never the checked-in training template"
        )
