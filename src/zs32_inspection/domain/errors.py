"""Fail-closed domain and configuration errors for ZS32 inspection."""

from __future__ import annotations


class ZS32ContractError(ValueError):
    """Base class for an invalid ZS32 business contract."""


class IdentityValidationError(ZS32ContractError):
    """Raised when part, capture, camera, or view identity is invalid."""


class TopologyValidationError(ZS32ContractError):
    """Raised when a camera topology is ambiguous or incomplete."""


class RoiValidationError(ZS32ContractError):
    """Raised when an authoritative ROI configuration is unusable."""


class RecipeValidationError(ZS32ContractError):
    """Raised when a deployment recipe is incomplete or inconsistent."""


class DeploymentContractError(ZS32ContractError):
    """Raised when an immutable deployment contract cannot be compiled."""


class EvidenceValidationError(ZS32ContractError):
    """Raised when model or template evidence violates its identity contract."""


class DecisionValidationError(ZS32ContractError):
    """Raised when an inspection decision represents an impossible state."""


class SchemaValidationError(ZS32ContractError):
    """Raised when an input mapping does not match a supported schema."""


class ConfigLoadError(ZS32ContractError):
    """Raised when a configuration file cannot be loaded safely."""
